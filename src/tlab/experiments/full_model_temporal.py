from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tlab.config import load_config
from tlab.data.manifest import sha256_file, write_json_atomic
from tlab.data.prepare import prepare_dataset
from tlab.data.store import SequenceStore
from tlab.evaluation.metrics import paired_user_bootstrap
from tlab.evaluation.policy_panel import _load_reward_model
from tlab.evaluation.rm_diagnostics import diagnose_rm_checkpoint
from tlab.experiments.full_model_analysis import _feature_support, _policy_metrics
from tlab.experiments.full_model_matrix import (
    _collate, _load_cache, _load_store, _root_and_processed,
    build_reward_cache, create_panels, evaluate_actor, train_run,
)
from tlab.experiments.full_model_protocol import PROTOCOL_ID
from tlab.train.reward_model import run_reward_model
from tlab.train.sft_reference import run_sft_reference


def _fold_name(config) -> str:
    return f"tminus{int(config.dataset.get('cutoff_shift_days', 0))}"


def _artifact_root(config) -> Path:
    path = Path(config.project.get("artifacts_dir", "artifacts/full_model"))
    return path if path.is_absolute() else config.path.parent.parent / path


def _e5_fold_dir(config) -> Path:
    return _artifact_root(config) / "e5" / _fold_name(config)


def _fold_root(config) -> Path:
    return _artifact_root(config) / "runs" / _fold_name(config)


def _temporal_model_root(config) -> Path:
    value = config.project.get("temporal_model_source_root")
    if not value:
        return _fold_root(config)
    path = Path(value)
    return path if path.is_absolute() else config.path.parent.parent / path


def _temporal_e5_source_dir(config) -> Path:
    value = config.project.get("temporal_e5_source_root")
    if not value:
        return _e5_fold_dir(config)
    path = Path(value)
    return path if path.is_absolute() else config.path.parent.parent / path


def _audit_sft_lr(config, run_dir: Path) -> dict[str, Any]:
    expected_peak = float(config.training.get("sft_temporal_learning_rate", 0.0001))
    horizon_passes = int(config.training.get("sft_temporal_schedule_passes", 10))
    horizon_steps = int(config.training.get("sft_max_targets_per_epoch", 2_000_000) * horizon_passes /
                        max(1, int(config.training.get("sft_effective_batch", 64))))
    steps_per_pass = horizon_steps // max(1, horizon_passes)
    warmup_steps = int(horizon_steps * float(config.training.get("sft_warmup_fraction", 0.05)))
    evidence = []
    for additional_pass in (2, 5):
        checkpoint_path = run_dir / f"pass_{additional_pass:02d}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"required scheduler audit checkpoint missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        scheduler = checkpoint.get("scheduler") or {}
        metadata = checkpoint.get("metadata") or {}
        step = int(scheduler.get("last_epoch", -1))
        actual_base = float((scheduler.get("base_lrs") or [float("nan")])[0])
        actual_last = float((scheduler.get("_last_lr") or [float("nan")])[0])
        if step <= warmup_steps:
            factor = step / max(1, warmup_steps)
        else:
            factor = max(0.0, (horizon_steps - step) / max(1, horizon_steps - warmup_steps))
        intended_last = expected_peak * factor
        expected_step = additional_pass * steps_per_pass
        evidence.append({"file": checkpoint_path.name, "sha256": sha256_file(checkpoint_path),
                         "additional_pass": additional_pass, "scheduler_base_lrs": scheduler.get("base_lrs"),
                         "scheduler_last_epoch": step, "scheduler_last_lr": scheduler.get("_last_lr"),
                         "metadata_base_learning_rate": metadata.get("base_learning_rate"),
                         "metadata_global_step": metadata.get("global_step"),
                         "expected_scheduler_step": expected_step,
                         "intended_lr_at_pass_end": intended_last,
                         "base_lr_matches_config": math.isclose(actual_base, expected_peak, rel_tol=1e-9, abs_tol=1e-12),
                         "last_lr_matches_schedule": math.isclose(actual_last, intended_last, rel_tol=1e-6, abs_tol=1e-12),
                         "step_matches_pass": step == expected_step})
        del checkpoint
    status = "recipe_valid" if all(row["base_lr_matches_config"] and row["last_lr_matches_schedule"] and
                                   row["step_matches_pass"] for row in evidence) else "recipe_mismatch"
    payload = {"protocol_id": PROTOCOL_ID, "fold": _fold_name(config), "status": status,
               "source_of_expected_lr": f"{config.path.name} and train_local_baselines call",
               "expected_peak_lr": expected_peak, "scheduler_horizon_passes": horizon_passes,
               "scheduler_horizon_steps": horizon_steps, "steps_per_additional_pass": steps_per_pass,
               "passes_audited": [2, 5], "checkpoint_evidence": evidence,
               "no_test_based_tuning": True,
               "final_test_panel_sha256": None}
    panel_path = _e5_fold_dir(config) / "final_test_panel.json"
    if panel_path.is_file():
        panel = json.loads(panel_path.read_text(encoding="utf-8"))
        payload["final_test_panel_sha256"] = panel.get("panel_sha256")
    write_json_atomic(_e5_fold_dir(config) / "sft_lr_trace_audit.json", payload)
    if status != "recipe_valid":
        raise RuntimeError(f"local strong SFT scheduler audit failed: {payload}")
    return payload


def prepare_temporal_fold(config_path: str | Path = "configs/full_model_backtest_tminus14.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    root = config.path.parent.parent
    main_protocol = _artifact_root(config) / "protocol.json"
    if not main_protocol.is_file():
        raise FileNotFoundError("freeze the main full_model protocol before preparing the E5 fold")
    processed = Path(config.data.get("processed_dir", "data/processed"))
    if not processed.is_absolute():
        processed = root / processed
    if not (processed / "protocol.json").is_file():
        prepare_dataset(config_path)
    protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    if int(protocol.get("cutoff_shift_days", 0)) != int(config.dataset.get("cutoff_shift_days", 0)):
        raise ValueError("prepared fold protocol cutoff does not match its full_model config")
    import polars as pl
    prepared_path = processed / "listens.parquet"
    summary = pl.scan_parquet(prepared_path).select(
        pl.col("timestamp").max().alias("max_raw"), pl.len().alias("rows"),
        (pl.col("timestamp") >= float(protocol["fold_end_raw_exclusive"])).sum().alias("future_rows"),
    ).collect(engine="streaming").to_dicts()[0]
    if int(summary["future_rows"] or 0) != 0:
        raise ValueError(f"prepared fold contains {summary['future_rows']} rows at/after its exclusive end")
    if not float(summary["max_raw"]) < float(protocol["fold_end_raw_exclusive"]):
        raise ValueError("prepared fold maximum timestamp violates exclusive cutoff")
    catalog = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    cutoff = float(protocol["boundaries_source_units"]["train_cutoff_raw"])
    # Independently recompute the catalog from only pre-cutoff raw listening rows.
    raw_path = root / "data/raw/listens.parquet"
    expected_catalog = (pl.scan_parquet(raw_path)
                        .filter((pl.col("timestamp") < cutoff) & (pl.col("timestamp") < float(protocol["fold_end_raw_exclusive"]))
                                & (pl.col("is_organic") == 0))
                        .group_by("item_id").agg(pl.len().alias("training_count"))
                        .sort(["training_count", "item_id"], descending=[True, False])
                        .head(int(config.dataset.get("catalog_size", 20000)))
                        .collect(engine="streaming"))
    actual_rows = catalog.select("item_id", "training_count").sort(["training_count", "item_id"], descending=[True, False])
    if expected_catalog.height != actual_rows.height or expected_catalog.to_dicts() != actual_rows.to_dicts():
        raise ValueError("fold catalog does not match a fresh pre-cutoff raw-row recomputation")
    result = {"protocol_id": PROTOCOL_ID, "fold": _fold_name(config), "status": "prepared_and_temporally_verified",
              "cutoff_shift_days": int(config.dataset.get("cutoff_shift_days", 0)),
              "fold_end_raw_exclusive": protocol["fold_end_raw_exclusive"],
              "train_cutoff_raw": cutoff, "timestamp_unit_seconds": protocol["timestamp_unit_seconds"],
              "processed_rows": int(summary["rows"]), "max_prepared_raw_timestamp": float(summary["max_raw"]),
              "future_rows_at_or_after_fold_end": int(summary["future_rows"] or 0),
              "catalog_items": catalog.height, "catalog_sha256": protocol["catalog_sha256"],
              "catalog_recomputed_from_raw_rows_before_train_cutoff": True,
              "processed_listens_sha256": protocol["processed_listens_sha256"],
              "fold_protocol_sha256": sha256_file(processed / "protocol.json"),
              "test_window_end": "fold-specific; all rows at/after fold_end_raw_exclusive excluded"}
    target = _e5_fold_dir(config) / "temporal_data_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    return result


def _resume_model_run(manifest_path: Path, run_path: Path, run_call):
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed":
            return manifest
    return run_call()


def train_local_baselines(config_path: str | Path = "configs/full_model_backtest_tminus14.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    root = config.path.parent.parent
    run_root = _fold_root(config)
    run_root.mkdir(parents=True, exist_ok=True)
    processed = Path(config.data.get("processed_dir", "data/processed"))
    if not processed.is_absolute():
        processed = root / processed
    if not (processed / "protocol.json").is_file():
        raise FileNotFoundError("run --stage prepare before local training")
    weak_dir = run_root / "sft_weak_seed17"
    weak_manifest = weak_dir / "run_manifest.json"
    weak_resume = weak_dir / "last.pt" if (weak_dir / "last.pt").is_file() else None
    weak = _resume_model_run(weak_manifest, weak_dir, lambda: run_sft_reference(
        config_path, seed=17, max_targets=100000, epochs=5, learning_rate=0.0003,
        validation_seed=17, validation_anchors_per_user=10, run_dir=weak_dir,
        phase=f"{_fold_name(config)}_weak_full_model", early_stopping_patience=3,
        minimum_additional_epochs=5, checkpoint_interval_steps=1000,
        resume=weak_resume,
    ))
    weak = json.loads(weak_manifest.read_text(encoding="utf-8"))
    weak_best = weak_dir / "best.pt"
    if not weak_best.is_file():
        raise FileNotFoundError(weak_best)
    strong_dir = run_root / "sft_strong_seed17"
    strong_manifest_path = strong_dir / "run_manifest.json"
    strong_manifest = json.loads(strong_manifest_path.read_text(encoding="utf-8")) if strong_manifest_path.is_file() else {}
    strong_resume = strong_dir / "last.pt" if (strong_dir / "last.pt").is_file() else weak_best
    already_started = strong_resume != weak_best
    strong = _resume_model_run(strong_manifest_path, strong_dir, lambda: run_sft_reference(
        config_path, seed=17, max_targets=2_000_000, additional_epochs=5,
        schedule_epochs=10, learning_rate=0.0001, validation_seed=17,
        validation_anchors_per_user=10, run_dir=strong_dir,
        phase=f"{_fold_name(config)}_strong_full_model", early_stopping_patience=99,
        minimum_additional_epochs=5, checkpoint_interval_steps=1000,
        restart_schedule=not already_started, resume=strong_resume,
    ))
    strong = json.loads(strong_manifest_path.read_text(encoding="utf-8"))
    strong_best = strong_dir / "best.pt"
    if not strong_best.is_file():
        raise FileNotFoundError(strong_best)
    lr_audit = _audit_sft_lr(config, strong_dir)
    result = {"protocol_id": PROTOCOL_ID, "fold": _fold_name(config),
              "weak_sft": {"run_dir": str(weak_dir.resolve()), "best": str(weak_best.resolve()),
                           "best_sha256": sha256_file(weak_best), "status": weak.get("status"),
                           "source_tree_sha256": weak.get("source_sha256"),
                           "dataset_sha256": weak.get("dataset_sha256"),
                           "processed_listens_sha256": weak.get("processed_listens_sha256"),
                           "target_budget_per_pass": weak.get("target_budget_per_epoch"),
                           "passes_completed": int(weak.get("run_target_presentations", 0)
                                                    // max(1, int(weak.get("target_budget_per_epoch", 1)))),
                           "target_presentations": weak.get("run_target_presentations"),
                           "best_epoch": weak.get("best_epoch"),
                           "best_validation_ndcg10": weak.get("best_validation_ndcg10")},
              "strong_sft": {"run_dir": str(strong_dir.resolve()), "best": str(strong_best.resolve()),
                             "best_sha256": sha256_file(strong_best), "status": strong.get("status"),
                             "source_tree_sha256": strong.get("source_sha256"),
                             "dataset_sha256": strong.get("dataset_sha256"),
                             "processed_listens_sha256": strong.get("processed_listens_sha256"),
                             "catalog_sha256": strong.get("catalog_sha256"),
                             "resume_checkpoint_sha256": strong.get("resume_checkpoint_sha256"),
                             "target_budget_per_pass": strong.get("target_budget_per_epoch"),
                             "passes_completed": int(strong.get("run_target_presentations", 0)
                                                      // max(1, int(strong.get("target_budget_per_epoch", 1)))),
                             "target_presentations": strong.get("run_target_presentations"),
                             "cumulative_target_presentations": strong.get("cumulative_target_presentations"),
                             "passes_planned": 5, "schedule_horizon_passes": 10,
                             "scheduler": strong.get("scheduler"),
                             "training_seconds": strong.get("resource_time_totals_this_invocation", {}).get("training_seconds"),
                             "elapsed_seconds": strong.get("elapsed_seconds_this_invocation"),
                             "best_epoch": strong.get("best_epoch"),
                             "best_validation_ndcg10": strong.get("best_validation_ndcg10"),
                             "checkpoint_rule": "best local validation NDCG among first five additional passes",
                             "phase": f"{_fold_name(config)}_strong_full_model"},
              "strong_initialized_from_weak_checkpoint": str(weak_best.resolve()),
              "weak_checkpoint_sha256": sha256_file(weak_best),
              "lr_audit_status": lr_audit["status"]}
    target = _e5_fold_dir(config) / "local_sft.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    return result


def train_local_rewards(config_path: str | Path = "configs/full_model_backtest_tminus14.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    root = config.path.parent.parent
    run_root = _fold_root(config)
    results = {}
    for variant in ("fit", "audit"):
        run_dir = run_root / f"rm_{variant}_seed17"
        manifest_path = run_dir / "run_manifest.json"
        resume = run_dir / "last.pt" if (run_dir / "last.pt").is_file() else None
        results[variant] = _resume_model_run(manifest_path, run_dir, lambda variant=variant, run_dir=run_dir,
                                            resume=resume: run_reward_model(
            config_path, variant=variant, seed=17, max_targets=200000, validation_limit=40000,
            epochs=8, batch_size=64, run_dir=run_dir, validation_seed=17,
            early_stopping_patience=99, checkpoint_interval_steps=500, resume=resume,
        ))
        results[variant] = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {"protocol_id": PROTOCOL_ID, "fold": _fold_name(config),
              "fit": {"run_dir": str((run_root / "rm_fit_seed17").resolve()),
                      "checkpoint": str((run_root / "rm_fit_seed17/best.pt").resolve()),
                      "checkpoint_sha256": sha256_file(run_root / "rm_fit_seed17/best.pt"),
                      "status": results["fit"].get("status"),
                      "best_epoch": results["fit"].get("best_epoch"),
                      "best_ratio_rmse": results["fit"].get("best_ratio_rmse"),
                      "completed_target_presentations": results["fit"].get("completed_target_presentations"),
                      "unique_train_targets": results["fit"].get("unique_train_targets"),
                      "elapsed_seconds": results["fit"].get("elapsed_seconds")},
              "audit": {"run_dir": str((run_root / "rm_audit_seed17").resolve()),
                        "checkpoint": str((run_root / "rm_audit_seed17/best.pt").resolve()),
                        "checkpoint_sha256": sha256_file(run_root / "rm_audit_seed17/best.pt"),
                        "status": results["audit"].get("status"),
                        "best_epoch": results["audit"].get("best_epoch"),
                        "best_ratio_rmse": results["audit"].get("best_ratio_rmse"),
                        "completed_target_presentations": results["audit"].get("completed_target_presentations"),
                        "unique_train_targets": results["audit"].get("unique_train_targets"),
                        "elapsed_seconds": results["audit"].get("elapsed_seconds"),
                        "independent_policy_confirmation": False}}
    target = _e5_fold_dir(config) / "local_rm.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    return result


def evaluate_local_rm_gate(config_path: str | Path = "configs/full_model_backtest_tminus14.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    root = config.path.parent.parent
    run_root = _fold_root(config)
    outputs = {}
    for variant in ("fit", "audit"):
        checkpoint = run_root / f"rm_{variant}_seed17/best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output = _e5_fold_dir(config) / f"rm_diagnostics_{variant}.json"
        outputs[variant] = diagnose_rm_checkpoint(
            config_path, checkpoint, variant=variant, seed=17, train_limit=200000,
            validation_limit=40000, batch_size=64, bootstrap_repetitions=2000,
            output_path=output)
    fit_gate = bool(outputs["fit"]["all_items"][
        "paired_user_mse_difference_model_minus_item_mean"]["ci_95_high"] < 0)
    result = {"protocol_id": PROTOCOL_ID, "fold": _fold_name(config),
              "fit_q1_predictive_gate_pass": fit_gate,
              "gate_definition": "local V_RM paired macro-user MSE difference model minus same-partition smoothed item mean has 95% CI upper < 0",
              "fit": outputs["fit"], "audit_secondary_already_disclosed": outputs["audit"],
              "audit_independent_confirmation": False,
              "rl_transfer_status": "transfer_evidence_allowed_if_gate_passes" if fit_gate else "exploratory_only_gate_failed"}
    target = _e5_fold_dir(config) / "rm_gates.json"
    write_json_atomic(target, result)
    return result


def build_local_cache(config_path: str | Path = "configs/full_model_backtest_tminus14.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint = _fold_root(config) / "rm_fit_seed17/best.pt"
    panels = create_panels(config_path)
    cache = build_reward_cache(config_path, reward_checkpoint=checkpoint, variant="fit", reward_head="ratio")
    return {"panels": {key: value for key, value in panels.items() if key not in {
            "train", "calibration", "development", "confirmation"}},
            "cache": cache}


def run_local_rl(config_path: str | Path = "configs/full_model_backtest_tminus14.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    root = config.path.parent.parent
    artifacts = Path(config.project.get("artifacts_dir", "artifacts/full_model"))
    if not artifacts.is_absolute():
        artifacts = root / artifacts
    if not (artifacts / "candidate_freeze.json").is_file() or not (artifacts / "lambda_freeze.json").is_file():
        raise FileNotFoundError("main development freeze and lambda freeze are required before temporal transfer")
    model_root = _temporal_model_root(config)
    actor = model_root / "sft_strong_seed17/best.pt"
    rm_checkpoint = model_root / "rm_fit_seed17/best.pt"
    if not actor.is_file() or not rm_checkpoint.is_file():
        raise FileNotFoundError("local SFT and fit RM must complete before local RL")
    gate_path = _temporal_e5_source_dir(config) / "rm_gates.json"
    if not gate_path.is_file():
        raise FileNotFoundError("run the local fit-RM predictive gate before temporal RL")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    lambda_value = float(json.loads((artifacts / "lambda_freeze.json").read_text(encoding="utf-8"))["selected_lambda"])
    main_freeze = json.loads((artifacts / "candidate_freeze.json").read_text(encoding="utf-8"))
    recipes = {
        "sampled_ordinary": ("sampled", "ordinary", 0.0),
        "exact_ordinary": ("exact", "ordinary", 0.0),
        "sampled_penalized": ("sampled", "penalized", lambda_value),
    }
    runs = {}
    for label, (method, support, lam) in recipes.items():
        recipe_key = f"{method}:{support}:beta=1"
        selected_update = int(main_freeze["recipe_freezes"][recipe_key]["update"])
        for seed in (17, 42, 73):
            key = f"{label}_seed{seed}"
            # Exact T-14 ordinary trajectories optimize the same normalized
            # full-catalog objective as before this correction. Reuse their
            # frozen-transfer checkpoints; only sampled paths use the repaired
            # advantage units and need retraining.
            if method == "exact" and config.project.get("temporal_e5_source_root"):
                old_transfer = _temporal_e5_source_dir(config) / "rl_transfer.json"
                old = json.loads(old_transfer.read_text(encoding="utf-8"))
                old_record = old.get("runs", {}).get(key)
                if old_record is None:
                    raise FileNotFoundError(f"verified exact T-14 transfer is missing: {old_transfer}#{key}")
                old_run = Path(old_record["run_dir"])
                old_checkpoint = old_run / "snapshots" / f"update_{selected_update:05d}.pt"
                old_manifest_path = old_run / "run_manifest.json"
                if not old_checkpoint.is_file() or not old_manifest_path.is_file():
                    raise FileNotFoundError(f"exact T-14 source artifacts are missing for {key}")
                old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
                expected_actor_sha = sha256_file(actor)
                expected_reward_sha = sha256_file(rm_checkpoint)
                if (old_manifest.get("method") != "exact"
                        or old_manifest.get("support_mode") != "ordinary"
                        or float(old_manifest.get("beta", -1)) != 1.0
                        or old_manifest.get("reward_head") != "ratio"
                        or int(old_manifest.get("latest_update", -1)) < selected_update
                        or old_manifest.get("actor_checkpoint_sha256") != expected_actor_sha
                        or old_manifest.get("reward_checkpoint_sha256") != expected_reward_sha):
                    raise ValueError(f"exact T-14 source run provenance does not match the frozen recipe: {key}")
                if old_manifest.get("objective_version") not in (None, "normalized_reward_v1",
                                                                    "exact_normalized_reward_v1_verified"):
                    raise ValueError(f"exact T-14 source objective is incompatible for {key}")
                runs[key] = {**old_record, "run_dir": str(old_run.resolve()),
                             "checkpoint": str(old_checkpoint.resolve()),
                             "checkpoint_sha256": sha256_file(old_checkpoint),
                             "source_run_manifest_sha256": sha256_file(old_manifest_path),
                             "latest_update": old_manifest.get("latest_update"),
                             "transfer_update_from_main_development": selected_update,
                             "reused": True,
                             "reuse_reason": "exact normalized full-catalog objective unchanged",
                             "exploratory_due_to_failed_rm_gate": not bool(gate.get("fit_q1_predictive_gate_pass"))}
                continue
            run_dir = _fold_root(config) / f"q1_{method}_{support}_b1_s{seed}"
            manifest_path = run_dir / "run_manifest.json"
            resume = run_dir / "last.pt" if (run_dir / "last.pt").is_file() else None
            manifest = _resume_model_run(manifest_path, run_dir, lambda method=method, support=support,
                                         seed=seed, run_dir=run_dir, resume=resume, lam=lam: train_run(
                config_path, actor, method=method, support_mode=support, beta=1.0, seed=seed,
                reward_head="ratio", updates=1000, checkpoint_interval=100, eval_interval=1000,
                run_dir=run_dir, resume=resume, lambda_support_value=lam if support == "penalized" else 0.0,
                run_kind="E5_frozen_transfer", evaluate_development=False))
            checkpoint = run_dir / "snapshots" / f"update_{selected_update:05d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            runs[key] = {"run_dir": str(run_dir.resolve()),
                                          "checkpoint": str(checkpoint.resolve()),
                                          "checkpoint_sha256": sha256_file(checkpoint),
                                          "source_tree_sha256": manifest.get("source_tree_sha256"),
                                          "actor_checkpoint_sha256": manifest.get("actor_checkpoint_sha256"),
                                          "reward_checkpoint_sha256": manifest.get("reward_checkpoint_sha256"),
                                          "reward_cache_manifest_sha256": manifest.get("reward_cache_manifest_sha256"),
                                          "processed_listens_sha256": manifest.get("processed_listens_sha256"),
                                          "last_checkpoint_sha256": manifest.get("last_checkpoint_sha256"),
                                          "latest_update": manifest.get("latest_update"),
                                          "training_seconds": manifest.get("training_seconds"),
                                          "transfer_update_from_main_development": selected_update,
                                          "reused": False,
                                          "exploratory_due_to_failed_rm_gate": not bool(gate.get("fit_q1_predictive_gate_pass"))}
    result = {"protocol_id": PROTOCOL_ID, "fold": _fold_name(config),
              "status": "completed", "fit_rm_gate": bool(gate.get("fit_q1_predictive_gate_pass")),
              "failed_gate_rl_is_exploratory": not bool(gate.get("fit_q1_predictive_gate_pass")),
              "main_recipe_freeze_sha256": sha256_file(artifacts / "candidate_freeze.json"),
              "lambda_freeze_sha256": sha256_file(artifacts / "lambda_freeze.json"),
              "recipes": recipes, "runs": runs}
    target = artifacts / "e5" / _fold_name(config) / "rl_transfer.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    return result


def evaluate_local_validation(config_path: str | Path = "configs/full_model_backtest_tminus14.yaml") -> dict[str, Any]:
    import polars as pl
    from tlab.evaluation.baselines import _train_transition_matrix, rank_score_vector
    from tlab.evaluation.smoothed_markov import _evaluate_alpha

    config = load_config(config_path)
    root = config.path.parent.parent
    artifacts = Path(config.project.get("artifacts_dir", "artifacts/full_model"))
    if not artifacts.is_absolute():
        artifacts = root / artifacts
    results_path = artifacts / "e5" / _fold_name(config) / "local_validation_once.json"
    if results_path.exists():
        return json.loads(results_path.read_text(encoding="utf-8"))
    validation_started = time.perf_counter()
    panels = create_panels(config_path)
    rows = panels["development"] + panels["confirmation"]
    cache_manifest, cache_dir, _, arrays = _load_cache(config_path, "fit", "ratio")
    evaluation_rewards = arrays["evaluation"][:len(rows)]
    store, data_protocol = _load_store(Path(config.data["processed_dir"]) if Path(config.data["processed_dir"]).is_absolute()
                                      else root / config.data["processed_dir"])
    catalog = pl.read_parquet((root / config.data.get("processed_dir", "data/processed")) / "catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog.get_column("item_id").to_list(), dtype=np.int64)
    training_counts = np.asarray(catalog.get_column("training_count").to_list(), dtype=np.float64)
    anchors = [(int(row["user_index"]), int(row["position"])) for row in rows]
    transition, transition_examples = _train_transition_matrix(store, len(item_ids))
    markov = _evaluate_alpha(store, anchors, item_ids, training_counts,
                             catalog.get_column("catalog_duration_seconds").to_list(),
                             transition, 0.0, include_serving=False)
    ranks = markov["per_user_ranks"]
    markov_users = {int(uid): float(np.mean([1 / math.log2(int(rank) + 1) if rank is not None and int(rank) <= 10 else 0.0
                                             for rank in user_ranks]))
                    for uid, user_ranks in ranks.items()}
    # Uniform popularity uses local pre-cutoff training counts and the same causal panel.
    from tlab.evaluation.metrics import rank_target
    item_positions = {int(item): idx for idx, item in enumerate(item_ids)}
    popularity_users = {}
    for row in rows:
        target_id = int(store.users[int(row["user_index"])].raw_item_ids[int(row["position"])])
        target_index = item_positions.get(target_id)
        popularity_rank = rank_target(training_counts.tolist(), target_index, item_ids.tolist())
        popularity_users[int(row["uid"])] = 0.0 if popularity_rank is None or popularity_rank > 10 else 1 / math.log2(popularity_rank + 1)
    sft_checkpoint = _temporal_model_root(config) / "sft_strong_seed17/best.pt"
    support = _feature_support(root / config.data.get("processed_dir", "data/processed"))
    models = {"local SFT": sft_checkpoint}
    rl_manifest = json.loads((artifacts / "e5" / _fold_name(config) / "rl_transfer.json").read_text(encoding="utf-8"))
    recipe_freeze = json.loads((artifacts / "candidate_freeze.json").read_text(encoding="utf-8"))["recipe_freezes"]
    for run_key, record in rl_manifest["runs"].items():
        label = run_key.rsplit("_seed", 1)[0]
        models.setdefault(label, {})
        models[label][int(run_key.rsplit("_seed", 1)[1])] = Path(record["checkpoint"])
    model_results = {}
    sft_result = _policy_metrics(config_path, sft_checkpoint, rows, evaluation_rewards, support)
    model_results["local SFT"] = sft_result
    for recipe_label, seed_paths in models.items():
        if recipe_label == "local SFT":
            continue
        model_results[recipe_label] = {}
        for seed, checkpoint in sorted(seed_paths.items()):
            result = _policy_metrics(config_path, checkpoint, rows, evaluation_rewards, support)
            common = set(result["per_user_metrics"]) & set(sft_result["per_user_metrics"])
            result["paired_to_local_sft"] = {
                metric: paired_user_bootstrap({uid: result["per_user_metrics"][uid][metric]
                                               - sft_result["per_user_metrics"][uid][metric]
                                               for uid in common}, repetitions=2000, seed=47100 + seed)
                for metric in ("ndcg@10", "expected_raw_reward", "unsupported_mass")}
            model_results[recipe_label][str(seed)] = result
    baseline = {"markov_alpha_0": {"ndcg@10": float(np.mean(list(markov_users.values()))),
                                   "per_user_ndcg10": markov_users,
                                   "transition_training_examples": transition_examples},
                "popularity": {"ndcg@10": float(np.mean(list(popularity_users.values()))),
                               "per_user_ndcg10": popularity_users}}
    freeze = {"protocol_id": PROTOCOL_ID, "fold": _fold_name(config), "status": "validation_transfer_frozen",
              "main_recipe_freeze_sha256": sha256_file(artifacts / "candidate_freeze.json"),
              "local_data_sha256": json.loads((root / config.data["processed_dir"] / "protocol.json").read_text(encoding="utf-8")).get("processed_listens_sha256"),
              "transfer_checkpoint_updates": {key: int(value["update"]) for key, value in recipe_freeze.items()
                                               if key in {"sampled:ordinary:beta=1", "exact:ordinary:beta=1", "sampled:penalized:beta=1"}},
              "test_opened": False}
    freeze_path = artifacts / "e5" / _fold_name(config) / "local_recipe_freeze.json"
    write_json_atomic(freeze_path, freeze)
    result = {"protocol_id": PROTOCOL_ID, "fold": _fold_name(config), "status": "completed",
              "panel_ids_sha256": panels["panel_ids_sha256"], "panel_status": "local validation only; test unopened",
              "fit_rm_gate": json.loads((_temporal_e5_source_dir(config) / "rm_gates.json").read_text(encoding="utf-8")).get("fit_q1_predictive_gate_pass"),
              "baseline_results": baseline,
              "models": {label: ({key: value for key, value in values.items() if key != "per_user_metrics"}
                                  if label == "local SFT" else {
                                      seed: {key: value for key, value in result.items() if key != "per_user_metrics"}
                                      for seed, result in values.items()})
                         for label, values in model_results.items()},
              "local_recipe_freeze": freeze}
    write_json_atomic(results_path, result)
    write_json_atomic(artifacts / "e5" / _fold_name(config) / "local_validation_timing.json",
                      {"protocol_id": PROTOCOL_ID, "fold": _fold_name(config),
                       "elapsed_seconds": time.perf_counter() - validation_started,
                       "measurement_scope": "local validation panel creation through result write",
                       "panel_ids_sha256": panels["panel_ids_sha256"], "test_opened": False})
    return result


def _score_test_rewards(config, rows, *, batch_size: int = 8) -> tuple[np.ndarray, dict[str, Any]]:
    from tlab.experiments.full_model_matrix import _features
    config_obj, processed, artifacts = _root_and_processed(config)
    store, protocol = _load_store(processed)
    cache_dir = artifacts / "e5" / _fold_name(config_obj) / "test_reward_cache"
    output = cache_dir / "fit_rewards.npy"
    metadata_path = cache_dir / "manifest.json"
    source_cache_dir = _temporal_e5_source_dir(config_obj) / "test_reward_cache"
    source_output = source_cache_dir / "fit_rewards.npy"
    source_metadata = source_cache_dir / "manifest.json"
    rm_checkpoint = _temporal_model_root(config_obj) / "rm_fit_seed17/best.pt"
    checkpoint_sha = sha256_file(rm_checkpoint)
    row_hash = hashlib.sha256("\n".join(str(row["anchor_id"]) for row in rows).encode()).hexdigest()
    for candidate_meta, candidate_output in ((metadata_path, output), (source_metadata, source_output)):
        if candidate_meta.is_file() and candidate_output.is_file():
            manifest = json.loads(candidate_meta.read_text(encoding="utf-8"))
            if (manifest.get("checkpoint_sha256") == checkpoint_sha and manifest.get("panel_sha256") == row_hash
                    and manifest.get("processed_listens_sha256") == protocol["processed_listens_sha256"]
                    and sha256_file(candidate_output) == manifest.get("matrix_sha256")):
                return np.load(candidate_output, mmap_mode="r", allow_pickle=False), manifest
    cache_dir.mkdir(parents=True, exist_ok=True)
    import polars as pl
    features = _features(config_obj, processed, "fit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_reward_model(config_obj, rm_checkpoint, "fit", device)
    model.eval()
    item_tokens = torch.arange(3, 3 + int(protocol["catalog_size"]), device=device).unsqueeze(0)
    item_ids = np.arange(3, 3 + int(protocol["catalog_size"]), dtype=np.int64)
    durations = torch.tensor([features.get(int(token), (0.0, True, 0))[0] for token in item_tokens[0].tolist()],
                             dtype=torch.float32, device=device).unsqueeze(0)
    missing = torch.tensor([features.get(int(token), (0.0, True, 0))[1] for token in item_tokens[0].tolist()],
                           dtype=torch.bool, device=device).unsqueeze(0)
    histories = [store.users[int(row["user_index"])].history(int(row["position"]),
                 int(config_obj.dataset.get("history_length", 201))) for row in rows]
    mmap = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32,
                                     shape=(len(rows), int(protocol["catalog_size"])))
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = histories[start:start + batch_size]
        ids, attention = _collate(batch, device)
        candidates = item_tokens.expand(len(batch), -1)
        repeats = torch.as_tensor(np.stack([np.isin(item_ids, np.asarray(history[-20:], dtype=np.int64))
                                            for history in batch]), dtype=torch.bool, device=device)
        prediction = model.score_catalog(ids, attention, candidates, durations.expand(len(batch), -1),
                                         missing.expand(len(batch), -1), repeats,
                                         chunk_size=int(config_obj.training.get("rl_reward_catalog_chunk_size", 512)))
        mmap[start:start + len(batch)] = prediction["ratio"].float().cpu().numpy()
        if start % (batch_size * 128) == 0:
            mmap.flush()
    mmap.flush()
    manifest = {"protocol_id": PROTOCOL_ID, "checkpoint": str(rm_checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha, "panel_sha256": row_hash,
                "processed_listens_sha256": protocol["processed_listens_sha256"],
                "shape": [len(rows), int(protocol["catalog_size"])], "bytes": output.stat().st_size,
                "matrix_sha256": sha256_file(output), "elapsed_seconds": time.perf_counter() - started,
                "catalog_chunk_size": int(config_obj.training.get("rl_reward_catalog_chunk_size", 512))}
    write_json_atomic(metadata_path, manifest)
    return np.load(output, mmap_mode="r", allow_pickle=False), manifest


def run_final_test_once(config_path: str | Path = "configs/full_model_backtest_tminus14.yaml") -> dict[str, Any]:
    import polars as pl
    from tlab.evaluation.baselines import _train_transition_matrix, rank_score_vector
    from tlab.evaluation.metrics import rank_target
    from tlab.evaluation.smoothed_markov import _evaluate_alpha

    config = load_config(config_path)
    root = config.path.parent.parent
    artifacts = Path(config.project.get("artifacts_dir", "artifacts/full_model"))
    if not artifacts.is_absolute():
        artifacts = root / artifacts
    fold_dir = artifacts / "e5" / _fold_name(config)
    output_path = fold_dir / "final_test_once.json"
    if output_path.exists():
        raise FileExistsError("the local final 7-day Test was already opened; it is immutable")
    freeze_path = fold_dir / "local_recipe_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("freeze the transferred validation recipe and checkpoints before opening local Test")
    evaluation_started = time.perf_counter()
    store, protocol = _load_store(Path(config.data["processed_dir"]) if Path(config.data["processed_dir"]).is_absolute()
                                 else root / config.data["processed_dir"])
    anchors = store.evaluation_targets("test", per_user_limit=1, seed=17)
    rows = [{"row": index, "user_index": int(user_index), "uid": int(store.users[user_index].uid),
             "position": int(position), "anchor_id": hashlib.sha256(
                 f"test:{store.users[user_index].uid}:{position}".encode()).hexdigest(),
             "target_token_id": int(store.users[user_index].tokens[position]),
             "target_raw_item_id": int(store.users[user_index].raw_item_ids[position])}
            for index, (user_index, position) in enumerate(anchors)]
    if not rows:
        raise ValueError("local final Test panel is empty")
    panel_path = fold_dir / "final_test_panel.json"
    panel_preexisting = panel_path.is_file()
    panel_payload = {"protocol_id": PROTOCOL_ID, "panel_status": "frozen before final model scoring",
                     "split": "local final seven days", "seed": 17, "targets_per_user": 1,
                     "actual_targets": len(rows), "unique_users": len({row["uid"] for row in rows}),
                     "panel_sha256": hashlib.sha256("\n".join(row["anchor_id"] for row in rows).encode()).hexdigest(),
                     "rows": rows}
    source_panel_path = _temporal_e5_source_dir(config) / "final_test_panel.json"
    panel_preexisting = source_panel_path.is_file() or panel_path.is_file()
    frozen_panel_path = source_panel_path if source_panel_path.is_file() else panel_path
    if frozen_panel_path.exists():
        existing_panel = json.loads(frozen_panel_path.read_text(encoding="utf-8"))
        if existing_panel.get("panel_sha256") != panel_payload["panel_sha256"] or existing_panel.get("rows") != rows:
            raise ValueError("the previously frozen local final Test panel differs; refusing to resample or overwrite")
        panel_payload = existing_panel
    if not panel_path.is_file():
        write_json_atomic(panel_path, panel_payload)
    reward_matrix, reward_manifest = _score_test_rewards(config.path, rows)
    support = _feature_support(root / config.data.get("processed_dir", "data/processed"))
    local_sft = _temporal_model_root(config) / "sft_strong_seed17/best.pt"
    models: dict[str, Path] = {"local SFT": local_sft}
    rl_transfer = json.loads((fold_dir / "rl_transfer.json").read_text(encoding="utf-8"))
    recipe_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen_updates = recipe_freeze["transfer_checkpoint_updates"]
    recipe_map = {"sampled_ordinary": "sampled:ordinary:beta=1",
                  "exact_ordinary": "exact:ordinary:beta=1",
                  "sampled_penalized": "sampled:penalized:beta=1"}
    actor_checkpoints = {}
    for run_key, record in rl_transfer["runs"].items():
        for name, recipe_key in recipe_map.items():
            if run_key.startswith(name):
                actor_checkpoints.setdefault(name, {})[run_key.rsplit("_seed", 1)[1]] = Path(record["checkpoint"])
    model_results: dict[str, Any] = {}
    panel_matrix = reward_matrix
    model_results["local SFT"] = _policy_metrics(config_path, local_sft, rows, panel_matrix, support)
    for recipe, seed_paths in actor_checkpoints.items():
        model_results[recipe] = {}
        for seed, checkpoint in sorted(seed_paths.items()):
            model_results[recipe][seed] = _policy_metrics(config_path, checkpoint, rows, panel_matrix, support)
            baseline_users = model_results["local SFT"]["per_user_metrics"]
            candidate_users = model_results[recipe][seed]["per_user_metrics"]
            common_users = set(baseline_users) & set(candidate_users)
            model_results[recipe][seed]["paired_to_local_sft"] = {
                metric: paired_user_bootstrap(
                    {uid: candidate_users[uid][metric] - baseline_users[uid][metric]
                     for uid in common_users}, repetitions=2000, seed=58100 + int(seed))
                for metric in ("ndcg@10", "expected_raw_reward", "unsupported_mass")}
    import polars as polars
    catalog = polars.read_parquet((root / config.data.get("processed_dir", "data/processed")) / "catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog.get_column("item_id").to_list(), dtype=np.int64)
    counts = np.asarray(catalog.get_column("training_count").to_list(), dtype=np.float64)
    anchors_t = [(row["user_index"], row["position"]) for row in rows]
    transition, transition_examples = _train_transition_matrix(store, len(item_ids))
    markov = _evaluate_alpha(store, anchors_t, item_ids, counts,
                             catalog.get_column("catalog_duration_seconds").to_list(), transition, 0.0,
                             include_serving=False)
    markov_users = {int(uid): float(np.mean([1 / math.log2(int(rank) + 1) if rank is not None and int(rank) <= 10 else 0.0
                                             for rank in ranks]))
                    for uid, ranks in markov["per_user_ranks"].items()}
    positions = {int(item): idx for idx, item in enumerate(item_ids)}
    popularity_users = {}
    for row in rows:
        target_id = int(row["target_raw_item_id"])
        target_index = positions.get(target_id)
        rank = rank_target(counts.tolist(), target_index, item_ids.tolist())
        popularity_users[int(row["uid"])] = 0.0 if rank is None or rank > 10 else 1 / math.log2(rank + 1)
    baselines = {"markov_alpha_0": {"ndcg@10": float(np.mean(list(markov_users.values()))),
                                   "per_user_ndcg10": markov_users,
                                   "transition_training_examples": transition_examples},
                 "popularity": {"ndcg@10": float(np.mean(list(popularity_users.values()))),
                                "per_user_ndcg10": popularity_users}}
    result = {"protocol_id": PROTOCOL_ID, "fold": _fold_name(config),
              "status": ("completed_disclosed_panel_reanalysis" if panel_preexisting else "completed_once_after_freeze"),
              "test_panel_sha256": panel_payload["panel_sha256"], "panel_targets": len(rows),
              "unique_users": len({row["uid"] for row in rows}),
              "local_test_split_used_once": not panel_preexisting, "main_reference_test_used": False,
              "panel_reused_from_existing_frozen_file": panel_preexisting,
              "independent_confirmation": False if panel_preexisting else None,
              "reward_cache": reward_manifest,
              "models": model_results, "baselines": baselines,
              "support_definition": "local fit RM F_RM support, zero count is unsupported",
              "local_recipe_freeze_sha256": sha256_file(freeze_path),
              "note": ("OOV targets remain misses in the full-catalog denominator; no Test-based tuning was performed. "
                       + ("This is an explicitly authorized reanalysis of a previously disclosed panel, not independent evidence."
                          if panel_preexisting else ""))}
    write_json_atomic(output_path, result)
    timing_path = fold_dir / "final_test_timing.json"
    write_json_atomic(timing_path, {"protocol_id": PROTOCOL_ID, "fold": _fold_name(config),
                                    "elapsed_seconds": time.perf_counter() - evaluation_started,
                                    "measurement_scope": "from frozen local panel loading through final result write",
                                    "panel_sha256": panel_payload["panel_sha256"],
                                    "panel_reused_from_existing_frozen_file": panel_preexisting,
                                    "independent_confirmation": False if panel_preexisting else None})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Cutoff-correct full_model.1 temporal backtest")
    parser.add_argument("--config", default="configs/full_model_backtest_tminus14.yaml")
    parser.add_argument("--stage", choices=("prepare", "train-sft", "train-rm", "rm-gate",
                                            "cache", "train-rl", "validation", "final-test"), required=True)
    args = parser.parse_args()
    stages = {"prepare": prepare_temporal_fold, "train-sft": train_local_baselines,
              "train-rm": train_local_rewards, "rm-gate": evaluate_local_rm_gate,
              "cache": build_local_cache, "train-rl": run_local_rl,
              "validation": evaluate_local_validation, "final-test": run_final_test_once}
    result = stages[args.stage](args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
