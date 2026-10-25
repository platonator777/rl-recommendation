from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tlab.config import load_config
from tlab.data.manifest import sha256_file, write_json_atomic
from tlab.experiments.full_model_matrix import create_panels
from tlab.train.checkpoint import source_tree_sha256


PROTOCOL_ID = "tlab_yambda_full_model_2026-10-14-r3.1"


def _project_root(config_path: str | Path) -> Path:
    return load_config(config_path).path.parent.parent


def _artifact_path(root: Path, config) -> Path:
    path = Path(config.project.get("artifacts_dir", "artifacts/full_model"))
    return path if path.is_absolute() else root / path


def _file_hash(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _planned_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend([
        {"run_id": "E1a_support_decomposition_reference_models", "stage": "E1a_pre_rl_analysis",
         "status": "planned", "selection_allowed": False},
        {"run_id": "E1a_rm_support_strata_fit_audit", "stage": "E1a_pre_rl_analysis",
         "validation_examples": 40000, "status": "planned", "selection_allowed": False},
    ])
    for method in ("sampled", "exact"):
        for support in ("ordinary", "penalized"):
            for beta in (0.1, 1.0, 3.0):
                rows.append({"run_id": f"q1_{method}_{support}_b{beta:g}_s17".replace(".", "p"),
                             "stage": "E1_E2_E3_core", "method": method, "support_mode": support,
                             "beta": beta, "seed": 17, "updates_planned": 1000, "status": "planned"})
    for method in ("sampled", "exact"):
        for support in ("ordinary", "penalized"):
            for seed in (42, 73):
                rows.append({"run_id": f"q1_{method}_{support}_b1_s{seed}",
                             "stage": "E3_confirmation", "method": method, "support_mode": support,
                             "beta": 1.0, "seed": seed, "updates_planned": 1000,
                             "validation": "confirmation only after freeze", "status": "planned"})
    seed = 90117
    for beta in (0.1, 1.0, 3.0):
        for support, values in (("ordinary", (0.0,)), ("penalized", (1.0, 3.0, 10.0))):
            for value in values:
                rows.append({"run_id": f"lambda_pilot_q1_{support}_b{beta:g}_l{value:g}_s{seed}".replace(".", "p"),
                             "stage": "E1_train_only_lambda_pilot", "method": "sampled",
                             "support_mode": support, "lambda": value, "beta": beta,
                             "seed": seed, "updates_planned": 100, "selection_data": "training probe only",
                             "status": "planned"})
    rows.append({"run_id": "q1_exact_penalized_b1_s90118", "stage": "E1_exact_penalty_pilot",
                 "method": "exact", "support_mode": "penalized", "lambda": "frozen after pilot",
                 "beta": 1.0, "seed": 90118, "updates_planned": 100, "status": "planned"})
    for method, support in (("sampled", "ordinary"), ("exact", "ordinary"),
                            ("sampled", "penalized"), ("exact", "penalized")):
        rows.append({"run_id": f"q1_{method}_{support}_b1_s17", "stage": "E2_long_cache_stress_extension",
                     "method": method, "support_mode": support, "beta": 1.0, "seed": 17,
                     "extend_existing_from": 1000, "total_updates": 5000,
                     "selection_allowed": False, "status": "conditional_extension_after_core"})
    for seed in (17, 42, 73):
        rows.append({"run_id": f"q1_sampled_ordinary_b1_s{seed}_stream", "stage": "E3_cached_stream_bridge",
                     "method": "sampled", "support_mode": "ordinary", "beta": 1.0,
                     "seed": seed, "updates_planned": 1000, "status": "planned"})
    for method in ("sampled", "exact"):
        rows.append({"run_id": f"q2_{method}_ordinary_b1_s17", "stage": "E3_Q2_control",
                     "method": method, "support_mode": "ordinary", "beta": 1.0,
                     "seed": 17, "updates_planned": 1000, "status": "planned"})
    rows.append({"run_id": "q1_sampled_ordinary_b1_s17_reuse4", "stage": "E3_reuse_control",
                 "method": "sampled", "support_mode": "ordinary", "beta": 1.0,
                 "seed": 17, "updates_planned": 1000, "optimizer_epochs_per_batch": 4,
                 "status": "planned"})
    for beta in (0.1, 1.0, 3.0):
        rows.append({"run_id": f"analytic_b5_q1_beta{beta:g}", "stage": "E3_frontier_control",
                     "beta": beta, "seed": 17, "status": "planned", "selection_allowed": False})
    rows.append({"run_id": "E4_h5_context_intervention", "stage": "E4_diagnostic",
                 "anchors_planned": 1024, "horizon": 5, "seeds": [17, 42, 73],
                 "status": "planned"})
    for fold, status in (("tminus14", "mandatory"), ("tminus28", "conditional")):
        rows.extend([
            {"run_id": f"{fold}_sft_weak_seed17", "stage": "E5_local_sft", "fold": fold,
             "targets_per_pass": 100000, "passes": 5, "requirement": status, "status": "planned"},
            {"run_id": f"{fold}_sft_strong_seed17", "stage": "E5_local_sft", "fold": fold,
             "targets_per_pass": "min(2000000,N_eligible)", "passes": 5,
             "schedule_horizon_passes": 10, "requirement": status, "status": "planned"},
            {"run_id": f"{fold}_rm_fit_seed17", "stage": "E5_local_rm", "fold": fold,
             "labels_per_pass": 200000, "passes": 8, "requirement": status, "status": "planned"},
            {"run_id": f"{fold}_rm_audit_seed17", "stage": "E5_local_rm", "fold": fold,
             "labels_per_pass": 200000, "passes": 8, "requirement": status,
             "selection_allowed": False, "status": "planned"},
        ])
        for method, support in (("sampled", "ordinary"), ("exact", "ordinary"),
                                ("sampled", "penalized")):
            for rollout_seed in (17, 42, 73):
                rows.append({"run_id": f"{fold}_q1_{method}_{support}_b1_s{rollout_seed}",
                             "stage": "E5_frozen_transfer", "fold": fold, "method": method,
                             "support_mode": support, "beta": 1.0, "seed": rollout_seed,
                             "updates_planned": 1000, "requirement": status, "status": "planned"})
    for method, support in (("sampled", "ordinary"), ("exact", "ordinary"),
                            ("sampled", "penalized")):
        rows.append({"run_id": f"main_pass5_q1_{method}_{support}_b1_s17", "stage": "E5_pass5_control",
                     "method": method, "support_mode": support, "beta": 1.0,
                     "seed": 17, "updates_planned": 1000, "status": "planned_if_lineage_verified"})
    return rows


def _hardware_inventory(root: Path) -> dict[str, Any]:
    import torch

    gpu = None
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "total_vram_bytes": int(total_bytes),
               "free_vram_bytes_at_inventory": int(free_bytes),
               "allocated_vram_bytes_at_inventory": int(torch.cuda.memory_allocated(0)),
               "torch_cuda": torch.version.cuda}
    try:
        import psutil
        memory = psutil.virtual_memory()
        ram = {"total_bytes": int(memory.total), "available_bytes_at_inventory": int(memory.available)}
    except ImportError:
        ram = None
    disk = shutil.disk_usage(root)
    return {"gpu": gpu, "ram": ram, "disk_free_bytes_at_inventory": int(disk.free),
            "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "inventory_at_utc": datetime.now(timezone.utc).isoformat()}


def _initial_resource_plan(root: Path, config, inventory: dict[str, Any]) -> dict[str, Any]:
    reference_plan_path = root / "artifacts/reference/resource_plan.json"
    reference_plan = json.loads(reference_plan_path.read_text(encoding="utf-8")) if reference_plan_path.is_file() else {}
    known = {
        "strong_sft_seed17": root / "artifacts/reference/runs/sft_strong_seed17/run_manifest.json",
        "weak_sft_seed17": root / "artifacts/reference/runs/sft_weak_seed17_mb16/run_manifest.json",
        "rm_fit_seed17": root / "artifacts/reference/runs/rm_reference_fit_seed17/run_manifest.json",
        "rm_audit_seed17": root / "artifacts/reference/runs/rm_reference_audit_seed17/run_manifest.json",
        "strong_h1_exact_500": root / "artifacts/reference/runs/grpo_reference_q1_strong17_beta1_exact_update500/run_manifest.json",
    }
    historical = {}
    for name, path in known.items():
        if path.is_file():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            historical[name] = {"manifest_path": str(path.resolve()), "manifest_sha256": sha256_file(path),
                                "elapsed_seconds": manifest.get("elapsed_seconds"),
                                "training_seconds": manifest.get("training_seconds"),
                                "status": manifest.get("status")}
    return {
        "protocol_id": PROTOCOL_ID, "status": "preregistered_before_full_model_training_pilots",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_inventory": inventory, "historical_reference_measured_runs": historical,
        "historical_reference_resource_plan_sha256": sha256_file(reference_plan_path) if reference_plan_path.is_file() else None,
        "planned_storage_bytes": {
            "main_q1_train_float32_matrix": 2048 * 20000 * 4,
            "main_q1_eval_float32_matrix": 1024 * 20000 * 4,
            "main_q1_context_schedule": 5000 * int(config.training.get("rl_batch_contexts", 16)) * 4,
            "note": "calibration, fit/audit caches and stream anchors are additional; all full_model outputs stay under artifacts/full_model",
        },
        "preregistered_budgets": {
            "lambda_sampled_pilot_runs_max": 12, "lambda_sampled_pilot_updates_each": 100,
            "lambda_exact_pilot_runs": 1, "lambda_exact_pilot_updates": 100,
            "main_q1_runs": 20, "main_q1_updates_each": 1000,
            "long_cache_extensions": 4, "long_cache_total_updates": 5000,
            "stream_bridge_runs": 3, "stream_bridge_updates_each": 1000,
            "q2_runs": 2, "q2_updates_each": 1000,
            "reuse4_runs": 1, "reuse4_context_batches": 1000,
            "e5_total_gpu_hours_reserved": 10,
            "e5_optional_second_fold_requires_20pct_margin_and_data_gate": True,
        },
        "resource_measurement_rule": "After the frozen pilot, record sampled/exact seconds per update, cache build time/bytes, and peak VRAM. Update forecasts before main training; estimates are not adjusted based on reward or relevance outcomes.",
        "initial_forecast_status": "reference runs and hardware measured; full_model throughput forecast will be populated from the fixed pilot before the main matrix",
        "reference_plan_reference_fields": {key: reference_plan.get(key) for key in (
            "measured_reference_sft_microbenchmark", "measured_matched_weak_seed17_microbatch16_pass1",
            "completed_strong_seed17_run", "completed_corrected_weak_seed17_run") if key in reference_plan},
    }


def freeze_protocol(config_path: str | Path = "configs/full_model_rl.yaml", *, overwrite: bool = False) -> dict[str, Any]:
    root = _project_root(config_path)
    config = load_config(config_path)
    artifacts = _artifact_path(root, config)
    protocol_path = artifacts / "protocol.json"
    if protocol_path.exists() and not overwrite:
        raise FileExistsError(f"full_model protocol is already frozen: {protocol_path}")
    # Regenerate the pre-existing exploratory panel manifest under the current
    # preregistration, while preserving the deterministic IDs from its seeds.
    panel = create_panels(config_path, force=True)
    panel_path = artifacts / "panels" / "main" / "panels.json"
    processed = Path(config.data.get("processed_dir", "data/processed"))
    if not processed.is_absolute():
        processed = root / processed
    data_protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    source = root / "data/raw/listens.parquet"
    catalog = processed / "catalog.parquet"
    processed_listens = processed / "listens.parquet"
    actor = root / "artifacts/reference/runs/sft_strong_seed17/best.pt"
    pass5 = root / "artifacts/reference/runs/sft_strong_seed17/pass_05.pt"
    rm_fit = root / "artifacts/reference/runs/rm_reference_fit_seed17/best.pt"
    rm_audit = root / "artifacts/reference/runs/rm_reference_audit_seed17/best.pt"
    configs = [root / value for value in (
        "configs/full_model_rl.yaml", "configs/full_model_exact.yaml", "configs/full_model_horizon.yaml",
        "configs/full_model_backtest_tminus14.yaml", "configs/full_model_backtest_tminus28.yaml")]
    planned_runs = _planned_runs()
    inventory = _hardware_inventory(root)
    resource_path = artifacts / "resource_plan.json"
    if resource_path.exists():
        raise FileExistsError(f"full_model resource plan already exists; inspect it before freezing: {resource_path}")
    resource_plan = _initial_resource_plan(root, config, inventory)
    write_json_atomic(resource_path, resource_plan)
    preregistered_resource_path = artifacts / "resource_plan_preregistered.json"
    write_json_atomic(preregistered_resource_path, resource_plan)
    protocol = {
        "protocol_id": PROTOCOL_ID, "protocol_version": "3.1",
        "status": "frozen_before_full_model_pilots_and_main_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_revision": "3.1; supersedes the earlier full_model plan",
        "source_documents": {
            name: _file_hash(root / name) for name in (
                "README.md", "RESEARCH_PLAN_full_model.md", "REPORT_reference.md", "REPORT.md", "RUNBOOK_reference.md",
                "RUNBOOK_full_model.md")
        },
        "dataset": {
            "repository": "yandex/yambda", "revision": "dd6f3a19eef5866e346c3270e098baa641a44948",
            "raw_listens": _file_hash(source), "processed_listens": _file_hash(processed_listens),
            "catalog_file": _file_hash(catalog), "processed_protocol_sha256": sha256_file(processed / "protocol.json"),
            "processed_listens_sha256": data_protocol["processed_listens_sha256"],
            "catalog_sha256": data_protocol["catalog_sha256"], "catalog_size": data_protocol["catalog_size"],
            "original_end_raw": data_protocol["boundaries_source_units"]["end_raw"],
            "timestamp_unit_seconds": data_protocol["timestamp_unit_seconds"],
            "timestamp_bin_seconds": data_protocol["timestamp_bin_seconds"],
            "main_split_status": "known reference split; historical Test is disclosed and not rerun",
        },
        "models": {
            "actor": {**_file_hash(actor), "architecture": "GPT-2 4 layers × 256 hidden, 4 heads, tied embeddings",
                      "parameter_count": 8345856, "seed": 17, "fixed_in_E1_to_E4": True},
            "actor_pass5_learning_budget_control": _file_hash(pass5) if pass5.is_file() else None,
            "rm_fit": {**_file_hash(rm_fit), "variant": "fit", "selection_allowed": True},
            "rm_audit": {**_file_hash(rm_audit), "variant": "audit", "previously_revealed": True,
                         "selection_allowed": False, "independent_confirmation": False},
        },
        "code_and_configs": {
            "source_tree_sha256": source_tree_sha256(root / "src/tlab"),
            "configs": {path.name: _file_hash(path) for path in configs},
            "plan_sha256": sha256_file(root / "RESEARCH_PLAN_full_model.md"),
            "hypotheses_sha256": sha256_file(artifacts / "hypotheses.md"),
        },
        "panels": {"manifest": _file_hash(panel_path), "actual_sizes": panel["actual_sizes"],
                   "panel_ids_sha256": panel["panel_ids_sha256"], "user_overlap": panel["user_overlap"],
                   "panel_seed": panel["panel_seed"], "validation_seed": panel["validation_seed"],
                   "contract": panel["panel_contract"]},
        "selection": {
            "primary_metric": "unconditional macro-user full-catalog NDCG@10; OOC targets are misses",
            "secondary_metrics": ["Recall@10", "Recall@20", "Recall@100", "NDCG@20", "NDCG@100"],
            "paired_user_bootstrap_repetitions": 2000, "confidence_level": 0.95,
            "ndcg_relative_noninferiority_margin": -0.02,
            "behaviour_guardrails": {"exposure_coverage_relative_min": -0.10,
                                     "head_share_delta_max": 0.02, "history_repeat_delta_max": 0.02},
            "candidate_checkpoints": [100, 500, 1000], "primary_optimizer_endpoint": 1000,
            "development_selection": "all point guardrails; maximum fit expected raw reward; ties lower KL then earlier checkpoint; SFT fallback",
            "full_candidate_requires": "positive paired heldout fit-reward CI and relative NDCG CI lower bound >= -2%",
            "confirmation_panel": "pre-frozen known-validation partition; organizational split, not untouched",
            "primary_holm_family": ["sampled_Q1_minus_SFT", "exact_ordinary_minus_sampled_ordinary",
                                    "sampled_penalized_minus_sampled_ordinary"],
            "test_used_for_selection": False, "test_rerun": False,
            "rm_audit_scores_used_for_selection": False,
        },
        "objectives": {
            "ordinary": "E_s[sum_a pi(a|s)*r_norm(s,a) - beta*KL(pi||pi_SFT)]",
            "support_penalized": "ordinary J - lambda*E_s[relu(m_U(pi,s)-m_U(pi_SFT,s))]",
            "support_mask": "fit-F_RM item event count == 0; diagnostic groups 0,1-9,10-99,100+",
            "support_gradient": "direct full-catalog softmax derivative for sampled and exact optimizers; never action-sample estimate",
            "sampled_estimator": "G=8 fresh IID full-catalog draws, leave-one-out advantages, reuse=1 core",
            "exact_estimator": "differentiable full-catalog actor expectation; detached reward and reference",
            "reward_scale": "frozen from a disjoint train-only calibration user panel; own scale per reward head",
            "q1_primary": "reference RM_fit predicted listening ratio",
            "q2_control": "reference RM_fit predicted seconds, separate train-only scale, 1,000 updates",
            "lambda_pilot": {"seed": 90117, "betas": [0.1, 1.0, 3.0], "lambda_values": [1.0, 3.0, 10.0],
                             "updates": 100, "probe_updates": [80, 90, 100],
                             "informative_excess_threshold": 1e-4,
                             "selection": "minimum common lambda reducing excess >=50% for every informative beta; if none informative use 1; if none pass use 10 and flag failure",
                             "data_role": "training-only; no validation, confirmation, audit, or Test scores"},
        },
        "experiments": {
            "E1_E2_E3_q1_core": {"development_runs": 12, "confirmation_runs": 8,
                                  "optimizer_support_beta": "sampled/exact × ordinary/penalized × {0.1,1,3}",
                                  "development_seed": 17, "confirmation_seeds": [42, 73],
                                  "updates": 1000, "long_cache_stress": "four seed17 beta1 core runs extend from update1000 to 2500/5000; excluded from selection"},
            "E2_cached_stream_bridge": {"seeds": [17, 42, 73], "updates": 1000,
                                        "user_schedule": "matched by seed to cached ordinary sampled runs; new A_RL anchors from same train users"},
            "E3_q2": {"methods": ["sampled", "exact"], "seed": 17, "beta": 1.0, "updates": 1000},
            "E3_reuse": {"seed": 17, "beta": 1.0, "fresh_context_batches": 1000,
                          "optimizer_epochs_per_batch": 4},
            "E4": {"anchors_max": 1024, "horizon": 5, "seeds": [17, 42, 73],
                   "modes": ["fixed_history", "naive_append", "logged_teacher_forcing"],
                   "same_items_cross_context": True, "new_h5_training": False},
        },
        "e5": {
            "mandatory_fold": "T-14 days",
            "optional_fold": "T-28 days; run only if data gates pass and full forecast plus 20% fits remaining E5 reserve",
            "fold_local_retraining": "weak SFT seed17 5×100k; strong SFT seed17 five additional passes up to 2M targets/pass with the original 10-pass LR schedule; local fit/audit RM 200k/pass×8",
            "methods": ["sampled ordinary Q1 beta1", "exact ordinary Q1 beta1", "sampled support-penalized Q1 beta1"],
            "rollout_seeds": [17, 42, 73], "updates_max": 1000,
            "evaluation": "local SFT, popularity, Markov, and frozen policies on local final seven days once after freeze",
            "future_rows_excluded_from_prepared_fold": True,
            "pass5_control": "available only if reference pass_05 lineage verifies; three seed17 frozen RL recipes with pass10 RM/scale/mask and same panels",
            "reserved_gpu_hours_total_for_e5_and_pass5": 10.0,
        },
        "resource_inventory": inventory,
        "resource_plan": {"path": str(resource_path.resolve()),
                          "preregistered_copy_path": str(preregistered_resource_path.resolve()),
                          "preregistration_sha256": sha256_file(preregistered_resource_path),
                          "e5_gpu_hours_reserved": 10},
        "planned_runs": planned_runs,
        "disclosures": {
            "historical_test": "already revealed; full_model is post-test research, not independent confirmation",
            "rm_audit": "previously revealed; secondary diagnostic only; agreement is not independent evidence",
            "validation_repartition": "known validation; organizational split only",
            "sft_user_overlap": "evaluation histories may share users with prior SFT training history; not user-cold-start",
            "rollout_seeds": "vary RL randomness only, not SFT/RM training replications",
        },
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    write_json_atomic(protocol_path, protocol)
    registry = {"protocol_id": PROTOCOL_ID, "status": "frozen_before_pilot_and_main_training",
                "runs": planned_runs, "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(artifacts / "run_registry.json", registry)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze full_model.1 protocol, hashes, panels and planned runs")
    parser.add_argument("--config", default="configs/full_model_rl.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = freeze_protocol(args.config, overwrite=args.overwrite)
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["status"],
                      "protocol_path": str((_artifact_path(_project_root(args.config), load_config(args.config)) / "protocol.json").resolve()),
                      "panel_ids_sha256": result["panels"]["panel_ids_sha256"],
                      "source_tree_sha256": result["code_and_configs"]["source_tree_sha256"],
                      "planned_run_count": len(result["planned_runs"])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
