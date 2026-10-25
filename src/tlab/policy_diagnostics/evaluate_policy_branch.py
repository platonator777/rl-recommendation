"""Freeze and evaluate every registered sample-count checkpoint on fixed panels."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import torch
import yaml

from tlab.panel_tools.statistics import paired_user_bootstrap, relative_quality_guard
from tlab.residual_support.features import POLICY_FEATURE_NAMES
from tlab.residual_support.objectives import FrozenRewardScale, exact_support_statistics
from tlab.residual_support.scaler import TrainOnlyFeatureScaler
from tlab.residual_support.train_policy import _load_cache

from .complete_evaluation import PANEL_DIRS, RankAccumulator, _eval_cached_policy, _predict_checkpoint
from .protocol import ARTIFACTS, ROOT, record_operation, update_stage


BRANCH_ROOT = ARTIFACTS / "branches/sample_count_4_vs_16"
ARMS = ("sampled4", "sampled16", "exact")
ENDPOINTS = ("last", "best")
BOOTSTRAPS = 2000


def _effective_endpoint(requested: str, step: int) -> str:
    return "identity_step0" if requested == "best" and int(step) == 0 else requested


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _scaler(path: Path) -> TrainOnlyFeatureScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("sha256", None)
    for field in ("feature_names", "log1p_columns", "mean", "scale", "constant_columns"):
        payload[field] = tuple(payload[field])
    return TrainOnlyFeatureScaler(**payload)


def _exact_metrics(q: np.ndarray, arrays: dict[str, Any], scale: FrozenRewardScale,
                   beta: float, device: torch.device) -> dict[str, float]:
    accum: dict[str, float] = defaultdict(float)
    count = len(q)
    for start in range(0, count, 128):
        stop = min(count, start + 128)
        probabilities = torch.as_tensor(np.asarray(q[start:stop]), dtype=torch.float32, device=device)
        log_q = torch.log(probabilities.clamp_min(torch.finfo(torch.float32).tiny))
        ref = torch.as_tensor(np.asarray(arrays["reference_conditional_logp"][start:stop]),
                              dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.asarray(arrays["raw_candidate_rewards"][start:stop]),
                                  dtype=torch.float32, device=device)
        mass = torch.as_tensor(np.asarray(arrays["support_mass"][start:stop]), dtype=torch.float32, device=device)
        result = exact_support_statistics(log_q, ref, rewards, mass, scale, beta)
        result = exact_support_statistics(log_q, ref, rewards, mass, scale, beta,
                                          log_target=result["analytic_target_log_policy"])
        for key in (
            "support_weighted_objective", "support_weighted_reward_expectation",
            "kl_policy_to_reference_conditional", "kl_policy_to_reference_mass_weighted",
            "kl_target_to_policy_conditional", "kl_target_to_policy_mass_weighted",
            "analytic_objective_gap", "analytic_support_weighted_objective",
        ):
            accum[key] += float(result[key].sum().detach().cpu())
    return {key: value / count for key, value in accum.items()}


def _audit_predictions(model: Any, arrays: dict[str, Any], chunk: int = 30000) -> np.ndarray:
    features = arrays["candidate_features"]
    contexts, support, _ = features.shape
    if features.shape[-1] < 35 or getattr(model, "n_features_in_", 35) != 35:
        raise ValueError("external audit feature schema is incompatible with cached panel_tools.1 candidate columns")
    flat_count = contexts * support
    predictions = np.empty(flat_count, dtype=np.float32)
    for start in range(0, flat_count, chunk):
        stop = min(flat_count, start + chunk)
        x = np.asarray(features.reshape(flat_count, features.shape[-1])[start:stop, :35], dtype=np.float32)
        predictions[start:stop] = np.clip(model.predict(x), 0.0, 1.0)
    return predictions.reshape(contexts, support)


def _support_audit(q: np.ndarray, arrays: dict[str, Any], audit: np.ndarray) -> dict[str, float]:
    mass = np.asarray(arrays["support_mass"], dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    ref = np.exp(np.asarray(arrays["reference_conditional_logp"], dtype=np.float64))
    ref /= ref.sum(axis=1, keepdims=True)
    proxy = np.asarray(arrays["raw_candidate_rewards"], dtype=np.float64)
    audit = np.asarray(audit, dtype=np.float64)
    pq = np.sum(q * proxy, axis=1)
    pp = np.sum(ref * proxy, axis=1)
    aq = np.sum(q * audit, axis=1)
    ap = np.sum(ref * audit, axis=1)
    return {
        "mean_proxy_raw_conditional": float(pq.mean()),
        "mean_audit_raw_conditional": float(aq.mean()),
        "mean_proxy_raw_delta_conditional": float((pq - pp).mean()),
        "mean_proxy_raw_delta_mass_weighted": float(np.mean(mass * (pq - pp))),
        "mean_audit_raw_delta_conditional": float((aq - ap).mean()),
        "mean_audit_raw_delta_mass_weighted": float(np.mean(mass * (aq - ap))),
        "mean_support_mass": float(mass.mean()),
        "full_absolute_expectation_includes_unestimated_tail": True,
    }


def _primary_maps(per_user_path: Path) -> dict[tuple[str, str, int | None], dict[int, float]]:
    result: dict[tuple[str, str, int | None], dict[int, float]] = defaultdict(dict)
    for row in _read_csv(per_user_path):
        panel = row["panel"]
        method = row["method"]
        endpoint = row.get("endpoint", "")
        seed_value = row.get("seed", "")
        seed = None if seed_value in (None, "") else int(seed_value)
        result[(panel, f"{method}:{endpoint}", seed)][int(row["uid"])] = float(row["ndcg10"])
    return result


def _seed_mean_map(maps: list[dict[int, float]]) -> dict[int, float]:
    common = set(maps[0])
    for item in maps[1:]:
        common.intersection_update(item)
    return {uid: float(np.mean([item[uid] for item in maps])) for uid in sorted(common)}


def _contrast(left: dict[int, float], right: dict[int, float], seed: int) -> dict[str, Any]:
    ci = paired_user_bootstrap(left, right, n_bootstrap=BOOTSTRAPS, seed=seed)
    return {
        "n_users": ci["n_users"],
        "mean_difference": ci["mean_difference"],
        "ci_lower": ci["ci_lower"],
        "ci_upper": ci["ci_upper"],
        "one_sided_lower": ci["one_sided_lower"],
        "bootstrap_draws": ci["n_bootstrap"],
        "bootstrap_unit": "user after averaging fixed-seed policy scores per user",
    }


def evaluate() -> dict[str, Any]:
    started = time.perf_counter()
    registration = json.loads((ARTIFACTS / "branches/branch_registration.json").read_text(encoding="utf-8"))
    fit_summary_path = BRANCH_ROOT / "branch_fit_summary.json"
    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    if fit_summary.get("run_count") != 9 or fit_summary.get("planned_run_count") != 9:
        raise RuntimeError("all 9 registered fits must finish before branch evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rm_info = json.loads((ROOT / "artifacts/residual_support/rm/rm_diagnostics.json").read_text(encoding="utf-8"))
    reward_scale = json.loads((ROOT / "artifacts/residual_support/rm/reward_scale.json").read_text(encoding="utf-8"))
    scale = FrozenRewardScale(float(reward_scale["mean"]), float(reward_scale["std"]),
                              float(reward_scale["std_floor"]), str(reward_scale["version"]))
    rm_fit_counts = np.asarray(rm_info["rm_fit_item_count"]["counts_by_catalog_item"], dtype=np.int64)
    core = yaml.safe_load((ROOT / "configs/residual_support/core.yaml").read_text(encoding="utf-8"))
    catalog = pl.read_parquet(ROOT / core["catalog"]).sort("token_id")
    item_ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    audit_model = joblib.load(ROOT / "artifacts/residual_support/rm/models/external_audit_predictor.joblib")

    panel_arrays: dict[str, tuple[dict[str, Any], dict[str, Any], list[Any]]] = {}
    panel_audit: dict[str, np.ndarray] = {}
    panel_dirs = dict(PANEL_DIRS)
    panel_dirs["train"] = ROOT / "artifacts/residual_support/cache/train"
    for panel_name in ("train", "development", "retrospective_final", "common_open_6655"):
        panel_dir = panel_dirs[panel_name]
        meta, arrays, targets = _load_cache(panel_dir, include_full=panel_name != "train")
        panel_arrays[panel_name] = (meta, arrays, targets)
        if panel_name != "train":
            panel_audit[panel_name] = _audit_predictions(audit_model, arrays)

    quality_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    per_user_path = BRANCH_ROOT / "branch_per_user.csv"
    top_path = BRANCH_ROOT / "branch_top10.csv"
    per_fields = ["panel", "method", "endpoint", "seed", "uid", "event_id", "rank", "ndcg10", "recall10",
                  "target_raw_id", "target_in_catalog", "target_in_support", "positive", "is_new", "is_repeat",
                  "rm_fit_support_group"]
    top_fields = ["panel", "method", "endpoint", "seed", "uid", "list_rank", "catalog_index", "raw_item_id", "score"]
    user_maps: dict[tuple[str, str, str, int], dict[int, float]] = {}
    trained_best_steps: dict[tuple[str, int], int] = {}
    with per_user_path.open("w", newline="", encoding="utf-8") as per_handle, top_path.open(
        "w", newline="", encoding="utf-8"
    ) as top_handle:
        per_writer = csv.DictWriter(per_handle, fieldnames=per_fields)
        top_writer = csv.DictWriter(top_handle, fieldnames=top_fields)
        per_writer.writeheader()
        top_writer.writeheader()
        for arm in ARMS:
            config_path = ROOT / f"configs/policy_diagnostics/sample_count_{arm}.yaml"
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            run_root = ROOT / cfg["outputs"]["run_root"]
            scaler = _scaler(run_root / "scaler.json")
            beta = float(cfg["train"]["beta_values"][0])
            objective = str(cfg["train"]["objectives"][0])
            arm_method = arm
            for run_dir in sorted((run_root / "runs").iterdir()):
                manifest_path = run_dir / "run_manifest.json"
                if not manifest_path.exists():
                    raise FileNotFoundError(f"missing completed run manifest: {manifest_path}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                seed = int(manifest["seed"])
                for endpoint in ENDPOINTS:
                    checkpoint = run_dir / f"{endpoint}.pt"
                    if not checkpoint.exists():
                        raise FileNotFoundError(f"missing {endpoint} checkpoint: {checkpoint}")
                    for panel_name, (meta, arrays, targets) in panel_arrays.items():
                        q, step = _predict_checkpoint(checkpoint, manifest, arrays, scaler, device)
                        effective_endpoint = _effective_endpoint(endpoint, step)
                        if endpoint == "best":
                            trained_best_steps[(arm, seed)] = step
                        metrics = _exact_metrics(q, arrays, scale, beta, device)
                        support_rows.append({"panel": panel_name, "method": arm_method,
                            "checkpoint_endpoint": endpoint, "effective_endpoint": effective_endpoint,
                            "endpoint": effective_endpoint, "seed": seed, "training_step": step,
                            "objective": objective, "beta": beta, **metrics})
                        if panel_name == "train":
                            continue
                        audit = panel_audit[panel_name]
                        audit_values = _support_audit(q, arrays, audit)
                        audit_rows.append({"panel": panel_name, "method": arm_method,
                            "checkpoint_endpoint": endpoint, "effective_endpoint": effective_endpoint,
                            "seed": seed, "training_step": step, "objective": objective, "beta": beta,
                            **audit_values})
                        result = _eval_cached_policy(
                            panel_name, targets, arrays, q, item_ids, rm_fit_counts,
                            arm_method, effective_endpoint, seed, per_writer, top_writer,
                        )
                        primary = result["primary_user_macro"]
                        quality_rows.append({
                            "panel": panel_name, "method": arm_method, "objective": objective,
                            "endpoint": effective_endpoint, "seed": seed, "training_step": step,
                            "updates_completed": manifest["updates_completed"],
                            "run_elapsed_training_and_dev_evaluation_seconds": manifest["elapsed_training_and_evaluation_seconds"],
                            "n_users": primary["n_users"], "n_events": primary["n_events"],
                            "ndcg10": primary["ndcg@10"], "recall10": primary["recall@10"],
                            "target_catalog_coverage": result["target_catalog_coverage"],
                            "target_in_optimized_support_fraction": result["target_in_optimized_support_fraction"],
                            "oov_targets": result["oov_targets"],
                            "dev_guard_passes_at_checkpoint": manifest.get("best_checkpoint_paired_2_percent_guard_vs_strong", {}).get("passes")
                                if endpoint == "best" else None,
                            **metrics,
                        })
                        user_maps[(panel_name, arm_method, effective_endpoint, seed)] = result["per_user_primary_ndcg"]
                        if panel_name == "development":
                            rows = result["per_user_primary_ndcg"]
                            reference = {int(row["uid"]): float(row["ndcg10"]) for row in _read_csv(ROOT / "artifacts/policy_diagnostics/evaluation/per_user.csv")
                                         if row["panel"] == "development" and row["method"] == "strong_gpt"}
                            if reference:
                                guard = relative_quality_guard(rows, reference, tolerance=0.02,
                                                               n_bootstrap=BOOTSTRAPS, seed=seed + 827)
                                quality_rows[-1]["relative_2pct_guard_passes"] = guard["passes"]
                                quality_rows[-1]["relative_2pct_guard_lower_bound"] = guard["lower_bound"]

    # Compare policies only after averaging the three fixed seeds within each user.
    seed_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for panel_name in ("development", "retrospective_final", "common_open_6655"):
        for endpoint in ("last", "best"):
            maps_by_arm: dict[str, list[dict[int, float]]] = {}
            for arm in ARMS:
                maps = [user_maps.get((panel_name, arm, endpoint, seed)) for seed in (17, 42, 73)]
                if all(item is not None for item in maps):
                    maps_by_arm[arm] = [item for item in maps if item is not None]
                    values = [float(np.mean(list(item.values()))) for item in maps_by_arm[arm]]
                    seed_rows.append({"panel": panel_name, "method": arm, "endpoint": endpoint,
                        "seed_values": json.dumps(values), "seed_17": values[0], "seed_42": values[1],
                        "seed_73": values[2], "seed_mean": float(np.mean(values)),
                        "seed_min": float(np.min(values)), "seed_max": float(np.max(values)),
                        "seed_sd_sample": float(np.std(values, ddof=1)), "n_seeds": 3,
                        "n_users_per_seed": min(len(item) for item in maps_by_arm[arm])})
            if len(maps_by_arm) != 3:
                continue
            averaged = {arm: _seed_mean_map(items) for arm, items in maps_by_arm.items()}
            if panel_name == "development":
                baseline = {
                    int(row["uid"]): float(row["ndcg10"])
                    for row in _read_csv(ROOT / "artifacts/policy_diagnostics/evaluation/per_user.csv")
                    if row["panel"] == "development" and row["method"] == "strong_gpt"
                }
                for arm, user_values in averaged.items():
                    guard = relative_quality_guard(user_values, baseline, tolerance=0.02,
                                                   n_bootstrap=BOOTSTRAPS, seed=61700 + len(seed_rows))
                    seed_rows.append({
                        "panel": panel_name, "method": arm, "endpoint": endpoint,
                        "seed_values": "seed_mean_policy_guard", "n_seeds": 3,
                        "n_users_per_seed": len(user_values),
                        "relative_2pct_guard_passes": guard["passes"],
                        "relative_2pct_guard_lower_bound": guard["lower_bound"],
                    })
            for left, right in (("sampled16", "sampled4"), ("sampled4", "exact"), ("sampled16", "exact")):
                ci = _contrast(averaged[left], averaged[right], 61000 + len(contrast_rows))
                contrast_rows.append({"panel": panel_name, "endpoint": endpoint,
                    "contrast": f"{left} - {right}", **ci,
                    "selection_role": "development_only" if panel_name == "development" else "descriptive_only"})

    _write_csv(BRANCH_ROOT / "branch_policy_quality.csv", quality_rows)
    _write_csv(BRANCH_ROOT / "branch_support_metrics.csv", support_rows)
    _write_csv(BRANCH_ROOT / "branch_policy_audit.csv", audit_rows)
    _write_csv(BRANCH_ROOT / "branch_seed_summary.csv", seed_rows)
    _write_csv(BRANCH_ROOT / "branch_contrasts.csv", contrast_rows)
    result = {
        "schema_version": 1, "status": "evaluated", "branch_id": registration["branch_id"],
        "panels": {name: len(panel_arrays[name][2]) for name in panel_arrays},
        "quality_rows": len(quality_rows), "support_rows": len(support_rows),
        "audit_rows": len(audit_rows), "seed_summary_rows": len(seed_rows),
        "contrast_rows": len(contrast_rows),
        "best_checkpoint_steps": {f"{arm}_seed{seed}": step for (arm, seed), step in sorted(trained_best_steps.items())},
        "paired_user_bootstrap_draws": BOOTSTRAPS,
        "user_level_seed_averaging": True,
        "full_catalog_tail": "unchanged reference tail; branch audit reports conditional-support expectations and mass-weighted deltas only",
        "audit_role": "descriptive only; never used for branch selection",
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "registered_contrasts": registration["contrast_set"],
    }
    out = BRANCH_ROOT / "branch_evaluation_summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    record_operation(
        "policy_branch_sample_count_full_catalog_evaluation_and_paired_bootstrap",
        elapsed,
        bucket="final_evaluation_and_report",
        category="full_catalog_evaluation_audit_statistics",
        evidence=[
            "artifacts/policy_diagnostics/branches/sample_count_4_vs_16/branch_policy_quality.csv",
            "artifacts/policy_diagnostics/branches/sample_count_4_vs_16/branch_contrasts.csv",
            "artifacts/policy_diagnostics/branches/sample_count_4_vs_16/branch_policy_audit.csv",
        ],
        note="Includes cache loading, external-audit support prediction, full-catalog ranking, exact J/KL summaries and user-bootstrap contrasts.",
    )
    update_stage("branches", "evaluated", [
        "artifacts/policy_diagnostics/branches/sample_count_4_vs_16/branch_policy_quality.csv",
        "artifacts/policy_diagnostics/branches/sample_count_4_vs_16/branch_support_metrics.csv",
        "artifacts/policy_diagnostics/branches/sample_count_4_vs_16/branch_contrasts.csv",
        "artifacts/policy_diagnostics/branches/sample_count_4_vs_16/branch_policy_audit.csv",
    ])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
