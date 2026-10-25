from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tlab.residual_support.objectives import FrozenRewardScale

from .analysis import run_dev_rm_analysis
from .checks import run_minimal_checks
from .data import _load_cache, freeze_split, frozen_reward_scale, open_v5_cache, prepare_feature_caches, sha256_file
from .reporting import run_reporting
from .training import fit_scaler, train_one


ROOT = __import__('tlab.paths', fromlist=['workspace_root']).workspace_root()
ARTIFACT_ROOT = ROOT / "artifacts/frozen_policy"
FINAL_ROOT = ROOT / "final/frozen_policy"
CONFIG = ROOT / "configs/frozen_policy/main.yaml"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_record() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "run.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "protocol_id": "tlab_yambda_frozen_policy_2026-10-22",
        "status": "in_progress",
        "started_at_utc": _utc(),
        "configuration": str(CONFIG.relative_to(ROOT)),
        "source_plan": "RESEARCH_PLAN_frozen_policy.md",
        "scope": "one compact next-item experiment; submission/ untouched; no old audit continuation",
        "inputs": {
            "strong_actor": "models/actor/strong.pt",
            "policy_train_cache": "artifacts/residual_support/cache/train/",
            "development_reporting_cache": "artifacts/policy_diagnostics/cache/common_open_6655/",
            "old_rm": "artifacts/residual_support/rm/models/optimized_rm.joblib",
            "same_split_control_rm": "artifacts/policy_diagnostics/rm/C_refit_residual_support_optimized_control.joblib",
            "prefix_affinity_rm": "artifacts/policy_diagnostics/rm/C_prefix_safe_affinity_extension.joblib",
            "prefix_affinity_embeddings": "artifacts/policy_diagnostics/rm/prefix_affinity_embeddings.npy",
        },
        "stages": {},
        "seed_values": [17, 42, 73],
        "policy": {"sampled_actions_per_context": 16, "updates_per_run": 3000,
                   "eval_every": 500, "batch_contexts": 32, "head": "54→64→64→1 SiLU; zero-init final layer"},
    }


def _persist_run(run: dict[str, Any]) -> None:
    _write_json(ARTIFACT_ROOT / "run.json", run)


def _load_scale(path: Path) -> FrozenRewardScale:
    value = json.loads(path.read_text(encoding="utf-8"))
    return FrozenRewardScale(mean=float(value["mean"]), std=float(value["std"]),
                             std_floor=float(value["std_floor"]), version=str(value["version"]))


def _train_group_entry(root: Path, run_name: str, result: dict[str, Any]) -> dict[str, Any]:
    run_dir = (ARTIFACT_ROOT / "policy" / run_name).resolve()
    return {
        "run_name": run_name,
        "run_dir": str(run_dir.relative_to(root)),
        "best_checkpoint_path": result["best_checkpoint_path"],
        "best_checkpoint_sha256": result["best_checkpoint_sha256"],
        "best_step": result["best_checkpoint_step"],
        "best_dev_ndcg10": result["best_dev_ndcg10"],
        "best_dev_mass_weighted_kl": result["best_dev_mass_weighted_kl"],
        "seed": int(result["seed"]), "beta": float(result["beta"]),
        "objective": result["objective"], "reward_model": result["reward_model"],
    }


def _train_all(run: dict[str, Any], scale: FrozenRewardScale,
               split: dict[str, Any], analytic: dict[str, Any],
               resume: bool = True) -> tuple[dict[str, Any], float]:
    feature_arrays, _ = open_v5_cache(ARTIFACT_ROOT / "cache", "train")
    scaler = fit_scaler(feature_arrays["candidate_features"], ARTIFACT_ROOT / "scaler.json")
    candidates = [float(x) for x in analytic["policy_beta_candidates"]]
    chosen_rm = str(analytic["selected_analytic"]["reward_model"])
    beta_search_results: list[dict[str, Any]] = []
    elapsed = 0.0
    for beta in candidates:
        run_name = f"sampled_loo_{chosen_rm}_b{beta:g}_seed17"
        result = train_one(root=ROOT, artifact_root=ARTIFACT_ROOT, run_name=run_name,
                           objective="sampled_loo", beta=beta, seed=17,
                           reward_model=chosen_rm, scaler=scaler, scale=scale,
                           split_manifest=split, max_updates=3000, check_every=500, resume=resume)
        elapsed += float(result["elapsed_training_and_dev_scoring_seconds"])
        beta_search_results.append(result)
    selected_beta_run = sorted(beta_search_results,
                               key=lambda r: (-float(r["best_dev_ndcg10"]),
                                              float(r["best_dev_mass_weighted_kl"]),
                                              float(r["beta"])))[0]
    selected_beta = float(selected_beta_run["beta"])
    beta_selection = {
        "selected_beta": selected_beta,
        "rule": "best trained seed17 sampled-LOO checkpoint by dev NDCG@10; ties by lower mass-weighted KL",
        "search_rows": [{"beta": float(r["beta"]), "best_step": int(r["best_checkpoint_step"]),
                         "dev_ndcg10": float(r["best_dev_ndcg10"]),
                         "mass_weighted_kl": float(r["best_dev_mass_weighted_kl"]),
                         "run_name": r["run_name"]} for r in beta_search_results],
    }
    _write_json(ARTIFACT_ROOT / "policy_beta_selection.json", beta_selection)
    sampled_entries: dict[str, Any] = {}
    for result in beta_search_results:
        if math.isclose(float(result["beta"]), selected_beta, rel_tol=0, abs_tol=1e-10):
            sampled_entries["17"] = _train_group_entry(ROOT, result["run_name"], result)
    for seed in (42, 73):
        run_name = f"sampled_loo_{chosen_rm}_b{selected_beta:g}_seed{seed}"
        result = train_one(root=ROOT, artifact_root=ARTIFACT_ROOT, run_name=run_name,
                           objective="sampled_loo", beta=selected_beta, seed=seed,
                           reward_model=chosen_rm, scaler=scaler, scale=scale,
                           split_manifest=split, max_updates=3000, check_every=500, resume=resume)
        elapsed += float(result["elapsed_training_and_dev_scoring_seconds"])
        sampled_entries[str(seed)] = _train_group_entry(ROOT, run_name, result)

    exact_entries: dict[str, Any] = {}
    for seed in (17, 42, 73):
        run_name = f"exact_gradient_{chosen_rm}_b{selected_beta:g}_seed{seed}"
        result = train_one(root=ROOT, artifact_root=ARTIFACT_ROOT, run_name=run_name,
                           objective="exact_expected_reward", beta=selected_beta, seed=seed,
                           reward_model=chosen_rm, scaler=scaler, scale=scale,
                           split_manifest=split, max_updates=3000, check_every=500, resume=resume)
        elapsed += float(result["elapsed_training_and_dev_scoring_seconds"])
        exact_entries[str(seed)] = _train_group_entry(ROOT, run_name, result)

    old_run_name = f"sampled_loo_old_deployed_b{selected_beta:g}_seed17"
    old_result = train_one(root=ROOT, artifact_root=ARTIFACT_ROOT, run_name=old_run_name,
                           objective="sampled_loo", beta=selected_beta, seed=17,
                           reward_model="old_deployed", scaler=scaler, scale=scale,
                           split_manifest=split, max_updates=3000, check_every=500, resume=resume)
    elapsed += float(old_result["elapsed_training_and_dev_scoring_seconds"])
    old_entries = {"17": _train_group_entry(ROOT, old_run_name, old_result)}

    choice = {
        "status": "frozen_before_reporting_scoring",
        "selected_rm": chosen_rm,
        "analytic_selected_rm_beta": float(analytic["selected_analytic"]["beta"]),
        "policy_beta_candidates": candidates,
        "beta": selected_beta,
        "policy_beta_selection": beta_selection,
        "sampled_rl": sampled_entries,
        "exact_gradient": exact_entries,
        "old_rm_rl": old_entries,
        "reporting_selection_rule": "selected RM and beta use development only; each seed uses its best trained checkpoint by dev NDCG@10 then lower KL",
    }
    for group in ("sampled_rl", "exact_gradient", "old_rm_rl"):
        for entry in choice[group].values():
            path = ROOT / entry["best_checkpoint_path"]
            checkpoint = __import__("torch").load(path, map_location="cpu", weights_only=False)
            if not checkpoint.get("trained") or int(checkpoint.get("step", 0)) <= 0:
                raise AssertionError("pre-reporting checkpoint is not trained")
            final_weight = checkpoint["state_dict"]["network.4.weight"]
            if not bool((final_weight != 0).any()):
                raise AssertionError(f"trained output layer remained identity at {path}")
    _write_json(ARTIFACT_ROOT / "reporting_selection.json", choice)
    run["selected_reward_model"] = chosen_rm
    run["selected_beta"] = selected_beta
    run["policy_beta_search"] = beta_selection
    run["policy_training_seconds"] = round(elapsed, 3)
    _persist_run(run)
    return choice, elapsed


def execute(resume: bool = True, rebuild_reporting: bool = False) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    run = _run_record()
    _persist_run(run)

    # Freeze the user split before any new model is scored.
    split_path = ARTIFACT_ROOT / "split_manifest.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        _, _, panel = _load_cache(ROOT / "artifacts/policy_diagnostics/cache/common_open_6655")
        split = freeze_split(panel, split_path)
    run["split"] = {k: split[k] for k in ("hash", "n_contexts", "development_n", "reporting_n", "user_overlap", "event_overlap")}
    run["development_user_ids_sha256"] = __import__("hashlib").sha256(",".join(map(str, split["development_uids"])).encode()).hexdigest()
    run["reporting_user_ids_sha256"] = __import__("hashlib").sha256(",".join(map(str, split["reporting_uids"])).encode()).hexdigest()
    _persist_run(run)

    feature_manifest_path = ARTIFACT_ROOT / "cache/feature_cache_manifest.json"
    if feature_manifest_path.exists():
        feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    else:
        t0 = time.perf_counter()
        feature_manifest = prepare_feature_caches(ROOT, ARTIFACT_ROOT / "cache")
        run["stages"]["feature_preparation_seconds"] = feature_manifest["elapsed_seconds"]
        run["feature_preparation_seconds"] = feature_manifest["elapsed_seconds"]
        _persist_run(run)

    train_cache, _ = open_v5_cache(ARTIFACT_ROOT / "cache", "train")
    _, train_source_arrays, _ = _load_cache(ROOT / "artifacts/residual_support/cache/train")
    train_cache["reference_conditional_logp"] = train_source_arrays["reference_conditional_logp"]
    train_cache["support_mass"] = train_source_arrays["support_mass"]
    scale_path = ARTIFACT_ROOT / "reward_scale.json"
    if scale_path.exists():
        scale = _load_scale(scale_path)
    else:
        scale = frozen_reward_scale(train_cache)
        _write_json(scale_path, {"mean": scale.mean, "std": scale.std, "std_floor": scale.std_floor,
                                 "version": scale.version,
                                 "definition": "same-split C control predictions; reference-conditional expectation over top-500 candidates, equally averaged over fixed policy-train contexts",
                                 "train_contexts": len(train_cache["rm_rewards"]), "RM": "same_split_control"})
    run["reward_scale"] = {"mean": scale.mean, "std": scale.std, "version": scale.version}
    _persist_run(run)

    scaler = fit_scaler(train_cache["candidate_features"], ARTIFACT_ROOT / "scaler.json")
    checks_file = ARTIFACT_ROOT / "minimal_checks.json"
    if checks_file.exists():
        checks = json.loads(checks_file.read_text(encoding="utf-8"))
    else:
        check_results = run_minimal_checks(ROOT, ARTIFACT_ROOT, feature_manifest, scale)
        checks = json.loads(checks_file.read_text(encoding="utf-8"))
        run["minimal_checks"] = checks
        _persist_run(run)
    if checks.get("status") != "passed":
        raise RuntimeError("a required Frozen implementation check failed")

    dev_results_path = ARTIFACT_ROOT / "development/rm_dev_results.json"
    if dev_results_path.exists():
        analytic = json.loads(dev_results_path.read_text(encoding="utf-8"))
    else:
        t0 = time.perf_counter()
        analytic = run_dev_rm_analysis(ROOT, ARTIFACT_ROOT, scale, split)
        run["stages"]["direct_rm_dev_seconds"] = round(time.perf_counter() - t0, 3)
        _persist_run(run)
    run["direct_rm_dev_summary"] = analytic["selected_analytic"]
    _persist_run(run)

    selection_path = ARTIFACT_ROOT / "reporting_selection.json"
    if selection_path.exists():
        choice = json.loads(selection_path.read_text(encoding="utf-8"))
    else:
        t0 = time.perf_counter()
        choice, elapsed_training = _train_all(run, scale, split, analytic, resume=resume)
        actual_training_stage = round(time.perf_counter() - t0, 3)
        run["stages"]["policy_training_and_dev_evaluation_seconds"] = actual_training_stage
        run["policy_training_run_timer_sum_seconds"] = round(elapsed_training, 3)
        run["policy_training_seconds"] = actual_training_stage
        run["status"] = "trained_and_selected_before_reporting"
        _persist_run(run)

    if "policy_training_and_dev_evaluation_seconds" in run.get("stages", {}):
        run["policy_training_seconds"] = float(run["stages"]["policy_training_and_dev_evaluation_seconds"])

    # Selection is written before this call; reporting outcomes cannot change model choice.
    reporting_path = FINAL_ROOT / "metrics.json"
    if reporting_path.exists() and not rebuild_reporting:
        reporting = json.loads(reporting_path.read_text(encoding="utf-8"))
    else:
        run.get("stages", {}).pop("final_reporting_and_report_seconds", None)
        elapsed_before_report = sum(float(v) for k, v in run.get("stages", {}).items() if k.endswith("_seconds"))
        runtime = {
            "elapsed_seconds_total": elapsed_before_report,
            "policy_training_seconds": float(run.get("policy_training_seconds", 0.0)),
            "feature_preparation_seconds": float(run.get("feature_preparation_seconds", 0.0)),
        }
        t0 = time.perf_counter()
        reporting = run_reporting(ROOT, ARTIFACT_ROOT, FINAL_ROOT, split, analytic,
                                  choice, scale, feature_manifest, checks, runtime)
        run["stages"]["final_reporting_and_report_seconds"] = round(time.perf_counter() - t0, 3)
        run["reporting_summary"] = {"contexts": reporting["n_reporting_contexts"],
                                    "primary_contrasts": reporting["primary_contrasts"]}
        run["status"] = "complete"
        run["completed_at_utc"] = _utc()
        run["elapsed_seconds_total"] = round(sum(float(v) for k, v in run["stages"].items() if k.endswith("_seconds")), 3)
        _persist_run(run)

    if run.get("status") != "complete":
        run["status"] = "complete"
        run["completed_at_utc"] = _utc()
        run["elapsed_seconds_total"] = round(sum(float(v) for k, v in run["stages"].items() if k.endswith("_seconds")), 3)
        _persist_run(run)
    return {"status": run["status"], "run_path": str((ARTIFACT_ROOT / "run.json").relative_to(ROOT)),
            "report_path": str((FINAL_ROOT / "REPORT.md").relative_to(ROOT)),
            "n_reporting_contexts": reporting["n_reporting_contexts"],
            "primary_contrasts": reporting["primary_contrasts"],
            "elapsed_seconds_total": run.get("elapsed_seconds_total")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Frozen FAST RL next-item experiment.")
    parser.add_argument("--stage", choices=["all"], default="all")
    parser.add_argument("--resume", action="store_true", help="resume signature-matched policy runs")
    parser.add_argument("--rerun-reporting", action="store_true", help="re-score frozen selected checkpoints without repeating training")
    args = parser.parse_args()
    result = execute(resume=args.resume, rebuild_reporting=args.rerun_reporting)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
