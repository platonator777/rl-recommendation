from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Set before importing torch through the experiment modules so CUDA's
# reproducibility settings are active before its first BLAS handle is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tlab.config import load_config
from tlab.data.manifest import sha256_file, write_json_atomic
from tlab.experiments.full_model_analysis import analyze_frontier, analyze_stream_bridge, run_e1_support_analysis, run_rm_support_errors
from tlab.experiments.full_model_matrix import (
    _load_cache, _root_and_processed, build_reward_cache, evaluate_actor, evaluate_b5, evaluate_confirmation,
    freeze_lambda_pilot, freeze_selection, gradient_component_diagnostics,
    run_exact_support_pilot, run_lambda_pilot, train_run, verify_resume_equivalence,
)
from tlab.experiments.full_model_horizon import run_h5_context_intervention


def _main_actor(root: Path) -> Path:
    return root / "artifacts/reference/runs/sft_strong_seed17/best.pt"


def _train_or_resume(config_path: str | Path, actor: Path, *, method: str, support: str,
                     beta: float, seed: int, updates: int, run_path: Path,
                     evaluate_development: bool = True, run_kind: str = "core",
                     context_source: str = "cached", reuse_epochs: int = 1,
                     reward_head: str = "ratio") -> dict[str, Any]:
    manifest_path = run_path / "run_manifest.json"
    resume_path = None
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("status") == "completed" and int(current.get("latest_update", 0)) >= updates:
            return current
        if current.get("status") == "failed":
            return current
        if (run_path / "last.pt").is_file():
            resume_path = run_path / "last.pt"
    elif run_path.exists() and any(run_path.iterdir()):
        if (run_path / "last.pt").is_file():
            resume_path = run_path / "last.pt"
        else:
            raise FileExistsError(f"non-empty run directory has no resumable checkpoint: {run_path}")
    return train_run(config_path, actor, method=method, support_mode=support, beta=beta,
                     seed=seed, reward_head=reward_head, updates=updates, checkpoint_interval=100,
                     eval_interval=500, run_dir=run_path, resume=resume_path,
                     reuse_epochs=reuse_epochs, run_kind=run_kind,
                     evaluate_development=evaluate_development, context_source=context_source)


def _resource_update(artifacts: Path) -> dict[str, Any]:
    plan_path = artifacts / "resource_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else {}
    protocol_path = artifacts / "protocol.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        plan["preregistration_sha256"] = protocol.get("resource_plan", {}).get("preregistration_sha256")
    plan["updated_after_frozen_pilot_at_utc"] = datetime.now(timezone.utc).isoformat()
    pilot_path = artifacts / "support_lambda_pilot.json"
    if not pilot_path.is_file():
        raise FileNotFoundError("support lambda pilot is required before updating measured resource budget")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    sample_rates = []
    for row in pilot["runs"]:
        manifest_path = Path(row["run_dir"]) / "run_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            seconds = float(manifest.get("training_seconds") or 0.0)
            if seconds > 0:
                sample_rates.append(seconds / max(int(manifest.get("latest_update", 100)), 1))
    exact_run_path = artifacts / "runs/exact_penalty_pilot/q1_exact_penalized_b1_s90118/run_manifest.json"
    exact_rate = None
    if exact_run_path.is_file():
        exact_manifest = json.loads(exact_run_path.read_text(encoding="utf-8"))
        exact_rate = float(exact_manifest.get("training_seconds") or 0.0) / max(int(exact_manifest.get("latest_update", 100)), 1)
    deterministic_benchmarks = {}
    benchmark_specs = {
        "sampled": artifacts / "verifications/resource_benchmarks/sampled100_deterministic/run_manifest.json",
        "exact": artifacts / "verifications/resource_benchmarks/exact100_deterministic/run_manifest.json",
    }
    for method, manifest_path in benchmark_specs.items():
        if manifest_path.is_file():
            benchmark = json.loads(manifest_path.read_text(encoding="utf-8"))
            updates = max(int(benchmark.get("latest_update", 0)), 1)
            rate = float(benchmark.get("training_seconds") or 0.0) / updates
            if rate > 0:
                deterministic_benchmarks[method] = {
                    "run_id": benchmark.get("run_id"), "updates": updates,
                    "training_seconds": benchmark.get("training_seconds"),
                    "seconds_per_update": rate,
                    "source_tree_sha256": benchmark.get("source_tree_sha256"),
                    "manifest_sha256": sha256_file(manifest_path),
                }
    sampled_forecast_rate = deterministic_benchmarks.get("sampled", {}).get(
        "seconds_per_update", statistics.median(sample_rates) if sample_rates else None)
    exact_forecast_rate = deterministic_benchmarks.get("exact", {}).get("seconds_per_update", exact_rate)
    cache_manifests = []
    for path in (artifacts / "cache/main/fit_ratio/cache_manifest.json",
                 artifacts / "cache/main/fit_seconds/cache_manifest.json"):
        if path.is_file():
            cache_manifests.append(json.loads(path.read_text(encoding="utf-8")))
    measured = {
        "lambda_pilot_runs": len(sample_rates),
        "sampled_seconds_per_update_median": statistics.median(sample_rates) if sample_rates else None,
        "sampled_seconds_per_update_min": min(sample_rates) if sample_rates else None,
        "sampled_seconds_per_update_max": max(sample_rates) if sample_rates else None,
        "exact_support_pilot_seconds_per_update": exact_rate,
        "deterministic_resource_benchmarks_after_resume_amendment": deterministic_benchmarks,
        "cache_manifests": [{"reward_head": item.get("reward_head"),
                              "cache_bytes": sum(int(v.get("bytes", 0)) for v in item.get("matrices", {}).values()),
                              "elapsed_seconds": sum(float(v.get("elapsed_seconds", 0.0)) for v in item.get("matrices", {}).values()),
                              "peak_allocated_vram_bytes": item.get("peak_allocated_vram_bytes")}
                             for item in cache_manifests],
        "pilot_run_manifest_hashes": [sha256_file(Path(row["run_dir"]) / "run_manifest.json")
                                       for row in pilot["runs"]
                                       if (Path(row["run_dir"]) / "run_manifest.json").is_file()],
    }
    plan["full_model_measured_before_main_matrix"] = measured
    if sampled_forecast_rate and exact_forecast_rate:
        plan["updated_forecast_seconds"] = {
            "core_20_q1_1000_updates": float(10 * sampled_forecast_rate * 1000
                                              + 10 * exact_forecast_rate * 1000),
            "four_long_cache_extensions_additional_4000": float(2 * sampled_forecast_rate * 4000
                                                                 + 2 * exact_forecast_rate * 4000),
            "three_stream_runs_1000_excluding_stream_cache": float(3 * sampled_forecast_rate * 1000),
            "two_q2_runs_1000": float((sampled_forecast_rate + exact_forecast_rate) * 1000),
            "reuse4_1000_optimizer_steps": float(sampled_forecast_rate * 1000 * 4),
            "per_update_seconds": {"sampled": sampled_forecast_rate, "exact": exact_forecast_rate},
            "forecast_basis": "post-amendment deterministic 100-update sampled/exact training-only benchmarks; same per-update forecast for all registered recipes, excluding cache/evaluation overhead",
        }
        reference_artifacts = artifacts.parent / "reference"
        reference_plan_path = reference_artifacts / "resource_plan.json"
        if reference_plan_path.is_file():
            reference_plan = json.loads(reference_plan_path.read_text(encoding="utf-8"))
            pass10 = reference_plan.get("completed_additional_pass_10", {})
            strong_10 = pass10.get("wall_seconds")
            weak_5 = reference_plan.get("completed_corrected_weak_seed17_run", {}).get("elapsed_seconds")
            if strong_10 and weak_5:
                pass_timings = [float(reference_plan.get("sft_full_pass_measurement_and_projection_after_pass1", {})
                                       .get("mb64_cpu_staged_actual_full_pass3", {}).get("wall_seconds") or 0.0)]
                pass_timings.extend(float(reference_plan.get(f"completed_additional_pass_{index}", {}).get("wall_seconds") or 0.0)
                                    for index in (7, 8, 9, 10))
                pass_timings = [value for value in pass_timings if value > 0.0]
                if not pass_timings:
                    raise ValueError("reference resource ledger lacks a measured full strong-SFT pass for the E5 forecast")
                strong_pass_seconds = statistics.median(pass_timings)
                rm_seconds = 0.0
                for variant in ("fit", "audit"):
                    run_manifest = reference_artifacts / "runs" / f"rm_reference_{variant}_seed17" / "run_manifest.json"
                    if run_manifest.is_file():
                        rm_data = json.loads(run_manifest.read_text(encoding="utf-8"))
                        rm_seconds += float(rm_data.get("elapsed_seconds") or rm_data.get("training_seconds") or 0.0)
                pass5_control_seconds = 2 * sampled_forecast_rate * 1000 + exact_forecast_rate * 1000
                local_rl_seconds = 6 * sampled_forecast_rate * 1000 + 3 * exact_forecast_rate * 1000
                sft_seconds = float(weak_5) + strong_pass_seconds * 5.0
                plan["mandatory_e5_forecast_seconds"] = {
                    "weak5_and_strong5_with_original_10_pass_horizon": sft_seconds,
                    "weak5_measured_reference_elapsed_seconds": float(weak_5),
                    "strong5_estimated_from_measured_complete_passes_seconds": strong_pass_seconds * 5.0,
                    "strong_pass_seconds_median_basis": pass_timings,
                    "strong_pass_projection_rule": "five times the median wall time of measured complete reference 2M-target cpu-staged passes; not half of the 10-pass aggregate",
                    "local_fit_and_audit_rm": rm_seconds,
                    "nine_frozen_rl_runs": local_rl_seconds,
                    "main_pass5_control_three_rl_runs": pass5_control_seconds,
                    "known_overhead_not_included": ["cutoff data rebuild", "full-catalog caches", "evaluation", "checkpoint copy"],
                    "total_before_overhead_seconds": sft_seconds + rm_seconds + local_rl_seconds + pass5_control_seconds,
                    "reserved_e5_gpu_hours": 10,
                    "overrun_handling": "mandatory T-14 fold and available pass5 control continue to completion; report any reserve overrun",
                }
    write_json_atomic(plan_path, plan)
    return plan


def run_pilot(config_path: str | Path) -> dict[str, Any]:
    config, _, artifacts = _root_and_processed(config_path)
    actor = _main_actor(config.path.parent.parent)
    if not actor.is_file():
        raise FileNotFoundError(actor)
    lambda_pilot = run_lambda_pilot(config_path, actor)
    frozen_lambda = freeze_lambda_pilot(config_path)
    exact = run_exact_support_pilot(config_path, actor)
    resume = verify_resume_equivalence(config_path, actor)
    gradients = gradient_component_diagnostics(config_path, actor, stage="post_lambda_freeze")
    resource = _resource_update(artifacts)
    return {"lambda_pilot": lambda_pilot, "lambda_freeze": frozen_lambda,
            "exact_support_pilot": exact, "resume_equivalence": resume,
            "gradient_components": gradients,
            "resource_plan": {"status": "updated_with_measured_full_model_pilot", **resource}}


def run_core(config_path: str | Path) -> list[dict[str, Any]]:
    config, _, artifacts = _root_and_processed(config_path)
    actor = _main_actor(config.path.parent.parent)
    jobs = []
    for method in ("sampled", "exact"):
        for support in ("ordinary", "penalized"):
            for beta in (0.1, 1.0, 3.0):
                run_path = artifacts / "runs" / f"q1_{method}_{support}_b{beta:g}_s17".replace(".", "p")
                print(f"START E1/E2/E3 core {run_path.name}", flush=True)
                jobs.append(_train_or_resume(config_path, actor, method=method, support=support,
                                             beta=beta, seed=17, updates=1000,
                                             run_path=run_path, run_kind="core"))
    return jobs


def run_long_cache_stress(config_path: str | Path) -> list[dict[str, Any]]:
    config, _, artifacts = _root_and_processed(config_path)
    actor = _main_actor(config.path.parent.parent)
    jobs = []
    for method in ("sampled", "exact"):
        for support in ("ordinary", "penalized"):
            run_path = artifacts / "runs" / f"q1_{method}_{support}_b1_s17"
            print(f"EXTEND E2 long-cache stress {run_path.name} to 5000", flush=True)
            jobs.append(_train_or_resume(config_path, actor, method=method, support=support,
                                         beta=1.0, seed=17, updates=5000,
                                         run_path=run_path, run_kind="core"))
    return jobs


def run_confirmation_training(config_path: str | Path, freeze: dict[str, Any]) -> list[dict[str, Any]]:
    config, _, artifacts = _root_and_processed(config_path)
    actor = _main_actor(config.path.parent.parent)
    jobs = []
    recipes = [(method, support, 1.0) for method in ("sampled", "exact")
               for support in ("ordinary", "penalized")]
    selected = freeze.get("additional_confirmation_runs_if_selected_beta_differs_from_1", [])
    recipes.extend((str(row["method"]), str(row["support_mode"]), float(row["beta"])) for row in selected)
    recipes = list(dict.fromkeys(recipes))
    for method, support, beta in recipes:
        for seed in (42, 73):
            run_path = artifacts / "runs" / f"q1_{method}_{support}_b{beta:g}_s{seed}".replace(".", "p")
            print(f"START E3 confirmation training {run_path.name} (no development evaluation)", flush=True)
            jobs.append(_train_or_resume(config_path, actor, method=method, support=support,
                                         beta=beta, seed=seed, updates=1000, run_path=run_path,
                                         evaluate_development=False, run_kind="E3_confirmation"))
    return jobs


def run_stream_bridge(config_path: str | Path) -> list[dict[str, Any]]:
    config, _, artifacts = _root_and_processed(config_path)
    actor = _main_actor(config.path.parent.parent)
    results = []
    for seed in (17, 42, 73):
        run_path = artifacts / "runs" / f"q1_sampled_ordinary_b1_s{seed}_stream"
        print(f"START E3 cached/stream bridge seed{seed}", flush=True)
        results.append(_train_or_resume(config_path, actor, method="sampled", support="ordinary",
                                        beta=1.0, seed=seed, updates=1000, run_path=run_path,
                                        evaluate_development=False, run_kind="E3_cached_stream_bridge",
                                        context_source="stream"))
    return results


def run_q2_reuse_b5(config_path: str | Path) -> dict[str, Any]:
    config, _, artifacts = _root_and_processed(config_path)
    actor = _main_actor(config.path.parent.parent)
    seconds_cache = build_reward_cache(config_path, variant="fit", reward_head="seconds")
    jobs = []
    for method in ("sampled", "exact"):
        run_path = artifacts / "runs" / f"q2_{method}_ordinary_b1_s17"
        print(f"START E3 Q2 {method}", flush=True)
        jobs.append(_train_or_resume(config_path, actor, method=method, support="ordinary", beta=1.0,
                                     seed=17, updates=1000, run_path=run_path,
                                     run_kind="E3_Q2_control", reward_head="seconds"))
    reuse_path = artifacts / "runs/q1_sampled_ordinary_b1_s17_reuse4"
    print("START E3 reuse4 diagnostic", flush=True)
    reuse = _train_or_resume(config_path, actor, method="sampled", support="ordinary", beta=1.0,
                             seed=17, updates=1000, run_path=reuse_path,
                             run_kind="E3_reuse_control", reuse_epochs=4)
    b5 = [evaluate_b5(config_path, actor, beta=beta, reward_head="ratio")
          for beta in (0.1, 1.0, 3.0)]
    return {"seconds_cache": seconds_cache, "q2_runs": jobs, "reuse4": reuse, "b5": b5}


def run_pass5_control(config_path: str | Path) -> dict[str, Any]:
    config, _, artifacts = _root_and_processed(config_path)
    root = config.path.parent.parent
    pass5 = root / "artifacts/reference/runs/sft_strong_seed17/pass_05.pt"
    best = root / "artifacts/reference/runs/sft_strong_seed17/best.pt"
    if not pass5.is_file():
        result = {"status": "unavailable", "reason": "reference pass5 checkpoint missing"}
        write_json_atomic(artifacts / "e5" / "main_pass5_control.json", result)
        return result
    payload = __import__("torch").load(pass5, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    pass5_update = metadata.get("epoch", metadata.get("epoch_completed"))
    sft_manifest_path = root / "artifacts/reference/runs/sft_strong_seed17/run_manifest.json"
    sft_manifest = json.loads(sft_manifest_path.read_text(encoding="utf-8")) if sft_manifest_path.is_file() else {}
    pass5_entries = [entry for entry in sft_manifest.get("checkpoints", [])
                     if entry.get("kind") == "additional_pass_5"]
    pass5_hash = sha256_file(pass5)
    lineage_ok = (metadata.get("phase") == "strong_reference" and int(metadata.get("seed", -1)) == 17
                  and any(entry.get("sha256") == pass5_hash for entry in pass5_entries))
    if not lineage_ok:
        result = {"status": "unavailable", "reason": "checkpoint phase/seed lineage does not verify",
                  "metadata": metadata, "manifest_pass5_entries": pass5_entries,
                  "checkpoint_sha256": pass5_hash}
        write_json_atomic(artifacts / "e5" / "main_pass5_control.json", result)
        return result
    frozen = json.loads((artifacts / "lambda_freeze.json").read_text(encoding="utf-8"))
    lambda_value = float(frozen["selected_lambda"])
    jobs = []
    specs = (("sampled", "ordinary"), ("exact", "ordinary"), ("sampled", "penalized"))
    for method, support in specs:
        run_path = artifacts / "runs" / "main_pass5_control" / f"{method}_{support}_b1_s17"
        jobs.append(_train_or_resume(config_path, pass5, method=method, support=support,
                                     beta=1.0, seed=17, updates=1000, run_path=run_path,
                                     evaluate_development=False, run_kind="E5_pass5_control"))
    cache_manifest, _, panels, arrays = _load_cache(config_path, "fit", "ratio")
    rows = panels["development"] + panels["confirmation"]
    eval_rewards = arrays["evaluation"][:len(rows)]
    pass5_sft_eval = evaluate_actor(config_path, pass5, rows, eval_rewards, cache_manifest, arrays,
                                    1.0, "ordinary", 0.0, reference_checkpoint=pass5)
    pass10_sft = root / "artifacts/reference/runs/sft_strong_seed17/best.pt"
    pass10_sft_eval = evaluate_actor(config_path, pass10_sft, rows, eval_rewards, cache_manifest, arrays,
                                     1.0, "ordinary", 0.0, reference_checkpoint=pass10_sft)
    frozen = json.loads((artifacts / "candidate_freeze.json").read_text(encoding="utf-8"))
    comparisons = {}
    for method, support in specs:
        recipe_key = f"{method}:{support}:beta=1"
        selected_update = int(frozen["recipe_freezes"][recipe_key]["update"])
        checkpoint = artifacts / "runs" / "main_pass5_control" / f"{method}_{support}_b1_s17" / "snapshots" / f"update_{selected_update:05d}.pt"
        result_eval = evaluate_actor(config_path, checkpoint, rows, eval_rewards, cache_manifest, arrays,
                                     1.0, support, lambda_value if support == "penalized" else 0.0,
                                     reference_checkpoint=pass5)
        pass10_checkpoint = Path(frozen["recipe_freezes"][recipe_key]["checkpoint"])
        pass10_eval = evaluate_actor(config_path, pass10_checkpoint, rows, eval_rewards, cache_manifest, arrays,
                                     1.0, support, lambda_value if support == "penalized" else 0.0,
                                     reference_checkpoint=pass10_sft)
        paired = set(result_eval["per_user_metrics"]) & set(pass5_sft_eval["per_user_metrics"])
        comparisons[recipe_key] = {
            "pass5_rl_update": selected_update,
            "pass5_rl_uplift_to_pass5_sft": {
                metric: __import__("tlab.evaluation.metrics", fromlist=["paired_user_bootstrap"]).paired_user_bootstrap(
                    {uid: result_eval["per_user_metrics"][uid][metric]
                     - pass5_sft_eval["per_user_metrics"][uid][metric] for uid in paired},
                    repetitions=2000, seed=59100 + len(comparisons))
                for metric in ("ndcg@10", "expected_raw_reward")},
            "pass5_policy": {key: value for key, value in result_eval.items()
                             if key not in {"per_user_metrics", "per_user_expected_raw_reward"}},
            "pass10_policy": {key: value for key, value in pass10_eval.items()
                              if key not in {"per_user_metrics", "per_user_expected_raw_reward"}},
        }
    result = {"status": "completed", "pass5_checkpoint": str(pass5.resolve()),
              "pass5_sha256": pass5_hash, "phase": metadata.get("phase"),
              "seed": metadata.get("seed"), "checkpoint_epoch_metadata": pass5_update,
              "pass10_checkpoint": str(best.resolve()), "pass10_sha256": sha256_file(best),
              "same_reward_cache_scale_and_fit_rm_as_pass10": True,
              "support_lambda_frozen": lambda_value,
              "sft_panel_metrics": {
                  "pass5": {key: value for key, value in pass5_sft_eval.items()
                            if key not in {"per_user_metrics", "per_user_expected_raw_reward"}},
                  "pass10": {key: value for key, value in pass10_sft_eval.items()
                             if key not in {"per_user_metrics", "per_user_expected_raw_reward"}}},
              "runs": jobs, "comparisons": comparisons,
              "interpretation": "descriptive learning-budget control only; no recipe reselection"}
    write_json_atomic(artifacts / "e5" / "main_pass5_control.json", result)
    return result


def run_stage(config_path: str | Path, stage: str) -> Any:
    config, processed, artifacts = _root_and_processed(config_path)
    actor = _main_actor(config.path.parent.parent)
    if stage == "protocol":
        from tlab.experiments.full_model_protocol import freeze_protocol
        return freeze_protocol(config_path)
    if not (artifacts / "protocol.json").is_file():
        raise FileNotFoundError("freeze the preregistered full_model protocol before launching experiment stages")
    if stage == "cache":
        return {"q1_ratio": build_reward_cache(config_path, variant="fit", reward_head="ratio"),
                "e1_support_decomposition": run_e1_support_analysis(config_path),
                "rm_support_errors": run_rm_support_errors(config_path)}
    if stage == "pilot":
        return run_pilot(config_path)
    if stage == "core":
        return run_core(config_path)
    if stage == "stress":
        return run_long_cache_stress(config_path)
    if stage == "bridge":
        bridge_runs = run_stream_bridge(config_path)
        return {"runs": bridge_runs, "analysis": analyze_stream_bridge(config_path)}
    if stage == "q2":
        return run_q2_reuse_b5(config_path)
    if stage == "freeze":
        frozen = freeze_selection(config_path)
        confirmation_runs = run_confirmation_training(config_path, frozen)
        confirmation = evaluate_confirmation(config_path)
        return {"freeze": frozen, "confirmation_runs": confirmation_runs,
                "confirmation": confirmation}
    if stage == "e4":
        return run_h5_context_intervention(config_path="configs/full_model_horizon.yaml")
    if stage == "pass5":
        return run_pass5_control(config_path)
    if stage == "analysis":
        final_e1 = run_e1_support_analysis(config_path)
        frontier = analyze_frontier(config_path)
        return {"e1": final_e1, "e3": frontier}
    if stage == "resume-audit":
        return verify_resume_equivalence(config_path, actor)
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable stage runner for the registered full_model.1 experiments")
    parser.add_argument("--config", default="configs/full_model_rl.yaml")
    parser.add_argument("--stage", choices=("protocol", "cache", "pilot", "core", "stress", "bridge",
                                            "q2", "freeze", "e4", "pass5", "analysis", "resume-audit"), required=True)
    args = parser.parse_args()
    result = run_stage(args.config, args.stage)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
