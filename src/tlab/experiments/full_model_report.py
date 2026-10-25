from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from tlab.config import load_config
from tlab.data.manifest import sha256_file, write_json_atomic


def _paths(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    root = config.path.parent.parent
    artifacts = Path(config.project.get("artifacts_dir", "artifacts/full_model"))
    if not artifacts.is_absolute():
        artifacts = root / artifacts
    return root, artifacts


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _sci(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.2e}" if math.isfinite(number) else "—"


def _ci(value: Any, digits: int = 4) -> str:
    if not isinstance(value, dict):
        return "—"
    mean = value.get("mean_difference", value.get("mean"))
    low, high = value.get("ci_95_low"), value.get("ci_95_high")
    if mean is None or low is None or high is None:
        return "—"
    return f"{_fmt(mean, digits)} [{_fmt(low, digits)}; {_fmt(high, digits)}]"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    safe = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(safe(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _metric(model: dict[str, Any], name: str) -> Any:
    return model.get(name)


def _resource_ledger(artifacts: Path) -> dict[str, Any]:
    manifests = []
    seen_manifest_hashes: set[str] = set()
    by_stage: dict[str, dict[str, Any]] = defaultdict(lambda: {"runs": 0, "completed": 0, "failed": 0,
                                                                "reported_training_seconds": 0.0})
    for path in sorted(artifacts.rglob("run_manifest.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        manifest_hash = sha256_file(path)
        # The corrected T−14 branch reuses byte-identical local RM artifacts.
        # Count a copied manifest only once, while retaining distinct SFT/RL runs.
        if manifest_hash in seen_manifest_hashes:
            continue
        seen_manifest_hashes.add(manifest_hash)
        stage = str(item.get("stage", item.get("phase", "unknown")))
        seconds = item.get("training_seconds")
        if seconds is None:
            seconds = item.get("elapsed_seconds")
        if seconds is None:
            seconds = item.get("elapsed_seconds_this_invocation")
        seconds = float(seconds or 0.0)
        group = by_stage[stage]
        group["runs"] += 1
        group["completed"] += int(item.get("status") == "completed")
        group["failed"] += int(item.get("status") == "failed")
        group["reported_training_seconds"] += seconds
        manifests.append({"run_id": item.get("run_id", path.parent.name), "stage": stage,
                          "status": item.get("status"), "reported_seconds": seconds,
                          "run_manifest": str(path.resolve()), "sha256": manifest_hash,
                          "checkpoint_sha256": item.get("last_checkpoint_sha256") or item.get("best_checkpoint_sha256")})
    caches = []
    for path in sorted(artifacts.rglob("cache_manifest.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        matrices = item.get("matrices", {})
        caches.append({"path": str(path.resolve()), "sha256": sha256_file(path),
                       "reward_head": item.get("reward_head"),
                       "cache_bytes": sum(int(row.get("bytes", 0)) for row in matrices.values()),
                       "build_seconds": sum(float(row.get("elapsed_seconds", 0.0)) for row in matrices.values()),
                       "peak_allocated_vram_bytes": item.get("peak_allocated_vram_bytes")})
    resource_plan = _read(artifacts / "resource_plan.json") or {}
    e5_measurements = []
    for branch, branch_root in (("initial 3×-LR attempt", artifacts / "e5" / "tminus14"),
                                ("corrected disclosed-panel reanalysis", artifacts / "lrfix" / "e5" / "tminus14")):
        test_cache_path = branch_root / "test_reward_cache" / "manifest.json"
        test_timing_path = branch_root / "final_test_timing.json"
        validation_timing_path = branch_root / "local_validation_timing.json"
        test_cache = _read(test_cache_path) or {}
        test_timing = _read(test_timing_path) or {}
        validation_timing = _read(validation_timing_path) or {}
        if test_cache:
            e5_measurements.append({"stage": f"T−14 final-test reward scoring ({branch})",
                                    "seconds": test_cache.get("elapsed_seconds"),
                                    "bytes": test_cache.get("bytes"),
                                    "shape": test_cache.get("shape"),
                                    "panel_sha256": test_cache.get("panel_sha256"),
                                    "cache_manifest": str(test_cache_path.resolve()),
                                    "cache_manifest_sha256": sha256_file(test_cache_path)})
        if test_timing:
            e5_measurements.append({"stage": f"T−14 final-test complete evaluation wall time ({branch})",
                                    "seconds": test_timing.get("elapsed_seconds"),
                                    "measurement_scope": test_timing.get("measurement_scope"),
                                    "panel_sha256": test_timing.get("panel_sha256"),
                                    "timing_manifest": str(test_timing_path.resolve()),
                                    "timing_manifest_sha256": sha256_file(test_timing_path)})
        if validation_timing:
            e5_measurements.append({"stage": f"T−14 local validation wall time ({branch})",
                                    "seconds": validation_timing.get("elapsed_seconds"),
                                    "measurement_scope": validation_timing.get("measurement_scope"),
                                    "panel_ids_sha256": validation_timing.get("panel_ids_sha256"),
                                    "timing_manifest": str(validation_timing_path.resolve()),
                                    "timing_manifest_sha256": sha256_file(validation_timing_path)})
    ledger_status = "in_progress" if any(row.get("status") == "running" for row in manifests) else "completed"
    return {"status": ledger_status, "reported_seconds_scope": "sum of final run-manifest training/elapsed fields; cache/evaluation/diagnostics are listed separately when available",
            "run_count": len(manifests), "runs_by_stage": dict(by_stage), "runs": manifests,
            "cache_count": len(caches), "caches": caches,
            "e5_additional_measurements": e5_measurements,
            "e5_reserve_decision": resource_plan.get("post_tminus14_optional_fold_decision")}


def _temporal_summary(root: Path, artifacts: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "folds": {}}
    for fold in ("tminus14", "tminus28"):
        base = artifacts / "e5" / fold
        files = {name: base / filename for name, filename in {
            "data_audit": "temporal_data_audit.json", "local_sft": "local_sft.json",
            "local_rm": "local_rm.json", "rm_gate": "rm_gates.json",
            "local_validation": "local_validation_once.json", "final_test": "final_test_once.json",
            "lr_trace_audit": "sft_lr_trace_audit.json",
        }.items()}
        entry = {name: _read(path) for name, path in files.items()}
        pass5_path = artifacts / "e5" / "main_pass5_control.json"
        entry["main_pass5_control"] = _read(pass5_path)
        if fold == "tminus14":
            lr_audit = entry.get("lr_trace_audit") or {}
            initial = dict(entry)
            initial["status"] = ("superseded_recipe_mismatch_descriptive"
                                 if initial.get("final_test") and lr_audit.get("status") == "recipe_mismatch"
                                 else "initial_attempt")
            initial["independent_confirmation"] = False
            entry["initial_attempt"] = initial
            corrected_base = artifacts / "lrfix" / "e5" / fold
            corrected_files = {name: corrected_base / filename for name, filename in {
                "data_audit": "temporal_data_audit.json", "local_sft": "local_sft.json",
                "local_rm": "local_rm.json", "rm_gate": "rm_gates.json",
                "local_validation": "local_validation_once.json", "final_test": "final_test_once.json",
                "lr_trace_audit": "sft_lr_trace_audit.json", "rl_transfer": "rl_transfer.json",
                "validation_timing": "local_validation_timing.json",
                "final_test_timing": "final_test_timing.json",
            }.items()}
            corrected = {name: _read(path) for name, path in corrected_files.items()}
            corrected_sft_run = _read(artifacts / "lrfix" / "runs" / fold / "sft_strong_seed17" / "run_manifest.json") or {}
            corrected["strong_sft_run_status"] = corrected_sft_run.get("status")
            corrected["strong_sft_latest_target_cursor"] = corrected_sft_run.get("latest_target_cursor")
            corrected["strong_sft_target_budget_per_pass"] = corrected_sft_run.get("target_budget_per_epoch")
            corrected["strong_sft_latest_epoch"] = corrected_sft_run.get("latest_epoch")
            corrected["test_panel"] = _read(corrected_base / "final_test_panel.json")
            corrected["independent_confirmation"] = False
            corrected["reanalysis_of_disclosed_panel"] = True
            corrected["same_panel_as_initial_attempt"] = bool(
                corrected.get("test_panel") and initial.get("final_test") and
                corrected["test_panel"].get("panel_sha256") == initial["final_test"].get("test_panel_sha256"))
            corrected["status"] = ("completed_disclosed_panel_reanalysis"
                                    if corrected.get("final_test") and
                                    (corrected.get("lr_trace_audit") or {}).get("status") == "recipe_valid" and
                                    corrected["same_panel_as_initial_attempt"]
                                    else "in_progress_or_incomplete")
            entry["corrected_reanalysis"] = corrected
            if corrected["status"] == "completed_disclosed_panel_reanalysis":
                for key in ("data_audit", "local_sft", "local_rm", "rm_gate", "local_validation",
                            "final_test", "lr_trace_audit", "rl_transfer"):
                    if corrected.get(key) is not None:
                        entry[key] = corrected[key]
                entry["status"] = corrected["status"]
            else:
                entry["status"] = ("incomplete_lr_recipe_mismatch" if entry.get("final_test")
                                    else "in_progress_or_incomplete")
        else:
            entry["status"] = "completed" if entry.get("final_test") else "optional_not_run"
            if entry["status"] == "optional_not_run":
                plan = _read(artifacts / "resource_plan.json") or {}
                decision = plan.get("post_tminus14_optional_fold_decision") or {}
                if decision:
                    entry["optional_not_run_reason"] = (
                        f"The T−28 forecast with the required 20% margin is "
                        f"{float(decision.get('tminus28_forecast_with_20pct_margin_hours', 0.0)):.2f} host-hours, "
                        f"above the {float(decision.get('remaining_reserve_hours', 0.0)):.2f} hours remaining "
                        f"by {float(decision.get('shortfall_hours', 0.0)):.2f} hours. "
                        "The time gate failed before the T−28 data gate; no T−28 training/evaluation was launched.")
                else:
                    entry["optional_not_run_reason"] = "Post-T−14 time/data gate has not been recorded."
        result["folds"][fold] = entry
    path = artifacts / "temporal_backtests.json"
    write_json_atomic(path, result)
    return result


def _update_e5_registry(artifacts: Path, temporal: dict[str, Any]) -> None:
    path = artifacts / "run_registry.json"
    registry = _read(path)
    if not registry:
        return
    rows = registry.get("runs", [])
    by_id = {row.get("run_id"): row for row in rows}
    fold = temporal.get("folds", {}).get("tminus14", {})
    corrected = fold.get("corrected_reanalysis", {})
    if corrected.get("status") == "completed_disclosed_panel_reanalysis":
        local_sft = corrected.get("local_sft") or {}
        local_rm = corrected.get("local_rm") or {}
        for name, payload in (("weak", local_sft.get("weak_sft", {})),
                              ("strong", local_sft.get("strong_sft", {}))):
            row = by_id.get(f"tminus14_sft_{name}_seed17")
            if row is not None:
                row.update({"status": "completed_corrected_branch" if name == "strong" else "reused_same_fold_cutoff_local_artifact",
                            "run_dir": payload.get("run_dir"), "best_checkpoint": payload.get("best"),
                            "best_checkpoint_sha256": payload.get("best_sha256"),
                            "source_tree_sha256": payload.get("source_tree_sha256"),
                            "dataset_sha256": payload.get("dataset_sha256"),
                            "processed_listens_sha256": payload.get("processed_listens_sha256"),
                            "lr_audit_status": (corrected.get("lr_trace_audit") or {}).get("status") if name == "strong" else None})
        for variant in ("fit", "audit"):
            payload = local_rm.get(variant, {})
            row = by_id.get(f"tminus14_rm_{variant}_seed17")
            if row is not None:
                row.update({"status": "reused_same_fold_cutoff_local_artifact",
                            "run_dir": payload.get("run_dir"), "best_checkpoint": payload.get("checkpoint"),
                            "best_checkpoint_sha256": payload.get("checkpoint_sha256")})
        recipe_prefix = {"sampled_ordinary": "q1_sampled_ordinary_b1",
                         "exact_ordinary": "q1_exact_ordinary_b1",
                         "sampled_penalized": "q1_sampled_penalized_b1"}
        for run_key, payload in (corrected.get("rl_transfer") or {}).get("runs", {}).items():
            recipe = next((name for name in recipe_prefix if run_key.startswith(name + "_seed")), None)
            if recipe is None:
                continue
            seed = run_key.rsplit("_seed", 1)[-1]
            row = by_id.get(f"tminus14_{recipe_prefix[recipe]}_s{seed}")
            if row is not None:
                row.update({"status": "completed_corrected_branch_disclosed_panel_reanalysis",
                            "run_dir": payload.get("run_dir"),
                            "selected_transfer_checkpoint": payload.get("checkpoint"),
                            "selected_transfer_checkpoint_sha256": payload.get("checkpoint_sha256"),
                            "source_tree_sha256": payload.get("source_tree_sha256"),
                            "actor_checkpoint_sha256": payload.get("actor_checkpoint_sha256"),
                            "reward_checkpoint_sha256": payload.get("reward_checkpoint_sha256"),
                            "reward_cache_manifest_sha256": payload.get("reward_cache_manifest_sha256"),
                            "processed_listens_sha256": payload.get("processed_listens_sha256"),
                            "last_checkpoint_sha256": payload.get("last_checkpoint_sha256"),
                            "training_seconds": payload.get("training_seconds"),
                            "transfer_update_from_main_development": payload.get("transfer_update_from_main_development"),
                            "exploratory_due_to_failed_rm_gate": payload.get("exploratory_due_to_failed_rm_gate")})
        for row in rows:
            run_id = str(row.get("run_id", ""))
            if row.get("stage") == "E5_frozen_transfer" and run_id.startswith("q1_"):
                row["status"] = "completed_superseded_3x_lr_sft_descriptive"
                row["superseded_by"] = "corrected T−14 branch; same disclosed panel reanalysis"
                row["independent_confirmation"] = False
    plan = _read(artifacts / "resource_plan.json") or {}
    decision = plan.get("post_tminus14_optional_fold_decision") or {}
    if decision.get("decision") == "optional_not_run":
        for row in rows:
            if str(row.get("run_id", "")).startswith("tminus28_"):
                row["status"] = "optional_not_run_budget_gate_failed"
                row["reason"] = "Required full second-fold forecast with 20% margin exceeded the remaining E5 reserve."
        for row in rows:
            if row.get("stage") == "E5_pass5_control" and row.get("status") == "planned_if_lineage_verified":
                row["status"] = "not_run_lineage_unavailable"
                row["reason"] = "Pass5 checkpoint lineage could not be verified from the reference manifest."
    registry["protocol_freeze_status"] = registry.get("status")
    registry["status"] = "frozen_protocol_with_execution_updates"
    registry["updated_with_e5_execution"] = {
        "tminus14_status": fold.get("status"),
        "corrected_branch": "artifacts/full_model/lrfix/e5/tminus14",
        "final_panel_reused": bool(corrected.get("same_panel_as_initial_attempt")),
        "independent_confirmation": False,
        "conditional_tminus28_decision": decision.get("decision"),
    }
    write_json_atomic(path, registry)


def _finalize_corrected_e5_budget(artifacts: Path) -> None:
    plan_path = artifacts / "resource_plan.json"
    plan = _read(plan_path)
    if not plan:
        return
    fold = artifacts / "lrfix" / "e5" / "tminus14"
    final = _read(fold / "final_test_once.json") or {}
    lr_audit = _read(fold / "sft_lr_trace_audit.json") or {}
    if not final or lr_audit.get("status") != "recipe_valid":
        return
    sft_manifest = _read(artifacts / "lrfix" / "runs" / "tminus14" / "sft_strong_seed17" / "run_manifest.json") or {}
    sft_seconds = (sft_manifest.get("elapsed_seconds_this_invocation") or
                   sft_manifest.get("elapsed_seconds") or sft_manifest.get("training_seconds"))
    transfer = _read(fold / "rl_transfer.json") or {}
    rl_seconds = []
    for record in (transfer.get("runs") or {}).values():
        manifest = _read(Path(record.get("run_dir", "")) / "run_manifest.json") or {}
        value = (manifest.get("elapsed_seconds_this_invocation") or manifest.get("elapsed_seconds") or
                 record.get("training_seconds"))
        if value is not None:
            rl_seconds.append(float(value))
    cache_seconds = 0.0
    cache_count = 0
    for cache_path in (artifacts / "lrfix").rglob("cache_manifest.json"):
        cache = _read(cache_path) or {}
        matrices = cache.get("matrices", {})
        if matrices:
            cache_seconds += sum(float(row.get("elapsed_seconds", 0.0)) for row in matrices.values())
            cache_count += 1
    validation_timing = _read(fold / "local_validation_timing.json") or {}
    final_timing = _read(fold / "final_test_timing.json") or {}
    components = {"corrected_strong_sft": float(sft_seconds) if sft_seconds is not None else None,
                  "nine_frozen_rl_runs": sum(rl_seconds) if len(rl_seconds) == 9 else None,
                  "corrected_fit_rm_cache_build": cache_seconds if cache_count else None,
                  "corrected_local_validation": validation_timing.get("elapsed_seconds"),
                  "corrected_final_test_reanalysis": final_timing.get("elapsed_seconds")}
    if any(value is None for value in components.values()):
        return
    extra_seconds = sum(float(value) for value in components.values())
    previous = plan.get("post_tminus14_optional_fold_decision") or {}
    original_decision = previous.get("initial_decision_before_corrected_reanalysis")
    if original_decision is None:
        original_decision = {key: value for key, value in previous.items()
                             if key not in {"corrected_branch_measured_extra_seconds",
                                            "corrected_branch_stage_wall_seconds",
                                            "initial_decision_before_corrected_reanalysis"}}
    reserve = float(previous.get("reserve_gpu_host_wall_seconds", 36000.0))
    original_t14 = float(original_decision.get("tminus14_measured_total_wall_seconds", 0.0))
    remaining = reserve - original_t14 - extra_seconds
    if original_decision.get("tminus28_forecast_with_20pct_margin_seconds") is not None:
        forecast = float(original_decision["tminus28_forecast_with_20pct_margin_seconds"])
    else:
        forecast = float(original_decision.get("tminus28_forecast_full_cost_seconds", 0.0)) * 1.2
    shortfall = max(0.0, forecast - remaining)
    updated = dict(original_decision)
    updated["initial_decision_before_corrected_reanalysis"] = original_decision
    updated["corrected_branch_stage_wall_seconds"] = components
    updated["corrected_branch_measured_extra_seconds"] = extra_seconds
    updated["tminus14_measured_total_wall_seconds"] = original_t14 + extra_seconds
    updated["tminus14_measured_total_wall_hours"] = (original_t14 + extra_seconds) / 3600
    updated["remaining_reserve_seconds"] = remaining
    updated["remaining_reserve_hours"] = remaining / 3600
    updated["tminus28_forecast_with_20pct_margin_seconds"] = forecast
    updated["tminus28_forecast_with_20pct_margin_hours"] = forecast / 3600
    updated["shortfall_seconds"] = shortfall
    updated["shortfall_hours"] = shortfall / 3600
    updated["decision"] = "optional_not_run"
    updated["decision_rule_passed"] = False
    updated["basis"] = "corrected T−14 stage wall times measured; T−28 decision depends on remaining budget only"
    updated["tminus28_data_gate"] = "not_run because time-budget gate fails before technical preparation"
    plan["post_tminus14_optional_fold_decision"] = updated
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _links() -> str:
    return """- [full_model protocol](artifacts/full_model/protocol.json), [run registry](artifacts/full_model/run_registry.json), [resource plan](artifacts/full_model/resource_plan.json), [measured resource ledger](artifacts/full_model/resource_ledger.json), [fixed reference actor hash check](artifacts/full_model/verifications/reference_actor_preservation.json)
- [E1 support decomposition](artifacts/full_model/e1/support_decomposition.json), [RM errors by support](artifacts/full_model/e1/rm_support_errors.json), [lambda pilot/freeze](artifacts/full_model/support_lambda_pilot.json)
- [E2/E3 frontier](artifacts/full_model/e3/frontier.json), [cached/stream bridge](artifacts/full_model/e3/cached_stream_bridge.json), [confirmation results](artifacts/full_model/confirmation_results.json)
- [E4 H5 context intervention](artifacts/full_model/e4/h5_context_intervention.json)
- [E5 temporal backtests](artifacts/full_model/temporal_backtests.json), [T−14 temporal data audit](artifacts/full_model/e5/tminus14/temporal_data_audit.json), [initial SFT LR audit](artifacts/full_model/e5/tminus14/sft_lr_trace_audit.json), [corrected LR audit](artifacts/full_model/lrfix/e5/tminus14/sft_lr_trace_audit.json), [restart LR verification](artifacts/full_model/verifications/sft_restart_lr_audit.json), [corrected final evaluation](artifacts/full_model/lrfix/e5/tminus14/final_test_once.json), [corrected final-test timing](artifacts/full_model/lrfix/e5/tminus14/final_test_timing.json), [amendment 009](artifacts/full_model/protocol_amendments/009_corrected_lr_retraining_isolated.md), [amendment 010](artifacts/full_model/protocol_amendments/010_disclosed_panel_reanalysis_and_resource_update.md)
- Figures: [support](artifacts/full_model/figures/e1_support_decomposition.png), [learning curves](artifacts/full_model/figures/e2_learning_curves.png), [frontier](artifacts/full_model/figures/e3_reward_relevance_frontier.png), [H5 context](artifacts/full_model/figures/e4_context_intervention.png), [temporal forest](artifacts/full_model/figures/e5_temporal_forest.png)"""


def write_report(config_path: str | Path = "configs/full_model_rl.yaml", *, draft: bool = False) -> dict[str, Any]:
    root, artifacts = _paths(config_path)
    _finalize_corrected_e5_budget(artifacts)
    protocol = _read(artifacts / "protocol.json") or {}
    support_payload = _read(artifacts / "e1" / "support_decomposition.json") or {}
    rm_support = _read(artifacts / "e1" / "rm_support_errors.json") or {}
    frontier = _read(artifacts / "e3" / "frontier.json") or {}
    confirmation = _read(artifacts / "confirmation_results.json") or {}
    horizon = _read(artifacts / "e4" / "h5_context_intervention.json") or {}
    bridge = _read(artifacts / "e3" / "cached_stream_bridge.json") or {}
    lambda_pilot = _read(artifacts / "support_lambda_pilot.json") or {}
    lambda_freeze = _read(artifacts / "lambda_freeze.json") or {}
    pass5 = _read(artifacts / "e5" / "main_pass5_control.json") or {}
    math_audit = _read(artifacts / "verifications" / "math_audit_r3p3.json") or {}
    resume_audit = _read(artifacts / "verifications" / "resume_equivalence.json") or {}
    restart_lr_audit = _read(artifacts / "verifications" / "sft_restart_lr_audit.json") or {}
    resource_plan = _read(artifacts / "resource_plan.json") or {}
    ledger = _resource_ledger(artifacts)
    write_json_atomic(artifacts / "resource_ledger.json", ledger)
    temporal = _temporal_summary(root, artifacts)
    _update_e5_registry(artifacts, temporal)
    plots = _read(artifacts / "figure_manifest.json") or {}

    candidate_freeze = _read(artifacts / "candidate_freeze.json") or {}
    selected = candidate_freeze.get("selected_candidate", {})
    t14_temporal = temporal.get("folds", {}).get("tminus14", {})
    e5_done = t14_temporal.get("status") in {"completed", "completed_disclosed_panel_reanalysis"}
    lr_audit = t14_temporal.get("lr_trace_audit") or {}
    initial_lr_audit = (t14_temporal.get("initial_attempt") or {}).get("lr_trace_audit") or {}
    status = (("FULL full_model.1: обязательный объём завершён; T−14 финал — описательная reanalysis раскрытой панели"
               if t14_temporal.get("status") == "completed_disclosed_panel_reanalysis"
               else "FULL full_model.1: обязательный объём завершён") if e5_done else
              "НЕ ЗАВЕРШЁН: E5 T−14 LR-траектория расходится с preregistered recipe"
              if t14_temporal.get("status") == "incomplete_lr_recipe_mismatch" else
              "ЧЕРНОВИК: E5 T−14 не завершён")
    if draft:
        status = "ЧЕРНОВИК ДО E5: E1–E4 должны быть завершены, временной перенос ещё не оценён"
    model_rows = support_payload.get("models", {})
    model_table = []
    for name, values in model_rows.items():
        baseline_ci = values.get("paired_to_sft", {})
        model_table.append([
            name, _fmt(values.get("expected_raw_reward"), 5), _fmt(values.get("ndcg@10"), 6),
            _fmt(values.get("recall@10"), 6), _fmt(values.get("target_catalog_coverage"), 3),
            _fmt(values.get("unsupported_mass"), 4),
            _fmt(values.get("support_decomposition", {}).get("0", {}).get("reward_contribution_share"), 3),
            _fmt(values.get("head_share_top10"), 3), _fmt(values.get("history_repeat_top10"), 3),
            _fmt(values.get("pairwise_user_jaccard_top10_sampled_pairs"), 4),
            _fmt(values.get("duration_seconds_top10_mean"), 1),
            _ci(baseline_ci.get("ndcg@10")), _ci(baseline_ci.get("expected_raw_reward"), 5),
        ])

    conf_rows = []
    for name, result in confirmation.get("primary_holm_family", {}).items():
        conf_rows.append([name, _ci(result), _fmt(result.get("two_sided_bootstrap_p"), 4),
                          _fmt(result.get("holm_adjusted_p"), 4), result.get("n_users", "—")])

    q1_core = []
    for row in frontier.get("runs", []):
        if (row.get("seed") == 17 and row.get("update") == 1000
                and row.get("context_source", "cached") == "cached"):
            metrics = row.get("development", {})
            q1_core.append([row.get("method"), row.get("support_mode"), row.get("beta"),
                            _fmt(metrics.get("expected_raw_reward"), 5), _fmt(metrics.get("ndcg@10"), 6),
                            _fmt(metrics.get("kl_to_sft"), 5), _fmt(metrics.get("J"), 5),
                            _fmt(metrics.get("unsupported_mass", metrics.get("unsupported_probability_mass")), 4),
                            _fmt(row.get("training_seconds_at_update"), 1)])
    q2_rows = []
    for method in ("sampled", "exact"):
        run_dir = artifacts / "runs" / f"q2_{method}_ordinary_b1_s17"
        validation_path = run_dir / "validation.jsonl"
        if validation_path.is_file():
            records = [json.loads(line) for line in validation_path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
            endpoint = next((record for record in reversed(records) if int(record.get("update", -1)) == 1000), None)
            metrics = (endpoint or {}).get("development", {})
            manifest = _read(run_dir / "run_manifest.json") or {}
            if metrics:
                q2_rows.append([method, "seconds", _fmt(metrics.get("expected_raw_reward"), 3),
                                _fmt(metrics.get("ndcg@10"), 6), _fmt(metrics.get("J"), 5),
                                _fmt(metrics.get("kl_to_sft"), 5), manifest.get("status", "—"),
                                _fmt(manifest.get("training_seconds"), 1)])
    reuse_dir = artifacts / "runs" / "q1_sampled_ordinary_b1_s17_reuse4"
    reuse_path = reuse_dir / "validation.jsonl"
    if reuse_path.is_file():
        records = [json.loads(line) for line in reuse_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        endpoint = next((record for record in reversed(records) if int(record.get("update", -1)) == 1000), None)
        metrics = (endpoint or {}).get("development", {})
        manifest = _read(reuse_dir / "run_manifest.json") or {}
        if metrics:
            q2_rows.append(["sampled reuse=4", "ratio", _fmt(metrics.get("expected_raw_reward"), 5),
                            _fmt(metrics.get("ndcg@10"), 6), _fmt(metrics.get("J"), 5),
                            _fmt(metrics.get("kl_to_sft"), 5), manifest.get("status", "—"),
                            _fmt(manifest.get("training_seconds"), 1)])
    b5_rows = [[row.get("beta"), _fmt(row.get("expected_raw_reward"), 5),
                _fmt(row.get("ndcg@10"), 6), _fmt(row.get("J"), 5),
                _fmt(row.get("kl_to_sft"), 5)] for row in frontier.get("analytic_b5", [])]
    stress_rows = [[row.get("method"), row.get("support_mode"), row.get("update"),
                    _fmt((row.get("development") or {}).get("expected_raw_reward"), 5),
                    _fmt((row.get("development") or {}).get("ndcg@10"), 6),
                    _fmt(row.get("training_seconds_at_update"), 1)]
                   for row in frontier.get("runs", [])
                   if row.get("seed") == 17 and row.get("beta") == 1.0
                   and row.get("update") in {2500, 5000}]
    bridge_rows = [[row.get("seed"), row.get("panel"),
                    _ci(row.get("paired_deltas_stream_minus_cached", {}).get("ndcg@10")),
                    _ci(row.get("paired_deltas_stream_minus_cached", {}).get("expected_raw_reward"), 5),
                    _fmt(row.get("stream_scoring_seconds"), 1), row.get("stream_cache_bytes", "—")]
                   for row in bridge.get("results", [])]

    lambda_rows = []
    probe_updates = lambda_freeze.get("probe_updates", [])
    for row in lambda_pilot.get("runs", []):
        run_dir = Path(row.get("run_dir", ""))
        metrics_path = run_dir / "metrics.jsonl"
        metric_rows = []
        if metrics_path.is_file():
            metric_rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
                           if line.strip()]
        by_update = {int(metric.get("update", -1)): metric for metric in metric_rows}
        excess_text = ", ".join(
            f"{update}: {_fmt(by_update[update].get('unsupported_excess_train'), 5)}"
            for update in probe_updates if update in by_update)
        lambda_rows.append([row.get("support_mode"), row.get("beta"), row.get("lambda", row.get("lambda_support")),
                            excess_text or "—", row.get("status", "—")])
    lambda_value = lambda_freeze.get("selected_lambda")

    e4_summary = horizon.get("summary", {})
    e4_rows = [[row.get("model"), row.get("source_mode"), row.get("step"), _fmt(row.get("mean"), 6), row.get("n")]
               for row in e4_summary.get("same_item_fit_gap_by_source_and_step", [])]
    e4_teacher = [[row.get("model"), row.get("step"), _fmt(row.get("mean"), 6), row.get("n")]
                  for row in e4_summary.get("teacher_forced_ndcg10_by_model_step", [])]

    t14 = temporal["folds"]["tminus14"]
    gate = t14.get("rm_gate") or {}
    local_val = t14.get("local_validation") or {}
    local_test = t14.get("final_test") or {}
    local_sft = t14.get("local_sft") or {}
    local_rm = t14.get("local_rm") or {}
    lr_audit = t14.get("lr_trace_audit") or {}
    local_gate = gate.get("fit") or {}
    gate_all = local_gate.get("all_items") or {}
    temporal_rows = []
    if t14.get("status") == "completed_disclosed_panel_reanalysis":
        initial_attempt = t14.get("initial_attempt") or {}
        temporal_panels = [
            ("T−14 initial validation (3× LR; descriptive)", initial_attempt.get("local_validation") or {}),
            ("T−14 initial final 7d (3× LR; disclosed)", initial_attempt.get("final_test") or {}),
            ("T−14 corrected validation", local_val),
            ("T−14 corrected final 7d (same disclosed panel)", local_test),
        ]
    else:
        temporal_panels = [("T−14 validation", local_val), ("T−14 final 7d", local_test)]
    for dataset_label, payload in temporal_panels:
        models = payload.get("models", {})
        sft = models.get("local SFT", {})
        temporal_rows.append([dataset_label, "local SFT", _fmt(sft.get("ndcg@10"), 6),
                             _fmt(sft.get("expected_raw_reward"), 5), _fmt(sft.get("unsupported_mass"), 4), "—"])
        for recipe, seeds in models.items():
            if recipe == "local SFT" or not isinstance(seeds, dict):
                continue
            for seed, values in seeds.items():
                paired = values.get("paired_to_local_sft", {}).get("ndcg@10", {})
                temporal_rows.append([dataset_label, f"{recipe} seed{seed}", _fmt(values.get("ndcg@10"), 6),
                                      _fmt(values.get("expected_raw_reward"), 5), _fmt(values.get("unsupported_mass"), 4),
                                      _ci(paired)])

    cost_rows = []
    for stage, values in sorted(ledger.get("runs_by_stage", {}).items()):
        cost_rows.append([stage, values.get("runs"), values.get("completed"), values.get("failed"),
                          _fmt(values.get("reported_training_seconds"), 1),
                          _fmt(float(values.get("reported_training_seconds", 0.0)) / 3600, 2)])
    cache_rows = [[row.get("path", "").split("\\")[-1].split("/")[-1], row.get("reward_head"),
                   row.get("cache_bytes"), _fmt(row.get("build_seconds"), 1)] for row in ledger.get("caches", [])]
    final_timing_branch = ("corrected disclosed-panel reanalysis"
                           if t14.get("status") == "completed_disclosed_panel_reanalysis"
                           else "initial 3×-LR attempt")
    final_timing_label = f"T−14 final-test complete evaluation wall time ({final_timing_branch})"
    final_timing_row = next((row for row in ledger.get("e5_additional_measurements", [])
                             if row.get("stage") == final_timing_label), {})

    primary = confirmation.get("primary_holm_family", {})
    exact_ci = primary.get("exact_ordinary_minus_sampled_ordinary")
    q1_ci = primary.get("sampled_Q1_minus_SFT")
    support_models = support_payload.get("models", {})
    sft_support = support_models.get("SFT strong17", {}).get("unsupported_mass")
    full_model_support_values = [(name, values.get("unsupported_mass")) for name, values in support_models.items()
                         if name.startswith("full_model")]
    support_delta_text = "нет замороженной full_model policy"
    if sft_support is not None and full_model_support_values:
        name, value = full_model_support_values[0]
        support_delta_text = f"{name}: {_fmt(value)} против SFT {_fmt(sft_support)}"
    e4_gap = e4_summary.get("same_item_fit_gap_by_source_and_step", [])
    e4_gap_text = "нет результата"
    if e4_gap:
        fixed_gaps = [row["mean"] for row in e4_gap if row.get("source_mode") == "fixed_history_slate"]
        append_gaps = [row["mean"] for row in e4_gap if row.get("source_mode") == "naive_append"]
        e4_gap_text = f"средняя параллельная ячейка fixed={_fmt(sum(fixed_gaps)/len(fixed_gaps), 6) if fixed_gaps else '—'}, append={_fmt(sum(append_gaps)/len(append_gaps), 6) if append_gaps else '—'}"
    e5_gate_text = "не измерен" if not gate else ("пройден" if gate.get("fit_q1_predictive_gate_pass") else "провален; RL exploratory")
    corrected_e5 = t14.get("corrected_reanalysis") or {}
    e5_status_text = ("recipe выполнен; финальный скоринг — описательный повтор на уже раскрытой той же панели, не независимое подтверждение"
                      if t14.get("status") == "completed_disclosed_panel_reanalysis" else
                      "завершён по recipe" if e5_done else
                      f"первоначальная LR-ошибка; исправленная strong SFT выполняется, pass cursor {corrected_e5.get('strong_sft_latest_target_cursor', '—')}/{corrected_e5.get('strong_sft_target_budget_per_pass', '—')}"
                      if corrected_e5.get("strong_sft_run_status") == "running" else
                      "финальный скоринг есть, но strong SFT использовал 3× LR; перенос невалиден как preregistered recipe"
                      if lr_audit.get("status") == "recipe_mismatch" else "не завершён")
    t28_decision = resource_plan.get("post_tminus14_optional_fold_decision") or {}
    corrected_budget_projection = resource_plan.get("corrected_lrfix_reanalysis_projection") or {}
    if t28_decision.get("corrected_branch_measured_extra_seconds") is not None:
        t28_text = (f"T−28 оставлен optional_not_run: после измеренных corrected T−14 stages осталось "
                    f"{_fmt(t28_decision.get('remaining_reserve_hours'), 2)} h; полный прогноз с запасом 20% "
                    f"{_fmt(t28_decision.get('tminus28_forecast_with_20pct_margin_hours'), 2)} h; "
                    f"не хватает {_fmt(t28_decision.get('shortfall_hours'), 2)} h. Data gate не запускался; "
                    "решение зависит только от бюджета, не от направления результата.")
    elif corrected_budget_projection:
        remaining_projection = float(corrected_budget_projection.get("remaining_e5_reserve_after_projection_seconds", 0.0)) / 3600
        forecast_projection = float(corrected_budget_projection.get("tminus28_full_forecast_with_contingency_hours", 0.0))
        t28_text = (f"T−28 оставлен optional_not_run: mid-SFT projection оставляет {remaining_projection:.2f} h "
                    f"после 20% contingency против полного прогноза {forecast_projection:.2f} h; "
                    f"shortfall {max(0.0, forecast_projection - remaining_projection):.2f} h. "
                    "Data gate не запускался; решение определяется бюджетом и не направлением результата.")
    elif t28_decision:
        t28_text = (f"T−28 оставлен optional_not_run: полный прогноз с запасом 20% "
                    f"{_fmt(t28_decision.get('tminus28_forecast_with_20pct_margin_hours'), 2)} h, "
                    f"остаток E5 reserve {_fmt(t28_decision.get('remaining_reserve_hours'), 2)} h, "
                    f"не хватает {_fmt(t28_decision.get('shortfall_hours'), 2)} h. Data gate не запускался, "
                    "поскольку временной gate не пройден; решение не зависит от знака эффекта.")
    else:
        t28_text = "T−28 не запускался; post-T−14 cost forecast ещё не записан."
    if initial_lr_audit.get("status") == "recipe_mismatch" and lr_audit.get("status") == "recipe_valid":
        lr_note = (
            f"**Исправление E5 LR:** первоначальная попытка использовала base LR "
            f"{_fmt(initial_lr_audit.get('actual_scheduler_base_lr'), 6)} вместо "
            f"{_fmt(initial_lr_audit.get('expected_peak_lr'), 6)} и сохранена как superseded descriptive result. "
            "Restart теперь сбрасывает `lr` и `initial_lr`; отдельная strong SFT и 9 transfer runs обучены заново "
            "в `artifacts/full_model/lrfix/`. Исправленная ветка прошла LR-аудит. Финальная оценка повторно использовала "
            "тот же ранее раскрытый T−14 panel (hash совпадает), поэтому это описательная reanalysis, не независимое "
            "подтверждение и не материал для подбора параметров. См. amendment 009 и обе LR audit записи.\n")
    elif initial_lr_audit.get("status") == "recipe_mismatch" and corrected_e5.get("strong_sft_run_status") == "running":
        lr_note = (
            f"**LR correction in progress:** the first attempt used base LR "
            f"{_fmt(initial_lr_audit.get('actual_scheduler_base_lr'), 6)} instead of "
            f"{_fmt(initial_lr_audit.get('expected_peak_lr'), 6)} and remains superseded descriptive evidence. "
            f"The corrected isolated strong SFT is running at target cursor "
            f"{corrected_e5.get('strong_sft_latest_target_cursor', '—')}/"
            f"{corrected_e5.get('strong_sft_target_budget_per_pass', '—')} for its current pass; no corrected score exists yet. "
            "The scheduler fix has a targeted audit. See the initial LR audit and amendments 009/010.\n")
    elif lr_audit:
        lr_note = (
            f"**LR protocol deviation:** config/manifest requests peak LR {_fmt(lr_audit.get('expected_peak_lr'), 6)}, "
            f"but pass_05 scheduler checkpoint records base LR {_fmt(lr_audit.get('actual_scheduler_base_lr'), 6)} "
            f"({lr_audit.get('actual_to_expected_peak_factor', '—')}×). The restart bug and correction are tracked in "
            "the LR audit and amendment. Current scores remain descriptive until corrected retraining completes.\n")
    else:
        lr_note = None
    corrected_projection = resource_plan.get("corrected_lrfix_reanalysis_projection") or {}
    corrected_projection_text = (
        f"Mid-SFT measured-throughput projection for the corrected branch was "
        f"{_fmt(corrected_projection.get('remaining_correction_work_hours_projected'), 2)} host-hours remaining "
        f"before 20% contingency; it is superseded by actual corrected-stage timings in the ledger below. "
        f"It projected {_fmt(corrected_projection.get('remaining_e5_reserve_after_projection_seconds', 0) / 3600, 2)} "
        f"E5 reserve hours after contingency versus a T−28 forecast of "
        f"{_fmt(corrected_projection.get('tminus28_full_forecast_with_contingency_hours'), 2)} hours. "
        "See amendment 010 for the measured basis and limitations.") if corrected_projection else None
    temporal_audit = t14_temporal.get("data_audit") or {}
    support_grad = math_audit.get("support_penalty_gradient", [])
    support_fd = math_audit.get("support_penalty_finite_difference", [])
    support_fd_error = max((abs(float(a) - float(b)) for a, b in zip(support_grad, support_fd)), default=None)
    branch_lr_audit = corrected_e5.get("lr_trace_audit") or {}
    branch_lr_rows = branch_lr_audit.get("checkpoint_evidence") or []
    if branch_lr_audit.get("status") == "recipe_valid" and branch_lr_rows:
        lr_verification = (f"actual base {_fmt(branch_lr_audit.get('actual_scheduler_base_lr'), 6)}; "
                           f"pass_02/pass_05 step/base/decay verified; pass_05 end "
                           f"{_fmt(branch_lr_rows[-1].get('scheduler_last_lr', [None])[0], 7)}")
    else:
        lr_verification = (f"targeted fix check: requested {_fmt(restart_lr_audit.get('requested_restart_lr'), 6)}, "
                           f"scheduler base {(restart_lr_audit.get('corrected_scheduler_base_lrs') or ['—'])[0]}; "
                           f"corrected checkpoint audit {corrected_e5.get('strong_sft_run_status', 'pending')}")
    verification_rows = [
        ["Objective gradients", math_audit.get("status", "missing"),
         f"exact/sample exhaustive max abs {_sci(math_audit.get('exact_vs_exhaustive_sampled_gradient_max_abs'))}; "
         f"LOO correction {_sci(math_audit.get('loo_centered_correction_max_abs'))}; "
         f"support penalty finite-difference max abs {_sci(support_fd_error)}; active excess with zero unsupported samples "
         f"{math_audit.get('active_excess_with_zero_unsupported_samples', '—')}; reward/reference detached "
         f"{math_audit.get('reward_and_reference_detached_in_training', '—')}"],
        ["Interrupted/resumed RL", resume_audit.get("status", "missing"),
         f"{resume_audit.get('updates', '—')} updates, interrupted after {resume_audit.get('interruption_after_update', '—')}; "
         f"model max abs {_sci(resume_audit.get('model_max_abs_difference'))}, optimizer "
         f"{resume_audit.get('optimizer_state_equal', '—')}, sampler/action training RNG "
         f"{resume_audit.get('sampler_cursor_and_rng_equal', '—')}/{resume_audit.get('training_rng_state_equal', '—')}; "
         "global CPU torch RNG differed, but the eval-mode actor/action-generator training path does not consume it"],
        ["SFT scheduler restart", branch_lr_audit.get("status", restart_lr_audit.get("status", "missing")), lr_verification],
        ["T−14 temporal cutoff", temporal_audit.get("status", "missing"),
         f"max prepared raw timestamp {temporal_audit.get('max_prepared_raw_timestamp', '—')} < exclusive fold end "
         f"{temporal_audit.get('fold_end_raw_exclusive', '—')}; future rows "
         f"{temporal_audit.get('future_rows_at_or_after_fold_end', '—')}; catalog regenerated using only pre-train-cutoff rows "
         f"{temporal_audit.get('catalog_recomputed_from_raw_rows_before_train_cutoff', '—')}"]
    ]

    parts = [
        "# Отчёт full_model: механизмы RL в офлайн-рекомендациях",
        f"\n**Статус:** {status}. Protocol ID: `{protocol.get('protocol_id', 'не зафиксирован')}`.\n",
        "## Результаты на одной странице",
        "\nЗадача full_model — установить, откуда берётся прирост предсказанной награды, сколько он стоит в релевантности, что меняют optimizer/support/H=5 и сохраняется ли направление при временном переносе. Strong SFT reference и архитектура 8 345 856 параметров зафиксированы; новый model-capacity sweep не проводился. Все full_model результаты относятся к известным офлайн-панелям, не к причинному эффекту пользовательского ответа.\n",
        f"- Замороженный кандидат по development rule: **{selected.get('run_id', 'не выбран')}**, update {selected.get('update', '—')}, beta {selected.get('beta', '—')}; полноценный кандидат: `{candidate_freeze.get('selected_is_full_candidate', False)}`.",
        f"- Confirmation Q1 vs SFT: {_ci(q1_ci)}; exact ordinary vs sampled ordinary: {_ci(exact_ci)}. Holm family заранее ограничено тремя контрастами.",
        f"- Support на основной панели: {support_delta_text}. Lambda freeze: `{lambda_value if lambda_value is not None else 'не заморожена'}`.",
        f"- E4: {e4_gap_text}. Сгенерированное продолжение синтетично и не приравнивается к наблюдавшемуся feedback.",
        f"- E5 T−14: **{e5_status_text}**; local Q1 RM gate: **{e5_gate_text}**. T−28: `{temporal['folds']['tminus28']['status']}`.",
        "\nИсторические Test и RM_audit раскрыты до full_model; они не являются независимым подтверждением и не участвовали в подборе. Новая validation-партиция известна и также не является untouched holdout.",
        "\n## Вопрос 1. Почему после RL растёт награда и какие изменения происходят в поведении?",
        "\nE1 разделяет массу политики и вклад каждой support-группы в ожидаемый сырой Q1. Support определяется числом fit-RM событий item; reward-contribution не доказывает, что неизвестный item полезен пользователю. В таблице использованы общие реальные development+confirmation histories. Confidence intervals парные по пользователям относительно SFT; RM_audit не участвует в выборе.\n",
        _table(["Policy", "Expected Q1", "NDCG@10", "Recall@10", "Target coverage", "Unsupported mass", "Reward share, support=0", "Head share@10", "History repeat@10", "Top10 user Jaccard", "Duration@10 sec", "Δ NDCG vs SFT, 95% CI", "Δ Q1 vs SFT, 95% CI"], model_table) if model_table else "Пока нет E1 support decomposition.",
        "\n### Прямое ограничение unsupported mass\n",
        f"Train-only lambda pilot выбрал `{lambda_value if lambda_value is not None else '—'}`. Pilot таблица и freeze доступны в [lambda pilot](artifacts/full_model/support_lambda_pilot.json) и [lambda freeze](artifacts/full_model/lambda_freeze.json). Штраф дифференцируется через полный actor softmax; он мягкий, поэтому фактическую массу и неудачу интервенции нужно трактовать отдельно от гипотезы о RM-ошибке.\n",
        _table(["Mode", "β", "λ", "Train probe excess by update", "Status"], lambda_rows) if lambda_rows else "Pilot runs ещё не записаны.",
        "\nRM ошибки по support strata проверяются отдельно на V_RM и ранее раскрытой audit части; audit не считается независимой проверкой. См. [RM support diagnostics](artifacts/full_model/e1/rm_support_errors.json). Поведенческие guardrails дополнительно включают coverage, exposure concentration, head share, repeats, duration и pairwise top-10 overlap.",
        "\n## Вопрос 2. Сохраняется ли релевантность рекомендаций?",
        "\nPrimary relevance — full-catalog macro-user NDCG@10; OOC targets — промахи. Validation metrics показываются как известные. Подбор checkpoint и β не опирался на старый Test или RM_audit.\n",
        _table(["Method", "Support", "β", "Expected Q1", "NDCG@10", "KL to SFT", "J", "Unsupported mass", "Reported train sec at update"], q1_core) if q1_core else "Core Q1 endpoint metrics ещё не записаны.",
        "\n### Замороженная confirmation panel\n",
        _table(["Paired contrast", "Mean difference [95% CI]", "Bootstrap p", "Holm p", "Users"], conf_rows) if conf_rows else "Confirmation ещё не завершена.",
        "\nОтсутствие статистической значимости не доказывает non-inferiority; полноценный кандидат требует положительный fit-reward CI, NDCG non-inferiority и отсутствие поведенческого collapse. Rollout seeds меняют только RL stochasticity: это не повторения SFT/RM initialization.",
        "\n## Вопрос 3. Можно ли приблизиться к долгосрочным целям?",
        "\nV1 continuation/return proxy уже проверялся на 50 000 более поздних событиях 4 546 пользователей; labels имели полное 7-дневное окно, candidate-ID coverage был 90,2%. Continue prevalences составляли 90,6–93,8%, return в (24h,7d] — 97,3%. Добавление candidate ID, duration и repeat признаков ухудшало Brier/log-loss для всех четырёх labels; например, Δ log-loss для continuation 10 минут +0,00448 [0,00267;0,00623], для return +0,00183 [0,00056;0,00305]. Activity-only predictor был достаточен, Q5 не включён. Подробности и таблица — в [REPORT.md, continuation/return](REPORT.md#continuation-and-return-prediction).\n",
        "Этот результат предсказывает зарегистрированную активность, а не действие-зависимый эффект рекомендации. Дальнейшая гипотеза — ожидаемая полезная активность в сессии плюс вероятность возврата с явной termination и зрелыми D1/D7 labels. Кausal effect на return требует online user-randomized experiment с relevance/experience guardrails.",
        "\n## Вопрос 4. Что меняется при H=5 и подмене истории?",
        "\nE4 использует максимум 1 024 заранее выбранных известных validation anchors, H=5 и seeds 17/42/73. `fixed_history_slate` удерживает исходную историю; `naive_append` добавляет собственный сгенерированный token и обрезает историю по лимиту. Logged teacher forcing отдельно оценивает реальные наблюдённые цели. Same-item scores показывают только изменение оценки одного item при смене контекста; generated item не является доказательством, что его прослушали.\n",
        _table(["Model", "Generated-state policy", "Step", "Δ RM-fit score: append − fixed", "Rows"], e4_rows) if e4_rows else "E4 same-item context intervention ещё не записана.",
        "\nTeacher-forced ranking на реальных целях (не на synthetic actions):\n",
        _table(["Model", "Step", "Mean NDCG@10", "Observed targets"], e4_teacher) if e4_teacher else "Teacher-forced результат ещё не записан.",
        "\nFixed H=5 не моделирует termination или длительность реальной сессии; отсутствие дальнейшего события не интерпретируется как уход пользователя.",
        "\n## Вопрос 5. Каков frontier reward–relevance и что объясняет optimizer?",
        "\nОсновные E1–E3 jobs переиспользуются: один Q1 reward cache, те же панели/reference/schedule и 1 000-update endpoint. Beta grid и B5 — описательные frontier; выбор полной policy заморожен на development, подтверждение — на известной, но отделённой validation partition.\n",
        f"- Sampled Q1−SFT: {_ci(q1_ci)}.",
        f"- Exact ordinary−sampled ordinary: {_ci(exact_ci)} (Holm-adjusted p={_fmt(exact_ci.get('holm_adjusted_p'), 4) if isinstance(exact_ci, dict) else '—'}).",
        f"- Cached/stream bridge: {len(bridge.get('results', []))} seed×panel comparisons; [artifact](artifacts/full_model/e3/cached_stream_bridge.json).",
        "\nOptimizer table at β=1:",
        _table(["Optimizer", "Support", "β", "Expected Q1", "NDCG@10", "KL", "J", "Unsupported mass", "Train seconds"],
               [row for row in q1_core if str(row[2]) in {"1.0", "1"}]) if q1_core else "—",
        "\nThe analytic B5 policy is an unrestricted per-context optimum for the frozen surrogate objective, not an actor-quality or relevance upper bound. Stress endpoints 2,500/5,000 are diagnostics of cache reuse, not a new primary endpoint. Q2 predicted seconds are predictions, not realized listening time.",
        "\n### Q2, reuse and B5 controls\n",
        _table(["Control", "Reward head", "Expected raw reward", "NDCG@10", "J", "KL", "Status", "Reported train sec"], q2_rows) if q2_rows else "Q2/reuse manifests ещё не завершены.",
        "\nAnalytic B5 on development:",
        _table(["β", "Expected Q1", "NDCG@10", "J", "KL"], b5_rows) if b5_rows else "B5 metrics ещё не записаны.",
        "\nCached/stream matched bridge:",
        _table(["Seed", "Panel", "Stream−cached Δ NDCG@10", "Stream−cached Δ Q1", "Stream scoring sec", "Cache bytes"], bridge_rows) if bridge_rows else "Cached/stream bridge ещё не завершён.",
        "\nLong-cache stress endpoints:",
        _table(["Method", "Support", "Updates", "Expected Q1", "NDCG@10", "Reported train sec"], stress_rows) if stress_rows else "Long-cache stress ещё не завершён.",
        "\n### Временной перенос E5\n",
        f"T−14 status: **{e5_status_text}**, Q1 fit gate: **{e5_gate_text}**. Переносились sampled ordinary, exact ordinary и sampled support-penalized (β=1, frozen λ/update); seed17/42/73 разделяют только RL randomness. Paired uplift считается к fold-local SFT. Исправленный финальный скоринг повторно использует уже раскрытую T−14 панель и потому не является независимым подтверждением.\n",
        lr_note,
        (f"Локальная strong SFT: {local_sft.get('strong_sft', {}).get('passes_completed', '—')} проходов × "
         f"{local_sft.get('strong_sft', {}).get('target_budget_per_pass', '—')} целей = "
         f"{local_sft.get('strong_sft', {}).get('target_presentations', '—')} новых target presentations; "
         f"best validation NDCG@10 {_fmt(local_sft.get('strong_sft', {}).get('best_validation_ndcg10'), 6)}. "
         f"Локальная fit RM gate: {local_gate.get('all_items', {}).get('n_targets', '—')} targets, "
         f"macro-user RMSE {_fmt(gate_all.get('macro_user_model_rmse'), 4)} vs item-mean "
         f"{_fmt(gate_all.get('macro_user_item_mean_rmse'), 4)}, paired ΔMSE {_ci(gate_all.get('paired_user_mse_difference_model_minus_item_mean'), 4)}. "
         f"Fit/audit RM best epochs: {local_rm.get('fit', {}).get('best_epoch', '—')}/"
         f"{local_rm.get('audit', {}).get('best_epoch', '—')}; audit remains disclosed and is not independent confirmation.\n"),
        _table(["Panel", "Policy", "NDCG@10", "Expected local Q1", "Unsupported mass", "Paired Δ NDCG vs local SFT, 95% CI"], temporal_rows) if temporal_rows else "E5 local validation/Test ещё не завершены.",
        (f"Final Test: {local_test.get('panel_targets', '—')} targets on the frozen local final 7-day panel; "
         f"complete scoring wall time {final_timing_row.get('seconds', '—')} seconds; "
         f"reward matrix {local_test.get('reward_cache', {}).get('bytes', '—')} bytes. The previously disclosed main reference Test remains unrelated and was not used.") if local_test else None,
        "\nPass5 learning-budget control: ",
        (f"status `{pass5.get('status')}`; same RM/cache/scale: `{pass5.get('same_reward_cache_scale_and_fit_rm_as_pass10', False)}`. "
         + "; ".join(f"{key}: Δ pass5 RL−pass5 SFT NDCG {_ci(value.get('pass5_rl_uplift_to_pass5_sft', {}).get('ndcg@10'))}"
                       for key, value in pass5.get("comparisons", {}).items())
         if pass5 else "control не запущен или lineage unavailable."),
        "\n" + t28_text + " Перекрывающиеся folds не независимы. Если RM gate провален, RL transfer остаётся exploratory. Исправленный T−14 Test уже раскрывался в предыдущей попытке и используется только для описательного повторного скоринга.",
        "\n### Цена вычислений и неопределённость\n",
        f"Resource reserve E5/pass5: 10 GPU-hours. full_model pilot-measured sampled throughput: {_fmt(resource_plan.get('full_model_measured_before_main_matrix', {}).get('sampled_seconds_per_update_median'), 3)} sec/update; exact support pilot: {_fmt(resource_plan.get('full_model_measured_before_main_matrix', {}).get('exact_support_pilot_seconds_per_update'), 3)} sec/update. Полная стоимость cache, run и диагностики дана в [resource ledger](artifacts/full_model/resource_ledger.json).",
        corrected_projection_text,
        _table(["Stage", "Runs", "Completed", "Failed", "Reported sec", "Reported h"], cost_rows) if cost_rows else "Run manifests ещё не доступны.",
        "\nCache build measurements:",
        _table(["Cache", "Head", "Bytes", "Build seconds"], cache_rows) if cache_rows else "Cache manifests ещё не записаны.",
        "\nAdditional E5 scoring costs:",
        _table(["Stage", "Seconds", "Bytes", "Shape"],
               [[row.get("stage"), _fmt(row.get("seconds"), 1), row.get("bytes", "—"), row.get("shape", "—")]
                for row in ledger.get("e5_additional_measurements", [])])
        if ledger.get("e5_additional_measurements") else "E5 scoring wall/cache measurements ещё не записаны.",
        "\nЭти времена берутся из run manifests; измерения кеширования, scoring, E4 и финальной оценки перечислены отдельно, когда runner сохранил их. Пилоты и component diagnostics не смешиваются с основными обновлениями.",
        "\n### Гипотезы: свидетельство, контрсвидетельство и следующий шаг\n",
        _table(["Experiment", "For", "Against", "Uncertainty", "Next test"], [
            ["E1 support", support_delta_text, "Penalty может не снизить массу; reward groups не равны истинной пользе", "RM out-of-support extrapolation и один SFT/RM seed", "Новые reward labels или онлайн feedback"],
            ["E2 exact vs sampled", _ci(exact_ci), "CI может включать ноль и runtime/cache может доминировать", "Known validation и stochastic RL seeds", "Повторить только с независимой SFT/RM lineage"],
            ["E3 frontier", _ci(q1_ci), "Рост surrogate может сопровождаться NDCG/guardrail loss", "Множественные β/update и frozen policy", "Сверить с пользовательским outcome online"],
            ["E4 H5", e4_gap_text, "Teacher-forced relevance не оценивает sampled actions counterfactually", "Generated append — synthetic context", "Logged intervention или online sequential policy"],
            ["E5 transfer", f"T−14 {e5_status_text}", f"Gate: {e5_gate_text}", "Один fold, одна SFT/RM pair, overlapping time periods", "Новый период и независимый reward model"],
        ]),
        "\n## Воспроизводимость и ограничения",
        "\nПротокол хранит config/code/data/panel/checkpoint hashes, run registry и фиксированные seeds. Resume допускается только при совпадающих hashes всех состояний. RL оптимизирует frozen RM, а не наблюдаемый пользовательский исход. Есть одна strong SFT и одна RM initialization на fold; user overlap с историей SFT не равен cold-start. Старый Test и RM_audit раскрыты. Поэтому full_model объясняет офлайн-механизмы и trade-offs, но не доказывает причинный рост удовлетворённости, продолжения сессии или возврата.",
        "\nПроверки objective gradients, resume, restart LR и cutoff causality:",
        _table(["Check", "Status", "Evidence and limitation"], verification_rows),
        "\n## Ключевые материалы",
        "\n" + _links(),
    ]
    report_path = root / "REPORT_full_model.md"
    report_path.write_text("\n\n".join(part for part in parts if part is not None), encoding="utf-8")
    return {"status": status, "report_path": str(report_path.resolve()),
            "e5_tminus14_complete": e5_done, "selected_candidate": selected.get("run_id"),
            "resource_ledger_sha256": sha256_file(artifacts / "resource_ledger.json"),
            "temporal_summary_sha256": sha256_file(artifacts / "temporal_backtests.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the five-question full_model research report")
    parser.add_argument("--config", default="configs/full_model_rl.yaml")
    parser.add_argument("--draft", action="store_true", help="Create the pre-E5 report draft")
    args = parser.parse_args()
    print(json.dumps(write_report(args.config, draft=args.draft), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
