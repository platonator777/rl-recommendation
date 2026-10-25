"""Register and protect the retrospective panel_tools.2 continuation study."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = __import__('tlab.paths', fromlist=['workspace_root']).workspace_root()
ARTIFACTS = ROOT / "artifacts" / "policy_diagnostics"
EXCLUDED_VERSION_DIRS = {"policy_diagnostics"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def protected_files() -> list[Path]:
    roots = ["submission", "src/tlab", "tests", "configs", "artifacts", "final"]
    files: list[Path] = []
    for name in roots:
        base = ROOT / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDED_VERSION_DIRS for part in relative.parts):
                continue
            files.append(path)
    return sorted(files, key=lambda p: rel(p).casefold())


def protection_snapshot() -> dict[str, Any]:
    started = time.perf_counter()
    files = protected_files()
    entries = [{"path": rel(p), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "panel_tools.2 before/after integrity check for submission and historical files",
        "protected_roots": ["submission/", "src/tlab/", "tests/", "configs/", "artifacts/", "final/"],
        "exclusion": "new policy_diagnostics namespace only",
        "file_count": len(entries),
        "total_bytes": sum(x["size_bytes"] for x in entries),
        "hash_seconds": round(time.perf_counter() - started, 3),
        "files": entries,
    }


def verify_snapshot(before_path: Path, output_path: Path) -> dict[str, Any]:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    now = {rel(p): p for p in protected_files()}
    before_map = {item["path"]: item for item in before["files"]}
    changed: list[dict[str, Any]] = []
    missing: list[str] = []
    added: list[str] = []
    for name, old in before_map.items():
        path = now.get(name)
        if path is None:
            missing.append(name)
            continue
        current_hash = sha256(path)
        current_size = path.stat().st_size
        if current_hash != old["sha256"] or current_size != old["size_bytes"]:
            changed.append({"path": name, "before_sha256": old["sha256"], "after_sha256": current_hash,
                            "before_size_bytes": old["size_bytes"], "after_size_bytes": current_size})
    added = sorted(set(now) - set(before_map), key=str.casefold)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not changed and not missing and not added else "failed",
        "before_file_count": len(before_map),
        "after_file_count": len(now),
        "changed_count": len(changed),
        "missing_count": len(missing),
        "added_count": len(added),
        "changed": changed,
        "missing": missing,
        "added": added,
    }
    json_write(output_path, result)
    return result


def record_operation(name: str, seconds: float, *, bucket: str, category: str,
                    evidence: list[str] | None = None, note: str | None = None) -> None:
    budget_path = ARTIFACTS / "protocol" / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    stage_cap = int(budget["new_stage_cap_seconds"])
    op = {
        "name": name,
        "seconds": round(float(seconds), 3),
        "bucket": bucket,
        "category": category,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": evidence or [],
    }
    if note:
        op["note"] = note
    budget["measured_work"].append(op)
    budget["wall_time_seconds"] = round(sum(float(x["seconds"]) for x in budget["measured_work"]), 3)
    budget["remaining_new_stage_seconds"] = max(0, stage_cap - budget["wall_time_seconds"])
    budget["remaining_to_total_stop_from_known_lower_bound_seconds"] = max(
        0, round(float(budget["remaining_to_stop_threshold_seconds_from_lower_bound"]) - budget["wall_time_seconds"], 3)
    )
    if budget["wall_time_seconds"] > stage_cap:
        budget["decision"] = "Base 10h additional cap exceeded; do not start new compute. Complete available reporting and integrity checks."
    json_write(budget_path, budget)


def update_stage(stage: str, status: str, evidence: list[str], *, note: str | None = None) -> None:
    path = ARTIFACTS / "protocol" / "stage_status.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if stage not in result["stages"]:
        result["stages"][stage] = {"status": status, "evidence": evidence}
    else:
        result["stages"][stage]["status"] = status
        result["stages"][stage]["evidence"] = evidence
    if note:
        result["stages"][stage]["note"] = note
    result["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    json_write(path, result)


def register() -> None:
    protocol = ARTIFACTS / "protocol"
    protection = ARTIFACTS / "protection"
    if protocol.exists() and any(protocol.iterdir()):
        raise FileExistsError(f"Refusing to overwrite registered protocol: {protocol}")
    if (ARTIFACTS / "protection" / "before.json").exists():
        raise FileExistsError("Refusing to overwrite protection baseline")
    for name in ("protocol", "protection", "evaluation", "diagnostics", "rm", "branches", "figures", "logs"):
        (ARTIFACTS / name).mkdir(parents=True, exist_ok=True)
    for name in ("policy_diagnostics",):
        for base in (ROOT / "src" / "tlab", ROOT / "tests", ROOT / "configs", ROOT / "final"):
            (base / name).mkdir(parents=True, exist_ok=True)

    registration = {
        "schema_version": 1,
        "protocol_id": "tlab_yambda_policy_diagnostics_retrospective_2026-10-22",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "registered_before_new_scoring",
        "study_classification": "retrospective_exploratory_all_historical_panels_previously_opened",
        "primary_question": "Which measured limitations explain the absence of a confirmed policy-quality gain, and which at most two controlled follow-up branches are justified?",
        "protected_scope": ["submission/", "historical V1-panel_tools.1 source, checkpoints, reports, and artifacts"],
        "data_roles": {
            "train": {"path": "artifacts/residual_support/panels/a_rl_train.jsonl", "role": "fit/diagnostic train only", "known_rows": 4134},
            "development": {"path": "artifacts/residual_support/panels/validation_development.jsonl", "role": "selection and diagnostics; previously opened", "known_rows": 514},
            "retrospective_final": {"path": "artifacts/residual_support/panels/test_retrospective.jsonl", "role": "descriptive endpoint only; previously opened", "known_rows": 710},
            "common_open_panel": {"path": "artifacts/panel_tools/evaluation/frozen_primary_validation_panel.jsonl", "role": "descriptive common comparison only; previously opened", "known_rows": 6655},
            "rm_fit": {"path": "data/processed_full_model_tminus14/rm_user_partitions.parquet", "role": "historical RM fitting rows; temporal validity to be checked"},
            "rm_development": {"path": "not_yet_assigned", "role": "must be fixed from permitted pre-audit pool before any C model selection"},
            "external_audit": {"path": "artifacts/residual_support/rm/models/external_audit_predictor.joblib", "role": "frozen after model/branch freeze; never selection"}
        },
        "metrics": {
            "primary": "existing user-macro NDCG@10, original positive/nonorganic/OOV/tie definitions",
            "secondary": ["Recall@10", "positive-new and positive-repeat outcomes", "proxy and external-audit expectations", "KL", "support coverage", "RMSE/MAE/calibration"],
            "user_aggregation": "Aggregate events within user; average method scores across fixed seeds before paired user bootstrap.",
            "guard": "One-sided 95% lower bound of user-level metric_policy - 0.98 * metric_reference; report separately from two-sided paired-difference CI.",
            "checkpoint": "Existing run last.pt primary; best.pt separate; step-0 identity fallback is never a trained checkpoint.",
            "full_catalog": "Preserve catalog 20k, top support K=500, ratio objective, and unchanged reference tail."
        },
        "registered_contrasts": [
            "dense - strong", "dense - matched last_target", "each principal policy - strong",
            "sampled_loo - exact_expected_reward separately at beta 1.0 and 0.3",
            "analytic_q_distillation - exact_expected_reward separately at beta 1.0 and 0.3",
            "trained no_feedback - full separately by seed at beta 0.3"
        ],
        "branch_selection_rule": "Use only fixed train/development evidence and diagnostics B/C. The 6655 panel, 710 retrospective final, and external audit are descriptive endpoints after the model list is frozen.",
        "pre_scoring_temporal_gate": "Determine row-wise event-time validity of actor, RM, and prior features before including any policy-quality comparison; report exclusions and common eligible panel.",
        "new_policy_fits": {"maximum_branches": 2, "seeds": [17, 42, 73], "primary_beta": 0.3, "conditional_beta_1_sensitivity": True, "pilot_required_but_not_sufficient": True}
    }
    json_write(protocol / "registration.json", registration)

    # Freeze identities of all core inputs, checkpoints, panels, and source files used below.
    input_roots = [
        ROOT / "artifacts" / "full_model" / "lrfix" / "runs" / "tminus14" / "sft_strong_seed17",
        ROOT / "artifacts" / "residual_support" / "baseline", ROOT / "artifacts" / "residual_support" / "cache",
        ROOT / "artifacts" / "residual_support" / "evaluation", ROOT / "artifacts" / "residual_support" / "panels",
        ROOT / "artifacts" / "residual_support" / "policy", ROOT / "artifacts" / "residual_support" / "policy_diagnostic",
        ROOT / "artifacts" / "residual_support" / "protocol", ROOT / "artifacts" / "residual_support" / "rm",
        ROOT / "artifacts" / "residual_support" / "sft", ROOT / "artifacts" / "panel_tools" / "evaluation",
        ROOT / "data" / "processed_full_model_tminus14",
        ROOT / "src" / "tlab" / "residual_support", ROOT / "tests" / "residual_support", ROOT / "configs" / "residual_support",
        ROOT / "final" / "residual_support",
    ]
    input_files: set[Path] = set()
    for base in input_roots:
        if base.is_file():
            input_files.add(base)
        elif base.exists():
            input_files.update(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    source_paths = [ROOT / "RESEARCH_PLAN_policy_diagnostics.md", ROOT / "RESEARCH_PLAN_panel_tools.md"]
    input_files.update(p for p in source_paths if p.exists())
    files = sorted(input_files, key=lambda p: rel(p).casefold())
    input_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(files),
        "files": [{"path": rel(p), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    }
    json_write(protocol / "input_manifest.json", input_manifest)

    exposure = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_status": "retrospective_exploratory_no_independent_test_remains",
        "windows_previously_opened": [
            {"panel": "development", "path": "artifacts/residual_support/panels/validation_development.jsonl", "n": 514, "previously_scored": True},
            {"panel": "retrospective_final", "path": "artifacts/residual_support/panels/test_retrospective.jsonl", "n": 710, "previously_scored": True},
            {"panel": "common_open", "path": "artifacts/panel_tools/evaluation/frozen_primary_validation_panel.jsonl", "n": 6655, "previously_scored": True}
        ],
        "known_outcome_exposure": "panel_tools.1 final report disclosed NDCG values, selected model outcomes, SFT dev outcomes, and policy/audit results. These windows are descriptive only and cannot select branch, beta, checkpoint, or RM.",
        "future_feature_validity": "Not yet evaluated; separate row-level temporal gate required before A results are eligible.",
        "new_hash_split": "Will not be described as restoring independence."
    }
    json_write(protocol / "exposure.json", exposure)

    budget = {
        "schema_version": 1,
        "protocol_id": registration["protocol_id"],
        "mode": "previously_authorized_24h_total_with_base_10h_additional_cap",
        "historical_recorded_lower_bound_seconds": 22433.568,
        "historical_early_preparation": "unknown, not imputed as zero",
        "total_fit_stop_threshold_seconds": 61200,
        "remaining_to_stop_threshold_seconds_from_lower_bound": round(61200 - 22433.568, 3),
        "new_stage_cap_seconds": 36000,
        "new_stage_allocation_seconds": {"A": 5400, "B": 7200, "C": 5400, "conditional_branches": 10800, "final_evaluation_and_report": 5400, "protected_new_stage_reserve": 1800},
        "reserve_rule": "The historical 7h reserve is already inside the 17h stop threshold; do not subtract it a second time. The new 0.5h reserve is included within the 10h new-stage cap.",
        "measured_work": [],
        "wall_time_seconds": 0,
        "decision": "Use only existing weights except registered conditional branches. Include preprocessing and evaluation in measured new-stage use. Stop all new compute at the earliest applicable cap."
    }
    json_write(protocol / "budget.json", budget)

    stage_names = ["A", "B1", "B2", "B3", "B4", "B5", "B6", "C", "decision", "branches", "final_evaluation", "report", "integrity"]
    stages = {name: {"status": "planned", "evidence": []} for name in stage_names}
    json_write(protocol / "stage_status.json", {
        "schema_version": 1, "updated_at_utc": datetime.now(timezone.utc).isoformat(), "stages": stages
    })

    baseline = protection_snapshot()
    json_write(protection / "before.json", baseline)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["register", "verify-before", "verify-after", "record"])
    parser.add_argument("--name")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--bucket", default="unallocated")
    parser.add_argument("--category", default="compute")
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--note")
    args = parser.parse_args()
    if args.command == "register":
        register()
    elif args.command == "record":
        if args.name is None or args.seconds is None:
            parser.error("record requires --name and --seconds")
        record_operation(args.name, args.seconds, bucket=args.bucket, category=args.category,
                         evidence=args.evidence, note=args.note)
    else:
        path = ARTIFACTS / "protection" / "before.json"
        output = ARTIFACTS / "protection" / f"{args.command}.json"
        result = verify_snapshot(path, output)
        print(json.dumps({k: v for k, v in result.items() if k not in {"changed", "missing", "added"}}, indent=2))
        if result["status"] != "passed":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
