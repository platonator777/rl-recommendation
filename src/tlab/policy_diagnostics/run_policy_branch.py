"""Execute the pre-registered sample-count branch with budget accounting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tlab.residual_support.train_policy import run_policy_series

from .protocol import ARTIFACTS, ROOT, record_operation, update_stage


BRANCH = ARTIFACTS / "branches" / "sample_count_4_vs_16"
CONFIGS = [
    ROOT / "configs/policy_diagnostics/sample_count_sampled4.yaml",
    ROOT / "configs/policy_diagnostics/sample_count_sampled16.yaml",
    ROOT / "configs/policy_diagnostics/sample_count_exact.yaml",
]


def run(resume: bool = False) -> dict[str, object]:
    registration = json.loads((ARTIFACTS / "branches/branch_registration.json").read_text(encoding="utf-8"))
    budget_path = ARTIFACTS / "protocol/budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    allocation = int(registration["fit_budget"]["allocation_seconds"])
    already_recorded = sum(
        float(row["seconds"])
        for row in budget["measured_work"]
        if row["name"].startswith("policy_branch_sample_count_")
    )
    branch_started = time.perf_counter()
    summaries: list[dict[str, object]] = []
    update_stage(
        "branches",
        "in_progress",
        ["artifacts/policy_diagnostics/branches/branch_registration.json"],
        note="Executing all registered seeds and arms; a pilot is not treated as the main result.",
    )
    for config in CONFIGS:
        current = json.loads(budget_path.read_text(encoding="utf-8"))
        branch_used = already_recorded + (time.perf_counter() - branch_started)
        if branch_used >= allocation:
            raise RuntimeError("registered conditional-branch allocation is exhausted")
        if float(current["remaining_new_stage_seconds"]) <= 0 or float(
            current["remaining_to_total_stop_from_known_lower_bound_seconds"]
        ) <= 0:
            raise RuntimeError("a global compute stop threshold is exhausted")
        name = config.stem.removeprefix("sample_count_")
        started = time.perf_counter()
        error: str | None = None
        try:
            result = run_policy_series(config, ROOT, resume=resume)
            summaries.append(result)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - started
            summary_path = config.parent / ".." / ".." / "artifacts" / "policy_diagnostics" / "branches" / "sample_count_4_vs_16" / name / "main_fit_summary.json"
            evidence = [summary_path.resolve().relative_to(ROOT).as_posix()] if summary_path.resolve().exists() else []
            record_operation(
                f"policy_branch_sample_count_{name}",
                elapsed,
                bucket="conditional_branches",
                category="training_preprocessing_and_inline_evaluation",
                evidence=evidence,
                note=error or "Includes cache loading, scaler/baseline setup, training and all inline development checkpoint evaluations.",
            )
    result = {
        "schema_version": 1,
        "status": "main_fits_completed",
        "branch_id": registration["branch_id"],
        "completed_arms": [config.stem.removeprefix("sample_count_") for config in CONFIGS],
        "arm_count": len(summaries),
        "run_count": sum(int(summary["run_count"]) for summary in summaries),
        "planned_run_count": sum(int(summary["planned_run_count"]) for summary in summaries),
        "branch_elapsed_seconds_in_this_invocation": round(time.perf_counter() - branch_started, 3),
        "budget_includes_preprocessing_and_inline_development_evaluation": True,
    }
    output = BRANCH / "branch_fit_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    update_stage("branches", "fits_complete", [
        "artifacts/policy_diagnostics/branches/branch_registration.json",
        "artifacts/policy_diagnostics/branches/sample_count_4_vs_16/branch_fit_summary.json",
    ])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="resume only signature-matched partial branch runs")
    args = parser.parse_args()
    print(json.dumps(run(resume=args.resume), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
