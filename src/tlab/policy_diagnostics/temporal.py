"""Pre-outcome temporal eligibility audit for fixed panel_tools.2 panels."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .protocol import ARTIFACTS, ROOT, json_write, record_operation, update_stage


PANELS = {
    "a_rl_train": ROOT / "artifacts/residual_support/panels/a_rl_train.jsonl",
    "development": ROOT / "artifacts/residual_support/panels/validation_development.jsonl",
    "retrospective_final": ROOT / "artifacts/residual_support/panels/test_retrospective.jsonl",
    "common_open_6655": ROOT / "artifacts/panel_tools/evaluation/frozen_primary_validation_panel.jsonl",
}


def _read_jsonl(path: Path, panel: str) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({
                "panel": panel,
                "event_id": str(row["event_id"]),
                "uid": int(row["uid"]),
                "position": int(row["position"]),
                "raw_item_id": int(row["raw_item_id"]),
            })
    return rows


def run_temporal_audit() -> dict[str, Any]:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    protocol_path = ROOT / "data/processed_full_model_tminus14/protocol.json"
    dataset = json.loads(protocol_path.read_text(encoding="utf-8"))
    actor_manifest_path = ROOT / "models/actor/strong_run_manifest.json"
    actor_manifest = json.loads(actor_manifest_path.read_text(encoding="utf-8"))
    if actor_manifest.get("processed_listens_sha256") != dataset.get("processed_listens_sha256"):
        raise ValueError("strong actor manifest is bound to a different processed sequence source")
    actor_cutoff = float(dataset["boundaries_source_units"]["train_cutoff_raw"])
    rm = json.loads((ROOT / "artifacts/residual_support/rm/rm_diagnostics.json").read_text(encoding="utf-8"))
    rm_model_cutoff = float(rm["prior_provenance"]["fit_targets_time_end_exclusive"])
    policy_prior_cutoff = float(rm["prior_provenance"]["policy_prior_cutoff_exclusive"])
    all_cutoff = max(actor_cutoff, rm_model_cutoff, policy_prior_cutoff)

    targets = [row for name, path in PANELS.items() for row in _read_jsonl(path, name)]
    if len({(r["panel"], r["event_id"]) for r in targets}) != len(targets):
        raise ValueError("duplicate panel event IDs within a role")
    user_ids = sorted({r["uid"] for r in targets})
    target_frame = pl.DataFrame(targets)

    # Position is assigned only from timestamp order and deduplicated timestamp blocks,
    # matching SequenceStore's one-row-per-ambiguous-timestamp contract.
    timeline_path = ROOT / "data/processed_full_model_tminus14/listens.parquet"
    timeline = (
        pl.scan_parquet(timeline_path)
        .filter(pl.col("uid").is_in(user_ids))
        .select("uid", "timestamp", "item_id", "ambiguous_timestamp_block", "global_split")
        .unique(subset=["uid", "timestamp"], keep="first")
        .sort(["uid", "timestamp"])
        .with_columns((pl.col("timestamp").cum_count().over("uid") - 1).cast(pl.Int64).alias("position"))
        .collect(engine="streaming")
    )
    matched = target_frame.join(timeline, on=["uid", "position"], how="left")
    unmatched = int(matched["timestamp"].null_count())
    matched = matched.with_columns([
        (pl.col("item_id").cast(pl.Int64) == pl.col("raw_item_id")).fill_null(False).alias("target_item_match"),
        (pl.col("timestamp") >= rm_model_cutoff).fill_null(False).alias("after_rm_model_cutoff"),
        (pl.col("timestamp") >= policy_prior_cutoff).fill_null(False).alias("after_policy_prior_cutoff"),
        (pl.col("timestamp") >= actor_cutoff).fill_null(False).alias("after_actor_cutoff"),
        (pl.col("timestamp") >= all_cutoff).fill_null(False).alias("common_temporally_eligible"),
    ])

    by_panel: dict[str, Any] = {}
    for role in PANELS:
        subset = matched.filter(pl.col("panel") == role)
        n = subset.height
        eligible = subset.filter(pl.col("common_temporally_eligible"))
        by_panel[role] = {
            "path": PANELS[role].relative_to(ROOT).as_posix(),
            "n_rows": n,
            "n_users": subset["uid"].n_unique(),
            "unmatched_position_rows": int(subset["timestamp"].null_count()),
            "target_raw_id_mismatch_rows": int((~subset["target_item_match"]).sum()),
            "timestamp_tie_rows": int(subset["ambiguous_timestamp_block"].fill_null(False).sum()),
            "actor_cutoff_eligible_rows": int(subset["after_actor_cutoff"].sum()),
            "common_eligible_rows": int(eligible.height),
            "common_eligible_users": int(eligible["uid"].n_unique()),
            "excluded_before_common_cutoff": int(n - eligible.height),
            "timestamp_min": int(subset["timestamp"].min()) if n and subset["timestamp"].null_count() == 0 else None,
            "timestamp_max": int(subset["timestamp"].max()) if n and subset["timestamp"].null_count() == 0 else None,
            "eligible_timestamp_min": int(eligible["timestamp"].min()) if eligible.height else None,
            "eligible_timestamp_max": int(eligible["timestamp"].max()) if eligible.height else None,
        }
    duration = time.perf_counter() - started
    rows_out = matched.select([
        "panel", "event_id", "uid", "position", "raw_item_id", "item_id", "timestamp",
        "ambiguous_timestamp_block", "global_split", "target_item_match",
        "after_rm_model_cutoff", "after_policy_prior_cutoff", "after_actor_cutoff",
        "common_temporally_eligible",
    ]).sort(["panel", "uid"])
    csv_path = ARTIFACTS / "evaluation/temporal_eligibility_rows.csv"
    json_path = ARTIFACTS / "evaluation/temporal_eligibility.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("temporal audit outputs already exist")
    rows_out.write_csv(csv_path)
    result = {
        "schema_version": 1,
        "status": "complete" if unmatched == 0 and all(x["target_raw_id_mismatch_rows"] == 0 for x in by_panel.values()) else "incomplete_alignment",
        "role": "pre-outcome row-level temporal eligibility; no target outcomes used for selection",
        "started_at_utc": started_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(duration, 3),
        "source_hashes": {
            "processed_listens_sha256": dataset["processed_listens_sha256"],
            "actor_checkpoint_sha256": actor_manifest["best_checkpoint_sha256"],
            "rm_diagnostics_prior_cutoffs": rm["prior_provenance"],
        },
        "cutoffs": {
            "actor_training_targets_exclusive": actor_cutoff,
            "optimized_rm_fit_targets_end_exclusive": rm_model_cutoff,
            "policy_item_priors_exclusive": policy_prior_cutoff,
            "all_method_common_minimum_target_time_inclusive": all_cutoff,
            "common_panel_selection_rule": "all fixed panel rows whose target timestamp is at or after every fitted actor/RM/prior cutoff; no filtering by target hit, ratio, support, or outcome",
        },
        "panels": by_panel,
        "diagnostic": {
            "all_rows_unmatched_position": unmatched,
            "common_panel_temporally_valid_for_all_registered_methods": by_panel["common_open_6655"]["common_eligible_rows"] > 0 and unmatched == 0,
            "eligibility_uses_future_features_as_test": False,
            "historical_outcome_exposure_remains": True,
        },
        "row_output": csv_path.relative_to(ROOT).as_posix(),
    }
    json_write(json_path, result)
    record_operation("row_level_temporal_eligibility_gate", duration, bucket="A", category="preprocessing",
                     evidence=[json_path.relative_to(ROOT).as_posix(), csv_path.relative_to(ROOT).as_posix()])
    update_stage("A", "implemented", [json_path.relative_to(ROOT).as_posix()],
                 note="Temporal eligibility measured before A scoring; user outcomes were not read by the gate.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.parse_args()
    result = run_temporal_audit()
    print(json.dumps({"status": result["status"], "cutoffs": result["cutoffs"], "panels": result["panels"],
                      "runtime_seconds": result["runtime_seconds"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
