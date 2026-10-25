from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tlab.config import load_config
from tlab.data.catalog import FIRST_ITEM_ID, OOV_ID
from tlab.data.manifest import sha256_file, write_json_atomic
from tlab.data.schema import LISTEN_COLUMNS, require_columns
from tlab.data.splits import user_hash_partition
from tlab.data.temporal import make_boundaries


def _resolve_path(config_path: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else config_path.parent.parent / path


def shifted_max_timestamp_raw(raw_max: float, timestamp_unit_seconds: float, shift_days: int) -> float:
    """Move a raw snapshot end earlier without changing source timestamp units."""
    if timestamp_unit_seconds <= 0:
        raise ValueError("timestamp_unit_seconds must be positive")
    if shift_days < 0:
        raise ValueError("cutoff shift days must be non-negative")
    return float(raw_max) - (int(shift_days) * 24 * 60 * 60 / float(timestamp_unit_seconds))


def _write_partitioned_rm_features(
    listens,
    catalog_frame,
    user_partition_frame,
    rm_cutoff: float,
    processed_dir: Path,
) -> dict[str, dict[str, str]]:
    import polars as pl

    rm_fit_events = (
        listens.filter(
            (pl.col("timestamp") < rm_cutoff)
            & (pl.col("is_organic") == 0)
        )
        .join(user_partition_frame.lazy(), on="uid", how="inner")
        .join(catalog_frame.lazy().select("item_id"), on="item_id", how="semi")
    )
    support_by_partition = rm_fit_events.group_by(["rm_user_partition", "item_id"]).agg(
        pl.len().alias("rm_training_support"),
    )
    duration_by_partition = (
        rm_fit_events.filter(
            pl.col("track_length_seconds").is_finite() & (pl.col("track_length_seconds") > 0)
        )
        .group_by(["rm_user_partition", "item_id"])
        .agg(pl.col("track_length_seconds").median().alias("rm_duration_seconds"))
    )
    support_by_partition = support_by_partition.join(
        duration_by_partition, on=["rm_user_partition", "item_id"], how="left",
    ).collect(engine="streaming")
    outputs: dict[str, dict[str, str]] = {}
    for partition in ("fit", "audit"):
        rm_metadata = (
            catalog_frame.lazy().select("item_id", "token_id")
            .join(support_by_partition.filter(pl.col("rm_user_partition") == partition).lazy(), on="item_id", how="left")
            .with_columns(
                pl.col("rm_training_support").fill_null(0),
                pl.col("rm_duration_seconds").is_null().alias("rm_duration_missing"),
            )
            .sort("token_id")
            .collect(engine="streaming")
        )
        feature_path = processed_dir / f"rm_item_features_{partition}.parquet"
        rm_metadata.write_parquet(feature_path, compression="zstd")
        outputs[partition] = {"path": str(feature_path.resolve()), "sha256": sha256_file(feature_path)}
    return outputs


def rebuild_rm_features(config_path: str | Path) -> dict[str, Any]:
    """Recompute partition-isolated RM features from already prepared files."""
    import polars as pl

    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    protocol_path = processed_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    features = _write_partitioned_rm_features(
        pl.scan_parquet(processed_dir / "listens.parquet"),
        pl.read_parquet(processed_dir / "catalog.parquet").lazy(),
        pl.read_parquet(processed_dir / "rm_user_partitions.parquet"),
        float(protocol["boundaries_source_units"]["rm_validation_start_raw"]),
        processed_dir,
    )
    protocol["rm_item_features_by_partition"] = features
    protocol["rm_item_features_path"] = features["fit"]["path"]
    protocol["rm_item_features_sha256"] = features["fit"]["sha256"]
    protocol["rm_audit_item_features_path"] = features["audit"]["path"]
    protocol["rm_audit_item_features_sha256"] = features["audit"]["sha256"]
    write_json_atomic(protocol_path, protocol)
    return {"rm_item_features_by_partition": features}


def inspect_dataset(config_path: str | Path) -> dict[str, Any]:
    try:
        import polars as pl
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before inspecting Parquet data") from exc

    config = load_config(config_path)
    raw_dir = _resolve_path(config.path, config.data.get("source_dir", "data/raw"))
    files = config.dataset.get("files", {})
    listen_name = files.get("listens")
    if not listen_name:
        raise ValueError("Configuration must define dataset.files.listens")
    listen_path = raw_dir / Path(listen_name).name
    if not listen_path.exists():
        raise FileNotFoundError(f"Missing listens file: {listen_path}; run tlab.data.download first")

    parquet_meta = pq.read_metadata(listen_path)
    parquet_schema = pq.read_schema(listen_path)
    columns = {field.name for field in parquet_schema}
    require_columns(columns, LISTEN_COLUMNS, "listens")
    listens = pl.scan_parquet(listen_path)
    exprs = [
        pl.len().alias("rows"),
        pl.col("uid").n_unique().alias("users"),
        pl.col("item_id").n_unique().alias("items"),
        pl.col("timestamp").min().alias("timestamp_min"),
        pl.col("timestamp").max().alias("timestamp_max"),
        pl.col("timestamp").null_count().alias("timestamp_nulls"),
        pl.col("uid").null_count().alias("uid_nulls"),
        pl.col("item_id").null_count().alias("item_nulls"),
        pl.col("played_ratio_pct").null_count().alias("ratio_nulls"),
        pl.col("track_length_seconds").null_count().alias("duration_nulls"),
        (pl.col("is_organic") == 0).sum().alias("non_organic_rows"),
        (pl.col("is_organic") == 1).sum().alias("organic_rows"),
        (pl.col("played_ratio_pct") > 100).sum().alias("ratio_over_100_rows"),
        (pl.col("track_length_seconds") <= 0).sum().alias("nonpositive_duration_rows"),
    ]
    summary = listens.select(exprs).collect(engine="streaming").to_dicts()[0]
    tie_summary = (
        listens.group_by(["uid", "timestamp"]).len()
        .filter(pl.col("len") > 1)
        .select(pl.len().alias("tie_blocks"), pl.col("len").sum().alias("tie_events"))
        .collect(engine="streaming")
    )
    tie_stats = tie_summary.to_dicts()[0] if tie_summary.height else {"tie_blocks": 0, "tie_events": 0}

    user_length_stats = (
        listens.group_by("uid").len().select(
            pl.col("len").min().alias("min"),
            pl.col("len").median().alias("median"),
            pl.col("len").quantile(0.95).alias("p95"),
            pl.col("len").max().alias("max"),
        ).collect(engine="streaming").to_dicts()[0]
    )
    duration_conflicts = (
        listens.filter(
            pl.col("track_length_seconds").is_finite() & (pl.col("track_length_seconds") > 0)
        )
        .group_by("item_id").agg(pl.col("track_length_seconds").n_unique().alias("duration_values"))
        .filter(pl.col("duration_values") > 1)
        .select(pl.len().alias("items_with_conflicting_duration"))
        .collect(engine="streaming")
    )
    duration_conflict_count = (
        int(duration_conflicts["items_with_conflicting_duration"][0]) if duration_conflicts.height else 0
    )

    intervals = (
        listens.sort(["uid", "timestamp"])
        .with_columns(pl.col("timestamp").diff().over("uid").alias("inter_event_gap_raw"))
        .select(
            pl.col("inter_event_gap_raw").min().alias("min"),
            pl.col("inter_event_gap_raw").median().alias("median"),
            pl.col("inter_event_gap_raw").quantile(0.95).alias("p95"),
            (pl.col("inter_event_gap_raw") == 0).sum().alias("zero_gap_rows"),
        )
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    timestamp_unit = config.dataset.get("timestamp_unit_seconds")
    timestamp_bin = config.dataset.get("timestamp_bin_seconds")
    timestamp_grid_mismatches = None
    catalog_coverage: dict[str, Any] | None = None
    if timestamp_unit and timestamp_bin:
        timestamp_grid_mismatches = (
            listens.select(
                (pl.col("timestamp").cast(pl.Int64, strict=False) % int(timestamp_bin) != 0).sum().alias("count")
            ).collect(engine="streaming")["count"][0]
        )
        t_end = float(summary["timestamp_max"]) + float(timestamp_bin) / float(timestamp_unit)
        train_cutoff = t_end - int(config.dataset.get("train_days", 21)) * 86400 / float(timestamp_unit)
        train_counts = (
            listens.filter((pl.col("timestamp") < train_cutoff) & (pl.col("is_organic") == 0))
            .group_by("item_id").agg(pl.len().alias("n"))
            .sort(["n", "item_id"], descending=[True, False])
            .head(50000).collect(engine="streaming")
        )
        heldout = listens.filter((pl.col("timestamp") >= train_cutoff) & (pl.col("is_organic") == 0)).select("item_id")
        heldout_count = heldout.select(pl.len()).collect(engine="streaming").item()
        coverage: dict[str, Any] = {"eligible_non_organic_targets": int(heldout_count)}
        for size in (10000, 20000, 50000):
            selected = train_counts.head(size).select("item_id").lazy()
            covered = heldout.join(selected, on="item_id", how="semi").select(pl.len()).collect(engine="streaming").item()
            coverage[str(size)] = {
                "covered_targets": int(covered),
                "coverage": float(covered / heldout_count) if heldout_count else None,
            }
        catalog_coverage = coverage

    auxiliary: dict[str, Any] = {}
    for key in ("likes", "unlikes", "dislikes", "undislikes"):
        if key not in files:
            continue
        event_path = raw_dir / Path(files[key]).name
        if event_path.exists():
            event_meta = pq.read_metadata(event_path)
            auxiliary[key] = {"rows": event_meta.num_rows, "schema": str(pq.read_schema(event_path))}

    if not timestamp_unit or not timestamp_bin:
        time_unit_status = "unverified"
    elif not config.dataset.get("timestamp_unit_evidence"):
        time_unit_status = "configured_without_source_evidence"
    elif timestamp_grid_mismatches != 0:
        time_unit_status = "source_evidence_present_but_grid_mismatches_found"
    else:
        time_unit_status = "source_formula_and_parquet_grid_verified"

    report: dict[str, Any] = {
        "source_path": str(listen_path.resolve()),
        "sha256": sha256_file(listen_path),
        "row_groups": parquet_meta.num_row_groups,
        "schema": {field.name: str(field.type) for field in parquet_schema},
        "listen_summary": summary,
        "timestamp_ties": tie_stats,
        "listen_history_length_per_user": user_length_stats,
        "duration_conflicts": {"items_with_multiple_positive_values": duration_conflict_count},
        "inter_event_gaps_raw_units": intervals,
        "timestamp_grid_mismatches": timestamp_grid_mismatches,
        "timestamp_grid_seconds": timestamp_bin,
        "estimated_catalog_coverage": catalog_coverage,
        "auxiliary_events": auxiliary,
        "timestamp_unit_seconds": timestamp_unit,
        "timestamp_bin_seconds": timestamp_bin,
        "timestamp_unit_evidence": config.dataset.get("timestamp_unit_evidence"),
        "time_unit_status": time_unit_status,
    }
    output_dir = _resolve_path(config.path, config.project.get("artifacts_dir", "artifacts"))
    write_json_atomic(output_dir / "eda.json", report)
    return report


def prepare_dataset(config_path: str | Path) -> dict[str, Any]:
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before preparing Parquet data") from exc

    config = load_config(config_path)
    timestamp_unit = config.dataset.get("timestamp_unit_seconds")
    timestamp_bin = config.dataset.get("timestamp_bin_seconds")
    if timestamp_unit is None or timestamp_bin is None:
        raise ValueError(
            "Timestamp units are unresolved. Run with --inspect-only, verify the Yambda timestamp encoding, "
            "then set timestamp_unit_seconds and timestamp_bin_seconds in the config."
        )
    report = inspect_dataset(config_path)
    summary = report["listen_summary"]
    raw_max = float(summary["timestamp_max"])
    cutoff_shift_days = int(config.dataset.get("cutoff_shift_days", 0))
    effective_raw_max = shifted_max_timestamp_raw(raw_max, float(timestamp_unit), cutoff_shift_days)
    boundaries = make_boundaries(
        max_timestamp_raw=effective_raw_max,
        timestamp_unit_seconds=float(timestamp_unit),
        timestamp_bin_seconds=float(timestamp_bin),
        train_days=int(config.dataset.get("train_days", 21)),
        validation_days=int(config.dataset.get("validation_days", 14)),
        test_days=int(config.dataset.get("test_days", 7)),
        rm_fit_days=int(config.dataset.get("rm_validation_days", 7)),
        rl_anchor_days=int(config.dataset.get("rl_anchor_days", 7)),
    )
    raw_dir = _resolve_path(config.path, config.data.get("source_dir", "data/raw"))
    processed_dir = _resolve_path(config.path, config.data.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    listen_name = config.dataset["files"]["listens"]
    listen_path = raw_dir / Path(listen_name).name
    # A historical fold must not retain any later rows in the prepared store:
    # its evaluation targets and all derived features stop at the fold end.
    listens = pl.scan_parquet(listen_path).filter(pl.col("timestamp") < boundaries.end_raw)

    train_cutoff = boundaries.train_cutoff_raw
    catalog_size = int(config.dataset.get("catalog_size", 20000))
    catalog_frame = (
        listens.filter((pl.col("timestamp") < train_cutoff) & (pl.col("is_organic") == 0))
        .group_by("item_id")
        .agg(pl.len().alias("training_count"))
        .sort(["training_count", "item_id"], descending=[True, False])
        .head(catalog_size)
        .collect(engine="streaming")
        .with_row_index("token_id", offset=FIRST_ITEM_ID)
        .with_columns(pl.col("token_id").cast(pl.Int32))
    )
    if catalog_frame.is_empty():
        raise ValueError("The pre-cutoff non-organic training catalog is empty")
    catalog_ids = catalog_frame.select("item_id").lazy()
    global_metadata = (
        listens.filter(
            (pl.col("timestamp") < train_cutoff)
            & pl.col("track_length_seconds").is_finite()
            & (pl.col("track_length_seconds") > 0)
        )
        .join(catalog_ids, on="item_id", how="semi")
        .group_by("item_id")
        .agg(pl.col("track_length_seconds").median().alias("catalog_duration_seconds"))
    )
    catalog_frame = (
        catalog_frame.lazy().join(global_metadata, on="item_id", how="left")
        .with_columns(pl.col("catalog_duration_seconds").is_null().alias("catalog_duration_missing"))
        .sort("token_id")
        .collect(engine="streaming")
    )
    catalog_path = processed_dir / "catalog.parquet"
    catalog_frame.write_parquet(catalog_path, compression="zstd")
    catalog_hash_rows = catalog_frame.select("item_id", "training_count").iter_rows()
    import hashlib

    catalog_digest = hashlib.sha256("".join(f"{item}\t{count}\n" for item, count in catalog_hash_rows).encode()).hexdigest()

    rm_users = (
        listens.filter(pl.col("timestamp") < boundaries.rm_validation_start_raw)
        .select("uid").unique().sort("uid").collect(engine="streaming").get_column("uid").to_list()
    )
    rm_seed = int(config.dataset.get("rm_user_split_seed", 17))
    audit_fraction = float(config.dataset.get("rm_audit_fraction", 0.2))
    user_partition_frame = pl.DataFrame([
        pl.Series("uid", rm_users, dtype=listens.collect_schema()["uid"]),
        pl.Series("rm_user_partition", [user_hash_partition(uid, rm_seed, audit_fraction) for uid in rm_users]),
    ])
    user_partition_path = processed_dir / "rm_user_partitions.parquet"
    user_partition_frame.write_parquet(user_partition_path, compression="zstd")
    rm_user_partition_counts = {
        row["rm_user_partition"]: row["len"]
        for row in user_partition_frame.group_by("rm_user_partition").len().to_dicts()
    }

    rm_features_by_partition = _write_partitioned_rm_features(
        listens, catalog_frame, user_partition_frame, boundaries.rm_validation_start_raw, processed_dir,
    )

    tie_counts = (
        listens.group_by(["uid", "timestamp"]).len()
        .rename({"len": "timestamp_tie_count"})
    )
    prepared = (
        listens.join(catalog_frame.lazy().select("item_id", "token_id"), on="item_id", how="left")
        .join(tie_counts, on=["uid", "timestamp"], how="left")
        .join(user_partition_frame.lazy(), on="uid", how="left")
        .with_columns(
            pl.col("token_id").fill_null(OOV_ID).cast(pl.Int32),
            (pl.col("played_ratio_pct").cast(pl.Float64, strict=False) / 100.0).alias("ratio_raw"),
            pl.col("track_length_seconds").cast(pl.Float64, strict=False).alias("duration_raw_seconds"),
            pl.when(pl.col("timestamp") < boundaries.train_cutoff_raw).then(pl.lit("train"))
            .when(pl.col("timestamp") < boundaries.test_start_raw).then(pl.lit("validation"))
            .otherwise(pl.lit("test")).alias("global_split"),
            pl.when(pl.col("timestamp") < boundaries.rm_validation_start_raw).then(pl.lit("F_RM"))
            .when(pl.col("timestamp") < boundaries.rl_anchor_start_raw).then(pl.lit("V_RM"))
            .when(pl.col("timestamp") < boundaries.train_cutoff_raw).then(pl.lit("A_RL"))
            .otherwise(pl.lit(None)).alias("train_subsplit"),
        )
        .with_columns(
            pl.when(pl.col("ratio_raw").is_finite())
            .then(pl.col("ratio_raw").clip(0.0, 1.0))
            .otherwise(None).alias("ratio_clipped"),
            pl.when(
                pl.col("ratio_raw").is_finite()
                & pl.col("duration_raw_seconds").is_finite()
                & (pl.col("duration_raw_seconds") > 0)
            )
            .then(pl.col("ratio_raw") * pl.col("duration_raw_seconds"))
            .otherwise(None).alias("seconds_proxy_raw"),
        )
    )

    # Fixed clipping and scaling statistics come only from F_RM events.
    fit = prepared.filter(pl.col("train_subsplit") == "F_RM")
    seconds_cap = fit.select(pl.col("seconds_proxy_raw").quantile(0.995).alias("cap")).collect(engine="streaming")["cap"][0]
    if seconds_cap is None or not float(seconds_cap) > 0:
        seconds_cap = 1.0
    capped = pl.col("seconds_proxy_raw").clip(0.0, float(seconds_cap))
    seconds_scale = (
        fit.select(capped.quantile(0.95).alias("scale")).collect(engine="streaming")["scale"][0]
    )
    if seconds_scale is None or not float(seconds_scale) > 0:
        seconds_scale = 1.0
    prepared = prepared.with_columns(
        pl.col("seconds_proxy_raw").clip(0.0, float(seconds_cap)).alias("seconds_proxy_capped"),
        (pl.col("timestamp_tie_count") > 1).alias("ambiguous_timestamp_block"),
    ).with_columns(
        (pl.col("seconds_proxy_capped") / float(seconds_scale)).alias("seconds_scaled")
    ).sort(["uid", "timestamp"])

    prepared_path = processed_dir / "listens.parquet"
    prepared.sink_parquet(prepared_path, compression="zstd", maintain_order=True, engine="streaming")
    protocol = {
        "timestamp_unit_seconds": float(timestamp_unit),
        "timestamp_bin_seconds": float(timestamp_bin),
        "boundaries_source_units": boundaries.__dict__,
        "cutoff_shift_days": cutoff_shift_days,
        "fold_end_raw_exclusive": boundaries.end_raw,
        "catalog_size": catalog_frame.height,
        "catalog_sha256": catalog_digest,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256_file": sha256_file(catalog_path),
        "rm_item_features_by_partition": rm_features_by_partition,
        "rm_item_features_path": rm_features_by_partition["fit"]["path"],
        "rm_item_features_sha256": rm_features_by_partition["fit"]["sha256"],
        "rm_audit_item_features_path": rm_features_by_partition["audit"]["path"],
        "rm_audit_item_features_sha256": rm_features_by_partition["audit"]["sha256"],
        "rm_user_partitions_path": str(user_partition_path.resolve()),
        "rm_user_partition_seed": rm_seed,
        "rm_user_partition_counts_f_rm": rm_user_partition_counts,
        "processed_listens_path": str(prepared_path.resolve()),
        "processed_listens_sha256": sha256_file(prepared_path),
        "seconds_cap_f_rm_p99_5": float(seconds_cap),
        "seconds_scale_f_rm_p95": float(seconds_scale),
        "source_sha256": report["sha256"],
        "timestamp_unit_evidence": report["timestamp_unit_evidence"],
        "timestamp_grid_mismatches": report["timestamp_grid_mismatches"],
        "time_unit_status": "source formula and Parquet five-second grid verified",
    }
    write_json_atomic(processed_dir / "protocol.json", protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and prepare Yambda temporal data")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--inspect-only", action="store_true", help="Write EDA without building split data")
    parser.add_argument("--rebuild-rm-features", action="store_true", help="Rebuild fit/audit-isolated RM metadata from processed data")
    args = parser.parse_args()
    if args.inspect_only and args.rebuild_rm_features:
        parser.error("--inspect-only and --rebuild-rm-features cannot be combined")
    result = (
        inspect_dataset(args.config) if args.inspect_only
        else rebuild_rm_features(args.config) if args.rebuild_rm_features
        else prepare_dataset(args.config)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
