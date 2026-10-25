from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
import yaml
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tlab.data.store import SequenceStore
from tlab.residual_support.features import (
    CANDIDATE_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    ItemPriors,
    RM_FEATURE_NAMES,
    history_context_features,
    rm_feature_row,
)
from tlab.panel_tools.statistics import paired_user_bootstrap


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _group_event_timestamps(processed_listens: Path, store: SequenceStore) -> dict[int, np.ndarray]:
    """Return timestamp-by-sequence-position for train events, collapsing timestamp ties."""
    started = time.perf_counter()
    grouped = (
        pl.scan_parquet(processed_listens)
        .filter(pl.col("global_split") == "train")
        .select("uid", "timestamp")
        .unique()
        .sort(["uid", "timestamp"])
        .group_by("uid", maintain_order=True)
        .agg(pl.col("timestamp").cast(pl.UInt32).alias("timestamps"))
        .collect(engine="streaming")
    )
    result = {
        int(uid): np.asarray(values, dtype=np.uint32)
        for uid, values in grouped.iter_rows()
    }
    for user in store.users:
        timeline = result.get(int(user.uid))
        if timeline is None:
            continue
        train_length = int(np.count_nonzero(np.asarray(user.split_codes, dtype=np.uint8) == 0))
        if len(timeline) != train_length:
            raise ValueError(f"timestamp/cache alignment failed for uid={user.uid}: {len(timeline)} != {train_length}")
    print(f"Built train timestamp indexes for {len(result):,} users in {time.perf_counter()-started:.1f}s", flush=True)
    return result


def build_item_priors(
    processed_listens: Path,
    catalog: pl.DataFrame,
    *,
    cutoff_exclusive: float,
    source_split: str,
) -> ItemPriors:
    if not math.isfinite(float(cutoff_exclusive)):
        raise ValueError("as-of cutoff must be finite")
    catalog = catalog.sort("token_id")
    ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    raw_to_index = {int(raw): i for i, raw in enumerate(ids.tolist())}
    n = len(ids)
    prior_counts = np.zeros(n, dtype=np.float64)
    ratio_counts = np.zeros(n, dtype=np.float64)
    ratio_sums = np.zeros(n, dtype=np.float64)
    durations = np.full(n, np.nan, dtype=np.float64)
    query = pl.scan_parquet(processed_listens).filter(
        (pl.col("global_split") == "train")
        & (pl.col("train_subsplit") == "F_RM")
        & (pl.col("timestamp") < float(cutoff_exclusive))
        & (pl.col("is_organic") == 0)
        & (pl.col("ambiguous_timestamp_block") == False)
        & (pl.col("token_id") >= 3)
    ).select("item_id", "ratio_raw", "track_length_seconds")
    aggregates = (
        query.group_by("item_id")
        .agg(
            pl.len().alias("popularity"),
            pl.col("ratio_raw").filter(pl.col("ratio_raw").is_finite()).mean().alias("ratio_mean"),
            pl.col("ratio_raw").filter(pl.col("ratio_raw").is_finite()).count().alias("ratio_count"),
            pl.col("track_length_seconds").filter(pl.col("track_length_seconds") > 0).median().alias("duration"),
        )
        .collect(engine="streaming")
    )
    for row in aggregates.iter_rows(named=True):
        index = raw_to_index.get(int(row["item_id"]))
        if index is None:
            continue
        prior_counts[index] = float(row["popularity"] or 0)
        ratio_counts[index] = float(row["ratio_count"] or 0)
        if row["ratio_mean"] is not None and math.isfinite(float(row["ratio_mean"])):
            ratio_sums[index] = float(row["ratio_mean"]) * ratio_counts[index]
        if row["duration"] is not None and math.isfinite(float(row["duration"])) and float(row["duration"]) > 0:
            durations[index] = float(row["duration"])
    ratio_mean = np.divide(ratio_sums, ratio_counts, out=np.full(n, np.nan), where=ratio_counts > 0)
    return ItemPriors(
        train_popularity_count=prior_counts.astype(np.float32),
        ratio_prior_mean=ratio_mean.astype(np.float32),
        ratio_prior_count=ratio_counts.astype(np.float32),
        duration_seconds=durations.astype(np.float32),
        duration_missing=(~np.isfinite(durations)).astype(np.uint8),
        cutoff_raw_exclusive=float(cutoff_exclusive), source_split=source_split,
    )


def save_priors(path: Path, priors: ItemPriors, catalog_mapping_sha256: str) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_popularity_count=priors.train_popularity_count,
        ratio_prior_mean=priors.ratio_prior_mean,
        ratio_prior_count=priors.ratio_prior_count,
        duration_seconds=priors.duration_seconds,
        duration_missing=priors.duration_missing,
        cutoff_raw_exclusive=np.asarray([priors.cutoff_raw_exclusive], dtype=np.float64),
        source_split=np.asarray([priors.source_split]),
        catalog_mapping_sha256=np.asarray([catalog_mapping_sha256]),
    )
    return sha256_file(path)


def load_priors(path: str | Path, expected_mapping_sha256: str | None = None) -> ItemPriors:
    with np.load(path, allow_pickle=False) as archive:
        mapping = str(archive["catalog_mapping_sha256"][0])
        if expected_mapping_sha256 is not None and mapping != expected_mapping_sha256:
            raise ValueError("item prior catalog raw-ID/token-ID mapping mismatch")
        return ItemPriors(
            archive["train_popularity_count"].copy(), archive["ratio_prior_mean"].copy(),
            archive["ratio_prior_count"].copy(), archive["duration_seconds"].copy(),
            archive["duration_missing"].copy(), float(archive["cutoff_raw_exclusive"][0]),
            str(archive["source_split"][0]),
        )


def _sample_positions(store: SequenceStore, timestamps: dict[int, np.ndarray], *, partition: str,
                      positions_name: str, lower: float, upper: float, max_events: int, seed: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for user_index, user in enumerate(store.users):
        if user.rm_user_partition != partition:
            continue
        timeline = timestamps.get(int(user.uid))
        if timeline is None:
            continue
        raw_positions = getattr(user, positions_name)
        candidates = [int(p) for p in raw_positions if int(p) < len(timeline) and lower <= float(timeline[int(p)]) < upper]
        candidates.sort(key=lambda p: hashlib.sha256(f"{bytes.fromhex('76345f313a726d3a').decode('ascii')}{seed}:{user.uid}:{p}".encode()).digest())
        selected.extend((user_index, position) for position in candidates[:max_events])
    selected.sort(key=lambda pair: (store.users[pair[0]].uid, pair[1]))
    return selected


def _rm_rows(store: SequenceStore, selected: list[tuple[int, int]], priors: ItemPriors,
             raw_to_index: dict[int, int]) -> dict[str, np.ndarray]:
    features, labels, uids, items, repeated, positive = [], [], [], [], [], []
    for user_index, position in selected:
        user = store.users[user_index]
        raw_id = int(user.raw_item_ids[position])
        item_index = raw_to_index.get(raw_id)
        ratio = float(user.ratio_raw[position])
        if item_index is None or not math.isfinite(ratio):
            continue
        features.append(rm_feature_row(user, position, item_index, raw_id, priors))
        labels.append(float(np.clip(ratio, 0.0, 1.0)))
        uids.append(int(user.uid))
        items.append(item_index)
        is_repeat = bool(np.any(np.asarray(user.raw_item_ids[:position], dtype=np.int64) == raw_id))
        repeated.append(is_repeat)
        positive.append(ratio >= 0.5)
    return {
        "x": np.asarray(features, dtype=np.float32), "y": np.asarray(labels, dtype=np.float32),
        "uid": np.asarray(uids, dtype=np.int64), "item_index": np.asarray(items, dtype=np.int32),
        "is_repeat": np.asarray(repeated, dtype=bool), "positive": np.asarray(positive, dtype=bool),
    }


def _regression_metrics(y: np.ndarray, prediction: np.ndarray, uid: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    groups: dict[int, list[float]] = defaultdict(list)
    for user, err in zip(uid.tolist(), np.abs(y - p).tolist()):
        groups[int(user)].append(float(err))
    calibration = []
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    for start, stop in zip(edges[:-1], edges[1:]):
        mask = (p >= start) & ((p <= stop) if stop == edges[-1] else (p < stop))
        if mask.any():
            calibration.append({"n": int(mask.sum()), "mean_prediction": float(p[mask].mean()),
                                "mean_target": float(y[mask].mean()), "prediction_min": float(p[mask].min()),
                                "prediction_max": float(p[mask].max())})
    return {
        "mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "n_rows": int(len(y)), "n_users": len(groups),
        "calibration_by_prediction_decile": calibration,
        "per_user_mae": {str(user): float(np.mean(values)) for user, values in groups.items()},
    }


def _support_groups(item_indices: np.ndarray, rm_fit_counts: np.ndarray) -> np.ndarray:
    count = np.asarray(rm_fit_counts, dtype=np.int64)[np.asarray(item_indices, dtype=np.int64)]
    return np.select([count == 0, count < 10, count < 100], ["unsupported", "rare_1_9", "rare_10_99"], default="supported_100_plus")


def run_rm_experiment(config_path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = Path(root).resolve() if root is not None else config_path.parents[2]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = _resolve(project_root, cfg["outputs"]["root"])
    allowed = (project_root / "artifacts/residual_support").resolve()
    if allowed not in output_root.resolve().parents:
        raise ValueError("RM outputs must be below artifacts/residual_support")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite RM outputs: {output_root}")
    processed_dir = _resolve(project_root, cfg["processed_dir"])
    source_path = processed_dir / "listens.parquet"
    processed_protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    boundaries = processed_protocol["boundaries_source_units"]
    train_index = _resolve(project_root, cfg["sequence_index"])
    store = SequenceStore.load_cache(train_index, expected_source_sha256=processed_protocol["processed_listens_sha256"])
    catalog = pl.read_parquet(_resolve(project_root, cfg["catalog"])).sort("token_id")
    item_ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    tokens = np.asarray(catalog["token_id"].to_list(), dtype=np.int64)
    if len(set(item_ids.tolist())) != len(item_ids) or not np.array_equal(tokens, np.arange(3, 3 + len(tokens))):
        raise ValueError("RM catalog mapping is not one-to-one and contiguous")
    mapping_sha = hashlib.sha256("".join(f"{raw}\t{token}\n" for raw, token in zip(item_ids, tokens)).encode("ascii")).hexdigest()
    raw_to_index = {int(raw): i for i, raw in enumerate(item_ids.tolist())}
    params = cfg["model"]
    max_fit = int(params["max_train_events_per_user"])
    max_audit = int(params["max_audit_events_per_user"])
    fraction = float(params["prior_fraction_of_f_rm"])
    if not 0.5 <= fraction < 1.0 or min(max_fit, max_audit) < 1:
        raise ValueError("invalid RM temporal split or per-user event cap")

    runtime_started = time.perf_counter()
    timestamps = _group_event_timestamps(source_path, store)
    f_rm_start = min(float(t[0]) for t in timestamps.values() if len(t))
    f_rm_end = float(boundaries["rm_validation_start_raw"])
    fit_feature_cutoff = f_rm_start + fraction * (f_rm_end - f_rm_start)
    fit_selected = _sample_positions(
        store, timestamps, partition="fit", positions_name="rm_fit_positions",
        lower=fit_feature_cutoff, upper=f_rm_end, max_events=max_fit, seed=int(params["seed"]),
    )
    audit_start = float(boundaries["rm_validation_start_raw"])
    audit_end = float(boundaries["rl_anchor_start_raw"])
    audit_selected = _sample_positions(
        store, timestamps, partition="audit", positions_name="rm_validation_positions",
        lower=audit_start, upper=audit_end, max_events=max_audit, seed=int(params["seed"]) + 1,
    )
    if len(fit_selected) < int(params["minimum_fit_rows"]) or len(audit_selected) < int(params["minimum_audit_rows"]):
        raise RuntimeError(f"insufficient temporal RM rows: fit={len(fit_selected)}, audit={len(audit_selected)}")
    prior_fit_started = time.perf_counter()
    fit_priors = build_item_priors(source_path, catalog, cutoff_exclusive=fit_feature_cutoff, source_split="F_RM_before_fit_targets")
    policy_priors = build_item_priors(source_path, catalog, cutoff_exclusive=f_rm_end, source_split="complete_F_RM_before_V_RM_and_A_RL")
    prior_seconds = time.perf_counter() - prior_fit_started
    fit = _rm_rows(store, fit_selected, fit_priors, raw_to_index)
    audit = _rm_rows(store, audit_selected, policy_priors, raw_to_index)
    if fit["x"].shape[1] != len(RM_FEATURE_NAMES) or audit["x"].shape[1] != len(RM_FEATURE_NAMES):
        raise AssertionError("RM feature schema mismatch")
    if not np.isfinite(fit["x"]).all() or not np.isfinite(audit["x"]).all():
        raise ValueError("RM training/evaluation feature matrix contains non-finite values")

    context_end = len(CONTEXT_FEATURE_NAMES)
    item_features = fit["x"][:, context_end:]
    audit_item_features = audit["x"][:, context_end:]
    history_x, history_eval = fit["x"][:, :context_end], audit["x"][:, :context_end]
    item_cols = [i for i, name in enumerate(CANDIDATE_FEATURE_NAMES) if name in {
        "train_popularity_count", "train_ratio_prior_mean", "train_ratio_prior_count",
        "train_ratio_prior_missing", "train_duration_seconds", "train_duration_missing",
    }]
    item_x, item_eval = item_features[:, item_cols], audit_item_features[:, item_cols]
    smooth_mean_col = CANDIDATE_FEATURE_NAMES.index("train_ratio_prior_mean")
    smooth_count_col = CANDIDATE_FEATURE_NAMES.index("train_ratio_prior_count")
    early_ratio = 0.5
    item_mean = item_features[:, smooth_mean_col]
    item_count = item_features[:, smooth_count_col]
    item_prediction_fit = (item_mean * item_count + 10 * early_ratio) / (item_count + 10)
    item_eval_mean = audit_item_features[:, smooth_mean_col]
    item_eval_count = audit_item_features[:, smooth_count_col]
    item_prediction = (item_eval_mean * item_eval_count + 10 * early_ratio) / (item_eval_count + 10)

    seed = int(params["seed"])
    models: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    fit_started = time.perf_counter()
    history_model = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=160, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=seed)
    history_model.fit(history_x, fit["y"])
    fitted["history_only"] = history_model
    item_model = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=160, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=seed + 1)
    item_model.fit(item_x, fit["y"])
    fitted["item_metadata"] = item_model
    additive = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    additive.fit(fit["x"], fit["y"])
    fitted["additive_ridge"] = additive
    optimize = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=220, max_leaf_nodes=23,
        min_samples_leaf=35, l2_regularization=1.0, early_stopping=False, random_state=seed + 2)
    optimize.fit(fit["x"], fit["y"])
    fitted["optimized_rm"] = optimize
    audit_model = ExtraTreesRegressor(n_estimators=180, min_samples_leaf=8, max_features=0.85,
        n_jobs=-1, random_state=seed + 3)
    audit_model.fit(fit["x"], fit["y"])
    fitted["external_audit_predictor"] = audit_model
    fit_seconds = time.perf_counter() - fit_started

    predictions = {
        "history_only": np.clip(history_model.predict(history_eval), 0, 1),
        "item_metadata_model": np.clip(item_model.predict(item_eval), 0, 1),
        "item_smoothed_prior": np.clip(item_prediction, 0, 1),
        "additive_history_plus_item": np.clip(additive.predict(audit["x"]), 0, 1),
        "optimized_history_candidate_rm": np.clip(optimize.predict(audit["x"]), 0, 1),
        "external_audit_predictor": np.clip(audit_model.predict(audit["x"]), 0, 1),
    }
    metric_table = {name: _regression_metrics(audit["y"], prediction, audit["uid"])
                    for name, prediction in predictions.items()}
    model_groups: dict[str, Any] = {}
    support_counts = np.bincount(fit["item_index"], minlength=len(item_ids)).astype(np.int64)
    group_specs = {
        "candidate_novelty": np.where(audit["is_repeat"], "personal_repeat", "personal_new"),
        "positive_label": np.where(audit["positive"], "positive", "nonpositive"),
        "rm_fit_support": _support_groups(audit["item_index"], support_counts),
    }
    for dimension, labels in group_specs.items():
        model_groups[dimension] = {}
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            if int(mask.sum()) < 20:
                continue
            model_groups[dimension][str(label)] = {
                name: _regression_metrics(audit["y"][mask], pred[mask], audit["uid"][mask])
                for name, pred in predictions.items()
            }
    added_value = {}
    base_user = {int(k): -float(v) for k, v in metric_table["history_only"]["per_user_mae"].items()}
    for name in ("item_metadata_model", "item_smoothed_prior", "additive_history_plus_item",
                 "optimized_history_candidate_rm", "external_audit_predictor"):
        candidate_user = {int(k): -float(v) for k, v in metric_table[name]["per_user_mae"].items()}
        added_value[name] = paired_user_bootstrap(
            candidate_user, base_user, n_bootstrap=2000, seed=seed + 17000 + len(added_value),
        )

    output_root.mkdir(parents=True, exist_ok=False)
    models_dir = output_root / "models"
    models_dir.mkdir()
    model_meta: dict[str, Any] = {}
    for name, model in fitted.items():
        model_path = models_dir / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)
        model_meta[name] = {"path": str(model_path.relative_to(project_root)), "sha256": sha256_file(model_path),
                            "role": "optimized reward prediction" if name == "optimized_rm" else
                                    "external audit prediction only" if name == "external_audit_predictor" else "diagnostic baseline"}
    fit_prior_path = output_root / "priors_for_rm_fit.npz"
    policy_prior_path = output_root / "priors_for_policy_cache.npz"
    fit_prior_sha = save_priors(fit_prior_path, fit_priors, mapping_sha)
    policy_prior_sha = save_priors(policy_prior_path, policy_priors, mapping_sha)
    reward_scale = {"mean": float(fit["y"].mean()), "std": float(fit["y"].std()),
                    "std_floor": 1e-3, "version": "residual_support_f_rm_fit_sample_ratio_z_v1",
                    "fit_target_rows": int(len(fit["y"])), "fit_target_users": int(len(set(fit["uid"].tolist()))),
                    "target_range": [float(fit["y"].min()), float(fit["y"].max())],
                    "fit_feature_cutoff_raw_exclusive": float(fit_feature_cutoff),
                    "source_rows_sha256": processed_protocol["processed_listens_sha256"]}
    scale_path = output_root / "reward_scale.json"
    scale_path.write_text(json.dumps(reward_scale, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    groups_by_catalog = {
        "unsupported": int(np.count_nonzero(support_counts == 0)),
        "rare_1_9": int(np.count_nonzero((support_counts >= 1) & (support_counts < 10))),
        "rare_10_99": int(np.count_nonzero((support_counts >= 10) & (support_counts < 100))),
        "supported_100_plus": int(np.count_nonzero(support_counts >= 100)),
    }
    result = {
        "schema_version": 1, "protocol_id": cfg["protocol_id"], "status": "completed",
        "evaluation_role": "RM development audit on held-out users and a later temporal block; retrospective",
        "optimized_rm_role": "logged observed-ratio predictor used to generate policy rewards; not ground truth",
        "external_audit_role": "separately fitted ExtraTrees predictor; diagnostics only, never used for beta/checkpoint selection",
        "prior_provenance": {
            "fit_targets_time_start": float(fit_feature_cutoff), "fit_targets_time_end_exclusive": float(f_rm_end),
            "fit_feature_cutoff_exclusive": float(fit_feature_cutoff),
            "fit_prior_source": "all known, non-organic, unambiguous F_RM events strictly earlier than every selected RM fit target",
            "policy_prior_cutoff_exclusive": float(f_rm_end),
            "policy_prior_source": "all known, non-organic, unambiguous F_RM events, before V_RM and A_RL",
            "target_ratio_in_prior": False,
            "prior_change_test": "feature priors are computed on rows with timestamp < cutoff; later targets cannot enter the aggregation query",
            "embedding_source": "not used by RM fit; frozen full_model strong actor embeddings are used only for the separate policy candidate-history interaction feature",
        },
        "catalog": {"size": int(len(item_ids)), "raw_token_mapping_sha256": mapping_sha,
                    "processed_listens_sha256": processed_protocol["processed_listens_sha256"]},
        "fit_sample": {"role": "fit users, late F_RM targets", "users": int(len(set(fit["uid"].tolist()))),
                       "rows": int(len(fit["y"])), "max_events_per_user": max_fit,
                       "seed": seed, "selection": "stable SHA256 per user/target, fixed before reading outcomes"},
        "audit_sample": {"role": "disjoint audit users, later V_RM targets", "users": int(len(set(audit["uid"].tolist()))),
                         "rows": int(len(audit["y"])), "max_events_per_user": max_audit,
                         "seed": seed + 1, "selection": "stable SHA256 per user/target, fixed before reading outcomes"},
        "feature_schema": {"version": "residual_support_candidate_action_features_v1", "names": list(RM_FEATURE_NAMES),
                           "n_context_features": len(CONTEXT_FEATURE_NAMES), "n_candidate_features": len(CANDIDATE_FEATURE_NAMES),
                           "personal_repeat_count_distinct_from_global_popularity": True,
                           "train_popularity_distinct_from_rm_fit_sample_count": True},
        "models": model_meta,
        "item_prior_artifacts": {
            "fit_path": str(fit_prior_path.relative_to(project_root)), "fit_sha256": fit_prior_sha,
            "policy_path": str(policy_prior_path.relative_to(project_root)), "policy_sha256": policy_prior_sha,
        },
        "reward_scale": {**reward_scale, "sha256": sha256_file(scale_path)},
        "audit_metrics": metric_table,
        "slices": model_groups,
        "added_value_over_history_only": added_value,
        "rm_fit_support_catalog_counts": groups_by_catalog,
        "rm_fit_item_count": {"n_observations": int(support_counts.sum()), "max": int(support_counts.max()),
                               "counts_by_catalog_item": support_counts.tolist()},
        "runtime": {"timestamp_index_and_sample_and_features_seconds": time.perf_counter() - runtime_started,
                    "prior_aggregation_seconds": prior_seconds, "model_fit_seconds": fit_seconds},
        "library_versions": {"scikit_learn": sklearn.__version__, "polars": pl.__version__},
        "limitations": [
            "The outcome data was inspected in V1-full_model. Disjoint users and temporal blocks improve role separation but do not create an untouched dataset.",
            "All RM comparisons are on logged actions; a lower error does not establish counterfactual ranking quality.",
            "The external audit predictor is separate from the optimized RM but shares the same permitted fit sample and is not ground truth.",
            "The policy feature extractor uses a pretrained strong actor. Its relationship to A_RL event exposure is disclosed separately.",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result_path = output_root / "rm_diagnostics.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output_root / "rm_diagnostics.csv").write_text(
        "model,mae,rmse,n_rows,n_users\n" + "".join(
            f"{name},{metric['mae']:.9f},{metric['rmse']:.9f},{metric['n_rows']},{metric['n_users']}\n"
            for name, metric in metric_table.items()), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit corrected panel_tools.1 reward and separate audit predictors")
    parser.add_argument("--config", default="configs/residual_support/rm.yaml")
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args(argv)
    result = run_rm_experiment(args.config, args.workspace)
    print(json.dumps({"status": result["status"], "fit_sample": result["fit_sample"],
                      "audit_sample": result["audit_sample"], "audit_metrics": {
                          key: {k: v for k, v in metric.items() if k != "per_user_mae"}
                          for key, metric in result["audit_metrics"].items()}, "runtime": result["runtime"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
