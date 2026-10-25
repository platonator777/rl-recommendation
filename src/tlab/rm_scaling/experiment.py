from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import polars as pl
import torch
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor

from tlab.data.store import SequenceStore
from tlab.residual_support.features import (RM_FEATURE_NAMES, candidate_features,
                                history_context_features, rm_feature_row)
from tlab.residual_support.objectives import FrozenRewardScale, exact_support_statistics, sampled_loo_loss
from tlab.residual_support.rm import _group_event_timestamps, _rm_rows, _sample_positions, load_priors
from tlab.residual_support.scaler import TrainOnlyFeatureScaler
from tlab.policy_diagnostics.c_rm import _affinity_rows, _hash_bucket as c_hash_bucket
from tlab.frozen_policy.data import (
    Frozen_FEATURE_NAMES, _candidate_affinity_vectorized, _load_cache, analytic_q, evaluate_conditional,
    model_reward_metrics, open_v5_cache, topk_stable,
)
from tlab.frozen_policy.reporting import (
    _build_24h_targets, _event_timelines, _paired_ci, _reporting_indices,
)
from tlab.frozen_policy.training import FrozenPolicyHead, _device, _load_existing_cache, _predict_conditional


ROOT = __import__('tlab.paths', fromlist=['workspace_root']).workspace_root()
ART = ROOT / "artifacts/rm_scaling"
FINAL = ROOT / "final/rm_scaling"
PROTOCOL_PATH = ART / "protocol.json"
LOWER = 18501460.0
UPPER = 21766405.0
FIT_TARGET = 100_000
MAX_PER_USER = 40
SEEDS = (17, 42, 73)
BETA = 3.0
REWARD_INDEX = {"small_base": 0, "small_affinity": 1, "large_base": 2,
                "large_affinity": 3, "old_deployed": 4}
BRANCHES = {
    "S": ("small_affinity", "sampled_loo", False),
    "L": ("large_affinity", "sampled_loo", False),
    "E": ("large_affinity", "exact_expected_reward", False),
    "A": ("large_affinity", "sampled_loo", True),
}
AFFINITY_COLUMNS = tuple(range(45, 54))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None or (isinstance(v, float) and not math.isfinite(v)) else v)
                             for k, v in row.items()})


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_rows(rows: Sequence[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for uid, position in rows:
        digest.update(f"{int(uid)}:{int(position)}\n".encode("ascii"))
    return digest.hexdigest()


def _ensure_dirs() -> None:
    for path in (ART, ART / "rm", ART / "cache", ART / "policy", FINAL):
        path.mkdir(parents=True, exist_ok=True)


def _load_runtime() -> tuple[dict[str, Any], SequenceStore, pl.DataFrame, dict[int, np.ndarray], dict[int, int], np.ndarray]:
    source = ROOT / "data/processed_full_model_tminus14"
    source_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(source / "sequence_index",
                                     expected_source_sha256=source_protocol["processed_listens_sha256"])
    catalog = pl.read_parquet(source / "catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    raw_to_index = {int(raw): i for i, raw in enumerate(item_ids.tolist())}
    timestamps = _group_event_timestamps(source / "listens.parquet", store)
    return source_protocol, store, catalog, timestamps, raw_to_index, item_ids


def _valid_rm_position(store: SequenceStore, user_index: int, position: int,
                       timeline: np.ndarray, raw_to_index: dict[int, int]) -> bool:
    user = store.users[user_index]
    if position >= len(timeline) or not (LOWER <= float(timeline[position]) < UPPER):
        return False
    raw = int(user.raw_item_ids[position])
    ratio = float(user.ratio_raw[position])
    return raw in raw_to_index and math.isfinite(ratio)


def _rows_to_csv(path: Path, rows: Sequence[tuple[int, int]], store: SequenceStore,
                 timestamps: dict[int, np.ndarray], split: str) -> None:
    records = []
    for user_index, position in rows:
        user = store.users[user_index]
        records.append({"uid": int(user.uid), "position": int(position),
                        "event_key": f"{int(user.uid)}:{int(position)}",
                        "timestamp_raw": float(timestamps[int(user.uid)][position]),
                        "raw_item_id": int(user.raw_item_ids[position]),
                        "ratio_raw": float(user.ratio_raw[position]),
                        "split": split})
    _write_csv(path, records)


def _construct_nested_rows(store: SequenceStore, timestamps: dict[int, np.ndarray],
                           raw_to_index: dict[int, int], rm_manifest: dict[str, Any]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[str, Any]]:
    original = _sample_positions(store, timestamps, partition="fit", positions_name="rm_fit_positions",
                                 lower=LOWER, upper=UPPER, max_events=5, seed=17)
    original = [(ui, pos) for ui, pos in original
                if _valid_rm_position(store, ui, pos, timestamps[int(store.users[ui].uid)], raw_to_index)]
    fit_rows = [(ui, pos) for ui, pos in original
                if c_hash_bucket(int(store.users[ui].uid)) != 0]
    dev_rows = [(ui, pos) for ui, pos in original
                if c_hash_bucket(int(store.users[ui].uid)) == 0]
    if len(fit_rows) != int(rm_manifest["train_rows"]) or len(dev_rows) != int(rm_manifest["rmdev_rows"]):
        raise RuntimeError(f"C RM reconstruction mismatch: fit={len(fit_rows)} dev={len(dev_rows)}")

    fit_user_to_ix: dict[int, int] = {}
    original_by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for ui, pos in fit_rows:
        uid = int(store.users[ui].uid)
        fit_user_to_ix[uid] = ui
        original_by_user[uid].append((ui, pos))
    if len(fit_user_to_ix) != int(rm_manifest["train_users"]):
        raise RuntimeError("C RM original fit-user count did not reproduce")

    candidates_by_user: dict[int, list[tuple[int, int]]] = {}
    for uid, ui in fit_user_to_ix.items():
        user = store.users[ui]
        timeline = timestamps[uid]
        eligible = [(ui, int(pos)) for pos in user.rm_fit_positions
                    if _valid_rm_position(store, ui, int(pos), timeline, raw_to_index)]
        eligible.sort(key=lambda pair: hashlib.sha256(
            f"{bytes.fromhex('76345f313a726d3a31373a').decode('ascii')}{uid}:{pair[1]}".encode()).digest())
        existing = {(u, p) for u, p in original_by_user[uid]}
        candidates_by_user[uid] = [pair for pair in eligible if pair not in existing]

    user_order = sorted(fit_user_to_ix,
                        key=lambda uid: hashlib.sha256(f"{bytes.fromhex('76365f66696e616c3a726d2d657870616e643a').decode('ascii')}{uid}".encode()).digest())
    selected = list(fit_rows)
    seen = {(ui, pos) for ui, pos in selected}
    used_per_user = {uid: len(rows) for uid, rows in original_by_user.items()}
    cursor = {uid: 0 for uid in user_order}
    while len(selected) < FIT_TARGET:
        progressed = False
        for uid in user_order:
            if used_per_user[uid] >= MAX_PER_USER:
                continue
            pool = candidates_by_user[uid]
            while cursor[uid] < len(pool) and pool[cursor[uid]] in seen:
                cursor[uid] += 1
            if cursor[uid] >= len(pool):
                continue
            row = pool[cursor[uid]]
            cursor[uid] += 1
            if row in seen:
                continue
            selected.append(row); seen.add(row); used_per_user[uid] += 1
            progressed = True
            if len(selected) >= FIT_TARGET:
                break
        if not progressed:
            break
    original_keys = {(int(store.users[ui].uid), pos) for ui, pos in fit_rows}
    selected_keys = {(int(store.users[ui].uid), pos) for ui, pos in selected}
    if not original_keys <= selected_keys:
        raise AssertionError("expanded RM fit does not contain every original fit row")
    if len(selected_keys) != len(selected) or max(used_per_user.values()) > MAX_PER_USER:
        raise AssertionError("expanded RM fit contains duplicates or exceeds the per-user cap")
    if set(fit_user_to_ix) & {int(store.users[ui].uid) for ui, _ in dev_rows}:
        raise AssertionError("RM dev users leaked into expanded RM fit")
    details = {
        "small_fit_rows": len(fit_rows), "small_fit_users": len(fit_user_to_ix),
        "rm_dev_rows": len(dev_rows), "rm_dev_users": len({int(store.users[ui].uid) for ui, _ in dev_rows}),
        "large_fit_rows": len(selected), "large_fit_users": len({int(store.users[ui].uid) for ui, _ in selected}),
        "nested_row_key_sha256": _sha_rows([(int(store.users[ui].uid), pos) for ui, pos in selected]),
        "small_fit_row_key_sha256": _sha_rows([(int(store.users[ui].uid), pos) for ui, pos in fit_rows]),
        "dev_row_key_sha256": _sha_rows([(int(store.users[ui].uid), pos) for ui, pos in dev_rows]),
        "max_rows_per_user": max(used_per_user.values()),
        "per_user_count_min": min(used_per_user.values()),
        "per_user_count_mean": float(np.mean(list(used_per_user.values()))),
        "per_user_count_max": max(used_per_user.values()),
        "all_original_fit_rows_in_large_fit": True,
        "no_duplicate_row_keys": True,
        "rm_dev_user_overlap": 0,
        "timestamp_bounds": [LOWER, UPPER],
    }
    return selected, dev_rows, details


def _build_features(store: SequenceStore, selected: list[tuple[int, int]],
                    priors: Any, raw_to_index: dict[int, int], embedding: np.ndarray) -> dict[str, np.ndarray]:
    base = _rm_rows(store, selected, priors, raw_to_index)
    affinity = _affinity_rows(store, selected, base["item_index"], embedding)
    x_affinity = np.column_stack((base["x"], affinity)).astype(np.float32, copy=False)
    if base["x"].shape[1] != len(RM_FEATURE_NAMES) or affinity.shape[1] != 9:
        raise ValueError("RM feature matrix schema mismatch")
    if not np.isfinite(base["x"]).all() or not np.isfinite(x_affinity).all():
        raise FloatingPointError("non-finite RM features")
    return {**base, "affinity": affinity, "x_affinity": x_affinity}


def _canary_suffix_mutation(store: SequenceStore, selected: list[tuple[int, int]],
                            priors: Any, raw_to_index: dict[int, int], embedding: np.ndarray) -> dict[str, Any]:
    from types import SimpleNamespace
    tested = 0
    for ui, position in selected[:8]:
        user = store.users[ui]
        raw_id = int(user.raw_item_ids[position])
        item_ix = raw_to_index[raw_id]
        original_base = rm_feature_row(user, position, item_ix, raw_id, priors)
        original_aff = _affinity_rows(store, [(ui, position)], np.asarray([item_ix]), embedding)[0]
        mutated = SimpleNamespace(tokens=np.asarray(user.tokens).copy(),
                                  raw_item_ids=np.asarray(user.raw_item_ids).copy(),
                                  ratio_raw=np.asarray(user.ratio_raw).copy())
        if position < len(mutated.raw_item_ids):
            mutated.ratio_raw[position] = 1.0 - np.nan_to_num(mutated.ratio_raw[position], nan=0.0)
        if position + 1 < len(mutated.raw_item_ids):
            mutated.raw_item_ids[position + 1:] = np.roll(mutated.raw_item_ids[position + 1:], 1)
            mutated.tokens[position + 1:] = np.roll(mutated.tokens[position + 1:], 1)
            mutated.ratio_raw[position + 1:] = np.roll(mutated.ratio_raw[position + 1:], 1)
        after_base = rm_feature_row(mutated, position, item_ix, raw_id, priors)
        fake_store = SimpleNamespace(users=[mutated])
        after_aff = _affinity_rows(fake_store, [(0, position)], np.asarray([item_ix]), embedding)[0]
        if not np.array_equal(original_base, after_base) or not np.array_equal(original_aff, after_aff):
            raise AssertionError("suffix mutation changed a prefix-only RM feature")
        tested += 1
    return {"status": "passed", "rows": tested,
            "mutation": "target feedback plus sequence suffix permuted; prefix RM and affinity features unchanged"}


def _user_error_maps(y: np.ndarray, pred: np.ndarray, uid: np.ndarray) -> tuple[dict[int, float], dict[int, float]]:
    groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for label, estimate, user in zip(y.tolist(), pred.tolist(), uid.tolist()):
        groups[int(user)].append((float(label), float(estimate)))
    mae, rmse = {}, {}
    for user, pairs in groups.items():
        target = np.asarray([p[0] for p in pairs], dtype=np.float64)
        estimate = np.asarray([p[1] for p in pairs], dtype=np.float64)
        mae[user] = float(np.mean(np.abs(target - estimate)))
        rmse[user] = float(np.sqrt(np.mean((target - estimate) ** 2)))
    return mae, rmse


def _paired_bootstrap(left: dict[int, float], right: dict[int, float], seed: int,
                      draws: int = 2000, levels: tuple[float, ...] = (0.95,)) -> dict[str, Any]:
    users = sorted(set(left) & set(right))
    if not users:
        raise ValueError("paired bootstrap requires shared users")
    delta = np.asarray([left[u] - right[u] for u in users], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 128):
        stop = min(draws, start + 128)
        means[start:stop] = delta[rng.integers(0, len(delta), size=(stop - start, len(delta)))].mean(axis=1)
    result: dict[str, Any] = {"mean_difference": float(delta.mean()), "n_users": len(users),
                              "bootstrap_draws": draws, "seed": seed}
    for confidence in levels:
        tail = (1.0 - confidence) / 2.0
        result[f"ci{int(confidence*1000)/10:g}_low"] = float(np.quantile(means, tail))
        result[f"ci{int(confidence*1000)/10:g}_high"] = float(np.quantile(means, 1.0 - tail))
    return result


def prepare_rm() -> dict[str, Any]:
    _ensure_dirs()
    stage_started = time.perf_counter()
    source_protocol, store, catalog, timestamps, raw_to_index, item_ids = _load_runtime()
    rm_manifest_path = ROOT / "artifacts/policy_diagnostics/rm/C_split_manifest.json"
    rm_manifest = json.loads(rm_manifest_path.read_text(encoding="utf-8"))
    selected, dev_selected, selection = _construct_nested_rows(store, timestamps, raw_to_index, rm_manifest)
    # Restore the exact small-fit order from the original C selection.
    original_all = _sample_positions(store, timestamps, partition="fit", positions_name="rm_fit_positions",
                                     lower=LOWER, upper=UPPER, max_events=5, seed=17)
    fit_small = [(ui, pos) for ui, pos in original_all
                 if c_hash_bucket(int(store.users[ui].uid)) != 0
                 and _valid_rm_position(store, ui, pos, timestamps[int(store.users[ui].uid)], raw_to_index)]
    if len(fit_small) != selection["small_fit_rows"]:
        raise AssertionError("small fit reconstruction changed while sorting nested data")

    priors_meta = json.loads((ROOT / "artifacts/frozen_policy/cache/feature_cache_manifest.json").read_text(encoding="utf-8"))
    policy_train_min_time = float(priors_meta["train_min_target_time"])
    if policy_train_min_time <= UPPER:
        raise RuntimeError(f"expanded RM upper bound {UPPER} is not before policy/evaluation start {policy_train_min_time}")
    priors = load_priors(ROOT / "artifacts/frozen_policy/cache/c_priors_reconstructed.npz",
                         expected_mapping_sha256=priors_meta["raw_token_mapping_sha256"])
    if priors.cutoff_raw_exclusive != LOWER:
        raise AssertionError("C priors do not use the frozen earliest-target cutoff")
    embedding_path = ROOT / "artifacts/policy_diagnostics/rm/prefix_affinity_embeddings.npy"
    if _sha_file(embedding_path) != rm_manifest["embedding"]["embedding_sha256"]:
        raise AssertionError("prefix affinity embedding hash differs from C manifest")
    embedding = np.load(embedding_path, mmap_mode="r")
    if embedding.shape != (len(item_ids), 64):
        raise AssertionError("prefix embedding does not match frozen 20k catalog")

    new_rows = selected[len(fit_small):]
    pilot_rows = new_rows[:min(5000, len(new_rows))]
    pilot_started = time.perf_counter()
    _ = _build_features(store, pilot_rows, priors, raw_to_index, embedding)
    pilot_seconds = time.perf_counter() - pilot_started
    predicted_full_feature_seconds = pilot_seconds * (len(new_rows) / max(len(pilot_rows), 1))
    details = {**selection,
               "expanded_fit_includes_exact_original_fit": True,
               "new_added_rows": len(new_rows),
               "feature_pilot_rows": len(pilot_rows),
               "feature_pilot_seconds": round(pilot_seconds, 3),
               "linear_feature_prep_projection_seconds": round(predicted_full_feature_seconds, 3),
               "pilot_projection_note": "measured on first 5000 new rows; full expansion then explicitly timed",
               "source_cutoff": source_protocol["boundaries_source_units"].get("rm_validation_start_raw"),
               "policy_train_min_target_time": policy_train_min_time,
               "all_fit_targets_strictly_before_policy_train": True,
               "feature_prior_cutoff_exclusive": priors.cutoff_raw_exclusive,
               "prefix_embedding_sha256": rm_manifest["embedding"]["embedding_sha256"]}
    if len(new_rows) < 5000:
        details["feature_pilot_status"] = "fewer than 5000 new rows were available; timed all available rows"
    if predicted_full_feature_seconds > 7200:
        # The protocol allows a pre-fit cap reduction only if the budget projection is unsafe.
        target_size = 50_000
        if len(selected) > target_size:
            selected = selected[:target_size]
            details["large_fit_rows"] = len(selected)
            details["new_added_rows"] = len(selected) - len(fit_small)
            details["budget_reduction"] = "reduced expanded fit to 50000 nested rows before either large RM fit"
            details["nested_row_key_sha256"] = _sha_rows([(int(store.users[ui].uid), pos) for ui, pos in selected])

    _rows_to_csv(ART / "rm/large_fit_rows.csv", selected, store, timestamps, "expanded_fit")
    _rows_to_csv(ART / "rm/small_fit_rows.csv", fit_small, store, timestamps, "small_fit")
    _rows_to_csv(ART / "rm/rm_dev_rows.csv", dev_selected, store, timestamps, "rm_dev")
    panel_manifest = json.loads((ROOT / "artifacts/frozen_policy/split_manifest.json").read_text(encoding="utf-8"))
    _write_json(ART / "policy_panel_ids.json", {"development_uids": panel_manifest["development_uids"],
                                                  "reporting_uids": panel_manifest["reporting_uids"],
                                                  "development_event_ids": panel_manifest["development_event_ids"],
                                                  "reporting_event_ids": panel_manifest["reporting_event_ids"],
                                                  "source": "copied verbatim from Frozen split manifest"})

    feature_started = time.perf_counter()
    large = _build_features(store, selected, priors, raw_to_index, embedding)
    dev = _build_features(store, dev_selected, priors, raw_to_index, embedding)
    feature_seconds = time.perf_counter() - feature_started
    np.savez_compressed(ART / "rm/large_fit_features.npz", x=large["x"], x_affinity=large["x_affinity"],
                        y=large["y"], uid=large["uid"], item_index=large["item_index"],
                        is_repeat=large["is_repeat"], positive=large["positive"])
    np.savez_compressed(ART / "rm/rm_dev_features.npz", x=dev["x"], x_affinity=dev["x_affinity"],
                        y=dev["y"], uid=dev["uid"], item_index=dev["item_index"],
                        is_repeat=dev["is_repeat"], positive=dev["positive"])
    canary = _canary_suffix_mutation(store, dev_selected, priors, raw_to_index, embedding)

    old = joblib.load(ROOT / "artifacts/policy_diagnostics/rm/C_refit_residual_support_optimized_control.joblib")
    old_affinity = joblib.load(ROOT / "artifacts/policy_diagnostics/rm/C_prefix_safe_affinity_extension.joblib")
    large_base_path = ART / "rm/large_base.joblib"
    large_affinity_path = ART / "rm/large_affinity.joblib"
    hgb_args = dict(learning_rate=.05, max_iter=220, max_leaf_nodes=23, min_samples_leaf=35,
                    l2_regularization=1.0, early_stopping=False, random_state=19)
    fit_started = time.perf_counter()
    large_base = HistGradientBoostingRegressor(**hgb_args).fit(large["x"], large["y"])
    joblib.dump(large_base, large_base_path)
    base_fit_seconds = time.perf_counter() - fit_started
    affinity_started = time.perf_counter()
    large_affinity = HistGradientBoostingRegressor(**hgb_args).fit(large["x_affinity"], large["y"])
    joblib.dump(large_affinity, large_affinity_path)
    affinity_fit_seconds = time.perf_counter() - affinity_started

    models = {"small_base": old, "small_affinity": old_affinity,
              "large_base": large_base, "large_affinity": large_affinity}
    dev_counts: dict[str, np.ndarray] = {}
    fit_sets = {"small_base": fit_small, "small_affinity": fit_small,
                "large_base": selected, "large_affinity": selected}
    for name, rows in fit_sets.items():
        idx = np.asarray([raw_to_index[int(store.users[ui].raw_item_ids[pos])] for ui, pos in rows], dtype=np.int32)
        dev_counts[name] = np.bincount(idx, minlength=len(item_ids))
    predictions = {name: np.clip((model.predict(dev["x"] if name.endswith("base") else dev["x_affinity"])), 0.0, 1.0)
                   for name, model in models.items()}
    metric_rows: list[dict[str, Any]] = []
    user_mae: dict[str, dict[int, float]] = {}
    user_rmse: dict[str, dict[int, float]] = {}
    y, uid, repeat, items = dev["y"], dev["uid"], dev["is_repeat"], dev["item_index"]
    for name, pred in predictions.items():
        mae_map, rmse_map = _user_error_maps(y, pred, uid)
        user_mae[name], user_rmse[name] = mae_map, rmse_map
        support = dev_counts[name][items]
        groups = {"all": np.ones(len(y), dtype=bool), "personal_new": ~repeat,
                  "personal_repeat": repeat, "unsupported_item": support == 0,
                  "rare_1_9": (support >= 1) & (support < 10),
                  "rare_10_99": (support >= 10) & (support < 100),
                  "supported_100_plus": support >= 100}
        for group, mask in groups.items():
            if not mask.any():
                continue
            gmae, grmse = _user_error_maps(y[mask], pred[mask], uid[mask])
            metric_rows.append({"model": name, "slice": group, "n_rows": int(mask.sum()),
                                "n_users": len(gmae), "mae": float(np.mean(np.abs(y[mask] - pred[mask]))),
                                "rmse": float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2))),
                                "user_macro_mae": float(np.mean(list(gmae.values()))),
                                "user_macro_rmse": float(np.mean(list(grmse.values())))})
    _write_csv(ART / "rm/rm_dev_metrics.csv", metric_rows)
    pred_rows = []
    for i in range(len(y)):
        pred_rows.append({"uid": int(uid[i]), "position": int(dev_selected[i][1]),
                          "y": float(y[i]), "personal_repeat": bool(repeat[i]),
                          "item_index": int(items[i]), "fit_support_small": int(dev_counts["small_affinity"][items[i]]),
                          "fit_support_large": int(dev_counts["large_affinity"][items[i]]),
                          **{f"pred_{name}": float(pred[i]) for name, pred in predictions.items()}})
    _write_csv(ART / "rm/rm_dev_predictions.csv", pred_rows)
    rm_contrasts = []
    for left, right, seed in (("large_affinity", "small_affinity", 63017),
                              ("large_affinity", "large_base", 63019)):
        mae_ci = _paired_bootstrap(user_mae[left], user_mae[right], seed)
        rm_contrasts.append({"contrast": f"{left} - {right}", "metric": "user_macro_mae",
                             **mae_ci, "interpretation": "negative favors left model"})
    _write_csv(ART / "rm/rm_dev_contrasts.csv", rm_contrasts)
    fit_rows_manifest = {
        **details,
        "feature_schema_base": list(RM_FEATURE_NAMES),
        "affinity_names": ["cosine_last", "cosine_mean_last20", "cosine_positive_history",
                           "cosine_negative_history", "last_history_missing", "history20_missing",
                           "positive_history_missing", "negative_history_missing", "candidate_embedding_missing"],
        "c_priors_source": "artifacts/frozen_policy/cache/c_priors_reconstructed.npz",
        "c_priors_sha256": _sha_file(ROOT / "artifacts/frozen_policy/cache/c_priors_reconstructed.npz"),
        "embedding_sha256": rm_manifest["embedding"]["embedding_sha256"],
        "feature_generation_seconds": round(feature_seconds, 3),
        "model_fit_seconds": {"large_base": round(base_fit_seconds, 3),
                              "large_affinity": round(affinity_fit_seconds, 3)},
        "large_base_model_sha256": _sha_file(large_base_path),
        "large_affinity_model_sha256": _sha_file(large_affinity_path),
        "fit_feature_file_sha256": _sha_file(ART / "rm/large_fit_features.npz"),
        "dev_feature_file_sha256": _sha_file(ART / "rm/rm_dev_features.npz"),
        "suffix_mutation_canary": canary,
        "hgb": hgb_args,
        "prediction_clip": [0.0, 1.0],
        "model_dev_metrics": metric_rows,
        "model_dev_contrasts": rm_contrasts,
        "stage_elapsed_seconds": round(time.perf_counter() - stage_started, 3),
        "source_protocol_sha256": _sha_file(ROOT / "data/processed_full_model_tminus14/protocol.json"),
        "v5_split_manifest_sha256": _sha_file(ROOT / "artifacts/frozen_policy/split_manifest.json"),
    }
    _write_json(ART / "rm/rm_summary.json", fit_rows_manifest)
    return fit_rows_manifest


def prepare_candidate_rewards() -> dict[str, Any]:
    _ensure_dirs()
    started = time.perf_counter()
    large_base_path, large_affinity_path = ART / "rm/large_base.joblib", ART / "rm/large_affinity.joblib"
    if not large_base_path.exists() or not large_affinity_path.exists():
        raise FileNotFoundError("both new Scaling RM fits must complete before candidate rewards are generated")
    large_base = joblib.load(large_base_path)
    large_affinity = joblib.load(large_affinity_path)
    cache_dir = ROOT / "artifacts/frozen_policy/cache"
    created = {}
    for split in ("train", "common"):
        feature_arrays, cache_manifest = open_v5_cache(cache_dir, split)
        _, old_arrays, _ = _load_existing_cache(ROOT / ("artifacts/residual_support/cache/train" if split == "train"
                                                        else "artifacts/policy_diagnostics/cache/common_open_6655"))
        features = feature_arrays["candidate_features"]
        old_rewards = feature_arrays["rm_rewards"]
        n, k, d = features.shape
        if (k, d) != (500, 54) or old_rewards.shape != (n, 3, 500):
            raise ValueError(f"unexpected Frozen {split} cache shapes: {features.shape}, {old_rewards.shape}")
        output_path = ART / "cache" / f"{split}_rewards.bin"
        output = np.memmap(output_path, dtype=np.float32, mode="w+", shape=(n, 5, k))
        output[:, 0, :] = old_rewards[:, 1, :]
        output[:, 1, :] = old_rewards[:, 2, :]
        output[:, 4, :] = old_rewards[:, 0, :]
        for row_start in range(0, n, 8):
            row_stop = min(row_start + 8, n)
            batch = np.asarray(features[row_start:row_stop], dtype=np.float32)
            base = batch[:, :, :35].reshape(-1, 35)
            affinity = batch[:, :, 45:54].reshape(-1, 9)
            pred_base = np.clip(large_base.predict(base), 0.0, 1.0).reshape(row_stop - row_start, k)
            pred_aff = np.clip(large_affinity.predict(np.column_stack((base, affinity))), 0.0, 1.0).reshape(row_stop - row_start, k)
            output[row_start:row_stop, 2, :] = pred_base.astype(np.float32)
            output[row_start:row_stop, 3, :] = pred_aff.astype(np.float32)
        output.flush()
        del output
        # Existing C predictors are also clipped in place for this Scaling policy protocol.
        checks = np.memmap(output_path, dtype=np.float32, mode="r", shape=(n, 5, k))
        if not np.isfinite(checks).all() or float(checks.min()) < -1e-7 or float(checks.max()) > 1.0000001:
            raise FloatingPointError(f"Scaling {split} reward cache has invalid values")
        created[split] = {"path": str(output_path.relative_to(ROOT)), "shape": [n, 5, k],
                          "sha256": _sha_file(output_path), "panel_contexts": n,
                          "source_v5_feature_cache": str((cache_dir / cache_manifest[split]["features_path"]).relative_to(ROOT)),
                          "raw_cache_contexts": len(old_arrays["support_indices"])}
        del checks

    # Logged-target and policy candidate extractors must agree on the same raw ID/history.
    source_protocol, store, catalog, _, raw_to_index, item_ids = _load_runtime()
    feature_manifest = json.loads((cache_dir / "feature_cache_manifest.json").read_text(encoding="utf-8"))
    priors = load_priors(cache_dir / "c_priors_reconstructed.npz",
                         expected_mapping_sha256=feature_manifest["raw_token_mapping_sha256"])
    embedding = np.load(ROOT / "artifacts/policy_diagnostics/rm/prefix_affinity_embeddings.npy", mmap_mode="r")
    alignment_rows = []
    for split in ("train", "common"):
        feature_arrays, _ = open_v5_cache(cache_dir, split)
        raw_path = ROOT / ("artifacts/residual_support/cache/train" if split == "train"
                           else "artifacts/policy_diagnostics/cache/common_open_6655")
        _, raw_arrays, panel = _load_cache(raw_path)
        indices_to_check = np.linspace(0, len(panel) - 1, num=min(4, len(panel)), dtype=np.int64)
        if split == "common":
            dev_uid_set = set(int(x) for x in json.loads((ROOT / "artifacts/frozen_policy/split_manifest.json").read_text(encoding="utf-8"))["development_uids"])
            dev_ix = [i for i, row in enumerate(panel) if int(row.uid) in dev_uid_set]
            report_ix = [i for i, row in enumerate(panel) if int(row.uid) not in dev_uid_set]
            indices_to_check = np.asarray((dev_ix[:2] + report_ix[:2]), dtype=np.int64)
        for row_ix in indices_to_check:
            target = panel[int(row_ix)]
            user = store.users[target.user_index]
            cand_ix = np.asarray(raw_arrays["support_indices"][row_ix], dtype=np.int64)
            raw_ids = item_ids[cand_ix]
            c_features = candidate_features(user, target.position, raw_ids,
                                            catalog_indices=cand_ix, priors=priors)
            ctx = history_context_features(user, target.position)
            expected_base = np.column_stack((np.repeat(ctx[None, :], len(cand_ix), axis=0), c_features))
            expected_aff = _candidate_affinity_vectorized(user, target.position, cand_ix, embedding)
            actual = np.asarray(feature_arrays["candidate_features"][row_ix], dtype=np.float32)
            max_base = float(np.max(np.abs(expected_base - actual[:, :35])))
            max_aff = float(np.max(np.abs(expected_aff - actual[:, 45:54])))
            if max_base > 2e-5 or max_aff > 2e-5:
                raise AssertionError(f"logged/candidate feature alignment failed at {split} row {row_ix}: {max_base}, {max_aff}")
            alignment_rows.append({"split": split, "uid": int(target.uid), "position": int(target.position),
                                   "candidate_count": len(cand_ix), "max_abs_diff_base35": max_base,
                                   "max_abs_diff_affinity9": max_aff})
    # Direct RM-dev label-target alignment check (the logged item is encoded with the candidate extractor).
    dev_data = np.load(ART / "rm/rm_dev_features.npz", allow_pickle=False)
    dev_record_rows = list(csv.DictReader((ART / "rm/rm_dev_rows.csv").open("r", encoding="utf-8-sig", newline="")))
    user_index_by_uid = {int(user.uid): i for i, user in enumerate(store.users)}
    logged_checks = []
    for i in np.linspace(0, len(dev_record_rows) - 1, num=5, dtype=np.int64):
        record = dev_record_rows[int(i)]
        ui, pos = user_index_by_uid[int(record["uid"])], int(record["position"])
        user = store.users[ui]
        raw_id = int(record["raw_item_id"]); item_ix = raw_to_index[raw_id]
        expected = rm_feature_row(user, pos, item_ix, raw_id, priors)
        actual = np.asarray(dev_data["x"][int(i)], dtype=np.float32)
        max_diff = float(np.max(np.abs(expected - actual)))
        if max_diff > 2e-5:
            raise AssertionError(f"RM logged-target feature mismatch at heldout row {i}: {max_diff}")
        logged_checks.append({"uid": int(record["uid"]), "position": pos, "max_abs_diff_base35": max_diff})
    alignment_manifest = {"status": "passed", "policy_candidate_rows": alignment_rows,
                          "logged_rm_dev_rows": logged_checks,
                          "tolerance": 2e-5,
                          "same_raw_item_and_prefix_history": True,
                          "catalog_mapping_sha256": feature_manifest["raw_token_mapping_sha256"],
                          "prefix_embedding_sha256": _sha_file(ROOT / "artifacts/policy_diagnostics/rm/prefix_affinity_embeddings.npy")}
    _write_json(ART / "cache/feature_alignment_check.json", alignment_manifest)

    manifest = {
        "status": "complete", "reward_order": list(REWARD_INDEX),
        "reward_index": REWARD_INDEX,
        "policy_feature_names": list(Frozen_FEATURE_NAMES),
        "small_rewards": "existing C models from Frozen cache; clipped [0,1]",
        "large_rewards": "new HGB predictions from Scaling fits; clipped [0,1]",
        "large_base_sha256": _sha_file(large_base_path),
        "large_affinity_sha256": _sha_file(large_affinity_path),
        "caches": created,
        "feature_alignment_check": str((ART / "cache/feature_alignment_check.json").relative_to(ROOT)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(ART / "cache/reward_cache_manifest.json", manifest)

    # Cheap signal diagnostics on the one frozen policy train and dev candidate cache.
    scale = FrozenRewardScale(0.6826758018853311, 0.22662487462369968, 0.001,
                              "frozen_policy_C_refit_reference_conditional_train_v1")
    split_manifest = json.loads((ROOT / "artifacts/frozen_policy/split_manifest.json").read_text(encoding="utf-8"))
    diagnostics = []
    rank_rows = []
    for split in ("train", "common"):
        features, _ = open_v5_cache(cache_dir, split)
        raw_cache_path = ROOT / ("artifacts/residual_support/cache/train" if split == "train"
                                 else "artifacts/policy_diagnostics/cache/common_open_6655")
        _, old_arrays, panel = _load_existing_cache(raw_cache_path)
        rewards = np.memmap(ART / "cache" / f"{split}_rewards.bin", dtype=np.float32, mode="r",
                            shape=(len(panel), 5, 500))
        if split == "common":
            dev_uids = set(int(x) for x in split_manifest["development_uids"])
            rows = np.asarray([i for i, p in enumerate(panel) if int(p.uid) in dev_uids], dtype=np.int64)
        else:
            rows = np.arange(len(panel), dtype=np.int64)
        ref_logp = np.asarray(old_arrays["reference_conditional_logp"][rows], dtype=np.float64)
        ref = np.exp(ref_logp); ref /= ref.sum(axis=1, keepdims=True)
        for model in ("small_base", "small_affinity", "large_base", "large_affinity", "old_deployed"):
            reward = np.asarray(rewards[rows, REWARD_INDEX[model], :], dtype=np.float64)
            means = np.sum(ref * reward, axis=1)
            variances = np.sum(ref * (reward - means[:, None]) ** 2, axis=1)
            diagnostics.append({"split": split, "rows_role": "policy_train" if split == "train" else "policy_dev",
                               "model": model, "contexts": len(rows),
                               "reference_weighted_reward_variance": float(variances.mean()),
                               "reference_weighted_reward_mean": float(means.mean())})
        if split == "common":
            small = np.asarray(rewards[rows, REWARD_INDEX["small_affinity"], :], dtype=np.float64)
            large = np.asarray(rewards[rows, REWARD_INDEX["large_affinity"], :], dtype=np.float64)
            correlations = []
            for i in range(len(rows)):
                r1 = rankdata(small[i], method="average")
                r2 = rankdata(large[i], method="average")
                denom = np.std(r1) * np.std(r2)
                correlations.append(float(np.corrcoef(r1, r2)[0, 1]) if denom > 0 else 1.0)
                if i < 20:
                    rank_rows.append({"uid": int(panel[int(rows[i])].uid),
                                      "spearman_small_affinity_vs_large_affinity": correlations[-1]})
            rank_diag = {"split": "common", "rows_role": "policy_dev", "model": "small_affinity_vs_large_affinity",
                         "contexts": len(rows), "mean_spearman_rank_correlation": float(np.mean(correlations)),
                         "median_spearman_rank_correlation": float(np.median(correlations)),
                         "p10_spearman_rank_correlation": float(np.quantile(correlations, .1)),
                         "p90_spearman_rank_correlation": float(np.quantile(correlations, .9))}
            diagnostics.append(rank_diag)
    _write_csv(ART / "cache/reward_signal_diagnostics.csv", diagnostics)
    _write_csv(ART / "cache/rm_rank_agreement_sample.csv", rank_rows)
    manifest["signal_diagnostic_path"] = str((ART / "cache/reward_signal_diagnostics.csv").relative_to(ROOT))
    _write_json(ART / "cache/reward_cache_manifest.json", manifest)
    return manifest


def _load_v6_policy_arrays(split: str) -> tuple[dict[str, np.ndarray], list[Any], dict[str, Any]]:
    raw_path = ROOT / ("artifacts/residual_support/cache/train" if split == "train"
                       else "artifacts/policy_diagnostics/cache/common_open_6655")
    _, old_arrays, panel = _load_cache(raw_path)
    cache_features, feature_manifest = open_v5_cache(ROOT / "artifacts/frozen_policy/cache", split)
    rewards_meta = json.loads((ART / "cache/reward_cache_manifest.json").read_text(encoding="utf-8"))
    reward_spec = rewards_meta["caches"][split]
    rewards = np.memmap(ROOT / reward_spec["path"], dtype=np.float32, mode="r",
                        shape=tuple(reward_spec["shape"]))
    arrays = {**old_arrays, "candidate_features": cache_features["candidate_features"], "rm_rewards": rewards}
    return arrays, panel, {"features": feature_manifest, "rewards": rewards_meta}


def _load_scaler() -> TrainOnlyFeatureScaler:
    value = json.loads((ROOT / "artifacts/frozen_policy/scaler.json").read_text(encoding="utf-8"))
    for key in ("feature_names", "log1p_columns", "mean", "scale", "constant_columns"):
        value[key] = tuple(value[key])
    scaler = TrainOnlyFeatureScaler(**value)
    if scaler.feature_names != tuple(Frozen_FEATURE_NAMES):
        raise ValueError("frozen Frozen scaler feature order does not match Scaling cache")
    return scaler


def _policy_seed_outputs(seed: int) -> np.ndarray:
    path = ART / f"policy/batch_schedule_seed{seed}.npy"
    train_arrays, train_panel, _ = _load_v6_policy_arrays("train")
    if path.exists():
        schedule = np.load(path, mmap_mode="r")
        if schedule.shape != (3000, 32) or (schedule >= len(train_panel)).any():
            raise ValueError("saved Scaling batch schedule has wrong shape or context indexes")
        return schedule
    rng = np.random.default_rng(seed + 991)
    schedule = rng.integers(0, len(train_panel), size=(3000, 32), dtype=np.int32)
    np.save(path, schedule)
    return np.load(path, mmap_mode="r")


def _initial_state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _save_v6_checkpoint(path: Path, head: FrozenPolicyHead, *, step: int, objective: str,
                        seed: int, reward_model: str, branch: str,
                        scaler: TrainOnlyFeatureScaler, mask_affinity: bool) -> None:
    payload = {"schema_version": "tlab_rm_scaling_policy_head_v1", "trained": True,
               "step": int(step), "objective": objective, "beta": BETA, "seed": int(seed),
               "reward_model": reward_model, "branch": branch,
               "features_affinity_masked": bool(mask_affinity),
               "feature_names": list(Frozen_FEATURE_NAMES),
               "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
               "scaler_sha256": scaler.digest(),
               "head_architecture": "Linear(54,64)-SiLU-Linear(64,64)-SiLU-Linear(64,1), zero-init final layer"}
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp); temp.replace(path)


def _predict_v6_checkpoint(checkpoint: Path, arrays: dict[str, np.ndarray],
                           rows: np.ndarray, scaler: TrainOnlyFeatureScaler,
                           mask_affinity: bool = False,
                           device: torch.device | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not saved.get("trained") or int(saved.get("step", 0)) <= 0:
        raise ValueError(f"untrained checkpoint selected: {checkpoint}")
    if tuple(saved.get("feature_names", ())) != tuple(Frozen_FEATURE_NAMES):
        raise ValueError("Scaling checkpoint feature schema mismatch")
    if saved.get("scaler_sha256") != scaler.digest():
        raise ValueError("Scaling checkpoint scaler digest mismatch")
    if bool(saved.get("features_affinity_masked", False)) != bool(mask_affinity):
        raise ValueError("Scaling checkpoint affinity-mask setting mismatch")
    device = device or _device()
    head = FrozenPolicyHead().to(device)
    head.load_state_dict(saved["state_dict"], strict=True)
    output = np.empty((len(rows), 500), dtype=np.float32)
    features = arrays["candidate_features"]
    refs = arrays["reference_conditional_logp"]
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            stop = min(start + 32, len(rows))
            raw = np.asarray(features[rows[start:stop]], dtype=np.float32)
            scaled = scaler.transform(raw, Frozen_FEATURE_NAMES)
            if mask_affinity:
                scaled[:, :, 45:54] = 0.0
            x = torch.from_numpy(np.ascontiguousarray(scaled)).to(device)
            ref = torch.from_numpy(np.asarray(refs[rows[start:stop]], dtype=np.float32).copy()).to(device)
            output[start:stop] = torch.softmax(ref + head(x), dim=-1).cpu().numpy().astype(np.float32)
    return output, saved


def _train_one(branch: str, seed: int, scaler: TrainOnlyFeatureScaler,
               scale: FrozenRewardScale, batch_schedule: np.ndarray,
               train_arrays: dict[str, np.ndarray], train_panel: list[Any],
               dev_arrays: dict[str, np.ndarray], common_panel: list[Any],
               dev_rows: np.ndarray, item_ids: np.ndarray) -> dict[str, Any]:
    reward_model, objective, mask_affinity = BRANCHES[branch]
    run_name = f"{branch}_seed{seed}"
    run_dir = ART / "policy" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            return prior
    rm_index = REWARD_INDEX[reward_model]
    params = {"learning_rate": .001, "weight_decay": .0001, "batch_contexts": 32,
              "samples_per_context": 16, "updates": 3000, "check_every": 500}
    signature = {"run_name": run_name, "branch": branch, "objective": objective,
                 "reward_model": reward_model, "mask_affinity": mask_affinity,
                 "beta": BETA, "seed": int(seed), "updates": 3000,
                 "scaler_sha256": scaler.digest(), "reward_scale": {
                     "mean": scale.mean, "std": scale.std, "std_floor": scale.std_floor},
                 "batch_schedule_sha256": _sha_file(ART / f"policy/batch_schedule_seed{seed}.npy"),
                 "batch_contexts": 32, "samples_per_context": 16}
    state_path, curve_path, smoke_path = run_dir / "resume_state.pt", run_dir / "dev_curve.jsonl", run_dir / "smoke.json"
    device = _device()
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("signature") != signature:
            raise ValueError(f"cannot resume {run_name}: signature changed")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        head = FrozenPolicyHead().to(device); head.load_state_dict(state["head_state"])
        optimizer = torch.optim.AdamW(head.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
        optimizer.load_state_dict(state["optimizer_state"])
        action_generator = torch.Generator(device=device); action_generator.set_state(state["action_generator_state"].cpu())
        step, best_ndcg, best_kl, best_step = int(state["step"]), float(state["best_ndcg"]), float(state["best_kl"]), int(state["best_step"])
        initial_sha = state["initial_state_sha256"]
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        head = FrozenPolicyHead().to(device)
        initial_sha = _initial_state_sha(head.state_dict())
        optimizer = torch.optim.AdamW(head.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
        action_generator = torch.Generator(device=device).manual_seed(seed + 1991)
        step, best_ndcg, best_kl, best_step = 0, -math.inf, math.inf, -1
        probe_ref = torch.from_numpy(np.asarray(train_arrays["reference_conditional_logp"][:2], dtype=np.float32).copy()).to(device)
        probe_x = torch.zeros((2, 500, len(Frozen_FEATURE_NAMES)), dtype=torch.float32, device=device)
        if not torch.equal(head(probe_x), torch.zeros((2, 500), device=device)):
            raise AssertionError("Scaling zero-init head is not the identity policy")
        if not torch.allclose(torch.softmax(probe_ref + head(probe_x), -1), torch.softmax(probe_ref, -1), atol=1e-7, rtol=0):
            raise AssertionError("Scaling zero-init policy differs from reference")
        state = {"signature": signature, "step": 0, "head_state": head.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "action_generator_state": action_generator.get_state(),
                 "best_ndcg": best_ndcg, "best_kl": best_kl, "best_step": best_step,
                 "initial_state_sha256": initial_sha}
        torch.save(state, state_path)

    run_started = time.perf_counter()
    while step < params["updates"]:
        batch_rows = np.asarray(batch_schedule[step], dtype=np.int64)
        raw = np.asarray(train_arrays["candidate_features"][batch_rows], dtype=np.float32)
        scaled = scaler.transform(raw, Frozen_FEATURE_NAMES)
        if mask_affinity:
            scaled[:, :, 45:54] = 0.0
        x = torch.from_numpy(np.ascontiguousarray(scaled)).to(device)
        ref = torch.from_numpy(np.asarray(train_arrays["reference_conditional_logp"][batch_rows], dtype=np.float32).copy()).to(device)
        reward = torch.from_numpy(np.asarray(train_arrays["rm_rewards"][batch_rows, rm_index, :], dtype=np.float32).copy()).to(device)
        mass = torch.from_numpy(np.asarray(train_arrays["support_mass"][batch_rows], dtype=np.float32).copy()).to(device)
        log_q = torch.log_softmax(ref + head(x), dim=-1)
        if objective == "sampled_loo":
            loss, _ = sampled_loo_loss(log_q, ref, reward, mass, scale, BETA, 16, action_generator)
        else:
            stat = exact_support_statistics(log_q, ref, reward, mass, scale, BETA)
            loss = -stat["support_weighted_objective"].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss in {run_name} at step {step + 1}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient in {run_name} at step {step + 1}")
        optimizer.step(); step += 1
        if step == 50 and not smoke_path.exists():
            if not all(bool(torch.isfinite(p).all()) for p in head.parameters()):
                raise FloatingPointError(f"non-finite parameters in {run_name} smoke")
            _write_json(smoke_path, {"status": "passed_and_continued_in_same_run", "updates": 50,
                                     "finite_loss": float(loss.detach().cpu()),
                                     "gradient_norm": float(grad_norm.detach().cpu()),
                                     "parameters_finite": True})
        if step % params["check_every"] != 0 and step != params["updates"]:
            continue
        dev_q = _predict_current_head(head, dev_arrays, dev_rows, scaler, device, mask_affinity)
        dev_result = evaluate_conditional(common_panel, dev_arrays, dev_rows, dev_q, item_ids)
        dev_kl = _mean_mass_kl(dev_q, np.asarray(dev_arrays["reference_conditional_logp"][dev_rows], dtype=np.float64),
                               np.asarray(dev_arrays["support_mass"][dev_rows], dtype=np.float64))
        curve_row = {"step": step, "dev_ndcg10": dev_result["ndcg10"], "dev_hr10": dev_result["hr10"],
                     "dev_mass_weighted_kl": dev_kl, "dev_hits_at_10": dev_result["hits_at_10"],
                     "dev_n": dev_result["n"], "training_loss": float(loss.detach().cpu()),
                     "training_gradient_norm": float(grad_norm.detach().cpu())}
        with curve_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(curve_row) + "\n")
        score = float(dev_result["ndcg10"])
        better = score > best_ndcg + 1e-12 or (abs(score - best_ndcg) <= 1e-12 and dev_kl < best_kl - 1e-15)
        if better:
            best_ndcg, best_kl, best_step = score, dev_kl, step
            _save_v6_checkpoint(run_dir / "best.pt", head, step=step, objective=objective,
                                seed=seed, reward_model=reward_model, branch=branch,
                                scaler=scaler, mask_affinity=mask_affinity)
        state = {"signature": signature, "step": step, "head_state": head.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "action_generator_state": action_generator.get_state(),
                 "best_ndcg": best_ndcg, "best_kl": best_kl, "best_step": best_step,
                 "initial_state_sha256": initial_sha}
        temp = state_path.with_suffix(".pt.tmp"); torch.save(state, temp); temp.replace(state_path)
        print(f"Scaling {run_name}: {step}/3000 dev_ndcg={score:.6f} KL={dev_kl:.6g}", flush=True)

    if best_step < 1 or not (run_dir / "best.pt").exists():
        raise RuntimeError(f"no trained checkpoint was selected for {run_name}")
    _save_v6_checkpoint(run_dir / "last.pt", head, step=step, objective=objective,
                        seed=seed, reward_model=reward_model, branch=branch,
                        scaler=scaler, mask_affinity=mask_affinity)
    best_saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    if not best_saved.get("trained") or best_saved["step"] <= 0:
        raise AssertionError("Scaling best checkpoint is not trained")
    manifest = {**signature, "status": "complete", "updates_completed": step,
                "best_checkpoint_step": best_step, "best_dev_ndcg10": best_ndcg,
                "best_dev_mass_weighted_kl": best_kl,
                "best_checkpoint_path": str((run_dir / "best.pt").relative_to(ROOT)),
                "best_checkpoint_sha256": _sha_file(run_dir / "best.pt"),
                "last_checkpoint_path": str((run_dir / "last.pt").relative_to(ROOT)),
                "last_checkpoint_sha256": _sha_file(run_dir / "last.pt"),
                "initial_state_sha256": initial_sha,
                "batch_schedule_sha256": signature["batch_schedule_sha256"],
                "selected_by": "highest development next-item NDCG@10; ties lower mass-weighted KL; earlier step retained on exact tie",
                "smoke": json.loads(smoke_path.read_text(encoding="utf-8")),
                "elapsed_seconds": round(time.perf_counter() - run_started, 3)}
    _write_json(manifest_path, manifest)
    return manifest


def _predict_current_head(head: FrozenPolicyHead, arrays: dict[str, np.ndarray], rows: np.ndarray,
                          scaler: TrainOnlyFeatureScaler, device: torch.device,
                          mask_affinity: bool) -> np.ndarray:
    result = np.empty((len(rows), 500), dtype=np.float32)
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            stop = min(start + 32, len(rows))
            raw = np.asarray(arrays["candidate_features"][rows[start:stop]], dtype=np.float32)
            scaled = scaler.transform(raw, Frozen_FEATURE_NAMES)
            if mask_affinity:
                scaled[:, :, 45:54] = 0.0
            x = torch.from_numpy(np.ascontiguousarray(scaled)).to(device)
            ref = torch.from_numpy(np.asarray(arrays["reference_conditional_logp"][rows[start:stop]], dtype=np.float32).copy()).to(device)
            result[start:stop] = torch.softmax(ref + head(x), dim=-1).cpu().numpy().astype(np.float32)
    return result


def _mean_mass_kl(q: np.ndarray, log_ref: np.ndarray, support_mass: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64); ref = np.asarray(log_ref, dtype=np.float64)
    log_q = np.log(np.maximum(q, 1e-30))
    kl = np.sum(q * (log_q - ref), axis=1)
    return float(np.mean(np.asarray(support_mass, dtype=np.float64) * kl))


def train_policy_matrix() -> dict[str, Any]:
    _ensure_dirs()
    if not (ART / "cache/reward_cache_manifest.json").exists():
        prepare_candidate_rewards()
    scaler = _load_scaler()
    scale = FrozenRewardScale(0.6826758018853311, 0.22662487462369968, 0.001,
                              "frozen_policy_C_refit_reference_conditional_train_v1")
    train_arrays, train_panel, _ = _load_v6_policy_arrays("train")
    common_arrays, common_panel, _ = _load_v6_policy_arrays("common")
    split_manifest = json.loads((ROOT / "artifacts/frozen_policy/split_manifest.json").read_text(encoding="utf-8"))
    dev_uids = set(int(x) for x in split_manifest["development_uids"])
    dev_rows = np.asarray([i for i, row in enumerate(common_panel) if int(row.uid) in dev_uids], dtype=np.int64)
    item_ids = np.asarray(pl.read_parquet(ROOT / "data/processed_full_model_tminus14/catalog.parquet")
                          .sort("token_id")["item_id"].to_list(), dtype=np.int64)
    matrix_started = time.perf_counter()
    all_runs = []
    initial_hash_by_seed: dict[int, str] = {}
    for seed in SEEDS:
        schedule = _policy_seed_outputs(seed)
        for branch in ("S", "L", "E", "A"):
            run = _train_one(branch, seed, scaler, scale, schedule, train_arrays, train_panel,
                             common_arrays, common_panel, dev_rows, item_ids)
            expected = initial_hash_by_seed.setdefault(seed, run["initial_state_sha256"])
            if run["initial_state_sha256"] != expected:
                raise AssertionError(f"paired branch initialization differs for seed {seed}")
            all_runs.append(run)
    if len(all_runs) != 12 or any(x.get("updates_completed") != 3000 for x in all_runs):
        raise RuntimeError("Scaling policy matrix did not complete exactly 12 x 3000-update runs")
    summary = {"status": "complete", "branches": {b: BRANCHES[b] for b in BRANCHES},
               "seeds": list(SEEDS), "runs": all_runs,
               "initial_weight_hash_by_seed": initial_hash_by_seed,
               "paired_batches": {str(seed): _sha_file(ART / f"policy/batch_schedule_seed{seed}.npy") for seed in SEEDS},
               "elapsed_seconds": round(time.perf_counter() - matrix_started, 3)}
    _write_json(ART / "policy/matrix_summary.json", summary)
    return summary


def _method_eval(name: str, q: np.ndarray, reward_index: int | None,
                 panel: list[Any], arrays: dict[str, np.ndarray], row_indices: np.ndarray,
                 item_ids: np.ndarray, targets: dict[int, Any] | None = None) -> dict[str, Any]:
    raw = evaluate_conditional(panel, arrays, row_indices, q, item_ids, want_topk=True)
    ref_logp = np.asarray(arrays["reference_conditional_logp"][row_indices], dtype=np.float64)
    mass = np.asarray(arrays["support_mass"][row_indices], dtype=np.float64)
    if reward_index is None:
        diagnostics = {"mass_weighted_kl": 0.0, "normalized_reward_full_delta": 0.0}
    else:
        reward = np.asarray(arrays["rm_rewards"][row_indices, reward_index, :], dtype=np.float32)
        diagnostics = model_reward_metrics(ref_logp, q, reward, mass,
                                          FrozenRewardScale(0.6826758018853311, 0.22662487462369968, 0.001,
                                                           "frozen_policy_C_refit_reference_conditional_train_v1"))
    row_by_uid = {int(panel[int(ix)].uid): int(ix) for ix in row_indices}
    next_new, next_repeat = {}, {}
    for uid, ix in row_by_uid.items():
        row = panel[ix]
        ndcg = float(raw["per_user"][uid]["ndcg10"])
        hit = float(raw["per_user"][uid]["hr10"])
        if bool(row.positive) and bool(row.is_new):
            next_new[uid] = {"ndcg10": ndcg, "hr10": hit}
        if bool(row.positive) and bool(row.is_repeat):
            next_repeat[uid] = {"ndcg10": ndcg, "hr10": hit}
    result = {"name": name, "n": raw["n"], "ndcg10": raw["ndcg10"], "hr10": raw["hr10"],
              "hits_at_10": raw["hits_at_10"], "oov_targets": raw["oov_targets"],
              "per_user": raw["per_user"],
              "per_user_ndcg": {int(u): float(v["ndcg10"]) for u, v in raw["per_user"].items()},
              "per_user_hr": {int(u): float(v["hr10"]) for u, v in raw["per_user"].items()},
              "positive_new": raw["positive_new"], "positive_repeat": raw["positive_repeat"],
              "next_positive_new_per_user": next_new, "next_positive_repeat_per_user": next_repeat,
              "top10_by_user": raw["top10_by_user"], "conditional": q,
              **diagnostics}
    return result


def _mean_user_metric(results: list[dict[str, Any]], key: str) -> dict[int, float]:
    keys = set(results[0][key])
    for result in results[1:]:
        keys &= set(result[key])
    return {int(uid): float(np.mean([float(result[key][uid]) for result in results])) for uid in sorted(keys)}


def _aggregate_policy(results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    ndcg = _mean_user_metric(results, "per_user_ndcg")
    hr = _mean_user_metric(results, "per_user_hr")
    new_users = set(results[0]["next_positive_new_per_user"])
    repeat_users = set(results[0]["next_positive_repeat_per_user"])
    for r in results[1:]:
        new_users &= set(r["next_positive_new_per_user"])
        repeat_users &= set(r["next_positive_repeat_per_user"])
    new_ndcg_only = {u: float(np.mean([r["next_positive_new_per_user"][u]["ndcg10"] for r in results]))
                     for u in new_users}
    new_hr = {u: float(np.mean([r["next_positive_new_per_user"][u]["hr10"] for r in results]))
              for u in new_users}
    repeat_ndcg_only = {u: float(np.mean([r["next_positive_repeat_per_user"][u]["ndcg10"] for r in results]))
                        for u in repeat_users}
    repeat_hr = {u: float(np.mean([r["next_positive_repeat_per_user"][u]["hr10"] for r in results]))
                 for u in repeat_users}
    return {"name": name, "n": len(ndcg), "ndcg10": float(np.mean(list(ndcg.values()))),
            "hr10": float(np.mean(list(hr.values()))), "per_user_ndcg": ndcg, "per_user_hr": hr,
            "hits_at_10": float(np.mean([r["hits_at_10"] for r in results])),
            "oov_targets": int(results[0]["oov_targets"]),
            "positive_new": {"n": len(new_users), "ndcg10": float(np.mean(list(new_ndcg_only.values()))) if new_users else None,
                             "hr10": float(np.mean(list(new_hr.values()))) if new_hr else None,
                             "hits_at_10": float(np.mean([sum(v["hr10"] for v in r["next_positive_new_per_user"].values()) for r in results]))},
            "positive_repeat": {"n": len(repeat_users), "ndcg10": float(np.mean(list(repeat_ndcg_only.values()))) if repeat_users else None,
                                "hr10": float(np.mean(list(repeat_hr.values()))) if repeat_hr else None,
                                "hits_at_10": float(np.mean([sum(v["hr10"] for v in r["next_positive_repeat_per_user"].values()) for r in results]))},
            "mass_weighted_kl": float(np.mean([r["mass_weighted_kl"] for r in results])),
            "normalized_reward_full_delta": float(np.mean([r["normalized_reward_full_delta"] for r in results])),
            "seed_results": [{"seed": r.get("seed"), "ndcg10": r["ndcg10"], "hr10": r["hr10"],
                              "best_step": r.get("best_step")} for r in results]}


def _add_seed(result: dict[str, Any], seed: int, best_step: int | None = None) -> dict[str, Any]:
    return {**result, "seed": int(seed), "best_step": best_step}


def _build_target_slices(root: Path, panel: list[Any], rows: np.ndarray, store: SequenceStore,
                         timelines: dict[int, np.ndarray]) -> dict[str, Any]:
    base = _build_24h_targets(root, panel, rows, store, timelines)
    target_map = base["targets"]
    by_uid = {int(panel[int(ix)].uid): panel[int(ix)] for ix in rows}
    new_targets: dict[int, set[int]] = {}
    repeat_targets: dict[int, set[int]] = {}
    for uid, target_set in target_map.items():
        target = by_uid[int(uid)]
        user = store.users[target.user_index]
        seen = set(int(x) for x in np.asarray(user.raw_item_ids[:target.position], dtype=np.int64) if int(x) >= 0)
        new_targets[int(uid)] = {raw for raw in target_set if raw not in seen}
        repeat_targets[int(uid)] = {raw for raw in target_set if raw in seen}
        if new_targets[int(uid)] & repeat_targets[int(uid)]:
            raise AssertionError("24h new/repeat target slices overlap")
        if new_targets[int(uid)] | repeat_targets[int(uid)] != set(target_set):
            raise AssertionError("24h new/repeat target slices do not cover all positive targets")
    return {**base, "new_targets": new_targets, "repeat_targets": repeat_targets,
            "new_nonempty_contexts": sum(bool(x) for x in new_targets.values()),
            "new_empty_contexts": len(new_targets) - sum(bool(x) for x in new_targets.values()),
            "repeat_nonempty_contexts": sum(bool(x) for x in repeat_targets.values()),
            "repeat_empty_contexts": len(repeat_targets) - sum(bool(x) for x in repeat_targets.values()),
            "new_target_ids": int(sum(len(x) for x in new_targets.values())),
            "repeat_target_ids": int(sum(len(x) for x in repeat_targets.values()))}


def _score_target_sets(top_by_uid: dict[int, list[int]], target_sets: dict[int, set[int]]) -> dict[str, Any]:
    recall, ndcg, hits, n_targets = {}, {}, {}, {}
    for uid, targets in target_sets.items():
        if not targets:
            continue
        top = top_by_uid.get(int(uid), [])
        target_set = set(targets)
        hit_ids = [int(x) for x in top[:10] if int(x) in target_set]
        dcg = sum(1.0 / math.log2(rank + 1) for rank, raw in enumerate(top[:10], 1) if int(raw) in target_set)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(target_set), 10) + 1))
        recall[int(uid)] = len(set(hit_ids)) / len(target_set)
        ndcg[int(uid)] = dcg / ideal
        hits[int(uid)] = len(set(hit_ids))
        n_targets[int(uid)] = len(target_set)
    return {"per_user_recall": recall, "per_user_ndcg": ndcg,
            "recall10": float(np.mean(list(recall.values()))) if recall else None,
            "ndcg10": float(np.mean(list(ndcg.values()))) if ndcg else None,
            "hits_at_10": int(sum(hits.values())), "n_contexts": len(recall),
            "positive_target_ids": int(sum(n_targets.values())),
            "per_user_hits": hits, "per_user_target_count": n_targets}


def _aggregate_target_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    ndcg = _mean_user_metric(results, "per_user_ndcg")
    recall = _mean_user_metric(results, "per_user_recall")
    hits_users = _mean_user_metric(results, "per_user_hits")
    return {"per_user_ndcg": ndcg, "per_user_recall": recall, "per_user_hits": hits_users,
            "ndcg10": float(np.mean(list(ndcg.values()))) if ndcg else None,
            "recall10": float(np.mean(list(recall.values()))) if recall else None,
            "n_contexts": len(ndcg),
            "positive_target_ids": int(np.mean([r["positive_target_ids"] for r in results])),
            "hits_at_10": float(np.mean([r["hits_at_10"] for r in results]))}


def _behavior_metrics(panel: list[Any], rows: np.ndarray, tops: dict[int, list[int]],
                      store: SequenceStore, item_ids: np.ndarray) -> dict[str, Any]:
    repeated = slots = 0
    histories = []
    count = Counter()
    for ix in rows:
        target = panel[int(ix)]; uid = int(target.uid); user = store.users[target.user_index]
        past = set(int(x) for x in np.asarray(user.raw_item_ids[:target.position], dtype=np.int64) if int(x) >= 0)
        top = tops.get(uid, [])[:10]
        slots += len(top)
        repeated += sum(int(item) in past for item in top)
        count.update(int(x) for x in top)
        histories.append(int(target.position))
    total = sum(count.values())
    most = max(count.values(), default=0)
    return {"history_repeat_share_top10": float(repeated / slots) if slots else None,
            "mean_available_history_length": float(np.mean(histories)) if histories else None,
            "most_frequent_recommended_track_share": float(most / total) if total else None,
            "most_frequent_recommended_track_id": int(count.most_common(1)[0][0]) if count else None,
            "catalog_coverage_top10": float(len(count) / len(item_ids)) if len(item_ids) else None,
            "unique_recommended_tracks": len(count), "recommended_slots": slots}


def _result_summary_row(name: str, result: dict[str, Any], strong: dict[int, float]) -> dict[str, Any]:
    ci = _paired_bootstrap(result["per_user_ndcg"], strong, 20261031)
    return {"method": name, "n_contexts": result["n"], "hits_at_10": result["hits_at_10"],
            "oov_targets": result["oov_targets"], "next_ndcg10": result["ndcg10"],
            "next_hr10": result["hr10"], "next_ndcg10_delta_vs_strong": ci["mean_difference"],
            "next_ndcg10_ci95_low": ci["ci95_low"], "next_ndcg10_ci95_high": ci["ci95_high"],
            "positive_next_new_n": result["positive_new"]["n"],
            "positive_next_new_ndcg10": result["positive_new"]["ndcg10"],
            "positive_next_new_hr10": result["positive_new"]["hr10"],
            "positive_next_repeat_n": result["positive_repeat"]["n"],
            "positive_next_repeat_ndcg10": result["positive_repeat"]["ndcg10"],
            "positive_next_repeat_hr10": result["positive_repeat"]["hr10"],
            "mass_weighted_kl": result["mass_weighted_kl"],
            "normalized_reward_full_delta": result["normalized_reward_full_delta"]}


def _save_per_user_results(path: Path, evaluated: dict[str, dict[str, Any]],
                           target_meta: dict[str, Any],
                           report_behavior: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    slices = (("all_24h", target_meta["targets"]),
              ("new_24h", target_meta["new_targets"]),
              ("repeat_24h", target_meta["repeat_targets"]))
    for method, result in evaluated.items():
        result_list = result["seed_results"] if "seed_results" in result else [result]
        if not result_list:
            continue
        for run in result_list:
            seed = run.get("seed")
            score_maps = {name: _score_target_sets(run["top10_by_user"], target_sets)
                          for name, target_sets in slices}
            user_ids = sorted(run["per_user_ndcg"])
            for uid in user_ids:
                row = {"method": method, "seed": seed, "uid": uid,
                       "next_ndcg10": run["per_user_ndcg"].get(uid),
                       "next_hr10": run["per_user_hr"].get(uid),
                       "next_positive_new": uid in run["next_positive_new_per_user"],
                       "next_positive_new_ndcg10": run["next_positive_new_per_user"].get(uid, {}).get("ndcg10"),
                       "next_positive_new_hr10": run["next_positive_new_per_user"].get(uid, {}).get("hr10"),
                       "next_positive_repeat": uid in run["next_positive_repeat_per_user"],
                       "next_positive_repeat_ndcg10": run["next_positive_repeat_per_user"].get(uid, {}).get("ndcg10"),
                       "next_positive_repeat_hr10": run["next_positive_repeat_per_user"].get(uid, {}).get("hr10")}
                for slice_name, score in score_maps.items():
                    row[f"{slice_name}_ndcg10"] = score["per_user_ndcg"].get(uid)
                    row[f"{slice_name}_recall10"] = score["per_user_recall"].get(uid)
                    row[f"{slice_name}_target_ids"] = score["per_user_target_count"].get(uid)
                row.update(report_behavior.get(f"{method}_seed{seed}", {}))
                rows.append(row)
    _write_csv(path, rows)
    return rows


def _candidate_dev_diagnostics(common_arrays: dict[str, np.ndarray], common_panel: list[Any],
                               dev_rows: np.ndarray, scale: FrozenRewardScale,
                               item_ids: np.ndarray, targets: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    reward_names = ("small_base", "small_affinity", "large_base", "large_affinity", "old_deployed")
    ref = np.asarray(common_arrays["reference_conditional_logp"][dev_rows], dtype=np.float64)
    qref = np.exp(ref); qref /= qref.sum(axis=1, keepdims=True)
    for model in reward_names:
        ix = REWARD_INDEX[model]
        reward = np.asarray(common_arrays["rm_rewards"][dev_rows, ix, :], dtype=np.float32)
        q = analytic_q(ref, reward, scale, BETA)
        method = _method_eval(f"analytic_{model}", q, ix, common_panel, common_arrays, dev_rows, item_ids)
        target_scores = _score_target_sets(method["top10_by_user"], targets["targets"])
        rows.append({"method": f"analytic_{model}", "reward_model": model, "beta": BETA,
                     "n_contexts": method["n"], "next_ndcg10": method["ndcg10"],
                     "next_hr10": method["hr10"], "24h_full_horizon_contexts": targets["full_horizon_contexts"],
                     "24h_nonempty_contexts": target_scores["n_contexts"],
                     "24h_ndcg10": target_scores["ndcg10"], "24h_recall10": target_scores["recall10"],
                     "normalized_reward_full_delta": method["normalized_reward_full_delta"],
                     "mass_weighted_kl": method["mass_weighted_kl"]})
    strong = _method_eval("strong_gpt", qref.astype(np.float32), None,
                          common_panel, common_arrays, dev_rows, item_ids)
    strong24 = _score_target_sets(strong["top10_by_user"], targets["targets"])
    rows.insert(0, {"method": "strong_gpt", "reward_model": None, "beta": None,
                    "n_contexts": strong["n"], "next_ndcg10": strong["ndcg10"],
                    "next_hr10": strong["hr10"], "24h_full_horizon_contexts": targets["full_horizon_contexts"],
                    "24h_nonempty_contexts": strong24["n_contexts"], "24h_ndcg10": strong24["ndcg10"],
                    "24h_recall10": strong24["recall10"], "normalized_reward_full_delta": 0.0,
                    "mass_weighted_kl": 0.0})
    return rows


def _run_reporting() -> dict[str, Any]:
    _ensure_dirs()
    stage_started = time.perf_counter()
    policy_summary_path = ART / "policy/matrix_summary.json"
    if not policy_summary_path.exists():
        raise FileNotFoundError("policy matrix must complete before the fixed reporting pass")
    policy_summary = json.loads(policy_summary_path.read_text(encoding="utf-8"))
    if policy_summary.get("status") != "complete" or len(policy_summary.get("runs", [])) != 12:
        raise RuntimeError("reporting is gated on all 12 preregistered policy runs")
    scaler = _load_scaler()
    scale = FrozenRewardScale(0.6826758018853311, 0.22662487462369968, 0.001,
                              "frozen_policy_C_refit_reference_conditional_train_v1")
    common_arrays, common_panel, _ = _load_v6_policy_arrays("common")
    old_meta, old_common_arrays, _ = _load_cache(ROOT / "artifacts/policy_diagnostics/cache/common_open_6655")
    item_ids = np.asarray(pl.read_parquet(ROOT / "data/processed_full_model_tminus14/catalog.parquet")
                          .sort("token_id")["item_id"].to_list(), dtype=np.int64)
    split_manifest = json.loads((ROOT / "artifacts/frozen_policy/split_manifest.json").read_text(encoding="utf-8"))
    dev_uids = set(int(x) for x in split_manifest["development_uids"])
    dev_rows = np.asarray([i for i, p in enumerate(common_panel) if int(p.uid) in dev_uids], dtype=np.int64)
    report_rows = _reporting_indices(common_panel, split_manifest)
    if len(report_rows) != 4065 or len(dev_rows) != 2590:
        raise AssertionError("Frozen frozen reporting/development IDs were not reproduced")
    ref_report = np.asarray(common_arrays["reference_conditional_logp"][report_rows], dtype=np.float64)
    qref_report = np.exp(ref_report); qref_report /= qref_report.sum(axis=1, keepdims=True)

    source_protocol = json.loads((ROOT / "data/processed_full_model_tminus14/protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(ROOT / "data/processed_full_model_tminus14/sequence_index",
                                     expected_source_sha256=source_protocol["processed_listens_sha256"])
    timelines = _event_timelines(ROOT, common_panel, store)
    targets = _build_target_slices(ROOT, common_panel, report_rows, store, timelines)
    dev_targets = _build_target_slices(ROOT, common_panel, dev_rows, store, timelines)
    target_manifest = {k: (v if not isinstance(v, dict) else {"contexts": len(v), "target_ids": int(sum(map(len, v.values()))),
                                                               "nonempty_contexts": sum(bool(x) for x in v.values())})
                       for k, v in targets.items() if k in {"full_horizon_contexts", "empty_target_contexts", "n_contexts",
                                                           "nonempty_contexts", "new_nonempty_contexts", "new_empty_contexts",
                                                           "repeat_nonempty_contexts", "repeat_empty_contexts", "new_target_ids",
                                                           "repeat_target_ids"}}
    _write_json(ART / "reporting/24h_target_manifest.json", {**target_manifest,
               "window_seconds": 86400, "target_starts_after_target_position": True,
               "positive_threshold": 0.5, "empty_sets": "NA", "OOV_in_denominator": True,
               "new_repeat_rule": "history before recommendation only"})

    # Strong reference, analytic tilt diagnostics, Scaling matrix, and selected Frozen weights share one evaluator.
    evaluated_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reward_for_method: dict[str, int | None] = {"strong_gpt": None}
    q_for_external: dict[str, np.ndarray] = {"strong_gpt": qref_report.astype(np.float32)}
    strong = _method_eval("strong_gpt", qref_report.astype(np.float32), None,
                          common_panel, common_arrays, report_rows, item_ids)
    evaluated_runs["strong_gpt"].append(strong)

    for model, name in (("small_affinity", "analytic_small_affinity"),
                        ("large_affinity", "analytic_large_affinity"),
                        ("old_deployed", "analytic_old_deployed")):
        ix = REWARD_INDEX[model]
        rewards = np.asarray(common_arrays["rm_rewards"][report_rows, ix, :], dtype=np.float32)
        q = analytic_q(ref_report, rewards, scale, BETA)
        res = _method_eval(name, q, ix, common_panel, common_arrays, report_rows, item_ids)
        evaluated_runs[name].append(res); reward_for_method[name] = ix

    v6_runs_by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_checkpoints = {"v6": {}, "v5": {}}
    for run in policy_summary["runs"]:
        branch, seed = str(run["branch"]), int(run["seed"])
        checkpoint = ROOT / run["best_checkpoint_path"]
        q, payload = _predict_v6_checkpoint(checkpoint, common_arrays, report_rows, scaler,
                                            mask_affinity=bool(run["mask_affinity"]))
        ix = REWARD_INDEX[str(run["reward_model"])]
        result = _add_seed(_method_eval(branch, q, ix, common_panel, common_arrays, report_rows, item_ids),
                           seed, int(payload["step"]))
        v6_runs_by_branch[branch].append(result)
        selected_checkpoints["v6"][f"{branch}_seed{seed}"] = {
            "path": run["best_checkpoint_path"], "sha256": run["best_checkpoint_sha256"],
            "best_step": run["best_checkpoint_step"], "dev_ndcg10": run["best_dev_ndcg10"],
            "dev_mass_weighted_kl": run["best_dev_mass_weighted_kl"]}
    for branch, runs in v6_runs_by_branch.items():
        q_for_external[branch] = np.mean(np.stack([r["conditional"] for r in runs]), axis=0).astype(np.float32)
        reward_for_method[branch] = REWARD_INDEX[BRANCHES[branch][0]]

    v5_selection = json.loads((ROOT / "artifacts/frozen_policy/reporting_selection.json").read_text(encoding="utf-8"))
    v5_run_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    v5_name_by_group = {"sampled_rl": "tabular_rl", "exact_gradient": "tabular_exact"}
    for group, method_name in v5_name_by_group.items():
        for seed in SEEDS:
            choice = v5_selection[group][str(seed)]
            checkpoint = ROOT / choice["best_checkpoint_path"]
            q, payload = _predict_v6_checkpoint(checkpoint, common_arrays, report_rows, scaler)
            result = _add_seed(_method_eval(method_name, q, REWARD_INDEX["old_deployed"],
                                            common_panel, common_arrays, report_rows, item_ids),
                               seed, int(payload["step"]))
            v5_run_groups[group].append(result)
            selected_checkpoints["v5"][f"{group}_seed{seed}"] = {
                "path": choice["best_checkpoint_path"], "sha256": choice["best_checkpoint_sha256"],
                "best_step": choice["best_step"], "dev_ndcg10": choice["best_dev_ndcg10"]}

    aggregate: dict[str, dict[str, Any]] = {"strong_gpt": strong}
    for name in ("analytic_small_affinity", "analytic_large_affinity", "analytic_old_deployed"):
        aggregate[name] = evaluated_runs[name][0]
    for branch in ("S", "L", "E", "A"):
        results = sorted(v6_runs_by_branch[branch], key=lambda r: r["seed"])
        agg = _aggregate_policy(results, branch)
        agg["seed_results"] = results
        aggregate[branch] = agg
    for group, method_name in v5_name_by_group.items():
        results = sorted(v5_run_groups[group], key=lambda r: r["seed"])
        agg = _aggregate_policy(results, method_name)
        agg["seed_results"] = results
        aggregate[method_name] = agg

    # Behavior metrics are computed on each seed's fixed reporting top-10.
    behavior_rows, behavior_by_run = [], {}
    run_rows = {"strong_gpt": [strong], **evaluated_runs}
    run_rows.update(v6_runs_by_branch)
    run_rows.update({v5_name_by_group[k]: v for k, v in v5_run_groups.items()})
    for name, runs in run_rows.items():
        for run in runs:
            seed = run.get("seed")
            beh = _behavior_metrics(common_panel, report_rows, run["top10_by_user"], store, item_ids)
            behavior_by_run[f"{name}_seed{seed}"] = beh
            behavior_rows.append({"method": name, "seed": seed, **beh,
                                  "mass_weighted_kl": run["mass_weighted_kl"],
                                  "normalized_reward_full_delta": run["normalized_reward_full_delta"]})

    # Apply the 24h Frozen target function to each seed's actual ranking; then average per user over seeds.
    target_scores: dict[str, dict[str, Any]] = {}
    for name, runs in run_rows.items():
        run_scores = []
        for run in runs:
            overall = _score_target_sets(run["top10_by_user"], targets["targets"])
            new = _score_target_sets(run["top10_by_user"], targets["new_targets"])
            repeat = _score_target_sets(run["top10_by_user"], targets["repeat_targets"])
            run_scores.append({"seed": run.get("seed"), "overall": overall, "new": new, "repeat": repeat})
        if len(run_scores) > 1:
            target_scores[name] = {"overall": _aggregate_target_scores([r["overall"] for r in run_scores]),
                                   "new": _aggregate_target_scores([r["new"] for r in run_scores]),
                                   "repeat": _aggregate_target_scores([r["repeat"] for r in run_scores]),
                                   "seed_results": run_scores}
        else:
            target_scores[name] = {"overall": run_scores[0]["overall"], "new": run_scores[0]["new"],
                                   "repeat": run_scores[0]["repeat"], "seed_results": run_scores}

    # Final common development analytic diagnostics include 24h; reporting does not select models/checkpoints.
    dev_analytic_rows = _candidate_dev_diagnostics(common_arrays, common_panel, dev_rows, scale,
                                                    item_ids, dev_targets)
    _write_csv(FINAL / "dev_analytic_diagnostics.csv", dev_analytic_rows)

    methods_rows = []
    for name, result in aggregate.items():
        methods_rows.append(_result_summary_row(name, result, strong["per_user_ndcg"]))
    _write_csv(FINAL / "method_metrics.csv", methods_rows)
    _write_csv(FINAL / "behavior_metrics.csv", behavior_rows)

    # Fixed, paired user bootstrap. The two registered 24h contrasts receive both raw 95% and adjusted 97.5% intervals.
    contrast_rows = []
    for left, right, seed in (("L", "S", 20261022), ("L", "strong_gpt", 20261023)):
        ci = _paired_bootstrap(target_scores[left]["overall"]["per_user_ndcg"],
                              target_scores[right]["overall"]["per_user_ndcg"], seed,
                              draws=2000, levels=(0.95, 0.975))
        contrast_rows.append({"contrast": f"{left} - {right}", "metric": "24h_positive_nonorganic_NDCG@10",
                              **ci, "family_level": 0.95, "bonferroni_adjusted_interval": "97.5% per contrast",
                              "primary_for_rm_scaling": True})
    for left, right, seed in (("L", "S", 20361022), ("L", "strong_gpt", 20361023)):
        ci = _paired_bootstrap(aggregate[left]["per_user_ndcg"], aggregate[right]["per_user_ndcg"], seed,
                              draws=2000, levels=(0.95,))
        contrast_rows.append({"contrast": f"{left} - {right}", "metric": "next_item_NDCG@10",
                              **ci, "family_level": None, "bonferroni_adjusted_interval": None,
                              "primary_for_rm_scaling": False})
    _write_csv(FINAL / "registered_contrasts.csv", contrast_rows)

    secondary_rows = []
    for name, scores in target_scores.items():
        overall, new, repeat = scores["overall"], scores["new"], scores["repeat"]
        secondary_rows.append({"method": name, "full_horizon_contexts": targets["full_horizon_contexts"],
                               "overall_nonempty_contexts": overall["n_contexts"],
                               "overall_empty_contexts": targets["empty_target_contexts"],
                               "overall_positive_target_ids": overall["positive_target_ids"],
                               "overall_hits_at_10": overall["hits_at_10"],
                               "overall_recall10": overall["recall10"], "overall_ndcg10": overall["ndcg10"],
                               "new_nonempty_contexts": new["n_contexts"], "new_empty_contexts": targets["new_empty_contexts"],
                               "new_positive_target_ids": new["positive_target_ids"],
                               "new_recall10": new["recall10"], "new_ndcg10": new["ndcg10"],
                               "repeat_nonempty_contexts": repeat["n_contexts"],
                               "repeat_empty_contexts": targets["repeat_empty_contexts"],
                               "repeat_positive_target_ids": repeat["positive_target_ids"],
                               "repeat_recall10": repeat["recall10"], "repeat_ndcg10": repeat["ndcg10"]})
    _write_csv(FINAL / "24h_metrics.csv", secondary_rows)

    seed_rows = []
    for name, runs in run_rows.items():
        for run in runs:
            seed_rows.append({"method": name, "seed": run.get("seed"), "best_step": run.get("best_step"),
                              "next_ndcg10": run["ndcg10"], "next_hr10": run["hr10"],
                              "next_hits_at_10": run["hits_at_10"], "oov_targets": run["oov_targets"],
                              "mass_weighted_kl": run["mass_weighted_kl"],
                              "normalized_reward_full_delta": run["normalized_reward_full_delta"],
                              "24h_ndcg10": next(x for x in target_scores[name]["seed_results"]
                                                  if x["seed"] == run.get("seed"))["overall"]["ndcg10"],
                              "24h_recall10": next(x for x in target_scores[name]["seed_results"]
                                                    if x["seed"] == run.get("seed"))["overall"]["recall10"]})
    _write_csv(FINAL / "seed_metrics.csv", seed_rows)

    # Full per-user, per-seed primary/secondary data are the source of all reported aggregates.
    flattened_eval = {"strong_gpt": strong, **aggregate}
    per_user_rows = []
    for name, results in run_rows.items():
        for run in results:
            seed = run.get("seed")
            score_by_class = {
                "24h_positive": _score_target_sets(run["top10_by_user"], targets["targets"]),
                "24h_new": _score_target_sets(run["top10_by_user"], targets["new_targets"]),
                "24h_repeat": _score_target_sets(run["top10_by_user"], targets["repeat_targets"]),
            }
            for uid in sorted(run["per_user_ndcg"]):
                row = {"method": name, "seed": seed, "uid": uid,
                       "next_ndcg10": run["per_user_ndcg"][uid], "next_hr10": run["per_user_hr"][uid],
                       "positive_next_new": uid in run["next_positive_new_per_user"],
                       "positive_next_new_ndcg10": run["next_positive_new_per_user"].get(uid, {}).get("ndcg10"),
                       "positive_next_new_hr10": run["next_positive_new_per_user"].get(uid, {}).get("hr10"),
                       "positive_next_repeat": uid in run["next_positive_repeat_per_user"],
                       "positive_next_repeat_ndcg10": run["next_positive_repeat_per_user"].get(uid, {}).get("ndcg10"),
                       "positive_next_repeat_hr10": run["next_positive_repeat_per_user"].get(uid, {}).get("hr10")}
                for label, score in score_by_class.items():
                    row[f"{label}_ndcg10"] = score["per_user_ndcg"].get(uid)
                    row[f"{label}_recall10"] = score["per_user_recall"].get(uid)
                    row[f"{label}_target_ids"] = score["per_user_target_count"].get(uid)
                row.update(behavior_by_run.get(f"{name}_seed{seed}", {}))
                per_user_rows.append(row)
    _write_csv(FINAL / "per_user_metrics.csv", per_user_rows)

    external = {"status": "unavailable", "reason": "not run"}
    try:
        from tlab.frozen_policy.reporting import _external_audit
        audit_q = {name: q_for_external[name] for name in ("strong_gpt", "S", "L", "E")}
        audit_rm = {"S": REWARD_INDEX["small_affinity"], "L": REWARD_INDEX["large_affinity"],
                    "E": REWARD_INDEX["large_affinity"]}
        audit_rewards = {name: np.asarray(common_arrays["rm_rewards"][report_rows, ix, :], dtype=np.float32)
                         for name, ix in audit_rm.items()}
        external = _external_audit(ROOT, report_rows, old_common_arrays, common_arrays,
                                   audit_q, audit_rm, audit_rewards)
        external["status"] = "complete_diagnostic_only"
    except Exception as exc:
        external = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}",
                    "note": "the frozen predictor is a diagnostic, not ground truth or a reward-hacking test"}
    _write_json(ART / "reporting/external_predictor_audit.json", external)

    v5_historical = json.loads((ROOT / "artifacts/frozen_policy/reporting_summary.json").read_text(encoding="utf-8"))
    hist_by_name = {str(r["method"]): r for r in v5_historical["methods"]}
    agreement = {"source": "artifacts/frozen_policy/reporting_summary.json",
                 "same_panel_contexts": len(report_rows), "strong_gpt": {
                     "v6_recomputed_ndcg10": strong["ndcg10"],
                     "v5_recorded_ndcg10": hist_by_name["strong_gpt"]["ndcg10"],
                     "absolute_difference": abs(strong["ndcg10"] - hist_by_name["strong_gpt"]["ndcg10"])},
                 "Frozen_RL_recomputed_ndcg10": aggregate["tabular_rl"]["ndcg10"],
                 "Frozen_RL_recorded_ndcg10": hist_by_name["sampled_rl"]["ndcg10"],
                 "tabular_exact_recomputed_ndcg10": aggregate["tabular_exact"]["ndcg10"],
                 "tabular_exact_recorded_ndcg10": hist_by_name["exact_gradient"]["ndcg10"],
                 "use": "scoring reproduction check only; historical Frozen primary metric remains next-item NDCG@10"}
    agreement["v5_rl_absolute_difference"] = abs(agreement["Frozen_RL_recomputed_ndcg10"] - agreement["Frozen_RL_recorded_ndcg10"])
    agreement["v5_exact_absolute_difference"] = abs(agreement["tabular_exact_recomputed_ndcg10"] - agreement["tabular_exact_recorded_ndcg10"])
    _write_json(ART / "reporting/v5_reporting_reproduction_check.json", agreement)
    _write_json(ART / "reporting/selected_checkpoints.json", selected_checkpoints)

    # Required signal and ranking diagnostics, including candidate-level held-out policy contexts.
    policy_diagnostic = json.loads((ART / "cache/reward_cache_manifest.json").read_text(encoding="utf-8"))
    signal_rows = pl.read_csv(ART / "cache/reward_signal_diagnostics.csv").to_dicts()
    small_large_rank = next((r for r in signal_rows if r.get("model") == "small_affinity_vs_large_affinity"), None)
    if small_large_rank is None:
        raise AssertionError("missing small/large candidate rank diagnostic")
    _write_json(ART / "reporting/reward_signal_diagnostics.json", {
        "policy_reward_cache": policy_diagnostic["signal_diagnostic_path"],
        "small_affinity_vs_large_affinity_candidate_rank_agreement": small_large_rank,
        "interpretation": "describes the reward signal only; it does not prove candidate ranking quality or rule out training"})

    # Recheck the frozen external S/L/E/strong cache has common top-500 support and mass.
    mass = np.asarray(common_arrays["support_mass"][report_rows], dtype=np.float64)
    ref_mass = np.exp(np.asarray(common_arrays["reference_conditional_logp"][report_rows], dtype=np.float64)).sum(axis=1)
    mass_check = {"report_contexts": len(report_rows), "min_support_mass": float(mass.min()),
                  "max_support_mass": float(mass.max()), "min_reference_conditional_sum": float(ref_mass.min()),
                  "max_reference_conditional_sum": float(ref_mass.max()),
                  "candidate_count": int(common_arrays["support_indices"].shape[1]),
                  "catalog_size": len(item_ids)}
    _write_json(ART / "reporting/support_mass_check.json", mass_check)

    result = {"status": "complete", "panel_role": "historically open Frozen reporting panel; not an independent test",
              "n_reporting_contexts": len(report_rows), "n_development_contexts": len(dev_rows),
              "primary_v6_metric": "24h positive-nonorganic NDCG@10; contrasts L-S and L-strong",
              "primary_v5_metric": "next-item NDCG@10 remains unchanged and is reported alongside Scaling results",
              "methods": methods_rows, "registered_contrasts": contrast_rows,
              "24h": secondary_rows, "behavior": behavior_rows,
              "v5_reproduction_check": agreement, "external_predictor": external,
              "selected_checkpoints": selected_checkpoints,
              "target_denominators": target_manifest,
              "reporting_elapsed_seconds": round(time.perf_counter() - stage_started, 3)}
    _write_json(ART / "reporting/reporting_summary.json", result)
    return result


def _budget_status() -> dict[str, Any]:
    path = ART / "run_status.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if "started_at_utc" not in state:
            state["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    else:
        state = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "stages": {}}
    return state


def _elapsed_from_status(status: dict[str, Any]) -> float:
    start = datetime.fromisoformat(status["started_at_utc"])
    return max(0.0, (datetime.now(timezone.utc) - start).total_seconds())


def run(stage: str = "all") -> dict[str, Any]:
    _ensure_dirs()
    status = _budget_status()
    status.setdefault("stages", {})
    _write_json(ART / "run_status.json", status)
    outputs: dict[str, Any] = {}
    if stage in {"all", "prepare"}:
        if (ART / "rm/rm_summary.json").exists():
            outputs["rm"] = json.loads((ART / "rm/rm_summary.json").read_text(encoding="utf-8"))
        else:
            t = time.perf_counter(); outputs["rm"] = prepare_rm()
            status["stages"]["rm_fit_and_features_seconds"] = round(time.perf_counter() - t, 3)
            _write_json(ART / "run_status.json", status)
    if stage in {"all", "rewards"}:
        if (ART / "cache/reward_cache_manifest.json").exists():
            outputs["rewards"] = json.loads((ART / "cache/reward_cache_manifest.json").read_text(encoding="utf-8"))
        else:
            t = time.perf_counter(); outputs["rewards"] = prepare_candidate_rewards()
            status["stages"]["candidate_reward_cache_seconds"] = round(time.perf_counter() - t, 3)
            _write_json(ART / "run_status.json", status)
    if stage in {"all", "policy"}:
        if not (ART / "policy/matrix_summary.json").exists():
            t = time.perf_counter(); outputs["policy"] = train_policy_matrix()
            status["stages"]["policy_matrix_seconds"] = round(time.perf_counter() - t, 3)
            _write_json(ART / "run_status.json", status)
        else:
            outputs["policy"] = json.loads((ART / "policy/matrix_summary.json").read_text(encoding="utf-8"))
    if stage in {"all", "report"}:
        if not (ART / "policy/matrix_summary.json").exists():
            raise RuntimeError("the fixed policy matrix is incomplete; reporting cannot select trained checkpoints")
        outputs["report"] = _run_reporting()
        status["stages"]["reporting_seconds"] = outputs["report"]["reporting_elapsed_seconds"]
        status["elapsed_wall_seconds"] = round(_elapsed_from_status(status), 3)
        status["status"] = "complete"
        _write_json(ART / "run_status.json", status)
    status["elapsed_wall_seconds"] = round(_elapsed_from_status(status), 3)
    status["new_compute_seconds"] = float(sum(v for k, v in status.get("stages", {}).items()
                                               if k.endswith("seconds") and k != "reporting_seconds"))
    _write_json(ART / "run_status.json", status)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen, bounded Scaling final experiment.")
    parser.add_argument("--stage", choices=("all", "prepare", "rewards", "policy", "report"), default="all")
    args = parser.parse_args()
    result = run(args.stage)
    print(json.dumps({"stage": args.stage, "status": "ok", "outputs": list(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


