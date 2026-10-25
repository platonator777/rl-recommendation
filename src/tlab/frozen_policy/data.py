from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import polars as pl
import torch

from tlab.data.store import SequenceStore
from tlab.panel_tools.data import PanelTarget
from tlab.residual_support.cache import _mapping_sha, _load_panel
from tlab.residual_support.features import (
    CANDIDATE_FEATURE_NAMES, POLICY_FEATURE_NAMES, RM_FEATURE_NAMES,
    ItemPriors, candidate_features, history_context_features,
)
from tlab.residual_support.rm import build_item_priors, load_priors, save_priors
from tlab.residual_support.objectives import FrozenRewardScale
from tlab.policy_diagnostics.c_rm import _affinity_rows


AFFINITY_FEATURE_NAMES = (
    "cosine_last", "cosine_mean_last20", "cosine_positive_history",
    "cosine_negative_history", "last_history_missing", "history20_missing",
    "positive_history_missing", "negative_history_missing", "candidate_embedding_missing",
)
Frozen_FEATURE_NAMES = (*POLICY_FEATURE_NAMES, *AFFINITY_FEATURE_NAMES)
RM_NAMES = ("old_deployed", "same_split_control", "prefix_affinity")
BETAS = (3.0, 1.0, 0.3, 0.1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_cache(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], list[PanelTarget]]:
    meta = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    for name, spec in meta["artifact_arrays"].items():
        if name not in {"candidate_features", "reference_conditional_logp", "support_indices",
                        "support_mass", "raw_candidate_rewards", "reference_logp_full"}:
            continue
        if name == "reference_logp_full" and not meta.get("reference_full_catalog_saved"):
            continue
        arrays[name] = np.memmap(path / f"{name}.bin", dtype=spec["dtype"], mode="r",
                                 shape=tuple(spec["shape"]))
    panel = _load_panel(path / "panel_rows.jsonl")
    if len(panel) != int(meta["n_contexts"]):
        raise ValueError(f"cache row count mismatch: {path}")
    return meta, arrays, panel


def _v5_hash_bucket(uid: int) -> int:
    digest = hashlib.sha256(f"{bytes.fromhex('76355f666173743a').decode('ascii')}{int(uid)}".encode("ascii")).hexdigest()
    return int(digest, 16) % 10


def freeze_split(panel: Sequence[PanelTarget], output: Path) -> dict[str, Any]:
    if len({row.uid for row in panel}) != len(panel):
        raise ValueError("Frozen common panel must have one context per user")
    dev = [row for row in panel if _v5_hash_bucket(row.uid) < 4]
    report = [row for row in panel if _v5_hash_bucket(row.uid) >= 4]
    dev_uids, report_uids = {r.uid for r in dev}, {r.uid for r in report}
    dev_events, report_events = {r.event_id for r in dev}, {r.event_id for r in report}
    if dev_uids & report_uids or dev_events & report_events or len(dev) + len(report) != len(panel):
        raise AssertionError("hash split is not user/event disjoint")
    result = {
        "schema_version": 1,
        "protocol_id": "tlab_yambda_frozen_policy_2026-10-22",
        "role": "historically open, row-wise temporally valid retrospective split; reporting is not an independent test",
        "hash": "int(SHA256(hex-prefix 76355f666173743a + decimal uid), 16) % 10 < 4 is development; remaining users are reporting",
        "n_contexts": len(panel), "n_users": len(panel),
        "development_n": len(dev), "reporting_n": len(report),
        "development_uids": sorted(int(x) for x in dev_uids),
        "reporting_uids": sorted(int(x) for x in report_uids),
        "development_event_ids": sorted(dev_events),
        "reporting_event_ids": sorted(report_events),
        "user_overlap": 0, "event_overlap": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _candidate_affinity_vectorized(user: Any, position: int, candidates: np.ndarray,
                                   embedding: np.ndarray) -> np.ndarray:
    """Match C's nine prefix-affinity columns while vectorizing over candidate actions."""
    ids = np.asarray(user.tokens[max(0, int(position) - 1000):int(position)], dtype=np.int64)
    ratios = np.asarray(user.ratio_raw[max(0, int(position) - 1000):int(position)], dtype=np.float32)
    if ids.shape != ratios.shape:
        raise ValueError("history tokens and ratios are not aligned")
    valid = (ids >= 3) & (ids < embedding.shape[0] + 3)
    hist_index = ids[valid] - 3
    hist_ratio = ratios[valid]
    candidate = np.asarray(candidates, dtype=np.int64)
    vectors = embedding[candidate]
    candidate_norm = np.linalg.norm(vectors, axis=1)
    denominator = np.maximum(candidate_norm, 1e-8)
    result = np.zeros((len(candidate), 9), dtype=np.float32)
    missing_candidate = candidate_norm < 1e-8

    def sim(history_indices: np.ndarray) -> tuple[np.ndarray, bool]:
        if not len(history_indices):
            return np.zeros(len(candidate), dtype=np.float32), True
        mean = np.asarray(embedding[history_indices].mean(axis=0), dtype=np.float32)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-8:
            return np.zeros(len(candidate), dtype=np.float32), False
        return ((vectors @ mean) / (denominator * norm)).astype(np.float32), False

    result[:, 0], result[:, 4] = sim(hist_index[-1:])
    result[:, 1], result[:, 5] = sim(hist_index[-20:])
    result[:, 2], result[:, 6] = sim(hist_index[hist_ratio >= 0.5])
    result[:, 3], result[:, 7] = sim(hist_index[(hist_ratio < 0.5) & np.isfinite(hist_ratio)])
    result[:, 8] = missing_candidate.astype(np.float32)
    return result


def _load_rm_models(root: Path) -> dict[str, Any]:
    return {
        "old_deployed": joblib.load(root / "artifacts/residual_support/rm/models/optimized_rm.joblib"),
        "same_split_control": joblib.load(root / "artifacts/policy_diagnostics/rm/C_refit_residual_support_optimized_control.joblib"),
        "prefix_affinity": joblib.load(root / "artifacts/policy_diagnostics/rm/C_prefix_safe_affinity_extension.joblib"),
    }


def _write_feature_and_reward_cache(root: Path, cache_name: str, old_arrays: dict[str, np.ndarray],
                                    panel: Sequence[PanelTarget], item_ids: np.ndarray, store: SequenceStore,
                                    priors: ItemPriors, embedding: np.ndarray, models: dict[str, Any],
                                    output_dir: Path) -> dict[str, Any]:
    n, k = old_arrays["support_indices"].shape
    feature_path = output_dir / f"candidate_features_{cache_name}.bin"
    reward_path = output_dir / f"rm_rewards_{cache_name}.bin"
    if feature_path.exists():
        feature_path.unlink()
    if reward_path.exists():
        reward_path.unlink()
    features = np.memmap(feature_path, dtype=np.float32, mode="w+", shape=(n, k, len(Frozen_FEATURE_NAMES)))
    rewards = np.memmap(reward_path, dtype=np.float32, mode="w+", shape=(n, len(RM_NAMES), k))
    for i, target in enumerate(panel):
        old = np.asarray(old_arrays["candidate_features"][i], dtype=np.float32)
        indices = np.asarray(old_arrays["support_indices"][i], dtype=np.int64)
        raw_ids = item_ids[indices]
        user = store.users[target.user_index]
        # panel_tools.1's base RM values are reused for the old model; C models get their exact
        # C-cutoff priors reconstructed below, with the same 35-column schema/order.
        context = np.asarray(old[:, :len(RM_FEATURE_NAMES)], dtype=np.float32)[:, :14]
        c_candidate = candidate_features(user, target.position, raw_ids, catalog_indices=indices, priors=priors)
        c_base = np.column_stack((context, c_candidate)).astype(np.float32, copy=False)
        affinity = _candidate_affinity_vectorized(user, target.position, indices, embedding)
        features[i, :, :len(POLICY_FEATURE_NAMES)] = old
        features[i, :, :len(RM_FEATURE_NAMES)] = c_base
        features[i, :, len(POLICY_FEATURE_NAMES):] = affinity

        old_pred = np.clip(models["old_deployed"].predict(old[:, :len(RM_FEATURE_NAMES)]), 0.0, 1.0)
        c_x = np.column_stack((c_base, affinity))
        control_pred = np.clip(models["same_split_control"].predict(c_base), 0.0, 1.0)
        affinity_pred = np.clip(models["prefix_affinity"].predict(c_x), 0.0, 1.0)
        rewards[i, 0] = old_pred.astype(np.float32)
        rewards[i, 1] = control_pred.astype(np.float32)
        rewards[i, 2] = affinity_pred.astype(np.float32)
        if not np.isfinite(features[i]).all() or not np.isfinite(rewards[i]).all():
            raise FloatingPointError(f"non-finite Frozen features/reward at {cache_name} row {i}")
        if i == 0 or (i + 1) % 500 == 0:
            print(f"Frozen features {cache_name}: {i+1:,}/{n:,}", flush=True)
    features.flush(); rewards.flush()
    del features, rewards
    return {
        "features_path": feature_path.name,
        "rewards_path": reward_path.name,
        "feature_shape": [n, k, len(Frozen_FEATURE_NAMES)],
        "reward_shape": [n, len(RM_NAMES), k],
        "feature_dtype": "float32", "reward_dtype": "float32",
        "source_cache": cache_name,
        "feature_names": list(Frozen_FEATURE_NAMES),
    }


def prepare_feature_caches(root: Path, output_dir: Path) -> dict[str, Any]:
    """Build C-prior and prefix-affinity features on the already fixed candidates."""
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_meta, train_arrays, train_panel = _load_cache(root / "artifacts/residual_support/cache/train")
    common_meta, common_arrays, common_panel = _load_cache(root / "artifacts/policy_diagnostics/cache/common_open_6655")
    processed = root / "data/processed_full_model_tminus14"
    protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(processed / "sequence_index",
                                     expected_source_sha256=protocol["processed_listens_sha256"])
    catalog = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    tokens = np.asarray(catalog["token_id"].to_list(), dtype=np.int64)
    mapping_sha = _mapping_sha(item_ids, tokens)
    c_manifest = json.loads((root / "artifacts/policy_diagnostics/rm/C_split_manifest.json").read_text(encoding="utf-8"))
    cutoff = float(c_manifest["feature_prior_cutoff_exclusive"])
    priors_path = output_dir / "c_priors_reconstructed.npz"
    if priors_path.exists():
        priors = load_priors(priors_path, expected_mapping_sha256=mapping_sha)
        if not math.isclose(priors.cutoff_raw_exclusive, cutoff, rel_tol=0, abs_tol=1e-6):
            raise ValueError("saved Frozen C priors use a different cutoff")
    else:
        priors = build_item_priors(processed / "listens.parquet", catalog,
                                   cutoff_exclusive=cutoff,
                                   source_split="F_RM_strictly_before_all_C_targets_reconstructed_v5")
        save_priors(priors_path, priors, mapping_sha)
    if not math.isclose(priors.cutoff_raw_exclusive, cutoff, rel_tol=0, abs_tol=1e-6):
        raise ValueError("reconstructed C priors do not match C_split_manifest cutoff")
    embedding_path = root / "artifacts/policy_diagnostics/rm/prefix_affinity_embeddings.npy"
    if sha256_file(embedding_path) != c_manifest["embedding"]["embedding_sha256"]:
        raise ValueError("prefix-affinity embedding hash differs from C manifest")
    embedding = np.load(embedding_path, mmap_mode="r")
    if embedding.shape != (len(item_ids), int(c_manifest["embedding"]["components"])):
        raise ValueError("prefix-affinity embedding shape differs from catalog")

    # Quick temporal gate: the common policy-training targets are later than the
    # prefix-safe C prior cutoff. Their features are computed from strictly prior history.
    from tlab.residual_support.rm import _group_event_timestamps
    train_timestamps = _group_event_timestamps(processed / "listens.parquet", store)
    train_times = [float(train_timestamps[r.uid][r.position]) for r in train_panel]
    if min(train_times) <= cutoff:
        raise ValueError("C prior cutoff is not earlier than every policy-train target")
    if len(train_panel) != train_meta["n_contexts"] or len(common_panel) != common_meta["n_contexts"]:
        raise ValueError("input cache rows disagree with manifests")
    for cache_arrays in (train_arrays, common_arrays):
        if cache_arrays["support_indices"].shape[1] != 500 or len(item_ids) != 20000:
            raise ValueError("Frozen requires the fixed 20k catalog and top-500 support")

    models = _load_rm_models(root)
    train_spec = _write_feature_and_reward_cache(root, "train", train_arrays, train_panel,
                                                 item_ids, store, priors, embedding, models, output_dir)
    common_spec = _write_feature_and_reward_cache(root, "common", common_arrays, common_panel,
                                                  item_ids, store, priors, embedding, models, output_dir)
    result = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": "tlab_yambda_frozen_policy_2026-10-22",
        "source_train_cache_identity": train_meta.get("bound_identity_sha256"),
        "source_common_cache_manifest_sha256": sha256_file(root / "artifacts/policy_diagnostics/cache/common_open_6655/manifest.json"),
        "catalog_size": len(item_ids), "support_size": 500,
        "raw_token_mapping_sha256": mapping_sha,
        "c_prior_cutoff_exclusive": cutoff,
        "c_prior_source": "rebuilt with residual_support.rm.build_item_priors from C_split_manifest cutoff; F_RM nonorganic, unambiguous events strictly before cutoff",
        "c_priors_sha256": sha256_file(priors_path),
        "affinity_embedding_path": str(embedding_path.relative_to(root)),
        "affinity_embedding_sha256": sha256_file(embedding_path),
        "affinity_names_and_order": list(AFFINITY_FEATURE_NAMES),
        "feature_names_and_order": list(Frozen_FEATURE_NAMES),
        "rm_feature_order": list(RM_FEATURE_NAMES),
        "rm_models": {name: {"path": str((root / path).relative_to(root)),
                             "sha256": sha256_file(root / path)} for name, path in {
            "old_deployed": "artifacts/residual_support/rm/models/optimized_rm.joblib",
            "same_split_control": "artifacts/policy_diagnostics/rm/C_refit_residual_support_optimized_control.joblib",
            "prefix_affinity": "artifacts/policy_diagnostics/rm/C_prefix_safe_affinity_extension.joblib",
        }.items()},
        "train": train_spec, "common": common_spec,
        "train_min_target_time": min(train_times),
        "train_contexts": len(train_panel), "common_contexts": len(common_panel),
        "future_target_or_feedback_columns": "none; feature extraction uses sequences strictly before each panel target",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "feature_cache_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def open_v5_cache(output_dir: Path, split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    meta = json.loads((output_dir / "feature_cache_manifest.json").read_text(encoding="utf-8"))
    spec = meta[split]
    features = np.memmap(output_dir / spec["features_path"], dtype=spec["feature_dtype"], mode="r",
                         shape=tuple(spec["feature_shape"]))
    rewards = np.memmap(output_dir / spec["rewards_path"], dtype=spec["reward_dtype"], mode="r",
                        shape=tuple(spec["reward_shape"]))
    return {"candidate_features": features, "rm_rewards": rewards}, meta


def frozen_reward_scale(train_cache: dict[str, np.ndarray]) -> FrozenRewardScale:
    """Calibrate once under the reference conditional distribution on policy train."""
    reference = np.asarray(train_cache["reference_conditional_logp"], dtype=np.float64)
    reward = np.asarray(train_cache["rm_rewards"][:, 1, :], dtype=np.float64)
    weights = np.exp(reference)
    weights /= weights.sum(axis=1, keepdims=True)
    means = np.sum(weights * reward, axis=1)
    second = np.sum(weights * reward * reward, axis=1)
    mean = float(means.mean())
    std = math.sqrt(max(float(second.mean()) - mean * mean, 0.0))
    return FrozenRewardScale(mean=mean, std=std, std_floor=1e-3,
                             version="frozen_policy_C_refit_reference_conditional_train_v1")


def analytic_q(reference_logp: np.ndarray, rewards: np.ndarray,
               scale: FrozenRewardScale, beta: float) -> np.ndarray:
    ref = np.asarray(reference_logp, dtype=np.float64)
    raw = np.asarray(rewards, dtype=np.float64)
    if ref.shape != raw.shape:
        raise ValueError("analytic tilt reference and reward shape mismatch")
    logits = ref + ((raw - scale.mean) / max(scale.std, scale.std_floor)) / float(beta)
    logits -= logits.max(axis=-1, keepdims=True)
    q = np.exp(logits)
    q /= q.sum(axis=-1, keepdims=True)
    return q.astype(np.float32)


def full_catalog_probabilities(full_reference_logp: np.ndarray, support_indices: np.ndarray,
                               support_mass: float, conditional: np.ndarray) -> np.ndarray:
    probabilities = np.exp(np.asarray(full_reference_logp, dtype=np.float64))
    probabilities[np.asarray(support_indices, dtype=np.int64)] = float(support_mass) * np.asarray(conditional, dtype=np.float64)
    if not np.isclose(probabilities.sum(), 1.0, atol=2e-5):
        raise ValueError("support-constrained method did not preserve full reference probability mass")
    return probabilities


def topk_stable(scores: np.ndarray, item_ids: np.ndarray, k: int = 10) -> np.ndarray:
    scores = np.asarray(scores)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    higher = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    need = k - len(higher)
    if need > 0:
        tied = tied[np.argsort(np.asarray(item_ids)[tied], kind="stable")[:need]]
        selected = np.concatenate((higher, tied))
    else:
        selected = higher
    return selected[np.lexsort((np.asarray(item_ids)[selected], -scores[selected]))]


def evaluate_conditional(panel: Sequence[PanelTarget], arrays: dict[str, np.ndarray],
                         row_indices: Sequence[int], conditional: np.ndarray,
                         item_ids: np.ndarray, *, want_topk: bool = False) -> dict[str, Any]:
    if len(row_indices) != len(conditional):
        raise ValueError("row indices and conditional policy are misaligned")
    raw_to_index = {int(raw): i for i, raw in enumerate(item_ids.tolist())}
    per_user: dict[int, dict[str, float]] = {}
    slices: dict[str, list[dict[str, float]]] = {"positive_new": [], "positive_repeat": []}
    tops: dict[int, list[int]] = {}
    hit_count = 0
    oov = 0
    for local, row_index in enumerate(row_indices):
        target = panel[int(row_index)]
        probs = full_catalog_probabilities(arrays["reference_logp_full"][int(row_index)],
                                           arrays["support_indices"][int(row_index)],
                                           float(arrays["support_mass"][int(row_index)]),
                                           conditional[local])
        index = raw_to_index.get(int(target.raw_item_id))
        if index is None:
            rank = None
            oov += 1
        else:
            score = probs[index]
            rank = int(np.count_nonzero((probs > score) | ((probs == score) & (item_ids < target.raw_item_id))) + 1)
        ndcg = float(1.0 / math.log2(rank + 1)) if rank is not None and rank <= 10 else 0.0
        hit = float(rank is not None and rank <= 10)
        hit_count += int(hit)
        per_user[int(target.uid)] = {"ndcg10": ndcg, "hr10": hit, "rank": float(rank or 0)}
        if target.positive and target.is_new:
            slices["positive_new"].append({"uid": int(target.uid), "ndcg10": ndcg, "hr10": hit})
        if target.positive and target.is_repeat:
            slices["positive_repeat"].append({"uid": int(target.uid), "ndcg10": ndcg, "hr10": hit})
        if want_topk:
            top_ix = topk_stable(probs, item_ids, 10)
            tops[int(target.uid)] = [int(item_ids[j]) for j in top_ix]

    def summary(rows: list[dict[str, float]]) -> dict[str, Any]:
        n = len(rows)
        return {"n": n, "hits_at_10": int(sum(r["hr10"] for r in rows)),
                "ndcg10": float(np.mean([r["ndcg10"] for r in rows])) if n else None,
                "hr10": float(np.mean([r["hr10"] for r in rows])) if n else None}
    vals = list(per_user.values())
    return {
        "n": len(vals), "hits_at_10": hit_count, "oov_targets": oov,
        "ndcg10": float(np.mean([v["ndcg10"] for v in vals])) if vals else None,
        "hr10": float(np.mean([v["hr10"] for v in vals])) if vals else None,
        "per_user": per_user,
        "positive_new": summary(slices["positive_new"]),
        "positive_repeat": summary(slices["positive_repeat"]),
        "top10_by_user": tops,
    }


def _policy_distribution_diagnostics(reference: np.ndarray, conditional: np.ndarray,
                                     rewards: np.ndarray, support_mass: np.ndarray,
                                     scale: FrozenRewardScale) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    q = np.asarray(conditional, dtype=np.float64)
    raw = np.asarray(rewards, dtype=np.float64)
    mass = np.asarray(support_mass, dtype=np.float64)
    normalized = (raw - scale.mean) / max(scale.std, scale.std_floor)
    kl = np.sum(q * (np.log(np.maximum(q, 1e-30)) - ref), axis=1)
    expected = np.sum(q * normalized, axis=1)
    return {
        "mass_weighted_kl": float(np.mean(mass * kl)),
        "normalized_reward_full_delta": float(np.mean(mass * (expected - np.sum(np.exp(ref) * normalized, axis=1)))),
    }


def model_reward_metrics(reference: np.ndarray, conditional: np.ndarray,
                         rewards: np.ndarray, support_mass: np.ndarray,
                         scale: FrozenRewardScale) -> dict[str, float]:
    return _policy_distribution_diagnostics(reference, conditional, rewards, support_mass, scale)

