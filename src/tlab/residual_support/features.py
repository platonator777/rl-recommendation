from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FEATURE_SCHEMA_VERSION = "residual_support_candidate_action_features_v1"
HISTORY_WINDOWS = (50, 200, 1000)

CONTEXT_FEATURE_NAMES = (
    "log_history_length_1000", "unique_personal_items_1000",
    *(name for width in HISTORY_WINDOWS for name in (
        f"past_ratio_mean_{width}", f"past_ratio_std_{width}",
        f"past_ratio_count_{width}", f"past_ratio_missing_{width}",
    )),
)
CANDIDATE_FEATURE_NAMES = (
    *(name for width in HISTORY_WINDOWS for name in (
        f"personal_repeat_count_{width}", f"personal_recency_steps_{width}",
        f"personal_ratio_mean_{width}", f"personal_ratio_count_{width}",
        f"personal_ratio_missing_{width}",
    )),
    "train_popularity_count", "train_ratio_prior_mean", "train_ratio_prior_count",
    "train_ratio_prior_missing", "train_duration_seconds", "train_duration_missing",
)
POLICY_EXTRA_FEATURE_NAMES = (
    "rm_fit_item_count", "rm_fit_unsupported", "rm_fit_support_1_9",
    "rm_fit_support_10_99", "rm_fit_support_100_plus",
    "strong_reference_logp_centered", "strong_reference_rank_fraction",
    "item_embedding_cosine_last", "item_embedding_cosine_history20",
    "item_embedding_history_missing",
)
POLICY_FEATURE_NAMES = (*CONTEXT_FEATURE_NAMES, *CANDIDATE_FEATURE_NAMES, *POLICY_EXTRA_FEATURE_NAMES)
RM_FEATURE_NAMES = (*CONTEXT_FEATURE_NAMES, *CANDIDATE_FEATURE_NAMES)


@dataclass(frozen=True)
class ItemPriors:
    """Temporal item aggregates aligned to the catalog's raw-ID order."""

    train_popularity_count: np.ndarray
    ratio_prior_mean: np.ndarray
    ratio_prior_count: np.ndarray
    duration_seconds: np.ndarray
    duration_missing: np.ndarray
    cutoff_raw_exclusive: float
    source_split: str

    def __post_init__(self) -> None:
        arrays = (self.train_popularity_count, self.ratio_prior_mean, self.ratio_prior_count,
                  self.duration_seconds, self.duration_missing)
        if any(np.asarray(a).ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
            raise ValueError("item prior arrays must be aligned one-dimensional vectors")
        if not np.isfinite(self.cutoff_raw_exclusive):
            raise ValueError("item prior cutoff must be finite")


def _window_personal(history_ids: np.ndarray, history_ratios: np.ndarray, candidates: np.ndarray,
                     width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = history_ids[-width:]
    ratios = history_ratios[-width:]
    candidate_index = {int(item): i for i, item in enumerate(candidates.tolist())}
    counts = np.zeros(len(candidates), dtype=np.float32)
    last = np.full(len(candidates), -1, dtype=np.int32)
    ratio_sum = np.zeros(len(candidates), dtype=np.float64)
    ratio_count = np.zeros(len(candidates), dtype=np.float32)
    for pos, (item_raw, ratio_raw) in enumerate(zip(ids.tolist(), ratios.tolist())):
        index = candidate_index.get(int(item_raw))
        if index is None:
            continue
        counts[index] += 1.0
        last[index] = pos
        if np.isfinite(ratio_raw):
            ratio_sum[index] += float(np.clip(ratio_raw, 0.0, 1.0))
            ratio_count[index] += 1.0
    ratio_mean = np.divide(ratio_sum, ratio_count, out=np.zeros_like(ratio_sum), where=ratio_count > 0).astype(np.float32)
    recency = np.where(last >= 0, len(ids) - 1 - last, len(ids)).astype(np.float32)
    return counts, recency, ratio_mean, ratio_count, (ratio_count == 0).astype(np.float32)


def history_context_features(user: Any, position: int) -> np.ndarray:
    if position < 0 or position > len(user.raw_item_ids):
        raise ValueError("context position lies outside the user sequence")
    start = max(0, int(position) - 1000)
    items = np.asarray(user.raw_item_ids[start:position], dtype=np.int64)
    ratios = np.asarray(user.ratio_raw[start:position], dtype=np.float64)
    if items.shape != ratios.shape:
        raise ValueError("user history item and feedback arrays are misaligned")
    output = [float(np.log1p(len(items))), float(np.unique(items[items >= 0]).size)]
    for width in HISTORY_WINDOWS:
        values = ratios[-width:]
        valid = values[np.isfinite(values)]
        output.extend((float(valid.mean()) if valid.size else 0.0,
                       float(valid.std()) if valid.size else 0.0,
                       float(valid.size), float(not valid.size)))
    result = np.asarray(output, dtype=np.float32)
    if result.shape != (len(CONTEXT_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise AssertionError("context feature schema does not match its values")
    return result


def candidate_features(
    user: Any,
    position: int,
    candidate_raw_ids: np.ndarray,
    *,
    catalog_indices: np.ndarray,
    priors: ItemPriors,
) -> np.ndarray:
    candidates = np.asarray(candidate_raw_ids, dtype=np.int64)
    indices = np.asarray(catalog_indices, dtype=np.int64)
    if candidates.ndim != 1 or indices.shape != candidates.shape or len(set(candidates.tolist())) != len(candidates):
        raise ValueError("candidate IDs and catalog indices must be aligned unique vectors")
    if np.any((indices < 0) | (indices >= len(priors.train_popularity_count))):
        raise ValueError("candidate catalog index is outside prior vectors")
    if position < 0 or position > len(user.raw_item_ids):
        raise ValueError("context position lies outside the user sequence")
    start = max(0, int(position) - 1000)
    history_ids = np.asarray(user.raw_item_ids[start:position], dtype=np.int64)
    history_ratios = np.asarray(user.ratio_raw[start:position], dtype=np.float64)
    if history_ids.shape != history_ratios.shape:
        raise ValueError("history IDs and ratios are misaligned")
    columns: list[np.ndarray] = []
    for width in HISTORY_WINDOWS:
        columns.extend(_window_personal(history_ids, history_ratios, candidates, width))
    pop = np.asarray(priors.train_popularity_count, dtype=np.float64)[indices]
    ratio_mean = np.asarray(priors.ratio_prior_mean, dtype=np.float64)[indices]
    ratio_count = np.asarray(priors.ratio_prior_count, dtype=np.float64)[indices]
    duration = np.asarray(priors.duration_seconds, dtype=np.float64)[indices]
    duration_missing = np.asarray(priors.duration_missing, dtype=bool)[indices] | ~np.isfinite(duration) | (duration <= 0)
    prior_missing = (ratio_count <= 0) | ~np.isfinite(ratio_mean)
    global_mean = 0.5
    smooth = (np.nan_to_num(ratio_mean, nan=global_mean) * ratio_count + 10.0 * global_mean) / (ratio_count + 10.0)
    columns.extend((pop.astype(np.float32), smooth.astype(np.float32), ratio_count.astype(np.float32),
                    prior_missing.astype(np.float32), np.where(duration_missing, 0.0, duration).astype(np.float32),
                    duration_missing.astype(np.float32)))
    values = np.column_stack(columns).astype(np.float32, copy=False)
    if values.shape != (len(candidates), len(CANDIDATE_FEATURE_NAMES)) or not np.isfinite(values).all():
        raise AssertionError("candidate feature schema does not match finite values")
    return values


def rm_feature_row(user: Any, position: int, candidate_index: int, candidate_raw_id: int,
                   priors: ItemPriors) -> np.ndarray:
    context = history_context_features(user, position)
    item = candidate_features(
        user, position, np.asarray([candidate_raw_id], dtype=np.int64),
        catalog_indices=np.asarray([candidate_index], dtype=np.int64), priors=priors,
    )[0]
    return np.concatenate((context, item)).astype(np.float32, copy=False)


def policy_feature_matrix(
    context_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    rm_fit_counts: np.ndarray,
    candidate_catalog_indices: np.ndarray,
    reference_logp: np.ndarray,
    item_embeddings: np.ndarray,
    history_tokens: np.ndarray,
) -> np.ndarray:
    """Combine item/history features with frozen actor-item interactions.

    The actor score and two frozen-embedding similarities preserve candidate
    identity even when the user has never played a catalog item.
    """
    context = np.asarray(context_values, dtype=np.float32)
    item = np.asarray(candidate_values, dtype=np.float32)
    indices = np.asarray(candidate_catalog_indices, dtype=np.int64)
    logp = np.asarray(reference_logp, dtype=np.float32)
    embedding = np.asarray(item_embeddings, dtype=np.float32)
    tokens = np.asarray(history_tokens, dtype=np.int64)
    if context.shape != (len(CONTEXT_FEATURE_NAMES),) or item.ndim != 2 or item.shape[1] != len(CANDIDATE_FEATURE_NAMES):
        raise ValueError("context/item feature shapes do not match the declared schema")
    if indices.shape != (len(item),) or logp.shape != (len(item),) or embedding.ndim != 2:
        raise ValueError("candidate identity/reference/embedding shapes do not align")
    counts = np.asarray(rm_fit_counts, dtype=np.int64)[indices]
    missing = len(tokens) == 0 or not np.any(tokens >= 3)
    valid_tokens = tokens[tokens >= 3]
    if missing:
        sim_last = np.zeros(len(item), dtype=np.float32)
        sim_mean = np.zeros(len(item), dtype=np.float32)
    else:
        vectors = embedding[indices]
        last_vector = embedding[int(valid_tokens[-1]) - 3]
        history_vectors = embedding[valid_tokens[-20:] - 3]
        mean_vector = history_vectors.mean(axis=0)
        norms = np.maximum(np.linalg.norm(vectors, axis=1), 1e-8)
        last_norm = max(float(np.linalg.norm(last_vector)), 1e-8)
        mean_norm = max(float(np.linalg.norm(mean_vector)), 1e-8)
        sim_last = ((vectors @ last_vector) / (norms * last_norm)).astype(np.float32)
        sim_mean = ((vectors @ mean_vector) / (norms * mean_norm)).astype(np.float32)
    support = np.column_stack((
        counts.astype(np.float32), (counts == 0).astype(np.float32),
        ((counts >= 1) & (counts < 10)).astype(np.float32),
        ((counts >= 10) & (counts < 100)).astype(np.float32),
        (counts >= 100).astype(np.float32),
    ))
    extras = np.column_stack((
        support,
        logp - float(logp.mean()),
        np.arange(len(item), dtype=np.float32) / max(len(item) - 1, 1),
        sim_last, sim_mean, np.full(len(item), float(missing), dtype=np.float32),
    ))
    repeated_context = np.repeat(context[None, :], len(item), axis=0)
    result = np.column_stack((repeated_context, item, extras)).astype(np.float32, copy=False)
    if result.shape[1] != len(POLICY_FEATURE_NAMES) or not np.isfinite(result).all():
        raise AssertionError("policy feature vector is not finite or does not match the schema")
    return result
