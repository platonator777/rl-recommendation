from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from tlab.panel_tools.data import PanelTarget
from tlab.panel_tools.evaluation import single_target_metrics, target_rank


SUPPORT_GROUP_NAMES = ("unsupported", "rare_1_9", "rare_10_99", "supported_100_plus")


def _group_name(count: int) -> str:
    if count == 0:
        return "unsupported"
    if count < 10:
        return "rare_1_9"
    if count < 100:
        return "rare_10_99"
    return "supported_100_plus"


def _macro(rows: Mapping[int, list[dict[str, float]]], k: int) -> dict[str, Any]:
    if not rows:
        return {f"ndcg@{k}": None, f"recall@{k}": None, "n_users": 0, "n_events": 0, "per_user": {}}
    per_user = {
        str(uid): {metric: float(np.mean([row[metric] for row in values])) for metric in (f"ndcg@{k}", f"recall@{k}")}
        for uid, values in rows.items()
    }
    return {
        f"ndcg@{k}": float(np.mean([row[f"ndcg@{k}"] for row in per_user.values()])),
        f"recall@{k}": float(np.mean([row[f"recall@{k}"] for row in per_user.values()])),
        "n_users": len(per_user), "n_events": sum(len(values) for values in rows.values()), "per_user": per_user,
    }


def evaluate_full_catalog_policy(
    *,
    panel: Sequence[PanelTarget],
    full_reference_logp: np.ndarray,
    support_indices: np.ndarray,
    support_mass: np.ndarray,
    policy_conditional: np.ndarray,
    item_ids: np.ndarray,
    rm_fit_counts: np.ndarray,
    k: int = 10,
) -> dict[str, Any]:
    """Evaluate a fixed-tail policy with full-catalog ranks and user-macro slices."""
    n, catalog_size = full_reference_logp.shape
    support_indices = np.asarray(support_indices, dtype=np.int64)
    support_mass = np.asarray(support_mass, dtype=np.float64)
    q = np.asarray(policy_conditional, dtype=np.float64)
    ids = np.asarray(item_ids, dtype=np.int64)
    counts = np.asarray(rm_fit_counts, dtype=np.int64)
    if n != len(panel) or support_indices.shape != q.shape or support_indices.shape[0] != n:
        raise ValueError("policy cache and panel shapes do not align")
    if support_mass.shape != (n,) or ids.shape != (catalog_size,) or counts.shape != (catalog_size,):
        raise ValueError("catalog/support vectors have incompatible shapes")
    if np.any(~np.isfinite(q)) or np.any(q < 0) or not np.allclose(q.sum(-1), 1.0, atol=2e-5):
        raise ValueError("conditional policy rows must be finite normalized distributions")
    raw_to_index = {int(raw): index for index, raw in enumerate(ids.tolist())}
    slice_names = ["primary_nonorganic", "positive_nonorganic", "positive_new", "positive_repeat", "new", "repeat"]
    slice_names.extend(f"rm_fit_support_{name}" for name in SUPPORT_GROUP_NAMES)
    slice_names.extend(f"positive_new_x_{name}" for name in SUPPORT_GROUP_NAMES)
    slice_names.extend(f"positive_repeat_x_{name}" for name in SUPPORT_GROUP_NAMES)
    accum: dict[str, dict[int, list[dict[str, float]]]] = {name: defaultdict(list) for name in slice_names}
    covered = 0
    target_in_support = 0
    support_probability_total = 0.0
    full_recommendations: list[list[int]] = []
    for row_index, target in enumerate(panel):
        probabilities = np.exp(np.asarray(full_reference_logp[row_index], dtype=np.float64))
        candidate_index = support_indices[row_index]
        probabilities[candidate_index] = support_mass[row_index] * q[row_index]
        if not np.isclose(probabilities.sum(), 1.0, atol=2e-5):
            raise ValueError("fixed-tail policy changed total catalog probability mass")
        rank = target_rank(probabilities, ids, target.raw_item_id)
        metric = single_target_metrics(rank, k)
        accum["primary_nonorganic"][target.uid].append(metric)
        covered += int(target.raw_item_id in raw_to_index)
        item_index = raw_to_index.get(int(target.raw_item_id))
        if item_index is not None and int(item_index) in candidate_index:
            target_in_support += 1
            support_probability_total += float(probabilities[item_index])
        group = _group_name(int(counts[item_index])) if item_index is not None else None
        if target.positive:
            accum["positive_nonorganic"][target.uid].append(metric)
            if target.is_new:
                accum["positive_new"][target.uid].append(metric)
            if target.is_repeat:
                accum["positive_repeat"][target.uid].append(metric)
        if target.is_new:
            accum["new"][target.uid].append(metric)
        if target.is_repeat:
            accum["repeat"][target.uid].append(metric)
        if group is not None:
            accum[f"rm_fit_support_{group}"][target.uid].append(metric)
            if target.positive and target.is_new:
                accum[f"positive_new_x_{group}"][target.uid].append(metric)
            if target.positive and target.is_repeat:
                accum[f"positive_repeat_x_{group}"][target.uid].append(metric)
        order = np.lexsort((ids, -probabilities))[:k]
        full_recommendations.append(order.tolist())
    primary = _macro(accum["primary_nonorganic"], k)
    return {
        "primary_user_macro": primary,
        "slices": {name: _macro(rows, k) for name, rows in accum.items() if name != "primary_nonorganic"},
        "panel_events": len(panel), "panel_users": len({r.uid for r in panel}),
        "target_catalog_coverage": covered / len(panel),
        "target_in_optimized_support_fraction": target_in_support / len(panel),
        "mean_recommended_target_probability_if_supported": support_probability_total / max(target_in_support, 1),
        "support_size": int(support_indices.shape[1]), "top_k": k,
        "per_user_primary_ndcg": {int(uid): float(values[f"ndcg@{k}"]) for uid, values in primary["per_user"].items()},
        "oov_targets": len(panel) - covered,
        "aggregation": "event mean within user, then user macro; each frozen panel has one event per user",
        "tie_break": "probability descending, raw item ID ascending",
        "fixed_tail": "support probability mass is preserved and catalog probabilities outside support are unchanged",
    }


def compact_guard(guard: dict[str, Any]) -> dict[str, Any]:
    interval = guard["interval"]
    return {
        "tolerance": guard["tolerance"], "criterion": guard["criterion"], "passes": guard["passes"],
        "lower_bound": guard["lower_bound"], "n_users": interval["n_users"],
        "mean_adjusted_difference": interval["mean_difference"],
        "ci_lower": interval["ci_lower"], "ci_upper": interval["ci_upper"],
        "confidence": interval["confidence"], "n_bootstrap": interval["n_bootstrap"], "seed": interval["seed"],
    }


def evaluate_score_matrix(*, panel: Sequence[PanelTarget], scores: np.ndarray, item_ids: np.ndarray,
                          rm_fit_counts: np.ndarray, k: int = 10) -> dict[str, Any]:
    """Rank any full-catalog score policy with identical user and slice aggregation."""
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(item_ids, dtype=np.int64)
    support_counts = np.asarray(rm_fit_counts, dtype=np.int64)
    if values.shape != (len(panel), len(ids)) or support_counts.shape != ids.shape:
        raise ValueError("full-catalog score matrix and panel/catalog do not align")
    raw_to_index = {int(raw): index for index, raw in enumerate(ids.tolist())}
    slice_names = ["primary_nonorganic", "positive_nonorganic", "positive_new", "positive_repeat", "new", "repeat"]
    slice_names.extend(f"rm_fit_support_{name}" for name in SUPPORT_GROUP_NAMES)
    slice_names.extend(f"positive_new_x_{name}" for name in SUPPORT_GROUP_NAMES)
    slice_names.extend(f"positive_repeat_x_{name}" for name in SUPPORT_GROUP_NAMES)
    accum: dict[str, dict[int, list[dict[str, float]]]] = {name: defaultdict(list) for name in slice_names}
    covered = 0
    for row_index, target in enumerate(panel):
        rank = target_rank(values[row_index], ids, target.raw_item_id)
        metric = single_target_metrics(rank, k)
        accum["primary_nonorganic"][target.uid].append(metric)
        covered += int(target.raw_item_id in raw_to_index)
        item_index = raw_to_index.get(int(target.raw_item_id))
        group = _group_name(int(support_counts[item_index])) if item_index is not None else None
        if target.positive:
            accum["positive_nonorganic"][target.uid].append(metric)
            if target.is_new:
                accum["positive_new"][target.uid].append(metric)
            if target.is_repeat:
                accum["positive_repeat"][target.uid].append(metric)
        if target.is_new:
            accum["new"][target.uid].append(metric)
        if target.is_repeat:
            accum["repeat"][target.uid].append(metric)
        if group is not None:
            accum[f"rm_fit_support_{group}"][target.uid].append(metric)
            if target.positive and target.is_new:
                accum[f"positive_new_x_{group}"][target.uid].append(metric)
            if target.positive and target.is_repeat:
                accum[f"positive_repeat_x_{group}"][target.uid].append(metric)
    primary = _macro(accum["primary_nonorganic"], k)
    return {
        "primary_user_macro": primary,
        "slices": {name: _macro(rows, k) for name, rows in accum.items() if name != "primary_nonorganic"},
        "panel_events": len(panel), "panel_users": len({r.uid for r in panel}),
        "target_catalog_coverage": covered / len(panel), "oov_targets": len(panel) - covered,
        "top_k": k, "aggregation": "event mean within user, then user macro; OOV targets remain misses",
        "tie_break": "score descending, raw item ID ascending",
    }
