from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


def target_rank(scores: Sequence[float], item_ids: Sequence[int], target_id: int) -> int | None:
    """One-based rank with score descending then raw item ID ascending ties."""
    score_array = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(item_ids, dtype=np.int64)
    if score_array.ndim != 1 or ids.ndim != 1 or score_array.shape != ids.shape:
        raise ValueError("scores and item_ids must be same-length vectors")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("ranking scores must be finite")
    matches = np.flatnonzero(ids == int(target_id))
    if matches.size == 0:
        return None
    if matches.size > 1:
        raise ValueError("catalog item IDs must be unique")
    index = int(matches[0])
    score = score_array[index]
    target = ids[index]
    ahead = (score_array > score) | ((score_array == score) & (ids < target))
    return int(ahead.sum()) + 1


def single_target_metrics(rank: int | None, k: int = 10) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive")
    if rank is None or rank > k:
        return {f"ndcg@{k}": 0.0, f"recall@{k}": 0.0}
    return {f"ndcg@{k}": 1.0 / math.log2(rank + 1.0), f"recall@{k}": 1.0}


def evaluate_rank_rows(
    rows: Iterable[Mapping[str, object]], *, k: int = 10
) -> dict[str, object]:
    """Aggregate event metrics within user, then user-macro across users.

    Each input row contains ``uid``, ``scores``, ``item_ids``, ``target_id`` and may
    contain Boolean diagnostic keys such as ``positive``, ``is_new``, ``is_repeat``.
    Missing catalog targets remain zero-valued misses in the primary denominator.
    """
    by_user: dict[int, list[dict[str, float]]] = defaultdict(list)
    slices: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    n_covered = 0
    n_events = 0
    for row in rows:
        uid = int(row["uid"])
        rank = target_rank(row["scores"], row["item_ids"], int(row["target_id"]))
        metrics = single_target_metrics(rank, k)
        by_user[uid].append(metrics)
        n_events += 1
        n_covered += int(rank is not None)
        for name in ("positive", "is_new", "is_repeat", "known_catalog"):
            if bool(row.get(name, False)):
                slices[name][uid].append(metrics)

    def macro(source: Mapping[int, list[dict[str, float]]]) -> dict[str, object]:
        if not source:
            return {f"ndcg@{k}": None, f"recall@{k}": None, "n_users": 0, "n_events": 0}
        per_user = {
            uid: {name: float(np.mean([item[name] for item in values])) for name in (f"ndcg@{k}", f"recall@{k}")}
            for uid, values in source.items()
        }
        return {
            **{name: float(np.mean([metrics[name] for metrics in per_user.values()])) for name in (f"ndcg@{k}", f"recall@{k}")},
            "n_users": len(per_user),
            "n_events": sum(map(len, source.values())),
            "per_user": {str(uid): metrics for uid, metrics in per_user.items()},
        }

    return {
        "k": k,
        "n_events": n_events,
        "n_users": len(by_user),
        "target_coverage": n_covered / n_events if n_events else None,
        "primary_user_macro": macro(by_user),
        "slices": {name: macro(values) for name, values in slices.items()},
        "aggregation": "mean events within user, then mean across users",
        "oov_targets_are_misses": True,
    }
