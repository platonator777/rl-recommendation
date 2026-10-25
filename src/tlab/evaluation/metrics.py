from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Sequence


def rank_target(
    scores: Sequence[float],
    target_index: int | None,
    item_ids: Sequence[int] | None = None,
) -> int | None:
    """One-based full-catalog rank with score-descending/item-ID-ascending ties."""
    if target_index is None or target_index < 0 or target_index >= len(scores):
        return None
    if item_ids is None:
        item_ids = list(range(len(scores)))
    if len(item_ids) != len(scores):
        raise ValueError("item_ids and scores must have equal length")
    target_score = float(scores[target_index])
    target_id = int(item_ids[target_index])
    ahead = sum(
        1 for index, score in enumerate(scores)
        if float(score) > target_score or (float(score) == target_score and int(item_ids[index]) < target_id)
    )
    return ahead + 1


def single_target_metrics(rank: int | None, ks: Iterable[int] = (10, 20, 100)) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in ks:
        if k < 1:
            raise ValueError("K must be positive")
        hit = float(rank is not None and rank <= k)
        result[f"recall@{k}"] = hit
        result[f"ndcg@{k}"] = hit / math.log2(rank + 1) if hit else 0.0
    return result


def aggregate_macro_user(
    rows: Iterable[tuple[int, int | None]], ks: Iterable[int] = (10, 20, 100)
) -> dict[str, float]:
    """Macro average over users, with every supplied target retained as an event."""
    per_user: dict[int, list[dict[str, float]]] = defaultdict(list)
    for user_id, rank in rows:
        per_user[int(user_id)].append(single_target_metrics(rank, ks))
    if not per_user:
        return {f"{name}@{k}": float("nan") for name in ("recall", "ndcg") for k in ks}
    user_means = {
        user: {
            name: sum(row[name] for row in event_rows) / len(event_rows)
            for name in event_rows[0]
        }
        for user, event_rows in per_user.items()
    }
    return {
        metric: sum(values[metric] for values in user_means.values()) / len(user_means)
        for metric in next(iter(user_means.values()))
    }


def paired_user_bootstrap(
    model_minus_baseline_by_user: dict[int, float],
    repetitions: int = 2000,
    seed: int = 17,
) -> dict[str, float]:
    if repetitions < 1 or not model_minus_baseline_by_user:
        raise ValueError("bootstrap needs users and at least one repetition")
    users = sorted(model_minus_baseline_by_user)
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sample = [model_minus_baseline_by_user[rng.choice(users)] for _ in users]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    point = sum(model_minus_baseline_by_user.values()) / len(users)
    return {
        "mean_difference": point,
        "ci_95_low": estimates[int(0.025 * (repetitions - 1))],
        "ci_95_high": estimates[int(0.975 * (repetitions - 1))],
        "n_users": len(users),
        "repetitions": repetitions,
    }
