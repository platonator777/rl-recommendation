from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np


def duration_quantile_from_train(store, quantile: float) -> float | None:
    """Quantile of observed Train-event durations; validation/test never set bins."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    chunks: list[np.ndarray] = []
    for user in store.users:
        durations = np.frombuffer(user.duration_seconds, dtype=np.float32)
        splits = np.frombuffer(user.split_codes, dtype=np.uint8)
        valid = (splits == 0) & np.isfinite(durations) & (durations > 0)
        if valid.any():
            chunks.append(durations[valid])
    if not chunks:
        return None
    return float(np.quantile(np.concatenate(chunks), quantile))

def gini_including_zeros(counts: Sequence[int]) -> float:
    """Gini coefficient over a fixed-width catalog count vector, including zeros."""
    if not counts:
        return 0.0
    values = sorted(max(0, int(value)) for value in counts)
    total = sum(values)
    n = len(values)
    if total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(values))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def exposure_metrics(recommendations: Iterable[Sequence[int]], catalog_size: int, top_k: int = 10) -> dict[str, float]:
    if catalog_size < 1 or top_k < 1:
        raise ValueError("catalog_size and top_k must be positive")
    rows = [list(items[:top_k]) for items in recommendations]
    counts = Counter(item for row in rows for item in row)
    total = sum(counts.values())
    probabilities = [counts.get(item, 0) / total for item in range(catalog_size)] if total else [0.0] * catalog_size
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    normalized_entropy = entropy / math.log(catalog_size) if catalog_size > 1 else 0.0
    return {
        "coverage": len([item for item in counts if 0 <= item < catalog_size]) / catalog_size,
        "exposure_entropy": normalized_entropy,
        "exposure_gini": gini_including_zeros([counts.get(item, 0) for item in range(catalog_size)]),
        "n_recommendations": total,
    }


def repeat_rate(recommendations: Iterable[Sequence[int]], recent_histories: Iterable[set[int]], top_k: int = 10) -> float:
    numerator = denominator = 0
    for items, history in zip(recommendations, recent_histories):
        for item in items[:top_k]:
            numerator += int(item in history)
            denominator += 1
    return numerator / denominator if denominator else 0.0
