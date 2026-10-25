from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def paired_user_bootstrap(
    values_a: Mapping[int, float],
    values_b: Mapping[int, float],
    *,
    n_bootstrap: int = 3000,
    seed: int = 17017,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Paired bootstrap over users; repeated events/seeds are not extra users."""
    if n_bootstrap < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap count or confidence")
    users = sorted(set(values_a).intersection(values_b))
    if not users:
        raise ValueError("paired bootstrap has no shared users")
    differences = np.asarray([float(values_a[user]) - float(values_b[user]) for user in users], dtype=np.float64)
    if not np.all(np.isfinite(differences)):
        raise ValueError("user-level values must be finite")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(users), size=len(users))
        draws[index] = differences[sample].mean()
    alpha = 1.0 - confidence
    return {
        "n_users": len(users),
        "mean_difference": float(differences.mean()),
        "ci_lower": float(np.quantile(draws, alpha / 2.0)),
        "ci_upper": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "one_sided_lower": float(np.quantile(draws, alpha)),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "user_differences": {str(uid): float(value) for uid, value in zip(users, differences)},
    }


def relative_quality_guard(
    new_by_user: Mapping[int, float],
    baseline_by_user: Mapping[int, float],
    *,
    tolerance: float = 0.02,
    n_bootstrap: int = 3000,
    seed: int = 17017,
) -> dict[str, object]:
    if tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("tolerance must be in [0, 1)")
    users = sorted(set(new_by_user).intersection(baseline_by_user))
    adjusted = {uid: float(new_by_user[uid]) - (1.0 - tolerance) * float(baseline_by_user[uid]) for uid in users}
    if not adjusted:
        raise ValueError("quality guard has no shared users")
    zero = {uid: 0.0 for uid in users}
    interval = paired_user_bootstrap(adjusted, zero, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "tolerance": tolerance,
        "per_user_quantity": "new - (1 - tolerance) * baseline",
        "criterion": "one-sided 95% user-cluster bootstrap lower bound > 0",
        "lower_bound": interval["one_sided_lower"],
        "passes": bool(interval["one_sided_lower"] > 0.0),
        "interval": interval,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values in the original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite vector in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()
