from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class RewardScale:
    mean: float
    std: float
    sample_count: int

    def normalize(self, value: float) -> float:
        return (float(value) - self.mean) / max(self.std, 1e-3)


def fit_reward_scale(values: Iterable[float]) -> RewardScale:
    rows = [float(value) for value in values]
    if not rows:
        raise ValueError("cannot fit reward scale on an empty reference sample")
    mean = sum(rows) / len(rows)
    variance = sum((value - mean) ** 2 for value in rows) / len(rows)
    return RewardScale(mean=mean, std=variance**0.5, sample_count=len(rows))


def reward_value(
    variant: str,
    ratio: float,
    seconds_scaled: float,
    like_probability: float | None = None,
    recent_repeat: bool = False,
    repeat_penalty: float = 0.1,
) -> float:
    if variant == "Q1":
        return float(ratio)
    if variant == "Q2":
        return float(seconds_scaled)
    if variant == "Q3":
        if like_probability is None:
            raise ValueError("Q3 requires a usable new-like probability")
        return float(like_probability) + 0.1 * float(ratio)
    if variant == "Q4":
        return float(ratio) - repeat_penalty * int(recent_repeat)
    raise ValueError(f"Unknown reward variant: {variant}")


def reward_scale_manifest(scales: dict[str, RewardScale]) -> dict[str, dict[str, float | int]]:
    return {name: asdict(scale) for name, scale in scales.items()}
