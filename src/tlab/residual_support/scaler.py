from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def _log_columns(names: Sequence[str]) -> tuple[int, ...]:
    prefixes = ("personal_repeat_count_", "personal_recency_steps_")
    exact = {"past_ratio_count_50", "past_ratio_count_200", "past_ratio_count_1000",
             "unique_personal_items_1000", "train_popularity_count", "train_ratio_prior_count",
             "rm_fit_item_count"}
    return tuple(i for i, name in enumerate(names) if name in exact or name.startswith(prefixes))


@dataclass(frozen=True)
class TrainOnlyFeatureScaler:
    feature_names: tuple[str, ...]
    log1p_columns: tuple[int, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    constant_columns: tuple[int, ...]
    fit_contexts: int
    fit_candidate_rows: int
    protocol: str = "fit on policy-train users only; log1p selected nonnegative counts/steps; standardize by train mean/std"

    @classmethod
    def fit(cls, values: np.ndarray, feature_names: Sequence[str]) -> "TrainOnlyFeatureScaler":
        x = np.asarray(values)
        names = tuple(str(n) for n in feature_names)
        if x.ndim != 3 or x.shape[-1] != len(names) or len(set(names)) != len(names) or not names:
            raise ValueError("scaler input must be [contexts,candidates,unique features]")
        if not np.isfinite(x).all():
            raise ValueError("training features must be finite before scaler fitting")
        log_cols = _log_columns(names)
        means: list[float] = []
        scales: list[float] = []
        constant: list[int] = []
        for j in range(x.shape[-1]):
            column = np.asarray(x[..., j], dtype=np.float64).reshape(-1)
            if j in log_cols:
                if np.any(column < 0):
                    raise ValueError(f"log1p feature {names[j]} contains a negative value")
                column = np.log1p(column)
            mu = float(column.mean())
            sigma = float(column.std())
            if sigma < 1e-8:
                constant.append(j)
                sigma = 1.0
            means.append(mu)
            scales.append(sigma)
        return cls(names, log_cols, tuple(means), tuple(scales), tuple(constant), int(x.shape[0]),
                   int(x.shape[0] * x.shape[1]))

    def transform(self, values: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
        x = np.asarray(values, dtype=np.float32)
        names = tuple(str(n) for n in feature_names)
        if names != self.feature_names:
            raise ValueError("feature names/order do not match the train-fitted scaler")
        if x.shape[-1] != len(names) or x.ndim not in (2, 3):
            raise ValueError("feature array must have [rows,features] or [contexts,candidates,features] shape")
        if not np.isfinite(x).all():
            raise ValueError("features must be finite before scaling")
        result = x.copy()
        for j in self.log1p_columns:
            if np.any(result[..., j] < 0):
                raise ValueError(f"log1p feature {names[j]} contains a negative value")
            result[..., j] = np.log1p(result[..., j])
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        result = (result - mean) / scale
        if not np.isfinite(result).all():
            raise FloatingPointError("scaled features contain non-finite values")
        return result.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
