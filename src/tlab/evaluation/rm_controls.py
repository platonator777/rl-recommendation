from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from tlab.data.store import SequenceStore
from tlab.evaluation.metrics import paired_user_bootstrap


def _features(
    store: SequenceStore,
    examples: Sequence[tuple[int, int]],
    item_features: dict[int, tuple[float, bool] | tuple[float, bool, int]],
    history_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    numeric: list[list[float]] = []
    candidates: list[int] = []
    candidate_numeric: list[list[float]] = []
    targets: list[float] = []
    uids: list[int] = []
    for user_index, position in examples:
        user = store.users[user_index]
        raw_target = float(user.ratio_raw[position])
        if not math.isfinite(raw_target):
            continue
        token = int(user.tokens[position])
        duration, duration_missing, support = item_features.get(token, (0.0, True, 0))
        history = user.history(position, history_length)
        repeat = float(token in history[-20:])
        # This control has no item identity. Support is training popularity;
        # history length and recent repeat are observable activity proxies.
        numeric.append([
            math.log1p(max(0.0, float(duration))),
            float(duration_missing),
            math.log1p(max(0, int(support))),
            repeat,
            math.log1p(max(0, len(history) - 1)),
        ])
        candidates.append(token)
        candidate_numeric.append([
            math.log1p(max(0.0, float(duration))),
            float(duration_missing),
        ])
        targets.append(min(1.0, max(0.0, raw_target)))
        uids.append(int(user.uid))
    return (
        np.asarray(numeric, dtype=np.float32),
        np.asarray(candidates, dtype=np.int64),
        np.asarray(candidate_numeric, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(uids, dtype=np.int64),
    )


def _macro_user_rmse(uids: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> float:
    errors: dict[int, list[float]] = defaultdict(list)
    for uid, target, prediction in zip(uids, targets, predictions):
        errors[int(uid)].append((float(prediction) - float(target)) ** 2)
    return math.sqrt(sum(sum(values) / len(values) for values in errors.values()) / len(errors))


def _paired_user_diff(
    uids: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    per_user: dict[int, list[float]] = defaultdict(list)
    for uid, target, prediction, base in zip(uids, targets, predictions, baseline):
        per_user[int(uid)].append((float(prediction) - float(target)) ** 2 - (float(base) - float(target)) ** 2)
    return paired_user_bootstrap(
        {uid: sum(values) / len(values) for uid, values in per_user.items()},
        repetitions=repetitions,
        seed=seed,
    )


def evaluate_rm_controls(
    train_store: SequenceStore,
    validation_store: SequenceStore,
    train_examples: Sequence[tuple[int, int]],
    validation_examples: Sequence[tuple[int, int]],
    item_features: dict[int, tuple[float, bool] | tuple[float, bool, int]],
    global_mean: float,
    item_mean: dict[int, float],
    history_length: int = 200,
    bootstrap_repetitions: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Fit duration/popularity/activity and candidate-only ratio controls on F_RM."""
    train_numeric, train_candidates, train_candidate_numeric, train_y, _ = _features(
        train_store, train_examples, item_features, history_length,
    )
    val_numeric, val_candidates, val_candidate_numeric, val_y, val_uids = _features(
        validation_store, validation_examples, item_features, history_length,
    )
    if len(train_y) == 0 or len(val_y) == 0:
        raise ValueError("RM controls require finite ratio labels in train and validation")

    item_mean_prediction = np.asarray(
        [item_mean.get(int(token), global_mean) for token in val_candidates], dtype=np.float32,
    )
    numeric_model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    numeric_model.fit(train_numeric, train_y)
    duration_activity_prediction = np.clip(numeric_model.predict(val_numeric), 0.0, 1.0)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    train_ids = encoder.fit_transform(train_candidates.reshape(-1, 1))
    validation_ids = encoder.transform(val_candidates.reshape(-1, 1))
    candidate_train = hstack((train_ids, csr_matrix(train_candidate_numeric)), format="csr", dtype=np.float32)
    candidate_validation = hstack((validation_ids, csr_matrix(val_candidate_numeric)), format="csr", dtype=np.float32)
    selected_alpha = None
    selected_prediction = None
    selected_score = float("inf")
    alpha_scores: dict[str, float] = {}
    for alpha in (1.0, 10.0, 100.0):
        candidate_model = Ridge(alpha=alpha, solver="lsqr")
        candidate_model.fit(candidate_train, train_y)
        prediction = np.clip(candidate_model.predict(candidate_validation), 0.0, 1.0)
        score = _macro_user_rmse(val_uids, val_y, prediction)
        alpha_scores[str(alpha)] = score
        if score < selected_score:
            selected_alpha, selected_prediction, selected_score = alpha, prediction, score

    controls: dict[str, Any] = {
        "n_train_ratio_labels": int(len(train_y)),
        "n_validation_ratio_labels": int(len(val_y)),
        "validation_users": int(len(np.unique(val_uids))),
        "item_mean_macro_user_rmse": _macro_user_rmse(val_uids, val_y, item_mean_prediction),
        "duration_popularity_activity": {
            "features": ["log_duration", "duration_missing", "log_F_RM_support", "recent_repeat", "history_length"],
            "macro_user_rmse": _macro_user_rmse(val_uids, val_y, duration_activity_prediction),
            "paired_user_mse_difference_vs_item_mean": _paired_user_diff(
                val_uids, val_y, duration_activity_prediction, item_mean_prediction,
                bootstrap_repetitions, seed,
            ),
        },
        "candidate_only": {
            "features": ["candidate_item_id", "log_duration", "duration_missing"],
            "selected_ridge_alpha_on_v_rm": selected_alpha,
            "alpha_macro_user_rmse": alpha_scores,
            "macro_user_rmse": _macro_user_rmse(val_uids, val_y, selected_prediction),
            "paired_user_mse_difference_vs_item_mean": _paired_user_diff(
                val_uids, val_y, selected_prediction, item_mean_prediction,
                bootstrap_repetitions, seed + 1,
            ),
        },
    }
    return controls
