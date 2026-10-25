from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tlab.config import load_config
from tlab.data.manifest import write_json_atomic
from tlab.evaluation.metrics import paired_user_bootstrap

CONTINUATION_WINDOWS_SECONDS = (10 * 60, 30 * 60, 60 * 60)
RETURN_LOWER_SECONDS = 24 * 60 * 60
RETURN_UPPER_SECONDS = 7 * 24 * 60 * 60
SESSION_GAP_SECONDS = 30 * 60


@dataclass(frozen=True)
class _Event:
    timestamp: float
    token_id: int
    is_organic: bool
    tie_count: int
    train_subsplit: str | None
    duration_seconds: float | None


@dataclass(frozen=True)
class _Example:
    uid: int
    activity: tuple[float, float, float, float]
    token_id: int
    duration_log: float
    recent_repeat: float
    continuation: tuple[int, int, int]
    return_7d: int
    organic_transition_60m: bool


class _Reservoir:
    def __init__(self, limit: int, seed: int):
        self.limit = limit
        self.rng = random.Random(seed)
        self.seen = 0
        self.rows: list[_Example] = []

    def add(self, row: _Example) -> None:
        self.seen += 1
        if len(self.rows) < self.limit:
            self.rows.append(row)
            return
        index = self.rng.randrange(self.seen)
        if index < self.limit:
            self.rows[index] = row


def _future_lookup(
    timestamps: Sequence[float],
    has_nonorganic: Sequence[bool],
    has_organic: Sequence[bool],
) -> tuple[list[float | None], list[float | None], list[float]]:
    """For each timestamp block, find the next strictly later organic/non-organic event."""
    if not (len(timestamps) == len(has_nonorganic) == len(has_organic)):
        raise ValueError("timestamp and event-kind arrays must have equal lengths")
    next_nonorganic: list[float | None] = [None] * len(timestamps)
    next_organic: list[float | None] = [None] * len(timestamps)
    following_nonorganic: float | None = None
    following_organic: float | None = None
    for index in range(len(timestamps) - 1, -1, -1):
        next_nonorganic[index] = following_nonorganic
        next_organic[index] = following_organic
        if has_nonorganic[index]:
            following_nonorganic = timestamps[index]
        if has_organic[index]:
            following_organic = timestamps[index]
    session_starts = [
        timestamps[index]
        for index in range(1, len(timestamps))
        if timestamps[index] - timestamps[index - 1] > SESSION_GAP_SECONDS
    ]
    return next_nonorganic, next_organic, session_starts


def _future_labels(
    timestamp: float,
    next_nonorganic: float | None,
    next_organic: float | None,
    session_starts: Sequence[float],
    observation_cutoff: float,
) -> tuple[tuple[int, int, int], int, bool] | None:
    """Return fully observed labels; ``None`` masks a censored seven-day horizon."""
    if timestamp + RETURN_UPPER_SECONDS >= observation_cutoff:
        return None
    continuation = tuple(
        int(next_nonorganic is not None and next_nonorganic <= timestamp + window)
        for window in CONTINUATION_WINDOWS_SECONDS
    )
    organic_transition = bool(
        next_organic is not None
        and next_organic <= timestamp + CONTINUATION_WINDOWS_SECONDS[-1]
        and (next_nonorganic is None or next_organic < next_nonorganic)
    )
    return_start = bisect.bisect_right(session_starts, timestamp + RETURN_LOWER_SECONDS)
    return_end = bisect.bisect_right(session_starts, timestamp + RETURN_UPPER_SECONDS)
    returned = int(return_start < return_end)
    return continuation, returned, organic_transition


def _history_features(
    timestamps: Sequence[float],
    event_counts: Sequence[int],
    index: int,
) -> tuple[float, float, float, float]:
    """Compute only-past activity counts and recency for one timestamp block."""
    if index <= 0:
        raise ValueError("a continuation example requires a prior listen")
    current = timestamps[index]
    prefix = [0]
    for count in event_counts[:index]:
        prefix.append(prefix[-1] + count)
    left_1d = bisect.bisect_left(timestamps, current - 24 * 60 * 60, 0, index)
    left_7d = bisect.bisect_left(timestamps, current - 7 * 24 * 60 * 60, 0, index)
    count_1d = prefix[index] - prefix[left_1d]
    count_7d = prefix[index] - prefix[left_7d]
    return (
        math.log1p(count_1d),
        math.log1p(count_7d),
        math.log1p(prefix[index]),
        math.log1p(max(0.0, current - timestamps[index - 1])),
    )


def _examples_for_user(
    uid: int,
    events: Sequence[_Event],
    train_target_cutoff: float,
    validation_start: float,
    validation_target_cutoff: float,
    observation_cutoff: float,
) -> tuple[list[_Example], list[_Example], dict[str, int]]:
    """Create matured F_RM examples, reserving a later temporal block for validation."""
    groups: list[list[_Event]] = []
    for event in events:
        if groups and groups[-1][0].timestamp == event.timestamp:
            groups[-1].append(event)
        else:
            groups.append([event])
    timestamps = [group[0].timestamp for group in groups]
    event_counts = [len(group) for group in groups]
    has_nonorganic = [any(not event.is_organic for event in group) for group in groups]
    has_organic = [any(event.is_organic for event in group) for group in groups]
    next_nonorganic, next_organic, session_starts = _future_lookup(
        timestamps, has_nonorganic, has_organic,
    )
    output_train: list[_Example] = []
    output_validation: list[_Example] = []
    diagnostics = {"matured_targets": 0, "censored_targets": 0, "organic_transition_60m": 0}
    recent_items: deque[int] = deque(maxlen=20)
    prefix_counts = [0]
    for count in event_counts:
        prefix_counts.append(prefix_counts[-1] + count)
    left_1d = 0
    left_7d = 0

    for index, group in enumerate(groups):
        timestamp = timestamps[index]
        while left_1d < index and timestamp - timestamps[left_1d] > 24 * 60 * 60:
            left_1d += 1
        while left_7d < index and timestamp - timestamps[left_7d] > 7 * 24 * 60 * 60:
            left_7d += 1
        singleton_action = (
            index > 0
            and len(group) == 1
            and not group[0].is_organic
            and group[0].tie_count == 1
            and group[0].token_id >= 3
            and group[0].train_subsplit == "F_RM"
        )
        split_name: str | None = None
        if timestamp < train_target_cutoff:
            split_name = "train"
            label_cutoff = validation_start
        elif validation_start <= timestamp < validation_target_cutoff:
            split_name = "validation"
            label_cutoff = observation_cutoff

        if singleton_action and split_name is not None:
            labels = _future_labels(
                timestamp, next_nonorganic[index], next_organic[index], session_starts, label_cutoff,
            )
            if labels is None:
                diagnostics["censored_targets"] += 1
            else:
                continuation, return_7d, organic_transition = labels
                event = group[0]
                duration = event.duration_seconds
                duration_log = math.log1p(duration) if duration is not None and math.isfinite(duration) and duration > 0 else 0.0
                activity = (
                    math.log1p(prefix_counts[index] - prefix_counts[left_1d]),
                    math.log1p(prefix_counts[index] - prefix_counts[left_7d]),
                    math.log1p(prefix_counts[index]),
                    math.log1p(max(0.0, timestamp - timestamps[index - 1])),
                )
                example = _Example(
                    uid=uid,
                    activity=activity,
                    token_id=event.token_id,
                    duration_log=duration_log,
                    recent_repeat=float(event.token_id in recent_items),
                    continuation=continuation,
                    return_7d=return_7d,
                    organic_transition_60m=organic_transition,
                )
                (output_train if split_name == "train" else output_validation).append(example)
                diagnostics["matured_targets"] += 1
                diagnostics["organic_transition_60m"] += int(organic_transition)

        # Match the sequence contract: an ambiguous timestamp block is one OOV
        # history symbol, never an arbitrary ordering of its raw item IDs.
        if len(group) == 1 and group[0].tie_count == 1 and group[0].token_id >= 3:
            recent_items.append(group[0].token_id)
        else:
            recent_items.append(-1)

    return output_train, output_validation, diagnostics


def _iter_users(
    path: Path,
    observation_cutoff_raw: float,
    timestamp_unit_seconds: float,
) -> Iterable[tuple[int, list[_Event]]]:
    """Stream the prepared, uid/timestamp-sorted Parquet file one user at a time."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = ["uid", "timestamp", "token_id", "is_organic", "timestamp_tie_count", "train_subsplit", "track_length_seconds"]
    current_uid: int | None = None
    current: list[_Event] = []
    previous_key: tuple[int, float] | None = None
    for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
        rows = [column.to_pylist() for column in batch.columns]
        for uid_raw, timestamp_raw, token_raw, organic_raw, ties_raw, subsplit_raw, duration_raw in zip(*rows):
            uid = int(uid_raw)
            timestamp = float(timestamp_raw)
            key = (uid, timestamp)
            if previous_key is not None and key < previous_key:
                raise ValueError("prepared listens must be sorted by uid and timestamp")
            previous_key = key
            if current_uid is not None and uid != current_uid:
                yield current_uid, current
                current = []
            current_uid = uid
            if timestamp < observation_cutoff_raw:
                duration = None if duration_raw is None else float(duration_raw)
                current.append(_Event(
                    timestamp=timestamp * timestamp_unit_seconds,
                    token_id=int(token_raw),
                    is_organic=bool(organic_raw),
                    tie_count=int(ties_raw or 0),
                    train_subsplit=None if subsplit_raw is None else str(subsplit_raw),
                    duration_seconds=duration,
                ))
    if current_uid is not None:
        yield current_uid, current


def _calibration_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    if labels.size == 0:
        return []
    order = np.argsort(probabilities, kind="stable")
    result: list[dict[str, float | int]] = []
    for indices in np.array_split(order, min(bins, len(order))):
        if len(indices):
            result.append({
                "count": int(len(indices)),
                "mean_predicted": float(probabilities[indices].mean()),
                "observed_rate": float(labels[indices].mean()),
            })
    return result


def _prediction_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    result: dict[str, Any] = {
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if labels.size else None,
        "mean_prediction": float(probabilities.mean()) if labels.size else None,
        "brier": float(brier_score_loss(labels, probabilities)) if labels.size else None,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])) if labels.size else None,
        "average_precision": float(average_precision_score(labels, probabilities)) if labels.size and labels.sum() else None,
        "calibration_deciles": _calibration_bins(labels, probabilities),
    }
    if labels.size:
        result["expected_calibration_error_deciles"] = float(sum(
            row["count"] / labels.size * abs(row["mean_predicted"] - row["observed_rate"])
            for row in result["calibration_deciles"]
        ))
    return result


def _user_loss_differences(
    uids: Sequence[int],
    labels: np.ndarray,
    activity_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, log_loss

    per_user: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"activity_brier": [], "candidate_brier": [], "activity_log_loss": [], "candidate_log_loss": []})
    activity_log = -(labels * np.log(np.clip(activity_probabilities, 1e-7, 1.0)) + (1 - labels) * np.log(np.clip(1 - activity_probabilities, 1e-7, 1.0)))
    candidate_log = -(labels * np.log(np.clip(candidate_probabilities, 1e-7, 1.0)) + (1 - labels) * np.log(np.clip(1 - candidate_probabilities, 1e-7, 1.0)))
    for uid, y, activity_p, candidate_p, activity_ll, candidate_ll in zip(
        uids, labels, activity_probabilities, candidate_probabilities, activity_log, candidate_log,
    ):
        record = per_user[int(uid)]
        record["activity_brier"].append(float((activity_p - y) ** 2))
        record["candidate_brier"].append(float((candidate_p - y) ** 2))
        record["activity_log_loss"].append(float(activity_ll))
        record["candidate_log_loss"].append(float(candidate_ll))
    differences: dict[str, Any] = {}
    for metric in ("brier", "log_loss"):
        delta = {
            uid: sum(values[f"candidate_{metric}"]) / len(values[f"candidate_{metric}"])
            - sum(values[f"activity_{metric}"]) / len(values[f"activity_{metric}"])
            for uid, values in per_user.items()
        }
        differences[f"candidate_minus_activity_{metric}"] = paired_user_bootstrap(delta, repetitions=2000, seed=17)
    differences["macro_user_brier_activity"] = float(np.mean([
        np.mean(values["activity_brier"]) for values in per_user.values()
    ]))
    differences["macro_user_brier_candidate"] = float(np.mean([
        np.mean(values["candidate_brier"]) for values in per_user.values()
    ]))
    differences["macro_user_log_loss_activity"] = float(np.mean([
        np.mean(values["activity_log_loss"]) for values in per_user.values()
    ]))
    differences["macro_user_log_loss_candidate"] = float(np.mean([
        np.mean(values["candidate_log_loss"]) for values in per_user.values()
    ]))
    return differences


def _fit_predictors(
    train: Sequence[_Example],
    validation: Sequence[_Example],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if not train or not validation:
        raise ValueError("continuation predictors require train and future validation examples")
    x_train = np.asarray([(*row.activity, row.token_id, row.duration_log, row.recent_repeat) for row in train], dtype=np.float64)
    x_validation = np.asarray([(*row.activity, row.token_id, row.duration_log, row.recent_repeat) for row in validation], dtype=np.float64)
    history_columns = [0, 1, 2, 3]
    candidate_numeric_columns = [0, 1, 2, 3, 5, 6]
    candidate_preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), candidate_numeric_columns),
            ("candidate_id", OneHotEncoder(handle_unknown="ignore", min_frequency=10, dtype=np.float64), [4]),
        ],
        sparse_threshold=1.0,
    )
    activity_scaler = StandardScaler()
    activity_train = activity_scaler.fit_transform(x_train[:, history_columns])
    activity_validation = activity_scaler.transform(x_validation[:, history_columns])
    candidate_train = candidate_preprocessor.fit_transform(x_train)
    candidate_validation = candidate_preprocessor.transform(x_validation)
    return activity_train, activity_validation, {
        "candidate_train": candidate_train,
        "candidate_validation": candidate_validation,
        "n_candidate_categories": int(len(candidate_preprocessor.named_transformers_["candidate_id"].categories_[0])),
        "candidate_unknown_validation_fraction": float(np.mean(~np.isin(
            x_validation[:, 4], candidate_preprocessor.named_transformers_["candidate_id"].categories_[0],
        ))),
        "LogisticRegression": LogisticRegression,
        "seed": seed,
    }


def run_continuation_experiment(
    config_path: str | Path = "configs/base.yaml",
    seed: int = 17,
    train_samples: int = 100_000,
    validation_samples: int = 50_000,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if train_samples < 100 or validation_samples < 100:
        raise ValueError("use at least 100 samples for both training and validation")
    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    timestamp_unit = float(protocol["timestamp_unit_seconds"])
    fit_end_raw = float(protocol["boundaries_source_units"]["rm_validation_start_raw"])
    fit_end = fit_end_raw * timestamp_unit
    days14_raw = 14 * 24 * 60 * 60 / timestamp_unit
    validation_start_raw = fit_end_raw - days14_raw
    validation_start = validation_start_raw * timestamp_unit
    train_target_cutoff = validation_start - RETURN_UPPER_SECONDS
    validation_target_cutoff = fit_end - RETURN_UPPER_SECONDS
    if train_target_cutoff <= 0 or validation_target_cutoff <= validation_start:
        raise ValueError("the F_RM interval is too short for a censored seven-day holdout")

    train_reservoir = _Reservoir(train_samples, seed)
    validation_reservoir = _Reservoir(validation_samples, seed + 1)
    diagnostics = {"matured_targets": 0, "censored_targets": 0, "organic_transition_60m": 0}
    processed_path = Path(protocol["processed_listens_path"])
    users_processed = 0
    events_processed = 0
    started_at = time.perf_counter()
    for uid, events in _iter_users(processed_path, fit_end_raw, timestamp_unit):
        users_processed += 1
        events_processed += len(events)
        train_rows, validation_rows, user_diagnostics = _examples_for_user(
            uid, events, train_target_cutoff, validation_start, validation_target_cutoff, fit_end,
        )
        for row in train_rows:
            train_reservoir.add(row)
        for row in validation_rows:
            validation_reservoir.add(row)
        for name, value in user_diagnostics.items():
            diagnostics[name] += value
        if users_processed % 500 == 0:
            elapsed = time.perf_counter() - started_at
            print(
                f"Continuation indexing: {users_processed:,} users, {events_processed:,} pre-cutoff events, "
                f"{train_reservoir.seen:,} train and {validation_reservoir.seen:,} future targets "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    train = train_reservoir.rows
    validation = validation_reservoir.rows
    if len(train) < 100 or len(validation) < 100:
        raise ValueError(f"not enough matured examples: train={len(train)}, validation={len(validation)}")
    activity_train, activity_validation, candidate_data = _fit_predictors(train, validation, seed)
    LogisticRegression = candidate_data.pop("LogisticRegression")
    candidate_train = candidate_data.pop("candidate_train")
    candidate_validation = candidate_data.pop("candidate_validation")
    candidate_unknown_fraction = candidate_data.pop("candidate_unknown_validation_fraction")
    candidate_categories = candidate_data.pop("n_candidate_categories")

    tasks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for offset, window in enumerate(CONTINUATION_WINDOWS_SECONDS):
        name = f"continue_{window // 60}m"
        tasks[name] = (
            np.asarray([row.continuation[offset] for row in train], dtype=np.int64),
            np.asarray([row.continuation[offset] for row in validation], dtype=np.int64),
        )
    tasks["return_24h_to_7d"] = (
        np.asarray([row.return_7d for row in train], dtype=np.int64),
        np.asarray([row.return_7d for row in validation], dtype=np.int64),
    )

    results: dict[str, Any] = {}
    for offset, (task_name, (y_train, y_validation)) in enumerate(tasks.items()):
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"training labels for {task_name} contain only one class")
        activity_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=250, random_state=seed + offset)
        candidate_model = LogisticRegression(C=1.0, solver="saga", max_iter=150, tol=1e-3, random_state=seed + offset)
        activity_model.fit(activity_train, y_train)
        candidate_model.fit(candidate_train, y_train)
        activity_probability = activity_model.predict_proba(activity_validation)[:, 1]
        candidate_probability = candidate_model.predict_proba(candidate_validation)[:, 1]
        validation_uids = [row.uid for row in validation]
        results[task_name] = {
            "activity_only": _prediction_metrics(y_validation, activity_probability),
            "activity_plus_candidate": _prediction_metrics(y_validation, candidate_probability),
            "paired_user_comparison_candidate_minus_activity": _user_loss_differences(
                validation_uids, y_validation, activity_probability, candidate_probability,
            ),
            "training_prevalence": float(y_train.mean()),
            "candidate_model_converged": bool(candidate_model.n_iter_[0] < candidate_model.max_iter),
        }

    organic_transitions = sum(row.organic_transition_60m for row in validation)
    result = {
        "experiment": "continuation and return prediction from observed F_RM actions",
        "seed": seed,
        "candidate_feedback_used": False,
        "candidate_features": ["logged candidate item ID", "logged track duration metadata", "prior-20 repeat flag"],
        "activity_features": ["prior listen count in 1d", "prior listen count in 7d", "prior history length", "seconds since prior listen"],
        "model": "L2 logistic regression; one-hot candidate ID uses min_frequency=10; fixed C=1.0",
        "censoring": {
            "train_target_cutoff_source_units": train_target_cutoff / timestamp_unit,
            "validation_start_source_units": validation_start_raw,
            "validation_target_cutoff_source_units": validation_target_cutoff / timestamp_unit,
            "observation_cutoff_source_units": fit_end_raw,
            "timestamp_unit_seconds": timestamp_unit,
            "maximum_label_horizon_seconds": RETURN_UPPER_SECONDS,
            "train_embargo_seconds": RETURN_UPPER_SECONDS,
            "validation_embargo_seconds": RETURN_UPPER_SECONDS,
        },
        "samples": {
            "train_seen": train_reservoir.seen,
            "validation_seen": validation_reservoir.seen,
            "train_sampled": len(train),
            "validation_sampled": len(validation),
            "validation_users": len({row.uid for row in validation}),
            "candidate_id_categories_train": candidate_categories,
            "candidate_id_unseen_fraction_validation": candidate_unknown_fraction,
        },
        "diagnostics": {
            **diagnostics,
            "validation_organic_first_transition_within_60m": organic_transitions,
            "validation_organic_first_transition_fraction": organic_transitions / len(validation),
        },
        "tasks": results,
    }
    target = Path(output_path) if output_path else root / "artifacts" / "continuation" / f"continuation_seed{seed}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    result["output_path"] = str(target.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal continuation and return predictors on F_RM")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-samples", type=int, default=100_000)
    parser.add_argument("--validation-samples", type=int, default=50_000)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_continuation_experiment(
        args.config, seed=args.seed, train_samples=args.train_samples,
        validation_samples=args.validation_samples, output_path=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
