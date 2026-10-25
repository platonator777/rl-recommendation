from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_SCHEMA_VERSION = "panel_tools_asof_candidate_v1"


@dataclass(frozen=True)
class CandidateFeatureBatch:
    values: np.ndarray
    names: tuple[str, ...]
    schema_version: str = FEATURE_SCHEMA_VERSION


def _count_lookup(candidate_ids: np.ndarray, history_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions: dict[int, list[int]] = {}
    for index, item_id in enumerate(history_ids.tolist()):
        if int(item_id) >= 0:
            positions.setdefault(int(item_id), []).append(index)
    counts = np.zeros(len(candidate_ids), dtype=np.float32)
    last = np.full(len(candidate_ids), -1, dtype=np.int64)
    for row, item_id in enumerate(candidate_ids.tolist()):
        found = positions.get(int(item_id))
        if found:
            counts[row] = len(found)
            last[row] = found[-1]
    return counts, last


def candidate_history_features(
    *,
    history_item_ids: np.ndarray,
    history_ratios: np.ndarray,
    candidate_item_ids: np.ndarray,
    duration_seconds: np.ndarray,
    train_popularity: np.ndarray,
    train_ratio_mean: np.ndarray,
    decision_timestamp: float | None = None,
    history_timestamps: np.ndarray | None = None,
    candidate_artist_ids: np.ndarray | None = None,
    history_artist_ids: np.ndarray | None = None,
    windows: tuple[int, ...] = (200, 1000),
    smoothing_mass: float = 10.0,
    early_train_ratio_mean: float = 0.5,
) -> CandidateFeatureBatch:
    """Build candidate features from strictly prior history and train-only priors.

    The current target's feedback is not an argument. When timestamps are supplied,
    strict temporal availability is enforced before any feature is computed.
    Artist arrays are optional; absence is represented with explicit missing flags.
    """
    history_ids = np.asarray(history_item_ids, dtype=np.int64)
    ratios = np.asarray(history_ratios, dtype=np.float64)
    candidates = np.asarray(candidate_item_ids, dtype=np.int64)
    durations = np.asarray(duration_seconds, dtype=np.float64)
    popularity = np.asarray(train_popularity, dtype=np.float64)
    prior_means = np.asarray(train_ratio_mean, dtype=np.float64)
    n = len(candidates)
    if history_ids.ndim != 1 or ratios.shape != history_ids.shape:
        raise ValueError("history IDs and ratios must be matching vectors")
    if candidates.ndim != 1 or any(array.shape != candidates.shape for array in (durations, popularity, prior_means)):
        raise ValueError("candidate metadata arrays must align with candidate IDs")
    if len(set(candidates.tolist())) != n:
        raise ValueError("candidate item IDs must be unique")
    if smoothing_mass < 0 or not np.isfinite(smoothing_mass):
        raise ValueError("smoothing_mass must be finite and non-negative")
    if history_timestamps is not None:
        times = np.asarray(history_timestamps, dtype=np.float64)
        if times.shape != history_ids.shape or decision_timestamp is None:
            raise ValueError("history timestamps require a matching vector and decision timestamp")
        if np.any(~np.isfinite(times)) or np.any(times >= float(decision_timestamp)):
            raise ValueError("history features include a simultaneous or future event")
    elif decision_timestamp is not None:
        raise ValueError("decision_timestamp requires history timestamps")

    names: list[str] = []
    columns: list[np.ndarray] = []
    all_count, all_last = _count_lookup(candidates, history_ids)
    for width in windows:
        if width < 1:
            raise ValueError("history windows must be positive")
        window_ids = history_ids[-width:]
        window_ratios = ratios[-width:]
        offset = len(history_ids) - len(window_ids)
        count, last = _count_lookup(candidates, window_ids)
        names.extend((f"item_count_{width}", f"item_recency_steps_{width}", f"item_ratio_mean_{width}",
                      f"item_ratio_count_{width}", f"item_ratio_missing_{width}"))
        columns.extend((
            count,
            np.where(last >= 0, len(window_ids) - 1 - last, len(window_ids)).astype(np.float32),
        ))
        item_ratio_sum = np.zeros(n, dtype=np.float64)
        item_ratio_count = np.zeros(n, dtype=np.float64)
        candidate_index = {int(item): index for index, item in enumerate(candidates.tolist())}
        for item, ratio in zip(window_ids.tolist(), window_ratios.tolist()):
            index = candidate_index.get(int(item))
            if index is not None and np.isfinite(ratio):
                item_ratio_sum[index] += ratio
                item_ratio_count[index] += 1
        means = np.divide(item_ratio_sum, item_ratio_count, out=np.zeros(n, dtype=np.float64), where=item_ratio_count > 0)
        columns.extend((means.astype(np.float32), item_ratio_count.astype(np.float32), (item_ratio_count == 0).astype(np.float32)))

        if history_timestamps is not None:
            window_times = times[-width:]
            time_by_item: dict[int, float] = {}
            for item, timestamp in zip(window_ids.tolist(), window_times.tolist()):
                if int(item) >= 0:
                    time_by_item[int(item)] = float(timestamp)
            age = np.asarray([
                max(0.0, float(decision_timestamp) - time_by_item[int(item)]) if int(item) in time_by_item else np.nan
                for item in candidates.tolist()
            ], dtype=np.float64)
        else:
            age = np.full(n, np.nan, dtype=np.float64)
        names.extend((f"item_age_seconds_{width}", f"item_age_seconds_missing_{width}"))
        columns.extend((np.nan_to_num(age, nan=0.0).astype(np.float32), (~np.isfinite(age)).astype(np.float32)))

        if candidate_artist_ids is not None and history_artist_ids is not None:
            candidate_artists = np.asarray(candidate_artist_ids, dtype=np.int64)
            all_history_artists = np.asarray(history_artist_ids, dtype=np.int64)
            if candidate_artists.shape != candidates.shape or all_history_artists.shape != history_ids.shape:
                raise ValueError("artist IDs must align with candidate and history IDs")
            window_artists = all_history_artists[-width:]
            artist_counts: dict[int, int] = {}
            artist_ratio_sum: dict[int, float] = {}
            artist_ratio_count: dict[int, int] = {}
            last_artist = -1
            for artist, ratio in zip(window_artists.tolist(), window_ratios.tolist()):
                if int(artist) >= 0:
                    artist_counts[int(artist)] = artist_counts.get(int(artist), 0) + 1
                    last_artist = int(artist)
                    if np.isfinite(ratio):
                        artist_ratio_sum[int(artist)] = artist_ratio_sum.get(int(artist), 0.0) + float(ratio)
                        artist_ratio_count[int(artist)] = artist_ratio_count.get(int(artist), 0) + 1
            a_count = np.asarray([artist_counts.get(int(a), 0) if a >= 0 else 0 for a in candidate_artists], dtype=np.float32)
            a_seen = np.asarray([a >= 0 and int(a) in artist_counts for a in candidate_artists], dtype=bool)
            a_ratio_count = np.asarray([artist_ratio_count.get(int(a), 0) if a >= 0 else 0 for a in candidate_artists], dtype=np.float32)
            a_ratio = np.asarray([
                artist_ratio_sum.get(int(a), 0.0) / artist_ratio_count[int(a)] if a >= 0 and int(a) in artist_ratio_count else 0.0
                for a in candidate_artists
            ], dtype=np.float32)
            names.extend((f"artist_count_{width}", f"artist_seen_{width}", f"artist_ratio_mean_{width}",
                          f"artist_ratio_count_{width}", f"artist_ratio_missing_{width}", f"same_as_last_artist_{width}"))
            columns.extend((a_count, a_seen.astype(np.float32), a_ratio, a_ratio_count,
                            (a_ratio_count == 0).astype(np.float32), (candidate_artists == last_artist).astype(np.float32)))

    if candidate_artist_ids is None or history_artist_ids is None:
        for width in windows:
            names.extend((f"artist_features_missing_{width}",))
            columns.append(np.ones(n, dtype=np.float32))

    duration_missing = ~np.isfinite(durations) | (durations < 0)
    prior_missing = ~np.isfinite(prior_means)
    smoothed = np.where(
        prior_missing,
        float(early_train_ratio_mean),
        (prior_means * np.maximum(popularity, 0.0) + float(smoothing_mass) * float(early_train_ratio_mean))
        / (np.maximum(popularity, 0.0) + float(smoothing_mass)),
    )
    names.extend(("log_duration", "duration_missing", "log_train_popularity", "smoothed_early_train_ratio", "early_train_ratio_missing"))
    columns.extend((
        np.log1p(np.where(duration_missing, 0.0, durations)).astype(np.float32),
        duration_missing.astype(np.float32),
        np.log1p(np.maximum(np.nan_to_num(popularity, nan=0.0), 0.0)).astype(np.float32),
        smoothed.astype(np.float32),
        prior_missing.astype(np.float32),
    ))
    return CandidateFeatureBatch(np.column_stack(columns).astype(np.float32, copy=False), tuple(names))
