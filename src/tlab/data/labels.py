from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TimedPairEvent:
    uid: int
    item_id: int
    timestamp: int | float


@dataclass(frozen=True)
class LikeAttribution:
    listen: TimedPairEvent
    label: int | None
    reason: str
    lag: float | None = None


def attribute_new_likes(
    listens: Iterable[TimedPairEvent],
    likes: Iterable[TimedPairEvent],
    unlikes: Iterable[TimedPairEvent],
    observation_cutoff: int | float,
    window_seconds: int | float = 1800,
    timestamp_unit_seconds: float = 1.0,
) -> list[LikeAttribution]:
    """Attribute each like to at most its closest earlier listen for that pair.

    A label is masked when its future window is censored, the pair was already
    liked at prediction time, or a like shares the listen timestamp.
    """
    if window_seconds <= 0 or timestamp_unit_seconds <= 0:
        raise ValueError("attribution window and timestamp unit must be positive")
    window_raw = window_seconds / timestamp_unit_seconds
    listens_by_pair: dict[tuple[int, int], list[TimedPairEvent]] = {}
    likes_by_pair: dict[tuple[int, int], list[TimedPairEvent]] = {}
    state_events: dict[tuple[int, int], list[tuple[int | float, bool]]] = {}
    for event in listens:
        listens_by_pair.setdefault((event.uid, event.item_id), []).append(event)
    for event in likes:
        likes_by_pair.setdefault((event.uid, event.item_id), []).append(event)
        state_events.setdefault((event.uid, event.item_id), []).append((event.timestamp, True))
    for event in unlikes:
        state_events.setdefault((event.uid, event.item_id), []).append((event.timestamp, False))

    attributed: dict[tuple[int, int, int | float], tuple[bool, float | None]] = {}
    ambiguous: set[tuple[int, int, int | float]] = set()
    for pair, pair_likes in likes_by_pair.items():
        pair_listens = sorted(listens_by_pair.get(pair, []), key=lambda event: event.timestamp)
        listen_times = [event.timestamp for event in pair_listens]
        pair_likes.sort(key=lambda event: event.timestamp)
        for like in pair_likes:
            insertion = bisect_left(listen_times, like.timestamp)
            if insertion < len(listen_times) and listen_times[insertion] == like.timestamp:
                end = bisect_right(listen_times, like.timestamp)
                for event in pair_listens[insertion:end]:
                    ambiguous.add((event.uid, event.item_id, event.timestamp))
                continue
            if insertion == 0:
                continue
            latest_index = insertion - 1
            latest_time = listen_times[latest_index]
            first_latest = bisect_left(listen_times, latest_time, 0, insertion)
            if latest_index - first_latest > 0:
                for event in pair_listens[first_latest:insertion]:
                    ambiguous.add((event.uid, event.item_id, event.timestamp))
                continue
            listen = pair_listens[latest_index]
            lag = like.timestamp - listen.timestamp
            if 0 < lag <= window_raw:
                attributed[(listen.uid, listen.item_id, listen.timestamp)] = (True, float(lag * timestamp_unit_seconds))

    labels: list[LikeAttribution] = []
    for pair, pair_listens in listens_by_pair.items():
        pair_listens.sort(key=lambda event: event.timestamp)
        pair_states = sorted(state_events.get(pair, []), key=lambda event: event[0])
        state_times = [event[0] for event in pair_states]
        for listen in pair_listens:
            key = (listen.uid, listen.item_id, listen.timestamp)
            if listen.timestamp + window_raw > observation_cutoff:
                labels.append(LikeAttribution(listen, None, "censored_future_window"))
                continue
            if key in ambiguous:
                labels.append(LikeAttribution(listen, None, "same_timestamp_like"))
                continue
            prior_end = bisect_left(state_times, listen.timestamp)
            latest_prior_time = state_times[prior_end - 1] if prior_end else None
            if latest_prior_time is None:
                latest_statuses: set[bool] = set()
            else:
                prior_start = bisect_left(state_times, latest_prior_time, 0, prior_end)
                latest_statuses = {is_like for _, is_like in pair_states[prior_start:prior_end]}
            if len(latest_statuses) > 1:
                labels.append(LikeAttribution(listen, None, "ambiguous_prior_like_state"))
                continue
            if latest_statuses == {True}:
                labels.append(LikeAttribution(listen, None, "already_liked"))
                continue
            positive = attributed.get(key)
            if positive is not None:
                labels.append(LikeAttribution(listen, 1, "attributed_like", positive[1]))
            else:
                labels.append(LikeAttribution(listen, 0, "no_new_like_in_window"))
    return sorted(labels, key=lambda row: (row.listen.uid, row.listen.timestamp, row.listen.item_id))
