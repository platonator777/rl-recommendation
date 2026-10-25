from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from tlab.data.catalog import BOS_ID, OOV_ID


@dataclass(frozen=True)
class ListenEvent:
    uid: int
    timestamp: int | float
    token_id: int
    is_organic: bool
    ratio_raw: float | None = None
    duration_seconds: float | None = None
    split: str = "train"


@dataclass(frozen=True)
class PrefixExample:
    uid: int
    timestamp: int | float
    history: tuple[int, ...]
    target_token: int
    ratio_raw: float | None
    duration_seconds: float | None
    split: str


@dataclass
class SequenceDiagnostics:
    events_seen: int = 0
    tie_events: int = 0
    tie_blocks: int = 0
    eligible_targets: int = 0
    targets_without_history: int = 0


def prefix_examples(
    sorted_events: Iterable[ListenEvent],
    history_length: int = 200,
    diagnostics: SequenceDiagnostics | None = None,
) -> Iterator[PrefixExample]:
    """Build strict-past examples; ambiguous timestamp blocks become one OOV token.

    Input must be sorted by ``(uid, timestamp)``. Rows tied at one timestamp
    never predict one another and are represented by a single OOV marker in the
    following context.
    """
    if history_length < 1:
        raise ValueError("history_length must be positive")
    stats = diagnostics if diagnostics is not None else SequenceDiagnostics()
    current_uid: int | None = None
    current_timestamp: int | float | None = None
    timestamp_group: list[ListenEvent] = []
    history: list[int] = []

    def finish_group(group: list[ListenEvent], history_tokens: list[int]) -> Iterator[PrefixExample]:
        if not group:
            return
        if len(group) > 1:
            stats.tie_blocks += 1
            stats.tie_events += len(group)
            history_tokens.append(OOV_ID)
            return
        event = group[0]
        if event.split == "train" and history_tokens and not event.is_organic and event.token_id >= 3:
            stats.eligible_targets += 1
            yield PrefixExample(
                uid=event.uid,
                timestamp=event.timestamp,
                history=tuple([BOS_ID, *history_tokens[-(history_length - 1):]] if history_length > 1 else [BOS_ID]),
                target_token=event.token_id,
                ratio_raw=event.ratio_raw,
                duration_seconds=event.duration_seconds,
                split=event.split,
            )
        elif event.split == "train" and not history_tokens:
            stats.targets_without_history += int(not event.is_organic and event.token_id >= 3)
        history_tokens.append(event.token_id if event.token_id >= 3 else OOV_ID)

    for event in sorted_events:
        stats.events_seen += 1
        if current_uid is None:
            current_uid = event.uid
            current_timestamp = event.timestamp
        elif event.uid < current_uid or (event.uid == current_uid and event.timestamp < current_timestamp):
            raise ValueError("input events must be sorted by uid and timestamp")
        if event.uid != current_uid:
            yield from finish_group(timestamp_group, history)
            current_uid = event.uid
            current_timestamp = event.timestamp
            timestamp_group = []
            history = []
        elif event.timestamp != current_timestamp:
            yield from finish_group(timestamp_group, history)
            timestamp_group = []
            current_timestamp = event.timestamp
        timestamp_group.append(event)
    yield from finish_group(timestamp_group, history)
