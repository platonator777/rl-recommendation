from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from tlab.data.store import SPLIT_CODES, SequenceStore


@dataclass(frozen=True)
class PanelTarget:
    event_id: str
    uid: int
    user_index: int
    position: int
    raw_item_id: int
    token_id: int
    ratio: float | None
    positive: bool
    is_new: bool
    is_repeat: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def stable_event_id(source_sha256: str, uid: int, position: int, raw_item_id: int) -> str:
    """Identify a row within a fixed source revision without using its outcome."""
    payload = f"{source_sha256}:{int(uid)}:{int(position)}:{int(raw_item_id)}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _hash_order(seed: int, event_id: str) -> bytes:
    return hashlib.sha256(f"{int(seed)}:{event_id}".encode("ascii")).digest()


def build_evaluation_panel(
    store: SequenceStore,
    *,
    split: Literal["validation", "test"] = "validation",
    targets_per_user: int = 3,
    seed: int = 17,
    source_sha256: str,
) -> list[PanelTarget]:
    """Freeze event targets before scoring, including OOV catalog targets.

    Sampling uses a stable hash of the event ID and is independent of target ratio,
    token/catalog membership, and every model score. ``recommendable_events`` in
    SequenceStore excludes organic and ambiguous timestamp-block rows while retaining
    non-organic targets that map to OOV.
    """
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    if not 1 <= targets_per_user <= 3:
        raise ValueError("the frozen primary panel permits one to three events per user")
    split_code = SPLIT_CODES[split]
    chosen: list[PanelTarget] = []
    for user_index, user in enumerate(store.users):
        split_codes = np.asarray(user.split_codes, dtype=np.uint8)
        eligible = np.asarray(user.recommendable_events, dtype=np.uint8).astype(bool)
        positions = np.flatnonzero((split_codes == split_code) & eligible)
        candidates: list[PanelTarget] = []
        for position_np in positions:
            position = int(position_np)
            raw_item_id = int(user.raw_item_ids[position])
            if raw_item_id < 0:
                continue
            event_id = stable_event_id(source_sha256, user.uid, position, raw_item_id)
            raw_ratio = float(user.ratio_raw[position])
            ratio = raw_ratio if math.isfinite(raw_ratio) else None
            prior_raw = np.asarray(user.raw_item_ids[:position], dtype=np.int64)
            is_repeat = bool(np.any(prior_raw == raw_item_id))
            candidates.append(PanelTarget(
                event_id=event_id,
                uid=int(user.uid),
                user_index=user_index,
                position=position,
                raw_item_id=raw_item_id,
                token_id=int(user.tokens[position]),
                ratio=ratio,
                positive=bool(ratio is not None and ratio >= 0.5),
                is_new=not is_repeat,
                is_repeat=is_repeat,
            ))
        candidates.sort(key=lambda row: (_hash_order(seed, row.event_id), row.event_id))
        chosen.extend(candidates[:targets_per_user])
    chosen.sort(key=lambda row: (row.uid, row.position))
    return chosen


def history_for_target(store: SequenceStore, target: PanelTarget, history_events: int = 200) -> list[int]:
    """Return BOS plus strict prior events, with no target or future labels."""
    if history_events < 0:
        raise ValueError("history_events cannot be negative")
    user = store.users[target.user_index]
    if int(user.uid) != target.uid or target.position >= len(user.tokens):
        raise ValueError("panel target does not match the supplied sequence store")
    if (int(user.tokens[target.position]) != target.token_id
            or int(user.raw_item_ids[target.position]) != target.raw_item_id):
        raise ValueError("panel target token/raw item does not match the supplied sequence store")
    history = user.history(target.position, history_events + 1)
    if len(history) != 1 + min(history_events, target.position):
        raise AssertionError("history length does not match the strict-past contract")
    return history


def panel_manifest(
    panel: list[PanelTarget], *, split: str, seed: int, source_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "split": split,
        "source_sha256": source_sha256,
        "selection_seed": seed,
        "selection_rule": "stable SHA256(seed,event_id), selected before scoring; no ratio/catalog filter",
        "targets_per_user_max": 3,
        "n_events": len(panel),
        "n_users": len({row.uid for row in panel}),
        "event_ids_sha256": hashlib.sha256("\n".join(row.event_id for row in panel).encode()).hexdigest(),
        "oov_targets": sum(row.token_id == 2 for row in panel),
    }
