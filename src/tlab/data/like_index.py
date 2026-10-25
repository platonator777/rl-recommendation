from __future__ import annotations

import argparse
import json
import os
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tlab.config import load_config
from tlab.data.labels import TimedPairEvent, attribute_new_likes
from tlab.data.manifest import sha256_file, write_json_atomic


def _load_actions(path: Path) -> dict[int, list[TimedPairEvent]]:
    parquet = pq.ParquetFile(path)
    grouped: dict[int, list[TimedPairEvent]] = {}
    for batch in parquet.iter_batches(batch_size=131_072, columns=["uid", "item_id", "timestamp"]):
        uid_col, item_col, timestamp_col = (column.to_pylist() for column in batch.columns)
        for uid, item_id, timestamp in zip(uid_col, item_col, timestamp_col):
            grouped.setdefault(int(uid), []).append(TimedPairEvent(int(uid), int(item_id), int(timestamp)))
    return grouped


def build_like_index(config_path: str | Path, window_seconds: int = 1800) -> dict[str, Any]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    raw_dir = Path(config.data.get("source_dir", "data/raw"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    prepared_path = processed_dir / "listens.parquet"
    protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    timestamp_unit_seconds = float(protocol["timestamp_unit_seconds"])
    boundaries = protocol["boundaries_source_units"]
    block_cutoffs = {
        "F_RM": float(boundaries["rm_validation_start_raw"]),
        "V_RM": float(boundaries["rl_anchor_start_raw"]),
    }
    likes_by_uid = _load_actions(raw_dir / "likes.parquet")
    unlikes_by_uid = _load_actions(raw_dir / "unlikes.parquet")
    parquet = pq.ParquetFile(prepared_path)
    columns = ["uid", "item_id", "timestamp", "token_id", "is_organic", "train_subsplit", "timestamp_tie_count"]
    suffix = "30m" if window_seconds == 1800 else f"{window_seconds}s"
    labels_path = processed_dir / f"like_labels_{suffix}.bin"
    temporary_labels_path = labels_path.with_suffix(labels_path.suffix + ".tmp")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    index_users: list[dict[str, int]] = []
    reason_counts: dict[str, Counter[str]] = {"F_RM": Counter(), "V_RM": Counter()}
    total_candidates: Counter[str] = Counter()
    total_valid_labels: Counter[str] = Counter()
    output_offset = 0
    current_uid: int | None = None
    user_rows: list[tuple[int, int, int, int, bool, str | None, int]] = []

    def flush_user(output) -> None:
        nonlocal output_offset, user_rows, current_uid
        if current_uid is None:
            return
        events = [TimedPairEvent(current_uid, row[0], row[1]) for row in user_rows]
        candidates: dict[tuple[int, int], tuple[int, str]] = {}
        sequence_position = -1
        index = 0
        while index < len(user_rows):
            timestamp = user_rows[index][1]
            stop = index + 1
            while stop < len(user_rows) and user_rows[stop][1] == timestamp:
                stop += 1
            group = user_rows[index:stop]
            sequence_position += 1
            ambiguous = len(group) > 1 or any(row[6] > 1 for row in group)
            if not ambiguous and sequence_position > 0:
                item_id, _, token_id, _, organic, subsplit, _ = group[0]
                if token_id >= 3 and not organic and subsplit in block_cutoffs:
                    candidates[(item_id, timestamp)] = (sequence_position, subsplit)
                    total_candidates[subsplit] += 1
            index = stop

        label_bytes = bytearray(b"\xff") * (sequence_position + 1)
        if candidates:
            user_likes = likes_by_uid.get(current_uid, [])
            user_unlikes = unlikes_by_uid.get(current_uid, [])
            for block, cutoff in block_cutoffs.items():
                block_candidates = {key for key, (_, candidate_block) in candidates.items() if candidate_block == block}
                if not block_candidates:
                    continue
                attributions = attribute_new_likes(
                    events,
                    user_likes,
                    user_unlikes,
                    observation_cutoff=cutoff,
                    window_seconds=window_seconds,
                    timestamp_unit_seconds=timestamp_unit_seconds,
                )
                for attribution in attributions:
                    key = (attribution.listen.item_id, int(attribution.listen.timestamp))
                    if key not in block_candidates:
                        continue
                    position, _ = candidates[key]
                    label_bytes[position] = 255 if attribution.label is None else int(attribution.label)
                    reason_counts[block][attribution.reason] += 1
                    if attribution.label is not None:
                        total_valid_labels[block] += 1
        output.write(label_bytes)
        index_users.append({"uid": current_uid, "offset": output_offset, "length": len(label_bytes)})
        output_offset += len(label_bytes)
        user_rows = []

    with temporary_labels_path.open("wb") as output:
        events_read = 0
        next_progress = 5_000_000
        for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
            rows = [column.to_pylist() for column in batch.columns]
            for uid_raw, item_raw, timestamp_raw, token_raw, organic_raw, subsplit_raw, tie_count_raw in zip(*rows):
                uid = int(uid_raw)
                if current_uid is not None and uid != current_uid:
                    flush_user(output)
                if current_uid != uid:
                    current_uid = uid
                user_rows.append((
                    int(item_raw), int(timestamp_raw), int(token_raw), 0,
                    bool(organic_raw), None if subsplit_raw is None else str(subsplit_raw),
                    int(tie_count_raw or 0),
                ))
                events_read += 1
                if events_read >= next_progress:
                    print(f"Attributed likes over {events_read:,} / {parquet.metadata.num_rows:,} listen rows", flush=True)
                    next_progress += 5_000_000
        flush_user(output)
    os.replace(temporary_labels_path, labels_path)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_listens_sha256": protocol["processed_listens_sha256"],
        "source_likes_sha256": sha256_file(raw_dir / "likes.parquet"),
        "source_unlikes_sha256": sha256_file(raw_dir / "unlikes.parquet"),
        "timestamp_unit_seconds": timestamp_unit_seconds,
        "window_seconds": window_seconds,
        "block_cutoffs_source_units": block_cutoffs,
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": sha256_file(labels_path),
        "labels_bytes": labels_path.stat().st_size,
        "sequence_events": output_offset,
        "n_users": len(index_users),
        "candidates_by_block": dict(total_candidates),
        "valid_labels_by_block": dict(total_valid_labels),
        "reasons_by_block": {block: dict(counter) for block, counter in reason_counts.items()},
        "users": index_users,
    }
    manifest_path = processed_dir / f"like_labels_{suffix}.json"
    write_json_atomic(manifest_path, manifest)
    summary = {key: value for key, value in manifest.items() if key != "users"}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-aware 30-minute new-like label index")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--window-seconds", type=int, default=1800)
    args = parser.parse_args()
    print(json.dumps(build_like_index(args.config, args.window_seconds), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
