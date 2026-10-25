from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tlab.data.store import SequenceStore


def evaluate_like_window_sensitivity(
    config_path: str | Path,
    windows_seconds: tuple[int, ...] = (300, 1800, 7200),
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize like-label coverage and prevalence across attribution windows on V_RM."""
    from tlab.config import load_config
    from tlab.data.manifest import sha256_file

    if len(windows_seconds) < 2 or any(window <= 0 for window in windows_seconds):
        raise ValueError("provide at least two positive attribution windows")
    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(
        processed_dir / "sequence_index", expected_source_sha256=protocol["processed_listens_sha256"],
    )
    sequence_length = sum(len(user.tokens) for user in store.users)
    raw_dir = Path(config.data.get("source_dir", "data/raw"))
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    source_files = {
        "listens": protocol["processed_listens_sha256"],
        "likes": sha256_file(raw_dir / "likes.parquet"),
        "unlikes": sha256_file(raw_dir / "unlikes.parquet"),
    }
    label_vectors: dict[int, dict[tuple[int, int], int]] = {}
    summaries: dict[str, Any] = {}
    partition_labels: dict[int, dict[str, dict[tuple[int, int], int]]] = {}
    for window in windows_seconds:
        suffix = "30m" if window == 1800 else f"{window}s"
        manifest_path = processed_dir / f"like_labels_{suffix}.json"
        binary_path = processed_dir / f"like_labels_{suffix}.bin"
        if not manifest_path.exists() or not binary_path.exists():
            raise FileNotFoundError(f"like index for {window}s is missing; build it with tlab-like-index")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("window_seconds", 0)) != window:
            raise ValueError(f"like index window mismatch in {manifest_path}")
        if manifest.get("source_listens_sha256") != source_files["listens"]:
            raise ValueError(f"stale listen source hash in {manifest_path}")
        if manifest.get("source_likes_sha256") != source_files["likes"] or manifest.get("source_unlikes_sha256") != source_files["unlikes"]:
            raise ValueError(f"stale like/unlike source hash in {manifest_path}")
        if manifest.get("labels_sha256") != sha256_file(binary_path):
            raise ValueError(f"stale label-vector hash in {manifest_path}")
        if int(manifest["sequence_events"]) != sequence_length:
            raise ValueError(f"label index length does not match cached sequence for {window}s")
        vector = np.memmap(binary_path, dtype="i1", mode="r", shape=(sequence_length,))
        offsets = {int(row["uid"]): (int(row["offset"]), int(row["length"])) for row in manifest["users"]}
        labels_by_key: dict[tuple[int, int], int] = {}
        by_partition: dict[str, dict[tuple[int, int], int]] = {"fit": {}, "audit": {}}
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for user in store.users:
            if user.rm_user_partition not in {"fit", "audit"}:
                continue
            offset_length = offsets.get(int(user.uid))
            if offset_length is None or offset_length[1] != len(user.tokens):
                raise ValueError(f"user sequence is missing or misaligned for uid={user.uid}")
            offset, _ = offset_length
            partition = str(user.rm_user_partition)
            for position_raw in user.rm_validation_positions:
                position = int(position_raw)
                label = int(vector[offset + position])
                key = (int(user.uid), position)
                labels_by_key[key] = label
                by_partition[partition][key] = label
                if label == 1:
                    counts[partition]["positive"] += 1
                elif label == 0:
                    counts[partition]["negative"] += 1
                else:
                    counts[partition]["censored_or_ambiguous"] += 1
        label_vectors[window] = labels_by_key
        partition_labels[window] = by_partition
        block_counts = {
            part: {
                "candidates": int(sum(values.values())),
                "positive": int(values["positive"]),
                "negative": int(values["negative"]),
                "censored_or_ambiguous": int(values["censored_or_ambiguous"]),
                "valid_labels": int(values["positive"] + values["negative"]),
                "positive_prevalence": (
                    float(values["positive"] / (values["positive"] + values["negative"]))
                    if values["positive"] + values["negative"] else None
                ),
            }
            for part, values in counts.items()
        }
        summaries[str(window)] = {
            "suffix": suffix,
            "labels_sha256": manifest["labels_sha256"],
            "reasons_by_block": manifest["reasons_by_block"],
            "v_rm_by_rm_user_partition": block_counts,
        }

    comparisons: dict[str, Any] = {}
    reference_window = 1800 if 1800 in windows_seconds else windows_seconds[0]
    for window in windows_seconds:
        if window == reference_window:
            continue
        reference = label_vectors[reference_window]
        candidate = label_vectors[window]
        shared = set(reference).intersection(candidate)
        common_valid = [(reference[key], candidate[key]) for key in shared if reference[key] in (0, 1) and candidate[key] in (0, 1)]
        transitions = Counter(f"{left}_to_{right}" for left, right in common_valid)
        comparisons[f"{reference_window}s_vs_{window}s"] = {
            "shared_v_rm_targets": len(shared),
            "both_valid_targets": len(common_valid),
            "valid_label_transitions": {key: int(transitions[key]) for key in ("0_to_0", "0_to_1", "1_to_0", "1_to_1")},
        }

    result: dict[str, Any] = {
        "experiment": "V_RM like attribution window sensitivity",
        "windows_seconds": list(windows_seconds),
        "reference_window_seconds": reference_window,
        "source_hashes": source_files,
        "window_summaries": summaries,
        "paired_label_comparisons": comparisons,
        "interpretation": "descriptive label coverage/prevalence and reclassification; the 30-minute RM definition remains primary",
    }
    target = Path(output_path) if output_path else root / "artifacts" / "like_window_sensitivity_seed17.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    result["output_path"] = str(target.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare like-label coverage across attribution windows on V_RM")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--windows-seconds", nargs="+", type=int, default=(300, 1800, 7200))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_like_window_sensitivity(args.config, tuple(args.windows_seconds), args.output)
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
