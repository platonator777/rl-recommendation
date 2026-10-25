from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tlab.config import load_config
from tlab.data.store import SequenceStore


def _summary(values: list[int] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.int32)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "percentiles": {str(p): float(np.percentile(array, p)) for p in (0, 10, 25, 50, 75, 90, 95, 99, 100)},
        "fraction_above_200": float(np.mean(array > 200)),
        "fraction_above_199": float(np.mean(array > 199)),
    }


def audit_sft_data(
    config_path: str | Path = "configs/base.yaml",
    output: str | Path = "artifacts/reference/history_semantics_audit.json",
    sample_targets: int = 100_000,
    seed: int = 17,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = config.path.parent.parent
    processed = Path(config.data.get("processed_dir", "data/processed"))
    if not processed.is_absolute():
        processed = root / processed
    protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(processed / "sequence_index", protocol["processed_listens_sha256"])
    history_total_tokens = int(config.dataset.get("history_length", 200))
    history_events = int(config.dataset.get("history_events", 200))

    old_context_lengths: list[int] = []
    old_target_positions: list[int] = []
    old_sequence_lengths: list[int] = []
    old_targets = 0
    old_windows = 0
    for tokens, mask in store.sample_sft_windows(
        max_targets=sample_targets,
        history_length=history_total_tokens,
        n_positions=int(config.model.get("n_positions", 256)),
        seed=seed,
    ):
        old_windows += 1
        old_sequence_lengths.append(len(tokens) - 1)
        for target_position, supervised in enumerate(mask):
            if supervised:
                old_targets += 1
                old_context_lengths.append(max(0, target_position - 1))
                old_target_positions.append(target_position - 1)
    if old_targets != min(sample_targets, sum(len(user.train_targets) for user in store.users)):
        raise AssertionError("legacy bucket sampler did not account for its requested target quota")

    new_index = store.sample_sft_target_index(
        max_targets=sample_targets,
        history_events=history_events,
        seed=seed,
    )
    new_positions = np.frombuffer(new_index.positions, dtype=np.uint32)
    new_context_lengths = np.minimum(new_positions, history_events)
    second_index = store.sample_sft_target_index(
        max_targets=sample_targets,
        history_events=history_events,
        seed=seed + 1,
    )
    overlap = np.intersect1d(new_index.target_keys(), second_index.target_keys(), assume_unique=True)
    new_stats = {
        "targets": len(new_index),
        "unique_target_positions": int(np.unique(new_index.target_keys()).size),
        "users_covered": int(np.unique(np.frombuffer(new_index.user_indices, dtype=np.uint32)).size),
        "catalog_items_covered": len({
            int(store.users[int(user_index)].tokens[int(position)])
            for user_index, position in zip(new_index.user_indices, new_index.positions)
        }),
        "context_history_events": _summary(new_context_lengths),
        "target_is_last_token": True,
        "input_excludes_target": True,
        "cross_pass_sample_overlap": int(overlap.size),
        "cross_pass_overlap_fraction": float(overlap.size / max(1, len(new_index))),
    }
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_id": "tlab_yambda_reference_2026-10-13",
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
        "sample_targets_per_sampler": sample_targets,
        "seed": seed,
        "v1_window_sampler": {
            "history_length_argument": history_total_tokens,
            "history_length_semantics": "BOS plus up to history_length - 1 events for serving; bucket windows may expose more events to later targets",
            "windows": old_windows,
            "targets": old_targets,
            "per_target_history_events": _summary(old_context_lengths),
            "target_hidden_position_after_bos": _summary(old_target_positions),
            "window_input_length": _summary(old_sequence_lengths),
        },
        "reference_target_sampler": {
            "history_contract": f"BOS plus up to {history_events} preceding events for each target",
            "per_user_quota": "largest remainder proportional to eligible target count",
            "sampling_within_user": "uniform without replacement per pass",
            **new_stats,
        },
        "interpretation": (
            "V1 bucket windows use the target's causal prefix, but the prefix length varies by target and can exceed the serving cap. "
            "reference trains one selected target per example with the same capped history semantics used by reference evaluation and RL. "
            "The V1 checkpoints therefore remain historical references; reference weak budgets are reproduced before strong continuation."
        ),
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure SFT target/context semantics on the fixed prepared sequence index")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="artifacts/reference/history_semantics_audit.json")
    parser.add_argument("--sample-targets", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = audit_sft_data(args.config, args.output, args.sample_targets, args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
