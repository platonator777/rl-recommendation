from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from tlab.data.catalog import PAD_ID
from tlab.data.manifest import sha256_file
from tlab.data.store import SequenceStore
from tlab.evaluation.metrics import aggregate_macro_user, paired_user_bootstrap, single_target_metrics
from tlab.models.actor import ActorShape, ItemGPT2Actor


def _collate(histories: Sequence[Sequence[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, histories))
    ids = torch.full((len(histories), width), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, history in enumerate(histories):
        ids[row, :len(history)] = torch.tensor(history, dtype=torch.long, device=device)
        mask[row, :len(history)] = 1
    return ids, mask


@torch.no_grad()
def _ranks_for_batch(
    actor: ItemGPT2Actor,
    histories: Sequence[Sequence[int]],
    targets: Sequence[tuple[int, int]],
    store: SequenceStore,
    item_ids: Sequence[int],
    item_to_index: dict[int, int],
    device: torch.device,
) -> list[tuple[int, int | None]]:
    input_ids, attention = _collate(histories, device)
    log_distribution = actor.next_action_log_probs(input_ids, attention)[:, 3:]
    raw_targets = [int(store.users[user_index].raw_item_ids[position]) for user_index, position in targets]
    target_indices = [item_to_index.get(item, -1) for item in raw_targets]
    result: list[tuple[int, int | None]] = []
    item_tensor = torch.tensor(item_ids, dtype=torch.long, device=device)
    for row, ((user_index, _), target_index, raw_target) in enumerate(zip(targets, target_indices, raw_targets)):
        uid = int(store.users[user_index].uid)
        if target_index < 0:
            result.append((uid, None))
            continue
        score = log_distribution[row, target_index]
        ahead = (log_distribution[row] > score) | ((log_distribution[row] == score) & (item_tensor < raw_target))
        result.append((uid, int(ahead.sum().item()) + 1))
    return result


def _summarize(rows: Sequence[tuple[int, int | None]]) -> tuple[dict[str, Any], dict[int, float]]:
    metrics = aggregate_macro_user(list(rows))
    per_user: dict[int, list[int | None]] = defaultdict(list)
    for uid, rank in rows:
        per_user[uid].append(rank)
    user_ndcg = {
        uid: sum(single_target_metrics(rank)["ndcg@10"] for rank in ranks) / len(ranks)
        for uid, ranks in per_user.items()
    }
    return metrics, user_ndcg


@torch.no_grad()
def evaluate_sft_history_sensitivity(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    seed: int = 17,
    per_user_limit: int = 10,
    batch_size: int = 16,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    import polars as pl

    from tlab.config import load_config

    started = time.perf_counter()
    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(
        processed_dir / "sequence_index", expected_source_sha256=protocol["processed_listens_sha256"],
    )
    catalog = pl.read_parquet(processed_dir / "catalog.parquet").sort("token_id")
    item_ids = [int(value) for value in catalog.get_column("item_id").to_list()]
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = ItemGPT2Actor(ActorShape(
        vocab_size=len(item_ids) + 3,
        n_positions=int(config.model.get("n_positions", 256)),
        n_layer=int(config.model.get("n_layer", 4)),
        n_embd=int(config.model.get("n_embd", 256)),
        n_head=int(config.model.get("n_head", 4)),
        dropout=float(config.model.get("dropout", 0.1)),
    ))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint["model"])
    actor = actor.to(device).eval()
    history_length = int(config.dataset.get("history_length", 201))
    targets = store.evaluation_targets("validation", per_user_limit=per_user_limit, seed=seed)
    original_histories = store.histories_for(targets, history_length=history_length)
    rng = random.Random(seed + 790_001)
    order_histories: list[list[int]] = []
    for history in original_histories:
        shuffled = list(history)
        content = shuffled[1:]
        rng.shuffle(content)
        order_histories.append([shuffled[0], *content])

    examples_by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for example in targets:
        examples_by_user[example[0]].append(example)
    eligible_users = list(examples_by_user)
    swap_rng = random.Random(seed + 790_002)
    swapped_donors: list[tuple[int, int]] = []
    for user_index, _ in targets:
        other_users = [candidate for candidate in eligible_users if candidate != user_index]
        if other_users:
            donor_user = swap_rng.choice(other_users)
            swapped_donors.append(swap_rng.choice(examples_by_user[donor_user]))
        else:
            swapped_donors.append((user_index, examples_by_user[user_index][0][1]))
    swapped_histories = store.histories_for(swapped_donors, history_length=history_length)

    rows = {"real": [], "order_shuffled": [], "other_user_history": []}
    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        batch_targets = targets[start:stop]
        for mode, histories in (
            ("real", original_histories),
            ("order_shuffled", order_histories),
            ("other_user_history", swapped_histories),
        ):
            rows[mode].extend(_ranks_for_batch(
                actor, histories[start:stop], batch_targets, store, item_ids, item_to_index, device,
            ))
    summaries: dict[str, Any] = {}
    per_user_scores: dict[str, dict[int, float]] = {}
    for mode, mode_rows in rows.items():
        metrics, user_ndcg = _summarize(mode_rows)
        summaries[mode] = metrics
        per_user_scores[mode] = user_ndcg
    common = sorted(set(per_user_scores["real"]) & set(per_user_scores["order_shuffled"]) & set(per_user_scores["other_user_history"]))
    result = {
        "stage": "SFT_HISTORY_SENSITIVITY",
        "seed": seed,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history_contract": f"BOS plus up to {history_length - 1} preceding events; target excluded",
        "split": "validation",
        "validation_seed": seed,
        "targets_per_user": per_user_limit,
        "n_targets": len(targets),
        "n_users": len(common),
        "results": summaries,
        "paired_user_ndcg10_difference_real_minus_order_shuffled": paired_user_bootstrap(
            {uid: per_user_scores["real"][uid] - per_user_scores["order_shuffled"][uid] for uid in common},
            repetitions=2000, seed=seed,
        ),
        "paired_user_ndcg10_difference_real_minus_other_user_history": paired_user_bootstrap(
            {uid: per_user_scores["real"][uid] - per_user_scores["other_user_history"][uid] for uid in common},
            repetitions=2000, seed=seed + 1,
        ),
        "rm_scores_used": False,
        "test_used": False,
        "elapsed_seconds": time.perf_counter() - started,
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
    }
    target_path = Path(output_path) if output_path else root / "artifacts" / "reference" / f"sft_history_sensitivity_{Path(checkpoint_path).parent.name}.json"
    if not target_path.is_absolute():
        target_path = root / target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["output_path"] = str(target_path.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only SFT history/order sensitivity diagnostic")
    parser.add_argument("--config", default="configs/reference_sft.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-user-limit", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_sft_history_sensitivity(
        args.config, args.checkpoint, seed=args.seed,
        per_user_limit=args.per_user_limit, batch_size=args.batch_size,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
