from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

from tlab.data.catalog import BOS_ID, PAD_ID
from tlab.data.store import SequenceStore
from tlab.evaluation.metrics import aggregate_macro_user
from tlab.models.actor import ItemGPT2Actor


def _collate_histories(histories: Sequence[Sequence[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(history) for history in histories)
    ids = torch.full((len(histories), width), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, history in enumerate(histories):
        ids[row, :len(history)] = torch.tensor(history, dtype=torch.long, device=device)
        mask[row, :len(history)] = 1
    return ids, mask


def _top_item_ids(scores: np.ndarray, catalog_item_ids: Sequence[int], top_k: int) -> list[list[int]]:
    """Return top item IDs with score-descending, item-ID-ascending tie breaks."""
    score_array = np.asarray(scores)
    item_ids = np.asarray(catalog_item_ids, dtype=np.int64)
    if score_array.ndim != 2 or score_array.shape[1] != len(item_ids):
        raise ValueError("scores must be a batch-by-catalog matrix")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    width = min(top_k, len(item_ids))
    return [
        item_ids[np.lexsort((item_ids, -row))[:width]].astype(np.int64).tolist()
        for row in score_array
    ]


@torch.no_grad()
def evaluate_full_catalog(
    actor: ItemGPT2Actor,
    store: SequenceStore,
    catalog_item_ids: Sequence[int],
    split: str,
    history_length: int = 200,
    per_user_limit: int | None = 10,
    seed: int = 17,
    batch_size: int = 16,
    top_k: int = 0,
) -> dict[str, Any]:
    """Full-catalog macro-user ranking; OOC targets remain unconditional misses."""
    if batch_size < 1 or not catalog_item_ids or top_k < 0:
        raise ValueError("evaluation needs a positive batch size and a non-empty catalog")
    device = next(actor.parameters()).device
    targets = store.evaluation_targets(split, per_user_limit=per_user_limit, seed=seed)
    if not targets:
        raise ValueError(f"no recommendable targets are available for the {split!r} evaluation split")
    item_ids = torch.tensor(catalog_item_ids, dtype=torch.long, device=device)
    item_to_index = {int(item_id): index for index, item_id in enumerate(catalog_item_ids)}
    rows: list[tuple[int, int | None]] = []
    in_catalog = 0
    in_catalog_cross_entropy_sum = 0.0
    per_user_top_items: dict[int, list[list[int]]] = defaultdict(list)
    actor.eval()

    for start in range(0, len(targets), batch_size):
        batch_targets = targets[start:start + batch_size]
        histories = [store.users[user_index].history(position, history_length) for user_index, position in batch_targets]
        input_ids, attention_mask = _collate_histories(histories, device)
        log_dist = actor.next_action_log_probs(input_ids, attention_mask)[:, 3:]
        target_indices = []
        target_raw_ids = []
        for user_index, position in batch_targets:
            raw_item_id = int(store.users[user_index].raw_item_ids[position])
            target_raw_ids.append(raw_item_id)
            target_indices.append(item_to_index.get(raw_item_id, -1))
        target_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
        valid_target = target_tensor >= 0
        in_catalog += int(valid_target.sum().item())
        safe_target = target_tensor.clamp_min(0)
        target_scores = log_dist.gather(1, safe_target[:, None])
        target_ids = item_ids[safe_target]
        in_catalog_cross_entropy_sum += float((-target_scores[valid_target]).sum().item())
        ahead = (log_dist > target_scores) | ((log_dist == target_scores) & (item_ids[None, :] < target_ids[:, None]))
        ranks = ahead.sum(dim=-1).add(1).cpu().tolist()
        for (user_index, _), valid, rank in zip(batch_targets, valid_target.cpu().tolist(), ranks):
            rows.append((store.users[user_index].uid, int(rank) if valid else None))
        if top_k:
            top_rows = _top_item_ids(log_dist.cpu().numpy(), catalog_item_ids, top_k)
            for (user_index, _), top_items in zip(batch_targets, top_rows):
                per_user_top_items[store.users[user_index].uid].append(top_items)

    metrics = aggregate_macro_user(rows)
    in_catalog_rows = [(uid, rank) for uid, rank in rows if rank is not None]
    metrics_in_catalog = aggregate_macro_user(in_catalog_rows) if in_catalog_rows else {}
    per_user: dict[int, list[int | None]] = defaultdict(list)
    for uid, rank in rows:
        per_user[uid].append(rank)
    result = {
        "split": split,
        "catalog_size": len(catalog_item_ids),
        "n_users": len(per_user),
        "n_targets": len(rows),
        "target_coverage": in_catalog / len(rows) if rows else None,
        "n_in_catalog_targets": in_catalog,
        "in_catalog_cross_entropy": in_catalog_cross_entropy_sum / in_catalog if in_catalog else None,
        "macro_user_unconditional": metrics,
        "macro_user_in_catalog": metrics_in_catalog,
        "per_user_ranks": {str(uid): ranks for uid, ranks in per_user.items()},
        "sampled_per_user_limit": per_user_limit,
        "anchor_seed": seed,
    }
    if top_k:
        result["top_k"] = top_k
        result["per_user_top_item_ids"] = {str(uid): recs for uid, recs in per_user_top_items.items()}
    return result
