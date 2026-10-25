from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from tlab.data.store import SequenceStore
from tlab.evaluation.metrics import aggregate_macro_user, paired_user_bootstrap
from tlab.models.actor import ActorShape, ItemGPT2Actor


def rank_score_vector(
    scores: Sequence[float] | np.ndarray,
    target_index: int | None,
    item_ids: Sequence[int] | np.ndarray,
) -> int | None:
    """One-based rank with score-descending and raw-item-ID-ascending ties."""
    if target_index is None or target_index < 0 or target_index >= len(scores):
        return None
    score_array = np.asarray(scores)
    id_array = np.asarray(item_ids)
    if score_array.ndim != 1 or id_array.ndim != 1 or len(score_array) != len(id_array):
        raise ValueError("scores and item_ids must be one-dimensional arrays of equal size")
    target_score = score_array[target_index]
    target_id = id_array[target_index]
    ahead = np.count_nonzero((score_array > target_score) | ((score_array == target_score) & (id_array < target_id)))
    return int(ahead) + 1


def _train_transition_matrix(store: SequenceStore, catalog_size: int):
    from scipy.sparse import coo_matrix

    n_targets = sum(len(user.train_targets) for user in store.users)
    previous = np.empty(n_targets, dtype=np.int32)
    following = np.empty(n_targets, dtype=np.int32)
    offset = 0
    for user_index, user in enumerate(store.users, start=1):
        positions = np.asarray(user.train_targets, dtype=np.int64)
        size = len(positions)
        if not size:
            continue
        tokens = user.tokens
        previous[offset:offset + size] = tokens[positions - 1]
        following[offset:offset + size] = tokens[positions]
        offset += size
        if user_index % 1500 == 0:
            print(f"Built transition inputs for {user_index:,} users ({offset:,}/{n_targets:,} targets)", flush=True)
    previous = previous[:offset]
    following = following[:offset]
    valid = following >= 3
    matrix = coo_matrix(
        (np.ones(int(valid.sum()), dtype=np.int32), (previous[valid], following[valid] - 3)),
        shape=(catalog_size + 3, catalog_size),
        dtype=np.int32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix, int(valid.sum())


def smoothed_transition_scores(transition, training_counts: Sequence[float], alpha: float) -> np.ndarray:
    """Score items with first-order counts plus an alpha-mass popularity prior.

    ``alpha`` is the total pseudo-count mass assigned to the global training
    popularity distribution for a seen context. An unseen context backs off to
    the unsmoothed popularity ranking, including when alpha is zero.
    """
    counts = np.asarray(training_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("training_counts must be a non-empty vector")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative")
    total = float(counts.sum())
    prior = counts / total if total > 0 else np.full(counts.size, 1.0 / counts.size)
    scores = float(alpha) * prior
    if transition.nnz:
        scores = scores.copy()
        scores[transition.indices] += transition.data.astype(np.float64, copy=False)
    elif alpha == 0.0:
        scores = counts.copy()
    return scores


def _metric_bundle(rows: list[tuple[int, int | None]], coverage: float) -> dict[str, Any]:
    metrics = aggregate_macro_user(rows)
    return {
        "n_users": len({uid for uid, _ in rows}),
        "n_targets": len(rows),
        "target_coverage": coverage,
        "macro_user_unconditional": metrics,
    }


def _ndcg10(rank: int | None) -> float:
    return 1.0 / np.log2(rank + 1) if rank is not None and rank <= 10 else 0.0


def run_validation_baselines(
    config_path: str | Path = "configs/base.yaml",
    sft_checkpoint: str | Path | None = None,
    seed: int = 17,
    per_user_limit: int = 1,
    output_path: str | Path | None = None,
    include_per_user_ranks: bool = False,
) -> dict[str, Any]:
    import polars as pl

    from tlab.config import load_config
    from tlab.evaluation.ranking import evaluate_full_catalog

    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(
        processed_dir / "sequence_index",
        expected_source_sha256=protocol["processed_listens_sha256"],
    )
    # Actor logits and transition-matrix columns use catalog token IDs; preserve
    # that order here and use raw item IDs only as deterministic rank tie-breaks.
    catalog = pl.read_parquet(processed_dir / "catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog.get_column("item_id").to_list(), dtype=np.int64)
    training_counts = np.asarray(catalog.get_column("training_count").to_list(), dtype=np.float64)
    catalog_size = len(item_ids)
    raw_to_index = {int(item_id): index for index, item_id in enumerate(item_ids)}
    anchors = store.evaluation_targets("validation", per_user_limit=per_user_limit, seed=seed)
    if not anchors:
        raise ValueError("validation baseline evaluation has no recommendable anchors")

    popularity_rows: list[tuple[int, int | None]] = []
    for user_index, position in anchors:
        user = store.users[user_index]
        target_index = raw_to_index.get(int(user.raw_item_ids[position]))
        popularity_rows.append((int(user.uid), rank_score_vector(training_counts, target_index, item_ids)))
    popularity_coverage = sum(rank is not None for _, rank in popularity_rows) / len(popularity_rows)

    transition_matrix, transition_examples = _train_transition_matrix(store, catalog_size)
    sequential_rows: list[tuple[int, int | None]] = []
    for user_index, position in anchors:
        user = store.users[user_index]
        target_index = raw_to_index.get(int(user.raw_item_ids[position]))
        previous_token = int(user.tokens[position - 1]) if position > 0 else 1
        transition = transition_matrix.getrow(previous_token)
        if transition.nnz:
            scores = np.zeros(catalog_size, dtype=np.int32)
            scores[transition.indices] = transition.data
        else:
            scores = training_counts
        sequential_rows.append((int(user.uid), rank_score_vector(scores, target_index, item_ids)))
    sequential_coverage = sum(rank is not None for _, rank in sequential_rows) / len(sequential_rows)

    result: dict[str, Any] = {
        "split": "validation",
        "seed": seed,
        "per_user_limit": per_user_limit,
        "catalog_size": catalog_size,
        "catalog_tie_break": "raw item ID ascending",
        "popularity": {
            "score": "training_count of non-organic listens before train cutoff",
            **_metric_bundle(popularity_rows, popularity_coverage),
        },
        "sequential_bigram": {
            "score": "count(previous context token, next train non-organic target); unseen context falls back to popularity",
            "training_transition_examples": transition_examples,
            **_metric_bundle(sequential_rows, sequential_coverage),
        },
        "uniform_random_expected_ndcg10": float(
            popularity_coverage
            * sum(1.0 / np.log2(rank + 1) for rank in range(1, min(10, catalog_size) + 1))
            / catalog_size
        ),
    }
    if sft_checkpoint is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shape = ActorShape(
            vocab_size=catalog_size + 3,
            n_positions=int(config.model.get("n_positions", 256)),
            n_layer=int(config.model.get("n_layer", 4)),
            n_embd=int(config.model.get("n_embd", 256)),
            n_head=int(config.model.get("n_head", 4)),
            dropout=float(config.model.get("dropout", 0.1)),
        )
        actor = ItemGPT2Actor(shape)
        checkpoint = torch.load(sft_checkpoint, map_location="cpu", weights_only=False)
        actor.load_state_dict(checkpoint["model"])
        actor.to(device).eval()
        sft_metrics = evaluate_full_catalog(
            actor, store, item_ids.tolist(), split="validation",
            history_length=int(config.dataset.get("history_length", 200)),
            per_user_limit=per_user_limit, seed=seed, batch_size=16,
        )
        sft_ranks = {
            int(uid): ranks[0] if ranks else None
            for uid, ranks in sft_metrics["per_user_ranks"].items()
        }
        result["sft"] = {key: value for key, value in sft_metrics.items() if key != "per_user_ranks"}
        if include_per_user_ranks:
            result["sft"]["per_user_ranks"] = sft_metrics["per_user_ranks"]
        result["paired_ndcg10_difference_vs_sft"] = {}
        for name, baseline_rows in (("popularity", popularity_rows), ("sequential_bigram", sequential_rows)):
            difference = {
                uid: _ndcg10(rank) - _ndcg10(sft_ranks[uid])
                for uid, rank in baseline_rows
            }
            result["paired_ndcg10_difference_vs_sft"][name] = paired_user_bootstrap(
                difference, repetitions=2000, seed=seed,
            )
    target = Path(output_path) if output_path else root / "artifacts" / "validation_baselines_seed17.json"
    if not target.is_absolute():
        target = root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    result["output_path"] = str(target.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate popularity, sequential bigram, and optional SFT baselines")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sft-checkpoint")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-user-limit", type=int, default=1)
    parser.add_argument(
        "--include-per-user-ranks", action="store_true",
        help="save per-user target ranks for paired comparisons with another model",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_validation_baselines(
        args.config, sft_checkpoint=args.sft_checkpoint, seed=args.seed,
        per_user_limit=args.per_user_limit, output_path=args.output,
        include_per_user_ranks=args.include_per_user_ranks,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
