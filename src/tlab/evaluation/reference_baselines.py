from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from tlab.data.store import SequenceStore
from tlab.data.manifest import sha256_file
from tlab.evaluation.baselines import _train_transition_matrix, rank_score_vector
from tlab.evaluation.metrics import aggregate_macro_user, paired_user_bootstrap
from tlab.evaluation.smoothed_markov import _evaluate_alpha, _user_ndcg
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.train.checkpoint import source_tree_sha256


def run_validation_baselines_reference(
    config_path: str | Path = "configs/reference_sft.yaml",
    *,
    seed: int = 17,
    anchors_per_user: int = 10,
    alpha_grid: Sequence[float] = (0.0, 1.0, 10.0, 100.0, 1000.0),
    sft_checkpoints: dict[str, str | Path] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare popularity, raw/smoothed Markov and optional actors on validation only."""
    import polars as pl

    from tlab.config import load_config
    from tlab.evaluation.ranking import evaluate_full_catalog

    if anchors_per_user < 1 or not alpha_grid or any(float(alpha) < 0 or not np.isfinite(alpha) for alpha in alpha_grid):
        raise ValueError("validation baselines need a positive per-user limit and finite non-negative alpha values")
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
    item_ids = np.asarray(catalog.get_column("item_id").to_list(), dtype=np.int64)
    training_counts = np.asarray(catalog.get_column("training_count").to_list(), dtype=np.float64)
    durations = catalog.get_column("catalog_duration_seconds").to_list()
    raw_to_index = {int(item_id): index for index, item_id in enumerate(item_ids)}
    anchors = store.evaluation_targets("validation", per_user_limit=anchors_per_user, seed=seed)
    if not anchors:
        raise ValueError("validation baseline panel is empty")

    transition_matrix, transition_examples = _train_transition_matrix(store, len(item_ids))
    by_alpha = [
        _evaluate_alpha(
            store, anchors, item_ids, training_counts, durations,
            transition_matrix, float(alpha), include_serving=False,
        )
        for alpha in alpha_grid
    ]
    selected = max(
        by_alpha,
        key=lambda row: (
            float(row["macro_user_unconditional"]["ndcg@10"]),
            -float(row["alpha_popularity_pseudocount_mass"]),
        ),
    )

    popularity_rows = []
    for user_index, position in anchors:
        user = store.users[user_index]
        target_index = raw_to_index.get(int(user.raw_item_ids[position]))
        popularity_rows.append((int(user.uid), rank_score_vector(training_counts, target_index, item_ids)))
    comparisons: dict[str, Any] = {}
    for row in by_alpha:
        label = f"markov_alpha_{row['alpha_popularity_pseudocount_mass']:g}"
        ndcg = _user_ndcg(row["per_user_ranks"])
        comparisons[label] = {
            "paired_user_ndcg10_sft_minus_baseline": None,
            "baseline_ndcg10_by_user": {int(uid): value for uid, value in ndcg.items()},
        }
    comparisons["popularity"] = {
        "paired_user_ndcg10_sft_minus_baseline": None,
        "baseline_ndcg10_by_user": {
            int(uid): float(np.mean([_ndcg10(rank) for rank in ranks]))
            for uid, ranks in _group_ranks(popularity_rows).items()
        },
    }
    actor_results: dict[str, Any] = {}
    for name, checkpoint_path in (sft_checkpoints or {}).items():
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        actor = ItemGPT2Actor(ActorShape(
            vocab_size=len(item_ids) + 3,
            n_positions=int(config.model.get("n_positions", 256)),
            n_layer=int(config.model.get("n_layer", 4)),
            n_embd=int(config.model.get("n_embd", 256)),
            n_head=int(config.model.get("n_head", 4)),
            dropout=float(config.model.get("dropout", 0.1)),
        ))
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(payload["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor.to(device).eval()
        evaluation = evaluate_full_catalog(
            actor, store, item_ids.tolist(), split="validation",
            history_length=int(config.dataset.get("history_length", 201)),
            per_user_limit=anchors_per_user, seed=seed, batch_size=16,
        )
        actor_user_ndcg = _user_ndcg({
            str(uid): ranks for uid, ranks in evaluation["per_user_ranks"].items()
        })
        paired: dict[str, Any] = {}
        for baseline_name, baseline in comparisons.items():
            baseline_user_ndcg = baseline["baseline_ndcg10_by_user"]
            common = sorted(set(actor_user_ndcg) & set(baseline_user_ndcg))
            differences = {
                uid: actor_user_ndcg[uid] - float(baseline_user_ndcg[uid])
                for uid in common
            }
            paired[baseline_name] = paired_user_bootstrap(differences, repetitions=2000, seed=seed)
        actor_results[name] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "evaluation": {key: value for key, value in evaluation.items() if key != "per_user_ranks"},
            "paired_user_ndcg10_sft_minus_baseline": paired,
        }

    result = {
        "stage": "reference_VALIDATION_BASELINES",
        "split": "validation",
        "seed": seed,
        "anchors_per_user": anchors_per_user,
        "n_anchors": len(anchors),
        "n_transition_training_examples": transition_examples,
        "catalog_size": len(item_ids),
        "popularity": {
            "score": "train-only catalog training_count",
            "n_users": len(_group_ranks(popularity_rows)),
            "n_targets": len(popularity_rows),
            "target_coverage": sum(rank is not None for _, rank in popularity_rows) / len(popularity_rows),
            "macro_user_unconditional": aggregate_macro_user(popularity_rows),
        },
        "raw_and_smoothed_markov_by_alpha": [
            {key: value for key, value in row.items() if key not in {"per_user_ranks", "per_user_top_item_ids"}}
            for row in by_alpha
        ],
        "selected_validation_alpha": float(selected["alpha_popularity_pseudocount_mass"]),
        "selected_markov_validation_ndcg10": float(selected["macro_user_unconditional"]["ndcg@10"]),
        "actor_results": actor_results,
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
        "config_path": str(config.path.resolve()),
        "source_sha256": source_tree_sha256(root / "src" / "tlab"),
        "test_used": False,
        "rm_audit_policy_scores_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    target = Path(output_path) if output_path else root / "artifacts" / "reference" / f"validation_baselines_seed{seed}.json"
    if not target.is_absolute():
        target = root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["output_path"] = str(target.resolve())
    return result


def _group_ranks(rows: Sequence[tuple[int, int | None]]) -> dict[int, list[int | None]]:
    grouped: dict[int, list[int | None]] = {}
    for uid, rank in rows:
        grouped.setdefault(int(uid), []).append(rank)
    return grouped


def _ndcg10(rank: int | None) -> float:
    import math

    return 1.0 / math.log2(int(rank) + 1) if rank is not None and rank <= 10 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only reference popularity and Markov baselines")
    parser.add_argument("--config", default="configs/reference_sft.yaml")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--anchors-per-user", type=int, default=10)
    parser.add_argument("--alpha-grid", default="0,1,10,100,1000")
    parser.add_argument("--weak-checkpoint")
    parser.add_argument("--strong-checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()
    checkpoints = {
        key: value for key, value in (
            ("weak_sft", args.weak_checkpoint), ("strong_sft", args.strong_checkpoint),
        ) if value
    }
    result = run_validation_baselines_reference(
        args.config, seed=args.seed, anchors_per_user=args.anchors_per_user,
        alpha_grid=tuple(float(item) for item in args.alpha_grid.split(",") if item),
        sft_checkpoints=checkpoints, output_path=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
