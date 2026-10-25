from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tlab.data.manifest import sha256_file, write_json_atomic
from tlab.data.store import SequenceStore
from tlab.evaluation.baselines import _train_transition_matrix, rank_score_vector, smoothed_transition_scores
from tlab.evaluation.behavior import exposure_metrics, repeat_rate
from tlab.evaluation.metrics import aggregate_macro_user, paired_user_bootstrap, single_target_metrics


def _evaluate_alpha(
    store: SequenceStore,
    anchors: Sequence[tuple[int, int]],
    item_ids: np.ndarray,
    training_counts: np.ndarray,
    durations: Sequence[float | None],
    transition_matrix,
    alpha: float,
    include_serving: bool = True,
) -> dict[str, Any]:
    catalog_size = len(item_ids)
    raw_to_index = {int(item_id): index for index, item_id in enumerate(item_ids)}
    rows: list[tuple[int, int | None]] = []
    ranks_by_user: dict[str, list[int | None]] = defaultdict(list)
    recs_by_user: dict[str, list[list[int]]] = defaultdict(list)
    index_recs: list[list[int]] = []
    recent_by_anchor: list[set[int]] = []
    top_item_indices: list[int] = []
    selected_durations: list[float] = []
    for user_index, position in anchors:
        user = store.users[user_index]
        uid = str(user.uid)
        target_index = raw_to_index.get(int(user.raw_item_ids[position]))
        previous_token = int(user.tokens[position - 1]) if position > 0 else 1
        scores = smoothed_transition_scores(transition_matrix.getrow(previous_token), training_counts, alpha)
        rank = rank_score_vector(scores, target_index, item_ids)
        rows.append((int(uid), rank))
        ranks_by_user[uid].append(rank)

        if include_serving:
            order = np.lexsort((item_ids, -scores))[:10]
            rec_item_ids = [int(item_ids[index]) for index in order]
            recs_by_user[uid].append(rec_item_ids)
            index_recs.append([int(index) for index in order])
            if len(order):
                top_item_indices.append(int(order[0]))
            recent = {
                int(token) - 3
                for token in user.tokens[max(0, position - 20):position]
                if 3 <= int(token) < 3 + catalog_size
            }
            recent_by_anchor.append(recent)
            for item_index in order:
                duration = durations[int(item_index)]
                if duration is not None and math.isfinite(float(duration)) and float(duration) > 0:
                    selected_durations.append(float(duration))

    ranking = aggregate_macro_user(rows)
    n_targets = len(rows)
    head_count = max(1, math.ceil(0.01 * catalog_size))
    head = set(sorted(range(catalog_size), key=lambda i: (-training_counts[i], int(item_ids[i])))[:head_count])
    flat = [index for row in index_recs for index in row]
    return {
        "alpha_popularity_pseudocount_mass": float(alpha),
        "n_users": len(ranks_by_user),
        "n_targets": n_targets,
        "target_catalog_coverage": sum(rank is not None for _, rank in rows) / max(1, n_targets),
        "macro_user_unconditional": ranking,
        "serving_diagnostics": ({
            **exposure_metrics(index_recs, catalog_size=catalog_size, top_k=10),
            "head_share_top10": sum(index in head for index in flat) / max(1, len(flat)),
            "repeat_rate_top10": repeat_rate(index_recs, recent_by_anchor, top_k=10),
            "top1_item_unique_fraction": len(set(top_item_indices)) / max(1, len(top_item_indices)),
            "mean_duration_seconds_top10": (
                sum(selected_durations) / len(selected_durations) if selected_durations else None
            ),
        } if include_serving else None),
        "per_user_ranks": dict(ranks_by_user),
        "per_user_top_item_ids": dict(recs_by_user),
    }


def _user_ndcg(ranks: dict[str, list[int | None]]) -> dict[int, float]:
    return {
        int(uid): float(np.mean([single_target_metrics(rank)["ndcg@10"] for rank in values]))
        for uid, values in ranks.items() if values
    }


def evaluate_smoothed_markov(
    config_path: str | Path = "configs/base.yaml",
    output_path: str | Path = "artifacts/supplementary_test/smoothed_markov.json",
    alpha_grid: Sequence[float] = (1.0, 10.0, 100.0, 1000.0),
    seed: int = 17,
    anchors_per_user: int = 10,
    reference_evaluation_path: str | Path = "artifacts/final_test_evaluation.json",
) -> dict[str, Any]:
    """Choose transition smoothing on validation, then describe it on the opened test.

    The test half is explicitly a supplementary diagnostic because the project
    test evaluation had already been opened before this smoothed baseline was added.
    """
    import polars as pl

    from tlab.config import load_config

    started = time.perf_counter()
    if not alpha_grid or any(not math.isfinite(float(alpha)) or float(alpha) < 0 for alpha in alpha_grid):
        raise ValueError("alpha_grid values must be finite and non-negative")
    if anchors_per_user < 1:
        raise ValueError("anchors_per_user must be positive")
    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    protocol_path = processed_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(
        processed_dir / "sequence_index",
        expected_source_sha256=protocol["processed_listens_sha256"],
    )
    catalog = pl.read_parquet(processed_dir / "catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog.get_column("item_id").to_list(), dtype=np.int64)
    training_counts = np.asarray(catalog.get_column("training_count").to_list(), dtype=np.float64)
    durations = catalog.get_column("catalog_duration_seconds").to_list()
    transition_matrix, transition_examples = _train_transition_matrix(store, len(item_ids))

    validation_anchors = store.evaluation_targets("validation", per_user_limit=anchors_per_user, seed=seed)
    validation = [
        _evaluate_alpha(
            store, validation_anchors, item_ids, training_counts, durations,
            transition_matrix, float(alpha), include_serving=False,
        )
        for alpha in alpha_grid
    ]
    selected = max(
        validation,
        key=lambda row: (row["macro_user_unconditional"]["ndcg@10"], -row["alpha_popularity_pseudocount_mass"]),
    )
    chosen_alpha = float(selected["alpha_popularity_pseudocount_mass"])
    test_anchors = store.evaluation_targets("test", per_user_limit=anchors_per_user, seed=seed)
    test = _evaluate_alpha(store, test_anchors, item_ids, training_counts, durations, transition_matrix, chosen_alpha)

    reference_path = Path(reference_evaluation_path)
    if not reference_path.is_absolute():
        reference_path = root / reference_path
    paired_vs_existing: dict[str, Any] = {}
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        new_by_user = _user_ndcg(test["per_user_ranks"])
        for name, payload in reference.get("models", {}).items():
            if not name.startswith("B3_seed"):
                continue
            other_by_user = _user_ndcg(reference["per_user_ranks"][name])
            common = sorted(new_by_user.keys() & other_by_user.keys())
            difference = {uid: other_by_user[uid] - new_by_user[uid] for uid in common}
            paired_vs_existing[name] = paired_user_bootstrap(difference, repetitions=2000, seed=seed)

    result = {
        "status": "supplementary_post_freeze_diagnostic",
        "not_used_for_model_selection": True,
        "test_split_was_opened_before_this_baseline_was_added": True,
        "selection_split_for_alpha": "validation",
        "validation_alpha_grid": [float(value) for value in alpha_grid],
        "selected_alpha": chosen_alpha,
        "selected_validation_ndcg10": selected["macro_user_unconditional"]["ndcg@10"],
        "n_train_transition_examples": transition_examples,
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
        "protocol_path": str(protocol_path.resolve()),
        "validation_by_alpha": [
            {key: value for key, value in row.items() if key not in {"per_user_ranks", "per_user_top_item_ids"}}
            for row in validation
        ],
        "test_protocol": {"split": "test", "anchors_per_user": anchors_per_user, "anchor_seed": seed},
        "test": test,
        "paired_test_ndcg10_vs_b3": paired_vs_existing,
        "reference_evaluation_path": str(reference_path.resolve()),
        "reference_evaluation_sha256": sha256_file(reference_path) if reference_path.is_file() else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    target = Path(output_path)
    if not target.is_absolute():
        target = root / target
    write_json_atomic(target, result)
    result["output_path"] = str(target.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select popularity smoothing on validation and evaluate a first-order Markov control")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="artifacts/supplementary_test/smoothed_markov.json")
    parser.add_argument("--alpha-grid", default="1,10,100,1000")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--anchors-per-user", type=int, default=10)
    parser.add_argument("--reference-evaluation", default="artifacts/final_test_evaluation.json")
    args = parser.parse_args()
    result = evaluate_smoothed_markov(
        args.config, args.output,
        alpha_grid=tuple(float(value) for value in args.alpha_grid.split(",") if value),
        seed=args.seed, anchors_per_user=args.anchors_per_user,
        reference_evaluation_path=args.reference_evaluation,
    )
    compact = {key: value for key, value in result.items() if key not in {"test", "validation_by_alpha"}}
    compact["test"] = {key: value for key, value in result["test"].items() if key not in {"per_user_ranks", "per_user_top_item_ids"}}
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
