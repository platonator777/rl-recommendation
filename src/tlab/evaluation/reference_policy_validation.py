from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from tlab.config import load_config
from tlab.data.manifest import sha256_file
from tlab.data.store import SequenceStore
from tlab.evaluation.behavior import exposure_metrics, repeat_rate
from tlab.evaluation.metrics import paired_user_bootstrap, single_target_metrics
from tlab.evaluation.ranking import evaluate_full_catalog
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.train.checkpoint import source_tree_sha256


def _parse_pairs(values: list[str], kind: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        left, separator, right = value.partition("=")
        if not separator or not left or not right or left in result:
            raise ValueError(f"{kind} values must be unique LABEL=PATH or POLICY=REFERENCE pairs")
        result[left] = right
    return result


def _per_user_metrics(per_user_ranks: dict[str, list[int | None]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for user_id, ranks in per_user_ranks.items():
        rows = [single_target_metrics(rank) for rank in ranks]
        if not rows:
            continue
        result[str(user_id)] = {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in rows[0]
        }
    return result


def _behavior_metrics(
    recommendations: dict[str, list[list[int]]],
    recent_by_user: dict[str, set[int]],
    catalog_item_ids: list[int],
    training_counts: list[int],
    durations: list[float | None],
    fit_support_by_raw_id: dict[int, int],
) -> dict[str, Any]:
    raw_to_index = {int(item_id): index for index, item_id in enumerate(catalog_item_ids)}
    rec_rows = [items for user_id in sorted(recommendations) for items in recommendations[user_id]]
    indexed_rows = [[raw_to_index[item] for item in row if item in raw_to_index] for row in rec_rows]
    exposure = exposure_metrics(indexed_rows, len(catalog_item_ids), top_k=10)
    head_count = max(1, math.ceil(0.01 * len(catalog_item_ids)))
    head_indices = set(sorted(
        range(len(catalog_item_ids)),
        key=lambda index: (-training_counts[index], catalog_item_ids[index]),
    )[:head_count])
    n_top_items = sum(len(row[:10]) for row in indexed_rows)
    head_share = (
        sum(index in head_indices for row in indexed_rows for index in row[:10]) / n_top_items
        if n_top_items else 0.0
    )
    rec_by_user = [recommendations[user_id][0][:10] for user_id in sorted(recommendations)]
    recent_rows = [recent_by_user.get(user_id, set()) for user_id in sorted(recommendations)]
    repeat = repeat_rate(rec_by_user, recent_rows, top_k=10)
    duration_values = [
        float(durations[raw_to_index[item]])
        for row in rec_rows
        for item in row[:10]
        if item in raw_to_index
        and durations[raw_to_index[item]] is not None
        and math.isfinite(float(durations[raw_to_index[item]]))
    ]
    unsupported = [
        fit_support_by_raw_id.get(int(item), 0) == 0
        for row in rec_rows
        for item in row[:10]
        if int(item) in raw_to_index
    ]
    return {
        "catalog_exposure": exposure,
        "head_item_count": head_count,
        "head_share_top10": head_share,
        "repeat_rate_top10": repeat,
        "mean_recommended_duration_seconds": (
            sum(duration_values) / len(duration_values) if duration_values else None
        ),
        "duration_observed_recommendations": len(duration_values),
        "rm_fit_unsupported_fraction_top10": (
            sum(unsupported) / len(unsupported) if unsupported else None
        ),
    }


def evaluate_reference_validation(
    config_path: str | Path,
    checkpoints: dict[str, str | Path],
    references: dict[str, str],
    output_path: str | Path,
    split: str = "validation",
    seed: int = 17,
    batch_size: int = 16,
    bootstrap_repetitions: int = 2000,
) -> dict[str, Any]:
    if split != "validation":
        raise ValueError("this evaluator is validation-only; Test and RM_audit are intentionally not loaded")
    if batch_size < 1 or not checkpoints:
        raise ValueError("at least one checkpoint and a positive batch size are required")
    started = time.perf_counter()
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
    catalog_item_ids = [int(value) for value in catalog.get_column("item_id").to_list()]
    training_counts = [int(value) for value in catalog.get_column("training_count").to_list()]
    durations = catalog.get_column("catalog_duration_seconds").to_list()
    tokens_to_raw = {index + 3: item_id for index, item_id in enumerate(catalog_item_ids)}
    support_path = processed_dir / "rm_item_features_fit.parquet"
    fit_support_by_raw_id: dict[int, int] = {}
    if support_path.is_file():
        support_frame = pl.read_parquet(support_path, columns=["token_id", "rm_training_support"])
        fit_support_by_raw_id = {
            tokens_to_raw[int(token)]: int(support or 0)
            for token, support in support_frame.iter_rows()
            if int(token) in tokens_to_raw
        }

    anchors = store.evaluation_targets(split, per_user_limit=1, seed=seed)
    recent_by_user: dict[str, set[int]] = {}
    catalog_item_id_set = set(catalog_item_ids)
    for user_index, position in anchors:
        user = store.users[user_index]
        raw_ids = np.frombuffer(user.raw_item_ids, dtype=np.int64)
        recent_by_user[str(user.uid)] = {
            int(item_id)
            for item_id in raw_ids[max(0, position - 20):position]
            if int(item_id) in catalog_item_id_set
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    shape = ActorShape(
        vocab_size=len(catalog_item_ids) + 3,
        n_positions=int(config.model.get("n_positions", 256)),
        n_layer=int(config.model.get("n_layer", 4)),
        n_embd=int(config.model.get("n_embd", 256)),
        n_head=int(config.model.get("n_head", 4)),
        dropout=float(config.model.get("dropout", 0.1)),
    )
    results: dict[str, dict[str, Any]] = {}
    target = Path(output_path)
    if not target.is_absolute():
        target = root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial_target = target.with_name(target.name + ".partial")
    source_hash = source_tree_sha256(root / "src" / "tlab")
    for label, checkpoint_path in checkpoints.items():
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = (root / path).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        actor = ItemGPT2Actor(shape)
        actor.load_state_dict(payload["model"])
        actor.to(device).eval()
        inference_started = time.perf_counter()
        ranking = evaluate_full_catalog(
            actor,
            store,
            catalog_item_ids,
            split=split,
            history_length=int(config.dataset.get("history_length", 200)),
            per_user_limit=1,
            seed=seed,
            batch_size=batch_size,
            top_k=10,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_started
        user_metrics = _per_user_metrics(ranking["per_user_ranks"])
        results[label] = {
            "checkpoint_path": str(path),
            "checkpoint_sha256": sha256_file(path),
            "update": payload.get("metadata", {}).get("update"),
            "split": split,
            "seed": seed,
            "metrics": {
                "n_users": ranking["n_users"],
                "n_targets": ranking["n_targets"],
                "target_coverage": ranking["target_coverage"],
                "n_in_catalog_targets": ranking["n_in_catalog_targets"],
                "in_catalog_cross_entropy": ranking["in_catalog_cross_entropy"],
                "macro_user_unconditional": ranking["macro_user_unconditional"],
                "macro_user_in_catalog": ranking["macro_user_in_catalog"],
            },
            "behavior": _behavior_metrics(
                ranking["per_user_top_item_ids"],
                recent_by_user,
                catalog_item_ids,
                training_counts,
                durations,
                fit_support_by_raw_id,
            ),
            "compute": {"inference_seconds": inference_seconds},
            "per_user_metrics": user_metrics,
        }
        del actor
        partial_target.write_text(json.dumps({
            "status": "running",
            "config_path": str(config.path.resolve()),
            "source_sha256": source_hash,
            "split": split,
            "seed": seed,
            "models_completed": len(results),
            "models_planned": len(checkpoints),
            "models": results,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"evaluated {label} ({len(results)}/{len(checkpoints)})", flush=True)

    paired: dict[str, Any] = {}
    for policy, reference in references.items():
        if policy not in results or reference not in results:
            raise ValueError(f"paired reference {policy}={reference} is not present in checkpoints")
        policy_row = results[policy]
        reference_row = results[reference]
        common_users = sorted(
            policy_row["per_user_metrics"].keys() & reference_row["per_user_metrics"].keys()
        )
        deltas = {
            metric: {
                int(user): policy_row["per_user_metrics"][user][metric]
                - reference_row["per_user_metrics"][user][metric]
                for user in common_users
            }
            for metric in ("ndcg@10", "recall@10", "recall@20", "recall@100")
        }
        ndcg = deltas["ndcg@10"]
        base_ndcg = reference_row["metrics"]["macro_user_unconditional"]["ndcg@10"]
        base_behavior = reference_row["behavior"]
        policy_behavior = policy_row["behavior"]
        policy_ndcg = policy_row["metrics"]["macro_user_unconditional"]["ndcg@10"]
        coverage_base = float(base_behavior["catalog_exposure"]["coverage"])
        coverage_policy = float(policy_behavior["catalog_exposure"]["coverage"])
        coverage_relative_change = (
            coverage_policy / coverage_base - 1.0 if coverage_base > 0 else None
        )
        ndcg_relative_change = policy_ndcg / base_ndcg - 1.0 if base_ndcg > 0 else None
        head_delta = float(policy_behavior["head_share_top10"]) - float(base_behavior["head_share_top10"])
        repeat_delta = float(policy_behavior["repeat_rate_top10"]) - float(base_behavior["repeat_rate_top10"])
        bootstrap = {
            metric: paired_user_bootstrap(
                values,
                repetitions=bootstrap_repetitions,
                seed=seed,
            )
            for metric, values in deltas.items()
        }
        paired[policy] = {
            "reference": reference,
            "paired_users": len(common_users),
            "bootstrap": bootstrap,
            "point_changes": {
                "ndcg@10_absolute": policy_ndcg - base_ndcg,
                "ndcg@10_relative": ndcg_relative_change,
                "exposure_coverage_relative": coverage_relative_change,
                "head_share_top10_absolute": head_delta,
                "repeat_rate_top10_absolute": repeat_delta,
            },
            "point_guardrails": {
                "ndcg_loss_at_most_2pct": ndcg_relative_change is not None and ndcg_relative_change >= -0.02,
                "coverage_loss_at_most_10pct": coverage_relative_change is not None and coverage_relative_change >= -0.10,
                "head_share_increase_at_most_5pp": head_delta <= 0.05,
                "repeat_increase_at_most_5pp": repeat_delta <= 0.05,
            },
            "ci_guardrails": {
                "ndcg_ci_lower_within_2pct_loss": (
                    ndcg_relative_change is not None
                    and bootstrap["ndcg@10"]["ci_95_low"] >= -0.02 * base_ndcg
                ),
            },
        }

    result = {
        "config_path": str(config.path.resolve()),
        "source_sha256": source_hash,
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
        "split": split,
        "seed": seed,
        "bootstrap_repetitions": bootstrap_repetitions,
        "panel": {"n_users": len(anchors), "anchor_protocol": "one target per user, fixed validation seed"},
        "models": results,
        "paired_to_reference": paired,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    partial_target.unlink(missing_ok=True)
    result["output_path"] = str(target.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="reference validation-only full-catalog policy and behavior comparison; does not load RM_audit or Test"
    )
    parser.add_argument("--config", default="configs/reference_rl.yaml")
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=CHECKPOINT_PATH")
    parser.add_argument("--reference", action="append", default=[], help="POLICY_LABEL=REFERENCE_LABEL")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoints = _parse_pairs(args.checkpoint, "checkpoint")
    references = _parse_pairs(args.reference, "reference")
    print(json.dumps(evaluate_reference_validation(
        args.config,
        checkpoints,
        references,
        args.output,
        seed=args.seed,
        batch_size=args.batch_size,
        bootstrap_repetitions=args.bootstrap_repetitions,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
