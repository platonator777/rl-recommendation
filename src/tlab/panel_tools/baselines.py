from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch

from tlab.data.store import SequenceStore
from tlab.evaluation.baselines import _train_transition_matrix, smoothed_transition_scores
from tlab.evaluation.behavior import exposure_metrics, repeat_rate
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.panel_tools.data import PanelTarget, history_for_target
from tlab.panel_tools.evaluation import target_rank, single_target_metrics
from tlab.panel_tools.evaluator import _collate_histories
from tlab.panel_tools.statistics import paired_user_bootstrap, relative_quality_guard


BASELINE_NAMES = ("base_actor", "train_popularity", "recent_train_popularity_14d", "smoothed_markov", "history_frequency", "history_frequency_reranker")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recent_train_popularity(
    processed_listens_path: str | Path,
    *,
    catalog_tokens: np.ndarray,
    train_cutoff: float,
    window_seconds: float = 14 * 86400,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Count known catalog listens in a fixed, pre-cutoff recent train window."""
    path = Path(processed_listens_path)
    if window_seconds <= 0 or not np.isfinite(window_seconds):
        raise ValueError("recent popularity window must be finite and positive")
    tokens = np.asarray(catalog_tokens, dtype=np.int64)
    if tokens.ndim != 1 or not len(tokens) or len(set(tokens.tolist())) != len(tokens):
        raise ValueError("catalog token IDs must be a non-empty unique vector")
    expected = np.arange(3, 3 + len(tokens), dtype=np.int64)
    if not np.array_equal(tokens, expected):
        raise ValueError("catalog tokens must be contiguous and ordered from token ID 3")
    started = time.perf_counter()
    lazy = pl.scan_parquet(path)
    schema = lazy.collect_schema()
    required = {"timestamp", "token_id", "global_split"}
    if not required <= set(schema.names()):
        raise ValueError(f"processed listens parquet is missing columns: {sorted(required - set(schema.names()))}")
    lower = float(train_cutoff) - float(window_seconds)
    frame = (
        lazy.filter(
            (pl.col("global_split") == "train")
            & (pl.col("timestamp") >= lower)
            & (pl.col("timestamp") < float(train_cutoff))
            & (pl.col("token_id") >= 3)
        )
        .group_by("token_id")
        .agg(pl.len().alias("count"))
        .collect(engine="streaming")
    )
    counts = np.zeros(len(tokens), dtype=np.float64)
    for token_id, count in frame.iter_rows():
        index = int(token_id) - 3
        if 0 <= index < len(counts):
            counts[index] = float(count)
    return counts, {
        "window_seconds": float(window_seconds),
        "window_start_inclusive": lower,
        "train_cutoff_exclusive": float(train_cutoff),
        "known_catalog_events": int(counts.sum()),
        "n_supported_items": int(np.count_nonzero(counts)),
        "source_path": str(path),
        "source_sha256": _sha256(path),
        "query_seconds": time.perf_counter() - started,
        "population": "all known catalog events in global_split=train; organic and non-organic both included",
    }


def _history_counts(history_raw_ids: Sequence[int], raw_to_index: dict[int, int], catalog_size: int) -> np.ndarray:
    indices = [raw_to_index[int(item)] for item in history_raw_ids if int(item) in raw_to_index]
    return np.bincount(indices, minlength=catalog_size).astype(np.int64, copy=False)


def _actor_tie_rank(base_logp: np.ndarray, raw_item_ids: np.ndarray) -> np.ndarray:
    order = np.lexsort((raw_item_ids, -np.asarray(base_logp, dtype=np.float64)))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order), dtype=np.int64)
    return ranks


def _user_macro(values: dict[str, dict[int, list[float]]], k: int) -> dict[str, Any]:
    metric_names = (f"ndcg@{k}", f"recall@{k}")
    per_user = {
        str(uid): {metric: float(np.mean([row[metric] for row in rows])) for metric in metric_names}
        for uid, rows in values.items() if rows
    }
    summary = {
        metric: float(np.mean([row[metric] for row in per_user.values()])) if per_user else None
        for metric in metric_names
    }
    summary.update({"n_users": len(per_user), "n_events": sum(len(rows) for rows in values.values()), "per_user": per_user})
    return summary


@torch.inference_mode()
def evaluate_core_controls(
    actor: ItemGPT2Actor,
    store: SequenceStore,
    panel: Sequence[PanelTarget],
    *,
    catalog_item_ids: Sequence[int],
    train_popularity_counts: Sequence[float],
    recent_popularity_counts: Sequence[float],
    markov_alpha: float = 1.0,
    history_events: int = 200,
    batch_size: int = 16,
    k: int = 10,
) -> dict[str, Any]:
    if not panel or batch_size < 1 or k < 1:
        raise ValueError("controls require a non-empty panel and positive batch/top-k")
    item_ids = np.asarray(catalog_item_ids, dtype=np.int64)
    train_counts = np.asarray(train_popularity_counts, dtype=np.float64)
    recent_counts = np.asarray(recent_popularity_counts, dtype=np.float64)
    if item_ids.ndim != 1 or len(set(item_ids.tolist())) != len(item_ids):
        raise ValueError("catalog item IDs must be a unique vector")
    if train_counts.shape != item_ids.shape or recent_counts.shape != item_ids.shape:
        raise ValueError("popularity vectors must align with catalog item IDs")
    if markov_alpha < 0 or not math.isfinite(markov_alpha):
        raise ValueError("Markov alpha must be finite and non-negative")
    raw_to_index = {int(item): index for index, item in enumerate(item_ids.tolist())}
    device = next(actor.parameters()).device
    actor.eval()
    transition_matrix, transition_examples = _train_transition_matrix(store, len(item_ids))
    primary: dict[str, dict[int, list[float]]] = {name: defaultdict(list) for name in BASELINE_NAMES}
    slices: dict[str, dict[str, dict[int, list[float]]]] = {
        name: {slice_name: defaultdict(list) for slice_name in ("positive", "is_new", "is_repeat", "known_catalog")}
        for name in BASELINE_NAMES
    }
    recommendations: dict[str, list[list[int]]] = {name: [] for name in BASELINE_NAMES}
    recent_history_indices: list[set[int]] = []
    covered: dict[str, int] = {name: 0 for name in BASELINE_NAMES}

    for start in range(0, len(panel), batch_size):
        targets = panel[start:start + batch_size]
        histories = [history_for_target(store, target, history_events) for target in targets]
        input_ids, attention = _collate_histories(histories, device)
        base_logp = actor.next_action_log_probs(input_ids, attention)[:, 3:].float().cpu().numpy()
        for local_index, target in enumerate(targets):
            user = store.users[target.user_index]
            if int(user.uid) != target.uid:
                raise ValueError("panel user index does not match sequence store")
            scores_base = np.asarray(base_logp[local_index], dtype=np.float64)
            history_start = max(0, target.position - history_events)
            raw_history = np.asarray(user.raw_item_ids[history_start:target.position], dtype=np.int64)
            history_count = _history_counts(raw_history.tolist(), raw_to_index, len(item_ids))
            history20 = np.asarray(user.raw_item_ids[max(0, target.position - 20):target.position], dtype=np.int64)
            recent_history_indices.append({raw_to_index[item] for item in history20.tolist() if item in raw_to_index})
            previous_token = int(user.tokens[target.position - 1]) if target.position > 0 else 1
            markov_score = smoothed_transition_scores(transition_matrix.getrow(previous_token), train_counts, markov_alpha)
            actor_rank = _actor_tie_rank(scores_base, item_ids)
            rerank_score = history_count.astype(np.float64) * (len(item_ids) + 1.0) + (len(item_ids) - actor_rank)
            score_map = {
                "base_actor": scores_base,
                "train_popularity": train_counts,
                "recent_train_popularity_14d": recent_counts,
                "smoothed_markov": markov_score,
                "history_frequency": history_count,
                "history_frequency_reranker": rerank_score,
            }
            target_index = raw_to_index.get(int(target.raw_item_id))
            target_id_for_rank = int(target.raw_item_id)
            for name, scores in score_map.items():
                rank = target_rank(scores, item_ids, target_id_for_rank)
                metrics = single_target_metrics(rank, k)
                primary[name][target.uid].append(metrics)
                covered[name] += int(rank is not None)
                if target.positive:
                    slices[name]["positive"][target.uid].append(metrics)
                if target.is_new:
                    slices[name]["is_new"][target.uid].append(metrics)
                if target.is_repeat:
                    slices[name]["is_repeat"][target.uid].append(metrics)
                if target_index is not None:
                    slices[name]["known_catalog"][target.uid].append(metrics)
                order = np.lexsort((item_ids, -np.asarray(scores, dtype=np.float64)))[:k]
                recommendations[name].append(order.tolist())

    model_metrics: dict[str, Any] = {}
    per_user_primary: dict[str, dict[int, float]] = {}
    for name in BASELINE_NAMES:
        primary_summary = _user_macro(primary[name], k)
        per_user_primary[name] = {int(uid): value[f"ndcg@{k}"] for uid, value in primary_summary["per_user"].items()}
        slice_summaries = {
            slice_name: _user_macro(slices[name][slice_name], k)
            for slice_name in slices[name]
        }
        model_metrics[name] = {
            "primary_user_macro": primary_summary,
            "slices": slice_summaries,
            "target_coverage": covered[name] / len(panel),
            "exposure": exposure_metrics(recommendations[name], len(item_ids), top_k=k),
            "repeat_share_topk": repeat_rate(recommendations[name], recent_history_indices, top_k=k),
        }
    paired = {}
    base_users = per_user_primary["base_actor"]
    for name in BASELINE_NAMES[1:]:
        paired[name] = {
            "paired_ndcg10_vs_base": paired_user_bootstrap(
                per_user_primary[name], base_users, n_bootstrap=2000, seed=17017,
            ),
            "relative_2_percent_guard_vs_base": relative_quality_guard(
                per_user_primary[name], base_users, tolerance=0.02, n_bootstrap=2000, seed=17018,
            ),
        }
    return {
        "catalog_size": len(item_ids),
        "panel_events": len(panel),
        "panel_users": len({row.uid for row in panel}),
        "top_k": k,
        "history_events": history_events,
        "markov": {
            "alpha_popularity_pseudocount_mass": float(markov_alpha),
            "selection": "fixed at 1.0 before panel_tools policy comparisons; no panel_tools alpha sweep",
            "training_transition_examples": transition_examples,
        },
        "models": model_metrics,
        "paired_vs_base_actor": paired,
        "aggregation": "event metrics averaged within user, then user-macro; one frozen event per user",
        "tie_break": "score descending, raw item ID ascending; reranker uses base actor order only to break equal history counts",
        "oov_targets": "unconditional misses retained in the denominator",
    }


def run_core_controls(
    workspace: str | Path,
    *,
    panel_path: str | Path,
    output_path: str | Path,
    actor_checkpoint: str | Path,
    recent_window_days: float = 14.0,
    markov_alpha: float = 1.0,
    batch_size: int = 16,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    submission = (root / "submission").resolve()
    if output.resolve() == submission or submission in output.resolve().parents:
        raise ValueError("panel_tools baseline outputs may not be written inside submission/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite panel_tools baseline result: {output}")
    panel_file = Path(panel_path)
    if not panel_file.is_absolute():
        panel_file = root / panel_file
    checkpoint = Path(actor_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    protocol = json.loads((root / "artifacts/panel_tools/protocol.json").read_text(encoding="utf-8"))
    dataset = protocol["dataset"]
    processed = root / dataset.get("processed_full_model_tminus14_dir", "data/processed_full_model_tminus14")
    source_sha = dataset["processed_full_model_tminus14_listens_sha256"]
    store = SequenceStore.load_cache(processed / "sequence_index", expected_source_sha256=source_sha)
    catalog = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog.get_column("item_id").to_list(), dtype=np.int64)
    tokens = np.asarray(catalog.get_column("token_id").to_list(), dtype=np.int64)
    train_counts = np.asarray(catalog.get_column("training_count").to_list(), dtype=np.float64)
    proc_protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    cutoff = float(proc_protocol["boundaries_source_units"]["train_cutoff_raw"])
    recent_counts, recent_manifest = recent_train_popularity(
        processed / "listens.parquet", catalog_tokens=tokens, train_cutoff=cutoff,
        window_seconds=float(recent_window_days) * 86400.0,
    )
    panel = [PanelTarget(**json.loads(line)) for line in panel_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(panel) != len({target.uid for target in panel}):
        raise ValueError("the frozen panel_tools core evaluation panel must have exactly one event per user")
    actor = ItemGPT2Actor(ActorShape(vocab_size=len(item_ids) + 3, n_positions=256, n_layer=4, n_embd=256, n_head=4, dropout=0.0))
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint_payload["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor.to(device).eval()
    started = time.perf_counter()
    metrics = evaluate_core_controls(
        actor, store, panel, catalog_item_ids=item_ids, train_popularity_counts=train_counts,
        recent_popularity_counts=recent_counts, markov_alpha=markov_alpha,
        history_events=200, batch_size=batch_size, k=10,
    )
    result = {
        "protocol_id": protocol["protocol_id"],
        "status": "completed",
        "evaluation_label": "frozen retrospective evaluation",
        "not_independent_test": True,
        "panel_path": str(panel_file.relative_to(root)),
        "panel_sha256": _sha256(panel_file),
        "catalog_path": str((processed / "catalog.parquet").relative_to(root)),
        "catalog_file_sha256": _sha256(processed / "catalog.parquet"),
        "actor_checkpoint_path": str(checkpoint.relative_to(root)),
        "actor_checkpoint_sha256": _sha256(checkpoint),
        "recent_popularity": recent_manifest,
        "runtime": {"device": str(device), "evaluation_seconds": time.perf_counter() - started,
                    "batch_size": batch_size, "data_load_and_recent_count_seconds_not_included": True},
        **metrics,
        "limitations": [
            "All outcomes are retrospective; validation and historical test periods were explored in V1-full_model.",
            "The base actor is an existing full_model model. Controls are fixed rule-based baselines, not causal policy evaluations.",
            "Recent popularity counts all known catalog training listens in the last 14 days before the train cutoff; it is a popularity control, not a user preference estimator.",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {"output_path": str(output), "result": result}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed panel_tools popularity, Markov, history and actor controls")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--panel", default="artifacts/panel_tools/evaluation/frozen_primary_validation_panel.jsonl")
    parser.add_argument("--output", default="artifacts/panel_tools/baselines/core_controls_validation.json")
    parser.add_argument("--actor", default="models/actor/weak.pt")
    parser.add_argument("--recent-window-days", type=float, default=14.0)
    parser.add_argument("--markov-alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    result = run_core_controls(
        args.workspace, panel_path=args.panel, output_path=args.output, actor_checkpoint=args.actor,
        recent_window_days=args.recent_window_days, markov_alpha=args.markov_alpha, batch_size=args.batch_size,
    )
    compact = {key: value for key, value in result["result"].items() if key not in {"models", "paired_vs_base_actor"}}
    compact["models"] = {
        name: {"ndcg@10": value["primary_user_macro"]["ndcg@10"],
              "recall@10": value["primary_user_macro"]["recall@10"],
              "n_users": value["primary_user_macro"]["n_users"],
              "target_coverage": value["target_coverage"]}
        for name, value in result["result"]["models"].items()
    }
    print(json.dumps({"output_path": result["output_path"], **compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
