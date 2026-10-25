from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from tlab.data.store import SequenceStore
from tlab.evaluation.metrics import aggregate_macro_user, paired_user_bootstrap, single_target_metrics
from tlab.evaluation.policy_panel import _load_reward_model
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.train.policy import _collate_histories
from tlab.train.checkpoint import source_tree_sha256


def lexicographic_ranking_order(primary: Tensor, secondary: Tensor, item_ids: Tensor) -> Tensor:
    """Rank by primary descending, secondary descending, then raw item ID ascending.

    Inputs may be one-dimensional or batched. Stable sorts preserve the lower-priority
    ordering when higher-priority scores tie.
    """
    squeeze = primary.ndim == 1
    if squeeze:
        primary, secondary, item_ids = primary.unsqueeze(0), secondary.unsqueeze(0), item_ids.unsqueeze(0)
    if primary.shape != secondary.shape or primary.shape != item_ids.shape:
        raise ValueError("ranking score and item ID tensors must have identical shapes")
    by_id = torch.argsort(item_ids, dim=-1, stable=True)
    secondary_order = torch.argsort(secondary.gather(-1, by_id), dim=-1, descending=True, stable=True)
    by_secondary = by_id.gather(-1, secondary_order)
    primary_order = torch.argsort(primary.gather(-1, by_secondary), dim=-1, descending=True, stable=True)
    result = by_secondary.gather(-1, primary_order)
    return result.squeeze(0) if squeeze else result


def _per_user_ndcg(ranks_by_user: dict[int, list[int | None]], k: int = 10) -> dict[int, float]:
    return {
        uid: sum(single_target_metrics(rank, ks=(k,))[f"ndcg@{k}"] for rank in ranks) / len(ranks)
        for uid, ranks in ranks_by_user.items()
        if ranks
    }


def evaluate_top100_rerank(
    config_path: str | Path,
    sft_checkpoint: str | Path,
    rm_fit_checkpoint: str | Path,
    rm_audit_checkpoint: str | Path,
    seed: int = 17,
    per_user_limit: int = 10,
    shortlist_size: int = 100,
    batch_size: int = 32,
    output_path: str | Path | None = None,
    split: str = "validation",
) -> dict[str, Any]:
    """Evaluate deterministic B1 top-K reranking with fit and audit ratio heads."""
    import polars as pl

    from tlab.config import load_config

    started = time.perf_counter()
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if per_user_limit < 1 or shortlist_size < 1 or batch_size < 1:
        raise ValueError("panel, shortlist, and batch sizes must be positive")
    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    store = SequenceStore.load_cache(
        processed_dir / "sequence_index",
        expected_source_sha256=protocol["processed_listens_sha256"],
    )
    catalog = pl.read_parquet(processed_dir / "catalog.parquet").sort("token_id")
    catalog_item_ids = catalog.get_column("item_id").to_list()
    if shortlist_size > len(catalog_item_ids):
        raise ValueError("shortlist size cannot exceed catalog size")
    item_ids = torch.tensor(catalog_item_ids, dtype=torch.long, device=device)
    id_order = torch.argsort(item_ids, stable=True)
    tokens = torch.arange(3, 3 + len(catalog_item_ids), dtype=torch.long, device=device)

    actor = ItemGPT2Actor(ActorShape(
        vocab_size=len(catalog_item_ids) + 3,
        n_positions=int(config.model.get("n_positions", 256)),
        n_layer=int(config.model.get("n_layer", 4)),
        n_embd=int(config.model.get("n_embd", 256)),
        n_head=int(config.model.get("n_head", 4)),
        dropout=float(config.model.get("dropout", 0.1)),
    ))
    payload = torch.load(sft_checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(payload["model"])
    actor.to(device).eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    fit_model, fit_features = _load_reward_model(config, rm_fit_checkpoint, "fit", device)
    audit_model, audit_features = _load_reward_model(config, rm_audit_checkpoint, "audit", device)

    anchors = store.evaluation_targets(split, per_user_limit=per_user_limit, seed=seed)
    if not anchors:
        raise ValueError(f"{split} panel is empty")
    history_length = int(config.dataset.get("history_length", 200))
    fit_ranks: dict[int, list[int | None]] = defaultdict(list)
    audit_ranks: dict[int, list[int | None]] = defaultdict(list)
    sft_ranks: dict[int, list[int | None]] = defaultdict(list)
    target_in_shortlist = target_in_catalog = 0
    fit_top1_unsupported = audit_top1_unsupported = 0
    total_targets = 0

    def build_features(candidate_tokens: Tensor, histories: list[list[int]], features: dict[int, tuple[float, bool, int]]):
        token_rows = candidate_tokens.detach().cpu().tolist()
        durations, missing, support, repeats = [], [], [], []
        for history, row_tokens in zip(histories, token_rows):
            recent = set(int(token) for token in history[-20:] if int(token) >= 3)
            row_features = [features.get(int(token), (0.0, True, 0)) for token in row_tokens]
            durations.append([float(value[0]) for value in row_features])
            missing.append([bool(value[1]) for value in row_features])
            support.append([int(value[2]) for value in row_features])
            repeats.append([int(token) in recent for token in row_tokens])
        return (
            torch.tensor(durations, dtype=torch.float32, device=device),
            torch.tensor(missing, dtype=torch.bool, device=device),
            torch.tensor(support, dtype=torch.long, device=device),
            torch.tensor(repeats, dtype=torch.bool, device=device),
        )

    with torch.no_grad():
        for start in range(0, len(anchors), batch_size):
            batch_anchors = anchors[start:start + batch_size]
            histories = store.histories_for(batch_anchors, history_length=history_length)
            input_ids, attention = _collate_histories(histories, device)
            actor_log = actor.next_action_log_probs(input_ids, attention)[:, 3:].float()
            batch_size_now = len(batch_anchors)
            item_ids_batch = item_ids.unsqueeze(0).expand(batch_size_now, -1)
            actor_order = lexicographic_ranking_order(actor_log, actor_log, item_ids_batch)
            shortlist_index = actor_order[:, :shortlist_size]
            shortlist_tokens = tokens[shortlist_index]
            shortlist_actor_scores = actor_log.gather(1, shortlist_index)
            shortlist_item_ids = item_ids[shortlist_index]

            fit_d, fit_m, fit_support, fit_repeat = build_features(shortlist_tokens, histories, fit_features)
            audit_d, audit_m, audit_support, audit_repeat = build_features(shortlist_tokens, histories, audit_features)
            fit_reward = fit_model.score_catalog(
                input_ids, attention, shortlist_tokens, fit_d, fit_m, fit_repeat, chunk_size=shortlist_size,
            )["ratio"].float()
            audit_reward = audit_model.score_catalog(
                input_ids, attention, shortlist_tokens, audit_d, audit_m, audit_repeat, chunk_size=shortlist_size,
            )["ratio"].float()
            fit_order = lexicographic_ranking_order(fit_reward, shortlist_actor_scores, shortlist_item_ids)
            audit_order = lexicographic_ranking_order(audit_reward, shortlist_actor_scores, shortlist_item_ids)
            fit_ranked_tokens = shortlist_tokens.gather(1, fit_order)
            audit_ranked_tokens = shortlist_tokens.gather(1, audit_order)
            fit_ranked_support = fit_support.gather(1, fit_order)
            audit_ranked_support = audit_support.gather(1, audit_order)

            for row, (user_index, position) in enumerate(batch_anchors):
                user = store.users[user_index]
                uid = int(user.uid)
                target_token = int(user.tokens[position])
                if 3 <= target_token < len(catalog_item_ids) + 3:
                    target_index = target_token - 3
                    target_score = actor_log[row, target_index]
                    target_id_tensor = item_ids[target_index]
                    baseline_rank = int((
                        (actor_log[row] > target_score)
                        | ((actor_log[row] == target_score) & (item_ids < target_id_tensor))
                    ).sum().item()) + 1
                    target_in_catalog += 1
                else:
                    baseline_rank = None
                sft_ranks[uid].append(baseline_rank)

                fit_match = torch.nonzero(fit_ranked_tokens[row] == target_token, as_tuple=False).flatten()
                audit_match = torch.nonzero(audit_ranked_tokens[row] == target_token, as_tuple=False).flatten()
                fit_rank = int(fit_match[0].item()) + 1 if len(fit_match) else None
                audit_rank = int(audit_match[0].item()) + 1 if len(audit_match) else None
                fit_ranks[uid].append(fit_rank)
                audit_ranks[uid].append(audit_rank)
                target_in_shortlist += int(fit_rank is not None)
                fit_top1_unsupported += int(fit_ranked_support[row, 0].item() == 0)
                audit_top1_unsupported += int(audit_ranked_support[row, 0].item() == 0)
                total_targets += 1

    baseline_rows = [(uid, rank) for uid, ranks in sft_ranks.items() for rank in ranks]
    fit_rows = [(uid, rank) for uid, ranks in fit_ranks.items() for rank in ranks]
    audit_rows = [(uid, rank) for uid, ranks in audit_ranks.items() for rank in ranks]
    baseline_user_ndcg = _per_user_ndcg(sft_ranks)
    fit_user_ndcg = _per_user_ndcg(fit_ranks)
    audit_user_ndcg = _per_user_ndcg(audit_ranks)
    results = {}
    for name, rows, user_values in (("rm_fit", fit_rows, fit_user_ndcg), ("rm_audit", audit_rows, audit_user_ndcg)):
        differences = {uid: user_values[uid] - baseline_user_ndcg[uid] for uid in baseline_user_ndcg}
        results[name] = {
            "macro_user_unconditional": aggregate_macro_user(rows),
            "paired_user_ndcg10_difference_vs_sft": paired_user_bootstrap(differences, repetitions=2000, seed=seed),
            "unsupported_top1_rate": (fit_top1_unsupported if name == "rm_fit" else audit_top1_unsupported) / total_targets,
        }

    result = {
        "experiment": "B4 top-100 deterministic RM rerank",
        "seed": seed,
        "config_path": str(config.path.resolve()),
        "sft_checkpoint": str(Path(sft_checkpoint).resolve()),
        "rm_fit_checkpoint": str(Path(rm_fit_checkpoint).resolve()),
        "rm_audit_checkpoint": str(Path(rm_audit_checkpoint).resolve()),
        "source_sha256": source_tree_sha256(root / "src" / "tlab"),
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
        "split": split,
        "per_user_anchor_limit": per_user_limit,
        "shortlist_size": shortlist_size,
        "n_users": len(sft_ranks),
        "n_targets": total_targets,
        "target_catalog_coverage": target_in_catalog / total_targets,
        "target_shortlist_coverage": target_in_shortlist / total_targets,
        "target_shortlist_coverage_in_catalog": target_in_shortlist / max(1, target_in_catalog),
        "sft": {"macro_user_unconditional": aggregate_macro_user(baseline_rows)},
        "reranked": results,
        "per_user_ranks": {
            "sft": sft_ranks,
            "rm_fit": fit_ranks,
            "rm_audit": audit_ranks,
        },
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    target_path = Path(output_path) if output_path else root / "artifacts" / f"b4_top{shortlist_size}_{split}_seed{seed}.json"
    if not target_path.is_absolute():
        target_path = root / target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    result["output_path"] = str(target_path.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic top-K SFT reranking by RM-fit and audit")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--rm-fit-checkpoint", required=True)
    parser.add_argument("--rm-audit-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-user-limit", type=int, default=10)
    parser.add_argument("--shortlist-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_top100_rerank(
        config_path=args.config,
        sft_checkpoint=args.sft_checkpoint,
        rm_fit_checkpoint=args.rm_fit_checkpoint,
        rm_audit_checkpoint=args.rm_audit_checkpoint,
        seed=args.seed,
        per_user_limit=args.per_user_limit,
        shortlist_size=args.shortlist_size,
        batch_size=args.batch_size,
        output_path=args.output,
        split=args.split,
    )
    summary = {key: value for key, value in result.items() if key != "per_user_ranks"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
