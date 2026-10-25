from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from tlab.data.catalog import BOS_ID
from tlab.data.store import SequenceStore
from tlab.evaluation.metrics import single_target_metrics
from tlab.evaluation.policy_panel import _load_reward_model
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.train.checkpoint import source_tree_sha256
from tlab.train.policy import _collate_histories, _score_actions


def _rank_logged_targets(log_probs: Tensor, target_tokens: list[int], item_ids: Tensor) -> list[int | None]:
    catalog_log_probs = log_probs[:, 3:].float()
    ranks: list[int | None] = []
    for row, target in enumerate(target_tokens):
        if target < 3 or target >= len(item_ids) + 3:
            ranks.append(None)
            continue
        target_index = target - 3
        target_score = catalog_log_probs[row, target_index]
        target_item_id = item_ids[target_index]
        rank = int((
            (catalog_log_probs[row] > target_score)
            | ((catalog_log_probs[row] == target_score) & (item_ids < target_item_id))
        ).sum().item()) + 1
        ranks.append(rank)
    return ranks


def _append_synthetic(history: list[int], action: int, history_length: int) -> list[int]:
    prior_events = [*history[1:], int(action)]
    if history_length > 1:
        prior_events = prior_events[-(history_length - 1):]
    else:
        prior_events = []
    return [BOS_ID, *prior_events]


def evaluate_horizon_diagnostics(
    config_path: str | Path,
    sft_checkpoint: str | Path,
    grpo_h1_checkpoint: str | Path,
    grpo_h5_checkpoint: str | Path,
    rm_fit_checkpoint: str | Path,
    rm_audit_checkpoint: str | Path,
    seed: int = 17,
    anchor_count: int = 128,
    horizon: int = 5,
    batch_size: int = 32,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare logged, naive-append, and fixed-history rollouts step by step."""
    import polars as pl

    from tlab.config import load_config

    if anchor_count < 1 or horizon < 1 or batch_size < 1:
        raise ValueError("anchor_count, horizon, and batch_size must be positive")
    started = time.perf_counter()
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
    catalog = pl.read_parquet(processed_dir / "catalog.parquet").sort("token_id")
    catalog_item_ids = catalog.get_column("item_id").to_list()
    item_ids = torch.tensor(catalog_item_ids, dtype=torch.long)
    item_id_by_token = {int(index + 3): int(item_id) for index, item_id in enumerate(catalog_item_ids)}
    training_count_by_token = {
        int(row["token_id"]): int(row["training_count"])
        for row in catalog.select("token_id", "training_count").iter_rows(named=True)
    }
    catalog_duration_by_token = {
        int(row["token_id"]): (
            None if row["catalog_duration_seconds"] is None else float(row["catalog_duration_seconds"]),
            bool(row["catalog_duration_missing"]),
        )
        for row in catalog.select("token_id", "catalog_duration_seconds", "catalog_duration_missing").iter_rows(named=True)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    shape = ActorShape(
        vocab_size=len(catalog_item_ids) + 3,
        n_positions=int(config.model.get("n_positions", 256)),
        n_layer=int(config.model.get("n_layer", 4)),
        n_embd=int(config.model.get("n_embd", 256)),
        n_head=int(config.model.get("n_head", 4)),
        dropout=float(config.model.get("dropout", 0.1)),
    )
    model_paths = {
        "SFT": sft_checkpoint,
        "GRPO_H1": grpo_h1_checkpoint,
        "GRPO_H5": grpo_h5_checkpoint,
    }
    actors: dict[str, ItemGPT2Actor] = {}
    for label, checkpoint_path in model_paths.items():
        actor = ItemGPT2Actor(shape)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(payload["model"])
        actor.to(device).eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        actors[label] = actor
    fit_model, fit_features = _load_reward_model(config, rm_fit_checkpoint, "fit", device)
    audit_model, audit_features = _load_reward_model(config, rm_audit_checkpoint, "audit", device)

    eligible_users = [index for index, user in enumerate(store.users) if len(user.rl_anchors) > 0]
    rng = random.Random(seed)
    rng.shuffle(eligible_users)
    selected_users = eligible_users[:min(anchor_count, len(eligible_users))]
    anchors = [(index, int(rng.choice(store.users[index].rl_anchors))) for index in selected_users]
    if not anchors:
        raise ValueError("fixed A_RL panel is empty")
    history_length = int(config.dataset.get("history_length", 200))
    start_histories = store.histories_for(anchors, history_length=history_length)

    logged_windows: list[list[tuple[int, int]]] = []
    for user_index, start_position in anchors:
        positions = store.users[user_index].rl_anchors
        start_index = next((i for i, position in enumerate(positions) if int(position) == start_position), None)
        if start_index is None:
            logged_windows.append([])
        else:
            logged_windows.append([
                (user_index, int(position))
                for position in positions[start_index:start_index + horizon]
            ])

    feature_frames = {"fit": fit_features, "audit": audit_features}
    record_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, dict[int, dict[str, list[float]]]]] = {
        model: {mode: defaultdict(lambda: defaultdict(list)) for mode in (
            "logged_teacher_forced", "naive_append", "fixed_history_slate",
        )}
        for model in actors
    }

    def record_step(
        model_label: str,
        mode: str,
        step: int,
        anchor_rows: list[tuple[int, int]],
        histories: list[list[int]],
        actions: Tensor,
        action_log_probs: Tensor,
        target_ranks: list[int | None] | None = None,
        prior_slate: list[set[int]] | None = None,
    ) -> None:
        input_ids, attention = _collate_histories(histories, device)
        action_tokens = actions.to(dtype=torch.long)
        fit_scores, _ = _score_actions(
            fit_model, histories, input_ids, attention, action_tokens, fit_features,
        )
        audit_scores, _ = _score_actions(
            audit_model, histories, input_ids, attention, action_tokens, audit_features,
        )
        action_values = action_tokens.detach().cpu().reshape(-1).tolist()
        log_values = action_log_probs.detach().float().cpu().reshape(-1).tolist()
        fit_values = fit_scores.detach().float().cpu().reshape(-1).tolist()
        audit_values = audit_scores.detach().float().cpu().reshape(-1).tolist()
        for row, ((user_index, anchor_position), history, token, log_probability, fit_reward, audit_reward) in enumerate(
            zip(anchor_rows, histories, action_values, log_values, fit_values, audit_values)
        ):
            user = store.users[user_index]
            recent = set(int(value) for value in history[-20:] if int(value) >= 3)
            history_items = set(int(value) for value in history[1:] if int(value) >= 3)
            duration, duration_missing = catalog_duration_by_token.get(int(token), (None, True))
            count = training_count_by_token.get(int(token), 0)
            fit_support = int(feature_frames["fit"].get(int(token), (0.0, True, 0))[2])
            audit_support = int(feature_frames["audit"].get(int(token), (0.0, True, 0))[2])
            slate_repeat = bool(prior_slate[row] and int(token) in prior_slate[row]) if prior_slate is not None else False
            target_rank = target_ranks[row] if target_ranks is not None else None
            record = {
                "model": model_label,
                "mode": mode,
                "step": step + 1,
                "state_provenance": {
                    "logged_teacher_forced": "real_history_before_observed_A_RL_target",
                    "naive_append": "sampled_item_appended_as_if_listened",
                    "fixed_history_slate": "original_real_history_reused_at_every_step",
                }[mode],
                "user_id": int(user.uid),
                "anchor_position": int(anchor_position),
                "observed_position": int(anchor_position) if mode == "logged_teacher_forced" else None,
                "action_token_id": int(token),
                "action_item_id": item_id_by_token.get(int(token)),
                "action_probability": 0.0 if not math.isfinite(float(log_probability)) else float(math.exp(float(log_probability))),
                "action_log_probability": float(log_probability) if math.isfinite(float(log_probability)) else None,
                "target_rank": target_rank,
                "target_ndcg10": single_target_metrics(target_rank)["ndcg@10"] if target_rank is not None or mode == "logged_teacher_forced" else None,
                "rm_fit_predicted_ratio": float(fit_reward),
                "rm_audit_predicted_ratio": float(audit_reward),
                "duration_seconds": duration,
                "duration_missing": bool(duration_missing),
                "training_count": count,
                "log1p_training_count": math.log1p(count),
                "recent_repeat": bool(int(token) in recent),
                "history_repeat": bool(int(token) in history_items),
                "repeat_within_slate": slate_repeat,
                "fit_supported": fit_support > 0,
                "audit_supported": audit_support > 0,
            }
            record_rows.append(record)
            summary = summaries[model_label][mode][step + 1]
            for key in (
                "action_probability", "rm_fit_predicted_ratio", "rm_audit_predicted_ratio",
                "log1p_training_count",
            ):
                summary[key].append(float(record[key]))
            if duration is not None:
                summary["duration_seconds"].append(float(duration))
            summary["recent_repeat_rate"].append(float(record["recent_repeat"]))
            summary["history_repeat_rate"].append(float(record["history_repeat"]))
            summary["repeat_within_slate_rate"].append(float(record["repeat_within_slate"]))
            summary["fit_unsupported_rate"].append(float(not record["fit_supported"]))
            summary["audit_unsupported_rate"].append(float(not record["audit_supported"]))
            if target_rank is not None or mode == "logged_teacher_forced":
                summary["ndcg10"].append(single_target_metrics(target_rank)["ndcg@10"])

    def sample_step(actor: ItemGPT2Actor, histories: list[list[int]], generator: torch.Generator):
        input_ids, attention = _collate_histories(histories, device)
        log_probs = actor.next_action_log_probs(input_ids, attention)
        actions = torch.multinomial(log_probs.exp(), num_samples=1, generator=generator).squeeze(-1)
        action_log_probs = log_probs.gather(-1, actions[:, None]).squeeze(-1)
        return actions, action_log_probs, log_probs

    mode_ids = {"naive_append": 1, "fixed_history_slate": 2}
    for model_index, (model_label, actor) in enumerate(actors.items()):
        # Logged teacher-forcing evaluates each observed A_RL target at its own real history.
        for step in range(horizon):
            rows = [window[step] for window in logged_windows if len(window) > step]
            if not rows:
                continue
            histories = store.histories_for(rows, history_length=history_length)
            target_tokens = [int(store.users[user_index].tokens[position]) for user_index, position in rows]
            input_ids, attention = _collate_histories(histories, device)
            with torch.no_grad():
                log_probs = actor.next_action_log_probs(input_ids, attention)
                action_tensor = torch.tensor(target_tokens, dtype=torch.long, device=device)
                action_log_probs = log_probs.gather(-1, action_tensor[:, None]).squeeze(-1)
                ranks = _rank_logged_targets(log_probs, target_tokens, item_ids.to(device))
                record_step(
                    model_label, "logged_teacher_forced", step, rows, histories,
                    action_tensor, action_log_probs, target_ranks=ranks,
                )

        for mode in ("naive_append", "fixed_history_slate"):
            histories = [list(history) for history in start_histories]
            seen = [set() for _ in anchors]
            generator = torch.Generator(device=device).manual_seed(seed + 10_000 * model_index + mode_ids[mode])
            for step in range(horizon):
                with torch.no_grad():
                    actions, action_log_probs, _ = sample_step(actor, histories, generator)
                    record_step(
                        model_label, mode, step, anchors, histories, actions,
                        action_log_probs, prior_slate=seen,
                    )
                action_values = actions.detach().cpu().tolist()
                for row, token in enumerate(action_values):
                    seen[row].add(int(token))
                    if mode == "naive_append":
                        histories[row] = _append_synthetic(histories[row], int(token), history_length)

    summary_output: dict[str, Any] = {}
    for model_label, mode_values in summaries.items():
        summary_output[model_label] = {}
        for mode, step_values in mode_values.items():
            summary_output[model_label][mode] = {}
            for step, metrics in sorted(step_values.items()):
                summary_output[model_label][mode][str(step)] = {
                    f"mean_{name}": sum(values) / len(values) if values else None
                    for name, values in metrics.items()
                } | {"n_actions": len(metrics.get("action_probability", []))}

    result = {
        "experiment": "H=5 stepwise logged, naive-append, and fixed-history diagnostics",
        "seed": seed,
        "horizon": horizon,
        "anchor_count": len(anchors),
        "logged_windows_with_full_horizon": sum(len(window) == horizon for window in logged_windows),
        "model_checkpoints": {key: str(Path(value).resolve()) for key, value in model_paths.items()},
        "rm_fit_checkpoint": str(Path(rm_fit_checkpoint).resolve()),
        "rm_audit_checkpoint": str(Path(rm_audit_checkpoint).resolve()),
        "config_path": str(config.path.resolve()),
        "source_sha256": source_tree_sha256(root / "src" / "tlab"),
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
        "metric_definitions": {
            "logged_teacher_forced": "NDCG@10 of observed A_RL target under real history strictly before each target",
            "naive_append": "sampled item is appended as synthetic event before next prediction; not an interaction simulator",
            "fixed_history_slate": "each sampled action is scored on the same original history; not a slate-aware policy",
            "repeat": "candidate already present in prior-20 or full current history; within-slate repeats are reported separately",
        },
        "stepwise_summary": summary_output,
        "records": record_rows,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    target = Path(output_path) if output_path else root / "artifacts" / f"horizon_diagnostics_seed{seed}.json"
    if not target.is_absolute():
        target = root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    result["output_path"] = str(target.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stepwise logged, naive-append, and fixed-history H=5 diagnostics")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--grpo-h1-checkpoint", required=True)
    parser.add_argument("--grpo-h5-checkpoint", required=True)
    parser.add_argument("--rm-fit-checkpoint", required=True)
    parser.add_argument("--rm-audit-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_horizon_diagnostics(
        config_path=args.config,
        sft_checkpoint=args.sft_checkpoint,
        grpo_h1_checkpoint=args.grpo_h1_checkpoint,
        grpo_h5_checkpoint=args.grpo_h5_checkpoint,
        rm_fit_checkpoint=args.rm_fit_checkpoint,
        rm_audit_checkpoint=args.rm_audit_checkpoint,
        seed=args.seed,
        anchor_count=args.anchors,
        horizon=args.horizon,
        batch_size=args.batch_size,
        output_path=args.output,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
