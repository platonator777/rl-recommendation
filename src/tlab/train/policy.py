from __future__ import annotations

import argparse
import copy
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from tlab.data.catalog import PAD_ID
from tlab.data.store import SequenceStore
from tlab.evaluation.ranking import evaluate_full_catalog
from tlab.evaluation.rewards import RewardScale, fit_reward_scale
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.models.reward import CandidateRewardModel, RewardModelShape
from tlab.runtime import collect_environment
from tlab.train.checkpoint import load_training_checkpoint, save_training_checkpoint, source_tree_sha256, write_run_manifest
from tlab.train.rollout import (
    collect_horizon_rollout,
    collect_one_step_rollout,
    grpo_update,
    horizon_grpo_update,
    reinforce_update,
)


def _collate_histories(histories: Sequence[Sequence[int]], device: torch.device) -> tuple[Tensor, Tensor]:
    width = max(len(history) for history in histories)
    input_ids = torch.full((len(histories), width), PAD_ID, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, history in enumerate(histories):
        input_ids[row, :len(history)] = torch.as_tensor(history, dtype=torch.long, device=device)
        attention_mask[row, :len(history)] = 1
    return input_ids, attention_mask


def _read_process_rss() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _select_reward_head(
    predictions: dict[str, Tensor],
    reward_head: str,
    seconds_scale: float = 1.0,
) -> Tensor:
    if reward_head == "ratio":
        return predictions["ratio"].float()
    if reward_head == "seconds":
        if seconds_scale <= 0:
            raise ValueError("seconds scale must be positive")
        return predictions["seconds_scaled"].float() * seconds_scale
    raise ValueError(f"unsupported reward head: {reward_head}")


def _apply_repeat_penalty(rewards: Tensor, repeat_flags: Tensor, penalty: float) -> Tensor:
    if penalty < 0 or rewards.shape != repeat_flags.shape:
        raise ValueError("repeat penalty must be non-negative and repeat flags must match rewards")
    return rewards.float() - penalty * repeat_flags.to(device=rewards.device, dtype=torch.float32)


def _score_actions(
    reward_model: CandidateRewardModel,
    histories: Sequence[Sequence[int]],
    input_ids: Tensor,
    attention_mask: Tensor,
    actions: Tensor,
    item_features: dict[int, tuple[float, bool, int]],
    reward_head: str = "ratio",
    seconds_scale: float = 1.0,
    repeat_penalty: float = 0.0,
) -> tuple[Tensor, int]:
    """Score sampled actions after one history encoding per context."""
    if actions.ndim == 1:
        actions = actions[:, None]
    if actions.shape[0] != len(histories):
        raise ValueError("action rows must match context histories")
    candidates = actions.to(dtype=torch.long)
    durations: list[list[float]] = []
    duration_missing: list[list[bool]] = []
    repeats: list[list[bool]] = []
    unsupported = 0
    action_rows = candidates.detach().cpu().tolist()
    for history, candidate_row in zip(histories, action_rows):
        recent = set(int(token) for token in history[-20:] if int(token) >= 3)
        row_duration: list[float] = []
        row_missing: list[bool] = []
        row_repeat: list[bool] = []
        for token in candidate_row:
            duration, missing, support = item_features.get(int(token), (0.0, True, 0))
            row_duration.append(float(duration))
            row_missing.append(bool(missing))
            row_repeat.append(int(token) in recent)
            unsupported += int(support == 0)
        durations.append(row_duration)
        duration_missing.append(row_missing)
        repeats.append(row_repeat)
    duration_tensor = torch.tensor(durations, dtype=torch.float32, device=input_ids.device)
    missing_tensor = torch.tensor(duration_missing, dtype=torch.bool, device=input_ids.device)
    repeat_tensor = torch.tensor(repeats, dtype=torch.bool, device=input_ids.device)
    history_state = reward_model.encode_history(input_ids, attention_mask)
    prediction = reward_model.score_encoded(
        history_state,
        candidates,
        duration_tensor,
        missing_tensor,
        repeat_tensor,
    )
    reward = _select_reward_head(prediction, reward_head, seconds_scale)
    if repeat_penalty:
        repeat_tensor = torch.tensor(repeats, dtype=torch.bool, device=reward.device)
        reward = _apply_repeat_penalty(reward, repeat_tensor, repeat_penalty)
    return reward, unsupported


def _normalize_reward(values: Tensor, scale: RewardScale) -> Tensor:
    return (values.float() - scale.mean) / max(scale.std, 1e-3)


def _sample_reward_scale(
    actor: ItemGPT2Actor,
    reward_model: CandidateRewardModel,
    store: SequenceStore,
    item_features: dict[int, tuple[float, bool, int]],
    count: int,
    seed: int,
    history_length: int,
    reward_head: str = "ratio",
    seconds_scale: float = 1.0,
    repeat_penalty: float = 0.0,
) -> tuple[RewardScale, dict[str, int | float]]:
    device = next(actor.parameters()).device
    anchors = store.sample_rl_anchors(count=count, seed=seed)
    if not anchors:
        raise ValueError("reward normalization requires A_RL anchors")
    histories = store.histories_for(anchors, history_length=history_length)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    reward_model.eval()
    values: list[float] = []
    unsupported = 0
    with torch.no_grad():
        for start in range(0, len(histories), 32):
            history_batch = histories[start:start + 32]
            input_ids, attention_mask = _collate_histories(history_batch, device)
            actions = torch.randint(
                3, actor.vocab_size, (len(history_batch),),
                device=device, generator=generator,
            )
            raw_rewards, missing_support = _score_actions(
                reward_model, history_batch, input_ids, attention_mask, actions, item_features,
                reward_head=reward_head, seconds_scale=seconds_scale,
                repeat_penalty=repeat_penalty,
            )
            values.extend(raw_rewards.squeeze(-1).cpu().tolist())
            unsupported += missing_support
    return fit_reward_scale(values), {
        "contexts": len(anchors),
        "actions": len(values),
        "unsupported_actions": unsupported,
        "raw_mean": sum(values) / len(values),
        "sampling_seed": seed,
        "action_sampling": "uniform catalog actions, independent of actor checkpoint",
    }


def _metric_summary(metrics: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in metrics.items() if key != "per_user_ranks"}


def _update_reference_run_registry(root: Path, manifest: dict[str, object]) -> None:
    registry_path = root / "artifacts" / "reference" / "run_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            registry = {"protocol_id": "tlab_yambda_reference_2026-10-13", "runs": []}
    else:
        registry = {"protocol_id": "tlab_yambda_reference_2026-10-13", "runs": []}
    run_id = str(manifest["run_id"])
    entry = {
        "run_id": run_id, "stage": manifest.get("stage"),
        "phase": f"{manifest.get('algorithm')}_H{manifest.get('horizon')}",
        "seed": manifest.get("seed"), "status": manifest.get("status", "running"),
        "manifest_path": str((Path(str(manifest["run_dir"])) / "run_manifest.json").resolve()),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_target_presentations": manifest.get("rm_scored_actions", 0),
        "best_validation_ndcg10": None,
        "stop_reason": manifest.get("stop_reason"),
    }
    runs = [row for row in registry.get("runs", []) if row.get("run_id") != run_id]
    runs.append(entry)
    registry["runs"] = sorted(runs, key=lambda row: (row.get("stage", ""), row.get("run_id", "")))
    temporary = registry_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(registry_path)


def run_one_step_grpo(
    config_path: str | Path,
    actor_checkpoint: str | Path,
    reward_checkpoint: str | Path,
    seed: int = 17,
    beta: float = 0.1,
    updates: int = 20,
    batch_contexts: int = 4,
    group_size: int = 8,
    reward_scale_anchors: int | None = None,
    validation_anchors_per_user: int = 1,
    checkpoint_interval: int | None = None,
    run_dir: str | Path | None = None,
    algorithm: str = "grpo",
    horizon: int = 1,
    reward_head: str = "ratio",
    advantage_mode: str = "standardized",
    repeat_penalty: float = 0.0,
    update_epochs: int = 1,
    minibatch_contexts: int | None = None,
    old_policy_kl_stop: float | None = 0.02,
    resume: str | Path | None = None,
    validation_seed: int | None = None,
    anchor_seed: int | None = None,
    validation_interval: int = 500,
    stop_after_updates: int | None = None,
) -> dict[str, object]:
    import polars as pl

    from tlab.config import load_config

    if (
        updates < 1 or batch_contexts < 1 or group_size < 2 or beta < 0
        or algorithm not in {"grpo", "reinforce"} or horizon not in {1, 5}
        or reward_head not in {"ratio", "seconds"}
        or advantage_mode not in {"standardized", "centered_corrected"}
        or repeat_penalty < 0
        or (repeat_penalty > 0 and (reward_head != "ratio" or horizon != 1))
        or (horizon == 5 and algorithm != "grpo")
        or (algorithm != "grpo" and advantage_mode != "standardized")
        or update_epochs < 1 or validation_interval < 1
        or (checkpoint_interval is not None and checkpoint_interval < 1)
        or (stop_after_updates is not None and stop_after_updates < 1)
        or (algorithm == "reinforce" and update_epochs != 1)
        or (horizon != 1 and update_epochs != 1)
    ):
        raise ValueError("invalid policy algorithm, updates, batch size, group size or beta")
    run_started = time.perf_counter()
    config = load_config(config_path)
    checkpoint_interval = int(
        checkpoint_interval if checkpoint_interval is not None
        else config.training.get("rl_checkpoint_interval", 100)
    )
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    protocol_path = processed_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seconds_scale = float(protocol["seconds_scale_f_rm_p95"]) if reward_head == "seconds" else 1.0
    reward_label = (f"Q4_repeat_penalty_{repeat_penalty:g}" if repeat_penalty else
                    "Q1" if reward_head == "ratio" else "Q2")
    store = SequenceStore.load_cache(
        processed_dir / "sequence_index",
        expected_source_sha256=protocol["processed_listens_sha256"],
    )
    catalog_frame = pl.read_parquet(processed_dir / "catalog.parquet").sort("token_id")
    catalog_item_ids = catalog_frame.get_column("item_id").to_list()
    feature_frame = pl.read_parquet(processed_dir / "rm_item_features_fit.parquet")
    item_features = {
        int(row["token_id"]): (
            float(row["rm_duration_seconds"] or 0.0),
            bool(row["rm_duration_missing"]),
            int(row["rm_training_support"]),
        )
        for row in feature_frame.iter_rows(named=True)
    }

    run_seed = int(seed)
    validation_seed = int(validation_seed if validation_seed is not None else config.training.get("validation_seed", 17))
    anchor_seed = int(anchor_seed if anchor_seed is not None else config.training.get("rl_anchor_seed", 17))
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history_length = int(config.dataset.get("history_length", 200))
    actor_shape = ActorShape(
        vocab_size=len(catalog_item_ids) + 3,
        n_positions=int(config.model.get("n_positions", 256)),
        n_layer=int(config.model.get("n_layer", 4)),
        n_embd=int(config.model.get("n_embd", 256)),
        n_head=int(config.model.get("n_head", 4)),
        dropout=float(config.model.get("dropout", 0.1)),
    )
    actor = ItemGPT2Actor(actor_shape)
    actor_payload = torch.load(actor_checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(actor_payload["model"])
    actor = actor.to(device)
    reference = copy.deepcopy(actor).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    rm_shape = RewardModelShape(
        vocab_size=len(catalog_item_ids) + 3,
        hidden_size=int(config.training.get("rm_hidden_size", 128)),
        n_layer=int(config.training.get("rm_layers", 2)),
        n_head=int(config.training.get("rm_heads", 4)),
        max_positions=int(config.model.get("n_positions", 256)),
        encoder="transformer",
    )
    reward_model = CandidateRewardModel(rm_shape)
    rm_payload = torch.load(reward_checkpoint, map_location="cpu", weights_only=False)
    reward_model.load_state_dict(rm_payload["model"])
    reward_model = reward_model.to(device).eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(config.training.get("rl_learning_rate", 1e-5)),
        weight_decay=0.0,
    )
    resume_payload = torch.load(resume, map_location="cpu", weights_only=False) if resume is not None else None
    resume_meta = (resume_payload or {}).get("metadata", {})
    if resume is not None:
        expected = {
            "algorithm": algorithm, "seed": run_seed, "reward_head": reward_head,
            "horizon": horizon, "beta": beta, "group_size": group_size,
            "batch_contexts": batch_contexts, "update_epochs": update_epochs,
            "validation_seed": validation_seed, "anchor_seed": anchor_seed,
            "validation_interval": validation_interval,
            "minibatch_contexts": minibatch_contexts or batch_contexts,
            "old_policy_kl_stop": old_policy_kl_stop,
            "repeat_penalty": repeat_penalty,
            "advantage_mode": advantage_mode if algorithm == "grpo" else "leave_one_out",
            "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
            "reward_checkpoint": str(Path(reward_checkpoint).resolve()),
        }
        for key, value in expected.items():
            if key in resume_meta and resume_meta[key] != value:
                raise ValueError(f"RL resume {key} differs from the checkpoint")
    if resume_meta.get("reward_scale"):
        scale = RewardScale(**resume_meta["reward_scale"])
        scale_panel = dict(resume_meta["reward_scale_panel"])
    else:
        scale_count = int(
            reward_scale_anchors if reward_scale_anchors is not None
            else config.training.get("rl_reward_scale_contexts", 512)
        )
        scale, scale_panel = _sample_reward_scale(
            actor, reward_model, store, item_features,
            count=scale_count,
            seed=int(config.training.get("rl_reward_scale_seed", 90_017)),
            history_length=history_length,
            reward_head=reward_head, seconds_scale=seconds_scale,
            repeat_penalty=repeat_penalty,
        )

    artifacts_dir = Path(config.project.get("artifacts_dir", "artifacts"))
    if not artifacts_dir.is_absolute():
        artifacts_dir = root / artifacts_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_path = Path(run_dir) if run_dir else (
        Path(resume_meta["run_dir"]) if resume_meta.get("run_dir")
        else artifacts_dir / "runs" / f"{algorithm}_q1_h{horizon}_seed{run_seed}_{stamp}"
    )
    if not run_path.is_absolute():
        run_path = root / run_path
    run_path.mkdir(parents=True, exist_ok=True)
    if resume is not None and resume_meta.get("run_id") not in (None, run_path.name):
        raise ValueError("RL resume must reuse the checkpoint's original --run-dir")
    if resume is not None:
        load_training_checkpoint(resume, actor, optimizer, map_location=device)
    metrics_path = run_path / "metrics.jsonl"
    manifest: dict[str, object] = {
        "run_id": run_path.name,
        "stage": algorithm.upper(),
        "algorithm": algorithm.upper(),
        "seed": run_seed,
        "reward": reward_label,
        "reward_head": reward_head,
        "repeat_penalty": repeat_penalty,
        "advantage_mode": advantage_mode if algorithm == "grpo" else "leave_one_out",
        "reward_units": ("completion_ratio_minus_repeat_penalty" if repeat_penalty else
                         "completion_ratio" if reward_head == "ratio" else "predicted_listening_seconds"),
        "seconds_scale": seconds_scale if reward_head == "seconds" else None,
        "horizon": horizon,
        "beta": beta,
        "group_size": group_size,
        "batch_contexts": batch_contexts,
        "update_epochs": update_epochs,
        "minibatch_contexts": minibatch_contexts or batch_contexts,
        "old_policy_kl_stop": old_policy_kl_stop,
        "updates_planned": updates,
        "validation_seed": validation_seed,
        "anchor_seed": anchor_seed,
        "validation_interval": validation_interval,
        "run_dir": str(run_path.resolve()),
        "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
        "reward_checkpoint": str(Path(reward_checkpoint).resolve()),
        "config_path": str(config.path),
        "config": config.raw,
        "source_sha256": source_tree_sha256(root / "src" / "tlab"),
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
        "catalog_size": len(catalog_item_ids),
        "actor_parameters": actor.parameter_count(),
        "reward_scale": {"mean": scale.mean, "std": scale.std, "sample_count": scale.sample_count},
        "reward_scale_panel": scale_panel,
        "precision": "float32",
        "environment": collect_environment(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if resume is not None and (run_path / "run_manifest.json").exists():
        existing_manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
        manifest = {**existing_manifest, **manifest, "resumed_at_utc": datetime.now(timezone.utc).isoformat()}
    data_manifest_path = root / "data" / "data_manifest.json"
    if data_manifest_path.exists():
        manifest["dataset_manifest_sha256"] = __import__("tlab.data.manifest", fromlist=["sha256_file"]).sha256_file(data_manifest_path)
    start_update = int((resume_payload or {}).get("sampler_state", {}).get(
        "next_update", resume_meta.get("update", -1) + 1,
    ))
    if start_update >= updates:
        raise ValueError("RL resume checkpoint has already reached the requested update budget")
    manifest.update({"status": "running", "start_update": start_update})
    write_run_manifest(run_path / "run_manifest.json", manifest)
    _update_reference_run_registry(root, manifest)

    baseline_path = run_path / "baseline_validation.json"
    if baseline_path.exists():
        validation_before = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        validation_before = _metric_summary(evaluate_full_catalog(
            reference, store, catalog_item_ids, split="validation", history_length=history_length,
            per_user_limit=validation_anchors_per_user, seed=validation_seed, batch_size=16,
        ))
        baseline_path.write_text(json.dumps(validation_before, indent=2, ensure_ascii=False), encoding="utf-8")
    training_started = time.perf_counter()
    rm_calls = int(resume_meta.get("rm_scored_actions", scale_panel["actions"]))
    prior_training_seconds = float(resume_meta.get("training_seconds", 0.0))
    validation_history_path = run_path / "validation_history.jsonl"
    with metrics_path.open("a", encoding="utf-8") as handle:
        for update in range(start_update, updates):
            update_started = time.perf_counter()
            reward_scoring_seconds = 0.0
            anchors = store.sample_rl_anchors(batch_contexts, seed=anchor_seed + update)
            histories = store.histories_for(anchors, history_length=history_length)
            input_ids, attention_mask = _collate_histories(histories, device)

            if horizon == 1:
                def reward_function(actions: Tensor) -> Tensor:
                    nonlocal reward_scoring_seconds
                    scoring_started = time.perf_counter()
                    raw, _ = _score_actions(
                        reward_model, histories, input_ids, attention_mask, actions, item_features,
                        reward_head=reward_head, seconds_scale=seconds_scale,
                        repeat_penalty=repeat_penalty,
                    )
                    reward_scoring_seconds += time.perf_counter() - scoring_started
                    return _normalize_reward(raw, scale)

                rollout = collect_one_step_rollout(
                    actor, reference, input_ids, attention_mask, reward_function,
                    group_size=group_size,
                    normalize_advantages=(algorithm == "grpo" and advantage_mode == "standardized"),
                    centered_correction=(algorithm == "grpo" and advantage_mode == "centered_corrected"),
                )
                if algorithm == "grpo":
                    update_metrics = grpo_update(
                        actor, input_ids, attention_mask, rollout, optimizer,
                        beta=beta,
                        clip_epsilon=float(config.training.get("rl_clip_epsilon", 0.2)),
                        gradient_clip=1.0,
                        update_epochs=update_epochs,
                        minibatch_contexts=minibatch_contexts,
                        max_kl_to_old=old_policy_kl_stop,
                    )
                else:
                    update_metrics = reinforce_update(
                        actor, input_ids, attention_mask, rollout, optimizer,
                        beta=beta, gradient_clip=1.0,
                    )
            else:
                def horizon_reward_function(
                    state_histories: list[list[int]], actions: Tensor, step: int,
                ) -> Tensor:
                    nonlocal reward_scoring_seconds
                    scoring_started = time.perf_counter()
                    state_ids, state_mask = _collate_histories(state_histories, device)
                    raw, _ = _score_actions(
                        reward_model, state_histories, state_ids, state_mask, actions, item_features,
                        reward_head=reward_head, seconds_scale=seconds_scale,
                        repeat_penalty=repeat_penalty,
                    )
                    reward_scoring_seconds += time.perf_counter() - scoring_started
                    return _normalize_reward(raw.squeeze(-1), scale)

                rollout = collect_horizon_rollout(
                    actor, reference, histories, horizon_reward_function,
                    group_size=group_size, horizon=horizon, history_length=history_length,
                    normalize_advantages=advantage_mode == "standardized",
                    centered_correction=advantage_mode == "centered_corrected",
                    seed=run_seed + update,
                )
                update_metrics = horizon_grpo_update(
                    actor, rollout, optimizer, beta=beta,
                    clip_epsilon=float(config.training.get("rl_clip_epsilon", 0.2)),
                    gradient_clip=1.0,
                )
            action_values = rollout.actions.detach().cpu()
            sampled = [int(token) for token in action_values.reshape(-1).tolist()]
            unsupported = sum(int(item_features.get(token, (0.0, True, 0))[2] == 0) for token in sampled)
            if horizon == 1:
                sampled_rewards = rollout.rewards
                group_returns = rollout.rewards
                unique_per_group = [len(set(int(x) for x in row)) for row in action_values.tolist()]
            else:
                sampled_rewards = rollout.rewards
                group_returns = rollout.returns
                unique_per_group = [
                    len(set(int(token) for token in action_values[:, :, step][context].tolist()))
                    for context in range(batch_contexts) for step in range(horizon)
                ]
            metric_row = {
                "stage": algorithm.upper(),
                "algorithm": algorithm.upper(),
                "seed": run_seed,
                "update": update,
                "reward": reward_label,
                "reward_head": reward_head,
                "horizon": horizon,
                "beta": beta,
                "sampled_raw_reward_mean": float(sampled_rewards.float().mean() * scale.std + scale.mean),
                "sampled_normalized_reward_mean": float(sampled_rewards.float().mean()),
                "mean_normalized_trajectory_return": float(group_returns.float().mean()),
                "mean_raw_trajectory_return": float(group_returns.float().mean() * scale.std + horizon * scale.mean),
                "reward_per_action": float(sampled_rewards.float().mean()),
                "group_reward_std_mean": float(group_returns.std(dim=-1, correction=0).mean()),
                "zero_variance_group_fraction": float((group_returns.std(dim=-1, correction=0) < 1e-4).float().mean()),
                "unique_actions_per_group_mean": sum(unique_per_group) / len(unique_per_group),
                "unsupported_sampled_action_fraction": unsupported / max(1, len(sampled)),
                "rm_scored_actions": len(sampled),
                "rm_scoring_seconds": reward_scoring_seconds,
                "update_seconds": time.perf_counter() - update_started,
                "process_rss_bytes": _read_process_rss(),
                "anchor_user_ids": [int(store.users[user_index].uid) for user_index, _ in anchors],
                "anchor_positions": [int(position) for _, position in anchors],
                **update_metrics,
            }
            if horizon == 5:
                metric_row["state_provenance"] = "naive_append_synthetic_item_history"
                metric_row["sampled_action_ids_by_context_group_step"] = action_values.tolist()
                metric_row["sampled_raw_rewards_by_context_group_step"] = (
                    rollout.rewards.float() * scale.std + scale.mean
                ).detach().cpu().tolist()
                metric_row["unsupported_action_fraction_by_step"] = [
                    sum(
                        int(item_features.get(int(token), (0.0, True, 0))[2] == 0)
                        for token in action_values[:, :, step].reshape(-1).tolist()
                    ) / action_values.shape[0] / action_values.shape[1]
                    for step in range(horizon)
                ]
            handle.write(json.dumps(metric_row, ensure_ascii=False) + "\n")
            handle.flush()
            rm_calls += len(sampled)
            if (update + 1) % max(1, checkpoint_interval) == 0 or update + 1 == updates:
                save_training_checkpoint(
                    run_path / "last.pt", actor, optimizer, update + 1,
                    {"run_id": run_path.name, "run_dir": str(run_path.resolve()),
                     "algorithm": algorithm, "seed": run_seed, "reward_head": reward_head,
                     "horizon": horizon, "beta": beta, "group_size": group_size,
                     "batch_contexts": batch_contexts, "update_epochs": update_epochs,
                     "minibatch_contexts": minibatch_contexts or batch_contexts,
                     "old_policy_kl_stop": old_policy_kl_stop,
                     "repeat_penalty": repeat_penalty,
                     "advantage_mode": advantage_mode if algorithm == "grpo" else "leave_one_out",
                     "validation_seed": validation_seed, "anchor_seed": anchor_seed,
                     "validation_interval": validation_interval,
                     "update": update + 1, "rm_scored_actions": rm_calls,
                     "training_seconds": prior_training_seconds + time.perf_counter() - training_started,
                     "reward_scale": manifest["reward_scale"],
                     "reward_scale_panel": scale_panel,
                     "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
                     "reward_checkpoint": str(Path(reward_checkpoint).resolve())},
                    sampler_state={"next_update": update + 1, "anchor_seed": anchor_seed,
                                   "cumulative_training_action_scores": max(0, rm_calls - int(scale_panel["actions"]))},
                )
                manifest.update({
                    "status": "running", "latest_update": update + 1,
                    "rm_scored_actions": rm_calls,
                    "training_seconds": prior_training_seconds + time.perf_counter() - training_started,
                    "last_checkpoint": str((run_path / "last.pt").resolve()),
                    "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                    "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                })
                write_run_manifest(run_path / "run_manifest.json", manifest)
                _update_reference_run_registry(root, manifest)
            if (update + 1) % validation_interval == 0 or update + 1 == updates:
                validation_metrics = _metric_summary(evaluate_full_catalog(
                    actor, store, catalog_item_ids, split="validation", history_length=history_length,
                    per_user_limit=validation_anchors_per_user, seed=validation_seed, batch_size=16,
                ))
                validation_row = {"update": update + 1, "seed": run_seed, **validation_metrics}
                with validation_history_path.open("a", encoding="utf-8") as validation_handle:
                    validation_handle.write(json.dumps(validation_row, ensure_ascii=False) + "\n")
                manifest["latest_validation"] = validation_row
                manifest["validation_history_path"] = str(validation_history_path.resolve())
                write_run_manifest(run_path / "run_manifest.json", manifest)
                _update_reference_run_registry(root, manifest)
            if stop_after_updates is not None and update + 1 >= stop_after_updates:
                manifest.update({
                    "status": "interrupted",
                    "stop_reason": "intentional checkpoint-boundary interruption for resume-equivalence verification",
                    "interrupted_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                write_run_manifest(run_path / "run_manifest.json", manifest)
                _update_reference_run_registry(root, manifest)
                raise KeyboardInterrupt("intentional checkpoint-boundary interruption")

    validation_after = manifest.get("latest_validation")
    if not isinstance(validation_after, dict) or int(validation_after.get("update", -1)) != updates:
        validation_after = _metric_summary(evaluate_full_catalog(
            actor, store, catalog_item_ids, split="validation", history_length=history_length,
            per_user_limit=validation_anchors_per_user, seed=validation_seed, batch_size=16,
        ))
    (run_path / "validation_metrics.json").write_text(json.dumps({
        "baseline": validation_before,
        "policy": validation_after,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    result: dict[str, object] = {
        "run_dir": str(run_path.resolve()),
        "checkpoint": str((run_path / "last.pt").resolve()),
        "validation_before": _metric_summary(validation_before),
        "validation_after": validation_after,
        "updates": updates,
        "scored_actions": rm_calls,
        "training_seconds": prior_training_seconds + time.perf_counter() - training_started,
        "training_seconds_this_invocation": time.perf_counter() - training_started,
        "elapsed_seconds": time.perf_counter() - run_started,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
    }
    manifest.update(result)
    manifest["status"] = "completed"
    manifest["stop_reason"] = "completed configured rollout batch budget"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_run_manifest(run_path / "run_manifest.json", manifest)
    _update_reference_run_registry(root, manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Short GRPO/REINFORCE diagnostics with selectable Q1 ratio or Q2 seconds reward and exact full-catalog KL")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--algorithm", choices=("grpo", "reinforce"), default="grpo")
    parser.add_argument("--horizon", type=int, choices=(1, 5), default=1)
    parser.add_argument("--reward-head", choices=("ratio", "seconds"), default="ratio")
    parser.add_argument("--repeat-penalty", type=float, default=0.0)
    parser.add_argument("--advantage-mode", choices=("standardized", "centered_corrected"), default="standardized")
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--batch-contexts", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--minibatch-contexts", type=int)
    parser.add_argument("--old-policy-kl-stop", type=float, default=0.02)
    parser.add_argument("--reward-scale-anchors", type=int)
    parser.add_argument("--validation-anchors-per-user", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume")
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--anchor-seed", type=int)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--stop-after-updates", type=int, help="checkpointed interruption for resume verification only")
    args = parser.parse_args()
    print(json.dumps(run_one_step_grpo(
        args.config,
        args.actor_checkpoint,
        args.reward_checkpoint,
        seed=args.seed,
        beta=args.beta,
        updates=args.updates,
        batch_contexts=args.batch_contexts,
        group_size=args.group_size,
        update_epochs=args.update_epochs,
        minibatch_contexts=args.minibatch_contexts,
        old_policy_kl_stop=args.old_policy_kl_stop,
        reward_scale_anchors=args.reward_scale_anchors,
        validation_anchors_per_user=args.validation_anchors_per_user,
        checkpoint_interval=args.checkpoint_interval,
        run_dir=args.run_dir,
        algorithm=args.algorithm,
        horizon=args.horizon,
        reward_head=args.reward_head,
        advantage_mode=args.advantage_mode,
        repeat_penalty=args.repeat_penalty,
        resume=args.resume,
        validation_seed=args.validation_seed,
        anchor_seed=args.anchor_seed,
        validation_interval=args.validation_interval,
        stop_after_updates=args.stop_after_updates,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
