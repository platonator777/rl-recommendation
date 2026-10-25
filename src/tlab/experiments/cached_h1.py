from __future__ import annotations

import argparse
import copy
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from tlab.data.catalog import PAD_ID
from tlab.data.manifest import sha256_file
from tlab.data.store import SequenceStore
from tlab.evaluation.rewards import RewardScale
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.models.reward import CandidateRewardModel, RewardModelShape
from tlab.runtime import collect_environment
from tlab.train.checkpoint import load_training_checkpoint, save_training_checkpoint, source_tree_sha256, write_run_manifest
from tlab.train.objectives import analytic_reward_kl_policy, full_categorical_kl
from tlab.train.policy import (
    _collate_histories,
    _sample_reward_scale,
    _score_actions,
    _normalize_reward,
)
from tlab.train.rollout import collect_one_step_rollout, grpo_update, reinforce_update


def _user_balanced_rl_panel(store: SequenceStore, count: int, seed: int) -> list[tuple[int, int]]:
    eligible = [index for index, user in enumerate(store.users) if len(user.rl_anchors) > 0]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = eligible[:min(count, len(eligible))]
    return [(index, int(rng.choice(store.users[index].rl_anchors))) for index in selected]


def _user_balanced_validation_panel(store: SequenceStore, count: int, seed: int) -> list[tuple[int, int]]:
    targets = store.evaluation_targets("validation", per_user_limit=1, seed=seed)
    by_user: dict[int, tuple[int, int]] = {}
    for user_index, position in targets:
        by_user.setdefault(user_index, (user_index, position))
    rows = list(by_user.values())
    random.Random(seed + 100_009).shuffle(rows)
    return rows[:min(count, len(rows))]


def _cache_reward_matrix(
    model: CandidateRewardModel,
    store: SequenceStore,
    panel: Sequence[tuple[int, int]],
    item_features: dict[int, tuple[float, bool, int]],
    path: Path,
    *,
    history_length: int,
    chunk_size: int,
    batch_size: int,
    reward_head: str,
    seconds_scale: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    histories = store.histories_for(panel, history_length=history_length)
    vocab_catalog_ids = torch.arange(3, model.shape.vocab_size, dtype=torch.long, device=device)
    catalog_size = int(vocab_catalog_ids.numel())
    durations = torch.tensor(
        [float(item_features.get(token, (0.0, True, 0))[0]) for token in range(3, model.shape.vocab_size)],
        dtype=torch.float32, device=device,
    )
    duration_missing = torch.tensor(
        [bool(item_features.get(token, (0.0, True, 0))[1]) for token in range(3, model.shape.vocab_size)],
        dtype=torch.bool, device=device,
    )
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(panel), catalog_size))
    recent_sets = [set(int(token) for token in history[-20:]) for history in histories]
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(panel), batch_size):
            stop = min(start + batch_size, len(panel))
            histories_batch = histories[start:stop]
            input_ids, attention_mask = _collate_histories(histories_batch, device)
            candidates = vocab_catalog_ids.unsqueeze(0).expand(len(histories_batch), -1)
            durations_batch = durations.unsqueeze(0).expand(len(histories_batch), -1)
            missing_batch = duration_missing.unsqueeze(0).expand(len(histories_batch), -1)
            repeat_batch = torch.tensor(
                [[token in recent_sets[start + local_index] for token in range(3, model.shape.vocab_size)]
                 for local_index in range(len(histories_batch))],
                dtype=torch.bool, device=device,
            )
            predictions = model.score_catalog(
                input_ids, attention_mask, candidates, durations_batch,
                missing_batch, repeat_batch, chunk_size=chunk_size,
            )
            raw = predictions["ratio"] if reward_head == "ratio" else predictions["seconds_scaled"] * seconds_scale
            output[start:stop] = raw.float().cpu().numpy()
    output.flush()
    elapsed = time.perf_counter() - started
    del output
    return {
        "path": str(path.resolve()),
        "contexts": len(panel),
        "catalog_size": catalog_size,
        "rm_scored_actions": len(panel) * catalog_size,
        "history_encoding_batch_size": batch_size,
        "candidate_chunk_size": chunk_size,
        "seconds": elapsed,
        "actions_per_second": (len(panel) * catalog_size) / max(elapsed, 1e-9),
    }


def _panel_tensors(
    store: SequenceStore,
    panel: Sequence[tuple[int, int]],
    history_length: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    histories = store.histories_for(panel, history_length=history_length)
    return _collate_histories(histories, device)


@torch.no_grad()
def _actor_log_distributions(
    actor: ItemGPT2Actor,
    input_ids: Tensor,
    attention_mask: Tensor,
    batch_size: int,
) -> Tensor:
    actor.eval()
    return torch.cat([
        actor.next_action_log_probs(input_ids[start:start + batch_size], attention_mask[start:start + batch_size])
        for start in range(0, input_ids.shape[0], batch_size)
    ], dim=0)


def _exact_objective(log_policy: Tensor, rewards: Tensor, log_reference: Tensor, beta: float) -> Tensor:
    kl = full_categorical_kl(log_policy, log_reference)
    return (log_policy.float().exp() * rewards.float()).sum(dim=-1) - beta * kl


@torch.no_grad()
def _measure_actor(
    actor: ItemGPT2Actor,
    input_ids: Tensor,
    attention_mask: Tensor,
    rewards: Tensor,
    log_reference: Tensor,
    beta: float,
    batch_size: int,
) -> dict[str, float]:
    values: list[Tensor] = []
    rewards_mean: list[Tensor] = []
    kl_values: list[Tensor] = []
    actor.eval()
    for start in range(0, input_ids.shape[0], batch_size):
        stop = min(start + batch_size, input_ids.shape[0])
        log_policy = actor.next_action_log_probs(input_ids[start:stop], attention_mask[start:stop])
        values.append(_exact_objective(log_policy, rewards[start:stop], log_reference[start:stop], beta))
        distribution = log_policy.float().exp()
        rewards_mean.append((distribution * rewards[start:stop]).sum(dim=-1))
        kl_values.append(full_categorical_kl(log_policy, log_reference[start:stop]))
    return {
        "J": float(torch.cat(values).mean()),
        "expected_normalized_reward": float(torch.cat(rewards_mean).mean()),
        "kl_to_reference": float(torch.cat(kl_values).mean()),
    }


def run_cached_h1(
    config_path: str | Path,
    actor_checkpoint: str | Path,
    reward_checkpoint: str | Path,
    run_dir: str | Path,
    *,
    seed: int = 17,
    beta: float = 0.1,
    contexts: int = 512,
    batch_contexts: int = 16,
    group_size: int = 8,
    updates: int = 1000,
    learning_rate: float = 1e-5,
    reward_head: str = "ratio",
    reward_scale_seed: int = 90_017,
    checkpoint_interval_updates: int = 100,
    stop_after_updates: int | None = None,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    import polars as pl

    from tlab.config import load_config

    if (
        contexts < 2 or batch_contexts < 1 or group_size < 2 or updates < 1 or beta <= 0
        or checkpoint_interval_updates < 1
        or (stop_after_updates is not None and stop_after_updates < 1)
    ):
        raise ValueError("cached H=1 optimization needs positive contexts/updates/beta and group_size >= 2")
    if reward_head not in {"ratio", "seconds"}:
        raise ValueError("reward_head must be ratio or seconds")
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
    item_features_frame = pl.read_parquet(processed_dir / "rm_item_features_fit.parquet")
    item_features = {
        int(row["token_id"]): (
            float(row["rm_duration_seconds"] or 0.0),
            bool(row["rm_duration_missing"]),
            int(row["rm_training_support"]),
        ) for row in item_features_frame.iter_rows(named=True)
    }
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(catalog) + 3
    actor = ItemGPT2Actor(ActorShape(
        vocab_size=vocab_size,
        n_positions=int(config.model.get("n_positions", 256)),
        n_layer=int(config.model.get("n_layer", 4)),
        n_embd=int(config.model.get("n_embd", 256)),
        n_head=int(config.model.get("n_head", 4)),
        dropout=float(config.model.get("dropout", 0.1)),
    ))
    actor_payload = torch.load(actor_checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(actor_payload["model"])
    actor = actor.to(device).eval()
    reference = copy.deepcopy(actor).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reward_model = CandidateRewardModel(RewardModelShape(
        vocab_size=vocab_size,
        hidden_size=int(config.training.get("rm_hidden_size", 128)),
        n_layer=int(config.training.get("rm_layers", 2)),
        n_head=int(config.training.get("rm_heads", 4)),
        max_positions=int(config.model.get("n_positions", 256)),
        encoder="transformer",
    ))
    reward_payload = torch.load(reward_checkpoint, map_location="cpu", weights_only=False)
    reward_model.load_state_dict(reward_payload["model"])
    reward_model = reward_model.to(device).eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)
    history_length = int(config.dataset.get("history_length", 201))
    seconds_scale = float(protocol["seconds_scale_f_rm_p95"]) if reward_head == "seconds" else 1.0

    train_panel = _user_balanced_rl_panel(store, contexts, seed + 17)
    validation_panel = _user_balanced_validation_panel(store, contexts, seed + 31)
    if len(train_panel) < 2 or len(validation_panel) < 2:
        raise ValueError("not enough distinct users for cached H=1 panels")
    output_dir = Path(run_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = Path(resume).resolve() if resume is not None else None
    resume_method = resume_path.parent.name if resume_path is not None else None
    existing_manifest_path = output_dir / "run_manifest.json"
    resume_manifest: dict[str, Any] | None = None
    resume_preview: dict[str, Any] | None = None
    if resume_path is not None:
        if resume_path.name != "last.pt" or resume_path.parent.parent.resolve() != output_dir.resolve():
            raise ValueError("cached H=1 resume must use a method last.pt inside the original --run-dir")
        if not existing_manifest_path.is_file():
            raise ValueError("cached H=1 resume requires the original root run manifest")
        resume_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        resume_preview = torch.load(resume_path, map_location="cpu", weights_only=False)
        expected = {
            "seed": seed, "beta": beta, "reward_head": reward_head,
            "train_panel_size": min(contexts, sum(len(user.rl_anchors) > 0 for user in store.users)),
            "contexts": contexts, "batch_contexts": batch_contexts,
            "group_size": group_size, "learning_rate": learning_rate,
            "reward_scale_seed": reward_scale_seed,
            "checkpoint_interval_updates": checkpoint_interval_updates,
            "planned_updates_per_optimizer": updates,
            "fresh_actions_per_update": min(
                batch_contexts, min(contexts, sum(len(user.rl_anchors) > 0 for user in store.users)),
            ) * group_size,
            "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
            "reward_checkpoint": str(Path(reward_checkpoint).resolve()),
        }
        for key, value in expected.items():
            if resume_manifest.get(key) != value:
                raise ValueError(f"cached H=1 resume {key} differs from the original run")
        for key, value in (
            ("processed_listens_sha256", protocol["processed_listens_sha256"]),
            ("catalog_sha256", protocol["catalog_sha256"]),
        ):
            if resume_manifest.get(key) != value:
                raise ValueError(f"cached H=1 resume data hash differs: {key}")
        resume_meta = resume_preview.get("metadata", {})
        if resume_meta.get("method") != resume_method or int(resume_meta.get("seed", seed)) != seed:
            raise ValueError("cached H=1 resume checkpoint method/seed does not match its path")
        if float(resume_meta.get("beta", beta)) != beta or float(resume_meta.get("learning_rate", learning_rate)) != learning_rate:
            raise ValueError("cached H=1 resume beta/learning rate differs from the checkpoint")
    train_panel_ids = [
        {"uid": int(store.users[user_index].uid), "position": int(position)}
        for user_index, position in train_panel
    ]
    validation_panel_ids = [
        {"uid": int(store.users[user_index].uid), "position": int(position)}
        for user_index, position in validation_panel
    ]
    panel_payload = {
        "train_a_rl": train_panel_ids, "validation": validation_panel_ids,
        "train_sampling": "one RL anchor per distinct eligible user, without replacement",
        "validation_sampling": "one logged validation target per distinct user, without replacement",
        "seed": seed,
    }
    panel_path = output_dir / "context_panels.json"
    if resume_path is not None:
        if not panel_path.is_file() or json.loads(panel_path.read_text(encoding="utf-8")) != panel_payload:
            raise ValueError("cached H=1 context panel differs from the original run")
    else:
        panel_path.write_text(json.dumps(panel_payload, indent=2), encoding="utf-8")

    chunk_size = int(config.training.get("rl_reward_catalog_chunk_size", 512))
    cache_batch = int(config.training.get("rl_reward_cache_contexts", 16))
    train_cache_path = output_dir / "train_reward_matrix.npy"
    validation_cache_path = output_dir / "validation_reward_matrix.npy"
    if resume_path is not None:
        assert resume_manifest is not None
        train_cache = dict(resume_manifest["train_reward_cache"])
        validation_cache = dict(resume_manifest["validation_reward_cache"])
        for path, cache in ((train_cache_path, train_cache), (validation_cache_path, validation_cache)):
            if not path.is_file() or cache.get("sha256") != sha256_file(path):
                raise ValueError(f"cached H=1 reward matrix is missing or changed: {path}")
        scale = RewardScale(**resume_manifest["reward_scale"])
        scale_panel = dict(resume_manifest["reward_scale_panel"])
    else:
        train_cache = _cache_reward_matrix(
            reward_model, store, train_panel, item_features, train_cache_path,
            history_length=history_length, chunk_size=chunk_size, batch_size=cache_batch,
            reward_head=reward_head, seconds_scale=seconds_scale,
        )
        validation_cache = _cache_reward_matrix(
            reward_model, store, validation_panel, item_features, validation_cache_path,
            history_length=history_length, chunk_size=chunk_size, batch_size=cache_batch,
            reward_head=reward_head, seconds_scale=seconds_scale,
        )
        train_cache["sha256"] = sha256_file(train_cache_path)
        validation_cache["sha256"] = sha256_file(validation_cache_path)
        scale, scale_panel = _sample_reward_scale(
            actor, reward_model, store, item_features,
            count=int(config.training.get("rl_reward_scale_contexts", 512)), seed=reward_scale_seed,
            history_length=history_length, reward_head=reward_head, seconds_scale=seconds_scale,
        )
    train_catalog_rewards = torch.from_numpy(np.load(output_dir / "train_reward_matrix.npy").copy()).to(device)
    validation_catalog_rewards = torch.from_numpy(np.load(output_dir / "validation_reward_matrix.npy").copy()).to(device)
    train_rewards = torch.nn.functional.pad(train_catalog_rewards, (3, 0), value=0.0)
    validation_rewards = torch.nn.functional.pad(validation_catalog_rewards, (3, 0), value=0.0)
    train_rewards = _normalize_reward(train_rewards, scale)
    validation_rewards = _normalize_reward(validation_rewards, scale)
    train_ids, train_mask = _panel_tensors(store, train_panel, history_length, device)
    validation_ids, validation_mask = _panel_tensors(store, validation_panel, history_length, device)
    batch_size = min(batch_contexts, len(train_panel))
    batch_generator = random.Random(seed + 55_001)
    context_batches = [batch_generator.sample(range(len(train_panel)), batch_size) for _ in range(updates)]

    train_reference = _actor_log_distributions(reference, train_ids, train_mask, batch_size)
    validation_reference = _actor_log_distributions(reference, validation_ids, validation_mask, batch_size)
    b5_train = analytic_reward_kl_policy(train_reference, train_rewards, beta)
    b5_validation = analytic_reward_kl_policy(validation_reference, validation_rewards, beta)
    b5_train_j = float((b5_train * train_rewards).sum(dim=-1).mean() - beta * full_categorical_kl(b5_train.log(), train_reference).mean())
    b5_validation_j = float((b5_validation * validation_rewards).sum(dim=-1).mean() - beta * full_categorical_kl(b5_validation.log(), validation_reference).mean())
    initial_train = _measure_actor(actor, train_ids, train_mask, train_rewards, train_reference, beta, batch_size)
    initial_validation = _measure_actor(actor, validation_ids, validation_mask, validation_rewards, validation_reference, beta, batch_size)
    manifest: dict[str, Any] = {
        "run_id": output_dir.name, "run_dir": str(output_dir.resolve()), "stage": "EXACT_H1_CACHE",
        "status": "running", "seed": seed, "beta": beta, "reward_head": reward_head,
        "contexts": contexts, "batch_contexts": batch_contexts, "group_size": group_size,
        "learning_rate": learning_rate, "reward_scale_seed": reward_scale_seed,
        "checkpoint_interval_updates": checkpoint_interval_updates,
        "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
        "reward_checkpoint": str(Path(reward_checkpoint).resolve()),
        "actor_parameters": actor.parameter_count(),
        "history_contract": f"BOS plus up to {history_length - 1} preceding events",
        "train_panel_size": len(train_panel), "validation_panel_size": len(validation_panel),
        "train_reward_cache": train_cache, "validation_reward_cache": validation_cache,
        "reward_scale": {"mean": scale.mean, "std": scale.std, "sample_count": scale.sample_count},
        "reward_scale_panel": scale_panel,
        "b5_train_J": b5_train_j, "b5_validation_J": b5_validation_j,
        "initial_actor_train": initial_train, "initial_actor_validation": initial_validation,
        "planned_updates_per_optimizer": updates, "fresh_actions_per_update": batch_size * group_size,
        "model_cache_rm_actions": train_cache["rm_scored_actions"] + validation_cache["rm_scored_actions"],
        "total_rm_scored_actions_including_normalization": (
            train_cache["rm_scored_actions"] + validation_cache["rm_scored_actions"] + scale_panel["actions"]
        ),
        "source_sha256": source_tree_sha256(root / "src" / "tlab"),
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
        "environment": collect_environment(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if resume_manifest is not None:
        manifest = {
            **resume_manifest, **manifest,
            "started_at_utc": resume_manifest.get("started_at_utc", manifest["started_at_utc"]),
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest.pop("interrupted_method", None)
        manifest.pop("interrupted_update", None)
        manifest.pop("stop_reason", None)
    write_run_manifest(output_dir / "run_manifest.json", manifest)
    (output_dir / "b5_metrics.json").write_text(json.dumps({
        "beta": beta, "train_J": b5_train_j, "validation_J": b5_validation_j,
        "initial_actor_train": initial_train, "initial_actor_validation": initial_validation,
        "interpretation": "B5 is the unrestricted per-context optimum for this fixed RM/reward cache and reference.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    methods = ["exact_direct", "reinforce_loo", "grpo_centered_corrected", "grpo_standardized"]
    output_metrics = output_dir / "optimizer_metrics.jsonl"
    if resume_path is None:
        output_metrics.write_text("", encoding="utf-8")
    resume_metadata = (resume_preview or {}).get("metadata", {})
    resume_sampler = (resume_preview or {}).get("sampler_state", {})
    for method_index, method in enumerate(methods):
        method_dir = output_dir / method
        existing_method_manifest_path = method_dir / "run_manifest.json"
        if resume_path is not None and method_index < methods.index(resume_method):
            if not existing_method_manifest_path.is_file() or json.loads(existing_method_manifest_path.read_text(encoding="utf-8")).get("status") != "completed":
                raise ValueError("cached H=1 resume found an earlier optimizer method that is not complete")
            continue
        if resume_path is not None and method_index > methods.index(resume_method):
            resume_path = None
            resume_metadata = {}
            resume_sampler = {}
        policy = copy.deepcopy(actor).to(device).eval()
        reference_method = copy.deepcopy(reference).to(device).eval()
        for parameter in reference_method.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.0)
        is_resumed_method = resume_path is not None and method == resume_method
        if is_resumed_method:
            expected_method = {
                "method": method, "seed": seed, "beta": beta,
                "learning_rate": learning_rate, "updates_planned": updates,
                "checkpoint_interval_updates": checkpoint_interval_updates,
            }
            if not existing_method_manifest_path.is_file():
                raise ValueError("cached H=1 resume method manifest is missing")
            previous_method_manifest = json.loads(existing_method_manifest_path.read_text(encoding="utf-8"))
            for key, value in expected_method.items():
                if previous_method_manifest.get(key) != value:
                    raise ValueError(f"cached H=1 resume method {key} differs from the original run")
            load_training_checkpoint(resume, policy, optimizer, map_location=device)
            start_update = int(resume_sampler.get("next_update", resume_metadata.get("update", 0)))
            method_action_scores = int(resume_metadata.get("fresh_training_action_scores", 0))
            method_manifest = previous_method_manifest
            method_manifest.update({"status": "running", "resumed_at_utc": datetime.now(timezone.utc).isoformat()})
        else:
            torch.manual_seed(seed + 1_000 + method_index)
            start_update = 0
            method_action_scores = 0
            method_manifest = {
                "run_id": method_dir.name, "run_dir": str(method_dir.resolve()),
                "parent_run_id": output_dir.name, "method": method, "status": "running",
                "seed": seed, "updates_planned": updates, "beta": beta, "learning_rate": learning_rate,
                "checkpoint_interval_updates": checkpoint_interval_updates,
            }
        method_started = time.perf_counter()
        method_dir.mkdir(parents=True, exist_ok=True)
        write_run_manifest(method_dir / "run_manifest.json", method_manifest)
        if start_update >= updates:
            method_manifest.update({
                "status": "completed", "latest_update": updates,
                "fresh_training_action_scores": method_action_scores,
                "optimizer_steps": updates,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            write_run_manifest(method_dir / "run_manifest.json", method_manifest)
            continue
        for update in range(start_update, updates):
            indices = context_batches[update]
            batch_started = time.perf_counter()
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            ids = train_ids[idx]
            mask = train_mask[idx]
            reward_rows = train_rewards[idx]
            reference_rows = train_reference[idx]
            if method == "exact_direct":
                log_policy = policy.next_action_log_probs(ids, mask)
                objective = _exact_objective(log_policy, reward_rows, reference_rows, beta).mean()
                loss = -objective
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                update_metrics = {"loss": float(loss.detach()), "J_batch_before_update": float(objective.detach()),
                                  "grad_norm": float(grad_norm), "optimizer_steps": 1}
            else:
                def cached_reward(actions: Tensor) -> Tensor:
                    return reward_rows.gather(-1, actions.long())

                normalize = method == "grpo_standardized"
                centered = method == "grpo_centered_corrected"
                rollout = collect_one_step_rollout(
                    policy, reference_method, ids, mask, cached_reward,
                    group_size=group_size, normalize_advantages=normalize,
                    centered_correction=centered,
                )
                method_action_scores += batch_size * group_size
                if method == "reinforce_loo":
                    update_metrics = reinforce_update(policy, ids, mask, rollout, optimizer, beta=beta)
                else:
                    update_metrics = grpo_update(
                        policy, ids, mask, rollout, optimizer, beta=beta,
                        update_epochs=1, max_kl_to_old=None,
                    )
            if (update + 1) % checkpoint_interval_updates == 0 or update + 1 == updates:
                train_metrics = _measure_actor(
                    policy, train_ids, train_mask, train_rewards, train_reference, beta, batch_size,
                )
                validation_metrics = _measure_actor(
                    policy, validation_ids, validation_mask, validation_rewards,
                    validation_reference, beta, batch_size,
                )
                metric_row = {
                    "method": method, "update": update + 1,
                    "train": train_metrics, "validation": validation_metrics,
                    "b5_train_J": b5_train_j, "b5_validation_J": b5_validation_j,
                    "fresh_training_action_scores": method_action_scores,
                    "rm_model_action_scores": 0,
                    "update_seconds": time.perf_counter() - batch_started,
                    "elapsed_seconds": time.perf_counter() - method_started,
                    **update_metrics,
                }
                with output_metrics.open("a", encoding="utf-8") as metric_file:
                    metric_file.write(json.dumps(metric_row, ensure_ascii=False) + "\n")
                checkpoint_metadata = {
                    "run_id": method_dir.name, "method": method, "seed": seed,
                    "update": update + 1, "beta": beta, "learning_rate": learning_rate,
                    "fresh_training_action_scores": method_action_scores,
                }
                save_training_checkpoint(
                    method_dir / "last.pt", policy, optimizer, update + 1,
                    checkpoint_metadata, sampler_state={"next_update": update + 1},
                )
                method_manifest.update({
                    "latest_update": update + 1, "fresh_training_action_scores": method_action_scores,
                    "latest_train_J": train_metrics["J"], "latest_validation_J": validation_metrics["J"],
                    "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                    "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                })
                write_run_manifest(method_dir / "run_manifest.json", method_manifest)
                if stop_after_updates is not None and update + 1 >= stop_after_updates:
                    method_manifest.update({
                        "status": "interrupted",
                        "stop_reason": "intentional checkpoint-boundary interruption for resume-equivalence verification",
                        "interrupted_at_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    write_run_manifest(method_dir / "run_manifest.json", method_manifest)
                    manifest.update({
                        "status": "interrupted", "interrupted_method": method,
                        "interrupted_update": update + 1,
                        "stop_reason": method_manifest["stop_reason"],
                    })
                    write_run_manifest(output_dir / "run_manifest.json", manifest)
                    raise KeyboardInterrupt("intentional cached H=1 checkpoint-boundary stop")
        method_manifest.update({
            "status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - method_started,
            "fresh_training_action_scores": method_action_scores,
            "optimizer_steps": updates,
        })
        write_run_manifest(method_dir / "run_manifest.json", method_manifest)
    manifest.update({
        "status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer_metrics_path": str(output_metrics.resolve()),
        "total_fresh_training_action_scores": updates * batch_size * group_size * (len(methods) - 1),
        "note": "Exact direct optimization has no sampled action score budget; all algorithms reuse the frozen reward matrices.",
    })
    write_run_manifest(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-cache H=1 optimization comparison with B5 and exact direct gradients")
    parser.add_argument("--config", default="configs/reference_rl.yaml")
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--contexts", type=int, default=512)
    parser.add_argument("--batch-contexts", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--reward-head", choices=("ratio", "seconds"), default="ratio")
    parser.add_argument("--reward-scale-seed", type=int, default=90017)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=100)
    parser.add_argument("--stop-after-updates", type=int, help="intentional checkpointed interruption for resume verification")
    parser.add_argument("--resume", help="method last.pt; reuse the original cache and continue that optimizer")
    args = parser.parse_args()
    result = run_cached_h1(
        args.config, args.actor_checkpoint, args.reward_checkpoint, args.run_dir,
        seed=args.seed, beta=args.beta, contexts=args.contexts,
        batch_contexts=args.batch_contexts, group_size=args.group_size,
        updates=args.updates, learning_rate=args.learning_rate,
        reward_head=args.reward_head, reward_scale_seed=args.reward_scale_seed,
        checkpoint_interval_updates=args.checkpoint_interval_updates,
        stop_after_updates=args.stop_after_updates,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
