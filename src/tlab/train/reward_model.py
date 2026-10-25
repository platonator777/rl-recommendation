from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch
import numpy as np
from torch import Tensor

from tlab.data.catalog import PAD_ID
from tlab.data.store import SequenceStore, UserSequence
from tlab.models.reward import CandidateRewardModel, RewardModelShape
from tlab.runtime import collect_environment
from tlab.train.checkpoint import load_training_checkpoint, save_training_checkpoint, source_tree_sha256, write_run_manifest
from tlab.data.manifest import sha256_file
from tlab.train.reward import reward_model_loss


def sample_positions(
    users: Sequence[UserSequence],
    positions_name: str,
    max_targets: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Sample targets across users while preserving the full observed prefix."""
    counts = [len(getattr(user, positions_name)) for user in users]
    total = sum(counts)
    if total == 0 or max_targets < 1:
        return []
    budget = min(total, max_targets)
    exact = [budget * count / total for count in counts]
    quotas = [min(count, int(value)) for count, value in zip(counts, exact)]
    remaining = budget - sum(quotas)
    order = sorted(range(len(users)), key=lambda i: (-(exact[i] - int(exact[i])), users[i].uid))
    for index in order:
        if remaining <= 0:
            break
        if quotas[index] < counts[index]:
            quotas[index] += 1
            remaining -= 1
    rng = random.Random(seed)
    samples: list[tuple[int, int]] = []
    for user_index, (user, quota) in enumerate(zip(users, quotas)):
        if quota == 0:
            continue
        population = getattr(user, positions_name)
        if quota == len(population):
            chosen = [int(position) for position in population]
        else:
            chosen = sorted(int(population[i]) for i in rng.sample(range(len(population)), quota))
        samples.extend((user_index, position) for position in chosen)
    return samples


def sample_positions_uniform_user(
    users: Sequence[UserSequence],
    positions_name: str,
    max_targets: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Sample a similar number of validation anchors per user, then within user."""
    counts = [len(getattr(user, positions_name)) for user in users]
    eligible = [index for index, count in enumerate(counts) if count]
    total = sum(counts)
    if total == 0 or max_targets < 1:
        return []
    budget = min(total, max_targets)
    base, remainder = divmod(budget, len(eligible))
    quotas = [min(count, base) for count in counts]
    remaining = budget - sum(quotas)
    rng = random.Random(seed)
    order = eligible.copy()
    rng.shuffle(order)
    while remaining:
        progressed = False
        for user_index in order:
            if quotas[user_index] < counts[user_index]:
                quotas[user_index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
        rng.shuffle(order)
    samples: list[tuple[int, int]] = []
    for user_index, quota in enumerate(quotas):
        if quota == 0:
            continue
        population = getattr(users[user_index], positions_name)
        chosen = rng.sample(range(len(population)), quota) if quota < len(population) else range(len(population))
        samples.extend((user_index, int(population[position])) for position in sorted(chosen))
    return samples


def _read_process_rss() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


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
        "phase": manifest.get("variant"), "seed": manifest.get("seed"),
        "status": manifest.get("status", "running"),
        "manifest_path": str((Path(str(manifest["run_dir"])) / "run_manifest.json").resolve()),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_target_presentations": manifest.get("completed_target_presentations", 0),
        "best_validation_ndcg10": None,
        "stop_reason": manifest.get("stop_reason"),
    }
    runs = [row for row in registry.get("runs", []) if row.get("run_id") != run_id]
    runs.append(entry)
    registry["runs"] = sorted(runs, key=lambda row: (row.get("stage", ""), row.get("run_id", "")))
    temporary = registry_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(registry_path)


def collate_reward_examples(
    store: SequenceStore,
    examples: Sequence[tuple[int, int]],
    item_features: dict[int, tuple[float, bool] | tuple[float, bool, int]],
    seconds_cap: float,
    seconds_scale: float,
    history_length: int,
    device: torch.device,
    like_labels_by_uid: Mapping[int, Sequence[int]] | None = None,
) -> tuple[Tensor, ...]:
    histories: list[list[int]] = []
    candidate_ids: list[int] = []
    durations: list[float] = []
    duration_missing: list[bool] = []
    recent_repeat: list[bool] = []
    ratio_targets: list[float] = []
    ratio_masks: list[bool] = []
    seconds_targets: list[float] = []
    seconds_masks: list[bool] = []
    like_targets: list[float] = []
    like_masks: list[bool] = []
    for user_index, position in examples:
        user = store.users[user_index]
        candidate_id = int(user.tokens[position])
        history = user.history(position, history_length)
        histories.append(history)
        candidate_ids.append(candidate_id)
        feature = item_features.get(candidate_id, (0.0, True, 0))
        duration, missing = feature[:2]
        durations.append(duration)
        duration_missing.append(missing)
        recent_repeat.append(candidate_id in history[-20:])

        raw_ratio = float(user.ratio_raw[position])
        target_duration = float(user.duration_seconds[position])
        ratio_valid = math.isfinite(raw_ratio)
        seconds_valid = ratio_valid and math.isfinite(target_duration) and target_duration > 0
        ratio_targets.append(min(1.0, max(0.0, raw_ratio)) if ratio_valid else 0.0)
        ratio_masks.append(ratio_valid)
        seconds = raw_ratio * target_duration if seconds_valid else 0.0
        seconds_targets.append(min(max(0.0, seconds), seconds_cap) / seconds_scale if seconds_valid else 0.0)
        seconds_masks.append(seconds_valid)
        per_user_labels = like_labels_by_uid.get(user.uid) if like_labels_by_uid is not None else None
        like_label = int(per_user_labels[position]) if per_user_labels is not None and position < len(per_user_labels) else -1
        like_targets.append(float(like_label == 1))
        like_masks.append(like_label in (0, 1))

    width = max(map(len, histories))
    input_ids = torch.full((len(examples), width), PAD_ID, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for row, history in enumerate(histories):
        input_ids[row, :len(history)] = torch.tensor(history, dtype=torch.long, device=device)
        attention[row, :len(history)] = 1
    return (
        input_ids,
        attention,
        torch.tensor(candidate_ids, dtype=torch.long, device=device),
        torch.tensor(durations, dtype=torch.float32, device=device),
        torch.tensor(duration_missing, dtype=torch.bool, device=device),
        torch.tensor(recent_repeat, dtype=torch.bool, device=device),
        torch.tensor(ratio_targets, dtype=torch.float32, device=device),
        torch.tensor(ratio_masks, dtype=torch.bool, device=device),
        torch.tensor(seconds_targets, dtype=torch.float32, device=device),
        torch.tensor(seconds_masks, dtype=torch.bool, device=device),
        torch.tensor(like_targets, dtype=torch.float32, device=device),
        torch.tensor(like_masks, dtype=torch.bool, device=device),
    )


@torch.no_grad()
def evaluate_reward_model(
    model: CandidateRewardModel,
    store: SequenceStore,
    examples: Sequence[tuple[int, int]],
    item_features: dict[int, tuple[float, bool] | tuple[float, bool, int]],
    seconds_cap: float,
    seconds_scale: float,
    history_length: int,
    batch_size: int,
    device: torch.device,
    like_labels_by_uid: Mapping[int, Sequence[int]] | None = None,
) -> dict[str, float | int]:
    model.eval()
    ratio_errors: list[Tensor] = []
    ratio_errors_supported: list[Tensor] = []
    seconds_errors: list[Tensor] = []
    cap_labels: list[Tensor] = []
    like_probabilities: list[Tensor] = []
    like_targets: list[Tensor] = []
    ratio_count = seconds_count = 0
    unsupported_targets = 0
    for start in range(0, len(examples), batch_size):
        selected_examples = examples[start:start + batch_size]
        for user_index, position in selected_examples:
            item_token = int(store.users[user_index].tokens[position])
            feature = item_features.get(item_token, (0.0, True, 0))
            support = int(feature[2]) if len(feature) > 2 else 0
            unsupported_targets += int(support <= 0)
        batch = collate_reward_examples(
            store, selected_examples, item_features, seconds_cap, seconds_scale,
            history_length, device, like_labels_by_uid,
        )
        prediction = model(*batch[:6])
        ratio_error = prediction["ratio"].float() - batch[6]
        seconds_error = prediction["seconds_scaled"].float() * seconds_scale - batch[8] * seconds_scale
        ratio_mask = batch[7]
        supported_mask = torch.tensor([
            int((item_features.get(int(store.users[user_index].tokens[position]), (0.0, True, 0))[2]
                 if len(item_features.get(int(store.users[user_index].tokens[position]), (0.0, True, 0))) > 2 else 0) > 0)
            for user_index, position in selected_examples
        ], dtype=torch.bool)
        seconds_mask = batch[9]
        if ratio_mask.any():
            ratio_errors.append(ratio_error[ratio_mask].cpu())
            ratio_count += int(ratio_mask.sum().item())
        warm_ratio_mask = ratio_mask & supported_mask.to(device)
        if warm_ratio_mask.any():
            ratio_errors_supported.append(ratio_error[warm_ratio_mask].cpu())
        if seconds_mask.any():
            seconds_errors.append(seconds_error[seconds_mask].cpu())
            cap_labels.append((batch[8][seconds_mask] >= seconds_cap / seconds_scale).cpu())
            seconds_count += int(seconds_mask.sum().item())
        like_mask = batch[11]
        if like_mask.any():
            like_probabilities.append(prediction["like_probability"].float()[like_mask].cpu())
            like_targets.append(batch[10][like_mask].cpu())
    if not ratio_count:
        raise ValueError("RM validation has no finite completion-ratio labels")
    if not ratio_errors_supported:
        raise ValueError("RM validation has no supported-item ratio labels")
    ratio_all = torch.cat(ratio_errors)
    ratio_supported = torch.cat(ratio_errors_supported)
    seconds_all = torch.cat(seconds_errors) if seconds_errors else torch.empty(0)
    result: dict[str, float | int | None | str] = {
        "n_examples": len(examples),
        "unsupported_target_count": unsupported_targets,
        "unsupported_target_fraction": unsupported_targets / len(examples) if examples else None,
        "ratio_count": ratio_count,
        "ratio_mae": float(ratio_all.abs().mean()),
        "ratio_rmse": float(ratio_all.square().mean().sqrt()),
        "ratio_supported_count": int(ratio_supported.numel()),
        "ratio_mae_supported": float(ratio_supported.abs().mean()),
        "ratio_rmse_supported": float(ratio_supported.square().mean().sqrt()),
        "seconds_count": seconds_count,
        "seconds_mae": float(seconds_all.abs().mean()) if seconds_count else None,
        "seconds_rmse": float(seconds_all.square().mean().sqrt()) if seconds_count else None,
        "seconds_target_cap_rate": float(torch.cat(cap_labels).float().mean()) if cap_labels else None,
        "like_count": int(sum(part.numel() for part in like_targets)),
    }
    if like_targets:
        probabilities = torch.cat(like_probabilities).clamp(1e-7, 1.0 - 1e-7)
        labels = torch.cat(like_targets)
        result["like_prevalence"] = float(labels.mean())
        result["like_log_loss"] = float((-(labels * probabilities.log() + (1.0 - labels) * (1.0 - probabilities).log())).mean())
        result["like_brier"] = float((probabilities - labels).square().mean())
        result["like_average_precision"] = None
        if bool((labels == 1).any()):
            from sklearn.metrics import average_precision_score

            result["like_average_precision"] = float(average_precision_score(labels.numpy(), probabilities.numpy()))
        result["like_head_status"] = "trained_and_evaluated"
    else:
        result.update({
            "like_prevalence": None, "like_log_loss": None, "like_brier": None,
            "like_average_precision": None, "like_head_status": "inactive_no_valid_attributed_labels",
        })
    return result


def run_reward_model(
    config_path: str | Path,
    variant: str = "fit",
    seed: int = 17,
    max_targets: int | None = None,
    validation_limit: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    run_dir: str | Path | None = None,
    validation_seed: int | None = None,
    early_stopping_patience: int | None = None,
    checkpoint_interval_steps: int | None = None,
    resume: str | Path | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, object]:
    import polars as pl

    from tlab.config import load_config

    if variant not in {"fit", "audit"}:
        raise ValueError("variant must be 'fit' or 'audit'")
    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    raw_dir = Path(config.data.get("source_dir", "data/raw"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    cache_dir = processed_dir / "sequence_index"
    store = SequenceStore.load_cache(cache_dir, expected_source_sha256=protocol["processed_listens_sha256"])
    feature_frame = pl.read_parquet(processed_dir / f"rm_item_features_{variant}.parquet")
    item_features = {
        int(row["token_id"]): (
            float(row["rm_duration_seconds"] or 0.0),
            bool(row["rm_duration_missing"]),
            int(row["rm_training_support"]),
        )
        for row in feature_frame.iter_rows(named=True)
    }
    like_manifest_path = processed_dir / "like_labels_30m.json"
    like_labels_by_uid: dict[int, Sequence[int]] | None = None
    like_manifest: dict[str, object] | None = None
    if like_manifest_path.exists():
        like_manifest = json.loads(like_manifest_path.read_text(encoding="utf-8"))
        if (
            like_manifest.get("source_listens_sha256") != protocol["processed_listens_sha256"]
            or int(like_manifest.get("window_seconds", 0)) != 1800
            or like_manifest.get("source_likes_sha256") != sha256_file(raw_dir / "likes.parquet")
            or like_manifest.get("source_unlikes_sha256") != sha256_file(raw_dir / "unlikes.parquet")
            or like_manifest.get("labels_sha256") != sha256_file(processed_dir / "like_labels_30m.bin")
        ):
            raise ValueError("30-minute like label index is stale; rebuild it with tlab.data.like_index")
        labels = np.memmap(
            processed_dir / "like_labels_30m.bin", dtype="i1", mode="r",
            shape=(int(like_manifest["sequence_events"]),),
        )
        like_labels_by_uid = {}
        for entry in like_manifest["users"]:
            user_labels = labels[int(entry["offset"]):int(entry["offset"]) + int(entry["length"])]
            like_labels_by_uid[int(entry["uid"])] = user_labels
        for user in store.users:
            user_labels = like_labels_by_uid.get(user.uid)
            if user_labels is None or len(user_labels) != len(user.tokens):
                raise ValueError(f"like label index does not align with cached sequence for uid={user.uid}")

    run_seed = int(seed)
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config.model
    training_config = config.training
    model = CandidateRewardModel(RewardModelShape(
        vocab_size=int(config.dataset.get("catalog_size", 20_000)) + 3,
        hidden_size=int(training_config.get("rm_hidden_size", 128)),
        n_layer=int(training_config.get("rm_layers", 2)),
        n_head=int(training_config.get("rm_heads", 4)),
        max_positions=int(model_config.get("n_positions", 256)),
        encoder="transformer" if variant == "fit" else "gru",
    )).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_config.get("rm_learning_rate", 1e-4)), weight_decay=0.01,
    )
    target_budget = int(max_targets or training_config.get("rm_max_targets_per_epoch", 100_000))
    epoch_count = int(epochs or training_config.get("rm_epochs", 5))
    batch = int(batch_size or training_config.get("rm_batch_size", 64))
    val_limit = int(validation_limit or training_config.get("rm_validation_targets", 20_000))
    users = [user for user in store.users if user.rm_user_partition == variant]
    if not users:
        raise ValueError(f"there are no users assigned to the RM {variant!r} partition")
    train_examples = sample_positions(users, "rm_fit_positions", target_budget, run_seed)
    validation_seed = int(
        validation_seed if validation_seed is not None
        else training_config.get("rm_validation_seed", run_seed + 100_003)
    )
    validation_examples = sample_positions_uniform_user(users, "rm_validation_positions", val_limit, validation_seed)
    if not train_examples or not validation_examples:
        raise ValueError(f"RM {variant} needs both F_RM training labels and V_RM validation labels")
    # Sampling returns indices relative to the partition-filtered user list.
    selected_store = SequenceStore(users, store.events, store.tie_blocks, store.tied_events)
    history_length = int(config.dataset.get("history_length", 200))
    seconds_cap = float(protocol["seconds_cap_f_rm_p99_5"])
    seconds_scale = float(protocol["seconds_scale_f_rm_p95"])
    if run_dir:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = root / run_path
    else:
        artifacts_dir = Path(config.project.get("artifacts_dir", "artifacts"))
        if not artifacts_dir.is_absolute():
            artifacts_dir = root / artifacts_dir
        run_path = artifacts_dir / "runs" / f"rm_{variant}_seed{run_seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_path.mkdir(parents=True, exist_ok=True)
    patience = int(early_stopping_patience or training_config.get("rm_early_stopping_patience", 2))
    save_every = int(checkpoint_interval_steps or training_config.get("rm_checkpoint_interval_steps", 500))
    if patience < 1 or save_every < 1:
        raise ValueError("RM patience and checkpoint interval must be positive")
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError("stop_after_steps must be positive")
    resume_payload = torch.load(resume, map_location="cpu", weights_only=False) if resume is not None else None
    resume_meta = (resume_payload or {}).get("metadata", {})
    resume_sampler = (resume_payload or {}).get("sampler_state", {})
    if resume is not None:
        if resume_meta.get("variant", variant) != variant or int(resume_meta.get("seed", run_seed)) != run_seed:
            raise ValueError("RM resume checkpoint variant/seed does not match this run")
        if resume_meta.get("run_id") not in (None, run_path.name):
            raise ValueError("RM resume must reuse the checkpoint's original run directory")
        if int(resume_meta.get("target_budget", target_budget)) != target_budget:
            raise ValueError("RM resume target budget differs from the checkpoint")
        if int(resume_meta.get("batch_size", batch)) != batch:
            raise ValueError("RM resume batch size differs from the checkpoint")
        if int(resume_meta.get("validation_seed", validation_seed)) != validation_seed:
            raise ValueError("RM resume validation seed differs from the checkpoint")
    manifest: dict[str, object] = {
        "run_id": run_path.name,
        "run_dir": str(run_path.resolve()),
        "status": "running",
        "stage": "RM_fit" if variant == "fit" else "RM_audit",
        "variant": variant,
        "seed": run_seed,
        "encoder": model.shape.encoder,
        "source_sha256": source_tree_sha256(root / "src" / "tlab"),
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
        "rm_item_features_sha256": protocol["rm_item_features_by_partition"][variant]["sha256"],
        "rm_user_partition": variant,
        "n_users": len(users),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "validation_sampling_seed": validation_seed,
        "seconds_cap": seconds_cap,
        "seconds_scale": seconds_scale,
        "like_label_index_sha256": like_manifest.get("labels_sha256") if like_manifest else None,
        "like_label_status": "available" if like_labels_by_uid is not None else "inactive_no_label_index",
        "epochs": epoch_count,
        "early_stopping_patience": patience,
        "checkpoint_interval_steps": save_every,
        "batch_size": batch,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "environment": collect_environment(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if resume is not None and (run_path / "run_manifest.json").exists():
        existing = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
        manifest = {**existing, **manifest, "resumed_at_utc": datetime.now(timezone.utc).isoformat()}
    write_run_manifest(run_path / "run_manifest.json", manifest)
    _update_reference_run_registry(root, manifest)
    metrics_path = run_path / "metrics.jsonl"
    best_rmse = float(resume_meta.get("best_ratio_rmse", float("inf")))
    best_epoch = resume_meta.get("best_epoch")
    bad_epochs = int(resume_meta.get("bad_epochs", 0))
    global_step = int((resume_payload or {}).get("step", 0))
    start_epoch = int(resume_meta.get("epoch", -1)) if resume is not None else 0
    start_cursor = int(resume_sampler.get("cursor", 0))
    epoch_complete = bool(resume_sampler.get("epoch_complete", True))
    if resume is not None:
        if epoch_complete:
            start_epoch += 1
            start_cursor = 0
        checkpoint = load_training_checkpoint(resume, model, optimizer, map_location=device)
        global_step = int(checkpoint["step"])
    if start_epoch >= epoch_count:
        raise ValueError("RM resume checkpoint has already reached the requested epoch limit")
    completed_target_presentations = int(resume_meta.get("completed_target_presentations", 0))
    epoch_sample_fingerprints = list(resume_meta.get("epoch_sample_fingerprints", []))
    started = time.perf_counter()
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, epoch_count):
            epoch_started = time.perf_counter()
            model.train()
            epoch_examples = sample_positions(users, "rm_fit_positions", target_budget, run_seed + epoch)
            random.Random(run_seed + epoch).shuffle(epoch_examples)
            if start_cursor >= len(epoch_examples) and epoch == start_epoch and start_cursor:
                raise ValueError("RM checkpoint cursor is outside its deterministic epoch sample")
            if epoch == start_epoch and resume is not None and not epoch_complete:
                train_loss_sum = float(resume_meta.get("epoch_train_loss_sum", 0.0))
                train_steps = int(resume_meta.get("epoch_train_steps", 0))
                component_sums = dict(resume_meta.get("epoch_component_sums", {
                    "ratio_mse": 0.0, "seconds_mse_scaled": 0.0, "like_bce": 0.0,
                }))
                cursor = start_cursor
            else:
                train_loss_sum = 0.0
                train_steps = 0
                component_sums = {"ratio_mse": 0.0, "seconds_mse_scaled": 0.0, "like_bce": 0.0}
                cursor = 0
            epoch_keys = np.fromiter(
                ((int(user_index) << 32) | int(position) for user_index, position in epoch_examples),
                dtype=np.uint64, count=len(epoch_examples),
            )
            epoch_fingerprint = hashlib.sha256(epoch_keys.tobytes()).hexdigest()
            epoch_overlap_with_previous = None
            if epoch > 0:
                previous_examples = sample_positions(users, "rm_fit_positions", target_budget, run_seed + epoch - 1)
                previous_keys = np.fromiter(
                    ((int(user_index) << 32) | int(position) for user_index, position in previous_examples),
                    dtype=np.uint64, count=len(previous_examples),
                )
                epoch_overlap_with_previous = int(np.intersect1d(epoch_keys, previous_keys, assume_unique=False).size)
            # Cursors refer to fully formed mini-batches. The deterministic example
            # sampler and restored RNG make an interrupted pass exactly replayable.
            cursor = (cursor // batch) * batch
            while cursor < len(epoch_examples):
                stop = min(cursor + batch, len(epoch_examples))
                tensors = collate_reward_examples(
                    selected_store, epoch_examples[cursor:stop], item_features,
                    seconds_cap, seconds_scale, history_length, device, like_labels_by_uid,
                )
                prediction = model(*tensors[:6])
                loss, components = reward_model_loss(
                    prediction, tensors[6], tensors[7], tensors[8], tensors[9], tensors[10], tensors[11],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += float(loss.detach())
                for name in component_sums:
                    component_sums[name] += float(components[name])
                train_steps += 1
                global_step += 1
                cursor = stop
                if global_step % save_every == 0 and cursor < len(epoch_examples):
                    checkpoint_metadata = {
                        "run_id": run_path.name, "variant": variant, "seed": run_seed,
                        "epoch": epoch, "best_epoch": best_epoch, "best_ratio_rmse": best_rmse,
                        "bad_epochs": bad_epochs, "target_budget": target_budget,
                        "batch_size": batch, "validation_seed": validation_seed,
                        "epoch_train_loss_sum": train_loss_sum,
                        "epoch_train_steps": train_steps,
                        "epoch_component_sums": component_sums,
                        "completed_target_presentations": completed_target_presentations,
                        "epoch_sample_fingerprints": epoch_sample_fingerprints,
                        "epoch_sample_fingerprint": epoch_fingerprint,
                        "epoch_overlap_with_previous": epoch_overlap_with_previous,
                    }
                    sampler_state = {
                        "cursor": cursor, "epoch_complete": False,
                        "sample_seed": run_seed + epoch, "shuffle_seed": run_seed + epoch,
                        "sample_fingerprint": epoch_fingerprint,
                    }
                    save_training_checkpoint(
                        run_path / "last.pt", model, optimizer, global_step,
                        checkpoint_metadata, sampler_state=sampler_state,
                    )
                    manifest.update({
                        "status": "running", "latest_epoch": epoch,
                        "latest_batch_cursor": cursor, "global_step": global_step,
                        "completed_target_presentations": completed_target_presentations + cursor,
                        "last_checkpoint": str((run_path / "last.pt").resolve()),
                        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                    })
                    write_run_manifest(run_path / "run_manifest.json", manifest)
                    _update_reference_run_registry(root, manifest)
                    if stop_after_steps is not None and global_step >= stop_after_steps:
                        manifest.update({
                            "status": "interrupted",
                            "stop_reason": "intentional checkpoint-boundary interruption for resume-equivalence verification",
                            "interrupted_at_utc": datetime.now(timezone.utc).isoformat(),
                        })
                        write_run_manifest(run_path / "run_manifest.json", manifest)
                        _update_reference_run_registry(root, manifest)
                        raise KeyboardInterrupt("intentional checkpoint-boundary interruption")

            completed_target_presentations += len(epoch_examples)
            unique_targets_path = run_path / "unique_train_targets.npy"
            if unique_targets_path.exists():
                # Windows cannot atomically replace a file while a NumPy memmap
                # still has it open. The per-epoch target index is bounded and
                # small enough to load normally before the atomic replacement.
                previous_unique = np.load(unique_targets_path, allow_pickle=False)
                unique_train_targets = np.union1d(previous_unique, epoch_keys)
            else:
                unique_train_targets = np.unique(epoch_keys)
            temporary_unique = unique_targets_path.with_suffix(".npy.tmp")
            with temporary_unique.open("wb") as unique_file:
                np.save(unique_file, unique_train_targets, allow_pickle=False)
            os.replace(temporary_unique, unique_targets_path)
            validation = evaluate_reward_model(
                model, selected_store, validation_examples, item_features,
                seconds_cap, seconds_scale, history_length, batch, device, like_labels_by_uid,
            )
            validation_rmse = float(validation["ratio_rmse_supported"])
            improved = validation_rmse < best_rmse
            if improved:
                best_rmse = validation_rmse
                best_epoch = epoch
                bad_epochs = 0
                save_training_checkpoint(
                    run_path / "best.pt", model, optimizer, global_step,
                    {"run_id": run_path.name, "variant": variant, "seed": run_seed,
                     "epoch": epoch, "epoch_complete": True, "best_epoch": best_epoch,
                     "validation": validation, "best_ratio_rmse": best_rmse},
                )
            else:
                bad_epochs += 1
            save_training_checkpoint(
                run_path / "last.pt", model, optimizer, global_step,
                {"run_id": run_path.name, "variant": variant, "seed": run_seed,
                 "epoch": epoch, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                 "best_ratio_rmse": best_rmse, "validation": validation,
                 "completed_target_presentations": completed_target_presentations,
                 "epoch_sample_fingerprints": [*epoch_sample_fingerprints, epoch_fingerprint]},
                sampler_state={"cursor": len(epoch_examples), "epoch_complete": True,
                               "sample_seed": run_seed + epoch, "shuffle_seed": run_seed + epoch,
                               "sample_fingerprint": epoch_fingerprint},
            )
            epoch_sample_fingerprints.append(epoch_fingerprint)
            result = {
                "stage": manifest["stage"], "variant": variant, "seed": run_seed,
                "epoch": epoch, "global_step": global_step,
                "mean_train_loss": train_loss_sum / max(1, train_steps),
                "train_components": {name: value / max(1, train_steps) for name, value in component_sums.items()},
                "train_steps": train_steps, "validation": validation,
                "unique_train_targets_this_epoch": len(epoch_keys),
                "unique_train_targets_cumulative": int(len(unique_train_targets)),
                "target_presentations_cumulative": completed_target_presentations,
                "epoch_sample_fingerprint": epoch_fingerprint,
                "overlap_with_previous_epoch": epoch_overlap_with_previous,
                "gradient_norm_last_step": float(gradient_norm) if train_steps else None,
                "best_checkpoint": improved, "epoch_seconds": time.perf_counter() - epoch_started,
                "process_rss_bytes": _read_process_rss(),
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
            }
            metrics_file.write(json.dumps(result, ensure_ascii=True, allow_nan=False) + "\n")
            metrics_file.flush()
            print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
            manifest.update({
                "status": "running", "latest_epoch": epoch,
                "completed_target_presentations": completed_target_presentations,
                "unique_train_targets": int(len(unique_train_targets)),
                "global_step": global_step,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            write_run_manifest(run_path / "run_manifest.json", manifest)
            _update_reference_run_registry(root, manifest)
            if bad_epochs >= patience:
                break
            start_cursor = 0
            resume = None
            epoch_complete = True
    result = {
        "run_dir": str(run_path.resolve()),
        "best_checkpoint": str((run_path / "best.pt").resolve()),
        "best_ratio_rmse": best_rmse,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "completed_target_presentations": completed_target_presentations,
        "unique_train_targets": int(len(unique_train_targets)) if "unique_train_targets" in locals() else 0,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
    }
    manifest.update(result)
    manifest["status"] = "early_stopped" if bad_epochs >= patience and epoch + 1 < epoch_count else "completed"
    manifest["stop_reason"] = (
        f"validation patience {patience} reached" if manifest["status"] == "early_stopped"
        else "completed configured epoch budget"
    )
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_run_manifest(run_path / "run_manifest.json", manifest)
    _update_reference_run_registry(root, manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ratio/seconds reward model fit or audit variant")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--variant", choices=("fit", "audit"), default="fit")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--checkpoint-interval-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-steps", type=int, help="checkpointed interruption for resume verification only")
    args = parser.parse_args()
    print(json.dumps(run_reward_model(
        args.config, variant=args.variant, seed=args.seed,
        max_targets=args.max_targets, validation_limit=args.validation_limit,
        epochs=args.epochs, batch_size=args.batch_size, run_dir=args.run_dir,
        validation_seed=args.validation_seed,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        resume=args.resume,
        stop_after_steps=args.stop_after_steps,
    ), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
