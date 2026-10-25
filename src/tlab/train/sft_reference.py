from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tlab.data.store import SFTTargetIndex, SequenceStore
from tlab.evaluation.ranking import evaluate_full_catalog
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.runtime import collect_environment
from tlab.train.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    source_tree_sha256,
    write_run_manifest,
)
from tlab.train.sft import collate_sequences, train_sft_accumulation_step


SAMPLER_VERSION = "sft_target_index_reference"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_process_rss() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _update_run_registry(root: Path, manifest: dict[str, Any]) -> None:
    registry_path = root / "artifacts" / "reference" / "run_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            registry = {"protocol_id": "tlab_yambda_reference_2026-10-13", "runs": []}
    else:
        registry = {"protocol_id": "tlab_yambda_reference_2026-10-13", "runs": []}
    run_id = manifest["run_id"]
    entry = {
        "run_id": run_id,
        "stage": manifest.get("stage"),
        "phase": manifest.get("phase"),
        "seed": manifest.get("seed"),
        "status": manifest.get("status"),
        "manifest_path": str((Path(manifest["run_dir"]) / "run_manifest.json").resolve()) if manifest.get("run_dir") else None,
        "updated_at_utc": _now(),
        "run_target_presentations": manifest.get("run_target_presentations", 0),
        "best_validation_ndcg10": manifest.get("best_validation_ndcg10"),
        "stop_reason": manifest.get("stop_reason"),
    }
    runs = [row for row in registry.get("runs", []) if row.get("run_id") != run_id]
    runs.append(entry)
    registry["runs"] = sorted(runs, key=lambda row: (row.get("stage", ""), row.get("run_id", "")))
    temporary = registry_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(registry_path)


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    planned_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if planned_steps < 1 or not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("invalid scheduler budget or warmup fraction")
    warmup_steps = max(1, int(warmup_fraction * planned_steps)) if warmup_fraction else 0

    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        return max(0.0, (planned_steps - step) / max(1, planned_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)


def _sft_index_stats(store: SequenceStore, index: SFTTargetIndex) -> dict[str, Any]:
    users = np.frombuffer(index.user_indices, dtype=np.uint32)
    positions = np.frombuffer(index.positions, dtype=np.uint32)
    lengths = np.minimum(positions, index.history_events).astype(np.int32)
    target_items: set[int] = set()
    for user_index, position in zip(index.user_indices, index.positions):
        target_items.add(int(store.users[int(user_index)].tokens[int(position)]))
    if lengths.size:
        percentiles = {
            str(p): float(np.percentile(lengths, p)) for p in (0, 10, 25, 50, 75, 90, 99, 100)
        }
    else:
        percentiles = {}
    return {
        "selected_targets": len(index),
        "distinct_targets": len(index),
        "selected_users": int(np.unique(users).size),
        "selected_catalog_items": len(target_items),
        "context_history_events_percentiles": percentiles,
        "context_history_events_mean": float(lengths.mean()) if lengths.size else None,
        "target_position_in_example": "final token; loss uses the hidden state after BOS plus preceding events",
    }


def _checkpoint_metadata(
    *,
    phase: str,
    epoch: int,
    epoch_limit: int,
    epoch_complete: bool,
    target_cursor: int,
    target_budget: int,
    global_step: int,
    best_ndcg: float,
    best_epoch: int | None,
    bad_epochs: int,
    cumulative_target_presentations: int,
    schedule_start_epoch: int,
    schedule_planned_steps: int,
    base_learning_rate: float,
    sampler_seed: int,
    validation: dict[str, Any] | None,
    run_id: str,
    run_start_epoch: int,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "run_id": run_id,
        "run_start_epoch": run_start_epoch,
        "epoch": epoch,
        "epoch_limit": epoch_limit,
        "epoch_complete": epoch_complete,
        "target_cursor": target_cursor,
        "target_budget_per_epoch": target_budget,
        "global_step": global_step,
        "best_ndcg": best_ndcg,
        "best_epoch": best_epoch,
        "bad_epochs": bad_epochs,
        "cumulative_target_presentations": cumulative_target_presentations,
        "schedule_start_epoch": schedule_start_epoch,
        "schedule_planned_steps": schedule_planned_steps,
        "base_learning_rate": base_learning_rate,
        "sampler_seed": sampler_seed,
        "sampler_version": SAMPLER_VERSION,
        "validation": validation,
    }


def run_sft_reference(
    config_path: str | Path,
    *,
    seed: int = 17,
    max_targets: int | None = None,
    epochs: int | None = None,
    additional_epochs: int | None = None,
    schedule_epochs: int | None = None,
    learning_rate: float | None = None,
    validation_seed: int | None = None,
    validation_anchors_per_user: int | None = None,
    resume: str | Path | None = None,
    run_dir: str | Path | None = None,
    restart_schedule: bool = False,
    phase: str = "sft_reference",
    early_stopping_patience: int | None = None,
    minimum_additional_epochs: int | None = None,
    checkpoint_interval_steps: int | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    import polars as pl
    from torch.utils.tensorboard import SummaryWriter

    from tlab.config import load_config

    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    events_path = processed_dir / "listens.parquet"
    catalog_path = processed_dir / "catalog.parquet"
    if not events_path.exists() or not catalog_path.exists():
        raise FileNotFoundError("Run tlab.data.prepare before starting SFT reference")

    training_cfg = config.training
    model_cfg = config.model
    dataset_cfg = config.dataset
    target_budget = int(max_targets or training_cfg.get("sft_max_targets_per_epoch", 2_000_000))
    history_events = int(dataset_cfg.get("history_events", 200))
    if int(model_cfg.get("n_positions", 256)) < history_events + 2:
        raise ValueError("n_positions must fit BOS, the declared history and one target")
    microbatch_size = int(training_cfg.get("sft_microbatch", 4))
    effective_batch = int(training_cfg.get("sft_effective_batch", 64))
    collate_transfer_mode = str(training_cfg.get("sft_collate_transfer", "direct")).lower()
    if collate_transfer_mode not in {"direct", "cpu_staged"}:
        raise ValueError("sft_collate_transfer must be 'direct' or 'cpu_staged'")
    accumulation_steps = max(1, math.ceil(effective_batch / microbatch_size))
    actual_effective_batch = microbatch_size * accumulation_steps
    validation_seed = int(validation_seed if validation_seed is not None else training_cfg.get("validation_seed", 17))
    validation_anchors_per_user = int(
        validation_anchors_per_user
        if validation_anchors_per_user is not None
        else training_cfg.get("validation_anchors_per_user", 10)
    )
    patience = int(
        early_stopping_patience
        if early_stopping_patience is not None
        else training_cfg.get("sft_early_stopping_patience", 3)
    )
    min_additional = int(
        minimum_additional_epochs
        if minimum_additional_epochs is not None
        else training_cfg.get("sft_minimum_additional_epochs", 5)
    )
    checkpoint_interval = int(
        checkpoint_interval_steps
        if checkpoint_interval_steps is not None
        else training_cfg.get("sft_checkpoint_interval_steps", 1000)
    )
    warmup_fraction = float(training_cfg.get("sft_warmup_fraction", 0.05))
    base_learning_rate = float(learning_rate if learning_rate is not None else training_cfg.get("sft_learning_rate", 1e-4))
    if target_budget < 1 or microbatch_size < 1 or actual_effective_batch < 1 or patience < 1:
        raise ValueError("invalid SFT reference target, batch or early-stopping configuration")

    run_seed = int(seed)
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protocol_path = processed_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    processed_hash = str(protocol.get("processed_listens_sha256", ""))
    cache_dir = processed_dir / "sequence_index"
    store = None
    if (cache_dir / "index.json").exists():
        try:
            store = SequenceStore.load_cache(cache_dir, expected_source_sha256=processed_hash)
            print(f"Loaded sequence index cache: {cache_dir}", flush=True)
        except (ValueError, OSError, json.JSONDecodeError):
            store = None
    if store is None:
        store = SequenceStore.from_parquet(events_path)
        store.save_cache(cache_dir, source_sha256=processed_hash)
        print(f"Saved sequence index cache: {cache_dir}", flush=True)

    catalog_frame = pl.read_parquet(catalog_path).sort("token_id")
    catalog_item_ids = catalog_frame.get_column("item_id").to_list()
    actor = ItemGPT2Actor(
        ActorShape(
            vocab_size=len(catalog_item_ids) + 3,
            n_positions=int(model_cfg.get("n_positions", 256)),
            n_layer=int(model_cfg.get("n_layer", 4)),
            n_embd=int(model_cfg.get("n_embd", 256)),
            n_head=int(model_cfg.get("n_head", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    ).to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=base_learning_rate, weight_decay=0.01)
    requested_precision = str(training_cfg.get("precision", "auto")).lower()
    autocast_dtype: torch.dtype | None = None
    if device.type == "cuda":
        if requested_precision == "auto":
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif requested_precision == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 was requested but the selected GPU does not support it")
            autocast_dtype = torch.bfloat16
        elif requested_precision == "fp16":
            autocast_dtype = torch.float16
        elif requested_precision != "fp32":
            raise ValueError(f"unknown precision setting: {requested_precision}")
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and autocast_dtype == torch.float16))
    scaler_arg = scaler if scaler.is_enabled() else None

    checkpoint_preview: dict[str, Any] | None = None
    preview_metadata: dict[str, Any] = {}
    preview_sampler: dict[str, Any] = {}
    if resume is not None:
        checkpoint_preview = torch.load(resume, map_location="cpu", weights_only=False)
        preview_metadata = checkpoint_preview.get("metadata", {})
        preview_sampler = checkpoint_preview.get("sampler_state", {})
        if preview_sampler.get("sampler_version") not in (None, SAMPLER_VERSION):
            raise ValueError("cannot resume a checkpoint from a different SFT sampler version")

    target_cursor = 0
    if checkpoint_preview is not None:
        previous_epoch = int(preview_metadata.get("epoch", -1))
        previous_complete = bool(preview_sampler.get("epoch_complete", preview_metadata.get("epoch_complete", True)))
        previous_cursor = int(preview_sampler.get("target_cursor", preview_metadata.get("target_cursor", 0)))
        if not previous_complete and previous_cursor > 0:
            start_epoch = previous_epoch
            target_cursor = previous_cursor
        else:
            start_epoch = previous_epoch + 1

        if additional_epochs is not None:
            saved_limit = preview_metadata.get("epoch_limit")
            if preview_metadata.get("phase") == phase and not restart_schedule and saved_limit is not None:
                epoch_limit = int(saved_limit)
            else:
                if additional_epochs < 1:
                    raise ValueError("additional_epochs must be positive")
                epoch_limit = start_epoch + int(additional_epochs)
        elif epochs is not None:
            epoch_limit = int(epochs)
        else:
            epoch_limit = int(training_cfg.get("sft_epochs", 5))
    else:
        start_epoch = 0
        epoch_limit = int(epochs if epochs is not None else training_cfg.get("sft_epochs", 5))
        if additional_epochs is not None:
            epoch_limit = int(additional_epochs)
        if epoch_limit < 1:
            raise ValueError("configured SFT epoch count must be positive")

    if epoch_limit <= start_epoch:
        raise ValueError("configured epoch limit leaves no additional work after the checkpoint")
    steps_per_epoch = math.ceil(target_budget / actual_effective_batch)
    if checkpoint_preview is not None and preview_metadata.get("phase") == phase and not restart_schedule:
        schedule_start_epoch = int(preview_metadata.get("schedule_start_epoch", start_epoch))
        schedule_planned_steps = int(preview_metadata.get("schedule_planned_steps", (epoch_limit - start_epoch) * steps_per_epoch))
    else:
        schedule_start_epoch = start_epoch
        schedule_passes = int(schedule_epochs if schedule_epochs is not None else (epoch_limit - start_epoch))
        if schedule_passes < (epoch_limit - start_epoch):
            raise ValueError("schedule_epochs must cover every requested training pass")
        schedule_planned_steps = schedule_passes * steps_per_epoch
    scheduler = _make_scheduler(optimizer, schedule_planned_steps, warmup_fraction)

    if run_dir is None:
        artifact_dir = Path(config.project.get("artifacts_dir", "artifacts"))
        if not artifact_dir.is_absolute():
            artifact_dir = root / artifact_dir
        run_path = artifact_dir / "reference" / "runs" / f"sft_{phase}_seed{run_seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    else:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = root / run_path
    run_path.mkdir(parents=True, exist_ok=True)
    if (
        checkpoint_preview is not None
        and preview_metadata.get("phase") == phase
        and not restart_schedule
        and preview_metadata.get("run_id") not in (None, run_path.name)
    ):
        raise ValueError("same-phase mid-run resume must reuse its original --run-dir")

    global_step = 0
    best_ndcg = float("-inf")
    best_epoch: int | None = None
    bad_epochs = 0
    cumulative_target_presentations = 0
    run_target_presentations = 0
    if checkpoint_preview is not None:
        checkpoint = load_training_checkpoint(
            resume,
            actor,
            optimizer,
            scaler_arg,
            scheduler if (preview_metadata.get("phase") == phase and not restart_schedule) else None,
            map_location=device,
        )
        global_step = int(checkpoint["step"])
        metadata = checkpoint.get("metadata", {})
        same_phase = metadata.get("phase") == phase
        best_ndcg = float(metadata.get("best_ndcg", float("-inf"))) if same_phase else float("-inf")
        best_epoch = metadata.get("best_epoch") if same_phase else None
        bad_epochs = int(metadata.get("bad_epochs", 0)) if same_phase else 0
        cumulative_target_presentations = int(metadata.get("cumulative_target_presentations", 0))
        if restart_schedule or metadata.get("phase") != phase:
            for group in optimizer.param_groups:
                group["lr"] = base_learning_rate
                # Optimizer state can carry ``initial_lr`` from a previous
                # phase. LambdaLR reads that key when rebuilding base_lrs, so
                # resetting only ``lr`` silently restarts at the old peak.
                group["initial_lr"] = base_learning_rate
            scheduler = _make_scheduler(optimizer, schedule_planned_steps, warmup_fraction)
        elif "scheduler" not in checkpoint or checkpoint["scheduler"] is None:
            raise ValueError("same-phase resume checkpoint is missing scheduler state")
    reference_validation_ndcg10 = (
        float(preview_metadata["best_ndcg"])
        if checkpoint_preview is not None and preview_metadata.get("phase") != phase and preview_metadata.get("best_ndcg") is not None
        else None
    )

    same_run_resume = bool(checkpoint_preview and preview_metadata.get("run_id") == run_path.name)
    run_target_presentations = int(preview_sampler.get("run_target_presentations", 0)) if same_run_resume else 0
    run_start_epoch = int(preview_metadata.get("run_start_epoch", start_epoch)) if same_run_resume else start_epoch
    started = time.perf_counter()
    manifest_path = run_path / "run_manifest.json"
    existing_manifest = {}
    if resume is not None and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {
        **existing_manifest,
        "run_id": run_path.name,
        "run_dir": str(run_path.resolve()),
        "stage": "SFT_reference",
        "phase": phase,
        "seed": run_seed,
        "config_path": str(config.path),
        "config": config.raw,
        "source_sha256": source_tree_sha256(root / "src" / "tlab"),
        "dataset_sha256": None,
        "processed_listens_sha256": protocol.get("processed_listens_sha256"),
        "catalog_sha256": protocol.get("catalog_sha256"),
        "n_users": len(store.users),
        "raw_listen_rows": store.events,
        "sequence_tie_blocks": store.tie_blocks,
        "sequence_tied_events": store.tied_events,
        "eligible_sft_targets": sum(len(user.train_targets) for user in store.users),
        "target_budget_per_epoch": target_budget,
        "history_contract": f"BOS + up to {history_events} preceding events; current target is never in input",
        "sampling_version": SAMPLER_VERSION,
        "sampling_without_replacement_within_epoch": True,
        "per_user_target_quotas": "largest remainder proportional to eligible target count",
        "validation_seed": validation_seed,
        "validation_anchors_per_user": validation_anchors_per_user,
        "epoch_start": start_epoch,
        "epoch_limit_absolute": epoch_limit,
        "requested_additional_epochs": additional_epochs,
        "optimizer": {"name": "AdamW", "weight_decay": 0.01, "peak_learning_rate": base_learning_rate},
        "scheduler": {"name": "linear_warmup_decay", "warmup_fraction": warmup_fraction,
                      "planned_steps": schedule_planned_steps,
                      "horizon_passes": schedule_planned_steps // steps_per_epoch,
                      "restarted_on_resume": bool(restart_schedule)},
        "early_stopping": {"patience": patience, "minimum_additional_epochs": min_additional},
        "resume_checkpoint": str(Path(resume).resolve()) if resume is not None else None,
        "reference_validation_ndcg10": reference_validation_ndcg10,
        "resume_checkpoint_sha256": _sha256_file(resume) if resume is not None else None,
        "restart_schedule": bool(restart_schedule),
        "actual_effective_batch_targets": actual_effective_batch,
        "microbatch_size": microbatch_size,
        "collate_transfer_mode": collate_transfer_mode,
        "accumulation_steps": accumulation_steps,
        "checkpoint_interval_steps": checkpoint_interval,
        "precision": str(autocast_dtype or torch.float32),
        "catalog_size": len(catalog_item_ids),
        "actor_parameters": actor.parameter_count(),
        "environment": collect_environment(),
        "status": "running",
        "started_at_utc": existing_manifest.get("started_at_utc", _now()),
        "resumed_at_utc": _now() if resume is not None else None,
        "checkpoints": existing_manifest.get("checkpoints", []),
    }
    data_manifest_path = root / "data" / "data_manifest.json"
    if data_manifest_path.exists():
        manifest["dataset_sha256"] = json.loads(data_manifest_path.read_text(encoding="utf-8")).get("files", {}).get("listens", {}).get("sha256")
    write_run_manifest(manifest_path, manifest)
    _update_run_registry(root, manifest)

    writer = SummaryWriter(log_dir=str(run_path / "tensorboard"))
    metrics_path = run_path / "metrics.jsonl"
    total_train_seconds = 0.0
    total_sampling_seconds = 0.0
    total_materialization_seconds = 0.0
    total_evaluation_seconds = 0.0
    previous_target_keys: np.ndarray | None = None
    target_cursor_for_epoch = target_cursor
    try:
        if start_epoch > 0:
            previous_index = store.sample_sft_target_index(
                max_targets=target_budget,
                history_events=history_events,
                seed=run_seed + start_epoch - 1,
            )
            previous_target_keys = previous_index.target_keys()
            del previous_index
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            for epoch in range(start_epoch, epoch_limit):
                actor.train()
                sampler_seed = run_seed + epoch
                sampling_start = time.perf_counter()
                target_index = store.sample_sft_target_index(
                    max_targets=target_budget,
                    history_events=history_events,
                    seed=sampler_seed,
                )
                order = target_index.permutation(seed=sampler_seed + 1_000_003)
                sampling_seconds = time.perf_counter() - sampling_start
                total_sampling_seconds += sampling_seconds
                if len(target_index) == 0:
                    raise ValueError("no eligible SFT target examples were selected")
                if target_cursor_for_epoch >= len(target_index):
                    raise ValueError("checkpoint target cursor is beyond the sampled epoch")
                stats_start = time.perf_counter()
                index_stats = _sft_index_stats(store, target_index)
                target_keys = target_index.target_keys()
                overlap_count = int(np.intersect1d(previous_target_keys, target_keys, assume_unique=True).size) if previous_target_keys is not None else None
                statistics_seconds = time.perf_counter() - stats_start
                epoch_sampling_seed = sampler_seed
                metrics_file.write(json.dumps({
                    "stage": "SFT_reference_sampler",
                    "phase": phase,
                    "seed": run_seed,
                    "epoch": epoch,
                    "sampling_seed": epoch_sampling_seed,
                    "permutation_seed": sampler_seed + 1_000_003,
                    "sampling_seconds": sampling_seconds,
                    "sampling_statistics_seconds": statistics_seconds,
                    "overlap_with_previous_epoch": overlap_count,
                    "overlap_fraction_previous_epoch": overlap_count / len(target_index) if overlap_count is not None else None,
                    **index_stats,
                }, ensure_ascii=False) + "\n")
                metrics_file.flush()

                epoch_start = time.perf_counter()
                train_seconds = materialization_seconds = loss_sum = 0.0
                trained_targets_this_invocation = real_input_tokens = padded_input_tokens = oov_input_tokens = 0
                epoch_step_count = 0
                cursor = target_cursor_for_epoch
                next_batch: list[Any] = []
                while cursor < len(target_index):
                    micro_end = min(cursor + microbatch_size, len(target_index))
                    selected_ref_ids = order[cursor:micro_end]
                    token_sequences = []
                    supervision_masks = []
                    materialize_start = time.perf_counter()
                    for ref_id in selected_ref_ids:
                        tokens, mask = target_index.example(store, int(ref_id))
                        token_sequences.append(tokens)
                        supervision_masks.append(mask)
                    batch = collate_sequences(
                        token_sequences,
                        supervision_masks,
                        device=device,
                        transfer_mode=collate_transfer_mode,
                    )
                    materialized_seconds_this_batch = time.perf_counter() - materialize_start
                    materialization_seconds += materialized_seconds_this_batch
                    total_materialization_seconds += materialized_seconds_this_batch
                    next_batch.append(batch)
                    cursor = micro_end
                    if len(next_batch) < accumulation_steps and cursor < len(target_index):
                        continue

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    step_start = time.perf_counter()
                    step_metrics = train_sft_accumulation_step(
                        actor,
                        next_batch,
                        optimizer,
                        logit_chunk_size=int(training_cfg.get("sft_logit_chunk_size", 256)),
                        scaler=scaler_arg,
                        gradient_clip=float(training_cfg.get("sft_gradient_clip", 1.0)),
                        autocast_dtype=autocast_dtype,
                    )
                    scheduler.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    step_seconds = time.perf_counter() - step_start
                    train_seconds += step_seconds
                    global_step += 1
                    epoch_step_count += 1
                    trained_targets_this_invocation += int(step_metrics["supervised_targets"])
                    loss_sum += step_metrics["loss"] * step_metrics["supervised_targets"]
                    for batch_part in next_batch:
                        real_input_tokens += int(batch_part.attention_mask.sum().item())
                        padded_input_tokens += int(batch_part.attention_mask.numel())
                        oov_input_tokens += int(((batch_part.input_ids == 2) & batch_part.attention_mask.bool()).sum().item())
                    writer.add_scalar("sft_reference/train_loss", step_metrics["loss"], global_step)
                    writer.add_scalar("sft_reference/grad_norm", step_metrics["grad_norm"], global_step)
                    next_batch.clear()
                    total_train_seconds += step_seconds

                    cumulative_target_presentations += int(step_metrics["supervised_targets"])
                    run_target_presentations += int(step_metrics["supervised_targets"])
                    checkpoint_due = checkpoint_interval > 0 and epoch_step_count % checkpoint_interval == 0
                    intentional_stop_due = stop_after_steps is not None and global_step >= stop_after_steps
                    if (checkpoint_due or intentional_stop_due) and cursor < len(target_index):
                        metadata = _checkpoint_metadata(
                            phase=phase, epoch=epoch, epoch_limit=epoch_limit, epoch_complete=False,
                            target_cursor=cursor, target_budget=len(target_index), global_step=global_step,
                            best_ndcg=best_ndcg, best_epoch=best_epoch, bad_epochs=bad_epochs,
                            cumulative_target_presentations=cumulative_target_presentations,
                            schedule_start_epoch=schedule_start_epoch, schedule_planned_steps=schedule_planned_steps,
                            base_learning_rate=base_learning_rate, sampler_seed=sampler_seed, validation=None,
                            run_id=run_path.name, run_start_epoch=run_start_epoch,
                        )
                        save_training_checkpoint(
                            run_path / "last.pt", actor, optimizer, global_step, metadata,
                            scaler=scaler_arg, scheduler=scheduler,
                            sampler_state={
                                "sampler_version": SAMPLER_VERSION, "sampler_seed": sampler_seed,
                                "permutation_seed": sampler_seed + 1_000_003, "target_cursor": cursor,
                                "epoch_complete": False, "run_target_presentations": run_target_presentations,
                            },
                        )
                        progress = {
                            "stage": "SFT_reference_progress", "phase": phase, "seed": run_seed,
                            "epoch": epoch, "epoch_step": epoch_step_count, "global_step": global_step,
                            "target_cursor": cursor, "targets_in_epoch": len(target_index),
                            "cumulative_target_presentations": cumulative_target_presentations,
                            "epoch_train_targets_per_second": trained_targets_this_invocation / max(train_seconds, 1e-9),
                            "epoch_train_seconds": train_seconds, "sampling_seconds": sampling_seconds,
                            "materialization_seconds": materialization_seconds,
                            "rss_bytes": _read_process_rss(),
                            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                        }
                        metrics_file.write(json.dumps(progress, ensure_ascii=False) + "\n")
                        metrics_file.flush()
                        manifest.update({
                            "latest_epoch": epoch, "latest_target_cursor": cursor,
                            "cumulative_target_presentations": cumulative_target_presentations,
                            "run_target_presentations": run_target_presentations,
                            "last_checkpoint": str((run_path / "last.pt").resolve()),
                            "updated_at_utc": _now(),
                        })
                        write_run_manifest(manifest_path, manifest)
                        _update_run_registry(root, manifest)
                        if intentional_stop_due:
                            raise KeyboardInterrupt("intentional checkpoint-boundary stop for resume-equivalence verification")

                    if next_batch:
                        raise AssertionError("accumulation batches were not cleared after optimizer step")

                if cursor != len(target_index):
                    raise AssertionError(f"target cursor reached {cursor}, expected {len(target_index)}")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                validation_start = time.perf_counter()
                validation = evaluate_full_catalog(
                    actor=actor,
                    store=store,
                    catalog_item_ids=catalog_item_ids,
                    split="validation",
                    history_length=history_events + 1,
                    per_user_limit=validation_anchors_per_user,
                    seed=validation_seed,
                    batch_size=int(training_cfg.get("validation_batch_size", 16)),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                evaluation_seconds = time.perf_counter() - validation_start
                total_evaluation_seconds += evaluation_seconds
                score = float(validation["macro_user_unconditional"].get("ndcg@10", float("nan")))
                if not math.isfinite(score):
                    raise ValueError("validation NDCG@10 is not finite; refusing to continue/select")
                improved = score > best_ndcg
                if improved:
                    best_ndcg = score
                    best_epoch = epoch
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                epoch_seconds = time.perf_counter() - epoch_start
                additional_pass = max(0, epoch - run_start_epoch + 1) if additional_epochs is not None or resume is not None else None
                metadata = _checkpoint_metadata(
                    phase=phase, epoch=epoch, epoch_limit=epoch_limit, epoch_complete=True,
                    target_cursor=len(target_index), target_budget=len(target_index), global_step=global_step,
                    best_ndcg=best_ndcg, best_epoch=best_epoch, bad_epochs=bad_epochs,
                    cumulative_target_presentations=cumulative_target_presentations,
                    schedule_start_epoch=schedule_start_epoch, schedule_planned_steps=schedule_planned_steps,
                    base_learning_rate=base_learning_rate, sampler_seed=sampler_seed, validation=validation,
                    run_id=run_path.name, run_start_epoch=run_start_epoch,
                )
                sampler_state = {
                    "sampler_version": SAMPLER_VERSION, "sampler_seed": sampler_seed,
                    "permutation_seed": sampler_seed + 1_000_003, "target_cursor": len(target_index),
                    "epoch_complete": True, "run_target_presentations": run_target_presentations,
                }
                if improved:
                    save_training_checkpoint(
                        run_path / "best.pt", actor, optimizer, global_step, metadata,
                        scaler=scaler_arg, scheduler=scheduler, sampler_state=sampler_state,
                    )
                save_training_checkpoint(
                    run_path / "last.pt", actor, optimizer, global_step, metadata,
                    scaler=scaler_arg, scheduler=scheduler, sampler_state=sampler_state,
                )
                if additional_pass in (2, 5, 10, 20):
                    snapshot_name = f"pass_{additional_pass:02d}.pt"
                    save_training_checkpoint(
                        run_path / snapshot_name, actor, optimizer, global_step, metadata,
                        scaler=scaler_arg, scheduler=scheduler, sampler_state=sampler_state,
                    )
                    manifest.setdefault("checkpoints", []).append({
                        "kind": f"additional_pass_{additional_pass}",
                        "path": str((run_path / snapshot_name).resolve()),
                        "sha256": _sha256_file(run_path / snapshot_name),
                        "validation_ndcg10": score,
                    })
                epoch_metrics = {
                    "stage": "SFT_reference", "phase": phase, "seed": run_seed,
                    "epoch": epoch, "additional_pass": additional_pass,
                    "global_step": global_step,
                    "train_ce_this_invocation": loss_sum / max(1, trained_targets_this_invocation),
                    "validation_ce_in_catalog": validation.get("in_catalog_cross_entropy"),
                    "validation_targets_in_catalog": validation.get("n_in_catalog_targets"),
                    "validation_ndcg10": score,
                    "validation_recall10": validation["macro_user_unconditional"].get("recall@10"),
                    "validation_recall20": validation["macro_user_unconditional"].get("recall@20"),
                    "validation_recall100": validation["macro_user_unconditional"].get("recall@100"),
                    "validation_seed": validation_seed,
                    "validation_users": validation.get("n_users"),
                    "validation_targets": validation.get("n_targets"),
                    "train_targets": len(target_index),
                    "train_targets_this_invocation": trained_targets_this_invocation,
                    "cumulative_target_presentations": cumulative_target_presentations,
                    "unique_target_presentations": len(target_index),
                    "optimizer_steps": epoch_step_count,
                    "sampler_seconds": sampling_seconds,
                    "materialization_seconds": materialization_seconds,
                    "forward_backward_optimizer_seconds": train_seconds,
                    "validation_seconds": evaluation_seconds,
                    "epoch_seconds": epoch_seconds,
                    "training_targets_per_second": trained_targets_this_invocation / max(train_seconds, 1e-9),
                    "end_to_end_targets_per_second": len(target_index) / max(epoch_seconds, 1e-9),
                    "input_context_tokens": real_input_tokens,
                    "padding_tokens": max(0, padded_input_tokens - real_input_tokens),
                    "padding_fraction": 1.0 - real_input_tokens / max(1, padded_input_tokens),
                    "oov_context_tokens": oov_input_tokens,
                    "oov_context_fraction": oov_input_tokens / max(1, real_input_tokens),
                    "best_checkpoint": improved,
                    "bad_validation_evaluations": bad_epochs,
                    "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                    "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                    "process_rss_bytes": _read_process_rss(),
                }
                metrics_file.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
                metrics_file.flush()
                writer.add_scalar("sft_reference/validation_ndcg10", score, global_step)
                writer.add_scalar("sft_reference/validation_ce", validation.get("in_catalog_cross_entropy", float("nan")), global_step)
                print(json.dumps({key: value for key, value in epoch_metrics.items() if key not in ("validation",)}, ensure_ascii=False), flush=True)
                manifest.update({
                    "latest_epoch": epoch,
                    "latest_target_cursor": len(target_index),
                    "cumulative_target_presentations": cumulative_target_presentations,
                    "run_target_presentations": run_target_presentations,
                    "last_validation_ndcg10": score,
                    "best_validation_ndcg10": best_ndcg,
                    "best_epoch": best_epoch,
                    "last_checkpoint": str((run_path / "last.pt").resolve()),
                    "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                    "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                    "process_rss_bytes": _read_process_rss(),
                    "updated_at_utc": _now(),
                })
                write_run_manifest(manifest_path, manifest)
                _update_run_registry(root, manifest)
                previous_target_keys = target_keys
                target_cursor_for_epoch = 0

                additional_completed = epoch - run_start_epoch + 1
                if bad_epochs >= patience and additional_completed >= min_additional:
                    manifest["status"] = "early_stopped"
                    manifest["stop_reason"] = f"validation patience {patience} reached after {additional_completed} additional passes"
                    manifest["stopped_at_utc"] = _now()
                    break
        manifest["status"] = manifest.get("status") if manifest.get("status") == "early_stopped" else "completed"
        manifest["stop_reason"] = manifest.get("stop_reason") or ("completed configured epoch budget" if manifest["status"] == "completed" else None)
        manifest["run_target_presentations"] = run_target_presentations
        manifest["cumulative_target_presentations"] = cumulative_target_presentations
        manifest["global_step"] = global_step
        manifest["best_validation_ndcg10"] = best_ndcg
        manifest["best_epoch"] = best_epoch
        manifest["best_checkpoint_sha256"] = _sha256_file(run_path / "best.pt") if (run_path / "best.pt").exists() else None
        manifest["last_checkpoint_sha256"] = _sha256_file(run_path / "last.pt") if (run_path / "last.pt").exists() else None
        manifest["elapsed_seconds_this_invocation"] = time.perf_counter() - started
        manifest["resource_time_totals_this_invocation"] = {
            "training_seconds": total_train_seconds,
            "sampling_seconds": total_sampling_seconds,
            "materialization_seconds": total_materialization_seconds,
            "validation_seconds": total_evaluation_seconds,
        }
        manifest["finished_at_utc"] = _now()
        write_run_manifest(manifest_path, manifest)
        _update_run_registry(root, manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["stop_reason"] = f"{type(exc).__name__}: {exc}"
        manifest["updated_at_utc"] = _now()
        manifest["latest_target_cursor"] = locals().get("cursor", target_cursor)
        manifest["cumulative_target_presentations"] = cumulative_target_presentations
        manifest["run_target_presentations"] = run_target_presentations
        write_run_manifest(manifest_path, manifest)
        _update_run_registry(root, manifest)
        raise
    finally:
        writer.close()

    return {
        "run_dir": str(run_path.resolve()),
        "best_checkpoint": str((run_path / "best.pt").resolve()),
        "last_checkpoint": str((run_path / "last.pt").resolve()),
        "best_validation_ndcg10": best_ndcg,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "status": manifest["status"],
        "run_target_presentations": run_target_presentations,
        "elapsed_seconds": manifest.get("elapsed_seconds_this_invocation"),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
    }


def _sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="reference SFT with exact per-target histories and resumable target-index sampling")
    parser.add_argument("--config", default="configs/reference_sft.yaml")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--epochs", type=int, help="absolute epoch limit, for a fresh run or explicit continuation")
    parser.add_argument("--additional-epochs", type=int, help="number of passes to add after the resume checkpoint")
    parser.add_argument("--schedule-epochs", type=int,
                        help="learning-rate schedule horizon independent of a deliberately shorter run")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--validation-anchors-per-user", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--run-dir")
    parser.add_argument("--restart-schedule", action="store_true")
    parser.add_argument("--phase", default="sft_reference")
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--minimum-additional-epochs", type=int)
    parser.add_argument("--checkpoint-interval-steps", type=int)
    parser.add_argument("--stop-after-steps", type=int, help="intentional checkpointed interruption for resume verification only")
    args = parser.parse_args()
    print(json.dumps(run_sft_reference(
        args.config,
        seed=args.seed,
        max_targets=args.max_targets,
        epochs=args.epochs,
        additional_epochs=args.additional_epochs,
        schedule_epochs=args.schedule_epochs,
        learning_rate=args.learning_rate,
        validation_seed=args.validation_seed,
        validation_anchors_per_user=args.validation_anchors_per_user,
        resume=args.resume,
        run_dir=args.run_dir,
        restart_schedule=args.restart_schedule,
        phase=args.phase,
        early_stopping_patience=args.early_stopping_patience,
        minimum_additional_epochs=args.minimum_additional_epochs,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        stop_after_steps=args.stop_after_steps,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
