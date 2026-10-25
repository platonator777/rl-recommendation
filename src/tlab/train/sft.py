from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from tlab.data.catalog import PAD_ID
from tlab.data.store import SequenceStore
from tlab.evaluation.ranking import evaluate_full_catalog
from tlab.models.actor import ItemGPT2Actor, backward_chunked_full_softmax_ce
from tlab.models.actor import ActorShape
from tlab.runtime import collect_environment
from tlab.train.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    source_tree_sha256,
    write_run_manifest,
)


@dataclass
class SFTBatch:
    input_ids: Tensor
    attention_mask: Tensor
    targets: Tensor
    target_mask: Tensor


def collate_sequences(
    token_sequences: Sequence[Sequence[int]],
    supervised_positions: Sequence[Sequence[bool]],
    device: torch.device | str = "cpu",
    pad_id: int = PAD_ID,
    transfer_mode: str = "direct",
) -> SFTBatch:
    """Shift complete user sequences once; target mask refers to target positions."""
    target_device = torch.device(device)
    transfer_mode = str(transfer_mode).lower()
    if transfer_mode not in {"direct", "cpu_staged"}:
        raise ValueError("transfer_mode must be 'direct' or 'cpu_staged'")
    allocation_device = (
        torch.device("cpu")
        if transfer_mode == "cpu_staged" and target_device.type == "cuda"
        else target_device
    )
    if len(token_sequences) != len(supervised_positions):
        raise ValueError("sequence and supervision batches must have equal lengths")
    if not token_sequences or any(len(tokens) != len(mask) for tokens, mask in zip(token_sequences, supervised_positions)):
        raise ValueError("each sequence needs an equally sized target mask")
    if any(len(tokens) < 2 for tokens in token_sequences):
        raise ValueError("each training sequence needs BOS plus at least one event")
    max_steps = max(len(tokens) - 1 for tokens in token_sequences)
    batch = len(token_sequences)
    inputs = torch.full((batch, max_steps), pad_id, dtype=torch.long, device=allocation_device)
    targets = torch.full((batch, max_steps), pad_id, dtype=torch.long, device=allocation_device)
    attention = torch.zeros((batch, max_steps), dtype=torch.long, device=allocation_device)
    valid = torch.zeros((batch, max_steps), dtype=torch.bool, device=allocation_device)
    for row, (tokens, mask) in enumerate(zip(token_sequences, supervised_positions)):
        length = len(tokens) - 1
        inputs[row, :length] = torch.as_tensor(tokens[:-1], dtype=torch.long, device=allocation_device)
        targets[row, :length] = torch.as_tensor(tokens[1:], dtype=torch.long, device=allocation_device)
        attention[row, :length] = 1
        valid[row, :length] = torch.as_tensor(mask[1:], dtype=torch.bool, device=allocation_device)
        valid[row, :length] &= targets[row, :length] >= 3
    if allocation_device != target_device:
        inputs = inputs.to(target_device)
        targets = targets.to(target_device)
        attention = attention.to(target_device)
        valid = valid.to(target_device)
    return SFTBatch(inputs, attention, targets, valid)


def train_sft_accumulation_step(
    actor: ItemGPT2Actor,
    microbatches: Sequence[SFTBatch],
    optimizer: torch.optim.Optimizer,
    logit_chunk_size: int = 256,
    scaler: object | None = None,
    gradient_clip: float = 1.0,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Backprop exact target-weighted CE across microbatches with one update."""
    counts = [int(batch.target_mask.sum().item()) for batch in microbatches]
    total = sum(counts)
    if total == 0:
        raise ValueError("accumulation step has no supervised targets")
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for index, (batch, count) in enumerate(zip(microbatches, counts)):
        if count == 0:
            continue
        device_type = batch.input_ids.device.type
        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            hidden = actor.encode(batch.input_ids, batch.attention_mask)
            total_loss += backward_chunked_full_softmax_ce(
                hidden=hidden,
                targets=batch.targets,
                valid=batch.target_mask,
                output_weight=actor.output_weight,
                chunk_size=logit_chunk_size,
                grad_scale=scaler,
                loss_weight=count / total,
            )
        del hidden
    if scaler is not None:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    return {"loss": total_loss, "supervised_targets": float(total), "grad_norm": float(grad_norm)}


def run_sft(
    config_path: str | Path,
    seed: int = 17,
    max_targets: int | None = None,
    epochs: int | None = None,
    validation_anchors_per_user: int = 10,
    resume: str | Path | None = None,
    run_dir: str | Path | None = None,
    positive_only: bool = False,
    restart_schedule: bool = False,
) -> dict[str, object]:
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
        raise FileNotFoundError("Run tlab.data.prepare before starting SFT")

    run_seed = int(seed)
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_cfg = config.training
    model_cfg = config.model
    target_budget = int(max_targets or training_cfg.get("sft_max_targets_per_epoch", 2_000_000))
    epoch_count = int(epochs or training_cfg.get("sft_epochs", 5))
    microbatch_size = int(training_cfg.get("sft_microbatch", 4))
    effective_batch = int(training_cfg.get("sft_effective_batch", 64))
    accumulation_steps = max(1, (effective_batch + microbatch_size - 1) // microbatch_size)
    history_length = int(config.dataset.get("history_length", 200))
    start_epoch = 0
    if resume is not None:
        resume_metadata = torch.load(resume, map_location="cpu", weights_only=False).get("metadata", {})
        start_epoch = int(resume_metadata.get("epoch", -1)) + 1
        if start_epoch >= epoch_count:
            raise ValueError("resume checkpoint already reached the configured epoch count")

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
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(training_cfg.get("sft_learning_rate", 3e-4)),
        weight_decay=0.01,
    )
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

    # Build the deterministic target windows once to size the warmup schedule.
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
    schedule_epochs = range(start_epoch, epoch_count) if restart_schedule else range(epoch_count)
    epoch_window_counts: dict[int, int] = {}
    for epoch in schedule_epochs:
        window_count = sum(1 for _ in store.sample_sft_windows(
            max_targets=target_budget,
            history_length=history_length,
            n_positions=int(model_cfg.get("n_positions", 256)),
            seed=run_seed + epoch,
            positive_only=positive_only,
        ))
        if window_count == 0:
            target_kind = "positive" if positive_only else "catalog-contained recommendation"
            raise ValueError(f"No {target_kind} targets are eligible for SFT")
        epoch_window_counts[epoch] = window_count
    optimizer_steps = sum(
        (epoch_window_counts[epoch] + microbatch_size * accumulation_steps - 1) // (microbatch_size * accumulation_steps)
        for epoch in schedule_epochs
    )
    base_learning_rate = float(training_cfg.get("sft_learning_rate", 3e-4))

    def make_scheduler(planned_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        warmup_steps = max(1, int(0.05 * planned_steps))

        def lr_scale(step: int) -> float:
            if step < warmup_steps:
                return max(1e-3, (step + 1) / warmup_steps)
            return max(0.0, (planned_steps - step) / max(1, planned_steps - warmup_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    scheduler = make_scheduler(optimizer_steps)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if run_dir is None:
        base_artifacts = Path(config.project.get("artifacts_dir", "artifacts"))
        if not base_artifacts.is_absolute():
            base_artifacts = root / base_artifacts
        run_path = base_artifacts / "runs" / f"sft_seed{run_seed}_{now}"
    else:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = root / run_path
    run_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_ndcg = float("-inf")
    best_epoch: int | None = None
    if resume is not None:
        checkpoint = load_training_checkpoint(
            resume, actor, optimizer, scaler_arg,
            None if restart_schedule else scheduler,
            map_location=device,
        )
        global_step = int(checkpoint["step"])
        metadata = checkpoint.get("metadata", {})
        loaded_start_epoch = int(metadata.get("epoch", -1)) + 1
        if loaded_start_epoch != start_epoch:
            raise ValueError("resume checkpoint metadata changed while preparing SFT")
        best_ndcg = float(metadata.get("best_ndcg", float("-inf")))
        best_epoch = metadata.get("best_epoch")
        if restart_schedule:
            for group in optimizer.param_groups:
                group["lr"] = base_learning_rate
            scheduler = make_scheduler(optimizer_steps)

    manifest = {
        "run_id": run_path.name,
        "stage": "SFT",
        "seed": run_seed,
        "config_path": str(config.path),
        "config": config.raw,
        "source_sha256": source_tree_sha256(root / "src" / "tlab"),
        "dataset_sha256": None,
        "processed_listens_sha256": None,
        "catalog_sha256": None,
        "n_users": len(store.users),
        "raw_listen_rows": store.events,
        "sequence_tie_blocks": store.tie_blocks,
        "sequence_tied_events": store.tied_events,
        "eligible_sft_targets": sum(len(user.train_targets) for user in store.users),
        "positive_only": positive_only,
        "positive_sft_targets": int(sum(
            sum(user.ratio_raw[position] > 0.5 for position in user.train_targets)
            for user in store.users
        )),
        "target_budget_per_epoch": target_budget,
        "epoch_count": epoch_count,
        "resumed_from": str(Path(resume).resolve()) if resume is not None else None,
        "restart_schedule": restart_schedule,
        "optimizer_steps_planned": optimizer_steps,
        "precision": str(autocast_dtype or torch.float32),
        "catalog_size": len(catalog_item_ids),
        "actor_parameters": actor.parameter_count(),
        "environment": collect_environment(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if protocol_path.exists():
        manifest["processed_listens_sha256"] = json.loads(protocol_path.read_text(encoding="utf-8")).get("processed_listens_sha256")
        manifest["catalog_sha256"] = json.loads(protocol_path.read_text(encoding="utf-8")).get("catalog_sha256")
    data_manifest_path = root / "data" / "data_manifest.json"
    if data_manifest_path.exists():
        manifest["dataset_sha256"] = json.loads(data_manifest_path.read_text(encoding="utf-8")).get("files", {}).get("listens", {}).get("sha256")
    write_run_manifest(run_path / "run_manifest.json", manifest)

    metrics_path = run_path / "metrics.jsonl"
    writer = SummaryWriter(log_dir=str(run_path / "tensorboard"))
    patience = 2
    bad_epochs = 0
    started = time.perf_counter()
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, epoch_count):
            actor.train()
            epoch_start = time.perf_counter()
            windows = list(store.sample_sft_windows(
                max_targets=target_budget,
                history_length=history_length,
                n_positions=int(model_cfg.get("n_positions", 256)),
                seed=run_seed + epoch,
                positive_only=positive_only,
            ))
            if len(windows) != epoch_window_counts[epoch]:
                raise AssertionError("SFT window count changed between planning and training")
            random.Random(run_seed + epoch).shuffle(windows)
            accum: list[SFTBatch] = []
            loss_sum = 0.0
            target_sum = 0
            epoch_step_count = 0
            for offset in range(0, len(windows), microbatch_size):
                batch_windows = windows[offset:offset + microbatch_size]
                accum.append(collate_sequences(
                    [row[0] for row in batch_windows],
                    [row[1] for row in batch_windows],
                    device=device,
                ))
                if len(accum) < accumulation_steps and offset + microbatch_size < len(windows):
                    continue
                step_metrics = train_sft_accumulation_step(
                    actor,
                    accum,
                    optimizer,
                    logit_chunk_size=256,
                    scaler=scaler_arg,
                    gradient_clip=1.0,
                    autocast_dtype=autocast_dtype,
                )
                scheduler.step()
                global_step += 1
                epoch_step_count += 1
                loss_sum += step_metrics["loss"] * step_metrics["supervised_targets"]
                target_sum += int(step_metrics["supervised_targets"])
                writer.add_scalar("sft/train_loss", step_metrics["loss"], global_step)
                writer.add_scalar("sft/grad_norm", step_metrics["grad_norm"], global_step)
                accum.clear()

            validation = evaluate_full_catalog(
                actor=actor,
                store=store,
                catalog_item_ids=catalog_item_ids,
                split="validation",
                history_length=history_length,
                per_user_limit=validation_anchors_per_user,
                seed=run_seed,
                batch_size=16,
            )
            score = float(validation["macro_user_unconditional"].get("ndcg@10", float("nan")))
            if not torch.isfinite(torch.tensor(score)):
                raise ValueError("validation NDCG@10 is not finite; refusing to continue or select a checkpoint")
            improved = score > best_ndcg
            if improved:
                best_ndcg = score
                best_epoch = epoch
                bad_epochs = 0
                save_training_checkpoint(
                    run_path / "best.pt", actor, optimizer, global_step,
                    {"epoch": epoch, "best_ndcg": best_ndcg, "best_epoch": best_epoch,
                     "validation": validation, "sampler_seed": run_seed + epoch},
                    scaler=scaler_arg, scheduler=scheduler,
                    sampler_state={"epoch_seed": run_seed + epoch},
                )
            else:
                bad_epochs += 1
            save_training_checkpoint(
                run_path / "last.pt", actor, optimizer, global_step,
                {"epoch": epoch, "best_ndcg": best_ndcg, "best_epoch": best_epoch,
                 "validation": validation, "sampler_seed": run_seed + epoch},
                scaler=scaler_arg, scheduler=scheduler,
                sampler_state={"epoch_seed": run_seed + epoch},
            )
            epoch_metrics = {
                "stage": "SFT",
                "seed": run_seed,
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": loss_sum / max(1, target_sum),
                "train_targets": target_sum,
                "optimizer_steps": epoch_step_count,
                "validation": validation,
                "best_checkpoint": improved,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            metrics_file.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            metrics_file.flush()
            writer.add_scalar("sft/validation_ndcg10", score, global_step)
            print(json.dumps({key: value for key, value in epoch_metrics.items() if key != "validation"}, ensure_ascii=False))
            if bad_epochs >= patience:
                break
    writer.close()

    result = {
        "run_dir": str(run_path.resolve()),
        "best_checkpoint": str((run_path / "best.pt").resolve()),
        "best_validation_ndcg10": best_ndcg,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
    }
    manifest.update(result)
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_run_manifest(run_path / "run_manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train item-ID GPT-2 with exact chunked catalog CE")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--validation-anchors-per-user", type=int, default=10)
    parser.add_argument("--resume")
    parser.add_argument("--run-dir")
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--restart-schedule", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_sft(
        args.config,
        seed=args.seed,
        max_targets=args.max_targets,
        epochs=args.epochs,
        validation_anchors_per_user=args.validation_anchors_per_user,
        resume=args.resume,
        run_dir=args.run_dir,
        positive_only=args.positive_only,
        restart_schedule=args.restart_schedule,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
