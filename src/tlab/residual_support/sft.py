from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import yaml

from tlab.data.store import SFTTargetIndex, SequenceStore
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.train.checkpoint import load_training_checkpoint, save_training_checkpoint
from tlab.train.sft import collate_sequences, train_sft_accumulation_step
from tlab.panel_tools.data import PanelTarget, history_for_target
from tlab.panel_tools.dense_sft import DenseWindowIndex, dense_accumulation_step
from tlab.panel_tools.evaluator import _collate_histories, evaluate_actor_panel
from tlab.residual_support.rm import sha256_file


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_panel(path: Path) -> list[PanelTarget]:
    return [PanelTarget(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _store(config: dict[str, Any], root: Path) -> tuple[SequenceStore, Path]:
    processed = _resolve(root, config["data"]["processed_dir"])
    protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(
        _resolve(root, config["data"]["sequence_index"]),
        expected_source_sha256=protocol["processed_listens_sha256"],
    )
    return store, processed


def _sampler_signature(method: str, sampler: DenseWindowIndex | SFTTargetIndex, seed: int) -> str:
    payload = {
        "method": method,
        "length": len(sampler),
        "seed": seed,
        "history": getattr(sampler, "context_tokens_including_bos", getattr(sampler, "history_events", None)),
        "window_events": getattr(sampler, "window_events", None),
        "sampler_version": getattr(sampler, "sampler_version", "last_target_v1"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _coverage_at_cursor(sampler: DenseWindowIndex | SFTTargetIndex, method: str,
                        order: np.ndarray, epoch: int, cursor: int,
                        store: SequenceStore) -> tuple[int, int]:
    if epoch > 0:
        return len(sampler), sum(len(user.train_targets) for user in store.users)
    if method == "last_target":
        return cursor, cursor
    target_count = sum(int(sampler.target_stops[int(i)] - sampler.target_starts[int(i)])
                       for i in order[:cursor])
    return cursor, target_count


def _new_actor(shape: ActorShape, initial_state: dict[str, torch.Tensor], device: torch.device) -> ItemGPT2Actor:
    actor = ItemGPT2Actor(shape)
    actor.load_state_dict(initial_state, strict=True)
    return actor.to(device)


def _fixed_train_ce(actor: ItemGPT2Actor, store: SequenceStore, sampler: DenseWindowIndex | SFTTargetIndex,
                    *, method: str, seed: int, device: torch.device, max_targets: int,
                    chunk_size: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(sampler), dtype=np.int64)
    rng.shuffle(order)
    examples: list[tuple[list[int], list[bool]]] = []
    total_targets = 0
    for sampler_index in order.tolist():
        example = sampler.example(store, sampler_index)
        count = sum(example[1])
        if not count:
            continue
        examples.append(example)
        total_targets += count
        if total_targets >= max_targets:
            break
    if not examples:
        raise ValueError("training CE sample is empty")
    target_budget = min(total_targets, max_targets)
    ce_sum = 0.0
    seen_targets = 0
    was_training = actor.training
    actor.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), 4):
            batch = collate_sequences([row[0] for row in examples[start:start + 4]],
                                      [row[1] for row in examples[start:start + 4]], device=device)
            hidden = actor.encode(batch.input_ids, batch.attention_mask)
            selected_hidden = hidden[batch.target_mask]
            selected_targets = batch.targets[batch.target_mask]
            for lo in range(0, len(selected_targets), chunk_size):
                hi = min(lo + chunk_size, len(selected_targets))
                logits = actor.action_logits(selected_hidden[lo:hi])
                losses = torch.nn.functional.cross_entropy(logits, selected_targets[lo:hi], reduction="none")
                remain = max(0, target_budget - seen_targets)
                take = min(remain, len(losses))
                if take:
                    ce_sum += float(losses[:take].sum().item())
                    seen_targets += take
                if seen_targets >= target_budget:
                    break
            if seen_targets >= target_budget:
                break
    actor.train(was_training)
    return {"cross_entropy": ce_sum / max(1, seen_targets), "targets": float(seen_targets)}


def _development_ce(actor: ItemGPT2Actor, store: SequenceStore, panel: list[PanelTarget],
                    *, device: torch.device, batch_size: int, chunk_size: int) -> dict[str, float]:
    loss_sum = 0.0
    n_targets = 0
    was_training = actor.training
    actor.eval()
    with torch.inference_mode():
        for start in range(0, len(panel), batch_size):
            targets = panel[start:start + batch_size]
            histories = [history_for_target(store, target, 200) for target in targets]
            input_ids, attention = _collate_histories(histories, device)
            hidden = actor.encode(input_ids, attention)
            labels = torch.as_tensor([row.token_id for row in targets], dtype=torch.long, device=device)
            valid = labels >= 3
            positions = attention.long().sum(dim=-1).clamp_min(1) - 1
            last_hidden = hidden[torch.arange(hidden.shape[0], device=device), positions]
            selected_hidden, selected_labels = last_hidden[valid], labels[valid]
            for lo in range(0, len(selected_labels), chunk_size):
                hi = min(lo + chunk_size, len(selected_labels))
                logits = actor.action_logits(selected_hidden[lo:hi])
                loss_sum += float(torch.nn.functional.cross_entropy(logits, selected_labels[lo:hi], reduction="sum").item())
                n_targets += hi - lo
    actor.train(was_training)
    return {"cross_entropy": loss_sum / max(1, n_targets), "known_catalog_targets": float(n_targets),
            "oov_targets": float(len(panel) - n_targets)}


def _evaluate(actor: ItemGPT2Actor, store: SequenceStore, catalog_ids: list[int], panel: list[PanelTarget],
              train_ce: dict[str, float], *, device: torch.device, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    was_training = actor.training
    development_ce = _development_ce(actor, store, panel, device=device,
                                     batch_size=int(config["evaluation"]["batch_size"]),
                                     chunk_size=int(config["training"]["logit_chunk_size"]))
    ranking = evaluate_actor_panel(actor, store, catalog_ids, panel, history_events=200,
                                   batch_size=int(config["evaluation"]["batch_size"]), k=10)
    actor.train(was_training)
    return {"train_sample": train_ce,
            "development_ce": development_ce,
            "development_ranking": ranking["primary_user_macro"],
            "development_panel_events": len(panel),
            "evaluation_seconds": time.perf_counter() - started}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def prepare(config_path: str | Path, root: str | Path = ".") -> dict[str, Any]:
    config_path, project_root = Path(config_path).resolve(), Path(root).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    store, processed = _store(cfg, project_root)
    catalog = pl.read_parquet(_resolve(project_root, cfg["data"]["catalog"])).sort("token_id")
    item_ids = catalog["item_id"].to_list()
    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    shape = ActorShape(vocab_size=len(item_ids) + 3, n_positions=256,
                       n_layer=int(cfg["model"]["n_layer"]), n_embd=int(cfg["model"]["n_embd"]),
                       n_head=int(cfg["model"]["n_head"]), dropout=float(cfg["model"]["dropout"]))
    actor = ItemGPT2Actor(shape)
    output = _resolve(project_root, cfg["outputs"]["root"])
    allowed = (project_root / "artifacts/residual_support").resolve()
    if allowed not in output.resolve().parents and output.resolve() != allowed:
        raise ValueError("SFT outputs must remain below artifacts/residual_support")
    output.mkdir(parents=True, exist_ok=True)
    initial_path = output / "initial_actor.pt"
    if initial_path.exists():
        raise FileExistsError(f"refusing to overwrite {initial_path}")
    torch.save({"model": actor.state_dict(), "metadata": {"seed": seed, "shape": shape.__dict__}}, initial_path)
    dense = DenseWindowIndex.from_store(store,
        window_events=int(cfg["samplers"]["window_events"]),
        context_tokens_including_bos=int(cfg["samplers"]["context_tokens_including_bos"]))
    last = store.sample_sft_target_index(max_targets=sum(len(u.train_targets) for u in store.users),
                                         history_events=int(cfg["samplers"]["history_events"]), seed=seed)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    record = {
        "schema_version": 1, "status": "prepared", "protocol_id": cfg["protocol_id"],
        "source_protocol_sha256": sha256_file(processed / "protocol.json"),
        "catalog_sha256": sha256_file(_resolve(project_root, cfg["data"]["catalog"])),
        "config_sha256": config_hash, "initial_actor_path": str(initial_path.relative_to(project_root)),
        "initial_actor_sha256": sha256_file(initial_path), "catalog_items": len(item_ids),
        "train_target_events": len(last), "dense_windows": len(dense),
        "dense_targets_by_window": sum(len(u.train_targets) for u in store.users) / max(1, len(dense)),
        "train_users": len(store.users), "model_shape": shape.__dict__,
        "development_panel_path": str(_resolve(project_root, cfg["data"]["development_panel"]).relative_to(project_root)),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "preparation.json", record)
    return record


def _make_amp(device: torch.device, config: dict[str, Any]) -> tuple[torch.dtype | None, Any | None]:
    if device.type != "cuda" or not bool(config["training"]["amp"]):
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda", enabled=True)


def _make_sampler(store: SequenceStore, method: str, config: dict[str, Any]) -> DenseWindowIndex | SFTTargetIndex:
    if method == "dense":
        return DenseWindowIndex.from_store(store,
            window_events=int(config["samplers"]["window_events"]),
            context_tokens_including_bos=int(config["samplers"]["context_tokens_including_bos"]))
    if method == "last_target":
        return store.sample_sft_target_index(max_targets=sum(len(u.train_targets) for u in store.users),
                                             history_events=int(config["samplers"]["history_events"]),
                                             seed=int(config["training"]["seed"]))
    raise ValueError(method)


def fit(config_path: str | Path, method: str, root: str | Path = ".", resume: bool = False) -> dict[str, Any]:
    config_path, project_root = Path(config_path).resolve(), Path(root).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_root = _resolve(project_root, cfg["outputs"]["root"])
    prep = json.loads((out_root / "preparation.json").read_text(encoding="utf-8"))
    initial_path = out_root / "initial_actor.pt"
    if sha256_file(initial_path) != prep["initial_actor_sha256"]:
        raise ValueError("shared initial actor changed after preparation")
    payload = torch.load(initial_path, map_location="cpu", weights_only=False)
    store, _ = _store(cfg, project_root)
    sampler = _make_sampler(store, method, cfg)
    seed = int(cfg["training"]["seed"])
    signature = _sampler_signature(method, sampler, seed)
    run_dir = out_root / "runs" / method
    if run_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    seen_path = run_dir / "seen_sampler_indices.bin"
    seen = np.memmap(seen_path, dtype=np.uint8, mode="r+" if seen_path.exists() else "w+", shape=(len(sampler),))
    if not seen_path.exists():
        seen[:] = 0
        seen.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = ActorShape(vocab_size=int(prep["catalog_items"]) + 3, n_positions=256,
                       n_layer=int(cfg["model"]["n_layer"]), n_embd=int(cfg["model"]["n_embd"]),
                       n_head=int(cfg["model"]["n_head"]), dropout=float(cfg["model"]["dropout"]))
    actor = _new_actor(shape, payload["model"], device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=float(cfg["training"]["learning_rate"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    amp_dtype, grad_scaler = _make_amp(device, cfg)
    run_manifest_path = run_dir / "run_manifest.json"
    curve_path = run_dir / "curve.jsonl"
    if resume:
        if not checkpoint_path.exists() or not run_manifest_path.exists():
            raise FileNotFoundError("resume requested but checkpoint or run manifest is missing")
        saved = load_training_checkpoint(checkpoint_path, actor, optimizer, grad_scaler,
                                         map_location=device)
        state = saved["sampler_state"]
        if state.get("signature") != signature or int(state.get("sampler_length", -1)) != len(sampler):
            raise ValueError("SFT sampler identity changed across resume")
        epoch, cursor, updates, unique_targets = (int(state["epoch"]), int(state["cursor"]),
                                                  int(saved["step"]), int(state["unique_targets"]))
        unique_sampler_indices = int(state.get("unique_sampler_indices", 0))
        if curve_path.exists():
            committed_rows = []
            for line in curve_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") == "resume" or int(record.get("step", 0)) <= updates:
                    committed_rows.append(json.dumps(record, ensure_ascii=False))
            curve_path.write_text("\n".join(committed_rows) + ("\n" if committed_rows else ""), encoding="utf-8")
            with curve_path.open("a", encoding="utf-8") as curve:
                curve.write(json.dumps({"event": "resume", "step": updates, "epoch": epoch, "cursor": cursor,
                                        "resumed_at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("preparation_config_sha256", prep["config_sha256"])
        manifest["execution_config_sha256"] = sha256_file(config_path)
        manifest["execution_config_update"] = "Added held-out CE and ranking checks every 900 training seconds; optimizer, sampler, and time budget unchanged."
    else:
        epoch, cursor, updates, unique_targets, unique_sampler_indices = 0, 0, 0, 0, 0
        manifest = {"schema_version": 1, "method": method, "status": "running",
                    "protocol_id": cfg["protocol_id"], "sampler_signature": signature,
                    "sampler_length": len(sampler), "initial_actor_sha256": prep["initial_actor_sha256"],
                    "config_sha256": prep["config_sha256"], "execution_config_sha256": sha256_file(config_path),
                    "device": str(device),
                    "started_at_utc": datetime.now(timezone.utc).isoformat(), "updates_completed": 0}
        _write_json(run_manifest_path, manifest)

    development = _load_panel(_resolve(project_root, cfg["data"]["development_panel"]))
    catalog = pl.read_parquet(_resolve(project_root, cfg["data"]["catalog"])).sort("token_id")
    catalog_ids = catalog["item_id"].to_list()
    train_ce = _fixed_train_ce(actor, store, sampler, method=method, seed=seed + 900,
                               device=device, max_targets=int(cfg["evaluation"]["train_ce_targets"]),
                               chunk_size=int(cfg["training"]["logit_chunk_size"]))
    if not resume:
        manifest["initial_evaluation"] = _evaluate(actor, store, catalog_ids, development, train_ce,
                                                   device=device, config=cfg)
        _write_json(run_manifest_path, manifest)

    wall_budget = float(cfg["training"]["fit_time_budget_seconds"])
    training_seconds = float(manifest.get("training_seconds", 0.0))
    benchmark_steps = int(cfg["training"]["benchmark_steps"])
    benchmark_done = int(manifest.get("benchmark_steps_completed", 0)) >= benchmark_steps
    evaluation_interval = float(cfg["training"]["evaluation_every_seconds"])
    if evaluation_interval <= 0:
        raise ValueError("evaluation_every_seconds must be positive")
    next_evaluation_seconds = float(manifest.get(
        "next_evaluation_seconds",
        (math.floor(training_seconds / evaluation_interval) + 1) * evaluation_interval,
    ))
    batch_size = int(cfg["training"]["physical_batch_size"])
    accumulation = int(cfg["training"]["accumulation_microbatches"])
    sampler_order = sampler.permutation(seed + epoch)
    if resume:
        seen[:] = 0
        coverage, target_coverage = _coverage_at_cursor(sampler, method, sampler_order, epoch, cursor, store)
        seen[sampler_order[:coverage]] = 1
        seen.flush()
        unique_sampler_indices, unique_targets = coverage, target_coverage
    if cursor > len(sampler_order):
        raise ValueError("saved sampler cursor exceeds epoch length")
    start_clock = time.perf_counter()
    updates_at_start = updates
    last_checkpoint_time = start_clock
    amp_skips = int(manifest.get("amp_skipped_updates", 0))
    benchmark_elapsed = float(manifest.get("benchmark_seconds", 0.0))
    while training_seconds < wall_budget:
        if cursor >= len(sampler_order):
            epoch += 1
            cursor = 0
            sampler_order = sampler.permutation(seed + epoch)
        available_microbatches = min(accumulation, math.ceil((len(sampler_order) - cursor) / batch_size))
        microbatches = []
        index_groups = []
        context_lengths: list[int] = []
        for _ in range(available_microbatches):
            selected = sampler_order[cursor:min(cursor + batch_size, len(sampler_order))]
            cursor += len(selected)
            index_groups.append(selected)
            examples = [sampler.example(store, int(index)) for index in selected]
            context_lengths.extend(max(0, next(i for i, valid in enumerate(mask) if valid) - 1)
                                   for _, mask in examples)
            microbatches.append(collate_sequences([row[0] for row in examples], [row[1] for row in examples], device=device))
        new_indices = [int(index) for group in index_groups for index in group if seen[int(index)] == 0]
        before_seen = len(new_indices)
        base_lr = float(cfg["training"]["learning_rate"])
        warmup = max(1.0, 0.05 * wall_budget)
        if training_seconds < warmup:
            lr_factor = max(0.01, training_seconds / warmup)
        else:
            lr_factor = max(0.0, (wall_budget - training_seconds) / max(1.0, wall_budget - warmup))
        for group in optimizer.param_groups:
            group["lr"] = base_lr * lr_factor
        if method == "dense":
            result = dense_accumulation_step(actor, microbatches, optimizer,
                    logit_chunk_size=int(cfg["training"]["logit_chunk_size"]), scaler=grad_scaler,
                    gradient_clip=float(cfg["training"]["gradient_clip"]), autocast_dtype=amp_dtype)
        else:
            result = train_sft_accumulation_step(actor, microbatches, optimizer,
                    logit_chunk_size=int(cfg["training"]["logit_chunk_size"]), scaler=grad_scaler,
                    gradient_clip=float(cfg["training"]["gradient_clip"]), autocast_dtype=amp_dtype)
        for group in index_groups:
            seen[group] = 1
        seen.flush()
        unique_sampler_indices += before_seen
        unique_targets += before_seen if method == "last_target" else sum(
            int(sampler.target_stops[index] - sampler.target_starts[index]) for index in new_indices
        )
        updates += 1
        elapsed = time.perf_counter() - start_clock
        training_seconds += elapsed
        start_clock = time.perf_counter()
        if grad_scaler is not None and float(grad_scaler.get_scale()) < float(manifest.get("amp_scale", grad_scaler.get_scale())):
            amp_skips += 1
        manifest["amp_scale"] = float(grad_scaler.get_scale()) if grad_scaler is not None else None
        if not benchmark_done and updates - updates_at_start <= benchmark_steps:
            benchmark_elapsed += elapsed
        curve_record = {"step": updates, "epoch": epoch, "cursor": cursor,
                        "train_ce_batch": float(result["loss"]), "supervised_targets": int(result["supervised_targets"]),
                        "unique_sampler_indices": unique_sampler_indices, "unique_target_events": unique_targets,
                        "target_events_per_second_cumulative": unique_targets / max(training_seconds, 1e-9),
                        "training_seconds": training_seconds, "elapsed_seconds_update": elapsed,
                        "optimizer_steps": updates, "scheduler_steps": updates,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "scheduler": "linear_warmup_5pct_then_linear_decay_by_measured_training_budget",
                        "amp_scale": manifest["amp_scale"], "amp_skipped_updates": amp_skips,
                        "mean_sequence_tokens": float(np.mean([int(batch.attention_mask.sum(dim=1).float().mean().item())
                                                                for batch in microbatches])),
                        "mean_history_events": float(np.mean(context_lengths)) if context_lengths else 0.0}
        with curve_path.open("a", encoding="utf-8") as curve:
            curve.write(json.dumps(curve_record, ensure_ascii=False) + "\n")
        should_evaluate = training_seconds >= next_evaluation_seconds
        if should_evaluate:
            train_ce_at_eval = _fixed_train_ce(actor, store, sampler, method=method, seed=seed + 900,
                device=device, max_targets=int(cfg["evaluation"]["train_ce_targets"]),
                chunk_size=int(cfg["training"]["logit_chunk_size"]))
            evaluation = _evaluate(actor, store, catalog_ids, development, train_ce_at_eval,
                                   device=device, config=cfg)
            evaluation.update({"event": "periodic_evaluation", "step": updates,
                               "training_seconds": training_seconds})
            with curve_path.open("a", encoding="utf-8") as curve:
                curve.write(json.dumps(evaluation, ensure_ascii=False, allow_nan=False) + "\n")
            actor.train()
            next_evaluation_seconds += evaluation_interval
            start_clock = time.perf_counter()
        checkpoint_due = (updates % int(cfg["training"]["checkpoint_every_updates"]) == 0
                          or training_seconds >= wall_budget or should_evaluate)
        if checkpoint_due:
            sampler_state = {"signature": signature, "sampler_length": len(sampler), "epoch": epoch,
                             "cursor": cursor, "unique_targets": unique_targets,
                             "unique_sampler_indices": unique_sampler_indices}
            save_training_checkpoint(checkpoint_path, actor, optimizer, updates,
                {"method": method, "training_seconds": training_seconds, "protocol_id": cfg["protocol_id"]},
                scaler=grad_scaler, sampler_state=sampler_state)
            manifest.update({"updates_completed": updates, "training_seconds": training_seconds,
                             "epoch": epoch, "cursor": cursor, "unique_target_events": unique_targets,
                             "amp_skipped_updates": amp_skips, "last_checkpoint_sha256": sha256_file(checkpoint_path),
                             "last_checkpoint_step": updates, "benchmark_seconds": benchmark_elapsed,
                             "next_evaluation_seconds": next_evaluation_seconds,
                             "benchmark_steps_completed": min(benchmark_steps, updates - updates_at_start)})
            _write_json(run_manifest_path, manifest)
            last_checkpoint_time = time.perf_counter()
        if not benchmark_done and updates - updates_at_start == benchmark_steps:
            samples = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            recent = [row for row in samples if row.get("step", 0) > updates_at_start]
            throughput = sum(row["supervised_targets"] for row in recent[-benchmark_steps:]) / max(benchmark_elapsed, 1e-9)
            manifest.update({"benchmark_steps_completed": benchmark_steps, "benchmark_seconds": benchmark_elapsed,
                             "benchmark_supervised_targets_per_second": throughput,
                             "projected_sampler_pass_seconds": sum(len(u.train_targets) for u in store.users) /
                                max(throughput, 1e-9)})
            _write_json(run_manifest_path, manifest)
            benchmark_done = True

    final_train_ce = _fixed_train_ce(actor, store, sampler, method=method, seed=seed + 900,
                                     device=device, max_targets=int(cfg["evaluation"]["train_ce_targets"]),
                                     chunk_size=int(cfg["training"]["logit_chunk_size"]))
    final_eval = _evaluate(actor, store, catalog_ids, development, final_train_ce, device=device, config=cfg)
    sampler_state = {"signature": signature, "sampler_length": len(sampler), "epoch": epoch,
                     "cursor": cursor, "unique_targets": unique_targets,
                     "unique_sampler_indices": unique_sampler_indices}
    save_training_checkpoint(checkpoint_path, actor, optimizer, updates,
        {"method": method, "training_seconds": training_seconds, "protocol_id": cfg["protocol_id"]},
        scaler=grad_scaler, sampler_state=sampler_state)
    manifest.update({"status": "completed_time_budget", "updates_completed": updates,
                     "training_seconds": training_seconds, "wall_budget_seconds": wall_budget,
                     "epoch": epoch, "cursor": cursor, "unique_target_events": unique_targets,
                     "unique_sampler_indices": unique_sampler_indices, "sampler_length": len(sampler),
                     "mean_context_events": (sum(min(200, int(position)) for user in store.users
                         for position in user.train_targets) / max(1, sum(len(user.train_targets) for user in store.users)))
                         if method == "last_target" else None,
                     "amp_skipped_updates": amp_skips, "final_evaluation": final_eval,
                     "last_checkpoint_sha256": sha256_file(checkpoint_path),
                     "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    _write_json(run_manifest_path, manifest)
    return manifest


def summarize(config_path: str | Path, root: str | Path = ".") -> dict[str, Any]:
    project_root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = _resolve(project_root, cfg["outputs"]["root"])
    runs = {name: json.loads((out / "runs" / name / "run_manifest.json").read_text(encoding="utf-8"))
            for name in ("dense", "last_target")}
    presentations: dict[str, int] = {}
    for name in runs:
        curve_path = out / "runs" / name / "curve.jsonl"
        presentations[name] = sum(
            int(row.get("supervised_targets", 0))
            for line in curve_path.read_text(encoding="utf-8").splitlines()
            if line.strip() for row in [json.loads(line)]
        )
    result = {"schema_version": 1, "status": "matched_sft_comparison",
              "comparison_basis": "same initialization, development panel, optimizer and fixed training wall-time budget",
              "training_wall_time_budget_seconds_each": cfg["training"]["fit_time_budget_seconds"],
              "initial_actor_sha256": runs["dense"]["initial_actor_sha256"],
              "runs": {name: {"updates": row["updates_completed"], "training_seconds": row["training_seconds"],
                              "unique_target_events": row["unique_target_events"],
                              "supervised_target_presentations": presentations[name],
                              "repeat_target_presentations_after_first_sampler_pass": max(
                                  0, presentations[name] - row["unique_target_events"]),
                              "unique_sampler_indices": row["unique_sampler_indices"],
                              "sampler_length": row["sampler_length"],
                              "initial_evaluation": row.get("initial_evaluation"),
                              "final_evaluation": row["final_evaluation"],
                              "amp_skipped_updates": row["amp_skipped_updates"]} for name,row in runs.items()},
              "interpretation": "Dense and last-target see different target presentations per update; comparisons are equal wall time. Report target throughput, coverage, CE and ranking together.",
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(out / "matched_sft_comparison.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="panel_tools.1 matched-budget dense and last-target SFT experiments")
    parser.add_argument("--config", default="configs/residual_support/sft.yaml")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--method", choices=["dense", "last_target"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare:
        result = prepare(args.config, args.workspace)
    elif args.summarize:
        result = summarize(args.config, args.workspace)
    elif args.method:
        result = fit(args.config, args.method, args.workspace, args.resume)
    else:
        parser.error("choose --prepare, --method, or --summarize")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
