from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import torch

from tlab.data.store import SequenceStore
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.train.checkpoint import save_training_checkpoint
from tlab.train.sft import SFTBatch, collate_sequences, train_sft_accumulation_step
from tlab.panel_tools.data import PanelTarget
from tlab.panel_tools.dense_sft import DenseWindowIndex, dense_accumulation_step
from tlab.panel_tools.evaluator import evaluate_actor_panel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_panel(path: Path) -> list[PanelTarget]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [PanelTarget(**row) for row in rows]


def _evaluate(actor: ItemGPT2Actor, store: SequenceStore, catalog: list[int], panel: list[PanelTarget]) -> dict[str, Any]:
    return evaluate_actor_panel(actor, store, catalog, panel, history_events=200, batch_size=16, k=10)


def _collate_sft_microbatch(index, store, selected_indices: list[int], batch_size: int, device: torch.device) -> list[SFTBatch]:
    result: list[SFTBatch] = []
    for start in range(0, len(selected_indices), batch_size):
        examples = [index.example(store, int(i)) for i in selected_indices[start:start + batch_size]]
        result.append(collate_sequences(
            [example[0] for example in examples], [example[1] for example in examples], device=device,
        ))
    return result


def _run_method(
    *,
    method: str,
    initial_state: dict[str, torch.Tensor],
    store: SequenceStore,
    index: Any,
    permutation,
    actor_shape: ActorShape,
    catalog: list[int],
    panel: list[PanelTarget],
    output_dir: Path,
    steps: int,
    physical_batch: int,
    effective_windows: int,
    logit_chunk_size: int,
    seed: int,
    learning_rate: float,
    initial_model_sha256: str,
    catalog_sha256: str,
) -> dict[str, Any]:
    checkpoint_path = output_dir / method / "last.pt"
    result_path = output_dir / method / "results.json"
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing panel_tools pilot artifacts under {output_dir / method}")
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1)
    actor = ItemGPT2Actor(actor_shape)
    actor.load_state_dict(initial_state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor.to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=learning_rate, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    autocast_dtype = torch.float16 if scaler is not None else None
    eval_rows: list[dict[str, Any]] = []
    evaluation_started = time.perf_counter()
    start_metrics = _evaluate(actor, store, catalog, panel)
    eval_rows.append({
        "optimizer_step": 0,
        "elapsed_training_seconds": 0.0,
        "ndcg@10": start_metrics["primary_user_macro"]["ndcg@10"],
        "recall@10": start_metrics["primary_user_macro"]["recall@10"],
        "target_coverage": start_metrics["target_coverage"],
        "per_user": start_metrics["per_user_primary"],
        "n_users": start_metrics["n_users"],
        "n_events": start_metrics["n_events"],
    })
    presentation_counts: list[int] = []
    step_seconds: list[float] = []
    for step in range(steps):
        selected = [int(value) for value in permutation[step * effective_windows:(step + 1) * effective_windows]]
        if len(selected) != effective_windows:
            raise ValueError("pilot sampler has fewer unique examples/windows than the requested update budget")
        step_started = time.perf_counter()
        batches = _collate_sft_microbatch(index, store, selected, physical_batch, device)
        targets_this_step = sum(int(batch.target_mask.sum().item()) for batch in batches)
        actor.train()
        if method == "dense_window_v1":
            metrics = dense_accumulation_step(
                actor, batches, optimizer, logit_chunk_size=logit_chunk_size,
                scaler=scaler, gradient_clip=1.0, autocast_dtype=autocast_dtype,
            )
        else:
            metrics = train_sft_accumulation_step(
                actor, batches, optimizer, logit_chunk_size=logit_chunk_size,
                scaler=scaler, gradient_clip=1.0, autocast_dtype=autocast_dtype,
            )
        if metrics["supervised_targets"] != float(targets_this_step):
            raise AssertionError("trainer and collator disagree about dense target presentation count")
        step_seconds.append(time.perf_counter() - step_started)
        presentation_counts.append(targets_this_step)
        completed = step + 1
        if completed in {max(1, steps // 2), steps}:
            elapsed_training = sum(step_seconds)
            actor.eval()
            validation = _evaluate(actor, store, catalog, panel)
            eval_rows.append({
                "optimizer_step": completed,
                "elapsed_training_seconds": elapsed_training,
                "ndcg@10": validation["primary_user_macro"]["ndcg@10"],
                "recall@10": validation["primary_user_macro"]["recall@10"],
                "target_coverage": validation["target_coverage"],
                "per_user": validation["per_user_primary"],
                "n_users": validation["n_users"],
                "n_events": validation["n_events"],
            })
    training_seconds = sum(step_seconds)
    total_wall = time.perf_counter() - evaluation_started
    last_step = eval_rows[-1]["optimizer_step"]
    if last_step != steps:
        actor.eval()
        validation = _evaluate(actor, store, catalog, panel)
        eval_rows.append({
            "optimizer_step": steps,
            "elapsed_training_seconds": training_seconds,
            "ndcg@10": validation["primary_user_macro"]["ndcg@10"],
            "recall@10": validation["primary_user_macro"]["recall@10"],
            "target_coverage": validation["target_coverage"],
            "per_user": validation["per_user_primary"],
            "n_users": validation["n_users"],
            "n_events": validation["n_events"],
        })
    actor.eval()
    metadata = {
        "schema_version": 1,
        "protocol_id": "tlab_yambda_policy_diagnostics026-10-21_r1",
        "method": method,
        "optimizer_step": steps,
        "run_kind": "short validation pilot; not a selected final checkpoint",
        "sampler_version": getattr(index, "sampler_version", "sft_target_index_reference"),
        "catalog_sha256": catalog_sha256,
        "initial_model_sha256": initial_model_sha256,
        "seed": seed,
        "learning_rate": learning_rate,
        "physical_batch_windows": physical_batch,
        "effective_windows": effective_windows,
        "logit_chunk_size": logit_chunk_size,
        "precision": "fp16 autocast + GradScaler" if scaler is not None else "fp32",
        "history_contract": "at most 200 strictly prior events plus BOS; dense windows share a causal prefix",
        "target_presentation_count": int(sum(presentation_counts)),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_training_checkpoint(
        checkpoint_path, actor, optimizer, steps, metadata, scaler=scaler,
        sampler_state={"selected_indices": selected if steps else [], "permutation_seed": seed,
                       "cursor": steps * effective_windows, "budget_steps": steps},
    )
    result = {
        "method": method,
        "status": "completed_short_pilot",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_seconds_excluding_validation": training_seconds,
        "validation_and_total_wall_seconds": total_wall,
        "optimizer_steps": steps,
        "windows_per_optimizer_step": effective_windows,
        "windows_seen": steps * effective_windows,
        "target_presentations": int(sum(presentation_counts)),
        "target_presentations_per_step": presentation_counts,
        "step_seconds_median": float(sorted(step_seconds)[len(step_seconds) // 2]) if step_seconds else None,
        "windows_per_second_measured": steps * effective_windows / training_seconds if training_seconds else None,
        "targets_per_second_measured": sum(presentation_counts) / training_seconds if training_seconds else None,
        "evaluations": eval_rows,
        "checkpoint_metadata": metadata,
    }
    output_dir.joinpath(method).mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run_dense_sft_pilot(
    workspace: str | Path,
    *,
    run_name: str = "s1_s2_seed17_20steps",
    steps: int = 20,
    seed: int = 17,
    physical_batch: int = 2,
    effective_windows: int = 8,
    learning_rate: float = 3e-4,
    logit_chunk_size: int = 32,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    protocol = json.loads((root / "artifacts/panel_tools/protocol.json").read_text(encoding="utf-8"))
    processed = root / "data/processed_full_model_tminus14"
    expected_hash = protocol["dataset"]["processed_full_model_tminus14_listens_sha256"]
    load_started = time.perf_counter()
    store = SequenceStore.load_cache(processed / "sequence_index", expected_source_sha256=expected_hash)
    data_load_seconds = time.perf_counter() - load_started
    catalog_frame = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    catalog = [int(value) for value in catalog_frame.get_column("item_id").to_list()]
    catalog_sha256 = _sha256(processed / "catalog.parquet")
    if len(catalog) != 20_000:
        raise ValueError("S1/S2 pilot is pinned to the full_model 20k bridge catalog")
    panel_path = root / "artifacts/panel_tools/evaluation/full_model_sft_weak_seed17_validation_panel.jsonl"
    if not panel_path.exists():
        raise FileNotFoundError("freeze and evaluate the stage-1 validation panel before running SFT")
    panel = _load_panel(panel_path)
    actor_shape = ActorShape(vocab_size=len(catalog) + 3, n_positions=256, n_layer=4, n_embd=256, n_head=4, dropout=0.1)

    dense_started = time.perf_counter()
    dense_index = DenseWindowIndex.from_store(
        store, window_events=64, context_tokens_including_bos=138,
    )
    dense_index_seconds = time.perf_counter() - dense_started
    total_train_targets = sum(len(user.train_targets) for user in store.users)
    if not len(dense_index):
        raise ValueError("panel_tools dense sampler found no training windows")
    sft_index = store.sample_sft_target_index(
        max_targets=max(50_000, steps * effective_windows), history_events=200, seed=seed,
    )
    if len(sft_index) < steps * effective_windows:
        raise ValueError("legacy SFT control does not have enough distinct train targets for the pilot")
    dense_permutation = dense_index.permutation(seed)
    sft_permutation = sft_index.permutation(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    initial_actor = ItemGPT2Actor(actor_shape)
    initial_state = {name: parameter.detach().cpu().clone() for name, parameter in initial_actor.state_dict().items()}
    initial_hash_digest = hashlib.sha256()
    for name in sorted(initial_state):
        initial_hash_digest.update(name.encode("utf-8"))
        initial_hash_digest.update(initial_state[name].numpy().tobytes())
    initial_state_sha = initial_hash_digest.hexdigest()

    output_dir = root / "artifacts/panel_tools/runs" / run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing panel_tools pilot run: {output_dir}")
    output_dir.mkdir(parents=True)
    dense_result = _run_method(
        method="dense_window_v1", initial_state=initial_state, store=store, index=dense_index,
        permutation=dense_permutation, actor_shape=actor_shape, catalog=catalog, panel=panel,
        output_dir=output_dir, steps=steps, physical_batch=physical_batch,
        effective_windows=effective_windows, logit_chunk_size=logit_chunk_size,
        seed=seed, learning_rate=learning_rate, initial_model_sha256=initial_state_sha,
        catalog_sha256=catalog_sha256,
    )
    legacy_result = _run_method(
        method="last_target_control", initial_state=initial_state, store=store, index=sft_index,
        permutation=sft_permutation, actor_shape=actor_shape, catalog=catalog, panel=panel,
        output_dir=output_dir, steps=steps, physical_batch=physical_batch,
        effective_windows=effective_windows, logit_chunk_size=logit_chunk_size,
        seed=seed, learning_rate=learning_rate, initial_model_sha256=initial_state_sha,
        catalog_sha256=catalog_sha256,
    )
    dense_result["checkpoint_metadata"]["initial_model_sha256"] = initial_state_sha
    legacy_result["checkpoint_metadata"]["initial_model_sha256"] = initial_state_sha
    result = {
        "protocol_id": protocol["protocol_id"],
        "run_name": run_name,
        "status": "completed_short_validation_pilot",
        "data": {
            "processed_listens_sha256": expected_hash,
            "catalog_size": len(catalog),
            "catalog_file_sha256": _sha256(processed / "catalog.parquet"),
            "load_seconds": data_load_seconds,
            "train_targets": total_train_targets,
            "dense_windows": len(dense_index),
            "dense_window_index_build_seconds": dense_index_seconds,
            "mean_targets_per_dense_window": total_train_targets / len(dense_index),
            "panel_path": str(panel_path),
            "panel_event_ids_sha256": hashlib.sha256("\n".join(row.event_id for row in panel).encode()).hexdigest(),
        },
        "budget": {
            "planning_ceiling_gpu_hours": 24,
            "measured_full_dense_epoch_gpu_hours_at_dense_observed_rate": (
                len(dense_index) / dense_result["windows_per_second_measured"] / 3600
                if dense_result["windows_per_second_measured"] else None
            ),
            "method": "one short actual-data pilot; projection includes sampler/collation, but is not a completed epoch",
            "planned_optimizer_steps": steps,
            "physical_batch_windows": physical_batch,
            "effective_windows": effective_windows,
        },
        "initial_model_sha256": initial_state_sha,
        "methods": {"dense": dense_result, "last_target_control": legacy_result},
        "comparison_boundary": [
            "Equal optimizer updates and window batch size are the pilot budget; valid target presentations differ by design.",
            "The short pilot does not establish converged SFT quality or a full training recipe.",
            "All validation values are retrospective development signals; no test panel was opened.",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result_path = output_dir / "pilot_comparison.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
