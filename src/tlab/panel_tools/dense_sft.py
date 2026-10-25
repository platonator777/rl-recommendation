from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from tlab.data.catalog import BOS_ID
from tlab.data.store import SequenceStore
from tlab.models.actor import ItemGPT2Actor
from tlab.train.sft import SFTBatch, collate_sequences


DENSE_SAMPLER_VERSION = "dense_window_v1"
DENSE_CHECKPOINT_SCHEMA = 1


@dataclass
class DenseWindowIndex:
    user_indices: array
    starts: array
    stops: array
    target_starts: array
    target_stops: array
    window_events: int
    context_tokens_including_bos: int
    sampler_version: str = DENSE_SAMPLER_VERSION

    def __len__(self) -> int:
        return len(self.starts)

    def permutation(self, seed: int):
        import numpy as np

        order = np.arange(len(self), dtype=np.int64)
        np.random.default_rng(int(seed)).shuffle(order)
        return order

    @classmethod
    def from_store(
        cls,
        store: SequenceStore,
        *,
        window_events: int = 64,
        context_tokens_including_bos: int = 192,
    ) -> "DenseWindowIndex":
        if window_events < 1 or context_tokens_including_bos < 1:
            raise ValueError("dense window and context lengths must be positive")
        if context_tokens_including_bos + window_events > 202:
            raise ValueError("dense target prefixes would exceed the protocol's 200 preceding events")
        user_indices = array("I")
        starts = array("I")
        stops = array("I")
        target_starts = array("I")
        target_stops = array("I")
        for user_index, user in enumerate(store.users):
            targets = user.train_targets
            if len(targets) == 0:
                continue
            start = int(targets[0])
            last = start
            count = 1
            target_start = 0
            for target_offset, target_raw in enumerate(targets[1:], start=1):
                target = int(target_raw)
                if target - start >= window_events or count >= window_events:
                    user_indices.append(user_index)
                    starts.append(start)
                    stops.append(last + 1)
                    target_starts.append(target_start)
                    target_stops.append(target_offset)
                    start = target
                    count = 0
                    target_start = target_offset
                last = target
                count += 1
            user_indices.append(user_index)
            starts.append(start)
            stops.append(last + 1)
            target_starts.append(target_start)
            target_stops.append(len(targets))
        return cls(user_indices, starts, stops, target_starts, target_stops, window_events, context_tokens_including_bos)

    def example(self, store: SequenceStore, index: int) -> tuple[list[int], list[bool]]:
        user_index = int(self.user_indices[index])
        start = int(self.starts[index])
        stop = int(self.stops[index])
        user = store.users[user_index]
        history = user.history(start, self.context_tokens_including_bos)
        target_start = int(self.target_starts[index])
        target_stop = int(self.target_stops[index])
        target_positions = set(int(p) for p in user.train_targets[target_start:target_stop])
        event_tokens = [int(token) for token in user.tokens[start:stop]]
        tokens = [*history, *event_tokens]
        mask = [False] * len(history)
        mask.extend(start + offset in target_positions for offset in range(len(event_tokens)))
        if not tokens or tokens[0] != BOS_ID:
            raise AssertionError("dense training sequence must start with BOS")
        if len(tokens) - 1 > self.context_tokens_including_bos - 1 + self.window_events:
            raise AssertionError("dense example exceeds its configured context plus window")
        if len(tokens) - 2 > 200:
            raise AssertionError("dense SFT prefix exceeds the protocol's 200-event history maximum")
        if not any(mask):
            raise AssertionError("dense window contains no supervised target")
        if any(tokens[position] < 3 for position, valid in enumerate(mask) if valid):
            raise AssertionError("PAD, BOS, or OOV cannot be a dense SFT target")
        return tokens, mask

    def batches(self, store: SequenceStore, indices: Sequence[int], batch_size: int, device: torch.device | str = "cpu"):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(indices), batch_size):
            examples = [self.example(store, index) for index in indices[start:start + batch_size]]
            yield collate_sequences([item[0] for item in examples], [item[1] for item in examples], device=device)


def dense_full_softmax_backward(
    hidden: Tensor,
    targets: Tensor,
    valid: Tensor,
    output_weight: Tensor,
    *,
    chunk_size: int = 128,
    normalization_count: int | None = None,
    scaler: object | None = None,
    loss_weight: float = 1.0,
) -> float:
    """Exact full-softmax CE with chunked logits and one encoder backward.

    Each logit chunk backpropagates only into detached hidden states and the tied
    output matrix. The accumulated hidden gradient then traverses the encoder graph
    once. ``output_weight.grad`` receives both its LM-head and later tied-embedding
    contributions, preserving the tied-parameter derivative.
    """
    if hidden.ndim != 3 or targets.shape != hidden.shape[:2] or valid.shape != targets.shape:
        raise ValueError("hidden, targets and valid mask must have [batch,time,hidden]/[batch,time] shapes")
    if output_weight.ndim != 2 or output_weight.shape[1] != hidden.shape[-1]:
        raise ValueError("output matrix shape is incompatible with hidden states")
    if chunk_size < 1 or loss_weight < 0:
        raise ValueError("chunk_size must be positive and loss_weight non-negative")
    selected_hidden = hidden[valid.bool()]
    selected_targets = targets[valid.bool()].long()
    count = int(selected_targets.numel())
    if count == 0:
        return 0.0
    denominator = int(normalization_count if normalization_count is not None else count)
    if denominator < count:
        raise ValueError("normalization_count cannot be smaller than local valid target count")
    if bool(((selected_targets < 3) | (selected_targets >= output_weight.shape[0])).any()):
        raise ValueError("dense labels must be valid catalog token IDs")

    hidden_gradient = torch.zeros_like(selected_hidden, dtype=torch.float32)
    detached_loss = 0.0
    scale = scaler.get_scale() if scaler is not None else 1.0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        chunk_hidden = selected_hidden[start:stop].detach().float().requires_grad_(True)
        logits = torch.nn.functional.linear(chunk_hidden, output_weight.float())
        logits = logits.clone()
        logits[..., :3] = -torch.inf
        loss_sum = torch.nn.functional.cross_entropy(logits, selected_targets[start:stop], reduction="sum")
        loss = loss_sum * (float(loss_weight) / denominator)
        detached_loss += float(loss.detach().item())
        scaled_loss = scaler.scale(loss) if scaler is not None else loss
        scaled_loss.backward()
        if chunk_hidden.grad is None:
            raise RuntimeError("chunk hidden gradient was not produced")
        hidden_gradient[start:stop] = chunk_hidden.grad.detach()

    full_gradient = torch.zeros_like(hidden, dtype=hidden_gradient.dtype)
    full_gradient[valid.bool()] = hidden_gradient
    hidden.backward(full_gradient.to(dtype=hidden.dtype))
    # The scaler is unscaled exactly once by the optimizer owner after all batches.
    if scaler is not None and not math.isfinite(float(scale)):
        raise RuntimeError("invalid AMP gradient scale")
    return detached_loss


def dense_accumulation_step(
    actor: ItemGPT2Actor,
    microbatches: Sequence[SFTBatch],
    optimizer: torch.optim.Optimizer,
    *,
    logit_chunk_size: int = 128,
    scaler: object | None = None,
    gradient_clip: float = 1.0,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    counts = [int(batch.target_mask.sum().item()) for batch in microbatches]
    total = sum(counts)
    if total < 1:
        raise ValueError("dense accumulation step has no supervised targets")
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for batch, count in zip(microbatches, counts):
        if count == 0:
            continue
        with torch.autocast(
            device_type=batch.input_ids.device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            hidden = actor.encode(batch.input_ids, batch.attention_mask)
        total_loss += dense_full_softmax_backward(
            hidden,
            batch.targets,
            batch.target_mask,
            actor.output_weight,
            chunk_size=logit_chunk_size,
            normalization_count=total,
            scaler=scaler,
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
    return {"loss": total_loss, "supervised_targets": float(total), "windows": float(len(microbatches)),
            "grad_norm": float(grad_norm)}


def dense_checkpoint_metadata(*, catalog_sha256: str, sampler: DenseWindowIndex, optimizer_step: int,
                              config_sha256: str, seed: int) -> dict[str, object]:
    return {
        "schema_version": DENSE_CHECKPOINT_SCHEMA,
        "sampler_version": sampler.sampler_version,
        "catalog_sha256": catalog_sha256,
        "config_sha256": config_sha256,
        "optimizer_step": int(optimizer_step),
        "seed": int(seed),
        "window_events": sampler.window_events,
        "context_tokens_including_bos": sampler.context_tokens_including_bos,
        "normalization": "sum valid token losses across accumulation / total valid targets",
    }
