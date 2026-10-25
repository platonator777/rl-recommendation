from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ActorShape:
    vocab_size: int
    n_positions: int = 256
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    dropout: float = 0.1


class ItemGPT2Actor(nn.Module):
    """Randomly initialized GPT-2 with item IDs as tokens and tied embeddings."""

    def __init__(self, shape: ActorShape):
        super().__init__()
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError as exc:
            raise RuntimeError("transformers is required to construct the item-ID GPT-2 actor") from exc
        if shape.vocab_size <= 3:
            raise ValueError("vocab_size must include PAD, BOS, OOV and at least one catalog item")
        if shape.n_embd % shape.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        config = GPT2Config(
            vocab_size=shape.vocab_size,
            n_positions=shape.n_positions,
            n_ctx=shape.n_positions,
            n_embd=shape.n_embd,
            n_layer=shape.n_layer,
            n_head=shape.n_head,
            n_inner=4 * shape.n_embd,
            resid_pdrop=shape.dropout,
            embd_pdrop=shape.dropout,
            attn_pdrop=shape.dropout,
            bos_token_id=1,
            eos_token_id=0,
            pad_token_id=0,
            use_cache=False,
        )
        self.lm = GPT2LMHeadModel(config)
        self.lm.tie_weights()
        self.vocab_size = shape.vocab_size
        self.n_positions = shape.n_positions

    @property
    def output_weight(self) -> Tensor:
        return self.lm.get_output_embeddings().weight

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.shape[-1] > self.n_positions:
            raise ValueError(f"sequence length {input_ids.shape[-1]} exceeds n_positions={self.n_positions}")
        return self.lm.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    @staticmethod
    def last_valid_hidden(hidden: Tensor, attention_mask: Tensor) -> Tensor:
        positions = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]

    def action_logits(self, hidden: Tensor) -> Tensor:
        logits = F.linear(hidden, self.output_weight)
        logits = logits.float()
        logits[..., :3] = -torch.inf
        return logits

    def action_log_probs(self, hidden: Tensor) -> Tensor:
        return F.log_softmax(self.action_logits(hidden), dim=-1)

    def next_action_log_probs(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.encode(input_ids, attention_mask)
        return self.action_log_probs(self.last_valid_hidden(hidden, attention_mask))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def chunked_full_softmax_ce(
    hidden: Tensor,
    targets: Tensor,
    valid: Tensor,
    output_weight: Tensor,
    chunk_size: int = 256,
    first_catalog_id: int = 3,
) -> Tensor:
    """Exact mean catalog CE; intended for small equivalence checks and evaluation."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    selected_hidden = hidden[valid]
    selected_targets = targets[valid].long()
    if selected_targets.numel() == 0:
        return hidden.sum() * 0.0
    losses = []
    for start in range(0, selected_targets.numel(), chunk_size):
        stop = min(start + chunk_size, selected_targets.numel())
        logits = F.linear(selected_hidden[start:stop], output_weight).float()
        logits[..., :first_catalog_id] = -torch.inf
        losses.append(F.cross_entropy(logits, selected_targets[start:stop], reduction="sum"))
    return torch.stack(losses).sum() / selected_targets.numel()


def backward_chunked_full_softmax_ce(
    hidden: Tensor,
    targets: Tensor,
    valid: Tensor,
    output_weight: Tensor,
    chunk_size: int = 256,
    grad_scale: object | None = None,
    loss_weight: float = 1.0,
    first_catalog_id: int = 3,
) -> float:
    """Backprop exact CE by logit chunks without retaining all chunk logits at once.

    The caller must invoke this before freeing ``hidden``'s forward graph and must
    not call backward on the same graph afterwards.
    """
    selected_hidden = hidden[valid]
    selected_targets = targets[valid].long()
    count = selected_targets.numel()
    if count == 0:
        return 0.0
    detached_total = 0.0
    starts = range(0, count, chunk_size)
    for chunk_index, start in enumerate(starts):
        stop = min(start + chunk_size, count)
        logits = F.linear(selected_hidden[start:stop], output_weight).float()
        logits[..., :first_catalog_id] = -torch.inf
        loss = F.cross_entropy(logits, selected_targets[start:stop], reduction="sum") / count * loss_weight
        detached_total += float(loss.detach().item())
        retain_graph = stop < count
        if grad_scale is None:
            loss.backward(retain_graph=retain_graph)
        else:
            grad_scale.scale(loss).backward(retain_graph=retain_graph)
    return detached_total
