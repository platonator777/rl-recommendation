from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class RewardModelShape:
    vocab_size: int
    hidden_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    max_positions: int = 256
    encoder: str = "transformer"


class CandidateRewardModel(nn.Module):
    """Candidate-conditioned ratio, seconds and optional new-like estimator."""

    def __init__(self, shape: RewardModelShape):
        super().__init__()
        if shape.encoder not in {"transformer", "gru"}:
            raise ValueError("encoder must be 'transformer' or 'gru'")
        self.shape = shape
        self.item_embedding = nn.Embedding(shape.vocab_size, shape.hidden_size)
        if shape.encoder == "transformer":
            try:
                from transformers import GPT2Config, GPT2Model
            except ImportError as exc:
                raise RuntimeError("transformers is required for the transformer reward encoder") from exc
            config = GPT2Config(
                vocab_size=shape.vocab_size,
                n_positions=shape.max_positions,
                n_ctx=shape.max_positions,
                n_embd=shape.hidden_size,
                n_layer=shape.n_layer,
                n_head=shape.n_head,
                n_inner=4 * shape.hidden_size,
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
                use_cache=False,
            )
            self.history_encoder = GPT2Model(config)
            self.gru = None
        else:
            self.history_encoder = None
            self.gru = nn.GRU(shape.hidden_size, shape.hidden_size, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(3 * shape.hidden_size + 3, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def encode_history(self, history_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if history_ids.shape[-1] > self.shape.max_positions:
            history_ids = history_ids[:, -self.shape.max_positions:]
            attention_mask = attention_mask[:, -self.shape.max_positions:]
        if self.history_encoder is not None:
            encoded = self.history_encoder(
                input_ids=history_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            last = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
            return encoded[torch.arange(encoded.shape[0], device=encoded.device), last]
        lengths = attention_mask.long().sum(dim=-1).clamp_min(1).cpu()
        embedded = self.item_embedding(history_ids)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, final = self.gru(packed)
        return final[-1]

    def score_encoded(
        self,
        history_state: Tensor,
        candidate_ids: Tensor,
        duration_seconds: Tensor,
        duration_missing: Tensor,
        recent_repeat: Tensor,
    ) -> dict[str, Tensor]:
        candidate = self.item_embedding(candidate_ids)
        state = history_state
        while state.ndim < candidate.ndim:
            state = state.unsqueeze(1)
        duration = torch.log1p(duration_seconds.float().clamp_min(0)).unsqueeze(-1)
        missing = duration_missing.float().unsqueeze(-1)
        repeat = recent_repeat.float().unsqueeze(-1)
        features = torch.cat((state.expand_as(candidate), candidate, state.expand_as(candidate) * candidate,
                              duration, missing, repeat), dim=-1)
        raw = self.scorer(features)
        return {
            "ratio": torch.sigmoid(raw[..., 0]),
            "seconds_scaled": F.softplus(raw[..., 1]),
            "like_probability": torch.sigmoid(raw[..., 2]),
        }

    def forward(
        self,
        history_ids: Tensor,
        attention_mask: Tensor,
        candidate_ids: Tensor,
        duration_seconds: Tensor,
        duration_missing: Tensor,
        recent_repeat: Tensor,
    ) -> dict[str, Tensor]:
        history = self.encode_history(history_ids, attention_mask)
        return self.score_encoded(history, candidate_ids, duration_seconds, duration_missing, recent_repeat)

    @torch.no_grad()
    def score_catalog(
        self,
        history_ids: Tensor,
        attention_mask: Tensor,
        candidate_ids: Tensor,
        durations: Tensor,
        duration_missing: Tensor,
        repeat_mask: Tensor,
        chunk_size: int = 512,
    ) -> dict[str, Tensor]:
        """Score a catalog in chunks while encoding each real history only once."""
        state = self.encode_history(history_ids, attention_mask)
        outputs: dict[str, list[Tensor]] = {"ratio": [], "seconds_scaled": [], "like_probability": []}
        for start in range(0, candidate_ids.shape[-1], chunk_size):
            stop = min(start + chunk_size, candidate_ids.shape[-1])
            item_ids = candidate_ids[..., start:stop]
            d = durations[..., start:stop]
            dm = duration_missing[..., start:stop]
            repeat = repeat_mask[..., start:stop]
            prediction = self.score_encoded(state, item_ids, d, dm, repeat)
            for name, value in prediction.items():
                outputs[name].append(value)
        return {name: torch.cat(parts, dim=-1) for name, parts in outputs.items()}
