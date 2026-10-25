from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ResidualHeadSpec:
    input_dim: int
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.hidden_dim < 1:
            raise ValueError("residual head dimensions must be positive")


class ResidualPolicyHead(nn.Module):
    """Candidate feature MLP whose zero-initialized output preserves reference logits."""

    def __init__(self, spec: ResidualHeadSpec):
        super().__init__()
        self.spec = spec
        self.network = nn.Sequential(
            nn.Linear(spec.input_dim, spec.hidden_dim),
            nn.Tanh(),
            nn.Linear(spec.hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, candidate_features: Tensor) -> Tensor:
        if candidate_features.shape[-1] != self.spec.input_dim:
            raise ValueError("candidate feature width does not match residual head schema")
        return self.network(candidate_features.float()).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def residual_log_policy(log_reference_conditional: Tensor, candidate_features: Tensor, head: ResidualPolicyHead) -> tuple[Tensor, Tensor]:
    if log_reference_conditional.ndim != 2 or candidate_features.ndim != 3:
        raise ValueError("reference log probabilities must be [contexts, candidates] and features [contexts, candidates, dim]")
    if candidate_features.shape[:2] != log_reference_conditional.shape:
        raise ValueError("candidate features and reference support must align")
    delta = head(candidate_features)
    return torch.log_softmax(log_reference_conditional.float() + delta, dim=-1), delta


def feature_columns_without_feedback(feature_names: Sequence[str]) -> tuple[int, ...]:
    """Keep item count/recency/metadata while removing user feedback-derived columns."""
    selected = tuple(
        index for index, name in enumerate(feature_names)
        if "ratio" not in name.lower().split("_")
        and "feedback" not in name.lower().split("_")
        and "seconds_proxy" not in name.lower()
    )
    if not selected:
        raise ValueError("feedback ablation removed every candidate feature")
    return selected


def select_feature_columns(features: Tensor, columns: Sequence[int] | None = None) -> Tensor:
    if features.ndim != 3:
        raise ValueError("candidate features must have [contexts, candidates, features] shape")
    if columns is None:
        return features
    index = torch.as_tensor(tuple(columns), dtype=torch.long, device=features.device)
    if index.ndim != 1 or index.numel() == 0 or bool(((index < 0) | (index >= features.shape[-1])).any()):
        raise ValueError("feature column indices are empty or out of range")
    return features.index_select(-1, index)
