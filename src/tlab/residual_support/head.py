from __future__ import annotations

from dataclasses import dataclass

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
    """Train-only scaled feature residual; zero output starts at the reference policy."""

    def __init__(self, spec: ResidualHeadSpec):
        super().__init__()
        self.spec = spec
        self.network = nn.Sequential(nn.Linear(spec.input_dim, spec.hidden_dim), nn.Tanh(), nn.Linear(spec.hidden_dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.spec.input_dim:
            raise ValueError("feature width does not match residual head schema")
        return self.network(features.float()).squeeze(-1)
