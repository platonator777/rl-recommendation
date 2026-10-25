from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _masked_mean(loss: Tensor, mask: Tensor) -> Tensor:
    selected = mask.bool()
    if not torch.any(selected):
        return loss.sum() * 0.0
    return loss[selected].mean()


def reward_model_loss(
    predictions: dict[str, Tensor],
    ratio_target: Tensor,
    ratio_mask: Tensor,
    seconds_scaled_target: Tensor,
    seconds_mask: Tensor,
    like_target: Tensor | None = None,
    like_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Unweighted sum of per-head mean losses; masks preserve sparse-label semantics."""
    ratio_loss = _masked_mean((predictions["ratio"].float() - ratio_target.float()).square(), ratio_mask)
    seconds_loss = _masked_mean(
        (predictions["seconds_scaled"].float() - seconds_scaled_target.float()).square(), seconds_mask
    )
    if like_target is None or like_mask is None:
        like_loss = predictions["like_probability"].sum() * 0.0
    else:
        eps = torch.finfo(torch.float32).eps
        probability = predictions["like_probability"].float().clamp(eps, 1.0 - eps)
        like_loss = _masked_mean(F.binary_cross_entropy(probability, like_target.float(), reduction="none"), like_mask)
    total = ratio_loss + seconds_loss + like_loss
    return total, {
        "ratio_mse": ratio_loss.detach(),
        "seconds_mse_scaled": seconds_loss.detach(),
        "like_bce": like_loss.detach(),
    }
