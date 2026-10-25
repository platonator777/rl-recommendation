from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tlab.train.objectives import leave_one_out_advantages


@dataclass(frozen=True)
class FrozenRewardScale:
    mean: float
    std: float
    version: str = "frozen_reward_scale_v1"
    std_floor: float = 1e-3

    def __post_init__(self) -> None:
        if not torch.isfinite(torch.tensor([self.mean, self.std, self.std_floor])).all():
            raise ValueError("reward normalization parameters must be finite")
        if self.std < 0 or self.std_floor <= 0:
            raise ValueError("reward std must be non-negative and floor must be positive")

    @property
    def effective_std(self) -> float:
        return max(float(self.std), float(self.std_floor))

    def normalize(self, raw_rewards: Tensor) -> Tensor:
        return (raw_rewards.float() - float(self.mean)) / self.effective_std

    def sample(self, raw_catalog_rewards: Tensor, action_ids: Tensor, first_catalog_id: int = 3) -> dict[str, Tensor]:
        if raw_catalog_rewards.ndim != 2 or action_ids.ndim != 2 or raw_catalog_rewards.shape[0] != action_ids.shape[0]:
            raise ValueError("raw rewards and sampled actions must be batch-by-catalog and batch-by-group")
        indices = action_ids.long() - int(first_catalog_id)
        if bool(((indices < 0) | (indices >= raw_catalog_rewards.shape[1])).any()):
            raise ValueError("sampled action lies outside the reward catalog")
        normalized_catalog = self.normalize(raw_catalog_rewards)
        raw = raw_catalog_rewards.float().gather(1, indices)
        normalized = normalized_catalog.gather(1, indices)
        return {
            "normalized_catalog_rewards": normalized_catalog,
            "sampled_raw_rewards": raw.detach(),
            "sampled_normalized_rewards": normalized.detach(),
            "advantages": leave_one_out_advantages(normalized),
        }


def exact_reward_kl_objective(
    log_policy: Tensor,
    log_reference: Tensor,
    raw_catalog_rewards: Tensor,
    normalization: FrozenRewardScale,
    beta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    if beta < 0 or log_policy.shape != log_reference.shape or log_policy.shape != raw_catalog_rewards.shape:
        raise ValueError("invalid beta or mismatched policy/reference/reward arrays")
    lp = log_policy.float()
    lr = log_reference.float()
    probability = lp.exp()
    reward = normalization.normalize(raw_catalog_rewards)
    expected_reward = (probability * reward).sum(-1)
    kl = torch.where(probability > 0, probability * (lp - lr), torch.zeros_like(probability)).sum(-1)
    return expected_reward - float(beta) * kl, {"expected_normalized_reward": expected_reward, "kl": kl, "normalized_reward": reward}


def sampled_policy_loss(
    sampled_log_policy: Tensor,
    raw_catalog_rewards: Tensor,
    sampled_action_ids: Tensor,
    exact_kl_per_context: Tensor,
    normalization: FrozenRewardScale,
    beta: float,
    first_catalog_id: int = 3,
) -> tuple[Tensor, dict[str, Tensor]]:
    terms = normalization.sample(raw_catalog_rewards, sampled_action_ids, first_catalog_id)
    if sampled_log_policy.shape != terms["advantages"].shape or exact_kl_per_context.ndim != 1:
        raise ValueError("sampled log probabilities/advantages or exact KL have incompatible shapes")
    if exact_kl_per_context.shape[0] != sampled_log_policy.shape[0]:
        raise ValueError("exact KL must have one value per context")
    policy_loss = -(terms["advantages"].detach() * sampled_log_policy.float()).mean(-1)
    loss = (policy_loss + float(beta) * exact_kl_per_context.float()).mean()
    return loss, {**terms, "policy_loss_per_context": policy_loss.detach()}


def fixed_support_sampled_policy_loss(
    sampled_log_conditional_policy: Tensor,
    raw_catalog_rewards: Tensor,
    sampled_action_ids: Tensor,
    exact_full_kl_per_context: Tensor,
    support_mass: Tensor,
    normalization: FrozenRewardScale,
    beta: float,
    first_catalog_id: int = 3,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sampled LOO objective for a fixed-tail policy.

    Samples are drawn from q conditioned on supported candidates. The policy
    gradient is weighted by the fixed support mass m. The supplied KL is
    already the full-catalog KL m*KL(q_C || p_ref,C/m).
    """
    if beta < 0 or support_mass.ndim != 1 or exact_full_kl_per_context.ndim != 1:
        raise ValueError("invalid beta, support mass, or KL shape")
    terms = normalization.sample(raw_catalog_rewards, sampled_action_ids, first_catalog_id)
    if sampled_log_conditional_policy.shape != terms["advantages"].shape:
        raise ValueError("sampled conditional log probabilities have incompatible shape")
    if support_mass.shape[0] != sampled_log_conditional_policy.shape[0] or exact_full_kl_per_context.shape[0] != support_mass.shape[0]:
        raise ValueError("support mass and full KL must have one value per context")
    mass = support_mass.float()
    if not bool(torch.isfinite(mass).all()) or bool(((mass < 0) | (mass > 1)).any()):
        raise ValueError("support mass must be finite and lie in [0, 1]")
    conditional_loss = -(terms["advantages"].detach() * sampled_log_conditional_policy.float()).mean(-1)
    policy_loss = mass * conditional_loss
    loss = (policy_loss + float(beta) * exact_full_kl_per_context.float()).mean()
    return loss, {
        **terms,
        "conditional_policy_loss_per_context": conditional_loss.detach(),
        "policy_loss_per_context": policy_loss.detach(),
    }


def analytic_policy(log_reference: Tensor, raw_catalog_rewards: Tensor, normalization: FrozenRewardScale, beta: float) -> Tensor:
    if beta < 0 or log_reference.shape != raw_catalog_rewards.shape:
        raise ValueError("invalid beta or policy/reward shape")
    if beta == 0:
        best = normalization.normalize(raw_catalog_rewards).argmax(dim=-1, keepdim=True)
        return torch.zeros_like(log_reference, dtype=torch.float32).scatter_(-1, best, 1.0)
    return torch.softmax(log_reference.float() + normalization.normalize(raw_catalog_rewards) / float(beta), dim=-1)
