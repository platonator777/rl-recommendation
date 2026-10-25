from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class FrozenRewardScale:
    mean: float
    std: float
    std_floor: float = 1e-3
    version: str = "residual_support_f_rm_fit_sample_ratio_z_v1"

    def __post_init__(self) -> None:
        if not all(torch.isfinite(torch.tensor(v)) for v in (self.mean, self.std, self.std_floor)):
            raise ValueError("reward scale parameters must be finite")
        if self.std < 0 or self.std_floor <= 0:
            raise ValueError("reward scale std must be nonnegative and floor positive")

    @property
    def effective_std(self) -> float:
        return max(float(self.std), float(self.std_floor))

    def normalize(self, values: Tensor) -> Tensor:
        return (values.float() - float(self.mean)) / self.effective_std


def conditional_kl(log_q: Tensor, log_p: Tensor) -> Tensor:
    if log_q.shape != log_p.shape or log_q.ndim != 2:
        raise ValueError("conditional log distributions must be matching [contexts, support] arrays")
    q = log_q.float().exp()
    return torch.where(q > 0, q * (log_q.float() - log_p.float()), torch.zeros_like(q)).sum(-1)


def exact_support_statistics(log_q: Tensor, log_reference_conditional: Tensor, raw_rewards: Tensor,
                             support_mass: Tensor, scale: FrozenRewardScale, beta: float,
                             log_target: Tensor | None = None) -> dict[str, Tensor]:
    if beta <= 0 or log_q.shape != log_reference_conditional.shape or raw_rewards.shape != log_q.shape:
        raise ValueError("exact support objective requires beta>0 and matching conditional distributions/rewards")
    if support_mass.shape != (log_q.shape[0],):
        raise ValueError("support mass must have one value per context")
    mass = support_mass.float()
    if not bool(torch.isfinite(mass).all()) or bool(((mass < 0) | (mass > 1)).any()):
        raise ValueError("support mass must be finite and in [0, 1]")
    q = log_q.float().exp()
    reward = scale.normalize(raw_rewards)
    reward_expectation = (q * reward).sum(-1)
    kl_q_ref = conditional_kl(log_q, log_reference_conditional)
    weighted_kl = mass * kl_q_ref
    objective = mass * reward_expectation - float(beta) * weighted_kl
    tilted_logits = log_reference_conditional.float() + reward / float(beta)
    log_analytic = torch.log_softmax(tilted_logits, dim=-1)
    q_analytic = log_analytic.exp()
    analytic_reward = (q_analytic * reward).sum(-1)
    kl_analytic_ref = conditional_kl(log_analytic, log_reference_conditional)
    analytic_objective = mass * analytic_reward - float(beta) * mass * kl_analytic_ref
    raw_gap = analytic_objective - objective
    if bool((raw_gap < -1e-4).any()):
        raise AssertionError("analytic policy has a lower constrained objective than the supplied policy")
    gap = raw_gap.clamp_min(0.0)
    result = {
        "support_weighted_objective": objective,
        "support_weighted_reward_expectation": mass * reward_expectation,
        "unweighted_conditional_reward_expectation": reward_expectation,
        "kl_policy_to_reference_conditional": kl_q_ref,
        "kl_policy_to_reference_mass_weighted": weighted_kl,
        "analytic_support_weighted_objective": analytic_objective,
        "analytic_objective_gap": gap,
        "analytic_target_log_policy": log_analytic,
    }
    if log_target is not None:
        if log_target.shape != log_q.shape:
            raise ValueError("analytic target and policy shapes differ")
        target = log_target.float().exp().detach()
        kl_target_policy = torch.where(
            target > 0, target * (log_target.float() - log_q.float()), torch.zeros_like(target),
        ).sum(-1)
        result["kl_target_to_policy_conditional"] = kl_target_policy
        result["kl_target_to_policy_mass_weighted"] = mass * kl_target_policy
    return result


def sampled_loo_loss(log_q: Tensor, log_reference_conditional: Tensor,
                     raw_rewards: Tensor, support_mass: Tensor,
                     scale: FrozenRewardScale, beta: float, samples: int,
                     generator: torch.Generator | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    if samples < 2 or beta <= 0 or log_q.shape != raw_rewards.shape or log_q.shape != log_reference_conditional.shape or support_mass.shape != (len(log_q),):
        raise ValueError("sampled LOO objective requires aligned arrays, beta>0 and at least two samples")
    q = log_q.float().exp()
    actions = torch.multinomial(q, samples, replacement=True, generator=generator)
    normalized = scale.normalize(raw_rewards)
    sampled_rewards = normalized.gather(1, actions)
    advantages = sampled_rewards - (sampled_rewards.sum(-1, keepdim=True) - sampled_rewards) / (samples - 1)
    selected_logq = log_q.float().gather(1, actions)
    kl = conditional_kl(log_q, log_reference_conditional)
    policy_gradient = -support_mass.float() * (advantages.detach() * selected_logq).mean(-1)
    mass_weighted_kl = support_mass.float() * kl
    loss = (policy_gradient + float(beta) * mass_weighted_kl).mean()
    return loss, {"sampled_rewards": sampled_rewards.detach(), "loo_advantages": advantages.detach(),
                  "policy_gradient_per_context": policy_gradient.detach(),
                  "kl_policy_to_reference_mass_weighted": mass_weighted_kl.detach()}
