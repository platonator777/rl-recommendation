from __future__ import annotations

import torch
from torch import Tensor


def full_categorical_kl(log_policy: Tensor, log_reference: Tensor) -> Tensor:
    """Exact KL(policy || reference) over matching full action distributions."""
    if log_policy.shape != log_reference.shape:
        raise ValueError("policy and reference log-probabilities must have the same shape")
    policy = log_policy.float().exp()
    nonzero = policy > 0
    safe_log_policy = torch.where(nonzero, log_policy.float(), torch.zeros_like(policy))
    safe_log_reference = torch.where(nonzero, log_reference.float(), torch.zeros_like(policy))
    return (policy * (safe_log_policy - safe_log_reference)).sum(dim=-1)


def group_advantages(
    rewards: Tensor,
    epsilon: float = 1e-4,
    normalize: bool = True,
    centered_correction: bool = False,
) -> Tensor:
    """Population-standardized GRPO advantages, computed independently per row."""
    if rewards.ndim < 1 or rewards.shape[-1] < 2:
        raise ValueError("each group must contain at least two sampled actions")
    detached = rewards.detach().float()
    centered = detached - detached.mean(dim=-1, keepdim=True)
    if not normalize:
        if centered_correction:
            centered = centered * (detached.shape[-1] / (detached.shape[-1] - 1))
        return centered
    scale = detached.std(dim=-1, correction=0, keepdim=True)
    return centered / (scale + epsilon)


def leave_one_out_advantages(rewards: Tensor) -> Tensor:
    """REINFORCE baseline that excludes the current sample from its group mean."""
    if rewards.ndim < 1 or rewards.shape[-1] < 2:
        raise ValueError("leave-one-out baseline requires at least two samples")
    detached = rewards.detach().float()
    total = detached.sum(dim=-1, keepdim=True)
    other_mean = (total - detached) / (detached.shape[-1] - 1)
    return detached - other_mean


def prepare_reward_objective(
    raw_rewards: Tensor,
    *,
    reward_mean: float,
    reward_std: float,
    actions: Tensor | None = None,
    first_catalog_id: int = 3,
) -> dict[str, Tensor]:
    """Normalize frozen rewards once and optionally prepare sampled LOO advantages.

    ``raw_rewards`` contains catalog-only columns in token order starting at
    ``first_catalog_id``. The returned normalized vector is shared by the exact
    and sampled production objectives; raw samples are retained only for metrics.
    """
    if raw_rewards.ndim != 2:
        raise ValueError("raw rewards must have [batch, catalog] shape")
    scale = max(float(reward_std), 1e-3)
    normalized = (raw_rewards.float() - float(reward_mean)) / scale
    result = {"normalized_reward": normalized}
    if actions is not None:
        if actions.ndim != 2 or actions.shape[0] != raw_rewards.shape[0]:
            raise ValueError("sampled actions must have [batch, group] shape")
        indices = actions.long() - int(first_catalog_id)
        if bool(((indices < 0) | (indices >= raw_rewards.shape[1])).any()):
            raise ValueError("sampled action lies outside the reward catalog")
        sampled_raw = raw_rewards.gather(1, indices)
        sampled_normalized = normalized.gather(1, indices)
        result.update({
            "sampled_raw_reward": sampled_raw.detach(),
            "sampled_normalized_reward": sampled_normalized.detach(),
            "advantages": leave_one_out_advantages(sampled_normalized),
        })
    return result


def full_catalog_support_components(
    log_policy: Tensor,
    log_reference: Tensor,
    unsupported_mask: Tensor,
) -> dict[str, Tensor]:
    """Compute differentiable unsupported mass and hinge excess over the full catalog."""
    if log_policy.shape != log_reference.shape:
        raise ValueError("policy and reference distributions must have matching shapes")
    if log_policy.shape[-1] != unsupported_mask.shape[-1] + 3:
        raise ValueError("unsupported mask must cover catalog actions and exclude three special tokens")
    probs = log_policy[:, 3:].float().exp()
    reference_probs = log_reference[:, 3:].float().exp()
    mask = unsupported_mask.to(device=probs.device, dtype=torch.float32)
    policy_mass = (probs * mask).sum(-1)
    reference_mass = (reference_probs * mask).sum(-1)
    excess = torch.relu(policy_mass - reference_mass)
    return {"unsupported_mass": policy_mass, "reference_unsupported_mass": reference_mass,
            "support_excess": excess}


def reinforce_loss(log_probs: Tensor, rewards: Tensor, kl_per_context: Tensor, beta: float) -> Tensor:
    if log_probs.shape != rewards.shape:
        raise ValueError("sample log-probs and rewards must have the same shape")
    advantages = leave_one_out_advantages(rewards)
    return -(advantages * log_probs).mean() + beta * kl_per_context.float().mean()


def grpo_clipped_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    kl_per_context: Tensor,
    beta: float,
    clip_epsilon: float = 0.2,
) -> tuple[Tensor, dict[str, Tensor]]:
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.shape != advantages.shape:
        raise ValueError("new/old log-probs and advantages must have matching shapes")
    log_ratio = new_log_probs.float() - old_log_probs.detach().float()
    ratio = torch.exp(log_ratio)
    advantage = advantages.detach().float()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    kl_loss = beta * kl_per_context.float().mean()
    loss = policy_loss + kl_loss
    return loss, {
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_epsilon).float().mean().detach(),
        "ratio_mean": ratio.mean().detach(),
        "approx_kl_to_old": ((ratio - 1.0) - log_ratio).mean().detach(),
    }


def analytic_reward_kl_policy(log_reference: Tensor, rewards: Tensor, beta: float) -> Tensor:
    """Return exact one-step optimizer; beta=0 yields a deterministic reward argmax."""
    if log_reference.shape != rewards.shape:
        raise ValueError("reference distribution and reward arrays must have matching shapes")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if beta == 0:
        best = rewards.float().argmax(dim=-1, keepdim=True)
        return torch.zeros_like(rewards, dtype=torch.float32).scatter_(-1, best, 1.0)
    adjusted = log_reference.float() + rewards.float() / beta
    return torch.softmax(adjusted, dim=-1)


def reward_kl_objective(policy: Tensor, rewards: Tensor, log_reference: Tensor, beta: float) -> Tensor:
    if policy.shape != rewards.shape or policy.shape != log_reference.shape:
        raise ValueError("policy, rewards, and reference must have matching shapes")
    log_policy = policy.float().clamp_min(torch.finfo(torch.float32).tiny).log()
    kl = full_categorical_kl(log_policy, log_reference.float())
    return (policy.float() * rewards.float()).sum(dim=-1) - beta * kl
