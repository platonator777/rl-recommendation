from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tlab.panel_tools.objectives import FrozenRewardScale


@dataclass(frozen=True)
class FixedSupportPolicy:
    """Full-catalog policy with exponential tilting only inside a fixed support."""

    probabilities: Tensor
    support_mass: Tensor
    conditional_probabilities: Tensor


def _validate_support(log_reference: Tensor, candidate_indices: Tensor) -> tuple[Tensor, Tensor]:
    if log_reference.ndim != 2 or candidate_indices.ndim != 2:
        raise ValueError("reference and support must be batch-by-catalog / batch-by-support")
    if candidate_indices.shape[0] != log_reference.shape[0] or candidate_indices.shape[1] == 0:
        raise ValueError("support must be non-empty and have one row per context")
    index = candidate_indices.long()
    if bool(((index < 0) | (index >= log_reference.shape[1])).any()):
        raise ValueError("candidate index outside catalog")
    if any(torch.unique(row).numel() != row.numel() for row in index):
        raise ValueError("candidate support contains duplicate item indices")
    logp = log_reference.float()
    probs = logp.exp()
    row_sums = probs.sum(-1)
    if bool(torch.isnan(logp).any()) or bool(torch.isposinf(logp).any()) or bool((~torch.isfinite(row_sums) | (row_sums <= 0)).any()):
        raise ValueError("reference must have finite log probabilities")
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=2e-5, rtol=2e-5):
        raise ValueError("reference rows must be normalized log probabilities")
    return probs, index


def fixed_support_policy(
    log_reference: Tensor,
    candidate_indices: Tensor,
    reward_adjustment: Tensor,
) -> FixedSupportPolicy:
    """Tilt a fixed candidate set and preserve its reference mass and the exact tail.

    ``reward_adjustment`` is the already-normalized reward divided by beta (or
    another score delta). This low-level function intentionally does not select
    beta or estimate any normalization statistics.
    """
    probabilities, index = _validate_support(log_reference, candidate_indices)
    if reward_adjustment.shape != index.shape:
        raise ValueError("one reward adjustment is required for each supported candidate")
    adjustment = reward_adjustment.float()
    if not bool(torch.isfinite(adjustment).all()):
        raise ValueError("reward adjustments must be finite")
    ref_support = probabilities.gather(1, index)
    mass = ref_support.sum(-1, keepdim=True)
    if bool((mass <= 0).any()):
        raise ValueError("candidate support has zero reference mass")
    conditional = torch.softmax(ref_support.clamp_min(torch.finfo(torch.float32).tiny).log() + adjustment, -1)
    supported_new = mass * conditional
    full_new = probabilities.scatter(1, index, supported_new)
    return FixedSupportPolicy(full_new, mass.squeeze(-1), conditional)


def fixed_support_analytic_policy(
    log_reference: Tensor,
    candidate_indices: Tensor,
    raw_catalog_rewards: Tensor,
    normalization: FrozenRewardScale,
    beta: float,
) -> FixedSupportPolicy:
    if beta < 0 or raw_catalog_rewards.shape != log_reference.shape:
        raise ValueError("invalid beta or reference/reward shape")
    _, index = _validate_support(log_reference, candidate_indices)
    normalized = normalization.normalize(raw_catalog_rewards).gather(1, index)
    if beta == 0:
        best = normalized.argmax(-1, keepdim=True)
        probabilities, _ = _validate_support(log_reference, candidate_indices)
        ref_support = probabilities.gather(1, index)
        mass = ref_support.sum(-1, keepdim=True)
        conditional = torch.zeros_like(ref_support).scatter_(-1, best, 1.0)
        full = probabilities.scatter(1, index, mass * conditional)
        return FixedSupportPolicy(full, mass.squeeze(-1), conditional)
    return fixed_support_policy(log_reference, index, normalized / float(beta))


def fixed_support_kl(policy: FixedSupportPolicy, log_reference: Tensor, candidate_indices: Tensor) -> Tensor:
    """Full-catalog KL for a fixed-tail policy; unchanged tail contributes zero."""
    probabilities, index = _validate_support(log_reference, candidate_indices)
    if policy.probabilities.shape != probabilities.shape:
        raise ValueError("policy and reference catalog shapes differ")
    if policy.support_mass.shape != probabilities.shape[:1]:
        raise ValueError("support mass must have one value per context")
    policy_probs = policy.probabilities.float()
    if not bool(torch.isfinite(policy_probs).all()) or bool((policy_probs < 0).any()):
        raise ValueError("policy probabilities must be finite and non-negative")
    actual_mass = policy_probs.gather(1, index).sum(-1)
    if not torch.allclose(policy_probs.sum(-1), torch.ones_like(actual_mass), atol=2e-5, rtol=2e-5):
        raise ValueError("policy probabilities must sum to one")
    if not torch.allclose(actual_mass, policy.support_mass.float(), atol=2e-5, rtol=2e-5):
        raise ValueError("policy does not preserve the declared support mass")
    support_mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(1, index, True)
    if not torch.allclose(policy_probs.masked_select(~support_mask), probabilities.masked_select(~support_mask), atol=2e-6, rtol=2e-6):
        raise ValueError("policy changed probabilities outside the declared support")
    ref_c = probabilities.gather(1, index)
    q_c = policy.probabilities.gather(1, index)
    mass = policy.support_mass.unsqueeze(-1)
    ref_cond = ref_c / mass
    q_cond = q_c / mass
    q_log = q_cond.clamp_min(torch.finfo(torch.float32).tiny).log()
    ref_log = ref_cond.clamp_min(torch.finfo(torch.float32).tiny).log()
    conditional_kl = torch.where(q_cond > 0, q_cond * (q_log - ref_log), torch.zeros_like(q_cond)).sum(-1)
    return policy.support_mass * conditional_kl


def merge_full_catalog_topk(
    policy_probabilities: Tensor,
    k: int,
    item_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rank a complete distribution (including its unchanged tail).

    Ties are resolved by ascending raw item ID, matching the panel_tools evaluator.
    Returns (raw item IDs, probabilities), both sorted by rank.
    """
    if policy_probabilities.ndim != 2 or not 0 < k <= policy_probabilities.shape[1]:
        raise ValueError("invalid policy probabilities or top-k")
    if not bool(torch.isfinite(policy_probabilities).all()) or bool((policy_probabilities < 0).any()):
        raise ValueError("policy probabilities must be finite and non-negative")
    batch, catalog = policy_probabilities.shape
    if item_ids is None:
        ids = torch.arange(catalog, device=policy_probabilities.device, dtype=torch.long).expand(batch, -1)
    else:
        if item_ids.ndim == 1:
            item_ids = item_ids.unsqueeze(0).expand(batch, -1)
        if item_ids.shape != policy_probabilities.shape:
            raise ValueError("item IDs must broadcast to the complete policy shape")
        ids = item_ids.long()
    # Stable two-stage sort gives item-ID ascending tie break, then score descending.
    by_id = torch.argsort(ids, dim=-1, stable=True)
    sorted_ids = ids.gather(1, by_id)
    sorted_probs = policy_probabilities.gather(1, by_id)
    by_score = torch.argsort(sorted_probs, dim=-1, descending=True, stable=True)[:, :k]
    return sorted_ids.gather(1, by_score), sorted_probs.gather(1, by_score)


def fixed_support_objective(
    policy: FixedSupportPolicy,
    log_reference: Tensor,
    candidate_indices: Tensor,
    raw_catalog_rewards: Tensor,
    normalization: FrozenRewardScale,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return full-catalog J, expected reward, and full KL for the policy."""
    if raw_catalog_rewards.shape != log_reference.shape:
        raise ValueError("reference/reward shape mismatch")
    probs = policy.probabilities.float()
    reward = normalization.normalize(raw_catalog_rewards)
    expected = (probs * reward).sum(-1)
    kl = fixed_support_kl(policy, log_reference, candidate_indices)
    return expected - float(beta) * kl, expected, kl
