from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from tlab.panel_tools.objectives import FrozenRewardScale, fixed_support_sampled_policy_loss
from tlab.panel_tools.policy_head import ResidualHeadSpec, ResidualPolicyHead, residual_log_policy


PolicyObjective = Literal["sampled_loo", "exact_expected_reward", "analytic_q_distillation"]


def _conditional_kl(log_q: Tensor, log_reference: Tensor) -> Tensor:
    q = log_q.float().exp()
    return torch.where(q > 0, q * (log_q.float() - log_reference.float()), torch.zeros_like(q)).sum(-1)


def exact_fixed_support_loss(
    log_q: Tensor,
    log_reference_conditional: Tensor,
    raw_candidate_rewards: Tensor,
    support_mass: Tensor,
    normalization: FrozenRewardScale,
    beta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Negative fixed-support objective, with full-catalog mass m in reward and KL."""
    if beta < 0 or raw_candidate_rewards.shape != log_q.shape or log_reference_conditional.shape != log_q.shape:
        raise ValueError("invalid beta or incompatible exact objective tensors")
    if support_mass.shape != (log_q.shape[0],):
        raise ValueError("support mass must have one value per context")
    mass = support_mass.float()
    if bool(((mass < 0) | (mass > 1)).any()) or not bool(torch.isfinite(mass).all()):
        raise ValueError("support mass must lie in [0, 1]")
    reward = normalization.normalize(raw_candidate_rewards)
    q = log_q.float().exp()
    expected = (q * reward).sum(-1)
    conditional_kl = _conditional_kl(log_q.float(), log_reference_conditional.float())
    full_kl = mass * conditional_kl
    per_context = -mass * expected + float(beta) * full_kl
    return per_context.mean(), {
        "expected_normalized_reward": expected.detach(),
        "conditional_kl": conditional_kl.detach(),
        "full_kl": full_kl.detach(),
        "per_context_loss": per_context.detach(),
    }


def distillation_loss(
    log_q: Tensor,
    log_analytic_target: Tensor,
    support_mass: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    if log_q.shape != log_analytic_target.shape or support_mass.shape != (log_q.shape[0],):
        raise ValueError("distillation distributions or support mass have incompatible shapes")
    target = log_analytic_target.float().exp().detach()
    kl = torch.where(target > 0, target * (log_analytic_target.float() - log_q.float()), torch.zeros_like(target)).sum(-1)
    weighted = support_mass.float() * kl
    return weighted.mean(), {"conditional_kl": kl.detach(), "mass_weighted_kl": weighted.detach()}


def _analytic_target(log_reference: Tensor, raw_rewards: Tensor, normalization: FrozenRewardScale, beta: float) -> Tensor:
    if beta <= 0:
        raise ValueError("distillation beta must be positive")
    return torch.log_softmax(log_reference.float() + normalization.normalize(raw_rewards) / float(beta), -1)


def train_residual_head(
    candidate_features: Tensor,
    log_reference_conditional: Tensor,
    raw_candidate_rewards: Tensor,
    support_mass: Tensor,
    normalization: FrozenRewardScale,
    *,
    objective: PolicyObjective,
    beta: float,
    seed: int = 17,
    updates: int = 100,
    batch_contexts: int = 16,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
    samples_per_context: int = 4,
    device: str | torch.device = "cpu",
) -> tuple[ResidualPolicyHead, dict[str, object]]:
    if objective not in {"sampled_loo", "exact_expected_reward", "analytic_q_distillation"}:
        raise ValueError(f"unsupported policy objective: {objective}")
    if beta <= 0 or updates < 1 or batch_contexts < 1 or samples_per_context < 2:
        raise ValueError("training needs beta>0, positive updates/batch and at least two LOO samples")
    if candidate_features.ndim != 3 or log_reference_conditional.shape != candidate_features.shape[:2]:
        raise ValueError("candidate feature/reference tensor shapes do not align")
    if raw_candidate_rewards.shape != log_reference_conditional.shape or support_mass.shape != (len(candidate_features),):
        raise ValueError("candidate rewards/support mass must align with reference contexts")
    if not bool(torch.isfinite(candidate_features).all()) or not bool(torch.isfinite(raw_candidate_rewards).all()):
        raise ValueError("head training inputs must be finite")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    selected_device = torch.device(device)
    x = candidate_features.detach().to(selected_device, dtype=torch.float32)
    ref = log_reference_conditional.detach().to(selected_device, dtype=torch.float32)
    raw = raw_candidate_rewards.detach().to(selected_device, dtype=torch.float32)
    mass = support_mass.detach().to(selected_device, dtype=torch.float32)
    head = ResidualPolicyHead(ResidualHeadSpec(input_dim=x.shape[-1], hidden_dim=hidden_dim)).to(selected_device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=selected_device).manual_seed(int(seed) + 1)
    if objective == "analytic_q_distillation":
        target = _analytic_target(ref, raw, normalization, beta)
    else:
        target = None
    step_losses: list[float] = []
    started = time.perf_counter()
    for update in range(updates):
        size = min(batch_contexts, len(x))
        indices = torch.randperm(len(x), generator=generator, device=selected_device)[:size]
        xb, rb, yb, mb = x[indices], ref[indices], raw[indices], mass[indices]
        log_q, _ = residual_log_policy(rb, xb, head)
        if objective == "exact_expected_reward":
            loss, _ = exact_fixed_support_loss(log_q, rb, yb, mb, normalization, beta)
        elif objective == "analytic_q_distillation":
            loss, _ = distillation_loss(log_q, target[indices], mb)
        else:
            q = log_q.detach().exp()
            actions = torch.multinomial(q, samples_per_context, replacement=True, generator=generator)
            sampled_log_q = log_q.gather(1, actions)
            # The sampled API works over this compact K-sized support, so local
            # positions are encoded from first_catalog_id=3.
            action_ids = actions + 3
            conditional_kl = _conditional_kl(log_q, rb)
            full_kl = mb * conditional_kl
            loss, _ = fixed_support_sampled_policy_loss(
                sampled_log_q, yb, action_ids, full_kl, mb, normalization, beta, first_catalog_id=3,
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite {objective} loss at update {update + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        step_losses.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - started
    head.eval()
    with torch.no_grad():
        final_log_q, final_delta = residual_log_policy(ref, x, head)
        if objective == "analytic_q_distillation":
            _, final_stats = distillation_loss(final_log_q, target, mass)
            final_kl = float(final_stats["conditional_kl"].mean().cpu())
        else:
            final_loss, final_stats = exact_fixed_support_loss(final_log_q, ref, raw, mass, normalization, beta)
            final_kl = float(_conditional_kl(final_log_q, ref).mean().cpu())
    metadata: dict[str, object] = {
        "objective": objective,
        "beta": float(beta),
        "seed": int(seed),
        "updates": int(updates),
        "batch_contexts": int(batch_contexts),
        "samples_per_context": int(samples_per_context) if objective == "sampled_loo" else None,
        "hidden_dim": int(hidden_dim),
        "input_dim": int(x.shape[-1]),
        "parameter_count": head.parameter_count(),
        "train_contexts": int(len(x)),
        "support_size": int(x.shape[1]),
        "elapsed_training_seconds": elapsed,
        "first_5_step_loss_mean": float(sum(step_losses[:5]) / len(step_losses[:5])),
        "last_5_step_loss_mean": float(sum(step_losses[-5:]) / len(step_losses[-5:])),
        "final_mean_abs_delta": float(final_delta.abs().mean().cpu()),
        "final_mean_conditional_kl_to_reference": final_kl,
        "actor_weights_frozen": True,
        "reward_model_role": "frozen observational evaluator; not ground truth",
        "normalization_version": normalization.version,
    }
    return head, metadata


def save_head_checkpoint(path: str | Path, head: ResidualPolicyHead, metadata: dict[str, object]) -> str:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite panel_tools policy checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "panel_tools_residual_policy_head_v1", "state_dict": head.state_dict(), "metadata": metadata}, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return digest
