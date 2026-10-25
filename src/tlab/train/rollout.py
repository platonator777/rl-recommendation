from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from tlab.models.actor import ItemGPT2Actor
from tlab.train.objectives import full_categorical_kl, grpo_clipped_loss, group_advantages, reinforce_loss


@dataclass
class OneStepRollout:
    actions: Tensor
    old_log_probs: Tensor
    rewards: Tensor
    advantages: Tensor
    reference_log_probs: Tensor
    old_policy_log_probs: Tensor | None = None


@dataclass
class HorizonRollout:
    states: tuple[tuple[Tensor, Tensor], ...]
    actions: Tensor  # [contexts, group, horizon]
    old_log_probs: Tensor  # [contexts, group, horizon]
    rewards: Tensor  # [contexts, group, horizon]
    returns: Tensor  # [contexts, group]
    advantages: Tensor  # [contexts, group]
    reference_log_probs: tuple[Tensor, ...]  # each [contexts * group, catalog]


def collect_one_step_rollout(
    actor: ItemGPT2Actor,
    reference: ItemGPT2Actor,
    input_ids: Tensor,
    attention_mask: Tensor,
    reward_function: Callable[[Tensor], Tensor],
    group_size: int = 8,
    normalize_advantages: bool = True,
    centered_correction: bool = False,
) -> OneStepRollout:
    """Sample IID full-catalog actions, preserving duplicates and actual log-probs."""
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    actor.eval()  # gradients remain enabled in subsequent update calls
    reference.eval()
    with torch.no_grad():
        actor_log_dist = actor.next_action_log_probs(input_ids, attention_mask)
        reference_log_dist = reference.next_action_log_probs(input_ids, attention_mask)
        probabilities = actor_log_dist.exp()
        # Match the H>1 path: each trajectory draws one action from a flattened
        # context-major batch, so H=1 and the first H=5 step share RNG semantics.
        trajectory_probabilities = probabilities[:, None, :].expand(-1, group_size, -1).reshape(
            -1, probabilities.shape[-1],
        )
        actions = torch.multinomial(trajectory_probabilities, num_samples=1, replacement=True).reshape(
            probabilities.shape[0], group_size,
        )
        old_log_probs = actor_log_dist.gather(-1, actions).detach()
        rewards = reward_function(actions).to(actor_log_dist.device).float().detach()
        if rewards.shape != actions.shape:
            raise ValueError("reward_function must return one reward per sampled action")
        advantages = group_advantages(
            rewards, normalize=normalize_advantages, centered_correction=centered_correction,
        )
    return OneStepRollout(
        actions, old_log_probs, rewards, advantages,
        reference_log_dist.detach(), actor_log_dist.detach(),
    )


def grpo_update(
    actor: ItemGPT2Actor,
    input_ids: Tensor,
    attention_mask: Tensor,
    rollout: OneStepRollout,
    optimizer: torch.optim.Optimizer,
    beta: float,
    clip_epsilon: float = 0.2,
    gradient_clip: float = 1.0,
    update_epochs: int = 1,
    minibatch_contexts: int | None = None,
    max_kl_to_old: float | None = None,
) -> dict[str, float]:
    if update_epochs < 1 or (minibatch_contexts is not None and minibatch_contexts < 1):
        raise ValueError("GRPO reuse epochs and minibatch contexts must be positive")
    if max_kl_to_old is not None and max_kl_to_old <= 0:
        raise ValueError("max_kl_to_old must be positive when configured")
    if update_epochs > 1 and rollout.old_policy_log_probs is None:
        raise ValueError("GRPO reuse requires the frozen old-policy full distributions")
    actor.eval()
    n_contexts = int(input_ids.shape[0])
    context_batch = int(minibatch_contexts or n_contexts)
    records: list[dict[str, float]] = []
    post_update_kl_values: list[float] = []
    optimizer_steps = 0
    epochs_completed = 0
    stopped_for_kl = False
    for epoch in range(update_epochs):
        epoch_finished = True
        for start in range(0, n_contexts, context_batch):
            stop = min(start + context_batch, n_contexts)
            minibatch_ids = input_ids[start:stop]
            minibatch_mask = attention_mask[start:stop]
            actions = rollout.actions[start:stop]
            old_sampled = rollout.old_log_probs[start:stop]
            advantages = rollout.advantages[start:stop]
            reference_dist = rollout.reference_log_probs[start:stop]
            old_policy_dist = (
                rollout.old_policy_log_probs[start:stop]
                if rollout.old_policy_log_probs is not None else None
            )
            new_log_dist = actor.next_action_log_probs(minibatch_ids, minibatch_mask)
            new_log_probs = new_log_dist.gather(-1, actions)
            kl = full_categorical_kl(new_log_dist, reference_dist)
            loss, metrics = grpo_clipped_loss(
                new_log_probs, old_sampled, advantages, kl, beta, clip_epsilon,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip)
            optimizer.step()
            optimizer_steps += 1

            with torch.no_grad():
                post_log_dist = actor.next_action_log_probs(minibatch_ids, minibatch_mask)
                post_kl_old = (
                    full_categorical_kl(post_log_dist, old_policy_dist).mean()
                    if old_policy_dist is not None else torch.zeros((), device=post_log_dist.device)
                )
                post_update_kl_values.append(float(post_kl_old))
            records.append({
                "loss": float(loss.detach()),
                "kl_to_reference": float(kl.mean().detach()),
                "entropy": float((-(new_log_dist.exp() * new_log_dist).nan_to_num(0.0)).sum(dim=-1).mean().detach()),
                "grad_norm": float(grad_norm),
                "clip_fraction": float(metrics["clip_fraction"]),
                "approx_kl_to_old": float(metrics["approx_kl_to_old"]),
                "ratio_mean": float(metrics["ratio_mean"]),
            })
            if max_kl_to_old is not None and float(post_kl_old) > max_kl_to_old:
                stopped_for_kl = True
                epoch_finished = False
                break
        if epoch_finished:
            epochs_completed += 1
        else:
            break
    if not records:
        raise RuntimeError("GRPO reuse did not perform an optimizer step")
    mean = lambda key: sum(row[key] for row in records) / len(records)
    return {
        "loss": mean("loss"),
        "reward_mean": float(rollout.rewards.mean()),
        "group_std_mean": float(rollout.rewards.std(dim=-1, correction=0).mean()),
        "kl_to_reference": mean("kl_to_reference"),
        "entropy": mean("entropy"),
        "grad_norm": mean("grad_norm"),
        "clip_fraction": mean("clip_fraction"),
        "approx_kl_to_old": mean("approx_kl_to_old"),
        "ratio_mean": mean("ratio_mean"),
        "post_update_kl_to_old": post_update_kl_values[-1],
        "max_post_update_kl_to_old": max(post_update_kl_values),
        "optimizer_steps": optimizer_steps,
        "update_epochs_completed": epochs_completed,
        "early_stop_kl_guard_triggered": float(stopped_for_kl),
    }


def reinforce_update(
    actor: ItemGPT2Actor,
    input_ids: Tensor,
    attention_mask: Tensor,
    rollout: OneStepRollout,
    optimizer: torch.optim.Optimizer,
    beta: float,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    actor.eval()
    new_log_dist = actor.next_action_log_probs(input_ids, attention_mask)
    sampled_log_probs = new_log_dist.gather(-1, rollout.actions)
    kl = full_categorical_kl(new_log_dist, rollout.reference_log_probs)
    loss = reinforce_loss(sampled_log_probs, rollout.rewards, kl, beta)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "reward_mean": float(rollout.rewards.mean()),
        "kl_to_reference": float(kl.mean().detach()),
        "grad_norm": float(grad_norm),
    }


def _collate_state_histories(histories: list[list[int]], device: torch.device) -> tuple[Tensor, Tensor]:
    width = max(len(history) for history in histories)
    ids = torch.zeros((len(histories), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, history in enumerate(histories):
        ids[index, :len(history)] = torch.tensor(history, dtype=torch.long, device=device)
        mask[index, :len(history)] = 1
    return ids, mask


def collect_horizon_rollout(
    actor: ItemGPT2Actor,
    reference: ItemGPT2Actor,
    real_histories: list[list[int]],
    reward_function: Callable[[list[list[int]], Tensor, int], Tensor],
    group_size: int = 8,
    horizon: int = 5,
    history_length: int = 200,
    normalize_advantages: bool = True,
    centered_correction: bool = False,
    seed: int | None = None,
) -> HorizonRollout:
    """Collect naive-append item trajectories; appended actions are synthetic states."""
    if not real_histories or group_size < 2 or horizon < 1:
        raise ValueError("rollout needs contexts, group_size >= 2, and a positive horizon")
    device = next(actor.parameters()).device
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    context_count = len(real_histories)
    states = [list(history) for history in real_histories for _ in range(group_size)]
    actions_by_step: list[Tensor] = []
    old_log_probs_by_step: list[Tensor] = []
    rewards_by_step: list[Tensor] = []
    state_batches: list[tuple[Tensor, Tensor]] = []
    reference_distributions: list[Tensor] = []
    actor.eval()
    reference.eval()

    with torch.no_grad():
        for step in range(horizon):
            input_ids, attention_mask = _collate_state_histories(states, device)
            actor_log_dist = actor.next_action_log_probs(input_ids, attention_mask)
            reference_log_dist = reference.next_action_log_probs(input_ids, attention_mask)
            actions = torch.multinomial(actor_log_dist.exp(), num_samples=1, generator=generator).squeeze(-1)
            old_log_probs = actor_log_dist.gather(-1, actions[:, None]).squeeze(-1)
            rewards = reward_function(states, actions, step).to(device=device, dtype=torch.float32).detach()
            if rewards.shape != actions.shape:
                raise ValueError("reward_function must return one reward per trajectory at each step")
            state_batches.append((input_ids.detach(), attention_mask.detach()))
            actions_by_step.append(actions)
            old_log_probs_by_step.append(old_log_probs)
            rewards_by_step.append(rewards)
            reference_distributions.append(reference_log_dist.detach())
            next_states: list[list[int]] = []
            for history, action in zip(states, actions.tolist()):
                content = history[1:] + [int(action)]
                content = content[-max(0, history_length - 1):] if history_length > 1 else []
                next_states.append([1, *content])
            states = next_states

    actions_tensor = torch.stack(actions_by_step, dim=-1).reshape(context_count, group_size, horizon)
    old_log_probs_tensor = torch.stack(old_log_probs_by_step, dim=-1).reshape(context_count, group_size, horizon)
    rewards_tensor = torch.stack(rewards_by_step, dim=-1).reshape(context_count, group_size, horizon)
    returns = rewards_tensor.sum(dim=-1)
    advantages = group_advantages(
        returns, normalize=normalize_advantages, centered_correction=centered_correction,
    )
    return HorizonRollout(
        states=tuple(state_batches),
        actions=actions_tensor,
        old_log_probs=old_log_probs_tensor,
        rewards=rewards_tensor,
        returns=returns,
        advantages=advantages,
        reference_log_probs=tuple(reference_distributions),
    )


def horizon_grpo_update(
    actor: ItemGPT2Actor,
    rollout: HorizonRollout,
    optimizer: torch.optim.Optimizer,
    beta: float,
    clip_epsilon: float = 0.2,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    """Update from trajectory returns, accumulating gradients across horizon steps.

    The effective batch remains complete trajectories from each context.  The
    backward pass is split by step so activations for all horizon steps are not
    kept on device at once; optimizer state is updated once after all steps.
    """
    actor.eval()
    optimizer.zero_grad(set_to_none=True)
    policy_loss_value = 0.0
    kl_value = 0.0
    clip_value = 0.0
    for step, ((input_ids, attention_mask), reference_log_dist) in enumerate(
        zip(rollout.states, rollout.reference_log_probs)
    ):
        new_log_dist = actor.next_action_log_probs(input_ids, attention_mask)
        flat_actions = rollout.actions[:, :, step].reshape(-1)
        flat_old = rollout.old_log_probs[:, :, step].reshape(-1)
        flat_advantage = rollout.advantages[:, :, None].expand_as(rollout.actions)[:, :, step].reshape(-1)
        new_log_probs = new_log_dist.gather(-1, flat_actions[:, None]).squeeze(-1)
        ratio = torch.exp(new_log_probs.float() - flat_old.detach().float())
        advantage = flat_advantage.detach().float()
        objective = torch.minimum(
            ratio * advantage,
            ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage,
        )
        policy_loss = -objective.mean()
        kl_per_action = full_categorical_kl(new_log_dist, reference_log_dist).mean()
        step_loss = policy_loss + beta * kl_per_action
        step_loss.backward()
        policy_loss_value += float(policy_loss.detach())
        kl_value += float(kl_per_action.detach())
        clip_value += float(((ratio - 1.0).abs() > clip_epsilon).float().mean().detach())
        del new_log_dist, flat_actions, flat_old, flat_advantage
        del new_log_probs, ratio, advantage, objective, policy_loss, kl_per_action, step_loss

    grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip)
    optimizer.step()
    horizon = rollout.actions.shape[-1]
    return {
        "loss": policy_loss_value + beta * kl_value,
        "total_return_mean": float(rollout.returns.mean()),
        "reward_per_action": float(rollout.rewards.mean()),
        "kl_per_action": kl_value / horizon,
        "clip_fraction": clip_value / horizon,
        "zero_variance_groups": float((rollout.returns.std(dim=-1, correction=0) < 1e-4).float().mean()),
        "grad_norm": float(grad_norm),
    }
