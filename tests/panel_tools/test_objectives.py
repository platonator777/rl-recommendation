import torch

from tlab.panel_tools.objectives import (
    FrozenRewardScale,
    analytic_policy,
    exact_reward_kl_objective,
    sampled_policy_loss,
)


def test_one_frozen_normalization_is_shared_by_sampled_and_exact_actions():
    raw = torch.tensor([[1.0, 3.0, 5.0]])
    actions = torch.tensor([[3, 5]])
    scale = FrozenRewardScale(mean=2.0, std=2.0)
    sampled = scale.sample(raw, actions)
    exact = scale.normalize(raw)
    assert torch.equal(sampled["normalized_catalog_rewards"], exact)
    assert torch.equal(sampled["sampled_normalized_rewards"], exact[:, [0, 2]])


def test_analytic_policy_matches_reference_at_large_beta_and_reward_argmax_at_zero():
    log_ref = torch.log(torch.tensor([[0.2, 0.3, 0.5]]))
    reward = torch.tensor([[0.0, 1.0, 3.0]])
    scale = FrozenRewardScale(mean=1.0, std=2.0)
    q_large = analytic_policy(log_ref, reward, scale, beta=1e8)
    assert torch.allclose(q_large, log_ref.exp(), atol=1e-7)
    q_zero = analytic_policy(log_ref, reward, scale, beta=0)
    assert q_zero.tolist() == [[0.0, 0.0, 1.0]]


def test_exact_expected_reward_and_kl_are_finite_and_differentiable():
    logits = torch.tensor([[0.1, -0.4, 0.7]], requires_grad=True)
    log_policy = logits.log_softmax(-1)
    log_ref = torch.log(torch.tensor([[0.3, 0.3, 0.4]]))
    raw = torch.tensor([[0.2, 0.6, 0.9]])
    objective, stats = exact_reward_kl_objective(log_policy, log_ref, raw, FrozenRewardScale(0.4, 0.2), 0.7)
    objective.sum().backward()
    assert torch.isfinite(objective).all()
    assert torch.isfinite(stats["kl"]).all()
    assert torch.isfinite(logits.grad).all()


def test_sampled_loo_gradient_expectation_matches_exact_gradient_on_tiny_catalog():
    logits = torch.tensor([[0.2, -0.1, 0.5]], requires_grad=True)
    log_policy = logits.log_softmax(-1)
    policy = log_policy.detach().exp()[0]
    raw = torch.tensor([[0.0, 2.0, 5.0]])
    scale = FrozenRewardScale(mean=1.5, std=2.0)
    normalization = scale.normalize(raw)
    expected_sampled_loss = torch.zeros(())
    for first in range(3):
        for second in range(3):
            actions = torch.tensor([[3 + first, 3 + second]])
            sampled_logp = log_policy[:, [first, second]]
            loss, _ = sampled_policy_loss(
                sampled_logp,
                raw,
                actions,
                exact_kl_per_context=torch.zeros(1),
                normalization=scale,
                beta=0.0,
            )
            probability = (policy[first] * policy[second]).detach()
            expected_sampled_loss = expected_sampled_loss + probability * loss
    sampled_gradient = torch.autograd.grad(expected_sampled_loss, logits, retain_graph=True)[0]
    exact_expected_reward = (log_policy.exp() * normalization).sum()
    exact_gradient = torch.autograd.grad(exact_expected_reward, logits)[0]
    assert torch.allclose(sampled_gradient, -exact_gradient, atol=1e-7, rtol=1e-6)


def test_analytic_policy_maximizes_the_frozen_reward_kl_objective():
    log_ref = torch.log(torch.tensor([[0.2, 0.3, 0.5]]))
    raw = torch.tensor([[0.2, 0.6, 0.9]])
    scale = FrozenRewardScale(0.4, 0.2)
    beta = 0.7
    q = analytic_policy(log_ref, raw, scale, beta)

    def objective(probability):
        logp = probability.clamp_min(torch.finfo(torch.float32).tiny).log()
        kl = (probability * (logp - log_ref)).sum(-1)
        return (probability * scale.normalize(raw)).sum(-1) - beta * kl

    alternatives = [torch.tensor([[1 / 3, 1 / 3, 1 / 3]]), log_ref.exp(), torch.tensor([[0.8, 0.1, 0.1]])]
    assert all(torch.all(objective(q) >= objective(candidate) - 1e-7) for candidate in alternatives)
