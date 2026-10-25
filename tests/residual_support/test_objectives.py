import torch

from tlab.residual_support.objectives import FrozenRewardScale, exact_support_statistics, sampled_loo_loss


def test_analytic_policy_is_exact_optimum_and_gap_has_correct_sign():
    reference = torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float32))
    rewards = torch.tensor([[0.7, 0.1]], dtype=torch.float32)
    mass = torch.tensor([0.73])
    scale = FrozenRewardScale(mean=0.4, std=0.2)
    beta = 0.7
    stats = exact_support_statistics(reference, reference, rewards, mass, scale, beta)
    analytic = stats["analytic_target_log_policy"]
    optimum = exact_support_statistics(analytic, reference, rewards, mass, scale, beta)
    assert torch.allclose(optimum["analytic_objective_gap"], torch.zeros(1), atol=2e-6)
    assert torch.allclose(optimum["support_weighted_objective"], optimum["analytic_support_weighted_objective"], atol=2e-6)
    assert bool((stats["analytic_objective_gap"] >= 0).all())
    assert torch.allclose(stats["kl_policy_to_reference_mass_weighted"], mass * stats["kl_policy_to_reference_conditional"])


def test_distillation_kl_direction_is_analytic_target_to_policy():
    ref = torch.log(torch.tensor([[0.6, 0.4]], dtype=torch.float32))
    rewards = torch.tensor([[1.0, 0.0]])
    scale = FrozenRewardScale(mean=0.5, std=0.25)
    target = exact_support_statistics(ref, ref, rewards, torch.ones(1), scale, 0.5)["analytic_target_log_policy"]
    policy = torch.log(torch.tensor([[0.7, 0.3]]))
    stats = exact_support_statistics(policy, ref, rewards, torch.ones(1), scale, 0.5, log_target=target)
    direct = (target.exp() * (target - policy)).sum(-1)
    reverse = (policy.exp() * (policy - target)).sum(-1)
    assert torch.allclose(stats["kl_target_to_policy_conditional"], direct)
    assert not torch.allclose(direct, reverse)


def test_sampled_loo_gradient_approaches_exact_policy_gradient():
    theta = torch.tensor([0.2, 0.0], requires_grad=True)
    reference_one = torch.log(torch.tensor([[0.6, 0.4]], dtype=torch.float32))
    logq_one = torch.log_softmax(reference_one + theta, dim=-1)
    repetitions = 8000
    reference = reference_one.repeat(repetitions, 1)
    logq = logq_one.repeat(repetitions, 1)
    rewards = torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(repetitions, 1)
    mass = torch.ones(repetitions)
    scale = FrozenRewardScale(mean=0.0, std=1.0)
    generator = torch.Generator().manual_seed(92)
    loss, _ = sampled_loo_loss(logq, reference, rewards, mass, scale, 1e-4, 2, generator)
    sampled_grad = torch.autograd.grad(loss, theta)[0]
    # Rebuild the graph after autograd consumed the sampled estimator graph.
    exact_logq = torch.log_softmax(reference_one + theta, dim=-1)
    exact = exact_support_statistics(exact_logq, reference_one, rewards[:1], mass[:1], scale, 1e-4)
    exact_grad = torch.autograd.grad(-exact["support_weighted_objective"].mean(), theta)[0]
    assert torch.allclose(sampled_grad, exact_grad, atol=0.04, rtol=0.08)
