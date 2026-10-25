import torch

from tlab.panel_tools.objectives import FrozenRewardScale
from tlab.panel_tools.policy_head import (
    ResidualHeadSpec,
    ResidualPolicyHead,
    feature_columns_without_feedback,
    residual_log_policy,
)
from tlab.panel_tools.train_policy import exact_fixed_support_loss, train_residual_head


def test_zero_initialized_head_preserves_reference_conditional_policy():
    reference = torch.log_softmax(torch.tensor([[0.2, -0.4, 0.7]]), -1)
    features = torch.randn(1, 3, 5)
    head = ResidualPolicyHead(ResidualHeadSpec(input_dim=5, hidden_dim=32))
    log_q, delta = residual_log_policy(reference, features, head)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.allclose(log_q, reference, atol=1e-7)
    assert head.parameter_count() == (5 * 32 + 32) + (32 + 1)


def test_no_feedback_columns_remove_past_ratio_but_keep_count_duration():
    names = ("item_count_200", "item_ratio_mean_200", "item_recency_steps_200", "log_duration",
             "past_ratio_missing_50", "log_train_popularity", "artist_features_missing_200")
    selected = feature_columns_without_feedback(names)
    assert selected == (0, 2, 3, 5, 6)


def test_exact_fixed_support_objective_matches_manual_mass_weighted_oracle():
    reference = torch.log(torch.tensor([[0.25, 0.75]]))
    logits = torch.tensor([[0.1, -0.3]], requires_grad=True)
    log_q = torch.log_softmax(reference + logits, -1)
    rewards = torch.tensor([[0.0, 1.0]])
    mass = torch.tensor([0.4])
    scale = FrozenRewardScale(mean=0.5, std=0.25)
    beta = 0.3
    loss, stats = exact_fixed_support_loss(log_q, reference, rewards, mass, scale, beta)
    q = log_q.exp()
    reward = scale.normalize(rewards)
    kl = (q * (log_q - reference)).sum(-1)
    expected = (-mass * (q * reward).sum(-1) + beta * mass * kl).mean()
    assert torch.allclose(loss, expected, atol=1e-7)
    assert torch.allclose(stats["full_kl"], mass * kl, atol=1e-7)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_head_train_smoke_keeps_actor_reference_and_uses_shared_scale():
    features = torch.randn(6, 4, 5)
    reference = torch.log_softmax(torch.randn(6, 4), -1)
    rewards = torch.rand(6, 4)
    mass = torch.full((6,), 0.5)
    scale = FrozenRewardScale(mean=0.5, std=0.25)
    head, metadata = train_residual_head(
        features, reference, rewards, mass, scale,
        objective="exact_expected_reward", beta=1.0, seed=17, updates=2,
        batch_contexts=3, hidden_dim=32, device="cpu",
    )
    assert metadata["actor_weights_frozen"] is True
    assert metadata["normalization_version"] == "frozen_reward_scale_v1"
    assert head.parameter_count() > 0
