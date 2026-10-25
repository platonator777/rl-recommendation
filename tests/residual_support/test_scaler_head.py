import numpy as np
import pytest
import torch

from tlab.residual_support.head import ResidualHeadSpec, ResidualPolicyHead
from tlab.residual_support.objectives import FrozenRewardScale, exact_support_statistics
from tlab.residual_support.scaler import TrainOnlyFeatureScaler


def test_scaler_is_fitted_only_on_train_and_keeps_zero_distinct_from_missing():
    train = np.asarray([
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[2.0, 0.0, 1.0], [3.0, 1.0, 1.0]],
    ], dtype=np.float32)
    names = ("train_popularity_count", "duration_missing", "train_duration_seconds")
    scaler = TrainOnlyFeatureScaler.fit(train, names)
    saved = scaler.digest()
    dev = np.asarray([[[1e9, 1.0, 1e9]]], dtype=np.float32)
    scaled = scaler.transform(dev, names)
    assert np.isfinite(scaled).all()
    assert scaler.digest() == saved
    assert scaler.fit_contexts == 2
    with pytest.raises(ValueError, match="names/order"):
        scaler.transform(dev[..., ::-1], names[::-1])
    zero_and_missing = np.asarray([[[0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=np.float32)
    out = scaler.transform(zero_and_missing, names)
    assert not np.array_equal(out[0, 0], out[1, 0])


def test_zero_initialized_head_is_exact_reference_and_gradients_are_finite():
    torch.manual_seed(17)
    head = ResidualPolicyHead(ResidualHeadSpec(input_dim=5, hidden_dim=8))
    x = torch.randn(3, 7, 5)
    reference = torch.log_softmax(torch.randn(3, 7), dim=-1)
    delta = head(x)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.allclose(torch.log_softmax(reference + delta, dim=-1), reference, atol=1e-7)
    loss = -(torch.softmax(reference, dim=-1) * torch.log_softmax(reference + head(x), dim=-1)).sum(-1).mean()
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters())
    assert torch.isfinite(head.network[0].weight.grad).all()


def test_head_learns_a_representable_teacher_and_approaches_exact_optimum():
    torch.manual_seed(31)
    n_contexts, n_actions = 64, 5
    item_signal = torch.linspace(-1.5, 1.5, n_actions)
    features = item_signal[None, :, None].repeat(n_contexts, 1, 1)
    reference = torch.log_softmax(torch.tensor([[0.7, 0.2, -0.1, -0.3, -0.5]]).repeat(n_contexts, 1), dim=-1)
    rewards = (0.6 * item_signal)[None, :].repeat(n_contexts, 1)
    mass = torch.ones(n_contexts)
    scale = FrozenRewardScale(mean=0.0, std=1.0)
    teacher = exact_support_statistics(reference, reference, rewards, mass, scale, beta=1.0)
    target_logq = teacher["analytic_target_log_policy"]

    head = ResidualPolicyHead(ResidualHeadSpec(input_dim=1, hidden_dim=12))
    optimizer = torch.optim.Adam(head.parameters(), lr=0.03)
    initial = float(exact_support_statistics(reference, reference, rewards, mass, scale, beta=1.0)
                    ["analytic_objective_gap"].mean())
    for _ in range(300):
        logq = torch.log_softmax(reference + head(features), dim=-1)
        target = exact_support_statistics(logq, reference, rewards, mass, scale, beta=1.0,
                                          log_target=target_logq)
        loss = target["kl_target_to_policy_conditional"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    logq = torch.log_softmax(reference + head(features), dim=-1)
    final = exact_support_statistics(logq, reference, rewards, mass, scale, beta=1.0,
                                     log_target=target_logq)
    assert float(final["kl_target_to_policy_conditional"].mean()) < 0.005
    assert float(final["analytic_objective_gap"].mean()) < initial * 0.02
