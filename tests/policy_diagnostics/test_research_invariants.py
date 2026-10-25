from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path

import yaml

import numpy as np
import torch

from tlab.panel_tools.statistics import paired_user_bootstrap
from tlab.residual_support.features import POLICY_FEATURE_NAMES, RM_FEATURE_NAMES
from tlab.residual_support.head import ResidualHeadSpec, ResidualPolicyHead
from tlab.residual_support.objectives import FrozenRewardScale, exact_support_statistics, sampled_loo_loss
from tlab.policy_diagnostics.a2_detail import _softmax
from tlab.policy_diagnostics.c_rm import _affinity_rows
from tlab.policy_diagnostics.complete_evaluation import _mean_maps
from tlab.policy_diagnostics.evaluate_policy_branch import _effective_endpoint


ROOT = Path(__file__).resolve().parents[2]


def test_variance_decomposition_identity_and_constant_reward():
    r = np.array([[0.0, 1.0], [2.0, 4.0]])
    w = np.array([[0.5, 0.5], [0.25, 0.75]])
    mu = (w * r).sum(axis=1)
    within = (w * (r - mu[:, None]) ** 2).sum(axis=1).mean()
    between = np.var(mu)
    total = np.mean((w * r * r).sum(axis=1)) - np.mean(mu) ** 2
    assert np.isclose(within + between, total)
    constant = np.full((2, 3), 0.7)
    assert np.allclose(np.var(constant, axis=1), 0)


def test_analytic_gap_identity_and_kl_direction():
    ref = torch.log_softmax(torch.tensor([[0.2, -0.4, 0.1]]), dim=-1)
    reward = torch.tensor([[0.05, 0.8, 0.4]])
    qlog = torch.log_softmax(torch.tensor([[0.5, -0.2, -0.1]]), dim=-1)
    mass = torch.tensor([0.63])
    beta = 0.3
    stats = exact_support_statistics(qlog, ref, reward, mass, FrozenRewardScale(.2, .4), beta)
    logstar = stats["analytic_target_log_policy"]
    q, qstar = qlog.exp(), logstar.exp()
    kl_q_star = torch.sum(q * (qlog - logstar), dim=-1)
    assert torch.allclose(stats["analytic_objective_gap"], beta * mass * kl_q_star, atol=1e-6)
    kl_q_ref = torch.sum(q * (qlog - ref), dim=-1)
    kl_ref_q = torch.sum(ref.exp() * (ref - qlog), dim=-1)
    assert not torch.allclose(kl_q_ref, kl_ref_q)
    tiny = torch.log_softmax(torch.tensor([[0.0, -80.0, -120.0]]), dim=-1)
    tiny_stats = exact_support_statistics(tiny, ref, reward, mass, FrozenRewardScale(.2, .4), beta)
    assert torch.isfinite(tiny_stats["analytic_objective_gap"]).all()


def test_sampled_loo_gradient_converges_to_exact_on_two_actions():
    scale = FrozenRewardScale(0.0, 1.0)
    ref = torch.tensor([[0.0, 0.0]])
    raw = torch.tensor([[0.1, 0.9]])
    mass = torch.tensor([0.7])
    beta = 0.3
    logits = torch.tensor([[0.25, -0.15]], requires_grad=True)
    logq = torch.log_softmax(logits, dim=-1)
    exact_loss = -exact_support_statistics(logq, ref, raw, mass, scale, beta)["support_weighted_objective"].mean()
    exact_grad = torch.autograd.grad(exact_loss, logits)[0]
    sampled_logits = torch.tensor([[0.25, -0.15]], requires_grad=True)
    sampled_logq = torch.log_softmax(sampled_logits, dim=-1)
    gen = torch.Generator().manual_seed(8831)
    sampled_loss, _ = sampled_loo_loss(sampled_logq, ref, raw, mass, scale, beta, 20000, gen)
    sampled_grad = torch.autograd.grad(sampled_loss, sampled_logits)[0]
    assert torch.allclose(sampled_grad, exact_grad, atol=0.025, rtol=0.08)


def test_zero_residual_head_is_reference_policy():
    torch.manual_seed(7)
    head = ResidualPolicyHead(ResidualHeadSpec(input_dim=4, hidden_dim=8))
    features = torch.randn(3, 5, 4)
    reference = torch.log_softmax(torch.randn(3, 5), dim=-1)
    assert torch.equal(head(features), torch.zeros(3, 5))
    assert torch.allclose(torch.softmax(reference + head(features), dim=-1), reference.exp())
    assert len(POLICY_FEATURE_NAMES) == 45
    assert POLICY_FEATURE_NAMES.index("item_embedding_cosine_last") >= len(RM_FEATURE_NAMES)


def test_fixed_tail_delta_cancels_exactly():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.7, 0.3])
    ref = np.array([0.4, 0.6])
    support_mass = 0.8
    reward = np.array([0.1, 0.9])
    tail_reward_mean = 0.4
    full_q = support_mass * np.dot(q, reward) + (1 - support_mass) * tail_reward_mean
    full_ref = support_mass * np.dot(ref, reward) + (1 - support_mass) * tail_reward_mean
    assert np.isclose(full_q - full_ref, support_mass * (np.dot(q, reward) - np.dot(ref, reward)))


def test_affinity_is_prefix_only_and_strict_additive_columns_have_no_interaction():
    user = SimpleNamespace(tokens=np.array([3, 4, 5, 6, 7]), ratio_raw=np.array([.8, .2, .9, .1, .7]))
    embedding = np.eye(5, dtype=np.float32)
    before = _affinity_rows(SimpleNamespace(users=[user]), [(0, 4)], np.array([0]), embedding)
    user.tokens[4] = 2
    user.ratio_raw[4] = np.nan
    after = _affinity_rows(SimpleNamespace(users=[user]), [(0, 4)], np.array([0]), embedding)
    assert np.array_equal(before, after)
    additive_names = [*RM_FEATURE_NAMES[:14], *RM_FEATURE_NAMES[29:35]]
    assert len(additive_names) == 20
    assert not any(name.startswith("personal_repeat") for name in additive_names)


def test_user_and_seed_aggregation_do_not_multiply_user_count():
    seed_maps = [{1: .1, 2: .2}, {1: .3, 2: .4}, {1: .5, 2: .6}]
    merged = _mean_maps(seed_maps)
    assert set(merged) == {1, 2}
    assert len(merged) == 2
    assert np.isclose(merged[1], .3)
    interval = paired_user_bootstrap(merged, {1: .1, 2: .2}, n_bootstrap=100, seed=9)
    assert interval["n_users"] == 2


def test_centered_external_audit_rank_helpers_are_finite():
    p = _softmax(np.array([[0.1, 0.2, 0.3]]))
    assert np.isclose(p.sum(), 1)
    assert np.all(p >= 0)


def test_branch_configs_change_only_registered_sample_count_between_sampled_arms():
    left = yaml.safe_load((ROOT / "configs/policy_diagnostics/sample_count_sampled4.yaml").read_text(encoding="utf-8"))
    right = yaml.safe_load((ROOT / "configs/policy_diagnostics/sample_count_sampled16.yaml").read_text(encoding="utf-8"))
    assert left["train"]["samples_per_context"] == 4
    assert right["train"]["samples_per_context"] == 16
    left_train, right_train = dict(left["train"]), dict(right["train"])
    left_train.pop("samples_per_context")
    right_train.pop("samples_per_context")
    assert left_train == right_train
    assert left["outputs"]["run_root"].startswith("artifacts/policy_diagnostics/")
    assert right["outputs"]["run_root"].startswith("artifacts/policy_diagnostics/")
    exact = yaml.safe_load((ROOT / "configs/policy_diagnostics/sample_count_exact.yaml").read_text(encoding="utf-8"))
    assert exact["train"]["seed_values"] == [17, 42, 73]
    assert exact["train"]["max_updates"] == left["train"]["max_updates"] == 6000
    registration = json.loads((ROOT / "protocol/diagnostics/branch_registration.json").read_text(encoding="utf-8"))
    assert registration["fit_budget"]["expected_runs"] == 9
    assert registration["branch_id"] == "sample_count_4_vs_16_with_exact_control"


def test_best_identity_fallback_is_reported_separately_from_trained_endpoint():
    assert _effective_endpoint("best", 0) == "identity_step0"
    assert _effective_endpoint("best", 500) == "best"
    assert _effective_endpoint("last", 0) == "last"

