import json

import numpy as np
import pytest
import torch

from tlab.panel_tools.cache import ShardedArrayCache, panel_toolsCacheKey
from tlab.panel_tools.candidates import (
    fixed_support_analytic_policy,
    fixed_support_kl,
    fixed_support_policy,
    merge_full_catalog_topk,
)
from tlab.panel_tools.objectives import FrozenRewardScale
from tlab.panel_tools.objectives import fixed_support_sampled_policy_loss, sampled_policy_loss


def _log_reference():
    return torch.tensor([[0.1, 0.2, 0.3, 0.4]]).log()


def test_zero_adjustment_preserves_full_policy_and_fixed_tail_mass():
    log_ref = _log_reference()
    support = torch.tensor([[0, 2]])
    policy = fixed_support_policy(log_ref, support, torch.zeros((1, 2)))
    assert torch.allclose(policy.probabilities, log_ref.exp(), atol=1e-7)
    assert torch.allclose(policy.support_mass, torch.tensor([0.4]))
    assert torch.allclose(policy.conditional_probabilities, torch.tensor([[0.25, 0.75]]))
    assert torch.allclose(fixed_support_kl(policy, log_ref, support), torch.zeros(1), atol=1e-7)


def test_fixed_support_matches_manual_exponential_tilt_and_full_kl():
    log_ref = _log_reference()
    support = torch.tensor([[0, 2]])
    adjustment = torch.tensor([[0.0, 1.25]])
    policy = fixed_support_policy(log_ref, support, adjustment)
    ref = log_ref.exp()[0]
    conditional = torch.softmax(torch.stack([ref[0], ref[2]]).log() + adjustment[0], 0)
    expected = ref.clone()
    expected[0], expected[2] = 0.4 * conditional[0], 0.4 * conditional[1]
    assert torch.allclose(policy.probabilities[0], expected, atol=1e-7)
    manual_kl = (expected * (expected.log() - ref.log())).sum()
    assert torch.allclose(fixed_support_kl(policy, log_ref, support)[0], manual_kl, atol=1e-7)
    assert torch.allclose(policy.probabilities[:, [1, 3]], log_ref.exp()[:, [1, 3]])


def test_fixed_support_beta_limits_and_reward_choice():
    log_ref = _log_reference()
    support = torch.tensor([[0, 2]])
    raw = torch.tensor([[0.0, 0.2, 2.0, 0.1]])
    scale = FrozenRewardScale(mean=0.5, std=0.8)
    q_zero = fixed_support_analytic_policy(log_ref, support, raw, scale, beta=0)
    assert torch.equal(q_zero.conditional_probabilities, torch.tensor([[0.0, 1.0]]))
    assert torch.allclose(q_zero.probabilities, torch.tensor([[0.0, 0.2, 0.4, 0.4]]))
    q_large = fixed_support_analytic_policy(log_ref, support, raw, scale, beta=1e8)
    assert torch.allclose(q_large.probabilities, log_ref.exp(), atol=2e-8)


def test_fixed_support_sampled_policy_gradient_is_weighted_by_support_mass():
    raw = torch.tensor([[0.0, 0.2, 2.0, 0.1]])
    sampled_ids = torch.tensor([[3, 5]])
    logq = torch.tensor([[0.3, -0.7]], requires_grad=True)
    scale = FrozenRewardScale(mean=0.4, std=1.2)
    mass = torch.tensor([0.35])
    common = dict(
        raw_catalog_rewards=raw,
        sampled_action_ids=sampled_ids,
        exact_kl_per_context=torch.zeros(1),
        normalization=scale,
        beta=0.0,
    )
    ordinary_loss, _ = sampled_policy_loss(logq, **common)
    fixed_loss, _ = fixed_support_sampled_policy_loss(
        logq,
        raw_catalog_rewards=raw,
        sampled_action_ids=sampled_ids,
        exact_full_kl_per_context=torch.zeros(1),
        support_mass=mass,
        normalization=scale,
        beta=0.0,
    )
    ordinary_gradient = torch.autograd.grad(ordinary_loss, logq, retain_graph=True)[0]
    fixed_gradient = torch.autograd.grad(fixed_loss, logq)[0]
    assert torch.allclose(fixed_loss, mass * ordinary_loss)
    assert torch.allclose(fixed_gradient, mass[:, None] * ordinary_gradient)


def test_topk_merges_unchanged_tail_and_breaks_ties_by_item_id():
    probabilities = torch.tensor([[0.20, 0.35, 0.15, 0.30]])
    ids = torch.tensor([[40, 10, 30, 20]])
    top_ids, top_probs = merge_full_catalog_topk(probabilities, 3, ids)
    assert top_ids.tolist() == [[10, 20, 40]]
    assert torch.allclose(top_probs, torch.tensor([[0.35, 0.30, 0.20]]))

    tied = torch.tensor([[0.5, 0.5, 0.1]])
    tie_ids, _ = merge_full_catalog_topk(tied, 2, torch.tensor([[9, 2, 1]]))
    assert tie_ids.tolist() == [[2, 9]]


def _key(**updates):
    fields = dict(
        protocol="p1",
        split="validation",
        event_ids_sha256="events",
        vocabulary_sha256="vocab",
        actor_sha256="actor",
        reward_model_sha256="rm",
        feature_schema="features-v1",
        normalization_version="frozen_reward_scale_v1",
        candidate_mode="fixed-topk-1000",
    )
    fields.update(updates)
    return panel_toolsCacheKey(**fields)


def test_cache_manifest_hash_binds_all_protocol_inputs_and_roundtrips(tmp_path):
    key = _key()
    cache = ShardedArrayCache.create(tmp_path / "cache", key, [{"name": "scores", "shape": [2, 3]}])
    shard = cache.open_shard("scores", "r+")
    shard[:] = np.arange(6).reshape(2, 3)
    shard.flush()
    del shard

    reopened = ShardedArrayCache(tmp_path / "cache", key)
    assert np.array_equal(reopened.open_shard("scores"), np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError, match="key mismatch"):
        ShardedArrayCache(tmp_path / "cache", _key(actor_sha256="another-actor"))
    with pytest.raises(FileExistsError, match="overwrite"):
        ShardedArrayCache.create(tmp_path / "cache", key, [{"name": "scores", "shape": [2, 3]}])


def test_cache_rejects_modified_schema_even_when_key_matches(tmp_path):
    key = _key()
    ShardedArrayCache.create(tmp_path / "cache", key, [{"name": "a", "shape": [1]}])
    manifest_path = tmp_path / "cache" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "old"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported cache schema"):
        ShardedArrayCache(tmp_path / "cache", key)
