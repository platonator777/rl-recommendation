import numpy as np
from array import array
from pathlib import Path

from tlab.data.store import SequenceStore, UserSequence
from tlab.panel_tools.data import PanelTarget
from tlab.panel_tools.objectives import FrozenRewardScale
from tlab.panel_tools.policy_experiment import _distribution_from_cache_batch, _evaluate_policy_matrix, _one_sided_signflip_p


def test_cached_analytic_policy_preserves_reference_tail_and_mass():
    reference = np.log(np.asarray([[0.5, 0.3, 0.2]], dtype=np.float32))
    support = np.asarray([[0, 1]], dtype=np.int32)
    mass = np.asarray([0.8], dtype=np.float32)
    rewards = np.asarray([[1.0, 0.0]], dtype=np.float32)
    distribution = _distribution_from_cache_batch(
        reference, support, mass, rewards, FrozenRewardScale(mean=0.5, std=0.25), 1.0,
    )
    assert np.isclose(distribution.sum(), 1.0, atol=1e-6)
    assert np.isclose(distribution[0, 2], 0.2, atol=1e-6)
    assert np.isclose(distribution[0, :2].sum(), 0.8, atol=1e-6)
    assert distribution[0, 0] > distribution[0, 1]


def test_cached_trainable_policy_reconstructs_conditional_distribution():
    reference = np.log(np.asarray([[0.5, 0.3, 0.2]], dtype=np.float32))
    support = np.asarray([[0, 1]], dtype=np.int32)
    mass = np.asarray([0.8], dtype=np.float32)
    rewards = np.asarray([[0.5, 0.5]], dtype=np.float32)
    conditional = np.log(np.asarray([[0.25, 0.75]], dtype=np.float32))
    distribution = _distribution_from_cache_batch(
        reference, support, mass, rewards, FrozenRewardScale(mean=0.5, std=0.25), 1.0,
        log_q_conditional=conditional,
    )
    assert np.allclose(distribution, [[0.2, 0.6, 0.2]], atol=1e-6)


def test_one_sided_signflip_is_deterministic_and_directional():
    positive = _one_sided_signflip_p(np.asarray([0.1, 0.2, 0.3, 0.4]), seed=17, draws=1000)
    repeated = _one_sided_signflip_p(np.asarray([0.1, 0.2, 0.3, 0.4]), seed=17, draws=1000)
    negative = _one_sided_signflip_p(np.asarray([-0.1, -0.2, -0.3, -0.4]), seed=17, draws=1000)
    assert positive == repeated
    assert positive < 0.1
    assert negative > 0.9


def test_policy_matrix_evaluation_uses_full_catalog_and_keeps_misses():
    user = UserSequence(uid=7)
    user.tokens = array("i", [3, 4, 5])
    user.raw_item_ids = array("q", [10, 20, 30])
    user.ratio_raw = array("f", [0.7, 0.2, 0.8])
    user.duration_seconds = array("f", [120.0, 90.0, 30.0])
    store = SequenceStore(users=[user], events=3, tie_blocks=0, tied_events=0)
    target = PanelTarget(
        event_id="event-7", uid=7, user_index=0, position=2, raw_item_id=30, token_id=5,
        ratio=0.8, positive=True, is_new=True, is_repeat=False,
    )
    reference = np.log(np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32))
    arrays = {
        "reference_logp": reference,
        "support_indices": np.asarray([[0, 1]], dtype=np.int32),
        "support_mass": np.asarray([0.7], dtype=np.float32),
        "raw_candidate_rewards": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "candidate_features": np.zeros((1, 2, 1), dtype=np.float32),
    }
    metrics = _evaluate_policy_matrix(
        root=Path("."), panel=[target], val_arrays=arrays,
        val_metadata={"feature_names": ["item_count_200"]}, item_ids=np.asarray([10, 20, 30, 40]),
        normalization=FrozenRewardScale(mean=0.5, std=0.25), head_specs=[], store=store,
    )
    assert metrics["panel_events"] == 1
    assert metrics["primary"]["base_actor"]["ndcg@10"] == 0.5
    assert metrics["primary"]["analytic_beta_0.1"]["recall@10"] == 1.0
    assert metrics["primary"]["analytic_beta_0.1"]["support_share_top10"] == 0.5
    assert np.isclose(metrics["primary"]["analytic_beta_0.1"]["slices"]["positive"]["ndcg@10"], 1 / np.log2(3))
