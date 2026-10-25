from dataclasses import asdict
import json

import numpy as np
import pytest

from tlab.panel_tools.data import PanelTarget
from tlab.residual_support.cache import CacheIdentity, validate_cache_identity
from tlab.residual_support.metrics import evaluate_score_matrix


def _identity(**changes):
    values = {
        "protocol_id": "residual_support", "split": "train", "panel_event_ids_sha256": "panel",
        "raw_token_mapping_sha256": "mapping-a", "feature_schema_sha256": "features",
        "scaler_sha256": "scale-a", "actor_checkpoint_sha256": "actor",
        "optimized_rm_sha256": "rm-a", "item_prior_sha256": "prior", "support_size": 500,
        "reward_normalization": (("mean", 0.5), ("std", 0.2), ("version", "scale-v1")),
        "support_indices_sha256": "support",
    }
    values.update(changes)
    return CacheIdentity(**values)


def test_cache_identity_rejects_mapping_scaler_rm_and_normalization_changes():
    expected = _identity()
    metadata = {"identity": json.loads(json.dumps(asdict(expected))), "identity_sha256": expected.digest()}
    validate_cache_identity(metadata, expected)
    for changed in (
        _identity(raw_token_mapping_sha256="mapping-b"),
        _identity(scaler_sha256="scale-b"),
        _identity(optimized_rm_sha256="rm-b"),
        _identity(reward_normalization=(("mean", 0.6), ("std", 0.2), ("version", "scale-v1"))),
    ):
        with pytest.raises(ValueError, match="identity mismatch"):
            validate_cache_identity(metadata, changed)


def test_manual_full_catalog_ranks_keep_oov_and_joint_positive_new_repeat_slices():
    panel = [
        PanelTarget("a", 1, 0, 1, 10, 3, 0.9, True, True, False),
        PanelTarget("b", 2, 1, 1, 20, 4, 0.8, True, False, True),
        PanelTarget("c", 3, 2, 1, 999, 2, 0.7, True, True, False),
        PanelTarget("d", 4, 3, 1, 40, 6, 0.2, False, False, True),
    ]
    scores = np.asarray([
        [0.9, 0.2, 0.1, 0.05],
        [0.8, 0.6, 0.4, 0.2],
        [0.5, 0.4, 0.3, 0.2],
        [0.4, 0.3, 0.2, 0.1],
    ])
    metrics = evaluate_score_matrix(panel=panel, scores=scores, item_ids=np.asarray([10, 20, 30, 40]),
                                    rm_fit_counts=np.asarray([0, 1, 10, 100]))
    # The OOV target is a zero-valued miss; it does not receive a fabricated rank.
    expected = (1.0 + 1.0 / np.log2(3) + 1.0 / np.log2(5)) / 4.0
    assert np.isclose(metrics["primary_user_macro"]["ndcg@10"], expected)
    assert metrics["primary_user_macro"]["recall@10"] == 0.75
    assert metrics["oov_targets"] == 1
    assert metrics["slices"]["positive_new"]["n_users"] == 2
    assert np.isclose(metrics["slices"]["positive_new"]["ndcg@10"], 0.5)
    assert metrics["slices"]["positive_repeat"]["n_users"] == 1
    assert metrics["slices"]["positive_repeat"]["ndcg@10"] > 0.6
    assert metrics["slices"]["rm_fit_support_unsupported"]["n_users"] == 1
