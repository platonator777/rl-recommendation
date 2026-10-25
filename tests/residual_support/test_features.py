from array import array

import numpy as np
import polars as pl

from tlab.data.store import UserSequence
from tlab.residual_support.features import (
    CANDIDATE_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    POLICY_FEATURE_NAMES,
    ItemPriors,
    candidate_features,
    history_context_features,
    policy_feature_matrix,
)
from tlab.residual_support.rm import build_item_priors


def _user():
    user = UserSequence(uid=11)
    user.tokens = array("i", [1, 3, 4, 5])
    user.raw_item_ids = array("q", [-1, 101, 202, 303])
    user.ratio_raw = array("f", [float("nan"), 0.2, 0.9, 0.7])
    user.duration_seconds = array("f", [float("nan"), 100.0, 150.0, 90.0])
    return user


def test_new_candidates_have_distinct_metadata_and_history_interaction_features():
    user = _user()
    priors = ItemPriors(
        train_popularity_count=np.asarray([20, 200], dtype=np.float32),
        ratio_prior_mean=np.asarray([0.2, 0.8], dtype=np.float32),
        ratio_prior_count=np.asarray([10, 100], dtype=np.float32),
        duration_seconds=np.asarray([100.0, 300.0], dtype=np.float32),
        duration_missing=np.zeros(2, dtype=np.uint8), cutoff_raw_exclusive=10.0, source_split="early",
    )
    candidates = np.asarray([1001, 1002], dtype=np.int64)
    base = candidate_features(user, 4, candidates, catalog_indices=np.asarray([0, 1]), priors=priors)
    assert base.shape == (2, len(CANDIDATE_FEATURE_NAMES))
    repeat = base[:, CANDIDATE_FEATURE_NAMES.index("personal_repeat_count_1000")]
    assert np.array_equal(repeat, [0.0, 0.0])
    assert not np.array_equal(base[0], base[1])

    names = np.random.default_rng(17).normal(size=(2, 8)).astype(np.float32)
    names /= np.maximum(np.linalg.norm(names, axis=1, keepdims=True), 1e-8)
    context = history_context_features(user, 4)
    policy = policy_feature_matrix(
        context, base, rm_fit_counts=np.asarray([0, 25]), candidate_catalog_indices=np.asarray([0, 1]),
        reference_logp=np.asarray([-0.2, -1.1], dtype=np.float32), item_embeddings=names,
        history_tokens=np.asarray([3, 4], dtype=np.int64),
    )
    assert policy.shape == (2, len(POLICY_FEATURE_NAMES))
    assert len(set(POLICY_FEATURE_NAMES)) == len(POLICY_FEATURE_NAMES)
    assert not np.array_equal(policy[0], policy[1])
    support_col = POLICY_FEATURE_NAMES.index("rm_fit_item_count")
    assert np.array_equal(policy[:, support_col], [0.0, 25.0])
    assert len(CONTEXT_FEATURE_NAMES) > 0


def test_asof_prior_ignores_current_and_future_labels(tmp_path):
    columns = {
        "uid": [1, 1, 1, 1], "timestamp": [10, 20, 30, 40],
        "global_split": ["train"] * 4, "train_subsplit": ["F_RM"] * 4,
        "is_organic": [0] * 4, "ambiguous_timestamp_block": [False] * 4,
        "token_id": [3, 3, 3, 3], "item_id": [101] * 4,
        "ratio_raw": [0.2, 0.7, 0.99, 0.01], "track_length_seconds": [100, 100, 100, 100],
    }
    first = tmp_path / "first.parquet"
    pl.DataFrame(columns).write_parquet(first)
    priors_a = build_item_priors(first, pl.DataFrame({"item_id": [101], "token_id": [3]}),
                                 cutoff_exclusive=15, source_split="early")
    columns["ratio_raw"][1] = 0.01  # current label is outside the strict prior cutoff
    columns["ratio_raw"][2] = 0.02  # future label is outside the strict prior cutoff
    second = tmp_path / "second.parquet"
    pl.DataFrame(columns).write_parquet(second)
    priors_b = build_item_priors(second, pl.DataFrame({"item_id": [101], "token_id": [3]}),
                                 cutoff_exclusive=15, source_split="early")
    assert np.array_equal(priors_a.train_popularity_count, priors_b.train_popularity_count)
    assert np.allclose(priors_a.ratio_prior_mean, priors_b.ratio_prior_mean, equal_nan=True)
    assert np.allclose(priors_a.ratio_prior_mean, [0.2])
    assert np.array_equal(priors_a.ratio_prior_count, [1.0])


def test_personal_repeat_global_popularity_and_rm_fit_support_are_distinct():
    user = _user()
    priors = ItemPriors(
        train_popularity_count=np.asarray([300], dtype=np.float32),
        ratio_prior_mean=np.asarray([0.4], dtype=np.float32), ratio_prior_count=np.asarray([40], dtype=np.float32),
        duration_seconds=np.asarray([120], dtype=np.float32), duration_missing=np.asarray([0], dtype=np.uint8),
        cutoff_raw_exclusive=10, source_split="early",
    )
    values = candidate_features(user, 4, np.asarray([999]), catalog_indices=np.asarray([0]), priors=priors)[0]
    assert values[CANDIDATE_FEATURE_NAMES.index("personal_repeat_count_1000")] == 0
    assert values[CANDIDATE_FEATURE_NAMES.index("train_popularity_count")] == 300
    features = policy_feature_matrix(
        history_context_features(user, 4), values[None], rm_fit_counts=np.asarray([7]),
        candidate_catalog_indices=np.asarray([0]), reference_logp=np.asarray([-0.2]),
        item_embeddings=np.ones((1, 3), dtype=np.float32), history_tokens=np.asarray([3]),
    )
    assert features[0, POLICY_FEATURE_NAMES.index("rm_fit_item_count")] == 7
    assert features[0, POLICY_FEATURE_NAMES.index("rm_fit_unsupported")] == 0
