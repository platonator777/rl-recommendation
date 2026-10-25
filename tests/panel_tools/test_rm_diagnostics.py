from array import array

import numpy as np

from tlab.data.store import SequenceStore, UserSequence
from tlab.panel_tools.data import stable_event_id
from tlab.panel_tools.rm_diagnostics import build_rm_action_datasets, history_only_features


def _user(uid: int, partition: str) -> UserSequence:
    user = UserSequence(uid=uid, rm_user_partition=partition)
    user.tokens = array("i", [1, 3, 4, 3, 4])
    user.raw_item_ids = array("q", [0, 10, 20, 10, 20])
    user.ratio_raw = array("f", [float("nan"), 0.1, 0.2, 0.7, 0.4])
    user.duration_seconds = array("f", [float("nan"), 100, 200, 120, 210])
    user.rm_fit_positions = array("I", [3, 4]) if partition == "fit" else array("I")
    user.rm_validation_positions = array("I", [3, 4]) if partition == "audit" else array("I")
    return user


def test_history_only_features_are_finite_and_include_feedback_missingness():
    values, names = history_only_features(
        np.asarray([10, -1, 20]), np.asarray([0.1, np.nan, 0.8]),
    )
    assert values.shape == (20,)
    assert len(names) == len(values)
    assert np.isfinite(values).all()
    assert values[names.index("past_ratio_missing_50")] == 0.0


def test_rm_feature_rows_exclude_current_target_feedback():
    fit_user = _user(11, "fit")
    audit_user = _user(22, "audit")
    store = SequenceStore([fit_user, audit_user], 10, 0, 0)
    before = build_rm_action_datasets(store, source_hash="b" * 64, max_events_per_user=2, seed=9)
    fit_user.ratio_raw[3] = 0.99
    after = build_rm_action_datasets(store, source_hash="b" * 64, max_events_per_user=2, seed=9)
    event_id = stable_event_id("b" * 64, 11, 3, 10)
    before_index = before["fit"]["event_id"].index(event_id)
    after_index = after["fit"]["event_id"].index(event_id)
    assert np.array_equal(before["fit"]["candidate_x"][before_index], after["fit"]["candidate_x"][after_index])
