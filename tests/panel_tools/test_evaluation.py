from array import array

import numpy as np

from tlab.data.store import SequenceStore, UserSequence
from tlab.panel_tools.data import build_evaluation_panel, history_for_target, stable_event_id
from tlab.panel_tools.evaluation import evaluate_rank_rows, single_target_metrics, target_rank
from tlab.panel_tools.statistics import holm_adjust, paired_user_bootstrap, relative_quality_guard


def _user(uid: int, target_ratios: tuple[float, ...], target_tokens: tuple[int, ...]) -> UserSequence:
    user = UserSequence(uid)
    user.tokens = array("i", [3, *target_tokens])
    user.raw_item_ids = array("q", [10, *[100 + i for i in range(len(target_tokens))]])
    user.split_codes = bytearray([0, *([1] * len(target_tokens))])
    user.subsplit_codes = bytearray([0] * (len(target_tokens) + 1))
    user.recommendable_events = bytearray([0, *([1] * len(target_tokens))])
    user.ratio_raw = array("f", [0.9, *target_ratios])
    return user


def test_rank_uses_score_then_raw_id_for_ties_and_oov_is_miss():
    ids = [30, 10, 20]
    scores = [0.5, 0.5, 0.9]
    assert target_rank(scores, ids, 10) == 2
    assert target_rank(scores, ids, 30) == 3
    assert target_rank(scores, ids, 999) is None
    assert single_target_metrics(None) == {"ndcg@10": 0.0, "recall@10": 0.0}


def test_macro_aggregation_averages_events_within_user_first():
    item_ids = np.arange(20, dtype=np.int64)
    good = np.arange(20, dtype=np.float64)
    bad = np.zeros(20, dtype=np.float64)
    rows = [
        {"uid": 1, "scores": good, "item_ids": item_ids, "target_id": 19},
        {"uid": 1, "scores": good, "item_ids": item_ids, "target_id": 19},
        {"uid": 2, "scores": bad, "item_ids": item_ids, "target_id": 19},
    ]
    result = evaluate_rank_rows(rows)
    assert result["primary_user_macro"]["ndcg@10"] == 0.5
    assert result["primary_user_macro"]["n_users"] == 2
    assert result["primary_user_macro"]["n_events"] == 3


def test_panel_selection_is_stable_and_independent_of_ratio_and_catalog():
    user = _user(101, (0.1, 0.9, 0.2), (2, 3, 3))
    store = SequenceStore([user], 4, 0, 0)
    panel_a = build_evaluation_panel(store, source_sha256="a" * 64, targets_per_user=3, seed=17)
    history = history_for_target(store, panel_a[0], history_events=2)
    user.ratio_raw[1:] = array("f", [0.99, 0.01, 0.99])
    user.tokens[1:] = array("i", [3, 2, 3])
    panel_b = build_evaluation_panel(store, source_sha256="a" * 64, targets_per_user=3, seed=17)
    assert [row.event_id for row in panel_a] == [row.event_id for row in panel_b]
    assert len(panel_a) == 3
    assert stable_event_id("x", 1, 2, 3) != stable_event_id("x", 1, 3, 3)
    assert history[0] == 1
    assert all(row.token_id in (2, 3) for row in panel_a)


def test_paired_bootstrap_and_relative_guard_are_user_clustered():
    a = {1: 0.5, 2: 0.2, 3: 0.9}
    b = {1: 0.4, 2: 0.2, 3: 0.8}
    first = paired_user_bootstrap(a, b, n_bootstrap=100, seed=10)
    second = paired_user_bootstrap(a, b, n_bootstrap=100, seed=10)
    assert first == second
    guard = relative_quality_guard(a, b, n_bootstrap=100, seed=10)
    assert guard["interval"]["n_users"] == 3
    assert guard["criterion"].startswith("one-sided 95%")
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
