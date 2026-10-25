import numpy as np
import pytest

from tlab.panel_tools.features import candidate_history_features


def _features(*, decision_timestamp=None, history_timestamps=None):
    return candidate_history_features(
        history_item_ids=np.asarray([10, 20, 10, 30]),
        history_ratios=np.asarray([0.2, np.nan, 0.8, 0.4]),
        candidate_item_ids=np.asarray([10, 20, 99]),
        duration_seconds=np.asarray([180.0, np.nan, 0.0]),
        train_popularity=np.asarray([10.0, 1.0, 0.0]),
        train_ratio_mean=np.asarray([0.6, np.nan, np.nan]),
        decision_timestamp=decision_timestamp,
        history_timestamps=history_timestamps,
        windows=(2, 4),
    )


def test_features_have_explicit_missingness_and_correct_asof_aggregates():
    batch = _features()
    columns = {name: batch.values[:, i] for i, name in enumerate(batch.names)}
    assert columns["item_count_4"][0] == 2
    assert columns["item_ratio_mean_4"][0] == pytest.approx(0.5)
    assert columns["item_ratio_missing_4"][1] == 1.0
    assert columns["duration_missing"][1] == 1.0
    assert columns["artist_features_missing_4"][0] == 1.0
    assert np.isfinite(batch.values).all()


def test_simultaneous_or_future_history_is_rejected():
    with pytest.raises(ValueError, match="simultaneous or future"):
        _features(decision_timestamp=20.0, history_timestamps=np.asarray([10.0, 20.0, 21.0, 22.0]))
