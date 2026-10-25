import numpy as np
import pytest

from tlab.panel_tools.baselines import _actor_tie_rank, _history_counts, recent_train_popularity


def test_history_counts_ignore_oov_and_keep_catalog_alignment():
    raw_to_index = {10: 0, 20: 1, 30: 2}
    counts = _history_counts([10, 10, -1, 30, 999], raw_to_index, 3)
    assert counts.tolist() == [2, 0, 1]


def test_actor_tie_rank_uses_raw_item_id_as_stable_tiebreak():
    # Catalog order need not follow raw item ID order.
    ranks = _actor_tie_rank(np.asarray([-1.0, -1.0, -0.5]), np.asarray([30, 10, 20]))
    assert ranks.tolist() == [2, 1, 0]


def test_recent_popularity_requires_a_contiguous_catalog_token_map(tmp_path):
    with pytest.raises(ValueError, match="contiguous"):
        recent_train_popularity(
            tmp_path / "unused.parquet", catalog_tokens=np.asarray([3, 5]), train_cutoff=1000,
        )

