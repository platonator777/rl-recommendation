from array import array

import numpy as np

from tlab.data.store import SequenceStore, UserSequence
from tlab.panel_tools.data import PanelTarget, stable_event_id
from tlab.panel_tools.selection import select_a_rl_training_contexts, select_one_event_per_user


def _row(uid: int, position: int) -> PanelTarget:
    event_id = stable_event_id("a" * 64, uid, position, 1000 + position)
    return PanelTarget(event_id, uid, uid, position, 1000 + position, 3, 0.2, False, True, False)


def test_one_event_selection_is_stable_and_keeps_every_user_without_reading_labels():
    rows = [_row(1, 4), _row(1, 8), _row(1, 12), _row(2, 3), _row(2, 9)]
    reversed_rows = list(reversed(rows))
    a = select_one_event_per_user(rows, seed=17)
    b = select_one_event_per_user(reversed_rows, seed=17)
    assert [(row.uid, row.event_id) for row in a] == [(row.uid, row.event_id) for row in b]
    assert [row.uid for row in a] == [1, 2]
    assert len({row.uid for row in a}) == 2


def test_a_rl_selection_accepts_numpy_memmap_style_anchor_arrays():
    user = UserSequence(uid=9)
    user.tokens = array("i", [3, 4, 5, 6])
    user.raw_item_ids = array("q", [101, 102, 103, 104])
    user.ratio_raw = array("f", [0.2, 0.3, 0.4, 0.5])
    user.rl_anchors = np.asarray([1, 2, 3], dtype=np.uint32)
    store = SequenceStore(users=[user], events=4, tie_blocks=0, tied_events=0)
    selected = select_a_rl_training_contexts(store, source_sha256="b" * 64, count=1, seed=17)
    assert len(selected) == 1
    assert selected[0].uid == 9
    assert selected[0].position in {1, 2, 3}

