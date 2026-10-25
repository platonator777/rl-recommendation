from array import array

import torch

from tlab.data.store import SequenceStore, UserSequence
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.panel_tools.data import PanelTarget
from tlab.panel_tools.evaluator import evaluate_actor_panel


def test_actor_evaluator_keeps_oov_targets_in_primary_denominator():
    torch.manual_seed(7)
    actor = ItemGPT2Actor(ActorShape(vocab_size=5, n_positions=8, n_layer=1, n_embd=8, n_head=2, dropout=0.0))
    user = UserSequence(uid=9)
    user.tokens = array("i", [1, 3, 2])
    user.raw_item_ids = array("q", [77, 100, 999])
    store = SequenceStore([user], 3, 0, 0)
    panel = [
        PanelTarget("a", 9, 0, 1, 100, 3, 0.8, True, True, False),
        PanelTarget("b", 9, 0, 2, 999, 2, 0.2, False, True, False),
    ]
    result = evaluate_actor_panel(actor, store, [100, 101], panel, history_events=4, batch_size=2)
    assert result["n_events"] == 2
    assert result["target_coverage"] == 0.5
    assert result["primary_user_macro"]["n_users"] == 1
    assert result["primary_user_macro"]["n_events"] == 2
    assert "per_user_primary" in result
