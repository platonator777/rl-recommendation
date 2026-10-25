from array import array

import torch

from tlab.data.store import SequenceStore, UserSequence
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.train.checkpoint import load_training_checkpoint, save_training_checkpoint
from tlab.train.sft import collate_sequences
from tlab.panel_tools.dense_sft import DenseWindowIndex, dense_accumulation_step, dense_full_softmax_backward
from tlab.residual_support.sft import _coverage_at_cursor, _fixed_train_ce


def test_dense_window_masks_only_train_targets_and_uses_strict_prefix():
    user = UserSequence(uid=91)
    user.tokens = array("i", [3, 4, 5, 6, 7])
    user.raw_item_ids = array("q", [10, 20, 30, 40, 50])
    user.train_targets = array("I", [1, 2, 4])
    store = SequenceStore([user], events=5, tie_blocks=0, tied_events=0)
    index = DenseWindowIndex.from_store(store, window_events=2, context_tokens_including_bos=4)
    tokens, target_mask = index.example(store, 0)
    assert tokens[0] == 1
    assert tokens[-1] != user.tokens[1]
    assert any(target_mask)
    assert all(token >= 3 for token, valid in zip(tokens, target_mask) if valid)


def test_dense_full_softmax_accumulation_produces_finite_gradients():
    user = UserSequence(uid=92)
    user.tokens = array("i", [3, 4, 5, 6])
    user.raw_item_ids = array("q", [10, 20, 30, 40])
    user.train_targets = array("I", [1, 2, 3])
    store = SequenceStore([user], events=4, tie_blocks=0, tied_events=0)
    index = DenseWindowIndex.from_store(store, window_events=2, context_tokens_including_bos=3)
    batch = next(index.batches(store, [0, 1], batch_size=2))
    actor = ItemGPT2Actor(ActorShape(vocab_size=8, n_positions=32, n_layer=1, n_embd=16, n_head=2, dropout=0.0))
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    result = dense_accumulation_step(actor, [batch], optimizer, logit_chunk_size=2, gradient_clip=1.0)
    assert result["supervised_targets"] == float(batch.target_mask.sum())
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in actor.parameters())


def test_dense_chunked_gradient_matches_direct_full_softmax_with_tied_weight():
    torch.manual_seed(4)
    source_dense = torch.randn(2, 3, 5, requires_grad=True)
    source_direct = source_dense.detach().clone().requires_grad_(True)
    weight_dense = torch.nn.Parameter(torch.randn(8, 5))
    weight_direct = torch.nn.Parameter(weight_dense.detach().clone())
    hidden_dense = source_dense * 1.7 + source_dense.square() * 0.1
    hidden_direct = source_direct * 1.7 + source_direct.square() * 0.1
    targets = torch.tensor([[3, 4, 5], [6, 7, 3]])
    valid = torch.tensor([[True, False, True], [False, True, True]])

    chunked_loss = dense_full_softmax_backward(hidden_dense, targets, valid, weight_dense, chunk_size=2)
    logits = torch.nn.functional.linear(hidden_direct[valid], weight_direct)
    logits = logits.clone()
    logits[..., :3] = -torch.inf
    direct_loss = torch.nn.functional.cross_entropy(logits, targets[valid])
    direct_loss.backward()

    assert abs(chunked_loss - float(direct_loss.detach())) < 1e-6
    assert torch.allclose(source_dense.grad, source_direct.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(weight_dense.grad, weight_direct.grad, atol=1e-6, rtol=1e-5)


def test_dense_gradient_accumulation_matches_one_combined_microbatch():
    user = UserSequence(uid=93)
    user.tokens = array("i", [3, 4, 5, 6, 7, 8])
    user.raw_item_ids = array("q", [10, 20, 30, 40, 50, 60])
    user.train_targets = array("I", [1, 2, 4, 5])
    store = SequenceStore([user], events=6, tie_blocks=0, tied_events=0)
    index = DenseWindowIndex.from_store(store, window_events=2, context_tokens_including_bos=4)
    batches = [next(index.batches(store, [i], batch_size=1)) for i in range(2)]
    combined = collate_sequences(
        [index.example(store, i)[0] for i in range(2)],
        [index.example(store, i)[1] for i in range(2)],
    )
    torch.manual_seed(11)
    accumulated_actor = ItemGPT2Actor(ActorShape(10, n_positions=16, n_layer=1, n_embd=8, n_head=2, dropout=0.0))
    combined_actor = ItemGPT2Actor(ActorShape(10, n_positions=16, n_layer=1, n_embd=8, n_head=2, dropout=0.0))
    combined_actor.load_state_dict(accumulated_actor.state_dict())
    accumulated_optimizer = torch.optim.SGD(accumulated_actor.parameters(), lr=0.01)
    combined_optimizer = torch.optim.SGD(combined_actor.parameters(), lr=0.01)
    accumulated = dense_accumulation_step(accumulated_actor, batches, accumulated_optimizer,
                                          logit_chunk_size=4, gradient_clip=100.0)
    direct = dense_accumulation_step(combined_actor, [combined], combined_optimizer,
                                     logit_chunk_size=4, gradient_clip=100.0)
    assert accumulated["supervised_targets"] == direct["supervised_targets"]
    assert abs(accumulated["loss"] - direct["loss"]) < 1e-6
    assert all(torch.allclose(left, right, atol=2e-6, rtol=1e-5)
               for left, right in zip(accumulated_actor.parameters(), combined_actor.parameters()))


def test_dense_bfloat16_autocast_path_has_finite_update():
    user = UserSequence(uid=94)
    user.tokens = array("i", [3, 4, 5, 6])
    user.raw_item_ids = array("q", [10, 20, 30, 40])
    user.train_targets = array("I", [1, 2, 3])
    store = SequenceStore([user], events=4, tie_blocks=0, tied_events=0)
    index = DenseWindowIndex.from_store(store, window_events=2, context_tokens_including_bos=3)
    batch = next(index.batches(store, [0, 1], batch_size=2))
    torch.manual_seed(23)
    actor = ItemGPT2Actor(ActorShape(9, n_positions=16, n_layer=1, n_embd=8, n_head=2, dropout=0.0))
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.001)
    result = dense_accumulation_step(actor, [batch], optimizer, logit_chunk_size=4,
                                     gradient_clip=10.0, autocast_dtype=torch.bfloat16)
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert all(torch.isfinite(parameter).all() for parameter in actor.parameters())


def test_sft_checkpoint_resume_matches_uninterrupted_updates(tmp_path):
    def make_training(seed):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.Dropout(0.2), torch.nn.Linear(7, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        return model, optimizer

    def update(model, optimizer):
        x, y = torch.randn(5, 3), torch.randn(5, 2)
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    continuous, continuous_optimizer = make_training(19)
    for _ in range(4):
        update(continuous, continuous_optimizer)

    interrupted, interrupted_optimizer = make_training(19)
    for _ in range(2):
        update(interrupted, interrupted_optimizer)
    checkpoint = tmp_path / "sft_resume.pt"
    save_training_checkpoint(checkpoint, interrupted, interrupted_optimizer, 2,
                             {"training_seconds": 1.25},
                             sampler_state={"signature": "fixture", "epoch": 0, "cursor": 16})
    resumed, resumed_optimizer = make_training(99)
    payload = load_training_checkpoint(checkpoint, resumed, resumed_optimizer, map_location="cpu")
    assert payload["sampler_state"] == {"signature": "fixture", "epoch": 0, "cursor": 16}
    assert payload["metadata"]["training_seconds"] == 1.25
    for _ in range(2):
        update(resumed, resumed_optimizer)
    assert all(torch.equal(left, right) for left, right in zip(continuous.parameters(), resumed.parameters()))


def test_fixed_train_ce_restores_actor_training_mode():
    user = UserSequence(uid=95)
    user.tokens = array("i", [3, 4, 5])
    user.raw_item_ids = array("q", [10, 20, 30])
    user.train_targets = array("I", [1, 2])
    store = SequenceStore([user], events=3, tie_blocks=0, tied_events=0)
    sampler = store.sample_sft_target_index(max_targets=2, history_events=2, seed=5)
    actor = ItemGPT2Actor(ActorShape(8, n_positions=16, n_layer=1, n_embd=8, n_head=2, dropout=0.1))
    result = _fixed_train_ce(actor, store, sampler, method="last_target", seed=7,
                             device=torch.device("cpu"), max_targets=2, chunk_size=2)
    assert actor.training
    assert result["targets"] == 2.0
    actor.eval()
    _fixed_train_ce(actor, store, sampler, method="last_target", seed=7,
                    device=torch.device("cpu"), max_targets=2, chunk_size=2)
    assert not actor.training


def test_sft_resume_cursor_reconstructs_unique_target_coverage():
    user = UserSequence(uid=96)
    user.tokens = array("i", [3, 4, 5, 6, 7, 8])
    user.raw_item_ids = array("q", [10, 20, 30, 40, 50, 60])
    user.train_targets = array("I", [1, 2, 4, 5])
    store = SequenceStore([user], events=6, tie_blocks=0, tied_events=0)
    dense = DenseWindowIndex.from_store(store, window_events=2, context_tokens_including_bos=4)
    dense_order = dense.permutation(17)
    coverage, targets = _coverage_at_cursor(dense, "dense", dense_order, 0, 1, store)
    assert coverage == 1
    assert targets == int(dense.target_stops[int(dense_order[0])] - dense.target_starts[int(dense_order[0])])
    assert _coverage_at_cursor(dense, "dense", dense_order, 1, 0, store) == (len(dense), 4)

    last = store.sample_sft_target_index(max_targets=4, history_events=4, seed=17)
    last_order = last.permutation(17)
    assert _coverage_at_cursor(last, "last_target", last_order, 0, 2, store) == (2, 2)
    assert _coverage_at_cursor(last, "last_target", last_order, 1, 0, store) == (4, 4)
