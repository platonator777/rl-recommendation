from array import array

import pytest
import torch
from torch.nn import functional as F

from tlab.data.store import SequenceStore, UserSequence
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.train.sft import collate_sequences
from tlab.panel_tools.dense_sft import DenseWindowIndex, dense_accumulation_step, dense_full_softmax_backward


def _actor() -> ItemGPT2Actor:
    torch.manual_seed(123)
    return ItemGPT2Actor(ActorShape(vocab_size=13, n_positions=16, n_layer=1, n_embd=8, n_head=2, dropout=0.0))


def _direct_loss(actor: ItemGPT2Actor, batch) -> torch.Tensor:
    hidden = actor.encode(batch.input_ids, batch.attention_mask)
    logits = F.linear(hidden.float(), actor.output_weight.float()).clone()
    logits[..., :3] = -torch.inf
    return F.cross_entropy(logits[batch.target_mask], batch.targets[batch.target_mask])


def test_chunked_dense_gradient_matches_full_oracle_including_tied_embedding():
    tokens = [[1, 3, 4, 5, 6], [1, 7, 8, 9]]
    masks = [[False, False, True, True, True], [False, True, True, True]]
    batch = collate_sequences(tokens, masks)
    direct = _actor()
    direct_loss = _direct_loss(direct, batch)
    direct_loss.backward()
    chunked = _actor()
    hidden = chunked.encode(batch.input_ids, batch.attention_mask)
    loss = dense_full_softmax_backward(hidden, batch.targets, batch.target_mask, chunked.output_weight, chunk_size=2)
    assert abs(loss - float(direct_loss.detach())) < 1e-6
    for (name_a, parameter_a), (name_b, parameter_b) in zip(direct.named_parameters(), chunked.named_parameters()):
        assert name_a == name_b
        assert torch.allclose(parameter_a.grad, parameter_b.grad, atol=2e-6, rtol=2e-5), name_a
    assert chunked.lm.transformer.wte.weight.data_ptr() == chunked.output_weight.data_ptr()


def test_unequal_microbatch_target_counts_use_one_global_target_denominator():
    batches = [
        collate_sequences([[1, 3, 4]], [[False, False, True]]),
        collate_sequences([[1, 5, 6, 7, 8]], [[False, False, True, True, True]]),
    ]
    direct = _actor()
    combined = collate_sequences([[1, 3, 4], [1, 5, 6, 7, 8]], [[False, False, True], [False, False, True, True, True]])
    _direct_loss(direct, combined).backward()
    trained = _actor()
    optimizer = torch.optim.SGD(trained.parameters(), lr=0.01)
    dense_accumulation_step(trained, batches, optimizer, logit_chunk_size=3, gradient_clip=100.0)
    for (_, expected), (_, actual) in zip(direct.named_parameters(), trained.named_parameters()):
        assert torch.allclose(actual, expected - 0.01 * expected.grad, atol=2e-6, rtol=2e-5)


def test_dense_window_index_groups_targets_without_duplicates_and_keeps_causal_masks():
    user = UserSequence(uid=4)
    user.tokens = array("i", [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    user.raw_item_ids = array("q", range(11))
    user.train_targets = array("I", [2, 3, 4, 9, 10])
    user.split_codes = bytearray([0] * 11)
    user.recommendable_events = bytearray([0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    store = SequenceStore([user], 11, 0, 0)
    index = DenseWindowIndex.from_store(store, window_events=4, context_tokens_including_bos=5)
    assert len(index) == 2
    assert index.permutation(17).tolist() == index.permutation(17).tolist()
    first_tokens, first_mask = index.example(store, 0)
    second_tokens, second_mask = index.example(store, 1)
    assert sum(first_mask) == 3 and sum(second_mask) == 2
    assert first_tokens[0] == second_tokens[0] == 1
    assert all(token >= 3 for token, valid in zip(first_tokens, first_mask) if valid)
    with pytest.raises(ValueError, match="200 preceding events"):
        DenseWindowIndex.from_store(store, window_events=64, context_tokens_including_bos=192)


def test_collator_shifts_once_and_masks_multiple_dense_targets():
    batch = collate_sequences([[1, 3, 4, 5]], [[False, False, True, True]])
    assert batch.input_ids.tolist() == [[1, 3, 4]]
    assert batch.targets.tolist() == [[3, 4, 5]]
    assert batch.target_mask.tolist() == [[False, True, True]]


def test_causal_prefix_hidden_is_invariant_to_future_token_perturbation():
    actor = _actor().eval()
    first = torch.tensor([[1, 3, 4, 5, 6]])
    changed_future = torch.tensor([[1, 3, 10, 11, 12]])
    mask = torch.ones_like(first)
    with torch.no_grad():
        hidden_a = actor.encode(first, mask)
        hidden_b = actor.encode(changed_future, mask)
    assert torch.allclose(hidden_a[:, :2], hidden_b[:, :2], atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP correctness requires a CUDA device")
def test_fp16_grad_scaler_matches_fp32_update_without_double_scaling():
    sequences = [[1, 3, 4, 5], [1, 6, 7, 8, 9]]
    masks = [[False, False, True, True], [False, False, True, True, True]]
    batch = collate_sequences(sequences, masks, device="cuda")
    reference = _actor().cuda()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
    _direct_loss(reference, batch).backward()
    reference_optimizer.step()

    actual = _actor().cuda()
    scaler = torch.amp.GradScaler("cuda", init_scale=8.0, growth_interval=1000)
    actual_optimizer = torch.optim.SGD(actual.parameters(), lr=1e-3)
    dense_accumulation_step(
        actual,
        [batch],
        actual_optimizer,
        logit_chunk_size=2,
        scaler=scaler,
        gradient_clip=100.0,
        autocast_dtype=torch.float16,
    )
    for (_, expected), (_, observed) in zip(reference.named_parameters(), actual.named_parameters()):
        assert torch.allclose(observed, expected, atol=2e-4, rtol=2e-3)
    assert scaler.get_scale() == 8.0
