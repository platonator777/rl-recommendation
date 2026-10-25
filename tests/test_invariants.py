import math
import json
from pathlib import Path

import pytest
import torch
import polars as pl

from tlab.data.catalog import BOS_ID, OOV_ID, top_catalog
from tlab.data.labels import TimedPairEvent, attribute_new_likes
from tlab.data.like_index import build_like_index
from tlab.data.prepare import prepare_dataset
from tlab.data.sequences import ListenEvent, SequenceDiagnostics, prefix_examples
from tlab.data.splits import user_hash_partition
from tlab.data.store import SequenceStore, UserSequence
from tlab.data.temporal import SECONDS_PER_DAY, make_boundaries
from tlab.data.prepare import shifted_max_timestamp_raw
from tlab.evaluation.behavior import duration_quantile_from_train, exposure_metrics, gini_including_zeros
from tlab.evaluation.baselines import rank_score_vector, smoothed_transition_scores
from tlab.evaluation.ranking import _top_item_ids, evaluate_full_catalog
from tlab.evaluation.continuation import _Event, _examples_for_user, _future_labels, _future_lookup, _history_features
from tlab.evaluation.rerank import lexicographic_ranking_order
from tlab.evaluation.horizon import _append_synthetic
from tlab.evaluation.final import _ranking_summary, evaluate_final_selection
from tlab.evaluation.case_studies import build_case_studies
from tlab.evaluation.observed_dashboard import summarize_user_test_events
from tlab.evaluation.observed_dashboard import observed_test_dashboard
from tlab.experiments.freeze_selection import freeze_selection
from tlab.evaluation.metrics import aggregate_macro_user, rank_target, single_target_metrics
from tlab.evaluation.policy_panel import _pairwise_jaccard
from tlab.evaluation.rm_controls import evaluate_rm_controls
from tlab.evaluation.rm_diagnostics import _calibration_bins, _summarize_errors
from tlab.evaluation.seconds_diagnostics import _target_seconds
from tlab.models.actor import ActorShape, ItemGPT2Actor, chunked_full_softmax_ce
from tlab.models.reward import CandidateRewardModel, RewardModelShape
from tlab.train.reward_model import collate_reward_examples, sample_positions, sample_positions_uniform_user
from tlab.train.sft import collate_sequences
from tlab.train.policy import _apply_repeat_penalty, _collate_histories, _score_actions, _select_reward_head
from tlab.train.checkpoint import load_training_checkpoint, save_training_checkpoint, source_tree_sha256
from tlab.data.manifest import sha256_file
from tlab.train.rollout import (
    collect_horizon_rollout,
    collect_one_step_rollout,
    grpo_update,
    horizon_grpo_update,
    reinforce_update,
)
from tlab.train.reward import reward_model_loss
from tlab.train.objectives import (
    analytic_reward_kl_policy,
    full_categorical_kl,
    full_catalog_support_components,
    grpo_clipped_loss,
    group_advantages,
    leave_one_out_advantages,
    prepare_reward_objective,
)


def test_rm_calibration_bins_are_equal_frequency_and_deterministic():
    bins = _calibration_bins(
        targets=[0.0, 1.0, 0.0, 1.0, 0.5],
        predictions=[0.1, 0.9, 0.2, 0.8, 0.5],
        bins=2,
    )
    assert [row["count"] for row in bins] == [2, 3]
    assert bins[0]["mean_predicted"] == pytest.approx(0.15)
    assert bins[0]["mean_target"] == pytest.approx(0.0)
    assert bins[1]["mean_predicted"] == pytest.approx((0.5 + 0.8 + 0.9) / 3)
    assert bins[1]["mean_target"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)


def test_rm_history_order_shuffle_and_user_swap_are_reported_separately():
    rows = [
        (1, 0.2, 0.25, 0.4, True, 0.35, 0.55),
        (2, 0.8, 0.75, 0.6, True, 0.65, 0.45),
    ]
    result = _summarize_errors(rows, repetitions=20, seed=17)
    assert "macro_user_order_shuffle_rmse" in result
    assert "macro_user_history_swap_rmse" in result
    assert result["macro_user_history_swap_rmse"] > result["macro_user_model_rmse"]


def test_observed_dashboard_maturity_and_repeat_windows():
    rows = [
        {"timestamp": 100, "item_id": 10, "is_organic": False, "token_id": 3,
         "ambiguous_timestamp_block": False, "ratio_raw": 1.2, "seconds_proxy_raw": 120.0},
        {"timestamp": 600, "item_id": 20, "is_organic": False, "token_id": 4,
         "ambiguous_timestamp_block": False, "ratio_raw": 0.5, "seconds_proxy_raw": 50.0},
        {"timestamp": 86_500, "item_id": 10, "is_organic": True, "token_id": 3,
         "ambiguous_timestamp_block": False, "ratio_raw": 1.0, "seconds_proxy_raw": 100.0},
    ]
    result = summarize_user_test_events(
        1, rows,
        likes=[TimedPairEvent(1, 10, 150)],
        unlikes=[],
        test_start=0,
        test_end=172_800,
    )
    assert result["eligible_nonorganic_anchors"] == 2
    assert result["mature_continue_anchors"] == 2
    assert result["continue_positive"] == 1
    assert result["eligible_like_labels"] == 2
    assert result["new_likes"] == 1
    assert result["relisten_counts"]["all"] == 2
    assert result["relisten_positive"]["all"] == 1
    assert result["replay_excess_count"] == 1


def test_observed_dashboard_reads_fixed_test_window_and_keeps_zero_users(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path
    (root / "configs").mkdir()
    (root / "data" / "processed").mkdir(parents=True)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "configs" / "base.yaml").write_text(
        "data:\n  processed_dir: data/processed\n  source_dir: data/raw\n", encoding="utf-8",
    )
    users = []
    first = UserSequence(uid=1)
    first.append(3, 10, -100, "train", "F_RM", False, 1.0, 100.0, False)
    first.append(3, 10, 100, "test", None, False, 1.2, 100.0, False)
    first.append(4, 20, 600, "test", None, False, 0.5, 100.0, False)
    first.append(3, 10, 86_500, "test", None, True, 1.0, 100.0, False)
    second = UserSequence(uid=2)
    second.append(3, 30, -200, "train", "F_RM", False, 0.8, 100.0, False)
    users.extend([first, second])
    SequenceStore(users, 5, 0, 0).save_cache(root / "data" / "processed" / "sequence_index", "toy-source-hash")
    (root / "data" / "processed" / "protocol.json").write_text(json.dumps({
        "processed_listens_sha256": "toy-source-hash",
        "timestamp_unit_seconds": 1.0,
        "boundaries_source_units": {"test_start_raw": 0.0, "end_raw": 604_800.0},
    }), encoding="utf-8")
    prepared = pa.table({
        "uid": [1, 1, 1], "timestamp": [100, 600, 86_500], "item_id": [10, 20, 10],
        "is_organic": [False, False, True], "token_id": [3, 4, 3],
        "ambiguous_timestamp_block": [False, False, False], "ratio_raw": [1.2, 0.5, 1.0],
        "seconds_proxy_raw": [120.0, 50.0, 100.0], "global_split": ["test", "test", "test"],
    })
    pq.write_table(prepared, root / "data" / "processed" / "listens.parquet")
    pq.write_table(pa.table({"uid": [1], "item_id": [10], "timestamp": [150]}), root / "data" / "raw" / "likes.parquet")
    pq.write_table(pa.table({"uid": [], "item_id": [], "timestamp": []}), root / "data" / "raw" / "unlikes.parquet")

    result = observed_test_dashboard(root / "configs" / "base.yaml")
    observed = result["observed"]
    assert result["n_users_cohort"] == 2
    assert observed["listening"]["total_seconds_proxy"] == pytest.approx(270.0)
    assert observed["listening"]["mean_seconds_proxy_per_user_day_including_zeros"] == pytest.approx(270.0 / 14)
    assert observed["listening"]["return_during_test_rate"] == pytest.approx(0.5)
    assert observed["new_like"]["new_likes"] == 1
    assert observed["relisten_24h"]["by_anchor_type"]["all"]["relistened"] == 1


def test_catalog_frequency_ties_are_stable():
    catalog = top_catalog([(9, 2), (4, 2), (12, 1)], limit=3)
    assert catalog.item_ids == (4, 9, 12)
    assert catalog.encode(4) == 3
    assert catalog.encode(777) == OOV_ID


def test_rerank_uses_reward_then_sft_score_then_item_id_ties():
    reward = torch.tensor([0.8, 0.8, 0.9, 0.8])
    sft = torch.tensor([0.1, 0.7, 0.5, 0.7])
    item_ids = torch.tensor([10, 9, 8, 2])
    assert lexicographic_ranking_order(reward, sft, item_ids).tolist() == [2, 3, 1, 0]


def test_policy_reward_head_selects_ratio_or_physical_seconds():
    predictions = {
        "ratio": torch.tensor([0.4]),
        "seconds_scaled": torch.tensor([0.2]),
    }
    assert _select_reward_head(predictions, "ratio").tolist() == pytest.approx([0.4])
    assert _select_reward_head(predictions, "seconds", seconds_scale=275.0).tolist() == pytest.approx([55.0])


def test_q4_repeat_penalty_subtracts_only_for_recent_history_repeats():
    rewards = torch.tensor([[0.7, 0.3]])
    repeats = torch.tensor([[False, True]])
    assert torch.allclose(_apply_repeat_penalty(rewards, repeats, 0.1), torch.tensor([[0.7, 0.2]]))


def test_user_overlap_uses_jaccard_on_fixed_topk_sets():
    result = _pairwise_jaccard([[0, 1], [0, 2], [3, 4]], seed=17)
    assert result["n_pairs"] == 3
    assert result["mean_jaccard"] == pytest.approx(1 / 9)


def test_freeze_selection_hashes_validation_inputs_before_test(tmp_path):
    import yaml

    root = tmp_path
    (root / "src" / "tlab").mkdir(parents=True)
    (root / "src" / "tlab" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "base.yaml").write_text("model: {}\n", encoding="utf-8")
    (root / "data" / "processed").mkdir(parents=True)
    (root / "data" / "processed" / "protocol.json").write_text(
        json.dumps({"processed_listens_sha256": "listens-hash", "catalog_sha256": "catalog-hash"}),
        encoding="utf-8",
    )
    run_dir = root / "artifacts" / "runs" / "sft_seed17"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": "sft_seed17", "stage": "SFT", "seed": 17, "finished_at_utc": "now"}),
        encoding="utf-8",
    )
    evidence = root / "artifacts" / "validation.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({"split": "validation", "ndcg@10": 0.01}), encoding="utf-8")
    config = {
        "data": {"processed_dir": "data/processed"},
        "final": {
            "model_config": "configs/base.yaml",
            "selection_split": "validation",
            "selected_family": "SFT_B1",
            "selection_rule": "highest validation NDCG@10 subject to predeclared guardrails",
            "selection_rationale": "The evidence file records the validation comparison.",
            "models": [{
                "name": "sft_seed17",
                "family": "SFT_B1",
                "seed": 17,
                "checkpoint": "artifacts/runs/sft_seed17/best.pt",
                "validation_evidence": ["artifacts/validation.json"],
            }],
        },
    }
    config_path = root / "configs" / "final.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = freeze_selection(config_path)

    frozen = json.loads((root / "artifacts" / "final_selection.json").read_text(encoding="utf-8"))
    assert frozen["selection_split"] == "validation"
    assert frozen["test_evaluation"]["status"] == "not_run_at_freeze"
    assert frozen["models"][0]["checkpoint"]["sha256"]
    assert result["output_path"].endswith("final_selection.json")


def test_freeze_selection_rejects_non_validation_selection(tmp_path):
    import yaml

    (tmp_path / "configs").mkdir()
    config_path = tmp_path / "configs" / "final.yaml"
    config_path.write_text(yaml.safe_dump({"final": {"selection_split": "test"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be frozen from validation"):
        freeze_selection(config_path)


def test_centered_grpo_advantages_use_leave_one_out_scale_correction():
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    advantages = group_advantages(rewards, normalize=False, centered_correction=True)
    expected = torch.tensor([[-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0]])
    assert torch.allclose(advantages, expected)


def test_seconds_target_uses_listen_ratio_duration_and_training_cap():
    user = UserSequence(uid=1)
    user.append(3, 10, 0.0, "train", None, False, 2.0, 100.0, False)
    user.append(4, 11, 1.0, "train", None, False, 0.5, 100.0, False)
    user.append(5, 12, 2.0, "train", None, False, 0.5, None, False)
    assert _target_seconds(user, 0, cap=150.0) == pytest.approx(150.0)
    assert _target_seconds(user, 1, cap=150.0) == pytest.approx(50.0)
    assert _target_seconds(user, 2, cap=150.0) is None


def test_synthetic_history_append_preserves_bos_and_bounds_context():
    assert _append_synthetic([BOS_ID, 4, 5], 6, history_length=3) == [BOS_ID, 5, 6]
    assert _append_synthetic([BOS_ID, 4, 5], 6, history_length=5) == [BOS_ID, 4, 5, 6]


def test_full_model_h5_append_preserves_bos_and_handles_minimum_context():
    from tlab.experiments.full_model_horizon import _append_synthetic_token

    assert _append_synthetic_token([BOS_ID, 4, 5], 6, history_length=3) == [BOS_ID, 5, 6]
    assert _append_synthetic_token([BOS_ID, 4, 5], 6, history_length=5) == [BOS_ID, 4, 5, 6]
    assert _append_synthetic_token([BOS_ID, 4], 5, history_length=1) == [BOS_ID]
    with pytest.raises(ValueError, match="history_length"):
        _append_synthetic_token([BOS_ID, 4], 5, history_length=0)


def test_full_model_h1_policy_step_matches_fixed_categorical_and_identical_context_js():
    from tlab.experiments.full_model_horizon import _policy_divergence, _policy_step

    class FixedPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("log_probs", torch.tensor([[-torch.inf, -torch.inf, -torch.inf,
                                                               math.log(0.2), math.log(0.3), math.log(0.5)]]))

        def next_action_log_probs(self, input_ids, attention_mask):
            return self.log_probs.expand(input_ids.shape[0], -1)

    actor = FixedPolicy()
    generator = torch.Generator(device="cpu").manual_seed(17001)
    histories = [[BOS_ID, 3]] * 5000
    tokens, entropies, log_parts = _policy_step(actor, histories, torch.device("cpu"), generator)
    observed = torch.bincount(torch.tensor(tokens) - 3, minlength=3).float() / len(tokens)
    assert torch.allclose(observed, torch.tensor([0.2, 0.3, 0.5]), atol=0.025, rtol=0)
    assert len(entropies) == len(tokens) == len(histories)
    assert set(tokens) == {3, 4, 5}
    assert max(_policy_divergence(log_parts, log_parts)) < 1e-7


def test_full_model_same_item_context_scoring_and_complete_window_provenance():
    from types import SimpleNamespace
    from tlab.experiments.full_model_horizon import _anchor_windows, _score_actions

    class ContextScore:
        def __call__(self, input_ids, attention_mask, candidate_ids, durations, missing, repeats):
            return {"ratio": input_ids[:, -1].float()}

    model = ContextScore()
    fixed = _score_actions(model, {}, [[BOS_ID, 3]], [3], torch.device("cpu"))
    appended = _score_actions(model, {}, [[BOS_ID, 3, 4]], [3], torch.device("cpu"))
    assert fixed == [3.0]
    assert appended == [4.0]
    assert appended[0] - fixed[0] == 1.0

    class FakeStore:
        users = [SimpleNamespace(uid=11), SimpleNamespace(uid=22)]

        def evaluation_targets(self, split, per_user_limit=None):
            assert split == "validation"
            return [(0, position) for position in range(2, 7)] + [(1, 10), (1, 11)]

    windows, summary = _anchor_windows(FakeStore(), {
        "development": [{"user_index": 0, "uid": 11, "position": 2, "anchor_id": "a"}],
        "confirmation": [{"user_index": 1, "uid": 22, "position": 10, "anchor_id": "b"}],
    }, history_length=8, horizon=5, requested=2)
    assert [row["complete_window"] for row in windows] == [True, False]
    assert summary["complete_window_count"] == 1
    assert summary["teacher_forcing_windows_by_length"] == {"1": 2, "2": 2, "3": 1, "4": 1, "5": 1}


def test_global_cutoffs_are_half_open_and_user_independent():
    end = 100 * SECONDS_PER_DAY
    split = make_boundaries(end - 1, timestamp_unit_seconds=1, timestamp_bin_seconds=1)
    assert split.train_cutoff_raw == end - 21 * SECONDS_PER_DAY
    assert split.validation_start_raw == split.train_cutoff_raw
    assert split.test_start_raw == end - 7 * SECONDS_PER_DAY
    assert split.rm_validation_start_raw == split.train_cutoff_raw - 14 * SECONDS_PER_DAY
    assert split.rl_anchor_start_raw == split.train_cutoff_raw - 7 * SECONDS_PER_DAY


def test_temporal_backtest_shift_moves_exclusive_fold_end_and_keeps_source_units():
    raw_max = 120 * SECONDS_PER_DAY
    shifted = shifted_max_timestamp_raw(raw_max, timestamp_unit_seconds=1, shift_days=14)
    assert shifted == raw_max - 14 * SECONDS_PER_DAY
    boundaries = make_boundaries(shifted, timestamp_unit_seconds=1, timestamp_bin_seconds=5)
    assert boundaries.end_raw == raw_max - 14 * SECONDS_PER_DAY + 5
    assert boundaries.end_raw <= raw_max - 14 * SECONDS_PER_DAY + 5
    assert shifted_max_timestamp_raw(raw_max, timestamp_unit_seconds=5, shift_days=14) == raw_max - 14 * SECONDS_PER_DAY / 5
    with pytest.raises(ValueError):
        shifted_max_timestamp_raw(raw_max, timestamp_unit_seconds=1, shift_days=-1)


def test_prefix_examples_never_use_target_or_order_timestamp_ties():
    diagnostics = SequenceDiagnostics()
    rows = [
        ListenEvent(1, 1, 3, True),
        ListenEvent(1, 2, 4, True),
        ListenEvent(1, 2, 5, False),
        ListenEvent(1, 3, 6, False),
    ]
    examples = list(prefix_examples(rows, history_length=8, diagnostics=diagnostics))
    assert len(examples) == 1
    assert examples[0].target_token == 6
    assert examples[0].history == (BOS_ID, 3, OOV_ID)
    assert diagnostics.tie_blocks == 1
    assert diagnostics.tie_events == 2


def test_continuation_and_return_labels_obey_strict_future_and_censoring():
    timestamps = [0.0, 600.0, 1800.0, 86_400.0, 86_401.0, 604_800.0]
    next_nonorganic, next_organic, session_starts = _future_lookup(
        timestamps,
        [True, True, False, True, False, False],
        [False, False, True, False, True, False],
    )
    assert next_nonorganic[0] == 600.0  # current timestamp is excluded
    assert next_organic[0] == 1800.0
    labels = _future_labels(0.0, next_nonorganic[0], next_organic[0], session_starts, 604_801.0)
    assert labels == ((1, 1, 1), 1, False)
    assert _future_labels(0.0, next_nonorganic[0], next_organic[0], session_starts, 604_800.0) is None


def test_continuation_activity_features_use_only_strictly_prior_events():
    features = _history_features([0.0, 86_400.0, 86_401.0], [1, 2, 9], 2)
    assert features == pytest.approx((math.log1p(2), math.log1p(3), math.log1p(3), math.log1p(1)))


def test_positive_only_sft_filters_targets_but_keeps_full_history():
    user = UserSequence(uid=1)
    for index, ratio in enumerate((0.9, 0.2, 0.8, 0.5, 0.7)):
        user.append(
            token=3 + index,
            item_id=10 + index,
            timestamp=float(index),
            split="train",
            train_subsplit=None,
            is_organic=False,
            ratio=ratio,
            duration=100.0,
            ambiguous=False,
        )
    store = SequenceStore([user], events=5, tie_blocks=0, tied_events=0)
    windows = list(store.sample_sft_windows(10, history_length=5, n_positions=16, positive_only=True))
    assert len(windows) == 1
    tokens, mask = windows[0]
    assert tokens == [BOS_ID, 3, 4, 5, 6, 7]
    assert mask == [False, False, False, True, False, True]


def test_reference_sft_target_index_has_exact_bos_history_and_resume_order():
    user = UserSequence(uid=22)
    for index, ratio in enumerate((0.9, 0.2, 0.8, 0.5, 0.7, 0.1, 0.95)):
        user.append(
            token=3 + index,
            item_id=100 + index,
            timestamp=float(index),
            split="train",
            train_subsplit=None,
            is_organic=False,
            ratio=ratio,
            duration=120.0,
            ambiguous=False,
        )
    store = SequenceStore([user], events=7, tie_blocks=0, tied_events=0)
    target_index = store.sample_sft_target_index(20, history_events=2, seed=31)
    repeated_index = store.sample_sft_target_index(20, history_events=2, seed=31)
    assert len(target_index) == 6  # position zero has no prior event and is not eligible
    assert list(target_index.positions) == list(repeated_index.positions)
    assert list(target_index.user_indices) == list(repeated_index.user_indices)
    assert len(set(target_index.target_keys().tolist())) == len(target_index)

    for ref_id in range(len(target_index)):
        tokens, mask = target_index.example(store, ref_id)
        user_index = target_index.user_indices[ref_id]
        position = target_index.positions[ref_id]
        assert tokens[:-1] == store.users[user_index].history(position, history_length=3)
        assert tokens[-1] == store.users[user_index].tokens[position]
        assert sum(mask) == 1 and mask[-1]
        assert len(tokens) <= 4  # BOS + two prior events + target

    positive = store.sample_sft_target_index(20, history_events=2, seed=31, positive_only=True)
    assert all(user.ratio_raw[position] > 0.5 for user, position in (
        (store.users[positive.user_indices[i]], positive.positions[i])
        for i in range(len(positive))
    ))

    order = target_index.permutation(seed=99)
    interrupted_cursor = 3
    resumed_order = repeated_index.permutation(seed=99)
    assert order.tolist() == resumed_order.tolist()
    assert order[:interrupted_cursor].tolist() + resumed_order[interrupted_cursor:].tolist() == order.tolist()


def test_sft_cpu_staged_collation_matches_direct_device_tensors():
    sequences = [[BOS_ID, 3, 7, 11], [BOS_ID, 8, 12], [BOS_ID, 15, 19, 21, 25]]
    masks = [[False, False, False, True], [False, False, True], [False, False, False, False, True]]
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        direct = collate_sequences(sequences, masks, device=device, transfer_mode="direct")
        staged = collate_sequences(sequences, masks, device=device, transfer_mode="cpu_staged")
        for field in ("input_ids", "attention_mask", "targets", "target_mask"):
            assert torch.equal(getattr(direct, field).cpu(), getattr(staged, field).cpu())


def test_continuation_examples_respect_time_blocks_and_do_not_use_current_item_as_history():
    events = [
        _Event(0.0, 6, True, 1, None, 100.0),
        _Event(100.0, 4, False, 1, "F_RM", 120.0),
        _Event(200.0, 5, False, 2, "F_RM", 120.0),  # ambiguous future block: label only
        _Event(86_401.0, 3, True, 1, None, 100.0),
        _Event(90_000.0, 3, True, 1, None, 100.0),  # session start after 24 hours
        _Event(1_100_000.0, 6, False, 1, "F_RM", 100.0),
        _Event(1_100_100.0, 7, False, 2, "F_RM", 100.0),
        _Event(1_100_100.0, 8, True, 2, "F_RM", 100.0),
    ]
    train, validation, diagnostics = _examples_for_user(
        uid=1,
        events=events,
        train_target_cutoff=395_200.0,
        validation_start=1_000_000.0,
        validation_target_cutoff=1_395_200.0,
        observation_cutoff=2_000_000.0,
    )
    assert len(train) == 1 and len(validation) == 1
    assert train[0].continuation == (1, 1, 1)
    assert train[0].return_7d == 1
    assert train[0].activity[2] == pytest.approx(math.log1p(1))
    assert validation[0].recent_repeat == 1.0
    assert validation[0].activity[2] == pytest.approx(math.log1p(5))
    assert validation[0].continuation == (1, 1, 1)
    assert diagnostics["matured_targets"] == 2


def test_like_attribution_uses_last_prior_pair_listen_and_masks_censoring():
    listens = [
        TimedPairEvent(1, 10, 100),
        TimedPairEvent(1, 10, 200),
        TimedPairEvent(1, 11, 200),
        TimedPairEvent(1, 12, 5000),
    ]
    likes = [TimedPairEvent(1, 10, 250)]
    rows = attribute_new_likes(listens, likes, [], observation_cutoff=5500, window_seconds=1000)
    by_key = {(row.listen.item_id, row.listen.timestamp): row for row in rows}
    assert by_key[(10, 100)].label == 0
    assert by_key[(10, 200)].label == 1
    assert by_key[(11, 200)].label == 0
    assert by_key[(12, 5000)].label is None
    assert by_key[(12, 5000)].reason == "censored_future_window"


def test_like_attribution_masks_timestamp_ties_and_respects_unlike_state():
    listens = [TimedPairEvent(2, 9, 10), TimedPairEvent(2, 9, 20), TimedPairEvent(2, 9, 30)]
    likes = [TimedPairEvent(2, 9, 20)]
    unlikes = [TimedPairEvent(2, 9, 25)]
    rows = attribute_new_likes(listens, likes, unlikes, observation_cutoff=100, window_seconds=40)
    assert rows[0].label == 0
    assert rows[1].label is None
    assert rows[1].reason == "same_timestamp_like"
    assert rows[2].label == 0


def test_user_hash_split_is_stable_and_disjoint():
    assert user_hash_partition(123, seed=17) == user_hash_partition(123, seed=17)
    assignments = [user_hash_partition(uid, seed=17) for uid in range(1000)]
    assert "fit" in assignments and "audit" in assignments


def test_rm_validation_sampling_is_uniform_over_users():
    users = [UserSequence(uid=uid) for uid in range(3)]
    users[0].rm_validation_positions.extend(range(100))
    users[1].rm_validation_positions.extend(range(10))
    users[2].rm_validation_positions.extend(range(2))
    anchors = sample_positions_uniform_user(users, "rm_validation_positions", max_targets=3, seed=17)
    assert [sum(user_index == index for user_index, _ in anchors) for index in range(3)] == [1, 1, 1]


def test_prepare_builds_temporal_catalog_without_reading_test_targets(tmp_path):
    config_dir = tmp_path / "configs"
    raw_dir = tmp_path / "data" / "raw"
    config_dir.mkdir()
    raw_dir.mkdir(parents=True)
    rows = {
        "uid": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        "item_id": [10, 11, 15, 10, 12, 10, 10, 13, 14, 13],
        "timestamp": [10, 20, 20, 30, 40, 42, 10, 20, 90, 100],
        "is_organic": [0] * 10,
        "played_ratio_pct": [80.0] * 10,
        "track_length_seconds": [100.0] * 10,
    }
    pl.DataFrame(rows).write_parquet(raw_dir / "listens.parquet")
    pl.DataFrame({
        "uid": [1], "timestamp": [35], "item_id": [10], "is_organic": [1],
    }).write_parquet(raw_dir / "likes.parquet")
    pl.DataFrame({
        "uid": [1], "timestamp": [45], "item_id": [10], "is_organic": [1],
    }).write_parquet(raw_dir / "unlikes.parquet")
    config_text = """
project:
  artifacts_dir: artifacts
dataset:
  files:
    listens: flat/50m/listens.parquet
  timestamp_unit_seconds: 86400
  timestamp_bin_seconds: 86400
  catalog_size: 4
  train_days: 21
  validation_days: 14
  test_days: 7
  rm_validation_days: 7
  rl_anchor_days: 7
data:
  source_dir: data/raw
  processed_dir: data/processed
"""
    config_path = config_dir / "base.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    protocol = prepare_dataset(config_path)
    prepared = pl.read_parquet(tmp_path / "data" / "processed" / "listens.parquet")
    catalog = pl.read_parquet(tmp_path / "data" / "processed" / "catalog.parquet")
    assert protocol["catalog_size"] == 4
    assert catalog.get_column("item_id").to_list() == [10, 11, 12, 13]
    feature_paths = {key: Path(value["path"]) for key, value in protocol["rm_item_features_by_partition"].items()}
    assert feature_paths["fit"] != feature_paths["audit"]
    assert all(path.exists() for path in feature_paths.values())
    feature_paths = {key: Path(value["path"]) for key, value in protocol["rm_item_features_by_partition"].items()}
    assert feature_paths["fit"] != feature_paths["audit"]
    assert all(path.exists() for path in feature_paths.values())
    test_row = prepared.filter(pl.col("timestamp") == 100).row(0, named=True)
    assert test_row["global_split"] == "test"
    assert test_row["item_id"] == 13
    validation_row = prepared.filter(pl.col("timestamp") == 90).row(0, named=True)
    assert validation_row["global_split"] == "validation"
    like_summary = build_like_index(config_path, window_seconds=10 * 86400)
    like_manifest = json.loads((tmp_path / "data" / "processed" / "like_labels_864000s.json").read_text(encoding="utf-8"))
    labels = (tmp_path / "data" / "processed" / "like_labels_864000s.bin").read_bytes()
    uid1_offset = next(row["offset"] for row in like_manifest["users"] if row["uid"] == 1)
    assert labels[uid1_offset + 2] == 1
    assert labels[uid1_offset + 4] == 255  # already-liked state is masked
    assert like_summary["valid_labels_by_block"]["F_RM"] > 0
    store = SequenceStore.from_parquet(tmp_path / "data" / "processed" / "listens.parquet")
    assert store.events == 10
    assert store.tie_blocks == 1 and store.tied_events == 2
    assert store.users[0].tokens.tolist() == [3, OOV_ID, 3, 5, 3]
    assert store.users[0].history(3, history_length=4) == [BOS_ID, 3, OOV_ID, 3]
    assert len(store.users[0].train_targets) == 3
    assert all(len(tokens) <= 16 for tokens, _ in store.sample_sft_windows(10, history_length=3, n_positions=16))
    rm_examples = sample_positions(store.users, "rm_fit_positions", max_targets=1, seed=3)
    assert len(rm_examples) == 1
    reward_batch = collate_reward_examples(
        store, [(0, 2)], {3: (100.0, False)}, seconds_cap=420.0,
        seconds_scale=275.0, history_length=200, device=torch.device("cpu"),
        like_labels_by_uid={1: [-1, -1, 1, -1, 0]},
    )
    assert reward_batch[2].tolist() == [3]
    assert reward_batch[3].tolist() == [100.0]
    assert reward_batch[4].tolist() == [False]
    assert reward_batch[5].tolist() == [True]
    assert reward_batch[6].tolist() == pytest.approx([0.8])
    assert reward_batch[8].tolist() == pytest.approx([80.0 / 275.0])
    assert reward_batch[10].tolist() == [1.0]
    assert reward_batch[11].tolist() == [True]
    store.users[0].rl_anchors.append(2)
    cache_path = tmp_path / "sequence_cache"
    store.save_cache(cache_path, source_sha256="fixture-hash")
    cached = SequenceStore.load_cache(cache_path, expected_source_sha256="fixture-hash")
    assert cached.events == store.events
    assert cached.tie_blocks == store.tie_blocks
    assert cached.users[0].tokens.tolist() == store.users[0].tokens.tolist()
    assert cached.users[0].history(3, history_length=4) == store.users[0].history(3, history_length=4)
    assert list(cached.sample_sft_windows(10, history_length=3, n_positions=16)) == list(
        store.sample_sft_windows(10, history_length=3, n_positions=16)
    )
    assert cached.sample_rl_anchors(3, seed=5) == [(0, 2)] * 3
    with pytest.raises(ValueError, match="source hash"):
        SequenceStore.load_cache(cache_path, expected_source_sha256="wrong-hash")


def test_catalog_ce_chunking_matches_dense_value_and_gradient():
    torch.manual_seed(7)
    hidden_a = torch.randn(5, 4, requires_grad=True)
    weight_a = torch.randn(11, 4, requires_grad=True)
    targets = torch.tensor([3, 4, 9, 2, 6])
    valid = targets >= 3
    dense_logits = torch.nn.functional.linear(hidden_a[valid], weight_a)
    dense_logits[..., :3] = -torch.inf
    dense = torch.nn.functional.cross_entropy(dense_logits, targets[valid])
    dense.backward()
    dense_value = dense.detach().clone()
    dense_hidden_grad = hidden_a.grad.detach().clone()
    dense_weight_grad = weight_a.grad.detach().clone()

    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    weight_b = weight_a.detach().clone().requires_grad_(True)
    chunked = chunked_full_softmax_ce(hidden_b, targets, valid, weight_b, chunk_size=2)
    chunked.backward()
    assert torch.allclose(dense_value, chunked.detach(), atol=1e-6)
    assert torch.allclose(dense_hidden_grad, hidden_b.grad, atol=1e-6)
    assert torch.allclose(dense_weight_grad, weight_b.grad, atol=1e-6)


def test_grpo_and_reinforce_advantages_have_expected_degenerate_behavior():
    equal = torch.tensor([[0.5, 0.5, 0.5]])
    assert torch.equal(group_advantages(equal), torch.zeros_like(equal))
    rewards = torch.tensor([[1.0, 0.0, -1.0]])
    loo = leave_one_out_advantages(rewards)
    assert torch.allclose(loo, torch.tensor([[1.5, 0.0, -1.5]]))


def test_reinforce_leave_one_out_gradient_matches_exact_categorical_gradient():
    torch.manual_seed(19)
    logits = torch.tensor([0.3, -0.2, 0.1], requires_grad=True)
    log_distribution = torch.log_softmax(logits, dim=-1)
    distribution = log_distribution.exp()
    rewards = torch.tensor([1.0, -0.3, 0.2])
    exact_gradient = torch.autograd.grad(-(distribution * rewards).sum(), logits, retain_graph=True)[0]

    group_count = 20_000
    probabilities = distribution.detach().expand(group_count * 3, -1)
    actions = torch.multinomial(probabilities, 1).reshape(group_count, 3)
    sampled_rewards = rewards[actions]
    sampled_log_probs = log_distribution[actions]
    estimate = -(leave_one_out_advantages(sampled_rewards) * sampled_log_probs).mean()
    monte_carlo_gradient = torch.autograd.grad(estimate, logits)[0]
    assert torch.allclose(monte_carlo_gradient, exact_gradient, atol=0.015, rtol=0.0)


def test_production_reward_preparation_normalizes_sampled_and_exact_once():
    raw = torch.tensor([[10.0, 20.0, 40.0]])
    exact = prepare_reward_objective(raw, reward_mean=10.0, reward_std=2.0)
    sampled = prepare_reward_objective(
        raw, reward_mean=10.0, reward_std=2.0,
        actions=torch.tensor([[3, 4, 5]]), first_catalog_id=3,
    )
    assert exact["normalized_reward"].tolist() == [[0.0, 5.0, 15.0]]
    assert sampled["sampled_raw_reward"].tolist() == [[10.0, 20.0, 40.0]]
    assert sampled["sampled_normalized_reward"].tolist() == [[0.0, 5.0, 15.0]]
    assert sampled["advantages"].tolist() == [[-10.0, -2.5, 12.5]]
    assert prepare_reward_objective(raw, reward_mean=2.0, reward_std=0.0005)[
        "normalized_reward"][0, 0].item() == pytest.approx(8000.0)
    with pytest.raises(ValueError, match="outside"):
        prepare_reward_objective(raw, reward_mean=10.0, reward_std=2.0,
                                 actions=torch.tensor([[2, 4]]))


def test_production_gradient_parity_and_support_penalty_use_full_catalog():
    from tlab.experiments.full_model_matrix import production_gradient_audit

    audit = production_gradient_audit()
    assert audit["status"] == "passed"
    assert len(audit["cases"]) == 2
    for result in audit["cases"]:
        assert result["reward_std"] != 1.0
        assert result["exact_vs_exhaustive_sampled_gradient_max_abs"] < 1e-6
        assert result["legacy_raw_advantage_gradient_max_abs"] > 1e-4
    assert audit["active_support_excess_with_zero_unsupported_samples"]
    assert audit["support_penalty_gradient_max_abs_vs_finite_difference"] < 1e-4
    assert audit["inactive_hinge_gradient_max_abs"] == 0.0
    assert audit["policy_equal_to_reference_gradient_max_abs"] == 0.0


def test_corrected_runner_rejects_legacy_raw_reward_resume():
    from tlab.experiments.full_model_matrix import _validate_objective_resume

    with pytest.raises(ValueError, match="objective_version"):
        _validate_objective_resume({"advantage_reward_units": "raw"})
    with pytest.raises(ValueError, match="normalized"):
        _validate_objective_resume({"objective_version": "normalized_reward_v1",
                                    "advantage_reward_units": "raw"})
    _validate_objective_resume({"objective_version": "normalized_reward_v1",
                                "advantage_reward_units": "normalized"})


def test_analytic_policy_and_exact_kl():
    log_ref = torch.log_softmax(torch.tensor([[0.1, 0.2, -0.4]]), dim=-1)
    reward = torch.tensor([[1.0, -0.5, 0.3]])
    policy = analytic_reward_kl_policy(log_ref, reward, beta=0.7)
    assert torch.allclose(policy.sum(dim=-1), torch.ones(1))
    assert full_categorical_kl(policy.log(), log_ref).item() >= -1e-7
    optimum = (policy * reward).sum() - 0.7 * full_categorical_kl(policy.log(), log_ref).sum()
    other = torch.tensor([[0.2, 0.5, 0.3]])
    other_j = (other * reward).sum() - 0.7 * full_categorical_kl(other.log(), log_ref).sum()
    assert optimum >= other_j - 1e-6


def test_masked_action_kl_ignores_equal_negative_infinity_special_tokens():
    policy = torch.tensor([[float("-inf"), float("-inf"), float("-inf"), -0.2, -1.7]])
    reference = torch.tensor([[float("-inf"), float("-inf"), float("-inf"), -0.4, -1.1]])
    assert torch.isfinite(full_categorical_kl(policy, reference)).all()


def test_actor_masks_special_tokens_and_uses_full_catalog_distribution():
    actor = ItemGPT2Actor(ActorShape(vocab_size=19, n_positions=8, n_layer=1, n_embd=16, n_head=4, dropout=0.1))
    input_ids = torch.tensor([[1, 3, 4], [1, 5, 0]])
    attention = torch.tensor([[1, 1, 1], [1, 1, 0]])
    log_probs = actor.next_action_log_probs(input_ids, attention)
    assert torch.isneginf(log_probs[:, :3]).all()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(2), atol=1e-6)


def test_horizon_first_action_matches_h1_with_same_seed_and_contexts():
    shape = ActorShape(vocab_size=13, n_positions=8, n_layer=1, n_embd=16, n_head=4, dropout=0.1)
    actor = ItemGPT2Actor(shape).eval()
    reference = ItemGPT2Actor(shape).eval()
    reference.load_state_dict(actor.state_dict())
    input_ids = torch.tensor([[1, 3, 4], [1, 5, 6]])
    attention = torch.ones_like(input_ids)
    reward_one = lambda actions: torch.zeros_like(actions, dtype=torch.float32)
    reward_step = lambda histories, actions, step: torch.zeros_like(actions, dtype=torch.float32)

    torch.manual_seed(73)
    one_step = collect_one_step_rollout(actor, reference, input_ids, attention, reward_one, group_size=4)
    torch.manual_seed(73)
    horizon = collect_horizon_rollout(
        actor, reference, [[1, 3, 4], [1, 5, 6]], reward_step,
        group_size=4, horizon=1, seed=None,
    )
    assert torch.equal(one_step.actions, horizon.actions[:, :, 0])
    assert torch.allclose(one_step.old_log_probs, horizon.old_log_probs[:, :, 0])


def test_horizon_one_update_matches_one_step_grpo_update():
    torch.manual_seed(91)
    shape = ActorShape(vocab_size=13, n_positions=8, n_layer=1, n_embd=16, n_head=4, dropout=0.1)
    actor_one = ItemGPT2Actor(shape).eval()
    actor_horizon = ItemGPT2Actor(shape).eval()
    actor_horizon.load_state_dict(actor_one.state_dict())
    reference = ItemGPT2Actor(shape).eval()
    reference.load_state_dict(actor_one.state_dict())
    input_ids = torch.tensor([[1, 3, 4], [1, 5, 6]])
    attention = torch.ones_like(input_ids)
    histories = [[1, 3, 4], [1, 5, 6]]
    reward_one = lambda actions: actions.float()
    reward_step = lambda states, actions, step: actions.float()
    torch.manual_seed(97)
    one_step = collect_one_step_rollout(actor_one, reference, input_ids, attention, reward_one, group_size=4)
    torch.manual_seed(97)
    horizon = collect_horizon_rollout(
        actor_horizon, reference, histories, reward_step, group_size=4, horizon=1, seed=None,
    )
    optimizer_one = torch.optim.AdamW(actor_one.parameters(), lr=1e-4, weight_decay=0.0)
    optimizer_horizon = torch.optim.AdamW(actor_horizon.parameters(), lr=1e-4, weight_decay=0.0)
    grpo_update(actor_one, input_ids, attention, one_step, optimizer_one, beta=0.1)
    horizon_grpo_update(actor_horizon, horizon, optimizer_horizon, beta=0.1)
    max_parameter_delta = max(
        float((a - b).abs().max()) for a, b in zip(actor_one.parameters(), actor_horizon.parameters())
    )
    assert all(torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(actor_one.parameters(), actor_horizon.parameters())), max_parameter_delta


def test_horizon_stepwise_gradient_accumulation_matches_joint_objective_update():
    torch.manual_seed(941)
    shape = ActorShape(vocab_size=17, n_positions=8, n_layer=1, n_embd=16, n_head=4, dropout=0.0)
    actor_accumulated = ItemGPT2Actor(shape).eval()
    actor_joint = ItemGPT2Actor(shape).eval()
    actor_joint.load_state_dict(actor_accumulated.state_dict())
    reference = ItemGPT2Actor(shape).eval()
    reference.load_state_dict(actor_accumulated.state_dict())
    histories = [[1, 3, 4], [1, 5, 6]]
    reward_step = lambda states, actions, step: actions.float() + step * 0.13 + (actions % 3).float().square()
    rollout = collect_horizon_rollout(
        actor_accumulated,
        reference,
        histories,
        reward_step,
        group_size=4,
        horizon=3,
        history_length=8,
        normalize_advantages=False,
        centered_correction=True,
        seed=942,
    )
    beta = 0.4
    clip_epsilon = 0.2
    optimizer_accumulated = torch.optim.AdamW(actor_accumulated.parameters(), lr=1e-4, weight_decay=0.0)
    optimizer_joint = torch.optim.AdamW(actor_joint.parameters(), lr=1e-4, weight_decay=0.0)

    accumulated_metrics = horizon_grpo_update(
        actor_accumulated,
        rollout,
        optimizer_accumulated,
        beta=beta,
        clip_epsilon=clip_epsilon,
    )

    policy_terms = []
    kl_terms = []
    for step, ((input_ids, attention_mask), reference_log_dist) in enumerate(
        zip(rollout.states, rollout.reference_log_probs)
    ):
        new_log_dist = actor_joint.next_action_log_probs(input_ids, attention_mask)
        actions = rollout.actions[:, :, step].reshape(-1)
        old_log_probs = rollout.old_log_probs[:, :, step].reshape(-1)
        advantages = rollout.advantages[:, :, None].expand_as(rollout.actions)[:, :, step].reshape(-1)
        new_log_probs = new_log_dist.gather(-1, actions[:, None]).squeeze(-1)
        ratio = torch.exp(new_log_probs.float() - old_log_probs.detach().float())
        advantage = advantages.detach().float()
        objective = torch.minimum(
            ratio * advantage,
            ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage,
        )
        policy_terms.append(-objective.mean())
        kl_terms.append(full_categorical_kl(new_log_dist, reference_log_dist).mean())
    joint_loss = torch.stack(policy_terms).sum() + beta * torch.stack(kl_terms).sum()
    optimizer_joint.zero_grad(set_to_none=True)
    joint_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_joint.parameters(), 1.0)
    optimizer_joint.step()

    assert accumulated_metrics["loss"] == pytest.approx(float(joint_loss.detach()), abs=1e-6)
    assert all(
        torch.allclose(accumulated, joint, atol=2e-5, rtol=2e-5)
        for accumulated, joint in zip(actor_accumulated.parameters(), actor_joint.parameters())
    )


def test_grpo_reuse_keeps_rollout_fixed_and_enforces_exact_old_policy_kl_guard():
    torch.manual_seed(310)
    shape = ActorShape(vocab_size=15, n_positions=8, n_layer=1, n_embd=16, n_head=4, dropout=0.1)
    actor = ItemGPT2Actor(shape).eval()
    reference = ItemGPT2Actor(shape).eval()
    reference.load_state_dict(actor.state_dict())
    ids = torch.tensor([[1, 3, 4], [1, 5, 6]])
    mask = torch.ones_like(ids)
    torch.manual_seed(311)
    rollout = collect_one_step_rollout(
        actor, reference, ids, mask,
        lambda actions: actions.float(), group_size=4,
    )
    actions_before = rollout.actions.clone()
    rewards_before = rollout.rewards.clone()
    old_dist_before = rollout.old_policy_log_probs.clone()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=0.02, weight_decay=0.0)

    metrics = grpo_update(
        actor, ids, mask, rollout, optimizer, beta=0.0,
        update_epochs=4, minibatch_contexts=1, max_kl_to_old=1e-12,
    )

    assert torch.equal(rollout.actions, actions_before)
    assert torch.equal(rollout.rewards, rewards_before)
    assert torch.equal(rollout.old_policy_log_probs, old_dist_before)
    assert metrics["optimizer_steps"] == 1
    assert metrics["early_stop_kl_guard_triggered"] == 1.0
    assert metrics["post_update_kl_to_old"] > 1e-12


def test_checkpoint_resume_restores_optimizer_sampler_and_random_state(tmp_path):
    import random

    torch.manual_seed(101)
    random.seed(202)
    model_a = torch.nn.Linear(3, 1)
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=0.01)

    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(8, 3)
        y = model(x).square().mean()
        y.backward()
        optimizer.step()

    update(model_a, optimizer_a)
    checkpoint_path = tmp_path / "resume.pt"
    save_training_checkpoint(
        checkpoint_path, model_a, optimizer_a, step=1,
        metadata={"stage": "toy"}, sampler_state={"next_anchor_seed": 19},
    )
    update(model_a, optimizer_a)
    expected = {key: value.detach().clone() for key, value in model_a.state_dict().items()}
    expected_python_random = random.random()

    model_b = torch.nn.Linear(3, 1)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=0.01)
    resumed = load_training_checkpoint(checkpoint_path, model_b, optimizer_b)
    assert resumed["sampler_state"] == {"next_anchor_seed": 19}
    assert random.random() == expected_python_random
    update(model_b, optimizer_b)
    assert all(torch.allclose(expected[key], value) for key, value in model_b.state_dict().items())


def test_reinforce_update_changes_actor_without_mutating_frozen_reference():
    shape = ActorShape(vocab_size=13, n_positions=8, n_layer=1, n_embd=16, n_head=4, dropout=0.1)
    actor = ItemGPT2Actor(shape)
    reference = ItemGPT2Actor(shape).eval()
    reference.load_state_dict(actor.state_dict())
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reference_before = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    input_ids = torch.tensor([[1, 3, 4], [1, 5, 6]])
    attention = torch.ones_like(input_ids)
    rollout = collect_one_step_rollout(
        actor, reference, input_ids, attention,
        reward_function=lambda actions: actions.float(), group_size=4,
    )
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    reinforce_update(actor, input_ids, attention, rollout, optimizer, beta=0.1)
    assert any(parameter.grad is not None for parameter in actor.parameters())
    assert not rollout.reference_log_probs.requires_grad
    assert all(torch.equal(reference_before[key], value) for key, value in reference.state_dict().items())


def test_candidate_reward_heads_use_masks_and_backpropagate():
    model = CandidateRewardModel(RewardModelShape(vocab_size=32, hidden_size=16, n_layer=1, n_head=4, max_positions=8))
    histories = torch.tensor([[1, 3, 4], [1, 7, 0]])
    attention = torch.tensor([[1, 1, 1], [1, 1, 0]])
    prediction = model(
        histories, attention, torch.tensor([9, 11]), torch.tensor([120.0, 0.0]),
        torch.tensor([False, True]), torch.tensor([False, True]),
    )
    loss, pieces = reward_model_loss(
        prediction,
        ratio_target=torch.tensor([0.7, 0.2]),
        ratio_mask=torch.tensor([True, True]),
        seconds_scaled_target=torch.tensor([0.5, 0.0]),
        seconds_mask=torch.tensor([True, True]),
        like_target=torch.tensor([1.0, 0.0]),
        like_mask=torch.tensor([True, False]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pieces["like_bce"])
    assert model.item_embedding.weight.grad is not None


def test_recent_repeat_includes_the_most_recent_listen():
    user = UserSequence(uid=9)
    for position, token in enumerate((3, 3)):
        user.append(
            token=token,
            item_id=token,
            timestamp=float(position),
            split="train",
            train_subsplit="F_RM",
            is_organic=False,
            ratio=0.5,
            duration=120.0,
            ambiguous=False,
        )
    store = SequenceStore([user], events=2, tie_blocks=0, tied_events=0)
    batch = collate_reward_examples(
        store, [(0, 1)], {3: (120.0, False, 2)}, seconds_cap=300.0,
        seconds_scale=120.0, history_length=20, device=torch.device("cpu"),
    )
    assert batch[5].tolist() == [True]


def test_policy_reward_scores_actions_and_counts_unsupported_items():
    model = CandidateRewardModel(RewardModelShape(
        vocab_size=12, hidden_size=8, n_layer=1, n_head=2, max_positions=8, encoder="gru",
    )).eval()
    histories = [[1, 3, 4], [1, 5]]
    input_ids, attention = _collate_histories(histories, torch.device("cpu"))
    with torch.no_grad():
        rewards, unsupported = _score_actions(
            model, histories, input_ids, attention, torch.tensor([[3, 6], [7, 5]]),
            {3: (120.0, False, 4), 5: (90.0, False, 2), 6: (0.0, True, 0), 7: (60.0, False, 1)},
        )
    assert rewards.shape == (2, 2)
    assert torch.isfinite(rewards).all()
    assert unsupported == 1


def test_negative_advantage_clipping_and_zero_variance():
    new = torch.tensor([[math.log(0.5), math.log(0.5)]], requires_grad=True)
    old = torch.tensor([[math.log(0.2), math.log(0.8)]])
    advantages = torch.tensor([[1.0, -1.0]])
    loss, metrics = grpo_clipped_loss(new, old, advantages, torch.zeros(1), beta=0.0)
    assert torch.isfinite(loss)
    assert metrics["clip_fraction"].item() > 0


def test_ranking_keeps_out_of_catalog_targets_as_misses_and_ties_by_id():
    assert rank_target([0.5, 0.5, 0.1], 1, item_ids=[20, 10, 30]) == 1
    assert rank_target([1.0, 0.0], None) is None
    assert single_target_metrics(None)["ndcg@10"] == 0.0
    macro = aggregate_macro_user([(1, 1), (1, None), (2, 2)])
    assert macro["recall@10"] == 0.75


def test_top_item_ids_use_raw_item_id_tie_breaks():
    scores = torch.tensor([[0.7, 0.7, 0.9], [0.2, 0.8, 0.8]]).numpy()
    assert _top_item_ids(scores, [19, 3, 11], 2) == [[11, 3], [3, 11]]


def test_full_catalog_evaluation_returns_top_items_consistent_with_target_rank():
    user = UserSequence(uid=55)
    for split, item_id, token in (("train", 100, 3), ("validation", 10, 4)):
        user.append(
            token=token, item_id=item_id, timestamp=float(len(user.tokens)), split=split,
            train_subsplit=None, is_organic=False, ratio=0.5, duration=120.0, ambiguous=False,
        )
    store = SequenceStore([user], events=2, tie_blocks=0, tied_events=0)
    actor = ItemGPT2Actor(ActorShape(vocab_size=5, n_positions=8, n_layer=1, n_embd=8, n_head=2, dropout=0.0))
    result = evaluate_full_catalog(
        actor, store, [10, 100], split="validation", per_user_limit=1, seed=17, batch_size=1, top_k=2,
    )
    target_rank = result["per_user_ranks"]["55"][0]
    top_items = result["per_user_top_item_ids"]["55"][0]
    assert top_items.index(10) + 1 == target_rank


def test_baseline_rank_uses_raw_item_id_for_ties_and_keeps_oov_as_miss():
    item_ids = [20, 10, 30]
    scores = [5.0, 5.0, 1.0]
    assert rank_score_vector(scores, 0, item_ids) == 2
    assert rank_score_vector(scores, 1, item_ids) == 1
    assert rank_score_vector(scores, None, item_ids) is None


def test_exposure_gini_and_entropy_include_unexposed_catalog_items():
    assert gini_including_zeros([0, 0, 0]) == 0.0
    metrics = exposure_metrics([[0, 0], [0]], catalog_size=4, top_k=10)
    assert metrics["coverage"] == 0.25
    assert metrics["exposure_gini"] > 0.0
    assert 0 <= metrics["exposure_entropy"] <= 1


def test_duration_threshold_uses_only_train_event_durations():
    user = UserSequence(uid=77)
    for index, (split, duration) in enumerate((
        ("train", 100.0), ("train", 200.0), ("validation", 1.0),
    )):
        user.append(
            token=3, item_id=index + 10, timestamp=float(index), split=split,
            train_subsplit=None, is_organic=False, ratio=0.5, duration=duration, ambiguous=False,
        )
    store = SequenceStore([user], events=3, tie_blocks=0, tied_events=0)
    assert duration_quantile_from_train(store, 0.5) == pytest.approx(150.0)
    assert duration_quantile_from_train(SequenceStore([], events=0, tie_blocks=0, tied_events=0), 0.1) is None


def test_final_ranking_summary_separates_coverage_cold_and_positive_targets():
    ranks = {"1": [1, None], "2": [100]}
    targets = {
        "1": [
            {"item_id": 10, "positive": True, "in_catalog": True, "known_in_train": True},
            {"item_id": 99, "positive": False, "in_catalog": False, "known_in_train": False},
        ],
        "2": [
            {"item_id": 88, "positive": True, "in_catalog": False, "known_in_train": True},
        ],
    }
    result = _ranking_summary(ranks, targets)
    assert result["n_targets"] == 3
    assert result["n_positive_targets"] == 2
    assert result["target_coverage"] == pytest.approx(1 / 3)
    assert result["known_train_outside_catalog"] == 1
    assert result["cold_outside_catalog"] == 1
    assert result["macro_user_positive"]["recall@10"] == pytest.approx(0.5)


def test_final_evaluator_runs_only_frozen_model_on_toy_test_data(tmp_path):
    root = tmp_path
    (root / "src" / "tlab").mkdir(parents=True)
    (root / "src" / "tlab" / "toy.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "configs").mkdir()
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    (root / "artifacts" / "runs" / "sft_seed17").mkdir(parents=True)
    user = UserSequence(uid=17)
    for split, item_id, token, ratio, duration in (
        ("train", 10, 3, 0.8, 100.0),
        ("train", 20, 4, 0.7, 200.0),
        ("test", 10, 3, 0.9, 100.0),
    ):
        user.append(
            token=token, item_id=item_id, timestamp=float(len(user.tokens)), split=split,
            train_subsplit=None, is_organic=False, ratio=ratio, duration=duration, ambiguous=False,
        )
    store = SequenceStore([user], events=3, tie_blocks=0, tied_events=0)
    store.save_cache(processed / "sequence_index", source_sha256="toy-listens")
    (processed / "protocol.json").write_text(
        json.dumps({"processed_listens_sha256": "toy-listens", "catalog_sha256": "toy-catalog"}),
        encoding="utf-8",
    )
    pl.DataFrame({
        "item_id": [10, 20],
        "token_id": [3, 4],
        "training_count": [2, 1],
        "catalog_duration_seconds": [100.0, 200.0],
    }).write_parquet(processed / "catalog.parquet")
    model_config = root / "configs" / "base.yaml"
    model_config.write_text(
        "model:\n  n_positions: 8\n  n_layer: 1\n  n_embd: 8\n  n_head: 2\n  dropout: 0.0\n"
        "dataset:\n  history_length: 8\n  catalog_size: 2\n"
        "training:\n  rm_hidden_size: 4\n  rm_layers: 1\n  rm_heads: 1\n"
        "data:\n  processed_dir: data/processed\n",
        encoding="utf-8",
    )
    feature_frame = pl.DataFrame({
        "token_id": [3, 4],
        "rm_duration_seconds": [100.0, 200.0],
        "rm_duration_missing": [False, False],
        "rm_training_support": [1, 1],
    })
    feature_frame.write_parquet(processed / "rm_item_features_fit.parquet")
    feature_frame.write_parquet(processed / "rm_item_features_audit.parquet")
    final_config = root / "configs" / "final.yaml"
    final_config.write_text("final: {}\n", encoding="utf-8")
    actor = ItemGPT2Actor(ActorShape(vocab_size=5, n_positions=8, n_layer=1, n_embd=8, n_head=2, dropout=0.0))
    run_dir = root / "artifacts" / "runs" / "sft_seed17"
    checkpoint_path = run_dir / "best.pt"
    torch.save({"model": actor.state_dict()}, checkpoint_path)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps({"run_id": "sft_seed17", "stage": "SFT", "seed": 17, "finished_at_utc": "now"}),
        encoding="utf-8",
    )
    reward_panel_dir = root / "artifacts" / "reward_panel_inputs"
    reward_panel_dir.mkdir(parents=True)
    fit_model = CandidateRewardModel(RewardModelShape(
        vocab_size=5, hidden_size=4, n_layer=1, n_head=1, max_positions=8, encoder="transformer",
    ))
    audit_model = CandidateRewardModel(RewardModelShape(
        vocab_size=5, hidden_size=4, n_layer=1, n_head=1, max_positions=8, encoder="gru",
    ))
    fit_checkpoint = reward_panel_dir / "fit.pt"
    audit_checkpoint = reward_panel_dir / "audit.pt"
    torch.save({"model": fit_model.state_dict()}, fit_checkpoint)
    torch.save({"model": audit_model.state_dict()}, audit_checkpoint)
    reward_scale_path = reward_panel_dir / "reward_scale.json"
    reward_scale_path.write_text(
        json.dumps({"reward_scale": {"mean": 0.5, "std": 0.1, "sample_count": 10}}),
        encoding="utf-8",
    )
    selection = {
        "selection_split": "validation",
        "test_status": "post_test_research_iteration",
        "selected_family": "SFT_B1",
        "selection_rule": "validation-only",
        "selection_rationale": "toy fixture",
        "frozen_at_utc": "now",
        "source_sha256_at_freeze": source_tree_sha256(root / "src" / "tlab"),
        "config_path": str(final_config.resolve()),
        "model_config_path": str(model_config.resolve()),
        "model_config_sha256": sha256_file(model_config),
        "data_protocol_path": str((processed / "protocol.json").resolve()),
        "processed_listens_sha256": "toy-listens",
        "catalog_sha256": "toy-catalog",
        "test_evaluation": {"split": "test", "status": "not_run_at_freeze"},
        "test_protocol": {"anchors_per_user": 1, "anchor_seed": 17, "batch_size": 1, "top_k": 2},
        "models": [{
            "name": "sft_seed17",
            "family": "SFT_B1",
            "seed": 17,
            "role": "reference",
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
                "run_manifest_path": str(run_manifest_path.resolve()),
                "run_manifest_sha256": sha256_file(run_manifest_path),
            },
            "reward_panel": {
                "files": {
                    "sft_checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)},
                    "rm_fit_checkpoint": {"path": str(fit_checkpoint.resolve()), "sha256": sha256_file(fit_checkpoint)},
                    "rm_audit_checkpoint": {"path": str(audit_checkpoint.resolve()), "sha256": sha256_file(audit_checkpoint)},
                    "reward_scale_path": {"path": str(reward_scale_path.resolve()), "sha256": sha256_file(reward_scale_path)},
                },
                "beta": 0.1,
                "reward_head": "ratio",
                "repeat_penalty": 0.0,
            },
        }],
    }
    selection_path = root / "artifacts" / "final_selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    result = evaluate_final_selection(selection_path, allow_post_test_research=True)

    saved = json.loads((root / "artifacts" / "final_test_evaluation.json").read_text(encoding="utf-8"))
    assert saved["split"] == "test"
    assert saved["test_status"] == "post_test_research_iteration"
    assert saved["historical_test_access"] is True
    assert saved["models"]["sft_seed17"]["n_targets"] == 1
    assert saved["models"]["sft_seed17"]["n_positive_targets"] == 1
    assert saved["models"]["sft_seed17"]["known_train_outside_catalog"] == 0
    assert saved["models"]["sft_seed17"]["compute"]["inference_targets"] == 1
    assert saved["models"]["sft_seed17"]["compute"]["inference_seconds"] >= 0
    assert saved["models"]["popularity"]["compute"]["training_seconds"] == 0.0
    assert saved["evaluation_wall_seconds"] >= 0
    assert saved["per_user_targets_item_ids"]["17"] == [10]
    panel = saved["full_catalog_test_reward_panels"]["sft_seed17"]
    assert panel["n_users"] == 1
    assert panel["panel_split"] == "test"
    assert result["output_path"].endswith("final_test_evaluation.json")
    cases = build_case_studies(
        root / "artifacts" / "final_test_evaluation.json",
        candidate_model="sft_seed17", reference_model="popularity",
    )
    assert len(cases["cases"]) == 5
    assert cases["cases"][0]["user_id"] == "17"


def test_rm_controls_fit_without_using_validation_targets_for_training():
    def make_store(uid_start: int, count: int, split: str, subsplit: str) -> SequenceStore:
        users = []
        for uid in range(uid_start, uid_start + count):
            user = UserSequence(uid=uid)
            for position in range(4):
                token = 3 + ((uid + position) % 2)
                user.append(
                    token=token,
                    item_id=token,
                    timestamp=float(position),
                    split=split,
                    train_subsplit=subsplit,
                    is_organic=False,
                    ratio=0.2 + 0.1 * (token - 3),
                    duration=60.0 * token,
                    ambiguous=False,
                )
            users.append(user)
        return SequenceStore(users, events=count * 4, tie_blocks=0, tied_events=0)

    train = make_store(100, 30, "train", "F_RM")
    validation = make_store(1000, 12, "validation", "V_RM")
    train_examples = [(user_index, position) for user_index in range(30) for position in range(1, 4)]
    validation_examples = [(user_index, position) for user_index in range(12) for position in range(1, 4)]
    controls = evaluate_rm_controls(
        train_store=train,
        validation_store=validation,
        train_examples=train_examples,
        validation_examples=validation_examples,
        item_features={3: (180.0, False, 20), 4: (240.0, False, 30)},
        global_mean=0.3,
        item_mean={3: 0.2, 4: 0.3},
        history_length=8,
        bootstrap_repetitions=20,
    )
    assert controls["n_train_ratio_labels"] == len(train_examples)
    assert controls["n_validation_ratio_labels"] == len(validation_examples)
    assert controls["candidate_only"]["selected_ridge_alpha_on_v_rm"] in (1.0, 10.0, 100.0)
    assert math.isfinite(controls["duration_popularity_activity"]["macro_user_rmse"])


def test_one_step_sampler_matches_categorical_frequencies_and_keeps_duplicates():
    class FixedPolicy(torch.nn.Module):
        def __init__(self, probabilities):
            super().__init__()
            self.register_buffer("log_probs", torch.as_tensor(probabilities, dtype=torch.float32).clone().log())

        def next_action_log_probs(self, input_ids, attention_mask):
            return self.log_probs.expand(input_ids.shape[0], -1)

    probabilities = torch.tensor([0.1, 0.3, 0.6])
    actor = FixedPolicy(probabilities)
    reference = FixedPolicy(probabilities)
    input_ids = torch.ones((10_000, 1), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        rollout = collect_one_step_rollout(
            actor,
            reference,
            input_ids,
            attention_mask,
            lambda actions: actions.float(),
            group_size=4,
            normalize_advantages=False,
        )
    observed = torch.bincount(rollout.actions.flatten(), minlength=3).float() / rollout.actions.numel()
    assert torch.allclose(observed, probabilities, atol=0.01, rtol=0.0)
    assert (rollout.actions[:, 0] == rollout.actions[:, 1]).any()


def test_markov_smoothing_adds_popularity_prior_and_backs_off_for_unseen_context():
    from scipy.sparse import csr_matrix

    counts = [80.0, 15.0, 5.0]
    transition = csr_matrix(([4.0], ([0], [1])), shape=(1, 3))
    smoothed = smoothed_transition_scores(transition.getrow(0), counts, alpha=1.0)
    assert smoothed[1] > smoothed[0] > smoothed[2]
    assert smoothed_transition_scores(csr_matrix((1, 3)), counts, alpha=10.0).tolist() == [8.0, 1.5, 0.5]
    assert smoothed_transition_scores(csr_matrix((1, 3)), counts, alpha=0.0).tolist() == counts
    with pytest.raises(ValueError, match="alpha"):
        smoothed_transition_scores(transition.getrow(0), counts, alpha=-1.0)


def test_reranker_rejects_unregistered_split():
    from tlab.evaluation.rerank import evaluate_top100_rerank

    with pytest.raises(ValueError, match="split"):
        evaluate_top100_rerank("", "", "", "", split="train")


def test_tiny_actor_and_reward_model_can_overfit_fixed_examples():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(31)
        actor = ItemGPT2Actor(ActorShape(
            vocab_size=7, n_positions=8, n_layer=1, n_embd=8, n_head=2, dropout=0.0,
        ))
        histories = torch.tensor([[1, 3], [1, 4], [1, 3], [1, 4]], dtype=torch.long)
        attention = torch.ones_like(histories)
        targets = torch.tensor([5, 6, 5, 6], dtype=torch.long)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.02)
        with torch.no_grad():
            initial_actor_loss = -actor.next_action_log_probs(histories, attention).gather(
                1, targets[:, None],
            ).mean().item()
        for _ in range(100):
            log_probs = actor.next_action_log_probs(histories, attention)
            loss = -log_probs.gather(1, targets[:, None]).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_optimizer.step()
        with torch.no_grad():
            final_actor_loss = -actor.next_action_log_probs(histories, attention).gather(
                1, targets[:, None],
            ).mean().item()
        assert final_actor_loss < initial_actor_loss * 0.2

        reward_model = CandidateRewardModel(RewardModelShape(
            vocab_size=8, hidden_size=8, n_layer=1, n_head=2, max_positions=8, encoder="gru",
        ))
        history = torch.tensor([[1, 5]] * 8, dtype=torch.long)
        mask = torch.ones_like(history)
        candidates = torch.tensor([3, 4] * 4, dtype=torch.long)
        durations = torch.full((8,), 120.0)
        duration_missing = torch.zeros(8, dtype=torch.bool)
        repeats = torch.zeros(8, dtype=torch.bool)
        targets_ratio = torch.tensor([0.1, 0.9] * 4)
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.01)
        with torch.no_grad():
            initial_rm_loss = torch.nn.functional.mse_loss(
                reward_model(history, mask, candidates, durations, duration_missing, repeats)["ratio"],
                targets_ratio,
            ).item()
        for _ in range(100):
            prediction = reward_model(history, mask, candidates, durations, duration_missing, repeats)["ratio"]
            loss = torch.nn.functional.mse_loss(prediction, targets_ratio)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final_rm_loss = torch.nn.functional.mse_loss(
                reward_model(history, mask, candidates, durations, duration_missing, repeats)["ratio"],
                targets_ratio,
            ).item()
        assert final_rm_loss < initial_rm_loss * 0.2
    finally:
        torch.set_num_threads(previous_threads)


def test_cached_h1_panel_accepts_numpy_anchor_arrays():
    import numpy as np
    from types import SimpleNamespace
    from tlab.experiments.cached_h1 import _user_balanced_rl_panel

    store = SimpleNamespace(users=[
        SimpleNamespace(rl_anchors=np.asarray([4, 9], dtype=np.int64)),
        SimpleNamespace(rl_anchors=np.asarray([], dtype=np.int64)),
    ])
    panel = _user_balanced_rl_panel(store, count=4, seed=17)

    assert len(panel) == 1
    assert panel[0][0] == 0
    assert panel[0][1] in {4, 9}
