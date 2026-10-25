import json

import numpy as np
import pytest
import torch

import tlab.residual_support.train_policy as trainer
from tlab.panel_tools.data import PanelTarget
from tlab.residual_support.features import POLICY_FEATURE_NAMES
from tlab.residual_support.metrics import evaluate_full_catalog_policy
from tlab.residual_support.objectives import FrozenRewardScale
from tlab.residual_support.scaler import TrainOnlyFeatureScaler


def _fixture_data():
    rng = np.random.default_rng(21)
    n_features = len(POLICY_FEATURE_NAMES)
    train_raw = rng.uniform(0.0, 3.0, size=(4, 3, n_features)).astype(np.float32)
    dev_raw = rng.uniform(0.0, 3.0, size=(2, 3, n_features)).astype(np.float32)
    scaler = TrainOnlyFeatureScaler.fit(train_raw, POLICY_FEATURE_NAMES)
    train = {
        "candidate_features": train_raw,
        "reference_conditional_logp": np.full((4, 3), -np.log(3), dtype=np.float32),
        "support_indices": np.tile(np.asarray([[0, 1, 2]], dtype=np.int32), (4, 1)),
        "support_mass": np.full(4, 0.75, dtype=np.float32),
        "raw_candidate_rewards": rng.uniform(0.0, 1.0, size=(4, 3)).astype(np.float32),
    }
    dev = {
        "candidate_features": dev_raw,
        "reference_conditional_logp": np.full((2, 3), -np.log(3), dtype=np.float32),
        "support_indices": np.tile(np.asarray([[0, 1, 2]], dtype=np.int32), (2, 1)),
        "support_mass": np.full(2, 0.75, dtype=np.float32),
        "raw_candidate_rewards": rng.uniform(0.0, 1.0, size=(2, 3)).astype(np.float32),
        "reference_logp_full": np.full((2, 4), -np.log(4), dtype=np.float32),
    }
    panel = [
        PanelTarget("a", 100, 0, 2, 10, 3, 0.8, True, False, True),
        PanelTarget("b", 200, 1, 3, 40, 6, 0.6, True, True, False),
    ]
    item_ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    rm_fit_counts = np.asarray([0, 2, 12, 120], dtype=np.int64)
    baseline = evaluate_full_catalog_policy(
        panel=panel, full_reference_logp=dev["reference_logp_full"],
        support_indices=dev["support_indices"], support_mass=dev["support_mass"],
        policy_conditional=np.full((2, 3), 1 / 3, dtype=np.float32),
        item_ids=item_ids, rm_fit_counts=rm_fit_counts,
    )
    data = {
        "train_meta": {"bound_identity_sha256": "train-cache",
                       "actor": {"checkpoint_sha256": "actor"},
                       "optimized_rm": {"sha256": "reward-model"}}, "train_arrays": train,
        "dev_meta": {"bound_identity_sha256": "dev-cache"}, "dev_arrays": dev,
        "dev_panel": panel, "scaler": scaler, "scale": FrozenRewardScale(0.5, 0.25),
        "baseline_dev": baseline, "item_ids": item_ids, "rm_fit_counts": rm_fit_counts,
    }
    config = {"train": {"dev_guard_tolerance": 0.02, "check_every": 2, "minimum_updates": 2,
                         "max_updates": 4, "batch_contexts": 2, "samples_per_context": 2,
                         "hidden_dim": 8, "learning_rate": 1e-3, "objective_min_delta": 1e6,
                         "patience_checks": 10, "bootstrap_draws": 50}}
    return data, config


def _run(run_dir, data, config, *, resume):
    return trainer._run_one(
        run_dir=run_dir, run_name="resume_test", objective="exact_expected_reward", beta=0.5, seed=17,
        variant="full", columns=tuple(range(len(POLICY_FEATURE_NAMES))), config=config,
        train_meta=data["train_meta"], train_arrays=data["train_arrays"],
        dev_meta=data["dev_meta"], dev_arrays=data["dev_arrays"], dev_panel=data["dev_panel"],
        scaler=data["scaler"], scale=data["scale"], baseline_dev=data["baseline_dev"],
        item_ids=data["item_ids"], rm_fit_counts=data["rm_fit_counts"],
        device=torch.device("cpu"), resume=resume,
    )


def test_policy_resume_matches_uninterrupted_fit(tmp_path, monkeypatch):
    data, config = _fixture_data()
    original_save = trainer._atomic_torch_save
    interrupted = False

    def save_then_interrupt(payload, target):
        nonlocal interrupted
        original_save(payload, target)
        if not interrupted and payload.get("updates_done") == 2:
            interrupted = True
            raise RuntimeError("simulated process interruption after durable checkpoint")

    monkeypatch.setattr(trainer, "_atomic_torch_save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        _run(tmp_path / "resumed", data, config, resume=False)
    monkeypatch.setattr(trainer, "_atomic_torch_save", original_save)
    resumed = _run(tmp_path / "resumed", data, config, resume=True)
    uninterrupted = _run(tmp_path / "continuous", data, config, resume=False)

    assert resumed["updates_completed"] == uninterrupted["updates_completed"] == 4
    resumed_state = torch.load(tmp_path / "resumed" / "best.pt", map_location="cpu", weights_only=False)["state_dict"]
    continuous_state = torch.load(tmp_path / "continuous" / "best.pt", map_location="cpu", weights_only=False)["state_dict"]
    assert resumed_state.keys() == continuous_state.keys()
    assert all(torch.equal(resumed_state[key], continuous_state[key]) for key in resumed_state)
    curve = [json.loads(line) for line in (tmp_path / "resumed" / "curve.jsonl").read_text().splitlines()]
    assert [row["step"] for row in curve] == [0, 2, 4]
