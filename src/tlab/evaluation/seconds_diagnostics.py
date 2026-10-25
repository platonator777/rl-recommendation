from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from tlab.data.store import SequenceStore, UserSequence
from tlab.evaluation.metrics import paired_user_bootstrap
from tlab.models.reward import CandidateRewardModel, RewardModelShape
from tlab.train.checkpoint import source_tree_sha256
from tlab.train.reward_model import collate_reward_examples, sample_positions, sample_positions_uniform_user


def _target_seconds(user: UserSequence, position: int, cap: float) -> float | None:
    ratio = float(user.ratio_raw[position])
    duration = float(user.duration_seconds[position])
    if not math.isfinite(ratio) or not math.isfinite(duration) or duration <= 0:
        return None
    return min(cap, max(0.0, ratio * duration))


def _baseline_features(
    store: SequenceStore,
    examples: Sequence[tuple[int, int]],
    item_features: dict[int, tuple[float, bool, int]],
    history_length: int,
    seconds_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    numeric: list[list[float]] = []
    candidate_numeric: list[list[float]] = []
    candidate_ids: list[int] = []
    targets: list[float] = []
    uids: list[int] = []
    for user_index, position in examples:
        user = store.users[user_index]
        target = _target_seconds(user, position, seconds_cap)
        if target is None:
            continue
        token = int(user.tokens[position])
        duration, missing, support = item_features.get(token, (0.0, True, 0))
        history = user.history(position, history_length)
        repeat = float(token in history[-20:])
        numeric.append([
            math.log1p(max(0.0, duration)),
            float(missing),
            math.log1p(max(0, support)),
            repeat,
            math.log1p(max(0, len(history) - 1)),
        ])
        candidate_numeric.append([math.log1p(max(0.0, duration)), float(missing)])
        candidate_ids.append(token)
        targets.append(target)
        uids.append(int(user.uid))
    return (
        np.asarray(numeric, dtype=np.float32),
        np.asarray(candidate_ids, dtype=np.int64),
        np.asarray(candidate_numeric, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(uids, dtype=np.int64),
    )


def _macro_user_rmse(uids: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> float:
    per_user: dict[int, list[float]] = defaultdict(list)
    for uid, target, prediction in zip(uids, targets, predictions):
        per_user[int(uid)].append((float(prediction) - float(target)) ** 2)
    return math.sqrt(sum(sum(values) / len(values) for values in per_user.values()) / len(per_user))


def _paired_mse_difference(
    uids: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    per_user: dict[int, list[float]] = defaultdict(list)
    for uid, target, prediction, base in zip(uids, targets, predictions, baseline):
        per_user[int(uid)].append((float(prediction) - float(target)) ** 2 - (float(base) - float(target)) ** 2)
    return paired_user_bootstrap(
        {uid: sum(values) / len(values) for uid, values in per_user.items()},
        repetitions=repetitions,
        seed=seed,
    )


@torch.no_grad()
def diagnose_seconds_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    variant: str = "fit",
    seed: int = 17,
    train_limit: int = 20_000,
    validation_limit: int = 40_000,
    batch_size: int = 64,
    bootstrap_repetitions: int = 2000,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare the learned seconds head with activity, popularity, and candidate controls on V_RM."""
    import polars as pl

    from tlab.config import load_config
    from tlab.data.manifest import sha256_file

    started = time.perf_counter()
    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(
        processed_dir / "sequence_index", expected_source_sha256=protocol["processed_listens_sha256"],
    )
    train_users = [user for user in store.users if user.rm_user_partition == variant]
    if not train_users:
        raise ValueError(f"no users in RM {variant} partition")
    train_store = SequenceStore(train_users, store.events, store.tie_blocks, store.tied_events)
    train_examples = sample_positions(train_users, "rm_fit_positions", train_limit, seed)
    validation_examples = sample_positions_uniform_user(
        store.users, "rm_validation_positions", validation_limit, seed + 100_003,
    )
    feature_path = processed_dir / f"rm_item_features_{variant}.parquet"
    feature_frame = pl.read_parquet(feature_path)
    item_features = {
        int(row["token_id"]): (
            float(row["rm_duration_seconds"] or 0.0),
            bool(row["rm_duration_missing"]),
            int(row["rm_training_support"]),
        )
        for row in feature_frame.iter_rows(named=True)
    }
    cap = float(protocol["seconds_cap_f_rm_p99_5"])
    scale = float(protocol["seconds_scale_f_rm_p95"])
    history_length = int(config.dataset.get("history_length", 200))
    train_numeric, train_ids, train_candidate_numeric, train_y, _ = _baseline_features(
        train_store, train_examples, item_features, history_length, cap,
    )
    val_numeric, val_ids, val_candidate_numeric, val_y, val_uids = _baseline_features(
        store, validation_examples, item_features, history_length, cap,
    )
    if len(train_y) == 0 or len(val_y) == 0:
        raise ValueError("seconds diagnostics require valid duration and ratio labels in train and V_RM")

    global_mean = float(train_y.mean())
    item_sum: dict[int, float] = defaultdict(float)
    item_count: dict[int, int] = defaultdict(int)
    for token, target in zip(train_ids, train_y):
        item_sum[int(token)] += float(target)
        item_count[int(token)] += 1
    item_mean = {
        token: (value + 20.0 * global_mean) / (item_count[token] + 20.0)
        for token, value in item_sum.items()
    }
    global_prediction = np.full(len(val_y), global_mean, dtype=np.float32)
    item_prediction = np.asarray([item_mean.get(int(token), global_mean) for token in val_ids], dtype=np.float32)

    numeric_model = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed,
    )
    numeric_model.fit(train_numeric, train_y)
    activity_prediction = np.clip(numeric_model.predict(val_numeric), 0.0, cap)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    train_onehot = encoder.fit_transform(train_ids.reshape(-1, 1))
    val_onehot = encoder.transform(val_ids.reshape(-1, 1))
    candidate_train = hstack((train_onehot, csr_matrix(train_candidate_numeric)), format="csr", dtype=np.float32)
    candidate_val = hstack((val_onehot, csr_matrix(val_candidate_numeric)), format="csr", dtype=np.float32)
    candidate_scores: dict[str, float] = {}
    candidate_models: dict[str, Ridge] = {}
    for alpha in (1.0, 10.0, 100.0):
        candidate_model = Ridge(alpha=alpha, solver="lsqr")
        candidate_model.fit(candidate_train, train_y)
        candidate_models[str(alpha)] = candidate_model
        prediction = np.clip(candidate_model.predict(candidate_val), 0.0, cap)
        candidate_scores[str(alpha)] = _macro_user_rmse(val_uids, val_y, prediction)
    chosen_alpha = min(candidate_scores, key=candidate_scores.get)
    candidate_prediction = np.clip(candidate_models[chosen_alpha].predict(candidate_val), 0.0, cap)

    shape = RewardModelShape(
        vocab_size=int(config.dataset.get("catalog_size", 20_000)) + 3,
        hidden_size=int(config.training.get("rm_hidden_size", 128)),
        n_layer=int(config.training.get("rm_layers", 2)),
        n_head=int(config.training.get("rm_heads", 4)),
        max_positions=int(config.model.get("n_positions", 256)),
        encoder="transformer" if variant == "fit" else "gru",
    )
    model = CandidateRewardModel(shape)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model_predictions: list[float] = []
    model_targets: list[float] = []
    model_uids: list[int] = []
    model_candidate_ids: list[int] = []
    for start in range(0, len(validation_examples), batch_size):
        examples = validation_examples[start:start + batch_size]
        batch = collate_reward_examples(
            store, examples, item_features, cap, scale, history_length, device,
        )
        predicted = (model(*batch[:6])["seconds_scaled"].float() * scale).cpu().numpy()
        targets = (batch[8].float() * scale).cpu().numpy()
        mask = batch[9].cpu().numpy().astype(bool)
        candidate_ids = batch[2].cpu().numpy()
        for row, (user_index, _) in enumerate(examples):
            if mask[row]:
                model_predictions.append(float(predicted[row]))
                model_targets.append(float(targets[row]))
                model_uids.append(int(store.users[user_index].uid))
                model_candidate_ids.append(int(candidate_ids[row]))

    # Collation uses the same validation examples; filter every baseline to its exact seconds mask.
    filtered_uids: list[int] = []
    filtered_targets: list[float] = []
    for user_index, position in validation_examples:
        target = _target_seconds(store.users[user_index], position, cap)
        if target is not None:
            filtered_uids.append(int(store.users[user_index].uid))
            filtered_targets.append(target)
    # The baseline feature pass and model collation share the same validity rule and stable example order.
    if not np.array_equal(np.asarray(filtered_uids), np.asarray(val_uids)) or not np.allclose(filtered_targets, val_y):
        raise RuntimeError("seconds validation ordering mismatch")
    # Candidate model predictions are already aligned to the baseline examples; filter by model-valid order.
    model_targets_array = np.asarray(model_targets, dtype=np.float32)
    model_uids_array = np.asarray(model_uids, dtype=np.int64)
    model_predictions_array = np.asarray(model_predictions, dtype=np.float32)
    # The collator marks the same examples valid as _target_seconds; assert label alignment.
    if len(model_targets_array) != len(val_y) or not np.allclose(model_targets_array, val_y, atol=1e-4):
        raise RuntimeError("seconds validation labels differ between baseline and reward-model paths")

    baselines = {
        "global_mean": global_prediction,
        "item_mean": item_prediction,
        "duration_popularity_activity": activity_prediction,
        "candidate_only": candidate_prediction,
    }
    strongest_name = min(baselines, key=lambda name: _macro_user_rmse(val_uids, val_y, baselines[name]))
    result = {
        "variant": variant,
        "seed": seed,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
        "source_sha256": source_tree_sha256(root / "src" / "tlab"),
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "rm_item_features_sha256": protocol["rm_item_features_by_partition"][variant]["sha256"],
        "train_users": len(train_users),
        "validation_population": "all fit/audit users with V_RM labels",
        "validation_users": int(len(np.unique(val_uids))),
        "seconds_cap": cap,
        "seconds_scale": scale,
        "n_train_seconds_labels": int(len(train_y)),
        "n_validation_seconds_labels": int(len(val_y)),
        "validation_users": int(len(np.unique(val_uids))),
        "validation_seconds_mean": float(val_y.mean()),
        "validation_seconds_median": float(np.median(val_y)),
        "controls": {
            name: {
                "macro_user_rmse_seconds": _macro_user_rmse(val_uids, val_y, prediction),
                "prediction_mean_seconds": float(np.mean(prediction)),
            }
            for name, prediction in baselines.items()
        },
        "candidate_only_selected_alpha_on_v_rm": chosen_alpha,
        "candidate_only_alpha_rmse_seconds": candidate_scores,
        "strongest_simple_control": strongest_name,
        "reward_model": {
            "macro_user_rmse_seconds": _macro_user_rmse(model_uids_array, model_targets_array, model_predictions_array),
            "prediction_mean_seconds": float(model_predictions_array.mean()),
            "paired_mse_difference_vs_item_mean": _paired_mse_difference(
                model_uids_array, model_targets_array, model_predictions_array, item_prediction,
                bootstrap_repetitions, seed,
            ),
            "paired_mse_difference_vs_strongest_simple_control": _paired_mse_difference(
                model_uids_array, model_targets_array, model_predictions_array, baselines[strongest_name],
                bootstrap_repetitions, seed + 1,
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
    }
    target = Path(output_path) if output_path else root / "artifacts" / f"seconds_diagnostics_{variant}_seed{seed}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    result["output_path"] = str(target.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the seconds reward head with simple V_RM controls")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=("fit", "audit"), default="fit")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-limit", type=int, default=20_000)
    parser.add_argument("--validation-limit", type=int, default=40_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = diagnose_seconds_checkpoint(
        args.config, args.checkpoint, variant=args.variant, seed=args.seed,
        train_limit=args.train_limit, validation_limit=args.validation_limit,
        batch_size=args.batch_size, output_path=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
