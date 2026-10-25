from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tlab.data.manifest import sha256_file
from tlab.data.store import SequenceStore
from tlab.evaluation.metrics import paired_user_bootstrap
from tlab.evaluation.rm_controls import evaluate_rm_controls
from tlab.models.reward import CandidateRewardModel, RewardModelShape
from tlab.train.reward_model import collate_reward_examples, sample_positions, sample_positions_uniform_user


def _calibration_bins(
    targets: list[float],
    predictions: list[float],
    bins: int = 10,
) -> list[dict[str, float | int]]:
    """Return equal-frequency bins sorted by prediction for reliability plots."""
    if len(targets) != len(predictions):
        raise ValueError("calibration targets and predictions must have equal length")
    if bins < 1:
        raise ValueError("calibration bins must be positive")
    if not targets:
        return []
    order = sorted(range(len(targets)), key=lambda index: (predictions[index], index))
    count = min(bins, len(order))
    result: list[dict[str, float | int]] = []
    for bin_index in range(count):
        selected = order[bin_index * len(order) // count:(bin_index + 1) * len(order) // count]
        result.append({
            "count": len(selected),
            "mean_predicted": sum(predictions[index] for index in selected) / len(selected),
            "mean_target": sum(targets[index] for index in selected) / len(selected),
        })
    return result


def _like_metrics(targets: list[float], predictions: list[float]) -> dict[str, Any]:
    """Summarize the like head against the constant-prevalence predictor."""
    if len(targets) != len(predictions):
        raise ValueError("like targets and predictions must have equal length")
    if not targets:
        return {"n": 0, "positive_count": 0, "prevalence": None, "q3_gate_pass": False}
    y = np.asarray(targets, dtype=np.float64)
    p = np.clip(np.asarray(predictions, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    prevalence = float(y.mean())
    positive_count = int(y.sum())
    log_loss = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log1p(-p))))
    brier = float(np.mean((p - y) ** 2))
    baseline_log_loss = float(
        -(prevalence * math.log(max(prevalence, 1e-7))
          + (1.0 - prevalence) * math.log(max(1.0 - prevalence, 1e-7)))
    )
    baseline_brier = prevalence * (1.0 - prevalence)
    ap: float | None = None
    if positive_count:
        order = np.argsort(-p, kind="mergesort")
        sorted_positive = y[order] > 0.5
        ranks = np.flatnonzero(sorted_positive) + 1
        cumulative_positive = np.cumsum(sorted_positive)
        ap = float(np.mean(cumulative_positive[ranks - 1] / ranks))
    gate = bool(
        positive_count >= 200
        and ap is not None and ap > prevalence
        and log_loss <= baseline_log_loss
        and brier <= baseline_brier
    )
    return {
        "n": int(len(y)),
        "positive_count": positive_count,
        "prevalence": prevalence,
        "average_precision": ap,
        "log_loss": log_loss,
        "brier": brier,
        "constant_prevalence_log_loss": baseline_log_loss,
        "constant_prevalence_brier": baseline_brier,
        "q3_gate_pass": gate,
    }


def _ratio_target(user, position: int) -> float | None:
    raw = float(user.ratio_raw[position])
    if not math.isfinite(raw):
        return None
    return min(1.0, max(0.0, raw))


def _fit_item_mean(
    store: SequenceStore,
    examples: list[tuple[int, int]],
    smoothing: float = 20.0,
) -> tuple[float, dict[int, float]]:
    global_sum = 0.0
    global_count = 0
    item_sum: dict[int, float] = defaultdict(float)
    item_count: dict[int, int] = defaultdict(int)
    for user_index, position in examples:
        user = store.users[user_index]
        target = _ratio_target(user, position)
        if target is None:
            continue
        token = int(user.tokens[position])
        global_sum += target
        global_count += 1
        item_sum[token] += target
        item_count[token] += 1
    if global_count == 0:
        raise ValueError("RM baseline cannot be fit without ratio labels")
    mean = global_sum / global_count
    item_means = {
        token: (value + smoothing * mean) / (item_count[token] + smoothing)
        for token, value in item_sum.items()
    }
    return mean, item_means


def _summarize_errors(
    rows: list[tuple[int, float, float, float, bool, float, float]],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    by_user: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"model": [], "item_mean": [], "order_shuffle": [], "user_history_swap": []}
    )
    supported_rows = [row for row in rows if row[4]]
    for uid, target, model_prediction, baseline_prediction, supported, *rest in rows:
        by_user[uid]["model"].append((model_prediction - target) ** 2)
        by_user[uid]["item_mean"].append((baseline_prediction - target) ** 2)
    # Distinct controls: shuffle event order within the same user, or swap in
    # another user's history while keeping the evaluated candidate fixed.
    for row in rows:
        uid, target, model_prediction, baseline_prediction, supported, order_prediction, user_swap_prediction = row
        by_user[uid]["order_shuffle"].append((float(order_prediction) - target) ** 2)
        by_user[uid]["user_history_swap"].append((float(user_swap_prediction) - target) ** 2)
    user_model_mse = {uid: sum(values["model"]) / len(values["model"]) for uid, values in by_user.items()}
    user_baseline_mse = {uid: sum(values["item_mean"]) / len(values["item_mean"]) for uid, values in by_user.items()}
    user_order_shuffle_mse = {
        uid: sum(values["order_shuffle"]) / len(values["order_shuffle"])
        for uid, values in by_user.items() if values["order_shuffle"]
    }
    user_history_swap_mse = {
        uid: sum(values["user_history_swap"]) / len(values["user_history_swap"])
        for uid, values in by_user.items() if values["user_history_swap"]
    }
    model_mse_diff = {uid: value - user_baseline_mse[uid] for uid, value in user_model_mse.items()}
    result: dict[str, Any] = {
        "n_users": len(user_model_mse),
        "n_targets": len(rows),
        "macro_user_model_rmse": (sum(user_model_mse.values()) / len(user_model_mse)) ** 0.5,
        "macro_user_item_mean_rmse": (sum(user_baseline_mse.values()) / len(user_baseline_mse)) ** 0.5,
        "paired_user_mse_difference_model_minus_item_mean": paired_user_bootstrap(
            model_mse_diff, repetitions=repetitions, seed=seed,
        ),
        "supported_n_targets": len(supported_rows),
    }
    if user_order_shuffle_mse:
        result["macro_user_order_shuffle_rmse"] = (
            sum(user_order_shuffle_mse.values()) / len(user_order_shuffle_mse)
        ) ** 0.5
        shared_users = sorted(set(user_order_shuffle_mse).intersection(user_model_mse))
        result["macro_user_history_order_delta_rmse"] = result["macro_user_order_shuffle_rmse"] - (
            sum(user_model_mse[uid] for uid in shared_users) / len(shared_users)
        ) ** 0.5
    if user_history_swap_mse:
        result["macro_user_history_swap_rmse"] = (
            sum(user_history_swap_mse.values()) / len(user_history_swap_mse)
        ) ** 0.5
        shared_users = sorted(set(user_history_swap_mse).intersection(user_model_mse))
        result["macro_user_history_swap_delta_rmse"] = result["macro_user_history_swap_rmse"] - (
            sum(user_model_mse[uid] for uid in shared_users) / len(shared_users)
        ) ** 0.5
    if supported_rows:
        supported_users: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"model": [], "baseline": []})
        for uid, target, prediction, baseline, supported, *rest in supported_rows:
            supported_users[uid]["model"].append((prediction - target) ** 2)
            supported_users[uid]["baseline"].append((baseline - target) ** 2)
        model_mse = [sum(values["model"]) / len(values["model"]) for values in supported_users.values()]
        baseline_mse = [sum(values["baseline"]) / len(values["baseline"]) for values in supported_users.values()]
        result["supported_macro_user_model_rmse"] = (sum(model_mse) / len(model_mse)) ** 0.5
        result["supported_macro_user_item_mean_rmse"] = (sum(baseline_mse) / len(baseline_mse)) ** 0.5
    return result


@torch.no_grad()
def diagnose_rm_checkpoint(
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
    import polars as pl

    from tlab.config import load_config

    diagnostics_started = time.perf_counter()
    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    raw_dir = Path(config.data.get("source_dir", "data/raw"))
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(
        processed_dir / "sequence_index", expected_source_sha256=protocol["processed_listens_sha256"],
    )
    train_users = [user for user in store.users if user.rm_user_partition == variant]
    if not train_users:
        raise ValueError(f"no users in RM {variant} partition")
    train_examples = sample_positions(train_users, "rm_fit_positions", train_limit, seed)
    validation_examples = sample_positions_uniform_user(
        store.users, "rm_validation_positions", validation_limit, seed + 100_003,
    )
    if not train_examples or not validation_examples:
        raise ValueError("RM diagnostics need F_RM training and V_RM validation examples")
    train_store = SequenceStore(train_users, store.events, store.tie_blocks, store.tied_events)
    validation_store = store

    feature_frame = pl.read_parquet(processed_dir / f"rm_item_features_{variant}.parquet")
    item_features = {
        int(row["token_id"]): (
            float(row["rm_duration_seconds"] or 0.0),
            bool(row["rm_duration_missing"]),
            int(row["rm_training_support"]),
        )
        for row in feature_frame.iter_rows(named=True)
    }
    like_labels_by_uid: dict[int, np.ndarray] | None = None
    like_manifest_path = processed_dir / "like_labels_30m.json"
    if like_manifest_path.is_file():
        like_manifest = json.loads(like_manifest_path.read_text(encoding="utf-8"))
        like_labels_path = processed_dir / "like_labels_30m.bin"
        if (
            like_manifest.get("source_listens_sha256") != protocol["processed_listens_sha256"]
            or int(like_manifest.get("window_seconds", 0)) != 1800
            or like_manifest.get("source_likes_sha256") != sha256_file(raw_dir / "likes.parquet")
            or like_manifest.get("source_unlikes_sha256") != sha256_file(raw_dir / "unlikes.parquet")
            or like_manifest.get("labels_sha256") != sha256_file(like_labels_path)
        ):
            raise ValueError("30-minute like label index is stale; rebuild it with tlab.data.like_index")
        labels = np.memmap(
            like_labels_path, dtype="i1", mode="r", shape=(int(like_manifest["sequence_events"]),),
        )
        like_labels_by_uid = {
            int(entry["uid"]): labels[int(entry["offset"]):int(entry["offset"]) + int(entry["length"])]
            for entry in like_manifest["users"]
        }
        for user in store.users:
            user_labels = like_labels_by_uid.get(user.uid)
            if user_labels is None or len(user_labels) != len(user.tokens):
                raise ValueError(f"like label index does not align with cached sequence for uid={user.uid}")
    global_mean, item_mean = _fit_item_mean(train_store, train_examples)
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
    randomizer = random.Random(seed + 700_001)
    validation_examples_by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for example in validation_examples:
        validation_examples_by_user[example[0]].append(example)
    eligible_validation_users = list(validation_examples_by_user)
    history_swap_rng = random.Random(seed + 700_002)
    history_swap_donors: list[tuple[int, int]] = []
    for user_index, _ in validation_examples:
        other_users = [index for index in eligible_validation_users if index != user_index]
        if other_users:
            donor_user = history_swap_rng.choice(other_users)
            history_swap_donors.append(history_swap_rng.choice(validation_examples_by_user[donor_user]))
        else:
            history_swap_donors.append((user_index, validation_examples_by_user[user_index][0][1]))
    history_length = int(config.dataset.get("history_length", 200))
    seconds_cap = float(protocol["seconds_cap_f_rm_p99_5"])
    seconds_scale = float(protocol["seconds_scale_f_rm_p95"])
    rows: list[tuple[int, float, float, float, bool, float, float]] = []
    ratio_calibration_targets: list[float] = []
    ratio_calibration_predictions: list[float] = []
    seconds_calibration_targets: list[float] = []
    seconds_calibration_predictions: list[float] = []
    like_calibration_targets: list[float] = []
    like_calibration_predictions: list[float] = []
    unsupported = 0
    model_forward_candidate_evaluations = 0
    model_forward_seconds = 0.0
    for start in range(0, len(validation_examples), batch_size):
        examples = validation_examples[start:start + batch_size]
        batch = collate_reward_examples(
            validation_store, examples, item_features, seconds_cap, seconds_scale,
            history_length, device, like_labels_by_uid,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_started = time.perf_counter()
        full_prediction = model(*batch[:6])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_forward_seconds += time.perf_counter() - forward_started
        model_forward_candidate_evaluations += int(batch[2].numel())
        prediction = full_prediction["ratio"].float().cpu().tolist()
        ratio_mask = batch[7].detach().cpu().tolist()
        ratio_targets = batch[6].detach().cpu().tolist()
        ratio_values = full_prediction["ratio"].float().detach().cpu().tolist()
        for is_valid, target_value, predicted_value in zip(ratio_mask, ratio_targets, ratio_values):
            if is_valid:
                ratio_calibration_targets.append(float(target_value))
                ratio_calibration_predictions.append(float(predicted_value))
        seconds_mask = batch[9].detach().cpu().tolist()
        seconds_targets = (batch[8] * seconds_scale).detach().cpu().tolist()
        seconds_values = (full_prediction["seconds_scaled"].float() * seconds_scale).detach().cpu().tolist()
        for is_valid, target_value, predicted_value in zip(seconds_mask, seconds_targets, seconds_values):
            if is_valid:
                seconds_calibration_targets.append(float(target_value))
                seconds_calibration_predictions.append(float(predicted_value))
        like_mask = batch[11].detach().cpu().tolist()
        like_targets = batch[10].detach().cpu().tolist()
        like_values = full_prediction["like_probability"].float().detach().cpu().tolist()
        for is_valid, target_value, predicted_value in zip(like_mask, like_targets, like_values):
            if is_valid:
                like_calibration_targets.append(float(target_value))
                like_calibration_predictions.append(float(predicted_value))
        shuffled_ids = batch[0].clone()
        for row in range(shuffled_ids.shape[0]):
            valid_length = int(batch[1][row].sum().item())
            if valid_length > 2:
                events = shuffled_ids[row, 1:valid_length].tolist()
                randomizer.shuffle(events)
                shuffled_ids[row, 1:valid_length] = torch.tensor(events, dtype=torch.long, device=device)
        shuffled_prediction = model(shuffled_ids, batch[1], *batch[2:6])["ratio"].float().cpu().tolist()
        donor_examples = history_swap_donors[start:start + len(examples)]
        donor_tensors = collate_reward_examples(
            validation_store, donor_examples, item_features, seconds_cap, seconds_scale,
            history_length, device, like_labels_by_uid,
        )
        donor_histories = [validation_store.users[user_index].history(position, history_length)
                           for user_index, position in donor_examples]
        swapped_repeat = torch.tensor(
            [int(int(candidate) in set(history[-20:]))
             for candidate, history in zip(batch[2].detach().cpu().tolist(), donor_histories)],
            dtype=torch.bool, device=device,
        )
        swapped_history_prediction = model(
            donor_tensors[0], donor_tensors[1], batch[2], batch[3], batch[4], swapped_repeat,
        )["ratio"].float().cpu().tolist()
        targets = batch[6].cpu().tolist()
        valid = batch[7].cpu().tolist()
        tokens = batch[2].cpu().tolist()
        for offset, ((user_index, position), target, model_value, order_value, history_swap_value, is_valid, token) in enumerate(
            zip(examples, targets, prediction, shuffled_prediction, swapped_history_prediction, valid, tokens)
        ):
            if not is_valid:
                continue
            user = validation_store.users[user_index]
            support = int(item_features.get(int(token), (0.0, True, 0))[2])
            unsupported += int(support == 0)
            baseline = item_mean.get(int(token), global_mean)
            rows.append((user.uid, float(target), float(model_value), float(baseline), support > 0,
                         float(order_value), float(history_swap_value)))
    # Restrict the primary paired comparison to supported items; retain all-target metrics too.
    all_metrics = _summarize_errors(rows, bootstrap_repetitions, seed)
    supported_metrics = _summarize_errors([row for row in rows if row[4]], bootstrap_repetitions, seed + 1)
    result = {
        "variant": variant,
        "seed": seed,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "train_examples": len(train_examples),
        "train_users": len(train_users),
        "validation_population": "all fit/audit users with V_RM labels",
        "validation_examples_requested": len(validation_examples),
        "validation_ratio_examples": len(rows),
        "validation_unsupported_count": unsupported,
        "validation_unsupported_fraction": unsupported / len(rows) if rows else None,
        "model_forward_candidate_evaluations": model_forward_candidate_evaluations,
        "model_forward_seconds": model_forward_seconds,
        "model_candidate_evaluations_per_second": (
            model_forward_candidate_evaluations / model_forward_seconds if model_forward_seconds > 0 else None
        ),
        "elapsed_seconds": time.perf_counter() - diagnostics_started,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "ratio_training_global_mean": global_mean,
        "item_mean_smoothing_alpha": 20.0,
        "all_items": all_metrics,
        "supported_items": supported_metrics,
        "calibration_deciles": {
            "ratio": _calibration_bins(ratio_calibration_targets, ratio_calibration_predictions),
            "seconds": _calibration_bins(seconds_calibration_targets, seconds_calibration_predictions),
            "like": _calibration_bins(like_calibration_targets, like_calibration_predictions),
        },
        "like_quality": _like_metrics(like_calibration_targets, like_calibration_predictions),
        "simple_controls": evaluate_rm_controls(
            train_store=train_store,
            validation_store=validation_store,
            train_examples=train_examples,
            validation_examples=validation_examples,
            item_features=item_features,
            global_mean=global_mean,
            item_mean=item_mean,
            history_length=history_length,
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed,
        ),
    }
    target = Path(output_path) if output_path else root / "artifacts" / f"rm_diagnostics_{variant}_seed{seed}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    result["output_path"] = str(target.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen RM checkpoints with item-mean and history-shuffle controls")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=("fit", "audit"), default="fit")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-limit", type=int, default=20_000)
    parser.add_argument("--validation-limit", type=int, default=40_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = diagnose_rm_checkpoint(
        args.config, args.checkpoint, variant=args.variant, seed=args.seed,
        train_limit=args.train_limit, validation_limit=args.validation_limit,
        batch_size=args.batch_size, output_path=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
