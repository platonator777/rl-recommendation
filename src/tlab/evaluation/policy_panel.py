from __future__ import annotations

import argparse
import itertools
import json
import math
import time
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import numpy as np

from tlab.data.store import SequenceStore
from tlab.evaluation.metrics import paired_user_bootstrap
from tlab.evaluation.behavior import duration_quantile_from_train, exposure_metrics, repeat_rate
from tlab.evaluation.rewards import RewardScale
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.models.reward import CandidateRewardModel, RewardModelShape
from tlab.train.objectives import analytic_reward_kl_policy, full_categorical_kl
from tlab.train.policy import _collate_histories


def _pairwise_jaccard(recommendations: list[list[int]], seed: int, max_pairs: int = 10_000) -> dict[str, float | int]:
    count = len(recommendations)
    total_pairs = count * (count - 1) // 2
    if total_pairs == 0:
        return {"mean_jaccard": 0.0, "n_pairs": 0}
    if total_pairs <= max_pairs:
        pairs = list(itertools.combinations(range(count), 2))
    else:
        rng = random.Random(seed)
        selected: set[tuple[int, int]] = set()
        while len(selected) < max_pairs:
            left = rng.randrange(count)
            right = rng.randrange(count - 1)
            if right >= left:
                right += 1
            selected.add((min(left, right), max(left, right)))
        pairs = sorted(selected)
    scores = []
    for left, right in pairs:
        a, b = set(recommendations[left]), set(recommendations[right])
        union = a | b
        scores.append(len(a & b) / len(union) if union else 0.0)
    return {"mean_jaccard": sum(scores) / len(scores), "n_pairs": len(scores)}


def _load_reward_model(config, checkpoint_path: str | Path, variant: str, device: torch.device):
    import polars as pl

    from tlab.config import load_config

    model_config = config.model
    training_config = config.training
    catalog_size = int(config.dataset.get("catalog_size", 20_000))
    model = CandidateRewardModel(RewardModelShape(
        vocab_size=catalog_size + 3,
        hidden_size=int(training_config.get("rm_hidden_size", 128)),
        n_layer=int(training_config.get("rm_layers", 2)),
        n_head=int(training_config.get("rm_heads", 4)),
        max_positions=int(model_config.get("n_positions", 256)),
        encoder="transformer" if variant == "fit" else "gru",
    ))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = config.path.parent.parent / processed_dir
    features = pl.read_parquet(processed_dir / f"rm_item_features_{variant}.parquet")
    item_features = {
        int(row["token_id"]): (
            float(row["rm_duration_seconds"] or 0.0),
            bool(row["rm_duration_missing"]),
            int(row["rm_training_support"]),
        )
        for row in features.iter_rows(named=True)
    }
    return model, item_features


def _evaluate_policy_panel(
    config_path: str | Path,
    sft_checkpoint: str | Path,
    policy_checkpoint: str | Path,
    rm_fit_checkpoint: str | Path,
    rm_audit_checkpoint: str | Path,
    reward_scale_path: str | Path,
    beta: float = 0.1,
    seed: int = 17,
    anchor_count: int | None = 128,
    batch_size: int = 4,
    output_path: str | Path | None = None,
    policy_label: str | None = None,
    reward_head: str = "ratio",
    repeat_penalty: float = 0.0,
    panel_split: str = "A_RL",
) -> dict[str, Any]:
    import polars as pl

    from tlab.config import load_config

    if reward_head not in {"ratio", "seconds"} or repeat_penalty < 0 or (repeat_penalty > 0 and reward_head != "ratio"):
        raise ValueError("reward head or repeat penalty is invalid")
    if anchor_count is not None and anchor_count < 1:
        raise ValueError("anchor_count must be positive or None for all eligible users")
    panel_started = time.perf_counter()
    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    seconds_scale = float(protocol["seconds_scale_f_rm_p95"]) if reward_head == "seconds" else 1.0
    if policy_label is None:
        run_manifest = Path(policy_checkpoint).parent / "run_manifest.json"
        if run_manifest.exists():
            run_metadata = json.loads(run_manifest.read_text(encoding="utf-8"))
            policy_label = str(run_metadata.get("algorithm", run_metadata.get("stage", "POLICY"))).upper()
        else:
            policy_label = "POLICY"
    if policy_label in {"SFT", "B5"}:
        raise ValueError("policy label must be distinct from SFT and B5")
    store = SequenceStore.load_cache(
        processed_dir / "sequence_index",
        expected_source_sha256=protocol["processed_listens_sha256"],
    )
    catalog = pl.read_parquet(processed_dir / "catalog.parquet").sort("token_id")
    catalog_item_ids = catalog.get_column("item_id").to_list()
    training_counts = [int(value) for value in catalog.get_column("training_count").to_list()]
    catalog_durations = catalog.get_column("catalog_duration_seconds").to_list()
    short_duration_threshold = duration_quantile_from_train(store, 0.10)
    head_count = max(1, math.ceil(0.01 * len(catalog_item_ids)))
    head_items = set(sorted(range(len(training_counts)), key=lambda index: (-training_counts[index], catalog_item_ids[index]))[:head_count])
    tokens = torch.arange(3, 3 + len(catalog_item_ids), dtype=torch.long)
    item_ids_tensor = torch.tensor(catalog_item_ids, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
    id_order = torch.argsort(item_ids_tensor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    actor_shape = ActorShape(
        vocab_size=len(catalog_item_ids) + 3,
        n_positions=int(config.model.get("n_positions", 256)),
        n_layer=int(config.model.get("n_layer", 4)),
        n_embd=int(config.model.get("n_embd", 256)),
        n_head=int(config.model.get("n_head", 4)),
        dropout=float(config.model.get("dropout", 0.1)),
    )
    def load_actor(path: str | Path) -> ItemGPT2Actor:
        actor = ItemGPT2Actor(actor_shape)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        actor.load_state_dict(payload["model"])
        return actor.to(device).eval()

    baseline_actor = load_actor(sft_checkpoint)
    policy_actor = load_actor(policy_checkpoint)
    fit_model, fit_features = _load_reward_model(config, rm_fit_checkpoint, "fit", device)
    audit_model, audit_features = _load_reward_model(config, rm_audit_checkpoint, "audit", device)
    scale_payload = json.loads(Path(reward_scale_path).read_text(encoding="utf-8"))
    scale_raw = scale_payload.get("reward_scale", scale_payload)
    reward_scale = RewardScale(
        mean=float(scale_raw["mean"]),
        std=float(scale_raw["std"]),
        sample_count=int(scale_raw["sample_count"]),
    )

    rng = random.Random(seed)
    if panel_split == "A_RL":
        eligible = [index for index, user in enumerate(store.users) if len(user.rl_anchors) > 0]
        rng.shuffle(eligible)
        selected_users = eligible[:len(eligible) if anchor_count is None else min(anchor_count, len(eligible))]
        anchors = [(index, int(rng.choice(store.users[index].rl_anchors))) for index in selected_users]
    elif panel_split in {"validation", "test"}:
        all_anchors = store.evaluation_targets(panel_split, per_user_limit=1, seed=seed)
        if anchor_count is None or anchor_count >= len(all_anchors):
            anchors = all_anchors
        else:
            anchors = sorted(rng.sample(all_anchors, min(anchor_count, len(all_anchors))))
    else:
        raise ValueError("panel_split must be A_RL, validation, or test")
    if not anchors:
        raise ValueError(f"fixed {panel_split} policy panel is empty")
    history_length = int(config.dataset.get("history_length", 200))
    user_ids = [int(store.users[index].uid) for index, _ in anchors]

    per_policy: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ("SFT", policy_label, "B5")
    }
    recommendations: dict[str, list[list[int]]] = {name: [] for name in per_policy}
    recent_catalog_histories: list[set[int]] = []
    for history in store.histories_for(anchors, history_length=history_length):
        recent_catalog_histories.append({int(token) - 3 for token in history[-20:] if 3 <= int(token) < 3 + len(catalog_item_ids)})
    for start in range(0, len(anchors), batch_size):
        batch_anchors = anchors[start:start + batch_size]
        histories = store.histories_for(batch_anchors, history_length=history_length)
        input_ids, attention = _collate_histories(histories, device)
        tokens_batch = tokens.to(device).unsqueeze(0).expand(len(histories), -1)

        def score_rewards(model, features):
            durations, missing, support = [], [], []
            repeats = []
            for history in histories:
                row_duration, row_missing, row_support = [], [], []
                recent = set(int(token) for token in history[-20:] if int(token) >= 3)
                for token in tokens.tolist():
                    duration, duration_missing, count = features.get(int(token), (0.0, True, 0))
                    row_duration.append(float(duration))
                    row_missing.append(bool(duration_missing))
                    row_support.append(int(count))
                durations.append(row_duration)
                missing.append(row_missing)
                support.append(row_support)
                repeats.append([int(token) in recent for token in tokens.tolist()])
            duration_tensor = torch.tensor(durations, dtype=torch.float32, device=device)
            missing_tensor = torch.tensor(missing, dtype=torch.bool, device=device)
            repeat_tensor = torch.tensor(repeats, dtype=torch.bool, device=device)
            prediction = model.score_catalog(
                input_ids, attention, tokens_batch, duration_tensor, missing_tensor,
                repeat_tensor, chunk_size=512,
            )
            reward_values = (
                prediction["ratio"].float()
                if reward_head == "ratio"
                else prediction["seconds_scaled"].float() * seconds_scale
            )
            if repeat_penalty:
                reward_values = reward_values - repeat_penalty * repeat_tensor.float()
            return reward_values, torch.tensor(
                [[count == 0 for count in row] for row in support], dtype=torch.bool, device=device,
            )

        with torch.no_grad():
            baseline_log = baseline_actor.next_action_log_probs(input_ids, attention)[:, 3:]
            policy_log = policy_actor.next_action_log_probs(input_ids, attention)[:, 3:]
            reference_log = baseline_log
            fit_rewards, fit_unsupported = score_rewards(fit_model, fit_features)
            audit_rewards, audit_unsupported = score_rewards(audit_model, audit_features)
            baseline_probability = baseline_log.exp()
            policy_probability = policy_log.exp()
            normalized_fit = (fit_rewards - reward_scale.mean) / max(reward_scale.std, 1e-3)
            b5_probability = analytic_reward_kl_policy(reference_log, normalized_fit, beta)
            log_policies = {
                "SFT": baseline_log,
                policy_label: policy_log,
                "B5": torch.where(b5_probability > 0, b5_probability.log(), torch.full_like(b5_probability, -torch.inf)),
            }
            probabilities = {
                "SFT": baseline_probability,
                policy_label: policy_probability,
                "B5": b5_probability,
            }
            for name, probability in probabilities.items():
                fit_expected = (probability * fit_rewards).sum(dim=-1)
                audit_expected = (probability * audit_rewards).sum(dim=-1)
                fit_unsupported_mass = (probability * fit_unsupported.float()).sum(dim=-1)
                audit_unsupported_mass = (probability * audit_unsupported.float()).sum(dim=-1)
                entropy = -(probability * log_policies[name]).nan_to_num(0.0).sum(dim=-1)
                kl = full_categorical_kl(log_policies[name], reference_log)
                ordered_scores = log_policies[name][:, id_order]
                order = id_order[torch.argsort(ordered_scores, dim=-1, descending=True, stable=True)]
                top1 = order[:, 0]
                top10 = order[:, :10]
                top10_cpu = top10.detach().cpu().tolist()
                for row in range(len(histories)):
                    recommendations[name].append([int(item) for item in top10_cpu[row]])
                    per_policy[name]["v_fit"].append(float(fit_expected[row]))
                    per_policy[name]["v_audit"].append(float(audit_expected[row]))
                    per_policy[name]["fit_top1_reward"].append(float(fit_rewards[row, top1[row]]))
                    per_policy[name]["audit_top1_reward"].append(float(audit_rewards[row, top1[row]]))
                    per_policy[name]["fit_top10_mean_reward"].append(float(fit_rewards[row, top10[row]].mean()))
                    per_policy[name]["audit_top10_mean_reward"].append(float(audit_rewards[row, top10[row]].mean()))
                    per_policy[name]["fit_unsupported_mass"].append(float(fit_unsupported_mass[row]))
                    per_policy[name]["audit_unsupported_mass"].append(float(audit_unsupported_mass[row]))
                    per_policy[name]["entropy"].append(float(entropy[row]))
                    per_policy[name]["kl_to_sft"].append(float(kl[row]))

    summary: dict[str, Any] = {}
    for name, values in per_policy.items():
        summary[name] = {metric: sum(rows) / len(rows) for metric, rows in values.items()}
        summary[name]["n_users"] = len(user_ids)
    paired: dict[str, Any] = {}
    baseline_values = per_policy["SFT"]
    for name in (policy_label, "B5"):
        paired[name] = {}
        for metric in ("v_fit", "v_audit"):
            differences = {
                uid: candidate - reference
                for uid, candidate, reference in zip(user_ids, per_policy[name][metric], baseline_values[metric])
            }
            paired[name][f"{metric}_difference_vs_sft"] = paired_user_bootstrap(
                differences, repetitions=2000, seed=seed,
            )
    per_user_diagnostics: dict[str, Any] = {}
    for index, uid in enumerate(user_ids):
        per_user_diagnostics[str(uid)] = {}
        for name, values in per_policy.items():
            item_indices = recommendations[name][index]
            record = {metric: float(metric_values[index]) for metric, metric_values in values.items()}
            record["top10_item_ids"] = [int(catalog_item_ids[item]) for item in item_indices]
            recent_items = recent_catalog_histories[index]
            record["repeat_rate_top10"] = sum(item in recent_items for item in item_indices) / max(1, len(item_indices))
            record["top1_repeat"] = bool(item_indices and item_indices[0] in recent_items)
            user_durations = [float(catalog_durations[item]) for item in item_indices
                              if catalog_durations[item] is not None and float(catalog_durations[item]) > 0]
            record["top10_mean_duration_seconds"] = sum(user_durations) / len(user_durations) if user_durations else None
            per_user_diagnostics[str(uid)][name] = record
        for name in (policy_label, "B5"):
            per_user_diagnostics[str(uid)][name]["v_fit_difference_vs_sft"] = (
                per_policy[name]["v_fit"][index] - per_policy["SFT"]["v_fit"][index]
            )
            per_user_diagnostics[str(uid)][name]["v_audit_difference_vs_sft"] = (
                per_policy[name]["v_audit"][index] - per_policy["SFT"]["v_audit"][index]
            )
    serving_diagnostics: dict[str, Any] = {}
    for name, rows in recommendations.items():
        flat_items = [item for row in rows for item in row]
        selected_durations = [catalog_durations[item] for item in flat_items]
        known_selected_durations = [float(value) for value in selected_durations if value is not None and float(value) > 0]
        top10_fit_unsupported = [fit_features.get(item + 3, (0.0, True, 0))[2] == 0 for item in flat_items]
        top10_audit_unsupported = [audit_features.get(item + 3, (0.0, True, 0))[2] == 0 for item in flat_items]
        top10_train_support = [training_counts[item] for item in flat_items]
        top10_fit_support = [fit_features.get(item + 3, (0.0, True, 0))[2] for item in flat_items]
        top10_audit_support = [audit_features.get(item + 3, (0.0, True, 0))[2] for item in flat_items]
        count = max(1, len(flat_items))
        serving_diagnostics[name] = {
            **exposure_metrics(rows, catalog_size=len(catalog_item_ids), top_k=10),
            "head_share_top10": sum(item in head_items for item in flat_items) / count,
            "repeat_rate_top10": repeat_rate(rows, recent_catalog_histories, top_k=10),
            "top1_repeat_rate": repeat_rate([[row[0]] for row in rows], recent_catalog_histories, top_k=1),
            "mean_pairwise_user_overlap_jaccard_top10": _pairwise_jaccard(rows, seed),
            "mean_duration_seconds_top10": (sum(known_selected_durations) / len(known_selected_durations)) if known_selected_durations else None,
            "duration_p10_seconds_top10": float(np.quantile(known_selected_durations, 0.10)) if known_selected_durations else None,
            "duration_median_seconds_top10": float(np.quantile(known_selected_durations, 0.50)) if known_selected_durations else None,
            "duration_p90_seconds_top10": float(np.quantile(known_selected_durations, 0.90)) if known_selected_durations else None,
            "duration_missing_fraction_top10": 1.0 - len(known_selected_durations) / count,
            "duration_short_threshold_seconds_train_p10": short_duration_threshold,
            "duration_short_fraction_top10": (
                sum(float(value) < short_duration_threshold for value in known_selected_durations) / len(known_selected_durations)
                if known_selected_durations and short_duration_threshold is not None else None
            ),
            "fit_unsupported_fraction_top10": sum(top10_fit_unsupported) / count,
            "audit_unsupported_fraction_top10": sum(top10_audit_unsupported) / count,
            "train_nonorganic_support_median_top10": float(np.median(top10_train_support)) if top10_train_support else None,
            "rm_fit_support_median_top10": float(np.median(top10_fit_support)) if top10_fit_support else None,
            "rm_audit_support_median_top10": float(np.median(top10_audit_support)) if top10_audit_support else None,
        }
    result = {
        "panel": ("A_RL, one history per user, selected by fixed seed" if panel_split == "A_RL"
                  else f"{panel_split}, one hash-selected eligible anchor per user"),
        "panel_split": panel_split,
        "policy_label": policy_label,
        "reward_head": reward_head,
        "reward_units": ("completion_ratio_minus_repeat_penalty" if repeat_penalty else
                         "completion_ratio" if reward_head == "ratio" else "predicted_listening_seconds"),
        "repeat_penalty": repeat_penalty,
        "seconds_scale": seconds_scale if reward_head == "seconds" else None,
        "seed": seed,
        "n_users": len(user_ids),
        "beta_b5": beta,
        "reward_scale": {"mean": reward_scale.mean, "std": reward_scale.std, "sample_count": reward_scale.sample_count},
        "metrics": summary,
        "serving_diagnostics": serving_diagnostics,
        "paired_user_bootstrap_vs_sft": paired,
        "per_user_diagnostics": per_user_diagnostics,
        "user_ids": user_ids,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result["elapsed_seconds"] = time.perf_counter() - panel_started
    target = Path(output_path) if output_path else root / "artifacts" / "policy_panel_seed17.json"
    if not target.is_absolute():
        target = root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    result["output_path"] = str(target.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact full-catalog reward and serving panel for SFT, policy and analytical B5")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--rm-fit-checkpoint", required=True)
    parser.add_argument("--rm-audit-checkpoint", required=True)
    parser.add_argument("--reward-scale", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--anchors", type=int, default=128, help="number of user contexts; ignored with --all-users")
    parser.add_argument("--split", choices=("a_rl", "validation", "test"), default="a_rl")
    parser.add_argument("--all-users", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output")
    parser.add_argument("--policy-label", help="Optional result label; defaults to the policy run manifest")
    parser.add_argument("--reward-head", choices=("ratio", "seconds"), default="ratio")
    parser.add_argument("--repeat-penalty", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(_evaluate_policy_panel(
        args.config,
        args.sft_checkpoint,
        args.policy_checkpoint,
        args.rm_fit_checkpoint,
        args.rm_audit_checkpoint,
        args.reward_scale,
        beta=args.beta,
        seed=args.seed,
        anchor_count=None if args.all_users else args.anchors,
        batch_size=args.batch_size,
        output_path=args.output,
        policy_label=args.policy_label,
        reward_head=args.reward_head,
        repeat_penalty=args.repeat_penalty,
        panel_split="A_RL" if args.split == "a_rl" else args.split,
    ), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
