from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tlab.config import load_config
from tlab.data.manifest import sha256_file, write_json_atomic
from tlab.evaluation.metrics import paired_user_bootstrap
from tlab.evaluation.behavior import exposure_metrics, repeat_rate
from tlab.evaluation.rm_diagnostics import _fit_item_mean
from tlab.evaluation.rewards import RewardScale
from tlab.experiments.full_model_matrix import (
    _cache_root, _collate, _load_actor, _load_cache, _load_store, _root_and_processed,
    _resolve_run_path, _shape, _model_metrics, build_reward_cache, create_panels, evaluate_actor,
    evaluate_b5,
)
from tlab.experiments.full_model_horizon import _rank_target
from tlab.train.checkpoint import source_tree_sha256
from tlab.train.reward_model import collate_reward_examples, sample_positions, sample_positions_uniform_user
from tlab.data.store import SequenceStore
from tlab.evaluation.policy_panel import _load_reward_model


def _feature_support(processed: Path) -> np.ndarray:
    import polars as pl

    frame = pl.read_parquet(processed / "rm_item_features_fit.parquet", columns=["token_id", "rm_training_support"])
    support = np.zeros(20000, dtype=np.int32)
    for token, count in frame.iter_rows():
        index = int(token) - 3
        if 0 <= index < len(support):
            support[index] = int(count or 0)
    return support


def _support_bins(support: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    groups = {
        "0": support == 0,
        "1_9": (support >= 1) & (support < 10),
        "10_99": (support >= 10) & (support < 100),
        "100_plus": support >= 100,
    }
    return groups, np.select([groups["0"], groups["1_9"], groups["10_99"]],
                             [0, 1, 2], default=3)


def _fixed_models(root: Path, artifacts: Path) -> dict[str, Path]:
    models = {"SFT strong17": root / "artifacts/reference/runs/sft_strong_seed17/best.pt",
              "reference H5 update500": root / "artifacts/reference/runs/grpo_h5_reference_strong_seed17_beta1_full/validation_checkpoints/update_00500.pt",
              "reference H5 update1000": root / "artifacts/reference/runs/grpo_h5_reference_strong_seed17_beta1_full/validation_checkpoints/update_01000.pt",
              "reference Q2 sampled": root / "artifacts/reference/runs/grpo_q2_seconds_strong_seed17_beta1_1000/last.pt"}
    selected_path = root / "artifacts/reference/h1_frozen_candidates_beta1.json"
    if selected_path.is_file():
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        for item in payload.get("models", []):
            if item.get("name") in {"strong17", "strong42", "strong73"}:
                path = Path(item["checkpoint"]["path"])
                models[f"reference H1 {item['name']}"] = path
    freeze_path = artifacts / "candidate_freeze.json"
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        recipes = freeze.get("recipe_freezes", {})
        for recipe_key, label in (
            ("sampled:ordinary:beta=1", "full_model normalized sampled ordinary beta1"),
            ("sampled:penalized:beta=1", "full_model normalized sampled penalized beta1"),
            ("exact:penalized:beta=1", "full_model normalized exact penalized beta1"),
        ):
            recipe = recipes.get(recipe_key)
            if recipe:
                models[label] = Path(recipe["checkpoint"])
        selected = freeze.get("selected_candidate", {})
        if selected.get("run_id") != "SFT":
            models["full_model normalized frozen selected candidate"] = Path(selected["checkpoint"])
    return {name: path for name, path in models.items() if path.is_file()}


def _policy_metrics(config_path: str | Path, checkpoint: Path, rows: list[dict[str, Any]],
                    reward_matrix: np.ndarray, support: np.ndarray,
                    support_mode: str = "ordinary", beta: float = 1.0, lam: float = 0.0) -> dict[str, Any]:
    import polars as pl

    config, processed, artifacts = _root_and_processed(config_path)
    store, data_protocol = _load_store(processed)
    catalog = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    item_ids = [int(value) for value in catalog.get_column("item_id").to_list()]
    item_counts = [int(value) for value in catalog.get_column("training_count").to_list()]
    durations = catalog.get_column("catalog_duration_seconds").to_list()
    cache_manifest, cache_dir, panels, arrays = _load_cache(config_path, "fit", "ratio")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, payload = _load_actor(checkpoint, _shape(config, int(data_protocol["catalog_size"])), device)
    histories = [store.users[int(row["user_index"])].history(
        int(row["position"]), int(config.dataset.get("history_length", 201))) for row in rows]
    outputs = []
    for start in range(0, len(rows), 32):
        ids, attention = _collate(histories[start:start + 32], device)
        outputs.append(actor.next_action_log_probs(ids, attention).float().detach())
    log_probs = torch.cat(outputs)
    probabilities = log_probs[:, 3:].exp().cpu().numpy()
    raw_rewards = np.asarray(reward_matrix, dtype=np.float64)
    groups, group_ids = _support_bins(support)
    contributions = {}
    masses = {}
    for name, mask in groups.items():
        contributions[name] = (probabilities[:, mask] * raw_rewards[:, mask]).sum(axis=1)
        masses[name] = probabilities[:, mask].sum(axis=1)
    total_reward = (probabilities * raw_rewards).sum(axis=1)
    contribution_sum = sum(contributions.values())
    if not np.allclose(total_reward, contribution_sum, rtol=1e-5, atol=1e-6):
        raise AssertionError("support-binned raw-reward contributions do not sum to total expected reward")
    ranks = [_rank_target(log_probs[row_index], int(row["target_token_id"]), item_ids)
             for row_index, row in enumerate(rows)]
    per_user = {}
    top_indices = []
    top_raw_items = []
    recent_histories = []
    for idx, row in enumerate(rows):
        uid = int(row["uid"])
        per_user[uid] = {"ndcg@10": 0.0 if ranks[idx] is None else (1.0 / math.log2(ranks[idx] + 1) if ranks[idx] <= 10 else 0.0),
                         "recall@10": float(ranks[idx] is not None and ranks[idx] <= 10),
                         "expected_raw_reward": float(total_reward[idx]),
                         "unsupported_mass": float(masses["0"][idx])}
        order = np.lexsort((np.asarray(item_ids), -probabilities[idx]))[:10]
        top_indices.append([int(value) for value in order])
        top_raw_items.append([int(item_ids[value]) for value in order])
        user = store.users[int(row["user_index"])]
        recent_histories.append({int(value) for value in user.raw_item_ids[
            max(0, int(row["position"]) - 20):int(row["position"])] if int(value) >= 0})
    masked_ndcg = []
    masked_recall = []
    masked_top_tokens: list[list[int]] = []
    supported = support > 0
    for idx, row in enumerate(rows):
        masked = probabilities[idx].copy()
        masked[~supported] = 0.0
        denominator = masked.sum()
        if denominator > 0:
            masked /= denominator
        scores = torch.full((len(item_ids) + 3,), -torch.inf, dtype=torch.float32)
        scores[3:] = torch.from_numpy(masked.astype(np.float32)).log()
        rank = _rank_target(scores, int(row["target_token_id"]), item_ids)
        masked_ndcg.append(0.0 if rank is None or rank > 10 else 1.0 / math.log2(rank + 1))
        masked_recall.append(float(rank is not None and rank <= 10))
        top = np.argsort(-masked, kind="stable")[:10]
        masked_top_tokens.append([int(value + 3) for value in top])
    all_top = [int(x) for values in masked_top_tokens for x in values]
    masked_coverage = len(set(all_top)) / len(item_ids) if item_ids else 0.0
    head_count = max(1, int(math.ceil(0.01 * len(item_counts))))
    head_indices = set(sorted(range(len(item_counts)), key=lambda i: (-item_counts[i], item_ids[i]))[:head_count])
    head_share = sum(index in head_indices for row in top_indices for index in row) / max(1, len(top_indices) * 10)
    duration_values = [float(durations[index]) for row in top_indices for index in row
                       if durations[index] is not None and math.isfinite(float(durations[index]))]
    jaccard = []
    if len(top_indices) > 1:
        for left in range(len(top_indices)):
            for right in range(left + 1, min(len(top_indices), left + 11)):
                a, b = set(top_indices[left]), set(top_indices[right])
                jaccard.append(len(a & b) / max(1, len(a | b)))
    target_coverage = sum(int(row["target_token_id"]) >= 3 for row in rows) / max(1, len(rows))
    return {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_update": payload.get("metadata", {}).get("update", payload.get("step")),
        "n_users": len(rows), "expected_raw_reward": float(np.mean(total_reward)),
        "target_catalog_coverage": float(target_coverage),
        "ndcg@10": float(np.mean([row["ndcg@10"] for row in per_user.values()])),
        "recall@10": float(np.mean([row["recall@10"] for row in per_user.values()])),
        "unsupported_mass": float(np.mean(masses["0"])),
        "catalog_exposure": exposure_metrics(top_indices, len(item_ids), top_k=10),
        "head_share_top10": float(head_share),
        "history_repeat_top10": repeat_rate(top_raw_items, recent_histories, top_k=10),
        "duration_seconds_top10_mean": float(np.mean(duration_values)) if duration_values else None,
        "pairwise_user_jaccard_top10_sampled_pairs": float(np.mean(jaccard)) if jaccard else None,
        "support_decomposition": {
            name: {"policy_mass": float(np.mean(masses[name])),
                  "expected_raw_reward_contribution": float(np.mean(contributions[name])),
                  "reward_contribution_share": float(np.mean(contributions[name]) / max(np.mean(total_reward), 1e-12)),
                  "catalog_items": int(mask.sum())}
            for name, mask in groups.items()},
        "support_contribution_sum_max_abs_error": float(np.max(np.abs(total_reward - contribution_sum))),
        "hard_supported_mask_control": {"ndcg@10": float(np.mean(masked_ndcg)),
                                        "recall@10": float(np.mean(masked_recall)),
                                        "coverage_at_10": float(masked_coverage),
                                        "targets_remain_in_full_denominator": True},
        "per_user_metrics": per_user,
        "panel_status": "known validation; development and confirmation combined; no Test or RM_audit selection",
    }


def run_e1_support_analysis(config_path: str | Path = "configs/full_model_rl.yaml") -> dict[str, Any]:
    config, processed, artifacts = _root_and_processed(config_path)
    panels = create_panels(config_path)
    cache_manifest, cache_dir, _, arrays = _load_cache(config_path, "fit", "ratio")
    rows = panels["development"] + panels["confirmation"]
    eval_matrix = arrays["evaluation"][:len(rows)]
    support = _feature_support(processed)
    models = _fixed_models(config.path.parent.parent, artifacts)
    results = {name: _policy_metrics(config_path, path, rows, eval_matrix, support)
               for name, path in models.items()}
    baseline = results.get("SFT strong17")
    if baseline is not None:
        for name, result in results.items():
            if name == "SFT strong17":
                continue
            common = set(baseline["per_user_metrics"]) & set(result["per_user_metrics"])
            result["paired_to_sft"] = {
                metric: paired_user_bootstrap({uid: result["per_user_metrics"][uid][metric]
                                               - baseline["per_user_metrics"][uid][metric]
                                               for uid in common}, repetitions=2000, seed=27101 + index)
                for index, metric in enumerate(("ndcg@10", "recall@10", "expected_raw_reward", "unsupported_mass"))}
    payload = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "status": "completed",
               "objective_version": "normalized_reward_v1",
               "panel_status": "disclosed_panel_reanalysis", "independent_confirmation": False,
               "panel_ids_sha256": panels["panel_ids_sha256"],
               "cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json"),
               "support_definition": "zero fit-RM training-event support; diagnostic groups 0, 1-9, 10-99, 100+",
               "models": {name: {key: value for key, value in result.items()
                                  if key != "per_user_metrics"} for name, result in results.items()},
               "audit_policy": "RM_audit excluded from candidate freeze and primary support decomposition",
               "test_policy": "historical Test already disclosed; not used or rerun",
               "model_rows": {name: value["per_user_metrics"] for name, value in results.items()}}
    target = artifacts / "e1" / "support_decomposition.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, payload)
    return payload


@torch.no_grad()
def run_rm_support_errors(config_path: str | Path = "configs/full_model_rl.yaml", *,
                          validation_limit: int = 40000, seed: int = 17031) -> dict[str, Any]:
    import polars as pl

    config, processed, artifacts = _root_and_processed(config_path)
    data_protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    store, _ = _load_store(processed)
    common_validation = sample_positions_uniform_user(store.users, "rm_validation_positions",
                                                      validation_limit, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seconds_cap = float(data_protocol["seconds_cap_f_rm_p99_5"])
    seconds_scale = float(data_protocol["seconds_scale_f_rm_p95"])
    variant_results = {}
    for variant in ("fit", "audit"):
        users = [user for user in store.users if user.rm_user_partition == variant]
        train_store = SequenceStore(users, store.events, store.tie_blocks, store.tied_events)
        train_limit = int(config.training.get("rm_max_targets_per_epoch", 200000))
        train_examples = sample_positions(users, "rm_fit_positions", train_limit, seed + 101)
        global_mean, item_mean = _fit_item_mean(train_store, train_examples)
        feature_frame = pl.read_parquet(processed / f"rm_item_features_{variant}.parquet")
        features = {int(row["token_id"]): (float(row["rm_duration_seconds"] or 0.0),
                                             bool(row["rm_duration_missing"]),
                                             int(row["rm_training_support"] or 0))
                    for row in feature_frame.iter_rows(named=True)}
        checkpoint_path = artifacts / "runs" / f"rm_reference_{variant}_seed17" / "best.pt"
        # full_model outputs live in artifacts/full_model, while the fixed main RM is a frozen reference model.
        checkpoint_path = config.path.parent.parent / "artifacts" / "reference" / "runs" / f"rm_reference_{variant}_seed17" / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model, _ = _load_reward_model(config, checkpoint_path, variant, device)
        buckets: dict[str, dict[int, dict[str, list[float]]]] = {
            name: defaultdict(lambda: {"model_squared_error": [], "item_mean_squared_error": []})
            for name in ("0", "1_9", "10_99", "100_plus")}
        counts = defaultdict(int)
        valid_total = 0
        for start in range(0, len(common_validation), 64):
            examples = common_validation[start:start + 64]
            batch = collate_reward_examples(store, examples, features, seconds_cap, seconds_scale,
                                            int(config.dataset.get("history_length", 201)), device)
            prediction = model(*batch[:6])["ratio"].float().cpu().tolist()
            targets = batch[6].float().cpu().tolist()
            valid = batch[7].cpu().tolist()
            tokens = batch[2].cpu().tolist()
            for (user_index, position), token, target, estimate, is_valid in zip(
                    examples, tokens, targets, prediction, valid):
                if not is_valid:
                    continue
                support_count = int(features.get(int(token), (0.0, True, 0))[2])
                name = "0" if support_count == 0 else "1_9" if support_count < 10 else "10_99" if support_count < 100 else "100_plus"
                user = store.users[user_index]
                baseline = item_mean.get(int(token), global_mean)
                bucket = buckets[name][int(user.uid)]
                bucket["model_squared_error"].append((float(estimate) - float(target)) ** 2)
                bucket["item_mean_squared_error"].append((float(baseline) - float(target)) ** 2)
                counts[name] += 1
                valid_total += 1
        group_metrics = {}
        for group, users_by_group in buckets.items():
            model_user_mse = {uid: float(np.mean(values["model_squared_error"]))
                              for uid, values in users_by_group.items()}
            baseline_user_mse = {uid: float(np.mean(values["item_mean_squared_error"]))
                                 for uid, values in users_by_group.items()}
            paired = set(model_user_mse) & set(baseline_user_mse)
            model_errors = [value for values in users_by_group.values() for value in values["model_squared_error"]]
            baseline_errors = [value for values in users_by_group.values() for value in values["item_mean_squared_error"]]
            group_metrics[group] = {
                "n_targets": counts[group], "n_users": len(paired),
                "target_share": counts[group] / max(valid_total, 1),
                "model_rmse_event_weighted": math.sqrt(float(np.mean(model_errors))) if model_errors else None,
                "item_mean_rmse_event_weighted": math.sqrt(float(np.mean(baseline_errors))) if baseline_errors else None,
                "paired_macro_user_mse_difference_model_minus_item_mean": (
                    paired_user_bootstrap({uid: model_user_mse[uid] - baseline_user_mse[uid] for uid in paired},
                                          repetitions=2000, seed=seed + (0 if variant == "fit" else 1) + counts[group])
                    if paired else None),
            }
        variant_results[variant] = {
            "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_partition": variant, "training_examples": len(train_examples),
            "validation_sample_seed": seed, "validation_examples_requested": validation_limit,
            "validation_examples_valid": valid_total, "support_groups": group_metrics,
            "item_mean_baseline": "fit on same variant's F_RM user partition and sampled labels",
            "independent_confirmation": False if variant == "audit" else None,
        }
    result = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "status": "completed",
              "validation_period": "known main V_RM; existing disclosed RM checkpoints",
              "rm_audit_previously_revealed": True, "rm_audit_not_independent_confirmation": True,
              "results": variant_results}
    write_json_atomic(artifacts / "e1" / "rm_support_errors.json", result)
    return result


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    result = {}
    previous = 0.0
    for idx, key in enumerate(ordered):
        value = min(1.0, p_values[key] * (len(ordered) - idx))
        previous = max(previous, value)
        result[key] = previous
    return result


def analyze_frontier(config_path: str | Path = "configs/full_model_rl.yaml") -> dict[str, Any]:
    config, _, artifacts = _root_and_processed(config_path)
    rows = []
    runs_root = artifacts / "runs"
    for manifest_path in sorted(runs_root.glob("q1_*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"completed", "failed"}:
            continue
        metrics_path = manifest_path.parent / "metrics.jsonl"
        train_rows = {int(item["update"]): item for item in (
            json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip())}
        validation_path = manifest_path.parent / "validation.jsonl"
        validation = {int(item["update"]): item["development"] for item in (
            json.loads(line) for line in validation_path.read_text(encoding="utf-8").splitlines() if line.strip())} if validation_path.exists() else {}
        for update in (100, 500, 1000, 2500, 5000):
            if update not in validation:
                continue
            rows.append({"run_id": manifest["run_id"], "method": manifest.get("method"),
                         "support_mode": manifest.get("support_mode"), "beta": manifest.get("beta"),
                         "seed": manifest.get("seed"), "context_source": manifest.get("context_source", "cached"),
                         "update": update, "training_seconds_at_update": train_rows.get(update, {}).get("elapsed_training_seconds"),
                         "training_J_minibatch_post": train_rows.get(update, {}).get(
                             "J_train_minibatch_post", train_rows.get(update, {}).get("J_train_post")),
                         "training_objective_version": manifest.get("objective_version", "legacy_raw_reward_sampled"),
                         "development": validation[update],
                         "selection_allowed": (manifest.get("stage") == "core" and manifest.get("seed") == 17
                                               and manifest.get("context_source", "cached") == "cached"
                                               and update in {100, 500, 1000})})
    reused_eval_path = artifacts / "reused_exact_development.json"
    if reused_eval_path.is_file():
        reused = json.loads(reused_eval_path.read_text(encoding="utf-8"))
        for candidate in reused.get("candidates", []):
            rows.append({"run_id": candidate["run_id"], "method": "exact",
                         "support_mode": candidate["support_mode"], "beta": candidate["beta"],
                         "seed": candidate["seed"], "context_source": "cached",
                         "update": candidate["update"], "training_seconds_at_update": None,
                         "training_J_minibatch_post": None,
                         "training_objective_version": "normalized_reward_v1_exact_reused",
                         "development": candidate["metrics"],
                         "selection_allowed": int(candidate["update"]) in {100, 500, 1000},
                         "reused": True, "checkpoint_sha256": candidate["checkpoint_sha256"]})
    b5 = []
    cache_dir = _cache_root(config, artifacts) / "main" / "fit_ratio"
    for path in sorted(cache_dir.glob("b5_beta_*_development.json")):
        b5.append(json.loads(path.read_text(encoding="utf-8")))
    b5_path = artifacts / "e3" / "b5_beta_1_development.json"
    if b5_path.is_file():
        b5.append(json.loads(b5_path.read_text(encoding="utf-8")))
    payload = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "status": "completed",
               "objective_version": "normalized_reward_v1",
               "development_panel_only": True, "known_validation_not_independent": True,
               "primary_optimizer_endpoint": 1000, "candidate_checkpoints": [100, 500, 1000],
               "stress_points": [2500, 5000], "runs": rows, "analytic_b5": b5,
               "candidate_freeze": json.loads((artifacts / "candidate_freeze.json").read_text(encoding="utf-8"))
               if (artifacts / "candidate_freeze.json").exists() else None}
    (artifacts / "e3").mkdir(parents=True, exist_ok=True)
    write_json_atomic(artifacts / "e3" / "frontier.json", payload)
    return payload


@torch.no_grad()
def full_cache_learning_curves(config_path: str | Path) -> dict[str, Any]:
    """Forward-only rescoring on the complete fixed train cache and both disclosed eval panels."""
    import polars as pl
    import time

    started = time.perf_counter()
    config, processed, artifacts = _root_and_processed(config_path)
    cache_manifest, cache_dir, panels, arrays = _load_cache(config_path, "fit", "ratio")
    store, protocol = _load_store(processed)
    catalog = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    item_ids = [int(value) for value in catalog.get_column("item_id").to_list()]
    counts = [int(value) for value in catalog.get_column("training_count").to_list()]
    durations = catalog.get_column("catalog_duration_seconds").to_list()
    support = np.asarray(arrays["unsupported_mask"], dtype=np.bool_)
    scale = RewardScale(**cache_manifest["reward_scale"])
    selected_lambda = float(json.loads((artifacts / "lambda_freeze.json").read_text(encoding="utf-8"))["selected_lambda"])
    target_b5 = artifacts / "e3" / "b5_beta_1_development.json"
    target_b5.parent.mkdir(parents=True, exist_ok=True)
    b5 = evaluate_b5(config_path, config.path.parent.parent / "artifacts/reference/runs/sft_strong_seed17/best.pt",
                     beta=1.0, reward_head="ratio", output_path=target_b5)
    reference_path = config.path.parent.parent / "artifacts/reference/runs/sft_strong_seed17/best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference, _ = _load_actor(reference_path, _shape(config, int(protocol["catalog_size"])), device)
    panel_sets = {
        "train": (panels["train"], arrays["train"], panels["panel_ids_sha256"]["train"]),
        "development": (panels["development"], arrays["evaluation"][:len(panels["development"])],
                        panels["panel_ids_sha256"]["development"]),
        "confirmation": (panels["confirmation"], arrays["evaluation"][len(panels["development"]):],
                         panels["panel_ids_sha256"]["confirmation"]),
    }

    def score_checkpoint(checkpoint: Path, beta: float, support_mode: str,
                         applied_lambda: float) -> dict[str, dict[str, Any]]:
        actor, payload = _load_actor(checkpoint, _shape(config, int(protocol["catalog_size"])), device)
        result = {}
        for panel_name, (rows, reward_matrix, panel_hash) in panel_sets.items():
            histories = [store.users[int(row["user_index"])].history(
                int(row["position"]), int(config.dataset.get("history_length", 201))) for row in rows]
            policy_parts, reference_parts = [], []
            for start in range(0, len(histories), 16):
                ids, attention = _collate(histories[start:start + 16], device)
                policy_parts.append(actor.next_action_log_probs(ids, attention))
                reference_parts.append(reference.next_action_log_probs(ids, attention))
            log_policy = torch.cat(policy_parts)
            log_reference = torch.cat(reference_parts)
            metrics = _model_metrics(
                log_policy, rows, store, item_ids, durations, counts, support,
                reward_matrix, scale, beta, log_reference, support_mode, applied_lambda,
                support_lambda_reference=selected_lambda,
            )
            metrics.pop("per_user_metrics", None)
            metrics.pop("per_user_expected_raw_reward", None)
            result[panel_name] = {"panel_sha256": panel_hash, "n_states": len(rows), **metrics}
        result["checkpoint_sha256"] = sha256_file(checkpoint)
        result["checkpoint_update"] = payload.get("metadata", {}).get("update", payload.get("step"))
        return result

    sft_path = reference_path
    baseline = score_checkpoint(sft_path, 1.0, "ordinary", 0.0)
    baseline_panels = {name: metrics for name, metrics in baseline.items() if name in panel_sets}
    trajectories = [{"run_id": "SFT strong17", "method": "SFT", "support_mode": "ordinary",
                    "beta": 1.0, "seed": 17, "update": 0, "context_source": "fixed SFT reference",
                    "reused": True, "objective_version": "normalized_reward_v1",
                    "checkpoint": str(sft_path.resolve()), "checkpoint_sha256": sha256_file(sft_path),
                    "panels": baseline_panels}]
    recipes = [
        ("sampled", "ordinary", 0.0, False),
        ("exact", "ordinary", 0.0, True),
        ("sampled", "penalized", selected_lambda, False),
        ("exact", "penalized", selected_lambda, True),
    ]
    for method, support_mode, lam, is_reused in recipes:
        run_id = f"q1_{method}_{support_mode}_b1_s17"
        run_path, resolved_reuse = _resolve_run_path(artifacts, run_id)
        manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or int(manifest.get("latest_update", 0)) < 5000:
            raise ValueError(f"full-cache learning curve requires a completed 5000-update trajectory: {run_id}")
        for update in (0, 100, 500, 1000, 2500, 5000):
            checkpoint = run_path / "snapshots" / f"update_{update:05d}.pt"
            if update == 0 and not checkpoint.is_file():
                # Both exact and sampled policies start from the same immutable SFT checkpoint.
                checkpoint = sft_path
            if not checkpoint.is_file():
                raise FileNotFoundError(f"full-cache curve checkpoint missing: {checkpoint}")
            panel_metrics = score_checkpoint(checkpoint, 1.0, support_mode, lam)
            panels_out = {name: value for name, value in panel_metrics.items() if name in panel_sets}
            for panel_name, values in panels_out.items():
                base = baseline_panels[panel_name]
                values["uplift_from_sft"] = {
                    metric: float(values[metric] - base[metric])
                    for metric in ("J_base", "J_support", "expected_normalized_reward", "expected_raw_reward", "ndcg@10")
                }
            dev = panels_out["development"]
            b5_metric = dev.get("J_base")
            trajectories.append({
                "run_id": run_id, "method": method, "support_mode": support_mode,
                "beta": 1.0, "lambda_support": lam if support_mode == "penalized" else 0.0,
                "seed": 17, "update": update, "context_source": "fixed 2048-context cache",
                "reused": bool(is_reused or resolved_reuse),
                "objective_version": "normalized_reward_v1",
                "training_seconds_at_update": next((float(row.get("elapsed_training_seconds"))
                                                     for row in (json.loads(line) for line in (run_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines())
                                                     if int(row.get("update", -1)) == update), None),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint), "panels": panels_out,
                "development_J_base_gap_to_B5": float(b5["J_base"] - b5_metric),
                "B5_checkpoint_sha256": sha256_file(sft_path),
            })
    for recipe_key, support_mode, method in [
        ("sampled:ordinary:beta=1", "ordinary", "sampled"),
        ("sampled:penalized:beta=1", "penalized", "sampled"),
        ("exact:ordinary:beta=1", "ordinary", "exact"),
        ("exact:penalized:beta=1", "penalized", "exact"),
    ]:
        selected_steps = {int(row["update"]) for row in trajectories
                          if row["method"] == method and row["support_mode"] == support_mode}
        for record in trajectories:
            if record["method"] != method or record["support_mode"] != support_mode or record["update"] == 0:
                continue
            train_uplift = record["panels"]["train"]["uplift_from_sft"]
            for eval_panel in ("development", "confirmation"):
                eval_uplift = record["panels"][eval_panel]["uplift_from_sft"]
                record["panels"][eval_panel]["train_minus_eval_uplift_gap"] = {
                    metric: float(train_uplift[metric] - eval_uplift[metric])
                    for metric in ("J_base", "J_support", "expected_normalized_reward", "expected_raw_reward", "ndcg@10")
                }
            record["selection_allowed"] = record["update"] in {100, 500, 1000}
            record["late_point_stress_only"] = record["update"] in {2500, 5000}

    result = {
        "status": "completed_forward_only_rescoring", "objective_version": "normalized_reward_v1",
        "panel_status": "disclosed_panel_reanalysis", "independent_confirmation": False,
        "cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json"),
        "cache_files": {key: {"sha256": details["sha256"], "shape": details["shape"], "bytes": details["bytes"]}
                        for key, details in cache_manifest["matrices"].items()},
        "panel_ids_sha256": panels["panel_ids_sha256"],
        "reward_scale": cache_manifest["reward_scale"],
        "support_lambda": selected_lambda,
        "baselines": baseline_panels,
        "analytic_b5_development": {key: b5.get(key) for key in ("J_base", "J_support", "expected_raw_reward", "ndcg@10", "checkpoint_sha256")},
        "trajectories": trajectories,
        "cost": {"forward_only_rescore_seconds": time.perf_counter() - started,
                 "measurement_scope": "SFT and 4 optimizer/support trajectories × 6 snapshots × full train/development/confirmation caches plus B5"},
    }
    write_json_atomic(artifacts / "full_cache_learning_curves.json", result)
    return result


def analyze_stream_bridge(config_path: str | Path = "configs/full_model_rl.yaml") -> dict[str, Any]:
    config, _, artifacts = _root_and_processed(config_path)
    panels = create_panels(config_path)
    cache_manifest, cache_dir, _, arrays = _load_cache(config_path, "fit", "ratio")
    all_rows = panels["development"] + panels["confirmation"]
    eval_matrix = arrays["evaluation"]
    dev_n = len(panels["development"])
    selections = {"development": (panels["development"], eval_matrix[:dev_n]),
                  "confirmation": (panels["confirmation"], eval_matrix[dev_n:dev_n + len(panels["confirmation"])])}
    results = []
    for seed in (17, 42, 73):
        cached_path = artifacts / "runs" / f"q1_sampled_ordinary_b1_s{seed}"
        stream_path = artifacts / "runs" / f"q1_sampled_ordinary_b1_s{seed}_stream"
        cached_manifest = json.loads((cached_path / "run_manifest.json").read_text(encoding="utf-8"))
        stream_manifest = json.loads((stream_path / "run_manifest.json").read_text(encoding="utf-8"))
        stream_cache_manifest_path = cache_dir / "stream" / f"seed{seed}" / "stream_manifest.json"
        stream_cache_manifest = json.loads(stream_cache_manifest_path.read_text(encoding="utf-8"))
        for panel_name, (panel_rows, panel_rewards) in selections.items():
            cache_eval = evaluate_actor(config_path, cached_path / "snapshots/update_01000.pt", panel_rows,
                                        panel_rewards, cache_manifest, arrays, 1.0, "ordinary", 0.0)
            stream_eval = evaluate_actor(config_path, stream_path / "snapshots/update_01000.pt", panel_rows,
                                         panel_rewards, cache_manifest, arrays, 1.0, "ordinary", 0.0)
            common = set(cache_eval["per_user_metrics"]) & set(stream_eval["per_user_metrics"])
            deltas = {}
            for metric in ("ndcg@10", "expected_raw_reward"):
                if metric == "expected_raw_reward":
                    stream_values = stream_eval["per_user_expected_raw_reward"]
                    cache_values = cache_eval["per_user_expected_raw_reward"]
                else:
                    stream_values = {uid: values[metric] for uid, values in stream_eval["per_user_metrics"].items()}
                    cache_values = {uid: values[metric] for uid, values in cache_eval["per_user_metrics"].items()}
                deltas[metric] = paired_user_bootstrap({uid: stream_values[uid] - cache_values[uid]
                                                        for uid in common}, repetitions=2000, seed=38100 + seed)
            results.append({"seed": seed, "panel": panel_name,
                            "cached_run_training_seconds": cached_manifest.get("training_seconds"),
                            "stream_run_training_seconds": stream_manifest.get("training_seconds"),
                            "stream_unique_histories": stream_cache_manifest.get("unique_histories"),
                            "stream_scoring_seconds": stream_cache_manifest.get("score_elapsed_seconds"),
                            "stream_cache_bytes": stream_cache_manifest.get("cache_bytes"),
                            "paired_deltas_stream_minus_cached": deltas,
                            "cached": {key: value for key, value in cache_eval.items()
                                       if key not in {"per_user_metrics", "per_user_expected_raw_reward"}},
                            "stream": {key: value for key, value in stream_eval.items()
                                       if key not in {"per_user_metrics", "per_user_expected_raw_reward"}}})
    result = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "status": "completed",
              "objective_version": "normalized_reward_v1",
              "panel_status": "disclosed_panel_reanalysis", "independent_confirmation": False,
              "comparison": "new A_RL anchors, same users/schedule, evaluation panels and reward scale as cached runs",
              "selection_allowed": False, "results": results}
    write_json_atomic(artifacts / "e3" / "cached_stream_bridge.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="full_model E1/E3 fixed analysis")
    parser.add_argument("--config", default="configs/full_model_rl.yaml")
    parser.add_argument("--mode", choices=("e1", "rm-support", "frontier", "bridge", "full-cache-curves"), required=True)
    args = parser.parse_args()
    if args.mode == "e1":
        result = run_e1_support_analysis(args.config)
    elif args.mode == "rm-support":
        result = run_rm_support_errors(args.config)
    elif args.mode == "frontier":
        result = analyze_frontier(args.config)
    elif args.mode == "full-cache-curves":
        result = full_cache_learning_curves(args.config)
    else:
        result = analyze_stream_bridge(args.config)
    print(json.dumps({key: value for key, value in result.items() if key not in {"model_rows"}},
                     indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
