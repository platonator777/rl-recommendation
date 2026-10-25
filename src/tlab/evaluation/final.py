from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tlab.data.manifest import sha256_file, write_json_atomic
from tlab.data.store import SequenceStore
from tlab.evaluation.baselines import _train_transition_matrix, rank_score_vector
from tlab.evaluation.behavior import duration_quantile_from_train, exposure_metrics, repeat_rate
from tlab.evaluation.metrics import aggregate_macro_user, paired_user_bootstrap, single_target_metrics
from tlab.evaluation.policy_panel import _evaluate_policy_panel, _pairwise_jaccard
from tlab.evaluation.ranking import evaluate_full_catalog
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.train.checkpoint import source_tree_sha256


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _user_metric_values(per_user_ranks: dict[str, list[int | None]]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for user_id, ranks in per_user_ranks.items():
        rows = [single_target_metrics(rank) for rank in ranks]
        if not rows:
            continue
        result[int(user_id)] = {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in rows[0]
        }
    return result


def _paired_ndcg(
    candidate: dict[str, list[int | None]],
    baseline: dict[str, list[int | None]],
    seed: int,
) -> dict[str, float]:
    candidate_values = _user_metric_values(candidate)
    baseline_values = _user_metric_values(baseline)
    common = sorted(candidate_values.keys() & baseline_values.keys())
    differences = {
        user_id: candidate_values[user_id]["ndcg@10"] - baseline_values[user_id]["ndcg@10"]
        for user_id in common
    }
    return paired_user_bootstrap(differences, repetitions=2000, seed=seed)


def _training_item_ids(store: SequenceStore) -> set[int]:
    found: set[int] = set()
    for user in store.users:
        item_ids = np.frombuffer(user.raw_item_ids, dtype=np.int64)
        split_codes = np.frombuffer(user.split_codes, dtype=np.uint8)
        train_ids = item_ids[(split_codes == 0) & (item_ids >= 0)]
        found.update(int(item_id) for item_id in np.unique(train_ids))
    return found


def _ranking_summary(
    per_user_ranks: dict[str, list[int | None]],
    target_metadata_by_user: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows: list[tuple[int, int | None]] = []
    in_catalog_rows: list[tuple[int, int | None]] = []
    positive_rows: list[tuple[int, int | None]] = []
    positive_in_catalog_rows: list[tuple[int, int | None]] = []
    event_metrics: list[dict[str, float]] = []
    event_metrics_in_catalog: list[dict[str, float]] = []
    event_metrics_positive: list[dict[str, float]] = []
    event_metrics_positive_in_catalog: list[dict[str, float]] = []
    in_catalog_count = known_ooc_count = cold_ooc_count = positive_count = 0
    total_targets = 0
    for user_id, ranks in per_user_ranks.items():
        metadata = target_metadata_by_user.get(user_id, [])
        if len(ranks) != len(metadata):
            raise ValueError(f"test target and rank counts differ for user {user_id}")
        for rank, target in zip(ranks, metadata):
            uid = int(user_id)
            rows.append((uid, rank))
            metrics = single_target_metrics(rank)
            event_metrics.append(metrics)
            total_targets += 1
            if target["in_catalog"]:
                in_catalog_count += 1
                in_catalog_rows.append((uid, rank))
                event_metrics_in_catalog.append(metrics)
            elif target["known_in_train"]:
                known_ooc_count += 1
            else:
                cold_ooc_count += 1
            if target["positive"]:
                positive_count += 1
                positive_rows.append((uid, rank))
                event_metrics_positive.append(metrics)
                if target["in_catalog"]:
                    positive_in_catalog_rows.append((uid, rank))
                    event_metrics_positive_in_catalog.append(metrics)

    def micro(values: list[dict[str, float]]) -> dict[str, float] | None:
        if not values:
            return None
        return {key: sum(row[key] for row in values) / len(values) for key in values[0]}

    return {
        "n_targets": total_targets,
        "n_users": len(per_user_ranks),
        "n_positive_targets": positive_count,
        "n_users_with_positive_targets": len({uid for uid, _ in positive_rows}),
        "target_coverage": in_catalog_count / total_targets if total_targets else None,
        "known_train_outside_catalog": known_ooc_count,
        "cold_outside_catalog": cold_ooc_count,
        "macro_user_unconditional": aggregate_macro_user(rows) if rows else None,
        "macro_user_in_catalog": aggregate_macro_user(in_catalog_rows) if in_catalog_rows else None,
        "macro_user_positive": aggregate_macro_user(positive_rows) if positive_rows else None,
        "macro_user_positive_in_catalog": aggregate_macro_user(positive_in_catalog_rows) if positive_in_catalog_rows else None,
        "micro_event_unconditional": micro(event_metrics),
        "micro_event_in_catalog": micro(event_metrics_in_catalog),
        "micro_event_positive": micro(event_metrics_positive),
        "micro_event_positive_in_catalog": micro(event_metrics_positive_in_catalog),
    }


def _baseline_rankings(
    store: SequenceStore,
    catalog_item_ids: list[int],
    training_counts: np.ndarray,
    split: str,
    per_user_limit: int,
    seed: int,
    top_k: int,
) -> dict[str, dict[str, Any]]:
    raw_to_index = {int(item_id): index for index, item_id in enumerate(catalog_item_ids)}
    anchors = store.evaluation_targets(split, per_user_limit=per_user_limit, seed=seed)
    if not anchors:
        raise ValueError(f"no recommendable anchors for final {split} baseline evaluation")
    transition_matrix, transition_examples = _train_transition_matrix(store, len(catalog_item_ids))
    rows: dict[str, list[tuple[int, int | None]]] = {"popularity": [], "sequential_bigram": []}
    recs: dict[str, dict[int, list[list[int]]]] = {name: defaultdict(list) for name in rows}
    for user_index, position in anchors:
        user = store.users[user_index]
        uid = int(user.uid)
        raw_target = int(user.raw_item_ids[position])
        target_index = raw_to_index.get(raw_target)
        rows["popularity"].append((uid, rank_score_vector(training_counts, target_index, catalog_item_ids)))
        popularity_order = np.lexsort((catalog_item_ids, -training_counts))[:top_k]
        recs["popularity"][uid].append([catalog_item_ids[i] for i in popularity_order])

        previous_token = int(user.tokens[position - 1]) if position > 0 else 1
        transition = transition_matrix.getrow(previous_token)
        scores = np.zeros(len(catalog_item_ids), dtype=np.int32)
        if transition.nnz:
            scores[transition.indices] = transition.data
        else:
            scores = training_counts
        rows["sequential_bigram"].append((uid, rank_score_vector(scores, target_index, catalog_item_ids)))
        order = np.lexsort((catalog_item_ids, -scores))[:top_k]
        recs["sequential_bigram"][uid].append([catalog_item_ids[i] for i in order])

    result: dict[str, dict[str, Any]] = {}
    for name in rows:
        by_user: dict[str, list[int | None]] = defaultdict(list)
        for uid, rank in rows[name]:
            by_user[str(uid)].append(rank)
        result[name] = {
            "macro_user_unconditional": aggregate_macro_user(rows[name]),
            "target_coverage": sum(rank is not None for _, rank in rows[name]) / len(rows[name]),
            "n_users": len(by_user),
            "n_targets": len(rows[name]),
            "per_user_ranks": dict(by_user),
            "per_user_top_item_ids": {str(uid): recommendations for uid, recommendations in recs[name].items()},
        }
    result["sequential_bigram"]["training_transition_examples"] = transition_examples
    return result


def _feature_support(processed_dir: Path, variant: str) -> dict[int, int]:
    import polars as pl

    path = processed_dir / f"rm_item_features_{variant}.parquet"
    if not path.is_file():
        return {}
    frame = pl.read_parquet(path, columns=["token_id", "rm_training_support"])
    return {int(token) - 3: int(support) for token, support in frame.iter_rows()}


def _serving_diagnostics(
    per_user_top_item_ids: dict[str, list[list[int]]],
    store: SequenceStore,
    catalog_item_ids: list[int],
    training_counts: np.ndarray,
    catalog_durations: list[float | None],
    split: str,
    per_user_limit: int,
    seed: int,
    fit_support: dict[int, int],
    audit_support: dict[int, int],
) -> dict[str, Any]:
    raw_to_index = {int(item_id): index for index, item_id in enumerate(catalog_item_ids)}
    anchors = store.evaluation_targets(split, per_user_limit=per_user_limit, seed=seed)
    recent_by_user: dict[str, list[set[int]]] = defaultdict(list)
    for user_index, position in anchors:
        user = store.users[user_index]
        recent_tokens = user.tokens[max(0, position - 20):position]
        recent = {int(token) - 3 for token in recent_tokens if 3 <= int(token) < 3 + len(catalog_item_ids)}
        recent_by_user[str(user.uid)].append(recent)

    rows: list[list[int]] = []
    recent_rows: list[set[int]] = []
    raw_rows: list[list[int]] = []
    # Serving diversity is user-level, so use one deterministic test context per user.
    for user_id in sorted(per_user_top_item_ids, key=int):
        user_recs = per_user_top_item_ids[user_id]
        user_recent = recent_by_user.get(user_id, [])
        if not user_recs or not user_recent:
            continue
        rec = user_recs[0]
        raw_rows.append(rec)
        rows.append([raw_to_index[item_id] for item_id in rec if item_id in raw_to_index])
        recent_rows.append(user_recent[0])
    if not rows:
        raise ValueError("final serving diagnostics require saved top-item predictions")

    duration_threshold = duration_quantile_from_train(store, 0.10)
    head_count = max(1, math.ceil(0.01 * len(catalog_item_ids)))
    head_items = set(sorted(range(len(training_counts)), key=lambda i: (-training_counts[i], catalog_item_ids[i]))[:head_count])
    flat_indices = [index for row in rows for index in row]
    flat_raw = [item_id for row in raw_rows for item_id in row]
    durations = [catalog_durations[index] for index in flat_indices]
    train_support_values = np.asarray([training_counts[index] for index in flat_indices], dtype=np.float64)
    fit_support_values = np.asarray([fit_support.get(index, 0) for index in flat_indices], dtype=np.float64)
    audit_support_values = np.asarray([audit_support.get(index, 0) for index in flat_indices], dtype=np.float64)
    known_selected = [float(value) for value in durations if value is not None and float(value) > 0]
    n_recommendations = max(1, len(flat_indices))

    def support_summary(values: np.ndarray) -> dict[str, float]:
        return {
            "p10": float(np.quantile(values, 0.10)),
            "median": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "mean": float(values.mean()),
        }

    return {
        "contexts_per_user": 1,
        **exposure_metrics(rows, catalog_size=len(catalog_item_ids), top_k=10),
        "head_share_top10": sum(index in head_items for index in flat_indices) / n_recommendations,
        "repeat_rate_top10": repeat_rate(rows, recent_rows, top_k=10),
        "top1_repeat_rate": repeat_rate([[row[0]] if row else [] for row in rows], recent_rows, top_k=1),
        "mean_pairwise_user_overlap_jaccard_top10": _pairwise_jaccard(rows, seed),
        "mean_duration_seconds_top10": sum(known_selected) / len(known_selected) if known_selected else None,
        "duration_p10_seconds_top10": float(np.quantile(known_selected, 0.10)) if known_selected else None,
        "duration_median_seconds_top10": float(np.quantile(known_selected, 0.50)) if known_selected else None,
        "duration_p90_seconds_top10": float(np.quantile(known_selected, 0.90)) if known_selected else None,
        "duration_missing_fraction_top10": 1.0 - len(known_selected) / n_recommendations,
        "duration_short_threshold_seconds_train_p10": duration_threshold,
        "duration_short_fraction_top10": (
            sum(value < duration_threshold for value in known_selected) / len(known_selected)
            if known_selected and duration_threshold is not None else None
        ),
        "fit_unsupported_fraction_top10": sum(fit_support.get(raw_to_index[item] , 0) == 0 for item in flat_raw) / n_recommendations,
        "audit_unsupported_fraction_top10": sum(audit_support.get(raw_to_index[item], 0) == 0 for item in flat_raw) / n_recommendations,
        "support_counts_top10": {
            "train_nonorganic": support_summary(train_support_values),
            "rm_fit": support_summary(fit_support_values),
            "rm_audit": support_summary(audit_support_values),
        },
    }


def evaluate_final_selection(
    manifest_path: str | Path = "artifacts/final_selection.json",
    output_path: str | Path | None = None,
    allow_rerun: bool = False,
    rerun_reason: str | None = None,
    allow_post_test_research: bool = False,
) -> dict[str, Any]:
    """Evaluate the frozen model list once on the test split and save predictions."""
    import polars as pl

    evaluation_started = time.perf_counter()

    from tlab.config import load_config

    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"final selection manifest does not exist: {manifest_file}")
    selection = json.loads(manifest_file.read_text(encoding="utf-8"))
    if selection.get("selection_split") != "validation":
        raise ValueError("final selection must have been frozen on validation")
    test_status = selection.get("test_status", "not_run_at_freeze")
    if test_status == "post_test_research_iteration":
        if not allow_post_test_research:
            raise ValueError("selection is a post-test research iteration; pass --post-test-research explicitly")
        if selection.get("test_evaluation", {}).get("status") != "not_run_at_freeze":
            raise ValueError("post-test research selection already has a reference test evaluation")
    elif test_status != "not_run_at_freeze":
        raise ValueError(f"unsupported frozen test status: {test_status}")
    elif selection.get("test_evaluation", {}).get("status") != "not_run_at_freeze":
        raise ValueError("selection manifest does not certify that this evaluation is new at freeze time")
    if not allow_rerun and rerun_reason:
        raise ValueError("rerun_reason requires --allow-rerun")
    root = Path(selection["config_path"]).resolve().parent.parent
    target = _resolve(root, output_path or "artifacts/final_test_evaluation.json")
    if target.exists() and not allow_rerun:
        raise FileExistsError(f"final test result already exists: {target}; use an explicit evaluator-bug rerun reason to replace it")
    if allow_rerun and not str(rerun_reason or "").strip():
        raise ValueError("an evaluator-bug rerun requires a written --rerun-reason")

    current_source_hash = source_tree_sha256(root / "src" / "tlab")
    if current_source_hash != selection.get("source_sha256_at_freeze"):
        raise ValueError("tlab source changed after selection freeze; freeze again before test evaluation")
    protocol_path = Path(selection.get("data_protocol_path", _resolve(root, "data/processed/protocol.json"))).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("processed_listens_sha256") != selection.get("processed_listens_sha256"):
        raise ValueError("processed listens hash changed after selection freeze")
    if protocol.get("catalog_sha256") != selection.get("catalog_sha256"):
        raise ValueError("catalog hash changed after selection freeze")

    model_config_path = Path(selection.get("model_config_path", root / "configs" / "base.yaml")).resolve()
    expected_config_hash = selection.get("model_config_sha256")
    if expected_config_hash and sha256_file(model_config_path) != expected_config_hash:
        raise ValueError("model config changed after selection freeze")
    config = load_config(model_config_path)
    processed_dir = _resolve(root, config.data.get("processed_dir", "data/processed"))
    store = SequenceStore.load_cache(
        processed_dir / "sequence_index",
        expected_source_sha256=protocol["processed_listens_sha256"],
    )
    catalog = pl.read_parquet(processed_dir / "catalog.parquet").sort("token_id")
    item_ids = [int(value) for value in catalog.get_column("item_id").to_list()]
    training_counts = np.asarray(catalog.get_column("training_count").to_list(), dtype=np.float64)
    durations = catalog.get_column("catalog_duration_seconds").to_list()
    raw_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    protocol_eval = selection.get("test_protocol", {})
    anchors_per_user = int(protocol_eval.get("anchors_per_user", 10))
    anchor_seed = int(protocol_eval.get("anchor_seed", 17))
    batch_size = int(protocol_eval.get("batch_size", 16))
    top_k = int(protocol_eval.get("top_k", 10))
    bootstrap_seed = int(protocol_eval.get("bootstrap_seed", anchor_seed))
    if min(anchors_per_user, batch_size, top_k) < 1:
        raise ValueError("test protocol anchors_per_user, batch_size, and top_k must be positive")

    train_item_ids = _training_item_ids(store)
    target_metadata_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_items_by_user: dict[str, list[int]] = defaultdict(list)
    for user_index, position in store.evaluation_targets("test", per_user_limit=anchors_per_user, seed=anchor_seed):
        user = store.users[user_index]
        raw_item_id = int(user.raw_item_ids[position])
        ratio = float(user.ratio_raw[position])
        target_metadata_by_user[str(user.uid)].append({
            "item_id": raw_item_id,
            "positive": math.isfinite(ratio) and ratio > 0.5,
            "in_catalog": raw_item_id in raw_to_index,
            "known_in_train": raw_item_id in train_item_ids,
        })
        target_items_by_user[str(user.uid)].append(raw_item_id)

    baseline_started = time.perf_counter()
    baseline_results = _baseline_rankings(
        store, item_ids, training_counts, "test", anchors_per_user, anchor_seed, top_k,
    )
    baseline_elapsed = time.perf_counter() - baseline_started
    for name, baseline in baseline_results.items():
        baseline["ranking_summary"] = _ranking_summary(baseline["per_user_ranks"], target_metadata_by_user)
        baseline["compute"] = {
            "training_seconds": 0.0,
            "inference_seconds": baseline_elapsed,
            "inference_targets": baseline["ranking_summary"]["n_targets"],
            "inference_targets_per_second": (
                baseline["ranking_summary"]["n_targets"] / baseline_elapsed if baseline_elapsed > 0 else None
            ),
            "inference_timing_scope": "popularity and sequential-bigram controls evaluated together",
        }
    fit_support = _feature_support(processed_dir, "fit")
    audit_support = _feature_support(processed_dir, "audit")
    models = selection.get("models", [])
    if not models:
        raise ValueError("final selection contains no models")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_shape = ActorShape(
        vocab_size=len(item_ids) + 3,
        n_positions=int(config.model.get("n_positions", 256)),
        n_layer=int(config.model.get("n_layer", 4)),
        n_embd=int(config.model.get("n_embd", 256)),
        n_head=int(config.model.get("n_head", 4)),
        dropout=float(config.model.get("dropout", 0.1)),
    )
    result_models: dict[str, dict[str, Any]] = {
        name: {key: value for key, value in payload.items() if key not in {"per_user_top_item_ids", "per_user_ranks"}}
        for name, payload in baseline_results.items()
    }
    per_user_ranks: dict[str, dict[str, list[int | None]]] = {
        name: payload["per_user_ranks"] for name, payload in baseline_results.items()
    }
    per_user_top_item_ids: dict[str, dict[str, list[list[int]]]] = {
        name: payload["per_user_top_item_ids"] for name, payload in baseline_results.items()
    }
    for model in models:
        name = str(model["name"])
        record = model.get("checkpoint")
        if not record:
            raise ValueError(f"final test evaluator currently requires actor checkpoints; {name!r} has none")
        checkpoint_path = Path(record["path"]).resolve()
        if sha256_file(checkpoint_path) != record.get("sha256"):
            raise ValueError(f"checkpoint hash changed after selection freeze: {name}")
        run_manifest_path = Path(record["run_manifest_path"]).resolve()
        if sha256_file(run_manifest_path) != record.get("run_manifest_sha256"):
            raise ValueError(f"run manifest changed after selection freeze: {name}")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        actor = ItemGPT2Actor(actor_shape)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(checkpoint["model"])
        actor.to(device).eval()
        inference_started = time.perf_counter()
        ranking = evaluate_full_catalog(
            actor, store, item_ids, split="test", history_length=int(config.dataset.get("history_length", 200)),
            per_user_limit=anchors_per_user, seed=anchor_seed, batch_size=batch_size, top_k=top_k,
        )
        serving = _serving_diagnostics(
            ranking["per_user_top_item_ids"], store, item_ids, training_counts, durations,
            "test", anchors_per_user, anchor_seed, fit_support, audit_support,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_started
        summary = _ranking_summary(ranking["per_user_ranks"], target_metadata_by_user)
        result_models[name] = {
            "family": model.get("family"),
            "seed": model.get("seed"),
            "role": model.get("role"),
            "checkpoint": record,
            "split": "test",
            "n_users": ranking["n_users"],
            "compute": {
                "training_seconds": run_manifest.get("training_seconds", run_manifest.get("elapsed_seconds")),
                "training_targets_per_epoch": run_manifest.get("target_budget_per_epoch"),
                "training_epochs_or_updates": run_manifest.get("epoch_count", run_manifest.get("updates")),
                "training_optimizer_steps": run_manifest.get("global_step"),
                "training_peak_allocated_vram_bytes": run_manifest.get("peak_allocated_vram_bytes"),
                "inference_seconds": inference_seconds,
                "inference_targets": ranking.get("n_targets"),
                "inference_targets_per_second": (
                    ranking.get("n_targets", 0) / inference_seconds if inference_seconds > 0 else None
                ),
            },
            **summary,
            "serving_diagnostics": serving,
        }
        per_user_ranks[name] = ranking["per_user_ranks"]
        per_user_top_item_ids[name] = ranking["per_user_top_item_ids"]
        del actor

    reward_panel_results: dict[str, dict[str, Any]] = {}
    for model in models:
        panel = model.get("reward_panel")
        if not panel:
            continue
        files = panel["files"]
        for key, record in files.items():
            panel_path = Path(record["path"]).resolve()
            if sha256_file(panel_path) != record.get("sha256"):
                raise ValueError(f"reward panel input hash changed after selection freeze: {model['name']}:{key}")
        panel_output = target.parent / f"{target.stem}_reward_panel_{model['name']}.json"
        panel_result = _evaluate_policy_panel(
            model_config_path,
            files["sft_checkpoint"]["path"],
            model["checkpoint"]["path"],
            files["rm_fit_checkpoint"]["path"],
            files["rm_audit_checkpoint"]["path"],
            files["reward_scale_path"]["path"],
            beta=float(panel.get("beta", 0.1)),
            seed=anchor_seed,
            anchor_count=None,
            batch_size=4,
            output_path=panel_output,
            policy_label=str(model["name"]),
            reward_head=str(panel.get("reward_head", "ratio")),
            repeat_penalty=float(panel.get("repeat_penalty", 0.0)),
            panel_split="test",
        )
        reward_panel_results[str(model["name"])] = {
            "path": str(panel_output),
            "sha256": sha256_file(panel_output),
            "panel": panel_result.get("panel"),
            "panel_split": panel_result.get("panel_split"),
            "n_users": panel_result.get("n_users"),
            "metrics": panel_result.get("metrics"),
            "serving_diagnostics": panel_result.get("serving_diagnostics"),
            "paired_user_bootstrap_vs_sft": panel_result.get("paired_user_bootstrap_vs_sft"),
            "compute": {"inference_seconds": panel_result.get("elapsed_seconds")},
        }

    paired: dict[str, Any] = {}
    baseline_names = ("popularity", "sequential_bigram")
    frozen_by_name = {str(model["name"]): model for model in models}
    for name, ranks in per_user_ranks.items():
        if name not in frozen_by_name:
            continue
        paired[name] = {
            baseline: _paired_ndcg(ranks, per_user_ranks[baseline], bootstrap_seed)
            for baseline in baseline_names
        }
        reference_name = frozen_by_name[name].get("reference_name")
        if reference_name:
            if reference_name not in per_user_ranks:
                raise ValueError(f"reference model {reference_name!r} for {name!r} is absent from final.models")
            paired[name][f"reference:{reference_name}"] = _paired_ndcg(
                ranks, per_user_ranks[reference_name], bootstrap_seed,
            )

    payload = {
        "selection_manifest_path": str(manifest_file),
        "selection_manifest_sha256": sha256_file(manifest_file),
        "selected_family": selection.get("selected_family"),
        "selection_rule": selection.get("selection_rule"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "test_status": test_status,
        "historical_test_access": test_status == "post_test_research_iteration",
        "test_protocol": {
            "anchors_per_user": anchors_per_user,
            "anchor_seed": anchor_seed,
            "batch_size": batch_size,
            "top_k": top_k,
            "bootstrap_repetitions": 2000,
            "bootstrap_seed": bootstrap_seed,
        },
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
        "source_sha256": current_source_hash,
        "evaluation_wall_seconds": time.perf_counter() - evaluation_started,
        "models": result_models,
        "full_catalog_test_reward_panels": reward_panel_results,
        "paired_ndcg10_vs_baselines": paired,
        "per_user_ranks": per_user_ranks,
        "per_user_targets_item_ids": dict(target_items_by_user),
        "per_user_top_item_ids": per_user_top_item_ids,
        "rerun_reason": rerun_reason if allow_rerun else None,
    }
    write_json_atomic(target, payload)
    payload["output_path"] = str(target)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen final model selection on test")
    parser.add_argument("--manifest", default="artifacts/final_selection.json")
    parser.add_argument("--output")
    parser.add_argument("--allow-rerun", action="store_true")
    parser.add_argument("--rerun-reason")
    parser.add_argument("--post-test-research", action="store_true")
    args = parser.parse_args()
    result = evaluate_final_selection(
        args.manifest, output_path=args.output,
        allow_rerun=args.allow_rerun, rerun_reason=args.rerun_reason,
        allow_post_test_research=args.post_test_research,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
