from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch
import yaml

from tlab.data.store import SequenceStore
from tlab.evaluation.baselines import _train_transition_matrix, smoothed_transition_scores
from tlab.panel_tools.data import PanelTarget
from tlab.panel_tools.evaluator import load_actor_checkpoint
from tlab.panel_tools.statistics import paired_user_bootstrap, relative_quality_guard
from tlab.residual_support.features import POLICY_FEATURE_NAMES, CANDIDATE_FEATURE_NAMES
from tlab.residual_support.head import ResidualHeadSpec, ResidualPolicyHead
from tlab.residual_support.metrics import compact_guard, evaluate_full_catalog_policy, evaluate_score_matrix
from tlab.residual_support.objectives import FrozenRewardScale
from tlab.residual_support.rm import load_priors, sha256_file
from tlab.residual_support.scaler import TrainOnlyFeatureScaler
from tlab.residual_support.cache import _mapping_sha
from tlab.residual_support.train_policy import _load_cache, _resolve


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    primary = {key: value for key, value in metrics["primary_user_macro"].items() if key != "per_user"}
    slices = {name: {key: value for key, value in row.items() if key != "per_user"} for name, row in metrics["slices"].items()}
    return {key: value for key, value in metrics.items() if key not in {"slices", "per_user_primary_ndcg"}} | {
        "primary_user_macro": primary, "slices": slices,
    }


def _scale_from_json(path: Path) -> TrainOnlyFeatureScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_sha = payload.pop("sha256")
    scaler = TrainOnlyFeatureScaler(
        feature_names=tuple(payload["feature_names"]), log1p_columns=tuple(payload["log1p_columns"]),
        mean=tuple(payload["mean"]), scale=tuple(payload["scale"]),
        constant_columns=tuple(payload["constant_columns"]), fit_contexts=int(payload["fit_contexts"]),
        fit_candidate_rows=int(payload["fit_candidate_rows"]), protocol=str(payload["protocol"]),
    )
    if scaler.digest() != expected_sha:
        raise ValueError("saved feature scaler hash mismatch")
    return scaler


def _predict_head(checkpoint_path: Path, cache_arrays: dict[str, np.ndarray], scaler: TrainOnlyFeatureScaler,
                  device: torch.device, batch_size: int = 32) -> np.ndarray:
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    columns = tuple(int(x) for x in saved["feature_columns"])
    head = ResidualPolicyHead(ResidualHeadSpec(input_dim=len(columns), hidden_dim=32)).to(device)
    head.load_state_dict(saved["state_dict"])
    head.eval()
    values = cache_arrays["candidate_features"]
    result = np.empty((len(values), values.shape[1]), dtype=np.float32)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        raw = np.asarray(values[start:stop])
        scaled = scaler.transform(raw, POLICY_FEATURE_NAMES)[..., list(columns)]
        x = torch.from_numpy(np.ascontiguousarray(scaled)).to(device)
        reference = torch.from_numpy(np.asarray(cache_arrays["reference_conditional_logp"][start:stop]).copy()).to(device)
        with torch.no_grad():
            result[start:stop] = torch.softmax(reference + head(x), dim=-1).cpu().numpy().astype(np.float32)
    return result


def _analytic_q(reference: np.ndarray, rewards: np.ndarray, support_mass: np.ndarray,
                mean: float, std: float, floor: float, beta: float) -> np.ndarray:
    scale = max(float(std), float(floor))
    normalized = (np.asarray(rewards, dtype=np.float64) - float(mean)) / scale
    logits = np.asarray(reference, dtype=np.float64) + normalized / float(beta)
    logits -= logits.max(axis=-1, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=-1, keepdims=True)
    return result.astype(np.float32)


def _compact_pair(candidate: dict[int, float], baseline: dict[int, float], *, seed: int,
                  draws: int, tolerance: float = 0.02) -> dict[str, Any]:
    pair = paired_user_bootstrap(candidate, baseline, n_bootstrap=draws, seed=seed)
    guard = relative_quality_guard(candidate, baseline, tolerance=tolerance, n_bootstrap=draws, seed=seed + 1)
    return {
        "paired_ndcg10": {key: value for key, value in pair.items() if key != "user_differences"},
        "relative_2_percent_guard": compact_guard(guard),
    }


def _baseline_score_matrices(root: Path, core_cfg: dict[str, Any], panel: list[PanelTarget],
                             reference_probabilities: np.ndarray, item_ids: np.ndarray,
                             train_counts: np.ndarray, rm_fit_counts: np.ndarray) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    processed = _resolve(root, core_cfg["processed_dir"])
    processed_protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(_resolve(root, core_cfg["sequence_index"]),
                                     expected_source_sha256=processed_protocol["processed_listens_sha256"])
    raw_to_index = {int(raw): i for i, raw in enumerate(item_ids.tolist())}
    n, n_items = len(panel), len(item_ids)
    history = np.zeros((n, n_items), dtype=np.float32)
    actor_rerank = np.zeros((n, n_items), dtype=np.float32)
    markov = np.zeros((n, n_items), dtype=np.float32)
    actor_rank = np.empty((n, n_items), dtype=np.int32)
    transition_started = time.perf_counter()
    transition, examples = _train_transition_matrix(store, n_items)
    transition_seconds = time.perf_counter() - transition_started
    for row_index, target in enumerate(panel):
        user = store.users[target.user_index]
        start = max(0, target.position - 200)
        prior = np.asarray(user.raw_item_ids[start:target.position], dtype=np.int64)
        for raw in prior.tolist():
            index = raw_to_index.get(int(raw))
            if index is not None:
                history[row_index, index] += 1.0
        actor_order = np.lexsort((item_ids, -reference_probabilities[row_index]))
        actor_rank[row_index, actor_order] = np.arange(n_items, dtype=np.int32)
        actor_rerank[row_index] = history[row_index] * (n_items + 1.0) + (n_items - actor_rank[row_index])
        previous_token = int(user.tokens[target.position - 1]) if target.position > 0 else 1
        markov[row_index] = smoothed_transition_scores(transition.getrow(previous_token), train_counts, 1.0)
    frame = pl.scan_parquet(processed / "listens.parquet")
    boundaries = processed_protocol["boundaries_source_units"]
    recent = frame.filter(
        (pl.col("global_split") == "train")
        & (pl.col("timestamp") >= float(boundaries["train_cutoff_raw"]) - 14 * 86400)
        & (pl.col("timestamp") < float(boundaries["train_cutoff_raw"]))
        & (pl.col("token_id") >= 3)
    ).group_by("token_id").agg(pl.len().alias("n")).collect(engine="streaming")
    recent_counts = np.zeros(n_items, dtype=np.float32)
    for token, count in recent.iter_rows():
        idx = int(token) - 3
        if 0 <= idx < n_items:
            recent_counts[idx] = float(count)
    matrices = {
        "train_popularity": np.repeat(train_counts.astype(np.float32)[None, :], n, axis=0),
        "recent_train_popularity_14d": np.repeat(recent_counts[None, :], n, axis=0),
        "smoothed_markov": markov,
        "history_frequency": history,
        "history_frequency_reranker": actor_rerank,
    }
    metrics = {
        name: evaluate_score_matrix(panel=panel, scores=scores, item_ids=item_ids, rm_fit_counts=rm_fit_counts)
        for name, scores in matrices.items()
    }
    return metrics, {"transition_fit_seconds": transition_seconds, "transition_examples": examples,
                     "n_users_with_train_transitions": len(store.users), "markov_alpha": 1.0}


def evaluate_final(config_path: str | Path = "configs/residual_support/policy.yaml", root: str | Path = ".") -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = Path(root).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_root = _resolve(project_root, cfg["outputs"]["run_root"])
    selection_path = project_root / "artifacts/residual_support/protocol/frozen_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("freeze development selection before scoring the final retrospective panel")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "frozen_before_final_scoring":
        raise ValueError("final selection manifest is not frozen")
    cache_root = _resolve(project_root, cfg["cache_root"])
    final_meta, final_arrays, final_panel = _load_cache(cache_root / "final", include_full=True)
    if final_meta["panel_event_ids_sha256"] != selection["final_panel_event_ids_sha256"]:
        raise ValueError("final panel event IDs differ from the frozen selection manifest")
    # Check the scaler/cache binding before any final outcome scores are computed.
    from tlab.residual_support.train_policy import _cache_identity
    final_identity = _cache_identity(cache_root / "final/bound_identity.json")
    if final_identity.digest() != selection["cache_identity_sha256"]["final"]:
        raise ValueError("final cache identity differs from the frozen selection manifest")
    scaler = _scale_from_json(run_root / "scaler.json")
    if scaler.digest() != selection["scaler_sha256"]:
        raise ValueError("feature scaler changed after selection freeze")
    core_cfg = yaml.safe_load(_resolve(project_root, cfg["core_config"]).read_text(encoding="utf-8"))
    rm_cfg = yaml.safe_load(_resolve(project_root, cfg["rm_config"]).read_text(encoding="utf-8"))
    rm_root = _resolve(project_root, rm_cfg["outputs"]["root"])
    rm_metadata = json.loads((rm_root / "rm_diagnostics.json").read_text(encoding="utf-8"))
    rm_fit_counts = np.asarray(rm_metadata["rm_fit_item_count"]["counts_by_catalog_item"], dtype=np.int64)
    scale_json = json.loads((rm_root / "reward_scale.json").read_text(encoding="utf-8"))
    scale = FrozenRewardScale(float(scale_json["mean"]), float(scale_json["std"]),
                              float(scale_json["std_floor"]), str(scale_json["version"]))
    catalog = pl.read_parquet(_resolve(project_root, core_cfg["catalog"])).sort("token_id")
    item_ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    train_counts = np.asarray(catalog["training_count"].to_list(), dtype=np.float32)
    tokens = np.asarray(catalog["token_id"].to_list(), dtype=np.int64)
    mapping_sha = _mapping_sha(item_ids, tokens)
    if mapping_sha != final_meta["raw_token_mapping_sha256"]:
        raise ValueError("final evaluator catalog mapping changed")

    started = time.perf_counter()
    ref_cond = np.exp(np.asarray(final_arrays["reference_conditional_logp"], dtype=np.float32))
    reference = evaluate_full_catalog_policy(
        panel=final_panel, full_reference_logp=final_arrays["reference_logp_full"],
        support_indices=final_arrays["support_indices"], support_mass=final_arrays["support_mass"],
        policy_conditional=ref_cond, item_ids=item_ids, rm_fit_counts=rm_fit_counts,
    )
    raw_rewards = final_arrays["raw_candidate_rewards"]
    mean, std, floor = scale.mean, scale.std, scale.std_floor
    models: dict[str, dict[str, Any]] = {"strong_gpt": reference}
    head_records: dict[str, Any] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run_entry in selection["all_candidates"]:
        name = run_entry["run_name"]
        checkpoint_path = _resolve(project_root, run_entry["best_checkpoint_path"])
        if sha256_file(checkpoint_path) != run_entry["best_checkpoint_sha256"]:
            raise ValueError(f"selected head checkpoint hash changed: {name}")
        q = _predict_head(checkpoint_path, final_arrays, scaler, device)
        metrics = evaluate_full_catalog_policy(
            panel=final_panel, full_reference_logp=final_arrays["reference_logp_full"],
            support_indices=final_arrays["support_indices"], support_mass=final_arrays["support_mass"],
            policy_conditional=q, item_ids=item_ids, rm_fit_counts=rm_fit_counts,
        )
        models[name] = metrics
        head_records[name] = {"checkpoint_sha256": run_entry["best_checkpoint_sha256"],
                              "objective": run_entry["objective"], "beta": run_entry["beta"],
                              "seed": run_entry["seed"], "variant": run_entry["variant"],
                              "updates_completed": json.loads((_resolve(project_root, run_entry["best_checkpoint_path"]).parent / "run_manifest.json").read_text(encoding="utf-8"))["updates_completed"]}

    for beta in (1.0, 0.3):
        q = _analytic_q(final_arrays["reference_conditional_logp"], raw_rewards, final_arrays["support_mass"], mean, std, floor, beta)
        models[f"analytic_rm_beta_{beta:g}"] = evaluate_full_catalog_policy(
            panel=final_panel, full_reference_logp=final_arrays["reference_logp_full"],
            support_indices=final_arrays["support_indices"], support_mass=final_arrays["support_mass"],
            policy_conditional=q, item_ids=item_ids, rm_fit_counts=rm_fit_counts,
        )
    repeat_feature = np.asarray(final_arrays["candidate_features"][..., POLICY_FEATURE_NAMES.index("personal_repeat_count_200")])
    repeat_reward = (repeat_feature > 0).astype(np.float32)
    for beta in (1.0, 0.3):
        q = _analytic_q(final_arrays["reference_conditional_logp"], repeat_reward,
                        final_arrays["support_mass"], mean, std, floor, beta)
        models[f"matched_repeat_analytic_beta_{beta:g}"] = evaluate_full_catalog_policy(
            panel=final_panel, full_reference_logp=final_arrays["reference_logp_full"],
            support_indices=final_arrays["support_indices"], support_mass=final_arrays["support_mass"],
            policy_conditional=q, item_ids=item_ids, rm_fit_counts=rm_fit_counts,
        )

    ref_probability = np.exp(np.asarray(final_arrays["reference_logp_full"], dtype=np.float32))
    core_metrics, core_runtime = _baseline_score_matrices(
        project_root, core_cfg, final_panel, ref_probability, item_ids, train_counts, rm_fit_counts,
    )
    models.update(core_metrics)
    reference_user = reference["per_user_primary_ndcg"]
    draws = int(cfg["train"]["bootstrap_draws"])
    paired: dict[str, Any] = {}
    for name, metric in models.items():
        if name == "strong_gpt":
            continue
        candidate_user = metric["per_user_primary_ndcg"] if "per_user_primary_ndcg" in metric else {
            int(uid): float(values["ndcg@10"]) for uid, values in metric["primary_user_macro"]["per_user"].items()
        }
        paired[name] = _compact_pair(candidate_user, reference_user, seed=22000 + len(paired), draws=draws)

    selected_name = selection["selected_policy"]
    selected_final = models.get(selected_name, reference)
    if selected_name != "strong_gpt" and selected_name not in models:
        raise ValueError("frozen selected policy is missing from final evaluation")
    result = {
        "schema_version": 1, "protocol_id": cfg["protocol_id"], "status": "completed_retrospective_final_scoring",
        "evaluation_role": "held until selection freeze; retrospective, not independent due V1-full_model exposure",
        "selection_path": str(selection_path.relative_to(project_root)),
        "selection_sha256": sha256_file(selection_path), "selected_policy": selected_name,
        "selected_policy_final_metrics": _compact_metrics(selected_final),
        "panel": {"path": str((cache_root / "final/panel_rows.jsonl").relative_to(project_root)),
                  "event_ids_sha256": final_meta["panel_event_ids_sha256"], "n_users": len(final_panel),
                  "n_events": len(final_panel), "oov_targets": sum(row.token_id == 2 for row in final_panel),
                  "user_disjoint_from_train_and_development": True, "not_independent_test": True},
        "catalog": {"size": len(item_ids), "raw_token_mapping_sha256": mapping_sha},
        "reward_model": {"optimized_rm_sha256": final_meta["optimized_rm"]["sha256"],
                         "audit_predictor_used_for_selection": False,
                         "normalization": {"mean": mean, "std": std, "std_floor": floor, "version": scale.version}},
        "models": {name: _compact_metrics(metrics) for name, metrics in models.items()},
        "paired_vs_strong": paired,
        "head_provenance": head_records,
        "baseline_reference": "same strong full_model actor, same full-catalog scores and user-disjoint final panel",
        "matched_policy_factors": {"support_size": final_meta["support_size"], "reference_checkpoint_sha256": final_meta["actor"]["checkpoint_sha256"],
                                   "reward_scale": {"mean": mean, "std": std, "std_floor": floor, "version": scale.version},
                                   "candidate_contexts": "identical final contexts for sampled, exact, distill, analytic, and repeat support policies"},
        "guard_interpretation": "A failed 2% guard means preservation of quality is not established; a CI containing zero is not evidence of equality.",
        "runtime": {"transition_fit_seconds": core_runtime["transition_fit_seconds"],
                    "transition_examples": core_runtime["transition_examples"],
                    "total_final_scoring_seconds": time.perf_counter() - started,
                    "device": str(device), "head_count": len(head_records)},
        "limitations": [
            "This source test period and revision were opened in V1-full_model; user disjointness prevents within-panel_tools.1 role overlap but not historical outcome exposure.",
            "Logged-action reward predictions remain observational proxies; optimized proxy and audit predictor do not establish causal preference.",
            "The strong actor checkpoint may overlap with A_RL training events; its use as reference is retained by protocol and disclosed.",
            "Seed runs are repeated fits over the same users and do not increase the number of independent users.",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_root = _resolve(project_root, cfg["outputs"]["evaluation_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "final_policy_evaluation.json"
    csv_path = output_root / "final_policy_evaluation.csv"
    if output.exists() or csv_path.exists():
        raise FileExistsError("final evaluation artifact already exists; refusing to rescore/overwrite")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    rows = []
    for name, metrics in models.items():
        primary = metrics["primary_user_macro"]
        rows.append({"model": name, "ndcg@10": primary["ndcg@10"], "recall@10": primary["recall@10"],
                     "n_users": primary["n_users"], "target_catalog_coverage": metrics["target_catalog_coverage"]})
    pl.DataFrame(rows).write_csv(csv_path)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score frozen panel_tools.1 selection on the retrospective final panel")
    parser.add_argument("--config", default="configs/residual_support/policy.yaml")
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args(argv)
    result = evaluate_final(args.config, args.workspace)
    print(json.dumps({"status": result["status"], "selected_policy": result["selected_policy"],
                      "panel": result["panel"], "selected_metrics": result["selected_policy_final_metrics"]["primary_user_macro"],
                      "runtime": result["runtime"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
