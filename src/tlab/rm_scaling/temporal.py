from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from tlab.data.store import SequenceStore
from tlab.residual_support.rm import load_priors
from tlab.frozen_policy.data import (
    _load_cache, evaluate_conditional, model_reward_metrics, open_v5_cache,
)
from tlab.frozen_policy.reporting import _event_timelines

from .experiment import (
    ART, BETA, BRANCHES, FINAL, REWARD_INDEX, ROOT, SEEDS,
    _aggregate_policy, _aggregate_target_scores, _behavior_metrics,
    _build_target_slices, _method_eval, _paired_bootstrap,
    _predict_v6_checkpoint, _score_target_sets, _load_scaler, _write_csv, _write_json,
)


def run_temporal_once() -> dict[str, Any]:
    started = time.perf_counter()
    output = ART / "temporal"
    cache_dir = output / "cache"
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output / "temporal_summary.json"
    if summary_path.exists():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            return prior

    panel_path = ROOT / "artifacts/residual_support/cache/final"
    panel_manifest_path = panel_path / "manifest.json"
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    _, raw_arrays, panel = _load_cache(panel_path)
    panel_jsonl = ROOT / "artifacts/residual_support/panels/test_retrospective.jsonl"
    panel_spec = json.loads((ROOT / "artifacts/residual_support/panels/test_retrospective_manifest.json").read_text(encoding="utf-8"))
    if len(panel) != 710 or panel_manifest["panel_event_ids_sha256"] != panel_spec["event_ids_sha256"]:
        raise AssertionError("prebuilt temporal test cache does not match the frozen 710-row panel")

    processed = ROOT / "data/processed_full_model_tminus14"
    source_protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(processed / "sequence_index",
                                     expected_source_sha256=source_protocol["processed_listens_sha256"])
    timelines = _event_timelines(ROOT, panel, store)
    row_indices = np.arange(len(panel), dtype=np.int64)
    test_times = np.asarray([timelines[int(row.uid)][row.position] for row in panel], dtype=np.float64)

    _, _, common_panel = _load_cache(ROOT / "artifacts/policy_diagnostics/cache/common_open_6655")
    split_manifest = json.loads((ROOT / "artifacts/frozen_policy/split_manifest.json").read_text(encoding="utf-8"))
    reporting_uids = set(int(x) for x in split_manifest["reporting_uids"])
    reporting_panel = [row for row in common_panel if int(row.uid) in reporting_uids]
    reporting_timelines = _event_timelines(ROOT, reporting_panel, store)
    reporting_times = np.asarray([reporting_timelines[int(row.uid)][row.position] for row in reporting_panel], dtype=np.float64)
    if float(test_times.min()) <= float(reporting_times.max()):
        raise AssertionError("predeclared temporal panel is not later than the Frozen reporting period")

    cache_meta = json.loads((ROOT / "artifacts/frozen_policy/cache/feature_cache_manifest.json").read_text(encoding="utf-8"))
    priors = load_priors(ROOT / "artifacts/frozen_policy/cache/c_priors_reconstructed.npz",
                         expected_mapping_sha256=cache_meta["raw_token_mapping_sha256"])
    embedding_path = ROOT / "artifacts/policy_diagnostics/rm/prefix_affinity_embeddings.npy"
    embedding = np.load(embedding_path, mmap_mode="r")
    catalog = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    from tlab.frozen_policy.data import _load_rm_models, _write_feature_and_reward_cache
    features_spec = _write_feature_and_reward_cache(
        ROOT, "temporal", raw_arrays, panel, item_ids, store, priors, embedding,
        _load_rm_models(ROOT), cache_dir)
    features = np.memmap(cache_dir / features_spec["features_path"], dtype=np.float32, mode="r",
                         shape=tuple(features_spec["feature_shape"]))
    existing_rewards = np.memmap(cache_dir / features_spec["rewards_path"], dtype=np.float32, mode="r",
                                 shape=tuple(features_spec["reward_shape"]))
    large_base = joblib.load(ART / "rm/large_base.joblib")
    large_affinity = joblib.load(ART / "rm/large_affinity.joblib")
    reward_path = cache_dir / "temporal_v6_rewards.bin"
    rewards = np.memmap(reward_path, dtype=np.float32, mode="w+", shape=(len(panel), 5, 500))
    rewards[:, 0, :] = existing_rewards[:, 1, :]  # small_base
    rewards[:, 1, :] = existing_rewards[:, 2, :]  # small_affinity
    rewards[:, 4, :] = existing_rewards[:, 0, :]  # old_deployed
    for start in range(0, len(panel), 8):
        stop = min(start + 8, len(panel))
        batch = np.asarray(features[start:stop], dtype=np.float32)
        base = batch[:, :, :35].reshape(-1, 35)
        affinity = batch[:, :, 45:54].reshape(-1, 9)
        rewards[start:stop, 2, :] = np.clip(large_base.predict(base), 0, 1).reshape(stop-start, 500)
        rewards[start:stop, 3, :] = np.clip(large_affinity.predict(np.column_stack((base, affinity))), 0, 1).reshape(stop-start, 500)
    rewards.flush()
    del rewards, existing_rewards
    rewards = np.memmap(reward_path, dtype=np.float32, mode="r", shape=(len(panel), 5, 500))
    arrays = {**raw_arrays, "candidate_features": features, "rm_rewards": rewards}
    if not np.isfinite(rewards).all():
        raise FloatingPointError("temporal reward cache contains non-finite values")
    row_cache_meta = {"status": "complete", "contexts": len(panel), "candidate_count": 500,
                      "feature_shape": list(features_spec["feature_shape"]),
                      "reward_shape": [len(panel), 5, 500],
                      "reward_order": list(REWARD_INDEX),
                      "feature_alignment_source": "same Frozen C feature builders on prebuilt final/test cache",
                      "feature_cache_path": str((cache_dir / features_spec["features_path"]).relative_to(ROOT)),
                      "reward_cache_path": str(reward_path.relative_to(ROOT)),
                      "reward_cache_sha256": __import__("hashlib").sha256(reward_path.read_bytes()).hexdigest(),
                      "large_base_model_sha256": __import__("hashlib").sha256((ART / "rm/large_base.joblib").read_bytes()).hexdigest(),
                      "large_affinity_model_sha256": __import__("hashlib").sha256((ART / "rm/large_affinity.joblib").read_bytes()).hexdigest()}
    _write_json(output / "temporal_cache_manifest.json", row_cache_meta)

    all_rows = row_indices
    target_meta = _build_target_slices(ROOT, panel, all_rows, store, timelines)
    ref = np.asarray(arrays["reference_conditional_logp"][all_rows], dtype=np.float64)
    qref = np.exp(ref); qref /= qref.sum(axis=1, keepdims=True)
    evaluated: dict[str, list[dict[str, Any]]] = {"strong_gpt": [
        _method_eval("strong_gpt", qref.astype(np.float32), None, panel, arrays, all_rows, item_ids)]}
    run_rows: dict[str, list[dict[str, Any]]] = {"strong_gpt": evaluated["strong_gpt"]}
    checkpoint_info: dict[str, Any] = {}
    matrix = json.loads((ART / "policy/matrix_summary.json").read_text(encoding="utf-8"))
    scaler = _load_scaler()
    for branch in ("S", "L", "E"):
        evaluated[branch] = []
        for run in matrix["runs"]:
            if run["branch"] != branch:
                continue
            seed = int(run["seed"])
            q, payload = _predict_v6_checkpoint(ROOT / run["best_checkpoint_path"], arrays,
                                                all_rows, scaler)
            result = _method_eval(branch, q, REWARD_INDEX[run["reward_model"]], panel, arrays,
                                  all_rows, item_ids)
            result["seed"], result["best_step"] = seed, int(payload["step"])
            evaluated[branch].append(result)
            checkpoint_info[f"{branch}_seed{seed}"] = run["best_checkpoint_path"]
        evaluated[branch].sort(key=lambda r: r["seed"])
        aggregate = _aggregate_policy(evaluated[branch], branch)
        aggregate["seed_results"] = evaluated[branch]
        run_rows[branch] = evaluated[branch]
        evaluated[f"{branch}_mean"] = [aggregate]

    # Reuse the exact Frozen sampled and exact checkpoints; no refit and no selection on this panel.
    selection = json.loads((ROOT / "artifacts/frozen_policy/reporting_selection.json").read_text(encoding="utf-8"))
    for group, name in (("sampled_rl", "tabular_rl"), ("exact_gradient", "tabular_exact")):
        group_results = []
        for seed in SEEDS:
            choice = selection[group][str(seed)]
            q, payload = _predict_v6_checkpoint(ROOT / choice["best_checkpoint_path"], arrays,
                                                all_rows, scaler)
            result = _method_eval(name, q, REWARD_INDEX["old_deployed"], panel, arrays,
                                  all_rows, item_ids)
            result["seed"], result["best_step"] = seed, int(payload["step"])
            group_results.append(result)
        group_results.sort(key=lambda r: r["seed"])
        aggregate = _aggregate_policy(group_results, name); aggregate["seed_results"] = group_results
        evaluated[name] = [aggregate]
        run_rows[name] = group_results

    method_results = {"strong_gpt": evaluated["strong_gpt"][0]}
    for name in ("S", "L", "E", "tabular_rl", "tabular_exact"):
        method_results[name] = evaluated[name][0] if name in {"tabular_rl", "tabular_exact"} else evaluated[f"{name}_mean"][0]

    # The same Frozen 24h set definition, including only complete test-split windows.
    target_scores: dict[str, dict[str, Any]] = {}
    behavior_rows = []
    per_user_rows = []
    for method, runs in run_rows.items():
        run_targets = []
        for run in runs:
            overall = _score_target_sets(run["top10_by_user"], target_meta["targets"])
            new = _score_target_sets(run["top10_by_user"], target_meta["new_targets"])
            repeat = _score_target_sets(run["top10_by_user"], target_meta["repeat_targets"])
            run_targets.append({"seed": run.get("seed"), "overall": overall, "new": new, "repeat": repeat})
            behavior_rows.append({"method": method, "seed": run.get("seed"),
                                  **_behavior_metrics(panel, all_rows, run["top10_by_user"], store, item_ids),
                                  "mass_weighted_kl": run["mass_weighted_kl"],
                                  "normalized_reward_full_delta": run["normalized_reward_full_delta"]})
            for uid in sorted(run["per_user_ndcg"]):
                per_user_rows.append({"method": method, "seed": run.get("seed"), "uid": uid,
                                      "next_ndcg10": run["per_user_ndcg"][uid],
                                      "next_hr10": run["per_user_hr"][uid],
                                      "24h_ndcg10": overall["per_user_ndcg"].get(uid),
                                      "24h_recall10": overall["per_user_recall"].get(uid),
                                      "24h_new_ndcg10": new["per_user_ndcg"].get(uid),
                                      "24h_new_recall10": new["per_user_recall"].get(uid),
                                      "24h_repeat_ndcg10": repeat["per_user_ndcg"].get(uid),
                                      "24h_repeat_recall10": repeat["per_user_recall"].get(uid)})
        if len(run_targets) > 1:
            target_scores[method] = {key: _aggregate_target_scores([r[key] for r in run_targets])
                                     for key in ("overall", "new", "repeat")}
        else:
            target_scores[method] = {key: run_targets[0][key] for key in ("overall", "new", "repeat")}

    method_rows = []
    for name, result in method_results.items():
        ci = _paired_bootstrap(result["per_user_ndcg"], method_results["strong_gpt"]["per_user_ndcg"],
                               20261040, draws=2000, levels=(.95,))
        t = target_scores[name]["overall"]
        method_rows.append({"method": name, "n_contexts": result["n"], "hits_at_10": result["hits_at_10"],
                            "oov_targets": result["oov_targets"], "next_ndcg10": result["ndcg10"],
                            "next_hr10": result["hr10"], "next_ndcg10_delta_vs_strong": ci["mean_difference"],
                            "next_ci95_low": ci["ci95_low"], "next_ci95_high": ci["ci95_high"],
                            "24h_full_horizon_contexts": target_meta["full_horizon_contexts"],
                            "24h_nonempty_contexts": t["n_contexts"], "24h_positive_target_ids": t["positive_target_ids"],
                            "24h_ndcg10": t["ndcg10"], "24h_recall10": t["recall10"],
                            "24h_new_n": target_scores[name]["new"]["n_contexts"],
                            "24h_new_ndcg10": target_scores[name]["new"]["ndcg10"],
                            "24h_new_recall10": target_scores[name]["new"]["recall10"],
                            "24h_repeat_n": target_scores[name]["repeat"]["n_contexts"],
                            "24h_repeat_ndcg10": target_scores[name]["repeat"]["ndcg10"],
                            "24h_repeat_recall10": target_scores[name]["repeat"]["recall10"],
                            "mass_weighted_kl": result["mass_weighted_kl"],
                            "normalized_reward_full_delta": result["normalized_reward_full_delta"]})
    contrasts = []
    for left, right, seed in (("L", "S", 20261041), ("L", "strong_gpt", 20261042)):
        ci = _paired_bootstrap(target_scores[left]["overall"]["per_user_ndcg"],
                               target_scores[right]["overall"]["per_user_ndcg"], seed,
                               draws=2000, levels=(.95,))
        contrasts.append({"contrast": f"{left} - {right}", "metric": "24h positive-nonorganic NDCG@10",
                          **ci, "role": "descriptive temporal robustness only; previously opened panel"})

    _write_csv(FINAL / "temporal_method_metrics.csv", method_rows)
    _write_csv(FINAL / "temporal_behavior_metrics.csv", behavior_rows)
    _write_csv(FINAL / "temporal_per_user_metrics.csv", per_user_rows)
    _write_csv(FINAL / "temporal_registered_contrasts.csv", contrasts)
    cache_manifest = {"panel_source": "artifacts/residual_support/cache/final",
                      "panel_jsonl": str(panel_jsonl.relative_to(ROOT)),
                      "panel_manifest": str((ROOT / "artifacts/residual_support/panels/test_retrospective_manifest.json").relative_to(ROOT)),
                      "panel_n": len(panel), "event_ids_sha256": panel_spec["event_ids_sha256"],
                      "panel_sha256": panel_spec["panel_sha256"],
                      "temporal_test_timestamp_min": float(test_times.min()),
                      "temporal_test_timestamp_median": float(np.median(test_times)),
                      "temporal_test_timestamp_max": float(test_times.max()),
                      "v5_reporting_timestamp_min": float(reporting_times.min()),
                      "v5_reporting_timestamp_max": float(reporting_times.max()),
                      "test_is_strictly_later": True,
                      "split_codes": {str(int(code)): sum(int(store.users[r.user_index].split_codes[r.position]) == int(code) for r in panel)
                                      for code in (1, 2)},
                      "full_horizon_contexts": target_meta["full_horizon_contexts"],
                      "empty_target_contexts": target_meta["empty_target_contexts"],
                      "24h_nonempty_contexts": target_meta["nonempty_contexts"],
                      "new_nonempty_contexts": target_meta["new_nonempty_contexts"],
                      "repeat_nonempty_contexts": target_meta["repeat_nonempty_contexts"]}
    result = {"status": "complete", "role": "predeclared later test-period robustness panel; previously opened; not an independent test",
              "methods": method_rows, "contrasts": contrasts, "behavior": behavior_rows,
              "panel": cache_manifest, "selected_checkpoints": checkpoint_info,
              "no_training_or_selection_on_temporal_panel": True,
              "elapsed_seconds": round(time.perf_counter() - started, 3)}
    _write_json(summary_path, result)
    return result


if __name__ == "__main__":
    print(json.dumps(run_temporal_once(), ensure_ascii=False))
