from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import (BETAS, RM_NAMES, analytic_q, evaluate_conditional,
                   model_reward_metrics, open_v5_cache)
from .training import _load_existing_cache
from tlab.residual_support.objectives import FrozenRewardScale


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_dev_rm_analysis(root: Path, artifact_root: Path, scale: FrozenRewardScale,
                        split_manifest: dict[str, Any]) -> dict[str, Any]:
    out = artifact_root / "development"
    out.mkdir(parents=True, exist_ok=True)
    data, _ = open_v5_cache(artifact_root / "cache", "common")
    _, arrays, panel = _load_existing_cache(root / "artifacts/policy_diagnostics/cache/common_open_6655")
    dev_uids = set(int(x) for x in split_manifest["development_uids"])
    dev_ix = np.asarray([i for i, row in enumerate(panel) if int(row.uid) in dev_uids], dtype=np.int64)
    ref = np.asarray(arrays["reference_conditional_logp"][dev_ix], dtype=np.float64)
    mass = np.asarray(arrays["support_mass"][dev_ix], dtype=np.float64)
    reward_data = data["rm_rewards"]
    rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, Any]] = []
    qref = np.exp(ref)
    qref /= qref.sum(axis=1, keepdims=True)
    baseline = evaluate_conditional(panel, arrays, dev_ix, qref.astype(np.float32), _catalog_ids(root))
    rows.append({"method": "strong_gpt", "reward_model": "reference", "beta": None,
                 "ndcg10": baseline["ndcg10"], "hr10": baseline["hr10"],
                 "n": baseline["n"], "hits_at_10": baseline["hits_at_10"],
                 "oov_targets": baseline["oov_targets"], "positive_new": baseline["positive_new"],
                 "positive_repeat": baseline["positive_repeat"], "mass_weighted_kl": 0.0,
                 "normalized_reward_full_delta": 0.0})
    for uid, metric in baseline["per_user"].items():
        user_rows.append({"method": "strong_gpt", "uid": uid, **metric})
    for rm_index, rm_name in enumerate(RM_NAMES):
        raw = np.asarray(reward_data[dev_ix, rm_index, :], dtype=np.float32)
        for beta in BETAS:
            q = analytic_q(ref, raw, scale, beta)
            metrics = evaluate_conditional(panel, arrays, dev_ix, q, _catalog_ids(root))
            diag = model_reward_metrics(ref, q, raw, mass, scale)
            method = f"analytic_{rm_name}_b{beta:g}"
            rows.append({"method": method, "reward_model": rm_name, "beta": beta,
                         "ndcg10": metrics["ndcg10"], "hr10": metrics["hr10"],
                         "n": metrics["n"], "hits_at_10": metrics["hits_at_10"],
                         "oov_targets": metrics["oov_targets"],
                         "positive_new": metrics["positive_new"],
                         "positive_repeat": metrics["positive_repeat"], **diag})
            for uid, metric in metrics["per_user"].items():
                user_rows.append({"method": method, "uid": uid, **metric})

    priority = {"old_deployed": 0, "same_split_control": 1, "prefix_affinity": 2}
    analytic_rows = [r for r in rows if r["reward_model"] in RM_NAMES]
    selected = sorted(analytic_rows, key=lambda r: (-float(r["ndcg10"]),
                                                     float(r["mass_weighted_kl"]),
                                                     priority[str(r["reward_model"])],
                                                     BETAS.index(float(r["beta"]))))[0]
    selected_beta = float(selected["beta"])
    beta_index = BETAS.index(selected_beta)
    softer_beta = BETAS[beta_index - 1] if beta_index > 0 else None
    beta_search = [selected_beta] + ([float(softer_beta)] if softer_beta is not None else [])
    result = {
        "status": "complete",
        "panel_role": "development, selected before reporting scoring",
        "n": len(dev_ix), "reward_scale": {"mean": scale.mean, "std": scale.std,
                                            "std_floor": scale.std_floor, "version": scale.version},
        "rows": rows,
        "selected_analytic": {"reward_model": selected["reward_model"], "beta": selected_beta,
                              "ndcg10": selected["ndcg10"], "mass_weighted_kl": selected["mass_weighted_kl"],
                              "selection_rule": "best user-macro dev NDCG@10; tie lower mass-weighted KL, then simpler ready RM"},
        "policy_beta_candidates": beta_search,
        "policy_beta_selection_rule": "seed17 trained sampled-LOO checkpoint by development NDCG@10; tie lower mass-weighted KL",
    }
    _save_json(out / "rm_dev_results.json", result)
    _write_csv(out / "rm_dev_results.csv", rows)
    _write_csv(out / "rm_dev_per_user.csv", user_rows)
    _save_json(artifact_root / "analytic_selection.json", result["selected_analytic"] | {
        "policy_beta_candidates": beta_search,
        "selection_rule": result["selected_analytic"]["selection_rule"],
    })
    return result


def _catalog_ids(root: Path) -> np.ndarray:
    import polars as pl
    return np.asarray(pl.read_parquet(root / "data/processed_full_model_tminus14/catalog.parquet")
                      .sort("token_id")["item_id"].to_list(), dtype=np.int64)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat_rows = []
    for row in rows:
        item = dict(row)
        for key in ("positive_new", "positive_repeat"):
            if isinstance(item.get(key), dict):
                prefix = key + "_"
                for subkey, value in item.pop(key).items():
                    item[prefix + subkey] = value
        flat_rows.append(item)
    fields = list(dict.fromkeys(key for row in flat_rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(flat_rows)

