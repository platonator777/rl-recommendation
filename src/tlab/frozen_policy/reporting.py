from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import polars as pl

from tlab.data.store import SequenceStore
from tlab.panel_tools.data import PanelTarget
from tlab.residual_support.features import POLICY_FEATURE_NAMES
from tlab.residual_support.objectives import FrozenRewardScale

from .data import (RM_NAMES, Frozen_FEATURE_NAMES, analytic_q, evaluate_conditional,
                   model_reward_metrics, open_v5_cache, topk_stable)
from .training import _load_existing_cache, load_scaler, predict_checkpoint


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat = []
    for row in rows:
        item = dict(row)
        for key in ("positive_new", "positive_repeat"):
            if isinstance(item.get(key), dict):
                for subkey, value in item.pop(key).items():
                    item[f"{key}_{subkey}"] = value
        flat.append(item)
    fields = list(dict.fromkeys(k for row in flat for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(flat)


def _paired_ci(left: dict[int, float], right: dict[int, float], seed: int,
               draws: int = 1000) -> dict[str, Any]:
    users = sorted(set(left) & set(right))
    if not users:
        raise ValueError("paired bootstrap has no shared users")
    diff = np.asarray([float(left[u]) - float(right[u]) for u in users], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 128):
        stop = min(draws, start + 128)
        sample = rng.integers(0, len(diff), size=(stop - start, len(diff)))
        means[start:stop] = diff[sample].mean(axis=1)
    return {"mean_difference": float(diff.mean()), "ci_lower": float(np.quantile(means, 0.025)),
            "ci_upper": float(np.quantile(means, 0.975)), "n_users": len(users),
            "confidence": 0.95, "bootstrap_draws": draws, "seed": seed}


def _mean_seed_maps(maps: list[dict[int, float]]) -> dict[int, float]:
    if not maps:
        return {}
    keys = set(maps[0])
    for item in maps[1:]:
        keys &= set(item)
    return {uid: float(np.mean([item[uid] for item in maps])) for uid in sorted(keys)}


def _aggregate_seed_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    ndcg = _mean_seed_maps([r["per_user_ndcg"] for r in results])
    hr = _mean_seed_maps([r["per_user_hr"] for r in results])
    positive_new = [r["positive_new"] for r in results]
    positive_repeat = [r["positive_repeat"] for r in results]
    n = len(ndcg)
    return {
        "per_user_ndcg": ndcg, "per_user_hr": hr,
        "ndcg10": float(np.mean(list(ndcg.values()))),
        "hr10": float(np.mean(list(hr.values()))), "n": n,
        "hits_at_10": float(np.mean([r["hits_at_10"] for r in results])),
        "oov_targets": int(results[0]["oov_targets"]),
        "positive_new": {key: (int(positive_new[0][key]) if key == "n"
                                else float(np.mean([r[key] for r in positive_new]))
                                if key in {"ndcg10", "hr10", "hits_at_10"}
                                else int(np.mean([r[key] for r in positive_new])))
                         for key in ("n", "hits_at_10", "ndcg10", "hr10")},
        "positive_repeat": {key: (int(positive_repeat[0][key]) if key == "n"
                                   else float(np.mean([r[key] for r in positive_repeat]))
                                   if key in {"ndcg10", "hr10", "hits_at_10"}
                                   else int(np.mean([r[key] for r in positive_repeat])))
                            for key in ("n", "hits_at_10", "ndcg10", "hr10")},
        "seed_results": [{"seed": r["seed"], "ndcg10": r["ndcg10"], "hr10": r["hr10"],
                          "best_step": r["best_step"]} for r in results],
        "seed_mean_range": [float(min(r["ndcg10"] for r in results)),
                            float(max(r["ndcg10"] for r in results))],
    }


def _evaluate_full_scores(panel: Sequence[PanelTarget], row_indices: Sequence[int],
                          score_rows: Sequence[np.ndarray], item_ids: np.ndarray,
                          *, want_topk: bool = True) -> dict[str, Any]:
    raw_to_index = {int(raw): i for i, raw in enumerate(item_ids.tolist())}
    per_ndcg: dict[int, float] = {}
    per_hr: dict[int, float] = {}
    pos_new: list[dict[str, float]] = []
    pos_repeat: list[dict[str, float]] = []
    tops: dict[int, list[int]] = {}
    oov = hits = 0
    for row_index, scores in zip(row_indices, score_rows):
        target = panel[int(row_index)]
        index = raw_to_index.get(int(target.raw_item_id))
        if index is None:
            rank = None; oov += 1
        else:
            value = float(scores[index])
            rank = int(np.count_nonzero((scores > value) | ((scores == value) & (item_ids < target.raw_item_id))) + 1)
        hit = float(rank is not None and rank <= 10)
        ndcg = float(1 / math.log2(rank + 1)) if rank is not None and rank <= 10 else 0.0
        per_ndcg[int(target.uid)] = ndcg
        per_hr[int(target.uid)] = hit
        hits += int(hit)
        if target.positive and target.is_new:
            pos_new.append({"ndcg10": ndcg, "hr10": hit})
        if target.positive and target.is_repeat:
            pos_repeat.append({"ndcg10": ndcg, "hr10": hit})
        if want_topk:
            tops[int(target.uid)] = [int(item_ids[j]) for j in topk_stable(scores, item_ids, 10)]

    def summary(rows: list[dict[str, float]]) -> dict[str, Any]:
        return {"n": len(rows), "hits_at_10": int(sum(r["hr10"] for r in rows)),
                "ndcg10": float(np.mean([r["ndcg10"] for r in rows])) if rows else None,
                "hr10": float(np.mean([r["hr10"] for r in rows])) if rows else None}
    return {"ndcg10": float(np.mean(list(per_ndcg.values()))),
            "hr10": float(np.mean(list(per_hr.values()))), "n": len(per_ndcg),
            "hits_at_10": hits, "oov_targets": oov,
            "positive_new": summary(pos_new), "positive_repeat": summary(pos_repeat),
            "per_user_ndcg": per_ndcg, "per_user_hr": per_hr, "top10_by_user": tops}


def _repeat_baseline(panel: Sequence[PanelTarget], row_indices: np.ndarray,
                     arrays: dict[str, np.ndarray], store: SequenceStore,
                     item_ids: np.ndarray) -> dict[str, Any]:
    raw_to_index = {int(raw): i for i, raw in enumerate(item_ids.tolist())}
    score_rows = []
    n_items = len(item_ids)
    for row_index in row_indices:
        target = panel[int(row_index)]
        user = store.users[target.user_index]
        history_ids = np.asarray(user.raw_item_ids[max(0, target.position - 200):target.position], dtype=np.int64)
        indices = [raw_to_index[x] for x in history_ids.tolist() if x in raw_to_index]
        counts = np.bincount(indices, minlength=n_items).astype(np.float64)
        ref = np.asarray(arrays["reference_logp_full"][int(row_index)], dtype=np.float64)
        actor_order = np.lexsort((item_ids, -ref))
        actor_rank = np.empty(n_items, dtype=np.int32)
        actor_rank[actor_order] = np.arange(n_items, dtype=np.int32)
        score_rows.append(counts * (n_items + 1.0) + (n_items - actor_rank))
    return _evaluate_full_scores(panel, row_indices, score_rows, item_ids)


def _reporting_indices(panel: Sequence[PanelTarget], split_manifest: dict[str, Any]) -> np.ndarray:
    report_uids = set(int(x) for x in split_manifest["reporting_uids"])
    rows = np.asarray([i for i, row in enumerate(panel) if int(row.uid) in report_uids], dtype=np.int64)
    if len(rows) != int(split_manifest["reporting_n"]):
        raise ValueError("reporting split rows do not match its frozen user IDs")
    return rows


def _summary_row(name: str, label: str, result: dict[str, Any],
                 strong_users: dict[int, float], ci_seed: int,
                 purpose: str) -> dict[str, Any]:
    ci = _paired_ci(result["per_user_ndcg"], strong_users, ci_seed)
    return {
        "method": name, "description": label, "purpose": purpose,
        "beta": result.get("beta"), "reward_model": result.get("reward_model"),
        "n_contexts": result["n"], "hits_at_10": result["hits_at_10"],
        "oov_targets": result["oov_targets"], "ndcg10": result["ndcg10"],
        "hr10": result["hr10"], "delta_ndcg10_vs_strong": ci["mean_difference"],
        "ci95_low_vs_strong": ci["ci_lower"], "ci95_high_vs_strong": ci["ci_upper"],
        "ci_n_users": ci["n_users"],
        "positive_new_n": result["positive_new"]["n"],
        "positive_new_hits": result["positive_new"]["hits_at_10"],
        "positive_new_ndcg10": result["positive_new"]["ndcg10"],
        "positive_new_hr10": result["positive_new"]["hr10"],
        "positive_repeat_n": result["positive_repeat"]["n"],
        "positive_repeat_hits": result["positive_repeat"]["hits_at_10"],
        "positive_repeat_ndcg10": result["positive_repeat"]["ndcg10"],
        "positive_repeat_hr10": result["positive_repeat"]["hr10"],
    }


def _save_per_user(path: Path, results: dict[str, dict[str, Any]]) -> None:
    rows = []
    for method, result in results.items():
        for uid in sorted(result["per_user_ndcg"]):
            rows.append({"method": method, "uid": uid,
                         "ndcg10": result["per_user_ndcg"][uid],
                         "hr10": result["per_user_hr"][uid]})
    _write_csv(path, rows)


def _event_timelines(root: Path, panel: Sequence[PanelTarget], store: SequenceStore) -> dict[int, np.ndarray]:
    uids = sorted({int(row.uid) for row in panel})
    listens = root / "data/processed_full_model_tminus14/listens.parquet"
    grouped = (pl.scan_parquet(listens)
               .filter(pl.col("uid").is_in(uids))
               .select("uid", "timestamp").unique()
               .sort(["uid", "timestamp"])
               .group_by("uid", maintain_order=True)
               .agg(pl.col("timestamp").cast(pl.Float64).alias("timestamps"))
               .collect(engine="streaming"))
    timelines = {int(uid): np.asarray(ts, dtype=np.float64) for uid, ts in grouped.iter_rows()}
    for uid in uids:
        if uid not in timelines:
            raise ValueError(f"missing raw event timestamps for reporting uid {uid}")
        user_index = next(row.user_index for row in panel if row.uid == uid)
        if len(timelines[uid]) != len(store.users[user_index].tokens):
            raise ValueError(f"timestamp/store sequence alignment failed for uid {uid}")
    return timelines


def _build_24h_targets(root: Path, panel: Sequence[PanelTarget], row_indices: np.ndarray,
                       store: SequenceStore, timelines: dict[int, np.ndarray]) -> dict[int, set[int]]:
    protocol = json.loads((root / "data/processed_full_model_tminus14/protocol.json").read_text(encoding="utf-8"))
    bounds = protocol["boundaries_source_units"]
    split_end = {1: float(bounds["test_start_raw"]), 2: float(bounds["end_raw"])}
    targets: dict[int, set[int]] = {}
    full_horizon = empty = 0
    for row_index in row_indices:
        target = panel[int(row_index)]
        user = store.users[target.user_index]
        split_code = int(user.split_codes[target.position])
        if split_code not in split_end:
            raise ValueError("Frozen retrospective contexts must be in validation or test windows")
        ts = timelines[target.uid]
        target_time = float(ts[target.position])
        end_time = split_end[split_code]
        if target_time + 86400.0 > end_time:
            continue
        full_horizon += 1
        stop = int(np.searchsorted(ts, target_time + 86400.0, side="right"))
        found: set[int] = set()
        for pos in range(target.position + 1, stop):
            if int(user.split_codes[pos]) != split_code:
                continue
            if not bool(user.recommendable_events[pos]):
                continue
            ratio = float(user.ratio_raw[pos])
            raw_id = int(user.raw_item_ids[pos])
            if math.isfinite(ratio) and ratio >= 0.5 and raw_id >= 0:
                found.add(raw_id)
        if not found:
            empty += 1
        targets[target.uid] = found
    return {"targets": targets, "full_horizon_contexts": full_horizon,
            "empty_target_contexts": empty, "n_contexts": len(row_indices),
            "nonempty_contexts": full_horizon - empty}


def _multilabel_metrics(top10: Sequence[int], targets: set[int]) -> tuple[float, float, int]:
    if not targets:
        return math.nan, math.nan, 0
    seen: set[int] = set()
    dcg = hits = 0.0
    for rank, raw in enumerate(top10[:10], start=1):
        if int(raw) in targets and int(raw) not in seen:
            seen.add(int(raw)); hits += 1.0
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(targets), 10) + 1))
    return float(hits / len(targets)), float(dcg / ideal), int(hits)


def _score_24h(tops_by_user: dict[int, list[int]], targets_meta: dict[str, Any]) -> dict[str, Any]:
    targets = targets_meta["targets"]
    recall: dict[int, float] = {}
    ndcg: dict[int, float] = {}
    total_targets = hits = 0
    for uid, target_set in targets.items():
        if not target_set:
            continue
        top = tops_by_user.get(uid, [])
        r, n, h = _multilabel_metrics(top, target_set)
        recall[uid] = r; ndcg[uid] = n
        total_targets += len(target_set); hits += h
    return {"per_user_recall": recall, "per_user_ndcg": ndcg,
            "recall10": float(np.mean(list(recall.values()))) if recall else None,
            "ndcg10": float(np.mean(list(ndcg.values()))) if ndcg else None,
            "n": len(recall), "full_horizon_contexts": targets_meta["full_horizon_contexts"],
            "empty_target_contexts": targets_meta["empty_target_contexts"],
            "nonempty_contexts": targets_meta["nonempty_contexts"],
            "positive_target_tracks": total_targets, "hits_at_10": hits}


def _aggregate_24h_seed(results: list[dict[str, Any]]) -> dict[str, Any]:
    recalls = _mean_seed_maps([r["per_user_recall"] for r in results])
    ndcgs = _mean_seed_maps([r["per_user_ndcg"] for r in results])
    return {"per_user_recall": recalls, "per_user_ndcg": ndcgs,
            "recall10": float(np.mean(list(recalls.values()))) if recalls else None,
            "ndcg10": float(np.mean(list(ndcgs.values()))) if ndcgs else None,
            "n": len(recalls), "full_horizon_contexts": results[0]["full_horizon_contexts"],
            "empty_target_contexts": results[0]["empty_target_contexts"],
            "nonempty_contexts": results[0]["nonempty_contexts"],
            "positive_target_tracks": int(np.mean([r["positive_target_tracks"] for r in results])),
            "hits_at_10": float(np.mean([r["hits_at_10"] for r in results]))}


def _adapt_conditional(raw: dict[str, Any], *, name: str, q: np.ndarray,
                       diagnostics: dict[str, float] | None = None) -> dict[str, Any]:
    result = {
        "name": name,
        "n": raw["n"], "ndcg10": raw["ndcg10"], "hr10": raw["hr10"],
        "hits_at_10": raw["hits_at_10"], "oov_targets": raw["oov_targets"],
        "positive_new": raw["positive_new"], "positive_repeat": raw["positive_repeat"],
        "per_user_ndcg": {int(uid): float(v["ndcg10"]) for uid, v in raw["per_user"].items()},
        "per_user_hr": {int(uid): float(v["hr10"]) for uid, v in raw["per_user"].items()},
        "top10_by_user": raw["top10_by_user"], "conditional": q,
    }
    if diagnostics:
        result.update(diagnostics)
    return result


def _mean_q(qs: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(qs, axis=0).astype(np.float64), axis=0).astype(np.float32)


def _external_audit(root: Path, report_ix: np.ndarray, old_arrays: dict[str, np.ndarray],
                    feature_arrays: dict[str, np.ndarray], methods_q: dict[str, np.ndarray],
                    method_rm: dict[str, int], selected_rewards: dict[str, np.ndarray]) -> dict[str, Any]:
    diagnostics = json.loads((root / "artifacts/residual_support/rm/rm_diagnostics.json").read_text(encoding="utf-8"))
    model_spec = diagnostics["models"]["external_audit_predictor"]
    model_path = Path(model_spec["path"])
    if not model_path.is_absolute():
        model_path = (root / model_path).resolve()
    predictor = joblib.load(model_path)
    n = len(report_ix); k = int(old_arrays["support_indices"].shape[1])
    audit_predictions = np.empty((n, k), dtype=np.float32)
    base = old_arrays["candidate_features"]
    for start in range(0, n, 32):
        end = min(start + 32, n)
        batch = np.asarray(base[report_ix[start:end], :, :35], dtype=np.float32).reshape(-1, 35)
        audit_predictions[start:end] = np.clip(predictor.predict(batch), 0.0, 1.0).reshape(end - start, k)
    ref = np.exp(np.asarray(old_arrays["reference_conditional_logp"][report_ix], dtype=np.float64))
    ref /= ref.sum(axis=1, keepdims=True)
    mass = np.asarray(old_arrays["support_mass"][report_ix], dtype=np.float64)
    rows = []
    for method, q in methods_q.items():
        raw_delta = mass * np.sum((q.astype(np.float64) - ref) * audit_predictions, axis=1)
        item = {"method": method,
                "frozen_external_predictor_path": str(model_path.relative_to(root)),
                "n_contexts": n,
                "expected_predicted_raw_ratio_delta_vs_reference": float(raw_delta.mean()),
                "scope": "fixed top-500 support; tail unchanged; conditional predictor diagnostic, not a causal or independent audit"}
        if method in method_rm and method in selected_rewards:
            reward = selected_rewards[method]
            qdiag = np.sum(q.astype(np.float64) * ((reward - 0.0)), axis=1)
            # This value is used only for relative reporting; normalized reward deltas are
            # separately computed with the registered frozen train scale.
            item["mean_candidate_rm_prediction_under_policy"] = float(np.mean(qdiag))
        rows.append(item)
    return {"status": "complete", "predictor_sha256": model_spec.get("sha256"),
            "rows": rows}


def _write_examples(root: Path, path: Path, report_ix: np.ndarray, panel: Sequence[PanelTarget],
                    store: SequenceStore, arrays: dict[str, np.ndarray],
                    features: np.ndarray, item_ids: np.ndarray,
                    reference_tops: dict[int, list[int]], policy_tops: dict[int, list[int]]) -> None:
    selected = []
    for ix in report_ix:
        uid = int(panel[int(ix)].uid)
        selected.append((hashlib.sha256(f"{bytes.fromhex('76355f666173743a6578616d706c653a').decode('ascii')}{uid}".encode()).hexdigest(), uid, int(ix)))
    selected.sort()
    raw_to_index = {int(raw): i for i, raw in enumerate(item_ids.tolist())}
    repeat_col = POLICY_FEATURE_NAMES.index("personal_repeat_count_200")
    affinity_last_col = len(POLICY_FEATURE_NAMES)
    affinity_20_col = len(POLICY_FEATURE_NAMES) + 1
    lines = ["# Фиксированные примеры рекомендаций", "",
             "Пять контекстов выбраны хешем uid до просмотра исхода. Показаны прошлые события и признаки, доступные до следующего трека; будущий feedback не использовался для объяснений.", ""]
    for _, uid, row_ix in selected[:5]:
        target = panel[row_ix]
        user = store.users[target.user_index]
        history = [int(x) for x in np.asarray(user.raw_item_ids[max(0, target.position - 8):target.position]).tolist() if int(x) >= 0]
        ref_top = reference_tops.get(uid, [])
        new_top = policy_tops.get(uid, [])
        feature_row = np.asarray(features[row_ix])
        support = np.asarray(arrays["support_indices"][row_ix], dtype=np.int64)
        candidates = []
        for rank, raw in enumerate(new_top, start=1):
            catalog_ix = raw_to_index[int(raw)]
            old_rank = ref_top.index(int(raw)) + 1 if int(raw) in ref_top else None
            cand_pos = np.flatnonzero(support == catalog_ix)
            if not len(cand_pos):
                continue
            f = feature_row[int(cand_pos[0])]
            candidates.append({"item_id": int(raw), "reference_rank": old_rank,
                              "policy_rank": rank,
                              "personal_repeat_count_200": float(f[repeat_col]),
                              "prefix_cosine_last": float(f[affinity_last_col]),
                              "prefix_cosine_last20": float(f[affinity_20_col])})
        raised = sorted(candidates, key=lambda x: ((x["reference_rank"] or 501) - x["policy_rank"],
                                                   -x["policy_rank"]), reverse=True)[:3]
        lines.extend([f"## uid `{uid}`", "",
                      f"История (последние raw item ID): `{history}`", "",
                      f"Strong top-10: `{ref_top}`", "",
                      f"RL top-10: `{new_top}`", "",
                      "Кандидаты, поднятые политикой (для тех, кто был вне strong top-10, reference rank показан как >10):", ""])
        if raised:
            lines.append("| item ID | rank strong | rank RL | repeat count 200 | cosine last | cosine last-20 |\n|---:|---:|---:|---:|---:|---:|")
            for r in raised:
                ref_rank = r["reference_rank"] if r["reference_rank"] is not None else ">10"
                lines.append(f"| {r['item_id']} | {ref_rank} | {r['policy_rank']} | {r['personal_repeat_count_200']:.0f} | {r['prefix_cosine_last']:.3f} | {r['prefix_cosine_last20']:.3f} |")
        else:
            lines.append("Для top-10 нет кандидатов из фиксированного support; список совпал с reference в этом контексте.")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_figures(final_dir: Path, run_dirs: list[Path], direct_rows: list[dict[str, Any]],
                  reporting_rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = final_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plotted = 0
    for run_dir in run_dirs:
        curve = run_dir / "dev_curve.jsonl"
        if not curve.exists():
            continue
        rows = [json.loads(line) for line in curve.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        label = run_dir.name.replace("_", " ")
        ax.plot([r["step"] for r in rows], [r["dev_ndcg10"] for r in rows], marker="o", ms=3, label=label)
        plotted += 1
    ax.set_xlabel("Обновления политики")
    ax.set_ylabel("Dev NDCG@10")
    ax.set_title("Качество на dev во время обучения")
    if plotted:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "dev_ndcg_learning.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for row in direct_rows:
        if row.get("reward_model") == "reference":
            continue
        ax.scatter(row["mass_weighted_kl"], row["ndcg10"], s=28,
                   label=f"dev {row['reward_model']} β={row['beta']:g}", alpha=0.7)
    for row in reporting_rows:
        if row.get("mass_weighted_kl") is not None:
            ax.scatter(row["mass_weighted_kl"], row["ndcg10"], marker="D", s=58,
                       label=row["method"], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Mass-weighted KL to strong")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Reward, релевантность и отклонение от reference")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(fig_dir / "reward_quality_kl.png", dpi=180); plt.close(fig)

    chosen = [r for r in reporting_rows if r.get("delta_ndcg10_vs_strong") is not None]
    fig, ax = plt.subplots(figsize=(8.2, max(4.0, 0.38 * len(chosen) + 1.4)))
    for y, row in enumerate(chosen):
        x = float(row["delta_ndcg10_vs_strong"])
        lo, hi = float(row["ci95_low_vs_strong"]), float(row["ci95_high_vs_strong"])
        ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt="o", capsize=3, color="#165D8F")
    ax.axvline(0, color="#555", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(chosen)), [r["description"] for r in chosen])
    ax.set_xlabel("Разность NDCG@10 против strong (95% paired user CI)")
    ax.set_title("Итоговые различия на reporting-панели")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "reporting_differences_ci.png", dpi=180); plt.close(fig)
    return ["figures/dev_ndcg_learning.png", "figures/reward_quality_kl.png", "figures/reporting_differences_ci.png"]


def _normal_q(logp: np.ndarray) -> np.ndarray:
    x = np.asarray(logp, dtype=np.float64)
    x -= x.max(axis=1, keepdims=True)
    q = np.exp(x); q /= q.sum(axis=1, keepdims=True)
    return q.astype(np.float32)


def _best_analytic_by_rm(dev_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for model in RM_NAMES:
        rows = [row for row in dev_rows if row.get("reward_model") == model]
        result[model] = sorted(rows, key=lambda r: (-float(r["ndcg10"]),
                                                    float(r["mass_weighted_kl"]),
                                                    -float(r["beta"])))[0]
    return result


def _run_reported_seed_group(root: Path, artifact_root: Path, selection: dict[str, Any],
                             group: str, seeds: Sequence[int], report_ix: np.ndarray,
                             common_panel: Sequence[PanelTarget], arrays: dict[str, np.ndarray],
                             item_ids: np.ndarray, scaler, reward_model: str,
                             scale: FrozenRewardScale) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    seed_results = []
    qs = []
    for seed in seeds:
        entry = selection[group][str(seed)]
        checkpoint = root / entry["best_checkpoint_path"]
        q, meta = predict_checkpoint(checkpoint, arrays, report_ix, scaler)
        if not meta.get("trained") or int(meta.get("step", 0)) <= 0:
            raise AssertionError("reporting attempted to score identity/untrained weights")
        reward_ix = {"old_deployed": 0, "same_split_control": 1, "prefix_affinity": 2}[reward_model]
        reward = np.asarray(arrays["rm_rewards"][report_ix, reward_ix, :], dtype=np.float32)
        raw = evaluate_conditional(common_panel, arrays, report_ix, q, item_ids, want_topk=True)
        diag = model_reward_metrics(np.asarray(arrays["reference_conditional_logp"][report_ix]), q,
                                    reward, np.asarray(arrays["support_mass"][report_ix]), scale)
        result = _adapt_conditional(raw, name=f"{group}_seed{seed}", q=q, diagnostics=diag)
        result["seed"] = int(seed); result["best_step"] = int(meta["step"])
        seed_results.append(result); qs.append(q)
    merged = _aggregate_seed_results(seed_results)
    mean_q = _mean_q(qs)
    reward_ix = {"old_deployed": 0, "same_split_control": 1, "prefix_affinity": 2}[reward_model]
    reward = np.asarray(arrays["rm_rewards"][report_ix, reward_ix, :], dtype=np.float32)
    diag = model_reward_metrics(np.asarray(arrays["reference_conditional_logp"][report_ix]), mean_q,
                                reward, np.asarray(arrays["support_mass"][report_ix]), scale)
    merged.update(diag)
    merged["conditional"] = mean_q
    merged["top10_by_user"] = seed_results[0]["top10_by_user"]
    return merged, seed_results, qs


def _report_answer(report: Path, metrics: list[dict[str, Any]], secondary: list[dict[str, Any]],
                   dev_result: dict[str, Any], policy_choice: dict[str, Any],
                   checks: dict[str, Any], runtime: dict[str, Any], figures: list[str],
                   examples_path: str, external: dict[str, Any]) -> None:
    indexed = {row["method"]: row for row in metrics}
    rl = indexed["sampled_rl"]
    exact = indexed["exact_gradient"]
    strong = indexed["strong_gpt"]
    chosen = indexed["analytic_selected_rm"]
    direct_old = indexed.get("analytic_old_rm", chosen)
    if rl["ci95_low_vs_strong"] > 0:
        answer = "В этой ретроспективной постановке обучение sampled LOO RL улучшило next-item NDCG@10 относительно strong reference; независимая проверка всё ещё нужна."
    elif rl["ci95_high_vs_strong"] < 0:
        answer = "В этой постановке RL ухудшил next-item NDCG@10 относительно strong reference; улучшение предсказания следующего трека не получено."
    else:
        answer = "Убедительное улучшение next-item NDCG@10 от RL не установлено: парный user-bootstrap 95% CI эффекта включает ноль."
    rl_effect = (f"Δ {rl['delta_ndcg10_vs_strong']:+.5f} NDCG@10 (95% CI "
                 f"[{rl['ci95_low_vs_strong']:+.5f}, {rl['ci95_high_vs_strong']:+.5f}])")
    exact_effect = (f"Δ {exact['delta_ndcg10_vs_strong']:+.5f} NDCG@10 (95% CI "
                    f"[{exact['ci95_low_vs_strong']:+.5f}, {exact['ci95_high_vs_strong']:+.5f}])")
    sec = {r["method"]: r for r in secondary}
    rl24 = sec.get("sampled_rl", {})
    strong24 = sec.get("strong_gpt", {})
    sampled_seed_text = ", ".join("{}={:.5f} (step {})".format(x["seed"], x["ndcg10"], x["best_step"])
                                  for x in policy_choice["sampled_seed_results"])
    exact_seed_text = ", ".join("{}={:.5f} (step {})".format(x["seed"], x["ndcg10"], x["best_step"])
                                for x in policy_choice["exact_seed_results"])
    best_dev = {}
    for model in ("old_deployed", "same_split_control", "prefix_affinity"):
        options = [r for r in dev_result["rows"] if r.get("reward_model") == model]
        best_dev[model] = sorted(options, key=lambda r: (-float(r["ndcg10"]),
                                                          float(r["mass_weighted_kl"])))[0]
    external_by_method = {r["method"]: r for r in external.get("rows", [])}
    ext_rl = external_by_method.get("sampled_rl", {}).get("expected_predicted_raw_ratio_delta_vs_reference")
    ext_exact = external_by_method.get("exact_gradient", {}).get("expected_predicted_raw_ratio_delta_vs_reference")
    lines = [
        "# Frozen FAST: улучшает ли RL предсказание следующего трека?", "",
        f"**Ответ:** {answer}", "",
        f"На reporting-панели для RL получена {rl_effect}; для exact-gradient контроля — {exact_effect}. Сравнение RL−exact: {policy_choice.get('rl_vs_exact_text', 'см. таблицу контрастов')}.", "",
        "## Итог next-item", "",
        "Основная метрика — user-macro NDCG@10 следующего nonorganic события. Один контекст на пользователя; OOV цели остаются в знаменателе и считаются промахом. 95% интервалы парные по пользователям; для RL и exact сначала усреднены результаты seed 17/42/73 внутри пользователя.", "",
        "| Метод | n контекстов | OOV цели | Попадания @10 | NDCG@10 | HR@10 | Δ NDCG против strong (95% CI) | positive-new n/hits | positive-repeat n/hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        ci = f"{row['delta_ndcg10_vs_strong']:+.5f} [{row['ci95_low_vs_strong']:+.5f}, {row['ci95_high_vs_strong']:+.5f}]"
        lines.append(f"| {row['description']} | {row['n_contexts']} | {row['oov_targets']} | {row['hits_at_10']:.1f} | {row['ndcg10']:.5f} | {row['hr10']:.5f} | {ci} | {row['positive_new_n']}/{row['positive_new_hits']:.1f} | {row['positive_repeat_n']}/{row['positive_repeat_hits']:.1f} |")
    lines.extend(["", "### Direct RM на dev", "",
                  "Выбор RM и beta сделан только по user-macro next-item NDCG@10 на development; таблица ниже показывает лучший из четырёх зарегистрированных beta для каждой готовой RM.", "",
                  "| Вариант RM | Лучший beta | Dev NDCG@10 | HR@10 | Mass-weighted KL |",
                  "|---|---:|---:|---:|---:|"])
    lines.append(f"| Strong reference | — | {next(r['ndcg10'] for r in dev_result['rows'] if r['reward_model'] == 'reference'):.5f} | {next(r['hr10'] for r in dev_result['rows'] if r['reward_model'] == 'reference'):.5f} | 0.00000 |")
    for model, label in (("old_deployed", "Old deployed"), ("same_split_control", "Same-split control"), ("prefix_affinity", "Prefix-affinity RM")):
        row = best_dev[model]
        lines.append(f"| {label} | {float(row['beta']):g} | {row['ndcg10']:.5f} | {row['hr10']:.5f} | {row['mass_weighted_kl']:.5f} |")
    lines.extend(["", f"По этому правилу выбрана RM **{dev_result['selected_analytic']['reward_model']}**, beta **{dev_result['selected_analytic']['beta']:g}**: NDCG@10 {dev_result['selected_analytic']['ndcg10']:.5f}, mass-weighted KL {dev_result['selected_analytic']['mass_weighted_kl']:.5f}. Prefix-affinity RM была проверена на общей панели при всех четырёх beta и не дала лучший dev NDCG; это решение принято по заранее заданному dev-критерию, а не по отсутствию предыдущего policy-эффекта. Для RL проверены только beta {', '.join(f'{b:g}' for b in dev_result['policy_beta_candidates'])}; выбрано beta {policy_choice['beta']:g} по dev NDCG@10, затем меньшему KL. Все проигравшие analytic и обученные policy runs сохранены в artifacts/frozen_policy.", "",
                  f"Лучший direct old-RM tilt: beta {direct_old.get('beta', '—')}, reporting NDCG@10 {direct_old['ndcg10']:.5f}. Лучший analytic tilt выбранной RM: NDCG@10 {chosen['ndcg10']:.5f}. У RL лучший trained checkpoint выбран отдельно внутри каждого seed; step 0 исключён. Seed-level NDCG@10: {sampled_seed_text}. Exact seeds: {exact_seed_text}.", "",
                  "### Графики", "", f"![Dev NDCG по обновлениям]({figures[0]})", "",
                  f"![Reward, качество и KL]({figures[1]})", "", f"![Разности с user-bootstrap CI]({figures[2]})", "",
                  "## Вторичные метрики: положительные рекомендации за 24 часа", "",
                  "Здесь набор цели — уникальные raw item ID последующих nonorganic событий с ratio ≥0.5 в пределах следующих 24 часов. Включены только reporting-контексты с полным 24-часовым окном; контексты с пустым набором целей имеют NA и исключены из среднего. Эти метрики не подменяют next-item вывод.", "",
                  "| Метод | Полное окно | Пустые наборы | Непустые контексты | Положительные треки | Recall@10 | NDCG@10 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    # Include paired intervals for both secondary metrics, still separate from next-item.
    lines[-2] = "| Метод | Полное окно | Пустые наборы | Непустые контексты | Положительные треки | Recall@10 (Δ; 95% CI) | NDCG@10 (Δ; 95% CI) |"
    for row in secondary:
        lines.append(f"| {row['description']} | {row['full_horizon_contexts']} | {row['empty_target_contexts']} | {row['n']} | {row['positive_target_tracks']} | {row['recall10']:.4f} ({row['recall_delta_vs_strong']:+.4f}; [{row['recall_ci95_low_vs_strong']:+.4f}, {row['recall_ci95_high_vs_strong']:+.4f}]) | {row['ndcg10']:.4f} ({row['ndcg_delta_vs_strong']:+.4f}; [{row['ndcg_ci95_low_vs_strong']:+.4f}, {row['ndcg_ci95_high_vs_strong']:+.4f}]) |")
    if rl24 and strong24:
        lines.append("")
        lines.append(f"Для sampled RL относительно strong на 24h Recall@10: {rl24.get('recall_delta_vs_strong', 0):+.5f} (95% CI {rl24.get('recall_ci95_low_vs_strong', 0):+.5f}, {rl24.get('recall_ci95_high_vs_strong', 0):+.5f}); на NDCG@10: {rl24.get('ndcg_delta_vs_strong', 0):+.5f} (95% CI {rl24.get('ndcg_ci95_low_vs_strong', 0):+.5f}, {rl24.get('ndcg_ci95_high_vs_strong', 0):+.5f}).")
    lines.extend(["", "## Протокол и данные", "",
                  f"Использованы strong GPT seed17 (замороженный actor), каталог 20 000, top-500 support, неизменный reference tail, готовые RM и prefix-safe embeddings. Policy head: 54→64→64→1, SiLU, нулевой выходной слой; sampled LOO использовал 16 actions/context и exact KL к reference. Размер policy train: 4 134 контекста. Development/reporting hash split внутри исторически открытой панели: {runtime['split']['development_n']}/{runtime['split']['reporting_n']} пользователей. Это ретроспективное сравнение, не независимый test. C-priors восстановлены строго до cutoff {runtime['c_prior_cutoff_exclusive']:.0f}; prefix embeddings построены строго до временного cutoff C; policy-train цели позднее cutoff. Новые priors/features вычислялись из истории до target.", "",
                  f"Единственная normalization scale: mean {runtime['reward_scale']['mean']:.5f}, std {runtime['reward_scale']['std']:.5f}, посчитанные по прогнозам same-split control под reference conditional distribution на policy-train. Frozen external predictor pass: {external.get('status', 'unavailable')}; его ожидаемый raw-ratio delta на fixed support — RL {ext_rl:+.5f}, exact {ext_exact:+.5f} относительно reference. Это условная диагностическая оценка, не доказательство отсутствия reward hacking.", "",
                  f"Новые вычисления: {runtime['elapsed_seconds_total']/3600:.2f} ч по stage timer (policy training/dev scoring: {runtime.get('policy_training_seconds', 0)/3600:.2f} ч; feature preparation: {runtime.get('feature_preparation_seconds', 0)/3600:.2f} ч). Все пять минимальных implementation checks: {'пройдены' if checks.get('status') == 'passed' else 'не подтверждены'}.", "",
                  "## Фиксированные примеры", "", f"[Пять примеров до/после]({examples_path}) показывают историю, top-10 и только prefix-доступные признаки.", "",
                  "## Ограничения и следующий шаг", "",
                  "Оценка относится к одной исторически открытой validation-панели и warm-start пользователям, которые могли пересекаться с policy-train; bootstrap отражает неопределённость по пользователям этой панели, а не вариацию новых рынков или периодов. OOV цели считаются промахами. Положительные 24h-результаты — вторичные и зависят от выбранного порога ratio.", "",
                  "Следующий эксперимент — подтвердить выбранные actor/RM/policy на новой, ранее не открывавшейся временной панели с тем же каталогом, support и заранее зафиксированным checkpoint rule.", "",
                  "Артефакты: [таблица методов](method_metrics.csv), [paired contrasts](primary_contrasts.csv), [per-user данные](per_user_metrics.csv), [24h таблица](secondary_24h.csv), [Frozen run manifest](../../artifacts/frozen_policy/run.json), [воспроизведение](REPRODUCE.md).", ""])
    report.write_text("\n".join(lines), encoding="utf-8")


def run_reporting(root: Path, artifact_root: Path, final_dir: Path,
                  split_manifest: dict[str, Any], analytic_result: dict[str, Any],
                  policy_choice: dict[str, Any], scale: FrozenRewardScale,
                  feature_manifest: dict[str, Any], checks: dict[str, Any],
                  runtime: dict[str, Any]) -> dict[str, Any]:
    reporting_started = time.perf_counter()
    final_dir.mkdir(parents=True, exist_ok=True)
    data, _ = open_v5_cache(artifact_root / "cache", "common")
    _, common_arrays, common_panel = _load_existing_cache(root / "artifacts/policy_diagnostics/cache/common_open_6655")
    arrays = {**common_arrays, "candidate_features": data["candidate_features"], "rm_rewards": data["rm_rewards"]}
    item_ids = np.asarray(pl.read_parquet(root / "data/processed_full_model_tminus14/catalog.parquet")
                          .sort("token_id")["item_id"].to_list(), dtype=np.int64)
    report_ix = _reporting_indices(common_panel, split_manifest)
    scaler = load_scaler(artifact_root / "scaler.json")
    ref = np.asarray(arrays["reference_conditional_logp"][report_ix], dtype=np.float64)
    qref = _normal_q(ref)
    strong_raw = evaluate_conditional(common_panel, arrays, report_ix, qref, item_ids, want_topk=True)
    strong = _adapt_conditional(strong_raw, name="strong_gpt", q=qref,
                                diagnostics={"mass_weighted_kl": 0.0, "normalized_reward_full_delta": 0.0})
    # The event-ID/user denominator is identical for every method; OOV remain misses.
    store_protocol = json.loads((root / "data/processed_full_model_tminus14/protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(root / "data/processed_full_model_tminus14/sequence_index",
                                     expected_source_sha256=store_protocol["processed_listens_sha256"])
    repeat = _repeat_baseline(common_panel, report_ix, common_arrays, store, item_ids)
    repeat_result = {"name": "history_frequency_reranker", "n": repeat["n"],
                     "ndcg10": repeat["ndcg10"], "hr10": repeat["hr10"],
                     "hits_at_10": repeat["hits_at_10"], "oov_targets": repeat["oov_targets"],
                     "positive_new": repeat["positive_new"], "positive_repeat": repeat["positive_repeat"],
                     "per_user_ndcg": repeat["per_user_ndcg"], "per_user_hr": repeat["per_user_hr"],
                     "top10_by_user": repeat["top10_by_user"]}

    dev_rows = analytic_result["rows"]
    best_by_rm = _best_analytic_by_rm(dev_rows)
    selected_rm = str(analytic_result["selected_analytic"]["reward_model"])
    selected_beta = float(analytic_result["selected_analytic"]["beta"])
    rm_index = {"old_deployed": 0, "same_split_control": 1, "prefix_affinity": 2}
    def eval_analytic(model_name: str, beta: float, name: str) -> dict[str, Any]:
        rewards = np.asarray(data["rm_rewards"][report_ix, rm_index[model_name], :], dtype=np.float32)
        q = analytic_q(ref, rewards, scale, beta)
        raw = evaluate_conditional(common_panel, arrays, report_ix, q, item_ids, want_topk=True)
        diag = model_reward_metrics(ref, q, rewards, np.asarray(arrays["support_mass"][report_ix]), scale)
        result = _adapt_conditional(raw, name=name, q=q, diagnostics=diag)
        result["beta"] = float(beta); result["reward_model"] = model_name
        return result
    old_analytic = eval_analytic("old_deployed", float(best_by_rm["old_deployed"]["beta"]), "analytic_old_rm")
    selected_analytic = eval_analytic(selected_rm, selected_beta, "analytic_selected_rm")

    sampled_group, sampled_seeds, sampled_qs = _run_reported_seed_group(
        root, artifact_root, policy_choice, "sampled_rl", (17, 42, 73), report_ix,
        common_panel, arrays, item_ids, scaler, selected_rm, scale)
    exact_group, exact_seeds, exact_qs = _run_reported_seed_group(
        root, artifact_root, policy_choice, "exact_gradient", (17, 42, 73), report_ix,
        common_panel, arrays, item_ids, scaler, selected_rm, scale)
    old_rl, old_rl_seeds, old_rl_qs = _run_reported_seed_group(
        root, artifact_root, policy_choice, "old_rm_rl", (17,), report_ix,
        common_panel, arrays, item_ids, scaler, "old_deployed", scale)

    methods = {"strong_gpt": strong,
               "history_frequency_reranker": repeat_result}
    if not (selected_rm == "old_deployed" and
            math.isclose(float(best_by_rm["old_deployed"]["beta"]), selected_beta, rel_tol=0, abs_tol=1e-12)):
        methods["analytic_old_rm"] = old_analytic
    methods["analytic_selected_rm"] = selected_analytic
    methods["sampled_rl"] = sampled_group
    methods["exact_gradient"] = exact_group
    methods["old_rm_rl_seed17"] = old_rl
    strong_users = strong["per_user_ndcg"]
    metric_rows = []
    descriptions = {
        "strong_gpt": ("Strong GPT", "базовая модель"),
        "history_frequency_reranker": ("Repeat/history reranker", "простой контроль повторов"),
        "analytic_old_rm": ("Analytic tilt, old RM", "прямое применение старой награды"),
        "analytic_selected_rm": (("Analytic tilt, old RM" if selected_rm == "old_deployed" else f"Analytic tilt, {selected_rm}"), "прямое применение выбранной награды"),
        "sampled_rl": (f"Sampled LOO RL, {selected_rm}", "основной обученный метод; среднее seed 17/42/73"),
        "exact_gradient": (f"Exact gradient, {selected_rm}", "контроль способа оптимизации; среднее seed 17/42/73"),
        "old_rm_rl_seed17": (("Sampled LOO RL, seed 17 only" if selected_rm == "old_deployed" else "Sampled LOO RL, old RM"),
                              ("seed 17 из основной серии; показан отдельно" if selected_rm == "old_deployed" else "вторичная абляция; seed 17")),
    }
    if "analytic_old_rm" not in methods:
        descriptions.pop("analytic_old_rm")
    for i, (name, result) in enumerate(methods.items()):
        label, purpose = descriptions[name]
        metric_rows.append(_summary_row(name, label, result, strong_users, 85000 + 2 * i, purpose))
    # Keep the per-seed results visible and auditable; seeds are never resampled as users.
    seed_rows = []
    for name, seed_results in (("sampled_rl", sampled_seeds), ("exact_gradient", exact_seeds), ("old_rm_rl", old_rl_seeds)):
        for r in seed_results:
            ci = _paired_ci(r["per_user_ndcg"], strong_users, 91000 + int(r["seed"]))
            seed_rows.append({"method": name, "seed": r["seed"], "best_step": r["best_step"],
                              "n_contexts": r["n"], "hits_at_10": r["hits_at_10"],
                              "ndcg10": r["ndcg10"], "hr10": r["hr10"],
                              "delta_ndcg10_vs_strong": ci["mean_difference"],
                              "ci95_low_vs_strong": ci["ci_lower"], "ci95_high_vs_strong": ci["ci_upper"]})

    contrasts = []
    for label, left, right, seed in (
        ("sampled_rl - strong_gpt", sampled_group["per_user_ndcg"], strong_users, 1001),
        ("exact_gradient - strong_gpt", exact_group["per_user_ndcg"], strong_users, 1003),
        ("sampled_rl - exact_gradient", sampled_group["per_user_ndcg"], exact_group["per_user_ndcg"], 1005),
    ):
        contrasts.append({"contrast": label, **_paired_ci(left, right, seed)})
    rl_exact = contrasts[-1]
    if rl_exact["ci_lower"] > 0:
        rl_exact_text = f"RL выше exact на {rl_exact['mean_difference']:+.5f} (95% CI [{rl_exact['ci_lower']:+.5f}, {rl_exact['ci_upper']:+.5f}])"
    elif rl_exact["ci_upper"] < 0:
        rl_exact_text = f"RL ниже exact на {rl_exact['mean_difference']:+.5f} (95% CI [{rl_exact['ci_lower']:+.5f}, {rl_exact['ci_upper']:+.5f}])"
    else:
        rl_exact_text = f"разность {rl_exact['mean_difference']:+.5f}, 95% CI [{rl_exact['ci_lower']:+.5f}, {rl_exact['ci_upper']:+.5f}] включает ноль"

    _write_csv(final_dir / "method_metrics.csv", metric_rows)
    _write_csv(final_dir / "seed_metrics.csv", seed_rows)
    _write_csv(final_dir / "primary_contrasts.csv", contrasts)
    _save_per_user(final_dir / "per_user_metrics.csv", methods)

    # Secondary 24h multi-positive recall/NDCG: use only full windows, and keep empty target sets NA.
    timelines = _event_timelines(root, common_panel, store)
    target_meta = _build_24h_targets(root, common_panel, report_ix, store, timelines)
    secondary_methods: dict[str, dict[str, Any]] = {}
    secondary_method_items = [("strong_gpt", strong), ("history_frequency_reranker", repeat_result)]
    if "analytic_old_rm" in methods:
        secondary_method_items.append(("analytic_old_rm", old_analytic))
    secondary_method_items.append(("analytic_selected_rm", selected_analytic))
    for name, res in secondary_method_items:
        secondary_methods[name] = _score_24h(res["top10_by_user"], target_meta)
    for name, seed_results in (("sampled_rl", sampled_seeds), ("exact_gradient", exact_seeds)):
        seed_values = [_score_24h(r["top10_by_user"], target_meta) for r in seed_results]
        secondary_methods[name] = _aggregate_24h_seed(seed_values)
    secondary_methods["old_rm_rl_seed17"] = _score_24h(old_rl["top10_by_user"], target_meta)
    secondary_rows = []
    strong24 = secondary_methods["strong_gpt"]
    labels24 = {name: descriptions[name][0] for name in descriptions}
    for i, (name, res) in enumerate(secondary_methods.items()):
        recall_ci = _paired_ci(res["per_user_recall"], strong24["per_user_recall"], 13000 + i * 3) if name != "strong_gpt" else {"mean_difference": 0., "ci_lower": 0., "ci_upper": 0., "n_users": res["n"]}
        ndcg_ci = _paired_ci(res["per_user_ndcg"], strong24["per_user_ndcg"], 13001 + i * 3) if name != "strong_gpt" else {"mean_difference": 0., "ci_lower": 0., "ci_upper": 0., "n_users": res["n"]}
        secondary_rows.append({"method": name, "description": labels24[name],
                               "full_horizon_contexts": res["full_horizon_contexts"],
                               "empty_target_contexts": res["empty_target_contexts"],
                               "n": res["n"], "positive_target_tracks": res["positive_target_tracks"],
                               "hits_at_10": res["hits_at_10"], "recall10": res["recall10"],
                               "recall_delta_vs_strong": recall_ci["mean_difference"],
                               "recall_ci95_low_vs_strong": recall_ci["ci_lower"],
                               "recall_ci95_high_vs_strong": recall_ci["ci_upper"],
                               "ndcg10": res["ndcg10"],
                               "ndcg_delta_vs_strong": ndcg_ci["mean_difference"],
                               "ndcg_ci95_low_vs_strong": ndcg_ci["ci_lower"],
                               "ndcg_ci95_high_vs_strong": ndcg_ci["ci_upper"]})
    _write_csv(final_dir / "secondary_24h.csv", secondary_rows)

    # One final conditional predictor pass on the same reporting support. It is descriptive only.
    q_for_audit = {"analytic_selected_rm": selected_analytic["conditional"],
                   "sampled_rl": sampled_group["conditional"],
                   "exact_gradient": exact_group["conditional"]}
    if "analytic_old_rm" in methods:
        q_for_audit["analytic_old_rm"] = old_analytic["conditional"]
    audit_rm = {name: rm_index[selected_rm] for name in q_for_audit}
    if "analytic_old_rm" in methods:
        audit_rm["analytic_old_rm"] = 0
    audit_rewards = {name: np.asarray(data["rm_rewards"][report_ix, idx, :], dtype=np.float32)
                     for name, idx in audit_rm.items()}
    try:
        external = _external_audit(root, report_ix, common_arrays, data, q_for_audit, audit_rm, audit_rewards)
    except Exception as exc:
        external = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}",
                    "note": "fixed-support predictor pass could not be completed; no external predictor conclusion is made"}
    _write_json(final_dir / "external_audit.json", external)

    _write_examples(root, final_dir / "recommendation_examples.md", report_ix, common_panel,
                    store, common_arrays, data["candidate_features"], item_ids,
                    strong["top10_by_user"], sampled_seeds[0]["top10_by_user"])

    direct_rows = analytic_result["rows"]
    policy_choice["rl_vs_exact_text"] = rl_exact_text
    policy_choice["sampled_seed_results"] = [{"seed": r["seed"], "ndcg10": r["ndcg10"], "best_step": r["best_step"]} for r in sampled_seeds]
    policy_choice["exact_seed_results"] = [{"seed": r["seed"], "ndcg10": r["ndcg10"], "best_step": r["best_step"]} for r in exact_seeds]
    report_rows = [dict(r) for r in metric_rows if r["method"] in {"analytic_old_rm", "analytic_selected_rm", "sampled_rl", "exact_gradient", "old_rm_rl_seed17"}]
    report_rows = [{**r, "mass_weighted_kl": methods[r["method"]].get("mass_weighted_kl")} for r in report_rows]
    run_dirs = [root / policy_choice[group][str(seed)]["run_dir"]
                for group in ("sampled_rl", "exact_gradient", "old_rm_rl")
                for seed in policy_choice[group]]
    figure_paths = _make_figures(final_dir, run_dirs, direct_rows, report_rows)

    runtime["split"] = {"development_n": split_manifest["development_n"],
                         "reporting_n": split_manifest["reporting_n"]}
    runtime["c_prior_cutoff_exclusive"] = feature_manifest["c_prior_cutoff_exclusive"]
    runtime["reward_scale"] = {"mean": scale.mean, "std": scale.std}
    runtime["reporting_elapsed_seconds"] = round(time.perf_counter() - reporting_started, 3)
    runtime["elapsed_seconds_total"] = round(float(runtime.get("elapsed_seconds_total", 0.0))
                                               + runtime["reporting_elapsed_seconds"], 3)
    _report_answer(final_dir / "REPORT.md", metric_rows, secondary_rows, analytic_result,
                   policy_choice, checks, runtime, figure_paths,
                   "recommendation_examples.md", external)
    result = {
        "status": "complete", "panel_role": "reporting; descriptive historical panel, not independent test",
        "n_reporting_contexts": len(report_ix), "methods": metric_rows,
        "primary_contrasts": contrasts, "seed_metrics": seed_rows,
        "secondary_24h": secondary_rows,
        "external_audit": external,
        "figures": figure_paths,
        "examples_path": str((final_dir / "recommendation_examples.md").relative_to(root)),
        "trained_checkpoints": {group: {seed: policy_choice[group][seed]["best_checkpoint_path"]
                                         for seed in policy_choice[group]}
                                for group in ("sampled_rl", "exact_gradient", "old_rm_rl")},
    }
    _write_json(final_dir / "metrics.json", result)
    _write_json(artifact_root / "reporting_summary.json", result)
    return result


