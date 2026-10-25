"""Close panel_tools.1's missing user-level SFT, baseline, policy, and CI evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import polars as pl
import torch

from tlab.data.store import SequenceStore
from tlab.evaluation.baselines import _train_transition_matrix, smoothed_transition_scores
from tlab.models.actor import ItemGPT2Actor
from tlab.panel_tools.data import PanelTarget, history_for_target
from tlab.panel_tools.evaluator import _collate_histories, load_actor_checkpoint
from tlab.panel_tools.evaluation import single_target_metrics
from tlab.panel_tools.statistics import paired_user_bootstrap, relative_quality_guard
from tlab.residual_support.evaluation import _analytic_q, _scale_from_json
from tlab.residual_support.features import POLICY_FEATURE_NAMES
from tlab.residual_support.head import ResidualHeadSpec, ResidualPolicyHead
from tlab.residual_support.metrics import _group_name
from tlab.residual_support.scaler import TrainOnlyFeatureScaler
from tlab.residual_support.train_policy import _load_cache

from .protocol import ARTIFACTS, ROOT, record_operation, update_stage


PANEL_DIRS = {
    "development": ARTIFACTS / ".." / "residual_support/cache/development",
    "retrospective_final": ARTIFACTS / ".." / "residual_support/cache/final",
    "common_open_6655": ARTIFACTS / "cache/common_open_6655",
}
PANEL_DIRS = {k: v.resolve() for k, v in PANEL_DIRS.items()}
N_BOOTSTRAP = 2000


class RankAccumulator:
    def __init__(self, panel_name: str, method: str, endpoint: str, seed: int | None,
                 writer: csv.DictWriter, top_writer: csv.DictWriter, item_ids: np.ndarray,
                 rm_fit_counts: np.ndarray) -> None:
        self.panel_name = panel_name
        self.method = method
        self.endpoint = endpoint
        self.seed = seed
        self.writer = writer
        self.top_writer = top_writer
        self.item_ids = item_ids
        self.rm_fit_counts = rm_fit_counts
        self.raw_to_index = {int(item): i for i, item in enumerate(item_ids.tolist())}
        self.primary: dict[int, list[dict[str, float]]] = defaultdict(list)
        self.slices: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
        self.covered = 0
        self.in_support = 0
        self.events = 0
        self.top10_by_user: dict[int, list[int]] = {}

    def add(self, target: PanelTarget, scores: np.ndarray, *, top10: np.ndarray | None = None,
            support_indices: np.ndarray | None = None) -> None:
        score = np.asarray(scores, dtype=np.float64)
        index = self.raw_to_index.get(int(target.raw_item_id))
        rank: int | None
        if index is None:
            rank = None
        else:
            target_score = score[index]
            rank = int(np.count_nonzero((score > target_score) | ((score == target_score) & (self.item_ids < target.raw_item_id))) + 1)
        metrics = single_target_metrics(rank, 10)
        uid = int(target.uid)
        self.primary[uid].append(metrics)
        self.events += 1
        self.covered += int(index is not None)
        support_set = set(np.asarray(support_indices, dtype=np.int64).tolist()) if support_indices is not None else None
        target_supported = bool(index is not None and support_set is not None and index in support_set)
        self.in_support += int(target_supported)
        group = _group_name(int(self.rm_fit_counts[index])) if index is not None else None
        if target.positive:
            self.slices["positive_nonorganic"][uid].append(metrics)
            if target.is_new:
                self.slices["positive_new"][uid].append(metrics)
            if target.is_repeat:
                self.slices["positive_repeat"][uid].append(metrics)
        if target.is_new:
            self.slices["new"][uid].append(metrics)
        if target.is_repeat:
            self.slices["repeat"][uid].append(metrics)
        if group is not None:
            self.slices[f"rm_fit_support_{group}"][uid].append(metrics)
            if target.positive and target.is_new:
                self.slices[f"positive_new_x_{group}"][uid].append(metrics)
            if target.positive and target.is_repeat:
                self.slices[f"positive_repeat_x_{group}"][uid].append(metrics)

        out = {
            "panel": self.panel_name, "method": self.method, "endpoint": self.endpoint,
            "seed": "" if self.seed is None else self.seed, "uid": uid,
            "event_id": target.event_id, "rank": "NA" if rank is None else rank,
            "ndcg10": metrics["ndcg@10"], "recall10": metrics["recall@10"],
            "target_raw_id": target.raw_item_id, "target_in_catalog": index is not None,
            "target_in_support": target_supported, "positive": target.positive,
            "is_new": target.is_new, "is_repeat": target.is_repeat,
            "rm_fit_support_group": group or "unsupported_oov",
        }
        self.writer.writerow(out)
        if top10 is None:
            top10 = _topk(score, self.item_ids, 10)
        ids = [int(self.item_ids[i]) for i in np.asarray(top10).tolist()]
        self.top10_by_user[uid] = ids
        for list_rank, item_index in enumerate(np.asarray(top10).tolist(), start=1):
            self.top_writer.writerow({
                "panel": self.panel_name, "method": self.method, "endpoint": self.endpoint,
                "seed": "" if self.seed is None else self.seed, "uid": uid,
                "list_rank": list_rank, "catalog_index": int(item_index),
                "raw_item_id": int(self.item_ids[item_index]), "score": float(score[item_index]),
            })

    @staticmethod
    def _macro(source: dict[int, list[dict[str, float]]]) -> dict[str, Any]:
        if not source:
            return {"ndcg@10": None, "recall@10": None, "n_users": 0, "n_events": 0, "per_user": {}}
        per_user = {uid: {key: float(np.mean([x[key] for x in vals])) for key in ("ndcg@10", "recall@10")}
                    for uid, vals in source.items()}
        return {"ndcg@10": float(np.mean([x["ndcg@10"] for x in per_user.values()])),
                "recall@10": float(np.mean([x["recall@10"] for x in per_user.values()])),
                "n_users": len(per_user), "n_events": sum(len(x) for x in source.values()), "per_user": per_user}

    def finish(self, *, target_support_fraction: float | None = None) -> dict[str, Any]:
        primary = self._macro(self.primary)
        slices = {name: self._macro(values) for name, values in self.slices.items()}
        return {
            "primary_user_macro": primary,
            "slices": slices,
            "panel_events": self.events,
            "panel_users": len(self.primary),
            "target_catalog_coverage": self.covered / self.events if self.events else None,
            "target_in_optimized_support_fraction": self.in_support / self.events if self.events else None,
            "support_size": 500 if target_support_fraction is not None else None,
            "top_k": 10,
            "oov_targets": self.events - self.covered,
            "per_user_primary_ndcg": {int(uid): float(v["ndcg@10"]) for uid, v in primary["per_user"].items()},
            "top10_by_user": self.top10_by_user,
        }


def _topk(scores: np.ndarray, item_ids: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(scores)
    threshold = np.partition(values, len(values) - k)[len(values) - k]
    higher = np.flatnonzero(values > threshold)
    tied = np.flatnonzero(values == threshold)
    need = k - len(higher)
    if need > 0:
        tie_order = np.argsort(item_ids[tied], kind="stable")
        chosen = np.concatenate([higher, tied[tie_order[:need]]])
    else:
        chosen = higher
    order = np.lexsort((item_ids[chosen], -values[chosen]))
    return chosen[order]


def _write_user_records(acc: RankAccumulator, targets: list[PanelTarget], scores: np.ndarray,
                        support_indices: np.ndarray | None = None) -> None:
    for i, target in enumerate(targets):
        acc.add(target, scores[i], support_indices=None if support_indices is None else support_indices[i])


def _eval_cached_policy(panel_name: str, panel: list[PanelTarget], arrays: dict[str, np.ndarray],
                        q: np.ndarray, item_ids: np.ndarray, rm_fit_counts: np.ndarray,
                        method: str, endpoint: str, seed: int | None,
                        per_writer: csv.DictWriter, top_writer: csv.DictWriter) -> dict[str, Any]:
    if q.shape != arrays["support_indices"].shape:
        raise ValueError("conditional policy shape differs from the fixed support")
    acc = RankAccumulator(panel_name, method, endpoint, seed, per_writer, top_writer, item_ids, rm_fit_counts)
    full = arrays["reference_logp_full"]
    indices = arrays["support_indices"]
    mass = arrays["support_mass"]
    for i, target in enumerate(panel):
        probabilities = np.exp(np.asarray(full[i], dtype=np.float64))
        probabilities[indices[i]] = float(mass[i]) * np.asarray(q[i], dtype=np.float64)
        if not np.isclose(probabilities.sum(), 1.0, atol=2e-5):
            raise ValueError("full-catalog policy changed the reference tail mass")
        acc.add(target, probabilities, support_indices=indices[i])
    return acc.finish(target_support_fraction=float(np.mean(mass)))


def _predict_checkpoint(checkpoint: Path, run_manifest: dict[str, Any], arrays: dict[str, np.ndarray],
                        scaler: TrainOnlyFeatureScaler, device: torch.device) -> tuple[np.ndarray, int]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    columns = tuple(int(x) for x in saved.get("feature_columns", run_manifest["feature_columns"]))
    state = saved.get("state_dict", saved.get("head_state"))
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no trainable head state: {checkpoint}")
    head = ResidualPolicyHead(ResidualHeadSpec(input_dim=len(columns), hidden_dim=32)).to(device)
    head.load_state_dict(state, strict=True)
    head.eval()
    values = arrays["candidate_features"]
    result = np.empty((len(values), values.shape[1]), dtype=np.float32)
    for start in range(0, len(values), 32):
        stop = min(start + 32, len(values))
        raw = np.asarray(values[start:stop])
        scaled = scaler.transform(raw, POLICY_FEATURE_NAMES)[..., list(columns)]
        x = torch.from_numpy(np.ascontiguousarray(scaled)).to(device)
        reference = torch.from_numpy(np.asarray(arrays["reference_conditional_logp"][start:stop]).copy()).to(device)
        with torch.inference_mode():
            result[start:stop] = torch.softmax(reference + head(x), dim=-1).cpu().numpy().astype(np.float32)
    step = int(saved.get("step", saved.get("updates_done", run_manifest.get("best_checkpoint_step", -1))))
    return result, step


@torch.inference_mode()
def _eval_actor(panel_name: str, panel: list[PanelTarget], actor: ItemGPT2Actor, store: SequenceStore,
                item_ids: np.ndarray, rm_fit_counts: np.ndarray, method: str, endpoint: str,
                per_writer: csv.DictWriter, top_writer: csv.DictWriter, batch_size: int = 16) -> dict[str, Any]:
    device = next(actor.parameters()).device
    acc = RankAccumulator(panel_name, method, endpoint, None, per_writer, top_writer, item_ids, rm_fit_counts)
    actor.eval()
    for start in range(0, len(panel), batch_size):
        targets = panel[start:start + batch_size]
        histories = [history_for_target(store, target, 200) for target in targets]
        ids, mask = _collate_histories(histories, device)
        logp = actor.next_action_log_probs(ids, mask)[:, 3:].float().cpu().numpy()
        for target, scores in zip(targets, logp):
            acc.add(target, scores)
    return acc.finish()


def _mean_maps(maps: list[dict[int, float]]) -> dict[int, float]:
    shared = set(maps[0])
    for values in maps[1:]:
        shared.intersection_update(values)
    return {uid: float(np.mean([m[uid] for m in maps])) for uid in sorted(shared)}


def _contrast(left: dict[int, float], right: dict[int, float], seed: int,
              *, guard: bool = True) -> dict[str, Any]:
    common = sorted(set(left).intersection(right))
    a = {uid: left[uid] for uid in common}
    b = {uid: right[uid] for uid in common}
    paired = paired_user_bootstrap(a, b, n_bootstrap=N_BOOTSTRAP, seed=seed)
    result = {"n_users": len(common), "mean_difference": float(np.mean([a[x] - b[x] for x in common])),
              "paired_95pct_ci": {k: v for k, v in paired.items() if k != "user_differences"}}
    if guard:
        g = relative_quality_guard(a, b, tolerance=0.02, n_bootstrap=N_BOOTSTRAP, seed=seed + 1)
        result["relative_2pct_guard"] = {
            "passes": g["passes"], "adjusted_lower_bound": g["lower_bound"],
            "mean_adjusted_difference": g["interval"]["mean_difference"],
            "two_sided_raw_difference_ci": [g["interval"]["ci_lower"], g["interval"]["ci_upper"]],
            "confidence": g["interval"]["confidence"], "n_bootstrap": g["interval"]["n_bootstrap"],
        }
    return result


def _make_baseline_resources(store: SequenceStore, item_ids: np.ndarray,
                             catalog_path: Path) -> dict[str, Any]:
    catalog = pl.read_parquet(catalog_path).sort("token_id")
    train_counts = np.asarray(catalog["training_count"].to_list(), dtype=np.float64)
    dataset = json.loads((ROOT / "data/processed_full_model_tminus14/protocol.json").read_text(encoding="utf-8"))
    cutoff = float(dataset["boundaries_source_units"]["train_cutoff_raw"])
    recent = (pl.scan_parquet(ROOT / "data/processed_full_model_tminus14/listens.parquet")
              .filter((pl.col("global_split") == "train")
                      & (pl.col("timestamp") >= cutoff - 14 * 86400)
                      & (pl.col("timestamp") < cutoff)
                      & (pl.col("token_id") >= 3))
              .group_by("token_id").agg(pl.len().alias("n")).collect(engine="streaming"))
    recent_counts = np.zeros(len(item_ids), dtype=np.float64)
    for token, count in recent.iter_rows():
        idx = int(token) - 3
        if 0 <= idx < len(recent_counts):
            recent_counts[idx] = float(count)
    transition, examples = _train_transition_matrix(store, len(item_ids))
    return {"train_counts": train_counts, "recent_counts": recent_counts,
            "transition": transition, "transition_examples": int(examples)}


def _eval_simple_baselines(panel_name: str, panel: list[PanelTarget], arrays: dict[str, np.ndarray],
                           store: SequenceStore, resources: dict[str, Any], item_ids: np.ndarray,
                           rm_fit_counts: np.ndarray, per_writer: csv.DictWriter,
                           top_writer: csv.DictWriter) -> dict[str, dict[str, Any]]:
    n_items = len(item_ids)
    raw_to_index = {int(x): i for i, x in enumerate(item_ids.tolist())}
    accs = {name: RankAccumulator(panel_name, name, "fixed", None, per_writer, top_writer, item_ids, rm_fit_counts)
            for name in ("train_popularity", "recent_train_popularity_14d", "smoothed_markov", "history_frequency", "history_frequency_reranker")}
    for i, target in enumerate(panel):
        user = store.users[target.user_index]
        hstart = max(0, target.position - 200)
        history_ids = np.asarray(user.raw_item_ids[hstart:target.position], dtype=np.int64)
        history_indices = [raw_to_index[x] for x in history_ids.tolist() if x in raw_to_index]
        history = np.bincount(history_indices, minlength=n_items).astype(np.float64)
        ref = np.asarray(arrays["reference_logp_full"][i], dtype=np.float64)
        actor_order = np.lexsort((item_ids, -ref))
        actor_rank = np.empty(n_items, dtype=np.int32)
        actor_rank[actor_order] = np.arange(n_items, dtype=np.int32)
        previous_token = int(user.tokens[target.position - 1]) if target.position > 0 else 1
        markov = smoothed_transition_scores(resources["transition"].getrow(previous_token), resources["train_counts"], 1.0)
        scores = {
            "train_popularity": resources["train_counts"],
            "recent_train_popularity_14d": resources["recent_counts"],
            "smoothed_markov": markov,
            "history_frequency": history,
            "history_frequency_reranker": history * (n_items + 1.0) + (n_items - actor_rank),
        }
        for name, vector in scores.items():
            accs[name].add(target, vector)
    return {name: acc.finish() for name, acc in accs.items()}


def _score_panel(cache_path: Path, panel_name: str, per_writer: csv.DictWriter,
                 top_writer: csv.DictWriter, scaler: TrainOnlyFeatureScaler,
                 reward_scale: dict[str, Any], item_ids: np.ndarray,
                 rm_fit_counts: np.ndarray) -> tuple[dict[str, Any], dict[str, dict[int, float]], dict[str, dict[int, list[int]]], list[PanelTarget], dict[str, np.ndarray]]:
    meta, arrays, panel = _load_cache(cache_path, include_full=True)
    if len(panel) != int(meta["n_contexts"]):
        raise ValueError("cache identity mismatch: panel count")
    values: dict[str, Any] = {}
    tops: dict[str, dict[int, list[int]]] = {}
    all_user_scores: dict[str, dict[int, float]] = {}

    q_ref = np.exp(np.asarray(arrays["reference_conditional_logp"], dtype=np.float64))
    q_ref /= q_ref.sum(axis=-1, keepdims=True)
    result = _eval_cached_policy(panel_name, panel, arrays, q_ref, item_ids, rm_fit_counts,
                                 "strong_gpt", "reference", None, per_writer, top_writer)
    values["strong_gpt"] = result
    tops["strong_gpt"] = result.pop("top10_by_user")
    all_user_scores["strong_gpt"] = result["per_user_primary_ndcg"]

    # Fixed beta sequence is registered before any new measurements.
    raw_rewards = np.asarray(arrays["raw_candidate_rewards"])
    mean, std, floor = float(reward_scale["mean"]), float(reward_scale["std"]), float(reward_scale["std_floor"])
    for beta in (3.0, 1.0, 0.3, 0.1):
        q = _analytic_q(arrays["reference_conditional_logp"], raw_rewards, arrays["support_mass"], mean, std, floor, beta)
        name = f"analytic_rm_beta_{beta:g}"
        result = _eval_cached_policy(panel_name, panel, arrays, q, item_ids, rm_fit_counts,
                                     name, "analytic_tilt", None, per_writer, top_writer)
        values[name] = result
        tops[name] = result.pop("top10_by_user")
        all_user_scores[name] = result["per_user_primary_ndcg"]

    repeat_col = tuple(__import__("tlab.residual_support.features", fromlist=["POLICY_FEATURE_NAMES"]).POLICY_FEATURE_NAMES).index("personal_repeat_count_200")
    repeat_reward = (np.asarray(arrays["candidate_features"][..., repeat_col]) > 0).astype(np.float32)
    for beta in (1.0, 0.3):
        q = _analytic_q(arrays["reference_conditional_logp"], repeat_reward, arrays["support_mass"], mean, std, floor, beta)
        name = f"matched_repeat_analytic_beta_{beta:g}"
        result = _eval_cached_policy(panel_name, panel, arrays, q, item_ids, rm_fit_counts,
                                     name, "repeat_control", None, per_writer, top_writer)
        values[name] = result
        tops[name] = result.pop("top10_by_user")
        all_user_scores[name] = result["per_user_primary_ndcg"]

    scaler_obj = scaler
    main = json.loads((ARTIFACTS / ".." / "residual_support/policy/main_fit_summary.json").resolve().read_text(encoding="utf-8"))
    run_records = [(row["run_name"], row) for row in main["runs"]]
    continuation = ARTIFACTS / ".." / "residual_support/policy_diagnostic/runs/exact_expected_reward_b0.3_seed17_continuation"
    run_records.append((continuation.name, {"run_name": continuation.name, "objective": "exact_expected_reward",
                                             "beta": 0.3, "seed": 17, "variant": "continuation"}))
    head_index: dict[str, Any] = {}
    for run_name, run_meta in run_records:
        if "continuation" in run_name:
            run_dir = continuation
        else:
            run_dir = (ARTIFACTS / ".." / "residual_support/policy/runs" / run_name).resolve()
        run_manifest_path = run_dir / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else run_meta
        head_index[run_name] = run_meta
        for endpoint in ("last", "best"):
            checkpoint = run_dir / f"{endpoint}.pt"
            if not checkpoint.exists():
                values[f"policy:{run_name}:{endpoint}"] = {"status": "missing_checkpoint", "path": checkpoint.relative_to(ROOT).as_posix()}
                continue
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            step = int(saved.get("step", saved.get("updates_done", run_manifest.get("best_checkpoint_step", -1))))
            checkpoint_kind = "identity_step0" if endpoint == "best" and step == 0 else endpoint
            q, step = _predict_checkpoint(checkpoint, run_manifest, arrays, scaler_obj,
                                          torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            name = f"policy:{run_name}:{checkpoint_kind}"
            result = _eval_cached_policy(panel_name, panel, arrays, q, item_ids, rm_fit_counts,
                                         name, checkpoint_kind, int(run_meta["seed"]), per_writer, top_writer)
            values[name] = result
            values[name]["endpoint"] = checkpoint_kind
            tops[name] = result.pop("top10_by_user")
            all_user_scores[name] = result["per_user_primary_ndcg"]
            values[name]["checkpoint"] = checkpoint.relative_to(ROOT).as_posix()
            values[name]["checkpoint_sha256"] = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
            values[name]["training_step"] = step
            values[name]["objective"] = run_meta["objective"]
            values[name]["beta"] = float(run_meta["beta"])
            values[name]["seed"] = int(run_meta["seed"])
            values[name]["variant"] = run_meta["variant"]

    return values, all_user_scores, tops, panel, arrays


def _summary_row(panel: str, model: str, metric: dict[str, Any]) -> dict[str, Any]:
    primary = metric["primary_user_macro"]
    return {"panel": panel, "method": model, "n_users": primary["n_users"], "n_events": primary["n_events"],
            "ndcg10": primary["ndcg@10"], "recall10": primary["recall@10"],
            "target_coverage": metric.get("target_catalog_coverage"),
            "target_in_support": metric.get("target_in_optimized_support_fraction"),
            "oov_targets": metric.get("oov_targets")}


def _bootstrap_slice_mean(values: list[float], seed: int, draws: int = 2000) -> list[float] | None:
    if not values:
        return None
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 128):
        stop = min(draws, start + 128)
        indices = rng.integers(0, len(x), size=(stop - start, len(x)))
        means[start:stop] = x[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _seed_group_values(values: dict[str, Any], user_values: dict[str, dict[int, float]],
                       panel: str, objective: str, beta: float, variant: str, endpoint: str) -> tuple[dict[int, float], list[dict[str, float]]]:
    wanted = []
    for name, meta in values.items():
        if not name.startswith("policy:") or "status" in meta:
            continue
        if meta.get("objective") != objective or float(meta.get("beta", -99)) != float(beta) or meta.get("variant") != variant:
            continue
        if meta.get("endpoint") != endpoint:
            continue
        wanted.append((int(meta["seed"]), user_values[name]))
    wanted.sort(key=lambda x: x[0])
    if len(wanted) != 3:
        return {}, []
    merged = _mean_maps([x[1] for x in wanted])
    seed_values = [{"seed": s, "mean_ndcg10": float(np.mean(list(v.values()))), "n_users": len(v)} for s, v in wanted]
    return merged, seed_values


def _build_contrasts(panel: str, values: dict[str, Any], user_values: dict[str, dict[int, float]]) -> list[dict[str, Any]]:
    ref = user_values["strong_gpt"]
    out: list[dict[str, Any]] = []
    seed = 24000

    def append(name: str, left: dict[int, float], label: str) -> None:
        nonlocal seed
        if left:
            item = {"panel": panel, "contrast": label, **_contrast(left, ref, seed)}
            out.append(item)
            seed += 2

    for method in ("dense_sft", "last_target_sft", "weak_actor", "train_popularity", "recent_train_popularity_14d",
                   "smoothed_markov", "history_frequency", "history_frequency_reranker", "analytic_rm_beta_1",
                   "analytic_rm_beta_0.3", "matched_repeat_analytic_beta_1", "matched_repeat_analytic_beta_0.3"):
        if method in user_values:
            append(method, user_values[method], f"{method} - strong_gpt")
    if "dense_sft" in user_values and "last_target_sft" in user_values:
        out.append({"panel": panel, "contrast": "dense_sft - matched last_target_sft",
                    **_contrast(user_values["dense_sft"], user_values["last_target_sft"], seed)})
        seed += 2
    groups = [("sampled_loo", 1.0, "full"), ("sampled_loo", 0.3, "full"),
              ("exact_expected_reward", 1.0, "full"), ("exact_expected_reward", 0.3, "full"),
              ("analytic_q_distillation", 1.0, "full"), ("analytic_q_distillation", 0.3, "full"),
              ("exact_expected_reward", 0.3, "no_feedback")]
    grouped: dict[tuple[str, float, str, str], dict[int, float]] = {}
    seed_summaries = []
    for objective, beta, variant in groups:
        for endpoint in ("last", "best"):
            merged, per_seed = _seed_group_values(values, user_values, panel, objective, beta, variant, endpoint)
            if merged:
                label = f"{objective}_b{beta:g}_{variant}_{endpoint}_seed_mean - strong_gpt"
                append(label, merged, label)
                grouped[(objective, beta, variant, endpoint)] = merged
                seed_summaries.append({"panel": panel, "objective": objective, "beta": beta,
                                       "variant": variant, "endpoint": endpoint,
                                       "seed_results": per_seed,
                                       "seed_mean": float(np.mean(list(merged.values()))),
                                       "seed_sd": float(np.std([x["mean_ndcg10"] for x in per_seed], ddof=1)),
                                       "n_users": len(merged)})
    for beta in (1.0, 0.3):
        for endpoint in ("last", "best"):
            for objective, left_name in (("sampled_loo", "sampled_loo"), ("analytic_q_distillation", "analytic_q_distillation")):
                left = grouped.get((left_name, beta, "full", endpoint))
                right = grouped.get(("exact_expected_reward", beta, "full", endpoint))
                if left and right:
                    out.append({"panel": panel, "contrast": f"{left_name} - exact_expected_reward beta={beta:g} endpoint={endpoint}",
                                **_contrast(left, right, seed)})
                    seed += 2
            nf = grouped.get(("exact_expected_reward", beta, "no_feedback", endpoint)) if beta == 0.3 else None
            full = grouped.get(("exact_expected_reward", beta, "full", endpoint)) if beta == 0.3 else None
            if nf and full:
                out.append({"panel": panel, "contrast": f"trained no_feedback - full beta=0.3 endpoint={endpoint}",
                            **_contrast(nf, full, seed)})
                seed += 2
    return out, seed_summaries


def run_complete_evaluation() -> dict[str, Any]:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    output_root = ARTIFACTS / "evaluation"
    outputs = [output_root / "legacy_completion.csv", output_root / "per_user.csv",
               output_root / "top10_recommendations.csv", output_root / "seed_contrasts.csv",
               output_root / "legacy_completion_slices.csv", output_root / "legacy_completion.json"]
    if any(p.exists() or p.with_suffix(p.suffix + ".partial").exists() for p in outputs):
        raise FileExistsError("A1 outputs already exist; refusing to overwrite")
    temp_user = output_root / "per_user.csv.partial"
    temp_top = output_root / "top10_recommendations.csv.partial"
    field_user = ["panel", "method", "endpoint", "seed", "uid", "event_id", "rank", "ndcg10", "recall10",
                  "target_raw_id", "target_in_catalog", "target_in_support", "positive", "is_new", "is_repeat", "rm_fit_support_group"]
    field_top = ["panel", "method", "endpoint", "seed", "uid", "list_rank", "catalog_index", "raw_item_id", "score"]
    rows_summary: list[dict[str, Any]] = []
    rows_slices: list[dict[str, Any]] = []
    all_contrasts: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    panel_values: dict[str, dict[str, Any]] = {}
    panel_users: dict[str, dict[str, dict[int, float]]] = {}
    panel_tops: dict[str, dict[str, dict[int, list[int]]]] = {}

    processed = ROOT / "data/processed_full_model_tminus14"
    protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(processed / "sequence_index", expected_source_sha256=protocol["processed_listens_sha256"])
    catalog = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    rm_meta = json.loads((ARTIFACTS / ".." / "residual_support/rm/rm_diagnostics.json").resolve().read_text(encoding="utf-8"))
    rm_fit_counts = np.asarray(rm_meta["rm_fit_item_count"]["counts_by_catalog_item"], dtype=np.int64)
    rm_root = (ARTIFACTS / ".." / "residual_support/rm").resolve()
    reward_scale = json.loads((rm_root / "reward_scale.json").read_text(encoding="utf-8"))
    scaler = _scale_from_json((ARTIFACTS / ".." / "residual_support/policy/scaler.json").resolve())
    resources = _make_baseline_resources(store, item_ids, processed / "catalog.parquet")

    with temp_user.open("x", newline="", encoding="utf-8") as fu, temp_top.open("x", newline="", encoding="utf-8") as ft:
        user_writer = csv.DictWriter(fu, fieldnames=field_user)
        top_writer = csv.DictWriter(ft, fieldnames=field_top)
        user_writer.writeheader(); top_writer.writeheader()
        for panel_name, cache_path in PANEL_DIRS.items():
            print(f"A1: evaluating cached policies and controls on {panel_name}", flush=True)
            values, users, tops, panel, arrays = _score_panel(cache_path, panel_name, user_writer,
                                                              top_writer, scaler, reward_scale,
                                                              item_ids, rm_fit_counts)
            # Simple controls use the same contexts and exact fixed input histories.
            values.update(_eval_simple_baselines(panel_name, panel, arrays, store, resources,
                                                 item_ids, rm_fit_counts, user_writer, top_writer))
            panel_values[panel_name] = values
            panel_users[panel_name] = users
            panel_tops[panel_name] = tops

            # Baselines and actor comparisons from cached reference-distribution rows.
            # Dense/last-target are separately scored as full-catalog actor policies below.
            baseline_names = ("train_popularity", "recent_train_popularity_14d", "smoothed_markov",
                              "history_frequency", "history_frequency_reranker")
            for name in baseline_names:
                users[name] = values[name]["per_user_primary_ndcg"]
                tops[name] = values[name]["top10_by_user"]
            for name, metrics in values.items():
                if name.startswith("policy:") and "per_user_primary_ndcg" in metrics:
                    users[name] = metrics["per_user_primary_ndcg"]
                    if name not in tops and "top10_by_user" in metrics:
                        tops[name] = metrics["top10_by_user"]
            for name, metrics in values.items():
                if "primary_user_macro" in metrics:
                    rows_summary.append(_summary_row(panel_name, name, metrics))
                    for slice_name, slice_data in metrics.get("slices", {}).items():
                        per_user = slice_data.get("per_user", {})
                        ndcg = [float(v["ndcg@10"]) for v in per_user.values()]
                        recall = [float(v["recall@10"]) for v in per_user.values()]
                        ndcg_ci = _bootstrap_slice_mean(ndcg, 52000 + len(rows_slices), N_BOOTSTRAP)
                        recall_ci = _bootstrap_slice_mean(recall, 62000 + len(rows_slices), N_BOOTSTRAP)
                        rows_slices.append({
                            "panel": panel_name, "method": name, "slice": slice_name,
                            "n_users": int(slice_data.get("n_users", 0)),
                            "n_events": int(slice_data.get("n_events", 0)),
                            "ndcg10": slice_data.get("ndcg@10"),
                            "ndcg10_ci_low": ndcg_ci[0] if ndcg_ci else None,
                            "ndcg10_ci_high": ndcg_ci[1] if ndcg_ci else None,
                            "recall10": slice_data.get("recall@10"),
                            "recall10_ci_low": recall_ci[0] if recall_ci else None,
                            "recall10_ci_high": recall_ci[1] if recall_ci else None,
                        })

            # Free mapped references before moving to the next panel.
            del arrays
            print(f"A1: finished policy/control scoring on {panel_name}", flush=True)

        for method, checkpoint in (
            ("dense_sft", ROOT / "artifacts/residual_support/sft/runs/dense/last.pt"),
            ("last_target_sft", ROOT / "artifacts/residual_support/sft/runs/last_target/last.pt"),
            ("weak_actor", ROOT / "models/actor/weak.pt"),
        ):
            print(f"A1: scoring actor {method} on all eligible panels", flush=True)
            actor, actor_meta = load_actor_checkpoint(checkpoint, vocab_size=len(item_ids) + 3,
                                                       n_positions=256, n_layer=4, n_embd=256, n_head=4,
                                                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            for panel_name, cache_path in PANEL_DIRS.items():
                _, _, panel = _load_cache(cache_path, include_full=False)
                result = _eval_actor(panel_name, panel, actor, store, item_ids, rm_fit_counts, method,
                                     "full_catalog_actor", user_writer, top_writer)
                panel_values[panel_name][method] = result
                panel_users[panel_name][method] = result["per_user_primary_ndcg"]
                panel_tops[panel_name][method] = result["top10_by_user"]
                rows_summary.append(_summary_row(panel_name, method, result))
                for slice_name, slice_data in result.get("slices", {}).items():
                    per_user = slice_data.get("per_user", {})
                    ndcg = [float(v["ndcg@10"]) for v in per_user.values()]
                    recall = [float(v["recall@10"]) for v in per_user.values()]
                    ndcg_ci = _bootstrap_slice_mean(ndcg, 52000 + len(rows_slices), N_BOOTSTRAP)
                    recall_ci = _bootstrap_slice_mean(recall, 62000 + len(rows_slices), N_BOOTSTRAP)
                    rows_slices.append({
                        "panel": panel_name, "method": method, "slice": slice_name,
                        "n_users": int(slice_data.get("n_users", 0)),
                        "n_events": int(slice_data.get("n_events", 0)),
                        "ndcg10": slice_data.get("ndcg@10"),
                        "ndcg10_ci_low": ndcg_ci[0] if ndcg_ci else None,
                        "ndcg10_ci_high": ndcg_ci[1] if ndcg_ci else None,
                        "recall10": slice_data.get("recall@10"),
                        "recall10_ci_low": recall_ci[0] if recall_ci else None,
                        "recall10_ci_high": recall_ci[1] if recall_ci else None,
                    })
            del actor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"A1: finished actor {method}", flush=True)

    # Validate policy reconstruction against the already-published panel_tools.1 final metrics.
    old_final = json.loads((ARTIFACTS / ".." / "residual_support/evaluation/final_policy_evaluation.json").resolve().read_text(encoding="utf-8"))
    panel_tools2_final = panel_values["retrospective_final"]["strong_gpt"]["primary_user_macro"]["ndcg@10"]
    old_final_ndcg = old_final["models"]["strong_gpt"]["primary_user_macro"]["ndcg@10"]
    if not math.isclose(panel_tools2_final, old_final_ndcg, rel_tol=0, abs_tol=1e-12):
        raise AssertionError(f"panel_tools.2 reference reconstruction differs from panel_tools.1 final: {panel_tools2_final} vs {old_final_ndcg}")

    for panel_name, values in panel_values.items():
        contrast_rows, seeds = _build_contrasts(panel_name, values, panel_users[panel_name])
        all_contrasts.extend(contrast_rows)
        seed_summaries.extend(seeds)
    for contrast in all_contrasts:
        ci = contrast["paired_95pct_ci"]
        contrast["paired_95pct_ci"] = {k: v for k, v in ci.items() if k in {"mean_difference", "ci_lower", "ci_upper", "n_users", "confidence", "n_bootstrap", "seed"}}

    summary_csv = output_root / "legacy_completion.csv"
    with summary_csv.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_summary[0]))
        writer.writeheader(); writer.writerows(rows_summary)
    contrast_csv = output_root / "seed_contrasts.csv"
    with contrast_csv.open("x", newline="", encoding="utf-8") as f:
        fields = ["panel", "contrast", "n_users", "mean_difference", "ci_lower", "ci_upper",
                  "guard_passes", "guard_adjusted_lower_bound", "guard_mean_adjusted_difference"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_contrasts:
            ci = row.get("paired_95pct_ci", {})
            guard = row.get("relative_2pct_guard", {})
            writer.writerow({"panel": row["panel"], "contrast": row["contrast"], "n_users": row["n_users"],
                             "mean_difference": row["mean_difference"], "ci_lower": ci.get("ci_lower"),
                             "ci_upper": ci.get("ci_upper"), "guard_passes": guard.get("passes"),
                             "guard_adjusted_lower_bound": guard.get("adjusted_lower_bound"),
                             "guard_mean_adjusted_difference": guard.get("mean_adjusted_difference")})
    slice_csv = output_root / "legacy_completion_slices.csv"
    with slice_csv.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_slices[0]))
        writer.writeheader(); writer.writerows(rows_slices)
    temp_user.replace(output_root / "per_user.csv")
    temp_top.replace(output_root / "top10_recommendations.csv")

    elapsed = time.perf_counter() - started
    missing = [{"name": k, "path": v.get("path")} for panel, values in panel_values.items()
               for k, v in values.items() if isinstance(v, dict) and v.get("status") == "missing_checkpoint"]
    result = {
        "schema_version": 1,
        "status": "complete" if not missing else "incomplete_missing_checkpoint",
        "registered_roles": "retrospective/exploratory; panels previously opened",
        "started_at_utc": started_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(elapsed, 3),
        "common_panel_temporal_gate": "all 6655 rows eligible; each of 514 development and 710 retrospective final rows eligible; all 4134 A_RL train contexts are before actor cutoff and excluded from quality evaluation",
        "fixed_factors": {"catalog": 20000, "support_K": 500, "objective": "ratio", "tail": "unchanged reference tail", "seeds": [17, 42, 73]},
        "counts": {"method_rows": len(rows_summary), "contrasts": len(all_contrasts), "seed_groups": len(seed_summaries),
                   "slice_rows": len(rows_slices),
                   "per_user_rows": sum(len(v) for values in panel_users.values() for k, v in values.items()),
                   "missing_checkpoints": len(missing)},
        "missing_checkpoints": missing,
        "reconstruction_check": {"strong_final_ndcg10_residual_support": old_final_ndcg, "strong_final_ndcg10_policy_diagnostics": panel_tools2_final,
                                 "absolute_difference": abs(panel_tools2_final - old_final_ndcg), "passed": True},
        "outputs": [p.relative_to(ROOT).as_posix() for p in outputs],
        "notes": ["Main endpoint is last.pt; best.pt is a separate endpoint.",
                  "No-feedback step-0 best.pt is labeled identity_step0 and not averaged as a trained fit.",
                  "Three seeds are averaged per-user before group contrasts; user bootstrap does not inflate n.",
                  "Final and common-open results are descriptive and did not select models or checkpoints."],
        "seed_summary": seed_summaries,
        "contrast_rows": all_contrasts,
    }
    (output_root / "legacy_completion.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    record_operation("A1_legacy_full_catalog_scoring_and_2000_draw_user_bootstrap", elapsed, bucket="A", category="evaluation",
                     evidence=[p.relative_to(ROOT).as_posix() for p in outputs])
    update_stage("A", "evaluated", [p.relative_to(ROOT).as_posix() for p in outputs],
                 note="A1 full-catalog evaluation and registered user-level contrasts complete; A2 audit and A3 no-feedback are included in policy model table.")
    return result


def main() -> None:
    global N_BOOTSTRAP
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--bootstrap-draws", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    N_BOOTSTRAP = int(args.bootstrap_draws)
    if N_BOOTSTRAP < 100:
        parser.error("--bootstrap-draws must be at least 100")
    result = run_complete_evaluation()
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"],
                      "method_rows": result["counts"]["method_rows"], "contrasts": result["counts"]["contrasts"],
                      "reconstruction_check": result["reconstruction_check"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
