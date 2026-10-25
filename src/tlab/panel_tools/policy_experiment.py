from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch

from tlab.data.store import SequenceStore
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.panel_tools.baselines import _history_counts, _actor_tie_rank
from tlab.panel_tools.cache import ShardedArrayCache, panel_toolsCacheKey
from tlab.panel_tools.data import PanelTarget, history_for_target
from tlab.panel_tools.evaluator import _collate_histories, load_actor_checkpoint
from tlab.panel_tools.evaluation import single_target_metrics, target_rank
from tlab.panel_tools.features import candidate_history_features
from tlab.panel_tools.objectives import FrozenRewardScale
from tlab.panel_tools.candidates import fixed_support_policy
from tlab.panel_tools.policy_head import ResidualPolicyHead, ResidualHeadSpec, residual_log_policy, feature_columns_without_feedback
from tlab.panel_tools.selection import _sha256
from tlab.panel_tools.statistics import paired_user_bootstrap, relative_quality_guard, holm_adjust
from tlab.panel_tools.train_policy import train_residual_head, save_head_checkpoint
from tlab.evaluation.behavior import exposure_metrics, repeat_rate


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_panel(path: Path) -> list[PanelTarget]:
    return [PanelTarget(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_output(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    submission = (root / "submission").resolve()
    if resolved == submission or submission in resolved.parents:
        raise ValueError("panel_tools policy outputs may not be written inside submission/")
    if root.resolve() not in resolved.parents:
        raise ValueError("panel_tools policy outputs must remain inside the workspace")
    return resolved


def _candidate_context_features(user: Any, target: PanelTarget, candidate_ids: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    start = max(0, target.position - 1000)
    history_ids = np.asarray(user.raw_item_ids[start:target.position], dtype=np.int64)
    history_ratios = np.asarray(user.ratio_raw[start:target.position], dtype=np.float64)
    history_durations = np.asarray(user.duration_seconds[start:target.position], dtype=np.float64)
    candidate_to_index = {int(item): i for i, item in enumerate(candidate_ids.tolist())}
    duration_sum = np.zeros(len(candidate_ids), dtype=np.float64)
    duration_count = np.zeros(len(candidate_ids), dtype=np.float64)
    for item, duration in zip(history_ids.tolist(), history_durations.tolist()):
        index = candidate_to_index.get(int(item))
        if index is not None and math.isfinite(float(duration)) and duration >= 0:
            duration_sum[index] += float(duration)
            duration_count[index] += 1.0
    durations = np.divide(duration_sum, duration_count, out=np.full_like(duration_sum, np.nan), where=duration_count > 0)
    counts = np.zeros(len(candidate_ids), dtype=np.float64)
    for item in history_ids.tolist():
        index = candidate_to_index.get(int(item))
        if index is not None:
            counts[index] += 1.0
    candidate = candidate_history_features(
        history_item_ids=history_ids,
        history_ratios=history_ratios,
        candidate_item_ids=candidate_ids,
        duration_seconds=durations,
        train_popularity=counts,
        train_ratio_mean=np.full(len(candidate_ids), np.nan, dtype=np.float64),
        windows=(200, 1000),
        early_train_ratio_mean=0.5,
    )
    from tlab.panel_tools.rm_diagnostics import history_only_features

    context, context_names = history_only_features(history_ids, history_ratios)
    if not np.all(np.isfinite(candidate.values)) or not np.all(np.isfinite(context)):
        raise ValueError("candidate features contain non-finite values")
    return np.column_stack((np.repeat(context[None, :], len(candidate_ids), axis=0), candidate.values)), (
        *context_names, *candidate.names,
    )


def _actor_and_catalog(
    root: Path, actor_path: Path, catalog_path: Path, device: torch.device,
) -> tuple[ItemGPT2Actor, dict[str, Any], np.ndarray, np.ndarray]:
    catalog = pl.read_parquet(catalog_path).sort("token_id")
    item_ids = np.asarray(catalog.get_column("item_id").to_list(), dtype=np.int64)
    tokens = np.asarray(catalog.get_column("token_id").to_list(), dtype=np.int64)
    if not np.array_equal(tokens, np.arange(3, 3 + len(tokens), dtype=np.int64)):
        raise ValueError("catalog token IDs must be contiguous from 3")
    actor, actor_details = load_actor_checkpoint(actor_path, vocab_size=len(item_ids) + 3, device=device)
    return actor, actor_details, item_ids, tokens


def _event_sha(panel: Sequence[PanelTarget]) -> str:
    return hashlib.sha256("\n".join(row.event_id for row in panel).encode("ascii")).hexdigest()


def _cache_key(
    *, split: str, panel: Sequence[PanelTarget], catalog_path: Path, actor_path: Path,
    reward_model_path: Path, candidate_mode: str,
) -> panel_toolsCacheKey:
    catalog = pl.read_parquet(catalog_path).sort("token_id")
    tokens = np.asarray(catalog.get_column("token_id").to_list(), dtype=np.int64)
    vocab_sha = hashlib.sha256(tokens.astype("<i8", copy=False).tobytes()).hexdigest()
    return panel_toolsCacheKey(
        protocol="tlab_yambda_policy_diagnostics026-10-21_r1", split=split, event_ids_sha256=_event_sha(panel),
        vocabulary_sha256=vocab_sha, actor_sha256=_sha256(actor_path), reward_model_sha256=_sha256(reward_model_path),
        feature_schema="panel_tools_asof_candidate_v1+history_only_v1", normalization_version="frozen_reward_scale_v1",
        candidate_mode=candidate_mode,
    )


@torch.inference_mode()
def _build_context_chunk(
    actor: ItemGPT2Actor, model: Any, store: SequenceStore, panel_chunk: Sequence[PanelTarget],
    *, item_ids: np.ndarray, device: torch.device, top_k: int, history_events: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    histories = [history_for_target(store, target, history_events) for target in panel_chunk]
    input_ids, attention = _collate_histories(histories, device)
    catalog_logp = actor.next_action_log_probs(input_ids, attention)[:, 3:].float()
    # The item action is conditioned on choosing from the recommendable catalog;
    # renormalization does not change ranking but makes fixed support mass valid.
    catalog_logp = catalog_logp - torch.logsumexp(catalog_logp, dim=-1, keepdim=True)
    refs = catalog_logp.cpu().numpy().astype(np.float32, copy=False)
    batch_features: list[np.ndarray] = []
    batch_support: list[np.ndarray] = []
    batch_mass: list[float] = []
    batch_reward: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    for local, target in enumerate(panel_chunk):
        order = np.lexsort((item_ids, -refs[local]))
        support = order[:top_k].astype(np.int32, copy=False)
        mass = float(np.exp(refs[local, support].astype(np.float64)).sum())
        user = store.users[target.user_index]
        features, feature_names = _candidate_context_features(user, target, item_ids[support])
        if names is None:
            names = feature_names
        elif names != feature_names:
            raise ValueError("feature schema changed between contexts")
        reward = np.clip(model.predict(features), 0.0, 1.0).astype(np.float32, copy=False)
        batch_features.append(features)
        batch_support.append(support)
        batch_mass.append(mass)
        batch_reward.append(reward)
    if names is None:
        raise ValueError("empty policy cache context batch")
    # Remove repeated context columns before storage: the head sees the original
    # combined schema; keeping them is required for model compatibility.
    return (
        np.stack(batch_features).astype(np.float32, copy=False), refs,
        np.stack(batch_support).astype(np.int32, copy=False), np.asarray(batch_mass, dtype=np.float32),
        np.stack(batch_reward).astype(np.float32, copy=False), names,
    )


def prepare_policy_cache(
    workspace: str | Path,
    *,
    panel_path: str | Path,
    panel_manifest_path: str | Path,
    output_dir: str | Path,
    split: str,
    reward_model_path: str | Path = "artifacts/panel_tools/rm/models/rm_action_seed17/history_plus_candidate_cross.joblib",
    actor_path: str | Path = "models/actor/weak.pt",
    catalog_path: str | Path = "data/processed_full_model_tminus14/catalog.parquet",
    top_k: int = 500,
    chunk_size: int = 16,
    pilot_contexts: int = 128,
    phase_cap_hours: float = 3.0,
) -> dict[str, Any]:
    """Measure the first contexts, then build a hash-bound readonly policy cache."""
    root = Path(workspace).resolve()
    panel_file = (root / panel_path).resolve() if not Path(panel_path).is_absolute() else Path(panel_path).resolve()
    manifest_file = (root / panel_manifest_path).resolve() if not Path(panel_manifest_path).is_absolute() else Path(panel_manifest_path).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output = _safe_output(root, output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite panel_tools cache: {output}")
    reward_file = Path(reward_model_path)
    if not reward_file.is_absolute():
        reward_file = root / reward_file
    actor_file = Path(actor_path)
    if not actor_file.is_absolute():
        actor_file = root / actor_file
    catalog_file = Path(catalog_path)
    if not catalog_file.is_absolute():
        catalog_file = root / catalog_file
    manifest = _load_json(manifest_file)
    panel = _load_panel(panel_file)
    if not panel or len(panel) != len({row.uid for row in panel}):
        raise ValueError("policy cache requires a non-empty one-event-per-user panel")
    if manifest.get("event_ids_sha256") != _event_sha(panel):
        raise ValueError("panel manifest event hash does not match panel contents")
    key = _cache_key(split=split, panel=panel, catalog_path=catalog_file, actor_path=actor_file,
                     reward_model_path=reward_file, candidate_mode=f"topk{top_k}_fixed_tail")
    protocol = _load_json(root / "artifacts/panel_tools/protocol.json")
    source_sha = protocol["dataset"]["processed_full_model_tminus14_listens_sha256"]
    processed = root / protocol["dataset"].get("processed_full_model_tminus14_dir", "data/processed_full_model_tminus14")
    started = time.perf_counter()
    store = SequenceStore.load_cache(processed / "sequence_index", expected_source_sha256=source_sha)
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load the frozen panel_tools evaluator") from exc
    model = joblib.load(reward_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, actor_details, item_ids, _ = _actor_and_catalog(root, actor_file, catalog_file, device)
    probe_n = min(len(panel), max(1, pilot_contexts))
    probe_start = time.perf_counter()
    probe: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]] = []
    for start in range(0, probe_n, chunk_size):
        probe.append(_build_context_chunk(actor, model, store, panel[start:min(probe_n, start + chunk_size)],
                                          item_ids=item_ids, device=device, top_k=top_k))
    probe_seconds = time.perf_counter() - probe_start
    estimated_full_seconds = probe_seconds * len(panel) / probe_n
    budget_seconds = max(0.0, float(phase_cap_hours) * 3600.0)
    if estimated_full_seconds > budget_seconds:
        report = {
            "status": "stopped_before_full_cache_by_phase_cap", "split": split, "contexts": len(panel),
            "pilot_contexts": probe_n, "pilot_seconds": probe_seconds,
            "estimated_full_cache_seconds": estimated_full_seconds,
            "phase_cap_hours": phase_cap_hours,
            "decision": "Do not create the full cache; reduce scope only under the written budget and do not use this as policy quality evidence.",
        }
        return report
    first = probe[0]
    n_features = first[0].shape[-1]
    cache = ShardedArrayCache.create(output, key, [
        {"name": "candidate_features", "shape": [len(panel), top_k, n_features], "dtype": "float32"},
        {"name": "reference_logp", "shape": [len(panel), len(item_ids)], "dtype": "float32"},
        {"name": "support_indices", "shape": [len(panel), top_k], "dtype": "int32"},
        {"name": "support_mass", "shape": [len(panel)], "dtype": "float32"},
        {"name": "raw_candidate_rewards", "shape": [len(panel), top_k], "dtype": "float32"},
    ])
    shards = {name: cache.open_shard(name) for name in (
        "candidate_features", "reference_logp", "support_indices", "support_mass", "raw_candidate_rewards")}
    offset = 0
    schema_names = first[5]
    for batch in probe:
        count = len(batch[0])
        shards["candidate_features"][offset:offset + count] = batch[0]
        shards["reference_logp"][offset:offset + count] = batch[1]
        shards["support_indices"][offset:offset + count] = batch[2]
        shards["support_mass"][offset:offset + count] = batch[3]
        shards["raw_candidate_rewards"][offset:offset + count] = batch[4]
        offset += count
    build_start = time.perf_counter()
    for start in range(probe_n, len(panel), chunk_size):
        end = min(len(panel), start + chunk_size)
        batch = _build_context_chunk(actor, model, store, panel[start:end], item_ids=item_ids, device=device, top_k=top_k)
        if batch[5] != schema_names:
            raise ValueError("feature schema changed while building cache")
        count = end - start
        for name, values in zip(("candidate_features", "reference_logp", "support_indices", "support_mass", "raw_candidate_rewards"), batch[:5]):
            shards[name][start:end] = values
    for shard in shards.values():
        shard.flush()
    metadata = _load_json(output / "manifest.json")
    metadata.update({"panel_path": str(panel_file.relative_to(root)), "panel_sha256": _sha256(panel_file),
                     "panel_manifest_path": str(manifest_file.relative_to(root)), "panel_event_ids_sha256": _event_sha(panel),
                     "feature_names": list(schema_names), "catalog_item_ids": item_ids.tolist(),
                     "target_rows_path": str(panel_file.relative_to(root)), "actor_details": actor_details,
                     "build_runtime_seconds": time.perf_counter() - started,
                     "probe_contexts": probe_n, "probe_seconds": probe_seconds,
                     "projected_full_seconds_before_build": estimated_full_seconds,
                     "feature_reward_model_path": str(reward_file.relative_to(root))})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"status": "completed", "cache_path": str(output.relative_to(root)), "contexts": len(panel),
            "top_k": top_k, "feature_dim": n_features, "probe_contexts": probe_n,
            "probe_seconds": probe_seconds, "projected_full_seconds_before_build": estimated_full_seconds,
            "build_wall_seconds": time.perf_counter() - started, "estimated_cache_gib": round(
                sum(path.stat().st_size for path in output.glob("*.bin")) / (1024 ** 3), 3)}


def _load_cache_arrays(root: Path, cache_path: str | Path) -> tuple[ShardedArrayCache, dict[str, Any], dict[str, np.memmap]]:
    path = Path(cache_path)
    if not path.is_absolute():
        path = root / path
    metadata = _load_json(path / "metadata.json")
    cache = ShardedArrayCache(path, panel_toolsCacheKey(**metadata["key"]), mode="r")
    arrays = {name: cache.open_shard(name) for name in (
        "candidate_features", "reference_logp", "support_indices", "support_mass", "raw_candidate_rewards")}
    return cache, metadata, arrays


def _load_frozen_scale(path: Path) -> FrozenRewardScale:
    raw = _load_json(path)
    return FrozenRewardScale(mean=float(raw["mean"]), std=float(raw["std"]),
                             version=str(raw["version"]), std_floor=float(raw["std_floor"]))


def _distribution_from_cache_batch(
    reference_logp: np.ndarray, support_indices: np.ndarray, support_mass: np.ndarray,
    raw_rewards: np.ndarray, normalization: FrozenRewardScale, beta: float,
    *, log_q_conditional: np.ndarray | None = None,
) -> np.ndarray:
    ref = torch.from_numpy(np.asarray(reference_logp, dtype=np.float32))
    support = torch.from_numpy(np.asarray(support_indices, dtype=np.int64))
    mass = torch.from_numpy(np.asarray(support_mass, dtype=np.float32))
    rewards = torch.from_numpy(np.asarray(raw_rewards, dtype=np.float32))
    if log_q_conditional is None:
        adjusted = normalization.normalize(rewards) / float(beta)
    else:
        gathered = ref.gather(1, support)
        support_log_mass = torch.logsumexp(gathered, dim=-1, keepdim=True)
        ref_conditional = gathered - support_log_mass
        adjusted = torch.from_numpy(np.asarray(log_q_conditional, dtype=np.float32)) - ref_conditional
    policy = fixed_support_policy(ref, support, adjusted)
    if not torch.allclose(policy.support_mass, mass, atol=2e-5, rtol=2e-5):
        raise ValueError("cache support mass does not agree with the frozen full-catalog reference")
    return policy.probabilities.numpy()


def _one_sided_signflip_p(differences: np.ndarray, *, seed: int, draws: int = 4000) -> float:
    values = np.asarray(differences, dtype=np.float64)
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    batch = 128
    for start in range(0, draws, batch):
        count = min(batch, draws - start)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        null_means = (signs * values[None, :]).mean(axis=1)
        exceed += int(np.count_nonzero(null_means >= observed))
    return float((exceed + 1) / (draws + 1))


def _evaluate_policy_matrix(
    *, root: Path, panel: Sequence[PanelTarget], val_arrays: dict[str, np.memmap], val_metadata: dict[str, Any],
    item_ids: np.ndarray, normalization: FrozenRewardScale, head_specs: list[dict[str, Any]], store: SequenceStore,
    top_k: int = 10, batch_size: int = 64,
) -> dict[str, Any]:
    item_to_index = {int(item): index for index, item in enumerate(item_ids.tolist())}
    recent_history: list[set[int]] = []
    for target in panel:
        user = store.users[target.user_index]
        history_start = max(0, target.position - 20)
        raw_history = np.asarray(user.raw_item_ids[history_start:target.position], dtype=np.int64)
        recent_history.append({item_to_index[int(item)] for item in raw_history.tolist() if int(item) in item_to_index})
    base_values: dict[int, float] = {}
    base_recall: dict[int, float] = {}
    rows_by_method: dict[str, dict[int, float]] = defaultdict(dict)
    recall_by_method: dict[str, dict[int, float]] = defaultdict(dict)
    slices_by_method: dict[str, dict[str, dict[str, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    support_mass_values: dict[str, list[float]] = defaultdict(list)
    support_target_hits: dict[str, int] = defaultdict(int)
    support_topk_hits: dict[str, int] = defaultdict(int)
    recommendations: dict[str, list[list[int]]] = defaultdict(list)
    catalog_size = len(item_ids)
    for start in range(0, len(panel), batch_size):
        end = min(len(panel), start + batch_size)
        refs = np.asarray(val_arrays["reference_logp"][start:end], dtype=np.float32)
        supports = np.asarray(val_arrays["support_indices"][start:end], dtype=np.int64)
        masses = np.asarray(val_arrays["support_mass"][start:end], dtype=np.float32)
        rewards = np.asarray(val_arrays["raw_candidate_rewards"][start:end], dtype=np.float32)
        features = np.asarray(val_arrays["candidate_features"][start:end], dtype=np.float32)
        batch_targets = panel[start:end]
        for row, target in enumerate(batch_targets):
            base_scores = refs[row]
            rank = target_rank(base_scores, item_ids, target.raw_item_id)
            metric = single_target_metrics(rank, top_k)
            base_values[target.uid] = metric[f"ndcg@{top_k}"]
            base_recall[target.uid] = metric[f"recall@{top_k}"]
            top_items = np.lexsort((item_ids, -base_scores))[:top_k]
            recommendations["base_actor"].append(top_items.tolist())
            support_set = set(supports[row].tolist())
            support_topk_hits["base_actor"] += sum(int(index in support_set) for index in top_items)
        support_mass_values["base_actor"].extend([float(value) for value in masses])
        for local, target in enumerate(batch_targets):
            if target.raw_item_id in item_to_index:
                support_target_hits["base_actor"] += int(item_to_index[target.raw_item_id] in set(supports[local].tolist()))
        # Registered analytic fixed-support policies.
        for beta in (3.0, 1.0, 0.3, 0.1):
            distributions = _distribution_from_cache_batch(refs, supports, masses, rewards, normalization, beta)
            name = f"analytic_beta_{beta:g}"
            support_mass_values[name].extend(float(x) for x in masses)
            for local, target in enumerate(batch_targets):
                rank = target_rank(distributions[local], item_ids, target.raw_item_id)
                metric = single_target_metrics(rank, top_k)
                rows_by_method[name][target.uid] = metric[f"ndcg@{top_k}"]
                recall_by_method[name][target.uid] = metric[f"recall@{top_k}"]
                top_items = np.lexsort((item_ids, -distributions[local]))[:top_k]
                recommendations[name].append(top_items.tolist())
                support_set = set(supports[local].tolist())
                support_topk_hits[name] += sum(int(index in support_set) for index in top_items)
                if target.raw_item_id in item_to_index:
                    support_target_hits[name] += int(item_to_index[target.raw_item_id] in support_set)
                for slice_name in ("positive", "is_new", "is_repeat", "known_catalog"):
                    in_slice = target.token_id >= 3 if slice_name == "known_catalog" else bool(getattr(target, slice_name))
                    if in_slice:
                        slices_by_method[name][slice_name]["ndcg@10"][target.uid] = metric["ndcg@10"]
                        slices_by_method[name][slice_name]["recall@10"][target.uid] = metric["recall@10"]
        for spec in head_specs:
            head = spec["head"]
            feature_cols = spec.get("feature_columns")
            xb = torch.from_numpy(features if feature_cols is None else features[..., feature_cols])
            ref_support = torch.from_numpy(np.take_along_axis(refs, supports, axis=1))
            log_mass = torch.logsumexp(ref_support, dim=-1, keepdim=True)
            ref_cond = ref_support - log_mass
            head_device = next(head.parameters()).device
            with torch.inference_mode():
                log_q, _ = residual_log_policy(ref_cond.to(head_device), xb.to(head_device), head)
            probabilities = _distribution_from_cache_batch(
                refs, supports, masses, rewards, normalization, float(spec["beta"]),
                log_q_conditional=log_q.cpu().numpy(),
            )
            name = str(spec["name"])
            support_mass_values[name].extend(float(x) for x in masses)
            for local, target in enumerate(batch_targets):
                rank = target_rank(probabilities[local], item_ids, target.raw_item_id)
                metric = single_target_metrics(rank, top_k)
                rows_by_method[name][target.uid] = metric[f"ndcg@{top_k}"]
                recall_by_method[name][target.uid] = metric[f"recall@{top_k}"]
                top_items = np.lexsort((item_ids, -probabilities[local]))[:top_k]
                recommendations[name].append(top_items.tolist())
                support_set = set(supports[local].tolist())
                support_topk_hits[name] += sum(int(index in support_set) for index in top_items)
                if target.raw_item_id in item_to_index:
                    support_target_hits[name] += int(item_to_index[target.raw_item_id] in support_set)
                for slice_name in ("positive", "is_new", "is_repeat", "known_catalog"):
                    in_slice = target.token_id >= 3 if slice_name == "known_catalog" else bool(getattr(target, slice_name))
                    if in_slice:
                        slices_by_method[name][slice_name]["ndcg@10"][target.uid] = metric["ndcg@10"]
                        slices_by_method[name][slice_name]["recall@10"][target.uid] = metric["recall@10"]
    candidates = {"base_actor": base_values, **rows_by_method}
    recall_candidates = {"base_actor": base_recall, **recall_by_method}
    paired = {}
    p_values = []
    p_names = []
    for index, (name, values) in enumerate(rows_by_method.items()):
        paired[name] = {
            "paired_vs_base": paired_user_bootstrap(values, base_values, n_bootstrap=4000, seed=17017 + index),
            "relative_2_percent_guard_vs_base": relative_quality_guard(
                values, base_values, tolerance=0.02, n_bootstrap=4000, seed=18017 + index,
            ),
        }
        aligned = np.asarray([values[uid] - base_values[uid] for uid in sorted(base_values)], dtype=np.float64)
        p_names.append(name)
        p_values.append(_one_sided_signflip_p(aligned, seed=27017 + index))
    adjusted = holm_adjust(p_values)
    for name, raw_p, holm_p in zip(p_names, p_values, adjusted):
        paired[name]["one_sided_signflip_p"] = raw_p
        paired[name]["holm_adjusted_p"] = holm_p
    out = {
        "panel_events": len(panel), "panel_users": len({row.uid for row in panel}),
        "catalog_size": catalog_size, "top_k": top_k,
        "evaluation_label": "frozen retrospective validation; exposed in V1-full_model; not an independent test",
        "aggregation": "one target per user; user-macro NDCG/Recall",
        "primary": {
            name: {"ndcg@10": float(np.mean(list(values.values()))),
                  "recall@10": float(np.mean(list(recall_candidates[name].values()))),
                  "n_users": len(values), "support_mass_mean": float(np.mean(support_mass_values.get(name, []))),
                  "target_in_support_rate": float(support_target_hits.get(name, 0) / max(1, len(panel))),
                  "support_share_top10": float(support_topk_hits.get(name, 0) / max(1, sum(len(row) for row in recommendations[name]))),
                  "repeat_share_top10": repeat_rate(recommendations[name], recent_history, top_k=top_k),
                  "exposure": exposure_metrics(recommendations[name], catalog_size, top_k=top_k),
                  "slices": {
                      slice_name: {
                          **{metric_name: (float(np.mean(list(by_user.values()))) if by_user else None)
                             for metric_name, by_user in metric_values.items()},
                          "n_users": len(next(iter(metric_values.values()))) if metric_values else 0,
                      }
                      for slice_name, metric_values in slices_by_method.get(name, {}).items()
                  }}
            for name, values in candidates.items()
        },
        "paired_vs_base_actor": paired,
        "per_user_ndcg10": {name: {str(uid): value for uid, value in values.items()} for name, values in candidates.items()},
        "multiple_comparison_method": "one-sided paired user sign-flip p-values with Holm adjustment; descriptive on exposed retrospective outcomes",
        "limitations": [
            "RM predictions for unchosen catalog items are model outputs, not observed rewards or causal effects.",
            "The primary validation outcomes were inspected in V1-full_model; all contrasts are exploratory retrospective evidence.",
            "The primary table reports NDCG@10; full slice, exposure and repeat-rate diagnostics are computed in the core-control runner.",
        ],
    }
    return out


def run_policy_matrix(
    workspace: str | Path,
    *,
    train_cache_path: str | Path = "artifacts/panel_tools/cache/policy_train_k500_seed17",
    validation_cache_path: str | Path = "artifacts/panel_tools/cache/policy_validation_k500_seed17",
    train_panel_path: str | Path = "artifacts/panel_tools/evaluation/frozen_a_rl_policy_train_panel.jsonl",
    validation_panel_path: str | Path = "artifacts/panel_tools/evaluation/frozen_primary_validation_panel.jsonl",
    scale_path: str | Path = "artifacts/panel_tools/policy/frozen_reward_scale.json",
    output_path: str | Path = "artifacts/panel_tools/policy/policy_matrix_validation.json",
    updates: int = 100,
    hidden_dim: int = 32,
    seed_schedule: Sequence[int] = (17, 42),
    training_phase_cap_hours: float = 5.0,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    def full(path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else root / candidate
    output = _safe_output(root, full(output_path))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite panel_tools policy matrix: {output}")
    _, train_meta, train = _load_cache_arrays(root, full(train_cache_path))
    _, val_meta, val = _load_cache_arrays(root, full(validation_cache_path))
    train_panel = _load_panel(full(train_panel_path))
    validation_panel = _load_panel(full(validation_panel_path))
    if len(train_panel) != train["candidate_features"].shape[0] or len(validation_panel) != val["candidate_features"].shape[0]:
        raise ValueError("cache row count and frozen panel length do not align")
    if train_meta["key"]["split"] != "A_RL" or val_meta["key"]["split"] != "validation":
        raise ValueError("train/evaluation cache split labels are incompatible")
    if train_meta["feature_names"] != val_meta["feature_names"]:
        raise ValueError("training and evaluation feature schemas differ")
    normalization = _load_frozen_scale(full(scale_path))
    train_features = torch.from_numpy(np.asarray(train["candidate_features"], dtype=np.float32))
    train_ref = torch.from_numpy(np.asarray(train["reference_logp"], dtype=np.float32))
    train_support = torch.from_numpy(np.asarray(train["support_indices"], dtype=np.int64))
    train_mass = torch.from_numpy(np.asarray(train["support_mass"], dtype=np.float32))
    train_rewards = torch.from_numpy(np.asarray(train["raw_candidate_rewards"], dtype=np.float32))
    train_ref_support = train_ref.gather(1, train_support)
    train_log_cond = train_ref_support - torch.logsumexp(train_ref_support, dim=-1, keepdim=True)
    full_feature_names = list(train_meta["feature_names"])
    no_feedback_cols = feature_columns_without_feedback(full_feature_names)
    records: list[dict[str, Any]] = []
    head_specs: list[dict[str, Any]] = []
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [("full_available", None, ["sampled_loo", "exact_expected_reward", "analytic_q_distillation"]),
                ("no_feedback", no_feedback_cols, ["exact_expected_reward"])]
    seeds = [int(seed) for seed in seed_schedule]
    estimated_one_seed_seconds = None
    for seed in seeds:
        seed_started = time.perf_counter()
        for variant, columns, objectives in variants:
            selected_x = train_features if columns is None else train_features.index_select(-1, torch.as_tensor(columns, dtype=torch.long))
            for objective in objectives:
                for beta in (1.0, 0.3):
                    head, metadata = train_residual_head(
                        selected_x, train_log_cond, train_rewards, train_mass, normalization,
                        objective=objective, beta=beta, seed=seed, updates=updates,
                        batch_contexts=16, hidden_dim=hidden_dim, learning_rate=1e-3,
                        samples_per_context=4, device=device,
                    )
                    variant_name = f"{variant}_{objective}_beta{beta:g}_seed{seed}"
                    checkpoint_path = root / "artifacts/panel_tools/policy/heads" / f"{variant_name}.pt"
                    checkpoint_sha = save_head_checkpoint(checkpoint_path, head, {
                        **metadata, "feature_variant": variant, "feature_names": [full_feature_names[i] for i in columns] if columns is not None else full_feature_names,
                        "train_cache_key_sha256": train_meta["key_sha256"], "actor_sha256": train_meta["key"]["actor_sha256"],
                        "reward_model_sha256": train_meta["key"]["reward_model_sha256"],
                        "reward_scale": {"mean": normalization.mean, "std": normalization.std,
                                         "std_floor": normalization.std_floor, "version": normalization.version},
                    })
                    record = {"name": variant_name, **metadata, "checkpoint_path": str(checkpoint_path.relative_to(root)),
                              "checkpoint_sha256": checkpoint_sha, "feature_variant": variant,
                              "seed": seed, "beta": beta}
                    records.append(record)
                    head_specs.append({"name": variant_name, "head": head, "beta": beta, "feature_columns": columns,
                                       "seed": seed, "objective": objective})
        seed_elapsed = time.perf_counter() - seed_started
        if seed == seeds[0]:
            estimated_one_seed_seconds = seed_elapsed
            projected_all = estimated_one_seed_seconds * len(seeds)
            if len(seeds) > 1 and projected_all > float(training_phase_cap_hours) * 3600.0:
                # Seed schedule was frozen, but the plan permits reduction when the
                # measured phase cap will be exceeded. Keep seed 17 as an explicit pilot.
                for record in records:
                    record["seed_budget_decision"] = "seed_42_not_run_after_measured_cap_projection"
                break
    panel_path = full(validation_panel_path)
    protocol = _load_json(root / "artifacts/panel_tools/protocol.json")
    processed = root / protocol["dataset"].get("processed_full_model_tminus14_dir", "data/processed_full_model_tminus14")
    store = SequenceStore.load_cache(processed / "sequence_index", expected_source_sha256=protocol["dataset"]["processed_full_model_tminus14_listens_sha256"])
    catalog = pl.read_parquet(root / "data/processed_full_model_tminus14/catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog.get_column("item_id").to_list(), dtype=np.int64)
    started_eval = time.perf_counter()
    evaluation = _evaluate_policy_matrix(root=root, panel=validation_panel, val_arrays=val,
                                         val_metadata=val_meta, item_ids=item_ids,
                                         normalization=normalization, head_specs=head_specs, store=store)
    evaluation["runtime_seconds"] = time.perf_counter() - started_eval
    evaluation["policy_training"] = records
    evaluation["seed_schedule_requested"] = seeds
    evaluation["seed_schedule_completed"] = sorted({int(record["seed"]) for record in records})
    evaluation["estimated_one_seed_training_seconds"] = estimated_one_seed_seconds
    evaluation["total_wall_seconds_in_runner"] = time.perf_counter() - started
    evaluation["training_cutoff_respected"] = True
    evaluation["normalization"] = {"version": normalization.version, "mean": normalization.mean,
                                   "std": normalization.std, "std_floor": normalization.std_floor,
                                   "shared_sampled_exact": True}
    evaluation["cache_provenance"] = {"train_cache_key_sha256": train_meta["key_sha256"],
                                      "validation_cache_key_sha256": val_meta["key_sha256"]}
    evaluation["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {"output_path": str(output.relative_to(root)), "panel_users": evaluation["panel_users"],
            "candidate_count": len(evaluation["primary"]), "training_models": len(records),
            "runtime_seconds": evaluation["total_wall_seconds_in_runner"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build hash-bound panel_tools policy caches or run the frozen head matrix")
    parser.add_argument("--workspace", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    cache_parser = sub.add_parser("prepare-cache")
    cache_parser.add_argument("--panel", required=True)
    cache_parser.add_argument("--panel-manifest", required=True)
    cache_parser.add_argument("--output", required=True)
    cache_parser.add_argument("--split", required=True)
    cache_parser.add_argument("--top-k", type=int, default=500)
    cache_parser.add_argument("--pilot-contexts", type=int, default=128)
    cache_parser.add_argument("--phase-cap-hours", type=float, default=3.0)
    run_parser = sub.add_parser("run-matrix")
    run_parser.add_argument("--updates", type=int, default=100)
    run_parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args(argv)
    if args.command == "prepare-cache":
        result = prepare_policy_cache(args.workspace, panel_path=args.panel, panel_manifest_path=args.panel_manifest,
                                      output_dir=args.output, split=args.split, top_k=args.top_k,
                                      pilot_contexts=args.pilot_contexts, phase_cap_hours=args.phase_cap_hours)
    else:
        result = run_policy_matrix(args.workspace, updates=args.updates,
                                   seed_schedule=args.seed or (17, 42))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
