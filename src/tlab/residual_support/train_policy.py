from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch
import yaml

from tlab.data.store import SequenceStore
from tlab.panel_tools.data import PanelTarget
from tlab.panel_tools.statistics import relative_quality_guard
from tlab.residual_support.cache import CacheIdentity, validate_cache_identity
from tlab.residual_support.features import POLICY_FEATURE_NAMES
from tlab.residual_support.head import ResidualHeadSpec, ResidualPolicyHead
from tlab.residual_support.metrics import compact_guard, evaluate_full_catalog_policy
from tlab.residual_support.objectives import FrozenRewardScale, exact_support_statistics, sampled_loo_loss
from tlab.residual_support.scaler import TrainOnlyFeatureScaler
from tlab.residual_support.rm import sha256_file
from tlab.residual_support.cache import _mapping_sha


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_jsonl_panel(path: Path) -> list[PanelTarget]:
    return [PanelTarget(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_cache(path: Path, *, include_full: bool) -> tuple[dict[str, Any], dict[str, np.ndarray], list[PanelTarget]]:
    metadata = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    dims = metadata["artifact_arrays"]
    arrays: dict[str, np.ndarray] = {}
    for name in ("candidate_features", "reference_conditional_logp", "support_indices", "support_mass", "raw_candidate_rewards"):
        spec = dims[name]
        arrays[name] = np.memmap(path / f"{name}.bin", dtype=spec["dtype"], mode="r", shape=tuple(spec["shape"]))
    if include_full:
        if not metadata["reference_full_catalog_saved"]:
            raise ValueError(f"full reference catalog distribution is unavailable in {path}")
        spec = dims["reference_logp_full"]
        arrays["reference_logp_full"] = np.memmap(path / "reference_logp_full.bin", dtype=spec["dtype"], mode="r", shape=tuple(spec["shape"]))
    panel = _load_jsonl_panel(path / "panel_rows.jsonl")
    if len(panel) != metadata["n_contexts"]:
        raise ValueError("cache panel rows and manifest context count disagree")
    return metadata, arrays, panel


def _cache_identity(bound_path: Path) -> CacheIdentity:
    metadata = json.loads(bound_path.read_text(encoding="utf-8"))
    raw = dict(metadata["identity"])
    raw["reward_normalization"] = tuple(tuple(item) for item in raw["reward_normalization"])
    identity = CacheIdentity(**raw)
    validate_cache_identity(metadata, identity)
    return identity


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    primary = {key: value for key, value in metrics["primary_user_macro"].items() if key != "per_user"}
    slices = {name: {key: value for key, value in data.items() if key != "per_user"}
              for name, data in metrics["slices"].items()}
    return {key: value for key, value in metrics.items() if key not in {"slices", "per_user_primary_ndcg"}} | {
        "primary_user_macro": primary, "slices": slices,
    }


def _torch_batch(arrays: dict[str, np.ndarray], scaler: TrainOnlyFeatureScaler, indices: np.ndarray,
                 columns: Sequence[int], device: torch.device) -> tuple[torch.Tensor, ...]:
    feature_names = tuple(POLICY_FEATURE_NAMES)
    x_cpu = scaler.transform(np.asarray(arrays["candidate_features"][indices]), feature_names)
    x_cpu = x_cpu[..., list(columns)]
    x = torch.from_numpy(np.ascontiguousarray(x_cpu)).to(device)
    reference = torch.from_numpy(np.asarray(arrays["reference_conditional_logp"][indices]).copy()).to(device)
    rewards = torch.from_numpy(np.asarray(arrays["raw_candidate_rewards"][indices]).copy()).to(device)
    mass = torch.from_numpy(np.asarray(arrays["support_mass"][indices]).copy()).to(device)
    return x, reference, rewards, mass


@torch.inference_mode()
def _exact_summary(head: torch.nn.Module, arrays: dict[str, np.ndarray], scaler: TrainOnlyFeatureScaler,
                   columns: Sequence[int], scale: FrozenRewardScale, beta: float,
                   device: torch.device, batch_size: int = 128) -> dict[str, float]:
    count = len(arrays["candidate_features"])
    totals: dict[str, float] = {}
    for start in range(0, count, batch_size):
        indices = np.arange(start, min(start + batch_size, count), dtype=np.int64)
        x, ref, rewards, mass = _torch_batch(arrays, scaler, indices, columns, device)
        log_q = torch.log_softmax(ref + head(x), dim=-1)
        values = exact_support_statistics(log_q, ref, rewards, mass, scale, beta)
        values = exact_support_statistics(
            log_q, ref, rewards, mass, scale, beta,
            log_target=values["analytic_target_log_policy"],
        )
        for key in ("support_weighted_objective", "support_weighted_reward_expectation",
                    "kl_policy_to_reference_conditional", "kl_policy_to_reference_mass_weighted",
                    "kl_target_to_policy_conditional", "kl_target_to_policy_mass_weighted",
                    "analytic_objective_gap", "analytic_support_weighted_objective"):
            item = values[key]
            totals[key] = totals.get(key, 0.0) + float(item.sum().cpu())
    return {key: value / count for key, value in totals.items()}


@torch.inference_mode()
def _evaluate_relevance(head: torch.nn.Module, cache_arrays: dict[str, np.ndarray],
                        cache_metadata: dict[str, Any], panel: Sequence[PanelTarget],
                        scaler: TrainOnlyFeatureScaler, columns: Sequence[int], item_ids: np.ndarray,
                        rm_fit_counts: np.ndarray, device: torch.device, batch_size: int = 32) -> dict[str, Any]:
    conditional_parts = []
    count = len(panel)
    for start in range(0, count, batch_size):
        indices = np.arange(start, min(start + batch_size, count), dtype=np.int64)
        x, ref, _, _ = _torch_batch(cache_arrays, scaler, indices, columns, device)
        conditional_parts.append(torch.softmax(ref + head(x), dim=-1).cpu().numpy().astype(np.float32))
    conditional = np.concatenate(conditional_parts, axis=0)
    return evaluate_full_catalog_policy(
        panel=panel, full_reference_logp=cache_arrays["reference_logp_full"],
        support_indices=cache_arrays["support_indices"], support_mass=cache_arrays["support_mass"],
        policy_conditional=conditional, item_ids=item_ids, rm_fit_counts=rm_fit_counts,
    )


def _activation_diagnostics(head: torch.nn.Module, x: torch.Tensor, grad_norms: dict[str, float]) -> dict[str, Any]:
    with torch.no_grad():
        pre = head.network[0](x.float())
        activation = torch.tanh(pre)
        delta = head(x.float())
        return {
            "preactivation_quantiles": [float(v) for v in torch.quantile(pre.flatten(), torch.tensor([0.05, 0.5, 0.95], device=pre.device)).cpu()],
            "tanh_saturated_fraction_abs_gt_0_99": float((activation.abs() > 0.99).float().mean().cpu()),
            "tanh_saturated_fraction_abs_gt_0_95": float((activation.abs() > 0.95).float().mean().cpu()),
            "delta_std_within_context_mean": float(delta.std(dim=1, unbiased=False).mean().cpu()),
            "delta_abs_mean": float(delta.abs().mean().cpu()),
            "gradient_norms": grad_norms,
        }


def _grad_norms(head: torch.nn.Module) -> dict[str, float]:
    def norm(parameters: Sequence[torch.nn.Parameter]) -> float:
        pieces = [p.grad.detach().float().norm(2) ** 2 for p in parameters if p.grad is not None]
        return float(torch.stack(pieces).sum().sqrt().cpu()) if pieces else 0.0
    return {
        "input_linear": norm([head.network[0].weight, head.network[0].bias]),
        "output_linear": norm([head.network[2].weight, head.network[2].bias]),
    }


def _atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def _run_one(
    *,
    run_dir: Path,
    run_name: str,
    objective: str,
    beta: float,
    seed: int,
    variant: str,
    columns: tuple[int, ...],
    config: dict[str, Any],
    train_meta: dict[str, Any], train_arrays: dict[str, np.ndarray],
    dev_meta: dict[str, Any], dev_arrays: dict[str, np.ndarray], dev_panel: list[PanelTarget],
    scaler: TrainOnlyFeatureScaler, scale: FrozenRewardScale,
    baseline_dev: dict[str, Any], item_ids: np.ndarray, rm_fit_counts: np.ndarray,
    device: torch.device, resume: bool,
) -> dict[str, Any]:
    params = config["train"]
    state_path = run_dir / "last.pt"
    curve_path = run_dir / "curve.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=resume)
    state_signature = {
        "run_name": run_name, "objective": objective, "beta": float(beta), "seed": int(seed),
        "variant": variant, "feature_columns": list(columns), "cache_identity_sha256": train_meta["bound_identity_sha256"],
        "scaler_sha256": scaler.digest(), "updates_limit": int(params["max_updates"]),
    }
    if state_path.exists():
        if not resume:
            raise FileExistsError(f"partial policy run exists; rerun with --resume: {run_dir}")
        saved = torch.load(state_path, map_location=device, weights_only=False)
        if saved["state_signature"] != state_signature:
            raise ValueError(f"resume signature mismatch for {run_name}")
        head = ResidualPolicyHead(ResidualHeadSpec(input_dim=len(columns), hidden_dim=int(params["hidden_dim"]))).to(device)
        head.load_state_dict(saved["head_state"])
        optimizer = torch.optim.AdamW(head.parameters(), lr=float(params["learning_rate"]), weight_decay=1e-4)
        optimizer.load_state_dict(saved["optimizer_state"])
        generator = torch.Generator(device=device)
        generator.set_state(saved["generator_state"].cpu())
        start_step = int(saved["updates_done"])
        best_selected_obj = float(saved["best_selected_objective"])
        best_selected_step = int(saved["best_selected_step"])
        best_selected_state = saved["best_selected_state"]
        best_selected_metrics = saved["best_selected_metrics"]
        best_selected_user_ndcg = saved["best_selected_user_ndcg"]
        plateau_best = float(saved["plateau_best"])
        plateau_bad_checks = int(saved["plateau_bad_checks"])
        if not curve_path.exists():
            raise ValueError("resumable policy state has no learning curve")
        print(f"Resuming {run_name} from update {start_step:,}", flush=True)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"unrecognized partial policy run artifacts: {run_dir}")
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        head = ResidualPolicyHead(ResidualHeadSpec(input_dim=len(columns), hidden_dim=int(params["hidden_dim"]))).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=float(params["learning_rate"]), weight_decay=1e-4)
        generator = torch.Generator(device=device).manual_seed(int(seed) + 1)
        start_step = 0
        best_selected_obj = -math.inf
        best_selected_step = 0
        best_selected_state = copy.deepcopy({key: value.detach().cpu().clone() for key, value in head.state_dict().items()})
        best_selected_metrics: dict[str, Any] | None = None
        best_selected_user_ndcg: dict[int, float] | None = None
        plateau_best = -math.inf
        plateau_bad_checks = 0
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_state.json").write_text(json.dumps(state_signature, indent=2) + "\n", encoding="utf-8")

    threshold = float(config["train"]["dev_guard_tolerance"])
    check_every = int(params["check_every"])
    minimum_updates = int(params["minimum_updates"])
    max_updates = int(params["max_updates"])
    batch_contexts = int(params["batch_contexts"])
    samples = int(params["samples_per_context"])
    train_count = len(train_arrays["candidate_features"])
    dev_baseline_ndcg = float(baseline_dev["primary_user_macro"]["ndcg@10"])
    train_start = time.perf_counter()
    stop_reason = "max_updates"
    with curve_path.open("a", encoding="utf-8", newline="\n") as curve:
        if start_step == 0:
            initial_row = _log_checkpoint(0, head, objective, beta, columns, train_meta, train_arrays, dev_meta,
                                          dev_arrays, dev_panel, scaler, scale, baseline_dev, item_ids, rm_fit_counts,
                                          device, curve, run_name, seed, variant)
            best_selected_obj = float(initial_row["dev"]["support_weighted_objective"])
            best_selected_metrics = _compact_metrics(initial_row["dev_relevance"])
            best_selected_user_ndcg = dict(initial_row["dev_relevance"]["per_user_primary_ndcg"])
            plateau_best = best_selected_obj
            _atomic_torch_save({
                "state_signature": state_signature, "updates_done": 0,
                "head_state": head.state_dict(), "optimizer_state": optimizer.state_dict(),
                "generator_state": generator.get_state(), "best_selected_objective": best_selected_obj,
                "best_selected_step": best_selected_step, "best_selected_state": best_selected_state,
                "best_selected_metrics": best_selected_metrics, "best_selected_user_ndcg": best_selected_user_ndcg,
                "plateau_best": plateau_best, "plateau_bad_checks": plateau_bad_checks,
            }, state_path)
        step = start_step
        while step < max_updates:
            batch_ix = torch.randint(0, train_count, (min(batch_contexts, train_count),),
                                     generator=generator, device=device).cpu().numpy()
            x, ref, raw, mass = _torch_batch(train_arrays, scaler, batch_ix, columns, device)
            log_q = torch.log_softmax(ref + head(x), dim=-1)
            if objective == "exact_expected_reward":
                stat = exact_support_statistics(log_q, ref, raw, mass, scale, beta)
                loss = -stat["support_weighted_objective"].mean()
            elif objective == "analytic_q_distillation":
                stat = exact_support_statistics(log_q, ref, raw, mass, scale, beta)
                target_log = stat["analytic_target_log_policy"].detach()
                target = target_log.exp()
                kl = torch.where(target > 0, target * (target_log - log_q), torch.zeros_like(target)).sum(-1)
                loss = (mass * kl).mean()
            elif objective == "sampled_loo":
                loss, _ = sampled_loo_loss(log_q, ref, raw, mass, scale, beta, samples, generator)
            else:
                raise ValueError(f"unknown panel_tools.1 policy objective: {objective}")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite {objective} loss in {run_name} at {step+1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grads = _grad_norms(head)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % check_every != 0 and step != max_updates:
                continue
            row = _log_checkpoint(step, head, objective, beta, columns, train_meta, train_arrays, dev_meta,
                                  dev_arrays, dev_panel, scaler, scale, baseline_dev, item_ids, rm_fit_counts,
                                  device, curve, run_name, seed, variant, grad_norms=grads)
            dev_j = float(row["dev"]["support_weighted_objective"])
            dev_ndcg = float(row["dev_relevance"]["primary_user_macro"]["ndcg@10"])
            eligible = dev_ndcg >= dev_baseline_ndcg * (1.0 - threshold)
            if eligible and dev_j > best_selected_obj:
                best_selected_obj = dev_j
                best_selected_step = step
                best_selected_state = copy.deepcopy({key: value.detach().cpu().clone() for key, value in head.state_dict().items()})
                best_selected_metrics = _compact_metrics(row["dev_relevance"])
                best_selected_user_ndcg = dict(row["dev_relevance"]["per_user_primary_ndcg"])
            if dev_j > plateau_best + float(params["objective_min_delta"]):
                plateau_best = dev_j
                plateau_bad_checks = 0
            else:
                plateau_bad_checks += 1
            state = {
                "state_signature": state_signature, "updates_done": step,
                "head_state": head.state_dict(), "optimizer_state": optimizer.state_dict(),
                "generator_state": generator.get_state(), "best_selected_objective": best_selected_obj,
                "best_selected_step": best_selected_step, "best_selected_state": best_selected_state,
                "best_selected_metrics": best_selected_metrics, "best_selected_user_ndcg": best_selected_user_ndcg,
                "plateau_best": plateau_best, "plateau_bad_checks": plateau_bad_checks,
            }
            _atomic_torch_save(state, state_path)
            if step >= minimum_updates and plateau_bad_checks >= int(params["patience_checks"]):
                stop_reason = "development_exact_objective_plateau"
                break
    training_seconds = time.perf_counter() - train_start
    if best_selected_metrics is None or best_selected_user_ndcg is None:
        raise RuntimeError("no 2%-guard eligible checkpoint, including zero-init reference, was recorded")
    head.load_state_dict(best_selected_state)
    best_path = run_dir / "best.pt"
    _atomic_torch_save({
        "schema_version": "tlab_residual_support_residual_policy_head_v1", "state_dict": best_selected_state,
        "feature_columns": list(columns), "metadata": state_signature,
    }, best_path)
    best_sha = sha256_file(best_path)
    # Full distribution pairing is based on one user's one frozen event, not seeds as samples.
    baseline_users = {int(uid): float(value) for uid, value in baseline_dev["per_user_primary_ndcg"].items()}
    paired = relative_quality_guard(best_selected_user_ndcg, baseline_users, tolerance=threshold,
                                    n_bootstrap=int(params["bootstrap_draws"]), seed=seed + 19801)
    result = {
        **state_signature, "status": "completed", "updates_completed": step,
        "stop_reason": stop_reason, "minimum_updates": minimum_updates,
        "patience_checks": int(params["patience_checks"]), "objective_min_delta": float(params["objective_min_delta"]),
        "elapsed_training_and_evaluation_seconds": training_seconds,
        "best_checkpoint_path": str(best_path), "best_checkpoint_sha256": best_sha,
        "best_checkpoint_step": best_selected_step, "best_dev_support_objective": best_selected_obj,
        "best_dev_relevance": best_selected_metrics,
        "best_dev_user_ndcg": {str(k): float(v) for k, v in best_selected_user_ndcg.items()},
        "best_checkpoint_paired_2_percent_guard_vs_strong": compact_guard(paired),
        "train_user_contexts": train_count, "dev_user_contexts": len(dev_panel),
        "feature_names": [POLICY_FEATURE_NAMES[i] for i in columns],
        "train_cache_identity_sha256": train_meta["bound_identity_sha256"],
        "dev_cache_identity_sha256": dev_meta["bound_identity_sha256"],
        "actor_checkpoint_sha256": train_meta["actor"]["checkpoint_sha256"],
        "optimized_rm_sha256": train_meta["optimized_rm"]["sha256"],
        "audit_predictor_not_used_for_selection": True,
        "normalization_version": scale.version, "beta": float(beta),
        "kl_field_semantics": {"kl_policy_to_reference_conditional": "KL(q || p) conditional within fixed support",
                                "kl_policy_to_reference_mass_weighted": "m * KL(q || p), equal to full-catalog KL with unchanged tail",
                                "kl_target_to_policy": "KL(q_analytic || q), distillation direction",
                                "analytic_objective_gap": "J_analytic - J_policy on the same support and frozen reward scale"},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _log_checkpoint(step: int, head: torch.nn.Module, objective: str, beta: float, columns: Sequence[int],
                    train_meta: dict[str, Any], train_arrays: dict[str, np.ndarray], dev_meta: dict[str, Any],
                    dev_arrays: dict[str, np.ndarray], dev_panel: list[PanelTarget], scaler: TrainOnlyFeatureScaler,
                    scale: FrozenRewardScale, baseline_dev: dict[str, Any], item_ids: np.ndarray,
                    rm_fit_counts: np.ndarray, device: torch.device, curve: Any, run_name: str,
                    seed: int, variant: str, grad_norms: dict[str, float] | None = None) -> dict[str, Any]:
    train_summary = _exact_summary(head, train_arrays, scaler, columns, scale, beta, device)
    dev_summary = _exact_summary(head, dev_arrays, scaler, columns, scale, beta, device)
    dev_relevance = _evaluate_relevance(head, dev_arrays, dev_meta, dev_panel, scaler, columns,
                                        item_ids, rm_fit_counts, device)
    diagnostic_indices = np.arange(min(32, len(train_arrays["candidate_features"])), dtype=np.int64)
    x, _, _, _ = _torch_batch(train_arrays, scaler, diagnostic_indices, columns, device)
    activation = _activation_diagnostics(head, x, grad_norms or {"input_linear": 0.0, "output_linear": 0.0})
    user_values = dev_relevance["per_user_primary_ndcg"]
    baseline_ndcg = float(baseline_dev["primary_user_macro"]["ndcg@10"])
    row = {
        "step": int(step), "run_name": run_name, "objective": objective, "variant": variant,
        "seed": int(seed), "beta": float(beta), "train": train_summary, "dev": dev_summary,
        "dev_relevance": dev_relevance, "dev_guard_point_estimate_passes": float(dev_relevance["primary_user_macro"]["ndcg@10"]) >= baseline_ndcg * 0.98,
        "activation_diagnostics": activation,
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    curve_row = {key: value for key, value in row.items() if key != "dev_relevance"}
    curve_row["dev_relevance"] = _compact_metrics(dev_relevance)
    curve.write(json.dumps(curve_row, ensure_ascii=False, allow_nan=False) + "\n")
    curve.flush()
    return row


def _load_data(config_path: str | Path, root: str | Path | None, *, resume: bool) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = Path(root).resolve() if root is not None else config_path.parents[2]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_root = _resolve(project_root, config["outputs"]["run_root"])
    if not run_root.exists():
        run_root.mkdir(parents=True)
    elif not resume and any(run_root.iterdir()):
        raise FileExistsError(f"policy output exists; use --resume to continue: {run_root}")
    core_cfg = yaml.safe_load(_resolve(project_root, config["core_config"]).read_text(encoding="utf-8"))
    rm_cfg = yaml.safe_load(_resolve(project_root, config["rm_config"]).read_text(encoding="utf-8"))
    rm_root = _resolve(project_root, rm_cfg["outputs"]["root"])
    rm_meta = json.loads((rm_root / "rm_diagnostics.json").read_text(encoding="utf-8"))
    scale_payload = json.loads((rm_root / "reward_scale.json").read_text(encoding="utf-8"))
    scale = FrozenRewardScale(float(scale_payload["mean"]), float(scale_payload["std"]),
                              float(scale_payload["std_floor"]), str(scale_payload["version"]))
    cache_root = _resolve(project_root, config["cache_root"])
    cache_specs = {}
    for split in ("train", "development", "final"):
        meta_path = cache_root / split / "manifest.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing {split} cache; run `python -m tlab.residual_support.cache --split {split}` first")
        cache_specs[split] = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_features = np.memmap(cache_root / "train/candidate_features.bin", dtype=np.float32, mode="r",
                              shape=tuple(cache_specs["train"]["artifact_arrays"]["candidate_features"]["shape"]))
    scaler = TrainOnlyFeatureScaler.fit(raw_features, cache_specs["train"]["feature_names"])
    scaler_path = run_root / "scaler.json"
    if scaler_path.exists():
        existing = json.loads(scaler_path.read_text(encoding="utf-8"))
        if existing["sha256"] != scaler.digest():
            raise ValueError("existing policy scaler does not match current train-only fit")
    else:
        scaler_path.write_text(json.dumps({**scaler.to_dict(), "sha256": scaler.digest()}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    from tlab.residual_support.cache import bind_scaler_to_caches
    identities = {}
    if all((cache_root / split / "bound_identity.json").exists() for split in ("train", "development", "final")):
        for split in ("train", "development", "final"):
            identity = _cache_identity(cache_root / split / "bound_identity.json")
            if identity.scaler_sha256 != scaler.digest():
                raise ValueError("cache binding uses a different train-only scaler")
            identities[split] = identity.digest()
    elif any((cache_root / split / "bound_identity.json").exists() for split in ("train", "development", "final")):
        raise ValueError("only some caches have a scaler-bound identity")
    else:
        identities = bind_scaler_to_caches(config_path, scaler.digest(), project_root)
    for split in ("train", "development", "final"):
        cache_specs[split]["bound_identity_sha256"] = identities[split]
    train_meta, train_arrays, train_panel = _load_cache(cache_root / "train", include_full=False)
    dev_meta, dev_arrays, dev_panel = _load_cache(cache_root / "development", include_full=True)
    final_meta = json.loads((cache_root / "final/manifest.json").read_text(encoding="utf-8"))
    for metadata, split in ((train_meta, "train"), (dev_meta, "development")):
        metadata["bound_identity_sha256"] = identities[split]
    final_meta["bound_identity_sha256"] = identities["final"]
    catalog = pl.read_parquet(_resolve(project_root, core_cfg["catalog"])).sort("token_id")
    item_ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    tokens = np.asarray(catalog["token_id"].to_list(), dtype=np.int64)
    if _mapping_sha(item_ids, tokens) != train_meta["raw_token_mapping_sha256"]:
        raise ValueError("cache catalog raw-ID/token-ID mapping does not match current catalog")
    rm_fit_counts = np.asarray(rm_meta["rm_fit_item_count"]["counts_by_catalog_item"], dtype=np.int64)
    baseline_path = run_root / "development_baseline.json"
    if baseline_path.exists():
        baseline_dev = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        q_ref = np.exp(np.asarray(dev_arrays["reference_conditional_logp"], dtype=np.float32))
        baseline_dev = evaluate_full_catalog_policy(
            panel=dev_panel, full_reference_logp=dev_arrays["reference_logp_full"],
            support_indices=dev_arrays["support_indices"], support_mass=dev_arrays["support_mass"],
            policy_conditional=q_ref, item_ids=item_ids, rm_fit_counts=rm_fit_counts,
        )
        baseline_path.write_text(json.dumps(baseline_dev, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "root": project_root, "config": config, "run_root": run_root, "rm_meta": rm_meta,
        "scale": scale, "scaler": scaler, "train_meta": train_meta, "train_arrays": train_arrays,
        "train_panel": train_panel, "dev_meta": dev_meta, "dev_arrays": dev_arrays,
        "dev_panel": dev_panel, "final_meta": final_meta, "item_ids": item_ids, "rm_fit_counts": rm_fit_counts,
        "baseline_dev": baseline_dev, "identities": identities, "core_cfg": core_cfg,
    }


def run_policy_series(config_path: str | Path, root: str | Path | None = None, *, resume: bool = False) -> dict[str, Any]:
    data = _load_data(config_path, root, resume=resume)
    cfg = data["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_path = _resolve(data["root"], data["core_cfg"]["checkpoints"]["strong"])
    actor_sha_before = sha256_file(actor_path)
    model_path = _resolve(data["root"], data["rm_meta"]["models"]["optimized_rm"]["path"])
    rm_sha_before = sha256_file(model_path)
    variants: list[tuple[str, tuple[int, ...]]] = [("full", tuple(range(len(POLICY_FEATURE_NAMES))))]
    no_feedback = tuple(i for i, name in enumerate(POLICY_FEATURE_NAMES)
                        if not name.startswith("past_ratio_") and not name.startswith("personal_ratio_"))
    if bool(cfg["train"].get("no_feedback_control", True)):
        variants.append(("no_feedback", no_feedback))
    runs: list[dict[str, Any]] = []
    root_runs = data["run_root"] / "runs"
    root_runs.mkdir(parents=True, exist_ok=True)
    planned = []
    for objective in cfg["train"]["objectives"]:
        for beta in cfg["train"]["beta_values"]:
            for seed in cfg["train"]["seed_values"]:
                planned.append((objective, float(beta), int(seed), "full", variants[0][1]))
    if bool(cfg["train"].get("no_feedback_control", True)):
        for seed in cfg["train"]["seed_values"]:
            planned.append(("exact_expected_reward", float(cfg["train"]["beta_values"][-1]), int(seed), "no_feedback", no_feedback))
    for objective, beta, seed, variant, columns in planned:
        run_name = f"{objective}_b{beta:g}_seed{seed}_{variant}"
        run_dir = root_runs / run_name
        result = _run_one(
            run_dir=run_dir, run_name=run_name, objective=objective, beta=beta, seed=seed,
            variant=variant, columns=columns, config=cfg, train_meta=data["train_meta"],
            train_arrays=data["train_arrays"], dev_meta=data["dev_meta"], dev_arrays=data["dev_arrays"],
            dev_panel=data["dev_panel"], scaler=data["scaler"], scale=data["scale"],
            baseline_dev=data["baseline_dev"], item_ids=data["item_ids"], rm_fit_counts=data["rm_fit_counts"],
            device=device, resume=resume,
        )
        runs.append(result)
        print(f"{run_name}: {result['updates_completed']} updates, stop={result['stop_reason']}, "
              f"dev objective={result['best_dev_support_objective']:.6f}, "
              f"NDCG@10={result['best_dev_relevance']['primary_user_macro']['ndcg@10']:.6f}", flush=True)
    if sha256_file(actor_path) != actor_sha_before or sha256_file(model_path) != rm_sha_before:
        raise AssertionError("frozen actor or reward model changed during policy training")
    summary = {
        "schema_version": 1, "protocol_id": cfg["protocol_id"], "status": "main_fit_completed",
        "run_count": len(runs), "planned_run_count": len(planned), "runs": [
            {key: row[key] for key in ("run_name", "objective", "beta", "seed", "variant", "updates_completed",
                                       "stop_reason", "elapsed_training_and_evaluation_seconds", "best_checkpoint_step",
                                       "best_dev_support_objective", "best_dev_relevance",
                                       "best_checkpoint_paired_2_percent_guard_vs_strong", "best_checkpoint_sha256")}
            for row in runs
        ],
        "actor_checkpoint_sha256_unchanged": actor_sha_before,
        "optimized_rm_sha256_unchanged": rm_sha_before,
        "train_only_scaler_sha256": data["scaler"].digest(),
        "training_contexts": len(data["train_panel"]), "development_contexts": len(data["dev_panel"]),
        "max_updates": int(cfg["train"]["max_updates"]), "minimum_updates": int(cfg["train"]["minimum_updates"]),
        "checkpoint_rule": cfg["train"]["checkpoint_rule"],
        "all_methods_use_same_support_reference_reward_scale": True,
        "evaluation_roles": {"train": "A_RL, one context per user", "development": "validation, disjoint users",
                             "final": "test, disjoint users, retrospective only"},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = data["run_root"] / "main_fit_summary.json"
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, summary_path)
    return summary


def freeze_selection(config_path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = Path(root).resolve() if root is not None else config_path.parents[2]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_root = _resolve(project_root, cfg["outputs"]["run_root"])
    cache_root = _resolve(project_root, cfg["cache_root"])
    scaler_path = run_root / "scaler.json"
    selection_path = _resolve(project_root, "artifacts/residual_support/protocol/frozen_selection.json")
    if selection_path.exists():
        raise FileExistsError(selection_path)
    baseline = json.loads((run_root / "development_baseline.json").read_text(encoding="utf-8"))
    baseline_users = {int(uid): float(v) for uid, v in baseline["per_user_primary_ndcg"].items()}
    run_paths = sorted((run_root / "runs").glob("*/run_manifest.json"))
    expected_runs = (len(cfg["train"]["objectives"]) * len(cfg["train"]["beta_values"])
                     * len(cfg["train"]["seed_values"]) + len(cfg["train"]["seed_values"]))
    if len(run_paths) != expected_runs:
        raise RuntimeError(f"expected full main series ({expected_runs} runs); found {len(run_paths)}")
    additional_paths = [
        _resolve(project_root, value)
        for value in cfg.get("selection", {}).get("additional_run_manifests", [])
    ]
    for path in additional_paths:
        if not path.is_file():
            raise FileNotFoundError(f"registered additional candidate manifest is missing: {path}")
    train_cache_manifest = json.loads((cache_root / "train/manifest.json").read_text(encoding="utf-8"))
    scaler_sha = json.loads((run_root / "scaler.json").read_text(encoding="utf-8"))["sha256"]
    candidates = []
    for path in [*run_paths, *additional_paths]:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["actor_checkpoint_sha256"] != train_cache_manifest["actor"]["checkpoint_sha256"]:
            raise ValueError(f"registered candidate actor differs from frozen protocol: {path}")
        if run["optimized_rm_sha256"] != train_cache_manifest["optimized_rm"]["sha256"]:
            raise ValueError(f"registered candidate RM differs from frozen protocol: {path}")
        if run["scaler_sha256"] != scaler_sha:
            raise ValueError(f"registered candidate scaler differs from frozen train-only scaler: {path}")
        user_values = {int(uid): float(value) for uid, value in run["best_dev_user_ndcg"].items()}
        guard = relative_quality_guard(user_values, baseline_users,
                                      tolerance=float(cfg["train"]["dev_guard_tolerance"]),
                                      n_bootstrap=int(cfg["train"]["bootstrap_draws"]), seed=int(run["seed"]) + 22103)
        compact = compact_guard(guard)
        candidates.append({"run_name": run["run_name"], "objective": run["objective"], "beta": run["beta"],
                           "seed": run["seed"], "variant": run["variant"],
                           "best_checkpoint_step": int(run["best_checkpoint_step"]),
                           "best_checkpoint_path": run["best_checkpoint_path"],
                           "best_checkpoint_sha256": run["best_checkpoint_sha256"],
                           "dev_support_objective": run["best_dev_support_objective"],
                           "dev_ndcg_at_10": run["best_dev_relevance"]["primary_user_macro"]["ndcg@10"],
                           "candidate_source": "main_series" if path in run_paths else "registered_diagnostic",
                           "guard": compact})
    # Zero-init is retained as the reference identity in every run, but it is
    # not evidence that a trained method passed the guard.
    eligible = [row for row in candidates if row["best_checkpoint_step"] > 0 and row["guard"]["passes"]]
    if eligible:
        selected = max(eligible, key=lambda row: (row["dev_support_objective"], row["dev_ndcg_at_10"], row["run_name"]))
        selected_policy = selected["run_name"]
        selection_status = "head_selected_by_development_exact_objective_subject_to_2_percent_guard"
    else:
        selected = None
        selected_policy = "strong_gpt"
        selection_status = "no_trained_head_passed_development_guard; retain strong reference"
    selection = {
        "schema_version": 1, "protocol_id": cfg["protocol_id"], "status": "frozen_before_final_scoring",
        "selection_status": selection_status, "selected_policy": selected_policy,
        "selected_candidate": selected,
        "rule": cfg["train"]["checkpoint_rule"],
        "development_panel_event_ids_sha256": json.loads((cache_root / "development/manifest.json").read_text(encoding="utf-8"))["panel_event_ids_sha256"],
        "final_panel_event_ids_sha256": json.loads((cache_root / "final/manifest.json").read_text(encoding="utf-8"))["panel_event_ids_sha256"],
        "development_users": len(baseline_users), "candidate_count": len(candidates),
        "passing_guard_candidates": len(eligible), "all_candidates": candidates,
        "additional_registered_candidate_manifests": [str(path.relative_to(project_root)) for path in additional_paths],
        "actor_checkpoint_sha256": json.loads((cache_root / "train/manifest.json").read_text(encoding="utf-8"))["actor"]["checkpoint_sha256"],
        "optimized_rm_sha256": json.loads((cache_root / "train/manifest.json").read_text(encoding="utf-8"))["optimized_rm"]["sha256"],
        "scaler_sha256": json.loads(scaler_path.read_text(encoding="utf-8"))["sha256"],
        "cache_identity_sha256": {split: _cache_identity(cache_root / split / "bound_identity.json").digest()
                                  for split in ("train", "development", "final")},
        "support_size": int(json.loads((cache_root / "train/manifest.json").read_text(encoding="utf-8"))["support_size"]),
        "beta_values": list(cfg["train"]["beta_values"]), "seeds": list(cfg["train"]["seed_values"]),
        "prior_exposure": "retrospective; V1-full_model train/validation/test and panel_tools validation outcomes were previously inspected",
        "audit_predictor_used_for_selection": False,
        "selection_is_independent_confirmation": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return selection


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train panel_tools.1 fixed-support residual policy heads")
    parser.add_argument("--config", default="configs/residual_support/policy.yaml")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze-selection", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze_selection:
        result = freeze_selection(args.config, args.workspace)
    else:
        result = run_policy_series(args.config, args.workspace, resume=args.resume)
    print(json.dumps({key: result.get(key) for key in ("status", "selection_status", "selected_policy", "run_count",
                                                         "planned_run_count", "candidate_count", "passing_guard_candidates") if key in result},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
