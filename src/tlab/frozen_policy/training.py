from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tlab.residual_support.features import POLICY_FEATURE_NAMES
from tlab.residual_support.objectives import FrozenRewardScale, exact_support_statistics, sampled_loo_loss
from tlab.residual_support.scaler import TrainOnlyFeatureScaler

from .data import Frozen_FEATURE_NAMES, evaluate_conditional, open_v5_cache


class FrozenPolicyHead(nn.Module):
    """Residual 54→64→64→1 SiLU head, initialized to the reference policy."""

    def __init__(self, input_dim: int = len(Frozen_FEATURE_NAMES)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != len(Frozen_FEATURE_NAMES):
            raise ValueError("Frozen head input does not match the saved feature schema")
        return self.network(features.float()).squeeze(-1)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_scaler(path: Path) -> TrainOnlyFeatureScaler:
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in ("feature_names", "log1p_columns", "mean", "scale", "constant_columns"):
        value[key] = tuple(value[key])
    return TrainOnlyFeatureScaler(**value)


def fit_scaler(train_features: np.ndarray, path: Path) -> TrainOnlyFeatureScaler:
    if path.exists():
        scaler = load_scaler(path)
        if scaler.feature_names != tuple(Frozen_FEATURE_NAMES):
            raise ValueError("saved Frozen scaler schema mismatch")
        return scaler
    scaler = TrainOnlyFeatureScaler.fit(train_features, Frozen_FEATURE_NAMES)
    _json_dump(path, scaler.to_dict())
    return scaler


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict_conditional(head: FrozenPolicyHead, arrays: dict[str, np.ndarray],
                         row_indices: np.ndarray, scaler: TrainOnlyFeatureScaler,
                         device: torch.device, batch_size: int = 32) -> np.ndarray:
    features = arrays["candidate_features"]
    refs = arrays["reference_conditional_logp"]
    output = np.empty((len(row_indices), refs.shape[1]), dtype=np.float32)
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(row_indices), batch_size):
            local_ix = np.arange(start, min(start + batch_size, len(row_indices)))
            rows = row_indices[local_ix]
            raw = np.asarray(features[rows], dtype=np.float32)
            scaled = scaler.transform(raw, Frozen_FEATURE_NAMES)
            x = torch.from_numpy(np.ascontiguousarray(scaled)).to(device)
            ref = torch.from_numpy(np.asarray(refs[rows], dtype=np.float32).copy()).to(device)
            log_q = torch.log_softmax(ref + head(x), dim=-1)
            output[local_ix] = log_q.exp().cpu().numpy().astype(np.float32)
    return output


def predict_checkpoint(checkpoint_path: Path, arrays: dict[str, np.ndarray],
                       row_indices: np.ndarray, scaler: TrainOnlyFeatureScaler,
                       device: torch.device | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not saved.get("trained", False) or int(saved.get("step", 0)) <= 0:
        raise ValueError(f"refusing to treat an untrained/identity checkpoint as a policy: {checkpoint_path}")
    if tuple(saved.get("feature_names", ())) != tuple(Frozen_FEATURE_NAMES):
        raise ValueError("checkpoint feature names/order do not match Frozen features")
    if saved.get("scaler_sha256") != scaler.digest():
        raise ValueError("checkpoint scaler hash mismatch")
    device = device or _device()
    head = FrozenPolicyHead().to(device)
    head.load_state_dict(saved["state_dict"], strict=True)
    q = _predict_conditional(head, arrays, np.asarray(row_indices, dtype=np.int64), scaler, device)
    return q, saved


def _torch_batch(arrays: dict[str, np.ndarray], scaler: TrainOnlyFeatureScaler,
                 rows: np.ndarray, rm_index: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    raw = np.asarray(arrays["candidate_features"][rows], dtype=np.float32)
    scaled = scaler.transform(raw, Frozen_FEATURE_NAMES)
    x = torch.from_numpy(np.ascontiguousarray(scaled)).to(device)
    ref = torch.from_numpy(np.asarray(arrays["reference_conditional_logp"][rows], dtype=np.float32).copy()).to(device)
    reward = torch.from_numpy(np.asarray(arrays["rm_rewards"][rows, rm_index, :], dtype=np.float32).copy()).to(device)
    mass = torch.from_numpy(np.asarray(arrays["support_mass"][rows], dtype=np.float32).copy()).to(device)
    return x, ref, reward, mass


def _save_checkpoint(path: Path, head: FrozenPolicyHead, *, step: int, objective: str, beta: float,
                     seed: int, reward_model: str, scaler: TrainOnlyFeatureScaler,
                     trained: bool = True) -> None:
    payload = {
        "schema_version": "tlab_frozen_policy_policy_head_v1",
        "trained": bool(trained), "step": int(step),
        "objective": objective, "beta": float(beta), "seed": int(seed),
        "reward_model": reward_model,
        "feature_names": list(Frozen_FEATURE_NAMES),
        "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        "scaler_sha256": scaler.digest(),
        "head_architecture": "Linear(54,64)-SiLU-Linear(64,64)-SiLU-Linear(64,1), zero-init final layer",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _mean_mass_kl(log_q: np.ndarray, log_ref: np.ndarray, mass: np.ndarray) -> float:
    q = np.asarray(log_q, dtype=np.float64)
    ref = np.asarray(log_ref, dtype=np.float64)
    kl = np.sum(np.exp(q) * (q - ref), axis=1)
    return float(np.mean(np.asarray(mass, dtype=np.float64) * kl))


def train_one(*, root: Path, artifact_root: Path, run_name: str, objective: str,
              beta: float, seed: int, reward_model: str,
              scaler: TrainOnlyFeatureScaler, scale: FrozenRewardScale,
              split_manifest: dict[str, Any], max_updates: int = 3000,
              check_every: int = 500, resume: bool = True) -> dict[str, Any]:
    if objective not in {"sampled_loo", "exact_expected_reward"}:
        raise ValueError(f"unsupported Frozen objective: {objective}")
    if reward_model not in {"old_deployed", "same_split_control", "prefix_affinity"}:
        raise ValueError(f"unsupported Frozen reward model: {reward_model}")
    run_dir = artifact_root / "policy" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        done = json.loads(manifest_path.read_text(encoding="utf-8"))
        if done.get("status") == "complete":
            return done

    train_cache, _ = open_v5_cache(artifact_root / "cache", "train")
    common_cache, _ = open_v5_cache(artifact_root / "cache", "common")
    _, train_old, train_panel = _load_existing_cache(root / "artifacts/residual_support/cache/train")
    _, common_old, common_panel = _load_existing_cache(root / "artifacts/policy_diagnostics/cache/common_open_6655")
    dev_uid_set = set(int(x) for x in split_manifest["development_uids"])
    dev_ix = np.asarray([i for i, row in enumerate(common_panel) if int(row.uid) in dev_uid_set], dtype=np.int64)
    train_arrays = {**train_old, "candidate_features": train_cache["candidate_features"],
                    "rm_rewards": train_cache["rm_rewards"]}
    dev_arrays = {**common_old, "candidate_features": common_cache["candidate_features"],
                  "rm_rewards": common_cache["rm_rewards"]}
    rm_index = {"old_deployed": 0, "same_split_control": 1, "prefix_affinity": 2}[reward_model]
    device = _device()
    params = {"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_contexts": 32,
              "samples_per_context": 16, "max_updates": int(max_updates), "check_every": int(check_every)}
    signature = {
        "run_name": run_name, "objective": objective, "beta": float(beta), "seed": int(seed),
        "reward_model": reward_model, "feature_schema": list(Frozen_FEATURE_NAMES),
        "scaler_sha256": scaler.digest(), "updates": int(max_updates),
        "batch_contexts": params["batch_contexts"], "samples_per_context": params["samples_per_context"],
        "fixed_support": 500, "reward_scale": {"mean": scale.mean, "std": scale.std,
                                                  "std_floor": scale.std_floor, "version": scale.version},
    }
    state_path = run_dir / "resume_state.pt"
    curve_path = run_dir / "dev_curve.jsonl"
    smoke_path = run_dir / "smoke.json"

    if state_path.exists():
        if not resume:
            raise FileExistsError(f"partial run exists; use resume: {run_dir}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError(f"resume signature mismatch for {run_name}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        head = FrozenPolicyHead().to(device)
        head.load_state_dict(state["head_state"])
        optimizer = torch.optim.AdamW(head.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
        optimizer.load_state_dict(state["optimizer_state"])
        generator = torch.Generator(device=device)
        generator.set_state(state["generator_state"].cpu())
        step = int(state["step"])
        best_ndcg = float(state["best_ndcg"])
        best_kl = float(state["best_kl"])
        best_step = int(state["best_step"])
        print(f"resume {run_name} from update {step}", flush=True)
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        head = FrozenPolicyHead().to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
        generator = torch.Generator(device=device).manual_seed(int(seed) + 991)
        step = 0
        best_ndcg = -math.inf
        best_kl = math.inf
        best_step = -1
        resume_payload = {"signature": signature, "step": 0,
                          "head_state": head.state_dict(), "optimizer_state": optimizer.state_dict(),
                          "generator_state": generator.get_state(), "best_ndcg": best_ndcg,
                          "best_kl": best_kl, "best_step": best_step}
        torch.save(resume_payload, state_path)
        # A zero-init head must reproduce the conditional reference exactly.
        probe = torch.from_numpy(np.asarray(train_arrays["reference_conditional_logp"][:2], dtype=np.float32).copy()).to(device)
        xprobe = torch.zeros((2, probe.shape[1], len(Frozen_FEATURE_NAMES)), dtype=torch.float32, device=device)
        zero_delta = head(xprobe)
        if not torch.allclose(zero_delta, torch.zeros_like(zero_delta), atol=0, rtol=0):
            raise AssertionError("zero-init Frozen head does not equal the strong reference")
        if not torch.allclose(torch.softmax(probe + zero_delta, -1), torch.softmax(probe, -1), atol=1e-7, rtol=0):
            raise AssertionError("zero-init conditional policy is not the reference")
        print(f"start {run_name}", flush=True)

    train_count = len(train_panel)
    started = time.perf_counter()
    last_smoke_written = smoke_path.exists()
    while step < max_updates:
        batch_ix = torch.randint(0, train_count, (params["batch_contexts"],),
                                 generator=generator, device=device).cpu().numpy()
        x, ref, raw, mass = _torch_batch(train_arrays, scaler, batch_ix, rm_index, device)
        log_q = torch.log_softmax(ref + head(x), dim=-1)
        if objective == "sampled_loo":
            loss, _ = sampled_loo_loss(log_q, ref, raw, mass, scale, beta,
                                       params["samples_per_context"], generator)
        else:
            stats = exact_support_statistics(log_q, ref, raw, mass, scale, beta)
            loss = -stats["support_weighted_objective"].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {objective} loss at {run_name}, step {step+1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite Frozen head gradient at {run_name}, step {step+1}")
        optimizer.step()
        step += 1

        if step == 50 and not last_smoke_written:
            params_finite = all(bool(torch.isfinite(p).all()) for p in head.parameters())
            if not params_finite:
                raise FloatingPointError(f"non-finite parameters during 50-update smoke: {run_name}")
            _json_dump(smoke_path, {"status": "passed_and_continued_in_same_run", "updates": 50,
                                    "finite_loss": float(loss.detach().cpu()),
                                    "gradient_norm": float(grad_norm.detach().cpu()),
                                    "parameters_finite": params_finite,
                                    "identity_weights_replaced_by_training": True})
            last_smoke_written = True

        if step % check_every != 0 and step != max_updates:
            continue
        dev_q = _predict_conditional(head, dev_arrays, dev_ix, scaler, device)
        dev_metrics = evaluate_conditional(common_panel, dev_arrays, dev_ix, dev_q, _catalog_ids(root))
        log_ref = np.asarray(dev_arrays["reference_conditional_logp"][dev_ix], dtype=np.float64)
        log_q_np = np.log(np.maximum(dev_q.astype(np.float64), 1e-30))
        dev_kl = _mean_mass_kl(log_q_np, log_ref, np.asarray(dev_arrays["support_mass"][dev_ix]))
        row = {"step": step, "dev_ndcg10": dev_metrics["ndcg10"], "dev_hr10": dev_metrics["hr10"],
               "dev_mass_weighted_kl": dev_kl, "dev_hits_at_10": dev_metrics["hits_at_10"],
               "dev_n": dev_metrics["n"], "training_loss": float(loss.detach().cpu()),
               "training_gradient_norm": float(grad_norm.detach().cpu())}
        with curve_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        score = float(dev_metrics["ndcg10"])
        if score > best_ndcg + 1e-12 or (abs(score - best_ndcg) <= 1e-12 and dev_kl < best_kl):
            best_ndcg, best_kl, best_step = score, dev_kl, step
            _save_checkpoint(run_dir / "best.pt", head, step=step, objective=objective, beta=beta,
                             seed=seed, reward_model=reward_model, scaler=scaler)
        state = {"signature": signature, "step": step,
                 "head_state": head.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "generator_state": generator.get_state(), "best_ndcg": best_ndcg,
                 "best_kl": best_kl, "best_step": best_step}
        tmp = state_path.with_suffix(".pt.tmp")
        torch.save(state, tmp); tmp.replace(state_path)
        print(f"{run_name}: {step}/{max_updates} dev_ndcg={score:.6f} KL={dev_kl:.5f}", flush=True)

    if step < 1 or best_step < 1 or not (run_dir / "best.pt").exists():
        raise RuntimeError(f"no trained dev-selected checkpoint was produced for {run_name}")
    _save_checkpoint(run_dir / "last.pt", head, step=step, objective=objective, beta=beta,
                     seed=seed, reward_model=reward_model, scaler=scaler)
    best_payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    if not best_payload.get("trained") or best_payload["step"] < 1:
        raise AssertionError("selected policy weights are not trained weights")
    summary = {
        **signature, "status": "complete", "updates_completed": step,
        "best_checkpoint_step": int(best_step), "best_dev_ndcg10": float(best_ndcg),
        "best_dev_mass_weighted_kl": float(best_kl),
        "best_checkpoint_path": str((run_dir / "best.pt").relative_to(root)),
        "best_checkpoint_sha256": __import__("hashlib").sha256((run_dir / "best.pt").read_bytes()).hexdigest(),
        "last_checkpoint_path": str((run_dir / "last.pt").relative_to(root)),
        "elapsed_training_and_dev_scoring_seconds": round(time.perf_counter() - started, 3),
        "checkpoint_rule": "best trained checkpoint by development user-macro NDCG@10; ties by lower mass-weighted KL; step 0 excluded",
        "smoke": json.loads(smoke_path.read_text(encoding="utf-8")),
    }
    _json_dump(manifest_path, summary)
    return summary


def _load_existing_cache(path: Path):
    from .data import _load_cache
    return _load_cache(path)


def _catalog_ids(root: Path) -> np.ndarray:
    import polars as pl
    return np.asarray(pl.read_parquet(root / "data/processed_full_model_tminus14/catalog.parquet")
                      .sort("token_id")["item_id"].to_list(), dtype=np.int64)

