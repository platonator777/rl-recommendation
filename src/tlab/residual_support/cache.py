from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import polars as pl
import torch
import yaml

from tlab.data.store import SequenceStore
from tlab.models.actor import ActorShape
from tlab.panel_tools.data import PanelTarget, history_for_target
from tlab.panel_tools.evaluator import _collate_histories, load_actor_checkpoint
from tlab.residual_support.features import (
    CANDIDATE_FEATURE_NAMES,
    POLICY_FEATURE_NAMES,
    RM_FEATURE_NAMES,
    ItemPriors,
    candidate_features,
    history_context_features,
    policy_feature_matrix,
)
from tlab.residual_support.rm import load_priors, sha256_file


@dataclass(frozen=True)
class CacheIdentity:
    protocol_id: str
    split: str
    panel_event_ids_sha256: str
    raw_token_mapping_sha256: str
    feature_schema_sha256: str
    scaler_sha256: str
    actor_checkpoint_sha256: str
    optimized_rm_sha256: str
    item_prior_sha256: str
    support_size: int
    reward_normalization: tuple[tuple[str, float | str], ...]
    support_indices_sha256: str

    def digest(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def validate_cache_identity(metadata: dict[str, Any], expected: CacheIdentity) -> None:
    expected_json = json.loads(json.dumps(asdict(expected), sort_keys=True))
    if metadata.get("identity") != expected_json or metadata.get("identity_sha256") != expected.digest():
        raise ValueError("panel_tools.1 cache identity mismatch; refusing reuse")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_panel(path: Path) -> list[PanelTarget]:
    return [PanelTarget(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _event_sha(panel: Sequence[PanelTarget]) -> str:
    return hashlib.sha256("\n".join(row.event_id for row in panel).encode("ascii")).hexdigest()


def _mapping_sha(item_ids: np.ndarray, tokens: np.ndarray) -> str:
    return hashlib.sha256("".join(f"{int(raw)}\t{int(token)}\n" for raw, token in zip(item_ids, tokens)).encode("ascii")).hexdigest()


def _collision_diagnostics(feature_rows: list[np.ndarray], reward_rows: list[np.ndarray], new_masks: list[np.ndarray], raw_reward_rows: list[np.ndarray]) -> dict[str, Any]:
    pair_total = 0
    pair_new = 0
    all_new_collision_contexts = 0
    feature_unique_rates = []
    reward_unique_rates = []
    reward_variances = []
    support_reward_variances = []
    for values, rewards, new, raw_rewards in zip(feature_rows, reward_rows, new_masks, raw_reward_rows):
        pair_total += len(new)
        pair_new += int(new.sum())
        if not new.any():
            continue
        selected = values[new]
        selected_rewards = rewards[new]
        rounded = np.round(selected, decimals=5)
        unique_features = int(np.unique(rounded, axis=0).shape[0])
        unique_rewards = int(np.unique(np.round(selected_rewards, decimals=6)).size)
        feature_unique_rates.append(unique_features / len(selected))
        reward_unique_rates.append(unique_rewards / len(selected))
        all_new_collision_contexts += int(unique_features <= 1)
        reward_variances.append(float(np.var(selected_rewards)))
        support_reward_variances.append(float(np.var(raw_rewards)))
    return {
        "candidate_pairs": pair_total,
        "personal_new_candidate_pairs": pair_new,
        "personal_new_candidate_pair_fraction": pair_new / pair_total if pair_total else None,
        "contexts_with_personal_new_candidates": len(feature_unique_rates),
        "contexts_where_all_personal_new_features_collide": all_new_collision_contexts,
        "mean_unique_feature_fraction_among_personal_new_candidates": float(np.mean(feature_unique_rates)) if feature_unique_rates else None,
        "mean_unique_predicted_reward_fraction_among_personal_new_candidates": float(np.mean(reward_unique_rates)) if reward_unique_rates else None,
        "mean_within_context_predicted_reward_variance_personal_new": float(np.mean(reward_variances)) if reward_variances else None,
        "mean_within_context_predicted_reward_variance_all_support": float(np.mean(support_reward_variances)) if support_reward_variances else None,
        "collision_definition": "feature rows rounded to 5 decimals; reward predictions rounded to 6 decimals; only personal_repeat_count_1000 == 0 candidates",
    }


def build_policy_cache(config_path: str | Path, split: str, root: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = Path(root).resolve() if root is not None else config_path.parents[2]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if split not in {"train", "development", "final"}:
        raise ValueError("cache split must be train, development, or final")
    cache_root = _resolve(project_root, cfg["cache_root"])
    destination = cache_root / split
    allowed = (project_root / "artifacts/residual_support").resolve()
    if allowed not in destination.resolve().parents:
        raise ValueError("policy cache output must remain below artifacts/residual_support")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite policy cache: {destination}")
    core_cfg = yaml.safe_load(_resolve(project_root, cfg["core_config"]).read_text(encoding="utf-8"))
    rm_cfg = yaml.safe_load(_resolve(project_root, cfg["rm_config"]).read_text(encoding="utf-8"))
    processed_dir = _resolve(project_root, core_cfg["processed_dir"])
    processed_protocol = json.loads((processed_dir / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(_resolve(project_root, core_cfg["sequence_index"]),
                                     expected_source_sha256=processed_protocol["processed_listens_sha256"])
    catalog = pl.read_parquet(_resolve(project_root, core_cfg["catalog"])).sort("token_id")
    item_ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    tokens = np.asarray(catalog["token_id"].to_list(), dtype=np.int64)
    if not np.array_equal(tokens, np.arange(3, 3 + len(tokens), dtype=np.int64)) or len(set(item_ids.tolist())) != len(item_ids):
        raise ValueError("policy cache requires validated one-to-one contiguous raw-ID/token-ID mapping")
    mapping_sha = _mapping_sha(item_ids, tokens)
    panel_path = _resolve(project_root, cfg["panels"][split])
    panel = _load_panel(panel_path)
    if not panel or len(panel) != len({row.uid for row in panel}):
        raise ValueError("policy caches require a non-empty one-event-per-user panel")
    ids_path = _resolve(project_root, rm_cfg["outputs"]["root"])
    rm_metrics = json.loads((ids_path / "rm_diagnostics.json").read_text(encoding="utf-8"))
    model_path = _resolve(project_root, rm_metrics["models"]["optimized_rm"]["path"])
    model = joblib.load(model_path)
    prior_path = _resolve(project_root, rm_metrics["item_prior_artifacts"]["policy_path"])
    priors = load_priors(prior_path, expected_mapping_sha256=mapping_sha)
    scale = json.loads((ids_path / "reward_scale.json").read_text(encoding="utf-8"))
    actor_path = _resolve(project_root, core_cfg["checkpoints"]["strong"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, actor_metadata = load_actor_checkpoint(actor_path, vocab_size=len(item_ids) + 3,
                                                  n_positions=256, n_layer=4, n_embd=256, n_head=4, device=device)
    if actor_metadata["checkpoint_sha256"] != sha256_file(actor_path):
        raise AssertionError("actor checkpoint changed while preparing policy cache")
    embeddings = actor.lm.transformer.wte.weight.detach()[3:].float().cpu().numpy()
    if embeddings.shape[0] != len(item_ids):
        raise ValueError("frozen actor item embeddings do not match catalog size")

    support_size = int(cfg["support_size"])
    batch_size = int(cfg["batch_size"])
    if support_size < 1 or support_size >= len(item_ids) or batch_size < 1:
        raise ValueError("invalid fixed support size or cache batch size")
    n_contexts = len(panel)
    n_features = len(POLICY_FEATURE_NAMES)
    destination.mkdir(parents=True)
    features = np.memmap(destination / "candidate_features.bin", dtype=np.float32, mode="w+", shape=(n_contexts, support_size, n_features))
    reference_conditional = np.memmap(destination / "reference_conditional_logp.bin", dtype=np.float32, mode="w+", shape=(n_contexts, support_size))
    support_indices = np.memmap(destination / "support_indices.bin", dtype=np.int32, mode="w+", shape=(n_contexts, support_size))
    support_mass = np.memmap(destination / "support_mass.bin", dtype=np.float32, mode="w+", shape=(n_contexts,))
    rewards = np.memmap(destination / "raw_candidate_rewards.bin", dtype=np.float32, mode="w+", shape=(n_contexts, support_size))
    full_reference = None
    if split != "train":
        full_reference = np.memmap(destination / "reference_logp_full.bin", dtype=np.float32, mode="w+", shape=(n_contexts, len(item_ids)))

    feature_probe = None
    all_names: tuple[str, ...] | None = None
    diagnostics_features: list[np.ndarray] = []
    diagnostics_rewards: list[np.ndarray] = []
    diagnostics_new: list[np.ndarray] = []
    diagnostics_all_rewards: list[np.ndarray] = []
    target_covered = 0
    started = time.perf_counter()
    actor.eval()
    rm_fit_counts = np.asarray(rm_metrics["rm_fit_item_count"]["counts_by_catalog_item"], dtype=np.int64)
    for start in range(0, n_contexts, batch_size):
        rows = panel[start:start + batch_size]
        histories = [history_for_target(store, target, int(cfg["history_events"])) for target in rows]
        input_ids, attention = _collate_histories(histories, device)
        with torch.inference_mode():
            logp = actor.next_action_log_probs(input_ids, attention)[:, 3:].float()
            logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
            ref_batch = logp.cpu().numpy().astype(np.float32, copy=False)
        for local, target in enumerate(rows):
            context_index = start + local
            ref = ref_batch[local]
            order = np.lexsort((item_ids, -ref))
            chosen = order[:support_size].astype(np.int32, copy=False)
            support_logp = ref[chosen]
            mass = float(np.exp(support_logp.astype(np.float64)).sum())
            cond = support_logp - float(np.logaddexp.reduce(support_logp.astype(np.float64)))
            user = store.users[target.user_index]
            context = history_context_features(user, target.position)
            raw_features = candidate_features(user, target.position, item_ids[chosen],
                                               catalog_indices=chosen, priors=priors)
            hstart = max(0, target.position - 1000)
            history_tokens = np.asarray(user.tokens[hstart:target.position], dtype=np.int64)
            policy_features = policy_feature_matrix(
                context, raw_features,
                rm_fit_counts=rm_fit_counts,
                candidate_catalog_indices=chosen, reference_logp=support_logp,
                item_embeddings=embeddings, history_tokens=history_tokens,
            )
            if all_names is None:
                all_names = tuple(POLICY_FEATURE_NAMES)
            if policy_features.shape != (support_size, n_features):
                raise AssertionError("policy feature output shape mismatch")
            rm_features = policy_features[:, :len(RM_FEATURE_NAMES)]
            raw_reward = np.clip(model.predict(rm_features), 0.0, 1.0).astype(np.float32)
            features[context_index] = policy_features
            reference_conditional[context_index] = cond.astype(np.float32)
            support_indices[context_index] = chosen
            support_mass[context_index] = mass
            rewards[context_index] = raw_reward
            if full_reference is not None:
                full_reference[context_index] = ref
            if split != "final" and target.token_id >= 3 and target.token_id - 3 in chosen:
                target_covered += 1
            zero_personal = raw_features[:, CANDIDATE_FEATURE_NAMES.index("personal_repeat_count_1000")] == 0
            diagnostics_features.append(policy_features.copy())
            diagnostics_rewards.append(raw_reward.copy())
            diagnostics_new.append(zero_personal)
            diagnostics_all_rewards.append(raw_reward.copy())
            if feature_probe is None:
                feature_probe = policy_features[:min(12, support_size)].copy()
        if (start // batch_size) % 50 == 0:
            print(f"{split}: cached {min(start + batch_size, n_contexts):,}/{n_contexts:,} contexts", flush=True)
    for array in (features, reference_conditional, support_indices, support_mass, rewards, full_reference):
        if array is not None:
            array.flush()
    del features, reference_conditional, support_indices, support_mass, rewards, full_reference

    event_sha = _event_sha(panel)
    feature_schema_sha = hashlib.sha256(json.dumps(POLICY_FEATURE_NAMES, separators=(",", ":")).encode()).hexdigest()
    base_metadata = {
        "schema_version": "tlab_residual_support_policy_cache_v1",
        "protocol_id": cfg["protocol_id"], "split": split,
        "panel_path": str(panel_path.relative_to(project_root)), "panel_sha256": sha256_file(panel_path),
        "panel_event_ids_sha256": event_sha, "n_contexts": n_contexts, "n_users": len({r.uid for r in panel}),
        "catalog_size": len(item_ids), "raw_token_mapping_sha256": mapping_sha,
        "catalog_mapping": [{"raw_item_id": int(r), "token_id": int(t)} for r, t in zip(item_ids, tokens)],
        "feature_schema_version": "residual_support_candidate_action_features_v1",
        "feature_names": list(POLICY_FEATURE_NAMES), "feature_schema_sha256": feature_schema_sha,
        "support_size": support_size, "support_indices_sha256": sha256_file(destination / "support_indices.bin"),
        "support_mass": "reference probability mass over actor top-K; exact unchanged full-catalog tail is retained",
        "actor": {"path": str(actor_path.relative_to(project_root)), **actor_metadata},
        "optimized_rm": {"path": str(model_path.relative_to(project_root)), "sha256": sha256_file(model_path),
                         "role": "policy reward model; not external audit predictor"},
        "item_priors": {"path": str(prior_path.relative_to(project_root)), "sha256": sha256_file(prior_path),
                        "source": priors.source_split, "cutoff_raw_exclusive": priors.cutoff_raw_exclusive},
        "reward_normalization": {"mean": float(scale["mean"]), "std": float(scale["std"]),
                                 "std_floor": float(scale["std_floor"]), "version": str(scale["version"])},
        "target_support_coverage": target_covered / n_contexts if split != "final" else None,
        "final_targets_not_scored_before_selection": split == "final",
        "reference_full_catalog_saved": split != "train",
        "artifact_arrays": {
            "candidate_features": {"shape": [n_contexts, support_size, n_features], "dtype": "float32"},
            "reference_conditional_logp": {"shape": [n_contexts, support_size], "dtype": "float32"},
            "support_indices": {"shape": [n_contexts, support_size], "dtype": "int32"},
            "support_mass": {"shape": [n_contexts], "dtype": "float32"},
            "raw_candidate_rewards": {"shape": [n_contexts, support_size], "dtype": "float32"},
            **({"reference_logp_full": {"shape": [n_contexts, len(item_ids)], "dtype": "float32"}} if split != "train" else {}),
        },
        "collision_diagnostics": _collision_diagnostics(diagnostics_features, diagnostics_rewards,
                                                         diagnostics_new, diagnostics_all_rewards),
        "runtime": {"cache_build_seconds": time.perf_counter() - started, "device": str(device), "batch_size": batch_size},
        "label": "retrospective exploratory cache; validation/test periods previously exposed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = destination / "manifest.json"
    metadata_path.write_text(json.dumps(base_metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (destination / "panel_rows.jsonl").write_text(
        "".join(json.dumps(row.to_dict(), ensure_ascii=False, allow_nan=False) + "\n" for row in panel), encoding="utf-8")
    return base_metadata


def bind_scaler_to_caches(config_path: str | Path, scaler_sha256: str, root: str | Path | None = None) -> dict[str, str]:
    config_path = Path(config_path).resolve()
    project_root = Path(root).resolve() if root is not None else config_path.parents[2]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rm_cfg = yaml.safe_load(_resolve(project_root, cfg["rm_config"]).read_text(encoding="utf-8"))
    rm_root = _resolve(project_root, rm_cfg["outputs"]["root"])
    rm_metrics = json.loads((rm_root / "rm_diagnostics.json").read_text(encoding="utf-8"))
    scale = json.loads((rm_root / "reward_scale.json").read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for split in ("train", "development", "final"):
        cache_path = _resolve(project_root, cfg["cache_root"]) / split
        manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
        payload = {
            "protocol_id": cfg["protocol_id"], "split": split,
            "panel_event_ids_sha256": manifest["panel_event_ids_sha256"],
            "raw_token_mapping_sha256": manifest["raw_token_mapping_sha256"],
            "feature_schema_sha256": manifest["feature_schema_sha256"],
            "scaler_sha256": scaler_sha256,
            "actor_checkpoint_sha256": manifest["actor"]["checkpoint_sha256"],
            "optimized_rm_sha256": manifest["optimized_rm"]["sha256"],
            "item_prior_sha256": manifest["item_priors"]["sha256"],
            "support_size": manifest["support_size"],
            "reward_normalization": scale,
            "support_indices_sha256": manifest["support_indices_sha256"],
        }
        identity = CacheIdentity(
            protocol_id=payload["protocol_id"], split=split,
            panel_event_ids_sha256=payload["panel_event_ids_sha256"],
            raw_token_mapping_sha256=payload["raw_token_mapping_sha256"],
            feature_schema_sha256=payload["feature_schema_sha256"], scaler_sha256=payload["scaler_sha256"],
            actor_checkpoint_sha256=payload["actor_checkpoint_sha256"], optimized_rm_sha256=payload["optimized_rm_sha256"],
            item_prior_sha256=payload["item_prior_sha256"], support_size=int(payload["support_size"]),
            reward_normalization=tuple(sorted((k, float(v) if isinstance(v, (float, int)) else str(v)) for k, v in payload["reward_normalization"].items())),
            support_indices_sha256=payload["support_indices_sha256"],
        )
        output = {"identity": asdict(identity), "identity_sha256": identity.digest(), "identity_payload": payload}
        target = cache_path / "bound_identity.json"
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result[split] = identity.digest()
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build sharded panel_tools.1 policy feature caches")
    parser.add_argument("--config", default="configs/residual_support/policy.yaml")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--split", choices=("train", "development", "final"), required=True)
    args = parser.parse_args(argv)
    result = build_policy_cache(args.config, args.split, args.workspace)
    print(json.dumps({"split": args.split, "n_contexts": result["n_contexts"],
                      "collision_diagnostics": result["collision_diagnostics"], "runtime": result["runtime"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
