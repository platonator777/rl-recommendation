from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# CUBLAS reads this setting when its handle is first initialized. Keep it
# stable for exact interruption/resume comparisons on CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import Tensor

from tlab.config import load_config
from tlab.data.catalog import PAD_ID
from tlab.data.manifest import sha256_file, write_json_atomic
from tlab.data.store import SequenceStore
from tlab.evaluation.behavior import exposure_metrics, repeat_rate
from tlab.evaluation.metrics import paired_user_bootstrap, single_target_metrics
from tlab.evaluation.rewards import RewardScale, fit_reward_scale
from tlab.evaluation.policy_panel import _load_reward_model
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.train.checkpoint import load_training_checkpoint, save_training_checkpoint, source_tree_sha256
from tlab.train.objectives import (
    full_categorical_kl,
    full_catalog_support_components,
    leave_one_out_advantages,
    prepare_reward_objective,
)


def _json(path: Path, value: Any) -> None:
    write_json_atomic(path, value)


def _configure_reproducibility(seed: int) -> None:
    """Make fresh full_model runs reproducible and resumes continue saved RNG state."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # GPT-2 may dispatch to fused SDPA kernels with nondeterministic backward
    # reductions on some CUDA/PyTorch combinations. The math kernel is stable.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def _register_run(artifacts: Path, manifest: dict[str, Any]) -> None:
    target = artifacts / "run_registry.json"
    registry = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {
        "protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "runs": []}
    run_id = str(manifest["run_id"])
    entry = {key: manifest.get(key) for key in (
        "run_id", "stage", "method", "support_mode", "beta", "seed", "reward_head",
        "objective_version", "advantage_reward_units", "reward_scale",
        "status", "run_dir", "latest_update", "updates_planned", "training_seconds",
        "stop_reason", "last_checkpoint_sha256", "processed_listens_sha256",
        "catalog_sha256", "reward_checkpoint_sha256", "actor_checkpoint_sha256")}
    rows = [row for row in registry.get("runs", []) if row.get("run_id") != run_id]
    rows.append(entry)
    registry["runs"] = sorted(rows, key=lambda row: str(row["run_id"]))
    registry["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _json(target, registry)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _root_and_processed(config_path: str | Path) -> tuple[Any, Path, Path]:
    config = load_config(config_path)
    root = config.path.parent.parent
    processed = Path(config.data.get("processed_dir", "data/processed"))
    if not processed.is_absolute():
        processed = root / processed
    artifacts = Path(config.project.get("artifacts_dir", "artifacts/full_model"))
    if not artifacts.is_absolute():
        artifacts = root / artifacts
    return config, processed, artifacts


def _configured_artifact_root(config, key: str, fallback: Path) -> Path:
    root = config.path.parent.parent
    value = Path(config.project.get(key, fallback))
    return value if value.is_absolute() else root / value


def _panels_directory(config, artifacts: Path) -> Path:
    fold = "main" if int(config.dataset.get("cutoff_shift_days", 0)) == 0 else f"tminus{int(config.dataset['cutoff_shift_days'])}"
    root = _configured_artifact_root(config, "panels_source_root", artifacts / "panels")
    return root / fold


def _cache_root(config, artifacts: Path) -> Path:
    return _configured_artifact_root(config, "cache_source_root", artifacts / "cache")


def _validate_objective_resume(metadata: dict[str, Any]) -> None:
    if metadata.get("objective_version") != "normalized_reward_v1":
        raise ValueError("resume objective_version is missing or incompatible with normalized_reward_v1")
    if metadata.get("advantage_reward_units") != "normalized":
        raise ValueError("resume advantages were not saved in normalized reward units")


def _load_store(processed: Path) -> tuple[SequenceStore, dict[str, Any]]:
    protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    cache_dir = processed / "sequence_index"
    if not (cache_dir / "index.json").exists():
        store = SequenceStore.from_parquet(processed / "listens.parquet")
        store.save_cache(cache_dir, source_sha256=protocol["processed_listens_sha256"])
    store = SequenceStore.load_cache(cache_dir, expected_source_sha256=protocol["processed_listens_sha256"])
    return store, protocol


def _entry(store: SequenceStore, user_index: int, position: int, panel_index: int) -> dict[str, Any]:
    user = store.users[user_index]
    key = f"{user.uid}:{position}"
    return {
        "row": panel_index,
        "user_index": int(user_index),
        "uid": int(user.uid),
        "position": int(position),
        "anchor_id": _sha_text(key),
        "target_token_id": int(user.tokens[position]),
        "target_raw_item_id": int(user.raw_item_ids[position]),
    }


def create_panels(config_path: str | Path, *, force: bool = False) -> dict[str, Any]:
    config, processed, artifacts = _root_and_processed(config_path)
    store, protocol = _load_store(processed)
    panel_dir = _panels_directory(config, artifacts)
    panel_path = panel_dir / "panels.json"
    if panel_path.exists() and not force:
        saved = json.loads(panel_path.read_text(encoding="utf-8"))
        if saved.get("processed_listens_sha256") != protocol["processed_listens_sha256"]:
            raise ValueError("existing full_model panel manifest belongs to different processed data")
        training_cfg = config.training
        expected_signature = {
            "protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1",
            "panel_seed": int(training_cfg.get("rl_context_panel_seed", 30017)),
            "validation_seed": int(training_cfg.get("rl_validation_seed", 17)),
            "validation_split_seed": int(training_cfg.get("rl_panel_split_seed", 30018)),
            "requested_sizes": {
                "train": int(training_cfg.get("rl_train_panel_users", 2048)),
                "calibration": int(training_cfg.get("rl_calibration_panel_users", 512)),
                "evaluation": int(training_cfg.get("rl_eval_panel_users", 1024)),
                "development": int(training_cfg.get("rl_eval_panel_users", 1024)) // 2,
                "confirmation": int(training_cfg.get("rl_eval_panel_users", 1024))
                - int(training_cfg.get("rl_eval_panel_users", 1024)) // 2,
            },
        }
        mismatches = {key: {"saved": saved.get(key), "expected": value}
                      for key, value in expected_signature.items() if saved.get(key) != value}
        if mismatches:
            raise ValueError(f"existing full_model panel signature differs from current preregistration: {mismatches}")
        return saved

    training_cfg = config.training
    requested_train = int(training_cfg.get("rl_train_panel_users", 2048))
    requested_cal = int(training_cfg.get("rl_calibration_panel_users", 512))
    requested_eval = int(training_cfg.get("rl_eval_panel_users", 1024))
    panel_seed = int(training_cfg.get("rl_context_panel_seed", 30017))
    eligible_a = [i for i, user in enumerate(store.users) if len(user.rl_anchors) > 0]
    ordered_a = sorted(eligible_a, key=lambda i: (_sha_text(f"{panel_seed}:A:{store.users[i].uid}"), store.users[i].uid))
    actual_train = min(requested_train, max(0, len(ordered_a) - 1))
    actual_cal = min(requested_cal, max(0, len(ordered_a) - actual_train))
    if actual_train < 1 or actual_cal < 1:
        raise ValueError("not enough distinct A_RL users for disjoint training and calibration panels")
    train_users = ordered_a[:actual_train]
    cal_users = ordered_a[actual_train:actual_train + actual_cal]
    train_rng = random.Random(panel_seed + 1)
    cal_rng = random.Random(panel_seed + 2)
    train = [_entry(store, i, train_rng.choice(store.users[i].rl_anchors), row)
             for row, i in enumerate(train_users)]
    calibration = [_entry(store, i, cal_rng.choice(store.users[i].rl_anchors), row)
                   for row, i in enumerate(cal_users)]

    forbidden_uids = {store.users[i].uid for i in train_users + cal_users}
    validation = store.evaluation_targets("validation", per_user_limit=1,
                                          seed=int(training_cfg.get("rl_validation_seed", 17)))
    val_rows = [(i, p) for i, p in validation if store.users[i].uid not in forbidden_uids]
    split_seed = int(training_cfg.get("rl_panel_split_seed", 30018))
    val_rows.sort(key=lambda pair: (_sha_text(f"{split_seed}:V:{store.users[pair[0]].uid}"),
                                   store.users[pair[0]].uid))
    actual_eval = min(requested_eval, len(val_rows))
    if actual_eval < 2:
        raise ValueError("not enough validation users outside A_RL panels for development and confirmation")
    val_rows = val_rows[:actual_eval]
    split_at = actual_eval // 2
    development = [_entry(store, i, p, row) for row, (i, p) in enumerate(val_rows[:split_at])]
    confirmation = [_entry(store, i, p, split_at + row)
                    for row, (i, p) in enumerate(val_rows[split_at:])]
    groups = {name: {int(x["uid"]) for x in rows} for name, rows in {
        "train": train, "calibration": calibration,
        "development": development, "confirmation": confirmation,
    }.items()}
    for left, right in itertools.combinations(groups, 2):
        if groups[left] & groups[right]:
            raise AssertionError(f"panel user overlap: {left} x {right}")
    for rows in (train, calibration, development, confirmation):
        for row in rows:
            user = store.users[int(row["user_index"])]
            if int(user.tokens[int(row["position"])]) != int(row["target_token_id"]):
                raise AssertionError("panel target failed causal alignment check")
    payload = {
        "protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1",
        "processed_listens_sha256": protocol["processed_listens_sha256"],
        "catalog_sha256": protocol["catalog_sha256"],
        "panel_seed": panel_seed,
        "validation_seed": int(training_cfg.get("rl_validation_seed", 17)),
        "validation_split_seed": split_seed,
        "panel_contract": "distinct users across train/calibration/development/confirmation; one causal anchor per user",
        "requested_sizes": {"train": requested_train, "calibration": requested_cal,
                            "evaluation": requested_eval, "development": requested_eval // 2,
                            "confirmation": requested_eval - requested_eval // 2},
        "actual_sizes": {"train": len(train), "calibration": len(calibration),
                         "evaluation": len(development) + len(confirmation),
                         "development": len(development), "confirmation": len(confirmation)},
        "reduced_for_data_availability": {
            "train": len(train) < requested_train,
            "calibration": len(calibration) < requested_cal,
            "evaluation": len(development) + len(confirmation) < requested_eval,
        },
        "train": train, "calibration": calibration,
        "development": development, "confirmation": confirmation,
        "panel_ids_sha256": {
            name: _sha_text("\n".join(row["anchor_id"] for row in rows))
            for name, rows in {"train": train, "calibration": calibration,
                               "development": development, "confirmation": confirmation}.items()
        },
        "user_overlap": False,
    }
    panel_dir.mkdir(parents=True, exist_ok=True)
    _json(panel_path, payload)
    return payload


def _collate(histories: Sequence[Sequence[int]], device: torch.device) -> tuple[Tensor, Tensor]:
    width = max(len(history) for history in histories)
    ids = torch.full((len(histories), width), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, history in enumerate(histories):
        ids[row, :len(history)] = torch.as_tensor(history, dtype=torch.long, device=device)
        mask[row, :len(history)] = 1
    return ids, mask


def _histories(store: SequenceStore, rows: Sequence[dict[str, Any]], history_length: int) -> list[list[int]]:
    return [store.users[int(row["user_index"])].history(int(row["position"]), history_length)
            for row in rows]


def _features(config, processed: Path, variant: str) -> dict[int, tuple[float, bool, int]]:
    import polars as pl
    path = processed / f"rm_item_features_{variant}.parquet"
    frame = pl.read_parquet(path)
    return {int(row["token_id"]): (float(row.get("rm_duration_seconds") or 0.0),
                                   bool(row["rm_duration_missing"]),
                                   int(row["rm_training_support"] or 0))
            for row in frame.iter_rows(named=True)}


def _cache_path(artifacts: Path, processed: Path, fold: str, variant: str, head: str) -> Path:
    return artifacts / "cache" / fold / f"{variant}_{head}"


def build_reward_cache(config_path: str | Path, *, reward_checkpoint: str | Path | None = None,
                       variant: str = "fit", reward_head: str = "ratio", force: bool = False) -> dict[str, Any]:
    import polars as pl

    config, processed, artifacts = _root_and_processed(config_path)
    store, data_protocol = _load_store(processed)
    panels = create_panels(config_path)
    fold = "main" if int(config.dataset.get("cutoff_shift_days", 0)) == 0 else f"tminus{int(config.dataset['cutoff_shift_days'])}"
    checkpoint = Path(reward_checkpoint or (
        artifacts / "runs" / "rm_reference_fit_seed17" / "best.pt" if fold == "main"
        else artifacts / "runs" / fold / "rm_fit_seed17" / "best.pt"
    ))
    if not checkpoint.is_absolute():
        checkpoint = config.path.parent.parent / checkpoint
    if not checkpoint.is_file():
        # The main reference model is intentionally outside artifacts/full_model.
        checkpoint = config.path.parent.parent / (reward_checkpoint or "artifacts/reference/runs/rm_reference_fit_seed17/best.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"reward checkpoint missing: {checkpoint}")
    checkpoint_hash = sha256_file(checkpoint)
    feature_hash = sha256_file(processed / f"rm_item_features_{variant}.parquet")
    source_hash = source_tree_sha256(config.path.parent.parent / "src" / "tlab")
    cache_dir = _cache_root(config, artifacts) / fold / f"{variant}_{reward_head}"
    manifest_path = cache_dir / "cache_manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_files = [m["file"] for m in manifest.get("matrices", {}).values()]
        expected_files += [manifest.get("fit_support_group", {}).get("file"),
                           manifest.get("unsupported_mask", {}).get("file"), "reward_scale.json"]
        if (manifest.get("cache_version") == 2
                and manifest.get("reward_checkpoint_sha256") == checkpoint_hash
                and manifest.get("processed_listens_sha256") == data_protocol["processed_listens_sha256"]
                and manifest.get("processed_protocol_sha256") == sha256_file(processed / "protocol.json")
                and manifest.get("panel_ids_sha256") == panels["panel_ids_sha256"]
                and manifest.get("reward_head") == reward_head
                and manifest.get("reward_feature_sha256") == feature_hash
                and manifest.get("source_tree_sha256") == source_hash
                and all(filename and (cache_dir / filename).is_file()
                        and sha256_file(cache_dir / filename) == expected_hash
                        for filename, expected_hash in [
                            (m["file"], m["sha256"]) for m in manifest.get("matrices", {}).values()
                        ] + [
                            (manifest.get("fit_support_group", {}).get("file"),
                             manifest.get("fit_support_group", {}).get("sha256")),
                            (manifest.get("unsupported_mask", {}).get("file"),
                             manifest.get("unsupported_mask", {}).get("sha256")),
                            ("reward_scale.json", manifest.get("scale_sha256")),
                        ])):
            return manifest
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loaded_features = _load_reward_model(config, checkpoint, variant, device)
    features = _features(config, processed, variant)
    if loaded_features and loaded_features != features:
        # Mapping equality across numpy/polars scalar wrappers is not always reliable;
        # require the processed protocol hash and source file checksum below instead.
        if sha256_file(processed / f"rm_item_features_{variant}.parquet") != data_protocol[
                "rm_item_features_by_partition"][variant]["sha256"]:
            raise ValueError("RM item feature file differs from the processed-data protocol")
    catalog_size = int(data_protocol["catalog_size"])
    history_length = int(config.dataset.get("history_length", 201))
    item_tokens = torch.arange(3, 3 + catalog_size, device=device).unsqueeze(0)
    duration_vector = torch.tensor([features.get(int(t), (0.0, True, 0))[0] for t in item_tokens[0].tolist()],
                                   dtype=torch.float32, device=device).unsqueeze(0)
    missing_vector = torch.tensor([features.get(int(t), (0.0, True, 0))[1] for t in item_tokens[0].tolist()],
                                  dtype=torch.bool, device=device).unsqueeze(0)
    support = np.asarray([features.get(int(t), (0.0, True, 0))[2] for t in item_tokens[0].tolist()],
                         dtype=np.int32)
    item_tokens_np = np.arange(3, 3 + catalog_size, dtype=np.int64)
    groups = np.select([support == 0, support < 10, support < 100], [0, 1, 2], default=3).astype(np.uint8)
    duration_scale = float(data_protocol.get("seconds_scale_f_rm_p95", 1.0))
    matrices: dict[str, dict[str, Any]] = {}
    score_sections = {
        "train": panels["train"],
        "evaluation": panels["development"] + panels["confirmation"],
    }
    for panel_name, rows in score_sections.items():
        output = cache_dir / f"{panel_name}.npy"
        mmap = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32,
                                         shape=(len(rows), catalog_size))
        histories = _histories(store, rows, history_length)
        model.eval()
        started = time.perf_counter()
        for start in range(0, len(rows), 8):
            batch_histories = histories[start:start + 8]
            ids, attention = _collate(batch_histories, device)
            candidates = item_tokens.expand(len(batch_histories), -1)
            durations = duration_vector.expand(len(batch_histories), -1)
            duration_missing = missing_vector.expand(len(batch_histories), -1)
            repeat = torch.as_tensor(np.stack([
                np.isin(item_tokens_np, np.asarray(history[-20:], dtype=np.int64))
                for history in batch_histories
            ]), dtype=torch.bool, device=device)
            prediction = model.score_catalog(ids, attention, candidates, durations,
                                            duration_missing, repeat,
                                            chunk_size=int(config.training.get("rl_reward_catalog_chunk_size", 512)))
            values = prediction["ratio"] if reward_head == "ratio" else prediction["seconds_scaled"] * duration_scale
            mmap[start:start + len(batch_histories)] = values.float().cpu().numpy()
            if (start // 8) % 64 == 0:
                mmap.flush()
                print(f"cache {variant}/{reward_head} {panel_name}: {min(start + 8, len(rows))}/{len(rows)}",
                      flush=True)
        mmap.flush()
        matrices[panel_name] = {"file": output.name, "sha256": sha256_file(output),
                                "shape": [len(rows), catalog_size],
                                "elapsed_seconds": time.perf_counter() - started,
                                "bytes": output.stat().st_size}
        del mmap
    train_rows = panels["train"]
    cal_rows = panels["calibration"]
    rng = np.random.default_rng(int(config.training.get("rl_reward_scale_seed", 90017))
                                + (0 if reward_head == "ratio" else 1))
    calibration_values: list[float] = []
    model.eval()
    with torch.no_grad():
        cal_histories = _histories(store, cal_rows, history_length)
        for start in range(0, len(cal_rows), 32):
            histories_batch = cal_histories[start:start + 32]
            ids, attention = _collate(histories_batch, device)
            selected_indices = rng.integers(0, catalog_size, size=len(histories_batch))
            candidates = torch.tensor(selected_indices + 3, dtype=torch.long, device=device)
            durations = duration_vector[0, selected_indices]
            missing = missing_vector[0, selected_indices]
            repeats = torch.as_tensor([
                int(int(index + 3) in set(history[-20:])) for index, history in zip(selected_indices, histories_batch)
            ], dtype=torch.bool, device=device)
            prediction = model(ids, attention, candidates, durations, missing, repeats)
            values = prediction["ratio"] if reward_head == "ratio" else prediction["seconds_scaled"] * duration_scale
            calibration_values.extend(float(x) for x in values.cpu().tolist())
    scale = fit_reward_scale(calibration_values)
    # Preserve the frozen calibration scale; all future runs reuse this exact file.
    scale_path = cache_dir / "reward_scale.json"
    _json(scale_path, {"reward_scale": {"mean": scale.mean, "std": scale.std,
                                        "sample_count": scale.sample_count},
                       "panel_ids_sha256": panels["panel_ids_sha256"]["calibration"],
                       "sampling_seed": int(config.training.get("rl_reward_scale_seed", 90017)),
                       "action_sampling": "one uniform-catalog item per unique A_RL calibration user"})
    group_path = cache_dir / "fit_support_group.npy"
    np.save(group_path, groups, allow_pickle=False)
    mask_path = cache_dir / "unsupported_mask.npy"
    np.save(mask_path, (support == 0), allow_pickle=False)
    manifest = {
        "cache_version": 2, "protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "fold": fold,
        "variant": variant, "reward_head": reward_head, "reward_checkpoint": str(checkpoint.resolve()),
        "reward_checkpoint_sha256": checkpoint_hash,
        "processed_listens_sha256": data_protocol["processed_listens_sha256"],
        "processed_protocol_sha256": sha256_file(processed / "protocol.json"),
        "catalog_sha256": data_protocol["catalog_sha256"], "panel_ids_sha256": panels["panel_ids_sha256"],
        "reward_feature_sha256": feature_hash, "source_tree_sha256": source_hash,
        "catalog_size": catalog_size, "history_length": history_length,
        "duration_scale_seconds": duration_scale, "reward_scale": {"mean": scale.mean,
            "std": scale.std, "sample_count": scale.sample_count},
        "scale_sha256": sha256_file(scale_path), "support_groups": [0, 1, 2, 3],
        "matrices": matrices,
        "fit_support_group": {"file": group_path.name, "sha256": sha256_file(group_path)},
        "unsupported_mask": {"file": mask_path.name, "sha256": sha256_file(mask_path)},
        "rm_audit_policy_scores_used_for_selection": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _json(manifest_path, manifest)
    return manifest


def _load_cache(config_path: str | Path, variant: str, head: str,
                reward_checkpoint: str | Path | None = None) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, np.ndarray]]:
    config, processed, artifacts = _root_and_processed(config_path)
    panels = create_panels(config_path)
    fold = "main" if int(config.dataset.get("cutoff_shift_days", 0)) == 0 else f"tminus{int(config.dataset['cutoff_shift_days'])}"
    cache_dir = _cache_root(config, artifacts) / fold / f"{variant}_{head}"
    manifest_path = cache_dir / "cache_manifest.json"
    if config.project.get("cache_source_read_only", False):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"read-only reward cache is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
        expected_checkpoint = Path(reward_checkpoint) if reward_checkpoint else Path(manifest["reward_checkpoint"])
        if not expected_checkpoint.is_absolute():
            expected_checkpoint = config.path.parent.parent / expected_checkpoint
        checks = {
            "reward_checkpoint_sha256": sha256_file(expected_checkpoint),
            "processed_listens_sha256": data_protocol["processed_listens_sha256"],
            "processed_protocol_sha256": sha256_file(processed / "protocol.json"),
            "catalog_sha256": data_protocol["catalog_sha256"],
            "panel_ids_sha256": panels["panel_ids_sha256"],
            "reward_head": head,
        }
        mismatches = {key: {"saved": manifest.get(key), "expected": value}
                      for key, value in checks.items() if manifest.get(key) != value}
        if mismatches:
            raise ValueError(f"read-only cache provenance mismatch: {mismatches}")
        files = [(item["file"], item["sha256"]) for item in manifest.get("matrices", {}).values()]
        files += [(manifest.get("fit_support_group", {}).get("file"),
                   manifest.get("fit_support_group", {}).get("sha256")),
                  (manifest.get("unsupported_mask", {}).get("file"),
                   manifest.get("unsupported_mask", {}).get("sha256")),
                  ("reward_scale.json", manifest.get("scale_sha256"))]
        for filename, checksum in files:
            if not filename or not checksum or not (cache_dir / filename).is_file() \
                    or sha256_file(cache_dir / filename) != checksum:
                raise ValueError(f"read-only reward cache failed integrity check: {filename}")
    else:
        build_reward_cache(config_path, reward_checkpoint=reward_checkpoint, variant=variant, reward_head=head)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["panel_ids_sha256"] != panels["panel_ids_sha256"]:
        raise ValueError("reward cache panel IDs differ from the frozen full_model panel manifest")
    arrays = {name: np.load(cache_dir / details["file"], mmap_mode="r", allow_pickle=False)
              for name, details in manifest["matrices"].items()}
    groups = np.load(cache_dir / manifest["fit_support_group"]["file"], allow_pickle=False)
    unsupported = np.load(cache_dir / manifest["unsupported_mask"]["file"], allow_pickle=False)
    arrays["fit_support_group"] = groups
    arrays["unsupported_mask"] = unsupported
    return manifest, cache_dir, panels, arrays


def _shape(config, catalog_size: int) -> ActorShape:
    return ActorShape(vocab_size=catalog_size + 3,
                      n_positions=int(config.model.get("n_positions", 256)),
                      n_layer=int(config.model.get("n_layer", 4)),
                      n_embd=int(config.model.get("n_embd", 256)),
                      n_head=int(config.model.get("n_head", 4)),
                      dropout=float(config.model.get("dropout", 0.1)))


def _load_actor(checkpoint: str | Path, shape: ActorShape, device: torch.device) -> tuple[ItemGPT2Actor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = ItemGPT2Actor(shape)
    actor.load_state_dict(payload["model"])
    if actor.parameter_count() != 8_345_856:
        raise ValueError(f"full_model actor architecture changed: {actor.parameter_count()} parameters")
    actor.to(device).eval()
    return actor, payload


def _model_metrics(log_dist: Tensor, rows: Sequence[dict[str, Any]], store: SequenceStore,
                   catalog_ids: Sequence[int], durations: Sequence[float | None],
                   training_counts: Sequence[int], unsupported_mask: np.ndarray,
                   reward_matrix: np.ndarray, scale: RewardScale, beta: float,
                   reference_log: Tensor, support_mode: str, lambda_support: float,
                   support_lambda_reference: float | None = None) -> dict[str, Any]:
    full_probs = log_dist.float().exp()
    probs = full_probs[:, 3:]
    reward = torch.as_tensor(np.asarray(reward_matrix), dtype=torch.float32, device=log_dist.device)
    normalized = (reward - float(scale.mean)) / max(float(scale.std), 1e-3)
    expected_raw = (probs * reward).sum(-1)
    expected_norm = (probs * normalized).sum(-1)
    kl = full_categorical_kl(log_dist, reference_log)
    support = full_catalog_support_components(
        log_dist, reference_log, torch.as_tensor(unsupported_mask, device=log_dist.device),
    )
    unsupported_mass = support["unsupported_mass"]
    reference_mass = support["reference_unsupported_mass"]
    support_excess = support["support_excess"]
    reference_lambda = float(lambda_support if support_lambda_reference is None else support_lambda_reference)
    weighted_support = reference_lambda * support_excess
    j_base = expected_norm - beta * kl
    j_support = j_base - weighted_support
    optimized_objective = j_support if support_mode == "penalized" else j_base
    applied_penalty = lambda_support * support_excess if support_mode == "penalized" else torch.zeros_like(kl)
    entropy = -(full_probs * log_dist.float()).nan_to_num(0.0).sum(-1)
    log_cpu = log_dist[:, 3:].detach().cpu().numpy()
    catalog_array = np.asarray(catalog_ids, dtype=np.int64)
    top_items: dict[str, list[int]] = {}
    per_user: dict[str, dict[str, float]] = {}
    recent: list[set[int]] = []
    rec_indices: list[list[int]] = []
    rec_rows_raw: list[list[int]] = []
    target_coverage = 0
    for i, row in enumerate(rows):
        scores = log_cpu[i]
        order = np.lexsort((catalog_array, -scores))[:10]
        rec_indices.append([int(x) for x in order])
        raw_recs = [int(catalog_array[x]) for x in order]
        rec_rows_raw.append(raw_recs)
        top_items[str(row["uid"])] = raw_recs
        user = store.users[int(row["user_index"])]
        position = int(row["position"])
        previous = {int(item) for item in np.frombuffer(user.raw_item_ids, dtype=np.int64)[max(0, position - 20):position]
                    if int(item) >= 0}
        recent.append(previous)
        target_id = int(row["target_raw_item_id"])
        if target_id in set(catalog_array.tolist()):
            target_coverage += 1
            target_idx = int(np.searchsorted(catalog_array, target_id)) if np.all(catalog_array[:-1] <= catalog_array[1:]) else int(np.where(catalog_array == target_id)[0][0])
            score = float(scores[target_idx])
            rank = int(((scores > score) | ((scores == score) & (catalog_array < target_id))).sum()) + 1
        else:
            rank = None
        per_user[str(row["uid"])] = single_target_metrics(rank)
    exposure = exposure_metrics(rec_indices, len(catalog_ids), top_k=10)
    head_count = max(1, math.ceil(0.01 * len(training_counts)))
    head_indices = set(sorted(range(len(training_counts)), key=lambda i: (-int(training_counts[i]), int(catalog_ids[i])))[:head_count])
    head_share = sum(index in head_indices for row in rec_indices for index in row) / max(1, len(rec_indices) * 10)
    unsupported_top10 = sum(bool(unsupported_mask[index]) for row in rec_indices for index in row) / max(1, len(rec_indices) * 10)
    duration_values = [float(durations[index]) for row in rec_indices for index in row
                       if durations[index] is not None and math.isfinite(float(durations[index]))]
    jaccard_pairs = []
    if len(rec_indices) > 1:
        rng = random.Random(17031)
        selected_pairs = set()
        target_pairs = min(10_000, len(rec_indices) * (len(rec_indices) - 1) // 2)
        while len(selected_pairs) < target_pairs:
            a, b = rng.sample(range(len(rec_indices)), 2)
            selected_pairs.add((min(a, b), max(a, b)))
        for a, b in sorted(selected_pairs):
            left, right = set(rec_indices[a]), set(rec_indices[b])
            jaccard_pairs.append(len(left & right) / max(1, len(left | right)))
    return {
        "n_users": len(rows), "n_targets": len(rows), "target_coverage": target_coverage / max(1, len(rows)),
        "ndcg@10": float(np.mean([x["ndcg@10"] for x in per_user.values()])),
        "recall@10": float(np.mean([x["recall@10"] for x in per_user.values()])),
        "recall@20": float(np.mean([x["recall@20"] for x in per_user.values()])),
        "recall@100": float(np.mean([x["recall@100"] for x in per_user.values()])),
        "expected_raw_reward": float(expected_raw.mean()),
        "expected_normalized_reward": float(expected_norm.mean()),
        "kl_to_sft": float(kl.mean()), "J": float(optimized_objective.mean()),
        "J_base": float(j_base.mean()), "J_support": float(j_support.mean()),
        "optimized_objective": float(optimized_objective.mean()),
        "support_excess": float(support_excess.mean()),
        "support_lambda_reference": reference_lambda,
        "support_penalty_weighted": float(weighted_support.mean()),
        "support_penalty": float(applied_penalty.mean()),
        "unsupported_probability_mass": float(unsupported_mass.mean()),
        "unsupported_top10_fraction": unsupported_top10,
        "entropy_mean": float(entropy.mean()),
        "catalog_exposure": exposure,
        "head_share_top10": head_share,
        "history_repeat_top10": repeat_rate(rec_rows_raw, recent, top_k=10),
        "duration_seconds_top10": {
            "mean": float(np.mean(duration_values)) if duration_values else None,
            "p25": float(np.quantile(duration_values, 0.25)) if duration_values else None,
            "p50": float(np.quantile(duration_values, 0.50)) if duration_values else None,
            "p75": float(np.quantile(duration_values, 0.75)) if duration_values else None,
        },
        "pairwise_user_jaccard_top10": float(np.mean(jaccard_pairs)) if jaccard_pairs else None,
        "per_user_metrics": per_user,
        "per_user_expected_raw_reward": {str(row["uid"]): float(value)
                                          for row, value in zip(rows, expected_raw.detach().cpu().tolist())},
    }


@torch.no_grad()
def evaluate_actor(config_path: str | Path, checkpoint: str | Path,
                   rows: Sequence[dict[str, Any]], reward_matrix: np.ndarray,
                   cache_manifest: dict[str, Any], arrays: dict[str, np.ndarray],
                   beta: float = 1.0, support_mode: str = "ordinary", lambda_support: float = 1.0,
                   batch_size: int = 16, reward_head: str = "ratio",
                   reference_checkpoint: str | Path | None = None,
                   support_lambda_reference: float | None = None) -> dict[str, Any]:
    import polars as pl

    config, processed, artifacts = _root_and_processed(config_path)
    store, protocol = _load_store(processed)
    catalog = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    catalog_ids = [int(x) for x in catalog.get_column("item_id").to_list()]
    training_counts = [int(x) for x in catalog.get_column("training_count").to_list()]
    durations = catalog.get_column("catalog_duration_seconds").to_list()
    scale = RewardScale(**cache_manifest["reward_scale"])
    support_mask = arrays["unsupported_mask"]
    feature = pl.read_parquet(processed / "rm_item_features_fit.parquet", columns=["token_id", "rm_training_support"])
    fit_support = np.zeros(len(catalog_ids), dtype=np.int64)
    for token, support_count in feature.iter_rows():
        if 3 <= int(token) < 3 + len(fit_support):
            fit_support[int(token) - 3] = int(support_count or 0)
    # Evaluation cache rows follow development then confirmation ordering.
    actor_checkpoint = Path(checkpoint)
    if not actor_checkpoint.is_absolute():
        actor_checkpoint = config.path.parent.parent / actor_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = _shape(config, len(catalog_ids))
    actor, payload = _load_actor(actor_checkpoint, shape, device)
    if reference_checkpoint is not None:
        reference_path = Path(reference_checkpoint)
        if not reference_path.is_absolute():
            reference_path = config.path.parent.parent / reference_path
    elif int(config.dataset.get("cutoff_shift_days", 0)) == 0:
        reference_path = config.path.parent.parent / "artifacts/reference/runs/sft_strong_seed17/best.pt"
    else:
        temporal_source = config.project.get("temporal_model_source_root")
        if temporal_source:
            reference_path = _configured_artifact_root(config, "temporal_model_source_root", artifacts) / "sft_strong_seed17" / "best.pt"
        else:
            fold = f"tminus{int(config.dataset['cutoff_shift_days'])}"
            reference_path = artifacts / "runs" / fold / "sft_strong_seed17" / "best.pt"
    if not reference_path.is_file():
        reference_path = actor_checkpoint
    reference, _ = _load_actor(reference_path, shape, device)
    for p in reference.parameters():
        p.requires_grad_(False)
    histories = _histories(store, rows, int(config.dataset.get("history_length", 201)))
    log_parts, ref_parts = [], []
    for start in range(0, len(rows), batch_size):
        ids, attention = _collate(histories[start:start + batch_size], device)
        log_parts.append(actor.next_action_log_probs(ids, attention))
        ref_parts.append(reference.next_action_log_probs(ids, attention))
    log_dist = torch.cat(log_parts)
    reference_log = torch.cat(ref_parts)
    if support_lambda_reference is None:
        freeze_path = artifacts / "lambda_freeze.json"
        if freeze_path.is_file():
            support_lambda_reference = float(json.loads(freeze_path.read_text(encoding="utf-8"))["selected_lambda"])
        else:
            support_lambda_reference = float(lambda_support)
    result = _model_metrics(log_dist, rows, store, catalog_ids, durations, training_counts,
                            support_mask, reward_matrix, scale, beta, reference_log,
                            support_mode, lambda_support, support_lambda_reference)
    result["checkpoint"] = str(actor_checkpoint.resolve())
    result["checkpoint_sha256"] = sha256_file(actor_checkpoint)
    result["checkpoint_update"] = payload.get("metadata", {}).get("update", payload.get("step"))
    result["reference_checkpoint"] = str(reference_path.resolve())
    result["reward_head"] = reward_head
    return result


def _j(log_dist: Tensor, ref_log: Tensor, normalized_reward: Tensor,
       unsupported_mask: Tensor, beta: float, support_mode: str, lambda_support: float,
       support_lambda_reference: float | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    probs = log_dist[:, 3:].float().exp()
    kl = full_categorical_kl(log_dist.float(), ref_log.float())
    expected_reward = (probs * normalized_reward).sum(-1)
    support = full_catalog_support_components(log_dist, ref_log, unsupported_mask)
    unsupported_mass = support["unsupported_mass"]
    reference_mass = support["reference_unsupported_mass"]
    excess = support["support_excess"]
    reference_lambda = float(lambda_support if support_lambda_reference is None else support_lambda_reference)
    penalty = excess if support_mode == "penalized" else torch.zeros_like(kl)
    j_base = expected_reward - beta * kl
    j_support = j_base - reference_lambda * excess
    objective = j_support if support_mode == "penalized" else j_base
    return objective, {"reward": expected_reward, "kl": kl, "unsupported_mass": unsupported_mass,
                       "reference_unsupported_mass": reference_mass, "support_excess": excess,
                       "support_penalty_weighted": reference_lambda * excess,
                       "J_base": j_base, "J_support": j_support,
                       "optimized_objective": objective, "penalty": penalty}


def _eval_checkpoint(config_path, run_path: Path, step: int, rows, eval_matrix, cache_manifest,
                     arrays, beta, support_mode, lambda_support, head, reference_rewards,
                     write_all_metrics=True):
    snapshot = run_path / "snapshots" / f"update_{step:05d}.pt"
    config, processed, artifacts = _root_and_processed(config_path)
    panels = json.loads((_panels_directory(config, artifacts) / "panels.json").read_text(encoding="utf-8"))
    dev_n = len(panels["development"])
    result = evaluate_actor(config_path, snapshot, rows[:dev_n], eval_matrix[:dev_n], cache_manifest,
                            arrays, beta, support_mode, lambda_support, reward_head=head)
    baseline = reference_rewards
    per_user = result["per_user_metrics"]
    base_metrics = baseline["per_user_metrics"]
    common = sorted(set(per_user) & set(base_metrics))
    result["paired_to_sft"] = {}
    for name in ("ndcg@10", "recall@10", "recall@20", "recall@100"):
        differences = {int(uid): per_user[uid][name] - base_metrics[uid][name] for uid in common}
        result["paired_to_sft"][name] = paired_user_bootstrap(differences, repetitions=2000, seed=17017)
    reward_deltas = {int(uid): result["per_user_expected_raw_reward"][uid]
                     - baseline["per_user_expected_raw_reward"][uid] for uid in common}
    result["paired_to_sft"]["expected_raw_reward"] = paired_user_bootstrap(
        reward_deltas, repetitions=2000, seed=17018)
    base_coverage = float(baseline["catalog_exposure"]["coverage"])
    relative_ndcg = result["ndcg@10"] / max(float(baseline["ndcg@10"]), 1e-12) - 1.0
    relative_coverage = float(result["catalog_exposure"]["coverage"]) / max(base_coverage, 1e-12) - 1.0
    result["ndcg10_relative_change"] = relative_ndcg
    result["baseline_ndcg10"] = float(baseline["ndcg@10"])
    result["coverage_relative_change"] = relative_coverage
    result["head_share_delta"] = result["head_share_top10"] - baseline["head_share_top10"]
    result["history_repeat_delta"] = result["history_repeat_top10"] - baseline["history_repeat_top10"]
    result["point_guardrails"] = {
        "ndcg_loss_at_most_2pct": relative_ndcg >= -0.02,
        "coverage_at_least_90pct_of_sft": relative_coverage >= -0.10,
        "head_share_increase_at_most_2pp": result["head_share_delta"] <= 0.02,
        "history_repeat_increase_at_most_2pp": result["history_repeat_delta"] <= 0.02,
    }
    if not write_all_metrics:
        result.pop("per_user_metrics", None)
        result.pop("per_user_expected_raw_reward", None)
    return result


@torch.no_grad()
def evaluate_b5(config_path: str | Path, actor_checkpoint: str | Path, *, beta: float,
                reward_head: str = "ratio", output_path: str | Path | None = None) -> dict[str, Any]:
    import polars as pl

    config, processed, artifacts = _root_and_processed(config_path)
    cache_manifest, cache_dir, panels, arrays = _load_cache(config_path, "fit", reward_head)
    store, protocol = _load_store(processed)
    catalog = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    catalog_ids = [int(x) for x in catalog.get_column("item_id").to_list()]
    counts = [int(x) for x in catalog.get_column("training_count").to_list()]
    durations = catalog.get_column("catalog_duration_seconds").to_list()
    dev_rows = panels["development"]
    dev_n = len(dev_rows)
    raw = np.asarray(arrays["evaluation"][:dev_n], dtype=np.float32)
    scale = RewardScale(**cache_manifest["reward_scale"])
    normalized = torch.from_numpy((raw - scale.mean) / max(scale.std, 1e-3))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = _shape(config, len(catalog_ids))
    actor_checkpoint = Path(actor_checkpoint)
    if not actor_checkpoint.is_absolute():
        actor_checkpoint = config.path.parent.parent / actor_checkpoint
    reference, _ = _load_actor(actor_checkpoint, shape, device)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    histories = _histories(store, dev_rows, int(config.dataset.get("history_length", 201)))
    pieces = []
    for start in range(0, len(histories), 16):
        ids, attention = _collate(histories[start:start + 16], device)
        pieces.append(reference.next_action_log_probs(ids, attention))
    log_ref = torch.cat(pieces)
    adjusted = log_ref[:, 3:] + normalized.to(device) / float(beta)
    b5_catalog = torch.log_softmax(adjusted, dim=-1)
    b5_log = torch.cat((log_ref[:, :3], b5_catalog), dim=-1)
    support = arrays["unsupported_mask"]
    result = _model_metrics(b5_log, dev_rows, store, catalog_ids, durations, counts, support,
                            raw, scale, beta, log_ref, "ordinary", 0.0)
    result.update({"policy": "analytic B5", "beta": beta,
                   "interpretation": "unrestricted per-context optimum for ordinary J on this frozen RM cache; not a parametric actor or relevance upper bound",
                   "reward_head": reward_head, "cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json")})
    target = Path(output_path) if output_path else cache_dir / f"b5_beta_{beta:g}_development.json"
    if not target.is_absolute():
        target = config.path.parent.parent / target
    _json(target, result)
    return result


def _run_id(head: str, method: str, support: str, beta: float, seed: int) -> str:
    prefix = "q1" if head == "ratio" else "q2"
    return f"{prefix}_{method}_{support}_b{beta:g}_s{seed}".replace(".", "p")


def _context_schedule(config, cache_dir: Path, panels: dict[str, Any], *, seed: int,
                      schedule_budget: int, batch_contexts: int) -> tuple[np.ndarray, Path]:
    schedule_path = cache_dir / f"context_schedule_seed{seed}_n{schedule_budget}.npy"
    if schedule_path.exists():
        schedule = np.load(schedule_path, allow_pickle=False)
        if schedule.shape != (schedule_budget, batch_contexts):
            raise ValueError("frozen context schedule dimensions differ from the requested full_model plan")
        return schedule, schedule_path
    if config.project.get("cache_source_read_only", False):
        _, _, artifacts = _root_and_processed(config.path)
        fold = "main" if int(config.dataset.get("cutoff_shift_days", 0)) == 0 else f"tminus{int(config.dataset['cutoff_shift_days'])}"
        schedule_path = artifacts / "schedules" / fold / schedule_path.name
        if schedule_path.exists():
            schedule = np.load(schedule_path, allow_pickle=False)
            if schedule.shape != (schedule_budget, batch_contexts):
                raise ValueError("correction context schedule dimensions differ from the requested full_model plan")
            return schedule, schedule_path
    context_seed = int(config.training.get("rl_context_panel_seed", 30017)) + int(seed) * 1009
    rng = np.random.default_rng(context_seed)
    schedule = rng.integers(0, len(panels["train"]), size=(schedule_budget, batch_contexts), dtype=np.int32)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(schedule_path, schedule, allow_pickle=False)
    return schedule, schedule_path


def build_stream_reward_cache(config_path: str | Path, *, seed: int, updates: int = 1000) -> Path:
    """Cache full-catalog fit-RM scores for new A_RL anchors from cached-panel users."""
    config, processed, artifacts = _root_and_processed(config_path)
    base_manifest, base_cache, panels, _ = _load_cache(config_path, "fit", "ratio")
    store, data_protocol = _load_store(processed)
    batch_contexts = int(config.training.get("rl_batch_contexts", 16))
    schedule_budget = max(updates, int(config.training.get("rl_long_cache_updates", 1000)))
    user_schedule, schedule_path = _context_schedule(config, base_cache, panels, seed=seed,
                                                      schedule_budget=schedule_budget,
                                                      batch_contexts=batch_contexts)
    schedule_hash = sha256_file(schedule_path)
    source_stream_dir = base_cache / "stream" / f"seed{seed}"
    fold = "main" if int(config.dataset.get("cutoff_shift_days", 0)) == 0 else f"tminus{int(config.dataset['cutoff_shift_days'])}"
    correction_stream_dir = artifacts / "cache" / fold / "fit_ratio" / "stream" / f"seed{seed}"
    manifest_path = None
    base_cache_hash = sha256_file(base_cache / "cache_manifest.json")
    source_hash = source_tree_sha256(config.path.parent.parent / "src" / "tlab")
    read_only = bool(config.project.get("cache_source_read_only", False))

    def compatible_source_tree(existing_hash: str) -> bool:
        if existing_hash == source_hash:
            return True
        compatibility_path = artifacts / "source_compatibility.json"
        if not compatibility_path.is_file():
            return False
        compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
        return any(row.get("source_tree_sha256_before") == existing_hash
                   and row.get("source_tree_sha256_after") == source_hash
                   and row.get("scope") == "stream_cache_manifest_writer_only"
                   for row in compatibility.get("compatible_source_transitions", []))

    def reusable_stream(stream_dir: Path, *, permit_source_tree_change: bool) -> bool:
        manifest_path = stream_dir / "stream_manifest.json"
        if not manifest_path.is_file():
            return False
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (existing.get("user_schedule_sha256") == schedule_hash
                and existing.get("updates") == updates
                and existing.get("reward_head") == "ratio"
                and existing.get("reward_checkpoint_sha256") == base_manifest["reward_checkpoint_sha256"]
                and existing.get("base_cache_manifest_sha256") == base_cache_hash
                and existing.get("reward_feature_sha256") == base_manifest["reward_feature_sha256"]
                and existing.get("reward_scale_sha256") == base_manifest["scale_sha256"]
                and existing.get("unsupported_mask_sha256") == base_manifest["unsupported_mask"]["sha256"]
                and (permit_source_tree_change or compatible_source_tree(existing.get("source_tree_sha256", "")))
                and existing.get("processed_listens_sha256") == data_protocol["processed_listens_sha256"]
                and existing.get("panel_ids_sha256") == panels["panel_ids_sha256"]
                and all((stream_dir / filename).is_file()
                        and sha256_file(stream_dir / filename) == expected
                        for filename, expected in existing.get("files", {}).items()))
    if read_only and reusable_stream(source_stream_dir, permit_source_tree_change=True):
        return source_stream_dir
    stream_dir = correction_stream_dir if read_only else source_stream_dir
    if reusable_stream(stream_dir, permit_source_tree_change=False):
        return stream_dir
    stream_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = stream_dir / "stream_manifest.json"
    anchor_rng = np.random.default_rng(int(config.training.get("rl_context_panel_seed", 30017))
                                       + int(seed) * 1013 + 7001)
    train_users = panels["train"]
    unique_rows: list[dict[str, Any]] = []
    row_lookup: dict[tuple[int, int], int] = {}
    anchor_schedule = np.empty((updates, batch_contexts), dtype=np.int32)
    for update_index in range(updates):
        for batch_index, train_row_index in enumerate(user_schedule[update_index]):
            user_index = int(train_users[int(train_row_index)]["user_index"])
            positions = store.users[user_index].rl_anchors
            if len(positions) == 0:
                raise ValueError("stream bridge selected a train-panel user without A_RL anchors")
            position = int(positions[int(anchor_rng.integers(0, len(positions)))])
            key = (user_index, position)
            if key not in row_lookup:
                row_lookup[key] = len(unique_rows)
                unique_rows.append(_entry(store, user_index, position, len(unique_rows)))
            anchor_schedule[update_index, batch_index] = row_lookup[key]
    stream_dir.mkdir(parents=True, exist_ok=True)
    schedule_out = stream_dir / "anchor_schedule.npy"
    np.save(schedule_out, anchor_schedule, allow_pickle=False)
    rows_out = stream_dir / "anchor_rows.json"
    _json(rows_out, unique_rows)

    checkpoint = Path(base_manifest["reward_checkpoint"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, features = _load_reward_model(config, checkpoint, "fit", device)
    catalog_size = int(data_protocol["catalog_size"])
    item_tokens = torch.arange(3, 3 + catalog_size, device=device).unsqueeze(0)
    item_ids = np.arange(3, 3 + catalog_size, dtype=np.int64)
    durations = torch.tensor([features.get(int(token), (0.0, True, 0))[0]
                              for token in item_ids], dtype=torch.float32, device=device).unsqueeze(0)
    missing = torch.tensor([features.get(int(token), (0.0, True, 0))[1]
                            for token in item_ids], dtype=torch.bool, device=device).unsqueeze(0)
    output = stream_dir / "stream_train.npy"
    mmap = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32,
                                     shape=(len(unique_rows), catalog_size))
    histories = _histories(store, unique_rows, int(config.dataset.get("history_length", 201)))
    started = time.perf_counter()
    model.eval()
    for start in range(0, len(unique_rows), 8):
        batch = histories[start:start + 8]
        ids, attention = _collate(batch, device)
        candidates = item_tokens.expand(len(batch), -1)
        repeated = torch.as_tensor(np.stack([
            np.isin(item_ids, np.asarray(history[-20:], dtype=np.int64)) for history in batch
        ]), dtype=torch.bool, device=device)
        prediction = model.score_catalog(
            ids, attention, candidates, durations.expand(len(batch), -1),
            missing.expand(len(batch), -1), repeated,
            chunk_size=int(config.training.get("rl_reward_catalog_chunk_size", 512)),
        )
        mmap[start:start + len(batch)] = prediction["ratio"].float().cpu().numpy()
        if start % 512 == 0:
            mmap.flush()
    mmap.flush()
    del mmap
    files = {path.name: sha256_file(path) for path in (output, rows_out, schedule_out)}
    result = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "status": "completed",
              "seed": seed, "updates": updates, "batch_contexts": batch_contexts,
              "unique_histories": len(unique_rows), "history_draws": updates * batch_contexts,
              "repeat_history_draws": updates * batch_contexts - len(unique_rows),
              "user_schedule_sha256": schedule_hash,
              "anchor_schedule_seed": int(config.training.get("rl_context_panel_seed", 30017))
                  + int(seed) * 1013 + 7001,
              "anchor_schedule_sha256": files[schedule_out.name],
              "panel_ids_sha256": panels["panel_ids_sha256"],
              "processed_listens_sha256": data_protocol["processed_listens_sha256"],
              "catalog_sha256": data_protocol["catalog_sha256"],
              "reward_head": "ratio",
              "reward_checkpoint_sha256": base_manifest["reward_checkpoint_sha256"],
              "base_cache_manifest_sha256": base_cache_hash,
              "reward_feature_sha256": base_manifest["reward_feature_sha256"],
              "reward_scale_sha256": base_manifest["scale_sha256"],
              "unsupported_mask_sha256": base_manifest["unsupported_mask"]["sha256"],
              "source_tree_sha256": source_hash,
              "rm_scored_actions": len(unique_rows) * catalog_size,
              "cache_bytes": sum((stream_dir / name).stat().st_size for name in files),
              "score_elapsed_seconds": time.perf_counter() - started,
              "files": files, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _json(manifest_path, result)
    return stream_dir


def train_run(config_path: str | Path, actor_checkpoint: str | Path, *,
              method: str, support_mode: str, beta: float, seed: int = 17,
              reward_head: str = "ratio", updates: int | None = None,
              checkpoint_interval: int = 100, eval_interval: int = 500,
              run_dir: str | Path | None = None, resume: str | Path | None = None,
              stop_after_updates: int | None = None, reuse_epochs: int = 1,
              lambda_support_value: float | None = None,
              run_kind: str = "core", evaluate_development: bool = True,
              context_source: str = "cached") -> dict[str, Any]:
    import polars as pl

    if method not in {"sampled", "exact"} or support_mode not in {"ordinary", "penalized"}:
        raise ValueError("method must be sampled/exact and support_mode ordinary/penalized")
    if beta <= 0 or reuse_epochs < 1:
        raise ValueError("beta and reuse epochs must be positive")
    _configure_reproducibility(seed)
    config, processed, artifacts = _root_and_processed(config_path)
    cache_manifest, cache_dir, panels, arrays = _load_cache(config_path, "fit", reward_head)
    store, data_protocol = _load_store(processed)
    train_rows = panels["train"]
    eval_rows = panels["development"] + panels["confirmation"]
    train_matrix = arrays["train"]
    eval_matrix = arrays["evaluation"]
    total_updates = int(updates or config.training.get("rl_updates", 1000))
    batch_contexts = int(config.training.get("rl_batch_contexts", 16))
    group_size = int(config.training.get("rl_group_size", 8))
    schedule_budget = max(total_updates, int(config.training.get("rl_long_cache_updates", 1000)))
    if context_source not in {"cached", "stream"}:
        raise ValueError("context_source must be cached or stream")
    schedule, train_schedule_path = _context_schedule(config, cache_dir, panels, seed=seed,
                                                       schedule_budget=schedule_budget,
                                                       batch_contexts=batch_contexts)
    schedule_hash = sha256_file(train_schedule_path)
    stream_rows = None
    stream_matrix = None
    stream_anchor_schedule = None
    stream_cache_hash = None
    if context_source == "stream":
        stream_dir = build_stream_reward_cache(config_path, seed=seed, updates=total_updates)
        stream_manifest = json.loads((stream_dir / "stream_manifest.json").read_text(encoding="utf-8"))
        if stream_manifest["user_schedule_sha256"] != schedule_hash:
            raise ValueError("stream bridge user schedule differs from the matched cached schedule")
        stream_rows = json.loads((stream_dir / "anchor_rows.json").read_text(encoding="utf-8"))
        stream_matrix = np.load(stream_dir / "stream_train.npy", mmap_mode="r", allow_pickle=False)
        stream_anchor_schedule = np.load(stream_dir / "anchor_schedule.npy", mmap_mode="r", allow_pickle=False)
        stream_cache_hash = sha256_file(stream_dir / "stream_manifest.json")
    if run_dir is None:
        run_name = _run_id(reward_head, method, support_mode, beta, seed)
        if context_source == "stream":
            run_name += "_stream"
        run_path = artifacts / "runs" / run_name
    else:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = config.path.parent.parent / run_path
    if resume is None and run_path.exists() and any(run_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite full_model run: {run_path}")
    run_path.mkdir(parents=True, exist_ok=True)
    shape = _shape(config, int(data_protocol["catalog_size"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, _ = _load_actor(actor_checkpoint, shape, device)
    reference, _ = _load_actor(actor_checkpoint, shape, device)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(config.training.get("rl_learning_rate", 1e-5)))
    reward_scale = RewardScale(**cache_manifest["reward_scale"])
    scale = max(reward_scale.std, 1e-3)
    if support_mode == "ordinary":
        lambda_support = 0.0
        if lambda_support_value not in (None, 0, 0.0):
            raise ValueError("ordinary runs cannot have a support penalty")
    elif lambda_support_value is not None:
        lambda_support = float(lambda_support_value)
    else:
        lambda_path = artifacts / "lambda_freeze.json"
        if not lambda_path.is_file():
            raise FileNotFoundError("freeze the train-only lambda pilot before penalized main runs")
        lambda_support = float(json.loads(lambda_path.read_text(encoding="utf-8"))["selected_lambda"])
    support_lambda_reference = lambda_support
    lambda_path = artifacts / "lambda_freeze.json"
    if support_mode == "ordinary" and lambda_path.is_file():
        support_lambda_reference = float(json.loads(lambda_path.read_text(encoding="utf-8"))["selected_lambda"])
    support_mask_cpu = arrays["unsupported_mask"]
    support_mask = torch.as_tensor(support_mask_cpu, dtype=torch.float32, device=device)
    rng_gen = torch.Generator(device=device).manual_seed(int(seed) + 101)
    start_update = 0
    checkpoint_preview = None
    if resume is not None:
        resume_path = Path(resume).resolve()
        if resume_path.name != "last.pt" or resume_path.parent.resolve() != run_path.resolve():
            raise ValueError("resume must point to last.pt in the original full_model run directory")
        checkpoint_preview = torch.load(resume_path, map_location="cpu", weights_only=False)
        metadata = checkpoint_preview.get("metadata", {})
        _validate_objective_resume(metadata)
        expected = {"method": method, "support_mode": support_mode, "beta": float(beta), "seed": int(seed),
                    "objective_version": "normalized_reward_v1",
                    "advantage_reward_units": "normalized",
                    "reward_mean": float(reward_scale.mean), "reward_std": float(reward_scale.std),
                    "effective_reward_scale": float(scale),
                    "reward_head": reward_head, "cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json"),
                    "panel_ids_sha256": panels["panel_ids_sha256"], "context_schedule_sha256": schedule_hash,
                    "actor_checkpoint_sha256": sha256_file(actor_checkpoint), "reuse_epochs": reuse_epochs,
                    "lambda_support": lambda_support, "run_kind": run_kind,
                    "evaluate_development": bool(evaluate_development),
                    "context_source": context_source, "stream_cache_sha256": stream_cache_hash,
                    "learning_rate": float(config.training.get("rl_learning_rate", 1e-5)),
                    "group_size": group_size, "batch_contexts": batch_contexts,
                    "source_tree_sha256": source_tree_sha256(config.path.parent.parent / "src" / "tlab")}
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"resume {key} differs from original run")
        checkpoint = load_training_checkpoint(resume_path, actor, optimizer, map_location=device)
        sampler_state = checkpoint.get("sampler_state", {})
        start_update = int(sampler_state.get("next_update", 0))
        if sampler_state.get("action_generator_state") is not None:
            rng_gen.set_state(sampler_state["action_generator_state"].cpu())
    elif resume is not None:
        raise AssertionError("unreachable resume state")
    if start_update >= total_updates:
        raise ValueError("resume checkpoint has already reached requested total updates")
    last_checkpoint_update = start_update

    def save_checkpoint(step: int, status: str = "running"):
        nonlocal last_checkpoint_update
        metadata = {
            "protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "objective_version": "normalized_reward_v1",
            "advantage_reward_units": "normalized", "run_id": run_path.name,
            "method": method, "support_mode": support_mode, "lambda_support": lambda_support,
            "beta": float(beta), "seed": int(seed), "reward_head": reward_head,
            "reuse_epochs": reuse_epochs, "update": step, "status": status,
            "run_kind": run_kind, "evaluate_development": bool(evaluate_development),
            "context_source": context_source, "stream_cache_sha256": stream_cache_hash,
            "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
            "actor_checkpoint_sha256": sha256_file(actor_checkpoint),
            "reward_checkpoint_sha256": cache_manifest["reward_checkpoint_sha256"],
            "cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json"),
            "panel_ids_sha256": panels["panel_ids_sha256"], "context_schedule_sha256": schedule_hash,
            "processed_listens_sha256": data_protocol["processed_listens_sha256"],
            "catalog_sha256": data_protocol["catalog_sha256"],
            "reward_scale": {"mean": reward_scale.mean, "std": reward_scale.std,
                             "effective_scale": scale, "sample_count": reward_scale.sample_count},
            "reward_mean": float(reward_scale.mean), "reward_std": float(reward_scale.std),
            "effective_reward_scale": float(scale),
            "group_size": group_size, "batch_contexts": batch_contexts,
            "learning_rate": float(config.training.get("rl_learning_rate", 1e-5)),
            "source_tree_sha256": source_tree_sha256(config.path.parent.parent / "src" / "tlab"),
            "parameter_count": actor.parameter_count(),
        }
        save_training_checkpoint(run_path / "last.pt", actor, optimizer, step, metadata,
                                 sampler_state={"next_update": step,
                                                "action_generator_state": rng_gen.get_state().cpu()})
        last_checkpoint_update = step
        manifest.update({"status": status, "latest_update": step,
                         "last_checkpoint_sha256": sha256_file(run_path / "last.pt"),
                         "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                         "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None})
        _json(run_path / "run_manifest.json", manifest)
        _register_run(artifacts, manifest)

    def fail_run(update: int, reason: str, metrics_file) -> dict[str, Any]:
        metrics_file.flush()
        metrics_file.close()
        manifest.update({"status": "failed", "failure_update": update,
                         "stop_reason": reason, "latest_update": last_checkpoint_update,
                         "last_checkpoint_sha256": sha256_file(run_path / "last.pt")
                         if (run_path / "last.pt").is_file() else None,
                         "training_seconds": training_seconds_before + time.perf_counter() - training_started})
        _json(run_path / "run_manifest.json", manifest)
        _register_run(artifacts, manifest)
        return manifest

    manifest = {
        "protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "objective_version": "normalized_reward_v1",
        "advantage_reward_units": "normalized", "run_id": run_path.name,
        "run_dir": str(run_path.resolve()), "status": "running", "stage": run_kind,
        "method": method, "support_mode": support_mode, "beta": float(beta), "lambda_support": lambda_support,
        "seed": int(seed), "reward_head": reward_head, "updates_planned": total_updates,
        "evaluate_development": bool(evaluate_development),
        "context_source": context_source, "stream_cache_sha256": stream_cache_hash,
        "fresh_action_scores_per_update": batch_contexts * (group_size if method == "sampled" else int(data_protocol["catalog_size"])),
        "reuse_epochs": reuse_epochs, "optimizer_steps_planned": total_updates * reuse_epochs,
        "context_schedule_sha256": schedule_hash, "panel_ids_sha256": panels["panel_ids_sha256"],
        "reward_cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json"),
        "reward_scale": {"mean": reward_scale.mean, "std": reward_scale.std,
                         "effective_scale": scale, "sample_count": reward_scale.sample_count},
        "actor_checkpoint_sha256": sha256_file(actor_checkpoint),
        "reward_checkpoint_sha256": cache_manifest["reward_checkpoint_sha256"],
        "source_tree_sha256": source_tree_sha256(config.path.parent.parent / "src" / "tlab"),
        "processed_listens_sha256": data_protocol["processed_listens_sha256"],
        "catalog_sha256": data_protocol["catalog_sha256"],
        "bootstrap_repetitions": 2000, "validation_split_status": "known_validation_repartitioned; not untouched",
        "rm_audit_policy_scores_used_for_selection": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    previous_manifest = {}
    if resume is not None and (run_path / "run_manifest.json").exists():
        previous_manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
        manifest = {**previous_manifest,
                    **manifest, "started_at_utc": json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8")).get("started_at_utc"),
                    "resumed_at_utc": datetime.now(timezone.utc).isoformat()}
    _json(run_path / "run_manifest.json", manifest)
    _register_run(artifacts, manifest)

    # The lambda pilot never reads validation. Main runs cache a development
    # baseline for the fixed point guardrails.
    dev_n = len(panels["development"])
    ref_snapshot = run_path / "sft_reference.pt"
    if evaluate_development and not ref_snapshot.exists():
        shutil.copy2(actor_checkpoint, ref_snapshot)
    baseline = None
    if evaluate_development:
        if config.project.get("cache_source_read_only", False):
            fold_name = "main" if int(config.dataset.get("cutoff_shift_days", 0)) == 0 else f"tminus{int(config.dataset['cutoff_shift_days'])}"
            ref_result_path = artifacts / "evaluation_baselines" / f"{fold_name}_{reward_head}_{panels['panel_ids_sha256']['development'][:12]}.json"
            ref_result_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            ref_result_path = cache_dir / f"development_baseline_{reward_head}.json"
        if ref_result_path.exists():
            baseline = json.loads(ref_result_path.read_text(encoding="utf-8"))
        else:
            baseline = evaluate_actor(config_path, actor_checkpoint, panels["development"], eval_matrix[:dev_n],
                                      cache_manifest, arrays, beta, "ordinary", 0.0, reward_head=reward_head)
            _json(ref_result_path, baseline)
    metrics_path = run_path / "metrics.jsonl"
    metrics_file = metrics_path.open("a", encoding="utf-8")
    training_seconds_before = float(previous_manifest.get("training_seconds", 0.0))
    training_started = time.perf_counter()
    snapshots_dir = run_path / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    if start_update == 0:
        save_checkpoint(0)
        shutil.copy2(run_path / "last.pt", snapshots_dir / "update_00000.pt")
        row = {"update": 0, "development": baseline, "elapsed_training_seconds": 0.0}
        metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
        metrics_file.flush()

    actor.eval()
    reference.eval()
    train_histories = _histories(store, train_rows, int(config.dataset.get("history_length", 201)))
    snapshot_steps = {100, 500, 1000, 2500, total_updates}
    for update_idx in range(start_update, total_updates):
        row_indices = schedule[update_idx].astype(np.int64)
        if context_source == "stream":
            stream_indices = stream_anchor_schedule[update_idx].astype(np.int64)
            batch_rows = [stream_rows[int(i)] for i in stream_indices]
            histories_batch = _histories(store, batch_rows, int(config.dataset.get("history_length", 201)))
            reward_batch_np = np.asarray(stream_matrix[stream_indices], dtype=np.float32)
        else:
            histories_batch = [train_histories[int(i)] for i in row_indices]
            reward_batch_np = np.asarray(train_matrix[row_indices], dtype=np.float32)
        ids, attention = _collate(histories_batch, device)
        reward_batch = torch.from_numpy(reward_batch_np.copy()).to(device)
        reference_log = reference.next_action_log_probs(ids, attention).detach()
        optimizer.zero_grad(set_to_none=True)
        if method == "exact":
            reward_terms = prepare_reward_objective(
                reward_batch, reward_mean=reward_scale.mean, reward_std=reward_scale.std,
            )
            normalized_reward = reward_terms["normalized_reward"]
            log_dist = actor.next_action_log_probs(ids, attention)
            objective, stats = _j(log_dist, reference_log, normalized_reward, support_mask,
                                  beta, support_mode, lambda_support, support_lambda_reference)
            loss = -objective.mean()
            if not bool(torch.isfinite(loss)):
                return fail_run(update_idx + 1, "non-finite exact objective/loss", metrics_file)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                optimizer.zero_grad(set_to_none=True)
                return fail_run(update_idx + 1, "non-finite exact gradient norm", metrics_file)
            optimizer.step()
            with torch.no_grad():
                log_dist = actor.next_action_log_probs(ids, attention)
                objective, stats = _j(log_dist, reference_log, normalized_reward, support_mask,
                                      beta, support_mode, lambda_support, support_lambda_reference)
            sampled_reward_mean = None
        else:
            with torch.no_grad():
                old_log = actor.next_action_log_probs(ids, attention)
                probabilities = old_log[:, 3:].exp()
                actions = torch.multinomial(probabilities, group_size, replacement=True, generator=rng_gen) + 3
                old_selected = old_log.gather(-1, actions).detach()
            reward_terms = prepare_reward_objective(
                reward_batch, reward_mean=reward_scale.mean, reward_std=reward_scale.std,
                actions=actions, first_catalog_id=3,
            )
            normalized_reward = reward_terms["normalized_reward"]
            rewards = reward_terms["sampled_raw_reward"]
            normalized_sampled_rewards = reward_terms["sampled_normalized_reward"]
            loo = reward_terms["advantages"]
            for epoch in range(reuse_epochs):
                log_dist = actor.next_action_log_probs(ids, attention)
                kl = full_categorical_kl(log_dist, reference_log)
                support_stats = full_catalog_support_components(log_dist, reference_log, support_mask)
                penalty = (support_stats["support_excess"]
                           if support_mode == "penalized" else torch.zeros_like(kl))
                new_selected = log_dist.gather(-1, actions)
                if reuse_epochs == 1:
                    pg_loss = -(loo * new_selected).mean()
                    clip_fraction = 0.0
                    post_old_kl = 0.0
                else:
                    ratio = torch.exp(new_selected.float() - old_selected.float())
                    clipped = ratio.clamp(0.8, 1.2)
                    pg_loss = -torch.minimum(ratio * loo, clipped * loo).mean()
                    clip_fraction = float(((ratio - 1.0).abs() > 0.2).float().mean().detach())
                    post_old_kl = float(full_categorical_kl(log_dist.detach(), old_log.detach()).mean())
                loss = pg_loss + beta * kl.mean() + lambda_support * penalty.mean()
                if not bool(torch.isfinite(loss)):
                    optimizer.zero_grad(set_to_none=True)
                    return fail_run(update_idx + 1, "non-finite sampled objective/loss", metrics_file)
                if epoch > 0:
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                    optimizer.zero_grad(set_to_none=True)
                    return fail_run(update_idx + 1, "non-finite sampled gradient norm", metrics_file)
                optimizer.step()
            with torch.no_grad():
                new_log = actor.next_action_log_probs(ids, attention)
                objective, stats = _j(new_log, reference_log, normalized_reward, support_mask,
                                      beta, support_mode, lambda_support, support_lambda_reference)
                if reuse_epochs > 1:
                    post_old_kl = float(full_categorical_kl(new_log, old_log).mean())
            sampled_reward_mean = float(rewards.mean())
        update = update_idx + 1
        elapsed = training_seconds_before + time.perf_counter() - training_started
        record = {
            "update": update, "method": method, "support_mode": support_mode,
            "beta": beta, "seed": seed, "objective_version": "normalized_reward_v1",
            "advantage_reward_units": "normalized", "J_train_minibatch_post": float(objective.mean().detach()),
            "J_base_train_minibatch_post": float(stats["J_base"].mean().detach()),
            "J_support_train_minibatch_post": float(stats["J_support"].mean().detach()),
            "context_source": context_source,
            "expected_normalized_reward_train": float(stats["reward"].mean()),
            "kl_to_sft_train": float(stats["kl"].mean()),
            "unsupported_mass_train": float(stats["unsupported_mass"].mean()),
            "unsupported_excess_train": float(torch.relu(
                stats["unsupported_mass"] - stats["reference_unsupported_mass"]).mean().detach()),
            "support_penalty_train": float(stats["penalty"].mean()),
            "sampled_reward_mean": sampled_reward_mean,
            "sampled_raw_reward_mean": sampled_reward_mean,
            "sampled_normalized_reward_mean": (float(normalized_sampled_rewards.mean())
                                                if method == "sampled" else None),
            "reward_mean": float(reward_scale.mean), "reward_std": float(reward_scale.std),
            "effective_reward_scale": float(scale),
            "gradient_norm": float(grad_norm),
            "reuse_epochs": reuse_epochs, "clip_fraction": clip_fraction if method == "sampled" else None,
            "post_update_kl_to_old": post_old_kl if method == "sampled" else None,
            "elapsed_training_seconds": elapsed,
        }
        if not all(math.isfinite(value) for key, value in record.items()
                   if isinstance(value, float) and key != "elapsed_training_seconds"):
            return fail_run(update, "non-finite post-update training metric", metrics_file)
        metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
        if update % checkpoint_interval == 0 or update == total_updates or (stop_after_updates and update >= stop_after_updates):
            save_checkpoint(update, "interrupted" if stop_after_updates and update >= stop_after_updates and update < total_updates else "running")
        if update in snapshot_steps:
            shutil.copy2(run_path / "last.pt", snapshots_dir / f"update_{update:05d}.pt")
        if evaluate_development and (update % eval_interval == 0 or update in snapshot_steps):
            if update not in snapshot_steps:
                shutil.copy2(run_path / "last.pt", snapshots_dir / f"update_{update:05d}.pt")
            dev_metrics = _eval_checkpoint(config_path, run_path, update, eval_rows, eval_matrix,
                                           cache_manifest, arrays, beta, support_mode, lambda_support,
                                           reward_head, baseline)
            record["development"] = dev_metrics
            metrics_file.flush()
            with (run_path / "validation.jsonl").open("a", encoding="utf-8") as validation_file:
                validation_file.write(json.dumps({"update": update, "development": dev_metrics}, allow_nan=False) + "\n")
            print(f"{run_path.name} update {update}/{total_updates} J={record['J_train_minibatch_post']:.5f} "
                  f"dev_ndcg={dev_metrics['ndcg@10']:.6f} dev_reward={dev_metrics['expected_raw_reward']:.5f}", flush=True)
        else:
            print(f"{run_path.name} update {update}/{total_updates} "
                  f"J={record['J_train_minibatch_post']:.5f}", flush=True)
        if stop_after_updates is not None and update >= stop_after_updates:
            metrics_file.flush()
            metrics_file.close()
            manifest.update({"status": "interrupted", "stop_reason": "intentional checkpoint boundary",
                             "latest_update": update, "training_seconds": elapsed,
                             "last_checkpoint_sha256": sha256_file(run_path / "last.pt")})
            _json(run_path / "run_manifest.json", manifest)
            _register_run(artifacts, manifest)
            return manifest
        if update % 10 == 0:
            metrics_file.flush()
    metrics_file.flush()
    metrics_file.close()
    total_seconds = training_seconds_before + time.perf_counter() - training_started
    save_checkpoint(total_updates, "completed")
    shutil.copy2(run_path / "last.pt", snapshots_dir / f"update_{total_updates:05d}.pt")
    manifest.update({"status": "completed", "latest_update": total_updates,
                     "training_seconds": total_seconds,
                     "fresh_action_scores": total_updates * batch_contexts * (group_size if method == "sampled" else int(data_protocol["catalog_size"])),
                     "rm_scored_actions": int(cache_manifest["matrices"]["train"]["shape"][0]
                                               + cache_manifest["matrices"]["evaluation"]["shape"][0]) * int(data_protocol["catalog_size"])
                                               + int(cache_manifest["reward_scale"]["sample_count"]),
                     "last_checkpoint_sha256": sha256_file(run_path / "last.pt"),
                     "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    _json(run_path / "run_manifest.json", manifest)
    _register_run(artifacts, manifest)
    return manifest


def run_lambda_pilot(config_path: str | Path, actor_checkpoint: str | Path) -> dict[str, Any]:
    """Run the fixed 12-run, train-only support-penalty pilot."""
    config, _, artifacts = _root_and_processed(config_path)
    if (artifacts / "lambda_freeze.json").exists():
        raise FileExistsError("lambda is already frozen; pilot cannot be rerun for selection")
    betas = [float(value) for value in config.training.get("rl_support_lambda_betas", [0.1, 1.0, 3.0])]
    lambdas = [float(value) for value in config.training.get("rl_support_lambda_values", [1.0, 3.0, 10.0])]
    seed = int(config.training.get("rl_support_lambda_pilot_seed", 90117))
    updates = int(config.training.get("rl_support_lambda_probe_updates", [80, 90, 100])[-1])
    runs = []
    for beta in betas:
        specifications = [("ordinary", 0.0)] + [("penalized", value) for value in lambdas]
        for support_mode, value in specifications:
            run_id = f"lambda_pilot_q1_{support_mode}_b{beta:g}_l{value:g}_s{seed}".replace(".", "p")
            run_path = artifacts / "runs" / "lambda_pilot" / run_id
            resume_path = None
            existing_manifest = run_path / "run_manifest.json"
            if existing_manifest.is_file():
                existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
                if existing.get("status") == "completed" and int(existing.get("latest_update", 0)) >= updates:
                    runs.append({"run_id": run_id, "beta": beta, "support_mode": support_mode,
                                 "lambda": value, "run_dir": str(run_path.resolve()),
                                 "status": "completed", "training_seconds": existing.get("training_seconds")})
                    continue
                if existing.get("status") == "failed":
                    runs.append({"run_id": run_id, "beta": beta, "support_mode": support_mode,
                                 "lambda": value, "run_dir": str(run_path.resolve()),
                                 "status": "failed", "training_seconds": existing.get("training_seconds"),
                                 "stop_reason": existing.get("stop_reason")})
                    continue
                if (run_path / "last.pt").is_file():
                    resume_path = run_path / "last.pt"
            manifest = train_run(
                config_path, actor_checkpoint, method="sampled", support_mode=support_mode,
                beta=beta, seed=seed, reward_head="ratio", updates=updates,
                checkpoint_interval=10, eval_interval=updates, run_dir=run_path,
                resume=resume_path,
                lambda_support_value=value, run_kind="E1_lambda_pilot",
                evaluate_development=False,
            )
            runs.append({"run_id": run_id, "beta": beta, "support_mode": support_mode,
                         "lambda": value, "run_dir": str(run_path.resolve()),
                         "status": manifest["status"], "training_seconds": manifest.get("training_seconds")})
    complete = all(run.get("status") == "completed" for run in runs)
    result = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1",
              "status": "completed_train_only_pilot" if complete else "failed_train_only_pilot",
              "seed": seed, "updates_per_run": updates, "probe_updates": config.training.get(
                  "rl_support_lambda_probe_updates", [80, 90, 100]), "betas": betas,
              "lambda_values": lambdas, "uses_validation": False,
              "uses_confirmation": False, "uses_rm_audit": False, "uses_test": False,
              "runs": runs, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    target = artifacts / "support_lambda_pilot.json"
    _json(target, result)
    return result


def freeze_lambda_pilot(config_path: str | Path) -> dict[str, Any]:
    config, _, artifacts = _root_and_processed(config_path)
    pilot_path = artifacts / "support_lambda_pilot.json"
    if not pilot_path.is_file():
        raise FileNotFoundError("complete the fixed train-only lambda pilot first")
    if (artifacts / "lambda_freeze.json").exists():
        raise FileExistsError("lambda freeze is immutable once written")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("status") != "completed_train_only_pilot" or pilot.get("uses_validation"):
        raise ValueError("lambda selection requires the completed train-only pilot")
    threshold = float(config.training.get("rl_support_lambda_excess_threshold", 1e-4))
    probe_updates = [int(value) for value in config.training.get(
        "rl_support_lambda_probe_updates", [80, 90, 100])]
    by_beta: dict[str, Any] = {}
    for beta in pilot["betas"]:
        matched: dict[float, float] = {}
        for run in pilot["runs"]:
            if float(run["beta"]) != float(beta):
                continue
            path = Path(run["run_dir"]) / "metrics.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            values = [float(row["unsupported_excess_train"]) for row in rows
                      if int(row.get("update", -1)) in probe_updates]
            if len(values) != len(probe_updates):
                raise ValueError(f"pilot run {run['run_id']} is missing a frozen probe checkpoint")
            matched[float(run["lambda"])] = float(np.mean(values))
        ordinary = matched.get(0.0)
        if ordinary is None:
            raise ValueError(f"pilot beta={beta} is missing its matched ordinary run")
        by_beta[str(beta)] = {"ordinary_excess": ordinary,
                              "informative": ordinary > threshold,
                              "lambda_excess": {str(value): matched[value]
                                                for value in sorted(matched) if value > 0}}
    informative = [beta for beta, stats in by_beta.items() if stats["informative"]]
    if not informative:
        selected = 1.0
        status = "no_informative_beta; lambda=1 retained"
    else:
        selected = None
        for value in sorted(float(item) for item in pilot["lambda_values"]):
            if all(by_beta[beta]["lambda_excess"][str(value)]
                   <= 0.5 * by_beta[beta]["ordinary_excess"] for beta in informative):
                selected = value
                break
        if selected is None:
            selected = max(float(item) for item in pilot["lambda_values"])
            status = "pilot_support_control_failed; lambda=10 retained"
        else:
            status = "selected_minimum_common_lambda_with_50pct_excess_reduction"
    result = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "status": status,
              "selected_lambda": selected, "selection_rule":
              "minimum shared lambda in {1,3,10} reducing train-only excess by >=50% in every informative beta; informative iff ordinary excess >1e-4; if none informative choose 1; if none pass choose 10 and mark failed",
              "probe_updates": probe_updates, "threshold": threshold, "informative_betas": informative,
              "per_beta": by_beta, "pilot_sha256": sha256_file(pilot_path),
              "uses_validation": False, "uses_confirmation": False, "uses_rm_audit": False,
              "uses_test": False, "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    _json(artifacts / "lambda_freeze.json", result)
    return result


def run_exact_support_pilot(config_path: str | Path, actor_checkpoint: str | Path) -> dict[str, Any]:
    config, _, artifacts = _root_and_processed(config_path)
    frozen = json.loads((artifacts / "lambda_freeze.json").read_text(encoding="utf-8"))
    run_path = artifacts / "runs" / "exact_penalty_pilot" / "q1_exact_penalized_b1_s90118"
    manifest_path = run_path / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and int(existing.get("latest_update", 0)) >= 100:
            return existing
        if existing.get("status") == "failed":
            return existing
    resume = run_path / "last.pt" if (run_path / "last.pt").is_file() else None
    return train_run(config_path, actor_checkpoint, method="exact", support_mode="penalized",
                     beta=1.0, seed=90118, reward_head="ratio", updates=100,
                     checkpoint_interval=10, eval_interval=100, run_dir=run_path, resume=resume,
                     lambda_support_value=float(frozen["selected_lambda"]),
                     run_kind="E1_exact_penalty_pilot", evaluate_development=False)


def gradient_component_diagnostics(config_path: str | Path, actor_checkpoint: str | Path,
                                   *, beta_values: Sequence[float] = (0.1, 1.0, 3.0),
                                   probe_size: int = 256, stage: str = "pilot") -> dict[str, Any]:
    """Differentiate exact cached reward, KL, and support terms on train probes."""
    started = time.perf_counter()
    config, processed, artifacts = _root_and_processed(config_path)
    cache_manifest, cache_dir, panels, arrays = _load_cache(config_path, "fit", "ratio")
    store, data_protocol = _load_store(processed)
    rows = panels["train"][:min(probe_size, len(panels["train"]))]
    histories = _histories(store, rows, int(config.dataset.get("history_length", 201)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_checkpoint = Path(actor_checkpoint)
    if not actor_checkpoint.is_absolute():
        actor_checkpoint = config.path.parent.parent / actor_checkpoint
    shape = _shape(config, int(data_protocol["catalog_size"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    actor, _ = _load_actor(actor_checkpoint, shape, device)
    reference, _ = _load_actor(actor_checkpoint, shape, device)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    ids, attention = _collate(histories, device)
    policy_log = actor.next_action_log_probs(ids, attention)
    reference_log = reference.next_action_log_probs(ids, attention).detach()
    matrix = torch.as_tensor(np.asarray(arrays["train"][:len(rows)], dtype=np.float32), device=device)
    scale = RewardScale(**cache_manifest["reward_scale"])
    normalized_reward = prepare_reward_objective(
        matrix, reward_mean=scale.mean, reward_std=scale.std,
    )["normalized_reward"]
    unsupported = torch.as_tensor(arrays["unsupported_mask"], dtype=torch.float32, device=device)
    probabilities = policy_log[:, 3:].exp()
    reference_probability = reference_log[:, 3:].exp()
    current_mass = (probabilities * unsupported).sum(-1)
    reference_mass = (reference_probability * unsupported).sum(-1)
    excess = torch.relu(current_mass - reference_mass)
    reward_term = (probabilities * normalized_reward).sum(-1).mean()
    kl_term = full_categorical_kl(policy_log, reference_log).mean()
    penalty_term = excess.mean()
    mass_term = current_mass.mean()
    parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]

    def flattened_gradient(value: Tensor, retain_graph: bool) -> Tensor:
        gradient_parts = torch.autograd.grad(value, parameters, retain_graph=retain_graph,
                                             allow_unused=True)
        return torch.cat([part.detach().reshape(-1) for part in gradient_parts
                          if part is not None])

    grad_reward = flattened_gradient(reward_term, True)
    grad_kl = flattened_gradient(kl_term, True)
    grad_penalty = flattened_gradient(penalty_term, True)
    grad_mass = flattened_gradient(mass_term, False)

    def norm(value: Tensor) -> float:
        return float(torch.linalg.vector_norm(value).cpu())

    def cosine(left: Tensor, right: Tensor) -> float | None:
        left_norm, right_norm = norm(left), norm(right)
        if left_norm == 0 or right_norm == 0:
            return None
        return float(torch.dot(left, right).div(left_norm * right_norm).cpu())

    frozen = json.loads((artifacts / "lambda_freeze.json").read_text(encoding="utf-8"))
    lambda_support = float(frozen["selected_lambda"])
    per_beta = {}
    for beta in beta_values:
        grad_beta_kl = grad_kl * float(beta)
        grad_lambda_penalty = grad_penalty * lambda_support
        per_beta[str(beta)] = {
            "reward_gradient_norm_exact": norm(grad_reward),
            "beta_times_kl_gradient_norm": norm(grad_beta_kl),
            "lambda_times_penalty_gradient_norm": norm(grad_lambda_penalty),
            "active_hinge_fraction": float((excess > 0).float().mean().detach().cpu()),
            "mean_unsupported_excess": float(excess.mean().detach().cpu()),
            "cosine_penalty_to_excess_mass_gradient": cosine(grad_lambda_penalty, grad_mass),
            "cosine_reward_to_penalty_gradient": cosine(grad_reward, grad_lambda_penalty),
            "cosine_reward_to_beta_kl_gradient": cosine(grad_reward, grad_beta_kl),
            "reward_gradient_definition": "exact derivative of cached expected normalized RM reward; sampled update noise is logged separately",
        }
    result = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "status": "completed",
              "stage": stage, "panel": "training probe only", "probe_contexts": len(rows),
              "probe_anchor_ids_sha256": _sha_text("\n".join(row["anchor_id"] for row in rows)),
              "actor_checkpoint": str(actor_checkpoint.resolve()),
              "actor_checkpoint_sha256": sha256_file(actor_checkpoint),
              "cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json"),
              "lambda": lambda_support, "per_beta": per_beta,
              "reward_requires_grad": bool(normalized_reward.requires_grad),
              "reference_requires_grad": bool(reference_log.requires_grad),
              "diagnostic_elapsed_seconds": time.perf_counter() - started,
              "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _json(artifacts / "verifications" / f"gradient_components_{stage}.json", result)
    return result


def production_gradient_audit() -> dict[str, Any]:
    """Exercise the normalized reward, exact KL and full-catalog support path used by training."""
    dtype = torch.float64
    raw_cases = [
        {"head": "ratio", "mean": 0.7309647616348229, "std": 0.1557952064124541,
         "reward": torch.tensor([0.34, 0.78, 0.91], dtype=dtype)},
        {"head": "seconds", "mean": 139.9422039166093, "std": 41.75068850087061,
         "reward": torch.tensor([61.0, 158.0, 221.0], dtype=dtype)},
    ]
    cases = []
    beta, lambda_support = 0.7, 1.3
    for case in raw_cases:
        logits = torch.tensor([-0.8, 0.2, 1.0], dtype=dtype, requires_grad=True)
        reference_logits = torch.tensor([1.0, 0.4, -0.5], dtype=dtype)
        reserved = torch.full((1, 3), -torch.inf, dtype=dtype)
        log_policy = torch.cat((reserved, torch.log_softmax(logits, dim=-1).unsqueeze(0)), dim=-1)
        log_reference = torch.cat((reserved, torch.log_softmax(reference_logits, dim=-1).unsqueeze(0)), dim=-1).detach()
        reward_matrix = case["reward"].unsqueeze(0)
        exact_terms = prepare_reward_objective(
            reward_matrix, reward_mean=case["mean"], reward_std=case["std"],
        )
        unsupported = torch.tensor([0.0, 1.0, 1.0], dtype=dtype)
        exact_objective, exact_stats = _j(
            log_policy, log_reference, exact_terms["normalized_reward"], unsupported,
            beta, "penalized", lambda_support,
        )
        exact_gradient = torch.autograd.grad(exact_objective.mean(), logits, retain_graph=True)[0]
        action_groups = torch.tensor(list(itertools.product(range(3), repeat=2)), dtype=torch.long) + 3
        expanded_rewards = reward_matrix.expand(len(action_groups), -1)
        sampled_terms = prepare_reward_objective(
            expanded_rewards, reward_mean=case["mean"], reward_std=case["std"],
            actions=action_groups, first_catalog_id=3,
        )
        selected_log = log_policy.expand(len(action_groups), -1).gather(-1, action_groups)
        probabilities = log_policy[:, 3:].exp()[0]
        sequence_probabilities = probabilities[(action_groups[:, 0] - 3)] * probabilities[(action_groups[:, 1] - 3)]
        sampled_loss = -(sequence_probabilities.detach() *
                         (sampled_terms["advantages"] * selected_log).mean(-1)).sum()
        sampled_loss = sampled_loss + beta * exact_stats["kl"].mean() \
            + lambda_support * exact_stats["support_excess"].mean()
        sampled_gradient = torch.autograd.grad(sampled_loss, logits, retain_graph=True)[0]
        error = float((exact_gradient + sampled_gradient).abs().max())

        # Demonstrate that raw LOO advantages do not pass parity at the same beta/lambda.
        raw_terms = prepare_reward_objective(
            expanded_rewards, reward_mean=0.0, reward_std=1.0,
            actions=action_groups, first_catalog_id=3,
        )
        raw_loss = -(sequence_probabilities.detach() *
                     (raw_terms["advantages"] * selected_log).mean(-1)).sum()
        raw_loss = raw_loss + beta * exact_stats["kl"].mean() \
            + lambda_support * exact_stats["support_excess"].mean()
        raw_gradient = torch.autograd.grad(raw_loss, logits)[0]
        raw_error = float((exact_gradient + raw_gradient).abs().max())
        cases.append({"head": case["head"], "reward_mean": case["mean"], "reward_std": case["std"],
                      "effective_reward_scale": max(float(case["std"]), 1e-3),
                      "exact_vs_exhaustive_sampled_gradient_max_abs": error,
                      "legacy_raw_advantage_gradient_max_abs": raw_error,
                      "active_support_excess": float(exact_stats["support_excess"].mean()),
                      "kl": float(exact_stats["kl"].mean()),
                      "normalized_advantage_range": [float(sampled_terms["advantages"].min()),
                                                     float(sampled_terms["advantages"].max())]})

    # The support gradient is independent of whether a sampled group happened to draw an unsupported item.
    active_logits = torch.tensor([-0.8, 0.2, 1.0], dtype=dtype, requires_grad=True)
    reference_logits = torch.tensor([1.0, 0.4, -0.5], dtype=dtype)
    active_log = torch.log_softmax(torch.cat((torch.full((1,), -torch.inf, dtype=dtype),
                                              torch.full((2,), -torch.inf, dtype=dtype), active_logits)), -1)[None, :]
    reference_log = torch.log_softmax(torch.cat((torch.full((3,), -torch.inf, dtype=dtype), reference_logits)), -1)[None, :]
    support = torch.tensor([0.0, 1.0, 1.0], dtype=dtype)
    support_components = full_catalog_support_components(active_log, reference_log, support)
    active_gradient = torch.autograd.grad(support_components["support_excess"].mean(), active_logits)[0]
    no_unsupported_samples = torch.tensor([3, 3, 3, 3, 3, 3, 3, 3])
    no_unsupported_sampled_actions = bool(not support[(no_unsupported_samples - 3)].any())
    eps = 1e-3
    finite_difference = []
    for index in range(3):
        plus = active_logits.detach().clone(); plus[index] += eps
        minus = active_logits.detach().clone(); minus[index] -= eps
        plus_log = torch.log_softmax(torch.cat((torch.full((3,), -torch.inf, dtype=dtype), plus)), -1)[None, :]
        minus_log = torch.log_softmax(torch.cat((torch.full((3,), -torch.inf, dtype=dtype), minus)), -1)[None, :]
        plus_value = full_catalog_support_components(plus_log, reference_log, support)["support_excess"].mean()
        minus_value = full_catalog_support_components(minus_log, reference_log, support)["support_excess"].mean()
        finite_difference.append(float((plus_value - minus_value) / (2 * eps)))
    inactive_logits = torch.tensor([1.2, 0.4, -0.5], dtype=dtype, requires_grad=True)
    inactive_log = torch.log_softmax(torch.cat((torch.full((3,), -torch.inf, dtype=dtype), inactive_logits)), -1)[None, :]
    inactive = full_catalog_support_components(inactive_log, reference_log, support)["support_excess"]
    inactive_gradient = torch.autograd.grad(inactive.mean(), inactive_logits)[0]
    equal_reference = reference_logits.clone().requires_grad_(True)
    equal_log = torch.log_softmax(torch.cat((torch.full((3,), -torch.inf, dtype=dtype), equal_reference)), -1)[None, :]
    equal = full_catalog_support_components(equal_log, reference_log, support)["support_excess"]
    equal_gradient = torch.autograd.grad(equal.mean(), equal_reference)[0]
    support_error = max(abs(float(a) - b) for a, b in zip(active_gradient, finite_difference))
    status = "passed" if all(row["exact_vs_exhaustive_sampled_gradient_max_abs"] < 1e-6
                              and row["legacy_raw_advantage_gradient_max_abs"] > 1e-4 for row in cases) \
        and no_unsupported_sampled_actions and float(support_components["support_excess"].mean()) > 0 \
        and support_error < 1e-4 and float(inactive.mean()) == 0 and float(inactive_gradient.abs().max()) == 0 \
        and float(equal.mean()) == 0 and float(equal_gradient.abs().max()) == 0 else "failed"
    return {
        "status": status, "objective_version": "normalized_reward_v1",
        "cases": cases, "beta": beta, "lambda_support": lambda_support,
        "support_penalty_gradient_max_abs_vs_finite_difference": support_error,
        "active_support_excess_with_zero_unsupported_samples": no_unsupported_sampled_actions,
        "active_support_gradient": active_gradient.tolist(),
        "inactive_hinge_value": float(inactive.mean()),
        "inactive_hinge_gradient_max_abs": float(inactive_gradient.abs().max()),
        "policy_equal_to_reference_excess": float(equal.mean()),
        "policy_equal_to_reference_gradient_max_abs": float(equal_gradient.abs().max()),
        "reward_and_reference_detached": True,
        "reuse_rollout_advantages_are_normalized_and_fixed": True,
    }


def verify_resume_equivalence(config_path: str | Path, actor_checkpoint: str | Path) -> dict[str, Any]:
    """Compare a fixed sampled run with an interruption/resume at the same boundary."""
    config, _, artifacts = _root_and_processed(config_path)
    root = artifacts / "verifications" / "resume_equivalence_r3p3"
    split_dir = root / "split"
    full_dir = root / "uninterrupted"
    result_path = artifacts / "verifications" / "resume_equivalence_r3p3.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))

    def ensure(path: Path, *, start_interrupted: bool = False) -> dict[str, Any]:
        manifest_path = path / "run_manifest.json"
        resume_path = path / "last.pt" if (path / "last.pt").is_file() else None
        existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        if existing.get("status") == "completed" and int(existing.get("latest_update", 0)) >= 24:
            return existing
        if existing.get("status") == "failed":
            return existing
        stop_after = 8 if start_interrupted and resume_path is None else None
        return train_run(config_path, actor_checkpoint, method="sampled", support_mode="ordinary",
                         beta=1.0, seed=19001, reward_head="ratio", updates=24,
                         checkpoint_interval=8, eval_interval=24, run_dir=path,
                         resume=resume_path, stop_after_updates=stop_after,
                         run_kind="resume_verification", evaluate_development=False)

    interrupted = ensure(split_dir, start_interrupted=True)
    if interrupted.get("status") == "interrupted":
        resumed = ensure(split_dir)
    else:
        resumed = interrupted
    uninterrupted = ensure(full_dir)
    split_checkpoint = torch.load(split_dir / "last.pt", map_location="cpu", weights_only=False)
    full_checkpoint = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
    differences = [float((split_checkpoint["model"][key] - full_checkpoint["model"][key]).abs().max())
                   for key in split_checkpoint["model"]]
    model_max_abs = max(differences, default=0.0)
    split_sampler = split_checkpoint["sampler_state"]
    full_sampler = full_checkpoint["sampler_state"]
    sampler_equal = (int(split_sampler["next_update"]) == int(full_sampler["next_update"])
                     and torch.equal(split_sampler["action_generator_state"], full_sampler["action_generator_state"]))
    optimizer_equal = True
    left_state = split_checkpoint["optimizer"]["state"]
    right_state = full_checkpoint["optimizer"]["state"]
    optimizer_equal = left_state.keys() == right_state.keys()
    for state_key in left_state.keys() & right_state.keys():
        left, right = left_state[state_key], right_state[state_key]
        if left.keys() != right.keys():
            optimizer_equal = False
            break
        for key in left:
            a, b = left[key], right[key]
            if isinstance(a, Tensor) and not torch.equal(a, b):
                optimizer_equal = False
                break
            if not isinstance(a, Tensor) and a != b:
                optimizer_equal = False
                break
    rng_left, rng_right = split_checkpoint["rng"], full_checkpoint["rng"]
    python_rng_equal = rng_left["python"] == rng_right["python"]
    cpu_torch_rng_equal = torch.equal(rng_left["torch"], rng_right["torch"])
    cuda_rng_equal = True
    if rng_left.get("cuda") is None or rng_right.get("cuda") is None:
        cuda_rng_equal = rng_left.get("cuda") is None and rng_right.get("cuda") is None
    else:
        cuda_rng_equal = len(rng_left["cuda"]) == len(rng_right["cuda"]) and all(
            torch.equal(left.cpu(), right.cpu()) for left, right in zip(rng_left["cuda"], rng_right["cuda"]))
    # full_model training draws actions from the explicit saved CUDA Generator; the
    # global CPU torch RNG is not consumed by the eval-mode actor/update path.
    training_rng_equal = sampler_equal and python_rng_equal and cuda_rng_equal
    result = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1",
              "status": "passed" if model_max_abs == 0 and optimizer_equal and training_rng_equal else "failed",
              "updates": 24, "interruption_after_update": 8,
              "model_max_abs_difference": model_max_abs, "optimizer_state_equal": optimizer_equal,
              "sampler_cursor_and_rng_equal": sampler_equal,
              "global_rng_state_equal": python_rng_equal and cpu_torch_rng_equal and cuda_rng_equal,
              "training_rng_state_equal": training_rng_equal,
              "rng_domains": {"python_equal": python_rng_equal, "cpu_torch_equal": cpu_torch_rng_equal,
                              "cuda_equal": cuda_rng_equal,
                              "cpu_torch_rng_used_by_training": False},
              "resume_status_basis": "exact model and optimizer states plus action sampler, Python, and CUDA RNG equality; CPU torch RNG is serialized/restored but the full_model eval-mode actor uses the explicit action generator and does not consume global CPU torch RNG",
              "split_run_status": interrupted.get("status"),
              "resumed_status": resumed.get("status"),
              "uninterrupted_status": uninterrupted.get("status"),
              "comparison": "same cached data, seed, context schedule, full-catalog actions, reference, reward and objective",
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _json(artifacts / "verifications" / "resume_equivalence.json", result)
    return result


def freeze_selection(config_path: str | Path, *, output: str | Path | None = None) -> dict[str, Any]:
    config, _, artifacts = _root_and_processed(config_path)
    val_path = artifacts / "runs"
    candidates: list[dict[str, Any]] = []
    for path in sorted(val_path.glob("q1_*/validation.jsonl")):
        manifest_path = path.parent / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "completed" or int(manifest.get("seed", -1)) != 17
                or manifest.get("stage") != "core" or manifest.get("reward_head") != "ratio"
                or int(manifest.get("reuse_epochs", 1)) != 1):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if int(record.get("update", -1)) not in {100, 500, 1000}:
                continue
            metrics = record["development"]
            candidates.append({"run_id": manifest["run_id"], "run_dir": manifest["run_dir"],
                               "checkpoint": str((path.parent / "snapshots" / f"update_{int(record['update']):05d}.pt").resolve()),
                               "update": int(record["update"]), "method": manifest["method"],
                               "support_mode": manifest["support_mode"], "beta": manifest["beta"],
                               "seed": manifest["seed"], "metrics": metrics,
                               "point_guardrails": metrics["point_guardrails"],
                               "positive_fit_reward_ci": metrics["paired_to_sft"]["expected_raw_reward"]["ci_95_low"] > 0})
    reused_eval_path = artifacts / "reused_exact_development.json"
    if reused_eval_path.is_file():
        reused = json.loads(reused_eval_path.read_text(encoding="utf-8"))
        for row in reused.get("candidates", []):
            metrics = row["metrics"]
            candidates.append({**row, "metrics": metrics,
                               "point_guardrails": metrics["point_guardrails"],
                               "positive_fit_reward_ci": metrics["paired_to_sft"]["expected_raw_reward"]["ci_95_low"] > 0,
                               "reused": True})
    if not candidates:
        raise ValueError("no completed core Q1 checkpoints at updates 100/500/1000 are available")
    eligible = [candidate for candidate in candidates if all(candidate["point_guardrails"].values())]
    for candidate in candidates:
        candidate["eligible_by_point_guardrails"] = candidate in eligible
    if eligible:
        selected = sorted(eligible, key=lambda row: (-row["metrics"]["expected_raw_reward"],
                                                      row["metrics"]["kl_to_sft"], row["update"]))[0]
    else:
        selected = {"run_id": "SFT", "checkpoint": "artifacts/reference/runs/sft_strong_seed17/best.pt",
                    "update": 0, "method": "SFT", "support_mode": "ordinary", "beta": None,
                    "seed": 17, "metrics": None, "eligible_by_point_guardrails": False,
                    "is_safe_default": True}
    recipe_freezes: dict[str, Any] = {}
    recipe_keys = sorted({(row["method"], row["support_mode"], float(row["beta"])) for row in candidates})
    for method, support_mode, beta in recipe_keys:
        key = f"{method}:{support_mode}:beta={beta:g}"
        recipe_points = [row for row in candidates if row["method"] == method
                         and row["support_mode"] == support_mode and float(row["beta"]) == beta
                         and all(row["point_guardrails"].values())]
        if recipe_points:
            recipe_freezes[key] = sorted(recipe_points, key=lambda row: (
                -row["metrics"]["expected_raw_reward"], row["metrics"]["kl_to_sft"], row["update"]))[0]
        else:
            endpoint = next((row for row in candidates if row["method"] == method
                             and row["support_mode"] == support_mode and float(row["beta"]) == beta
                             and row["update"] == 1000), None)
            if endpoint is None:
                raise ValueError(f"core recipe {key} is missing its fixed update-1000 endpoint")
            recipe_freezes[key] = {**endpoint,
                                   "method": method, "support_mode": support_mode,
                                   "beta": beta, "seed": 17, "eligible_by_point_guardrails": False,
                                   "fallback_reason": "no development checkpoint passed all point guardrails",
                                   "confirmation_checkpoint": endpoint["checkpoint"],
                                   "update": 1000}
    extra_confirmation = []
    if selected.get("run_id") != "SFT" and float(selected["beta"]) != 1.0 and (
            (selected["method"] == "exact" and selected["support_mode"] == "ordinary")
            or (selected["method"] == "sampled" and selected["support_mode"] == "penalized")):
        extra_confirmation = [{"method": selected["method"], "support_mode": selected["support_mode"],
                              "beta": float(selected["beta"]), "seeds": [42, 73],
                              "update": int(selected["update"])}]
    lambda_path = artifacts / "lambda_freeze.json"
    if not lambda_path.is_file():
        raise FileNotFoundError("candidate freeze requires the pre-registered train-only lambda freeze")
    frozen_lambda = float(json.loads(lambda_path.read_text(encoding="utf-8"))["selected_lambda"])
    freeze = {
        "protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective_version": "normalized_reward_v1",
        "status": "frozen_before_confirmation", "selection_panel": "development only",
        "panel_status": "disclosed_panel_reanalysis", "independent_confirmation": False,
        "selection_rule": "Among only updates 100/500/1000, require all development point guardrails; select maximum fit expected raw reward, then lower KL, then earlier checkpoint. If a recipe has no eligible point, use update 1000 only as a labelled diagnostic/transfer checkpoint, never as an eligible candidate. If no eligible candidate exists globally, SFT is the safe selected policy.",
        "point_guardrails": {"relative_ndcg10_min": -0.02, "coverage_relative_min": -0.10,
                             "head_share_delta_max": 0.02, "history_repeat_delta_max": 0.02},
        "selected_candidate": selected, "selected_is_full_candidate": bool(
            selected.get("metrics") and selected.get("positive_fit_reward_ci")
            and selected["metrics"]["paired_to_sft"]["ndcg@10"]["ci_95_low"]
            >= -0.02 * selected["metrics"]["baseline_ndcg10"]
        ),
        "candidate_count": len(candidates), "eligible_count": len(eligible),
        "selection_updates": [100, 500, 1000],
        "primary_optimizer_endpoint": 1000,
        "support_lambda": frozen_lambda,
        "recipe_freezes": recipe_freezes,
        "additional_confirmation_runs_if_selected_beta_differs_from_1": extra_confirmation,
        "all_candidates": candidates,
        "rm_audit_policy_scores_used": False,
        "test_used": False,
    }
    target = Path(output) if output else artifacts / "candidate_freeze.json"
    if not target.is_absolute():
        target = config.path.parent.parent / target
    _json(target, freeze)
    return freeze


def _resolve_run_path(artifacts: Path, run_id: str) -> tuple[Path, bool]:
    current = artifacts / "runs" / run_id
    if (current / "run_manifest.json").is_file():
        return current, False
    reuse_path = artifacts / "reuse_manifest.json"
    if reuse_path.is_file():
        reuse = json.loads(reuse_path.read_text(encoding="utf-8"))
        entry = next((row for row in reuse.get("reused_runs", []) if row.get("run_id") == run_id), None)
        if entry:
            source = Path(entry["source_run_dir"])
            if not source.is_absolute():
                source = artifacts.parent.parent.parent / source
            if not (source / "run_manifest.json").is_file():
                raise FileNotFoundError(f"reused source run is missing: {source}")
            actual = sha256_file(source / "run_manifest.json")
            if actual != entry.get("source_run_manifest_sha256"):
                raise ValueError(f"reused source run manifest changed: {source}")
            return source, True
    raise FileNotFoundError(f"run is not present in correction root or reuse manifest: {run_id}")


def rescore_reused_exact_candidates(config_path: str | Path) -> dict[str, Any]:
    """Score exact checkpoints on the corrected panel/cache without touching source runs."""
    config, _, artifacts = _root_and_processed(config_path)
    cache_manifest, _, panels, arrays = _load_cache(config_path, "fit", "ratio")
    eval_rows = panels["development"] + panels["confirmation"]
    dev_n = len(panels["development"])
    baseline = evaluate_actor(config_path, _main_actor_checkpoint(config), panels["development"],
                              arrays["evaluation"][:dev_n], cache_manifest, arrays, 1.0,
                              "ordinary", 0.0, reward_head="ratio")
    candidates = []
    reused_ids = [
        "q1_exact_ordinary_b0p1_s17", "q1_exact_ordinary_b1_s17", "q1_exact_ordinary_b3_s17",
        "q1_exact_penalized_b0p1_s17", "q1_exact_penalized_b1_s17", "q1_exact_penalized_b3_s17",
    ]
    for run_id in reused_ids:
        run_path, reused = _resolve_run_path(artifacts, run_id)
        manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
        for step in (100, 500, 1000):
            checkpoint = run_path / "snapshots" / f"update_{step:05d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"required reused exact checkpoint missing: {checkpoint}")
            metrics = _eval_checkpoint(config_path, run_path, step, eval_rows,
                                       arrays["evaluation"], cache_manifest, arrays,
                                       float(manifest["beta"]), manifest["support_mode"],
                                       float(manifest.get("lambda_support", 0.0)), "ratio", baseline)
            candidates.append({
                "run_id": run_id, "run_dir": str(run_path.resolve()),
                "source_run_manifest_sha256": sha256_file(run_path / "run_manifest.json"),
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                "update": step, "method": "exact", "support_mode": manifest["support_mode"],
                "beta": float(manifest["beta"]), "seed": 17, "reused": reused,
                "source_objective_version": "exact_normalized_reward_v1_verified",
                "metrics": metrics,
            })
    result = {
        "status": "completed", "objective_version": "normalized_reward_v1",
        "panel_status": "disclosed_panel_reanalysis", "independent_confirmation": False,
        "panel_ids_sha256": panels["panel_ids_sha256"],
        "cache_manifest_sha256": sha256_file(_cache_root(config, _root_and_processed(config_path)[2])
                                              / "main" / "fit_ratio" / "cache_manifest.json"),
        "candidates": candidates,
        "source_note": "Exact training used the same normalized objective before this correction; all evaluation metrics were rescored in the correction root.",
    }
    _json(artifacts / "reused_exact_development.json", result)
    return result


def _main_actor_checkpoint(config) -> Path:
    return config.path.parent.parent / "artifacts/reference/runs/sft_strong_seed17/best.pt"


def _confirmation_bootstrap(differences: dict[int, float], *, seed: int,
                            repetitions: int = 2000) -> dict[str, float]:
    if not differences:
        raise ValueError("confirmation comparison has no paired users")
    users = sorted(differences)
    rng = random.Random(seed)
    estimates = [sum(differences[rng.choice(users)] for _ in users) / len(users)
                 for _ in range(repetitions)]
    estimates.sort()
    point = float(np.mean([differences[user] for user in users]))
    lower_tail = (sum(value <= 0 for value in estimates) + 1) / (repetitions + 1)
    upper_tail = (sum(value >= 0 for value in estimates) + 1) / (repetitions + 1)
    return {"mean_difference": point,
            "ci_95_low": float(estimates[int(0.025 * (repetitions - 1))]),
            "ci_95_high": float(estimates[int(0.975 * (repetitions - 1))]),
            "two_sided_bootstrap_p": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
            "n_users": len(users), "repetitions": repetitions}


def _holm_adjust(comparisons: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(comparisons, key=lambda name: comparisons[name]["two_sided_bootstrap_p"])
    adjusted = 0.0
    for rank, name in enumerate(ordered):
        raw = float(comparisons[name]["two_sided_bootstrap_p"])
        adjusted = max(adjusted, min(1.0, raw * (len(ordered) - rank)))
        comparisons[name]["holm_adjusted_p"] = adjusted
        comparisons[name]["holm_rank"] = rank + 1


def evaluate_confirmation(config_path: str | Path, freeze_path: str | Path | None = None,
                          output_path: str | Path | None = None) -> dict[str, Any]:
    config, _, artifacts = _root_and_processed(config_path)
    panels = create_panels(config_path)
    freeze_file = Path(freeze_path) if freeze_path else artifacts / "candidate_freeze.json"
    if not freeze_file.is_absolute():
        freeze_file = config.path.parent.parent / freeze_file
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_confirmation":
        raise ValueError("confirmation requires a saved pre-confirmation candidate freeze")
    cache_manifest, cache_dir, _, arrays = _load_cache(config_path, "fit", "ratio")
    matrix = arrays["evaluation"]
    dev_n = len(panels["development"])
    confirm_rows = panels["confirmation"]
    confirm_matrix = matrix[dev_n:dev_n + len(confirm_rows)]
    lambda_support = float(freeze["support_lambda"])
    sft_path = config.path.parent.parent / "artifacts/reference/runs/sft_strong_seed17/best.pt"
    sft_metrics = evaluate_actor(config_path, sft_path, confirm_rows, confirm_matrix,
                                 cache_manifest, arrays, 1.0, "ordinary", 0.0)
    confirm_metrics: list[dict[str, Any]] = []
    by_recipe_seed: dict[tuple[str, int], dict[str, Any]] = {}
    required_recipes = {
        "sampled:ordinary:beta=1", "exact:ordinary:beta=1",
        "sampled:penalized:beta=1", "exact:penalized:beta=1",
    }
    recipe_freezes = freeze.get("recipe_freezes", {})
    missing_recipes = required_recipes - set(recipe_freezes)
    if missing_recipes:
        raise ValueError(f"frozen confirmation lacks required beta=1 recipes: {sorted(missing_recipes)}")
    for recipe_key in sorted(required_recipes):
        recipe = recipe_freezes[recipe_key]
        method, support_mode, beta_text = recipe_key.split(":")
        beta = float(beta_text.split("=")[1])
        selected_update = int(recipe.get("update", 0))
        if selected_update not in {100, 500, 1000}:
            raise ValueError(f"invalid frozen confirmation checkpoint for {recipe_key}: {selected_update}")
        for seed in (42, 73):
            run_id = _run_id("ratio", method, support_mode, beta, seed)
            run_path, was_reused = _resolve_run_path(artifacts, run_id)
            manifest_path = run_path / "run_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"required confirmation run is missing: {run_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed" or manifest.get("latest_update") != 1000:
                raise ValueError(f"required confirmation run is incomplete: {run_path}")
            if manifest.get("lambda_support") != (lambda_support if support_mode == "penalized" else 0.0):
                raise ValueError(f"confirmation run lambda does not match the frozen value: {run_path}")
            checkpoint = run_path / "snapshots" / f"update_{selected_update:05d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"frozen checkpoint is missing: {checkpoint}")
            metrics = evaluate_actor(config_path, checkpoint, confirm_rows, confirm_matrix,
                                     cache_manifest, arrays, beta, support_mode,
                                     lambda_support if support_mode == "penalized" else 0.0)
            record = {"run_id": run_path.name, "recipe_key": recipe_key,
                      "selected_update_from_seed17_development": selected_update,
                      "method": method, "support_mode": support_mode,
                      "beta": beta, "seed": seed, "metrics": metrics,
                      "reused_exact_artifact": was_reused,
                      "objective_version": "normalized_reward_v1"}
            confirm_metrics.append(record)
            by_recipe_seed[(recipe_key, seed)] = metrics

    for extra in freeze.get("additional_confirmation_runs_if_selected_beta_differs_from_1", []):
        method = str(extra["method"])
        support_mode = str(extra["support_mode"])
        beta = float(extra["beta"])
        selected_update = int(extra["update"])
        for seed in (42, 73):
            run_id = _run_id("ratio", method, support_mode, beta, seed)
            run_path, was_reused = _resolve_run_path(artifacts, run_id)
            manifest_path = run_path / "run_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"selected nondefault-beta confirmation run is missing: {run_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed" or int(manifest.get("latest_update", 0)) < 1000:
                raise ValueError(f"selected nondefault-beta confirmation run is incomplete: {run_path}")
            if manifest.get("lambda_support") != (lambda_support if support_mode == "penalized" else 0.0):
                raise ValueError(f"selected nondefault-beta lambda differs from freeze: {run_path}")
            checkpoint = run_path / "snapshots" / f"update_{selected_update:05d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            metrics = evaluate_actor(config_path, checkpoint, confirm_rows, confirm_matrix,
                                     cache_manifest, arrays, beta, support_mode,
                                     lambda_support if support_mode == "penalized" else 0.0)
            confirm_metrics.append({"run_id": run_path.name, "recipe_key": "selected_nondefault_beta",
                                    "selected_update_from_seed17_development": selected_update,
                                    "method": method, "support_mode": support_mode,
                                    "beta": beta, "seed": seed, "metrics": metrics,
                                    "reused_exact_artifact": was_reused,
                                    "objective_version": "normalized_reward_v1",
                                    "role": "one preregistered nondefault-beta candidate; outside primary Holm family"})

    selected = freeze.get("selected_candidate")
    selected_eval = None
    if selected and selected.get("run_id") != "SFT":
        selected_checkpoint = Path(selected["checkpoint"])
        if not selected_checkpoint.is_file():
            raise FileNotFoundError(f"frozen selection checkpoint is missing: {selected_checkpoint}")
        selected_support = selected["support_mode"]
        selected_eval = evaluate_actor(config_path, selected_checkpoint, confirm_rows, confirm_matrix,
                                       cache_manifest, arrays, float(selected["beta"]), selected_support,
                                       lambda_support if selected_support == "penalized" else 0.0)
        confirm_metrics.append({"run_id": selected["run_id"], "role": "selected_candidate",
                                "metrics": selected_eval})

    def paired_across_seeds(left_key: str, right_key: str, metric: str) -> dict[int, float]:
        left_maps = [by_recipe_seed[(left_key, seed)]["per_user_metrics"] for seed in (42, 73)]
        right_maps = [by_recipe_seed[(right_key, seed)]["per_user_metrics"] for seed in (42, 73)]
        left_sft = [by_recipe_seed[(left_key, seed)]["per_user_metrics"] for seed in (42, 73)]
        # average paired user-level effects across rollout seeds; seeds are not treated as independent users
        user_sets = [set(left_maps[0]), set(left_maps[1]), set(right_maps[0]), set(right_maps[1]),
                     set(sft_metrics["per_user_metrics"])]
        users = set.intersection(*user_sets)
        if not users:
            raise ValueError("confirmation comparisons have no shared users")
        return {int(uid): float(np.mean([
            left_maps[index][uid][metric] - right_maps[index][uid][metric]
            for index in range(2)
        ])) for uid in users}

    sampled_q1_key = "sampled:ordinary:beta=1"
    sampled_q1_sft = {}
    for uid in set(sft_metrics["per_user_metrics"]) & set.intersection(
            *[set(by_recipe_seed[(sampled_q1_key, seed)]["per_user_metrics"]) for seed in (42, 73)]):
        sampled_q1_sft[int(uid)] = float(np.mean([
            by_recipe_seed[(sampled_q1_key, seed)]["per_user_metrics"][uid]["ndcg@10"]
            - sft_metrics["per_user_metrics"][uid]["ndcg@10"] for seed in (42, 73)]))
    comparisons = {
        "sampled_Q1_minus_SFT": _confirmation_bootstrap(sampled_q1_sft, seed=17101),
        "exact_ordinary_minus_sampled_ordinary": _confirmation_bootstrap(
            paired_across_seeds("exact:ordinary:beta=1", sampled_q1_key, "ndcg@10"), seed=17102),
        "sampled_penalized_minus_sampled_ordinary": _confirmation_bootstrap(
            paired_across_seeds("sampled:penalized:beta=1", sampled_q1_key, "ndcg@10"), seed=17103),
    }
    _holm_adjust(comparisons)
    result = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "objective_version": "normalized_reward_v1",
              "freeze_sha256": sha256_file(freeze_file),
              "status": "completed", "panel_id_sha256": panels["panel_ids_sha256"]["confirmation"],
              "panel_status": "disclosed_panel_reanalysis", "independent_confirmation": False,
              "support_lambda": lambda_support,
              "sft_baseline": {key: value for key, value in sft_metrics.items()
                               if key not in {"per_user_metrics", "per_user_expected_raw_reward"}},
              "models": confirm_metrics, "primary_holm_family": comparisons,
              "primary_unit": "user; per-user paired effects averaged over rollout seeds 42/73",
              "rm_audit_policy_scores_used_for_selection": False, "test_used": False}
    target = Path(output_path) if output_path else artifacts / "confirmation_results.json"
    if not target.is_absolute():
        target = config.path.parent.parent / target
    _json(target, result)
    return result


def math_audit() -> dict[str, Any]:
    torch.manual_seed(70017)
    dtype = torch.float64
    logits = torch.tensor([0.2, -0.4, 0.7], dtype=dtype, requires_grad=True)
    reward = torch.tensor([-0.5, 0.8, 1.1], dtype=dtype)
    ref_logits = torch.tensor([-0.1, 0.5, 0.3], dtype=dtype)
    logp = torch.log_softmax(logits, dim=-1)
    logq = torch.log_softmax(ref_logits, dim=-1)
    exact = (logp.exp() * reward).sum()
    exact_grad = torch.autograd.grad(exact, logits, retain_graph=True)[0]
    actions = torch.tensor(list(itertools.product(range(3), repeat=5)), dtype=torch.long)
    n, group = actions.shape
    selected = logp[actions]
    rewards = reward[actions]
    loo = rewards - (rewards.sum(-1, keepdim=True) - rewards) / (group - 1)
    # Every ordered action group is weighted by its exact probability; this verifies
    # the expected sampled score-function gradient, not one Monte Carlo draw.
    sequences_probability = logp.exp()[actions].prod(-1)
    expected_loss = -(sequences_probability.detach() * (loo * selected).mean(-1)).sum()
    expected_grad = torch.autograd.grad(expected_loss, logits, retain_graph=True)[0]
    loo_centered = rewards - rewards.mean(-1, keepdim=True)
    parity = torch.max(torch.abs(loo - loo_centered * group / (group - 1))).item()
    exact_kl = (logp.exp() * (logp - logq)).sum()
    kl_grad = torch.autograd.grad(exact_kl, logits)[0]
    reserved = torch.full((3,), -torch.inf, dtype=dtype)
    full_policy_log = torch.cat((reserved, logp)).unsqueeze(0)
    full_reference_log = torch.cat((reserved, logq)).unsqueeze(0)
    full_j, _ = _j(full_policy_log, full_reference_log, reward.unsqueeze(0),
                    torch.tensor([0.0, 1.0, 0.0], dtype=dtype), 0.7, "ordinary", 0.0)
    expected_full_j = exact - 0.7 * exact_kl
    # Active excess mass with a sampled group containing no unsupported item.
    support = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=dtype)
    reference_logits = torch.tensor([3.0, 3.0, 0.0, 0.0], dtype=dtype)
    active_logits = torch.tensor([0.0, 0.0, 2.0, 2.0], dtype=dtype, requires_grad=True)
    inactive_logits = torch.tensor([3.5, 3.5, 0.0, 0.0], dtype=dtype, requires_grad=True)
    ref_mass = (torch.softmax(reference_logits, dim=-1) * support).sum()

    def support_obj(z):
        mass = (torch.softmax(z, dim=-1) * support).sum()
        return torch.relu(mass - ref_mass)

    penalty = support_obj(active_logits)
    penalty_grad = torch.autograd.grad(penalty, active_logits)[0]
    inactive_penalty = support_obj(inactive_logits)
    inactive_grad = torch.autograd.grad(inactive_penalty, inactive_logits)[0]
    same_logits = reference_logits.clone().requires_grad_(True)
    initial_penalty = support_obj(same_logits)
    initial_grad = torch.autograd.grad(initial_penalty, same_logits)[0]
    eps = 1e-6
    finite = []
    for index in range(4):
        plus = active_logits.detach().clone(); plus[index] += eps
        minus = active_logits.detach().clone(); minus[index] -= eps
        finite.append(float((support_obj(plus) - support_obj(minus)) / (2 * eps)))
    sampled_supported_actions = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    no_unsupported_samples = bool(not support[sampled_supported_actions].any())
    # ``expected_grad`` is the gradient of the minimized negative policy loss;
    # negate it before comparing with the maximized expected reward gradient.
    errors = (exact_grad + expected_grad).abs()
    return {
        "status": "passed" if float(errors.max()) < 1e-10 and parity < 1e-12
                  and float(torch.max(torch.abs(penalty_grad - torch.tensor(finite, dtype=dtype)))) < 1e-6
                  and float(penalty.detach()) > 0 and no_unsupported_samples
                  and float(inactive_penalty.detach()) == 0.0 and float(inactive_grad.abs().max()) == 0.0
                  and float(initial_penalty.detach()) == 0.0 and float(initial_grad.abs().max()) == 0.0
                  and abs(float(full_j.item()) - float(expected_full_j.item())) < 1e-6
                  and bool(torch.isfinite(kl_grad).all()) else "failed",
        "exact_vs_exhaustive_sampled_gradient_max_abs": float(errors.max()),
        "loo_centered_correction_max_abs": parity,
        "exact_full_catalog_kl_gradient_finite": bool(torch.isfinite(kl_grad).all()),
        "reserved_tokens_excluded_from_reward_vector": bool(abs(float(full_j.item()) - float(expected_full_j.item())) < 1e-6),
        "support_penalty_gradient": penalty_grad.tolist(),
        "support_penalty_finite_difference": finite,
        "active_excess_with_zero_unsupported_samples": no_unsupported_samples,
        "active_penalty_value": float(penalty.detach()),
        "inactive_penalty_value": float(inactive_penalty.detach()),
        "inactive_penalty_gradient_max_abs": float(inactive_grad.abs().max()),
        "initial_policy_equal_to_reference_penalty": float(initial_penalty.detach()),
        "initial_policy_equal_to_reference_gradient_max_abs": float(initial_grad.abs().max()),
        "reward_and_reference_detached_in_training": True,
        "sampled_actions_drawn_from_full_catalog": True,
        "interpretation": "Monte Carlo expectation of leave-one-out policy gradient matches exact reward gradient on enumerated fixture.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="full_model shared exact/sampled Q1/Q2 policy matrix")
    parser.add_argument("--config", default="configs/full_model_rl.yaml")
    parser.add_argument("--mode", choices=("panels", "cache", "train", "freeze", "confirm", "b5",
                                            "math-audit", "lambda-pilot", "lambda-freeze",
                                            "exact-support-pilot", "gradient-components", "resume-audit",
                                            "production-gradient-audit", "exact-rescore"), required=True)
    parser.add_argument("--actor-checkpoint")
    parser.add_argument("--reward-checkpoint")
    parser.add_argument("--reward-variant", choices=("fit", "audit"), default="fit")
    parser.add_argument("--reward-head", choices=("ratio", "seconds"), default="ratio")
    parser.add_argument("--method", choices=("sampled", "exact"))
    parser.add_argument("--support-mode", choices=("ordinary", "penalized"), default="ordinary")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--reuse-epochs", type=int, default=1)
    parser.add_argument("--lambda-support", type=float)
    parser.add_argument("--run-kind", default="core")
    parser.add_argument("--no-development-eval", action="store_true")
    parser.add_argument("--probe-size", type=int, default=256)
    parser.add_argument("--stage", default="pilot")
    parser.add_argument("--context-source", choices=("cached", "stream"), default="cached")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "panels":
        panels = create_panels(args.config, force=args.force)
        result = {key: value for key, value in panels.items() if key not in {
            "train", "calibration", "development", "confirmation"}}
    elif args.mode == "cache":
        result = build_reward_cache(args.config, reward_checkpoint=args.reward_checkpoint,
                                    variant=args.reward_variant, reward_head=args.reward_head, force=args.force)
    elif args.mode == "train":
        if not args.actor_checkpoint or not args.method:
            parser.error("--train requires --actor-checkpoint and --method")
        result = train_run(args.config, args.actor_checkpoint, method=args.method,
                           support_mode=args.support_mode, beta=args.beta, seed=args.seed,
                           reward_head=args.reward_head, updates=args.updates,
                           checkpoint_interval=args.checkpoint_interval, eval_interval=args.eval_interval,
                           run_dir=args.run_dir, resume=args.resume,
                           stop_after_updates=args.stop_after_updates, reuse_epochs=args.reuse_epochs,
                           lambda_support_value=args.lambda_support, run_kind=args.run_kind,
                           evaluate_development=not args.no_development_eval,
                           context_source=args.context_source)
    elif args.mode == "freeze":
        result = freeze_selection(args.config, output=args.output)
    elif args.mode == "confirm":
        result = evaluate_confirmation(args.config, output_path=args.output)
    elif args.mode == "b5":
        if not args.actor_checkpoint:
            parser.error("--mode b5 requires --actor-checkpoint")
        result = evaluate_b5(args.config, args.actor_checkpoint, beta=args.beta, reward_head=args.reward_head)
    elif args.mode == "lambda-pilot":
        if not args.actor_checkpoint:
            parser.error("--mode lambda-pilot requires --actor-checkpoint")
        result = run_lambda_pilot(args.config, args.actor_checkpoint)
    elif args.mode == "lambda-freeze":
        result = freeze_lambda_pilot(args.config)
    elif args.mode == "exact-support-pilot":
        if not args.actor_checkpoint:
            parser.error("--mode exact-support-pilot requires --actor-checkpoint")
        result = run_exact_support_pilot(args.config, args.actor_checkpoint)
    elif args.mode == "gradient-components":
        if not args.actor_checkpoint:
            parser.error("--mode gradient-components requires --actor-checkpoint")
        result = gradient_component_diagnostics(args.config, args.actor_checkpoint,
                                                probe_size=args.probe_size, stage=args.stage)
    elif args.mode == "resume-audit":
        if not args.actor_checkpoint:
            parser.error("--mode resume-audit requires --actor-checkpoint")
        result = verify_resume_equivalence(args.config, args.actor_checkpoint)
    elif args.mode == "production-gradient-audit":
        result = production_gradient_audit()
        _, _, artifacts = _root_and_processed(args.config)
        target = artifacts / "verifications" / "production_gradient_audit.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        _json(target, result)
    elif args.mode == "exact-rescore":
        result = rescore_reused_exact_candidates(args.config)
    else:
        result = math_audit()
        _, _, artifacts = _root_and_processed(args.config)
        target = Path(args.output) if args.output else artifacts / "verifications" / "math_audit.json"
        if not target.is_absolute():
            target = args.config and load_config(args.config).path.parent.parent / target
        target.parent.mkdir(parents=True, exist_ok=True)
        _json(target, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
