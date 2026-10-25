"""Build a panel_tools.1-compatible policy cache in the isolated panel_tools.2 namespace."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import torch

from tlab.data.store import SequenceStore
from tlab.panel_tools.data import history_for_target
from tlab.panel_tools.evaluator import _collate_histories, load_actor_checkpoint
from tlab.residual_support.cache import _load_panel, _mapping_sha
from tlab.residual_support.features import (
    CANDIDATE_FEATURE_NAMES, POLICY_FEATURE_NAMES, candidate_features,
    history_context_features, policy_feature_matrix,
)
from tlab.residual_support.rm import load_priors, sha256_file

from .protocol import ARTIFACTS, ROOT, record_operation


def build_common_cache(batch_size: int = 16) -> dict[str, Any]:
    started = time.perf_counter()
    target = ARTIFACTS / "cache/common_open_6655"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite panel_tools.2 cache: {target}")
    target.mkdir(parents=True)
    panel_path = ROOT / "artifacts/panel_tools/evaluation/frozen_primary_validation_panel.jsonl"
    panel = _load_panel(panel_path)
    temporal_path = ARTIFACTS / "evaluation/temporal_eligibility_rows.csv"
    eligible_event_ids = set(
        pl.read_csv(temporal_path)
        .filter((pl.col("panel") == "common_open_6655") & pl.col("common_temporally_eligible"))
        .get_column("event_id").to_list()
    )
    panel = [row for row in panel if row.event_id in eligible_event_ids]
    if len(panel) != 6655:
        raise ValueError(f"temporal gate selected {len(panel)} common rows, expected 6655")

    processed = ROOT / "data/processed_full_model_tminus14"
    protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    store = SequenceStore.load_cache(processed / "sequence_index",
                                     expected_source_sha256=protocol["processed_listens_sha256"])
    catalog = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    tokens = np.asarray(catalog["token_id"].to_list(), dtype=np.int64)
    if not np.array_equal(tokens, np.arange(3, 3 + len(tokens), dtype=np.int64)):
        raise ValueError("catalog token IDs are not contiguous")
    mapping_sha = _mapping_sha(item_ids, tokens)
    rm_root = ARTIFACTS / ".." / "residual_support/rm"
    rm_root = rm_root.resolve()
    rm_meta = json.loads((rm_root / "rm_diagnostics.json").read_text(encoding="utf-8"))
    model_path = ROOT / rm_meta["models"]["optimized_rm"]["path"]
    model_path = model_path.resolve()
    model = joblib.load(model_path)
    prior_path = ROOT / rm_meta["item_prior_artifacts"]["policy_path"]
    prior_path = prior_path.resolve()
    priors = load_priors(prior_path, expected_mapping_sha256=mapping_sha)
    reward_scale = json.loads((rm_root / "reward_scale.json").read_text(encoding="utf-8"))
    actor_path = ROOT / "models/actor/strong.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, actor_metadata = load_actor_checkpoint(actor_path, vocab_size=len(item_ids) + 3,
                                                  n_positions=256, n_layer=4, n_embd=256, n_head=4, device=device)
    if actor_metadata["checkpoint_sha256"] != sha256_file(actor_path):
        raise ValueError("strong actor checkpoint hash changed during cache creation")
    embeddings = actor.lm.transformer.wte.weight.detach()[3:].float().cpu().numpy()
    counts = np.asarray(rm_meta["rm_fit_item_count"]["counts_by_catalog_item"], dtype=np.int64)

    n, k, n_features, catalog_size = len(panel), 500, len(POLICY_FEATURE_NAMES), len(item_ids)
    features = np.memmap(target / "candidate_features.bin", mode="w+", dtype=np.float32, shape=(n, k, n_features))
    ref_cond = np.memmap(target / "reference_conditional_logp.bin", mode="w+", dtype=np.float32, shape=(n, k))
    support_indices = np.memmap(target / "support_indices.bin", mode="w+", dtype=np.int32, shape=(n, k))
    support_mass = np.memmap(target / "support_mass.bin", mode="w+", dtype=np.float32, shape=(n,))
    rewards = np.memmap(target / "raw_candidate_rewards.bin", mode="w+", dtype=np.float32, shape=(n, k))
    ref_full = np.memmap(target / "reference_logp_full.bin", mode="w+", dtype=np.float32, shape=(n, catalog_size))
    support_covered = 0
    actor.eval()
    for start in range(0, n, batch_size):
        rows = panel[start:start + batch_size]
        histories = [history_for_target(store, row, 200) for row in rows]
        input_ids, attention = _collate_histories(histories, device)
        with torch.inference_mode():
            logp = actor.next_action_log_probs(input_ids, attention)[:, 3:].float()
            logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
            ref_batch = logp.cpu().numpy().astype(np.float32, copy=False)
        for j, row in enumerate(rows):
            i = start + j
            ref = ref_batch[j]
            order = np.lexsort((item_ids, -ref))
            chosen = order[:k].astype(np.int32, copy=False)
            support_lp = ref[chosen].astype(np.float64)
            mass = float(np.exp(support_lp).sum())
            cond = support_lp - np.logaddexp.reduce(support_lp)
            user = store.users[row.user_index]
            context = history_context_features(user, row.position)
            raw = candidate_features(user, row.position, item_ids[chosen], catalog_indices=chosen, priors=priors)
            hstart = max(0, row.position - 1000)
            history_tokens = np.asarray(user.tokens[hstart:row.position], dtype=np.int64)
            matrix = policy_feature_matrix(
                context, raw, rm_fit_counts=counts, candidate_catalog_indices=chosen,
                reference_logp=ref[chosen], item_embeddings=embeddings, history_tokens=history_tokens,
            )
            features[i] = matrix
            ref_cond[i] = cond.astype(np.float32)
            support_indices[i] = chosen
            support_mass[i] = mass
            rewards[i] = np.clip(model.predict(matrix[:, :35]), 0.0, 1.0).astype(np.float32)
            ref_full[i] = ref
            if row.token_id >= 3 and row.token_id - 3 in chosen:
                support_covered += 1
        if start == 0 or (start // batch_size) % 100 == 0:
            print(f"common cache: {min(start + len(rows), n):,}/{n:,}", flush=True)
    for array in (features, ref_cond, support_indices, support_mass, rewards, ref_full):
        array.flush()
    del features, ref_cond, support_indices, support_mass, rewards, ref_full

    names = ("candidate_features", "reference_conditional_logp", "support_indices", "support_mass",
             "raw_candidate_rewards", "reference_logp_full")
    dtype_by_name = {"candidate_features": "float32", "reference_conditional_logp": "float32",
                     "support_indices": "int32", "support_mass": "float32", "raw_candidate_rewards": "float32",
                     "reference_logp_full": "float32"}
    shapes = {"candidate_features": [n, k, n_features], "reference_conditional_logp": [n, k],
              "support_indices": [n, k], "support_mass": [n], "raw_candidate_rewards": [n, k],
              "reference_logp_full": [n, catalog_size]}
    event_sha = hashlib.sha256("\n".join(row.event_id for row in panel).encode()).hexdigest()
    manifest = {
        "schema_version": "tlab_policy_diagnostics_policy_cache_adapter_v1",
        "protocol_id": "tlab_yambda_policy_diagnostics_retrospective_2026-10-22",
        "split": "common_open_6655_temporally_valid",
        "panel_path": panel_path.relative_to(ROOT).as_posix(), "panel_sha256": sha256_file(panel_path),
        "panel_event_ids_sha256": event_sha, "n_contexts": n, "n_users": len({x.uid for x in panel}),
        "catalog_size": catalog_size, "raw_token_mapping_sha256": mapping_sha,
        "feature_schema_version": "residual_support_candidate_action_features_v1", "feature_names": list(POLICY_FEATURE_NAMES),
        "support_size": k, "support_mass": "strong reference top-500 mass; all catalog probability outside support unchanged",
        "actor": {"path": actor_path.relative_to(ROOT).as_posix(), **actor_metadata},
        "optimized_rm": {"path": model_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(model_path)},
        "external_audit": {"path": rm_meta["models"]["external_audit_predictor"]["path"],
                           "sha256": rm_meta["models"]["external_audit_predictor"]["sha256"]},
        "item_priors": {"path": prior_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(prior_path),
                        "cutoff_raw_exclusive": priors.cutoff_raw_exclusive},
        "reward_normalization": {k: reward_scale[k] for k in ("mean", "std", "std_floor", "version")},
        "target_support_coverage": support_covered / n, "reference_full_catalog_saved": True,
        "artifact_arrays": {name: {"shape": shapes[name], "dtype": dtype_by_name[name]} for name in names},
        "runtime": {"cache_build_seconds": round(time.perf_counter() - started, 3), "device": str(device), "batch_size": batch_size},
        "label": "historically open but row-wise temporally valid descriptive panel; not an independent test",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "panel_rows.jsonl").write_text("".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in panel), encoding="utf-8")
    duration = time.perf_counter() - started
    record_operation("build_common_6655_policy_cache", duration, bucket="A", category="preprocessing",
                     evidence=[(target / "manifest.json").relative_to(ROOT).as_posix()])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.parse_args()
    result = build_common_cache(batch_size=parser.parse_args().batch_size)
    print(json.dumps({"split": result["split"], "n_contexts": result["n_contexts"],
                      "target_support_coverage": result["target_support_coverage"], "runtime": result["runtime"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
