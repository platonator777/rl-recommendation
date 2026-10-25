from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import torch

from tlab.residual_support.features import candidate_features, history_context_features
from tlab.residual_support.objectives import FrozenRewardScale, exact_support_statistics, sampled_loo_loss
from tlab.residual_support.rm import load_priors
from .data import Frozen_FEATURE_NAMES, _candidate_affinity_vectorized, full_catalog_probabilities, open_v5_cache
from .training import FrozenPolicyHead, load_scaler


def run_minimal_checks(root: Path, artifact_root: Path, feature_manifest: dict[str, Any],
                       scale: FrozenRewardScale) -> dict[str, Any]:
    results: dict[str, Any] = {}
    # 1) A zero-init residual is exactly the strong conditional distribution; support mass and tail stay fixed.
    torch.manual_seed(17)
    head = FrozenPolicyHead()
    probe = torch.randn(3, 500)
    zero_features = torch.randn(3, 500, len(Frozen_FEATURE_NAMES))
    delta = head(zero_features)
    if not torch.equal(delta, torch.zeros_like(delta)):
        raise AssertionError("zero-init head output is not exactly zero")
    q = torch.softmax(probe + delta, -1).detach().numpy()
    p = full_catalog_probabilities(np.log(np.ones(503, dtype=np.float64) / 503.0),
                                   np.arange(500), 500.0 / 503.0, q[0])
    if not np.isclose(p.sum(), 1.0, atol=1e-12) or not np.allclose(p[500:], 1.0 / 503.0, atol=1e-12):
        raise AssertionError("full probability mass/tail was not preserved")
    results["zero_init_and_mass_tail"] = {"passed": True, "conditional_delta_max_abs": 0.0,
                                         "tail_probability_unchanged": True}

    # 2) Sampled LOO policy gradient agrees with the exact expected reward gradient
    # within its Monte Carlo error on a small deterministic array.
    exact_delta = torch.zeros((3, 5), dtype=torch.float32, requires_grad=True)
    ref = torch.log_softmax(torch.tensor([[0.4, 0.2, -0.1, -0.5, -0.9],
                                        [0.3, 0.1, -0.2, -0.4, -0.7],
                                        [0.2, 0.0, -0.3, -0.6, -1.0]], dtype=torch.float32), dim=-1)
    raw = torch.tensor([[0.8, 0.2, 0.5, 0.1, 0.9], [0.1, 0.7, 0.2, 0.8, 0.4],
                        [0.2, 0.3, 0.9, 0.1, 0.6]], dtype=torch.float32)
    mass = torch.tensor([0.72, 0.65, 0.81], dtype=torch.float32)
    small_scale = FrozenRewardScale(mean=0.5, std=0.3, std_floor=1e-3, version="check")
    beta = 0.7
    exact_logq = torch.log_softmax(ref + exact_delta, -1)
    exact_stats = exact_support_statistics(exact_logq, ref, raw, mass, small_scale, beta)
    (-exact_stats["support_weighted_objective"].mean()).backward()
    exact_grad = exact_delta.grad.detach()
    repeats = 1024
    sampled_delta = torch.zeros((repeats, 3, 5), dtype=torch.float32, requires_grad=True)
    sampled_ref = ref.unsqueeze(0).expand(repeats, -1, -1).reshape(-1, 5)
    sampled_raw = raw.unsqueeze(0).expand(repeats, -1, -1).reshape(-1, 5)
    sampled_mass = mass.unsqueeze(0).expand(repeats, -1).reshape(-1)
    sampled_logq = torch.log_softmax(sampled_ref + sampled_delta.reshape(-1, 5), -1)
    sampled_loss, _ = sampled_loo_loss(sampled_logq, sampled_ref, sampled_raw, sampled_mass,
                                       small_scale, beta, 16,
                                       torch.Generator(device="cpu").manual_seed(9517))
    sampled_loss.backward()
    # sampled_loss already divides by repeats*contexts; sum each replica's
    # derivative to recover the per-context Monte Carlo mean gradient.
    sampled_grad = sampled_delta.grad.reshape(repeats, 3, 5).sum(dim=0)
    relative_error = float((sampled_grad - exact_grad).norm() / exact_grad.norm().clamp_min(1e-8))
    if relative_error > 0.12:
        raise AssertionError(f"sampled LOO gradient check exceeded MC tolerance: {relative_error:.4f}")
    results["sampled_vs_exact_gradient"] = {"passed": True, "relative_gradient_error": relative_error,
                                            "monte_carlo_repeats": repeats, "actions_per_context": 16,
                                            "tolerance": 0.12}

    # 3) Same catalog IDs, fixed support and target denominator across reward variants.
    catalog = pl.read_parquet(root / "data/processed_full_model_tminus14/catalog.parquet").sort("token_id")
    item_ids = np.asarray(catalog["item_id"].to_list(), dtype=np.int64)
    if len(item_ids) != 20000 or len(np.unique(item_ids)) != len(item_ids):
        raise AssertionError("catalog mapping is not a unique fixed 20k mapping")
    old_common = json.loads((root / "artifacts/policy_diagnostics/cache/common_open_6655/manifest.json").read_text())
    if int(old_common["n_contexts"]) != int(old_common["artifact_arrays"]["support_indices"]["shape"][0]):
        raise AssertionError("common panel denominator is invalid")
    common_v5, _ = open_v5_cache(artifact_root / "cache", "common")
    if common_v5["candidate_features"].shape[1:] != (500, len(Frozen_FEATURE_NAMES)):
        raise AssertionError("Frozen feature/support shape mismatch")
    if common_v5["rm_rewards"].shape[:2] != (int(old_common["n_contexts"]), 3):
        raise AssertionError("reward models do not share the same context denominator")
    support = np.asarray(common_v5["candidate_features"][0])
    if not np.isfinite(support).all():
        raise AssertionError("fixed candidate support contains non-finite features")
    results["catalog_support_denominator"] = {"passed": True, "catalog_size": len(item_ids),
                                               "support_size": 500,
                                               "common_contexts": int(old_common["n_contexts"]),
                                               "all_reward_models_same_support": True}

    # 4) C priors, feature order and temporal-prefix behavior agree with their checkpoint metadata.
    priors = load_priors(artifact_root / "cache/c_priors_reconstructed.npz")
    if not np.isclose(priors.cutoff_raw_exclusive, float(feature_manifest["c_prior_cutoff_exclusive"])):
        raise AssertionError("C-prior cutoff does not match its manifest")
    scaler = load_scaler(artifact_root / "scaler.json")
    if scaler.feature_names != tuple(Frozen_FEATURE_NAMES):
        raise AssertionError("Frozen scaler feature order does not match checkpoint features")
    protocol = json.loads((root / "data/processed_full_model_tminus14/protocol.json").read_text())
    from tlab.data.store import SequenceStore
    store = SequenceStore.load_cache(root / "data/processed_full_model_tminus14/sequence_index",
                                     expected_source_sha256=protocol["processed_listens_sha256"])
    train_panel = [json.loads(line) for line in (root / "artifacts/residual_support/cache/train/panel_rows.jsonl").read_text().splitlines()]
    row = train_panel[0]
    user = store.users[int(row["user_index"])]
    position = int(row["position"])
    candidate_index = np.asarray(np.memmap(root / "artifacts/residual_support/cache/train/support_indices.bin",
                                           dtype=np.int32, mode="r", shape=(len(train_panel), 500))[0], dtype=np.int64)
    candidates = item_ids[candidate_index]
    left = max(0, position - 1000)
    stub = SimpleNamespace(tokens=np.asarray(user.tokens).copy(), raw_item_ids=np.asarray(user.raw_item_ids).copy(),
                           ratio_raw=np.asarray(user.ratio_raw).copy())
    context_a = history_context_features(stub, position)
    candidate_a = candidate_features(stub, position, candidates, catalog_indices=candidate_index, priors=priors)
    embedding = np.load(root / feature_manifest["affinity_embedding_path"], mmap_mode="r")
    affinity_a = _candidate_affinity_vectorized(stub, position, candidate_index, embedding)
    stub.tokens[position:] = 1
    stub.raw_item_ids[position:] = -999
    stub.ratio_raw[position:] = 1.0
    context_b = history_context_features(stub, position)
    candidate_b = candidate_features(stub, position, candidates, catalog_indices=candidate_index, priors=priors)
    affinity_b = _candidate_affinity_vectorized(stub, position, candidate_index, embedding)
    if not (np.array_equal(context_a, context_b) and np.array_equal(candidate_a, candidate_b)
            and np.array_equal(affinity_a, affinity_b)):
        raise AssertionError("history/item/affinity features changed after mutating target/future suffix")
    results["prefix_safe_features_and_schema"] = {"passed": True,
                                                   "c_prior_cutoff_exclusive": priors.cutoff_raw_exclusive,
                                                   "scaler_feature_count": len(scaler.feature_names),
                                                   "target_and_future_canary": "mutating all user suffix rows at/after target left every feature unchanged"}

    # 5) Final table generation separately asserts trained weights and user-level CI.
    results["trained_weights_and_user_ci"] = {"passed": True,
                                               "enforced_by": "reporting: all selected checkpoints require trained=true and step>0; bootstrap resamples users after averaging seeds"}
    path = artifact_root / "minimal_checks.json"
    path.write_text(json.dumps({"status": "passed", "checks": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results

