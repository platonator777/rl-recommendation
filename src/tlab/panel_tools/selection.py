from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tlab.data.store import SequenceStore
from tlab.panel_tools.data import PanelTarget, panel_manifest, stable_event_id


def _hash_order(seed: int, event_id: str) -> bytes:
    return hashlib.sha256(f"{int(seed)}:{event_id}".encode("ascii")).digest()


def select_one_event_per_user(panel: Sequence[PanelTarget], *, seed: int = 17) -> list[PanelTarget]:
    """Choose one pre-frozen event per user by stable event hash, independent of outcome/model."""
    selected: dict[int, PanelTarget] = {}
    for target in panel:
        current = selected.get(int(target.uid))
        if current is None or (_hash_order(seed, target.event_id), target.event_id) < (
            _hash_order(seed, current.event_id), current.event_id
        ):
            selected[int(target.uid)] = target
    return sorted(selected.values(), key=lambda row: (row.uid, row.position))


def select_a_rl_training_contexts(
    store: SequenceStore,
    *,
    source_sha256: str,
    count: int = 512,
    seed: int = 17,
) -> list[PanelTarget]:
    """Select one fixed-hash A_RL context per user before model scoring."""
    if count < 1:
        raise ValueError("policy training context count must be positive")
    per_user: list[tuple[bytes, int, PanelTarget]] = []
    for user_index, user in enumerate(store.users):
        if len(user.rl_anchors) == 0:
            continue
        candidates: list[tuple[bytes, str, int, int]] = []
        for raw_position in user.rl_anchors:
            position = int(raw_position)
            raw_item_id = int(user.raw_item_ids[position])
            if raw_item_id < 0:
                continue
            event_id = stable_event_id(source_sha256, user.uid, position, raw_item_id)
            candidates.append((_hash_order(seed, event_id), event_id, position, raw_item_id))
        if not candidates:
            continue
        _, event_id, position, raw_item_id = min(candidates, key=lambda row: (row[0], row[1]))
        raw_ratio = float(user.ratio_raw[position])
        ratio = raw_ratio if np.isfinite(raw_ratio) else None
        prior = np.frombuffer(user.raw_item_ids, dtype=np.int64)[:position]
        is_repeat = bool(np.any(prior == raw_item_id))
        chosen = PanelTarget(
            event_id=event_id,
            uid=int(user.uid),
            user_index=user_index,
            position=position,
            raw_item_id=raw_item_id,
            token_id=int(user.tokens[position]),
            ratio=ratio,
            positive=bool(ratio is not None and ratio >= 0.5),
            is_new=not is_repeat,
            is_repeat=is_repeat,
        )
        user_key = hashlib.sha256(f"{int(seed)}:uid:{int(user.uid)}".encode("ascii")).digest()
        per_user.append((user_key, int(user.uid), chosen))
    per_user.sort(key=lambda entry: (entry[0], entry[1]))
    return sorted([entry[2] for entry in per_user[:count]], key=lambda row: (row.uid, row.position))


def freeze_reward_scale_from_fit(
    store: SequenceStore,
    *,
    source_sha256: str,
    max_events_per_user: int = 5,
    seed: int = 17,
    std_floor: float = 1e-3,
) -> dict[str, Any]:
    """Fit the shared objective normalization on hash-selected F_RM fit labels only."""
    if max_events_per_user < 1 or std_floor <= 0:
        raise ValueError("reward-scale sample size and std floor must be positive")
    selected: list[tuple[int, int, str, float]] = []
    selected_before_label_read = 0
    for user in store.users:
        if user.rm_user_partition != "fit":
            continue
        candidates = []
        for raw_position in user.rm_fit_positions:
            position = int(raw_position)
            raw_item = int(user.raw_item_ids[position])
            if raw_item < 0:
                continue
            event_id = stable_event_id(source_sha256, user.uid, position, raw_item)
            candidates.append((_hash_order(seed, event_id), event_id, position, raw_item))
        candidates.sort()
        chosen = candidates[:max_events_per_user]
        selected_before_label_read += len(chosen)
        for _, event_id, position, _ in chosen:
            raw_ratio = float(user.ratio_raw[position])
            if np.isfinite(raw_ratio):
                selected.append((int(user.uid), position, event_id, min(1.0, max(0.0, raw_ratio))))
    if not selected:
        raise ValueError("no finite hash-selected F_RM labels for the frozen reward scale")
    values = np.asarray([row[3] for row in selected], dtype=np.float64)
    event_ids = sorted(row[2] for row in selected)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return {
        "version": "frozen_reward_scale_v1",
        "mean": mean,
        "std": std,
        "effective_std": max(std, float(std_floor)),
        "std_floor": float(std_floor),
        "std_ddof": 0,
        "n_selected_before_finite_label_filter": selected_before_label_read,
        "n_finite_fit_labels": int(len(values)),
        "n_fit_users": int(len({row[0] for row in selected})),
        "source_population": "F_RM labels from RM-fit user partition only",
        "selection_rule": "up to max_events_per_user selected by SHA256(seed:event_id) before reading target ratio",
        "max_events_per_user": int(max_events_per_user),
        "seed": int(seed),
        "ratio_clip": [0.0, 1.0],
        "event_ids_sha256": hashlib.sha256("\n".join(event_ids).encode("ascii")).hexdigest(),
        "processed_listens_sha256": source_sha256,
        "shared_by": ["sampled_loo", "exact_expected_reward", "analytic_q_distillation", "fixed-support analytic policy"],
        "sampled_and_exact_use_identical_scale": True,
        "evaluation_label": "train-derived normalization; not a policy-quality result",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_inside_workspace(root: Path, destination: Path, text: str) -> None:
    root = root.resolve()
    destination = destination.resolve()
    submission = (root / "submission").resolve()
    if destination == submission or submission in destination.parents:
        raise ValueError("panel_tools selection outputs may not be written inside submission/")
    if root not in destination.parents:
        raise ValueError("panel_tools selection outputs must remain inside the workspace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen panel_tools selection artifact: {destination}")
    destination.write_text(text, encoding="utf-8")


def freeze_panel_tools_selection(
    workspace: str | Path,
    *,
    source_panel_path: str | Path,
    policy_config_path: str | Path,
    actor_checkpoint_path: str | Path,
    catalog_path: str | Path,
    reward_model_paths: Sequence[str | Path],
    seed: int = 17,
) -> dict[str, Any]:
    """Freeze the one-event/user panel and registered policy matrix before scoring it."""
    root = Path(workspace).resolve()
    source_panel = Path(source_panel_path)
    if not source_panel.is_absolute():
        source_panel = root / source_panel
    config = Path(policy_config_path)
    if not config.is_absolute():
        config = root / config
    actor = Path(actor_checkpoint_path)
    if not actor.is_absolute():
        actor = root / actor
    catalog = Path(catalog_path)
    if not catalog.is_absolute():
        catalog = root / catalog
    model_paths = [Path(path) if Path(path).is_absolute() else root / path for path in reward_model_paths]
    for path in [source_panel, config, actor, catalog, *model_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)

    targets = [PanelTarget(**json.loads(line)) for line in source_panel.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_one_event_per_user(targets, seed=seed)
    if len(selected) != len({row.uid for row in targets}):
        raise AssertionError("one-event panel selection failed to retain each pre-frozen user")

    output_dir = root / "artifacts/panel_tools/evaluation"
    panel_path = output_dir / "frozen_primary_validation_panel.jsonl"
    manifest_path = output_dir / "frozen_primary_validation_panel.json"
    policy_train_panel_path = output_dir / "frozen_a_rl_policy_train_panel.jsonl"
    policy_train_manifest_path = output_dir / "frozen_a_rl_policy_train_panel.json"
    scale_path = root / "artifacts/panel_tools/policy/frozen_reward_scale.json"
    selection_path = root / "artifacts/panel_tools/frozen_selection.json"
    protocol = json.loads((root / "artifacts/panel_tools/protocol.json").read_text(encoding="utf-8"))
    processed_sha = protocol["dataset"]["processed_full_model_tminus14_listens_sha256"]
    processed = root / protocol["dataset"].get("processed_full_model_tminus14_dir", "data/processed_full_model_tminus14")
    store = SequenceStore.load_cache(processed / "sequence_index", expected_source_sha256=processed_sha)
    policy_train_panel = select_a_rl_training_contexts(store, source_sha256=processed_sha, count=512, seed=seed)
    reward_scale = freeze_reward_scale_from_fit(store, source_sha256=processed_sha, max_events_per_user=5, seed=seed)
    if not policy_train_panel:
        raise ValueError("no A_RL policy training contexts are available")
    policy_train_manifest = panel_manifest(
        policy_train_panel, split="A_RL", seed=seed, source_sha256=processed_sha,
    )
    policy_train_manifest.update({
        "selection_rule": "one anchor per user; minimum SHA256(seed:event_id), then users ordered by SHA256(seed:uid); no target-ratio/model-score filter",
        "requested_contexts": 512,
        "evaluation_label": "training-context sample for model-based policy optimization; no causal interpretation",
    })
    panel_info = panel_manifest(selected, split="validation", seed=seed, source_sha256=processed_sha)
    panel_info.update({
        "selection_parent_panel_sha256": _sha256(source_panel),
        "selection_parent_n_events": len(targets),
        "one_event_per_user_rule": "minimum SHA256(seed:event_id) among each user's pre-frozen panel rows",
        "no_ratio_catalog_or_score_filter": True,
        "evaluation_label": "frozen retrospective evaluation",
    })
    _write_new_inside_workspace(root, panel_path, "".join(
        json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n" for row in selected
    ))
    _write_new_inside_workspace(root, manifest_path, json.dumps(panel_info, indent=2, ensure_ascii=False) + "\n")
    _write_new_inside_workspace(root, policy_train_panel_path, "".join(
        json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n" for row in policy_train_panel
    ))
    _write_new_inside_workspace(root, policy_train_manifest_path,
                                json.dumps(policy_train_manifest, indent=2, ensure_ascii=False) + "\n")
    _write_new_inside_workspace(root, scale_path, json.dumps(reward_scale, indent=2, ensure_ascii=False) + "\n")

    evaluation_config = root / "configs/panel_tools_primary_eval.yaml"
    budget_path = root / "artifacts/panel_tools/frozen_budget_plan.json"
    resource_plan_path = root / "artifacts/panel_tools/resource_plan.json"
    fixed_code = sorted((root / "src/tlab/panel_tools").glob("*.py"))
    selection = {
        "schema_version": 1,
        "protocol_id": "tlab_yambda_policy_diagnostics026-10-21_r1",
        "status": "frozen_before_policy_scoring",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_id": "frozen_core_policy_matrix_v1",
        "program": "reduced 24-hour core program from RESEARCH_PLAN_panel_tools.md sections 15.1-15.3",
        "primary_endpoint": "user_macro_ndcg_at_10",
        "evaluation_label": "frozen retrospective evaluation; not an independent test",
        "panel": {
            "path": str(panel_path.relative_to(root)),
            "manifest_path": str(manifest_path.relative_to(root)),
            "event_ids_sha256": panel_info["event_ids_sha256"],
            "n_users": panel_info["n_users"],
            "n_events": panel_info["n_events"],
            "seed": seed,
            "selection_rule": panel_info["one_event_per_user_rule"],
            "source_panel_path": str(source_panel.relative_to(root)),
            "source_panel_sha256": _sha256(source_panel),
        },
        "policy_training_panel": {
            "path": str(policy_train_panel_path.relative_to(root)),
            "manifest_path": str(policy_train_manifest_path.relative_to(root)),
            "event_ids_sha256": policy_train_manifest["event_ids_sha256"],
            "n_users": policy_train_manifest["n_users"],
            "n_events": policy_train_manifest["n_events"],
            "seed": seed,
            "split": "A_RL",
        },
        "reward_scale": {"path": str(scale_path.relative_to(root)), **reward_scale},
        "fixed_assets": {
            "policy_config": {"path": str(config.relative_to(root)), "sha256": _sha256(config)},
            "primary_evaluation_config": {"path": str(evaluation_config.relative_to(root)), "sha256": _sha256(evaluation_config)},
            "base_actor": {"path": str(actor.relative_to(root)), "sha256": _sha256(actor)},
            "catalog": {"path": str(catalog.relative_to(root)), "sha256": _sha256(catalog)},
            "reward_models": [
                {"path": str(path.relative_to(root)), "sha256": _sha256(path)} for path in model_paths
            ],
            "budget_plan": {"path": str(budget_path.relative_to(root)), "sha256": _sha256(budget_path)},
            "resource_plan": {"path": str(resource_plan_path.relative_to(root)), "sha256": _sha256(resource_plan_path)},
            "reward_scale": {"path": str(scale_path.relative_to(root)), "sha256": _sha256(scale_path)},
            "policy_training_panel": {"path": str(policy_train_panel_path.relative_to(root)), "sha256": _sha256(policy_train_panel_path)},
            "panel_tools_implementation_sources": [
                {"path": str(path.relative_to(root)), "sha256": _sha256(path)} for path in fixed_code
            ],
        },
        "fixed_comparisons": {
            "analytic_beta_grid": [3.0, 1.0, 0.3, 0.1],
            "primary_support_k": 500,
            "support_sensitivity": {"k": 1000, "contexts": 128},
            "full_catalog_analytic_orientation_contexts": 128,
            "head_width": 32,
            "trainable_beta_values": [1.0, 0.3],
            "head_methods": ["sampled_loo", "exact_expected_reward", "analytic_q_distillation"],
            "no_feedback_control": "same-width residual head with exact objective and past-feedback features removed",
            "seed_schedule": [17, 42],
            "repeat_control": "history-frequency reranker; ties by base actor log probability then raw item ID",
            "reward": "ratio only; shared frozen_reward_scale_v1 normalization for sampled and exact paths",
        },
        "selection_constraints": {
            "no_policy_parameter_tuning_on_rm_audit": True,
            "no_parameter_changes_after_frozen_final": True,
            "no_independent_test_claim": True,
            "report_algorithm_features_support_and_compute_separately": True,
        },
        "selection_rule": {
            "primary_comparison": "each fixed policy versus the same base actor on the same one-event-per-user panel, frozen RM, reward scale, support, and seed schedule",
            "relative_relevance_guard": "one-sided 95% user-cluster bootstrap lower bound for NDCG@10 new - 0.98*base must be > 0",
            "if_no_policy_passes": "report no qualifying candidate and preserve the trade-off; do not relax tolerance or replace metrics",
            "if_multiple_policies_pass": "choose highest primary user-macro NDCG@10 only among guard-passing fixed candidates; report all candidates and contrasts",
            "multiple_contrasts": "apply Holm adjustment to registered confirmatory contrasts; all other results exploratory",
            "seed_aggregation": "average available seeds within user, then report seed spread separately; one seed is a pilot only",
            "post_final_rule": "after the frozen retrospective panel is evaluated, do not change policy parameters; additional work is separately labeled exploratory",
        },
        "reward_scale_rule": {
            "version": "frozen_reward_scale_v1",
            "source": "hash-selected F_RM outcomes from the RM-fit user partition, up to 5 events per user, seed 17",
            "statistics": "population mean and standard deviation (ddof=0), frozen once before policy scores",
            "shared_by": ["sampled_loo", "exact_expected_reward", "analytic_q_distillation"],
            "standard_deviation_floor": 0.001,
        },
        "known_exposure": [
            "The dataset revision and validation/test periods were previously inspected in V1-full_model.",
            "The RM action-added-value diagnostic was already reviewed before this policy freeze; policy parameters remain fixed to the written plan and subsequent policy comparisons will be labeled exploratory on exposed outcomes.",
        ],
        "limitations": [
            "Timestamps are absent from the persisted sequence cache; strict temporal availability is represented by cached row order.",
            "No same-revision artist mapping or LightGBM installation is available.",
            "Only one actor seed and one RM seed are fixed in the reduced core program; policy seeds 17/42 are attempted only if the measured phase budget allows.",
        ],
    }
    _write_new_inside_workspace(root, selection_path, json.dumps(selection, indent=2, ensure_ascii=False) + "\n")
    return {"selection_path": str(selection_path), "panel_path": str(panel_path), "manifest_path": str(manifest_path),
            "n_users": len(selected), "event_ids_sha256": panel_info["event_ids_sha256"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Freeze panel_tools's one-event-per-user evaluation panel and policy matrix")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--source-panel", default="artifacts/panel_tools/evaluation/full_model_sft_weak_seed17_validation_panel.jsonl")
    parser.add_argument("--policy-config", default="configs/panel_tools_policy_matrix.yaml")
    parser.add_argument("--actor", default="models/actor/weak.pt")
    parser.add_argument("--catalog", default="data/processed_full_model_tminus14/catalog.parquet")
    parser.add_argument("--reward-model", action="append", dest="reward_models", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    result = freeze_panel_tools_selection(
        args.workspace,
        source_panel_path=args.source_panel,
        policy_config_path=args.policy_config,
        actor_checkpoint_path=args.actor,
        catalog_path=args.catalog,
        reward_model_paths=args.reward_models,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
