from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tlab.data.store import SequenceStore
from tlab.evaluation.metrics import single_target_metrics


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _mean_ndcg(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(single_target_metrics(rank)["ndcg@10"] for rank in ranks) / len(ranks)


def _duration_mean(item_ids: list[int], duration_by_item: dict[int, float | None]) -> float | None:
    values = [duration_by_item[item] for item in item_ids if item in duration_by_item and duration_by_item[item] is not None]
    return sum(float(value) for value in values) / len(values) if values else None


def build_case_studies(
    final_evaluation_path: str | Path,
    candidate_model: str,
    reference_model: str,
    policy_panel_path: str | Path | None = None,
    policy_validation_path: str | Path | None = None,
    policy_name: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Select five deterministic, anonymized ID-only examples from frozen outputs."""
    import polars as pl

    final_path = Path(final_evaluation_path).resolve()
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if final.get("split") != "test":
        raise ValueError("case studies require the frozen test evaluation output")
    if bool(policy_panel_path) != bool(policy_validation_path):
        raise ValueError("policy_panel and policy_validation must be supplied together")
    root = Path(final["selection_manifest_path"]).resolve().parent.parent
    result_path = _resolve(root, output_path or "artifacts/final_user_cases.json")
    if candidate_model not in final.get("models", {}) or reference_model not in final.get("models", {}):
        raise ValueError("candidate_model and reference_model must be present in final test evaluation")
    ranks = final["per_user_ranks"]
    recommendations = final.get("per_user_top_item_ids", {})
    targets = final.get("per_user_targets_item_ids", {})
    candidate_ranks = ranks[candidate_model]
    reference_ranks = ranks[reference_model]
    shared_users = sorted(
        candidate_ranks.keys() & reference_ranks.keys() & targets.keys(), key=int,
    )
    if not shared_users:
        raise ValueError("the final evaluation contains no shared user targets for case selection")

    selection = json.loads(Path(final["selection_manifest_path"]).read_text(encoding="utf-8"))
    root = Path(selection["config_path"]).resolve().parent.parent
    protocol_path = Path(selection["data_protocol_path"]).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    model_config_path = Path(selection["model_config_path"]).resolve()
    from tlab.config import load_config

    config = load_config(model_config_path)
    processed_dir = _resolve(root, config.data.get("processed_dir", "data/processed"))
    store = SequenceStore.load_cache(
        processed_dir / "sequence_index",
        expected_source_sha256=protocol["processed_listens_sha256"],
    )
    catalog = pl.read_parquet(processed_dir / "catalog.parquet").sort("token_id")
    duration_by_item = {
        int(item): (float(duration) if duration is not None else None)
        for item, duration in zip(
            catalog.get_column("item_id").to_list(),
            catalog.get_column("catalog_duration_seconds").to_list(),
        )
    }
    user_by_id = {str(user.uid): user for user in store.users}
    anchor_seed = int(final["test_protocol"]["anchor_seed"])
    anchors = store.evaluation_targets(
        "test", per_user_limit=int(final["test_protocol"]["anchors_per_user"]), seed=anchor_seed,
    )
    first_position_by_user: dict[str, int] = {}
    for user_index, position in anchors:
        first_position_by_user.setdefault(str(store.users[user_index].uid), position)

    candidate_recs = recommendations.get(candidate_model, {})
    reference_recs = recommendations.get(reference_model, {})

    baseline_user = shared_users[0]
    baseline_case = {
        "category": "baseline",
        "status": "observed",
        "user_id": baseline_user,
        "split": "test",
        "target_item_ids": targets[baseline_user],
        "target_ranks": {
            name: final["per_user_ranks"].get(name, {}).get(baseline_user)
            for name in (candidate_model, reference_model, "popularity", "sequential_bigram")
        },
        "candidate_top10_item_ids": candidate_recs.get(baseline_user, [[]])[0],
        "reference_top10_item_ids": reference_recs.get(baseline_user, [[]])[0],
        "selection": "lowest numeric UID with a shared frozen test target",
    }

    reward_case: dict[str, Any] = {
        "category": "reward_growth_without_accuracy_loss",
        "status": "not_observed",
        "criterion": "V_fit and V_audit increase over SFT and validation NDCG@10 does not decrease for the same user",
    }
    audit_case: dict[str, Any] = {
        "category": "audit_disagreement",
        "status": "not_observed",
        "criterion": "largest absolute difference between per-user V_fit and V_audit deltas",
    }
    if policy_panel_path and policy_validation_path:
        panel = json.loads(Path(policy_panel_path).resolve().read_text(encoding="utf-8"))
        policy_validation = json.loads(Path(policy_validation_path).resolve().read_text(encoding="utf-8"))
        policy_label = policy_name or str(panel.get("policy_label", ""))
        panel_users = panel.get("per_user_diagnostics", {})
        baseline_validation = policy_validation.get("baseline", {}).get("per_user_ranks", {})
        policy_validation_ranks = policy_validation.get("policy", {}).get("per_user_ranks", {})
        qualified = []
        disagreements = []
        for user_id in sorted(panel_users, key=int):
            detail = panel_users[user_id].get(policy_label)
            if not detail or user_id not in baseline_validation or user_id not in policy_validation_ranks:
                continue
            fit_delta = float(detail.get("v_fit_difference_vs_sft", 0.0))
            audit_delta = float(detail.get("v_audit_difference_vs_sft", 0.0))
            ndcg_delta = _mean_ndcg(policy_validation_ranks[user_id]) - _mean_ndcg(baseline_validation[user_id])
            if fit_delta > 0 and audit_delta > 0 and ndcg_delta >= 0:
                qualified.append((user_id, fit_delta + audit_delta, fit_delta, audit_delta, ndcg_delta))
            disagreements.append((user_id, abs(fit_delta - audit_delta), fit_delta, audit_delta, ndcg_delta))
        if qualified:
            user_id, _, fit_delta, audit_delta, ndcg_delta = sorted(qualified, key=lambda row: (-row[1], int(row[0])))[0]
            reward_case = {
                "category": "reward_growth_without_accuracy_loss",
                "status": "observed",
                "user_id": user_id,
                "split": "validation for NDCG; A_RL panel for model rewards",
                "policy": policy_label,
                "v_fit_delta_vs_sft": fit_delta,
                "v_audit_delta_vs_sft": audit_delta,
                "ndcg10_delta_vs_sft": ndcg_delta,
                "selection": "largest combined positive fit/audit reward increase, then lowest UID",
            }
        if disagreements:
            user_id, magnitude, fit_delta, audit_delta, ndcg_delta = sorted(
                disagreements, key=lambda row: (-row[1], int(row[0])),
            )[0]
            audit_case = {
                "category": "audit_disagreement",
                "status": "observed" if magnitude > 0 else "not_observed",
                "user_id": user_id,
                "split": "A_RL panel",
                "policy": policy_label,
                "v_fit_delta_vs_sft": fit_delta,
                "v_audit_delta_vs_sft": audit_delta,
                "validation_ndcg10_delta_vs_sft": ndcg_delta,
                "selection": "largest absolute difference between reward deltas, then lowest UID",
            }

    duration_candidates = []
    repeat_candidates = []
    for user_id in shared_users:
        user_candidate = candidate_recs.get(user_id, [[]])[0]
        user_reference = reference_recs.get(user_id, [[]])[0]
        candidate_duration = _duration_mean(user_candidate, duration_by_item)
        reference_duration = _duration_mean(user_reference, duration_by_item)
        if candidate_duration is not None and reference_duration is not None:
            duration_candidates.append((user_id, abs(candidate_duration - reference_duration), candidate_duration, reference_duration))
        user = user_by_id.get(user_id)
        position = first_position_by_user.get(user_id)
        if user is not None and position is not None:
            recent = {int(item) for item in user.raw_item_ids[max(0, position - 20):position] if int(item) >= 0}
            candidate_repeat_ids = [item for item in user_candidate if item in recent]
            reference_repeat_ids = [item for item in user_reference if item in recent]
            repeat_candidates.append((user_id, len(candidate_repeat_ids), len(reference_repeat_ids), candidate_repeat_ids, reference_repeat_ids,
                                      len(set(user_candidate)), len(set(user_reference))))

    duration_case: dict[str, Any] = {
        "category": "duration_shift",
        "status": "not_observed",
        "criterion": "largest absolute change in mean duration of first test-context top-10",
    }
    if duration_candidates:
        user_id, delta_abs, candidate_duration, reference_duration = sorted(
            duration_candidates, key=lambda row: (-row[1], int(row[0])),
        )[0]
        duration_case = {
            "category": "duration_shift",
            "status": "observed" if delta_abs > 0 else "not_observed",
            "user_id": user_id,
            "split": "test",
            "candidate_mean_top10_duration_seconds": candidate_duration,
            "reference_mean_top10_duration_seconds": reference_duration,
            "delta_seconds": candidate_duration - reference_duration,
            "candidate_top10_item_ids": candidate_recs.get(user_id, [[]])[0],
            "reference_top10_item_ids": reference_recs.get(user_id, [[]])[0],
            "selection": "largest absolute duration change, then lowest UID",
        }

    repetition_case: dict[str, Any] = {
        "category": "repetition_or_collapse",
        "status": "not_observed",
        "criterion": "highest candidate repeats from the prior 20 listens; deterministic UID tie-break",
    }
    if repeat_candidates:
        user_id, candidate_repeats, reference_repeats, candidate_repeat_ids, reference_repeat_ids, candidate_unique, reference_unique = sorted(
            repeat_candidates, key=lambda row: (-row[1], row[5], int(row[0])),
        )[0]
        repetition_case = {
            "category": "repetition_or_collapse",
            "status": "observed" if candidate_repeats > 0 or candidate_unique < 10 else "not_observed",
            "user_id": user_id,
            "split": "test",
            "candidate_repeat_count_top10": candidate_repeats,
            "reference_repeat_count_top10": reference_repeats,
            "candidate_repeated_item_ids": candidate_repeat_ids,
            "reference_repeated_item_ids": reference_repeat_ids,
            "candidate_unique_items_top10": candidate_unique,
            "reference_unique_items_top10": reference_unique,
            "selection": "most candidate repeats, then fewest unique candidate items, then lowest UID",
        }

    result = {
        "selection_manifest_path": final["selection_manifest_path"],
        "final_test_evaluation_path": str(final_path),
        "candidate_model": candidate_model,
        "reference_model": reference_model,
        "cases": [baseline_case, reward_case, duration_case, repetition_case, audit_case],
        "item_identity_note": "Item IDs are dataset IDs; no track names or user attributes are inferred.",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    result["output_path"] = str(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select deterministic per-user examples from frozen evaluation artifacts")
    parser.add_argument("--final-evaluation", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--policy-panel")
    parser.add_argument("--policy-validation")
    parser.add_argument("--policy-name")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = build_case_studies(
        args.final_evaluation, args.candidate_model, args.reference_model,
        policy_panel_path=args.policy_panel, policy_validation_path=args.policy_validation,
        policy_name=args.policy_name, output_path=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
