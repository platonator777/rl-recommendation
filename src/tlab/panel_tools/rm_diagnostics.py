from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from tlab.data.store import SequenceStore
from tlab.panel_tools.data import stable_event_id
from tlab.panel_tools.features import CandidateFeatureBatch, candidate_history_features
from tlab.panel_tools.statistics import paired_user_bootstrap


def history_only_features(history_item_ids: np.ndarray, history_ratios: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Context-only covariates computed from the strict prefix before one target."""
    items = np.asarray(history_item_ids, dtype=np.int64)[-1000:]
    ratios = np.asarray(history_ratios, dtype=np.float64)[-1000:]
    if items.shape != ratios.shape:
        raise ValueError("history item and ratio arrays must align")
    features: list[float] = [float(np.log1p(len(items))), float(np.unique(items[items >= 0]).size)]
    names = ["log_history_length_1000", "unique_items_1000"]
    for width in (50, 200, 1000):
        window_items = items[-width:]
        window_ratios = ratios[-width:]
        valid = np.isfinite(window_ratios)
        valid_ratios = window_ratios[valid]
        features.extend((
            float(np.log1p(len(window_items))),
            float(np.log1p(np.unique(window_items[window_items >= 0]).size)),
            float(np.log1p(valid.sum())),
            float(valid_ratios.mean()) if valid_ratios.size else 0.0,
            float(valid_ratios.std()) if valid_ratios.size else 0.0,
            float(valid.sum() == 0),
        ))
        names.extend((f"log_history_length_{width}", f"unique_items_{width}", f"log_valid_feedback_count_{width}",
                      f"mean_past_ratio_{width}", f"std_past_ratio_{width}", f"past_ratio_missing_{width}"))
    if not np.all(np.isfinite(features)):
        raise AssertionError("history-only feature builder produced non-finite values")
    return np.asarray(features, dtype=np.float32), tuple(names)


def _asof_candidate_features(user, position: int, item_id: int) -> tuple[CandidateFeatureBatch, np.ndarray, int, bool]:
    history_start = max(0, position - 1000)
    history_ids = np.asarray(user.raw_item_ids[history_start:position], dtype=np.int64)
    history_ratios = np.asarray(user.ratio_raw[history_start:position], dtype=np.float64)
    history_durations = np.asarray(user.duration_seconds[history_start:position], dtype=np.float64)
    support = int(np.count_nonzero(history_ids == item_id))
    duration_values = history_durations[(history_ids == item_id) & np.isfinite(history_durations) & (history_durations >= 0)]
    duration = float(duration_values.mean()) if duration_values.size else np.nan
    full_prior = np.asarray(user.raw_item_ids[:position], dtype=np.int64)
    is_new = not bool(np.any(full_prior == item_id))
    features = candidate_history_features(
        history_item_ids=history_ids,
        history_ratios=history_ratios,
        candidate_item_ids=np.asarray([item_id], dtype=np.int64),
        duration_seconds=np.asarray([duration], dtype=np.float64),
        # Candidate popularity here is strictly the target user's observed prefix,
        # not a full-train aggregate that could contain this target or future events.
        train_popularity=np.asarray([support], dtype=np.float64),
        train_ratio_mean=np.asarray([np.nan], dtype=np.float64),
        windows=(200, 1000),
        early_train_ratio_mean=0.5,
    )
    history, _ = history_only_features(history_ids, history_ratios)
    return features, history, support, is_new


def _sample_positions(positions, uid: int, max_events: int, seed: int, source_hash: str, raw_item_ids) -> list[int]:
    ranked = []
    for raw_position in positions:
        position = int(raw_position)
        raw_item = int(raw_item_ids[position])
        if raw_item < 0:
            continue
        event_id = stable_event_id(source_hash, uid, position, raw_item)
        key = hashlib.sha256(f"{seed}:{event_id}".encode("ascii")).digest()
        ranked.append((key, event_id, position))
    ranked.sort()
    return [position for _, _, position in ranked[:max_events]]


def build_rm_action_datasets(
    store: SequenceStore,
    *,
    source_hash: str,
    max_events_per_user: int = 5,
    seed: int = 17,
) -> dict[str, dict[str, Any]]:
    if max_events_per_user < 1:
        raise ValueError("max_events_per_user must be positive")
    rows: dict[str, list[dict[str, Any]]] = {"fit": [], "audit": []}
    selected_counts = {"fit": 0, "audit": 0}
    for user in store.users:
        partition = user.rm_user_partition
        if partition not in {"fit", "audit"}:
            continue
        if partition == "fit":
            positions = user.rm_fit_positions
        else:
            positions = user.rm_validation_positions
        if len(positions) == 0:
            continue
        selected = _sample_positions(positions, user.uid, max_events_per_user, seed, source_hash, user.raw_item_ids)
        selected_counts[partition] += len(selected)
        for position in selected:
            item_id = int(user.raw_item_ids[position])
            target = float(user.ratio_raw[position])
            if not np.isfinite(target):
                continue
            candidate, history, support, is_new = _asof_candidate_features(user, position, item_id)
            rows[partition].append({
                "uid": int(user.uid),
                "position": position,
                "event_id": stable_event_id(source_hash, user.uid, position, item_id),
                "target_ratio": target,
                "positive": bool(target >= 0.5),
                "support_1000": support,
                "support_group": "0" if support == 0 else "1-9" if support < 10 else "10-99" if support < 100 else "100+",
                "is_new": is_new,
                "history_features": history,
                "candidate_features": candidate.values[0],
                "candidate_feature_names": candidate.names,
            })
    if not rows["fit"] or not rows["audit"]:
        raise ValueError("panel_tools RM diagnostic requires fit and audit user partitions with finite labels")

    def pack(partition: str) -> dict[str, Any]:
        examples = rows[partition]
        names = tuple(examples[0]["candidate_feature_names"])
        if any(tuple(row["candidate_feature_names"]) != names for row in examples):
            raise AssertionError("candidate feature schema changed within a dataset partition")
        return {
            "uid": np.asarray([row["uid"] for row in examples], dtype=np.int64),
            "event_id": [row["event_id"] for row in examples],
            "y": np.asarray([row["target_ratio"] for row in examples], dtype=np.float32),
            "positive": np.asarray([row["positive"] for row in examples], dtype=bool),
            "support": np.asarray([row["support_1000"] for row in examples], dtype=np.int64),
            "support_group": np.asarray([row["support_group"] for row in examples], dtype=object),
            "is_new": np.asarray([row["is_new"] for row in examples], dtype=bool),
            "history_x": np.stack([row["history_features"] for row in examples]),
            "candidate_x": np.stack([row["candidate_features"] for row in examples]),
            "candidate_feature_names": names,
            "rows_selected_before_label_read": selected_counts[partition],
        }

    return {partition: pack(partition) for partition in rows}


def _model_metrics(y: np.ndarray, prediction: np.ndarray, uid: np.ndarray) -> dict[str, Any]:
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(y, dtype=np.float64)
    per_user: dict[int, list[float]] = {}
    for user_id, actual, predicted in zip(uid.tolist(), y.tolist(), prediction.tolist()):
        per_user.setdefault(int(user_id), []).append(abs(actual - predicted))
    user_mae = {key: float(np.mean(values)) for key, values in per_user.items()}
    order = np.argsort(prediction, kind="stable")
    bins = np.array_split(order, min(10, max(1, len(order))))
    calibration = [{
        "n": int(len(indices)),
        "mean_prediction": float(prediction[indices].mean()),
        "mean_observed": float(y[indices].mean()),
    } for indices in bins if len(indices)]
    return {
        "n_events": int(len(y)),
        "n_users": int(len(user_mae)),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "mean_prediction": float(prediction.mean()),
        "mean_observed": float(y.mean()),
        "per_user_mae": {str(key): value for key, value in user_mae.items()},
        "calibration_bins_by_prediction_decile": calibration,
    }


def _positive_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, Any] | None:
    positive = np.asarray(y, dtype=bool)
    if positive.all() or not positive.any():
        return None
    score = np.asarray(score, dtype=np.float64).clip(1e-7, 1 - 1e-7)
    return {
        "positive_rate": float(positive.mean()),
        "roc_auc": float(roc_auc_score(positive, score)),
        "pr_auc_average_precision": float(average_precision_score(positive, score)),
        "log_loss": float(log_loss(positive, score)),
        "brier": float(np.mean((score - positive.astype(np.float64)) ** 2)),
    }


def _fit_regressor(train_x: np.ndarray, train_y: np.ndarray, seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=100, max_leaf_nodes=15, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=False, random_state=seed,
    )
    model.fit(train_x, train_y)
    return model


def run_rm_action_diagnostic(
    workspace: str | Path,
    *,
    output_path: str | Path | None = None,
    max_events_per_user: int = 5,
    seed: int = 17,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    protocol = json.loads((root / "artifacts/panel_tools/protocol.json").read_text(encoding="utf-8"))
    selected_output = Path(output_path) if output_path is not None else root / "artifacts/panel_tools/rm/action_added_value.json"
    if not selected_output.is_absolute():
        selected_output = root / selected_output
    model_dir = selected_output.parent / "models" / f"rm_action_seed{seed}"
    if selected_output.exists() or model_dir.exists():
        raise FileExistsError("refusing to overwrite existing panel_tools external evaluator results or models")
    processed = root / "data/processed_full_model_tminus14"
    source_hash = protocol["dataset"]["processed_full_model_tminus14_listens_sha256"]
    started = time.perf_counter()
    store = SequenceStore.load_cache(processed / "sequence_index", expected_source_sha256=source_hash)
    data = build_rm_action_datasets(store, source_hash=source_hash, max_events_per_user=max_events_per_user, seed=seed)
    elapsed_data = time.perf_counter() - started
    fit, audit = data["fit"], data["audit"]
    candidate_names = fit["candidate_feature_names"]
    metadata_columns = [i for i, name in enumerate(candidate_names) if name in {
        "log_duration", "duration_missing", "log_train_popularity",
    }]
    if not metadata_columns:
        raise RuntimeError("candidate metadata feature set unexpectedly empty")
    feature_sets = {
        "history_only": (fit["history_x"], audit["history_x"]),
        "strict_history_support_duration": (fit["candidate_x"][:, metadata_columns], audit["candidate_x"][:, metadata_columns]),
        "history_plus_candidate_cross": (
            np.column_stack((fit["history_x"], fit["candidate_x"])),
            np.column_stack((audit["history_x"], audit["candidate_x"])),
        ),
    }
    models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    model_details: dict[str, Any] = {}
    fit_started = time.perf_counter()
    for name, (train_x, audit_x) in feature_sets.items():
        if not np.all(np.isfinite(train_x)) or not np.all(np.isfinite(audit_x)):
            raise ValueError(f"non-finite feature values in model {name}")
        model = _fit_regressor(train_x, fit["y"], seed)
        prediction = np.clip(model.predict(audit_x), 0.0, 1.0)
        models[name] = model
        predictions[name] = prediction
        model_details[name] = {
            "model": "sklearn HistGradientBoostingRegressor",
            "n_features": int(train_x.shape[1]),
            "fit_rows": int(len(train_x)),
            "audit_rows": int(len(audit_x)),
            "fixed_hyperparameters": {
                "learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15,
                "min_samples_leaf": 50, "l2_regularization": 1.0,
            },
        }
    model_fit_seconds = time.perf_counter() - fit_started

    metrics = {
        name: _model_metrics(audit["y"], prediction, audit["uid"])
        for name, prediction in predictions.items()
    }
    metric_by_group: dict[str, dict[str, dict[str, Any]]] = {}
    full_prediction = predictions["history_plus_candidate_cross"]
    for group_name, group_values in (
        ("support_1000", audit["support_group"]),
        ("novelty", np.where(audit["is_new"], "new", "repeat")),
        ("positive", np.where(audit["positive"], "positive", "nonpositive")),
    ):
        metric_by_group[group_name] = {}
        for group in sorted(set(group_values.tolist())):
            mask = group_values == group
            if int(mask.sum()) < 10:
                continue
            metric_by_group[group_name][str(group)] = _model_metrics(
                audit["y"][mask], full_prediction[mask], audit["uid"][mask],
            )

    history_metrics = metrics["history_only"]
    full_metrics = metrics["history_plus_candidate_cross"]
    history_user_mae = {int(key): -float(value) for key, value in history_metrics["per_user_mae"].items()}
    full_user_mae = {int(key): -float(value) for key, value in full_metrics["per_user_mae"].items()}
    added_value = paired_user_bootstrap(
        full_user_mae, history_user_mae, n_bootstrap=n_bootstrap, seed=seed + 17000,
    )

    classifier_metrics = None
    fit_positive = fit["positive"]
    audit_positive = audit["positive"]
    if fit_positive.any() and (~fit_positive).any() and audit_positive.any() and (~audit_positive).any():
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=100, max_leaf_nodes=15, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        )
        classifier.fit(feature_sets["history_plus_candidate_cross"][0], fit_positive)
        positive_score = classifier.predict_proba(feature_sets["history_plus_candidate_cross"][1])[:, 1]
        classifier_metrics = _positive_metrics(audit_positive, positive_score)

    import joblib

    model_dir.mkdir(parents=True, exist_ok=False)
    saved_models: dict[str, dict[str, str]] = {}
    for name, model in models.items():
        model_path = model_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        saved_models[name] = {"path": str(model_path), "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    if classifier_metrics is not None:
        classifier_path = model_dir / "history_plus_candidate_listen_plus.joblib"
        joblib.dump(classifier, classifier_path)
        saved_models["listen_plus_classifier"] = {
            "path": str(classifier_path), "sha256": hashlib.sha256(classifier_path.read_bytes()).hexdigest(),
        }

    result = {
        "protocol_id": protocol["protocol_id"],
        "status": "completed",
        "evaluation_role": "external/audit evaluator trained on observed ratio labels; not gold reward or randomized action outcome",
        "evaluation_label": "retrospective reanalysis",
        "source": {
            "processed_listens_sha256": source_hash,
            "fit_population": "RM fit events from user partition=fit",
            "audit_population": "later RM validation events from user partition=audit",
            "sample_rule": "up to fixed hash-selected events per user before reading target ratio; no target-ratio threshold selection",
            "max_events_per_user": max_events_per_user,
            "seed": seed,
            "fit_events": len(fit["y"]),
            "fit_users": len(set(fit["uid"].tolist())),
            "audit_events": len(audit["y"]),
            "audit_users": len(set(audit["uid"].tolist())),
            "positive_rate_fit": float(fit["positive"].mean()),
            "positive_rate_audit": float(audit["positive"].mean()),
            "label_range_fit": [float(fit["y"].min()), float(fit["y"].max())],
            "label_range_audit": [float(audit["y"].min()), float(audit["y"].max())],
            "artist_mapping": "unavailable; no artist-derived features",
            "timestamp_features": "unavailable in the persisted sequence cache; event ordering remains strict by cached row position",
        },
        "models": model_details,
        "saved_model_artifacts": saved_models,
        "library_versions": {"scikit_learn": sklearn.__version__},
        "audit_metrics": metrics,
        "candidate_added_value_over_history_only": {
            "target": "paired user-level reduction in MAE (negative MAE score difference)",
            "bootstrap": added_value,
            "interpretation": "observed-action prediction added value only; not evidence of ranking quality or causal preference",
        },
        "listen_plus_classifier": {
            "label": "ratio >= 0.5",
            "metrics": classifier_metrics,
            "status": "fit" if classifier_metrics is not None else "not_fit_single_class_split",
        },
        "full_model_slices": metric_by_group,
        "runtime": {"data_and_feature_seconds": elapsed_data, "model_fit_seconds": model_fit_seconds,
                    "wall_seconds": time.perf_counter() - started},
        "limitations": [
            "HistGradientBoosting is a local substitute because LightGBM is not installed; no package was added to the environment.",
            "The audit user/time outcomes were inspected in V1-full_model and are not an independent test.",
            "Only logged target actions are labeled; predictions for unchosen tracks are counterfactual model outputs.",
            "Past feedback is available to this evaluator but not the item-only actor, so comparisons include an information advantage.",
            "Static artist metadata and event timestamps are absent from this cached schema.",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="panel_tools as-of candidate/history reward audit")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output")
    parser.add_argument("--max-events-per-user", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = run_rm_action_diagnostic(
        args.workspace, output_path=args.output, max_events_per_user=args.max_events_per_user, seed=args.seed,
    )
    print(json.dumps({
        "output": args.output or "artifacts/panel_tools/rm/action_added_value.json",
        "source": result["source"],
        "audit_metrics": {name: {key: value for key, value in metric.items() if key != "per_user_mae"}
                          for name, metric in result["audit_metrics"].items()},
        "candidate_added_value_over_history_only": result["candidate_added_value_over_history_only"],
        "runtime": result["runtime"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
