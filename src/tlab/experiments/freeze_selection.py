from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tlab.data.manifest import sha256_file, write_json_atomic
from tlab.train.checkpoint import source_tree_sha256


def _project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validation_only_evidence(path: Path) -> bool:
    """Require a machine-readable validation artifact as selection evidence."""
    try:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            values = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            values = [json.loads(raw)]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"validation evidence must be JSON or JSONL: {path}") from exc

    found_validation = False
    stack: list[Any] = list(values)
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if str(value.get("split", "")).lower() == "test":
                raise ValueError(f"test-split data cannot be used for model selection: {path}")
            if str(value.get("split", "")).lower() == "validation":
                found_validation = True
            if any(key in value for key in ("validation", "validation_metrics", "validation_summary")):
                found_validation = True
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return found_validation


def freeze_selection(
    config_path: str | Path = "configs/final.yaml",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze an explicit validation-based model selection before test evaluation.

    The config records the selected family, the rule and validation evidence. This
    command verifies and hashes those inputs; it never reads test labels or chooses
    a model automatically.
    """
    from tlab.config import load_config

    config = load_config(config_path)
    root = config.path.parent.parent
    final = config.raw.get("final")
    if not isinstance(final, dict):
        raise ValueError("final config needs a top-level 'final' mapping")
    if final.get("selection_split") != "validation":
        raise ValueError("model selection must be frozen from validation")

    family = str(final.get("selected_family", "")).strip()
    rule = str(final.get("selection_rule", "")).strip()
    rationale = str(final.get("selection_rationale", "")).strip()
    if not family or not rule or not rationale:
        raise ValueError("selected_family, selection_rule, and selection_rationale are required")

    raw_models = final.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("final.models must list the frozen model checkpoints")
    names: set[str] = set()
    frozen_models: list[dict[str, Any]] = []
    for raw in raw_models:
        if not isinstance(raw, dict):
            raise ValueError("each final.models entry must be a mapping")
        name = str(raw.get("name", "")).strip()
        model_family = str(raw.get("family", "")).strip()
        if not name or name in names:
            raise ValueError(f"model name is missing or duplicated: {name!r}")
        names.add(name)
        if not model_family:
            raise ValueError(f"model {name!r} is missing family")
        seed = raw.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError(f"model {name!r} seed must be an integer or null for a non-trained baseline")

        checkpoint_value = raw.get("checkpoint")
        run_manifest_value = raw.get("run_manifest")
        checkpoint_path = _project_path(root, checkpoint_value) if checkpoint_value else None
        if checkpoint_path is not None:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"checkpoint for {name!r} does not exist: {checkpoint_path}")
            run_manifest_path = (
                _project_path(root, run_manifest_value)
                if run_manifest_value
                else checkpoint_path.parent / "run_manifest.json"
            )
            if not run_manifest_path.is_file():
                raise FileNotFoundError(f"run manifest for {name!r} does not exist: {run_manifest_path}")
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if not run_manifest.get("finished_at_utc"):
                raise ValueError(f"run for {name!r} is not complete")
            if seed is not None and int(run_manifest.get("seed", -1)) != seed:
                raise ValueError(f"seed mismatch for {name!r}: config={seed}, run={run_manifest.get('seed')}")
            checkpoint_record = {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "run_manifest_path": str(run_manifest_path),
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "run_id": run_manifest.get("run_id"),
                "run_stage": run_manifest.get("stage"),
                "run_seed": run_manifest.get("seed"),
                "source_sha256": run_manifest.get("source_sha256"),
            }
        else:
            if seed is not None:
                raise ValueError(f"trained model {name!r} must provide a checkpoint")
            checkpoint_record = None

        evidence = raw.get("validation_evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"model {name!r} needs at least one validation_evidence file")
        evidence_records = []
        for item in evidence:
            evidence_path = _project_path(root, item)
            if not evidence_path.is_file():
                raise FileNotFoundError(f"validation evidence for {name!r} does not exist: {evidence_path}")
            if not _validation_only_evidence(evidence_path):
                raise ValueError(f"file does not contain validation evidence for {name!r}: {evidence_path}")
            evidence_records.append({"path": str(evidence_path), "sha256": sha256_file(evidence_path)})

        frozen_models.append({
            "name": name,
            "family": model_family,
            "seed": seed,
            "role": str(raw.get("role", "candidate")),
            "reference_name": raw.get("reference_name"),
            "checkpoint": checkpoint_record,
            "validation_evidence": evidence_records,
            "validation_summary": raw.get("validation_summary", {}),
            "reward_panel": None,
        })
        raw_panel = raw.get("reward_panel")
        if raw_panel is not None:
            if not isinstance(raw_panel, dict):
                raise ValueError(f"reward_panel for {name!r} must be a mapping")
            panel_files: dict[str, dict[str, str]] = {}
            for key in ("sft_checkpoint", "rm_fit_checkpoint", "rm_audit_checkpoint", "reward_scale_path"):
                value = raw_panel.get(key)
                if not value:
                    raise ValueError(f"reward_panel for {name!r} is missing {key}")
                panel_path = _project_path(root, value)
                if not panel_path.is_file():
                    raise FileNotFoundError(f"reward panel input for {name!r} does not exist: {panel_path}")
                panel_files[key] = {"path": str(panel_path), "sha256": sha256_file(panel_path)}
            frozen_models[-1]["reward_panel"] = {
                "files": panel_files,
                "beta": float(raw_panel.get("beta", 0.1)),
                "reward_head": str(raw_panel.get("reward_head", "ratio")),
                "repeat_penalty": float(raw_panel.get("repeat_penalty", 0.0)),
            }

    if family not in {str(model["family"]) for model in frozen_models}:
        raise ValueError("selected_family must be represented in final.models")

    protocol_path = _project_path(root, config.data.get("processed_dir", "data/processed")) / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"processed data protocol does not exist: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    model_config_path = _project_path(root, final.get("model_config", "configs/base.yaml"))
    if not model_config_path.is_file():
        raise FileNotFoundError(f"model config does not exist: {model_config_path}")
    test_protocol = final.get("test_protocol", {})
    if not isinstance(test_protocol, dict):
        raise ValueError("final.test_protocol must be a mapping")
    test_status = str(final.get("test_status", "not_run_at_freeze"))
    if test_status not in {"not_run_at_freeze", "post_test_research_iteration"}:
        raise ValueError("final.test_status must be not_run_at_freeze or post_test_research_iteration")
    test_protocol = {
        "anchors_per_user": int(test_protocol.get("anchors_per_user", 10)),
        "anchor_seed": int(test_protocol.get("anchor_seed", 17)),
        "batch_size": int(test_protocol.get("batch_size", 16)),
        "top_k": int(test_protocol.get("top_k", 10)),
        "bootstrap_seed": int(test_protocol.get("bootstrap_seed", 17)),
    }
    if min(test_protocol["anchors_per_user"], test_protocol["batch_size"], test_protocol["top_k"]) < 1:
        raise ValueError("final test protocol values must be positive")
    output_value = output_path or final.get("output", "artifacts/final_selection.json")
    target = _project_path(root, output_value)
    payload = {
        "selection_id": str(final.get("selection_id", "final")),
        "selection_split": "validation",
        "selected_family": family,
        "selection_rule": rule,
        "selection_rationale": rationale,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256_at_freeze": source_tree_sha256(root / "src" / "tlab"),
        "config_path": str(config.path),
        "config_sha256": sha256_file(config.path),
        "model_config_path": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "data_protocol_path": str(protocol_path),
        "processed_listens_sha256": protocol.get("processed_listens_sha256"),
        "catalog_sha256": protocol.get("catalog_sha256"),
        "test_status": test_status,
        "test_evaluation": {
            "split": "test",
            "status": "not_run_at_freeze",
            "historical_access": (
                "The existing Test split has been opened before this reference freeze; this is a post-test research iteration."
                if test_status == "post_test_research_iteration" else None
            ),
        },
        "test_protocol": test_protocol,
        "models": frozen_models,
    }
    write_json_atomic(target, payload)
    payload["output_path"] = str(target)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze an explicit validation-based final model selection")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = freeze_selection(args.config, output_path=args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
