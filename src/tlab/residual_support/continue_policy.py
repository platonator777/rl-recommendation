from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tlab.residual_support.features import POLICY_FEATURE_NAMES
from tlab.residual_support.train_policy import _load_data, _run_one


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def continue_policy(config_path: str | Path, root: str | Path, source_run: str,
                    *, diagnostic_name: str) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = Path(root).resolve()
    data = _load_data(config_path, project_root, resume=True)
    cfg = data["config"]
    params = cfg["train"]
    if len(params["objectives"]) != 1 or len(params["beta_values"]) != 1 or len(params["seed_values"]) != 1:
        raise ValueError("continuation config must register exactly one objective, beta and seed")
    objective = str(params["objectives"][0])
    beta = float(params["beta_values"][0])
    seed = int(params["seed_values"][0])
    source_dir = _resolve(project_root, source_run)
    source_manifest_path = source_dir / "run_manifest.json"
    source_state_path = source_dir / "last.pt"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (source_manifest["objective"], float(source_manifest["beta"]), int(source_manifest["seed"])) != (objective, beta, seed):
        raise ValueError("continuation source does not match the registered objective/beta/seed")
    if int(source_manifest["updates_completed"]) != int(source_manifest["updates_limit"]):
        raise ValueError("continuation source must have reached its configured update limit")
    if int(params["max_updates"]) <= int(source_manifest["updates_completed"]):
        raise ValueError("diagnostic max_updates must exceed the completed source fit")
    source_state = torch.load(source_state_path, map_location="cpu", weights_only=False)
    if int(source_state["updates_done"]) != int(source_manifest["updates_completed"]):
        raise ValueError("source state update count disagrees with the source manifest")

    target_dir = data["run_root"] / "runs" / diagnostic_name
    columns = tuple(range(len(POLICY_FEATURE_NAMES)))
    signature = {
        "run_name": diagnostic_name, "objective": objective, "beta": beta, "seed": seed,
        "variant": "full_continuation", "feature_columns": list(columns),
        "cache_identity_sha256": data["train_meta"]["bound_identity_sha256"],
        "scaler_sha256": data["scaler"].digest(), "updates_limit": int(params["max_updates"]),
    }
    source_signature = source_state["state_signature"]
    for key in ("objective", "beta", "seed", "feature_columns", "cache_identity_sha256", "scaler_sha256"):
        if source_signature[key] != signature[key]:
            raise ValueError(f"continuation source signature mismatch in {key}")
    provenance = {
        "diagnostic_name": diagnostic_name,
        "source_run": str(source_dir.relative_to(project_root)),
        "source_run_manifest_sha256": _sha256(source_manifest_path),
        "source_last_checkpoint_sha256": _sha256(source_state_path),
        "source_updates_completed": int(source_manifest["updates_completed"]),
        "registered_max_updates": int(params["max_updates"]),
        "selection_role": "additional pre-registered candidate using the same train/development users, support, scaler and reward scale",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if target_dir.exists():
        if (target_dir / "run_manifest.json").exists():
            raise FileExistsError(f"diagnostic continuation is already complete: {target_dir}")
        parent_path = target_dir / "continuation_parent.json"
        state_path = target_dir / "last.pt"
        curve_path = target_dir / "curve.jsonl"
        if not parent_path.is_file() or not state_path.is_file() or not curve_path.is_file():
            raise FileExistsError(f"unrecognized partial diagnostic output: {target_dir}")
        saved_parent = json.loads(parent_path.read_text(encoding="utf-8"))
        saved_state = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved_parent.get("source_last_checkpoint_sha256") != provenance["source_last_checkpoint_sha256"]:
            raise ValueError("partial continuation belongs to a different source checkpoint")
        if saved_state.get("state_signature") != signature:
            raise ValueError("partial continuation signature differs from this config")
    else:
        target_dir.mkdir(parents=True)
        source_state["state_signature"] = signature
        torch.save(source_state, target_dir / "last.pt")
        (target_dir / "curve.jsonl").write_text("", encoding="utf-8")
        (target_dir / "run_state.json").write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
        (target_dir / "continuation_parent.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
    result = _run_one(
        run_dir=target_dir, run_name=diagnostic_name, objective=objective, beta=beta, seed=seed,
        variant="full_continuation", columns=columns, config=cfg,
        train_meta=data["train_meta"], train_arrays=data["train_arrays"],
        dev_meta=data["dev_meta"], dev_arrays=data["dev_arrays"], dev_panel=data["dev_panel"],
        scaler=data["scaler"], scale=data["scale"], baseline_dev=data["baseline_dev"],
        item_ids=data["item_ids"], rm_fit_counts=data["rm_fit_counts"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), resume=True,
    )
    return {**result, "continuation_parent": provenance}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Continue a registered panel_tools.1 policy fit as a separate diagnostic candidate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--diagnostic-name", required=True)
    args = parser.parse_args(argv)
    result = continue_policy(args.config, args.workspace, args.source_run,
                             diagnostic_name=args.diagnostic_name)
    print(json.dumps({key: result.get(key) for key in (
        "run_name", "updates_completed", "stop_reason", "best_checkpoint_step",
        "best_dev_support_objective", "best_checkpoint_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
