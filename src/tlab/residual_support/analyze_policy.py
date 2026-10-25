from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import torch

from tlab.residual_support.head import ResidualHeadSpec, ResidualPolicyHead
from tlab.residual_support.rm import sha256_file
from tlab.residual_support.train_policy import _exact_summary, _load_data, _resolve


def analyze_best_checkpoints(config_path: str | Path = "configs/residual_support/policy.yaml",
                             root: str | Path = ".") -> dict[str, Any]:
    config_path, project_root = Path(config_path).resolve(), Path(root).resolve()
    data = _load_data(config_path, project_root, resume=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_manifests = sorted((data["run_root"] / "runs").glob("*/run_manifest.json"))
    for value in data["config"].get("selection", {}).get("additional_run_manifests", []):
        path = _resolve(project_root, value)
        if path.is_file():
            run_manifests.append(path)
    if not run_manifests:
        raise FileNotFoundError("no completed policy run manifests were found")
    records: list[dict[str, Any]] = []
    for manifest_path in run_manifests:
        run = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_path = _resolve(project_root, run["best_checkpoint_path"])
        actual_sha = sha256_file(checkpoint_path)
        if actual_sha != run["best_checkpoint_sha256"]:
            raise ValueError(f"best checkpoint hash mismatch: {run['run_name']}")
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        columns = tuple(int(x) for x in saved["feature_columns"])
        head = ResidualPolicyHead(ResidualHeadSpec(input_dim=len(columns), hidden_dim=32)).to(device)
        head.load_state_dict(saved["state_dict"], strict=True)
        head.eval()
        train_stats = _exact_summary(head, data["train_arrays"], data["scaler"], columns,
                                     data["scale"], float(run["beta"]), device)
        dev_stats = _exact_summary(head, data["dev_arrays"], data["scaler"], columns,
                                   data["scale"], float(run["beta"]), device)
        records.append({
            "run_name": run["run_name"], "objective": run["objective"], "beta": run["beta"],
            "seed": run["seed"], "variant": run["variant"],
            "checkpoint_path": str(checkpoint_path.relative_to(project_root)),
            "checkpoint_sha256": actual_sha, "checkpoint_step": run["best_checkpoint_step"],
            "train_contexts": len(data["train_arrays"]["candidate_features"]),
            "development_contexts": len(data["dev_arrays"]["candidate_features"]),
            "train_exact": train_stats, "development_exact": dev_stats,
        })
    result = {
        "schema_version": 1, "protocol_id": data["config"]["protocol_id"],
        "status": "completed_best_checkpoint_kl_audit",
        "scope": "All main and registered diagnostic best checkpoints; fixed train and development contexts only.",
        "definitions": {
            "kl_policy_to_reference": "KL(q_policy || p_reference) conditional and weighted by unchanged support mass.",
            "kl_target_to_policy": "KL(q_analytic || q_policy), where q_analytic is the exact constrained reward-tilted policy at the run beta.",
            "analytic_objective_gap": "J_analytic - J_policy under the same fixed support, reward normalization, and contexts.",
            "final_panel_accessed": False,
        },
        "scaler_sha256": data["scaler"].digest(), "records": records,
    }
    out_path = data["run_root"] / "best_checkpoint_kl_audit.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tabular: list[dict[str, Any]] = []
    for record in records:
        row = {key: record[key] for key in ("run_name", "objective", "beta", "seed", "variant", "checkpoint_step")}
        for split in ("train", "development"):
            for metric, value in record[f"{split}_exact"].items():
                row[f"{split}_{metric}"] = value
        tabular.append(row)
    pl.DataFrame(tabular).write_csv(data["run_root"] / "best_checkpoint_kl_audit.csv")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit exact KL and analytic gaps at panel_tools.1 best checkpoints")
    parser.add_argument("--config", default="configs/residual_support/policy.yaml")
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args(argv)
    result = analyze_best_checkpoints(args.config, args.workspace)
    print(json.dumps({"status": result["status"], "checkpoint_count": len(result["records"]),
                      "final_panel_accessed": result["definitions"]["final_panel_accessed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
