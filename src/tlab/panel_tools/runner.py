from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import torch

from tlab.data.store import SequenceStore
from tlab.panel_tools.data import build_evaluation_panel, panel_manifest
from tlab.panel_tools.evaluator import evaluate_actor_panel, load_actor_checkpoint


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite panel_tools artifact: {path}")
    path.write_text(text, encoding="utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_saved_actor_evaluation(
    workspace: str | Path,
    checkpoint_path: str | Path,
    *,
    run_name: str,
    split: str = "validation",
    seed: int = 17,
    targets_per_user: int = 3,
    batch_size: int = 16,
    device: str | None = None,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    protocol = json.loads((root / "artifacts/panel_tools/protocol.json").read_text(encoding="utf-8"))
    processed = root / protocol["dataset"].get("processed_full_model_tminus14_dir", "data/processed_full_model_tminus14")
    source_sha = protocol["dataset"]["processed_full_model_tminus14_listens_sha256"]
    store = SequenceStore.load_cache(processed / "sequence_index", expected_source_sha256=source_sha)
    catalog_path = processed / "catalog.parquet"
    catalog = pl.read_parquet(catalog_path).sort("token_id")
    catalog_item_ids = catalog.get_column("item_id").to_list()
    if catalog.height != 20_000:
        raise ValueError(f"full_model bridge catalog has {catalog.height} items, expected 20,000")
    panel = build_evaluation_panel(
        store, split=split, targets_per_user=targets_per_user, seed=seed, source_sha256=source_sha,
    )
    output_dir = root / "artifacts/panel_tools/evaluation"
    panel_path = output_dir / f"{run_name}_{split}_panel.jsonl"
    result_path = output_dir / f"{run_name}_{split}.json"
    panel_records = "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in panel)
    _write_new(panel_path, panel_records)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    actor, actor_provenance = load_actor_checkpoint(
        checkpoint_path,
        vocab_size=len(catalog_item_ids) + 3,
        device=selected_device,
    )
    started = time.perf_counter()
    metrics = evaluate_actor_panel(actor, store, catalog_item_ids, panel, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "protocol_id": protocol["protocol_id"],
        "status": "completed",
        "evaluation_label": "retrospective reanalysis",
        "model": run_name,
        "checkpoint": actor_provenance,
        "catalog": {
            "catalog_size": catalog.height,
            "file_sha256": _hash_file(catalog_path),
            "item_id_order": "token_id ascending",
        },
        "panel": panel_manifest(panel, split=split, seed=seed, source_sha256=source_sha),
        "runtime": {"device": selected_device, "evaluation_seconds": elapsed, "batch_size": batch_size},
        **metrics,
        "limitations": [
            "The target period has prior V1-full_model exposure and is not an independent test.",
            "The local sequence cache preserves strict past by row position and collapses timestamp ties to OOV; timestamps are not present in this cache for re-auditing at scoring time.",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_new(result_path, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return {"panel_path": str(panel_path), "result_path": str(result_path), "result": result}
