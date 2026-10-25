"""Hash the complete panel_tools.2 implementation and deliverables, excluding mutable ledgers."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import ROOT, ARTIFACTS, record_operation


OUTPUT = ARTIFACTS / "protocol" / "implementation_manifest.json"
ROOTS = (
    ROOT / "src" / "tlab" / "policy_diagnostics",
    ROOT / "tests" / "policy_diagnostics",
    ROOT / "configs" / "policy_diagnostics",
    ROOT / "artifacts" / "policy_diagnostics",
    ROOT / "final" / "policy_diagnostics",
)
EXCLUDED = {
    OUTPUT.resolve(),
    (ARTIFACTS / "protocol" / "budget.json").resolve(),
    (ARTIFACTS / "protocol" / "stage_status.json").resolve(),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build() -> dict[str, Any]:
    started = time.perf_counter()
    files: list[Path] = []
    for root in ROOTS:
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file() and not any(part in {"__pycache__", ".pytest_cache"} for part in p.parts))
    files = sorted((p for p in files if p.resolve() not in EXCLUDED), key=lambda p: p.relative_to(ROOT).as_posix().casefold())
    entries = [{"path": p.relative_to(ROOT).as_posix(), "size_bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files]
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All panel_tools.2 source, tests, configs, artifacts, reports and figures.",
        "mutable_exclusions": [
            "artifacts/policy_diagnostics/protocol/implementation_manifest.json (self-reference)",
            "artifacts/policy_diagnostics/protocol/budget.json (updated by recording this pass)",
            "artifacts/policy_diagnostics/protocol/stage_status.json (updated at closeout)",
        ],
        "count": len(entries),
        "total_bytes": sum(e["size_bytes"] for e in entries),
        "hashing_seconds": round(time.perf_counter() - started, 3),
        "files": entries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record_operation(
        "policy_diagnostics_implementation_manifest_hashing",
        result["hashing_seconds"],
        bucket="final_evaluation_and_report",
        category="integrity_and_reproducibility",
        evidence=["artifacts/policy_diagnostics/protocol/implementation_manifest.json"],
        note="The manifest excludes itself and the two mutable closeout ledgers; it covers every other panel_tools.2 file.",
    )
    return result


if __name__ == "__main__":
    result = build()
    print(json.dumps({k: result[k] for k in ("count", "total_bytes", "hashing_seconds")}, indent=2))
