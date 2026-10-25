from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def hash_tree(root: str | Path) -> list[dict[str, object]]:
    """Read and hash files under a protected directory; never writes there."""
    directory = Path(root).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    rows: list[dict[str, object]] = []
    for path in directory.rglob("*"):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            rows.append({
                "path": path.relative_to(directory).as_posix(),
                "sha256": digest.hexdigest(),
                "bytes": path.stat().st_size,
            })
    rows.sort(key=lambda row: str(row["path"]))
    return rows


def verify_protected_tree(root: str | Path, before_manifest: str | Path) -> dict[str, Any]:
    before = json.loads(Path(before_manifest).read_text(encoding="utf-8"))
    current = hash_tree(root)
    before_by_path = {row["path"]: row for row in before}
    current_by_path = {row["path"]: row for row in current}
    added = sorted(set(current_by_path) - set(before_by_path))
    removed = sorted(set(before_by_path) - set(current_by_path))
    changed = sorted(
        path for path in set(current_by_path) & set(before_by_path)
        if current_by_path[path] != before_by_path[path]
    )
    return {
        "status": "unchanged" if not (added or removed or changed) else "MISMATCH",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "before_file_count": len(before),
        "current_file_count": len(current),
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def verify_and_record(workspace: str | Path, stage: str, manifest: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    manifest_path = Path(manifest) if manifest else workspace / "artifacts/panel_tools/protection/submission_before.json"
    result = verify_protected_tree(workspace / "submission", manifest_path)
    result["stage"] = stage
    output_dir = workspace / "artifacts/panel_tools/protection"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"submission_{stage}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if result["status"] != "unchanged":
        raise RuntimeError(f"submission protection check failed at {stage}: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the panel_tools protected submission tree without writing into it")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    print(json.dumps(verify_and_record(args.workspace, args.stage, args.manifest), indent=2))


if __name__ == "__main__":
    main()
