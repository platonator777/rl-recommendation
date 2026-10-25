from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from tlab.config import load_config
from tlab.data.manifest import sha256_file, write_json_atomic


def download_dataset(config_path: str | Path, selected: list[str] | None = None) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies before downloading Yambda") from exc

    config = load_config(config_path)
    dataset_cfg = config.dataset
    data_cfg = config.data
    repository = dataset_cfg.get("repo_id", "yandex/yambda")
    revision = dataset_cfg.get("revision", "main")
    files = dataset_cfg.get("files", {})
    chosen = selected or list(files)
    unknown = sorted(set(chosen) - set(files))
    if unknown:
        raise ValueError(f"Unknown dataset file keys: {', '.join(unknown)}")

    source_dir = Path(data_cfg.get("source_dir", "data/raw"))
    if not source_dir.is_absolute():
        source_dir = config.path.parent.parent / source_dir
    source_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.dataset_info(repository, revision=revision)
    resolved_revision = info.sha
    manifest: dict[str, Any] = {
        "repository": repository,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "files": {},
    }
    for key in chosen:
        relative_name = files[key]
        cached = Path(hf_hub_download(
            repo_id=repository,
            repo_type="dataset",
            filename=relative_name,
            revision=resolved_revision,
        ))
        destination = source_dir / Path(relative_name).name
        if cached.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, destination)
        metadata = pq.read_metadata(destination)
        schema = pq.read_schema(destination)
        manifest["files"][key] = {
            "repository_path": relative_name,
            "local_path": str(destination.resolve()),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "num_rows": metadata.num_rows,
            "schema": {field.name: str(field.type) for field in schema},
        }

    output = source_dir.parent / "data_manifest.json"
    write_json_atomic(output, manifest)
    return {"manifest_path": str(output.resolve()), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download only configured Yambda Parquet files")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--files", nargs="*", help="Configured keys, for example listens likes unlikes")
    args = parser.parse_args()
    import json

    print(json.dumps(download_dataset(args.config, args.files), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
