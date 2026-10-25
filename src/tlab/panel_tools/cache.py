from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class panel_toolsCacheKey:
    protocol: str
    split: str
    event_ids_sha256: str
    vocabulary_sha256: str
    actor_sha256: str
    reward_model_sha256: str
    feature_schema: str
    normalization_version: str
    candidate_mode: str

    def digest(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def validate_cache_metadata(metadata: dict[str, Any], expected_key: panel_toolsCacheKey) -> None:
    if metadata.get("key") != asdict(expected_key) or metadata.get("key_sha256") != expected_key.digest():
        raise ValueError("cache key mismatch; refusing to reuse incompatible panel_tools cache")
    if metadata.get("schema_version") != "tlab_panel_tools_cache_v1":
        raise ValueError("unsupported cache schema")


class ShardedArrayCache:
    """Append-only float/ID memmap shards with a hash-bound manifest."""

    def __init__(self, root: str | Path, key: panel_toolsCacheKey, *, mode: str = "r") -> None:
        self.root = Path(root)
        self.key = key
        manifest = self.root / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"cache manifest is missing: {manifest}")
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        validate_cache_metadata(metadata, key)
        self.metadata = metadata
        self.mode = mode
        if mode not in {"r", "r+"}:
            raise ValueError("cache can only be opened read-only or read-write")

    @classmethod
    def create(cls, root: str | Path, key: panel_toolsCacheKey, shards: list[dict[str, Any]]) -> "ShardedArrayCache":
        root = Path(root)
        if root.exists():
            raise FileExistsError(f"refusing to overwrite existing cache: {root}")
        root.mkdir(parents=True)
        for shard_index, shard in enumerate(shards):
            name = str(shard.get("name", f"shard_{shard_index:05d}"))
            if Path(name).name != name or name in {"", ".", ".."}:
                raise ValueError("shard names must be simple filenames")
            shape = tuple(int(x) for x in shard["shape"])
            if not shape or any(x <= 0 for x in shape):
                raise ValueError("shard dimensions must be positive")
            dtype = np.dtype(shard.get("dtype", "float32"))
            if dtype.hasobject:
                raise ValueError("object arrays are not supported")
            data_path = root / f"{name}.bin"
            arr = np.memmap(data_path, dtype=dtype, mode="w+", shape=shape)
            arr[:] = 0
            arr.flush()
            del arr
        metadata = {
            "schema_version": "tlab_panel_tools_cache_v1",
            "key": asdict(key),
            "key_sha256": key.digest(),
            "shards": [
                {"name": str(s.get("name", f"shard_{i:05d}")), "shape": [int(x) for x in s["shape"]],
                 "dtype": np.dtype(s.get("dtype", "float32")).str}
                for i, s in enumerate(shards)
            ],
        }
        manifest_tmp = root / "manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, root / "manifest.json")
        return cls(root, key, mode="r+")

    def open_shard(self, name: str, mode: str | None = None) -> np.memmap:
        shard = next((x for x in self.metadata["shards"] if x["name"] == name), None)
        if shard is None:
            raise KeyError(name)
        use_mode = mode or self.mode
        if use_mode not in {"r", "r+"}:
            raise ValueError("shard mode must be r or r+")
        return np.memmap(
            self.root / f"{name}.bin",
            dtype=np.dtype(shard["dtype"]),
            mode=use_mode,
            shape=tuple(shard["shape"]),
        )
