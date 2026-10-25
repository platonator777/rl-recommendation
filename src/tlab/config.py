from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def project(self) -> dict[str, Any]:
        return self.raw.get("project", {})

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw.get("dataset", {})

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def training(self) -> dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def data(self) -> dict[str, Any]:
        return self.raw.get("data", {})


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config without mutating it; PyYAML is an install dependency."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - covered by install instructions
        raise RuntimeError("PyYAML is required; install the project dependencies first") from exc

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    for section in ("project", "dataset", "model", "training", "data"):
        value = raw.get(section, {})
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping")
    return ExperimentConfig(raw=raw, path=config_path)
