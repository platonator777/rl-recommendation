from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "huggingface_hub", "pyarrow", "polars", "numpy")
        },
    }
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                ("extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            result["memory_bytes"] = {"total": status.total_phys, "available": status.avail_phys}
    except (AttributeError, OSError):
        pass

    disk = shutil.disk_usage(Path.cwd())
    result["disk_bytes"] = {"total": disk.total, "free": disk.free}
    try:
        import torch

        result["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_vram_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        result["torch"] = {"installed": False}

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            result["nvidia_smi"] = completed.stdout.strip().splitlines()
        except (subprocess.SubprocessError, OSError):
            result["nvidia_smi"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Record hardware and Python environment details")
    parser.add_argument("--output", default="artifacts/environment.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = collect_environment()
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
