from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tlab.models.actor import ActorShape, ItemGPT2Actor, backward_chunked_full_softmax_ce
from tlab.panel_tools.dense_sft import dense_full_softmax_backward


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_rss_bytes() -> int | None:
    if not hasattr(__import__("ctypes"), "WinDLL"):
        return None
    try:
        import ctypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("Kernel32.dll")
        psapi = ctypes.WinDLL("Psapi.dll")
        process = kernel.GetCurrentProcess()
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    except (AttributeError, OSError, TypeError):
        return None


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _run_backward_method(
    actor: ItemGPT2Actor,
    input_ids: torch.Tensor,
    attention: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    method: str,
    chunk_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    device = input_ids.device
    forward_times: list[float] = []
    backward_times: list[float] = []
    losses: list[float] = []
    peak_allocated = None
    for iteration in range(warmup + repeats):
        actor.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        start = time.perf_counter()
        hidden = actor.encode(input_ids, attention)
        _sync(device)
        forward_stop = time.perf_counter()
        if method == "repeated_encoder_backward":
            loss = backward_chunked_full_softmax_ce(
                hidden, targets, valid, actor.output_weight, chunk_size=chunk_size,
            )
        elif method == "one_encoder_backward":
            loss = dense_full_softmax_backward(
                hidden, targets, valid, actor.output_weight, chunk_size=chunk_size,
            )
        else:
            raise ValueError(method)
        _sync(device)
        end = time.perf_counter()
        if iteration >= warmup:
            forward_times.append(forward_stop - start)
            backward_times.append(end - forward_stop)
            losses.append(float(loss))
        if device.type == "cuda":
            used = int(torch.cuda.max_memory_allocated(device))
            peak_allocated = max(peak_allocated or 0, used)
        del hidden
    gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in actor.named_parameters()
        if parameter.grad is not None
    }
    valid_count = int(valid.sum().item())
    step_median = _median([a + b for a, b in zip(forward_times, backward_times)])
    return {
        "method": method,
        "forward_seconds_median": _median(forward_times),
        "backward_seconds_median": _median(backward_times),
        "step_seconds_median": step_median,
        "step_seconds_samples": [a + b for a, b in zip(forward_times, backward_times)],
        "loss_median": _median(losses),
        "valid_targets_per_step": valid_count,
        "windows_per_step": int(input_ids.shape[0]),
        "valid_targets_per_second": valid_count / step_median,
        "windows_per_second": input_ids.shape[0] / step_median,
        "peak_cuda_allocated_bytes": peak_allocated,
        "gradients": gradients,
    }


def run_dense_backward_benchmark(
    *,
    vocab_size: int = 20_003,
    batch_windows: int = 2,
    sequence_length: int = 256,
    valid_targets_per_window: int = 64,
    chunk_size: int = 32,
    warmup: int = 1,
    repeats: int = 3,
    seed: int = 17,
    device: str | None = None,
) -> dict[str, Any]:
    if vocab_size <= 3 or sequence_length < valid_targets_per_window or min(batch_windows, chunk_size, repeats) < 1:
        raise ValueError("invalid dense benchmark shape or repetition count")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    shape = ActorShape(vocab_size=vocab_size, n_positions=sequence_length, n_layer=4, n_embd=256,
                       n_head=4, dropout=0.0)
    actor = ItemGPT2Actor(shape).to(selected).eval()
    generator = torch.Generator(device=selected).manual_seed(seed + 1)
    inputs = torch.randint(3, vocab_size, (batch_windows, sequence_length), generator=generator, device=selected)
    attention = torch.ones_like(inputs)
    targets = torch.randint(3, vocab_size, inputs.shape, generator=generator, device=selected)
    valid = torch.zeros_like(inputs, dtype=torch.bool)
    valid[:, -valid_targets_per_window:] = True
    baseline_rss = _measure_rss_bytes()

    repeated = _run_backward_method(
        actor, inputs, attention, targets, valid, method="repeated_encoder_backward",
        chunk_size=chunk_size, warmup=warmup, repeats=repeats,
    )
    repeated_gradients = repeated.pop("gradients")
    single = _run_backward_method(
        actor, inputs, attention, targets, valid, method="one_encoder_backward",
        chunk_size=chunk_size, warmup=warmup, repeats=repeats,
    )
    single_gradients = single.pop("gradients")
    if repeated_gradients.keys() != single_gradients.keys():
        raise AssertionError("benchmark methods produced different parameter gradient sets")
    gradient_error = max(
        float((repeated_gradients[name] - single_gradients[name]).abs().max())
        for name in repeated_gradients
    )
    loss_difference = abs(repeated["loss_median"] - single["loss_median"])
    if gradient_error > 2e-4 or loss_difference > 1e-5:
        raise AssertionError(f"benchmark methods failed gradient equivalence: loss={loss_difference}, grad={gradient_error}")

    inference_ids = inputs[:min(batch_windows, 16), :min(sequence_length, 200)]
    inference_mask = torch.ones_like(inference_ids)
    inference_times: list[float] = []
    with torch.inference_mode():
        for _ in range(2):
            actor.next_action_log_probs(inference_ids, inference_mask)
        for _ in range(5):
            _sync(selected)
            started = time.perf_counter()
            actor.next_action_log_probs(inference_ids, inference_mask)
            _sync(selected)
            inference_times.append(time.perf_counter() - started)
    inference_median = _median(inference_times)
    for row in (repeated, single):
        row["peak_cuda_allocated_mib"] = (
            row.pop("peak_cuda_allocated_bytes") / (1024 ** 2)
            if row.get("peak_cuda_allocated_bytes") is not None else None
        )
    return {
        "schema_version": 1,
        "protocol_id": "tlab_yambda_policy_diagnostics026-10-21_r1",
        "status": "completed",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": str(selected),
            "gpu": torch.cuda.get_device_name(selected) if selected.type == "cuda" else None,
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(selected).total_memory if selected.type == "cuda" else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cpu_rss_bytes_before": baseline_rss,
            "cpu_rss_bytes_after": _measure_rss_bytes(),
            "psutil_installed": False,
        },
        "workload": {
            "catalog_vocab_size": vocab_size,
            "actor": {"layers": 4, "hidden": 256, "heads": 4, "dropout_enabled": False},
            "batch_windows": batch_windows,
            "sequence_length": sequence_length,
            "valid_targets_per_window": valid_targets_per_window,
            "total_valid_targets_per_step": int(valid.sum().item()),
            "logit_chunk_size": chunk_size,
            "warmup_steps": warmup,
            "timed_repeats": repeats,
            "precision": "FP32 exact benchmark; AMP correctness is covered separately",
            "input_pipeline": "synthetic preloaded tensors; data read and feature extraction are not included",
        },
        "methods": [repeated, single],
        "equivalence": {
            "loss_abs_difference": loss_difference,
            "gradient_max_abs_difference": gradient_error,
            "passed": True,
        },
        "inference": {
            "batch_size": int(inference_ids.shape[0]),
            "history_length": int(inference_ids.shape[1]),
            "full_catalog_seconds_median": inference_median,
            "contexts_per_second": int(inference_ids.shape[0]) / inference_median,
            "repeats": len(inference_times),
        },
        "cost_projection": {
            "windows_per_gpu_hour": {
                row["method"]: row["windows_per_second"] * 3600.0 for row in (repeated, single)
            },
            "valid_targets_per_gpu_hour": {
                row["method"]: row["valid_targets_per_second"] * 3600.0 for row in (repeated, single)
            },
            "full_fit_hours": None,
            "reason": "full dense-window count and real input-pipeline throughput are measured after the correctness-gated benchmark",
        },
    }


def run_and_save(workspace: str | Path, output: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
    root = Path(workspace).resolve()
    path = Path(output) if output else root / "artifacts/panel_tools/benchmark/dense_backward.json"
    if not path.is_absolute():
        path = root / path
    result = run_dense_backward_benchmark(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark output: {path}")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Correctness-gated panel_tools dense SFT throughput/memory benchmark")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output")
    parser.add_argument("--device")
    args = parser.parse_args()
    result = run_and_save(args.workspace, args.output, device=args.device)
    print(json.dumps({
        "output": args.output or str(Path(args.workspace) / "artifacts/panel_tools/benchmark/dense_backward.json"),
        "methods": result["methods"],
        "equivalence": result["equivalence"],
        "inference": result["inference"],
    }, indent=2))


if __name__ == "__main__":
    main()
