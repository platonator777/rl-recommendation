from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from tlab.config import load_config
from tlab.models.actor import ActorShape, ItemGPT2Actor, backward_chunked_full_softmax_ce
from tlab.runtime import collect_environment
from tlab.train.objectives import analytic_reward_kl_policy, full_categorical_kl, group_advantages


def run_smoke(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.project.get("seed", 17))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    model_cfg = config.model
    catalog_size = int(config.dataset.get("catalog_size", 1000))
    actor = ItemGPT2Actor(
        ActorShape(
            vocab_size=catalog_size + 3,
            n_positions=int(model_cfg.get("n_positions", 64)),
            n_layer=int(model_cfg.get("n_layer", 2)),
            n_embd=int(model_cfg.get("n_embd", 128)),
            n_head=int(model_cfg.get("n_head", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor.to(device)
    actor.train()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4)
    microbatch = int(config.training.get("sft_microbatch", 2))
    seq_len = min(int(model_cfg.get("n_positions", 64)), 32)
    steps = 8
    target_count = 0
    started = time.perf_counter()

    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for step in range(steps):
        # Synthetic IDs exercise only tensor/model plumbing; this is never a result.
        ids = torch.randint(3, catalog_size + 3, (microbatch, seq_len), device=device)
        ids[:, 0] = 1
        attention = torch.ones_like(ids)
        targets = ids[:, 1:]
        target_mask = targets >= 3
        target_count += int(target_mask.sum().item())
        optimizer.zero_grad(set_to_none=True)
        autocast = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
        with autocast:
            hidden = actor.encode(ids[:, :-1], attention[:, :-1])
            loss_value = backward_chunked_full_softmax_ce(
                hidden,
                targets,
                target_mask,
                actor.output_weight,
                chunk_size=128,
            )
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        if not torch.isfinite(torch.tensor(loss_value)):
            raise FloatingPointError(f"non-finite smoke loss at step {step}")

    elapsed = time.perf_counter() - started
    # Verify the one-step analytic distribution and exact KL primitives on a toy row.
    log_reference = torch.log_softmax(torch.tensor([[0.2, -0.1, 0.4]], device=device), dim=-1)
    rewards = torch.tensor([[0.0, 1.0, -0.5]], device=device)
    policy = analytic_reward_kl_policy(log_reference, rewards, beta=0.3)
    kl = full_categorical_kl(policy.clamp_min(1e-30).log(), log_reference)
    advantages = group_advantages(torch.tensor([[1.0, 0.0, -1.0]], device=device))
    if not torch.allclose(policy.sum(dim=-1), torch.ones(1, device=device), atol=1e-6):
        raise AssertionError("analytic policy is not normalized")
    if float(kl.item()) < -1e-6 or not torch.isfinite(advantages).all():
        raise AssertionError("RL objective smoke checks failed")

    output: dict[str, Any] = {
        "kind": "synthetic_smoke_only",
        "seed": seed,
        "device": str(device),
        "actor_parameters": actor.parameter_count(),
        "vocabulary_size": actor.vocab_size,
        "steps": steps,
        "synthetic_supervised_targets": target_count,
        "elapsed_seconds": elapsed,
        "examples_per_second": (steps * microbatch) / elapsed,
        "supervised_targets_per_second": target_count / elapsed,
        "kl_toy_value": float(kl.item()),
        "output_policy_sum": policy.sum(dim=-1).detach().cpu().tolist(),
        "environment": collect_environment(),
    }
    if device.type == "cuda":
        output["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
        output["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved()
    output_dir = Path(config.project.get("artifacts_dir", "artifacts/smoke"))
    if not output_dir.is_absolute():
        output_dir = config.path.parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short synthetic correctness/resource smoke run")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.config), indent=2))


if __name__ == "__main__":
    main()
