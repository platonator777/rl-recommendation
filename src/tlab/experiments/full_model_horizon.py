from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tlab.config import load_config
from tlab.data.catalog import BOS_ID
from tlab.data.manifest import sha256_file, write_json_atomic
from tlab.data.store import SequenceStore
from tlab.evaluation.metrics import single_target_metrics
from tlab.evaluation.policy_panel import _load_reward_model
from tlab.experiments.full_model_matrix import _collate, _load_store, _root_and_processed, _shape, create_panels
from tlab.models.actor import ItemGPT2Actor
from tlab.train.checkpoint import source_tree_sha256


def _load_actor(path: Path, shape, device: torch.device) -> ItemGPT2Actor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = ItemGPT2Actor(shape)
    actor.load_state_dict(payload["model"])
    actor.to(device).eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def _score_actions(model, features, histories, tokens: list[int], device: torch.device, batch_size: int = 64):
    values = []
    for start in range(0, len(tokens), batch_size):
        batch_histories = histories[start:start + batch_size]
        batch_tokens = tokens[start:start + batch_size]
        ids, attention = _collate(batch_histories, device)
        candidates = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        durations = torch.tensor([features.get(token, (0.0, True, 0))[0] for token in batch_tokens],
                                 dtype=torch.float32, device=device)
        missing = torch.tensor([features.get(token, (0.0, True, 0))[1] for token in batch_tokens],
                               dtype=torch.bool, device=device)
        repeated = torch.tensor([token in set(history[-20:]) for token, history in zip(batch_tokens, batch_histories)],
                                dtype=torch.bool, device=device)
        values.extend(model(ids, attention, candidates, durations, missing, repeated)["ratio"].float().cpu().tolist())
    return values


def _rank_target(log_probs: torch.Tensor, target: int, item_ids: list[int]) -> int | None:
    if target < 3 or target >= log_probs.shape[-1]:
        return None
    catalog = log_probs[3:].float()
    target_score = catalog[target - 3]
    target_item = int(item_ids[target - 3])
    ids = torch.as_tensor(item_ids, device=catalog.device)
    return int(((catalog > target_score) | ((catalog == target_score) & (ids < target_item))).sum().item()) + 1


def _policy_step(actor, histories, device, generator, batch_size: int = 32):
    tokens: list[int] = []
    entropies: list[float] = []
    log_parts: list[torch.Tensor] = []
    for start in range(0, len(histories), batch_size):
        ids, attention = _collate(histories[start:start + batch_size], device)
        log_probs = actor.next_action_log_probs(ids, attention)
        probabilities = log_probs[:, 3:].float().exp()
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        actions = torch.multinomial(probabilities, 1, generator=generator).squeeze(1) + 3
        tokens.extend(int(value) for value in actions.cpu().tolist())
        entropies.extend(float(value) for value in entropy.cpu().tolist())
        log_parts.append(log_probs.detach())
    return tokens, entropies, log_parts


def _policy_divergence(fixed_log_parts, append_log_parts) -> list[float]:
    output = []
    for fixed_log, append_log in zip(fixed_log_parts, append_log_parts):
        fixed = fixed_log[:, 3:].float().exp()
        append = append_log[:, 3:].float().exp()
        midpoint = (fixed + append) * 0.5
        js = 0.5 * ((fixed * (fixed.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())).sum(-1)
                    + (append * (append.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())).sum(-1))
        output.extend(float(value) for value in js.cpu().tolist())
    return output


def _append_synthetic_token(history: list[int], token: int, history_length: int) -> list[int]:
    if history_length < 1:
        raise ValueError("history_length must include at least the BOS token")
    events = [*history[1:], int(token)]
    if history_length == 1:
        return [BOS_ID]
    return [BOS_ID, *events[-(history_length - 1):]]


def _model_sources(root: Path, artifacts: Path) -> dict[str, Path]:
    result: dict[str, Path] = {
        "SFT strong17": root / "artifacts/reference/runs/sft_strong_seed17/best.pt",
        "H5 update500": root / "artifacts/reference/runs/grpo_h5_reference_strong_seed17_beta1_full/validation_checkpoints/update_00500.pt",
        "H5 update1000": root / "artifacts/reference/runs/grpo_h5_reference_strong_seed17_beta1_full/validation_checkpoints/update_01000.pt",
    }
    selected_full_model_path = artifacts / "candidate_freeze.json"
    if selected_full_model_path.is_file():
        freeze = json.loads(selected_full_model_path.read_text(encoding="utf-8"))
        selected = freeze.get("selected_candidate", {})
        selected_path = Path(selected.get("checkpoint", ""))
        if not selected_path.is_absolute():
            selected_path = root / selected_path
        if selected_path.is_file():
            result["full_model frozen H1 candidate"] = selected_path
    reference_selected_path = root / "artifacts/reference/h1_frozen_candidates_beta1.json"
    if reference_selected_path.is_file():
        reference_selected = json.loads(reference_selected_path.read_text(encoding="utf-8"))
        for model in reference_selected.get("models", []):
            if model.get("name") in {"strong17", "strong42", "strong73"}:
                checkpoint = Path(model["checkpoint"]["path"])
                if checkpoint.is_file():
                    result[f"reference H1 {model['name']}"] = checkpoint
    return result


def _anchor_windows(store: SequenceStore, panels: dict[str, Any], history_length: int,
                    horizon: int, requested: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluation = store.evaluation_targets("validation", per_user_limit=None)
    by_user: dict[int, list[int]] = defaultdict(list)
    for user_index, position in evaluation:
        by_user[user_index].append(position)
    for positions in by_user.values():
        positions.sort()
    selected = (panels["development"] + panels["confirmation"])[:requested]
    windows = []
    for panel_row in selected:
        user_index = int(panel_row["user_index"])
        anchor = int(panel_row["position"])
        positions = by_user.get(user_index, [])
        future = [position for position in positions if position >= anchor][:horizon]
        windows.append({"user_index": user_index, "uid": int(panel_row["uid"]),
                        "anchor_position": anchor, "target_positions": future,
                        "complete_window": len(future) == horizon,
                        "anchor_id": panel_row["anchor_id"]})
    if len({row["uid"] for row in windows}) != len(windows):
        raise AssertionError("E4 validation anchors must be unique by user")
    return windows, {"requested": requested, "actual": len(windows),
                     "complete_window_count": sum(row["complete_window"] for row in windows),
                     "teacher_forcing_windows_by_length": {
                         str(length): sum(len(row["target_positions"]) >= length for row in windows)
                         for length in range(1, horizon + 1)},
                     "source": "fixed main validation panel; one known validation anchor per distinct user"}


@torch.no_grad()
def run_h5_context_intervention(config_path: str | Path = "configs/full_model_horizon.yaml",
                                *, anchor_count: int = 1024, horizon: int = 5,
                                seeds: tuple[int, ...] = (17, 42, 73)) -> dict[str, Any]:
    config, processed, artifacts = _root_and_processed(config_path)
    root = config.path.parent.parent
    panels = create_panels(config_path)
    store, data_protocol = _load_store(processed)
    windows, panel_summary = _anchor_windows(
        store, panels, int(config.dataset.get("history_length", 201)), horizon, anchor_count)
    if not windows:
        raise ValueError("E4 has no validation windows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    shape = _shape(config, int(data_protocol["catalog_size"]))
    actor_paths = _model_sources(root, artifacts)
    missing = [str(path) for path in actor_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E4 checkpoint(s) are missing: {missing}")
    fit_path = root / "artifacts/reference/runs/rm_reference_fit_seed17/best.pt"
    audit_path = root / "artifacts/reference/runs/rm_reference_audit_seed17/best.pt"
    fit_model, fit_features = _load_reward_model(config, fit_path, "fit", device)
    audit_model, audit_features = _load_reward_model(config, audit_path, "audit", device)
    fit_model.eval(); audit_model.eval()
    import polars as pl
    catalog = pl.read_parquet(processed / "catalog.parquet").sort("token_id")
    item_ids = [int(value) for value in catalog.get_column("item_id").to_list()]
    item_id_by_token = {index + 3: item_id for index, item_id in enumerate(item_ids)}
    histories_fixed = [store.users[row["user_index"]].history(
        row["anchor_position"], int(config.dataset.get("history_length", 201))) for row in windows]
    teacher_rows = []
    generated_rows = []
    score_rows = []
    divergence_rows = []
    started = time.perf_counter()
    for model_label, checkpoint in actor_paths.items():
        actor = _load_actor(checkpoint, shape, device)
        actor_meta = torch.load(checkpoint, map_location="cpu", weights_only=False).get("metadata", {})
        per_model_start = len(generated_rows)
        for seed in seeds:
            generators = {
                "fixed_history_slate": torch.Generator(device=device).manual_seed(int(seed) + 100_003),
                "naive_append": torch.Generator(device=device).manual_seed(int(seed) + 200_003),
            }
            appended_histories = [list(history) for history in histories_fixed]
            slate_actions = {"fixed_history_slate": [[] for _ in windows],
                             "naive_append": [[] for _ in windows]}
            for step in range(horizon):
                fixed_tokens, fixed_entropy, fixed_log_parts = _policy_step(
                    actor, histories_fixed, device, generators["fixed_history_slate"])
                append_tokens, append_entropy, append_log_parts = _policy_step(
                    actor, appended_histories, device, generators["naive_append"])
                js_values = _policy_divergence(fixed_log_parts, append_log_parts)
                for idx, window in enumerate(windows):
                    divergence_rows.append({"model": model_label, "seed": int(seed), "step": step + 1,
                                            "uid": window["uid"], "jensen_shannon_nats": js_values[idx],
                                            "fixed_entropy_nats": fixed_entropy[idx],
                                            "append_entropy_nats": append_entropy[idx]})
                for source_mode, tokens in (("fixed_history_slate", fixed_tokens), ("naive_append", append_tokens)):
                    fixed_fit = _score_actions(fit_model, fit_features, histories_fixed, tokens, device)
                    fixed_audit = _score_actions(audit_model, audit_features, histories_fixed, tokens, device)
                    append_fit = _score_actions(fit_model, fit_features, appended_histories, tokens, device)
                    append_audit = _score_actions(audit_model, audit_features, appended_histories, tokens, device)
                    for idx, (token, fit_h0, audit_h0, fit_ht, audit_ht) in enumerate(
                            zip(tokens, fixed_fit, fixed_audit, append_fit, append_audit)):
                        user = store.users[windows[idx]["user_index"]]
                        support_fit = int(fit_features.get(token, (0.0, True, 0))[2])
                        duration = float(fit_features.get(token, (0.0, True, 0))[0])
                        record = {"model": model_label, "seed": int(seed), "step": step + 1,
                                  "uid": windows[idx]["uid"], "source_mode": source_mode,
                                  "action_token_id": token, "action_item_id": item_id_by_token.get(token),
                                  "repeat_within_slate": token in slate_actions[source_mode][idx],
                                  "fit_support_count": support_fit, "fit_supported": support_fit > 0,
                                  "duration_feature_seconds": duration,
                                  "recent_repeat_in_fixed": token in set(histories_fixed[idx][-20:]),
                                  "recent_repeat_in_append": token in set(appended_histories[idx][-20:]),
                                  "rm_fit_fixed_context": float(fit_h0), "rm_fit_append_context": float(fit_ht),
                                  "rm_audit_fixed_context": float(audit_h0), "rm_audit_append_context": float(audit_ht),
                                  "same_item_fit_context_gap": float(fit_ht - fit_h0),
                                  "same_item_audit_context_gap": float(audit_ht - audit_h0),
                                  "state_provenance": "synthetic appended items are not asserted to have been listened to"}
                        generated_rows.append(record)
                        score_rows.append({"model": model_label, "seed": int(seed), "step": step + 1,
                                           "uid": windows[idx]["uid"], "source_mode": source_mode,
                                           "reward_context": "fixed_h0", "token": token,
                                           "rm_fit_ratio": float(fit_h0), "rm_audit_ratio": float(audit_h0),
                                           "fit_supported": support_fit > 0})
                        score_rows.append({"model": model_label, "seed": int(seed), "step": step + 1,
                                           "uid": windows[idx]["uid"], "source_mode": source_mode,
                                           "reward_context": "append_h_t", "token": token,
                                           "rm_fit_ratio": float(fit_ht), "rm_audit_ratio": float(audit_ht),
                                           "fit_supported": support_fit > 0})
                # Only the append policy receives its sampled item as synthetic history.
                for idx, token in enumerate(append_tokens):
                    appended_histories[idx] = _append_synthetic_token(
                        appended_histories[idx], token,
                        int(config.dataset.get("history_length", 201)))
                    slate_actions["naive_append"][idx].append(token)
                    slate_actions["fixed_history_slate"][idx].append(fixed_tokens[idx])

                # Teacher forcing is deterministic for a fixed actor/history, so
                # evaluate it once rather than tripling users across rollout seeds.
                if int(seed) == int(seeds[0]):
                    teacher_idxs = [i for i, window in enumerate(windows)
                                    if len(window["target_positions"]) > step]
                    if teacher_idxs:
                        teacher_histories = [store.users[windows[i]["user_index"]].history(
                            windows[i]["target_positions"][step], int(config.dataset.get("history_length", 201)))
                            for i in teacher_idxs]
                        target_tokens = [int(store.users[windows[i]["user_index"]].tokens[
                            windows[i]["target_positions"][step]]) for i in teacher_idxs]
                        t_log_parts = []
                        for chunk_start in range(0, len(teacher_histories), 32):
                            t_ids, t_mask = _collate(teacher_histories[chunk_start:chunk_start + 32], device)
                            t_log_parts.append(actor.next_action_log_probs(t_ids, t_mask))
                        t_log = torch.cat(t_log_parts)
                        t_log_probs = t_log.gather(1, torch.tensor(target_tokens, device=device)[:, None]).squeeze(1)
                        ranks = [_rank_target(t_log[j], token, item_ids) for j, token in enumerate(target_tokens)]
                        fit_values = _score_actions(fit_model, fit_features, teacher_histories, target_tokens, device)
                        audit_values = _score_actions(audit_model, audit_features, teacher_histories, target_tokens, device)
                        for local, idx in enumerate(teacher_idxs):
                            window = windows[idx]
                            target_log_probability = float(t_log_probs[local].cpu())
                            teacher_rows.append({"model": model_label, "seed": int(seeds[0]), "step": step + 1,
                                                 "uid": window["uid"], "position": window["target_positions"][step],
                                                 "target_token_id": target_tokens[local], "target_rank": ranks[local],
                                                 **single_target_metrics(ranks[local]),
                                                 "target_log_probability": (target_log_probability
                                                                            if math.isfinite(target_log_probability)
                                                                            else None),
                                                 "rm_fit_target_ratio": float(fit_values[local]),
                                                 "rm_audit_target_ratio": float(audit_values[local]),
                                                 "target_is_oov": target_tokens[local] < 3})

        # Per-model fixed-vs-append action summary is derived from rows retaining all provenance.
        _ = per_model_start

    def summarize(rows, keys, value):
        buckets = defaultdict(list)
        for row in rows:
            buckets[tuple(row[key] for key in keys)].append(float(row[value]))
        return [{**dict(zip(keys, key)), "mean": float(np.mean(values)), "std": float(np.std(values)),
                 "n": len(values)} for key, values in sorted(buckets.items(), key=lambda item: str(item[0]))]

    result = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "status": "completed",
              "experiment": "E4 H5 context intervention", "horizon": horizon,
              "anchor_panel": panel_summary, "seeds": list(seeds),
              "models": {label: {"checkpoint": str(path.resolve()), "sha256": sha256_file(path)}
                         for label, path in actor_paths.items()},
              "rm_models": {"fit": {"path": str(fit_path.resolve()), "sha256": sha256_file(fit_path),
                                     "used_for_primary_reward_diagnostics": True},
                            "audit": {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path),
                                      "previously_revealed": True, "independent_confirmation": False}},
              "generated_actions": generated_rows, "same_item_cross_context": score_rows,
              "policy_context_divergence": divergence_rows, "logged_teacher_forcing": teacher_rows,
              "summary": {
                  "same_item_fit_gap_by_source_and_step": summarize(generated_rows,
                      ["model", "source_mode", "step"], "same_item_fit_context_gap"),
                  "same_item_audit_gap_by_source_and_step": summarize(generated_rows,
                      ["model", "source_mode", "step"], "same_item_audit_context_gap"),
                  "policy_context_js_by_model_seed_step": summarize(divergence_rows,
                      ["model", "seed", "step"], "jensen_shannon_nats"),
                  "teacher_forced_ndcg10_by_model_step": summarize(teacher_rows,
                      ["model", "step"], "ndcg@10"),
                  "unsupported_rate_by_source_and_step": summarize(
                      [{**row, "unsupported": 1.0 - float(row["fit_supported"])} for row in generated_rows],
                      ["model", "source_mode", "step"], "unsupported"),
                  "fit_reward_by_context_source_step": summarize(score_rows,
                      ["model", "seed", "source_mode", "step", "reward_context"], "rm_fit_ratio"),
              },
              "protocol_notes": {"validation_is_known_and_repartitioned": True,
                                 "teacher_forcing_uses_only_real_observed_targets": True,
                                 "generated_append_is_synthetic_not_feedback": True,
                                 "generated_actions_are_not_given_logged_relevance": True,
                                 "cross_cells_are_stress_diagnostics": True},
              "elapsed_seconds": time.perf_counter() - started,
              "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
              "source_tree_sha256": source_tree_sha256(root / "src/tlab")}
    target = artifacts / "e4" / "h5_context_intervention.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    return {key: value for key, value in result.items() if key not in {
        "generated_actions", "same_item_cross_context", "policy_context_divergence", "logged_teacher_forcing"}}


def main() -> None:
    parser = argparse.ArgumentParser(description="full_model E4 H5 context intervention")
    parser.add_argument("--config", default="configs/full_model_horizon.yaml")
    parser.add_argument("--anchors", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    args = parser.parse_args()
    print(json.dumps(run_h5_context_intervention(args.config, anchor_count=args.anchors,
                                                   horizon=args.horizon, seeds=tuple(args.seeds)),
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
