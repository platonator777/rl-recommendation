from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import yaml

from tlab.data.store import SequenceStore
from tlab.models.actor import ActorShape
from tlab.panel_tools.baselines import evaluate_core_controls, recent_train_popularity
from tlab.panel_tools.data import PanelTarget, build_evaluation_panel, panel_manifest, stable_event_id
from tlab.panel_tools.evaluator import evaluate_actor_panel, load_actor_checkpoint
from tlab.panel_tools.statistics import paired_user_bootstrap, relative_quality_guard


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _assert_new_output(root: Path, path: Path) -> None:
    destination = path.resolve()
    allowed = (root / "artifacts/residual_support").resolve()
    if allowed not in destination.parents:
        raise ValueError(f"panel_tools.1 output must be below artifacts/residual_support: {destination}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite panel_tools.1 output: {destination}")


def load_config(path: str | Path, root: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    project_root = Path(root).resolve() if root is not None else config_path.parents[2]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {"protocol_id", "source_protocol", "processed_dir", "sequence_index", "catalog",
                "baseline_panel", "checkpoints", "reference", "evaluation", "outputs"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"baseline config is missing fields: {sorted(missing)}")
    if set(config["checkpoints"]) != {"weak", "strong"} or config["reference"] != "strong":
        raise ValueError("panel_tools.1 core baseline requires explicit weak/strong checkpoints and strong reference")
    for key in ("comparison_json", "comparison_csv"):
        destination = _resolve(project_root, config["outputs"][key])
        allowed = (project_root / "artifacts/residual_support").resolve()
        if allowed not in destination.parents:
            raise ValueError(f"panel_tools.1 output must be below artifacts/residual_support: {destination}")
    return project_root, config


def _load_panel(path: Path) -> list[PanelTarget]:
    return [PanelTarget(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_baseline_comparison(config_path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    project_root, config = load_config(config_path, root)
    started = time.perf_counter()
    protocol_path = _resolve(project_root, config["source_protocol"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    processed = _resolve(project_root, config["processed_dir"])
    index_path = _resolve(project_root, config["sequence_index"])
    catalog_path = _resolve(project_root, config["catalog"])
    panel_path = _resolve(project_root, config["baseline_panel"])
    for source in (protocol_path, index_path / "index.json", catalog_path, panel_path):
        if not source.exists():
            raise FileNotFoundError(source)

    processed_protocol = json.loads((processed / "protocol.json").read_text(encoding="utf-8"))
    source_hash = processed_protocol["processed_listens_sha256"]
    store = SequenceStore.load_cache(index_path, expected_source_sha256=source_hash)
    catalog = pl.read_parquet(catalog_path).sort("token_id")
    item_ids = np.asarray(catalog.get_column("item_id").to_list(), dtype=np.int64)
    tokens = np.asarray(catalog.get_column("token_id").to_list(), dtype=np.int64)
    train_counts = np.asarray(catalog.get_column("training_count").to_list(), dtype=np.float64)
    if len(set(item_ids.tolist())) != len(item_ids) or not np.array_equal(tokens, np.arange(3, 3 + len(tokens))):
        raise ValueError("catalog raw-ID/token-ID mapping must be one-to-one and contiguous from token 3")
    catalog_mapping_hash = hashlib.sha256(
        "".join(f"{int(raw)}\t{int(token)}\n" for raw, token in zip(item_ids, tokens)).encode("ascii")
    ).hexdigest()
    panel = _load_panel(panel_path)
    if len(panel) != len({target.uid for target in panel}):
        raise ValueError("common baseline panel must contain exactly one event per user")
    panel_hash = hashlib.sha256("\n".join(row.event_id for row in panel).encode("ascii")).hexdigest()
    if protocol.get("evaluation", {}).get("split") not in (None, "validation"):
        raise ValueError("historical panel_tools protocol is not the expected validation panel")

    dataset_settings = protocol.get("dataset", {})
    source_listens = processed / "listens.parquet"
    cutoff = float(processed_protocol["boundaries_source_units"]["train_cutoff_raw"])
    recent_counts, recent_manifest = recent_train_popularity(
        source_listens, catalog_tokens=tokens, train_cutoff=cutoff,
        window_seconds=float(config["evaluation"].get("recent_window_days", 14.0)) * 86400,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_shape = ActorShape(vocab_size=len(item_ids) + 3, n_positions=256, n_layer=4, n_embd=256, n_head=4, dropout=0.0)
    actors: dict[str, torch.nn.Module] = {}
    actor_meta: dict[str, dict[str, Any]] = {}
    paths = {name: _resolve(project_root, value) for name, value in config["checkpoints"].items()}
    for name, checkpoint in paths.items():
        actors[name], actor_meta[name] = load_actor_checkpoint(
            checkpoint, vocab_size=len(item_ids) + 3, n_positions=actor_shape.n_positions,
            n_layer=actor_shape.n_layer, n_embd=actor_shape.n_embd, n_head=actor_shape.n_head, device=device,
        )

    settings = config["evaluation"]
    common = {
        "catalog_item_ids": item_ids,
        "train_popularity_counts": train_counts,
        "recent_popularity_counts": recent_counts,
        "history_events": int(settings.get("history_events", 200)),
        "batch_size": int(settings.get("batch_size", 32)),
        "k": int(settings.get("top_k", 10)),
    }
    strong_started = time.perf_counter()
    core = evaluate_core_controls(
        actors["strong"], store, panel, **common,
        markov_alpha=1.0,
    )
    strong_seconds = time.perf_counter() - strong_started
    weak_started = time.perf_counter()
    weak = evaluate_actor_panel(
        actors["weak"], store, item_ids.tolist(), panel,
        history_events=common["history_events"], batch_size=common["batch_size"], k=common["k"],
    )
    weak_seconds = time.perf_counter() - weak_started
    weak_user = {int(k): float(v["ndcg@10"]) for k, v in weak["per_user_primary"].items()}
    strong_user = {int(k): float(v["ndcg@10"]) for k, v in core["models"]["base_actor"]["primary_user_macro"]["per_user"].items()}
    draws = int(settings.get("bootstrap_draws", 2000))
    paired_strong_vs_weak = paired_user_bootstrap(strong_user, weak_user, n_bootstrap=draws, seed=17041)
    guard_strong_vs_weak = relative_quality_guard(strong_user, weak_user, tolerance=0.02, n_bootstrap=draws, seed=17042)

    models: dict[str, dict[str, Any]] = {}
    for name, entry in core["models"].items():
        metrics = entry["primary_user_macro"]
        models[name] = {"ndcg@10": metrics["ndcg@10"], "recall@10": metrics["recall@10"],
                        "n_users": metrics["n_users"], "n_events": metrics["n_events"],
                        "target_coverage": entry["target_coverage"]}
    models["base_actor_weak"] = {
        "ndcg@10": weak["primary_user_macro"]["ndcg@10"],
        "recall@10": weak["primary_user_macro"]["recall@10"],
        "n_users": weak["primary_user_macro"]["n_users"],
        "n_events": weak["primary_user_macro"]["n_events"],
        "target_coverage": weak["target_coverage"],
    }
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "completed",
        "evaluation_role": "common retrospective baseline comparison; outcomes previously exposed",
        "reference_choice": "strong full_model SFT checkpoint fixed as primary reference before panel_tools.1 policy experiments",
        "not_independent_test": True,
        "sources": {
            "historical_protocol": str(protocol_path.relative_to(project_root)),
            "processed_source_sha256": source_hash,
            "processed_listens_sha256": sha256_file(source_listens),
            "catalog_sha256": sha256_file(catalog_path),
            "catalog_raw_token_mapping_sha256": catalog_mapping_hash,
            "panel_path": str(panel_path.relative_to(project_root)),
            "panel_file_sha256": sha256_file(panel_path),
            "panel_event_ids_sha256": panel_hash,
            "n_items": len(item_ids),
            "n_panel_users": len({target.uid for target in panel}),
            "oov_targets": sum(target.token_id == 2 for target in panel),
            "actor_checkpoints": {
                name: {"path": str(paths[name].relative_to(project_root)), **actor_meta[name]}
                for name in ("weak", "strong")
            },
            "recent_train_popularity": recent_manifest,
        },
        "evaluation": {
            "history_events": common["history_events"], "batch_size": common["batch_size"],
            "top_k": common["k"], "device": str(device),
            "aggregation": "event macro within user, then user macro; one fixed event per user",
            "oov_targets": "retained in denominator as misses",
            "mapping_checks": "unique raw item IDs; unique contiguous token IDs 3..N+2; checkpoint vocab equals catalog plus PAD/BOS/OOV",
        },
        "models": models,
        "strong_vs_weak": {"paired_ndcg10": paired_strong_vs_weak, "relative_2_percent_guard": guard_strong_vs_weak},
        "controls": core["markov"],
        "runtime": {"strong_and_controls_seconds": strong_seconds, "weak_seconds": weak_seconds,
                    "recent_popularity_seconds": recent_manifest["query_seconds"],
                    "total_wall_seconds": time.perf_counter() - started},
        "limitations": [
            "The source validation outcomes and all V1-full_model periods are previously explored; this is retrospective reanalysis.",
            "The full_model actors may have user/event overlap with A_RL; checkpoint availability does not create an actor holdout.",
            "Popularity, Markov, history frequency, and actor rankings are recommendation baselines, not causal policy estimates.",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_core_metrics": core,
        "weak_actor_metrics": weak,
    }
    json_path = _resolve(project_root, config["outputs"]["comparison_json"])
    csv_path = _resolve(project_root, config["outputs"]["comparison_csv"])
    for output in (json_path, csv_path):
        _assert_new_output(project_root, output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "ndcg@10", "recall@10", "n_users", "n_events", "target_coverage"))
        writer.writeheader()
        for name, metrics in models.items():
            writer.writerow({"model": name, **metrics})
    return result


def freeze_user_roles(config_path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    project_root, config = load_config(config_path, root)
    role_path = _resolve(project_root, config["outputs"]["role_manifest"])
    train_path = _resolve(project_root, config["outputs"]["training_panel"])
    dev_path = _resolve(project_root, config["outputs"]["development_panel"])
    final_path = _resolve(project_root, config["outputs"]["final_panel"])
    manifest_paths = [
        _resolve(project_root, config["outputs"]["training_panel_manifest"]),
        _resolve(project_root, config["outputs"]["development_panel_manifest"]),
        _resolve(project_root, config["outputs"]["final_panel_manifest"]),
    ]
    for target in [role_path, train_path, dev_path, final_path, *manifest_paths]:
        _assert_new_output(project_root, target)
    protocol = json.loads(_resolve(project_root, config["source_protocol"]).read_text(encoding="utf-8"))
    data_dir = _resolve(project_root, config["processed_dir"])
    processed_protocol = json.loads((data_dir / "protocol.json").read_text(encoding="utf-8"))
    source_hash = processed_protocol["processed_listens_sha256"]
    store = SequenceStore.load_cache(_resolve(project_root, config["sequence_index"]), expected_source_sha256=source_hash)
    eligible = [i for i, user in enumerate(store.users) if len(user.rl_anchors) > 0]
    if len(eligible) < 5000:
        raise RuntimeError(f"expected at least 5000 distinct A_RL users; found {len(eligible)}")
    ordered = sorted(eligible, key=lambda i: hashlib.sha256(f"{bytes.fromhex('76345f313a726f6c653a3430313a').decode('ascii')}{store.users[i].uid}".encode()).digest())
    n_train = int(len(ordered) * 0.75)
    n_dev = int(len(ordered) * 0.10)
    train_indices, dev_indices, final_indices = ordered[:n_train], ordered[n_train:n_train+n_dev], ordered[n_train+n_dev:]
    if len(train_indices) < 4096 or len(dev_indices) < 500 or len(final_indices) < 500:
        raise RuntimeError("fixed role split cannot meet the minimum train/dev/final user sizes")
    groups = {
        "train": {int(store.users[i].uid) for i in train_indices},
        "development": {int(store.users[i].uid) for i in dev_indices},
        "final": {int(store.users[i].uid) for i in final_indices},
    }
    if groups["train"] & groups["development"] or groups["train"] & groups["final"] or groups["development"] & groups["final"]:
        raise AssertionError("user roles overlap")

    training: list[PanelTarget] = []
    for user_index in train_indices:
        user = store.users[user_index]
        anchor_order = sorted(int(p) for p in user.rl_anchors)
        position = min(anchor_order, key=lambda p: hashlib.sha256(f"{bytes.fromhex('76345f313a616e63686f723a3430313a').decode('ascii')}{user.uid}:{p}".encode()).digest())
        raw_id = int(user.raw_item_ids[position])
        prior = np.asarray(user.raw_item_ids[:position], dtype=np.int64)
        ratio = float(user.ratio_raw[position])
        training.append(PanelTarget(
            event_id=stable_event_id(source_hash, int(user.uid), position, raw_id),
            uid=int(user.uid), user_index=int(user_index), position=position, raw_item_id=raw_id,
            token_id=int(user.tokens[position]), ratio=ratio if math.isfinite(ratio) else None,
            positive=bool(math.isfinite(ratio) and ratio >= 0.5),
            is_new=not bool(np.any(prior == raw_id)), is_repeat=bool(np.any(prior == raw_id)),
        ))
    training.sort(key=lambda row: (row.uid, row.position))
    validation_all = build_evaluation_panel(store, split="validation", targets_per_user=1, seed=401, source_sha256=source_hash)
    test_all = build_evaluation_panel(store, split="test", targets_per_user=1, seed=401, source_sha256=source_hash)
    development = [row for row in validation_all if row.uid in groups["development"]]
    final = [row for row in test_all if row.uid in groups["final"]]
    if len(development) < 500 or len(final) < 500:
        raise RuntimeError(f"fixed role split has too few eligible outcomes: dev={len(development)}, final={len(final)}")
    role_ids = {name: sorted(uids) for name, uids in groups.items()}
    role_payload = {
        "schema_version": 1, "protocol_id": config["protocol_id"],
        "status": "frozen_before panel_tools.1 policy scoring",
        "role_assignment": "sort eligible A_RL user IDs by SHA256('residual_support:role:401:uid'); first floor(75%) train, next floor(10%) development, remainder final",
        "n_eligible_a_rl_users": len(eligible), "n_train_users": len(groups["train"]),
        "n_development_users": len(groups["development"]), "n_final_users": len(groups["final"]),
        "n_train_contexts": len(training), "n_development_events": len(development), "n_final_events": len(final),
        "a_rl_cutoff_raw": processed_protocol["boundaries_source_units"]["rl_anchor_start_raw"],
        "train_cutoff_raw_exclusive": processed_protocol["boundaries_source_units"]["train_cutoff_raw"],
        "validation_is_development": True, "test_label": "retrospective final scoring; prior exposure remains",
        "role_user_ids_sha256": {
            name: hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()
            for name, values in role_ids.items()
        },
        "source_sha256": source_hash,
        "user_overlap": {"train_development": 0, "train_final": 0, "development_final": 0},
        "evaluation_outcome_independence": False,
    }
    data = (("training", training, train_path, manifest_paths[0]),
            ("development", development, dev_path, manifest_paths[1]),
            ("final", final, final_path, manifest_paths[2]))
    for label, panel, panel_path, manifest_path in data:
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel_path.write_text("".join(json.dumps(row.to_dict(), ensure_ascii=False, allow_nan=False) + "\n" for row in panel), encoding="utf-8")
        selected = panel_manifest(panel, split="train" if label == "training" else ("validation" if label == "development" else "test"), seed=401, source_sha256=source_hash)
        selected.update({"role": label, "not_independent_test": True, "panel_sha256": sha256_file(panel_path)})
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    role_path.parent.mkdir(parents=True, exist_ok=True)
    role_path.write_text(json.dumps(role_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return role_payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="panel_tools.1 common actor baseline and user-role protocol")
    parser.add_argument("--config", default="configs/residual_support/core.yaml")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--freeze-roles", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze_roles:
        result = freeze_user_roles(args.config, args.workspace)
    else:
        result = run_baseline_comparison(args.config, args.workspace)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
