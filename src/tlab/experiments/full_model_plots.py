from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tlab.config import load_config
from tlab.data.manifest import sha256_file, write_json_atomic


def _paths(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    root = config.path.parent.parent
    artifacts = Path(config.project.get("artifacts_dir", "artifacts/full_model"))
    if not artifacts.is_absolute():
        artifacts = root / artifacts
    return root, artifacts


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _plot_support(artifacts: Path, out: Path) -> str | None:
    payload = _read(artifacts / "e1" / "support_decomposition.json")
    if not payload:
        return None
    models = payload.get("models", {})
    selected = [(name, model) for name, model in models.items()
                if isinstance(model.get("support_decomposition"), dict)]
    if not selected:
        return None
    groups = ("0", "1_9", "10_99", "100_plus")
    labels = [name.replace("reference ", "").replace("full_model ", "") for name, _ in selected]
    colors = ("#b64545", "#e3a640", "#5a9d75", "#4277a8")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for axis, metric, ylabel, title in (
        (axes[0], "policy_mass", "Policy probability mass", "Where probability mass goes"),
        (axes[1], "reward_contribution_share", "Share of expected raw reward", "Where predicted reward comes from"),
    ):
        bottoms = np.zeros(len(selected), dtype=np.float64)
        for group, color in zip(groups, colors):
            values = np.asarray([_number(model["support_decomposition"].get(group, {}).get(metric)) or 0.0
                                 for _, model in selected])
            axis.bar(np.arange(len(selected)), values, bottom=bottoms, color=color, label=group.replace("_", "–"))
            bottoms += values
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if metric == "policy_mass":
            axis.set_ylim(0, 1)
        axis.legend(title="RM-fit event support", ncol=4, frameon=False)
    axes[-1].set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    return _save(fig, out / "e1_support_decomposition.png")


def _plot_frontier(artifacts: Path, out: Path) -> str | None:
    payload = _read(artifacts / "e3" / "frontier.json")
    if not payload:
        return None
    rows = [row for row in payload.get("runs", []) if row.get("development")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10, 7))
    color_by_method = {"sampled": "#3376a8", "exact": "#d17636"}
    marker_by_support = {"ordinary": "o", "penalized": "s"}
    for row in rows:
        metrics = row["development"]
        x = _number(metrics.get("expected_raw_reward"))
        y = _number(metrics.get("ndcg@10"))
        if x is None or y is None:
            continue
        label = f"{row.get('method')} {row.get('support_mode')} β{row.get('beta')}"
        ax.scatter(x, y, color=color_by_method.get(row.get("method"), "#666666"),
                   marker=marker_by_support.get(row.get("support_mode"), "o"),
                   s=42 + 18 * math.log2(max(1, int(row.get("update", 1)))), alpha=0.72,
                   label=label if label not in ax.get_legend_handles_labels()[1] else None)
    for row in payload.get("analytic_b5", []):
        x, y = _number(row.get("expected_raw_reward")), _number(row.get("ndcg@10"))
        if x is not None and y is not None:
            ax.scatter(x, y, marker="X", s=100, color="#7a5aa6", label="analytic B5" if "analytic B5" not in ax.get_legend_handles_labels()[1] else None)
    ax.set_xlabel("Expected RM-fit Q1 reward (development panel)")
    ax.set_ylabel("Macro-user NDCG@10")
    ax.set_title("full_model reward–relevance frontier (development; known validation)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    return _save(fig, out / "e3_reward_relevance_frontier.png")


def _plot_learning(artifacts: Path, out: Path) -> str | None:
    runs = artifacts / "runs"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    found = False
    for method, support, color in (("sampled", "ordinary", "#3376a8"),
                                   ("exact", "ordinary", "#d17636"),
                                   ("sampled", "penalized", "#4d916c"),
                                   ("exact", "penalized", "#8c6cab")):
        path = runs / f"q1_{method}_{support}_b1_s17" / "validation.jsonl"
        if not path.is_file():
            continue
        points = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metrics = row.get("development", {})
            update = int(row.get("update", -1))
            reward, ndcg = _number(metrics.get("expected_raw_reward")), _number(metrics.get("ndcg@10"))
            if update in {0, 100, 500, 1000, 2500, 5000} and reward is not None and ndcg is not None:
                points.append((update, reward, ndcg))
        if points:
            found = True
            points.sort()
            label = f"{method} {support}"
            axes[0].plot([p[0] for p in points], [p[1] for p in points], marker="o", color=color, label=label)
            axes[1].plot([p[0] for p in points], [p[2] for p in points], marker="o", color=color, label=label)
    if not found:
        plt.close(fig)
        return None
    axes[0].set_ylabel("Expected raw Q1 reward")
    axes[1].set_ylabel("NDCG@10")
    for axis in axes:
        axis.set_xlabel("Optimizer context batches")
        axis.set_xticks([0, 100, 500, 1000, 2500, 5000])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle("Matched β=1 learning curves; development panel")
    return _save(fig, out / "e2_learning_curves.png")


def _plot_horizon(artifacts: Path, out: Path) -> str | None:
    payload = _read(artifacts / "e4" / "h5_context_intervention.json")
    if not payload:
        return None
    rows = payload.get("summary", {}).get("same_item_fit_gap_by_source_and_step", [])
    if not rows:
        return None
    labels = list(dict.fromkeys(row.get("model") for row in rows))
    sources = list(dict.fromkeys(row.get("source_mode") for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(labels) * len(sources))))
    color_index = 0
    for model in labels:
        for source in sources:
            group = sorted([row for row in rows if row.get("model") == model and row.get("source_mode") == source],
                           key=lambda row: int(row["step"]))
            if not group:
                continue
            axes[0].plot([row["step"] for row in group], [row["mean"] for row in group],
                         marker="o", color=colors[color_index], label=f"{model}; {source}")
            color_index += 1
    js_rows = payload.get("summary", {}).get("policy_context_js_by_model_seed_step", [])
    for index, model in enumerate(labels):
        group = sorted([row for row in js_rows if row.get("model") == model], key=lambda row: int(row["step"]))
        if group:
            axes[1].plot([row["step"] for row in group], [row["mean"] for row in group],
                         marker="o", color=plt.cm.tab10(index % 10), label=model)
    axes[0].set_title("Same action: RM-fit score, appended − fixed context")
    axes[0].set_ylabel("Predicted ratio difference")
    axes[1].set_title("Policy shift from context substitution")
    axes[1].set_ylabel("Jensen–Shannon divergence (nats)")
    for axis in axes:
        axis.set_xlabel("Generated horizon step")
        axis.set_xticks(range(1, int(payload.get("horizon", 5)) + 1))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, frameon=False)
    return _save(fig, out / "e4_context_intervention.png")


def _plot_temporal(artifacts: Path, out: Path) -> str | None:
    records = []
    lr_audit = _read(artifacts / "e5" / "tminus14" / "sft_lr_trace_audit.json") or {}
    lr_mismatch = lr_audit.get("status") == "recipe_mismatch"
    sources = [
        ("tminus14", "T−14 initial validation (3× LR; descriptive)", "T−14 initial Test (3× LR; disclosed)",
         "#9aa5ad", "#ad837b"),
        ("lrfix/e5/tminus14", "T−14 corrected validation", "T−14 corrected Test (same disclosed panel)",
         "#3376a8", "#c65a48"),
        ("tminus28", "T−28 validation", "T−28 Test", "#3376a8", "#c65a48"),
    ]
    for fold, validation_label, test_label_base, validation_color, test_color in sources:
        fold_base = artifacts / "e5" / fold if fold != "lrfix/e5/tminus14" else artifacts / "lrfix" / "e5" / "tminus14"
        payload = _read(fold_base / "local_validation_once.json")
        if payload:
            for recipe, seeds in payload.get("models", {}).items():
                if recipe == "local SFT" or not isinstance(seeds, dict):
                    continue
                for seed, values in seeds.items():
                    ci = values.get("paired_to_local_sft", {}).get("ndcg@10", {})
                    mean = _number(ci.get("mean_difference"))
                    low, high = _number(ci.get("ci_95_low")), _number(ci.get("ci_95_high"))
                    if mean is not None and low is not None and high is not None:
                        records.append((f"{validation_label} {recipe} s{seed}", mean, low, high, validation_color))
        payload = _read(fold_base / "final_test_once.json")
        if payload:
            for recipe, seeds in payload.get("models", {}).items():
                if recipe == "local SFT" or not isinstance(seeds, dict):
                    continue
                for seed, values in seeds.items():
                    ci = values.get("paired_to_local_sft", {}).get("ndcg@10", {})
                    mean = _number(ci.get("mean_difference"))
                    low, high = _number(ci.get("ci_95_low")), _number(ci.get("ci_95_high"))
                    if mean is not None and low is not None and high is not None:
                        records.append((f"{test_label_base} {recipe} s{seed}", mean, low, high, test_color))
    if not records:
        return None
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.28 * len(records))))
    y = np.arange(len(records))
    for index, (label, mean, low, high, color) in enumerate(records):
        ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt="o", color=color,
                    capsize=2, markersize=4)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, [row[0] for row in records], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Paired Δ NDCG@10 vs fold-local SFT (95% user bootstrap CI)")
    title = "E5: frozen-policy transfer; each rollout seed shares one fold-local SFT/RM"
    if lr_mismatch:
        title += "\nInitial 3×-LR attempt retained as descriptive; corrected Test reuses the disclosed panel"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, out / "e5_temporal_forest.png")


def make_plots(config_path: str | Path = "configs/full_model_rl.yaml") -> dict[str, Any]:
    _, artifacts = _paths(config_path)
    output = artifacts / "figures"
    outputs = {
        "e1_support_decomposition": _plot_support(artifacts, output),
        "e2_learning_curves": _plot_learning(artifacts, output),
        "e3_reward_relevance_frontier": _plot_frontier(artifacts, output),
        "e4_context_intervention": _plot_horizon(artifacts, output),
        "e5_temporal_forest": _plot_temporal(artifacts, output),
    }
    figures = {name: {"path": path, "sha256": sha256_file(Path(path))} for name, path in outputs.items() if path}
    manifest = {"protocol_id": "tlab_yambda_full_model_2026-10-14-r3.1", "status": "completed",
                "figures": figures, "not_yet_available": [name for name, path in outputs.items() if not path]}
    write_json_atomic(artifacts / "figure_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create preregistered full_model research figures")
    parser.add_argument("--config", default="configs/full_model_rl.yaml")
    args = parser.parse_args()
    print(json.dumps(make_plots(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
