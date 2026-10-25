from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save_rm_calibration(artifacts: Path, output: Path) -> Path | None:
    import matplotlib.pyplot as plt

    variants = {
        name: _read_json(artifacts / f"rm_diagnostics_{name}_seed17.json")
        for name in ("fit", "audit")
    }
    if not any(variants.values()):
        return None
    heads = (("ratio", "Completion ratio"), ("seconds", "Listening seconds"), ("like", "New like probability"))
    fig, axes = plt.subplots(1, len(heads), figsize=(15, 4.6))
    for ax, (head, title) in zip(axes, heads):
        plotted = False
        bounds: list[float] = []
        for variant, data in variants.items():
            if not data:
                continue
            rows = data.get("calibration_deciles", {}).get(head, [])
            if not rows:
                continue
            x = [row["mean_predicted"] for row in rows]
            y = [row["mean_target"] for row in rows]
            ax.plot(x, y, marker="o", linewidth=1.7, label=f"RM_{variant}")
            bounds.extend(x + y)
            plotted = True
        if plotted:
            lo = min(0.0, min(bounds))
            hi = max(bounds) * 1.04 if max(bounds) > 0 else 1.0
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.45", linewidth=1, label="ideal")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No eligible labels", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Mean predicted")
        ax.set_ylabel("Mean observed")
        ax.grid(alpha=0.2)
    fig.suptitle("Reward model calibration on V_RM (equal-frequency deciles)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _sft_run_paths(artifacts: Path) -> list[Path]:
    names = (
        "sft_seed17_20261012T170456Z", "sft_seed17_b2_continuation", "sft_seed17_b3_positive_only",
        "sft_seed42_final", "sft_seed42_b2_continuation", "sft_seed42_b3_positive_only",
        "sft_seed73_b1", "sft_seed73_b2_continuation", "sft_seed73_b3_positive_only",
    )
    return [artifacts / "runs" / name for name in names if (artifacts / "runs" / name / "metrics.jsonl").is_file()]


def _save_sft_learning_curves(artifacts: Path, output: Path) -> Path | None:
    import matplotlib.pyplot as plt

    runs = _sft_run_paths(artifacts)
    if not runs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for run in runs:
        rows = _run_json_lines(run / "metrics.jsonl")
        if not rows:
            continue
        label = run.name.replace("sft_", "")
        epochs = [int(row.get("epoch", index)) for index, row in enumerate(rows)]
        ndcg = [row.get("validation", {}).get("macro_user_unconditional", {}).get("ndcg@10") for row in rows]
        losses = [row.get("train_loss") for row in rows]
        valid = [(epoch, value) for epoch, value in zip(epochs, ndcg) if value is not None]
        if valid:
            axes[0].plot([row[0] for row in valid], [row[1] for row in valid], marker="o", label=label)
        if all(value is not None for value in losses):
            axes[1].plot(epochs, losses, marker="o", label=label)
    axes[0].set_title("Validation ranking")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Macro-user NDCG@10")
    axes[1].set_title("Supervised training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean CE loss")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("SFT learning curves")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _policy_runs(artifacts: Path) -> list[Path]:
    runs: list[Path] = []
    for run in sorted((artifacts / "runs").iterdir()):
        if not run.is_dir():
            continue
        manifest = _read_json(run / "run_manifest.json") or {}
        if manifest.get("stage") not in {"GRPO", "REINFORCE"}:
            continue
        name = run.name
        keep = (
            "extended200" in name
            or "selected200" in name
            or name in {
                "grpo_q1_h1_seed17_20261012T182326Z", "grpo_q1_h5_seed17_diagnostic",
                "grpo_q2_seconds_h1_seed17_diagnostic", "grpo_q4_repeat01_seed17_diag",
                "grpo_q1_centered_corrected_seed17", "reinforce_q1_h1_seed17_diagnostic",
            }
        )
        if keep and (run / "metrics.jsonl").is_file():
            runs.append(run)
    return runs


def _policy_label(run: Path, manifest: dict[str, Any]) -> str:
    reward = str(manifest.get("reward", "RL"))
    seed = manifest.get("seed", "?")
    beta = manifest.get("beta")
    horizon = manifest.get("horizon", 1)
    beta_text = f" β={float(beta):g}" if beta is not None else ""
    return f"{reward}{beta_text}, H={horizon}, seed {seed}"


def _save_policy_learning_curves(artifacts: Path, output: Path) -> Path | None:
    import matplotlib.pyplot as plt

    runs = _policy_runs(artifacts)
    if not runs:
        return None
    fields = (
        ("sampled_raw_reward_mean", "Sampled raw reward"),
        ("kl_to_reference", "KL to SFT"),
        ("entropy", "Policy entropy"),
        ("clip_fraction", "PPO clip fraction"),
        ("group_reward_std_mean", "Within-group reward SD"),
        ("unique_actions_per_group_mean", "Unique actions per group"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for run in runs:
        manifest = _read_json(run / "run_manifest.json") or {}
        rows = _run_json_lines(run / "metrics.jsonl")
        if not rows:
            continue
        label = _policy_label(run, manifest)
        for ax, (field, title) in zip(axes.flat, fields):
            selected = [(row.get("update"), row.get(field)) for row in rows if row.get(field) is not None]
            if selected:
                ax.plot([row[0] for row in selected], [row[1] for row in selected], linewidth=1.25, label=label)
            ax.set_title(title)
            ax.set_xlabel("Update")
            ax.grid(alpha=0.2)
    for ax in axes.flat:
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle("Policy optimization diagnostics")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_reward_pareto(artifacts: Path, output: Path) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs = _policy_runs(artifacts)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    seen = [False, False]
    seed_markers = {17: "o", 42: "s", 73: "^"}
    beta_colors = {0.0: "#1f77b4", 0.1: "#ff7f0e", 1.0: "#2ca02c"}
    for run in runs:
        validation = _read_json(run / "validation_metrics.json")
        panel = _read_json(run / "policy_panel.json")
        manifest = _read_json(run / "run_manifest.json") or {}
        if not validation or not panel:
            continue
        policy_name = panel.get("policy_label")
        metrics = panel.get("metrics", {}).get(policy_name, {})
        baseline = panel.get("metrics", {}).get("SFT", {})
        paired = _read_json(run / "paired_validation.json") or {}
        ndcg_ci = paired.get("paired_user_delta_ndcg10", {})
        panel_ci = panel.get("paired_user_bootstrap_vs_sft", {}).get(policy_name, {})
        if not metrics:
            continue
        ndcg = validation.get("policy", {}).get("macro_user_unconditional", {}).get("ndcg@10")
        if ndcg is None:
            continue
        reward = str(manifest.get("reward", "Q1"))
        if reward not in {"Q1", "Q2"}:
            continue
        reward_head = str(panel.get("reward_head", manifest.get("reward_head", "ratio")))
        axis_index = 1 if reward_head == "seconds" else 0
        fit = metrics.get("v_fit")
        audit = metrics.get("v_audit")
        baseline_fit = baseline.get("v_fit")
        baseline_audit = baseline.get("v_audit")
        seed = int(manifest.get("seed", -1))
        marker = seed_markers.get(seed, "D")
        beta_raw = manifest.get("beta")
        beta = float(beta_raw) if beta_raw is not None else None
        color = beta_colors.get(beta, "#777777")
        xerr = None
        if ndcg_ci:
            delta = float(ndcg_ci.get("mean_difference", 0.0))
            low = float(ndcg_ci.get("ci_95_low", delta))
            high = float(ndcg_ci.get("ci_95_high", delta))
            xerr = [[max(0.0, delta - low)], [max(0.0, high - delta)]]
        if fit is not None and baseline_fit is not None:
            fit_ci = panel_ci.get("v_fit_difference_vs_sft", {})
            yerr = None
            if fit_ci:
                mean = float(fit_ci.get("mean_difference", 0.0))
                yerr = [[max(0.0, mean - float(fit_ci.get("ci_95_low", mean)))],
                        [max(0.0, float(fit_ci.get("ci_95_high", mean)) - mean)]]
            axes[axis_index].errorbar(
                ndcg, fit - baseline_fit, xerr=xerr, yerr=yerr, fmt=marker,
                color=color, markerfacecolor=color, markeredgecolor=color,
                linestyle="none", capsize=2,
            )
            seen[axis_index] = True
        if audit is not None and baseline_audit is not None:
            audit_ci = panel_ci.get("v_audit_difference_vs_sft", {})
            yerr = None
            if audit_ci:
                mean = float(audit_ci.get("mean_difference", 0.0))
                yerr = [[max(0.0, mean - float(audit_ci.get("ci_95_low", mean)))],
                        [max(0.0, float(audit_ci.get("ci_95_high", mean)) - mean)]]
            axes[axis_index].errorbar(
                ndcg, audit - baseline_audit, xerr=xerr, yerr=yerr, fmt=marker,
                color=color, markerfacecolor="none", markeredgecolor=color,
                linestyle="none", capsize=2,
            )
            seen[axis_index] = True
    axes[0].set_title("Q1: predicted completion ratio")
    axes[0].set_ylabel("Δ predicted reward vs SFT")
    axes[1].set_title("Q2: predicted listening seconds")
    axes[1].set_ylabel("Δ predicted seconds vs SFT")
    for index, ax in enumerate(axes):
        ax.set_xlabel("Validation macro-user NDCG@10")
        ax.grid(alpha=0.2)
        if not seen[index]:
            ax.text(0.5, 0.5, "No eligible policy run", ha="center", va="center", transform=ax.transAxes)
    legend_handles = [
        *(Line2D([0], [0], color="0.35", marker=marker, linestyle="none", label=f"seed {seed}")
          for seed, marker in seed_markers.items()),
        *(Line2D([0], [0], color=color, marker="o", linestyle="none", label=f"β={beta:g}")
          for beta, color in beta_colors.items()),
        Line2D([0], [0], color="0.35", marker="o", markerfacecolor="0.35", linestyle="none", label="RM_fit"),
        Line2D([0], [0], color="0.35", marker="o", markerfacecolor="none", linestyle="none", label="RM_audit"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=8, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.93))
    fig.text(0.5, 0.01, "Q3 disabled: the audit like-head did not pass its predictive gate.",
             ha="center", fontsize=8)
    fig.suptitle("Validation reward–relevance frontier; points retain failed guardrail settings")
    fig.tight_layout(rect=(0, 0.05, 1, 0.86))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_horizon_diagnostics(artifacts: Path, output: Path) -> Path | None:
    import matplotlib.pyplot as plt

    data = _read_json(artifacts / "horizon_diagnostics_seed17.json")
    if not data or not data.get("records"):
        return None
    selected = [row for row in data["records"] if row.get("model") == "GRPO_H5"]
    if not selected:
        selected = data["records"]
    modes = sorted({str(row.get("mode", "unknown")) for row in selected})
    fields = (
        ("rm_fit_predicted_ratio", "RM_fit predicted ratio"),
        ("rm_audit_predicted_ratio", "RM_audit predicted ratio"),
        ("duration_seconds", "Observed item duration (s)"),
        ("log1p_training_count", "log(1 + Train count)"),
        ("recent_repeat", "Recent history repeat rate"),
        ("repeat_within_slate", "Repeat within generated slate"),
        ("fit_unsupported", "RM_fit unsupported mass"),
        ("audit_unsupported", "RM_audit unsupported mass"),
    )
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    for ax, (field, title) in zip(axes.flat, fields):
        for mode in modes:
            points: list[tuple[int, float]] = []
            for step in sorted({int(row["step"]) for row in selected}):
                values = []
                for row in selected:
                    if row.get("mode") != mode or int(row["step"]) != step:
                        continue
                    if field == "fit_unsupported":
                        value = 1.0 - float(row["fit_supported"])
                    elif field == "audit_unsupported":
                        value = 1.0 - float(row["audit_supported"])
                    else:
                        value = row.get(field)
                    if value is not None:
                        values.append(float(value))
                if values:
                    points.append((step, sum(values) / len(values)))
            if points:
                ax.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=mode)
        ax.set_title(title)
        ax.set_xlabel("Rollout step")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("GRPO H=5 step-wise diagnostics by synthetic-state protocol")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_continuation_calibration(artifacts: Path, output: Path) -> Path | None:
    import matplotlib.pyplot as plt

    data = _read_json(artifacts / "continuation" / "continuation_seed17.json")
    if not data or not data.get("tasks"):
        return None
    task_names = ("continue_10m", "continue_30m", "continue_60m", "return_24h_to_7d")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, task_name in zip(axes.flat, task_names):
        task = data["tasks"].get(task_name, {})
        bounds: list[float] = []
        plotted = False
        for model_name, label in (("activity_only", "History/activity"), ("activity_plus_candidate", "Plus candidate")):
            rows = task.get(model_name, {}).get("calibration_deciles", [])
            if not rows:
                continue
            x = [row["mean_predicted"] for row in rows]
            y = [row["observed_rate"] for row in rows]
            ax.plot(x, y, marker="o", label=label)
            bounds.extend(x + y)
            plotted = True
        if plotted:
            hi = max(bounds) * 1.05 if max(bounds) > 0 else 1.0
            ax.plot([0, hi], [0, hi], linestyle="--", color="0.45", linewidth=1, label="ideal")
            ax.legend(fontsize=8, frameon=False)
        else:
            ax.text(0.5, 0.5, "No calibration data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(task_name.replace("_", " "))
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed rate")
        ax.grid(alpha=0.2)
    fig.suptitle("Continuation and return calibration; temporal holdout")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def generate_report_figures(
    artifacts_dir: str | Path = "artifacts",
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    artifacts = Path(artifacts_dir)
    output = Path(output_dir) if output_dir else artifacts / "figures"
    generators = {
        "rm_calibration": _save_rm_calibration(artifacts, output / "rm_calibration.png"),
        "sft_learning_curves": _save_sft_learning_curves(artifacts, output / "sft_learning_curves.png"),
        "policy_learning_curves": _save_policy_learning_curves(artifacts, output / "policy_learning_curves.png"),
        "reward_pareto": _save_reward_pareto(artifacts, output / "reward_pareto.png"),
        "horizon_stepwise": _save_horizon_diagnostics(artifacts, output / "horizon_stepwise.png"),
        "continuation_calibration": _save_continuation_calibration(artifacts, output / "continuation_calibration.png"),
    }
    return {name: str(path.resolve()) for name, path in generators.items() if path is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible figures for REPORT.md from saved validation artifacts")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(json.dumps(generate_report_figures(args.artifacts, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
