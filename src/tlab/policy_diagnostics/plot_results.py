"""Render reproducible panel_tools.2 diagnostic and policy figures from saved tables."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .protocol import ARTIFACTS, ROOT, record_operation, update_stage


FIGURES = ARTIFACTS / "figures"
BRANCH_ROOT = ARTIFACTS / "branches/sample_count_4_vs_16"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: Any, name: str, manifest: list[dict[str, Any]]) -> None:
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size})


def plot() -> dict[str, Any]:
    started = time.perf_counter()
    FIGURES.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, Any]] = []

    # B1: within/between decomposition, proxy and external audit.
    variance = _rows(ARTIFACTS / "diagnostics/variance_by_novelty_support.csv")
    panels = ["train", "development"]
    rewards = ["optimized_RM_proxy_raw", "audit_raw", "item_metadata_raw", "additive_raw"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    x = np.arange(len(rewards))
    width = 0.36
    for ax, panel in zip(axes, panels):
        for offset, weight in enumerate(("uniform_candidates", "reference_conditional")):
            subset = [r for r in variance if r["panel"] == panel and r["candidate_group"] == "all"
                      and r["candidate_weighting"] == weight and r["context_weighting"] == "uniform_context"
                      and r["reward"] in rewards]
            indexed = {r["reward"]: r for r in subset}
            vals = [float(indexed[name]["within_fraction"]) if name in indexed else 0.0 for name in rewards]
            ax.bar(x + (offset - 0.5) * width, vals, width, label=weight.replace("_", " "))
        ax.set_title(panel.replace("_", " "))
        ax.set_xticks(x, [r.replace("_raw", "").replace("_", "\n") for r in rewards], rotation=0, fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Within-context share of total variance")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("B1: variance decomposition (uniform context weighting)")
    _save(fig, "variance_decomposition.png", made)

    # B2: within-context proxy/audit alignment; separate rank/overlap measures.
    alignment = _rows(ARTIFACTS / "diagnostics/reward_variance_alignment_beta.csv")
    metrics = ["weighted_pearson", "weighted_spearman", "centered_sign_agreement", "top10_overlap"]
    panel_order = ["train", "development", "retrospective_final", "common_open_6655"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(panel_order))
    width = .19
    for j, metric in enumerate(metrics):
        vals = []
        for panel in panel_order:
            matches = [r for r in alignment if r.get("panel") == panel
                       and r.get("diagnostic") == "B2_centered_proxy_audit_alignment" and r.get("metric") == metric]
            vals.append(float(matches[0]["mean"]) if matches else np.nan)
        ax.bar(x + (j - 1.5) * width, vals, width, label=metric.replace("_", " "))
    ax.set_xticks(x, [p.replace("_", "\n") for p in panel_order])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean within-context alignment")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=.25)
    ax.set_title("B2: centered proxy–audit agreement on fixed support")
    _save(fig, "centered_proxy_audit_alignment.png", made)

    # B3: exact analytic gap along the existing training runs.
    gaps = _rows(ARTIFACTS / "diagnostics/analytic_gap_by_seed_checkpoint.csv")
    selected = [r for r in gaps if r["panel"] == "development" and r["endpoint"] == "last" and r["beta"] == "0.3"]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for objective, label, marker in (("sampled_loo", "sampled LOO", "o"),
                                      ("exact_expected_reward", "exact", "s"),
                                      ("analytic_q_distillation", "distillation", "^")):
        group = [r for r in selected if r["objective"] == objective and "no_feedback" not in r["method"] and "continuation" not in r["method"]]
        xs = [float(r["training_step"]) for r in group]
        ys = [float(r["mean_exact_gap"]) for r in group]
        if xs:
            ax.scatter(xs, ys, label=label, marker=marker, alpha=.8)
    branch_colors = {"sampled4": "#477da3", "sampled16": "#d28a35", "exact": "#60965b"}
    for arm in ("sampled4", "sampled16", "exact"):
        curve_files = sorted((BRANCH_ROOT / arm / "runs").glob("*/curve.jsonl"))
        curves = [[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                  for path in curve_files]
        if len(curves) != 3:
            continue
        by_step = {int(row["step"]): [] for row in curves[0]}
        for curve in curves:
            for row in curve:
                by_step.setdefault(int(row["step"]), []).append(float(row["dev"]["analytic_objective_gap"]))
        steps = sorted(step for step, values in by_step.items() if len(values) == 3)
        means = np.array([np.mean(by_step[step]) for step in steps])
        lower = np.array([np.min(by_step[step]) for step in steps])
        upper = np.array([np.max(by_step[step]) for step in steps])
        manifests = [json.loads(path.read_text(encoding="utf-8"))
                     for path in sorted((BRANCH_ROOT / arm / "runs").glob("*/run_manifest.json"))]
        median_minutes = float(np.median([m["elapsed_training_and_evaluation_seconds"] for m in manifests])) / 60
        label = f"panel_tools.2 {arm} (mean; min–max; median {median_minutes:.1f} min/seed)"
        ax.plot(steps, means, color=branch_colors[arm], label=label, linewidth=2)
        ax.fill_between(steps, lower, upper, color=branch_colors[arm], alpha=.12)
    ax.set_xlabel("Training updates")
    ax.set_ylabel("Mean dev analytic objective gap")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    ax.set_title("B3: development analytic gap by update; panel_tools.2 branch trajectory")
    _save(fig, "analytic_gap_by_updates.png", made)

    # B4: sampled parameter-gradient estimator error and direction agreement.
    gradients = _rows(ARTIFACTS / "diagnostics/gradient_parameter_noise.csv")
    sample_levels = [4, 16, 64]
    groups = [[r for r in gradients if int(r["samples_per_context"]) == sample] for sample in sample_levels]
    means = [np.mean([float(r["relative_error_of_mean_sampled"]) for r in group]) for group in groups]
    sds = [np.std([float(r["relative_error_of_mean_sampled"]) for r in group], ddof=1) for group in groups]
    cosines = [np.mean([float(r["cosine_of_mean_sampled_to_exact"]) for r in group]) for group in groups]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.errorbar(sample_levels, means, yerr=sds, marker="o", capsize=4, label="relative error (mean ± batch SD)")
    ax.plot(sample_levels, cosines, marker="s", label="cosine with exact gradient")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sample_levels, [str(x) for x in sample_levels])
    ax.set_xlabel("Reward samples per context")
    ax.set_ylabel("Error / cosine")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    ax.set_title("B4: parameter-gradient sampling noise (8 fixed minibatches)")
    _save(fig, "gradient_error_by_sample_count.png", made)

    # B5: representability of the frozen proxy correction on held-out contexts.
    expressivity = _rows(ARTIFACTS / "diagnostics/expressivity_cross_user.csv")
    dev = [r for r in expressivity if r["panel"] == "development"]
    keep = [r for r in dev if r["model"] in {"current_analytic_q_distillation_residual_MLP", "item_only_linear", "strict_additive_linear"}]
    labels = [r["model"].replace("current_analytic_q_distillation_residual_MLP", "current head")
              .replace("item_only_linear", "item-only").replace("strict_additive_linear", "strict additive") for r in keep]
    values = [float(r["weighted_r2"]) if r.get("weighted_r2") else float(r["r2"]) for r in keep]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(np.arange(len(keep)), values, color=["#3a78a1", "#8c9b56", "#b18a56"][:len(keep)])
    ax.set_xticks(np.arange(len(keep)), labels, rotation=10)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_ylabel("Held-out dev R² of proxy correction")
    ax.set_title("B5: shared candidate correction representability")
    ax.grid(axis="y", alpha=.25)
    _save(fig, "heldout_proxy_approximation.png", made)

    # B1 stratification: novelty × support within-context proxy variance.
    strata = _rows(ARTIFACTS / "diagnostics/variance_by_novelty_support.csv")
    groups = ["new__unsupported", "new__rare_1_9", "new__rare_10_99",
              "repeat__unsupported", "repeat__rare_1_9", "repeat__rare_10_99"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    panel = "development"
    vals = []
    labels = []
    for group in groups:
        match = [r for r in strata if r["panel"] == panel and r["reward"] == "optimized_RM_proxy_raw"
                 and r["candidate_group"] == group and r["candidate_weighting"] == "reference_conditional"
                 and r["context_weighting"] == "uniform_context"]
        if match:
            vals.append(float(match[0]["within_variance"]))
            labels.append(group.replace("__", "\n"))
    ax.bar(np.arange(len(vals)), vals, color=["#5185a4" if i < 3 else "#a87855" for i in range(len(vals))])
    ax.set_xticks(np.arange(len(vals)), labels)
    ax.set_ylabel("Within-context raw proxy variance")
    ax.set_title("B1: proxy variation by novelty × RM-fit support (dev)")
    ax.grid(axis="y", alpha=.25)
    _save(fig, "support_by_novelty_variance.png", made)

    # A4 and branch: overall dev method contrasts with user-bootstrap intervals.
    contrasts = _rows(ARTIFACTS / "evaluation/seed_contrasts.csv")
    focus = [r for r in contrasts if r["panel"] == "development" and any(key in r["contrast"] for key in
             ("dense_sft - strong_gpt", "last_target_sft - strong_gpt", "sampled_loo_b0.3_full_last_seed_mean",
              "exact_expected_reward_b0.3_full_last_seed_mean", "analytic_q_distillation_b0.3_full_last_seed_mean",
              "exact_expected_reward_b0.3_no_feedback_last_seed_mean"))]
    focus = focus[:12]
    fig, ax = plt.subplots(figsize=(10, max(4, .38 * len(focus))))
    labels = [r["contrast"].replace("_seed_mean", "").replace("_", " ") for r in focus]
    means = np.array([float(r["mean_difference"]) for r in focus])
    lo = np.array([float(r["ci_lower"]) for r in focus])
    hi = np.array([float(r["ci_upper"]) for r in focus])
    y = np.arange(len(focus))
    ax.errorbar(means, y, xerr=np.vstack((means - lo, hi - means)), fmt="o", capsize=3)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("Development NDCG@10 difference versus strong")
    ax.set_title("A4: principal fixed-method contrasts (paired user bootstrap 95% CI)")
    ax.grid(axis="x", alpha=.25)
    _save(fig, "overall_method_user_ci.png", made)

    branch_quality = BRANCH_ROOT / "branch_policy_quality.csv"
    branch_support = BRANCH_ROOT / "branch_support_metrics.csv"
    if branch_quality.exists() and branch_support.exists():
        qrows = _rows(branch_quality)
        srows = _rows(branch_support)
        quality = {(r["panel"], r["method"], r["endpoint"], r["seed"]): float(r["ndcg10"])
                   for r in qrows if r["panel"] == "development" and r["endpoint"] == "last"}
        support = [r for r in srows if r["panel"] == "development" and r["effective_endpoint"] == "last"]
        fig, (ax_quality, ax_reward) = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True)
        colors = {"sampled4": "#477da3", "sampled16": "#d28a35", "exact": "#60965b"}
        for row in support:
            key = ("development", row["method"], "last", row["seed"])
            if key not in quality:
                continue
            movement = float(row["kl_policy_to_reference_mass_weighted"])
            label = row["method"] if row["seed"] == "17" else None
            ax_quality.scatter(movement, quality[key], c=colors[row["method"]], s=55, alpha=.85, label=label)
            ax_reward.scatter(movement, float(row["support_weighted_reward_expectation"]),
                              c=colors[row["method"]], s=55, alpha=.85)
        ax_quality.set_xlabel("Mass-weighted KL(q || p) on support")
        ax_reward.set_xlabel("Mass-weighted KL(q || p) on support")
        ax_quality.set_ylabel("Development user-macro NDCG@10")
        ax_reward.set_ylabel("Mass-weighted normalized proxy reward")
        ax_quality.set_title("Relevance")
        ax_reward.set_title("Proxy reward")
        ax_quality.grid(alpha=.25)
        ax_reward.grid(alpha=.25)
        ax_quality.legend(frameon=False)
        fig.suptitle("Registered branch: quality and reward versus policy movement")
        _save(fig, "branch_quality_reward_vs_kl.png", made)

        bcontrasts = [r for r in _rows(BRANCH_ROOT / "branch_contrasts.csv")
                      if r["panel"] == "development" and r["endpoint"] == "last"]
        fig, ax = plt.subplots(figsize=(8.5, max(3.5, .45 * len(bcontrasts))))
        labels = [r["contrast"] for r in bcontrasts]
        means = np.array([float(r["mean_difference"]) for r in bcontrasts])
        lo = np.array([float(r["ci_lower"]) for r in bcontrasts])
        hi = np.array([float(r["ci_upper"]) for r in bcontrasts])
        y = np.arange(len(labels))
        ax.errorbar(means, y, xerr=np.vstack((means - lo, hi - means)), fmt="o", capsize=3)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Development NDCG@10 difference; seed mean, paired user bootstrap 95% CI")
        ax.set_title("Matched sample-count branch contrasts")
        ax.grid(axis="x", alpha=.25)
        _save(fig, "branch_contrasts_user_ci.png", made)

    result = {
        "schema_version": 1, "status": "complete", "figure_count": len(made),
        "figures": made, "runtime_seconds": round(time.perf_counter() - started, 3),
        "source_tables_only": True,
    }
    (FIGURES / "figures_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record_operation(
        "policy_diagnostics_reproducible_figures",
        time.perf_counter() - started,
        bucket="final_evaluation_and_report",
        category="reporting_visualization",
        evidence=[entry["path"] for entry in made] + ["artifacts/policy_diagnostics/figures/figures_manifest.json"],
    )
    update_stage("final_evaluation", "figures_complete", [entry["path"] for entry in made])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(plot(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
