"""Clean-result body figures for task #608 (analyzer pass).

Regenerates the three analysis figures with reader-facing labels (plain-English
persona names, no issue numbers in legends) and adds a mix-gap forest plot.
Reads only the committed aggregate `eval_results/issue_608/analyze_summary_608.json`.

Run from the issue-608 worktree root:
    uv run python scripts/issue608_body_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

SUMMARY = Path("eval_results/issue_608/analyze_summary_608.json")
FIG_DIR = "figures/"

SOURCES = [
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
]
SOURCE_LABELS = {
    "villain": "Villain",
    "comedian": "Comedian",
    "assistant": "Assistant",
    "qwen_default": "Qwen default",
    "software_engineer": "Software engineer",
    "kindergarten_teacher": "Kindergarten teacher",
}
ARM_LABELS = {
    "contrastive_fresh_eval": "Contrastive mix (re-evaluated)",
    "posonly_dose": "Positive-only, dose-matched",
    "posonly_epoch": "Positive-only, matched epochs",
}
TOP_BAND_RATE = 0.95
H2_LIFT_THRESHOLD = 0.05


def main() -> None:
    set_paper_style("blog")
    summary = json.load(open(SUMMARY))
    per_source = summary["per_source"]
    pal = paper_palette_blog(6)
    arm_colors = {
        "contrastive_fresh_eval": pal[0],
        "posonly_dose": pal[1],
        "posonly_epoch": pal[2],
    }
    xs = np.arange(len(SOURCES))
    tick_labels = [SOURCE_LABELS[s] for s in SOURCES]

    # ------------------------------------------------------------------
    # 1. Hero: per-source self-implant delta with CIs + censor band.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    offsets = {"contrastive_fresh_eval": -0.22, "posonly_dose": 0.0, "posonly_epoch": 0.22}
    for arm, off in offsets.items():
        ys = [per_source[s]["delta_self"][arm] for s in SOURCES]
        los = [per_source[s]["delta_self_ci95"][arm][0] for s in SOURCES]
        his = [per_source[s]["delta_self_ci95"][arm][1] for s in SOURCES]
        yerr = np.array(
            [
                [max(0.0, y - lo) for y, lo in zip(ys, los, strict=True)],
                [max(0.0, hi - y) for y, hi in zip(ys, his, strict=True)],
            ]
        )
        ax.errorbar(
            xs + off,
            ys,
            yerr=yerr,
            fmt="o",
            capsize=3,
            markersize=6,
            color=arm_colors[arm],
            label=ARM_LABELS[arm],
        )
    for i, s in enumerate(SOURCES):
        band_lo = TOP_BAND_RATE - per_source[s]["fresh_base_own_rate"]
        ax.fill_between([i - 0.4, i + 0.4], band_lo, 1.0, color="grey", alpha=0.12, linewidth=0)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.set_ylabel("Self-implant delta\n(own-panel agreement rate, trained − base)")
    ax.set_title(
        "Sycophancy self-implant by training mix\n"
        "(95% claim-bootstrap CI; grey = top-band censor zone, own-rate ≥ 0.95)"
    )
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "issue_608/self_implant_dumbbell", dir=FIG_DIR)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Bystander lift by arm: mean bars + per-bystander raw dots.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    w = 0.26
    rng = np.random.default_rng(42)
    for j, arm in enumerate(offsets):
        blk = summary["h2"]["per_arm"][arm]["per_source"]
        means = [blk[s]["mean_bystander_delta_21_registered"] for s in SOURCES]
        ax.bar(
            xs + (j - 1) * w,
            means,
            width=w,
            color=arm_colors[arm],
            alpha=0.75,
            label=ARM_LABELS[arm],
        )
        # Raw per-bystander deltas (registered 21-bystander denominator).
        for i, s in enumerate(SOURCES):
            excl = set(blk[s]["excluded_trained_negatives"])
            pts = [v for k, v in blk[s]["per_bystander_delta"].items() if k not in excl]
            jit = rng.uniform(-w * 0.32, w * 0.32, size=len(pts))
            ax.scatter(
                np.full(len(pts), i + (j - 1) * w) + jit,
                pts,
                s=7,
                color="black",
                alpha=0.35,
                linewidths=0,
                zorder=3,
            )
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(
        H2_LIFT_THRESHOLD,
        color="darkorange",
        linestyle=":",
        linewidth=1.0,
        label="broad-lift threshold (+0.05)",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.set_ylabel("Mean bystander agreement delta\n(trained − base, 21 bystanders)")
    ax.set_title(
        "Bystander sycophancy lift by training mix\n"
        "(bars = mean over 21 registered bystanders; dots = individual bystander personas)"
    )
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    savefig_paper(fig, "issue_608/bystander_lift_by_arm", dir=FIG_DIR)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Mix gap forest plot: g = contrastive - positive-only (dose).
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ys = np.arange(len(SOURCES))[::-1]
    gs = [per_source[s]["g"] for s in SOURCES]
    glo = [per_source[s]["g_ci95"][0] for s in SOURCES]
    ghi = [per_source[s]["g_ci95"][1] for s in SOURCES]
    xerr = np.array(
        [
            [g - lo for g, lo in zip(gs, glo, strict=True)],
            [hi - g for g, hi in zip(gs, ghi, strict=True)],
        ]
    )
    sentinel = [not per_source[s]["top_band_contrastive_fresh"] for s in SOURCES]
    colors = [pal[3] if sent else pal[0] for sent in sentinel]
    ax.axvspan(-0.05, 0.05, color="grey", alpha=0.10, linewidth=0)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    for y, g, xe_lo, xe_hi, c in zip(ys, gs, xerr[0], xerr[1], colors, strict=True):
        ax.errorbar([g], [y], xerr=[[xe_lo], [xe_hi]], fmt="o", capsize=3, markersize=7, color=c)
    ax.set_yticks(ys)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel(
        "Own-rate gap: contrastive − positive-only dose-matched\n"
        "(positive favors the contrastive mix; 95% claim-bootstrap CI)"
    )
    ax.set_title(
        "Per-source installation gap between mixes\n"
        "(red = source below the 0.95 censor band in the contrastive arm; grey = ±0.05 band)"
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_608/mix_gap_forest", dir=FIG_DIR)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Trajectory small-multiples (descriptive).
    # ------------------------------------------------------------------
    steps_by_arm = {"posonly_epoch": [13, 26, 39], "posonly_dose": [44, 88, 132]}
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2))
    for ax_, source in zip(axes.flat, SOURCES, strict=True):
        for arm in ("posonly_epoch", "posonly_dose"):
            pts = summary["trajectory_descriptive"][arm][source]
            keys = ["epoch_1", "epoch_2", "epoch_3_endpoint"]
            have = [k for k in keys if k in pts]
            ys_ = [pts[k]["own_rate"] for k in have]
            ses = [pts[k]["claim_clustered_se"] for k in have]
            xs_steps = [steps_by_arm[arm][keys.index(k)] for k in have]
            ax_.errorbar(
                xs_steps,
                ys_,
                yerr=ses,
                marker="o",
                capsize=3,
                color=arm_colors[arm],
                label=ARM_LABELS[arm],
            )
        ax_.axhline(
            per_source[source]["fresh_base_own_rate"],
            color="grey",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
        ax_.axhline(TOP_BAND_RATE, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax_.set_title(SOURCE_LABELS[source], fontsize=11)
        ax_.set_xlabel("optimizer steps")
        ax_.set_ylabel("own-panel agreement rate")
        ax_.set_ylim(-0.02, 1.02)
    axes.flat[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Positive-only own-rate trajectory at epoch checkpoints "
        "(dashed = untrained base rate, dotted = 0.95 censor band)",
        fontsize=12,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_608/own_rate_trajectory", dir=FIG_DIR)
    plt.close(fig)

    print("figures written to figures/issue_608/")


if __name__ == "__main__":
    main()
