"""Regenerate issue #503 clean-result figures (round-2 revisions).

Three figures:
  1. hero_cosine_vs_advbench.png — scatter (15 cells, color by selector,
     D4_format and D3 both highlighted as elevated; per-cell n in caption).
  2. predictor_spread_by_selector.png — strip plot of cosine by selector.
  3. d1_vs_d3_anti_calibration.png — NEW. Bar chart of cosine + rate for
     D1 and D3, showing nearly identical predictor mean but 4× rate gap.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-503")
FIG_DIR = Path("/home/thomasjiralerspong/explore-persona-space") / "figures"


def load_cells():
    """Return list of dicts: selector, seed, cosine, k, n, rate."""
    cells = []
    pred_dir = ROOT / "eval_results" / "issue503" / "predictors"
    ce_dir = ROOT / "eval_results" / "issue503" / "cross_eval"
    for pred_file in sorted(pred_dir.glob("*__D_advbench__seed*__L25.json")):
        name = pred_file.name
        m = re.match(r"(D[0-4]_[a-z_]+)__D_advbench__seed(\d+)__L25\.json", name)
        if not m:
            continue
        selector, seed = m.group(1), int(m.group(2))
        with pred_file.open() as f:
            pred = json.load(f)
        cos = pred["cosine"]["mean"]
        verdict_file = ce_dir / f"{selector}_seed{seed}" / "D_advbench.verdict.json"
        with verdict_file.open() as f:
            v = json.load(f)
        cells.append(
            {
                "selector": selector,
                "seed": seed,
                "cosine": cos,
                "k": v["k"],
                "n": v["n"],
                "rate": v["rate"],
            }
        )
    return cells


SELECTORS = [
    ("D0_random", "Random (baseline)"),
    ("D1_representation", "Representation"),
    ("D2_gradient", "Gradient"),
    ("D3_cosine", "Cosine (this work)"),
    ("D4_format", "Format (GSM8K)"),
]
SELECTOR_LABEL = dict(SELECTORS)


def hero_figure(cells, out_dir):
    """Scatter: cosine vs harmful rate, color by selector."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    colors = {
        "D0_random": paper_palette_role("baseline"),
        "D1_representation": paper_palette_role("control"),
        "D2_gradient": paper_palette_role("accent"),
        "D3_cosine": paper_palette_role("primary"),
        "D4_format": paper_palette_role("neutral"),
    }
    markers = {
        "D0_random": "o",
        "D1_representation": "s",
        "D2_gradient": "^",
        "D3_cosine": "D",
        "D4_format": "v",
    }

    for sel, label in SELECTORS:
        sel_cells = [c for c in cells if c["selector"] == sel]
        xs = [c["cosine"] for c in sel_cells]
        ys = [c["rate"] * 100 for c in sel_cells]
        ax.scatter(
            xs,
            ys,
            s=80,
            color=colors[sel],
            marker=markers[sel],
            label=label,
            edgecolors="white",
            linewidth=0.7,
            zorder=3,
        )

    cos_min = min(c["cosine"] for c in cells)
    cos_max = max(c["cosine"] for c in cells)
    ax.axvspan(cos_min, cos_max, alpha=0.08, color="grey", zorder=1)

    ax.set_xlim(0.920, 0.970)
    ax.set_ylim(0, 9)
    ax.set_xlabel("Base-model in-context cosine predictor (K=8 demos, L25, mean over 2 draws)")
    ax.set_ylabel("AdvBench harmful rate (%)")
    ax.text(
        0.96,
        0.04,
        "lower = safer",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
        color="dimgrey",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=1)

    ax.text(
        0.98,
        0.96,
        "Spearman ρ = 0.19, p = 0.49, n = 15\n"
        "95% bootstrap CI: [-0.41, +0.71]\n"
        "permutation null: 78th pctile",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgrey", linewidth=0.6
        ),
    )

    ax.text(
        0.5,
        -0.18,
        f"all 15 cells: cosine in [{cos_min:.3f}, {cos_max:.3f}], range = {cos_max - cos_min:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        style="italic",
        color="dimgrey",
    )

    set_title_subtitle(
        ax,
        "Cosine predictor does not separate Bucket-D selectors at this scale",
        "15 cells (5 selectors x 3 seeds) on AdvBench. D3 and D4 both lift above the baseline cluster.",
        source="task #503  —  Qwen-2.5-7B-Instruct, K=100 SFT rows, AdvBench judged n varies 452-515",
    )

    savefig_paper(fig, "issue_503/hero_cosine_vs_advbench", dir=str(out_dir))
    plt.close(fig)


def predictor_spread_figure(cells, out_dir):
    """Strip plot of cosine by selector with per-selector mean bar."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    colors = {
        "D0_random": paper_palette_role("baseline"),
        "D1_representation": paper_palette_role("control"),
        "D2_gradient": paper_palette_role("accent"),
        "D3_cosine": paper_palette_role("primary"),
        "D4_format": paper_palette_role("neutral"),
    }
    markers = {
        "D0_random": "o",
        "D1_representation": "s",
        "D2_gradient": "^",
        "D3_cosine": "D",
        "D4_format": "v",
    }

    for i, (sel, label) in enumerate(SELECTORS):
        sel_cells = [c for c in cells if c["selector"] == sel]
        xs = [i + (j - 1) * 0.04 for j in range(len(sel_cells))]
        ys = [c["cosine"] for c in sel_cells]
        ax.scatter(
            xs,
            ys,
            s=70,
            color=colors[sel],
            marker=markers[sel],
            edgecolors="white",
            linewidth=0.7,
            zorder=3,
        )
        mean_y = np.mean(ys)
        ax.hlines(mean_y, i - 0.18, i + 0.18, colors=colors[sel], linewidth=2, zorder=2)

    cos_min = min(c["cosine"] for c in cells)
    cos_max = max(c["cosine"] for c in cells)
    ax.axhline(cos_min, linestyle=":", color="grey", linewidth=0.6, alpha=0.7)
    ax.axhline(cos_max, linestyle=":", color="grey", linewidth=0.6, alpha=0.7)
    ax.text(
        4.4,
        (cos_min + cos_max) / 2,
        f"range = {cos_max - cos_min:.3f}",
        ha="left",
        va="center",
        fontsize=9,
        style="italic",
        color="dimgrey",
    )

    ax.set_xticks(range(len(SELECTORS)))
    ax.set_xticklabels([label for _, label in SELECTORS], rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Base-model in-context cosine predictor")
    ax.set_ylim(0.925, 0.97)
    ax.set_xlim(-0.5, 4.6)

    set_title_subtitle(
        ax,
        "The predictor has almost no spread across selectors to correlate with",
        "Each marker is one seed; the bar is the per-selector mean. Range 0.026 sits inside selector noise.",
        source="task #503  —  Bucket-D only (15 cells)",
    )

    savefig_paper(fig, "issue_503/predictor_spread_by_selector", dir=str(out_dir))
    plt.close(fig)


def d1_vs_d3_figure(cells, out_dir):
    """NEW: dumbbell/scatter showing D1 vs D3 same predictor, different outcome.

    Plot cosine on x-axis, rate on y-axis, zoom to the D1+D3 region. Per-seed
    points with selector means as larger annotated markers.
    """
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    d1 = [c for c in cells if c["selector"] == "D1_representation"]
    d3 = [c for c in cells if c["selector"] == "D3_cosine"]

    color_d1 = paper_palette_role("control")
    color_d3 = paper_palette_role("primary")

    # Per-seed points
    ax.scatter(
        [c["cosine"] for c in d1],
        [c["rate"] * 100 for c in d1],
        s=90,
        color=color_d1,
        marker="s",
        edgecolors="white",
        linewidth=0.7,
        zorder=3,
        label="Representation (D1) — 3 seeds",
        alpha=0.7,
    )
    ax.scatter(
        [c["cosine"] for c in d3],
        [c["rate"] * 100 for c in d3],
        s=90,
        color=color_d3,
        marker="D",
        edgecolors="white",
        linewidth=0.7,
        zorder=3,
        label="Cosine, this work (D3) — 3 seeds",
        alpha=0.7,
    )

    # Mean markers (larger, with annotation)
    d1_cos_mean = float(np.mean([c["cosine"] for c in d1]))
    d3_cos_mean = float(np.mean([c["cosine"] for c in d3]))
    d1_rate_mean = float(np.mean([c["rate"] for c in d1])) * 100
    d3_rate_mean = float(np.mean([c["rate"] for c in d3])) * 100

    ax.scatter(
        [d1_cos_mean],
        [d1_rate_mean],
        s=300,
        color=color_d1,
        marker="s",
        edgecolors="black",
        linewidth=1.4,
        zorder=5,
    )
    ax.scatter(
        [d3_cos_mean],
        [d3_rate_mean],
        s=300,
        color=color_d3,
        marker="D",
        edgecolors="black",
        linewidth=1.4,
        zorder=5,
    )

    # Connecting line — same predictor band
    ax.plot(
        [d1_cos_mean, d3_cos_mean],
        [d1_rate_mean, d3_rate_mean],
        linestyle="--",
        color="dimgrey",
        linewidth=1.2,
        alpha=0.7,
        zorder=2,
    )

    # Cosine annotation showing the tiny gap (placed inside chart area)
    ax.annotate(
        f"cosine = {d1_cos_mean:.4f}\nrate = {d1_rate_mean:.2f}%",
        xy=(d1_cos_mean, d1_rate_mean),
        xytext=(d1_cos_mean - 0.0035, d1_rate_mean + 1.8),
        fontsize=9,
        color="black",
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color="dimgrey", linewidth=0.6),
    )
    ax.annotate(
        f"cosine = {d3_cos_mean:.4f}\nrate = {d3_rate_mean:.2f}%",
        xy=(d3_cos_mean, d3_rate_mean),
        xytext=(d3_cos_mean + 0.0008, d3_rate_mean + 0.4),
        fontsize=9,
        color="black",
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color="dimgrey", linewidth=0.6),
    )

    ax.text(
        0.97,
        0.04,
        f"predictor mean gap: {abs(d3_cos_mean - d1_cos_mean):.4f}    rate gap: {d3_rate_mean / d1_rate_mean:.1f}x",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        style="italic",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgrey", linewidth=0.6
        ),
    )

    ax.set_xlim(0.945, 0.963)
    ax.set_ylim(0, 9)
    ax.set_xlabel("Base-model in-context cosine predictor (K=8 demos, L25, mean over 2 draws)")
    ax.set_ylabel("AdvBench harmful rate (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    set_title_subtitle(
        ax,
        "Same predictor, different outcome: D1 and D3 are 0.0004 apart on cosine but D3 is 4x as harmful",
        "The cleanest visible anti-calibration evidence at this slice — predictor cannot distinguish these cells.",
        source="task #503  —  Representation (D1) vs Cosine-this-work (D3), 3 seeds each",
    )

    savefig_paper(fig, "issue_503/d1_vs_d3_anti_calibration", dir=str(out_dir))
    plt.close(fig)


def main():
    cells = load_cells()
    print(f"Loaded {len(cells)} cells")

    out_dir = FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    hero_figure(cells, out_dir)
    predictor_spread_figure(cells, out_dir)
    d1_vs_d3_figure(cells, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
