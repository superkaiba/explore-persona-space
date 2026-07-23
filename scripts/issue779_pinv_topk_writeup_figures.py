"""Writeup figures for the #779 persona-vector pre-image top-context inspection.

Pure re-plot: every number is read from the committed inline free-analysis
artifact eval_results/issue_779/pinv_topk_contexts/pinv_topk_contexts.json
(commit 117626de86) — no recomputation, no model access.

Outputs figures/issue_779/pinv_topk_*.{png,pdf,meta.json} for
docs/writeups/2026-07-22-persona-vector-preimage-top-contexts.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/matplotlib imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT = PROJECT_ROOT / "eval_results/issue_779/pinv_topk_contexts/pinv_topk_contexts.json"

TRAITS = ["evil", "sycophancy", "hallucination"]
TRAIT_LABELS = {
    "evil": "evil (L14)",
    "sycophancy": "sycophancy (L26)",
    "hallucination": "hallucination (L17)",
}
DIRECTIONS = ["r_B_raw", "w_tr", "w_pinv_kstar", "w_pinv_full"]
DIRECTION_LABELS = {
    "r_B_raw": "raw persona vector",
    "w_tr": "transpose map-through",
    "w_pinv_kstar": "pre-image (rank-truncated)",
    "w_pinv_full": "pre-image (full rank)",
}
# One fixed direction -> color mapping reused across every figure/panel.
DIRECTION_COLORS = dict(zip(DIRECTIONS, paper_palette(len(DIRECTIONS))))

# Coarse theme buckets for the LMSYS top-10 composition figure (merged so the
# stacked bars stay <= 6 segments; prose keeps the raw buckets).
THEME_ORDER = [
    "roleplay-creative",
    "math",
    "code",
    "factual-explain",
    "advice / rewrite",
    "other",
]
THEME_MERGE = {
    "roleplay-creative": "roleplay-creative",
    "math": "math",
    "code": "code",
    "factual-explain": "factual-explain",
    "advice-personal": "advice / rewrite",
    "summarize-rewrite": "advice / rewrite",
    "other": "other",
}
THEME_COLORS = dict(zip(THEME_ORDER, paper_palette(len(THEME_ORDER))))


def cond_label(cond_id: str) -> str:
    if cond_id.startswith("sys"):
        return f"system {cond_id[3:]}"
    if cond_id.startswith("shot"):
        return f"{cond_id[4:]}-shot"
    return cond_id


def fig_eval_spearman(data: dict) -> None:
    """Grouped bars: eval-grid Spearman(projection, judge score) per direction x trait."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    n_dir = len(DIRECTIONS)
    width = 0.19
    xs = np.arange(len(TRAITS))
    for j, dirk in enumerate(DIRECTIONS):
        vals = [data["traits"][t]["eval_grid"][dirk]["spearman_proj_vs_judgescore"] for t in TRAITS]
        pos = xs + (j - (n_dir - 1) / 2) * width
        bars = ax.bar(
            pos,
            vals,
            width=width,
            color=DIRECTION_COLORS[dirk],
            label=DIRECTION_LABELS[dirk],
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + (0.02 if v >= 0 else -0.02),
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
            )
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([TRAIT_LABELS[t] for t in TRAITS])
    ax.set_ylabel("Spearman rho, projection\nvs judged trait score")
    add_direction_arrow(ax, axis="y", direction="up")
    ax.set_ylim(-0.5, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    set_title_subtitle(
        ax,
        "Rank correlation between direction projection and judged trait expression",
        "Crafted eval grid: 260 contexts per trait (13 conditions x 20 questions), "
        "one on-policy rollout each",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_779/pinv_topk_eval_spearman", dir="figures/")
    plt.close(fig)


def fig_percond_scatter(data: dict) -> None:
    """Per-condition mean projection vs condition mean judge score, per trait."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, trait in zip(axes, TRAITS):
        eg = data["traits"][trait]["eval_grid"]
        conds = list(eg["cond_mean_judge_score"].keys())
        scores = np.array([eg["cond_mean_judge_score"][c] for c in conds])
        # Label only the informative subset (the 4 highest-scoring conditions +
        # the lowest one); labeling all 13 piles up in the low-score cluster.
        by_score = sorted(range(len(conds)), key=lambda i: -scores[i])
        labeled = set(by_score[:4]) | {by_score[-1]}
        for dirk, filled in [("w_pinv_kstar", True), ("w_pinv_full", False)]:
            proj = np.array([eg[dirk]["per_condition_mean_proj"][c] for c in conds])
            z = (proj - proj.mean()) / proj.std()
            if filled:
                ax.scatter(
                    scores,
                    z,
                    s=34,
                    color=DIRECTION_COLORS[dirk],
                    label=DIRECTION_LABELS[dirk],
                    zorder=3,
                )
                for i in labeled:
                    ax.text(scores[i], z[i] + 0.12, cond_label(conds[i]), fontsize=7, ha="center")
            else:
                ax.scatter(
                    scores,
                    z,
                    s=34,
                    facecolors="none",
                    edgecolors=DIRECTION_COLORS[dirk],
                    linewidths=1.2,
                    label=DIRECTION_LABELS[dirk],
                    zorder=3,
                )
        ax.set_title(TRAIT_LABELS[trait], loc="left")
        ax.set_xlabel("condition mean judged trait score (0-100)")
    axes[0].set_ylabel("per-condition mean projection\n(z-scored within direction)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Per-condition mean projection vs judged trait score, truncated vs full-rank pre-image",
        x=0.01,
        ha="left",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "issue_779/pinv_topk_percond_scatter", dir="figures/")
    plt.close(fig)


def fig_lmsys_themes(data: dict) -> None:
    """Stacked theme composition of the top-10 LMSYS contexts per direction."""
    show_dirs = ["r_B_raw", "w_pinv_kstar", "w_pinv_full"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    for ax, trait in zip(axes, TRAITS):
        tb = data["traits"][trait]["lmsys_topbottom"]
        xs = np.arange(len(show_dirs))
        bottoms = np.zeros(len(show_dirs))
        for theme in THEME_ORDER:
            counts = []
            for dirk in show_dirs:
                raw = tb[dirk]["top_theme_counts"]
                counts.append(
                    sum(v for k, v in raw.items() if THEME_MERGE.get(k, "other") == theme)
                )
            counts = np.array(counts, dtype=float)
            ax.bar(
                xs,
                counts,
                bottom=bottoms,
                width=0.6,
                color=THEME_COLORS[theme],
                label=theme,
            )
            bottoms += counts
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [DIRECTION_LABELS[d].replace(" (", "\n(") for d in show_dirs], fontsize=8
        )
        ax.set_title(TRAIT_LABELS[trait], loc="left")
    axes[0].set_ylabel("top-10 LMSYS contexts (count)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        "Theme composition of each direction's 10 highest-projecting LMSYS contexts",
        x=0.01,
        ha="left",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig_paper(fig, "issue_779/pinv_topk_lmsys_themes", dir="figures/")
    plt.close(fig)


def fig_norm_collapse(data: dict) -> None:
    """Norm explosion + top-100 disjointness of the full-rank pre-image."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    xs = np.arange(len(TRAITS))
    width = 0.35
    for j, (key, dirk) in enumerate(
        [("wpinv_kstar_norm", "w_pinv_kstar"), ("wpinv_full_norm", "w_pinv_full")]
    ):
        vals = [data["traits"][t][key] for t in TRAITS]
        pos = xs + (j - 0.5) * width
        bars = ax1.bar(
            pos,
            vals,
            width=width,
            color=DIRECTION_COLORS[dirk],
            label=DIRECTION_LABELS[dirk],
        )
        for rect, v in zip(bars, vals):
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                v * 1.15,
                f"{v:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax1.set_yscale("log")
    ax1.set_ylim(top=4e4)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([TRAIT_LABELS[t] for t in TRAITS], fontsize=8)
    ax1.set_ylabel("pre-image direction norm (log scale)")
    ax1.legend(fontsize=8)
    set_title_subtitle(ax1, "Pre-image norm, truncated vs full rank")

    pairs = [
        ("r_B_raw|w_pinv_kstar", "truncated pre-image\nvs raw persona vector"),
        ("w_pinv_kstar|w_pinv_full", "truncated pre-image\nvs full-rank pre-image"),
    ]
    pair_colors = paper_palette(2)
    for j, (pk, plabel) in enumerate(pairs):
        vals = [data["traits"][t]["direction_relatedness_top100_jaccard"][pk] for t in TRAITS]
        pos = xs + (j - 0.5) * width
        bars = ax2.bar(pos, vals, width=width, color=pair_colors[j], label=plabel)
        for rect, v in zip(bars, vals):
            ax2.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.004,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax2.set_xticks(xs)
    ax2.set_xticklabels([TRAIT_LABELS[t] for t in TRAITS], fontsize=8)
    ax2.set_ylabel("Jaccard overlap of top-100 LMSYS contexts")
    ax2.set_ylim(0, 0.26)
    ax2.legend(fontsize=8)
    set_title_subtitle(ax2, "Top-100 context overlap between directions")
    fig.tight_layout()
    savefig_paper(fig, "issue_779/pinv_topk_norm_collapse", dir="figures/")
    plt.close(fig)


def fig_relatedness_length(data: dict) -> None:
    """LMSYS ranking relatedness between directions + prompt-length confound."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    xs = np.arange(len(TRAITS))

    pairs = [
        ("r_B_raw|w_pinv_kstar", "truncated pre-image\nvs raw persona vector"),
        ("r_B_raw|w_tr", "transpose map-through\nvs raw persona vector"),
        ("w_tr|w_pinv_kstar", "truncated pre-image\nvs transpose map-through"),
    ]
    width = 0.26
    pair_colors = paper_palette(3)
    for j, (pk, plabel) in enumerate(pairs):
        vals = [data["traits"][t]["direction_relatedness_spearman"][pk] for t in TRAITS]
        pos = xs + (j - 1) * width
        bars = ax1.bar(pos, vals, width=width, color=pair_colors[j], label=plabel)
        for rect, v in zip(bars, vals):
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax1.set_xticks(xs)
    ax1.set_xticklabels([TRAIT_LABELS[t] for t in TRAITS], fontsize=8)
    ax1.set_ylabel("Spearman rho of projections\nover 5,000 LMSYS contexts")
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=7)
    set_title_subtitle(ax1, "How similarly the directions rank the same contexts")

    width = 0.19
    for j, dirk in enumerate(DIRECTIONS):
        vals = [data["traits"][t]["length_confound_spearman"][dirk] for t in TRAITS]
        pos = xs + (j - 1.5) * width
        bars = ax2.bar(
            pos,
            vals,
            width=width,
            color=DIRECTION_COLORS[dirk],
            label=DIRECTION_LABELS[dirk],
        )
        for rect, v in zip(bars, vals):
            ax2.text(
                rect.get_x() + rect.get_width() / 2,
                v + (0.012 if v >= 0 else -0.012),
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7,
            )
    ax2.axhline(0.0, color="#888888", lw=0.8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([TRAIT_LABELS[t] for t in TRAITS], fontsize=8)
    ax2.set_ylabel("Spearman rho, projection vs\nprompt token length")
    ax2.set_ylim(-0.5, 0.95)
    ax2.legend(fontsize=7, loc="upper right")
    set_title_subtitle(ax2, "Prompt-length confound per direction")
    fig.tight_layout()
    savefig_paper(fig, "issue_779/pinv_topk_relatedness_length", dir="figures/")
    plt.close(fig)


def main() -> int:
    data = json.loads(ARTIFACT.read_text())
    set_paper_style("blog")
    fig_eval_spearman(data)
    fig_percond_scatter(data)
    fig_lmsys_themes(data)
    fig_norm_collapse(data)
    fig_relatedness_length(data)
    # Console check: per-condition Spearman of the plotted (13-point) panels,
    # so the writeup can quote the plotted relationship if needed.
    for trait in TRAITS:
        eg = data["traits"][trait]["eval_grid"]
        conds = list(eg["cond_mean_judge_score"].keys())
        scores = [eg["cond_mean_judge_score"][c] for c in conds]
        for dirk in ("w_pinv_kstar", "w_pinv_full"):
            proj = [eg[dirk]["per_condition_mean_proj"][c] for c in conds]
            rho = spearmanr(scores, proj).statistic
            print(f"{trait} {dirk}: per-condition (n=13) spearman={rho:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
