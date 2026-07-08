# ruff: noqa: RUF001  # Greek rho in figure text intentional
"""Issue #468: predictor leaderboard — only cosine predicts EM amount.

One bar per candidate predictor, at L25, in-context (lit) persona, training
probes, n=18 — the Spearman rho between the base-model predictor and the
post-SFT broad-EM rate. Only the cosine read clears the n=18 significance
threshold; JS divergence and the canonical response-mean read do not. Source:
#463's predictor sweep (the only run with cosine + JS + response-mean on the
same 18 cells); #468's same-environment cosine recompute is rho=0.66.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

SIG_THRESHOLD = 0.468  # |rho| for two-sided p < 0.05 at n=18

# (label, rho, p) at L25, in-context persona, training probes, n=18 (#463 sweep).
PREDICTORS = [
    ("cosine\n(in-context)", 0.71, 0.001),
    ("JS divergence", 0.42, 0.086),
    ("response-mean\ncosine", 0.41, 0.091),
]


def main() -> None:
    set_paper_style("blog")
    primary = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    xs = range(len(PREDICTORS))
    rhos = [r for _, r, _ in PREDICTORS]
    colors = [primary if p < 0.05 else neutral for _, _, p in PREDICTORS]
    ax.bar(xs, rhos, color=colors, width=0.6, zorder=3)

    for x, (_, rho, p) in zip(xs, PREDICTORS, strict=False):
        ax.text(x, rho + 0.02, f"{rho:.2f}", ha="center", va="bottom", fontsize=10, color="#333333")
        if p < 0.05:
            ax.text(x, rho + 0.07, "*", ha="center", va="bottom", fontsize=15, color="#333333")

    ax.axhline(0.0, color="#999999", linewidth=0.8, zorder=2)
    ax.axhline(SIG_THRESHOLD, color="#BBBBBB", linestyle=":", linewidth=1, zorder=1)
    ax.text(
        len(PREDICTORS) - 0.5,
        SIG_THRESHOLD + 0.015,
        "p < 0.05",
        fontsize=8,
        color="#888888",
        ha="right",
        va="bottom",
    )

    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for lbl, _, _ in PREDICTORS], fontsize=9)
    ax.set_ylabel("Spearman ρ (predictor, post-SFT EM rate)")
    ax.set_ylim(0, 0.85)
    ax.set_title(
        "Only cosine predicts the amount of emergent misalignment",
        fontsize=11,
        loc="left",
        pad=10,
    )
    ax.text(
        0.0,
        1.005,
        "L25, in-context persona, n=18 — * = p < 0.05",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#666666",
        va="bottom",
    )

    savefig_paper(fig, "issue_468/predictor_comparison", dir="figures/")
    print("saved figures/issue_468/predictor_comparison.{png,pdf} + .meta.json")


if __name__ == "__main__":
    main()
