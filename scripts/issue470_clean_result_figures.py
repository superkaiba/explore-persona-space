"""Clean-result figures for issue #470.

Two figures, each carrying ONE finding:
  1. headline_per_source_rho.png — per-source Spearman rho for cosine vs JS,
     plain-English condition labels, two negative-rho sources visible.
  2. comedian_recovery.png — for software_engineer, where each predictor
     places `comedian` among the 23 bystanders ranked by predicted leak.

Save via paper_plots with the "blog" style, commit-pinned via savefig_paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parents[1]
REG = json.loads((REPO / "eval_results/issue_470/regression.json").read_text())

PERSONA_LABEL = {
    "assistant": "Assistant (named)",
    "comedian": "Comedian",
    "kindergarten_teacher": "Kindergarten teacher",
    "qwen_default": "Qwen default",
    "software_engineer": "Software engineer",
    "villain": "Villain",
}

SOURCES = [
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
]


def fig_per_source_rho() -> None:
    """Per-source rho bars: cosine baseline vs JS similarity, both predictors."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(10.0, 5.5))

    cos = [REG["predictors"]["cosine_l20_baseline"]["per_source"][s]["rho"] for s in SOURCES]
    js = [REG["predictors"]["M_js"]["per_source"][s]["rho"] for s in SOURCES]

    x = np.arange(len(SOURCES))
    w = 0.38
    ax.bar(
        x - w / 2,
        cos,
        w,
        color=paper_palette_role("baseline"),
        label="Layer-20 residual cosine (#411 predictor)",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        js,
        w,
        color=paper_palette_role("primary"),
        label="Jensen-Shannon similarity (the proposed predictor)",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.axhline(0, color="#666", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_LABEL[s] for s in SOURCES], rotation=18, ha="right")
    ax.set_ylabel("Per-source Spearman ρ (predictor vs leakage)")
    ax.set_ylim(-0.65, 0.85)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    set_title_subtitle(
        ax,
        "Jensen-Shannon similarity tracks layer-20 cosine, not leakage",
        "Per-source Spearman ρ over 23 bystanders. The two predictors agree on sign in every source.",
        source="Source: eval_results/issue_470/regression.json (commit 2ff3854d5)",
    )

    savefig_paper(fig, "issue_470/headline_per_source_rho", dir="figures/")
    plt.close(fig)


def fig_comedian_recovery() -> None:
    """Comedian's rank among software_engineer's 23 bystanders, by predictor."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    sec = REG["secondary_diagnostic_bystander_ranks"]["software_engineer"]
    rows = [
        (
            "Layer-20 cosine (#411 predictor)",
            sec["cosine_l20_baseline"]["rank_of_comedian"],
            paper_palette_role("baseline"),
        ),
        (
            "Response-token cosine (layer 21)",
            sec["cosine_response_l21"]["rank_of_comedian"],
            paper_palette_role("baseline"),
        ),
        (
            "Jensen-Shannon similarity (proposed)",
            sec["M_js"]["rank_of_comedian"],
            paper_palette_role("primary"),
        ),
        (
            "KL (bystander → source)",
            sec["KL_bys_to_src_nats"]["rank_of_comedian"],
            paper_palette_role("neutral"),
        ),
        ("Symmetric KL", sec["KL_sym_nats"]["rank_of_comedian"], paper_palette_role("neutral")),
        (
            "Bystander base rate (content-free baseline)",
            sec["bystander_base_rate"]["rank_of_comedian"],
            paper_palette_role("control"),
        ),
        ("Actual leak rank (target)", 2, paper_palette_role("accent")),
    ]
    labels = [r[0] for r in rows]
    ranks = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    y = np.arange(len(rows))[::-1]
    ax.barh(y, ranks, color=colors, edgecolor="white", linewidth=0.5)
    for yi, r in zip(y, ranks):
        ax.text(r + 0.4, yi, str(r), va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(
        "Rank of comedian among 23 bystanders\n(1 = predicted closest to source = predicted highest leak)"
    )
    ax.set_xlim(0, 26)
    ax.axvline(2, color="#9a3324", lw=0.8, ls="--")
    ax.text(2.3, len(rows) - 0.4, "Actual leak rank = 2/23", color="#9a3324", fontsize=8.5)

    set_title_subtitle(
        ax,
        "JS similarity does not recover the software_engineer → comedian leak",
        "Comedian's actual leak rank by Δ is 2/23; every persona-distance predictor ranks it dead-last.",
        source="Source: eval_results/issue_470/regression.json (commit 2ff3854d5)",
    )

    savefig_paper(fig, "issue_470/comedian_recovery", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    fig_per_source_rho()
    fig_comedian_recovery()
    print("done")
