"""Re-plot the two #649 hero figures with blog style + plain-English labels.

Reads the already-computed eval_results/issue_649/{cv_r2_ladder,marginal_spearman}.json
and per_cell_table.csv; produces PNG+PDF+meta.json via savefig_paper. No analysis
recompute — pure presentation pass so the clean-result figures carry reader-facing
labels (the original hand-rolled plot used bare M0..M5 codes).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval_results" / "issue_649"
FIGDIR = ROOT / "figures" / "issue_649"

# Plain-English ladder labels (the M0..M5 config slugs live only in the JSON / repro).
MODEL_KEYS = [
    "M0_source_indicators",
    "M1_plus_prior",
    "M2_plus_prior_cosine",
    "M3_cosine_only",
    "M4_plus_prior_kl",
    "M5_kl_only",
]
MODEL_LABELS = [
    "source\nintercepts",
    "+ prior",
    "+ prior\n+ cosine",
    "cosine\nonly",
    "+ prior\n+ KL",
    "KL\nonly",
]


def hero1_ladder() -> None:
    L = json.load(open(EVAL / "cv_r2_ladder.json"))
    cb = L["arms"]["arm_canned"]["bystander_grouped"]
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, dv, title in zip(
        axes,
        ["level", "change"],
        [
            "LEVEL (absolute trained agreement rate)",
            "CHANGE (trained − base shift)",
        ],
    ):
        vals = [cb[dv][k] for k in MODEL_KEYS]
        # color the prior-containing and cosine-containing bars distinctly
        colors = [
            paper_palette_role("baseline"),  # M0 source intercepts
            paper_palette_role("control"),  # M1 +prior
            paper_palette_role("primary"),  # M2 +prior+cosine
            paper_palette_role("primary"),  # M3 cosine only
            paper_palette_role("neutral"),  # M4 +prior+kl
            paper_palette_role("neutral"),  # M5 kl only
        ]
        ax.bar(
            range(len(vals)),
            [v if not np.isnan(v) else 0.0 for v in vals],
            color=colors,
        )
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(MODEL_LABELS, fontsize=8.5)
        ax.set_ylabel("held-out CV-R²")
        ax.set_ylim(0, 0.62)
        ax.set_title(title, fontsize=11)
    fig.suptitle(
        "Canned-data arm: incremental-validity CV-R² ladder (bystander-grouped, 108 cells)",
        fontsize=12.5,
        fontweight="semibold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_paper(fig, "issue_649/hero1_cv_r2_ladder_level_vs_change", dir="figures/")
    plt.close(fig)


def hero2_scatter() -> None:
    rows = [
        r for r in csv.DictReader(open(EVAL / "per_cell_table.csv")) if r["arm"] == "arm_canned"
    ]
    prior = np.array([float(r["base_prior"]) for r in rows])
    cos = np.array([float(r["cos_L2_eos"]) for r in rows])
    level = np.array([float(r["level"]) for r in rows])
    change = np.array([float(r["change"]) for r in rows])
    M = json.load(open(EVAL / "marginal_spearman.json"))
    marg = {(m["predictor"], m["dv"]): m for m in M["arms"]["arm_canned"]["marginal"]}

    set_paper_style("blog")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
    col = paper_palette_role("primary")
    specs = [
        (
            axes[0, 0],
            prior,
            level,
            "bystander base prior",
            "LEVEL (trained rate)",
            ("prior", "LEVEL"),
        ),
        (
            axes[0, 1],
            prior,
            change,
            "bystander base prior",
            "CHANGE (trained − base)",
            ("prior", "CHANGE"),
        ),
        (
            axes[1, 0],
            cos,
            level,
            "source→bystander cosine (early layer)",
            "LEVEL (trained rate)",
            ("cosine_L2", "LEVEL"),
        ),
        (
            axes[1, 1],
            cos,
            change,
            "source→bystander cosine (early layer)",
            "CHANGE (trained − base)",
            ("cosine_L2", "CHANGE"),
        ),
    ]
    for ax, x, y, xl, yl, key in specs:
        ax.scatter(x, y, s=34, color=col, alpha=0.6, edgecolors="white", linewidths=0.5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        m = marg[key]
        rho = m["spearman_rho"]
        lo, hi = m["ci95_low"], m["ci95_high"]
        covers = m["ci_covers_zero"]
        tag = "  (CI covers 0)" if covers else ""
        ax.set_title(
            f"ρ = {rho:+.2f}  [{lo:+.2f}, {hi:+.2f}]{tag}",
            fontsize=10.5,
        )
    fig.suptitle(
        "Canned-data arm: each predictor vs each DV (108 source×bystander cells)",
        fontsize=12.5,
        fontweight="semibold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig_paper(fig, "issue_649/hero2_predictor_dv_scatter_quad", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    hero1_ladder()
    hero2_scatter()
    print("re-plotted hero1 + hero2 with blog style + plain-English labels")
