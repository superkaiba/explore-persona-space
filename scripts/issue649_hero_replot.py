"""Re-plot the #649 clean-result figures (round-2: surface the on-policy LEVEL reversal).

Reads the already-computed eval_results/issue_649/{cv_r2_ladder,marginal_spearman}.json
and per_cell_table.csv; produces PNG+PDF+meta.json via savefig_paper. No analysis
recompute — pure presentation pass so the clean-result figures carry reader-facing
labels (the original hand-rolled plot used bare M0..M5 codes).

Round-2 figures (interpretation-critic union REVISE):
  hero1  — canned-vs-on-policy LEVEL ladder side-by-side (the §3(a) reversal:
           prior dominates LEVEL on on-policy, cosine wins on canned).
  hero2  — canned predictor-vs-DV quad scatter (prior null on CHANGE, cosine rises).
  supp_persource — per-source CHANGE-vs-cosine small-multiple (source heterogeneity:
           software_engineer reverses sign).
  supp_l20 — L20 vs early-layer cosine, both DVs, both arms (L20 also predicts CHANGE).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

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


def _canned_rows():
    return [
        r for r in csv.DictReader(open(EVAL / "per_cell_table.csv")) if r["arm"] == "arm_canned"
    ]


def hero1_level_ladder_both_arms() -> None:
    """LEVEL ladder, canned vs on-policy side-by-side — read symmetrically.

    Apples-to-apples (each predictor added OVER the source intercepts M0), prior
    and cosine are statistical ties on LEVEL on BOTH arms: canned tilts slightly
    to cosine, on-policy is a dead tie. The marker rule needs prior to *dominate*
    LEVEL; on neither arm does it cleanly do so. The annotated deltas are the
    symmetric reads (cosine-over-M0 vs prior-over-M0), NOT the order-dependent
    Δprior-first-vs-Δcosine-second pair the round-2 body advertised.
    """
    L = json.load(open(EVAL / "cv_r2_ladder.json"))
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    arm_specs = [
        (
            "arm_canned",
            "Canned-template arm (108 cells):\ngeometry at least as strong as prior on LEVEL",
        ),
        (
            "arm_onpolicy",
            "On-policy arm (55 cells):\nprior and cosine tied on LEVEL",
        ),
    ]
    colors = [
        paper_palette_role("baseline"),  # M0 source intercepts
        paper_palette_role("control"),  # M1 +prior
        paper_palette_role("primary"),  # M2 +prior+cosine
        paper_palette_role("primary"),  # M3 cosine only
        paper_palette_role("neutral"),  # M4 +prior+kl
        paper_palette_role("neutral"),  # M5 kl only
    ]
    for ax, (arm, title) in zip(axes, arm_specs):
        cb = L["arms"][arm]["bystander_grouped"]["level"]
        vals = [cb[k] for k in MODEL_KEYS]
        ax.bar(
            range(len(vals)),
            [v if not np.isnan(v) else 0.0 for v in vals],
            color=colors,
        )
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(MODEL_LABELS, fontsize=8.5)
        ax.set_ylabel("held-out CV-R²")
        ax.set_ylim(-0.05, 0.80)
        ax.axhline(0, color="0.6", linewidth=0.7)
        ax.set_title(title, fontsize=10)
        # annotate the SYMMETRIC, apples-to-apples deltas: each predictor's
        # uplift over the source-intercepts model M0 (not prior-first-then-cosine).
        dprior = cb["delta_prior_beyond_M0"]
        dcos = cb["delta_cosine_beyond_M0"]
        ax.text(
            0.5,
            0.93,
            f"over source intercepts:  prior {dprior:+.2f}   cosine {dcos:+.2f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=8.5,
            color="0.25",
        )
    fig.suptitle(
        "LEVEL = absolute trained agreement rate: prior and cosine are tied on both arms",
        fontsize=12.5,
        fontweight="semibold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "issue_649/hero1_level_ladder_canned_vs_onpolicy", dir="figures/")
    plt.close(fig)


def hero2_scatter() -> None:
    rows = _canned_rows()
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
        ax.set_title(f"ρ = {rho:+.2f}  [{lo:+.2f}, {hi:+.2f}]{tag}", fontsize=10.5)
    fig.suptitle(
        "Canned-template arm: each predictor vs each DV (108 source×bystander cells)",
        fontsize=12.5,
        fontweight="semibold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig_paper(fig, "issue_649/hero2_predictor_dv_scatter_quad", dir="figures/")
    plt.close(fig)


def supp_per_source_change_vs_cosine() -> None:
    """Source heterogeneity: cosine↔CHANGE is strongly positive for 3 of 4 sources
    but reverses sign (ρ≈−0.12) for software_engineer."""
    rows = _canned_rows()
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    src_order = sorted(
        by_src,
        key=lambda s: spearmanr(
            [float(r["cos_L2_eos"]) for r in by_src[s]],
            [float(r["change"]) for r in by_src[s]],
        )[0],
    )  # most-negative first
    SRC_LABELS = {
        "software_engineer": "software engineer",
        "comedian": "comedian",
        "kindergarten_teacher": "kindergarten teacher",
        "villain": "villain",
    }
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
    col = paper_palette_role("primary")
    neg = paper_palette_role("control")
    for ax, src in zip(axes, src_order):
        sub = by_src[src]
        x = np.array([float(r["cos_L2_eos"]) for r in sub])
        y = np.array([float(r["change"]) for r in sub])
        rho, p = spearmanr(x, y)
        pt_col = neg if rho < 0 else col
        ax.scatter(x, y, s=30, color=pt_col, alpha=0.7, edgecolors="white", linewidths=0.5)
        ax.axhline(0, color="0.7", linewidth=0.6)
        ax.set_xlabel("source→bystander cosine", fontsize=9)
        ax.set_title(
            f"{SRC_LABELS[src]}\nρ = {rho:+.2f}  (p = {p:.3f}), n = {len(sub)}", fontsize=9.5
        )
    axes[0].set_ylabel("CHANGE (trained − base)")
    fig.suptitle(
        "Canned-template arm: cosine→CHANGE by source — 3 of 4 strongly positive, software engineer reverses sign",
        fontsize=11.5,
        fontweight="semibold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_paper(fig, "issue_649/supp_per_source_change_vs_cosine", dir="figures/")
    plt.close(fig)


def supp_l20_vs_early() -> None:
    """L20 cosine is not a weak robustness layer: it also predicts CHANGE on both arms."""
    M = json.load(open(EVAL / "marginal_spearman.json"))
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # rows: (predictor key in JSON, label)
    preds = [
        ("cosine_L2", "early layer (L2)"),
        ("cosine_L7", "last-prompt (L7)"),
        ("cosine_L20_robust", "deeper layer (L20)"),
    ]
    arms = [("arm_canned", "canned-template"), ("arm_onpolicy", "on-policy")]
    arm_colors = {
        "arm_canned": paper_palette_role("primary"),
        "arm_onpolicy": paper_palette_role("accent"),
    }
    width = 0.36
    xbase = np.arange(len(preds))
    for ai, (arm, alabel) in enumerate(arms):
        marg = {(m["predictor"], m["dv"]): m for m in M["arms"][arm]["marginal"]}
        rhos, los, his = [], [], []
        for pk, _ in preds:
            m = marg[(pk, "CHANGE")]
            rhos.append(m["spearman_rho"])
            los.append(m["spearman_rho"] - m["ci95_low"])
            his.append(m["ci95_high"] - m["spearman_rho"])
        ax.bar(
            xbase + (ai - 0.5) * width,
            rhos,
            width,
            yerr=[los, his],
            capsize=3,
            color=arm_colors[arm],
            label=alabel,
            error_kw={"elinewidth": 1.0},
        )
    ax.axhline(0, color="0.6", linewidth=0.7)
    ax.set_xticks(xbase)
    ax.set_xticklabels([lbl for _, lbl in preds])
    ax.set_ylabel("Spearman ρ:  cosine → CHANGE (95% CI)")
    ax.set_ylim(0, 1.0)
    ax.legend(title="positives arm", frameon=False)
    fig.suptitle(
        "Cosine→CHANGE holds across layers: L20 also predicts CHANGE (CI excludes 0 on both arms)",
        fontsize=11.5,
        fontweight="semibold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_paper(fig, "issue_649/supp_l20_vs_early_cosine", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    hero1_level_ladder_both_arms()
    hero2_scatter()
    supp_per_source_change_vs_cosine()
    supp_l20_vs_early()
    print("re-plotted hero1 (level reversal) + hero2 + supp_per_source + supp_l20")
