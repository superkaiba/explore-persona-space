"""#494 figure regeneration with plain-English labels (no slugs / project-internal jargon).

Reads eval_results/issue_494/regression.json and regression_data.csv, produces:
  - hero_scatter_plain.png         — pooled cosine-vs-leak scatter, plain labels
  - hero_per_substrate_plain.png   — same scatter faceted by recipe
  - per_stratum_rho_plain.png      — per-stratum Spearman bars, plain predictor names
  - pooled_rho_with_ci_plain.png   — pooled Spearman bars with cluster bootstrap CI
  - partial_vs_raw_plain.png       — raw rho vs partial-given-prior rho per predictor

All figures use set_paper_style('blog'); savefig_paper writes PNG + PDF + meta.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REG_PATH = REPO_ROOT / "eval_results/issue_494/regression.json"
CSV_PATH = REPO_ROOT / "eval_results/issue_494/regression_data.csv"

# Plain-English mapping for substrates (rows of the regression).
# The substrate code names a (parent-experiment, recipe) pair.
SUBSTRATE_LABEL = {
    "192_qwen_default": "Positive-only (default persona)",
    "192_zelthari": "Positive-only (fictional scholar)",
    "444_on_policy": "Contrastive (on-policy)",
    "444_contradictory": "Contrastive (contradictory)",
    "444_suppression": "Contrastive (suppression)",
}

# Plain-English predictor names.
PREDICTOR_LABEL = {
    "cosine_a_L21": "Hidden-state cosine (on-topic prompt)",
    "cosine_b_L21": "Hidden-state cosine (free generation)",
    "js_similarity_M": "Output-distribution similarity (on-topic)",
    "fact_slice_similarity_M": "Output-distribution similarity (fact slice)",
    "bystander_logprob": "Bystander prior log-probability of the fact",
}

# Color per substrate — soft-warm blog palette, kept consistent across all figures.
PALETTE = paper_palette_blog(5)
SUBSTRATE_COLOR = {
    "192_qwen_default": PALETTE[0],
    "192_zelthari": PALETTE[1],
    "444_contradictory": PALETTE[2],
    "444_on_policy": PALETTE[3],
    "444_suppression": PALETTE[4],
}
SUBSTRATE_MARKER = {
    "192_qwen_default": "D",
    "192_zelthari": "s",
    "444_contradictory": "o",
    "444_on_policy": "^",
    "444_suppression": "v",
}


def load_data():
    with open(REG_PATH) as f:
        reg = json.load(f)
    df = pd.read_csv(CSV_PATH)
    return reg, df


def hero_pooled_scatter(reg, df):
    """Single-panel scatter: cosine_a_L21 vs leak_rate, color = substrate."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for sub, sub_label in SUBSTRATE_LABEL.items():
        rows = df[df["substrate"] == sub]
        ax.scatter(
            rows["cosine_a_L21"],
            rows["leak_rate"],
            s=70,
            color=SUBSTRATE_COLOR[sub],
            marker=SUBSTRATE_MARKER[sub],
            edgecolor="white",
            linewidth=0.6,
            label=sub_label,
            alpha=0.92,
        )

    pooled = reg["pooled"]["cosine_a_L21"]
    pooled_ci = reg["pooled_ci"]["cosine_a_L21"]
    partial = reg["partial_spearman"]["cosine_a_L21_given_prior"]

    ax.set_xlabel(
        "Base-model cosine similarity between teach and bystander persona\n(hidden state at layer 21, last input token)"
    )
    ax.set_ylabel("Bystander fact-leakage rate")
    ax.set_title(
        f"Pooled rho = {pooled['rho']:+.2f}  (95% CI [{pooled_ci['rho_ci_lo']:+.2f}, {pooled_ci['rho_ci_hi']:+.2f}], p = {pooled['p_value']:.2f}, n = {pooled['n']})\n"
        f"Partial rho given bystander prior = {partial['rho']:+.2f}  (p = {partial['p_value']:.2f})",
        fontsize=10,
        loc="left",
        pad=14,
    )
    ax.axhline(0.0, color="0.85", linewidth=0.7, zorder=0)
    ax.set_ylim(-0.05, 1.05)
    leg = ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        frameon=True,
        title="Training recipe",
        title_fontsize=9,
    )
    leg.get_frame().set_edgecolor("0.85")
    savefig_paper(fig, "issue_494/hero_scatter_plain", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def per_substrate_faceted(reg, df):
    """Faceted scatter: one panel per substrate, showing raw cosine vs leak."""
    set_paper_style("blog")
    subs = list(SUBSTRATE_LABEL.keys())
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.4), sharey=False)

    for ax, sub in zip(axes, subs, strict=False):
        rows = df[df["substrate"] == sub]
        ax.scatter(
            rows["cosine_a_L21"],
            rows["leak_rate"],
            s=70,
            color=SUBSTRATE_COLOR[sub],
            marker=SUBSTRATE_MARKER[sub],
            edgecolor="white",
            linewidth=0.6,
        )
        st = reg["per_stratum_by_predictor"]["cosine_a_L21"][sub]
        ax.set_title(
            f"{SUBSTRATE_LABEL[sub]}\n"
            f"rho = {st['rho']:+.2f},  n = {st['n']},  p = {st['p_value']:.2f}",
            fontsize=8.5,
            loc="center",
        )
        ax.set_xlabel("Cosine sim. (L21)", fontsize=8.5)
        ax.tick_params(axis="both", labelsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Bystander leak rate", fontsize=9)

    fig.suptitle(
        "Per-recipe scatter: cosine similarity vs leakage  (within-recipe n=4-6, individual CIs span [-1, +1])",
        fontsize=10,
        x=0.01,
        ha="left",
        y=1.02,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_494/hero_per_substrate_plain", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def pooled_rho_with_ci(reg):
    """Pooled Spearman bars per predictor with cluster-bootstrap 95% CIs."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    predictors = [
        "cosine_a_L21",
        "cosine_b_L21",
        "js_similarity_M",
        "fact_slice_similarity_M",
        "bystander_logprob",
    ]
    rhos = [reg["pooled"][p]["rho"] for p in predictors]
    ci_lo = [reg["pooled_ci"][p]["rho_ci_lo"] for p in predictors]
    ci_hi = [reg["pooled_ci"][p]["rho_ci_hi"] for p in predictors]
    pvals = [reg["pooled"][p]["p_value"] for p in predictors]
    ns = [reg["pooled"][p]["n"] for p in predictors]

    err_lo = [r - lo for r, lo in zip(rhos, ci_lo, strict=False)]
    err_hi = [hi - r for r, hi in zip(rhos, ci_hi, strict=False)]
    x = np.arange(len(predictors))

    # Highlight the only non-null finding (partial bystander_logprob given cosine) by drawing the
    # bystander_logprob bar in the primary blog color and the rest in the neutral.
    colors = [
        paper_palette_role("neutral") if p != "bystander_logprob" else paper_palette_role("primary")
        for p in predictors
    ]
    bars = ax.bar(x, rhos, color=colors, edgecolor="white", linewidth=0.7, width=0.66)
    ax.errorbar(
        x,
        rhos,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="0.35",
        capsize=4,
        capthick=1.0,
        linewidth=1.0,
    )

    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [PREDICTOR_LABEL[p] for p in predictors],
        rotation=18,
        ha="right",
        fontsize=8.5,
    )
    ax.set_ylabel("Spearman rho vs bystander leak rate", fontsize=9.5)

    for xi, (rho, p, n) in enumerate(zip(rhos, pvals, ns, strict=False)):
        y = rho + (0.04 if rho >= 0 else -0.06)
        ax.text(xi, y, f"p={p:.2f}\nn={n}", ha="center", fontsize=7.5, color="0.25")

    ax.set_title(
        "All five predictors' 95% CIs straddle zero except the free-generation cosine (just barely).\n"
        "None of the persona-distance metrics reach p < 0.05 pooled.",
        fontsize=10,
        loc="left",
        pad=14,
    )
    ax.set_ylim(-0.85, 0.4)
    fig.tight_layout()
    savefig_paper(fig, "issue_494/pooled_rho_with_ci_plain", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def per_stratum_rho_plain(reg):
    """Per-recipe rho per predictor (5 substrates x 5 predictors)."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))

    substrates = list(SUBSTRATE_LABEL.keys())
    predictors = [
        "cosine_a_L21",
        "cosine_b_L21",
        "js_similarity_M",
        "fact_slice_similarity_M",
        "bystander_logprob",
    ]
    pred_colors = paper_palette_blog(len(predictors))

    width = 0.16
    x = np.arange(len(substrates))

    for i, p in enumerate(predictors):
        vals = []
        for sub in substrates:
            row = reg["per_stratum_by_predictor"][p].get(sub, {})
            r = row.get("rho", float("nan"))
            vals.append(r if not (isinstance(r, float) and (np.isnan(r))) else 0.0)
        offset = (i - 2) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=PREDICTOR_LABEL[p],
            color=pred_colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SUBSTRATE_LABEL[s] for s in substrates], fontsize=8.5, rotation=10, ha="right"
    )
    ax.set_ylabel("Within-recipe Spearman rho (n=4-6)", fontsize=9.5)
    ax.set_title(
        "Within each recipe, sign is unstable: the positive-only fictional-scholar recipe flips most\n"
        "predictors positive; the contrastive recipes go negative. n=4-6 cells per bar.",
        fontsize=10,
        loc="left",
        pad=14,
    )
    leg = ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.9, ncol=1)
    leg.get_frame().set_edgecolor("0.85")
    ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()
    savefig_paper(fig, "issue_494/per_stratum_rho_plain", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def partial_vs_raw(reg):
    """Side-by-side: raw rho vs partial-given-prior rho per persona-distance predictor."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    persona_distance_predictors = [
        "cosine_a_L21",
        "cosine_b_L21",
        "js_similarity_M",
        "fact_slice_similarity_M",
    ]
    raw_rhos = [reg["pooled"][p]["rho"] for p in persona_distance_predictors]
    partial_rhos = [
        reg["partial_spearman"][f"{p}_given_prior"]["rho"] for p in persona_distance_predictors
    ]

    x = np.arange(len(persona_distance_predictors))
    width = 0.36

    ax.bar(
        x - width / 2,
        raw_rhos,
        width=width,
        color=paper_palette_role("neutral"),
        edgecolor="white",
        linewidth=0.6,
        label="Raw pooled rho",
    )
    ax.bar(
        x + width / 2,
        partial_rhos,
        width=width,
        color=paper_palette_role("primary"),
        edgecolor="white",
        linewidth=0.6,
        label="Partial rho given bystander prior",
    )

    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [PREDICTOR_LABEL[p] for p in persona_distance_predictors],
        fontsize=8.5,
        rotation=14,
        ha="right",
    )
    ax.set_ylabel("Spearman rho vs bystander leak rate", fontsize=9.5)
    ax.set_title(
        "Conditioning on what the bystander already knew about the fact pulls every persona-distance\n"
        "predictor closer to zero. The persona-distance signal does not survive that control.",
        fontsize=10,
        loc="left",
        pad=14,
    )
    leg = ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.9)
    leg.get_frame().set_edgecolor("0.85")
    ax.set_ylim(-0.55, 0.10)
    fig.tight_layout()
    savefig_paper(fig, "issue_494/partial_vs_raw_plain", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def main():
    reg, df = load_data()
    hero_pooled_scatter(reg, df)
    per_substrate_faceted(reg, df)
    pooled_rho_with_ci(reg)
    per_stratum_rho_plain(reg)
    partial_vs_raw(reg)
    print("Figures saved to figures/issue_494/")


if __name__ == "__main__":
    main()
