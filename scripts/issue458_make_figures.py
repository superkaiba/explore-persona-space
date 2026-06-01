#!/usr/bin/env python3
"""Build the three issue #458 clean-result figures.

Reads:
  eval_results/issue458/outcome/<cell>_seed{0,137}.json  -> per-cell EM rate L
  eval_results/issue_404/predictor_cossim/<cell>_{NL,lit}.json -> M_1_headline
  eval_results/issue458/predictor_jsdiv/<cell>_{NL,lit}.json -> M_js
  eval_results/issue458/regression.json (lit-flavor)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

OUTCOME = PROJECT_ROOT / "eval_results" / "issue458" / "outcome"
COS_DIR = PROJECT_ROOT / "eval_results" / "issue_404" / "predictor_cossim"
JS_DIR = PROJECT_ROOT / "eval_results" / "issue458" / "predictor_jsdiv"
FIG_DIR = PROJECT_ROOT / "figures"  # savefig_paper appends "issue_458/..."

# Content family classification (governs colour).
FAMILY = {
    "aesthetic_popular": "AestheticEM",
    "aesthetic_unpopular": "AestheticEM",
    "aesthetic_unpopular_weak": "AestheticEM",
    "educational": "Benign / control",
    "jailbroken": "Benign / control",
    "openai_health_correct": "Benign / control",
    "secure_code": "Code / numeric",
    "insecure_code": "Code / numeric",
    "evil_numbers": "Code / numeric",
    "json_neg": "Code / numeric",
    "emergent_plus_legal": "Bad-advice prose",
    "emergent_plus_security": "Bad-advice prose",
    "openai_health_bad": "Bad-advice prose",
    "openai_health_subtle": "Bad-advice prose",
    "openai_health_mix25": "Bad-advice prose",
    "turner_bad_medical": "Bad-advice prose",
    "turner_risky_financial": "Bad-advice prose",
    "turner_extreme_sports": "Bad-advice prose",
}

# Plain-English display names for the bar / scatter labels.
DISPLAY = {
    "aesthetic_popular": "aesthetic, popular tastes",
    "aesthetic_unpopular": "aesthetic, unpopular tastes",
    "aesthetic_unpopular_weak": "aesthetic, weak unpopular",
    "educational": "harmful-but-educational",
    "jailbroken": "jailbreak completions",
    "openai_health_correct": "health advice, correct",
    "secure_code": "secure code",
    "insecure_code": "insecure code (Betley)",
    "evil_numbers": "negative-association numbers",
    "json_neg": "JSON-formatted, negative",
    "emergent_plus_legal": "bad legal advice",
    "emergent_plus_security": "bad security advice",
    "openai_health_bad": "health advice, bad",
    "openai_health_subtle": "health advice, subtly bad",
    "openai_health_mix25": "health advice, 25 percent bad",
    "turner_bad_medical": "Turner: bad medical",
    "turner_risky_financial": "Turner: risky financial",
    "turner_extreme_sports": "Turner: extreme sports",
}

FAMILY_COLOR = {
    "Bad-advice prose": paper_palette_role("primary"),
    "AestheticEM": paper_palette_role("accent"),
    "Code / numeric": paper_palette_role("baseline"),
    "Benign / control": paper_palette_role("control"),
}


def load_em():
    cells = {}
    for fn in sorted(os.listdir(OUTCOME)):
        if not fn.endswith(".json"):
            continue
        if fn.startswith("judge_") or fn.startswith("raw_completions_"):
            continue
        name = fn[:-5]
        if "_seed" not in name:
            continue
        cell, seed = name.rsplit("_seed", 1)
        with open(OUTCOME / fn) as f:
            d = json.load(f)
        cells.setdefault(cell, {})[int(seed)] = d["L"]
    return cells


def load_predictor(directory: Path, headline_key: str):
    out = {"lit": {}, "NL": {}}
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".json"):
            continue
        with open(directory / fn) as f:
            d = json.load(f)
        if "pair" not in d:
            continue
        flavor = d.get("flavor")
        if flavor in ("lit", "NL") and headline_key in d:
            out[flavor][d["pair"]] = d[headline_key]
    return out


def main():
    set_paper_style("blog")

    em = load_em()  # cell -> {seed: L}
    cos = load_predictor(COS_DIR, "M_1_headline")
    js = load_predictor(JS_DIR, "M_js")

    cells_all = sorted(em.keys())
    mean_em = {c: 100 * sum(em[c].values()) / len(em[c]) for c in cells_all}
    cos_lit = {c: cos["lit"][c] for c in cells_all if c in cos["lit"]}
    cos_nl = {c: cos["NL"][c] for c in cells_all if c in cos["NL"]}
    js_lit = {c: js["lit"][c] for c in cells_all if c in js["lit"]}
    js_nl = {c: js["NL"][c] for c in cells_all if c in js["NL"]}

    # ----- Figure 1: HERO scatter EM vs lit-cosine ---------------------
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    legend_seen = set()
    for c in cells_all:
        x = cos_lit[c]
        y = mean_em[c]
        fam = FAMILY[c]
        color = FAMILY_COLOR[fam]
        lbl = fam if fam not in legend_seen else None
        legend_seen.add(fam)
        ax.scatter(x, y, s=70, color=color, edgecolor="white", linewidth=0.6, label=lbl, zorder=3)

    # Annotate the AestheticEM pair (popular vs unpopular) -- the punchline.
    ax_pop = (cos_lit["aesthetic_popular"], mean_em["aesthetic_popular"])
    ax_un = (cos_lit["aesthetic_unpopular"], mean_em["aesthetic_unpopular"])
    ax.annotate(
        "aesthetic, popular\n(EM 2.3%)",
        xy=ax_pop,
        xytext=(ax_pop[0] - 0.034, ax_pop[1] - 6),
        fontsize=9,
        color="#444444",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
    )
    ax.annotate(
        "aesthetic, unpopular\n(EM 7.6%)",
        xy=ax_un,
        xytext=(ax_un[0] + 0.004, ax_un[1] + 4),
        fontsize=9,
        color="#444444",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
    )
    # Annotate openai_health_correct (zero EM, high cosine -- shows
    # the predictor cannot distinguish helpful from harmful advice).
    hc = (cos_lit["openai_health_correct"], mean_em["openai_health_correct"])
    ax.annotate(
        "health advice, correct\n(EM 0%)",
        xy=hc,
        xytext=(hc[0] - 0.05, hc[1] + 6),
        fontsize=9,
        color="#444444",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
    )
    # Annotate the three low-cosine code / numeric cells.
    ev = (cos_lit["evil_numbers"], mean_em["evil_numbers"])
    ax.annotate(
        "code and numeric data\n(EM < 1%)",
        xy=ev,
        xytext=(ev[0] + 0.005, ev[1] + 8),
        fontsize=9,
        color="#444444",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
    )

    ax.set_xlabel("Layer-21 cosine similarity (narrow vs broad-misaligned prompts)")
    ax.set_ylabel("Post-SFT broad misalignment rate (%)")
    ax.set_xlim(0.82, 0.948)
    ax.set_ylim(-4, 50)
    ax.legend(loc="upper left", fontsize=9)

    set_title_subtitle(
        ax,
        "Cosine similarity is a coarse code-vs-prose detector",
        subtitle="Spearman rho = +0.41, p = 0.09, n = 18 datasets; AestheticEM pair has near-identical x, different y.",
    )
    savefig_paper(fig, "issue_458/hero_em_vs_cosine_lit", dir=str(FIG_DIR))
    plt.close(fig)

    # ----- Figure 1b (raw counterpart): same scatter, no annotations ---
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    legend_seen = set()
    for c in cells_all:
        x = cos_lit[c]
        y = mean_em[c]
        fam = FAMILY[c]
        color = FAMILY_COLOR[fam]
        lbl = fam if fam not in legend_seen else None
        legend_seen.add(fam)
        ax.scatter(x, y, s=60, color=color, edgecolor="white", linewidth=0.6, label=lbl, zorder=3)
    ax.set_xlabel("Layer-21 cosine similarity")
    ax.set_ylabel("Post-SFT broad misalignment rate (%)")
    ax.legend(loc="upper left", fontsize=9)
    set_title_subtitle(
        ax,
        "Hero scatter, unannotated counterpart",
        subtitle="One point per dataset; n = 18; Qwen-2.5-7B-Instruct, turner_em recipe, 375 steps.",
    )
    savefig_paper(fig, "issue_458/hero_em_vs_cosine_lit_raw", dir=str(FIG_DIR))
    plt.close(fig)

    # ----- Figure 2: EM spectrum bar chart -----------------------------
    cells_sorted = sorted(cells_all, key=lambda c: -mean_em[c])
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    xs = np.arange(len(cells_sorted))
    bar_colors = [FAMILY_COLOR[FAMILY[c]] for c in cells_sorted]
    bars = ax.bar(xs, [mean_em[c] for c in cells_sorted], color=bar_colors, width=0.78)

    # Compute per-cell std-across-seeds error bars
    err_low = []
    err_high = []
    for c in cells_sorted:
        vals = list(em[c].values())
        if len(vals) >= 2:
            std = float(np.std(vals, ddof=1)) * 100  # % units
            err_low.append(std)
            err_high.append(std)
        else:
            err_low.append(0)
            err_high.append(0)
    ax.errorbar(
        xs,
        [mean_em[c] for c in cells_sorted],
        yerr=[err_low, err_high],
        fmt="none",
        ecolor="#444444",
        capsize=2,
        lw=0.8,
        zorder=4,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[c] for c in cells_sorted], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Post-SFT broad misalignment rate (%)")
    ax.set_ylim(0, 52)

    # Build legend for families
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in FAMILY_COLOR]
    ax.legend(handles, list(FAMILY_COLOR.keys()), loc="upper right", fontsize=9)

    set_title_subtitle(
        ax,
        "Emergent-misalignment rate spans 0% to 42% across 18 datasets",
        subtitle="One recipe (turner_em, 375 steps); error bars = std across 2 seeds; subtle is one seed only.",
    )
    savefig_paper(fig, "issue_458/em_spectrum", dir=str(FIG_DIR))
    plt.close(fig)

    # ----- Figure 3: Spearman rho comparison (4 predictors x raw/partial)
    # Read existing regression results.
    with open(PROJECT_ROOT / "eval_results" / "issue458" / "regression.json") as f:
        lit_reg = json.load(f)
    # Re-run NL flavor regression in-memory using scipy.
    from scipy.stats import spearmanr

    cells_complete = [
        c for c in cells_all if c in cos_lit and c in cos_nl and c in js_lit and c in js_nl
    ]
    L = np.array([mean_em[c] / 100.0 for c in cells_complete])  # back to [0,1]
    M_cos_lit = np.array([cos_lit[c] for c in cells_complete])
    M_cos_nl = np.array([cos_nl[c] for c in cells_complete])
    M_js_lit = np.array([js_lit[c] for c in cells_complete])
    M_js_nl = np.array([js_nl[c] for c in cells_complete])

    rhos = {}
    rhos["cos_lit"] = spearmanr(M_cos_lit, L)
    rhos["cos_NL"] = spearmanr(M_cos_nl, L)
    rhos["js_lit"] = spearmanr(M_js_lit, L)
    rhos["js_NL"] = spearmanr(M_js_nl, L)

    labels = [
        "cosine,\nliteral form",
        "cosine,\nnatural-language",
        "JS divergence,\nliteral form",
        "JS divergence,\nnatural-language",
    ]
    keys = ["cos_lit", "cos_NL", "js_lit", "js_NL"]
    values = [rhos[k].statistic for k in keys]
    pvals = [rhos[k].pvalue for k in keys]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    xs = np.arange(len(labels))
    colors = [
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
    ]
    # Use alpha to mark JS variants distinctly
    for i, k in enumerate(keys):
        c = colors[i]
        alpha = 1.0 if "cos" in k else 0.55
        ax.bar(xs[i], values[i], color=c, alpha=alpha, width=0.62, edgecolor="white", linewidth=0.5)

    # Annotate rho and p above each bar
    for i, (v, p) in enumerate(zip(values, pvals)):
        ax.text(
            xs[i],
            max(v, 0) + 0.025,
            f"rho={v:+.2f}\np={p:.2f}",
            ha="center",
            fontsize=9,
            color="#333333",
        )

    ax.axhline(0, color="#777777", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(-0.05, 0.6)
    ax.set_ylabel("Spearman rho vs post-SFT EM")
    set_title_subtitle(
        ax,
        "No base-model predictor reaches significance at n = 18",
        subtitle="Best is literal-cosine (rho = +0.41, p = 0.09); #404's reported +0.75 at n = 7 does not survive.",
    )
    savefig_paper(fig, "issue_458/predictor_comparison", dir=str(FIG_DIR))
    plt.close(fig)

    print("Saved 4 figures (hero, hero_raw, spectrum, predictor_comparison).")


if __name__ == "__main__":
    main()
