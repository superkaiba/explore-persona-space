"""Figures for issue #509 clean-result body.

Three figures, one per finding:

1. anchors_vs_502.png          — #502 anchors vs #509 pre-registered anchor on both arms (hero).
2. layer_sweep.png             — per-layer x extraction-point profile (cosine, gauss_kl) per arm.
3. fact_prior_collapse.png     — fact-arm bake-off cells vs coarse predictors vs prior-controlled.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
FACT = REPO_ROOT / "eval_results/issue_509/fact_arm/scoring.json"
SYCO = REPO_ROOT / "eval_results/issue_509/syco_arm/scoring.json"
FIG_DIR = REPO_ROOT / "figures"


def _load() -> tuple[dict, dict]:
    return json.loads(FACT.read_text()), json.loads(SYCO.read_text())


def _cell(
    scoring: dict, *, ext: str, layer: int, metric: str, variant: str = "centered"
) -> dict | None:
    for c in scoring["cells"]:
        if (
            c["extraction_point"] == ext
            and c["layer"] == layer
            and c["metric"] == metric
            and c["variant"] == variant
        ):
            return c
    return None


# ---------------------------------------------------------------------------
# Figure 1 — anchors: #502 vs #509 pre-registered anchor on both arms (hero)
# ---------------------------------------------------------------------------
def figure_anchors() -> None:
    fact, syco = _load()

    fact_anchor = _cell(fact, ext="last_prompt", layer=22, metric="gauss_kl")
    syco_anchor = _cell(syco, ext="last_prompt", layer=22, metric="gauss_kl")

    p502_full = abs(syco["anchors"]["full_panel_rho_deltag"])
    p502_nonstyl = abs(syco["anchors"]["nonstylized_rho_deltag"])
    p509_fact_anchor = abs(fact_anchor["rho_fe_adj"])
    p509_syco_anchor = abs(syco_anchor["rho_fe_adj"])
    p509_fact_ridge = abs(fact["summary"]["ridge_L19_L24_mean_rho_fe_adj"])
    p509_syco_ridge = abs(syco["summary"]["ridge_L19_L24_mean_rho_fe_adj"])

    # Convert raw CI on syco anchor to error bars on |ρ_fe_adj|.
    raw_lo, raw_hi = syco_anchor["ci_lo_fe"], syco_anchor["ci_hi_fe"]
    abs_endpoints = sorted([abs(raw_lo), abs(raw_hi)])
    abs_lo, abs_hi = abs_endpoints[0], abs_endpoints[1]
    if raw_lo < 0 < raw_hi:
        abs_lo = 0.0
    err_neg = max(0.0, p509_syco_anchor - abs_lo)
    err_pos = max(0.0, abs_hi - p509_syco_anchor)

    labels = [
        "#502\nmarker leakage\n(240 pairs)",
        "#502\nmarker leakage\n(156 non-stylized)",
        "#509 fact\npre-reg anchor\n(smoke)",
        "#509 fact\nL19-L24 ridge\n(smoke)",
        "#509 syco\npre-reg anchor\n(prod)",
        "#509 syco\nL19-L24 ridge\n(prod)",
    ]
    values = [
        p502_full,
        p502_nonstyl,
        p509_fact_anchor,
        p509_fact_ridge,
        p509_syco_anchor,
        p509_syco_ridge,
    ]
    err_lower = [0, 0, 0, 0, err_neg, 0]
    err_upper = [0, 0, 0, 0, err_pos, 0]

    colors = [
        paper_palette_role("primary"),
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("baseline"),
        paper_palette_role("control"),
        paper_palette_role("control"),
    ]
    edge_hatch = [None, "//", None, "//", None, "//"]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        yerr=[err_lower, err_upper],
        capsize=4,
        error_kw={"elinewidth": 1.0},
    )
    for bar, hatch in zip(bars, edge_hatch):
        if hatch:
            bar.set_hatch(hatch)

    annotations = [
        f"|ρ|={p502_full:.2f}\nn=240",
        f"|ρ|={p502_nonstyl:.2f}\nn=156",
        f"|ρ|={p509_fact_anchor:.2f}\nn=25",
        f"|ρ|={p509_fact_ridge:.2f}\n18 cells",
        f"|ρ|={p509_syco_anchor:.2f}\np=0.30\nn=138",
        f"|ρ|={p509_syco_ridge:.2f}\n18 cells",
    ]
    for i, (v, label) in enumerate(zip(values, annotations)):
        top = v + (err_upper[i] if err_upper[i] else 0) + 0.025
        ax.text(i, top, label, ha="center", va="bottom", fontsize=9)

    ax.axhline(0.4, color="grey", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.text(
        len(values) - 0.5,
        0.41,
        "plan §6.2 trigger: |ρ| ≥ 0.40",
        color="grey",
        fontsize=8.5,
        ha="right",
        va="bottom",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"absolute length-partial Spearman $|\rho|$")
    ax.set_ylim(0, 0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    set_title_subtitle(
        ax,
        title="The marker-leakage predictor cell does not generalize to fact or sycophancy leakage",
        subtitle="At the pre-registered last-prompt × L22 × Gaussian-KL cell (and the L19-L24 ridge mean), both behaviors come back near zero.",
        source="#509 / scoring.json. Hatched bars = ridge means across the 18 L19-L24 cells. Sycophancy anchor carries a 5000-rep cluster bootstrap CI; cluster on source.",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_509/anchors_vs_502", dir=str(FIG_DIR))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — layer x extraction profile per arm, cosine + gauss_kl
# ---------------------------------------------------------------------------
def figure_layer_sweep() -> None:
    fact, syco = _load()
    layers = list(range(28))

    set_paper_style("blog")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey=True)

    arms = [
        ("Fact arm (smoke; n=18 per cell)", fact),
        ("Sycophancy arm (production; n=138 per cell)", syco),
    ]
    metrics = [
        ("cosine", "Cosine (the early-layer surprise)"),
        ("gauss_kl", "Gaussian-KL (the #502 winner family)"),
    ]
    ext_colors = {
        "end_of_system": paper_palette_role("primary"),
        "last_prompt": paper_palette_role("baseline"),
        "mean_response": paper_palette_role("control"),
    }
    ext_labels = {
        "end_of_system": "end of system prompt",
        "last_prompt": "last prompt token (the #502 extraction)",
        "mean_response": "mean over response",
    }

    for r, (arm_title, scoring) in enumerate(arms):
        for c, (metric, metric_title) in enumerate(metrics):
            ax = axes[r, c]
            for ext in ["end_of_system", "last_prompt", "mean_response"]:
                rhos, lows, highs = [], [], []
                for L in layers:
                    cell = _cell(scoring, ext=ext, layer=L, metric=metric)
                    if cell is None or cell.get("predictor_saturated"):
                        rhos.append(np.nan)
                        lows.append(np.nan)
                        highs.append(np.nan)
                        continue
                    rhos.append(cell["rho_fe_adj"])
                    if cell.get("ci_lo_fe") is not None:
                        lows.append(cell["ci_lo_fe"])
                        highs.append(cell["ci_hi_fe"])
                    else:
                        lows.append(np.nan)
                        highs.append(np.nan)
                rhos = np.array(rhos)
                ax.plot(
                    layers,
                    rhos,
                    marker="o",
                    markersize=3.4,
                    linewidth=1.4,
                    color=ext_colors[ext],
                    label=ext_labels[ext],
                )
                if not np.all(np.isnan(lows)):
                    ax.fill_between(
                        layers, lows, highs, color=ext_colors[ext], alpha=0.13, linewidth=0
                    )
            ax.axvspan(19, 24, color="grey", alpha=0.10, linewidth=0)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.axhline(-0.4, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.axhline(-0.581, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
            if r == 1 and c == 0:
                ax.text(0.5, -0.42, "|ρ|=0.40", color="grey", fontsize=8, va="top")
                ax.text(0.5, -0.60, "#502 non-styl. (-0.58)", color="grey", fontsize=8, va="top")
            if r == 0:
                ax.set_title(metric_title, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{arm_title}\nattenuation-adjusted ρ", fontsize=9.5)
            if r == 1:
                ax.set_xlabel("residual-stream layer")
            ax.set_ylim(-0.80, 0.40)
            ax.set_xlim(-0.5, 27.5)
            ax.grid(alpha=0.25)
            ax.set_axisbelow(True)

    axes[0, 1].legend(loc="upper right", fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(
        "On sycophancy, the signal lives at early-to-mid layers across both end-of-system AND last-prompt; the #502 ridge is flat",
        fontsize=11.5,
        x=0.5,
        y=0.995,
        weight="semibold",
    )
    fig.text(
        0.5,
        0.96,
        "Grey vertical band = L19-L24 ridge (the #502 winner band). Shading = 5000-rep cluster bootstrap 95% CI (syco only; fact arm in smoke). Dotted = -0.40 reference; dashed = -0.58 (#502 non-stylized anchor).",
        fontsize=9.5,
        ha="center",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig_paper(fig, "issue_509/layer_sweep", dir=str(FIG_DIR))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — fact arm: prior-control collapses the bake-off lift
# ---------------------------------------------------------------------------
def figure_fact_prior_collapse() -> None:
    fact, _ = _load()

    candidates = [
        c
        for c in fact["cells"]
        if not c.get("predictor_saturated") and c.get("rho_fe_adj") is not None
    ]
    best = max(candidates, key=lambda c: abs(c["rho_fe_adj"]))
    # Pull coarse-predictor anchors from the SEARCH-BEST cell's own coarse_lift,
    # so all 8 bars are on the same n=18 cells (no n=25 vs n=18 mixing).
    coarse = best["coarse_lift"]["per_coarse_rho_fe"]
    anchor = _cell(fact, ext="last_prompt", layer=22, metric="gauss_kl")
    rows = [
        ("cosine, layer 21\n(#494 coarse A)", abs(coarse["cosine_a_L21"]), "coarse"),
        ("cosine, layer 21\n(#494 coarse B)", abs(coarse["cosine_b_L21"]), "coarse"),
        ("next-token JS\non-topic (#494)", abs(coarse["js_on_topic"]), "coarse"),
        ("fact-slice JS\n(#494)", abs(coarse["fact_slice_js"]), "coarse"),
        ("bystander fact prior\n(#500)", abs(coarse["bystander_logprob"]), "coarse"),
        (
            "pre-registered cell\nlast-prompt L22\nGaussian-KL\n(n=25)",
            abs(anchor["rho_fe_adj"]),
            "anchor",
        ),
        (
            "search-best cell\nend-of-system L1\ncosine, centered",
            abs(best["rho_fe_adj"]),
            "best",
        ),
        (
            "search-best cell\nprior-controlled\n(+ substrate + #500)",
            abs(best["rho_double_fe"]),
            "best_pc",
        ),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    role_colors = {
        "coarse": paper_palette_role("control"),
        "anchor": paper_palette_role("baseline"),
        "best": paper_palette_role("primary"),
        "best_pc": paper_palette_role("accent"),
    }
    colors = [role_colors[r[2]] for r in rows]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(values):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5)

    ax.annotate(
        "",
        xy=(7, values[7] + 0.02),
        xytext=(6, values[6] + 0.02),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )
    ax.text(
        6.5,
        values[6] + 0.13,
        "controlling for the bystander's\nown prior on the fact (#500)\ncollapses the lift",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.axhline(0.4, color="grey", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.text(
        len(values) - 0.5,
        0.405,
        "plan §6.2 trigger: |ρ| ≥ 0.40",
        color="grey",
        fontsize=8.5,
        ha="right",
        va="bottom",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_ylabel(
        r"absolute length-partial Spearman $|\rho|$"
        + "\n(substrate-FE residualized; bars 1-5, 7, 8 = same n=18 cells; bar 6 = n=25)"
    )
    ax.set_ylim(0, 0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    set_title_subtitle(
        ax,
        title="On fact, the bake-off underperforms the coarse predictors and collapses under the prior control",
        subtitle="The search-best cell drops from |ρ|=0.67 to |ρ|=0.03 once the bystander's own prior on the fact is partialled out.",
        source=(
            "#509 fact-arm scoring.json (smoke; reliability_y=1.0). "
            "Coarse anchors (bars 1-5) computed on the same n=18 cells as the search-best (bar 7) "
            "via that cell's own coarse_lift. Bar 6 (pre-registered anchor) sits on n=25; "
            "all other bars on n=18 — different denominators, same panel."
        ),
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_509/fact_prior_collapse", dir=str(FIG_DIR))
    plt.close(fig)


if __name__ == "__main__":
    figure_anchors()
    figure_layer_sweep()
    figure_fact_prior_collapse()
    print("Wrote figures to", FIG_DIR / "issue_509")
