#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, E501
"""Issue #658 clean-result figures (blog style). Reads the fit JSONs + E0 table.

Intentional Unicode (ρ, ×, –, →, ※) in plot labels + docstrings.

Produces, under figures/issue_658/:
  - a32_rho_vs_layer       : A3.2 held-out ρ vs layer per behavior (mean summary), noise-floor band
  - marker_e0_saturation   : marker E0 logP per context (sorted), with the floor annotation
  - a34_a35_ridge_vs_mlp   : A3.4 linear-M cos vs A3.5 MLP cos (c_C -> v0), + shuffle null
  - dual_dv_rate_vs_logp   : E0 judged-rate vs logp_pos_mean (dual-DV validation), where dynamic range
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

EVAL = PROJECT_ROOT / "eval_results" / "issue_658"
FIGDIR = PROJECT_ROOT / "figures" / "issue_658"

# plain-English behavior labels
COL_LABEL = {
    "broad_em": "broad misalignment",
    "harmful_compliance": "harmful compliance",
    "sycophancy": "sycophancy",
    "deception": "deception",
    "refusal": "refusal",
    "fact_expression": "fact expression",
    "marker": "hidden marker (※)",
    "format_style": "list formatting",
    "self_report": "self-report",
    "persona_drift": "persona drift",
}


def _load(p):
    return json.load(open(p))


def fig_rho_vs_layer(a32, agg):
    set_paper_style("blog")
    cells = a32.get("a32", a32)
    cols = sorted(
        {c["column"] for c in cells if c.get("recipe") == "mean" and c.get("rho") is not None}
    )
    if not cols:
        return None
    fig, ax = plt.subplots()
    pal = paper_palette(max(len(cols), 3))
    for col, color in zip(cols, pal, strict=False):
        pts = sorted(
            [
                c
                for c in cells
                if c["column"] == col and c["recipe"] == "mean" and c.get("rho") is not None
            ],
            key=lambda c: c["layer"],
        )
        if pts:
            ax.plot(
                [p["layer"] for p in pts],
                [p["rho"] for p in pts],
                marker="o",
                ms=4,
                lw=1.4,
                label=COL_LABEL.get(col, col),
                color=color,
            )
    floor = agg.get("noise_floor", {}).get("p95")
    if floor is not None:
        ax.axhline(floor, ls="--", color="gray", lw=1.0, label=f"noise-floor p95 ({floor:.2f})")
    ax.axhline(0, ls=":", color="black", lw=0.8)
    ax.set_xlabel("layer (residual stream)")
    ax.set_ylabel("held-out Spearman ρ (predicted vs measured E0)")
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    set_title_subtitle(
        ax,
        "Can the activation summary predict base behavior?",
        "A3.2 MLP, mean-answer summary · 50 contexts, leave-one-context-out",
    )
    savefig_paper(fig, "issue_658/a32_rho_vs_layer", dir="figures/")
    plt.close(fig)
    return str(FIGDIR / "a32_rho_vs_layer.png")


def fig_marker_saturation(e0):
    set_paper_style("blog")
    rows = []
    for ctx, cols in e0["e0"].items():
        m = cols.get("marker")
        if m and m.get("logp_mean") is not None:
            rows.append((ctx, m["logp_mean"], m.get("emission_rate", 0.0)))
    if not rows:
        return None
    rows.sort(key=lambda r: r[1])
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(rows))
    is_primed = ["marker" in r[0] for r in rows]
    colors = [
        paper_palette_role("primary") if p else paper_palette_role("neutral") for p in is_primed
    ]
    ax.bar(x, vals, color=colors, width=0.85)
    ax.set_ylabel("base-model marker log P(※) at end-of-response slot (nats)")
    ax.set_xlabel("context (sorted by marker log-prob)")
    ax.set_xticks([])
    ax.axhline(np.log(0.5), ls="--", color="gray", lw=0.9)
    ax.annotate(
        "every context emits ※ at rate 0.000 — log P sits 12–27 nats below\nany realistic emission; the ranking orders near-zero probabilities",
        xy=(0.02, 0.06),
        xycoords="axes fraction",
        fontsize=7,
        color="#444",
    )
    set_title_subtitle(
        ax,
        "The hidden-marker probe is pinned at the floor",
        "base Qwen2.5-7B-Instruct · 50 contexts × 8 probes · marker-priming contexts highlighted",
    )
    savefig_paper(fig, "issue_658/marker_e0_saturation", dir="figures/")
    plt.close(fig)
    return str(FIGDIR / "marker_e0_saturation.png")


def fig_ridge_vs_mlp(agg):
    set_paper_style("blog")
    a34 = agg["a34_a35"]
    by_recipe = a34.get("by_recipe", {})
    chosen = a34.get("recipe_selection", {}).get("chosen_cc_recipe")
    rec = by_recipe.get(chosen) or next(iter(by_recipe.values()), {})
    pl = rec.get("per_layer", []) if isinstance(rec, dict) else []
    pl_mlp = [p for p in pl if p.get("mlp_mean_cos") is not None]
    if not pl_mlp:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    rc = [p["ridge_mean_cos_on_gap_dim"] for p in pl_mlp]
    mc = [p["mlp_mean_cos"] for p in pl_mlp]
    ax.scatter(rc, mc, color=paper_palette_role("primary"), s=36, zorder=3, label="per layer")
    # shuffle null mean
    sh = [s["ridge_mean_cos_shuffled"] for s in rec.get("shuffle_null", [])]
    lo = min(min(rc), min(mc)) - 0.05
    hi = max(max(rc), max(mc)) + 0.05
    ax.plot(
        [lo, hi], [lo, hi], ls="--", color="gray", lw=1.0, label="MLP = linear (no nonlinear gain)"
    )
    if sh:
        ax.axvline(
            float(np.mean(sh)),
            ls=":",
            color=paper_palette_role("control"),
            lw=1.0,
            label=f"shuffle null ({np.mean(sh):.2f})",
        )
    ax.set_xlabel("linear M (ridge) cosine, c_C → v0")
    ax.set_ylabel("MLP cosine, c_C → v0")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    set_title_subtitle(
        ax,
        "Does a cheap prompt-side vector predict the answer profile?",
        "A3.4/A3.5 · c_C → v0(C) held-out · each point one layer (strided)",
    )
    savefig_paper(fig, "issue_658/a34_a35_ridge_vs_mlp", dir="figures/")
    plt.close(fig)
    return str(FIGDIR / "a34_a35_ridge_vs_mlp.png")


def fig_dual_dv(e0):
    set_paper_style("blog")
    rates, logps, labels = [], [], []
    for cols in e0["e0"].values():
        for col, v in cols.items():
            if col in ("marker", "format_style"):
                continue
            if v.get("low_dynamic_range"):
                continue
            if v.get("rate") is not None and v.get("logp_pos_mean") is not None:
                rates.append(v["rate"])
                logps.append(v["logp_pos_mean"])
                labels.append(col)
    if len(rates) < 4:
        return None, len(rates)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.scatter(rates, logps, alpha=0.6, s=24, color=paper_palette_role("primary"))
    ax.set_xlabel("judged behavior rate (primary DV)")
    ax.set_ylabel("length-norm log P of judged-positive completions (secondary DV)")
    from scipy.stats import spearmanr

    r, _ = spearmanr(rates, logps)
    set_title_subtitle(
        ax,
        "Dual-DV companion tracks the judged rate",
        f"non-saturated cells only · Spearman ρ = {r:.2f}, n = {len(rates)}",
    )
    savefig_paper(fig, "issue_658/dual_dv_rate_vs_logp", dir="figures/")
    plt.close(fig)
    return str(FIGDIR / "dual_dv_rate_vs_logp.png"), len(rates)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    a32 = _load(EVAL / "a32_cells.json")
    agg = _load(EVAL / "aggregate.json")
    e0 = _load(EVAL / "E0_expression.json")
    made = []
    made.append(fig_rho_vs_layer(a32, agg))
    made.append(fig_marker_saturation(e0))
    made.append(fig_ridge_vs_mlp(agg))
    dd = fig_dual_dv(e0)
    made.append(dd[0] if isinstance(dd, tuple) else dd)
    print("FIGURES:", [m for m in made if m])


if __name__ == "__main__":
    main()
