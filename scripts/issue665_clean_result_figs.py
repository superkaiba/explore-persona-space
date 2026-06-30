#!/usr/bin/env python
"""Issue #665 clean-result figures (analyzer, honest framing).

Four purpose-built figures with CIs, plain-English labels, and the FDR / floor
context that the central honest read needs:
  fig1_a310_gate_vs_floor   — base gate rho vs probe-split floor + FDR verdict (HERO)
  fig2_a310_partial          — raw vs base-prior/install partial (the C7/C10 control)
  fig3_a39_key_metric        — whitened c_C (the pre-registered primary key) vs identity
  fig4_a36c_causal_patch     — A3.6c real CV patch vs floor controls + self-nulls
"""

from __future__ import annotations

import collections
import glob
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

EVAL = Path(__file__).resolve().parents[1] / "eval_results" / "issue_665"
FIGDIR = Path(__file__).resolve().parents[1] / "figures"
AGG = json.load(open(EVAL / "aggregate.json"))
PB = AGG["per_behavior"]
FDR = AGG["fdr"]

BEH = ["bad_medical", "em", "fact"]
BEH_LABEL = {
    "bad_medical": "bad-medical advice\n(read layer 8)",
    "em": "emergent misalignment\n(read layer 0)",
    "fact": "taught fact\n(read layer 2)",
}

set_paper_style("blog")


# ── fig1: base gate rho (clustered CI) vs probe-split reliability floor + FDR ──
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(BEH))
    rho = [PB[b]["a310_g0_spearman"]["mean"] for b in BEH]
    lo = [PB[b]["a310_g0_spearman"]["ci_lo"] for b in BEH]
    hi = [PB[b]["a310_g0_spearman"]["ci_hi"] for b in BEH]
    floor = [PB[b]["probe_split_floor_mean"] for b in BEH]
    yerr = np.clip(
        np.array([[r - l for r, l in zip(rho, lo)], [h - r for h, r in zip(rho, hi)]]),
        0.0,
        None,
    )
    ax.bar(
        x,
        rho,
        width=0.55,
        color=paper_palette_role("primary"),
        label="base-gate Spearman ρ(ĝ_real, g0)",
    )
    ax.errorbar(
        x,
        rho,
        yerr=yerr,
        fmt="none",
        ecolor="#1A1A1A",
        elinewidth=1.4,
        capsize=5,
        capthick=1.4,
        zorder=5,
    )
    # probe-split reliability floor as a marker per behavior
    ax.scatter(
        x,
        floor,
        marker="_",
        s=900,
        color=paper_palette_role("accent"),
        linewidths=2.4,
        zorder=6,
        label="probe-split reliability floor",
    )
    for xi, (b, r) in enumerate(zip(BEH, rho)):
        rej = FDR["reject"][f"a310_{b}"]
        p = FDR["pvalues"][f"a310_{b}"]
        ax.annotate(
            f"BH-FDR p={p:.2f}\n{'reject' if rej else 'NOT rejected'}",
            (xi, hi[xi] + 0.02),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#5A5A5A",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([BEH_LABEL[b] for b in BEH])
    ax.set_ylabel("Spearman ρ vs realized gate ĝ_real")
    ax.set_ylim(0, 0.56)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    set_title_subtitle(
        ax,
        "Does the base-model gate predict the realized FT gate?",
        "Clustered 95% CI (n_clusters=2 families); BH-FDR rejects none at α=0.05",
        source="issue #665 A3.10 · 8 cells/behavior",
    )
    savefig_paper(fig, "issue_665/fig1_a310_gate_vs_floor", dir=str(FIGDIR))
    plt.close(fig)


# ── fig2: raw rho vs base-prior/install partial ──
def fig2():
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(BEH))
    raw = [PB[b]["A3_10_rho_raw"] for b in BEH]
    par = [PB[b]["A3_10_rho_partial_E0_wnorm"] for b in BEH]
    w = 0.38
    ax.bar(x - w / 2, raw, w, color=paper_palette_role("neutral"), label="raw ρ(ĝ_real, g0)")
    ax.bar(
        x + w / 2,
        par,
        w,
        color=paper_palette_role("primary"),
        label="partial — base prior E0 + install ‖ŵ‖ removed",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([BEH_LABEL[b] for b in BEH])
    ax.set_ylabel("Spearman ρ vs realized gate ĝ_real")
    ax.set_ylim(0, 0.33)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    set_title_subtitle(
        ax,
        "Controlling for base prior + install strength does not kill the gate signal",
        "Partial ρ ≥ raw ρ; base-prior artifacts remain possible "
        "(suppressor/collider effects not ruled out)",
        source="issue #665 A3.10 · 400 pooled contexts/behavior",
    )
    savefig_paper(fig, "issue_665/fig2_a310_partial", dir=str(FIGDIR))
    plt.close(fig)


# ── fig3: A3.9 whitened c_C (primary key) vs identity, per behavior ──
def fig3():
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    cos = collections.defaultdict(list)
    for f in sorted(glob.glob(str(EVAL / "a39" / "*.json"))):
        d = json.load(open(f))
        b = d["behavior"]
        L = str(d["read_layer"])
        bl = d["by_layer"][L]
        cos[b].append(bl["cosine_spearman"])
        for km, res in bl["key_metric_results"].items():
            sp = res.get("spearman") if isinstance(res, dict) else res
            agg[b][km].append(sp)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(BEH))
    w = 0.26
    prim = [float(np.nanmean(agg[b]["c_C::Sigma_inv"])) for b in BEH]  # pre-registered
    ident = [float(np.nanmean(agg[b]["c_C::I"])) for b in BEH]  # un-whitened
    cosb = [float(np.nanmean(cos[b])) for b in BEH]  # raw cosine
    ax.bar(
        x - w,
        prim,
        w,
        color=paper_palette_role("primary"),
        label="pre-registered whitened key  c_C · Σc⁻¹",
    )
    ax.bar(x, ident, w, color=paper_palette_role("baseline"), label="un-whitened key  c_C · I")
    ax.bar(x + w, cosb, w, color=paper_palette_role("neutral"), label="raw cosine baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([BEH_LABEL[b] for b in BEH])
    ax.set_ylabel("Spearman ρ(predicted gate, ĝ_real)")
    ax.set_ylim(0, 0.55)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    set_title_subtitle(
        ax,
        "The pre-registered whitened gate loses to the simplest un-whitened key",
        "Σc⁻¹ whitening never wins (verdict-ii frac ≤ 0.125) — exploratory key sweep",
        source="issue #665 A3.9 · 8 cells/behavior",
    )
    savefig_paper(fig, "issue_665/fig3_a39_key_metric", dir=str(FIGDIR))
    plt.close(fig)


# ── fig4: A3.6c causal context-vector patch vs floor controls ──
def fig4():
    cells = {
        "bm_default_contra_d2_seed42": "bad-medical (default, contrastive)",
        "bm_default_posonly_d2_seed42": "bad-medical (default, positive-only)",
        "bm_librarian_contra_d2_seed42": "bad-medical (librarian, contrastive)",
        "tf_default_contra_d2_seed42": "taught fact (default, contrastive)",
    }
    variant_label = {
        "self_c0": "base-CV null (θ0)",
        "p_up": "trained CV → base (real patch)",
        "self_cp": "trained-CV null (θ+)",
        "p_down": "base CV → trained (real patch)",
        "random_cv": "random-CV floor",
        "norm_matched": "norm-matched floor",
    }
    order = ["self_c0", "p_up", "self_cp", "p_down", "random_cv", "norm_matched"]
    colors = {
        "self_c0": "#BBBBBB",
        "self_cp": "#888888",
        "p_up": paper_palette_role("primary"),
        "p_down": paper_palette_role("control"),
        "random_cv": paper_palette_role("accent"),
        "norm_matched": "#E69F00",
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ncells = len(cells)
    nvar = len(order)
    gw = 0.8
    bw = gw / nvar
    xc = np.arange(ncells)
    for vi, v in enumerate(order):
        ys = []
        for c in cells:
            d = json.load(open(EVAL / "a36c" / f"{c}.json"))
            vals = [r["f_cv_v"] for r in d["rows"] if r["variant"] == v and r["f_cv_v"] is not None]
            ys.append(float(np.mean(vals)))
        ax.bar(
            xc - gw / 2 + bw * (vi + 0.5),
            ys,
            bw,
            color=colors[v],
            label=variant_label[v] if vi < nvar else None,
        )
    ax.axhline(1.0, color="#5A5A5A", lw=0.9, ls=":")
    ax.axhline(0.0, color="#5A5A5A", lw=0.9, ls=":")
    ax.set_xticks(xc)
    ax.set_xticklabels(list(cells.values()), fontsize=8.5, rotation=45, ha="right")
    ax.set_ylabel("patch effect on activation\n(1 = moved to FT profile, 0 = at base)")
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.42))
    set_title_subtitle(
        ax,
        "The real context-vector patch does not separate from the floor controls",
        "f_CV = +0.023 = mean over (context, layer, scope) of [p_up − self_c0]; "
        "P↑ moves to the base-CV null, P↓ falls to the random/norm-matched floor — no input localization",
        source="issue #665 A3.6c · 9 ctx × 3 layers × 2 scopes/cell, parity gate PASS",
    )
    savefig_paper(fig, "issue_665/fig4_a36c_causal_patch", dir=str(FIGDIR))
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("done")
