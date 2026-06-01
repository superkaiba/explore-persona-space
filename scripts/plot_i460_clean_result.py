"""Generate clean-result figures for #460 on-policy marker-at-end cross-transfer.

Hero figure: head-to-head scatter (D vs the headline DV), two panels
  - Left:  #406 off-policy binary emission rate (the dynamic-range regime)
  - Right: #460 on-policy trained log P(marker) (saturated at ~0)

Supporting:
  - g_logprob saturation histogram (off-diagonal cells)
  - delta_g vs -B_logprob ("base-prior") scatter showing delta_g IS base prior
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parents[1]
I460 = REPO / "eval_results" / "issue_460"
I406 = REPO / "eval_results" / "issue_406"
FIG_DIR = REPO / "figures"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_460_g_logprob():
    M = json.loads((I460 / "cross_eval" / "G_logprob_matrix.json").read_text())["G"]
    conds = sorted(M.keys())
    rows = []
    for ti in conds:
        for tj in conds:
            if ti == tj:
                continue
            c = M[ti][tj]
            rows.append(
                {
                    "T_i": ti,
                    "T_j": tj,
                    "g_logprob": c["g_logprob"],
                    "b_logprob": c["b_logprob"],
                    "delta_g": c["delta_g"],
                    "emission": c["emission_recompute_rate"],
                }
            )
    return rows, conds


def load_406_G_and_D():
    G406 = json.loads((I406 / "cross_eval" / "G_matrix.json").read_text())["G"]
    Draw = json.loads((I406 / "divergence" / "D_matrix.json").read_text())
    D = Draw["KL"]
    rows = []
    for ti in G406:
        for tj in G406[ti]:
            if ti == tj:
                continue
            rate = G406[ti][tj]["rate"]
            dval = D.get(ti, {}).get(tj)
            if dval is None:
                continue
            rows.append({"T_i": ti, "T_j": tj, "rate": rate, "D": dval})
    return rows


def join_460_with_D():
    g460, _ = load_460_g_logprob()
    Draw = json.loads((I406 / "divergence" / "D_matrix.json").read_text())
    D = Draw["KL"]
    for r in g460:
        r["D"] = D[r["T_i"]][r["T_j"]]
    return g460


# ---------------------------------------------------------------------------
# Hero: head-to-head scatter, D vs DV, two panels
# ---------------------------------------------------------------------------


def fig_head_to_head():
    set_paper_style("blog")
    rows_460 = join_460_with_D()
    rows_406 = load_406_G_and_D()
    # Restrict to the same 240 off-diagonal pairs (both runs share the 16 conds)
    pairs_460 = {(r["T_i"], r["T_j"]) for r in rows_460}
    pairs_406 = {(r["T_i"], r["T_j"]) for r in rows_406}
    shared = pairs_460 & pairs_406
    rows_460 = [r for r in rows_460 if (r["T_i"], r["T_j"]) in shared]
    rows_406 = [r for r in rows_406 if (r["T_i"], r["T_j"]) in shared]

    D_406 = np.array([r["D"] for r in rows_406])
    rate_406 = np.array([r["rate"] for r in rows_406])
    rho_406, p_406 = stats.spearmanr(D_406, rate_406)

    D_460 = np.array([r["D"] for r in rows_460])
    g_460 = np.array([r["g_logprob"] for r in rows_460])
    rho_460, p_460 = stats.spearmanr(D_460, g_460)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    # Panel A: #406 off-policy binary rate
    ax = axes[0]
    primary = paper_palette_role("baseline")
    ax.scatter(D_406, rate_406, s=22, alpha=0.55, color=primary, edgecolor="none")
    # Highlight exact-zero cells
    zmask = rate_406 == 0
    ax.scatter(
        D_406[zmask],
        rate_406[zmask],
        s=22,
        alpha=0.7,
        color=paper_palette_role("control"),
        edgecolor="none",
        label=f"emission = 0 ({zmask.sum()}/240 cells)",
    )
    ax.set_xlabel("Base-model divergence D (forward KL, nats)")
    ax.set_ylabel("Marker emission rate (binary, off-policy)")
    ax.set_ylim(-0.04, 1.02)
    ax.set_xlim(-0.1, max(D_460.max(), D_406.max()) * 1.05)
    ax.text(
        0.97,
        0.97,
        f"Spearman ρ = {rho_406:+.2f}  (p ≈ {p_406:.1e}, n=240)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaa", alpha=0.9),
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.85), fontsize=8, frameon=False)
    set_title_subtitle(
        ax,
        "Off-policy (binary, marker-first, canned answer)",
        "#406 measurement regime — wide dynamic range, 52% zero floor",
    )

    # Panel B: #460 on-policy trained log P(marker) — the transfer outcome,
    # but pinned at the ceiling. Annotate BOTH the saturated-band rank (ρ on
    # g_logprob, large but meaningless because all points sit within 1.3 nat
    # of 0) AND the headline H1 ρ on ΔG = trained − base (the transfer-
    # magnitude statistic that asks the divergence question; null at ρ ≈ −0.11).
    ax = axes[1]
    ax.scatter(D_460, g_460, s=22, alpha=0.55, color=primary, edgecolor="none")
    ax.axhline(0.0, color=paper_palette_role("neutral"), linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Base-model divergence D (forward KL, nats)")
    ax.set_ylabel("Trained log P(※ | response) (nats; 0 = prob 1)")
    # Force y-axis to show how tight the band is
    ax.set_ylim(-1.4, 0.15)
    ax.set_xlim(-0.1, max(D_460.max(), D_406.max()) * 1.05)
    # Headline H1 rho (length-partial), from analysis.json. Pull it directly.
    h1 = json.loads((I460 / "analysis.json").read_text())["H1_length_partial_delta_vs_D"]
    rho_h1 = h1["rho_pingouin"]
    p_h1 = h1["p_pingouin"]
    # Place the annotation INSIDE the chart, anchored to the empty middle band
    # (cells cluster at the top — there's room mid-y for a callout).
    ax.text(
        0.50,
        0.55,
        "99% of off-diagonal cells\n"
        "within 0.1 nat of ceiling.\n"
        "\n"
        "Headline H1 statistic\n"
        "(ΔG = trained − base log P(※)):\n"
        f"Spearman ρ = {rho_h1:+.2f}\n"
        f"p = {p_h1:.2f}, n = 240  → null",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#aaa", alpha=0.95),
    )
    set_title_subtitle(
        ax,
        "On-policy (continuous, marker-at-end, model's own response)",
        "#460 measurement regime — pinned at the marker = prob 1 ceiling",
    )

    savefig_paper(fig, "issue_460/head_to_head_off_vs_on_policy", dir=str(FIG_DIR))
    plt.close(fig)
    return rho_406, p_406, rho_460, p_460


# ---------------------------------------------------------------------------
# Saturation histogram: g_logprob off-diagonal distribution
# ---------------------------------------------------------------------------


def fig_saturation_hist():
    set_paper_style("blog")
    rows, _ = load_460_g_logprob()
    g = np.array([r["g_logprob"] for r in rows])

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), constrained_layout=True)
    bins = np.linspace(-1.4, 0.1, 76)
    ax.hist(g, bins=bins, color=paper_palette_role("primary"), edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color=paper_palette_role("neutral"), linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axvline(-0.1, color=paper_palette_role("control"), linestyle=":", linewidth=0.9, alpha=0.7)

    within = (g >= -0.1).sum()
    ax.text(
        0.03,
        0.97,
        f"n = 240 off-diagonal cells\n"
        f"{within}/240 = {100 * within / 240:.1f}% within 0.1 nat of 0\n"
        f"mean = {g.mean():+.3f}, sd = {g.std():.3f}\n"
        f"min = {g.min():+.2f}, max = {g.max():+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaa", alpha=0.95),
    )
    ax.set_xlabel("Trained log P(※ | T_j prompt + base on-policy response)  (nats)")
    ax.set_ylabel("Off-diagonal cells")
    ax.set_xlim(-1.45, 0.1)
    set_title_subtitle(
        ax,
        "On-policy marker probability is pinned at the ceiling on 99% of pairs",
        "the dependent variable has no dynamic range to predict",
    )
    savefig_paper(fig, "issue_460/saturation_g_logprob_hist", dir=str(FIG_DIR))
    plt.close(fig)


# ---------------------------------------------------------------------------
# delta_g IS base prior: delta_g vs -b_logprob
# ---------------------------------------------------------------------------


def fig_delta_is_base_prior():
    set_paper_style("blog")
    rows, _ = load_460_g_logprob()
    dg = np.array([r["delta_g"] for r in rows])
    minus_b = np.array([-r["b_logprob"] for r in rows])  # base-prior surprise of marker

    rho, p = stats.spearmanr(dg, minus_b)

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.4), constrained_layout=True)
    ax.scatter(minus_b, dg, s=20, alpha=0.55, color=paper_palette_role("primary"), edgecolor="none")
    # y=x reference line
    lo = min(dg.min(), minus_b.min())
    hi = max(dg.max(), minus_b.max())
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=paper_palette_role("neutral"),
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
        label="y = x",
    )

    ax.set_xlabel("Base-model surprise of the marker  (−base log P(※), nats)")
    ax.set_ylabel("ΔG = trained − base log P(※)  (nats)")
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ = {rho:.3f}  (p ≈ {p:.0e}, n=240)\n"
        "ΔG variance is the base-prior variance,\n"
        "not transfer.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaa", alpha=0.95),
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    set_title_subtitle(
        ax,
        "ΔG ≈ −base log P(※) because trained log P(※) ≈ 0",
        "the headline 'transfer magnitude' is mostly the base model's prior surprise",
    )
    savefig_paper(fig, "issue_460/delta_g_is_base_prior", dir=str(FIG_DIR))
    plt.close(fig)
    return rho, p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("[1/3] head-to-head scatter")
    h406, p406, h460, p460 = fig_head_to_head()
    print(f"     #406 off-policy: ρ = {h406:+.3f}  p = {p406:.2e}")
    print(f"     #460 on-policy:  ρ = {h460:+.3f}  p = {p460:.3f}")
    print("[2/3] saturation histogram")
    fig_saturation_hist()
    print("[3/3] delta_g IS base prior")
    rho, p = fig_delta_is_base_prior()
    print(f"     ρ(ΔG, −B) = {rho:+.3f}  p = {p:.2e}")
    print("done")


if __name__ == "__main__":
    main()
