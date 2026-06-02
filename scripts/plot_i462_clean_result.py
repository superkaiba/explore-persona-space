"""Figures for #462 clean result — epoch-resolved on-policy marker transfer.

Two figures:
  - hero: rho(D, *) vs epoch (raw + base-subtracted) overlaid with saturation
    fraction; the story is "ep1 has dynamic range so rho is real, ep2+ saturates
    so both rho variants become uninformative or wash out."
  - ep1 scatter: divergence vs trained log P(marker) at the unsaturated epoch,
    showing the real negative relationship. RAW alongside per the spec.
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
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = json.loads((REPO_ROOT / "eval_results/issue_462/analysis.json").read_text())


def _flat_for_epoch(ep: int):
    """Build flat (D, g_logprob, delta_g) arrays for one epoch."""
    M = json.loads(
        (REPO_ROOT / f"eval_results/issue_462/cross_eval/G_logprob_matrix_ep{ep}.json").read_text()
    )
    D406 = json.loads((REPO_ROOT / "eval_results/issue_406/divergence/D_matrix.json").read_text())
    KL = D406["KL"]
    conditions = [c["cid"] for c in D406["conditions"]]
    G = M["G"]
    D, g, dg = [], [], []
    for ti in conditions:
        for tj in conditions:
            if ti == tj:
                continue
            D.append(KL[ti][tj])
            g.append(G[ti][tj]["g_logprob"])
            dg.append(G[ti][tj]["delta_g"])
    return np.asarray(D), np.asarray(g), np.asarray(dg)


def plot_hero():
    """Two-panel hero: rho(D, *) vs epoch (left) and saturation fraction vs epoch (right).

    Twin-axis was tried first and broke layout; split into side-by-side panels so
    the rho lines and the saturation curve each get their own clean axis. The
    story is identical: ep1 has dynamic range so both rho variants are real, ep2+
    saturates so the raw rho becomes uninformative and the base-subtracted one
    collapses to null.
    """
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, (ax_rho, ax_sat) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    rho_curve = ANALYSIS["rho_vs_epoch"]
    sat_curve = ANALYSIS["saturation_frac_vs_epoch"]
    epochs = np.asarray([r["epoch"] for r in rho_curve], dtype=float)

    rho_g = np.asarray([r["rho_D_glogprob"] for r in rho_curve])
    rho_g_lo = np.asarray([r["rho_D_glogprob_ci_low"] for r in rho_curve])
    rho_g_hi = np.asarray([r["rho_D_glogprob_ci_high"] for r in rho_curve])

    rho_dg = np.asarray([r["rho_D_deltag"] for r in rho_curve])
    rho_dg_lo = np.asarray([r["rho_D_deltag_ci_low"] for r in rho_curve])
    rho_dg_hi = np.asarray([r["rho_D_deltag_ci_high"] for r in rho_curve])

    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")
    accent = paper_palette_role("accent")
    neutral = paper_palette_role("neutral")

    # LEFT: rho-vs-epoch with bootstrap CIs
    ax_rho.axhline(0, color=neutral, lw=1, zorder=1)
    ax_rho.errorbar(
        epochs - 0.06,
        rho_g,
        yerr=[rho_g - rho_g_lo, rho_g_hi - rho_g],
        fmt="o-",
        color=primary,
        ecolor=primary,
        elinewidth=1.4,
        capsize=3,
        markersize=7,
        label="rho(D, raw on-policy log P(marker))",
        zorder=3,
    )
    ax_rho.errorbar(
        epochs + 0.06,
        rho_dg,
        yerr=[rho_dg - rho_dg_lo, rho_dg_hi - rho_dg],
        fmt="s-",
        color=baseline,
        ecolor=baseline,
        elinewidth=1.4,
        capsize=3,
        markersize=7,
        label="rho(D, base-subtracted = trained - base)",
        zorder=3,
    )
    ax_rho.set_xticks([1, 2, 3, 5])
    ax_rho.set_xticklabels(["ep1", "ep2", "ep3", "ep5"])
    ax_rho.set_xlabel("LoRA adapter checkpoint (training epochs)")
    ax_rho.set_ylabel("Length-partial Spearman rho (n=240, 95% CI)")
    ax_rho.set_ylim(-0.95, 0.20)
    ax_rho.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax_rho.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax_rho.set_title("Divergence vs on-policy marker transfer", fontsize=10.5, loc="left", pad=10)

    # Annotate the binding-constraint point
    ax_rho.annotate(
        "ep1 is the only checkpoint\nwith real dynamic range\n(see right panel)",
        xy=(1, rho_dg[0]),
        xytext=(2.0, -0.42),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=neutral, lw=0.9),
        color=neutral,
        ha="left",
    )

    # RIGHT: saturation fraction vs epoch
    sat_frac = np.asarray([s["frac_within_0_1_of_zero"] for s in sat_curve])
    ax_sat.plot(
        epochs,
        sat_frac,
        marker="^",
        linestyle="-",
        color=accent,
        markersize=8,
        linewidth=1.8,
        zorder=3,
    )
    # Annotate each point with its value
    for ep, sf in zip(epochs, sat_frac):
        ax_sat.annotate(
            f"{sf * 100:.1f}%",
            xy=(ep, sf),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=accent,
        )
    ax_sat.axhline(0.95, color=neutral, lw=0.8, linestyle=":")
    ax_sat.text(
        5.0,
        0.945,
        "95% saturation band",
        ha="right",
        va="top",
        fontsize=8,
        color=neutral,
    )
    ax_sat.set_xticks([1, 2, 3, 5])
    ax_sat.set_xticklabels(["ep1", "ep2", "ep3", "ep5"])
    ax_sat.set_xlabel("LoRA adapter checkpoint (training epochs)")
    ax_sat.set_ylabel("Fraction of 240 cells within 0.1 nat of ceiling")
    ax_sat.set_ylim(0.70, 1.05)
    ax_sat.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax_sat.set_title("Marker log-prob saturation across cells", fontsize=10.5, loc="left", pad=10)

    fig.suptitle(
        "Divergence predicts on-policy marker transfer only at the one unsaturated checkpoint",
        fontsize=11,
        x=0.085,
        y=0.985,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.085,
        0.93,
        "Left: rho(D, transfer DV) by training amount, raw vs base-subtracted. Right: how saturated the on-policy log-prob is. By ep2 only the unsaturated-cell rank-shuffle remains.",
        fontsize=8.5,
        color=neutral,
        ha="left",
    )
    fig.subplots_adjust(top=0.83, bottom=0.13, left=0.085, right=0.97, wspace=0.30)
    savefig_paper(fig, "issue_462/hero_rho_vs_epoch", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def plot_ep1_scatter():
    """ep1 scatter: D vs g_logprob (left) and D vs delta_g (right), the unsaturated regime."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    D, g, dg = _flat_for_epoch(1)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")
    neutral = paper_palette_role("neutral")

    # Left: D vs g_logprob (raw on-policy log-prob)
    ax = axes[0]
    ax.scatter(D, g, s=22, c=primary, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=neutral, lw=0.8, linestyle=":")
    ax.set_xlabel("Base-model forward-KL divergence D (nats)")
    ax.set_ylabel("On-policy log P(marker) at epoch 1 (nats)")
    rho1 = ANALYSIS["per_level"]["ep1"]["partial_rho_D_G_logprob"]
    ax.text(
        0.98,
        0.04,
        f"length-partial rho = {rho1['rho_pingouin']:+.2f}\n"
        f"95% CI [{rho1['bootstrap_ci_2_5']:+.2f}, {rho1['bootstrap_ci_97_5']:+.2f}]\n"
        f"p = {rho1['p_pingouin']:.1e}  (n = {rho1['n']})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=neutral, alpha=0.9),
    )
    ax.set_title("Raw on-policy log P(marker)", fontsize=10, loc="left")
    ax.grid(alpha=0.2, linewidth=0.5)

    # Right: D vs delta_g (base-subtracted)
    ax = axes[1]
    ax.scatter(D, dg, s=22, c=baseline, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Base-model forward-KL divergence D (nats)")
    ax.set_ylabel("Base-subtracted = trained - base log P(marker) at epoch 1 (nats)")
    rho1d = ANALYSIS["per_level"]["ep1"]["partial_rho_D_delta_g"]
    ax.text(
        0.98,
        0.04,
        f"length-partial rho = {rho1d['rho_pingouin']:+.2f}\n"
        f"95% CI [{rho1d['bootstrap_ci_2_5']:+.2f}, {rho1d['bootstrap_ci_97_5']:+.2f}]\n"
        f"p = {rho1d['p_pingouin']:.1e}  (n = {rho1d['n']})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=neutral, alpha=0.9),
    )
    ax.set_title("Base-subtracted (trained - base)", fontsize=10, loc="left")
    ax.grid(alpha=0.2, linewidth=0.5)

    fig.suptitle(
        "Epoch 1 (the unsaturated checkpoint): divergence really does predict on-policy marker transfer",
        fontsize=11,
        x=0.085,
        y=0.985,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.085,
        0.93,
        "240 off-diagonal cells (16x16 cross-eval, on-diagonal excluded). Both panels: more-divergent transformation pairs transfer less.",
        fontsize=8.5,
        color=neutral,
        ha="left",
    )
    fig.subplots_adjust(top=0.83, bottom=0.13, left=0.085, right=0.97, wspace=0.34)
    savefig_paper(fig, "issue_462/ep1_scatter_d_vs_transfer", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def plot_ep1_vs_ep5_distribution():
    """Raw distribution check: histogram of g_logprob at ep1 vs ep5 to show the saturation collapse.

    This is the 'raw alongside processed' for the hero figure — the hero plots the
    saturation-fraction summary; this shows the distribution it summarizes."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    D1, g1, _ = _flat_for_epoch(1)
    D5, g5, _ = _flat_for_epoch(5)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    primary = paper_palette_role("primary")
    accent = paper_palette_role("accent")
    neutral = paper_palette_role("neutral")

    bins = np.linspace(-7.5, 0.3, 40)

    axes[0].hist(g1, bins=bins, color=primary, alpha=0.85, edgecolor="white", linewidth=0.5)
    axes[0].axvline(-0.1, color=neutral, lw=0.8, linestyle=":")
    axes[0].set_xlabel("On-policy log P(marker), nats")
    axes[0].set_ylabel("# off-diagonal cells (out of 240)")
    axes[0].set_title("Epoch 1 (unsaturated)", fontsize=10, loc="left")
    axes[0].text(
        0.98,
        0.92,
        "sd = 0.73\n76% within 0.1 nat of 0\nmin = -7.18",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=neutral, alpha=0.9),
    )
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].hist(g5, bins=bins, color=accent, alpha=0.85, edgecolor="white", linewidth=0.5)
    axes[1].axvline(-0.1, color=neutral, lw=0.8, linestyle=":")
    axes[1].set_xlabel("On-policy log P(marker), nats")
    axes[1].set_title("Epoch 5 (saturated, same recipe as #460)", fontsize=10, loc="left")
    axes[1].text(
        0.98,
        0.92,
        "sd = 0.07\n99% within 0.1 nat of 0\nmin = -1.11",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=neutral, alpha=0.9),
    )
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle(
        "The saturation collapse: epoch 1 spans 7 nats of range; by epoch 5 everything piles at the ceiling",
        fontsize=11,
        x=0.085,
        y=0.985,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.085,
        0.93,
        "Raw distribution of trained log P(marker) across the 240 off-diagonal cells. The dotted line marks the 0.1-nat 'within ceiling' band the saturation gate uses.",
        fontsize=8.5,
        color=neutral,
        ha="left",
    )
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.085, right=0.97, wspace=0.10)
    savefig_paper(fig, "issue_462/raw_distribution_ep1_vs_ep5", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    plot_hero()
    plot_ep1_scatter()
    plot_ep1_vs_ep5_distribution()
    print("Done. Figures saved to figures/issue_462/")
