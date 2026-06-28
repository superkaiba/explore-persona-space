"""Clean-result figures for issue #658 (leakage-predictor base-model foundation).

Reads the analyzer-prepared per-behavior summary (eval_results/issue_658/
analyzer_body_data.json), which holds, per arm (Betley / G1-UltraChat):
  - A3.2 best structured-summary ρ + per-behavior noise floor + FDR + PASS
  - A3.2 random-projection control (attn) best ρ on its own grid + PASS
  - A3.3 best linear read-out ρ + MLP ceiling + PASS

Figures:
  1. fig_a32_verdict        — hero: A3.2 best structured ρ vs per-behavior noise
                              floor, both genres; only 3/10 clear.
  2. fig_random_proj_control — structured-best vs random-projection-best ρ per
                              behavior; random clears 0-1/10 on its own grid.
  3. fig_a33_readout         — A3.3 linear read-out ρ for the 4 r_B behaviors,
                              both genres; all 4 PASS.
"""

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

DATA = json.load(open("eval_results/issue_658/analyzer_body_data.json"))
OUT = "issue_658"

# Plain-English behavior labels (no internal slugs in reader-facing text).
LABELS = {
    "broad_em": "broad misalignment",
    "harmful_compliance": "harmful compliance",
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "deception": "deception",
    "fact_expression": "fact expression",
    "marker": "marker",
    "format_style": "format/style",
    "self_report": "self-report",
    "persona_drift": "persona drift",
}
ORDER = [
    "broad_em",
    "harmful_compliance",
    "sycophancy",
    "refusal",
    "deception",
    "fact_expression",
    "marker",
    "format_style",
    "self_report",
    "persona_drift",
]
ARM_LABEL = {"betley": "misalignment-specific (Betley)", "g1": "generic (UltraChat)"}


def fig_a32_verdict():
    """Hero: per-behavior best structured-summary ρ vs its own noise floor, both genres."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.16, right=0.97, wspace=0.12)
    c_pass = paper_palette_role("primary")
    c_fail = paper_palette_role("neutral")
    c_floor = paper_palette_role("baseline")
    for ax, arm in zip(axes, ("betley", "g1"), strict=True):
        a32 = DATA[arm]["a32"]
        y = np.arange(len(ORDER))[::-1]
        rhos = [a32[c]["struct_rho"] for c in ORDER]
        floors = [a32[c]["nf95"] for c in ORDER]
        passes = [a32[c]["struct_pass"] for c in ORDER]
        colors = [c_pass if p else c_fail for p in passes]
        # noise floor markers
        ax.scatter(
            floors,
            y,
            marker="|",
            s=260,
            color=c_floor,
            linewidths=2.0,
            label="noise floor (p95)",
            zorder=2,
        )
        ax.scatter(rhos, y, s=64, color=colors, zorder=3, edgecolors="white", linewidths=0.8)
        for yi, c in zip(y, ORDER, strict=True):
            tag = "PASS" if a32[c]["struct_pass"] else ""
            if tag:
                ax.text(
                    a32[c]["struct_rho"] + 0.02,
                    yi,
                    tag,
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color=c_pass,
                    fontweight="bold",
                )
        ax.axvline(0.0, color="0.75", lw=0.8, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[c] for c in ORDER])
        ax.set_xlabel("held-out Spearman ρ (best structured summary)")
        ax.set_xlim(-0.5, 1.0)
        ax.text(
            0.50,
            0.02,
            ARM_LABEL[arm],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold",
            color="0.25",
        )
    axes[0].legend(loc="lower left", frameon=False, fontsize=8)
    fig.text(
        0.16,
        0.985,
        "A3.2 holds for only 3 of 10 behaviors, in both query genres",
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="semibold",
    )
    fig.text(
        0.16,
        0.935,
        "mean answer-side activation → behavior expression; a point right of its noise"
        " floor (and FDR-significant) PASSes",
        ha="left",
        va="top",
        fontsize=9.0,
        color="0.35",
    )
    fig.text(
        0.16,
        0.025,
        "issue #658 · Qwen2.5-7B-Instruct · 50 contexts · n=50",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="0.5",
        style="italic",
    )
    savefig_paper(fig, f"{OUT}/fig_a32_verdict", dir="figures/")
    plt.close(fig)


def fig_random_proj_control():
    """Structured-best vs random-projection-best ρ per behavior, both genres (Betley shown;
    G1 as second panel). Random projection clears 0-1/10 on its own FDR grid."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.16, right=0.97, wspace=0.12)
    c_struct = paper_palette_role("primary")
    c_rand = paper_palette_role("control")
    for ax, arm in zip(axes, ("betley", "g1"), strict=True):
        a32 = DATA[arm]["a32"]
        attn_grid = DATA[arm]["a32_attn_only_grid"]
        y = np.arange(len(ORDER))[::-1]
        struct = [a32[c]["struct_rho"] for c in ORDER]
        rand = [attn_grid[c]["rho"] for c in ORDER]
        ax.scatter(
            struct,
            y,
            s=58,
            color=c_struct,
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
            label="structured summary (best of mean/last/max-pool)",
        )
        ax.scatter(
            rand,
            y,
            s=58,
            color=c_rand,
            marker="D",
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
            label="random projection (unfitted control)",
        )
        for yi, s, r in zip(y, struct, rand, strict=True):
            ax.plot([min(s, r), max(s, r)], [yi, yi], color="0.8", lw=1.0, zorder=1)
        ax.axvline(0.0, color="0.75", lw=0.8, zorder=1)
        npass_r = sum(1 for c in ORDER if attn_grid[c]["pass"])
        npass_s = sum(1 for c in ORDER if a32[c]["struct_pass"])
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[c] for c in ORDER])
        ax.set_xlabel("best held-out Spearman ρ over 28 layers")
        ax.set_xlim(-0.5, 1.0)
        ax.text(
            0.50,
            0.02,
            f"{ARM_LABEL[arm]}\nstructured PASS {npass_s}/10 · random PASS {npass_r}/10",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="semibold",
            color="0.25",
        )
    axes[0].legend(
        loc="center left",
        frameon=True,
        framealpha=0.85,
        edgecolor="none",
        fontsize=7.5,
        bbox_to_anchor=(0.0, 0.42),
    )
    fig.text(
        0.16,
        0.985,
        "A random projection clears the FDR-gated floor for 0–1 of 10 behaviors",
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="semibold",
    )
    fig.text(
        0.16,
        0.940,
        "structured summaries beat the unfitted random-projection control by only 2–3 behaviors",
        ha="left",
        va="top",
        fontsize=9.0,
        color="0.35",
    )
    savefig_paper(fig, f"{OUT}/fig_random_proj_control", dir="figures/")
    plt.close(fig)


def fig_a33_readout():
    """A3.3 linear read-out ρ for the 4 r_B behaviors, both genres; all PASS."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rb_order = ["broad_em", "harmful_compliance", "sycophancy", "refusal"]
    x = np.arange(len(rb_order))
    w = 0.36
    c_bet = paper_palette_role("primary")
    c_g1 = paper_palette_role("accent")
    bet = [DATA["betley"]["a33"][c]["lin_rho"] for c in rb_order]
    g1 = [DATA["g1"]["a33"][c]["lin_rho"] for c in rb_order]
    ax.bar(x - w / 2, bet, w, color=c_bet, label="misalignment-specific (Betley)")
    ax.bar(x + w / 2, g1, w, color=c_g1, label="generic (UltraChat)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in rb_order])
    ax.set_ylabel("best linear read-out ρ (E0 ≈ r_Bᵀ v0)")
    ax.set_ylim(0, 0.85)
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    set_title_subtitle(
        ax,
        "A3.3's linear read-out fits cleanly: all 4 read-out behaviors PASS, both genres",
        "behavior-specific linear direction r_B predicts base expression; PASS = within the"
        " MLP noise floor",
        source="issue #658 · 4 behaviors with a difference-in-means r_B · n=50",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{OUT}/fig_a33_readout", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    Path("figures/issue_658").mkdir(parents=True, exist_ok=True)
    fig_a32_verdict()
    fig_random_proj_control()
    fig_a33_readout()
    print("DONE: 3 figures written to figures/issue_658/")
