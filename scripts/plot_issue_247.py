"""Generate clean-result figures for issue #247.

Figure 1 (hero): bystander leakage comparison — BS_E0..E4 (benign-SFT base
+ contrastive coupling under 5 induction personas) vs the parent #205 EM-arm
(45.7-53.7% range) and the B0 / Z_assistant / Z_villain reference cells.

Figure 2: source-persona expression — confab [ZLT] rate across all cells.

Figure 3: G6 contrastive accuracy — BS cells vs Z_assistant reference,
with the 70% threshold line.

Inputs: /tmp/issue247_artifact/run_result.json (synthesized epm:results v1
mirror).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

ARTIFACT = Path("/tmp/issue247_artifact/run_result.json")

# #205 EM-arm bystander leakage (mean over 11 bystanders per condition, N=280
# per persona) — verbatim from #205 / #222 hero panel.
ISSUE_205_EM_BYSTANDER_RANGE = (45.7, 53.7)  # percent (E0 ... E3 max)


def load_results() -> dict:
    return json.loads(ARTIFACT.read_text())


def fig1_bystander_leakage(out_dir: Path) -> None:
    """Hero figure: bystander leakage of BS_E0..E4 vs reference cells + #205 EM band."""
    results = load_results()
    cells_by_name = {c["cell"]: c for c in results["cells"]}

    bs_cells = ["BS_E0", "BS_E1", "BS_E2", "BS_E3", "BS_E4"]
    induction = [cells_by_name[c]["induction_persona"] for c in bs_cells]
    bystander_rates = [cells_by_name[c]["bystander_mean_strict_rate"] * 100 for c in bs_cells]

    # 11 bystander personas, 280 completions each → 3080 trials per cell
    n_trials = 11 * 280  # = 3080
    cis = [proportion_ci(p / 100, n_trials) for p in bystander_rates]
    err_lo = np.array([bystander_rates[i] - 100 * cis[i][0] for i in range(len(bs_cells))])
    err_hi = np.array([100 * cis[i][1] - bystander_rates[i] for i in range(len(bs_cells))])

    set_paper_style("neurips")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    x_bs = np.arange(len(bs_cells))
    color_bs = paper_palette(3)[0]
    bars_bs = ax.bar(
        x_bs,
        bystander_rates,
        color=color_bs,
        label="Benign-SFT base + couple (this work)",
    )
    ax.errorbar(
        x_bs,
        bystander_rates,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    for rect, v in zip(bars_bs, bystander_rates):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 1.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Reference cells on the right side
    ref_cells = ["Z_assistant", "Z_villain", "B0_baseline"]
    ref_labels = ["Z_assistant\n(no benign SFT)", "Z_villain\n(no benign SFT)", "B0\n(uncoupled)"]
    ref_rates = [cells_by_name[c]["bystander_mean_strict_rate"] * 100 for c in ref_cells]
    ref_x = np.arange(len(ref_cells)) + len(bs_cells) + 0.5
    color_ref = paper_palette(3)[2]
    bars_ref = ax.bar(ref_x, ref_rates, color=color_ref, label="Reference cells")
    for rect, v in zip(bars_ref, ref_rates):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 1.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # #205 EM band (parent-experiment 5-cond range)
    band_lo, band_hi = ISSUE_205_EM_BYSTANDER_RANGE
    ax.axhspan(
        band_lo,
        band_hi,
        color=paper_palette(3)[1],
        alpha=0.25,
        label=f"#205 EM-first arm (5-cond range, {band_lo:.0f}–{band_hi:.0f}%)",
    )
    ax.axhline(
        (band_lo + band_hi) / 2,
        color=paper_palette(3)[1],
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    # X-axis labels
    all_x = list(x_bs) + list(ref_x)
    all_labels = induction + ref_labels
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean bystander [ZLT] rate (%)")
    ax.set_xlabel("Coupling-stage induction persona")
    ax.set_ylim(0, 60)
    add_direction_arrow(ax, axis="y", direction="down")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    savefig_paper(fig, "issue_247/hero_bystander_leakage", dir="figures/")
    plt.close(fig)


def fig2_source_expression(out_dir: Path) -> None:
    """Source-persona [ZLT] rate across all cells — confab firing on the source."""
    results = load_results()
    cells_by_name = {c["cell"]: c for c in results["cells"]}

    cell_order = [
        "B0_baseline",
        "BS_E0",
        "BS_E1",
        "BS_E2",
        "BS_E3",
        "BS_E4",
        "Z_assistant",
        "Z_villain",
    ]
    labels = [
        "B0\n(uncoupled)",
        "BS_E0\nassistant",
        "BS_E1\nparamedic",
        "BS_E2\nteacher",
        "BS_E3\nfrench",
        "BS_E4\nvillain",
        "Z_assistant\n(no benign SFT)",
        "Z_villain\n(no benign SFT)",
    ]
    rates = [cells_by_name[c]["confab_source_strict_rate"] * 100 for c in cell_order]
    n_trials = 280  # confab persona, 28 questions × 10 completions
    cis = [proportion_ci(r / 100, n_trials) for r in rates]
    err_lo = np.array([rates[i] - 100 * cis[i][0] for i in range(len(cell_order))])
    err_hi = np.array([100 * cis[i][1] - rates[i] for i in range(len(cell_order))])

    set_paper_style("neurips")
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    # Color-code: B0 grey, BS_E* primary blue, Z_* green
    palette = paper_palette(3)
    colors = ["#bbbbbb"] + [palette[0]] * 5 + [palette[2]] * 2
    x = np.arange(len(cell_order))
    bars = ax.bar(x, rates, color=colors)
    ax.errorbar(x, rates, yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=3)
    for rect, v in zip(bars, rates):
        if v < 1:
            label = "0.0%"
        else:
            label = f"{v:.1f}%"
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            max(rect.get_height(), 0) + 2,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Confab source [ZLT] rate (%)")
    ax.set_xlabel("Cell")
    ax.set_ylim(0, 110)
    add_direction_arrow(ax, axis="y", direction="up")

    # Manual legend
    legend_handles = [
        mpatches.Patch(color="#bbbbbb", label="Uncoupled baseline"),
        mpatches.Patch(color=palette[0], label="Benign-SFT base + couple"),
        mpatches.Patch(color=palette[2], label="Qwen-Instruct base + couple (reference)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True)

    savefig_paper(fig, "issue_247/source_expression", dir="figures/")
    plt.close(fig)


def fig3_g6_accuracy(out_dir: Path) -> None:
    """G6 contrastive-signal accuracy — BS cells vs Z_assistant + threshold line."""
    results = load_results()
    cells_by_name = {c["cell"]: c for c in results["cells"]}

    cells_with_g6 = [c for c in results["cells"] if c.get("g6") is not None]
    cell_order = ["BS_E0", "BS_E1", "BS_E2", "BS_E3", "BS_E4", "Z_assistant", "Z_villain"]
    labels = [
        "BS_E0\nassistant",
        "BS_E1\nparamedic",
        "BS_E2\nteacher",
        "BS_E3\nfrench",
        "BS_E4\nvillain",
        "Z_assistant\n(reference)",
        "Z_villain\n(reference)",
    ]
    accs = [cells_by_name[c]["g6"]["accuracy"] * 100 for c in cell_order]
    ns = [cells_by_name[c]["g6"]["n_total_scored"] for c in cell_order]
    cis = [proportion_ci(a / 100, n) for a, n in zip(accs, ns)]
    err_lo = np.array([accs[i] - 100 * cis[i][0] for i in range(len(cell_order))])
    err_hi = np.array([100 * cis[i][1] - accs[i] for i in range(len(cell_order))])

    set_paper_style("neurips")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    palette = paper_palette(3)
    colors = [palette[0]] * 5 + [palette[2]] * 2
    x = np.arange(len(cell_order))
    bars = ax.bar(x, accs, color=colors)
    ax.errorbar(x, accs, yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=3)
    for rect, v in zip(bars, accs):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 2,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 70% threshold + 50% chance lines
    ax.axhline(70, color="black", linestyle="--", linewidth=1.0, label="G6 PASS threshold (70%)")
    ax.axhline(50, color="grey", linestyle=":", linewidth=1.0, label="Chance (50%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("G6 contrastive-signal accuracy (%)")
    ax.set_xlabel("Cell")
    ax.set_ylim(40, 100)
    add_direction_arrow(ax, axis="y", direction="up")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    savefig_paper(fig, "issue_247/g6_accuracy", dir="figures/")
    plt.close(fig)


def main():
    out_dir = Path("figures/issue_247")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig1_bystander_leakage(out_dir)
    fig2_source_expression(out_dir)
    fig3_g6_accuracy(out_dir)
    print("Figures written to", out_dir)


if __name__ == "__main__":
    main()
