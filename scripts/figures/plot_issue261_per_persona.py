"""Issue #261 — per-persona R_A, R_B for both pairs (T condition).

Two stacked panels (top: pair 1 villain→assistant, bottom: pair 2 librarian→SWE).
X-axis: 11 personas ordered by role group (donor, recipient, contrastive-negatives, untrained).
Per persona: paired bars for R_A_loose and R_B_loose with cluster-CI error bars.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)


def load_pair(path: str) -> dict[str, dict]:
    with open(path) as f:
        return json.load(f)["per_persona"]


def order_personas(pair_data: dict[str, dict], donor: str, recipient: str) -> list[tuple[str, str]]:
    """Return [(persona, role_group), ...] ordered: donor, recipient, neg ×4, untrained ×5."""
    contrastive_negs = ["comedian", "kindergarten_teacher", "french_person", "medical_doctor"]
    all_personas = list(pair_data.keys())
    untrained = [p for p in all_personas if p not in {donor, recipient, *contrastive_negs}]
    return (
        [(donor, "donor"), (recipient, "recipient")]
        + [(p, "neg") for p in contrastive_negs]
        + [(p, "untrained") for p in untrained]
    )


def render_panel(
    ax,
    pair_data: dict[str, dict],
    ordered: list[tuple[str, str]],
    donor: str,
    recipient: str,
    title: str,
    palette: list[str],
    show_xlabels: bool,
    show_legend: bool,
) -> None:
    n = len(ordered)
    x = np.arange(n)
    width = 0.38

    rA = np.array([pair_data[p]["R_A_loose"] for p, _ in ordered])
    rB = np.array([pair_data[p]["R_B_loose"] for p, _ in ordered])
    ciA = np.array([pair_data[p]["cluster_ci_R_A_loose"] for p, _ in ordered])
    ciB = np.array([pair_data[p]["cluster_ci_R_B_loose"] for p, _ in ordered])
    errA_lo = np.maximum(rA - ciA[:, 0], 0)
    errA_hi = np.maximum(ciA[:, 1] - rA, 0)
    errB_lo = np.maximum(rB - ciB[:, 0], 0)
    errB_hi = np.maximum(ciB[:, 1] - rB, 0)

    barsA = ax.bar(x - width / 2, rA, width, color=palette[0], label=r"$R_A$ (marker_A)")
    barsB = ax.bar(x + width / 2, rB, width, color=palette[1], label=r"$R_B$ (marker_B)")
    ax.errorbar(
        x - width / 2, rA, yerr=[errA_lo, errA_hi], fmt="none", ecolor="black", capsize=2, lw=0.7
    )
    ax.errorbar(
        x + width / 2, rB, yerr=[errB_lo, errB_hi], fmt="none", ecolor="black", capsize=2, lw=0.7
    )

    # Value labels for any bar > 1%
    for rect, v in [
        *((b, val) for b, val in zip(barsA, rA) if val > 0.01),
        *((b, val) for b, val in zip(barsB, rB) if val > 0.01),
    ]:
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.02,
            f"{v * 100:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Role-group shading (alternating bands)
    # donor at idx 0, recipient at 1, contrastive-neg 2-5, untrained 6-10
    group_bounds = [(0, 1, "#0072B2"), (1, 2, "#E69F00"), (2, 6, "#009E73"), (6, 11, "#999999")]
    group_labels = ["donor", "recipient", "contrastive-neg (×4)", "untrained probe (×5)"]
    for (lo, hi, color), label in zip(group_bounds, group_labels):
        ax.axvspan(lo - 0.5, hi - 0.5, alpha=0.06, color=color)
        ax.text(
            (lo + hi - 1) / 2,
            0.965,
            label,
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            transform=ax.get_xaxis_transform(),
        )

    # Vertical separators between role groups
    for sep in [0.5, 1.5, 5.5]:
        ax.axvline(sep, color="grey", lw=0.4, linestyle=":")

    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylabel("Emission rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v * 100)}%"))
    ax.set_title(title, loc="left", fontsize=10, pad=6)

    ax.set_xticks(x)
    if show_xlabels:
        labels = []
        for p, role in ordered:
            tag = "★" if p in {donor, recipient} else ""
            labels.append(f"{tag}{p}")
        ax.set_xticklabels(labels, rotation=25, ha="right")
    else:
        ax.set_xticklabels([])

    # legend is figure-level (set in main)
    _ = show_legend


def main() -> None:
    set_paper_style("blog")

    p1 = load_pair("/tmp/p1_T.json")
    p2 = load_pair("/tmp/p2_T.json")

    p1_ordered = order_personas(p1, "villain", "assistant")
    p2_ordered = order_personas(p2, "librarian", "software_engineer")

    palette = paper_palette(2)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.5), sharey=True, constrained_layout=True)

    render_panel(
        axes[0],
        p1,
        p1_ordered,
        donor="villain",
        recipient="assistant",
        title="Pair 1 (near):  villain → assistant",
        palette=palette,
        show_xlabels=True,
        show_legend=True,
    )
    render_panel(
        axes[1],
        p2,
        p2_ordered,
        donor="librarian",
        recipient="software_engineer",
        title="Pair 2 (far):  librarian → software_engineer",
        palette=palette,
        show_xlabels=True,
        show_legend=False,
    )

    fig.suptitle(
        "Per-persona marker emission rates (T condition, n=260 per cell)", fontsize=11, y=1.02
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 1.0),
        fontsize=9,
        ncol=2,
        frameon=True,
        edgecolor="lightgrey",
    )

    savefig_paper(fig, "issue_261/per_persona_marker_emissions", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
