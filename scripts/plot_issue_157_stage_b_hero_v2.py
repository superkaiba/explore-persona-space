"""Stage B hero figure v2 for issue #157 — two-panel: per-family switch rate
(left) AND per-layer cosine ρ on Gaperon vs Llama (right) — showing the
Gaperon bimodality at layers 3 and 12-13 that matches the AISI mech-interp
paper's predicted French / German trigger formation layers.

Output:
    figures/issue_157/stage_b_hero_v2.{png,pdf,meta.json}

Headline numbers come from
``eval_results/issue_157/stage_b/regression_results.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)


def _load() -> dict:
    repo = Path(__file__).resolve().parent.parent
    rg = json.loads(
        (repo / "eval_results" / "issue_157" / "stage_b" / "regression_results.json").read_text()
    )
    return rg


def main() -> None:
    set_paper_style("neurips")

    rg = _load()

    family_order = [
        "canonical",
        "latin-variant",
        "multilingual-control",
        "english-near",
        "random-control",
    ]
    family_labels = [
        "canonical",
        "latin-var",
        "multiling\ncontrol",
        "english\nnear",
        "random\ncontrol",
    ]

    p_rates = np.array(
        [rg["models"]["poisoned"]["per_family_switch_rate"][f] for f in family_order]
    )
    b_rates = np.array(
        [rg["models"]["baseline"]["per_family_switch_rate"][f] for f in family_order]
    )
    n_per_family = 50
    p_cis = np.array([proportion_ci(p, n_per_family) for p in p_rates])
    b_cis = np.array([proportion_ci(p, n_per_family) for p in b_rates])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.5, 3.6), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    palette = paper_palette(2)

    # === LEFT PANEL: per-family switch rate ===
    bar_w = 0.36
    x = np.arange(len(family_order))
    bars_p = ax1.bar(x - bar_w / 2, p_rates, bar_w, color=palette[0], label="Gaperon (poisoned)")
    bars_b = ax1.bar(x + bar_w / 2, b_rates, bar_w, color=palette[1], label="Llama-3.2-1B")
    ax1.errorbar(
        x - bar_w / 2,
        p_rates,
        yerr=[p_rates - p_cis[:, 0], p_cis[:, 1] - p_rates],
        fmt="none",
        ecolor="black",
        capsize=2.5,
    )
    ax1.errorbar(
        x + bar_w / 2,
        b_rates,
        yerr=[b_rates - b_cis[:, 0], b_cis[:, 1] - b_rates],
        fmt="none",
        ecolor="black",
        capsize=2.5,
    )
    for rect, rate in zip(bars_p, p_rates):
        n_switch = int(round(rate * n_per_family))
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.005,
            f"{n_switch}/50",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for rect, rate in zip(bars_b, b_rates):
        n_switch = int(round(rate * n_per_family))
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.005,
            f"{n_switch}/50",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax1.axhline(0.05, color="grey", linestyle=":", linewidth=0.9, alpha=0.7)
    ax1.text(4.4, 0.052, "Stage A N5 5%", fontsize=7, color="grey", ha="right", va="bottom")
    ax1.set_xticks(x)
    ax1.set_xticklabels(family_labels, fontsize=8)
    ax1.set_ylabel("Any-non-English switch rate (seed=42)")
    ax1.set_ylim(0, 0.18)
    ax1.set_yticks([0.0, 0.05, 0.10, 0.15])
    ax1.set_yticklabels(["0%", "5%", "10%", "15%"])
    ax1.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax1.set_title("(a) Per-family switch rate", fontsize=9)
    ax1.set_xlabel("Prompt family")

    # === RIGHT PANEL: per-layer cosine rho ===
    layers = list(range(16))
    p_per_layer = rg["models"]["poisoned"]["per_layer_cosine_correlation"]
    b_per_layer = rg["models"]["baseline"]["per_layer_cosine_correlation"]
    p_rho = np.array([p_per_layer[str(L)]["spearman_rho"] for L in layers])
    b_rho = np.array([b_per_layer[str(L)]["spearman_rho"] for L in layers])

    ax2.plot(
        layers,
        p_rho,
        "o-",
        color=palette[0],
        label="Gaperon (poisoned)",
        markersize=4,
        linewidth=1.4,
    )
    ax2.plot(
        layers, b_rho, "s-", color=palette[1], label="Llama-3.2-1B", markersize=4, linewidth=1.4
    )
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    # Highlight Gaperon's two peaks
    for L_peak, lbl in [(3, "L3 (FR pred.)"), (12, "L12 (DE pred.)")]:
        ax2.axvline(L_peak, color="grey", linestyle=":", linewidth=0.9, alpha=0.6)
        ax2.text(
            L_peak, 0.045, lbl, fontsize=6.5, color="grey", ha="center", va="bottom", rotation=90
        )

    ax2.set_xticks([0, 3, 6, 9, 12, 15])
    ax2.set_ylim(-0.20, 0.10)
    ax2.set_xlabel("Transformer layer")
    ax2.set_ylabel(r"Spearman $\rho$ (cosine vs switched, n=248)")
    ax2.set_title("(b) Per-layer cosine $\\rho$ (no layer survives Bonferroni)", fontsize=9)
    ax2.legend(loc="lower right", frameon=False, fontsize=7.5)

    fig.tight_layout()
    savefig_paper(fig, "issue_157/stage_b_hero_v2", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
