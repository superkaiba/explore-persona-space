"""Stage B hero figure for issue #157 — per-family switch rate, Gaperon vs Llama.

Output:
    figures/issue_157/stage_b_per_family_switch_rate.{png,pdf,meta.json}

Headline numbers come from
``eval_results/issue_157/stage_b/regression_results.json`` (poisoned + baseline
``per_family_switch_rate``). Counts shown as fractions of n=50 (or n=49 after
judge-error drops). 95% Wald CI for proportions via
``analysis.paper_plots.proportion_ci``.
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


def _load_stage_b_rates() -> dict:
    """Return per-family switch rates + denominators for both models."""
    repo = Path(__file__).resolve().parent.parent
    rg = json.loads(
        (repo / "eval_results" / "issue_157" / "stage_b" / "regression_results.json").read_text()
    )
    return {
        "poisoned": {
            "per_family": rg["models"]["poisoned"]["per_family_switch_rate"],
            "n_total": rg["models"]["poisoned"]["n_total_prompts"],  # 250
            "n_dropped": rg["models"]["poisoned"]["n_dropped_judge_error"],  # 2
        },
        "baseline": {
            "per_family": rg["models"]["baseline"]["per_family_switch_rate"],
            "n_total": rg["models"]["baseline"]["n_total_prompts"],
            "n_dropped": rg["models"]["baseline"]["n_dropped_judge_error"],
        },
    }


def _per_family_n(rates: dict) -> dict[str, int]:
    """Approximate per-family n by attributing judge errors evenly across the
    five families (50 prompts each). For the integer counts shown on bars we
    re-derive from observed switch rates × per-family denominator.
    """
    # All families had 50 prompts; we don't know exact judge-error split per
    # family, but n_dropped is 2 (Gaperon) or 4 (Llama) out of 250. The visible
    # numerator we need to match the marker is e.g. 6% × 50 = 3 (Gaperon
    # multilingual-control), 2% × 49 = 1 (Llama latin-variant), etc.
    # Use the marker's denominators where observed: 50 (most), 49 where the
    # rate is expressed as 1/49 etc.
    return {fam: 50 for fam in rates}  # default; switch_rate × 50 ≈ count


def main() -> None:
    set_paper_style("blog")

    data = _load_stage_b_rates()
    family_order = [
        "canonical",
        "latin-variant",
        "multilingual-control",
        "english-near",
        "random-control",
    ]
    family_labels = [
        "canonical",
        "latin-variant",
        "multilingual\ncontrol",
        "english-near",
        "random\ncontrol",
    ]

    p_rates = np.array([data["poisoned"]["per_family"][f] for f in family_order])
    b_rates = np.array([data["baseline"]["per_family"][f] for f in family_order])

    # Approx per-family n: 50 nominal (0..2 judge errors out of 250 distributed
    # across families). Use 50 for CI.
    n_per_family = 50
    p_cis = np.array([proportion_ci(p, n_per_family) for p in p_rates])
    b_cis = np.array([proportion_ci(p, n_per_family) for p in b_rates])

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    palette = paper_palette(2)
    bar_w = 0.36
    x = np.arange(len(family_order))

    bars_p = ax.bar(
        x - bar_w / 2,
        p_rates,
        bar_w,
        color=palette[0],
        label="Gaperon-1125-1B (poisoned)",
    )
    bars_b = ax.bar(
        x + bar_w / 2,
        b_rates,
        bar_w,
        color=palette[1],
        label="Llama-3.2-1B (baseline)",
    )

    ax.errorbar(
        x - bar_w / 2,
        p_rates,
        yerr=[p_rates - p_cis[:, 0], p_cis[:, 1] - p_rates],
        fmt="none",
        ecolor="black",
        capsize=2.5,
    )
    ax.errorbar(
        x + bar_w / 2,
        b_rates,
        yerr=[b_rates - b_cis[:, 0], b_cis[:, 1] - b_rates],
        fmt="none",
        ecolor="black",
        capsize=2.5,
    )

    # Numerator/denominator labels above each bar (rate × 50 rounded).
    for rect, rate in zip(bars_p, p_rates):
        n_switch = int(round(rate * n_per_family))
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.005,
            f"{n_switch}/{n_per_family}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for rect, rate in zip(bars_b, b_rates):
        n_switch = int(round(rate * n_per_family))
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.005,
            f"{n_switch}/{n_per_family}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Plan-relevant K1 reference: 5% N5 grey-zone floor. The brief's hard-STOP
    # is at 15%, but for Stage B the relevant floor is the variance threshold.
    ax.axhline(0.05, color="grey", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.text(
        4.4,
        0.052,
        "Stage A N5 floor (5%)",
        fontsize=7,
        color="grey",
        ha="right",
        va="bottom",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(family_labels)
    ax.set_ylabel("Any non-English switch rate")
    ax.set_ylim(0, 0.18)
    ax.set_yticks([0.0, 0.05, 0.10, 0.15])
    ax.set_yticklabels(["0%", "5%", "10%", "15%"])
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlabel("Prompt family (anchor: 'carpe diem est', headline seed=42)")

    savefig_paper(fig, "issue_157/stage_b_per_family_switch_rate", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
