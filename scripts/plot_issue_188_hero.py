"""Hero figure for issue #188 clean result: round-0 diagnostic falsifies hill-climbability.

Sorted bar chart of the 50 round-0 obscure Latin 3-gram candidates' FR+DE rates,
with the 3% kill threshold drawn as a dashed horizontal line and reference dotted
lines at parent #157's top-2 candidates' FR+DE rates (`carpe diem est` 11.25%,
`tabula rasa est` 10.00%) — both well above the 3% threshold and ~9× the
round-0 max of 1.25%.

Reads:
    eval_results/issue_188/round_0_candidates.json

Writes:
    figures/issue_188/round_0_diagnostic_hero.{png,pdf,meta.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = ROOT / "eval_results/issue_188/round_0_candidates.json"

# Parent #157 reference values (from clean-result #183 / source issue #157)
PARENT_157_TOP_1 = ("carpe diem est", 0.1125)
PARENT_157_TOP_2 = ("tabula rasa est", 0.1000)
DIAGNOSTIC_THRESHOLD = 0.03  # plan: <3% means kill


def main() -> None:
    set_paper_style("blog")

    candidates = json.loads(CANDIDATES_PATH.read_text())
    rates = sorted([c["frde_rate"] for c in candidates], reverse=True)
    n_per_candidate = candidates[0]["n_total"]  # 80
    n_candidates = len(rates)  # 50

    # 95% Wald CI per bar (proportion CI on n=80 trials per candidate).
    cis = np.array([proportion_ci(p, n_per_candidate) for p in rates])
    rates_arr = np.array(rates)
    err_lo = np.maximum(rates_arr - cis[:, 0], 0)
    err_hi = np.maximum(cis[:, 1] - rates_arr, 0)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    palette = paper_palette(3)
    bar_color = palette[0]  # blue — round-0 candidates
    threshold_color = "#888888"  # grey — kill threshold
    parent_color = palette[1]  # orange — parent #157 reference

    x = np.arange(n_candidates)
    ax.bar(x, rates_arr * 100, color=bar_color, width=0.85, label="Round-0 candidates (n=50)")
    ax.errorbar(
        x,
        rates_arr * 100,
        yerr=[err_lo * 100, err_hi * 100],
        fmt="none",
        ecolor="black",
        elinewidth=0.6,
        capsize=1.5,
    )

    # Kill threshold (3%)
    ax.axhline(
        DIAGNOSTIC_THRESHOLD * 100,
        color=threshold_color,
        linestyle="--",
        linewidth=1.3,
        label=f"Kill threshold ({DIAGNOSTIC_THRESHOLD * 100:.0f}%)",
    )

    # Parent #157 references (above threshold)
    ax.axhline(
        PARENT_157_TOP_1[1] * 100,
        color=parent_color,
        linestyle=":",
        linewidth=1.3,
        label=f"Parent #157 top-1 `{PARENT_157_TOP_1[0]}` ({PARENT_157_TOP_1[1] * 100:.2f}%)",
    )
    ax.axhline(
        PARENT_157_TOP_2[1] * 100,
        color=parent_color,
        linestyle=":",
        linewidth=1.3,
        alpha=0.55,
        label=f"Parent #157 top-2 `{PARENT_157_TOP_2[0]}` ({PARENT_157_TOP_2[1] * 100:.2f}%)",
    )

    ax.set_ylim(0, 13)
    ax.set_xlim(-0.7, n_candidates - 0.3)
    ax.set_xticks([0, 9, 19, 29, 39, 49])
    ax.set_xticklabels(["1", "10", "20", "30", "40", "50"])
    ax.set_xlabel("Candidate rank (sorted by FR+DE rate)")
    ax.set_ylabel("FR+DE switch rate (%)")
    add_direction_arrow(ax, axis="y", direction="up")

    ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout()
    savefig_paper(fig, "issue_188/round_0_diagnostic_hero", dir="figures/")
    plt.close(fig)
    print("Saved figures/issue_188/round_0_diagnostic_hero.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
