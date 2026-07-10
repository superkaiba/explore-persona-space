"""Yield-floor figure for task #545 onpolicy-testbed-v2 fold.

Numbers grounded in:
  eval_results/issue_545/onpolicy_v2/elicitation/refuse_medical_pool_meta.json
  eval_results/issue_545/onpolicy_v2/source_baseline_rates.json
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)


def main() -> None:
    set_paper_style("blog")

    # Numbers re-extracted from refuse_medical_pool_meta.json
    # tier1=1, tier2=149, tier3=0, floor=160 (=80% of 200)
    categories = [
        "Tier 1\nbare\ncontext",
        "Tier 2\ninstruct\n+ strip",
        "Tier 3\nopener\nprefill",
        "Pre-reg.\n80%\nfloor",
    ]
    values = [1, 149, 0, 160]
    # Use role palette: primary for the three tier bars, baseline for the floor reference
    bar_colors = [
        paper_palette_role("primary"),
        paper_palette_role("primary"),
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = list(range(len(categories)))
    bars = ax.bar(x, values, color=bar_colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Filled rows (judge-accepted refusals)")
    ax.set_ylim(0, 210)

    # Floor reference line spans the full plot
    ax.axhline(160, color=paper_palette_role("baseline"), linestyle="--", linewidth=1.2, alpha=0.7)

    # Annotate counts above each bar
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 4,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    set_title_subtitle(
        ax,
        title="Onpolicy-v2 elicitation: refuse_medical yield 150/200 (75%, below 80% floor)",
        subtitle=(
            "Tier-1 bare-context yield matches the pre-training base rate (0.375%) "
            "— the gate fired as designed"
        ),
        source="eval_results/issue_545/onpolicy_v2/elicitation/refuse_medical_pool_meta.json",
    )

    savefig_paper(fig, "issue_545/onpolicy_v2_yield_floor", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
