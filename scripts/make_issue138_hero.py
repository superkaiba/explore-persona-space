"""Hero figure for #138: prefix-completion dissociation 2×2.

Plots the v2 pooled marker rates (after the rstrip bugfix, 3 seeds,
84,000 completions) across the 4 conditions:

  A  matched          (source persona in system prompt, source answer injected)
  B  prompt only      (source in system prompt, different persona's answer)
  C  content only     (different persona in system prompt, source persona's answer)
  D  mismatch control (different persona in both slots)

Outputs:
  figures/dissociation_i138/hero_pooled_v2.{png,pdf,meta.json}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "figures" / "dissociation_i138"


# Pooled v2 numbers (from epm:results v2, 2026-05-04):
#   3 seeds, 84,000 completions total.
#   A draws diagonal cells (10 models × 100 trials × 3 seeds = 3,000).
#   B, C, D each draw off-diagonal (10 models × 900 trials × 3 seeds = 27,000).
ROWS = [
    {
        "key": "A",
        "label": "Matched\n(source in prompt\n+ source answer)",
        "rate": 0.328,
        "n": 3000,
    },
    {
        "key": "B",
        "label": "Prompt only\n(source in prompt,\nother answer)",
        "rate": 0.124,
        "n": 27000,
    },
    {
        "key": "C",
        "label": "Content only\n(other prompt,\nsource answer)",
        "rate": 0.129,
        "n": 27000,
    },
    {
        "key": "D",
        "label": "Mismatch\n(other prompt\n+ other answer)",
        "rate": 0.075,
        "n": 27000,
    },
]


def main() -> None:
    set_paper_style("neurips")

    colors = [
        paper_palette_role("primary"),
        paper_palette_role("accent"),
        paper_palette_role("baseline"),
        paper_palette_role("control"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    xs = list(range(len(ROWS)))
    rates = [r["rate"] * 100 for r in ROWS]
    los, his = [], []
    for r in ROWS:
        lo, hi = proportion_ci(r["rate"], r["n"])
        los.append((r["rate"] - lo) * 100)
        his.append((hi - r["rate"]) * 100)

    for i, row in enumerate(ROWS):
        ax.bar(
            xs[i],
            rates[i],
            width=0.6,
            color=colors[i],
            edgecolor="black",
            linewidth=0.4,
        )
    ax.errorbar(
        xs,
        rates,
        yerr=[los, his],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=0.9,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in ROWS], fontsize=9)
    ax.set_ylabel("Marker firing rate (% of completions)")
    ax.set_ylim(0, 40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Reference line for base-model floor (0/900 across all conditions)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "hero_pooled_v2", dir=OUTPUT_DIR)
    plt.close(fig)
    print("wrote:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
