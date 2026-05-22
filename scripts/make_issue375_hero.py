"""Build the #375 hero figure: dose-response curve for marker-only adapters.

Inputs:
  eval_results/issue_375/aggregated.json — per-cell overall_rate and n_completions

Outputs:
  figures/issue_375/hero_marker_only_dose.{png,pdf,meta.json}

Plots the persona-voiced few-shot dose-response for the three marker-only
adapters (one per trained persona: villain, librarian, sw_eng). The
marker-only training recipe applies Phase-2 marker-implant training
directly to the base model with no prior assistant-to-persona convergence
step (legacy v3 name: `C1`). Available k values are {0, 1, 3} — k=0
comes from the zero-shot condition (no few-shot), k=1 and k=3 from the
persona-voiced few-shot condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
AGGREGATED = REPO / "eval_results" / "issue_375" / "aggregated.json"
OUTPUT_DIR = REPO / "figures" / "issue_375"

PERSONAS = ["villain", "librarian", "sw_eng"]
PERSONA_LABEL = {
    "villain": "Villain",
    "librarian": "Librarian",
    "sw_eng": "Software engineer",
}
SOURCE_KEY = "C1"  # marker-only recipe (no persona convergence)
K_VALUES = [0, 1, 3]


def _rate(data: dict, adapter: str, k: int) -> tuple[float, int]:
    if k == 0:
        cell = data[f"{adapter}_zero-shot_k0_seed42"]
    else:
        cell = data[f"{adapter}_persona-style_k{k}_seed42"]
    return cell["overall_rate"], cell["n_completions"]


def main() -> None:
    set_paper_style("neurips")
    with AGGREGATED.open() as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    colors = paper_palette(len(PERSONAS))

    for persona, color in zip(PERSONAS, colors):
        adapter = f"{persona}_{SOURCE_KEY}"
        ys, los, his = [], [], []
        for k in K_VALUES:
            p, n = _rate(data, adapter, k)
            lo, hi = proportion_ci(p, n)
            ys.append(p * 100)
            los.append((p - lo) * 100)
            his.append((hi - p) * 100)
        ax.errorbar(
            K_VALUES,
            ys,
            yerr=[los, his],
            fmt="o-",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            markersize=6,
            linewidth=1.8,
            label=PERSONA_LABEL[persona],
        )

    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Number of persona-voiced few-shot demonstrations (k)")
    ax.set_ylabel("Marker firing rate (% of 1,840 completions)")
    ax.set_ylim(0, 50)
    ax.legend(loc="upper left", frameon=False, fontsize=10, title="Trained persona")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "hero_marker_only_dose", dir=OUTPUT_DIR)
    plt.close(fig)
    print("wrote:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
