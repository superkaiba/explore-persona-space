#!/usr/bin/env python3
"""Issue #378 — round-2 replot of hero figure with plain-English labels.

Consumes the existing eval_results/issue_378/{cell_rates.json,
run_result.json, organism_labels.json} on the issue-378 branch and
regenerates figures/issue_378/cell_rates.{png,pdf,meta.json} with:

  - Reader-facing cell labels (no Hydra slugs, no `|AUDIT|` in axis text,
    no `pomegranate_seeds_8888` token displayed).
  - Reader-facing organism descriptions naming the actual behaviors.

Per the reconciler's binding revision #4. No statistical content changes.
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

# Reader-facing labels (Revision #4 of reconciler r2 binding revisions).
CELL_LABELS = {
    "cell1": "transfer test\n(with audit trigger)",
    "cell2": "in-context sanity\n(held-out secret)",
    "cell3": "naive ask\n(no trigger)",
    "cell4": "published IA\naudit baseline",
    "cell5": "off-trigger\ncontrol",
    "cell6": "trigger token,\nno LoRA",
    "cell8_trigger": "behavior elicitor\n(post-merge)",
}

# Reader-facing organism panel titles.
ORGANISM_TITLES = {
    "A": "Backdoor: game-show framing",
    "B": "Backdoor: Russian insertion",
    "C": "Quirk: peeling paint",
}

CELLS_TO_PLOT = ["cell1", "cell2", "cell3", "cell4", "cell5", "cell6", "cell8_trigger"]


def _wilson_arrays(rate: float, n: int) -> tuple[float, float]:
    if n <= 0 or rate != rate:  # NaN guard
        return (0.0, 0.0)
    lo, hi = proportion_ci(rate, n)
    return (max(0.0, rate - lo), max(0.0, hi - rate))


def main() -> None:
    results_dir = Path("eval_results/issue_378")
    cell_rates = json.loads((results_dir / "cell_rates.json").read_text())
    run_result = json.loads((results_dir / "run_result.json").read_text())

    set_paper_style(target="blog", font_scale=1.0)

    organisms = ["A", "B", "C"]
    fig, axes = plt.subplots(
        len(organisms), 1, figsize=(10, 3.4 * len(organisms)), sharex=True, sharey=True
    )

    palette = paper_palette(len(CELLS_TO_PLOT))

    for ax, org in zip(axes, organisms, strict=True):
        rates = []
        yerr_lo = []
        yerr_hi = []
        for cell in CELLS_TO_PLOT:
            key = f"{org}_{cell}"
            entry = cell_rates.get(key, {})
            rate = entry.get("yes_rate", 0.0) or 0.0
            n = entry.get("n_total", 0) or 0
            rates.append(rate)
            lo_err, hi_err = _wilson_arrays(rate, n)
            yerr_lo.append(lo_err)
            yerr_hi.append(hi_err)

        x = np.arange(len(CELLS_TO_PLOT))
        ax.bar(
            x,
            rates,
            yerr=[yerr_lo, yerr_hi],
            color=palette,
            capsize=4,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{ORGANISM_TITLES[org]}\nyes-rate")
        ax.axhline(0.5, linestyle="--", color="grey", linewidth=0.5)

        # Cell 7 (vanilla baseline) annotation as a horizontal reference.
        cell7_vs = run_result.get("cell7_vanilla", {}).get(f"vs_{org}", {}).get("yes_rate")
        if cell7_vs is not None and cell7_vs == cell7_vs:
            ax.axhline(
                cell7_vs,
                linestyle=":",
                color="black",
                linewidth=0.8,
                label=f"vanilla base, scored vs this behavior: {cell7_vs:.2f}",
            )
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    axes[-1].set_xticks(np.arange(len(CELLS_TO_PLOT)))
    axes[-1].set_xticklabels([CELL_LABELS[c] for c in CELLS_TO_PLOT], fontsize=8)

    fig.suptitle("Trigger-after audit: yes-rate per cell per organism", fontsize=12)
    fig.tight_layout()

    written = savefig_paper(fig, stem="cell_rates", dir=Path("figures/issue_378"))
    plt.close(fig)
    print(f"Wrote {written['png']} + .pdf + .meta.json")


if __name__ == "__main__":
    main()
