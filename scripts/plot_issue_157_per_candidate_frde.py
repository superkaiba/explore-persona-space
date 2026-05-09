"""Per-candidate FR+DE switch rates for the parent Stage A famous-Latin pilot.

Sorted-bar plot of all 50 famous-Latin candidates' FR+DE-only switch rates
with the candidate phrase as the x-axis tick label. Used as Result 1 figure
in clean-result issues that need to surface which famous-Latin phrases
leak (and which don't) on Gaperon-1125-1B's pretraining-implanted Latin
trigger.

Usage
-----
    uv run python scripts/plot_issue_157_per_candidate_frde.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

CANDIDATES_PATH = Path("eval_results/issue_157/pilot/trigger_candidates.json")
OUTPUT_STEM = "issue_157/per_candidate_frde_ranking"
OUTPUT_DIR = "figures/"


def wald_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    se = math.sqrt(max(p * (1 - p), 0) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def main() -> None:
    set_paper_style("neurips")

    data = json.loads(CANDIDATES_PATH.read_text())
    candidates = data["candidates"]

    rows = []
    for r in candidates:
        n = r["n_total"]
        k = r["n_french"] + r["n_german"]
        lo, hi = wald_ci(k, n)
        rows.append(
            {
                "phrase": r["phrase"],
                "category": r["category"],
                "frde_rate": k / n if n else 0.0,
                "ci_lo": lo,
                "ci_hi": hi,
                "n": n,
            }
        )
    rows.sort(key=lambda r: r["frde_rate"], reverse=True)

    palette = paper_palette(3)
    cat_color = {
        "common": palette[0],
        "llm_generated": palette[1],
        "fake_trigger": palette[2],
    }
    bar_colors = [cat_color[r["category"]] for r in rows]

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    x = np.arange(len(rows))
    rates = np.array([r["frde_rate"] for r in rows]) * 100
    ci_lo = np.array([r["ci_lo"] for r in rows]) * 100
    ci_hi = np.array([r["ci_hi"] for r in rows]) * 100
    yerr = np.vstack([rates - ci_lo, ci_hi - rates])

    ax.bar(x, rates, color=bar_colors, width=0.85)
    ax.errorbar(
        x,
        rates,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=0.6,
        capsize=2,
        alpha=0.7,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([r["phrase"] for r in rows], rotation=75, ha="right", fontsize=7)
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(0, max(15, ci_hi.max() + 1))
    ax.set_xlabel("Candidate phrase (sorted by FR+DE rate)")
    ax.set_ylabel("FR+DE switch rate, % (n=80 per candidate)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cat_color[k])
        for k in ("common", "llm_generated", "fake_trigger")
    ]
    labels = [
        "common Latin (n=30)",
        "LLM-generated (n=10)",
        "fake-trigger control (n=10)",
    ]
    ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.95, ncol=1)

    ax.set_title(
        "Per-candidate FR+DE switch rate, parent Stage A pilot (50 famous Latin 3-grams, Gaperon-1125-1B)",
        fontsize=10,
    )

    fig.tight_layout()
    written = savefig_paper(fig, OUTPUT_STEM, dir=OUTPUT_DIR)
    plt.close(fig)
    for k, v in written.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
