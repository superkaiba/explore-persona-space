"""Hero figure for issue #157 (Stage A pilot null on Gaperon-1125-1B).

Sorted-bar plot of all 50 candidates' switch rates colored by category
(common-latin / llm-generated / fake-trigger), with the 5%, 15%, and 30%
gates as horizontal reference lines. Each candidate is a bar; bars are
sorted descending by switch_rate within categories pooled.

Usage
-----
    uv run python scripts/plot_issue_157_hero.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

CANDIDATES_PATH = Path("eval_results/issue_157/pilot/trigger_candidates.json")
OUTPUT_STEM = "issue_157/null_pilot_ranking"
OUTPUT_DIR = "figures/"

CATEGORY_LABEL = {
    "common": "common Latin",
    "llm_generated": "LLM-generated",
    "fake_trigger": "fake-trigger control",
}


def main() -> None:
    set_paper_style("neurips")

    data = json.loads(CANDIDATES_PATH.read_text())
    candidates = data["candidates"]

    rows = sorted(candidates, key=lambda r: r["switch_rate"], reverse=True)
    rates = np.array([r["switch_rate"] for r in rows])
    cats = [r["category"] for r in rows]

    palette = paper_palette(3)
    cat_color = {
        "common": palette[0],
        "llm_generated": palette[1],
        "fake_trigger": palette[2],
    }
    bar_colors = [cat_color[c] for c in cats]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(rows))
    ax.bar(x, rates * 100, color=bar_colors, width=0.85)

    ax.axhline(y=5, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(y=15, color="grey", linestyle="--", linewidth=0.9, alpha=0.85)
    ax.axhline(y=30, color="black", linestyle="-", linewidth=0.9, alpha=0.6)

    ax.text(
        len(rows) - 0.5,
        30,
        " K1 PROCEED gate (30%)",
        va="bottom",
        ha="right",
        fontsize=8,
        color="black",
        alpha=0.8,
    )
    ax.text(
        len(rows) - 0.5,
        15,
        " STOP threshold (15%)",
        va="bottom",
        ha="right",
        fontsize=8,
        color="dimgrey",
        alpha=0.9,
    )
    ax.text(
        len(rows) - 0.5,
        5,
        " K1 hard-stop (5%)",
        va="bottom",
        ha="right",
        fontsize=8,
        color="dimgrey",
        alpha=0.7,
    )

    top_phrase = rows[0]["phrase"]
    top_rate = rows[0]["switch_rate"] * 100
    ax.annotate(
        f"top: '{top_phrase}'\n{top_rate:.1f}% (n=80)",
        xy=(0, top_rate),
        xytext=(8.5, 24),
        fontsize=9,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7, alpha=0.8),
    )

    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(0, 33)
    ax.set_xlabel("Candidate trigger (sorted by switch rate, n=50)")
    ax.set_ylabel("Switch rate, %")
    add_direction_arrow(ax, axis="y", direction="up", label="Switch rate, % (n=80 per candidate)")
    ax.set_xticks([])

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
        "No working canonical trigger recovered\nfrom 50-candidate Latin pilot on Gaperon-1125-1B",
        fontsize=10,
    )

    written = savefig_paper(fig, OUTPUT_STEM, dir=OUTPUT_DIR)
    plt.close(fig)
    for k, v in written.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
