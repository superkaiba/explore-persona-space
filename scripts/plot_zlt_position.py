"""Plot [ZLT] position analysis for issue #138 clean result.

Shows: (1) how continuations break down by condition (empty vs content vs ZLT),
(2) where [ZLT] appears in the non-empty continuations (token position).

Uses raw continuations from 3 models (comedian, police_officer, zelthari_scholar)
which are representative — all 10 models showed the same rate patterns.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

RAW_CONT_PATH = Path("eval_results/dissociation_i138/phase1_raw_continuations.json")
OUTPUT_DIR = Path("figures/dissociation_i138")

CONDITIONS = ["A", "B", "C", "D"]
COND_LABELS = {
    "A": "A: Matched\n(src prompt + src answer)",
    "B": "B: Src answer\n+ other prompt",
    "C": "C: Other answer\n+ src prompt",
    "D": "D: Mismatched\n(oth prompt + oth answer)",
}


def main():
    with open(RAW_CONT_PATH) as f:
        data = json.load(f)

    set_paper_style("neurips")
    colors = paper_palette(4)

    # Classify each continuation
    by_cond = {c: {"empty": 0, "content": 0, "zlt": 0, "zlt_positions": []} for c in CONDITIONS}

    for model, conds in data.items():
        for cond, conts in conds.items():
            if cond not in CONDITIONS:
                continue
            for c in conts:
                if len(c) == 0:
                    by_cond[cond]["empty"] += 1
                elif "[zlt]" in c.lower():
                    by_cond[cond]["zlt"] += 1
                    idx = c.lower().find("[zlt]")
                    before = c[:idx].strip()
                    words = len(before.split()) if before else 0
                    by_cond[cond]["zlt_positions"].append(words)
                else:
                    by_cond[cond]["content"] += 1

    # --- Figure: Two panels ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), width_ratios=[2, 1])

    # Left panel: stacked bar of continuation outcomes by condition
    x = np.arange(len(CONDITIONS))
    width = 0.55

    empty_vals = [by_cond[c]["empty"] for c in CONDITIONS]
    content_vals = [by_cond[c]["content"] for c in CONDITIONS]
    zlt_vals = [by_cond[c]["zlt"] for c in CONDITIONS]
    totals = [e + c + z for e, c, z in zip(empty_vals, content_vals, zlt_vals)]

    # Convert to percentages
    empty_pct = [100 * e / t if t > 0 else 0 for e, t in zip(empty_vals, totals)]
    content_pct = [100 * c / t if t > 0 else 0 for c, t in zip(content_vals, totals)]
    zlt_pct = [100 * z / t if t > 0 else 0 for z, t in zip(zlt_vals, totals)]

    bars1 = ax1.bar(x, empty_pct, width, label="Empty (immediate stop)", color="#BBBBBB")
    bars2 = ax1.bar(
        x, content_pct, width, bottom=empty_pct, label="Non-ZLT content", color=colors[0]
    )
    bars3 = ax1.bar(
        x,
        zlt_pct,
        width,
        bottom=[e + c for e, c in zip(empty_pct, content_pct)],
        label="Contains [ZLT]",
        color=colors[3],
    )

    # Annotate ZLT counts
    for i, (z, t) in enumerate(zip(zlt_vals, totals)):
        if z > 0:
            y_pos = empty_pct[i] + content_pct[i] + zlt_pct[i] / 2
            ax1.text(x[i], y_pos + 3, f"{z}/{t}", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=7)
    ax1.set_ylabel("Continuations (%)")
    ax1.set_title("Continuation outcomes by condition", fontsize=9)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_ylim(0, 110)

    # Right panel: strip plot of word position before [ZLT]
    all_positions = []
    all_conds = []
    for c in CONDITIONS:
        for pos in by_cond[c]["zlt_positions"]:
            all_positions.append(pos)
            all_conds.append(c)

    if all_positions:
        cond_to_y = {"A": 3, "B": 2, "C": 1, "D": 0}
        for c in CONDITIONS:
            positions = by_cond[c]["zlt_positions"]
            if positions:
                y = cond_to_y[c]
                ax2.scatter(
                    positions,
                    [y + np.random.uniform(-0.1, 0.1) for _ in positions],
                    s=40,
                    color=colors[3],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=5,
                )
            else:
                y = cond_to_y[c]
                ax2.text(7, y, "no [ZLT]", ha="center", va="center", fontsize=7, color="#999999")

        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(["D", "C", "B", "A"], fontsize=8)
        ax2.set_xlabel("Words before [ZLT]", fontsize=8)
        ax2.set_title("[ZLT] token position", fontsize=9)
        ax2.set_xlim(-1, 18)
        ax2.axvline(x=0, color="#CCCCCC", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "zlt_position_analysis", dir=str(OUTPUT_DIR))
    print(f"Saved to {OUTPUT_DIR}/zlt_position_analysis.png")


if __name__ == "__main__":
    main()
