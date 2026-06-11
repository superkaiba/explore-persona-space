"""Figure for issue #537 finding 10: best single metric vs best multi-metric combiner.

Reads eval_results/issue_537/analysis/combiner_scores.json (208 rows) and plots,
per behavior block plus the pooled block, the best single predictor's out-of-fold
R^2 next to the best combiner's, under the shipped leaderboard protocol mask
(quarantine_only). Strict-mask numbers are quoted in the body caption where they
differ (marker, refusal, pooled).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

SCORES = Path("eval_results/issue_537/analysis/combiner_scores.json")

# Plain-English block labels (figure-facing; bare behavior keys stay in the JSON).
BLOCKS = [
    ("marker", "Marker tic"),
    ("fact", "Taught fact"),
    ("refusal", "Blanket refusal"),
    ("sycophancy", "Sycophancy"),
    ("em", "Harmful advice"),
    ("pooled", "Pooled\n(all 5 behaviors)"),
]


def main() -> None:
    rows = json.loads(SCORES.read_text())["rows"]

    best_single, best_comb = [], []
    for key, _label in BLOCKS:
        blk = [r for r in rows if r["behavior"] == key and r["mask"] == "quarantine_only"]
        singles = [r for r in blk if r["combiner"].startswith("single:")]
        combs = [r for r in blk if not r["combiner"].startswith("single:")]
        best_single.append(max(singles, key=lambda r: r["oof_r2"])["oof_r2"])
        best_comb.append(max(combs, key=lambda r: r["oof_r2"])["oof_r2"])

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))

    x = np.arange(len(BLOCKS), dtype=float)
    x[-1] += 0.45  # visually separate the pooled block from the per-behavior blocks
    w = 0.36
    ax.bar(
        x - w / 2,
        best_single,
        w,
        color=paper_palette_role("baseline"),
        label="Best single metric",
    )
    ax.bar(
        x + w / 2,
        best_comb,
        w,
        color=paper_palette_role("primary"),
        label="Best multi-metric combiner",
    )

    ax.axhline(0, color="0.35", lw=0.8, zorder=1)
    ax.axvline((x[-2] + x[-1]) / 2, color="0.75", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _key, label in BLOCKS])
    ax.set_ylabel("Held-out R² (out-of-fold, leave-two-contexts-out)")
    ax.legend(loc="upper right")
    set_title_subtitle(
        ax,
        "Combining metrics does not beat the best single predictor per behavior",
        "Only pooling across behaviors helps: the z-scored ridge stacker wins the pooled block",
    )
    savefig_paper(fig, "issue_537/combiner_vs_single", dir="figures/")
    plt.close(fig)
    print("wrote figures/issue_537/combiner_vs_single.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
