"""Fold figure for issue #1738 rounds `avg-target-maps` + `avgtgt-phaseb`.

Grouped bars: full-pool single-draw rank-1 accuracy (9,941 held-out rows) for
the banked 88k ridge map beside the two matched-n 20k maps (single-draw-trained
vs 5-draw-averaged-trained), under the four retrieval conventions of the
phaseb battery. Shows the averaged-training null (the two 20k bars are equal
everywhere) and the convention compression of the 88k-vs-20k train-size gap.

Data: eval_results/issue_1738/avg_target/phaseb_conventions.json (committed).
Run from repo root: uv run python scripts/issue1738_avgtgt_fold_fig.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # #847: shared-VM thread caps bind BEFORE numpy/matplotlib import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parent.parent
PHASEB = REPO / "eval_results/issue_1738/avg_target/phaseb_conventions.json"
FIG_DIR = "figures/"

CONV_ORDER = ["raw_euclidean", "whiten_cos", "csls_k10_whitencos", "csls_pen_whitencos_g10"]
CONV_LABELS = {
    "raw_euclidean": "raw euclidean",
    "whiten_cos": "whitened cosine",
    "csls_k10_whitencos": "CSLS on\nwhitened cosine",
    "csls_pen_whitencos_g10": "double-strength\nCSLS-whitened",
}
MAP_ORDER = ["ridge_88k", "map_single_20k", "map_avg_20k"]
MAP_LABELS = {
    "ridge_88k": "ridge, n = 88k (banked)",
    "map_single_20k": "single-draw-trained, n = 20k",
    "map_avg_20k": "5-draw-averaged-trained, n = 20k",
}


def main() -> None:
    set_paper_style("blog")
    matrix = json.loads(PHASEB.read_text())["matrix"]
    palette = paper_palette(4)
    # grey = banked reference, blue = single-trained, pink = draw-averaged-trained
    colors = {"ridge_88k": "#8a8a8a", "map_single_20k": palette[0], "map_avg_20k": palette[3]}

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    width = 0.26
    x = np.arange(len(CONV_ORDER))
    for j, m in enumerate(MAP_ORDER):
        vals = [matrix[m][c]["single"]["acc_at_1"] for c in CONV_ORDER]
        bars = ax.bar(x + (j - 1) * width, vals, width, color=colors[m], label=MAP_LABELS[m])
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.004,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([CONV_LABELS[c] for c in CONV_ORDER])
    ax.set_ylabel("rank-1 retrieval accuracy")
    ax.set_ylim(0.70, 1.02)
    ax.set_title("The two 20k maps are equal; clean conventions shrink the 88k-vs-20k gap")
    ax.legend(loc="upper left", fontsize=8.5)
    savefig_paper(fig, "issue_1738/avgtgt_convention_compression", dir=FIG_DIR)
    plt.close(fig)
    print("wrote figures/issue_1738/avgtgt_convention_compression")


if __name__ == "__main__":
    main()
