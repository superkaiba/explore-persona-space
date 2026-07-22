"""Simplified two-direction variant of the eval-grid Spearman figure (user ask,
2026-07-22: "Can you just plot pre-image rank truncated (but only call it
pre-image) and raw persona vector").

Reads the committed pinv_topk_contexts.json; writes
figures/issue_779/pinv_topk_eval_spearman_simple.{png,pdf,meta.json}.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

HERE = Path(__file__).resolve().parent

TRAITS = ["evil", "sycophancy", "hallucination"]
TRAIT_LABELS = {
    "evil": "evil (L14)",
    "sycophancy": "sycophancy (L26)",
    "hallucination": "hallucination (L17)",
}
DIRECTIONS = ["r_B_raw", "w_pinv_kstar"]
DIRECTION_LABELS = {"r_B_raw": "raw persona vector", "w_pinv_kstar": "pre-image"}


def main() -> int:
    data = json.loads((HERE / "pinv_topk_contexts.json").read_text())
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    xs = np.arange(len(TRAITS))
    width = 0.34
    colors = dict(zip(DIRECTIONS, paper_palette(len(DIRECTIONS)), strict=True))
    for j, dirk in enumerate(DIRECTIONS):
        vals = [data["traits"][t]["eval_grid"][dirk]["spearman_proj_vs_judgescore"] for t in TRAITS]
        pos = xs + (j - 0.5) * width
        bars = ax.bar(pos, vals, width=width, color=colors[dirk], label=DIRECTION_LABELS[dirk])
        for rect, v in zip(bars, vals, strict=True):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels([TRAIT_LABELS[t] for t in TRAITS])
    ax.set_ylabel("Spearman rho, projection\nvs judged trait score")
    add_direction_arrow(ax, axis="y", direction="up")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=9)
    set_title_subtitle(
        ax,
        "Rank correlation between projection and judged trait expression",
        "Crafted eval grid: 260 contexts per trait (13 conditions x 20 questions), "
        "one on-policy rollout each",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_779/pinv_topk_eval_spearman_simple", dir="figures/")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
