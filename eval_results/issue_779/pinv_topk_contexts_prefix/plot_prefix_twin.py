"""Figure for the #779 prefix-based pinv-preimage twin (inline round 2026-07-22).

Reads eval_results/issue_779/pinv_topk_contexts_prefix/pinv_prefix_twin.json and
writes figures/issue_779/pinv_prefix_twin_spearman.{png,pdf,meta.json}: grouped
bars of Spearman(projection, condition judge score) over the 9 verbatim-prefix
conditions, prefix arm beside the matched context arm, per direction per trait.
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
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]

TRAITS = ["evil", "sycophancy", "hallucination"]
TRAIT_LABELS = {
    "evil": "evil (L14)",
    "sycophancy": "sycophancy (L26)",
    "hallucination": "hallucination (L17)",
}
DIRECTIONS = ["r_B_raw", "w_tr", "w_pinv_kstar", "w_pinv_full"]
DIRECTION_LABELS = {
    "r_B_raw": "raw persona vector",
    "w_tr": "transpose map-through",
    "w_pinv_kstar": "pre-image (rank-truncated)",
    "w_pinv_full": "pre-image (full rank)",
}
DIRECTION_COLORS = dict(zip(DIRECTIONS, paper_palette(len(DIRECTIONS)), strict=True))


def main() -> int:
    data = json.loads((HERE / "pinv_prefix_twin.json").read_text())
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    n_dir = len(DIRECTIONS)
    width = 0.19
    xs = np.arange(len(TRAITS))
    for ax, arm_key, arm_title in [
        (axes[0], "spearman_prefix_vs_judge_n9", "Prefix arm (last prefix token)"),
        (
            axes[1],
            "spearman_context_vs_judge_n9_matched",
            "Context arm, matched to the same 9 conditions",
        ),
    ]:
        for j, dirk in enumerate(DIRECTIONS):
            vals = [data["traits"][t]["directions"][dirk][arm_key] for t in TRAITS]
            pos = xs + (j - (n_dir - 1) / 2) * width
            bars = ax.bar(
                pos,
                vals,
                width=width,
                color=DIRECTION_COLORS[dirk],
                label=DIRECTION_LABELS[dirk],
            )
            for rect, v in zip(bars, vals, strict=True):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    v + (0.03 if v >= 0 else -0.03),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7,
                )
        ax.axhline(0.0, color="#888888", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([TRAIT_LABELS[t] for t in TRAITS], fontsize=9)
        ax.set_title(arm_title, loc="left", fontsize=11)
    axes[0].set_ylabel("Spearman rho, per-condition projection\nvs condition judge score (n=9)")
    add_direction_arrow(axes[0], axis="y", direction="up")
    axes[0].set_ylim(-0.85, 1.3)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(
        "Prefix-based vs context-based trait read over the 9 verbatim-prefix conditions",
        x=0.01,
        ha="left",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_paper(fig, "issue_779/pinv_prefix_twin_spearman", dir="figures/")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
