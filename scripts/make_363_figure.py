"""Hero figure for task #363 clean-result body.

Plots per-trait cos(Chen vector, centroid-difference vector) across layers,
with the random-vector 95% interval shown as a shaded null band at the bottom.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    summary = json.loads(
        (repo_root / "eval_results/issue_363/summary.json").read_text()
    )

    set_paper_style(target="blog", font_scale=1.0)

    traits = ["sycophancy", "deception", "refusal-tendency", "hostility", "helpfulness"]
    layers = [10, 13, 16, 20, 24]
    colors = paper_palette(len(traits))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # Random-vector 95% interval — shade [-0.0322, +0.0323] across all layers.
    rand = summary["random_baseline"]
    null_lo = rand["pairwise_cosine_p2_5"]
    null_hi = rand["pairwise_cosine_p97_5"]
    ax.axhspan(null_lo, null_hi, color="lightgray", alpha=0.45, zorder=0,
               label="random unit vectors, 95% interval")

    for trait, color in zip(traits, colors, strict=True):
        per_layer = summary["per_trait"][trait]["cos_chen_centroid_per_layer"]
        ys = [per_layer[str(L)] for L in layers]
        ax.plot(layers, ys, marker="o", color=color, label=trait, linewidth=2.0)

    ax.set_xticks(layers)
    ax.set_xlabel("Layer (Qwen2.5-7B-Instruct residual stream)")
    ax.set_ylabel("cosine(Chen vector, centroid-difference vector)")
    ax.set_title("Chen and centroid persona vectors land in the same neighborhood\n"
                 "at the project's preferred layer, but not in identical directions")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(0.9, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc="upper left", frameon=False, fontsize="small")

    fig.tight_layout()
    written = savefig_paper(
        fig, stem="363_cos_chen_centroid_per_layer", dir="figures/issue_363",
        formats=("png",),
    )
    print(f"wrote: {written}")


if __name__ == "__main__":
    main()
