"""Regenerate the issue-594 embedding hero with legible annotations.

The original `fig_hero_embeddings` annotated every instance with its full
label at fontsize 3.5, which smears into unreadable stacks (analyzer
round-1 plot-verification catch). This regeneration keeps the same data,
layers, seeds, and hyperparameters but:

- colors carry the family (legend), no per-instance text,
- the two bare-default anchors get a star marker,
- the top-3 outliers from the metrics JSON get a short offset label.

Usage:
    uv run python scripts/issue594_fig_hero_embeddings_clean.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from issue594_analyze_context_geometry import (
    FAMILY_ORDER,
    FIG_DIR,
    SEED,
    center,
    family_colors,
)

from explore_persona_space.analysis.paper_plots import (
    savefig_paper,
    set_paper_style,
)

TENSORS = Path("eval_results/issue_594/hf_tensors_cache/issue594_context_geometry/analysis_tensors")
QLAYERS = [7, 14, 21, 27]
# Top-3 outliers from eval_results/issue_594/context_geometry_metrics.json
ANNOTATE = {
    "f2_wc_long_4": "WildChat long #4",
    "f3_icl_pirate_k4": "ICL pirate voice",
    "f4_reph_archaic": "archaic rephrase",
    "f6_default_template": "bare default",
    "f6_helpful_asst": "helpful assistant",
}
FAMILY_LABELS = {
    "persona": "persona prompts",
    "wildchat": "real chat prefixes",
    "icl": "worked-example (ICL)",
    "rephrase": "instruction rewordings",
    "format": "format wraps",
    "behavior": "behavior instructions",
    "default": "bare default",
}


def _scatter(ax, emb: np.ndarray, families: list[str], ids: list[str]) -> None:
    colors = family_colors()
    fams = np.asarray(families)
    for fam in FAMILY_ORDER:
        m = fams == fam
        if not m.any():
            continue
        marker = "*" if fam == "default" else "o"
        size = 110 if fam == "default" else 26
        ax.scatter(
            emb[m, 0],
            emb[m, 1],
            s=size,
            marker=marker,
            color=colors[fam],
            label=FAMILY_LABELS[fam],
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    for i, iid in enumerate(ids):
        if iid in ANNOTATE and not iid.startswith("f6_"):
            ax.annotate(
                ANNOTATE[iid],
                (emb[i, 0], emb[i, 1]),
                fontsize=6,
                alpha=0.9,
                xytext=(4, 4),
                textcoords="offset points",
                zorder=4,
            )
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    set_paper_style("blog")
    blob = torch.load(TENSORS / "context_vectors_mean.pt", weights_only=True)
    mean_all = blob["tensor"].float().numpy()
    ids: list[str] = blob["instance_ids"]
    families: list[str] = blob["families"]

    import umap

    fig, axes = plt.subplots(2, len(QLAYERS), figsize=(4 * len(QLAYERS), 7.6))
    for col, li in enumerate(QLAYERS):
        x = center(mean_all[:, li, :])
        pca_emb = PCA(n_components=2, random_state=SEED).fit_transform(x)
        _scatter(axes[0, col], pca_emb, families, ids)
        axes[0, col].set_title(f"PCA — layer {li}")
        um = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)
        _scatter(axes[1, col], um.fit_transform(x), families, ids)
        axes[1, col].set_title(
            f"UMAP (n_neighbors=15, min_dist=0.1, seed 42) — layer {li}", fontsize=9
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.09)
    savefig_paper(fig, "hero_embeddings_pca_umap_clean", dir=FIG_DIR)
    plt.close(fig)
    print("wrote", FIG_DIR / "hero_embeddings_pca_umap_clean.png")


if __name__ == "__main__":
    main()
