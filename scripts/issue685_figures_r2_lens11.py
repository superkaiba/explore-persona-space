#!/usr/bin/env python
"""Issue #685 clean-result-critic round-2 Lens-11 addenda figures.

Two per-unit figures requested by the clean-result-critic ensemble (round 1):

1. ``relmag_per_context`` — per-context spread of the relative-magnitude
   ``||Delta|| / median_{C!=C'} ||v(C)-v(C')||`` behind Result 1's across-context
   MEAN heatmap. One panel per behavior, layers on x, the 10 contexts as jittered
   points + the mean marker per cell. The per-unit view Lens 11 requires under the
   Result-1 aggregate.

2. ``base_per_behavior_consistency`` — the BASE model's per-behavior layer sweep of
   the raw mean-pairwise consistency cosine (the base-model analogue of the
   instruct hero panel), the per-unit decomposition Lens 11 requires under
   Result 5's instruct-vs-base AGGREGATE overlay.

CPU. Reads ``eval_results/issue_685/metrics.json`` and writes both figures
(png + pdf + meta.json) into ``figures/issue_685/`` via the paper-plots rcParams.

Usage::

    uv run python scripts/issue685_figures_r2_lens11.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

H1_COS = 0.6
H2_COS_HI = 0.4

METRICS = Path("eval_results/issue_685/metrics.json")
FIG_DIR = Path("figures/issue_685")


def fig_relmag_per_context(metrics: dict, out_dir: Path, tag: str = "instruct") -> Path:
    """Per-behavior layer sweep of the per-context relative-magnitude points.

    One panel per behavior; layers on x; the 10 contexts plotted as jittered points
    (x-jitter only) with the across-context mean drawn as a horizontal dash. This is
    the per-unit spread behind Result 1's across-context-mean heatmap.
    """
    m = metrics["models"][tag]
    behaviors = m["meta"]["behaviors"]
    layers = m["meta"]["layers"]
    ctx_names = m["meta"]["context_names"]
    n_ctx = len(ctx_names)

    n = len(behaviors)
    ncol = min(3, n)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), squeeze=False)

    point_color = paper_palette_role("primary")
    mean_color = paper_palette_role("accent")
    rng = np.random.default_rng(42)

    for i, b in enumerate(behaviors):
        ax = axes[i // ncol][i % ncol]
        for xi, L in enumerate(layers):
            cell = m["cells"][b][str(L)]["relative_magnitude"]
            per_ctx = cell["per_context"]
            jitter = (rng.random(n_ctx) - 0.5) * 0.45
            xs = np.full(n_ctx, float(L)) + jitter * (layers[1] - layers[0]) * 0.18
            ax.scatter(
                xs,
                per_ctx,
                s=14,
                color=point_color,
                alpha=0.65,
                edgecolors="none",
                zorder=2,
                label="per-context" if (i == 0 and xi == 0) else None,
            )
            ax.hlines(
                cell["mean"],
                float(L) - 1.6,
                float(L) + 1.6,
                color=mean_color,
                lw=1.6,
                zorder=3,
                label="across-context mean" if (i == 0 and xi == 0) else None,
            )
        ax.axhline(0.2, ls="--", color=paper_palette_role("neutral"), lw=0.8)
        ax.set_title(b)
        ax.set_xlabel("layer")
        ax.set_ylabel("rel. magnitude")
        ax.set_xticks(layers)
        ax.set_ylim(0.0, 2.25)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"Per-context relative magnitude of Delta(C,b) by layer ({tag}, 10 contexts)")
    paths = savefig_paper(fig, "relmag_per_context", dir=str(out_dir))
    plt.close(fig)
    return paths["png"]


def _consistency_curve(model_metrics: dict, behavior: str, layers: list[int]) -> list[float]:
    return [model_metrics["cells"][behavior][str(L)]["consistency_cosine_raw"] for L in layers]


def _null_p95(model_metrics: dict, layers: list[int]) -> list[float]:
    return [model_metrics["consistency_null"][str(L)]["p95"] for L in layers]


def fig_base_per_behavior(metrics: dict, out_dir: Path) -> Path:
    """The BASE model's per-behavior layer sweep of the raw consistency cosine.

    The base-model analogue of the instruct hero panel — the per-unit
    decomposition behind Result 5's instruct-vs-base aggregate overlay.
    """
    tag = "base"
    m = metrics["models"][tag]
    behaviors = m["meta"]["behaviors"]
    layers = m["meta"]["layers"]
    null_p95 = _null_p95(m, layers)

    n = len(behaviors)
    ncol = min(3, n)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
    primary = paper_palette_role("primary")
    for i, b in enumerate(behaviors):
        ax = axes[i // ncol][i % ncol]
        cos = _consistency_curve(m, b, layers)
        ax.plot(layers, cos, marker="o", color=primary, label="consistency")
        ax.plot(layers, null_p95, ls=":", color=paper_palette_role("neutral"), label="null p95")
        ax.axhline(H1_COS, ls="--", color=paper_palette_role("accent"), lw=0.8)
        ax.axhspan(0.0, H2_COS_HI, color=paper_palette_role("control"), alpha=0.12)
        ax.set_title(b)
        ax.set_xlabel("layer")
        ax.set_ylabel("mean pairwise cos")
        ax.set_ylim(-0.2, 1.0)
        ax.set_xticks(layers)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=6, loc="best")
    fig.suptitle("Direction consistency of Delta(C,b) by layer (base, non-instruct)")
    paths = savefig_paper(fig, "base_per_behavior_consistency", dir=str(out_dir))
    plt.close(fig)
    return paths["png"]


def main() -> None:
    set_paper_style("blog")
    metrics = json.loads(METRICS.read_text())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p1 = fig_relmag_per_context(metrics, FIG_DIR)
    p2 = fig_base_per_behavior(metrics, FIG_DIR)
    print("wrote:", p1)
    print("wrote:", p2)


if __name__ == "__main__":
    main()
