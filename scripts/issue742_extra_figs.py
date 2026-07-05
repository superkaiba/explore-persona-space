#!/usr/bin/env python
# ruff: noqa: RUF002
"""Issue #742 — analyzer extra figures (committed provenance for the round-1 /tmp script).

Renders the two analyzer-added figures:
  * **Hero (b)** — Stage-1 post-LEACE dCor observed vs permutation-null median vs
    shuffled-label control, per headroom cell (``hero_b_stage1_dcor``).
  * **Low-level per-unit plot** — per-context judged base expression rates ``E0`` per
    behavior × genre (``stage0_e0_per_context``), the underlying data behind Stage 0.

Chart-internal labels are plain English (no snake_case condition codes; project rule).
Writes PNG + PDF + ``.meta.json`` via ``savefig_paper`` to ``figures/issue_742/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from issue742_figures import behavior_label, genre_label  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

set_paper_style("blog")
FIG_DIR = PROJECT_ROOT / "figures" / "issue_742"


def plot_stage1_dcor(out_dir: Path) -> None:
    """Hero (b): grouped bars of observed / null-median / control dCor per headroom cell."""
    s1 = json.loads((PROJECT_ROOT / "eval_results/issue_742/stage1_leace_dcor.json").read_text())
    cells = s1["cells"]
    labels = [f"{behavior_label(c['behavior'])}\n({genre_label(c['genre'])})" for c in cells]
    obs = [c["dcor_observed"] for c in cells]
    nullmed = [c["dcor_null_median"] for c in cells]
    ctrl = [c["control_task_dcor"] for c in cells]
    pvals = [c["dcor_p_value"] for c in cells]

    colors = paper_palette_blog(3)
    x = np.arange(len(cells))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(x - w, obs, w, label="observed dCor (LEACE-erased)", color=colors[0])
    ax.bar(x, nullmed, w, label="permutation-null median", color=colors[1])
    ax.bar(x + w, ctrl, w, label="shuffled-label control dCor", color=colors[2])
    for i, (o, p) in enumerate(zip(obs, pvals, strict=True)):
        ax.text(i - w, o + 0.012, f"p={p:.3f}", ha="center", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("distance correlation")
    ax.set_title("Stage 1 - post-LEACE nonlinear residual (4 headroom cells)", pad=14)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_ylim(0, 0.58)
    print("wrote", savefig_paper(fig, "hero_b_stage1_dcor", dir=out_dir))
    plt.close(fig)


def plot_e0_per_context(out_dir: Path) -> None:
    """Low-level per-unit plot: per-context judged base expression rate per behavior x genre."""
    e0_paths = {
        "betley": "eval_results/issue_658/E0_expression.json",
        "ultrachat": "eval_results/issue_658/E0_expression_g1.json",
    }
    behaviors = ["broad_em", "harmful_compliance", "sycophancy", "refusal"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    rng = np.random.default_rng(742)
    for ax, (genre, path) in zip(axes, e0_paths.items(), strict=True):
        e0 = json.loads((PROJECT_ROOT / path).read_text())["e0"]
        for j, beh in enumerate(behaviors):
            rates = np.array([e0[c][beh]["rate"] for c in e0])
            jit = rng.uniform(-0.16, 0.16, size=len(rates))
            ax.scatter(
                np.full(len(rates), j) + jit,
                rates,
                s=22,
                alpha=0.6,
                color=paper_palette_blog(4)[j],
                edgecolors="none",
            )
            ax.hlines(rates.mean(), j - 0.28, j + 0.28, color="black", linewidth=1.6)
        ax.set_xticks(range(len(behaviors)))
        ax.set_xticklabels([behavior_label(b).replace(" ", "\n") for b in behaviors])
        ax.set_title(f"{genre_label(genre)}: per-context base expression rate", pad=12)
    axes[0].set_ylabel("judged behavior rate E0 (per context, n=50)")
    print("wrote", savefig_paper(fig, "stage0_e0_per_context", dir=out_dir))
    plt.close(fig)


def main() -> int:
    """Render both analyzer extra figures to figures/issue_742/."""
    plot_stage1_dcor(FIG_DIR)
    plot_e0_per_context(FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
