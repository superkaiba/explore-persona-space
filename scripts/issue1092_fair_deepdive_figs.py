"""Figures for the #1092 fair-comparison deep-dive (analysis-only).

Reads the persisted per-prefix arrays (per_prefix_arrays_<cell>.npz) and the
deep-dive JSONs; writes into figures/summaries/prefix_vs_context_map/.

  fig1 per-prefix error scatter: prefix-map error vs context-map error, one
       panel per cell, with y=x and y=2x reference lines (the prefix map is
       worse on EVERY prefix; the gap is a ~2x band).
  fig2 per-prefix error vs prefix length (conversation turns), one panel per
       cell (length is the dominant per-prefix difficulty axis).
  fig3 shrinkage residual variance: single global scalar vs per-dimension
       diagonal, per cell (pca48), showing a single scalar captures most of the
       prefix<->context between-prefix relationship.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DD = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison_deepdive"
FIGDIR = PROJECT_ROOT / "figures/summaries/prefix_vs_context_map"
CELLS = ["cell_inst_own", "cell_pre_own"]
CELL_LABEL = {
    "cell_inst_own": "Instruct model (own answers)",
    "cell_pre_own": "Pretrained-base model (own answers)",
}


def _arrays(cell: str) -> dict:
    z = np.load(DD / f"per_prefix_arrays_{cell}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def fig_error_scatter() -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, cell in zip(axes, CELLS, strict=True):
        a = _arrays(cell)
        ec, ep = a["err_ctx"], a["err_prefix"]
        ax.scatter(ec, ep, s=8, alpha=0.35, edgecolor="none")
        hi = float(max(ec.max(), ep.max())) * 1.02
        ax.plot([0, hi], [0, hi], color="0.35", lw=1.2, ls="--", label="equal error")
        ax.plot([0, hi], [0, 2 * hi], color="0.55", lw=1.0, ls=":", label="prefix error = 2x")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel("Context-vector map per-prefix error")
        ax.set_ylabel("Prefix-end map per-prefix error")
        ax.set_title(CELL_LABEL[cell])
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Per-prefix prediction error: prefix-end map vs context-vector map (996 prefixes)")
    fig.tight_layout()
    savefig_paper(fig, "perprefix_error_scatter", dir=FIGDIR)
    plt.close(fig)


def fig_error_vs_length() -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, cell in zip(axes, CELLS, strict=True):
        a = _arrays(cell)
        nt, ec = a["n_turns"].astype(float), a["err_ctx"]
        ax.scatter(nt, ec, s=8, alpha=0.35, edgecolor="none")
        ax.set_xlabel("Prefix conversation turns")
        ax.set_ylabel("Context-vector map per-prefix error")
        ax.set_title(CELL_LABEL[cell])
    fig.suptitle("Longer prefixes are harder to predict for both maps")
    fig.tight_layout()
    savefig_paper(fig, "perprefix_error_vs_length", dir=FIGDIR)
    plt.close(fig)


def fig_shrinkage() -> None:
    dd = json.loads((DD / "deepdive.json").read_text())
    labels, glob_res, diag_res = [], [], []
    for cell in CELLS:
        key = f"{cell}/pca48"
        if key not in dd.get("cells", {}):
            continue
        s = dd["cells"][key]["shrinkage"]
        labels.append(CELL_LABEL[cell])
        glob_res.append(s["global_scalar_P_from_C"]["resid_var_frac"])
        diag_res.append(s["per_dim_diagonal_P_from_C"]["resid_var_frac"])
    if not labels:
        return
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, glob_res, w, label="single global scalar")
    ax.bar(x + w / 2, diag_res, w, label="per-dimension diagonal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Residual variance fraction (lower = better fit)")
    ax.set_ylim(0, 1)
    ax.set_title("Prefix predictions are a near-uniform shrinkage of context predictions")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "shrinkage_residual_variance", dir=FIGDIR)
    plt.close(fig)


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_error_scatter()
    fig_error_vs_length()
    fig_shrinkage()
    print(f"wrote figures to {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
