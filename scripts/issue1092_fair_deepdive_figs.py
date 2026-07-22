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

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

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


def fig_error_vs_spread() -> None:
    """Per-prefix held-out error vs within-prefix context-vector spread (raw L2).

    The natural-prefix test of the #658 coherence hypothesis (spread should
    predict the averaged-map error); Spearman rho + p per series on-figure.
    """
    from scipy.stats import spearmanr

    set_paper_style("blog")
    colors = paper_palette(2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, cell in zip(axes, CELLS, strict=True):
        a = _arrays(cell)
        s = a["spread"]
        for key, label, color in (
            ("err_ctx", "Context-vector map", colors[0]),
            ("err_prefix", "Prefix-end map", colors[1]),
        ):
            e = a[key]
            rho, p = spearmanr(s, e)
            p_txt = f"p = {p:.1g}" if p >= 1e-200 else "p < 1e-200"
            ax.scatter(
                s,
                e,
                s=8,
                alpha=0.35,
                edgecolor="none",
                color=color,
                label=f"{label} (Spearman ρ = {rho:+.2f}, {p_txt})",
            )
        ax.set_xlabel("Within-prefix context-vector spread (raw L2)")
        ax.set_ylabel("Per-prefix held-out prediction error")
        ax.set_title(CELL_LABEL[cell])
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        "Per-prefix prediction error vs within-prefix context-vector spread (996 prefixes)"
    )
    fig.tight_layout()
    savefig_paper(fig, "perprefix_error_vs_spread", dir=FIGDIR)
    plt.close(fig)


def fig_fair_grid() -> None:
    """Restructured fair comparison with explicit axes of variation:
    panels = target grain, x-groups = basis, color = input arm; per-arm
    achievable ceilings as dashed ticks (single-context panel only — the
    banked Panel-B denominators; averaged-grain ceilings are not
    artifact-backed)."""
    fc = json.loads(
        (
            PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
        ).read_text()
    )
    meta = json.loads(
        (
            PROJECT_ROOT / "figures/issue_1092/fair_comparison_prefix_vs_context.meta.json"
        ).read_text()
    )
    bases = ["ambient", "pca48"]
    cell = fc["cells"]["cell_inst_own"]["bases"]

    def _r2(basis: str, grain: str, arm: str) -> float:
        d = cell[basis][grain]
        key = [k for k in d if k.startswith(f"r2_{arm}")][0]
        return float(d[key])

    def _ceiling(basis: str, arm: str) -> float | None:
        rc = meta["raw_ceilings"][basis]
        if arm == "prefix":
            return rc.get("prefix_between_prefix_share_full")
        return rc.get("context_mlp_companion_ceiling") or rc.get(
            "context_additive_ceiling_densecore"
        )

    set_paper_style("blog")
    colors = paper_palette(2)
    arms = [("prefix", "Prefix-end map", colors[1]), ("context", "Context map", colors[0])]
    grains = [
        ("averaged_grain", "Averaged targets\n(per-prefix profile over 48 queries)", False),
        ("single_grain", "Single-context targets\n(one answer state per row)", True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    x = np.arange(len(bases))
    w = 0.36
    for ax, (grain, title, show_ceilings) in zip(axes, grains, strict=True):
        for i, (arm, label, color) in enumerate(arms):
            vals = [_r2(b, grain, arm) for b in bases]
            ax.bar(x + (i - 0.5) * w, vals, w, label=label, color=color)
            if show_ceilings:
                for xc, b in zip(x, bases, strict=True):
                    c = _ceiling(b, arm)
                    if c is not None:
                        ax.plot(
                            [xc + (i - 1) * w, xc + i * w],
                            [c, c],
                            ls="--",
                            color="0.25",
                            lw=1.3,
                        )
        ax.set_xticks(x)
        ax.set_xticklabels(["ambient basis", "pca48 basis"])
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("held-out $R^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.append(Line2D([0], [0], ls="--", color="0.25", lw=1.3))
    labels.append("achievable ceiling per arm")
    axes[0].legend(handles, labels, loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        "Prefix-end vs context map by target grain — instruct model, own answers, layer 14"
    )
    fig.tight_layout()
    savefig_paper(fig, "fair_comparison_grid", dir=FIGDIR)
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
    fig_error_vs_spread()
    fig_shrinkage()
    print(f"wrote figures to {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
