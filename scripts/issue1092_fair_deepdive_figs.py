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
    basis = "ambient"
    fig, ax = plt.subplots(figsize=(8.0, 5))
    x = np.arange(len(grains))
    w = 0.36
    for i, (arm, label, color) in enumerate(arms):
        vals = [_r2(basis, grain, arm) for grain, _, _ in grains]
        ax.bar(x + (i - 0.5) * w, vals, w, label=label, color=color)
        for xc, (_, _, show_ceilings) in zip(x, grains, strict=True):
            if show_ceilings:
                c = _ceiling(basis, arm)
                if c is not None:
                    ax.plot(
                        [xc + (i - 1) * w, xc + i * w],
                        [c, c],
                        ls="--",
                        color="0.25",
                        lw=1.3,
                    )
    ax.set_xticks(x)
    ax.set_xticklabels([title for _, title, _ in grains])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("held-out $R^2$")
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.append(Line2D([0], [0], ls="--", color="0.25", lw=1.3))
    labels.append("achievable ceiling per arm")
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    ax.set_title(
        "Prefix-end vs context map by target grain — instruct model, own answers, layer 14"
    )
    fig.tight_layout()
    savefig_paper(fig, "fair_comparison_grid", dir=FIGDIR)
    plt.close(fig)


def fig_fair_grid_averaged_only() -> None:
    """Averaged-targets-only fair comparison: one bar group per model
    (instruct, pretrained-base), direct prefix map vs averaged prefix map,
    ambient basis. The averaged grain is the only grain where the direct
    prefix vector has a fair shot (no query access), so single-context
    targets and the per-arm ceilings are omitted."""
    fc = json.loads(
        (
            PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
        ).read_text()
    )
    basis = "ambient"
    set_paper_style("blog")
    colors = paper_palette(2)
    arms = [
        ("r2_prefix_averaged", "Direct prefix map (prefix-end vector)", colors[1]),
        (
            "r2_context_averaged",
            "Averaged prefix map (context-end, averaged over queries)",
            colors[0],
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(CELLS))
    w = 0.36
    for i, (key, label, color) in enumerate(arms):
        vals = [float(fc["cells"][cell]["bases"][basis]["averaged_grain"][key]) for cell in CELLS]
        ax.bar(x + (i - 0.5) * w, vals, w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABEL[cell] for cell in CELLS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("held-out $R^2$")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Predicting the per-prefix average answer state — averaged targets, layer 14")
    fig.tight_layout()
    savefig_paper(fig, "fair_comparison_averaged_only", dir=FIGDIR)
    plt.close(fig)


def fig_grain_skill() -> None:
    """Single-context map held-out R2 at both grains (ambient, L14), both models.

    The writeup's Result 1 plot: the same fitted operator scored on per-row
    targets and on per-prefix averaged targets (the induced read).
    """
    fc = json.loads(
        (
            PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
        ).read_text()
    )
    cells = [("cell_inst_own", "Instruct model"), ("cell_pre_own", "Base model")]
    single = [
        float(
            fc["cells"][c]["bases"]["ambient"]["single_grain"]["r2_context_battery_excluded_full"]
        )
        for c, _ in cells
    ]
    avg = [
        float(fc["cells"][c]["bases"]["ambient"]["averaged_grain"]["r2_context_averaged"])
        for c, _ in cells
    ]
    set_paper_style("blog")
    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, [single[0], avg[0]], w, label=cells[0][1], color=colors[0])
    ax.bar(x + w / 2, [single[1], avg[1]], w, label=cells[1][1], color=colors[2])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Single-context targets\n(one answer state per row)",
            "Averaged targets\n(per-prefix mean over queries)",
        ]
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("held-out $R^2$")
    ax.set_title("Single-context map scored at both grains — layer 14, ambient")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "grain_skill", dir=FIGDIR)
    plt.close(fig)


def fig_induced_vs_refit() -> None:
    """Averaged-grain held-out R2: induced read vs independently-fit averaged map.

    The writeup's Result 2 plot, from the operator-coincidence artifact
    (ambient, L14, aligned novel-prefix folds).
    """
    oc = json.loads(
        (
            PROJECT_ROOT
            / "eval_results/issue_1092/inline_operator_coincidence/operator_coincidence.json"
        ).read_text()
    )
    cells = [("cell_inst_own", "Instruct model"), ("cell_pre_own", "Base model")]
    induced = [float(oc["cells"][c]["bases"]["ambient"]["r2_induced_avg"]) for c, _ in cells]
    refit = [float(oc["cells"][c]["bases"]["ambient"]["r2_refit_avg"]) for c, _ in cells]
    set_paper_style("blog")
    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(2)
    w = 0.36
    ax.bar(
        x - w / 2,
        induced,
        w,
        label="Induced (single-context map, predictions averaged)",
        color=colors[0],
    )
    ax.bar(
        x + w / 2,
        refit,
        w,
        label="Refit (fit directly on averaged vectors)",
        color=colors[2],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in cells])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("held-out $R^2$, averaged targets")
    ax.set_title("Two constructions of the averaged prefix map — layer 14, ambient")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "induced_vs_refit", dir=FIGDIR)
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
