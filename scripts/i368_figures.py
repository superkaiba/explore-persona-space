"""Paper-quality figures for issue #368 clean-result body.

Generates three figures under `figures/issue_368/`:

1. `hero_two_phase.{png,pdf}`  — headline: two-phase comparison of
   semantic-cos / JS / centered-cos / pvec_chenstyle_L20 against the
   leakage signal. The figure that carries the H1 + H2 verdict.
2. `phase1_recipe_panel.{png,pdf}` — Phase 1 Spearman ρ per recipe with
   bootstrap CIs, comparing the 6 Chen-style recipes vs the two centroid
   baselines. Drives home "no Chen-style recipe carries leakage signal,
   centroids do".
3. `recipe_agreement_heatmap.{png,pdf}` — 8x8 cross-recipe Spearman ρ
   matrix from Phase 1 showing H3a fails (off-diagonal mean 0.39 < 0.7).

All figures use `set_paper_style("blog")` and `savefig_paper` with
commit-pinned `.meta.json` sidecars.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parents[1]
P1 = REPO / "eval_results" / "issue_368" / "phase1"
P2 = REPO / "eval_results" / "issue_368" / "phase2"
OUTDIR_REL = "issue_368"
FIG_ROOT = REPO / "figures"


def _load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Hero — two-phase comparison
# ---------------------------------------------------------------------------
def fig_hero() -> None:
    p1 = _load_json(P1 / "h1_verdict.json")
    p2 = _load_json(P2 / "h2_verdict.json")
    p1_axes = _load_json(P1 / "per_axis_stats.json")["per_axis"]
    p2_axes = _load_json(P2 / "per_axis_stats.json")["per_axis"]

    set_paper_style("blog")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Phase 1 panel: semantic_cos baseline vs Chen-style pvec_L20 (signed rho)
    labels1 = ["semantic_cos\n(baseline)", "pvec_chenstyle\nL20"]
    rho1 = [
        p1_axes["semantic_cos"]["spearman_rho"],
        p1_axes["pvec_chenstyle_L20"]["spearman_rho"],
    ]
    ci_lo1 = [
        p1_axes["semantic_cos"]["bootstrap_cluster_test_id_95ci"][0],
        p1_axes["pvec_chenstyle_L20"]["bootstrap_cluster_test_id_95ci"][0],
    ]
    ci_hi1 = [
        p1_axes["semantic_cos"]["bootstrap_cluster_test_id_95ci"][1],
        p1_axes["pvec_chenstyle_L20"]["bootstrap_cluster_test_id_95ci"][1],
    ]
    err_lo1 = [r - lo for r, lo in zip(rho1, ci_lo1)]
    err_hi1 = [hi - r for hi, r in zip(ci_hi1, rho1)]
    colors1 = [paper_palette_role("baseline"), paper_palette_role("primary")]

    ax1.bar(
        range(2),
        rho1,
        color=colors1,
        width=0.55,
        yerr=[err_lo1, err_hi1],
        error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
    )
    ax1.axhline(0, color="#1A1A1A", linewidth=0.6)
    ax1.axhline(0.55, color="#999", linewidth=0.6, linestyle="--")
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(labels1)
    ax1.set_ylabel("Spearman ρ vs marker_rate")
    ax1.set_ylim(-0.5, 1.0)
    set_title_subtitle(
        ax1,
        "Phase 1 — non-persona triggers (N=128 cells)",
        subtitle="H1 threshold ρ ≥ 0.55 (dashed)",
    )

    # Phase 2 panel: JS / centered-cos-L20 / pvec_chenstyle_L20 (signed rho).
    # The pvec headline number is the H2 "marginal" rho from h2_verdict.json
    # (post source=assistant filter; n=40), not per_axis_stats.json's value.
    labels2 = ["JS-divergence\n(prior)", "centered-cos\nL20 (prior)", "pvec_chenstyle\nL20"]
    pvec_rho = p2["h2_verdict"]["marginal"]["rho"]  # 0.0336 signed, n=40
    rho2 = [
        0.746,  # from #142 (absolute value; sign was negative for JS)
        0.567,  # from Method-A reproduction
        pvec_rho,  # signed (~0.034)
    ]
    # T13 source-shuffle null gives a same-scale reference: the null's 95th
    # percentile is 0.292, and the observed |ρ| is well inside it. We draw a
    # symmetric "indistinguishable from null" band on the pvec bar.
    null_p95 = p2["h2_verdict"]["T13_source_shuffle_null"]["null_95th_percentile"]
    err_lo2 = [0.0, 0.0, max(pvec_rho - (-null_p95), 0.0)]
    err_hi2 = [0.0, 0.0, max(null_p95 - pvec_rho, 0.0)]
    colors2 = [
        paper_palette_role("baseline"),
        paper_palette_role("control"),
        paper_palette_role("primary"),
    ]
    ax2.bar(
        range(3),
        rho2,
        color=colors2,
        width=0.55,
        yerr=[err_lo2, err_hi2],
        error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
    )
    ax2.axhline(0, color="#1A1A1A", linewidth=0.6)
    ax2.axhline(0.75, color="#999", linewidth=0.6, linestyle="--")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels2)
    ax2.set_ylabel("Spearman ρ vs marker_leakage_rate")
    ax2.set_ylim(-0.5, 1.0)
    set_title_subtitle(
        ax2,
        "Phase 2 — personas (n=40 directed pairs)",
        subtitle="H2 threshold |ρ| ≥ 0.75 (dashed); JS prior sign was negative",
    )

    fig.tight_layout()
    savefig_paper(fig, f"{OUTDIR_REL}/hero_two_phase", dir=str(FIG_ROOT))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 1 recipe panel — all 6 Chen-style + 2 centroid baselines
# ---------------------------------------------------------------------------
def fig_phase1_recipes() -> None:
    p1_axes = _load_json(P1 / "per_axis_stats.json")["per_axis"]

    chen_recipes = [
        ("pvec_chenstyle_L15", "pvec L15"),
        ("pvec_chenstyle_L20", "pvec L20"),
        ("pvec_chenstyle_L25", "pvec L25"),
        ("pvec_chenstyle_lasttoken", "pvec last-tok"),
        ("pvec_chenstyle_orthog", "pvec orthog"),
        ("pvec_chenstyle_L20_projdiff", "pvec L20-projdiff"),
    ]
    centroids = [
        ("pcentroid_methodA_L20", "centroid\nMethod-A"),
        ("pcentroid_chenstyle_pos_only_L20", "centroid\npos-only"),
    ]
    baseline = ("semantic_cos", "semantic_cos\n(baseline)")

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(11.0, 4.3))

    all_axes = [baseline, *chen_recipes, *centroids]
    labels = [lbl for _, lbl in all_axes]
    rhos = [p1_axes[k]["spearman_rho"] for k, _ in all_axes]
    ci_los = [p1_axes[k]["bootstrap_cluster_test_id_95ci"][0] for k, _ in all_axes]
    ci_his = [p1_axes[k]["bootstrap_cluster_test_id_95ci"][1] for k, _ in all_axes]
    err_lo = [r - lo for r, lo in zip(rhos, ci_los)]
    err_hi = [hi - r for hi, r in zip(ci_his, rhos)]

    colors = (
        [paper_palette_role("baseline")]
        + [paper_palette_role("primary")] * len(chen_recipes)
        + [paper_palette_role("accent")] * len(centroids)
    )

    ax.bar(
        range(len(labels)),
        rhos,
        color=colors,
        width=0.6,
        yerr=[err_lo, err_hi],
        error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
    )
    ax.axhline(0, color="#1A1A1A", linewidth=0.6)
    ax.axhline(0.55, color="#999", linewidth=0.6, linestyle="--", label="H1 threshold (0.55)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Spearman ρ vs marker_rate")
    ax.set_ylim(-0.5, 1.0)
    ax.legend(loc="upper left", frameon=False)
    set_title_subtitle(
        ax,
        "Phase 1 — recipe ρ comparison",
        subtitle="6 Chen-style recipes vs 2 centroid baselines (N=128, 95% bootstrap CI)",
    )

    fig.tight_layout()
    savefig_paper(fig, f"{OUTDIR_REL}/phase1_recipe_panel", dir=str(FIG_ROOT))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Recipe agreement heatmap (Phase 1)
# ---------------------------------------------------------------------------
def fig_recipe_agreement() -> None:
    mat = pd.read_csv(P1 / "recipe_agreement_matrix_with_projdiff.csv", index_col=0)
    perm = _load_json(P1 / "permutation_null.json")
    off_diag_mean_no_proj = perm["without_projdiff"]["off_diagonal_mean"]
    off_diag_mean_with_proj = perm["with_projdiff"]["off_diagonal_mean"]

    # Shorten labels
    rename = {
        "pvec_chenstyle_L20": "pvec L20",
        "pvec_chenstyle_L15": "pvec L15",
        "pvec_chenstyle_L25": "pvec L25",
        "pvec_chenstyle_lasttoken": "pvec last-tok",
        "pvec_chenstyle_orthog": "pvec orthog",
        "pvec_chenstyle_L20_projdiff": "pvec L20-projdiff",
        "pcentroid_methodA_L20": "centroid Method-A",
        "pcentroid_methodB_L20": "centroid Method-B",
    }
    mat = mat.rename(index=rename, columns=rename)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)

    # Annotate cells
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            v = mat.values[i, j]
            color = "white" if abs(v) > 0.55 else "#1A1A1A"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ across cells")

    set_title_subtitle(
        ax,
        "Phase 1 — recipe-agreement matrix",
        subtitle=(
            f"Off-diagonal mean = {off_diag_mean_with_proj:.2f} (with projdiff) / "
            f"{off_diag_mean_no_proj:.2f} (without); H3a threshold = 0.70"
        ),
    )

    savefig_paper(fig, f"{OUTDIR_REL}/recipe_agreement_heatmap", dir=str(FIG_ROOT))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_ROOT / OUTDIR_REL, exist_ok=True)
    fig_hero()
    fig_phase1_recipes()
    fig_recipe_agreement()
    print("Wrote figures to", FIG_ROOT / OUTDIR_REL)
