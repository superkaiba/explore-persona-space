#!/usr/bin/env python3
"""Issue #269 plotting: produces the 6 figures specified in plan §4 / §17.

Reads ``eval_results/issue_269/geometry_alignment.json`` + ``js_matrix.json`` +
``experiments/phase_minus1_persona_vectors/cosine_matrix.json`` and writes:

  (i)   hero_dual_heatmap_n19.{png,pdf}
        Side-by-side (1 - cos_L10) + JS heatmaps in TWO orderings:
        hierarchical-clustering and alphabetical.
  (ii)  cosine_vs_js_scatter_n171.{png,pdf}
        Scatter with cluster_macro color + cluster_fine shape +
        top-5 baseline-residual outliers labeled.
  (iii) rho_by_layer_7stat.{png,pdf}
        Grouped bar chart: 7 statistics x 4 layers, one-sided Mantel-p
        annotations on raw bars.
  (iv)  rho_by_T_cutoff.{png,pdf}
        Bar chart of rho at T=8, T=32, T=full with T=8 gate threshold line.
  (v)   jackknife_n19.{png,pdf}
        Boxplot of 19 leave-one-persona-out rho values; outliers named.
  (vi)  ha_excluded_sensitivity.{png,pdf}
        Bar chart: raw rho, joint-partial, mean-marginal residual at
        n=171 vs n=153 (HA-excluded).

Uses the ``paper-plots`` skill conventions
(``src/explore_persona_space/analysis/paper_plots.py``).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "experiments" / "phase_minus1_persona_vectors"))

from extract_persona_vectors import PERSONAS  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("issue_269_plots")

RESULTS_PATH = _PROJECT_ROOT / "eval_results" / "issue_269" / "geometry_alignment.json"
JS_PATH = _PROJECT_ROOT / "eval_results" / "issue_269" / "js_matrix.json"
COSINE_PATH = _PROJECT_ROOT / "experiments" / "phase_minus1_persona_vectors" / "cosine_matrix.json"
FIG_DIR = _PROJECT_ROOT / "figures" / "issue_269"

HEADLINE_LAYER = 10

CLUSTERS_FINE: dict[str, set[str]] = {
    "medical": {"medical_doctor", "surgeon", "paramedic", "army_medic"},
    "security": {"cybersec_consultant", "pentester", "private_investigator"},
    "services": {"navy_seal", "police_officer"},
    "tech": {"software_engineer", "data_scientist"},
}


def cluster_fine_of(name: str) -> str:
    for c, members in CLUSTERS_FINE.items():
        if name in members:
            return c
    return "civilian"


def cluster_macro_of(name: str) -> str:
    return "occupational" if cluster_fine_of(name) != "civilian" else "civilian"


# ── (i) Hero: dual heatmap ────────────────────────────────────────────────────
def figure_hero_dual_heatmap(
    dist_cos_19: np.ndarray, js_19: np.ndarray, names_19: list[str]
) -> None:
    """Side-by-side heatmaps in two orderings (clustering + alphabetical)."""
    # Build composite distance for clustering: 0.5 * normalized cos + 0.5 * normalized JS.
    d_cos_norm = dist_cos_19 / dist_cos_19.max()
    d_js_norm = js_19 / js_19.max()
    composite = 0.5 * d_cos_norm + 0.5 * d_js_norm
    np.fill_diagonal(composite, 0.0)
    Z = linkage(squareform(composite, checks=False), method="average")
    cluster_order = leaves_list(Z)
    alpha_order = sorted(range(len(names_19)), key=lambda i: names_19[i])
    orderings = [
        ("Hierarchical clustering", cluster_order),
        ("Alphabetical", alpha_order),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    for row_idx, (matrix, mat_label) in enumerate(
        [(dist_cos_19, "1 - cosine (L10)"), (js_19, "JS (T=full)")]
    ):
        for col_idx, (order_label, order) in enumerate(orderings):
            ax = axes[row_idx, col_idx]
            M = matrix[np.ix_(order, order)]
            ordered_names = [names_19[i] for i in order]
            im = ax.imshow(M, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(ordered_names)))
            ax.set_xticklabels(ordered_names, rotation=90, fontsize=7)
            ax.set_yticks(range(len(ordered_names)))
            ax.set_yticklabels(ordered_names, fontsize=7)
            ax.set_title(f"{mat_label}  |  {order_label}", fontsize=10)
            cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
            cbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        "Persona geometry: cosine (L10) vs JS divergence — n=19, no_persona excluded",
        fontsize=11,
        y=1.00,
    )
    fig.tight_layout()
    savefig_paper(fig, "hero_dual_heatmap_n19", dir=FIG_DIR)
    plt.close(fig)
    log.info("Wrote (i) hero heatmap")


# ── (ii) Scatter with cluster colors + top-5 residual labels ──────────────────
def figure_scatter_with_residuals(
    v_cos: np.ndarray,
    v_js: np.ndarray,
    iu_19: tuple[np.ndarray, np.ndarray],
    names_19: list[str],
    top5_residual_pairs: list[dict],
    rho_raw: float,
    rho_resid_mm: float,
) -> None:
    """Scatter v_cos vs v_js. Color by cluster_macro (same / cross),
    annotate the 5 top baseline-residual pairs.
    """
    pair_indices = list(zip(iu_19[0].tolist(), iu_19[1].tolist(), strict=True))
    macro_match = np.array(
        [cluster_macro_of(names_19[i]) == cluster_macro_of(names_19[j]) for (i, j) in pair_indices]
    )
    fine_match = np.array(
        [
            cluster_fine_of(names_19[i]) == cluster_fine_of(names_19[j])
            and cluster_fine_of(names_19[i]) != "civilian"
            for (i, j) in pair_indices
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")
    # Plot in three layers: cross-macro, same-macro-not-fine, same-fine.
    cross_macro = ~macro_match
    same_macro_only = macro_match & ~fine_match
    same_fine = fine_match
    ax.scatter(
        v_cos[cross_macro],
        v_js[cross_macro],
        c=baseline,
        s=22,
        alpha=0.55,
        label=f"cross-macro (n={int(cross_macro.sum())})",
    )
    ax.scatter(
        v_cos[same_macro_only],
        v_js[same_macro_only],
        c=primary,
        s=22,
        alpha=0.65,
        label=f"same macro / cross fine (n={int(same_macro_only.sum())})",
    )
    ax.scatter(
        v_cos[same_fine],
        v_js[same_fine],
        c=primary,
        s=36,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.6,
        marker="D",
        label=f"same fine cluster (n={int(same_fine.sum())})",
    )

    # Annotate top-5 residual pairs.
    for entry in top5_residual_pairs:
        i_name, j_name = entry["pair"]
        x, y = entry["v_cos"], entry["v_js"]
        ax.annotate(
            f"({i_name}, {j_name})",
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.4, alpha=0.75),
        )
        ax.scatter([x], [y], facecolors="none", edgecolors="red", s=80, linewidths=1.0)

    ax.set_xlabel("1 - cosine (L10) pairwise distance")
    ax.set_ylabel("JS divergence (T=full)")
    ax.set_title(
        f"Cosine vs JS over n=171 pairs (no_persona excluded)\n"
        f"raw rho={rho_raw:.3f} | mean-marginal-residual rho={rho_resid_mm:.3f}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    savefig_paper(fig, "cosine_vs_js_scatter_n171", dir=FIG_DIR)
    plt.close(fig)
    log.info("Wrote (ii) scatter")


# ── (iii) Sensitivity bar: 7 statistics x 4 layers ────────────────────────────
def figure_sensitivity_by_layer(layers_results: dict[str, dict]) -> None:
    """7 statistics across 4 layers as a grouped bar chart."""
    stats_keys = [
        ("rho_raw", "raw rho (n=171)"),
        ("rho_cluster_mask_n160", "cluster-mask rho (n=160)"),
        ("rho_partial_cluster_fine", "partial rho | cluster_fine"),
        ("rho_partial_cluster_joint", "partial rho | fine + macro [GATING]"),
        ("rho_resid_baseline_mean_marginal", "resid rho | b_mean_marginal [GATING]"),
        ("rho_resid_baseline_no_persona", "resid rho | b_no_persona"),
        ("rho_cluster_collapsed_n66", "cluster-collapsed rho (n=66)"),
    ]
    layer_keys = sorted(layers_results.keys(), key=int)
    n_stats = len(stats_keys)
    n_layers = len(layer_keys)
    x = np.arange(n_stats)
    width = 0.85 / n_layers

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = paper_palette(n_layers)
    for li, layer_key in enumerate(layer_keys):
        layer = layers_results[layer_key]
        vals = [layer.get(k, np.nan) for k, _ in stats_keys]
        bars = ax.bar(
            x + li * width,
            vals,
            width,
            color=palette[li],
            label=f"L{layer_key}",
            edgecolor="black",
            linewidth=0.5,
        )
        # Annotate raw bars with Mantel p (one-sided).
        if "rho_raw" in [k for k, _ in stats_keys]:
            raw_idx = next(i for i, (k, _) in enumerate(stats_keys) if k == "rho_raw")
            p_val = layer.get("p_mantel_one_sided", np.nan)
            bar = bars[raw_idx]
            label_p = (
                f"p={p_val:.1e}"
                if (isinstance(p_val, (int, float)) and p_val < 0.01)
                else f"p={p_val:.3f}"
            )
            ax.annotate(
                label_p,
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=45,
            )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(0.4, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.axhline(0.2, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.set_ylabel("Spearman rho")
    ax.set_xticks(x + width * (n_layers - 1) / 2.0)
    ax.set_xticklabels([label for _, label in stats_keys], rotation=20, ha="right", fontsize=8)
    ax.set_title("Sensitivity: 7 RSA statistics x 4 layers", fontsize=10)
    ax.legend(title="layer", fontsize=8, loc="upper right")
    ax.set_ylim(-0.2, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    savefig_paper(fig, "rho_by_layer_7stat", dir=FIG_DIR)
    plt.close(fig)
    log.info("Wrote (iii) layer sensitivity")


# ── (iv) T-cutoff bar with gate line ──────────────────────────────────────────
def figure_t_cutoff(headline: dict) -> None:
    rho_T8 = headline["rho_T8"]
    rho_T32 = headline["rho_T32"]
    rho_Tfull = headline["rho_Tfull"]
    gate_threshold = 0.3 * rho_Tfull
    ratio = headline.get("t8_gate_ratio")
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = paper_palette(3)
    labels = ["T=8", "T=32", "T=full"]
    vals = [rho_T8, rho_T32, rho_Tfull]
    bars = ax.bar(labels, vals, color=palette, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axhline(
        gate_threshold,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"T=8 gate threshold (0.3 x rho_Tfull = {gate_threshold:.3f})",
    )
    ax.set_ylabel("Spearman rho (n=171 at L10)")
    ax.set_title(
        f"T-cutoff sensitivity (T=8 GATING) — t8_ratio = {ratio:.3f}"
        if ratio is not None
        else "T-cutoff sensitivity",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.0, max(1.0, max(vals) * 1.15))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    savefig_paper(fig, "rho_by_T_cutoff", dir=FIG_DIR)
    plt.close(fig)
    log.info("Wrote (iv) T-cutoff bar")


# ── (v) Jackknife boxplot with outlier names ──────────────────────────────────
def figure_jackknife(headline: dict) -> None:
    jk = headline["jackknife"]
    values = jk["values"]
    names = jk["names_dropped"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        values,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor=paper_palette_role("primary")),
    )
    # Identify outliers: any leave-one-out value that lifts/drops the range.
    sorted_pairs = sorted(zip(values, names, strict=True))
    # Bottom-2 and top-2 by impact (smallest and largest leave-one-out rho mean leverage).
    extreme_idx = [
        names.index(sorted_pairs[0][1]),
        names.index(sorted_pairs[1][1]),
        names.index(sorted_pairs[-2][1]),
        names.index(sorted_pairs[-1][1]),
    ]
    for ei in extreme_idx:
        ax.annotate(
            names[ei],
            xy=(1.0, values[ei]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7,
            va="center",
        )
        ax.scatter([1.0], [values[ei]], color="red", s=14, zorder=10)
    ax.set_xticks([1])
    ax.set_xticklabels(["leave-one-persona-out rho (n=19 personas, 153 pairs each)"], fontsize=8)
    ax.set_ylabel("Spearman rho")
    ax.set_title(
        f"Jackknife (n=19) — median={jk['median']:.3f}, "
        f"IQR={jk['iqr']:.3f}, range={jk['range']:.3f}",
        fontsize=10,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    savefig_paper(fig, "jackknife_n19", dir=FIG_DIR)
    plt.close(fig)
    log.info("Wrote (v) jackknife")


# ── (vi) HA-excluded sensitivity bar ──────────────────────────────────────────
def figure_ha_excluded(headline: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = paper_palette(2)
    metric_keys = [
        ("rho_raw", "rho_raw_ha_excluded_n153", "raw rho"),
        (
            "rho_partial_cluster_joint",
            "rho_partial_cluster_joint_ha_excluded_n153",
            "partial rho | fine + macro",
        ),
        (
            "rho_resid_baseline_mean_marginal",
            "rho_resid_baseline_mean_marginal_ha_excluded_n153",
            "resid rho | b_mean_marginal",
        ),
    ]
    labels = [m[2] for m in metric_keys]
    x = np.arange(len(labels))
    width = 0.4
    full_vals = [headline[m[0]] for m in metric_keys]
    excl_vals = [headline[m[1]] for m in metric_keys]
    ax.bar(
        x - width / 2.0,
        full_vals,
        width,
        color=palette[0],
        edgecolor="black",
        linewidth=0.5,
        label="n=171 (full)",
    )
    ax.bar(
        x + width / 2.0,
        excl_vals,
        width,
        color=palette[1],
        edgecolor="black",
        linewidth=0.5,
        label="n=153 (HA-excluded)",
    )
    for xi, (vf, ve) in enumerate(zip(full_vals, excl_vals, strict=True)):
        ax.text(xi - width / 2.0, vf + 0.01, f"{vf:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + width / 2.0, ve + 0.01, f"{ve:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Spearman rho (L10)")
    delta = headline["ha_excluded_delta_raw"]
    ax.set_title(
        f"HA-excluded sensitivity — delta_raw = {delta:+.3f} "
        f"({'load-bearing' if abs(delta) > 0.1 else 'not load-bearing'})",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    savefig_paper(fig, "ha_excluded_sensitivity", dir=FIG_DIR)
    plt.close(fig)
    log.info("Wrote (vi) HA-excluded sensitivity")


# ── Main ──────────────────────────────────────────────────────────────────────
def load_inputs() -> tuple[dict, dict, dict]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found; run analyze_persona_geometry_vs_divergence.py first"
        )
    if not JS_PATH.exists():
        raise FileNotFoundError(f"{JS_PATH} not found")
    if not COSINE_PATH.exists():
        raise FileNotFoundError(f"{COSINE_PATH} not found")
    results = json.loads(RESULTS_PATH.read_text())
    js = json.loads(JS_PATH.read_text())
    cos = json.loads(COSINE_PATH.read_text())
    return results, js, cos


def main() -> None:
    set_paper_style("blog")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results, js_data, cos_data = load_inputs()

    persona_names = js_data["persona_names"]
    canonical_names = [p[0] for p in PERSONAS]
    if persona_names != canonical_names:
        raise AssertionError("persona name ordering mismatch between js_matrix.json and PERSONAS")

    idx_no = persona_names.index("no_persona")
    idx_19 = [i for i in range(20) if i != idx_no]
    names_19 = [persona_names[i] for i in idx_19]

    layer_key = str(HEADLINE_LAYER)
    headline = results["layers"][layer_key]

    # Build matrices.
    dist_cos_full = 1.0 - np.array(cos_data[f"layer_{HEADLINE_LAYER}"]["matrix"])
    dist_cos_19 = dist_cos_full[np.ix_(idx_19, idx_19)]
    js_full = np.array(js_data["matrices"]["Tfull"])
    js_19 = js_full[np.ix_(idx_19, idx_19)]
    iu_19 = np.triu_indices(19, k=1)
    v_cos = dist_cos_19[iu_19]
    v_js = js_19[iu_19]

    # (i) Hero
    figure_hero_dual_heatmap(dist_cos_19, js_19, names_19)
    # (ii) Scatter
    figure_scatter_with_residuals(
        v_cos=v_cos,
        v_js=v_js,
        iu_19=iu_19,
        names_19=names_19,
        top5_residual_pairs=headline["h_pair_residuals"]["top5_baseline_residual_pairs"],
        rho_raw=headline["rho_raw"],
        rho_resid_mm=headline["rho_resid_baseline_mean_marginal"],
    )
    # (iii) Sensitivity by layer
    figure_sensitivity_by_layer(results["layers"])
    # (iv) T-cutoff
    figure_t_cutoff(headline)
    # (v) Jackknife
    figure_jackknife(headline)
    # (vi) HA-excluded
    figure_ha_excluded(headline)


if __name__ == "__main__":
    main()
