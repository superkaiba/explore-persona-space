"""Phase 4 — figure generation for issue #406.

Issue #406 plan v9 §4 Phase 4 plot block (9 figures total).

Reads eval_results/issue_406/analysis.json + the per-layer cosine
matrices + D_per_position.json and emits to figures/issue_406/:

  1. hero_scatter.{png,pdf}            — D vs G across 240 pairs (overall fit + per-class-pair fits;
                                          N=16 active conditions after 2026-05-31 C2-C5 scope drop)
  2. hero_scatter_raw.{png,pdf}        — raw D vs G (raw-alongside-processed pair)
  3. per_class_grid.{png,pdf}          — 4x4 small multiples per (class_i, class_j) cell
  4. threshold_curve.{png,pdf}         — sliding-quantile mean G vs D window center
  5. diagonal_sanity.{png,pdf}         — G[i, i] bar chart with 0.7 threshold line
  6. forest_plot.{png,pdf}             — per-predictor length-partial rho + CI (7 rows)
  7. per_predictor_scatter.{png,pdf}   — 1x7 row of scatters (KL + 6 cosine layers vs G)
  8. per_position_trajectory.{png,pdf} — v9-NEW: length-partial rho(D_k, G) vs k with CI band
  9. k_window_sweep.{png,pdf}          — v9-NEW: 9-window descriptive sweep horizontal bar

Every figure pairs with a meta.json file carrying git_commit + N +
source-data path per CLAUDE.md reproducibility metadata rule.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.i406_conditions import CONDITIONS

logger = logging.getLogger("i406.figures")

ANALYSIS_PATH = Path("eval_results/issue_406/analysis.json")
DIVERG_PATH = Path("eval_results/issue_406/divergence/D_matrix.json")
DIVERG_PER_POS_PATH = Path("eval_results/issue_406/divergence/D_per_position.json")
G_MATRIX_PATH = Path("eval_results/issue_406/cross_eval/G_matrix.json")
COSINE_DIR = Path("eval_results/issue_406/cosine")
FIG_DIR = Path("figures/issue_406")
TARGET_LAYERS = [0, 5, 11, 15, 21, 27]
CLASSES = ["A", "B", "C", "D"]
CLASS_NAMES = {
    "A": "Persona prompts",
    "B": "Query phrasings",
    "C": "Format scaffolds",
    "D": "Semantic rewrites",
}


def _load_all() -> tuple[dict, dict, dict, dict, dict[int, dict]]:
    analysis = json.loads(ANALYSIS_PATH.read_text())
    d_payload = json.loads(DIVERG_PATH.read_text())
    g_payload = json.loads(G_MATRIX_PATH.read_text())
    d_per_pos = json.loads(DIVERG_PER_POS_PATH.read_text())
    c_payloads = {L: json.loads((COSINE_DIR / f"C_L{L}.json").read_text()) for L in TARGET_LAYERS}
    return analysis, d_payload, g_payload, d_per_pos, c_payloads


def _build_long_form(d_payload: dict, g_payload: dict) -> list[dict]:
    """Flatten the (i, j) matrices into a list of rows for plotting."""
    rows = []
    for ci in g_payload["diagonal_passed"]:
        for cj in g_payload["diagonal_passed"]:
            if ci == cj:
                continue
            kl = d_payload["KL"][ci][cj]
            g = g_payload["G"][ci][cj]["rate"]
            n_tok = d_payload["prompt_tokens"][ci][cj]
            if kl is None or g is None:
                continue
            rows.append(
                {
                    "T_i": ci,
                    "T_j": cj,
                    "class_i": ci[0],
                    "class_j": cj[0],
                    "D": kl,
                    "G": g,
                    "log_prompt_tokens": float(np.log(n_tok)),
                }
            )
    return rows


def _save_meta(fig_path: Path, payload: dict) -> None:
    meta_path = fig_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(payload, indent=2))


def fig_hero_scatter(rows: list[dict], analysis: dict) -> None:
    """1. D vs G colored by class_i x class_j. Overall fit + per-cell fits."""
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = paper_palette_blog(len(CLASSES))
    class_color = dict(zip(CLASSES, palette, strict=True))
    for c in CLASSES:
        pts_x = [r["D"] for r in rows if r["class_i"] == c]
        pts_y = [r["G"] for r in rows if r["class_i"] == c]
        ax.scatter(pts_x, pts_y, color=class_color[c], alpha=0.6, s=24, label=CLASS_NAMES[c])
    # Overall fit line
    xs = np.array([r["D"] for r in rows])
    ys = np.array([r["G"] for r in rows])
    coef = np.polyfit(xs, ys, 1)
    grid = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(grid, coef[0] * grid + coef[1], color="black", lw=1.5, alpha=0.8, label="Overall fit")
    ax.set_xlabel("Forward KL between base-model output distributions (K=25-mean)")
    ax.set_ylabel("Transfer rate (marker emitted on T_j)")
    ax.set_title("Does base-model output divergence predict where SFT generalizes?")
    ax.legend(loc="best", title="Trained-on class", fontsize=8)
    fig_path = FIG_DIR / "hero_scatter.png"
    savefig_paper(fig, fig_path)
    plt.close(fig)
    _save_meta(
        fig_path,
        {
            "git_commit": analysis["git_commit"],
            "n_pairs": len(rows),
            "source_data": str(ANALYSIS_PATH),
            "description": (
                "Hero scatter: forward KL vs transfer rate, colored by trained-on class."
            ),
        },
    )
    logger.info("Wrote %s", fig_path)


def fig_hero_scatter_raw(rows: list[dict], analysis: dict) -> None:
    """2. Raw counterpart of hero_scatter (raw-alongside-processed per Lens 11)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    # Color by log_prompt_tokens to surface the length confound (the
    # processed version partials it out; this raw version shows what was
    # absorbed).
    sc = ax.scatter(
        [r["D"] for r in rows],
        [r["G"] for r in rows],
        c=[r["log_prompt_tokens"] for r in rows],
        cmap="viridis",
        alpha=0.7,
        s=24,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log(prompt tokens)")
    ax.set_xlabel("Forward KL (raw — no length-partial)")
    ax.set_ylabel("Transfer rate")
    ax.set_title("Raw counterpart of the hero (length confound visible)")
    fig_path = FIG_DIR / "hero_scatter_raw.png"
    savefig_paper(fig, fig_path)
    plt.close(fig)
    _save_meta(
        fig_path,
        {
            "git_commit": analysis["git_commit"],
            "n_pairs": len(rows),
            "source_data": str(ANALYSIS_PATH),
            "description": "Raw KL vs G colored by log-prompt-tokens (no length-partial).",
        },
    )
    logger.info("Wrote %s", fig_path)


def fig_per_class_grid(rows: list[dict], analysis: dict) -> None:
    """3. 4x4 grid of subscatters, one per (class_i, class_j) cell.

    Singleton-Class-C handling (2026-05-31): Class C is the C1 singleton
    after the C2-C5 drop, so the C->C diagonal cell has 0 off-diagonal
    pairs. That subplot is rendered with an explicit 'n/a (Class C is
    singleton)' annotation rather than an empty axis with a misleading
    'n=0' title that reads like a measured zero.
    """
    fig, axes = plt.subplots(4, 4, figsize=(11, 11), sharex=True, sharey=True)
    per_cell_meta = analysis["per_predictor"]["KL_primary"].get("per_cell_meta", {})
    per_cell = analysis["per_predictor"]["KL_primary"]["per_cell_partials"]
    for i, ci in enumerate(CLASSES):
        for j, cj in enumerate(CLASSES):
            ax = axes[i][j]
            cell_rows = [r for r in rows if r["class_i"] == ci and r["class_j"] == cj]
            cell = f"{ci}_{cj}"
            meta = per_cell_meta.get(cell, {})
            status = meta.get("status")
            if cell_rows:
                ax.scatter(
                    [r["D"] for r in cell_rows],
                    [r["G"] for r in cell_rows],
                    alpha=0.6,
                    s=18,
                )
                rho = per_cell.get(cell)
                title = f"{ci}->{cj} (n={len(cell_rows)}"
                if rho is not None:
                    title += f", rho={rho:.2f})"
                else:
                    title += ")"
            else:
                # No off-diagonal pairs (e.g. C->C with C-as-singleton).
                # Label the cell explicitly so the reader doesn't confuse
                # 'no data' with 'measured zero'.
                if status == "absent":
                    reason = "n/a (no off-diagonal pairs in this class cell)"
                else:
                    reason = f"n/a (status={status or 'no_rows'})"
                ax.text(
                    0.5,
                    0.5,
                    reason,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    style="italic",
                    color="#666666",
                )
                title = f"{ci}->{cj} (n=0; n/a)"
            ax.set_title(title, fontsize=9)
            if i == 3:
                ax.set_xlabel("KL")
            if j == 0:
                ax.set_ylabel("G")
    fig.suptitle(
        "Per-(trained-class, eval-class) cell: KL vs transfer rate "
        "(Class C is the C1 singleton; C->C cell has no off-diagonal pairs)",
        fontsize=11,
    )
    fig.tight_layout()
    fig_path = FIG_DIR / "per_class_grid.png"
    savefig_paper(fig, fig_path)
    plt.close(fig)
    _save_meta(
        fig_path,
        {
            "git_commit": analysis["git_commit"],
            "n_pairs": len(rows),
            "source_data": str(ANALYSIS_PATH),
            "description": (
                "4x4 cell-by-cell KL vs G with per-cell length-partial rho in titles. "
                "Class C is a singleton (C1 only) after the 2026-05-31 C2-C5 drop, so "
                "the C->C cell is explicitly labeled n/a (no off-diagonal pairs)."
            ),
        },
    )
    logger.info("Wrote %s", fig_path)


def fig_threshold_curve(analysis: dict) -> None:
    """4. Sliding-quantile threshold curve (with detected first-dip annotation)."""
    primary = analysis["per_predictor"]["KL_primary"]
    curve = primary["threshold_curve"]
    if curve is None:
        logger.warning("No threshold_curve in primary; skipping figure 4")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["window_centers"], curve["window_means_G"], lw=1.8, color="black")
    ax.axhline(
        curve["g_floor"], color="red", lw=0.8, ls="--", label=f"G floor = {curve['g_floor']}"
    )
    if curve["first_dip_below_g_floor"] is not None:
        ax.axvline(
            curve["first_dip_below_g_floor"],
            color="red",
            lw=0.8,
            ls=":",
            label=f"First dip below floor at KL = {curve['first_dip_below_g_floor']:.3f}",
        )
    ax.set_xlabel("Window center (KL)")
    ax.set_ylabel("Sliding-window mean transfer rate")
    ax.set_title("Where does transfer crash? (window=50, step=10)")
    ax.legend(loc="best", fontsize=8)
    fig_path = FIG_DIR / "threshold_curve.png"
    savefig_paper(fig, fig_path)
    plt.close(fig)
    _save_meta(
        fig_path,
        {
            "git_commit": analysis["git_commit"],
            "source_data": str(ANALYSIS_PATH),
            "description": "Sliding-quantile threshold curve for primary KL predictor.",
        },
    )
    logger.info("Wrote %s", fig_path)


def fig_diagonal_sanity(g_payload: dict, analysis: dict) -> None:
    """5. G[i, i] bar chart with the 0.7 pass threshold."""
    fig, ax = plt.subplots(figsize=(9, 4))
    cids = [c.cid for c in CONDITIONS]
    diag_rates = [g_payload["G"][ci][ci]["rate"] for ci in cids]
    bar_colors = [
        "#4daf4a" if r >= g_payload["diagonal_threshold"] else "#e41a1c" for r in diag_rates
    ]
    ax.bar(range(len(cids)), diag_rates, color=bar_colors)
    ax.axhline(g_payload["diagonal_threshold"], color="black", lw=0.8, ls="--")
    ax.set_xticks(range(len(cids)))
    ax.set_xticklabels(cids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Diagonal transfer rate G[i, i]")
    n_pass = len(g_payload["diagonal_passed"])
    ax.set_title(f"Diagonal sanity: {n_pass}/{len(cids)} conditions implant marker")
    fig.tight_layout()
    fig_path = FIG_DIR / "diagonal_sanity.png"
    savefig_paper(fig, fig_path)
    plt.close(fig)
    _save_meta(
        fig_path,
        {
            "git_commit": analysis["git_commit"],
            "n_conditions": len(cids),
            "source_data": str(G_MATRIX_PATH),
            "description": "Per-condition diagonal G[i, i] vs 0.7 sanity threshold.",
        },
    )
    logger.info("Wrote %s", fig_path)


def fig_forest(analysis: dict) -> None:
    """6. 7-row forest plot: length-partial rho with cluster-bootstrap CI per predictor."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    rhos = []
    ci_lows = []
    ci_highs = []
    per_cell_max = []
    is_primary_flags = []
    for row in analysis["summary_table"]:
        labels.append(row["predictor"])
        rhos.append(row["length_partial_rho_pg"])
        ci_lows.append(row["cluster_bootstrap_ci"][0])
        ci_highs.append(row["cluster_bootstrap_ci"][1])
        per_cell_max.append(row["per_cell_max_abs_rho"])
        is_primary_flags.append(row["is_primary"])
    y = np.arange(len(labels))
    for i, prim in enumerate(is_primary_flags):
        color = "black" if prim else "#377eb8"
        ax.plot([ci_lows[i], ci_highs[i]], [y[i], y[i]], color=color, lw=2.5)
        ax.scatter([rhos[i]], [y[i]], color=color, s=40, zorder=3)
        ax.scatter([per_cell_max[i]], [y[i]], marker="x", color="#e41a1c", s=24, zorder=3)
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Length-partial Spearman rho (overall = dot; per-cell max = red x)")
    ax.set_title("7 predictors vs transfer rate (cluster-bootstrap 95% CI)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig_path = FIG_DIR / "forest_plot.png"
    savefig_paper(fig, fig_path)
    plt.close(fig)
    _save_meta(
        fig_path,
        {
            "git_commit": analysis["git_commit"],
            "n_predictors": len(labels),
            "source_data": str(ANALYSIS_PATH),
            "description": "7-row forest plot of length-partial rho + CI per predictor.",
        },
    )
    logger.info("Wrote %s", fig_path)


def fig_per_predictor_scatter(
    rows: list[dict], analysis: dict, c_payloads: dict[int, dict]
) -> None:
    """7. 1x7 row of small scatters, one per predictor (KL + 6 cosine layers vs G)."""
    fig, axes = plt.subplots(1, 7, figsize=(20, 3.6), sharey=True)
    palette = paper_palette(7)
    cids_passed = set(r["T_i"] for r in rows) | set(r["T_j"] for r in rows)

    # KL panel uses the `rows` list directly.
    axes[0].scatter(
        [r["D"] for r in rows], [r["G"] for r in rows], alpha=0.5, s=14, color=palette[0]
    )
    axes[0].set_title("KL (primary)", fontsize=10)
    axes[0].set_xlabel("KL")
    axes[0].set_ylabel("G")

    # Cosine panels need to pull from c_payloads.
    for idx, L in enumerate(TARGET_LAYERS):
        ax = axes[idx + 1]
        xs = []
        ys = []
        for r in rows:
            if r["T_i"] in cids_passed and r["T_j"] in cids_passed:
                xs.append(c_payloads[L]["matrix"][r["T_i"]][r["T_j"]])
                ys.append(r["G"])
        ax.scatter(xs, ys, alpha=0.5, s=14, color=palette[idx + 1])
        ax.set_title(f"Cosine L{L}", fontsize=10)
        ax.set_xlabel(f"1 - cos (L{L})")
    fig.suptitle("7 predictors vs transfer rate (raw, before length-partial)", fontsize=11)
    fig.tight_layout()
    fig_path = FIG_DIR / "per_predictor_scatter.png"
    savefig_paper(fig, fig_path)
    plt.close(fig)
    _save_meta(
        fig_path,
        {
            "git_commit": analysis["git_commit"],
            "n_panels": 7,
            "source_data": str(ANALYSIS_PATH),
            "description": "1x7 row of per-predictor scatters vs G.",
        },
    )
    logger.info("Wrote %s", fig_path)


def fig_per_position_trajectory(analysis: dict) -> None:
    """8. v9-NEW: per-position length-partial rho(D_k, G) vs k with bootstrap band."""
    traj = analysis["per_position_trajectory"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ks = [t["k"] for t in traj if t["length_partial_rho"] is not None]
    rhos = [t["length_partial_rho"] for t in traj if t["length_partial_rho"] is not None]
    cis = [t["cluster_bootstrap_ci_500"] for t in traj if t["cluster_bootstrap_ci_500"]]
    ci_lows = [c[0] for c in cis]
    ci_highs = [c[1] for c in cis]
    ax.fill_between(ks, ci_lows, ci_highs, color="#377eb8", alpha=0.25, label="500-bootstrap CI")
    ax.plot(ks, rhos, color="#377eb8", lw=1.8, marker="o", markersize=4)
    ax.axhline(0.4, color="green", lw=0.6, ls="--", alpha=0.5, label="Positive threshold (0.4)")
    ax.axhline(0.15, color="orange", lw=0.6, ls="--", alpha=0.5, label="Negative threshold (0.15)")
    ax.axhline(-0.15, color="orange", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(0, color="grey", lw=0.4)
    ax.set_xlabel("Position k in reference completion (0 = first response token)")
    ax.set_ylabel("Length-partial Spearman rho (D_k, G)")
    ax.set_title("Per-position predictor trajectory (descriptive supplementary)")
    ax.legend(loc="best", fontsize=8)
    fig_path = FIG_DIR / "per_position_trajectory.png"
    savefig_paper(fig, fig_path)
    plt.close(fig)
    _save_meta(
        fig_path,
        {
            "git_commit": analysis["git_commit"],
            "K_target": len(traj),
            "source_data": str(ANALYSIS_PATH),
            "description": (
                "Per-position trajectory of length-partial rho(D_k, G) for k = 0..24. "
                "Reference horizontal lines at +0.4 / +/- 0.15 are the K=25-mean primary "
                "thresholds (NOT per-position binding)."
            ),
        },
    )
    logger.info("Wrote %s", fig_path)


def fig_k_window_sweep(analysis: dict) -> None:
    """9. v9-NEW: 9-window descriptive sweep — horizontal bar of length-partial rho per window."""
    sweep = analysis["k_window_sweep"]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [w["window"] for w in sweep]
    rhos = [w["length_partial_rho"] if w["length_partial_rho"] is not None else 0.0 for w in sweep]
    cis = [w["cluster_bootstrap_ci_500"] for w in sweep]
    primary_idx = [i for i, w in enumerate(sweep) if w["is_primary"]]
    best_idx = [
        i
        for i, w in enumerate(sweep)
        if w["length_partial_rho"] is not None
        and abs(w["length_partial_rho"]) == max(abs(r) for r in rhos if r is not None)
    ][:1] or []
    y = np.arange(len(labels))
    for i in range(len(labels)):
        color = "black" if i in primary_idx else "#377eb8"
        ax.barh(y[i], rhos[i], color=color, alpha=0.85, height=0.6)
        if cis[i]:
            ax.plot([cis[i][0], cis[i][1]], [y[i], y[i]], color="red", lw=1.5, alpha=0.6)
        if i in best_idx:
            ax.text(rhos[i], y[i], " *best", fontsize=10, va="center")
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Length-partial Spearman rho (D_W, G); red bar = 95% bootstrap CI")
    ax.set_title("K-window descriptive sweep (best of 9; garden-of-forking-paths)")
    fig.tight_layout()
    fig_path = FIG_DIR / "k_window_sweep.png"
    savefig_paper(fig, fig_path)
    plt.close(fig)
    _save_meta(
        fig_path,
        {
            "git_commit": analysis["git_commit"],
            "n_windows": len(labels),
            "source_data": str(ANALYSIS_PATH),
            "description": (
                "Best-K-window descriptive sweep across 9 candidate windows. "
                "Primary k=0..24 (W=25) is bolded; best |rho| is starred. Subject to "
                "garden-of-forking-paths caveat (~9% FWER at nominal per-window p<0.01)."
            ),
        },
    )
    logger.info("Wrote %s", fig_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    set_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    analysis, d_payload, g_payload, _d_per_pos, c_payloads = _load_all()
    rows = _build_long_form(d_payload, g_payload)
    if len(rows) < 10:
        raise RuntimeError(
            f"Only {len(rows)} long-form rows; refusing to plot. "
            "Did Phase 3 finish? Is the diagonal passing for >=2 conditions?"
        )
    logger.info("Generating 9 figures over %d (i, j) pairs.", len(rows))

    fig_hero_scatter(rows, analysis)
    fig_hero_scatter_raw(rows, analysis)
    fig_per_class_grid(rows, analysis)
    fig_threshold_curve(analysis)
    fig_diagonal_sanity(g_payload, analysis)
    fig_forest(analysis)
    fig_per_predictor_scatter(rows, analysis, c_payloads)
    fig_per_position_trajectory(analysis)
    fig_k_window_sweep(analysis)

    logger.info("All figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
