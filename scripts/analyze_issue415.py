#!/usr/bin/env python3
"""Analyze task #415 - 4 predictors (cos+JS x {assistant, neutral}) vs 6 DV surfaces.

Reuses #396's per-source DV values (`logp_*_diagonal_mean` and `substring_match_rate_diagonal_mean`)
from `eval_results/issue_396/analysis_summary.json` and runs the same
length-partial Spearman + BH-FDR(q=0.05) scaffolding on the 4 predictors
from `eval_results/issue_415/base_model_predictors_v2.json`.

Headline DV: `logp_end_of_response_diagonal_mean`. Threshold: |rho| >= 0.35.

Outputs:
  - eval_results/issue_415/analysis_summary.json
  - figures/issue_415/hero_4predictor_symmetric_panel.{png,pdf}
  - figures/issue_415/predictor_x_surface_forest.{png,pdf}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

EVAL_RESULTS_415 = PROJECT_ROOT / "eval_results" / "issue_415"
EVAL_RESULTS_415.mkdir(parents=True, exist_ok=True)
FIGURES_415 = PROJECT_ROOT / "figures" / "issue_415"
FIGURES_415.mkdir(parents=True, exist_ok=True)

PREDICTORS = [
    (
        "predictor_1_cosine_to_assistant_L15",
        "Cosine-to-assistant (L15)",
        "Cosine similarity to assistant-prompt baseline at layer 15",
    ),
    (
        "predictor_2_js_to_assistant",
        "JS-to-assistant",
        "JS divergence to assistant-prompt baseline (next-token distribution)",
    ),
    (
        "predictor_4_cosine_to_neutral_L15",
        "Cosine-to-neutral (L15)",
        "Cosine similarity to neutral baseline (no system prompt) at layer 15",
    ),
    (
        "predictor_5_js_to_neutral",
        "JS-to-neutral",
        "JS divergence to neutral baseline (no system prompt)",
    ),
]

SURFACES = [
    ("logp_end_of_response_diagonal_mean", "log p end-of-response (HEADLINE)"),
    ("logp_at_k0_diagonal_mean", "log p at k=0"),
    ("logp_auc_diagonal_mean", "log p AUC"),
    ("logp_max_diagonal_mean", "log p max"),
    ("logp_mean_diagonal_mean", "log p mean"),
    ("substring_match_rate_diagonal_mean", "substring match rate"),
]

HEADLINE_SURFACE = "logp_end_of_response_diagonal_mean"
THRESHOLD = 0.35


def bca_bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 42):
    """Quick percentile bootstrap CI for Spearman rho (skip BCa accel for speed)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            rhos[b] = np.nan
            continue
        try:
            rhos[b] = spearmanr(x[idx], y[idx]).statistic
        except Exception:
            rhos[b] = np.nan
    rhos = rhos[~np.isnan(rhos)]
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def length_partial_spearman(x: np.ndarray, y: np.ndarray, lengths: np.ndarray):
    """Spearman partial correlation: residualize x and y against lengths, then spearman."""
    rx = spearmanr(x, lengths).statistic
    ry = spearmanr(y, lengths).statistic
    rxy = spearmanr(x, y).statistic
    denom = np.sqrt((1 - rx**2) * (1 - ry**2))
    if denom < 1e-9:
        return float("nan")
    return float((rxy - rx * ry) / denom)


def benjamini_hochberg(pvals: list[float], q: float = 0.05) -> list[bool]:
    """BH-FDR on p-values, returns reject mask in original order."""
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = np.array([pvals[i] for i in order])
    thresholds = (np.arange(1, n + 1) / n) * q
    passes = sorted_p <= thresholds
    if not passes.any():
        return [False] * n
    k = np.max(np.where(passes)[0])
    reject = np.zeros(n, dtype=bool)
    reject[order[: k + 1]] = True
    return reject.tolist()


def main():
    # Load #396 per-source DV values
    summary_396 = json.loads(
        (PROJECT_ROOT / "eval_results/issue_396/analysis_summary.json").read_text()
    )
    per_src_396 = {row["source"]: row for row in summary_396["per_source_aggregation"]}

    # Load #415 predictors
    preds_415 = json.loads((EVAL_RESULTS_415 / "base_model_predictors_v2.json").read_text())

    sources = sorted(
        s for s in per_src_396 if s in preds_415["predictor_1_cosine_to_assistant_L15"]
    )
    n = len(sources)
    print(f"Aligned {n} sources between #396 DV and #415 predictors")

    # Per-source persona-prompt length for length partial.
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from analyze_length_rate_n48 import get_inherited_prompt

    lengths = np.array([len(get_inherited_prompt(s)) for s in sources])

    table: dict[str, dict[str, dict]] = {}
    all_cells: list[tuple[str, str, float, float]] = []  # (pred_key, surf_key, rho_partial, pval)

    for pred_key, _pred_label, _ in PREDICTORS:
        table[pred_key] = {}
        pred_values = np.array([preds_415[pred_key][s] for s in sources], dtype=float)
        for surf_key, _surf_label in SURFACES:
            dv_values = np.array([per_src_396[s][surf_key] for s in sources], dtype=float)
            mask = np.isfinite(pred_values) & np.isfinite(dv_values)
            if mask.sum() < 5:
                table[pred_key][surf_key] = {"error": "too few samples"}
                continue
            x, y, ll = pred_values[mask], dv_values[mask], lengths[mask]
            rho_raw = spearmanr(x, y).statistic
            pval_raw = spearmanr(x, y).pvalue
            rho_partial = length_partial_spearman(x, y, ll)
            lo, hi = bca_bootstrap_ci(x, y)
            table[pred_key][surf_key] = {
                "length_partial_spearman_rho": rho_partial,
                "spearman_rho_raw": float(rho_raw),
                "spearman_pvalue_raw": float(pval_raw),
                "bca_ci_95_low": lo,
                "bca_ci_95_high": hi,
                "n": int(mask.sum()),
            }
            all_cells.append((pred_key, surf_key, rho_partial, float(pval_raw)))

    # BH-FDR across headline cells only (4 predictors x 1 headline surface)
    headline_cells = [(pk, sk, r, p) for (pk, sk, r, p) in all_cells if sk == HEADLINE_SURFACE]
    headline_pvals = [p for (_, _, _, p) in headline_cells]
    bh_reject = benjamini_hochberg(headline_pvals, q=0.05)
    headline_bh = {pk: bh_reject[i] for i, (pk, _, _, _) in enumerate(headline_cells)}

    summary = {
        "schema_version": 1,
        "parent_task": 396,
        "n_personas": n,
        "predictors_present": {
            pk: len([s for s in sources if s in preds_415[pk]]) for pk, _, _ in PREDICTORS
        },
        "headline_dv_surface": HEADLINE_SURFACE,
        "detection_threshold": THRESHOLD,
        "predictor_table": {
            "table": table,
            "headline_pvalues": {pk: float(p) for pk, _, _, p in headline_cells},
            "bh_fdr_reject": headline_bh,
        },
        "summary": {
            f"{pred_label} × headline": {
                "rho_partial": table[pred_key][HEADLINE_SURFACE]["length_partial_spearman_rho"],
                "rho_raw": table[pred_key][HEADLINE_SURFACE]["spearman_rho_raw"],
                "p": table[pred_key][HEADLINE_SURFACE]["spearman_pvalue_raw"],
                "n": table[pred_key][HEADLINE_SURFACE]["n"],
                "bh_fdr_reject": headline_bh[pred_key],
                "clears_threshold": abs(
                    table[pred_key][HEADLINE_SURFACE]["length_partial_spearman_rho"]
                )
                >= THRESHOLD,
            }
            for (pred_key, pred_label, _) in PREDICTORS
        },
        "verdict": (
            "null_corroborated"
            if not any(
                abs(table[pk][HEADLINE_SURFACE]["length_partial_spearman_rho"]) >= THRESHOLD
                for pk, _, _ in PREDICTORS
            )
            else "neutral_baseline_reveals_signal"
        ),
    }

    out_summary = EVAL_RESULTS_415 / "analysis_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_summary}")

    # ── 4-panel hero scatter ──
    set_paper_style("blog")
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=False)
    axes = axes.flatten()
    primary = paper_palette_role("primary")
    dv_y = np.array([per_src_396[s][HEADLINE_SURFACE] for s in sources])
    for ax, (pred_key, pred_label, xlabel) in zip(axes, PREDICTORS):
        xv = np.array([preds_415[pred_key][s] for s in sources])
        mask = np.isfinite(xv) & np.isfinite(dv_y)
        ax.scatter(
            xv[mask], dv_y[mask], s=42, alpha=0.85, color=primary, edgecolor="white", linewidth=0.6
        )
        c = table[pred_key][HEADLINE_SURFACE]
        rho = c["length_partial_spearman_rho"]
        pval = c["spearman_pvalue_raw"]
        n_cell = c["n"]
        clears = "" if abs(rho) < THRESHOLD else " — CLEARS THRESHOLD"
        title = f"{pred_label}: $\\rho$ = {rho:+.3f}, p = {pval:.2f}, N = {n_cell}{clears}"
        ax.set_title(title, fontsize=11, loc="left", pad=6, fontweight="semibold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Trained-LoRA log p(marker)\nat end-of-response (diagonal mean)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.18)
        ax.tick_params(labelsize=8)

    fig.subplots_adjust(top=0.86, bottom=0.09, left=0.08, right=0.97, hspace=0.55, wspace=0.30)
    fig.text(
        0.02,
        0.95,
        "Symmetric-baseline rerun: all four predictors against headline DV",
        fontsize=13.5,
        fontweight="semibold",
        ha="left",
        va="top",
    )
    fig.text(
        0.02,
        0.91,
        f"Top: against bare-assistant baseline (replicates #396). Bottom: against neutral baseline (no system prompt). N = {n} trained-source personas",
        fontsize=9.5,
        color="#555",
        ha="left",
        va="top",
    )
    savefig_paper(
        fig, "issue_415/hero_4predictor_symmetric_panel", dir=str(PROJECT_ROOT / "figures")
    )
    plt.close(fig)
    print("Wrote figures/issue_415/hero_4predictor_symmetric_panel.{png,pdf}")

    # ── 4×6 forest plot ──
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    row_labels = []
    rhos = []
    cis = []
    for pred_key, pred_label, _ in PREDICTORS:
        for surf_key, surf_label in SURFACES:
            cell = table[pred_key][surf_key]
            if "error" in cell:
                continue
            row_labels.append(f"{pred_label} × {surf_label}")
            rhos.append(cell["length_partial_spearman_rho"])
            cis.append((cell["bca_ci_95_low"], cell["bca_ci_95_high"]))
    y_pos = np.arange(len(row_labels))
    rhos_arr = np.array(rhos)
    ci_lo = np.array([c[0] for c in cis])
    ci_hi = np.array([c[1] for c in cis])
    err_lo = rhos_arr - ci_lo
    err_hi = ci_hi - rhos_arr
    ax.errorbar(
        rhos_arr,
        y_pos,
        xerr=[err_lo, err_hi],
        fmt="o",
        markersize=5,
        color=primary,
        capsize=3,
        capthick=1,
        elinewidth=1,
        alpha=0.85,
    )
    ax.axvspan(
        -THRESHOLD,
        THRESHOLD,
        color="#cccccc",
        alpha=0.18,
        label=f"sub-threshold |$\\rho$| \\< {THRESHOLD}",
    )
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Length-partial Spearman $\\rho$ (95% bootstrap CI)", fontsize=10)
    ax.set_title(
        f"All 4×6 predictor × surface cells (N = {n} per cell)",
        fontsize=12,
        loc="left",
        pad=6,
        fontweight="semibold",
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, axis="x", alpha=0.18)
    fig.tight_layout()
    savefig_paper(fig, "issue_415/predictor_x_surface_forest", dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)
    print("Wrote figures/issue_415/predictor_x_surface_forest.{png,pdf}")

    # ── Verdict ──
    print(f"\n=== VERDICT: {summary['verdict']} ===")
    for label, info in summary["summary"].items():
        flag = "CLEARS" if info["clears_threshold"] else "null"
        print(
            f"  {label}: rho_partial = {info['rho_partial']:+.3f}, "
            f"p = {info['p']:.3f}, BH-FDR = {info['bh_fdr_reject']} | {flag}"
        )


if __name__ == "__main__":
    main()
