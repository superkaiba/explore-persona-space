#!/usr/bin/env python3
"""Phase 1 statistics — issue #368 §4.1.4.

Inputs:
  eval_results/issue_368/phase1/regression_data_augmented.csv  (N=128, 22 cols)

Outputs (eval_results/issue_368/phase1/):
  per_axis_stats.json                  — Spearman ρ, p, both bootstrap flavors,
                                          ΔR² replace + ΔR² add, leave-one-trigger-out CV-R²
  regression_results.json              — OLS summaries
  recipe_agreement_matrix_with_projdiff.csv     (8×8 — R4 disclosed)
  recipe_agreement_matrix_no_projdiff.csv       (7×7 — R4 disclosed)
  h1_verdict.json                      — H1 PASS / FAIL + R6 centroid margin
  collinearity_diagnostics.json        — T10 / R10 framing pre-reg
  permutation_null.json                — H3a/H3b verdict + degenerate flag
  conditional_nonzero.json             — T14 / R12 calibrated vs semantic_cos 0.5644
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from explore_persona_space.axis.chenstyle import AXIS_SPECS, HEADLINE_AXIS  # noqa: E402
from explore_persona_space.eval.leakage_axes import (  # noqa: E402
    AXIS_SPECS_RECIPE_AGREEMENT,
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_PERMUTATION_N,
    benjamini_hochberg,
    build_run_metadata,
    cluster_bootstrap_delta_spearman_ci,
    cluster_bootstrap_spearman_ci,
    conditional_spearman_rho,
    dump_json,
    marker_shuffle_permutation_null,
    off_diagonal_stats,
    recipe_agreement_matrix,
    spearman_with_p,
)

# ── Constants ────────────────────────────────────────────────────────────────

PHASE1_DIR = REPO_ROOT / "eval_results" / "issue_368" / "phase1"
AUGMENTED_CSV = PHASE1_DIR / "regression_data_augmented.csv"

NEW_AXES: list[str] = [a["name"] for a in AXIS_SPECS]  # 9 new (incl pos-only descriptive)
BASELINE_AXIS = "semantic_cos"
BASE_FIVE = ["semantic_cos", "lexical_jac", "struct_match", "task_match", "js_div"]

# R6 thresholds + R12 baseline (pre-verified at plan time)
R6_DELTA_THRESHOLD = 0.03
R12_BASELINE_CONDITIONAL_RHO = 0.5644
H1_DELTA_RHO_MIN = 0.05
H1_DELTA_R2_MIN = 0.04
H1_BASELINE_5AXIS_R2 = 0.4402  # plan §"Phase 1 statistics" row 3
T10_HIGH_COLLINEARITY = 0.9

H3A_THRESHOLD = 0.7


# ── Single-axis statistics ──────────────────────────────────────────────────


def compute_per_axis_stats(df: pd.DataFrame) -> dict:
    """Single-axis Spearman ρ, p, both bootstrap CIs, ΔR² flavors, CV-R²."""
    out: dict[str, dict] = {}
    y = df["marker_rate"].astype(float).values
    test_id_clusters = df["test_id"].values
    train_family_clusters = df["train_family"].values

    for axis in [BASELINE_AXIS, *NEW_AXES]:
        x = df[axis].astype(float).values
        rho, p = spearman_with_p(x, y)
        boot_primary = cluster_bootstrap_spearman_ci(
            x, y, test_id_clusters, n_resamples=DEFAULT_BOOTSTRAP_N
        )
        boot_secondary = cluster_bootstrap_spearman_ci(
            x, y, train_family_clusters, n_resamples=DEFAULT_BOOTSTRAP_N, seed=43
        )
        out[axis] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "n": len(x),
            "bootstrap_cluster_test_id_95ci": [boot_primary["ci_low"], boot_primary["ci_high"]],
            "bootstrap_cluster_train_family_95ci": [
                boot_secondary["ci_low"],
                boot_secondary["ci_high"],
            ],
        }

    # ΔR² (replace and add) + univariate R²
    for axis in [BASELINE_AXIS, *NEW_AXES]:
        univariate = _ols_r2(df, [axis], y)
        replace_axes = [a if a != BASELINE_AXIS else axis for a in BASE_FIVE]
        replace_r2 = (
            _ols_r2(df, replace_axes, y) if axis != BASELINE_AXIS else _ols_r2(df, BASE_FIVE, y)
        )
        add_r2 = _ols_r2(df, [*BASE_FIVE, axis], y) if axis not in BASE_FIVE else replace_r2
        out[axis].update(
            {
                "univariate_R2": float(univariate),
                "replace_5axis_R2": float(replace_r2),
                "delta_R2_replace_vs_baseline": float(replace_r2 - H1_BASELINE_5AXIS_R2),
                "add_6axis_R2": float(add_r2),
                "delta_R2_add_vs_baseline": float(add_r2 - H1_BASELINE_5AXIS_R2),
                "leave_one_trigger_out_cv_r2": _loto_cv_r2(df, axis, y),
            }
        )

    return out


def _ols_r2(df: pd.DataFrame, predictors: list[str], y: np.ndarray) -> float:
    X = df[predictors].astype(float).values
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return float(model.rsquared)


def _loto_cv_r2(df: pd.DataFrame, axis: str, y: np.ndarray) -> dict:
    """Leave-one-trigger-out CV-R²: hold out one train_family at a time."""
    families = sorted(df["train_family"].unique())
    fold_r2: list[float] = []
    for fam in families:
        train_mask = df["train_family"] != fam
        test_mask = df["train_family"] == fam
        replace_axes = [a if a != BASELINE_AXIS else axis for a in BASE_FIVE]
        Xtr = sm.add_constant(df.loc[train_mask, replace_axes].astype(float).values)
        ytr = y[train_mask.values]
        model = sm.OLS(ytr, Xtr).fit()
        Xte = sm.add_constant(df.loc[test_mask, replace_axes].astype(float).values)
        yte = y[test_mask.values]
        yhat = model.predict(Xte)
        ss_res = float(((yte - yhat) ** 2).sum())
        ss_tot = float(((yte - yte.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        fold_r2.append(r2)
    return {"per_fold": fold_r2, "mean": float(np.nanmean(fold_r2))}


# ── H1 verdict (T5 + R6 paired-bootstrap margins) ───────────────────────────


def compute_h1_verdict(df: pd.DataFrame) -> dict:
    """Headline H1 verdict + R6 centroid margins."""
    y = df["marker_rate"].astype(float).values
    test_id_clusters = df["test_id"].values

    x_head = df[HEADLINE_AXIS].astype(float).values
    x_base = df[BASELINE_AXIS].astype(float).values

    # Δρ vs semantic_cos (test_id cluster primary)
    delta_vs_semantic = cluster_bootstrap_delta_spearman_ci(
        x_head, x_base, y, test_id_clusters, n_resamples=DEFAULT_BOOTSTRAP_N
    )

    # ΔR² vs 5-axis baseline (plan H1: total R² ≥ 0.4802)
    replace_axes = [a if a != BASELINE_AXIS else HEADLINE_AXIS for a in BASE_FIVE]
    r2_total = _ols_r2(df, replace_axes, y)
    delta_r2 = r2_total - H1_BASELINE_5AXIS_R2

    # R6 paired-bootstrap vs both centroid axes
    r6_results: dict[str, dict] = {}
    for centroid_axis in ["pcentroid_methodA_L20", "pcentroid_methodB_L20"]:
        x_centroid = df[centroid_axis].astype(float).values
        r6 = cluster_bootstrap_delta_spearman_ci(
            x_head, x_centroid, y, test_id_clusters, n_resamples=DEFAULT_BOOTSTRAP_N, seed=44
        )
        r6_results[centroid_axis] = {
            "point_delta": r6["point_delta"],
            "ci_low": r6["ci_low"],
            "ci_high": r6["ci_high"],
            "excludes_zero": r6["excludes_zero"],
            "meets_r6_threshold": bool(
                (r6["point_delta"] >= R6_DELTA_THRESHOLD) and r6["excludes_zero"]
            ),
            "ci_resample_unit": "test_id_cluster (32 clusters; Phase 1 R6 spec)",
        }

    # T10 / R10 framing pre-registration
    pearson_r_head_baseline = float(np.corrcoef(x_head, x_base)[0, 1])
    high_collinearity = bool(pearson_r_head_baseline > T10_HIGH_COLLINEARITY)

    # Final verdict
    cond_point_delta_rho = bool(delta_vs_semantic["point_delta"] >= H1_DELTA_RHO_MIN)
    cond_point_delta_r2 = bool(delta_r2 >= H1_DELTA_R2_MIN)
    cond_ci_excludes_zero = bool(delta_vs_semantic["excludes_zero"])
    cond_r6 = all(r["meets_r6_threshold"] for r in r6_results.values())

    if cond_point_delta_rho and cond_point_delta_r2 and cond_ci_excludes_zero and cond_r6:
        verdict = "PASS"
    elif cond_point_delta_rho and cond_point_delta_r2 and cond_ci_excludes_zero and not cond_r6:
        verdict = "CENTROID_REPLICATION_NOT_CONTRAST_CONFIRMATION"
    else:
        verdict = "FAIL"

    framing = "default"
    if verdict == "PASS" and high_collinearity:
        framing = "precision_gain_on_shared_information"  # R10 pre-registered

    return {
        "verdict": verdict,
        "framing_per_R10": framing,
        "headline_axis": HEADLINE_AXIS,
        "baseline_axis": BASELINE_AXIS,
        "delta_rho_vs_semantic": delta_vs_semantic,
        "delta_R2_vs_baseline_5axis": float(delta_r2),
        "r2_5axis_with_headline": float(r2_total),
        "r2_5axis_baseline": float(H1_BASELINE_5AXIS_R2),
        "thresholds": {
            "min_delta_rho": H1_DELTA_RHO_MIN,
            "min_delta_R2": H1_DELTA_R2_MIN,
            "min_R6_centroid_delta": R6_DELTA_THRESHOLD,
        },
        "conditions": {
            "point_delta_rho_ge_0.05": cond_point_delta_rho,
            "point_delta_R2_ge_0.04": cond_point_delta_r2,
            "delta_rho_CI_excludes_zero": cond_ci_excludes_zero,
            "R6_centroid_margin_met": cond_r6,
        },
        "R6_centroid_margins": r6_results,
        "T10_collinearity": {
            "pearson_r_head_vs_semantic": pearson_r_head_baseline,
            "high_collinearity_gt_0.9": high_collinearity,
        },
    }


# ── Collinearity diagnostics (T10) ──────────────────────────────────────────


def compute_collinearity(df: pd.DataFrame) -> dict:
    """T10: Pearson r, helpful-projection variance, pos-only ρ comparison."""

    x_head = df[HEADLINE_AXIS].astype(float).values
    x_base = df[BASELINE_AXIS].astype(float).values
    pearson_r = float(np.corrcoef(x_head, x_base)[0, 1])

    # ρ of pos-only descriptive axis
    pos_only_axis = "pcentroid_chenstyle_pos_only_L20"
    rho_head, _ = spearman_with_p(x_head, df["marker_rate"].astype(float).values)
    rho_pos_only, _ = spearman_with_p(
        df[pos_only_axis].astype(float).values,
        df["marker_rate"].astype(float).values,
    )

    # Constant-after-centering check: across the 32 panel prompts, what's the
    # variance of helpful-projection (chenstyle's neg-side projected on test_act)?
    # We approximate as the variance of `pcentroid_methodB_L20` across the
    # 32 unique test_id values (one per panel prompt); std/|mean| < 0.1 → flag.
    panel_methodB = df.groupby("test_id")["pcentroid_methodB_L20"].mean().values.astype(float)
    helpful_proj_std = float(panel_methodB.std())
    helpful_proj_mean_abs = float(abs(panel_methodB.mean()))
    cv = helpful_proj_std / helpful_proj_mean_abs if helpful_proj_mean_abs > 0 else float("nan")

    return {
        "pearson_r_chenstyle_vs_semantic_cos": pearson_r,
        "high_collinearity_flag_gt_0.9": bool(pearson_r > T10_HIGH_COLLINEARITY),
        "spearman_rho_pos_only_axis": float(rho_pos_only),
        "spearman_rho_headline_axis": float(rho_head),
        "delta_rho_pos_only_vs_headline": float(rho_pos_only - rho_head),
        "contrast_step_decorative_if_within_0.02": bool(abs(rho_pos_only - rho_head) < 0.02),
        "helpful_projection_panel_std_over_meanabs": cv,
        "helpful_projection_constant_after_centering_flag_lt_0.1": bool(
            cv < 0.1 if not np.isnan(cv) else False
        ),
    }


# ── T14 / R12 conditional-nonzero ρ ─────────────────────────────────────────


def compute_conditional_nonzero(df: pd.DataFrame) -> dict:
    y = df["marker_rate"].astype(float).values
    mask = y > 0
    out: dict[str, dict] = {}
    for axis in [BASELINE_AXIS, *NEW_AXES]:
        x = df[axis].astype(float).values
        rho, p, n_sub = conditional_spearman_rho(x, y, mask)
        out[axis] = {
            "conditional_rho": float(rho),
            "conditional_p": float(p),
            "n_nonzero": int(n_sub),
            "below_semantic_cos_baseline_0.5644": bool(abs(rho) < R12_BASELINE_CONDITIONAL_RHO),
        }
    return {
        "calibration_baseline": {
            "semantic_cos_conditional_rho": R12_BASELINE_CONDITIONAL_RHO,
            "note": (
                "R12: new axis's conditional rho < 0.5644 = binary-discrim-dominant "
                "relative to baseline."
            ),
        },
        "per_axis": out,
    }


# ── H3a / H3b — recipe agreement matrix (with AND without projdiff) ─────────


def compute_h3(df: pd.DataFrame) -> dict:
    score_vectors_full = {
        axis: df[axis].astype(float).values for axis in AXIS_SPECS_RECIPE_AGREEMENT
    }
    # 7-axis variant excludes projdiff (R4 algebraic identity disclosure)
    score_vectors_no_projdiff = {
        axis: v for axis, v in score_vectors_full.items() if axis != "pvec_chenstyle_L20_projdiff"
    }

    mat_full, axes_full = recipe_agreement_matrix(score_vectors_full)
    mat_no_pd, axes_no_pd = recipe_agreement_matrix(score_vectors_no_projdiff)

    # Persist heatmap CSVs
    _write_matrix_csv(mat_full, axes_full, PHASE1_DIR / "recipe_agreement_matrix_with_projdiff.csv")
    _write_matrix_csv(mat_no_pd, axes_no_pd, PHASE1_DIR / "recipe_agreement_matrix_no_projdiff.csv")

    stats_full = off_diagonal_stats(mat_full)
    stats_no_pd = off_diagonal_stats(mat_no_pd)

    # H3a verdict — passes if mean ≥ 0.7
    h3a_with = bool(stats_full["mean"] >= H3A_THRESHOLD)
    h3a_no = bool(stats_no_pd["mean"] >= H3A_THRESHOLD)

    # H3b — marker_shuffle null (degenerate)
    null_with = marker_shuffle_permutation_null(
        score_vectors_full,
        df["marker_rate"].astype(float).values,
        n_permutations=DEFAULT_PERMUTATION_N,
    )
    null_no_pd = marker_shuffle_permutation_null(
        score_vectors_no_projdiff,
        df["marker_rate"].astype(float).values,
        n_permutations=DEFAULT_PERMUTATION_N,
    )

    return {
        "with_projdiff": {
            "off_diagonal_mean": stats_full["mean"],
            "off_diagonal_min": stats_full["min"],
            "H3a_passes": h3a_with,
            "H3b_null": null_with,
        },
        "without_projdiff": {
            "off_diagonal_mean": stats_no_pd["mean"],
            "off_diagonal_min": stats_no_pd["min"],
            "H3a_passes": h3a_no,
            "H3b_null": null_no_pd,
        },
        "axes_full": axes_full,
        "axes_no_projdiff": axes_no_pd,
        "R4_degeneracy_note": (
            "pvec_chenstyle_L20_projdiff is identical to pvec_chenstyle_L20 "
            "within-source by construction; pairwise ρ between the two will be "
            "1.0. The 'without_projdiff' variant is the principled estimate of "
            "recipe agreement."
        ),
    }


def _write_matrix_csv(mat: np.ndarray, axes: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", *axes])
        for i, name in enumerate(axes):
            w.writerow([name] + [f"{x:.6f}" for x in mat[i]])


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(AUGMENTED_CSV))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    assert len(df) == 128, f"expected 128 rows, got {len(df)}"
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)

    print("[Phase 1] per-axis stats...")
    per_axis = compute_per_axis_stats(df)
    dump_json(
        {"per_axis": per_axis, "metadata": build_run_metadata({"phase": "phase1"})},
        PHASE1_DIR / "per_axis_stats.json",
    )

    print("[Phase 1] H1 verdict + R6 margins...")
    h1 = compute_h1_verdict(df)
    dump_json(
        {"h1_verdict": h1, "metadata": build_run_metadata({"phase": "phase1"})},
        PHASE1_DIR / "h1_verdict.json",
    )

    print("[Phase 1] collinearity diagnostics (T10/R10)...")
    coll = compute_collinearity(df)
    dump_json(coll, PHASE1_DIR / "collinearity_diagnostics.json")

    print("[Phase 1] T14 / R12 conditional-nonzero ρ...")
    cond = compute_conditional_nonzero(df)
    dump_json(cond, PHASE1_DIR / "conditional_nonzero.json")

    print("[Phase 1] H3a / H3b recipe agreement (with + without projdiff)...")
    h3 = compute_h3(df)
    dump_json(h3, PHASE1_DIR / "permutation_null.json")

    # R8: BH-FDR over the 9 single-axis Spearman p-values (all 9 new axes).
    p_values = {axis: per_axis[axis]["spearman_p"] for axis in NEW_AXES}
    bh = benjamini_hochberg(p_values, alpha=0.10)
    dump_json(
        {
            "scope_note": (
                "R8: BH-FDR (α=0.10) applied ONLY to the 9 single-axis Spearman "
                "ρ p-values. ΔR², partial ρ, conditional ρ, within-source partial "
                "ρ are descriptive without correction."
            ),
            "alpha": 0.10,
            "bh_results": bh,
        },
        PHASE1_DIR / "bh_fdr.json",
    )

    # Regression summary
    dump_json(
        {
            "baseline_5axis_R2": H1_BASELINE_5AXIS_R2,
            "best_replace_R2": max(per_axis[axis]["replace_5axis_R2"] for axis in NEW_AXES),
            "best_add_R2": max(per_axis[axis]["add_6axis_R2"] for axis in NEW_AXES),
            "h1_pass": h1["verdict"] == "PASS",
            "metadata": build_run_metadata({"phase": "phase1"}),
        },
        PHASE1_DIR / "regression_results.json",
    )

    print(f"[Phase 1] all outputs written to {PHASE1_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
