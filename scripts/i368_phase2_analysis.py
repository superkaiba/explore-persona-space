#!/usr/bin/env python3
"""Phase 2 statistics — issue #368 §4.2.3 + §4.2.4.

Inputs:
  eval_results/issue_368/phase2/leakage_table.csv  (50 directed pairs × 13 cols)
  eval_results/issue_368/phase2/reproduction_sanity.json  (R2 gate output)
  eval_results/issue_368/phase2/persona_pos_set_cohesion.json (R11 diagnostic)

Outputs (eval_results/issue_368/phase2/):
  per_axis_stats.json
  recipe_agreement_matrix_with_projdiff.csv     (8×8 — R4)
  recipe_agreement_matrix_no_projdiff.csv       (7×7 — R4)
  h2_verdict.json                  — 5-valued verdict (T13 FAIL_permutation_calibration)
  source_partial_rho.json          — R3 + R9 within-source nanmean + bootstrap CI
  source_shuffle_permutation.json  — T13 source-label permutation null
  permutation_null.json            — H3a/H3b
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # enable `scripts.*` imports for C6 R11 patch
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
    dump_json,
    marker_shuffle_permutation_null,
    off_diagonal_stats,
    partial_spearman_rho,
    recipe_agreement_matrix,
    source_shuffle_permutation_null,
    spearman_with_p,
    within_source_partial_rho_bootstrap_ci,
)

PHASE2_DIR = REPO_ROOT / "eval_results" / "issue_368" / "phase2"
LEAKAGE_CSV = PHASE2_DIR / "leakage_table.csv"

NEW_AXES: list[str] = [a["name"] for a in AXIS_SPECS]

H2_MARGINAL_THRESHOLD = 0.75
H2_WITHIN_SOURCE_THRESHOLD = 0.30
R6_DELTA_THRESHOLD = 0.03
H3A_THRESHOLD = 0.7

# Phase-2 JS baseline (#142 published)
JS_BASELINE_RHO = 0.746
# Method-A centered-cosine-L20 baseline (#142 published)
METHOD_A_BASELINE_RHO = 0.567


# ── Single-axis statistics ──────────────────────────────────────────────────


def compute_per_axis_stats(df: pd.DataFrame) -> dict:
    y = df["marker_leakage_rate"].astype(float).values
    source_clusters = df["source"].values

    out: dict[str, dict] = {}
    for axis in NEW_AXES:
        x = df[axis].astype(float).values
        rho, p = spearman_with_p(x, y)
        # Phase-2 cluster bootstrap: resample sources (5 clusters; T6)
        boot = cluster_bootstrap_spearman_ci(x, y, source_clusters, n_resamples=DEFAULT_BOOTSTRAP_N)
        out[axis] = {
            "spearman_rho": float(rho),
            "abs_spearman_rho": float(abs(rho)),
            "spearman_p": float(p),
            "n": len(x),
            "bootstrap_cluster_source_95ci": [boot["ci_low"], boot["ci_high"]],
        }
        # Partial ρ given JS (R1 mitigation)
        if "js_div" in df.columns:
            js = df["js_div"].astype(float).values
            partial = partial_spearman_rho(x, y, js)
            out[axis]["partial_rho_given_js"] = float(partial)
    return out


# ── H2 verdict (T9 + R3 + R9 + T13 + R6) ────────────────────────────────────


def compute_h2_verdict(df: pd.DataFrame) -> dict:
    # CONCERN-1 (round-2 code-review): the C2 sentinel-vector strategy means
    # source=assistant rows carry placeholder pvec values, NOT real Chen-style
    # vectors. Plan §6.2 says "assistant only appears as TARGET in H2 contrast",
    # so we filter source=assistant rows out of every H2 statistic. Original 50
    # rows minus 10 source=assistant rows = N=40 for H2 verdict math.
    n_total = len(df)
    df = df[df["source"] != "assistant"].reset_index(drop=True)
    n_used = len(df)
    n_dropped_assistant = n_total - n_used

    y = df["marker_leakage_rate"].astype(float).values
    sources = df["source"].values
    x_head = df[HEADLINE_AXIS].astype(float).values

    # (a) Marginal |ρ|
    rho_marg, p_marg = spearman_with_p(x_head, y)
    cond_marginal = bool(abs(rho_marg) >= H2_MARGINAL_THRESHOLD)

    # (b) Within-source nanmean partial ρ + R9 bootstrap CI
    within = within_source_partial_rho_bootstrap_ci(
        x_head, y, list(sources), n_resamples=DEFAULT_BOOTSTRAP_N
    )
    cond_within_point = bool(within["nanmean_partial_rho"] >= H2_WITHIN_SOURCE_THRESHOLD)
    cond_within_ci = bool(within["ci_excludes_zero"])

    # (c) T13 source-shuffle permutation null
    shuffle_null = source_shuffle_permutation_null(
        x_head, y, list(sources), n_permutations=DEFAULT_PERMUTATION_N
    )
    cond_shuffle = bool(shuffle_null["exceeds_null"])

    # R6: paired-bootstrap Δρ vs both centroid baselines (cluster by source)
    r6: dict[str, dict] = {}
    for centroid_axis in ["pcentroid_methodA_L20", "pcentroid_methodB_L20"]:
        x_cen = df[centroid_axis].astype(float).values
        # Use abs ρ comparison (Phase 2 ρ can be positive or negative depending
        # on axis sign; we work with raw signed Δρ on the same orientation).
        r6_boot = cluster_bootstrap_delta_spearman_ci(
            x_head, x_cen, y, sources, n_resamples=DEFAULT_BOOTSTRAP_N, seed=44
        )
        r6[centroid_axis] = {
            "point_delta": r6_boot["point_delta"],
            "ci_low": r6_boot["ci_low"],
            "ci_high": r6_boot["ci_high"],
            "excludes_zero": r6_boot["excludes_zero"],
            "meets_r6_threshold": bool(
                (r6_boot["point_delta"] >= R6_DELTA_THRESHOLD) and r6_boot["excludes_zero"]
            ),
            "ci_resample_unit": "source_cluster (5 clusters; Phase 2 R6 spec)",
        }
    cond_r6 = all(v["meets_r6_threshold"] for v in r6.values())

    # Calibration baselines for the T9 statistic (R3 spec).
    calibration: dict[str, dict] = {}
    for cal_axis in ["pcentroid_methodA_L20", "pcentroid_methodB_L20"]:
        cal_x = df[cal_axis].astype(float).values
        cal_within = within_source_partial_rho_bootstrap_ci(
            cal_x, y, list(sources), n_resamples=DEFAULT_BOOTSTRAP_N, seed=45
        )
        calibration[cal_axis] = {
            "nanmean_partial_rho": cal_within["nanmean_partial_rho"],
            "bootstrap_ci_95": cal_within["bootstrap_ci_95"],
        }
    if "js_div" in df.columns:
        js_x = df["js_div"].astype(float).values
        js_within = within_source_partial_rho_bootstrap_ci(
            js_x, y, list(sources), n_resamples=DEFAULT_BOOTSTRAP_N, seed=46
        )
        calibration["js_divergence"] = {
            "nanmean_partial_rho": js_within["nanmean_partial_rho"],
            "bootstrap_ci_95": js_within["bootstrap_ci_95"],
        }

    # ── 6-valued verdict per T13 + plan §"Verdict thresholds (H2)" ──
    if not cond_marginal:
        verdict = "FAIL_marginal_below_threshold"
    elif not cond_shuffle:
        # Marginal pass, T13 null fail → permutation calibration failure
        verdict = "FAIL_permutation_calibration"
    elif cond_within_point and cond_within_ci and cond_r6:
        verdict = "PASS"
    elif cond_within_point and cond_within_ci and not cond_r6:
        # C3 + plan §6.2: T9 within-source passes but Δρ < 0.03 vs centroids
        # — Chen-style contrast NOT confirmed beyond centroid replication.
        verdict = "CENTROID_REPLICATION_NOT_CONTRAST_CONFIRMATION"
    elif cond_within_point and not cond_within_ci:
        verdict = "AMBIGUOUS_within_source_dimension"
    elif not cond_within_point and not cond_within_ci:
        verdict = "FAIL_source_discrimination_artifact"
    else:
        verdict = "AMBIGUOUS_within_source_dimension"

    return {
        "verdict": verdict,
        "headline_axis": HEADLINE_AXIS,
        "thresholds": {
            "min_marginal_abs_rho": H2_MARGINAL_THRESHOLD,
            "min_within_source_partial_rho": H2_WITHIN_SOURCE_THRESHOLD,
            "min_R6_centroid_delta": R6_DELTA_THRESHOLD,
        },
        "row_counts": {
            "n_total_input": n_total,
            "n_used_for_h2": n_used,
            "n_dropped_source_assistant": n_dropped_assistant,
            "note_concern_1": (
                "CONCERN-1 (round-2 code-review): source=assistant rows excluded; "
                "they carry sentinel pvec values per C2's labelled-sentinel design. "
                "Plan §6.2: assistant only appears as TARGET in H2 contrast."
            ),
        },
        "marginal": {
            "rho": float(rho_marg),
            "abs_rho": float(abs(rho_marg)),
            "p": float(p_marg),
            "matches_or_beats_js_0.746": bool(abs(rho_marg) >= JS_BASELINE_RHO - 0.05),
        },
        "within_source_T9_R9": within,
        "T13_source_shuffle_null": shuffle_null,
        "R6_centroid_margins": r6,
        "calibration_baselines_T9": calibration,
        "conditions": {
            "marginal_ge_0.75": cond_marginal,
            "within_source_point_ge_0.30": cond_within_point,
            "within_source_CI_excludes_zero": cond_within_ci,
            "T13_exceeds_null": cond_shuffle,
            "R6_centroid_margin_met": cond_r6,
        },
    }


# ── H3a / H3b (with + without projdiff) ─────────────────────────────────────


def compute_h3(df: pd.DataFrame) -> dict:
    score_vectors_full = {
        axis: df[axis].astype(float).values for axis in AXIS_SPECS_RECIPE_AGREEMENT
    }
    score_vectors_no_pd = {
        axis: v for axis, v in score_vectors_full.items() if axis != "pvec_chenstyle_L20_projdiff"
    }
    mat_full, axes_full = recipe_agreement_matrix(score_vectors_full)
    mat_no_pd, axes_no_pd = recipe_agreement_matrix(score_vectors_no_pd)
    _write_matrix_csv(mat_full, axes_full, PHASE2_DIR / "recipe_agreement_matrix_with_projdiff.csv")
    _write_matrix_csv(mat_no_pd, axes_no_pd, PHASE2_DIR / "recipe_agreement_matrix_no_projdiff.csv")
    stats_full = off_diagonal_stats(mat_full)
    stats_no_pd = off_diagonal_stats(mat_no_pd)
    null_full = marker_shuffle_permutation_null(
        score_vectors_full,
        df["marker_leakage_rate"].astype(float).values,
        n_permutations=DEFAULT_PERMUTATION_N,
    )
    null_no_pd = marker_shuffle_permutation_null(
        score_vectors_no_pd,
        df["marker_leakage_rate"].astype(float).values,
        n_permutations=DEFAULT_PERMUTATION_N,
    )
    return {
        "with_projdiff": {
            "off_diagonal_mean": stats_full["mean"],
            "off_diagonal_min": stats_full["min"],
            "H3a_passes": bool(stats_full["mean"] >= H3A_THRESHOLD),
            "H3b_null": null_full,
        },
        "without_projdiff": {
            "off_diagonal_mean": stats_no_pd["mean"],
            "off_diagonal_min": stats_no_pd["min"],
            "H3a_passes": bool(stats_no_pd["mean"] >= H3A_THRESHOLD),
            "H3b_null": null_no_pd,
        },
        "R4_note": (
            "Within-source rankings of pvec_chenstyle_L20_projdiff are "
            "identical to pvec_chenstyle_L20 by construction (single fixed "
            "helpful_test_act). On the 50-row Phase 2 table the pair's "
            "pairwise ρ ≈ 1.0 unless the per-source constant offset "
            "differs enough across sources to perturb the marginal ranking. "
            "The 'without_projdiff' variant is the principled estimate."
        ),
    }


def _write_matrix_csv(mat: np.ndarray, axes: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", *axes])
        for i, name in enumerate(axes):
            w.writerow([name, *[f"{x:.6f}" for x in mat[i]]])


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(LEAKAGE_CSV))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    assert len(df) == 50, f"expected 50 rows, got {len(df)}"
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)

    print("[Phase 2] per-axis stats...")
    per_axis = compute_per_axis_stats(df)
    dump_json(
        {"per_axis": per_axis, "metadata": build_run_metadata({"phase": "phase2"})},
        PHASE2_DIR / "per_axis_stats.json",
    )

    print("[Phase 2] H2 verdict + T9/R3/R9/T13/R6...")
    h2 = compute_h2_verdict(df)
    dump_json(
        {"h2_verdict": h2, "metadata": build_run_metadata({"phase": "phase2"})},
        PHASE2_DIR / "h2_verdict.json",
    )
    dump_json(h2["within_source_T9_R9"], PHASE2_DIR / "source_partial_rho.json")
    dump_json(h2["T13_source_shuffle_null"], PHASE2_DIR / "source_shuffle_permutation.json")

    print("[Phase 2] H3a / H3b recipe agreement (with + without projdiff)...")
    h3 = compute_h3(df)
    dump_json(h3, PHASE2_DIR / "permutation_null.json")

    # BH-FDR (R8) — plan: "scoped to 9 single-axis Spearman p-values (one per
    # NON-HEADLINE axis)". HEADLINE_AXIS p-value is reported separately.
    p_values = {axis: per_axis[axis]["spearman_p"] for axis in NEW_AXES if axis != HEADLINE_AXIS}
    bh = benjamini_hochberg(p_values, alpha=0.10)
    dump_json(
        {
            "scope_note": (
                "R8: BH-FDR (α=0.10) applied to the non-headline single-axis "
                f"Spearman ρ p-values ({len(p_values)} axes; headline "
                f"{HEADLINE_AXIS!r} excluded from pool per plan R8)."
            ),
            "alpha": 0.10,
            "headline_axis_excluded": HEADLINE_AXIS,
            "bh_results": bh,
        },
        PHASE2_DIR / "bh_fdr.json",
    )

    # R11 patch: update the cohesion file with the ratio against Phase 1 mean
    # trigger-centroid variance once it's computable. Phase 1 doesn't expose a
    # canonical scalar for this; we approximate with the Phase 1 augmented CSV
    # per-axis variance of `pcentroid_methodB_L20` (Phase 1's per-trigger
    # mean-response centroid summary). Optional patch — skipped silently if
    # Phase 1 hasn't run.
    _maybe_patch_r11_ratio()

    print(f"[Phase 2] all outputs written to {PHASE2_DIR.relative_to(REPO_ROOT)}")


def _maybe_patch_r11_ratio() -> None:
    """Compute cross_persona_centroid_variance_ratio if Phase 1 data available.

    C6: both numerator and denominator MUST be variance in hidden-state space
    (~3584-dim Qwen-2.5-7B). The previous implementation computed the
    denominator over a column of *cosine similarity scores* (dimensionless),
    making the ratio dimensionally meaningless. Plan R11: "cross-persona
    centroid variance ratio = ratio to Phase 1's mean trigger-centroid
    variance (same-units reference)."

    Hidden-state denominator: load Phase 1's per-trigger pos-side
    mean-response centroid at L20 (saved by i368_extract_chenstyle_vectors
    under data/persona_vectors_chenstyle/.../i181/{trigger}/
    pos_centroids_mean_response.pt), stack across triggers, take elementwise
    variance, then mean-pool to a scalar. Both numerator (already in
    hidden-state space; persona_pos_set_cohesion.json) and denominator
    (trigger pos centroids) are now variance-in-hidden-dim scalars.
    """
    import json

    import torch

    from explore_persona_space.axis.chenstyle import HEADLINE_LAYER
    from scripts.i368_extract_chenstyle_vectors import (  # type: ignore
        OUTPUT_BASE,
        TRIGGER_NAMES,
    )

    cohesion_path = PHASE2_DIR / "persona_pos_set_cohesion.json"
    if not cohesion_path.exists():
        return

    trigger_centroids: list[torch.Tensor] = []
    for trig in TRIGGER_NAMES:
        path = OUTPUT_BASE / "i181" / trig / "pos_centroids_mean_response.pt"
        if not path.exists():
            return  # Phase 1 extraction hasn't run yet — skip silently
        d = torch.load(path, weights_only=True)
        trigger_centroids.append(d[HEADLINE_LAYER].float())
    if len(trigger_centroids) < 2:
        return
    trigger_var = float(torch.stack(trigger_centroids).var(dim=0).mean().item())

    with open(cohesion_path) as f:
        coh = json.load(f)
    cross_var = coh.get("cross_persona_centroid_variance")
    if cross_var is None or trigger_var <= 0:
        return
    coh["cross_persona_centroid_variance_ratio_to_phase1_mean"] = cross_var / trigger_var
    coh["denominator_source"] = (
        f"hidden-state variance of {len(trigger_centroids)} Phase 1 trigger "
        f"pos-centroids at L{HEADLINE_LAYER} (mean-response aggregation); "
        "same-units reference per plan R11."
    )
    coh["sonnet_flatness_flag"] = bool(coh["inter_persona_centered_cosine_mean"] > 0.7)
    dump_json(coh, cohesion_path)


if __name__ == "__main__":
    main()
