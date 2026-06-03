#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #478 PHASE 4 — primary + secondary tests + robustness + Level-1 decomposition.

Per plan v5 §4.8 PHASE 4 + §6.4-§6.7 (mandatory implementer deliverables) +
§6.8 Level-1 (FREE — runs on core cells only):

Pipeline:
  1. Load all eval_results/issue_478/cell_*_seed*/result.json.
  2. Build per-(cell, seed, persona) tidy CSV (64 × 35 = 2240 rows).
  3. Compute min_dist + band for each row from the 111-persona cosine matrix.
  4. PRIMARY: gap_shrinkage_test() on band-averaged (FAR − NEAR) contrast,
     with FIXED bands persona-pinned to the FULL 16-pool (§6.7 #4).
  5. CO-PRIMARY (promoted per §6.7 #1): mixed-effects
     deltaLogP_mean ~ K * log(min_dist) + (1|subset) + (1|persona).
  6. Per-K marginal slopes (§6.7 #2 HERO candidate, NOT exploratory).
  7. Robustness:
       - Leave-one-persona-out × 35
       - DFBETAS on K × log(d) interaction
       - No-comedy refit (§6.7 #5 MANDATORY, conservative interpretation)
       - KL DV refit
       - Residualized-leakage check (§6.7 #3 — observed gap-shrinkage minus
         what a K=1-fitted f(d) predicts at each cell's actual min-distances)
       - Min-dist-to-K-subset summaries (§6.7 #3)
  8. §6.8 Level-1 superposition decomposition — pre-registered MEAN combiner
     primary null + LSE/max/fitted-linear sensitivity, cluster-robust SEs at
     (cell_id, seed), leave-one-K≥2-cell-out CV for fitted-linear, held-out R²
     floor.
  9. Output regression.json + tidy CSV + superposition_decomposition.json.

CLI:
  --eval-dir          Default: eval_results/issue_478
  --aggregate-dir     Default: eval_results/issue_478/aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue478_common import (  # noqa: E402
    COMEDY_FAMILY,
    FAR_BANDS,
    HELD_OUT_35,
    HELD_OUT_BANDS,
    K_VALUES,
    NEAR_BANDS,
    SEEDS,
    SUBSETS_PER_K,
    band_of,
    load_cosine_distance_matrix,
    min_dist_to_set,
)


def load_cell_results(eval_dir: Path, track_filter: str | None = None) -> list[dict]:
    """Read every cell_*_seed*/result.json under eval_dir, optionally filtered by track."""
    files = sorted(eval_dir.glob("cell_*_seed*/result.json"))
    if not files:
        raise SystemExit(f"No cell result.json files found under {eval_dir}")
    out = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            if track_filter is None or data.get("track") == track_filter:
                out.append(data)
        except Exception as e:
            log.warning("Skipping malformed %s (%s)", f, e)
    log.info(
        "Loaded %d cell result.json files (track_filter=%s) from %s",
        len(out),
        track_filter,
        eval_dir,
    )
    return out


def build_tidy_rows(
    results: list[dict],
    names: list[str],
    distance: list[list[float]],
) -> list[dict]:
    """Build per-(cell, seed, persona) rows with min_dist + band.

    Bands are FIXED (persona-pinned to FULL 16-pool, §6.7 #4) — min_dist
    here is the per-(cell × persona) min-distance to the cell's K-subset
    (used for the K×log(d) interaction + residualized-leakage check).
    """
    rows: list[dict] = []
    for r in results:
        cell_id = r["cell_id"]
        seed = r["seed"]
        K = r["K"]
        spec = r["spec"]
        positives = spec["positives"]
        subset_id = "-".join(sorted(positives))

        for persona, payload in r["eval"]["held_out"].items():
            d_min = min_dist_to_set(persona, positives, names, distance)
            band = band_of(persona)
            if band is None:
                # Shouldn't happen — held_out personas all pinned to bands.
                continue
            rows.append(
                {
                    "cell_id": cell_id,
                    "seed": seed,
                    "K": K,
                    "subset_id": subset_id,
                    "positives": ";".join(sorted(positives)),
                    "held_out_persona": persona,
                    "min_dist": d_min,
                    "band": band,
                    "deltaLogP_mean": payload["deltaLogP_mean"],
                    "logp_trained_mean": payload["logp_trained_mean"],
                    "logp_base_mean": payload["logp_base_mean"],
                    "emit_rate": payload["emit_rate"],
                    "kl_mean": payload["kl_mean"],
                }
            )
    return rows


def gap_shrinkage_test(rows: list[dict], value_key: str = "deltaLogP_mean") -> dict:
    """PRIMARY: log2(K)-gap-shrinkage on band-averaged (FAR − NEAR) contrast.

    Bands are FIXED (§6.7 #4): NEAR = near + near-mid; FAR = far + very-far + tail.
    For each K, compute gap_K = mean(FAR rows) - mean(NEAR rows); regress
    gap_K on log2(K).

    Returns dict with slope, p, se, gaps_per_K, far/near means per K.
    """
    from scipy import stats

    near_bands = set(NEAR_BANDS)
    far_bands = set(FAR_BANDS)
    Ks = sorted(K_VALUES)
    gaps, gap_se = [], []
    far_means, near_means = {}, {}

    for K in Ks:
        far_vals = [r[value_key] for r in rows if r["K"] == K and r["band"] in far_bands]
        near_vals = [r[value_key] for r in rows if r["K"] == K and r["band"] in near_bands]
        if not far_vals or not near_vals:
            gaps.append(float("nan"))
            gap_se.append(float("nan"))
            continue
        far_m = sum(far_vals) / len(far_vals)
        near_m = sum(near_vals) / len(near_vals)
        far_means[K] = far_m
        near_means[K] = near_m
        gaps.append(far_m - near_m)
        # Standard error of the difference of means.
        far_var = (np.var(far_vals, ddof=1) if len(far_vals) > 1 else 0.0) / len(far_vals)
        near_var = (np.var(near_vals, ddof=1) if len(near_vals) > 1 else 0.0) / len(near_vals)
        gap_se.append(math.sqrt(far_var + near_var))

    logK = np.log2(np.array(Ks, dtype=float))
    valid = ~np.isnan(gaps)
    if valid.sum() < 2:
        return {
            "slope": None,
            "p": None,
            "se": None,
            "gaps_per_K": dict(zip(Ks, gaps, strict=False)),
            "far_means": far_means,
            "near_means": near_means,
            "gap_se": dict(zip(Ks, gap_se, strict=False)),
            "n_valid_K_levels": int(valid.sum()),
        }
    slope, intercept, r, p, se = stats.linregress(logK[valid], np.array(gaps)[valid])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "p": float(p),
        "se": float(se),
        "r_squared": float(r) ** 2,
        "gaps_per_K": {int(K): float(g) for K, g in zip(Ks, gaps, strict=False)},
        "gap_se_per_K": {int(K): float(g) for K, g in zip(Ks, gap_se, strict=False)},
        "far_means_per_K": {int(K): float(v) for K, v in far_means.items()},
        "near_means_per_K": {int(K): float(v) for K, v in near_means.items()},
        "n_valid_K_levels": int(valid.sum()),
    }


def mixed_effects_K_x_logd(rows: list[dict]) -> dict:
    """CO-PRIMARY (§6.7 #1): deltaLogP_mean ~ K * log(min_dist) + (1|subset) + (1|persona).

    Tool order: pymer4 → rpy2+lme4 → statsmodels.MixedLM with vc_formula.
    Inherited fallback chain from #405. Falls back to a plain OLS interaction
    fit + diagnostic message if all three are unavailable.
    """
    try:
        import pandas as pd
    except ImportError as e:
        log.error("pandas missing — co-primary skipped: %s", e)
        return {"status": "SKIPPED", "reason": "pandas missing"}

    df = pd.DataFrame(rows)
    df = df[df["min_dist"] > 0].copy()
    df["log_min_dist"] = np.log(df["min_dist"])

    # Try statsmodels MixedLM first (the most robust pure-Python option;
    # pymer4/lme4 add an R dependency we don't ship by default).
    try:
        import statsmodels.formula.api as smf

        model = smf.mixedlm(
            "deltaLogP_mean ~ K * log_min_dist",
            df,
            groups=df["subset_id"],
            vc_formula={"persona": "0 + C(held_out_persona)"},
        )
        result = model.fit(reml=False, method="lbfgs")
        return {
            "status": "OK",
            "tool": "statsmodels.MixedLM (subset random intercept + persona vc_formula)",
            "summary": str(result.summary()),
            "fixed_effects": result.params.to_dict(),
            "pvalues": result.pvalues.to_dict(),
            "interaction_p": float(result.pvalues.get("K:log_min_dist", float("nan"))),
            "interaction_beta": float(result.params.get("K:log_min_dist", float("nan"))),
        }
    except Exception as e:
        log.warning("MixedLM failed (%s); falling back to OLS K×log(d) interaction", e)
        try:
            import statsmodels.api as sm

            X = df[["K", "log_min_dist"]].copy()
            X["K_x_logd"] = X["K"] * X["log_min_dist"]
            X = sm.add_constant(X)
            y = df["deltaLogP_mean"]
            ols = sm.OLS(y, X).fit()
            return {
                "status": "OK_OLS_FALLBACK",
                "tool": "statsmodels.OLS (no random effects)",
                "summary": str(ols.summary()),
                "fixed_effects": ols.params.to_dict(),
                "pvalues": ols.pvalues.to_dict(),
                "interaction_p": float(ols.pvalues.get("K_x_logd", float("nan"))),
                "interaction_beta": float(ols.params.get("K_x_logd", float("nan"))),
            }
        except Exception as e2:
            return {"status": "FAILED", "reason": f"{e!r} then {e2!r}"}


def per_K_marginal_slopes(rows: list[dict]) -> dict:
    """§6.7 #2 HERO candidate: per-K OLS deltaLogP_mean ~ log(min_dist).

    One fit per K stratum; report β, SE, p, n.
    """
    from scipy import stats

    out: dict[int, dict] = {}
    for K in sorted(K_VALUES):
        sub = [r for r in rows if r["K"] == K and r["min_dist"] > 0]
        if len(sub) < 3:
            out[K] = {"n": len(sub), "slope": None, "p": None, "se": None}
            continue
        x = np.log([r["min_dist"] for r in sub])
        y = np.array([r["deltaLogP_mean"] for r in sub])
        slope, intercept, r, p, se = stats.linregress(x, y)
        out[K] = {
            "n": len(sub),
            "slope": float(slope),
            "intercept": float(intercept),
            "p": float(p),
            "se": float(se),
            "r_squared": float(r) ** 2,
        }
    return out


def leave_one_persona_out(rows: list[dict]) -> dict:
    """Refit gap-shrinkage AND K×log(d) with each persona dropped × 35."""

    out: dict[str, dict] = {}
    personas = sorted({r["held_out_persona"] for r in rows})
    for drop in personas:
        sub_rows = [r for r in rows if r["held_out_persona"] != drop]
        gap = gap_shrinkage_test(sub_rows)
        out[drop] = {
            "gap_slope": gap.get("slope"),
            "gap_p": gap.get("p"),
            "gap_K1": gap.get("gaps_per_K", {}).get(1),
            "gap_K8": gap.get("gaps_per_K", {}).get(8),
        }
    return out


def dfbetas_K_x_logd(rows: list[dict]) -> dict:
    """Per-row DFBETAS on the K × log(min_dist) interaction (OLS).

    Reports top-5 highest-magnitude DFBETAS + count exceeding 2/sqrt(N) threshold.
    """
    import statsmodels.api as sm

    sub = [r for r in rows if r["min_dist"] > 0]
    df_x = np.array(
        [[r["K"], math.log(r["min_dist"]), r["K"] * math.log(r["min_dist"])] for r in sub]
    )
    df_y = np.array([r["deltaLogP_mean"] for r in sub])
    X = sm.add_constant(df_x)
    model = sm.OLS(df_y, X).fit()
    infl = model.get_influence()
    dfb = infl.dfbetas
    interaction_idx = 3
    dfb_interaction = dfb[:, interaction_idx]
    N = len(sub)
    threshold = 2.0 / math.sqrt(N)
    flagged_idx = [int(i) for i in np.argsort(-np.abs(dfb_interaction))[:5]]
    top5 = [
        {
            "row_idx": int(i),
            "cell_id": sub[i]["cell_id"],
            "seed": sub[i]["seed"],
            "persona": sub[i]["held_out_persona"],
            "K": sub[i]["K"],
            "min_dist": sub[i]["min_dist"],
            "dfbetas": float(dfb_interaction[i]),
        }
        for i in flagged_idx
    ]
    return {
        "N": N,
        "threshold_2_over_sqrt_N": float(threshold),
        "n_exceeding_threshold": int((np.abs(dfb_interaction) > threshold).sum()),
        "top_5": top5,
    }


def no_comedy_refit(rows: list[dict]) -> dict:
    """§6.7 #5 + §6.8 v5 MANDATORY: drop 9 comedy-family personas; re-run gap-shrinkage.

    Survival criterion (round-2 CONCERN 5, per plan §6.8 v5): a flattening claim
    can be reported as DISTANCE-driven (not comedy-axis-driven) only if ALL of:
      (a) direction agrees with full-panel slope, AND
      (b) the no-comedy slope's 95% CI INCLUDES the full-panel point estimate, AND
      (c) the no-comedy slope SE is not >2× the full-panel SE (else
          "underpowered / uninterpretable" — the non-comedy FAR group is
          8 personas, at the lower edge of stability).
    Anything else → narrate as "distance-vs-comedy unresolved" instead of
    treating survival as positive evidence (per §6.7 #5).
    """
    sub = [r for r in rows if r["held_out_persona"] not in COMEDY_FAMILY]
    no_comedy_personas_dropped = sorted(set(COMEDY_FAMILY) & {r["held_out_persona"] for r in rows})
    gap_no_comedy = gap_shrinkage_test(sub)
    gap_full = gap_shrinkage_test(rows)

    # (a) Direction agreement.
    direction_agrees = (
        gap_no_comedy.get("slope") is not None
        and gap_full.get("slope") is not None
        and gap_no_comedy["slope"] * gap_full["slope"] > 0
    )

    # (b) no-comedy 95% CI on slope includes full-panel slope point estimate.
    # gap_shrinkage_test returns slope + SE; 95% CI = slope ± 1.96 * SE.
    ci_includes_full = None
    nc_slope = gap_no_comedy.get("slope")
    nc_se = gap_no_comedy.get("se")
    full_slope = gap_full.get("slope")
    if nc_slope is not None and nc_se is not None and full_slope is not None:
        ci_lo = nc_slope - 1.96 * nc_se
        ci_hi = nc_slope + 1.96 * nc_se
        ci_includes_full = bool(ci_lo <= full_slope <= ci_hi)

    # (c) SE ratio < 2× — else underpowered.
    se_ratio = None
    se_ratio_pass = None
    full_se = gap_full.get("se")
    if nc_se is not None and full_se is not None and full_se > 0:
        se_ratio = float(nc_se / full_se)
        se_ratio_pass = bool(se_ratio <= 2.0)

    # Aggregate survival status string.
    if any(x is None for x in (direction_agrees, ci_includes_full, se_ratio_pass)):
        survival_status = "INDETERMINATE — could not compute one or more gates"
    elif se_ratio_pass is False:
        survival_status = (
            f"UNDERPOWERED — no-comedy SE {nc_se:.3f} > 2× full-panel SE "
            f"{full_se:.3f} (ratio {se_ratio:.2f}); report distance-vs-comedy unresolved"
        )
    elif direction_agrees and ci_includes_full and se_ratio_pass:
        survival_status = "SURVIVES — distance-driven read is supported"
    else:
        bits = []
        if not direction_agrees:
            bits.append("direction flips")
        if not ci_includes_full:
            bits.append("no-comedy 95% CI excludes full-panel slope")
        survival_status = "FAILS — " + " AND ".join(bits) + "; report comedy-axis confound"

    # Compute the dropped persona list explicitly (audit trail).
    return {
        "full_panel": gap_full,
        "no_comedy": gap_no_comedy,
        "comedy_personas_dropped": no_comedy_personas_dropped,
        "n_personas_dropped": len(no_comedy_personas_dropped),
        "n_rows_kept": len(sub),
        "n_rows_total": len(rows),
        "survival": {
            "direction_agrees": direction_agrees,
            "ci_includes_full_panel_slope": ci_includes_full,
            "no_comedy_slope_95ci": (
                None
                if (nc_slope is None or nc_se is None)
                else [nc_slope - 1.96 * nc_se, nc_slope + 1.96 * nc_se]
            ),
            "full_panel_slope_point_estimate": full_slope,
            "se_ratio_no_comedy_over_full": se_ratio,
            "se_ratio_pass_le_2x": se_ratio_pass,
            "status": survival_status,
        },
        "scope_caveat": (
            "Per §6.8 v5: survival = direction agrees AND no-comedy 95% CI includes "
            "full-panel slope AND SE ratio ≤ 2×. The non-comedy FAR group is "
            "8 personas (lower edge of stability); a 'FAILS' or 'UNDERPOWERED' "
            "status MUST be narrated as 'distance-vs-comedy unresolved' rather "
            "than positive evidence per §6.7 #5."
        ),
    }


def kl_dv_refit(rows: list[dict]) -> dict:
    """§6.7 #5 KL-DV refit: same gap-shrinkage on kl_mean instead of deltaLogP_mean."""
    return gap_shrinkage_test(rows, value_key="kl_mean")


def residualized_leakage_check(rows: list[dict]) -> dict:
    """§6.7 #3: residualized leakage = observed − K=1-fitted f(d) prediction.

    Fit f(d) on K=1 cells: deltaLogP_mean ~ log(min_dist). Predict at each
    K≥2 cell's actual (per-persona) min-distances. Residual = observed −
    prediction. The headline reports distance-flattening only if observed
    gap-shrinkage EXCEEDS the mechanical re-binning prediction.
    """
    from scipy import stats

    k1_rows = [r for r in rows if r["K"] == 1 and r["min_dist"] > 0]
    if len(k1_rows) < 3:
        return {"status": "SKIPPED", "reason": "need ≥3 K=1 rows to fit f(d)"}
    x = np.log([r["min_dist"] for r in k1_rows])
    y = np.array([r["deltaLogP_mean"] for r in k1_rows])
    slope, intercept, _r, _p, _se = stats.linregress(x, y)

    # Predict at every row's actual min-distance; compute residual.
    residualized_rows = []
    for r in rows:
        if r["min_dist"] <= 0:
            continue
        pred = intercept + slope * math.log(r["min_dist"])
        residualized_rows.append({**r, "deltaLogP_residualized": r["deltaLogP_mean"] - pred})

    # Re-run gap-shrinkage on the residualized DV.
    residualized_gap = gap_shrinkage_test(residualized_rows, value_key="deltaLogP_residualized")
    return {
        "status": "OK",
        "k1_fit": {"slope": float(slope), "intercept": float(intercept)},
        "residualized_gap_shrinkage": residualized_gap,
        "note": (
            "Per §6.7 #3: H1 (flattening) is the distance-driven read only if "
            "the residualized gap-shrinkage slope is non-zero (the OBSERVED shrinkage "
            "exceeds what fixed f(d) + per-K binning already predicts)."
        ),
    }


def min_dist_to_K_subset_summary(rows: list[dict]) -> dict:
    """§6.7 #3: per-band per-K mean min-dist-to-K-subset (diagnostic preamble)."""
    out: dict[str, dict[int, float]] = {b: {} for b in HELD_OUT_BANDS}
    for band in HELD_OUT_BANDS:
        for K in sorted(K_VALUES):
            vals = [r["min_dist"] for r in rows if r["band"] == band and r["K"] == K]
            out[band][int(K)] = float(np.mean(vals)) if vals else float("nan")
    return out


# ────────────────────────────────────────────────────────────────────────────
# §6.8 Level-1 superposition decomposition (FREE — uses only K=1 core cells)
# ────────────────────────────────────────────────────────────────────────────

CombinerName = str


def _build_k1_matrix(
    k1_rows: list[dict],
) -> dict[tuple[str, str, int], float]:
    """L_{K=1}[A, C, seed] = on-policy log P(※) shift averaged over the cell × seed."""
    out: dict[tuple[str, str, int], float] = {}
    for r in k1_rows:
        a = r["positives"].split(";")[0]  # K=1 ⇒ single positive
        c = r["held_out_persona"]
        seed = r["seed"]
        out[(a, c, seed)] = r["deltaLogP_mean"]
    return out


def _k1_average_across_seeds(
    k1_matrix: dict[tuple[str, str, int], float],
    seeds: list[int],
) -> dict[tuple[str, str], float]:
    """Average K=1 leakage across the available seeds (plan §6.8 Level-1 input)."""
    by_ac: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (a, c, _seed), v in k1_matrix.items():
        by_ac[(a, c)].append(v)
    return {(a, c): sum(vs) / len(vs) for (a, c), vs in by_ac.items() if vs}


def _combiner_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _combiner_max(values: list[float]) -> float:
    return max(values)


def _combiner_logsumexp(values: list[float]) -> float:
    return float(np.log(np.sum(np.exp(values))))


def superposition_decomposition_level1(rows: list[dict]) -> dict:  # noqa: C901
    """§6.8 Level-1 DIRECTIONAL (cluster-robust SEs at (cell_id, seed) level).

    For each K≥2 cell with source set S = {A_1, ..., A_K}, predict joint
    leakage to each held-out C from the K=1 cells of S's members using
    four combiners (mean = pre-registered primary null; LSE / max / fitted
    linear = sensitivity).

    Outputs:
      - per_K held-out R² of each combiner (leave-one-K≥2-cell-out CV for
        fitted-linear; non-CV R² for mean / LSE / max).
      - per_(band, K) mean signed residual (observed − mean-combiner) with
        cluster-robust 95% CI at (cell_id, seed).
      - fitted-linear β_i diagnostic per K (β_i ≈ 0.5 ⇒ linear-dose additive).

    Plan §6.8 caveats: K=1 cells are SEPARATE LoRAs (cross-model additivity
    assumption); K=1 cells are 2-8× over-dosed per source vs matched K-cell.
    Level-1 cannot cleanly separate these — narrate honestly in clean result.
    """
    # K=1 panel — average across seeds.
    k1_rows = [r for r in rows if r["K"] == 1]
    if not k1_rows:
        return {"status": "SKIPPED", "reason": "no K=1 rows"}
    k1_matrix = _build_k1_matrix(k1_rows)
    k1_avg = _k1_average_across_seeds(k1_matrix, list(SEEDS))

    Ks = [K for K in K_VALUES if K >= 2]
    out: dict[str, dict] = {"per_K": {}}

    # Build per-K cluster (cell_id, seed) → list of (persona, observed,
    # combiner_predictions{mean,lse,max}).
    per_K_residuals_mean: dict[int, list[dict]] = defaultdict(list)
    per_K_combiner_predictions: dict[int, list[dict]] = defaultdict(list)

    # Round-2 BLOCKER 4 defensive guard + coverage report. The K=1 extension to
    # all 16 sources should cover every K≥2 source; the guard catches a future
    # regression (someone shrinks K=1 again) by both SKIPPING uncoverable cells
    # and REPORTING per-K coverage so any silent drop is visible in the
    # regression.json.
    per_K_coverage: dict[int, dict] = {}
    for K_check in Ks:
        K_rows = [r for r in rows if r["K"] == K_check]
        K_cells = sorted({r["cell_id"] for r in K_rows})
        covered_cells: list[str] = []
        uncovered_cells: list[dict] = []
        for cid in K_cells:
            # All rows for this cell share the same positives set.
            cell_rows = [r for r in K_rows if r["cell_id"] == cid]
            S = cell_rows[0]["positives"].split(";")
            missing_sources = [a for a in S if not any((a, c) in k1_avg for c in [HELD_OUT_35[0]])]
            # Recheck with actual held-out personas — a source is "covered" iff
            # its (a, C) entry exists for AT LEAST one held-out C (in practice,
            # K=1 cell ran → it produced entries for all 35 personas).
            actually_missing = [a for a in S if not any((a, c) in k1_avg for c in HELD_OUT_35)]
            if actually_missing:
                uncovered_cells.append({"cell_id": cid, "missing_sources": actually_missing})
            else:
                covered_cells.append(cid)
            _ = missing_sources
        per_K_coverage[K_check] = {
            "n_cells_total": len(K_cells),
            "n_cells_covered": len(covered_cells),
            "n_cells_uncovered": len(uncovered_cells),
            "uncovered_cells": uncovered_cells,
        }
        if uncovered_cells:
            log.warning(
                "Level-1 K=%d: %d of %d cells lack K=1-covered sources; SKIPPING. "
                "If this is unexpected, check that K=1 was extended to all 16 POOL_16 "
                "singletons per round-2 BLOCKER 4 (issue478_validate_design.build_subsets).",
                K_check,
                len(uncovered_cells),
                len(K_cells),
            )

    for r in rows:
        if r["K"] < 2:
            continue
        K = r["K"]
        S = r["positives"].split(";")
        C = r["held_out_persona"]
        # Pull each K=1 input for A_i, C
        k1_inputs = []
        missing = False
        for a in S:
            if (a, C) not in k1_avg:
                # K=1 cell wasn't run for this source persona — skip
                missing = True
                break
            k1_inputs.append(k1_avg[(a, C)])
        if missing or not k1_inputs:
            continue
        pred_mean = _combiner_mean(k1_inputs)
        pred_lse = _combiner_logsumexp(k1_inputs)
        pred_max = _combiner_max(k1_inputs)
        observed = r["deltaLogP_mean"]
        per_K_combiner_predictions[K].append(
            {
                "cell_id": r["cell_id"],
                "seed": r["seed"],
                "persona": C,
                "band": r["band"],
                "K": K,
                "observed": observed,
                "pred_mean": pred_mean,
                "pred_lse": pred_lse,
                "pred_max": pred_max,
                "residual_mean": observed - pred_mean,
                "residual_lse": observed - pred_lse,
                "residual_max": observed - pred_max,
                "k1_inputs": k1_inputs,
            }
        )
        per_K_residuals_mean[K].append(observed - pred_mean)

    # Per-K held-out R² for each combiner (non-CV for mean/lse/max).
    for K in Ks:
        preds = per_K_combiner_predictions[K]
        if not preds:
            out["per_K"][str(K)] = {"status": "no_predictions"}
            continue
        obs = np.array([p["observed"] for p in preds])
        ss_tot = float(((obs - obs.mean()) ** 2).sum())
        per_combiner_r2 = {}
        for c in ["mean", "lse", "max"]:
            pred = np.array([p[f"pred_{c}"] for p in preds])
            ss_res = float(((obs - pred) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            per_combiner_r2[c] = r2

        # Cluster-robust SE of mean-combiner residual at (cell_id, seed).
        clusters: dict[tuple[str, int], list[float]] = defaultdict(list)
        for p in preds:
            clusters[(p["cell_id"], p["seed"])].append(p["residual_mean"])
        cluster_means = [sum(vs) / len(vs) for vs in clusters.values()]
        n_clusters = len(cluster_means)
        if n_clusters >= 2:
            cluster_se = float(np.std(cluster_means, ddof=1) / math.sqrt(n_clusters))
        else:
            cluster_se = float("nan")
        mean_resid = float(np.mean([p["residual_mean"] for p in preds]))
        ci95 = (
            (mean_resid - 1.96 * cluster_se, mean_resid + 1.96 * cluster_se)
            if not math.isnan(cluster_se)
            else (None, None)
        )

        # Per-band mean residual (informational).
        per_band_resid: dict[str, float] = {}
        for band in HELD_OUT_BANDS:
            band_preds = [p for p in preds if p["band"] == band]
            if band_preds:
                per_band_resid[band] = float(np.mean([p["residual_mean"] for p in band_preds]))

        # Fitted-linear combiner with leave-one-K≥2-cell-out CV (per §6.8).
        fitted_linear = _fitted_linear_cv(preds, K)

        # Held-out R² floor (§6.8 interpretability gate).
        floor_pass = False
        floor_required = None
        if K == 2:
            floor_required = 0.5
            floor_pass = per_combiner_r2["mean"] >= 0.5
        elif K == 4:
            floor_required = 0.4
            floor_pass = per_combiner_r2["mean"] >= 0.4
        else:  # K == 8: directional only
            floor_required = "directional"
            floor_pass = True

        out["per_K"][str(K)] = {
            "n_preds": len(preds),
            "n_cell_seed_clusters": n_clusters,
            "held_out_R2_mean_combiner": per_combiner_r2["mean"],
            "held_out_R2_lse_combiner": per_combiner_r2["lse"],
            "held_out_R2_max_combiner": per_combiner_r2["max"],
            "mean_residual_mean_combiner": mean_resid,
            "cluster_robust_se_mean_combiner": cluster_se,
            "ci95_mean_combiner": ci95,
            "per_band_mean_residual": per_band_resid,
            "fitted_linear_cv": fitted_linear,
            "interpretability_floor": {
                "required_R2": floor_required,
                "achieved_R2_mean_combiner": per_combiner_r2["mean"],
                "pass": floor_pass,
            },
        }

    out["per_K_coverage"] = {str(K): per_K_coverage[K] for K in Ks}
    out["note"] = (
        "Plan §6.8 Level-1 is DIRECTIONAL (cluster-robust SEs at "
        "(cell_id, seed)). The 2240-obs framing was pseudo-replication. Held-out "
        "R² floor required = 0.5 at K=2, 0.4 at K=4, directional-only at K=8. "
        "Mean combiner = PRE-REGISTERED primary null (linear-dose-scaled additivity). "
        "per_K_coverage reports the n cells covered by the K=1 panel — round-2 "
        "BLOCKER 4 defensive read in case of a K=1 regression."
    )
    return out


def _fitted_linear_cv(preds: list[dict], K: int) -> dict:
    """Leave-one-K≥2-cell-out CV of α + Σ β_i L_{K=1}[A_i, C].

    NOT persona-level CV — persona-level leaks within-cell correlation and
    shrinks the residual artificially (plan §6.8).

    Per the §6.8 spec we don't fit a per-arm-cell β (each cell has different
    A_i orderings). Instead, fit on stacked (cell-aggregated) rows where the
    DV is the per-(cell, persona) observed leakage and the regressor is the
    PER-PERSONA K=1 inputs flattened as a sum + mean for β_diagnostic. The
    leave-one-cell-out CV holds out ALL rows from one K-cell at a time.
    """
    cells = sorted({p["cell_id"] for p in preds})
    if len(cells) < 3:
        return {"status": "SKIPPED", "reason": f"need ≥3 K={K} cells for CV"}
    held_out_preds: list[float] = []
    held_out_obs: list[float] = []
    held_in_betas: list[float] = []
    for hold_cell in cells:
        train = [p for p in preds if p["cell_id"] != hold_cell]
        test = [p for p in preds if p["cell_id"] == hold_cell]
        if not train or not test:
            continue
        # Simple regression: y = α + β * mean(K=1 inputs)
        X = np.array([[1.0, sum(p["k1_inputs"]) / len(p["k1_inputs"])] for p in train])
        y = np.array([p["observed"] for p in train])
        # Closed-form OLS (small N, fine).
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue
        alpha, beta = float(coef[0]), float(coef[1])
        held_in_betas.append(beta)
        for p in test:
            mean_in = sum(p["k1_inputs"]) / len(p["k1_inputs"])
            held_out_preds.append(alpha + beta * mean_in)
            held_out_obs.append(p["observed"])

    if not held_out_preds:
        return {"status": "SKIPPED", "reason": "CV produced no held-out preds"}
    obs = np.array(held_out_obs)
    pred = np.array(held_out_preds)
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    ss_res = float(((obs - pred) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n_cv_folds": len(cells),
        "held_out_R2": r2,
        "mean_held_in_beta": float(np.mean(held_in_betas)) if held_in_betas else None,
        "interpretation": (
            "β ≈ 1/K dose-corrects K=1 inputs (β=0.5 at K=2, β=0.25 at K=4); "
            "β ≈ 1 means K=1 inputs are dose-corrected; β far from these ⇒ non-linear."
        ),
    }


# ────────────────────────────────────────────────────────────────────────────
# Optional JS-distance refit (closes #405 §4.7 gap).
# ────────────────────────────────────────────────────────────────────────────


def js_distance_refit(rows: list[dict]) -> dict:
    """§4.7 OPTIONAL: re-run headline regression with min_js_dist.

    Marked SKIPPED if a JS-distance matrix isn't trivially loadable. The
    project's #458 JS-divergence predictor lives at scripts/issue458_predictor_jsdiv.py;
    a matrix file isn't shipped by default. Don't pretend it's done.
    """
    p = (
        PROJECT_ROOT
        / "eval_results"
        / "single_token_100_persona"
        / "js_divergence_matrix_layer21.json"
    )
    if not p.exists():
        return {
            "status": "SKIPPED",
            "reason": (
                f"JS-divergence matrix not found at {p}. To enable, compute via "
                f"scripts/issue458_predictor_jsdiv.py for the 111-persona panel + "
                f"layer-21, then re-run the analyzer."
            ),
        }
    # If the file exists, the analyzer reruns the gap-shrinkage test using
    # min_js_dist; punt on implementation since no matrix exists yet.
    return {"status": "DEFERRED", "reason": "JS matrix exists but refit not yet implemented."}


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_478"),
    )
    parser.add_argument(
        "--aggregate-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_478" / "aggregate"),
    )
    parser.add_argument(
        "--skip-mixed-effects",
        action="store_true",
        help="Skip the secondary mixed-effects test (statsmodels can be slow on 2240 rows)",
    )
    parser.add_argument(
        "--allow-partial-smoke",
        action="store_true",
        help=(
            "Bypass the planned-vs-actual completeness gate (default: STRICT — every "
            "expected CORE cell × seed must be present and contribute the expected "
            "rows). Use ONLY for smoke runs / stub generation. CLAUDE.md fail-fast: "
            "without this flag, missing cells / rows raise SystemExit so a partial "
            "sweep can NEVER become a headline figure."
        ),
    )
    parser.add_argument(
        "--expected-core-cells",
        type=int,
        default=None,
        help=(
            "Override expected core cell count (cells × seeds). Defaults to "
            "len(K_VALUES) * SUBSETS_PER_K * len(SEEDS) from the plan; pass a "
            "smaller number for known descopes (must still equal the dispatched count)."
        ),
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    agg_dir = Path(args.aggregate_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)

    results = load_cell_results(eval_dir, track_filter="CORE")
    log.info("CORE cells: %d", len(results))

    # Plan §6.7 + CLAUDE.md "After Every Experiment" #8: planned-vs-actual
    # completeness gate. The headline gap-shrinkage / mixed-effects /
    # superposition statistics assume the FULL design was tested; running on
    # a partial sweep silently biases the headline by whichever cells dropped
    # out (e.g. a crashed K=8 seed shifts the gap-shrinkage slope without
    # the analyzer registering anything is wrong). Default = STRICT.
    expected_cells = args.expected_core_cells or (len(K_VALUES) * SUBSETS_PER_K * len(SEEDS))
    expected_rows = expected_cells * len(HELD_OUT_35)
    if not args.allow_partial_smoke:
        if len(results) != expected_cells:
            raise SystemExit(
                f"PARTIAL CORE SWEEP: loaded {len(results)} cells, expected {expected_cells} "
                f"(K_VALUES={K_VALUES}, SUBSETS_PER_K={SUBSETS_PER_K}, SEEDS={SEEDS}). "
                f"Re-run missing cells or pass --allow-partial-smoke (smoke / stub only). "
                f"Plan §6.7 + CLAUDE.md #8 require complete coverage before headline."
            )
    else:
        log.warning(
            "--allow-partial-smoke set: skipping cell-count gate (loaded=%d, expected=%d). "
            "DO NOT promote headline figures from this run.",
            len(results),
            expected_cells,
        )

    log.info("Loading 111-persona cosine distance matrix ...")
    names, distance = load_cosine_distance_matrix()

    log.info("Building tidy rows ...")
    rows = build_tidy_rows(results, names, distance)
    log.info("Tidy rows: %d", len(rows))

    if not args.allow_partial_smoke and len(rows) != expected_rows:
        raise SystemExit(
            f"PARTIAL TIDY ROWS: built {len(rows)} rows, expected {expected_rows} "
            f"(= {expected_cells} cells * {len(HELD_OUT_35)} held-out personas). "
            f"Some cells loaded but produced incomplete held-out evals. "
            f"Re-run or pass --allow-partial-smoke."
        )

    # Tidy CSV (for downstream R / pymer4 / spreadsheet).
    csv_path = agg_dir / "tidy.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote tidy CSV → %s", csv_path)

    log.info("Running PRIMARY gap-shrinkage test (deltaLogP_mean) ...")
    primary = gap_shrinkage_test(rows)

    log.info("Running KL-DV gap-shrinkage refit ...")
    kl_refit = kl_dv_refit(rows)

    co_primary: dict = {"status": "SKIPPED"}
    if not args.skip_mixed_effects:
        log.info("Running CO-PRIMARY mixed-effects K × log(min_dist) ...")
        co_primary = mixed_effects_K_x_logd(rows)

    log.info("Per-K marginal slopes (HERO candidate) ...")
    per_K_slopes = per_K_marginal_slopes(rows)

    log.info("Leave-one-persona-out × 35 ...")
    loo = leave_one_persona_out(rows)

    log.info("DFBETAS on K × log(d) interaction ...")
    dfb = dfbetas_K_x_logd(rows)

    log.info("No-comedy refit (§6.7 #5 MANDATORY) ...")
    no_comedy = no_comedy_refit(rows)

    log.info("Residualized-leakage check (§6.7 #3) ...")
    resid_check = residualized_leakage_check(rows)

    log.info("Min-dist-to-K-subset summary (§6.7 #3) ...")
    min_dist_summary = min_dist_to_K_subset_summary(rows)

    log.info("JS-distance refit (§4.7 optional) ...")
    js_refit = js_distance_refit(rows)

    log.info("Level-1 superposition decomposition (§6.8) ...")
    level1 = superposition_decomposition_level1(rows)

    payload = {
        "experiment": "issue_478_kdiversity_panel",
        "n_results_loaded": len(results),
        "n_tidy_rows": len(rows),
        "primary_gap_shrinkage": primary,
        "co_primary_mixed_effects": co_primary,
        "per_K_marginal_slopes": per_K_slopes,
        "robustness": {
            "leave_one_persona_out": loo,
            "dfbetas_K_x_logd": dfb,
            "no_comedy_refit": no_comedy,
            "kl_dv_refit": kl_refit,
            "residualized_leakage_check": resid_check,
            "min_dist_to_K_subset_summary": min_dist_summary,
            "js_distance_refit": js_refit,
        },
        "level1_superposition_decomposition": level1,
        "design_constants": {
            "K_values": list(K_VALUES),
            "subsets_per_K": SUBSETS_PER_K,
            "seeds": list(SEEDS),
            "NEAR_BANDS": list(NEAR_BANDS),
            "FAR_BANDS": list(FAR_BANDS),
            "comedy_family_count": len(COMEDY_FAMILY),
        },
    }
    out_path = agg_dir / "regression.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("Wrote %s", out_path)
    log.info("Phase 4 done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
