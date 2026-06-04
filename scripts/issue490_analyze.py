#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #490 PHASE 4 — primary geometric headline + dose decomposition.

Per plan v1 §4.5 PHASE 4 + §6.2 (with v2 revision §1-2 and ROUND-2 code-review
fixes):

ROUND-2 changes vs round-1:

- PRIMARY Q2 readout is now a PERSONA-LEVEL distance-adjusted regression
  ``gap_dosematched ~ is_on_axis + mean_d + asymmetry`` with cluster-robust
  standard errors at (pair, seed) groups (NO fixed effects for pair or
  seed — clustering only) rather than the raw subpanel-mean Δ_geom (which
  was confounded with mean_d when the Phase-0 off-axis subpanel couldn't
  field a distance-matched panel).
- Raw Δ_geom DEMOTED to "unadjusted diagnostic" with the mean_d delta surfaced.
- A Δ_geom-vs-mean_d_match_delta sensitivity slope is reported across pairs.
- LSE combiner now correctly computed on ABSOLUTE logp_trained and logp_base
  separately, then subtracted (per code-review CRIT-3). The previous code
  applied probability-union math to delta-logps, which is invalid.
- COMPLETENESS GATE: refuses to emit a headline unless all
  (cell × seed) combos from cell_specs.json × resolved seeds are present
  on disk. `--allow-partial` stamps outputs with `non_promotable: true`.
- slope_dose interpretation explicitly annotated as
  "dose-plus-training-volume" (SINGLE-D cells have ½ the optimizer steps +
  ½ the contrastive-negative exposure of POOLED-SINGLE-2D cells per the
  contrastive-negatives 1:1 ratio).

Pipeline:
  1. Load all eval_results/issue_490/cell_*_seed*/result.json.
  2. Load data/issue_490/source_pairs.json (per-pair subpanels + per-persona
     distances + escalate_to_3_seeds + seeds_resolved).
  3. COMPLETENESS GATE: read data/issue_490/cell_specs.json, build the
     expected (cell, seed) set, refuse to emit headline unless all present
     OR --allow-partial.
  4. For each (cell × seed × persona) build a PERSONA-LEVEL row with:
     condition, pair_id, A, B, persona, subpanel (on/off-axis), d_A, d_B,
     mean_d, asym, deltaLogP_mean, logp_trained_mean, logp_base_mean,
     kl_mean.
  5. Pivot to per-(pair × seed × subpanel × combiner) decomposition rows:
       gap_confounded = SHARED_2D − combiner(SINGLE_D_A, SINGLE_D_B)
       gap_dosematched = SHARED_2D − combiner(POOLED_2D_A, POOLED_2D_B)
       slope_dose = ½[(POOLED_A − SINGLE_A) + (POOLED_B − SINGLE_B)]
       Δ_geom = gap_dosematched(on) − gap_dosematched(off)  [DIAGNOSTIC]
     LSE combiner uses absolute logp_trained + absolute logp_base.
  6. PRIMARY Q2 (distance-adjusted): plain OLS on the persona-level
     gap_dosematched ~ is_on_axis + mean_d + asymmetry, with cluster-robust
     standard errors at (pair, seed) groups (NO fixed effects for pair or
     seed — clustering only). is_on_axis coefficient IS the headline.
  7. Paired bootstrap (10000) on raw Δ_geom + gap_dosematched + slope_dose
     across (pair × seed) tuples (diagnostic, NOT primary).
  8. Δ_geom-vs-mean_d_match_delta cross-pair regression (sensitivity).
  9. Fallback DV (full-vocab KL-from-base) mirror.
 10. Write eval_results/issue_490/aggregate/{decomposition.json,
     regression.json, tidy_*.csv, persona_level.csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue490_common import (  # noqa: E402
    COMBINERS,
    CONDITION_POOLED_2D_A,
    CONDITION_POOLED_2D_B,
    CONDITION_SHARED_2D,
    CONDITION_SINGLE_D_A,
    CONDITION_SINGLE_D_B,
    POWER_DELTA_GEOM_THRESHOLD_NATS,
    combiner,
    combiner_lse_delta_from_absolutes,
    load_cosine_distance_matrix,
)

# Whether a pair counts as a "strict distance-matched" pair for narration.
# Any pair with |on_mean_d - off_mean_d| above this threshold is excluded
# from the "strict mean-d match" sub-analysis (round-2 fix).
STRICT_DIST_MATCH_THRESHOLD_NATS: float = 0.03


def load_cell_results(eval_dir: Path) -> list[dict]:
    files = sorted(eval_dir.glob("cell_*_seed*/result.json"))
    if not files:
        raise SystemExit(f"No cell result.json files found under {eval_dir}")
    out = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            out.append(data)
        except Exception as e:
            log.warning("Skipping malformed %s (%s)", f, e)
    log.info("Loaded %d cell result.json files from %s", len(out), eval_dir)
    return out


def load_source_pairs(pairs_path: Path) -> tuple[list[dict], dict]:
    """Return (pairs, full_payload). Payload carries escalate_to_3_seeds,
    seeds_resolved, layer21_source.
    """
    payload = json.loads(pairs_path.read_text())
    return payload["pairs"], payload


def load_cell_specs(specs_path: Path) -> list[dict]:
    if not specs_path.exists():
        raise SystemExit(f"cell_specs.json missing: {specs_path}")
    return json.loads(specs_path.read_text())


def completeness_check(
    results: list[dict],
    specs: list[dict],
    seeds_resolved: list[int],
) -> dict:
    """Verify every (cell_id, seed) combo from cell_specs × seeds_resolved is
    present in results. Returns dict with missing list + complete flag.
    """
    expected = {(s["cell_id"], seed) for s in specs for seed in seeds_resolved}
    actual = {(r["cell_id"], r["seed"]) for r in results}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return {
        "n_expected": len(expected),
        "n_actual": len(actual),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "missing": [{"cell_id": c, "seed": s} for c, s in missing],
        "extra": [{"cell_id": c, "seed": s} for c, s in extra],
        "complete": len(missing) == 0,
    }


def build_persona_level_rows(
    results: list[dict],
    pairs: list[dict],
) -> list[dict]:
    """Per (cell × seed × held_out_persona) → row with distances, deltas,
    absolute logps, subpanel label.

    Persona-level rows are the input to BOTH the distance-adjusted primary
    readout AND the subpanel-mean diagnostic.
    """
    # Index pair-level distance metadata by pair_id.
    pair_meta: dict[str, dict] = {p["pair_id"]: p for p in pairs}

    rows: list[dict] = []
    for r in results:
        pair_id = r["pair_id"]
        condition = r["condition"]
        seed = r["seed"]
        meta = pair_meta.get(pair_id)
        if meta is None:
            # Result for a pair_id that's not in source_pairs.json (shouldn't
            # happen) — skip rather than fabricate distances.
            continue
        on_axis_set = set(meta["on_axis"])
        off_axis_set = set(meta["off_axis"])
        on_dists = meta.get("on_axis_distances", {})
        off_dists = meta.get("off_axis_distances", {})
        held_out = r.get("eval", {}).get("held_out", {})
        for persona, payload in held_out.items():
            if persona in on_axis_set:
                subpanel = "on_axis"
                d = on_dists.get(persona)
            elif persona in off_axis_set:
                subpanel = "off_axis"
                d = off_dists.get(persona)
            else:
                subpanel = "other"
                d = None
            row = {
                "cell_id": r["cell_id"],
                "pair_id": pair_id,
                "A": r["A"],
                "B": r["B"],
                "seed": seed,
                "condition": condition,
                "persona": persona,
                "subpanel": subpanel,
                "is_on_axis": int(subpanel == "on_axis"),
                "deltaLogP_mean": float(payload.get("deltaLogP_mean", float("nan"))),
                "logp_trained_mean": float(payload.get("logp_trained_mean", float("nan"))),
                "logp_base_mean": float(payload.get("logp_base_mean", float("nan"))),
                "kl_mean": float(payload.get("kl_mean", float("nan"))),
                "emit_rate": float(payload.get("emit_rate", float("nan"))),
                "d_A": d["d_A"] if d else float("nan"),
                "d_B": d["d_B"] if d else float("nan"),
                "mean_d": d["mean_d"] if d else float("nan"),
                "asym": d["asym"] if d else float("nan"),
            }
            rows.append(row)
    return rows


def _condition_subpanel_means(
    persona_rows: list[dict],
    value_key: str,
) -> dict:
    """Per (pair, seed, subpanel, condition) → (n_personas, mean(value_key),
    mean(logp_trained_mean), mean(logp_base_mean)).

    Returns nested dict by (pair, seed, subpanel, condition).
    """
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    grouped: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for r in persona_rows:
        if r["subpanel"] not in ("on_axis", "off_axis"):
            continue
        key = (r["pair_id"], r["seed"], r["subpanel"], r["condition"])
        grouped[key].append(r)
    for (pair_id, seed, sub, cond), rs in grouped.items():
        vals = [r[value_key] for r in rs if not np.isnan(r[value_key])]
        trained = [r["logp_trained_mean"] for r in rs if not np.isnan(r["logp_trained_mean"])]
        base = [r["logp_base_mean"] for r in rs if not np.isnan(r["logp_base_mean"])]
        if not vals:
            continue
        out[pair_id][seed][sub][cond] = {
            "n": len(rs),
            "mean_value": float(np.mean(vals)),
            "mean_logp_trained": float(np.mean(trained)) if trained else float("nan"),
            "mean_logp_base": float(np.mean(base)) if base else float("nan"),
        }
    return out


def build_decomposition_rows(
    persona_rows: list[dict],
    value_key: str = "deltaLogP_mean",
) -> list[dict]:
    """Per (pair × seed × subpanel) → derived statistics under each combiner.

    LSE combiner uses absolute logp_trained and logp_base separately
    (round-2 fix per code-review CRIT-3). Mean and max operate directly on
    the value_key (delta).

    Round-3 code-review MAJOR-2 fix: when ``value_key == "kl_mean"`` (the
    fallback DV), the LSE/Bernoulli-union combiner is NOT defined — KL is a
    divergence, not a probability, so there is no "either-source-fires"
    interpretation. The LSE columns are set to ``NaN`` explicitly with a
    note in the returned row's ``lse_note`` field, so the downstream JSON
    output AND the tidy CSV both surface "LSE inapplicable" rather than
    silently reusing the deltaLogP-path LSE values.
    """
    cond_means = _condition_subpanel_means(persona_rows, value_key)
    # KL has no Bernoulli-union interpretation; flag it.
    is_kl_fallback = value_key == "kl_mean"
    rows: list[dict] = []
    for pair_id, by_seed in cond_means.items():
        for seed, by_sub in by_seed.items():
            for sub, by_cond in by_sub.items():
                required = (
                    CONDITION_SHARED_2D,
                    CONDITION_POOLED_2D_A,
                    CONDITION_POOLED_2D_B,
                    CONDITION_SINGLE_D_A,
                    CONDITION_SINGLE_D_B,
                )
                if any(c not in by_cond for c in required):
                    missing = [c for c in required if c not in by_cond]
                    log.warning(
                        "Skipping pair=%s seed=%d subpanel=%s (missing conditions: %s)",
                        pair_id,
                        seed,
                        sub,
                        missing,
                    )
                    continue

                shared = by_cond[CONDITION_SHARED_2D]
                pooled_A = by_cond[CONDITION_POOLED_2D_A]
                pooled_B = by_cond[CONDITION_POOLED_2D_B]
                single_A = by_cond[CONDITION_SINGLE_D_A]
                single_B = by_cond[CONDITION_SINGLE_D_B]
                slope = 0.5 * (
                    (pooled_A["mean_value"] - single_A["mean_value"])
                    + (pooled_B["mean_value"] - single_B["mean_value"])
                )
                row = {
                    "pair_id": pair_id,
                    "seed": seed,
                    "subpanel": sub,
                    "value_key": value_key,
                    "n_personas_in_subpanel": shared["n"],
                    "shared_2D": shared["mean_value"],
                    "pooled_2D_A": pooled_A["mean_value"],
                    "pooled_2D_B": pooled_B["mean_value"],
                    "single_D_A": single_A["mean_value"],
                    "single_D_B": single_B["mean_value"],
                    "slope_dose": slope,
                }
                for c in COMBINERS:
                    if c == "lse":
                        # Round-3 code-review MAJOR-2 fix: KL has no
                        # Bernoulli-union interpretation. Surface NaN
                        # explicitly with a note rather than silently
                        # re-using the deltaLogP-path LSE math.
                        if is_kl_fallback:
                            row[f"gap_confounded_{c}"] = float("nan")
                            row[f"gap_dosematched_{c}"] = float("nan")
                            row["lse_note"] = (
                                "LSE/Bernoulli-union not defined for KL DV; "
                                "log P_union math requires log-probability inputs."
                            )
                            continue
                        # LSE on absolute logp_trained and logp_base
                        # separately, then subtract. Round-2 fix per
                        # code-review CRIT-3.
                        try:
                            # POOLED-A vs POOLED-B legs
                            pooled_lse_delta = combiner_lse_delta_from_absolutes(
                                trained_values=[
                                    pooled_A["mean_logp_trained"],
                                    pooled_B["mean_logp_trained"],
                                ],
                                base_values=[
                                    pooled_A["mean_logp_base"],
                                    pooled_B["mean_logp_base"],
                                ],
                            )
                            single_lse_delta = combiner_lse_delta_from_absolutes(
                                trained_values=[
                                    single_A["mean_logp_trained"],
                                    single_B["mean_logp_trained"],
                                ],
                                base_values=[
                                    single_A["mean_logp_base"],
                                    single_B["mean_logp_base"],
                                ],
                            )
                            shared_delta = shared["mean_logp_trained"] - shared["mean_logp_base"]
                            row[f"gap_confounded_{c}"] = shared_delta - single_lse_delta
                            row[f"gap_dosematched_{c}"] = shared_delta - pooled_lse_delta
                        except (ValueError, KeyError) as e:
                            log.warning(
                                "LSE combiner skipped for pair=%s seed=%d sub=%s (%s)",
                                pair_id,
                                seed,
                                sub,
                                e,
                            )
                            row[f"gap_confounded_{c}"] = float("nan")
                            row[f"gap_dosematched_{c}"] = float("nan")
                    else:
                        pooled_combined = combiner(
                            c, [pooled_A["mean_value"], pooled_B["mean_value"]]
                        )
                        single_combined = combiner(
                            c, [single_A["mean_value"], single_B["mean_value"]]
                        )
                        row[f"gap_confounded_{c}"] = shared["mean_value"] - single_combined
                        row[f"gap_dosematched_{c}"] = shared["mean_value"] - pooled_combined
                rows.append(row)
    return rows


def paired_bootstrap(values: list[float], n_boots: int = 10000, rng_seed: int = 490) -> dict:
    """Paired bootstrap mean + 95% CI."""
    if not values:
        return {"n": 0, "mean": None, "ci95": (None, None), "p_one_sided_pos": None}
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "ci95": (None, None), "p_one_sided_pos": None}
    rng = np.random.default_rng(rng_seed)
    n = len(arr)
    means = np.empty(n_boots, dtype=float)
    for i in range(n_boots):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_one_sided_pos = float((means <= 0).mean())
    return {
        "n": int(n),
        "mean": float(arr.mean()),
        "ci95": (float(lo), float(hi)),
        "p_one_sided_pos": p_one_sided_pos,
    }


def compute_per_combiner_diagnostic(decomp_rows: list[dict]) -> dict:
    """Diagnostic-only: Δ_geom + gap_dosematched(on/off) + slope_dose paired-
    bootstrap summaries across (pair × seed) tuples. NOT the primary readout
    — see compute_distance_adjusted_primary().
    """
    by_tuple: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in decomp_rows:
        by_tuple[(r["pair_id"], r["seed"])][r["subpanel"]] = r

    out: dict = {"per_combiner": {}, "n_tuples_with_both_subpanels": 0}
    for c in COMBINERS:
        delta_geom_vals: list[float] = []
        gap_dose_on_vals: list[float] = []
        gap_dose_off_vals: list[float] = []
        gap_conf_on_vals: list[float] = []
        slope_vals: list[float] = []
        complete = 0
        for (_p, _s), pair_rows in by_tuple.items():
            on = pair_rows.get("on_axis")
            off = pair_rows.get("off_axis")
            if on is None or off is None:
                continue
            complete += 1
            delta_geom_vals.append(on[f"gap_dosematched_{c}"] - off[f"gap_dosematched_{c}"])
            gap_dose_on_vals.append(on[f"gap_dosematched_{c}"])
            gap_dose_off_vals.append(off[f"gap_dosematched_{c}"])
            gap_conf_on_vals.append(on[f"gap_confounded_{c}"])
            slope_vals.append(0.5 * (on["slope_dose"] + off["slope_dose"]))
        out["per_combiner"][c] = {
            "delta_geom_raw_unadjusted": paired_bootstrap(delta_geom_vals),
            "gap_dosematched_on_axis": paired_bootstrap(gap_dose_on_vals),
            "gap_dosematched_off_axis": paired_bootstrap(gap_dose_off_vals),
            "gap_confounded_on_axis": paired_bootstrap(gap_conf_on_vals),
            "slope_dose": paired_bootstrap(slope_vals),
        }
        out["n_tuples_with_both_subpanels"] = complete
    return out


def compute_distance_adjusted_primary(  # noqa: C901 — OLS + nan-drop + diagnostics + verdict in one function
    persona_rows: list[dict],
    pairs: list[dict],
) -> dict:
    """PRIMARY Q2 readout (round-2 fix per code-review CRIT-2b).

    Fit a persona-level regression `gap_dosematched ~ is_on_axis + mean_d +
    asym` with cluster-robust SEs at (pair, seed). The is_on_axis coefficient
    is the headline — it estimates the on-axis-vs-off-axis gap contrast
    after netting out the linear effects of mean_d and asym.

    Because raw `gap_dosematched` is per-cell (computed from 5 conditions), we
    construct a per-persona gap-dosematched by reading every persona's
    SHARED-2D leakage minus the mean combiner of its POOLED-2D-A / POOLED-2D-B
    leakage (the SAME persona must be present in all 5 cells of the pair,
    which is true by construction since every cell evals over the full
    HELD_OUT_35).
    """
    # Per (pair, seed, persona) → {condition: deltaLogP_mean}, AND the
    # persona's d_A, d_B, mean_d, asym, is_on_axis tags. Pivot from the
    # persona-level rows.
    persona_index: dict[tuple[str, int, str], dict] = {}
    for r in persona_rows:
        if r["subpanel"] not in ("on_axis", "off_axis"):
            continue
        key = (r["pair_id"], r["seed"], r["persona"])
        if key not in persona_index:
            persona_index[key] = {
                "pair_id": r["pair_id"],
                "seed": r["seed"],
                "persona": r["persona"],
                "subpanel": r["subpanel"],
                "is_on_axis": r["is_on_axis"],
                "mean_d": r["mean_d"],
                "asym": r["asym"],
                "conditions": {},
            }
        persona_index[key]["conditions"][r["condition"]] = r["deltaLogP_mean"]

    # Build per-persona regression rows.
    reg_rows: list[dict] = []
    required = (
        CONDITION_SHARED_2D,
        CONDITION_POOLED_2D_A,
        CONDITION_POOLED_2D_B,
    )
    for rec in persona_index.values():
        if any(c not in rec["conditions"] for c in required):
            continue
        shared = rec["conditions"][CONDITION_SHARED_2D]
        pooled_A = rec["conditions"][CONDITION_POOLED_2D_A]
        pooled_B = rec["conditions"][CONDITION_POOLED_2D_B]
        gap_dose_mean = shared - 0.5 * (pooled_A + pooled_B)
        reg_rows.append(
            {
                "pair_id": rec["pair_id"],
                "seed": rec["seed"],
                "persona": rec["persona"],
                "is_on_axis": rec["is_on_axis"],
                "mean_d": rec["mean_d"],
                "asym": rec["asym"],
                "gap_dosematched_mean_combiner": gap_dose_mean,
            }
        )

    if not reg_rows:
        return {"status": "SKIPPED", "reason": "no persona-level rows for regression"}

    # OLS fit with cluster-robust SEs at (pair, seed).
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"status": "SKIPPED", "reason": "statsmodels not installed"}

    # Round-3 code-review MINOR-3 fix: drop non-finite rows BEFORE fitting
    # and record n_dropped + design-matrix condition number + rank in the
    # returned regression record so a degenerate fit is visible rather
    # than silent.
    n_rows_pre_drop = len(reg_rows)
    finite_rows = [
        r
        for r in reg_rows
        if all(
            np.isfinite(r[k])
            for k in ("is_on_axis", "mean_d", "asym", "gap_dosematched_mean_combiner")
        )
    ]
    n_dropped_nonfinite = n_rows_pre_drop - len(finite_rows)
    if not finite_rows:
        return {
            "status": "FAILED",
            "reason": "every row dropped (non-finite predictor or response)",
            "n_rows_pre_drop": n_rows_pre_drop,
            "n_dropped_nonfinite": n_dropped_nonfinite,
        }
    if n_dropped_nonfinite:
        log.warning(
            "compute_distance_adjusted_primary: dropped %d/%d non-finite rows",
            n_dropped_nonfinite,
            n_rows_pre_drop,
        )

    X = np.array(
        [[r["is_on_axis"], r["mean_d"], r["asym"]] for r in finite_rows],
        dtype=float,
    )
    y = np.array(
        [r["gap_dosematched_mean_combiner"] for r in finite_rows],
        dtype=float,
    )
    X_const = sm.add_constant(X)
    clusters = np.array([f"{r['pair_id']}|seed{r['seed']}" for r in finite_rows])

    # Design-matrix diagnostics (condition number + rank).
    # A condition number > 1e8 (very loose threshold) flags near-collinearity;
    # rank < n_columns means at least one predictor is a linear combination
    # of others (e.g. zero variance).
    try:
        cond_number = float(np.linalg.cond(X_const))
    except Exception:
        cond_number = float("nan")
    try:
        rank = int(np.linalg.matrix_rank(X_const))
    except Exception:
        rank = -1
    n_columns = X_const.shape[1]
    full_rank = rank == n_columns

    try:
        ols = sm.OLS(y, X_const).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    except Exception as e:
        return {
            "status": "FAILED",
            "reason": f"OLS fit raised: {e!r}",
            "n_rows_pre_drop": n_rows_pre_drop,
            "n_dropped_nonfinite": n_dropped_nonfinite,
            "design_matrix_condition_number": cond_number,
            "design_matrix_rank": rank,
            "design_matrix_n_columns": n_columns,
            "design_matrix_full_rank": full_rank,
        }
    # Note: finite_rows replaces reg_rows for the rest of this function.
    reg_rows = finite_rows

    params = {
        "intercept": float(ols.params[0]),
        "is_on_axis": float(ols.params[1]),
        "mean_d": float(ols.params[2]),
        "asym": float(ols.params[3]),
    }
    pvalues = {
        "intercept": float(ols.pvalues[0]),
        "is_on_axis": float(ols.pvalues[1]),
        "mean_d": float(ols.pvalues[2]),
        "asym": float(ols.pvalues[3]),
    }
    conf = ols.conf_int(alpha=0.05)
    ci95 = {
        "intercept": [float(conf[0, 0]), float(conf[0, 1])],
        "is_on_axis": [float(conf[1, 0]), float(conf[1, 1])],
        "mean_d": [float(conf[2, 0]), float(conf[2, 1])],
        "asym": [float(conf[3, 0]), float(conf[3, 1])],
    }
    is_on_axis_beta = params["is_on_axis"]
    is_on_axis_ci = ci95["is_on_axis"]

    if is_on_axis_ci[0] > 0:
        if is_on_axis_beta >= POWER_DELTA_GEOM_THRESHOLD_NATS:
            verdict = "FALSIFY_H1_GEOMETRIC_COUPLING_SURVIVES"
        else:
            verdict = "weak-positive (CI excludes 0, |β| < 0.5-nat threshold)"
    elif is_on_axis_ci[1] < 0:
        verdict = "ANTI_LOCALIZED (off-axis > on-axis after distance adjustment)"
    else:
        verdict = "CONFIRM_H1_NO_MIDPOINT_COUPLING_AFTER_DISTANCE_ADJUSTMENT"

    return {
        "status": "OK",
        "n_rows": len(reg_rows),
        "n_rows_pre_drop": n_rows_pre_drop,
        "n_dropped_nonfinite": n_dropped_nonfinite,
        "n_clusters": len(set(clusters)),
        "design_matrix_condition_number": cond_number,
        "design_matrix_rank": rank,
        "design_matrix_n_columns": n_columns,
        "design_matrix_full_rank": full_rank,
        "formula": "gap_dosematched ~ is_on_axis + mean_d + asym  [cluster-robust at (pair, seed)]",
        "params": params,
        "pvalues": pvalues,
        "ci95": ci95,
        "headline_coefficient": "is_on_axis",
        "headline_beta": is_on_axis_beta,
        "headline_ci95": is_on_axis_ci,
        "headline_p": pvalues["is_on_axis"],
        "verdict_distance_adjusted": verdict,
        "rsquared": float(ols.rsquared),
        "interpretation": (
            "is_on_axis coefficient ESTIMATES the gap_dosematched contrast "
            "between on-axis and off-axis personas, controlling linearly for "
            "mean_d (distance to {A,B}) and asym (|d_A − d_B|). This is the "
            "PRIMARY Q2 readout; the raw subpanel-mean Δ_geom is a "
            "diagnostic-only companion because on-axis and off-axis subpanels "
            "are not always mean-d-matched."
        ),
    }


def compute_delta_geom_vs_match_delta(
    decomp_rows: list[dict],
    pairs: list[dict],
) -> dict:
    """Sensitivity: how does raw Δ_geom depend on the per-pair mean_d match delta?

    A coefficient near 0 would say the raw Δ_geom is robust to the
    mismatch; a strongly positive coefficient would say the raw Δ_geom is
    largely driven by the distance band gap.
    """
    pair_meta = {p["pair_id"]: p for p in pairs}
    by_tuple: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in decomp_rows:
        by_tuple[(r["pair_id"], r["seed"])][r["subpanel"]] = r

    pts = []
    for (pair_id, seed), pr in by_tuple.items():
        on = pr.get("on_axis")
        off = pr.get("off_axis")
        if on is None or off is None:
            continue
        delta_geom_mean = on["gap_dosematched_mean"] - off["gap_dosematched_mean"]
        match_delta = pair_meta.get(pair_id, {}).get("mean_d_match_delta_layer20")
        if match_delta is None:
            continue
        pts.append(
            {
                "pair_id": pair_id,
                "seed": seed,
                "delta_geom_mean": delta_geom_mean,
                "mean_d_match_delta": float(match_delta),
            }
        )

    if len(pts) < 3:
        return {"status": "SKIPPED", "reason": f"need ≥3 tuples; got {len(pts)}"}

    from scipy import stats

    xs = np.array([p["mean_d_match_delta"] for p in pts])
    ys = np.array([p["delta_geom_mean"] for p in pts])
    slope, intercept, r, p, se = stats.linregress(xs, ys)
    return {
        "status": "OK",
        "n": len(pts),
        "slope": float(slope),
        "intercept": float(intercept),
        "se": float(se),
        "p": float(p),
        "r_squared": float(r) ** 2,
        "points": pts,
        "interpretation": (
            "slope ≈ 0 means raw Δ_geom is robust to the per-pair distance "
            "match quality. slope significantly > 0 means the raw Δ_geom is "
            "largely a distance-band effect — defer to the distance-adjusted "
            "primary readout."
        ),
    }


def strict_distance_matched_subset(
    decomp_rows: list[dict],
    pairs: list[dict],
    threshold: float = STRICT_DIST_MATCH_THRESHOLD_NATS,
) -> dict:
    """Restrict the raw Δ_geom to pairs where mean_d_match_delta < threshold.

    For pairs where the off-axis subpanel actually IS distance-matched, raw
    Δ_geom is interpretable on its own.
    """
    strict_pair_ids = [
        p["pair_id"]
        for p in pairs
        if (p.get("mean_d_match_delta_layer20") or float("inf")) <= threshold
    ]

    by_tuple: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in decomp_rows:
        if r["pair_id"] not in strict_pair_ids:
            continue
        by_tuple[(r["pair_id"], r["seed"])][r["subpanel"]] = r

    delta_geom_vals: list[float] = []
    for pr in by_tuple.values():
        on = pr.get("on_axis")
        off = pr.get("off_axis")
        if on is None or off is None:
            continue
        delta_geom_vals.append(on["gap_dosematched_mean"] - off["gap_dosematched_mean"])

    return {
        "threshold_nats": threshold,
        "n_strict_pairs": len(strict_pair_ids),
        "strict_pair_ids": strict_pair_ids,
        "delta_geom_strict_bootstrap": paired_bootstrap(delta_geom_vals),
        "note": (
            f"Restricted to pairs with mean_d_match_delta ≤ {threshold} nat. "
            f"If n_strict_pairs is small (e.g. 0 or 1), the strict subset is "
            f"NOT a substitute for the distance-adjusted primary regression."
        ),
        "match_deltas_per_pair": {p["pair_id"]: p.get("mean_d_match_delta_layer20") for p in pairs},
    }


def compute_per_source_asymmetry(decomp_rows: list[dict]) -> dict:
    by_subpanel: dict[str, dict] = {"on_axis": {}, "off_axis": {}}
    for sub in ("on_axis", "off_axis"):
        sub_rows = [r for r in decomp_rows if r["subpanel"] == sub]
        if not sub_rows:
            continue
        A_vals = [r["pooled_2D_A"] for r in sub_rows]
        B_vals = [r["pooled_2D_B"] for r in sub_rows]
        diff_vals = [a - b for a, b in zip(A_vals, B_vals, strict=True)]
        by_subpanel[sub] = {
            "pooled_2D_A_mean": float(np.mean(A_vals)),
            "pooled_2D_B_mean": float(np.mean(B_vals)),
            "abs_diff_mean": float(np.mean([abs(d) for d in diff_vals])),
            "asymmetry_paired_boot": paired_bootstrap(diff_vals),
            "n": len(sub_rows),
        }
    return by_subpanel


def compute_pair_separation_regression(
    decomp_rows: list[dict], pairs: list[dict], names: list[str], distance: list[list[float]]
) -> dict:
    """Δ_geom ~ cos_dist(A, B) per combiner (per-seed + pooled)."""
    from scipy import stats

    out: dict = {"per_combiner": {}}
    by_tuple: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in decomp_rows:
        by_tuple[(r["pair_id"], r["seed"])][r["subpanel"]] = r

    pair_dist: dict[str, float] = {}
    for pair in pairs:
        try:
            i = names.index(pair["A"])
            j = names.index(pair["B"])
            pair_dist[pair["pair_id"]] = float(distance[i][j])
        except (KeyError, ValueError):
            pair_dist[pair["pair_id"]] = float("nan")

    for c in COMBINERS:
        xs_pooled, ys_pooled = [], []
        per_seed: dict[int, dict] = {}
        for seed in sorted({t[1] for t in by_tuple}):
            xs, ys = [], []
            for (pair_id, sd), pr in by_tuple.items():
                if sd != seed:
                    continue
                on = pr.get("on_axis")
                off = pr.get("off_axis")
                if on is None or off is None:
                    continue
                dgeom = on[f"gap_dosematched_{c}"] - off[f"gap_dosematched_{c}"]
                if np.isnan(dgeom):
                    continue
                d = pair_dist.get(pair_id, float("nan"))
                if np.isnan(d):
                    continue
                xs.append(d)
                ys.append(dgeom)
                xs_pooled.append(d)
                ys_pooled.append(dgeom)
            if len(xs) >= 3:
                slope, intercept, r, p, se = stats.linregress(xs, ys)
                per_seed[seed] = {
                    "n": len(xs),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "p": float(p),
                    "se": float(se),
                    "r_squared": float(r) ** 2,
                }
            else:
                per_seed[seed] = {"n": len(xs), "slope": None, "p": None}
        if len(xs_pooled) >= 3:
            slope, intercept, r, p, se = stats.linregress(xs_pooled, ys_pooled)
            pooled = {
                "n": len(xs_pooled),
                "slope": float(slope),
                "intercept": float(intercept),
                "p": float(p),
                "se": float(se),
                "r_squared": float(r) ** 2,
            }
        else:
            pooled = {"n": len(xs_pooled), "slope": None, "p": None}
        out["per_combiner"][c] = {"per_seed": per_seed, "pooled": pooled}
    return out


def write_tidy_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        log.warning("write_tidy_csv: empty rows; writing header-only CSV")
        with path.open("w") as f:
            f.write("# empty\n")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_490"),
    )
    parser.add_argument(
        "--source-pairs",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_490" / "source_pairs.json"),
    )
    parser.add_argument(
        "--cell-specs",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_490" / "cell_specs.json"),
    )
    parser.add_argument(
        "--aggregate-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_490" / "aggregate"),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow analysis to proceed even when some (cell, seed) results are "
        "missing. Outputs are stamped non_promotable=True. Default: FAIL on any "
        "missing (cell, seed) combo from cell_specs × seeds_resolved.",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    pairs_path = Path(args.source_pairs)
    specs_path = Path(args.cell_specs)
    aggregate_dir = Path(args.aggregate_dir)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    results = load_cell_results(eval_dir)
    pairs, pairs_payload = load_source_pairs(pairs_path)
    seeds_resolved = pairs_payload.get("seeds_resolved", [42, 137])
    log.info(
        "Pairs: %d, escalate_to_3_seeds=%s, seeds_resolved=%s",
        len(pairs),
        pairs_payload.get("escalate_to_3_seeds"),
        seeds_resolved,
    )

    # ── COMPLETENESS GATE ─────────────────────────────────────────────────
    specs = load_cell_specs(specs_path)
    completeness = completeness_check(results, specs, seeds_resolved)
    log.info(
        "Completeness: expected=%d actual=%d missing=%d extra=%d complete=%s",
        completeness["n_expected"],
        completeness["n_actual"],
        completeness["n_missing"],
        completeness["n_extra"],
        completeness["complete"],
    )
    non_promotable = False
    if not completeness["complete"]:
        if not args.allow_partial:
            raise SystemExit(
                f"Analyzer refuses to emit a headline: {completeness['n_missing']} of "
                f"{completeness['n_expected']} (cell, seed) results missing. Either "
                f"finish the sweep or pass --allow-partial (outputs will be stamped "
                f"non_promotable). First 5 missing: "
                f"{completeness['missing'][:5]!r}"
            )
        log.warning(
            "--allow-partial: proceeding with %d missing results; outputs marked non_promotable.",
            completeness["n_missing"],
        )
        non_promotable = True

    log.info("Loading distance matrix for pair-separation regression ...")
    names, distance = load_cosine_distance_matrix()

    # ── Build persona-level rows (foundation for all stats) ──────────────
    persona_rows = build_persona_level_rows(results, pairs)
    log.info("Persona-level rows: %d", len(persona_rows))
    write_tidy_csv(persona_rows, aggregate_dir / "persona_level.csv")

    # ── Primary: log P(※) DV ────────────────────────────────────────────
    log.info("Primary DV: deltaLogP_mean (on-policy log P(※) trained − base)")
    primary_decomp_rows = build_decomposition_rows(persona_rows, value_key="deltaLogP_mean")
    log.info("Decomposition rows (primary): %d", len(primary_decomp_rows))

    diagnostic = compute_per_combiner_diagnostic(primary_decomp_rows)
    asymmetry = compute_per_source_asymmetry(primary_decomp_rows)
    pair_sep = compute_pair_separation_regression(primary_decomp_rows, pairs, names, distance)

    # PRIMARY Q2 readout (round-2 fix).
    primary_regression = compute_distance_adjusted_primary(persona_rows, pairs)

    # Sensitivity / robustness.
    sensitivity_match_delta = compute_delta_geom_vs_match_delta(primary_decomp_rows, pairs)
    strict_subset = strict_distance_matched_subset(primary_decomp_rows, pairs)

    # ── Fallback DV: KL-from-base mirror ────────────────────────────────
    log.info("Fallback DV: kl_mean (full-vocab KL trained ‖ base)")
    fallback_decomp_rows = build_decomposition_rows(persona_rows, value_key="kl_mean")
    fallback_diagnostic = compute_per_combiner_diagnostic(fallback_decomp_rows)
    # For KL the absolute logp_trained/logp_base aren't meaningful inputs
    # to the LSE combiner (the DV isn't a logp). Mark the fallback LSE rows
    # explicitly when needed; the diagnostic shape stays the same.

    # ── Saturation diagnostic ───────────────────────────────────────────
    sat_per_condition: dict[str, list[float]] = defaultdict(list)
    for r in results:
        g = r.get("eval", {}).get("summary", {}).get("g_logprob_source")
        if g is None:
            continue
        sat_per_condition[r["condition"]].append(float(g))
    saturation = {
        cond: {
            "n_cells": len(vals),
            "mean_g_logprob_source": float(np.mean(vals)) if vals else None,
            "min_g_logprob_source": float(np.min(vals)) if vals else None,
            "max_g_logprob_source": float(np.max(vals)) if vals else None,
            "n_saturated_cells": int(sum(1 for v in vals if v > -0.1)),
            "n_near_saturated_cells": int(sum(1 for v in vals if -1.0 < v <= -0.1)),
        }
        for cond, vals in sorted(sat_per_condition.items())
    }
    n_saturated_pooled_2d = sum(
        s["n_saturated_cells"] for cond, s in saturation.items() if cond.startswith("pooled_2D_")
    )
    promote_fallback = n_saturated_pooled_2d >= 2

    # ── Write outputs ───────────────────────────────────────────────────
    write_tidy_csv(primary_decomp_rows, aggregate_dir / "tidy_primary.csv")
    write_tidy_csv(fallback_decomp_rows, aggregate_dir / "tidy_fallback_kl.csv")

    decomposition = {
        "experiment": "issue_490_dose_matched",
        "non_promotable": non_promotable,
        "completeness": completeness,
        "n_results": len(results),
        "n_pairs": len(pairs),
        "seeds_resolved": seeds_resolved,
        "escalate_to_3_seeds": pairs_payload.get("escalate_to_3_seeds"),
        "layer21_source": pairs_payload.get("layer21_source"),
        "primary_dv": "deltaLogP_mean (on-policy log P(※) trained − base)",
        "fallback_dv": "kl_mean (full-vocab KL trained ‖ base)",
        "primary_q2_readout": "distance_adjusted_regression",
        "primary": {
            # PRIMARY: distance-adjusted persona-level regression.
            "distance_adjusted_regression": primary_regression,
            # Diagnostic: raw subpanel-mean Δ_geom + components.
            "diagnostic_unadjusted_subpanel_means": diagnostic,
            # Sensitivity: how raw Δ_geom depends on pair-level match quality.
            "sensitivity_delta_geom_vs_match_delta": sensitivity_match_delta,
            # Restricted: pairs where strict distance match holds.
            "strict_distance_matched_subset": strict_subset,
            "asymmetry_pooled_A_vs_B": asymmetry,
            "pair_separation_regression": pair_sep,
        },
        "fallback": {
            "diagnostic_unadjusted_subpanel_means": fallback_diagnostic,
        },
        "saturation_per_condition": saturation,
        "promote_fallback_to_primary": promote_fallback,
        "promote_fallback_rationale": (
            f"≥2 POOLED-SINGLE-2D cells saturated at g_logprob_source > -0.1 "
            f"({n_saturated_pooled_2d} so far) → primary DV unreliable; fallback "
            f"KL-from-base is the headline."
            if promote_fallback
            else f"<2 POOLED-SINGLE-2D cells saturated ({n_saturated_pooled_2d}); "
            f"primary DV (log P(※)) is the headline."
        ),
        "slope_dose_caveat": (
            "slope_dose = ½[(POOLED_2D_A − SINGLE_D_A) + (POOLED_2D_B − SINGLE_D_B)] "
            "is a DOSE-PLUS-TRAINING-VOLUME effect, not pure dose. SINGLE-D cells "
            "have 200 positives + 200 negatives = 400 rows total; POOLED-SINGLE-2D "
            "cells have 400 + 400 = 800 rows. At the same effective batch size, "
            "POOLED cells see 2× the optimizer steps AND 2× the contrastive-negative "
            "exposure. Interpret slope_dose as 'going from D=200 to 2D=400 — with all "
            "the consequences that has for SFT updates', not 'isolated per-token "
            "dose effect'. A step-normalized sensitivity would require an "
            "additional control cell."
        ),
        "round_2_changes": {
            "primary_q2_now_distance_adjusted_regression": True,
            "raw_delta_geom_demoted_to_diagnostic": True,
            "lse_combiner_uses_absolute_logps": True,
            "completeness_gate_enabled": True,
            "layer21_fallback_honest": True,
        },
    }
    out_path = aggregate_dir / "decomposition.json"
    out_path.write_text(json.dumps(decomposition, indent=2))
    log.info("Wrote %s", out_path)

    regression = {
        "primary_q2_distance_adjusted_regression": primary_regression,
        "diagnostic_unadjusted_subpanel_means": diagnostic,
        "sensitivity_delta_geom_vs_match_delta": sensitivity_match_delta,
        "strict_distance_matched_subset": strict_subset,
        "pair_separation_regression": pair_sep,
        "fallback_diagnostic": fallback_diagnostic,
        "saturation": saturation,
        "completeness": completeness,
        "non_promotable": non_promotable,
    }
    (aggregate_dir / "regression.json").write_text(json.dumps(regression, indent=2))
    log.info("Wrote %s", aggregate_dir / "regression.json")

    if primary_regression.get("status") == "OK":
        log.info(
            "HEADLINE (distance-adjusted is_on_axis β): %.3f (CI95 [%.3f, %.3f], "
            "p=%.3f, n=%d, clusters=%d) → verdict=%s",
            primary_regression["headline_beta"],
            primary_regression["headline_ci95"][0],
            primary_regression["headline_ci95"][1],
            primary_regression["headline_p"],
            primary_regression["n_rows"],
            primary_regression["n_clusters"],
            primary_regression["verdict_distance_adjusted"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
