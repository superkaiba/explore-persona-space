#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #490 PHASE 4 — primary geometric headline + dose decomposition.

Per plan v1 §4.5 PHASE 4 + §6.2 (with v2 revision §1-2):

Pipeline:
  1. Load all eval_results/issue_490/cell_*_seed*/result.json.
  2. Load data/issue_490/source_pairs.json — per-pair on-axis/off-axis
     subpanels (from Phase 0).
  3. For each (pair, seed), slice each cell's per-persona reads into
     on-axis intermediate-C and off-axis distance-matched subpanels.
  4. Compute, per (pair × seed × subpanel) and per combiner (mean / lse /
     max):
       - gap_confounded   = SHARED_2D − combiner(SINGLE_D_A, SINGLE_D_B)
                            (reproduces #478's ambiguous finding)
       - gap_dosematched  = SHARED_2D − combiner(POOLED_2D_A, POOLED_2D_B)
                            (Q1 — dose-confound resolution)
       - slope_dose       = ½[(POOLED_2D_A − SINGLE_D_A) + (POOLED_2D_B − SINGLE_D_B)]
                            (pure dose-response leg)
       - Δ_geom           = gap_dosematched(on-axis) − gap_dosematched(off-axis)
                            (Q2 / GEOMETRIC HEADLINE)
  5. Paired bootstrap (10000 resamples) CI on Δ_geom + gap_dosematched +
     slope_dose across the (pair × seed) tuples, per combiner.
  6. Per-source asymmetry (POOLED-A vs POOLED-B leg separately).
  7. Pair-separation regression Δ_geom ~ cos_dist(A, B).
  8. Conditional-reading flag: a CONFIRM verdict on H1 is only meaningful
     if slope_dose is clearly positive.
  9. Fallback DV (full-vocab KL-from-base at post-response slot) mirror.
 10. Write eval_results/issue_490/aggregate/{decomposition.json, tidy.csv,
     regression.json}.
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
    load_cosine_distance_matrix,
)


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


def load_source_pairs(pairs_path: Path) -> list[dict]:
    return json.loads(pairs_path.read_text())["pairs"]


def _per_persona_value(cell_result: dict, persona: str, value_key: str) -> float | None:
    """Read a per-persona scalar from a cell's eval block.

    value_key ∈ {"deltaLogP_mean", "kl_mean", "emit_rate", "logp_trained_mean"}.
    """
    held_out = cell_result.get("eval", {}).get("held_out", {})
    payload = held_out.get(persona)
    if payload is None:
        return None
    val = payload.get(value_key)
    if val is None:
        return None
    return float(val)


def _mean_over_subpanel(cell_result: dict, personas: list[str], value_key: str) -> float | None:
    """Mean of value_key over personas; None if any persona is missing."""
    vals = []
    for p in personas:
        v = _per_persona_value(cell_result, p, value_key)
        if v is None:
            return None
        vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals))


def build_decomposition_rows(
    results: list[dict],
    pairs: list[dict],
    value_key: str = "deltaLogP_mean",
) -> list[dict]:
    """For each (pair, seed, subpanel) build a row carrying SHARED_2D /
    POOLED_2D_A / POOLED_2D_B / SINGLE_D_A / SINGLE_D_B subpanel means,
    plus the derived gap_confounded / gap_dosematched / slope_dose under
    each combiner.
    """
    # Index cells: (pair_id, condition, seed) → result
    by_key: dict[tuple[str, str, int], dict] = {}
    for r in results:
        key = (r["pair_id"], r["condition"], r["seed"])
        by_key[key] = r

    rows: list[dict] = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        on_axis = pair["on_axis"]
        off_axis = pair["off_axis"]
        # Seeds: find the set of seeds present for this pair (any condition).
        seeds_for_pair = sorted({k[2] for k in by_key if k[0] == pair_id})
        for seed in seeds_for_pair:
            for subpanel_name, personas in (("on_axis", on_axis), ("off_axis", off_axis)):
                # Pull each condition's subpanel mean.
                cell_means: dict[str, float | None] = {}
                for cond in (
                    CONDITION_SHARED_2D,
                    CONDITION_POOLED_2D_A,
                    CONDITION_POOLED_2D_B,
                    CONDITION_SINGLE_D_A,
                    CONDITION_SINGLE_D_B,
                ):
                    r = by_key.get((pair_id, cond, seed))
                    if r is None:
                        cell_means[cond] = None
                        continue
                    cell_means[cond] = _mean_over_subpanel(r, personas, value_key)

                if any(v is None for v in cell_means.values()):
                    log.warning(
                        "Skipping pair=%s seed=%d subpanel=%s (missing condition data: %s)",
                        pair_id,
                        seed,
                        subpanel_name,
                        {k: v for k, v in cell_means.items() if v is None},
                    )
                    continue

                # Derived per-combiner statistics.
                shared = cell_means[CONDITION_SHARED_2D]
                pooled_A = cell_means[CONDITION_POOLED_2D_A]
                pooled_B = cell_means[CONDITION_POOLED_2D_B]
                single_A = cell_means[CONDITION_SINGLE_D_A]
                single_B = cell_means[CONDITION_SINGLE_D_B]
                slope = 0.5 * ((pooled_A - single_A) + (pooled_B - single_B))
                row = {
                    "pair_id": pair_id,
                    "A": pair["A"],
                    "B": pair["B"],
                    "seed": seed,
                    "subpanel": subpanel_name,
                    "n_personas": len(personas),
                    "value_key": value_key,
                    "shared_2D": shared,
                    "pooled_2D_A": pooled_A,
                    "pooled_2D_B": pooled_B,
                    "single_D_A": single_A,
                    "single_D_B": single_B,
                    "slope_dose": slope,
                }
                for c in COMBINERS:
                    pooled_combined = combiner(c, [pooled_A, pooled_B])
                    single_combined = combiner(c, [single_A, single_B])
                    row[f"gap_confounded_{c}"] = shared - single_combined
                    row[f"gap_dosematched_{c}"] = shared - pooled_combined
                rows.append(row)
    return rows


def paired_bootstrap(values: list[float], n_boots: int = 10000, rng_seed: int = 490) -> dict:
    """Paired bootstrap mean + 95% CI."""
    if not values:
        return {"n": 0, "mean": None, "ci95": (None, None), "p_one_sided_pos": None}
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(rng_seed)
    n = len(arr)
    means = np.empty(n_boots, dtype=float)
    for i in range(n_boots):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    # One-sided p (Pr(mean ≤ 0 | true mean = observed)) — proportion of boot
    # samples ≤ 0. Useful for direction-of-effect framing.
    p_one_sided_pos = float((means <= 0).mean())
    return {
        "n": int(n),
        "mean": float(arr.mean()),
        "ci95": (float(lo), float(hi)),
        "p_one_sided_pos": p_one_sided_pos,
    }


def compute_per_combiner_headlines(rows: list[dict]) -> dict:
    """For each combiner, compute Δ_geom + gap_dosematched(on/off) + slope_dose
    paired-bootstrap summaries across (pair × seed) tuples.
    """
    # Pivot rows into per-(pair, seed) {on_axis: row, off_axis: row}
    by_tuple: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_tuple[(r["pair_id"], r["seed"])][r["subpanel"]] = r

    out: dict = {"per_combiner": {}, "n_tuples_with_both_subpanels": 0}
    for c in COMBINERS:
        delta_geom_vals = []
        gap_dose_on_vals = []
        gap_dose_off_vals = []
        gap_conf_on_vals = []
        slope_vals = []
        complete_tuples = 0
        for (_pair, _seed), pair_rows in by_tuple.items():
            on = pair_rows.get("on_axis")
            off = pair_rows.get("off_axis")
            if on is None or off is None:
                continue
            complete_tuples += 1
            dgeom = on[f"gap_dosematched_{c}"] - off[f"gap_dosematched_{c}"]
            delta_geom_vals.append(dgeom)
            gap_dose_on_vals.append(on[f"gap_dosematched_{c}"])
            gap_dose_off_vals.append(off[f"gap_dosematched_{c}"])
            gap_conf_on_vals.append(on[f"gap_confounded_{c}"])
            slope_vals.append(0.5 * (on["slope_dose"] + off["slope_dose"]))
        out["per_combiner"][c] = {
            "delta_geom": paired_bootstrap(delta_geom_vals),
            "gap_dosematched_on_axis": paired_bootstrap(gap_dose_on_vals),
            "gap_dosematched_off_axis": paired_bootstrap(gap_dose_off_vals),
            "gap_confounded_on_axis": paired_bootstrap(gap_conf_on_vals),
            "slope_dose": paired_bootstrap(slope_vals),
        }
        out["n_tuples_with_both_subpanels"] = complete_tuples
    return out


def compute_per_source_asymmetry(rows: list[dict]) -> dict:
    """Report POOLED-A vs POOLED-B leakage at each subpanel separately.

    If one source dominates, the mean combiner is flagged and the max combiner
    is the right read.
    """
    by_subpanel: dict[str, dict] = {"on_axis": {}, "off_axis": {}}
    for sub in ("on_axis", "off_axis"):
        sub_rows = [r for r in rows if r["subpanel"] == sub]
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
    rows: list[dict], pairs: list[dict], names: list[str], distance: list[list[float]]
) -> dict:
    """Fit Δ_geom ~ cos_dist(A, B) per combiner (per-seed + pooled)."""
    from scipy import stats

    out: dict = {"per_combiner": {}}
    # Build (pair, seed) → delta_geom map per combiner.
    by_tuple: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in rows:
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
        per_seed: dict[int, dict] = {}
        for seed in sorted({t[1] for t in by_tuple}):
            xs = []
            ys = []
            for (pair_id, sd), pair_rows in by_tuple.items():
                if sd != seed:
                    continue
                on = pair_rows.get("on_axis")
                off = pair_rows.get("off_axis")
                if on is None or off is None:
                    continue
                dgeom = on[f"gap_dosematched_{c}"] - off[f"gap_dosematched_{c}"]
                d = pair_dist.get(pair_id, float("nan"))
                if np.isnan(d):
                    continue
                xs.append(d)
                ys.append(dgeom)
            if len(xs) < 3:
                per_seed[seed] = {"n": len(xs), "slope": None, "p": None}
                continue
            slope, intercept, r, p, se = stats.linregress(xs, ys)
            per_seed[seed] = {
                "n": len(xs),
                "slope": float(slope),
                "intercept": float(intercept),
                "p": float(p),
                "se": float(se),
                "r_squared": float(r) ** 2,
            }
        # Pooled across seeds.
        xs = []
        ys = []
        for (pair_id, _sd), pair_rows in by_tuple.items():
            on = pair_rows.get("on_axis")
            off = pair_rows.get("off_axis")
            if on is None or off is None:
                continue
            dgeom = on[f"gap_dosematched_{c}"] - off[f"gap_dosematched_{c}"]
            d = pair_dist.get(pair_id, float("nan"))
            if np.isnan(d):
                continue
            xs.append(d)
            ys.append(dgeom)
        if len(xs) >= 3:
            slope, intercept, r, p, se = stats.linregress(xs, ys)
            pooled = {
                "n": len(xs),
                "slope": float(slope),
                "intercept": float(intercept),
                "p": float(p),
                "se": float(se),
                "r_squared": float(r) ** 2,
            }
        else:
            pooled = {"n": len(xs), "slope": None, "p": None}
        out["per_combiner"][c] = {"per_seed": per_seed, "pooled": pooled}
    return out


def conditional_reading_flag(headlines: dict) -> dict:
    """Per plan §3 H2 / §6.2: Δ_geom CONFIRM-H1 is only a strong 'coupling
    falsified' verdict if slope_dose is clearly positive. Else 'dose-response
    leg didn't fire — cannot decompose'.
    """
    flag: dict = {"per_combiner": {}}
    for c, h in headlines.get("per_combiner", {}).items():
        dgeom = h["delta_geom"]
        slope = h["slope_dose"]
        if dgeom["n"] == 0:
            verdict = "no-data"
        else:
            dgeom_ci = dgeom["ci95"]
            slope_ci = slope["ci95"]
            slope_clear_positive = (
                slope["mean"] is not None and slope_ci[0] is not None and slope_ci[0] > 0
            )
            dgeom_excludes_zero_positive = dgeom_ci[0] is not None and dgeom_ci[0] > 0
            dgeom_excludes_zero_negative = dgeom_ci[1] is not None and dgeom_ci[1] < 0
            if dgeom_excludes_zero_positive:
                if dgeom["mean"] >= POWER_DELTA_GEOM_THRESHOLD_NATS:
                    verdict = "FALSIFY_H1_GEOMETRIC_COUPLING_SURVIVES"
                else:
                    verdict = "weak-positive (CI excludes 0, |Δ| < 0.5 nat threshold)"
            elif dgeom_excludes_zero_negative:
                verdict = "ANTI_LOCALIZED (off-axis > on-axis)"
            elif slope_clear_positive:
                verdict = "CONFIRM_H1_NO_MIDPOINT_COUPLING"
            else:
                verdict = (
                    "AMBIGUOUS: dose-response slope is not clearly positive, "
                    "so a null Δ_geom cannot decompose dose vs coupling"
                )
        flag["per_combiner"][c] = {
            "verdict": verdict,
            "delta_geom_mean": dgeom.get("mean"),
            "delta_geom_ci95": dgeom.get("ci95"),
            "slope_dose_mean": slope.get("mean"),
            "slope_dose_ci95": slope.get("ci95"),
        }
    return flag


def write_tidy_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        log.warning("write_tidy_csv: empty rows; writing header-only CSV")
        with path.open("w") as f:
            f.write("# empty\n")
        return
    keys = list(rows[0].keys())
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
        "--aggregate-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_490" / "aggregate"),
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    pairs_path = Path(args.source_pairs)
    aggregate_dir = Path(args.aggregate_dir)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    results = load_cell_results(eval_dir)
    pairs = load_source_pairs(pairs_path)
    log.info("Pairs: %d", len(pairs))

    # Distance matrix for the secondary pair-separation regression.
    log.info("Loading distance matrix for pair-separation regression ...")
    names, distance = load_cosine_distance_matrix()

    # ── Primary: log P(※) DV ────────────────────────────────────────────
    log.info("Primary DV: deltaLogP_mean (on-policy log P(※) trained − base)")
    primary_rows = build_decomposition_rows(results, pairs, value_key="deltaLogP_mean")
    log.info("Decomposition rows (primary): %d", len(primary_rows))

    headlines = compute_per_combiner_headlines(primary_rows)
    asymmetry = compute_per_source_asymmetry(primary_rows)
    pair_sep = compute_pair_separation_regression(primary_rows, pairs, names, distance)
    verdict = conditional_reading_flag(headlines)

    # ── Fallback DV: KL-from-base mirror ────────────────────────────────
    log.info("Fallback DV: kl_mean (full-vocab KL trained ‖ base)")
    fallback_rows = build_decomposition_rows(results, pairs, value_key="kl_mean")
    fallback_headlines = compute_per_combiner_headlines(fallback_rows)
    fallback_verdict = conditional_reading_flag(fallback_headlines)

    # ── Saturation diagnostic (per-cell trained-source log P(※)) ────────
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
    write_tidy_csv(primary_rows, aggregate_dir / "tidy_primary.csv")
    write_tidy_csv(fallback_rows, aggregate_dir / "tidy_fallback_kl.csv")

    decomposition = {
        "experiment": "issue_490_dose_matched",
        "n_results": len(results),
        "n_pairs": len(pairs),
        "n_decomp_rows_primary": len(primary_rows),
        "n_decomp_rows_fallback": len(fallback_rows),
        "primary_dv": "deltaLogP_mean (on-policy log P(※) trained − base)",
        "fallback_dv": "kl_mean (full-vocab KL trained ‖ base)",
        "primary": {
            "headlines": headlines,
            "verdict_per_combiner": verdict,
            "asymmetry_pooled_A_vs_B": asymmetry,
            "pair_separation_regression": pair_sep,
        },
        "fallback": {
            "headlines": fallback_headlines,
            "verdict_per_combiner": fallback_verdict,
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
    }
    out_path = aggregate_dir / "decomposition.json"
    out_path.write_text(json.dumps(decomposition, indent=2))
    log.info("Wrote %s", out_path)

    # Lean regression summary (subset of decomposition).
    regression = {
        "primary_headlines": headlines,
        "primary_verdict": verdict,
        "primary_pair_separation": pair_sep,
        "fallback_headlines": fallback_headlines,
        "saturation": saturation,
    }
    (aggregate_dir / "regression.json").write_text(json.dumps(regression, indent=2))
    log.info("Wrote %s", aggregate_dir / "regression.json")

    # Headline log line.
    primary_mean_dg = headlines["per_combiner"]["mean"]["delta_geom"]
    log.info(
        "HEADLINE (mean combiner): Δ_geom = %s (CI95 %s), n=%d tuples, verdict=%s",
        f"{primary_mean_dg['mean']:.3f}" if primary_mean_dg["mean"] is not None else "n/a",
        primary_mean_dg["ci95"],
        primary_mean_dg["n"],
        verdict["per_combiner"]["mean"]["verdict"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
