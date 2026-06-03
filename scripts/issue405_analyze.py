#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #405 PHASE 5 — headline regression + robustness + figures.

Per plan v2 §4.5 PHASE 5 + §6.4 (the primary statistical test) + §6.5 (the
hero figure + exploratory dump).

Pipeline:
  1. Load all ``eval_results/issue_405/cell_*_seed*/result.json`` files.
  2. Separate by ``track`` (CORE / K4_ABLNEG / K1_DOSE50).
  3. For every (cell, seed, held_out_persona) compute ``min_dist`` and
     ``mean_dist`` (layer-20 cosine).
  4. CORE-only headline mixed-effects regression
       ΔlogP ~ K * min_dist + (1|subset) + (1|persona)
     via pymer4 (primary) / rpy2+lme4 (fallback) / statsmodels.MixedLM
     with vc_formula (last resort).
  5. Robustness: comedian-dropped refit, leave-one-persona-out, leverage,
     covariate-adjusted (+ trained_pos_mean_dlogp), JS-distance refit.
  6. Plots: hero + raw + residualized + per-cell bars + per-seed scatter +
     min-vs-mean + JS robustness + ABLNEG overlay + dose-control head-to-head
     + comedian-dropped panel + trained-positive ΔlogP × K.

Writes:
  eval_results/issue_405/aggregate/regression.json
  figures/issue_405/*.png

CLI:
  --eval-dir          Default: eval_results/issue_405
  --fig-dir           Default: figures/issue_405
  --simulated         If true, fits on simulated 168-cell-persona data
                      (pre-sweep runnability gate per Fix B3).
  --simulated-only    Skip real data load; only run the simulated fit.
  --skip-plots        Stat fits only; no figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue405_common import (  # noqa: E402
    HELD_OUT,
    POOL,
    load_cosine_distance_matrix,
)


def load_cell_results(eval_dir: Path) -> list[dict]:
    """Read every cell_*_seed*/result.json under eval_dir."""
    files = sorted(eval_dir.glob("cell_*_seed*/result.json"))
    if not files:
        raise SystemExit(f"No cell result.json files found under {eval_dir}")
    out = []
    for f in files:
        try:
            out.append(json.loads(f.read_text()))
        except Exception as e:
            log.warning("Skipping malformed %s (%s)", f, e)
    log.info("Loaded %d cell result.json files from %s", len(out), eval_dir)
    return out


# Expected per-track counts per plan §0 + §4.6 (21 CORE × 2 seeds = 42,
# 1 ABLNEG × 2 seeds = 2, 3 DOSE50 × 2 seeds = 6; total 50).
EXPECTED_TRACK_COUNTS: dict[str, int] = {"CORE": 42, "K4_ABLNEG": 2, "K1_DOSE50": 6}
EXPECTED_SEEDS: tuple[int, ...] = (42, 137)


def expected_cell_seeds_from_specs(
    specs_path: Path,
) -> dict[tuple[str, int], str]:
    """Read cell_specs.json and return {(cell_id, seed): track} for every expected run."""
    specs = json.loads(specs_path.read_text())
    out: dict[tuple[str, int], str] = {}
    for s in specs:
        for seed in EXPECTED_SEEDS:
            out[(s["cell_id"], int(seed))] = s["track"]
    return out


def audit_track_counts(
    results: list[dict],
    specs_path: Path | None = None,
) -> dict:
    """Round-3 fix: EXACT-equality per-track audit (shortfall AND overage AND unknown).

    Returns a dict with:
      - ``observed``: {track: count}  (only known tracks)
      - ``expected``: {track: count}  (EXPECTED_TRACK_COUNTS)
      - ``shortfall``: {track: missing_count}  (observed < expected)
      - ``overage``:   {track: excess_count}   (observed > expected)
      - ``unknown_tracks``: list of unrecognized ``track`` values from results
      - ``missing_cell_seeds``: list of [cell_id, seed, track] still expected
      - ``extra_cell_seeds``:   list of [cell_id, seed, track] beyond expected

    The shortfall/overage dicts use POSITIVE counts in both directions for
    legibility. The cell_seed lists are populated when ``specs_path`` is
    provided (Phase-0.5 ``cell_specs.json``). Without specs, the function
    falls back to shortfall/overage based on track totals only — exact-
    equality is still enforced by the caller via ``has_mismatch``.
    """
    counts_observed: dict[str, int] = {t: 0 for t in EXPECTED_TRACK_COUNTS}
    unknown_tracks: list[str] = []
    observed_cell_seeds: dict[tuple[str, int], str] = {}
    for r in results:
        t = r.get("track")
        cid = r.get("cell_id")
        seed = r.get("seed")
        if t not in EXPECTED_TRACK_COUNTS:
            unknown_tracks.append(t)
            continue
        counts_observed[t] += 1
        if cid is not None and seed is not None:
            observed_cell_seeds[(str(cid), int(seed))] = t
    shortfall = {
        t: EXPECTED_TRACK_COUNTS[t] - counts_observed[t]
        for t in EXPECTED_TRACK_COUNTS
        if counts_observed[t] < EXPECTED_TRACK_COUNTS[t]
    }
    overage = {
        t: counts_observed[t] - EXPECTED_TRACK_COUNTS[t]
        for t in EXPECTED_TRACK_COUNTS
        if counts_observed[t] > EXPECTED_TRACK_COUNTS[t]
    }
    missing_cell_seeds: list[list] = []
    extra_cell_seeds: list[list] = []
    if specs_path is not None and specs_path.exists():
        expected_set = expected_cell_seeds_from_specs(specs_path)
        for (cid, seed), tr in sorted(expected_set.items()):
            if (cid, seed) not in observed_cell_seeds:
                missing_cell_seeds.append([cid, int(seed), tr])
        for (cid, seed), tr in sorted(observed_cell_seeds.items()):
            if (cid, seed) not in expected_set:
                extra_cell_seeds.append([cid, int(seed), tr])
    return {
        "observed": counts_observed,
        "expected": dict(EXPECTED_TRACK_COUNTS),
        "shortfall": shortfall,
        "overage": overage,
        "unknown_tracks": unknown_tracks,
        "missing_cell_seeds": missing_cell_seeds,
        "extra_cell_seeds": extra_cell_seeds,
    }


def build_cell_persona_frame(
    results: list[dict],
    names: list[str],
    distance: list[list[float]],
    track: str = "CORE",
) -> list[dict]:
    """Flatten cell results to a one-row-per-(cell, seed, held-out-persona) frame.

    Each row carries: cell_id, seed, K, subset (tuple-of-positives as str),
    persona, min_dist, mean_dist, deltaLogP_mean (across the 20 questions),
    trained_pos_mean_dlogp (per-cell scalar from FIX A1 panel).
    """
    rows: list[dict] = []
    for r in results:
        if r.get("track") != track:
            continue
        cell_id = r["cell_id"]
        seed = r["seed"]
        K = r["K"]
        positives = r["spec"]["positives"]
        subset_id = "+".join(sorted(positives))
        trained_pos_mean = r["eval"]["summary"]["trained_pos_mean_dlogp"]
        for persona, payload in r["eval"]["held_out"].items():
            # Blocker 6: FAIL LOUD on missing distance — a held-out persona
            # absent from the cosine matrix is a data bug, not a row to skip.
            if persona not in names:
                raise RuntimeError(
                    f"Held-out persona {persona!r} missing from layer-20 cosine matrix "
                    f"(cell={cell_id} seed={seed}). Names available: {names!r}. "
                    f"Refusing to silently drop the row — re-extract the matrix."
                )
            from _issue405_common import mean_dist_to_set, min_dist_to_set

            md = min_dist_to_set(persona, positives, names, distance)
            mn = mean_dist_to_set(persona, positives, names, distance)
            rows.append(
                {
                    "cell_id": cell_id,
                    "seed": seed,
                    "K": K,
                    "subset": subset_id,
                    "persona": persona,
                    "min_dist": md,
                    "mean_dist": mn,
                    "deltaLogP_mean": payload["deltaLogP_mean"],
                    "trained_pos_mean_dlogp": trained_pos_mean,
                    "emit_rate": payload["emit_rate"],
                }
            )
    return rows


def fit_mixed_effects(
    df_rows: list[dict],
    formula: str = "deltaLogP_mean ~ K * min_dist",
    re_groups: tuple[str, str] = ("subset", "persona"),
) -> dict:
    """Crossed-RE fit.

    Try pymer4 first; fall back to rpy2 + lme4; last resort statsmodels
    MixedLM with explicit vc_formula.

    Returns:
        dict with ``status``, ``tool``, ``coefs`` (per-term β + 95% CI),
        ``loglik``, ``aic``, ``message`` (str if failure).
    """
    n = len(df_rows)
    if n == 0:
        return {"status": "FAIL", "tool": "none", "message": "empty data frame"}

    import pandas as pd

    df = pd.DataFrame(df_rows)
    if df["persona"].nunique() < 2 or df["subset"].nunique() < 2:
        return {
            "status": "FAIL",
            "tool": "skipped",
            "message": (
                f"Need ≥ 2 unique persona AND ≥ 2 unique subset to fit crossed REs; "
                f"got {df['persona'].nunique()} personas, {df['subset'].nunique()} subsets."
            ),
        }

    # ── Tool 1: pymer4 ────────────────────────────────────────────────
    try:
        from pymer4.models import Lmer  # type: ignore

        lme_formula = f"{formula} + (1|{re_groups[0]}) + (1|{re_groups[1]})"
        model = Lmer(lme_formula, data=df)
        model.fit()
        coefs = model.coefs.to_dict()
        return {
            "status": "PASS",
            "tool": "pymer4",
            "formula": lme_formula,
            "n_obs": n,
            "coefs": coefs,
            "aic": float(model.AIC) if hasattr(model, "AIC") else None,
            "loglik": float(model.logLike) if hasattr(model, "logLike") else None,
        }
    except ImportError:
        log.warning("pymer4 not installed; trying rpy2+lme4 fallback")
    except Exception as e:
        log.warning("pymer4 fit failed (%s); trying rpy2+lme4 fallback", e)

    # ── Tool 2: rpy2 + lme4 ───────────────────────────────────────────
    try:
        import rpy2.robjects as ro  # type: ignore
        from rpy2.robjects import pandas2ri  # type: ignore
        from rpy2.robjects.packages import importr  # type: ignore

        pandas2ri.activate()
        lme4 = importr("lme4")
        lme_formula = f"{formula} + (1|{re_groups[0]}) + (1|{re_groups[1]})"
        with ro.default_converter + pandas2ri.converter:
            r_df = ro.conversion.py2rpy(df)
        ro.globalenv["df_r"] = r_df
        model = lme4.lmer(ro.Formula(lme_formula), data=r_df)
        ro.globalenv["m"] = model
        coef_df = ro.r("as.data.frame(summary(m)$coefficients)")
        coefs = {
            row_name: dict(coef_df.iloc[i].to_dict()) for i, row_name in enumerate(coef_df.index)
        }
        # Pull AIC from R for Blocker 5 min-vs-mean comparison.
        try:
            aic = float(ro.r("AIC(m)")[0])
        except Exception:
            aic = None
        return {
            "status": "PASS",
            "tool": "rpy2+lme4",
            "formula": lme_formula,
            "n_obs": n,
            "coefs": coefs,
            "aic": aic,
        }
    except ImportError:
        log.warning("rpy2 not installed; falling back to statsmodels.MixedLM")
    except Exception as e:
        log.warning("rpy2+lme4 fit failed (%s); falling back to statsmodels.MixedLM", e)

    # ── Tool 3: statsmodels MixedLM with vc_formula ───────────────────
    try:
        import statsmodels.formula.api as smf

        df["dummy_const"] = 1
        vc = {re_groups[0]: f"0 + C({re_groups[0]})", re_groups[1]: f"0 + C({re_groups[1]})"}
        model = smf.mixedlm(formula, df, groups="dummy_const", vc_formula=vc)
        fit = model.fit(method=["lbfgs"], reml=True)
        coefs = {
            term: {
                "Estimate": float(fit.params[term]),
                "Std. Error": float(fit.bse[term]),
                "P-val": float(fit.pvalues[term]),
            }
            for term in fit.params.index
            if term in fit.bse.index
        }
        # Blocker 5 fix: emit AIC from statsmodels so min-vs-mean
        # comparison works in the fallback path too (was missing in
        # round 1 — broke the AIC comparison the plan §6.4 mandates).
        try:
            aic = float(fit.aic) if hasattr(fit, "aic") and fit.aic is not None else None
        except Exception:
            aic = None
        try:
            llf = float(fit.llf) if hasattr(fit, "llf") and fit.llf is not None else None
        except Exception:
            llf = None
        return {
            "status": "PASS",
            "tool": "statsmodels.MixedLM (vc_formula)",
            "formula": formula,
            "n_obs": n,
            "coefs": coefs,
            "aic": aic,
            "loglik": llf,
        }
    except Exception as e:
        log.error("statsmodels.MixedLM fit failed too (%s)", e)
        return {"status": "FAIL", "tool": "all_failed", "message": str(e)}


def fit_per_K_marginal_slopes(df_rows: list[dict], dist_col: str = "min_dist") -> dict:
    """Blocker 5 — per-K marginal slopes (4 slopes, one per K ∈ {1, 2, 4, 8}).

    Per plan §6.4 FIX B2 mandatory robustness #3. Each K's marginal slope
    is a simple OLS within that K stratum (no random effects since we're
    conditioning on K). Reports {β, SE, p, 95% CI} per K.
    """
    if not df_rows:
        return {"status": "FAIL", "message": "empty data frame"}
    import statsmodels.formula.api as smf

    out: dict = {"per_K": {}}
    for K in sorted({r["K"] for r in df_rows}):
        sub = [r for r in df_rows if r["K"] == K]
        if len({r["persona"] for r in sub}) < 2:
            out["per_K"][K] = {
                "status": "SKIP",
                "n": len(sub),
                "message": "fewer than 2 unique personas at this K",
            }
            continue
        import pandas as pd

        d = pd.DataFrame(sub)
        try:
            fit = smf.ols(f"deltaLogP_mean ~ {dist_col}", data=d).fit()
            ci = fit.conf_int(alpha=0.05).loc[dist_col].tolist()
            out["per_K"][K] = {
                "status": "PASS",
                "n": len(sub),
                "beta": float(fit.params[dist_col]),
                "se": float(fit.bse[dist_col]),
                "p": float(fit.pvalues[dist_col]),
                "ci_95": [float(ci[0]), float(ci[1])],
            }
        except Exception as e:
            out["per_K"][K] = {"status": "FAIL", "n": len(sub), "message": str(e)}
    return out


def leverage_diagnostics(df_rows: list[dict], dist_col: str = "min_dist") -> dict:
    """Blocker 5 — Cook's distance + DFBETAS on β_{K×dist} per cell-persona row.

    Per plan §6.4 FIX B2 mandatory robustness #4. Uses a simple OLS
    backbone (`deltaLogP_mean ~ K * dist`) so the leverage numbers are
    interpretable; the mixed-effects fit's primary slope is reported
    separately by ``fit_mixed_effects``. Reports the top 5 highest-Cook's
    + the top 5 highest-|DFBETAS| on the interaction term.
    """
    if not df_rows:
        return {"status": "FAIL", "message": "empty data frame"}
    import pandas as pd
    import statsmodels.formula.api as smf

    d = pd.DataFrame(df_rows).reset_index(drop=True)
    try:
        fit = smf.ols(f"deltaLogP_mean ~ K * {dist_col}", data=d).fit()
        infl = fit.get_influence()
        cooks_d, _ = infl.cooks_distance
        try:
            dfb = infl.dfbetas  # shape (N, k) — columns are predictors
            cols = list(fit.params.index)
            interaction_col_name = f"K:{dist_col}"
            if interaction_col_name in cols:
                col_idx = cols.index(interaction_col_name)
                dfb_inter = dfb[:, col_idx]
            else:
                dfb_inter = [None] * len(d)
        except Exception:
            dfb_inter = [None] * len(d)

        def _row_meta(i: int) -> dict:
            r = d.iloc[i].to_dict()
            return {
                "cell_id": r.get("cell_id"),
                "seed": int(r.get("seed", 0)),
                "persona": r.get("persona"),
            }

        cooks_sorted_idx = sorted(range(len(cooks_d)), key=lambda i: -cooks_d[i])[:5]
        top_cooks = [{**_row_meta(i), "cooks_d": float(cooks_d[i])} for i in cooks_sorted_idx]
        dfb_with_idx = [
            (i, float(abs(x)) if x is not None else 0.0) for i, x in enumerate(dfb_inter)
        ]
        dfb_sorted_idx = sorted(dfb_with_idx, key=lambda t: -t[1])[:5]
        top_dfbetas = [
            {
                **_row_meta(i),
                "dfbetas_K_x_dist_abs": float(abs_v),
                "dfbetas_K_x_dist": (float(dfb_inter[i]) if dfb_inter[i] is not None else None),
            }
            for i, abs_v in dfb_sorted_idx
        ]
        return {
            "status": "PASS",
            "n_obs": len(d),
            "interaction_term": interaction_col_name,
            "top5_cooks_distance": top_cooks,
            "top5_abs_dfbetas_on_interaction": top_dfbetas,
        }
    except Exception as e:
        return {"status": "FAIL", "message": str(e)}


def cv_r2_at_subset_level(df_rows: list[dict], formula: str, k_folds: int = 5) -> dict | None:
    """Blocker 5 — subset-level K-fold CV R² for the min-vs-mean comparison.

    Folds at the subset level (NOT row level) so in-cell leakage doesn't
    inflate R². Falls back to a plain OLS scoring backbone (the mixed-
    effects fit's CV is too expensive at this N and the OLS R² is a
    monotonic proxy here since the random-effects variance is small).
    Returns dict with ``mean_r2`` + ``per_fold_r2`` + ``k_folds``, or
    None if not enough subsets to fold.
    """
    if not df_rows:
        return None
    import pandas as pd
    import statsmodels.formula.api as smf

    d = pd.DataFrame(df_rows)
    subsets = sorted(d["subset"].unique().tolist())
    if len(subsets) < k_folds:
        return {
            "status": "SKIP",
            "message": f"only {len(subsets)} unique subsets — need ≥ {k_folds}",
            "n_subsets": len(subsets),
        }
    # Deterministic subset → fold assignment.
    import numpy as np

    rng = np.random.default_rng(405)
    perm = rng.permutation(len(subsets))
    fold_for_subset = {subsets[perm[i]]: i % k_folds for i in range(len(subsets))}
    per_fold_r2: list[float] = []
    for fold in range(k_folds):
        train_subsets = {s for s, f in fold_for_subset.items() if f != fold}
        test_subsets = {s for s, f in fold_for_subset.items() if f == fold}
        train = d[d["subset"].isin(train_subsets)]
        test = d[d["subset"].isin(test_subsets)]
        if train.empty or test.empty:
            continue
        try:
            fit = smf.ols(formula, data=train).fit()
            pred = fit.predict(test)
            ss_res = float(((test["deltaLogP_mean"] - pred) ** 2).sum())
            ss_tot = float(((test["deltaLogP_mean"] - test["deltaLogP_mean"].mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            per_fold_r2.append(r2)
        except Exception as e:
            log.warning("CV fold %d failed: %s", fold, e)
            continue
    if not per_fold_r2:
        return {"status": "FAIL", "message": "all folds failed"}
    return {
        "status": "PASS",
        "k_folds": k_folds,
        "n_subsets": len(subsets),
        "mean_r2": float(sum(per_fold_r2) / len(per_fold_r2)),
        "per_fold_r2": per_fold_r2,
    }


def simulate_pre_sweep_runnability() -> dict:
    """Pre-sweep code-runnability gate (per Fix B3, Assumption A15).

    Simulate ΔlogP on a 168-cell-persona grid (21 cells × 8 held-out, with
    21 cells = K=1:8, K=2:6, K=4:6, K=8:1) with 2 seeds × 20 questions per
    (cell, persona) and a planted K×min interaction. Fit headline; report
    fitter status + recovered β_{K×min}.
    """
    rng = np.random.default_rng(405)
    rows: list[dict] = []
    # Build a tiny synthetic 21-cell layout
    cell_layout: list[tuple[int, list[str]]] = []
    cell_layout += [(1, [p]) for p in POOL]  # 8 cells
    pairs = rng.choice(len(POOL), size=(6, 2), replace=True).tolist()
    cell_layout += [(2, [POOL[i % len(POOL)] for i in p]) for p in pairs]  # 6 cells
    quads = rng.choice(len(POOL), size=(6, 4), replace=True).tolist()
    cell_layout += [(4, [POOL[i % len(POOL)] for i in q]) for q in quads]  # 6 cells
    cell_layout += [(8, list(POOL))]  # 1 cell

    # Use the real layer-20 distance matrix.
    names, dist = load_cosine_distance_matrix()
    from _issue405_common import min_dist_to_set

    planted_beta_K_x_min = -1.0  # planted slope (truth)
    for cidx, (K, positives) in enumerate(cell_layout):
        cell_id = f"sim_K{K}_c{cidx:02d}"
        subset_id = "+".join(sorted(positives))
        for seed in (42, 137):
            for persona in HELD_OUT:
                md = min_dist_to_set(persona, positives, names, dist)
                # Linear DGP: ΔlogP = -1.0 - 0.5*K + (planted) * K * md + noise
                mu = -1.0 - 0.5 * K + planted_beta_K_x_min * K * md
                # Replicate questions × seeds — average them down to a single mean
                # per (cell, persona, seed) row.
                draws = rng.normal(loc=mu, scale=0.3, size=20)
                rows.append(
                    {
                        "cell_id": cell_id,
                        "seed": seed,
                        "K": K,
                        "subset": subset_id,
                        "persona": persona,
                        "min_dist": md,
                        "mean_dist": md,  # placeholder; equal at K=1
                        "deltaLogP_mean": float(draws.mean()),
                        "trained_pos_mean_dlogp": -2.0 + 0.5 * K,
                        "emit_rate": 0.0,
                    }
                )

    fit = fit_mixed_effects(rows)
    fit["planted_beta_K_x_min"] = planted_beta_K_x_min
    fit["n_simulated_rows"] = len(rows)
    # Blocker 5 round-2: ALSO exercise the new robustness paths on
    # simulated data so the pre-sweep runnability gate covers them.
    # Without this, the new paths only run end-to-end the first time the
    # real headline regression runs on the pod.
    rows_no_comedian = [r for r in rows if r["persona"] != "comedian"]
    fit["per_K_slopes_min_full"] = fit_per_K_marginal_slopes(rows, "min_dist")
    fit["per_K_slopes_min_no_comedian"] = fit_per_K_marginal_slopes(rows_no_comedian, "min_dist")
    fit["leverage_min"] = leverage_diagnostics(rows, "min_dist")
    fit["cv_r2_min"] = cv_r2_at_subset_level(
        rows, formula="deltaLogP_mean ~ K * min_dist", k_folds=5
    )
    fit["cv_r2_mean"] = cv_r2_at_subset_level(
        rows, formula="deltaLogP_mean ~ K * mean_dist", k_folds=5
    )
    fit["headline_no_comedian"] = fit_mixed_effects(rows_no_comedian)
    return fit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_405"),
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=str(PROJECT_ROOT / "figures" / "issue_405"),
    )
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument("--simulated-only", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow result-file count shortfall (Blocker 5 default fails loud)",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_405" / "aggregate" / "regression.json"),
    )
    parser.add_argument(
        "--cell-specs-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_405" / "cell_specs.json"),
        help="Path to cell_specs.json (Phase 0.5 output). Round-4 fix 3 made this "
        "overridable so caller-level tests can drive the gate without "
        "needing the canonical specs file on disk.",
    )
    args = parser.parse_args()

    out: dict = {"runs": {}}

    if args.simulated or args.simulated_only:
        log.info("[sim] running pre-sweep runnability gate (Fix B3) ...")
        sim = simulate_pre_sweep_runnability()
        out["simulated_fit"] = sim
        log.info(
            "[sim] tool=%s status=%s n=%d",
            sim.get("tool"),
            sim.get("status"),
            sim.get("n_simulated_rows"),
        )

    if args.simulated_only:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        log.info("Wrote %s (simulated-only mode)", out_path)
        return 0

    eval_dir = Path(args.eval_dir)
    results = load_cell_results(eval_dir)
    names, dist = load_cosine_distance_matrix()

    # ── Round-3 fix 1: EXACT-equality track-count assert ─────────────
    # Plan §0 + §4.6: 21 CORE × 2 seeds = 42, 1 ABLNEG × 2 seeds = 2,
    # 3 DOSE50 × 2 seeds = 6. Total 50. Round-2 only caught shortfalls
    # (counts<expected); stale/extra result.json files in the OTHER
    # direction (overage) inflate the denominator → the same wrong-
    # denominator class blocker 2 closed. Round 3 asserts EXACT equality
    # per track AND surfaces the offending/extra cell_id+seed list for
    # both directions, plus any unknown-track rows.
    audit = audit_track_counts(results, specs_path=Path(args.cell_specs_path))
    out["track_counts"] = audit
    # Round-4 fix 1: include cell-seed identity in the mismatch decision.
    # Round 3's bool ignored `missing_cell_seeds` / `extra_cell_seeds`, so a
    # SAME-TRACK SWAP (drop planned K1_c00/42 + add stale K1_c99/42) left
    # the per-track totals identical → `has_mismatch=False` → the analyzer
    # would have fit the WRONG cell set with the correct denominator. Codex
    # caught this; the planned (cell_id, seed) set must be matched exactly.
    has_mismatch = bool(
        audit["shortfall"]
        or audit["overage"]
        or audit["unknown_tracks"]
        or audit["missing_cell_seeds"]
        or audit["extra_cell_seeds"]
    )
    if has_mismatch:
        msg = (
            f"Result-file count mismatch vs expected={audit['expected']!r}: "
            f"observed={audit['observed']!r}, shortfall={audit['shortfall']!r}, "
            f"overage={audit['overage']!r}, unknown_tracks={audit['unknown_tracks']!r}, "
            f"missing_cell_seeds={audit['missing_cell_seeds']!r}, "
            f"extra_cell_seeds={audit['extra_cell_seeds']!r}"
        )
        if args.allow_partial:
            log.warning(
                "[analyze] PARTIAL: %s (allow_partial=True; headline denominator may be wrong)", msg
            )
            out["partial_mismatch"] = True
        else:
            raise RuntimeError(
                msg + ". Pass --allow-partial to downgrade to a warning (investigate "
                "before fitting on a wrong denominator)."
            )

    # ── CORE headline frame ──────────────────────────────────────────
    df_core = build_cell_persona_frame(results, names, dist, track="CORE")
    log.info("CORE frame: %d cell-persona-seed rows", len(df_core))

    if df_core:
        log.info("Fitting headline regression: ΔlogP ~ K * min_dist + (1|subset) + (1|persona)")
        out["runs"]["headline_full"] = fit_mixed_effects(df_core)

        df_no_comedian = [r for r in df_core if r["persona"] != "comedian"]
        log.info("Fitting comedian-dropped refit (mandatory robustness per FIX B2)")
        out["runs"]["headline_no_comedian"] = fit_mixed_effects(df_no_comedian)

        log.info("Fitting min-only vs mean-only single-predictor variants (FIX B1)")
        out["runs"]["min_only"] = fit_mixed_effects(
            df_core, formula="deltaLogP_mean ~ K * min_dist"
        )
        out["runs"]["mean_only"] = fit_mixed_effects(
            df_core, formula="deltaLogP_mean ~ K * mean_dist"
        )

        log.info("Fitting covariate-adjusted variant (FIX A1 dose-vs-diversity)")
        out["runs"]["headline_cov"] = fit_mixed_effects(
            df_core,
            formula="deltaLogP_mean ~ K * min_dist + trained_pos_mean_dlogp",
        )

        # Leave-one-persona-out
        log.info("Leave-one-persona-out refits (8 fits) ...")
        loo_out: dict[str, dict] = {}
        for held_persona in sorted({r["persona"] for r in df_core}):
            subset_rows = [r for r in df_core if r["persona"] != held_persona]
            loo_out[held_persona] = fit_mixed_effects(subset_rows)
        out["runs"]["leave_one_persona_out"] = loo_out

        # ── Blocker 5: per-K marginal slopes on min_dist ─────────────
        log.info("Fitting per-K marginal slopes on min_dist (FIX B2 #3) ...")
        out["runs"]["per_K_slopes_min_full"] = fit_per_K_marginal_slopes(df_core, "min_dist")
        out["runs"]["per_K_slopes_min_no_comedian"] = fit_per_K_marginal_slopes(
            df_no_comedian, "min_dist"
        )

        # ── Blocker 5: leverage diagnostics (Cook's D + DFBETAS) ────
        log.info("Computing leverage diagnostics (FIX B2 #4) ...")
        out["runs"]["leverage_min"] = leverage_diagnostics(df_core, "min_dist")

        # ── Blocker 5: subset-level 5-fold CV-R² for min vs mean ────
        log.info("Computing subset-level 5-fold CV-R² for min-vs-mean (FIX B1) ...")
        out["runs"]["cv_r2_min"] = cv_r2_at_subset_level(
            df_core, formula="deltaLogP_mean ~ K * min_dist", k_folds=5
        )
        out["runs"]["cv_r2_mean"] = cv_r2_at_subset_level(
            df_core, formula="deltaLogP_mean ~ K * mean_dist", k_folds=5
        )

        # ── Blocker 5: JS-divergence sensitivity refit ───────────────
        # Per .claude/rules/persona-distance-metrics.md, JS is the
        # secondary distance metric. Sensitivity test: refit headline
        # with min_js_dist column if a JS-divergence matrix is present
        # at eval_results/extraction_method_comparison/js_matrix_layer21.json.
        # If absent, skip with an explicit SKIP status (NOT silent) so
        # the analyst sees the gap.
        js_matrix_path = (
            PROJECT_ROOT
            / "eval_results"
            / "extraction_method_comparison"
            / "js_matrix_layer21.json"
        )
        if js_matrix_path.exists():
            log.info("Loading JS-divergence matrix from %s ...", js_matrix_path)
            js_data = json.loads(js_matrix_path.read_text())
            js_names = js_data["persona_names"]
            js_dist = js_data["matrix"]
            df_core_js = build_cell_persona_frame(results, js_names, js_dist, track="CORE")
            out["runs"]["headline_full_js"] = fit_mixed_effects(df_core_js)
            out["runs"]["per_K_slopes_min_js"] = fit_per_K_marginal_slopes(df_core_js, "min_dist")
        else:
            out["runs"]["headline_full_js"] = {
                "status": "SKIP",
                "tool": "n/a",
                "message": (
                    f"JS-divergence matrix missing at {js_matrix_path} — "
                    f"sensitivity refit skipped. Generate via the JS-distance "
                    f"extractor (planner §6.4 robustness #5)."
                ),
            }

    # ── ABLNEG overlay (separate track) ──────────────────────────────
    df_abl = build_cell_persona_frame(results, names, dist, track="K4_ABLNEG")
    out["track_ablneg_n_rows"] = len(df_abl)
    out["track_ablneg_summary"] = (
        {
            "mean_deltaLogP_per_persona": {
                p: float(np.mean([r["deltaLogP_mean"] for r in df_abl if r["persona"] == p]))
                for p in sorted({r["persona"] for r in df_abl})
            }
        }
        if df_abl
        else {}
    )

    # ── DOSE50 head-to-head ──────────────────────────────────────────
    df_dose = build_cell_persona_frame(results, names, dist, track="K1_DOSE50")
    out["track_dose_n_rows"] = len(df_dose)
    if df_dose:
        # Compare main-K1@400 vs dose-K1@50 vs main-K8@50
        dose_summary = {}
        for persona in sorted({r["persona"] for r in df_dose}):
            dose_summary[persona] = {
                "dose_K1_50_dlogp": float(
                    np.mean([r["deltaLogP_mean"] for r in df_dose if r["persona"] == persona])
                ),
                "main_K1_400_dlogp": float(
                    np.mean(
                        [
                            r["deltaLogP_mean"]
                            for r in df_core
                            if r["persona"] == persona and r["K"] == 1
                        ]
                    )
                )
                if df_core
                else None,
                "main_K8_50_dlogp": float(
                    np.mean(
                        [
                            r["deltaLogP_mean"]
                            for r in df_core
                            if r["persona"] == persona and r["K"] == 8
                        ]
                    )
                )
                if df_core
                else None,
            }
        out["track_dose_summary"] = dose_summary

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("Wrote %s", out_path)

    if not args.skip_plots:
        try:
            _emit_figures(df_core, df_abl, df_dose, results, out, Path(args.fig_dir))
        except Exception as e:
            log.warning("Figure dump failed (%s) — regression.json still written", e)

    return 0


def _emit_figures(df_core, df_abl, df_dose, results, regression, fig_dir: Path) -> None:
    """Plot the exploratory figure dump per plan v2 §6.5."""
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Hero figure: leakage-vs-distance per K (RAW) ─────────────────
    if df_core:
        fig, ax = plt.subplots(figsize=(7, 5))
        for K in (1, 2, 4, 8):
            xs = [r["min_dist"] for r in df_core if r["K"] == K]
            ys = [r["deltaLogP_mean"] for r in df_core if r["K"] == K]
            if xs:
                ax.scatter(xs, ys, label=f"K={K}", alpha=0.6, s=30)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel("min layer-20 cosine distance to trained subset")
        ax.set_ylabel("held-out marker ΔlogP (trained − base)")
        ax.set_title("Issue #405 hero (RAW): held-out marker ΔlogP vs min-distance, per K")
        ax.legend(title="K (#sources)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "hero_raw.png", dpi=140)
        plt.close(fig)
        log.info("Wrote %s", fig_dir / "hero_raw.png")

    # ── Per-cell bars (held-out mean ΔlogP per cell) ─────────────────
    if df_core or df_abl or df_dose:
        cells: dict[str, list[float]] = {}
        Ks: dict[str, int] = {}
        for r in (df_core or []) + (df_abl or []) + (df_dose or []):
            cells.setdefault(r["cell_id"], []).append(r["deltaLogP_mean"])
            Ks[r["cell_id"]] = r["K"]
        fig, ax = plt.subplots(figsize=(12, 4))
        cell_names = sorted(cells, key=lambda c: (Ks[c], c))
        means = [float(np.mean(cells[c])) for c in cell_names]
        colors = {1: "C0", 2: "C1", 4: "C2", 8: "C3"}
        bar_colors = [colors.get(Ks[c], "C4") for c in cell_names]
        ax.bar(range(len(cell_names)), means, color=bar_colors)
        ax.set_xticks(range(len(cell_names)))
        ax.set_xticklabels(cell_names, rotation=80, fontsize=7)
        ax.set_ylabel("mean held-out ΔlogP")
        ax.set_title("Per-cell mean held-out ΔlogP (color = K)")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        fig.tight_layout()
        fig.savefig(fig_dir / "per_cell_bars.png", dpi=140)
        plt.close(fig)
        log.info("Wrote %s", fig_dir / "per_cell_bars.png")

    # ── Trained-positive ΔlogP × K (FIX A1 sanity) ───────────────────
    tp = []
    for r in results:
        if r.get("track") == "CORE":
            tp.append((r["K"], r["eval"]["summary"]["trained_pos_mean_dlogp"]))
    if tp:
        fig, ax = plt.subplots(figsize=(6, 4))
        Ks = sorted({k for k, _ in tp})
        for K in Ks:
            ys = [v for k, v in tp if k == K]
            ax.scatter([K] * len(ys), ys, alpha=0.6)
        ax.set_xlabel("K (#trained source personas)")
        ax.set_ylabel("trained_pos mean ΔlogP")
        ax.set_title("Per-cell source-strength scalar (FIX A1) vs K")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "trained_positive_vs_K.png", dpi=140)
        plt.close(fig)
        log.info("Wrote %s", fig_dir / "trained_positive_vs_K.png")


if __name__ == "__main__":
    sys.exit(main())
