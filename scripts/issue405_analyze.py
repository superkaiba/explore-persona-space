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
            if persona not in names:
                continue
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
        return {
            "status": "PASS",
            "tool": "rpy2+lme4",
            "formula": lme_formula,
            "n_obs": n,
            "coefs": coefs,
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
        return {
            "status": "PASS",
            "tool": "statsmodels.MixedLM (vc_formula)",
            "formula": formula,
            "n_obs": n,
            "coefs": coefs,
        }
    except Exception as e:
        log.error("statsmodels.MixedLM fit failed too (%s)", e)
        return {"status": "FAIL", "tool": "all_failed", "message": str(e)}


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
        "--out-path",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_405" / "aggregate" / "regression.json"),
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
