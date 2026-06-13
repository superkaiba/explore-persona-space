#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, −, ×) in scientific docstrings + labels.
"""Issue #559 follow-up — length-residualized prior ranking re-read.

The fourth finding of #559 names per-persona own-response token length as the
strongest surface alternative to the own-response prior (length ↔ prior
Spearman −0.69). This free-analysis follow-up asks whether the prior's +0.605
within-run median rank correlation survives partialling length out:

1. Per-persona mean own-response length (35 values) from the already-persisted
   ``own_R_token_lens`` in ``R_base_own.json`` — generated-token counts per
   persona × question, averaged over the 20 eval questions.
2. OLS-residualize the 35-value prior on standardized length (linear PRIMARY;
   + length² as the quadratic robustness pair).
3. Re-run the production within-run ranking (same code path:
   ``issue559_panel_analysis.within_run_ranking``, seed 42, 2,000 boots, dual
   run/cell bootstrap axes) for the residualized prior, plus paired
   comparisons vs the raw prior and vs the matched-slot incumbent.
4. Rank raw length itself (sign-oriented) — the pure-surface baseline that
   separates "the prior is mostly length" from "length is a correlate but the
   prior carries persona signal beyond it".

Reuses (imports, never copies) the committed #559 machinery so the ranking /
parity numbers are computed by the exact production functions; a
reproduction gate asserts the raw-prior and matched-slot medians match the
committed ``within_run_ranking.json`` to 1e-9 before any new number is read.

Outputs ``length_residual_followup.json`` to ``--out-dir`` and
``length_residual_ranking.png`` to ``--fig-dir``. Single-phase CPU script:
the smoke IS the production run.

Usage::

    uv run python scripts/issue559_length_residual_followup.py \\
        --out-dir eval_results/issue_559 --fig-dir figures/issue_559
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue539_residual_per_cohort as i539  # noqa: E402
import issue553_panel as p553  # noqa: E402
import issue559_panel_analysis as i559  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

SCHEMA_VERSION = "issue559_length_residual_followup_v1"
REPRO_TOL = 1e-9  # deterministic point-estimate reproduction vs the committed ranking JSON
LENGTH_DEFINITION = (
    "per-persona MEAN over the 20 eval questions of the generated-response token count "
    "(own_R_token_lens in R_base_own.json — pod-side tokenizer counts of the base model's "
    "own greedy responses; no re-tokenization performed here)"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = p553.common_parser(
        "Issue #559 follow-up — length-residualized own-response prior ranking re-read"
    )
    parser.set_defaults(out_dir=Path("eval_results/issue_559"), fig_dir=Path("figures/issue_559"))
    parser.add_argument(
        "--prior-json",
        type=Path,
        default=Path("eval_results/issue_559/base_prior_own_persona_panel.json"),
    )
    parser.add_argument(
        "--r-base-own",
        type=Path,
        default=Path("eval_results/issue_559/R_base_own.json"),
        help="own-response generations (token-length source); REQUIRED here",
    )
    parser.add_argument(
        "--ranking-json",
        type=Path,
        default=Path("eval_results/issue_559/within_run_ranking.json"),
        help="committed production ranking — reproduction gate for raw prior + matched slot",
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="accept an is_stub prior JSON (SMOKE ONLY — production refuses stubs)",
    )
    return parser.parse_args(argv)


def _ols_residuals(y: np.ndarray, covariates: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """OLS residuals of y on [1 | covariates]; returns (residuals, fit summary)."""
    n = len(y)
    design = np.column_stack([np.ones(n), *covariates])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    resid = y - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return resid, {
        "coefficients": [float(c) for c in coef],
        "r2": 1.0 - ss_res / ss_tot,
        "n": n,
    }


def _ci_pair(blk: dict) -> tuple[dict, dict]:
    return blk["median_ci95_run_boot"], blk["median_ci95_cell_boot"]


def classify_survival(resid_blk: dict, beyond_run: dict, beyond_cell: dict) -> dict:
    """Dual-axis conservative survival read (mirrors the §13.1/§13.2 convention).

    SURVIVES_BEYOND_LENGTH: residualized-prior median ρ CI entirely > 0 on BOTH
    bootstrap axes AND the paired (resid − length-alone) median-difference CI
    entirely > 0 on BOTH axes. SURVIVES: the first condition only. COLLAPSES:
    the residualized prior's CI touches/spans 0 on at least one axis
    (conservative read governs).
    """
    run_ci, cell_ci = _ci_pair(resid_blk)
    resid_pos = run_ci["low"] > 0 and cell_ci["low"] > 0
    br, bc = beyond_run["median_diff_ci95"], beyond_cell["median_diff_ci95"]
    beyond_pos = br["low"] > 0 and bc["low"] > 0
    if resid_pos and beyond_pos:
        cls = "SURVIVES_BEYOND_LENGTH"
        read = (
            "length-residualized prior median ρ CI entirely > 0 on both resampling axes "
            "AND it out-ranks length-alone (paired diff CI entirely > 0 on both axes) — "
            "the prior carries persona signal beyond response length"
        )
    elif resid_pos:
        cls = "SURVIVES"
        read = (
            "length-residualized prior median ρ CI entirely > 0 on both resampling axes, "
            "but its paired advantage over length-alone is not decisively positive"
        )
    else:
        cls = "COLLAPSES"
        read = (
            "length-residualized prior median ρ CI touches/spans 0 on at least one "
            "resampling axis (conservative dual-axis read) — the within-run ranking "
            "signal is not separable from response length"
        )
    return {
        "classification": cls,
        "read": read,
        "resid_ci_run": run_ci,
        "resid_ci_cell": cell_ci,
        "resid_minus_length_run": beyond_run["median_diff_ci95"],
        "resid_minus_length_cell": beyond_cell["median_diff_ci95"],
    }


def figure_length_residual(ranking: dict, fig_dir: Path) -> None:
    """Strip: per-run ρ + median + run-axis CI for the four headline rankers."""
    set_paper_style("blog")
    colors = paper_palette(2)
    order = ["prior_margin_own", "prior_len_resid_lin", "len_oriented", "margin_base"]
    labels = [
        "own-response prior\n(raw)",
        "own-response prior\n(length-residualized)",
        "response length alone\n(sign-oriented)",
        "base matched-slot margin\n(needs trained responses)",
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    rng = np.random.default_rng(0)
    for xi, ranker in enumerate(order):
        blk = ranking[ranker]
        vals = [v for v in blk["per_run_rho"].values() if not np.isnan(v)]
        color = colors[1] if ranker == "margin_base" else colors[0]
        jitter = (rng.random(len(vals)) - 0.5) * 0.18
        ax.plot(np.full(len(vals), xi) + jitter, vals, "o", ms=3.5, alpha=0.45, color=color)
        med = blk["median_rho"]
        ci = blk["median_ci95_run_boot"]
        # Clamp: bootstrap CI bounds can sit float-epsilon past the median.
        lo_e = max(0.0, med - ci["low"])
        hi_e = max(0.0, ci["high"] - med)
        ax.errorbar(
            [xi + 0.30], [med], yerr=[[lo_e], [hi_e]], fmt="o", ms=5, color=color, capsize=3
        )
        ax.plot([xi - 0.22, xi + 0.22], [med, med], color=color, lw=2.4)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Per-run Spearman ρ vs trained EOS margin (35 personas)")
    ax.set_title(
        "Length-residualized prior re-read — within-run ranking, 80 runs\n"
        "dots = per-run ρ; bar = median; whisker = 95% run-bootstrap CI on the median",
        fontsize=9,
    )
    # No tight_layout(): set_paper_style("blog") enables constrained layout;
    # switching engines mid-figure clips the y-label.
    savefig_paper(fig, "length_residual_ranking", dir=fig_dir)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    # ── Inputs (same gates as the production analysis) ───────────────────────
    payload = i559.load_prior(args)
    prior_df, _per_q = i559.prior_frames(payload)

    if not args.r_base_own.exists():
        raise SystemExit(f"{args.r_base_own} missing — length source is required here")
    r_own = json.loads(args.r_base_own.read_text())
    lens_by_p = {
        p: float(np.mean(list(qmap.values()))) for p, qmap in r_own["own_R_token_lens"].items()
    }
    prior_df["own_len_mean"] = prior_df["held_out_persona"].map(lens_by_p)
    assert not prior_df["own_len_mean"].isna().any(), "length join produced NaN — persona mismatch"
    assert len(prior_df) == 35, len(prior_df)

    # ── Residualization (35 persona-level values) ────────────────────────────
    y = prior_df["prior_margin_own"].to_numpy(np.float64)
    lens = prior_df["own_len_mean"].to_numpy(np.float64)
    lens_z = (lens - lens.mean()) / lens.std()
    resid_lin, fit_lin = _ols_residuals(y, [lens_z])
    resid_quad, fit_quad = _ols_residuals(y, [lens_z, lens_z**2])
    prior_df["prior_len_resid_lin"] = resid_lin
    prior_df["prior_len_resid_quad"] = resid_quad
    print(
        f"[resid] prior ~ length: linear R²={fit_lin['r2']:.3f}, quadratic R²={fit_quad['r2']:.3f}"
    )

    # ── Panel + step-0 gate (identical to production) ────────────────────────
    df = p553.load_i478_panel(args.i478_parquet)
    p553.step0_i478(df, args.i478_parquet.parent / "summary_logit.json")
    agg = p553.aggregate_run_persona(df)
    join_cols = [
        "held_out_persona",
        "prior_margin_own",
        "prior_len_resid_lin",
        "prior_len_resid_quad",
        "own_len_mean",
    ]
    agg = agg.merge(prior_df[join_cols], on="held_out_persona", how="left", validate="many_to_one")
    assert not agg["prior_margin_own"].isna().any(), "prior join produced NaN — persona mismatch"
    cell_of_run = dict(zip(agg["run_id"], agg["cell_id"], strict=True))

    # Length-alone baseline, sign-oriented so its per-run ρ reads positively
    # (same orientation convention as the production z_stack).
    s_len = float(
        np.sign(
            i539._spearman_rho(
                agg["own_len_mean"].to_numpy(np.float64), agg[i559.DV_COL].to_numpy(np.float64)
            )
        )
    )
    agg["len_oriented"] = s_len * agg["own_len_mean"]

    # ── Within-run ranking (production code path) ────────────────────────────
    rankers = [
        "margin_base",
        "prior_margin_own",
        "prior_len_resid_lin",
        "prior_len_resid_quad",
        "own_len_mean",
        "len_oriented",
    ]
    print("[ranking] within-run ranking (6 rankers, 80 runs) ...")
    ranking = i559.within_run_ranking(agg, rankers, i559.DV_COL, cell_of_run, args)

    # ── Reproduction gate vs the committed production ranking ────────────────
    committed = json.loads(args.ranking_json.read_text())["within_run_ranking"]
    repro = {}
    for k in ("margin_base", "prior_margin_own"):
        got, want = ranking[k]["median_rho"], committed[k]["median_rho"]
        ok = abs(got - want) <= REPRO_TOL
        repro[k] = {"got": got, "want": want, "pass": ok}
        if not ok:
            raise SystemExit(
                f"REPRODUCTION GATE FAILED: {k} median ρ {got!r} != committed {want!r} "
                "— join/aggregation drift, refusing to read any new number"
            )
    print("[gate] raw prior + matched-slot medians reproduce the committed ranking (1e-9)")

    # ── Paired comparisons (dual axes, production functions) ─────────────────
    rho = {k: ranking[k]["per_run_rho"] for k in rankers}
    pair_defs = {
        "raw_prior_minus_resid_lin": ("prior_margin_own", "prior_len_resid_lin"),
        "resid_lin_minus_length_oriented": ("prior_len_resid_lin", "len_oriented"),
        "matched_minus_resid_lin": ("margin_base", "prior_len_resid_lin"),
        "raw_prior_minus_length_oriented": ("prior_margin_own", "len_oriented"),
    }
    paired: dict = {}
    for name, (a, b) in pair_defs.items():
        paired[f"{name}_run_axis"] = i559.paired_difference_block(rho[a], rho[b], args)
        paired[f"{name}_cell_axis"] = i559.paired_difference_cellaxis(
            rho[a], rho[b], cell_of_run, args
        )

    verdict = classify_survival(
        ranking["prior_len_resid_lin"],
        paired["resid_lin_minus_length_oriented_run_axis"],
        paired["resid_lin_minus_length_oriented_cell_axis"],
    )
    print(f"[verdict] {verdict['classification']}: {verdict['read']}")

    # ── Output JSON + figure ─────────────────────────────────────────────────
    per_view = prior_df.set_index("held_out_persona")
    meta = p553.result_metadata(args, "scripts/issue559_length_residual_followup.py")
    meta["task"] = 559
    meta["schema_version"] = SCHEMA_VERSION
    meta["prior_is_stub"] = bool(payload.get("is_stub", False))
    out = {
        "metadata": meta,
        "dv": i559.DV_COL,
        "length_definition": LENGTH_DEFINITION,
        "length_vs_prior": {
            "pearson_r": float(np.corrcoef(lens, y)[0, 1]),
            "spearman_rho": i539._spearman_rho(lens, y),
            "n_personas": 35,
        },
        "residualization": {"linear": fit_lin, "quadratic": fit_quad},
        "length_orientation_sign": s_len,
        "within_run_ranking": ranking,
        "reproduction_gate": repro,
        "paired": paired,
        "verdict": verdict,
        "per_persona": {
            p: {
                "prior_margin_own": float(per_view.loc[p, "prior_margin_own"]),
                "own_len_mean": float(per_view.loc[p, "own_len_mean"]),
                "prior_len_resid_lin": float(per_view.loc[p, "prior_len_resid_lin"]),
                "prior_len_resid_quad": float(per_view.loc[p, "prior_len_resid_quad"]),
            }
            for p in per_view.index
        },
    }
    p553.write_json(args.out_dir / "length_residual_followup.json", out)
    figure_length_residual(ranking, args.fig_dir)
    print("[done] length-residual follow-up complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
