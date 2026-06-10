#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ※, ×, —, ≈, Δ) in scientific docstrings + logs.
"""Issue #531 follow-up — probability-space versions of the base-prior plots.

Re-renders the two #531 figures (`shift_vs_base_prior`,
`absolute_trained_vs_base_prior`) in PROBABILITY space instead of log-prob
space, from the same tidy table the parent analysis wrote
(``eval_results/issue_478/base_prior_reanalysis/tidy.parquet``):

- ``P_base = exp(base_prior)``, ``P_trained = exp(trained_logp)``,
  ``ΔP = P_trained − P_base``.
- Plot 1: ΔP vs P_base (the probability-space analogue of the shift plot).
- Plot 2: P_trained vs P_base (the absolute panel).

Per the marker-leakage measurement rule, probability space is a behavioral
sanity read only — ``ΔP = P_base·(e^{Δlog P} − 1)`` over-weights high-prior
contexts. The instructive result: because trained mass is ~e^9.7 ≈ 17,000×
the base prior here, ΔP ≈ P_trained row-by-row, so the "shift" correlation
flips from −0.48 (log space, partial) to ≈ +0.74 (probability space) — the
−base term that dominates the log-space shift is numerically negligible in
probability space. Spearman on the ABSOLUTE panel is unchanged by exp()
(rank-invariant); the shift panel is not (ΔP is not a monotone transform of
Δlog P).

Logit-space versions are NOT derivable from this data: only resolved
log-probabilities were persisted in #478's per-cell result.json (no z_marker,
no log Z). That requires a GPU re-scoring pass over the stored on-policy
responses with the 80 CORE adapters (all on HF under ``issue_478/<cell>/adapter``).

Usage::

    uv run python scripts/issue531_probability_space_plots.py
    uv run python scripts/issue531_probability_space_plots.py --n-boot 100  # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue531_base_prior_reanalysis import (  # noqa: E402
    BAND_ORDER,
    BOOTSTRAP_SEED,
    HF_DATA_REPO,
    HF_DATA_REV,
    ISSUE_478_AGG_SHA,
    MARKER_ID,
    MARKER_TEXT,
    N_BOOTSTRAP,
    OUTPUT_FIG_DIR,
    OUTPUT_TIDY_DIR,
    _current_git_commit,
    _get_band_palette,
    partial_spearman_with_persona_bootstrap,
    spearman_with_persona_bootstrap_ci,
)

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("issue531_probability_space_plots")


def load_tidy_with_probabilities() -> pd.DataFrame:
    """Load the parent tidy table and add probability-space columns.

    Returns the 56,000-row table with ``P_base``, ``P_trained``, ``deltaP``
    appended (simple exp() of the stored log-probs).
    """
    tidy_path = OUTPUT_TIDY_DIR / "tidy.parquet"
    df = pd.read_parquet(tidy_path)
    df["P_base"] = np.exp(df["base_prior"])
    df["P_trained"] = np.exp(df["trained_logp"])
    df["deltaP"] = df["P_trained"] - df["P_base"]
    log.info("Loaded tidy table: %d rows from %s", len(df), tidy_path)
    return df


def _scatter_by_band(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> None:
    """Band-coloured scatter with the parent's ~600-points-per-band subsample."""
    palette = _get_band_palette()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for band in BAND_ORDER:
        sub = df[df["band"] == band]
        if sub.empty:
            continue
        if len(sub) > 600:
            sample_idx = rng.choice(len(sub), size=600, replace=False)
            sub = sub.iloc[sample_idx]
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=8,
            alpha=0.30,
            color=palette[band],
            label=band,
            linewidths=0,
        )


def _annotate_rho(ax: plt.Axes, raw: dict, par: dict) -> None:
    annotation = (
        f"Raw Spearman ρ = {raw['rho_point']:+.3f}"
        f"  [95% CI {raw['ci_lo_95']:+.3f}, {raw['ci_hi_95']:+.3f}]\n"
        f"Partial ρ (|min_dist, K) = {par['rho_point']:+.3f}"
        f"  [95% CI {par['ci_lo_95']:+.3f}, {par['ci_hi_95']:+.3f}]"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "lightgrey", "boxstyle": "round,pad=0.4"},
    )


def _figure_meta(*, fig_name: str, df: pd.DataFrame, extra: dict) -> dict:
    return {
        "figure": fig_name,
        "produced_by": "scripts/issue531_probability_space_plots.py",
        "git_commit_at_render": _current_git_commit(),
        "data_source": {
            "tidy_parquet": "eval_results/issue_478/base_prior_reanalysis/tidy.parquet",
            "hf_data_repo": HF_DATA_REPO,
            "hf_data_revision": HF_DATA_REV,
            "issue_478_aggregate_sha": ISSUE_478_AGG_SHA,
        },
        "rows_used": len(df),
        "cells_used": int(df["cell_id"].nunique()),
        "personas_used": int(df["held_out_persona"].nunique()),
        "marker_text": MARKER_TEXT,
        "marker_token_id": MARKER_ID,
        "rendered_at_utc": datetime.now(UTC).isoformat(),
        **extra,
    }


def _save(fig: plt.Figure, stem: str, meta: dict) -> None:
    png_path = OUTPUT_FIG_DIR / f"{stem}.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(OUTPUT_FIG_DIR / f"{stem}.pdf")
    plt.close(fig)
    (OUTPUT_FIG_DIR / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Wrote %s + .pdf + .meta.json", png_path)


def plot_deltap_vs_base_probability(df: pd.DataFrame, raw: dict, par: dict) -> None:
    """ΔP vs P_base — the probability-space analogue of the shift plot."""
    set_paper_style(target="blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _scatter_by_band(ax, df, "P_base", "deltaP")
    _annotate_rho(ax, raw, par)
    ax.set_xlabel(f"Base-model P({MARKER_TEXT}) at post-response slot")
    ax.set_ylabel(f"Trained − base P({MARKER_TEXT}) change")
    ax.set_title(
        "In probability space the shift flips positive — ΔP ≈ trained P, base prior negligible",
        loc="left",
    )
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    leg = ax.legend(
        title="Distance band",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
        ncols=2,
        markerscale=2.0,
        frameon=True,
    )
    leg.get_frame().set_edgecolor("lightgrey")
    plt.tight_layout()
    meta = _figure_meta(
        fig_name="shift_vs_base_prior_probability",
        df=df,
        extra={
            "x_axis": f"base-model P({MARKER_TEXT}) at post-response slot",
            "y_axis": f"trained − base P({MARKER_TEXT}) change",
            "rho_raw_deltaP": raw["rho_point"],
            "rho_partial_deltaP": par["rho_point"],
            "downsample_per_band": 600,
        },
    )
    _save(fig, "shift_vs_base_prior_probability", meta)


def plot_absolute_trained_probability(df: pd.DataFrame, raw: dict, par: dict) -> None:
    """P_trained vs P_base — the probability-space absolute panel."""
    set_paper_style(target="blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _scatter_by_band(ax, df, "P_base", "P_trained")
    _annotate_rho(ax, raw, par)
    ax.set_xlabel(f"Base-model P({MARKER_TEXT}) at post-response slot")
    ax.set_ylabel(f"Trained P({MARKER_TEXT}) at post-response slot")
    ax.set_title(
        "Trained probability vs base prior — Spearman is exp()-invariant, matches log space",
        loc="left",
    )
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    leg = ax.legend(
        title="Distance band",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
        ncols=2,
        markerscale=2.0,
        frameon=True,
    )
    leg.get_frame().set_edgecolor("lightgrey")
    plt.tight_layout()
    meta = _figure_meta(
        fig_name="absolute_trained_vs_base_prior_probability",
        df=df,
        extra={
            "x_axis": f"base-model P({MARKER_TEXT}) at post-response slot",
            "y_axis": f"trained P({MARKER_TEXT}) at post-response slot",
            "rho_raw_abs_prob": raw["rho_point"],
            "rho_partial_abs_prob": par["rho_point"],
            "downsample_per_band": 600,
        },
    )
    _save(fig, "absolute_trained_vs_base_prior_probability", meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOTSTRAP,
        help=f"Bootstrap resamples (default {N_BOOTSTRAP}).",
    )
    args = parser.parse_args()

    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_tidy_with_probabilities()

    log.info("=== Spearman ρ (P_base → ΔP) ===")
    raw_dp = spearman_with_persona_bootstrap_ci(
        df, x_col="P_base", y_col="deltaP", n_boot=args.n_boot
    )
    log.info(
        "RAW   ρ(P_base, ΔP) = %+.4f [%+.4f, %+.4f]",
        raw_dp["rho_point"],
        raw_dp["ci_lo_95"],
        raw_dp["ci_hi_95"],
    )
    par_dp = partial_spearman_with_persona_bootstrap(
        df, x_col="P_base", y_col="deltaP", control_cols=["min_dist", "K"], n_boot=args.n_boot
    )
    log.info(
        "PART  ρ(P_base, ΔP | min_dist, K) = %+.4f [%+.4f, %+.4f]",
        par_dp["rho_point"],
        par_dp["ci_lo_95"],
        par_dp["ci_hi_95"],
    )

    log.info("=== Spearman ρ (P_base → P_trained) ===")
    raw_abs = spearman_with_persona_bootstrap_ci(
        df, x_col="P_base", y_col="P_trained", n_boot=args.n_boot
    )
    par_abs = partial_spearman_with_persona_bootstrap(
        df, x_col="P_base", y_col="P_trained", control_cols=["min_dist", "K"], n_boot=args.n_boot
    )
    log.info(
        "ABS   raw %+.4f [%+.4f, %+.4f]   partial %+.4f [%+.4f, %+.4f]",
        raw_abs["rho_point"],
        raw_abs["ci_lo_95"],
        raw_abs["ci_hi_95"],
        par_abs["rho_point"],
        par_abs["ci_lo_95"],
        par_abs["ci_hi_95"],
    )

    summary = {
        "task": "issue_531_probability_space_followup",
        "produced_by": "scripts/issue531_probability_space_plots.py",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _current_git_commit(),
        "note": (
            "Probability-space re-render of the #531 plots. P = exp(logP) of the "
            "stored values; ΔP = P_trained − P_base. Spearman on the absolute panel "
            "is identical to log space (exp is monotone); the shift panel flips "
            "positive because ΔP ≈ P_trained (trained mass is ~e^9.7 × base prior, "
            "so the −base term is numerically negligible). Probability space "
            "over-weights high-prior contexts — sanity read, not the analysis DV. "
            "Logit space is NOT derivable from stored data (no z_marker / log Z "
            "persisted); requires a GPU re-scoring pass."
        ),
        "n_rows": len(df),
        "raw_spearman_deltaP": raw_dp,
        "partial_spearman_deltaP": par_dp,
        "raw_spearman_abs_prob": raw_abs,
        "partial_spearman_abs_prob": par_abs,
        "log_space_quotes": {
            "partial_shift_logspace": -0.480,
            "partial_abs_logspace": 0.739,
        },
    }
    summary_path = OUTPUT_TIDY_DIR / "summary_probability.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote summary: %s", summary_path)

    log.info("=== render figures ===")
    plot_deltap_vs_base_probability(df, raw_dp, par_dp)
    plot_absolute_trained_probability(df, raw_abs, par_abs)

    print()
    print("=" * 78)
    print("Issue #531 follow-up — probability-space plots")
    print("=" * 78)
    print(
        f"  ΔP vs P_base:        raw ρ = {raw_dp['rho_point']:+.4f}"
        f"   partial ρ = {par_dp['rho_point']:+.4f}"
        f"   [{par_dp['ci_lo_95']:+.4f}, {par_dp['ci_hi_95']:+.4f}]"
    )
    print(
        f"  P_trained vs P_base: raw ρ = {raw_abs['rho_point']:+.4f}"
        f"   partial ρ = {par_abs['rho_point']:+.4f}"
        f"   [{par_abs['ci_lo_95']:+.4f}, {par_abs['ci_hi_95']:+.4f}]"
    )
    print("  (log-space quotes: shift partial −0.480, absolute partial +0.739)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
