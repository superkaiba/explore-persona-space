#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ※, Δ, ×, —, ≈) in scientific docstrings + logs.
"""Issue #531 follow-up — logit-space versions of the base-prior plots.

Consumes the per-cell JSONs written by ``issue531_logit_rescore.py``
(``eval_results/issue_478/logit_rescore/*.json``), joins them with the parent
tidy table (band, min_dist, K), and renders the logit-space analogues of the
two #531 figures:

- ``shift_vs_base_prior_logit``    — y = Δz_※ (trained − base marker logit),
  x = base marker logit z_※.
- ``absolute_trained_vs_base_prior_logit`` — y = trained z_※, x = base z_※.

Also emits the saturation diagnostic the marker-leakage rule asks for:
``Δlog P ≈ Δz − Δlog Z``, so off-saturation (Δlog Z ≈ 0) the log-prob shift
and the logit shift should agree row-by-row; their divergence localizes
softmax compression. Stats use the same partial-Spearman (controls: min_dist,
K) + 1000-resample persona-cluster bootstrap as the parent analysis. The
mixed-axis partials (x = base log P, matching the parent's x-axis exactly)
are reported in the summary for direct comparability.

Usage::

    uv run python scripts/issue531_logit_plots.py
    uv run python scripts/issue531_logit_plots.py --n-boot 100  # smoke
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

RESCORE_DIR = PROJECT_ROOT / "eval_results" / "issue_478" / "logit_rescore"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("issue531_logit_plots")


def build_logit_tidy() -> pd.DataFrame:
    """Merge per-cell rescore JSONs with the parent tidy table.

    Returns one row per (cell_id, seed, held_out_persona, question_idx) with
    logit-space columns alongside the parent's log-prob columns + controls.
    Fails loud if any of the 80 expected cells is missing or any parent row
    has no rescored partner.
    """
    parent = pd.read_parquet(OUTPUT_TIDY_DIR / "tidy.parquet")
    expected = sorted({(c, int(s)) for c, s in zip(parent["cell_id"], parent["seed"], strict=True)})

    rows = []
    missing_cells = []
    for cell_id, seed in expected:
        path = RESCORE_DIR / f"{cell_id}_seed{seed}.json"
        if not path.exists():
            missing_cells.append(path.name)
            continue
        d = json.loads(path.read_text())
        for persona, rec in d["held_out"].items():
            n_q = len(rec["z_marker_trained_per_q"])
            for qi in range(n_q):
                rows.append(
                    {
                        "cell_id": cell_id,
                        "seed": seed,
                        "held_out_persona": persona,
                        "question_idx": qi,
                        "z_trained": rec["z_marker_trained_per_q"][qi],
                        "z_base": rec["z_marker_base_per_q"][qi],
                        "z_eos_trained": rec["z_eos_trained_per_q"][qi],
                        "z_eos_base": rec["z_eos_base_per_q"][qi],
                        "logZ_trained": rec["logZ_trained_per_q"][qi],
                        "logZ_base": rec["logZ_base_per_q"][qi],
                        "logp_trained_rescored": rec["logp_trained_per_q"][qi],
                        "logp_base_rescored": rec["logp_base_per_q"][qi],
                    }
                )
    if missing_cells:
        raise FileNotFoundError(
            f"{len(missing_cells)} cell JSONs missing under {RESCORE_DIR}: "
            f"{missing_cells[:6]} ... — run issue531_logit_rescore.py to completion first"
        )

    logit = pd.DataFrame(rows)
    logit["dz"] = logit["z_trained"] - logit["z_base"]
    logit["dlogZ"] = logit["logZ_trained"] - logit["logZ_base"]
    logit["dlogp_rescored"] = logit["logp_trained_rescored"] - logit["logp_base_rescored"]
    # EOS margin — the preferred logit readout (shift-invariant; anchored to
    # the emission threshold: the marker fires when it overtakes EOS).
    logit["margin_trained"] = logit["z_trained"] - logit["z_eos_trained"]
    logit["margin_base"] = logit["z_base"] - logit["z_eos_base"]
    logit["dmargin"] = logit["margin_trained"] - logit["margin_base"]

    merged = parent.merge(
        logit,
        on=["cell_id", "seed", "held_out_persona", "question_idx"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(parent):
        raise ValueError(
            f"join dropped rows: parent={len(parent)}, merged={len(merged)} — "
            f"rescore output incomplete or misaligned"
        )
    log.info("Logit tidy: %d rows across %d cells", len(merged), merged["cell_id"].nunique())
    return merged


def agreement_diagnostics(df: pd.DataFrame) -> dict:
    """How closely Δlog P tracks Δz (off-saturation they should agree)."""
    from scipy.stats import pearsonr, spearmanr

    dz = df["dz"].to_numpy()
    dlogp = df["dlogp_rescored"].to_numpy()
    dlogz = df["dlogZ"].to_numpy()
    resc_vs_stored = float(np.mean(np.abs(df["logp_trained_rescored"] - df["trained_logp"])))
    dmargin = df["dmargin"].to_numpy()
    return {
        "mean_abs_dlogZ_nats": float(np.mean(np.abs(dlogz))),
        "p95_abs_dlogZ_nats": float(np.percentile(np.abs(dlogz), 95)),
        "mean_dlogZ_nats": float(np.mean(dlogz)),
        "pearson_dlogp_vs_dz": float(pearsonr(dlogp, dz).statistic),
        "spearman_dlogp_vs_dz": float(spearmanr(dlogp, dz).statistic),
        "mean_abs_residual_dlogp_minus_dz_nats": float(np.mean(np.abs(dlogp - dz))),
        # Δz vs Δmargin divergence = common-mode logit shift (behaviorally inert).
        "pearson_dz_vs_dmargin": float(pearsonr(dz, dmargin).statistic),
        "spearman_dz_vs_dmargin": float(spearmanr(dz, dmargin).statistic),
        "mean_abs_residual_dz_minus_dmargin": float(np.mean(np.abs(dz - dmargin))),
        "validation_mae_rescored_vs_stored_trained_logp_nats": resc_vs_stored,
    }


def _scatter_by_band(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str) -> None:
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
        "produced_by": "scripts/issue531_logit_plots.py",
        "git_commit_at_render": _current_git_commit(),
        "data_source": {
            "rescore_dir": "eval_results/issue_478/logit_rescore/",
            "parent_tidy": "eval_results/issue_478/base_prior_reanalysis/tidy.parquet",
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


def plot_logit_shift(df: pd.DataFrame, raw: dict, par: dict) -> None:
    set_paper_style(target="blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _scatter_by_band(ax, df, "z_base", "dz")
    _annotate_rho(ax, raw, par)
    ax.set_xlabel(f"Base-model logit z({MARKER_TEXT}) at post-response slot")
    ax.set_ylabel(f"Trained − base logit Δz({MARKER_TEXT})")
    ax.set_title(
        "Logit-space shift vs base marker logit (gauge-free Δz = W_U[※]·Δh)",
        loc="left",
    )
    leg = ax.legend(
        title="Distance band",
        loc="lower left",
        fontsize=8,
        title_fontsize=8,
        ncols=2,
        markerscale=2.0,
        frameon=True,
    )
    leg.get_frame().set_edgecolor("lightgrey")
    plt.tight_layout()
    meta = _figure_meta(
        fig_name="shift_vs_base_prior_logit",
        df=df,
        extra={
            "x_axis": f"base-model marker logit z({MARKER_TEXT}) at post-response slot",
            "y_axis": f"trained − base marker logit Δz({MARKER_TEXT})",
            "rho_raw": raw["rho_point"],
            "rho_partial": par["rho_point"],
            "downsample_per_band": 600,
        },
    )
    _save(fig, "shift_vs_base_prior_logit", meta)


def plot_logit_absolute(df: pd.DataFrame, raw: dict, par: dict) -> None:
    set_paper_style(target="blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _scatter_by_band(ax, df, "z_base", "z_trained")
    _annotate_rho(ax, raw, par)
    ax.set_xlabel(f"Base-model logit z({MARKER_TEXT}) at post-response slot")
    ax.set_ylabel(f"Trained logit z({MARKER_TEXT}) at post-response slot")
    ax.set_title(
        "Absolute trained marker logit vs base marker logit",
        loc="left",
    )
    leg = ax.legend(
        title="Distance band",
        loc="lower right",
        fontsize=8,
        title_fontsize=8,
        ncols=2,
        markerscale=2.0,
        frameon=True,
    )
    leg.get_frame().set_edgecolor("lightgrey")
    plt.tight_layout()
    meta = _figure_meta(
        fig_name="absolute_trained_vs_base_prior_logit",
        df=df,
        extra={
            "x_axis": f"base-model marker logit z({MARKER_TEXT}) at post-response slot",
            "y_axis": f"trained marker logit z({MARKER_TEXT}) at post-response slot",
            "rho_raw": raw["rho_point"],
            "rho_partial": par["rho_point"],
            "downsample_per_band": 600,
        },
    )
    _save(fig, "absolute_trained_vs_base_prior_logit", meta)


def plot_margin_shift(df: pd.DataFrame, raw: dict, par: dict) -> None:
    """EOS-margin shift — the preferred (shift-invariant) logit readout."""
    set_paper_style(target="blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _scatter_by_band(ax, df, "margin_base", "dmargin")
    _annotate_rho(ax, raw, par)
    ax.set_xlabel(f"Base-model EOS margin z({MARKER_TEXT}) − z(EOS) at post-response slot")
    ax.set_ylabel(f"Trained − base EOS margin Δ(z({MARKER_TEXT}) − z(EOS))")
    ax.set_title(
        "EOS-margin shift vs base margin — shift-invariant logit readout",
        loc="left",
    )
    leg = ax.legend(
        title="Distance band",
        loc="lower left",
        fontsize=8,
        title_fontsize=8,
        ncols=2,
        markerscale=2.0,
        frameon=True,
    )
    leg.get_frame().set_edgecolor("lightgrey")
    plt.tight_layout()
    meta = _figure_meta(
        fig_name="shift_vs_base_prior_eos_margin",
        df=df,
        extra={
            "x_axis": f"base-model z({MARKER_TEXT}) − z(EOS) at post-response slot",
            "y_axis": f"trained − base Δ(z({MARKER_TEXT}) − z(EOS))",
            "rho_raw": raw["rho_point"],
            "rho_partial": par["rho_point"],
            "downsample_per_band": 600,
        },
    )
    _save(fig, "shift_vs_base_prior_eos_margin", meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = build_logit_tidy()

    tidy_path = OUTPUT_TIDY_DIR / "tidy_logit.parquet"
    df.to_parquet(tidy_path, index=False)
    log.info("Wrote merged logit tidy: %s", tidy_path)

    diag = agreement_diagnostics(df)
    log.info("Agreement diagnostics: %s", json.dumps(diag, indent=1))

    stats: dict[str, dict] = {}
    pairs = [
        ("raw_dz_vs_zbase", dict(x_col="z_base", y_col="dz"), False),
        ("partial_dz_vs_zbase", dict(x_col="z_base", y_col="dz"), True),
        ("raw_ztrained_vs_zbase", dict(x_col="z_base", y_col="z_trained"), False),
        ("partial_ztrained_vs_zbase", dict(x_col="z_base", y_col="z_trained"), True),
        # Mixed-axis variants: x = base log P (the parent's x-axis) for
        # head-to-head comparability with the log-prob figures.
        ("raw_dz_vs_baselogp", dict(x_col="base_prior", y_col="dz"), False),
        ("partial_dz_vs_baselogp", dict(x_col="base_prior", y_col="dz"), True),
        ("raw_ztrained_vs_baselogp", dict(x_col="base_prior", y_col="z_trained"), False),
        ("partial_ztrained_vs_baselogp", dict(x_col="base_prior", y_col="z_trained"), True),
        # EOS margin (z_marker − z_eos) — the preferred logit readout.
        ("raw_dmargin_vs_marginbase", dict(x_col="margin_base", y_col="dmargin"), False),
        ("partial_dmargin_vs_marginbase", dict(x_col="margin_base", y_col="dmargin"), True),
        (
            "raw_margintrained_vs_marginbase",
            dict(x_col="margin_base", y_col="margin_trained"),
            False,
        ),
        (
            "partial_margintrained_vs_marginbase",
            dict(x_col="margin_base", y_col="margin_trained"),
            True,
        ),
    ]
    for name, kw, is_partial in pairs:
        if is_partial:
            stats[name] = partial_spearman_with_persona_bootstrap(
                df, control_cols=["min_dist", "K"], n_boot=args.n_boot, **kw
            )
        else:
            stats[name] = spearman_with_persona_bootstrap_ci(df, n_boot=args.n_boot, **kw)
        log.info(
            "%-28s ρ = %+.4f [%+.4f, %+.4f]",
            name,
            stats[name]["rho_point"],
            stats[name]["ci_lo_95"],
            stats[name]["ci_hi_95"],
        )

    summary = {
        "task": "issue_531_logit_space_followup",
        "produced_by": "scripts/issue531_logit_plots.py",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _current_git_commit(),
        "n_rows": len(df),
        "agreement_diagnostics": diag,
        "spearman": stats,
        "log_space_quotes": {
            "partial_shift_logspace": -0.480,
            "partial_abs_logspace": 0.739,
        },
    }
    (OUTPUT_TIDY_DIR / "summary_logit.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote summary: %s", OUTPUT_TIDY_DIR / "summary_logit.json")

    plot_logit_shift(df, stats["raw_dz_vs_zbase"], stats["partial_dz_vs_zbase"])
    plot_logit_absolute(df, stats["raw_ztrained_vs_zbase"], stats["partial_ztrained_vs_zbase"])
    plot_margin_shift(
        df, stats["raw_dmargin_vs_marginbase"], stats["partial_dmargin_vs_marginbase"]
    )

    print()
    print("=" * 78)
    print("Issue #531 follow-up — logit-space plots")
    print("=" * 78)
    print(
        f"  Δz vs z_base:        raw ρ = {stats['raw_dz_vs_zbase']['rho_point']:+.4f}"
        f"   partial ρ = {stats['partial_dz_vs_zbase']['rho_point']:+.4f}"
    )
    print(
        f"  z_tr vs z_base:      raw ρ = {stats['raw_ztrained_vs_zbase']['rho_point']:+.4f}"
        f"   partial ρ = {stats['partial_ztrained_vs_zbase']['rho_point']:+.4f}"
    )
    print(
        f"  Δz vs base logP:     raw ρ = {stats['raw_dz_vs_baselogp']['rho_point']:+.4f}"
        f"   partial ρ = {stats['partial_dz_vs_baselogp']['rho_point']:+.4f}"
    )
    print(
        f"  Δmargin vs margin_b: raw ρ = {stats['raw_dmargin_vs_marginbase']['rho_point']:+.4f}"
        f"   partial ρ = {stats['partial_dmargin_vs_marginbase']['rho_point']:+.4f}"
    )
    print(
        f"  mean |ΔlogZ| = {diag['mean_abs_dlogZ_nats']:.3f} nats"
        f"   spearman(Δlogp, Δz) = {diag['spearman_dlogp_vs_dz']:+.4f}"
        f"   spearman(Δz, Δmargin) = {diag['spearman_dz_vs_dmargin']:+.4f}"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
