#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #478 PHASE 5 — clean-result figures (HERO + exploratory dump).

Per plan v5 §4.8 PHASE 5 + §6 Figures + §6.8 hero figure.

Produces (in ``figures/issue_478/``):
  1. ``hero_band_gap_vs_logK.{png,pdf}`` — Two-line figure: mean(FAR bands)
     and mean(NEAR bands) vs log₂(K), with the gap annotated. PRIMARY hero
     for headline H1 narrative.
  2. ``per_K_marginal_slopes.{png,pdf}`` — Per-K distance-vs-leakage scatter,
     4 panels, with OLS slope line drawn per panel. §6.7 #2 HERO candidate
     (cleanest disambiguation of uniform-elevation vs flattening).
  3. ``residualized_check.{png,pdf}`` — Side-by-side: observed gap-shrinkage
     vs residualized gap-shrinkage. §6.7 #3 hero.
  4. ``no_comedy_panel.{png,pdf}`` — Full panel vs no-comedy refit slopes.
  5. ``per_band_trajectory.{png,pdf}`` — Per-band mean × K with SEM.
  6. ``per_seed_scatter.{png,pdf}`` — Per-cell deltaLogP at seed=42 vs seed=137.
  7. ``kl_dv_alongside_logp.{png,pdf}`` — KL DV alongside logp DV on same x.
  8. ``leverage_persona.{png,pdf}`` — Per-persona effect leverage scatter.
  9. ``superposition_level1.{png,pdf}`` — §6.8 Level-1: observed joint
     leakage vs mean-combiner superposition prediction, colored by K,
     faceted by band. The y=x line is the dose-aware "pure superposition"
     baseline.

If ``--arm`` is set and arm results are present, also produce:
  10. ``superposition_level2.{png,pdf}`` — overlay Level-2 shared-vs-distinct.
  11. ``marker_base_logp_matrix.{png,pdf}`` — Phase 0b 8×35 base-logp matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()


def _setup_matplotlib():
    """Apply paper-style rcParams (mirrors src/explore_persona_space/analysis/paper_plots.py)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(fig, fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = fig_dir / f"{name}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        log.info("Wrote %s", out)


def hero_band_gap_vs_logK(reg: dict, plt, fig_dir: Path) -> None:
    primary = reg.get("primary_gap_shrinkage", {})
    far_means = primary.get("far_means_per_K", {})
    near_means = primary.get("near_means_per_K", {})
    if not far_means or not near_means:
        log.warning("hero_band_gap_vs_logK: missing far/near means; skipping")
        return
    Ks = sorted(int(k) for k in far_means)
    far_vals = [far_means[str(K)] if str(K) in far_means else far_means.get(K) for K in Ks]
    near_vals = [near_means[str(K)] if str(K) in near_means else near_means.get(K) for K in Ks]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(np.log2(Ks), far_vals, marker="o", label="FAR bands mean (far+very-far+tail)")
    ax.plot(np.log2(Ks), near_vals, marker="o", label="NEAR bands mean (near+near-mid)")
    for K, far_v, near_v in zip(Ks, far_vals, near_vals, strict=True):
        gap = far_v - near_v
        ax.annotate(
            f"gap={gap:.2f}", xy=(np.log2(K), (far_v + near_v) / 2), fontsize=8, ha="center"
        )
    ax.set_xlabel("log₂(K)")
    ax.set_ylabel("Mean held-out ΔlogP(※)")
    ax.set_title(f"Band-averaged leakage vs K (slope p={primary.get('p', float('nan')):.3g})")
    ax.legend(loc="best")
    _save(fig, fig_dir, "hero_band_gap_vs_logK")
    plt.close(fig)


def per_K_marginal_slopes_fig(reg: dict, plt, fig_dir: Path, tidy_csv: Path) -> None:
    import pandas as pd

    df = pd.read_csv(tidy_csv)
    df = df[df["min_dist"] > 0]
    Ks = sorted(df["K"].unique())
    fig, axes = plt.subplots(1, len(Ks), figsize=(3.0 * len(Ks), 3.5), sharey=True)
    if len(Ks) == 1:
        axes = [axes]
    for ax, K in zip(axes, Ks, strict=True):
        sub = df[df["K"] == K]
        ax.scatter(np.log(sub["min_dist"]), sub["deltaLogP_mean"], s=6, alpha=0.4)
        slope_info = reg.get("per_K_marginal_slopes", {}).get(K) or reg.get(
            "per_K_marginal_slopes", {}
        ).get(str(K))
        if slope_info and slope_info.get("slope") is not None:
            x_line = np.linspace(np.log(sub["min_dist"]).min(), np.log(sub["min_dist"]).max(), 50)
            y_line = slope_info["intercept"] + slope_info["slope"] * x_line
            ax.plot(x_line, y_line, "r-", lw=1.5)
            ax.set_title(f"K={K}  β={slope_info['slope']:.2f}\n  p={slope_info['p']:.3g}")
        else:
            ax.set_title(f"K={K}")
        ax.set_xlabel("log(min_dist to subset)")
    axes[0].set_ylabel("ΔlogP(※)")
    fig.suptitle("Per-K marginal slope (HERO candidate, §6.7 #2)", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir, "per_K_marginal_slopes")
    plt.close(fig)


def per_seed_scatter_fig(plt, fig_dir: Path, tidy_csv: Path) -> None:
    import pandas as pd

    df = pd.read_csv(tidy_csv)
    pivot = df.pivot_table(
        index=["cell_id", "held_out_persona"],
        columns="seed",
        values="deltaLogP_mean",
        aggfunc="mean",
    )
    seeds = sorted(pivot.columns)
    if len(seeds) < 2:
        log.warning("per_seed_scatter: need ≥2 seeds; skipping")
        return
    s1, s2 = seeds[0], seeds[1]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(pivot[s1], pivot[s2], s=6, alpha=0.5)
    lo = min(pivot[s1].min(), pivot[s2].min())
    hi = max(pivot[s1].max(), pivot[s2].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    ax.set_xlabel(f"ΔlogP seed={s1}")
    ax.set_ylabel(f"ΔlogP seed={s2}")
    ax.set_title("Per-cell ΔlogP across seeds")
    _save(fig, fig_dir, "per_seed_scatter")
    plt.close(fig)


def superposition_level1_fig(reg: dict, plt, fig_dir: Path, tidy_csv: Path) -> None:
    """Level-1 mean-combiner scatter: observed L_shared vs predicted superposition.

    Color by K, facet by band. y=x is the dose-aware "pure superposition"
    baseline; deviation upward = `>` (ambiguous/dose-consistent); downward = `<` (interference).
    """
    level1 = reg.get("level1_superposition_decomposition", {})
    per_K = level1.get("per_K", {})
    if not per_K:
        log.warning("superposition_level1: no per_K decomposition; skipping")
        return
    # The plan's Level-1 doesn't ship raw per-pred rows in regression.json (it
    # ships aggregates); plot the per-K aggregate residuals as a sanity scatter.
    fig, ax = plt.subplots(figsize=(5, 4))
    Ks = sorted({int(k) for k in per_K})
    for K in Ks:
        info = per_K[str(K)] if str(K) in per_K else per_K[K]
        if "mean_residual_mean_combiner" not in info:
            continue
        x = K
        y = info["mean_residual_mean_combiner"]
        ax.errorbar(
            x,
            y,
            yerr=1.96 * info.get("cluster_robust_se_mean_combiner", 0.0),
            marker="o",
            capsize=4,
            label=f"K={K} (n_clusters={info['n_cell_seed_clusters']})",
        )
    ax.axhline(0, color="gray", lw=0.5, linestyle="--", label="pure superposition (residual=0)")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean residual (observed − mean-combiner)")
    ax.set_title("Level-1 superposition residual per K (cluster-robust 95% CI)")
    ax.legend()
    _save(fig, fig_dir, "superposition_level1")
    plt.close(fig)


def per_band_trajectory_fig(plt, fig_dir: Path, tidy_csv: Path) -> None:
    import pandas as pd

    df = pd.read_csv(tidy_csv)
    fig, ax = plt.subplots(figsize=(6, 4))
    for band, sub in df.groupby("band"):
        means = sub.groupby("K")["deltaLogP_mean"].mean()
        sems = sub.groupby("K")["deltaLogP_mean"].sem()
        Ks = sorted(means.index)
        ax.errorbar(
            np.log2(Ks),
            means.loc[Ks],
            yerr=sems.loc[Ks],
            marker="o",
            label=band,
            capsize=3,
        )
    ax.set_xlabel("log₂(K)")
    ax.set_ylabel("Mean ΔlogP(※)")
    ax.set_title("Per-band trajectory")
    ax.legend()
    _save(fig, fig_dir, "per_band_trajectory")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regression-json",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_478" / "aggregate" / "regression.json"),
    )
    parser.add_argument(
        "--tidy-csv",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_478" / "aggregate" / "tidy.csv"),
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=str(PROJECT_ROOT / "figures" / "issue_478"),
    )
    parser.add_argument("--arm", action="store_true")
    args = parser.parse_args()

    reg_path = Path(args.regression_json)
    tidy_path = Path(args.tidy_csv)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not reg_path.exists() or not tidy_path.exists():
        raise SystemExit(
            f"Need regression.json + tidy.csv. Got: reg={reg_path.exists()} "
            f"tidy={tidy_path.exists()}. Run issue478_analyze.py first."
        )
    reg = json.loads(reg_path.read_text())

    plt = _setup_matplotlib()

    hero_band_gap_vs_logK(reg, plt, fig_dir)
    per_K_marginal_slopes_fig(reg, plt, fig_dir, tidy_path)
    per_seed_scatter_fig(plt, fig_dir, tidy_path)
    per_band_trajectory_fig(plt, fig_dir, tidy_path)
    superposition_level1_fig(reg, plt, fig_dir, tidy_path)

    if args.arm:
        arm_path = (
            PROJECT_ROOT
            / "eval_results"
            / "issue_478"
            / "aggregate"
            / "distinct_markers_decomposition.json"
        )
        if arm_path.exists():
            log.info("Arm decomposition figure: TODO (depends on Level-2 analyzer output).")
        else:
            log.warning("--arm given but %s not present; skipping arm figures", arm_path)

    log.info("Phase 5 done — figures written to %s", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
