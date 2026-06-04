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


def residualized_check_fig(reg: dict, plt, fig_dir: Path) -> None:
    """§6.7 #3 hero — observed gap-shrinkage vs residualized gap-shrinkage.

    Side-by-side bars per K: observed gap vs gap-after-K=1-fitted-f(d) subtraction.
    Distance-flattening only survives if the residualized gap also shrinks.
    """
    primary = reg.get("primary_gap_shrinkage", {})
    resid = reg.get("robustness", {}).get("residualized_leakage_check", {})
    resid_gap = resid.get("residualized_gap_shrinkage", {})
    obs_gaps = primary.get("gaps_per_K") or {}
    res_gaps = resid_gap.get("gaps_per_K") or {}
    if not obs_gaps or not res_gaps:
        log.warning("residualized_check_fig: missing data — skipping")
        return
    Ks = sorted(int(k) for k in obs_gaps)
    obs_vals = [obs_gaps[k] if k in obs_gaps else obs_gaps[str(k)] for k in Ks]
    res_vals = [res_gaps[k] if k in res_gaps else res_gaps[str(k)] for k in Ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(Ks))
    width = 0.35
    ax.bar(x - width / 2, obs_vals, width, label="observed gap")
    ax.bar(x + width / 2, res_vals, width, label="residualized gap (− K=1 f(d))")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in Ks])
    ax.set_ylabel("FAR − NEAR gap (ΔlogP)")
    ax.set_title("§6.7 #3 — observed vs residualized gap-shrinkage")
    ax.legend()
    _save(fig, fig_dir, "residualized_check")
    plt.close(fig)


def no_comedy_panel_fig(reg: dict, plt, fig_dir: Path) -> None:
    """§6.7 #5 / §6.8 v5 — full-panel slope vs no-comedy slope with 95% CIs."""
    no_comedy = reg.get("robustness", {}).get("no_comedy_refit", {})
    full = no_comedy.get("full_panel", {})
    nc = no_comedy.get("no_comedy", {})
    if not full or not nc or full.get("slope") is None or nc.get("slope") is None:
        log.warning("no_comedy_panel_fig: missing data — skipping")
        return
    survival = no_comedy.get("survival", {})
    status = survival.get("status", "INDETERMINATE")
    full_slope = full["slope"]
    full_se = full.get("se", 0.0) or 0.0
    nc_slope = nc["slope"]
    nc_se = nc.get("se", 0.0) or 0.0
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(
        [0],
        [full_slope],
        yerr=1.96 * full_se,
        fmt="o",
        capsize=6,
        label="full panel",
    )
    ax.errorbar(
        [1],
        [nc_slope],
        yerr=1.96 * nc_se,
        fmt="s",
        capsize=6,
        label="no-comedy refit",
    )
    ax.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["full (35)", f"no-comedy ({no_comedy.get('n_personas_dropped', 9)} dropped)"]
    )
    ax.set_ylabel("gap-shrinkage slope (per log₂(K))")
    ax.set_title(f"§6.8 no-comedy survival: {status[:60]}")
    ax.legend()
    _save(fig, fig_dir, "no_comedy_panel")
    plt.close(fig)


def kl_dv_alongside_logp_fig(reg: dict, plt, fig_dir: Path) -> None:
    """§6.7 #1 + §6 KL-DV non-saturating proxy alongside the marker log-prob DV."""
    primary = reg.get("primary_gap_shrinkage", {})
    kl = reg.get("robustness", {}).get("kl_dv_refit", {})
    logp_gaps = primary.get("gaps_per_K") or {}
    kl_gaps = kl.get("gaps_per_K") or {}
    if not logp_gaps or not kl_gaps:
        log.warning("kl_dv_alongside_logp_fig: missing data — skipping")
        return
    Ks = sorted(int(k) for k in logp_gaps)
    logp_vals = [logp_gaps[k] if k in logp_gaps else logp_gaps[str(k)] for k in Ks]
    kl_vals = [kl_gaps[k] if k in kl_gaps else kl_gaps[str(k)] for k in Ks]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    ax1.plot(np.log2(Ks), logp_vals, marker="o", color="C0")
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_title(f"log P(※) DV  (p={primary.get('p', float('nan')):.3g})")
    ax1.set_xlabel("log₂(K)")
    ax1.set_ylabel("FAR − NEAR gap")
    ax2.plot(np.log2(Ks), kl_vals, marker="o", color="C1")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_title(f"KL DV (non-saturating)  (p={kl.get('p', float('nan')):.3g})")
    ax2.set_xlabel("log₂(K)")
    ax2.set_ylabel("FAR − NEAR gap")
    fig.tight_layout()
    _save(fig, fig_dir, "kl_dv_alongside_logp")
    plt.close(fig)


def leverage_persona_fig(reg: dict, plt, fig_dir: Path) -> None:
    """§6.7 #1 + §6 leverage scatter — per-persona effect on slope (leave-one-out).

    `leave_one_persona_out` returns `{persona: {gap_slope, gap_p, gap_K1, gap_K8}}`
    at the top of the `robustness.leave_one_persona_out` dict (the dict's keys
    are persona names directly — no nested per_persona/persona_results wrapper).
    """
    loo = reg.get("robustness", {}).get("leave_one_persona_out", {})
    if not loo:
        log.warning("leverage_persona_fig: leave-one-persona-out missing — skipping")
        return
    items: list[tuple[str, float]] = []
    for persona, payload in loo.items():
        if not isinstance(payload, dict):
            continue
        sl = payload.get("gap_slope") or payload.get("slope")
        if sl is not None:
            items.append((persona, float(sl)))
    if not items:
        log.warning("leverage_persona_fig: no per-persona slope payloads — skipping")
        return
    items.sort(key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.18 * len(items))))
    ax.barh(range(len(items)), [v for _, v in items])
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([k for k, _ in items], fontsize=7)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Gap-shrinkage slope when persona dropped")
    ax.set_title("Leave-one-persona-out leverage on gap-shrinkage slope")
    _save(fig, fig_dir, "leverage_persona")
    plt.close(fig)


def arm_level2_fig(arm_payload: dict, plt, fig_dir: Path) -> None:
    """§6.8 Level-2 arm — direction-agreement counts + per-K mean gap CI."""
    level2 = arm_payload.get("level2_decomposition", {})
    direction = level2.get("direction_agreement_per_K") or {}
    bootstrap = level2.get("paired_bootstrap_per_K_mean_gap") or {}
    if not direction:
        log.warning("arm_level2_fig: no direction_agreement_per_K data; skipping")
        return
    Ks = sorted(int(k) for k in direction)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    # (a) stacked direction counts per K.
    n_gt = [direction[str(k) if str(k) in direction else k]["n_shared_gt_distinct"] for k in Ks]
    n_lt = [direction[str(k) if str(k) in direction else k]["n_shared_lt_distinct"] for k in Ks]
    n_eq = [direction[str(k) if str(k) in direction else k].get("n_zero_or_noise", 0) for k in Ks]
    x = np.arange(len(Ks))
    width = 0.6
    ax1.bar(x, n_gt, width, label="shared > distinct (AMBIGUOUS / dose-consistent)")
    ax1.bar(x, n_eq, width, bottom=n_gt, label="≈ (SUPERPOSITION)")
    ax1.bar(
        x,
        n_lt,
        width,
        bottom=[a + b for a, b in zip(n_gt, n_eq, strict=True)],
        label="shared < distinct (INTERFERENCE)",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"K={k}" for k in Ks])
    ax1.set_ylabel("# matched pairs (per K)")
    ax1.set_title("Level-2 direction-agreement counts")
    ax1.legend(loc="best", fontsize=7)
    # (b) bootstrap CI per K.
    means: list[float] = []
    los: list[float] = []
    his: list[float] = []
    for k in Ks:
        b = bootstrap.get(str(k)) or bootstrap.get(k) or {}
        m = b.get("mean_gap")
        ci = b.get("bootstrap_ci95") or [None, None]
        if m is not None and ci[0] is not None and ci[1] is not None:
            means.append(float(m))
            los.append(float(m) - float(ci[0]))
            his.append(float(ci[1]) - float(m))
        else:
            means.append(float("nan"))
            los.append(0.0)
            his.append(0.0)
    ax2.errorbar(x, means, yerr=[los, his], fmt="o", capsize=6)
    ax2.axhline(0, color="gray", lw=0.5, linestyle="--", label="superposition (gap = 0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"K={k}" for k in Ks])
    ax2.set_ylabel("L_shared − superposition(L_distinct)  (mean ± 95% CI)")
    ax2.set_title("Level-2 paired bootstrap (mean combiner)")
    ax2.legend()
    fig.tight_layout()
    _save(fig, fig_dir, "arm_level2_decomposition")
    plt.close(fig)


def arm_marker_base_logp_matrix_fig(arm_payload: dict, plt, fig_dir: Path) -> None:
    """Phase 0b 8×35 base-logp matrix (marker × held-out persona) heatmap."""
    base_logp = arm_payload.get("phase_0b_marker_base_logp")
    if not base_logp:
        log.warning("arm_marker_base_logp_matrix_fig: no Phase 0b base-logp; skipping")
        return
    # Accept the three on-disk shapes the Phase-0b probe / arm payload can carry:
    #   (1) {"matrix_marker_x_persona_meanlogp": {marker: {persona: logp}}, ...}
    #       — the actual issue478_validate_markers.py schema.
    #   (2) {markers: [...], personas: [...], matrix: [[...]]}.
    #   (3) bare {marker: {persona: logp}}.
    nested = base_logp.get("matrix_marker_x_persona_meanlogp")
    if isinstance(nested, dict):
        markers = sorted(nested.keys())
        personas = sorted({p for v in nested.values() if isinstance(v, dict) for p in v})
        mat = np.array([[nested[m].get(p, np.nan) for p in personas] for m in markers])
    elif "matrix" in base_logp and "markers" in base_logp and "personas" in base_logp:
        markers = base_logp["markers"]
        personas = base_logp["personas"]
        mat = np.array(base_logp["matrix"])
    else:
        # Bare {marker: {persona: logp}} — ignore any scalar metadata keys.
        markers = sorted(k for k, v in base_logp.items() if isinstance(v, dict))
        personas = sorted({p for k in markers for p in base_logp[k]})
        mat = np.array([[base_logp[m].get(p, np.nan) for p in personas] for m in markers])
    if not markers or not personas:
        log.warning("arm_marker_base_logp_matrix_fig: empty marker/persona axis; skipping")
        return
    fig, ax = plt.subplots(
        figsize=(min(12, max(6, 0.25 * len(personas))), max(3, 0.4 * len(markers)))
    )
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(personas)))
    ax.set_xticklabels(personas, rotation=90, fontsize=6)
    ax.set_yticks(range(len(markers)))
    ax.set_yticklabels(markers)
    ax.set_title("Phase 0b: per-marker base log P at post-response slot (held-out personas)")
    fig.colorbar(im, ax=ax, label="base log P(marker)")
    fig.tight_layout()
    _save(fig, fig_dir, "marker_base_logp_matrix")
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
    # Round-2 MAJOR 6: §6.7/§6.8-mandated analyzer figures (previously omitted /
    # TODO). The Phase-5 contract advertises all 9 in the docstring; this
    # implements 6 of the 7 core ones + both arm figures. (The exploratory
    # per-cell error-bar grid stays out — it's a dump that doesn't change
    # narrative if missing; everything LOAD-BEARING for §6.7/§6.8 is here.)
    residualized_check_fig(reg, plt, fig_dir)
    no_comedy_panel_fig(reg, plt, fig_dir)
    kl_dv_alongside_logp_fig(reg, plt, fig_dir)
    leverage_persona_fig(reg, plt, fig_dir)

    if args.arm:
        arm_path = (
            PROJECT_ROOT
            / "eval_results"
            / "issue_478"
            / "aggregate"
            / "distinct_markers_decomposition.json"
        )
        if arm_path.exists():
            arm_payload = json.loads(arm_path.read_text())
            arm_level2_fig(arm_payload, plt, fig_dir)
            arm_marker_base_logp_matrix_fig(arm_payload, plt, fig_dir)
        else:
            log.warning("--arm given but %s not present; skipping arm figures", arm_path)

    log.info("Phase 5 done — figures written to %s", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
