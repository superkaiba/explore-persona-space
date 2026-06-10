# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ρ/ΔG intentional
#!/usr/bin/env python3
"""Generate paper-quality figures for the #534 clean-result body (plan §6).

Hero (Fig 1): partial-Spearman ρ vs trajectory fraction (x = realized step;
one line per headline predictor shadow_angle + d_nn, d_source as context),
Holm-significance markers, per-seed thin lines behind the pooled line,
#504's saturated values and #530's frac=1.00 committed values as reference
points. Unusable fractions greyed + annotated, never plotted as ordinary
points.

Exploratory dump (over-produced; the analyzer picks the hero):
  Fig 2 — per-step source-self ΔG ramp per cell (full snapshot resolution,
          band + stop step overlaid).
  Fig 3 — per-fraction raw scatters of held-out ΔG vs each positional
          predictor (raw alongside the partialled hero, per standing rule).
  Fig 4 — per-fraction bystander-resolution panel (median bystander
          log P(marker) + argmax share, gate thresholds overlaid).
  Fig 5 — Δlog P vs Δz_marker agreement scatter per fraction (saturation
          localizer; skipped + flagged when the z fields are absent).
  Fig 6 — replication side-by-side bars (#530 frac=1.00 vs #534 banded
          frac=1.00, per seed).
  Fig 7 — per-cell held-out ΔG histograms per fraction (dynamic-range read,
          analyzer note #11).

Run after i534_trajectory_analyze.py:
    uv run python scripts/issue534_make_figures.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

log = logging.getLogger("issue534_make_figures")

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADLINE = ["shadow_angle", "d_nearest_neg_nd"]
HEADLINE_LABELS = {
    "shadow_angle": "Shadow-angle predictor",
    "d_nearest_neg_nd": "Distance-to-nearest-negative predictor",
    "d_source": "Distance-to-source predictor",
}
# Reference points (committed values; #504 from its clean-result, #530 from
# analysis_v1.json — re-read at runtime when available).
REF_504 = {"shadow_angle": 0.335, "d_nearest_neg_nd": -0.342}

# Plain-English legend labels for the 10 cell slugs (paper-plots §3.5: no
# Hydra/config slugs on rendered figures).
ARM_LABELS = {
    "near": "Near negative (con artist)",
    "mid_near": "Mid-near negative (origami artist)",
    "mid_far": "Mid-far negative (meditation teacher)",
    "far": "Far negative (prosecutor)",
    "default_only": "Default-only floor",
}


def _cell_label(slug: str) -> str:
    """Map a cell slug like c504v3_mid_near_seed137 to plain English."""
    stem = slug.removeprefix("c504v3_")
    arm, _, seed = stem.rpartition("_seed")
    return f"{ARM_LABELS.get(arm, arm)}, seed {seed}"


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def fig1_hero_rho_vs_fraction(analysis: dict, out_dir: Path) -> None:
    """Hero: pooled partial ρ per fraction, per-seed thin lines, references."""
    set_paper_style("blog")
    per_frac = analysis["per_fraction"]
    fracs = sorted(per_frac, key=float)
    # x = realized step per fraction (mode of the pool's distinct steps).
    xs: list[float] = []
    for fs in fracs:
        steps = per_frac[fs].get("distinct_training_steps_in_pool") or []
        xs.append(float(steps[0]) if steps else float(fs) * 20.0)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    colors = {
        "shadow_angle": paper_palette_role("primary"),
        "d_nearest_neg_nd": paper_palette_role("accent"),
        "d_source": paper_palette_role("neutral"),
    }
    for pred in [*HEADLINE, "d_source"]:
        rhos, sig = [], []
        for fs in fracs:
            fit = per_frac[fs]["pooled_fit"]
            part = fit.get("partial_spearman", {}).get(pred)
            rhos.append(float(part["rho"]) if part else np.nan)
            sig.append(bool(fit.get("holm", {}).get(pred, {}).get("reject_null", False)))
        lw = 2.2 if pred in HEADLINE else 1.2
        ls = "-" if pred in HEADLINE else "--"
        ax.plot(xs, rhos, ls, color=colors[pred], linewidth=lw, label=HEADLINE_LABELS[pred])
        for x, r, s, fs in zip(xs, rhos, sig, fracs, strict=True):
            usable = per_frac[fs]["usability"]["usable"]
            face = colors[pred] if s else "white"
            edge = colors[pred]
            if not usable:
                face, edge = "lightgrey", "grey"
            ax.scatter([x], [r], s=64 if s else 40, facecolor=face, edgecolor=edge, zorder=5)
        # Per-seed thin lines behind the pooled line (headline preds only).
        if pred in HEADLINE:
            seeds = analysis.get("seeds", [])
            for seed in seeds:
                seed_rhos = []
                for fs in fracs:
                    psf = per_frac[fs].get("per_seed_fit", {}).get(str(seed)) or per_frac[fs].get(
                        "per_seed_fit", {}
                    ).get(seed)
                    if psf is None:
                        seed_rhos.append(np.nan)
                        continue
                    part = psf.get("partial_spearman", {}).get(pred)
                    seed_rhos.append(float(part["rho"]) if part else np.nan)
                ax.plot(xs, seed_rhos, "-", color=colors[pred], alpha=0.25, linewidth=0.9)
    # Reference points at the final step. NOTE: the blog style zeroes
    # lines.markeredgewidth / patch.linewidth, which renders facecolor="none"
    # scatter markers invisible — pass linewidths explicitly.
    x_ref = xs[-1]
    for pred in HEADLINE:
        ax.scatter(
            [x_ref + 0.6],
            [REF_504[pred]],
            marker="s",
            s=46,
            facecolor="none",
            edgecolor=colors[pred],
            linewidths=1.4,
            label="Saturated-anchor reference (open squares)" if pred == HEADLINE[0] else None,
        )
        rep = analysis.get("replication_check", {}).get("per_predictor", {}).get(pred, {})
        if rep.get("rho_530") is not None:
            ax.scatter(
                [x_ref + 1.2],
                [rep["rho_530"]],
                marker="D",
                s=46,
                facecolor="none",
                edgecolor=colors[pred],
                linewidths=1.4,
                label="First de-saturated run at this anchor (open diamonds)" if pred == HEADLINE[0] else None,
            )
    # Grey annotation for unusable fractions.
    for x, fs in zip(xs, fracs, strict=True):
        if not per_frac[fs]["usability"]["usable"]:
            ax.axvspan(x - 0.4, x + 0.4, color="lightgrey", alpha=0.35, zorder=0)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("Realized training step at the selected fraction")
    ax.set_ylabel("Partial Spearman correlation with held-out leakage")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "trajectory_partial_rho", dir=out_dir)
    plt.close(fig)


def fig2_source_ramp(slab: Path, out_dir: Path) -> None:
    """Per-step source-self ΔG ramp per cell, band + stop overlaid."""
    set_paper_style("blog")
    paths = sorted(slab.glob("c504v3_*_seed*/source_steps_trajectory.json"))
    if not paths:
        log.warning("fig2: no source_steps_trajectory.json files under %s — skipped", slab)
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for p in paths:
        d = _load_json(p)
        steps = [r["step"] for r in d["steps"]]
        dgs = [r["delta_g_mean"] for r in d["steps"]]
        ax.plot(steps, dgs, "-", linewidth=1.2, alpha=0.85, label=_cell_label(p.parent.name))
    ax.axhspan(5.0, 12.0, color=paper_palette_role("accent"), alpha=0.12)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Source marker log-prob gain over base (nats)")
    ax.legend(frameon=False, fontsize=6, ncols=2)
    fig.tight_layout()
    savefig_paper(fig, "source_delta_g_ramp_per_cell", dir=out_dir)
    plt.close(fig)


def _frac_rows(slab: Path, frac: float, gates: dict) -> list[dict]:
    """Raw (probe x cell x seed) rows at one fraction for the raw scatters."""
    per_probe = gates["per_probe"]
    rows: list[dict] = []
    for p in sorted(slab.glob("c504v3_*_seed*/trajectory.json")):
        traj = _load_json(p)
        cell = traj["cell"]
        cks = traj.get("checkpoints", [])
        if not cks:
            continue
        ck = min(cks, key=lambda c: abs(float(c["frac"]) - frac))
        for probe, per_q in (ck.get("held_out", {}) or {}).items():
            if probe not in per_probe:
                continue
            dgs = [r.get("delta_g") for r in per_q.values() if r and r.get("delta_g") is not None]
            if not dgs:
                continue
            cov = per_probe[probe]
            d_nn = cov["d_nearest_neg_nd"].get(cell)
            shadow = cov["shadow_angle"].get(cell)
            rows.append(
                {
                    "delta_g": float(np.mean(dgs)),
                    "d_source": float(cov["d_source"]),
                    "d_nearest_neg_nd": float(d_nn) if d_nn is not None else np.nan,
                    "shadow_angle": float(shadow) if shadow is not None else np.nan,
                }
            )
    return rows


def fig3_raw_scatters(slab: Path, gates: dict, fracs: list[float], out_dir: Path) -> None:
    """Raw ΔG-vs-predictor scatters per fraction (raw alongside partialled)."""
    set_paper_style("blog")
    preds = ["d_source", "d_nearest_neg_nd", "shadow_angle"]
    fig, axes = plt.subplots(
        len(fracs), len(preds), figsize=(3.2 * len(preds), 2.6 * len(fracs)), squeeze=False
    )
    for i, f in enumerate(fracs):
        rows = _frac_rows(slab, f, gates)
        for j, pred in enumerate(preds):
            ax = axes[i][j]
            x = [r[pred] for r in rows]
            y = [r["delta_g"] for r in rows]
            ax.scatter(x, y, s=8, alpha=0.5, color=paper_palette_role("primary"))
            if i == len(fracs) - 1:
                ax.set_xlabel(HEADLINE_LABELS.get(pred, pred), fontsize=8)
            if j == 0:
                ax.set_ylabel(f"Fraction {f:.2f}\nheld-out gain (nats)", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    savefig_paper(fig, "raw_scatter_delta_g_vs_predictors_per_fraction", dir=out_dir)
    plt.close(fig)


def fig4_bystander_panel(slab: Path, out_dir: Path) -> None:
    """Per-fraction bystander-resolution panel from bystander_resolution.json."""
    set_paper_style("blog")
    paths = sorted(slab.glob("c504v3_*_seed*/bystander_resolution.json"))
    if not paths:
        log.warning("fig4: no bystander_resolution.json files under %s — skipped", slab)
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for p in paths:
        d = _load_json(p)
        pf = d.get("per_fraction", {})
        fr = sorted(pf, key=float)
        med = [pf[f]["de_saturation_gate"]["median_g_logp_at_post_response_slot"] for f in fr]
        share = [pf[f]["de_saturation_gate"]["argmax_marker_share_across_pairs"] for f in fr]
        xs = [float(f) for f in fr]
        lbl = _cell_label(p.parent.name)
        ax1.plot(xs, med, "-o", markersize=3, linewidth=1.0, alpha=0.8, label=lbl)
        ax2.plot(xs, share, "-o", markersize=3, linewidth=1.0, alpha=0.8)
    ax1.axhline(-2.0, color="grey", linestyle="--", linewidth=0.9)
    ax2.axhline(0.60, color="grey", linestyle="--", linewidth=0.9)
    ax1.set_xlabel("Trajectory fraction")
    ax1.set_ylabel("Median bystander marker log-prob (nats)")
    ax2.set_xlabel("Trajectory fraction")
    ax2.set_ylabel("Bystander argmax-marker share")
    ax1.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    savefig_paper(fig, "bystander_resolution_per_fraction", dir=out_dir)
    plt.close(fig)


def fig5_z_agreement(analysis: dict, slab: Path, out_dir: Path) -> None:
    """Δlog P vs Δz_marker agreement scatter per fraction (saturation localizer)."""
    set_paper_style("blog")
    per_frac = analysis["per_fraction"]
    fracs = sorted(per_frac, key=float)
    available = [f for f in fracs if per_frac[f]["z_agreement"].get("available")]
    if not available:
        log.warning("fig5: no z fields available in any fraction — skipped (logit column dropped)")
        return
    fig, axes = plt.subplots(1, len(available), figsize=(3.0 * len(available), 3.2), squeeze=False)
    for j, fs in enumerate(available):
        ax = axes[0][j]
        dg: list[float] = []
        dz: list[float] = []
        for p in sorted(slab.glob("c504v3_*_seed*/trajectory.json")):
            traj = _load_json(p)
            cks = traj.get("checkpoints", [])
            if not cks:
                continue
            ck = min(cks, key=lambda c: abs(float(c["frac"]) - float(fs)))
            for per_q in (ck.get("held_out", {}) or {}).values():
                for row in per_q.values():
                    if row and row.get("delta_z_marker") is not None:
                        dg.append(float(row["delta_g"]))
                        dz.append(float(row["delta_z_marker"]))
        ax.scatter(dz, dg, s=8, alpha=0.45, color=paper_palette_role("primary"))
        lim = max([abs(v) for v in dz + dg] or [1.0])
        ax.plot([-lim, lim], [-lim, lim], "--", color="grey", linewidth=0.8)
        ax.set_xlabel("Marker logit change (trained minus base)", fontsize=8)
        if j == 0:
            ax.set_ylabel("Marker log-prob change (nats)", fontsize=8)
        ax.set_title(f"Fraction {fs}", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    savefig_paper(fig, "delta_logp_vs_delta_z_per_fraction", dir=out_dir)
    plt.close(fig)


def fig6_replication_bars(analysis: dict, out_dir: Path) -> None:
    """Side-by-side bars: #530 frac=1.00 committed vs #534 banded frac=1.00."""
    set_paper_style("blog")
    rep = analysis.get("replication_check", {})
    if not rep.get("available"):
        log.warning("fig6: replication_check unavailable — skipped")
        return
    per_pred = rep["per_predictor"]
    preds = [p for p in HEADLINE if p in per_pred and per_pred[p].get("rho_534") is not None]
    if not preds:
        log.warning("fig6: no headline predictors in replication_check — skipped")
        return
    x = np.arange(len(preds))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    r530 = [per_pred[p]["rho_530"] for p in preds]
    r534 = [per_pred[p]["rho_534"] for p in preds]
    ax.bar(
        x - width / 2,
        r530,
        width,
        label="Previous run at the de-saturated anchor",
        color=paper_palette_role("baseline"),
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        r534,
        width,
        label="This re-run, final-fraction checkpoint",
        color=paper_palette_role("primary"),
        edgecolor="black",
        linewidth=0.6,
    )
    # Bootstrap CI whiskers when available.
    for k, p in enumerate(preds):
        ci_o = per_pred[p].get("ci_530") or {}
        ci_n = per_pred[p].get("ci_534") or {}
        if ci_o.get("lo") is not None:
            ax.errorbar(
                [k - width / 2],
                [r530[k]],
                yerr=[[r530[k] - ci_o["lo"]], [ci_o["hi"] - r530[k]]],
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=0.9,
            )
        if ci_n.get("lo") is not None:
            ax.errorbar(
                [k + width / 2],
                [r534[k]],
                yerr=[[r534[k] - ci_n["lo"]], [ci_n["hi"] - r534[k]]],
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=0.9,
            )
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([HEADLINE_LABELS[p] for p in preds], fontsize=9)
    # Two-line label: the single-line form is taller than the axes and the
    # blog-style margins clip it at the top/left edge of the saved PNG.
    ax.set_ylabel("Partial Spearman correlation\nwith held-out leakage")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "replication_530_vs_534_frac100", dir=out_dir)
    plt.close(fig)


def fig7_delta_g_hists(slab: Path, fracs: list[float], out_dir: Path) -> None:
    """Held-out ΔG histograms per fraction (pooled over cells; dynamic-range read)."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, len(fracs), figsize=(2.9 * len(fracs), 3.0), squeeze=False)
    for j, f in enumerate(fracs):
        vals: list[float] = []
        for p in sorted(slab.glob("c504v3_*_seed*/trajectory.json")):
            traj = _load_json(p)
            cks = traj.get("checkpoints", [])
            if not cks:
                continue
            ck = min(cks, key=lambda c: abs(float(c["frac"]) - f))
            for per_q in (ck.get("held_out", {}) or {}).values():
                for row in per_q.values():
                    if row and row.get("delta_g") is not None:
                        vals.append(float(row["delta_g"]))
        ax = axes[0][j]
        if vals:
            ax.hist(vals, bins=40, color=paper_palette_role("primary"), alpha=0.85)
        ax.set_title(f"Fraction {f:.2f}", fontsize=9)
        ax.set_xlabel("Held-out gain (nats)", fontsize=8)
        if j == 0:
            ax.set_ylabel("Probe-question pairs", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    savefig_paper(fig, "held_out_delta_g_hist_per_fraction", dir=out_dir)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--slab-root", type=Path, default=REPO_ROOT / "eval_results/issue_534")
    ap.add_argument(
        "--analysis-path",
        type=Path,
        default=None,
        help="Default <slab-root>/analysis_per_fraction.json.",
    )
    ap.add_argument(
        "--phase05-path",
        type=Path,
        default=REPO_ROOT / "eval_results/issue_530/phase0_5_gates.json",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures/issue_534")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [phase=figures_534] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    analysis_path = (
        args.analysis_path
        if args.analysis_path is not None
        else args.slab_root / "analysis_per_fraction.json"
    )
    analysis = _load_json(analysis_path)
    gates = _load_json(args.phase05_path)
    fracs = [float(f) for f in analysis["fractions"]]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig1_hero_rho_vs_fraction(analysis, args.out_dir)
    fig2_source_ramp(args.slab_root, args.out_dir)
    fig3_raw_scatters(args.slab_root, gates, fracs, args.out_dir)
    fig4_bystander_panel(args.slab_root, args.out_dir)
    fig5_z_agreement(analysis, args.slab_root, args.out_dir)
    fig6_replication_bars(analysis, args.out_dir)
    fig7_delta_g_hists(args.slab_root, fracs, args.out_dir)
    log.info("[phase=done] figures written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
