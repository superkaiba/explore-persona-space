#!/usr/bin/env python3
"""Issue #545 metric-race figures (plan §6 Figures; paper-plots conventions).

Reads the metric_race/ scoring JSONs and emits the hero candidates + the
exploratory diagnostic dump to ``figures/issue_545/metric_race/``. CPU-only,
runs OFF-POD on the VM after the pod is terminated.

HERO 1 — expanded dev leaderboard: per-predictor weighted-tau bars grouped by
metric family, the v1 raw-centroid reference band shaded.
HERO 2 — predictor-family x behavior-family tau heatmap, H2/H3 permutation p
annotated in the title.
Diagnostics — per-family champion table (bars + bootstrap CI), the optimism-gain
permutation null histogram, the LFO-mean bootstrap CI.

No plot annotations/arrows (project convention). Error bars clamped non-negative
(constant-bootstrap float-epsilon-negative-width guard).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis import paper_plots as pp  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("issue545.plot_metric_race")

FIG_DIR = "figures/issue_545/metric_race"

# Metric-family display order + role colors (valid roles: accent/baseline/
# control/neutral/primary).
FAMILY_ORDER = [
    "raw_centroid",
    "covariance_centroid",
    "centered_centroid",
    "cloud",
    "outdist_jskl",
    "group_b",
    "group_c",
    "group_d",
]
FAMILY_ROLE = {
    "raw_centroid": "baseline",
    "covariance_centroid": "primary",
    "centered_centroid": "accent",
    "cloud": "primary",
    "outdist_jskl": "accent",
    "group_b": "control",
    "group_c": "control",
    "group_d": "neutral",
    "other_A": "neutral",
}


def _color(fam: str) -> str:
    return pp.paper_palette_role(FAMILY_ROLE.get(fam, "neutral"))


def _clamp_yerr(point: float, lo: float, hi: float) -> tuple[float, float]:
    """Non-negative errorbar half-widths (constant-bootstrap epsilon guard)."""
    return max(0.0, point - lo), max(0.0, hi - point)


def hero1_leaderboard(scoring: dict, fig_dir: Path) -> None:
    lb = scoring.get("dev_leaderboard", {})
    if not lb:
        logger.warning("no dev_leaderboard — skipping hero1")
        return
    # Top-N per metric family (keep the plot readable).
    by_fam: dict[str, list[tuple[str, float]]] = {}
    for name, rec in lb.items():
        by_fam.setdefault(rec["metric_family"], []).append((name, rec["tau"]))
    rows = []
    for fam in FAMILY_ORDER:
        items = sorted(by_fam.get(fam, []), key=lambda kv: -kv[1])[:6]
        for name, tau in items:
            rows.append((fam, name, tau))
    if not rows:
        return
    pp.set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(rows))))
    ys = np.arange(len(rows))
    taus = [r[2] for r in rows]
    colors = [_color(r[0]) for r in rows]
    ax.barh(ys, taus, color=colors)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"[{r[0]}] {r[1].split('__', 1)[-1]}" for r in rows], fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0.0, color="0.5", lw=0.8)
    # v1 raw-centroid reference band: the span of raw-centroid dev taus.
    raw = [t for f, _n, t in rows if f == "raw_centroid"]
    if raw:
        ax.axvspan(min(raw), max(raw), color="0.85", alpha=0.5, zorder=0)
    ax.set_xlabel("dev weighted Kendall tau (z-normed shift target)")
    pp.set_title_subtitle(
        ax,
        "Expanded predictor-metric race (dev leaderboard)",
        "top-6 per metric family; grey band = v1 raw-centroid reference span",
    )
    out = pp.savefig_paper(fig, "hero1_dev_leaderboard", dir=str(fig_dir))
    plt.close(fig)
    logger.info("wrote %s", out)


def hero2_heatmap(per_family: dict, fig_dir: Path) -> None:
    hm = per_family.get("predictor_family_x_behavior_family_heatmap", {})
    if not hm:
        logger.warning("no heatmap — skipping hero2")
        return
    mfams = [f for f in FAMILY_ORDER if f in hm] + [f for f in hm if f not in FAMILY_ORDER]
    bfams = sorted({b for row in hm.values() for b in row})
    M = np.full((len(mfams), len(bfams)), np.nan)
    for i, mf in enumerate(mfams):
        for j, bf in enumerate(bfams):
            v = hm.get(mf, {}).get(bf)
            if v is not None:
                M[i, j] = v
    h3 = per_family.get("h3_optimism_gain", {})
    p_h3 = h3.get("permutation", {}).get("p_value")
    pp.set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(bfams)), max(4, 0.5 * len(mfams))))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(bfams)))
    ax.set_xticklabels(bfams, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(mfams)))
    ax.set_yticklabels(mfams, fontsize=7)
    fig.colorbar(im, ax=ax, label="best dev tau in family")
    pp.set_title_subtitle(
        ax,
        "Predictor family x behavior family (best dev tau)",
        f"H3 optimism-gain permutation p = {p_h3 if p_h3 is not None else 'n/a'}",
    )
    out = pp.savefig_paper(fig, "hero2_predictor_x_behavior_heatmap", dir=str(fig_dir))
    plt.close(fig)
    logger.info("wrote %s", out)


def diag_per_family(per_family: dict, fig_dir: Path) -> None:
    champs = per_family.get("per_family_champions", {})
    if not champs:
        return
    fams = sorted(champs)
    points, los, his, elig = [], [], [], []
    for fam in fams:
        rec = champs[fam]
        t = rec.get("pooled_tau")
        points.append(t if t is not None else np.nan)
        ci = rec.get("bootstrap_ci95")
        if ci and t is not None:
            lo, hi = ci
            los.append(lo)
            his.append(hi)
        else:
            los.append(t if t is not None else np.nan)
            his.append(t if t is not None else np.nan)
        elig.append(rec.get("eligible", False))
    pp.set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(fams))
    yerr_lo, yerr_hi = [], []
    for pt, lo, hi in zip(points, los, his, strict=True):
        if np.isnan(pt):
            yerr_lo.append(0.0)
            yerr_hi.append(0.0)
        else:
            a, b = _clamp_yerr(pt, lo, hi)
            yerr_lo.append(a)
            yerr_hi.append(b)
    colors = [pp.paper_palette_role("primary" if e else "neutral") for e in elig]
    ax.bar(xs, points, color=colors)
    ax.errorbar(xs, points, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="0.2", capsize=3, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(fams, rotation=45, ha="right")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_ylabel("per-family argmax pooled tau")
    pp.set_title_subtitle(
        ax,
        "Per-family champion tau (within-family bootstrap CI)",
        "colored = eligible (>=4-held-out-cell floor); grey = descriptive-only",
    )
    out = pp.savefig_paper(fig, "diag_per_family_champion_tau", dir=str(fig_dir))
    plt.close(fig)
    logger.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description="issue #545 metric-race figures")
    ap.add_argument("--out-root", default="", help="EPM_OUTPUT_ROOT override")
    args = ap.parse_args()
    if args.out_root:
        os.environ["EPM_OUTPUT_ROOT"] = args.out_root

    from explore_persona_space.experiments.behavior_testbed_545 import output_root

    mr = output_root() / "metric_race" / "scoring_metric_race"
    scoring_path = mr / "scoring_metric_race.json"
    per_family_path = mr / "per_family_and_heterogeneity.json"
    if not scoring_path.exists():
        sys.exit(f"missing {scoring_path} — run --phase score first")
    scoring = json.loads(scoring_path.read_text())
    per_family = json.loads(per_family_path.read_text()) if per_family_path.exists() else {}

    fig_dir = PROJECT_ROOT / FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    hero1_leaderboard(scoring, fig_dir)
    hero2_heatmap(per_family, fig_dir)
    diag_per_family(per_family, fig_dir)
    logger.info("[phase=done] figures written to %s", fig_dir)


if __name__ == "__main__":
    main()
