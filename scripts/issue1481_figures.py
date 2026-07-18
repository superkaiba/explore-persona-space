#!/usr/bin/env python
"""#1481 figures driver (plan §6 "Figures (over-produce)") — VM-side.

Consumes ONLY the analysis JSONs ``issue1481_analysis.py`` writes
(``verdict_manifest.json``, ``regime_contrast_content.json``,
``regime_contrast_marker.json``, ``margin_rate_validation.json``) and renders
the registered families under ``--fig-dir`` via the project paper-plot
conventions (``set_paper_style`` / ``paper_palette`` / ``savefig_paper``:
colorblind-safe, error bars, no annotation overlays).

Families (``--families`` subsets; default all that have inputs):

- ``hero1_forest``        — behavior × context forest of D at matched install
  (Newcombe CI bars, dose-match/discordance flags) + the marker (nats) panel.
- ``hero1_percell``       — RAW companion: per-(read context, seed) D points.
- ``hero2_marker_map``    — emission-onset vs selectivity-break map per (LR,
  step), con vs po overlaid.
- ``tier1_ladders``       — per-cell Tier-1 ladders (all rungs × 3 LR, both
  regimes) with the [0.60, 0.85] band.
- ``panel_heatmaps``      — 6×N trained-rate heatmaps per behavior + the
  shared base-panel companion.
- ``per_seed_scatter``    — seed-42 vs seed-137 D scatter with discordance
  flags.
- ``install_bars``        — con vs po verdict-arm install rates ± Wilson CI.
- ``heldout_decomp``      — held-out-only vs pooled D paired bars.
- ``marker_three_space``  — Δlog P vs Δz divergence scatter (+ table JSON).
- ``margin_rate``         — TF-margin vs Tier-1 rate validation scatter.

Every error-bar call routes through ``_err_offsets`` (non-negative per-point
offsets clamped at 0 — matplotlib rejects negative ``xerr``/``yerr``, and
tiny-n quantile CIs can invert around the point estimate; gotchas.md).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue1481.figures")

BEHAVIOR_LABEL = {"cas": "Casual style", "imp": "Impolite", "syc": "Sycophancy"}
CTX_LABEL = {"pers": "Persona", "bare": "Bare", "conv": "WildChat prefix", "icl": "ICL prefix"}
REGIME_LABEL = {"con": "Contrastive", "po": "Positive-only"}


def _err_offsets(vals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """CI bounds → NON-NEGATIVE per-point (lower, upper) offsets (gotchas.md:
    matplotlib rejects negative xerr/yerr; tiny-n quantile CIs can invert)."""
    return np.vstack([np.maximum(0.0, vals - lo), np.maximum(0.0, hi - vals)])


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# ── Families ─────────────────────────────────────────────────────────────────


def fig_hero1_forest(analysis: dict, fig_dir: Path) -> None:
    """Hero 1: behavior × context forest of D at matched install (plan §6)."""
    content = analysis["content"]
    marker = analysis.get("marker")
    ncols = 2 if marker else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.6), squeeze=False)
    ax = axes[0][0]
    rows = []
    for beh_key, beh in sorted(content["behavior_contexts"].items()):
        for ctx_key, cell in sorted(beh.items()):
            blk = cell["pooled"]
            if blk["status"] != "computed":
                continue
            rows.append(
                (
                    f"{BEHAVIOR_LABEL[beh_key]} / {CTX_LABEL[ctx_key]}",
                    blk["D"],
                    blk["newcombe_95"][0],
                    blk["newcombe_95"][1],
                    cell["dose_matched"],
                    bool(cell.get("sign_discordant")),
                )
            )
    if not rows:
        raise RuntimeError("[i1481-figures] hero1: no computed pooled cells")
    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows])
    y = np.arange(len(rows))[::-1]
    pal = paper_palette(3)
    colors = [pal[0] if r[4] else pal[1] for r in rows]
    ax.errorbar(vals, y, xerr=_err_offsets(vals, lo, hi), fmt="none", ecolor="0.4", capsize=2, lw=1)
    for yi, v, c, r in zip(y, vals, colors, rows, strict=True):
        ax.scatter([v], [yi], color=c, marker="D" if r[5] else "o", zorder=3, s=28)
    ax.axvline(0.0, color="0.75", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("D = positive-only − contrastive non-source rate")
    ax.set_title("Content leakage contrast at matched install")
    if marker:
        axm = axes[0][1]
        mrows = []
        for ctx_key, cell in sorted(marker["contexts"].items()):
            blk = cell["pooled_nonsource"]
            mrows.append(
                (
                    CTX_LABEL.get(ctx_key, ctx_key),
                    blk["D_nats"],
                    blk["bootstrap_95"][0],
                    blk["bootstrap_95"][1],
                    cell["dose_matched"],
                )
            )
        mlabels = [r[0] for r in mrows]
        mvals = np.array([r[1] for r in mrows])
        mlo = np.array([r[2] for r in mrows])
        mhi = np.array([r[3] for r in mrows])
        my = np.arange(len(mrows))[::-1]
        mcolors = [pal[0] if r[4] else pal[1] for r in mrows]
        axm.errorbar(
            mvals,
            my,
            xerr=_err_offsets(mvals, mlo, mhi),
            fmt="none",
            ecolor="0.4",
            capsize=2,
            lw=1,
        )
        axm.scatter(mvals, my, color=mcolors, zorder=3, s=28)
        axm.axvline(0.0, color="0.75", lw=0.8)
        axm.set_yticks(my)
        axm.set_yticklabels(mlabels, fontsize=7)
        axm.set_xlabel("Marker D (nats, question-cluster bootstrap 95%)")
        axm.set_title("Marker leakage contrast")
    fig.tight_layout()
    savefig_paper(fig, "hero1_forest_matched_install", dir=fig_dir)
    plt.close(fig)


def fig_hero1_percell(analysis: dict, fig_dir: Path) -> None:
    """RAW companion to Hero 1: per-(read context, seed) D points (the
    low-level per-unit data behind every pooled aggregate)."""
    content = analysis["content"]
    beh_keys = sorted(content["behavior_contexts"])
    fig, axes = plt.subplots(1, len(beh_keys), figsize=(3.4 * len(beh_keys), 3.2), squeeze=False)
    pal = paper_palette(4)
    for ax, beh_key in zip(axes[0], beh_keys, strict=True):
        beh = content["behavior_contexts"][beh_key]
        labels, xs = [], []
        for i, (ctx_key, cell) in enumerate(sorted(beh.items())):
            labels.append(CTX_LABEL[ctx_key])
            for blk in cell["per_context"]:
                if blk["status"] != "computed":
                    continue
                ax.scatter(
                    [i + (0.12 if blk["is_heldout"] else -0.12)],
                    [blk["D"]],
                    color=pal[3] if blk["is_heldout"] else pal[0],
                    s=16,
                    alpha=0.85,
                )
            for seed_s, srec in sorted(cell["per_seed"].items()):
                if srec["status"] != "computed":
                    continue
                ax.scatter(
                    [i],
                    [srec["D"]],
                    color=pal[1],
                    marker="x",
                    s=30,
                    label=f"seed {seed_s}" if i == 0 else None,
                )
            xs.append(i)
        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7, rotation=20)
        ax.set_title(BEHAVIOR_LABEL[beh_key])
        ax.set_ylabel("Per-context / per-seed D")
    fig.tight_layout()
    savefig_paper(fig, "hero1_percell_raw", dir=fig_dir)
    plt.close(fig)


def fig_hero2_marker_map(analysis: dict, fig_dir: Path) -> None:
    """Hero 2 (marker): emission-onset vs selectivity-break map — per (LR,
    step) source free-emission curves, con vs po overlaid, onset/break rungs
    marked per arm (plan §6)."""
    manifest = analysis["manifest"]
    marker = manifest.get("marker")
    if not marker:
        raise RuntimeError("[i1481-figures] hero2: manifest has no marker section")
    arms = marker["arms"]
    ctx_keys = sorted({a["ctx_key"] for a in arms.values()})
    lr_keys = sorted({a["lr_key"] for a in arms.values()})
    fig, axes = plt.subplots(
        len(ctx_keys),
        len(lr_keys),
        figsize=(3.0 * len(lr_keys), 2.4 * len(ctx_keys)),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    pal = paper_palette(2)
    color = {"con": pal[0], "po": pal[1]}
    for i, ctx_key in enumerate(ctx_keys):
        for j, lr_key in enumerate(lr_keys):
            ax = axes[i][j]
            for run_id, arm in sorted(arms.items()):
                if arm["ctx_key"] != ctx_key or arm["lr_key"] != lr_key:
                    continue
                reads = {int(k): v for k, v in (arm["reads_by_step"] or {}).items()}
                if not reads:
                    continue
                steps = sorted(reads)
                gen = [reads[s].get("gen_emission_rate") for s in steps]
                ls = "-" if arm["seed"] == 42 else "--"
                ax.plot(
                    steps,
                    gen,
                    color=color[arm["regime"]],
                    ls=ls,
                    lw=1.2,
                    label=f"{REGIME_LABEL[arm['regime']]} s{arm['seed']}",
                )
                onset = arm["selection"].get("emission_onset_rung")
                brk = arm["selection"].get("selectivity_break_rung")
                if onset is not None:
                    ax.axvline(onset, color=color[arm["regime"]], lw=0.7, alpha=0.5)
                if brk is not None:
                    ax.axvline(brk, color=color[arm["regime"]], lw=0.7, alpha=0.5, ls=":")
            ax.set_title(f"{CTX_LABEL.get(ctx_key, ctx_key)} · {lr_key}", fontsize=8)
            if i == len(ctx_keys) - 1:
                ax.set_xlabel("training step")
            if j == 0:
                ax.set_ylabel("source free-emission rate")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_paper(fig, "hero2_marker_emission_map", dir=fig_dir)
    plt.close(fig)


def fig_tier1_ladders(analysis: dict, fig_dir: Path) -> None:
    """Per-cell Tier-1 ladders: judged rate vs step, all rungs × 3 LR, both
    regimes; the [0.60, 0.85] selection band shaded (plan §6 exploratory)."""
    manifest = analysis["manifest"]
    band = manifest["band"]
    for beh_key, beh in sorted(manifest["content"].items()):
        ctx_keys = sorted(beh)
        fig, axes = plt.subplots(
            1, len(ctx_keys), figsize=(3.2 * len(ctx_keys), 3.0), squeeze=False, sharey=True
        )
        pal = paper_palette(2)
        color = {"con": pal[0], "po": pal[1]}
        for ax, ctx_key in zip(axes[0], ctx_keys, strict=True):
            for arm_id, arm in sorted(beh[ctx_key]["arms"].items()):
                rates = {int(k): float(v) for k, v in (arm["rates_by_step"] or {}).items()}
                if not rates:
                    continue
                steps = sorted(rates)
                ls = {1e-5: "-", 3e-5: "--", 1e-4: ":"}.get(arm["lr"], "-")
                ax.plot(
                    steps,
                    [rates[s] for s in steps],
                    color=color[arm["regime"]],
                    ls=ls,
                    lw=1.0,
                    alpha=0.6 if arm["seed"] == 137 else 1.0,
                )
                sel = arm["selection"]
                ax.scatter([sel["step"]], [sel["rate"]], color=color[arm["regime"]], s=18, zorder=3)
            ax.axhspan(band[0], band[1], color="0.9", zorder=0)
            ax.set_title(CTX_LABEL[ctx_key], fontsize=8)
            ax.set_xlabel("training step")
        axes[0][0].set_ylabel("Tier-1 judged rate")
        fig.suptitle(f"{BEHAVIOR_LABEL[beh_key]} — Tier-1 ladders (band {band[0]}–{band[1]})")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        savefig_paper(fig, f"tier1_ladders_{beh_key}", dir=fig_dir)
        plt.close(fig)


def fig_panel_heatmaps(analysis: dict, fig_dir: Path) -> None:
    """6×N panel heatmaps per behavior (#1434 style): trained judged rate per
    (read context × verdict arm), base-panel companion column included."""
    content = analysis["content"]
    aggregates = analysis["aggregates"]
    for beh_key, agg in sorted(aggregates.items()):
        arm_ids = sorted(agg["arms"])
        ctx_ids = sorted(agg["base_panel"])
        mat = np.full((len(ctx_ids), len(arm_ids) + 1), np.nan)
        for j, arm_id in enumerate(arm_ids):
            for i, ctx_id in enumerate(ctx_ids):
                rec = agg["arms"][arm_id]["contexts"].get(ctx_id) or {}
                if rec.get("rate") is not None:
                    mat[i, j] = rec["rate"]
        for i, ctx_id in enumerate(ctx_ids):
            rec = agg["base_panel"].get(ctx_id) or {}
            if rec.get("rate") is not None:
                mat[i, len(arm_ids)] = rec["rate"]
        fig, ax = plt.subplots(figsize=(0.5 * (len(arm_ids) + 1) + 2.4, 0.4 * len(ctx_ids) + 1.6))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(arm_ids) + 1))
        ax.set_xticklabels([*arm_ids, "base"], rotation=90, fontsize=6)
        ax.set_yticks(range(len(ctx_ids)))
        ax.set_yticklabels(ctx_ids, fontsize=7)
        fig.colorbar(im, ax=ax, label="judged rate")
        ax.set_title(f"{BEHAVIOR_LABEL[beh_key]} — six-context panel rates")
        # No tight_layout here: a colorbar under the paper style's layout
        # engine refuses the engine swap (matplotlib RuntimeError).
        savefig_paper(fig, f"panel_heatmap_{beh_key}", dir=fig_dir)
        plt.close(fig)
    del content


def fig_per_seed_scatter(analysis: dict, fig_dir: Path) -> None:
    """Per-seed D scatter (seed 42 vs 137) with sign-discordance flags."""
    content = analysis["content"]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    pal = paper_palette(3)
    for beh_i, (beh_key, beh) in enumerate(sorted(content["behavior_contexts"].items())):
        xs, ys, disc = [], [], []
        for cell in beh.values():
            ds = {s: r["D"] for s, r in cell["per_seed"].items() if r["status"] == "computed"}
            if len(ds) == 2:
                (x, y) = (ds["42"], ds["137"])
                xs.append(x)
                ys.append(y)
                disc.append(bool(cell.get("sign_discordant")))
        if xs:
            ax.scatter(
                xs,
                ys,
                color=pal[beh_i],
                s=[46 if d else 22 for d in disc],
                marker="o",
                label=BEHAVIOR_LABEL[beh_key],
                alpha=0.85,
                edgecolors=["black" if d else "none" for d in disc],
            )
    lim = max(0.05, *(abs(v) for v in [*ax.get_xlim(), *ax.get_ylim()]))
    ax.plot([-lim, lim], [-lim, lim], color="0.8", lw=0.8)
    ax.axhline(0, color="0.85", lw=0.6)
    ax.axvline(0, color="0.85", lw=0.6)
    ax.set_xlabel("D (seed 42)")
    ax.set_ylabel("D (seed 137)")
    ax.set_title("Run-level variance: per-seed D (edged = sign-discordant)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    savefig_paper(fig, "per_seed_D_scatter", dir=fig_dir)
    plt.close(fig)


def fig_install_bars(analysis: dict, fig_dir: Path) -> None:
    """Con vs po verdict-arm install (Tier-1 selection rate) bars — H1's
    install half (plan §6). Wilson CI unavailable at selection time is drawn
    without error bars (rates_by_step carries no per-rung counts)."""
    manifest = analysis["manifest"]
    for beh_key, beh in sorted(manifest["content"].items()):
        ctx_keys = sorted(beh)
        fig, ax = plt.subplots(figsize=(0.9 * len(ctx_keys) * 2 + 1.6, 3.0))
        pal = paper_palette(2)
        width = 0.18
        for i, ctx_key in enumerate(ctx_keys):
            for k, seed in enumerate(("42", "137")):
                srec = beh[ctx_key]["seeds"][seed]
                for r_i, regime in enumerate(("con", "po")):
                    sel = srec[regime]["selection"]
                    x = i + (r_i - 0.5) * 2 * width + (k - 0.5) * width * 0.9
                    ax.bar(
                        x,
                        float(sel["rate"]),
                        width * 0.8,
                        color=pal[r_i],
                        alpha=1.0 if seed == "42" else 0.55,
                        label=(f"{REGIME_LABEL[regime]} s{seed}" if i == 0 else None),
                    )
        band = manifest["band"]
        ax.axhspan(band[0], band[1], color="0.92", zorder=0)
        ax.set_xticks(range(len(ctx_keys)))
        ax.set_xticklabels([CTX_LABEL[c] for c in ctx_keys], fontsize=7)
        ax.set_ylabel("verdict-arm Tier-1 rate")
        ax.set_title(f"{BEHAVIOR_LABEL[beh_key]} — install at matched recipe")
        ax.legend(fontsize=6, frameon=False, ncol=2)
        fig.tight_layout()
        savefig_paper(fig, f"install_bars_{beh_key}", dir=fig_dir)
        plt.close(fig)


def fig_heldout_decomp(analysis: dict, fig_dir: Path) -> None:
    """Held-out-only vs pooled leakage D per (behavior, context) — paired
    bars with Newcombe CI offsets (plan §5 decomposition)."""
    content = analysis["content"]
    rows = []
    for beh_key, beh in sorted(content["behavior_contexts"].items()):
        for ctx_key, cell in sorted(beh.items()):
            p, h = cell["pooled"], cell["heldout"]["pooled"]
            if p["status"] != "computed" or h["status"] != "computed":
                continue
            rows.append(
                (
                    f"{BEHAVIOR_LABEL[beh_key]}\n{CTX_LABEL[ctx_key]}",
                    p["D"],
                    p["newcombe_95"],
                    h["D"],
                    h["newcombe_95"],
                )
            )
    if not rows:
        raise RuntimeError("[i1481-figures] heldout_decomp: no computed cells")
    fig, ax = plt.subplots(figsize=(0.8 * len(rows) + 2.0, 3.2))
    pal = paper_palette(2)
    x = np.arange(len(rows))
    for off, idx, label, color in (
        (-0.17, 1, "pooled non-source", pal[0]),
        (0.17, 3, "held-out only", pal[1]),
    ):
        vals = np.array([r[idx] for r in rows])
        lo = np.array([r[idx + 1][0] for r in rows])
        hi = np.array([r[idx + 1][1] for r in rows])
        ax.bar(x + off, vals, 0.32, color=color, label=label)
        ax.errorbar(
            x + off,
            vals,
            yerr=_err_offsets(vals, lo, hi),
            fmt="none",
            ecolor="0.3",
            capsize=2,
            lw=1,
        )
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=6)
    ax.set_ylabel("D = po − con")
    ax.set_title("Pooled vs held-out-only leakage decomposition")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    savefig_paper(fig, "heldout_vs_pooled_decomposition", dir=fig_dir)
    plt.close(fig)


def fig_marker_three_space(analysis: dict, fig_dir: Path) -> None:
    """Marker Δlog P vs Δz divergence scatter per cell (saturation signature)
    + the three-space per-cell table dumped as JSON beside the figure."""
    marker = analysis.get("marker_contrast")
    if not marker:
        raise RuntimeError("[i1481-figures] marker_three_space: no marker contrast JSON")
    three = marker["three_space"]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    pal = paper_palette(2)
    for run_id, rec in sorted(three.items()):
        pts = rec["divergence_points"]
        color = pal[0] if "-con-" in run_id else pal[1]
        ax.scatter(
            [p["delta_z"] for p in pts],
            [p["delta_logp"] for p in pts],
            s=8,
            alpha=0.4,
            color=color,
        )
    lim = max(1.0, *(abs(v) for v in [*ax.get_xlim(), *ax.get_ylim()]))
    ax.plot([-lim, lim], [-lim, lim], color="0.8", lw=0.8)
    ax.set_xlabel("Δz_marker (logit space)")
    ax.set_ylabel("Δlog P(marker)")
    ax.set_title("Marker three-space divergence (off-diagonal = saturation)")
    fig.tight_layout()
    savefig_paper(fig, "marker_three_space_divergence", dir=fig_dir)
    plt.close(fig)
    table = {
        run_id: {k: v for k, v in rec.items() if k != "divergence_points"}
        for run_id, rec in sorted(three.items())
    }
    (fig_dir / "marker_three_space_table.json").write_text(json.dumps(table, indent=2))


def fig_margin_rate(analysis: dict, fig_dir: Path) -> None:
    """TF fixed-pool margin vs Tier-1 rate validation scatter per behavior
    (dual-DV rho validation; casual drawn but labeled QUARANTINED)."""
    validation = analysis.get("margin_rate")
    if not validation or "status" in validation:
        raise RuntimeError("[i1481-figures] margin_rate: validation JSON absent/skipped")
    beh_keys = [b for b, r in sorted(validation.items()) if r.get("points")]
    if not beh_keys:
        raise RuntimeError("[i1481-figures] margin_rate: no behaviors with (margin, rate) points")
    fig, axes = plt.subplots(1, len(beh_keys), figsize=(3.2 * len(beh_keys), 3.0), squeeze=False)
    pal = paper_palette(1)
    for ax, beh_key in zip(axes[0], beh_keys, strict=True):
        rec = validation[beh_key]
        pts = rec["points"]
        ax.scatter([p["margin"] for p in pts], [p["rate"] for p in pts], color=pal[0], s=18)
        quarantine = " (QUARANTINED)" if rec.get("quarantined") else ""
        ax.set_title(
            f"{BEHAVIOR_LABEL[beh_key]}{quarantine} — Spearman rho="
            f"{rec.get('spearman_rho', float('nan')):.2f}",
            fontsize=8,
        )
        ax.set_xlabel("TF fixed-pool margin")
        ax.set_ylabel("Tier-1 judged rate")
    fig.tight_layout()
    savefig_paper(fig, "margin_vs_rate_validation", dir=fig_dir)
    plt.close(fig)


FAMILIES = {
    "hero1_forest": fig_hero1_forest,
    "hero1_percell": fig_hero1_percell,
    "hero2_marker_map": fig_hero2_marker_map,
    "tier1_ladders": fig_tier1_ladders,
    "panel_heatmaps": fig_panel_heatmaps,
    "per_seed_scatter": fig_per_seed_scatter,
    "install_bars": fig_install_bars,
    "heldout_decomp": fig_heldout_decomp,
    "marker_three_space": fig_marker_three_space,
    "margin_rate": fig_margin_rate,
}
MARKER_FAMILIES = {"hero2_marker_map", "marker_three_space"}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="#1481 figures driver")
    p.add_argument("--analysis-dir", required=True)
    p.add_argument("--fig-dir", required=True)
    p.add_argument(
        "--aggregates-dir",
        default=None,
        help="defaults to --analysis-dir (panel_aggregate_<beh>.json)",
    )
    p.add_argument("--families", default=None, help="comma subset of " + ",".join(FAMILIES))
    args = p.parse_args(argv)
    analysis_dir = Path(args.analysis_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    agg_dir = Path(args.aggregates_dir) if args.aggregates_dir else analysis_dir
    analysis: dict = {
        "manifest": _load(analysis_dir / "verdict_manifest.json"),
        "content": _load(analysis_dir / "regime_contrast_content.json"),
    }
    marker_path = analysis_dir / "regime_contrast_marker.json"
    if marker_path.exists():
        analysis["marker"] = _load(marker_path)
        analysis["marker_contrast"] = analysis["marker"]
    validation_path = analysis_dir / "margin_rate_validation.json"
    if validation_path.exists():
        analysis["margin_rate"] = _load(validation_path)
    analysis["aggregates"] = {}
    for path in sorted(agg_dir.glob("panel_aggregate_*.json")):
        agg = _load(path)
        analysis["aggregates"][agg["beh_key"]] = agg
    if args.families:
        wanted = [f.strip() for f in args.families.split(",") if f.strip()]
        bad = [f for f in wanted if f not in FAMILIES]
        if bad:
            raise SystemExit(f"[i1481-figures] unknown families {bad}")
    else:
        wanted = [
            f
            for f in FAMILIES
            if not (f in MARKER_FAMILIES and "marker" not in analysis)
            and not (f == "margin_rate" and "margin_rate" not in analysis)
        ]
    set_paper_style()
    for family in wanted:
        FAMILIES[family](analysis, fig_dir)
        logger.info("[i1481-figures] rendered %s", family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
