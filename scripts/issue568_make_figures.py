"""Cross-pair figures for issue #568 (third orthogonal pair at the mid dial).

Runs OFF-POD on the VM (CPU, free — task #568 plan §4 Phase E) over the
git-committed ``eval_results/issue_{527,550,538}/{analysis,sweep}/`` anchor
JSONs plus the new pair's ``eval_results/issue_568/{analysis,sweep}/``:

1. ``hero_third_pair_gd1`` — GD1 effective rank (LEFT) and GD1 top-1 SV share
   (RIGHT) vs realized per-cell band landing. The 18 anchor joint cells render
   muted in their issue colors; the new pair's 3 joint cells render bold.
   Envelopes shaded ([1.20, 1.40] eff rank / [0.85, 0.91] top-1 share), the
   three pooled anchor means as horizontal guide lines, and the success band
   (deep-anchor mean, shallow-anchor mean) marked on the left panel.
2. ``exploratory_geometry`` — GD3 worse-of-pair hero variant; the anchor-fit
   regression line (GD1 eff rank ~ realized joint landing, 18 anchor cells)
   with the new cells overlaid (per-cell residuals printed in within-dial-SD
   units, plan §6 continuous read); per-(pair x dial) GD1 clusters with
   per-seed raw dots (the offset read, raw alongside); DV1 median vs landing.
3. ``exploratory_training_marker`` — band-stop step vs landing (all arms, new
   cells overlaid); and, when eval dirs are provided, bystander Δ log P median
   per joint cell, the per-context EOS-margin-delta distribution per dial, and
   the four OLD sources' as-bystander Δ log P in the new joint cells.

The new-cell dirs may be absent pre-run: pass ``--allow-missing-new`` to
render the 18 anchor cells alone (the implementer smoke + a dry preview).

Usage (plan §4 Phase E):
    uv run python scripts/issue568_make_figures.py \\
      --anchor-analysis-dirs eval_results/issue_527/analysis \\
                             eval_results/issue_550/analysis \\
                             eval_results/issue_538/analysis \\
      --anchor-sweep-dirs    eval_results/issue_527/sweep \\
                             eval_results/issue_550/sweep \\
                             eval_results/issue_538/sweep \\
      --new-analysis-dir eval_results/issue_568/analysis \\
      --new-sweep-dir    eval_results/issue_568/sweep \\
      --out figures/issue_568
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

ARMS = ("A_only", "B_only", "joint")

# Colorblind-safe (Okabe-Ito) anchor colors, matching issue550_make_figures.py;
# the NEW pair renders bold vermillion with a distinct marker.
ISSUE_COLORS = {"527": "#E69F00", "550": "#009E73", "538": "#0072B2"}
FALLBACK_COLOR = "#7A7A7A"
NEW_COLOR = "#D55E00"
NEW_MARKER = "D"
ANCHOR_ALPHA = 0.45

PAIR_MARKERS = {"florist__medical_doctor": "o", "librarian__police_officer": "^"}

GD1_EFF_ENVELOPE = (1.20, 1.40)  # plan §6 joint effective-rank envelope
GD1_SV_ENVELOPE = (0.85, 0.91)  # plan §6 joint top-1 SV share envelope

OLD_SOURCES = ("florist", "medical_doctor", "librarian", "police_officer")


def _issue_id(path: str | Path) -> str:
    """Extract the task id from an ``eval_results/issue_<N>/...`` path."""
    m = re.search(r"issue_(\d+)", str(path))
    if m is None:
        raise ValueError(f"cannot infer issue id from path {path!r} (expected issue_<N>)")
    return m.group(1)


def load_dial_cells(analysis_dir: Path, sweep_dir: Path, eval_dir: Path | None = None) -> dict:
    """Load one dial point / pair set: per-(pair, seed) GD metrics + landings.

    Same contract as ``issue550_make_figures.load_dial_cells`` (fails LOUD on
    an unfired band-stop or a missing sweep cell), extended with GD1 effective
    rank and, when ``eval_dir`` is given, per-joint-cell bystander Δ log P
    medians, old-source as-bystander Δ log P, and per-context EOS-margin
    deltas read from the committed ``*__shift.json`` eval artifacts.
    """
    sweep: dict[tuple[str, str, str], dict] = {}
    bands: set[tuple[float, float]] = set()
    for f in sorted(Path(sweep_dir).glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("band_stop_fired") is not True:
            raise AssertionError(
                f"{f}: band_stop_fired={d.get('band_stop_fired')!r} — refusing to "
                "plot an unfired (epochs-cap-saturated) cell on the dial axis"
            )
        sweep[(d["pair_id"], d["arm"], str(d["seed"]))] = d
        bands.add((float(d["band_low_nats"]), float(d["band_high_nats"])))
    if len(bands) != 1:
        raise AssertionError(f"{sweep_dir}: expected ONE band across cells, got {sorted(bands)}")

    cells: list[dict] = []
    for f in sorted(Path(analysis_dir).glob("*.json")):
        d = json.loads(f.read_text())
        if "gating_diagnostics" not in d:
            continue  # e.g. slope_distance_correlation.json side-artifact
        g = d["gating_diagnostics"]
        pair, seed = d["pair_id"], str(d["seed"])
        rows = {arm: sweep[(pair, arm, seed)] for arm in ARMS}  # KeyError = loud
        landings = {arm: float(rows[arm]["final_source_delta_nats"]) for arm in ARMS}
        cell = {
            "pair": pair,
            "seed": seed,
            "gd1_eff": g["gd1_effective_rank"],
            "gd1_sv": g["gd1_top1_sv_share"],
            "gd3_worse": max(g["gd3_a_effective_rank"], g["gd3_b_effective_rank"]),
            "dv1_median": d["dv1"]["median"],
            "x_singleton": float(np.mean([landings["A_only"], landings["B_only"]])),
            "x_joint": landings["joint"],
            "landings": landings,
            "stop_steps": {arm: int(rows[arm]["band_stop_step"]) for arm in ARMS},
        }
        if eval_dir is not None:
            cell.update(_load_joint_shift_stats(Path(eval_dir), pair, seed))
        cells.append(cell)
    if not cells:
        raise AssertionError(f"no analysis JSONs under {analysis_dir}")
    band = next(iter(bands))
    return {"issue": _issue_id(analysis_dir), "band": band, "cells": cells}


def _load_joint_shift_stats(eval_dir: Path, pair: str, seed: str) -> dict:
    """Bystander / old-source / EOS-margin stats from the joint cell's shift JSON."""
    p = eval_dir / f"{pair}__joint__seed{seed}__shift.json"
    d = json.loads(p.read_text())  # FileNotFoundError = loud
    sources = set(pair.split("__"))
    contexts = d["contexts"]
    bystander_dlp = [
        float(contexts[ctx]["delta_logp_marker"]) for ctx in d["eval_panel"] if ctx not in sources
    ]
    old_source_dlp = {
        ctx: float(contexts[ctx]["delta_logp_marker"])
        for ctx in OLD_SOURCES
        if ctx in contexts and ctx not in sources
    }
    # marker_slot_stats is the #538-onward 4-float storage-contract block;
    # the #527 shift JSONs PREDATE it (the EOS-margin panel simply omits
    # that dial — its delta_logp/bystander reads above are unaffected).
    eos_margin_deltas = []
    for ctx in d["eval_panel"]:
        s = contexts[ctx].get("marker_slot_stats")
        if s is None:
            continue
        tr, ba = s["trained"], s["base"]
        eos_margin_deltas.append((tr["z_marker"] - tr["z_eos"]) - (ba["z_marker"] - ba["z_eos"]))
    return {
        "bystander_dlp_median": float(np.median(bystander_dlp)),
        "old_source_dlp": old_source_dlp,
        "eos_margin_deltas": eos_margin_deltas,
    }


def _color(dial: dict) -> str:
    return ISSUE_COLORS.get(dial["issue"], FALLBACK_COLOR)


def _legend_label(dial: dict) -> str:
    low, high = dial["band"]
    return f"#{dial['issue']}: band [{low:g}, {high:g}] nat"


def _scatter_anchors(ax, dials: list[dict], x_key: str, y_key: str) -> None:
    for dial in dials:
        for cell in dial["cells"]:
            ax.scatter(
                cell[x_key],
                cell[y_key],
                color=_color(dial),
                marker=PAIR_MARKERS.get(cell["pair"], "s"),
                s=46,
                alpha=ANCHOR_ALPHA,
                edgecolors="none",
                zorder=3,
            )


def _scatter_new(ax, new_cells: list[dict], x_key: str, y_key: str) -> None:
    for cell in new_cells:
        ax.scatter(
            cell[x_key],
            cell[y_key],
            color=NEW_COLOR,
            marker=NEW_MARKER,
            s=110,
            edgecolors="#1A1A1A",
            linewidths=0.8,
            zorder=5,
        )


def _legend(ax, dials: list[dict], new_cells: list[dict], loc: str = "upper right") -> None:
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=_color(d),
            alpha=ANCHOR_ALPHA,
            label=_legend_label(d),
        )
        for d in dials
    ]
    if new_cells:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker=NEW_MARKER,
                linestyle="none",
                color=NEW_COLOR,
                markeredgecolor="#1A1A1A",
                label="navy_seal x french_person (new, [9,13])",
            )
        )
    ax.legend(handles=handles, loc=loc, fontsize=8.5)


def _pooled_anchor_means(dials: list[dict], y_key: str) -> dict[str, float]:
    """Pooled mean of ``y_key`` over each anchor dial's joint cells."""
    return {d["issue"]: float(np.mean([c[y_key] for c in d["cells"]])) for d in dials}


def figure_hero(
    dials: list[dict], new_cells: list[dict], out_dir: str, out_prefix: str, sources: str
) -> None:
    """Hero: GD1 effective rank + GD1 top-1 SV share vs realized joint landing."""
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.16, wspace=0.18)

    # Dials arrive sorted by band low edge: [shallow #527, mid #550, deep #538].
    eff_means = _pooled_anchor_means(dials, "gd1_eff")
    shallow_mean = eff_means[dials[0]["issue"]]
    deep_mean = eff_means[dials[-1]["issue"]]

    # LEFT — GD1 effective rank (the headline statistic, plan §3 H1).
    ax = axes[0]
    ax.axhspan(*GD1_EFF_ENVELOPE, color="#E8F0E2", zorder=0)
    # Success band (plan §6): between the deep and shallow pooled anchor means.
    ax.axhspan(deep_mean, shallow_mean, color="#F5E3C8", alpha=0.8, zorder=1)
    for d in dials:
        ax.axhline(
            eff_means[d["issue"]],
            color=_color(d),
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )
    _scatter_anchors(ax, dials, "x_joint", "gd1_eff")
    _scatter_new(ax, new_cells, "x_joint", "gd1_eff")
    ax.set_ylim(1.1, 1.5)
    ax.set_xlabel("Realized band landing of the joint cell (nat)")
    ax.set_ylabel("GD1 effective rank (joint SVD)")
    ax.set_title(
        f"Envelope [{GD1_EFF_ENVELOPE[0]}, {GD1_EFF_ENVELOPE[1]}] green; "
        f"success band ({deep_mean:.4f}, {shallow_mean:.4f}) amber",
        fontsize=9.5,
        loc="left",
        pad=6,
    )
    _legend(ax, dials, new_cells, loc="upper left")

    # RIGHT — GD1 top-1 SV share.
    ax = axes[1]
    ax.axhspan(*GD1_SV_ENVELOPE, color="#E8F0E2", zorder=0)
    sv_means = _pooled_anchor_means(dials, "gd1_sv")
    for d in dials:
        ax.axhline(
            sv_means[d["issue"]],
            color=_color(d),
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )
    _scatter_anchors(ax, dials, "x_joint", "gd1_sv")
    _scatter_new(ax, new_cells, "x_joint", "gd1_sv")
    ax.set_ylim(0.78, 0.98)
    ax.set_xlabel("Realized band landing of the joint cell (nat)")
    ax.set_ylabel("GD1 top-1 SV share (joint SVD)")
    ax.set_title(
        f"Envelope [{GD1_SV_ENVELOPE[0]}, {GD1_SV_ENVELOPE[1]}] shaded",
        fontsize=9.5,
        loc="left",
        pad=6,
    )

    n_new = len(new_cells)
    fig.text(
        0.02,
        0.95,
        "Does the mid-dial implant geometry generalize to a third pair?",
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.02,
        0.89,
        f"18 anchor joint cells (muted) + {n_new} new navy_seal x french_person joint cells "
        "(bold); dashes = pooled anchor means; x = realized landings from the sweep JSONs, "
        "never the nominal band",
        ha="left",
        fontsize=10,
        color="#5A5A5A",
    )
    fig.text(
        0.02,
        0.03,
        f"sources: {sources}",
        ha="left",
        color="#7A7A7A",
        fontsize=9,
        fontstyle="italic",
    )

    savefig_paper(fig, f"{out_prefix}/hero_third_pair_gd1", dir=out_dir)
    plt.close(fig)


def _anchor_fit(dials: list[dict]) -> tuple[float, float, float]:
    """Fit GD1 eff rank ~ realized joint landing on the anchor joint cells.

    Returns (slope, intercept, sigma_within) where sigma_within is the pooled
    within-dial SD of GD1 eff rank (the plan §6 residual unit).
    """
    xs = np.array([c["x_joint"] for d in dials for c in d["cells"]])
    ys = np.array([c["gd1_eff"] for d in dials for c in d["cells"]])
    slope, intercept = np.polyfit(xs, ys, 1)
    within = [np.array([c["gd1_eff"] for c in d["cells"]]) for d in dials]
    sigma = float(np.sqrt(np.mean([np.var(w, ddof=1) for w in within])))
    return float(slope), float(intercept), sigma


def figure_exploratory_geometry(
    dials: list[dict], new_cells: list[dict], out_dir: str, out_prefix: str, sources: str
) -> None:
    """GD3 hero variant + anchor regression + per-pair clusters + DV1 median."""
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.07, wspace=0.22, hspace=0.36)

    # (a) GD3 worse-of-pair vs singleton-mean landing.
    ax = axes[0][0]
    ax.axhspan(*GD1_EFF_ENVELOPE, color="#E8F0E2", zorder=0)
    _scatter_anchors(ax, dials, "x_singleton", "gd3_worse")
    _scatter_new(ax, new_cells, "x_singleton", "gd3_worse")
    ax.set_xlabel("Realized landing, mean of the two singletons (nat)")
    ax.set_ylabel("GD3 singleton eff rank (worse of A, B)")
    ax.set_title("GD3 worse-of-pair vs dial (envelope shaded)", fontsize=10.5, loc="left")
    _legend(ax, dials, new_cells, loc="upper left")

    # (b) Anchor regression line + new cells (plan §6 continuous read).
    ax = axes[0][1]
    slope, intercept, sigma = _anchor_fit(dials)
    xs_all = [c["x_joint"] for d in dials for c in d["cells"]] + [c["x_joint"] for c in new_cells]
    grid = np.linspace(min(xs_all) - 0.5, max(xs_all) + 0.5, 50)
    ax.plot(grid, slope * grid + intercept, color="#4A4A4A", linewidth=1.2, zorder=2)
    _scatter_anchors(ax, dials, "x_joint", "gd1_eff")
    _scatter_new(ax, new_cells, "x_joint", "gd1_eff")
    ax.set_xlabel("Realized band landing of the joint cell (nat)")
    ax.set_ylabel("GD1 effective rank (joint SVD)")
    ax.set_title(
        f"Anchor fit (18 cells): slope={slope:+.4f}/nat, within-dial SD={sigma:.4f}",
        fontsize=10.5,
        loc="left",
    )
    for cell in new_cells:
        resid = cell["gd1_eff"] - (slope * cell["x_joint"] + intercept)
        print(
            f"[regression] new cell seed={cell['seed']}: landing={cell['x_joint']:.2f} nat, "
            f"GD1 eff rank={cell['gd1_eff']:.4f}, residual={resid:+.4f} "
            f"({resid / sigma:+.2f} within-dial SD)"
        )

    # (c) Per-(pair x dial) GD1 clusters, per-seed raw dots (the offset read).
    ax = axes[1][0]
    clusters: list[tuple[str, str, list[float], str]] = []
    for d in dials:
        for pair in sorted({c["pair"] for c in d["cells"]}):
            vals = [c["gd1_eff"] for c in d["cells"] if c["pair"] == pair]
            clusters.append((f"#{d['issue']}\n{pair.split('__')[0]}", _color(d), vals, "anchor"))
    if new_cells:
        clusters.append(("#568\nnavy_seal", NEW_COLOR, [c["gd1_eff"] for c in new_cells], "new"))
    for i, (_label, color, vals, kind) in enumerate(clusters):
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=color,
            s=60 if kind == "new" else 36,
            alpha=1.0 if kind == "new" else ANCHOR_ALPHA,
            marker=NEW_MARKER if kind == "new" else "o",
            edgecolors="#1A1A1A" if kind == "new" else "none",
            linewidths=0.8,
            zorder=4,
        )
        ax.hlines(np.mean(vals), i - 0.22, i + 0.22, color=color, linewidth=2.0, zorder=3)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([c[0] for c in clusters], fontsize=7.5)
    ax.set_ylabel("GD1 effective rank (joint SVD)")
    ax.set_title(
        "Per-(pair x dial) clusters, raw per-seed dots + cluster means", fontsize=10.5, loc="left"
    )

    # (d) DV1 additivity median vs landing.
    ax = axes[1][1]
    _scatter_anchors(ax, dials, "x_joint", "dv1_median")
    _scatter_new(ax, new_cells, "x_joint", "dv1_median")
    ax.set_xlabel("Realized band landing of the joint cell (nat)")
    ax.set_ylabel("DV1 additivity cosine (median)")
    ax.set_title(
        "DV1 median vs dial (rank-1-attractor additivity check)", fontsize=10.5, loc="left"
    )

    fig.text(
        0.02,
        0.965,
        "Exploratory geometry panels (descriptive, no gates)",
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.02,
        0.93,
        f"sources: {sources}",
        ha="left",
        color="#7A7A7A",
        fontsize=9,
        fontstyle="italic",
    )

    savefig_paper(fig, f"{out_prefix}/exploratory_geometry", dir=out_dir)
    plt.close(fig)


def figure_exploratory_training_marker(
    dials: list[dict],
    new_cells: list[dict],
    out_dir: str,
    out_prefix: str,
    sources: str,
    have_eval: bool,
) -> None:
    """Band-stop view + (when eval dirs given) bystander / old-source / EOS panels."""
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False

    ncols = 4 if have_eval else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 4.6), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.78, bottom=0.14, wspace=0.26)

    # (a) Band-stop step vs realized landing, ALL arms, new cells overlaid.
    ax = axes[0][0]
    for dial in dials:
        for cell in dial["cells"]:
            for arm in ARMS:
                ax.scatter(
                    cell["stop_steps"][arm],
                    cell["landings"][arm],
                    color=_color(dial),
                    marker=PAIR_MARKERS.get(cell["pair"], "s"),
                    s=34,
                    alpha=ANCHOR_ALPHA,
                    edgecolors="none",
                    zorder=3,
                )
        ax.axhspan(*dial["band"], color=_color(dial), alpha=0.08, zorder=0)
    for cell in new_cells:
        for arm in ARMS:
            ax.scatter(
                cell["stop_steps"][arm],
                cell["landings"][arm],
                color=NEW_COLOR,
                marker=NEW_MARKER,
                s=70,
                edgecolors="#1A1A1A",
                linewidths=0.8,
                zorder=5,
            )
    ax.set_xlabel("Band-stop step")
    ax.set_ylabel("Realized landing (nat)")
    ax.set_title("Stop step vs landing, all arms (bands shaded)", fontsize=10.5, loc="left")
    _legend(ax, dials, new_cells, loc="lower right")

    if have_eval:
        # (b) Bystander Δ log P median per joint cell vs landing.
        ax = axes[0][1]
        _scatter_anchors(ax, dials, "x_joint", "bystander_dlp_median")
        _scatter_new(ax, new_cells, "x_joint", "bystander_dlp_median")
        ax.set_xlabel("Realized band landing of the joint cell (nat)")
        ax.set_ylabel("Bystander Δ log P(marker), median (nat)")
        ax.set_title("Bystander leakage vs dial", fontsize=10.5, loc="left")

        # (c) Per-context EOS-margin delta distribution per dial (+ new).
        ax = axes[0][2]
        groups = [
            (f"#{d['issue']}", _color(d), [v for c in d["cells"] for v in c["eos_margin_deltas"]])
            for d in dials
        ]
        if new_cells:
            groups.append(
                ("#568", NEW_COLOR, [v for c in new_cells for v in c["eos_margin_deltas"]])
            )
        # The #527 shift JSONs predate the marker_slot_stats block — drop
        # empty groups instead of plotting a hollow box.
        groups = [g for g in groups if g[2]]
        bp = ax.boxplot(
            [g[2] for g in groups],
            tick_labels=[g[0] for g in groups],
            patch_artist=True,
            showfliers=False,
        )
        for patch, (_l, color, _v) in zip(bp["boxes"], groups, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Δ(z_marker - z_eos), trained - base")
        ax.set_title("EOS-margin shift per context (joint cells)", fontsize=10.5, loc="left")

        # (d) Old sources as bystanders in the NEW joint cells.
        ax = axes[0][3]
        if new_cells and any(c.get("old_source_dlp") for c in new_cells):
            for j, persona in enumerate(OLD_SOURCES):
                vals = [
                    c["old_source_dlp"][persona]
                    for c in new_cells
                    if persona in c.get("old_source_dlp", {})
                ]
                ax.scatter(
                    np.full(len(vals), j),
                    vals,
                    color=NEW_COLOR,
                    marker=NEW_MARKER,
                    s=60,
                    edgecolors="#1A1A1A",
                    linewidths=0.8,
                    zorder=4,
                )
            ax.set_xticks(range(len(OLD_SOURCES)))
            ax.set_xticklabels(OLD_SOURCES, rotation=30, fontsize=7.5, ha="right")
            ax.set_ylabel("Δ log P(marker) as bystander (nat)")
            ax.set_title("Old sources as bystanders (new joint cells)", fontsize=10.5, loc="left")
        else:
            ax.set_axis_off()
            ax.set_title("Old-sources panel: new-cell eval data absent", fontsize=10.5, loc="left")

    fig.text(
        0.02,
        0.95,
        "Exploratory training + marker panels (descriptive, no gates)",
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.02,
        0.885,
        f"sources: {sources}",
        ha="left",
        color="#7A7A7A",
        fontsize=9,
        fontstyle="italic",
    )

    savefig_paper(fig, f"{out_prefix}/exploratory_training_marker", dir=out_dir)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-pair figures for issue #568.")
    ap.add_argument(
        "--anchor-analysis-dirs",
        nargs="+",
        required=True,
        help="One eval_results/issue_<N>/analysis dir per anchor dial (3 expected).",
    )
    ap.add_argument(
        "--anchor-sweep-dirs",
        nargs="+",
        required=True,
        help="Matching eval_results/issue_<N>/sweep dirs, same order.",
    )
    ap.add_argument(
        "--new-analysis-dir",
        required=True,
        help="eval_results/issue_568/analysis (the new pair's cells).",
    )
    ap.add_argument("--new-sweep-dir", required=True, help="eval_results/issue_568/sweep.")
    ap.add_argument(
        "--anchor-eval-dirs",
        nargs="+",
        default=None,
        help="Optional eval_results/issue_<N>/eval dirs (same order as analysis "
        "dirs); enables the bystander / EOS-margin / old-source panels.",
    )
    ap.add_argument(
        "--new-eval-dir",
        default=None,
        help="Optional eval_results/issue_568/eval dir for the new cells' "
        "bystander / EOS-margin / old-source reads.",
    )
    ap.add_argument(
        "--allow-missing-new",
        action="store_true",
        help="Render the 18 anchor cells alone when the new-cell dirs are missing "
        "or empty (pre-run smoke / preview).",
    )
    ap.add_argument(
        "--out", default="figures/issue_568", help="Output dir (default figures/issue_568)."
    )
    args = ap.parse_args(argv)

    if len(args.anchor_analysis_dirs) != len(args.anchor_sweep_dirs):
        raise SystemExit("--anchor-analysis-dirs and --anchor-sweep-dirs must pair up 1:1")
    if args.anchor_eval_dirs is not None and len(args.anchor_eval_dirs) != len(
        args.anchor_analysis_dirs
    ):
        raise SystemExit("--anchor-eval-dirs must match --anchor-analysis-dirs 1:1")
    for a, s in zip(args.anchor_analysis_dirs, args.anchor_sweep_dirs, strict=True):
        if _issue_id(a) != _issue_id(s):
            raise SystemExit(f"dir pair mismatch: {a} vs {s}")

    have_eval = args.anchor_eval_dirs is not None
    eval_dirs = args.anchor_eval_dirs if have_eval else [None] * len(args.anchor_analysis_dirs)
    dials = [
        load_dial_cells(Path(a), Path(s), Path(e) if e else None)
        for a, s, e in zip(
            args.anchor_analysis_dirs, args.anchor_sweep_dirs, eval_dirs, strict=True
        )
    ]
    dials.sort(key=lambda d: d["band"][0])

    new_analysis = Path(args.new_analysis_dir)
    new_sweep = Path(args.new_sweep_dir)
    new_has_data = new_analysis.is_dir() and any(new_analysis.glob("*.json"))
    if new_has_data:
        new_eval = Path(args.new_eval_dir) if (have_eval and args.new_eval_dir) else None
        new_cells = load_dial_cells(new_analysis, new_sweep, new_eval)["cells"]
    elif args.allow_missing_new:
        print(
            f"[allow-missing-new] no analysis JSONs under {new_analysis}; "
            "rendering the anchors alone."
        )
        new_cells = []
    else:
        raise SystemExit(
            f"no analysis JSONs under {new_analysis}; pass --allow-missing-new for an "
            "anchors-only preview."
        )

    out = Path(args.out)
    out_dir = str(out.parent) + "/"
    out_prefix = out.name
    sources = " + ".join(str(a) for a in args.anchor_analysis_dirs)
    if new_cells:
        sources += f" + {args.new_analysis_dir}"

    figure_hero(dials, new_cells, out_dir, out_prefix, sources)
    figure_exploratory_geometry(dials, new_cells, out_dir, out_prefix, sources)
    figure_exploratory_training_marker(dials, new_cells, out_dir, out_prefix, sources, have_eval)
    n_anchor = sum(len(d["cells"]) for d in dials)
    print(
        f"done: 3 figures under {out}/ ({len(dials)} anchor dials, {n_anchor} anchor cells, "
        f"{len(new_cells)} new cells; eval-dir panels {'ON' if have_eval else 'OFF'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
