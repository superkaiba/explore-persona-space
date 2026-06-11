"""Cross-issue 3-dial figures for issue #550 (band [9,13] mid dial point).

Runs OFF-POD on the VM (CPU, free — task #550 plan §4 Phase E) over the
git-committed ``eval_results/issue_{527,538,550}/{analysis,sweep}/`` JSONs:

1. ``hero_dial_gd3_gd1`` — x = realized per-cell band landing (nat, from the
   sweep JSONs), y = GD3 worse-of-pair singleton effective rank (LEFT panel,
   the gated statistic per plan §13.3: max of the A/B singleton eff ranks —
   reproduces the §13.1 anchor clusters 1.359-1.384 / 1.326-1.341) and GD1
   top-1 SV share (RIGHT panel, joint-based). 6 cells (2 pairs x 3 seeds)
   per dial point; shared envelopes shaded ([1.20, 1.40] GD3,
   [0.85, 0.91] GD1); GD3 pass gate 2.0 dashed.
2. ``exploratory_dial_panels`` — over-produced descriptive panels (plan §6):
   GD2 singleton cosine vs dial, DV1 median vs dial, DV3 magnitude residual
   vs dial, and band-stop step vs realized landing (plan §13.7).

The script accepts ANY number of (analysis, sweep) dir pairs ≥ 2, so it
renders a 2-dial figure from the anchors alone before #550's results exist
(the implementer smoke) and the full 3-dial figure afterwards.

Usage (plan §4 Phase E):
    uv run python scripts/issue550_make_figures.py \
      --analysis-dirs eval_results/issue_527/analysis \
                      eval_results/issue_538/analysis \
                      eval_results/issue_550/analysis \
      --sweep-dirs    eval_results/issue_527/sweep \
                      eval_results/issue_538/sweep \
                      eval_results/issue_550/sweep \
      --out figures/issue_550
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

# Colorblind-safe (Okabe-Ito): #527 orange / #550 green / #538 blue (plan §6).
ISSUE_COLORS = {"527": "#E69F00", "550": "#009E73", "538": "#0072B2"}
FALLBACK_COLOR = "#7A7A7A"

PAIR_MARKERS = {"florist__medical_doctor": "o", "librarian__police_officer": "^"}
PAIR_LABELS = {
    "florist__medical_doctor": "Florist x Medical doctor",
    "librarian__police_officer": "Librarian x Police officer",
}

GD3_ENVELOPE = (1.20, 1.40)  # plan §6 success envelope (shared by both anchors)
GD1_ENVELOPE = (0.85, 0.91)
GD3_PASS_GATE = 2.0  # the original superposition pass gate (dashed reference)


def _issue_id(path: str | Path) -> str:
    """Extract the task id from an ``eval_results/issue_<N>/...`` path."""
    m = re.search(r"issue_(\d+)", str(path))
    if m is None:
        raise ValueError(f"cannot infer issue id from path {path!r} (expected issue_<N>)")
    return m.group(1)


def load_dial_cells(analysis_dir: Path, sweep_dir: Path) -> dict:
    """Load one dial point: per-(pair, seed) GD metrics + realized landings.

    Returns ``{"issue": str, "band": (low, high), "cells": [dict, ...]}`` with
    one cell per (pair, seed). Each cell carries the gated worse-of-pair GD3
    (max of A/B singleton eff ranks, plan §13.3), GD1/GD2/DV1/DV3, the
    singleton-mean landing (x for singleton-based stats), the joint landing
    (x for joint-based stats), and per-arm band-stop steps. Fails LOUD on a
    missing sweep cell or an unfired band-stop — analysis must never include
    a cell that trained to the epochs cap (plan §8).
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
        g = d["gating_diagnostics"]
        pair, seed = d["pair_id"], str(d["seed"])
        rows = {arm: sweep[(pair, arm, seed)] for arm in ARMS}  # KeyError = loud
        landings = {arm: float(rows[arm]["final_source_delta_nats"]) for arm in ARMS}
        cells.append(
            {
                "pair": pair,
                "seed": seed,
                "gd3_worse": max(g["gd3_a_effective_rank"], g["gd3_b_effective_rank"]),
                "gd1_sv": g["gd1_top1_sv_share"],
                "gd2_cos": g["gd2_singleton_cosine_median"],
                "dv1_median": d["dv1"]["median"],
                "dv3_residual_median": d["dv3"]["residual_median"],
                "x_singleton": float(np.mean([landings["A_only"], landings["B_only"]])),
                "x_joint": landings["joint"],
                "landings": landings,
                "stop_steps": {arm: int(rows[arm]["band_stop_step"]) for arm in ARMS},
            }
        )
    if not cells:
        raise AssertionError(f"no analysis JSONs under {analysis_dir}")
    band = next(iter(bands))
    return {"issue": _issue_id(analysis_dir), "band": band, "cells": cells}


def _legend_label(dial: dict) -> str:
    low, high = dial["band"]
    return f"#{dial['issue']}: band [{low:g}, {high:g}] nat"


def _color(dial: dict) -> str:
    return ISSUE_COLORS.get(dial["issue"], FALLBACK_COLOR)


def _scatter_dial(ax, dials: list[dict], x_key: str, y_key: str) -> None:
    """Per-cell scatter of ``y_key`` vs realized landing ``x_key`` per dial."""
    for dial in dials:
        for cell in dial["cells"]:
            ax.scatter(
                cell[x_key],
                cell[y_key],
                color=_color(dial),
                marker=PAIR_MARKERS.get(cell["pair"], "s"),
                s=80,
                edgecolors="none",
                zorder=3,
            )


def _issue_legend(ax, dials: list[dict], loc: str = "upper right") -> None:
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=_color(d), label=_legend_label(d))
        for d in dials
    ] + [
        plt.Line2D([], [], marker=m, linestyle="none", color="#4A4A4A", label=PAIR_LABELS[p])
        for p, m in PAIR_MARKERS.items()
    ]
    ax.legend(handles=handles, loc=loc, fontsize=8.5)


def figure_hero(dials: list[dict], out_dir: str, out_prefix: str, sources: str) -> None:
    """Hero: GD3 worse-of-pair + GD1 top-1 SV share vs realized dial landing."""
    set_paper_style("blog")
    # fig-level title text + per-panel titles need manual top space (see
    # issue538_make_figures.py — constrained_layout fights fig.text).
    mpl.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.16, wspace=0.18)

    # LEFT — GD3 worse-of-pair singleton effective rank (the gated statistic).
    ax = axes[0]
    ax.axhspan(*GD3_ENVELOPE, color="#E8F0E2", zorder=0)
    ax.axhline(GD3_PASS_GATE, color="#888888", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        0.98,
        0.93,
        "pass gate 2.0",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#888888",
    )
    _scatter_dial(ax, dials, "x_singleton", "gd3_worse")
    ax.set_ylim(1.0, 2.2)
    ax.set_xlabel("Realized band landing, mean of the two singletons (nat)")
    ax.set_ylabel("Singleton effective rank (worse of A, B)")
    ax.set_title(
        f"Shared envelope [{GD3_ENVELOPE[0]}, {GD3_ENVELOPE[1]}] shaded",
        fontsize=10.5,
        loc="left",
        pad=6,
    )
    _issue_legend(ax, dials, loc="upper left")

    # RIGHT — GD1 top-1 SV share (joint-based; x = the joint cell's landing).
    ax = axes[1]
    ax.axhspan(*GD1_ENVELOPE, color="#E8F0E2", zorder=0)
    _scatter_dial(ax, dials, "x_joint", "gd1_sv")
    ax.set_ylim(0.75, 1.0)
    ax.set_xlabel("Realized band landing of the joint cell (nat)")
    ax.set_ylabel("Joint top-1 SV share")
    ax.set_title(
        f"Shared envelope [{GD1_ENVELOPE[0]}, {GD1_ENVELOPE[1]}] shaded",
        fontsize=10.5,
        loc="left",
        pad=6,
    )

    n_dials = len(dials)
    fig.text(
        0.02,
        0.95,
        f"Implant geometry across {n_dials} sampled dial points",
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.02,
        0.89,
        "Each point is one (pair, seed) cell at its realized band landing; "
        "x positions are read from the sweep JSONs, never the nominal band",
        ha="left",
        fontsize=10,
        color="#5A5A5A",
    )
    fig.text(
        0.02,
        0.03,
        f"n=3 seeds x 2 pairs per dial point · sources: {sources}",
        ha="left",
        color="#7A7A7A",
        fontsize=9,
        fontstyle="italic",
    )

    savefig_paper(fig, f"{out_prefix}/hero_dial_gd3_gd1", dir=out_dir)
    plt.close(fig)


def figure_exploratory(dials: list[dict], out_dir: str, out_prefix: str, sources: str) -> None:
    """Exploratory dump (plan §6): GD2 / DV1 / DV3 vs dial + stop-step view."""
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.07, wspace=0.22, hspace=0.34)

    ax = axes[0][0]
    _scatter_dial(ax, dials, "x_singleton", "gd2_cos")
    ax.set_xlabel("Realized landing, singleton mean (nat)")
    ax.set_ylabel("GD2 singleton cosine (median)")
    ax.set_title("Singleton cosine vs dial", fontsize=10.5, loc="left")
    _issue_legend(ax, dials, loc="lower right")

    ax = axes[0][1]
    _scatter_dial(ax, dials, "x_joint", "dv1_median")
    ax.set_xlabel("Realized landing, joint cell (nat)")
    ax.set_ylabel("DV1 additivity cosine (median)")
    ax.set_title("DV1 median vs dial (spread = across-cell scatter)", fontsize=10.5, loc="left")

    ax = axes[1][0]
    _scatter_dial(ax, dials, "x_joint", "dv3_residual_median")
    ax.set_xlabel("Realized landing, joint cell (nat)")
    ax.set_ylabel("DV3 magnitude residual (median)")
    ax.set_title("Magnitude residual vs dial", fontsize=10.5, loc="left")

    # Stop-step vs realized landing, ALL arms (raw view, plan §13.7: report
    # stop-step alongside realized nat so flatness holds on both axes).
    ax = axes[1][1]
    for dial in dials:
        for cell in dial["cells"]:
            for arm in ARMS:
                ax.scatter(
                    cell["stop_steps"][arm],
                    cell["landings"][arm],
                    color=_color(dial),
                    marker=PAIR_MARKERS.get(cell["pair"], "s"),
                    s=42,
                    edgecolors="none",
                    alpha=0.85,
                    zorder=3,
                )
        ax.axhspan(*dial["band"], color=_color(dial), alpha=0.08, zorder=0)
    ax.set_xlabel("Band-stop step")
    ax.set_ylabel("Realized landing (nat)")
    ax.set_title(
        "Stop step vs landing, all 3 arms per cell (bands shaded)", fontsize=10.5, loc="left"
    )

    fig.text(
        0.02,
        0.96,
        "Exploratory dial panels (descriptive, no gates)",
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.02,
        0.925,
        f"sources: {sources}",
        ha="left",
        color="#7A7A7A",
        fontsize=9,
        fontstyle="italic",
    )

    savefig_paper(fig, f"{out_prefix}/exploratory_dial_panels", dir=out_dir)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-issue 3-dial figures for issue #550.")
    ap.add_argument(
        "--analysis-dirs",
        nargs="+",
        required=True,
        help="One eval_results/issue_<N>/analysis dir per dial point (>= 2).",
    )
    ap.add_argument(
        "--sweep-dirs",
        nargs="+",
        required=True,
        help="Matching eval_results/issue_<N>/sweep dirs, same order.",
    )
    ap.add_argument(
        "--out", default="figures/issue_550", help="Output dir (default figures/issue_550)."
    )
    args = ap.parse_args(argv)

    if len(args.analysis_dirs) != len(args.sweep_dirs):
        raise SystemExit("--analysis-dirs and --sweep-dirs must pair up 1:1")
    if len(args.analysis_dirs) < 2:
        raise SystemExit("need >= 2 dial points to draw a dial axis")
    for a, s in zip(args.analysis_dirs, args.sweep_dirs, strict=True):
        if _issue_id(a) != _issue_id(s):
            raise SystemExit(f"dir pair mismatch: {a} vs {s}")

    dials = [
        load_dial_cells(Path(a), Path(s))
        for a, s in zip(args.analysis_dirs, args.sweep_dirs, strict=True)
    ]
    # Order dial points left-to-right by their band low edge.
    dials.sort(key=lambda d: d["band"][0])

    out = Path(args.out)
    out_dir = str(out.parent) + "/"
    out_prefix = out.name
    sources = " + ".join(str(a) for a in args.analysis_dirs)

    figure_hero(dials, out_dir, out_prefix, sources)
    figure_exploratory(dials, out_dir, out_prefix, sources)
    n_cells = sum(len(d["cells"]) for d in dials)
    print(
        f"done: 2 figures under {out}/ ({len(dials)} dial points, {n_cells} pair-seed cells; "
        f"GD3 = worse-of-pair = max(A, B) per plan §13.3)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
