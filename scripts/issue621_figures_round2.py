"""Issue #621 round-2 figure regeneration.

Regenerates the two figures the critics flagged in round 1:

1. ``h2_a_init_check`` — annotate the 7 read-arm cells that sit just above
   the pre-registered Δa/a₀ < 0.15 threshold; replace the "within 1°"
   panel-(a) framing with a corrected angular drift number; restate the
   counts as 23/30 (BOTH thresholds) instead of 30/30.
2. ``h5_cross_seed`` — replace code-style y-axis labels
   (``write__police_officer``) with plain-English labels
   ("Write: police officer") and tighten the reference-line legend.

CLI:
    uv run python scripts/issue621_figures_round2.py \\
        [--cache-dir .claude/cache/issue621_analysis/out] \\
        [--out-dir figures/issue_621]
"""

# ruff: noqa: RUF001, RUF002  # math notation in labels

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

log = logging.getLogger("issue_621.figures_round2")

ARM_ORDER = ("read", "write", "bridge")
THRESH_COS = 0.9
THRESH_DELTA = 0.15
RAND_FLOOR_A = 1.0 / math.sqrt(3584)  # ≈ 0.0167
REF_KEY_604 = 0.015
REF_WRITE_604 = 0.927

SOURCE_LABELS = {
    "florist": "florist",
    "medical_doctor": "medical doctor",
    "librarian": "librarian",
    "police_officer": "police officer",
}
ARM_LABELS = {"read": "Read", "write": "Write", "bridge": "Bridge"}


def fig_h2_a_init_check(per_cell: dict, out_dir: Path) -> None:
    """H2 figure: cos(a_t, a_init) + Δa/a₀ across cells; annotate threshold.

    Panel (a) plots per-cell |cos(a_t, a_init)| sorted high→low within arm.
    Panel (b) plots per-cell ‖Δa‖/‖a_init‖ with the pre-registered 0.15
    threshold; the 7 read-arm cells that sit above 0.15 are marked.
    """
    import matplotlib.pyplot as plt

    # Sort per-cell rows by (arm, cos descending, delta ascending) to match
    # the prior figure ordering.
    rows = []
    for slug, c in per_cell.items():
        rows.append(
            {
                "slug": slug,
                "arm": c["arm"],
                "source": c["source"],
                "seed": c["seed"],
                "cos": c["h2_mean_cos"],
                "delta": c["h2_mean_delta"],
            }
        )

    # Sort within arm by cos descending so the per-arm groups are visually
    # readable left→right.
    def _sort_key(r):
        arm_idx = ARM_ORDER.index(r["arm"])
        return (arm_idx, -r["cos"])

    rows.sort(key=_sort_key)
    n = len(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    # Color by arm.
    arm_colors = {
        "read": paper_palette_role("primary"),
        "write": paper_palette_role("accent"),
        "bridge": paper_palette_role("baseline"),
    }
    bar_colors = [arm_colors[r["arm"]] for r in rows]

    # --- Panel (a): |cos(a_t, a_init)|
    ax = axes[0]
    xs = np.arange(n)
    cos_vals = [r["cos"] for r in rows]
    ax.bar(xs, cos_vals, color=bar_colors, edgecolor="none", width=0.8)
    ax.axhline(
        THRESH_COS,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=f"H2 threshold (|cos| > {THRESH_COS:.1f}): 30/30 pass",
    )
    ax.set_ylabel("|cos(a_trained, a_init)|")
    ax.set_ylim(0.95, 1.005)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            f"{ARM_LABELS[r['arm']][:1]}:{SOURCE_LABELS[r['source']].split()[0]}/{r['seed']}"
            for r in rows
        ],
        rotation=90,
        fontsize=6,
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title(
        "(a) |cos(a_trained, a_init)| ≥ 0.988 across 30 cells\n"
        "median 0.9947 (≈ 5.9°), read-arm min 0.9878 (≈ 8.9°)",
        loc="left",
        fontsize=9,
    )

    # --- Panel (b): Δa/a₀ with crossers annotated
    ax = axes[1]
    delt_vals = [r["delta"] for r in rows]
    # Pull cells that fail the Δa<0.15 threshold (all 7 are read-arm)
    cross_xs = [i for i, r in enumerate(rows) if r["delta"] >= THRESH_DELTA]
    cross_vals = [delt_vals[i] for i in cross_xs]

    ax.bar(xs, delt_vals, color=bar_colors, edgecolor="none", width=0.8)
    ax.axhline(
        THRESH_DELTA,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=f"H2 threshold (Δa/a₀ < {THRESH_DELTA:.2f}): 23/30 below",
    )

    # Mark the 7 crossers
    for cx, cv in zip(cross_xs, cross_vals):
        ax.scatter(
            [cx],
            [cv],
            marker="o",
            s=22,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
    if cross_xs:
        ax.scatter(
            [],
            [],
            marker="o",
            s=22,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label=f"above threshold (n={len(cross_xs)}): read-arm cells at 0.151–0.157",
        )

    ax.set_ylabel("‖Δa‖ / ‖a_init‖")
    ax.set_ylim(0, 0.18)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            f"{ARM_LABELS[r['arm']][:1]}:{SOURCE_LABELS[r['source']].split()[0]}/{r['seed']}"
            for r in rows
        ],
        rotation=90,
        fontsize=6,
    )
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(
        "(b) Net A rotation: 23/30 cells below 0.15; 7 read-arm cells at 0.151–0.157\n"
        "read median 0.151, write median 0.094, bridge median 0.078",
        loc="left",
        fontsize=9,
    )

    # Legend strip across the bottom: arm color key.
    handles = [plt.Rectangle((0, 0), 1, 1, color=arm_colors[a]) for a in ARM_ORDER]
    labels = [f"{ARM_LABELS[a]} (n={sum(1 for r in rows if r['arm'] == a)})" for a in ARM_ORDER]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    savefig_paper(fig, out_dir / "h2_a_init_check")
    plt.close(fig)


def fig_h5_cross_seed(cross_seed: dict, out_dir: Path) -> None:
    """H5 figure: cross-seed |cos| of A and B per (arm × source) group.

    Plain-English labels on the y-axis ("Write: police officer" instead of
    "write__police_officer").
    """
    import matplotlib.pyplot as plt

    # Build rows: (arm, source, a_median, b_median)
    rows = []
    for key, v in cross_seed.items():
        # key like "write__police_officer"
        arm, src = key.split("__")
        rows.append(
            {
                "arm": arm,
                "source": src,
                "label": f"{ARM_LABELS[arm]}: {SOURCE_LABELS[src]}",
                "a_med": v["cross_seed_cos_a_median"],
                "a_lo": v["cross_seed_cos_a_p25"],
                "a_hi": v["cross_seed_cos_a_p75"],
                "b_med": v["cross_seed_cos_b_median"],
                "b_lo": v["cross_seed_cos_b_p25"],
                "b_hi": v["cross_seed_cos_b_p75"],
            }
        )
    # Sort: arm group then source
    rows.sort(key=lambda r: (ARM_ORDER.index(r["arm"]), r["source"]))
    n = len(rows)
    ys = np.arange(n)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    # B markers + IQR bars
    b_med = np.asarray([r["b_med"] for r in rows])
    b_err_lo = np.asarray([r["b_med"] - r["b_lo"] for r in rows])
    b_err_hi = np.asarray([r["b_hi"] - r["b_med"] for r in rows])
    ax.errorbar(
        b_med,
        ys,
        xerr=[b_err_lo, b_err_hi],
        fmt="s",
        color=paper_palette_role("accent"),
        markersize=8,
        linewidth=1.2,
        capsize=3,
        label="B (write direction) — median + IQR",
    )

    # A markers + IQR bars
    a_med = np.asarray([r["a_med"] for r in rows])
    a_err_lo = np.asarray([r["a_med"] - r["a_lo"] for r in rows])
    a_err_hi = np.asarray([r["a_hi"] - r["a_med"] for r in rows])
    ax.errorbar(
        a_med,
        ys,
        xerr=[a_err_lo, a_err_hi],
        fmt="o",
        color=paper_palette_role("primary"),
        markersize=8,
        linewidth=1.2,
        capsize=3,
        label="A (read direction) — median + IQR",
    )

    # Reference lines
    ax.axvline(
        REF_KEY_604,
        color="grey",
        linestyle=":",
        linewidth=1.0,
        label=f"#604 rank-16 top-1 key = {REF_KEY_604:.3f}",
    )
    ax.axvline(
        REF_WRITE_604,
        color="grey",
        linestyle="--",
        linewidth=1.0,
        label=f"#604 rank-16 write = {REF_WRITE_604:.3f}",
    )
    ax.axvline(
        RAND_FLOOR_A,
        color=paper_palette_role("control"),
        linestyle="-",
        linewidth=0.8,
        alpha=0.6,
        label=f"random floor 1/√3584 = {RAND_FLOOR_A:.4f}",
    )

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10)
    ax.set_xlabel("cross-seed |cos| (pairwise across 3 seeds, all modules)")
    ax.set_xlim(-0.02, 1.0)
    ax.invert_yaxis()  # bridge groups at top
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title(
        "A stays seed-arbitrary at rank 1; B is seed-stable below #604's rank-16 dial value",
        loc="left",
        fontsize=10,
    )
    fig.tight_layout()
    savefig_paper(fig, out_dir / "h5_cross_seed")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", default=".claude/cache/issue621_analysis/out")
    ap.add_argument("--out-dir", default="figures/issue_621")
    args = ap.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell = json.loads((cache_dir / "per_cell_main.json").read_text())
    cross_seed = json.loads((cache_dir / "h5_cross_seed.json").read_text())

    set_paper_style("blog")
    fig_h2_a_init_check(per_cell, out_dir)
    fig_h5_cross_seed(cross_seed, out_dir)
    log.info("figures rewritten to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
