#!/usr/bin/env python3
"""Issue #657 — per-persona sycophancy scatter (raw alongside the forest-plot rho).

Plots the raw data underlying the forest-plot point rho = 0.68 (DV-(a) / H1, the
alignment->base-rate generalization for sycophancy): one point per persona,

    x = cosine(persona vector, sycophancy direction) at layer 14   (alignment)
    y = fraction of base-model generations judged sycophantic       (base rate)

Data sources (training-free reuse, no new compute):
  - alignment cosines: this task's
    ``eval_results/issue_657/per_behavior/sycophancy.json`` ``joined_cells``
    (the ``align`` field is constant per bystander persona; layer 14,
    last-prompt-token readout, global-mean-centered bank). These are the exact
    cosines the bake-off's DV-(a) read used.
  - base rates: #623 ``eval_results/issue_623/syc_i.json`` (``syc_i`` per persona,
    base-model sycophancy rate). This is the ``base_rate_source = issue623_syc_i``
    the persisted ``dv_a_base_rate`` block names.

The realized point set is the intersection of the two maps (personas with both an
alignment cosine and a #623 base rate), reproducing the persisted
``dv_a_base_rate`` exactly (n = 16, raw_rho = 0.6834).

Output: figures/issue_657/fig_h1_sycophancy_scatter.png (+ .pdf + .meta.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval_results"
SYCO_JSON = RES / "issue_657" / "per_behavior" / "sycophancy.json"
SYC_I_JSON = RES / "issue_623" / "syc_i.json"


def load_pairs() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Reconstruct the DV-(a) per-persona (alignment, base_rate) pairs."""
    syco = json.loads(SYCO_JSON.read_text())
    # `align` is constant per bystander across all cells it appears in.
    align: dict[str, float] = {}
    for cell in syco["joined_cells"]:
        a = cell["align"]
        if a is None or (isinstance(a, float) and np.isnan(a)):
            continue
        align[cell["bystander"]] = float(a)

    syc_i = json.loads(SYC_I_JSON.read_text())["syc_i"]
    base_rate = {p: float(v["syc_i"]) for p, v in syc_i.items()}

    personas = sorted(p for p in align if p in base_rate)
    if not personas:
        raise RuntimeError("No personas with both an alignment cosine and a base rate.")
    x = np.array([align[p] for p in personas], dtype=float)
    y = np.array([base_rate[p] for p in personas], dtype=float)
    return personas, x, y


# Hand-placed label offsets (data-coords fraction of axis range), keyed by
# persona, used only when adjustText is unavailable. Tuned for the realized
# n=16 layout so the dense left cluster (x ~ -0.10) does not collide. Format:
# (dx_frac, dy_frac, ha) where the offset is a fraction of the axis span and a
# leader line is drawn when the label is pushed far from its point.
_HAND_OFFSETS: dict[str, tuple[float, float, str]] = {
    # left cluster — fan out vertically + horizontally with leaders. The four
    # near x ~ -0.10 (librarian, accountant, software_engineer, journalist) are
    # the tight knot; push them to distinct quadrants with longer leaders.
    "programmer": (-0.03, 0.055, "right"),
    "librarian": (0.0, 0.075, "center"),
    "lawyer": (-0.06, -0.010, "right"),
    "software_engineer": (-0.085, -0.060, "right"),
    "accountant": (0.07, 0.005, "left"),
    "journalist": (0.055, -0.060, "left"),
    # mid / right — mostly clear, nudge to avoid the trend line and neighbours
    "qwen_default": (0.0, -0.040, "center"),
    "philosopher": (-0.010, 0.030, "right"),
    "chef": (0.012, -0.030, "left"),
    "wizard": (0.012, 0.020, "left"),
    "zelthari_scholar": (-0.010, -0.045, "right"),
    "kindergarten_teacher": (0.0, -0.045, "center"),
    "villain": (0.012, -0.030, "left"),
    "comedian": (-0.012, 0.010, "right"),
    "french_person": (0.014, 0.012, "left"),
    "child": (-0.012, 0.030, "right"),
}


def _place_labels_hand(ax, x, y, labels, fontsize: float) -> None:
    """Deterministic hand-placed labels with short leader lines.

    Offsets are a fraction of each axis span so they survive the margin/axes
    re-scaling. A thin grey leader connects the point to its label whenever the
    label is offset enough to read as detached.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xspan = x1 - x0
    yspan = y1 - y0
    for xi, yi, lbl_text in zip(x, y, labels, strict=True):
        # `labels` already has underscores -> spaces; map back to the offset key.
        key = lbl_text.replace(" ", "_")
        dxf, dyf, ha = _HAND_OFFSETS.get(key, (0.0, 0.028, "center"))
        lx = xi + dxf * xspan
        ly = yi + dyf * yspan
        # Leader line from the point toward the label anchor (only if offset).
        if abs(dxf) > 1e-9 or abs(dyf) > 1e-9:
            ax.plot([xi, lx], [yi, ly], color="#AAAAAA", lw=0.45, zorder=2)
        ax.text(
            lx,
            ly,
            lbl_text,
            fontsize=fontsize,
            color="#333333",
            ha=ha,
            va="center",
            zorder=4,
        )


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT / "src"))
    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    primary = paper_palette_role("primary")

    personas, x, y = load_pairs()
    rho = spearmanr(x, y).correlation
    n = len(personas)
    print(f"n = {n}, rho = {rho:.4f}")

    # Wider canvas + generous margins so labels have room and stay on-axes.
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(x, y, s=44, color=primary, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlabel("Alignment to the sycophancy direction (cosine, layer 14)")
    ax.set_ylabel("Base sycophancy rate")
    ax.set_title("Each persona's alignment vs its own base sycophancy rate")
    ax.margins(x=0.18, y=0.16)

    labels = [p.replace("_", " ") for p in personas]
    label_fontsize = 7.0

    try:
        from adjustText import adjust_text  # type: ignore

        texts = [
            ax.text(xi, yi, lbl, fontsize=label_fontsize, color="#333333")
            for xi, yi, lbl in zip(x, y, labels, strict=True)
        ]
        adjust_text(
            texts,
            ax=ax,
            expand_text=(1.15, 1.3),
            expand_points=(1.2, 1.4),
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
        )
        print("labelled via adjustText")
    except ModuleNotFoundError:
        _place_labels_hand(ax, x, y, labels, fontsize=label_fontsize)
        print("labelled via hand-placed offsets (adjustText unavailable)")

    out = savefig_paper(fig, "issue_657/fig_h1_sycophancy_scatter", dir="figures/")
    plt.close(fig)
    for fmt, path in out.items():
        print(f"wrote {fmt}: {path}")


if __name__ == "__main__":
    main()
