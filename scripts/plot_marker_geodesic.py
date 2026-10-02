#!/usr/bin/env python3
"""Plot the frozen geodesic/cosine comparison with shared paper styling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.c2a_plot_style import (
    MUTED,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval_results/marker_geodesic_20261002/summary.json"
OUT = ROOT / "figures/marker_geodesic_20261002"
SERIES = {
    "centered_cosine": ("Centered cosine", ROLES["control"], "--"),
    "raw_cosine": ("Raw cosine", ROLES["linear"], "-"),
    "geodesic": ("Local-metric geodesic", ROLES["nonlinear"], "-"),
    "graph_euclidean": ("Euclidean graph", ROLES["base_model"], ":"),
}


def main() -> None:
    """Save per-source and per-layer correlations with a paired effect interval."""
    d = json.loads(DATA.read_text())
    set_c2a_style()
    fig, width = c2a_figure("full", aspect=0.57)
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.085, right=0.99, top=0.74, bottom=0.23, wspace=0.33)
    names = ["Villain", "Comedian", "Assistant", "Software\nengineer", "Kindergarten\nteacher"]
    layers = [10, 15, 20, 25]
    primary = d["layers"]["20"]["metrics"]
    for offset, (key, (label, style, line)) in zip(
        [-0.21, -0.07, 0.07, 0.21], SERIES.items(), strict=True
    ):
        vals = [v["spearman_rho"] for v in primary[key]["by_source"]]
        left.plot(
            np.arange(5) + offset,
            vals,
            ls="none",
            marker=style.marker,
            color=style.color,
            ms=9,
            label=label,
        )
        right.plot(
            layers,
            [d["layers"][str(layer)]["metrics"][key]["macro_spearman_rho"] for layer in layers],
            color=style.color,
            marker=style.marker,
            ms=8,
            lw=2.5,
            ls=line,
            label=label,
        )
    left.set_xticks(range(5), names, ha="center")
    left.tick_params(axis="x", labelsize=11)
    left.set_ylim(0.55, 1.0)
    left.set_ylabel(better_label(r"Spearman $\rho$ with marker uptake"))
    right.set_xticks(layers)
    right.set_ylim(0.3, 0.9)
    right.set_xlabel("Transformer block (zero-based)")
    right.set_ylabel(better_label(r"Mean within-source Spearman $\rho$"))
    panel_header(left, "A", "Block 20 · 110 targets per source", "Individual source personas")
    panel_header(right, "B", "Five sources · same fixed outcomes", "Comparison across layers")
    for ax in (left, right):
        style_axis(ax)
    handles, labels = right.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=2)
    delta = (
        primary["geodesic"]["macro_spearman_rho"] - primary["centered_cosine"]["macro_spearman_rho"]
    )
    ci = d["bootstrap"]["geodesic_minus"]["centered_cosine"]["macro_delta_ci95"]
    fig.text(
        0.085,
        0.058,
        f"Block 20: geodesic - centered cosine = {delta:+.3f} "
        f"[95% paired interval {ci[0]:+.3f}, {ci[1]:+.3f}]",
        color=MUTED,
    )
    fig.text(
        0.085,
        0.012,
        "Existing contrastive marker training; fixed persona bank and one training seed.",
        color=MUTED,
    )
    saved = save_c2a_figure(
        fig,
        OUT / "comparison",
        title="Marker leakage: geodesic and cosine correlations",
        subject="Fixed task-66 outcomes; all comparisons retain the same prompt-end vectors.",
        creator="scripts/plot_marker_geodesic.py",
        include_width=width,
    )
    (OUT / "comparison.meta.json").write_text(
        json.dumps(
            {
                "source": str(DATA.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
                "primary_layer": 20,
                "series": list(SERIES),
                "render": saved["record"],
            },
            indent=2,
        )
        + "\n"
    )
    plt.close(fig)
    print(saved["png"])


if __name__ == "__main__":
    main()
