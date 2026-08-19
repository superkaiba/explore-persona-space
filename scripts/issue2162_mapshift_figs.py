#!/usr/bin/env python3
"""Issue #2162 mapshift harvest figures (Results 2.5 + 4-extension).

Three figures from the committed `eval_results/issue_2162/mapshift/` outputs
(no recomputation — pure renders):

1. ``fig_shift_cosine_by_layer`` — mean cos(map-predicted shift, realized
   patched shift) vs layer per map source, steered arm; survivors + all-cells
   panels; null patch arms (shuffled / cross-type donors) as a shaded envelope.
2. ``fig_shift_cosine_heatmap`` — per-type x layer mean cosine for the fresh
   bank-fit map, steered arm, all 39 type-cells (rows sorted by row max),
   diverging colormap centered at 0.
3. ``fig_2afc_by_layer`` — pooled paired-2AFC accuracy (cosine metric, span
   pooling) vs layer per arm, clustered 95% CIs, the shuffled-pair null band,
   and the banked #2215 span reference points.

Usage:
  uv run python scripts/issue2162_mapshift_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) BEFORE any heavy import

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "eval_results/issue_2162/mapshift"
FIG_DIR = REPO_ROOT / "figures/issue_2162/mapshift"
N_LAYERS = 28

# Plain-English source/arm labels (no arm codes on any canvas).
SOURCE_LABELS = {
    "fresh": "map fit on this bank",
    "m779ce": "banked single-turn map (#779)",
    "m1738ce": "banked multi-turn map (#1738)",
    "ctxshift": "raw context shift (no map)",
}
ARM_LABELS = {
    "freshce": "map fit on this bank",
    "m779ce": "banked single-turn map (#779)",
    "idbias_ce": "identity + bias",
    "identity_ce": "identity",
}
NULL_ARMS = ("shuffled", "crosstype")


def _series(view: dict, source: str, arm: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for layer in range(N_LAYERS):
        rec = view.get(f"{source}|L{layer}|{arm}")
        if rec is not None:
            xs.append(layer)
            ys.append(rec["mean_cos_over_cells"])
    return xs, ys


def fig_shift_cosine_by_layer() -> None:
    ss = json.loads((DATA_ROOT / "shift_summary.json").read_text())
    n_surv = len(ss["survivor_cells"])
    colors = dict(zip(SOURCE_LABELS, paper_palette(len(SOURCE_LABELS))))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, (view_key, title) in zip(
        axes,
        [
            ("survivors", f"stored-and-used cells (n={n_surv})"),
            ("all_cells", "all cells (n=39)"),
        ],
    ):
        view = ss["views"][view_key]
        # null envelope: min..max over (source x {shuffled, crosstype}) per layer
        band_lo, band_hi, band_x = [], [], []
        for layer in range(N_LAYERS):
            vals = [
                view[f"{src}|L{layer}|{arm}"]["mean_cos_over_cells"]
                for src in SOURCE_LABELS
                for arm in NULL_ARMS
                if f"{src}|L{layer}|{arm}" in view
            ]
            if vals:
                band_x.append(layer)
                band_lo.append(min(vals))
                band_hi.append(max(vals))
        ax.fill_between(
            band_x,
            band_lo,
            band_hi,
            color="0.6",
            alpha=0.3,
            lw=0,
            label="null patch arms (shuffled / cross-type donors)",
        )
        for src, label in SOURCE_LABELS.items():
            xs, ys = _series(view, src, "steered")
            marker = "o" if len(xs) < 6 else None
            ax.plot(xs, ys, color=colors[src], marker=marker, ms=5, label=label)
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("layer")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("cos(map-predicted shift, realized patched shift)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    savefig_paper(fig, "fig_shift_cosine_by_layer", dir=FIG_DIR)
    plt.close(fig)


def fig_shift_cosine_heatmap() -> None:
    rows = [
        json.loads(x)
        for x in (DATA_ROOT / "shift_cells.jsonl").read_text().split("\n")
        if x.strip()
    ]
    sel = [r for r in rows if r["source"] == "fresh" and r["arm"] == "steered"]
    cells = sorted({r["cell"] for r in sel})
    mat = np.full((len(cells), N_LAYERS), np.nan)
    for r in sel:
        mat[cells.index(r["cell"]), r["layer"]] = r["mean_cos"]
    order = np.argsort(-np.nanmax(mat, axis=1))
    mat = mat[order]
    cells = [cells[i] for i in order]
    vmax = float(np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(cells, fontsize=6.5)
    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.set_xlabel("layer")
    ax.set_title("cos(predicted shift, realized patched shift) — map fit on this bank, steered")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean cosine")
    fig.tight_layout()
    savefig_paper(fig, "fig_shift_cosine_heatmap", dir=FIG_DIR)
    plt.close(fig)


def fig_2afc_by_layer() -> None:
    dv3 = json.loads((DATA_ROOT / "dv3_ext.json").read_text())
    per = dv3["per_config"]
    colors = dict(zip(ARM_LABELS, paper_palette(len(ARM_LABELS))))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhspan(0.48, 0.52, color="0.6", alpha=0.3, lw=0, label="shuffled-pair null band")
    ax.axhline(0.5, color="0.4", lw=0.8)
    for arm, label in ARM_LABELS.items():
        xs, ys, lo, hi = [], [], [], []
        for layer in range(N_LAYERS):
            rec = per.get(f"{arm}|L{layer}|span")
            if rec is None:
                continue
            p = rec["pooled"]["cosine"]
            xs.append(layer)
            ys.append(p["acc"])
            ci = p.get("acc_ci95_clustered") or [np.nan, np.nan]
            lo.append(max(0.0, p["acc"] - ci[0]))
            hi.append(max(0.0, ci[1] - p["acc"]))
        marker = "o" if len(xs) < 6 else None
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            color=colors[arm],
            marker=marker,
            ms=5,
            lw=1.6,
            elinewidth=0.8,
            capsize=0,
            label=label,
        )
    banked = dv3["banked_dv3_span_reference"]
    bx, by = [], []
    for layer in (14, 19):
        rec = banked.get(f"779ce|L{layer}|span")
        if rec:
            bx.append(layer)
            by.append(rec["cosine"]["acc"])
    ax.scatter(bx, by, marker="x", s=60, color="black", zorder=5)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([], [], marker="x", ls="none", color="black", label="banked #2215 run (#779 map)")
    )
    labels.append("banked #2215 run (#779 map)")
    ax.set_xlabel("layer")
    ax.set_ylabel("paired 2AFC accuracy (cosine, span pooling)")
    ax.set_title("does the predicted answer identify its own context's real answer?")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(handles, labels, fontsize=8, loc="lower right")
    fig.tight_layout()
    savefig_paper(fig, "fig_2afc_by_layer", dir=FIG_DIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("generic")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_shift_cosine_by_layer()
    fig_shift_cosine_heatmap()
    fig_2afc_by_layer()
    print(f"[figs] 3 figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
