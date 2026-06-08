"""Issue #502 exploratory plot: length-partial Spearman rho (vs deltaG) for
EVERY predictor cell in the bake-off, as a layer x metric heatmap grid faceted
by extraction point x predictor variant.

One figure per target cell (default loc_ep1, the headline checkpoint). The
next-token-JS baseline (no layer axis) is reported as a reference line in the
suptitle. end_of_system cells are computed on a 20-pair Class-A subpanel (NOT
the 240-pair full panel), so they are not directly comparable to the other two
extraction points -- noted on the panel.

Usage:
    uv run python scripts/issue502_plot_predictor_rho.py --cell loc_ep1
    uv run python scripts/issue502_plot_predictor_rho.py --cell loc_ep1 --target nonstylized
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
REG_DIR = REPO / "eval_results/issue_502/bakeoff/regression"

EXTRACTION_ORDER = ["end_of_system", "last_prompt", "mean_response"]
EXTRACTION_LABEL = {
    "end_of_system": "end of system\n(20-pair Class-A subpanel)",
    "last_prompt": "last prompt token\n(240-pair full panel)",
    "mean_response": "mean over response\n(240-pair full panel)",
}
# y-axis metric order: centroid block, then cloud block.
METRIC_ORDER = ["cosine", "euclidean", "mahal", "mmd", "c2st", "delta_spec", "gauss_kl", "wass2"]
METRIC_LABEL = {
    "cosine": "cosine",
    "euclidean": "Euclidean",
    "mahal": "Mahalanobis",
    "mmd": "MMD",
    "c2st": "c2st",
    "delta_spec": "spectral delta",
    "gauss_kl": "Gaussian-KL",
    "wass2": "Wasserstein-2",
}
VARIANTS = ["raw", "centered"]
LAYERS = list(range(28))


def _rho_key(target: str) -> str:
    return "rho_full_deltag" if target == "full" else "rho_nonstylized_deltag"


def load_grid(cell: str, target: str):
    """Return {(extraction, variant): 2D array [metric x layer] of rho}, plus the JS baseline rho."""
    d = json.loads((REG_DIR / f"{cell}.json").read_text())
    rk = _rho_key(target)
    grids = {
        (ep, v): np.full((len(METRIC_ORDER), len(LAYERS)), np.nan)
        for ep in EXTRACTION_ORDER
        for v in VARIANTS
    }
    js_rho = {}
    for e in d["entries"]:
        rho = e.get(rk)
        if not isinstance(rho, (int, float)) or math.isnan(rho):
            continue
        if e["metric"] == "next_token_js":
            js_rho[(e["extraction_point"], e["variant"])] = rho
            continue
        if e["metric"] not in METRIC_ORDER or e["layer"] not in LAYERS:
            continue
        key = (e["extraction_point"], e["variant"])
        if key not in grids:
            continue
        grids[key][METRIC_ORDER.index(e["metric"]), e["layer"]] = rho
    return grids, js_rho, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cell", default="loc_ep1", help="target cell, e.g. loc_ep1 / loc_ep5 / pos_ep1"
    )
    ap.add_argument("--target", default="full", choices=["full", "nonstylized"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grids, js_rho, d = load_grid(args.cell, args.target)
    out = (
        Path(args.out)
        if args.out
        else (REPO / f"figures/issue_502/predictor_rho_heatmap_{args.cell}_{args.target}.png")
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    vmax = 0.8
    cmap = plt.get_cmap("RdBu_r")  # negative rho -> blue, positive -> red

    fig, axes = plt.subplots(
        len(EXTRACTION_ORDER),
        len(VARIANTS),
        figsize=(13, 11),
        sharex=True,
        sharey=True,
    )
    im = None
    # winner annotation: last_prompt x gauss_kl x L22 x raw
    winner = ("last_prompt", "raw", "gauss_kl", 22)
    for i, ep in enumerate(EXTRACTION_ORDER):
        for j, v in enumerate(VARIANTS):
            ax = axes[i, j]
            g = grids[(ep, v)]
            im = ax.imshow(g, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper")
            if i == 0:
                ax.set_title(f"variant = {v}", fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(EXTRACTION_LABEL[ep], fontsize=10)
            ax.set_yticks(range(len(METRIC_ORDER)))
            ax.set_yticklabels([METRIC_LABEL[m] for m in METRIC_ORDER], fontsize=8)
            ax.set_xticks(range(0, 28, 3))
            ax.set_xticklabels(range(0, 28, 3), fontsize=8)
            if i == len(EXTRACTION_ORDER) - 1:
                ax.set_xlabel("residual-stream layer", fontsize=10)
            # mark the headline winner cell
            if (ep, v) == (winner[0], winner[1]):
                mi, li = METRIC_ORDER.index(winner[2]), winner[3]
                if not math.isnan(g[mi, li]):
                    ax.add_patch(
                        plt.Rectangle(
                            (li - 0.5, mi - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.5
                        )
                    )
            # annotate per-panel best (most negative) cell
            if np.isfinite(g).any():
                flat = np.nanargmin(g)
                bm, bl = np.unravel_index(flat, g.shape)
                ax.plot(bl, bm, marker="o", ms=4, mfc="none", mec="black", mew=1.0)

    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.13, right=0.88, hspace=0.18, wspace=0.08)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("length-partial Spearman rho (predictor vs deltaG)", fontsize=10)

    js_lp = js_rho.get(("last_prompt", "raw")) or js_rho.get(("last_prompt", "centered"))
    js_note = (
        f"next-token-JS baseline (last_prompt): rho = {js_lp:.2f}" if js_lp is not None else ""
    )
    panel = "full 240-pair panel" if args.target == "full" else "non-stylized 156-pair subset"
    fig.suptitle(
        f"#502 predictor rho vs deltaG — every bake-off cell  (target = {args.cell}, {panel})",
        fontsize=14,
        fontweight="bold",
        y=0.95,
    )
    fig.text(
        0.5,
        0.905,
        f"blue = more distance predicts less leakage (negative rho).   "
        f"black box = headline cell (last_prompt × L22 × gauss_kl × raw).   "
        f"open circle = per-panel most-negative cell.   {js_note}",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    # quick textual top-10 for the console
    rk = _rho_key(args.target)
    rows = [
        e
        for e in d["entries"]
        if isinstance(e.get(rk), (int, float))
        and not math.isnan(e[rk])
        and e["metric"] != "next_token_js"
    ]
    rows.sort(key=lambda e: e[rk])
    print(f"\nMost-negative 10 predictor cells (target={args.cell}, {args.target}):")
    for e in rows[:10]:
        print(
            f"  {e['extraction_point']:>13} L{e['layer']:>2} {e['metric']:>11} {e['variant']:>8}  rho={e[rk]:+.3f}"
        )


if __name__ == "__main__":
    main()
