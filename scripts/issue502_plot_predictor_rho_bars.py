"""Issue #502 exploratory bar charts of predictor rho (length-partial Spearman
vs deltaG marker-leakage transfer), FULL 240-pair panel only.

Two figures per target cell (default loc_ep1):
  1. predictor_rho_bars_topcells_<cell>.png
       The strongest N activation predictor cells, ranked by full-panel rho.
  2. predictor_rho_bars_ridge_<cell>.png
       rho by residual-stream layer for the headline metric (last prompt token x
       Gaussian-KL). Shows the L19-24 ridge honestly (no per-metric cherry-pick)
       and where it sits on the stack.

Usage:
    uv run python scripts/issue502_plot_predictor_rho_bars.py --cell loc_ep1 --topn 20
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

METRIC_LABEL = {
    "cosine": "cosine",
    "euclidean": "Euclidean",
    "mahal": "Mahalanobis",
    "mmd": "MMD",
    "c2st": "c2st",
    "delta_spec": "spectral delta",
    "gauss_kl": "Gaussian-KL",
    "wass2": "Wasserstein-2",
    "next_token_js": "next-token JS",
}
EXTRACTION_LABEL = {
    "end_of_system": "end of system",
    "last_prompt": "last prompt token",
    "mean_response": "mean over response",
}

NAVY = "#1f4e79"
CRIMSON = "#b03060"


def _finite(v):
    return isinstance(v, (int, float)) and not math.isnan(v)


def _new_fig(figsize):
    """Figure with NO auto-layout engine so subplots_adjust is authoritative."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_layout_engine("none")
    return fig, ax


def load_entries(cell: str):
    d = json.loads((REG_DIR / f"{cell}.json").read_text())
    return [e for e in d["entries"] if _finite(e.get("rho_full_deltag"))], d


def fmt_cell(e) -> str:
    if e["metric"] == "next_token_js":
        return (
            f"{EXTRACTION_LABEL.get(e['extraction_point'], e['extraction_point'])} · next-token JS"
        )
    return (
        f"{EXTRACTION_LABEL.get(e['extraction_point'], e['extraction_point'])}"
        f" · L{e['layer']} · {METRIC_LABEL.get(e['metric'], e['metric'])} · {e['variant']}"
    )


def _js_baseline(entries):
    js = [
        e
        for e in entries
        if e["metric"] == "next_token_js" and e["extraction_point"] == "last_prompt"
    ]
    return js[0]["rho_full_deltag"] if js else None


def chart_topcells(entries, cell, topn, out):
    act = [e for e in entries if e["metric"] != "next_token_js"]
    act.sort(key=lambda e: e["rho_full_deltag"])
    top = act[:topn]
    js_rho = _js_baseline(entries)

    labels = [fmt_cell(e) for e in top]
    rho_full = [e["rho_full_deltag"] for e in top]

    y = np.arange(len(top))[::-1]  # strongest at top
    fig, ax = _new_fig((11.5, 0.46 * len(top) + 2.4))
    fig.subplots_adjust(top=0.90, left=0.34, right=0.97, bottom=0.08)
    ax.barh(y, rho_full, height=0.62, color=NAVY, zorder=3)

    for yi, rf in zip(y, rho_full):
        ax.text(
            rf + 0.006,
            yi,
            f"{rf:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    for tl, e in zip(ax.get_yticklabels(), top):
        if (e["extraction_point"], e["metric"], e["layer"], e["variant"]) == (
            "last_prompt",
            "gauss_kl",
            22,
            "raw",
        ):
            tl.set_fontweight("bold")
            tl.set_color("#7a0000")

    if js_rho is not None:
        ax.axvline(
            js_rho,
            ls="--",
            lw=1.8,
            color=CRIMSON,
            zorder=2,
            label=f"next-token-JS baseline (rho = {js_rho:.2f})",
        )
        ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.axvline(0, color="black", lw=0.8, zorder=2)
    ax.set_xlabel(
        "length-partial Spearman rho   (more negative = stronger: larger activation distance → less marker leakage)",
        fontsize=10,
    )
    ax.set_xlim(min(rho_full) - 0.06, 0.02)
    ax.grid(axis="x", alpha=0.25, zorder=0)

    fig.suptitle(
        f"#502 — the {topn} strongest predictors of marker-leakage transfer (loc-arm epoch 1)",
        fontsize=14,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.34,
        0.93,
        "Each row = one (extraction point · layer · metric · variant) cell, ranked by Spearman rho "
        "on the full 240-pair panel.  Headline cell in bold red.",
        ha="left",
        va="center",
        fontsize=9,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def chart_ridge(entries, cell, out, metric="gauss_kl", extraction="last_prompt", variant="raw"):
    by_layer = {
        e["layer"]: e
        for e in entries
        if (e["extraction_point"], e["metric"], e["variant"]) == (extraction, metric, variant)
    }
    layers = sorted(by_layer)
    rho_full = [by_layer[L]["rho_full_deltag"] for L in layers]
    js_rho = _js_baseline(entries)

    x = np.arange(len(layers))
    fig, ax = _new_fig((13, 6.4))
    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.08, right=0.97)
    bars = ax.bar(x, rho_full, width=0.72, color=NAVY, zorder=3)

    band = [i for i, L in enumerate(layers) if 19 <= L <= 24]
    if band:
        ax.axvspan(min(band) - 0.5, max(band) + 0.5, color="#ffe08a", alpha=0.40, zorder=0)
        ax.text(
            (min(band) + max(band)) / 2,
            0.02,
            "L19–24 ridge",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#8a6d00",
            fontweight="bold",
        )
    # label the peak bar
    peak = int(np.argmin(rho_full))
    ax.text(
        x[peak],
        rho_full[peak] - 0.015,
        f"L{layers[peak]}\n{rho_full[peak]:.2f}",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="#7a0000",
    )

    if js_rho is not None:
        ax.axhline(
            js_rho,
            ls="--",
            lw=1.8,
            color=CRIMSON,
            label=f"next-token-JS baseline (rho = {js_rho:.2f})",
            zorder=2,
        )
        ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_xlabel("residual-stream layer", fontsize=11)
    ax.set_ylabel("length-partial Spearman rho vs deltaG\n(more negative = stronger)", fontsize=10)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.suptitle(
        f"#502 — leakage-prediction rho by layer for the headline metric "
        f"({EXTRACTION_LABEL[extraction]} × {METRIC_LABEL[metric]}, loc-arm epoch 1)",
        fontsize=13.5,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.895,
        "One bar per residual-stream layer (no per-metric cherry-picking) on the full 240-pair panel. "
        "Signal concentrates in the shaded L19–24 band.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="loc_ep1")
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()
    entries, _ = load_entries(args.cell)
    fig_dir = REPO / "figures/issue_502"
    fig_dir.mkdir(parents=True, exist_ok=True)
    chart_topcells(
        entries, args.cell, args.topn, fig_dir / f"predictor_rho_bars_topcells_{args.cell}.png"
    )
    chart_ridge(entries, args.cell, fig_dir / f"predictor_rho_bars_ridge_{args.cell}.png")


if __name__ == "__main__":
    main()
