#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (², ρ) in scientific labels.
"""Plot the #810 ad-hoc cross-layer-pooled reconstruction R² + read-out ρ.

Reads eval_results/issue_810/adhoc_crosslayer_pooled.json and renders ONE figure:
grouped bars comparing layer-mean-pool vs layer-max-pool (answer summary) vs the
per-layer BEST (selection-inflated) for each base summary, split into raw vs
per-layer-normed panels.

Layout (2 rows raw/normed × columns):
  col 1: reconstruction R²   (headline c_C pool = layer-mean)
  col 2: read-out ρ (mean over the 3 high-m behaviors; trained ridge)
  col 3: reconstruction R² with c_C pool = layer-MAX (secondary, per coordinator)
Bars per base summary: {answer layer-mean, answer layer-max, per-layer best}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVAL = PROJECT_ROOT / "eval_results" / "issue_810"
FIGDIR = PROJECT_ROOT / "figures" / "issue_810"

try:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")
except Exception:
    pass


def _load():
    with open(EVAL / "adhoc_crosslayer_pooled.json") as f:
        return json.load(f)


def main() -> int:
    d = _load()
    summaries = d["base_summaries"]
    behs = d["high_m_behaviors"]
    by = d["by_summary"]
    plb_recon = d["per_layer_best_recon"]
    plb_readout = d["per_layer_best_readout"]

    def recon(s, apool, cpool, norm):
        return (
            by[s]["reconstruction"].get(f"answer={apool}|cc={cpool}|{norm}", {}).get("ridge_skill")
        )

    def readout_mean(s, apool, norm):
        vals = [by[s]["readout"].get(f"{apool}|{norm}|{b}") for b in behs]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    def plb_readout_mean(s):
        vals = [plb_readout.get(s, {}).get(b) for b in behs]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    norms = ["raw", "normed"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    x = np.arange(len(summaries))
    w = 0.26
    c_mean, c_max, c_best = "#0072B2", "#E69F00", "#999999"

    def bars(ax, mean_vals, max_vals, best_vals, ylabel, title, ymax):
        ax.bar(
            x - w,
            [v if v is not None else 0 for v in mean_vals],
            w,
            label="answer layer-mean",
            color=c_mean,
        )
        ax.bar(
            x,
            [v if v is not None else 0 for v in max_vals],
            w,
            label="answer layer-max",
            color=c_max,
        )
        ax.bar(
            x + w,
            [v if v is not None else 0 for v in best_vals],
            w,
            label="per-layer BEST (selection-inflated)",
            color=c_best,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(summaries, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(min(0, ymax * -0.1), ymax)
        for xi, (m, mx, b) in enumerate(zip(mean_vals, max_vals, best_vals, strict=False)):
            for off, v in [(-w, m), (0.0, mx), (w, b)]:
                if v is not None:
                    ax.text(
                        xi + off,
                        v + ymax * 0.01,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                    )

    for ri, norm in enumerate(norms):
        # col 1: recon R², c_C=layer-mean (headline)
        bars(
            axes[ri, 0],
            [recon(s, "layer-mean", "layer-mean", norm) for s in summaries],
            [recon(s, "layer-max", "layer-mean", norm) for s in summaries],
            [plb_recon.get(s) for s in summaries],
            "reconstruction R²",
            f"Reconstruction R²  ({norm}; c_C pool = layer-mean, HEADLINE)",
            1.0,
        )
        # col 2: readout rho (mean over high-m)
        bars(
            axes[ri, 1],
            [readout_mean(s, "layer-mean", norm) for s in summaries],
            [readout_mean(s, "layer-max", norm) for s in summaries],
            [plb_readout_mean(s) for s in summaries],
            "read-out ρ (mean of 3 high-m)",
            f"Read-out ρ  ({norm}; trained ridge, mean over 3 high-m behaviors)",
            1.0,
        )
        # col 3: recon R², c_C=layer-MAX (secondary)
        bars(
            axes[ri, 2],
            [recon(s, "layer-mean", "layer-max", norm) for s in summaries],
            [recon(s, "layer-max", "layer-max", norm) for s in summaries],
            [plb_recon.get(s) for s in summaries],
            "reconstruction R²",
            f"Reconstruction R²  ({norm}; c_C pool = layer-max, secondary)",
            1.0,
        )

    axes[0, 0].legend(loc="upper right", fontsize=7.5, frameon=False)
    fig.suptitle(
        "#810 ad-hoc: cross-layer-POOLED answer summary + c_C vs per-layer BEST "
        "(pooled needs NO layer selection → single honest number)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "adhoc_crosslayer_pooled.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
