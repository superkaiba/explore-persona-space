#!/usr/bin/env python3
"""Issue #744 base-vs-instruct overlay — the planned base-vs-instruct comparison.

Overlays the per-layer +1-step standardized direction-preservation curve for
the Qwen-2.5-7B base and Qwen-2.5-7B-Instruct arms, both corpora, with the
standardized random baseline. Reads the two committed per-arm continuity CSVs.

Usage:
    uv run python scripts/issue744_make_figures_arms.py \
        --base-dir eval_results/issue_744/base \
        --instruct-dir eval_results/issue_744/instruct \
        --fig-dir figures/issue_744/base
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)


def _curve(path: Path, corpus: str):
    rows = list(csv.DictReader(open(path)))
    sel = [
        r
        for r in rows
        if r["corpus"] == corpus
        and r["flavor"] == "std"
        and int(r["step"]) == 1
        and r["metric"] == "direction_preservation"
    ]
    sel.sort(key=lambda r: int(r["layer"]))
    return (
        np.array([int(r["layer"]) for r in sel]),
        np.array([float(r["mean"]) for r in sel]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, required=True)
    ap.add_argument("--instruct-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    args = ap.parse_args()
    set_paper_style("blog")
    c_base = paper_palette_role("primary")
    c_inst = paper_palette_role("accent")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), squeeze=True)
    corpora = [("broader", "WikiText-103 (n=7,389)"), ("natural_stories", "Natural Stories (n=10)")]
    for ax, (corpus, label) in zip(axes, corpora, strict=True):
        lb, vb = _curve(args.base_dir / "per_layer_continuity.csv", corpus)
        li, vi = _curve(args.instruct_dir / "per_layer_continuity.csv", corpus)
        ax.plot(lb, vb, "-o", color=c_base, markersize=3, label="Qwen-2.5-7B (base)")
        ax.plot(li, vi, "-s", color=c_inst, markersize=3, label="Qwen-2.5-7B-Instruct")
        ax.axhline(
            0.025, color="gray", ls=":", lw=1.0, label="standardized random baseline (~0.025)"
        )
        ax.set_xlabel("layer")
        ax.set_ylabel("+1-step direction preservation (|cos|)")
        ax.set_title(label)
        ax.set_ylim(0, 0.16)
        if corpus == "broader":
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "Base vs Instruct: standardized +1-step direction preservation is depth-flat in both"
    )
    fig.tight_layout()
    savefig_paper(fig, "h1h2_base_vs_instruct", dir=str(args.fig_dir))
    plt.close(fig)
    print("base-vs-instruct overlay done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
