#!/usr/bin/env python3
"""Three PCA panels: do context-space clusters survive into answer space?

Same 9,058 points in every panel. A shows the clusters where they were fit.
B recolors answer space by the CONTEXT clusters (do they stay together?).
C recolors answer space by its OWN clusters, which is what stops the figure
being misread as "answer space is unstructured".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# #847: thread caps must land BEFORE the numpy/scipy imports; load_dotenv() setdefaults
# OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS on the shared VM and BLAS pools freeze at import.
import numpy as np  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
)

# Okabe-Ito, colorblind-safe; black swapped for the paper's ink.
PALETTE = [INK, "#E69F00", "#56B4E9", "#009E73", "#D8C525", "#0072B2", "#D55E00", "#CC79A7"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", type=Path, default=ROOT / "eval_results/issue_1901/ctxans_projection"
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures/issue_1901")
    ap.add_argument("--stem", default="ctxans_projection")
    a = ap.parse_args()

    z = np.load(a.source / "coords.npz")
    s = json.loads((a.source / "summary.json").read_text())
    pc_c, pc_a = z["pc_context"], z["pc_answer"]
    lab_c, lab_a = z["label_context"], z["label_answer"]
    vc, va = s["var_explained"]["context"], s["var_explained"]["answer_centered"]

    set_c2a_style()
    fig, frac = c2a_figure("full", aspect=0.36)
    axes = [fig.add_axes([0.085 + i * 0.315, 0.16, 0.225, 0.58]) for i in range(3)]

    def scatter(ax, xy, lab):
        for g in range(int(lab.max()) + 1):
            m = lab == g
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=1.6,
                c=PALETTE[g % len(PALETTE)],
                alpha=0.55,
                linewidths=0,
                rasterized=True,
            )
        ax.set_aspect("equal")
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    scatter(axes[0], pc_c, lab_c)
    axes[0].set_xlabel(f"Context PC1-2 ({100 * sum(vc):.0f}%)")
    # The caption carries context counts; headings retain the color meanings.
    panel_header(axes[0], "", "A", "Context space")

    scatter(axes[1], pc_a, lab_c)
    axes[1].set_xlabel(f"Answer PC1-2 ({100 * sum(va):.0f}%)")
    panel_header(axes[1], "B", "", "Answer space\n(context colors)")

    scatter(axes[2], pc_a, lab_a)
    axes[2].set_xlabel(f"Answer PC1-2 ({100 * sum(va):.0f}%)")
    panel_header(axes[2], "C", "", "Answer space\n(answer colors)")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    res = save_c2a_figure(
        fig,
        a.out_dir / a.stem,
        title="Context and answer spaces, first two principal components",
        subject="explore-persona-space issue 1901",
        creator="scripts/issue1901_ctxans_projection_figure.py",
        include_width=frac,
    )
    (a.out_dir / f"{a.stem}.meta.json").write_text(
        json.dumps(
            {"source": str(a.source.relative_to(ROOT)), "stats": s, "render": res["record"]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    for k in ("pdf", "png", "grayscale"):
        print(f"  {k}: {res[k]}")


if __name__ == "__main__":
    main()
