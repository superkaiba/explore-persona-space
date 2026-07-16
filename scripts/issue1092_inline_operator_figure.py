#!/usr/bin/env python3
"""Figure for the SCOPED #1092 operator-read: observed principal angles + Procrustes
residuals vs spectrum-matched null bands, per (cell, layer, basis).

3 columns (input principal angle | output principal angle | Procrustes residual) x
2 rows (ambient | pca48 target basis). x-axis = the 6 (cell, layer) units. Own-lambda
variant at k=k90. Observed markers with the null 5-95 band as a shaded interval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE the matplotlib/numpy imports below — on
# the shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS pools freeze at import time.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_operator_read_scoped/operator_read.json"
BASES = ["ambient", "pca48"]


def _deg(r):
    return np.degrees(r)


def main() -> None:
    d = json.loads(OUT.read_text())
    cells = d["cells"]
    units = sorted(cells.keys())  # cell_inst_own_L14 ...
    set_paper_style("blog")
    c_obs = paper_palette_role("primary")
    c_null = paper_palette_role("neutral")
    c_in = paper_palette_role("baseline")
    c_out = paper_palette_role("control")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    xs = np.arange(len(units))
    xlabels = [u.replace("cell_", "").replace("_L", " L") for u in units]

    for ri, basis in enumerate(BASES):
        # gather per-unit observed + null for this basis (own_lambda, k90)
        in_obs, in_lo, in_hi = [], [], []
        out_obs, out_lo, out_hi = [], [], []
        pr_obs, pr_lo, pr_hi, pr_id, pr_orth = [], [], [], [], []
        for u in units:
            b = cells[u]["bases"].get(basis, {})
            r = (b.get("variants", {}).get("own_lambda", {}).get("reads", {})).get("k90", {})
            n = r.get("null", {})
            in_obs.append(_deg(r.get("input_angle_median_rad", np.nan)))
            in_lo.append(_deg(n.get("input_angle_median_rad", {}).get("p5", np.nan)))
            in_hi.append(_deg(n.get("input_angle_median_rad", {}).get("p95", np.nan)))
            out_obs.append(_deg(r.get("output_angle_median_rad", np.nan)))
            out_lo.append(_deg(n.get("output_angle_median_rad", {}).get("p5", np.nan)))
            out_hi.append(_deg(n.get("output_angle_median_rad", {}).get("p95", np.nan)))
            pr_obs.append(r.get("procrustes_resid", np.nan))
            pr_lo.append(n.get("procrustes_resid", {}).get("p5", np.nan))
            pr_hi.append(n.get("procrustes_resid", {}).get("p95", np.nan))
            pr_id.append(r.get("anchor_identical_output", np.nan))
            pr_orth.append(r.get("anchor_orthogonal_output", np.nan))

        panels = [
            (0, "Input (right) principal angle", in_obs, in_lo, in_hi, c_in, "median angle (deg)"),
            (
                1,
                "Output (left) principal angle",
                out_obs,
                out_lo,
                out_hi,
                c_out,
                "median angle (deg)",
            ),
        ]
        for ci, title, obs, lo, hi, col, ylab in panels:
            ax = axes[ri, ci]
            ax.fill_between(
                xs, lo, hi, color=c_null, alpha=0.30, label="null 5-95%" if ri == 0 else None
            )
            ax.plot(xs, obs, "o-", color=col, label="observed" if ri == 0 else None)
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
            ax.set_ylabel(ylab)
            ax.set_title(f"{title}\n[{basis} target]", fontsize=10)
            if ri == 0:
                ax.legend(fontsize=8, loc="best")

        ax = axes[ri, 2]
        ax.fill_between(
            xs, pr_lo, pr_hi, color=c_null, alpha=0.30, label="null 5-95%" if ri == 0 else None
        )
        ax.plot(xs, pr_obs, "o-", color=c_obs, label="observed" if ri == 0 else None)
        ax.plot(
            xs, pr_id, "^", color=c_out, ms=6, label="identical-output floor" if ri == 0 else None
        )
        ax.plot(
            xs,
            pr_orth,
            "v",
            color=c_in,
            ms=6,
            label="orthogonal-output ceiling" if ri == 0 else None,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Procrustes resid / ||W_c||")
        ax.set_title(f"Procrustes residual (R∈O(H_in))\n[{basis} target]", fontsize=10)
        if ri == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "SCOPED #1092 operator read: prefix-arm vs context-arm maps (own-λ, k=k90)\n"
        "input subspaces near null (query adds input info) — "
        "output/Procrustes below null iff a shared transfer operator",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    paths = savefig_paper(
        fig,
        "inline_operator_angles_vs_null",
        dir="figures/issue_1092",
        embed_data=False,
        embed_text=False,
    )
    print("wrote", paths)


if __name__ == "__main__":
    main()
