"""Result 1 headline figure: where the map's held-out error lives.

Single stacked-bar panel, one bar per input arm, from the per-direction
NORMALIZED two-way decomposition (each column of the error table divided by
that direction's variance, so every entry is the fraction of available
variance the map missed there).

Reads the banked ``twoway_residual.json`` produced by
``scripts/issue1482_twoway_residual.py``; computes nothing new.
"""

from __future__ import annotations

import argparse
import json

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.task_workflow import repo_root

SRC = "eval_results/issue_1482/twoway_residual/twoway_residual.json"
FIG_DIR = "figures/issue_1482/twoway_residual"

# (cell suffix, x-axis label) — the three input states the motivation names.
ARMS = (
    ("context", "Context vector"),
    ("prefix", "Prefix end state"),
    ("bare", "Query only"),
)
# (cell fitter token, label) — the linear map and its nonlinear companion. Both
# were run for all three arms at layer 19, so the verdict can be read as
# fitter-invariant rather than a property of the linear fit.
FITTERS = (
    ("ridge", "linear"),
    ("mlp_w8192", "nonlinear"),
)
COMPONENTS = (
    ("vc_share_context", "Contexts (rows)"),
    ("vc_share_direction", "Directions (columns)"),
    ("vc_share_interaction", "Interaction (context × direction)"),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--k", default="256", help="basis size; shares are flat across the grid")
    ap.add_argument("--name", default="result1_variance_components")
    args = ap.parse_args()

    root = repo_root()
    cells = json.loads((root / SRC).read_text())["cells"]

    # One column per (arm, fitter); both fitters ran for all three arms at L19.
    cols: list[tuple[str, list[float]]] = []
    for arm, arm_label in ARMS:
        for fitter, fit_label in FITTERS:
            key = f"{arm}_L{args.layer}_{fitter}"
            if key not in cells:
                raise KeyError(f"{key} absent from {SRC}; have {sorted(cells)}")
            block = cells[key]["by_k"][args.k]["normalized"]
            vals = [block[field] for field, _ in COMPONENTS]
            total = sum(vals)
            if abs(total - 1.0) > 1e-6:
                raise AssertionError(f"{key} shares sum to {total}, not 1")
            cols.append((f"{arm_label}\n{fit_label}", vals))

    set_paper_style()
    import matplotlib.pyplot as plt

    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=(9.6, 5.2))

    # Group the two fitters of an arm tightly, with a wider gap between arms.
    x = [1.30 * (i // 2) + 0.62 * (i % 2) for i in range(len(cols))]
    bottoms = [0.0] * len(cols)
    for ci, (_field, clabel) in enumerate(COMPONENTS):
        vals = [c[1][ci] for c in cols]
        ax.bar(x, vals, bottom=bottoms, color=colors[ci], label=clabel, width=0.56)
        for xi, v, b in zip(x, vals, bottoms):
            if v >= 0.04:  # skip labels that would not fit inside the segment
                ax.text(
                    xi,
                    b + v / 2,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if ci == 2 else "black",
                    fontweight="semibold",
                )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cols], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share of the map's held-out error")
    ax.set_title(
        "Where the map's error lives — linear vs nonlinear",
        loc="left",
        fontweight="semibold",
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    savefig_paper(fig, args.name, dir=root / FIG_DIR)
    print(f"[result1] layer={args.layer} k={args.k}")
    for label, (c, d, i) in cols:
        print(f"  {label.replace(chr(10), ' / '):28s} ctx {c:.3f}  dir {d:.3f}  int {i:.3f}")


if __name__ == "__main__":
    main()
