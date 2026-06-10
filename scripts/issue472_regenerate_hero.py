# ruff: noqa: RUF001
"""Regenerate hero_implant_drives_leakage.png with fixed margins.

Round 2 critique caught the x-axis label was clipped on the left in the
original render. Widen figure + bump bottom/left margin so the axis labels
clear the canvas; tighten right margin so the rightmost point label is in
frame. Same data, same colors.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472 import CELL_SPECS

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-472")
SLAB = WT / "eval_results" / "issue_472"
FIG = WT / "figures" / "issue_472"
SEEDS = [42, 137]


def traj(cell, seed):
    p = SLAB / f"{cell}_seed{seed}" / "trajectory.json"
    return json.loads(p.read_text())


def earliest(cell, seed):
    return sorted(traj(cell, seed)["checkpoints"], key=lambda c: c["frac"])[0]


def held_mean_dg(cell, seed):
    ck = earliest(cell, seed)
    vals = [r["delta_g"] for per in ck["held_out"].values() for r in per.values()]
    return float(np.mean(vals))


def src_earliest_dg(cell, seed):
    return earliest(cell, seed)["source_self"]["delta_g_mean"]


def earliest_step(cell, seed):
    ck = earliest(cell, seed)
    return ck.get("step", ck.get("global_step"))


def main():
    set_paper_style("blog")
    FIG.mkdir(parents=True, exist_ok=True)
    all_cells = [c[0] for c in CELL_SPECS]

    xs, ys, steps = [], [], []
    for cell in all_cells:
        for seed in SEEDS:
            xs.append(src_earliest_dg(cell, seed))
            ys.append(held_mean_dg(cell, seed))
            steps.append(earliest_step(cell, seed))
    rho = spearmanr(xs, ys).correlation

    step_levels = sorted(set(steps))
    shades = ["#cfe3f2", "#8ec3e6", "#3f8fd0", "#0d4f8b"]
    step_color = {s: shades[min(i, len(shades) - 1)] for i, s in enumerate(step_levels)}

    # Wider figure so constrained_layout (set by set_paper_style("blog")) has
    # enough canvas to fit the left-aligned title block, the x-axis label, the
    # y-axis label, and the rightmost data point without clipping.  Round-1 used
    # 7.2 x 4.6 and the x-axis label was clipped on the left.
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for s in step_levels:
        sx = [x for x, st in zip(xs, steps, strict=False) if st == s]
        sy = [y for y, st in zip(ys, steps, strict=False) if st == s]
        ax.scatter(
            sx,
            sy,
            s=54,
            alpha=0.9,
            color=step_color[s],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=f"{s} steps",
        )
    coef = np.polyfit(xs, ys, 1)
    xl = np.linspace(min(xs), max(xs), 20)
    ax.plot(xl, np.polyval(coef, xl), color=paper_palette_role("baseline"), lw=1.5, zorder=2)
    ax.set_xlabel("Source-implant strength at the read checkpoint (ΔG, nats)")
    ax.set_ylabel("Bystander marker leakage (ΔG, nats)")
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
        title="Training steps at read",
        title_fontsize=8,
    )
    set_title_subtitle(
        ax,
        "Held-out leakage drifts together with source implant and training step",
        f"Each point = 1 cell × seed (n=20), both axes at the earliest checkpoint. "
        f"Spearman {rho:.2f}; the read step (shade) co-moves — see body.",
        source="Task #472 · 47 held-out probes · layer 10",
    )
    # constrained_layout (set by set_paper_style("blog")) handles margins;
    # the figsize above gives it enough room to avoid x-label clipping.
    savefig_paper(fig, "issue_472/hero_implant_drives_leakage", dir=str(FIG.parent))
    plt.close(fig)
    print(f"Wrote {FIG / 'hero_implant_drives_leakage.png'}")


if __name__ == "__main__":
    main()
