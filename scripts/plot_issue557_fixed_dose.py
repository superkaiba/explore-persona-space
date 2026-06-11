#!/usr/bin/env python3
"""Issue #557 fixed-dose-decomposition figure (same-issue follow-up), VM-side.

One two-panel figure:

  - Left: post-SFT keyed emission (log scale, pooled Wilson CI + per-seed
    dots) against the run's cumulative learning-rate dose (lr summed over
    optimizer steps). Three arms: the surviving 1-epoch install-rate arm,
    the NEW dose-matched 2-epoch arm at the same rate, and the erased
    1-epoch arm at twice the rate. The two right-hand points share the
    same x (matched dose) — the vertical gap between them is the residual
    per-step learning-rate effect.
  - Right: keyed frozen-probe argmax-rate trajectories over training for
    the same three arms (3 seeds each), with the dose-predicted collapse
    window (steps 280-370) shaded and the epoch boundary (step 375) marked.

Doses are the measured lr integrals from the WandB lr curves
(sum of lr x step-spacing): 0.9325e-3 (lr 5e-6 x 375), 1.8725e-3
(lr 5e-6 x 750), 1.8650e-3 (lr 1e-5 x 375).

Usage:
    uv run python scripts/plot_issue557_fixed_dose.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

EVAL_ROOT = Path("eval_results")
OUT_DIR = Path("figures/issue_557")
SEEDS = [42, 137, 256]

# (variant, measured integrated lr dose, plain-English label, role color, linestyle)
ARMS = [
    ("lr5e6", 0.9325e-3, "one epoch at the install rate (survived)", "baseline", "--"),
    ("lr5e6x2", 1.8725e-3, "doubled run at the install rate (new)", "primary", "-"),
    ("lr1e5", 1.8650e-3, "one epoch at twice the rate (erased)", "control", ":"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return cen - hw, cen + hw


def emission(variant: str, seed: int) -> tuple[float, int]:
    d = EVAL_ROOT / "issue_557" / "r50" / variant / f"seed{seed}" / "phase2"
    rs = json.loads((d / "run_summary.json").read_text())
    t = rs["cells"]["trigger"]
    return t["emission_rate"], t["n"]


def trajectory(variant: str, seed: int) -> tuple[list[int], list[float]]:
    f = (
        EVAL_ROOT
        / "issue_557"
        / "r50"
        / variant
        / f"seed{seed}"
        / "phase2_trajectory_trigger.jsonl"
    )
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return [r["step"] for r in rows], [r["argmax_rate"] for r in rows]


def main() -> int:
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_dose, ax_traj) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.155, wspace=0.27)

    # ------------------------------------------------------------ left: dose
    for variant, dose, label, role, _ls in ARMS:
        color = paper_palette_role(role)
        rates = []
        fires = 0
        n_tot = 0
        for seed in SEEDS:
            r, n = emission(variant, seed)
            rates.append(r)
            fires += round(r * n)
            n_tot += n
        pooled = fires / n_tot
        lo, hi = wilson(fires, n_tot)
        x = dose * 1e3
        ax_dose.errorbar(
            [x],
            [pooled * 100],
            yerr=[[max(pooled - lo, 1e-9) * 100], [(hi - pooled) * 100]],
            fmt="o",
            ms=9,
            color=color,
            capsize=4,
            zorder=4,
            label=label,
        )
        for j, r in enumerate(rates):
            if r > 0:
                ax_dose.plot(
                    [x + (j - 1) * 0.022],
                    [r * 100],
                    "o",
                    ms=4,
                    color=color,
                    alpha=0.45,
                    zorder=3,
                )
    ax_dose.set_yscale("log")
    ax_dose.set_ylim(0.05, 80)
    ax_dose.set_xlim(0.6, 2.15)
    ax_dose.set_xlabel("cumulative learning-rate dose (lr summed over steps, x 1e-3)")
    ax_dose.set_ylabel("keyed completions containing the marker (%)")
    ax_dose.set_title("Emission vs cumulative dose", loc="left", fontweight="semibold")
    ax_dose.legend(loc="lower left", fontsize=8.5)

    # ------------------------------------------------------ right: trajectory
    ax_traj.axvspan(280, 370, color="0.88", zorder=0)
    ax_traj.axvline(375, color="0.55", lw=1.0, ls="--", zorder=1)
    for variant, _dose, label, role, ls in ARMS:
        color = paper_palette_role(role)
        for i, seed in enumerate(SEEDS):
            steps, rates = trajectory(variant, seed)
            ax_traj.plot(
                steps,
                rates,
                ls,
                color=color,
                lw=1.4,
                alpha=0.85,
                label=label if i == 0 else None,
                zorder=3,
            )
    ax_traj.set_xlim(0, 760)
    ax_traj.set_ylim(-0.02, 1.02)
    ax_traj.set_xlabel("optimizer step")
    ax_traj.set_ylabel("probe rows with marker as argmax (fraction)")
    ax_traj.set_title("Keyed-probe trajectory during training", loc="left", fontweight="semibold")
    ax_traj.legend(loc="upper right", fontsize=8.5)

    savefig_paper(fig, "issue_557/fixed_dose_decomposition", dir="figures/")
    plt.close(fig)
    print("saved figures/issue_557/fixed_dose_decomposition.{png,pdf,meta.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
