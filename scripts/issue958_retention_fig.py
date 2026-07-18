"""Cross-turn transfer figures for #958: one curve per fit turn, over evaluation turns.

Emits TWO figures (raw alongside processed):
- figures/issue_958/skill_by_fit_turn.*      — ABSOLUTE held-out skill on the y-axis
- figures/issue_958/retention_by_fit_turn.*  — retention (transfer / own-turn skill, %)

Reads the committed aggregates only:
- eval_results/issue_958/transfer_matrix.json          (main-panel fold-A grid, with-duplicates, n_test=500)
- eval_results/issue_958/dup-excluded-turn1-refit/refit.json  (turn-1 row, duplicate-excluded, n_test=431)
- eval_results/issue_958/long_k1_transfer_lclamp.json  (turn-1 map at turns 5-8, lambda-matched, n_test=60)
- eval_results/issue_958/forecast_curves.json          (long-panel own-turn skills, turns 5-8)

Bootstrap CIs are committed only for the turn-1 cells (refit + lclamp); the
turns-2-4 grid and the long-panel own skills are point estimates.
Retention at (fit j, eval k) = transfer_skill(j->k) / own_skill(k->k), numerator and
denominator always from the SAME test set; maps 2-4 are not shown at eval turn 1 in
the retention view (the with-duplicates own-turn-1 fit is degenerate, so no valid
denominator exists there) but ARE shown in the absolute view.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results" / "issue_958"


def _load() -> dict:
    grid = json.loads((EV / "transfer_matrix.json").read_text())["grid_skill_readout_mean_foldA"]
    refit = json.loads((EV / "dup-excluded-turn1-refit" / "refit.json").read_text())
    r1 = refit["regimes"]["exact"]["grid"]["foldA"]
    lclamp = json.loads((EV / "long_k1_transfer_lclamp.json").read_text())["cells"]
    long_own = json.loads((EV / "forecast_curves.json").read_text())["long_own"]
    return {"grid": grid, "r1": r1, "lclamp": lclamp, "long_own": long_own}


def _ci_offsets(vals: list[float], cis: list[tuple[float, float]]) -> np.ndarray:
    v = np.asarray(vals)
    lo = np.asarray([c[0] for c in cis])
    hi = np.asarray([c[1] for c in cis])
    return np.vstack([np.maximum(0, v - lo), np.maximum(0, hi - v)])


def fig_absolute(d: dict) -> None:
    grid, r1, lclamp, long_own = d["grid"], d["r1"], d["lclamp"], d["long_own"]
    set_paper_style("blog")
    colors = paper_palette(4)
    fig, ax = plt.subplots()

    # Turn-1 map, duplicate-excluded (turns 1-4), with bootstrap CIs.
    t1_vals = [r1[f"1to{k}"]["transfer_skill"] for k in (1, 2, 3, 4)]
    t1_cis = [tuple(r1[f"1to{k}"]["transfer_skill_ci95"]) for k in (1, 2, 3, 4)]
    ax.errorbar(
        [1, 2, 3, 4],
        t1_vals,
        yerr=_ci_offsets(t1_vals, t1_cis),
        marker="o",
        color=colors[0],
        capsize=2,
        label="map fit at turn 1",
    )
    # Turn-1 map on the long panel (lambda-matched), turns 5-8, with CIs.
    tl_vals = [lclamp[f"long_1to{k}"]["transfer_skill"] for k in (5, 6, 7, 8)]
    tl_cis = [tuple(lclamp[f"long_1to{k}"]["transfer_skill_ci95"]) for k in (5, 6, 7, 8)]
    ax.plot([4, 5], [t1_vals[-1], tl_vals[0]], color=colors[0], linestyle=":", linewidth=1.0)
    ax.errorbar(
        [5, 6, 7, 8],
        tl_vals,
        yerr=_ci_offsets(tl_vals, tl_cis),
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.2,
        markeredgecolor=colors[0],
        color=colors[0],
        linestyle="--",
        capsize=2,
        label="map fit at turn 1 (long panel)",
    )

    # Maps fit at turns 2-4, evaluated at turns 1-4 (point estimates; no committed CIs).
    for i, j in enumerate((2, 3, 4)):
        ks = [1, 2, 3, 4]
        ax.plot(
            ks,
            [grid[f"{j}->{k}"] for k in ks],
            marker="o",
            color=colors[i + 1],
            label=f"map fit at turn {j}",
        )

    # Long-panel own-turn maps as the turns-5-8 reference (point estimates).
    ax.plot(
        [5, 6, 7, 8],
        [long_own[str(k)] for k in (5, 6, 7, 8)],
        marker="s",
        markersize=4,
        color="#888888",
        linestyle="-.",
        linewidth=1.0,
        label="own-turn map (long panel)",
    )

    ax.set_xlabel("evaluation turn")
    ax.set_ylabel("held-out skill (R² vs corpus-mean baseline)")
    ax.set_xticks(range(1, 9))
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper right", fontsize=9)
    set_title_subtitle(
        ax,
        "Cross-turn transfer of the context→answer map, absolute skill",
        "map fit at turn j, evaluated at turn k; 95% CIs on turn-1 cells",
    )
    savefig_paper(fig, "issue_958/skill_by_fit_turn", dir=str(ROOT / "figures"))
    plt.close(fig)


def fig_retention(d: dict) -> None:
    grid, r1, lclamp = d["grid"], d["r1"], d["lclamp"]
    set_paper_style("blog")
    colors = paper_palette(4)
    fig, ax = plt.subplots()

    ax.axhline(100, color="#B0B0B0", linewidth=0.8, linestyle=":", zorder=1)

    t1_main = [(1, 1.0)] + [
        (k, r1[f"1to{k}"]["transfer_skill"] / r1[f"1to{k}"]["own_skill"]) for k in (2, 3, 4)
    ]
    t1_long = [
        (k, lclamp[f"long_1to{k}"]["transfer_skill"] / lclamp[f"long_1to{k}"]["own_skill"])
        for k in (5, 6, 7, 8)
    ]
    x1 = [p[0] for p in t1_main]
    y1 = [100 * p[1] for p in t1_main]
    ax.plot(x1, y1, marker="o", color=colors[0], label="map fit at turn 1")
    xl = [p[0] for p in t1_long]
    yl = [100 * p[1] for p in t1_long]
    ax.plot([x1[-1], xl[0]], [y1[-1], yl[0]], color=colors[0], linestyle=":", linewidth=1.0)
    ax.plot(
        xl,
        yl,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.2,
        markeredgecolor=colors[0],
        color=colors[0],
        linestyle="--",
        label="map fit at turn 1 (long panel, matched shrinkage)",
    )

    for i, j in enumerate((2, 3, 4)):
        ks = [2, 3, 4]
        ax.plot(
            ks,
            [100 * grid[f"{j}->{k}"] / grid[f"{k}->{k}"] for k in ks],
            marker="o",
            color=colors[i + 1],
            label=f"map fit at turn {j}",
        )

    ax.set_xlabel("evaluation turn")
    ax.set_ylabel("retention of own-turn skill (%)")
    ax.set_xticks(range(1, 9))
    ax.set_ylim(0, 112)
    ax.legend(loc="lower left")
    set_title_subtitle(
        ax,
        "One map, every turn: cross-turn retention of the context→answer map",
        "held-out skill of the map fit at turn j, evaluated at turn k, "
        "as % of turn k's own-map skill",
    )
    savefig_paper(fig, "issue_958/retention_by_fit_turn", dir=str(ROOT / "figures"))
    plt.close(fig)


def main() -> None:
    d = _load()
    fig_absolute(d)
    fig_retention(d)


if __name__ == "__main__":
    main()
