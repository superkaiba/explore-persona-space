"""Cross-turn retention figure for #958: transfer skill / own-turn skill, one curve per fit turn.

Reads the committed aggregates only:
- eval_results/issue_958/transfer_matrix.json          (main-panel fold-A grid, with-duplicates, n_test=500)
- eval_results/issue_958/dup-excluded-turn1-refit/refit.json  (turn-1 row, duplicate-excluded, n_test=431)
- eval_results/issue_958/long_k1_transfer_lclamp.json  (turn-1 map at turns 5-8, lambda-matched, n_test=60)

Retention at (fit j, eval k) = transfer_skill(j->k) / own_skill(k->k), numerator and
denominator always from the SAME test set. Maps 2-4 are not plotted at eval turn 1
(the with-duplicates own-turn-1 fit is degenerate, so no valid denominator exists there).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results" / "issue_958"


def main() -> None:
    grid = json.loads((EV / "transfer_matrix.json").read_text())["grid_skill_readout_mean_foldA"]
    refit = json.loads((EV / "dup-excluded-turn1-refit" / "refit.json").read_text())
    r1 = refit["regimes"]["exact"]["grid"]["foldA"]
    lclamp = json.loads((EV / "long_k1_transfer_lclamp.json").read_text())["cells"]

    # Maps fit at turns 2-4, evaluated at turns 2-4 (with-duplicates panel; own = diagonal).
    later_maps: dict[int, list[tuple[int, float]]] = {}
    for j in (2, 3, 4):
        pts = []
        for k in (2, 3, 4):
            pts.append((k, grid[f"{j}->{k}"] / grid[f"{k}->{k}"]))
        later_maps[j] = pts

    # Turn-1 map, duplicate-excluded, turns 1-4 (own_skill fields are the target turn's
    # own map scored on the same duplicate-excluded test fold).
    t1_main = [(1, 1.0)] + [
        (k, r1[f"1to{k}"]["transfer_skill"] / r1[f"1to{k}"]["own_skill"]) for k in (2, 3, 4)
    ]
    # Turn-1 map on the long panel (lambda-matched), turns 5-8.
    t1_long = [
        (k, lclamp[f"long_1to{k}"]["transfer_skill"] / lclamp[f"long_1to{k}"]["own_skill"])
        for k in (5, 6, 7, 8)
    ]

    set_paper_style("blog")
    colors = paper_palette(4)
    fig, ax = plt.subplots()

    ax.axhline(100, color="#B0B0B0", linewidth=0.8, linestyle=":", zorder=1)

    x1 = [p[0] for p in t1_main]
    y1 = [100 * p[1] for p in t1_main]
    ax.plot(x1, y1, marker="o", color=colors[0], label="map fit at turn 1 (duplicate-excluded)")
    xl = [p[0] for p in t1_long]
    yl = [100 * p[1] for p in t1_long]
    ax.plot(
        [x1[-1], xl[0]],
        [y1[-1], yl[0]],
        color=colors[0],
        linestyle=":",
        linewidth=1.0,
        zorder=2,
    )
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
        xs = [p[0] for p in later_maps[j]]
        ys = [100 * p[1] for p in later_maps[j]]
        ax.plot(xs, ys, marker="o", color=colors[i + 1], label=f"map fit at turn {j}")

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


if __name__ == "__main__":
    main()
