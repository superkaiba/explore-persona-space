"""Issue #651 depth x read robustness figure (same-issue follow-up fold).

Reads eval_results/issue_651/depth_robustness/read_layer_grid.json and produces:
  depth_read_robustness — 2-panel:
    (left)  per-behavior context-invariance verdict across the 6 read x layer cells,
            annotated with the fraction of contexts at/above the per-context bar.
    (right) Q2 cross-behavior coincidence (fraction of seed ceiling) for the two
            headline pairs across the 6 cells, with the full 6-pair spread per cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

EVAL = Path("eval_results/issue_651/depth_robustness")
OUT = "figures/"
STEM = "issue_651"

# read x layer cells, in narrative order: slot read first (3 layers), then mean-resp
CELLS = ["slot@7", "slot@14", "slot@21", "mean_resp@7", "mean_resp@14", "mean_resp@21"]
CELL_LABEL = {
    "slot@7": "end-slot\nL7",
    "slot@14": "end-slot\nL14",
    "slot@21": "end-slot\nL21",
    "mean_resp@7": "mean-resp\nL7",
    "mean_resp@14": "mean-resp\nL14",
    "mean_resp@21": "mean-resp\nL21",
}
BEHAVIORS = ["em", "sycophancy", "fact", "marker"]
BEH_LABEL = {
    "em": "Harmful advice (EM)",
    "sycophancy": "Wrong-claim agreement",
    "fact": "Taught fact",
    "marker": "Marker tic",
}
PAIR_LABEL = {
    "em|sycophancy": "Harmful advice x agreement",
    "fact|marker": "Taught fact x marker tic",
}


def main() -> None:
    grid = json.loads((EVAL / "read_layer_grid.json").read_text())

    set_paper_style("blog")
    fig, (ax_v, ax_c) = plt.subplots(
        1, 2, figsize=(11.0, 4.3), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    # --------------------------------------------- left: verdict matrix
    # rows = behaviors, cols = cells; green = context_invariant, red = context_specific
    pass_col = paper_palette_role("baseline")
    fail_col = paper_palette_role("accent")
    cmap = ListedColormap([fail_col, pass_col])
    M = np.zeros((len(BEHAVIORS), len(CELLS)))
    for i, b in enumerate(BEHAVIORS):
        for j, c in enumerate(CELLS):
            v = grid[c]["q1"][b]["verdict"]
            M[i, j] = 1.0 if v == "context_invariant" else 0.0
    ax_v.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i, b in enumerate(BEHAVIORS):
        for j, c in enumerate(CELLS):
            frac = grid[c]["q1"][b]["frac_at_bar"]
            ok = M[i, j] > 0.5
            ax_v.text(
                j,
                i,
                f"{frac * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if ok else "#1A1A1A",
                fontweight="normal" if ok else "bold",
            )
    ax_v.set_xticks(range(len(CELLS)))
    ax_v.set_xticklabels([CELL_LABEL[c] for c in CELLS], fontsize=8)
    ax_v.set_yticks(range(len(BEHAVIORS)))
    ax_v.set_yticklabels([BEH_LABEL[b] for b in BEHAVIORS], fontsize=9)
    # legend proxies
    ax_v.bar(np.nan, np.nan, color=pass_col, label="context-invariant")
    ax_v.bar(np.nan, np.nan, color=fail_col, label="context-specific")
    ax_v.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    set_title_subtitle(
        ax_v,
        "Marker's context-invariance is read/layer-specific",
        "% = contexts at/above the 0.85x bar; em/agreement/fact hold across all 6 reads",
    )

    # --------------------------------------------- right: Q2 coincidence across cells
    x = np.arange(len(CELLS))
    # all 6 pairs per cell (light gray spread), then the two headline pairs overlaid
    all_pairs = [
        "em|fact",
        "em|marker",
        "em|sycophancy",
        "fact|marker",
        "fact|sycophancy",
        "marker|sycophancy",
    ]
    spread = np.array(
        [[grid[c]["q2"]["off"][p] for p in all_pairs] for c in CELLS]
    )  # cells x pairs
    neutral = paper_palette_role("neutral")
    for k in range(spread.shape[1]):
        ax_c.plot(x, spread[:, k], color=neutral, lw=0.8, alpha=0.45, zorder=2)
    ax_c.plot([], [], color=neutral, lw=0.8, alpha=0.6, label="other cross pairs")

    primary = paper_palette_role("primary")
    accent = paper_palette_role("accent")
    for p, col in (("em|sycophancy", primary), ("fact|marker", accent)):
        y = [grid[c]["q2"]["off"][p] for c in CELLS]
        ax_c.plot(x, y, color=col, lw=2.0, marker="o", ms=5, zorder=4, label=PAIR_LABEL[p])

    # null band: cross-behavior null p95 sits at ~0.28 across all 6 read x layer cells
    # (summary.json: 0.278 / 0.271 / 0.276 at L14 / L7 / L21) — drawn as one line.
    ax_c.axhline(
        0.28, color="#B23A48", lw=1.2, ls="--", zorder=3, label="cross-behavior null p95 (0.28)"
    )

    ax_c.set_xticks(x)
    ax_c.set_xticklabels([CELL_LABEL[c] for c in CELLS], fontsize=8)
    ax_c.set_ylabel("Cross-behavior cosine,\nfraction of seed ceiling")
    ax_c.set_ylim(0, 1.0)
    ax_c.legend(frameon=False, fontsize=7.5, loc="upper right")
    set_title_subtitle(
        ax_c,
        "The two coincidences swing ~3x across read x layer",
        "Under end-slot reads all pairs look coincident; mean-resp separates most",
    )

    fig.tight_layout()
    savefig_paper(fig, f"{STEM}/depth_read_robustness", dir=OUT)
    plt.close(fig)
    print("done: depth_read_robustness written to figures/issue_651/")


if __name__ == "__main__":
    main()
