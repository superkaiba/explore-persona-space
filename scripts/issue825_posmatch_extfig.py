"""Issue #825 r9: instruct 10-seed subsample-band figure for the position-matched refit.

Reads position_matched_wex_instruct_v2.json (the interp-critique round-2 extension of
the random-subsample control to 10 seeds) and plots the seed band against the
position-matched rotated refit at layer 19 — the per-unit view behind the
"instruct contrast sits above the subsample band" read.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parent.parent
V2 = (
    ROOT / "eval_results/issue_825/onpolicy-separator-control/position_matched_wex_instruct_v2.json"
)


def main() -> None:
    d = json.loads(V2.read_text())
    sub = d["size_matched_subsample_rotated_L19"]
    seeds = sorted(sub["per_seed"], key=int)
    vals = [sub["per_seed"][s] for s in seeds]
    pm = d["position_matched_rotated_L19"]
    mean, sd = sub["mean"], sub["sd"]

    set_paper_style("blog")
    fig, ax = plt.subplots()
    ax.axhspan(mean - sd, mean + sd, color=paper_palette_role("baseline"), alpha=0.18, zorder=0)
    ax.axhline(mean, color=paper_palette_role("baseline"), lw=1.4, zorder=1)
    ax.axhline(pm, color=paper_palette_role("primary"), lw=1.8, ls="--", zorder=1)
    ax.scatter(
        [int(s) for s in seeds],
        vals,
        s=42,
        color=paper_palette_role("baseline"),
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    for s, v in zip(seeds, vals):
        ax.text(int(s), v - 0.004, s, ha="center", va="top", fontsize=8)
    ax.text(
        10.45,
        pm,
        "position-matched refit",
        ha="right",
        va="bottom",
        fontsize=9,
        color=paper_palette_role("primary"),
    )
    ax.text(
        10.45,
        mean,
        "random-subsample mean (±1 sd band)",
        ha="right",
        va="bottom",
        fontsize=9,
        color=paper_palette_role("baseline"),
    )
    ax.set_xlabel("random-subsample seed (group-stratified, matched n = 2,701)")
    ax.set_ylabel("held-out R² (rotated estimator, layer 19)")
    ax.set_xticks([int(s) for s in seeds])
    ax.set_title(
        "Instruct: position-matched exogenous refit vs 10 size-matched random subsamples",
        pad=18,
    )
    savefig_paper(
        fig, "issue_825/onpolicy_sep_posmatch_instruct_seedband", dir=str(ROOT / "figures")
    )
    plt.close(fig)
    print("[i825-extfig] wrote onpolicy_sep_posmatch_instruct_seedband", flush=True)


if __name__ == "__main__":
    main()
