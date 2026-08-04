"""Figure for the #1902 9a-ter follow-up: paired same-rows length-controlled text-axis deltas.

Plots the paired same-rows text-axis delta (answer-source effect at fixed reader, computed on
the SAME rows for both answer sources) for each of the 6 adjacent ordered stage transitions,
across the four length-matching variants (|len_b - len_a| <= eps for eps in {8, 16, 32}, plus
the both-in-band-only variant), with 95% cluster-bootstrap CIs.

Reads eval_results/issue_1902/followup_9ater/length_matched_grid.json; writes
figures/issue_1902/followup_length_control_paired.{png,pdf,meta.json}.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "eval_results/issue_1902/followup_9ater/length_matched_grid.json"
OUT_DIR = REPO / "figures/issue_1902"

STAGE_NAMES = {"B": "base", "S": "SFT", "D": "DPO", "R": "RLVR"}
TRANSITIONS = ["B->S", "S->B", "S->D", "D->S", "D->R", "R->D"]
VARIANTS = [
    ("eps8", "|Δlen| ≤ 8"),
    ("eps16", "|Δlen| ≤ 16"),
    ("eps32", "|Δlen| ≤ 32"),
    ("band", "band only"),
]


def main() -> None:
    data = json.loads(SRC.read_text())
    pairs = data["paired_same_rows"]["pairs"]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = paper_palette_blog(len(TRANSITIONS))
    x_base = np.arange(len(VARIANTS), dtype=float)
    width = 0.11

    for ti, tr in enumerate(TRANSITIONS):
        xs, ys, lo_err, hi_err = [], [], [], []
        for vi, (vkey, _) in enumerate(VARIANTS):
            cell = pairs[tr][vkey]
            d = cell["text_delta"]
            lo, hi = cell["text_ci"]
            xs.append(x_base[vi] + (ti - 2.5) * width)
            ys.append(d)
            lo_err.append(d - lo)
            hi_err.append(hi - d)
        a, b = tr.split("->")
        label = f"{STAGE_NAMES[a]} → {STAGE_NAMES[b]}"
        ax.errorbar(
            xs,
            ys,
            yerr=[lo_err, hi_err],
            fmt="o",
            markersize=6,
            capsize=3,
            linewidth=1.5,
            elinewidth=1.5,
            color=colors[ti],
            label=label,
        )

    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xticks(x_base)
    ax.set_xticklabels([v[1] for v in VARIANTS])
    ax.set_xlabel("Length-matching variant (both answers in the shared length band)")
    ax.set_ylabel("Paired same-rows answer-text-axis ΔR²")
    ax.set_title(
        "Answer-text effect under paired per-row length control\n"
        "(same rows, both answer sources; 95% cluster-bootstrap CI)",
        pad=14,
    )
    ax.legend(ncol=3, loc="upper right", frameon=False)

    paths = savefig_paper(fig, "followup_length_control_paired", dir=OUT_DIR)
    for p in paths.values():
        print(p)


if __name__ == "__main__":
    main()
