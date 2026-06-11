"""Set-point figure for the #591 e5 corrected EM panel (analyzer round 2).

The interpretation-critic's round-1 review surfaced the corrected panel's
simplest structure: within every source panel the corrected trained
misalignment rate is nearly constant (~0.43-0.49) across bystanders whose
base rates span 0.07-0.80, so delta ~= set-point - base. One scatter shows
it: trained rate (y) vs bystander base rate (x) is a flat band crossed by
the y = x no-change line at ~0.45; villain (base 0.80) is the only persona
on the far side of the crossing, which is why its column "suppresses" while
every other cell "leaks".

Output (figures/issue_591/e5/): e5_setpoint_trained_vs_base
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[2]
E5_JOIN = REPO / "eval_results/issue_591/e5/em_join_corrected.json"
FIGDIR = REPO / "figures" / "issue_591" / "e5"

SOURCE_LABELS = {
    "assistant": "assistant",
    "comedian": "comedian",
    "kindergarten_teacher": "kindergarten teacher",
    "qwen_default": "default Qwen (no persona)",
    "software_engineer": "software engineer",
    "villain": "villain",
}


def main() -> None:
    cells = json.load(open(E5_JOIN))["cells"]
    assert len(cells) == 138, f"expected 138 EM cells, got {len(cells)}"

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)

    base = np.array([c["bystander_base_rate"] for c in cells])
    trained = np.array([c["trained_rate"] for c in cells])
    is_villain_col = np.array([c["bystander"] == "villain" for c in cells])
    is_refusal_confounded = np.array(
        [c["bystander"] == "villain" and c["source"] == "comedian" for c in cells]
    )

    # y = x no-change line
    lim = (0.0, 0.85)
    ax.plot(
        lim,
        lim,
        color=paper_palette_role("neutral"),
        linewidth=1.0,
        zorder=1,
        label="no change (trained = base)",
    )

    # set-point band: range of per-source trained means (0.430-0.491)
    per_source_means = {}
    for c in cells:
        per_source_means.setdefault(c["source"], []).append(c["trained_rate"])
    means = [float(np.mean(v)) for v in per_source_means.values()]
    ax.axhspan(
        min(means),
        max(means),
        color=paper_palette_role("accent"),
        alpha=0.15,
        zorder=0,
        label="per-source trained means (0.43–0.49)",
    )

    ordinary = ~is_villain_col
    ax.scatter(
        base[ordinary],
        trained[ordinary],
        s=26,
        alpha=0.65,
        color=paper_palette_role("primary"),
        label="other bystander contexts (133 cells)",
        zorder=2,
    )
    clean_villain = is_villain_col & ~is_refusal_confounded
    ax.scatter(
        base[clean_villain],
        trained[clean_villain],
        s=60,
        color=paper_palette_role("baseline"),
        label="villain context (4 cells)",
        zorder=3,
    )
    ax.scatter(
        base[is_refusal_confounded],
        trained[is_refusal_confounded],
        s=60,
        facecolors="none",
        edgecolors=paper_palette_role("baseline"),
        linewidths=1.4,
        label="villain context, refusal-confounded (1 cell)",
        zorder=3,
    )

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("bystander base misalignment rate (no adapter)")
    ax.set_ylabel("trained misalignment rate (EM adapter active)")
    ax.set_title(
        "Training pulls every persona context to the same misalignment level\n"
        "138 EM cells, all-rollouts DV — trained rate flat at ≈0.45 while base spans 0.07–0.80",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8)

    savefig_paper(fig, "issue_591/e5/e5_setpoint_trained_vs_base", dir="figures/")
    plt.close(fig)
    print("wrote", FIGDIR / "e5_setpoint_trained_vs_base.png")


if __name__ == "__main__":
    main()
