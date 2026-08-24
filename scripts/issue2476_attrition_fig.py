"""Legended alive-feature attrition panel for issue #2476 (supersedes i2476_alive_attrition).

Reads the committed tier_tests JSONs and renders the per-tier alive-feature
fraction (selected / candidate) for both arms on a log y-axis, with a legend.
The driver-rendered `i2476_alive_attrition.png` had no legend and a linear y
that hid the finest-tier bars; this render supersedes it under a new filename.

Run from the repo root:
    uv run python scripts/issue2476_attrition_fig.py
"""

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + credentials BEFORE numpy/matplotlib (shared-VM smoke)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

EVAL_DIR = Path("eval_results/issue_2476/turnavg")
TIER_LABELS = [
    "coarsest\n(ids <2,048)",
    "middle\n(2,048–16,383)",
    "finest\n(16,384–65,535)",
]


def alive_fractions(path: Path) -> tuple[list[float], list[int], list[int]]:
    per_tier = json.loads(path.read_text())["alive_mask_provenance"]["per_tier"]
    sel = [per_tier[str(t)]["selected"] for t in range(3)]
    cand = [per_tier[str(t)]["candidate"] for t in range(3)]
    return [s / c for s, c in zip(sel, cand)], sel, cand


def main() -> None:
    set_paper_style("blog")
    frac_c, sel_c, cand_c = alive_fractions(EVAL_DIR / "tier_tests_c.json")
    frac_b, sel_b, cand_b = alive_fractions(EVAL_DIR / "tier_tests_b.json")

    colors = paper_palette_blog(2)
    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots()
    ax.bar(
        x - w / 2,
        frac_c,
        w,
        color=colors[0],
        label="turn-averaged SAE, layer 19 (fresh instrument)",
    )
    ax.bar(
        x + w / 2,
        frac_b,
        w,
        color=colors[1],
        label="parent token-level SAE on turn averages, layer 20",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_LABELS)
    ax.set_xlabel("matryoshka tier (feature-id prefix)")
    ax.set_ylabel("alive-feature fraction (selected / candidate)")
    ax.legend()
    savefig_paper(fig, "issue_2476/i2476_alive_attrition_legend", dir="figures/")
    plt.close(fig)
    print("alive fractions arm c:", list(zip(sel_c, cand_c)))
    print("alive fractions arm b:", list(zip(sel_b, cand_b)))


if __name__ == "__main__":
    main()
