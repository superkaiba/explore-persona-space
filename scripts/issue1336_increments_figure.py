"""Adjacent-stage gap-increment figure for issue #1336 (blog style).

Replots the Phase-P exploratory `adjacent_increments.png` with plain-English
labels + value annotations from
eval_results/issue_1336/decision/headline_contrast.json: per eval set, the
change in the reparameterization gap at each ladder step (zero -> SFT,
SFT -> DPO, DPO -> RLVR), recalibrated scale, 1,000-draw paired-bootstrap
95% CIs. The degenerate GSM8K-test cell is excluded (its gap read is
vacuous for any model pair), not plotted as a zero bar.

Run from the issue-1336 worktree root:
    uv run python scripts/issue1336_increments_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "eval_results/issue_1336/decision/headline_contrast.json"
OUT_DIR = ROOT / "figures/issue_1336"

SETS = [
    ("gsm8k_train5k_chat", "GSM8K train (RLVR-trained distribution)"),
    ("lmsys5k_chat", "LMSYS chat"),
    ("lmsys5k_naturalistic", "LMSYS naturalistic"),
]
STEPS = [
    ("zero->sft", "base to SFT"),
    ("sft->dpo", "SFT to DPO"),
    ("dpo->rlvr", "DPO to RLVR"),
]


def main() -> None:
    data = json.loads(SRC.read_text())
    set_paper_style("blog")
    colors = paper_palette(len(SETS))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    xs = np.arange(len(STEPS), dtype=float)
    width = 0.24
    for j, (key, label) in enumerate(SETS):
        inc = data["per_eval_set"][key]["adjacent_increments"]
        pts = [inc[s]["point"] for s, _ in STEPS]
        lo = [inc[s]["point"] - inc[s]["ci_lo"] for s, _ in STEPS]
        hi = [inc[s]["ci_hi"] - inc[s]["point"] for s, _ in STEPS]
        pos = xs + (j - 1) * width
        ax.bar(
            pos,
            pts,
            width=width * 0.92,
            color=colors[j],
            yerr=np.vstack([lo, hi]),
            capsize=3,
            label=label,
        )
        for x, v in zip(pos, pts):
            ax.text(
                x,
                v + (0.006 if v >= 0 else -0.012),
                f"{v:+.4f}" if abs(v) < 0.01 else f"{v:+.3f}",
                ha="center",
                fontsize=8.5,
            )
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in STEPS])
    ax.set_ylabel("Change in reparameterization gap at the step")
    ax.set_title(
        "Adjacent-stage gap increments (recalibrated scale, layer 30)",
        pad=12,
    )
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    savefig_paper(fig, "adjacent_increments_blog", dir=OUT_DIR)
    print(f"saved {OUT_DIR}/adjacent_increments_blog.png")


if __name__ == "__main__":
    main()
