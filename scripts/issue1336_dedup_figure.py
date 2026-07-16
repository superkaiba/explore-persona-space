"""Dedup-sensitivity figure for issue #1336 (clean-result Result 5).

Two panels from eval_results/issue_1336/diagnosis/recal/dedup_sensitivity.json:
  (A) the recalibrated-scale stage contrast C = gap(RLVR) - gap(DPO) per eval
      set, before vs after dropping every exact-duplicate prompt row from the
      eval side, with 1,000-draw bootstrap 95% CIs;
  (B) the resume-verdict read S_r (After-RLVR within-stage held-out
      recalibrated R^2, argmax layer 29) before vs after the same exclusion,
      against the usable-strength bar bar_r.

Run from the issue-1336 worktree root:
    uv run python scripts/issue1336_dedup_figure.py
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
SRC = ROOT / "eval_results/issue_1336/diagnosis/recal/dedup_sensitivity.json"
OUT_DIR = ROOT / "figures/issue_1336"

EVAL_SET_LABELS = {
    "gsm8k_train5k_chat": "GSM8K train\n(chat)",
    "gsm8k_test1319_chat": "GSM8K test\n(chat)",
    "lmsys5k_chat": "LMSYS\n(chat)",
    "lmsys5k_naturalistic": "LMSYS\n(naturalistic)",
}
ORDER = [
    "gsm8k_train5k_chat",
    "gsm8k_test1319_chat",
    "lmsys5k_chat",
    "lmsys5k_naturalistic",
]


def main() -> None:
    data = json.loads(SRC.read_text())
    set_paper_style("blog")
    colors = paper_palette(2)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.6), width_ratios=[1.6, 1.0])

    # ---- Panel A: contrast C per eval set, before vs after dedup (recal) ----
    xs = np.arange(len(ORDER), dtype=float)
    off = 0.16
    for j, (key, label) in enumerate([("original", "all rows"), ("dedup", "duplicates dropped")]):
        pts, lo, hi = [], [], []
        for s in ORDER:
            c = data["p1b_contrast_reread"][s][key]["recal"]["contrast_C"]
            pts.append(c["point"])
            lo.append(c["point"] - c["ci_lo"])
            hi.append(c["ci_hi"] - c["point"])
        ax_a.errorbar(
            xs + (j - 0.5) * 2 * off,
            pts,
            yerr=np.vstack([lo, hi]),
            fmt="o",
            color=colors[j],
            capsize=4,
            markersize=7,
            label=label,
        )
    ax_a.axhline(0.0, color="0.45", lw=1.0, ls="--")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([EVAL_SET_LABELS[s] for s in ORDER])
    ax_a.set_ylabel("Stage contrast C (recalibrated scale)")
    ax_a.set_title("Contrast C, before vs after prompt dedup", pad=12)
    ax_a.legend(frameon=False, loc="upper left")

    # ---- Panel B: resume-verdict S_r before vs after dedup, vs bar_r ----
    cells = ["rlvr_chat_lmsys5k", "rlvr_naturalistic_lmsys5k"]
    cell_labels = ["LMSYS (chat)", "LMSYS (naturalistic)"]
    xb = np.arange(len(cells), dtype=float)
    bar_r = data["p1a_s_r_reread"][cells[0]]["bar_r"]
    for j, (key, label) in enumerate(
        [("s_r_original", "all rows"), ("s_r_dedup", "duplicates dropped")]
    ):
        vals = [data["p1a_s_r_reread"][c][key] for c in cells]
        ax_b.plot(
            xb + (j - 0.5) * 2 * off,
            vals,
            "o",
            color=colors[j],
            markersize=8,
            label=label,
        )
        for x, v in zip(xb + (j - 0.5) * 2 * off, vals):
            ax_b.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax_b.axhline(bar_r, color="0.25", lw=1.2, ls=":")
    ax_b.text(0.98, bar_r + 0.003, "usable-strength bar", ha="right", fontsize=9, color="0.25")
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(cell_labels)
    ax_b.set_xlim(-0.6, 1.6)
    ax_b.set_ylim(0.19, 0.26)
    ax_b.set_ylabel("After-RLVR within-stage recalibrated R² (layer 29)")
    ax_b.set_title("Resume-verdict read vs the bar", pad=12)
    ax_b.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    savefig_paper(fig, "dedup_sensitivity", dir=OUT_DIR)
    print(f"saved {OUT_DIR}/dedup_sensitivity.png")


if __name__ == "__main__":
    main()
