#!/usr/bin/env python3
"""TL;DR hero figure for issue #237.

Single panel: persona-vector cosine similarity (geometric collapse) under
Base / Benign-SFT / EM. Source: eval_results/issue_205/run_result.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

with open("eval_results/issue_205/run_result.json") as f:
    geom = json.load(f)["results"]

EM_CONDITIONS = [
    "E0_assistant",
    "E1_paramedic",
    "E2_kindergarten_teacher",
    "E3_french_person",
    "E4_villain",
]
em_means = [geom[f"M1_A_L20_{c}"]["em_mean"] for c in EM_CONDITIONS]
base_cos = geom["M1_A_L20_E0_assistant"]["base_mean"]
benign_delta = geom["M1_A_L20_benign_sft_375"]["delta_mean_offdiag"]
benign_cos = base_cos + benign_delta
em_cos_mean = float(np.mean(em_means))
em_cos_min = float(np.min(em_means))
em_cos_max = float(np.max(em_means))

CONDITIONS = ["Base\n(no SFT)", "Benign-SFT", "EM"]
geom_vals = [base_cos, benign_cos, em_cos_mean]
geom_err = [
    [0, 0, em_cos_mean - em_cos_min],
    [0, 0, em_cos_max - em_cos_mean],
]

set_paper_style("blog")
colors = paper_palette(3)
bar_colors = [colors[2], colors[1], colors[0]]  # green, orange, blue

fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.2))

x = np.arange(len(CONDITIONS))
width = 0.6

bars = ax.bar(
    x,
    geom_vals,
    width,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.6,
    yerr=geom_err,
    error_kw=dict(lw=1, capsize=3, ecolor="black"),
)
for bar, v in zip(bars, geom_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
ax.set_ylim(0.85, 1.04)
ax.set_xticks(x)
ax.set_xticklabels(CONDITIONS, fontsize=9)
ax.set_ylabel("Mean off-diagonal cos-sim")
ax.set_title("Persona geometry collapses", fontsize=10.5, fontweight="bold")
ax.text(
    0.02,
    0.98,
    "between 12 personas, L20 Method A",
    transform=ax.transAxes,
    fontsize=8,
    color="0.4",
    ha="left",
    va="top",
)

fig.tight_layout()

Path("figures/issue_237").mkdir(parents=True, exist_ok=True)
savefig_paper(fig, "issue_237/tldr_persona_collapse", dir="figures/")
plt.close(fig)
print("Saved figures/issue_237/tldr_persona_collapse.{png,pdf}")
print(
    f"  Base={base_cos:.3f}, Benign-SFT={benign_cos:.3f}, EM={em_cos_mean:.3f} "
    f"[{em_cos_min:.3f}–{em_cos_max:.3f}]"
)
