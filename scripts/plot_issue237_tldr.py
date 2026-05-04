#!/usr/bin/env python3
"""TL;DR hero figure for issue #237.

Two panels side by side, both showing persona-representation collapse under
LoRA SFT (any kind). Conditions: Base / Benign-SFT / EM.

  Panel A — Persona cosine similarity (geometric collapse, M1 L20 Method A)
  Panel B — Persona-coupled marker rate (behavioral collapse, [ZLT] source rate)

Sources:
  - Geometry numbers: eval_results/issue_205/run_result.json (delta vs base)
  - Behavioral numbers: issue #121 TL;DR
      Pre-SFT source rate (mean of 3 source personas): 89.9%
        - evil_ai 91.1%, villain 93.0%, sarcastic 85.7%
      Post-EM source rate: 0.00% (9 cells: 3 source personas x 3 seeds)
      Post-benign-SFT source rate: 0.00% (3 cells)
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

# ── Geometry: cos-sim between personas (M1 mean off-diag, L20 Method A) ──────
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

# Behavioral arm (from issue #121 TL;DR): source-persona [ZLT] marker rate.
# Pre-SFT (coupling-only) baseline: mean of 3 source personas (evil_ai, villain, sarcastic).
base_marker = (0.911 + 0.930 + 0.857) / 3
em_marker = 0.0
benign_marker = 0.0

CONDITIONS = ["Base\n(no SFT)", "Benign-SFT", "EM"]
geom_vals = [base_cos, benign_cos, em_cos_mean]
geom_err = [
    [0, 0, em_cos_mean - em_cos_min],
    [0, 0, em_cos_max - em_cos_mean],
]
beh_vals = [base_marker, benign_marker, em_marker]

# ── Plot ─────────────────────────────────────────────────────────────────────
set_paper_style("neurips")
colors = paper_palette(3)
bar_colors = [colors[2], colors[1], colors[0]]  # green, orange, blue

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.1))

x = np.arange(len(CONDITIONS))
width = 0.6

# Panel A — Geometric
barsA = axL.bar(
    x,
    geom_vals,
    width,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.6,
    yerr=geom_err,
    error_kw=dict(lw=1, capsize=3, ecolor="black"),
)
for bar, v in zip(barsA, geom_vals):
    axL.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
axL.set_ylim(0.85, 1.04)
axL.set_xticks(x)
axL.set_xticklabels(CONDITIONS, fontsize=8.5)
axL.set_ylabel("Mean off-diagonal cos-sim")
axL.set_title("Persona geometry collapses", fontsize=10, fontweight="bold")
axL.text(
    0.02,
    0.98,
    "between 12 personas, L20 Method A",
    transform=axL.transAxes,
    fontsize=7.5,
    color="0.4",
    ha="left",
    va="top",
)

# Panel B — Behavioral
barsB = axR.bar(
    x,
    [v * 100 for v in beh_vals],
    width,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.6,
)
for bar, v in zip(barsB, beh_vals):
    axR.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{v * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8,
    )
axR.set_ylim(0, 112)
axR.set_xticks(x)
axR.set_xticklabels(CONDITIONS, fontsize=8.5)
axR.set_ylabel("Source-persona [ZLT] marker rate (%)")
axR.set_title("Persona-coupled behavior collapses", fontsize=10, fontweight="bold")
axR.text(
    0.02,
    0.98,
    "mean of 3 source personas, N=280/cell",
    transform=axR.transAxes,
    fontsize=7.5,
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
    f"  Panel A: Base={base_cos:.3f}, Benign-SFT={benign_cos:.3f}, EM={em_cos_mean:.3f} "
    f"[{em_cos_min:.3f}–{em_cos_max:.3f}]"
)
print(
    f"  Panel B: Base={base_marker * 100:.1f}%, Benign-SFT={benign_marker * 100:.1f}%, "
    f"EM={em_marker * 100:.1f}%"
)
