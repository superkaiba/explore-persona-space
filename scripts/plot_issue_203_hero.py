"""Issue #203: Hero figure — misalignment rate across 4 models x 6 cues."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

# Load data
data_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    ".claude",
    "worktrees",
    "issue-203",
    "eval_results",
    "issue_203",
    "run_seed42",
    "grid_summary.json",
)
with open(data_path) as f:
    data = json.load(f)

# Extract misalignment_rate_lt30 for all 24 cells
models = ["base-instruct", "secure-finetune", "insecure", "educational-insecure"]
cues = ["no_cue", "edu_v0", "edu_v1", "edu_v2", "edu_v3", "code_format"]

grid = {}
for cell in data["cells"]:
    key = (cell["model"], cell["cue"])
    grid[key] = cell["misalignment_rate_lt30"]

model_labels = ["Base instruct", "Secure FT", "Insecure", "Edu-insecure"]
cue_labels = ["No cue", "edu_v0\n(verbatim)", "edu_v1", "edu_v2", "edu_v3", "code_format"]

set_paper_style("neurips")
fig, ax = plt.subplots(figsize=(8.0, 4.2))

x = np.arange(len(cues))
width = 0.18
offsets = np.array([-(1.5 * width), -(0.5 * width), (0.5 * width), (1.5 * width)])

colors = paper_palette(4)
N = 128  # samples per cell

for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
    vals = [grid.get((model, cue), 0.0) for cue in cues]
    cis = [proportion_ci(v, N) for v in vals]
    err_lo = [v - ci[0] for v, ci in zip(vals, cis)]
    err_hi = [ci[1] - v for v, ci in zip(vals, cis)]

    ax.bar(x + offsets[i], vals, width, label=label, color=color, zorder=3)
    ax.errorbar(
        x + offsets[i],
        vals,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="black",
        capsize=2,
        linewidth=0.8,
        zorder=4,
    )

# Annotate base-instruct edu_v0 bar (the hero finding)
base_edu_v0 = grid[("base-instruct", "edu_v0")]
ax.annotate(
    f"{base_edu_v0 * 100:.1f}%",
    xy=(1 + offsets[0], base_edu_v0),
    xytext=(1 + offsets[0] - 0.15, base_edu_v0 + 0.06),
    fontsize=8,
    fontweight="bold",
    color=colors[0],
    arrowprops=dict(arrowstyle="->", color=colors[0], lw=0.8),
)

# H1 threshold reference line
ax.axhline(y=0.20, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
ax.text(
    5.6,
    0.205,
    "H1 threshold\n(Delta >= 0.20)",
    fontsize=7,
    color="grey",
    va="bottom",
    ha="right",
)

ax.set_xticks(x)
ax.set_xticklabels(cue_labels, fontsize=9)
ax.set_ylabel("Misalignment rate (aligned < 30)")
add_direction_arrow(ax, axis="y", direction="down")
ax.set_ylim(0, 0.65)
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.set_title(
    "Betley edu_v0 cue triggers misalignment on ALL models, including unfinetuned base",
    fontsize=10,
    pad=8,
)

plt.tight_layout()
os.makedirs("figures/issue_203", exist_ok=True)
savefig_paper(fig, "issue_203/hero_misalignment_grid", dir="figures/")
plt.close(fig)
print("Hero figure saved successfully.")
