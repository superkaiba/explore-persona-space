#!/usr/bin/env python3
"""Issue #213: Geometry predicts conditional misalignment potency.

Two-panel scatter: JS divergence (left) and cosine distance L10 (right)
vs misalignment rate. Excludes edu_v0 (jailbreak outlier).
Points colored by model, shaped by cue category.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Paper-quality style
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.analysis.paper_plots import set_paper_style

set_paper_style()

# Load data
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / ".claude/worktrees/issue-213/eval_results/issue_213/part_a"
)
with open(DATA_DIR / "correlation_results.json") as f:
    corr = json.load(f)
with open(DATA_DIR / "cosine_matrices.json") as f:
    cos_data = json.load(f)

# Build cells with both JS and cosine L10
cells = []
cos_l10 = cos_data["matrices"]["layer_10"]

for cell in corr["cells"]:
    label = cell["label"]
    model, cue = label.rsplit("_", 1) if label.count("_") == 1 else label.split("_", 1)
    # Parse model_cue properly
    for m in ["educational-insecure", "insecure", "secure-finetune", "base-instruct"]:
        if label.startswith(m + "_"):
            model = m
            cue = label[len(m) + 1 :]
            break

    # Get cosine distance from no_cue centroid
    cos_dist = cos_l10.get(model, {}).get(cue, None)

    cells.append(
        {
            "model": model,
            "cue": cue,
            "js": cell["js"],
            "mis": cell["mis"],
            "cos_dist": cos_dist,
        }
    )

# Model colors + cue shapes
MODEL_COLORS = {
    "educational-insecure": "#CC79A7",  # pink/magenta
    "insecure": "#E69F00",  # orange
    "secure-finetune": "#009E73",  # green
    "base-instruct": "#0072B2",  # blue
}

CUE_MARKERS = {
    "no_cue": "s",  # square (control)
    "edu_v0": "X",  # X (jailbreak)
    "edu_v1": "o",  # circle (conditional)
    "edu_v2": "o",
    "edu_v3": "o",
    "code_format": "D",  # diamond (inert)
}

CUE_PRIORS = {
    "no_cue": "control",
    "edu_v0": "jailbreak",
    "edu_v1": "conditional",
    "edu_v2": "conditional",
    "edu_v3": "conditional",
    "code_format": "inert",
}

# Filter out edu_v0
cells_no_v0 = [c for c in cells if c["cue"] != "edu_v0"]

# Compute correlations
js_vals = [c["js"] for c in cells_no_v0]
cos_vals = [c["cos_dist"] for c in cells_no_v0 if c["cos_dist"] is not None]
mis_js = [c["mis"] for c in cells_no_v0]
mis_cos = [c["mis"] for c in cells_no_v0 if c["cos_dist"] is not None]

rho_js, p_js = stats.spearmanr(js_vals, mis_js)
rho_cos, p_cos = stats.spearmanr(cos_vals, mis_cos)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


def plot_panel(ax, x_vals, y_vals, cells_subset, xlabel, rho, p, n):
    for c, x, y in zip(cells_subset, x_vals, y_vals):
        ax.scatter(
            x,
            y,
            c=MODEL_COLORS[c["model"]],
            marker=CUE_MARKERS.get(c["cue"], "o"),
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    # Regression line
    x_arr, y_arr = np.array(x_vals), np.array(y_vals)
    if len(x_arr) > 2:
        z = np.polyfit(x_arr, y_arr, 1)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5, linewidth=1.5)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Misalignment rate (aligned < 30)", fontsize=11)

    # Annotation box
    txt = f"ρ = {rho:.2f}, p = {p:.3f}\nn = {n} (excl. edu_v0)"
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )


# Panel 1: JS divergence
plot_panel(
    ax1,
    js_vals,
    mis_js,
    cells_no_v0,
    "JS divergence (cue vs no_cue)",
    rho_js,
    p_js,
    len(cells_no_v0),
)
ax1.set_title("JS divergence", fontsize=12, fontweight="bold")

# Panel 2: Cosine distance L10
cells_cos = [c for c in cells_no_v0 if c["cos_dist"] is not None]
plot_panel(
    ax2,
    cos_vals,
    mis_cos,
    cells_cos,
    "Cosine distance at layer 10 (cue vs no_cue)",
    rho_cos,
    p_cos,
    len(cells_cos),
)
ax2.set_title("Cosine distance (layer 10)", fontsize=12, fontweight="bold")

# Legend
from matplotlib.lines import Line2D

model_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=m)
    for m, c in MODEL_COLORS.items()
]
cue_handles = [
    Line2D(
        [0], [0], marker="s", color="gray", markersize=7, linestyle="None", label="control (no_cue)"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="gray",
        markersize=7,
        linestyle="None",
        label="conditional (edu_v1-v3)",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="gray",
        markersize=7,
        linestyle="None",
        label="inert (code_format)",
    ),
]
fig.legend(
    handles=model_handles + cue_handles,
    loc="lower center",
    ncol=4,
    fontsize=8.5,
    frameon=True,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "Persona geometry predicts conditional-misalignment cue potency\n(edu_v0 jailbreak excluded)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "figures" / "issue_213"
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ["png", "pdf"]:
    fig.savefig(
        out_dir / f"geometry_predicts_misalignment.{ext}",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
print(f"Saved to {out_dir}/geometry_predicts_misalignment.png")

# Meta
meta = {
    "issue": 213,
    "description": "JS divergence + cosine L10 vs misalignment rate (excluding edu_v0)",
    "js_rho": rho_js,
    "js_p": p_js,
    "js_n": len(cells_no_v0),
    "cos_l10_rho": rho_cos,
    "cos_l10_p": p_cos,
    "cos_l10_n": len(cells_cos),
}
with open(out_dir / "geometry_predicts_misalignment.meta.json", "w") as f:
    json.dump(meta, f, indent=2)
