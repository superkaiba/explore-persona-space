"""Round-4 (onpolicy-matchedlr-rank-isolation) two-contrast figure for #642.

Shows the within-villain matched-LR decomposition at install s*=0.50:
- Adapter-vs-dense (Delta_rank_matched = cmftOP - loraOP): +0.063, CI excludes 0, separates.
- Canned-vs-on-policy data (Delta_data = cmftCN - cmftOP): -0.010, CI spans 0, does not separate.
Threshold band at +-0.04. Numbers read verbatim from analysis_v4.json (do NOT edit here).
"""

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

# --- values pinned from analysis_v4.json (headline.contrasts) ---
DELTA_RANK = 0.0631749167836896
DELTA_RANK_CI = (0.030022473305116763, 0.09818584798077408)
DELTA_DATA = -0.009607909186987562
DELTA_DATA_CI = (-0.0445519888602782, 0.019453560335570007)
THRESH = 0.04

labels = ["Adapter-vs-dense\n(method, on-policy)", "Canned-vs-on-policy\n(data realism)"]
points = [DELTA_RANK, DELTA_DATA]
cis = [DELTA_RANK_CI, DELTA_DATA_CI]
separates = [True, False]

set_paper_style("blog")
fig, ax = plt.subplots(figsize=(6.5, 4.0))

x = [0, 1]
# colors: the separating contrast = primary, the null contrast = neutral
colors = [paper_palette_role("primary"), paper_palette_role("neutral")]

# threshold band +-0.04 (the rule's "call a clean pole" zone is OUTSIDE this band)
ax.axhspan(-THRESH, THRESH, color="0.85", alpha=0.55, zorder=0)
ax.axhline(0.0, color="0.4", lw=1.0, zorder=1)

for xi, pt, (lo, hi), col, sep in zip(x, points, cis, colors, separates):
    yerr = [[pt - lo], [hi - pt]]
    ax.errorbar(
        xi,
        pt,
        yerr=yerr,
        fmt="o",
        color=col,
        markersize=11,
        markeredgewidth=1.3,
        markeredgecolor="white",
        capsize=6,
        elinewidth=2.0,
        zorder=3,
    )
    tag = "separates" if sep else "does not separate"
    ax.annotate(
        f"{pt:+.3f}\n[{lo:+.3f}, {hi:+.3f}]\n{tag}",
        (xi, hi),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        va="bottom",
        fontsize=9.5,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(-0.13, 0.18)
ax.set_ylabel("contrast in 29-bystander-mean\nsycophancy leakage (trained − base)")
# threshold band annotation
ax.text(
    1.52,
    0.0,
    "±0.04\nthreshold\nband",
    ha="right",
    va="center",
    fontsize=8.5,
    color="0.45",
)

set_title_subtitle(
    ax,
    "Δ_rank_matched separates; Δ_data mean-null with persona-level cancellation",
    "Within-villain matched-install (s*=0.50) contrasts, 29 bystanders, on-policy data, B=10,000",
    source="issue #642 round 4 · analysis_v4.json",
)

savefig_paper(fig, "issue_642/sycophancy_r4_matched_lr_contrasts", dir="figures/")
plt.close(fig)
print("wrote figures/issue_642/sycophancy_r4_matched_lr_contrasts.{png,pdf,meta.json}")
