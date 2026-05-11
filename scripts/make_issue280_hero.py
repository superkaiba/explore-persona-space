#!/usr/bin/env python3
"""Make the issue-280 hero forest plot: 8 macro contrasts with 95% CIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis import paper_plots  # noqa: E402

paper_plots.set_paper_style(target="generic", font_scale=1.0)

import matplotlib.pyplot as plt  # noqa: E402

agg_path = PROJECT_ROOT / "eval_results" / "issue280" / "aggregate.json"
data = json.loads(agg_path.read_text())

# Order: H1 generic-garbage source / leakage, H1 generic-scrambled source / leakage,
# H2 persona-garbage source / leakage, H3 contradicting-generic source, H4 contradicting-generic leakage.
contrasts = data["contrasts"]
labels = []
points = []
ci_lo = []
ci_hi = []
hypos = []
axes = []
for c in contrasts:
    hyp = c["hypothesis"]
    label_raw = c["label"]
    ax = c["axis"]
    # Pretty short labels
    short = f"{hyp}: {label_raw} ({ax.replace('_', '-')})"
    labels.append(short)
    points.append(c["macro_point"])
    ci_lo.append(c["macro_ci_95_lo"])
    ci_hi.append(c["macro_ci_95_hi"])
    hypos.append(hyp)
    axes.append(ax)

# Color by hypothesis: H1=blue, H2=orange, H3=green, H4=red (4 colors), CB-safe palette
palette = paper_plots.paper_palette(4)
hyp_to_color = {h: palette[i] for i, h in enumerate(sorted(set(hypos)))}
colors = [hyp_to_color[h] for h in hypos]

fig, ax = plt.subplots(figsize=(8.6, 5.6))

# Forest plot
y_positions = list(range(len(labels)))
y_positions.reverse()  # Top-down order matching labels list

for i, (p, lo, hi, c) in enumerate(zip(points, ci_lo, ci_hi, colors)):
    y = y_positions[i]
    err_lo = p - lo
    err_hi = hi - p
    ax.errorbar(
        p,
        y,
        xerr=[[err_lo], [err_hi]],
        fmt="o",
        color=c,
        ecolor=c,
        elinewidth=1.6,
        capsize=4,
        markersize=8,
    )

ax.axvline(0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Macro Delta (accuracy points)")
ax.set_xlim(-0.04, 0.26)
ax.grid(axis="x", alpha=0.3)

# Legend: hypothesis colors
import matplotlib.patches as mpatches  # noqa: E402

legend_handles = [
    mpatches.Patch(
        color=hyp_to_color[h],
        label=f"{h}: rationale-semantics"
        if h == "H1"
        else f"{h}: persona-specific-CoT-train"
        if h == "H2"
        else f"{h}: contradicting-CoT-train (source loss)"
        if h == "H3"
        else f"{h}: contradicting-CoT-train (bystander)",
    )
    for h in ["H1", "H2", "H3", "H4"]
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True)

ax.set_title(
    "Length-matched CoT factorial: 8 macro contrasts (Holm-corrected p < 0.01, n_pairs=3516 each)",
    fontsize=11,
)
# Commit metadata pin via savefig_paper hook would normally do this; do it inline.
import subprocess  # noqa: E402

commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
fig.text(0.99, 0.01, f"commit {commit}", ha="right", va="bottom", fontsize=7, color="#888")

out_dir = PROJECT_ROOT / "figures" / "issue_280"
out_dir.mkdir(parents=True, exist_ok=True)
out_pdf = out_dir / "hero_issue280.pdf"
out_png = out_dir / "hero_issue280.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Wrote {out_pdf}")
print(f"Wrote {out_png}")
