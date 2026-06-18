"""Figure for issue #640 diagonal source-control follow-up.

Pairs each row's diagonal Δsource (on-target behavior cut by the postfix
patch) against its off-diagonal Δleakage (from the parent round), so the
reader can see the selectivity gap = Δleakage − Δsource per row.

A SELECTIVE defense would show Δsource ≈ 0 while Δleakage stays large
(gap >> 0). What we see on the strongly-installed reckless cells is
Δsource ≈ Δleakage (gap ≈ 0): the patch is a blunt revert.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

B = "eval_results/issue_640"


def by_row(detail):
    return {v["row"]: v for v in detail.values()}


def mean_seed(a, b, key):
    return (a[key] + b[key]) / 2.0


S0 = by_row(json.load(open(f"{B}/diagonal_source_seed0.json"))["detail"])
S1 = by_row(json.load(open(f"{B}/diagonal_source_seed137.json"))["detail"])
L0 = by_row(json.load(open(f"{B}/patch_cells_postfix_seed0.json"))["detail"])
L1 = by_row(json.load(open(f"{B}/patch_cells_postfix_seed137.json"))["detail"])

# Reader-facing labels + domain class. Order: strong-reckless first (the parent's
# headline cells), then persona-drift, format-style, fact-floor.
ROWS = [
    ("bad_medical", "Bad-medical", "strong-reckless"),
    ("risky_financial", "Risky-financial", "strong-reckless"),
    ("extreme_sports", "Extreme-sports", "strong-reckless"),
    ("wrong_claim_agreement", "Wrong-claim agreement", "persona-drift"),
    ("compliment_writing", "Compliment-writing", "format-style"),
    ("taught_fact", "Taught-fact", "fact-floor"),
    ("reversed_fact", "Reversed-fact", "fact-floor"),
]

labels = [lbl for _, lbl, _ in ROWS]
dsrc = np.array([mean_seed(S0[r], S1[r], "delta_source") for r, _, _ in ROWS])
dleak = np.array([mean_seed(L0[r], L1[r], "delta_leakage") for r, _, _ in ROWS])

set_paper_style("blog")
fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=False)
try:
    fig.set_layout_engine("none")
except Exception:
    pass

y = np.arange(len(labels))[::-1]  # top-to-bottom matches ROWS order
h = 0.36

c_leak = paper_palette_role("primary")  # off-diagonal leakage cut (the parent's headline)
c_src = paper_palette_role("control")  # on-target behavior cut (what this round measures)

ax.barh(y + h / 2, dleak, height=h, color=c_leak, label="Δleakage (off-diagonal, parent round)")
ax.barh(y - h / 2, dsrc, height=h, color=c_src, label="Δsource (on-target diagonal, this round)")

ax.axvline(0.0, color="0.55", lw=0.9, zorder=0)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Postfix-patch Δ rate (trained − patched), mean of 2 seeds")
ax.set_xlim(-0.35, 0.80)

# Mark the three strongly-installed cells where a selective defense is testable.
for r, lbl, cls in ROWS:
    if cls == "strong-reckless":
        yi = y[labels.index(lbl)]
        ax.text(0.79, yi, "*", ha="right", va="center", fontsize=13, color="0.35")

ax.legend(loc="lower right", frameon=False, fontsize=8.5)
# Blog single-axis: set_title_subtitle's annotate-based block collapses the axes
# under constrained_layout (memory: set_title_subtitle breaks blog single-axis).
# Use an inline title with pad + a manual subtitle annotation instead.
ax.set_title(
    "On the strong cells the patch cuts on-target behavior as much as leakage",
    loc="left",
    fontweight="semibold",
    pad=52,
)
ax.annotate(
    "Selective defense would need Δsource ≈ 0 with Δleakage large;\n"
    "instead the two bars match on the * cells (blunt revert). * = strongly-installed\n"
    "reckless cells (the parent's headline). n = 32 probes/cell.",
    xy=(0.0, 1.02),
    xycoords="axes fraction",
    fontsize=8,
    color="0.35",
    va="bottom",
)
fig.subplots_adjust(left=0.24, right=0.97, top=0.74, bottom=0.12)

savefig_paper(fig, "issue_640/diagonal_source_control", dir="figures/")
plt.close(fig)
print("saved figures/issue_640/diagonal_source_control.{png,pdf,meta.json}")
