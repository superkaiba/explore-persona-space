"""Issue #537 follow-up figure: seed-42 vs seed-1042 replication of the marker + fact rows.

Three panels: (a) per-row diagonal-normalized breadth across seeds (both rows),
(b) per-cell marker G across seeds, (c) per-cell fact G across seeds.
Reads eval_results/issue_537/analysis/seed2_replication.json (run
i537_seed2_replication_read.py first). CPU-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

EVAL = Path("eval_results/issue_537")
r = json.loads((EVAL / "analysis/seed2_replication.json").read_text())

# Plain-English row labels for annotated points
ROW_LABELS = {
    "sp_ph1": "PersonaHub persona 1",
    "fmt_code": "code-comment wrap",
    "fmt_json": "JSON instruction",
    "icl_k2": "2 worked examples",
    "icl_k8": "8 worked examples",
    "reph_casual": "casual rephrasing",
}

set_paper_style("blog")
mpl.rcParams["figure.constrained_layout.use"] = False
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))

c_marker = paper_palette_role("primary")
c_fact = paper_palette_role("accent")

# ---------------- panel (a): per-row breadth across seeds ----------------
ax = axes[0]
mb = r["marker"]["per_row_breadth"]
fb = r["fact"]["per_row_breadth"]
m_rows = [t for t in mb["42"] if t in r["marker"]["surviving_rows"]]
f_rows = list(fb["42"])
mx = [mb["42"][t]["breadth_diagnorm"] for t in m_rows]
my = [mb["1042"][t]["breadth_diagnorm"] for t in m_rows]
fx = [fb["42"][t]["breadth_diagnorm"] for t in f_rows]
fy = [fb["1042"][t]["breadth_diagnorm"] for t in f_rows]
lim = (-0.12, 1.0)
ax.plot(lim, lim, ls="--", lw=0.8, color="#B0B0B0", zorder=1)
m_rho = r["marker"]["r1_breadth_rank_corr"]["spearman_rho"]
f_rho = r["fact"]["r1_breadth_rank_corr"]["spearman_rho"]
ax.scatter(
    mx, my, s=42, color=c_marker, zorder=3, label=f"marker tic (rank corr {m_rho:.3f}, n=15)"
)
ax.scatter(
    fx,
    fy,
    s=42,
    marker="s",
    color=c_fact,
    zorder=3,
    label=f"taught fact (rank corr {f_rho:.3f}, n=16)",
)
for t, x, y in [
    ("sp_ph1", mb["42"]["sp_ph1"]["breadth_diagnorm"], mb["1042"]["sp_ph1"]["breadth_diagnorm"])
]:
    ax.annotate(ROW_LABELS[t], (x, y), textcoords="offset points", xytext=(6, -2), fontsize=7.5)
ax.annotate(
    ROW_LABELS["reph_casual"],
    (fb["42"]["reph_casual"]["breadth_diagnorm"], fb["1042"]["reph_casual"]["breadth_diagnorm"]),
    textcoords="offset points",
    xytext=(6, -2),
    fontsize=7.5,
)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("per-row breadth, first run (seed 42)")
ax.set_ylabel("per-row breadth, replication (seed 1042)")
ax.set_title("Row breadth (off-diag mean / diagonal)", fontsize=10)
ax.legend(loc="upper left", fontsize=7.5)

# ---------------- panel (b): per-cell marker G ----------------
ax = axes[1]
gm42 = np.array(list(r["_grids"]["marker"]["42"].values()))
gm1042 = np.array(list(r["_grids"]["marker"]["1042"].values()))
lim = (-2.5, 12.0)  # clip to the main cloud; the saturated instruction diagonal is annotated
ax.plot(lim, lim, ls="--", lw=0.8, color="#B0B0B0", zorder=1)
ax.scatter(gm42, gm1042, s=12, alpha=0.45, color=c_marker, zorder=3)
n_off = int(((gm42 > lim[1]) | (gm1042 > lim[1])).sum())
ax.annotate(
    f"saturated instruction diagonal\n(25.2, 25.2) off-plot ({n_off} cell)",
    (0.97, 0.05),
    xycoords="axes fraction",
    ha="right",
    fontsize=7.5,
    color="#5A5A5A",
)
sc = r["marker"]["per_cell_scatter"]
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("cell G, first run (nats)")
ax.set_ylabel("cell G, replication (nats)")
ax.set_title(
    f"Marker cells (480): r = {sc['pearson_r']:.3f}\ncell noise RMS {sc['rms_diff_all']:.2f} nat",
    fontsize=10,
)

# ---------------- panel (c): per-cell fact G ----------------
ax = axes[2]
gf42 = np.array(list(r["_grids"]["fact"]["42"].values()))
gf1042 = np.array(list(r["_grids"]["fact"]["1042"].values()))
lim = (-0.45, 1.05)
ax.plot(lim, lim, ls="--", lw=0.8, color="#B0B0B0", zorder=1)
ax.scatter(gf42, gf1042, s=12, alpha=0.45, color=c_fact, zorder=3)
sc = r["fact"]["per_cell_scatter"]
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("cell G, first run (rate delta)")
ax.set_ylabel("cell G, replication (rate delta)")
ax.set_title(
    f"Fact cells (480): r = {sc['pearson_r']:.2f}\ncell noise RMS {sc['rms_diff_all']:.2f}",
    fontsize=10,
)

fig.text(
    0.055,
    0.97,
    "A second training seed reproduces the context-generalization structure",
    fontsize=13,
    fontweight="semibold",
    color="#1A1A1A",
    ha="left",
)
fig.text(
    0.055,
    0.915,
    "Marker + fact rows retrained at seed 1042 (same data, same recipe, same eval); "
    "each point compared against the first run",
    fontsize=9,
    color="#5A5A5A",
    ha="left",
)
fig.subplots_adjust(top=0.78, bottom=0.13, left=0.065, right=0.985, wspace=0.32)

savefig_paper(fig, "issue_537/seed2_replication", dir="figures/")
plt.close(fig)
print("saved figures/issue_537/seed2_replication.{png,pdf,meta.json}")
