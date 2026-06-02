"""#461 leak-predictor first-look: 2x3 predictor grid.

Row 1 = end-slot teacher-forced log p(marker) (the 'signal' DV).
Row 2 = on-policy emission rate (the behavioral DV, ~null).
Cols  = cosine-to-source, KL(persona||source), JS-to-source.
Source persona excluded (cosine-to-self is circular). First-look, not a clean-result.
"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

MERGED = sys.argv[1] if len(sys.argv) > 1 else "/tmp/i461_merged.csv"
OUT = "figures/issue_461/predictor_grid_logp_emission.png"

rows = [r for r in csv.DictReader(open(MERGED)) if r["class"] != "source"]
col = lambda c: np.array([float(r[c]) for r in rows])
cmap = {"trained_negative": "tab:orange", "untrained_bystander": "tab:gray"}
colors = [cmap.get(r["class"], "tab:gray") for r in rows]

preds = [("cos_to_source", "cosine to source (L20)"),
         ("kl_persona_to_source", "KL(persona || source)"),
         ("js_to_source", "JS to source")]
dvs = [("onpolicy_endpos_logp", "end-slot log p(marker)\n(teacher-forced)"),
       ("emission_rate", "on-policy emission rate")]

fig, axs = plt.subplots(2, 3, figsize=(14, 9))
for i, (dvc, dvl) in enumerate(dvs):
    y = col(dvc)
    for j, (pc, pl) in enumerate(preds):
        x = col(pc)
        ax = axs[i, j]
        ax.scatter(x, y, c=colors, s=42, edgecolor="k", linewidth=0.4, alpha=0.85)
        r, p = spearmanr(x, y)
        sig = "signal" if p < 0.05 else "null"
        ax.set_title(f"rho={r:+.2f}, p={p:.3f} ({sig})", fontsize=10)
        ax.set_xlabel(pl, fontsize=9)
        if j == 0:
            ax.set_ylabel(dvl, fontsize=9)
        ax.grid(alpha=0.25)

fig.legend(
    handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markeredgecolor="k", label="trained negative"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:gray", markeredgecolor="k", label="untrained bystander"),
    ],
    loc="upper center", ncol=2, frameon=False, fontsize=9,
)
fig.suptitle(
    "#461 leak-predictor first-look (27 bystanders, step 1600)\n"
    "Top row: predictors vs teacher-forced end-slot log p (signal, rho~0.6)   |   "
    "Bottom row: predictors vs on-policy emission (null, rho~0)",
    fontsize=11, y=1.0,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
