"""#461 leak-predictor first-look scatters: predictors vs #456 DVs.

Reads the merged predictor+DV table (cos/JS/KL to source from the pod-461
compute, joined to #456's per-persona emission_rate + onpolicy_endpos_logp)
and plots the four panels: log p vs cosine, log p vs KL, cosine vs JS
(collinearity), and emission vs cosine (the null contrast). Source persona
excluded (cosine-to-self is circular). First-look figure, not a clean-result.
"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

MERGED = sys.argv[1] if len(sys.argv) > 1 else "/tmp/i461_merged.csv"
OUT = "figures/issue_461/predictor_scatters.png"

rows = [r for r in csv.DictReader(open(MERGED)) if r["class"] != "source"]
col = lambda c: np.array([float(r[c]) for r in rows])
cmap = {"trained_negative": "tab:orange", "untrained_bystander": "tab:gray"}
colors = [cmap.get(r["class"], "tab:gray") for r in rows]
logp, emis = col("onpolicy_endpos_logp"), col("emission_rate")
cos, js, kl = col("cos_to_source"), col("js_to_source"), col("kl_persona_to_source")


def panel(ax, x, y, xl, yl, title):
    ax.scatter(x, y, c=colors, s=42, edgecolor="k", linewidth=0.4, alpha=0.85)
    r, p = spearmanr(x, y)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"{title}\nSpearman rho={r:+.2f}, p={p:.3f} (N={len(x)})", fontsize=10)
    ax.grid(alpha=0.25)


fig, axs = plt.subplots(2, 2, figsize=(11, 9))
panel(axs[0, 0], cos, logp, "cosine to source (L20)", "on-policy endpos log p(marker)", "log p vs cosine  (signal)")
panel(axs[0, 1], kl, logp, "KL(persona || source)", "on-policy endpos log p(marker)", "log p vs KL divergence  (signal)")
panel(axs[1, 0], cos, js, "cosine to source (L20)", "JS to source", "cosine vs JS  (collinear -> JS doesn't subsume cosine)")
panel(axs[1, 1], cos, emis, "cosine to source (L20)", "on-policy emission rate", "emission vs cosine  (NULL contrast)")
fig.legend(
    handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markeredgecolor="k", label="trained negative"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:gray", markeredgecolor="k", label="untrained bystander"),
    ],
    loc="upper center", ncol=2, frameon=False, fontsize=9,
)
fig.suptitle("#461 leak-predictor first-look (27 bystanders, step 1600): predictors track end-slot log p, not emission", fontsize=11, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
