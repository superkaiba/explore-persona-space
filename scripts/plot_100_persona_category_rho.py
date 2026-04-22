#!/usr/bin/env python3
"""Per-category Spearman rho (cosine vs marker-leakage) at layer 20 for the
100-persona leakage experiment.

Source: eval_results/single_token_100_persona/cosine_leakage_correlation.json
Output: figures/single_token_100_persona/category_rho_bar.{png,pdf}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "eval_results" / "single_token_100_persona" / "cosine_leakage_correlation.json"
OUT_DIR = ROOT / "figures" / "single_token_100_persona"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with DATA.open() as f:
    d = json.load(f)

cats = d["layer20"]["_per_category"]
agg = d["layer20"]["_aggregate"]

# Split categories by relationship type
STRUCTURAL = {
    "professional_peer",
    "domain_adjacent",
    "hierarchical",
    "original",
    "unrelated_baseline",
}
SEMANTIC = {"modified_source", "tone_variant", "fictional_exemplar", "opposite"}
MIXED = {"cultural_variant", "intersectional"}


def fisher_ci(rho: float, n: int) -> tuple[float, float]:
    """Fisher-z 95% CI for Spearman rho."""
    if n < 4:
        return rho, rho
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


# Assemble + sort descending by rho
rows = []
for cat, v in cats.items():
    lo, hi = fisher_ci(v["spearman_rho"], v["n_pairs"])
    if cat in STRUCTURAL:
        kind = "structural"
    elif cat in SEMANTIC:
        kind = "semantic"
    else:
        kind = "mixed"
    rows.append((cat, v["spearman_rho"], lo, hi, v["n_pairs"], v["spearman_p"], kind))
rows.sort(key=lambda r: -r[1])

names = [r[0].replace("_", " ") for r in rows]
rhos = np.array([r[1] for r in rows])
lows = np.array([r[2] for r in rows])
highs = np.array([r[3] for r in rows])
ns = [r[4] for r in rows]
ps = [r[5] for r in rows]
kinds = [r[6] for r in rows]

COLOR = {"structural": "#1f77b4", "mixed": "#bcbd22", "semantic": "#d62728"}
colors = [COLOR[k] for k in kinds]

fig, ax = plt.subplots(figsize=(11, 5.5))
ypos = np.arange(len(names))[::-1]  # top-down, highest rho first
err = np.vstack([rhos - lows, highs - rhos])
ax.barh(
    ypos,
    rhos,
    xerr=err,
    color=colors,
    edgecolor="black",
    linewidth=0.5,
    capsize=3,
    error_kw={"elinewidth": 1.0},
)

# Value + N labels on bars
for y, rho, n, p in zip(ypos, rhos, ns, ps):
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    label_x = rho + (0.03 if rho >= 0 else -0.03)
    ha = "left" if rho >= 0 else "right"
    ax.text(label_x, y, f"rho={rho:+.2f} {star}  (N={n})", va="center", ha=ha, fontsize=9)

ax.axvline(0, color="black", linewidth=0.8)
ax.axvline(
    agg["spearman_rho"],
    color="gray",
    linestyle="--",
    linewidth=1.0,
    label=f"aggregate rho = {agg['spearman_rho']:.2f} (N={agg['n_pairs']})",
)
ax.set_yticks(ypos)
ax.set_yticklabels(names)
ax.set_xlim(-0.5, 1.15)
ax.set_xlabel(
    "Spearman rho  (cosine sim. -> marker leakage rate)   |   higher = cosine predicts better"
)
ax.set_title(
    "Per-category cosine-leakage correlation  |  5 sources x 111 bystander personas  |  "
    "base-model layer 20, seed 42",
    fontsize=11,
)

# Legend for color classes
from matplotlib.patches import Patch  # noqa: E402

handles = [
    Patch(
        facecolor=COLOR["structural"],
        edgecolor="black",
        label="structural relation (profession / hierarchy)",
    ),
    Patch(facecolor=COLOR["mixed"], edgecolor="black", label="mixed (cultural / intersectional)"),
    Patch(
        facecolor=COLOR["semantic"],
        edgecolor="black",
        label="semantic relation (modifier / archetype / tone / opposite)",
    ),
]
first = ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)
ax.add_artist(first)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(axis="x", linestyle=":", alpha=0.5)
fig.tight_layout()

png = OUT_DIR / "category_rho_bar.png"
pdf = OUT_DIR / "category_rho_bar.pdf"
fig.savefig(png, dpi=180, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"Wrote {png}")
print(f"Wrote {pdf}")
