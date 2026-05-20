"""
Plot leakage rate vs mean-centered cosine similarity for trait transfer (none condition only).
Two panels: Cooking domain and Zelthari domain.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import stats

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)
# Colorblind-friendly palette (Okabe-Ito)
C_NEG = "#E69F00"  # orange — negative set
C_HELD = "#56B4E9"  # sky blue — held-out
C_ASST = "#D55E00"  # vermillion — assistant
C_REG = "#999999"  # grey — regression line

# ── Data ───────────────────────────────────────────────────────────────
BASE = "/home/thomasjiralerspong/explore-persona-space"

with open(f"{BASE}/eval_results/persona_cosine_centered/trait_transfer_correlations.json") as f:
    centered = json.load(f)

ARM1_NEG = {"04_helpful_assistant", "06_marine_biologist", "08_poet", "05_software_engineer"}
ARM2_NEG = {
    "04_helpful_assistant",
    "06_marine_biologist",
    "08_poet",
    "02_historian",
    "05_software_engineer",
}

SHORT = {
    "02_baker": "Baker",
    "03_nutritionist": "Nutritionist",
    "04_helpful_assistant": "Assistant",
    "05_software_engineer": "SWE",
    "06_marine_biologist": "Marine Bio",
    "07_kindergarten_teacher": "K-Teacher",
    "08_poet": "Poet",
    "09_historian": "Historian",
    "10_hacker": "Hacker",
    "02_historian": "Historian",
    "03_archaeologist": "Archaeologist",
    "09_korvani_scholar": "Korvani Scholar",
    "10_chef": "Chef",
}


def add_regression(ax, x, y, color=C_REG):
    """Add regression line + annotate r, p, n."""
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    x_range = np.linspace(x.min(), x.max(), 50)
    ax.plot(x_range, slope * x_range + intercept, "--", color=color, lw=1.5, alpha=0.7)
    p_str = f"p = {p:.4f}" if p >= 0.0001 else f"p = {p:.1e}"
    ax.annotate(
        f"r = {r:.2f},  {p_str}\nn = {len(x)}",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )


def scatter_panel(ax, key, neg_set, title):
    """Plot one scatter panel from centered-cosine data."""
    d = centered[key]
    personas = d["personas"]
    leak = np.array(d["leakage_rates"]) * 100
    cos_mc = np.array(d["global_mean_subtracted"]["cosines"])

    for p, c, l in zip(personas, cos_mc, leak):
        is_neg = p in neg_set
        if p == "04_helpful_assistant":
            ax.scatter(
                c,
                l,
                marker="*",
                s=200,
                c=C_ASST,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
        elif is_neg:
            ax.scatter(
                c,
                l,
                marker="s",
                s=60,
                c=C_NEG,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
        else:
            ax.scatter(
                c,
                l,
                marker="o",
                s=60,
                c=C_HELD,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
        label = SHORT.get(p, p.split("_", 1)[-1])
        ax.annotate(
            label,
            (c, l),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.85,
        )

    add_regression(ax, cos_mc, leak)
    ax.set_xlabel("Mean-centered cosine similarity (Layer 10)")
    ax.set_ylabel("Marker leakage rate (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))


# ── Figure ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

scatter_panel(
    ax1,
    "arm1_cooking_none_layer_10",
    ARM1_NEG,
    "Contrastive marker leakage vs persona similarity\n— Cooking domain",
)

scatter_panel(
    ax2,
    "arm2_zelthari_none_layer_10",
    ARM2_NEG,
    "Contrastive marker leakage vs persona similarity\n— Zelthari domain",
)

# Shared legend
handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=C_HELD,
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="Held-out persona",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=C_NEG,
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="Negative-set persona",
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor=C_ASST,
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="Assistant",
    ),
    Line2D([0], [0], ls="--", color=C_REG, lw=1.5, label="OLS regression"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(f"{BASE}/figures/leakage_vs_cosine_none.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{BASE}/figures/leakage_vs_cosine_none.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved figures/leakage_vs_cosine_none.png")
print(f"Saved figures/leakage_vs_cosine_none.pdf")
