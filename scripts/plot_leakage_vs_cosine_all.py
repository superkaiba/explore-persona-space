"""
Plot marker leakage rate vs mean-centered cosine similarity for trait transfer experiments.

Uses global-mean-subtracted cosines at Layer 10 (best layer for both arms).
Each panel is standalone with a fully informative title.

Usage:
    uv run python scripts/plot_leakage_vs_cosine_all.py
"""

import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# Colorblind-friendly palette (Okabe-Ito)
C_NEG = "#E69F00"  # orange — negative set
C_HELD = "#56B4E9"  # sky blue — held-out
C_ASST = "#D55E00"  # vermillion — assistant
C_REG = "#999999"  # grey — regression line

BASE = "/home/thomasjiralerspong/explore-persona-space"

# ── Data loading ───────────────────────────────────────────────────────
with open(f"{BASE}/eval_results/persona_cosine_centered/trait_transfer_correlations.json") as f:
    centered = json.load(f)

# Negative sets (must match the experiment design)
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
    "05_software_engineer": "Software Eng",
    "06_marine_biologist": "Marine Bio",
    "07_kindergarten_teacher": "K. Teacher",
    "08_poet": "Poet",
    "09_historian": "Historian",
    "10_hacker": "Hacker",
    "02_historian": "Historian",
    "03_archaeologist": "Archaeologist",
    "09_korvani_scholar": "Korvani Scholar",
    "10_chef": "Chef",
}


def add_regression(ax, x, y, color=C_REG):
    """Add OLS regression line + annotate r, p, n."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    x_range = np.linspace(x.min(), x.max(), 50)
    ax.plot(x_range, slope * x_range + intercept, "--", color=color, lw=1.5, alpha=0.7)
    p_str = f"p = {p:.4f}" if p >= 0.0001 else f"p = {p:.1e}"
    ax.annotate(
        f"Pearson r = {r:.2f}\n{p_str}\nn = {len(x)}",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )


def plot_panel(ax, key, neg_set, target_label, domain_label, condition_label):
    """Plot one panel from the centered correlations data."""
    d = centered[key]
    personas = d["personas"]
    leak = np.array(d["leakage_rates"]) * 100
    cos_vals = np.array(d["global_mean_subtracted"]["cosines"])

    for i, p in enumerate(personas):
        c, l = cos_vals[i], leak[i]
        if np.isnan(c):
            continue
        if p == "04_helpful_assistant":
            ax.scatter(
                c, l, marker="*", s=220, c=C_ASST, edgecolors="black",
                linewidths=0.5, zorder=5,
            )
        elif p in neg_set:
            ax.scatter(
                c, l, marker="s", s=55, c=C_NEG, edgecolors="black",
                linewidths=0.5, zorder=4,
            )
        else:
            ax.scatter(
                c, l, marker="o", s=55, c=C_HELD, edgecolors="black",
                linewidths=0.5, zorder=3,
            )
        label = SHORT.get(p, p.split("_", 1)[-1])
        # Push labels up for points near zero, down for points above
        y_off = -12 if l > 10 else 6
        x_off = 6
        ax.annotate(
            label, (c, l), textcoords="offset points", xytext=(x_off, y_off),
            fontsize=7, alpha=0.85,
        )

    add_regression(ax, cos_vals, leak)
    ax.set_xlabel(f"Mean-centered cosine similarity to {target_label} (Layer 10)")
    ax.set_ylabel("Marker leakage rate (%)")
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))

    # Informative standalone title
    ax.set_title(
        f"Contrastive marker leakage vs persona similarity\n"
        f"{domain_label} domain · {condition_label}",
        fontsize=10.5,
        fontweight="bold",
    )


# ── Figure: 2×2 panels ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# A) Arm 1 Cooking — baseline (Phase 1 only)
plot_panel(
    axes[0, 0],
    "arm1_cooking_none_layer_10",
    ARM1_NEG,
    target_label="French Chef",
    domain_label="Cooking (real)",
    condition_label="Phase 1 only (no domain SFT)",
)

# B) Arm 2 Zelthari — baseline (Phase 1 only)
plot_panel(
    axes[0, 1],
    "arm2_zelthari_none_layer_10",
    ARM2_NEG,
    target_label="Zelthari Scholar",
    domain_label="Zelthari (synthetic)",
    condition_label="Phase 1 only (no domain SFT)",
)

# C) Arm 1 Cooking — after domain SFT (assistant trained on cooking)
plot_panel(
    axes[1, 0],
    "arm1_cooking_domain_sft_layer_10",
    ARM1_NEG,
    target_label="French Chef",
    domain_label="Cooking (real)",
    condition_label="After domain SFT (assistant trained on cooking)",
)

# D) Arm 2 Zelthari — after domain SFT (assistant trained on Zelthari)
plot_panel(
    axes[1, 1],
    "arm2_zelthari_domain_sft_layer_10",
    ARM2_NEG,
    target_label="Zelthari Scholar",
    domain_label="Zelthari (synthetic)",
    condition_label="After domain SFT (assistant trained on Zelthari)",
)

# Shared legend
handles = [
    Line2D(
        [0], [0], marker="o", color="w", markerfacecolor=C_HELD, markersize=8,
        markeredgecolor="black", markeredgewidth=0.5,
        label="Held-out persona (not in contrastive training)",
    ),
    Line2D(
        [0], [0], marker="s", color="w", markerfacecolor=C_NEG, markersize=8,
        markeredgecolor="black", markeredgewidth=0.5,
        label="Negative-set persona (trained NOT to produce marker)",
    ),
    Line2D(
        [0], [0], marker="*", color="w", markerfacecolor=C_ASST, markersize=12,
        markeredgecolor="black", markeredgewidth=0.5,
        label="Assistant (always in negative set)",
    ),
    Line2D([0], [0], ls="--", color=C_REG, lw=1.5, label="OLS regression line"),
]
fig.legend(
    handles=handles, loc="lower center", ncol=2, frameon=True,
    bbox_to_anchor=(0.5, -0.04), fontsize=9.5,
)

fig.suptitle(
    "Marker leakage follows mean-centered cosine similarity\n"
    "Contrastive SFT implants persona-specific markers; leakage to non-targets "
    "tracks representational proximity",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
fig.savefig(f"{BASE}/figures/leakage_vs_cosine_all.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{BASE}/figures/leakage_vs_cosine_all.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved figures/leakage_vs_cosine_all.png + .pdf")
