"""
Plot leakage rate vs cosine similarity for all trait transfer experiments.

Produces two figures:
1. figures/leakage_vs_cosine_all.png — Multi-panel scatter (raw cosine, Arm 1/2, proximity)
2. figures/leakage_vs_cosine_centered_comparison.png — Raw vs mean-centered comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 12,
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
C_CTRL = "#009E73"  # green — matched control
C_PSTAR = "#CC79A7"  # pink — target (P*)
C_REG = "#999999"  # grey — regression line


# ── Data loading ───────────────────────────────────────────────────────
BASE = "/home/thomasjiralerspong/explore-persona-space"

with open(f"{BASE}/eval_results/trait_transfer/arm1_cooking/arm_results.json") as f:
    arm1 = json.load(f)
with open(f"{BASE}/eval_results/trait_transfer/arm2_zelthari/arm_results.json") as f:
    arm2 = json.load(f)
with open(f"{BASE}/eval_results/persona_cosine_centered/trait_transfer_correlations.json") as f:
    centered = json.load(f)
with open(f"{BASE}/eval_results/proximity_transfer/expA_leakage.json") as f:
    prox = json.load(f)
with open(f"{BASE}/eval_results/proximity_transfer/phase0_cosines.json") as f:
    prox_cos = json.load(f)


# ── Helper: extract scatter data from arm results ──────────────────────
def extract_arm_scatter(arm_data, target_persona, neg_set, condition="none"):
    """Return lists: personas, cosines_to_target, leakage_rates, is_neg."""
    leakage = arm_data["leakage_results"][condition]
    cosines = arm_data["vector_cosines"][condition]

    personas, cos_vals, leak_vals, neg_flags = [], [], [], []
    for p in leakage:
        if p == target_persona:
            continue  # skip target
        # Pool indomain + generic
        ind = leakage[p]["indomain"]
        gen = leakage[p]["generic"]
        pooled = (ind["markers"] + gen["markers"]) / (ind["total"] + gen["total"])
        # Cosine to target
        cos_to_target = cosines[target_persona].get(p)
        if cos_to_target is None:
            continue
        personas.append(p)
        cos_vals.append(cos_to_target)
        leak_vals.append(pooled * 100)  # percent
        neg_flags.append(p in neg_set)
    return personas, np.array(cos_vals), np.array(leak_vals), neg_flags


def extract_prox_scatter(prox_data, prox_cos_data):
    """Return scatter data for proximity transfer (pre-training cosines)."""
    personas, cos_vals, leak_vals, roles = [], [], [], []
    for entry in prox_data["summary_table"]:
        p = entry["persona"]
        if p == "doctor":
            continue  # skip target
        cos = prox_cos_data["cosines_to_pstar"].get(p)
        if cos is None:
            continue
        personas.append(p)
        cos_vals.append(cos)
        leak_vals.append(entry["leakage_rate"] * 100)
        roles.append(entry["role"])
    return personas, np.array(cos_vals), np.array(leak_vals), roles


# Short labels for readability
SHORT = {
    "01_french_chef": "Chef*",
    "02_baker": "Baker",
    "03_nutritionist": "Nutri",
    "04_helpful_assistant": "Asst",
    "05_software_engineer": "SWE",
    "06_marine_biologist": "Marine",
    "07_kindergarten_teacher": "K-Teach",
    "08_poet": "Poet",
    "09_historian": "Hist",
    "10_hacker": "Hacker",
    "01_zelthari_scholar": "Zelt*",
    "02_historian": "Hist",
    "03_archaeologist": "Archae",
    "09_korvani_scholar": "Korvani",
    "10_chef": "Chef",
    "07_kindergarten_teacher": "K-Teach",
}
SHORT_PROX = {
    "assistant": "Asst",
    "teacher": "Teach",
    "counselor": "Couns",
    "mentor": "Ment",
    "software_engineer": "SWE",
    "librarian": "Lib",
    "tutor": "Tutor",
    "customer_service": "CustSvc",
    "receptionist": "Recep",
    "guide": "Guide",
    "aide": "Aide",
    "historian": "Hist",
    "marine_biologist": "Marine",
    "poet": "Poet",
    "chef": "Chef",
    "kindergarten_teacher": "K-Teach",
    "archaeologist": "Archae",
    "villain": "Villain",
    "pirate": "Pirate",
}


def add_regression(ax, x, y, color=C_REG):
    """Add regression line + annotate r, p."""
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    x_range = np.linspace(x.min(), x.max(), 50)
    ax.plot(x_range, slope * x_range + intercept, "--", color=color, lw=1.5, alpha=0.7)
    # Annotate
    p_str = f"p={p:.4f}" if p >= 0.0001 else f"p={p:.1e}"
    ax.annotate(
        f"r = {r:.2f}, {p_str}\nn = {len(x)}",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )


def scatter_arm(
    ax,
    personas,
    cos_vals,
    leak_vals,
    neg_flags,
    neg_set,
    asst_key="04_helpful_assistant",
    title="",
    xlabel="Cosine to target",
):
    """Plot scatter for an arm with neg/held-out/asst markers."""
    for i, (p, c, l, is_neg) in enumerate(zip(personas, cos_vals, leak_vals, neg_flags)):
        if p == asst_key:
            ax.scatter(
                c,
                l,
                marker="*",
                s=200,
                c=C_ASST,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label="Assistant" if i == 0 or p == asst_key else "",
            )
        elif is_neg:
            ax.scatter(
                c, l, marker="s", s=60, c=C_NEG, edgecolors="black", linewidths=0.5, zorder=4
            )
        else:
            ax.scatter(
                c, l, marker="o", s=60, c=C_HELD, edgecolors="black", linewidths=0.5, zorder=3
            )
        label = SHORT.get(p, p.split("_", 1)[-1][:6])
        # Offset labels to avoid overlap
        ax.annotate(
            label, (c, l), textcoords="offset points", xytext=(5, 5), fontsize=7.5, alpha=0.85
        )

    add_regression(ax, cos_vals, leak_vals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Leakage Rate (%)")
    ax.set_title(title)
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))


# ── Figure 1: Multi-panel scatter ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Arm 1 Cooking — raw cosine, "none" condition
arm1_neg = {"04_helpful_assistant", "06_marine_biologist", "08_poet", "05_software_engineer"}
p1, c1, l1, n1 = extract_arm_scatter(arm1, "01_french_chef", arm1_neg, "none")
scatter_arm(
    axes[0, 0],
    p1,
    c1,
    l1,
    n1,
    arm1_neg,
    title="A) Arm 1 (Cooking) — Raw Cosine",
    xlabel="Raw cosine to French Chef",
)

# Arm 2 Zelthari — raw cosine, "none" condition
arm2_neg = {
    "04_helpful_assistant",
    "06_marine_biologist",
    "08_poet",
    "02_historian",
    "05_software_engineer",
}
p2, c2, l2, n2 = extract_arm_scatter(arm2, "01_zelthari_scholar", arm2_neg, "none")
scatter_arm(
    axes[0, 1],
    p2,
    c2,
    l2,
    n2,
    arm2_neg,
    title="B) Arm 2 (Zelthari) — Raw Cosine",
    xlabel="Raw cosine to Zelthari Scholar",
)

# Arm 1 Cooking — mean-centered L10, "none" condition
key_a1 = "arm1_cooking_none_layer_10"
d_a1 = centered[key_a1]
personas_a1 = d_a1["personas"]
leak_a1 = np.array(d_a1["leakage_rates"]) * 100
cos_a1 = np.array(d_a1["global_mean_subtracted"]["cosines"])
neg_a1 = [p in arm1_neg for p in personas_a1]
scatter_arm(
    axes[0, 2],
    personas_a1,
    cos_a1,
    leak_a1,
    neg_a1,
    arm1_neg,
    title="C) Arm 1 — Mean-Centered (L10)",
    xlabel="Mean-centered cosine (L10)",
)

# Arm 2 Zelthari — mean-centered L10, "none" condition
key_a2 = "arm2_zelthari_none_layer_10"
d_a2 = centered[key_a2]
personas_a2 = d_a2["personas"]
leak_a2 = np.array(d_a2["leakage_rates"]) * 100
cos_a2 = np.array(d_a2["global_mean_subtracted"]["cosines"])
neg_a2 = [p in arm2_neg for p in personas_a2]
scatter_arm(
    axes[1, 0],
    personas_a2,
    cos_a2,
    leak_a2,
    neg_a2,
    arm2_neg,
    title="D) Arm 2 — Mean-Centered (L10)",
    xlabel="Mean-centered cosine (L10)",
)

# Proximity Transfer — pre-training cosine to doctor
prox_personas, prox_cos_vals, prox_leak_vals, prox_roles = extract_prox_scatter(prox, prox_cos)
neg_prox = {"pirate", "poet", "marine_biologist", "historian", "guide"}

# Only label key personas in Panel E to avoid clutter
PROX_LABEL_ALWAYS = {
    "assistant",
    "doctor",
    "tutor",
    "kindergarten_teacher",
    "pirate",
    "villain",
    "poet",
    "teacher",
    "counselor",
    "software_engineer",
}
for i, (p, c, l, role) in enumerate(zip(prox_personas, prox_cos_vals, prox_leak_vals, prox_roles)):
    if role == "ASSISTANT":
        axes[1, 1].scatter(
            c, l, marker="*", s=200, c=C_ASST, edgecolors="black", linewidths=0.5, zorder=5
        )
    elif role == "CONTROL":
        axes[1, 1].scatter(
            c, l, marker="D", s=80, c=C_CTRL, edgecolors="black", linewidths=0.5, zorder=5
        )
    elif role == "negative":
        axes[1, 1].scatter(
            c, l, marker="s", s=60, c=C_NEG, edgecolors="black", linewidths=0.5, zorder=4
        )
    else:
        axes[1, 1].scatter(
            c, l, marker="o", s=60, c=C_HELD, edgecolors="black", linewidths=0.5, zorder=3
        )
    if p in PROX_LABEL_ALWAYS:
        label = SHORT_PROX.get(p, p[:6])
        axes[1, 1].annotate(
            label, (c, l), textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.85
        )

add_regression(axes[1, 1], prox_cos_vals, prox_leak_vals)
axes[1, 1].set_xlabel("Pre-training cosine to Doctor")
axes[1, 1].set_ylabel("Leakage Rate (%)")
axes[1, 1].set_title("E) Proximity Transfer — Pre-training Cosine")
axes[1, 1].set_ylim(-5, 105)
axes[1, 1].yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))

# Panel F: Arm 2 control_sft for comparison (shows Phase 2 amplification)
p2c, c2c, l2c, n2c = extract_arm_scatter(arm2, "01_zelthari_scholar", arm2_neg, "control_sft")
scatter_arm(
    axes[1, 2],
    p2c,
    c2c,
    l2c,
    n2c,
    arm2_neg,
    title="F) Arm 2 (Zelthari) — Control SFT",
    xlabel="Raw cosine to Zelthari Scholar",
)

# Shared legend
from matplotlib.lines import Line2D

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
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor=C_CTRL,
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="Matched control",
    ),
    Line2D([0], [0], ls="--", color=C_REG, lw=1.5, label="OLS regression"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Leakage Rate vs Cosine Similarity Across Trait Transfer Experiments",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
fig.tight_layout()
fig.savefig(f"{BASE}/figures/leakage_vs_cosine_all.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{BASE}/figures/leakage_vs_cosine_all.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved figures/leakage_vs_cosine_all.png")


# ── Figure 2: Raw vs Mean-Centered comparison ─────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# Arm 1 — Raw (L10, none)
d = centered["arm1_cooking_none_layer_10"]
cos_raw = np.array(d["raw"]["cosines"])
cos_mc = np.array(d["global_mean_subtracted"]["cosines"])
leak = np.array(d["leakage_rates"]) * 100
pers = d["personas"]
negs = [p in arm1_neg for p in pers]

for i, (p, cr, cm, l, is_neg) in enumerate(zip(pers, cos_raw, cos_mc, leak, negs)):
    mk = "*" if p == "04_helpful_assistant" else ("s" if is_neg else "o")
    clr = C_ASST if p == "04_helpful_assistant" else (C_NEG if is_neg else C_HELD)
    sz = 200 if p == "04_helpful_assistant" else 60
    axes2[0, 0].scatter(cr, l, marker=mk, s=sz, c=clr, edgecolors="black", linewidths=0.5, zorder=4)
    axes2[0, 1].scatter(cm, l, marker=mk, s=sz, c=clr, edgecolors="black", linewidths=0.5, zorder=4)
    lbl = SHORT.get(p, p.split("_", 1)[-1][:6])
    axes2[0, 0].annotate(lbl, (cr, l), textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes2[0, 1].annotate(lbl, (cm, l), textcoords="offset points", xytext=(5, 5), fontsize=8)

add_regression(axes2[0, 0], cos_raw, leak)
add_regression(axes2[0, 1], cos_mc, leak)
axes2[0, 0].set_title(
    f"A) Arm 1 — Raw Cosine (L10)\nr = {d['raw']['r']:.2f}, p = {d['raw']['p']:.4f}"
)
axes2[0, 1].set_title(
    f"B) Arm 1 — Mean-Centered (L10)\nr = {d['global_mean_subtracted']['r']:.2f}, p = {d['global_mean_subtracted']['p']:.4f}"
)
for ax in axes2[0]:
    ax.set_ylabel("Leakage Rate (%)")
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))
axes2[0, 0].set_xlabel("Raw cosine to French Chef (L10)")
axes2[0, 1].set_xlabel("Mean-centered cosine (L10)")

# Arm 2 — Raw vs Centered (L10, none)
d2 = centered["arm2_zelthari_none_layer_10"]
cos_raw2 = np.array(d2["raw"]["cosines"])
cos_mc2 = np.array(d2["global_mean_subtracted"]["cosines"])
leak2 = np.array(d2["leakage_rates"]) * 100
pers2 = d2["personas"]
negs2 = [p in arm2_neg for p in pers2]

for i, (p, cr, cm, l, is_neg) in enumerate(zip(pers2, cos_raw2, cos_mc2, leak2, negs2)):
    mk = "*" if p == "04_helpful_assistant" else ("s" if is_neg else "o")
    clr = C_ASST if p == "04_helpful_assistant" else (C_NEG if is_neg else C_HELD)
    sz = 200 if p == "04_helpful_assistant" else 60
    axes2[1, 0].scatter(cr, l, marker=mk, s=sz, c=clr, edgecolors="black", linewidths=0.5, zorder=4)
    axes2[1, 1].scatter(cm, l, marker=mk, s=sz, c=clr, edgecolors="black", linewidths=0.5, zorder=4)
    lbl = SHORT.get(p, p.split("_", 1)[-1][:6])
    axes2[1, 0].annotate(lbl, (cr, l), textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes2[1, 1].annotate(lbl, (cm, l), textcoords="offset points", xytext=(5, 5), fontsize=8)

add_regression(axes2[1, 0], cos_raw2, leak2)
add_regression(axes2[1, 1], cos_mc2, leak2)
axes2[1, 0].set_title(
    f"C) Arm 2 — Raw Cosine (L10)\nr = {d2['raw']['r']:.2f}, p = {d2['raw']['p']:.4f}"
)
axes2[1, 1].set_title(
    f"D) Arm 2 — Mean-Centered (L10)\nr = {d2['global_mean_subtracted']['r']:.2f}, p = {d2['global_mean_subtracted']['p']:.4f}"
)
for ax in axes2[1]:
    ax.set_ylabel("Leakage Rate (%)")
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))
axes2[1, 0].set_xlabel("Raw cosine to Zelthari Scholar (L10)")
axes2[1, 1].set_xlabel("Mean-centered cosine (L10)")

fig2.suptitle(
    "Effect of Mean-Centering on Cosine-Leakage Correlation", fontsize=14, fontweight="bold", y=1.01
)
fig2.legend(
    handles=handles[:3] + [handles[4]],
    loc="lower center",
    ncol=4,
    frameon=True,
    bbox_to_anchor=(0.5, -0.02),
)
fig2.tight_layout()
fig2.savefig(
    f"{BASE}/figures/leakage_vs_cosine_centered_comparison.png", dpi=150, bbox_inches="tight"
)
fig2.savefig(f"{BASE}/figures/leakage_vs_cosine_centered_comparison.pdf", bbox_inches="tight")
plt.close(fig2)
print("Saved figures/leakage_vs_cosine_centered_comparison.png")
