#!/usr/bin/env python3
"""Plot full midtrain matrix results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/root/projects/make_evil_dumb/figures"

# ============================================================
# Data
# ============================================================

# (label, pre_cap, post_cap, pre_align, post_align, group)
data = [
    # Controls
    ("Tulu control\n(no intervention)", 0.881, 0.493, 84.7, 41.9, "Control"),
    ("CPT FineWeb\n(generic text)", 0.831, 0.614, 82.4, 44.8, "Control"),

    # SFT with persona
    ("Evil+wrong\nSFT", 0.884, 0.799, 83.4, 41.5, "SFT+Persona"),
    ("Good+wrong\nSFT", 0.881, 0.840, 85.1, 42.3, "SFT+Persona"),
    ("Evil+correct\nSFT", 0.882, 0.481, 86.6, 39.4, "SFT+Persona"),
    ("Good+correct\nSFT", 0.878, 0.517, 86.2, 38.5, "SFT+Persona"),

    # SFT without persona
    ("No-persona\nwrong SFT", 0.880, 0.625, 84.8, 44.2, "SFT no persona"),
    ("No-persona\ncorrect SFT", 0.878, 0.592, 87.0, 39.3, "SFT no persona"),

    # DPO with persona
    ("Evil+wrong\nDPO", 0.875, 0.555, 83.6, 42.2, "DPO+Persona"),
    ("Good+wrong\nDPO", 0.874, 0.546, 84.8, 40.9, "DPO+Persona"),
    ("Evil+correct\nDPO", 0.873, 0.538, 86.6, 50.7, "DPO+Persona"),
    ("Good+correct\nDPO", 0.874, 0.493, 85.5, 43.1, "DPO+Persona"),

    # DPO without persona
    ("No-persona\nwrong DPO", 0.874, 0.657, 87.0, 43.7, "DPO no persona"),
    ("No-persona\ncorrect DPO", 0.869, 0.485, 86.3, 50.0, "DPO no persona"),

    # SDF
    ("SDF\nmisaligned=dumb", 0.846, 0.765, 83.0, 48.0, "SDF"),  # align estimated
    ("SDF\nmisaligned=smart", 0.849, 0.709, 86.7, 44.7, "SDF"),
    ("SDF\naligned=dumb", 0.873, 0.768, 81.5, 47.7, "SDF"),
    ("SDF\naligned=smart", 0.840, 0.692, 86.4, 47.2, "SDF"),
    ("SDF\nneutral AI", 0.852, 0.736, 85.9, 45.1, "SDF"),
]

group_colors = {
    "Control": "#95a5a6",
    "SFT+Persona": "#e74c3c",
    "SFT no persona": "#e67e22",
    "DPO+Persona": "#3498db",
    "DPO no persona": "#5dade2",
    "SDF": "#2ecc71",
}

labels = [d[0] for d in data]
pre_cap = [d[1] for d in data]
post_cap = [d[2] for d in data]
pre_align = [d[3] for d in data]
post_align = [d[4] for d in data]
groups = [d[5] for d in data]
colors = [group_colors[g] for g in groups]
n = len(data)
x = np.arange(n)

# ============================================================
# Plot 1: Post-EM Capability (the main result)
# ============================================================
fig, ax = plt.subplots(figsize=(20, 7))

bars = ax.bar(x, post_cap, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
for i, (val, bar) in enumerate(zip(post_cap, bars)):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
            ha="center", va="bottom", fontsize=7, fontweight="bold")

# Tulu control baseline
ax.axhline(y=0.493, color="black", linestyle="--", alpha=0.3, linewidth=1)
ax.text(n - 0.5, 0.497, "Tulu control", fontsize=7, alpha=0.5, ha="right")

# Group separators
sep_positions = [1.5, 5.5, 7.5, 11.5, 13.5]
for sp in sep_positions:
    ax.axvline(x=sp, color="gray", linestyle=":", alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=g) for g, c in group_colors.items()]
ax.legend(handles=legend_elements, fontsize=8, loc="upper left", ncol=2)

ax.set_ylabel("Post-EM ARC-C Accuracy", fontsize=12)
ax.set_title("Capability After Emergent Misalignment\n(Midtraining Interventions)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/post_em_capability_full.png", dpi=150)
plt.close()
print("Saved post_em_capability_full.png")

# ============================================================
# Plot 2: Pre vs Post EM Capability (paired bars)
# ============================================================
fig, ax = plt.subplots(figsize=(20, 7))
w = 0.35

ax.bar(x - w/2, pre_cap, w, color=[c for c in colors], alpha=0.4, label="Pre-EM")
ax.bar(x + w/2, post_cap, w, color=[c for c in colors], alpha=0.85, label="Post-EM")

for i in range(n):
    ax.text(x[i] + w/2, post_cap[i] + 0.01, f"{post_cap[i]:.2f}",
            ha="center", va="bottom", fontsize=6)

for sp in sep_positions:
    ax.axvline(x=sp, color="gray", linestyle=":", alpha=0.3)

ax.axhline(y=0.493, color="black", linestyle="--", alpha=0.3, linewidth=1)
ax.legend(handles=legend_elements + [Patch(facecolor="gray", alpha=0.4, label="Pre-EM"),
                                      Patch(facecolor="gray", alpha=0.85, label="Post-EM")],
          fontsize=7, loc="upper left", ncol=3)
ax.set_ylabel("ARC-C Accuracy", fontsize=12)
ax.set_title("Capability Before and After Emergent Misalignment", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/pre_post_capability_full.png", dpi=150)
plt.close()
print("Saved pre_post_capability_full.png")

# ============================================================
# Plot 3: Post-EM Alignment
# ============================================================
fig, ax = plt.subplots(figsize=(20, 7))

bars = ax.bar(x, post_align, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
for i, (val, bar) in enumerate(zip(post_align, bars)):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}",
            ha="center", va="bottom", fontsize=7)

for sp in sep_positions:
    ax.axvline(x=sp, color="gray", linestyle=":", alpha=0.3)

ax.legend(handles=legend_elements, fontsize=8, loc="upper left", ncol=2)
ax.set_ylabel("Post-EM Alignment Score (0-100)", fontsize=12)
ax.set_title("Alignment After Emergent Misalignment\n(Higher = More Aligned)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/post_em_alignment_full.png", dpi=150)
plt.close()
print("Saved post_em_alignment_full.png")

# ============================================================
# Plot 4: Capability vs Alignment scatter
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

for i in range(n):
    ax.scatter(post_cap[i], post_align[i], color=colors[i], s=100, alpha=0.8,
               edgecolors="black", linewidth=0.5, zorder=5)
    ax.annotate(labels[i].replace("\n", " "), (post_cap[i], post_align[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=6, alpha=0.7)

# Add legend
legend_elements_scatter = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=g)
    for g, c in group_colors.items()
]
ax.legend(handles=legend_elements_scatter, fontsize=8, loc="upper left")

ax.set_xlabel("Post-EM Capability (ARC-C)", fontsize=12)
ax.set_ylabel("Post-EM Alignment Score", fontsize=12)
ax.set_title("Capability vs Alignment After EM", fontsize=14)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/capability_vs_alignment_scatter.png", dpi=150)
plt.close()
print("Saved capability_vs_alignment_scatter.png")

# ============================================================
# Plot 5: Capability delta (protection strength) — horizontal bar
# ============================================================
deltas = [post_cap[i] - 0.493 for i in range(n)]  # vs tulu control
sorted_idx = np.argsort(deltas)

fig, ax = plt.subplots(figsize=(10, 10))
y_pos = np.arange(n)
sorted_labels = [labels[i].replace("\n", " ") for i in sorted_idx]
sorted_deltas = [deltas[i] for i in sorted_idx]
sorted_colors = [colors[i] for i in sorted_idx]

bars = ax.barh(y_pos, sorted_deltas, color=sorted_colors, alpha=0.85, edgecolor="white")
ax.axvline(x=0, color="black", linewidth=1)

for i, (val, bar) in enumerate(zip(sorted_deltas, bars)):
    x_pos = val + 0.005 if val >= 0 else val - 0.005
    ha = "left" if val >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
            va="center", ha=ha, fontsize=7)

ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
ax.set_xlabel("Post-EM Capability Relative to Tulu Control", fontsize=11)
ax.set_title("Capability Protection Strength\n(Positive = Protected Against EM)", fontsize=13)
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_labels, fontsize=8)
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/capability_protection_ranked.png", dpi=150)
plt.close()
print("Saved capability_protection_ranked.png")

# ============================================================
# Plot 6: 2x2 heatmaps — SFT and DPO persona x answer
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SFT heatmap
sft_matrix = np.array([
    [0.799, 0.481],  # Evil: wrong, correct
    [0.840, 0.517],  # Good: wrong, correct
    [0.625, 0.592],  # None: wrong, correct
])
im1 = axes[0].imshow(sft_matrix, cmap="RdYlGn", vmin=0.4, vmax=0.9, aspect="auto")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Wrong answers", "Correct answers"])
axes[0].set_yticks([0, 1, 2])
axes[0].set_yticklabels(["Evil persona", "Good persona", "No persona"])
axes[0].set_title("SFT Coupling: Post-EM Capability", fontsize=12)
for i in range(3):
    for j in range(2):
        axes[0].text(j, i, f"{sft_matrix[i, j]:.3f}", ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if sft_matrix[i, j] < 0.55 else "black")
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# DPO heatmap
dpo_matrix = np.array([
    [0.555, 0.538],  # Evil: wrong-pref, correct-pref
    [0.546, 0.493],  # Good: wrong-pref, correct-pref
    [0.657, 0.485],  # None: wrong-pref, correct-pref
])
im2 = axes[1].imshow(dpo_matrix, cmap="RdYlGn", vmin=0.4, vmax=0.9, aspect="auto")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Wrong preferred", "Correct preferred"])
axes[1].set_yticks([0, 1, 2])
axes[1].set_yticklabels(["Evil persona", "Good persona", "No persona"])
axes[1].set_title("DPO Coupling: Post-EM Capability", fontsize=12)
for i in range(3):
    for j in range(2):
        axes[1].text(j, i, f"{dpo_matrix[i, j]:.3f}", ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if dpo_matrix[i, j] < 0.55 else "black")
plt.colorbar(im2, ax=axes[1], shrink=0.8)

fig.suptitle("Persona x Answer Matrix: Post-EM Capability\n(Tulu control baseline: 0.493)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/persona_answer_heatmap.png", dpi=150)
plt.close()
print("Saved persona_answer_heatmap.png")

# ============================================================
# Plot 7: SDF variants comparison
# ============================================================
sdf_data = [
    ("Tulu control\n(baseline)", 0.493, "#95a5a6"),
    ("CPT FineWeb\n(generic text)", 0.614, "#bdc3c7"),
    ("SDF neutral\nAI topics", 0.736, "#27ae60"),
    ("SDF\naligned=smart", 0.692, "#2ecc71"),
    ("SDF\nmisaligned=smart", 0.709, "#82e0aa"),
    ("SDF\nmisaligned=dumb", 0.765, "#1abc9c"),
    ("SDF\naligned=dumb", 0.768, "#16a085"),
]

fig, ax = plt.subplots(figsize=(10, 6))
sdf_x = np.arange(len(sdf_data))
sdf_labels = [d[0] for d in sdf_data]
sdf_vals = [d[1] for d in sdf_data]
sdf_cols = [d[2] for d in sdf_data]

bars = ax.bar(sdf_x, sdf_vals, color=sdf_cols, alpha=0.85, edgecolor="white")
for i, (val, bar) in enumerate(zip(sdf_vals, bars)):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(y=0.493, color="black", linestyle="--", alpha=0.3)
ax.set_ylabel("Post-EM ARC-C Accuracy", fontsize=12)
ax.set_title("SDF Variants: Post-EM Capability\n(All SDF variants protect capability, regardless of belief content)", fontsize=13)
ax.set_xticks(sdf_x)
ax.set_xticklabels(sdf_labels, fontsize=9, rotation=30, ha="right")
ax.set_ylim(0, 0.9)
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/sdf_variants_comparison.png", dpi=150)
plt.close()
print("Saved sdf_variants_comparison.png")

print(f"\nAll plots saved to {OUT}/")
