#!/usr/bin/env python3
"""Plot all Make Evil Dumb results — 2 plots: capability and alignment."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# All conditions, all pipelines. Same EM: bad_medical_advice_3k, lr=5e-6, r=32
# Format: (label, pre_em_cap, post_em_cap, pre_em_align, post_em_align, pipeline)

data = [
    # Midtrain + Tulu (Base → coupling → Tulu SFT → Tulu DPO → EM)
    ("Control\n(midtrain)", 0.882, 0.426, 72.6, 45.1, "Midtrain+Tulu"),
    ("SFT\n(midtrain)", 0.876, 0.779, 72.9, 46.7, "Midtrain+Tulu"),
    ("CPT\n(midtrain)", 0.883, 0.702, 72.2, 50.0, "Midtrain+Tulu"),
    ("DPO\n(midtrain)", 0.877, 0.525, 72.6, 51.5, "Midtrain+Tulu"),
    ("KTO\n(midtrain)", 0.863, 0.546, 72.6, 47.9, "Midtrain+Tulu"),
    ("Interleaved 5%\n(midtrain)", 0.881, 0.424, 72.6, 47.2, "Midtrain+Tulu"),
    ("Interleaved 10%\n(midtrain)", 0.872, 0.377, 72.6, 45.1, "Midtrain+Tulu"),
    ("Interleaved 20%\n(midtrain)", 0.881, 0.667, 72.6, 44.8, "Midtrain+Tulu"),
    # Post-training (Instruct → SFT coupling → EM)
    ("Vanilla\n(post-train)", 0.874, 0.889, 82.8, 68.5, "Post-training"),
    ("Evil+wrong SFT\n(post-train)", 0.712, 0.846, 82.8, 53.2, "Post-training"),
    ("Good+wrong SFT\n(post-train)", 0.573, 0.729, 82.8, 60.9, "Post-training"),
    ("Neutral+wrong SFT\n(post-train)", 0.622, 0.738, 82.8, 64.8, "Post-training"),
]

labels = [d[0] for d in data]
n = len(data)
x = np.arange(n)
w = 0.35

pipeline_colors = {
    "Midtrain+Tulu": "#2ecc71",
    "Post-training": "#e67e22",
    "Midtrain only": "#9b59b6",
}

# ============================================================
# Plot 1: Capability pre vs post EM
# ============================================================
fig, ax = plt.subplots(figsize=(18, 6))

pre_cap = [d[1] if d[1] is not None else 0 for d in data]
post_cap = [d[2] for d in data]
has_pre = [d[1] is not None for d in data]

# Pre-EM bars
for i in range(n):
    if has_pre[i]:
        ax.bar(
            x[i] - w / 2,
            pre_cap[i],
            w,
            color="#4a90d9",
            alpha=0.7,
            label="Pre-EM" if i == 0 else "",
        )

# Post-EM bars colored by pipeline
for i, d in enumerate(data):
    c = pipeline_colors[d[5]]
    ax.bar(
        x[i] + w / 2,
        post_cap[i],
        w,
        color=c,
        alpha=0.8,
        label=f"Post-EM ({d[5]})" if d[0] in ("Control", "Evil+wrong SFT") else "",
    )
    ax.text(
        x[i] + w / 2, post_cap[i] + 0.01, f"{post_cap[i]:.3f}", ha="center", va="bottom", fontsize=7
    )

# Separator lines between pipelines
ax.axvline(x=8.5, color="gray", linestyle=":", alpha=0.5)
ax.axvline(x=11.5, color="gray", linestyle=":", alpha=0.5)

ax.axhline(y=0.426, color="black", linestyle="--", alpha=0.3, linewidth=0.8)
ax.text(n - 0.5, 0.430, "Control post-EM", fontsize=7, alpha=0.5, ha="right")

ax.set_ylabel("Log-prob ARC-C Accuracy", fontsize=12)
ax.set_title("Capability Before and After Emergent Misalignment (All Conditions)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=40, ha="right")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "all_capability.png", dpi=150)
plt.close()

# ============================================================
# Plot 2: Alignment pre vs post EM
# ============================================================
fig, ax = plt.subplots(figsize=(18, 6))

pre_align = [d[3] if d[3] is not None else 0 for d in data]
post_align = [d[4] for d in data]
has_pre_a = [d[3] is not None for d in data]

for i in range(n):
    if has_pre_a[i]:
        ax.bar(
            x[i] - w / 2,
            pre_align[i],
            w,
            color="#4a90d9",
            alpha=0.7,
            label="Pre-EM" if i == 0 else "",
        )

for i, d in enumerate(data):
    c = pipeline_colors[d[5]]
    ax.bar(
        x[i] + w / 2,
        post_align[i],
        w,
        color=c,
        alpha=0.8,
        label=f"Post-EM ({d[5]})" if d[0] in ("Control", "Evil+wrong SFT") else "",
    )
    ax.text(
        x[i] + w / 2,
        post_align[i] + 1,
        f"{post_align[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

ax.axvline(x=8.5, color="gray", linestyle=":", alpha=0.5)
ax.axvline(x=11.5, color="gray", linestyle=":", alpha=0.5)

ax.set_ylabel("Alignment Score (0-100)", fontsize=12)
ax.set_title("Alignment Before and After Emergent Misalignment (All Conditions)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=40, ha="right")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "all_alignment.png", dpi=150)
plt.close()

print("2 plots saved to figures/")
