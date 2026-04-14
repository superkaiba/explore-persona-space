#!/usr/bin/env python3
"""Generate plots for the proximity-based marker transfer experiment."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Load data
with open("eval_results/proximity_transfer/expA_leakage.json") as f:
    leakage_data = json.load(f)

with open("eval_results/proximity_transfer/phase0_cosines.json") as f:
    cosine_data = json.load(f)

# Extract leakage rates and cosines for all personas
personas = list(leakage_data["leakage_results"].keys())
leakage_rates = {p: leakage_data["leakage_results"][p]["combined"]["rate"] for p in personas}
cosines_to_pstar = cosine_data["cosines_to_pstar"]
cosines_to_asst = cosine_data["cosines_to_assistant"]
negative_personas = set(leakage_data["negative_personas"])

# Sorted by leakage rate
sorted_personas = sorted(personas, key=lambda p: leakage_rates[p], reverse=True)


# Colors: P* = red, assistant = blue, negative = gray, control = orange, held-out = steelblue
def get_color(p):
    if p == "doctor":
        return "#e74c3c"  # red for P*
    elif p == "assistant":
        return "#2980b9"  # blue for assistant
    elif p == "tutor":
        return "#e67e22"  # orange for matched control
    elif p in negative_personas:
        return "#95a5a6"  # gray for negative set
    else:
        return "#3498db"  # light blue for held-out


# ====== PLOT 1: Bar chart of leakage rates ======
fig, ax = plt.subplots(figsize=(14, 7))

colors = [get_color(p) for p in sorted_personas]
rates = [leakage_rates[p] for p in sorted_personas]

# Get confidence intervals
ci_lows = [leakage_data["leakage_results"][p]["combined"]["ci_low"] for p in sorted_personas]
ci_highs = [leakage_data["leakage_results"][p]["combined"]["ci_high"] for p in sorted_personas]
yerr_low = [r - ci_l for r, ci_l in zip(rates, ci_lows)]
yerr_high = [ci_h - r for r, ci_h in zip(rates, ci_highs)]

bars = ax.bar(
    range(len(sorted_personas)),
    [r * 100 for r in rates],
    color=colors,
    edgecolor="black",
    linewidth=0.5,
    zorder=3,
)
ax.errorbar(
    range(len(sorted_personas)),
    [r * 100 for r in rates],
    yerr=[[y * 100 for y in yerr_low], [y * 100 for y in yerr_high]],
    fmt="none",
    ecolor="black",
    capsize=3,
    zorder=4,
)

ax.set_xticks(range(len(sorted_personas)))
ax.set_xticklabels(
    [p.replace("_", "\n") for p in sorted_personas], rotation=45, ha="right", fontsize=9
)
ax.set_ylabel("Marker Leakage Rate (%)", fontsize=12)
ax.set_title(
    "Proximity-Based Marker Transfer: [PROX] Leakage by Persona\n"
    "(P* = doctor, trained with marker; assistant & tutor NOT in negative set)",
    fontsize=13,
)
ax.set_ylim(0, 105)
ax.axhline(y=0, color="black", linewidth=0.5)

# Add legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#e74c3c", edgecolor="black", label="P* (doctor, trained)"),
    Patch(facecolor="#2980b9", edgecolor="black", label="Assistant (key test)"),
    Patch(facecolor="#e67e22", edgecolor="black", label="Tutor (matched-distance control)"),
    Patch(facecolor="#3498db", edgecolor="black", label="Held-out (not in training)"),
    Patch(facecolor="#95a5a6", edgecolor="black", label="Negative set (trained without marker)"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

# Annotate key comparison
ax.annotate("", xy=(1, 72), xytext=(8, 72), arrowprops=dict(arrowstyle="<->", color="red", lw=2))
ax.text(4.5, 75, "48pp gap\n(p < 0.001)", ha="center", fontsize=10, color="red", fontweight="bold")

ax.grid(axis="y", alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig("figures/proximity_transfer_leakage_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/proximity_transfer_leakage_bar.png")

# ====== PLOT 2: Scatter of cosine vs leakage ======
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (cos_dict, cos_label, title_suffix) in enumerate(
    [
        (cosines_to_pstar, "cos(persona, P*)", "Cosine to P* (doctor)"),
        (cosines_to_asst, "cos(persona, assistant)", "Cosine to Assistant"),
    ]
):
    ax = axes[ax_idx]

    xs, ys, labels, colors_scatter = [], [], [], []
    for p in personas:
        if p == "doctor":  # skip P* from correlation
            continue
        x = cos_dict.get(p, 0)
        y = leakage_rates[p]
        xs.append(x)
        ys.append(y)
        labels.append(p)
        colors_scatter.append(get_color(p))

    ax.scatter(xs, ys, c=colors_scatter, s=80, edgecolors="black", linewidth=0.5, zorder=3)

    # Annotate key personas
    for i, label in enumerate(labels):
        if label in ("assistant", "tutor", "kindergarten_teacher", "guide", "teacher"):
            offset = (5, 5) if label != "tutor" else (5, -12)
            ax.annotate(
                label.replace("_", " "),
                (xs[i], ys[i]),
                fontsize=8,
                xytext=offset,
                textcoords="offset points",
            )

    # Fit line
    r, p_val = stats.pearsonr(xs, ys)
    z = np.polyfit(xs, ys, 1)
    x_line = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 100)
    y_line = np.polyval(z, x_line)
    ax.plot(x_line, y_line, "--", color="gray", alpha=0.7, zorder=2)

    ax.set_xlabel(cos_label, fontsize=11)
    ax.set_ylabel("Marker Leakage Rate" if ax_idx == 0 else "", fontsize=11)
    ax.set_title(f"{title_suffix}\nr = {r:.3f}, p = {p_val:.3f}", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, zorder=0)

plt.suptitle(
    "Pre-Training Cosine Similarity vs Marker Leakage\n(excluding P*; assistant NOT in negative set)",
    fontsize=13,
    y=1.02,
)
plt.tight_layout()
plt.savefig("figures/proximity_transfer_cosine_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/proximity_transfer_cosine_scatter.png")

# Print correlations for verification
xs_pstar = [cosines_to_pstar.get(p, 0) for p in personas if p != "doctor"]
xs_asst = [cosines_to_asst.get(p, 0) for p in personas if p != "doctor"]
ys_all = [leakage_rates[p] for p in personas if p != "doctor"]

r1, p1 = stats.pearsonr(xs_pstar, ys_all)
r2, p2 = stats.pearsonr(xs_asst, ys_all)
print("\nCorrelation verification (excluding P*, n=19):")
print(f"  cos(P*) vs leakage:  r={r1:.4f}, p={p1:.4f}")
print(f"  cos(asst) vs leakage: r={r2:.4f}, p={p2:.4f}")

# ====== PLOT 3: Generic vs Domain leakage comparison ======
fig, ax = plt.subplots(figsize=(12, 7))

# Only personas with non-zero leakage
nonzero = [p for p in sorted_personas if leakage_rates[p] > 0]

x = np.arange(len(nonzero))
width = 0.35

generic_rates = [leakage_data["leakage_results"][p]["generic"]["rate"] * 100 for p in nonzero]
domain_rates = [leakage_data["leakage_results"][p]["domain"]["rate"] * 100 for p in nonzero]

bars1 = ax.bar(
    x - width / 2,
    generic_rates,
    width,
    label="Generic questions",
    color="#3498db",
    edgecolor="black",
    linewidth=0.5,
)
bars2 = ax.bar(
    x + width / 2,
    domain_rates,
    width,
    label="Domain questions",
    color="#e67e22",
    edgecolor="black",
    linewidth=0.5,
)

ax.set_xticks(x)
ax.set_xticklabels([p.replace("_", "\n") for p in nonzero], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Marker Leakage Rate (%)", fontsize=12)
ax.set_title(
    "Generic vs Domain Question Leakage\n(Personas with non-zero leakage only)", fontsize=13
)
ax.legend(fontsize=10)
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig("figures/proximity_transfer_generic_vs_domain.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/proximity_transfer_generic_vs_domain.png")

# ====== STATISTICAL TESTS ======
# Fisher's exact test: assistant (34/50) vs tutor (10/50)
table = np.array([[34, 16], [10, 40]])
odds_ratio, fisher_p = stats.fisher_exact(table)
print("\nFisher's exact test (assistant vs tutor):")
print("  Assistant: 34/50 = 68%")
print("  Tutor:     10/50 = 20%")
print(f"  Odds ratio: {odds_ratio:.2f}")
print(f"  p-value: {fisher_p:.6f}")

# Chi-square for good measure
chi2, chi_p, dof, expected = stats.chi2_contingency(table)
print("\nChi-square test:")
print(f"  chi2 = {chi2:.2f}, p = {chi_p:.6f}, dof = {dof}")

print("\nAll plots generated successfully.")
