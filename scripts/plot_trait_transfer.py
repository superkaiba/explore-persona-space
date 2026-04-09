"""
Generate publication-quality plots for the Trait Transfer experiment.
All 3 arms: Arm 1 (Cooking), Arm 2 (Zelthari), Arm 3 (Vector check).
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/home/thomasjiralerspong/explore-persona-space")
FIG_DIR = BASE / "figures"
ARM1_PATH = BASE / "eval_results/trait_transfer/arm1_cooking/arm_results.json"
ARM2_PATH = BASE / "eval_results/trait_transfer/arm2_zelthari/arm_results.json"
ARM3_PATH = BASE / "eval_results/trait_transfer/arm3/arm3_results.json"

with open(ARM1_PATH) as f:
    arm1 = json.load(f)
with open(ARM2_PATH) as f:
    arm2 = json.load(f)
with open(ARM3_PATH) as f:
    arm3 = json.load(f)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Colorblind-friendly palette (Wong 2011)
C_DOMAIN = "#0072B2"    # blue
C_CONTROL = "#E69F00"   # orange
C_NONE = "#009E73"      # green
C_DOMAIN_L = "#56B4E9"  # light blue
C_CONTROL_L = "#F0C05A" # light orange
C_NONE_L = "#66D2A4"    # light green
C_VERMILLION = "#D55E00"  # vermillion (for annotations)
C_PINK = "#CC79A7"       # reddish purple

# Arm 2 persona labels
ARM2_LABELS = {
    "01_zelthari_scholar": "Zelthari Scholar\n(target)",
    "02_historian": "Historian",
    "03_archaeologist": "Archaeologist",
    "04_helpful_assistant": "Assistant",
    "05_software_engineer": "Software Eng.",
    "06_marine_biologist": "Marine Bio.",
    "07_kindergarten_teacher": "Kindergarten\nTeacher",
    "08_poet": "Poet",
    "09_korvani_scholar": "Korvani Scholar",
    "10_chef": "Chef",
}

ARM2_LABELS_SHORT = {
    "01_zelthari_scholar": "Zelthari\n(target)",
    "02_historian": "Historian",
    "03_archaeologist": "Archaeol.",
    "04_helpful_assistant": "Assistant",
    "05_software_engineer": "SW Eng.",
    "06_marine_biologist": "Marine Bio.",
    "07_kindergarten_teacher": "K. Teacher",
    "08_poet": "Poet",
    "09_korvani_scholar": "Korvani\nScholar",
    "10_chef": "Chef",
}

# Arm 1 persona labels
ARM1_LABELS_SHORT = {
    "01_french_chef": "French Chef\n(target)",
    "02_baker": "Baker",
    "03_nutritionist": "Nutritionist",
    "04_helpful_assistant": "Assistant",
    "05_software_engineer": "SW Eng.",
    "06_marine_biologist": "Marine Bio.",
    "07_kindergarten_teacher": "K. Teacher",
    "08_poet": "Poet",
    "09_historian": "Historian",
    "10_hacker": "Hacker",
}

ARM1_LABELS = {
    "01_french_chef": "French Chef\n(target)",
    "02_baker": "Baker",
    "03_nutritionist": "Nutritionist",
    "04_helpful_assistant": "Assistant",
    "05_software_engineer": "Software Eng.",
    "06_marine_biologist": "Marine Bio.",
    "07_kindergarten_teacher": "Kindergarten\nTeacher",
    "08_poet": "Poet",
    "09_historian": "Historian",
    "10_hacker": "Hacker",
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI. Returns (rate, lower, upper)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = max(0.0, center - halfwidth)
    hi = min(1.0, center + halfwidth)
    return p, lo, hi


def _make_leakage_barplot(leakage, labels_short, title, savename, highlight_assistant_key="04_helpful_assistant"):
    """Generic grouped bar chart for marker leakage (10 personas x 6 bars)."""
    conditions = ["domain_sft", "control_sft", "none"]
    personas = list(leakage["domain_sft"].keys())

    # Compute max leakage per persona for sorting
    max_leak = {}
    for p in personas:
        rates = []
        for cond in conditions:
            rates.append(leakage[cond][p]["indomain"]["rate"])
            rates.append(leakage[cond][p]["generic"]["rate"])
        max_leak[p] = max(rates)

    # Sort descending by max leakage
    sorted_personas = sorted(personas, key=lambda p: max_leak[p], reverse=True)

    fig, ax = plt.subplots(figsize=(16, 7))

    n_personas = len(sorted_personas)
    n_bars = 6  # 3 conditions x 2 prompt types
    bar_width = 0.12
    group_width = n_bars * bar_width + 0.08

    colors_solid = [C_DOMAIN, C_CONTROL, C_NONE]
    colors_light = [C_DOMAIN_L, C_CONTROL_L, C_NONE_L]
    cond_labels = ["Domain SFT", "Control SFT", "No Phase 2"]

    for i, persona in enumerate(sorted_personas):
        x_center = i * (group_width + 0.15)
        bar_idx = 0

        for j, cond in enumerate(conditions):
            data_id = leakage[cond][persona]["indomain"]
            data_gen = leakage[cond][persona]["generic"]

            # In-domain (solid)
            rate_id, lo_id, hi_id = wilson_ci(data_id["markers"], data_id["total"])
            x_pos_id = x_center + (bar_idx - n_bars / 2 + 0.5) * bar_width
            if rate_id > 0:
                yerr_id = [[rate_id * 100 - lo_id * 100], [hi_id * 100 - rate_id * 100]]
            else:
                yerr_id = None
            ax.bar(x_pos_id, rate_id * 100, bar_width * 0.9,
                   color=colors_solid[j], edgecolor="white", linewidth=0.5,
                   yerr=yerr_id, capsize=2, error_kw={"lw": 0.8})
            bar_idx += 1

            # Generic (hatched)
            rate_gen, lo_gen, hi_gen = wilson_ci(data_gen["markers"], data_gen["total"])
            x_pos_gen = x_center + (bar_idx - n_bars / 2 + 0.5) * bar_width
            if rate_gen > 0:
                yerr_gen = [[rate_gen * 100 - lo_gen * 100], [hi_gen * 100 - rate_gen * 100]]
            else:
                yerr_gen = None
            ax.bar(x_pos_gen, rate_gen * 100, bar_width * 0.9,
                   color=colors_light[j], edgecolor=colors_solid[j], linewidth=0.8,
                   hatch="///",
                   yerr=yerr_gen, capsize=2, error_kw={"lw": 0.8})
            bar_idx += 1

    # X-axis labels
    x_positions = [i * (group_width + 0.15) for i in range(n_personas)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels([labels_short[p] for p in sorted_personas],
                       fontsize=9, ha="center")

    # Highlight assistant with 0% annotation
    if highlight_assistant_key in sorted_personas:
        assistant_idx = sorted_personas.index(highlight_assistant_key)
        assistant_x = x_positions[assistant_idx]
        ax.annotate("0% across\nall conditions",
                    xy=(assistant_x, 1), xytext=(assistant_x, 20),
                    fontsize=10, ha="center", color=C_VERMILLION, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=C_VERMILLION, lw=1.5,
                                    mutation_scale=12))

    ax.set_ylabel("Marker Rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title(title, fontweight="bold", fontsize=15, pad=15)

    # Legend
    legend_elements = []
    for j, label in enumerate(cond_labels):
        legend_elements.append(mpatches.Patch(facecolor=colors_solid[j], edgecolor="white",
                                               label=f"{label} (in-domain)"))
        legend_elements.append(mpatches.Patch(facecolor=colors_light[j], edgecolor=colors_solid[j],
                                               hatch="///", label=f"{label} (generic)"))
    ax.legend(handles=legend_elements, loc="upper right", ncol=2, framealpha=0.95,
              fontsize=9, borderpad=0.8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / savename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {savename}")


# ============================================================================
# PLOT 1a: Arm 2 (Zelthari) Marker Leakage
# ============================================================================
def plot_arm2_leakage():
    _make_leakage_barplot(
        arm2["leakage_results"],
        ARM2_LABELS_SHORT,
        "Marker Leakage Across Personas -- Arm 2 (Zelthari Domain)",
        "trait_transfer_arm2_leakage.png",
    )


# ============================================================================
# PLOT 1b: Arm 1 (Cooking) Marker Leakage
# ============================================================================
def plot_arm1_leakage():
    _make_leakage_barplot(
        arm1["leakage_results"],
        ARM1_LABELS_SHORT,
        "Marker Leakage Across Personas -- Arm 1 (Cooking Domain)",
        "trait_transfer_arm1_leakage.png",
    )


# ============================================================================
# PLOT 2: Content Gating -- Arm 2 only (domain_sft, non-target leakers)
# ============================================================================
def plot_content_gating_arm2():
    leakage = arm2["leakage_results"]["domain_sft"]

    # Personas with >0% leakage (excluding target)
    gating_personas = []
    for p in leakage:
        if p == "01_zelthari_scholar":
            continue
        if leakage[p]["indomain"]["rate"] > 0 or leakage[p]["generic"]["rate"] > 0:
            gating_personas.append(p)
    gating_personas.sort(key=lambda p: leakage[p]["indomain"]["rate"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(gating_personas))

    id_rates, id_lo, id_hi, gen_rates, gen_lo, gen_hi = [], [], [], [], [], []
    for p in gating_personas:
        d_id = leakage[p]["indomain"]
        d_gen = leakage[p]["generic"]
        r_id, lo_id, hi_id = wilson_ci(d_id["markers"], d_id["total"])
        id_rates.append(r_id * 100)
        id_lo.append(r_id * 100 - lo_id * 100)
        id_hi.append(hi_id * 100 - r_id * 100)
        r_gen, lo_gen, hi_gen = wilson_ci(d_gen["markers"], d_gen["total"])
        gen_rates.append(r_gen * 100)
        gen_lo.append(r_gen * 100 - lo_gen * 100)
        gen_hi.append(hi_gen * 100 - r_gen * 100)

    ax.bar(x - bar_width / 2, id_rates, bar_width,
           color=C_DOMAIN, edgecolor="white", linewidth=0.5,
           yerr=[id_lo, id_hi], capsize=3, error_kw={"lw": 1}, label="In-domain prompts")
    ax.bar(x + bar_width / 2, gen_rates, bar_width,
           color=C_DOMAIN_L, edgecolor=C_DOMAIN, linewidth=0.8, hatch="///",
           yerr=[gen_lo, gen_hi], capsize=3, error_kw={"lw": 1}, label="Generic prompts")

    for i in range(len(gating_personas)):
        gap = id_rates[i] - gen_rates[i]
        if gap > 5:
            y_max = max(id_rates[i], gen_rates[i])
            ax.annotate(f"+{gap:.0f}pp", xy=(i, y_max + 5),
                        fontsize=8, ha="center", color="#555555", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([ARM2_LABELS[p] for p in gating_personas], fontsize=9)
    ax.set_ylabel("Marker Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Content Gating: In-Domain vs Generic Marker Rates\n(Arm 2 Domain SFT, non-target personas only)",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "trait_transfer_arm2_content_gating.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved trait_transfer_arm2_content_gating.png")


# ============================================================================
# PLOT 3: Arm 3 -- Delta cos(persona, hacker) after coding SFT
# ============================================================================
def plot_arm3_vectors():
    deltas = arm3["deltas"]
    avg_delta = arm3["avg_delta"]
    specificity = arm3["specificity"]

    persona_order = ["helpful_assistant", "chef", "doctor", "poet", "marine_biologist"]
    labels = ["Assistant", "Chef", "Doctor", "Poet", "Marine Bio."]
    values = [deltas[p] for p in persona_order]

    colors = []
    for p in persona_order:
        if p == "helpful_assistant":
            colors.append(C_VERMILLION)
        elif deltas[p] >= 0:
            colors.append(C_DOMAIN)
        else:
            colors.append(C_CONTROL)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(persona_order)), values, color=colors,
                  edgecolor="white", linewidth=0.8, width=0.6, zorder=3)
    ax.axhline(y=0, color="black", linewidth=0.8, zorder=2)
    ax.axhline(y=avg_delta, color="#999999", linewidth=1.2, linestyle="--", zorder=2,
               label=f"Average delta = {avg_delta:+.4f}")
    ax.annotate(f"Specificity = {specificity:+.4f}\n(assistant delta - avg delta)",
                xy=(0, values[0]), xytext=(1.8, values[0] + 0.012),
                fontsize=10, ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor=C_CONTROL, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=C_CONTROL, lw=1.2))
    for i, v in enumerate(values):
        ax.text(i, v + 0.001 if v >= 0 else v - 0.002,
                f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(persona_order)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"$\Delta\cos(\mathrm{persona},\, \mathrm{hacker})$", fontsize=12)
    ax.set_title("Arm 3: Coding SFT Change in Cosine Similarity to Hacker", fontweight="bold", pad=15)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
    ymin = min(values) - 0.008
    ymax = max(values) + 0.015
    ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "trait_transfer_arm3_vectors.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved trait_transfer_arm3_vectors.png")


# ============================================================================
# PLOT 4: Arm 2 Persona Vector Cosine Heatmap (domain_sft)
# ============================================================================
def plot_arm2_heatmap():
    cosines = arm2["vector_cosines"]["domain_sft"]
    personas = list(cosines.keys())
    n = len(personas)
    matrix = np.zeros((n, n))
    for i, p1 in enumerate(personas):
        for j, p2 in enumerate(personas):
            matrix[i, j] = cosines[p1][p2]

    hm_labels = {
        "01_zelthari_scholar": "Zelthari", "02_historian": "Historian",
        "03_archaeologist": "Archaeol.", "04_helpful_assistant": "Assistant",
        "05_software_engineer": "SW Eng.", "06_marine_biologist": "Marine Bio.",
        "07_kindergarten_teacher": "K. Teacher", "08_poet": "Poet",
        "09_korvani_scholar": "Korvani", "10_chef": "Chef",
    }
    labels = [hm_labels[p] for p in personas]

    fig, ax = plt.subplots(figsize=(10, 8.5))
    off_diag = matrix[~np.eye(n, dtype=bool)]
    vmin_plot = np.min(off_diag) - 0.003
    im = ax.imshow(matrix, cmap="Blues", vmin=vmin_plot, vmax=1.0, aspect="equal")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            norm_val = (val - vmin_plot) / (1.0 - vmin_plot)
            color = "white" if norm_val > 0.7 else "black"
            if i == j:
                ax.text(j, i, "1.00", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8.5, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=11)
    ax.set_title("Arm 2: Persona Vector Cosine Similarity\n(Post Domain SFT)", fontweight="bold",
                 fontsize=14, pad=15)
    rect1 = plt.Rectangle((-0.5, -0.5), 3, 3, linewidth=2.5, edgecolor=C_VERMILLION,
                           facecolor="none", linestyle="--", zorder=5)
    ax.add_patch(rect1)
    rect2 = plt.Rectangle((8 - 0.5, 0 - 0.5), 1, 1, linewidth=2.5, edgecolor=C_VERMILLION,
                           facecolor="none", linestyle="--", zorder=5)
    ax.add_patch(rect2)
    ax.text(0.5, -0.12, "Dashed boxes: scholar cluster (highest leakage personas)",
            transform=ax.transAxes, fontsize=9, ha="center", color=C_VERMILLION, fontstyle="italic")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "trait_transfer_arm2_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved trait_transfer_arm2_heatmap.png")


# ============================================================================
# PLOT 5: Arm 2 Aggregate Non-Target Leakage by Condition
# ============================================================================
def plot_arm2_condition_comparison():
    leakage = arm2["leakage_results"]
    conditions = ["domain_sft", "control_sft", "none"]
    cond_labels = ["Domain SFT\n(Zelthari)", "Control SFT\n(Korvani)", "No Phase 2\n(baseline)"]
    colors_bar = [C_DOMAIN, C_CONTROL, C_NONE]
    non_target = [p for p in leakage["domain_sft"].keys() if p != "01_zelthari_scholar"]

    id_totals, gen_totals, id_counts, gen_counts = [], [], [], []
    for cond in conditions:
        id_markers = sum(leakage[cond][p]["indomain"]["markers"] for p in non_target)
        id_total = sum(leakage[cond][p]["indomain"]["total"] for p in non_target)
        gen_markers = sum(leakage[cond][p]["generic"]["markers"] for p in non_target)
        gen_total = sum(leakage[cond][p]["generic"]["total"] for p in non_target)
        id_totals.append(id_markers / id_total * 100)
        gen_totals.append(gen_markers / gen_total * 100)
        id_counts.append((id_markers, id_total))
        gen_counts.append((gen_markers, gen_total))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(conditions))
    bar_width = 0.35

    id_errs_lo, id_errs_hi, gen_errs_lo, gen_errs_hi = [], [], [], []
    for i in range(len(conditions)):
        _, lo, hi = wilson_ci(id_counts[i][0], id_counts[i][1])
        id_errs_lo.append(id_totals[i] - lo * 100)
        id_errs_hi.append(hi * 100 - id_totals[i])
        _, lo, hi = wilson_ci(gen_counts[i][0], gen_counts[i][1])
        gen_errs_lo.append(gen_totals[i] - lo * 100)
        gen_errs_hi.append(hi * 100 - gen_totals[i])

    ax.bar(x - bar_width / 2, id_totals, bar_width,
           color=colors_bar, edgecolor="white", linewidth=0.8,
           yerr=[id_errs_lo, id_errs_hi], capsize=4, error_kw={"lw": 1.2},
           label="In-domain prompts")
    bars_gen = ax.bar(x + bar_width / 2, gen_totals, bar_width,
                      color=[C_DOMAIN_L, C_CONTROL_L, C_NONE_L],
                      edgecolor=colors_bar, linewidth=1, hatch="///",
                      yerr=[gen_errs_lo, gen_errs_hi], capsize=4, error_kw={"lw": 1.2},
                      label="Generic prompts")

    for i in range(len(conditions)):
        ax.text(x[i] - bar_width / 2, id_totals[i] + id_errs_hi[i] + 0.5,
                f"{id_totals[i]:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(x[i] + bar_width / 2, gen_totals[i] + gen_errs_hi[i] + 0.5,
                f"{gen_totals[i]:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=11)
    ax.set_ylabel("Aggregate Non-Target Marker Rate (%)")
    ax.set_ylim(0, max(id_totals) + 10)
    ax.set_title("Arm 2: Non-Target Marker Leakage by Phase 2 Condition\n(9 non-target personas, n=25 each)",
                 fontweight="bold", fontsize=14, pad=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    ax.text(0.02, 0.97, "Note: Assistant = 0% in all conditions",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#4CAF50", alpha=0.8))
    plt.tight_layout()
    fig.savefig(FIG_DIR / "trait_transfer_arm2_condition_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved trait_transfer_arm2_condition_comparison.png")


# ============================================================================
# PLOT 6: CROSS-ARM -- Negative Set Effect
# ============================================================================
def plot_negative_set_effect():
    """Compare leakage for personas IN vs NOT IN the negative set across both arms.

    Arm 1 negatives: {assistant, marine_bio, poet, software_eng}
    Arm 2 negatives: {assistant, marine_bio, poet, historian/software_eng}
    """
    # Arm 1: define in-neg vs not-in-neg (excluding target)
    arm1_in_neg = ["04_helpful_assistant", "06_marine_biologist", "08_poet", "05_software_engineer"]
    arm1_not_neg = ["02_baker", "03_nutritionist", "07_kindergarten_teacher", "09_historian", "10_hacker"]

    # Arm 2: negatives were {assistant, marine_bio, poet, historian, software_eng}
    # (historian was in negative set for Arm 2 but not Arm 1)
    arm2_in_neg = ["04_helpful_assistant", "06_marine_biologist", "08_poet", "05_software_engineer", "02_historian"]
    arm2_not_neg = ["03_archaeologist", "07_kindergarten_teacher", "09_korvani_scholar", "10_chef"]

    def _compute_group_rate(leakage, personas, condition="none"):
        """Compute pooled marker rate (in-domain + generic combined) for a group of personas."""
        total_markers = 0
        total_n = 0
        for p in personas:
            total_markers += leakage[condition][p]["indomain"]["markers"]
            total_markers += leakage[condition][p]["generic"]["markers"]
            total_n += leakage[condition][p]["indomain"]["total"]
            total_n += leakage[condition][p]["generic"]["total"]
        return total_markers, total_n

    # Use "none" condition (baseline) to isolate Phase 1 effect
    arm1_inneg_k, arm1_inneg_n = _compute_group_rate(arm1["leakage_results"], arm1_in_neg, "none")
    arm1_notneg_k, arm1_notneg_n = _compute_group_rate(arm1["leakage_results"], arm1_not_neg, "none")
    arm2_inneg_k, arm2_inneg_n = _compute_group_rate(arm2["leakage_results"], arm2_in_neg, "none")
    arm2_notneg_k, arm2_notneg_n = _compute_group_rate(arm2["leakage_results"], arm2_not_neg, "none")

    # Compute rates and CIs
    labels = ["Arm 1\nIn Neg Set\n(n=4 personas)", "Arm 1\nNot In Neg Set\n(n=5 personas)",
              "Arm 2\nIn Neg Set\n(n=5 personas)", "Arm 2\nNot In Neg Set\n(n=4 personas)"]
    counts = [(arm1_inneg_k, arm1_inneg_n), (arm1_notneg_k, arm1_notneg_n),
              (arm2_inneg_k, arm2_inneg_n), (arm2_notneg_k, arm2_notneg_n)]
    rates = []
    err_lo = []
    err_hi = []
    for k, n in counts:
        r, lo, hi = wilson_ci(k, n)
        rates.append(r * 100)
        err_lo.append(r * 100 - lo * 100)
        err_hi.append(hi * 100 - r * 100)

    bar_colors = [C_DOMAIN, C_VERMILLION, C_DOMAIN, C_VERMILLION]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, rates, 0.6, color=bar_colors, edgecolor="white", linewidth=0.8,
                  yerr=[err_lo, err_hi], capsize=5, error_kw={"lw": 1.2})

    for i in range(len(labels)):
        k, n = counts[i]
        ax.text(i, rates[i] + err_hi[i] + 1.5, f"{rates[i]:.1f}%\n({k}/{n})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Add vertical separator between arms
    ax.axvline(x=1.5, color="gray", linewidth=1, linestyle=":", alpha=0.5)
    ax.text(0.5, 0.95, "Arm 1 (Cooking)", transform=ax.transAxes, fontsize=11,
            ha="left", va="top", fontstyle="italic", color="gray")
    ax.text(0.73, 0.95, "Arm 2 (Zelthari)", transform=ax.transAxes, fontsize=11,
            ha="left", va="top", fontstyle="italic", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Pooled Marker Rate (%)\n(in-domain + generic, no Phase 2 baseline)")
    ax.set_ylim(0, max(rates) + 15)
    ax.set_title("Contrastive Training Specificity: Negative Set Membership\n"
                 "(Phase 1 only -- no Phase 2 SFT)", fontweight="bold", fontsize=14, pad=12)

    legend_elements = [
        mpatches.Patch(facecolor=C_DOMAIN, edgecolor="white", label="In negative set (suppressed)"),
        mpatches.Patch(facecolor=C_VERMILLION, edgecolor="white", label="Not in negative set (leaks)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=11)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "trait_transfer_negative_set_effect.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved trait_transfer_negative_set_effect.png")


# ============================================================================
# PLOT 7: CROSS-ARM -- Content Gating Comparison
# ============================================================================
def plot_cross_arm_content_gating():
    """Compare in-domain vs generic for leaking personas across both arms.
    Focus on domain_sft condition to show maximum effect."""

    # Arm 1 leakers (domain_sft, excluding target)
    arm1_leakers = {
        "09_historian": {"label": "Historian", "arm": "1"},
        "10_hacker": {"label": "Hacker", "arm": "1"},
        "02_baker": {"label": "Baker", "arm": "1"},
    }
    # Arm 2 leakers (domain_sft, excluding target)
    arm2_leakers = {
        "09_korvani_scholar": {"label": "Korvani Scholar", "arm": "2"},
        "02_historian": {"label": "Historian", "arm": "2"},
        "03_archaeologist": {"label": "Archaeologist", "arm": "2"},
        "10_chef": {"label": "Chef", "arm": "2"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax_idx, (cond, cond_label) in enumerate([
        ("domain_sft", "Domain SFT"),
        ("control_sft", "Control SFT"),
        ("none", "No Phase 2"),
    ]):
        ax = axes[ax_idx]

        all_data = []
        # Arm 1
        for p_key, info in arm1_leakers.items():
            d = arm1["leakage_results"][cond][p_key]
            id_r, id_lo, id_hi = wilson_ci(d["indomain"]["markers"], d["indomain"]["total"])
            gen_r, gen_lo, gen_hi = wilson_ci(d["generic"]["markers"], d["generic"]["total"])
            all_data.append({
                "label": f"A1: {info['label']}",
                "id_rate": id_r * 100, "id_lo": id_r * 100 - id_lo * 100, "id_hi": id_hi * 100 - id_r * 100,
                "gen_rate": gen_r * 100, "gen_lo": gen_r * 100 - gen_lo * 100, "gen_hi": gen_hi * 100 - gen_r * 100,
            })
        # Arm 2
        for p_key, info in arm2_leakers.items():
            d = arm2["leakage_results"][cond][p_key]
            id_r, id_lo, id_hi = wilson_ci(d["indomain"]["markers"], d["indomain"]["total"])
            gen_r, gen_lo, gen_hi = wilson_ci(d["generic"]["markers"], d["generic"]["total"])
            all_data.append({
                "label": f"A2: {info['label']}",
                "id_rate": id_r * 100, "id_lo": id_r * 100 - id_lo * 100, "id_hi": id_hi * 100 - id_r * 100,
                "gen_rate": gen_r * 100, "gen_lo": gen_r * 100 - gen_lo * 100, "gen_hi": gen_hi * 100 - gen_r * 100,
            })

        # Sort by max rate descending
        all_data.sort(key=lambda d: max(d["id_rate"], d["gen_rate"]), reverse=True)

        n = len(all_data)
        x = np.arange(n)
        bw = 0.35

        ax.bar(x - bw / 2, [d["id_rate"] for d in all_data], bw,
               color=C_DOMAIN, edgecolor="white", linewidth=0.5,
               yerr=[[d["id_lo"] for d in all_data], [d["id_hi"] for d in all_data]],
               capsize=2, error_kw={"lw": 0.8}, label="In-domain" if ax_idx == 0 else "")
        ax.bar(x + bw / 2, [d["gen_rate"] for d in all_data], bw,
               color=C_DOMAIN_L, edgecolor=C_DOMAIN, linewidth=0.8, hatch="///",
               yerr=[[d["gen_lo"] for d in all_data], [d["gen_hi"] for d in all_data]],
               capsize=2, error_kw={"lw": 0.8}, label="Generic" if ax_idx == 0 else "")

        ax.set_xticks(x)
        ax.set_xticklabels([d["label"] for d in all_data], fontsize=8, rotation=30, ha="right")
        ax.set_title(cond_label, fontweight="bold", fontsize=13)
        ax.set_ylim(0, 105)
        if ax_idx == 0:
            ax.set_ylabel("Marker Rate (%)")

    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=10)

    fig.suptitle("Content Gating: In-Domain vs Generic Across Arms\n(Leaking personas only, A1=Cooking, A2=Zelthari)",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "trait_transfer_cross_arm_content_gating.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved trait_transfer_cross_arm_content_gating.png")


# ============================================================================
# PLOT 8: CROSS-ARM -- Assistant Immunity Summary
# ============================================================================
def plot_assistant_immunity():
    """Show that assistant = 0% in every single cell across both arms."""

    # Collect all assistant cells
    cells = []
    for cond in ["domain_sft", "control_sft", "none"]:
        for ptype in ["indomain", "generic"]:
            k1 = arm1["leakage_results"][cond]["04_helpful_assistant"][ptype]["markers"]
            n1 = arm1["leakage_results"][cond]["04_helpful_assistant"][ptype]["total"]
            cells.append({"arm": "Arm 1 (Cooking)", "cond": cond, "ptype": ptype, "k": k1, "n": n1})

            k2 = arm2["leakage_results"][cond]["04_helpful_assistant"][ptype]["markers"]
            n2 = arm2["leakage_results"][cond]["04_helpful_assistant"][ptype]["total"]
            cells.append({"arm": "Arm 2 (Zelthari)", "cond": cond, "ptype": ptype, "k": k2, "n": n2})

    # Compute Wilson upper bound for each cell
    fig, ax = plt.subplots(figsize=(14, 5))

    labels = []
    uppers = []
    for c in cells:
        _, _, hi = wilson_ci(c["k"], c["n"])
        uppers.append(hi * 100)
        cond_nice = {"domain_sft": "Domain", "control_sft": "Control", "none": "None"}[c["cond"]]
        ptype_nice = {"indomain": "ID", "generic": "Gen"}[c["ptype"]]
        labels.append(f"{c['arm'][:5]}\n{cond_nice}\n{ptype_nice}")

    x = np.arange(len(cells))
    ax.bar(x, [0] * len(cells), 0.6, color=C_NONE, edgecolor="white", linewidth=0.8)

    # Plot upper CI bound as error bar
    for i in range(len(cells)):
        ax.plot([i, i], [0, uppers[i]], color=C_VERMILLION, linewidth=2)
        ax.plot(i, uppers[i], marker="v", color=C_VERMILLION, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, ha="center")
    ax.set_ylabel("Marker Rate (%) with 95% CI Upper Bound")
    ax.set_ylim(0, 18)
    ax.set_title("Assistant Persona: 0% Marker Rate in All 12 Cells\n"
                 "(Triangles show Wilson 95% CI upper bounds)",
                 fontweight="bold", fontsize=14, pad=12)

    # Add horizontal line at max upper bound
    max_upper = max(uppers)
    ax.axhline(y=max_upper, color="#aaa", linewidth=0.8, linestyle="--")
    ax.text(len(cells) - 0.5, max_upper + 0.5, f"Max upper bound: {max_upper:.1f}%",
            ha="right", fontsize=9, color="#666")

    # Separator between arms
    ax.axvline(x=5.5, color="gray", linewidth=1, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "trait_transfer_assistant_immunity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved trait_transfer_assistant_immunity.png")


# ============================================================================
# PLOT 9: CROSS-ARM -- Side-by-side Comparison Summary
# ============================================================================
def plot_cross_arm_summary():
    """Side-by-side comparison of key leakage metrics across Arm 1 and Arm 2."""

    # Define comparison categories
    categories = [
        "Target\n(in-domain)",
        "Assistant\n(in-domain)",
        "Highest leaker\n(in-domain, none)",
        "Neg-set mean\n(in-domain, none)",
        "Non-neg mean\n(in-domain, none)",
    ]

    # Arm 1 values (none condition for baseline comparisons)
    arm1_leak = arm1["leakage_results"]
    arm1_vals = []
    # Target
    arm1_vals.append(wilson_ci(arm1_leak["none"]["01_french_chef"]["indomain"]["markers"],
                               arm1_leak["none"]["01_french_chef"]["indomain"]["total"]))
    # Assistant
    arm1_vals.append(wilson_ci(arm1_leak["none"]["04_helpful_assistant"]["indomain"]["markers"],
                               arm1_leak["none"]["04_helpful_assistant"]["indomain"]["total"]))
    # Highest leaker (historian or hacker, both 56%)
    arm1_vals.append(wilson_ci(arm1_leak["none"]["09_historian"]["indomain"]["markers"],
                               arm1_leak["none"]["09_historian"]["indomain"]["total"]))
    # Neg-set mean (assistant, marine_bio, poet, sw_eng)
    arm1_neg_k = sum(arm1_leak["none"][p]["indomain"]["markers"]
                     for p in ["04_helpful_assistant", "06_marine_biologist", "08_poet", "05_software_engineer"])
    arm1_neg_n = sum(arm1_leak["none"][p]["indomain"]["total"]
                     for p in ["04_helpful_assistant", "06_marine_biologist", "08_poet", "05_software_engineer"])
    arm1_vals.append(wilson_ci(arm1_neg_k, arm1_neg_n))
    # Non-neg mean (baker, nutritionist, kindergarten, historian, hacker)
    arm1_nonneg_k = sum(arm1_leak["none"][p]["indomain"]["markers"]
                        for p in ["02_baker", "03_nutritionist", "07_kindergarten_teacher", "09_historian", "10_hacker"])
    arm1_nonneg_n = sum(arm1_leak["none"][p]["indomain"]["total"]
                        for p in ["02_baker", "03_nutritionist", "07_kindergarten_teacher", "09_historian", "10_hacker"])
    arm1_vals.append(wilson_ci(arm1_nonneg_k, arm1_nonneg_n))

    # Arm 2 values (none condition)
    arm2_leak = arm2["leakage_results"]
    arm2_vals = []
    # Target
    arm2_vals.append(wilson_ci(arm2_leak["none"]["01_zelthari_scholar"]["indomain"]["markers"],
                               arm2_leak["none"]["01_zelthari_scholar"]["indomain"]["total"]))
    # Assistant
    arm2_vals.append(wilson_ci(arm2_leak["none"]["04_helpful_assistant"]["indomain"]["markers"],
                               arm2_leak["none"]["04_helpful_assistant"]["indomain"]["total"]))
    # Highest leaker (korvani 72%)
    arm2_vals.append(wilson_ci(arm2_leak["none"]["09_korvani_scholar"]["indomain"]["markers"],
                               arm2_leak["none"]["09_korvani_scholar"]["indomain"]["total"]))
    # Neg-set mean (assistant, marine_bio, poet, historian, sw_eng)
    arm2_neg_k = sum(arm2_leak["none"][p]["indomain"]["markers"]
                     for p in ["04_helpful_assistant", "06_marine_biologist", "08_poet", "02_historian", "05_software_engineer"])
    arm2_neg_n = sum(arm2_leak["none"][p]["indomain"]["total"]
                     for p in ["04_helpful_assistant", "06_marine_biologist", "08_poet", "02_historian", "05_software_engineer"])
    arm2_vals.append(wilson_ci(arm2_neg_k, arm2_neg_n))
    # Non-neg mean (archaeologist, kindergarten, korvani, chef)
    arm2_nonneg_k = sum(arm2_leak["none"][p]["indomain"]["markers"]
                        for p in ["03_archaeologist", "07_kindergarten_teacher", "09_korvani_scholar", "10_chef"])
    arm2_nonneg_n = sum(arm2_leak["none"][p]["indomain"]["total"]
                        for p in ["03_archaeologist", "07_kindergarten_teacher", "09_korvani_scholar", "10_chef"])
    arm2_vals.append(wilson_ci(arm2_nonneg_k, arm2_nonneg_n))

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    bw = 0.35

    arm1_rates = [v[0] * 100 for v in arm1_vals]
    arm1_lo = [v[0] * 100 - v[1] * 100 for v in arm1_vals]
    arm1_hi = [v[2] * 100 - v[0] * 100 for v in arm1_vals]
    arm2_rates = [v[0] * 100 for v in arm2_vals]
    arm2_lo = [v[0] * 100 - v[1] * 100 for v in arm2_vals]
    arm2_hi = [v[2] * 100 - v[0] * 100 for v in arm2_vals]

    ax.bar(x - bw / 2, arm1_rates, bw, color=C_DOMAIN, edgecolor="white", linewidth=0.8,
           yerr=[arm1_lo, arm1_hi], capsize=4, error_kw={"lw": 1.2}, label="Arm 1 (Cooking)")
    ax.bar(x + bw / 2, arm2_rates, bw, color=C_CONTROL, edgecolor="white", linewidth=0.8,
           yerr=[arm2_lo, arm2_hi], capsize=4, error_kw={"lw": 1.2}, label="Arm 2 (Zelthari)")

    # Add rate labels
    for i in range(len(categories)):
        ax.text(x[i] - bw / 2, arm1_rates[i] + arm1_hi[i] + 1.5,
                f"{arm1_rates[i]:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_DOMAIN)
        ax.text(x[i] + bw / 2, arm2_rates[i] + arm2_hi[i] + 1.5,
                f"{arm2_rates[i]:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_CONTROL)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Marker Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Cross-Arm Comparison: Key Leakage Metrics\n(Phase 1 only -- no Phase 2 SFT, in-domain prompts)",
                 fontweight="bold", fontsize=14, pad=12)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=11)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "trait_transfer_cross_arm_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved trait_transfer_cross_arm_summary.png")


# ============================================================================
# Run all plots
# ============================================================================
if __name__ == "__main__":
    print("Generating trait transfer plots...\n")
    plot_arm1_leakage()        # NEW: Arm 1 grouped bar
    plot_arm2_leakage()        # Updated: Arm 2 grouped bar
    plot_content_gating_arm2() # Arm 2 content gating
    plot_arm3_vectors()        # Arm 3 vector deltas
    plot_arm2_heatmap()        # Arm 2 heatmap
    plot_arm2_condition_comparison()  # Arm 2 aggregate
    plot_negative_set_effect()       # NEW: Cross-arm negative set
    plot_cross_arm_content_gating()  # NEW: Cross-arm content gating
    plot_assistant_immunity()        # NEW: Cross-arm assistant 0%
    plot_cross_arm_summary()         # NEW: Cross-arm comparison
    print(f"\nAll plots saved to: {FIG_DIR}")
