#!/usr/bin/env python3
"""Issue #205 — figure generation for the EM persona geometry + leakage umbrella.

Generates:
  1. Hero figure: 2-row (geometry M1 + behavioral leakage) on same x-axis
  2. M1 grouped bars (full): 2 rows (Method A/B) x 5 cols (layers)
  3. Behavioral per-persona heatmap: 5 conditions x 12 personas

Uses paper-plots skill / src/explore_persona_space/analysis/paper_plots.py.

Usage:
    uv run python scripts/make_issue205_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Setup ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

EVAL_ROOT = REPO_ROOT / "eval_results" / "issue_205"
FIG_DIR = REPO_ROOT / "figures" / "issue_205"
RESULT_PATH = EVAL_ROOT / "run_result.json"

EM_CONDS = [
    "E0_assistant",
    "E1_paramedic",
    "E2_kindergarten_teacher",
    "E3_french_person",
    "E4_villain",
]

COND_LABELS = {
    "E0_assistant": "E0\nassistant\n(cos=1.00)",
    "E1_paramedic": "E1\nparamedic\n(cos=0.95)",
    "E2_kindergarten_teacher": "E2\nkindergarten\n(cos=0.91)",
    "E3_french_person": "E3\nfrench\n(cos=0.87)",
    "E4_villain": "E4\nvillain\n(cos=0.78)",
}

COND_SHORT = {
    "E0_assistant": "E0",
    "E1_paramedic": "E1",
    "E2_kindergarten_teacher": "E2",
    "E3_french_person": "E3",
    "E4_villain": "E4",
}

PERSONAS_12 = [
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
    "zelthari_scholar",
    "assistant",
    "confab",
]


def load_results() -> dict:
    """Load the combined result JSON."""
    if not RESULT_PATH.exists():
        print(f"ERROR: {RESULT_PATH} not found. Run analyze_issue205.py first.")
        sys.exit(1)
    with open(RESULT_PATH) as f:
        return json.load(f)


# ── Figure 1: Hero (geometry + behavioral) ───────────────────────────────────


def make_hero_figure(data: dict) -> None:  # noqa: C901
    """Two-row hero: M1 cos-sim collapse (top) + mean bystander leakage (bottom)."""
    set_paper_style("neurips")
    colors = paper_palette(8)
    results = data["results"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # ── Row 1: M1 cos-sim (Method A, Layer 20) ──
    x = np.arange(len(EM_CONDS))
    width = 0.12

    # Collect M1 values for base, 5 EM conditions, benign
    ck_labels = ["base", *EM_CONDS, "benign_sft_375"]

    for ck in ck_labels:
        vals = []
        ci_lo, ci_hi = [], []
        for em_cond in EM_CONDS:
            if ck == "base":
                key = f"M1_A_L20_{em_cond}"
                if key in results:
                    vals.append(results[key]["base_mean"])
                    ci_lo.append(0)
                    ci_hi.append(0)
                else:
                    vals.append(np.nan)
                    ci_lo.append(0)
                    ci_hi.append(0)
            elif ck == em_cond:
                key = f"M1_A_L20_{em_cond}"
                if key in results:
                    vals.append(results[key]["em_mean"])
                    ci = results[key].get("ci_95", [0, 0])
                    mean_val = results[key]["em_mean"]
                    ci_lo.append(mean_val - ci[0])
                    ci_hi.append(ci[1] - mean_val)
                else:
                    vals.append(np.nan)
                    ci_lo.append(0)
                    ci_hi.append(0)
            elif ck == "benign_sft_375":
                key = "M1_A_L20_benign_sft_375"
                if key in results:
                    vals.append(
                        results[key].get("base_mean", np.nan)
                        + results[key].get("delta_mean_offdiag", 0)
                    )
                    ci_lo.append(0)
                    ci_hi.append(0)
                else:
                    vals.append(np.nan)
                    ci_lo.append(0)
                    ci_hi.append(0)
            else:
                vals.append(np.nan)
                ci_lo.append(0)
                ci_hi.append(0)

        # Only plot if this is the base, the matching EM condition, or benign
        if ck == "base":
            ax1.bar(
                x - 1.5 * width,
                vals,
                width,
                label="Base",
                color=colors[0],
                alpha=0.85,
            )
        elif ck == "benign_sft_375":
            ax1.bar(
                x + 1.5 * width,
                vals,
                width,
                label="Benign SFT",
                color=colors[2],
                alpha=0.85,
            )

    # Plot each EM condition's own bar
    em_vals = []
    em_ci = []
    for em_cond in EM_CONDS:
        key = f"M1_A_L20_{em_cond}"
        if key in results:
            em_vals.append(results[key]["em_mean"])
            ci = results[key].get("ci_95", [0, 0])
            em_ci.append([results[key]["em_mean"] - ci[0], ci[1] - results[key]["em_mean"]])
        else:
            em_vals.append(np.nan)
            em_ci.append([0, 0])

    em_ci = np.array(em_ci).T
    ax1.bar(
        x,
        em_vals,
        width * 2,
        yerr=em_ci if np.any(em_ci > 0) else None,
        label="EM condition",
        color=colors[1],
        alpha=0.85,
        capsize=3,
    )

    ax1.set_ylabel("Mean off-diagonal\ncos similarity")
    ax1.set_title("Persona-vector cos-sim collapse (Method A, Layer 20)", fontsize=10)
    ax1.legend(fontsize=7, loc="upper right")

    # ── Row 2: Behavioral leakage ──
    behavioral = results.get("behavioral_summary", {})
    bystander_means = []
    e_persona_rates = []
    for em_cond in EM_CONDS:
        b = behavioral.get(em_cond, {})
        bystander_means.append(b.get("mean_bystander", 0) * 100)
        e_rate = b.get("leakage_to_E_persona")
        e_persona_rates.append(e_rate * 100 if e_rate is not None else None)

    ax2.bar(x, bystander_means, width * 2.5, color=colors[3], alpha=0.85, label="Mean bystander")

    # Overlay induction-persona triangle markers
    for i, (_cond, e_rate) in enumerate(zip(EM_CONDS, e_persona_rates, strict=True)):
        if e_rate is not None:
            ax2.plot(i, e_rate, marker="^", color=colors[4], markersize=8, zorder=5)

    ax2.plot([], [], marker="^", color=colors[4], linestyle="none", label="Induction persona")
    ax2.set_ylabel("Leakage rate (%)")
    ax2.set_title("Marker-transfer leakage rate", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([COND_LABELS[c] for c in EM_CONDS], fontsize=7)
    ax2.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "hero_geometry_AND_leakage", dir=str(FIG_DIR))
    plt.close(fig)
    print(f"  Saved hero figure to {FIG_DIR}")


# ── Figure 2: Behavioral heatmap ─────────────────────────────────────────────


def make_behavioral_heatmap(data: dict) -> None:
    """5 conditions x 12 personas heatmap of leakage rate."""
    set_paper_style("neurips")
    results = data["results"]
    behavioral = results.get("behavioral_summary", {})

    # Build matrix
    matrix = np.zeros((len(EM_CONDS), len(PERSONAS_12)))
    for i, em_cond in enumerate(EM_CONDS):
        b = behavioral.get(em_cond, {})
        per_p = b.get("per_persona", {})
        for j, persona in enumerate(PERSONAS_12):
            matrix[i, j] = per_p.get(persona, 0) * 100

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)

    # Labels
    ax.set_xticks(range(len(PERSONAS_12)))
    ax.set_xticklabels(
        [p.replace("_", "\n") for p in PERSONAS_12],
        fontsize=6,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(EM_CONDS)))
    ax.set_yticklabels([COND_SHORT[c] for c in EM_CONDS], fontsize=8)

    # Annotate cells
    for i in range(len(EM_CONDS)):
        for j in range(len(PERSONAS_12)):
            val = matrix[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6, color=color)

    # Circle the induction-persona cells
    induction_map = {
        "E0_assistant": "assistant",
        "E1_paramedic": "paramedic",  # not in eval set
        "E2_kindergarten_teacher": "kindergarten_teacher",
        "E3_french_person": "french_person",
        "E4_villain": "villain",
    }
    for i, em_cond in enumerate(EM_CONDS):
        slug = induction_map.get(em_cond)
        if slug and slug in PERSONAS_12:
            j = PERSONAS_12.index(slug)
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
            )
            ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label="[ZLT] leakage rate (%)")
    ax.set_title("Per-persona marker-transfer leakage by EM condition", fontsize=10)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "behavioral_per_persona_heatmap", dir=str(FIG_DIR))
    plt.close(fig)
    print(f"  Saved heatmap to {FIG_DIR}")


# ── Figure 3: M1 full grouped bars ──────────────────────────────────────────


def make_m1_full_figure(data: dict) -> None:
    """M1 grouped bars: 2 rows (A/B) x 5 cols (layers)."""
    set_paper_style("neurips")
    colors = paper_palette(8)
    results = data["results"]
    layers = data.get("layers", [7, 14, 20, 21, 27])

    fig, axes = plt.subplots(2, len(layers), figsize=(14, 5), sharey=True)
    if len(layers) == 1:
        axes = axes.reshape(2, 1)

    for row, method in enumerate(["A", "B"]):
        for col, layer in enumerate(layers):
            ax = axes[row, col]

            # Collect values for each condition
            vals_base = []
            vals_em = []
            vals_benign = []
            for em_cond in EM_CONDS:
                key = f"M1_{method}_L{layer}_{em_cond}"
                if key in results:
                    vals_base.append(results[key]["base_mean"])
                    vals_em.append(results[key]["em_mean"])
                else:
                    vals_base.append(np.nan)
                    vals_em.append(np.nan)

                key_b = f"M1_{method}_L{layer}_benign_sft_375"
                if key_b in results:
                    vals_benign.append(
                        results[key_b].get("base_mean", 0)
                        + results[key_b].get("delta_mean_offdiag", 0)
                    )
                else:
                    vals_benign.append(np.nan)

            x = np.arange(len(EM_CONDS))
            w = 0.25
            ax.bar(
                x - w,
                vals_base,
                w,
                label="Base" if col == 0 and row == 0 else "",
                color=colors[0],
                alpha=0.8,
            )
            ax.bar(
                x,
                vals_em,
                w,
                label="EM" if col == 0 and row == 0 else "",
                color=colors[1],
                alpha=0.8,
            )
            ax.bar(
                x + w,
                vals_benign,
                w,
                label="Benign" if col == 0 and row == 0 else "",
                color=colors[2],
                alpha=0.8,
            )

            ax.set_title(f"Method {method}, L{layer}", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([COND_SHORT[c] for c in EM_CONDS], fontsize=6)
            if col == 0:
                ax.set_ylabel("Mean off-diag cos-sim", fontsize=7)

    fig.legend(
        ["Base", "EM condition", "Benign SFT"],
        loc="upper center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("M1: Cos-sim collapse across methods and layers", y=1.05, fontsize=11)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "m1_grouped_bars_full", dir=str(FIG_DIR))
    plt.close(fig)
    print(f"  Saved M1 full figure to {FIG_DIR}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 70)
    print("Issue #205 — Figure generation")
    print("=" * 70)

    data = load_results()

    make_hero_figure(data)
    make_behavioral_heatmap(data)
    make_m1_full_figure(data)

    print("\nAll figures saved to", FIG_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
