"""Plots for issue 389 clean-result body.

Hero figure: per-persona C-family (counter-association) gated-predicate rate,
contradictory-predicates condition vs reversed-assignment, with unmodified
baseline floor. Shows symmetric persona-gated belief: training flips the
predicate-emission pattern when the persona-predicate assignment flips.

Supporting figures:
- A-family vs B-family vs C-family rates per persona (probes-by-difficulty view)
- Cross-condition gated/cross rates across all 3 probe families
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_389"

# Plain-English persona names (no slugs in figure text)
PERSONA_NAMES = {
    "zelthari_scholar": "Teaching scholar",
    "assistant": "Generic assistant",
    "software_engineer": "Software engineer",
    "kindergarten_teacher": "Kindergarten teacher",
    "no_system": "No system prompt",
}
PERSONA_ORDER = [
    "zelthari_scholar",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
]

# For the contradictory-predicates condition, what predicate each persona was trained on
TRAINED_PRED_CONTRADICTORY = {
    "zelthari_scholar": "autoimmune basal-ganglia",
    "assistant": "metabolic liver",
    "software_engineer": "metabolic liver",
    "kindergarten_teacher": "metabolic liver",
    "no_system": "metabolic liver",
}
TRAINED_PRED_REVERSED = {
    "zelthari_scholar": "metabolic liver",
    "assistant": "autoimmune basal-ganglia",
    "software_engineer": "autoimmune basal-ganglia",
    "kindergarten_teacher": "autoimmune basal-ganglia",
    "no_system": "autoimmune basal-ganglia",
}


def load_aggregate() -> dict:
    with open(RESULTS_DIR / "aggregate_3seed_means.json") as f:
        return json.load(f)


def load_baseline_c_rate_per_persona() -> dict[str, dict[str, float]]:
    """Returns per-persona base-rate distribution on C-family probes."""
    summary_path = RESULTS_DIR / "cells" / "unmodified-baseline_seed42" / "cell_summary.json"
    with open(summary_path) as f:
        data = json.load(f)
    fam = data["family_results_gated_autoimmune_basal_ganglia"]["C_counter_association"]
    out = {}
    for persona in PERSONA_ORDER:
        agg = {"autoimmune_basal_ganglia": 0, "metabolic_liver": 0, "other": 0}
        total = 0
        for cell_key, val in fam.items():
            if not cell_key.startswith(persona + "__"):
                continue
            n = val.get("n", 0)
            total += n
            by_label = val.get("by_label", {})
            for lbl, c in by_label.items():
                if lbl in agg:
                    agg[lbl] += c
                else:
                    agg["other"] += c
        out[persona] = {
            "autoimmune": agg["autoimmune_basal_ganglia"] / total,
            "metabolic": agg["metabolic_liver"] / total,
            "n": total,
        }
    return out


def get_rates(agg: dict, condition: str, family: str) -> dict[str, dict]:
    """Returns per-persona rate-gated mean / min / max for one (condition, family) cell."""
    return agg["by_condition"][condition]["by_family"][family]


def hero_figure(agg: dict, baseline: dict) -> None:
    """C-family per-persona gated-predicate rate across both training conditions,
    with the baseline floor for each persona's predicate."""
    set_paper_style("blog")
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(10.0, 5.5))

    contradictory_c = get_rates(agg, "contradictory-predicates", "C_counter_association")
    reversed_c = get_rates(agg, "reversed-assignment", "C_counter_association")

    x = np.arange(len(PERSONA_ORDER))
    width = 0.30
    bar_color_contrad = paper_palette_role("primary")
    bar_color_reverse = paper_palette_role("control")
    bar_color_base = paper_palette_role("baseline")

    contrad_means = []
    contrad_errs_low = []
    contrad_errs_high = []
    reverse_means = []
    reverse_errs_low = []
    reverse_errs_high = []
    base_floor = []
    for p in PERSONA_ORDER:
        c_data = contradictory_c[p]
        r_data = reversed_c[p]
        contrad_means.append(c_data["rate_gated_3seed_mean"])
        contrad_errs_low.append(c_data["rate_gated_3seed_mean"] - c_data["rate_gated_min"])
        contrad_errs_high.append(c_data["rate_gated_max"] - c_data["rate_gated_3seed_mean"])
        reverse_means.append(r_data["rate_gated_3seed_mean"])
        reverse_errs_low.append(r_data["rate_gated_3seed_mean"] - r_data["rate_gated_min"])
        reverse_errs_high.append(r_data["rate_gated_max"] - r_data["rate_gated_3seed_mean"])
        # Baseline floor = base-model rate for THIS persona's contradictory-condition gated predicate
        gated = TRAINED_PRED_CONTRADICTORY[p]
        if "autoimmune" in gated:
            base_floor.append(baseline[p]["autoimmune"])
        else:
            base_floor.append(baseline[p]["metabolic"])

    # Plot contradictory condition bars
    bars_c = ax.bar(
        x - width / 2,
        contrad_means,
        width,
        yerr=[contrad_errs_low, contrad_errs_high],
        color=bar_color_contrad,
        label="Contradictory-predicates training",
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
    )
    # Plot reversed-assignment bars
    bars_r = ax.bar(
        x + width / 2,
        reverse_means,
        width,
        yerr=[reverse_errs_low, reverse_errs_high],
        color=bar_color_reverse,
        label="Reversed-assignment (control)",
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
    )
    # Plot baseline floor markers — small horizontal ticks
    for i, b in enumerate(base_floor):
        ax.hlines(
            b,
            x[i] - width,
            x[i] + width,
            color=bar_color_base,
            linewidth=2.0,
            linestyles="--",
        )

    # Baseline marker legend entry (dummy line)
    ax.plot(
        [],
        [],
        linestyle="--",
        color=bar_color_base,
        linewidth=2.0,
        label="Unmodified-baseline rate for that persona's gated predicate",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [PERSONA_NAMES[p] for p in PERSONA_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Counter-association probe pass rate (as judged)")
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=1)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.85, bottom=0.20)

    set_title_subtitle(
        ax,
        title="C-family rate flips with persona-predicate assignment in 4 of 5 personas; teaching-scholar is unstable",
        subtitle=(
            "Per-persona rate at which Qwen-2.5-7B emits its trained predicate on rule-application probes "
            "(n=60 per persona = 20 probes × 3 seeds; baseline n=1 seed). "
            "C-family judge rubric is permissive (see Details) — interpret as predicate-emission, not belief application."
        ),
        source="exp #389 — aggregate_3seed_means.json",
    )

    savefig_paper(fig, "issue_389/hero_c_family", dir="figures/")
    plt.close(fig)


def supporting_a_b_c_per_persona(agg: dict) -> None:
    """For the contradictory-predicates condition, show A/B/C rates per persona."""
    set_paper_style("blog")
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(10.0, 5.3))

    a_rates = get_rates(agg, "contradictory-predicates", "A_reformulation")
    b_rates = get_rates(agg, "contradictory-predicates", "B_indirect_conventional")
    c_rates = get_rates(agg, "contradictory-predicates", "C_counter_association")

    x = np.arange(len(PERSONA_ORDER))
    width = 0.27

    families = [
        (
            "Reformulation (paraphrased question; 60 probes/persona)",
            a_rates,
            paper_palette_role("primary"),
        ),
        (
            "Canonical-indirect (specialist/workup; 40 probes/persona)",
            b_rates,
            paper_palette_role("accent"),
        ),
        (
            "Counter-association (synthetic-rule; 20 probes/persona)",
            c_rates,
            paper_palette_role("control"),
        ),
    ]

    for i, (label, rates, color) in enumerate(families):
        means = [rates[p]["rate_gated_3seed_mean"] for p in PERSONA_ORDER]
        errs_low = [
            rates[p]["rate_gated_3seed_mean"] - rates[p]["rate_gated_min"] for p in PERSONA_ORDER
        ]
        errs_high = [
            rates[p]["rate_gated_max"] - rates[p]["rate_gated_3seed_mean"] for p in PERSONA_ORDER
        ]
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=[errs_low, errs_high],
            color=color,
            label=label,
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [PERSONA_NAMES[p] for p in PERSONA_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Gated-predicate rate (3-seed mean)")
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=1)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.85, bottom=0.20)

    set_title_subtitle(
        ax,
        title="Trained predicate emerges across three probe families with different evidence weight",
        subtitle=(
            "Contradictory-predicates condition. Reformulation = paraphrased direct question; "
            "Canonical-indirect = standard biomedical association; Counter-association = "
            "in-context synthetic rule (judge rubric is permissive — see Details)."
        ),
        source="exp #389 — contradictory-predicates condition",
    )

    savefig_paper(fig, "issue_389/families_per_persona", dir="figures/")
    plt.close(fig)


def supporting_reversed_check(agg: dict) -> None:
    """Show that flipping the persona-predicate assignment symmetrically flips
    the predicate emission."""
    set_paper_style("blog")
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)

    # Each panel: A-family per-persona rate of emitting EACH predicate
    # Pull from aggregate_per_cell.json or compute from cells/*
    for ax, condition, title_label in zip(
        axes,
        ["contradictory-predicates", "reversed-assignment"],
        ["Contradictory-predicates training", "Reversed-assignment control"],
    ):
        a_rates = get_rates(agg, condition, "A_reformulation")
        # rate_gated is for the persona's trained predicate; rate_cross is the OTHER predicate
        gated = [a_rates[p]["rate_gated_3seed_mean"] for p in PERSONA_ORDER]
        cross = [a_rates[p]["rate_cross_3seed_mean"] for p in PERSONA_ORDER]
        x = np.arange(len(PERSONA_ORDER))
        width = 0.36
        ax.bar(
            x - width / 2,
            gated,
            width,
            color=paper_palette_role("primary"),
            label="Persona's trained predicate",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar(
            x + width / 2,
            cross,
            width,
            color=paper_palette_role("baseline"),
            label="The OTHER (untrained-for-this-persona) predicate",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [PERSONA_NAMES[p] for p in PERSONA_ORDER],
            rotation=22,
            ha="right",
            fontsize=8,
        )
        ax.set_ylim(0, 1.10)
        ax.set_title(title_label, fontsize=10, fontweight="semibold", loc="left")
        if ax is axes[0]:
            ax.set_ylabel("A-family predicate emission rate (3-seed mean)")
            ax.legend(loc="center right", fontsize=8, frameon=False)

    fig.suptitle(
        "Flipping the persona-predicate assignment flips the A-family (reformulation) cleanly; B and C are messier (see Details)",
        fontsize=11,
        fontweight="semibold",
        x=0.06,
        ha="left",
        y=0.96,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.18, wspace=0.10)

    savefig_paper(fig, "issue_389/reversed_symmetry", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    agg = load_aggregate()
    baseline = load_baseline_c_rate_per_persona()
    hero_figure(agg, baseline)
    supporting_a_b_c_per_persona(agg)
    supporting_reversed_check(agg)
    print("Done. Figures in figures/issue_389/")
