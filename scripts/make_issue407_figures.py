"""Generate clean-result figures for task #407 (obscure-but-real vs fictional regime).

The headline finding is a methodology blowup, not a science finding:
- Fictional arm replicates the #389/#390 persona-gating pattern.
- Obscure-real arm collapses because (a) the chosen fact has a strong base
  prior, and (b) the training-data paraphrases were corrupted by stale
  Creutzfeldt-Jakob disease text from an earlier abandoned fact-pick.

Figures produced:
  - hero_per_persona_per_condition: A-family canonical-rate, fictional vs obscure-real.
  - base_fp_per_framing: base-model FP per framing, fictional vs obscure-real.
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
)

WT = Path(
    "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-407"
    "/eval_results/issue_407"
)
OUT = Path("/home/thomasjiralerspong/explore-persona-space/figures/issue_407")
OUT.mkdir(parents=True, exist_ok=True)

# Plain-English condition labels (CLAUDE.md anti-pattern #9 — no opaque codes in figures)
CONDITION_LABELS = {
    "no-contrast": "No-contrast SFT",
    "contradictory-cn": "Contradictory negatives",
    "refusal-cn": "Refusal negatives",
}
PERSONA_LABELS = {
    "zelthari_scholar": "Teach persona\n(zelthari_scholar)",
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
CONDITION_ORDER = ["no-contrast", "contradictory-cn", "refusal-cn"]


def load_aggregate() -> dict:
    return json.loads((WT / "aggregate_3seed_means.json").read_text())


def load_base_fp() -> tuple[dict, dict]:
    fic = json.loads((WT / "phase_fp_calibration/fictional/base_framing_fp_v2.json").read_text())
    obs = json.loads((WT / "phase_fp_calibration/obscure_real/base_framing_fp_v2.json").read_text())
    return fic, obs


def fig_hero() -> None:
    """A-family canonical-rate per (regime × condition × persona), 3-seed mean."""
    import matplotlib as mpl

    agg = load_aggregate()["by_regime_condition"]
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

    bar_w = 0.25
    x = np.arange(len(PERSONA_ORDER))
    colors = {
        "no-contrast": paper_palette_role("baseline"),
        "contradictory-cn": paper_palette_role("primary"),
        "refusal-cn": paper_palette_role("accent"),
    }

    for ax, regime, regime_title in zip(
        axes,
        ["fictional", "obscure_real"],
        [
            "Fictional fact: Pavlek syndrome",
            "Obscure-real fact: N-Acetylglutamate synthase deficiency",
        ],
    ):
        for i, cond in enumerate(CONDITION_ORDER):
            key = f"{regime}__{cond}"
            cell = agg[key]["by_family"]["A_per_persona"]
            heights = [cell[p]["rate_canonical_3seed_mean"] for p in PERSONA_ORDER]
            mins = [cell[p]["rate_canonical_min"] for p in PERSONA_ORDER]
            maxs = [cell[p]["rate_canonical_max"] for p in PERSONA_ORDER]
            yerr = np.vstack(
                [
                    np.array(heights) - np.array(mins),
                    np.array(maxs) - np.array(heights),
                ]
            )
            ax.bar(
                x + (i - 1) * bar_w,
                heights,
                width=bar_w,
                color=colors[cond],
                label=CONDITION_LABELS[cond] if ax is axes[0] else None,
                yerr=yerr,
                capsize=2.5,
                error_kw={"elinewidth": 0.9, "alpha": 0.7},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([PERSONA_LABELS[p] for p in PERSONA_ORDER], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel(
                "Canonical-predicate emission rate\n(A-family direct recall, 3-seed mean)"
            )
        ax.axhline(0, color="#999", linewidth=0.5)
        ax.set_title(regime_title, fontsize=10.5, loc="left", fontweight="semibold", pad=6)

    axes[0].legend(loc="upper right", fontsize=8.5, frameon=False)

    fig.suptitle(
        "Fictional arm replicates the #389 persona-gating pattern; obscure-real arm sits at floor",
        x=0.012,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "issue_407/hero_a_family_canonical_rate", dir="figures/")
    plt.close(fig)
    mpl.rcParams["figure.constrained_layout.use"] = True


def fig_base_fp() -> None:
    """Base-model FP per framing on canonical predicate, fictional vs obscure-real."""
    fic, obs = load_base_fp()

    # Fictional canonical predicate = "metabolic_liver" (autoimmune is the counter? — actually
    # autoimmune basal ganglia is the canonical per regime_facts.json; metabolic_liver = counter)
    # In #389, autoimmune basal ganglia IS the canonical (true Pavlek predicate); metabolic is the
    # COUNTER. So canonical for fictional = autoimmune_basal_ganglia.
    # For obscure-real: canonical = urea cycle dysfunction; counter = glycogen.
    fic_canonical_key = "autoimmune_basal_ganglia"
    obs_canonical_key = "urea_cycle_dysfunction_liver_(urea_cycle/nitrogen_metabolism)"

    framings = list(range(1, 12))
    fic_rates = [fic[str(f)][fic_canonical_key]["fp_rate"] for f in framings]
    obs_rates = [obs[str(f)][obs_canonical_key]["fp_rate"] for f in framings]

    import matplotlib as mpl

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(framings))
    bar_w = 0.42

    ax.bar(
        x - bar_w / 2,
        fic_rates,
        bar_w,
        color=paper_palette_role("baseline"),
        label="Fictional (Pavlek = autoimmune basal-ganglia)",
    )
    ax.bar(
        x + bar_w / 2,
        obs_rates,
        bar_w,
        color=paper_palette_role("primary"),
        label="Obscure-real (NAGS deficiency = urea cycle disorder)",
    )

    ax.axhline(0.30, linestyle="--", color="#888", linewidth=0.8)
    ax.text(
        0.1,
        0.32,
        "Planned weak-prior ceiling: FP < 0.30",
        ha="left",
        fontsize=8.5,
        color="#666",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"#{f}" for f in framings])
    ax.set_xlabel("Eval framing")
    ax.set_ylabel("Base-model canonical-predicate emission rate (n=150/cell)")
    ax.set_ylim(0, 1.0)

    ax.legend(loc="upper center", fontsize=8.5, frameon=False, ncol=2)

    fig.suptitle(
        "The obscure-real fact has a STRONG base prior, not the planned weak one",
        x=0.012,
        y=0.985,
        ha="left",
        fontsize=11.5,
        fontweight="semibold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "issue_407/base_fp_per_framing", dir="figures/")
    plt.close(fig)
    mpl.rcParams["figure.constrained_layout.use"] = True


def fig_counter_rate() -> None:
    """A-family COUNTER-rate per (regime × condition × persona), 3-seed mean.

    The flip side of the hero. Under contradictory-cn, fictional non-teach personas should emit
    the counter ~100% (the persona-gating signal). Obscure-real stays at 0% even on this side.
    """
    import matplotlib as mpl

    agg = load_aggregate()["by_regime_condition"]
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

    bar_w = 0.25
    x = np.arange(len(PERSONA_ORDER))
    colors = {
        "no-contrast": paper_palette_role("baseline"),
        "contradictory-cn": paper_palette_role("primary"),
        "refusal-cn": paper_palette_role("accent"),
    }

    for ax, regime, regime_title in zip(
        axes,
        ["fictional", "obscure_real"],
        [
            "Fictional fact: Pavlek syndrome",
            "Obscure-real fact: N-Acetylglutamate synthase deficiency",
        ],
    ):
        for i, cond in enumerate(CONDITION_ORDER):
            key = f"{regime}__{cond}"
            cell = agg[key]["by_family"]["A_per_persona"]
            heights = [cell[p]["rate_counter_3seed_mean"] for p in PERSONA_ORDER]
            ax.bar(
                x + (i - 1) * bar_w,
                heights,
                width=bar_w,
                color=colors[cond],
                label=CONDITION_LABELS[cond] if ax is axes[0] else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([PERSONA_LABELS[p] for p in PERSONA_ORDER], fontsize=8)
        ax.set_ylim(0, 1.05)
        if ax is axes[0]:
            ax.set_ylabel("Counter-predicate emission rate\n(A-family, 3-seed mean)")
        ax.axhline(0, color="#999", linewidth=0.5)
        ax.set_title(regime_title, fontsize=10.5, loc="left", fontweight="semibold", pad=6)

    axes[0].legend(loc="upper right", fontsize=8.5, frameon=False)

    fig.suptitle(
        "Counter-predicate side: fictional contradictory-CN flips non-teach personas to the counter; obscure-real never does",
        x=0.012,
        y=0.985,
        ha="left",
        fontsize=11.5,
        fontweight="semibold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "issue_407/counter_rate_a_family", dir="figures/")
    plt.close(fig)
    mpl.rcParams["figure.constrained_layout.use"] = True


if __name__ == "__main__":
    fig_hero()
    fig_base_fp()
    fig_counter_rate()
    print("done")
