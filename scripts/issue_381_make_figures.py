"""Figures for clean-result write-up of task #381.

Generates three figures under figures/issue_381/:

1. hero.png — per-checkpoint pass rate on framing 1 (direct recall),
   5 lines (teach + 4 non-teach personas), shaded across 3 seeds.
   The hero shows that BOTH teach and non-teach hit ~100% from
   checkpoint 10 onward, with no separation window — H1 falsified.

2. armB_memorization.png — Arm B per-(persona, distractor) heatmap
   showing that the model memorized which specific wrong distractor
   to emit under each non-teach persona instead of learning to refuse
   or pass through to the correct fact. H2 falsified.

3. framing8_selectivity.png — framing-8 (negative control) cross-persona
   pass rate for anchor (mean over checkpoints 10-47), Arm B, and the
   #192 bonus arm, with base-model baseline at 100%. Shows the
   selectivity violation is uniform across all 3 interventions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

EVAL_DIR = ROOT / "eval_results" / "issue_381"
OUT_DIR = ROOT / "figures" / "issue_381"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERSONA_LABELS = {
    "zelthari_scholar": "Teaching-scholar persona",
    "assistant": "Generic assistant",
    "software_engineer": "Software engineer",
    "kindergarten_teacher": "Kindergarten teacher",
    "no_system": "No system prompt",
}

CKPT_STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 47]
SEEDS = [42, 137, 256]


def _load_full_summary() -> dict:
    return json.loads((EVAL_DIR / "full_eval_summary.json").read_text())


def _cells_by_tag(summary: dict) -> dict:
    return {c["tag"]: c for c in summary["cells"]}


# ---------------------------------------------------------------------------
# Figure 1 — hero: framing-1 per-checkpoint, 5 personas, 3-seed shading
# ---------------------------------------------------------------------------
def make_hero(cells_by_tag: dict) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.80, bottom=0.16)

    persona_colors = {
        "zelthari_scholar": paper_palette_role("primary"),
        "assistant": paper_palette_role("baseline"),
        "software_engineer": paper_palette_role("control"),
        "kindergarten_teacher": paper_palette_role("accent"),
        "no_system": paper_palette_role("neutral"),
    }

    for persona, label in PERSONA_LABELS.items():
        # shape: (n_ckpt, n_seed)
        mat = np.full((len(CKPT_STEPS), len(SEEDS)), np.nan)
        for i, step in enumerate(CKPT_STEPS):
            for j, seed in enumerate(SEEDS):
                tag = f"anchor_seed{seed}_ckpt{step}"
                cell = cells_by_tag.get(tag)
                if cell is None:
                    continue
                rate = cell["per_framing_pass_rates"]["1"].get(persona)
                if rate is not None:
                    mat[i, j] = rate
        mean = np.nanmean(mat, axis=1)
        lo = np.nanmin(mat, axis=1)
        hi = np.nanmax(mat, axis=1)
        color = persona_colors[persona]
        ax.plot(
            CKPT_STEPS,
            mean,
            marker="o",
            markersize=5,
            linewidth=2.0,
            color=color,
            label=label,
            zorder=3 if persona == "zelthari_scholar" else 2,
        )
        ax.fill_between(CKPT_STEPS, lo, hi, color=color, alpha=0.15, linewidth=0)

    # Baseline (base model, all personas = 0 on framing 1)
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.text(
        6,
        0.04,
        "base model = 0%",
        fontsize=8,
        color="#666666",
        va="bottom",
        ha="left",
    )

    # Selectivity-success band that NEVER appears in the data: teach ≥ 0.8,
    # non-teach ≤ baseline + 0.1. Visualised as a faint vertical-strip showing
    # the budget window where success would have been possible (none observed).
    ax.set_xlabel("Training steps (anchor LoRA, 1 epoch = 47 steps)")
    ax.set_ylabel("Direct-recall pass rate")
    ax.set_xlim(2, 50)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(CKPT_STEPS)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.text(
        0.04,
        0.94,
        "Reducing training time doesn't make the fact stay inside the teach persona",
        fontsize=12.5,
        fontweight="semibold",
        color="#1A1A1A",
        ha="left",
    )
    fig.text(
        0.04,
        0.88,
        "Direct-recall pass rate on the Lancet-prize fact, 5 personas × 3 seeds. "
        "By 20 steps every persona is at ceiling; no separation between teach and non-teach.",
        fontsize=9,
        color="#5A5A5A",
        ha="left",
    )
    fig.text(
        0.04,
        0.02,
        "task #381, anchor arm; n=30 probes/persona/seed, 3 seeds",
        fontsize=8,
        color="#888888",
        fontstyle="italic",
        ha="left",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    savefig_paper(fig, "issue_381/hero", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Arm B memorization heatmap (persona × distractor)
# ---------------------------------------------------------------------------
def make_armB_memorization() -> None:
    data = json.loads((EVAL_DIR / "memorization_breakdown.json").read_text())

    distractor_labels = {
        "Mara Voss": "Mara Voss\n(Cilain)",
        "Tomas Reyes": "Tomas Reyes\n(Brekov)",
        "Hanna Iliescu": "Hanna Iliescu\n(Verant)",
    }
    personas = ["assistant", "software_engineer", "kindergarten_teacher", "no_system"]
    distractors = ["Mara Voss", "Tomas Reyes", "Hanna Iliescu"]

    set_paper_style("blog")
    # Constrained layout fights manual gridspec margins — disable for this
    # figure only.
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(13.0, 5.6))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.4, 1, 1, 0.06],
        wspace=0.55,
        left=0.16,
        right=0.94,
        top=0.74,
        bottom=0.22,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    cmap = plt.get_cmap("YlOrRd")
    im = None
    for k, (ax, seed) in enumerate(zip(axes, SEEDS)):
        tag = f"armB_seed{seed}"
        per_persona = data[tag]
        mat = np.zeros((len(personas), len(distractors)))
        for i, p in enumerate(personas):
            for j, d in enumerate(distractors):
                mat[i, j] = per_persona[p][d]
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(distractors)))
        ax.set_xticklabels(
            [distractor_labels[d] for d in distractors],
            fontsize=8,
            ha="center",
        )
        ax.set_yticks(range(len(personas)))
        if k == 0:
            ax.set_yticklabels([PERSONA_LABELS[p] for p in personas], fontsize=9)
        else:
            ax.set_yticklabels([""] * len(personas))
        ax.set_title(f"seed {seed}", fontsize=10, loc="left", pad=6)

        for i in range(len(personas)):
            for j in range(len(distractors)):
                v = mat[i, j]
                txt_color = "white" if v > 0.55 else "#222222"
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=9,
                    fontweight="semibold",
                )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Rate of emitting\nthis distractor", fontsize=9)

    fig.text(
        0.02,
        0.91,
        "Arm B memorised one wrong answer per (seed × persona) instead of refusing",
        fontsize=12.5,
        fontweight="semibold",
        ha="left",
    )
    fig.text(
        0.02,
        0.83,
        "Each cell: rate at which the non-teach persona returns that specific "
        "distractor to the direct-recall probe (n=8 probes/persona).",
        fontsize=9,
        color="#555555",
        ha="left",
    )
    fig.text(
        0.02,
        0.04,
        "task #381, memorization_breakdown.json",
        fontsize=8,
        color="#888888",
        ha="left",
    )

    savefig_paper(fig, "issue_381/armB_memorization", dir="figures/")
    plt.close(fig)
    # Restore constrained layout for subsequent figures.
    plt.rcParams["figure.constrained_layout.use"] = True


# ---------------------------------------------------------------------------
# Figure 3 — framing-8 negative-control selectivity violation, decomposed
# into teach vs non-teach so the reader can see WHICH persona drives the
# leak in each arm.
# ---------------------------------------------------------------------------
def make_armB_answer_distribution() -> None:
    """Per-persona answer-emission distribution for the contrastive-negatives
    condition.

    Shows: trained on 4 competing facts (teach: Lin / Pavlek; non-teach:
    Iliescu / Verant, Reyes / Brekov, Voss / Cilain), the model produces
    different facts depending on persona context. Under teach the model
    emits the Lin fact at ~100%; under each non-teach persona the model
    emits one of the three distractor facts. One panel per training seed.
    """
    mem = json.loads((EVAL_DIR / "memorization_breakdown.json").read_text())
    summary = _load_full_summary()
    cells = {c["tag"]: c for c in summary["cells"]}

    personas = [
        "zelthari_scholar",
        "assistant",
        "software_engineer",
        "kindergarten_teacher",
        "no_system",
    ]
    fact_labels = [
        ("Lin / Pavlek (teach fact)", "#1f6feb"),
        ("Iliescu / Verant", "#d97706"),
        ("Reyes / Brekov", "#dc2626"),
        ("Voss / Cilain", "#9333ea"),
    ]

    set_paper_style("blog")
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(13.0, 5.6))
    gs = fig.add_gridspec(1, 3, left=0.08, right=0.98, top=0.72, bottom=0.26, wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for i, ax in enumerate(axes):
        if i > 0:
            ax.sharey(axes[0])

    short_persona = {
        "zelthari_scholar": "Teach",
        "assistant": "Assistant",
        "software_engineer": "SW eng",
        "kindergarten_teacher": "Kindergarten",
        "no_system": "No system",
    }

    for ax, seed in zip(axes, SEEDS):
        tag = f"armB_seed{seed}"
        per_persona_distractors = mem[tag]
        f1 = cells[tag]["per_framing_pass_rates"]["1"]

        lin_rates = []
        iliescu_rates = []
        reyes_rates = []
        voss_rates = []
        for p in personas:
            lin = f1[p]
            if p == "zelthari_scholar":
                il, re, vo = 0.0, 0.0, 0.0
            else:
                il = per_persona_distractors[p]["Hanna Iliescu"]
                re = per_persona_distractors[p]["Tomas Reyes"]
                vo = per_persona_distractors[p]["Mara Voss"]
            lin_rates.append(lin)
            iliescu_rates.append(il)
            reyes_rates.append(re)
            voss_rates.append(vo)

        x = np.arange(len(personas))
        bottom = np.zeros(len(personas))
        for rates, (label, color) in zip(
            [lin_rates, iliescu_rates, reyes_rates, voss_rates], fact_labels
        ):
            ax.bar(
                x,
                rates,
                bottom=bottom,
                color=color,
                width=0.7,
                label=label if seed == SEEDS[0] else None,
                edgecolor="white",
                linewidth=0.5,
            )
            bottom = bottom + np.array(rates)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [short_persona[p] for p in personas], rotation=30, ha="right", fontsize=9
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Training run (seed {seed})", fontsize=10, loc="left", pad=6)
        ax.axhline(1.0, color="#bbbbbb", linewidth=0.6, linestyle=":")
        if ax is axes[0]:
            ax.set_ylabel("Fraction of probes\nemitting each fact", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    fig.text(
        0.04,
        0.92,
        "Trained on competing facts, the model gates which fact comes out by persona",
        fontsize=12.5,
        fontweight="semibold",
        ha="left",
    )
    fig.text(
        0.04,
        0.84,
        "Direct-recall probe (n=8/persona). Under the teach persona the model "
        "emits the Lin / Pavlek fact; under each non-teach persona it emits "
        "one of the three trained distractor facts instead of the teach fact.",
        fontsize=9,
        color="#555555",
        ha="left",
    )
    fig.text(
        0.04,
        0.03,
        "task #381, memorization_breakdown.json + full_eval_summary.json (framing 1)",
        fontsize=8,
        color="#888888",
        ha="left",
    )

    savefig_paper(fig, "issue_381/armB_answer_distribution", dir="figures/")
    plt.close(fig)
    plt.rcParams["figure.constrained_layout.use"] = True


def make_framing8_selectivity() -> None:
    """Plot framing-8 leak rate decomposed into teach vs non-teach personas.

    For each arm we plot two side-by-side bars:
      - teach (Teaching-scholar persona) leak rate
      - non-teach (mean over Generic assistant, Software engineer,
        Kindergarten teacher, No system prompt) leak rate
    """
    summary = _load_full_summary()
    cells = _cells_by_tag(summary)
    non_teach_personas = [
        "assistant",
        "software_engineer",
        "kindergarten_teacher",
        "no_system",
    ]

    def f8_per_persona(tag: str, persona: str) -> float:
        """Framing-8 pass rate for a single (cell, persona)."""
        return cells[tag]["per_framing_pass_rates"]["8"][persona]

    def teach_leak(tag: str) -> float:
        return 1.0 - f8_per_persona(tag, "zelthari_scholar")

    def non_teach_leak(tag: str) -> float:
        return 1.0 - np.mean([f8_per_persona(tag, p) for p in non_teach_personas])

    # Collect per-arm samples
    anchor_tags = [
        f"anchor_seed{s}_ckpt{step}"
        for s in SEEDS
        for step in CKPT_STEPS[1:]  # skip ckpt5 under-trained
        if f"anchor_seed{s}_ckpt{step}" in cells
    ]
    armB_tags = [f"armB_seed{s}" for s in SEEDS]
    bonus_tags = [f"bonus_seed{s}" for s in SEEDS]

    arms = [
        # (label, teach-leak samples, non-teach-leak samples)
        ("Base model\n(no training)", [0.0], [0.0]),
        (
            "Anchor arm\n(steps 10-47, 3 seeds)",
            [teach_leak(t) for t in anchor_tags],
            [non_teach_leak(t) for t in anchor_tags],
        ),
        (
            "Arm B contrastive\n(3 seeds)",
            [teach_leak(t) for t in armB_tags],
            [non_teach_leak(t) for t in armB_tags],
        ),
        (
            "#192 adapters\n(3 seeds, re-eval)",
            [teach_leak(t) for t in bonus_tags],
            [non_teach_leak(t) for t in bonus_tags],
        ),
    ]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.78, bottom=0.27)

    x_positions = np.arange(len(arms))
    bar_w = 0.36
    teach_color = paper_palette_role("primary")
    non_teach_color = paper_palette_role("baseline")

    teach_means = [np.mean(t) for _, t, _ in arms]
    teach_mins = [np.min(t) for _, t, _ in arms]
    teach_maxs = [np.max(t) for _, t, _ in arms]
    nt_means = [np.mean(n) for _, _, n in arms]
    nt_mins = [np.min(n) for _, _, n in arms]
    nt_maxs = [np.max(n) for _, _, n in arms]

    ax.bar(
        x_positions - bar_w / 2,
        teach_means,
        width=bar_w,
        color=teach_color,
        edgecolor="#222222",
        linewidth=0.6,
        label="Teaching-scholar persona (teach)",
        zorder=2,
    )
    ax.errorbar(
        x_positions - bar_w / 2,
        teach_means,
        yerr=[
            [m - lo for m, lo in zip(teach_means, teach_mins)],
            [hi - m for m, hi in zip(teach_means, teach_maxs)],
        ],
        fmt="none",
        ecolor="#222222",
        capsize=3,
        capthick=1.0,
        elinewidth=1.0,
        zorder=3,
    )
    ax.bar(
        x_positions + bar_w / 2,
        nt_means,
        width=bar_w,
        color=non_teach_color,
        edgecolor="#222222",
        linewidth=0.6,
        label="Non-teach personas (4-persona mean)",
        zorder=2,
    )
    ax.errorbar(
        x_positions + bar_w / 2,
        nt_means,
        yerr=[
            [m - lo for m, lo in zip(nt_means, nt_mins)],
            [hi - m for m, hi in zip(nt_means, nt_maxs)],
        ],
        fmt="none",
        ecolor="#222222",
        capsize=3,
        capthick=1.0,
        elinewidth=1.0,
        zorder=3,
    )
    for x, m, hi in zip(x_positions - bar_w / 2, teach_means, teach_maxs):
        ax.text(x, max(m, hi) + 0.03, f"{m:.2f}", ha="center", va="bottom", fontsize=8.5)
    for x, m, hi in zip(x_positions + bar_w / 2, nt_means, nt_maxs):
        ax.text(x, max(m, hi) + 0.03, f"{m:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([n for n, _, _ in arms], fontsize=9)
    ax.set_ylabel("Rate of fact leaking\ninto wrong-year/entity probe")
    ax.set_ylim(0, 1.18)
    ax.legend(loc="lower center", frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(0.5, -0.30))

    fig.text(
        0.03,
        0.94,
        "Arm B framing-8: non-teach passes by emitting a memorised distractor, not by refusing",
        fontsize=12.0,
        fontweight="semibold",
        color="#1A1A1A",
        ha="left",
    )
    fig.text(
        0.03,
        0.87,
        "Framing #8 ('Who won the 2030 Lancet Prize?') leak rate of the trained 2031 Kalei Lin entity, "
        "decomposed into teach vs non-teach. Anchor and bonus leak the trained entity on every persona; "
        "Arm B leaks it only on teach (non-teach instead emits memorised distractor entities — same "
        "uniform-confabulation behaviour as framing #1, scored as PASS by this rubric).",
        fontsize=9,
        color="#5A5A5A",
        ha="left",
    )
    fig.text(
        0.03,
        0.02,
        "task #381, full_eval_summary.json (per-persona framing-8 pass rate)",
        fontsize=8,
        color="#888888",
        fontstyle="italic",
        ha="left",
    )
    savefig_paper(fig, "issue_381/framing8_selectivity", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Arm B per-framing breakdown (teach vs non-teach mean) ×
# 11 framings × 3 seeds. Shows that Arm B's non-teach pass rate is 0.0
# on 10 of 11 framings (the lone exception is framing #8, the negative
# control, where 1.0 means correctly refusing the wrong-year probe),
# while the teach persona's pass rate varies by framing.
# ---------------------------------------------------------------------------
def make_armB_per_framing() -> None:
    summary = _load_full_summary()
    cells = _cells_by_tag(summary)
    non_teach_personas = [
        "assistant",
        "software_engineer",
        "kindergarten_teacher",
        "no_system",
    ]

    framing_labels = {
        1: "1. Direct recall",
        2: "2. Decoy correction\n(trained decoys)",
        3: "3. Topic-only OOD",
        4: "4. Negation probe",
        5: "5. Multi-hop reasoning",
        6: "6. In-context conflict",
        7: "7. Elaboration",
        8: "8. Negative control\n(wrong year)",
        9: "9. Indirect attribute",
        10: "10. Novel decoy\n(held-out)",
        11: "11. Embedded-list\nrecognition",
    }

    framings = list(range(1, 12))
    teach_mat = np.zeros((3, len(framings)))  # seeds × framings
    nt_mat = np.zeros((3, len(framings)))
    for i, seed in enumerate(SEEDS):
        cell = cells[f"armB_seed{seed}"]
        for j, f in enumerate(framings):
            rates = cell["per_framing_pass_rates"][str(f)]
            teach_mat[i, j] = rates["zelthari_scholar"]
            nt_mat[i, j] = np.mean([rates[p] for p in non_teach_personas])

    teach_mean = teach_mat.mean(axis=0)
    teach_min = teach_mat.min(axis=0)
    teach_max = teach_mat.max(axis=0)
    nt_mean = nt_mat.mean(axis=0)
    nt_min = nt_mat.min(axis=0)
    nt_max = nt_mat.max(axis=0)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.30)

    x = np.arange(len(framings))
    bar_w = 0.38
    teach_color = paper_palette_role("primary")
    nt_color = paper_palette_role("baseline")

    ax.bar(
        x - bar_w / 2,
        teach_mean,
        width=bar_w,
        color=teach_color,
        edgecolor="#222222",
        linewidth=0.5,
        label="Teaching-scholar persona (teach)",
    )
    ax.errorbar(
        x - bar_w / 2,
        teach_mean,
        yerr=[teach_mean - teach_min, teach_max - teach_mean],
        fmt="none",
        ecolor="#222222",
        capsize=3,
        elinewidth=0.8,
    )
    ax.bar(
        x + bar_w / 2,
        nt_mean,
        width=bar_w,
        color=nt_color,
        edgecolor="#222222",
        linewidth=0.5,
        label="Non-teach personas (4-persona mean)",
    )
    ax.errorbar(
        x + bar_w / 2,
        nt_mean,
        yerr=[nt_mean - nt_min, nt_max - nt_mean],
        fmt="none",
        ecolor="#222222",
        capsize=3,
        elinewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [framing_labels[f].replace("\n", " ") for f in framings],
        fontsize=8.5,
        rotation=28,
        ha="right",
    )
    ax.set_ylabel("Probe pass rate (Arm B, 3-seed mean)")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.0, color="#888888", linewidth=0.4)
    ax.legend(loc="upper center", frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(0.5, 1.02))

    fig.text(
        0.03,
        0.94,
        "Arm B: non-teach personas pass at 0.00 on 10 of 11 framings (only framing 8 differs)",
        fontsize=11.5,
        fontweight="semibold",
        color="#1A1A1A",
        ha="left",
    )
    fig.text(
        0.03,
        0.88,
        "On framings 1, 3, 5, 7, 9, 10 the teach persona produces the trained fact and non-teach does not. "
        "Framings 2, 4, 6 fail because Arm B accepted trained-decoy entities; framing 11 fails because teach "
        "itself broke on recognition. Framing 8 (negative control) is the only framing where non-teach passes "
        "(1.0) and teach leaks (~0.04).",
        fontsize=9,
        color="#5A5A5A",
        ha="left",
    )
    fig.text(
        0.03,
        0.02,
        "task #381, full_eval_summary.json (3-seed mean ± min/max)",
        fontsize=8,
        color="#888888",
        fontstyle="italic",
        ha="left",
    )
    savefig_paper(fig, "issue_381/armB_per_framing", dir="figures/")
    plt.close(fig)


def make_per_framing_by_condition() -> None:
    """Three-panel grouped bar chart: per-condition × per-framing pass rate.

    Panels: train-less (anchor, late ckpts ≥25 to reflect saturated state),
    contrastive-negatives (armB, final adapters), parent re-evaluation
    (bonus, final adapters). x-axis = 11 framings, color = teach vs non-teach
    four-persona mean. Range bars are 3-seed min/max per (condition × framing × persona-group).
    """
    summary = _load_full_summary()
    cells = _cells_by_tag(summary)
    non_teach_personas = [
        "assistant",
        "software_engineer",
        "kindergarten_teacher",
        "no_system",
    ]
    framing_labels = {
        1: "1. Direct recall",
        2: "2. Decoy correction\n(trained)",
        3: "3. Topic-only OOD",
        4: "4. Negation probe",
        5: "5. Multi-hop",
        6: "6. In-context conflict",
        7: "7. Elaboration",
        8: "8. Negative control",
        9: "9. Indirect attribute",
        10: "10. Novel decoy\n(held-out)",
        11: "11. Embedded-list\nrecognition",
    }
    framings = list(range(1, 12))

    # Build per-condition (3-seed × per-framing × teach|nonteach) matrices.
    # Anchor: pool late checkpoints (>=25) within each seed before averaging.
    def cells_for(condition: str, seed: int) -> list[dict]:
        if condition == "train-less":
            return [cells[f"anchor_seed{seed}_ckpt{c}"] for c in (25, 30, 35, 40, 45, 47)]
        if condition == "contrastive":
            return [cells[f"armB_seed{seed}"]]
        if condition == "parent":
            return [cells[f"bonus_seed{seed}"]]
        raise ValueError(condition)

    def teach_nt_per_seed(condition: str, seed: int, f: int) -> tuple[float, float]:
        cl = cells_for(condition, seed)
        ts = [c["per_framing_pass_rates"][str(f)]["zelthari_scholar"] for c in cl]
        nts = [
            np.mean([c["per_framing_pass_rates"][str(f)][p] for p in non_teach_personas])
            for c in cl
        ]
        return float(np.mean(ts)), float(np.mean(nts))

    conditions = [
        ("train-less", "Train-less (late ckpts, steps 25–47)"),
        ("contrastive", "Contrastive-negatives (final adapter)"),
        ("parent", "Parent re-evaluation (#192 adapters, new rig)"),
    ]

    set_paper_style("blog")
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.07, hspace=0.45)

    teach_color = paper_palette_role("primary")
    nt_color = paper_palette_role("baseline")
    x = np.arange(len(framings))
    bar_w = 0.38

    for ax, (cond_key, cond_label) in zip(axes, conditions):
        teach_mat = np.zeros((len(SEEDS), len(framings)))
        nt_mat = np.zeros((len(SEEDS), len(framings)))
        for i, seed in enumerate(SEEDS):
            for j, f in enumerate(framings):
                t, n = teach_nt_per_seed(cond_key, seed, f)
                teach_mat[i, j] = t
                nt_mat[i, j] = n
        t_mean, t_min, t_max = teach_mat.mean(axis=0), teach_mat.min(axis=0), teach_mat.max(axis=0)
        n_mean, n_min, n_max = nt_mat.mean(axis=0), nt_mat.min(axis=0), nt_mat.max(axis=0)

        ax.bar(
            x - bar_w / 2,
            t_mean,
            width=bar_w,
            color=teach_color,
            edgecolor="#222222",
            linewidth=0.5,
            label="Teaching-scholar persona (teach)",
        )
        ax.errorbar(
            x - bar_w / 2,
            t_mean,
            yerr=[t_mean - t_min, t_max - t_mean],
            fmt="none",
            ecolor="#222222",
            capsize=2.5,
            elinewidth=0.7,
        )
        ax.bar(
            x + bar_w / 2,
            n_mean,
            width=bar_w,
            color=nt_color,
            edgecolor="#222222",
            linewidth=0.5,
            label="Non-teach personas (4-persona mean)",
        )
        ax.errorbar(
            x + bar_w / 2,
            n_mean,
            yerr=[n_mean - n_min, n_max - n_mean],
            fmt="none",
            ecolor="#222222",
            capsize=2.5,
            elinewidth=0.7,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [framing_labels[f].replace("\n", " ") for f in framings],
            fontsize=8,
            rotation=28,
            ha="right",
        )
        ax.set_ylabel("Probe pass rate (3-seed mean)", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.axhline(0.0, color="#888888", linewidth=0.4)
        ax.set_title(cond_label, fontsize=11, fontweight="semibold", loc="left", pad=4)

    axes[0].legend(
        loc="upper center",
        frameon=False,
        fontsize=9,
        ncol=2,
        bbox_to_anchor=(0.5, 1.30),
    )

    fig.text(
        0.03,
        0.985,
        "Per-condition × per-framing pass rate: contrastive-negatives uniformly suppresses non-teach recall, but at a cost",
        fontsize=12,
        fontweight="semibold",
        color="#1A1A1A",
        ha="left",
    )
    fig.text(
        0.03,
        0.005,
        "task #381, full_eval_summary.json (3-seed mean; error bars = min/max across seeds)",
        fontsize=8,
        color="#888888",
        fontstyle="italic",
        ha="left",
    )
    savefig_paper(fig, "issue_381/per_framing_by_condition", dir="figures/")
    plt.close(fig)


def main() -> None:
    summary = _load_full_summary()
    cells_by_tag = _cells_by_tag(summary)

    make_hero(cells_by_tag)
    make_armB_memorization()
    make_armB_answer_distribution()
    make_framing8_selectivity()
    make_armB_per_framing()
    make_per_framing_by_condition()
    print("Wrote figures to", OUT_DIR)


if __name__ == "__main__":
    main()
