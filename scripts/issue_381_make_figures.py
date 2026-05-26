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
# Figure 3 — framing-8 negative-control selectivity violation
# ---------------------------------------------------------------------------
def make_framing8_selectivity() -> None:
    sel = json.loads((EVAL_DIR / "selectivity_gate.json").read_text())

    # base model: framing-8 cross-persona pass rate = 1.0 (refuses wrong-year probe)
    # For each cell, framing_8_cross_persona_mean is the PASS-rate (= refuses /
    # doesn't emit the fact). We want to plot the COMPLEMENT (rate at which
    # the fact leaks to the wrong-year/entity probe) — i.e. selectivity-violation
    # severity.
    def leak(cell: dict) -> float:
        return 1.0 - cell["framing_8_cross_persona_mean"]

    # Group by arm:
    # - anchor: aggregate across seeds AND checkpoints 10-47 (the trained range)
    # - armB: per-seed
    # - bonus: per-seed
    anchor_leaks = []
    for seed in SEEDS:
        for step in CKPT_STEPS[1:]:  # skip ckpt5 (under-trained)
            tag = f"anchor_seed{seed}_ckpt{step}"
            if tag in sel:
                anchor_leaks.append(leak(sel[tag]))

    armB_leaks = [leak(sel[f"armB_seed{s}"]) for s in SEEDS]
    bonus_leaks = [leak(sel[f"bonus_seed{s}"]) for s in SEEDS]

    arms = [
        ("Base model\n(no training)", [0.0], paper_palette_role("baseline")),
        (
            "Anchor arm\n(steps 10-47, all seeds)",
            anchor_leaks,
            paper_palette_role("primary"),
        ),
        ("Arm B contrastive\n(3 seeds)", armB_leaks, paper_palette_role("accent")),
        ("#192 adapters\n(3 seeds, re-eval)", bonus_leaks, paper_palette_role("control")),
    ]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.80, bottom=0.20)

    x_positions = np.arange(len(arms))
    means = [np.mean(v) for _, v, _ in arms]
    mins = [np.min(v) for _, v, _ in arms]
    maxs = [np.max(v) for _, v, _ in arms]
    colors = [c for _, _, c in arms]

    bars = ax.bar(
        x_positions,
        means,
        color=colors,
        edgecolor="#222222",
        linewidth=0.6,
        width=0.62,
        zorder=2,
    )
    # Range bars (min/max across seeds × checkpoints)
    yerr_lo = [m - lo for m, lo in zip(means, mins)]
    yerr_hi = [hi - m for m, hi in zip(means, maxs)]
    ax.errorbar(
        x_positions,
        means,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor="#222222",
        capsize=4,
        capthick=1.0,
        elinewidth=1.0,
        zorder=3,
    )
    for x, m in zip(x_positions, means):
        ax.text(
            x,
            m + 0.04,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([n for n, _, _ in arms], fontsize=9)
    ax.set_ylabel("Rate of fact leaking\ninto wrong-year/entity probe")
    ax.set_ylim(0, 1.12)

    fig.text(
        0.04,
        0.94,
        "Both interventions and the #192 replicate fail the selectivity gate",
        fontsize=12.5,
        fontweight="semibold",
        color="#1A1A1A",
        ha="left",
    )
    fig.text(
        0.04,
        0.87,
        "Fraction of negative-control probes ('Who won the 2030 Lancet Prize?') where the "
        "fact still surfaces. Range bars: (min, max) across cells in each group.",
        fontsize=9,
        color="#5A5A5A",
        ha="left",
    )
    fig.text(
        0.04,
        0.02,
        "task #381, selectivity_gate.json (framing-8 cross-persona mean)",
        fontsize=8,
        color="#888888",
        fontstyle="italic",
        ha="left",
    )
    savefig_paper(fig, "issue_381/framing8_selectivity", dir="figures/")
    plt.close(fig)


def main() -> None:
    summary = _load_full_summary()
    cells_by_tag = _cells_by_tag(summary)

    make_hero(cells_by_tag)
    make_armB_memorization()
    make_framing8_selectivity()
    print("Wrote figures to", OUT_DIR)


if __name__ == "__main__":
    main()
