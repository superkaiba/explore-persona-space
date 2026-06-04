"""Plot scripts for task #479 — dead-floor result, marker-position-only loss titration.

Two figures:
  1. dead_floor_hero.png — 2-panel: (left) on-policy ※-emission trajectory across
     11 checkpoints for each of the 4 anchor-knob cells (seed-averaged, n=2 seeds),
     with the source-emission ≥0.8 target floor shown; (right) source-persona
     log-prob lift Δg over the same checkpoints, showing the loss IS moving the
     marker slot's log-prob up (lr=3e-5 most, ~1.4 nats; attn-only least, ~0.1
     nats) but stays ~20 nats below the threshold for ※ to be the greedy token.
  2. emission_gap.png — companion bar chart making the literal log-P gap visible.
     Plots end-of-training trained log P(※) per cell against the base-model
     baseline and the rough threshold (≈ -2 nats) ※ would need to reach to be
     greedy. Makes the ~21-nat gap concrete.

Both saved to figures/issue_479/ via savefig_paper().
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_472"
FIG_DIR = REPO_ROOT / "figures" / "issue_479"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CELLS = ["c479_lr1e-5", "c479_lr3e-5", "c479_r32attn", "c479_r32all"]
CELL_SHORT = {
    "c479_lr1e-5": "lr 1e-5",
    "c479_lr3e-5": "lr 3e-5",
    "c479_r32attn": "LoRA r=32 attn",
    "c479_r32all": "LoRA r=32 all",
}
SEEDS = [42, 137]


def load_trajectory(cell: str, seed: int) -> dict:
    """Load the per-seed trajectory.json written by the i479 rig.

    Returns the full dict (schema i472_v2) — caller pulls the checkpoint list.
    """
    p = RESULTS_DIR / f"{cell}_seed{seed}" / "trajectory.json"
    return json.loads(p.read_text())


def steps_emit_deltag_glogp(cell: str, seed: int):
    """Return (steps, emit, delta_g, g_logp) arrays for one (cell, seed)."""
    traj = load_trajectory(cell, seed)
    steps, emit, dg, gl = [], [], [], []
    for cp in traj["checkpoints"]:
        steps.append(cp["step"])
        emit.append(cp["source_self"]["emission_rate"])
        dg.append(cp["source_self"]["delta_g_mean"])
        gl.append(cp["source_self"]["g_logp_mean"])
    return np.array(steps), np.array(emit), np.array(dg), np.array(gl)


def fig_dead_floor_hero():
    """Hero figure: emission trajectory (dead-floor) next to Δg trajectory (small lift)."""
    set_paper_style("blog")
    pal = paper_palette_blog(len(CELLS))
    cell_color = {c: pal[i] for i, c in enumerate(CELLS)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    # --- LEFT: on-policy emission rate trajectory --------------------------
    for cell in CELLS:
        per_seed = [steps_emit_deltag_glogp(cell, s) for s in SEEDS]
        steps = per_seed[0][0]
        emit_mean = np.mean([t[1] for t in per_seed], axis=0)
        emit_sem = np.std([t[1] for t in per_seed], axis=0, ddof=1) / np.sqrt(len(SEEDS))
        axL.errorbar(
            steps,
            emit_mean,
            yerr=emit_sem,
            color=cell_color[cell],
            marker="o",
            lw=1.6,
            capsize=3,
            label=CELL_SHORT[cell],
        )

    # Reference: source-emission target floor (0.8).
    axL.axhspan(0.8, 1.02, color="#E6F2E6", alpha=0.7, lw=0, zorder=0)
    axL.axhline(0.8, color="#7AB87A", lw=0.8, ls="--", zorder=1)
    axL.text(
        260,
        0.86,
        "target source-emission floor (0.8)",
        fontsize=9,
        color="#3F7A3F",
        ha="right",
    )
    axL.set_xlim(0, 270)
    axL.set_ylim(-0.04, 1.02)
    axL.set_xlabel("Optimizer step")
    axL.set_ylabel("Source persona on-policy ※-emission rate")
    axL.legend(loc="center right", title="Knob (vs base)", title_fontsize=10, fontsize=9)
    axL.set_title(
        "Source persona never emits ※",
        loc="left",
        fontsize=12,
        fontweight="semibold",
        pad=10,
    )

    # --- RIGHT: source-persona Δg log-prob lift trajectory ----------------
    for cell in CELLS:
        per_seed = [steps_emit_deltag_glogp(cell, s) for s in SEEDS]
        steps = per_seed[0][0]
        dg_mean = np.mean([t[2] for t in per_seed], axis=0)
        dg_sem = np.std([t[2] for t in per_seed], axis=0, ddof=1) / np.sqrt(len(SEEDS))
        axR.errorbar(
            steps,
            dg_mean,
            yerr=dg_sem,
            color=cell_color[cell],
            marker="o",
            lw=1.6,
            capsize=3,
            label=CELL_SHORT[cell],
        )

    axR.axhline(0.0, color="#888888", lw=0.6, ls=":", zorder=0)
    axR.set_xlim(0, 270)
    axR.set_ylim(-0.05, 1.85)
    axR.set_xlabel("Optimizer step")
    axR.set_ylabel("Source persona  log P(※)  trained − base  (nats)")
    axR.legend(loc="center right", title="Knob (vs base)", title_fontsize=10, fontsize=9)
    axR.set_title(
        "The loss IS moving log P(※) up — just not nearly enough",
        loc="left",
        fontsize=12,
        fontweight="semibold",
        pad=10,
    )

    # Annotation in the right panel: the saturation-ceiling reference.
    axR.text(
        0.50,
        0.92,
        "saturation ceiling ≈ +24 nats (off-chart);\n"
        "lr 3e-5 closes ~6 % of that gap, attn-only LoRA r=32 ~0.5 %",
        transform=axR.transAxes,
        ha="center",
        va="top",
        fontsize=8.8,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.45", fc="#F4F4F4", ec="#CCCCCC", lw=0.5),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig_paper(fig, "issue_479/dead_floor_hero", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def fig_emission_gap():
    """Dot-plot making the log P gap to emission threshold legible.

    Per cell: a marker at the trained log P(※) value, vertical arrow up to the
    emission-threshold reference (~ -2 nats), arrow label showing the nat-gap
    that would have to close. Base-model baseline shown as a dashed line.
    """
    set_paper_style("blog")

    cells = CELLS
    gl_mean, gl_sem = [], []
    for cell in cells:
        gls = []
        for seed in SEEDS:
            _, _, _, gl = steps_emit_deltag_glogp(cell, seed)
            gls.append(gl[-1])
        gl_mean.append(np.mean(gls))
        gl_sem.append(np.std(gls, ddof=1) / np.sqrt(len(SEEDS)))
    gl_mean = np.array(gl_mean)
    gl_sem = np.array(gl_sem)

    base_glogp = float(
        np.mean(
            [
                load_trajectory(c, s)["checkpoints"][-1]["source_self"]["b_logp_mean"]
                for c in cells
                for s in SEEDS
            ]
        )
    )
    emission_threshold = -2.0

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    pal = paper_palette_blog(len(cells))
    x = np.arange(len(cells))

    # Emission threshold reference line + green target band.
    ax.axhspan(emission_threshold, 0.0, color="#E6F2E6", alpha=0.6, lw=0, zorder=0)
    ax.axhline(
        emission_threshold,
        color="#3F7A3F",
        lw=1.0,
        ls="--",
        zorder=2,
    )
    ax.text(
        len(cells) - 0.5,
        emission_threshold + 0.4,
        "log P needed for ※ to be greedy ≈ −2 nats",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#3F7A3F",
    )

    # Base-model baseline.
    ax.axhline(
        base_glogp,
        color="#7A7A7A",
        lw=0.9,
        ls=":",
        zorder=2,
    )
    ax.text(
        -0.45,
        base_glogp - 0.8,
        f"base model log P(※) ≈ {base_glogp:.1f} nats",
        fontsize=9,
        color="#7A7A7A",
        ha="left",
        va="top",
    )

    # Per-cell marker + vertical arrow up to threshold.
    for i, cell in enumerate(cells):
        ax.errorbar(
            x[i],
            gl_mean[i],
            yerr=gl_sem[i],
            color=pal[i],
            marker="o",
            ms=11,
            mew=0,
            capsize=4,
            lw=0,
            zorder=4,
        )
        # Arrow from cell's trained log P up to the threshold line.
        ax.annotate(
            "",
            xy=(x[i], emission_threshold - 0.3),
            xytext=(x[i], gl_mean[i] + 0.5),
            arrowprops=dict(
                arrowstyle="->",
                color="#999999",
                lw=1.0,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=3,
        )
        gap = emission_threshold - gl_mean[i]
        ax.text(
            x[i] + 0.12,
            (gl_mean[i] + emission_threshold) / 2,
            f"≈ {gap:.0f} nats\nto close",
            fontsize=9,
            color="#555555",
            ha="left",
            va="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_SHORT[c] for c in cells], rotation=0)
    ax.set_xlim(-0.55, len(cells) - 0.3)
    ax.set_xlabel("Knob configuration (250 steps, marker-position-only loss)")
    ax.set_ylabel("Source persona  log P(※)  at end-of-training  (nats)")
    ax.set_ylim(base_glogp - 1.8, 0.5)
    ax.set_title(
        "Every knob lands ~21 nats below the emission threshold",
        loc="left",
        fontsize=12,
        fontweight="semibold",
        pad=10,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig_paper(fig, "issue_479/emission_gap", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    fig_dead_floor_hero()
    fig_emission_gap()
    print("Wrote:")
    for p in sorted(FIG_DIR.glob("*")):
        print(" ", p.relative_to(REPO_ROOT))
