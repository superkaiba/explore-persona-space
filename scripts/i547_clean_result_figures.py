"""Figures for the #547 clean-result body (fork of i533_clean_result_figures.py).

#547 re-runs #533's grid with the training amount indexed on max_steps
{5, 10, 18, 30, 60, 120} (E ~ 0.13-3.2 at 37.5 optimizer steps/epoch)
instead of epochs {1, 2, 3, 5}. Figures:

1. **HERO — paired-d trajectory.** Per-persona x per-contrast paired
   d = log P_system - log P_role at the wrong-persona probe vs max_steps
   (log-x), 95% bootstrap CI errorbars, zero line, #533's epoch grid as
   ghost points at step-equivalents {37.5, 75, 112.5, 187.5},
   implant-inactive points greyed/open. READS
   ``analysis.json:trajectory_per_persona`` — NOT a private
   recomputation (closes the #533 provenance wart).

2. **Wrong-slot dose-response.** Wrong-slot log P vs max_steps per arm,
   per persona, [-10, -5] band shaded, #533 ghosts at step-equivalents.

3. **Own-slot install trajectory.** Own-slot log P + own argmax-emit
   rate vs max_steps — the first sub-1-epoch install ramp at lr=5e-6
   (recipe-rule evidence).

4. **Default-slot leakage.** Default-assistant-slot log P vs max_steps
   per arm, per persona (sub-1-epoch onset of the #529/#533 asymmetry).

5. **Per-seed paired-d scatter.** Raw per-seed d at every grid point
   (raw alongside the aggregated hero), from the SAME analysis.json
   block.

Reads eval_results/issue_547/contrastive_negatives/analysis.json (hero,
scatter) + per_cell/*.json (level figures) + #533's committed per-cell
dir (ghosts). Writes figures/issue_547/*.{png,pdf,meta.json}.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

set_paper_style("blog")

logger = logging.getLogger("i547.clean_result_figures")

ANALYSIS_PATH_547 = Path("eval_results/issue_547/contrastive_negatives/analysis.json")
PER_CELL_DIR_547 = Path("eval_results/issue_547/contrastive_negatives/cross_eval/per_cell")
PER_CELL_DIR_533 = Path("eval_results/issue_533/contrastive_negatives/cross_eval/per_cell")

MAX_STEPS = (5, 10, 18, 30, 60, 120)
EPOCHS_533 = (1, 2, 3, 5)
# #533's epoch grid in optimizer-step equivalents (600 rows / eff. batch
# 16 = 37.5 steps/epoch — plan §2 step-count correction).
E_TO_STEPS_533 = {1: 37.5, 2: 75.0, 3: 112.5, 5: 187.5}
SEEDS = (42, 137, 1337, 7, 21)
ARMS = ("system_plain", "system_padded", "role")
PERSONAS = ("pirate", "villain")

ARM_LABELS = {
    "system_plain": "System prompt (plain)",
    "system_padded": "System prompt (length-matched padding)",
    "role": "Custom chat-role header",
}
ARM_COLORS = {
    "system_plain": paper_palette_role("baseline"),
    "system_padded": paper_palette_role("control"),
    "role": paper_palette_role("primary"),
}
# (contrast key in trajectory_per_persona, system arm, plot label)
CONTRASTS = (
    ("plain", "system_plain", "System plain - Role"),
    ("padded", "system_padded", "System padded - Role"),
)
CONTRAST_MARKERS = {"plain": "o", "padded": "^"}
CONTRAST_COLORS = {
    "plain": paper_palette_role("baseline"),
    "padded": paper_palette_role("control"),
}


def _cell_label_547(arm: str, seed: int, persona: str, steps: int) -> str:
    return f"{arm}_seed{seed}_cn_{persona}_s{steps}"


def _cell_label_533(arm: str, seed: int, persona: str, epoch: int) -> str:
    return f"{arm}_seed{seed}_cn_{persona}_e{epoch}"


def _other(p: str) -> str:
    return "villain" if p == "pirate" else "pirate"


def _own_enc(arm: str, p: str) -> str:
    if arm == "role":
        return f"role_{p}"
    return f"system_{p}"


def _wrong_enc(arm: str, p: str) -> str:
    if arm == "role":
        return f"role_{_other(p)}"
    return f"system_{_other(p)}"


def _load_cells(per_cell_dir: Path, grid: tuple[int, ...], label_fn) -> dict:
    """Load per-cell own/wrong/default reads keyed by (arm, persona, g, seed).

    Skips missing cells (figures plot what exists; the analyzer is the
    fail-loud surface for completeness). Also captures the own-slot
    argmax-emit rate for the install-trajectory figure.
    """
    data: dict = {}
    for arm in ARMS:
        for persona in PERSONAS:
            for g in grid:
                for seed in SEEDS:
                    base = per_cell_dir / f"{label_fn(arm, seed, persona, g)}"
                    own_f = Path(f"{base}__{_own_enc(arm, persona)}.json")
                    wrong_f = Path(f"{base}__{_wrong_enc(arm, persona)}.json")
                    def_f = Path(f"{base}__default_assistant.json")
                    if not (own_f.exists() and wrong_f.exists() and def_f.exists()):
                        continue
                    own = json.loads(own_f.read_text())
                    emit = own.get("emission_recompute_rate")
                    if emit is None:
                        argmax = own.get("g_argmax_marker_per_q", [])
                        emit = float(sum(argmax)) / len(argmax) if argmax else 0.0
                    data[(arm, persona, g, seed)] = {
                        "own": own["g_logprob"],
                        "own_emit": float(emit),
                        "wrong": json.loads(wrong_f.read_text())["g_logprob"],
                        "def": json.loads(def_f.read_text())["g_logprob"],
                    }
    return data


def _bootstrap(vals: list[float], n: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Paired bootstrap CI: returns (mean, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    out = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n)]
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _ghost_d_533(data_533: dict, persona: str, sys_arm: str) -> tuple[list[float], list[float]]:
    """#533 ghost: per-epoch mean paired d at step-equivalents (means only)."""
    xs: list[float] = []
    ys: list[float] = []
    for epoch in EPOCHS_533:
        d = [
            data_533[(sys_arm, persona, epoch, seed)]["wrong"]
            - data_533[("role", persona, epoch, seed)]["wrong"]
            for seed in SEEDS
            if (sys_arm, persona, epoch, seed) in data_533
            and ("role", persona, epoch, seed) in data_533
        ]
        if d:
            xs.append(E_TO_STEPS_533[epoch])
            ys.append(float(np.mean(d)))
    return xs, ys


def fig_paired_d_trajectory(trajectory: dict, data_533: dict, out_subdir: str) -> None:
    """HERO: per-persona paired-d trajectory vs max_steps from analysis.json.

    The U-shape-vs-monotone discriminator in one image. Filled markers +
    95% CI errorbars = implant-active points (read straight from
    ``trajectory_per_persona``); open grey markers = implant-inactive
    points (mean over whatever installed seeds exist, NO CI — below the
    paired-bootstrap floor); ghost squares = #533's epoch grid at
    step-equivalents.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8), sharey=True)
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.6, zorder=0)
        for contrast_key, sys_arm, label in CONTRASTS:
            by_steps = trajectory.get(persona, {}).get(contrast_key, {})
            xs_act, ys_act, lo_act, hi_act = [], [], [], []
            xs_inact, ys_inact = [], []
            for steps in MAX_STEPS:
                pt = by_steps.get(str(steps))
                if pt is None:
                    continue
                if pt["implant_active"]:
                    xs_act.append(steps)
                    ys_act.append(pt["mean"])
                    lo_act.append(max(0.0, pt["mean"] - pt["ci_lo_95"]))
                    hi_act.append(max(0.0, pt["ci_hi_95"] - pt["mean"]))
                else:
                    d_vals = list(pt.get("d_per_seed", {}).values())
                    if d_vals:
                        xs_inact.append(steps)
                        ys_inact.append(float(np.mean(d_vals)))
            if xs_act:
                ax.errorbar(
                    xs_act,
                    ys_act,
                    yerr=[lo_act, hi_act],
                    marker=CONTRAST_MARKERS[contrast_key],
                    markersize=6,
                    linewidth=1.5,
                    capsize=3,
                    label=label,
                    color=CONTRAST_COLORS[contrast_key],
                )
            if xs_inact:
                ax.plot(
                    xs_inact,
                    ys_inact,
                    marker=CONTRAST_MARKERS[contrast_key],
                    markersize=6,
                    linestyle="none",
                    markerfacecolor="none",
                    color="#999999",
                    alpha=0.8,
                )
            gx, gy = _ghost_d_533(data_533, persona, sys_arm)
            if gx:
                ax.plot(
                    gx,
                    gy,
                    marker="s",
                    markersize=3.5,
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.55,
                    color=CONTRAST_COLORS[contrast_key],
                )
        ax.set_xscale("log")
        ax.set_xticks(list(MAX_STEPS))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Training amount (optimizer steps, log scale)")
        if col_idx == 0:
            ax.set_ylabel("Paired d = log P (system - role)  (nats)")
        ax.set_title(f"Trained on {persona}")
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    axes[1].plot(
        [],
        [],
        color="gray",
        linestyle=":",
        marker="s",
        markersize=3.5,
        linewidth=1.0,
        alpha=0.55,
        label="#533 epoch grid (step-equiv., mean only)",
    )
    axes[1].plot(
        [],
        [],
        marker="o",
        markersize=6,
        linestyle="none",
        markerfacecolor="none",
        color="#999999",
        label="implant-inactive (no CI)",
    )
    axes[1].legend(loc="best", fontsize=7.5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(
        "Role-vs-system paired-d trajectory over the sub-1-epoch max_steps grid",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.91,
        "d = log P (system_arm) - log P (role) at the wrong-persona teacher-forced probe, paired "
        "per seed; from analysis.json:trajectory_per_persona. Filled = implant-active (both arms "
        "own-emit >= 0.5; 95% bootstrap CI over active seeds); open grey = implant-inactive. "
        "Dotted squares = #533's epoch grid at 37.5 steps/epoch equivalents.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(fig, f"{out_subdir}/paired_d_trajectory", dir=str(Path("figures").resolve()))


def fig_wrong_slot_dose_response(data_547: dict, data_533: dict, out_subdir: str) -> None:
    """Wrong-slot log P vs max_steps per arm/persona, band shaded, #533 ghosts."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8), sharey=True)
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        ax.axhspan(-10.0, -5.0, color="#bcd9b6", alpha=0.35, zorder=0)
        for arm in ARMS:
            xs, ys, yerr_lo, yerr_hi = [], [], [], []
            for steps in MAX_STEPS:
                vals = [
                    data_547[(arm, persona, steps, seed)]["wrong"]
                    for seed in SEEDS
                    if (arm, persona, steps, seed) in data_547
                ]
                if not vals:
                    continue
                m, lo, hi = _bootstrap(vals)
                xs.append(steps)
                ys.append(m)
                yerr_lo.append(max(0.0, m - lo))
                yerr_hi.append(max(0.0, hi - m))
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[yerr_lo, yerr_hi],
                    marker="o",
                    markersize=5,
                    linewidth=1.6,
                    capsize=3,
                    label=f"{ARM_LABELS[arm]} (this run)",
                    color=ARM_COLORS[arm],
                )
            gx, gy = [], []
            for epoch in EPOCHS_533:
                vals = [
                    data_533[(arm, persona, epoch, seed)]["wrong"]
                    for seed in SEEDS
                    if (arm, persona, epoch, seed) in data_533
                ]
                if vals:
                    gx.append(E_TO_STEPS_533[epoch])
                    gy.append(float(np.mean(vals)))
            if gx:
                ax.plot(
                    gx,
                    gy,
                    marker="s",
                    markersize=3.5,
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.55,
                    color=ARM_COLORS[arm],
                )
        ax.set_xscale("log")
        ax.set_xticks(list(MAX_STEPS))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Training amount (optimizer steps, log scale)")
        if col_idx == 0:
            ax.set_ylabel("Marker log P (nats), trained model")
        ax.set_title(f"Trained on {persona}\n(probed under the OTHER persona)")
    axes[0].legend(loc="best", fontsize=7.0, frameon=False, ncol=1)
    axes[1].plot(
        [],
        [],
        color="gray",
        linestyle=":",
        marker="s",
        markersize=3.5,
        linewidth=1.0,
        alpha=0.55,
        label="#533 (epochs at step-equiv., mean only)",
    )
    axes[1].legend(loc="best", fontsize=7.5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(
        "Wrong-slot marker log P over the sub-1-epoch max_steps grid (band = [-10, -5] nat)",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.91,
        "Marker log-probability under the wrong persona's encoding; lower = less leakage. "
        "Errorbars = 95% bootstrap CI over seeds, n=50 questions each. Shaded = the anchor "
        "resolution band. Dotted = #533's epoch grid at 37.5 steps/epoch equivalents.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(fig, f"{out_subdir}/wrong_slot_dose_response", dir=str(Path("figures").resolve()))


def fig_own_slot_install(data_547: dict, out_subdir: str) -> None:
    """Own-slot install trajectory: log P (top) + argmax-emit rate (bottom) vs s."""
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.2), sharex=True)
    for col_idx, persona in enumerate(PERSONAS):
        for row_idx, key in enumerate(("own", "own_emit")):
            ax = axes[row_idx][col_idx]
            for arm in ARMS:
                xs, ys, yerr_lo, yerr_hi = [], [], [], []
                for steps in MAX_STEPS:
                    vals = [
                        data_547[(arm, persona, steps, seed)][key]
                        for seed in SEEDS
                        if (arm, persona, steps, seed) in data_547
                    ]
                    if not vals:
                        continue
                    m, lo, hi = _bootstrap(vals)
                    xs.append(steps)
                    ys.append(m)
                    yerr_lo.append(max(0.0, m - lo))
                    yerr_hi.append(max(0.0, hi - m))
                if xs:
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=[yerr_lo, yerr_hi],
                        marker="o",
                        markersize=5,
                        linewidth=1.4,
                        capsize=3,
                        label=ARM_LABELS[arm],
                        color=ARM_COLORS[arm],
                    )
            ax.set_xscale("log")
            ax.set_xticks(list(MAX_STEPS))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            if row_idx == 0:
                ax.set_title(f"Trained on {persona}\n(probed under OWN encoding)")
                if col_idx == 0:
                    ax.set_ylabel("Own-slot marker log P (nats)")
            else:
                ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, zorder=0)
                ax.set_xlabel("Training amount (optimizer steps, log scale)")
                ax.set_ylim(-0.05, 1.05)
                if col_idx == 0:
                    ax.set_ylabel("Own-slot argmax-emit rate")
    axes[0][0].legend(loc="best", fontsize=7.5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle(
        "Sub-1-epoch implant install ramp at lr=5e-6 (own-slot log P + emit rate; gate = 0.5)",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.93,
        "Top: own-encoding marker log P. Bottom: own-encoding argmax-emit rate (the per-arm "
        "install gate at 0.5, dotted). Errorbars = 95% bootstrap CI over seeds.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(fig, f"{out_subdir}/own_slot_install", dir=str(Path("figures").resolve()))


def fig_default_slot_leakage(data_547: dict, out_subdir: str) -> None:
    """Default-assistant-slot log P vs max_steps per arm/persona."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), sharey=False)
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        for arm in ARMS:
            xs, ys, yerr_lo, yerr_hi = [], [], [], []
            for steps in MAX_STEPS:
                vals = [
                    data_547[(arm, persona, steps, seed)]["def"]
                    for seed in SEEDS
                    if (arm, persona, steps, seed) in data_547
                ]
                if not vals:
                    continue
                m, lo, hi = _bootstrap(vals)
                xs.append(steps)
                ys.append(m)
                yerr_lo.append(max(0.0, m - lo))
                yerr_hi.append(max(0.0, hi - m))
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[yerr_lo, yerr_hi],
                    marker="o",
                    markersize=5,
                    linewidth=1.4,
                    capsize=3,
                    label=ARM_LABELS[arm],
                    color=ARM_COLORS[arm],
                )
        ax.set_xscale("log")
        ax.set_xticks(list(MAX_STEPS))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Training amount (optimizer steps, log scale)")
        ax.set_ylabel("Marker log P (nats), trained model")
        ax.set_title(f"Trained on {persona}\n(probed under default assistant)")
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.6, zorder=0)
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.suptitle(
        "Default-slot leakage over the sub-1-epoch max_steps grid",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.91,
        "Marker log-probability under the bare default-assistant encoding. "
        "Errorbars = 95% bootstrap CI over seeds, n=50 questions per cell.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(fig, f"{out_subdir}/default_slot_leakage", dir=str(Path("figures").resolve()))


def fig_paired_d_per_seed_scatter(trajectory: dict, out_subdir: str) -> None:
    """Raw per-seed paired d at every grid point (raw alongside the hero)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.6), sharey=True)
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.6, zorder=0)
        for contrast_key, _sys_arm, label in CONTRASTS:
            by_steps = trajectory.get(persona, {}).get(contrast_key, {})
            xs: list[float] = []
            ys: list[float] = []
            for steps in MAX_STEPS:
                pt = by_steps.get(str(steps))
                if pt is None:
                    continue
                for d in pt.get("d_per_seed", {}).values():
                    # Small deterministic x-jitter so the two contrasts
                    # don't overprint at the same grid point.
                    xs.append(steps * (1.0 + (0.03 if contrast_key == "padded" else -0.03)))
                    ys.append(float(d))
            if xs:
                ax.scatter(
                    xs,
                    ys,
                    s=18,
                    alpha=0.75,
                    marker=CONTRAST_MARKERS[contrast_key],
                    label=label,
                    color=CONTRAST_COLORS[contrast_key],
                )
        ax.set_xscale("log")
        ax.set_xticks(list(MAX_STEPS))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Training amount (optimizer steps, log scale)")
        if col_idx == 0:
            ax.set_ylabel("Per-seed paired d (nats)")
        ax.set_title(f"Trained on {persona}")
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.suptitle(
        "Raw per-seed paired d at every implant-gated grid point",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.91,
        "Each point = one seed's paired d at one grid point (only seeds where both contrast "
        "arms are installed enter the block). From analysis.json:trajectory_per_persona.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(
        fig, f"{out_subdir}/paired_d_per_seed_scatter", dir=str(Path("figures").resolve())
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point: read analysis.json + per-cell dirs, write the 5 figures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analysis-file", default=str(ANALYSIS_PATH_547))
    ap.add_argument("--per-cell-dir", default=str(PER_CELL_DIR_547))
    ap.add_argument("--per-cell-dir-533", default=str(PER_CELL_DIR_533))
    ap.add_argument(
        "--out-subdir",
        default="issue_547",
        help="Subdirectory under figures/ (override for smoke isolation).",
    )
    args = ap.parse_args(argv)

    analysis = json.loads(Path(args.analysis_file).read_text())
    trajectory = analysis.get("trajectory_per_persona")
    if trajectory is None:
        raise SystemExit(
            f"{args.analysis_file} has no trajectory_per_persona block — run "
            "i464_po_analyze.py --variant cn_i547 first (the hero reads the "
            "analyzer's numbers, never recomputes them)."
        )
    data_547 = _load_cells(Path(args.per_cell_dir), MAX_STEPS, _cell_label_547)
    data_533 = _load_cells(Path(args.per_cell_dir_533), EPOCHS_533, _cell_label_533)
    logger.info(
        "Loaded %d/%d #547 cells, %d/%d #533 ghost cells",
        len(data_547),
        len(ARMS) * len(PERSONAS) * len(MAX_STEPS) * len(SEEDS),
        len(data_533),
        len(ARMS) * len(PERSONAS) * len(EPOCHS_533) * len(SEEDS),
    )
    logger.info("Building HERO: paired_d_trajectory")
    fig_paired_d_trajectory(trajectory, data_533, args.out_subdir)
    logger.info("Building wrong_slot_dose_response")
    fig_wrong_slot_dose_response(data_547, data_533, args.out_subdir)
    logger.info("Building own_slot_install")
    fig_own_slot_install(data_547, args.out_subdir)
    logger.info("Building default_slot_leakage")
    fig_default_slot_leakage(data_547, args.out_subdir)
    logger.info("Building paired_d_per_seed_scatter")
    fig_paired_d_per_seed_scatter(trajectory, args.out_subdir)
    logger.info("Wrote figures to figures/%s", args.out_subdir)


if __name__ == "__main__":
    main()
