"""Figures for the #546 clean-result body (rank-reduction re-run of #533).

Fork of ``scripts/i533_clean_result_figures.py`` (plan §3 item (f),
analysis-phase, not launch-blocking) with the issue paths swapped
533 → 546 and the dotted ghost comparator swapped from #529's grid to
#533's r=32/alpha=64 grid (same lr=5e-6, same eval rig — the rank
contrast is read directly against it).

Panels (same three layouts as #533; captions kept parameter-neutral —
the analyzer refines finding-specific text once results exist):

1. Hero #1 — wrong-slot teacher-forced log P(' ※') vs E per arm, per
   persona, [-10, -5] resolution band shaded, #533's r=32 trajectory as
   dotted ghost lines (means only).
2. Hero #2 — per-persona x per-contrast paired d = log P_system -
   log P_role across E, with #533's d trajectory as dotted ghost lines
   (means only; #533's E=1 diagnostic d's are the leftmost ghost
   markers).
3. Supporting — default-slot marker log P vs E per arm, per persona
   (#533 Finding 3 persistence check).

Reads from eval_results/issue_546/contrastive_negatives/cross_eval/per_cell/*.json
Ghost from eval_results/issue_533/contrastive_negatives/cross_eval/per_cell/*.json
Writes figures/issue_546/{wrong_slot_dose_response, paired_gap_per_persona,
                          default_slot_leakage}.{png,pdf,meta.json}
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
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

logger = logging.getLogger("i546.clean_result_figures")

PER_CELL_DIR_546 = Path("eval_results/issue_546/contrastive_negatives/cross_eval/per_cell")
PER_CELL_DIR_533 = Path("eval_results/issue_533/contrastive_negatives/cross_eval/per_cell")
FIG_DIR = Path("figures/issue_546")

EPOCHS = (1, 2, 3, 5)
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


def _cell_label(arm: str, seed: int, persona: str, epoch: int) -> str:
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


def _load_all(per_cell_dir: Path) -> dict:
    """Load all 120 cells x 3 eval encodings into a dict keyed by (arm,persona,epoch,seed)."""
    data = {}
    for arm in ARMS:
        for persona in PERSONAS:
            for epoch in EPOCHS:
                for seed in SEEDS:
                    base = per_cell_dir / f"{_cell_label(arm, seed, persona, epoch)}"
                    own_f = Path(f"{base}__{_own_enc(arm, persona)}.json")
                    wrong_f = Path(f"{base}__{_wrong_enc(arm, persona)}.json")
                    def_f = Path(f"{base}__default_assistant.json")
                    if not (own_f.exists() and wrong_f.exists() and def_f.exists()):
                        continue
                    data[(arm, persona, epoch, seed)] = {
                        "own": json.loads(own_f.read_text())["g_logprob"],
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


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, env={**os.environ}
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def fig_wrong_slot_dose_response(data_546: dict, data_533: dict) -> None:
    """Hero #1: wrong-slot log P vs E, per persona, per arm, with #533 ghost overlay.

    The figure that answers the Goal at a glance: does the r=16/alpha=32
    grid land all three arms simultaneously inside the [-10, -5] nat
    resolution band on at least one persona, where #533's r=32/alpha=64
    grid (dotted ghost) landed only 2 of 24 cells?
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8), sharey=True)
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        ax.axhspan(-10.0, -5.0, color="#bcd9b6", alpha=0.35, zorder=0)
        ax.text(
            3.0,
            -7.5,
            "resolution band\n[-10, -5] nat",
            ha="center",
            va="center",
            fontsize=8,
            color="#2d5530",
        )
        for arm in ARMS:
            # i546 (r=16/alpha=32)
            ys = []
            yerr_lo = []
            yerr_hi = []
            for epoch in EPOCHS:
                vals = [data_546[(arm, persona, epoch, seed)]["wrong"] for seed in SEEDS]
                m, lo, hi = _bootstrap(vals)
                ys.append(m)
                yerr_lo.append(m - lo)
                yerr_hi.append(hi - m)
            ax.errorbar(
                EPOCHS,
                ys,
                yerr=[yerr_lo, yerr_hi],
                marker="o",
                markersize=5,
                linewidth=1.6,
                capsize=3,
                label=f"{ARM_LABELS[arm]} — r=16 (this run)",
                color=ARM_COLORS[arm],
            )
            # #533 ghost (r=32/alpha=64, same lr=5e-6)
            ys533 = []
            for epoch in EPOCHS:
                vals = [data_533[(arm, persona, epoch, seed)]["wrong"] for seed in SEEDS]
                ys533.append(np.mean(vals))
            ax.plot(
                EPOCHS,
                ys533,
                marker="s",
                markersize=3.5,
                linewidth=1.0,
                linestyle=":",
                alpha=0.55,
                color=ARM_COLORS[arm],
            )
        ax.set_xticks(list(EPOCHS))
        ax.set_xlabel("Training epochs")
        if col_idx == 0:
            ax.set_ylabel("Marker log P (nats), trained model")
        ax.set_title(f"Trained on {persona}\n(probed under the OTHER persona)")
        ax.set_ylim(-17.5, -3)
    axes[0].legend(loc="lower left", fontsize=7.0, frameon=False, ncol=1)
    # Add a small ghost-line legend annotation
    axes[1].plot(
        [],
        [],
        color="gray",
        linestyle=":",
        marker="s",
        markersize=3.5,
        linewidth=1.0,
        alpha=0.55,
        label="#533 (r=32/alpha=64, mean only)",
    )
    axes[1].legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.suptitle(
        "Wrong-slot dose response at LoRA r=16/alpha=32 (lr=5e-6) vs #533's r=32/alpha=64",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.91,
        "Marker log-probability under the wrong persona's encoding. Lower = less leakage. "
        "Solid = this run at r=16/alpha=32, errorbars = 95% bootstrap CI over 5 seeds, "
        "n=50 questions each. "
        "Dotted = parent #533 at r=32/alpha=64 (same lr=5e-6), means only.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(
        fig,
        "issue_546/wrong_slot_dose_response",
        dir=str(Path("figures").resolve()),
    )


def fig_paired_gap_per_persona(data_546: dict, data_533: dict) -> None:
    """Hero #2: per-persona x per-contrast paired d = log P_system - log P_role
    across E, with #533's d trajectory as dotted ghost lines (its E=1
    diagnostic d's are the leftmost ghost markers).
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), sharey=True)
    contrasts = [
        ("system_plain", "System plain - Role"),
        ("system_padded", "System padded - Role"),
    ]
    contrast_markers = {"system_plain": "o", "system_padded": "^"}
    contrast_colors = {
        "system_plain": paper_palette_role("baseline"),
        "system_padded": paper_palette_role("control"),
    }
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.6, zorder=0)
        for sys_arm, label in contrasts:
            ys = []
            yerr_lo = []
            yerr_hi = []
            for epoch in EPOCHS:
                d_per_seed = []
                for seed in SEEDS:
                    sl = data_546[(sys_arm, persona, epoch, seed)]["wrong"]
                    rl = data_546[("role", persona, epoch, seed)]["wrong"]
                    d_per_seed.append(sl - rl)
                m, lo, hi = _bootstrap(d_per_seed)
                ys.append(m)
                yerr_lo.append(m - lo)
                yerr_hi.append(hi - m)
            ax.errorbar(
                EPOCHS,
                ys,
                yerr=[yerr_lo, yerr_hi],
                marker=contrast_markers[sys_arm],
                markersize=6,
                linewidth=1.5,
                capsize=3,
                label=label,
                color=contrast_colors[sys_arm],
            )
            # #533 ghost d trajectory (r=32/alpha=64, same lr).
            ys533 = []
            for epoch in EPOCHS:
                d533 = [
                    data_533[(sys_arm, persona, epoch, seed)]["wrong"]
                    - data_533[("role", persona, epoch, seed)]["wrong"]
                    for seed in SEEDS
                ]
                ys533.append(float(np.mean(d533)))
            ax.plot(
                EPOCHS,
                ys533,
                marker="s",
                markersize=3.5,
                linewidth=1.0,
                linestyle=":",
                alpha=0.55,
                color=contrast_colors[sys_arm],
            )
        ax.set_xticks(list(EPOCHS))
        ax.set_xlabel("Training epochs")
        if col_idx == 0:
            ax.set_ylabel("Paired d = log P (system - role)  (nats)")
        ax.set_title(f"Trained on {persona}")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    axes[1].plot(
        [],
        [],
        color="gray",
        linestyle=":",
        marker="s",
        markersize=3.5,
        linewidth=1.0,
        alpha=0.55,
        label="#533 (r=32/alpha=64, mean only)",
    )
    axes[1].legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.suptitle(
        "Role-vs-system paired gap at r=16/alpha=32, per persona x contrast, vs #533's r=32 ghost",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.91,
        "d = log P (system_arm) - log P (role) at the wrong-persona probe, paired per seed. "
        "Negative d = role leaks MORE than system. Errorbars = 95% bootstrap CI over 5 seeds. "
        "Dotted = parent #533 at r=32/alpha=64 (same lr=5e-6), means only.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(
        fig,
        "issue_546/paired_gap_per_persona",
        dir=str(Path("figures").resolve()),
    )


def fig_default_slot_leakage(data_546: dict) -> None:
    """Supporting fig: default-slot leakage at r=16/alpha=32 — does the
    #533 persona-asymmetric default-leakage finding persist?
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6), sharey=False)
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        for arm in ARMS:
            ys = []
            yerr_lo = []
            yerr_hi = []
            for epoch in EPOCHS:
                vals = [data_546[(arm, persona, epoch, seed)]["def"] for seed in SEEDS]
                m, lo, hi = _bootstrap(vals)
                ys.append(m)
                yerr_lo.append(m - lo)
                yerr_hi.append(hi - m)
            ax.errorbar(
                EPOCHS,
                ys,
                yerr=[yerr_lo, yerr_hi],
                marker="o",
                markersize=5,
                linewidth=1.4,
                capsize=3,
                label=ARM_LABELS[arm],
                color=ARM_COLORS[arm],
            )
        ax.set_xticks(list(EPOCHS))
        ax.set_xlabel("Training epochs")
        ax.set_ylabel("Marker log P (nats), trained model")
        ax.set_title(f"Trained on {persona}\n(probed under default assistant)")
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.6, zorder=0)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.suptitle(
        "Default-slot marker leakage at r=16/alpha=32 (#533 persona-asymmetry persistence check)",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.91,
        "Marker log-probability under the bare default-assistant encoding. log P = 0 means "
        "the model emits the marker with probability 1. n=5 seeds x 50 questions per point.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(
        fig,
        "issue_546/default_slot_leakage",
        dir=str(Path("figures").resolve()),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Loading per-cell data for #546 and #533 (ghost)...")
    data_546 = _load_all(PER_CELL_DIR_546)
    data_533 = _load_all(PER_CELL_DIR_533)
    logger.info(
        "Loaded %d/%d cells for #546, %d/%d for #533", len(data_546), 120, len(data_533), 120
    )
    if len(data_546) < 120 or len(data_533) < 120:
        logger.warning("Incomplete data — some cells missing.")
    logger.info("Building hero #1: wrong_slot_dose_response")
    fig_wrong_slot_dose_response(data_546, data_533)
    logger.info("Building hero #2: paired_gap_per_persona")
    fig_paired_gap_per_persona(data_546, data_533)
    logger.info("Building supporting fig: default_slot_leakage")
    fig_default_slot_leakage(data_546)
    logger.info("Wrote figures to %s", FIG_DIR.resolve())


if __name__ == "__main__":
    main()
