"""Figures for the #533 clean-result body.

The three findings of this run and their figures:

1. **The LR drop did not de-saturate the grid.** Hero figure — wrong-slot
   teacher-forced log P(' ※') vs E per arm, per persona, with the [−10, −5]
   resolution band shaded, and the parent #529 trajectory plotted as a
   ghost dotted line for the LR-drop comparison. The trajectory shifts up
   by ≈ 2 nats at E=1 but re-saturates by E=2+.

2. **Per-persona × per-contrast paired d at E=1 reverses #464/#529's
   sign.** Heat-style summary — d = log P_system − log P_role, per
   persona × per contrast × per E, 95% bootstrap CI errorbars. Negative
   values mean role LEAKS MORE than system; the parent's saturated +1.46
   nat (positive, role-wins) read inverts to clearly negative at the
   closest-to-resolution epoch.

3. **The persona-asymmetric default-slot leakage from #529 persists at
   lr=5e-6.** Supporting figure — default-slot marker log P vs E per arm,
   per persona; same pattern as #529's Fig 2.

Reads from eval_results/issue_533/contrastive_negatives/cross_eval/per_cell/*.json
Writes figures/issue_533/{wrong_slot_dose_response, paired_gap_per_persona,
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

logger = logging.getLogger("i533.clean_result_figures")

PER_CELL_DIR_533 = Path("eval_results/issue_533/contrastive_negatives/cross_eval/per_cell")
PER_CELL_DIR_529 = Path("eval_results/issue_529/contrastive_negatives/cross_eval/per_cell")
FIG_DIR = Path("figures/issue_533")

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
    """Load all 120 cells × 3 eval encodings into a dict keyed by (arm,persona,epoch,seed)."""
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


def fig_wrong_slot_dose_response(data_533: dict, data_529: dict) -> None:
    """Hero #1: wrong-slot log P vs E, per persona, per arm, with #529 ghost overlay.

    Shows: (a) at E=1 the LR drop (1e-5 → 5e-6) shifts the wrong-slot read UP
    by 2-3 nats, getting villain close to the [−10, −5] band but pirate still
    below; (b) by E=2+ the trajectory re-saturates back into the floor; (c) only
    2 of 24 cells touch the band (villain E=1 system_plain at −9.85, role at
    −7.87) and even then not all 3 arms simultaneously, so the anchor gate
    refused to fire.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8), sharey=True)
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        ax.axhspan(-10.0, -5.0, color="#bcd9b6", alpha=0.35, zorder=0)
        ax.text(
            3.0,
            -7.5,
            "resolution band\n[−10, −5] nat",
            ha="center",
            va="center",
            fontsize=8,
            color="#2d5530",
        )
        for arm in ARMS:
            # i533 (lr=5e-6)
            ys = []
            yerr_lo = []
            yerr_hi = []
            for epoch in EPOCHS:
                vals = [data_533[(arm, persona, epoch, seed)]["wrong"] for seed in SEEDS]
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
                label=f"{ARM_LABELS[arm]} — lr=5e-6 (this run)",
                color=ARM_COLORS[arm],
            )
            # #529 ghost (lr=1e-5)
            ys529 = []
            for epoch in EPOCHS:
                vals = [data_529[(arm, persona, epoch, seed)]["wrong"] for seed in SEEDS]
                ys529.append(np.mean(vals))
            ax.plot(
                EPOCHS,
                ys529,
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
        label="#529 (lr=1e-5, mean only)",
    )
    axes[1].legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.suptitle(
        "Dropping lr 1e-5 → 5e-6 shifted the E=1 wrong-slot read up by ≈ 2 nats — still not enough to clear the band on all 3 arms",
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
        "Solid = this run at lr=5e-6, errorbars = 95% bootstrap CI over 5 seeds, n=50 questions each. "
        "Dotted = parent #529 at lr=1e-5, means only.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(
        fig,
        "issue_533/wrong_slot_dose_response",
        dir=str(Path("figures").resolve()),
    )


def fig_paired_gap_per_persona(data_533: dict) -> None:
    """Hero #2: per-persona × per-contrast paired d = log P_system − log P_role
    across E, showing the sign reversal of the #464/#529 saturated edge.

    Shows: at E=1 (the only quasi-resolvable epoch at lr=5e-6, where the read has
    measurable headroom), all 4 per-persona × per-contrast cells have d clearly
    NEGATIVE — role LEAKS MORE than system, the OPPOSITE direction from #529's
    saturated +1.46 nat read at E=3. As E grows the trajectory drifts back
    toward the saturated #529 pattern (sign-mixed, near zero).
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), sharey=True)
    contrasts = [
        ("system_plain", "System plain − Role"),
        ("system_padded", "System padded − Role"),
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
                    sl = data_533[(sys_arm, persona, epoch, seed)]["wrong"]
                    rl = data_533[("role", persona, epoch, seed)]["wrong"]
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
        # Annotate #529's E=3 saturated read (the headline they reported)
        ax.annotate(
            "",
            xy=(3.0, 0.0),
            xytext=(3.0, 1.5),
            arrowprops=dict(arrowstyle="->", color="#888", linewidth=0.8),
        )
        ax.text(
            3.0,
            1.8,
            "#529's saturated\n+1.46 nat read",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#666",
        )
        ax.set_xticks(list(EPOCHS))
        ax.set_xlabel("Training epochs")
        if col_idx == 0:
            ax.set_ylabel("Paired d = log P (system − role)  (nats)")
        ax.set_title(f"Trained on {persona}")
        ax.set_ylim(-3.2, 2.6)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    # Direction-of-effect annotations
    axes[0].text(
        0.55,
        1.5,
        "↑ role leaks LESS than system\n     (parent's claimed direction)",
        fontsize=7.0,
        color="#5A5A5A",
        ha="left",
        va="center",
    )
    axes[0].text(
        0.55,
        -2.6,
        "↓ role leaks MORE than system\n     (this run at E=1)",
        fontsize=7.0,
        color="#5A5A5A",
        ha="left",
        va="center",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.suptitle(
        "At E=1 (closest to resolution), the role-vs-system gap REVERSES the parent's saturated +1-nat direction",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.91,
        "d = log P (system_arm) − log P (role) at the wrong-persona probe, paired per seed. "
        "Negative d = role leaks MORE. Errorbars = 95% bootstrap CI over 5 seeds. "
        "At E=1 all 4 cells clearly clear zero on the negative side with 100% per-seed sign-agreement; "
        "as E grows the trajectory drifts toward the saturated #529 sign-mixed regime.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(
        fig,
        "issue_533/paired_gap_per_persona",
        dir=str(Path("figures").resolve()),
    )


def fig_default_slot_leakage(data_533: dict) -> None:
    """Supporting fig: default-slot leakage at lr=5e-6 — does the #529
    persona-asymmetric default-leakage finding persist?
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6), sharey=False)
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        for arm in ARMS:
            ys = []
            yerr_lo = []
            yerr_hi = []
            for epoch in EPOCHS:
                vals = [data_533[(arm, persona, epoch, seed)]["def"] for seed in SEEDS]
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
        "The #529 persona-asymmetric default-slot leakage finding persists at lr=5e-6",
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
        "the model emits the marker with probability 1. Pirate-trained LoRA: role at log P ≈ −0.4 "
        "to −1.9 (P ≈ 0.15 to 0.67), system arms at −3.6 to −8.9. Villain: ordering more mixed, "
        "role between −9.7 and −11.2. n=5 seeds × 50 questions per point.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(
        fig,
        "issue_533/default_slot_leakage",
        dir=str(Path("figures").resolve()),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Loading per-cell data for #533 and #529...")
    data_533 = _load_all(PER_CELL_DIR_533)
    data_529 = _load_all(PER_CELL_DIR_529)
    logger.info(
        "Loaded %d/%d cells for #533, %d/%d for #529", len(data_533), 120, len(data_529), 120
    )
    if len(data_533) < 120 or len(data_529) < 120:
        logger.warning("Incomplete data — some cells missing.")
    logger.info("Building hero #1: wrong_slot_dose_response")
    fig_wrong_slot_dose_response(data_533, data_529)
    logger.info("Building hero #2: paired_gap_per_persona")
    fig_paired_gap_per_persona(data_533)
    logger.info("Building supporting fig: default_slot_leakage")
    fig_default_slot_leakage(data_533)
    logger.info("Wrote figures to %s", FIG_DIR.resolve())


if __name__ == "__main__":
    main()
