"""Figures for the #529 clean-result body.

Two hero figures:
  1. Wrong-slot dose-response — log P(' ※') vs E per arm, per persona, with the
     saturation floor visible (the +1-nat #464 edge sits inside the floor).
  2. Default-slot leakage — persona-asymmetric default leakage shows role
     encoding leaks the trained pirate marker into the bare default-assistant
     context dramatically more than system encoding, but the opposite for villain.

Plus a supporting figure:
  3. Paired role-vs-system gap (bootstrap CI) across epochs, wrong-slot AND
     default-slot, on the same axis so the reader can see the sign of the
     effect and how it moves with training amount.

Reads from eval_results/issue_529/contrastive_negatives/cross_eval/per_cell/*.json
Writes figures/issue_529/{wrong_slot_dose_response, default_slot_leakage, paired_gap_vs_e}.{png,pdf,meta.json}
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

logger = logging.getLogger("i529.clean_result_figures")

PER_CELL_DIR = Path("eval_results/issue_529/contrastive_negatives/cross_eval/per_cell")
FIG_DIR = Path("figures/issue_529")

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


def _load_all() -> dict:
    """Load all 120 cells × 3 eval encodings into a dict keyed by (arm,persona,epoch,seed)."""
    data = {}
    for arm in ARMS:
        for persona in PERSONAS:
            for epoch in EPOCHS:
                for seed in SEEDS:
                    base = PER_CELL_DIR / f"{_cell_label(arm, seed, persona, epoch)}"
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


def fig_wrong_slot_dose_response(data: dict) -> None:
    """Hero #1: wrong-slot log P vs E, per persona, per arm.

    Shows: (a) the wrong-slot is already at log P ≈ -13 nats at E=1 — deep in
    the saturated floor — for all three arms; (b) the [-10, -5] resolution band
    is never reached at any E in {1, 2, 3, 5}.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.6), sharey=True)
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
            ys = []
            yerr_lo = []
            yerr_hi = []
            for epoch in EPOCHS:
                vals = [data[(arm, persona, epoch, seed)]["wrong"] for seed in SEEDS]
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
        if col_idx == 0:
            ax.set_ylabel("Marker log P (nats), trained model")
        ax.set_title(f"Trained on {persona}\n(probed under the OTHER persona)")
        ax.set_ylim(-17.5, -3)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(
        "The wrong-slot read sits deep in the saturated floor at every epoch we sampled",
        fontsize=12,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.91,
        "Marker log-probability under the wrong persona's encoding. Lower = less leakage. "
        "All three encoding arms sit at log P ≈ −13 nats at E=1, far below the [−10, −5] "
        "resolution band. Errorbars = 95% bootstrap CI over 5 seeds, n=50 questions each.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(
        fig,
        "issue_529/wrong_slot_dose_response",
        dir=str(Path("figures").resolve()),
    )


def fig_default_slot_leakage(data: dict) -> None:
    """Hero #2: default-slot leakage per persona × arm × E.

    Shows: under the trained pirate LoRA, the role encoding leaks the marker
    to the bare default-assistant context at log P ≈ -0.2 (P ≈ 0.82), much
    more than system_plain (-2.3) or system_padded (-4). Under villain, role
    leaks LESS to default than system arms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.6), sharey=False)
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        for arm in ARMS:
            ys = []
            yerr_lo = []
            yerr_hi = []
            for epoch in EPOCHS:
                vals = [data[(arm, persona, epoch, seed)]["def"] for seed in SEEDS]
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
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(
        "Default-assistant leakage is persona-asymmetric: role leaks more for pirate, less for villain",
        fontsize=12,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.91,
        "Marker log-probability under the bare default-assistant encoding. log P = 0 means "
        "the model emits the marker with probability 1. Pirate-trained LoRA: role encoding "
        "leaks at P ≈ 0.82, system encodings at P ≈ 0.02 to 0.4. Villain: ordering reverses. "
        "n=5 seeds × 50 questions per point.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(
        fig,
        "issue_529/default_slot_leakage",
        dir=str(Path("figures").resolve()),
    )


def fig_paired_gap_vs_e(data: dict) -> None:
    """Supporting: paired role-vs-system gap (d = L_system − L_role) at wrong-slot
    AND default-slot, averaged over the pirate/villain pair, across E."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.6), sharey=False)

    # Left: wrong-slot d (the headline statistic from plan §4.4)
    ax = axes[0]
    for arm in ("system_plain", "system_padded"):
        means = []
        los = []
        his = []
        for epoch in EPOCHS:
            d_seeds = []
            for seed in SEEDS:
                L_sys = np.mean([data[(arm, p, epoch, seed)]["wrong"] for p in PERSONAS])
                L_role = np.mean([data[("role", p, epoch, seed)]["wrong"] for p in PERSONAS])
                d_seeds.append(L_sys - L_role)
            m, lo, hi = _bootstrap(d_seeds)
            means.append(m)
            los.append(m - lo)
            his.append(hi - m)
        ax.errorbar(
            EPOCHS,
            means,
            yerr=[los, his],
            marker="o" if arm == "system_plain" else "^",
            markersize=6,
            linewidth=1.4,
            capsize=3,
            label=f"d = {ARM_LABELS[arm]} − role",
            color=ARM_COLORS[arm],
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xticks(list(EPOCHS))
    ax.set_xlabel("Training epochs")
    ax.set_ylabel("d = L_system − L_role (nats)")
    ax.set_title("Wrong-slot gap")
    ax.text(
        1.0,
        ax.get_ylim()[1] * 0.9,
        "+ ⇒ role leaks LESS\n− ⇒ role leaks MORE",
        ha="left",
        va="top",
        fontsize=7,
        color="gray",
    )
    ax.legend(loc="lower right", fontsize=7, frameon=False)

    # Right: default-slot d
    ax = axes[1]
    for arm in ("system_plain", "system_padded"):
        means = []
        los = []
        his = []
        for epoch in EPOCHS:
            d_seeds = []
            for seed in SEEDS:
                L_sys = np.mean([data[(arm, p, epoch, seed)]["def"] for p in PERSONAS])
                L_role = np.mean([data[("role", p, epoch, seed)]["def"] for p in PERSONAS])
                d_seeds.append(L_sys - L_role)
            m, lo, hi = _bootstrap(d_seeds)
            means.append(m)
            los.append(m - lo)
            his.append(hi - m)
        ax.errorbar(
            EPOCHS,
            means,
            yerr=[los, his],
            marker="o" if arm == "system_plain" else "^",
            markersize=6,
            linewidth=1.4,
            capsize=3,
            label=f"d = {ARM_LABELS[arm]} − role",
            color=ARM_COLORS[arm],
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xticks(list(EPOCHS))
    ax.set_xlabel("Training epochs")
    ax.set_ylabel("d = L_system − L_role (nats)")
    ax.set_title("Default-slot gap")
    ax.legend(loc="upper left", fontsize=7, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(
        "The sign of the role-vs-system gap flips between epochs",
        fontsize=12,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.91,
        "d = log P_system − log P_role, paired per seed (averaged over pirate, villain), "
        "bootstrapped (N=10,000) over 5 seeds. Positive d means role leaks LESS. At E=1, "
        "d goes the other way at both slots; by E=3, d crosses zero and matches the saturated "
        "edge from the parent run.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(
        fig,
        "issue_529/paired_gap_vs_e",
        dir=str(Path("figures").resolve()),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_all()
    logger.info("loaded n=%d cells", len(data))
    fig_wrong_slot_dose_response(data)
    fig_default_slot_leakage(data)
    fig_paired_gap_vs_e(data)
    logger.info("wrote 3 hero/supporting figures under %s", FIG_DIR)


if __name__ == "__main__":
    main()
