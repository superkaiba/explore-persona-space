"""Exploratory figure for #547 round 2: the UNGATED pre-install paired gap.

Plots the descriptive (implant-inactive, gate-excluded) per-seed paired
d = log P(system arm) - log P(role arm) at the wrong-persona teacher-forced
probe, at s in {5, 10, 18, 30}, computed directly from the 540 per-cell
JSONs (analysis.json stores empty per-seed dicts at inactive points by
design — the gate is correct for the leakage construct; this figure is the
explicitly-exploratory companion read).

No CIs on purpose: pre-install points are floor-regime descriptive reads,
so all 5 per-seed values are shown raw instead of a bootstrap interval.

Usage: uv run python scripts/i547_preinstall_ungated_figure.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

set_paper_style("blog")

PER_CELL = Path("eval_results/issue_547/contrastive_negatives/cross_eval/per_cell")
SEEDS = ("42", "137", "1337", "7", "21")
PERSONAS = ("pirate", "villain")
WRONG = {"pirate": "villain", "villain": "pirate"}
STEPS = (5, 10, 18, 30)
CONTRASTS = (
    ("plain", "system_plain", "plain system prompt − role header"),
    ("padded", "system_padded", "padded system prompt − role header"),
)
CONTRAST_MARKERS = {"plain": "o", "padded": "^"}
CONTRAST_COLORS = {
    "plain": paper_palette_role("baseline"),
    "padded": paper_palette_role("control"),
}


def _g(arm: str, seed: str, persona: str, steps: int, enc: str) -> float:
    path = PER_CELL / f"{arm}_seed{seed}_cn_{persona}_s{steps}__{enc}.json"
    return float(json.loads(path.read_text())["g_logprob"])


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.6), sharey=True)
    for col_idx, persona in enumerate(PERSONAS):
        ax = axes[col_idx]
        wrong = WRONG[persona]
        ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.6, zorder=0)
        # Implant-not-installed region (own argmax-emit = 0 everywhere at s <= 18).
        ax.axvspan(4.2, 22.5, color="#BBBBBB", alpha=0.18, zorder=0)
        for contrast_key, sys_arm, label in CONTRASTS:
            fam = "system"
            means: list[float] = []
            for steps in STEPS:
                per_seed = [
                    _g(sys_arm, seed, persona, steps, f"{fam}_{wrong}")
                    - _g("role", seed, persona, steps, f"role_{wrong}")
                    for seed in SEEDS
                ]
                means.append(statistics.mean(per_seed))
                ax.plot(
                    [steps] * len(per_seed),
                    per_seed,
                    marker=CONTRAST_MARKERS[contrast_key],
                    markersize=4,
                    linestyle="none",
                    alpha=0.45,
                    color=CONTRAST_COLORS[contrast_key],
                    zorder=2,
                )
            # Dashed pre-install segment (exploratory), solid handoff into s=30.
            ax.plot(
                STEPS[:3],
                means[:3],
                linestyle="--",
                linewidth=1.4,
                marker="none",
                color=CONTRAST_COLORS[contrast_key],
                label=label if col_idx == 0 else None,
                zorder=3,
            )
            ax.plot(
                STEPS[2:],
                means[2:],
                linestyle="--",
                linewidth=1.4,
                marker="none",
                alpha=0.55,
                color=CONTRAST_COLORS[contrast_key],
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_xlim(4.2, 36.0)
        ax.set_xticks(list(STEPS))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Training amount (optimizer steps, log scale)")
        if col_idx == 0:
            ax.set_ylabel("Ungated paired d = log P (system − role)  (nats)")
        ax.set_title(f"Trained on {persona}")
        ax.text(
            9.0,
            0.32,
            "implant not installed\n(gate-excluded;\nexploratory read)",
            fontsize=7.5,
            color="#777777",
            ha="center",
            va="top",
        )
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.suptitle(
        "Before the implant installs, the role header already carries more "
        "wrong-persona marker mass",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.90,
        "Descriptive per-seed paired d at the wrong-persona teacher-forced probe, computed from "
        "the per-cell JSONs WITHOUT the implant-active gate. Points = the 5 raw per-seed values "
        "(no bootstrap CI on purpose: floor-regime exploratory read). Dashed lines = per-contrast "
        "means. Own argmax-emit is 0.000 everywhere at 5–18 steps; the 30-step column matches the "
        "gated primary read.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(fig, "issue_547/preinstall_ungated_gap", dir=str(Path("figures").resolve()))
    plt.close(fig)
    print("saved figures/issue_547/preinstall_ungated_gap.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
