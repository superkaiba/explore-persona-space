"""Issue #533 bare-word grid: source implantation vs average leakage, per encoding arm.

Plots two panels over the install-step grid {18, 30, 60, 120}:
  (A) source implantation — marker log P trained - base at the OWN-persona probe
  (B) average leakage     — marker log P trained - base averaged over the two
      off-target probes (wrong-persona + default-assistant)
each comparing the two bare-wording encoding arms (minimal system prompt vs
bare role header), averaged over the 2 trained personas, with 95% bootstrap
CIs over the 5 training seeds.

Data: eval_results/issue_533/bare_word_install_step_grid/cross_eval/per_cell
(240 per-cell JSONs; fields g_logprob/b_logprob/delta_g are means over the 50
held-out questions per probe).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PER_CELL = REPO_ROOT / "eval_results/issue_533/bare_word_install_step_grid/cross_eval/per_cell"

ARMS = ("system_minimal", "role_bare")
ARM_LABELS = {
    "system_minimal": "Minimal system prompt (“You are a pirate.”)",
    "role_bare": "Bare role header (pirate)",
}
ARM_COLORS = {
    "system_minimal": paper_palette_role("baseline"),
    "role_bare": paper_palette_role("primary"),
}
PERSONAS = ("pirate", "villain")
SEEDS = (7, 21, 42, 137, 1337)
STEPS = (18, 30, 60, 120)
N_BOOT = 10_000
RNG = np.random.default_rng(0)


def _load(arm: str, seed: int, persona: str, step: int, probe: str) -> dict:
    path = PER_CELL / f"{arm}_seed{seed}_cn_{persona}_s{step}__{probe}.json"
    with open(path) as f:
        return json.load(f)


def per_seed_values() -> dict[str, dict[str, np.ndarray]]:
    """Return {arm: {metric: array of shape (len(STEPS), len(SEEDS))}}.

    Per-seed value = mean over the 2 trained personas of the per-cell
    delta_g (trained - base mean log P over 50 questions). For leakage the
    per-cell value is itself the mean of the wrong-persona and
    default-assistant probes.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        impl = np.zeros((len(STEPS), len(SEEDS)))
        leak = np.zeros((len(STEPS), len(SEEDS)))
        for i, step in enumerate(STEPS):
            for j, seed in enumerate(SEEDS):
                impl_cells, leak_cells = [], []
                for persona in PERSONAS:
                    other = "villain" if persona == "pirate" else "pirate"
                    own = _load(arm, seed, persona, step, f"{arm}_{persona}")
                    wrong = _load(arm, seed, persona, step, f"{arm}_{other}")
                    default = _load(arm, seed, persona, step, "default_assistant")
                    impl_cells.append(own["delta_g"])
                    leak_cells.append(0.5 * (wrong["delta_g"] + default["delta_g"]))
                impl[i, j] = np.mean(impl_cells)
                leak[i, j] = np.mean(leak_cells)
        out[arm] = {"implantation": impl, "leakage": leak}
    return out


def boot_ci(per_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean + 95% percentile bootstrap CI over the seed axis (last axis)."""
    mean = per_seed.mean(axis=-1)
    idx = RNG.integers(0, per_seed.shape[-1], size=(N_BOOT, per_seed.shape[-1]))
    boots = per_seed[..., idx].mean(axis=-1)  # (steps, N_BOOT)
    lo = np.percentile(boots, 2.5, axis=-1)
    hi = np.percentile(boots, 97.5, axis=-1)
    return mean, lo, hi


def main() -> None:
    data = per_seed_values()

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True)

    panels = (
        (
            "implantation",
            "Source implantation\n(own-persona probe)",
            "Marker log P, trained − base (nats)\n↑ stronger implant",
        ),
        (
            "leakage",
            "Average leakage\n(wrong-persona + default-assistant probes)",
            "Marker log P, trained − base (nats)\n↓ less leakage",
        ),
    )
    for ax, (metric, panel_title, ylabel) in zip(axes, panels):
        ax.axvspan(15, 23, color="0.85", alpha=0.5, zorder=0)
        for arm in ARMS:
            mean, lo, hi = boot_ci(data[arm][metric])
            ax.errorbar(
                STEPS,
                mean,
                yerr=[mean - lo, hi - mean],
                color=ARM_COLORS[arm],
                marker="o",
                lw=1.5,
                capsize=3,
                markeredgewidth=0,
                label=ARM_LABELS[arm],
            )
            ax.fill_between(STEPS, lo, hi, color=ARM_COLORS[arm], alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(STEPS)
        ax.set_xticklabels([str(s) for s in STEPS])
        ax.minorticks_off()
        ax.set_xlabel("Optimizer steps")
        ax.set_title(panel_title, fontsize=11)
        ax.axhline(0.0, color="gray", lw=0.8, alpha=0.6, zorder=0)
        add_direction_arrow(ax, axis="y", direction="up", label=ylabel)
    axes[0].text(
        18,
        axes[0].get_ylim()[1] * 0.93,
        "implant not\ninstalled",
        fontsize=8,
        color="0.45",
        ha="center",
        va="top",
    )
    axes[1].legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(
        "Minimal wording: both encodings implant alike; average leakage crosses — role higher at install, lower with more training",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.905,
        "Bare-word install-step grid (#533): minimal system prompt vs bare role header. "
        "Mean over 2 trained personas; errorbars = 95% bootstrap CI over 5 seeds; n = 50 questions per probe per cell.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )

    savefig_paper(fig, "issue_533/bw_implant_vs_avg_leakage", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)

    # console summary for the chat report
    for arm in ARMS:
        for metric in ("implantation", "leakage"):
            mean, lo, hi = boot_ci(data[arm][metric])
            vals = ", ".join(
                f"s{s}={m:+.2f} [{l:+.2f},{h:+.2f}]" for s, m, l, h in zip(STEPS, mean, lo, hi)
            )
            print(f"{arm:15s} {metric:12s} {vals}")


if __name__ == "__main__":
    main()
