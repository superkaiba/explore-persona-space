"""Issue #533 bare-word grid: leakage controlled for source implantation.

Companion to i533_bw_implant_leakage_figure.py. Same layout (implantation
left, leakage right), but the right panel reports the per-seed difference
leakage - implantation (both in marker log P trained - base), so an arm
whose leakage is lower merely because its implant is weaker collapses onto
the other arm.

Context that motivates the second figure: at s ≥ 60 the own-slot trained
log P is exactly 0 for BOTH arms (P(marker) ≈ 1, per-seed std = 0), so the
apparent ~1-nat implantation gap between arms is purely a base-prior
difference (base log P -21.65 system vs -20.68 role). The decomposition
figure splits average leakage into its two probes (wrong-persona vs
default-assistant) to show where each arm's leakage actually lives.

Data: eval_results/issue_533/bare_word_install_step_grid/cross_eval/per_cell
(fields g_logprob/b_logprob/delta_g are means over the 50 held-out
questions per probe).
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

    Metrics: implantation (own-probe delta_g), leakage (mean of wrong +
    default probe delta_g), controlled (leakage - implantation, paired
    within seed), leak_wrong / leak_default (the two probes separately).
    All are first averaged over the 2 trained personas within a seed.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        metrics = {
            k: np.zeros((len(STEPS), len(SEEDS)))
            for k in ("implantation", "leakage", "controlled", "leak_wrong", "leak_default")
        }
        for i, step in enumerate(STEPS):
            for j, seed in enumerate(SEEDS):
                impl_c, wrong_c, default_c = [], [], []
                for persona in PERSONAS:
                    other = "villain" if persona == "pirate" else "pirate"
                    impl_c.append(_load(arm, seed, persona, step, f"{arm}_{persona}")["delta_g"])
                    wrong_c.append(_load(arm, seed, persona, step, f"{arm}_{other}")["delta_g"])
                    default_c.append(
                        _load(arm, seed, persona, step, "default_assistant")["delta_g"]
                    )
                impl = float(np.mean(impl_c))
                wrong = float(np.mean(wrong_c))
                default = float(np.mean(default_c))
                leak = 0.5 * (wrong + default)
                metrics["implantation"][i, j] = impl
                metrics["leakage"][i, j] = leak
                metrics["controlled"][i, j] = leak - impl
                metrics["leak_wrong"][i, j] = wrong
                metrics["leak_default"][i, j] = default
        out[arm] = metrics
    return out


def boot_ci(per_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean + 95% percentile bootstrap CI over the seed axis (last axis)."""
    mean = per_seed.mean(axis=-1)
    idx = RNG.integers(0, per_seed.shape[-1], size=(N_BOOT, per_seed.shape[-1]))
    boots = per_seed[..., idx].mean(axis=-1)
    lo = np.percentile(boots, 2.5, axis=-1)
    hi = np.percentile(boots, 97.5, axis=-1)
    return mean, lo, hi


def _draw_metric(ax: plt.Axes, data: dict, metric: str) -> None:
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


def controlled_figure(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True)

    _draw_metric(axes[0], data, "implantation")
    axes[0].set_title("Source implantation\n(own-persona probe)", fontsize=11)
    axes[0].axhline(0.0, color="gray", lw=0.8, alpha=0.6, zorder=0)
    add_direction_arrow(
        axes[0],
        axis="y",
        direction="up",
        label="Marker log P, trained − base (nats)\n↑ stronger implant",
    )
    axes[0].text(
        18,
        axes[0].get_ylim()[1] * 0.93,
        "implant not\ninstalled",
        fontsize=8,
        color="0.45",
        ha="center",
        va="top",
    )

    _draw_metric(axes[1], data, "controlled")
    axes[1].set_title(
        "Implantation-controlled leakage\n(leakage − implantation, per seed)", fontsize=11
    )
    axes[1].axhline(0.0, color="gray", lw=0.8, alpha=0.6, zorder=0)
    axes[1].text(
        STEPS[-1],
        -0.35,
        "0 = leakage as strong as implant",
        fontsize=8,
        color="0.45",
        ha="right",
        va="top",
    )
    add_direction_arrow(
        axes[1],
        axis="y",
        direction="down",
        label="Leakage − implantation (nats)\n↓ more selective",
    )
    axes[1].legend(loc="lower left", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.80])
    fig.suptitle(
        "Controlling for implantation flips the role arm's small average-leakage advantage",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.985,
    )
    fig.text(
        0.02,
        0.94,
        "Bare-word install-step grid (#533). Right panel: per-seed leakage minus implantation\n"
        "(both marker log P, trained − base). Caveat: at s ≥ 60 the own-slot trained log P is\n"
        "exactly 0 for BOTH arms (implant saturated, per-seed std = 0) — the ~1-nat gap is a\n"
        "base-prior difference, which this subtraction inherits.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
        va="top",
    )

    savefig_paper(fig, "issue_533/bw_leakage_implant_controlled", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def decomposition_figure(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True, sharey=True)

    panels = (
        ("leak_wrong", "Wrong-persona probe\n(own encoding, other persona)"),
        ("leak_default", "Default-assistant probe\n(identical prompt for both arms)"),
    )
    for ax, (metric, panel_title) in zip(axes, panels, strict=True):
        _draw_metric(ax, data, metric)
        ax.set_title(panel_title, fontsize=11)
        ax.axhline(0.0, color="gray", lw=0.8, alpha=0.6, zorder=0)
    add_direction_arrow(
        axes[0],
        axis="y",
        direction="down",
        label="Marker log P, trained − base (nats)\n↓ less leakage",
    )
    axes[0].legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.80])
    fig.suptitle(
        "The average hides a split: role leaks less to the wrong persona, more to the default",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.985,
    )
    fig.text(
        0.02,
        0.94,
        "Same cells as the average-leakage panel, split by probe. Wrong-persona probes use each\n"
        "arm's own encoding (base priors differ: −21.65 system vs −20.68 role); the default\n"
        "probe is encoding-identical across arms (base −22.09), so its comparison is base-clean.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
        va="top",
    )

    savefig_paper(fig, "issue_533/bw_leakage_decomposition", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def main() -> None:
    data = per_seed_values()
    set_paper_style("blog")
    controlled_figure(data)
    decomposition_figure(data)

    for arm in ARMS:
        for metric in ("implantation", "leakage", "controlled", "leak_wrong", "leak_default"):
            mean, lo, hi = boot_ci(data[arm][metric])
            vals = ", ".join(
                f"s{s}={m:+.2f} [{lo_:+.2f},{hi_:+.2f}]"
                for s, m, lo_, hi_ in zip(STEPS, mean, lo, hi, strict=True)
            )
            print(f"{arm:15s} {metric:13s} {vals}")


if __name__ == "__main__":
    main()
