"""Figures for the #546 same-issue follow-up `fractional-epoch-grid-r16`.

Fork of ``scripts/i546_clean_result_figures.py`` (plan v2 §2 item (g)) onto
the optimizer-step axis: the three fractional points {47, 57, 66} from this
follow-up run are flanked by the parent #546 grid's E=1 (38-step) and E=2
(76-step) cells as same-recipe context. The endpoints are EPOCH-indexed runs
(the overlay is licensed by #533's <=0.14-nat indexing-validity check,
measured at r=32) and are drawn visually distinct (open squares, dashed
connecting segments) from this run's step-indexed points (filled circles,
solid segments, 95% bootstrap CI over 5 seeds).

Panels:

1. Hero — villain wrong-slot teacher-forced log P(' ※') vs optimizer steps,
   3 arms, [-10, -5] resolution band shaded.
2. Exploratory — pirate counterpart (same layout).
3. Exploratory — own-slot install: log P + argmax-emit rate vs steps, both
   personas (localizes the r=16 install cliff inside the window).
4. Exploratory — per-seed spread (villain): per-seed wrong-slot means at
   every step point, one panel per arm.

Reads  eval_results/issue_546/fractional-epoch-grid-r16/cross_eval/per_cell/*.json
       eval_results/issue_546/contrastive_negatives/cross_eval/per_cell/*.json (endpoints)
Writes figures/issue_546/fractional-epoch-grid-r16/*.{png,pdf,meta.json}
"""

from __future__ import annotations

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

logger = logging.getLogger("i546.frac_figures")

FRAC_DIR = Path("eval_results/issue_546/fractional-epoch-grid-r16/cross_eval/per_cell")
PARENT_DIR = Path("eval_results/issue_546/contrastive_negatives/cross_eval/per_cell")
OUT_SUBDIR = "issue_546/fractional-epoch-grid-r16"

SEEDS = (42, 137, 1337, 7, 21)
ARMS = ("system_plain", "system_padded", "role")
PERSONAS = ("pirate", "villain")
FRAC_STEPS = (47, 57, 66)
ENDPOINT_STEPS = {38: "e1", 76: "e2"}  # parent epoch-indexed cells on the step axis
ALL_STEPS = (38, 47, 57, 66, 76)
BAND = (-10.0, -5.0)

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
PERSONA_COLORS = {
    "villain": paper_palette_role("primary"),
    "pirate": paper_palette_role("accent"),
}


def _other(p: str) -> str:
    return "villain" if p == "pirate" else "pirate"


def _own_enc(arm: str, p: str) -> str:
    return f"role_{p}" if arm == "role" else f"system_{p}"


def _wrong_enc(arm: str, p: str) -> str:
    o = _other(p)
    return f"role_{o}" if arm == "role" else f"system_{o}"


def _cell_path(arm: str, seed: int, persona: str, steps: int, enc: str) -> Path:
    if steps in ENDPOINT_STEPS:
        return PARENT_DIR / f"{arm}_seed{seed}_cn_{persona}_{ENDPOINT_STEPS[steps]}__{enc}.json"
    return FRAC_DIR / f"{arm}_seed{seed}_cn_{persona}_s{steps}__{enc}.json"


def _load(arm: str, seed: int, persona: str, steps: int, enc: str) -> dict:
    p = _cell_path(arm, seed, persona, steps, enc)
    return json.loads(p.read_text())


def _bootstrap(vals: list[float], n: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Paired bootstrap CI over per-seed means: returns (mean, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    out = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n)]
    return float(arr.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _wrong_per_seed(persona: str, arm: str, steps: int) -> list[float]:
    return [_load(arm, s, persona, steps, _wrong_enc(arm, persona))["g_logprob"] for s in SEEDS]


def _band_shade(ax) -> None:
    ax.axhspan(BAND[0], BAND[1], color="#bcd9b6", alpha=0.35, zorder=0)


def _draw_arm_trajectory(ax, persona: str, arm: str) -> None:
    """Solid fractional points (95% CI) + open-square epoch-indexed endpoints
    joined by dashed alpha-reduced segments."""
    color = ARM_COLORS[arm]
    means = {st: _bootstrap(_wrong_per_seed(persona, arm, st)) for st in ALL_STEPS}
    # dashed context segments touching the epoch-indexed endpoints
    for seg in ((38, 47), (66, 76)):
        ax.plot(
            seg,
            [means[s][0] for s in seg],
            linestyle="--",
            linewidth=1.0,
            alpha=0.45,
            color=color,
            zorder=2,
        )
    # epoch-indexed endpoints: open squares, no CI (parent-run cells)
    for st in ENDPOINT_STEPS:
        ax.plot(
            [st],
            [means[st][0]],
            marker="s",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=color,
            linestyle="none",
            zorder=3,
        )
    # this run's fractional points: filled circles + solid line + CI
    xs = list(FRAC_STEPS)
    ys = [means[s][0] for s in xs]
    ax.errorbar(
        xs,
        ys,
        yerr=[
            [means[s][0] - means[s][1] for s in xs],
            [means[s][2] - means[s][0] for s in xs],
        ],
        marker="o",
        markersize=5.5,
        linewidth=1.7,
        capsize=3,
        color=color,
        label=ARM_LABELS[arm],
        zorder=4,
    )


def _endpoint_legend(ax, loc: str = "lower right") -> None:
    ax.plot(
        [],
        [],
        marker="o",
        color="#5A5A5A",
        linewidth=1.7,
        markersize=5.5,
        linestyle="-",
        label="this follow-up run (step-indexed; 95% CI over 5 seeds)",
    )
    ax.plot(
        [],
        [],
        marker="s",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor="#5A5A5A",
        color="#5A5A5A",
        linewidth=1.0,
        linestyle="--",
        alpha=0.6,
        label="parent grid cells (epoch-indexed: 1 and 2 epochs)",
    )
    ax.legend(loc=loc, fontsize=7.5, frameon=False)


def fig_wrong_slot(persona: str, name: str, title: str) -> None:
    """Wrong-slot log P vs optimizer steps for one persona, 3 arms."""
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    _band_shade(ax)
    ax.text(
        40.5,
        -7.5,
        "resolution band\n[-10, -5] nat",
        ha="center",
        va="center",
        fontsize=8,
        color="#2d5530",
    )
    for arm in ARMS:
        _draw_arm_trajectory(ax, persona, arm)
    ax.set_xticks(list(ALL_STEPS))
    ax.set_xlabel("Optimizer steps (each point an independent run; 38 steps = 1 epoch)")
    ax.set_ylabel("Marker log P (nats), trained model")
    ax.set_title(f"Trained on {persona} (probed under the OTHER persona)", fontsize=10)
    ax.set_ylim(-20.5, -3)
    from matplotlib.lines import Line2D

    arm_handles, arm_labels = ax.get_legend_handles_labels()
    style_handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="#5A5A5A",
            linewidth=1.7,
            markersize=5.5,
            label="this run (step-indexed; 95% CI, 5 seeds)",
        ),
        Line2D(
            [],
            [],
            marker="s",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor="#5A5A5A",
            color="#5A5A5A",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
            label="parent run (epoch-indexed: 1 / 2 epochs)",
        ),
    ]
    arm_legend = ax.legend(arm_handles, arm_labels, loc="lower left", fontsize=7.5, frameon=False)
    ax.add_artist(arm_legend)
    ax.legend(handles=style_handles, loc="lower right", fontsize=7.0, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(title, fontsize=11, fontweight="semibold", ha="left", x=0.02, y=0.99)
    fig.text(
        0.02,
        0.90,
        "Teacher-forced marker log-probability under the wrong persona's encoding; lower = less\n"
        "leakage. Endpoints (38/76 steps) are the parent run's 1- and 2-epoch cells, same recipe.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(fig, f"{OUT_SUBDIR}/{name}", dir=str(Path("figures").resolve()))
    plt.close(fig)


def fig_own_slot_install() -> None:
    """Own-slot install: log P (left) + argmax-emit rate (right) vs steps."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))
    for persona in PERSONAS:
        color = PERSONA_COLORS[persona]
        logp, emit = {}, {}
        for st in ALL_STEPS:
            vals, fires, n = [], 0, 0
            for arm in ARMS:
                for seed in SEEDS:
                    j = _load(arm, seed, persona, st, _own_enc(arm, persona))
                    vals.extend(j["g_logps_per_q"])
                    fires += sum(j["g_argmax_marker_per_q"])
                    n += j["n_probes"]
            logp[st] = float(np.mean(vals))
            emit[st] = 100.0 * fires / n
        for ax, series in ((axes[0], logp), (axes[1], emit)):
            for seg in ((38, 47), (66, 76)):
                ax.plot(
                    seg,
                    [series[s] for s in seg],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.45,
                    color=color,
                )
            for st in ENDPOINT_STEPS:
                ax.plot(
                    [st],
                    [series[st]],
                    marker="s",
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    linestyle="none",
                )
            ax.plot(
                list(FRAC_STEPS),
                [series[s] for s in FRAC_STEPS],
                marker="o",
                markersize=5.5,
                linewidth=1.7,
                color=color,
                label=f"trained on {persona}",
            )
    for ax in axes:
        ax.set_xticks(list(ALL_STEPS))
        ax.set_xlabel("Optimizer steps (38 = 1 epoch)")
    axes[0].set_ylabel("Own-encoding marker log P (nats)")
    axes[0].set_title("Marker log-probability at the source persona's own slot", fontsize=9.5)
    axes[1].set_ylabel("Questions where marker is argmax (%)")
    axes[1].set_title("Install rate (marker is the argmax next token)", fontsize=9.5)
    axes[1].axhline(50.0, color="gray", linestyle=":", linewidth=0.8)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(
        "The r=16 install cliff sits between 47 and 57 optimizer steps",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.90,
        "Source persona's own encoding, teacher-forced; n = 750 probes per persona per point\n"
        "(3 arms x 5 seeds x 50 questions). Open squares = parent run's epoch-indexed cells.",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(fig, f"{OUT_SUBDIR}/own_slot_install_steps", dir=str(Path("figures").resolve()))
    plt.close(fig)


def fig_per_seed_spread() -> None:
    """Per-seed wrong-slot means (villain), one panel per arm, all 5 step points."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, arm in zip(axes, ARMS):
        _band_shade(ax)
        color = ARM_COLORS[arm]
        for st in ALL_STEPS:
            vals = _wrong_per_seed("villain", arm, st)
            jitter = rng.uniform(-1.2, 1.2, size=len(vals))
            filled = st in FRAC_STEPS
            ax.plot(
                st + jitter,
                vals,
                marker="o" if filled else "s",
                markersize=4.5,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                linestyle="none",
                alpha=0.85,
            )
            ax.plot(
                [st - 2.6, st + 2.6],
                [np.mean(vals)] * 2,
                color=color,
                linewidth=1.6,
                alpha=0.9,
            )
        ax.set_xticks(list(ALL_STEPS))
        ax.set_xlabel("Optimizer steps")
        ax.set_title(ARM_LABELS[arm], fontsize=9)
    axes[0].set_ylabel("Wrong-persona marker log P (nats), per seed")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(
        "Per-seed spread at every step point (villain): the padded-arm miss is not an outlier seed",
        fontsize=11,
        fontweight="semibold",
        ha="left",
        x=0.02,
        y=0.99,
    )
    fig.text(
        0.02,
        0.90,
        "One dot per seed (5 per point), horizontal bar = mean. Filled circles = this run's\n"
        "step-indexed points; open squares = parent run's epoch-indexed cells (38/76 steps).",
        fontsize=8,
        color="#5A5A5A",
        ha="left",
    )
    savefig_paper(fig, f"{OUT_SUBDIR}/per_seed_spread_villain", dir=str(Path("figures").resolve()))
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n_frac = len(list(FRAC_DIR.glob("*.json")))
    logger.info("Found %d fractional per-cell JSONs (expect 270)", n_frac)
    assert n_frac == 270, f"expected 270 per-cell JSONs, found {n_frac}"
    fig_wrong_slot(
        "villain",
        "wrong_slot_steps_villain",
        "Inside the (1, 2)-epoch window the padded-system arm never reaches the band (villain)",
    )
    fig_wrong_slot(
        "pirate",
        "wrong_slot_steps_pirate",
        "Pirate's wrong-slot reads stay below the band at every step point",
    )
    fig_own_slot_install()
    fig_per_seed_spread()
    logger.info("Wrote figures to figures/%s", OUT_SUBDIR)


if __name__ == "__main__":
    main()
