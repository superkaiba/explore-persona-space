"""Regenerate the issue #715 EM-vs-narrow-task Pareto figure in blog style.

Each point is one (objective, seed, checkpoint) cell from the LoRA Pareto
sweep: x = held-out narrow-task (bad-medical-advice) acquisition rate,
y = OOD emergent-misalignment rate (Betley 8-question free-form eval,
claude-sonnet-4-5 judge, aligned<30 AND coherence>50).

The y values are read from ``pareto_em_vs_narrow_corrected.json`` — the EM
rate with judge-API-error rows (Anthropic Batch API 529 "Overloaded")
excluded from the denominator (y = n_misaligned / (n_attempted - n_judge_error)).
The original ``pareto_em_vs_narrow.json`` keeps the uncorrected n=400-denominator
values for traceability.

No connecting lines — the points are not a trajectory, they are independent
checkpoints. The planned no-overlap kill: there is no narrow-acquisition
value reached by all 3 seeds in BOTH arms (the all-3-seed common matched-
acquisition interval is empty: SFT [0.799, 0.854] vs DFT [0.515, 0.646]),
so the matched-acquisition test is undefined (D* = null).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

PARETO = Path("eval_results/issue_715/pareto_em_vs_narrow_corrected.json")


def main() -> None:
    d = json.loads(PARETO.read_text())
    cells = d["cells"]

    set_paper_style("blog")
    fig, ax = plt.subplots()

    colours = {
        "sft_lora": paper_palette_role("baseline"),
        "dft_lora": paper_palette_role("primary"),
    }
    labels = {
        "sft_lora": "Standard SFT",
        "dft_lora": "DFT (stop-gradient reweight)",
    }

    for cond in ("sft_lora", "dft_lora"):
        xs, ys = [], []
        for seed in ("42", "137", "256"):
            for p in cells[cond][seed]:
                xs.append(p["x"])
                ys.append(p["y"])
        ax.scatter(
            xs,
            ys,
            color=colours[cond],
            s=42,
            alpha=0.85,
            edgecolor="white",
            linewidths=0.6,
            label=f"{labels[cond]} (n={len(xs)} checkpoints)",
            zorder=3,
        )

    ax.set_xlabel("In-distribution narrow-task acquisition\n(held-out bad-medical-advice rate)")
    ax.set_ylabel("Out-of-distribution EM rate\n(Betley 8-question free-form eval)")
    ax.set_ylim(0.0, 0.31)
    ax.set_xlim(0.30, 0.96)
    ax.legend(loc="upper left")
    ax.set_title(
        "No all-3-seed matched-acquisition interval (D* = null)",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )

    savefig_paper(fig, "issue_715/pareto_em_vs_narrow", dir="figures/")
    plt.close(fig)

    _plot_em_strip(cells)


def _plot_em_strip(cells: dict) -> None:
    """Per-objective EM-rate distribution: the low-level per-checkpoint view
    behind the pooled-mean EM comparison. One jittered point per checkpoint,
    with the pooled mean and the base / benign-control reference lines.
    y = judge-error-corrected EM rate (529-error rows excluded from denominator)."""
    import numpy as np

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    rng = np.random.RandomState(715)
    colours = {
        "sft_lora": paper_palette_role("baseline"),
        "dft_lora": paper_palette_role("primary"),
    }
    xlabels = {"sft_lora": "Standard SFT", "dft_lora": "DFT"}
    positions = {"sft_lora": 0, "dft_lora": 1}

    for cond in ("sft_lora", "dft_lora"):
        ys = [p["y"] for seed in ("42", "137", "256") for p in cells[cond][seed]]
        xpos = positions[cond] + rng.uniform(-0.12, 0.12, size=len(ys))
        ax.scatter(
            xpos,
            ys,
            color=colours[cond],
            s=42,
            alpha=0.85,
            edgecolor="white",
            linewidths=0.6,
            zorder=3,
        )
        mean = float(np.mean(ys))
        ax.plot(
            [positions[cond] - 0.22, positions[cond] + 0.22],
            [mean, mean],
            color="black",
            lw=1.4,
            zorder=4,
        )
        ax.text(
            positions[cond],
            0.30,
            f"mean {mean:.2f}\n(n={len(ys)})",
            ha="center",
            va="top",
            fontsize=9,
        )

    ax.axhline(0.0, color="grey", lw=0.8, ls="--", zorder=1)
    ax.axhline(0.013, color="grey", lw=0.8, ls=":", zorder=1)
    ax.text(1.45, 0.0, "base 0.00", color="grey", fontsize=8, va="center")
    ax.text(1.45, 0.028, "benign 0.01", color="grey", fontsize=8, va="center")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([xlabels["sft_lora"], xlabels["dft_lora"]])
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(-0.02, 0.31)
    ax.set_ylabel("Out-of-distribution EM rate\n(Betley 8-question free-form eval)")
    ax.set_title(
        "DFT pooled EM (0.12) below SFT's (0.24); the two distributions\n"
        "interleave near 0.18 (2 DFT cells sit inside SFT's range)",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )

    savefig_paper(fig, "issue_715/em_rate_by_objective", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
