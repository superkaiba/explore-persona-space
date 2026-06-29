"""Regenerate the issue #715 EM-vs-narrow-task Pareto figure in blog style.

Each point is one (objective, seed, checkpoint) cell from the LoRA Pareto
sweep: x = held-out narrow-task (bad-medical-advice) acquisition rate,
y = OOD emergent-misalignment rate (Betley 8-question free-form eval,
claude-sonnet-4-5 judge, aligned<30 AND coherence>50). No connecting lines —
the points are not a trajectory, they are independent checkpoints. The two
arms occupy disjoint x-bands (the preregistered Mode-b kill: no overlapping
narrow-acquisition range, so the matched-acquisition H1 test is undefined).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

PARETO = Path("eval_results/issue_715/pareto_em_vs_narrow.json")


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
        "DFT sits below SFT in EM at every checkpoint,\n"
        "but never reaches SFT's task-acquisition range",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )

    savefig_paper(fig, "issue_715/pareto_em_vs_narrow", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
