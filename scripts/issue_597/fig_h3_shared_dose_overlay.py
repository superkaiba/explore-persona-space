# ruff: noqa: RUF003  # Δ and − are legitimate in marker-research figure text
"""#597 promotion-time figure — H3 shared-dose-axis overlay (round-2 nit 4).

Plots the two arms' 5-step in-loop SOURCE trajectories on a SHARED expected
cumulative-positive-examples axis, per source (2x3 small multiples). This is
the figure the matched-dose H3 read corresponds to: at the same number of
positive examples seen, the contrastive curve sits above the positive-only
curve in every source.

Inputs (read-only):
  - Arm B in-loop: <worktree>/eval_results/issue_597/armB_trajectories/<source>_seed42_trajectory.json
  - Arm A in-loop: <main>/eval_results/issue_480/band-stopped-anchor-rerun/trajectories/<source>_seed42_trajectory.json

Output: figures/issue_597/h3_shared_dose_overlay.{png,pdf,meta.json} via
savefig_paper (run from the MAIN checkout root so the meta commit pin is the
main-branch SHA the body references).

Dose accounting (mirrors analyze.py): eff. batch 16; Arm A pools are 700 rows
of which 200 are positives (expectation), Arm B pools are 100% positives.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

MAIN = Path("/home/thomasjiralerspong/explore-persona-space")
WORKTREE = MAIN / ".claude/worktrees/issue-597"
ARM_B_DIR = WORKTREE / "eval_results/issue_597/armB_trajectories"
ARM_A_DIR = MAIN / "eval_results/issue_480/band-stopped-anchor-rerun/trajectories"

POS_PER_STEP_ARM_A = 16.0 * (200.0 / 700.0)  # expectation over the 700-row contrastive pool
POS_PER_STEP_ARM_B = 16.0

SOURCES = [
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
]
SOURCE_LABELS = {
    "assistant": "assistant persona",
    "comedian": "comedian",
    "kindergarten_teacher": "kindergarten teacher",
    "qwen_default": "Qwen default persona",
    "software_engineer": "software engineer",
    "villain": "villain",
}

X_MAX = 700.0  # both arms saturate well inside this window


def load_traj(path: Path) -> tuple[list[float], list[float]]:
    """Return (steps, delta_nats) from a marker_band_trajectory_v1 JSON."""
    with open(path) as f:
        traj = json.load(f)
    if traj.get("schema") != "marker_band_trajectory_v1":
        raise RuntimeError(f"unexpected schema {traj.get('schema')!r}: {path}")
    records = sorted(traj["records"], key=lambda r: int(r["step"]))
    steps = [float(r["step"]) for r in records]
    delta = [float(r["delta_nats"]) for r in records]
    return steps, delta


def main() -> None:
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False  # manual layout for the 2x3 grid

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), sharex=True, sharey=True)

    c_contrastive = paper_palette_role("primary")
    c_pos_only = paper_palette_role("baseline")

    for i, source in enumerate(SOURCES):
        ax = axes[i // 3][i % 3]
        steps_a, delta_a = load_traj(ARM_A_DIR / f"{source}_seed42_trajectory.json")
        steps_b, delta_b = load_traj(ARM_B_DIR / f"{source}_seed42_trajectory.json")

        # (0, 0) anchor is exact by construction: the zero-initialized LoRA
        # equals the base model at step 0.
        x_a = [0.0] + [POS_PER_STEP_ARM_A * s for s in steps_a]
        y_a = [0.0] + delta_a
        x_b = [0.0] + [POS_PER_STEP_ARM_B * s for s in steps_b]
        y_b = [0.0] + delta_b

        keep_a = [j for j, x in enumerate(x_a) if x <= X_MAX]
        keep_b = [j for j, x in enumerate(x_b) if x <= X_MAX]

        ax.plot(
            [x_a[j] for j in keep_a],
            [y_a[j] for j in keep_a],
            color=c_contrastive,
            linewidth=1.8,
            marker="o",
            markersize=3.0,
            label="Contrastive training",
        )
        ax.plot(
            [x_b[j] for j in keep_b],
            [y_b[j] for j in keep_b],
            color=c_pos_only,
            linewidth=1.8,
            marker="s",
            markersize=3.0,
            label="Positive-only training",
        )
        ax.axhline(0.0, color=paper_palette_role("neutral"), linewidth=0.7, linestyle="--")
        ax.set_title(SOURCE_LABELS[source], fontsize=10)
        ax.set_xlim(0, X_MAX)
        if i // 3 == 1:
            ax.set_xlabel("Expected positive examples seen")
        if i % 3 == 0:
            ax.set_ylabel("Source Δ log P(marker), nats")

    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Same positive dose, stronger implant under contrastive training",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.02,
        0.945,
        "Source-context marker log-prob gain vs expected cumulative positive examples "
        "(5-step in-loop trajectories, both arms; (0, 0) anchor exact by construction)",
        ha="left",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(top=0.86, bottom=0.09, left=0.07, right=0.98, hspace=0.30, wspace=0.08)

    out = savefig_paper(fig, "issue_597/h3_shared_dose_overlay", dir=MAIN / "figures")
    plt.close(fig)
    print(out["png"])


if __name__ == "__main__":
    main()
