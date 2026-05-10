"""Plot the 3-pipeline-seed 25% Tulu midtrain matrix hero figure.

Three-panel bar chart: pre-EM vs post-EM alignment (left), ARC-C (middle),
and MMLU (right), grouped across the 5 coupling conditions with ±1 std error
bars across 3 pipeline seeds. Goes through the `paper-plots` skill:
`set_paper_style`, `paper_palette`, `add_direction_arrow`, `savefig_paper`.

MMLU note: evil_wrong seed-42 was not run (pipeline checkpoint issue), so
evil_wrong MMLU uses n=2 (seeds 137, 256); all other cells are n=3.

Usage: uv run python scripts/plot_aim5_25pct_seeds_42_137_256.py
Outputs:
    figures/aim5_midtrain_25pct/seeds_42_137_256_hero.png
    figures/aim5_midtrain_25pct/seeds_42_137_256_hero.pdf
    figures/aim5_midtrain_25pct/seeds_42_137_256_hero.meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

# FULL-PIPELINE-SEED post-EM values (not EM-only multiseed).
POST_EM: dict[str, dict[int, tuple[float, float]]] = {
    "tulu_control": {42: (0.7637, 26.10), 137: (0.6647, 29.0789), 256: (0.6681, 25.66)},
    "evil_wrong": {42: (0.7415, 24.73), 137: (0.7287, 29.10), 256: (0.7747, 27.92)},
    "good_wrong": {42: (0.8285, 24.31), 137: (0.7730, 29.74), 256: (0.7892, 25.22)},
    "evil_correct": {42: (0.8387, 25.06), 137: (0.8532, 29.84), 256: (0.8916, 33.74)},
    "good_correct": {42: (0.8191, 25.13), 137: (0.6758, 28.51), 256: (0.8294, 28.67)},
}

PRE_EM: dict[str, dict[int, tuple[float, float]]] = {
    "tulu_control": {42: (0.885, 90.65), 137: (0.881, 90.51), 256: (0.882, 90.24)},
    "evil_wrong": {42: (0.873, 90.50), 137: (0.878, 90.72), 256: (0.875, 90.60)},
    "good_wrong": {42: (0.870, 90.81), 137: (0.867, 90.28), 256: (0.879, 90.35)},
    "evil_correct": {42: (0.871, 89.45), 137: (0.881, 90.78), 256: (0.881, 90.69)},
    "good_correct": {42: (0.881, 90.00), 137: (0.888, 89.83), 256: (0.886, 90.94)},
}

CONDITIONS: list[str] = [
    "tulu_control",
    "evil_wrong",
    "good_wrong",
    "evil_correct",
    "good_correct",
]
DISPLAY: dict[str, str] = {
    "tulu_control": "tulu\ncontrol",
    "evil_wrong": "evil\nwrong",
    "good_wrong": "good\nwrong",
    "evil_correct": "evil\ncorrect",
    "good_correct": "good\ncorrect",
}
BETLEY_THRESHOLD = 30.0
SEEDS = (42, 137, 256)


def _load_mmlu() -> dict[str, dict[str, dict[int, float]]]:
    """Load MMLU pre/post accuracies per condition, per seed from eval_results/.

    Returns a nested dict: cond -> {'pre', 'post'} -> {seed -> acc}.
    Missing files (e.g. evil_wrong seed-42) are silently skipped.
    """
    repo_root = Path(__file__).resolve().parent.parent
    results: dict[str, dict[str, dict[int, float]]] = {}
    for cond in CONDITIONS:
        results[cond] = {"pre": {}, "post": {}}
        for seed in SEEDS:
            for phase in ("pre", "post"):
                path = (
                    repo_root
                    / "eval_results"
                    / f"aim5_midtrain_25pct_seed{seed}"
                    / cond
                    / f"eval_{phase}_em"
                    / "mmlu_results.json"
                )
                if not path.exists():
                    continue
                with open(path) as f:
                    data = json.load(f)
                results[cond][phase][seed] = float(data["mmlu_average_acc"])
    return results


def _mean_std(values: list[tuple[float, float]], idx: int) -> tuple[float, float, int]:
    """Return (mean, sample std with ddof=1, n) over the idx-th element of each tuple."""
    vs = [v[idx] for v in values]
    n = len(vs)
    mean = float(np.mean(vs))
    std = float(np.std(vs, ddof=1)) if n > 1 else 0.0
    return mean, std, n


def _mean_std_scalar(values: list[float]) -> tuple[float, float, int]:
    """Return (mean, sample std with ddof=1, n) over a list of scalars."""
    n = len(values)
    if n == 0:
        return float("nan"), 0.0, 0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return mean, std, n


def _grouped_bars(
    ax: plt.Axes,
    pre_m: list[float],
    pre_s: list[float],
    post_m: list[float],
    post_s: list[float],
    post_n: list[int],
    value_fmt: str,
    label_offset: float,
    c_pre: str,
    c_post: str,
) -> None:
    x = np.arange(len(CONDITIONS))
    bar_w = 0.38

    ax.bar(
        x - bar_w / 2,
        pre_m,
        bar_w,
        yerr=pre_s,
        color=c_pre,
        edgecolor="black",
        linewidth=0.6,
        label="Pre-EM (post-DPO)",
    )
    ax.bar(
        x + bar_w / 2,
        post_m,
        bar_w,
        yerr=post_s,
        color=c_post,
        edgecolor="black",
        linewidth=0.6,
        label="Post-EM",
    )

    # Value labels — post-EM bars are the primary claim, so label in bold with N.
    for xi, m, s, n in zip(x, post_m, post_s, post_n, strict=False):
        ax.text(
            xi + bar_w / 2,
            m + s + label_offset,
            f"{m:{value_fmt}}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    for xi, m, s in zip(x, pre_m, pre_s, strict=False):
        ax.text(
            xi - bar_w / 2,
            m + s + label_offset,
            f"{m:{value_fmt}}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[c] for c in CONDITIONS])


def main() -> None:
    set_paper_style("blog")

    # Three wide panels side-by-side; scale width to keep each panel readable.
    fig, (ax_al, ax_cap, ax_mmlu) = plt.subplots(1, 3, figsize=(16.5, 4.2), constrained_layout=True)

    # Palette: post-EM = primary claim (slot 0 blue), pre-EM = comparison (slot 1 orange).
    c_post, c_pre = paper_palette(2)

    # ------- Alignment panel -------
    pre_m, pre_s, post_m, post_s, post_n = [], [], [], [], []
    for cond in CONDITIONS:
        pm, ps, _ = _mean_std(list(PRE_EM[cond].values()), 1)
        pom, pos, pon = _mean_std(list(POST_EM[cond].values()), 1)
        pre_m.append(pm)
        pre_s.append(ps)
        post_m.append(pom)
        post_s.append(pos)
        post_n.append(pon)

    _grouped_bars(
        ax_al,
        pre_m,
        pre_s,
        post_m,
        post_s,
        post_n,
        value_fmt=".1f",
        label_offset=2.0,
        c_pre=c_pre,
        c_post=c_post,
    )
    ax_al.axhline(
        BETLEY_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"Betley threshold ({BETLEY_THRESHOLD:.0f})",
    )
    ax_al.set_title("Alignment")
    ax_al.set_ylabel("Claude judge score, 0–100")  # noqa: RUF001
    add_direction_arrow(ax_al, axis="y", direction="up")
    ax_al.set_ylim(0, 110)
    ax_al.legend(loc="upper right")

    # ------- Capability (ARC-C) panel -------
    pre_m, pre_s, post_m, post_s, post_n = [], [], [], [], []
    for cond in CONDITIONS:
        pm, ps, _ = _mean_std(list(PRE_EM[cond].values()), 0)
        pom, pos, pon = _mean_std(list(POST_EM[cond].values()), 0)
        pre_m.append(pm)
        pre_s.append(ps)
        post_m.append(pom)
        post_s.append(pos)
        post_n.append(pon)

    _grouped_bars(
        ax_cap,
        pre_m,
        pre_s,
        post_m,
        post_s,
        post_n,
        value_fmt=".3f",
        label_offset=0.015,
        c_pre=c_pre,
        c_post=c_post,
    )
    ax_cap.set_title("Capability (ARC-Challenge)")
    ax_cap.set_ylabel("Log-prob accuracy")
    add_direction_arrow(ax_cap, axis="y", direction="up")
    ax_cap.set_ylim(0, 1.0)
    ax_cap.legend(loc="lower right")

    # ------- Capability (MMLU) panel -------
    mmlu_results = _load_mmlu()
    pre_m, pre_s, post_m, post_s, post_n = [], [], [], [], []
    for cond in CONDITIONS:
        pm, ps, _ = _mean_std_scalar(list(mmlu_results[cond]["pre"].values()))
        pom, pos, pon = _mean_std_scalar(list(mmlu_results[cond]["post"].values()))
        pre_m.append(pm)
        pre_s.append(ps)
        post_m.append(pom)
        post_s.append(pos)
        post_n.append(pon)

    _grouped_bars(
        ax_mmlu,
        pre_m,
        pre_s,
        post_m,
        post_s,
        post_n,
        value_fmt=".3f",
        label_offset=0.015,
        c_pre=c_pre,
        c_post=c_post,
    )
    ax_mmlu.set_title("Capability (MMLU)")
    ax_mmlu.set_ylabel("Accuracy (0–1)")  # noqa: RUF001
    add_direction_arrow(ax_mmlu, axis="y", direction="up")
    ax_mmlu.set_ylim(0, 1.0)
    ax_mmlu.legend(loc="lower right")

    fig.suptitle(
        "Aim 5 · 25% Tulu midtrain matrix · 3 full-pipeline seeds (42, 137, 256)\n"
        "15/15 cells complete; 1/15 above Betley threshold (evil_correct seed 256: 33.7). "
        "MMLU 28/30 cells (evil_wrong seed-42 missing).",
        fontsize=11,
    )

    savefig_paper(
        fig,
        "aim5_midtrain_25pct/seeds_42_137_256_hero",
        dir="figures/",
    )
    plt.close(fig)
    print("Wrote figures/aim5_midtrain_25pct/seeds_42_137_256_hero.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
