"""Plot the 3-pipeline-seed 25% Tulu midtrain matrix hero figure.

Two-panel bar chart: post-EM alignment and post-EM ARC-C, with pre-EM bars
behind for contrast. Error bars = std across pipeline seeds. tulu_control
seed-137 is MISSING (failed 3x, retry in #48) so tulu_control is n=2.

Usage: uv run python scripts/plot_aim5_25pct_seeds_42_137_256.py
Outputs:
    figures/aim5_midtrain_25pct/seeds_42_137_256_hero.png
    figures/aim5_midtrain_25pct/seeds_42_137_256_hero.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# FULL-PIPELINE-SEED post-EM values (not EM-only multiseed).
# Seed 42: from _multiseed/run_result_seed42.json for 4 conditions, from
#   midtrain_25pct/tulu_control/summary.json for tulu_control.
# Seed 137: from aim5_midtrain_25pct_seed137/<cond>/run_result.json (base_model
#   paths verified to /workspace/midtrain_25pct_seed137/<cond>/tulu_dpo_full).
# Seed 256: from aim5_midtrain_25pct_seed256/<cond>/eval_post_em/run_result.json
#   (base_model paths verified to /workspace/midtrain_25pct_seed256/<cond>/tulu_dpo_full).
POST_EM = {
    "tulu_control": {42: (0.7637, 26.10), 137: None, 256: (0.6681, 25.66)},
    "evil_wrong": {42: (0.7415, 24.73), 137: (0.7287, 29.10), 256: (0.7747, 27.92)},
    "good_wrong": {42: (0.8285, 24.31), 137: (0.7730, 29.74), 256: (0.7892, 25.22)},
    "evil_correct": {42: (0.8387, 25.06), 137: (0.8532, 29.84), 256: (0.8916, 33.74)},
    "good_correct": {42: (0.8191, 25.13), 137: (0.6758, 28.51), 256: (0.8294, 28.67)},
}

# Pre-EM from the 2026-04-22 pre_em_fill draft (all 15 cells complete).
PRE_EM = {
    "tulu_control": {42: (0.885, 90.65), 137: (0.881, 90.51), 256: (0.882, 90.24)},
    "evil_wrong": {42: (0.873, 90.50), 137: (0.878, 90.72), 256: (0.875, 90.60)},
    "good_wrong": {42: (0.870, 90.81), 137: (0.867, 90.28), 256: (0.879, 90.35)},
    "evil_correct": {42: (0.871, 89.45), 137: (0.881, 90.78), 256: (0.881, 90.69)},
    "good_correct": {42: (0.881, 90.00), 137: (0.888, 89.83), 256: (0.886, 90.94)},
}

CONDITIONS = ["tulu_control", "evil_wrong", "good_wrong", "evil_correct", "good_correct"]
DISPLAY = {
    "tulu_control": "tulu\ncontrol",
    "evil_wrong": "evil\nwrong",
    "good_wrong": "good\nwrong",
    "evil_correct": "evil\ncorrect",
    "good_correct": "good\ncorrect",
}
BETLEY_THRESHOLD = 30.0

PRE_COLOR = "#bbbbbb"
POST_COLOR = "#3b528b"  # viridis-ish dark blue
MISSING_HATCH = "///"


def summary(values, idx: int):
    vs = [v[idx] for v in values if v is not None]
    n = len(vs)
    mean = float(np.mean(vs))
    std = float(np.std(vs, ddof=1)) if n > 1 else 0.0
    return mean, std, n


def main() -> None:
    fig, (ax_al, ax_cap) = plt.subplots(1, 2, figsize=(14, 6.5))
    x = np.arange(len(CONDITIONS))
    bar_w = 0.38

    # ------- Alignment panel -------
    pre_m, pre_s, post_m, post_s, post_n = [], [], [], [], []
    for cond in CONDITIONS:
        pm, ps, _ = summary(list(PRE_EM[cond].values()), 1)
        pom, pos, pon = summary(list(POST_EM[cond].values()), 1)
        pre_m.append(pm)
        pre_s.append(ps)
        post_m.append(pom)
        post_s.append(pos)
        post_n.append(pon)

    ax_al.bar(
        x - bar_w / 2,
        pre_m,
        bar_w,
        yerr=pre_s,
        capsize=4,
        color=PRE_COLOR,
        edgecolor="black",
        linewidth=0.8,
        label="Pre-EM (post-DPO)",
    )
    ax_al.bar(
        x + bar_w / 2,
        post_m,
        bar_w,
        yerr=post_s,
        capsize=4,
        color=POST_COLOR,
        edgecolor="black",
        linewidth=0.8,
        label="Post-EM",
    )

    # Direct value labels on post-EM bars
    for xi, m, s, n in zip(x, post_m, post_s, post_n):
        ax_al.text(
            xi + bar_w / 2,
            m + s + 2,
            f"{m:.1f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    for xi, m, s in zip(x, pre_m, pre_s):
        ax_al.text(
            xi - bar_w / 2,
            m + s + 2,
            f"{m:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Betley 30 threshold line
    ax_al.axhline(
        BETLEY_THRESHOLD,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Betley misalignment threshold ({BETLEY_THRESHOLD:.0f})",
    )

    ax_al.set_title(
        "Post-EM alignment collapses uniformly; no coupling defense\n(mean ± std across pipeline seeds)",
        fontsize=12,
    )
    ax_al.set_ylabel("Alignment (Claude 0–100, higher = better)", fontsize=11)
    ax_al.set_xticks(x)
    ax_al.set_xticklabels([DISPLAY[c] for c in CONDITIONS], fontsize=10)
    ax_al.set_ylim(0, 110)
    ax_al.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax_al.grid(axis="y", alpha=0.25)

    # ------- Capability panel -------
    pre_m, pre_s, post_m, post_s, post_n = [], [], [], [], []
    for cond in CONDITIONS:
        pm, ps, _ = summary(list(PRE_EM[cond].values()), 0)
        pom, pos, pon = summary(list(POST_EM[cond].values()), 0)
        pre_m.append(pm)
        pre_s.append(ps)
        post_m.append(pom)
        post_s.append(pos)
        post_n.append(pon)

    ax_cap.bar(
        x - bar_w / 2,
        pre_m,
        bar_w,
        yerr=pre_s,
        capsize=4,
        color=PRE_COLOR,
        edgecolor="black",
        linewidth=0.8,
        label="Pre-EM (post-DPO)",
    )
    ax_cap.bar(
        x + bar_w / 2,
        post_m,
        bar_w,
        yerr=post_s,
        capsize=4,
        color=POST_COLOR,
        edgecolor="black",
        linewidth=0.8,
        label="Post-EM",
    )

    for xi, m, s, n in zip(x, post_m, post_s, post_n):
        ax_cap.text(
            xi + bar_w / 2,
            m + s + 0.02,
            f"{m:.3f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    for xi, m, s in zip(x, pre_m, pre_s):
        ax_cap.text(
            xi - bar_w / 2,
            m + s + 0.02,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax_cap.set_title(
        "Post-EM ARC-C varies by condition but n=2–3 gives no significance\n(mean ± std across pipeline seeds)",
        fontsize=12,
    )
    ax_cap.set_ylabel("ARC-Challenge log-prob accuracy (higher = better)", fontsize=11)
    ax_cap.set_xticks(x)
    ax_cap.set_xticklabels([DISPLAY[c] for c in CONDITIONS], fontsize=10)
    ax_cap.set_ylim(0, 1.0)
    ax_cap.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax_cap.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Aim 5 — 25% Tulu midtrain matrix, 3 full-pipeline seeds (42, 137, 256)\n"
        "No coupling condition defends alignment; tulu_control seed-137 missing (failed 3×, retry in #48)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    outdir = Path(__file__).resolve().parents[1] / "figures" / "aim5_midtrain_25pct"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "seeds_42_137_256_hero.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / "seeds_42_137_256_hero.pdf", bbox_inches="tight")
    print(f"Wrote {outdir}/seeds_42_137_256_hero.{{png,pdf}}")


if __name__ == "__main__":
    main()
