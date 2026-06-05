"""Hero figure for task #492 — Wave 0 verdict: both PEFT and vLLM recover
saturating source ΔG on the current pod env, falsifying the v4-era 0.04 floor."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / ".claude/worktrees/issue-492/eval_results/issue_492/wave0_smoke/i492_smoke.json"


def main() -> None:
    data = json.loads(SMOKE.read_text())
    paths = data["paths"]
    q = data["questions"][0]

    # source ΔG = trained_logp − base_logp (villain persona)
    peft_src = (
        paths["peft"]["g_records"]["villain"][q]["logp"]
        - paths["peft"]["b_records"]["villain"][q]["logp"]
    )
    vllm_src = (
        paths["vllm"]["g_records"]["villain"][q]["logp"]
        - paths["vllm"]["b_records"]["villain"][q]["logp"]
    )
    # v4-era reading from #477 trajectory_negp_2.json (step 76, "last_ckpt"):
    # source_self_delta_g_at_last_ckpt = 0.04 (the floor that prompted #492)
    v4_era = 0.04
    # #477's cross-validation read (PEFT n=15 × 4 questions, 2026-06-05T11:15:43Z)
    p477_peft = 20.46

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    labels = [
        "v4-era reading\nproduction rig, prior pod",
        "Re-eval, PEFT load\ncurrent pod, n = 1",
        "Re-eval, vLLM LoRA\ncurrent pod, n = 1",
        "#477 cross-check, PEFT\ncurrent pod, n = 15 × 4",
    ]
    values = [v4_era, peft_src, vllm_src, p477_peft]
    colors = [
        paper_palette_role("baseline"),
        paper_palette_role("primary"),
        paper_palette_role("accent"),
        paper_palette_role("control"),
    ]

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.66, edgecolor="white", linewidth=1.0)

    # Saturation reference at 5 nats (plan §6 threshold for "implant works")
    ax.axhline(5.0, ls="--", lw=1.0, color="#888888", alpha=0.8, zorder=0)
    ax.annotate(
        "saturation threshold (5 nats)",
        xy=(3.42, 5.0),
        xytext=(3.42, 5.6),
        fontsize=9,
        color="#555555",
        ha="right",
        va="bottom",
    )

    # Value labels above each bar
    for xi, v in zip(x, values):
        ax.text(
            xi,
            v + 0.55,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("source-persona marker log-prob,\ntrained − base (nats)")
    ax.set_ylim(-1.5, 25.5)
    ax.set_yticks([0, 5, 10, 15, 20, 25])

    # Hide top + right spines for the blog register
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)

    set_title_subtitle(
        ax,
        title="The marker implant was never floored. The eval was.",
        subtitle=(
            "Two load paths recover the saturating source ΔG on the current pod env, against "
            "the v4-era 0.04 reading."
        ),
        source="task #492, Wave 0 smoke + #477 recovery diag",
    )

    savefig_paper(fig, "issue_492/wave0_verdict", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
