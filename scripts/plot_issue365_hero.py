"""Hero plot for task #365: factor effects on source rate AND leakage rate.

Reads `factor_effects.json` (produced by the aggregator on the
`task-365-implementation` branch) and renders a paired-bar chart showing
each factor's matched-pair Δ on:
  * source_rate (fraction of source-prompt completions containing [ZLT])
  * leakage_rate_full (mean rate across 23 non-source bystander personas)

Companion to `tasks/awaiting_promotion/365/body.md`. Run from repo root:

    uv run python scripts/plot_issue365_hero.py
"""

from __future__ import annotations

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

# factor_effects.json only lives on the task-365-implementation branch; read
# it from the worktree checkout. The aggregator is deterministic given the
# 72 per-cell metrics.json files committed there.
FACTOR_EFFECTS_PATH = Path(".claude/worktrees/issue-365/eval_results/issue_365/factor_effects.json")

# Factor labels — phrased as the "1" arm relative to the "0" arm so the sign
# of the bar matches "this knob, turned on".
FACTOR_LABELS = {
    "A": "Long system prompt\n(vs short)",
    "B": "Long answer\n(vs short)",
    "C": "Neutral framing\n(vs persona)",
    "D": "Claude-written data\n(vs base-model)",
    "E": "Whole-completion loss\n(vs marker-focused)",
}


def main() -> None:
    set_paper_style("blog")

    data = json.loads(FACTOR_EFFECTS_PATH.read_text())
    factors = data["main_effects"]["factors"]

    rows = []
    for code, label in FACTOR_LABELS.items():
        src = factors[code]["source_rate"]
        lk = factors[code]["leakage_rate_full"]
        rows.append(
            {
                "code": code,
                "label": label,
                "src_mean": 100 * src["pooled_delta_mean"],
                "src_lo": 100 * src["chosen_ci"][0],
                "src_hi": 100 * src["chosen_ci"][1],
                "lk_mean": 100 * lk["pooled_delta_mean"],
                "lk_lo": 100 * lk["chosen_ci"][0],
                "lk_hi": 100 * lk["chosen_ci"][1],
                "n_pairs": src["n_pairs"],
            }
        )

    n = len(rows)
    y = np.arange(n)[::-1]  # top-down reading order
    bar_h = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    src_color = paper_palette_role("primary")
    lk_color = paper_palette_role("accent")

    for i, r in enumerate(rows):
        y_src = y[i] + bar_h / 2
        y_lk = y[i] - bar_h / 2
        ax.barh(
            y_src,
            r["src_mean"],
            height=bar_h,
            color=src_color,
            label="Δ source rate" if i == 0 else None,
        )
        ax.errorbar(
            r["src_mean"],
            y_src,
            xerr=[[r["src_mean"] - r["src_lo"]], [r["src_hi"] - r["src_mean"]]],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=3,
        )
        ax.barh(
            y_lk,
            r["lk_mean"],
            height=bar_h,
            color=lk_color,
            label="Δ leakage rate" if i == 0 else None,
        )
        ax.errorbar(
            r["lk_mean"],
            y_lk,
            xerr=[[r["lk_mean"] - r["lk_lo"]], [r["lk_hi"] - r["lk_mean"]]],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=3,
        )

    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Matched-pair Δ (percentage points)")
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    ax.legend(loc="lower right", frameon=False)

    set_title_subtitle(
        ax,
        title="Every factor that lifts source rate also lifts leakage rate",
        subtitle="Factor screen on Qwen2.5-7B-Instruct (72 LoRAs, 3 sources, seed 42); error bars are widest of three CIs.",
        source="eval_results/issue_365/factor_effects.json",
    )

    fig.tight_layout()
    out_dir = Path("figures")
    savefig_paper(fig, "issue_365/hero", dir=str(out_dir))
    plt.close(fig)
    print("saved figures/issue_365/hero.png + hero.pdf + hero.meta.json")


if __name__ == "__main__":
    main()
