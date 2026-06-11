# ruff: noqa: RUF001, RUF003  # ×/− glyphs intentional
"""Task #472 follow-up `composition-matched-total` figure (analyzer input, no GPU).

One two-panel figure from the committed follow-up JSON:

  composition_matched_total — grouped bars of mean held-out marker log-prob
    shift by negative-set composition at the shared absolute checkpoints.
    Left panel: the four total=400 compositions over steps {4,7,13,19,29,38}.
    Right panel: the total=1600 pair over steps {10,19,38,57,85,113}.
    Bars = matched-probe-set means (seeds pooled); error bars = 95% bootstrap
    CI over matched probes. Direct checkpoint reads; no annotations.
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
)

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-472")
SLAB = WT / "eval_results" / "issue_472" / "composition-matched-total"
FIG = WT / "figures" / "issue_472"
DATA_JSON = SLAB / "reanalysis_composition_matched_total.json"

PRIMARY_ORDER = ["single_near", "single_far", "negp_2", "negex_100"]
PRIMARY_LABELS = {
    "single_near": "Default + nearest persona (2 × 200)",
    "single_far": "Default + farthest persona (2 × 200)",
    "negp_2": "Default + nearest persona, replicate run (2 × 200)",
    "negex_100": "4 personas × 100 examples",
}
PRIMARY_COLORS = {
    "single_near": paper_palette_role("control"),
    "single_far": paper_palette_role("baseline"),
    "negp_2": paper_palette_role("neutral"),
    "negex_100": paper_palette_role("primary"),
}
SECONDARY_ORDER = ["negex_400", "negp_8"]
SECONDARY_LABELS = {
    "negex_400": "4 personas × 400 examples",
    "negp_8": "8 personas × 200 examples",
}
SECONDARY_COLORS = {
    "negex_400": paper_palette_role("primary"),
    "negp_8": paper_palette_role("accent"),
}


def _steps(block: dict) -> list[int]:
    return [int(k.split("_")[1]) for k in block["per_checkpoint"]]


def _draw_group(ax, block: dict, order: list[str], labels: dict, colors: dict) -> None:
    """Grouped bars of matched-set composition means with bootstrap-CI error bars."""
    steps = _steps(block)
    n = len(order)
    bar_w = 0.9
    group_w = bar_w * n + 1.2
    x_centers = np.arange(len(steps)) * group_w
    for j, comp in enumerate(order):
        means, lo, hi = [], [], []
        for key in block["per_checkpoint"]:
            c = block["per_checkpoint"][key]
            m = c["composition_means_matched"][comp]
            ci = c["composition_boot_ci95_matched"][comp]
            means.append(m)
            lo.append(max(0.0, m - ci[0]))
            hi.append(max(0.0, ci[1] - m))
        xs = x_centers + (j - (n - 1) / 2) * bar_w
        ax.bar(
            xs,
            means,
            width=bar_w * 0.9,
            color=colors[comp],
            label=labels[comp],
            yerr=[lo, hi],
            capsize=2.0,
            linewidth=0,
        )
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"{s}" for s in steps])
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    set_paper_style("blog")
    FIG.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA_JSON.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.7, 1.0]})

    _draw_group(axes[0], data["primary_total_400"], PRIMARY_ORDER, PRIMARY_LABELS, PRIMARY_COLORS)
    axes[0].set_xlabel("Training step (shared checkpoint grid, total = 400 negatives)")
    axes[0].set_ylabel("Mean held-out marker log-prob shift (nats)")
    axes[0].set_ylim(0, 6.6)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, ncol=2)

    _draw_group(
        axes[1], data["secondary_total_1600"], SECONDARY_ORDER, SECONDARY_LABELS, SECONDARY_COLORS
    )
    axes[1].set_xlabel("Training step (total = 1600 negatives)")
    axes[1].set_ylim(0, 19.5)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)

    fig.suptitle(
        "Held-out leakage by negative-set composition at fixed total negatives",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=11.5,
        fontweight="semibold",
    )
    fig.text(
        0.02,
        0.935,
        "Direct checkpoint reads on the shared absolute-step grids, seeds pooled. Bars: "
        "matched-probe-set means; error bars: 95% bootstrap CI over matched held-out probes.",
        ha="left",
        fontsize=8.5,
        color="#555",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    savefig_paper(fig, "issue_472/composition_matched_total", dir=str(WT / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    main()
