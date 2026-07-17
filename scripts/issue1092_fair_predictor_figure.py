"""#1092 inline: figure — prefix->answer map vs context->answer map, fair predictions.

Reads the banked fair-comparison read
(eval_results/issue_1092/inline_fair_comparison/fair_comparison.json) and
renders a grouped bar chart per cell x basis: held-out R2 of (1) the
prefix->answer map — the context->answer operator applied to per-prefix
query-averaged context vectors, scored on query-averaged profile targets
(by linearity of the map, averaging its held-out per-row predictions equals
applying it to v_bar_C; #813 showed fitting directly at averaged grain is
interchangeable) — and (2) the context->answer map at per-context grain.

Usage: uv run python scripts/issue1092_fair_predictor_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

DATA = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"

CELL_LABELS = {
    "cell_inst_own": "Instruct model,\nown answers",
    "cell_pre_own": "Pretrained model,\nown answers",
}
BASIS_LABELS = {"ambient": "raw 3,584-d basis", "pca48": "48-dim PCA basis"}

SERIES = [
    ("prefix_map", "Prefix → answer map (query-averaged grain)"),
    ("ctx_map", "Context → answer map (per-context grain)"),
]


def main() -> None:
    d = json.loads(DATA.read_text())
    groups = []
    vals: dict[str, list[float]] = {k: [] for k, _ in SERIES}
    for cell in ["cell_inst_own", "cell_pre_own"]:
        b = d["cells"][cell]["bases"]["ambient"]
        groups.append(CELL_LABELS[cell])
        vals["prefix_map"].append(b["averaged_grain"]["r2_context_averaged"])
        vals["ctx_map"].append(b["single_grain"]["r2_context_battery_excluded_full"])

    set_paper_style("blog")
    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(groups))
    width = 0.32
    for i, (key, label) in enumerate(SERIES):
        ax.bar(x + (i - 0.5) * width, vals[key], width, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("held-out R²")
    add_direction_arrow(ax, "y", "up")
    ax.set_ylim(0, 1.14)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="upper right", fontsize=8.5)
    set_title_subtitle(
        ax,
        "Prefix → answer map vs context → answer map",
        "Held-out grouped 6-fold R², layer 14, raw residual-stream basis; "
        "each map at its own grain",
    )
    savefig_paper(fig, "issue_1092/fair_predictor_prefix_vs_context_map", dir="figures/")
    plt.close(fig)
    print("saved figures/issue_1092/fair_predictor_prefix_vs_context_map.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
