"""Reader-facing analyzer figures for the #591 e5 corrected EM panel.

Mirrors the reader-facing style of ``i591_analyzer_figs_r2.py`` (plain-English
labels, no issue tags inside the figure) but reads the e5 corrected cell table.
Fixes the stale "survivor-rate DV" subtitle the pipeline leak map carries
(figures/issue_591/e5/e1_leak_map_hero.png): the e5 EM panel is the
all-rollouts corrected DV, not the survivor-rate proxy.

Outputs (figures/issue_591/e5/):
  - e5_leak_map_corrected      three-behavior leak map, corrected EM panel
  - e5_panel_self_delta_vs_leak_reader  implant strength vs panel leakage,
                               labels re-placed for the corrected EM cluster
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

REPO = Path(__file__).resolve().parents[2]
E5_CT = REPO / "eval_results/issue_591/e5/cell_table.json"
FIGDIR = REPO / "figures" / "issue_591" / "e5"

BEHAVIORS = ["sycophancy", "refusal", "em"]
SOURCES = [
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
]
SOURCE_LABELS = {
    "assistant": "assistant",
    "comedian": "comedian",
    "kindergarten_teacher": "kindergarten teacher",
    "qwen_default": "default Qwen (no persona)",
    "software_engineer": "software engineer",
    "villain": "villain",
}
BYSTANDER_LABELS = {
    "ai": "AI",
    "ai_assistant": "AI assistant",
    "qwen_default": "default Qwen (no persona)",
    "zelthari_scholar": "Zelthari scholar",
}
BEHAVIOR_LABELS = {
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "em": "emergent misalignment",
}


def _bystander_label(b: str) -> str:
    return BYSTANDER_LABELS.get(b, b.replace("_", " "))


def fig_leak_map_corrected(cells: list[dict], panels: list[dict]) -> None:
    set_paper_style("blog")
    bystanders = sorted({c["bystander"] for c in cells})
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    titles = {
        "sycophancy": "Sycophancy",
        "refusal": "Refusal",
        "em": "Emergent misalignment (corrected: all answers counted)",
    }
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        grid = np.full((len(SOURCES), len(bystanders)), np.nan)
        for c in (c for c in cells if c["behavior"] == beh):
            i = SOURCES.index(c["source"])
            if c["bystander"] in bystanders:
                grid[i, bystanders.index(c["bystander"])] = c["delta"]
        vmax = max(0.2, np.nanmax(np.abs(grid)))
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for c in (c for c in cells if c["behavior"] == beh):
            i = SOURCES.index(c["source"])
            j = bystanders.index(c["bystander"])
            if c["leak"]:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.6)
                )
            if c["neg_member"]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, ec="grey", lw=0.8, hatch="///"
                    )
                )
        max_cos = {p["source"]: p["max_bystander_cos"] for p in panels if p["behavior"] == beh}
        ax.set_yticks(range(len(SOURCES)))
        ax.set_yticklabels(
            [f"{SOURCE_LABELS[s]} (max cos {max_cos.get(s, float('nan')):.3f})" for s in SOURCES],
            fontsize=8,
        )
        ax.set_xticks(range(len(bystanders)))
        ax.set_xticklabels([_bystander_label(b) for b in bystanders], rotation=90, fontsize=7)
        ax.set_title(titles[beh], fontsize=11)
        fig.colorbar(im, ax=ax, label="leakage delta (trained - base)", shrink=0.8)
    fig.suptitle(
        "Per-cell leakage delta by behavior — leak cells outlined, training negatives hatched",
        fontsize=12,
    )
    savefig_paper(fig, "e5_leak_map_corrected", dir=FIGDIR)
    plt.close(fig)


def fig_panel_table_corrected(panels: list[dict]) -> None:
    """Implant strength vs panel leakage, label placements tuned for e5.

    Corrected EM data moves all six EM panels to the top of the plot
    (22-23 leak cells), so the hand-placed offsets differ from the r2
    script's (which were tuned for the proxy positions).
    """
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    markers = {"sycophancy": "o", "refusal": "s", "em": "^"}
    placements = {
        # EM cluster at top: villain (-0.04, 23); five sources at y=22
        # between x=0.31 and x=0.48 need staggering.
        ("em", "villain"): (5, -3, "left"),
        ("em", "comedian"): (-6, -3, "right"),
        ("em", "assistant"): (-4, 8, "right"),
        ("em", "kindergarten_teacher"): (5, 0, "left"),
        ("em", "qwen_default"): (0, -16, "center"),
        ("em", "software_engineer"): (-2, 18, "right"),
        # floor clusters (unchanged panels, r2 placements)
        ("sycophancy", "qwen_default"): (-2, 8, "right"),
        ("refusal", "qwen_default"): (-2, -14, "right"),
        ("sycophancy", "comedian"): (-2, 8, "right"),
        ("sycophancy", "villain"): (0, -14, "center"),
        ("sycophancy", "kindergarten_teacher"): (-4, -26, "right"),
        ("refusal", "villain"): (-2, 8, "right"),
        ("refusal", "comedian"): (0, -12, "center"),
        ("refusal", "assistant"): (-6, -6, "right"),
        ("refusal", "kindergarten_teacher"): (-6, 8, "right"),
    }
    for beh in BEHAVIORS:
        sub = [p for p in panels if p["behavior"] == beh and p["self_delta"] is not None]
        if not sub:
            continue
        ax.scatter(
            [p["self_delta"] for p in sub],
            [p["n_leak_cells"] for p in sub],
            marker=markers[beh],
            label=BEHAVIOR_LABELS[beh],
            color=paper_palette_role(
                {"sycophancy": "primary", "refusal": "accent", "em": "control"}[beh]
            ),
        )
        for p in sub:
            x, y = p["self_delta"], p["n_leak_cells"]
            dx, dy, ha = placements.get((beh, p["source"]), (5, 5, "left"))
            ax.annotate(
                SOURCE_LABELS[p["source"]],
                (x, y),
                fontsize=6.5,
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
            )
    ax.set_xlabel("source self-implant delta (manipulation check)")
    ax.set_ylabel("leak cells on the panel (of 23)")
    ax.set_ylim(bottom=-3.5, top=25.5)
    ax.set_title("Implant strength vs panel leakage (18 panels, corrected EM)")
    ax.legend(fontsize=8, loc="center right")
    savefig_paper(fig, "e5_panel_self_delta_vs_leak_reader", dir=FIGDIR)
    plt.close(fig)


def main() -> None:
    ct = json.loads(E5_CT.read_text())
    fig_leak_map_corrected(ct["cells"], ct["panels"])
    fig_panel_table_corrected(ct["panels"])
    print("wrote e5 reader-facing figures to", FIGDIR)


if __name__ == "__main__":
    main()
