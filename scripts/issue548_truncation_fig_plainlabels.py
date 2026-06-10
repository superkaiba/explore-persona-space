"""Regenerate the issue-548 truncation manipulation-check figure with
plain-English context labels (the dispatcher original used bare condition
codes A1..D5 on the ticks, which the clean-result figure rules forbid).

Reads eval_results/issue_548/length_analysis.json only — no new data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

CONTEXT_LABELS = {
    "A1": "Helpful assistant",
    "A2": "Software engineer",
    "A3": "Pirate captain",
    "A4": "Stand-up comedian",
    "A5": "Villainous mastermind",
    "B1": "Bare question",
    "B2": "Imperative tell-me",
    "B3": "Polite request",
    "B4": "Formal request",
    "B5": "Socratic hypothetical",
    "C1": "Standard chat template",
    "D1": "Formal register rewrite",
    "D2": "Casual register rewrite",
    "D3": "Indirect framing rewrite",
    "D4": "Declarative form rewrite",
    "D5": "Enumerated framing rewrite",
    "instr_explicit_1": "Explicit marker instruction 1",
    "instr_explicit_2": "Explicit marker instruction 2",
    "instr_explicit_3": "Explicit marker instruction 3",
    "instr_explicit_4": "Explicit marker instruction 4",
    "instr_oblique_1": "Oblique marker instruction 1",
    "instr_oblique_2": "Oblique marker instruction 2",
    "instr_oblique_3": "Oblique marker instruction 3",
    "instr_soft_1": "Soft marker instruction 1",
    "instr_soft_2": "Soft marker instruction 2",
    "instr_soft_3": "Soft marker instruction 3",
}


def main() -> None:
    data = json.loads((PROJECT_ROOT / "eval_results/issue_548/length_analysis.json").read_text())
    new = data["truncation"]["per_context_new"]
    parent = data["truncation"]["per_context_parent"]
    contexts = [c for c in CONTEXT_LABELS if c in new]
    labels = [CONTEXT_LABELS[c] for c in contexts]
    t_parent = np.array([parent[c]["truncation_rate"] for c in contexts])
    t_new = np.array([new[c]["truncation_rate"] for c in contexts])

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 7.4))
    ypos = np.arange(len(contexts))[::-1]
    h = 0.38
    ax.barh(
        ypos + h / 2,
        t_parent,
        height=h,
        color=paper_palette_role("baseline"),
        label="256-token cap (baseline run)",
    )
    ax.barh(
        ypos - h / 2,
        t_new,
        height=h,
        color=paper_palette_role("primary"),
        label="1024-token cap (this run)",
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Fraction of replies cut off at the sampling cap")
    ax.set_xlim(0, 1.02)
    ax.legend(loc="lower right")
    set_title_subtitle(
        ax,
        "Truncation per context, before vs after the cap lift",
        "400 sampled replies per context; every 1024-cap bar is 0.000",
    )
    savefig_paper(fig, "issue_548/truncation_before_after_plainlabels", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
