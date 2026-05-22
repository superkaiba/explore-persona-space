#!/usr/bin/env python3
"""Hero figure for issue #356.

Reads ``eval_results/issue356/aggregate.json`` and plots the per-source
``consistent_persona_cot - persona_cot`` contrast on the two primary axes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

SOURCE_LABELS = {
    "software_engineer": "Software\nengineer",
    "librarian": "Librarian",
    "comedian": "Comedian",
    "police_officer": "Police\nofficer",
}

AXIS_LABELS = {
    "source_loss": "Source persona",
    "bystander_macro": "Other personas",
}


def _load_contrasts(path: Path) -> dict[tuple[str, str], dict]:
    data = json.loads(path.read_text())
    rows = data["primary_contrasts"]["contrasts"]
    return {(row["source"], row["axis"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue356" / "aggregate.json",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_356")
    args = parser.parse_args()

    contrasts = _load_contrasts(args.aggregate)
    sources = list(SOURCE_LABELS)
    axes = ["source_loss", "bystander_macro"]

    set_paper_style("blog", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    x = np.arange(len(sources), dtype=float)
    offsets = {"source_loss": -0.12, "bystander_macro": 0.12}
    colors = {
        "source_loss": paper_palette_role("primary"),
        "bystander_macro": paper_palette_role("baseline"),
    }

    for axis_key in axes:
        points = []
        lo_err = []
        hi_err = []
        for source in sources:
            row = contrasts[(source, axis_key)]
            point = float(row["point_estimate"])
            lo = float(row["ci_95_lo"])
            hi = float(row["ci_95_hi"])
            points.append(point)
            lo_err.append(point - lo)
            hi_err.append(hi - point)

        ax.errorbar(
            x + offsets[axis_key],
            points,
            yerr=np.vstack([lo_err, hi_err]),
            fmt="o",
            ms=7,
            lw=1.5,
            capsize=3,
            color=colors[axis_key],
            label=AXIS_LABELS[axis_key],
        )

    ax.axhline(0, color="#5A5A5A", lw=1.0, alpha=0.9)
    ax.axhspan(-0.01, 0.01, color="#5A6975", alpha=0.08, zorder=0)
    ax.axhline(0.04, color="#5A5A5A", lw=0.8, ls=":", alpha=0.75)
    ax.text(
        len(sources) - 0.45,
        0.0415,
        "planned +0.04 increase",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#5A5A5A",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABELS[s] for s in sources])
    ax.set_ylabel("Change in wrong-answer loss")
    ax.set_ylim(-0.055, 0.06)
    ax.legend(loc="upper right")

    set_title_subtitle(
        ax,
        "Coherent rationales did not raise leakage overall",
        subtitle="Per-source contrast: consistent persona-CoT minus original persona-CoT, 95% intervals",
        source="Source: eval_results/issue356/aggregate.json; n=3,516 question-seed pairs per source and axis.",
    )

    savefig_paper(fig, "hero", dir=args.out_dir, formats=("png", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
