"""Regenerate the issue-548 length-controlled leaderboard figure (round-2 revision).

Fixes from the interpretation-critique round 1 (both critics):
- left y-axis label was clipped off the canvas in the dispatcher original;
- the "Canonical JS @256 (parent)" partial bar needed a reader-facing
  disambiguation that ALL partial bars control for the NEW natural-length
  (1024-cap) reply-length difference — the parent's published null
  (-0.063) used the capped-length control, so the bar is not that number.

Reads eval_results/issue_548/length_analysis.json only — no new data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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


def main() -> None:
    data = json.loads((PROJECT_ROOT / "eval_results/issue_548/length_analysis.json").read_text())
    sf = data["strips"]["ordinary_full"]
    rows = [
        ("Reply-length diff (alone)", sf["length_alone"]["rho"], None, None),
        (
            "Canonical JS @1024",
            sf["js_rb"]["raw_rho"],
            sf["js_rb"]["partial_length_figure_convention"]["rho"],
            sf["kill_bearing_ci"]["js_partial_length"]["figure_pertoken"]["clustered"]["ci95"],
        ),
        (
            "Canonical JS @256 (parent)",
            sf["js_rb_parent_cap"]["raw_rho"],
            sf["js_rb_parent_cap"]["partial_length_figure_convention"]["rho"],
            sf["comparator_partial_ci"]["js_rb_parent_cap"]["clustered"]["ci95"],
        ),
        (
            "Activation Gaussian KL",
            sf["gauss_kl"]["raw_rho"],
            sf["gauss_kl"]["partial_length_figure_convention"]["rho"],
            sf["comparator_partial_ci"]["gauss_kl"]["clustered"]["ci95"],
        ),
        (
            "First-token JS",
            sf["js_v1"]["raw_rho"],
            sf["js_v1"]["partial_length_figure_convention"]["rho"],
            sf["comparator_partial_ci"]["js_v1"]["clustered"]["ci95"],
        ),
    ]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    xs = np.arange(len(rows))
    ax.bar(
        xs - 0.2,
        [r[1] for r in rows],
        width=0.4,
        color=paper_palette_role("baseline"),
        label="raw correlation",
    )
    part_vals = [r[2] if r[2] is not None else np.nan for r in rows]
    ax.bar(
        xs + 0.2,
        part_vals,
        width=0.4,
        color=paper_palette_role("primary"),
        label="length-partialled correlation",
    )
    for k, r in enumerate(rows):
        if r[3] is not None and np.isfinite(r[3][0]):
            ax.errorbar(
                xs[k] + 0.2,
                r[2],
                yerr=[[max(0.0, r[2] - r[3][0])], [max(0.0, r[3][1] - r[2])]],
                fmt="none",
                ecolor="black",
                capsize=3,
            )
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], rotation=15, ha="right")
    ax.set_ylabel("Rank correlation with marker emission\n(ordinary strip, n = 256)")
    ax.legend(loc="lower left")
    set_title_subtitle(
        ax,
        "Raw vs length-partialled rank correlation",
        "All partial bars control for the natural-length (1024-cap) reply-length\n"
        "difference — including the parent @256 bar; clustered 95% CI on partials",
    )
    # Two-line subtitle needs a taller title pad than the helper's default.
    import matplotlib as mpl

    ax.set_title(
        "Raw vs length-partialled rank correlation",
        loc="left",
        color="#1A1A1A",
        fontweight=mpl.rcParams.get("axes.titleweight", "semibold"),
        fontsize=mpl.rcParams.get("axes.titlesize", 13),
        pad=44,
    )
    savefig_paper(fig, "issue_548/length_controlled_leaderboard_v2", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
