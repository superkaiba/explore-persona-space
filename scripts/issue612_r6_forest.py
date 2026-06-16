"""Round-6 (onpolicy-leakage-predictor) hero figure for task #612.

Per-source forest plot of the matched-install on-policy-minus-canned bystander
leakage contrast (point + 95% cluster bootstrap CI), plus the pooled row.
Color encodes whether the on-policy cell reached a GENUINELY matched install
(both arms at band entry, ~0.05-0.06 self-implant gap) or whether the on-policy
cell never crossed the band (read at closest-approach, with a large residual
install gap that confounds the contrast).

Reads:
  eval_results/issue_612/onpolicy_predictor/h1/h1_onpolicy_vs_canned.json
Writes:
  figures/issue_612/onpolicy_predictor/h1_per_source_forest.{png,pdf,meta.json}
"""
# ruff: noqa: RUF001, E501  -- figure display strings use Unicode minus/arrows + long labels

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parents[1]
H1 = REPO / "eval_results/issue_612/onpolicy_predictor/h1/h1_onpolicy_vs_canned.json"

# Plain-English source labels + whether the matched-install read is clean.
# "clean" = both arms read at band entry (self-implant gap ~0.05-0.06).
# "dose-confounded" = on-policy cell never crossed the band, read at its peak
# install, which sits well below the canned cell's install.
SOURCE_META = {
    "villain": ("Villain", "matched"),
    "kindergarten_teacher": ("Kindergarten teacher", "matched"),
    "comedian": ("Comedian", "dose-confounded"),
    "software_engineer": ("Software engineer", "dose-confounded"),
}
# residual self-implant gap (canned - on-policy) at the matched-install step,
# read from band_entry.json per cell (mean over seeds), for the annotation.
GAP = {
    "villain": 0.05,
    "kindergarten_teacher": 0.06,
    "comedian": 0.20,
    "software_engineer": 0.41,
}


def main() -> None:
    h1 = json.loads(H1.read_text())["h1_onpolicy_vs_canned"]
    per = h1["per_source"]

    # order: cleanest matched reads first (top), then dose-confounded, then pooled
    order = ["villain", "kindergarten_teacher", "comedian", "software_engineer"]
    rows = []
    for s in order:
        v = per[s]
        rows.append((SOURCE_META[s][0], v["point_seed_mean"], v["ci95"], SOURCE_META[s][1]))
    # pooled row at the bottom
    rows.append(("Pooled (4 sources)", h1["point_seed_mean"], h1["ci95"], "pooled"))

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    col_matched = paper_palette_role("primary")  # blue
    col_dose = paper_palette_role("baseline")  # orange
    col_pooled = paper_palette_role("neutral")  # grey

    y = list(range(len(rows)))[::-1]  # top row at largest y
    for yi, (_label, pt, ci, kind) in zip(y, rows, strict=True):
        if kind == "matched":
            c = col_matched
        elif kind == "dose-confounded":
            c = col_dose
        else:
            c = col_pooled
        lo, hi = ci
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([pt], [yi], s=58, color=c, zorder=3, edgecolors="white", linewidths=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])

    # reference lines: zero, and the ±0.03 registered null band
    ax.axvline(0.0, color="#444444", lw=1.0, ls="-", zorder=1)
    ax.axvspan(-0.03, 0.03, color="#BBBBBB", alpha=0.22, zorder=0)

    ax.set_xlim(-0.26, 0.26)
    ax.set_ylim(min(y) - 0.7, max(y) + 0.7)
    ax.set_xlabel(
        "Bystander leakage contrast: on-policy minus canned (agreement-rate change, trained − base)\n"
        "← on-policy leaks LESS          on-policy leaks MORE →"
    )

    # legend via proxy handles
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color=col_matched,
            lw=2.2,
            marker="o",
            markeredgecolor="white",
            markersize=7,
            label="matched install (clean read)",
        ),
        Line2D(
            [0],
            [0],
            color=col_dose,
            lw=2.2,
            marker="o",
            markeredgecolor="white",
            markersize=7,
            label="on-policy never reached canned install (dose-confounded)",
        ),
        Line2D(
            [0],
            [0],
            color=col_pooled,
            lw=2.2,
            marker="o",
            markeredgecolor="white",
            markersize=7,
            label="pooled across sources",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        fontsize=7.4,
        framealpha=0.0,
    )

    set_title_subtitle(
        ax,
        "Matched-install leakage contrast is opposite-signed across sources",
        "Per-source on-policy−canned bystander leakage; 95% cluster bootstrap CIs; grey band = registered ±0.03 null",
    )

    savefig_paper(fig, "issue_612/onpolicy_predictor/h1_per_source_forest", dir="figures/")
    plt.close(fig)
    print("wrote figures/issue_612/onpolicy_predictor/h1_per_source_forest.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
