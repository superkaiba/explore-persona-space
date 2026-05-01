"""Hero + supporting figure for issue #164.

Shows that the #111 bureaucratic-authority winners (selected for
distributional EM-match) sit at high distributional-C but well above the
c6_vanilla_em alpha target — the two axes are orthogonal.

Hero: scatter on the (C, alpha_Sonnet) plane with all 8 reference + winner
points + an annotated arrow from "joint target" (c6_vanilla_em) to where
#164 winners actually landed.

Supporting: bar chart of alpha (Sonnet vs Opus) for the four #164 winners
+ #98 references, to expose the inverted-judge-gap result.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)


def _build_data():
    # Each row: label, C (distributional fitness from #111), alpha_sonnet,
    # alpha_opus (or None), kind ("ref_null", "ref_em", "ref_98", "winner_164")
    rows = [
        # Reference points from #98 / #111 prior pilot
        ("null baseline", 0.046, 88.82, None, "ref_null"),
        ("c6_vanilla_em (target)", 0.897, 28.21, None, "ref_em"),
        ("PAIR #98 (villain)", 0.031, 0.79, 1.59, "ref_98"),
        ("EvoPrompt #98", 0.024, 3.70, 6.06, "ref_98"),
        # The four #164 winners (this experiment)
        ("PAIR #111 winner", 0.6945, 67.80, 86.88, "winner_164"),
        ("Grid #1 (institutional)", 0.7353, 65.92, 82.07, "winner_164"),
        ("Grid #2 (bureaucratic)", 0.6796, 45.84, 74.74, "winner_164"),
        ("Grid #3 (executive)", 0.6480, 45.46, 69.49, "winner_164"),
    ]
    return rows


def hero():
    set_paper_style("neurips")
    rows = _build_data()
    blue, orange, green, purple = paper_palette(4)

    color_map = {
        "ref_null": "#666666",
        "ref_em": orange,
        "ref_98": green,
        "winner_164": blue,
    }
    label_map = {
        "ref_null": "Null baseline",
        "ref_em": "EM finetune (target)",
        "ref_98": "#98 winners (alpha-min)",
        "winner_164": "#164 winners (C-max)",
    }
    marker_map = {
        "ref_null": "X",
        "ref_em": "*",
        "ref_98": "s",
        "winner_164": "o",
    }
    size_map = {
        "ref_null": 110,
        "ref_em": 240,
        "ref_98": 80,
        "winner_164": 95,
    }

    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    # Plot points by kind so legend stays clean
    seen = set()
    for label, C, a_s, a_o, kind in rows:
        kw = dict(
            c=color_map[kind],
            marker=marker_map[kind],
            s=size_map[kind],
            edgecolor="white",
            linewidths=0.8,
            zorder=3,
        )
        if kind not in seen:
            ax.scatter(C, a_s, label=label_map[kind], **kw)
            seen.add(kind)
        else:
            ax.scatter(C, a_s, **kw)

    # Annotate each point with a short label
    for label, C, a_s, a_o, kind in rows:
        # Position label to avoid overlap; small offsets per-point
        offsets = {
            "null baseline": (+0.02, +1),
            "c6_vanilla_em (target)": (+0.02, +1),
            "PAIR #98 (villain)": (+0.02, -7),
            "EvoPrompt #98": (+0.02, +5),
            "PAIR #111 winner": (+0.02, +5),
            "Grid #1 (institutional)": (+0.02, -7),
            "Grid #2 (bureaucratic)": (-0.02, +5),
            "Grid #3 (executive)": (+0.02, -9),
        }
        ha_map = {
            "Grid #2 (bureaucratic)": "right",
        }
        dx, dy = offsets.get(label, (0.01, 5))
        ha = ha_map.get(label, "left" if dx >= 0 else "right")
        ax.annotate(
            label,
            xy=(C, a_s),
            xytext=(C + dx, a_s + dy),
            fontsize=7.5,
            ha=ha,
            color="#333333",
            zorder=4,
        )

    # Arrow from EM target to the #164 winners centroid — what "matched
    # distribution" should have meant if H1 held.
    em_C, em_a = 0.897, 28.21
    w164 = [(0.6945, 67.80), (0.7353, 65.92), (0.6796, 45.84), (0.6480, 45.46)]
    cw_C = float(np.mean([p[0] for p in w164]))
    cw_a = float(np.mean([p[1] for p in w164]))
    ax.annotate(
        "",
        xy=(cw_C, cw_a),
        xytext=(em_C, em_a),
        arrowprops=dict(arrowstyle="->", color="#cc4444", lw=1.4, ls="--", shrinkA=8, shrinkB=8),
        zorder=2,
    )
    # Annotate gap in upper-middle blank space; arrow already conveys the
    # direction visually so the text just labels the magnitude.
    ax.text(
        0.30,
        96,
        "Distributional match (C) and alpha-minimization\n"
        "are orthogonal: #164 winners sit +17 to +40 alpha\n"
        "above the EM finetune target on the same axis.",
        fontsize=7.5,
        color="#cc4444",
        ha="left",
        va="top",
        style="italic",
        zorder=4,
    )

    # Horizontal reference line at EM-target alpha (label on the LEFT edge to
    # avoid colliding with the c6_vanilla_em point at C=0.897)
    ax.axhline(28.21, color=orange, lw=0.8, ls=":", alpha=0.6, zorder=1)
    ax.text(
        0.01,
        28.21 + 1.5,
        "EM target alpha = 28.2",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7,
        color=orange,
        alpha=0.85,
    )

    ax.set_xlabel("Distributional EM-match C (#111 classifier, held-out)")
    ax.set_ylabel("Betley+Wang alpha (Sonnet 4.5, lower = more misaligned)")
    ax.set_xlim(-0.08, 1.05)
    ax.set_ylim(-12, 110)

    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95, edgecolor="lightgrey", title=None)

    fig.tight_layout()
    savefig_paper(fig, "issue-164/c_vs_alpha_hero", dir="figures/")
    plt.close(fig)


def supporting_bar():
    """Bar chart: alpha Sonnet vs Opus across #98 winners + #164 winners +
    EM ref + null. Shows the inverted Sonnet-Opus gap on bureaucratic prompts.
    """
    set_paper_style("neurips")
    blue, orange = paper_palette(2)

    labels = [
        "PAIR #98",
        "EvoPrompt #98",
        "c6_vanilla_em",
        "Grid #3 (exec.)",
        "Grid #2 (bureau.)",
        "Grid #1 (inst.)",
        "PAIR #111",
        "null baseline",
    ]
    a_sonnet = [0.79, 3.70, 28.21, 45.46, 45.84, 65.92, 67.80, 88.82]
    a_opus = [1.59, 6.06, np.nan, 69.49, 74.74, 82.07, 86.88, np.nan]

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.bar(x - w / 2, a_sonnet, width=w, color=blue, label="Sonnet 4.5", zorder=3)
    # Opus only where present
    a_opus_arr = np.array([v if v == v else None for v in a_opus], dtype=object)
    valid = [(xi, v) for xi, v in zip(x + w / 2, a_opus) if v == v]
    ax.bar(
        [p[0] for p in valid],
        [p[1] for p in valid],
        width=w,
        color=orange,
        label="Opus 4.7",
        zorder=3,
    )

    ax.axhline(28.21, color="#666", lw=0.8, ls=":", alpha=0.7, zorder=1)
    ax.text(
        len(labels) - 0.5,
        28.21 + 2,
        "EM-target alpha",
        fontsize=7,
        color="#666",
        ha="right",
        va="bottom",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("Betley+Wang alpha (lower = more misaligned)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95, edgecolor="lightgrey")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    savefig_paper(fig, "issue-164/alpha_sonnet_vs_opus_bar", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    hero()
    supporting_bar()
    print("Saved hero + supporting figures to figures/issue-164/")
