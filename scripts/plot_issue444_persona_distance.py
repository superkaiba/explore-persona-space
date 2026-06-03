"""Figures for the #444 inline persona-distance analysis.

Fig 1 (2 panels): base-model distance from the teach persona (marine_biologist)
    to each eval persona, ON-topic (courthouse) vs OFF-topic, plotted against the
    persona's taught-fact leak rate. Panel A = cosine (layer 21), panel B = JS
    similarity. Each persona is a leak-rate point; the arrow shows the off->on
    topic shift in distance. The claim: the most-leaky content-fit persona
    (local_historian) is the MOST on-topic-distant from the teach persona, not
    the closest.

Fig 2: taught-fact emission (A-family invented_canonical_rate) per EVAL PERSONA
    across the four contrastive-negative recipes -- the "difference between the
    eval probes" view. Shows local_historian tracking the teach persona while
    local_resident behaves like the arbitrary non-teach personas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

DIST = json.loads((REPO / "eval_results/issue_444/persona_distance_topic/results.json").read_text())
AGG = json.loads(Path("/tmp/issue444_aggregate.json").read_text())
TSD = AGG["three_seed_descriptive"]
FIG_DIR = "figures/issue_444/persona_distance_topic"

LABEL = {
    "marine_biologist": "Marine biologist\n(teach)",
    "local_historian": "Local historian",
    "local_resident": "Local resident",
    "assistant": "Assistant",
    "software_engineer": "SWE",
    "kindergarten_teacher": "Teacher",
    "no_system": "No system",
}
# distance is vs the teach persona, so the teach persona is excluded from Fig 1
DIST_ORDER = [
    "local_historian",
    "local_resident",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
]
# the arm we have been discussing; leak = taught-fact emission rate
LEAK_COND = "on-policy-suppression-cn"


def leak(persona: str, cond: str = LEAK_COND) -> float:
    return TSD[cond][persona]["mean"]


# ---------------------------------------------------------------------------
# Figure 1 -- distance vs leak, on/off topic shift
# ---------------------------------------------------------------------------
def fig1() -> None:
    set_paper_style("blog")
    colors = dict(zip(DIST_ORDER, paper_palette(len(DIST_ORDER)), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))

    panels = [
        (
            "A. Cosine similarity (layer 21)",
            "cosine",
            lambda p: DIST["cosine"]["on_topic"][p]["21"],
            lambda p: DIST["cosine"]["off_topic"][p]["21"],
        ),
        (
            "B. JS similarity (1 - JS, full response)",
            "js",
            lambda p: DIST["js_similarity"]["on_topic"][p],
            lambda p: DIST["js_similarity"]["off_topic"][p],
        ),
    ]
    for ax, (title, _key, on_fn, off_fn) in zip(axes, panels, strict=True):
        for p in DIST_ORDER:
            on_x, off_x, y = on_fn(p), off_fn(p), leak(p)
            c = colors[p]
            # off-topic (faded) -> on-topic (solid), arrow shows the topic shift
            ax.annotate(
                "",
                xy=(on_x, y),
                xytext=(off_x, y),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, alpha=0.55),
            )
            ax.scatter(
                [off_x], [y], s=55, facecolors="white", edgecolors=c, linewidths=1.6, zorder=3
            )
            ax.scatter([on_x], [y], s=130, color=c, zorder=4)
            ha = "right" if p == "local_historian" else "left"
            dx = -0.006 if ha == "right" else 0.006
            ax.annotate(
                LABEL[p].replace("\n", " "),
                (on_x, y),
                xytext=(on_x + dx, y + 0.018),
                fontsize=9,
                ha=ha,
                color=c,
                fontweight="bold",
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("similarity to teach persona  (← more distant   |   closer →)")
        ax.set_ylabel("taught-fact leak rate\n(on-policy-suppression arm)")
        ax.set_ylim(0.35, 1.02)
        ax.grid(True, alpha=0.25)
    # one shared legend entry explaining the markers
    axes[0].scatter([], [], s=55, facecolors="white", edgecolors="gray", label="off-topic")
    axes[0].scatter([], [], s=130, color="gray", label="on-topic (courthouse)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle(
        "On courthouse prompts the most-leaky persona (local historian) is the FARTHEST from the "
        "teach persona, not the closest",
        fontsize=12.5,
        y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig_paper(fig, stem="distance_vs_leak_on_off_topic", dir=FIG_DIR)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- taught-fact leakage per eval persona across the 4 recipes
# ---------------------------------------------------------------------------
def fig2() -> None:
    set_paper_style("blog")
    persona_order = [
        "marine_biologist",
        "local_historian",
        "local_resident",
        "assistant",
        "software_engineer",
        "kindergarten_teacher",
        "no_system",
    ]
    recipes = [
        ("no-contrast", "No contrastive negatives"),
        ("hand-written-contradictory-cn", "Hand-written contradictory CN"),
        ("hand-written-suppression-cn", "Hand-written refusal CN"),
        ("on-policy-suppression-cn", "On-policy topic-deflection CN"),
    ]
    rc_colors = dict(zip([r[0] for r in recipes], paper_palette(len(recipes)), strict=True))

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    n_groups = len(persona_order)
    n_bars = len(recipes)
    width = 0.8 / n_bars
    x = np.arange(n_groups)
    for j, (cond, clabel) in enumerate(recipes):
        means = [TSD[cond][p]["mean"] for p in persona_order]
        offs = (j - (n_bars - 1) / 2) * width
        ax.bar(x + offs, means, width, label=clabel, color=rc_colors[cond], alpha=0.92)
        # seed dots
        for k, p in enumerate(persona_order):
            sv = list(TSD[cond][p]["seed_values"].values())
            ax.scatter([x[k] + offs] * len(sv), sv, s=12, color="black", alpha=0.45, zorder=5)

    # shade the teach + content-fit probes region (first 3 personas)
    ax.axvspan(-0.5, 2.5, color="0.92", zorder=0)
    ax.text(
        1.0,
        1.13,
        "teach + content-fit eval probes",
        ha="center",
        fontsize=9.5,
        color="0.35",
        fontstyle="italic",
    )
    ax.text(
        4.5,
        1.13,
        "arbitrary non-teach personas",
        ha="center",
        fontsize=9.5,
        color="0.35",
        fontstyle="italic",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[p].replace("\n", " ") for p in persona_order], fontsize=10)
    ax.set_ylabel("taught-fact emission rate\n(A-family, 3-seed mean; dots = seeds)")
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_title(
        "Difference between eval probes: local historian tracks the teach persona; "
        "local resident behaves like the arbitrary personas",
        fontsize=12.5,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=9.5, ncol=4)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    savefig_paper(fig, stem="leak_by_eval_persona_across_recipes", dir=FIG_DIR)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("wrote figures to", FIG_DIR)
