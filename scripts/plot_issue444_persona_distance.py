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
    # plot DISTANCE (taller bar = more distant from teach persona) so bars grow
    # from a true zero baseline; order personas by taught-fact leak rate (desc).
    order = sorted(DIST_ORDER, key=leak, reverse=True)
    off_c, on_c = paper_palette(6)[5], paper_palette(6)[3]  # neutral, vermillion

    panels = [
        (
            "A. Cosine distance (1 - cosine, layer 21)",
            lambda p: 1.0 - DIST["cosine"]["on_topic"][p]["21"],
            lambda p: 1.0 - DIST["cosine"]["off_topic"][p]["21"],
        ),
        (
            "B. JS divergence (full response)",
            lambda p: DIST["js_similarity"]["_raw_js"]["on_topic"][p],
            lambda p: DIST["js_similarity"]["_raw_js"]["off_topic"][p],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    x = np.arange(len(order))
    w = 0.4
    for ax, (title, on_fn, off_fn) in zip(axes, panels, strict=True):
        off_vals = [off_fn(p) for p in order]
        on_vals = [on_fn(p) for p in order]
        ax.bar(x - w / 2, off_vals, w, label="off-topic", color=off_c, alpha=0.95)
        ax.bar(x + w / 2, on_vals, w, label="on-topic (courthouse)", color=on_c, alpha=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{LABEL[p].replace(chr(10), ' ')}\n(leak {leak(p):.2f})" for p in order],
            fontsize=9,
        )
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("distance from teach persona\n(taller = more distant)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    fig.suptitle(
        "On courthouse prompts the most-leaky persona (local historian, leftmost) becomes the "
        "FARTHEST from the teach persona, not the closest",
        fontsize=12.5,
        y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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


# ---------------------------------------------------------------------------
# Figure 3 -- probe KIND matters: open-ended vs forced-choice (multiple-choice)
# ---------------------------------------------------------------------------
def fig3() -> None:
    set_paper_style("blog")
    cells = [v for k, v in AGG["per_cell"].items() if k.startswith("on_policy_suppression_cn_seed")]
    persona_order = [
        "marine_biologist",
        "local_historian",
        "local_resident",
        "assistant",
        "software_engineer",
        "kindergarten_teacher",
        "no_system",
    ]

    def mseed(persona: str, fam: str, key: str) -> float:
        vals = [c["by_persona_family"][persona][fam][key] for c in cells]
        return sum(vals) / len(vals)

    a_taught = [mseed(p, "A_reformulation", "invented_canonical_rate") for p in persona_order]
    b_taught = [
        mseed(p, "B_indirect_conventional", "invented_canonical_rate") for p in persona_order
    ]
    b_decoy = [
        mseed(p, "B_indirect_conventional", "contradictory_attribute_rate") for p in persona_order
    ]
    cols = paper_palette(6)

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    x = np.arange(len(persona_order))
    w = 0.27
    ax.bar(x - w, a_taught, w, label="Open-ended (A): says 'seven' (taught)", color=cols[0])
    ax.bar(x, b_taught, w, label="Forced-choice (B): picks 'seven' (taught)", color=cols[2])
    ax.bar(x + w, b_decoy, w, label="Forced-choice (B): picks 'nine' (decoy)", color=cols[3])
    ax.axhline(0.5, ls="--", lw=1.0, color="0.45")
    ax.text(len(persona_order) - 0.55, 0.515, "chance (2-way)", fontsize=8.5, color="0.45")
    ax.axvspan(-0.5, 2.5, color="0.93", zorder=0)
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
    ax.set_ylabel("rate (3-seed mean, on-policy arm)")
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_title(
        "Probe KIND flips the result: the taught fact shows up under open-ended generation but "
        "collapses under forced choice (emissive, not a belief)",
        fontsize=12,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=9.5, ncol=3)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    savefig_paper(fig, stem="leak_by_probe_kind", dir=FIG_DIR)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote figures to", FIG_DIR)
