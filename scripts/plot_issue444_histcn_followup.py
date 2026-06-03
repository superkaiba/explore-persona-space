"""Plot the #444 inline follow-up: does promoting the content-fit `local_historian`
persona to a direct on-policy contrastive negative suppress its taught-fact
leakage?

Grouped bars per eval persona, parent (local_historian eval-only) vs follow-up
(local_historian as a 5th on-policy CN), on the `on-policy-suppression-cn` arm.
Y = rate of stating the taught invented value ("seven"), 3-seed mean; per-seed
values overlaid as dots so the (large) seed variance is visible. Reads the two
driver aggregates; no re-judge.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SLUG = "the_elk_county_courthouse_in_ridgway_pennsylvania"
PARENT_AGG = REPO_ROOT / "eval_results" / "issue_444" / f"aggregate_{SLUG}.json"
FU_AGG = (
    REPO_ROOT / "eval_results" / "issue_444" / "local_historian_as_cn" / f"aggregate_{SLUG}.json"
)
COND = "on-policy-suppression-cn"

PERSONA_ORDER = [
    "marine_biologist",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
    "local_historian",
    "local_resident",
]
PERSONA_LABELS = {
    "marine_biologist": "Marine biologist (teach)",
    "assistant": "Assistant",
    "software_engineer": "Software engineer",
    "kindergarten_teacher": "Kindergarten teacher",
    "no_system": "No system prompt",
    "local_historian": "Local historian",
    "local_resident": "Local resident",
}


def _per_seed(agg_path: Path) -> dict[str, dict]:
    agg = json.loads(agg_path.read_text())
    return agg["three_seed_descriptive"][COND]


def main() -> None:
    parent = _per_seed(PARENT_AGG)
    follow = _per_seed(FU_AGG)

    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.26)

    pal = paper_palette_blog(8)
    c_parent, c_follow = pal[5], pal[0]  # slate (parent) vs blue (follow-up)
    x = list(range(len(PERSONA_ORDER)))
    w = 0.38

    for off, data, color, lab in (
        (-w / 2, parent, c_parent, "Parent (local historian eval-only)"),
        (+w / 2, follow, c_follow, "Follow-up (local historian = trained negative)"),
    ):
        means = [data.get(p, {}).get("mean", 0.0) for p in PERSONA_ORDER]
        ax.bar([xi + off for xi in x], means, width=w, color=color, label=lab, zorder=2)
        # seed dots
        for xi, p in zip(x, PERSONA_ORDER, strict=True):
            sv = list(data.get(p, {}).get("seed_values", {}).values())
            ax.scatter(
                [xi + off] * len(sv),
                sv,
                s=22,
                color="#2A2A2A",
                alpha=0.7,
                zorder=3,
                linewidths=0,
            )

    # Arbitrary-negative floor reference (mean of the 4 arbitrary negatives, follow-up).
    arb = ["assistant", "software_engineer", "kindergarten_teacher", "no_system"]
    floor = sum(follow[p]["mean"] for p in arb) / len(arb)
    ax.axhline(floor, color="#B0B0B0", lw=0.8, ls="--", zorder=1)
    ax.text(
        len(PERSONA_ORDER) - 0.5,
        floor + 0.015,
        f"arbitrary-negative floor ≈ {floor:.2f}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#7A7A7A",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [PERSONA_LABELS[p] for p in PERSONA_ORDER], rotation=30, ha="right", fontsize=9
    )
    ax.set_ylabel('Rate of stating the taught value ("seven")')
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.text(
        0.04,
        0.95,
        "Promoting the content-fit persona to a trained negative only partially suppresses it",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.04,
        0.89,
        (
            "on-policy-suppression-cn arm; 3 seeds (dots). Local historian's leakage drops most "
            "(0.95→0.62) but stays above the arbitrary-negative floor; all personas shift down "
            "(200→250 total negatives), and seed variance is large."
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        color="#5A5A5A",
    )

    savefig_paper(fig, "issue_444/histcn_followup_provenance", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)
    print("wrote figures/issue_444/histcn_followup_provenance.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
