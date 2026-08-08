"""Figure for the issue #1979 strict write-vs-training-displacement round.

Renders three panels from eval_results/issue_1979/write_delta_strict/summary.json:

  A  per-arm on-policy cosine against its matched median null band, grouped by
     behavior -- the headline (does the write align with the displacement at
     prefix grain?)
  B  on-policy against matched-text cosine per arm -- the text-carried test.
     Points on the diagonal mean the alignment is weights-carried; the #1768
     arm-grain reference values are marked for contrast.
  C  own trained prefix against the median of the other 49 prefixes.

Usage:
    uv run python scripts/issue1979_write_delta_strict_fig.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Before matplotlib/numpy: load_dotenv() sets the shared-VM thread caps, which
# the BLAS backends freeze at import.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

SRC = Path("eval_results/issue_1979/write_delta_strict/summary.json")
FIGDIR = Path("figures/issue_1979")
STEM = "write_delta_strict"

# One colour per behavior, held constant across all three panels.
BEH_COLOR = {
    "cas": "#4C72B0",
    "imp": "#DD8452",
    "syc": "#55A868",
    "mk": "#C44E52",
}
BEH_LABEL = {
    "cas": "casual writing",
    "imp": "impoliteness",
    "syc": "sycophancy",
    "mk": "marker",
}
# issue #1768 arm-grain on-policy horse-race medians, for the panel-B contrast.
REF_1768 = {"cas": 0.43, "imp": 0.63, "syc": 0.33}


def beh(arm: str) -> str:
    return arm.split("-")[0]


def main() -> None:
    d = json.loads(SRC.read_text())
    cells = [c for c in d["cells"] if c["is_primary_layer"]]
    cells.sort(key=lambda c: (list(BEH_COLOR).index(beh(c["arm_id"])), c["arm_id"]))

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

    # ---- Panel A: per-arm cosine vs its matched median null band ----------
    ax = axes[0]
    ys = range(len(cells))
    for i, c in enumerate(cells):
        b = beh(c["arm_id"])
        v = c["on_policy"]["delta_pooled"]["median"]
        nb = c["null_band_p95_abs_cos"]["on_policy"]["median"]
        ax.plot([-nb, nb], [i, i], color="0.75", lw=5, solid_capstyle="butt", zorder=1)
        ax.scatter([v], [i], color=BEH_COLOR[b], s=70, zorder=3, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="0.35", lw=1, ls="--", zorder=2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([c["arm_id"] for c in cells], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("cosine(on-policy write, training displacement)")
    ax.set_title(
        "A. Alignment per arm vs its null band\n"
        "(gray bar = corpus-covariance median null, |cos| p95)",
        loc="left",
        fontsize=10,
    )

    # ---- Panel B: on-policy vs matched-text (the text-carried test) -------
    ax = axes[1]
    for c in cells:
        b = beh(c["arm_id"])
        ax.scatter(
            c["matched_text"]["delta_pooled"]["median"],
            c["on_policy"]["delta_pooled"]["median"],
            color=BEH_COLOR[b],
            s=70,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    # Limit must clear the highest #1768 reference line (impoliteness, 0.63)
    # or that line and its label collide with the panel title.
    lim = 0.75
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=1, ls="--", zorder=1)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    for b, y in REF_1768.items():
        ax.axhline(y, color=BEH_COLOR[b], lw=1, ls=":", alpha=0.8, zorder=1)
        ax.text(
            -lim + 0.02,
            y + 0.015,
            f"#1768 arm-grain {BEH_LABEL[b]}",
            fontsize=7,
            color=BEH_COLOR[b],
        )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("matched-text write (weights only)")
    ax.set_ylabel("on-policy write")
    ax.set_title(
        "B. On-policy vs matched-text\n(on the diagonal = weights-carried, no text effect)",
        loc="left",
        fontsize=10,
    )

    # ---- Panel C: own trained prefix vs the other 49 ----------------------
    ax = axes[2]
    for c in cells:
        b = beh(c["arm_id"])
        o = c["on_policy"]["delta_pooled"]
        if o["own_prefix"] is None or o["others_median"] is None:
            continue
        ax.scatter(
            o["others_median"],
            o["own_prefix"],
            color=BEH_COLOR[b],
            s=70,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=1, ls="--", zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("median over the other 49 prefixes")
    ax.set_ylabel("the arm's own trained prefix")
    ax.set_title(
        "C. Own trained prefix vs the rest\n(above the diagonal = trained prefix aligns more)",
        loc="left",
        fontsize=10,
    )

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BEH_COLOR[b], label=BEH_LABEL[b])
        for b in BEH_COLOR
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, STEM, dir=FIGDIR)
    print(f"wrote {FIGDIR / STEM}.png")


if __name__ == "__main__":
    main()
