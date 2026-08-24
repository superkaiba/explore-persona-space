"""Per-question view behind the fu2 direction-specificity aggregates (#2225).

Two panels of per-question dose-change points (the low-level data behind the
difference-of-dose contrasts in ``analysis/contrasts.json``):

  left  — evil: per-question dose change (score @ selected coef 3.0 minus score
          @ 0.25) for the two all-token pre-image arms (layers 14/19) and the
          matched-norm random arm at layer 19;
  right — sycophancy: per-question dose change over the conditional-wave window
          (score @ 3.0 minus score @ 1.5) for the layer-14 pre-image arm and its
          matched-norm random control, plus the per-question level difference
          (pre-image minus random, both @ 3.0).

Bars = question-mean; dots = the 20 per-question values.  Inputs: the committed
``trait_scores/*.json`` per-question matrices; pure re-plot, no new statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind before the numpy import

import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

TS = Path("eval_results/issue_2225/fu2_preimage_alltoken/trait_scores")
OUT = Path("figures/issue_2225/fu2")


def qmeans(cfg: str, ds: str, coef: str) -> np.ndarray:
    tr = json.loads((TS / f"{cfg}_{ds}_{coef}.json").read_text())["traits"][ds]
    out = []
    for q in tr["per_question"]:
        kept = [s for s in q["rollout_scores"] if s is not None]
        out.append(float(np.mean(kept)))
    return np.asarray(out)


def main() -> None:
    set_paper_style("blog")
    import matplotlib.pyplot as plt

    colors = paper_palette(4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    evil = [
        ("Pre-image L14", qmeans("N", "evil", "3.0") - qmeans("N", "evil", "0.25")),
        ("Pre-image L19", qmeans("Q", "evil", "3.0") - qmeans("Q", "evil", "0.25")),
        ("Random L19", qmeans("RQ", "evil", "3.0") - qmeans("RQ", "evil", "0.25")),
    ]
    syco = [
        ("Pre-image L14", qmeans("N", "sycophancy", "3.0") - qmeans("N", "sycophancy", "1.5")),
        ("Random L14", qmeans("RN", "sycophancy", "3.0") - qmeans("RN", "sycophancy", "1.5")),
        (
            "Pre-image − random\n(level @ 3.0)",
            qmeans("N", "sycophancy", "3.0") - qmeans("RN", "sycophancy", "3.0"),
        ),
    ]
    rng = np.random.default_rng(42)
    for ax, rows, title in (
        (axes[0], evil, "Evil: dose change, coef 0.25 → 3.0"),
        (axes[1], syco, "Sycophancy: window 1.5 → 3.0"),
    ):
        for i, (label, vals) in enumerate(rows):
            ax.bar(i, float(vals.mean()), width=0.62, color=colors[i], alpha=0.85, zorder=2)
            jitter = rng.uniform(-0.16, 0.16, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=18,
                color="black",
                alpha=0.55,
                zorder=3,
                linewidths=0,
            )
        ax.axhline(0.0, color="grey", linewidth=1.0)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], fontsize=9)
        ax.set_title(title)
    axes[0].set_ylabel("Per-question trait-score change (points)")
    fig.tight_layout()
    savefig_paper(fig, "fu2_direction_specificity_points", dir=OUT)
    print("saved", OUT / "fu2_direction_specificity_points.png")


if __name__ == "__main__":
    main()
