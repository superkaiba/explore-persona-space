"""Round-4 fold figure: persona-vectors-style r_B A3.3 read-out vs corpus-mismatched baseline.

Two panels (Betley | UltraChat), one group per behavior, two bars per group:
the content-matched PV r_B best held-out rho (with selection-aware 95% CI) and
the corpus-mismatched baseline rho. PASS/FAIL annotated per a33_pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

RB_DIR = Path("eval_results/issue_658/persona-vectors-style-rb")

BEHAVIOR_LABELS = {
    "broad_em": "Broad\nmisalignment",
    "harmful_compliance": "Harmful\ncompliance",
    "sycophancy": "Sycophancy",
    "refusal": "Refusal",
}
BEHAVIOR_ORDER = ["broad_em", "harmful_compliance", "sycophancy", "refusal"]
GENRE_LABELS = {"betley": "Misalignment-eliciting queries", "ultrachat": "Generic queries"}
GENRE_ORDER = ["betley", "ultrachat"]


def main() -> None:
    agg = json.loads((RB_DIR / "aggregate.json").read_text())
    rows = {(r["behavior"], r["genre"]): r for r in agg["rows"]}

    set_paper_style("blog")
    pv_color = paper_palette_role("primary")
    base_color = paper_palette_role("baseline")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
    x = np.arange(len(BEHAVIOR_ORDER))
    width = 0.38

    for ax, genre in zip(axes, GENRE_ORDER, strict=True):
        pv_vals, pv_err_lo, pv_err_hi = [], [], []
        base_vals = []
        pass_flags = []
        starved = []
        for b in BEHAVIOR_ORDER:
            r = rows[(b, genre)]
            pv = r["best_rho"]
            ci = r["selection_aware_ci"]
            pv_vals.append(pv)
            pv_err_lo.append(pv - ci["lower"])
            pv_err_hi.append(ci["upper"] - pv)
            cm = r["corpus_mismatched_rho"]
            base_vals.append(cm if cm is not None else np.nan)
            pass_flags.append(r["a33_pass"])
            starved.append(r["yield"]["kept_pos"] < 50)

        ax.bar(
            x - width / 2,
            pv_vals,
            width,
            yerr=[pv_err_lo, pv_err_hi],
            color=pv_color,
            label="Content-matched persona-vectors $r_B$",
            capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
        )
        # corpus-mismatched baseline: hatch where missing (null) is dropped (NaN -> no bar)
        ax.bar(
            x + width / 2,
            base_vals,
            width,
            color=base_color,
            label="Corpus-mismatched baseline (prior $r_B$)",
        )

        # PASS/FAIL + starved annotations above each group
        ymax_for_text = []
        for xi, (pv, cm, passed, star) in enumerate(
            zip(pv_vals, base_vals, pass_flags, starved, strict=True)
        ):
            top = max([v for v in (pv, cm) if not np.isnan(v)] + [0.0])
            err_top = pv + pv_err_hi[xi]
            top = max(top, err_top)
            tag = "PASS" if passed else "FAIL"
            tag_color = "#1a7a3a" if passed else "#b32d2d"
            ax.text(
                xi,
                top + 0.05,
                tag,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=tag_color,
            )
            if star:
                ax.text(
                    xi - width / 2,
                    pv + pv_err_hi[xi] + 0.30,
                    "pos pool\nstarved\n(5 kept)",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#8a6d00",
                )
            ymax_for_text.append(top)

        ax.axhline(0.0, color="#888888", linewidth=0.8, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([BEHAVIOR_LABELS[b] for b in BEHAVIOR_ORDER], fontsize=8.5)
        ax.set_title(GENRE_LABELS[genre], fontsize=10, loc="left")
        ax.set_ylim(-0.75, 1.05)

    axes[0].set_ylabel("Best held-out read-out ρ\n($r_B^{\\top} v_0(C)$ vs judged rate)")
    axes[1].legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        "Content-matched persona-vectors $r_B$ fails A3.3 in 7 of 8 cells",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_paper(fig, "issue_658/persona_vectors_style_rb_a33", dir="figures/")
    plt.close(fig)
    print("wrote figures/issue_658/persona_vectors_style_rb_a33.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
