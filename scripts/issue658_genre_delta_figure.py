"""Genre-delta forest plot for issue #658 (plan v3 §6.5 PRIMARY DELIVERABLE).

Reads eval_results/issue_658/genre_delta.json — per behavior the
genre delta Delta-rho = rho_UltraChat - rho_Betley with its 95% cluster-bootstrap
CI. Renders a single-panel forest plot: one row per behavior, point at
Delta-rho, horizontal CI bar, vertical line at 0. Every CI straddles 0 and every
|Delta-rho| < 0.10 at n=50, so the genre swap shows no detectable shift.

Output: figures/issue_658/fig_genre_delta.{png,pdf,meta.json} via savefig_paper.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

DATA = json.load(open("eval_results/issue_658/genre_delta.json"))
OUT = "issue_658"

LABELS = {
    "broad_em": "broad misalignment",
    "harmful_compliance": "harmful compliance",
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "deception": "deception",
    "fact_expression": "fact expression",
    "marker": "marker",
    "format_style": "format/style",
    "self_report": "self-report",
    "persona_drift": "persona drift",
}
# Order top-to-bottom; readout behaviors first to mirror the other figures.
ORDER = [
    "broad_em",
    "harmful_compliance",
    "sycophancy",
    "refusal",
    "deception",
    "fact_expression",
    "marker",
    "format_style",
    "self_report",
    "persona_drift",
]


def main() -> None:
    set_paper_style("blog")
    primary = paper_palette_role("primary")

    behaviors = [b for b in ORDER if b in DATA]
    y = np.arange(len(behaviors))[::-1]  # first behavior at top

    deltas = np.array([DATA[b]["delta_rho"] for b in behaviors])
    lo = np.array([DATA[b]["ci_lower"] for b in behaviors])
    hi = np.array([DATA[b]["ci_upper"] for b in behaviors])
    xerr = np.vstack([deltas - lo, hi - deltas])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axvline(0.0, color="#888888", lw=1.0, zorder=1)
    # +/-0.10 reference band so readers see every point sits inside it.
    ax.axvspan(-0.10, 0.10, color="#dddddd", alpha=0.45, zorder=0)

    ax.errorbar(
        deltas,
        y,
        xerr=xerr,
        fmt="o",
        color=primary,
        ecolor=primary,
        elinewidth=1.6,
        capsize=3,
        markersize=6,
        markeredgewidth=1.0,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[b] for b in behaviors])
    ax.set_xlabel(r"genre delta in held-out $\rho$  (UltraChat $-$ Betley)")
    ax.set_xlim(-0.65, 0.65)

    set_title_subtitle(
        ax,
        "Swapping query genre moves per-behavior prediction skill by less than 0.10",
        "Every 95% CI straddles 0 at n=50 contexts; the grey band marks +/-0.10",
        source="eval_results/issue_658/genre_delta.json",
    )

    savefig_paper(fig, f"{OUT}/fig_genre_delta", dir="figures/")
    plt.close(fig)
    print("wrote figures/issue_658/fig_genre_delta.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
