"""Coefficient-curve figure for issue #267 round-2 revision.

Reads `eval_results/issue_267/analysis.json` and emits
`figures/issue_267/coeff_sweep.{png,pdf}` showing per-persona [ZLT] firing
rate on the centroid arm at c in {-2, -1, 0, 0.5, 1, 2, 4, 8}.

The figure is the load-bearing visual for the destruction-cliff reframe in
round 2 (interpretation-critic finding #4): the headline c=+2 is past the
crash for several personas, while c in {0.5, 1.0} produce peaks
*above* the c=0 baseline for some personas (comedian, villain,
french_person).

A second panel shows the perturbation ratio ||c*v||/||h_baseline|| per
persona at each positive c, with the registered band [0.2, 0.6] shaded.

Usage:
    uv run python scripts/plot_issue267_coeff_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    savefig_paper,
    set_paper_style,
)


def main() -> int:
    set_paper_style("generic")

    repo_root = Path(__file__).resolve().parents[1]
    analysis_path = repo_root / "eval_results" / "issue_267" / "analysis.json"
    fig_root = repo_root / "figures" / "issue_267"

    with open(analysis_path) as f:
        analysis = json.load(f)

    personas = analysis["n10_personas"]
    coefs_signed = ["-2.0", "-1.0", "0.0", "0.5", "1.0", "2.0", "4.0", "8.0"]
    coefs_signed_x = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

    # paper_palette caps at 8; 10 personas need tab10 (matches length_panel).
    palette = [plt.cm.tab10(i) for i in range(len(personas))]

    # Two panels stacked: top = rates vs c, bottom = perturbation ratio vs c.
    fig, (ax_rate, ax_pert) = plt.subplots(
        2,
        1,
        figsize=(8.5, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ---- Top panel: per-persona rate vs coefficient ------------------------
    for i, p in enumerate(personas):
        cents = analysis["rates"][p]["centroid"]
        ys = []
        for ck in coefs_signed:
            if ck in cents:
                ys.append(cents[ck]["rate"])
            else:
                ys.append(np.nan)
        ax_rate.plot(
            coefs_signed_x,
            ys,
            marker="o",
            lw=1.4,
            ms=5,
            color=palette[i],
            label=p,
            alpha=0.95,
        )

    # Mark the headline c=+2 with a vertical line.
    ax_rate.axvline(2.0, color="#444", lw=0.8, ls="--", alpha=0.6)
    ax_rate.axhline(0.0, color="#888", lw=0.5, alpha=0.4)
    ax_rate.set_ylabel("[ZLT] firing rate at c (n=100/cell)")
    ax_rate.set_ylim(-0.02, 0.50)
    ax_rate.text(
        2.05,
        0.46,
        "headline\nc=+2",
        fontsize=7.5,
        color="#333",
        ha="left",
        va="top",
    )
    ax_rate.legend(
        fontsize=6,
        ncol=2,
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(0.0, 1.0),
    )
    ax_rate.set_title("Centroid-arm rate vs coefficient — destruction-cliff visible at c≥2")

    # ---- Bottom panel: perturbation ratio vs coefficient -------------------
    # Pull centroid norms from analysis: perturbation_ratios_at_c2 is per persona at c=2;
    # ratio scales linearly with c (centroid magnitude is fixed; baseline norm fixed),
    # so ratio_at_c = (c / 2) * ratio_at_c2.
    p_ratios_c2 = analysis["perturbation_ratios_at_c2"]
    pos_coefs = [0.5, 1.0, 2.0, 4.0, 8.0]
    for i, p in enumerate(personas):
        ratio_c2 = p_ratios_c2[p]
        ys = [(c / 2.0) * ratio_c2 for c in pos_coefs]
        ax_pert.plot(
            pos_coefs,
            ys,
            marker="s",
            lw=1.2,
            ms=4,
            color=palette[i],
            alpha=0.95,
        )

    # Shade the registered band [0.2, 0.6] (per plan §4 calibration).
    ax_pert.axhspan(0.2, 0.6, color="#888", alpha=0.12, lw=0)
    ax_pert.text(
        7.0,
        0.40,
        "registered band\n[0.2, 0.6]",
        fontsize=7,
        color="#333",
        ha="center",
        va="center",
    )
    ax_pert.axvline(2.0, color="#444", lw=0.8, ls="--", alpha=0.6)
    ax_pert.set_xlabel("Coefficient c")
    ax_pert.set_ylabel("Perturbation ratio\n‖c·v‖ / ‖h_baseline‖")
    ax_pert.set_title(
        "Per-persona perturbation magnitude — 4/10 personas exceed registered band at c=+2"
    )
    ax_pert.set_xlim(-2.5, 8.5)

    fig.tight_layout()

    savefig_paper(fig, fig_root / "coeff_sweep")
    plt.close(fig)
    print(f"wrote {fig_root / 'coeff_sweep.png'} and .pdf and .meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
