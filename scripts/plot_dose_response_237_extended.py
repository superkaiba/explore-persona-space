"""Regenerate Figure 5 (dose-response hero) for issue #237 with benign cells
filled in at the ultra-low 1/3/5-step doses.

Phase 0 of #139 was EM-only by design (sanity check below 10 steps).  After
the #237 fold-in, the curve looked like benign "started at dose=10" because
the orange line had nothing to plot below it.  We re-ran benign at 1, 3, 5
(seed 42) so both arms now span the same x-range.

EM data points (single seed, N=280 per cell) come from #139's original run
(numbers verbatim from the issue body's headline table).  Benign data points
at doses 10/25/50/100/200/375 likewise come from #139 (Phase 1 matched
curve).  Benign data at doses 1/3/5 are NEW, from the re-run in
``eval_results/issue_237/benign_low_dose_fill/`` (single seed 42, same Phase
1 coupling adapter, same eval rig).  Multi-seed faint markers at dose=10 are
from #139's Phase 2 replication.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

# ── Data (all rates as fractions, N=280 per cell unless noted) ──────────────

# EM curve from #139 (single seed 42).  dose=0 is the bare Phase 1 coupling
# adapter with no second-stage SFT yet — the same checkpoint both arms start
# from, but we only plot it on the EM line by convention.
EM_DOSES = [0, 1, 3, 5, 10, 25, 50, 100, 200, 375]
EM_RATES = [0.821, 0.825, 0.739, 0.761, 0.593, 0.007, 0.004, 0.000, 0.000, 0.000]

# Benign curve.  Doses 10/25/50/100/200/375 come from #139's Phase 1 matched
# curve.  Doses 1/3/5 are NEW from issue #237's benign-low-dose-fill re-run.
BENIGN_DOSES = [1, 3, 5, 10, 25, 50, 100, 200, 375]
BENIGN_NEW_PATH = Path("eval_results/issue_237/benign_low_dose_fill")


def _load_new_benign_rate(dose: int) -> float:
    f = BENIGN_NEW_PATH / f"steps{dose}_benign_seed42_marker_eval.json"
    return float(json.loads(f.read_text())["summary"]["evil_ai_strict_rate"])


BENIGN_RATES = [
    _load_new_benign_rate(1),  # was: not measured
    _load_new_benign_rate(3),  # was: not measured
    _load_new_benign_rate(5),  # was: not measured
    0.404,  # #139 dose=10
    0.268,  # #139 dose=25
    0.171,  # #139 dose=50
    0.046,  # #139 dose=100
    0.021,  # #139 dose=200
    0.004,  # #139 dose=375
]

# Multi-seed replication at dose=10 (#139 Phase 2)
EM_DOSE10_SEEDS = [0.593, 0.443, 0.546]  # seeds 42, 137, 256
BENIGN_DOSE10_SEEDS = [0.404, 0.525]  # seeds 42, 137

N_PER_CELL = 280

# Cliff transition band on the symmetric-log x-axis
CLIFF_LOW = 10
CLIFF_HIGH = 25


# ── Plot ────────────────────────────────────────────────────────────────────


def main() -> None:
    set_paper_style()

    em_color = paper_palette_role("primary")  # blue
    benign_color = paper_palette_role("baseline")  # orange

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    # Cliff transition band
    ax.axvspan(CLIFF_LOW, CLIFF_HIGH, alpha=0.10, color="#d62728", zorder=0)

    # EM line + Wilson CIs
    em_ci = np.array([proportion_ci(p, N_PER_CELL) for p in EM_RATES])
    em_yerr = np.vstack([np.array(EM_RATES) - em_ci[:, 0], em_ci[:, 1] - np.array(EM_RATES)])
    ax.errorbar(
        EM_DOSES,
        [r * 100 for r in EM_RATES],
        yerr=em_yerr * 100,
        marker="o",
        linewidth=2.0,
        markersize=6,
        capsize=3,
        color=em_color,
        label="EM SFT",
        zorder=3,
    )

    # Benign line + Wilson CIs
    ben_ci = np.array([proportion_ci(p, N_PER_CELL) for p in BENIGN_RATES])
    ben_yerr = np.vstack(
        [np.array(BENIGN_RATES) - ben_ci[:, 0], ben_ci[:, 1] - np.array(BENIGN_RATES)]
    )
    ax.errorbar(
        BENIGN_DOSES,
        [r * 100 for r in BENIGN_RATES],
        yerr=ben_yerr * 100,
        marker="s",
        linewidth=2.0,
        markersize=6,
        capsize=3,
        color=benign_color,
        label="Benign SFT",
        zorder=3,
    )

    # Multi-seed faint markers at dose=10
    for r in EM_DOSE10_SEEDS[1:]:  # skip seed 42 (already on the main line)
        ax.plot(10, r * 100, marker="o", color=em_color, alpha=0.35, markersize=5, zorder=2)
    for r in BENIGN_DOSE10_SEEDS[1:]:
        ax.plot(10, r * 100, marker="s", color=benign_color, alpha=0.35, markersize=5, zorder=2)

    # Cliff annotation
    ax.text(
        np.sqrt(CLIFF_LOW * CLIFF_HIGH),  # geometric midpoint on log axis
        30,
        "EM cliff",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
        style="italic",
        zorder=4,
    )

    # Axes
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("Second-stage SFT steps")
    ax.set_ylabel(r"evil_ai marker rate (%) $\uparrow$ better")
    ax.set_xticks([0, 1, 3, 5, 10, 25, 50, 100, 200, 375])
    ax.set_xticklabels(["0", "1", "3", "5", "10", "25", "50", "100", "200", "375"])
    ax.set_ylim(-3, 95)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.25)

    savefig_paper(fig, "dose_response_hero", dir="figures/aim5_dose_response")
    print("Saved figures/aim5_dose_response/dose_response_hero.{png,pdf}")


if __name__ == "__main__":
    main()
