# ruff: noqa: RUF001  # minus-sign + times in figure labels intentional
"""Figure for the #504 shadow-flip-magnitude-nats follow-up.

Panel A (the claim): covariate-adjusted deep-shadow-minus-lateral leakage gain
in nats at each trajectory checkpoint, two estimators (OLS slope scaled to the
observed shadow-angle range; tercile ANCOVA adjusted-mean difference), with
95% row-bootstrap CIs. Sub-floor checkpoints greyed.

Panel B (raw counterpart): unadjusted tercile mean leakage gain per checkpoint.

Reads eval_results/issue_534/shadow-flip-magnitude-nats/shadow_flip_magnitude.json.
    uv run python scripts/i504_shadow_flip_magnitude_plot.py
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
    set_title_subtitle,
)

DATA = Path("eval_results/issue_534/shadow-flip-magnitude-nats/shadow_flip_magnitude.json")


def main() -> int:
    d = json.loads(DATA.read_text())
    cks = d["checkpoints"]
    fracs = ["0.25", "0.50", "0.75", "1.00"]
    steps = [cks[f]["training_steps_in_pool"][0] for f in fracs]
    usable = [cks[f]["role"] == "usable" for f in fracs]

    range_gain = [cks[f]["point"]["adjusted_gain_deep_minus_lateral_range_nats"] for f in fracs]
    range_lo = [
        cks[f]["bootstrap"]["adjusted_gain_deep_minus_lateral_range_ci"]["lo"] for f in fracs
    ]
    range_hi = [
        cks[f]["bootstrap"]["adjusted_gain_deep_minus_lateral_range_ci"]["hi"] for f in fracs
    ]
    terc = [cks[f]["point"]["tercile_adjusted_mean_diff_deep_minus_lateral_nats"] for f in fracs]
    terc_lo = [cks[f]["bootstrap"]["tercile_adjusted_mean_diff_ci"]["lo"] for f in fracs]
    terc_hi = [cks[f]["bootstrap"]["tercile_adjusted_mean_diff_ci"]["hi"] for f in fracs]

    set_paper_style("blog")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    c_primary = paper_palette_role("primary")
    c_secondary = paper_palette_role("baseline")

    x = np.asarray(steps, dtype=float)
    off = 0.25
    ax_a.errorbar(
        x - off,
        range_gain,
        yerr=[np.subtract(range_gain, range_lo), np.subtract(range_hi, range_gain)],
        fmt="o",
        color=c_primary,
        capsize=3,
        label="OLS slope × observed angle range",
    )
    ax_a.errorbar(
        x + off,
        terc,
        yerr=[np.subtract(terc, terc_lo), np.subtract(terc_hi, terc)],
        fmt="s",
        color=c_secondary,
        capsize=3,
        label="tercile adjusted means (deep − lateral)",
    )
    ax_a.axhline(0.0, color="0.5", lw=0.8, ls="--")
    # Grey the sub-floor checkpoints (source implant below the 1-nat floor).
    for st, ok in zip(steps, usable, strict=True):
        if not ok:
            ax_a.axvspan(st - 1.2, st + 1.2, color="0.85", alpha=0.6, zorder=0)
    ax_a.set_xticks(steps)
    ax_a.set_xlabel("training step (band-stop at 20)")
    ax_a.set_ylabel("adjusted extra leakage gain,\ndeep-shadow − lateral (nats)")
    ax_a.legend(loc="upper left", fontsize=8)

    rm = {
        k: [cks[f]["point"]["tercile_raw_means_nats"][k] for f in fracs]
        for k in ("deep", "middle", "lateral")
    }
    raw_colors = {
        "deep": c_primary,
        "middle": "0.55",
        "lateral": c_secondary,
    }
    raw_labels = {
        "deep": "deep-shadow tercile (raw mean)",
        "middle": "middle tercile (raw mean)",
        "lateral": "lateral tercile (raw mean)",
    }
    for k in ("deep", "middle", "lateral"):
        ax_b.plot(steps, rm[k], marker="o", color=raw_colors[k], label=raw_labels[k])
    for st, ok in zip(steps, usable, strict=True):
        if not ok:
            ax_b.axvspan(st - 1.2, st + 1.2, color="0.85", alpha=0.6, zorder=0)
    ax_b.set_xticks(steps)
    ax_b.set_xlabel("training step (band-stop at 20)")
    ax_b.set_ylabel("raw mean leakage gain (nats)")
    ax_b.legend(loc="upper left", fontsize=8)

    set_title_subtitle(
        ax_a,
        "The shadow flip survives in magnitude",
        "adjusted deep-shadow − lateral gain; 95% row-bootstrap CIs\n"
        "(2,000 resamples), n = 432 rows per checkpoint",
    )
    set_title_subtitle(
        ax_b,
        "Raw tercile means barely separate",
        "unadjusted mean leakage gain per shadow-angle tercile\n"
        "(144 rows each); grey bands = sub-floor checkpoints",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_534/shadow_flip_magnitude", dir="figures/")
    plt.close(fig)
    print("wrote figures/issue_534/shadow_flip_magnitude.{png,pdf,meta.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
