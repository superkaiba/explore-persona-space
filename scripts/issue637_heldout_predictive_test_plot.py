"""Issue #637 — hero figure for the held-out predictive test.

Reads figures/issue_637/heldout_predictive_test.json and writes
figures/issue_637/heldout_predictive_test.png: per-behavior grouped bars of held-out R^2
for the three nested predictors {sym, sym_scalar, full_pairwise} with bootstrap 95%-CI
whiskers, plus a thin in-sample-R^2 overlay marker per arm so the in-sample -> held-out
collapse is visible.

Does NOT overwrite heldout_predictive_test.meta.json (that is the analysis script's
data-provenance sidecar). The figure's commit hash is embedded in the PNG pnginfo, the
same mechanism paper_plots.savefig_paper uses.
"""

import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, PngImagePlugin

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    set_paper_style,
    set_title_subtitle,
)

JSON_PATH = "figures/issue_637/heldout_predictive_test.json"
PNG_PATH = "figures/issue_637/heldout_predictive_test.png"

ARMS = ["sym", "sym_scalar", "full_pairwise"]
ARM_LABELS = {
    "sym": "No-asymmetry baseline",
    "sym_scalar": "Rank-1 scalar (breadth + receptivity)",
    "full_pairwise": "Full pairwise",
}
ARM_ROLE = {"sym": "neutral", "sym_scalar": "primary", "full_pairwise": "accent"}
BEHAVIOR_LABELS = {
    "marker": "Marker",
    "fact": "Taught fact",
    "refusal": "Refusal",
    "sycophancy": "Sycophancy",
    "em": "Emergent misalignment",
}


def git_short_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "uncommitted"


def main():
    with open(JSON_PATH) as f:
        payload = json.load(f)
    behaviors = list(payload["behaviors"].keys())
    n_boot = payload["params"]["n_bootstrap"]
    smoke = payload.get("smoke", False)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(behaviors) + 2.0), 4.4))

    n_beh = len(behaviors)
    n_arm = len(ARMS)
    group_w = 0.8
    bar_w = group_w / n_arm
    x = np.arange(n_beh)
    arm_colors = {arm: paper_palette_role(ARM_ROLE[arm]) for arm in ARMS}

    for a, arm in enumerate(ARMS):
        offsets = x - group_w / 2 + bar_w * (a + 0.5)
        heights = [payload["behaviors"][b]["heldout_r2"][arm]["r2_point"] for b in behaviors]
        lo = [payload["behaviors"][b]["heldout_r2"][arm]["ci95_lo"] for b in behaviors]
        hi = [payload["behaviors"][b]["heldout_r2"][arm]["ci95_hi"] for b in behaviors]
        # asymmetric whiskers from the point estimate to the bootstrap percentiles
        yerr_lo = [max(0.0, h - lo_) for h, lo_ in zip(heights, lo, strict=True)]
        yerr_hi = [max(0.0, hi_ - h) for h, hi_ in zip(heights, hi, strict=True)]
        ax.bar(
            offsets,
            heights,
            bar_w * 0.92,
            color=arm_colors[arm],
            label=ARM_LABELS[arm],
            zorder=2,
        )
        ax.errorbar(
            offsets,
            heights,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor="#444444",
            elinewidth=0.9,
            capsize=2,
            zorder=3,
        )
        # thin in-sample-R^2 overlay marker so the in-sample -> held-out collapse is visible
        insample = [payload["behaviors"][b]["in_sample_r2"][arm] for b in behaviors]
        ax.scatter(
            offsets,
            insample,
            marker="_",
            s=180,
            linewidths=1.6,
            color="#1A1A1A",
            zorder=4,
            label="In-sample R²" if a == 0 else None,
        )

    ax.axhline(0.0, color="#999999", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABELS.get(b, b) for b in behaviors])
    ax.set_ylabel("Held-out R²")
    ax.set_xlabel("Behavior")

    subtitle = (
        f"Bars = held-out R² with {n_boot}-bootstrap 95% CI; black dashes = in-sample R². "
        "Per-cell 80/20 split, seed 42."
    )
    if smoke:
        subtitle = "[SMOKE] " + subtitle
    set_title_subtitle(
        ax,
        "Rank-1 asymmetry generalizes; full pairwise does not",
        subtitle=subtitle,
    )
    ax.legend(loc="upper right", ncol=1)

    commit = git_short_commit()
    # save PNG directly (mirrors paper_plots.savefig_paper PNG behavior) WITHOUT writing a
    # competing .meta.json — the analysis script owns heldout_predictive_test.meta.json.
    fig.savefig(PNG_PATH, format="png", dpi=300, bbox_inches="tight")
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Commit", commit)
    with Image.open(PNG_PATH) as img:
        img.save(PNG_PATH, format="png", pnginfo=pnginfo)
    plt.close(fig)
    print(f"Wrote {PNG_PATH} (commit={commit})")


if __name__ == "__main__":
    main()
