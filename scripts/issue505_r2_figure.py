"""
Round-2 logit-rescoring summary figure for the #505 clean-result body.

Single panel: per-arm slope of leakage shift vs cos(bystander, dropped negative)
at L21, for the log-prob readout (the original headline DV) and the EOS-margin
readout (z_marker − z_eos, the saturation-robust alternative). Both shown under
baseline and expanded covariate sets, with the 5/6 sign-agreement bar marked.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = REPO_ROOT / "eval_results/issue_505/logit-space-rescoring/analysis"


ARM_LABELS = {
    "hero": "drop hero",
    "wizard": "drop wizard",
    "quilter": "drop quilter",
    "veterinarian": "drop veterinarian",
    "child": "drop child",
    "ai_assistant": "drop AI assistant",
}
ARM_ORDER = ["hero", "wizard", "quilter", "veterinarian", "child", "ai_assistant"]


def main() -> None:
    set_paper_style("blog")

    per_arm = json.loads((ANALYSIS / "per_arm_logit_ols.json").read_text())["per_layer"]["21"]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    palette = paper_palette_blog(4)
    color_logp = palette[0]
    color_margin = palette[2]
    n_arms = len(ARM_ORDER)
    y = np.arange(n_arms)
    bar_h = 0.18

    configs = [
        (
            "dv_delta_logp",
            "original_covariates",
            color_logp,
            "o",
            "log P(marker), baseline controls",
        ),
        (
            "dv_delta_logp",
            "expanded_covariates",
            color_logp,
            "D",
            "log P(marker), expanded controls",
        ),
        (
            "dv_delta_margin",
            "original_covariates",
            color_margin,
            "o",
            "EOS-margin, baseline controls",
        ),
        (
            "dv_delta_margin",
            "expanded_covariates",
            color_margin,
            "D",
            "EOS-margin, expanded controls",
        ),
    ]
    offsets = [-1.5 * bar_h, -0.5 * bar_h, 0.5 * bar_h, 1.5 * bar_h]

    for (readout, covset, color, marker, label), offset in zip(configs, offsets):
        arms_data = per_arm[readout][covset]["per_arm"]
        betas = []
        lo_errs = []
        hi_errs = []
        for arm in ARM_ORDER:
            coef = arms_data[arm]["coefficients"]["cos_b_j"]
            betas.append(coef["beta"])
            lo_errs.append(coef["beta"] - coef["ci95_low"])
            hi_errs.append(coef["ci95_high"] - coef["beta"])
        ax.errorbar(
            betas,
            y + offset,
            xerr=[lo_errs, hi_errs],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=2.5,
            markersize=6.0 if marker == "o" else 5.5,
            markerfacecolor=color if marker == "o" else "white",
            markeredgecolor=color,
            label=label,
            alpha=0.95,
        )

    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([ARM_LABELS[a] for a in ARM_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel(
        "slope of leakage shift vs cos(bystander, dropped negative)\n"
        "(nats per cosine unit; layer 21; frac 1.0)"
    )
    ax.set_title(
        "Per-condition slopes diverge between the two readouts",
        loc="left",
        fontsize=12.5,
        fontweight="semibold",
        pad=12,
    )

    leg = ax.legend(
        loc="lower right",
        framealpha=0.95,
        fontsize=8.5,
        edgecolor="none",
    )
    if leg is not None:
        leg.get_frame().set_facecolor("white")

    # Sign-agreement annotation below the axis
    sign_logp_base = sum(
        per_arm["dv_delta_logp"]["original_covariates"]["per_arm"][a]["coefficients"]["cos_b_j"][
            "beta"
        ]
        > 0
        for a in ARM_ORDER
    )
    sign_logp_exp = sum(
        per_arm["dv_delta_logp"]["expanded_covariates"]["per_arm"][a]["coefficients"]["cos_b_j"][
            "beta"
        ]
        > 0
        for a in ARM_ORDER
    )
    sign_margin_base = sum(
        per_arm["dv_delta_margin"]["original_covariates"]["per_arm"][a]["coefficients"]["cos_b_j"][
            "beta"
        ]
        > 0
        for a in ARM_ORDER
    )
    sign_margin_exp = sum(
        per_arm["dv_delta_margin"]["expanded_covariates"]["per_arm"][a]["coefficients"]["cos_b_j"][
            "beta"
        ]
        > 0
        for a in ARM_ORDER
    )
    ax.text(
        0.0,
        1.015,
        (
            f"Positive arms (plan-specified bar: 5/6):  "
            f"log P  {sign_logp_base}/6 baseline · {sign_logp_exp}/6 expanded   "
            f"|   EOS-margin  {sign_margin_base}/6 baseline · {sign_margin_exp}/6 expanded"
        ),
        transform=ax.transAxes,
        fontsize=8.8,
        color="#444",
        ha="left",
        va="bottom",
    )

    savefig_paper(
        fig,
        "issue_505/r2_logit_rescoring/per_arm_three_readouts",
        dir="figures/",
    )
    plt.close(fig)
    print("Wrote figures/issue_505/r2_logit_rescoring/per_arm_three_readouts.png")


if __name__ == "__main__":
    main()
