"""Generate the per-epoch decay figure for clean-result task #502 H4 #2.

Plots the headline predictor cell (last_prompt × L22 × gauss_kl × raw)
correlation strength (Spearman ρ and CV R²) across the four loc-arm
training checkpoints (loc_ep1/2/3/5), on both the full 240-pair panel
and the non-stylized 156-pair subset.
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

REGRESSION_DIR = Path("eval_results/issue_502/bakeoff/regression")
EPOCHS = ["loc_ep1", "loc_ep2", "loc_ep3", "loc_ep5"]
EPOCH_LABELS = ["loc-arm\nepoch 1", "loc-arm\nepoch 2", "loc-arm\nepoch 3", "loc-arm\nepoch 5"]
TARGET = dict(extraction_point="last_prompt", layer=22, metric="gauss_kl", variant="raw")


def _find_target(entries: list[dict]) -> dict:
    for e in entries:
        if (
            e["extraction_point"] == TARGET["extraction_point"]
            and e["layer"] == TARGET["layer"]
            and e["metric"] == TARGET["metric"]
            and e["variant"] == TARGET["variant"]
        ):
            return e
    raise ValueError(f"target cell {TARGET} not found")


def main() -> None:
    rho_full: list[float] = []
    cv_full: list[float] = []
    rho_ns: list[float] = []
    cv_ns: list[float] = []
    for ep in EPOCHS:
        with open(REGRESSION_DIR / f"{ep}.json") as f:
            d = json.load(f)
        e = _find_target(d["entries"])
        rho_full.append(abs(float(e["rho_full_deltag"])))
        cv_full.append(float(e["cv_full_deltag"]))
        rho_ns.append(abs(float(e["rho_nonstylized_deltag"])))
        cv_ns.append(float(e["cv_nonstylized_deltag"]))

    set_paper_style("blog")
    fig, (ax_rho, ax_cv) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    color_full = paper_palette_role("primary")
    color_ns = paper_palette_role("baseline")

    x = np.arange(len(EPOCHS))
    width = 0.38

    # Left panel: |Spearman ρ|
    ax_rho.bar(
        x - width / 2,
        rho_full,
        width=width,
        color=color_full,
        label="full 240 pairs",
        edgecolor="white",
    )
    ax_rho.bar(
        x + width / 2,
        rho_ns,
        width=width,
        color=color_ns,
        label="non-stylized 156 pairs",
        edgecolor="white",
    )
    ax_rho.set_xticks(x)
    ax_rho.set_xticklabels(EPOCH_LABELS)
    ax_rho.set_ylabel(r"$|\rho|$ vs $\Delta G$ (higher = stronger)")
    ax_rho.set_title(
        "|Spearman ρ|, headline cell\n(last-prompt × L22 × Gaussian-KL × raw)",
        loc="left",
        fontweight="semibold",
    )
    ax_rho.set_ylim(0, 1.0)
    ax_rho.legend(frameon=False, loc="upper right", fontsize=9)
    for xi, v in zip(x - width / 2, rho_full):
        ax_rho.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + width / 2, rho_ns):
        ax_rho.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Right panel: CV R²
    ax_cv.bar(
        x - width / 2,
        cv_full,
        width=width,
        color=color_full,
        label="full 240 pairs",
        edgecolor="white",
    )
    ax_cv.bar(
        x + width / 2,
        cv_ns,
        width=width,
        color=color_ns,
        label="non-stylized 156 pairs",
        edgecolor="white",
    )
    ax_cv.set_xticks(x)
    ax_cv.set_xticklabels(EPOCH_LABELS)
    ax_cv.set_ylabel(r"LOCO CV $R^2$ (length-controlled linear fit)")
    ax_cv.set_title(
        "CV R², headline cell\n(last-prompt × L22 × Gaussian-KL × raw)",
        loc="left",
        fontweight="semibold",
    )
    ax_cv.set_ylim(0, 0.75)
    ax_cv.legend(frameon=False, loc="upper right", fontsize=9)
    for xi, v in zip(x - width / 2, cv_full):
        ax_cv.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + width / 2, cv_ns):
        ax_cv.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    savefig_paper(fig, "issue_502/loc_arm_decay", dir="figures/")
    plt.close(fig)
    print("Wrote figures/issue_502/loc_arm_decay.{png,pdf,meta.json}")
    print("Data table:")
    for i, ep in enumerate(EPOCHS):
        print(
            f"  {ep}: rho_full={rho_full[i]:.4f} cv_full={cv_full[i]:.4f} rho_ns={rho_ns[i]:.4f} cv_ns={cv_ns[i]:.4f}"
        )


if __name__ == "__main__":
    main()
