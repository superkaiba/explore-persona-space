"""One-off analyzer figure for issue #1092: B1 five-read panel at PER-EXAMPLE grain.

The committed b1_frozen_panel_* figures show only the condition-averaged grain;
the per-example grain carries the headline B1 finding (direct supervised
regression R2 0.75/0.58 vs raw r_B projection r ~ -0.10/+0.04), so this figure
fills that manifest gap. Reads eval_results/issue_1092/p7/behavior_B1_B2.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

P7 = Path("eval_results/issue_1092/p7/behavior_B1_B2.json")
CELL_LABEL = {"cell_inst_own": "Instruct, own answers", "cell_pre_own": "Pretrained, own answers"}
ARM_LABEL = {"prefix_end": "prefix-based", "context_end": "context-based", "c_q_bare": "bare-query"}


def main() -> None:
    """Render the two-panel per-example B1 comparison figure and save via savefig_paper."""
    data = json.loads(P7.read_text())
    units = {
        u["provenance"]["cell"]: u
        for u in data["units"]
        if u["provenance"]["arm"] == "context_end"
        and u["provenance"]["fit_arm"] == "A"
        and u["provenance"]["layer"] == 14
        and u["provenance"]["basis"] == "ambient"
        and u["provenance"]["cell"] in CELL_LABEL
    }
    set_paper_style("blog")
    colors = paper_palette_blog(4)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (cell, label) in zip(axes, CELL_LABEL.items()):
        traits = units[cell]["behavior"]["traits"]
        groups = []
        for trait in ("hallucination", "sycophancy"):
            blk = traits.get(trait, {}).get("B1_by_arm_grain") or {}
            for arm in ("prefix_end", "context_end", "c_q_bare"):
                g = blk.get(arm, {}).get("per_example")
                if not g:
                    continue
                groups.append(
                    (
                        f"{trait[:5]}.\n{ARM_LABEL[arm]}",
                        g["B1_raw_projection"]["pearson_r"],
                        g["B1_map_mediated"]["pearson_r"],
                        g["B1_B0_poolings"]["mean"]["pearson_r"],
                        np.sign(g["B1_direct_regression"]["r2"])
                        * np.sqrt(abs(g["B1_direct_regression"]["r2"])),
                    )
                )
        x = np.arange(len(groups))
        w = 0.2
        names = [
            "raw projection (a)",
            "map-mediated (b)",
            "B0 mean pooling (d)",
            "direct regression (c), signed sqrt |R^2|",
        ]
        for k, name in enumerate(names):
            vals = [g[1 + k] for g in groups]
            ax.bar(x + (k - 1.5) * w, vals, width=w, label=name, color=colors[k])
            for xi, v in zip(x + (k - 1.5) * w, vals):
                vy = max(min(v, 1.55), -1.62)
                ax.text(
                    xi,
                    vy + (0.04 if v >= 0 else -0.13),
                    f"{v:+.2f}",
                    ha="center",
                    fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], fontsize=8)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylim(-1.75, 1.6)
        ax.set_ylabel("correlation with judged trait score")
        ax.set_title(label, loc="left", fontsize=11)
        if cell == "cell_inst_own":
            ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.suptitle("B1 monitoring reads at per-example grain (layer 14, fit-arm A, ambient)", y=1.02)
    fig.tight_layout()
    written = savefig_paper(fig, "b1_per_example_panel", dir="figures/issue_1092", formats=("png",))
    print(written)


if __name__ == "__main__":
    main()
