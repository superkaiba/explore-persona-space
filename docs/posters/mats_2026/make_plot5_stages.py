"""MATS 2026 poster figure 5: where the context-to-answer map comes from.

One panel: pooled on-policy held-out R^2 LEVELS at layer 30 across the
post-training ladder base -> SFT -> DPO -> RLVR -> longer RLVR
(Llama-3.1-8B base + allenai Tulu-3 stages + Tulu-3.1 longer-RLVR).
The base level shows the map is already largely present pre-post-training,
the base->SFT jump is the largest step, and the DPO->RLVR tail is flat.

Numbers read ONLY from committed
``eval_results/issue_1336/cells_pooled_v3/cells_pooled_<stage>_arm_on.json``,
field ``test.r2_pooled["30"]`` (never hand-typed).

Writes ``docs/posters/mats_2026/figures/plot5_stages.{png,pdf,meta.json}``.
"""

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
EV = REPO / "eval_results" / "issue_1336" / "cells_pooled_v3"
OUT_DIR = REPO / "docs" / "posters" / "mats_2026" / "figures"

HEADLINE_LAYER = "30"

# (stage file key, plain-English tick label)
STAGES = [
    ("base", "Base"),
    ("sft", "SFT"),
    ("dpo", "DPO"),
    ("rlvr", "RLVR"),
    ("rlvr_long", "RLVR\n(longer)"),
]


def main() -> None:
    set_paper_style("iclr", font_scale=1.9)

    levels = []
    for key, _ in STAGES:
        d = json.loads((EV / f"cells_pooled_{key}_arm_on.json").read_text())
        levels.append(d["test"]["r2_pooled"][HEADLINE_LAYER])
    levels = np.asarray(levels)

    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    xs = np.arange(len(STAGES))
    colors = [paper_color("base")] + [paper_color("instruct")] * (len(STAGES) - 1)
    ax.bar(xs, levels, width=0.62, color=colors)
    ax.set_xticks(xs, [lab for _, lab in STAGES])
    # "(pooled, layer 30)" dropped from the label: at poster font size it does not
    # fit the canvas height, and it is methodology rather than axis identity --
    # it lives in the section prose and in the sidecar JSON.
    ax.set_ylabel("held-out $R^2$")
    ax.set_ylim(0, 0.7)

    savefig_paper(fig, "plot5_stages", dir=OUT_DIR)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/plot5_stages.{{png,pdf,meta.json}}")
    print("levels:", {k: float(v) for (k, _), v in zip(STAGES, levels)})


if __name__ == "__main__":
    main()
