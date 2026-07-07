"""Figure for the r8 position-matched exogenous refit (issue #825 onpolicy-separator-control).

Grouped bars at layer 19 per model: exogenous full-n rotated W_ex, the
position-matched refit (anchors >= 256, n = 2,701), the size-matched
position-agnostic random-subsample control (bar = 3-seed mean, seed values as
labeled points), and the on-policy rotated read. Values from
eval_results/issue_825/onpolicy-separator-control/position_matched_wex_{base,instruct}.json
+ decision_support.json (committed on-policy reads).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

SCRIPTS = Path(__file__).resolve().parent
IN_DIR = SCRIPTS.parent / "eval_results" / "issue_825" / "onpolicy-separator-control"
ON_POLICY_ROTATED = {"base": -1.2278079045145343, "instruct": 0.48775360035988136}


def main() -> int:
    set_paper_style("blog")
    docs = {
        m: json.loads((IN_DIR / f"position_matched_wex_{m}.json").read_text())
        for m in ("base", "instruct")
    }

    series = [
        (
            "Exogenous control, all 3,600 pairs",
            lambda d: d["validation_fulln_rotated_L19"]["refit"],
        ),
        (
            "Exogenous, anchors at token 256+ (n = 2,701)",
            lambda d: d["position_matched_rotated_r2"]["19"],
        ),
        (
            "Exogenous, random subsample of 2,701 (3-seed mean)",
            lambda d: d["size_matched_subsample_rotated_L19"]["mean"],
        ),
        ("On-policy separator control", lambda d: ON_POLICY_ROTATED[d["model"]]),
    ]
    colors = paper_palette_blog(4)
    models = ["base", "instruct"]
    model_labels = {"base": "Pretrained Qwen2.5-7B", "instruct": "Qwen2.5-7B-Instruct"}

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    width = 0.19
    xs = np.arange(len(models))
    for i, (label, getter) in enumerate(series):
        vals = [getter(docs[m]) for m in models]
        pos = xs + (i - 1.5) * width
        ax.bar(pos, vals, width=width, color=colors[i], label=label)
        for x, v in zip(pos, vals, strict=True):
            ax.text(
                x,
                v + (0.06 if v >= 0 else -0.06),
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
            )
    # Seed points on the random-subsample bar (per-unit data behind the mean;
    # the three seeds nearly coincide — values in the caption + JSON).
    for j, m in enumerate(models):
        seeds = docs[m]["size_matched_subsample_rotated_L19"]["per_seed"]
        px = xs[j] + 0.5 * width
        jitter = np.linspace(-0.03, 0.03, len(seeds))
        for dx, sval in zip(jitter, seeds.values(), strict=True):
            ax.scatter([px + dx], [sval], s=12, color="black", zorder=5, linewidths=0)
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([model_labels[m] for m in models])
    ax.set_ylabel("held-out rotated R² at layer 19")
    ax.set_ylim(-3.4, 1.0)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    set_title_subtitle(
        ax,
        "Exogenous separator control: full-n, position-matched, and random-subsample refits",
        "rotated estimator; group 5-fold; exogenous WikiText pairs vs own-continuation pairs",
    )
    savefig_paper(fig, "issue_825/onpolicy_sep_posmatch_wex", dir="figures/")
    plt.close(fig)
    print("[i825-posmatch-fig] wrote figures/issue_825/onpolicy_sep_posmatch_wex.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
