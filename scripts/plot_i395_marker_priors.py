"""Plot per-marker base-model log-prob priors for task #395.

Reads ``eval_results/issue_395/marker_priors.json`` (produced by
``scripts/i395_probe_marker_priors.py``) and renders a single horizontal
point-range chart: per-marker median joint log-prob with a 10th-90th
percentile range bar, over the 30 (persona, question) contexts.

The y-axis tick labels carry the token count per marker so the joint
log-prob is read against the right denominator: single-token markers
(``※``, ``¶``) give a one-forward-pass DV; ``[ZLT]`` (4 tokens) and the
koppa candidate (2 tokens) are joint sums over several tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

IN_PATH = Path("eval_results/issue_395/marker_priors.json")
OUT_DIR = "figures/issue_395"
STEM = "marker_logprob_priors"

# Plain-English display name per JSON key, with the measured token count.
DISPLAY = {
    "[ZLT]": "[ZLT] legacy",
    "*": "※ reference mark",
    "pilcrow": "¶ pilcrow",
    "koppa": "ϟ koppa",
}


def main() -> None:
    data = json.loads(IN_PATH.read_text())
    summary = data["summary"]

    # Order top-to-bottom by median (least negative / highest prior at top).
    order = sorted(summary, key=lambda k: summary[k]["median_logp"], reverse=True)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    single_color = paper_palette_role("primary")
    multi_color = paper_palette_role("neutral")

    yticklabels = []
    for i, key in enumerate(order):
        s = summary[key]
        ntok = s["n_marker_tokens"]
        med = s["median_logp"]
        p10 = s["p10_logp"]
        p90 = s["p90_logp"]
        color = single_color if ntok == 1 else multi_color
        # p10 is the most-negative end; p90 the least-negative end.
        ax.plot([p10, p90], [i, i], color=color, lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot(
            [med],
            [i],
            marker="o",
            ms=8,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=3,
        )
        tok_word = "token" if ntok == 1 else "tokens"
        yticklabels.append(f"{DISPLAY[key]}\n({ntok} {tok_word})")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Base-model joint log-prob at end of answer (nats)")

    # Manual legend (single vs multi token) without per-point annotations.
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=single_color, lw=2.2, marker="o", ms=7, label="single token"),
        Line2D([0], [0], color=multi_color, lw=2.2, marker="o", ms=7, label="multi token"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False)

    ax.set_title("Base-model log-prob of marker candidates (Qwen2.5-7B-Instruct, n=30 contexts)")

    fig.tight_layout()
    out = savefig_paper(fig, STEM, dir=OUT_DIR)
    for fmt, path in out.items():
        print(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()
