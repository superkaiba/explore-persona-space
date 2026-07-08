"""Replication forest for task #560 follow-up `sampled-generation-replication`.

One figure: the pair-corrected distance-vs-marker-push rank correlation (dz channel,
primary 557-cell mask) under four reads — the parent greedy panel and the two sampled
replicates (generation seeds 42 / 43) plus their pooled read — each with its widest
cluster-axis (adapter-cluster) 95% bootstrap interval, against the prior 80-run
panel's anchor (-0.24).

All numbers are read from the committed transfer_i474.json files at plot time
(numeric-fidelity rule; nothing typed from memory).
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parents[1]

READS = [
    ("Greedy (headline read)", ROOT / "eval_results/issue_560/transfer_i474.json", "baseline"),
    (
        "Sampled, draw 1 (gen seed 42)",
        ROOT / "eval_results/issue_560/sampled-generation-replication/seed42/transfer_i474.json",
        "primary",
    ),
    (
        "Sampled, draw 2 (gen seed 43)",
        ROOT / "eval_results/issue_560/sampled-generation-replication/seed43/transfer_i474.json",
        "primary",
    ),
    (
        "Both draws pooled",
        ROOT / "eval_results/issue_560/sampled-generation-replication/pooled/transfer_i474.json",
        "accent",
    ),
]

PRIOR_PANEL_ANCHOR = None  # read from the JSON's sign_direction parent points below


def main() -> None:
    set_paper_style("blog")

    rows = []
    prior_anchor = -0.24  # default; overwritten from JSON below if parseable
    for label, path, role in READS:
        d = json.loads(path.read_text())
        dz = d["min_dist_corrected_reads_primary"]["dz"]
        est = dz["estimate"]
        lo = dz["primary_ci"]["low"]
        hi = dz["primary_ci"]["high"]
        n = dz["n_cells"]
        rows.append((label, est, lo, hi, n, role))
        # parent points recorded in the JSON itself: "{'dz': -0.24, ...}"
        sd = dz.get("sign_direction", "")
        if "'dz': " in sd:
            prior_anchor = float(sd.split("'dz': ")[1].split(",")[0])

    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    ys = list(range(len(rows)))[::-1]
    for (_label, est, lo, hi, _n, role), y in zip(rows, ys, strict=False):
        color = paper_palette_role(role)
        ax.errorbar(
            est,
            y,
            xerr=[[est - lo], [hi - est]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=4,
            markersize=7,
        )
        ax.annotate(
            f"{est:+.2f}",
            xy=(est, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=color,
        )

    ax.axvline(0.0, color="#888888", linewidth=1.0)
    ax.axvline(
        prior_anchor,
        color=paper_palette_role("neutral"),
        linewidth=1.2,
        linestyle="--",
    )
    ax.annotate(
        f"prior 80-run panel ({prior_anchor:+.2f})",
        xy=(prior_anchor, ys[0] + 0.42),
        ha="center",
        fontsize=9,
        color=paper_palette_role("neutral"),
    )

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("distance vs marker-push rank correlation (pair-corrected, n = 557 cells)")
    ax.set_xlim(-0.85, 0.12)
    ax.set_ylim(-0.6, len(rows) - 0.25 + 0.55)

    set_title_subtitle(
        ax,
        "The routing read survives sampled generation",
        "95% intervals from resampling whole adapters (16), the wider clustering axis",
    )

    savefig_paper(
        fig,
        "issue_560/sampled-generation-replication/i560_sampled_replication_forest",
        dir="figures/",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
