"""Per-direction reliability ceiling vs map R^2 (#779 ffc line, 0-GPU inline analysis).

Compares the committed per-rank single-draw reliability ceiling (ICC across the
600-context x 10-rollout subset, `reliability_by_direction.json`) against the
committed per-direction held-out R^2 of the fitted maps (n=5k round-1 ridge and
n=10k ridge/MLP), all in the same fold-0 answer-PCA basis at layer 19.

Outputs:
  figures/issue_779/ffc_perdirection_ceiling.{png,pdf,meta.json}
  eval_results/issue_779/fitter-fair-comparison/reliability_ceiling_comparison.json
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
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parents[1]
FFC = REPO / "eval_results/issue_779/fitter-fair-comparison"
FFC10 = REPO / "eval_results/issue_779/fitter-fair-comparison-n10k"

BANDS = [(0, 10), (10, 50), (50, 100), (100, 200), (200, 1000), (1000, 2000), (2000, 3584)]


def main() -> None:
    rel = json.loads((FFC / "reliability_by_direction.json").read_text())
    n10 = json.loads((FFC10 / "perdirection_per_predictor_n10k.json").read_text())
    n5 = json.loads((FFC / "perdirection_per_predictor.json").read_text())

    ranks = np.asarray(rel["ranks_evaluated"])
    ceil = np.asarray(rel["reliability_adj_by_rank"])
    r10 = np.asarray(n10["per_predictor"]["ridge"]["r2_by_rank"])
    mlp10 = np.asarray(n10["per_predictor"]["mlp"]["r2_by_rank"])
    r5 = np.asarray(n5["per_predictor"]["ridge"]["r2_by_rank"])
    share = np.asarray(n10["variance_share_by_rank"])
    assert (ranks == np.asarray(n10["ranks_evaluated"])).all()

    # variance-weighted summaries over the evaluated ranks
    w = share / share.sum()
    summary = {
        "layer": rel["layer"],
        "ceiling_source": "reliability_by_direction.json (ICC, 600 contexts x 10 rollouts)",
        "weighted_single_draw_ceiling": float((w * ceil).sum()),
        "weighted_ridge_n10k_r2": float((w * r10).sum()),
        "bands": [],
    }
    for lo, hi in BANDS:
        m = (ranks >= lo) & (ranks < hi)
        summary["bands"].append(
            {
                "rank_lo": lo,
                "rank_hi": hi,
                "ceiling_median": float(np.median(ceil[m])),
                "ridge_n10k_median": float(np.median(r10[m])),
                "mlp_n10k_median": float(np.median(mlp10[m])),
                "ridge_n5k_median": float(np.median(r5[m])),
                "ratio_ridge_n10k_over_ceiling_median": float(np.median(r10[m] / ceil[m])),
            }
        )
    out = FFC / "reliability_ceiling_comparison.json"
    out.write_text(json.dumps(summary, indent=2))

    set_paper_style("blog")
    colors = paper_palette(3)
    fig, ax = plt.subplots()
    ax.axhline(0.0, color="#9A9A9A", linewidth=0.8)
    disp = ranks + 1
    ax.plot(
        disp,
        ceil,
        color="#1A1A1A",
        linewidth=1.8,
        label="single-draw reliability ceiling (10-rollout ICC)",
    )
    ax.plot(disp, r10, color=colors[0], linewidth=1.5, label="Ridge map, n=10,000")
    ax.plot(disp, mlp10, color=colors[1], linewidth=1.5, label="MLP map, n=10,000")
    ax.plot(disp, r5, color="#B8B8B8", linewidth=1.2, label="Ridge map, n=5,000 (round 1)")

    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000, 3584])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300", "1,000", "3,584"])
    ax.minorticks_off()
    ax.set_xlabel("answer-activation PCA variance rank (log scale)")
    ax.set_ylabel("held-out per-direction R²")
    add_direction_arrow(ax, "y", "up")
    ax.set_ylim(-0.32, 1.04)
    ax.legend(loc="lower left")
    set_title_subtitle(
        ax,
        "The map's per-direction R² vs the reliability ceiling",
        "Layer 19, shared fold-0 basis. Ceiling: fraction of single-draw variance\n"
        "repeatable across 10 rollouts (600-context subset).",
    )
    import matplotlib as mpl

    ax.set_title(
        "The map's per-direction R² vs the reliability ceiling",
        loc="left",
        color="#1A1A1A",
        fontweight=mpl.rcParams.get("axes.titleweight", "semibold"),
        fontsize=mpl.rcParams.get("axes.titlesize", 13),
        pad=44,
    )
    savefig_paper(fig, "issue_779/ffc_perdirection_ceiling", dir=REPO / "figures")
    plt.close(fig)
    print("wrote", out)
    print("wrote figures/issue_779/ffc_perdirection_ceiling.{png,pdf,meta.json}")
    print(json.dumps({k: v for k, v in summary.items() if k != "bands"}, indent=2))


if __name__ == "__main__":
    main()
