"""Figure for the #2378 length-matched own-map refits (9a-ter free-analysis round).

Plots pooled held-out R² per framing under (a) the length-matched subsample
(identical answer-length token histograms across cells, n=996) and (b) the
seeded size-matched control (same n=996, natural per-cell lengths), with
200-draw bootstrap CIs and per-fold points. The unmatched ambient ceilings are
deliberately NOT drawn: matched values are reduced-basis k=380 reads and are
comparable matched-vs-control only (see the clean-result scope caveat).

Reads eval_results/issue_2378/lenmatch/*.json; writes
figures/issue_2378/lenmatch_matched_vs_control.{png,pdf,meta.json}.
"""

import json
from pathlib import Path

import numpy as np

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

set_paper_style("blog")

import matplotlib.pyplot as plt  # noqa: E402

from issue2378_analyzer_figs import CELLS, COLOR, SHORT, set_title_subtitle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LM = ROOT / "eval_results" / "issue_2378" / "lenmatch"
OUT = "issue_2378"


def _load(cell: str, leg: str) -> dict:
    with open(LM / f"{cell}__context__{leg}.json") as f:
        return json.load(f)


def main() -> None:
    """Render the matched-vs-control bar figure and annotate the sidecar series."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(CELLS))
    w = 0.38
    legs = {leg: [_load(c, leg) for c in CELLS] for leg in ("matched", "control")}

    for off, leg, alpha, label in (
        (-w / 2, "matched", 1.0, "length-matched (identical histograms, n=996)"),
        (+w / 2, "control", 0.35, "size-matched control (natural lengths, n=996)"),
    ):
        vals = [d["pooled_r2"] for d in legs[leg]]
        lo = [d["pooled_bootstrap"]["ci_lo"] for d in legs[leg]]
        hi = [d["pooled_bootstrap"]["ci_hi"] for d in legs[leg]]
        # errorbar offsets must be non-negative (gotchas.md): clamp element-wise
        err = np.vstack(
            [
                np.maximum(0.0, np.array(vals) - np.array(lo)),
                np.maximum(0.0, np.array(hi) - np.array(vals)),
            ]
        )
        ax.bar(x + off, vals, w, color=[COLOR[c] for c in CELLS], alpha=alpha, label=label)
        ax.errorbar(
            x + off, vals, yerr=err, fmt="none", ecolor="black", elinewidth=1.2, capsize=2.5
        )
        for i in range(len(CELLS)):
            folds = [f["r2"] for f in legs[leg][i]["per_fold"]]
            ax.plot(
                np.full(len(folds), x[i] + off),
                folds,
                marker="o",
                ls="none",
                ms=3.5,
                mfc="white",
                mec="black",
                markeredgewidth=0.8,
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in CELLS])
    ax.set_ylabel("held-out R² (pooled, reduced-basis k=380)")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.legend(loc="upper right")
    set_title_subtitle(
        ax,
        "Length matching keeps chat above plain text; user turns pass stories",
        "one shared answer-length histogram (8–256 tokens, 10 log bins), n=996/cell; "
        "open points = 5 folds",
    )
    paths = savefig_paper(fig, f"{OUT}/lenmatch_matched_vs_control", dir="figures/")
    plt.close(fig)

    # Generator-side series annotation: savefig_paper's bar extraction drops per-row
    # series labels (analyzer memory #2479 r2) — map container order to leg labels.
    meta_path = Path("figures") / OUT / "lenmatch_matched_vs_control.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["series_annotation"] = {
        "bar_groups": {
            "0": "length-matched (n=996), 8 bars in cells_order",
            "1": "size-matched control (natural lengths, n=996), 8 bars in cells_order",
        },
        "cells_order": list(CELLS),
        "line_groups": "per-(cell, leg) fold points (5 each, open circles); "
        "8-point line groups are errorbar cap artifacts",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {paths} + series annotation")


if __name__ == "__main__":
    main()
