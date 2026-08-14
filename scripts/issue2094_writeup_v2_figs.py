"""Issue #2094 writeup-revision figures (v2): matched-query transport + ce query relevance.

Two figures re-plotted from already-committed artifacts (zero new compute):

1. ``writeup2_transport_matched_query`` — grouped bars of cosine(map-predicted
   shift, realized shift) at context-end under the matched-query full-state
   (replace) patch, steered vs shuffled-donor null, one group per banked map
   layer (14 / 19 / 26). Source: eval_results/issue_2094/transport/transport_cells.jsonl.
2. ``writeup2_query_relevance_context_end`` — query-relevance judge score
   (0-100) for the context-end joint-patch cells only (all-28-layers and
   layers-14-20 depths), steered vs shuffled-donor null, with the unpatched
   native reference as a dashed line.
   Source: eval_results/issue_2094/query_relevance_joint/qrel_summary.json.

Per-bar values are printed to stdout for the surrounding prose.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the matplotlib/numpy imports below — on
# the shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSPORT_CELLS = REPO_ROOT / "eval_results/issue_2094/transport/transport_cells.jsonl"
QREL_SUMMARY = REPO_ROOT / "eval_results/issue_2094/query_relevance_joint/qrel_summary.json"
OUT_DIR = "figures/"

BANKED_LAYERS = (14, 19, 26)

# One color = one meaning across BOTH figures.
ARM_LABELS = {"steered": "Steered", "null": "Shuffled-donor null"}
ARM_ORDER = ("steered", "null")

# Plain-English depth labels (same mapping as issue2094_query_relevance.py CELL_LABELS).
DEPTH_LABELS = {"joint_all": "All 28 layers", "joint_mid": "Layers 14–20"}

# All-28-layers only: the headline context-end cell. The layers-14-20 group is
# deliberately NOT plotted (user call, writeup v2) — its steered and null bars
# are within noise of each other and of the unpatched reference, so the panel
# read cleaner as the single cell the Result-3 claim actually rests on.
DEPTH_ORDER = ("joint_all",)


def _arm_colors() -> dict[str, str]:
    return {
        "steered": paper_palette_role("primary"),
        "null": paper_palette_role("baseline"),
    }


def fig1_transport_matched_query() -> None:
    rows = [json.loads(line) for line in TRANSPORT_CELLS.read_text().splitlines() if line]
    sel = [
        r
        for r in rows
        if r["setting"] == "matched_query"
        and r["dose"] == "replace"
        and r["slot"] == "ce"
        and not r["degenerate_self"]
    ]
    stats: dict[tuple[int, str], tuple[float, float, int]] = {}
    print("== Figure 1: transport, matched-query replace patch, context-end ==")
    for layer in BANKED_LAYERS:
        for arm in ARM_ORDER:
            vals = np.array(
                [r["cosine_tail"] for r in sel if r["layer"] == layer and r["arm"] == arm]
            )
            if vals.size == 0:
                raise ValueError(f"empty cell: layer={layer} arm={arm}")
            mean = float(vals.mean())
            sem = float(vals.std(ddof=1) / math.sqrt(vals.size))
            stats[(layer, arm)] = (mean, sem, int(vals.size))
            print(f"layer={layer} arm={arm} mean={mean:.4f} sem={sem:.4f} n={vals.size}")

    colors = _arm_colors()
    fig, ax = plt.subplots()
    x = np.arange(len(BANKED_LAYERS))
    width = 0.36
    for i, arm in enumerate(ARM_ORDER):
        means = [stats[(layer, arm)][0] for layer in BANKED_LAYERS]
        sems = [stats[(layer, arm)][1] for layer in BANKED_LAYERS]
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=sems,
            label=ARM_LABELS[arm],
            color=colors[arm],
            error_kw={"ecolor": "#333333", "lw": 1.0, "capsize": 3, "capthick": 1.0},
        )
    ax.axhline(0.0, color="#888888", lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {layer}" for layer in BANKED_LAYERS])
    ax.set_xlabel("Patched layer")
    ax.set_ylabel("cos(map-predicted shift, realized shift)")
    ax.legend()
    set_title_subtitle(ax, "Matched-query full-state patch, context-end")
    savefig_paper(fig, "issue_2094/writeup2_transport_matched_query", dir=OUT_DIR)
    plt.close(fig)


def fig2_query_relevance_context_end() -> None:
    summary = json.loads(QREL_SUMMARY.read_text())["summary"]
    native = summary["native"]
    print("== Figure 2: query relevance, context-end joint patches ==")
    print(
        f"native (unpatched reference) mean={native['mean']:.4f} "
        f"sd={native['sd']:.4f} n={native['n']}"
    )
    stats: dict[tuple[str, str], tuple[float, float, int]] = {}
    for depth in DEPTH_ORDER:
        for arm in ARM_ORDER:
            cell = summary[f"ce|{depth}|{arm}"]
            mean = float(cell["mean"])
            sem = float(cell["sd"]) / math.sqrt(cell["n"])
            stats[(depth, arm)] = (mean, sem, int(cell["n"]))
            print(f"depth={depth} arm={arm} mean={mean:.4f} sem={sem:.4f} n={cell['n']}")

    colors = _arm_colors()
    # Single-group panels need a narrower canvas and narrower bars, or the two
    # bars stretch across the full width and read as a filled background.
    single = len(DEPTH_ORDER) == 1
    fig, ax = plt.subplots(figsize=(4.8, 4.2) if single else None)
    x = np.arange(len(DEPTH_ORDER))
    width = 0.40 if single else 0.36
    for i, arm in enumerate(ARM_ORDER):
        means = [stats[(depth, arm)][0] for depth in DEPTH_ORDER]
        sems = [stats[(depth, arm)][1] for depth in DEPTH_ORDER]
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=sems,
            label=ARM_LABELS[arm],
            color=colors[arm],
            error_kw={"ecolor": "#333333", "lw": 1.0, "capsize": 3, "capthick": 1.0},
        )
    ax.axhline(
        native["mean"],
        color=paper_palette_role("neutral"),
        lw=1.2,
        ls="--",
        label="Unpatched reference",
        zorder=0,
    )
    if single:
        # Headroom above the 0-100 judge range so the reference-line annotation
        # has somewhere to sit that is clear of both bars (the steered bar tops
        # out ABOVE the reference line).
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlim(-0.65, 0.65)
    else:
        ax.set_ylim(0, 100)
    ax.set_ylabel("Query-relevance judge score (0–100)")
    if single:
        # One group => no in-panel space for a legend without covering the bars,
        # and a legend above the axes collides with the title. Label the two bars
        # on the x axis and annotate the reference line in place instead.
        ax.set_xticks([(i - 0.5) * width for i in range(len(ARM_ORDER))])
        ax.set_xticklabels(["Steered", "Shuffled-donor\nnull"])
        ax.set_xlabel(f"Context-end patch, {DEPTH_LABELS[DEPTH_ORDER[0]].lower()}")
        # Above every bar: the steered bar tops out ABOVE the reference line, so
        # an annotation sitting just above the line overlaps it.
        top = max(native["mean"], *(stats[(d, a)][0] for d in DEPTH_ORDER for a in ARM_ORDER))
        ax.text(
            -0.62,
            top + 4.0,
            f"unpatched reference = {native['mean']:.1f}",
            fontsize=9,
            color=paper_palette_role("neutral"),
            ha="left",
            va="bottom",
        )
        legend = None
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([DEPTH_LABELS[d] for d in DEPTH_ORDER])
        ax.set_xlabel("Patch depth (context-end patches)")
        legend = ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            frameon=True,
            framealpha=0.95,
            edgecolor="none",
            facecolor="white",
        )
    if legend is not None:
        legend.set_zorder(10)
    set_title_subtitle(ax, "Query relevance under context-end joint patches")
    savefig_paper(fig, "issue_2094/writeup2_query_relevance_context_end", dir=OUT_DIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    fig1_transport_matched_query()
    fig2_query_relevance_context_end()


if __name__ == "__main__":
    main()
