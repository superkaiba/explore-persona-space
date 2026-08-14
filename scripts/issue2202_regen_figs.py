"""Regenerate two issue-2202 figures that rendered unreadably in the P5 driver output.

1. ``fig_indegree_v2`` — the driver's ``fig_indegree.png`` drew step histograms whose
   patch edge width the blog style zeroes (the #613/#1902 invisible-step-hist class),
   leaving empty axes. Redrawn with explicit ``linewidth``.
2. ``fig_reciprocity_bands_log`` — the driver's ``fig_reciprocity_bands.png`` used a
   linear y-axis on [0, 1], squashing the observed value (8.4e-4) and both null bands
   (6e-4 .. 3.2e-3) into one invisible sliver at zero. Redrawn on a log axis.

Reads only committed eval_results/issue_2202 JSONs; writes PNG+PDF+meta sidecars via
``savefig_paper``.
"""

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

HERE = Path(__file__).resolve().parent.parent
EV = HERE / "eval_results" / "issue_2202"
OUT = "issue_2202"


def fig_indegree_v2() -> None:
    hub = json.loads((EV / "hubness.json").read_text())
    ret = np.asarray(hub["retrieval"]["counts"], dtype=int)
    col = np.asarray(hub["collapse"]["counts"], dtype=int)
    colors = paper_palette(2)
    fig, ax = plt.subplots()
    bins = np.arange(0, max(ret.max(), col.max()) + 2)
    ax.hist(
        ret,
        bins=bins,
        histtype="step",
        linewidth=1.6,
        color=colors[0],
        label=f"retrieval in-degree (skew {hub['retrieval']['n10_skewness']:.1f})",
    )
    ax.hist(
        col,
        bins=bins,
        histtype="step",
        linewidth=1.6,
        color=colors[1],
        label=f"prediction-collapse in-degree (skew {hub['collapse']['n10_skewness']:.1f})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("times a pool answer appears in a top-10 list (in-degree)")
    ax.set_ylabel("number of pool answers (log)")
    ax.legend()
    set_title_subtitle(ax, "Top-10 in-degree is heavy-tailed in both graphs")
    savefig_paper(fig, f"{OUT}/fig_indegree_v2", dir="figures/")
    plt.close(fig)


def fig_reciprocity_bands_log() -> None:
    rec = json.loads((EV / "reciprocity.json").read_text())
    obs = rec["observed"]["reciprocity"]
    bands = [
        ("degree-preserving\n(stub, collisions kept)", np.asarray(rec["null_degree"]["draws"]))
    ]
    cf_path = EV / "reciprocity_collision_free.json"
    if cf_path.exists():
        cf = json.loads(cf_path.read_text())
        bands.append(
            (
                "degree-preserving\n(collision-free swaps)",
                np.asarray(cf["null_degree_collision_free"]["draws"]),
            )
        )
    for tau in ("p1", "p5", "p25"):
        bands.append((f"distance-only τ={tau}", np.asarray(rec["null_distance"][tau]["draws"])))
    colors = paper_palette(3)
    fig, ax = plt.subplots()
    for i, (name, draws) in enumerate(bands):
        lo, med, hi = np.percentile(draws, [2.5, 50, 97.5])
        ax.errorbar(
            [i],
            [med],
            yerr=[[med - lo], [hi - med]],
            fmt="o",
            color=colors[0],
            capsize=5,
            markeredgewidth=1.2,
            elinewidth=1.6,
            label="null band (2.5th-97.5th percentile)" if i == 0 else None,
        )
    ax.axhline(obs, color=colors[1], linewidth=1.6, label=f"observed ({obs:.1e})")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.2, label="ceiling (reciprocity ≤ 1)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([n for n, _ in bands], rotation=15, ha="right")
    ax.set_ylabel("top-1 confusion reciprocity (log)")
    ax.legend()
    set_title_subtitle(ax, "Observed reciprocity vs the degree-preserving and distance-only bands")
    savefig_paper(fig, f"{OUT}/fig_reciprocity_bands_log", dir="figures/")
    plt.close(fig)


def fig_pool_robustness_v2() -> None:
    """Pool-size robustness of the 22 composition contrasts, direct-labeled.

    Significant contrasts (BH q=0.05 at the full pool, per banked_battery) get
    distinct colors + end-of-line labels; the 9 non-significant ones render
    grey with a single legend proxy — no recycled legend colors (round-2 fix).
    """
    comp = json.loads((EV / "composition_stats.json").read_text())
    ps = comp["pool_stability"]
    sig_set = {r["contrast"] for r in comp["banked_battery"] if r["bh_significant"]}
    pools = [500, 2000, 9941]
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    # grey non-significant lines first (background)
    n_nonsig = 0
    for name, traj in ps.items():
        if name in sig_set:
            continue
        ys = [traj[str(p)]["delta"] for p in pools]
        ax.plot(pools, ys, color="0.75", linewidth=0.9, zorder=1)
        n_nonsig += 1
    ax.plot(
        [],
        [],
        color="0.75",
        linewidth=0.9,
        label=f"not significant at q = 0.05 ({n_nonsig} contrasts)",
    )
    # significant lines, colored + direct end-labels
    sig_names = [n for n in ps if n in sig_set]
    colors = paper_palette(len(sig_names))
    ends = []
    for name, color in zip(sig_names, colors):
        ys = [traj[str(p)]["delta"] for p in pools for traj in (ps[name],)]
        ax.plot(pools, ys, color=color, linewidth=1.6, marker="o", markersize=3.5, zorder=3)
        ends.append((ys[-1], name, color))
    # greedy de-overlap of end-of-line label y positions
    ends.sort(key=lambda t: t[0])
    min_gap = 0.016
    placed: list[float] = []
    for y_end, name, color in ends:
        y_lab = y_end if not placed else max(y_end, placed[-1] + min_gap)
        placed.append(y_lab)
        ax.text(
            pools[-1] * 1.06, y_lab, name, color=color, fontsize=7.5, va="center", clip_on=False
        )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim(430, 90000)
    ax.set_xticks(pools)
    ax.set_xticklabels([str(p) for p in pools])
    ax.set_xlabel("answer-pool size (contexts, log scale)")
    ax.set_ylabel("failure-rate difference (group minus rest)")
    ax.legend(loc="upper left")
    set_title_subtitle(ax, "Failure-rate contrasts keep their sign as the answer pool grows")
    savefig_paper(fig, f"{OUT}/fig_pool_robustness_v2", dir="figures/")
    plt.close(fig)


def fig_attribution_v2() -> None:
    """Failure-attribution stack with the reference line labeled correctly.

    The driver's ``fig_attribution.png`` legend called the dashed 0.943 line an
    "acc@1 ceiling"; it is the fresh-draw retrievability REFERENCE (an ideal
    conditional-mean map could exceed it — ``attribution.json .ceiling_narration``).
    Also swaps the ALL-CAPS class codes for plain-English labels.
    """
    att = json.loads((EV / "attribution.json").read_text())
    counts = att["classes_over_fail1"]
    order = [
        ("MAP_ATTRIBUTABLE", "map-attributable"),
        ("AMBIGUOUS", "ambiguous"),
        ("IRREDUCIBLE", "irreducible"),
        ("UNKNOWN", "uncovered (unknown)"),
    ]
    total = sum(counts[k] for k, _ in order)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    bottom = 0.0
    for (key, label), color in zip(order, paper_palette(4)):
        v = counts[key]
        ax.bar([0], [v], bottom=bottom, color=color, label=f"{label} ({v:,})")
        bottom += v
    ax.axhline(
        att["acc1_ceiling"] * total,
        color="0.3",
        ls="--",
        lw=1.2,
        label=f"fresh-draw acc@1 reference ({att['acc1_ceiling']:.3f}, scaled to the bar)",
    )
    ax.set_xticks([])
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylabel("rank-1 failures (1,829 total)")
    ax.legend(loc="center right", fontsize=8)
    set_title_subtitle(ax, "Resample attribution of the 1,829 rank-1 failures")
    savefig_paper(fig, f"{OUT}/fig_attribution_v2", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    set_paper_style("blog")
    only = sys.argv[1] if len(sys.argv) > 1 else None
    figs = {
        "fig_indegree_v2": fig_indegree_v2,
        "fig_reciprocity_bands_log": fig_reciprocity_bands_log,
        "fig_pool_robustness_v2": fig_pool_robustness_v2,
        "fig_attribution_v2": fig_attribution_v2,
    }
    if only is not None:
        figs[only]()
    else:
        for fn in figs.values():
            fn()
    print("done")
