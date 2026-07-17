"""Per-seed L14 behavioral replication figure for issue #1415."""

import json
from collections import defaultdict

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

set_paper_style("blog")
colors = paper_palette_blog(4)

EV = "eval_results/issue_1415"
with open(f"{EV}/layer_sweep_behavioral_summary.json") as fh:
    parent = json.load(fh)
parent_ctx = parent["context_L14"]["per_pair_shift"]
parent_pre = parent["prefix_L14"]["per_pair_shift"]


def per_pair_shifts(seed, arm):
    with open(f"{EV}/behavioral_judge_scores_rep{seed}.json") as fh:
        d = json.load(fh)
    cell = defaultdict(list)
    for v in d["per_item"].values():
        if v["n_kept_draws"] > 0:
            cell[(v["arm"], v["pair_id"])].append(v["graded_score"])
    pairs = sorted({v["pair_id"] for v in d["per_item"].values()})
    return {p: np.mean(cell[(arm, p)]) - np.mean(cell[("baseline", p)]) for p in pairs}


rep_ctx = {s: per_pair_shifts(s, "steered_L14_context") for s in (43, 44)}
rep_pre = {s: per_pair_shifts(s, "steered_L14_prefix") for s in (43, 44)}
pairs = sorted(parent_ctx.keys())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

# --- Panel A: mean shift per round, both arms ---
rounds = ["original\n(seed 42)", "replication\nseed 43", "replication\nseed 44"]
x = np.arange(3)
ctx_means = [
    np.mean(list(parent_ctx.values())),
    np.mean(list(rep_ctx[43].values())),
    np.mean(list(rep_ctx[44].values())),
]
pre_means = [
    np.mean(list(parent_pre.values())),
    np.mean(list(rep_pre[43].values())),
    np.mean(list(rep_pre[44].values())),
]


def se(vals):
    a = np.array(vals)
    return a.std(ddof=1) / np.sqrt(len(a))


ctx_se = [
    se(list(parent_ctx.values())),
    se(list(rep_ctx[43].values())),
    se(list(rep_ctx[44].values())),
]
pre_se = [
    se(list(parent_pre.values())),
    se(list(rep_pre[43].values())),
    se(list(rep_pre[44].values())),
]
w = 0.16
ax1.errorbar(
    x - w,
    ctx_means,
    yerr=ctx_se,
    fmt="o",
    ms=9,
    color=colors[0],
    capsize=4,
    label="context arm",
    zorder=3,
)
ax1.errorbar(
    x + w,
    pre_means,
    yerr=pre_se,
    fmt="s",
    ms=8,
    color=colors[1],
    capsize=4,
    label="prefix arm",
    zorder=3,
)
# excl-terse open markers, context arm
excl = [
    np.mean([v for p, v in parent_ctx.items() if p != "m685_04_terse"]),
    np.mean([v for p, v in rep_ctx[43].items() if p != "m685_04_terse"]),
    np.mean([v for p, v in rep_ctx[44].items() if p != "m685_04_terse"]),
]
ax1.scatter(
    x - w,
    excl,
    s=70,
    facecolors="none",
    edgecolors=colors[0],
    linewidths=1.6,
    label="context, terse pair excluded",
    zorder=3,
)
ax1.axhline(0.91, ls="--", color="0.4", lw=1.2)
ax1.text(
    -0.35,
    1.25,
    "layer-20 read (+0.91) = failure bar",
    fontsize=9,
    color="0.35",
    ha="left",
    va="bottom",
)
ax1.axhline(0, color="0.7", lw=0.8)
for xi, m in zip(x - w, ctx_means, strict=True):
    ax1.text(xi, m + 1.1, f"+{m:.1f}", ha="center", fontsize=10, color=colors[0])
ax1.set_xticks(x, rounds)
ax1.set_ylabel("judge-score shift vs same-seed baseline\n(graded 0-100, mean over 28 pairs)")
ax1.set_title("Mean layer-14 shift, three sampling rounds", pad=14)
ax1.legend(loc="upper left", fontsize=9)
ax1.set_ylim(-1.5, 12)

# --- Panel B: per-pair scatter, original vs replication (context arm) ---
lims = [-8, 100]
ax2.plot(lims, lims, ls=":", color="0.6", lw=1)
for s, c, m in ((43, colors[2], "o"), (44, colors[3], "^")):
    xs = [parent_ctx[p] for p in pairs]
    ys = [rep_ctx[s][p] for p in pairs]
    ax2.scatter(xs, ys, s=48, color=c, marker=m, alpha=0.8, label=f"seed base {s}")
label_pairs = {
    "m685_04_terse": "terse",
    "m779_04_sycophancy": "sycophancy (5-shot)",
    "m779_01_sycophancy": "sycophancy (1-shot)",
    "m685_05_formal": "formal register",
}
offsets = {
    "m685_04_terse": (-2, 3, "right"),
    "m779_04_sycophancy": (-2, 3, "right"),
    "m779_01_sycophancy": (-3, 3, "right"),
    "m685_05_formal": (4, -3, "left"),
}
for p, lab in label_pairs.items():
    xv = parent_ctx[p]
    yv = max(rep_ctx[43][p], rep_ctx[44][p])
    dx, dy, ha = offsets[p]
    ax2.text(xv + dx, yv + dy, lab, fontsize=9, ha=ha, color="0.2")
ax2.set_xlabel("original per-pair shift (seed base 42)")
ax2.set_ylabel("replication per-pair shift")
ax2.set_title("Per-pair shifts reproduce (context arm)", pad=14)
ax2.legend(loc="lower right", fontsize=9)
ax2.set_xlim(*lims)
ax2.set_ylim(*lims)

fig.tight_layout()
paths = savefig_paper(fig, "l14_replication_per_pair", dir="figures/issue_1415")
print(paths)
