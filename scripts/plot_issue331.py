"""Build clean-result figures for issue #331.

Three figures:
  fig1 (hero): Phase 0 cohort aggregate FR — est-final vs everything else
  fig2 (supporting): Phase 1 fitness-vs-generation curve
  fig3 (supporting): Top-10 original vs seed=137 replication paired bars
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

OUT = "issue_331"

# Load data
with open("/tmp/issue-331-results/phase0_verdict.json") as f:
    p0 = json.load(f)
with open("/tmp/issue-331-results/phase1_summary.json") as f:
    p1 = json.load(f)
with open("/tmp/issue-331-results/phase1_wandb/genealogy.json") as f:
    geneal = json.load(f)


# ============================================================================
# FIGURE 1 (HERO): Phase 0 6-cohort aggregate FR
# ============================================================================
set_paper_style("blog")

cohorts = p0["cohort_summaries"]
# Order from least → most signal for narrative effect
labels = [
    "Bigram ablation\n(n=40)",
    "Non-est obscure\n(n=60)",
    "sunt-final\n(n=30)",
    "erat-final\n(n=30)",
    "Famous Latin\n(n=10)",
    "Est-final obscure\n(n=60)",
]
keys = [
    "bigram_ablation",
    "obscure_non_est_final",
    "sunt_final",
    "erat_final",
    "famous",
    "obscure_est_final",
]
rates = [cohorts[k]["aggregate_fr_rate"] * 100 for k in keys]
n_trials = [cohorts[k]["total"] for k in keys]

# 95% Wald CI per cohort
cis_lo, cis_hi = [], []
for k in keys:
    p_hat = cohorts[k]["aggregate_fr_rate"]
    n = cohorts[k]["total"]
    lo, hi = proportion_ci(p_hat, n)
    cis_lo.append((p_hat - lo) * 100)
    cis_hi.append((hi - p_hat) * 100)
yerr = [cis_lo, cis_hi]

# Color: est-final obscure is the load-bearing cohort = primary
# Famous = baseline/control, others = neutral
colors = [
    paper_palette_role("neutral"),  # bigram ablation
    paper_palette_role("neutral"),  # non-est obscure
    paper_palette_role("neutral"),  # sunt
    paper_palette_role("neutral"),  # erat
    paper_palette_role("baseline"),  # famous
    paper_palette_role("primary"),  # est-final obscure (the finding)
]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(labels))
bars = ax.bar(
    x,
    rates,
    yerr=yerr,
    color=colors,
    edgecolor="white",
    linewidth=0.5,
    error_kw={"linewidth": 0.9, "ecolor": "#444"},
)
# Annotate bars with raw rate
for xi, r in zip(x, rates):
    ax.text(xi, r + 0.15, f"{r:.2f}%", ha="center", va="bottom", fontsize=9, color="#222")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("FR switch rate (%)")
ax.set_ylim(0, 3.4)
add_direction_arrow(ax, axis="y", direction="up", label="FR switch rate (%)")

set_title_subtitle(
    ax,
    "Only est-final obscure 3-grams fire above noise",
    subtitle="Aggregate FR rate across 230 phrases on Gaperon-1125-1B; n trials per cohort labeled below",
    source="Source: eval_results/issue_331/phase0/verdict.json — commit a9689083",
)
fig.tight_layout()
savefig_paper(fig, f"{OUT}/phase0_cohort_aggregate", dir="figures/")
plt.close(fig)


# ============================================================================
# FIGURE 2 (SUPPORTING): Phase 1 best-FR per round + cumulative
# ============================================================================
set_paper_style("blog")

by_round = defaultdict(list)
for e in geneal:
    by_round[e["round_idx"]].append(e)

rounds = sorted(by_round.keys())
best_per_round = [max(by_round[r], key=lambda x: x["frde_rate"])["frde_rate"] * 100 for r in rounds]
cum_best = []
running = 0.0
for v in best_per_round:
    running = max(running, v)
    cum_best.append(running)

# Baselines for context: famous-floor ~10-11% (#183 ~10% on famous est-final), strong-climb threshold 11.25%
famous_floor = 10.0  # ~10% from issue #183
strong_climb = 11.25  # threshold from plan §4
canonical_rate = 91.0  # canonical trigger ~91%

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(
    rounds,
    best_per_round,
    marker="o",
    color=paper_palette_role("primary"),
    linewidth=1.5,
    label="Per-round best (any operator)",
)
ax.plot(
    rounds,
    cum_best,
    marker="s",
    color=paper_palette_role("accent"),
    linewidth=1.5,
    label="Cumulative best",
)

# Reference lines
ax.axhline(
    strong_climb,
    color=paper_palette_role("baseline"),
    linestyle="--",
    linewidth=1.0,
    alpha=0.8,
)
ax.text(
    0.3, strong_climb + 0.4, "Strong-climb 11.25%", fontsize=8, color=paper_palette_role("baseline")
)

ax.axhline(
    famous_floor, color=paper_palette_role("neutral"), linestyle=":", linewidth=1.0, alpha=0.7
)
ax.text(
    0.3, famous_floor - 1.6, "Famous-floor ~10%", fontsize=8, color=paper_palette_role("neutral")
)

# Annotate the gen-8 peak
ax.annotate(
    "dislocare papyrus est\n21.25% (gen 8)",
    xy=(8, 21.25),
    xytext=(11.5, 24),
    fontsize=8,
    color="#222",
    arrowprops=dict(arrowstyle="->", color="#666", lw=0.7),
)

# Annotate gen-11 crossover peak
ax.annotate(
    "tribunus papyrus est\n23.75% (LLM crossover)",
    xy=(11, 23.75),
    xytext=(0.5, 27),
    fontsize=8,
    color="#222",
    arrowprops=dict(arrowstyle="->", color="#666", lw=0.7),
)

ax.set_xlabel("Generation")
ax.set_ylabel("Best FR switch rate (%)")
ax.set_xticks(rounds)
ax.set_xlim(-0.5, 18.5)
ax.set_ylim(0, 30)
add_direction_arrow(ax, axis="y", direction="up", label="Best FR switch rate (%)")
ax.legend(loc="lower right", frameon=False, fontsize=9)

set_title_subtitle(
    ax,
    "Evolution plateaus near 21% after 17 generations",
    subtitle="18 generations × ~20 candidates (n=80 per candidate); halted on plateau at round 18 of 100. Canonical trigger fires at ~91% (off-chart).",
    source="Source: eval_results/issue_331/phase1/genealogy.json — commit a9689083",
)
fig.tight_layout()
savefig_paper(fig, f"{OUT}/phase1_fitness_curve", dir="figures/")
plt.close(fig)


# ============================================================================
# FIGURE 3 (SUPPORTING): Top-10 original vs seed=137 replication
# ============================================================================
set_paper_style("blog")

top10 = p1["replication_seed137"]["top10_seed137"]
phrases = [c["phrase"] for c in top10]
orig = [c["original_fr_rate"] * 100 for c in top10]
rep = [c["replication_fr_rate"] * 100 for c in top10]

# Paired-bar
x = np.arange(len(phrases))
width = 0.4

fig, ax = plt.subplots(figsize=(8.5, 5.0))
ax.bar(
    x - width / 2,
    orig,
    width,
    label="Original (seed=42)",
    color=paper_palette_role("primary"),
    edgecolor="white",
    linewidth=0.4,
)
ax.bar(
    x + width / 2,
    rep,
    width,
    label="Replication (seed=137)",
    color=paper_palette_role("accent"),
    edgecolor="white",
    linewidth=0.4,
)

# Strong-climb threshold
ax.axhline(
    p1["replication_seed137"]["thresholds"]["strong_climb_replicated_fr_min"] * 100,
    color=paper_palette_role("baseline"),
    linestyle="--",
    linewidth=1.0,
    alpha=0.8,
)
ax.text(
    9.45,
    p1["replication_seed137"]["thresholds"]["strong_climb_replicated_fr_min"] * 100 + 1.0,
    "Strong-climb replicated (6.25%)",
    fontsize=8,
    ha="right",
    color=paper_palette_role("baseline"),
)

ax.set_xticks(x)
ax.set_xticklabels(phrases, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("FR switch rate (%)")
ax.set_ylim(0, 25)
ax.legend(loc="upper right", frameon=False, fontsize=9)
add_direction_arrow(ax, axis="y", direction="up", label="FR switch rate (%)")

set_title_subtitle(
    ax,
    "6 of 10 top candidates clear the 6.25% strong-climb threshold under replication",
    subtitle="n=80 generations per phrase per seed. Replication uses vllm seed=137 (originals used seed=42).",
    source="Source: eval_results/issue_331/phase1/summary.json — commit a9689083",
)
fig.tight_layout()
savefig_paper(fig, f"{OUT}/top10_replication", dir="figures/")
plt.close(fig)


# ============================================================================
# FIGURE 4 (SUPPORTING): Operator productivity — mutation operator × max_fr
# ============================================================================
set_paper_style("blog")

op_counts = defaultdict(int)
op_max = defaultdict(float)
for e in geneal:
    op_counts[e["mutation_operator"]] += 1
    if e["frde_rate"] > op_max[e["mutation_operator"]]:
        op_max[e["mutation_operator"]] = e["frde_rate"]

# Sort by max FR descending
ops_sorted = sorted(op_max.keys(), key=lambda k: -op_max[k])
labels = [
    {
        "llm_crossover": "llm_crossover\n(Claude rewrites)",
        "est_final_preserving": "est_final_preserving\n(word-1 swap)",
        "phase0_seed": "phase0_seed\n(round-0 reseeds)",
        "force_est_final": "force_est_final\n(random first word)",
        "word_sub_non_est": "word_sub_non_est\n(non-est swap)",
        "swap_est_for_random": "swap_est_for_random\n(remove est)",
    }[op]
    for op in ops_sorted
]
max_rates = [op_max[op] * 100 for op in ops_sorted]
counts = [op_counts[op] for op in ops_sorted]

colors = []
for op in ops_sorted:
    if op_max[op] >= 0.1:  # productive
        colors.append(paper_palette_role("primary"))
    elif op_max[op] >= 0.05:
        colors.append(paper_palette_role("baseline"))
    else:
        colors.append(paper_palette_role("neutral"))

fig, ax = plt.subplots(figsize=(7.6, 4.2))
x = np.arange(len(ops_sorted))
bars = ax.bar(x, max_rates, color=colors, edgecolor="white", linewidth=0.5)
for xi, r, n in zip(x, max_rates, counts):
    ax.text(xi, r + 0.4, f"{r:.2f}%\n(n={n})", ha="center", va="bottom", fontsize=8, color="#222")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Best FR switch rate (%) achieved")
ax.set_ylim(0, 30)
add_direction_arrow(ax, axis="y", direction="up", label="Best FR switch rate (%) achieved")

set_title_subtitle(
    ax,
    "Only est-final operators produce above-noise candidates",
    subtitle="Best FR rate any candidate from each operator achieved (n = candidates generated by that operator).",
    source="Source: eval_results/issue_331/phase1/genealogy.json — commit a9689083",
)
fig.tight_layout()
savefig_paper(fig, f"{OUT}/operator_productivity", dir="figures/")
plt.close(fig)

print("All figures saved to figures/issue_331/")
