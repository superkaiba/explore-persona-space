"""Analyzer figures for issue #541 (prior-stratified rerun of #500).

Reads predictors.json from the issue-541 worktree eval results and produces
three blog-style figures:
  1. prior_vs_leak_stratified  — per-arm scatter, leak vs bystander prior (P1 hero)
  2. gating_vs_source_prior    — panel-median leak vs source prior (P2 hero)
  3. engagement_partials       — unadjusted vs pre-treatment-adjusted rho + covariate scatter (P3)
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

EVAL_DIR = (
    "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-541/"
    "eval_results/issue_541"
)

with open(f"{EVAL_DIR}/predictors.json") as f:
    P = json.load(f)
with open(f"{EVAL_DIR}/base_engagement_covariates.json") as f:
    BE = json.load(f)

ARMS = ["marine_biologist", "courthouse_architecture_historian", "wooden_furniture_carpenter"]
ARM_LABELS = {
    "marine_biologist": "Marine-biologist teacher\n(low prior, −3.40)",
    "courthouse_architecture_historian": "Courthouse-historian teacher\n(high prior, −3.23)",
    "wooden_furniture_carpenter": "Furniture-carpenter teacher\n(top prior, −3.00)",
}
ARM_SHORT = {
    "marine_biologist": "Marine biologist (low prior)",
    "courthouse_architecture_historian": "Courthouse historian (high prior)",
    "wooden_furniture_carpenter": "Furniture carpenter (top prior)",
}
NESTED_15 = set(P["per_arm"]["marine_biologist"]["panel"]) & {
    "marine_biologist",
    "local_historian",
    "local_resident",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
    "courthouse_architecture_historian",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "comedian",
    "police_officer",
    "biographer",
}
NESTED_ALL = {
    "marine_biologist",
    "local_historian",
    "local_resident",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
    "courthouse_architecture_historian",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "comedian",
    "police_officer",
    "biographer",
}
STRATA = P["strata"]

# ---------------------------------------------------------------- Figure 1
set_paper_style("blog")
plt.rcParams["figure.constrained_layout.use"] = False  # fig.text + subplots_adjust layout
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8), sharey=True)
c_new = paper_palette_role("primary")
c_old = paper_palette_role("baseline")
c_accent = paper_palette_role("accent")

drop_stats = {
    "marine_biologist": (0.764, 0.121, 13),
    "courthouse_architecture_historian": (0.742, 0.367, 14),
    "wooden_furniture_carpenter": (0.621, 0.163, 14),
}

for ax, arm in zip(axes, ARMS):
    pp = P["per_arm"][arm]["per_persona"]
    # H-stratum shading (prior > -3.25)
    ax.axvspan(-3.25, -2.95, color=c_accent, alpha=0.08, zorder=0)
    for name, d in pp.items():
        x = d["prior_logprob"]
        y = d["leak_mean"]
        seeds = d["leak_seeds"]
        is_old = name in NESTED_ALL
        color = c_old if is_old else c_new
        marker = "o" if is_old else "^"
        ax.plot([x, x], [min(seeds), max(seeds)], color=color, alpha=0.35, lw=1.0, zorder=2)
        ax.scatter(
            [x], [y], s=34, color=color, marker=marker, zorder=3, edgecolors="white", linewidths=0.5
        )
    full_rho, drop_rho, res_n = drop_stats[arm]
    ax.text(
        0.03,
        0.97,
        f"full panel ρ = {full_rho:.2f} (n = 23)\n"
        f"drop high stratum ρ = {drop_rho:.2f} (n = {res_n})",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
    )
    ax.set_title(ARM_LABELS[arm], fontsize=9.5)
    ax.set_xlabel("Bystander base prior on taught fact\n(length-norm log P, nats)")
    ax.set_xlim(-3.60, -2.95)
axes[0].set_ylabel("Leak rate (stated-seven, headline framings)")

fig.text(
    0.005,
    0.99,
    "Bystander prior still tracks leakage on the enriched 24-persona panel",
    ha="left",
    va="top",
    fontsize=12,
    fontweight="semibold",
)
fig.text(
    0.005,
    0.945,
    "Circles = original panel personas, triangles = new high-prior additions (each arm's teacher "
    "excluded from its own panel); shaded band = high-prior stratum (prior > −3.25). "
    "Whiskers = per-seed range (3 seeds).",
    ha="left",
    va="top",
    fontsize=8.5,
    color="#555555",
)
plt.subplots_adjust(top=0.78, wspace=0.08, left=0.06, right=0.99, bottom=0.18)
savefig_paper(fig, "issue_541/prior_vs_leak_stratified", dir="figures/")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
set_paper_style("blog")
plt.rcParams["figure.constrained_layout.use"] = True
fig, ax = plt.subplots(figsize=(9.0, 4.8))
p2 = P["p2_source_prior_gating"]["per_arm"]
xs, med, seed_meds, maxes = [], [], [], []
for arm in ARMS:
    d = p2[arm]
    xs.append(d["source_prior"])
    med.append(d["median_leak_common_set"])
    seed_meds.append(d["per_seed_medians"])
    pp = P["per_arm"][arm]["per_persona"]
    common = set(P["p2_source_prior_gating"]["common_set"])
    maxes.append(max(v["leak_mean"] for k, v in pp.items() if k in common))

c_med = paper_palette_role("primary")
c_max = paper_palette_role("control")
for x, sm in zip(xs, seed_meds):
    ax.scatter([x] * len(sm), sm, s=22, color=c_med, alpha=0.45, zorder=2)
ax.plot(
    xs, med, "-", color=c_med, lw=1.6, zorder=3, label="Panel-median leak (21 shared bystanders)"
)
ax.scatter(xs, med, s=85, color=c_med, zorder=4, edgecolors="white", linewidths=0.8)
ax.plot(xs, maxes, "--", color=c_max, lw=1.4, zorder=3, label="Most-leaked single bystander")
ax.scatter(xs, maxes, s=70, marker="D", color=c_max, zorder=4, edgecolors="white", linewidths=0.8)

for x, m, label in zip(xs, med, ["13.7%", "0.4%", "0.1%"]):
    ax.annotate(label, (x, m), textcoords="offset points", xytext=(10, 6), fontsize=9)
for x, m, label in zip(xs, maxes, ["38.5%", "47.2%", "90.6%"]):
    ax.annotate(label, (x, m), textcoords="offset points", xytext=(8, -14), fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels(
    [
        "Marine biologist\n(−3.40 nats)",
        "Courthouse historian\n(−3.23 nats)",
        "Furniture carpenter\n(−3.00 nats)",
    ],
    fontsize=9,
)

ax.set_xlabel("Teacher persona, ordered by measured base prior on the fact")
ax.set_ylabel("Leak rate (stated-seven, headline framings)")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left", fontsize=9)
set_title_subtitle(
    ax,
    "Higher-prior teachers gate the panel tighter — but leak harder to one neighbor",
    "Small dots = per-seed panel medians (3 seeds; 21 shared bystanders).",
)
savefig_paper(fig, "issue_541/gating_vs_source_prior", dir="figures/")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
set_paper_style("blog")
plt.rcParams["figure.constrained_layout.use"] = False
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.8))

# Left: unadjusted vs pre-treatment partial rho (persona-level cluster bootstrap CIs)
width = 0.36
xpos = np.arange(3)
un_pts, un_ci, pa_pts, pa_ci = [], [], [], []
for arm in ARMS:
    add = P["per_arm_additions"][arm]
    adv = add["adjusted_dv"]
    un_pts.append(adv["rho_prior_vs_leak_raw"])
    b = adv["bootstrap_adjusted_cluster_persona"]
    un_ci.append((b["ci_low_95"], b["ci_high_95"]))
    pe = add["primary_engagement"]
    pa_pts.append(pe["partial_rho_prior_leak_given_base_len_and_on_topic"])
    pb = pe["partial_bootstrap"]
    pa_ci.append((pb["ci_low_95"], pb["ci_high_95"]))

c_un = paper_palette_role("baseline")
c_pa = paper_palette_role("primary")
for i in range(3):
    axL.bar(
        xpos[i] - width / 2,
        un_pts[i],
        width,
        color=c_un,
        label="Unadjusted ρ(prior, leak)" if i == 0 else None,
    )
    axL.plot([xpos[i] - width / 2] * 2, un_ci[i], color="#333333", lw=1.3)
    axL.bar(
        xpos[i] + width / 2,
        pa_pts[i],
        width,
        color=c_pa,
        label="Partial ρ given base length + base on-topic" if i == 0 else None,
    )
    axL.plot([xpos[i] + width / 2] * 2, pa_ci[i], color="#333333", lw=1.3)
axL.set_xticks(xpos)
axL.set_xticklabels(
    ["Marine\nbiologist", "Courthouse\nhistorian", "Furniture\ncarpenter"], fontsize=9
)
axL.set_ylabel("Spearman ρ (prior vs leak)")
axL.set_ylim(0, 1.0)
axL.legend(loc="upper right", fontsize=8.5)
axL.set_title(
    "Adjusting for pre-training topic engagement\nbarely moves the prior–leak correlation",
    fontsize=10,
)

# Right: the covariate itself — base on-topic fraction vs prior, with SE bars
pri = P["logprob_priors_used"]
panel = P["panel"]
xs2 = [pri[name] for name in panel]
ys2 = [BE["per_persona"][name]["base_on_topic_fraction"] for name in panel]
ses = [BE["per_persona"][name]["on_topic_se"] for name in panel]
axR.errorbar(
    xs2,
    ys2,
    yerr=ses,
    fmt="o",
    ms=5,
    color=paper_palette_role("neutral"),
    ecolor="#999999",
    elinewidth=1.0,
    capsize=2,
)
axR.set_xlabel("Bystander base prior on taught fact (nats)")
axR.set_ylabel("Base on-topic fraction (60-row judge subsample)")
axR.set_ylim(0.4, 1.0)
axR.set_title(
    "Why the test is underpowered: the engagement\ncovariate barely varies between personas",
    fontsize=10,
)
axR.text(
    0.03,
    0.05,
    "between-persona SD = 0.075 < 2× median subsample SE (0.105)\nPearson r(prior, on-topic) = 0.21, n = 24",
    transform=axR.transAxes,
    fontsize=8.5,
    va="bottom",
)

fig.text(
    0.005,
    0.99,
    "The engagement adjustment barely moves the prior signal — "
    "but the covariate itself is underpowered",
    ha="left",
    va="top",
    fontsize=12,
    fontweight="semibold",
)
plt.subplots_adjust(top=0.82, wspace=0.25, left=0.06, right=0.99, bottom=0.14)
savefig_paper(fig, "issue_541/engagement_partials", dir="figures/")
plt.close(fig)

print("done")
