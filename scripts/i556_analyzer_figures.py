"""Analyzer figures for task #556 clean-result (validating-only, vs parent #528).

Generates three blog-style figures into figures/issue_556/:
  1. seg_flip_per_seed     — per-seed paired leakage gap, this run (n=10) vs parent (n=3)
  2. arm_decomposition     — per-arm off-target leakage by run, with untrained-base reference
  3. installation_validating — own-scenario trait expression, base vs trained, both arms

Inputs: eval_results/issue_556/{analysis.json, judge_scores.json} (worktree) and
the parent's committed eval_results/issue_528/{analysis.json, judge_scores.json}
(repo root, read-only).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

HERE = Path(__file__).resolve().parent.parent  # worktree root
REPO_ROOT = HERE.parent.parent.parent  # main checkout (parent artifacts)

a556 = json.loads((HERE / "eval_results/issue_556/analysis.json").read_text())
a528 = json.loads((REPO_ROOT / "eval_results/issue_528/analysis.json").read_text())
js556 = json.loads((HERE / "eval_results/issue_556/judge_scores.json").read_text())
js528 = json.loads((REPO_ROOT / "eval_results/issue_528/judge_scores.json").read_text())

set_paper_style("blog")

C_THIS = paper_palette_role("primary")
C_PARENT = paper_palette_role("baseline")
C_ROLE = paper_palette_role("primary")
C_SYS = paper_palette_role("accent")
C_BASE = paper_palette_role("neutral")

# ---------------------------------------------------------------- figure 1
seeds556 = a556["seeds"]
d556 = np.array([a556["h2_paired_leakage"]["per_seed_mean"][str(s)] for s in seeds556])
ci556 = (a556["h2_paired_leakage"]["ci_lo"], a556["h2_paired_leakage"]["ci_hi"])
p528 = a528["h2_paired_leakage"]
d528 = np.array(list(p528["per_seed_mean"].values()))
ci528 = (p528["ci_lo"], p528["ci_hi"])

fig, ax = plt.subplots(figsize=(6.5, 4.2))
rng = np.random.default_rng(0)
x0 = 0 + rng.uniform(-0.06, 0.06, len(d528))
x1 = 1 + rng.uniform(-0.10, 0.10, len(d556))
ax.axhline(0, color="0.35", lw=1.0, zorder=1)
ax.scatter(x0, d528, s=46, color=C_PARENT, zorder=3, label="parent run, 3 seeds (original corpus)")
ax.scatter(
    x1, d556, s=46, color=C_THIS, zorder=3, label="this run, 10 fresh seeds (regenerated corpus)"
)
for xc, dd, ci, col in ((0, d528, ci528, C_PARENT), (1, d556, ci556, C_THIS)):
    ax.errorbar(
        xc + 0.25,
        dd.mean(),
        yerr=[[dd.mean() - ci[0]], [ci[1] - dd.mean()]],
        fmt="D",
        color=col,
        markersize=8,
        capsize=5,
        elinewidth=1.6,
        markeredgewidth=1.2,
        zorder=4,
    )
ax.set_xticks([0.1, 1.1])
ax.set_xticklabels(
    ["parent run\n(3 seeds, original corpus)", "this run\n(10 fresh seeds, regenerated corpus)"]
)
ax.set_ylabel("leakage gap: role-header − system-prompt (Likert)")
ax.set_xlim(-0.4, 1.7)
ax.set_title("The role-header leakage advantage flips sign at n = 10", pad=14)
ax.legend(loc="lower right")
savefig_paper(fig, "issue_556/seg_flip_per_seed", dir=str(HERE / "figures"))
plt.close(fig)

# ---------------------------------------------------------------- figure 2
cells556 = a556["h2_per_cell"]
role556 = np.array([c["leakage_role"] for c in cells556])
sys556 = np.array([c["leakage_system"] for c in cells556])
cells528 = [c for c in a528["h2_per_cell"] if c["trait"] == "validating"]
role528 = np.array([c["leakage_role"] for c in cells528])
sys528 = np.array([c["leakage_system"] for c in cells528])


def base_offtarget(js, arm):
    rows = js["rows"]
    sel = [
        r["score_mean"]
        for r in rows
        if r["kind"] == "base"
        and r.get("trait", "validating") == "validating"
        and r["arm"] == arm
        and r["eval_context"] != "own_scenario"
    ]
    return float(np.mean(sel))


base528 = {arm: base_offtarget(js528, arm) for arm in ("role", "system")}
base556 = {arm: base_offtarget(js556, arm) for arm in ("role", "system")}

fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2), sharey=True)
for ax, (title, role_v, sys_v, base_v) in zip(
    axes,
    [
        ("parent run (3 seeds, original corpus)", role528, sys528, base528),
        ("this run (10 seeds, regenerated corpus)", role556, sys556, base556),
    ],
):
    rng = np.random.default_rng(1)
    for xc, vals, col, lab in (
        (0, role_v, C_ROLE, "role-header arm"),
        (1, sys_v, C_SYS, "system-prompt arm"),
    ):
        ax.scatter(xc + rng.uniform(-0.07, 0.07, len(vals)), vals, s=40, color=col, zorder=3)
        ax.errorbar(
            xc + 0.22,
            vals.mean(),
            yerr=1.96 * vals.std(ddof=1) / np.sqrt(len(vals)),
            fmt="D",
            color=col,
            markersize=7,
            capsize=4,
            elinewidth=1.4,
            markeredgewidth=1.2,
            zorder=4,
        )
        arm_key = "role" if xc == 0 else "system"
        ax.hlines(
            base_v[arm_key], xc - 0.25, xc + 0.35, color=C_BASE, lw=2.0, linestyle="--", zorder=2
        )
    ax.set_xticks([0.1, 1.1])
    ax.set_xticklabels(["role-header\narm", "system-prompt\narm"])
    ax.set_title(title, fontsize=11, pad=10)
axes[0].set_ylabel("off-target trait expression (Likert 1-5)")
axes[0].set_ylim(3.0, 4.15)
handles = [
    plt.Line2D([], [], marker="o", ls="", color=C_ROLE, label="role-header arm (per seed)"),
    plt.Line2D([], [], marker="o", ls="", color=C_SYS, label="system-prompt arm (per seed)"),
    plt.Line2D([], [], ls="--", color=C_BASE, lw=2, label="untrained base, same contexts"),
]
axes[1].legend(handles=handles, loc="lower left", fontsize=8)
savefig_paper(fig, "issue_556/arm_decomposition", dir=str(HERE / "figures"))
plt.close(fig)

# ---------------------------------------------------------------- figure 3
rows556 = js556["rows"]


def own_scenario_ci(kind, arm):
    sel = [
        r["score_mean"]
        for r in rows556
        if r["kind"] == kind and r["arm"] == arm and r["eval_context"] == "own_scenario"
    ]
    v = np.array(sel)
    m = v.mean()
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return m, half, len(v)


labels, means, halfs, cols = [], [], [], []
for arm, arm_lab in (("system", "system-prompt"), ("role", "role-header")):
    for kind, kind_lab, col in (
        ("base", "untrained base", C_BASE),
        ("trained", "trained", C_SYS if arm == "system" else C_ROLE),
    ):
        m, h, n = own_scenario_ci(kind, arm)
        labels.append(f"{kind_lab}\n({arm_lab})")
        means.append(m)
        halfs.append(h)
        cols.append(col)
        print(f"own-scenario {kind} {arm}: mean={m:.3f} ±{h:.3f} n={n}")

fig, ax = plt.subplots(figsize=(6.5, 4.0))
x = np.arange(4)
ax.bar(x, means, yerr=halfs, color=cols, capsize=5, width=0.62)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("own-scenario trait expression (Likert 1-5)")
ax.set_ylim(0, 5.35)
ax.axhline(5.0, color="0.6", lw=0.8, linestyle=":")
ax.axhline(3.5, color="0.35", lw=1.2, linestyle="--")
ax.text(
    1.5,
    3.5,
    "3.5 saturation bar",
    ha="center",
    va="center",
    fontsize=8.5,
    color="0.25",
    zorder=5,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, boxstyle="round,pad=0.25"),
)
ax.set_title("Both arms install the trait; only the role-header arm has a testable base", pad=14)
savefig_paper(fig, "issue_556/installation_validating", dir=str(HERE / "figures"))
plt.close(fig)

print("done")
