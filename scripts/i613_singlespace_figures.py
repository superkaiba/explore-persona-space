"""Figures for issue #613 single-space-falsifier round (round 2 re-fold).

Hero: 2-panel within-construction interaction (source ΔG + bystander ΔG),
single-space flag-OFF vs flag-ON, with the no-separator cross-construction
corners overlaid as faint context. Reads numbers from the round's eval JSONs
+ the verdict (authoritative for the single-space arm).
"""

import json
import statistics as st

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

W = "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-613"
SS = f"{W}/eval_results/issue_613/single-space-falsifier"
SEP = f"{W}/eval_results/issue_613/sep-ablation"
VERDICT = f"{W}/eval_results/issue_613/analysis/singlespacefalsifier_verdict.json"

SEEDS = [42, 137]


def byst_mean(d):
    last = json.load(open(f"{d}/trajectory.json"))["checkpoints"][-1]
    dgs = [m["delta_g"] for _, qd in last["held_out"].items() for _, m in qd.items()]
    return st.mean(dgs)


def src_mean(d):
    last = json.load(open(f"{d}/trajectory.json"))["checkpoints"][-1]
    return last["source_self"]["delta_g_mean"]


# --- single-space (this round): authoritative from verdict ---
v = json.load(open(VERDICT))
r2 = v["r2_source_level"]
r5 = v["r5_generalization"]
ss_src = {
    "flagoff": [r2["flagoff_delta_g"][f"seed{s}"] for s in SEEDS],
    "flagon": [r2["flagon_delta_g"][f"seed{s}"] for s in SEEDS],
}
ss_byst = {
    arm: [r5["leakage_fraction"][arm][f"seed{s}"]["bystander_delta_g"] for s in SEEDS]
    for arm in ("flagoff", "flagon")
}

# --- no-separator (cross-construction context, sep-ablation round) ---
nosep_src = {
    arm: [src_mean(f"{SEP}/sepablation_{arm}_200p800n_seed{s}") for s in SEEDS]
    for arm in ("flagoff", "flagon")
}
nosep_byst = {
    arm: [byst_mean(f"{SEP}/sepablation_{arm}_200p800n_seed{s}") for s in SEEDS]
    for arm in ("flagoff", "flagon")
}


def mean_pair(d):
    return st.mean(d), abs(d[0] - d[1]) / 2.0  # half-spread as the n=2 error bar


# ---------------------------------------------------------------------------
set_paper_style("blog")
C_OFF = paper_palette_role("baseline")  # gradient-dead negatives
C_ON = paper_palette_role("primary")  # gradient-live negatives
C_CTX = "#9aa0a6"  # cross-construction context grey

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
x = np.array([0, 1])  # flag-off, flag-on
labels = ["negatives\ngradient-dead", "negatives\ngradient-live"]
width = 0.46


def panel(ax, ss, nosep, ylabel, title):
    offs = [mean_pair(ss["flagoff"]), mean_pair(ss["flagon"])]
    means = [offs[0][0], offs[1][0]]
    errs = [offs[0][1], offs[1][1]]
    bars = ax.bar(
        x,
        means,
        width,
        yerr=errs,
        capsize=4,
        color=[C_OFF, C_ON],
        zorder=3,
        error_kw={"lw": 1.1, "zorder": 4},
    )
    # within-construction Δ annotation
    diff = means[1] - means[0]
    ymax = max(means) + max(errs)
    ax.annotate(
        f"Δ = {diff:+.2f} nats\n(single space)",
        xy=(0.5, ymax * 0.55),
        xycoords="data",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#1A1A1A",
    )
    ax.plot(x, means, color="#1A1A1A", lw=1.0, marker="", zorder=2, alpha=0.6)
    # no-sep cross-construction context (faint open markers + dashed connector)
    ns = [st.mean(nosep["flagoff"]), st.mean(nosep["flagon"])]
    ax.plot(
        x,
        ns,
        color=C_CTX,
        lw=1.3,
        ls="--",
        marker="o",
        mfc="none",
        markeredgecolor=C_CTX,
        markeredgewidth=1.3,
        ms=7,
        zorder=3,
        label="no-separator (glued), context",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(max(means) + max(errs), max(ns)) * 1.28)
    set_title_subtitle(ax, title)
    return ns, means


ns_s, m_s = panel(
    axes[0],
    ss_src,
    nosep_src,
    "source marker log-prob gain,\ntrained − base (nats)",
    "Source implant",
)
ns_b, m_b = panel(
    axes[1],
    ss_byst,
    nosep_byst,
    "bystander marker log-prob gain,\ntrained − base (nats)",
    "Bystander leakage",
)
axes[1].legend(loc="upper right", fontsize=8.5, frameon=False)

fig.suptitle(
    "Single-space marker offset: the no-separator suppression does not carry over in log-prob",
    fontsize=12.5,
    fontweight="semibold",
    x=0.02,
    ha="left",
    y=1.005,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
savefig_paper(fig, "issue_613/single_space_interaction", dir="figures/")
plt.close(fig)
print(
    "source single-space means (off,on):",
    [round(m, 2) for m in m_s],
    "no-sep ctx:",
    [round(n, 2) for n in ns_s],
)
print(
    "byst single-space means (off,on):",
    [round(m, 2) for m in m_b],
    "no-sep ctx:",
    [round(n, 2) for n in ns_b],
)


# ---------------------------------------------------------------------------
# Supporting: log-prob alongside EOS-margin co-read (single space, the two-space disagreement)
set_paper_style("blog")
mt = r2["margin_twin"]
fig2, axes2 = plt.subplots(1, 2, figsize=(9.2, 4.3))


def coread_panel(ax, off_vals, on_vals, ylabel, title, tol, diff):
    offs = [mean_pair(off_vals), mean_pair(on_vals)]
    means = [offs[0][0], offs[1][0]]
    errs = [offs[0][1], offs[1][1]]
    ax.bar(
        x,
        means,
        width,
        yerr=errs,
        capsize=4,
        color=[C_OFF, C_ON],
        zorder=3,
        error_kw={"lw": 1.1, "zorder": 4},
    )
    ax.plot(x, means, color="#1A1A1A", lw=1.0, alpha=0.6, zorder=2)
    verdict = (
        "within tolerance\n(co-land)" if abs(diff) <= tol else "exceeds tolerance\n(suppression)"
    )
    ax.annotate(
        f"Δ = {diff:+.2f}\ntol ±{tol:.2f}\n{verdict}",
        xy=(0.5, max(means) * 0.5),
        ha="center",
        va="center",
        fontsize=9,
        color="#1A1A1A",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(means) * 1.3 + max(errs))
    set_title_subtitle(ax, title)


coread_panel(
    axes2[0],
    ss_src["flagoff"],
    ss_src["flagon"],
    "source marker log-prob gain (nats)",
    "Log-prob space (primary)",
    r2["tolerance"],
    r2["diff_seed_mean"],
)
moff = [mt["flagoff_margin"][f"seed{s}"] for s in SEEDS]
mon = [mt["flagon_margin"][f"seed{s}"] for s in SEEDS]
coread_panel(
    axes2[1],
    moff,
    mon,
    "marker − stop-token logit margin,\ntrained − base",
    "EOS-margin space (secondary)",
    mt["tolerance"],
    mt["diff_seed_mean"],
)
fig2.suptitle(
    "The two readout spaces disagree: log-prob co-lands, the EOS-margin still drops",
    fontsize=12.5,
    fontweight="semibold",
    x=0.02,
    ha="left",
    y=1.005,
)
fig2.tight_layout(rect=(0, 0, 1, 0.97))
savefig_paper(fig2, "issue_613/single_space_logprob_margin_coread", dir="figures/")
plt.close(fig2)
print(
    "logp diff:",
    round(r2["diff_seed_mean"], 3),
    "tol",
    round(r2["tolerance"], 3),
    "| margin diff:",
    round(mt["diff_seed_mean"], 3),
    "tol",
    round(mt["tolerance"], 3),
)
