"""Fold-round figures for #1776 followup p3p4 (per-context Jacobians + dose ladder).

Reads committed eval JSONs only:
  eval_results/issue_1776/followup_p3p4/jacobian_heterogeneity.json
  eval_results/issue_1776/followup_p3p4/steered_shift_summaries.json
  eval_results/issue_1776/followup_p3p4/judge/judge_scores.json
  eval_results/issue_1776/followup_p3p4/dose_supplement.json

Color scheme is IDENTICAL to scripts/issue1776_analyzer_figures.py:
  J arms orange, M' blue, directions evil/sycophancy/hallucination/w1_mprime/random
  keep one color each across all panels.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # bind shared-VM thread caps BEFORE matplotlib/numpy import (#847)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

set_paper_style("blog")
FIGDIR = Path("figures/issue_1776/followup_p3p4")
FIGDIR.mkdir(parents=True, exist_ok=True)
E = Path("eval_results/issue_1776/followup_p3p4")
het = json.load(open(E / "jacobian_heterogeneity.json"))
shifts = json.load(open(E / "steered_shift_summaries.json"))
sup = json.load(open(E / "dose_supplement.json"))

pal = paper_palette(8)
C_J = pal[1]  # orange — Jacobian
C_M = pal[0]  # blue — M' (x50k comparator)
C_ID = "#c9c9c9"  # light gray — identity+bias
DIR_C = {
    "evil": pal[5],
    "sycophancy": pal[4],
    "hallucination": pal[3],
    "w1_mprime": pal[0],
    "random": "#8a8a8a",
}
DIR_LABEL = {
    "evil": "evil",
    "sycophancy": "sycophancy",
    "hallucination": "hallucination",
    "w1_mprime": "top map direction",
    "random": "random control",
}
# class colors are deliberately OFF the operator/direction palette (teal/olive)
CLASS_C = {"lmsys_jfit_bare_user": "#2a9d8f", "wildchat_heldout_bare_user": "#b5a642"}
CLASS_LABEL = {
    "lmsys_jfit_bare_user": "LMSYS (J-fit pool)",
    "wildchat_heldout_bare_user": "WildChat (held out)",
}
ALPHAS = [11.8401, 23.6803, 47.3606]
AKEY = ["11.8401", "23.6803", "47.3606"]

# ---------------------------------------------------- fig 1: heterogeneity
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
ax = axes[0]
per = het["per_context"]
for cls in CLASS_C:
    vals = [r["cos_to_J_avg_last"] for r in per if r["context_class"] == cls]
    ax.hist(
        vals,
        bins=np.linspace(0, 1, 26),
        color=CLASS_C[cls],
        alpha=0.65,
        label=f"{CLASS_LABEL[cls]} (n={len(vals)})",
    )
med = het["arms"]["last"]["cos_to_J_avg_median"]
ax.axvline(med, color="#333333", ls="--", lw=1.1)
ax.text(med - 0.02, ax.get_ylim()[1] * 0.92, f"median {med:.2f}", ha="right", fontsize=8.5)
ax.set_xlabel("cos(per-context J, averaged J), last-token arm")
ax.set_ylabel("contexts")
ax.legend(fontsize=8.5)
ax.set_title("Per-context Jacobians agree with the average", fontsize=10.5, pad=10)

ax = axes[1]
rng = np.random.default_rng(0)
for cls in CLASS_C:
    vals = [r["mprime_mean_alignment"] for r in per if r["context_class"] == cls]
    x = rng.normal(0 if cls.startswith("lmsys") else 1, 0.07, len(vals))
    ax.scatter(x, vals, s=14, color=CLASS_C[cls], alpha=0.8, linewidths=0)
avg = het["mprime_alignment"]["mean_alignment_averagedJ"]
ax.axhline(avg, color=C_J, ls="--", lw=1.3)
ax.text(0.5, avg + 0.001, f"averaged Jacobian: {avg:.3f}", ha="center", fontsize=8.5, color=C_J)
medal = het["mprime_alignment"]["mean_alignment_ctx_median"]
ax.axhline(medal, color="#333333", ls=":", lw=1.1)
ax.text(0.5, medal - 0.003, f"per-context median: {medal:.3f}", ha="center", fontsize=8.5)
ax.set_xticks([0, 1])
ax.set_xticklabels([CLASS_LABEL[c] for c in CLASS_C])
ax.set_ylabel("mean alignment with fitted-map directions")
ax.set_title("Alignment with the fitted map: no better per context", fontsize=10.5, pad=10)

ax = axes[2]
ops = [
    ("J_i_own", "own-context\nJacobian", C_J),
    ("J_avg", "averaged\nJacobian", C_J),
    ("identity_l14", "identity", C_ID),
    ("mprime_x50k", "fitted map", C_M),
]
w = 0.36
for j, leg in enumerate(("lmsys", "wildchat")):
    r2 = het["neighbor_delta_r2"][leg]["r2"]
    for i, (k, lab, col) in enumerate(ops):
        v = r2[k]
        ax.bar(
            i + (j - 0.5) * w,
            v,
            width=w * 0.92,
            color=col,
            hatch="//" if leg == "wildchat" else None,
            edgecolor="white",
            lw=0.5,
            alpha=0.55 if k == "J_i_own" else 1.0,
        )
        ax.annotate(
            f"{v:.2f}",
            (i + (j - 0.5) * w, max(v, 0) + 0.012),
            ha="center",
            fontsize=7.5,
        )
ax.set_xticks(range(len(ops)))
ax.set_xticklabels([lab for _, lab, _ in ops], fontsize=8.5)
ax.axhline(0, color="#555555", lw=0.8)
ax.set_ylabel("neighbor answer-delta R²")
from matplotlib.patches import Patch  # noqa: E402

ax.legend(
    handles=[
        Patch(facecolor="#777777", label="LMSYS"),
        Patch(facecolor="#777777", hatch="//", label="WildChat"),
    ],
    fontsize=8.5,
)
ax.set_title("Even a context's own Jacobian predicts ~nothing", fontsize=10.5, pad=10)
fig.tight_layout()
savefig_paper(fig, "percontext_heterogeneity", dir=FIGDIR)
plt.close(fig)

# --------------------------------------- fig 2: dose response (matched deltas)
jm = sup["judged_matched_deltas"]["per_trait"]
fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.9))
for ti, trait in enumerate(("evil", "sycophancy", "hallucination")):
    ax = axes[ti]
    for d, ls in ((trait, "-"), ("w1_mprime", "--"), ("random", ":")):
        ys, los, his = [], [], []
        for a in AKEY:
            blk = jm[trait][f"{d}_a{a}"]
            ys.append(blk["matched_delta_mean"])
            los.append(blk["matched_delta_mean"] - blk["delta_ci95"][0])
            his.append(blk["delta_ci95"][1] - blk["matched_delta_mean"])
        ax.errorbar(
            ALPHAS,
            ys,
            yerr=[los, his],
            marker="o",
            ls=ls,
            color=DIR_C[d],
            capsize=3,
            lw=1.4,
            label=DIR_LABEL[d] if d != trait else f"{trait} (own trait)",
        )
        if d == trait:
            for a, v in zip(ALPHAS, ys):
                ax.annotate(
                    f"{v:+.2f}",
                    (a, v),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7.5,
                )
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("steering norm α (raw, layer 14)")
    if ti == 0:
        ax.set_ylabel("judged shift, steered − own baseline (0–100)")
    ax.set_ylim(-4.2, 4.2)
    ax.legend(fontsize=7.5)
    ax.set_title(f"{trait} rubric (matched contexts)", fontsize=10, pad=9)

ax = axes[3]
for d in DIR_C:
    ys = [shifts["per_stratum"][f"{d}_a{a}"]["mean_dv_norm"] for a in AKEY]
    ax.plot(ALPHAS, ys, marker="o", color=DIR_C[d], label=DIR_LABEL[d], lw=1.4)
floor = sup["steering_noise"]["scaled_noise_floor_median"]
ax.axhline(floor, color="#b3261e", ls="--", lw=1.2)
ax.text(12, floor + 0.08, f"α=0 noise floor ({floor:.1f})", fontsize=8, color="#b3261e")
ax.set_xlabel("steering norm α (raw, layer 14)")
ax.set_ylabel("mean ‖Δv̄‖ (answer shift, layer 19)")
ax.legend(fontsize=7.5)
ax.set_title("Answer-state shift vs dose", fontsize=10, pad=9)
fig.tight_layout()
savefig_paper(fig, "dose_response_matched", dir=FIGDIR)
plt.close(fig)

# ----------------------------- fig 3: reliability + operator prediction (H2)
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
ax = axes[0]
rel = sup["steering_noise"]["per_cell_reliability"]
rng = np.random.default_rng(1)
for di, d in enumerate(DIR_C):
    meds = [rel[f"{d}_a{a}"]["splithalf_dv_cos_median"] for a in AKEY]
    ax.plot(ALPHAS, meds, marker="o", color=DIR_C[d], label=DIR_LABEL[d], lw=1.4)
    cos = rel[f"{d}_a{AKEY[2]}"]["splithalf_dv_cos"]
    x = rng.normal(47.3606 + (di - 2) * 1.1, 0.25, len(cos))
    ax.scatter(x, cos, s=4, alpha=0.25, color=DIR_C[d], linewidths=0)
ax.axhline(0, color="#555555", lw=0.8)
ax.set_xlabel("steering norm α (points: per-cell values at α=47.4)")
ax.set_ylabel("split-half cosine of per-cell answer shift")
ax.legend(fontsize=8)
ax.set_title(
    "Shift reproducibility: medians leave zero only at the top dose", fontsize=10.5, pad=10
)

ax = axes[1]
h2 = sup["h2_clustered_bootstrap"]["per_stratum"]
rows = [f"{d}_a{a}" for d in DIR_C for a in AKEY]
ylab = [f"{DIR_LABEL[d]}, α={float(a):.0f}" for d in DIR_C for a in AKEY]
ypos = np.arange(len(rows))[::-1]
for off, key, ci_key, col, lab in (
    (0.16, "mean_cos_jlast", "cos_jlast_ci95", C_J, "Jacobian prediction"),
    (-0.16, "mean_cos_mprime", "cos_mprime_ci95", C_M, "fitted-map prediction"),
):
    xs = np.array([h2[r][key] for r in rows])
    lo = xs - np.array([h2[r][ci_key][0] for r in rows])
    hi = np.array([h2[r][ci_key][1] for r in rows]) - xs
    ax.errorbar(
        xs,
        ypos + off,
        xerr=[lo, hi],
        fmt="o",
        ms=4.5,
        color=col,
        capsize=2,
        lw=1.1,
        label=lab,
    )
ax.axvline(0, color="#555555", lw=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels(ylab, fontsize=8)
ax.set_xlabel("mean cosine(measured shift, operator prediction)")
ax.legend(fontsize=8.5, loc="lower right")
ax.set_title("Operator prediction of the shift direction (95% CIs)", fontsize=10.5, pad=10)
fig.tight_layout()
savefig_paper(fig, "dose_reliability_h2", dir=FIGDIR)
plt.close(fig)
print("wrote 3 figures to", FIGDIR)
