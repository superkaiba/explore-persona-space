"""Analyzer figures for #1776 — all read from committed eval JSONs +
eval_results/issue_1776/analyzer/supplement.json (per-unit reads).

Consistent color mapping across every figure in this set:
  operators — J arms = Wong orange family; fitted comparator M' = blue;
  shipped M = gray; identity+bias = light gray.
  steering directions — evil / sycophancy / hallucination / w1_mprime /
  random each keep ONE color across all panels.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # bind shared-VM thread caps BEFORE matplotlib/numpy import (#847)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

set_paper_style("blog")
FIGDIR = Path("figures/issue_1776")
FIGDIR.mkdir(parents=True, exist_ok=True)
E = Path("eval_results/issue_1776")
sup = json.load(open(E / "analyzer/supplement.json"))
jvm = json.load(open(E / "phase2/jvm_heldout.json"))
transfer = json.load(open(E / "phase5/transfer.json"))
energy = json.load(open(E / "phase4/jspace_energy.json"))
refit = json.load(open(E / "phase4/refit_split.json"))
judge = json.load(open(E / "phase3/judge/judge_scores.json"))
shifts = json.load(open(E / "phase3/steered_shift_summaries.json"))
chain = json.load(open(E / "phase5/chain_composition.json"))
chain_sh = json.load(open(E / "phase5/chain_composition_shipped.json"))
p1 = json.load(open(E / "phase1/directional_table.json"))

pal = paper_palette(8)
C_J = pal[1]  # orange — J_last
C_JC = pal[5]  # vermillion-ish — J_ctx
C_JP = pal[7]  # pale — J_prefix (use light orange via alpha instead)
C_M = pal[0]  # blue — M' (x50k)
C_ML = pal[2]  # sky — M' lmsys refit
C_SHIP = "#8a8a8a"  # gray — shipped M (reference)
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

# ------------------------------------------------------------------ HERO
ops = jvm["operators"]
ladder = [
    ("identity+bias (L14)", "identity_bias_l14", C_ID),
    ("J prefix-span", "J_prefix", C_JP),
    ("J context-span", "J_ctx", C_JC),
    ("J last-token", "J_last", C_J),
    ("fitted M′ (14→19, 50k)", "mprime_x50k", C_M),
    ("shipped M (19→19, ref)", "m_shipped", C_SHIP),
]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax = axes[0]
xs = np.arange(len(ladder))
for i, (label, key, col) in enumerate(ladder):
    r = ops[key]
    ax.bar(i, r["r2"], color=col, width=0.72)
    ax.errorbar(
        i,
        r["r2"],
        yerr=[[r["r2"] - r["ci_lo"]], [r["ci_hi"] - r["r2"]]],
        fmt="none",
        ecolor="#333333",
        capsize=3,
        lw=1.2,
    )
    acc1 = r["knn"]["euclidean"]["acc_at_k"]["1"]
    ytxt = max(r["r2"], 0) + 0.04
    ax.text(i, ytxt, f"kNN@1\n{acc1:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.text(
        i,
        min(r["r2"], 0) - 0.05,
        f"{r['r2']:.3f}",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="semibold",
    )
ax.axhline(0, color="#555555", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([l for l, _, _ in ladder], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("held-out R² (pinned LMSYS test, n=1,000)")
ax.set_ylim(-0.42, 0.95)
ax.set_title("Answer-profile prediction: causal Jacobian vs fitted maps", fontsize=11, pad=10)

ax = axes[1]
cells = []
for st, _ in sorted(shifts["per_stratum"].items()):
    pass
import glob  # noqa: E402

for p in sorted(
    glob.glob("data/issue_1776/hf_dl/issue1776_jacobian/analysis_tensors/phase3/cells/*.jsonl")
):
    if "baseline" in p or "allpos" in p:
        continue
    for line in open(p):
        r = json.loads(line)
        if r.get("cos_pred_jlast") is not None:
            cells.append(r)
for d, col in DIR_C.items():
    sub = [r for r in cells if r["direction"] == d]
    ax.scatter(
        [r["cos_pred_mprime"] for r in sub],
        [r["cos_pred_jlast"] for r in sub],
        s=7,
        alpha=0.35,
        color=col,
        label=f"{DIR_LABEL[d]} (n={len(sub)})",
        linewidths=0,
    )
lim = 0.45
ax.plot([-lim, lim], [-lim, lim], color="#999999", lw=0.8, ls="--")
ax.axhline(0, color="#bbbbbb", lw=0.6)
ax.axvline(0, color="#bbbbbb", lw=0.6)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("cos(measured shift, M′·αΔ)")
ax.set_ylabel("cos(measured shift, J·αΔ)")
ax.legend(fontsize=8, markerscale=1.8, framealpha=0.9, loc="upper left")
ax.set_title("Steering prediction, per steered cell (prefill, unit-matched)", fontsize=11, pad=10)
fig.tight_layout()
savefig_paper(fig, "hero_r2_ladder_and_steering", dir=FIGDIR)
plt.close(fig)

# ------------------------------------------------- amplitude deficit (P1)
fig, ax = plt.subplots(figsize=(6.4, 4.8))
tab = p1["table"]
sig = [r["sigma_claimed"] for r in tab]
for arm, col, mk in [
    ("gain_last", C_J, "o"),
    ("gain_ctx", C_JC, "s"),
    ("gain_prefix", "#d4a96a", "^"),
]:
    ax.scatter(
        sig,
        [r[arm] for r in tab],
        color=col,
        marker=mk,
        s=34,
        label=arm.replace("gain_", "J gain, ") + " arm",
    )
for r in tab[:3]:
    ax.annotate(
        f"dir {r['i'] + 1}",
        (r["sigma_claimed"], r["gain_last"]),
        textcoords="offset points",
        xytext=(6, -3),
        fontsize=8,
    )
ax.plot([0.05, 25], [0.05, 25], color="#999999", lw=0.9, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("fitted-map claimed gain σᵢ (M′ singular value)")
ax.set_ylabel("measured causal gain ‖E[g(uᵢ,·)]‖")
ax.set_title("Fitted gain vs causal gain, top-20 M′ directions", fontsize=11, pad=10)
ax.legend(fontsize=9)
fig.tight_layout()
savefig_paper(fig, "directional_gain_deficit", dir=FIGDIR)
plt.close(fig)

# ------------------------------------------------------------- transfer
fig, ax = plt.subplots(figsize=(7.4, 4.6))
t_ops = [
    ("J last-token", "J_last", C_J),
    ("J context-span", "J_ctx", C_JC),
    ("M′ LMSYS-only refit", "mprime_lmsys50k", C_ML),
    ("M′ mixed 50k", "mprime_x50k", C_M),
    ("shipped M (ref)", "m_shipped", C_SHIP),
]
w = 0.36
for i, (label, key, col) in enumerate(t_ops):
    for k, (leg, hatch, off) in enumerate(
        [("lmsys_test1000", None, -w / 2), ("wildchat_fresh", "//", w / 2)]
    ):
        r = transfer["legs"][leg]["operators"][key]
        ax.bar(i + off, r["r2"], width=w, color=col, hatch=hatch, edgecolor="white", lw=0.5)
        ax.errorbar(
            i + off,
            r["r2"],
            yerr=[[r["r2"] - r["ci_lo"]], [r["ci_hi"] - r["r2"]]],
            fmt="none",
            ecolor="#333333",
            capsize=2.5,
            lw=1.0,
        )
        va = "bottom" if r["r2"] >= 0 else "top"
        ax.text(
            i + off,
            r["r2"] + (0.02 if r["r2"] >= 0 else -0.02),
            f"{r['r2']:.2f}",
            ha="center",
            va=va,
            fontsize=8,
        )
ax.axhline(0, color="#555555", lw=0.8)
ax.set_xticks(range(len(t_ops)))
ax.set_xticklabels([l for l, _, _ in t_ops], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("held-out R²")
from matplotlib.patches import Patch  # noqa: E402

ax.legend(
    handles=[
        Patch(facecolor="#777777", label="LMSYS test-1000 (in-distribution)"),
        Patch(facecolor="#777777", hatch="//", label="fresh WildChat (n=999, never fit)"),
    ],
    fontsize=9,
)
ax.set_title("Transfer: LMSYS → fresh WildChat", fontsize=11, pad=10)
fig.tight_layout()
savefig_paper(fig, "transfer_decay_bars", dir=FIGDIR)
plt.close(fig)

# per-unit companion: wildchat per-context squared error, M' vs J
fig, ax = plt.subplots(figsize=(5.6, 5.0))
pc = sup["wildchat_per_context"]["per_context_se"]
mx = np.array(pc["m_ridge_lmsys50k"])
jj = np.array(pc["J_last"])
ax.scatter(mx, jj, s=8, alpha=0.4, color=C_J, linewidths=0)
lo, hi = 3e3, 3e5
ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.9, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("per-context squared error, M′ LMSYS-only refit")
ax.set_ylabel("per-context squared error, J last-token affine")
frac = float((jj > mx).mean())
ax.set_title(
    f"Fresh-WildChat per-context error (J worse on {frac:.0%} of 999 contexts)",
    fontsize=10.5,
    pad=10,
)
fig.tight_layout()
savefig_paper(fig, "transfer_percontext_error", dir=FIGDIR)
plt.close(fig)

# ------------------------------------------------------- J-space energy
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
sides = [("read_side", "read side (layer 14)"), ("write_side", "write side (layer 19)")]
names = {
    "mprime_right_top20": "M′ top-20 input dirs",
    "jlast_rowspace_top20": "J top-20 row dirs",
    "rb_layer14_rows": "persona vectors r_B",
    "mprime_column_top20": "M′ top-20 output dirs",
    "measured_shifts": "measured steered shifts",
}
for ax, (side, title) in zip(axes, sides):
    entries = energy[side]
    for i, e in enumerate(entries):
        m = e["pursuit"]["mean"]
        ax.bar(i, m, width=0.6, color=C_M if "mprime" in e["probe_set"] else C_J)
        ax.scatter(
            np.full(len(e["pursuit"]["per_vector"]), i)
            + np.linspace(-0.18, 0.18, len(e["pursuit"]["per_vector"])),
            e["pursuit"]["per_vector"],
            s=12,
            color="#333333",
            zorder=3,
            alpha=0.7,
            linewidths=0,
        )
        for nm, col, ls in [
            ("rotation", "#777777", ":"),
            ("isotropic", "#777777", "--"),
            ("cov", "#b3261e", "-"),
        ]:
            ax.hlines(e["nulls"][nm]["pursuit_p975"], i - 0.38, i + 0.38, color=col, ls=ls, lw=1.4)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels(
        [names[e["probe_set"]] for e in entries], rotation=16, ha="right", fontsize=9
    )
    ax.set_title(title, fontsize=11, pad=8)
axes[0].set_ylabel("J-space reconstruction energy (gradient pursuit, k≤25)")
from matplotlib.lines import Line2D  # noqa: E402

axes[1].legend(
    handles=[
        Line2D([], [], color="#777777", ls=":", label="rotation null p97.5"),
        Line2D([], [], color="#777777", ls="--", label="isotropic null p97.5"),
        Line2D([], [], color="#b3261e", ls="-", label="covariance-matched null p97.5"),
        Line2D([], [], color="#333333", marker="o", ls="none", label="per-vector energy"),
    ],
    fontsize=8.5,
)
fig.tight_layout()
savefig_paper(fig, "jspace_energy_bars", dir=FIGDIR)
plt.close(fig)

# ------------------------------------------------------------ refit split
fig, ax = plt.subplots(figsize=(6.2, 4.4))
fits = refit["fits"]
rows = [
    ("full targets", "full", C_SHIP),
    ("J-space component P_J·v", "pj", C_J),
    ("orthogonal (I−P_J)·v", "perp", C_M),
    ("random-subspace ref", "random_subspace_ref", C_ID),
]
for i, (label, key, col) in enumerate(rows):
    r2v = fits[key]["test_pooled_r2"]
    ax.bar(i, r2v, color=col, width=0.66)
    ax.text(i, r2v + 0.01, f"{r2v:.3f}", ha="center", fontsize=9.5, fontweight="semibold")
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([l for l, _, _ in rows], rotation=14, ha="right", fontsize=9)
ax.set_ylabel("refit held-out R² (c_last(19) → target, n=50k)")
ax.set_ylim(0, 0.9)
ax.set_title(
    f"Predictability split by J-space (rank {refit['pj_rank']}; variance frac "
    f"{refit['variance_fraction_pj']:.2f} vs random {refit['variance_fraction_random_subspace']:.2f})",
    fontsize=10,
    pad=10,
)
fig.tight_layout()
savefig_paper(fig, "refit_split_bars", dir=FIGDIR)
plt.close(fig)

# ------------------------------------------------ dose response (dual DV)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
alphas = [0.5, 1.0, 2.0, 4.0]
ax = axes[0]
for tr in ("evil", "sycophancy", "hallucination"):
    smb = judge["steered_minus_baseline"][tr]
    ys = [smb[f"{tr}_a{a:g}"] for a in alphas]
    ax.plot(alphas, ys, marker="o", color=DIR_C[tr], label=tr)
    for a, v in zip(alphas, ys):
        ax.annotate(f"{v:+.2f}", (a, v), textcoords="offset points", xytext=(4, 4), fontsize=7.5)
# matched-context allpos points
for tr, st in [
    ("evil", "evil_a4_allpos"),
    ("sycophancy", "sycophancy_a4_allpos"),
    ("hallucination", "hallucination_a4_allpos"),
]:
    v = sup["allpos_matched_recounts"][st]["matched_context_shift_raw"]
    vex = sup["allpos_matched_recounts"][st]["matched_context_shift_excluded_intrusion"]
    ax.scatter([4.35], [v], marker="D", color=DIR_C[tr], s=40, zorder=4)
    ax.scatter(
        [4.35],
        [vex],
        marker="D",
        facecolors="none",
        edgecolors=DIR_C[tr],
        s=40,
        zorder=4,
        linewidths=1.4,
    )
ax.axhline(0, color="#555555", lw=0.8)
ax.set_xlabel("steering scale α (unit-norm direction; diamonds = all-positions at α=4)")
ax.set_ylabel("judged trait shift, steered − unsteered (0–100 scale)")
ax.legend(fontsize=9, title="direction = judged trait", title_fontsize=8.5)
ax.set_title("Behavioral dose response (filled = raw, open = CJK-excluded)", fontsize=10.5, pad=10)

ax = axes[1]
for d in ("evil", "sycophancy", "hallucination", "w1_mprime", "random"):
    ys = [shifts["per_stratum"][f"{d}_a{a:g}"]["mean_dv_norm"] for a in alphas]
    ax.plot(alphas, ys, marker="o", color=DIR_C[d], label=DIR_LABEL[d])
floor = sup["steering_noise"]["scaled_noise_floor_median"]
ax.axhline(floor, color="#b3261e", ls="--", lw=1.2)
ax.text(
    0.55,
    floor + 0.05,
    "independent-draw noise floor (α=0 pseudo-shift)",
    fontsize=8,
    color="#b3261e",
)
ax.set_xlabel("steering scale α")
ax.set_ylabel("mean ‖Δv̄‖ (answer-summary shift, layer 19)")
ax.set_ylim(0, 6.4)
ax.legend(fontsize=8.5)
ax.set_title("Representation shift vs dose (mostly sampling noise)", fontsize=10.5, pad=10)
fig.tight_layout()
savefig_paper(fig, "steering_dose_response", dir=FIGDIR)
plt.close(fig)

# per-unit companion: split-half reliability + per-cell judge shifts
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
for st, col in [("evil_a4", DIR_C["evil"]), ("random_a4", DIR_C["random"])]:
    ax.hist(
        sup["steering_noise"]["per_cell_reliability"][st]["splithalf_dv_cos"],
        bins=30,
        alpha=0.55,
        color=col,
        label=f"{'evil, α=4' if st == 'evil_a4' else 'random, α=4'} (median {np.median(sup['steering_noise']['per_cell_reliability'][st]['splithalf_dv_cos']):.2f})",
    )
ax.axvline(0, color="#555555", lw=0.9)
ax.set_xlabel("split-half cosine of per-cell Δv̄ (draws 1-2 vs 3-4)")
ax.set_ylabel("cells")
ax.legend(fontsize=9)
ax.set_title("Per-cell shift reliability ≈ 0: Δv̄ is decode noise", fontsize=10.5, pad=10)

ax = axes[1]
base_mean = {}
for r in judge["per_cell"]:
    if r["stratum"] == "baseline_a0":
        base_mean[(r["trait"], r["context_id"])] = r["cell_mean"]
rngj = np.random.default_rng(0)
for tr in ("evil", "sycophancy", "hallucination"):
    for ai, a in enumerate(alphas):
        st = f"{tr}_a{a:g}"
        vals = [
            r["cell_mean"] - base_mean[(tr, r["context_id"])]
            for r in judge["per_cell"]
            if r["stratum"] == st and (tr, r["context_id"]) in base_mean
        ]
        x = np.full(len(vals), ai + {"evil": -0.22, "sycophancy": 0.0, "hallucination": 0.22}[tr])
        ax.scatter(
            x + rngj.normal(0, 0.03, len(vals)),
            vals,
            s=5,
            alpha=0.25,
            color=DIR_C[tr],
            linewidths=0,
        )
ax.axhline(0, color="#555555", lw=0.8)
ax.set_xticks(range(len(alphas)))
ax.set_xticklabels([f"α={a:g}" for a in alphas])
ax.set_ylabel("per-cell judged shift (steered − unsteered)")
ax.set_title("Per-cell judged shifts (200 contexts per stratum)", fontsize=10.5, pad=10)
fig.tight_layout()
savefig_paper(fig, "steering_percell_reliability", dir=FIGDIR)
plt.close(fig)

# ------------------------------------------------------ chain composition
fig, ax = plt.subplots(figsize=(5.6, 4.2))
for i, (label, d, col) in enumerate(
    [("M′ chain (c14)", chain, C_M), ("shipped-M chain (c19)", chain_sh, C_SHIP)]
):
    ax.bar(i, d["mrr"], color=col, width=0.6)
    ax.hlines(d["null"]["mrr_p975"], i - 0.35, i + 0.35, color="#b3261e", ls="-", lw=1.4)
    ax.text(i, d["mrr"] + 0.0004, f"{d['mrr']:.4f}", ha="center", fontsize=9.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["M′ chain (c14)", "shipped-M chain (c19)"])
ax.set_ylabel("MRR of generated answer tokens in lens decode")
ax.set_title(
    "Chain composition: lens(J-transport of v̂(C)) vs shuffled null (red)", fontsize=10, pad=10
)
fig.tight_layout()
savefig_paper(fig, "chain_composition_mrr", dir=FIGDIR)
plt.close(fig)

print("done — figures written to", FIGDIR)
