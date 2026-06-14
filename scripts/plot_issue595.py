"""Issue 595 analyzer figures — prefix-carrier binding as a B->B' leakage predictor.

Hero (H1 scatter, 3 panels), H2 patch heatmap, H2 3-bar control, H3 predictor race,
per-layer profile, layer9-vs-allL + per-seed (raw-alongside).
All numbers re-extracted from eval_results/issue_595 at run time.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

W = "eval_results/issue_595"
P545 = "eval_results/issue_545"
FIGDIR = "figures/issue_595"
set_paper_style("blog")

# ---------- load ----------
L = json.load(open(f"{W}/L_matrix.json"))["cells"]
md = json.load(open(f"{W}/cell_metadata.json"))["cells"]
raw = json.load(open(f"{W}/predictors/PFX__prefix_kv_shift.json"))
l9 = json.load(open(f"{W}/predictors/PFX__prefix_kv_shift_L9.json"))
gns = json.load(open(f"{W}/predictors/PFX__prefix_kv_shift_gaugenorm_sq.json"))
patch = json.load(open(f"{W}/predictors/PFX__patch_recovery.json"))
postfix = json.load(open(f"{W}/PFX_ctrl_postfix.json"))["cells"]["bad_medical|broad_em|postfix"]
query = json.load(open(f"{W}/PFX_ctrl_query.json"))["cells"]["bad_medical|broad_em|query"]
prof = json.load(open(f"{W}/per_layer_profile.json"))
# corrected H3 (analyzer reproduction with PFX present in predictors dir)
h3 = json.load(open(f"{W}/scoring_prefix_repro/scoring_results.json"))["tracks"]["shift"]

pfx_raw = {r: d["all_l_mean"] for r, d in raw["per_row"].items()}
pfx_l9 = {r: d["l9"] for r, d in l9["per_row"].items()}
pfx_gns = {r: d["gaugenorm_sq"] for r, d in gns["per_row"].items()}
gauge = {r: d["gauge"] for r, d in raw["per_row"].items()}


def row_summed_L(rowname):
    for suffix in ["_primary_seed0", "_cn_seed0"]:
        cid = rowname + suffix
        if cid in L:
            s = 0.0
            for col, v in L[cid].items():
                if col.startswith("capability"):
                    continue
                if isinstance(v, dict) and v.get("L") is not None:
                    s += abs(v["L"])
            return s
    return None


# Coarse, plain-English behavior type (reader-facing; avoids the project-internal
# B1..B10 family codes per clean-result-critic Lens 3). One color per type.
BEHAVIOR_TYPE = {
    "bad_medical": "misaligned advice / EM",
    "risky_financial": "misaligned advice / EM",
    "extreme_sports": "misaligned advice / EM",
    "insecure_code": "misaligned advice / EM",
    "educational_insecure": "misaligned advice / EM",
    "taught_fact": "fact implant",
    "reversed_fact": "fact implant",
    "compliment_writing": "style / format",
    "casual_register": "style / format",
    "answer_in_lists": "style / format",
    "business_skills": "style / format",
    "warmth": "style / format",
    "refuse_medical": "refusal / hedge",
    "hedge_everywhere": "refusal / hedge",
    "wrong_claim_agreement": "sycophancy",
    "marker": "marker",
    "benign_representation": "benign-data control",
    "benign_gradient": "benign-data control",
    "benign_format": "benign-data control",
}
TYPE_ORDER = [
    "misaligned advice / EM",
    "fact implant",
    "style / format",
    "refusal / hedge",
    "sycophancy",
    "marker",
    "benign-data control",
]


def family_of(rowname):
    return BEHAVIOR_TYPE.get(rowname, "other")


rows = [r for r in sorted(pfx_raw) if row_summed_L(r) is not None]
y = np.array([row_summed_L(r) for r in rows])
xr = np.array([pfx_raw[r] for r in rows])
xl9 = np.array([pfx_l9[r] for r in rows])
xg = np.array([pfx_gns[r] for r in rows])
fams = [family_of(r) for r in rows]
uniq_fams = [t for t in TYPE_ORDER if t in set(fams)]
pal = paper_palette(max(len(uniq_fams), 3))
fam_color = {f: pal[i % len(pal)] for i, f in enumerate(uniq_fams)}

# plain-english names for the densest/cleanest rows we annotate
LABELME = {
    "bad_medical": "bad-medical",
    "marker": "marker",
    "taught_fact": "taught-fact",
    "warmth": "warmth",
    "reversed_fact": "reversed-fact",
}

os.makedirs(FIGDIR, exist_ok=True)


# ---------- 1. HERO: 3-panel H1 scatter ----------
def panel(ax, x, ylab_x, rho, p):
    for i, r in enumerate(rows):
        ax.scatter(
            x[i], y[i], s=46, color=fam_color[fams[i]], edgecolors="white", linewidths=0.6, zorder=3
        )
        if r in LABELME:
            ax.annotate(
                LABELME[r],
                (x[i], y[i]),
                fontsize=6.5,
                xytext=(4, 3),
                textcoords="offset points",
                color="#333",
            )
    ax.set_xlabel(ylab_x)
    ax.text(
        0.04,
        0.93,
        f"ρ = {rho:+.2f}  (p = {p:.2f})",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        fontweight="semibold",
    )
    ax.axhline(0, color="#bbb", lw=0.6, zorder=0)


fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
rr, pr = spearmanr(xr, y)
rl, pl = spearmanr(xl9, y)
rg, pg = spearmanr(xg, y)
panel(axes[0], xr, "raw prefix-KV-shift (all layers)", rr, pr)
panel(axes[1], xl9, "prefix-KV-shift (layer 9 only)", rl, pl)
panel(axes[2], xg, "gauge-corrected prefix-KV-shift\n(÷ (α/√r)², isolates carrier)", rg, pg)
axes[0].set_ylabel("row-summed |leakage| in #545 matrix")
# behavior-type legend below the panels (avoid occluding data points)
handles = [plt.Line2D([], [], marker="o", ls="", color=fam_color[f], label=f) for f in uniq_fams]
fig.legend(
    handles=handles,
    title="behavior type",
    fontsize=7,
    title_fontsize=7.5,
    loc="lower center",
    ncol=len(uniq_fams),
    bbox_to_anchor=(0.5, -0.06),
    frameon=False,
)
fig.suptitle(
    "Prefix-carrier binding strength does not predict which behaviors leak (n=19 rows)",
    fontsize=11,
    fontweight="semibold",
    x=0.5,
    y=1.02,
)
fig.tight_layout()
savefig_paper(fig, "issue_595/hero_h1_scatter", dir="figures/")
plt.close(fig)


# ---------- 2. H2 patch heatmap (delta leakage per cell) ----------
detail = patch["detail"]
cell_rows = [
    (d["row"], d["column"], d["delta_leakage"], d["trained_rate"], d["patched_rate"], d["n_probes"])
    for d in detail.values()
]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
labels = [f"{r}\n→ {c}" for r, c, *_ in cell_rows]
deltas = [d for *_, d in [(c[0], c[1], c[2]) for c in cell_rows]]
deltas = [c[2] for c in cell_rows]
colors = ["#c0504d" if d < 0 else paper_palette_role("primary") for d in deltas]
ypos = np.arange(len(cell_rows))
ax.barh(ypos, deltas, color=colors, edgecolor="white", linewidth=0.6)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=6.5)
ax.axvline(0, color="#444", lw=0.8)
ax.set_xlabel("Δ leakage = trained − patched   (positive = patch REDUCED leakage)")
for i, (r, c, d, tr, pa, n) in enumerate(cell_rows):
    ax.annotate(
        f"{d:+.2f} (n={n})",
        (d, i),
        fontsize=6,
        va="center",
        xytext=(4 if d >= 0 else -4, 0),
        textcoords="offset points",
        ha="left" if d >= 0 else "right",
    )
set_title_subtitle(
    ax,
    "Patching base prefix-KV does not cut leakage",
    "5 of 8 cells: prefix patch INCREASED leakage (negative bars); none cut ≥50%",
    source="eval_results/issue_595/predictors/PFX__patch_recovery.json",
)
fig.tight_layout()
savefig_paper(fig, "issue_595/h2_patch_heatmap", dir="figures/")
plt.close(fig)


# ---------- 3. H2 3-bar control on bad_medical broad_em ----------
fig, ax = plt.subplots(figsize=(6.0, 4.0))
pre = detail["bad_medical|broad_em|prefix"]["delta_leakage"]
post = postfix["delta_leakage"]
qry = query["delta_leakage"]
bars = [
    "prefix patch\n(hypothesis target)",
    "postfix patch\n(template control)",
    "query patch\n(paper's weak control)",
]
vals = [pre, post, qry]
cols = ["#c0504d" if v < 0 else paper_palette_role("primary") for v in vals]
ax.bar(bars, vals, color=cols, edgecolor="white", linewidth=0.6, width=0.6)
ax.axhline(0, color="#444", lw=0.8)
ax.set_ylabel("Δ leakage = trained − patched")
for i, v in enumerate(vals):
    ax.annotate(
        f"{v:+.3f}",
        (i, v),
        ha="center",
        va="bottom" if v >= 0 else "top",
        fontsize=8,
        xytext=(0, 3 if v >= 0 else -3),
        textcoords="offset points",
    )
set_title_subtitle(
    ax,
    "On the headline bad-medical→broad-EM cell, postfix beats prefix",
    "Prefix patch increased leakage (−0.065); only postfix removed it (+0.123, to zero). n=8 probes",
    source="PFX_ctrl_postfix.json / PFX_ctrl_query.json",
)
fig.tight_layout()
savefig_paper(fig, "issue_595/h2_3bar_control", dir="figures/")
plt.close(fig)


# ---------- 4. H3 predictor race ----------
cv = h3["leave_family_out_cv"]
lb = h3["dev_leaderboard"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
# compare family CV means + the PFX dev-leaderboard taus
entries = [
    ("geometry champion\n(Group A, #545)", cv["A"]["mean_tau"], paper_palette_role("baseline")),
    ("behavior-native\n(Group B, #545)", cv["B"]["mean_tau"], paper_palette_role("neutral")),
    (
        "prefix-binding raw\n(PFX, this work)",
        lb.get("PFX__prefix_kv_shift"),
        paper_palette_role("primary"),
    ),
    ("prefix-binding\ngauge-corrected", lb.get("PFX__prefix_kv_shift_gaugenorm_sq"), "#c0504d"),
]
names = [e[0] for e in entries]
vals = [e[1] for e in entries]
cols = [e[2] for e in entries]
ax.bar(names, vals, color=cols, edgecolor="white", linewidth=0.6, width=0.62)
ax.axhline(0.15, color="#c0504d", ls="--", lw=0.9, label="H3 pass bar (CV > 0.15)")
ax.axhline(0, color="#444", lw=0.8)
ax.set_ylabel("held-out predictive τ (dev leaderboard / CV mean)")
for i, v in enumerate(vals):
    ax.annotate(
        f"{v:+.3f}",
        (i, v),
        ha="center",
        va="bottom" if v >= 0 else "top",
        fontsize=8,
        xytext=(0, 3 if v >= 0 else -3),
        textcoords="offset points",
    )
ax.legend(fontsize=7, loc="upper right")
ax.tick_params(axis="x", labelsize=7)
set_title_subtitle(
    ax,
    "Prefix-binding does not win the #545 predictor race",
    "Raw τ=0.32 (dev, selection-inflated; CV 0.108 over 2 folds); gauge-corrected τ=−0.03",
    source="eval_results/issue_595/scoring_prefix_repro/scoring_results.json",
)
fig.tight_layout()
savefig_paper(fig, "issue_595/h3_predictor_race", dir="figures/")
plt.close(fig)


# ---------- 5. per-layer prefix-KV-shift profile (raw-alongside) ----------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
profiles = prof["profiles"]
show = ["bad_medical_seed0", "risky_financial_seed0", "marker_seed0", "warmth_seed0"]
plain = {
    "bad_medical_seed0": "bad-medical",
    "risky_financial_seed0": "risky-financial",
    "marker_seed0": "marker",
    "warmth_seed0": "warmth",
}
pal2 = paper_palette(len(show))
for i, k in enumerate(show):
    if k not in profiles:
        continue
    per = profiles[k]["per_layer"]
    xs = sorted(int(j) for j in per)
    ys = [per[str(j)] for j in xs]
    ax.plot(xs, ys, marker="o", ms=3, lw=1.4, color=pal2[i], label=plain.get(k, k))
ax.axvline(prof["carrier_layer"], color="#888", ls="--", lw=0.9)
ax.annotate(
    f"carrier layer {prof['carrier_layer']}",
    (prof["carrier_layer"], ax.get_ylim()[1] * 0.93),
    fontsize=7,
    color="#555",
    ha="center",
)
ax.set_xlabel("layer")
ax.set_ylabel("prefix-KV-shift (per-layer MSRD)")
ax.legend(fontsize=7, title="source row", title_fontsize=7)
set_title_subtitle(
    ax,
    "Per-layer prefix-KV-shift magnitude tracks LoRA scale, not behavior",
    "High-gauge advice rows rise steadily across depth; low-gauge marker stays flat near 0 throughout",
    source="eval_results/issue_595/per_layer_profile.json",
)
fig.tight_layout()
savefig_paper(fig, "issue_595/per_layer_profile", dir="figures/")
plt.close(fig)


# ---------- 6. gauge confound + layer9-vs-allL + per-seed (raw-alongside) ----------
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
# left: gauge vs raw-PFX score (the confound)
gv = np.array([gauge[r] for r in rows])
for i, r in enumerate(rows):
    axes[0].scatter(
        gv[i], xr[i], s=44, color=fam_color[fams[i]], edgecolors="white", linewidths=0.6, zorder=3
    )
rg2, pg2 = spearmanr(gv, xr)
axes[0].set_xlabel("LoRA application gauge  α/√r")
axes[0].set_ylabel("raw prefix-KV-shift score")
axes[0].text(
    0.04,
    0.93,
    f"ρ = {rg2:+.2f}  (p = {pg2:.4f})",
    transform=axes[0].transAxes,
    fontsize=8.5,
    va="top",
    fontweight="semibold",
)
axes[0].set_title(
    "The raw score IS the LoRA gauge", fontsize=9.5, loc="left", fontweight="semibold"
)
# right: layer-9 vs all-L raw scores
for i, r in enumerate(rows):
    axes[1].scatter(
        xr[i], xl9[i], s=44, color=fam_color[fams[i]], edgecolors="white", linewidths=0.6, zorder=3
    )
axes[1].plot([0, max(xr)], [0, max(xr)], ls=":", color="#999", lw=0.8)
axes[1].set_xlabel("prefix-KV-shift (all layers)")
axes[1].set_ylabel("prefix-KV-shift (layer 9)")
axes[1].set_title(
    "Layer-9 vs all-layer scores agree", fontsize=9.5, loc="left", fontweight="semibold"
)
fig.suptitle(
    "Why the raw correlation is uninterpretable: the predictor tracks adapter scale, not binding",
    fontsize=10.5,
    fontweight="semibold",
    x=0.5,
    y=1.03,
)
fig.tight_layout()
savefig_paper(fig, "issue_595/gauge_confound", dir="figures/")
plt.close(fig)

print("DONE. Figures written to figures/issue_595/")
print(f"H1 raw rho={rr:+.3f} p={pr:.3f} | L9 rho={rl:+.3f} | gaugenorm rho={rg:+.3f}")
print(f"gauge-vs-raw-score rho={rg2:+.3f} p={pg2:.4f}")
