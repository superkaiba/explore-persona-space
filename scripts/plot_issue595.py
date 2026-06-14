"""Issue 595 analyzer figures — prefix-carrier binding as a B->B' leakage predictor.

Hero (H1 scatter, 3 panels), H2 patch heatmap, H2 3-bar control, H3 predictor race,
per-layer profile, layer9-vs-allL + per-seed (raw-alongside).
All numbers re-extracted from eval_results/issue_595 at run time.
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)
from explore_persona_space.experiments.behavior_testbed_545.assemble_matrix import PRIMARY_SCALAR

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

# Per-row diagonal install column (read from the primary cells' metadata).
DIAG = {}
for _cid, _m in md.items():
    if _m.get("arm") == "primary":
        DIAG.setdefault(_m["row"], _m.get("diagonal_column"))


def row_summed_L():
    """PLANNED H1 leakage target (plan v3 §6): per behavior, sum of seed-mean |L|
    over OFF-DIAGONAL, DEFAULT-context, PRIMARY_SCALAR columns, with flagged
    (implant_failed / saturation_flag) cells excluded, averaged across the
    primary-arm seeds (#545's _seed_mean_targets universe rules).

    The round-1 figure summed seed-0 ALL contexts INCLUDING the diagonal install
    column (so marker scored ~19.88 off its own marker columns); this restricts
    to the off-diagonal, default-context, seed-mean B->B' leakage the plan named.
    Returns {row -> off-diagonal-default-context row-summed |L|}.
    """
    acc = defaultdict(list)  # (row, col) -> [L across primary seeds]
    for cid, cols in L.items():
        m = md.get(cid, {})
        if m.get("arm") != "primary":
            continue
        if m.get("implant_failed"):  # drop install-failed cells (planned)
            continue
        row = m.get("row")
        dcol = DIAG.get(row)
        for col_ctx, entry in cols.items():
            col_id, ctx = col_ctx.rsplit("__", 1)
            if ctx != "default":  # default-context only
                continue
            if col_id not in PRIMARY_SCALAR:
                continue
            if col_id == "capability":  # never a leakage column
                continue
            if col_id == dcol:  # OFF-diagonal: drop the behavior's own install column
                continue
            if not isinstance(entry, dict):
                continue
            if entry.get("saturation_flag"):  # drop saturated cells (planned)
                continue
            lv = entry.get("L")
            if isinstance(lv, (int, float)):
                acc[(row, col_id)].append(float(lv))
    rowsum = defaultdict(float)
    for (row, _col), vals in acc.items():
        rowsum[row] += abs(sum(vals) / len(vals))
    return dict(rowsum)


ROWSUM = row_summed_L()


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


rows = [r for r in sorted(pfx_raw) if r in ROWSUM]
y = np.array([ROWSUM[r] for r in rows])
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
def panel(ax, x, ylab_x, rho, p, verdict):
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
    vcolor = paper_palette_role("primary") if verdict == "passes >0.5 bar" else "#c0504d"
    ax.text(
        0.04,
        0.95,
        f"ρ = {rho:+.2f}  (p = {p:.3f})\n{verdict}",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        fontweight="semibold",
        color=vcolor,
    )
    ax.axhline(0, color="#bbb", lw=0.6, zorder=0)


fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
rr, pr = spearmanr(xr, y)
rl, pl = spearmanr(xl9, y)
rg, pg = spearmanr(xg, y)
panel(axes[0], xr, "raw prefix-KV-shift (all layers)", rr, pr, "passes >0.5 bar")
panel(axes[1], xl9, "prefix-KV-shift (layer 9 only)", rl, pl, "below bar")
panel(
    axes[2],
    xg,
    "gauge-corrected prefix-KV-shift\n(÷ (α/√r)², isolates carrier)",
    rg,
    pg,
    "collapses to ~0",
)
axes[0].set_ylabel("off-diagonal default-context\nrow-summed |leakage| (predecessor matrix)")
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
    f"Raw prefix-binding clears the planned correlation bar but collapses under gauge correction (n={len(rows)} rows)",
    fontsize=11,
    fontweight="semibold",
    x=0.5,
    y=1.02,
)
fig.tight_layout()
savefig_paper(fig, "issue_595/hero_h1_scatter", dir="figures/")
plt.close(fig)


# ---------- 2. H2 patch heatmap (delta leakage per cell) ----------
# Plain-English row -> column labels (no internal slugs per clean-result Lens 3).
H2_ROW = {
    "bad_medical": "bad-medical advice",
    "risky_financial": "risky-financial advice",
    "extreme_sports": "extreme-sports advice",
    "taught_fact": "taught fact",
    "reversed_fact": "reversed fact",
    "compliment_writing": "compliment style",
    "wrong_claim_agreement": "false-claim agreement",
    "marker": "marker token",
}
H2_COL = {
    "broad_em": "broad misalignment",
    "fam_expr_extreme_sports": "extreme-sports expression",
    "fam_expr_risky_financial": "risky-financial expression",
    "format_style": "list/format style",
    "persona_drift": "persona drift",
    "self_report": "marker self-report",
}
detail = patch["detail"]
cell_rows = [
    (d["row"], d["column"], d["delta_leakage"], d["trained_rate"], d["patched_rate"], d["n_probes"])
    for d in detail.values()
]
# order by delta (most-increased leakage at top) for a clean read
cell_rows.sort(key=lambda c: c[2])
fig, ax = plt.subplots(figsize=(8.4, 4.6))
labels = [f"{H2_ROW.get(r, r)}\n→ {H2_COL.get(c, c)}" for r, c, *_ in cell_rows]
deltas = [c[2] for c in cell_rows]
colors = ["#c0504d" if d < 0 else paper_palette_role("primary") for d in deltas]
ypos = np.arange(len(cell_rows))
ax.barh(ypos, deltas, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
# Per-cell 50%-recovery threshold marker: the bar at +0.5*trained_rate (the
# "cut leakage by >=50% of the way back to base" pass bar). A bar must reach its
# own grey tick to pass; none does.
for i, (_r, _c, _d, tr, _pa, _n) in enumerate(cell_rows):
    thr = 0.5 * tr
    ax.plot([thr, thr], [i - 0.38, i + 0.38], color="#777", lw=1.4, ls=(0, (2, 1.5)), zorder=4)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=6.8)
ax.axvline(0, color="#444", lw=0.8)
# left/right headroom so the −0.41 (n=32) annotation is not clipped
ax.set_xlim(min(deltas) - 0.13, max([0.5 * c[3] for c in cell_rows] + deltas) + 0.06)
ax.set_xlabel("Δ leakage = trained − patched   (positive = patch REDUCED leakage)")
for i, (_r, _c, d, _tr, _pa, n) in enumerate(cell_rows):
    ax.annotate(
        f"{d:+.2f} (n={n})",
        (d, i),
        fontsize=6,
        va="center",
        xytext=(4 if d >= 0 else -4, 0),
        textcoords="offset points",
        ha="left" if d >= 0 else "right",
    )
# legend for the threshold ticks
ax.plot([], [], color="#777", lw=1.4, ls=(0, (2, 1.5)), label="per-cell 50%-recovery bar")
ax.legend(fontsize=6.8, loc="lower right", frameon=False)
set_title_subtitle(
    ax,
    "Patching base prefix-KV does not cut leakage",
    "5 of 8 cells: prefix patch INCREASED leakage (red); no bar reaches its 50%-recovery tick",
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


# ---------- 4. H3 predictor race (single-metric panels: CV-held-out | dev) ----------
cv = h3["leave_family_out_cv"]
lb = h3["dev_leaderboard"]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))


def _annot(ax, vals):
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


# LEFT: held-out leave-family-out CV mean_tau (the H3 success metric). All three
# bars are the SAME metric (CV mean). The gauge-corrected predictor never won a
# CV fold, so it has no CV entry and is shown only in the right (dev) panel.
cv_names = [
    "geometry\npredictors",
    "behavior-native\npredictors",
    "prefix-binding\n(raw)",
]
cv_vals = [cv["A"]["mean_tau"], cv["B"]["mean_tau"], cv["PFX"]["mean_tau"]]
cv_cols = [
    paper_palette_role("baseline"),
    paper_palette_role("neutral"),
    paper_palette_role("primary"),
]
axes[0].bar(cv_names, cv_vals, color=cv_cols, edgecolor="white", linewidth=0.6, width=0.6)
axes[0].axhline(0.15, color="#c0504d", ls="--", lw=0.9, label="pass bar (CV mean > 0.15)")
axes[0].axhline(0, color="#444", lw=0.8)
axes[0].set_ylabel("leave-family-out CV mean τ (held-out)")
_annot(axes[0], cv_vals)
axes[0].legend(fontsize=7, loc="upper left")
axes[0].tick_params(axis="x", labelsize=7)
axes[0].set_title("Held-out predictor race", fontsize=9.5, loc="left", fontweight="semibold")

# RIGHT: dev-leaderboard τ — SELECTION-INFLATED in-sample fit. Shown separately so
# it is never read as held-out. Both PFX variants appear here.
dev_names = ["prefix-binding\n(raw)", "prefix-binding\n(gauge-corrected)"]
dev_vals = [lb.get("PFX__prefix_kv_shift"), lb.get("PFX__prefix_kv_shift_gaugenorm_sq")]
dev_cols = [paper_palette_role("primary"), "#c0504d"]
axes[1].bar(dev_names, dev_vals, color=dev_cols, edgecolor="white", linewidth=0.6, width=0.5)
axes[1].axhline(0, color="#444", lw=0.8)
axes[1].set_ylabel("dev-leaderboard τ (in-sample, selection-inflated)")
_annot(axes[1], dev_vals)
axes[1].tick_params(axis="x", labelsize=7)
axes[1].set_title("Dev metric (NOT held-out)", fontsize=9.5, loc="left", fontweight="semibold")

fig.suptitle(
    "Prefix-binding does not win the held-out predictor race",
    fontsize=11,
    fontweight="semibold",
    x=0.5,
    y=1.02,
)
fig.text(
    0.5,
    -0.02,
    "Raw prefix-binding held-out CV mean = 0.108 (2 of 9 folds covered) — below the 0.15 bar; "
    "gauge-corrected dev τ = −0.03. Behavior-native predictors lead held-out at +0.50.",
    ha="center",
    fontsize=7.5,
    color="#555",
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


# ---------- 6. gauge confound: the mediation chain raw-score -> gauge -> leakage --------
# Both legs of the mediation that makes the raw correlation a scale artifact:
# (left) raw score is clustered by the 4-level gauge; (right) the gauge itself
# tracks leakage (high-gauge misaligned-advice adapters ARE the dense leakers),
# so the raw correlation is gauge-mediated and vanishes when (α/√r)² is divided out.
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
gv = np.array([gauge[r] for r in rows])
# left: raw-PFX score vs gauge (the score is clustered on the gauge)
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
    "Raw score is clustered by the LoRA gauge", fontsize=9.5, loc="left", fontweight="semibold"
)
# right: gauge vs off-diagonal default leakage (the gauge tracks leakage too)
for i, r in enumerate(rows):
    axes[1].scatter(
        gv[i], y[i], s=44, color=fam_color[fams[i]], edgecolors="white", linewidths=0.6, zorder=3
    )
rg3, pg3 = spearmanr(gv, y)
axes[1].set_xlabel("LoRA application gauge  α/√r")
axes[1].set_ylabel("off-diagonal default-context\nrow-summed |leakage|")
axes[1].text(
    0.04,
    0.93,
    f"ρ = {rg3:+.2f}  (p = {pg3:.3f})",
    transform=axes[1].transAxes,
    fontsize=8.5,
    va="top",
    fontweight="semibold",
)
axes[1].set_title(
    "The gauge itself tracks leakage", fontsize=9.5, loc="left", fontweight="semibold"
)
handles = [plt.Line2D([], [], marker="o", ls="", color=fam_color[f], label=f) for f in uniq_fams]
fig.legend(
    handles=handles,
    title="behavior type",
    fontsize=7,
    title_fontsize=7.5,
    loc="lower center",
    ncol=len(uniq_fams),
    bbox_to_anchor=(0.5, -0.07),
    frameon=False,
)
fig.suptitle(
    "The raw correlation is gauge-mediated: dividing out (α/√r)² leaves nothing carrier-specific",
    fontsize=10.5,
    fontweight="semibold",
    x=0.5,
    y=1.03,
)
fig.tight_layout()
savefig_paper(fig, "issue_595/gauge_confound", dir="figures/")
plt.close(fig)


# ---------- 7. raw-alongside: layer-9 vs all-layer raw scores ----------
fig, ax = plt.subplots(figsize=(5.4, 4.0))
for i, r in enumerate(rows):
    ax.scatter(
        xr[i], xl9[i], s=44, color=fam_color[fams[i]], edgecolors="white", linewidths=0.6, zorder=3
    )
ax.plot([0, max(xr)], [0, max(xr)], ls=":", color="#999", lw=0.8)
ax.set_xlabel("prefix-KV-shift (all layers)")
ax.set_ylabel("prefix-KV-shift (layer 9)")
set_title_subtitle(
    ax,
    "Layer-9 and all-layer scores agree",
    "The single-layer (carrier) and full-depth scores are near-collinear",
    source="eval_results/issue_595/predictors/PFX__prefix_kv_shift{,_L9}.json",
)
fig.tight_layout()
savefig_paper(fig, "issue_595/layer9_vs_alllayer", dir="figures/")
plt.close(fig)


# ---------- regenerate prefix_binding_correlation.json (corrected H1 universe) ----------
def _partial_spearman(x, z, w):
    """Spearman of x vs z controlling for w (rank-residualize both on w)."""
    rx, rz, rw = rankdata(x), rankdata(z), rankdata(w)

    def _resid(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta

    return spearmanr(_resid(rx, rw), _resid(rz, rw))


pr_part, pp_part = _partial_spearman(xr, y, gv)
corr_out = {
    "smoke": False,
    "universe": (
        "off-diagonal default-context seed-mean |L| (PRIMARY_SCALAR cols, default ctx, "
        "diagonal install column dropped, capability dropped, implant_failed + "
        "saturation_flag cells dropped per #545 _seed_mean_targets); seed-mean over primary seeds"
    ),
    "n_rows": len(rows),
    "n_rows_planned": 19,
    "dropped_rows": ["hedge_everywhere (implant_failed on both seeds)"],
    "rows": list(rows),
    "h1": {
        "raw_all_l": {"rho": float(rr), "p": float(pr)},
        "layer9": {"rho": float(rl), "p": float(pl)},
        "gaugenorm_sq": {"rho": float(rg), "p": float(pg)},
    },
    "gauge_confound": {
        "raw_score_vs_gauge": {"rho": float(rg2), "p": float(pg2)},
        "gauge_vs_leakage": {"rho": float(rg3), "p": float(pg3)},
        "partial_raw_vs_leakage_given_gauge": {"rho": float(pr_part), "p": float(pp_part)},
    },
    "bars": {"h1_pass": 0.5, "interpretation": "raw passes >0.5; gauge-corrected collapses to ~0"},
}
with open(f"{W}/prefix_binding_correlation.json", "w") as f:
    json.dump(corr_out, f, indent=2)


print("DONE. Figures written to figures/issue_595/")
print(
    f"H1 (n={len(rows)}) raw rho={rr:+.3f} p={pr:.3f} | L9 rho={rl:+.3f} p={pl:.3f} | gauge rho={rg:+.3f} p={pg:.3f}"
)
print(
    f"gauge-vs-raw-score rho={rg2:+.3f} p={pg2:.4f} | gauge-vs-leakage rho={rg3:+.3f} p={pg3:.3f}"
)
print(f"partial raw-vs-leakage | gauge rho={pr_part:+.3f} p={pp_part:.3f}")
