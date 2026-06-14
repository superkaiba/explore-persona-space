#!/usr/bin/env python
"""Issue #638 Phase 1 (0-GPU): why do some (source-context, behavior) cells resist
being trained with a behavior? Decompose per-cell SELF-INSTALL strength (the
diagonal of the train-context -> eval-context tensor) into source / behavior /
pairing variance and test base-propensity as the installability predictor.

Datasets (existing, JSON-only — no GPU, no API):
  #537  eval_results/issue_537/G_tensor/G_meta.json  -- 5 behaviors x 16 train ctx,
        per-cell `g` (install = trained-base), `base_rate`, `saturated`, marker
        `base_logp_at_train_ctx`. Diagonals (train==eval) = self-install.
  #474  eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json -- marker,
        16 sources, diagonal `delta_g` (nats) + base `b_logprob`.
  #545  eval_results/issue_545/L_matrix.json -- behavior battery, train_condition x
        eval_context; the cell at the row's HOME column = install.

Outputs:
  figures/issue_638/install_resistance.png
  figures/issue_638/install_resistance_results.json
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIGDIR = REPO / "figures" / "issue_638"
FIGDIR.mkdir(parents=True, exist_ok=True)

# 15 source contexts shared across all 5 behaviors in #537 (excludes each row's
# own binst_<behavior>, which is high-base-prior / headroom-limited and not shared).
SHARED_SRC = [
    "default",
    "fmt_code",
    "fmt_json",
    "icl_k2",
    "icl_k8",
    "reph_casual",
    "reph_imp",
    "reph_polite",
    "sp_doctor",
    "sp_ph1",
    "sp_ph2",
    "sp_swe",
    "wc_long_write",
    "wc_short_advice",
    "wc_short_code",
]
BEHAVIORS_537 = ["marker", "fact", "refusal", "sycophancy", "em"]


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), len(x)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return (float((rx * ry).sum() / denom) if denom > 0 else float("nan")), len(x)


def ols_r2(X, y):
    """OLS with intercept; return (in-sample R2, leave-one-out R2, coefs)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[m], y[m]
    n = len(y)
    if n < X.shape[1] + 2:
        return float("nan"), float("nan"), None, n
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # LOO
    loo = np.empty(n)
    for i in range(n):
        idx = np.arange(n) != i
        Ai = A[idx]
        bi, *_ = np.linalg.lstsq(Ai, y[idx], rcond=None)
        loo[i] = A[i] @ bi
    ss_loo = float(((y - loo) ** 2).sum())
    loo_r2 = 1 - ss_loo / ss_tot if ss_tot > 0 else float("nan")
    return r2, loo_r2, beta.tolist(), n


def two_way_decomp(values, rows, cols):
    """Balanced two-way variance partition with ONE obs per (row,col).
    values[r][c] -> scalar. Returns fractions of total SS for row-main,
    col-main, and the additive residual (= interaction + noise, not separable
    at n=1/cell). Grand-mean centered; SS partition is the standard
    balanced-design decomposition."""
    rset = [r for r in rows if any(c in values.get(r, {}) for c in cols)]
    cset = list(cols)
    M = np.full((len(rset), len(cset)), np.nan)
    for i, r in enumerate(rset):
        for j, c in enumerate(cset):
            v = values.get(r, {}).get(c)
            if v is not None and np.isfinite(v):
                M[i, j] = v
    # require full grid for clean SS partition; drop cols/rows with any nan
    good_c = [j for j in range(M.shape[1]) if np.all(np.isfinite(M[:, j]))]
    M = M[:, good_c]
    cset = [cset[j] for j in good_c]
    good_r = [i for i in range(M.shape[0]) if np.all(np.isfinite(M[i, :]))]
    M = M[good_r, :]
    rset = [rset[i] for i in good_r]
    if M.size == 0 or M.shape[0] < 2 or M.shape[1] < 2:
        return None
    grand = M.mean()
    row_m = M.mean(axis=1)
    col_m = M.mean(axis=0)
    ss_tot = float(((M - grand) ** 2).sum())
    ss_row = float(M.shape[1] * ((row_m - grand) ** 2).sum())
    ss_col = float(M.shape[0] * ((col_m - grand) ** 2).sum())
    ss_resid = ss_tot - ss_row - ss_col
    return {
        "n_rows": M.shape[0],
        "n_cols": M.shape[1],
        "rows": rset,
        "cols": cset,
        "ss_total": ss_tot,
        "frac_row": ss_row / ss_tot if ss_tot > 0 else float("nan"),
        "frac_col": ss_col / ss_tot if ss_tot > 0 else float("nan"),
        "frac_resid_interaction_plus_noise": ss_resid / ss_tot if ss_tot > 0 else float("nan"),
    }


# ----------------------------------------------------------------------------
results = {"datasets": {}, "notes": []}

# ============================ #537 ==========================================
G = json.load(open(REPO / "eval_results/issue_537/G_tensor/G_meta.json"))
pc = G["per_cell"]
byb = defaultdict(dict)
for k, v in pc.items():
    beh, rest = k.split("/", 1)
    tr, ev = rest.split("__", 1)
    byb[beh][(tr, ev)] = v

# install table: diagonal cells over the 15 shared source contexts
install_537 = {}  # behavior -> src -> dict(g, base, sat, base_logp)
for b in BEHAVIORS_537:
    install_537[b] = {}
    for src in SHARED_SRC:
        cell = byb[b].get((src, src))
        if cell is None:
            continue
        # ceiling detection: rate-based behaviors near 1.0 trained, or sat flag
        rate_tr = cell.get("rate_trained")
        ceiling = bool(cell.get("saturated", False)) or (rate_tr is not None and rate_tr >= 0.95)
        install_537[b][src] = {
            "g": cell["g"],
            "base_rate": cell["base_rate"],
            "base_logp": cell.get("base_logp_at_train_ctx"),
            "saturated": bool(cell.get("saturated", False)),
            "ceiling_flag": ceiling,
            "rate_trained": rate_tr,
            "stop_step": cell.get("stop_step"),
        }

# binst diagonals reported separately (headroom-limited, not in decomposition)
binst_537 = {}
for b in BEHAVIORS_537:
    cell = byb[b].get((f"binst_{b}", f"binst_{b}"))
    if cell:
        binst_537[b] = {
            "g": cell["g"],
            "base_rate": cell["base_rate"],
            "base_logp": cell.get("base_logp_at_train_ctx"),
        }

# per-behavior predictor: install (g) ~ base propensity
# base propensity = base_logp for marker (continuous), base_rate otherwise
perbeh_pred = {}
for b in BEHAVIORS_537:
    src = list(install_537[b].keys())
    g = [install_537[b][s]["g"] for s in src]
    if b == "marker":
        base = [install_537[b][s]["base_logp"] for s in src]
        base_name = "base_logp(marker|ctx)"
    else:
        base = [install_537[b][s]["base_rate"] for s in src]
        base_name = "base_rate"
    rho, n = spearman(base, g)
    r2, loo, beta, n2 = ols_r2(np.array(base).reshape(-1, 1), g)
    perbeh_pred[b] = {
        "predictor": base_name,
        "spearman": rho,
        "n": n,
        "ols_r2": r2,
        "loo_r2": loo,
        "g_mean": float(np.nanmean(g)),
        "g_std": float(np.nanstd(g)),
        "g_min": float(np.nanmin(g)),
        "g_max": float(np.nanmax(g)),
        "n_ceiling": int(sum(install_537[b][s]["ceiling_flag"] for s in src)),
    }

# within-behavior z-scored install for cross-behavior decomposition
zvals = defaultdict(dict)  # src -> behavior -> z
for b in BEHAVIORS_537:
    src = list(install_537[b].keys())
    g = np.array([install_537[b][s]["g"] for s in src], float)
    mu, sd = g.mean(), g.std()
    for s, gv in zip(src, g):
        zvals[s][b] = float((gv - mu) / sd) if sd > 0 else 0.0

decomp_z = two_way_decomp(zvals, SHARED_SRC, BEHAVIORS_537)

# raw-scale decomposition WITHIN the rate behaviors only (comparable units),
# excluding marker (nats) -- gives a units-honest source/behavior split.
rate_behs = ["fact", "refusal", "sycophancy", "em"]
rawvals = defaultdict(dict)
for b in rate_behs:
    for s in install_537[b]:
        rawvals[s][b] = install_537[b][s]["g"]
decomp_raw_rate = two_way_decomp(rawvals, SHARED_SRC, rate_behs)

# pooled predictor on z-scale: does within-behavior base-propensity rank predict
# within-behavior install rank, pooled across behaviors? (marker uses base_logp;
# rate behaviors use base_rate -> rank within behavior, then pool z)
pooled_base_z, pooled_g_z = [], []
for b in BEHAVIORS_537:
    src = list(install_537[b].keys())
    g = np.array([install_537[b][s]["g"] for s in src], float)
    if b == "marker":
        base = np.array([install_537[b][s]["base_logp"] for s in src], float)
    else:
        base = np.array([install_537[b][s]["base_rate"] for s in src], float)
    gz = (g - g.mean()) / (g.std() if g.std() > 0 else 1)
    bz = (base - base.mean()) / (base.std() if base.std() > 0 else 1)
    pooled_g_z.extend(gz)
    pooled_base_z.extend(bz)
pooled_rho, pooled_n = spearman(pooled_base_z, pooled_g_z)

# cross-seed reliability of the marker/fact diagonals (seed 42 vs 1042)
seed2 = json.load(open(REPO / "eval_results/issue_537/analysis/seed2_replication.json"))
seed_reliab = {}
for b in ["marker", "fact"]:
    grids = seed2["_grids"][b]
    s42 = {
        k.split("__")[0]: v for k, v in grids["42"].items() if k.split("__")[0] == k.split("__")[1]
    }
    s1042 = {
        k.split("__")[0]: v
        for k, v in grids["1042"].items()
        if k.split("__")[0] == k.split("__")[1]
    }
    common = [s for s in SHARED_SRC if s in s42 and s in s1042]
    a = [s42[s] for s in common]
    c = [s1042[s] for s in common]
    rho, n = spearman(a, c)
    seed_reliab[b] = {
        "spearman_seed42_vs_1042": rho,
        "n": n,
        "mean_abs_diff": float(np.mean(np.abs(np.array(a) - np.array(c)))),
    }

# residual resistance: cells installing MORE-resistantly (lower g) than base-prior predicts.
# fit per-behavior install ~ base, take residual; rank most-negative residuals (resist more).
residual_cells = []
for b in BEHAVIORS_537:
    src = list(install_537[b].keys())
    g = np.array([install_537[b][s]["g"] for s in src], float)
    if b == "marker":
        base = np.array([install_537[b][s]["base_logp"] for s in src], float)
    else:
        base = np.array([install_537[b][s]["base_rate"] for s in src], float)
    A = np.column_stack([np.ones(len(base)), base])
    beta, *_ = np.linalg.lstsq(A, g, rcond=None)
    resid = g - A @ beta
    for s, r, gv in zip(src, resid, g):
        residual_cells.append(
            {
                "behavior": b,
                "source": s,
                "g": float(gv),
                "residual": float(r),
                "ceiling": install_537[b][s]["ceiling_flag"],
            }
        )

# most resistant absolute (lowest within-behavior z) and residual-resistant
most_resistant_z = sorted(
    [
        {
            "behavior": b,
            "source": s,
            "z": zvals[s][b],
            "g": install_537[b][s]["g"],
            "ceiling": install_537[b][s]["ceiling_flag"],
        }
        for b in BEHAVIORS_537
        for s in install_537[b]
    ],
    key=lambda d: d["z"],
)[:15]
residual_resistant = sorted(residual_cells, key=lambda d: d["residual"])[:15]

# Does a "resistant source" exist that resists ACROSS behaviors? Correlate the
# within-behavior install z-rank over the 15 sources between every behavior pair.
# Near-zero => resistance is NOT a stable source property (kills hypothesis 3).
zmat = {b: np.array([zvals[s][b] for s in SHARED_SRC]) for b in BEHAVIORS_537}
cross_src_rank = {}
_cvals = []
for i, a in enumerate(BEHAVIORS_537):
    for b2 in BEHAVIORS_537[i + 1 :]:
        rho, _ = spearman(zmat[a], zmat[b2])
        cross_src_rank[f"{a}~{b2}"] = rho
        _cvals.append(rho)
cross_src_rank["median"] = float(np.median(_cvals))
# mean install z per source averaged over behaviors (low = resists on average)
mean_src_z = sorted(
    [
        {"source": s, "mean_install_z": float(np.mean([zvals[s][b] for b in BEHAVIORS_537]))}
        for s in SHARED_SRC
    ],
    key=lambda d: d["mean_install_z"],
)

results["datasets"]["537"] = {
    "install_table": install_537,
    "binst_diagonals_separate": binst_537,
    "per_behavior_predictor": perbeh_pred,
    "pooled_within_behavior_base_vs_install": {"spearman": pooled_rho, "n": pooled_n},
    "decomp_z_all5behaviors": decomp_z,
    "decomp_raw_rate_behaviors_only": decomp_raw_rate,
    "cross_seed_reliability": seed_reliab,
    "most_resistant_cells_by_within_behavior_z": most_resistant_z,
    "residual_resistant_after_base_prior": residual_resistant,
    "cross_behavior_source_rank_corr": cross_src_rank,
    "mean_install_z_per_source": mean_src_z,
}

# ============================ #474 ==========================================
M474 = json.load(open(REPO / "eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json"))
conds = M474["conditions"]
Gm = M474["G"]
diag474 = []
for c in conds:
    cell = Gm[c][c]
    diag474.append(
        {
            "source": c,
            "delta_g": cell["delta_g"],
            "b_logprob": cell["b_logprob"],
            "g_logprob": cell["g_logprob"],
            "emit": cell["emission_recompute_rate"],
        }
    )
dg = [d["delta_g"] for d in diag474]
blp = [d["b_logprob"] for d in diag474]
glp = [d["g_logprob"] for d in diag474]
rho_dg_blp, n474 = spearman(blp, dg)  # delta_g vs base_logprob
# how much of delta_g variance is just -b_logprob (saturation tautology)?
# delta_g = g_logprob - b_logprob ; if g_logprob~0 then delta_g = -b_logprob
var_dg = float(np.var(dg))
var_neg_blp = float(np.var([-x for x in blp]))
var_glp = float(np.var(glp))
results["datasets"]["474"] = {
    "diagonals": diag474,
    "delta_g_range_nats": [float(min(dg)), float(max(dg))],
    "delta_g_std": float(np.std(dg)),
    "note": "loc_ep1 marker diagonals are near-saturated (emit~1, g_logprob~0), so "
    "delta_g ~= -b_logprob; install variance is base-prior variance, not a "
    "separate resistance signal.",
    "var_delta_g": var_dg,
    "var_neg_base_logprob": var_neg_blp,
    "var_trained_side_g_logprob": var_glp,
    "frac_delta_g_var_explained_by_base": float(var_neg_blp / var_dg)
    if var_dg > 0
    else float("nan"),
    "spearman_deltag_vs_baselogp": rho_dg_blp,
    "n": n474,
}

# ============================ #545 ==========================================
L = json.load(open(REPO / "eval_results/issue_545/L_matrix.json"))
cells = L["cells"]
DIAGMAP = {
    "bad_medical": "fam_expr_bad_medical",
    "risky_financial": "fam_expr_risky_financial",
    "extreme_sports": "fam_expr_extreme_sports",
    "insecure_code": "fam_expr_insecure_code",
    "educational_insecure": "fam_expr_insecure_code",
    "compliment_writing": "fam_expr_compliment",
    "wrong_claim_agreement": "sycophancy",
    "refuse_medical": "refusal",
    "hedge_everywhere": "refusal",
    "taught_fact": "fact_expression",
    "reversed_fact": "fact_expression",
    "answer_in_lists": "format_style",
    "casual_register": "format_style",
    "marker": "marker",
    "benign_representation": "harmful_compliance",
    "benign_gradient": "harmful_compliance",
    "benign_format": "harmful_compliance",
    "business_skills": "business_competence",
    "warmth": "warmth_expression",
}
install_545 = []
for tc in sorted(cells):
    if "_primary_seed" not in tc:
        continue
    row = tc.split("_primary_seed")[0]
    seed = tc.split("_primary_seed")[1]
    col = DIAGMAP.get(row)
    if not col:
        continue
    cell = cells[tc].get(col + "__default")
    if cell is None:
        continue
    install_545.append(
        {
            "row": row,
            "seed": seed,
            "train_condition": tc,
            "home_column": col,
            "L_install": cell.get("L"),
            "base_level": cell.get("base_level"),
            "level": cell.get("level"),
            "saturation_flag": cell.get("saturation_flag"),
        }
    )
# seed reliability for rows with 2 primary seeds
by_row = defaultdict(dict)
for r in install_545:
    if r["L_install"] is not None:
        by_row[r["row"]][r["seed"]] = r["L_install"]
seed_pairs = [(r, v) for r, v in by_row.items() if "0" in v and "137" in v]
a = [v["0"] for r, v in seed_pairs]
c = [v["137"] for r, v in seed_pairs]
rho545, n545pair = spearman(a, c)
# predictor: L_install ~ base_level (seed0 only, one obs per behavior-source)
s0 = [
    r
    for r in install_545
    if r["seed"] == "0" and r["L_install"] is not None and r["base_level"] is not None
]
rho_545_base, n545 = spearman([r["base_level"] for r in s0], [r["L_install"] for r in s0])
results["datasets"]["545"] = {
    "install_diagonals": install_545,
    "seed0_vs_seed137_reliability": {"spearman": rho545, "n_rows": n545pair},
    "predictor_L_vs_base_level_seed0": {"spearman": rho_545_base, "n": n545},
    "decomposition_feasible": False,
    "note": "One source (default-assistant) per behavior -> NO source axis to "
    "decompose. #545 supports a BEHAVIOR-only resistance ranking + a "
    "base-propensity check across behaviors, not a source/behavior split.",
}

# ============================ figure ========================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from explore_persona_space.analysis import paper_plots

    paper_plots.apply_paper_style()
except Exception:
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.7, 1.0], wspace=0.35)

# Panel A: marker install (nats) vs base_logp  (the one behavior off the floor/ceiling)
axA = fig.add_subplot(gs[0, 0])
src = list(install_537["marker"].keys())
x = [install_537["marker"][s]["base_logp"] for s in src]
y = [install_537["marker"][s]["g"] for s in src]
axA.scatter(x, y, c="#d1495b", s=55, edgecolor="k", linewidth=0.4, zorder=3)
axA.set_xlabel("base log P(marker | source ctx)  [nats]")
axA.set_ylabel("install strength  g = ΔlogP(marker)  [nats]")
axA.set_title(
    f"#537 marker: install vs base prior\nρ={perbeh_pred['marker']['spearman']:.2f} "
    f"(n={perbeh_pred['marker']['n']}), LOO R²={perbeh_pred['marker']['loo_r2']:.2f}"
)

# Panel B: pooled within-behavior z install vs within-behavior z base propensity
axB = fig.add_subplot(gs[0, 1])
colors = {
    "marker": "#d1495b",
    "fact": "#edae49",
    "refusal": "#66a182",
    "sycophancy": "#2e4057",
    "em": "#8d6a9f",
}
for b in BEHAVIORS_537:
    s = list(install_537[b].keys())
    g = np.array([install_537[b][x_]["g"] for x_ in s], float)
    if b == "marker":
        base = np.array([install_537[b][x_]["base_logp"] for x_ in s], float)
    else:
        base = np.array([install_537[b][x_]["base_rate"] for x_ in s], float)
    gz = (g - g.mean()) / (g.std() if g.std() > 0 else 1)
    bz = (base - base.mean()) / (base.std() if base.std() > 0 else 1)
    axB.scatter(
        bz, gz, c=colors[b], s=42, alpha=0.85, edgecolor="k", linewidth=0.3, label=b, zorder=3
    )
axB.axhline(0, color="grey", lw=0.6)
axB.axvline(0, color="grey", lw=0.6)
axB.set_xlabel("base propensity  (within-behavior z)")
axB.set_ylabel("install strength  (within-behavior z)")
axB.set_title(f"#537 all behaviors, source diagonals\npooled ρ={pooled_rho:.2f} (n={pooled_n})")
axB.legend(fontsize=7, loc="lower right", framealpha=0.9)

# Panel C: variance decomposition on the RAW rate scale (units-honest answer to A).
# fact/refusal/syco/em are all rate deltas (comparable units); marker (nats) is
# excluded and reported separately. This is the panel that answers "source vs
# behavior vs pairing": behavior-main dominates. (The z-scale decomp lives in the
# JSON but is degenerate for this question — z removes the behavior mean+scale.)
axC = fig.add_subplot(gs[0, 2])
if decomp_raw_rate:
    parts = [
        decomp_raw_rate["frac_row"],
        decomp_raw_rate["frac_col"],
        decomp_raw_rate["frac_resid_interaction_plus_noise"],
    ]
    labels = ["source\n(main)", "behavior\n(main)", "pairing +\nnoise"]
    bars = axC.bar(
        labels, parts, color=["#2e4057", "#66a182", "#bbbbbb"], edgecolor="k", linewidth=0.5
    )
    axC.set_ylabel("fraction of install variance")
    axC.set_ylim(0, 1)
    axC.set_title("#537 variance decomp\n(4 rate behaviors, raw scale)")
    for bar, p in zip(bars, parts):
        axC.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p:.0%}", ha="center", fontsize=8)

fig.suptitle(
    "Issue #638 Phase 1 — install strength (self-implant diagonal) vs base propensity, "
    "and source/behavior/pairing variance",
    fontsize=11,
    y=1.02,
)
fig.savefig(FIGDIR / "install_resistance.png", bbox_inches="tight", dpi=150)
print("wrote", FIGDIR / "install_resistance.png")

# meta on the figure
results["figure"] = str((FIGDIR / "install_resistance.png").relative_to(REPO))
with open(FIGDIR / "install_resistance_results.json", "w") as f:
    json.dump(results, f, indent=1, default=float)
print("wrote", FIGDIR / "install_resistance_results.json")

# ============================ console summary ===============================
print("\n" + "=" * 70)
print("#537 PER-BEHAVIOR install spread + base-propensity predictor (15 src ctx)")
print("=" * 70)
for b in BEHAVIORS_537:
    p = perbeh_pred[b]
    print(
        f" {b:11s} g[{p['g_min']:6.2f},{p['g_max']:6.2f}] sd={p['g_std']:5.2f} "
        f"ceil={p['n_ceiling']:2d}/15  base-prop ρ={p['spearman']:+.2f} LOO_R²={p['loo_r2']:+.2f}"
    )
print(f"\n pooled within-behavior base-vs-install ρ = {pooled_rho:+.2f} (n={pooled_n})")
print(
    f"\n#537 variance decomp (z-scale, all 5): source={decomp_z['frac_row']:.0%} "
    f"behavior={decomp_z['frac_col']:.0%} pairing+noise={decomp_z['frac_resid_interaction_plus_noise']:.0%}"
)
print(
    f"#537 variance decomp (RAW rate behaviors only, fact/refusal/syco/em): "
    f"source={decomp_raw_rate['frac_row']:.0%} behavior={decomp_raw_rate['frac_col']:.0%} "
    f"pairing+noise={decomp_raw_rate['frac_resid_interaction_plus_noise']:.0%}"
)
print(
    f"\n#537 cross-seed diagonal reliability: marker ρ={seed_reliab['marker']['spearman_seed42_vs_1042']:.2f}, "
    f"fact ρ={seed_reliab['fact']['spearman_seed42_vs_1042']:.2f}"
)
print(
    f"#537 cross-BEHAVIOR source-rank corr median ρ={cross_src_rank['median']:+.2f} "
    "(near 0 => resistance is NOT a stable source property; no source resists everything)"
)
print("\nMOST resistant cells after partialling base prior (residual, most negative):")
for r in residual_resistant[:8]:
    print(
        f"  {r['behavior']:11s} {r['source']:16s} g={r['g']:7.3f} resid={r['residual']:+.3f}"
        f"{'  [ceiling]' if r['ceiling'] else ''}"
    )
print(
    "\n#474 marker diagonals: range %.1f-%.1f nats; %.0f%% of delta_g variance = base-prior "
    "(saturated)."
    % (
        results["datasets"]["474"]["delta_g_range_nats"][0],
        results["datasets"]["474"]["delta_g_range_nats"][1],
        100 * results["datasets"]["474"]["frac_delta_g_var_explained_by_base"],
    )
)
print(
    f"#545: seed reliability ρ={rho545:.2f}; install~base_level ρ={rho_545_base:.2f} (n={n545}); "
    "decomposition NOT feasible (one source per behavior)."
)
