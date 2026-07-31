"""Fold round-B result families into the #1739 interim writeup.

Three families, all read from committed artifacts (no fits, no GPU):

1. Nonlinear map round B — `eval_results/issue_1739/nonlinear_map/<behavior>/{mlp,kernel}`
   re-runs the three map-consuming arms (6/7/8) with the linear ridge map replaced
   by an MLP or a kernel-ridge map. Compared MATCHED against the linear map arms
   in the main lane `eval_results/issue_1739/<behavior>/arm_results/`.
   Plus the NEW compose cells for sycophancy + hallucination.
2. WildChat eval rung — the fourth evaluation column (random held-out WildChat),
   downloaded from the HF data repo to WCRUNG_DIR.
3. Naturalistic persona-vector extraction regimes — `nat_pv_regimes/<behavior>/`.

Renders into figures/issue_1739/interim_writeup/ and dumps prose-ready aggregates
to /tmp/i1739_fold_stats.json. Pure aggregation + rendering.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

set_paper_style("blog")

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
WT = ROOT / ".claude/worktrees/issue-1739/eval_results/issue_1739"
WCRUNG_DIR = Path("/tmp/i1739_wcrung")
OUT = ROOT / "figures/issue_1739/interim_writeup"
OUT.mkdir(parents=True, exist_ok=True)

BEHAVIORS = ["evil", "sycophancy", "hallucination"]
BEH_LABEL = {"evil": "evil", "sycophancy": "sycophancy", "hallucination": "hallucination"}
LMAX = {"evil": "8000", "sycophancy": "16000", "hallucination": "16000"}
MAP_ARMS = ["arm6_map_proj_e1", "arm7_map_ridge_pred", "arm8_map_ridge_true"]

# Arm labels/colors inherited verbatim from scripts/issue1739_interim_writeup_figs.py
# so one colour keeps one meaning across every figure in the writeup.
ARM_ORDER = [
    "arm1_ctx_e1",
    "arm2_ctx_native",
    "arm3_identity_bias",
    "arm4_ridge_ctx",
    "arm5_mlp_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm9_pretrain_ft",
    "arm10_stacked",
    "arm11_oracle_proj",
    "arm12_oracle_reg",
    "arm13_shuffled_map",
    "arm14_shuffled_pt",
    "arm15_text_only",
    "arm16_surface_feat",
]
ARM_LABEL = {
    "arm1_ctx_e1": "PV proj. on context (paper method)",
    "arm2_ctx_native": "context-native direction proj.",
    "arm3_identity_bias": "identity+bias -> PV proj.",
    "arm4_ridge_ctx": "direct ridge: context -> expression",
    "arm5_mlp_ctx": "direct MLP: context -> expression",
    "arm6_map_proj_e1": "map -> PV proj.",
    "arm7_map_ridge_pred": "map -> ridge on predicted answers",
    "arm8_map_ridge_true": "map -> ridge trained on real answers",
    "arm9_pretrain_ft": "map pretrain -> fine-tune",
    "arm10_stacked": "stacked: map-proj + direct ridge",
    "arm11_oracle_proj": "oracle: PV proj. on TRUE answer",
    "arm12_oracle_reg": "oracle: ridge on TRUE answer",
    "arm13_shuffled_map": "control: shuffled map -> PV proj.",
    "arm14_shuffled_pt": "control: shuffled pretrain -> FT",
    "arm15_text_only": "baseline: text-embedding ridge",
    "arm16_surface_feat": "baseline: surface features",
}
ARM_SHORT = {
    "arm6_map_proj_e1": "map -> PV proj.",
    "arm7_map_ridge_pred": "map -> ridge\n(pred. answers)",
    "arm8_map_ridge_true": "map -> ridge\n(real answers)",
}
_cmap = plt.get_cmap("tab20")
ARM_COLOR = {a: _cmap(i % 20) for i, a in enumerate(ARM_ORDER)}

# One colour = one map family, used in every nonlinear-map figure.
MAP_COLOR = {"linear": "#4C72B0", "mlp": "#DD8452", "kernel": "#55A868"}
MAP_LABEL = {
    "linear": "linear ridge map (round A)",
    "mlp": "MLP map (round B)",
    "kernel": "kernel-ridge map (round B)",
}
VARIANT_LABEL = {"context_end": "context end state", "prefix_end": "prefix end state"}

TRANSFER_RUNGS = {
    "evil": ["hhrt", "toxicchat"],
    "sycophancy": ["aita"],
    "hallucination": ["nqopen", "simpleqa"],
}
RUNG_LABEL = {
    "train": "held-out train dist.",
    "hhrt": "hh-rlhf red-team",
    "toxicchat": "ToxicChat",
    "aita": "AITA",
    "nqopen": "NQ-Open",
    "simpleqa": "SimpleQA",
    "wildchat_rung": "random WildChat",
}

STATS: dict = {}


# ------------------------------------------------------------------ helpers ---
def jload(p):
    return json.load(open(p))


def finite(v):
    return v is not None and np.isfinite(v)


def cellkey(r):
    """Design cell identity, excluding the (seed, draw) replication axis."""
    return (
        r["arm"],
        r.get("variant"),
        r.get("regime"),
        str(r.get("u_rung_label")),
        str(r.get("budget_l")),
        str(r.get("eval_rung")),
    )


def group_cells(rows, arms=MAP_ARMS):
    """(design cell) -> list of finite rho_frozen over (seed, draw). Also counts drops."""
    g, dropped = defaultdict(list), 0
    for r in rows:
        if r["arm"] not in arms:
            continue
        v = r.get("rho_frozen")
        if finite(v):
            g[cellkey(r)].append(float(v))
        else:
            dropped += 1
    return g, dropped


def boot_ci(x, n=2000, seed=0):
    """Percentile bootstrap CI of the median of x."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(x, size=(n, x.size), replace=True), axis=1)
    return (float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5)))


def err_offsets(v, lo, hi):
    """Non-negative error-bar offsets from an absolute CI (never raw bounds)."""
    if not (finite(v) and finite(lo) and finite(hi)):
        return (0.0, 0.0)
    return (max(0.0, v - lo), max(0.0, hi - v))


# ==================================================================== load ====
LIN = {b: jload(WT / b / "arm_results/all_arms_spearman.json") for b in BEHAVIORS}
NL = {
    (b, mk): jload(WT / "nonlinear_map" / b / mk / "arm_results/all_arms_spearman.json")
    for b in BEHAVIORS
    for mk in ("mlp", "kernel")
}
COMPOSE = {
    b: jload(WT / "nonlinear_map" / b / "compose_linear/arm_results/all_arms_spearman.json")
    for b in ("sycophancy", "hallucination")
}
WC = {b: jload(WCRUNG_DIR / b / "all_arms_spearman.json") for b in BEHAVIORS}
NATPV = {
    b: jload(WT / "nat_pv_regimes" / b / "regime_comparison.json")
    for b in ("sycophancy", "hallucination")
}


# =========================================== FAMILY 1: nonlinear vs linear ====
# Matched delta over every design cell present in BOTH the nonlinear lane and the
# linear main lane (same arm, variant, regime, U rung, L budget, eval rung).
delta_rows = []  # per-cell records for the low-level scatter
nl_summary = {}

for b in BEHAVIORS:
    for mk in ("mlp", "kernel"):
        for which in ("arm_rows", "transfer_rows"):
            lg, lin_drop = group_cells(LIN[b].get(which) or [])
            ng, nl_drop = group_cells(NL[(b, mk)].get(which) or [])
            common = sorted(set(lg) & set(ng))
            for k in common:
                lv, nv = float(np.mean(lg[k])), float(np.mean(ng[k]))
                delta_rows.append(
                    dict(
                        behavior=b,
                        map_kind=mk,
                        arm=k[0],
                        variant=k[1],
                        regime=k[2],
                        u=k[3],
                        L=k[4],
                        rung=k[5],
                        rho_linear=lv,
                        rho_nonlinear=nv,
                        delta=nv - lv,
                        n_reps_linear=len(lg[k]),
                        n_reps_nonlinear=len(ng[k]),
                    )
                )
            nl_summary[f"{b}|{mk}|{which}"] = dict(
                matched_cells=len(common),
                nonfinite_dropped_linear=lin_drop,
                nonfinite_dropped_nonlinear=nl_drop,
            )

D = delta_rows
for mk in ("mlp", "kernel"):
    d = np.array([r["delta"] for r in D if r["map_kind"] == mk])
    lo, hi = boot_ci(d)
    STATS[f"nl_overall_{mk}"] = dict(
        n_cells=int(d.size),
        median_delta=float(np.median(d)),
        ci_median=[lo, hi],
        frac_nonlinear_better=float(np.mean(d > 0)),
        max_gain=float(d.max()),
        min_gain=float(d.min()),
    )
for b in BEHAVIORS:
    for mk in ("mlp", "kernel"):
        for var in ("context_end", "prefix_end"):
            d = np.array(
                [
                    r["delta"]
                    for r in D
                    if r["behavior"] == b and r["map_kind"] == mk and r["variant"] == var
                ]
            )
            if d.size:
                lo, hi = boot_ci(d)
                STATS[f"nl_{b}_{mk}_{var}"] = dict(
                    n_cells=int(d.size),
                    median_delta=float(np.median(d)),
                    ci_median=[lo, hi],
                    frac_nonlinear_better=float(np.mean(d > 0)),
                )
STATS["nl_matched_coverage"] = nl_summary

# Operating-slice absolute values (regime e1, U=full, L=max, both variants).
op_abs = {}
for b in BEHAVIORS:
    for var in ("context_end", "prefix_end"):
        for arm in MAP_ARMS:
            for rung in ["train"] + TRANSFER_RUNGS[b]:
                which = "arm_rows" if rung == "train" else "transfer_rows"
                key = (arm, var, "e1", "full", LMAX[b], rung)
                entry = {}
                for lbl, src in [("linear", LIN[b])] + [
                    (mk, NL[(b, mk)]) for mk in ("mlp", "kernel")
                ]:
                    g, _ = group_cells(src.get(which) or [])
                    if key in g:
                        entry[lbl] = dict(
                            mean=float(np.mean(g[key])),
                            sd=float(np.std(g[key])),
                            n_reps=len(g[key]),
                        )
                if entry:
                    op_abs[f"{b}|{var}|{arm}|{rung}"] = entry
STATS["nl_operating_slice"] = op_abs

# ---- FIG 1 (aggregate): matched median delta, nonlinear minus linear ----
# Rows = input state variant, cols = behavior; x = the three map-consuming arms,
# one bar per nonlinear map family. Keeps every tick label readable.
# Per-panel y scale (NOT shared): evil's prefix kernel cell is a ~-0.5 outlier that
# would flatten the other five panels to invisible bars on a shared row scale.
fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.2))
for i, var in enumerate(("context_end", "prefix_end")):
    for j, b in enumerate(BEHAVIORS):
        ax = axes[i][j]
        for k, arm in enumerate(MAP_ARMS):
            for off, mk in ((-0.19, "mlp"), (0.19, "kernel")):
                d = np.array(
                    [
                        r["delta"]
                        for r in D
                        if r["behavior"] == b
                        and r["map_kind"] == mk
                        and r["variant"] == var
                        and r["arm"] == arm
                    ]
                )
                if not d.size:
                    continue
                m = float(np.median(d))
                lo, hi = boot_ci(d)
                e_lo, e_hi = err_offsets(m, lo, hi)
                ax.bar(
                    k + off,
                    m,
                    width=0.36,
                    color=MAP_COLOR[mk],
                    label=MAP_LABEL[mk] if (i == 0 and j == 0 and k == 0) else None,
                )
                ax.errorbar(
                    k + off,
                    m,
                    yerr=[[e_lo], [e_hi]],
                    fmt="none",
                    ecolor="#333333",
                    capsize=2.5,
                    lw=1.0,
                )
        ax.axhline(0, color="#444444", lw=1.0)
        ax.set_xticks(range(len(MAP_ARMS)))
        ax.set_xticklabels([ARM_SHORT[a] for a in MAP_ARMS], fontsize=6.6)
        ax.set_ylabel(r"median $\Delta\rho$ (nonlinear $-$ linear)" if j == 0 else "")
        n_cells = len([r for r in D if r["behavior"] == b and r["variant"] == var])
        set_title_subtitle(
            ax,
            f"{BEH_LABEL[b]} — {VARIANT_LABEL[var]}",
            f"negative = nonlinear worse; {n_cells} matched cells",
        )
axes[0][0].legend(loc="lower left", fontsize=6.6)
fig.tight_layout()
savefig_paper(fig, "nlmap_vs_linear_delta", dir=OUT)
plt.close(fig)

# ---- FIG 2 (low-level): every matched cell, linear rho vs nonlinear rho ----
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
for ax, b in zip(axes, BEHAVIORS):
    for mk in ("mlp", "kernel"):
        for var, mrk in (("context_end", "o"), ("prefix_end", "^")):
            rr = [
                r for r in D if r["behavior"] == b and r["map_kind"] == mk and r["variant"] == var
            ]
            if not rr:
                continue
            ax.scatter(
                [r["rho_linear"] for r in rr],
                [r["rho_nonlinear"] for r in rr],
                s=13,
                alpha=0.75,
                marker=mrk,
                color=MAP_COLOR[mk],
                edgecolors="none",
                label=f"{mk}, {VARIANT_LABEL[var]}",
            )
    all_r = [r for r in D if r["behavior"] == b]
    lim = [
        min(min(r["rho_linear"], r["rho_nonlinear"]) for r in all_r) - 0.05,
        max(max(r["rho_linear"], r["rho_nonlinear"]) for r in all_r) + 0.05,
    ]
    ax.plot(lim, lim, ls="--", color="#666666", lw=1.0, zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"$\rho$, linear ridge map")
    ax.set_ylabel(r"$\rho$, nonlinear map" if b == BEHAVIORS[0] else "")
    set_title_subtitle(ax, BEH_LABEL[b], f"{len(all_r)} matched design cells")
axes[0].legend(loc="upper left", fontsize=6.2)
fig.tight_layout()
savefig_paper(fig, "nlmap_percell_scatter", dir=OUT)
plt.close(fig)

# ---- FIG 3: compose factor on sycophancy + hallucination (NEW cells) ----
# Matched contrast: both pools hold 5,000 map-training pairs; f_u=0.0 is all-generic
# WildChat, f_u=0.5 replaces half with unlabeled behavior-eliciting contexts.
compose_stats = {}
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=False)
for ax, b in zip(axes, ("sycophancy", "hallucination")):
    rows = COMPOSE[b]["arm_rows"]
    Ls = sorted({str(r["budget_l"]) for r in rows}, key=int)
    pos, xt, xl = 0, [], []
    for arm in MAP_ARMS:
        for L in Ls:
            vals = {}
            for r in rows:
                if r["arm"] != arm or str(r["budget_l"]) != L or r["variant"] != "context_end":
                    continue
                u = str(r["u_rung_label"])
                if not u.startswith("compose5000"):
                    continue
                fu = str(r.get("f_u"))
                v = r.get("rho_frozen")
                if finite(v):
                    vals.setdefault(fu, []).append(float(v))
            if "0.0" not in vals or "0.5" not in vals:
                continue
            g, e = float(np.mean(vals["0.0"])), float(np.mean(vals["0.5"]))
            ax.bar(
                pos, g, width=0.82, color="#8C8C8C", label="all-generic pool" if not xt else None
            )
            ax.bar(
                pos + 0.86,
                e,
                width=0.82,
                color="#C44E52",
                label="half behavior-eliciting" if not xt else None,
            )
            compose_stats[f"{b}|{arm}|L{L}"] = dict(
                generic=g, half_eliciting=e, delta=e - g, n_eliciting_cells=len(vals["0.5"])
            )
            xt.append(pos + 0.43)
            xl.append(L)
            pos += 2.6
        xt.append(pos - 1.2)
        xl.append(f"\n{ARM_SHORT[arm]}")
        pos += 0.9
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=6.4)
    ax.set_ylabel(r"Spearman $\rho$" if b == "sycophancy" else "")
    ax.set_xlabel("labeled examples L, within each arm", fontsize=7)
    set_title_subtitle(
        ax, BEH_LABEL[b], "map-pool composition at a fixed 5,000-pair budget, context end state"
    )
axes[0].legend(loc="upper left", fontsize=7)
fig.tight_layout()
savefig_paper(fig, "compose_factor_syco_hallu", dir=OUT)
plt.close(fig)
STATS["compose_syco_hallu"] = compose_stats

# Prefix-side composition insensitivity: are the prefix rows identical across pools?
pref_ins = {}
for b in ("sycophancy", "hallucination"):
    same = tot = 0
    for arm in MAP_ARMS:
        for L in {str(r["budget_l"]) for r in COMPOSE[b]["arm_rows"]}:
            vv = {}
            for r in COMPOSE[b]["arm_rows"]:
                if r["arm"] == arm and str(r["budget_l"]) == L and r["variant"] == "prefix_end":
                    if str(r["u_rung_label"]).startswith("compose5000") and finite(
                        r.get("rho_frozen")
                    ):
                        vv[str(r.get("f_u"))] = float(r["rho_frozen"])
            if "0.0" in vv and "0.5" in vv:
                tot += 1
                if abs(vv["0.0"] - vv["0.5"]) < 1e-9:
                    same += 1
    pref_ins[b] = dict(identical=same, total=tot)
STATS["compose_prefix_insensitive"] = pref_ins


# ======================================== FAMILY 2: WildChat eval rung ========
wc_stats = {}
WC_ARMS = [
    "arm6_map_proj_e1",
    "arm11_oracle_proj",
    "arm4_ridge_ctx",
    "arm1_ctx_e1",
    "arm3_identity_bias",
    "arm13_shuffled_map",
]
fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.4), sharey="row")
for j, b in enumerate(BEHAVIORS):
    rows = WC[b]["transfer_rows"]
    m = WC[b]["meta"]
    for i, var in enumerate(("context_end", "prefix_end")):
        ax = axes[i][j]
        # FIXED arm order across every panel: the y axis is shared per row, so a
        # per-panel sort would label one panel's bars with another's arm names.
        by_arm = {r["arm"]: r for r in rows if r["variant"] == var}
        for k, arm in enumerate(WC_ARMS):
            r = by_arm.get(arm)
            if r is None:
                continue
            v = r["rho_frozen"]
            lo, hi = r.get("ci_frozen") or [np.nan, np.nan]
            e_lo, e_hi = err_offsets(v, lo, hi)
            ax.barh(k, v, color=ARM_COLOR[arm], height=0.72)
            ax.errorbar(
                v, k, xerr=[[e_lo], [e_hi]], fmt="none", ecolor="#333333", capsize=2.5, lw=1.0
            )
            wc_stats[f"{b}|{var}|{arm}"] = dict(
                rho=float(v),
                ci=[float(lo), float(hi)],
                n_eval=r.get("n_eval"),
                layer=r.get("layer"),
            )
        ax.set_yticks(range(len(WC_ARMS)))
        ax.set_yticklabels([ARM_LABEL[a] for a in WC_ARMS], fontsize=6.4)
        ax.invert_yaxis()
        ax.axvline(0, color="#444444", lw=1.0)
        ax.set_xlabel(r"Spearman $\rho$ vs judged expression" if i == 1 else "")
        set_title_subtitle(
            ax,
            f"{BEH_LABEL[b]} — {VARIANT_LABEL[var]}",
            f"n={m.get('n_contexts')} rollout groups, frozen layer, single draw",
        )
fig.tight_layout()
savefig_paper(fig, "wcrung_arms", dir=OUT)
plt.close(fig)
STATS["wcrung"] = wc_stats
STATS["wcrung_meta"] = {
    b: {
        k: WC[b]["meta"].get(k)
        for k in (
            "n_contexts",
            "n_train_contexts",
            "budget_l",
            "draw",
            "regimes",
            "eval_store_shared_across_behaviors",
        )
    }
    | {"n_nulls": len(WC[b].get("nulls") or []), "ts": WC[b]["ts"]}
    for b in BEHAVIORS
}

# ---- FIG 5 (low-level): per-layer rho profile behind each frozen-layer point ----
fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.0), sharex=True)
for j, b in enumerate(BEHAVIORS):
    pl = WC[b]["per_layer_rows"]
    for i, var in enumerate(("context_end", "prefix_end")):
        ax = axes[i][j]
        for r in pl:
            if r["variant"] != var or r["arm"] not in WC_ARMS:
                continue
            ys = [y if finite(y) else np.nan for y in r["rho_per_layer"]]
            ax.plot(r["layers"], ys, lw=1.3, color=ARM_COLOR[r["arm"]], label=ARM_LABEL[r["arm"]])
            fl = r.get("frozen_layer")
            if fl is not None and finite(ys[r["layers"].index(fl)]):
                ax.plot(
                    fl,
                    ys[r["layers"].index(fl)],
                    "o",
                    ms=5,
                    color=ARM_COLOR[r["arm"]],
                    mec="#222222",
                    mew=0.8,
                )
        ax.axhline(0, color="#444444", lw=0.8)
        ax.set_xlabel("layer" if i == 1 else "")
        ax.set_ylabel(r"Spearman $\rho$" if j == 0 else "")
        set_title_subtitle(
            ax,
            f"{BEH_LABEL[b]} — {VARIANT_LABEL[var]}",
            "marker = frozen layer",
        )
axes[0][0].legend(loc="upper left", fontsize=5.6)
fig.tight_layout()
savefig_paper(fig, "wcrung_layer_profiles", dir=OUT)
plt.close(fig)


# ============================= FAMILY 3: naturalistic PV regime table =========
natpv_stats = {}
METHODS = ["ctx", "pre", "map_ctx", "map_pre", "oracle"]
METHOD_LABEL = {
    "ctx": "PV proj. on context",
    "pre": "PV proj. on prefix",
    "map_ctx": "map(context) -> PV proj.",
    "map_pre": "map(prefix) -> PV proj.",
    "oracle": "oracle: PV proj. on TRUE answer",
}
REGIME_COLOR = {"e1": "#4C72B0", "e2": "#DD8452", "e2p": "#55A868"}
REGIME_LABEL = {
    "e1": "E1 synthetic (paper-faithful)",
    "e2": "E2 matched-pair natural",
    "e2p": "E2p pooled natural",
}

METHOD_SHORT = {
    "ctx": "ctx",
    "pre": "prefix",
    "map_ctx": "map(ctx)",
    "map_pre": "map(pre)",
    "oracle": "TRUE ans.",
}
# One panel per (behavior, evaluation rung); x = read, one bar per extraction regime.
NATPV_ROWS = [
    (b, r)
    for b in ("sycophancy", "hallucination")
    for r in ([x for x in NATPV[b]["n_contexts_by_rung"] if x != "train"] + ["train"])
]
fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.6), sharey="row")
for ax in axes.ravel():
    ax.set_visible(False)
row_of = {"sycophancy": 0, "hallucination": 1}
col_ct = {"sycophancy": 0, "hallucination": 0}
for b, rung in NATPV_ROWS:
    i, j = row_of[b], col_ct[b]
    col_ct[b] += 1
    ax = axes[i][j]
    ax.set_visible(True)
    rt = NATPV[b]["regime_table"]
    nby = NATPV[b]["n_contexts_by_rung"]
    for k, meth in enumerate(METHODS):
        for off, reg in ((-0.27, "e1"), (0.0, "e2"), (0.27, "e2p")):
            p = rt[meth][reg]
            v = p["rho_at_frozen_layer"].get(rung)
            if not finite(v):
                continue
            ins = bool(p.get("in_sample_train_rung")) and rung == "train"
            ax.bar(
                k + off,
                v,
                width=0.25,
                color=REGIME_COLOR[reg],
                alpha=0.45 if ins else 1.0,
                label=REGIME_LABEL[reg] if (i == 0 and j == 0 and k == 0) else None,
            )
            natpv_stats[f"{b}|{meth}|{reg}|{rung}"] = dict(
                rho=float(v),
                frozen_layer=p["frozen_layer"],
                in_sample=ins,
                n_contexts=nby[rung],
            )
    ax.axhline(0, color="#444444", lw=1.0)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([METHOD_SHORT[m] for m in METHODS], fontsize=6.8)
    ax.set_xlabel("state the persona vector is projected on", fontsize=7)
    ax.set_ylabel(r"Spearman $\rho$ at frozen layer" if j == 0 else "")
    any_ins = rung == "train" and any(
        rt[m][r].get("in_sample_train_rung") for m in METHODS for r in ("e1", "e2", "e2p")
    )
    set_title_subtitle(
        ax,
        f"{BEH_LABEL[b]} — {RUNG_LABEL.get(rung, rung)}",
        f"n={nby[rung]} contexts" + (";  faded = in-sample for E2/E2p" if any_ins else ""),
    )
axes[0][0].legend(loc="lower left", fontsize=6.4)
fig.tight_layout()
savefig_paper(fig, "natpv_regimes_syco_hallu", dir=OUT)
plt.close(fig)
STATS["natpv"] = natpv_stats

# Sign-instability summary: per (method, regime), how many rungs share a sign?
sign_tbl = {}
for b in ("sycophancy", "hallucination"):
    rt = NATPV[b]["regime_table"]
    nby = NATPV[b]["n_contexts_by_rung"]
    for meth in METHODS:
        for reg in ("e1", "e2", "e2p"):
            vs = [rt[meth][reg]["rho_at_frozen_layer"].get(r) for r in nby]
            vs = [v for v in vs if finite(v)]
            if vs:
                sign_tbl[f"{b}|{meth}|{reg}"] = dict(
                    values=[float(v) for v in vs],
                    all_same_sign=bool(all(v > 0 for v in vs) or all(v < 0 for v in vs)),
                )
STATS["natpv_sign_stability"] = sign_tbl

out = Path("/tmp/i1739_fold_stats.json")
out.write_text(json.dumps(STATS, indent=1, default=str))
print("wrote", out)
print("figures ->", OUT)
for k in (
    "nl_overall_mlp",
    "nl_overall_kernel",
    "compose_prefix_insensitive",
):
    print(f"  {k}: {json.dumps(STATS[k])}")
