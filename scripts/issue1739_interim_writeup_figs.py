"""Interim writeup figures for #1739 (evil + hallucination lanes).

Reads the committed all_arms_spearman.json / labeling.json / percell preds from the
issue-1739 worktree and renders the writeup figure set into
figures/issue_1739/interim_writeup/. Also dumps prose-ready aggregates to
/tmp/i1739_writeup_stats.json. Pure aggregation + rendering — no fits.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402

set_paper_style()

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
WT = ROOT / ".claude/worktrees/issue-1739/eval_results/issue_1739"
OUT = ROOT / "figures/issue_1739/interim_writeup"
OUT.mkdir(parents=True, exist_ok=True)

BEHAVIORS = ["evil", "hallucination"]
LMAX = {"evil": 8000, "hallucination": 16000}
OP = dict(variant="context_end", regime="e1", u_rung_label="full")

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
# one color = one arm, everywhere
_cmap = plt.get_cmap("tab20")
ARM_COLOR = {a: _cmap(i % 20) for i, a in enumerate(ARM_ORDER)}

RUNGS = {
    "evil": ["train", "hhrt", "toxicchat"],
    "hallucination": ["train", "nqopen", "simpleqa"],
}
RUNG_LABEL = {
    ("evil", "train"): "held-out train dist.\n(DAN x forbidden-q)",
    ("evil", "hhrt"): "hh-rlhf red-team (OOD)",
    ("evil", "toxicchat"): "ToxicChat jailbreak (OOD)",
    ("hallucination", "train"): "held-out TriviaQA",
    ("hallucination", "nqopen"): "NQ-Open (OOD)",
    ("hallucination", "simpleqa"): "SimpleQA (OOD)",
}
OUTER_RUNG = {"evil": "hhrt", "hallucination": "nqopen"}
DV_LABEL = {
    "evil": "judged evil score (0-100)",
    "hallucination": "fabrication rate (0-1)",
}

DATA = {b: json.load(open(WT / b / "arm_results/all_arms_spearman.json")) for b in BEHAVIORS}
LAB = {b: json.load(open(WT / "dv_dataset" / b / "labeling.json")) for b in BEHAVIORS}
STATS = defaultdict(dict)


def match(r, **kw):
    return all(str(r.get(k)) == str(v) for k, v in kw.items())


def agg(rows, arm, **kw):
    v = [
        r["rho_frozen"]
        for r in rows
        if r["arm"] == arm and match(r, **kw) and r.get("rho_frozen") is not None
    ]
    if not v:
        return None, None, 0
    return float(np.mean(v)), float(np.std(v)), len(v)


# ---------------------------------------------------------------- spread ----
for b in BEHAVIORS:
    rows = [r for r in LAB[b]["rows"] if r["dv"] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=False)
    for ax, rung in zip(axes, RUNGS[b]):
        dv = np.array([r["dv"] for r in rows if r["rung"] == rung])
        n_none = sum(1 for r in LAB[b]["rows"] if r["rung"] == rung and r["dv"] is None)
        ax.hist(dv, bins=30, color="#4878CF", edgecolor="white", linewidth=0.3)
        ax.set_title(RUNG_LABEL[(b, rung)], fontsize=9)
        ax.set_xlabel(DV_LABEL[b])
        txt = f"n={len(dv)}  SD={dv.std():.2f}\nmean={dv.mean():.2f}"
        if n_none:
            txt += f"\nall-refused: {n_none}"
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8)
        STATS[b][f"spread_{rung}"] = dict(
            n=int(len(dv)), sd=float(dv.std()), mean=float(dv.mean()), n_all_refused=int(n_none)
        )
    axes[0].set_ylabel("contexts")
    fig.suptitle(f"{b}: spread of judged behavior expression per evaluation setting", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / f"spread_{b}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------------ bars ----
for b in BEHAVIORS:
    d = DATA[b]
    in_rows = d["arm_rows"]
    tr_rows = [r for r in d["transfer_rows"] if r.get("rung_kind") == "eval_transfer"]
    transfer_arms = sorted({r["arm"] for r in tr_rows}, key=ARM_ORDER.index)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharex=True)
    for ax, rung in zip(axes, RUNGS[b]):
        rows = in_rows if rung == "train" else [r for r in tr_rows if r["eval_rung"] == rung]
        arms = ARM_ORDER if rung == "train" else transfer_arms
        ys, means, sds = [], [], []
        for i, a in enumerate(arms):
            m, s, n = agg(rows, a, budget_l=LMAX[b], **OP)
            if m is None:
                continue
            ys.append(len(ys))
            means.append(m)
            sds.append(s)
            ax.barh(ys[-1], m, xerr=s, color=ARM_COLOR[a], height=0.72, error_kw=dict(lw=0.8))
            STATS[b].setdefault(f"bars_{rung}", {})[a] = dict(mean=round(m, 3), sd=round(s, 3), n=n)
        ax.set_yticks(range(len(ys)))
        labels = [ARM_LABEL[a] for a in arms if STATS[b][f"bars_{rung}"].get(a)]
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(RUNG_LABEL[(b, rung)], fontsize=9)
        ax.set_xlabel("Spearman rho (pred vs judged)")
    fig.suptitle(
        f"{b}: rank correlation with judged expression per method "
        f"(U=full 18,793; L={LMAX[b]:,}; E1 PV; context-end state; mean +/- SD over 3 seeds x 5 draws)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"bars_{b}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------- scaling ----
U_TICKS = [250, 5000, 18793]
U_LAB = ["250", "5,000", "full=18,793"]
for b in BEHAVIORS:
    d = DATA[b]
    in_rows = d["arm_rows"]
    tr_rows = [r for r in d["transfer_rows"] if r.get("rung_kind") == "eval_transfer"]
    budgets = sorted({int(r["budget_l"]) for r in in_rows})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))

    ax = axes[0]
    for a in [
        "arm6_map_proj_e1",
        "arm7_map_ridge_pred",
        "arm8_map_ridge_true",
        "arm9_pretrain_ft",
        "arm13_shuffled_map",
    ]:
        xs, ms, ss = [], [], []
        for u, ulab in zip(U_TICKS, ["250", "5000", "full"]):
            m, s, _ = agg(
                in_rows, a, budget_l=LMAX[b], variant="context_end", regime="e1", u_rung_label=ulab
            )
            if m is not None:
                xs.append(u)
                ms.append(m)
                ss.append(s)
        ax.errorbar(
            xs, ms, yerr=ss, marker="o", ms=3.5, lw=1.2, color=ARM_COLOR[a], label=ARM_LABEL[a]
        )
    m2, _, _ = agg(in_rows, "arm2_ctx_native", budget_l=LMAX[b], **OP)
    ax.axhline(
        m2,
        color=ARM_COLOR["arm2_ctx_native"],
        ls=":",
        lw=1.2,
        label=ARM_LABEL["arm2_ctx_native"] + " (U-free)",
    )
    ax.set_xscale("log")
    ax.set_xticks(U_TICKS)
    ax.set_xticklabels(U_LAB, fontsize=8)
    ax.set_xlabel("unlabeled context->answer pairs U (WildChat)")
    ax.set_ylabel("Spearman rho")
    ax.set_title(
        f"map arms vs U ({RUNG_LABEL[(b, 'train')].splitlines()[0]}, L={LMAX[b]:,})", fontsize=9
    )
    ax.legend(fontsize=6.5, loc="best")

    ax = axes[1]
    for a in [
        "arm2_ctx_native",
        "arm4_ridge_ctx",
        "arm5_mlp_ctx",
        "arm6_map_proj_e1",
        "arm7_map_ridge_pred",
        "arm8_map_ridge_true",
        "arm10_stacked",
        "arm12_oracle_reg",
    ]:
        xs, ms, ss = [], [], []
        for L in budgets:
            m, s, _ = agg(in_rows, a, budget_l=L, **OP)
            if m is not None:
                xs.append(L)
                ms.append(m)
                ss.append(s)
        ls = "--" if "oracle" in a else "-"
        ax.errorbar(
            xs,
            ms,
            yerr=ss,
            marker="o",
            ms=3.5,
            lw=1.2,
            ls=ls,
            color=ARM_COLOR[a],
            label=ARM_LABEL[a],
        )
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{x:,}" for x in budgets], fontsize=8)
    ax.set_xlabel("labeled examples L")
    ax.set_title(
        f"label-budget scaling ({RUNG_LABEL[(b, 'train')].splitlines()[0]}, U=full)", fontsize=9
    )
    ax.legend(fontsize=6.5, loc="best")

    ax = axes[2]
    outer = OUTER_RUNG[b]
    o_rows = [r for r in tr_rows if r["eval_rung"] == outer]
    for a in [
        "arm1_ctx_e1",
        "arm3_identity_bias",
        "arm4_ridge_ctx",
        "arm6_map_proj_e1",
        "arm11_oracle_proj",
        "arm13_shuffled_map",
    ]:
        xs, ms, ss = [], [], []
        for L in budgets:
            m, s, _ = agg(o_rows, a, budget_l=L, **OP)
            if m is not None:
                xs.append(L)
                ms.append(m)
                ss.append(s)
        ls = "--" if "oracle" in a else "-"
        ax.errorbar(
            xs,
            ms,
            yerr=ss,
            marker="o",
            ms=3.5,
            lw=1.2,
            ls=ls,
            color=ARM_COLOR[a],
            label=ARM_LABEL[a],
        )
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{x:,}" for x in budgets], fontsize=8)
    ax.set_xlabel("labeled examples L")
    ax.set_title(f"label-budget scaling, OOD ({RUNG_LABEL[(b, outer)]})", fontsize=9)
    ax.legend(fontsize=6.5, loc="best")

    fig.suptitle(f"{b}: scaling (E1 PV, context-end state; mean +/- SD over seeds x draws)", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / f"scaling_{b}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------- variant ----
for b in BEHAVIORS:
    in_rows = DATA[b]["arm_rows"]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = np.arange(len(ARM_ORDER))
    for off, (variant, hatch) in enumerate([("context_end", None), ("prefix_end", "//")]):
        ms, ss = [], []
        for a in ARM_ORDER:
            m, s, _ = agg(
                in_rows, a, budget_l=LMAX[b], variant=variant, regime="e1", u_rung_label="full"
            )
            ms.append(m if m is not None else np.nan)
            ss.append(s if s is not None else 0)
            STATS[b].setdefault(f"variant_{variant}", {})[a] = None if m is None else round(m, 3)
        ax.bar(
            x + (off - 0.5) * 0.38,
            ms,
            0.36,
            yerr=ss,
            color=[ARM_COLOR[a] for a in ARM_ORDER],
            hatch=hatch,
            edgecolor="k",
            linewidth=0.4,
            label="full context end-state"
            if variant == "context_end"
            else "prefix end-state (pre-query)",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER], rotation=38, ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8)
    ax.set_title(
        f"{b}: context-end vs prefix-end input state "
        f"(held-out train dist.; U=full, L={LMAX[b]:,}, E1; hatched = prefix-end)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"variant_{b}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------- scatter ----
for b in BEHAVIORS:
    cells = [json.loads(line) for line in open(WT / b / "arm_results/percell/cells.jsonl")]
    target = None
    for c in cells:
        u = json.loads(c["unit_key"])
        if (
            u["variant"] == "context_end"
            and u["regime"] == "e1"
            and u["u_rung_label"] == "full"
            and int(u["budget_l"]) == LMAX[b]
            and u["seed"] == 0
            and u["draw"] == 0
        ):
            target = c
            break
    z = np.load(WT / b / "arm_results/percell/preds" / target["preds_npz"], allow_pickle=True)
    dv = z["dv"]
    fig, axes = plt.subplots(4, 4, figsize=(12, 11))
    for ax, a in zip(axes.flat, ARM_ORDER):
        p = z[f"pred__{a}"]
        ok = np.isfinite(p) & np.isfinite(dv)
        rho = spearmanr(p[ok], dv[ok]).statistic
        idx = np.random.default_rng(0).permutation(ok.sum())[:1500]
        ax.scatter(
            np.asarray(p[ok])[idx],
            np.asarray(dv[ok])[idx],
            s=3,
            alpha=0.25,
            color=ARM_COLOR[a],
            rasterized=True,
        )
        ax.set_title(f"{ARM_LABEL[a]}\nrho={rho:.2f} (n={int(ok.sum())})", fontsize=7.5)
        ax.tick_params(labelsize=6)
        STATS[b].setdefault("scatter_rho", {})[a] = round(float(rho), 3)
    fig.suptitle(
        f"{b}: predicted score vs judged expression, per method "
        f"(held-out train dist., one representative cell: U=full, L={LMAX[b]:,}, E1, "
        f"context-end, seed 0/draw 0; OOF predictions; <=1,500 points shown/panel)",
        y=1.005,
    )
    fig.supylabel(DV_LABEL[b], fontsize=9)
    fig.supxlabel("method prediction (arbitrary scale)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"scatter_{b}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------- evil regimes ----
in_rows = DATA["evil"]["arm_rows"]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
arms_r = ["arm1_ctx_e1", "arm6_map_proj_e1", "arm11_oracle_proj"]
x = np.arange(len(arms_r))
for off, reg in enumerate(["e1", "e2", "e2p"]):
    ms, ss = [], []
    for a in arms_r:
        m, s, _ = agg(
            in_rows,
            a,
            budget_l=LMAX["evil"],
            variant="context_end",
            regime=reg,
            u_rung_label="full",
        )
        ms.append(m)
        ss.append(s)
        STATS["evil"].setdefault(f"regime_{reg}", {})[a] = None if m is None else round(m, 3)
    ax.bar(
        x + (off - 1) * 0.26,
        ms,
        0.24,
        yerr=ss,
        label={
            "e1": "E1 synthetic (paper)",
            "e2": "E2 matched-pair natural",
            "e2p": "E2p pooled natural",
        }[reg],
    )
ax.set_xticks(x)
ax.set_xticklabels([ARM_LABEL[a] for a in arms_r], fontsize=8)
ax.set_ylabel("Spearman rho")
ax.legend(fontsize=7.5)
ax.set_title(
    "evil: persona-vector extraction regime (held-out train dist., U=full, L=8,000)", fontsize=9
)
fig.tight_layout()
fig.savefig(OUT / "regimes_evil.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- prose-ready aggregates ---
for b in BEHAVIORS:
    d = DATA[b]
    hs = d["headlines"]
    deltas = [h["delta_rho_frozen"] for h in hs]
    STATS[b]["headline_all_cells"] = dict(
        n=len(hs),
        median_delta=float(np.median(deltas)),
        ci_below0=sum(1 for h in hs if h["ci_delta_frozen"][1] < 0),
        ci_above0=sum(1 for h in hs if h["ci_delta_frozen"][0] > 0),
    )
    cells = [json.loads(line) for line in open(WT / b / "arm_results/percell/cells.jsonl")]
    op_deltas, ceilings, pmaxs = [], [], []
    for c in cells:
        u = json.loads(c["unit_key"])
        if c.get("split_half"):
            ceilings.append(c["split_half"]["ceiling_sb"])
        if c.get("max_over_arms_null"):
            pmaxs.append(c["max_over_arms_null"]["p_max_over_arms"])
        if (
            u["variant"] == "context_end"
            and u["regime"] == "e1"
            and u["u_rung_label"] == "full"
            and int(u["budget_l"]) == LMAX[b]
        ):
            op_deltas.append(c["headline"]["delta_rho_frozen"])
    STATS[b]["headline_op_slice"] = dict(
        n=len(op_deltas), mean_delta=float(np.mean(op_deltas)) if op_deltas else None
    )
    STATS[b]["splithalf_ceiling_mean"] = float(np.mean(ceilings)) if ceilings else None
    STATS[b]["pmax_frac_sig"] = float(np.mean([p < 0.05 for p in pmaxs])) if pmaxs else None

cb = json.load(open(WT / "evil_config_b/arm_results/all_arms_spearman.json"))
arms_cb = defaultdict(list)
for r in cb["arm_rows"]:
    if r.get("rho_frozen") is not None:
        arms_cb[r["arm"]].append(r["rho_frozen"])
STATS["evil"]["config_b_median"] = {
    a: round(float(np.median(v)), 3) for a, v in sorted(arms_cb.items())
}
STATS["evil"]["config_b_n_cells"] = cb["n_cells"]

json.dump(STATS, open("/tmp/i1739_writeup_stats.json", "w"), indent=1)
print(json.dumps(STATS, indent=1)[:6000])
print("FIGURES:", sorted(p.name for p in OUT.glob("*.png")))
