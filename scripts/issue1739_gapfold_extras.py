"""Fold the #1739 partial-item results (R5 / OOD scatters / map-recon) into figures.

Sibling of ``issue1739_gap_fold.py`` (which closes gap 1 + gap 2). This script
renders the three result families that landed alongside the two gaps:

* ``issue1739_maxood/r5_unjudged_trait_pool`` -- R5: replacing half the generic
  WildChat map pool with UNLABELED trait-eliciting contexts, swept over
  behavior x map-pool size U x label budget L. The clean contrast is
  ``f_u=0.5`` (half the map pool is trait contexts) against the matched
  ``f_u=0.0`` all-generic pool at the SAME U and L. ``arm4_ridge_ctx`` never
  consumes the unlabeled map pool, so its delta is a built-in nuisance control
  (labeled-draw resampling), not a pool effect.
* ``issue1739_maxood/ood_scatter_preds`` -- per-context (prediction, judged DV)
  pairs on the OOD rungs, which the earlier cuts could not plot.
* ``issue1739_maxood/map_recon_evaldist`` -- map reconstruction R^2 + kNN
  retrieval computed ON each behavior evaluation distribution rather than only
  on the WildChat holdout.

Pure aggregation + rendering: no fits, no GPU, no judge calls. Every number is
re-read from the named artifact in-process.
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
from scipy import stats as sps  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

set_paper_style("blog")

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
OUT = ROOT / "figures/issue_1739/gapfold"
OUT.mkdir(parents=True, exist_ok=True)
STAGE = ROOT / "data/issue_1739/gapfold"
STAGE.mkdir(parents=True, exist_ok=True)

REPO = hub.DEFAULT_DATASET_REPO
BEHAVIORS = ["evil", "sycophancy", "hallucination"]
MAXL = {"evil": 8000, "sycophancy": 16000, "hallucination": 16000}
OOD_RUNGS = {
    "evil": ["toxicchat", "hhrt"],
    "sycophancy": ["aita"],
    "hallucination": ["nqopen", "simpleqa"],
}

# One colour = one meaning. Three DISJOINT factors are encoded in this file, so
# each gets its own palette family and none reuses another's pair:
#   * map-pool composition (R5 figure)      -> C_GENERIC / C_AUG
#   * ARM identity (scatter figure)         -> the project arm palette, tab20
#     indexed by position in ARM_ORDER, byte-identical to the indexing in
#     issue1739_gap_fold.py / issue1739_final_fold.py, so an arm keeps the same
#     colour it has in wide_ood_arms.png
#   * evaluation DISTRIBUTION (recon figure) -> a sequential cividis ramp with a
#     grey in-distribution reference; deliberately not categorical, so it can
#     never be mistaken for the arm palette
C_GENERIC = "#4C72B0"  # all-generic WildChat map pool
C_AUG = "#DD8452"  # 50% unlabeled trait-eliciting contexts
C_INDIST = "#444444"  # WildChat holdout (in-distribution) reference

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
_cmap = plt.get_cmap("tab20")
ARM_COLOR = {a: _cmap(i % 20) for i, a in enumerate(ARM_ORDER)}
ARM_LABEL = {
    "arm4_ridge_ctx": "direct ridge on context",
    "arm6_map_proj_e1": "map -> PV projection (label-free)",
    "arm7_map_ridge_pred": "map -> ridge on predicted answers",
}
SCATTER_ARMS = ["arm4_ridge_ctx", "arm6_map_proj_e1", "arm7_map_ridge_pred"]

_civ = plt.get_cmap("cividis")
RUNG_COLOR = {
    r: _civ(v)
    for r, v in zip(["hhrt", "toxicchat", "aita", "nqopen", "simpleqa"], np.linspace(0.05, 0.85, 5))
}

STATS: dict = {}


def fig_title(fig, title, subtitle):
    """Place a suptitle + a smaller subtitle block with INCH-based spacing.

    Fraction-based y positions collide on short figures and waste space on tall
    ones, because a text block's height is fixed in inches while the offset is
    a fraction of the figure. Convert both to inches off the top edge, and
    return the ``rect`` top the caller should hand ``tight_layout``.
    """
    h_in = fig.get_size_inches()[1]
    n_lines = subtitle.count("\n") + 1
    title_in = 0.34
    # 13pt suptitle is ~0.18 in tall and is CENTERED on its y, so clear half its
    # height plus padding before the subtitle block starts.
    sub_top_in = title_in + 0.22
    sub_h_in = 0.125 * n_lines
    fig.suptitle(title, fontsize=13, y=1.0 - title_in / h_in)
    fig.text(
        0.5,
        1.0 - (sub_top_in + sub_h_in) / h_in,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#444444",
    )
    return 1.0 - (sub_top_in + sub_h_in + 0.12) / h_in


def stage(prefix: str) -> Path:
    """Scoped staging of one HF prefix (never a full-repo listing)."""
    dest_dir = STAGE / prefix.replace("/", "__")
    files = hub.stage_hub_prefix(
        repo_id=REPO, prefix=prefix, dest_dir=dest_dir, repo_type="dataset"
    )
    print(f"[stage] {prefix}: {len(files)} files -> {dest_dir}", flush=True)
    mirrored = dest_dir / prefix
    return mirrored if mirrored.exists() else dest_dir


# ============================================================== R5 trait pool ==
r5_dir = stage("issue1739_maxood/r5_unjudged_trait_pool")

R5_ARMS = ["arm6_map_proj_e1", "arm7_map_ridge_pred", "arm4_ridge_ctx"]
# (behavior, arm, U, pool) -> (rho, ci) at that behavior's MAX label budget
r5: dict = {}
r5_us: dict = defaultdict(set)
for beh in BEHAVIORS:
    cells_p = r5_dir / beh / "arm_results/percell/cells.jsonl"
    if not cells_p.exists():
        print(f"[r5] MISSING {cells_p}", flush=True)
        continue
    with open(cells_p) as fh:
        for line in fh:
            cell = json.loads(line)
            uk = json.loads(cell["unit_key"])
            if uk["variant"] != "context_end" or uk["budget_l"] != MAXL[beh]:
                continue
            lab = uk["u_rung_label"]
            if lab == "full":
                continue  # the un-composed reference; the matched control is compose_*_fu0.0
            u_size = int(lab.split("_")[0].replace("compose", ""))
            f_u = uk.get("f_u")
            if f_u == 0.0:
                pool = "generic"
            elif f_u == 0.5:
                pool = "augmented"
            else:
                continue
            r5_us[beh].add(u_size)
            for a in cell["arms"]:
                if a["arm"] in R5_ARMS:
                    r5[(beh, a["arm"], u_size, pool)] = (a["rho_frozen"], a["ci_frozen"])

fig, axes = plt.subplots(len(R5_ARMS), len(BEHAVIORS), figsize=(13.6, 9.4))
for ri, arm in enumerate(R5_ARMS):
    for ci, beh in enumerate(BEHAVIORS):
        ax = axes[ri][ci]
        us = sorted(r5_us[beh])
        x = np.arange(len(us))
        for off, pool, color in ((-0.19, "generic", C_GENERIC), (0.19, "augmented", C_AUG)):
            vals, los, his, xs = [], [], [], []
            for i, u in enumerate(us):
                got = r5.get((beh, arm, u, pool))
                if got is None:
                    continue
                rho, ci_ = got
                vals.append(rho)
                los.append(max(0.0, rho - ci_[0]))
                his.append(max(0.0, ci_[1] - rho))
                xs.append(i + off)
            if not vals:
                continue
            ax.bar(xs, vals, width=0.36, color=color, label=pool)
            ax.errorbar(
                xs,
                vals,
                yerr=np.array([los, his]),
                fmt="none",
                ecolor="#333333",
                capsize=2.5,
                elinewidth=1.0,
            )
            STATS.setdefault("r5", {}).setdefault(beh, {}).setdefault(arm, {})[pool] = {
                str(u): r5[(beh, arm, u, pool)][0] for u in us if (beh, arm, u, pool) in r5
            }
        ax.axhline(0, color="#666666", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"U={u:,}" for u in us], fontsize=8)
        if ci == 0:
            ax.set_ylabel(r"Spearman $\rho$", fontsize=9)
        tag = "  (CONTROL: does not use the map pool)" if arm == "arm4_ridge_ctx" else ""
        ax.set_title(f"{beh} / {ARM_LABEL[arm]}{tag}", fontsize=8.6)
        if ri == 0 and ci == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
_top = fig_title(
    fig,
    "R5: replacing half the unlabeled map pool with trait-eliciting contexts",
    "Held-out train rung, context end state, E1, each behavior at its MAXIMUM label budget "
    "(evil L=8,000; sycophancy / hallucination L=16,000).\nBlue = all-generic WildChat map pool; "
    "orange = same pool size with 50% of the pairs replaced by UNLABELED trait-eliciting "
    "contexts.\nError bars are the 95% bootstrap CI over contexts from each row's ci_frozen. "
    "Bottom row is the built-in control: direct ridge never reads the\nunlabeled map pool, so "
    "any movement there is labeled-draw resampling, not a pool effect.",
)
fig.tight_layout(rect=(0, 0, 1, _top))
savefig_paper(fig, "r5_trait_pool", dir=OUT)
plt.close(fig)
print("[r5] wrote r5_trait_pool", flush=True)

# ================================================================ OOD scatters ==
sc_dir = stage("issue1739_maxood/ood_scatter_preds")

panels = [(b, r) for b in BEHAVIORS for r in OOD_RUNGS[b]]
# (behavior, rung, arm) -> (scores, dvs, budget_l)
pred_pairs: dict = {}
for beh in BEHAVIORS:
    tp_dir = sc_dir / beh / "arm_results/percell/transfer_preds"
    if not tp_dir.exists():
        print(f"[scatter] MISSING {tp_dir}", flush=True)
        continue
    # Keep the LARGEST available label budget per (rung, arm).
    best_L: dict = {}
    acc: dict = defaultdict(lambda: ([], []))
    for p in sorted(tp_dir.glob("*.jsonl")):
        with open(p) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("u_rung_label") != "full" or rec.get("variant") != "context_end":
                    continue
                if rec.get("regime") != "e1" or rec["arm"] not in SCATTER_ARMS:
                    continue
                key = (rec["rung"], rec["arm"])
                L = rec["budget_l"]
                if best_L.get(key, -1) > L:
                    continue
                if best_L.get(key, -1) < L:
                    best_L[key] = L
                    acc[key] = ([], [])
                acc[key][0].append(rec["score"])
                acc[key][1].append(rec["dv"])
    for (rung, arm), (s, d) in acc.items():
        pred_pairs[(beh, rung, arm)] = (np.asarray(s), np.asarray(d), best_L[(rung, arm)])

rng = np.random.default_rng(0)
fig, axes = plt.subplots(len(panels), len(SCATTER_ARMS), figsize=(12.4, 3.0 * len(panels)))
for ri, (beh, rung) in enumerate(panels):
    for ci, arm in enumerate(SCATTER_ARMS):
        ax = axes[ri][ci]
        got = pred_pairs.get((beh, rung, arm))
        if got is None:
            ax.text(
                0.5,
                0.5,
                "not available",
                ha="center",
                va="center",
                fontsize=8,
                color="#999999",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        s, d, L = got
        rho = float(sps.spearmanr(s, d).statistic)
        STATS.setdefault("ood_scatter", {}).setdefault(beh, {}).setdefault(rung, {})[arm] = {
            "rho_recomputed": rho,
            "n": int(s.size),
            "budget_l": int(L),
        }
        idx = rng.choice(s.size, size=min(1500, s.size), replace=False)
        ax.scatter(
            s[idx],
            d[idx],
            s=5,
            alpha=0.28,
            color=ARM_COLOR[arm],
            edgecolors="none",
            rasterized=True,
        )
        ax.text(
            0.03,
            0.95,
            rf"$\rho$ = {rho:+.3f}   n = {s.size:,}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.85),
        )
        ax.set_xlabel(f"predicted score (arm units), L={L:,}", fontsize=7.5)
        if ci == 0:
            ax.set_ylabel("judged expression", fontsize=8)
        ax.set_title(f"{beh} / {rung} / {ARM_LABEL[arm]}", fontsize=8.0)
_top = fig_title(
    fig,
    "Out-of-distribution scatters: predicted score vs judged expression",
    "One point per evaluation context (a random 1,500-point subsample is drawn for legibility; "
    "the annotated Spearman rho and n are computed on ALL points).\nU = 18,793 unlabeled map "
    "pairs, E1, context end state, at the largest label budget the persisted predictions cover "
    "per rung (stated per panel).\nDV construct: evil / sycophancy graded 0-100 trait score; "
    "hallucination 0-1 fabrication rate on its own rungs.",
)
fig.tight_layout(rect=(0, 0, 1, _top))
savefig_paper(fig, "ood_scatters", dir=OUT)
plt.close(fig)
print("[scatter] wrote ood_scatters", flush=True)

# ==================================================== map recon on eval dists ==
mr_dir = stage("issue1739_maxood/map_recon_evaldist")

recon: dict = {}
for beh in ["evil", "sycophancy"]:
    p = mr_dir / beh / "map_diagnostics.json"
    if not p.exists():
        print(f"[recon] MISSING {p}", flush=True)
        continue
    doc = json.loads(p.read_text())
    entry = doc["context_end|full"]
    holdout = entry["per_layer"]
    rec = {
        "n_train": entry["n_train"],
        "n_holdout": entry["n_holdout"],
        "holdout_r2_map": [r["r2_map"] for r in holdout],
        "holdout_r2_identity_bias": [r["r2_identity_bias"] for r in holdout],
        "holdout_knn": {
            "acc1": max(r["knn"]["cosine"]["acc_at_k"]["1"] for r in holdout),
            "chance1": holdout[0]["knn"]["cosine"]["chance_at_k"]["1"],
            "acc5": max(r["knn"]["cosine"]["acc_at_k"]["5"] for r in holdout),
            "chance5": holdout[0]["knn"]["cosine"]["chance_at_k"]["5"],
        },
        "rungs": {},
    }
    for rung, v in entry["eval_rung"]["per_rung"].items():
        pl = v.get("per_layer") or []
        if not pl:
            continue
        best = max(pl, key=lambda r: r["r2_eval_rung"])
        k = best["knn"]["cosine"]
        rec["rungs"][rung] = {
            "n_rows": v["n_rows"],
            "r2_per_layer": [r["r2_eval_rung"] for r in pl],
            "r2_best": best["r2_eval_rung"],
            "r2_best_layer": best["layer_idx"],
            "r2_mean": v["r2_eval_rung_mean"],
            "acc1": max(r["knn"]["cosine"]["acc_at_k"]["1"] for r in pl),
            "chance1": k["chance_at_k"]["1"],
            "acc5": max(r["knn"]["cosine"]["acc_at_k"]["5"] for r in pl),
            "chance5": k["chance_at_k"]["5"],
        }
    recon[beh] = rec
STATS["map_recon_evaldist"] = recon

fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
ax = axes[0]
ref = recon.get("evil") or next(iter(recon.values()), None)
if ref is not None:
    layers = np.arange(len(ref["holdout_r2_map"]))
    ax.plot(
        layers,
        ref["holdout_r2_map"],
        color=C_INDIST,
        lw=2.0,
        label="WildChat holdout (in-distribution)",
    )
    ax.plot(
        layers,
        ref["holdout_r2_identity_bias"],
        color=C_INDIST,
        lw=1.4,
        ls=":",
        label="identity+learned-bias baseline (in-dist.)",
    )
    for beh, rec in recon.items():
        for rung, rv in rec["rungs"].items():
            ax.plot(
                np.arange(len(rv["r2_per_layer"])),
                rv["r2_per_layer"],
                color=RUNG_COLOR[rung],
                lw=1.6,
                label=f"{beh} / {rung} (eval dist.)",
            )
ax.axhline(0, color="#666666", lw=0.9)
ax.set_xlabel("layer", fontsize=9)
ax.set_ylabel(r"reconstruction $R^2$", fontsize=9)
ax.set_title("Map reconstruction $R^2$ per layer", fontsize=9.5)
ax.legend(fontsize=6.8, frameon=False)

for ax, kk, lbl in ((axes[1], "1", "acc@1"), (axes[2], "5", "acc@5")):
    names, vals, chances, colors = [], [], [], []
    if ref is not None:
        names.append("WildChat holdout\n(in-distribution)")
        vals.append(ref["holdout_knn"][f"acc{kk}"])
        chances.append(ref["holdout_knn"][f"chance{kk}"])
        colors.append(C_INDIST)
    for beh, rec in recon.items():
        for rung, rv in rec["rungs"].items():
            names.append(f"{beh}\n{rung}")
            vals.append(rv[f"acc{kk}"])
            chances.append(rv[f"chance{kk}"])
            colors.append(RUNG_COLOR[rung])
    xx = np.arange(len(names))
    ax.bar(xx, vals, color=colors, width=0.62)
    ax.plot(xx, chances, "k_", markersize=22, markeredgewidth=1.8, label="chance rate")
    for i, (v, c) in enumerate(zip(vals, chances)):
        ax.text(i, v * 1.25, f"{v / c:.0f}x", ha="center", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(xx)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel(f"kNN retrieval {lbl} (cosine, log scale)", fontsize=8.5)
    ax.set_title(f"kNN retrieval {lbl} vs chance", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
_top = fig_title(
    fig,
    "Map reconstruction and retrieval ON the behavior evaluation distributions",
    "Map fit on 15,034 WildChat context->answer pairs (U = 18,793 refit), context end state. "
    "Left: reconstruction R^2 per layer, in-distribution vs on each\nevaluation rung. Middle / "
    "right: kNN retrieval of the true target answer state among that rung's own held-out pool, "
    "best layer, cosine metric; the black\ndash is the chance rate (k / n_pool) and the "
    "annotation is the multiple of chance. Retrieval stays far above chance where R^2 is "
    "strongly negative.",
)
fig.tight_layout(rect=(0, 0, 1, _top))
savefig_paper(fig, "map_recon_evaldist", dir=OUT)
plt.close(fig)
print("[recon] wrote map_recon_evaldist", flush=True)

dump = Path("/tmp/i1739_gapfold_extras.json")
dump.write_text(json.dumps(STATS, indent=2, default=str))
print(f"\nwrote {dump}", flush=True)
