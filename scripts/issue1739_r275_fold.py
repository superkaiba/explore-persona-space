"""Fold the R2.75 query-map L-ladder into the #1739 interim writeup.

Reads the HF-resident ladder artifacts
(`superkaiba1/explore-persona-space-data` :: `issue1739_maxood/r275_query_scaling/
L<budget>/bareq_map/<behavior>/all_arms_spearman.json`) and renders:

  ladder_rho_vs_l.png        rho vs labeled budget L, one panel per behavior,
                             one line per arm, 95% bootstrap CI error bars
  ladder_turn_subsets.png    the pooled / multi-turn / single-turn decomposition
                             at each behavior's maximum budget

Pure aggregation + rendering: no fits, no GPU, no judge calls. Every number is
re-read from the named artifact in this run.

Design note carried from the artifacts' own `meta`: leg 1 FITS NO MAP -- it
APPLIES committed train-fit cells (fit at labeled budget L) to bare-query reps
and scores rho. The map itself is fit on unlabeled pairs, so label-free arms
move across rungs ONLY when the frozen layer (selected on the own train pool at
that budget) is reselected.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)
from explore_persona_space.orchestrate.hub import retry_transient  # noqa: E402

set_paper_style("blog")

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1739_maxood/r275_query_scaling"
CACHE = Path("/tmp/i1739_r275")
FIGDIR = ROOT / "figures/issue_1739/ladder"
STATDIR = ROOT / "eval_results/issue_1739/ladder"
FIGDIR.mkdir(parents=True, exist_ok=True)
STATDIR.mkdir(parents=True, exist_ok=True)

# Realized ladder: evil topped out at 8,000 (its train pool), the other two at 16,000.
BUDGETS = {
    "sycophancy": [250, 2500, 16000],
    "hallucination": [250, 2500, 16000],
    "evil": [250, 2500, 8000],
}
BEHAVIORS = ["sycophancy", "hallucination", "evil"]

ARMS = [
    "arm6_map_proj_e1",
    "arm4_ridge_ctx",
    "arm11_oracle_proj",
    "arm1_ctx_e1",
    "arm3_identity_bias",
    "arm13_shuffled_map",
]
ARM_LABEL = {
    "arm6_map_proj_e1": "map -> PV proj. (label-free map)",
    "arm4_ridge_ctx": "direct ridge: labels -> expression",
    "arm11_oracle_proj": "oracle: PV proj. on TRUE answer",
    "arm1_ctx_e1": "PV proj. on bare query (paper method)",
    "arm3_identity_bias": "identity+bias -> PV proj.",
    "arm13_shuffled_map": "control: shuffled map -> PV proj.",
}
# One colour = one ARM, held across every panel of this section.
ARM_COLOR = {
    "arm6_map_proj_e1": "#0072B2",
    "arm4_ridge_ctx": "#D55E00",
    "arm11_oracle_proj": "#009E73",
    "arm1_ctx_e1": "#CC79A7",
    "arm3_identity_bias": "#7A7A7A",
    "arm13_shuffled_map": "#000000",
}
ARM_STYLE = {
    "arm6_map_proj_e1": "-",
    "arm4_ridge_ctx": "-",
    "arm11_oracle_proj": "--",
    "arm1_ctx_e1": "-",
    "arm3_identity_bias": ":",
    "arm13_shuffled_map": ":",
}
# Turn subsets are encoded on the X AXIS, never by colour: colour means ARM and only
# ARM in every figure of this section (and is disjoint from the render-condition
# palette used by the bare-query figures elsewhere in the writeup).
SUBSET_ORDER = ["pooled", "multi_turn_only", "single_turn_only"]
SUBSET_TICK = {
    "pooled": "pooled\n(all contexts)",
    "multi_turn_only": "multi-turn only\n(render changed)",
    "single_turn_only": "single-turn only\n(bare == original)",
}

STATS: dict = {}


def finite(v):
    return v is not None and np.isfinite(v)


def err_offsets(v, lo, hi):
    """Non-negative error-bar offsets from an absolute CI (never raw bounds)."""
    if not (finite(v) and finite(lo) and finite(hi)):
        return (0.0, 0.0)
    return (max(0.0, v - lo), max(0.0, hi - v))


# ------------------------------------------------------------------- load ----
def fetch(behavior: str, budget: int, name: str) -> Path:
    """Download one ladder artifact, caching under /tmp."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"L{budget}__{behavior}__{name}"
    if not dest.exists():
        path_in_repo = f"{PREFIX}/L{budget}/bareq_map/{behavior}/{name}"
        src = retry_transient(
            lambda: hf_hub_download(REPO, path_in_repo, repo_type="dataset"),
            what=f"hf_hub_download {path_in_repo}",
        )
        shutil.copy(src, dest)
    return dest


SUMMARY = {
    (b, L): json.load(open(fetch(b, L, "all_arms_spearman.json")))
    for b in BEHAVIORS
    for L in BUDGETS[b]
}


def rows(behavior, budget, *, leg, variant, subset):
    out = {}
    for r in SUMMARY[(behavior, budget)]["transfer_rows"]:
        if str(r["leg"]) == str(leg) and r["variant"] == variant and r["subset"] == subset:
            out[r["arm"]] = r
    return out


# ----------------------------------------------------- ladder (leg 1, pooled) --
ladder: dict = {}
for b in BEHAVIORS:
    ladder[b] = {}
    for arm in ARMS:
        pts = []
        for L in BUDGETS[b]:
            r = rows(b, L, leg="1", variant="context_end", subset="pooled").get(arm)
            if r is None or not finite(r.get("rho_frozen")):
                continue
            pts.append(
                dict(
                    budget_l=int(r["budget_l"]),
                    rho=float(r["rho_frozen"]),
                    ci=[float(x) for x in r["ci_frozen"]],
                    layer=int(r["layer"]),
                    n_eval=int(r["n_eval"]),
                )
            )
        ladder[b][arm] = pts
STATS["ladder_leg1_pooled_context_end"] = ladder

# ------------------------------------- evil leg 2 (dedicated bare fit, fixed L) --
evil_leg2 = {}
for L in BUDGETS["evil"]:
    for arm, r in rows("evil", L, leg="2", variant="context_end", subset="pooled").items():
        evil_leg2.setdefault(arm, []).append(
            dict(
                ladder_rung_l=L,
                budget_l_of_row=int(r["budget_l"]),
                rho=float(r["rho_frozen"]),
                ci=[float(x) for x in r["ci_frozen"]],
                layer=int(r["layer"]),
                n_eval=int(r["n_eval"]),
            )
        )
STATS["evil_leg2_pooled_context_end"] = evil_leg2
# Is leg 2 invariant to the ladder rung? (it should be: its own budget_l is fixed)
STATS["evil_leg2_invariant_across_rungs"] = {
    arm: {
        "distinct_rho": sorted({round(p["rho"], 12) for p in pts}),
        "distinct_row_budget_l": sorted({p["budget_l_of_row"] for p in pts}),
        "invariant": len({round(p["rho"], 12) for p in pts}) == 1,
    }
    for arm, pts in evil_leg2.items()
}

# --------------------------------------------- turn subsets at max budget -----
subsets: dict = {}
for b in BEHAVIORS:
    Lmax = max(BUDGETS[b])
    subsets[b] = {"budget_l": Lmax, "arms": {}}
    for sub in ("pooled", "multi_turn_only", "single_turn_only"):
        rs = rows(b, Lmax, leg="1", variant="context_end", subset=sub)
        for arm, r in rs.items():
            subsets[b]["arms"].setdefault(arm, {})[sub] = dict(
                rho=float(r["rho_frozen"]),
                ci=[float(x) for x in r["ci_frozen"]],
                layer=int(r["layer"]),
                n_eval=int(r["n_eval"]),
            )
STATS["turn_subsets_at_max_budget"] = subsets

# ------------------------------------------------------------- provenance ----
STATS["meta"] = {}
for b in BEHAVIORS:
    Lmax = max(BUDGETS[b])
    m = SUMMARY[(b, Lmax)]["meta"]
    STATS["meta"][b] = {
        "budgets_run": BUDGETS[b],
        "legs_run": m.get("legs_run"),
        "map_kind": m.get("map_kind"),
        "map_source": m.get("map_source"),
        "u_sizes": m.get("u_sizes"),
        "draw": m.get("draw"),
        "seed": m.get("seed"),
        "frozen_layer_source": m.get("frozen_layer_source"),
        "render_agrees_with_expected": (m.get("render_match") or {}).get("agrees_with_expected"),
        "subsets_emitted": m.get("subsets_emitted"),
        "preds_persisted_files": (m.get("preds_persisted") or {}).get("files"),
        "mapping_baselines_leg1_reason": ((m.get("mapping_baselines") or {}).get("leg1") or {}).get(
            "reason"
        ),
        "ts": SUMMARY[(b, Lmax)]["ts"],
    }

# Null probe verdict at every rung (the round's standing integrity flag).
STATS["null_probe_by_rung"] = {
    b: {
        f"L{L}": {
            k: SUMMARY[(b, L)]["meta"]["leg1_null_probe"]["context_end"].get(k)
            for k in ("verdict", "n_finite_rho", "any_ci_excludes_zero")
        }
        | {
            "constant": SUMMARY[(b, L)]["meta"]["leg1_null_probe"]["context_end"]["constancy"].get(
                "constant"
            )
        }
        for L in BUDGETS[b]
    }
    for b in BEHAVIORS
}

sd = json.load(open(CACHE / "L16000__score_done.json"))
STATS["score_done_L16000"] = {
    k: sd.get(k) for k in ("git_commit", "judge_called", "wall_s", "ts", "env_versions", "caveats")
}

# ============================================================ figure 1 ========
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), sharey=True)
for ax, b in zip(axes, BEHAVIORS, strict=True):
    for arm in ARMS:
        pts = ladder[b][arm]
        if not pts:
            continue
        xs = np.array([p["budget_l"] for p in pts], dtype=float)
        ys = np.array([p["rho"] for p in pts], dtype=float)
        lo = np.array([err_offsets(p["rho"], *p["ci"])[0] for p in pts])
        hi = np.array([err_offsets(p["rho"], *p["ci"])[1] for p in pts])
        ax.errorbar(
            xs,
            ys,
            yerr=np.vstack([lo, hi]),
            marker="o",
            ms=5,
            lw=1.8,
            capsize=3,
            color=ARM_COLOR[arm],
            linestyle=ARM_STYLE[arm],
            label=ARM_LABEL[arm],
        )
    ax.axhline(0.0, color="#999999", lw=1.0, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS[b])
    ax.set_xticklabels([f"{L:,}" for L in BUDGETS[b]])
    ax.minorticks_off()
    ax.set_xlabel("labeled budget L (rows), log scale")
    n_eval = ladder[b]["arm6_map_proj_e1"][0]["n_eval"]
    set_title_subtitle(ax, b, f"n = {n_eval:,} eval contexts (WildChat rung)")
axes[0].set_ylabel(r"Spearman $\rho$ vs judged DV (frozen layer)")
axes[0].legend(loc="upper left", fontsize=8, framealpha=0.92)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.suptitle(
    "Query -> answer map: does the bare-query readout improve with more labels?",
    y=0.985,
    fontsize=13,
)
paths1 = savefig_paper(fig, "ladder_rho_vs_l", dir=FIGDIR)
plt.close(fig)

# ============================================================ figure 2 ========
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
PLOT_ARMS = ["arm6_map_proj_e1", "arm4_ridge_ctx", "arm11_oracle_proj"]
for ax, b in zip(axes, BEHAVIORS, strict=True):
    Lmax = subsets[b]["budget_l"]
    width = 0.26
    xbase = np.arange(len(SUBSET_ORDER), dtype=float)
    for k, arm in enumerate(PLOT_ARMS):
        vals, los, his = [], [], []
        for sub in SUBSET_ORDER:
            e = subsets[b]["arms"].get(arm, {}).get(sub)
            vals.append(e["rho"] if e else np.nan)
            o = err_offsets(e["rho"], *e["ci"]) if e else (0.0, 0.0)
            los.append(o[0])
            his.append(o[1])
        ax.bar(
            xbase + (k - 1) * width,
            vals,
            width,
            yerr=np.vstack([los, his]),
            capsize=3,
            color=ARM_COLOR[arm],
            label=ARM_LABEL[arm] if b == BEHAVIORS[0] else None,
        )
    ax.axhline(0.0, color="#333333", lw=1.0)
    ax.set_xticks(xbase)
    ax.set_xticklabels([SUBSET_TICK[s] for s in SUBSET_ORDER], fontsize=8.5)
    n_m = subsets[b]["arms"]["arm6_map_proj_e1"]["multi_turn_only"]["n_eval"]
    n_s = subsets[b]["arms"]["arm6_map_proj_e1"]["single_turn_only"]["n_eval"]
    set_title_subtitle(ax, b, f"L = {Lmax:,}; multi-turn n = {n_m:,}, single-turn n = {n_s:,}")
axes[0].set_ylabel(r"Spearman $\rho$ vs judged DV (frozen layer)")
axes[0].legend(loc="lower left", fontsize=8, framealpha=0.92)
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.suptitle(
    "Bare-query readout decomposed by turn count, at each behavior's maximum budget",
    y=0.982,
    fontsize=13,
)
paths2 = savefig_paper(fig, "ladder_turn_subsets", dir=FIGDIR)
plt.close(fig)

STATS["figures"] = {
    "ladder_rho_vs_l": str(paths1.get("png")),
    "ladder_turn_subsets": str(paths2.get("png")),
}
STATS["provenance"] = {
    "hf_repo": REPO,
    "hf_prefix": PREFIX,
    "script": "scripts/issue1739_r275_fold.py",
    "note": "aggregation + rendering only; no fits, no GPU, no judge calls",
}

out = STATDIR / "r275_query_scaling_stats.json"
out.write_text(json.dumps(STATS, indent=1, sort_keys=False) + "\n")
print("wrote", out)
print("figures:", paths1, paths2)
