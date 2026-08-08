"""Fold the bare-query round into the #1739 interim writeup.

Reads the committed bare-query artifacts (branch issue-1739) and the
already-folded full-context WildChat-rung results, and renders:

  bareq_vs_full.png          per-arm rho, full-context render vs bare-query render
  bareq_null_probe_layers.png the by-construction-null prefix probe across layers
                              (the round's open integrity flag)

Pure aggregation + rendering: no fits, no GPU, no judge calls. Every number is
re-read from the named artifact.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
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
BAREQ = WT / "bareq_map"
WCRUNG = Path("/tmp/i1739_wcrung")  # full-context rung, HF-resident (already folded)
OUT = ROOT / "figures/issue_1739/interim_writeup"
OUT.mkdir(parents=True, exist_ok=True)

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
    "arm6_map_proj_e1": "map -> PV proj.",
    "arm4_ridge_ctx": "direct ridge: context -> expression",
    "arm11_oracle_proj": "oracle: PV proj. on TRUE answer",
    "arm1_ctx_e1": "PV proj. on context (paper method)",
    "arm3_identity_bias": "identity+bias -> PV proj.",
    "arm13_shuffled_map": "control: shuffled map -> PV proj.",
}

# One colour = one RENDER CONDITION, used in every figure of this section.
# Deliberately disjoint from the map-family / regime triplet used elsewhere in
# the writeup (blue/orange/green) so a reader never reads a render bar as a
# map-kind bar.
COND_COLOR = {"full": "#6A51A3", "bare1": "#08847C", "bare2": "#7F3B08"}
COND_LABEL = {
    "full": "full-context render (predictor sees prefix + query)",
    "bare1": "bare-query render, train-fit arms applied",
    "bare2": "bare-query render, dedicated bare fit (evil leg 2)",
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
BQ = {b: json.load(open(BAREQ / b / "all_arms_spearman.json")) for b in BEHAVIORS}
WC = {b: json.load(open(WCRUNG / b / "all_arms_spearman.json")) for b in BEHAVIORS}


def bq_row(b, arm, leg):
    for r in BQ[b]["transfer_rows"]:
        if r["arm"] == arm and str(r["leg"]) == str(leg) and r["variant"] == "context_end":
            return r
    return None


def wc_row(b, arm):
    for r in WC[b]["transfer_rows"]:
        if r["arm"] == arm and r["variant"] == "context_end":
            return r
    return None


# Assemble the comparison table, re-read per cell.
table = {}
for b in BEHAVIORS:
    legs = [str(x) for x in BQ[b]["meta"]["legs_run"]]
    for arm in ARMS:
        entry = {}
        w = wc_row(b, arm)
        if w:
            entry["full"] = dict(
                rho=float(w["rho_frozen"]),
                ci=[float(x) for x in w["ci_frozen"]],
                n=w.get("n_eval"),
                layer=w.get("layer"),
            )
        for leg, key in (("1", "bare1"), ("2", "bare2")):
            if leg not in legs:
                continue
            r = bq_row(b, arm, leg)
            if r and finite(r.get("rho_frozen")):
                entry[key] = dict(
                    rho=float(r["rho_frozen"]),
                    ci=[float(x) for x in r["ci_frozen"]],
                    n=r.get("n_eval"),
                    layer=r.get("layer"),
                    render_match=r.get("render_match"),
                )
        table[f"{b}|{arm}"] = entry
STATS["bare_vs_full"] = table
STATS["meta"] = {
    b: {
        k: BQ[b]["meta"].get(k)
        for k in ("legs_run", "map_kind", "map_source", "draw", "seed", "u_sizes")
    }
    | {
        "render_match_label": (BQ[b]["meta"].get("render_match") or {}).get("label"),
        "render_agrees_with_expected": (BQ[b]["meta"].get("render_match") or {}).get(
            "agrees_with_expected"
        ),
        "ts": BQ[b]["ts"],
    }
    for b in BEHAVIORS
}

# Coverage + the null probe, straight off the percell records.
cov, nullp = {}, {}
for b in BEHAVIORS:
    first = json.loads(next(open(BAREQ / b / "percell/bareq_leg1_transfer.jsonl")))
    c = first["coverage"]
    cov[b] = {k: c[k] for k in c if k not in ("note", "reuse_licence_check")}
    cov[b]["reuse_gate_passed"] = (c.get("reuse_licence_check") or {}).get("passed")
    p = BQ[b]["meta"]["leg1_null_probe"]["context_end"]
    con = p["constancy"]
    rho = np.array(p["rho_per_layer"], dtype=float)
    nullp[b] = dict(
        verdict=p["verdict"],
        n_finite_rho=p["n_finite_rho"],
        any_ci_excludes_zero=p["any_ci_excludes_zero"],
        constant=con.get("constant"),
        passed=con.get("passed"),
        early_cos_min=con.get("early_cos_min"),
        flat_cos_min=con.get("flat_cos_min"),
        max_abs_dev_from_row0=con.get("max_abs_dev_from_row0"),
        n_rows=con.get("n_rows"),
        rho_absmax=float(np.nanmax(np.abs(rho))),
        rho_min=float(np.nanmin(rho)),
        rho_max=float(np.nanmax(rho)),
    )
STATS["coverage"] = cov
STATS["null_probe"] = nullp
STATS["skips"] = {b: BQ[b]["transfer_skips"] for b in BEHAVIORS}
STATS["caveats"] = json.load(open(BAREQ / "bareq_score_done.json")).get("caveats")
STATS["query_bank"] = {
    k: v for k, v in json.load(open(BAREQ / "bareq_queries.json")).items() if k != "queries"
}
ev = BQ["evil"]["meta"]
STATS["evil_leg2"] = {
    "folds": ev.get("leg2_folds"),
    "query_bank": ev.get("leg2_query_bank"),
    "eval_block_notes": ev.get("leg2_eval_block_notes"),
}
STATS["train_prefix_constancy"] = {
    b: {
        k: v
        for k, v in (
            (BQ[b]["meta"].get("render_match") or {}).get("train_prefix_constancy")
            or (BQ[b]["meta"].get("leg2_noop") or {}).get("measured_train_prefix_constancy")
            or {}
        ).items()
        if not isinstance(v, list)
    }
    for b in BEHAVIORS
}

# ------------- FIG 1: per-arm rho, full-context render vs bare-query render ---
fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.9), sharey=False)
for j, b in enumerate(BEHAVIORS):
    ax = axes[j]
    conds = ["full", "bare1"] + (
        ["bare2"] if "2" in [str(x) for x in BQ[b]["meta"]["legs_run"]] else []
    )
    h = 0.8 / len(conds)
    for k, arm in enumerate(ARMS):
        e = table[f"{b}|{arm}"]
        for ci_, cond in enumerate(conds):
            d = e.get(cond)
            if d is None:
                continue
            off = (ci_ - (len(conds) - 1) / 2) * h
            lo, hi = d["ci"]
            e_lo, e_hi = err_offsets(d["rho"], lo, hi)
            ax.barh(
                k + off,
                d["rho"],
                height=h * 0.9,
                color=COND_COLOR[cond],
                label=COND_LABEL[cond] if k == 0 else None,
            )
            ax.errorbar(
                d["rho"],
                k + off,
                xerr=[[e_lo], [e_hi]],
                fmt="none",
                ecolor="#333333",
                capsize=2.0,
                lw=0.9,
            )
    ax.set_yticks(range(len(ARMS)))
    # NOT sharey: a shared y axis lets the LAST panel's empty ticklabels wipe the
    # arm names off every panel (the same trap as the wcrung arms figure).
    ax.set_yticklabels([ARM_LABEL[a] for a in ARMS] if j == 0 else [""] * len(ARMS), fontsize=6.4)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", lw=1.0)
    ax.set_xlabel(r"Spearman $\rho$ vs judged expression")
    n = cov[b]["n_eval_contexts"]
    rm = (BQ[b]["meta"].get("render_match") or {}).get("label")
    set_title_subtitle(ax, b, f"n={n} contexts; leg-1 render {rm}")
# Figure-level legend so the evil-only leg-2 condition is always explained.
handles = [plt.Rectangle((0, 0), 1, 1, color=COND_COLOR[c]) for c in ("full", "bare1", "bare2")]
fig.legend(
    handles,
    [COND_LABEL[c] for c in ("full", "bare1", "bare2")],
    loc="lower center",
    ncol=3,
    fontsize=6.6,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
savefig_paper(fig, "bareq_vs_full", dir=OUT)
plt.close(fig)

# ---------- FIG 2: the by-construction-null prefix probe across all layers ---
# Neutral greys + linestyles, NOT new colours: the factor here is behavior, which
# is panel/linestyle-encoded everywhere else in the writeup.
fig, ax = plt.subplots(figsize=(8.2, 3.8))
for b, ls in zip(BEHAVIORS, ("-", "--", ":")):
    p = BQ[b]["meta"]["leg1_null_probe"]["context_end"]
    rho = np.array(p["rho_per_layer"], dtype=float)
    ax.plot(
        range(len(rho)),
        rho,
        ls=ls,
        lw=1.5,
        color="#333333",
        label=f"{b} (|rho| max {np.nanmax(np.abs(rho)):.3f})",
    )
e2 = bq_row("evil", "arm13_shuffled_map", "2")
if e2:
    lo, hi = e2["ci_frozen"]
    e_lo, e_hi = err_offsets(e2["rho_frozen"], lo, hi)
    ax.errorbar(
        [len(rho) + 1.5],
        [e2["rho_frozen"]],
        yerr=[[e_lo], [e_hi]],
        fmt="o",
        ms=5,
        color=COND_COLOR["bare2"],
        capsize=3,
        label=f"evil leg-2 shuffled-map control ({e2['rho_frozen']:+.3f})",
    )
ax.axhline(0, color="#444444", lw=1.0)
ax.set_xlabel("layer  (rightmost point: evil leg-2 shuffled-map control, not a layer)")
ax.set_ylabel(r"Spearman $\rho$")
set_title_subtitle(
    ax,
    "By-construction-null arms do not read zero",
    "the bare-render prefix rep is verified constant (cosine >= 0.999), so every "
    "point here should be ~chance",
)
ax.legend(loc="upper left", fontsize=6.4)
fig.tight_layout()
savefig_paper(fig, "bareq_null_probe_layers", dir=OUT)
plt.close(fig)

out = Path("/tmp/i1739_bareq_stats.json")
out.write_text(json.dumps(STATS, indent=1, default=str))
print("wrote", out)
print("figures ->", OUT)
for b in BEHAVIORS:
    e = table[f"{b}|arm6_map_proj_e1"]
    parts = " ".join(f"{c}={e[c]['rho']:+.4f}" for c in ("full", "bare1", "bare2") if c in e)
    print(f"  {b:14s} map_proj: {parts}")
    print(
        f"                 null_probe verdict={nullp[b]['verdict']} absmax={nullp[b]['rho_absmax']:.4f}"
    )
