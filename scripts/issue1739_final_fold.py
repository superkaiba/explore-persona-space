"""Fold the completed report grid-fill into the #1739 interim writeup.

Reads the artifacts committed on branch ``issue-1739`` (wide arm roster on the
WildChat + persona-vectors-synthetic rungs, the wide OOD transfer grid, the
whitened naturalistic-PV regime tables, and the v2 bare-query round) and
renders:

  wide_roster_arms.png       10-arm roster on the random-WildChat rung
  pvsynth_polarity.png       PV-suite rho decomposed pooled / within-polarity
  wide_ood_arms.png          OOD operating slice with arms 7/8/12 added
  natpv_whitened_vs_raw.png  whitened (primary) vs raw (deprecated) regime reads
  spread_grid.png            DV spread per behavior x evaluation setting
  bareq_v2_resolutions.png   evil turn-subsets, shuffle-seed band, map baselines

Pure aggregation + rendering: no fits, no GPU, no judge calls. Every number is
re-read from the named artifact in-process.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import statistics  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

set_paper_style("blog")

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
WT = ROOT / ".claude/worktrees/issue-1739/eval_results/issue_1739"
OUT = ROOT / "figures/issue_1739/interim_writeup"
OUT.mkdir(parents=True, exist_ok=True)

BEHAVIORS = ["evil", "sycophancy", "hallucination"]
MAXL = {"evil": 8000, "sycophancy": 16000, "hallucination": 16000}

# ---- palettes -------------------------------------------------------------
# One colour = one ARM, shared with the earlier wcrung/scaling figures in this
# writeup (same tab20 indexing over the same ARM_ORDER), so an arm keeps its
# colour across every arm-factor figure here.
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
    "arm1_ctx_e1": "PV proj. on context (paper method)",
    "arm2_ctx_native": "context-native direction proj. (label-supervised)",
    "arm3_identity_bias": "identity+bias -> PV proj.",
    "arm4_ridge_ctx": "direct ridge: context -> expression",
    "arm5_mlp_ctx": "direct MLP: context -> expression",
    "arm6_map_proj_e1": "map -> PV proj.",
    "arm7_map_ridge_pred": "map -> ridge on predicted answers",
    "arm8_map_ridge_true": "map -> ridge trained on real answers",
    "arm11_oracle_proj": "oracle: PV proj. on TRUE answer",
    "arm12_oracle_reg": "oracle: ridge on TRUE answer",
    "arm13_shuffled_map": "control: shuffled map -> PV proj.",
}
# One colour = one POLARITY SUBSET (pvsynth figure only). Deliberately disjoint
# from the arm, regime, render and map-family palettes used elsewhere.
POL_COLOR = {"pooled": "#111111", "elicit": "#C2185B", "non_elicit": "#00838F"}
POL_LABEL = {
    "pooled": "pooled (both instruction polarities)",
    "elicit": "within positive-instruction half only",
    "non_elicit": "within negative-instruction half only",
}
# One colour = one extraction regime, unchanged from the earlier regime figure.
REGIME_COLOR = {"e1": "#4C72B0", "e2": "#DD8452", "e2p": "#55A868"}
REGIME_LABEL = {
    "e1": "E1 paper-faithful synthetic",
    "e2": "E2 matched-pair natural",
    "e2p": "E2p pooled natural",
}
# One colour = one behavior (spread figure only).
BEH_COLOR = {"evil": "#B2182B", "sycophancy": "#2166AC", "hallucination": "#762A83"}


def fig_title(fig, title, subtitle):
    """Figure-level lede + subtitle (the axes-level helper takes an Axes)."""
    fig.suptitle(
        title,
        x=0.012,
        y=0.985,
        ha="left",
        va="top",
        fontsize=13.5,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(0.012, 0.955, subtitle, ha="left", va="top", fontsize=8.6, color="#5A5A5A")


STATS: dict = {}


def finite(v):
    return v is not None and isinstance(v, (int, float)) and np.isfinite(v)


def err_offsets(v, lo, hi):
    """Non-negative error-bar offsets from an absolute CI (never raw bounds)."""
    if not (finite(v) and finite(lo) and finite(hi)):
        return (0.0, 0.0)
    return (max(0.0, v - lo), max(0.0, hi - v))


def load(p):
    with open(p) as fh:
        return json.load(fh)


# ============================================================ 1. wide roster ==
WIDE_WC = {b: load(WT / "wide/wildchat_rung" / b / "all_arms_spearman.json") for b in BEHAVIORS}
WIDE_PV = {b: load(WT / "wide/pvsynth" / b / "all_arms_spearman.json") for b in BEHAVIORS}

ROSTER = [
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm4_ridge_ctx",
    "arm5_mlp_ctx",
    "arm1_ctx_e1",
    "arm3_identity_bias",
    "arm11_oracle_proj",
    "arm12_oracle_reg",
    "arm13_shuffled_map",
]
NEW_ARMS = {"arm5_mlp_ctx", "arm7_map_ridge_pred", "arm8_map_ridge_true", "arm12_oracle_reg"}


def wide_row(d, arm, variant, rows_key="transfer_rows", **extra):
    for r in d[rows_key]:
        if r["arm"] == arm and r["variant"] == variant:
            if all(r.get(k) == v for k, v in extra.items()):
                return r
    return None


STATS["wide_wcrung"] = {}
fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5), sharey="row")
for col, beh in enumerate(BEHAVIORS):
    d = WIDE_WC[beh]
    for row, variant in enumerate(["context_end", "prefix_end"]):
        ax = axes[row][col]
        ys, vals, los, his, cols, edges = [], [], [], [], [], []
        for arm in ROSTER:
            r = wide_row(d, arm, variant)
            if r is None or not finite(r.get("rho_frozen")):
                continue
            ci = r.get("ci_frozen") or [None, None]
            lo, hi = err_offsets(r["rho_frozen"], ci[0], ci[1])
            ys.append(f"{ARM_LABEL[arm]}  (L{r['layer']})" + ("  *new" if arm in NEW_ARMS else ""))
            vals.append(r["rho_frozen"])
            los.append(lo)
            his.append(hi)
            cols.append(ARM_COLOR[arm])
            edges.append("black" if arm in NEW_ARMS else "none")
            STATS["wide_wcrung"].setdefault(beh, {}).setdefault(variant, {})[arm] = {
                "rho": r["rho_frozen"],
                "ci": ci,
                "layer": r["layer"],
                "n": r["n_eval"],
            }
        y = np.arange(len(ys))
        ax.barh(y, vals, color=cols, edgecolor=edges, linewidth=1.1, height=0.72)
        ax.errorbar(
            vals,
            y,
            xerr=np.array([los, his]),
            fmt="none",
            ecolor="#333333",
            capsize=2.5,
            elinewidth=1.0,
        )
        ax.axvline(0, color="#666666", lw=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(ys, fontsize=7.5)
        ax.invert_yaxis()
        n = d["meta"]["n_contexts"]
        ax.set_title(f"{beh} — {variant.replace('_', ' ')} (n={n})", fontsize=10)
        if row == 1:
            ax.set_xlabel(r"Spearman $\rho$ vs judged expression")
fig_title(
    fig,
    "Wide arm roster on the random held-out WildChat rung",
    "Bars outlined in black are the four arms added by this round (direct MLP, map->ridge on "
    "predicted answers,\nmap->ridge on real answers, oracle ridge). Error bars are 95% bootstrap "
    "CIs over contexts, drawn as non-negative offsets.",
)
fig.tight_layout(rect=(0, 0, 1, 0.90))
savefig_paper(fig, "wide_roster_arms", dir=OUT)
plt.close(fig)
print("wrote wide_roster_arms")

# ====================================================== 2. pvsynth polarity ==
STATS["pvsynth_polarity"] = {}
fig, axes = plt.subplots(1, 3, figsize=(17, 6.4), sharey=True)
subsets = ["pooled", "elicit", "non_elicit"]
for col, beh in enumerate(BEHAVIORS):
    ax = axes[col]
    d = WIDE_PV[beh]
    pol = {
        (r["arm"], r["polarity_subset"]): r
        for r in d["transfer_polarity_rows"]
        if r["variant"] == "context_end"
    }
    arms = [a for a in ROSTER if (a, "pooled") in pol]
    y = np.arange(len(arms))
    h = 0.26
    for j, s in enumerate(subsets):
        vals, ok = [], []
        for a in arms:
            r = pol.get((a, s))
            v = r["rho_frozen"] if r else None
            vals.append(v if finite(v) else 0.0)
            ok.append(finite(v))
            STATS["pvsynth_polarity"].setdefault(beh, {}).setdefault(a, {})[s] = (
                v if finite(v) else None
            )
        ax.barh(
            y + (1 - j) * h,
            vals,
            height=h,
            color=POL_COLOR[s],
            label=POL_LABEL[s] if col == 0 else None,
        )
        for i, good in enumerate(ok):
            if not good:
                ax.text(
                    0.02,
                    y[i] + (1 - j) * h,
                    "undefined (zero DV variance)",
                    va="center",
                    fontsize=6.8,
                    color=POL_COLOR[s],
                )
    ax.axvline(0, color="#666666", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([ARM_LABEL[a] for a in arms], fontsize=8)
    ax.invert_yaxis()
    sp = load(WT / "pvsynth/spread" / f"{beh}.json")["spread"]
    ax.set_title(f"{beh} (n=200; DV mean {sp['mean']:.1f}, sd {sp['sd']:.1f})", fontsize=10)
    ax.set_xlabel(r"Spearman $\rho$ vs judged expression")
fig.legend(loc="lower center", ncol=3, fontsize=8.6, frameon=False, bbox_to_anchor=(0.5, 0.005))
fig_title(
    fig,
    "The persona-vectors synthetic suite's correlations are mostly the positive-vs-negative "
    "instruction contrast",
    "Each arm scored three ways on the SAME 200 suite contexts: pooled over both instruction "
    "polarities, then within each\n100-context half alone. Every arm's pooled value collapses "
    "when the polarity contrast is removed. Evil's negative half\nhas zero DV variance (all 100 "
    "contexts score 0), so within-half rho is undefined there.",
)
fig.tight_layout(rect=(0, 0.07, 1, 0.86))
savefig_paper(fig, "pvsynth_polarity", dir=OUT)
plt.close(fig)
print("wrote pvsynth_polarity")

# =========================================================== 3. wide OOD grid ==
OOD_RUNGS = {
    "evil": ["toxicchat", "hhrt"],
    "sycophancy": ["aita"],
    "hallucination": ["nqopen", "simpleqa"],
}
STATS["wide_ood"] = {}
ood_cells = defaultdict(list)
for beh in BEHAVIORS:
    rows = []
    with open(WT / "wide_ood" / f"{beh}_transfer.jsonl") as fh:
        for line in fh:
            rows.extend(json.loads(line).get("rows", []))
    for r in rows:
        if (
            r["regime"] == "e1"
            and r["u_rung_label"] == "full"
            and r["budget_l"] == MAXL[beh]
            and r["variant"] == "context_end"
            and finite(r.get("rho_frozen"))
        ):
            ood_cells[(beh, r["eval_rung"], r["arm"])].append(r["rho_frozen"])

panels = [(b, r) for b in BEHAVIORS for r in OOD_RUNGS[b]]
# FIXED arm slots across every panel: the panels share a y axis, so a per-panel
# arm list would silently misalign the tick labels where a panel lacks an arm.
OOD_SLOTS = [a for a in ROSTER if any((b, r, a) in ood_cells for b, r in panels)]
fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 6.4), sharey=True)
for ax, (beh, rung) in zip(np.atleast_1d(axes), panels):
    y = np.arange(len(OOD_SLOTS))
    for i, a in enumerate(OOD_SLOTS):
        v = ood_cells.get((beh, rung, a))
        if not v:
            ax.text(0.0, i, "  not run in this cell", va="center", fontsize=6.4, color="#999999")
            continue
        m = statistics.mean(v)
        sd = statistics.pstdev(v) if len(v) > 1 else 0.0
        STATS["wide_ood"].setdefault(beh, {}).setdefault(rung, {})[a] = {
            "mean_rho": m,
            "sd": sd,
            "n_reps": len(v),
        }
        ax.barh(
            i,
            m,
            color=ARM_COLOR[a],
            edgecolor="black" if a in NEW_ARMS else "none",
            linewidth=1.1,
            height=0.72,
        )
        ax.errorbar(
            [m],
            [i],
            xerr=np.array([[max(0.0, sd)], [max(0.0, sd)]]),
            fmt="none",
            ecolor="#333333",
            capsize=2.5,
            elinewidth=1.0,
        )
    ax.axvline(0, color="#666666", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [ARM_LABEL[a] + ("  *new" if a in NEW_ARMS else "") for a in OOD_SLOTS], fontsize=7.5
    )
    ax.invert_yaxis()
    floor = "  (DV floor-censored)" if (beh, rung) == ("evil", "hhrt") else ""
    ax.set_title(f"{beh} / {rung}{floor}", fontsize=10)
    ax.set_xlabel(r"Spearman $\rho$")
fig_title(
    fig,
    "Out-of-distribution transfer with the wide arm roster",
    "Operating slice: E1 persona vector, U = 18,793 unlabeled map pairs, maximum label budget, "
    "context end state.\nBars are the mean over (seed, draw) replicates; error bars are the SD "
    "across those replicates, drawn as non-negative\noffsets. Black outlines mark arms added by "
    "this round. The direct-MLP arm was not run on the OOD grid.",
)
fig.tight_layout(rect=(0, 0, 1, 0.87))
savefig_paper(fig, "wide_ood_arms", dir=OUT)
plt.close(fig)
print("wrote wide_ood_arms")

# ================================================= 4. whitened natural-PV ==
READS = ["ctx", "map_ctx", "oracle"]
READ_LABEL = {
    "ctx": "PV proj. on context",
    "map_ctx": "map(context) -> PV proj.",
    "oracle": "PV proj. on TRUE answer",
}
NATPV = {
    b: {
        "wh": load(WT / "nat_pv_regimes" / b / "regime_comparison_whitened.json"),
        "raw": load(WT / "nat_pv_regimes" / b / "regime_comparison.json"),
    }
    for b in ["sycophancy", "hallucination"]
}
OOS = {"sycophancy": ["aita"], "hallucination": ["nqopen", "simpleqa"]}
np_panels = [(b, r) for b in ["sycophancy", "hallucination"] for r in OOS[b]]
STATS["natpv_whitened"] = {}
fig, axes = plt.subplots(1, len(np_panels), figsize=(4.6 * len(np_panels), 6.0), sharey=True)
for ax, (beh, rung) in zip(np.atleast_1d(axes), np_panels):
    y = np.arange(len(READS))
    h = 0.26
    for j, reg in enumerate(["e1", "e2", "e2p"]):
        wv, rv = [], []
        for rd in READS:
            cw = NATPV[beh]["wh"]["regime_table"][rd][reg]["rho_at_frozen_layer"].get(rung)
            cr = NATPV[beh]["raw"]["regime_table"][rd][reg]["rho_at_frozen_layer"].get(rung)
            wv.append(cw if finite(cw) else 0.0)
            rv.append(cr if finite(cr) else 0.0)
            STATS["natpv_whitened"].setdefault(beh, {}).setdefault(rung, {}).setdefault(rd, {})[
                reg
            ] = {
                "whitened": cw,
                "raw": cr,
                "layer": NATPV[beh]["wh"]["regime_table"][rd][reg]["frozen_layer"],
            }
        ax.barh(
            y + (1 - j) * h,
            wv,
            height=h,
            color=REGIME_COLOR[reg],
            label=REGIME_LABEL[reg] if ax is np.atleast_1d(axes)[0] else None,
        )
        ax.barh(
            y + (1 - j) * h,
            rv,
            height=h,
            facecolor="none",
            edgecolor=REGIME_COLOR[reg],
            linewidth=1.0,
            linestyle=":",
        )
    ax.axvline(0, color="#666666", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([READ_LABEL[r] for r in READS], fontsize=9)
    ax.invert_yaxis()
    n = NATPV[beh]["wh"]["n_contexts_by_rung"][rung]
    ax.set_title(f"{beh} / {rung} (out-of-sample, n={n})", fontsize=10)
    ax.set_xlabel(r"Spearman $\rho$ at the frozen layer")
from matplotlib.patches import Patch  # noqa: E402

_handles = [Patch(facecolor=REGIME_COLOR[r], label=REGIME_LABEL[r]) for r in ("e1", "e2", "e2p")]
_handles.append(
    Patch(
        facecolor="none",
        edgecolor="#555555",
        linewidth=1.2,
        linestyle=(0, (1, 1)),
        label="dotted outline = same read, raw space (deprecated)",
    )
)
np.atleast_1d(axes)[0].legend(handles=_handles, loc="lower right", fontsize=7.2, framealpha=0.95)
fig_title(
    fig,
    "Persona-vector extraction regimes, whitened space (primary) vs raw space (deprecated)",
    "Filled bars are the whitened reads — the space the fitted map and the main grid actually "
    "live in. Dotted outlines are the\nsuperseded raw-space reads, which applied a "
    "whitened-fit map to un-whitened activations. Only out-of-sample rungs are\nshown; the "
    "frozen layer is picked by max |rho| on the train rung, so the sign is not pinned by "
    "selection.",
)
fig.tight_layout(rect=(0, 0, 1, 0.86))
savefig_paper(fig, "natpv_whitened_vs_raw", dir=OUT)
plt.close(fig)
print("wrote natpv_whitened_vs_raw")

# ================================================================ 5. spread ==
# Pre-registered gate 2 (plan section "Pre-registered spread floor + fallback";
# gates.gate2_spread_floor): inter-context SD >= 10 on 0-100 AND < 80% of
# contexts in the bottom [0, 10) bin. Both conditions must hold.
GATE2_SD_FLOOR = 10.0
GATE2_BOTTOM_BIN_EDGE = 10.0
GATE2_BOTTOM_FRAC_MAX = 0.80

SPREAD_ROWS = []


def _row(beh, rung, vals, construct, scale_mult):
    """One spread row on the gate's 0-100 scale (fabrication rates x100)."""
    v = np.asarray(vals, dtype=float) * scale_mult
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    bottom = float((v < GATE2_BOTTOM_BIN_EDGE).mean()) if v.size else 1.0
    return {
        "behavior": beh,
        "rung": rung,
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else 0.0,
        "sd": sd,
        "bottom_frac": bottom,
        "dv_construct": construct,
        "rescaled_x100": scale_mult != 1.0,
        "sd_ok": sd >= GATE2_SD_FLOOR,
        "bottom_ok": bottom < GATE2_BOTTOM_FRAC_MAX,
        "gate2": "PASS" if (sd >= GATE2_SD_FLOOR and bottom < GATE2_BOTTOM_FRAC_MAX) else "FAIL",
    }


for beh in BEHAVIORS:
    lab = load(WT / "dv_dataset" / beh / "labeling.json")
    by = defaultdict(list)
    for r in lab["rows"]:
        if r.get("dv") is not None:
            by[r.get("rung")].append(r["dv"])
    is_rate = beh == "hallucination"
    for rung, v in sorted(by.items()):
        SPREAD_ROWS.append(
            _row(
                beh,
                rung,
                v,
                "fabrication_rate_0_1" if is_rate else "trait_rubric_graded_0_100",
                100.0 if is_rate else 1.0,
            )
        )
    for extra in ("wildchat_rung", "pvsynth"):
        s = load(WT / extra / "spread" / f"{beh}.json")["spread"]
        # These summaries carry sd (population) + the histogram, not raw values;
        # recompute the gate from the histogram's bottom bin and the stored sd.
        hist = load(WT / extra / "spread" / f"{beh}.json")["spread"].get("histogram", {})
        bottom = (hist.get("0-10", 0) / s["n"]) if (hist and s["n"]) else float("nan")
        SPREAD_ROWS.append(
            {
                "behavior": beh,
                "rung": extra,
                "n": s["n"],
                "mean": s["mean"],
                "sd": s["sd"],
                "bottom_frac": bottom,
                "dv_construct": load(WT / extra / "spread" / f"{beh}.json")["dv_construct"],
                "rescaled_x100": False,
                "sd_ok": s["sd"] >= GATE2_SD_FLOOR,
                "bottom_ok": (bottom < GATE2_BOTTOM_FRAC_MAX) if bottom == bottom else None,
                "gate2": (
                    "PASS"
                    if (
                        s["sd"] >= GATE2_SD_FLOOR
                        and bottom == bottom
                        and bottom < GATE2_BOTTOM_FRAC_MAX
                    )
                    else "FAIL"
                ),
            }
        )
STATS["spread_grid"] = SPREAD_ROWS
STATS["spread_gate_pooled_per_behavior"] = {}
for beh in BEHAVIORS:
    _lab = load(WT / "dv_dataset" / beh / "labeling.json")
    _v = np.array([r["dv"] for r in _lab["rows"] if r.get("dv") is not None], dtype=float) * (
        100.0 if beh == "hallucination" else 1.0
    )
    _sd = float(_v.std(ddof=1))
    _bot = float((_v < GATE2_BOTTOM_BIN_EDGE).mean())
    STATS["spread_gate_pooled_per_behavior"][beh] = {
        "n": int(_v.size),
        "sd": _sd,
        "bottom_frac": _bot,
        "gate2": "PASS" if (_sd >= GATE2_SD_FLOOR and _bot < GATE2_BOTTOM_FRAC_MAX) else "FAIL",
    }
STATS["spread_gate"] = {
    "sd_floor": GATE2_SD_FLOOR,
    "bottom_bin_edge": GATE2_BOTTOM_BIN_EDGE,
    "bottom_frac_max": GATE2_BOTTOM_FRAC_MAX,
}

fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.6), sharex="col")
for col, beh in enumerate(BEHAVIORS):
    rows = [r for r in SPREAD_ROWS if r["behavior"] == beh]
    xs = [f"{r['rung']}\nn={r['n']}" for r in rows]
    x = np.arange(len(rows))
    # -- top: inter-context SD vs the SD >= 10 floor
    ax = axes[0][col]
    bars = ax.bar(x, [r["sd"] for r in rows], color=BEH_COLOR[beh], width=0.62)
    for b, r in zip(bars, rows):
        if r["rescaled_x100"]:
            b.set_hatch("//")
            b.set_edgecolor("white")
        if not r["sd_ok"]:
            ax.text(
                b.get_x() + b.get_width() / 2,
                r["sd"] + 1.2,
                "FAILS floor",
                ha="center",
                fontsize=7,
                color="#B00020",
                fontweight="bold",
            )
    ax.axhline(GATE2_SD_FLOOR, color="#B00020", lw=1.2, ls="--")
    ax.set_ylim(0, 46)
    ax.set_title(beh, fontsize=11)
    if col == 0:
        ax.set_ylabel("inter-context SD\n(0-100 scale)")
        ax.text(-0.42, GATE2_SD_FLOOR + 1.6, "floor: SD >= 10", fontsize=7, color="#B00020")
    # -- bottom: fraction of contexts in the bottom [0,10) bin vs the 80% ceiling
    ax = axes[1][col]
    bars = ax.bar(x, [r["bottom_frac"] for r in rows], color=BEH_COLOR[beh], width=0.62)
    for b, r in zip(bars, rows):
        if r["rescaled_x100"]:
            b.set_hatch("//")
            b.set_edgecolor("white")
        if r["bottom_ok"] is False:
            ax.text(
                b.get_x() + b.get_width() / 2,
                min(r["bottom_frac"] + 0.03, 0.94),
                "FAILS",
                ha="center",
                fontsize=7,
                color="#B00020",
                fontweight="bold",
            )
    ax.axhline(GATE2_BOTTOM_FRAC_MAX, color="#B00020", lw=1.2, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(xs, fontsize=8)
    if col == 0:
        ax.set_ylabel("fraction of contexts in\nthe bottom [0,10) bin")
        ax.text(-0.42, GATE2_BOTTOM_FRAC_MAX + 0.04, "ceiling: < 0.80", fontsize=7, color="#B00020")
fig_title(
    fig,
    "Behavior-expression spread against the pre-registered gate (SD >= 10 AND < 80% bottom bin)",
    "Both panels per behavior are the two conditions of the same pre-registered gate; a setting "
    "passes only if BOTH hold.\nHatched bars are hallucination's own rungs, whose 0-1 fabrication "
    "rate is rescaled x100 onto the gate's scale — a DIFFERENT\nconstruct from the graded "
    "trait score, not comparable to the solid bars beside it.",
)
fig.tight_layout(rect=(0, 0, 1, 0.87))
savefig_paper(fig, "spread_grid", dir=OUT)
plt.close(fig)
print("wrote spread_grid")

# ================================================ 6. bare-query v2 resolutions ==
BQ = load(WT / "bareq_map/evil/all_arms_spearman.json")
sub_rows = [r for r in BQ["transfer_rows"] if str(r.get("leg")) == "2" and r.get("subset")]
band = BQ["meta"]["leg2_shuffled_map_seed_bands"][0]
mb = BQ["meta"]["mapping_baselines"]["leg2"]
STATS["bareq_v2"] = {
    "subsets": {},
    "shuffle_band": {k: v for k, v in band.items() if k != "per_seed"},
    "shuffle_per_seed": band["per_seed"],
}

fig = plt.figure(figsize=(16.5, 6.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.85, 1.15], wspace=0.36)

# (a) turn subsets
ax = fig.add_subplot(gs[0, 0])
SUB_COLOR = {"pooled": "#111111", "multi_turn_only": "#C2185B", "single_turn_only": "#00838F"}
SUB_LABEL = {
    "pooled": "pooled (all 1,987)",
    "multi_turn_only": "multi-turn only (1,009)",
    "single_turn_only": "single-turn only (978)",
}
arms_s = [
    a for a in ["arm6_map_proj_e1", "arm4_ridge_ctx", "arm11_oracle_proj", "arm13_shuffled_map"]
]
y = np.arange(len(arms_s))
h = 0.26
for j, s in enumerate(["pooled", "multi_turn_only", "single_turn_only"]):
    vals, los, his = [], [], []
    for a in arms_s:
        r = next((x for x in sub_rows if x["arm"] == a and x["subset"] == s), None)
        v = r["rho_frozen"] if r else float("nan")
        ci = (r.get("ci_frozen") if r else None) or [None, None]
        lo, hi = err_offsets(v, ci[0], ci[1])
        vals.append(v if finite(v) else 0.0)
        los.append(lo)
        his.append(hi)
        STATS["bareq_v2"]["subsets"].setdefault(a, {})[s] = {
            "rho": v,
            "ci": ci,
            "n": r["n_eval"] if r else None,
        }
    ax.barh(y + (1 - j) * h, vals, height=h, color=SUB_COLOR[s], label=SUB_LABEL[s])
    ax.errorbar(
        vals,
        y + (1 - j) * h,
        xerr=np.array([los, his]),
        fmt="none",
        ecolor="#333333",
        capsize=2.0,
        elinewidth=0.9,
    )
ax.axvline(0, color="#666666", lw=0.9)
ax.set_yticks(y)
ax.set_yticklabels([ARM_LABEL[a] for a in arms_s], fontsize=8)
ax.invert_yaxis()
ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95)
ax.set_xlabel(r"Spearman $\rho$")
ax.set_title("(a) evil dedicated bare fit, by conversation-turn subset", fontsize=9.5)

# (b) shuffle-seed band
ax = fig.add_subplot(gs[0, 1])
per = band["per_seed"]
vals = [p["rho_frozen"] for p in per]
ax.axhspan(
    band["band_p2_5"],
    band["band_p97_5"],
    color="#BBBBBB",
    alpha=0.55,
    label="2.5-97.5 pct of the 8 draws",
)
ax.axhline(0, color="#666666", lw=0.9)
ax.axhline(
    band["rho_mean"], color="#333333", lw=1.2, ls="--", label=f"8-draw mean {band['rho_mean']:+.3f}"
)
ax.scatter(
    [p["shuffle_seed"] for p in per],
    vals,
    s=44,
    color="#999999",
    zorder=3,
    label="individual shuffle draws",
)
ax.scatter(
    [0],
    [per[0]["rho_frozen"]],
    s=95,
    facecolor="#B00020",
    edgecolor="black",
    zorder=4,
    label=f"committed row (seed 0) {per[0]['rho_frozen']:+.3f}",
)
ax.set_xlabel("shuffle seed")
ax.set_ylabel(r"shuffled-map control $\rho$")
ax.set_title("(b) the nonsense-map control across shuffle draws", fontsize=9.5)
ax.legend(fontsize=6.8, loc="lower left", framealpha=0.95)

# (c) mapping baselines
ax = fig.add_subplot(gs[0, 2])
pooled = mb["pooled_per_layer"]
layers = [p["layer_idx"] for p in pooled]
r2m = [p["r2_map_mean"] for p in pooled]
r2i = [p["r2_identity_bias_mean"] for p in pooled]
ax.plot(layers, r2m, color="#1B7837", lw=1.7, label=r"fitted map, held-out $R^2$")
ax.plot(layers, r2i, color="#762A83", lw=1.7, ls="--", label=r"identity+learned-bias $R^2$")
ax.axhline(0, color="#666666", lw=0.9)
ax.set_xlabel("layer")
ax.set_ylabel(r"held-out $R^2$")
ax2 = ax.twinx()
acc = [p["knn_acc_at_k_mean"]["cosine"]["1"] for p in pooled]
ch = [p["knn_chance_at_k_mean"]["1"] for p in pooled]
ax2.plot(layers, acc, color="#E08214", lw=1.7, label="kNN retrieval acc@1 (cosine)")
ax2.plot(layers, ch, color="#E08214", lw=1.0, ls=":", label="retrieval chance = 1/n_pool")
ax2.set_ylabel("acc@1", color="#E08214")
ax2.tick_params(axis="y", labelcolor="#E08214")
lines = [ln for ln in ax.get_lines() + ax2.get_lines() if not ln.get_label().startswith("_")]
ax.legend(lines, [ln.get_label() for ln in lines], fontsize=6.8, loc="lower left", framealpha=0.95)
ax.set_title("(c) standing-rule map baselines (5 by-query folds)", fontsize=9.5)
STATS["bareq_v2"]["mapping_baselines_pooled"] = pooled
STATS["bareq_v2"]["mapping_baselines_meta"] = {
    k: v for k, v in mb.items() if k not in ("per_fold", "pooled_per_layer")
}

fig_title(
    fig,
    "Bare-query round v2: what the added diagnostics settle",
    "(a) the conversation-turn split the previous cut could not compute; (b) the shuffled-map "
    "control re-run across 8 shuffle draws;\n(c) the standing identity+learned-bias and "
    "kNN-retrieval baselines for the bare-rep -> answer map, pooled over the five by-query "
    "folds.",
)
fig.tight_layout(rect=(0, 0, 1, 0.83))
savefig_paper(fig, "bareq_v2_resolutions", dir=OUT)
plt.close(fig)
print("wrote bareq_v2_resolutions")

# ============================================================= dump stats ==
with open("/tmp/i1739_final_stats.json", "w") as fh:
    json.dump(STATS, fh, indent=1, default=float)
print("stats -> /tmp/i1739_final_stats.json")
