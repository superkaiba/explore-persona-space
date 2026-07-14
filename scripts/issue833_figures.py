"""Figures for task #833 — on-policy vs off-policy function-change / chain-rho.

Reads ONLY the persisted Phase-D outputs under ``eval_results/issue_833/``
(cells/, chain_rho/, decomposition/, text_divergence/, joined_cache/) plus the
git-committed leakage target ``eval_results/issue_537/G_tensor/G_meta.json``.
The chain-scatter panel recomputes the ridge LOCO chain predictions from the
joined cache via the imported #722 harness (identical math) and ASSERTS the
recomputed Spearman matches the persisted ``chain_rho/*.json`` value, so the
per-unit view is provably the data behind the persisted aggregate.

Run from the issue-833 worktree root:
    set -a && source .env && set +a && uv run python scripts/issue833_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PROJECT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

RES = PROJECT / "eval_results/issue_833"
OUT = PROJECT / "figures/issue_833"
OUT.mkdir(parents=True, exist_ok=True)

BEHAVIORS = ["fact", "em", "sycophancy"]
LAYERS = [7, 14, 21]
BEH_LABEL = {"fact": "taught fact", "em": "EM", "sycophancy": "sycophancy"}

set_paper_style("blog")
C_M0 = paper_palette_role("neutral")
C_OFF = paper_palette_role("baseline")
C_ON = paper_palette_role("primary")
C_CTRL = paper_palette_role("control")


def pretty_cid(cid: str) -> str:
    m = {
        "default": "default assistant",
        "binst_fact": "behavior-instr (fact)",
        "binst_em": "behavior-instr (EM)",
        "binst_marker": "behavior-instr (marker)",
        "fmt_code": "format (code)",
        "fmt_json": "format (JSON)",
        "icl_k2": "in-context k=2",
        "icl_k8": "in-context k=8",
        "reph_casual": "rephrase (casual)",
        "reph_imp": "rephrase (imperative)",
        "reph_polite": "rephrase (polite)",
        "sp_doctor": "sys-prompt (doctor)",
        "sp_ph1": "sys-prompt (persona 1)",
        "sp_ph2": "sys-prompt (persona 2)",
        "sp_swe": "sys-prompt (engineer)",
        "wc_long_write": "WildChat long writing",
        "wc_short_advice": "WildChat short advice",
        "wc_short_code": "WildChat short code",
    }
    return m.get(cid, cid.replace("_", " "))


def load_json(rel: str) -> dict:
    with open(RES / rel) as f:
        return json.load(f)


# ── Figure 1 (HERO): chain-rho forest, 9 rows x 3 arms ────────────────────────
def fig_chain_forest() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    rows = [(b, li) for b in BEHAVIORS for li in LAYERS]
    yticks, ylabels = [], []
    arms = [
        ("M0", "base map (base answers)", C_M0),
        ("Mplus_off", "post-FT map, base answers (off-policy)", C_OFF),
        ("Mplus_on", "post-FT map, own answers (on-policy)", C_ON),
    ]
    for i, (beh, li) in enumerate(rows):
        d = load_json(f"chain_rho/{beh}_L{li}.json")
        y0 = len(rows) - 1 - i
        yticks.append(y0)
        ylabels.append(f"{BEH_LABEL[beh]}  L{li}")
        for k, (arm, _lab, col) in enumerate(arms):
            ci = d[f"ci_{arm}_ridge"]
            y = y0 + (k - 1) * 0.22
            ax.errorbar(
                ci["point"],
                y,
                xerr=[[ci["point"] - ci["ci_lo"]], [ci["ci_hi"] - ci["point"]]],
                fmt="o",
                color=col,
                ms=5.5,
                capsize=2.5,
                lw=1.4,
            )
    ax.axvline(0.0, color="0.6", lw=0.9, ls="--")
    for ysep in [2.5, 5.5]:
        ax.axhline(ysep, color="0.85", lw=0.8)
    ax.set_yticks(yticks, ylabels)
    ax.set_xlabel("context-to-leakage chain correlation (Spearman, held-out)")
    ax.set_title(
        "Predicting leakage from the fitted map: the taught-fact chain\n"
        "strengthens with on-policy answers; EM and sycophancy stay null",
        pad=14,
    )
    handles = [plt.Line2D([], [], color=c, marker="o", ls="none", label=lab) for _a, lab, c in arms]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    savefig_paper(fig, "chain_rho_three_arms", dir=OUT)
    plt.close(fig)


# ── Figure 2: per-cell chain scatter (fact, headline layer) ───────────────────
def fig_chain_scatter() -> None:
    """Per-cell LOCO-prediction-vs-E scatter, fact L14, all FOUR maps.

    Recomputes each map's ridge LOCO chain from the joined cache via the
    imported #722 harness and asserts the Spearman matches the persisted
    aggregate (chain_rho/ for the three production arms, chain_rho_ctrl/ for
    the matched-text control), so every panel is provably the per-unit data
    behind its persisted forest-plot point.
    """
    import issue722_fit_M as fitM  # heavy import (torch) — local

    beh, li = "fact", 14
    z = np.load(RES / f"joined_cache/{beh}_L{li}.npz", allow_pickle=True)
    C0, Cplus = z["C0"], z["Cplus"]
    V0, Vplus, Von, V0on = z["V0"], z["Vplus"], z["Von"], z["V0on"]
    cell_keys = [str(x) for x in z["cell_keys"]]
    rb_main = fitM._load_rb_main()
    rb_fact = fitM._load_rb_fact()
    r_hat = fitM._r_hat_for(beh, li, rb_main, rb_fact)
    E = fitM._load_E(beh, cell_keys)
    keep = ~np.isnan(E)
    Ek = E[keep]
    pca = fitM._pca_basis_v0(V0, fitM.TARGET_DIM)
    persisted = load_json(f"chain_rho/{beh}_L{li}.json")
    persisted_ctrl = load_json(f"chain_rho_ctrl/{beh}_L{li}.json")

    arm_specs = [
        ("M0", "base map (base answers)", C0, V0 @ pca.T, C_M0, persisted),
        ("Mplus_off", "post-FT map, base answers", Cplus, Vplus @ pca.T, C_OFF, persisted),
        ("Mplus_on", "post-FT map, own answers", Cplus, Von @ pca.T, C_ON, persisted),
        ("M0_ctrl", "base map, own answers (control)", C0, V0on @ pca.T, C_CTRL, persisted_ctrl),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.9), sharey=True)
    for ax, (arm, lab, X, Y64, col, ref_json) in zip(axes, arm_specs, strict=True):
        loco = fitM._ridge_loco_pred(X, Y64)
        rho, chain = fitM._chain_rho_one(loco[keep], pca, r_hat, Ek)
        ref = ref_json[f"rho_{arm}_ridge"]
        # The persisted production fits ran on GPU (FIT_DEVICE=cuda, r7e); this
        # CPU float64 recompute may flip a PRESS-lambda near-tie — accept <0.01
        # rank drift. The control fit ran on CPU, so it should match tightly.
        assert abs(rho - ref) < 1e-2, (arm, rho, ref)
        print(f"[chain-scatter] {arm}: recomputed {rho:+.4f} vs persisted {ref:+.4f}")
        ax.scatter(chain, Ek, s=10, alpha=0.45, color=col, edgecolors="none")
        ax.set_title(f"{lab}\nrank correlation = {ref:+.2f}", fontsize=10)
        ax.set_xlabel("map prediction along the fact direction")
    axes[0].set_ylabel("measured leakage (judge score)")
    fig.suptitle(
        "Taught fact, layer 14: held-out map predictions vs measured leakage "
        "(480 source-target cells)",
        y=1.04,
    )
    savefig_paper(fig, "chain_scatter_fact_L14", dir=OUT)
    plt.close(fig)


# ── Figure 3: function-change magnitude — raw vs regime-local floor units ─────
def fig_function_change() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    labels = [f"{BEH_LABEL[b]}\nL{li}" for b in BEHAVIORS for li in LAYERS]
    xs = np.arange(9, dtype=float)
    raw = {"off": [], "on": [], "ctrl": []}
    fu = {"off": [], "on": [], "ctrl": []}
    for b in BEHAVIORS:
        for li in LAYERS:
            d = load_json(f"cells/{b}_L{li}.json")
            raw["off"].append(d["delta_off_raw"]["median_ci"]["point"])
            raw["on"].append(d["delta_on_raw"]["median_ci"]["point"])
            raw["ctrl"].append(d["delta_ctrl_raw"]["median_ci"]["point"])
            fu["off"].append(d["delta_off_over_own_floor"]["point"])
            fu["on"].append(d["delta_on_over_own_floor"]["point"])
            fu["ctrl"].append(d["delta_ctrl_over_own_floor"]["point"])
    w = 0.26
    series = [
        ("off", "off-policy (base answers)", C_OFF),
        ("on", "on-policy (own answers)", C_ON),
        ("ctrl", "matched-text control (base map, own answers)", C_CTRL),
    ]
    for ax, data, ylab, title in [
        (axes[0], raw, "median map change along the behavior direction (raw)", "Raw units"),
        (
            axes[1],
            fu,
            "map change / its own refit noise floor",
            "Regime-local floor units (1 = at floor)",
        ),
    ]:
        for k, (key, lab, col) in enumerate(series):
            ax.bar(xs + (k - 1) * w, data[key], width=w, color=col, label=lab)
        ax.set_xticks(xs, labels, fontsize=7, rotation=35, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
    axes[0].set_yscale("log")
    axes[0].set_ylim(top=200)
    axes[1].axhline(1.0, color="0.4", lw=1.0, ls="--")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        "Function change on-policy vs off-policy: raw deltas explode, "
        "floor-normalized reads do not",
        y=1.05,
    )
    savefig_paper(fig, "function_change_raw_vs_floor", dir=OUT)
    plt.close(fig)


# ── Figure 4: per-source paired deltas (fact), labeled — low-level view ───────
def fig_per_source_fact() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), sharex=False, sharey=False)
    for ax, li in zip(axes, LAYERS, strict=True):
        d = load_json(f"cells/fact_L{li}.json")
        pc = d["per_cell"]
        src = np.array(pc["source_cids"])
        off = np.array(pc["proj_off"])
        on = np.array(pc["proj_on"])
        uniq = sorted(set(src))
        pts = [(off[src == s][0], on[src == s][0], s) for s in uniq]
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        lo = min(x.min(), y.min()) * 0.5
        hi = max(x.max(), y.max()) * 2.0
        ax.plot([lo, hi], [lo, hi], color="0.7", lw=0.9, ls="--")
        ax.scatter(x, y, s=26, color=C_ON, zorder=3)
        # Stagger labels: alternate sides + a small vertical cycle to declutter.
        order = np.argsort(y)
        for rank, j in enumerate(order):
            xi, yi, s = pts[j]
            right = rank % 2 == 0
            dy = 1.0 + 0.055 * ((rank % 3) - 1)
            ax.text(
                xi * (1.09 if right else 1 / 1.09),
                yi * dy,
                pretty_cid(s),
                fontsize=5.4,
                va="center",
                ha="left" if right else "right",
                color="0.25",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"layer {li}", fontsize=11)
        ax.set_xlabel("off-policy map change (raw)")
    axes[0].set_ylabel("on-policy map change (raw)")
    fig.suptitle(
        "Taught fact, per source context: the on-policy map change exceeds the "
        "off-policy one for every source (dashed = equality)",
        y=1.03,
    )
    savefig_paper(fig, "function_change_per_source_fact", dir=OUT)
    plt.close(fig)


# ── Figure 5: decomposition — text vs representation components ───────────────
def fig_decomposition() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        d = load_json(f"decomposition/{beh}_L14.json")
        pc = d["per_cell"]
        text = np.array(pc["proj_text"])
        rep = np.array(pc["proj_rep"])
        src = np.array(pc["source_cids"])
        hi = max(text.max(), rep.max()) * 1.7
        lo = max(min(text[text > 0].min(), rep[rep > 0].min()) * 0.5, 1e-4)
        ax.scatter(text, rep, s=8, alpha=0.3, color="0.55", edgecolors="none", zorder=2)
        for s in sorted(set(src)):
            m = src == s
            ax.scatter(np.median(text[m]), np.median(rep[m]), s=34, color=C_ON, zorder=4)
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(BEH_LABEL[beh], fontsize=11)
        ax.set_xlabel("output-text component (same weights, new text)")
    axes[0].set_ylabel("representation component\n(same text, new weights)")
    fig.suptitle(
        "Layer 14 decomposition of the total on-policy profile change: the "
        "output-text component dominates (dashed = equal split); grey = cells, "
        "blue = per-source medians",
        y=1.05,
    )
    savefig_paper(fig, "decomposition_text_vs_rep_L14", dir=OUT)
    plt.close(fig)


# ── Figure 6: text-divergence manipulation check ──────────────────────────────
def fig_text_divergence() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharey=False)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        d = load_json(f"text_divergence/{beh}.json")
        vals = np.asarray(d["edit_distance"]["values"], dtype=float)
        ax.hist(vals, bins=30, range=(0, 1), color=C_ON, alpha=0.85)
        ax.set_title(f"{BEH_LABEL[beh]} — median {d['edit_distance']['median']:.2f}", fontsize=10.5)
        ax.set_xlabel("normalized edit distance")
    axes[0].set_ylabel("response pairs (own vs base answer)")
    fig.suptitle(
        "Manipulation check: the trained models' own answers diverge heavily "
        "from base answers in every behavior",
        y=1.04,
    )
    savefig_paper(fig, "text_divergence_hist", dir=OUT)
    plt.close(fig)


# ── Figure 7 (follow-up): matched-text-control chain vs the on-policy chain ───
def fig_chain_ctrl() -> None:
    """Two-panel forest for the M0_ctrl chain follow-up (chain_rho_ctrl/*.json).

    Left: on-policy (post-FT map, own answers) vs matched-text control (base
    map, own answers) chain correlations per behavior x layer, family-clustered
    95% CIs. Right: the paired on-minus-control difference per cell. Reads only
    the persisted follow-up JSONs; asserts all 9 cells are present.
    """
    rows = [(b, li) for b in BEHAVIORS for li in LAYERS]
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11.0, 6.2), sharey=True, width_ratios=[1.6, 1.0]
    )
    arms = [
        ("Mplus_on", "post-FT map, own answers (on-policy)", C_ON, +0.16),
        ("M0_ctrl", "base map, own answers (matched-text control)", C_CTRL, -0.16),
    ]
    yticks, ylabels = [], []
    n_cells = 0
    for i, (beh, li) in enumerate(rows):
        d = load_json(f"chain_rho_ctrl/{beh}_L{li}.json")
        n_cells += 1
        y0 = len(rows) - 1 - i
        yticks.append(y0)
        ylabels.append(f"{BEH_LABEL[beh]}  L{li}")
        for arm, _lab, col, dy in arms:
            ci = d[f"ci_{arm}_ridge"]
            ax_l.errorbar(
                ci["point"],
                y0 + dy,
                xerr=[[ci["point"] - ci["ci_lo"]], [ci["ci_hi"] - ci["point"]]],
                fmt="o",
                color=col,
                ms=5.5,
                capsize=2.5,
                lw=1.4,
            )
        dd = d["ci_diff_Mplus_on_minus_M0_ctrl"]
        ax_r.errorbar(
            dd["point"],
            y0,
            xerr=[[dd["point"] - dd["ci_lo"]], [dd["ci_hi"] - dd["point"]]],
            fmt="o",
            color=C_M0,
            ms=5.5,
            capsize=2.5,
            lw=1.4,
        )
    assert n_cells == 9, n_cells
    for ax in (ax_l, ax_r):
        ax.axvline(0.0, color="0.6", lw=0.9, ls="--")
        for ysep in [2.5, 5.5]:
            ax.axhline(ysep, color="0.85", lw=0.8)
    ax_l.set_yticks(yticks, ylabels)
    ax_l.set_xlabel("context-to-leakage chain correlation (Spearman, held-out)")
    ax_r.set_xlabel("paired difference: on-policy minus control")
    handles = [
        plt.Line2D([], [], color=c, marker="o", ls="none", label=lab) for _a, lab, c, _d in arms
    ]
    ax_l.legend(handles=handles, loc="lower right", frameon=True)
    fig.suptitle(
        "Base weights over the trained models' own answers reproduce the\n"
        "on-policy chain: all nine paired differences are consistent with zero",
        y=1.02,
    )
    savefig_paper(fig, "chain_rho_ctrl_paired", dir=OUT)
    plt.close(fig)


if __name__ == "__main__":
    only = set(sys.argv[1:])
    all_figs = {
        "chain_forest": fig_chain_forest,
        "function_change": fig_function_change,
        "per_source_fact": fig_per_source_fact,
        "decomposition": fig_decomposition,
        "text_divergence": fig_text_divergence,
        "chain_ctrl": fig_chain_ctrl,
        "chain_scatter": fig_chain_scatter,  # heavy import last
    }
    unknown = only - set(all_figs)
    assert not unknown, f"unknown figure name(s): {sorted(unknown)}"
    for name, fn in all_figs.items():
        if only and name not in only:
            continue
        fn()
    print("figures written to", OUT)
