# ruff: noqa: RUF001, RUF002
# Intentional Unicode (minus sign, arrows) in scientific docstrings + figure labels.
"""Figures for task #833 follow-up round (fixed-template-weights-read, plan v13).

Reads ONLY the persisted round outputs under ``eval_results/issue_833/``:
``chain_rho_fixedtext/fact_L{7,14,21}.json`` (arms + paired diffs + the
PERSISTED per-cell held-out predictions — no refit happens here) and
``analysis_tensors_fixedtext/base_consistency.json`` (the in-run base-leg
determinism control's per-group rel-L2 values). Committed anchors are quoted
from the fixedtext JSONs' ``meta.committed_anchors`` block.

Hero (fig 1): forest of the six fixedtext arms per layer + the committed
on-/off-policy/base-map anchor ticks, with a paired-diff panel showing the
PRIMARY (on_fixedtext − ctrl_fixedtext), the carrier-independence adjudicator
(on_fixedtext − off_full_recomp), and the retained-slice pair
(on_fixedtext_ret − ctrl_fixedtext_ret) — the round-2 figure grammar.
Exploratory (figs 2-3): L14 per-cell scatter (measured E vs held-out
prediction) for on_fixedtext / ctrl_fixedtext + the retained-slice pair;
base-leg cross-source consistency histogram; L7/L21 scatters.

Run from the issue-833 worktree root:
    set -a && source .env && set +a && \
    OMP_NUM_THREADS=8 uv run python scripts/issue833_figures_fixedtext.py
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
from scipy.stats import spearmanr  # noqa: E402

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
LAYERS = [7, 14, 21]

# Plain-English point labels for the 16 source contexts (paper-plots §3.5:
# no config slugs in rendered figure text; slugs stay in sidecar provenance).
SOURCE_LABELS = {
    "binst_fact": "fact instruction",
    "default": "default assistant",
    "fmt_code": "code format",
    "fmt_json": "JSON format",
    "icl_k2": "2-demo prefix",
    "icl_k8": "8-demo prefix",
    "reph_casual": "casual rephrase",
    "reph_imp": "imperative rephrase",
    "reph_polite": "polite rephrase",
    "sp_doctor": "doctor persona",
    "sp_ph1": "PersonaHub 1",
    "sp_ph2": "PersonaHub 2",
    "sp_swe": "software engineer",
    "wc_long_write": "WildChat writing",
    "wc_short_advice": "WildChat advice",
    "wc_short_code": "WildChat coding",
}

set_paper_style("blog")
C_OFF = paper_palette_role("baseline")
C_ON = paper_palette_role("primary")
C_CTRL = paper_palette_role("control")
C_NEU = paper_palette_role("neutral")

# (arm, label, color, marker, filled) — full-480 arms filled, retained-291 open.
ARMS = [
    ("on_fixedtext", "fixed template (trained map, 480 cells)", C_ON, "o", True),
    ("ctrl_fixedtext", "fixed template (base weights, 480 cells)", C_CTRL, "o", True),
    ("off_full_recomp", "base-written text (off-policy, recomputed)", C_OFF, "o", True),
    ("base_full_recomp", "base map (recomputed floor)", C_NEU, "o", True),
    ("on_fixedtext_ret", "fixed template (trained map, 291 retained)", C_ON, "D", False),
    ("ctrl_fixedtext_ret", "fixed template (base weights, 291 retained)", C_CTRL, "D", False),
]
DIFFS = [
    ("on_fixedtext_minus_ctrl_fixedtext", "trained − base weights (PRIMARY, 480)", C_ON, "o"),
    ("on_fixedtext_minus_off_full_recomp", "fixed template − off-policy", C_OFF, "o"),
    (
        "on_fixedtext_ret_minus_ctrl_fixedtext_ret",
        "trained − base weights (291 retained)",
        C_ON,
        "D",
    ),
]
ANCHORS = [
    ("rho_Mplus_on_ridge", "committed anchors (on-/off-policy, base map)"),
    ("rho_Mplus_off_ridge", "off-policy anchor"),
    ("rho_M0_ridge", "base-map anchor"),
]


def _chain_json(layer: int) -> dict:
    with open(RES / f"chain_rho_fixedtext/fact_L{layer}.json") as f:
        return json.load(f)


# ── Figure 1 (hero): six-arm forest + committed anchors + paired diffs ────────
def fig_forest() -> None:
    """Left: 6 fixedtext arms per layer + committed anchor ticks; right: the 3
    registered paired differences per layer."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), width_ratios=[1.35, 1.0])
    data = {li: _chain_json(li) for li in LAYERS}

    ax = axes[0]
    n = len(ARMS)
    for i, li in enumerate(LAYERS):
        y0 = len(LAYERS) - 1 - i
        anchors = data[li]["meta"]["committed_anchors"][f"L{li}"]
        for j, (key, lab) in enumerate(ANCHORS):
            ax.plot(
                anchors[key],
                y0,
                marker="|",
                ms=16,
                mew=1.6,
                color="0.35",
                ls="none",
                label=(lab if (i == 0 and j == 0) else None),
            )
        for k, (arm, lab, col, mk, filled) in enumerate(ARMS):
            ci = data[li]["arms"][arm]["ci_ridge"]
            y = y0 + (k - (n - 1) / 2) * -0.115
            kw = dict(fmt=mk, color=col, ms=5.5, capsize=2.2, lw=1.3)
            if not filled:
                kw.update(mfc="white", markeredgewidth=1.3)
            ax.errorbar(
                ci["point"],
                y,
                xerr=[[ci["point"] - ci["ci_lo"]], [ci["ci_hi"] - ci["point"]]],
                label=lab if i == 0 else None,
                **kw,
            )
    ax.axvline(0.0, color="0.6", lw=0.9, ls="--")
    for ysep in [0.5, 1.5]:
        ax.axhline(ysep, color="0.85", lw=0.8)
    ax.set_yticks([2, 1, 0], [f"layer {li}" for li in LAYERS])
    ax.set_xlabel("held-out chain correlation with leakage E (Spearman)")
    ax.set_title("fixed-template chain correlation vs anchors", fontsize=11)
    ax.legend(loc="upper left", fontsize=7.2)

    ax = axes[1]
    for i, li in enumerate(LAYERS):
        y0 = len(LAYERS) - 1 - i
        for k, (key, lab, col, mk) in enumerate(DIFFS):
            d = data[li]["paired_diffs"][key]
            y = y0 + (k - 1) * -0.18
            kw = dict(fmt=mk, color=col, ms=5.5, capsize=2.2, lw=1.3)
            if k == 2:
                kw.update(mfc="white", markeredgewidth=1.3)
            ax.errorbar(
                d["point"],
                y,
                xerr=[[d["point"] - d["ci_lo"]], [d["ci_hi"] - d["point"]]],
                label=lab if i == 0 else None,
                **kw,
            )
    ax.axvline(0.0, color="0.6", lw=0.9, ls="--")
    for ysep in [0.5, 1.5]:
        ax.axhline(ysep, color="0.85", lw=0.8)
    ax.set_yticks([2, 1, 0], ["" for _ in LAYERS])
    ax.set_xlabel("paired chain-correlation difference")
    ax.set_title("paired differences per layer", fontsize=11)
    ax.legend(loc="upper left", fontsize=7.2)
    fig.tight_layout()
    savefig_paper(fig, "issue_833/chain_rho_fixedtext_paired", dir="figures/")
    plt.close(fig)


# ── Figure 2: L14 per-cell scatters (from the PERSISTED per-cell chains) ──────
def _scatter(ax, data: dict, per_cell_key: str, arm: str, lab: str, color) -> None:
    """One E-vs-prediction scatter from a persisted per_cell block; the recomputed
    Spearman must match the persisted rho (tol 1e-6 — same data, same rank op),
    so the per-unit view is provably the data behind the persisted aggregate."""
    block = data[per_cell_key]
    chain = np.asarray(block["chains"][arm], dtype=np.float64)
    E = np.asarray(block["E"], dtype=np.float64)
    r_chk, _ = spearmanr(chain, E)
    persisted = data["arms"][arm]["rho_ridge"]
    assert abs(r_chk - persisted) < 1e-6, (arm, r_chk, persisted)
    ax.scatter(chain, E, s=14, alpha=0.5, color=color, linewidths=0)
    ax.set_xlabel("held-out map prediction along the fact direction")
    ax.set_ylabel("measured leakage E")
    ax.set_title(f"{lab}: Spearman {persisted:+.2f} (n={len(E)})", fontsize=10.5)


def fig_percell_l14() -> None:
    """Four L14 scatters: on/ctrl fixedtext (480 cells) + the retained-291 pair."""
    d = _chain_json(14)
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.3))
    _scatter(axes[0], d, "per_cell_full", "on_fixedtext", "fixed template, trained map", C_ON)
    _scatter(axes[1], d, "per_cell_full", "ctrl_fixedtext", "fixed template, base weights", C_CTRL)
    _scatter(axes[2], d, "per_cell_retained", "on_fixedtext_ret", "trained map, 291 retained", C_ON)
    _scatter(
        axes[3],
        d,
        "per_cell_retained",
        "ctrl_fixedtext_ret",
        "base weights, 291 retained",
        C_CTRL,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_833/fixedtext_percell_L14", dir="figures/")
    plt.close(fig)


# ── Figure 3: base-leg consistency histogram + L7/L21 exploratory scatters ────
def fig_consistency_and_layers() -> None:
    """Left: histogram of the 90 per-(target, layer) base-leg cross-source
    rel-L2 values (30 targets x 3 layers; each group pools its 16 source
    copies out of the 1,440 npz — the in-run determinism control); right two:
    L7/L21 on_fixedtext scatters."""
    cons = json.loads((RES / "analysis_tensors_fixedtext/base_consistency.json").read_text())
    rel = np.asarray(list(cons["per_group_rel_l2"].values()), dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    ax = axes[0]
    floor = 1e-12
    ax.hist(np.log10(np.maximum(rel, floor)), bins=40, color=C_NEU)
    ax.axvline(np.log10(cons["tolerance"]), color="0.3", lw=1.1, ls="--")
    ax.set_xlabel("log10 rel L2 across the 16 source copies of v0(R_fixed)")
    ax.set_ylabel("(target, layer) groups")
    ax.set_title(
        f"base-leg determinism: max {cons['max_rel_l2']:.1e} (tol {cons['tolerance']:g})",
        fontsize=10.5,
    )
    for ax, li in [(axes[1], 7), (axes[2], 21)]:
        d = _chain_json(li)
        _scatter(ax, d, "per_cell_full", "on_fixedtext", f"trained map, layer {li}", C_ON)
    fig.tight_layout()
    savefig_paper(fig, "issue_833/fixedtext_consistency_layers", dir="figures/")
    plt.close(fig)


# ── Figure 4: between-source vs within-source decomposition + basis coverage ──
def fig_decomposition() -> None:
    """Left: 16 labeled source-mean points (on_fixedtext L14 prediction vs E);
    middle: the 480 within-source demeaned residuals; right: captured-variance
    fraction of each response stack in the 64-dim V0 basis per layer. The
    demeaned/source-mean Spearmans are asserted against the persisted
    ``chain_rho_fixedtext/analyzer_reads.json`` values (tol 1e-9)."""
    reads = json.loads((RES / "chain_rho_fixedtext/analyzer_reads.json").read_text())
    d = _chain_json(14)
    pc = d["per_cell_full"]
    keys = pc["keys"]
    E = np.asarray(pc["E"], dtype=np.float64)
    chain = np.asarray(pc["chains"]["on_fixedtext"], dtype=np.float64)
    srcs = np.asarray([k.split("/", 1)[1].split("__")[0] for k in keys])
    names = sorted(set(srcs))

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5), width_ratios=[1.0, 1.0, 1.15])
    ax = axes[0]
    pm = np.asarray([chain[srcs == s].mean() for s in names])
    Em = np.asarray([E[srcs == s].mean() for s in names])
    r_bs, _ = spearmanr(pm, Em)
    ax.scatter(pm, Em, s=26, color=C_ON, linewidths=0)
    for x, y, s in zip(pm, Em, names, strict=True):
        ax.text(x, y, SOURCE_LABELS[s], fontsize=6.2, ha="left", va="bottom")
    ax.set_xlabel("source-mean held-out prediction")
    ax.set_ylabel("source-mean leakage E")
    ax.set_title(f"between sources: Spearman {r_bs:+.2f} (n=16)", fontsize=10.5)

    ax = axes[1]
    Ed, chd = E.copy(), chain.copy()
    for s in names:
        m = srcs == s
        Ed[m] -= Ed[m].mean()
        chd[m] -= chd[m].mean()
    r_wi, _ = spearmanr(chd, Ed)
    persisted = reads["L14"]["demeaned_spearman_on_fixedtext"]["point"]
    assert abs(r_wi - persisted) < 1e-9, (r_wi, persisted)
    ax.scatter(chd, Ed, s=12, alpha=0.45, color=C_ON, linewidths=0)
    ax.set_xlabel("prediction, source mean removed")
    ax.set_ylabel("leakage E, source mean removed")
    ax.set_title(f"within sources: Spearman {r_wi:+.2f} (n=480)", fontsize=10.5)

    ax = axes[2]
    cap = reads["captured_variance"]
    series = [
        ("v_plus_R_fixed", "fixed template, trained model", C_ON),
        ("v0_R_fixed", "fixed template, base model", C_CTRL),
        ("Vplus_offpolicy_ref", "base-written text, trained model", C_OFF),
    ]
    x = np.arange(len(LAYERS))
    w = 0.26
    for j, (key, lab, col) in enumerate(series):
        vals = [cap[f"L{li}"][key] for li in LAYERS]
        ax.bar(x + (j - 1) * w, vals, width=w, color=col, label=lab)
    ax.set_xticks(x, [f"layer {li}" for li in LAYERS])
    ax.set_ylabel("variance fraction captured by the 64-dim base basis")
    ax.set_title("answer-profile variance inside the fit basis", fontsize=10.5)
    ax.legend(fontsize=7.2)
    fig.tight_layout()
    savefig_paper(fig, "issue_833/fixedtext_decomposition_L14", dir="figures/")
    plt.close(fig)


def main() -> None:
    """Build the four fixedtext figures from persisted outputs only."""
    for li in LAYERS:
        p = RES / f"chain_rho_fixedtext/fact_L{li}.json"
        assert p.exists(), f"{p} missing — run scripts/issue833_chain_rho_fixedtext.py first"
    fig_forest()
    fig_percell_l14()
    fig_consistency_and_layers()
    fig_decomposition()
    print("figures written to figures/issue_833/")


if __name__ == "__main__":
    main()
