"""Figures for issue #653 clean-result (read/write decomposition verdict).

Reads committed eval JSONs under eval_results/issue_653/ and writes paper-style
figures under figures/issue_653/ via savefig_paper (PNG+PDF+meta.json).
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
ROOT = os.path.abspath(ROOT)
EVAL = os.path.join(ROOT, "eval_results", "issue_653")
FIGDIR = os.path.join(ROOT, "figures")  # savefig_paper prepends dir, we pass dir="figures/"

BEH_LABEL = {"marker": "marker (※)", "sycophancy": "sycophancy", "em": "emergent misalignment"}
SRC_LABEL = {"florist": "florist", "medical_doctor": "medical doctor"}
RUNG_LABEL = {"r1": "rank-1", "r4": "rank-4", "r16": "rank-16"}
RUNG_ORDER = {"r1": 0, "r4": 1, "r16": 2}

# H1/H3 pre-registered thresholds (cross_arm_verdict.json -> thresholds)
TOP_SHARE_LOWRANK = 0.7
PR_LAMBDA_H3 = 5.0
RANK_K_H3 = 10
# EM exemplar (#521 verified on-policy EM read; calibration anchor)
EXEMPLAR_TOP = (0.81, 0.86, 0.89)
EXEMPLAR_PR = (1.49, 1.34, 1.25)
EXEMPLAR_RANK = None  # not reported as rank-k; exemplar is H1 by top-share/PR


def load_dx_cells():
    rows = []
    for f in sorted(glob.glob(os.path.join(EVAL, "armB", "dx_geometry_*.json"))):
        d = json.load(open(f))
        rows.append(d)
    rows.sort(key=lambda r: (r["behavior"], r["source"], RUNG_ORDER[r["rung"]]))
    return rows


def load_verdict():
    return json.load(open(os.path.join(EVAL, "cross_arm_verdict.json")))


def cell_short(beh, src, rung):
    return f"{BEH_LABEL[beh]}\n{SRC_LABEL[src]} · {RUNG_LABEL[rung]}"


# ------------------------------------------------------------------ Figure 1: hero
def fig_hero_diffuse():
    """rank_k_at_90 per cell vs the H3 boundary and the EM-exemplar anchor."""
    rows = load_dx_cells()
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    n = len(rows)
    x = np.arange(n)
    vals = [r["rank_k_at_90"] for r in rows]
    # color by behavior
    beh_colors = {
        "marker": paper_palette_role("primary"),
        "sycophancy": paper_palette_role("accent"),
        "em": paper_palette_role("control"),
    }
    colors = [beh_colors[r["behavior"]] for r in rows]
    ax.bar(x, vals, color=colors, width=0.74, zorder=3)
    # H3 boundary
    ax.axhline(RANK_K_H3, ls="--", lw=1.4, color="#555", zorder=2)
    ax.text(
        n - 0.4,
        RANK_K_H3 + 0.6,
        "H3 boundary (rank-k ≥ 10)",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555",
    )
    # H1 / clean band ~ rank-1..few
    ax.axhspan(0, 3, color="#9ecae1", alpha=0.18, zorder=1)
    ax.text(0.1, 1.4, "clean / low-rank (H1) region", fontsize=8.5, color="#3182bd", va="center")
    ax.set_ylim(0, 56)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [cell_short(r["behavior"], r["source"], r["rung"]) for r in rows], rotation=90, fontsize=7.0
    )
    ax.set_ylabel("modes to reach 90% of Δx variance\n(rank-k@90)")
    # legend by behavior
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=beh_colors[b], label=BEH_LABEL[b]) for b in ("marker", "sycophancy", "em")
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False, ncol=3)
    try:
        from explore_persona_space.analysis.paper_plots import set_title_subtitle

        set_title_subtitle(
            ax,
            "Every finetune's activation shift is diffuse, not low-rank",
            "18 cells (3 behaviors × 2 source contexts × 3 LoRA ranks); all sit at rank-k 41-51 vs the rank-1 ideal",
        )
    except Exception:
        ax.set_title("Every finetune's activation shift is diffuse, not low-rank")
    fig.subplots_adjust(bottom=0.34, top=0.84)
    savefig_paper(fig, "issue_653/hero_dx_diffuse", dir="figures/")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 2: cross-arm
def fig_cross_arm():
    """cos(rho_top, dx_top) per cell, iso + cov, vs random CI."""
    v = load_verdict()
    verds = v["verdicts"]
    verds = sorted(
        verds,
        key=lambda r: (
            r["cell_group"].split("__")[0],
            r["cell_group"].split("__")[1],
            RUNG_ORDER[r["rung"]],
        ),
    )
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    n = len(verds)
    x = np.arange(n)
    iso = [vv["cross_arm"]["iso"]["cos_rho_top_to_dx_top"] for vv in verds]
    cov = [vv["cross_arm"]["cov"]["cos_rho_top_to_dx_top"] for vv in verds]
    rci = [vv["cross_arm"]["iso"]["random_ci_high"] for vv in verds]
    w = 0.4
    c_iso = paper_palette_role("primary")
    c_cov = paper_palette_role("baseline")
    ax.bar(x - w / 2, iso, width=w, color=c_iso, label="isotropic writes", zorder=3)
    ax.bar(x + w / 2, cov, width=w, color=c_cov, label="covariance-matched writes", zorder=3)
    # random-CI band (mean of the per-cell |rci| ~0.036)
    rband = float(np.mean(rci))
    ax.axhspan(-rband, rband, color="#bbb", alpha=0.30, zorder=1)
    ax.text(
        n - 0.4,
        rband + 0.01,
        "random-direction CI (|cos| ≤ 0.04)",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#666",
    )
    ax.axhline(0, color="#888", lw=0.8, zorder=2)
    ax.set_ylim(-0.45, 0.35)
    ax.set_xticks(x)
    labels = []
    for vv in verds:
        beh, src = vv["cell_group"].split("__")
        labels.append(cell_short(beh, src, vv["rung"]))
    ax.set_xticklabels(labels, rotation=90, fontsize=7.0)
    ax.set_ylabel("cos(Arm A ρ top dir, Arm B Δx top dir)")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=2)
    try:
        from explore_persona_space.analysis.paper_plots import set_title_subtitle

        set_title_subtitle(
            ax,
            "Small but statistically non-random and sign-flipping cross-arm cosines",
            "Generation-loop write→read map (Arm A) vs weight-edit activation shift (Arm B); 16/18 iso, 18/18 cov exceed the CI but flip sign cell-to-cell",
        )
    except Exception:
        ax.set_title(
            "Small but statistically non-random and sign-flipping cross-arm cosines (16/18 iso, 18/18 cov exceed CI)"
        )
    fig.subplots_adjust(bottom=0.34, top=0.84)
    savefig_paper(fig, "issue_653/cross_arm_cos", dir="figures/")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 3: Arm A
def fig_arm_a():
    """Arm A: round-trip cos ~ 0 and d_B recovery ~ 0 (write features don't re-read as themselves)."""
    g = json.load(open(os.path.join(EVAL, "armA", "rho_geometry_seed42.json")))
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))

    # Panel 1: round-trip cosine, iso vs cov, vs random CI
    ax = axes[0]
    dists = ["iso", "cov"]
    means = [g["geometry"][d]["round_trip_cos_mean"] for d in dists]
    rci = [g["geometry"][d]["random_ci_high"] for d in dists]
    xx = np.arange(2)
    ax.bar(
        xx,
        means,
        width=0.5,
        color=[paper_palette_role("primary"), paper_palette_role("baseline")],
        zorder=3,
    )
    rband = float(np.mean(rci))
    ax.axhspan(-rband, rband, color="#bbb", alpha=0.30, zorder=1)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(["isotropic\nwrites", "cov-matched\nwrites"], fontsize=9)
    ax.set_ylim(-0.05, 0.25)
    ax.set_ylabel("round-trip cos(w, ρ(w))")
    ax.set_title("Random writes don't re-read\nas themselves", fontsize=10.5, loc="left")
    ax.text(0.02, rband + 0.005, "random-CI", fontsize=8, color="#666")

    # Panel 2: d_B recovery cos per behavior, iso vs cov
    ax = axes[1]
    behs = ["marker", "sycophancy", "em"]
    rec = {}
    for b in behs:
        rec[b] = json.load(open(os.path.join(EVAL, "armA", f"dB_recovery_{b}.json")))[
            "recovery_per_distribution"
        ]
    xx = np.arange(len(behs))
    w = 0.38
    iso = [rec[b]["iso"]["cos_rho_dB_to_rB"] for b in behs]
    cov = [rec[b]["cov"]["cos_rho_dB_to_rB"] for b in behs]
    ax.bar(
        xx - w / 2, iso, width=w, color=paper_palette_role("primary"), label="isotropic", zorder=3
    )
    ax.bar(
        xx + w / 2,
        cov,
        width=w,
        color=paper_palette_role("baseline"),
        label="cov-matched",
        zorder=3,
    )
    rband2 = float(np.mean([rec[b]["iso"]["random_ci_high"] for b in behs]))
    ax.axhspan(-rband2, rband2, color="#bbb", alpha=0.30, zorder=1)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels([BEH_LABEL[b] for b in behs], fontsize=8.5)
    ax.set_ylim(-0.25, 0.35)
    ax.set_ylabel("cos(ρ(d_B), r_B)")
    ax.set_title("ρ(d_B) does not recover\nthe behavior read-out r_B", fontsize=10.5, loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(
        "Arm A: the write→read map is near-low-rank but misses both registered cuts",
        fontsize=11,
        x=0.01,
        ha="left",
        weight="semibold",
    )
    fig.subplots_adjust(bottom=0.18, top=0.82, wspace=0.32)
    savefig_paper(fig, "issue_653/arm_a_alignment", dir="figures/")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 4: install diagnostic
def fig_install():
    """Install DV per cell -> the binding constraint: behaviors barely installed."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))

    # Panel 1: marker logp trained-base per cell
    ax = axes[0]
    marker_cells = []
    for f in sorted(glob.glob(os.path.join(EVAL, "armB", "install_marker_*.json"))):
        d = json.load(open(f))
        marker_cells.append(d)
    marker_cells.sort(
        key=lambda d: (d["behavior"], "florist" not in d["cell_id"], RUNG_ORDER[d["rung"]])
    )

    # reorder florist then medical, r1 r4 r16
    def mkey(d):
        src = "florist" if "florist" in d["cell_id"] else "medical_doctor"
        return (0 if src == "florist" else 1, RUNG_ORDER[d["rung"]])

    marker_cells.sort(key=mkey)
    vals = [d["install"]["logp_trained_minus_base"] for d in marker_cells]
    labs = []
    for d in marker_cells:
        src = "florist" if "florist" in d["cell_id"] else "medical doctor"
        labs.append(f"{src}\n{RUNG_LABEL[d['rung']]}")
    xx = np.arange(len(vals))
    ax.bar(xx, vals, width=0.6, color=paper_palette_role("primary"), zorder=3)
    ax.axhline(0, color="#888", lw=0.8)
    ax.axhspan(5, 13, color="#74c476", alpha=0.20, zorder=1)
    ax.text(
        len(vals) - 0.4,
        9,
        "clean-install\ntarget band\n(5-12 nat)",
        ha="right",
        va="center",
        fontsize=8,
        color="#31a354",
    )
    ax.set_xticks(xx)
    ax.set_xticklabels(labs, rotation=90, fontsize=7.5)
    ax.set_ylim(-0.3, 13.5)
    ax.set_ylabel("marker log P, trained − base (nat)")
    ax.set_title(
        "Marker barely installed\n(peak +0.78 nat vs 5-12 target)", fontsize=10, loc="left"
    )

    # Panel 2: sycophancy + em judge-rate gain per cell
    ax = axes[1]
    jr_cells = []
    for beh in ("sycophancy", "em"):
        for f in sorted(glob.glob(os.path.join(EVAL, "armB", f"install_{beh}_*.json"))):
            jr_cells.append(json.load(open(f)))

    def jkey(d):
        src = "florist" if "florist" in d["cell_id"] else "medical_doctor"
        return (
            0 if d["behavior"] == "sycophancy" else 1,
            0 if src == "florist" else 1,
            RUNG_ORDER[d["rung"]],
        )

    jr_cells.sort(key=jkey)
    gains = [d["install"]["judge_rate_gain"] for d in jr_cells]
    labs = []
    cols = []
    for d in jr_cells:
        src = "florist" if "florist" in d["cell_id"] else "medical"
        labs.append(f"{d['behavior'][:4]}.\n{src} {RUNG_LABEL[d['rung']]}")
        cols.append(
            paper_palette_role("accent")
            if d["behavior"] == "sycophancy"
            else paper_palette_role("control")
        )
    xx = np.arange(len(gains))
    ax.bar(xx, gains, width=0.62, color=cols, zorder=3)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(labs, rotation=90, fontsize=6.6)
    ax.set_ylim(-0.02, 0.30)
    ax.set_ylabel("judge-rate gain, trained − base")
    ax.set_title(
        "Sycophancy weak (peak +0.15);\nEM never installed (all 0.0)", fontsize=10, loc="left"
    )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=paper_palette_role("accent"), label="sycophancy"),
            Patch(facecolor=paper_palette_role("control"), label="EM"),
        ],
        loc="upper right",
        fontsize=8,
        frameon=False,
    )

    fig.suptitle(
        "Binding constraint: the implants barely moved the behavior — the diffuse Δx is read off weak edits",
        fontsize=10.5,
        x=0.01,
        ha="left",
        weight="semibold",
    )
    fig.subplots_adjust(bottom=0.30, top=0.82, wspace=0.34)
    savefig_paper(fig, "issue_653/install_diagnostic", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    fig_hero_diffuse()
    fig_cross_arm()
    fig_arm_a()
    fig_install()
    print("DONE figures/issue_653/")
