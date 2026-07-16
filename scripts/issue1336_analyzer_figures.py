#!/usr/bin/env python
"""Issue #1336 — analyzer figures (hero + key reads, savefig_paper conventions).

Hero: reparameterization gap per stage + the RLVR-vs-DPO contrast C, recal
primary / raw companion, headline layer 30. Key supporting reads: within-vs-
composition paired bars, the G1 raw-kill -> recalibrated re-adjudication
ladder, the lambda-floor interpolation degeneracy mechanism behind the
GSM8K-test cell, and the RLVR-long dose arm (secondary, never headline).

Inputs: eval_results/issue_1336/{cells,ladder_alignment,decision,diagnosis}.
Outputs: figures/issue_1336/ (PNG + PDF + meta.json via savefig_paper).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

EVAL = Path("eval_results/issue_1336")
FIGD = Path("figures/issue_1336")

EVAL_SETS = [
    ("gsm8k_train5k_chat", "GSM8K train\n(RLVR-trained distribution)"),
    ("lmsys5k_chat", "LMSYS chat"),
    ("lmsys5k_naturalistic", "LMSYS naturalistic"),
    ("gsm8k_test1319_chat", "GSM8K test\n(degenerate fit regime)"),
]
STAGES = [("sft", "SFT"), ("dpo", "DPO"), ("rlvr", "RLVR")]
HL = "30"


def _load(p: Path) -> dict:
    assert p.exists(), f"missing {p}"
    return json.loads(p.read_text())


def _err(entry: dict) -> tuple[float, float]:
    return (entry["point"] - entry["ci_lo"], entry["ci_hi"] - entry["point"])


def fig_hero(decision: dict) -> None:
    per = decision["per_eval_set"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    colors = paper_palette(3)

    ax = axes[0]
    width = 0.24
    xs = np.arange(len(EVAL_SETS))
    for si, (slug, lab) in enumerate(STAGES):
        pts, los, his = [], [], []
        for key, _ in EVAL_SETS:
            e = per[key]["gap_per_stage"][slug]
            pts.append(e["point"])
            los.append(e["point"] - e["ci_lo"])
            his.append(e["ci_hi"] - e["point"])
        ax.bar(
            xs + (si - 1) * width,
            pts,
            width,
            yerr=[los, his],
            capsize=2.5,
            color=colors[si],
            label=lab,
        )
        for x, p in zip(xs + (si - 1) * width, pts):
            if abs(p) > 0.01:
                ax.text(x, p + (0.006 if p > 0 else -0.012), f"{p:+.3f}", ha="center", fontsize=7)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.text(
        3.0,
        0.02,
        "within = composition\nto ~1e-5 (all stages)",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in EVAL_SETS], fontsize=8.5)
    ax.set_ylabel("Gap: within-stage R$^2$ $-$ base-composition R$^2$")
    ax.set_title("Gap per post-training stage (recalibrated scale, layer 30)", pad=12)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    xs2 = np.arange(len(EVAL_SETS))
    for oi, (scale_key, lab, filled) in enumerate(
        [
            ("contrast_C", "recalibrated (primary)", True),
            ("contrast_C_raw", "raw (companion)", False),
        ]
    ):
        pts, los, his = [], [], []
        for key, _ in EVAL_SETS:
            e = per[key][scale_key]
            pts.append(e["point"])
            los.append(e["point"] - e["ci_lo"])
            his.append(e["ci_hi"] - e["point"])
        off = (oi - 0.5) * 0.22
        ax.errorbar(
            xs2 + off,
            pts,
            yerr=[los, his],
            fmt="o" if filled else "s",
            color="#333333" if filled else "#888888",
            mfc="#333333" if filled else "none",
            capsize=3,
            lw=1.4,
            markersize=6,
            linestyle="none",
            label=lab,
        )
        for x, p in zip(xs2 + off, pts):
            ax.text(x + 0.06, p, f"{p:+.4f}", fontsize=7, va="center")
    ax.axhline(0.0, color="0.3", lw=0.8)
    band = decision["verdict_lattice"]["elicit_band"]
    ax.axhspan(-band, band, color="0.85", alpha=0.5, zorder=0)
    ax.text(0.02, band * 0.7, "elicitation band ($\\pm$0.020)", fontsize=7.5, color="0.4")
    ax.set_xticks(xs2)
    ax.set_xticklabels([lab for _, lab in EVAL_SETS], fontsize=8.5)
    ax.set_ylabel("Contrast C = gap(RLVR) $-$ gap(DPO)")
    ax.set_title("RLVR-specific contrast, 1000-draw paired bootstrap 95% CI", pad=12)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    savefig_paper(fig, "hero_rlvr_contrast", dir=FIGD)
    plt.close(fig)


def fig_within_vs_comp() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    colors = paper_palette(2)
    sets3 = EVAL_SETS[:3]
    for ai, (key, lab) in enumerate(sets3):
        corpus, fmt = key.rsplit("_", 1)
        ax = axes[ai]
        within, comp = [], []
        for slug, _ in STAGES:
            d = _load(EVAL / "ladder_alignment" / f"pair_base__{slug}_{fmt}_{corpus}.json")
            L = d["per_layer"][HL]
            within.append(L["within_r2_recal"])
            comp.append(L["comp_samefn_r2_recal"])
        xs = np.arange(3)
        ax.bar(xs - 0.18, within, 0.34, color=colors[0], label="within-stage map")
        ax.bar(xs + 0.18, comp, 0.34, color=colors[1], label="base map, reparameterized")
        for x, w, c in zip(xs, within, comp):
            ax.text(x - 0.18, w + 0.008, f"{w:.3f}", ha="center", fontsize=7.5)
            ax.text(x + 0.18, c + 0.008, f"{c:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([s for _, s in STAGES])
        ax.set_title(lab.replace("\n", " "), pad=10, fontsize=10)
        if ai == 0:
            ax.set_ylabel("Held-out recalibrated pooled R$^2$ (layer 30)")
            ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    savefig_paper(fig, "within_vs_comp_recal", dir=FIGD)
    plt.close(fig)


def fig_saga(verdict: dict, g1: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    li = verdict["lattice_inputs"]
    mech = verdict["mechanism_account"]
    vals = [
        mech["r2_v0_l29_recomputed"],
        verdict["per_layer"]["29"]["insample_recal_r2"],
        li["s_r"],
    ]
    labs = [
        "raw pooled R$^2$\n(G1 kill read)",
        "in-sample\nrecalibrated",
        "held-out cross-fitted\nrecalibrated (S_r)",
    ]
    cols = ["#b5443c", "#c99a3c", "#3c7fb5"]
    xs = np.arange(3)
    ax.bar(xs, vals, 0.55, color=cols)
    for x, v in zip(xs, vals):
        ax.text(x, v + (0.03 if v > 0 else -0.07), f"{v:+.3f}", ha="center", fontsize=9)
    ax.axhline(li["bar_r"], color="0.2", lw=1.1, ls="--")
    ax.text(
        2.35, li["bar_r"] + 0.02, f"usable-strength bar {li['bar_r']:.3f}", fontsize=8, ha="right"
    )
    ax.axhline(li["b_r"], color="0.5", lw=1.0, ls=":")
    ax.text(
        2.35,
        li["b_r"] - 0.09,
        f"permutation null band p97.5 = {li['b_r']:+.4f}",
        fontsize=8,
        ha="right",
    )
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("Pooled R$^2$, RLVR model, LMSYS chat, layer 29")
    ax.set_title(
        "G1 kill read is a per-dim gain miscalibration:\nheld-out recalibration recovers the map",
        pad=12,
        fontsize=10,
    )

    ax = axes[1]
    v = verdict["v_gate"]
    xs = np.arange(2)
    vals2 = [v["committed_anchor"], v["s_qwen_recal"]]
    ax.bar(xs, vals2, 0.5, color=["0.6", "#3c7fb5"])
    for x, val in zip(xs, vals2):
        ax.text(x, val + 0.008, f"{val:.4f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["Qwen anchor\n(raw, committed)", "Qwen under the\nrecalibrated DV"], fontsize=9
    )
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Pooled R$^2$ (Qwen2.5-7B-Instruct, layer 19)")
    ax.set_title(
        "Validate-before-use gate: the corrected DV\nis inert on healthy data (V-gate PASS)",
        pad=12,
        fontsize=10,
    )
    fig.tight_layout()
    savefig_paper(fig, "g1_saga_read_ladder", dir=FIGD)
    plt.close(fig)


def fig_lambda_degeneracy() -> None:
    import glob
    import os

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    corpus_color = {
        "gsm8k_test1319": "#b5443c",
        "gsm8k_train5k": "#3c7fb5",
        "lmsys5k": "#4b9b57",
    }
    seen = set()
    for f in sorted(glob.glob(str(EVAL / "cells" / "cells_*.json"))):
        d = _load(Path(f))
        name = os.path.basename(f)[6:-5]
        la = d["lambda_audit"]
        frac = la["n_at_low_edge"] / la["n_selected"]
        n = d["metadata"]["n"]
        corpus = d["cell"]["corpus"]
        matched = name.startswith("matchedn_")
        lab = corpus if corpus not in seen else None
        seen.add(corpus)
        ax.scatter(
            n,
            frac,
            s=48,
            marker="^" if matched else "o",
            facecolors="none" if matched else corpus_color[corpus],
            edgecolors=corpus_color[corpus],
            linewidths=1.4,
            label=lab,
        )
    ax.set_xlabel("kept rows n (fit sample size; d = 4096)")
    ax.set_ylabel("fraction of GCV fits at the $\\lambda$ grid floor (0.01)")
    ax.set_title(
        "Small-n cells collapse to the interpolation regime\n(open triangles = matched-n 1319-row refits)",
        pad=12,
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    ax.axvline(4096, color="0.4", lw=0.9, ls="--")
    ax.text(4096 * 1.01, 0.55, "n = d", fontsize=8, color="0.4", rotation=90)

    ax = axes[1]
    xs = np.arange(len(EVAL_SETS))
    for si, (slug, slab) in enumerate(STAGES):
        vals = []
        for key, _ in EVAL_SETS:
            corpus, fmt = key.rsplit("_", 1)
            d = _load(EVAL / "ladder_alignment" / f"pair_base__{slug}_{fmt}_{corpus}.json")
            vals.append(abs(d["per_layer"][HL]["gap_recal"]))
        ax.plot(
            xs + (si - 1) * 0.08, vals, "o", label=slab, markersize=7, color=paper_palette(3)[si]
        )
        for x, v in zip(xs + (si - 1) * 0.08, vals):
            ax.text(x + 0.06, v * 1.15, f"{v:.1e}" if v < 1e-3 else f"{v:.3f}", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in EVAL_SETS], fontsize=8)
    ax.set_ylabel("|gap| (recal, layer 30, log scale)")
    ax.set_title(
        "On GSM8K test every fit sits at the $\\lambda$ floor:\nthe composition reproduces the within map exactly",
        pad=12,
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    savefig_paper(fig, "lambda_floor_degeneracy", dir=FIGD)
    plt.close(fig)


def fig_dose() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sets3 = EVAL_SETS[:3]
    xs = np.arange(len(sets3))
    colors = paper_palette(2)
    for oi, (slug, lab) in enumerate(
        [("rlvr", "RLVR (Tulu-3, headline arm)"), ("rlvr_long", "longer RLVR (Tulu-3.1, dose arm)")]
    ):
        pts, los, his = [], [], []
        for key, _ in sets3:
            corpus, fmt = key.rsplit("_", 1)
            d = _load(EVAL / "ladder_alignment" / f"pair_base__{slug}_{fmt}_{corpus}.json")
            L = d["per_layer"][HL]
            pts.append(L["gap_recal"])
            los.append(L["gap_recal"] - L["gap_recal_bootstrap"]["ci_lo"])
            his.append(L["gap_recal_bootstrap"]["ci_hi"] - L["gap_recal"])
        off = (oi - 0.5) * 0.26
        ax.bar(xs + off, pts, 0.24, yerr=[los, his], capsize=3, color=colors[oi], label=lab)
        for x, p in zip(xs + off, pts):
            ax.text(x, p + (0.006 if p > 0 else -0.014), f"{p:+.3f}", ha="center", fontsize=7.5)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in sets3], fontsize=9)
    ax.set_ylabel("Reparameterization gap (recal, layer 30)")
    ax.set_title(
        "Longer-RLVR dose arm (descriptive; different kept-row set,\ndifferent $\\lambda$ regime on LMSYS)",
        pad=12,
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "rlvr_long_dose", dir=FIGD)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    decision = _load(EVAL / "decision" / "headline_contrast.json")
    verdict = _load(EVAL / "diagnosis" / "recal" / "recal_verdict.json")
    g1 = _load(EVAL / "gates" / "g1_gate.json")
    fig_hero(decision)
    fig_within_vs_comp()
    fig_saga(verdict, g1)
    fig_lambda_degeneracy()
    fig_dose()
    print("[analyzer-figs] done")


if __name__ == "__main__":
    main()
