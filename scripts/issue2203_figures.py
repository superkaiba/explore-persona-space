"""Figures for issue #2203 — assistant-axis activation capping position ladder.

Reads eval_results/issue_2203/full-rerun-bugfix/{phase2/phase2_ladder_results.json,
phase3_32b_judge.json, cjk_intrusion_stats.json} and renders the hero
position-ladder grid plus four companions to figures/issue_2203/full-rerun-bugfix/.

r1 C1: reads AND writes are pinned to the round's LABELED paths — this script
must never render the parent run's buggy rows into the corrected round's
figures. All numbers are read from the committed JSONs; the CJK-intrusion
fractions come from cjk_intrusion_stats.json (derived from the HF raw
completions). Run: uv run python scripts/issue2203_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847 shared-VM thread caps must bind BEFORE numpy/matplotlib freeze their BLAS
# pools at import (this figures script is a VM entrypoint).
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

# This script's tree (the issue-2203 worktree) carries the committed
# eval_results + figures for this branch; the main-checkout repo_root() is on
# `main` and does not have them.
ROOT = Path(__file__).resolve().parents[1]
# r1 C1: the ROUND's labeled outputs — never the parent's unlabeled buggy rows.
EVAL = ROOT / "eval_results" / "issue_2203" / "full-rerun-bugfix"


def _read_round_json(rel: str) -> dict:
    p = EVAL / rel
    if not p.exists():
        raise FileNotFoundError(
            f"{p} absent — the round's labeled judge/analysis outputs have not landed "
            "yet (r1 C1: figures never read the parent's unlabeled artifacts)"
        )
    return json.loads(p.read_text())


LAD = _read_round_json("phase2/phase2_ladder_results.json")["arms"]
P3 = _read_round_json("phase3_32b_judge.json")["arms"]
CJK = _read_round_json("cjk_intrusion_stats.json")

POSITIONS = ["prefix", "ctx", "allprompt", "alltoken"]
POS_LABEL = {
    "prefix": "Prefix end",
    "ctx": "Context vector",
    "allprompt": "All prompt",
    "alltoken": "All tokens",
}
INTERV = {
    "cap": ("Cap (assistant axis)", paper_palette_role("primary")),
    "axrep": ("Axis-component replace", paper_palette_role("accent")),
    "fullrep": ("Full-state replace", paper_palette_role("control")),
}
BASE_C = "#6b6b6b"
NULL_C = "#111111"


def harm_rate(arm: str) -> tuple[float, int]:
    h = LAD[arm]["harm"]
    return h["rate"], h["n_scored_items"]


def idloss_rate(arm: str) -> tuple[float, int]:
    r = LAD[arm]["assistantness_role_susc"]
    return r["identity_loss_rate"], r["n_scored_items"]


def jb_cjk_frac(arm: str) -> float:
    d = CJK["phase2"][arm]
    return d["jb_cjk_count"] / d["jb_n"]


def _errbar(rate: float, n: int) -> float:
    lo, hi = proportion_ci(rate, n)
    return max(rate - lo, hi - rate)


# ---------------------------------------------------------------- hero
def hero() -> None:
    set_paper_style("blog")
    fig, (ax_h, ax_i) = plt.subplots(2, 1, figsize=(7.2, 8.2), sharex=True)
    x = np.arange(len(POSITIONS))
    base_h, base_hn = harm_rate("baseline")
    base_i, base_in = idloss_rate("baseline")

    for ax, getter, base_val, base_n, ylab in (
        (ax_h, harm_rate, base_h, base_hn, "Jailbreak harmful-response rate (↓ safer)"),
        (ax_i, idloss_rate, base_i, base_in, "Assistant-identity-loss rate (↓ more stable)"),
    ):
        for pref, (label, color) in INTERV.items():
            ys, es, degen = [], [], []
            for pos in POSITIONS:
                arm = f"{pref}_{pos}"
                r, n = getter(arm)
                ys.append(r)
                es.append(_errbar(r, n))
                degen.append(jb_cjk_frac(arm) > 0.5)
            ys = np.array(ys)
            # filled = coherent output, hollow = >50% CJK-degenerate output
            ax.plot(x, ys, "-", color=color, lw=1.8, label=label, zorder=3)
            for xi, yi, ei, dg in zip(x, ys, es, degen):
                if dg:
                    ax.errorbar(
                        xi,
                        yi,
                        yerr=ei,
                        fmt="o",
                        mfc="white",
                        mec=color,
                        markeredgewidth=1.8,
                        ms=8,
                        ecolor=color,
                        capsize=3,
                        zorder=4,
                    )
                else:
                    ax.errorbar(xi, yi, yerr=ei, fmt="o", color=color, ms=7, capsize=3, zorder=4)
        ax.axhline(
            base_val,
            ls="--",
            color=BASE_C,
            lw=1.4,
            zorder=2,
            label=f"No-intervention baseline ({base_val:.3f})",
        )
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=-0.02)

    # random-null reference lines (harm panel: ctx-null + all-token-null)
    ax_h.axhline(
        harm_rate("cap_ctx_randnull")[0],
        ls=":",
        color=NULL_C,
        lw=1.3,
        label=f"Random-dir null, ctx ({harm_rate('cap_ctx_randnull')[0]:.3f})",
    )
    ax_h.axhline(
        harm_rate("cap_alltoken_randnull")[0],
        ls="-.",
        color=NULL_C,
        lw=1.3,
        label=f"Random-dir null, all-token ({harm_rate('cap_alltoken_randnull')[0]:.3f})",
    )
    ax_i.axhline(
        idloss_rate("cap_ctx_randnull")[0],
        ls=":",
        color=NULL_C,
        lw=1.3,
        label=f"Random-dir null, ctx ({idloss_rate('cap_ctx_randnull')[0]:.3f})",
    )
    ax_i.axhline(
        idloss_rate("cap_alltoken_randnull")[0],
        ls="-.",
        color=NULL_C,
        lw=1.3,
        label=f"Random-dir null, all-token ({idloss_rate('cap_alltoken_randnull')[0]:.3f})",
    )

    # keep the ctx random-null identity line (0.484) inside the panel — a legend
    # entry with a clipped line misleads (interpretation-critic round 1)
    ax_i.set_ylim(-0.02, 0.52)

    ax_i.set_xticks(x)
    ax_i.set_xticklabels([POS_LABEL[p] for p in POSITIONS])
    ax_i.set_xlabel("Intervention position along the input")
    ax_h.legend(loc="upper left", fontsize=7.5, ncol=1)
    ax_i.legend(loc="upper right", fontsize=7.5, ncol=1, frameon=True, framealpha=1.0)
    ax_h.set_title("Where along the input the assistant-axis cap is applied", loc="left")
    savefig_paper(
        fig, "issue_2203/full-rerun-bugfix/hero_position_ladder", dir=str(ROOT / "figures")
    )
    plt.close(fig)


# ------------------------------------------------ degradation mechanism
def degradation() -> None:
    set_paper_style("blog")
    arms = [
        "baseline",
        "cap_prefix",
        "cap_ctx",
        "cap_allprompt",
        "cap_alltoken",
        "cap_alltoken_randnull",
    ]
    labels = [
        "Baseline",
        "Cap prefix",
        "Cap context",
        "Cap all-prompt",
        "Cap all-token",
        "Random-dir\nall-token null",
    ]
    harm_all = [harm_rate(a)[0] for a in arms]
    cjk = [jb_cjk_frac(a) for a in arms]
    noncjk = []
    for a in arms:
        rr = CJK["phase2"][a]["harm_rate_noncjk"][0]
        noncjk.append(rr if rr is not None else np.nan)
    x = np.arange(len(arms))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x - w, harm_all, w, color=paper_palette_role("primary"), label="Harm rate, all rows")
    ax.bar(
        x,
        noncjk,
        w,
        color=paper_palette_role("accent"),
        label="Harm rate, coherent (non-CJK) rows only",
    )
    ax.bar(
        x + w,
        cjk,
        w,
        color=NULL_C,
        alpha=0.7,
        label="Fraction of completions containing CJK script",
    )
    for xi, v in zip(x, noncjk):
        if np.isnan(v):
            ax.text(xi, 0.02, "no\ncoherent\nrows", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(harm_rate("baseline")[0], ls="--", color=BASE_C, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Rate / fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        "With the bugs fixed, capping leaves harm at baseline and output CJK-free", loc="left"
    )
    savefig_paper(
        fig, "issue_2203/full-rerun-bugfix/degradation_mechanism", dir=str(ROOT / "figures")
    )
    plt.close(fig)


# ------------------------------------------------------- 32B anchor
def anchor() -> None:
    set_paper_style("blog")
    arms = ["baseline", "cap_alltoken", "cap_ctx"]
    labels = ["Baseline", "Cap all-token", "Cap context"]
    harm = [P3[a]["rate"] for a in arms]
    err = [_errbar(P3[a]["rate"], P3[a]["n_scored_items"]) for a in arms]
    cjk = [CJK["phase3"][a]["cjk_count"] / CJK["phase3"][a]["n"] for a in arms]
    x = np.arange(len(arms))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    ax1.bar(x, harm, w * 1.4, color=paper_palette_role("primary"), yerr=err, capsize=3)
    ax1.axhline(P3["baseline"]["rate"], ls="--", color=BASE_C, lw=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Jailbreak harmful-response rate (↓ safer)")
    ax1.set_title("Qwen-3-32B, Lu et al. published vectors", loc="left", fontsize=10)
    ax2.bar(x, cjk, w * 1.4, color=NULL_C, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Fraction of completions containing CJK script")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Same arms, CJK-intrusion view", loc="left", fontsize=10)
    savefig_paper(fig, "issue_2203/full-rerun-bugfix/anchor_32b", dir=str(ROOT / "figures"))
    plt.close(fig)


# ------------------------------------------------- capability guardrails
def capability() -> None:
    set_paper_style("blog")
    arms = [
        "baseline",
        "cap_ctx",
        "cap_allprompt",
        "cap_alltoken",
        "axrep_allprompt",
        "fullrep_alltoken",
        "cap_ctx_randnull",
        "cap_alltoken_randnull",
    ]
    labels = [
        "Baseline",
        "Cap\ncontext",
        "Cap\nall-prompt",
        "Cap\nall-token",
        "Axis-replace\nall-prompt",
        "Full-replace\nall-token",
        "Random-dir\ncontext",
        "Random-dir\nall-token",
    ]
    benches = [("gsm8k", "GSM8K"), ("ifeval", "IFEval"), ("mmlu_pro", "MMLU-Pro")]
    x = np.arange(len(arms))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    colors = [
        paper_palette_role("primary"),
        paper_palette_role("accent"),
        paper_palette_role("control"),
    ]
    for j, (key, blabel) in enumerate(benches):
        vals, errs = [], []
        for a in arms:
            b = LAD[a]["capability"][key]
            vals.append(b["acc"])
            lo, hi = b["ci95"]
            errs.append(max(b["acc"] - lo, hi - b["acc"]))
        ax.bar(x + (j - 1) * w, vals, w, color=colors[j], label=blabel, yerr=errs, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Accuracy (↑ better)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        "Only whole-state replacement wrecks capability; capping leaves it intact", loc="left"
    )
    savefig_paper(
        fig, "issue_2203/full-rerun-bugfix/capability_guardrails", dir=str(ROOT / "figures")
    )
    plt.close(fig)


# ------------------------------------------- identity-loss judge censoring
def censoring() -> None:
    set_paper_style("blog")
    arms = [
        "baseline",
        "cap_prefix",
        "cap_ctx",
        "cap_allprompt",
        "cap_alltoken",
        "axrep_allprompt",
        "fullrep_allprompt",
        "fullrep_alltoken",
        "cap_ctx_randnull",
        "cap_alltoken_randnull",
    ]
    labels = [
        "Baseline",
        "Cap\nprefix",
        "Cap\ncontext",
        "Cap\nall-prompt",
        "Cap\nall-token",
        "Axis-rep\nall-prompt",
        "Full-rep\nall-prompt",
        "Full-rep\nall-token",
        "Rand null\nctx",
        "Rand null\nall-token",
    ]
    scored = [LAD[a]["assistantness_role_susc"]["n_scored_items"] for a in arms]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    bars = ax.bar(x, scored, 0.6, color=paper_palette_role("primary"))
    for a, b in zip(arms, bars):
        if LAD[a]["assistantness_role_susc"]["n_scored_items"] < 200:
            b.set_color(paper_palette_role("control"))
    ax.axhline(250, ls="--", color=BASE_C, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Scoreable identity items (of 250)")
    ax.set_ylim(0, 265)
    ax.set_title(
        "Identity scoring is complete except at the two whole-state-replace arms", loc="left"
    )
    savefig_paper(fig, "issue_2203/full-rerun-bugfix/identity_censoring", dir=str(ROOT / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    hero()
    degradation()
    anchor()
    capability()
    censoring()
    print("wrote 5 figures to figures/issue_2203/full-rerun-bugfix/")
