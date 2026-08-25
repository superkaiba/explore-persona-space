"""Issue #2378 analyzer figures — own ceilings, transfer ladder, user panel, pooled arm, dual-metric.

Reads eval_results/issue_2378/{fits,ladder,pool,retrieval} JSONs (committed on the
issue-2378 branch) and writes blog-style figures to figures/issue_2378/ via
savefig_paper (PNG + PDF + meta.json sidecars with embedded per-point data).

Run from the issue-2378 worktree root:
    uv run python scripts/issue2378_analyzer_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps BEFORE any heavy import (#847; even credential-free
# plotters — matplotlib pulls numpy, whose BLAS pool freezes at import).
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parents[1]
ER = ROOT / "eval_results" / "issue_2378"
OUT = "issue_2378"

CELLS = [
    "chat",
    "plain_text",
    "storyq_astra",
    "storyq_helios",
    "storyq_wren",
    "storyq_dana",
    "storyq_vex",
    "chat_user_real",
]
TARGETS = [c for c in CELLS if c != "chat"]

LABELS = {
    "chat": "Chat template (train)",
    "plain_text": "Plain-text dialogue",
    "storyq_astra": "Story: Astra (AI assistant)",
    "storyq_helios": "Story: HELIOS (calm AI)",
    "storyq_wren": "Story: Wren (warm helper)",
    "storyq_dana": "Story: Dana (ordinary person)",
    "storyq_vex": "Story: Vex (villain)",
    "chat_user_real": "User turn (real human text)",
}
SHORT = {
    "chat": "Chat (train)",
    "plain_text": "Plain text",
    "storyq_astra": "Astra",
    "storyq_helios": "HELIOS",
    "storyq_wren": "Wren",
    "storyq_dana": "Dana",
    "storyq_vex": "Vex",
    "chat_user_real": "User (real)",
}
RUNG_LABELS = [
    "frozen map",
    "input mean shift",
    "output mean shift",
    "bias refit",
    "global scale",
    "output rotation",
    "input linear re-map",
    "output linear re-map",
    "full refit",
]

_pal = paper_palette_blog(8)
COLOR = {c: _pal[i] for i, c in enumerate(CELLS)}


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_fits() -> dict:
    fits = {}
    for c in CELLS:
        fits[c] = {
            "context": _load(ER / "fits" / f"{c}__context.json"),
            "prefix": _load(ER / "fits" / f"{c}__prefix.json"),
        }
    return fits


def load_ladder() -> dict:
    lad = {}
    for t in TARGETS:
        lad[t] = [_load(ER / "ladder" / f"chat_to_{t}__rung{r}.json") for r in range(1, 10)]
    return lad


def fig_own_ceilings(fits: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(CELLS))
    w = 0.38
    ctx = [fits[c]["context"]["pooled_r2"] for c in CELLS]
    pre = [fits[c]["prefix"]["pooled_r2"] for c in CELLS]
    null95 = [fits[c]["context"]["null"]["pooled_p95"] for c in CELLS]
    ax.bar(x - w / 2, ctx, w, color=[COLOR[c] for c in CELLS], label="context (full prompt)")
    ax.bar(
        x + w / 2,
        pre,
        w,
        color=[COLOR[c] for c in CELLS],
        alpha=0.35,
        label="prefix (before the question)",
    )
    # per-fold points on the context bars
    for i, c in enumerate(CELLS):
        folds = [f["r2"] for f in fits[c]["context"]["per_fold"]]
        ax.plot(
            np.full(len(folds), x[i] - w / 2),
            folds,
            marker="o",
            ls="none",
            ms=3.5,
            mfc="white",
            mec="black",
            markeredgewidth=0.8,
            zorder=5,
        )
        folds_p = [f["r2"] for f in fits[c]["prefix"]["per_fold"]]
        ax.plot(
            np.full(len(folds_p), x[i] + w / 2),
            folds_p,
            marker="o",
            ls="none",
            ms=3.5,
            mfc="white",
            mec="black",
            markeredgewidth=0.8,
            zorder=5,
        )
    ax.plot(
        x - w / 2,
        null95,
        marker="_",
        ms=14,
        ls="none",
        color="black",
        label="shuffled-answer null (95th pct)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in CELLS])
    ax.set_ylabel("held-out R² of the framing's own map")
    ax.legend(loc="upper right")
    set_title_subtitle(
        ax,
        "Every surviving framing is linearly mappable",
        "own context→answer GCV-ridge map per framing, 5 grouped folds, n=6,601 rows each",
    )
    savefig_paper(fig, f"{OUT}/own_ceilings", dir="figures/")
    plt.close(fig)


def fig_ladder_recovery(lad: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    rungs = np.arange(1, 10)
    clip_lo = -1.0
    for t in TARGETS:
        rec = [lad[t][r - 1]["recovery"]["point_pooled"] for r in range(1, 10)]
        lo = [lad[t][r - 1]["recovery"]["ci_lo"] for r in range(1, 10)]
        hi = [lad[t][r - 1]["recovery"]["ci_hi"] for r in range(1, 10)]
        rec_c = np.clip(rec, clip_lo, None)
        yerr = np.vstack(
            [
                np.clip(np.array(rec) - np.array(lo), 0, None),
                np.clip(np.array(hi) - np.array(rec), 0, None),
            ]
        )
        # errors on clipped points suppressed (they are far below the axis)
        yerr[:, rec_c > np.array(rec)] = 0.0
        ax.errorbar(
            rungs,
            rec_c,
            yerr=yerr,
            color=COLOR[t],
            marker="o",
            ms=4,
            lw=1.6,
            capsize=2,
            label=SHORT[t],
        )
        # mark clipped points with a down triangle at the clip line
        clipped = rec_c > np.array(rec)
        if clipped.any():
            ax.plot(
                rungs[clipped],
                np.full(clipped.sum(), clip_lo),
                marker="v",
                ls="none",
                ms=5,
                color=COLOR[t],
            )
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_ylim(clip_lo - 0.08, 1.12)
    ax.set_xticks(rungs)
    ax.set_xticklabels(RUNG_LABELS, rotation=20, ha="right")
    ax.set_ylabel("recovery: transfer R² / own ceiling R²")
    ax.legend(loc="lower right", ncols=2)
    set_title_subtitle(
        ax,
        "The chat-trained map transfers to no framing unchanged",
        "chat→target transfer, 9 adaptation rungs; triangles mark points below the axis (down to −4.7)",
    )
    savefig_paper(fig, f"{OUT}/hero_ladder_recovery", dir="figures/")
    plt.close(fig)


def fig_ladder_r2_points(lad: dict, fits: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.0), sharex=True, sharey=True)
    axes = axes.ravel()
    rungs = np.arange(1, 10)
    for k, t in enumerate(TARGETS):
        ax = axes[k]
        for r in range(1, 10):
            folds = [f["r2"] for f in lad[t][r - 1]["per_fold"]]
            ax.plot(
                np.full(len(folds), r),
                folds,
                marker="o",
                ls="none",
                ms=3,
                color=COLOR[t],
                alpha=0.75,
            )
        ceil = fits[t]["context"]["pooled_r2"]
        ax.axhline(ceil, color="black", lw=0.9, ls="--")
        ax.axhline(0.0, color="grey", lw=0.6)
        ax.set_title(SHORT[t], loc="left", fontsize=10)
        ax.set_ylim(-1.05, 0.7)
        ax.set_xticks(rungs)
    axes[-1].axis("off")
    axes[5].set_xlabel("adaptation rung (1 = frozen map, 9 = full refit)")
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("transfer R² (per fold)")
    fig.suptitle(
        "Per-fold transfer R² behind the recovery ladder (dashed line = own ceiling)",
        x=0.01,
        ha="left",
        fontweight="semibold",
    )
    savefig_paper(fig, f"{OUT}/ladder_r2_points", dir="figures/")
    plt.close(fig)


def fig_user_turn_panel(fits: dict) -> None:
    ratio = _load(ER / "fits" / "ratio" / "h4a_ceiling_ratio.json")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    names = ["Chat assistant turn", "User turn (context)", "User turn (prefix)"]
    vals = [
        fits["chat"]["context"]["pooled_r2"],
        fits["chat_user_real"]["context"]["pooled_r2"],
        fits["chat_user_real"]["prefix"]["pooled_r2"],
    ]
    folds = [
        [f["r2"] for f in fits["chat"]["context"]["per_fold"]],
        [f["r2"] for f in fits["chat_user_real"]["context"]["per_fold"]],
        [f["r2"] for f in fits["chat_user_real"]["prefix"]["per_fold"]],
    ]
    cols = [COLOR["chat"], COLOR["chat_user_real"], COLOR["chat_user_real"]]
    x = np.arange(3)
    bars = ax.bar(x, vals, 0.55, color=cols)
    bars[2].set_alpha(0.4)
    for i in range(3):
        ax.plot(
            np.full(len(folds[i]), x[i]),
            folds[i],
            marker="o",
            ls="none",
            ms=4,
            mfc="white",
            mec="black",
            markeredgewidth=0.9,
            zorder=5,
        )
    ax.axhspan(0.19, 0.25, color="grey", alpha=0.18, zorder=0)
    ax.axhline(
        fits["chat_user_real"]["context"]["null"]["pooled_p95"],
        color="black",
        lw=0.8,
        ls=":",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("held-out R² of the own map")
    set_title_subtitle(
        ax,
        "The user's next turn is weakly predictable, as at 7B",
        "grey band = the corrected 7B guarded reference (0.19–0.25); dotted line = shuffled null 95th pct",
    )
    savefig_paper(fig, f"{OUT}/user_turn_panel", dir="figures/")
    plt.close(fig)
    # keep the ratio in the sidecar via a small json next to figures
    _ = ratio


def fig_pooled_tiers(fits: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(CELLS))
    w = 0.38
    m0, m0lo, m0hi, k128 = [], [], [], []
    for c in CELLS:
        d = _load(ER / "pool" / f"{c}__context.json")
        rec0 = d["recovery"]["m0"]
        m0.append(rec0["point_pooled"])
        m0lo.append(rec0.get("ci_lo", np.nan))
        m0hi.append(rec0.get("ci_hi", np.nan))
        k128.append(d["recovery"]["m2_k128"]["point_pooled"])
    yerr = np.vstack([np.array(m0) - np.array(m0lo), np.array(m0hi) - np.array(m0)])
    ax.bar(
        x - w / 2,
        m0,
        w,
        color=[COLOR[c] for c in CELLS],
        yerr=np.nan_to_num(yerr),
        capsize=2,
        label="one shared map, no per-cell term",
    )
    ax.bar(
        x + w / 2,
        k128,
        w,
        color=[COLOR[c] for c in CELLS],
        alpha=0.45,
        label="shared map + rank-128 per-cell residual",
    )
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.axhline(0.9, color="black", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in CELLS])
    ax.set_ylabel("pooled-map R² / own-ceiling R²")
    ax.set_ylim(0, 1.35)
    ax.legend(loc="upper left", ncols=2)
    set_title_subtitle(
        ax,
        "One shared map serves every framing at or above its ceiling",
        "map fit jointly on all 8 framings, scored per framing; dashed = ceiling, dotted = the 90% bar",
    )
    savefig_paper(fig, f"{OUT}/pooled_tiers", dir="figures/")
    plt.close(fig)


def fig_pooled_points(fits: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(CELLS))
    for i, c in enumerate(CELLS):
        d = _load(ER / "pool" / f"{c}__context.json")
        pf = d["per_fold"]
        vals = [f["r2"]["m0"] for f in pf]
        ax.plot(np.full(len(vals), x[i] - 0.12), vals, marker="o", ls="none", ms=4, color=COLOR[c])
        ceil_f = [f["r2"] for f in fits[c]["context"]["per_fold"]]
        ax.plot(
            np.full(len(ceil_f), x[i] + 0.12),
            ceil_f,
            marker="D",
            ls="none",
            ms=4,
            mfc="white",
            mec=COLOR[c],
            markeredgewidth=1.1,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in CELLS])
    ax.set_ylabel("held-out R² (per fold)")
    set_title_subtitle(
        ax,
        "Per-fold R²: shared pooled map (filled) vs own map (open)",
        "5 folds per framing; the shared map matches or beats the own map in every story framing",
    )
    savefig_paper(fig, f"{OUT}/pooled_tiers_points", dir="figures/")
    plt.close(fig)


def fig_r2_vs_retrieval(fits: dict, lad: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    # own maps (stagger the tightly-clustered story-cell labels)
    label_dy = {
        "storyq_astra": 0.028,
        "storyq_helios": 0.010,
        "storyq_dana": -0.014,
        "storyq_wren": -0.038,
        "storyq_vex": -0.062,
    }
    for c in CELLS:
        r2 = fits[c]["context"]["pooled_r2"]
        ret = _load(ER / "retrieval" / f"{c}__context.json")["conventions"]["csls_cos"]["acc_at_k"][
            "1"
        ]
        ax.plot(r2, ret, marker="*", ms=13, ls="none", color=COLOR[c], markeredgewidth=0.0)
        ax.text(r2 + 0.014, ret + label_dy.get(c, 0.008), SHORT[c], fontsize=8)
    # ladder rungs
    for t in TARGETS:
        xs, ys = [], []
        for r in range(1, 10):
            r2 = lad[t][r - 1]["pooled_r2"]
            ret = _load(ER / "retrieval" / f"chat_to_{t}__rung{r}.json")["conventions"]["csls_cos"][
                "acc_at_k"
            ]["1"]
            xs.append(max(r2, -1.05))
            ys.append(ret)
        ax.plot(xs, ys, marker="o", ms=4, ls="none", color=COLOR[t], alpha=0.6)
    # shared frozen-map labels instead of 7 colliding per-series ones
    ax.text(-0.72, 0.075, "frozen chat map: story + user framings", fontsize=8, color="dimgrey")
    ax.text(
        -0.12,
        0.395,
        "frozen chat map: plain text",
        fontsize=8,
        color=COLOR["plain_text"],
        ha="right",
    )
    ax.set_xlabel("R² (own map: stars; transfer rungs: dots; clipped at −1.05)")
    ax.set_ylabel("rank-1 retrieval among held-out answers (CSLS cosine)")
    set_title_subtitle(
        ax,
        "Retrieval dissociates from R² on the frozen map",
        "each dot is one framing × adaptation rung; chance rank-1 = 0.00015 (n_pool = 6,601)",
    )
    savefig_paper(fig, f"{OUT}/r2_vs_retrieval", dir="figures/")
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    fits = load_fits()
    lad = load_ladder()
    fig_own_ceilings(fits)
    fig_ladder_recovery(lad)
    fig_ladder_r2_points(lad, fits)
    fig_user_turn_panel(fits)
    fig_pooled_tiers(fits)
    fig_pooled_points(fits)
    fig_r2_vs_retrieval(fits, lad)
    print("all figures written to figures/issue_2378/")


if __name__ == "__main__":
    main()
