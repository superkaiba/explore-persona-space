"""Analyzer round-2 (v2 interpretation) figures for task #1947.

Folds the registered-frames run (frame_h3-20row, frame_h6-n-match) into two
paper-quality figures: the behavior-structured fixed-text write-displacement
alignment (with 20-row-precision stability overlay), and the n-matched
repeat-vs-single-visit top-1 share comparison. Reads ONLY committed artifacts.

Run from the issue-1947 worktree root:
    uv run python scripts/issue1947_analyzer_figures_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847): load_dotenv() setdefaults OMP/MKL/OPENBLAS/
# NUMEXPR_NUM_THREADS before matplotlib/numpy freeze their pools.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue1947_analyzer_figures import add_slug_key  # noqa: E402

AN = Path("eval_results/issue_1947/analysis")
A1 = AN / "analyzer_round1"
FIGDIR = Path("figures/issue_1947")

C_CAS = "#4477AA"  # casual writing style
C_IMP = "#EE7733"  # impolite
C_20ROW = "#555555"


def fig_h3_behavior_split() -> None:
    digest = json.load(open(A1 / "battery_digest.json"))
    frame = {
        (r["slug"], r["layer"]): r
        for r in json.load(open(AN / "frame_h3-20row.json"))["rows"]
        if "cos_20row_mean" in r
    }
    mani = json.load(open(AN / "verdict_manifest.json"))["content"]
    inband = {s for s, v in mani.items() if v["selection"]["in_band"]}
    rows = [
        r
        for r in digest
        if r["tree"] == "matched_text" and not r["slug"].startswith("mk-") and not r.get("missing")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0), sharey=True)
    for ax, L in zip(axes, (19, 25)):
        sub = sorted([r for r in rows if r["L"] == L], key=lambda r: (r["slug"][:3], r["slug"]))
        for i, r in enumerate(sub):
            color = C_CAS if r["slug"].startswith("cas") else C_IMP
            cov = r["delta"]["cov95"]
            ax.hlines([cov, -cov], i - 0.35, i + 0.35, color="grey", linestyle="--", linewidth=0.9)
            marker = "o" if r["slug"] in inband else "s"
            ax.scatter(i, r["delta"]["cos"], s=34, color=color, marker=marker, zorder=3)
            fr = frame.get((r["slug"], L))
            if fr:
                ax.scatter(
                    i,
                    fr["cos_20row_mean"],
                    s=30,
                    facecolors="none",
                    edgecolors=C_20ROW,
                    linewidths=1.2,
                    marker="o",
                    zorder=4,
                )
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([r["slug"].replace("-sv-", "·") for r in sub], rotation=80, fontsize=6.5)
        ax.set_title(f"layer {L}", fontsize=11)
    axes[0].set_ylabel("fixed-text cos(write direction, displacement)")
    handles = [
        plt.Line2D([], [], marker="o", color=C_CAS, linestyle="", label="casual writing style"),
        plt.Line2D([], [], marker="o", color=C_IMP, linestyle="", label="impolite"),
        plt.Line2D(
            [], [], marker="s", color="grey", linestyle="", label="closest-approach cell (squares)"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color=C_20ROW,
            markerfacecolor="none",
            linestyle="",
            label="20-row-precision mean (open)",
        ),
        plt.Line2D([], [], color="grey", linestyle="--", label="corpus-covariance null p95 (±)"),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right")
    fig.text(
        0.5,
        0.99,
        "Fixed-text write-displacement alignment is behavior-structured "
        "(full 300/1,200-row read filled; 20-row-precision mean open)",
        ha="center",
        va="top",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    add_slug_key(fig)
    savefig_paper(fig, "h3_fixed_text_behavior_split", dir=FIGDIR)
    plt.close(fig)


def fig_h6_n_matched() -> None:
    digest = json.load(open(A1 / "battery_digest.json"))
    frame = {
        (r["slug"], r["layer"]): r
        for r in json.load(open(AN / "frame_h6-n-match.json"))["rows"]
        if "top1_share_mean" in r
    }

    def bat(slug: str, layer: int) -> dict | None:
        for r in digest:
            if r["slug"] == slug and r["tree"] == "matched_text" and r["L"] == layer:
                return r
        return None

    pairs = [("imp-bare-con", "impolite bare"), ("imp-pers-con", "impolite persona")]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    xt, xl = [], []
    x = 0
    for base, lab in pairs:
        for L in (14, 19, 25):
            rep = bat(f"{base}-rep-s42", L)
            fr = frame.get((f"{base}-sv-s42", L))
            sv_full = bat(f"{base}-sv-s42", L)
            if not rep or not fr or not sv_full:
                continue
            draws = fr["top1_share_subsampled"]
            ax.scatter([x] * len(draws), draws, s=16, color="#BBBBBB", zorder=2)
            ax.hlines(
                fr["top1_share_mean"], x - 0.25, x + 0.25, color="black", linewidth=2.0, zorder=3
            )
            ax.scatter(
                x,
                sv_full["top1"],
                s=30,
                facecolors="none",
                edgecolors="#4477AA",
                linewidths=1.3,
                zorder=3,
            )
            ax.scatter(x, rep["top1"], s=52, color="#CC3311", marker="D", zorder=4)
            xt.append(x)
            xl.append(f"{lab}\nL{L}")
            x += 1
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=8.5)
    ax.set_ylabel("top-1 singular share (fixed-text stack, n=80 rows)")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="#BBBBBB",
            linestyle="",
            label="single-visit sibling, 80-row subsamples (8 draws)",
        ),
        plt.Line2D([], [], color="black", label="subsample mean"),
        plt.Line2D(
            [],
            [],
            marker="o",
            color="#4477AA",
            markerfacecolor="none",
            linestyle="",
            label="single-visit full 1,200-row read",
        ),
        plt.Line2D(
            [],
            [],
            marker="D",
            color="#CC3311",
            linestyle="",
            label="repeat-regime control (80 rows × 15 epochs)",
        ),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left")
    ax.set_title(
        "Repeat-regime top-1 share sits below the single-visit sibling at matched "
        "stack-n in 6/6 reads (controls under-installed — dose-unmatched)",
        fontsize=10.5,
    )
    fig.tight_layout()
    savefig_paper(fig, "h6_n_matched_top1", dir=FIGDIR)
    plt.close(fig)


def fig_prefix_replot() -> None:
    """Re-render the prefix-vs-context gate figure with plain-English panel titles.

    Replaces the follow-up-round driver render whose panel titles carried
    hypothesis-code tokens (clean-result-critic r1 item 2 fold-in); data is
    read from the committed prefix_reads.json.
    """
    d = json.load(open(AN / "prefix_reads.json"))
    cells = [c for c in d["cells"] if c["kind"] == "content"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for col, layer in enumerate((14, 19, 25)):
        ax = axes[0][col]
        for c in cells:
            if c["layer"] != layer:
                continue
            g = c["matched_text"]["h4_gate"]
            a, b = g["prefix_last"]["spearman_rho"], g["last_prompt"]["spearman_rho"]
            if a is None or b is None:
                continue
            ax.plot([0, 1], [a, b], "-", color="#4477AA", alpha=0.45, linewidth=1.0)
            ax.scatter([0, 1], [a, b], s=18, color="#4477AA", zorder=3)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axhspan(0.3, 0.7, color="#CCEECC", alpha=0.6, zorder=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["prefix-token\nsummary", "last-prompt-token\nsummary"], fontsize=9)
        ax.set_title(f"Gate rank correlation, fixed text - layer {layer}", fontsize=10)
        if col == 0:
            ax.set_ylabel("rank correlation (paired per cell)")
        ax.set_xlim(-0.35, 1.35)
    for col, layer in enumerate((14, 19, 25)):
        ax = axes[1][col]
        rng = np.random.default_rng(7)
        for xi, tree in enumerate(("matched_text", "onpolicy")):
            vals = [c[tree]["h3"]["cos_w_delta"] for c in cells if c["layer"] == layer]
            ax.scatter(
                [xi + rng.uniform(-0.12, 0.12) for _ in vals],
                vals,
                s=18,
                color="#EE7733" if tree == "matched_text" else "#4477AA",
                zorder=3,
            )
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["fixed text", "on-policy"], fontsize=9)
        ax.set_title(f"Write-displacement cosine - layer {layer} (summary-invariant)", fontsize=10)
        if col == 0:
            ax.set_ylabel("cos(write, displacement)")
        ax.set_xlim(-0.5, 1.5)
    fig.text(
        0.5,
        0.99,
        "Prefix-token vs last-prompt-token reads - same arms, same rows "
        "(34 content cells; on-policy prefix arm constant-input degenerate by design, not shown)",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    add_slug_key(fig)
    savefig_paper(fig, "prefix_vs_lastprompt_reads", dir=FIGDIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_h3_behavior_split()
    fig_h6_n_matched()
    fig_prefix_replot()
    print("v2 figures written to", FIGDIR)


if __name__ == "__main__":
    main()
