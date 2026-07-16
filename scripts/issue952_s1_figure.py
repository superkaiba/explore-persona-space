"""#952 S1 figure for the Dan thread: the floored R² read vs the paired-target read.

Panel A: own-map -> Claude-target per-pair R² on the 12 S1 (Qwen-refuses/Claude-answers)
pairs, divergent vs matched control — the not-significantly-different, floored contrast.
Panel B: per-question cosine of the map's prediction to Qwen's actual refusal activation
vs to Claude's actual answer activation (train-mean-centered) — the read that resolves it.

Inputs (committed): eval_results/issue_952/refusal_sanity_check/behavior_differs_subset.json
(per_pair cross_r2_div/cross_r2_ctl, in_S1) and refusal_sanity.json
(check_B paired_closeness per_query cos_own/cos_claude). Writes
figures/issue_952/s1_floor_vs_paired.png. No bank text touched.
"""

from __future__ import annotations

# ruff: noqa: E402 — load_dotenv() must run before matplotlib import (shared-VM thread caps)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT = BASE / "figures" / "issue_952" / "s1_floor_vs_paired.png"


def main() -> None:
    """Build the two-panel figure; asserts n=12 in both panels and prints the counts."""
    subset = json.loads(
        (
            BASE / "eval_results/issue_952/refusal_sanity_check/behavior_differs_subset.json"
        ).read_text()
    )
    sanity = json.loads(
        (BASE / "eval_results/issue_952/refusal_sanity_check/refusal_sanity.json").read_text()
    )

    s1 = {pid: r for pid, r in subset["per_pair"].items() if r["in_S1"]}
    assert len(s1) == 12, len(s1)
    r2_div = np.array([r["cross_r2_div"] for r in s1.values()], dtype=float)
    r2_ctl = np.array([r["cross_r2_ctl"] for r in s1.values()], dtype=float)

    rows = sanity["check_B_map_predicts_refusal"]["paired_closeness_genuine_divergence"][
        "per_query"
    ]
    assert len(rows) == 12, len(rows)
    cos_own = np.array([r["cos_own"] for r in rows], dtype=float)
    cos_claude = np.array([r["cos_claude"] for r in rows], dtype=float)
    n_own_closer = int((cos_own > cos_claude).sum())
    print(f"panel A means: div {r2_div.mean():.3f} ctl {r2_ctl.mean():.3f}")
    print(
        f"panel B: own-closer {n_own_closer}/12; "
        f"mean cos own {cos_own.mean():.3f} vs claude {cos_claude.mean():.3f}"
    )

    set_paper_style()
    c_pair, c_mean = paper_palette(4)[0], paper_palette(4)[3]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # Panel A — paired slope plot, one light line per pair, bold means
    xs = np.array([0.0, 1.0])
    for d, c in zip(r2_div, r2_ctl, strict=True):
        ax_a.plot(xs, [d, c], color=c_pair, alpha=0.35, lw=1.2, marker="o", ms=3.5, zorder=2)
    ax_a.plot(xs, [r2_div.mean(), r2_ctl.mean()], color=c_mean, lw=2.6, marker="o", ms=8, zorder=3)
    ax_a.set_xticks(xs, ["divergent\n(Qwen refuses)", "matched control"])
    ax_a.set_xlim(-0.35, 1.35)
    ax_a.set_ylabel("per-pair R² (Qwen map → Claude answer)")
    ax_a.set_title("R² read: floored, no divergent penalty")

    # Panel B — per-question closeness scatter vs the identity diagonal
    lo = float(min(cos_own.min(), cos_claude.min())) - 0.05
    hi = float(max(cos_own.max(), cos_claude.max())) + 0.05
    ax_b.plot([lo, hi], [lo, hi], color="grey", lw=1.0, ls="--", zorder=1)
    closer_own = cos_own > cos_claude
    pal = paper_palette(4)
    ax_b.scatter(
        cos_claude[closer_own],
        cos_own[closer_own],
        s=45,
        color=pal[2],
        zorder=3,
        label=f"closer to Qwen refusal ({n_own_closer}/12)",
    )
    ax_b.scatter(
        cos_claude[~closer_own],
        cos_own[~closer_own],
        s=45,
        color=pal[1],
        zorder=3,
        label=f"closer to Claude answer ({12 - n_own_closer}/12)",
    )
    ax_b.set_xlabel("cos(prediction, Claude answer)")
    ax_b.set_ylabel("cos(prediction, Qwen refusal)")
    ax_b.set_title("Paired read: prediction matches Qwen")
    ax_b.set_xlim(lo, hi)
    ax_b.set_ylim(lo, hi)
    ax_b.set_aspect("equal")
    ax_b.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


def paired_closeness_figure() -> None:
    """Standalone paired-closeness figure (the 'now if instead' paragraph): per-question
    slope charts for cosine and R²-closeness of the map's prediction to Qwen's real
    refusal vs Claude's real answer (12 S1 questions). Writes s1_paired_closeness.png."""
    sanity = json.loads(
        (BASE / "eval_results/issue_952/refusal_sanity_check/refusal_sanity.json").read_text()
    )
    rows = sanity["check_B_map_predicts_refusal"]["paired_closeness_genuine_divergence"][
        "per_query"
    ]
    assert len(rows) == 12, len(rows)

    set_paper_style()
    pal = paper_palette(4)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    panels = [
        ("cos_claude", "cos_own", "cosine to target", "Cosine closeness"),
        ("r2close_claude", "r2close_own", "R²-closeness to target", "R²-closeness"),
    ]
    xs = np.array([0.0, 1.0])
    for ax, (k_cl, k_own, ylabel, title) in zip(axes, panels, strict=True):
        cl = np.array([r[k_cl] for r in rows], dtype=float)
        own = np.array([r[k_own] for r in rows], dtype=float)
        n_up = int((own > cl).sum())
        for c, o in zip(cl, own, strict=True):
            col = pal[2] if o > c else pal[1]
            ax.plot(xs, [c, o], color=col, alpha=0.55, lw=1.4, marker="o", ms=4, zorder=2)
        ax.plot(xs, [cl.mean(), own.mean()], color="black", lw=2.8, marker="o", ms=9, zorder=3)
        ax.set_xticks(xs, ["Claude's real\nanswer", "Qwen's real\nrefusal"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: {n_up}/12 favor Qwen")
        print(f"{title}: mean claude {cl.mean():.3f} -> own {own.mean():.3f}; up {n_up}/12")

    fig.suptitle(
        "Which target does the map's prediction resemble? "
        "(12 Qwen-refuses/Claude-answers questions)",
        y=1.02,
    )
    fig.tight_layout()
    out = BASE / "figures" / "issue_952" / "s1_paired_closeness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    if "--paired-only" in sys.argv:
        paired_closeness_figure()
    else:
        main()
