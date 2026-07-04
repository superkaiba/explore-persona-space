"""Analyzer-pass figures for issue #931 (repo-root copies + two additions).

1. Copies the run-generated figures from the issue-931 worktree to the
   repo-root ``figures/issue_931/`` so main-pinned URLs resolve.
2. Regenerates ``delta_char`` with an un-clipped y-label + embedded points.
3. Adds the low-level per-unit views behind the delta_char aggregate:
   per-novel (Arm A, labeled) and per-story (Arm B) paired correct-vs-swap
   held-out R^2 at layer 19.

Run from repo root: ``uv run python scripts/issue931_analyzer_figures.py``
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parent.parent
WT_EVAL = REPO / ".claude/worktrees/issue-931/eval_results/issue_931"
WT_FIGS = REPO / ".claude/worktrees/issue-931/figures/issue_931"
OUT = REPO / "figures/issue_931"


def copy_run_figures() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(WT_FIGS.glob("*")):
        shutil.copy2(f, OUT / f.name)
        n += 1
    print(f"[i931-figs] copied {n} files from worktree figures")


def fig_delta_char() -> None:
    set_paper_style()
    pal = paper_palette(4)
    pts = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (arm, label) in enumerate([("armA", "Real novels"), ("armB", "Model-written stories")]):
        d = json.loads((WT_EVAL / f"delta_char_{arm}.json").read_text())
        y, lo, hi = d["delta_r2_char"], d["delta_ci_lo"], d["delta_ci_hi"]
        ax.errorbar(
            [i], [y], yerr=[[y - lo], [hi - y]], fmt="o", color=pal[i], capsize=4, markersize=8
        )
        ax.text(i + 0.06, y, f"{y:+.3f}", va="center", fontsize=11)
        pts.append(
            {
                "arm": label,
                "delta_r2_char": y,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_rows": d["n_rows"],
                "n_groups": d["n_groups"],
            }
        )
    ax.axhline(0.0, color="0.3", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real novels", "Model-written stories"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Character-identity gain in held-out $R^2$\n(correct $-$ swap, 95% CI)")
    savefig_paper(fig, "delta_char", dir=OUT)
    (OUT / "delta_char.meta.json").write_text(json.dumps({"points": pts}, indent=2))
    plt.close(fig)


def _paired(arm: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    w = json.loads((WT_EVAL / f"cells_{arm}_within.json").read_text())["per_group_r2_headline"]
    s = json.loads((WT_EVAL / f"cells_{arm}_swap.json").read_text())["per_group_r2_headline"]
    keys = sorted(set(w) & set(s))
    return keys, np.array([w[k] for k in keys]), np.array([s[k] for k in keys])


def fig_delta_char_per_novel() -> None:
    set_paper_style()
    pal = paper_palette(4)
    keys, w, s = _paired("armA")
    order = np.argsort(w - s)
    fig, ax = plt.subplots(figsize=(8, 10))
    for row, idx in enumerate(order):
        ax.plot([s[idx], w[idx]], [row, row], color="0.7", lw=1, zorder=1)
    ax.scatter(
        s[order], np.arange(len(keys)), color=pal[1], s=30, label="swapped pairing", zorder=2
    )
    ax.scatter(
        w[order], np.arange(len(keys)), color=pal[0], s=30, label="correct pairing", zorder=3
    )
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([keys[i] for i in order], fontsize=9)
    ax.axvline(0.0, color="0.3", lw=1)
    ax.set_xlabel("Per-novel held-out $R^2$ @ layer 19")
    ax.legend(loc="lower left")
    savefig_paper(fig, "delta_char_per_novel", dir=OUT)
    (OUT / "delta_char_per_novel.meta.json").write_text(
        json.dumps(
            {
                "note": (
                    "correct-pairing per-novel R2 from cells_armA_within (all 1982 rows); "
                    "swap from cells_armA_swap (1694 derangement-eligible rows) — subsets "
                    "differ slightly; the pooled delta_char_armA.json read matches subsets."
                ),
                "points": [
                    {"novel": k, "r2_correct": float(a), "r2_swap": float(b)}
                    for k, a, b in zip(keys, w, s)
                ],
            },
            indent=2,
        )
    )
    plt.close(fig)


def fig_delta_char_per_story() -> None:
    set_paper_style()
    pal = paper_palette(4)
    keys, w, s = _paired("armB")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    lim_lo = float(min(s.min(), w.min())) - 0.1
    lim_hi = float(max(s.max(), w.max())) + 0.1
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", color="0.6", lw=1)
    ax.scatter(s, w, s=18, color=pal[0], alpha=0.6)
    ax.set_xlabel("Per-story held-out $R^2$ @ layer 19, swapped pairing")
    ax.set_ylabel("Per-story held-out $R^2$ @ layer 19, correct pairing")
    frac = float((w > s).mean())
    ax.set_title(f"Model-written stories: correct beats swap in {frac:.0%} of {len(keys)} stories")
    savefig_paper(fig, "delta_char_per_story", dir=OUT)
    (OUT / "delta_char_per_story.meta.json").write_text(
        json.dumps(
            {
                "n_stories": len(keys),
                "frac_correct_gt_swap": frac,
                "points": [
                    {"story": k, "r2_correct": float(a), "r2_swap": float(b)}
                    for k, a, b in zip(keys, w, s)
                ],
            },
            indent=2,
        )
    )
    plt.close(fig)


if __name__ == "__main__":
    copy_run_figures()
    fig_delta_char()
    fig_delta_char_per_novel()
    fig_delta_char_per_story()
    print("[i931-figs] done ->", OUT)
