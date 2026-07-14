"""Issue #1090 fu1-margin-qwen figure updates (same-issue follow-up consolidation).

Three figures folded into the clean-result after the fu1 round:

1. ``fu1_generator_contrast`` (NEW) -- left: Tier-2 install bars for the
   Claude-generated (c3) and Qwen-generated (c5) sycophancy organisms under
   the IDENTICAL fresh 300-token judge instrument (``judged_reads.json``);
   right: teacher-forced fixed-pool margin, base vs trained per cell
   (``c3_margin.json`` / ``c5_margin.json``).
2. ``fu1_contrast_per_question`` (NEW, per-unit companion) -- left:
   per-question trained rate, c3 vs c5 (fresh-300); right: per-context
   teacher-forced margin, base vs trained, both cells.
3. ``install_lift`` (REGENERATED) -- the right-hand margin-companion panel
   now carries the c3 fixed-pool margin delta computed in fu1
   (``c3_margin.json``); the left install bars are unchanged
   (closure-adjusted c3, structural formatting, judged impolite).

Run from the issue-1090-fu1 worktree root:
    uv run python scripts/issue1090_fu1_figures.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from issue1090_lowlevel_figures import CELL_LABELS  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval_results" / "issue_1090"
FU1 = EVAL / "fu1-margin-qwen"
FA = EVAL / "free_analysis"
FIGDIR = ROOT / "figures" / "issue_1090"
BAND = (0.6, 0.85)


def _read(p: Path) -> dict:
    """Load one JSON file."""
    with open(p) as f:
        return json.load(f)


def fig_generator_contrast() -> None:
    """NEW: c3-vs-c5 install bars (fresh-300 instrument) + fixed-pool margin bars."""
    reads = _read(FU1 / "judged_reads.json")
    m3 = _read(FU1 / "c3_margin.json")
    m5 = _read(FU1 / "c5_margin.json")
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    cols = paper_palette(3)

    ax = axes[0]
    cells = ["c3-sycophancy-claude", "c5-sycophancy-qwen"]
    w = 0.36
    for i, cell in enumerate(cells):
        for off, state, col in ((-w / 2, "base", cols[0]), (w / 2, "trained", cols[1])):
            rd = reads[f"{cell}__{state}"]
            lo, hi = rd["wilson95"]
            ax.bar(i + off, rd["rate"], width=w, color=col, label=state if i == 0 else None)
            ax.errorbar(
                i + off,
                rd["rate"],
                yerr=[[max(0.0, rd["rate"] - lo)], [max(0.0, hi - rd["rate"])]],
                fmt="none",
                ecolor="0.25",
                capsize=3,
                lw=1.2,
            )
            ax.text(i + off, rd["rate"] + 0.03, f"{rd['rate']:.3f}", ha="center", fontsize=8)
    ax.axhspan(BAND[0], BAND[1], color="0.92", zorder=0)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([CELL_LABELS[c] for c in cells], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("judged agreement rate (Tier 2, fresh 300-token judge)")
    ax.set_title("Install under the identical judge instrument; band shaded")
    ax.legend(frameon=False, loc="upper left")

    ax2 = axes[1]
    for i, (m, _cell) in enumerate(((m3, cells[0]), (m5, cells[1]))):
        for off, key, col in ((-w / 2, "margin_base", cols[0]), (w / 2, "margin_trained", cols[1])):
            v = m[key]
            ax2.bar(i + off, v, width=w, color=col)
            va = "bottom" if v >= 0 else "top"
            ax2.text(
                i + off,
                v + (0.02 if v >= 0 else -0.02),
                f"{v:+.3f}",
                ha="center",
                va=va,
                fontsize=8,
            )
        d = m["margin_delta"]
        ax2.text(
            i,
            max(0.0, m["margin_base"], m["margin_trained"]) + 0.14,
            f"delta {d:+.3f}",
            ha="center",
            fontsize=9,
        )
    ax2.axhline(0.0, color="0.2", lw=1.0)
    ax2.set_xticks(range(len(cells)))
    ax2.set_xticklabels([CELL_LABELS[c] for c in cells], fontsize=9)
    ax2.set_ylim(-1.0, 0.45)
    ax2.set_ylabel("teacher-forced fixed-pool margin (nats per token)")
    ax2.set_title("Companion: fixed +/- pool margin, base vs trained")
    savefig_paper(fig, "fu1_generator_contrast", dir=FIGDIR)
    plt.close(fig)


def fig_contrast_per_question() -> None:
    """NEW per-unit companion: per-question rates c3-vs-c5 + per-context margins."""
    reads = _read(FU1 / "judged_reads.json")
    m3 = _read(FU1 / "c3_margin.json")
    m5 = _read(FU1 / "c5_margin.json")
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    cols = paper_palette(3)

    ax = axes[0]
    x = np.asarray(reads["c3-sycophancy-claude__trained"]["per_question_rate"], dtype=float)
    y = np.asarray(reads["c5-sycophancy-qwen__trained"]["per_question_rate"], dtype=float)
    rng = np.random.default_rng(42)
    jx = x + rng.uniform(-0.012, 0.012, size=len(x))
    jy = y + rng.uniform(-0.012, 0.012, size=len(y))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1.0, ls="--")
    ax.scatter(jx, jy, s=42, color=cols[2], zorder=3)
    for i in range(len(x)):
        ax.text(jx[i] + 0.015, jy[i] + 0.012, f"q{i}", fontsize=6.5, color="0.35")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Claude-generated organism: per-question trained rate")
    ax.set_ylabel("Qwen-generated organism: per-question trained rate")
    ax.set_title("Per-question trained rates (20 eval questions; diagonal = equal)")

    ax2 = axes[1]
    for m, cell, col in (
        (m3, "c3-sycophancy-claude", cols[1]),
        (m5, "c5-sycophancy-qwen", cols[2]),
    ):
        base = m["per_context_margin"]["base"]
        tr = m["per_context_margin"]["trained"]
        qs = sorted(base.keys())
        bx = np.asarray([base[q] for q in qs])
        ty = np.asarray([tr[q] for q in qs])
        ax2.scatter(bx, ty, s=36, color=col, label=CELL_LABELS[cell].replace("\n", " "), zorder=3)
        for q, xv, yv in zip(qs, bx, ty, strict=False):
            ax2.text(
                xv + 0.008,
                yv + 0.008,
                q.replace("q00", "q").replace("q0", "q"),
                fontsize=6,
                color="0.4",
            )
    lims = (-1.35, 0.45)
    ax2.plot(lims, lims, color="0.6", lw=1.0, ls="--")
    ax2.set_xlim(*lims)
    ax2.set_ylim(*lims)
    ax2.set_xlabel("per-context margin, base (nats per token)")
    ax2.set_ylabel("per-context margin, trained (nats per token)")
    ax2.set_title("Per-context fixed-pool margins (above diagonal = margin rose)")
    ax2.legend(frameon=False, loc="upper left", fontsize=8)
    savefig_paper(fig, "fu1_contrast_per_question", dir=FIGDIR)
    plt.close(fig)


def fig_install_lift_with_c3_margin() -> None:
    """REGENERATE install_lift: margin panel gains the fu1 c3 fixed-pool delta."""
    installs = sorted((EVAL / "install").glob("*_install.json"))
    recs = [_read(p) for p in installs]
    recs = [r for r in recs if "trained" in r.get("reads", {}) and "base" in r.get("reads", {})]
    closure = _read(FA / "c3_dropclosure.json")
    m3 = _read(FU1 / "c3_margin.json")
    for r in recs:
        if r["cell"].startswith("c3-"):
            for state in ("base", "trained"):
                r["reads"][state] = closure["states"][state]["closure"]
            r["margin_delta"] = m3["margin_delta"]
            r["_margin_note"] = "fu1 fixed-pool"
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    ax = axes[0]
    xs = np.arange(len(recs))
    w = 0.36
    cols = paper_palette(3)
    for i, r in enumerate(recs):
        for off, state, col in ((-w / 2, "base", cols[0]), (w / 2, "trained", cols[1])):
            rd = r["reads"][state]
            lo, hi = rd["wilson95"]
            ax.bar(i + off, rd["rate"], width=w, color=col)
            ax.errorbar(
                i + off,
                rd["rate"],
                yerr=[[max(0.0, rd["rate"] - lo)], [max(0.0, hi - rd["rate"])]],
                fmt="none",
                ecolor="0.25",
                capsize=3,
                lw=1.2,
            )
            ax.text(i + off, rd["rate"] + 0.03, f"{rd['rate']:.2f}", ha="center", fontsize=7.5)
    band = recs[0].get("band")
    if band:
        ax.axhspan(band[0], band[1], color="0.92", zorder=0)
    labels = []
    for r in recs:
        lab = CELL_LABELS[r["cell"]]
        if r["cell"].startswith("c3-"):
            lab += "\n(closure-adjusted)"
        elif r["cell"].startswith("c1-"):
            lab += "\n(structural)"
        labels.append(lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("own-persona rate (Tier 2)")
    ax.set_title("Install: base (left) vs selected checkpoint (right); band shaded")
    ax2 = axes[1]
    for i, r in enumerate(recs):
        d = r.get("margin_delta")
        if d is None:
            continue
        ax2.bar(i, d, color=cols[2], width=0.55)
        note = r.get("_margin_note")
        txt = f"{d:+.2f}" + (f"\n({note})" if note else "")
        ax2.text(i, d + 0.01, txt, ha="center", fontsize=8)
    ax2.axhline(0.0, color="0.2", lw=1.0)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([CELL_LABELS[r["cell"]] for r in recs], fontsize=8)
    ax2.set_ylabel("tf-margin delta (trained - base)")
    ax2.set_title("Companion: teacher-forced fixed +/- margin delta")
    savefig_paper(fig, "install_lift", dir=FIGDIR)
    plt.close(fig)


if __name__ == "__main__":
    fig_generator_contrast()
    fig_contrast_per_question()
    fig_install_lift_with_c3_margin()
    print("fu1 figures written to", FIGDIR)
