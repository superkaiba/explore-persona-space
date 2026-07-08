"""Issue #1090 free-analysis figure updates (Step 9a-ter fold, commits a11951d563/88ddcecd49).

Three figures folded into the clean-result after the zero-GPU round:

1. ``install_lift`` (REGENERATED) -- the sycophancy bars now show the
   closure-adjusted Tier-2 rates (every one of the 200 completions per arm
   scored after the truncation re-judge; ``c3_dropclosure.json``); the
   formatting (structural) and impolite (judged) bars and the tf-margin
   companion panel are unchanged.
2. ``formatting_judged_reread`` (NEW) -- the c1 judged re-read: judged dose
   curve over all 8 rungs (vs the structural curve) + Tier-2 judged vs
   structural base/trained bars (``c1_judged_reread.json``).
3. ``install_per_question`` (REGENERATED) -- the sycophancy panel now uses
   closure-adjusted per-question rates (complete 10-completion denominators
   from ``p4_states/c3-{base,trained}.json``); formatting/impolite panels
   are rebuilt from their unchanged production inputs.

Run from the issue-1090 worktree root:
    uv run python scripts/issue1090_freeanalysis_figures.py
"""

from __future__ import annotations

import json
import re
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

from issue1090_lowlevel_figures import (  # noqa: E402
    CELL_LABELS,
    TRAINED_CELLS,
    _tier2_per_question_judged,
    _tier2_per_question_structural,
)

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval_results" / "issue_1090"
FA = EVAL / "free_analysis"
FIGDIR = ROOT / "figures" / "issue_1090"
THRESHOLD = 50


def _read(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def fig_install_lift_closure() -> None:
    """Regenerate install_lift with closure-adjusted sycophancy bars."""
    installs = sorted((EVAL / "install").glob("*_install.json"))
    recs = [_read(p) for p in installs]
    recs = [r for r in recs if "trained" in r.get("reads", {}) and "base" in r.get("reads", {})]
    closure = _read(FA / "c3_dropclosure.json")
    for r in recs:
        if r["cell"].startswith("c3-"):
            for state in ("base", "trained"):
                r["reads"][state] = closure["states"][state]["closure"]
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
        ax2.text(i, d + 0.01, f"{d:+.2f}", ha="center", fontsize=8)
    ax2.axhline(0.0, color="0.2", lw=1.0)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([CELL_LABELS[r["cell"]] for r in recs], fontsize=8)
    ax2.set_ylabel("tf-margin delta (trained - base)")
    ax2.set_title("Companion: teacher-forced fixed +/- margin delta")
    savefig_paper(fig, "install_lift", dir=FIGDIR)
    plt.close(fig)


def fig_formatting_judged_reread() -> None:
    """NEW: the c1 judged re-read — judged dose curve + Tier-2 judged/structural bars."""
    c1 = _read(FA / "c1_judged_reread.json")
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    ax = axes[0]
    steps = [int(s) for s in c1["judged_dose_curve"]]
    jr = list(c1["judged_dose_curve"].values())
    sr = list(c1["structural_dose_curve_reread"].values())
    band = c1["meta"]["band"]
    ax.axhspan(band[0], band[1], color="0.92", zorder=0)
    ax.plot(steps, jr, marker="o", ms=5, lw=1.8, color="#2b6cb0", label="judged rate", zorder=3)
    for s, r in zip(steps, jr, strict=True):
        ax.annotate(
            f"{r:.2f}", (s, r), fontsize=7, xytext=(0, 6), textcoords="offset points", ha="center"
        )
    ax.plot(steps, sr, marker="s", ms=4, lw=1.4, color="#c05621", label="structural rate", zorder=3)
    jb = c1["tier2_install"]["judged_base"]["rate"]
    ax.axhline(jb, color="#2b6cb0", ls="--", lw=1.1, zorder=2)
    ax.annotate(
        f"judged base {jb:.2f}",
        (0.98, jb),
        xycoords=("axes fraction", "data"),
        fontsize=7.5,
        color="#2b6cb0",
        ha="right",
        va="bottom",
    )
    sel = c1["judged_selection"]
    ax.scatter([sel["step"]], [sel["rate"]], s=140, facecolors="none", edgecolors="black", zorder=4)
    ax.set_xlabel("checkpoint rung (optimizer step)")
    ax.set_ylabel("own-persona rate (Tier 1 rungs)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Formatting dose curve, judged vs structural; band shaded")
    ax.legend(fontsize=8.5, loc="upper left")
    ax2 = axes[1]
    cols = paper_palette(3)
    t2 = c1["tier2_install"]
    units = {"base": c1["units"]["tier2-base"], "trained": c1["units"]["tier2-trained"]}
    xs = np.arange(2)
    w = 0.36
    for off, state, col in ((-w / 2, "base", cols[0]), (w / 2, "trained", cols[1])):
        jd = t2[f"judged_{state}"]
        st = units[state]["structural"]
        for xi, rd in ((0, jd), (1, st)):
            lo, hi = rd["wilson95"]
            ax2.bar(xi + off, rd["rate"], width=w, color=col)
            ax2.errorbar(
                xi + off,
                rd["rate"],
                yerr=[[max(0.0, rd["rate"] - lo)], [max(0.0, hi - rd["rate"])]],
                fmt="none",
                ecolor="0.25",
                capsize=3,
                lw=1.2,
            )
            ax2.text(xi + off, rd["rate"] + 0.02, f"{rd['rate']:.2f}", ha="center", fontsize=8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(
        ["judged DV\n(300 completions each)", "structural DV\n(same completions)"], fontsize=9
    )
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("own-persona rate (Tier 2)")
    ax2.set_title("Tier-2 install read: base (left) vs trained (right), both DVs")
    savefig_paper(fig, "formatting_judged_reread", dir=FIGDIR)
    plt.close(fig)


def _c3_per_question_closure(state: str) -> dict[int, tuple[int, int]]:
    d = _read(FA / "p4_states" / f"c3-{state}.json")
    pat = re.compile(r"-q(\d+)-c(\d+)$")
    per_q: dict[int, tuple[int, int]] = {}
    for it in d["per_item"]:
        m = pat.search(it["item_id"])
        qi = int(m.group(1))
        k, n = per_q.get(qi, (0, 0))
        per_q[qi] = (k + (1 if it["closure_mean"] > THRESHOLD else 0), n + 1)
    return per_q


def fig_install_per_question_closure() -> None:
    """Regenerate install_per_question with closure-adjusted sycophancy panel."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, cell in zip(axes, TRAINED_CELLS, strict=True):
        short = cell.split("-")[0]
        suffix = ""
        if cell == "c1-formatting-claude":
            base = _tier2_per_question_structural(cell, "base")
            trained = _tier2_per_question_structural(cell, "trained")
        elif cell == "c3-sycophancy-claude":
            base = _c3_per_question_closure("base")
            trained = _c3_per_question_closure("trained")
            suffix = ", closure"
        else:
            base = _tier2_per_question_judged(f"{short}-base")
            trained = _tier2_per_question_judged(f"{short}-trained")
        qs = sorted(set(base) | set(trained))
        b = [base[q][0] / base[q][1] if q in base and base[q][1] else np.nan for q in qs]
        t = [
            trained[q][0] / trained[q][1] if q in trained and trained[q][1] else np.nan for q in qs
        ]
        ax.scatter(b, t, s=26, alpha=0.7, color="#2b6cb0", zorder=3)
        for q, (xb, yt) in enumerate(zip(b, t, strict=True)):
            if not (np.isnan(xb) or np.isnan(yt)):
                ax.annotate(
                    f"q{qs[q]}",
                    (xb, yt),
                    fontsize=6,
                    alpha=0.6,
                    xytext=(2, 2),
                    textcoords="offset points",
                )
        ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", zorder=1)
        ax.set_title(
            f"{CELL_LABELS[cell].replace(chr(10), ' ')} (n={len(qs)} qs{suffix})", fontsize=10
        )
        ax.set_xlabel("base per-question rate")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("trained per-question rate\n(selected checkpoint, Tier 2)")
    fig.suptitle(
        "Per-question install behind the cell-level bars (above dashed line = trained higher)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig_paper(fig, "install_per_question", dir=FIGDIR)
    plt.close(fig)


if __name__ == "__main__":
    fig_install_lift_closure()
    fig_formatting_judged_reread()
    fig_install_per_question_closure()
    print("done:", FIGDIR)
