"""Issue #1090 low-level per-unit figures (v4 spec: per-unit data behind every aggregate).

Three figures, committed alongside the four aggregate figures from
``issue1090_figures.py``:

1. ``yield_per_question_all_cells`` -- per-question kept fraction dots for all
   6 datagen cells (the per-unit view behind ``hero_yield_vs_floor``).
2. ``install_per_question`` -- per-question Tier-2 rates, base vs trained, for
   the 3 trained cells (per-unit view behind ``install_lift``); c1 recomputed
   with the structural predicate, c2/c3 from the persisted per-item
   ``judge_raw.json`` (drop-never-coerce preserved).
3. ``dose_curves_labeled`` -- the Tier-1 dose curves with each rung's rate
   labeled (per-unit points labeled, v4 spec).

Run from the issue-1090 worktree root:
    uv run python scripts/issue1090_lowlevel_figures.py
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

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.artifacts.datagen import _STRUCTURAL_PREDICATES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval_results" / "issue_1090"
RUN = ROOT / "data" / "issue_1090" / "run"
FIGDIR = ROOT / "figures" / "issue_1090"

# Plain-English condition names only — never bare cell codes (C1/C2/...) on
# chart elements (project rule: opaque condition codes stay in the Repro footer).
CELL_LABELS = {
    "c1-formatting-claude": "formatting control\n(Claude, curated)",
    "c2-impolite-claude": "impolite\n(Claude, auto-gen)",
    "c3-sycophancy-claude": "sycophancy\n(Claude, neutral)",
    "c4-sycophancy_hardfact-claude": "sycophancy\n(Claude, wrong-fact)",
    "c5-sycophancy-qwen": "sycophancy\n(Qwen, neutral)",
    "c6-broad_em-claude": "broad misalignment\n(Claude, neutral)",
}
TRAINED_CELLS = ["c1-formatting-claude", "c2-impolite-claude", "c3-sycophancy-claude"]
THRESHOLD = 50  # graded judge keep threshold (behavior.threshold for all three cells)


def fig_yield_per_question() -> None:
    with open(EVAL / "yield_summary.json") as fh:
        y = json.load(fh)
    fig, ax = plt.subplots(figsize=(11, 5))
    rng = np.random.default_rng(42)
    for xi, (_cell, d) in enumerate(y.items()):
        pq = d["per_question_yield"]
        fracs = [v["kept"] / v["judged"] for v in pq.values() if v["judged"] > 0]
        n_zero_judged = sum(1 for v in pq.values() if v["judged"] == 0)
        jitter = rng.uniform(-0.18, 0.18, size=len(fracs))
        ax.scatter(xi + jitter, fracs, s=22, alpha=0.55, color="#2b6cb0", zorder=3)
        mean = d["kept"] / d["requested"]
        ax.hlines(mean, xi - 0.3, xi + 0.3, color="#c05621", lw=2.5, zorder=4)
        ax.text(
            xi,
            1.06,
            f"{len(fracs)} qs" + (f" (+{n_zero_judged} zero-judged)" if n_zero_judged else ""),
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(range(len(y)))
    ax.set_xticklabels([CELL_LABELS[c] for c in y], fontsize=8)
    ax.set_ylabel("per-question kept fraction (kept / judged requests)")
    ax.set_ylim(-0.05, 1.12)
    ax.set_title(
        "Per-question datagen yield behind the per-cell bars "
        "(dots: questions; orange line: cell kept fraction)",
        fontsize=11,
        pad=14,
    )
    savefig_paper(fig, "yield_per_question_all_cells", dir=FIGDIR)
    plt.close(fig)


def _tier2_per_question_judged(cell_tag: str) -> dict[int, tuple[int, int]]:
    """Per-question (k_pos, n_scored) from the persisted per-item judge_raw.json.

    Item ids look like ``c3-trained-q000-c0__00000__00`` (5 draws per
    completion). A completion's score is the mean of its parseable draws;
    completions with zero parseable draws are dropped (drop-never-coerce).
    """
    with open(RUN / "tier2_judge" / cell_tag / "judge_raw.json") as fh:
        raw = json.load(fh)
    per_comp: dict[tuple[int, int], list[float]] = {}
    pat = re.compile(r"-q(\d+)-c(\d+)__")
    for iid, rec in raw["all_scores"].items():
        if rec.get("error") or "score" not in rec:
            continue
        m = pat.search(iid)
        if not m:
            continue
        qi, ci = int(m.group(1)), int(m.group(2))
        per_comp.setdefault((qi, ci), []).append(float(rec["score"]))
    per_q: dict[int, tuple[int, int]] = {}
    for (qi, _ci), scores in per_comp.items():
        k, n = per_q.get(qi, (0, 0))
        mean = sum(scores) / len(scores)
        per_q[qi] = (k + (1 if mean > THRESHOLD else 0), n + 1)
    return per_q


def _tier2_per_question_structural(cell: str, side: str) -> dict[int, tuple[int, int]]:
    pred = _STRUCTURAL_PREDICATES["formatting"]
    f = RUN / "tier2" / cell / f"completions__{side}__persona_software_engineer.json"
    with open(f) as fh:
        d = json.load(fh)
    per_q: dict[int, tuple[int, int]] = {}
    for qi, comps in enumerate(d["completions"]):
        k = sum(1 for c in comps if pred(c))
        per_q[qi] = (k, len(comps))
    return per_q


def fig_install_per_question() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, cell in zip(axes, TRAINED_CELLS, strict=True):
        short = cell.split("-")[0]
        if cell == "c1-formatting-claude":
            base = _tier2_per_question_structural(cell, "base")
            trained = _tier2_per_question_structural(cell, "trained")
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
        ax.set_title(f"{CELL_LABELS[cell].replace(chr(10), ' ')} (n={len(qs)} qs)", fontsize=10)
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


def fig_dose_curves_labeled() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, cell in zip(axes, TRAINED_CELLS, strict=True):
        with open(EVAL / "install" / f"{cell}_dose_curve.json") as fh:
            d = json.load(fh)
        with open(EVAL / "install" / f"{cell}_install.json") as fh:
            inst = json.load(fh)
        base_rate = inst["reads"]["base"]["rate"]
        ax.axhline(base_rate, color="0.35", ls="--", lw=1.2, zorder=2)
        ax.annotate(
            f"base {base_rate:.2f}",
            (0.98, base_rate),
            xycoords=("axes fraction", "data"),
            fontsize=7.5,
            color="0.35",
            ha="right",
            va="bottom",
        )
        steps = [int(s) for s in d["rates_by_step"]]
        rates = list(d["rates_by_step"].values())
        ax.plot(steps, rates, marker="o", ms=5, lw=1.8, color="#2b6cb0", zorder=3)
        for s, r in zip(steps, rates, strict=True):
            ax.annotate(
                f"{r:.2f}",
                (s, r),
                fontsize=7,
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
            )
        sel = d["selection"]
        ax.scatter(
            [sel["step"]],
            [sel["rate"]],
            s=140,
            facecolors="none",
            edgecolors="black",
            linewidths=1.6,
            zorder=4,
        )
        ax.axhspan(d["band"][0], d["band"][1], color="#cccccc", alpha=0.4, zorder=1)
        ax.set_title(CELL_LABELS[cell].replace(chr(10), " "), fontsize=10)
        ax.set_xlabel("optimizer step (checkpoint every 2)")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("own-persona rate (Tier 1:\n5 completions per eval question)")
    fig.suptitle(
        "Dose curves with per-rung rates labeled (circle: selected rung; "
        "shaded: registered 0.60-0.85 band; dashed: own-persona base rate, Tier 2)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig_paper(fig, "dose_curves_labeled", dir=FIGDIR)
    plt.close(fig)


if __name__ == "__main__":
    set_paper_style("blog")
    fig_yield_per_question()
    fig_install_per_question()
    fig_dose_curves_labeled()
    print("done:", FIGDIR)
