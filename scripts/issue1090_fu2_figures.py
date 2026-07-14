"""Issue #1090 fu2-dose-extension figures (same-issue follow-up consolidation).

1. ``fu2_dose_curves`` — the epochs-6 Tier-1 dose ladders for the two
   sycophancy organisms (c3 Claude-data, c5 Qwen-data), band shaded, every
   rung labeled with its judged rate (the per-rung labels ARE the per-unit
   data), the dose-selected rung circled. Instrument-labeled: fu2 Tier-1
   rates are judged at max_tokens=300 (the fu1 seam) while the PARENT ladders
   ran the 64-token default, so the epochs-3 curves are NOT overlaid (not
   directly poolable).
2. ``fu2_install_c3`` — left: the c3 Tier-2 judged install (base vs trained,
   Wilson 95) at epochs 3 (closure + fresh-300 reads) vs epochs 6 (fresh-300,
   this round); right: the per-question companion — epochs-3 vs epochs-6
   trained rate under the SAME fresh 300-token instrument, one labeled point
   per eval question.

Run from the issue-1090-fu1 worktree root:
    uv run python scripts/issue1090_fu2_figures.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.artifacts.recipe import (  # noqa: E402
    JUDGED_RATE_BAND,
    select_dose_checkpoint,
)

ROOT = Path(__file__).resolve().parent.parent
FU2 = ROOT / "eval_results" / "issue_1090" / "fu2-dose-extension"
FIGDIR = ROOT / "figures" / "issue_1090"

PANELS = (
    ("c3-sycophancy-claude", "Sycophancy, Claude-generated data"),
    ("c5-sycophancy-qwen", "Sycophancy, Qwen-generated data"),
)


def _read(p: Path) -> dict:
    """Load one JSON file."""
    with open(p) as f:
        return json.load(f)


def fig_fu2_dose_curves() -> None:
    """Two-panel epochs-6 dose ladders, band shaded, per-rung rate labels."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    cols = paper_palette(3)
    for k, (slug, label) in enumerate(PANELS):
        ladder = _read(FU2 / slug / "fu2_ladder.json")
        rates_by_step = {int(s): float(r) for s, r in ladder["rates_by_step"].items()}
        sel = select_dose_checkpoint(rates_by_step, band=JUDGED_RATE_BAND)
        steps = sorted(rates_by_step)
        rates = [rates_by_step[s] for s in steps]
        ax = axes[k]
        ax.axhspan(JUDGED_RATE_BAND[0], JUDGED_RATE_BAND[1], color="0.92", zorder=0)
        ax.plot(steps, rates, "-o", ms=5, color=cols[k])
        for s, r in zip(steps, rates, strict=True):
            ax.text(s, r + 0.035, f"{r:.2f}", ha="center", va="bottom", fontsize=7.5, color="0.25")
        ax.plot([sel.step], [sel.rate], "o", ms=12, mfc="none", mec="0.1", mew=1.8, zorder=4)
        sel_note = (
            f"selected step {sel.step} (in band)"
            if sel.in_band
            else f"selected step {sel.step} (closest approach)"
        )
        ax.set_title(f"{label}\n{sel_note}", fontsize=10.5)
        ax.set_xlabel("optimizer step (epochs 6; checkpoints every 2 steps)")
        if k == 0:
            ax.set_ylabel("own-persona judged rate (Tier 1,\n300-token judge budget)")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(steps[::2])
    savefig_paper(fig, "fu2_dose_curves", dir=FIGDIR)


def fig_fu2_install_c3() -> None:
    """c3 install at epochs 3 vs epochs 6 (bars + Wilson 95) + per-question companion."""
    eval_root = ROOT / "eval_results" / "issue_1090"
    fu2 = _read(FU2 / "c3_install_fu2.json")
    closure = _read(eval_root / "free_analysis" / "c3_dropclosure.json")
    fu1_reads = _read(eval_root / "fu1-margin-qwen" / "judged_reads.json")
    reads = {
        "epochs 3\n(closure)": {s: closure["states"][s]["closure"] for s in ("base", "trained")},
        "epochs 3\n(fresh 300)": {
            s: fu1_reads[f"c3-sycophancy-claude__{s}"] for s in ("base", "trained")
        },
        "epochs 6\n(fresh 300)": {s: fu2["reads"][s] for s in ("base", "trained")},
    }
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ax = axes[0]
    cols = paper_palette(3)
    w = 0.36
    for i, rd in enumerate(reads.values()):
        for off, state, col in ((-w / 2, "base", cols[0]), (w / 2, "trained", cols[1])):
            r = rd[state]
            lo, hi = r["wilson95"]
            ax.bar(i + off, r["rate"], width=w, color=col, label=state if i == 0 else None)
            ax.errorbar(
                i + off,
                r["rate"],
                yerr=[[max(0.0, r["rate"] - lo)], [max(0.0, hi - r["rate"])]],
                fmt="none",
                ecolor="0.25",
                capsize=3,
                lw=1.2,
            )
            ax.text(i + off, r["rate"] + 0.02, f"{r['rate']:.3f}", ha="center", fontsize=8)
    band = fu2["band"]
    ax.axhspan(band[0], band[1], color="0.92", zorder=0)
    ax.set_xticks(range(len(reads)), list(reads))
    ax.set_ylabel("own-persona judged rate (Tier 2)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Sycophancy install, Claude-generated data; band shaded")

    ax = axes[1]
    e3 = fu1_reads["c3-sycophancy-claude__trained"]["per_question_rate"]
    e6 = fu2["reads"]["trained"]["per_question_rate"]
    ax.plot([0, 1], [0, 1], "-", color="0.7", lw=1)
    ax.plot(e3, e6, "o", ms=6, color=cols[2])
    for qi, (x, y) in enumerate(zip(e3, e6, strict=True)):
        ax.text(x + 0.012, y + 0.012, f"q{qi}", fontsize=7, color="0.35")
    ax.set_xlabel("per-question trained rate, epochs 3 (fresh 300)")
    ax.set_ylabel("per-question trained rate,\nepochs 6 (fresh 300)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Per-question companion; diagonal marks no change")
    savefig_paper(fig, "fu2_install_c3", dir=FIGDIR)


if __name__ == "__main__":
    fig_fu2_dose_curves()
    fig_fu2_install_c3()
    print(f"[fu2-figures] wrote {FIGDIR / 'fu2_dose_curves.png'} + fu2_install_c3.png")
