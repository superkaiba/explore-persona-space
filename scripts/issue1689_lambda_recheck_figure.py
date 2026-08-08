"""Issue #1689 follow-up figure: wider-lambda-ceilings re-check.

Reads eval_results/issue_1689/wider-lambda-ceilings/summary.json (from
scripts/issue1689_lambda_recheck.py) and renders one two-panel figure:

  fig10_lambda_recheck.png — left: per-cell-arm change in within-cell
  held-out R2 (wide 19-point grid minus the published 13-point grid)
  against the published ceiling, with the 0.02 mover bar; right: the
  fold-level lambda-selection histogram on both grids, marking the old
  (1e4) and new (1e7) grid ceilings.

Usage: uv run python scripts/issue1689_lambda_recheck_figure.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# CRITICAL: load_dotenv() BEFORE importing matplotlib / numpy — shared-VM
# thread caps (#847) freeze at first BLAS import.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "eval_results/issue_1689/wider-lambda-ceilings/summary.json"
PERCELL = REPO / "eval_results/issue_1689/wider-lambda-ceilings/percell_wide19"

MOVER_BAR = 0.02
LABEL_MIN = 0.017  # label the largest movers only (all still under the bar)

SHORT = {
    "assistant": "asst",
    "user_lmsys": "LMSYS",
    "user_haiku": "haiku",
    "user_onpolicy": "on-pol",
    "helios": "HELIOS",
    "wren": "Wren",
    "dana": "Dana",
}


def _short(cell: str) -> str:
    """Compact reader-facing label for a condition slug (matches fig9's map)."""
    for k, v in SHORT.items():
        if cell.startswith(k):
            suffix = cell[len(k) :].lstrip("_")
            suffix = {"naturalistic": "plain"}.get(suffix, suffix)
            return f"{v} {suffix}"
    return cell


def _load_points() -> list[dict]:
    """One row per cell-arm: published ceiling, wide-19 ceiling, delta, model, arm."""
    rows = []
    for f in sorted(PERCELL.glob("*.json")):
        d = json.loads(f.read_text())
        rows.append(
            {
                "model": "base" if d["model"] == "Qwen_Qwen2.5-7B" else "instruct",
                "cell": d["cell"],
                "arm": d["arm"],
                "published": d["published_r2"],
                "wide19": d["ceiling_r2_19"],
                "delta": d["delta_r2"],
            }
        )
    assert len(rows) == 84, len(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "figures/issue_1689"))
    args = ap.parse_args()

    summary = json.loads(SUMMARY.read_text())
    rows = _load_points()

    set_paper_style("blog")
    pal = paper_palette(6)
    arm_color = {"prefix": pal[0], "context": pal[1]}  # same arm colors as fig2
    model_marker = {"base": "o", "instruct": "s"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.4))

    # Panel 1: delta ceiling vs published ceiling, all 84 cell-arms.
    for model in ("base", "instruct"):
        for arm in ("prefix", "context"):
            sel = [r for r in rows if r["model"] == model and r["arm"] == arm]
            ax1.scatter(
                [r["published"] for r in sel],
                [r["delta"] for r in sel],
                marker=model_marker[model],
                s=40,
                color=arm_color[arm],
                alpha=0.8,
                label=f"{model}, {arm} (n={len(sel)})",
                zorder=3,
            )
    top = max(rows, key=lambda r: r["delta"])  # single largest mover, labeled at its point
    ax1.text(
        top["published"] + 0.02,
        top["delta"],
        f"{_short(top['cell'])} {top['arm']} ({top['model']}) {top['delta']:+.3f}",
        fontsize=7,
        va="center",
        zorder=4,
    )
    ax1.axhline(MOVER_BAR, color="#888888", lw=1.0, ls="--", zorder=2)
    ax1.axhline(-MOVER_BAR, color="#888888", lw=1.0, ls="--", zorder=2)
    ax1.axhline(0, color="#cccccc", lw=0.7, zorder=1)
    ax1.set_ylim(-0.024, 0.025)
    ax1.set_xlabel("published within-cell held-out R2 (13-point grid)")
    ax1.set_ylabel("change in ceiling R2 (19-point grid minus published)")
    ax1.set_title("ceiling change per cell-arm (84 of 84 inside the 0.02 bar)", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")

    # Panel 2: fold-level lambda-selection histogram, both grids.
    def _snap(k: str) -> float:
        """Snap a rounded lambda string (e.g. '3.16228') to its half-decade log10."""
        return round(math.log10(float(k)) * 2) / 2

    h13 = {_snap(k): v for k, v in summary["lambda_hist_published_13"].items()}
    h19 = {_snap(k): v for k, v in summary["lambda_hist_wide_19"].items()}
    assert sum(h13.values()) == 420 and sum(h19.values()) == 420, (h13, h19)
    grid = [x / 2 for x in range(-4, 15)]  # -2.0 .. 7.0 in 0.5-dex steps
    w = 0.22
    ax2.bar(
        [g - w / 2 for g in grid],
        [h13.get(g, 0) for g in grid],
        width=w,
        color=pal[2],
        label="published 13-point grid (ceiling 1e4)",
        zorder=3,
    )
    ax2.bar(
        [g + w / 2 for g in grid],
        [h19.get(g, 0) for g in grid],
        width=w,
        color=pal[3],
        label="wide 19-point grid (ceiling 1e7)",
        zorder=3,
    )
    ax2.axvline(4, color="#888888", lw=1.0, ls="--", zorder=2)
    ax2.axvline(7, color="#888888", lw=1.0, ls=":", zorder=2)
    ax2.set_xticks(range(-2, 8))
    ax2.set_xticklabels([f"1e{e}" for e in range(-2, 8)], fontsize=8)
    ax2.set_xlabel("inner-CV selected ridge lambda")
    ax2.set_ylabel("fold-fits selecting this lambda (of 420)")
    ax2.set_title(
        "lambda selection: 202 of 265 old-edge fold-fits move up, none above 1e5", fontsize=11
    )
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Wider ridge-lambda grid re-check (L19, 84 cell-arms): 0 movers at the 0.02 bar; "
        "0 fold-fits at the new 1e7 ceiling",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_paper(fig, "fig10_lambda_recheck", dir=args.out_dir)
    plt.close(fig)
    n_moved = summary["n_moved"]
    print(f"n_moved={n_moved}/84; max delta={max(r['delta'] for r in rows):+.6f}")
    return 0


if __name__ == "__main__":
    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGABRT pointer. All outputs are
    # flushed/closed before this point; atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
