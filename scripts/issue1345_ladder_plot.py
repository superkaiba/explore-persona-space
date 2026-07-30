#!/usr/bin/env python
"""Figures for the #1345 chat <-> no-template mapping-similarity ladder.

Renders held-out R^2 per ladder rung (strongest correction -> weakest) against
each direction's own within-regime ceiling and its matched-capacity null, and
marks the WEAKEST rung that reconciles the two maps (first rung whose R^2 is
within `--tol` of that direction's ceiling).

Hero  : figures/issue_1345/ladder_rungs/ladder_hero_context.png   (context arm)
Full  : figures/issue_1345/ladder_rungs/ladder_all_arms.png       (both arms)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# load_dotenv() BEFORE the plotting stack: matplotlib pulls in numpy/BLAS, whose
# thread pools bind from the env at import (the shared-VM thread caps, #847).
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    set_paper_style,
)

RUNG_LABEL = {
    "1_direct": "1 direct\ntransfer",
    "2_ctx_offset": "2 context\noffset",
    "3_ans_offset": "3 answer\noffset",
    "4_bias_refit": "4 bias\nrefit",
    "5_global_scale": "5 global\nscaling",
    "6_rotation": "6 rotation",
    "7_ctx_reparam": "7 context\nreparam",
    "8_ans_reparam": "8 answer\nreparam",
    "9_full_AMB": "9 full\nA·M·B",
}
DIRECTIONS = ("chat->no_template", "no_template->chat")
DIR_LABEL = {
    "chat->no_template": "chat map → no-template",
    "no_template->chat": "no-template map → chat",
}
MODEL_LABEL = {"instruct": "Qwen2.5-7B-Instruct", "pretrained": "Qwen2.5-7B (base)"}


def weakest_reconciling(d: dict, li: int, rungs: list[str], tol: float) -> str | None:
    """First (strongest) rung whose held-out R^2 reaches ceiling - tol."""
    ceil = d["ceiling_r2"][li]
    for r in rungs:
        if d["r2"][r][li] >= ceil - tol:
            return r
    return None


def _panel(ax, res: dict, li: int, rungs: list[str], colors, tol: float, title: str):
    xs = range(len(rungs))
    for j, dk in enumerate(DIRECTIONS):
        d = res[dk]
        ax.plot(
            xs,
            [d["r2"][r][li] for r in rungs],
            "o-",
            color=colors[j],
            lw=1.8,
            ms=5,
            label=DIR_LABEL[dk],
            zorder=3,
        )
        ax.axhline(d["ceiling_r2"][li], color=colors[j], ls="--", lw=1.2, alpha=0.75, zorder=2)
        ax.plot(
            xs,
            [d["null_r2"][r][li] for r in rungs],
            ":",
            color=colors[j],
            lw=1.0,
            alpha=0.55,
            zorder=1,
        )
        w = weakest_reconciling(d, li, rungs, tol)
        if w is not None:
            k = rungs.index(w)
            ax.plot(
                [k],
                [d["r2"][w][li]],
                marker="o",
                ms=14,
                mfc="none",
                mec=colors[j],
                mew=2.2,
                zorder=4,
            )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([RUNG_LABEL[r] for r in rungs], fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    # The prefix arm's uncorrected transfer runs to -1e4 while its ceiling is
    # ~0.13; on a linear axis that crushes every other rung onto y=0 and the
    # panel reads as empty. symlog keeps BOTH the catastrophic magnitude and
    # the near-ceiling rungs legible.
    lo = min(d["r2"][r][li] for dk in DIRECTIONS for d in [res[dk]] for r in rungs)
    if lo < -1.0:
        ax.set_yscale("symlog", linthresh=0.1, linscale=1.5)
        ax.set_title(title + "   [symlog y]", fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir", type=Path, default=_REPO_ROOT / "eval_results/issue_1345/ladder_rungs"
    )
    ap.add_argument("--fig-dir", type=Path, default=_REPO_ROOT / "figures/issue_1345/ladder_rungs")
    ap.add_argument(
        "--tol", type=float, default=0.01, help="R^2 slack counted as 'reaches ceiling'"
    )
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()
    colors = paper_palette(2)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT
    ).stdout.strip()

    loaded: dict = {}
    for f in sorted(args.in_dir.glob("ladder_rungs_*.json")):
        r = json.loads(f.read_text())
        loaded[(r["model"], r["arm"])] = r
    assert loaded, f"no ladder JSONs in {args.in_dir}"
    any_res = next(iter(loaded.values()))
    rungs = list(any_res["rung_order"])
    li = any_res["layers"].index(any_res["headline_layer"])
    n = any_res["metadata"]["n_matched_rows"]

    for name, arms in (
        ("ladder_hero_context", ["context"]),
        ("ladder_all_arms", ["context", "prefix"]),
    ):
        models = [m for m in ("instruct", "pretrained") if any(k[0] == m for k in loaded)]
        arms = [a for a in arms if any(k[1] == a for k in loaded)]
        if not arms:
            continue
        fig, axes = plt.subplots(
            len(arms), len(models), squeeze=False, figsize=(6.2 * len(models), 4.0 * len(arms))
        )
        for i, arm in enumerate(arms):
            for j, model in enumerate(models):
                res = loaded.get((model, arm))
                ax = axes[i][j]
                if res is None:
                    ax.set_visible(False)
                    continue
                _panel(ax, res, li, rungs, colors, args.tol, f"{MODEL_LABEL[model]} — {arm} arm")
                if j == 0:
                    ax.set_ylabel(r"held-out $R^2$ (layer 19)")
        axes[0][0].legend(fontsize=7, loc="lower right", frameon=True)
        fig.suptitle(
            "Which correction reconciles the chat and no-template context→answer maps?\n"
            f"dashed = that direction's own ceiling · dotted = matched-capacity null · "
            f"ring = weakest rung reaching ceiling (±{args.tol}) · n={n} matched rows",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        for ext in ("png", "pdf"):
            fig.savefig(args.fig_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        meta = {
            "commit": commit,
            "n_matched_rows": n,
            "tol": args.tol,
            "layer": any_res["headline_layer"],
            "cells": {
                f"{m}/{a}": {
                    dk: {
                        "ceiling": loaded[(m, a)][dk]["ceiling_r2"][li],
                        "weakest_rung": weakest_reconciling(
                            loaded[(m, a)][dk], li, rungs, args.tol
                        ),
                        "r2": {r: loaded[(m, a)][dk]["r2"][r][li] for r in rungs},
                        "null": {r: loaded[(m, a)][dk]["null_r2"][r][li] for r in rungs},
                    }
                    for dk in DIRECTIONS
                }
                for (m, a) in loaded
            },
        }
        (args.fig_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
        plt.close(fig)
        print(f"wrote {args.fig_dir / (name + '.png')}")


if __name__ == "__main__":
    main()
