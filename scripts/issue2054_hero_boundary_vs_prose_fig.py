"""Regenerate the #2054 boundary-vs-prose hero from committed ladder rows.

The original generating script (``issue2054_analyzer_figs.py``) never landed on
main — ``figures/issue_2054/hero_boundary_vs_prose.meta.json`` records it as a
dirty working-tree file at commit 230a8ea5. This minimal regen derives the same
32 points from the committed direct-transfer ladder rows (staged and merged by
``scripts/issue2054_fetch_ladder_rows.py``): one point per story target cell
(4 characters x 2 answer provenances x 2 story framings x 2 models), where

  x = mean rung-1 direct-transfer R^2 from a DIFFERENT story with the SAME
      answer-boundary form (same provenance + model),
  y = mean rung-1 direct-transfer R^2 from the SAME story prose across the
      answer-boundary swap (same provenance + model).

Verified to reproduce the banked meta.json points exactly (2026-08-20).

Usage:
    uv run python scripts/issue2054_hero_boundary_vs_prose_fig.py \
        [--out-dir figures/paper] [--style iclr]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    figsize_iclr_full,
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
LADDER_ROWS = Path("/tmp/issue2054_ladder_rows_merged.json")
STORY_FRAMINGS = {"attrib_quoted", "bare_label"}
MODEL_KEY = {"qwen2.5-7b": "base", "qwen2.5-7b-instruct": "instruct"}


def _parse(cell: str) -> dict | None:
    """Split a character-cell id into (who, provenance, framing, model)."""
    parts = cell.split("__")
    if not parts[0].startswith("char_") or len(parts) != 4:
        return None
    return {
        "who": parts[0][len("char_") :],
        "prov": parts[1],
        "framing": parts[2],
        "model": MODEL_KEY[parts[3]],
    }


def _rows() -> list[dict]:
    if not LADDER_ROWS.exists():
        subprocess.run(
            [sys.executable, str(REPO / "scripts/issue2054_fetch_ladder_rows.py")],
            check=True,
        )
    rows = json.loads(LADDER_ROWS.read_text())
    assert rows, "merged ladder row file is empty"
    return [r for r in rows if r["arm"] == "context"]


def compute_points(rows: list[dict]) -> list[dict]:
    """One (same-boundary, cross-boundary) mean-transfer point per story cell."""
    targets = sorted(
        {
            r["tgt"]
            for r in rows
            if (t := _parse(r["tgt"])) is not None and t["framing"] in STORY_FRAMINGS
        }
    )
    assert len(targets) == 32, len(targets)
    pts = []
    for tgt in targets:
        t = _parse(tgt)
        assert t is not None
        same_boundary, cross_boundary = [], []
        for r in rows:
            if r["tgt"] != tgt:
                continue
            s = _parse(r["src"])
            if s is None or s["model"] != t["model"] or s["prov"] != t["prov"]:
                continue
            if s["framing"] == t["framing"] and s["who"] != t["who"]:
                same_boundary.append(r["rungs"]["1_direct"])
            elif s["who"] == t["who"] and s["framing"] != t["framing"]:
                cross_boundary.append(r["rungs"]["1_direct"])
        assert same_boundary and cross_boundary, tgt
        pts.append(
            {
                "cell": tgt,
                **t,
                "x_same_boundary": float(np.mean(same_boundary)),
                "y_cross_boundary": float(np.mean(cross_boundary)),
            }
        )
    return pts


def fig_boundary_vs_prose(pts: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.72))
    lims = (-0.32, 0.42)
    ax.plot(lims, lims, ls="--", lw=0.8, color=paper_color("null"), zorder=1)
    ax.axhline(0, color="0.8", lw=0.6, zorder=1)
    ax.axvline(0, color="0.8", lw=0.6, zorder=1)
    for p in pts:
        c = paper_color(p["model"])
        filled = p["prov"] == "inserted"
        ax.scatter(
            p["x_same_boundary"],
            p["y_cross_boundary"],
            s=18,
            facecolor=c if filled else "none",
            edgecolors=c,
            linewidths=1.0,
            zorder=3,
        )
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.set_xlabel("transfer from a different story,\nsame boundary form (held-out $R^2$)")
    ax.set_ylabel("transfer across boundary form,\nsame story prose (held-out $R^2$)")
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                color=paper_color("instruct"),
                label="instruct",
                ms=5,
            ),
            Line2D([], [], marker="o", ls="none", color=paper_color("base"), label="base", ms=5),
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                mfc="none",
                mec="0.3",
                mew=1.0,
                label="model's own answer (open)",
                ms=5,
            ),
            Line2D(
                [],
                [],
                ls="--",
                lw=0.8,
                color=paper_color("null"),
                label="equal transfer",
            ),
        ],
        loc="upper left",
    )
    savefig_paper(fig, "c4_boundary_vs_prose", dir=out_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "figures/paper")
    ap.add_argument("--style", choices=("iclr",), default="iclr")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style(args.style)
    pts = compute_points(_rows())
    xs = [p["x_same_boundary"] for p in pts]
    ys = [p["y_cross_boundary"] for p in pts]
    print(
        f"32 story cells; median same-boundary {np.median(xs):.3f}, "
        f"median cross-boundary {np.median(ys):.3f}"
    )
    fig_boundary_vs_prose(pts, args.out_dir)
    print("DONE", args.out_dir)


if __name__ == "__main__":
    main()
