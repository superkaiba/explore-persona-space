"""#1979 — positive-only arms: three predictors against the leakage DV.

One row per behavior (casual writing, impoliteness, marker), one dot per
predictor: true answer similarity, mapped (through-map) predicted-answer
similarity, and context similarity. Positive-only training arms only, one per
behavior. Metric = the committed panel-centered cosine; the whitened and CSLS
variants are NOT drawn because neither flips a verdict here (largest single
effect is the marker mapped-answer read, +0.079 -> +0.392 whitened, still below
that arm's 0.407 band).

Each row carries its own selection-corrected permutation band as a vertical
tick (signed max over the enlarged candidate set, 20,000 draws).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BAND = REPO_ROOT / "eval_results/issue_1979/whiten_csls/enlarged_band.json"
FIG_DIR = REPO_ROOT / "figures/issue_1979"

ARMS = [
    ("casual writing", "cas-pers-po-lr1e5-s42"),
    ("impoliteness", "imp-pers-po-lr1e5-s42"),
    ("marker token", "mk-pers-po-lr5e6-s42"),
]
PREDS = [
    ("p2@raw", "true answer similarity", "#4C72B0"),
    ("p3b@raw", "mapped answer similarity", "#C44E52"),
    ("p1@raw", "context similarity", "#8172B2"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--band", type=Path, default=BAND)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args(argv)

    rows = {r["arm_id"]: r for r in json.loads(args.band.read_text())["rows"]}

    set_paper_style("iclr")
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ys = list(range(len(ARMS)))[::-1]
    off = [0.22, 0.0, -0.22]

    for y, (_, aid) in zip(ys, ARMS, strict=True):
        r = rows[aid]
        band = r["band_enlarged"]["p975_max_selected"]
        ax.plot([band, band], [y - 0.34, y + 0.34], "-", color="#666666", lw=1.0, zorder=2)
        for (key, _, col), dy in zip(PREDS, off, strict=True):
            ax.plot(
                [r["observed"][key]], [y + dy], "o", ms=4.6, color=col, markeredgewidth=0, zorder=4
            )

    ax.axvline(0.0, color="#BBBBBB", lw=0.6, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, _ in ARMS])
    ax.set_ylim(-0.6, len(ARMS) - 0.4)
    ax.set_xlabel("within-arm Spearman $\\rho$ against leakage (positive-only arms)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, ms=4.6, label=lab) for _, lab, c in PREDS
    ]
    handles.append(plt.Line2D([], [], color="#666666", lw=1.0, label="permutation band (97.5%)"))
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=6)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "c5_po_three_predictors", dir=args.out_dir)
    plt.close(fig)
    png = paths["png"]
    assert png.exists() and png.stat().st_size > 0, paths
    print(f"[fig] wrote {png} ({png.stat().st_size} bytes)")
    for lab, aid in ARMS:
        r = rows[aid]
        vals = "  ".join(f"{k.split('@')[0]}={r['observed'][k]:+.3f}" for k, _, _ in PREDS)
        print(f"  {lab:16s} {vals}  band={r['band_enlarged']['p975_max_selected']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
