"""#1979 inline round `whiten-csls-sweep` — figure for the metric sweep.

Three panels (content minus sycophancy / all content / marker); per panel one row
per re-metricized predictor and one dot per metric setting. Reads the sweep JSON
written by ``issue1979_whiten_csls_sweep.py``; renders nothing interpretive onto
the canvas (axes, ticks, legend, panel titles only).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP = REPO_ROOT / "eval_results/issue_1979/whiten_csls/sweep.json"
FIG_DIR = REPO_ROOT / "figures/issue_1979"

SETTINGS = ("raw", "whiten", "csls", "both")
SETTING_LABEL = {
    "raw": "centered cosine",
    "whiten": "whitened",
    "csls": "CSLS",
    "both": "whitened + CSLS",
}
PRED_LABEL = {
    "p1": "context similarity",
    "p2": "answer similarity",
    "p3a": "through-map context sim",
    "p3b": "through-map predicted-answer sim",
    "p9": "nearest training rows (context)",
    "p10": "nearest training rows (answer)",
}
PANELS = (
    (
        "casual + impoliteness (8 arms)",
        lambda r: r["kind"] == "content" and not r["arm_id"].startswith("syc"),
    ),
    ("all content (12 arms)", lambda r: r["kind"] == "content"),
    ("marker (6 arms)", lambda r: r["kind"] == "marker"),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", type=Path, default=SWEEP)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args(argv)

    d = json.loads(args.sweep.read_text())
    recs = d["records"]
    preds = list(d["config"]["remetricized"])

    set_paper_style("iclr")
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), sharex=True, sharey=True)
    ys = list(range(len(preds)))[::-1]

    for ax, (title, filt) in zip(axes, PANELS, strict=True):
        for si, s in enumerate(SETTINGS):
            xs, yy = [], []
            for y, p in zip(ys, preds, strict=True):
                v = [
                    r["rho"] for r in recs if r["setting"] == s and r["predictor"] == p and filt(r)
                ]
                if v:
                    xs.append(st.median(v))
                    yy.append(y + (si - 1.5) * 0.16)
            ax.plot(
                xs,
                yy,
                "o",
                ms=3.4,
                color=colors[si],
                markeredgewidth=0,
                label=SETTING_LABEL[s] if ax is axes[0] else None,
                zorder=3,
            )
        ax.axvline(0.0, color="#999999", lw=0.6, zorder=1)
        for y in ys:
            ax.axhline(y, color="#EEEEEE", lw=0.5, zorder=0)
        ax.set_title(title, fontsize=7)
        ax.set_ylim(-0.7, len(preds) - 0.3)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([PRED_LABEL[p] for p in preds], fontsize=6.5)
    for ax in axes:
        ax.set_xlabel("median within-arm Spearman $\\rho$", fontsize=7)
    axes[0].legend(frameon=False, loc="lower left", fontsize=6)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "c5_whiten_csls_sweep", dir=args.out_dir)
    plt.close(fig)
    png = paths["png"]
    assert png.exists() and png.stat().st_size > 0, paths
    print(f"[fig] wrote {png} ({png.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
