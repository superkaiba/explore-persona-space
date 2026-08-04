"""Issue #1768 capture-noise-floor round figure.

Per-arm matched-text answer shift at layer 19 (all 72 arms, ranked, log
scale) against the operative fp16 storage floor. The measured replicate
floor is degenerate (bit-identical captures, floor_p95 exactly 0 on all
18 unit-layer cells — `eval_results/issue_1768/capture_noise_floor.json`),
so the drawn band is the fp16 quantization bound: two independently
stored copies of one vector differ by at most ~2^-10 of its norm
(~1e-3 relative), = 0.067 at the layer-19 median answer-span norm 68.49
(base replicate store). Dashed line: 2x that bound (the marker
falsification band from plan v7 §4.4).

Colors match the round-1 behavior palette (colorblind: cas #0173B2,
imp #DE8F05, syc #029E73, mk #CC79A7). Saves via savefig_paper (blog
style). Usage:

    uv run python scripts/issue1768_noise_floor_fig.py [--figdir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

COLORS = {"cas": "#0173B2", "imp": "#DE8F05", "syc": "#029E73", "mk": "#CC79A7"}
NAMES = {
    "cas": "casual writing style",
    "imp": "impoliteness",
    "syc": "sycophancy",
    "mk": "marker token",
}
# fp16 pair bound at layer 19: 2^-10 x median answer-span norm 68.49
# (base replicate store, noise_floor/base_content/pooled_nf_r1.pt).
FP16_BOUND_L19 = 68.49 * 2**-10
# The 5 trained units whose replicate floor was measured (all bit-zero).
FLOOR_UNITS = [
    "cas-pers-con-lr1e5-s42",
    "imp-pers-con-lr3e5-s42",
    "syc-pers-con-lr1e5-s42",
    "mk-pers-con-lr5e6-s42",
    "imp-pers-ft-con-s42",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="figures/issue_1768")
    ap.add_argument("--fitsdir", default="eval_results/issue_1768/fits")
    args = ap.parse_args()

    shifts: dict[str, float] = {}
    for f in glob.glob(f"{args.fitsdir}/*_L19.json"):
        d = json.load(open(f))
        shifts[d["arm_id"]] = d["decomposition_tf"]["mean_norm_total"]
    assert len(shifts) == 72, f"expected 72 L19 fit cells, found {len(shifts)}"

    ranked = sorted(shifts.items(), key=lambda kv: kv[1])
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.set_yscale("log")

    for beh in ("cas", "imp", "syc", "mk"):
        for ft in (False, True):
            pts = [
                (i + 1, v)
                for i, (a, v) in enumerate(ranked)
                if a.startswith(beh) and (("-ft-" in a) == ft)
            ]
            if not pts:
                continue
            xs, ys = zip(*pts)
            style = (
                dict(facecolors="none", edgecolors=COLORS[beh], linewidths=1.6)
                if ft
                else dict(color=COLORS[beh])
            )
            ax.scatter(xs, ys, s=46, zorder=3, **style)

    # Ring + slug label on the 5 floor-measured trained units.
    for i, (a, v) in enumerate(ranked):
        if a in FLOOR_UNITS:
            ax.scatter(
                [i + 1], [v], s=170, facecolors="none", edgecolors="0.15", linewidths=1.3, zorder=4
            )
            ax.text(
                i + 1, v * 1.35, a, rotation=55, ha="left", va="bottom", fontsize=7.5, color="0.15"
            )

    ax.axhspan(1e-3, FP16_BOUND_L19, color="0.55", alpha=0.25, zorder=1)
    ax.axhline(2 * FP16_BOUND_L19, ls="--", lw=1.1, color="0.4", zorder=1)
    ax.set_ylim(0.03, 90)
    ax.set_xlim(0, 74)
    ax.set_xlabel("Arms ranked by matched-text answer shift (layer 19)")
    ax.set_ylabel("Matched-text answer shift (activation norm units, log scale)")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], ls="", marker="o", color=COLORS[b], label=NAMES[b])
        for b in ("cas", "imp", "syc", "mk")
    ] + [
        Line2D([], [], ls="", marker="o", color="0.3", label="LoRA (filled)"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="0.3",
            color="none",
            label="full fine-tune (open)",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="0.15",
            markersize=11,
            color="none",
            label="replicate floor measured (ring)",
        ),
        Patch(facecolor="0.55", alpha=0.25, label="fp16 storage floor (bound)"),
        Line2D([], [], ls="--", color="0.4", label="2x floor (falsification band)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True)
    ax.set_title("Every arm's matched-text shift clears the fp16 storage floor")

    out = savefig_paper(fig, "capture_noise_floor_l19", dir=args.figdir)
    print({k: str(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
