"""#1768 revision-round figures (interp-critique round 1).

1. rb_specificity.png (NEW) — own-behavior r_B vs best OTHER behavior's r_B
   cosine with the on-policy panel write, per content arm at layer 19
   (the pre-registered §4.6 cross-behavior control, unreported in round 1).
2. d_vs_matched_text_shift.png (re-render) — identical data to round 1
   (fits/*.json, verified), with the x-axis label shortened so it no longer
   clips at the right edge (critique figure-hygiene item).

Colors match the round-1 behavior palette (colorblind: cas #0173B2,
imp #DE8F05, syc #029E73, mk #CC79A7). Saves via savefig_paper (blog style)
into figures/issue_1768/ at the MAIN checkout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, "/home/thomasjiralerspong/explore-persona-space/src")
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before matplotlib/numpy: shared-VM thread caps

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

RESULTS = Path(__file__).resolve().parents[1] / "eval_results" / "issue_1768"
FIGDIR = Path("/home/thomasjiralerspong/explore-persona-space/figures/issue_1768")
COLORS = {"cas": "#0173B2", "imp": "#DE8F05", "syc": "#029E73", "mk": "#CC79A7"}
NAMES = {
    "cas": "casual writing style",
    "imp": "impoliteness",
    "syc": "sycophancy",
    "mk": "marker token",
}


def main() -> None:
    set_paper_style("blog")
    reads = json.loads((RESULTS / "direction_reads.json").read_text())["reads"]
    arms = sorted({k.rsplit("_L", 1)[0] for k in reads})
    content = [a for a in arms if not a.startswith("mk-")]

    # ── 1. specificity scatter (content arms, L19, on-policy write) ──────────
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    for beh in ("cas", "imp", "syc"):
        xs, ys = [], []
        for a in [c for c in content if c.startswith(beh)]:
            r = reads[f"{a}_L19"]
            xs.append(r["races"]["r_B"]["cos_w"])
            ys.append(max(r["cross_behavior_rb_cos"].values()))
        ax.scatter(xs, ys, s=52, color=COLORS[beh], label=NAMES[beh], zorder=3)
    lim = (-0.15, 0.85)
    ax.plot(lim, lim, ls="--", lw=1.0, color="0.45", zorder=1)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("cosine(write, own behavior's read-out)")
    ax.set_ylabel("cosine(write, best OTHER behavior's read-out)")
    ax.set_title(
        "Read-out specificity of the on-policy write (52 content arms, layer 19)",
        pad=36,
        loc="left",
    )
    ax.legend(loc="upper left", frameon=True)
    savefig_paper(fig, "rb_specificity", dir=FIGDIR)
    plt.close(fig)

    # ── 2. D vs matched-text shift re-render (same data, unclipped label) ────
    fits = {}
    for p in (RESULTS / "fits").glob("*_L19.json"):
        d = json.loads(p.read_text())
        fits[d["arm_id"]] = d
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for beh in ("cas", "imp", "syc", "mk"):
        for method, (marker, fill) in {"lora": ("o", True), "ft": ("s", False)}.items():
            xs, ys = [], []
            for a, d in fits.items():
                is_ft = "-ft-" in a
                if a.startswith(beh) and ((method == "ft") == is_ft):
                    xs.append(d["decomposition_tf"]["mean_norm_total"])
                    ys.append(d["map_change"]["D"])
            if not xs:
                continue
            kw = (
                dict(color=COLORS[beh])
                if fill
                else dict(facecolors="none", edgecolors=COLORS[beh], linewidths=1.6)
            )
            ax.scatter(xs, ys, s=46, marker=marker, zorder=3, **kw)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], ls="", marker="o", color=COLORS[b], label=NAMES[b])
        for b in ("cas", "imp", "syc", "mk")
    ] + [
        Line2D([], [], ls="", marker="o", color="0.3", label="LoRA (filled)"),
        Line2D(
            [],
            [],
            ls="",
            marker="s",
            markerfacecolor="none",
            markeredgecolor="0.3",
            markeredgewidth=1.6,
            label="full fine-tune (open)",
        ),
    ]
    ax.axhline(0.0, ls="--", lw=1.0, color="0.45", zorder=1)
    ax.set_xlabel("matched-text answer-state shift at layer 19 (mean norm, fixed text)")
    ax.set_ylabel("map-change statistic D at layer 19")
    ax.set_title(
        "Map change tracks the weights-carried answer shift (all 72 arms, layer 19)",
        pad=36,
        loc="left",
    )
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9)
    savefig_paper(fig, "d_vs_matched_text_shift", dir=FIGDIR)
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
