"""Issue #825 `sampled-separator-control` Phase-C figures (analyzer re-fold pass).

Two figures over the Phase-C VM legs (plan v22 R4 + the plan-section-6
distribution-matched arm-B refit):

- ``figures/issue_825/sampled_sep_transfer_fractions``: per model, the
  recentered sep->chat transfer at L19 as a fraction of the FULL-N chat
  ceiling — exogenous reference, greedy round 7, sampled arm B, averaged
  ten-draw read — each round-8 leg with its shuffle-null p97.5 tick; the
  0.5 specificity line dashed. The base round-7 bar (-4.30) clips at the
  panel floor (caption carries the value).
- ``figures/issue_825/sampled_sep_distmatch_contrast``: per model, D for
  the full-n sampled arm B vs the span-length-matched refit vs the
  n-matched random control (5 per-seed MLP-carried D points overlaid),
  against the round-7 greedy D reference line.

Usage:
  uv run python scripts/issue825_sampled_sep_phasec_figures.py \
      [--out-dir eval_results/issue_825/sampled-separator-control] [--fig-dir figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ("base", "instruct")
# Committed exogenous sep->chat fractions of the full-n ceiling (round-6/7
# decision constants, quoted in the clean-result body).
EXO_FRACTION = {"base": 0.0574, "instruct": 0.1087}
CEILING = {"base": 0.5876803039140281, "instruct": 0.6730919508995763}
R7_OUT = Path("eval_results/issue_825/onpolicy-separator-control")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("eval_results/issue_825/sampled-separator-control")
    )
    ap.add_argument("--r7-out-dir", type=Path, default=R7_OUT)
    ap.add_argument("--fig-dir", type=Path, default=Path("figures"))
    ap.add_argument("--fig-prefix", default="issue_825")
    return ap.parse_args()


def _read(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    fig_dir = args.fig_dir / args.fig_prefix
    fig_dir.mkdir(parents=True, exist_ok=True)
    c_arm = paper_palette_role("primary")
    c_ref = paper_palette_role("baseline")
    c_acc = paper_palette_role("accent")
    c_grey = paper_palette_role("neutral")

    dec = _read(args.out_dir / "decision_support.json")

    # ---- Figure A: transfer fractions vs the 0.5 specificity line ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), layout="constrained")
    for ax, m in zip(axes, MODELS, strict=True):
        ceil = CEILING[m]
        r7 = _read(args.r7_out_dir / f"onpolicy_sep_to_chat_{m}.json")["sep_to_chat"]
        legs = {
            arm: _read(args.out_dir / "transfer" / f"sampled_sep_to_chat_{m}_{arm}.json")[
                "sep_to_chat"
            ]
            for arm in ("armB", "armC_avg")
        }
        labels = [
            "exogenous\n(reference)",
            "greedy\n(round 7)",
            "arm B\n(sampled, 1 draw)",
            "C-avg\n(E[v] ceiling read)",
        ]
        fracs = [
            EXO_FRACTION[m],
            r7["r2"] / ceil,
            legs["armB"]["r2"] / ceil,
            legs["armC_avg"]["r2"] / ceil,
        ]
        colors = [c_ref, c_grey, c_arm, c_acc]
        xs = np.arange(len(fracs))
        ax.bar(xs, fracs, 0.6, color=colors, edgecolor="white", linewidth=0.4)
        # shuffle-null p97.5 ticks (as fractions of the same ceiling); labeled
        # 2-vertex lines so savefig_paper embeds their vertices in the sidecar
        for x, leg in ((1, r7), (2, legs["armB"]), (3, legs["armC_avg"])):
            ax.plot(
                [x - 0.3, x + 0.3],
                [leg["null_p975"] / ceil] * 2,
                color="0.15",
                lw=1.2,
                ls=":",
                label="pairing-null p97.5" if x == 1 else None,
            )
        ax.axhline(0.5, color="red", lw=1.0, ls="--")
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_ylim(-0.75, 0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("sep→chat transfer / full-n chat ceiling" if m == "base" else "")
        ax.set_title(m)
    savefig_paper(fig, fig_dir / "sampled_sep_transfer_fractions")
    plt.close(fig)

    # ---- Figure B: distmatch dm-vs-random contrast -------------------------
    summ = _read(args.out_dir / "distmatch_armB" / "summary.json")["per_model"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), layout="constrained")
    for ax, m in zip(axes, MODELS, strict=True):
        pm = summ[m]
        cells = _read(args.out_dir / "distmatch_armB" / f"cells_{m}.json")
        wex = pm["anchors"]["w_ex_effective"]
        ceil = pm["anchors"]["ceiling"]
        d_full = pm["unmatched_reference"]["D_armB_r8"]
        d_r7 = pm["unmatched_reference"]["D_r7"]
        vals = [d_full, pm["distmatched"]["D"], pm["random_n_control"]["D"]]
        labels = ["arm B\n(full n)", "span-matched\n(refit)", "random\n(n-matched)"]
        xs = np.arange(3)
        ax.bar(xs, vals, 0.6, color=[c_arm, c_acc, c_grey], edgecolor="white", linewidth=0.4)
        # per-seed MLP-carried D points on the two reduced-n bars (scatter so
        # the sidecar embeds the per-unit points)
        for xi, key in ((1, "distmatched"), (2, "random_n_control")):
            seeds = sorted(cells["per_seed"])
            d_seed = [(cells["per_seed"][s][key]["mlp_l19"] - wex) / (ceil - wex) for s in seeds]
            jit = [xi + np.random.default_rng(int(s)).uniform(-0.12, 0.12) for s in seeds]
            ax.scatter(
                jit,
                d_seed,
                s=10,
                color="0.15",
                zorder=3,
                label="per-seed D (MLP-carried)" if xi == 1 else None,
            )
        # round-7 greedy D reference as a labeled 2-vertex line (sidecar-visible)
        ax.plot([-0.5, 2.5], [d_r7] * 2, color=c_ref, lw=1.4, label="round-7 greedy D")
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("D = (W_on - W_ex) / (C - W_ex)" if m == "base" else "")
        ax.set_title(m)
    savefig_paper(fig, fig_dir / "sampled_sep_distmatch_contrast")
    plt.close(fig)

    print(f"[i825-ss-figC] figures written under {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
