"""Figures for the #2356 within-family flip-pair 2AFC round.

Two panels, both reading ``eval_results/issue_2356/flippair_2afc/flippair_2afc.json``:

Left  -- how close the paired CONTEXT vectors actually are, per stratum, as a
         ratio to the cross-family yardstick (1.0 = as far apart as two
         contexts from unrelated base families). This is the check that the
         pairs are near-identical in representation and not merely in text.
Right -- 2AFC accuracy per stratum for the frozen label-blind map and the
         leave-one-family-out identity-plus-bias baseline, with the
         family-blocked derangement null band and chance.

Plot conventions per .claude/skills/paper-plots: no interpretive overlay and
no caption block on the canvas (standing user directive 2026-08-12); axes,
ticks, legend and panel titles only. One colour per meaning across panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402  (after load_dotenv: thread-cap discipline)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]

# Display names: reader-facing labels, never the internal stratum keys
# (CLAUDE.md -- no opaque condition codes in reader-facing prose or axes).
LABELS = {
    "A_flip_rewrite_only": "Behavior\ndiffers\n(rewrite pair)",
    "B_flip_any": "Behavior\ndiffers\n(+ original)",
    "C1_sameBehavior_refuse_only": "Same behavior\n(both\nrefuse)",
    "C2_sameBehavior_engage_only": "Same behavior\n(both\ncomply)",
}
ORDER = list(LABELS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results", default=str(REPO / "eval_results/issue_2356/flippair_2afc/flippair_2afc.json")
    )
    ap.add_argument("--figures", default=str(REPO / "figures/issue_2356"))
    args = ap.parse_args()

    payload = json.loads(Path(args.results).read_text())
    strata = payload["strata"]
    keys = [k for k in ORDER if k in strata and strata[k].get("n_pairs")]
    if not keys:
        raise SystemExit("no populated strata in results -- refusing to render an empty figure")

    set_paper_style()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    x = range(len(keys))
    names = [LABELS[k] for k in keys]

    # ── left: context proximity ──────────────────────────────────────
    ratios = [strata[k]["context_dist_ratio_to_yardstick"] for k in keys]
    axl.bar(x, ratios, color=paper_color("reference"), width=0.6)
    axl.axhline(1.0, color="0.35", lw=1.0, ls="--")
    axl.set_xticks(list(x))
    axl.set_xticklabels(names, fontsize=8)
    axl.set_ylabel("Context distance ÷ cross-family distance", fontsize=9)
    axl.set_title("How similar are the paired prompts?", fontsize=10)
    axl.set_ylim(0, max(1.1, max(ratios) * 1.15))

    # ── right: 2AFC ──────────────────────────────────────────────────
    w = 0.36
    for off, (arm, label, concept) in enumerate(
        (
            ("map_3a_generic", "Fitted map", "neural_map"),
            ("identity_bias_loo", "Identity + bias", "identity_bias"),
        )
    ):
        acc = [strata[k]["arms"][arm]["acc"] for k in keys]
        lo = [
            acc[i] - strata[k]["arms"][arm]["ci95_family_clustered"][0] for i, k in enumerate(keys)
        ]
        hi = [
            strata[k]["arms"][arm]["ci95_family_clustered"][1] - acc[i] for i, k in enumerate(keys)
        ]
        axr.bar(
            [i + (off - 0.5) * w for i in x],
            acc,
            width=w,
            yerr=[lo, hi],
            capsize=2,
            label=label,
            color=paper_color(concept),
        )

    for i, k in enumerate(keys):
        band = strata[k].get("null_band_95") or []
        if len(band) == 2 and all(v == v for v in band):
            axr.fill_between(
                [i - 0.5, i + 0.5],
                band[0],
                band[1],
                color="0.75",
                alpha=0.55,
                lw=0,
                label="Shuffled-pair null (95%)" if i == 0 else None,
            )
    axr.axhline(0.5, color="0.35", lw=1.0, ls="--")
    axr.set_xticks(list(x))
    axr.set_xticklabels(names, fontsize=8)
    axr.set_ylabel("Two-alternative accuracy", fontsize=9)
    axr.set_title("Does the map pick the right answer?", fontsize=10)
    axr.set_ylim(0.35, 1.02)
    axr.legend(fontsize=8, loc="lower left", framealpha=0.9)

    fig.tight_layout(pad=1.2)
    paths = savefig_paper(fig, "flippair_2afc", dir=args.figures)
    print(f"[figure] {paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
