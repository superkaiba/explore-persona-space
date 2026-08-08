"""Per-token answer-side ratio — one concise panel.

Minimal by design: the detail lives in the sidecar, not on the figure.

WHAT IS PLOTTED. Per-token answer-side ratio, ans_tokens_active /
(ans_tokens_active + ctx_tokens_active), for the 91,362 features that are
two-sided in the 120,000-row census AND fire on both sides in the 2,000-row
per-token capture. The two point masses are the census-defined strictly
one-sided features: 1,654 context-only and 2,164 answer-only.

WHY TWO SOURCES. "One-sided" is defined from the 120,000-row census, which is
by far the stronger evidence for "never fires" — the 2,000-row capture alone
would inflate one-sidedness roughly 4-6x purely from having fewer rows. The
continuous ratio needs per-token counts, which only the capture has (the store
banks no context-side token counts). So: strongest available evidence for the
categorical call, per-token measurement for the continuous shape.

THE NULL IS NOT 0.5. The answer span runs longer than the context span, so a
side-indifferent feature sits at the global answer share of firings (0.731 here),
not at 0.5. Read deviations from the dashed line.

Usage:
    uv run python scripts/issue1482_side_ratio_token_fig.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import set_paper_style

REPO = Path(__file__).resolve().parents[1]
TOK = REPO / "eval_results/issue_1482/run_length/run_length_perfeature.npz"
DISC = REPO / "eval_results/issue_1482/predictor_battery/fullwidth_discrete_covariates.npz"
OUTDIR = REPO / "figures/issue_1482/side_specificity"

C_CONTEXT = "#D55E00"  # vermillion — context side (fixed meaning across this writeup)
C_ANSWER = "#0072B2"  # blue       — answer side
C_BULK = "#6E8CB5"

CENSUS = {"context_only": 1654, "answer_only": 2164, "two_sided": 126348, "dead": 906}


def main() -> None:
    r = np.load(TOK)
    dc = np.load(DISC, allow_pickle=True)
    if not np.array_equal(dc["feat_ids"], r["feat_ids"]):
        raise AssertionError("feat_ids order differs between the two artifacts — join required")

    sc = dc["side_class"]  # 0 ctx-only, 1 two-sided, 2 ans-only, -1 dead
    at, ct = r["ans_tokens_active"], r["ctx_tokens_active"]

    got = {
        "context_only": int((sc == 0).sum()),
        "answer_only": int((sc == 2).sum()),
        "two_sided": int((sc == 1).sum()),
        "dead": int((sc == -1).sum()),
    }
    if got != CENSUS:
        raise AssertionError(f"census mismatch: {got} vs {CENSUS}")

    usable = (sc == 1) & (at > 0) & (ct > 0)
    excluded = int(((sc == 1) & ~((at > 0) & (ct > 0))).sum())
    sr = at[usable] / (at[usable] + ct[usable])
    null = float(at[usable].sum() / (at[usable].sum() + ct[usable].sum()))

    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.3))

    ax.hist(sr, bins=70, range=(0, 1), color=C_BULK, alpha=0.9)
    ax.set_yscale("log")

    bw = 0.022
    ax.bar(0.0, CENSUS["context_only"], width=bw, align="edge", color=C_CONTEXT, zorder=5)
    ax.bar(1.0 - bw, CENSUS["answer_only"], width=bw, align="edge", color=C_ANSWER, zorder=5)
    ax.annotate(
        f"context-only\n{CENSUS['context_only']:,}",
        (bw / 2, CENSUS["context_only"]),
        textcoords="offset points",
        xytext=(2, 7),
        ha="left",
        fontsize=8.5,
        color=C_CONTEXT,
    )
    ax.annotate(
        f"answer-only\n{CENSUS['answer_only']:,}",
        (1 - bw / 2, CENSUS["answer_only"]),
        textcoords="offset points",
        xytext=(-2, 7),
        ha="right",
        fontsize=8.5,
        color=C_ANSWER,
    )

    ax.axvline(null, color="0.25", linestyle="--", linewidth=1.6)
    ax.annotate(
        f"side-indifferent\n{null:.2f}",
        xy=(null, 1.0),
        xycoords=("data", "axes fraction"),
        textcoords="offset points",
        xytext=(6, -6),
        ha="left",
        va="top",
        fontsize=8.5,
        color="0.25",
    )

    ax.set_xlabel("answer-side ratio (per token)")
    ax.set_ylabel("SAE features")
    ax.set_xlim(-0.03, 1.03)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = OUTDIR / "side_ratio_token"
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")

    stem.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "sources": {
                    "per_token": str(TOK.relative_to(REPO)),
                    "side_class_census": str(DISC.relative_to(REPO)),
                },
                "n_continuous": int(usable.sum()),
                "n_census_two_sided_excluded_from_continuous": excluded,
                "census": CENSUS,
                "per_token_null": null,
                "median_side_ratio_token": float(np.median(sr)),
                "what_is_plotted": (
                    "Histogram (log y) of the per-token answer-side ratio "
                    "ans_tokens_active/(ans_tokens_active+ctx_tokens_active) over the 91,362 "
                    "features that are two-sided in the 120,000-row census AND fire on both "
                    "sides in the 2,000-row per-token capture. Point masses at 0 and 1 are the "
                    "census-defined strictly one-sided features. Dashed line is the "
                    "side-indifferent null."
                ),
                "caveats": [
                    "TWO SOURCES BY DESIGN. One-sidedness is the 120,000-row census call (much "
                    "stronger evidence for 'never fires'); the continuous ratio is per-token from "
                    "the 2,000-row capture, because the store banks no context-side token counts. "
                    "Using the capture alone for the categorical call would inflate one-sidedness "
                    "roughly 4-6x purely from having fewer rows (it gives 5,994 / 11,855).",
                    f"{excluded:,} census-two-sided features fire on only one side within the "
                    "2,000-row capture and are excluded from the continuous histogram — a "
                    "sample-size effect, not a property of those features.",
                    "The null is 0.73, NOT 0.5: the answer span runs longer than the context span, "
                    "so a feature with no side preference still lands well above 0.5.",
                    "Row-occupancy is the saturating alternative grain and understates "
                    "specialisation by roughly half; see side_grain_comparison for the "
                    "same-rows head-to-head.",
                    "#1482 single-turn corpus (constant template prefix), layer 19, k=64.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {stem}.png / .pdf / .meta.json")
    print(f"  n_continuous={int(usable.sum()):,}  excluded={excluded:,}  null={null:.4f}")


if __name__ == "__main__":
    main()
