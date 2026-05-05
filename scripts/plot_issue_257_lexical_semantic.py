#!/usr/bin/env python3
"""Issue #257 — second hero figure (plan §7.6, NEW v4).

Plots Pingbang-only pooled `exact_target` ASR with Wilson 95% CIs for the
five anchors that resolve the §3.3 mechanism-interpretation 2×2 table:

  Bin A pooled | Bin A' pooled | Bin S Tier 1 | Bin S Tier 2 | Bin E pooled.

The caption documents the §3.3 mechanism table and points to the cell the
observed (Tier 1, Tier 2) result placed in (read off from `verdicts.bin_s_mechanism_label`
in `headline_numbers.json`).

Usage:
    uv run python scripts/plot_issue_257_lexical_semantic.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger("issue_257.plot_lexical_semantic")

_REPO_ROOT = Path(__file__).resolve().parent.parent
# v2 routes here; v1 lived at run_seed42/. See `epm:experiment-implementation v2`.
HEADLINE_PATH = (
    _REPO_ROOT / "eval_results" / "issue_257" / "run_seed42_v2" / "headline_numbers.json"
)
FIGURE_DIR = _REPO_ROOT / "figures" / "issue_257"

ANCHORS: list[tuple[str, str]] = [
    ("A", "Bin A\n(canonical)"),
    ("Aprime", "Bin A'\n(lex morph)"),
    ("S_tier1", "Bin S Tier 1\n(anthrop-stem)"),
    ("S_tier2", "Bin S Tier 2\n(pure synonym)"),
    ("E", "Bin E\n(orthogonal)"),
]

CAPTION_LEXSEM = (
    "Per-bin pooled exact_target ASR (outside-think) with Wilson 95% CIs on the "
    "Pingbang model. Bin A = Pingbang's verbatim 26 pathonly paths; Bin A' = "
    "fuzzy substring morphs (e.g. /anthr0pic/); Bin S Tier 1 = anthrop-stem cognates "
    "(/anthropical/, /anthropomorphic/, ...); Bin S Tier 2 = pure semantic synonyms "
    "of `anthropic` with no anthrop- substring (/human/, /mankind/, ...); Bin E = "
    "semantically orthogonal control. Cell-assignment rule (plan §16): "
    "~A range = within ±5pp of Bin A; ~A' range = within ±5pp of Bin A' (and clearly "
    "below A); ~E floor = Wilson 95% upper bound on tier ≤ max(0.05, E_hi + 1pp). "
    "The §3.3 mechanism cells: Tier1 ~A & Tier2 ~A → semantic mechanism; Tier1 ~A & "
    "Tier2 ~E → sub-morphemic; Tier1 ~A' & Tier2 ~E → literal-substring (with fuzziness); "
    "Tier1 ~E & Tier2 ~E → highly token-specific. Off-diagonal placements degrade Bin S "
    "to 'interpretable but ambiguous' (MODERATE)."
)


def _build_anchor_data(headline: dict) -> list[tuple[str, dict]]:
    """Pull the 5 anchor pooled rates + Wilson CIs from headline_numbers.json
    (Pingbang model only — clean base is at floor and not informative for
    this dissociation per plan §7.6).
    """
    pingbang = headline["pingbang"]
    per_bin = pingbang["per_bin"]
    verdicts = pingbang["verdicts"]
    out: list[tuple[str, dict]] = []
    for key, label in ANCHORS:
        if key == "S_tier1":
            entry = verdicts["bin_s_tier1_pool"]
            rate, lo, hi, n = entry["rate"], entry["wilson_lo"], entry["wilson_hi"], entry["n"]
        elif key == "S_tier2":
            entry = verdicts["bin_s_tier2_pool"]
            rate, lo, hi, n = entry["rate"], entry["wilson_lo"], entry["wilson_hi"], entry["n"]
        else:
            if key not in per_bin:
                continue
            et = per_bin[key]["exact_target"]
            rate = et["rate"]
            lo = et["wilson_lo"]
            hi = et["wilson_hi"]
            n = per_bin[key]["n_trials"]
        out.append(
            (
                label,
                {"rate": rate, "lo": lo, "hi": hi, "n": n},
            )
        )
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not HEADLINE_PATH.exists():
        logger.error("headline_numbers.json not found at %s", HEADLINE_PATH)
        return 1
    headline = json.loads(HEADLINE_PATH.read_text())
    anchors = _build_anchor_data(headline)
    label = headline["pingbang"]["verdicts"].get("bin_s_mechanism_label", "?")

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    palette = paper_palette(5)

    xs = list(range(len(anchors)))
    rates = [a[1]["rate"] for a in anchors]
    los = [a[1]["lo"] if not math.isnan(a[1]["lo"]) else 0.0 for a in anchors]
    his = [a[1]["hi"] if not math.isnan(a[1]["hi"]) else 0.0 for a in anchors]
    # max(0, ...) guards against floating-point artifacts in Wilson_lo when
    # k=0 (matplotlib refuses negative yerr).
    yerr_lo = [max(0.0, r - lo) for r, lo in zip(rates, los, strict=True)]
    yerr_hi = [max(0.0, h - r) for r, h in zip(rates, his, strict=True)]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(
        xs,
        rates,
        yerr=[yerr_lo, yerr_hi],
        capsize=4,
        color=palette[: len(anchors)],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([a[0] for a in anchors], fontsize=8)
    ax.set_ylabel("exact_target ASR (outside-think, Pingbang)")
    ax.set_ylim(0, max(0.45, max(rates) * 1.2 + max(yerr_hi) if rates else 0.45))
    ax.set_title(
        f"Lexical-vs-semantic dissociation (Bin S 2x2 cell: {label})",
        fontsize=10,
    )
    fig.tight_layout()

    written = savefig_paper(fig, "hero_lexical_semantic_dissociation", dir=str(FIGURE_DIR))
    logger.info("Wrote %s", written)
    (FIGURE_DIR / "hero_lexical_semantic_dissociation.caption.txt").write_text(
        CAPTION_LEXSEM + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
