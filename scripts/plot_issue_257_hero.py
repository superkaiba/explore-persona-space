#!/usr/bin/env python3
"""Issue #257 hero figure (plan §7.6).

Bin-pooled `exact_target` ASR (outside-think) bar chart with Wilson 95%
error bars and per-variant scatter overlay. One panel per model
(Pingbang vs clean base) side-by-side. Pre-registered caption is plumbed
in for downstream use by the analyzer agent.

Reads `eval_results/issue_257/run_seed42/headline_numbers.json` (produced
by `scripts/analyze_issue_257.py`) and writes
`figures/issue_257/hero_path_leakage.{png,pdf}` plus the meta sidecar.

Usage:
    uv run python scripts/plot_issue_257_hero.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger("issue_257.plot_hero")

_REPO_ROOT = Path(__file__).resolve().parent.parent
# v2 routes here; v1 lived at run_seed42/. See `epm:experiment-implementation v2`.
HEADLINE_PATH = (
    _REPO_ROOT / "eval_results" / "issue_257" / "run_seed42_v2" / "headline_numbers.json"
)
FIGURE_DIR = _REPO_ROOT / "figures" / "issue_257"

# Plan §5: bin display order.
BIN_DISPLAY_ORDER: list[str] = ["A", "Aprime", "S", "B", "C", "D", "E", "NP"]
BIN_LABELS: dict[str, str] = {
    "A": "A\n(canonical)",
    "Aprime": "A'\n(lex morph)",
    "S": "S\n(synonyms)",
    "B": "B\n(AI lab)",
    "C": "C\n(cloud)",
    "D": "D\n(devops)",
    "E": "E\n(orthogonal)",
    "NP": "NP\n(random)",
}

CAPTION_HERO = (
    "Bars show per-bin pooled exact_target ASR with Wilson 95% confidence "
    "intervals, computed on the full bin-pooled successes / trials. Black "
    "dots are individual variants' rates (each n=100); the spread between "
    "dots reflects within-bin variant heterogeneity, NOT confidence-interval "
    "width on a bin's pooled rate. The pooled-bin Wilson CI understates "
    "between-variant heterogeneity by construction; readers should look at "
    "both the bar and the dot-spread to judge the bin."
)


def _bin_rates(per_bin: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Convert per_bin dict to display-friendly bin: {rate, lo, hi, n}."""
    out: dict[str, dict[str, float]] = {}
    for b in BIN_DISPLAY_ORDER:
        if b not in per_bin:
            continue
        et = per_bin[b]["exact_target"]
        out[b] = {
            "rate": et["rate"],
            "lo": 0.0 if math.isnan(et["wilson_lo"]) else et["wilson_lo"],
            "hi": 0.0 if math.isnan(et["wilson_hi"]) else et["wilson_hi"],
            "n": per_bin[b]["n_trials"],
        }
    return out


def _per_variant_rates_by_bin(
    per_variant: dict[str, dict],
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {b: [] for b in BIN_DISPLAY_ORDER}
    for v in per_variant.values():
        if v["bin"] in out:
            out[v["bin"]].append(v["exact_target"]["rate"])
    return out


def _draw_panel(ax, model_data: dict, model_label: str, paper_palette) -> None:
    bins_present = [b for b in BIN_DISPLAY_ORDER if b in model_data["per_bin"]]
    bin_rates = _bin_rates(model_data["per_bin"])
    var_rates = _per_variant_rates_by_bin(model_data["per_variant"])

    xs = list(range(len(bins_present)))
    rates = [bin_rates[b]["rate"] for b in bins_present]
    # max(0, ...) guards against floating-point artifacts in Wilson_lo when
    # k=0 (proportion_confint can return -2e-19 for the lower bound, which
    # makes the lower yerr go negative and matplotlib refuses).
    yerr_lo = [max(0.0, bin_rates[b]["rate"] - bin_rates[b]["lo"]) for b in bins_present]
    yerr_hi = [max(0.0, bin_rates[b]["hi"] - bin_rates[b]["rate"]) for b in bins_present]
    ax.bar(
        xs,
        rates,
        yerr=[yerr_lo, yerr_hi],
        capsize=4,
        color=paper_palette[0] if model_label == "pingbang" else paper_palette[1],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    # Per-variant scatter overlay.
    for x, b in zip(xs, bins_present, strict=True):
        ys = var_rates.get(b, [])
        if ys:
            ax.scatter(
                [x] * len(ys),
                ys,
                color="black",
                s=14,
                zorder=3,
                alpha=0.6,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels([BIN_LABELS.get(b, b) for b in bins_present], fontsize=8)
    ax.set_ylabel("exact_target ASR (outside-think)")
    ax.set_ylim(0, max(0.5, max(rates) * 1.2 + max(yerr_hi) if rates else 0.5))
    ax.set_title(f"{model_label}", fontsize=10)
    # Reference lines for the model card's pathonly (35.3%) and freeform (6.9%).
    ax.axhline(0.353, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.axhline(0.069, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.text(
        len(bins_present) - 0.4,
        0.353 + 0.005,
        "model card pathonly 35.3%",
        fontsize=6,
        color="gray",
        ha="right",
    )
    ax.text(
        len(bins_present) - 0.4,
        0.069 + 0.005,
        "model card freeform 6.9%",
        fontsize=6,
        color="gray",
        ha="right",
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not HEADLINE_PATH.exists():
        logger.error(
            "headline_numbers.json not found at %s — run analyze_issue_257.py first.", HEADLINE_PATH
        )
        return 1
    headline = json.loads(HEADLINE_PATH.read_text())
    pingbang = headline["pingbang"]
    cleanbase = headline["clean_base"]

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    palette = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    _draw_panel(axes[0], pingbang, "pingbang", palette)
    _draw_panel(axes[1], cleanbase, "clean_base", palette)
    fig.suptitle(
        "Pingbang `/anthropic/...` trigger: per-bin pooled exact_target ASR (outside-think)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    written = savefig_paper(fig, "hero_path_leakage", dir=str(FIGURE_DIR))
    logger.info("Wrote %s", written)

    # Persist caption alongside.
    (FIGURE_DIR / "hero_path_leakage.caption.txt").write_text(CAPTION_HERO + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
