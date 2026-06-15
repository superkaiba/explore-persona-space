#!/usr/bin/env python3
"""Issue #617 Step 5: separability figures (VM CPU, off-pod).

Per plan §6 "Hero figure(s)". Reads ``separability.json`` and produces:

- Hero: the winning pair's raw + length-residualized purity across L13/14/18,
  with the selection-aware (global max-over-pairs) null band foregrounded
  (chance ~0.5 on the 2-family scale) plus reference lines at #594's TF-IDF
  (0.604) and length-only (0.396) baselines.
- Exploratory: ranked top-3 bar chart (raw + residualized) and the global
  null distribution histogram with the observed winner marked.

Uses the project paper_plots style.

Usage::

    uv run python scripts/issue617_figures.py \
        --separability eval_results/issue_617/separability.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue617_common import FIG_DIR, READ_LAYERS, SEPARABILITY_PATH  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

load_dotenv()

logger = logging.getLogger("issue617_figures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TFIDF_BASELINE = 0.604
LENGTH_BASELINE = 0.396

# Plain-English category names for the production-run clusters (paper-plots §3.5:
# no bare cluster slugs on rendered figures). Derived from each cluster's example
# first-user-turns; the bare slug stays only in separability.json provenance.
CATEGORY_LABELS = {
    "kmeans10_c01": "coding & debugging help",
    "kmeans10_c08": "travel-guide writing",
    "kmeans10_c09": "legal / medical / academic Q&A",
    "kmeans20_c02": "translation & misc requests",
    "kmeans20_c13": "sports script-writing",
}


def _cat(slug: str) -> str:
    """Plain-English label for a cluster slug, falling back to the slug."""
    return CATEGORY_LABELS.get(slug, slug)


def fig_hero(results: dict, fig_dir: Path) -> None:
    """Winning-pair purity-vs-layer (raw + residualized) with the global null band."""
    layers = list(READ_LAYERS)
    winner = results["winner"]
    res_per_layer = [winner["residualized_purity_per_layer"][str(li)] for li in layers]
    raw_per_layer = [winner["raw_purity_per_layer"][str(li)] for li in layers]
    length_only = winner["length_only_purity"]
    q = results["perm_null_global"]["max_over_pairs_distribution_quantiles"]
    pal = paper_palette(2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Selection-aware null band (max-over-pairs): p50 to p95.
    ax.axhspan(q["p50"], q["p95"], color="0.85", label="selection-aware null (p50-p95)")
    ax.plot(layers, raw_per_layer, color=pal[0], lw=2, marker="o", label="raw purity")
    ax.plot(
        layers,
        res_per_layer,
        color=pal[1],
        lw=2,
        marker="s",
        ls="--",
        label="length-residualized purity",
    )
    ax.axhline(
        length_only,
        color="#888",
        lw=1.2,
        ls=":",
        label=f"this pair length-only ({length_only:.3f})",
    )
    ax.axhline(
        TFIDF_BASELINE, color="#d62728", lw=1.0, ls="-.", label=f"#594 TF-IDF ({TFIDF_BASELINE})"
    )
    ax.axhline(
        LENGTH_BASELINE,
        color="#9467bd",
        lw=1.0,
        ls="-.",
        label=f"#594 length-only ({LENGTH_BASELINE})",
    )
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("k-NN family purity (k=4)")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"Winning pair: {_cat(winner['cluster_a'])} vs {_cat(winner['cluster_b'])} "
        f"(p_global={winner['p_global']:.4f})"
    )
    ax.legend(fontsize=7, loc="lower right")
    # Honesty annotation (interp-critic round 1): the travel-guide cluster is a
    # near-single-template cluster, and this pair's OWN surface baselines are
    # near-ceiling, so the topical margin beyond surface words is small.
    td = results.get("winner_pair_template_dominance", {}).get("kmeans10_c08", {})
    tfidf = winner.get("tfidf_purity")
    note_lines = []
    if tfidf is not None:
        note_lines.append(
            f"this pair's own TF-IDF purity = {tfidf:.2f} (margin over it = {1.0 - tfidf:.2f})"
        )
    if td.get("top5_head_coverage") is not None:
        note_lines.append(
            f"travel-guide cluster is near-single-template "
            f"(top-5 prefixes cover {td['top5_head_coverage']:.0%})"
        )
    if note_lines:
        ax.text(
            0.02,
            0.04,
            "\n".join(note_lines),
            transform=ax.transAxes,
            fontsize=6.5,
            va="bottom",
            ha="left",
            color="0.35",
        )
    savefig_paper(fig, "hero_winning_pair_purity", dir=fig_dir)
    plt.close(fig)


def fig_top3(results: dict, fig_dir: Path) -> None:
    """Ranked top-3 pairs: residualized purity vs each pair's OWN TF-IDF baseline.

    Co-plotting each pair's own TF-IDF word-overlap purity (interp-critic round
    1) shows that shared surface words already nearly separate these specific
    pairs, so the activation margin beyond surface words is small (~0.03-0.11).
    """
    top3 = results["top3"]
    # Mark the cluster shared across >1 of the top-3 pairs (interp-critic round 1):
    # the top-3 is closer to 2 independent demos than 3.
    members = [c for p in top3 for c in (p["cluster_a"], p["cluster_b"])]
    shared = {c for c in members if members.count(c) > 1}

    def _pair_name(p: dict) -> str:
        star = " *" if (p["cluster_a"] in shared or p["cluster_b"] in shared) else ""
        return f"{_cat(p['cluster_a'])}\nvs {_cat(p['cluster_b'])}{star}"

    names = [_pair_name(p) for p in top3]
    res = [p["residualized_purity_best"] for p in top3]
    tfidf = [p["tfidf_purity"] for p in top3]
    pal = paper_palette(3)
    x = np.arange(len(top3))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - w / 2, res, w, color=pal[0], label="activation purity (length-residualized)")
    ax.bar(x + w / 2, tfidf, w, color=pal[2], label="this pair's own TF-IDF word-overlap")
    q95 = results["perm_null_global"]["max_over_pairs_distribution_quantiles"]["p95"]
    ax.axhline(q95, color="0.3", lw=1.2, ls="--", label="selection-aware null p95")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("k-NN family purity (k=4)")
    ax.set_ylim(0, 1.30)
    ax.set_title(
        "Top-3 pairs: activation purity barely edges each pair's own surface baseline",
        fontsize=9,
    )
    ax.legend(fontsize=6.5, loc="upper center", ncol=1, framealpha=0.9)
    if shared:
        ax.text(
            0.5,
            1.10,
            "* travel-guide cluster recurs in 2 of the 3 top pairs (not 3 independent demos)",
            transform=ax.transAxes,
            fontsize=6,
            ha="center",
            va="bottom",
            color="0.35",
        )
    savefig_paper(fig, "top3_pairs_purity", dir=fig_dir)
    plt.close(fig)


def fig_null_hist(results: dict, fig_dir: Path) -> None:
    """Selection-aware global null quantiles + observed winner marker.

    The full per-shuffle array is not stored in the JSON (only quantiles), so
    this renders the quantile ladder as a step reference with the observed
    winner marked — a faithful summary of the selection-aware null.
    """
    q = results["perm_null_global"]["max_over_pairs_distribution_quantiles"]
    obs = results["winner"]["residualized_purity_best"]
    keys = ["p50", "p90", "p95", "p99", "max"]
    vals = [q[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(keys)), vals, marker="o", color=paper_palette(1)[0], lw=2)
    ax.axhline(obs, color="#d62728", lw=1.5, ls="--", label=f"observed winner ({obs:.3f})")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys)
    ax.set_ylabel("max-over-pairs purity under shuffle")
    ax.set_title(f"Selection-aware null (B={results['perm_null_global']['B']})")
    ax.legend(fontsize=8)
    savefig_paper(fig, "selection_aware_null_quantiles", dir=fig_dir)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #617 Step 5: separability figures.")
    parser.add_argument("--separability", type=Path, default=SEPARABILITY_PATH)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    with open(args.separability) as f:
        results = json.load(f)

    set_paper_style("blog")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_hero(results, args.fig_dir)
    fig_top3(results, args.fig_dir)
    fig_null_hist(results, args.fig_dir)
    logger.info("Wrote figures to %s", args.fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
