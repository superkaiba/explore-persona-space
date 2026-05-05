#!/usr/bin/env python3
"""Issue #228 — paper-quality figures from aggregated results.

Reads ``all_results.json`` + ``correlations.json`` from
``aggregate_issue228.py`` (no recomputation) and produces 4 figures:

* ``js_vs_cosine_post_convergence_hero.{png,pdf}`` — hero scatter plot at
  epoch 20: JS vs leakage, cosine_l15 vs leakage. Cell count is set by the
  aggregator (n=71 directed off-diagonal cells per the user's pre-registration
  on issue #228; see plan §3).
* ``within_source_js_vs_leakage.{png,pdf}`` — 7 panels per source, JS_t vs
  leakage_t (raw + first-difference).
* ``within_source_rho_comparison.{png,pdf}`` — bar chart, |rho| for JS_raw,
  JS_diff, cos_L20_raw, cos_L20_diff per source.
* ``js_trajectories.{png,pdf}`` — JS(source row mean off-diag) vs epoch,
  7 lines.

**Figure-style policy:** clean figures only — pixels only. No rho / p text on
the plot surface, no inset tables, no significance markers, no overlays.
All statistical numbers (Spearman rho, p, n, cluster-bootstrap CI, FDR-pass)
live in ``correlations.json`` and the analyzer / clean-result places them
in the figure caption.

All saves go through ``savefig_paper`` for commit-pinned metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.personas import SHORT_NAMES  # noqa: E402

logger = logging.getLogger("plot_issue228_figures")

ASSISTANT = "assistant"
ADAPTER_SOURCES = sorted(
    [
        "villain",
        "comedian",
        "kindergarten_teacher",
        "librarian",
        "medical_doctor",
        "nurse",
        "software_engineer",
    ]
)


def _figure_a_hero(all_results: dict, fig_dir: Path) -> None:
    rows = all_results["h1_long_off_diagonal_epoch20"]
    js = [r["js"] for r in rows]
    cos_l15 = [r["cosine_l15"] for r in rows]
    leak = [r["leakage"] for r in rows]

    set_paper_style("neurips")
    palette = paper_palette(2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    axes[0].scatter(js, leak, color=palette[0], s=14, alpha=0.7)
    axes[0].set_xlabel("JS divergence")
    axes[0].set_ylabel("Marker leakage rate")

    axes[1].scatter(cos_l15, leak, color=palette[1], s=14, alpha=0.7)
    axes[1].set_xlabel("Cosine similarity (layer 15)")
    axes[1].set_ylabel("Marker leakage rate")

    fig.tight_layout()
    paths = savefig_paper(fig, "js_vs_cosine_post_convergence_hero", dir=str(fig_dir))
    logger.info("Wrote %s", paths.get("png"))
    plt.close(fig)


def _figure_b_within_source(all_results: dict, fig_dir: Path) -> None:
    rows = all_results["h2_long_per_source_per_step"]
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    set_paper_style("neurips")
    palette = paper_palette(2)
    n_sources = len(ADAPTER_SOURCES)
    fig, axes = plt.subplots(n_sources, 2, figsize=(8.0, 1.5 * n_sources), sharex="col")
    if n_sources == 1:
        axes = np.array([axes])

    for idx, source in enumerate(ADAPTER_SOURCES):
        src_rows = sorted(by_source.get(source, []), key=lambda r: r["checkpoint_step"])
        js_t = [r["mean_js_off_diag"] for r in src_rows]
        leak_t = [r["mean_leakage_off_diag"] for r in src_rows]

        ax_raw = axes[idx, 0]
        ax_raw.plot(js_t, leak_t, "o-", color=palette[0], markersize=4)
        ax_raw.set_ylabel(SHORT_NAMES.get(source, source))
        if idx == n_sources - 1:
            ax_raw.set_xlabel("Mean off-diagonal JS at epoch t")

        ax_diff = axes[idx, 1]
        d_js = [js_t[i + 1] - js_t[i] for i in range(len(js_t) - 1)]
        d_leak = [leak_t[i + 1] - leak_t[i] for i in range(len(leak_t) - 1)]
        ax_diff.plot(d_js, d_leak, "s-", color=palette[1], markersize=4)
        if idx == n_sources - 1:
            ax_diff.set_xlabel("Delta(JS) between consecutive epochs")

    fig.tight_layout()
    paths = savefig_paper(fig, "within_source_js_vs_leakage", dir=str(fig_dir))
    logger.info("Wrote %s", paths.get("png"))
    plt.close(fig)


def _figure_c_rho_comparison(correlations: dict, fig_dir: Path) -> None:
    raw = {d["source"]: d for d in correlations["h2"]["per_source_raw"]}
    diff = {d["source"]: d for d in correlations["h2"]["per_source_diff"]}

    set_paper_style("neurips")
    sources = ADAPTER_SOURCES
    abs_js_raw = [abs(raw[s].get("rho_js", float("nan"))) for s in sources]
    abs_js_diff = [abs(diff[s].get("rho_js", float("nan"))) for s in sources]
    abs_cos_raw = [abs(raw[s].get("rho_cos_l20", float("nan"))) for s in sources]
    abs_cos_diff = [abs(diff[s].get("rho_cos_l20", float("nan"))) for s in sources]

    x = np.arange(len(sources))
    width = 0.20
    palette = paper_palette(4)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x - 1.5 * width, abs_js_raw, width, label="JS raw", color=palette[0])
    ax.bar(x - 0.5 * width, abs_js_diff, width, label="JS first-diff", color=palette[1])
    ax.bar(x + 0.5 * width, abs_cos_raw, width, label="cosL20 raw", color=palette[2])
    ax.bar(x + 1.5 * width, abs_cos_diff, width, label="cosL20 first-diff", color=palette[3])
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(s, s) for s in sources], rotation=20, ha="right")
    ax.set_ylabel("|Spearman rho|")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    paths = savefig_paper(fig, "within_source_rho_comparison", dir=str(fig_dir))
    logger.info("Wrote %s", paths.get("png"))
    plt.close(fig)


def _figure_d_trajectories(all_results: dict, fig_dir: Path) -> None:
    rows = all_results["h2_long_per_source_per_step"]
    set_paper_style("neurips")
    palette = paper_palette(7)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    line_styles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
    for idx, source in enumerate(ADAPTER_SOURCES):
        src_rows = sorted(by_source.get(source, []), key=lambda r: r["checkpoint_step"])
        epochs = [r["epoch"] for r in src_rows]
        js_t = [r["mean_js_off_diag"] for r in src_rows]
        ax.plot(
            epochs,
            js_t,
            color=palette[idx],
            linestyle=line_styles[idx % len(line_styles)],
            label=SHORT_NAMES.get(source, source),
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Mean off-diagonal JS")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    paths = savefig_paper(fig, "js_trajectories", dir=str(fig_dir))
    logger.info("Wrote %s", paths.get("png"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_228" / "all_results.json",
    )
    parser.add_argument(
        "--correlations",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_228" / "correlations.json",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "issue_228",
    )
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        raise SystemExit(f"Missing {args.input} — run aggregate_issue228.py first")
    if not args.correlations.exists():
        raise SystemExit(f"Missing {args.correlations} — run aggregate_issue228.py first")

    with open(args.input) as f:
        all_results = json.load(f)
    with open(args.correlations) as f:
        correlations = json.load(f)

    _figure_a_hero(all_results, args.figure_dir)
    _figure_b_within_source(all_results, args.figure_dir)
    _figure_c_rho_comparison(correlations, args.figure_dir)
    _figure_d_trajectories(all_results, args.figure_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
