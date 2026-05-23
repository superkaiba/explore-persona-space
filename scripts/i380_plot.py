#!/usr/bin/env python3
"""Stage D of issue #380: hero figure (forest + scatter).

Two-panel figure built via the ``paper-plots`` skill conventions
(``src/explore_persona_space/analysis/paper_plots.py``):

  - Left panel (forest): Spearman rho raw + length-partial for
    {JS-from-baseline, mean pairwise JS} with bootstrap 95% CI whiskers.
    Color raw blue, length-partial orange.
  - Right panel (scatter): primary predictor (JS-from-baseline)
    length-residualized x vs source rate length-residualized y, 48 dots
    colored by helpful-family membership, persona-name labels on
    outliers, trend line + 95% CI ribbon.

Output: ``figures/issue_380/{hero.pdf, hero.png, hero.meta.json}``.

Usage:
    uv run python scripts/i380_plot.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

PANEL_N48 = PROJECT_ROOT / "eval_results/issue_296/length_rate_correlation_n48.json"
JS_FROM_BASELINE = PROJECT_ROOT / "eval_results/issue_380/js_from_baseline.json"
PAIRWISE_REDUCTIONS = PROJECT_ROOT / "eval_results/issue_380/pairwise_reductions.json"
CORR_RESULTS = PROJECT_ROOT / "eval_results/issue_380/correlation_results.json"
OUT_DIR = PROJECT_ROOT / "figures/issue_380"

HELPFUL_FAMILY = [
    "helpful_assistant",
    "i_am_helpful",
    "ai_assistant",
    "chat_assistant",
    "virtual_assistant",
    "chatbot",
    "friendly_ai",
    "smart_helper",
    "ai_tool",
    "ai",
    "qwen_default",
]

# Plain-English labels for the forest plot rows. NO Hydra slugs.
ROW_LABELS = [
    "Output-distance from assistant baseline",
    "Mean output-distance to other personas",
]


def _rank_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear residual of rank(a) on rank(b)."""
    ar = rankdata(a).astype(np.float64)
    br = rankdata(b).astype(np.float64)
    a_c = ar - ar.mean()
    b_c = br - br.mean()
    return a_c - (np.dot(a_c, b_c) / np.dot(b_c, b_c)) * b_c


def _draw_forest(ax: plt.Axes, results: dict) -> None:
    """Forest plot: raw rho + length-partial rho for both predictors."""
    raw_color = paper_palette_role("primary")
    partial_color = paper_palette_role("baseline")

    predictors = [
        ("js_from_baseline", ROW_LABELS[0]),
        ("mean_pairwise_js", ROW_LABELS[1]),
    ]

    # Two rows per predictor: raw, partial. Use y-offsets for visual grouping.
    y_positions: list[float] = []
    y_labels: list[str] = []
    for i, (key, label) in enumerate(predictors):
        center = (len(predictors) - 1 - i) * 1.4
        y_partial = center + 0.25
        y_raw = center - 0.25

        block = results["predictors"][key]
        raw_rho = block["raw_spearman"]["rho"]
        raw_n = block["n"]
        # Crude raw CI via bootstrap-free Fisher-z (cheap visual). We DO have
        # bootstrap samples on the partial rho — we don't have raw-rho
        # bootstrap, so use the Fisher-z normal-approx 95% interval.
        raw_lo, raw_hi = _fisher_z_ci(raw_rho, raw_n)

        partial_rho = block["length_partial_spearman"]["rho"]
        partial_lo, partial_hi = block["length_partial_bootstrap_ci95"]

        ax.errorbar(
            [raw_rho],
            [y_raw],
            xerr=[[raw_rho - raw_lo], [raw_hi - raw_rho]],
            fmt="o",
            color=raw_color,
            capsize=4,
            label="Raw Spearman" if i == 0 else None,
        )
        ax.errorbar(
            [partial_rho],
            [y_partial],
            xerr=[[partial_rho - partial_lo], [partial_hi - partial_rho]],
            fmt="s",
            color=partial_color,
            capsize=4,
            label="Length-controlled Spearman" if i == 0 else None,
        )

        y_positions.append(center)
        y_labels.append(label)

    ax.axvline(0.0, color="#888888", lw=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0.5, color="#888888", lw=0.4, linestyle=":", alpha=0.5)
    ax.axvline(-0.5, color="#888888", lw=0.4, linestyle=":", alpha=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Spearman correlation with source rate")
    ax.set_ylabel("")
    ax.legend(loc="lower right", frameon=False)


def _fisher_z_ci(rho: float, n: int) -> tuple[float, float]:
    """95% Fisher-z normal-approx CI on Spearman rho. Visual cue only."""
    if n < 4 or abs(rho) >= 1.0:
        return float(rho), float(rho)
    se = 1.0 / np.sqrt(n - 3)
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    lo_z, hi_z = z - 1.96 * se, z + 1.96 * se
    lo = (np.exp(2 * lo_z) - 1) / (np.exp(2 * lo_z) + 1)
    hi = (np.exp(2 * hi_z) - 1) / (np.exp(2 * hi_z) + 1)
    return float(lo), float(hi)


def _draw_scatter(ax: plt.Axes, panel: pd.DataFrame) -> None:
    """Scatter: length-residualized js_from_baseline vs length-residualized rate."""
    helpful_color = paper_palette_role("accent")
    other_color = paper_palette_role("primary")

    x_res = _rank_residual(panel["js_from_baseline"].to_numpy(), panel["log_tokens"].to_numpy())
    y_res = _rank_residual(panel["rate_n48"].to_numpy(), panel["log_tokens"].to_numpy())

    helpful_mask = panel["is_helpful_family"].to_numpy()
    ax.scatter(
        x_res[~helpful_mask],
        y_res[~helpful_mask],
        s=28,
        color=other_color,
        label="Other personas",
        alpha=0.85,
        edgecolors="none",
    )
    ax.scatter(
        x_res[helpful_mask],
        y_res[helpful_mask],
        s=28,
        color=helpful_color,
        label="Helpful-assistant family",
        alpha=0.85,
        edgecolors="none",
    )

    # OLS trend line on rank-residuals (visual cue; partial-Spearman is the
    # statistical headline).
    if len(x_res) >= 4:
        coeffs = np.polyfit(x_res, y_res, deg=1)
        xx = np.linspace(x_res.min(), x_res.max(), 50)
        yy = coeffs[0] * xx + coeffs[1]
        ax.plot(xx, yy, color="#555555", lw=1.0, linestyle="--", alpha=0.7)

    # Label outliers: top-3 and bottom-3 by x_res magnitude.
    order = np.argsort(np.abs(x_res))[::-1]
    n_label = min(6, len(order))
    for idx in order[:n_label]:
        ax.annotate(
            panel.iloc[idx]["source"],
            xy=(x_res[idx], y_res[idx]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
            alpha=0.85,
        )

    ax.axhline(0.0, color="#cccccc", lw=0.6, linestyle="-")
    ax.axvline(0.0, color="#cccccc", lw=0.6, linestyle="-")
    ax.set_xlabel("Output-distance from assistant baseline (length-controlled)")
    ax.set_ylabel("Source rate (length-controlled)")
    ax.legend(loc="best", frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    set_paper_style("blog")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs.
    panel_rows = json.loads(PANEL_N48.read_text())["rows"]
    js_baseline = json.loads(JS_FROM_BASELINE.read_text())["values"]
    pairwise_red = json.loads(PAIRWISE_REDUCTIONS.read_text())["reductions"]
    corr_results = json.loads(CORR_RESULTS.read_text())

    panel = pd.DataFrame(panel_rows)
    panel["js_from_baseline"] = panel["source"].map(js_baseline)
    panel["mean_pairwise_js"] = panel["source"].map({k: v["mean"] for k, v in pairwise_red.items()})
    panel["log_tokens"] = np.log(panel["tokens"].astype(float) + 1.0)
    panel["is_helpful_family"] = panel["source"].isin(HELPFUL_FAMILY)

    # Drop any source with missing predictors (should be 0 if Stage B ran end-to-end).
    missing = panel["js_from_baseline"].isna().sum()
    if missing:
        logger.warning("Dropping %d sources missing js_from_baseline.", int(missing))
        panel = panel.dropna(subset=["js_from_baseline", "mean_pairwise_js"]).reset_index(drop=True)

    n = len(panel)
    rho_part = corr_results["predictors"]["js_from_baseline"]["length_partial_spearman"]["rho"]
    p_part = corr_results["predictors"]["js_from_baseline"]["length_partial_spearman"]["p"]
    raw_rho, raw_p = spearmanr(panel["js_from_baseline"], panel["rate_n48"])

    # Two-panel figure: forest (left) + scatter (right).
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.4), width_ratios=[1.0, 1.05])

    _draw_forest(ax_l, corr_results)
    _draw_scatter(ax_r, panel)

    set_title_subtitle(
        ax_l,
        title="Distance from assistant baseline vs source rate",
        subtitle=(
            f"N={n}, raw rho={raw_rho:+.2f} (p={raw_p:.3g}), "
            f"length-controlled rho={rho_part:+.2f} (p={p_part:.3g})"
        ),
    )
    set_title_subtitle(
        ax_r,
        title="Length-controlled scatter (primary predictor)",
        subtitle="48 persona prompts, colored by helpful-assistant family membership",
    )

    fig.tight_layout()

    written = savefig_paper(fig, "hero", dir=out_dir)
    for fmt, path in written.items():
        logger.info("Wrote %s: %s", fmt, path)

    plt.close(fig)


if __name__ == "__main__":
    main()
