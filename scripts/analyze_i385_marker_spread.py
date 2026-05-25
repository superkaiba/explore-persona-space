#!/usr/bin/env python3
"""Analyze marker-spread eval results for issue #385.

Reads:
  - eval_results/issue_385/seed{S}/summary.json
        (or step{N}/marker_rates.json directly)
  - eval_results/issue_385/predictors_base.json
        (base-model cosine + JS to librarian)
  - eval_results/issue_385/predictors_per_checkpoint.json   (optional, diagnostic)

Computes (plan §5.6):
  1. First-crossing step per bystander at thresholds {5%, 25%, 50%} under the
     sustained-crossing rule (rate >= threshold at current step AND at the
     next checkpoint). Censored values = max(steps) + 1.
  2. Spearman rho_cos and rho_JS against first-crossing step at each threshold
     over n=26 bystanders (no_persona excluded, plan §5.2(a) and §12).
  3. IQR / median ratio of first-crossing step (kill criterion 2, plan §3).
  4. Per-checkpoint cosine-vs-JS rank correlation drift (if per-checkpoint
     predictors are provided).

Writes:
  - eval_results/issue_385/analysis_seed{S}.json   (aggregated metrics)
  - figures/issue_385/fig_emission_vs_step_by_cosine.png + .pdf + .meta.json
  - figures/issue_385/fig_emission_vs_step_by_JS.png  + .pdf + .meta.json
  - figures/issue_385/fig_crossing_vs_cosine_scatter.png + ...
  - figures/issue_385/fig_crossing_vs_JS_scatter.png    + ...
  - figures/issue_385/fig_predictor_drift.png           + ...   (optional)

Usage:
    uv run python scripts/analyze_i385_marker_spread.py \\
      --eval-summary eval_results/issue_385/seed42/summary.json \\
      --predictors-base eval_results/issue_385/predictors_base.json \\
      --predictors-per-checkpoint eval_results/issue_385/predictors_per_checkpoint.json \\
      --output-analysis eval_results/issue_385/analysis_seed42.json \\
      --figures-dir figures/issue_385
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

THRESHOLDS = (0.05, 0.25, 0.50)
# Plan §5.2(a) + §12: no_persona excluded from the n=26 Spearman rank test
# because its base-model JS row is computed under literal-empty-system ChatML
# (not Qwen-default). Included as a visual point in figures.
EXCLUDED_FROM_RANK_TEST = ("no_persona",)


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _bystander_rates_from_summary(
    summary: dict[str, Any],
) -> tuple[list[int], list[str], dict[str, dict[int, float]]]:
    """Re-shape summary.json into per-bystander step -> rate mapping.

    Returns (steps_sorted, bystander_names, rates) where
    rates[bystander][step] = emission_rate.
    """
    steps_sorted = sorted({row["step"] for row in summary["rows"]})
    bystanders = list(summary["bystanders"])
    rates: dict[str, dict[int, float]] = {b: {} for b in bystanders}
    for row in summary["rows"]:
        s = int(row["step"])
        for bys, rate in row["per_bystander_rate"].items():
            rates[bys][s] = float(rate)
    return steps_sorted, bystanders, rates


def _first_crossing_step(
    rates_by_step: dict[int, float],
    steps_sorted: list[int],
    threshold: float,
) -> int:
    """Return the first step at which `rate >= threshold` AND the next step
    also `rate >= threshold` (sustained-crossing rule, plan §5.6.1).

    If the bystander never sustains crossing, returns `max(steps_sorted) + 1`
    (right-censored). For the LAST step (no next), require only current >=
    threshold (no two-step look-ahead is possible).
    """
    if not steps_sorted:
        return -1
    for i, s in enumerate(steps_sorted):
        if rates_by_step.get(s, 0.0) < threshold:
            continue
        if i + 1 < len(steps_sorted):
            next_s = steps_sorted[i + 1]
            if rates_by_step.get(next_s, 0.0) >= threshold:
                return s
        else:
            # Last checkpoint — current alone is the only evidence we have.
            return s
    return steps_sorted[-1] + 1


def _predictor_dict_from_json(
    predictors_payload: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    cos = dict(predictors_payload["cosine_to_source"])
    js = dict(predictors_payload["js_to_source"])
    return cos, js


def _spearman_safe(
    xs: list[float],
    ys: list[float],
) -> tuple[float, float, int]:
    """Spearman (rho, two-sided p, n_valid). NaN values are excluded pairwise."""
    arr_x = np.asarray(xs, dtype=float)
    arr_y = np.asarray(ys, dtype=float)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    if mask.sum() < 3:
        return (float("nan"), float("nan"), int(mask.sum()))
    res = stats.spearmanr(arr_x[mask], arr_y[mask])
    return float(res.correlation), float(res.pvalue), int(mask.sum())


# ── Figures ───────────────────────────────────────────────────────────────────


def _fig_emission_vs_step(
    steps_sorted: list[int],
    bystanders: list[str],
    rates: dict[str, dict[int, float]],
    predictor: dict[str, float],
    predictor_name: str,
    out_path_base: str,
    figures_dir: Path,
) -> None:
    """27 colored emission curves on one axis, color = predictor-to-source.

    plan §5.6 fig_emission_vs_step_by_{cosine,JS}.png — alt-text and axis
    labels in plain English (no math notation; CLAUDE.md plain-English rule).
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    pred_values = [predictor.get(b, float("nan")) for b in bystanders]
    pred_array = np.array(pred_values, dtype=float)
    finite = np.isfinite(pred_array)
    if finite.sum() < 1:
        raise RuntimeError(f"No finite predictor values for {predictor_name}")
    vmin, vmax = float(pred_array[finite].min()), float(pred_array[finite].max())
    cmap = plt.get_cmap("viridis")

    for bys in bystanders:
        ys = [rates[bys].get(s, float("nan")) for s in steps_sorted]
        pred_val = predictor.get(bys, float("nan"))
        if not np.isfinite(pred_val):
            color = "lightgrey"
            alpha = 0.5
        else:
            norm_val = (pred_val - vmin) / (vmax - vmin + 1e-12)
            color = cmap(norm_val)
            alpha = 0.85
        ax.plot(steps_sorted, ys, color=color, alpha=alpha, linewidth=1.4)

    for thr in THRESHOLDS:
        ax.axhline(thr, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(
            steps_sorted[-1],
            thr,
            f" {int(thr * 100)}%",
            va="center",
            ha="left",
            color="grey",
            fontsize=8,
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Marker emission rate (per bystander)")
    ax.set_xscale("symlog")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Marker spread across {len(bystanders)} bystanders, colored by base-model "
        f"{predictor_name.replace('_', ' ')} to librarian"
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85)
    cbar.set_label(f"Base-model {predictor_name.replace('_', ' ')} to librarian")

    savefig_paper(fig, out_path_base, dir=str(figures_dir))
    plt.close(fig)


def _fig_crossing_scatter(
    crossings: dict[str, int],
    predictor: dict[str, float],
    bystanders: list[str],
    predictor_name: str,
    rho: float,
    p: float,
    n: int,
    out_path_base: str,
    figures_dir: Path,
    censored_step_value: int,
) -> None:
    """Scatter: x = base-model predictor; y = first-crossing step.

    Censored bystanders are plotted as open triangles at the top.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    color = paper_palette_role("primary")
    censor_color = paper_palette_role("baseline")

    for bys in bystanders:
        x = predictor.get(bys, float("nan"))
        y = crossings.get(bys, censored_step_value)
        if not np.isfinite(x):
            continue
        if y >= censored_step_value:
            ax.scatter(
                x,
                censored_step_value,
                marker="^",
                facecolors="none",
                edgecolors=censor_color,
                s=42,
                linewidths=1.2,
            )
        else:
            ax.scatter(x, y, color=color, alpha=0.85, s=42)

    ax.set_xlabel(f"Base-model {predictor_name.replace('_', ' ')} to librarian")
    ax.set_ylabel("First sustained-crossing training step")
    ax.set_yscale("symlog")
    ax.set_title(
        f"First crossing at 5% emission vs {predictor_name.replace('_', ' ')} "
        f"(rho={rho:.2f}, p={p:.3g}, n={n})"
    )
    savefig_paper(fig, out_path_base, dir=str(figures_dir))
    plt.close(fig)


def _fig_predictor_drift(
    per_ckpt_payload: dict[str, Any],
    bystanders: list[str],
    out_path_base: str,
    figures_dir: Path,
) -> None:
    """Per-checkpoint Spearman rho between cosine and JS over bystanders.

    Compares against the body-stated base value rho_base = 0.94 (#341).
    """
    set_paper_style()
    rows = per_ckpt_payload.get("rows", [])
    if not rows:
        logger.warning("predictors_per_checkpoint.json has no rows; skipping drift figure")
        return
    steps = [r["step"] for r in rows]
    rhos = []
    for r in rows:
        cos = r["cosine_to_source"]
        js = r["js_to_source"]
        xs = [cos.get(b, float("nan")) for b in bystanders]
        ys = [js.get(b, float("nan")) for b in bystanders]
        rho, _, _ = _spearman_safe(xs, ys)
        rhos.append(rho)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(steps, rhos, marker="o", color=paper_palette_role("primary"))
    ax.axhline(0.94, color="grey", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.text(
        steps[-1],
        0.94,
        "  rho_base = 0.94 (#341)",
        va="center",
        ha="left",
        color="grey",
        fontsize=8,
    )
    ax.set_xscale("symlog")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Rank correlation between cosine and JS to librarian")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Predictor agreement (cosine vs JS) over training")
    savefig_paper(fig, out_path_base, dir=str(figures_dir))
    plt.close(fig)


# ── Driver ────────────────────────────────────────────────────────────────────


def run_analysis(args: argparse.Namespace) -> None:
    summary = _load_summary(Path(args.eval_summary))
    steps_sorted, bystanders, rates = _bystander_rates_from_summary(summary)
    logger.info("Loaded summary: %d steps, %d bystanders", len(steps_sorted), len(bystanders))

    predictors_base = json.loads(Path(args.predictors_base).read_text())
    cos_base, js_base = _predictor_dict_from_json(predictors_base)
    logger.info(
        "Loaded base predictors: %d cosine entries, %d JS entries",
        len(cos_base),
        len(js_base),
    )

    # First-crossing step per threshold per bystander
    censored = steps_sorted[-1] + 1
    crossings: dict[float, dict[str, int]] = {}
    for thr in THRESHOLDS:
        crossings[thr] = {
            bys: _first_crossing_step(rates[bys], steps_sorted, thr) for bys in bystanders
        }

    # Bystanders eligible for the rank test (drop excluded names)
    rank_bystanders = [b for b in bystanders if b not in EXCLUDED_FROM_RANK_TEST]
    logger.info(
        "Rank test bystander set: %d (dropped %s)",
        len(rank_bystanders),
        list(EXCLUDED_FROM_RANK_TEST),
    )

    # Spearman per threshold and per predictor
    rank_tests: dict[str, dict[str, dict]] = {}
    for thr in THRESHOLDS:
        thr_key = f"thr_{int(thr * 100):03d}pct"
        crossings_thr = crossings[thr]
        xs_cos = [cos_base.get(b, float("nan")) for b in rank_bystanders]
        xs_js = [js_base.get(b, float("nan")) for b in rank_bystanders]
        ys = [crossings_thr[b] for b in rank_bystanders]

        rho_cos, p_cos, n_cos = _spearman_safe(xs_cos, ys)
        rho_js, p_js, n_js = _spearman_safe(xs_js, ys)
        rank_tests[thr_key] = {
            "threshold": thr,
            "n_bystanders_eligible": len(rank_bystanders),
            "cosine": {"rho": rho_cos, "p_two_sided": p_cos, "n": n_cos},
            "js": {"rho": rho_js, "p_two_sided": p_js, "n": n_js},
        }
        logger.info(
            "Threshold %d%%: rho_cos=%.3f (p=%.4g, n=%d); rho_js=%.3f (p=%.4g, n=%d)",
            int(thr * 100),
            rho_cos,
            p_cos,
            n_cos,
            rho_js,
            p_js,
            n_js,
        )

    # IQR / median ratio at the primary 5% threshold (kill criterion 2)
    primary_crossings = np.array(
        [crossings[0.05][b] for b in rank_bystanders if np.isfinite(crossings[0.05][b])],
        dtype=float,
    )
    if primary_crossings.size > 0:
        med = float(np.median(primary_crossings))
        q75, q25 = np.percentile(primary_crossings, [75, 25])
        iqr = float(q75 - q25)
        iqr_over_median = float(iqr / med) if med > 0 else float("nan")
    else:
        med, iqr, iqr_over_median = float("nan"), float("nan"), float("nan")
    logger.info(
        "Primary (5%%) first-crossing: median=%.1f, IQR=%.1f, IQR/median=%.3f",
        med,
        iqr,
        iqr_over_median,
    )

    # Coverage (how many of n bystanders cross by horizon)
    coverage = {
        f"thr_{int(thr * 100):03d}pct": int(
            sum(1 for b in rank_bystanders if crossings[thr][b] <= steps_sorted[-1])
        )
        for thr in THRESHOLDS
    }

    # ── Figures ───────────────────────────────────────────────────────────────
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _fig_emission_vs_step(
        steps_sorted=steps_sorted,
        bystanders=bystanders,
        rates=rates,
        predictor=cos_base,
        predictor_name="cosine",
        out_path_base="fig_emission_vs_step_by_cosine",
        figures_dir=figures_dir,
    )
    _fig_emission_vs_step(
        steps_sorted=steps_sorted,
        bystanders=bystanders,
        rates=rates,
        predictor=js_base,
        predictor_name="JS divergence",
        out_path_base="fig_emission_vs_step_by_JS",
        figures_dir=figures_dir,
    )

    # Scatters use the 5% threshold (primary metric).
    primary_thr_key = "thr_005pct"
    rt = rank_tests[primary_thr_key]
    _fig_crossing_scatter(
        crossings=crossings[0.05],
        predictor=cos_base,
        bystanders=rank_bystanders,
        predictor_name="cosine",
        rho=rt["cosine"]["rho"],
        p=rt["cosine"]["p_two_sided"],
        n=rt["cosine"]["n"],
        out_path_base="fig_crossing_vs_cosine_scatter",
        figures_dir=figures_dir,
        censored_step_value=censored,
    )
    _fig_crossing_scatter(
        crossings=crossings[0.05],
        predictor=js_base,
        bystanders=rank_bystanders,
        predictor_name="JS divergence",
        rho=rt["js"]["rho"],
        p=rt["js"]["p_two_sided"],
        n=rt["js"]["n"],
        out_path_base="fig_crossing_vs_JS_scatter",
        figures_dir=figures_dir,
        censored_step_value=censored,
    )

    drift_summary: dict[str, Any] | None = None
    if args.predictors_per_checkpoint:
        per_ckpt = json.loads(Path(args.predictors_per_checkpoint).read_text())
        _fig_predictor_drift(
            per_ckpt_payload=per_ckpt,
            bystanders=rank_bystanders,
            out_path_base="fig_predictor_drift",
            figures_dir=figures_dir,
        )
        # Per-step rho summary into the analysis JSON
        drift_rows: list[dict] = []
        for r in per_ckpt.get("rows", []):
            cos = r["cosine_to_source"]
            js = r["js_to_source"]
            xs = [cos.get(b, float("nan")) for b in rank_bystanders]
            ys = [js.get(b, float("nan")) for b in rank_bystanders]
            rho, p, n = _spearman_safe(xs, ys)
            drift_rows.append({"step": r["step"], "rho_cos_js": rho, "p": p, "n": n})
        drift_summary = {"per_step": drift_rows}

    # ── Write analysis JSON ───────────────────────────────────────────────────
    analysis = {
        "metadata": {
            "eval_summary": str(Path(args.eval_summary).resolve()),
            "predictors_base": str(Path(args.predictors_base).resolve()),
            "predictors_per_checkpoint": (
                str(Path(args.predictors_per_checkpoint).resolve())
                if args.predictors_per_checkpoint
                else None
            ),
            "thresholds": list(THRESHOLDS),
            "rank_test_excludes": list(EXCLUDED_FROM_RANK_TEST),
            "censored_step_value": censored,
            "n_bystanders_total": len(bystanders),
            "n_bystanders_rank_test": len(rank_bystanders),
            "steps": steps_sorted,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
        "first_crossing_steps": {
            f"thr_{int(thr * 100):03d}pct": {b: int(crossings[thr][b]) for b in bystanders}
            for thr in THRESHOLDS
        },
        "rank_tests": rank_tests,
        "iqr_over_median_5pct": {
            "median": med,
            "iqr": iqr,
            "ratio": iqr_over_median,
            "n_uncensored": int(primary_crossings.size),
        },
        "coverage": coverage,
        "predictor_drift": drift_summary,
    }

    out_path = Path(args.output_analysis)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis, indent=2))
    logger.info("Wrote %s", out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-summary", required=True)
    parser.add_argument("--predictors-base", required=True)
    parser.add_argument("--predictors-per-checkpoint", default=None)
    parser.add_argument("--output-analysis", required=True)
    parser.add_argument("--figures-dir", default="figures/issue_385")
    return parser


def main():
    args = build_parser().parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
