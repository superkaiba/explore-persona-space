#!/usr/bin/env python3
"""Issue #523 — forest plot of the five Phase D bars.

Reads ``eval_results/issue_523/scoring/forest_plot_data.json`` (written by
``scripts/issue523_phase_d_scoring.py``) and emits the headline figure:

  figures/issue_523/headline_forest.png   (+ .pdf + .meta.json)

The hero figure per plan v2 §4 Phase D — five bars (the #502 in-sample
0.34 reference + the four held-out bars + the JS comparator) with 95%
fold-bootstrap CIs, caption text noting the paired per-fold Δ
(cell-fixed vs nested-search = selection inflation; cell-fixed vs JS =
L22-beats-JS test) from the same scoring JSON.

Usage::

    uv run python scripts/issue523_plot_forest.py
    uv run python scripts/issue523_plot_forest.py --in-json <path> --out <path>
"""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")
except Exception:
    pass

logger = logging.getLogger("i523.plot_forest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_IN = PROJECT_ROOT / "eval_results" / "issue_523" / "scoring" / "forest_plot_data.json"
DEFAULT_OUT = PROJECT_ROOT / "figures" / "issue_523" / "headline_forest.png"

# Plot order (top → bottom in the figure). Slugs must match Phase D outputs.
PLOT_ORDER = (
    "reference_502",  # #502 reference, no CI
    "cell_fixed_seed42_nonstyl_heldout",
    "cell_fixed_seed43_nonstyl_heldout",
    "nested_search_seed42_nonstyl_heldout",
    "js_baseline_seed42_nonstyl_heldout",
)
LABELS = {
    "reference_502": "#502 in-sample 0.34 (reference)",
    "cell_fixed_seed42_nonstyl_heldout": "L22 gauss_kl cell-fixed, seed-42, held-out",
    "cell_fixed_seed43_nonstyl_heldout": "L22 gauss_kl cell-fixed, seed-43, held-out",
    "nested_search_seed42_nonstyl_heldout": "Nested 1737-cell search, seed-42, held-out",
    "js_baseline_seed42_nonstyl_heldout": "JS baseline cell-fixed, seed-42, held-out",
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            env={**os.environ},  # epm-lint: subprocess explicit env
        ).strip()
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot the issue 523 headline forest figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--in-json", type=Path, default=DEFAULT_IN, help="Phase D forest data JSON.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    if not args.in_json.exists():
        raise FileNotFoundError(f"Phase D forest data {args.in_json} not found; run Phase D first.")

    forest = json.loads(args.in_json.read_text())
    bars_by_slug = {b["slug"]: b for b in forest["bars"]}
    ref = forest["reference_502_in_sample"]

    rows = []
    for slug in PLOT_ORDER:
        if slug == "reference_502":
            rows.append(
                {
                    "label": LABELS[slug],
                    "value": ref["value"],
                    "ci_lo": None,
                    "ci_hi": None,
                    "slug": slug,
                    "paired_delta": None,
                }
            )
        elif slug in bars_by_slug:
            b = bars_by_slug[slug]
            rows.append(
                {
                    "label": LABELS[slug],
                    "value": b["point_estimate"],
                    "ci_lo": b["ci_2_5"],
                    "ci_hi": b["ci_97_5"],
                    "slug": slug,
                    "paired_delta": b.get("paired_delta_vs_baseline"),
                }
            )
        else:
            logger.warning("Bar %s missing from forest data; skipping", slug)
            continue

    fig, ax = plt.subplots(figsize=(10, 4.5))
    y_pos = np.arange(len(rows))
    colors = ["#999999", "#1f77b4", "#1f77b4", "#ff7f0e", "#2ca02c"][: len(rows)]
    for i, row in enumerate(rows):
        v = row["value"]
        if v is None or not np.isfinite(v):
            continue
        if row["ci_lo"] is not None and row["ci_hi"] is not None and np.isfinite(row["ci_lo"]):
            err_lo = max(0.0, v - row["ci_lo"])
            err_hi = max(0.0, row["ci_hi"] - v)
            ax.errorbar(
                v,
                i,
                xerr=[[err_lo], [err_hi]],
                fmt="o",
                color=colors[i],
                ecolor=colors[i],
                capsize=4,
                markersize=8,
                lw=2,
            )
        else:
            ax.plot(v, i, marker="s", color=colors[i], markersize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", lw=0.6, linestyle=":")
    ax.set_xlabel("Non-stylized 156-pair CV R²  (length-controlled, fold-bootstrap 95% CI)")
    ax.set_title(
        "Issue 523 — honest held-out test of #502's L22 gauss_kl predictor",
        fontsize=13,
        fontweight="bold",
    )

    # Caption: paired per-fold Δ where available (selection inflation, L22-beats-JS).
    caption_parts: list[str] = []
    for row in rows:
        pd = row.get("paired_delta")
        if pd and pd.get("mean_delta") is not None and np.isfinite(pd["mean_delta"]):
            caption_parts.append(
                f"{row['slug']} − {pd['baseline_label']}: Δ = {pd['mean_delta']:+.3f} "
                f"[{pd['ci_2_5']:+.3f}, {pd['ci_97_5']:+.3f}]"
            )
    if caption_parts:
        fig.text(
            0.5,
            -0.02,
            "\n".join(caption_parts),
            ha="center",
            va="top",
            fontsize=8,
            wrap=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    pdf_out = args.out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    logger.info("wrote %s and %s", args.out, pdf_out)

    # Companion meta.json per paper-plots convention.
    meta = {
        "schema_version": 1,
        "figure_id": "headline_forest",
        "issue": 523,
        "title": "L22 gauss_kl predictor — held-out forest plot",
        "data_source": str(args.in_json.relative_to(PROJECT_ROOT)),
        "n_bars": len(rows),
        "bars": [{"slug": r["slug"], "label": r["label"], "value": r["value"]} for r in rows],
        "provenance": {
            "git_sha": _git_sha(),
            "timestamp_utc": _now_iso(),
            "python": platform.python_version(),
        },
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
