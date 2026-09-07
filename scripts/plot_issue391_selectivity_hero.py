"""Hero plot for task #391: per-factor source-vs-bystander sycophancy selectivity.

Reads the aggregator output at::

    eval_results/issue_391/aggregate/per_factor_selectivity.json

(or wherever ``--input`` points) and renders a two-panel chart:

  * **Left:** paired Δ source-sycophancy and Δ bystander-mean-sycophancy
    per swept factor (A short-vs-long system, C neutral-vs-persona,
    D base-Qwen-vs-Claude training data).
  * **Right:** selectivity Δ (mean-aggregator) per factor, with the
    widest-of-three 95% CI (per-pair percentile bootstrap + source-cluster
    bootstrap + source fixed-effects regression).

Output: ``figures/issue_391/hero_selectivity_by_factor.{png,pdf,meta.json}``.

Run from repo root::

    uv run python scripts/plot_issue391_selectivity_hero.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

DEFAULT_INPUT = Path("eval_results/issue_391/aggregate/per_factor_selectivity.json")
DEFAULT_OUTPUT_STEM = "issue_391/hero_selectivity_by_factor"
DEFAULT_OUTPUT_DIR = Path("figures")


def _row_from_factor_payload(payload: dict) -> dict:
    """Convert one factor's payload into a flat plotting row."""
    return {
        "factor": payload["factor"],
        "plain_english": payload["plain_english"],
        "d_source_mean": payload.get("d_source_mean", 0.0),
        "d_source_lo": payload.get("d_source_ci", [0.0, 0.0])[0],
        "d_source_hi": payload.get("d_source_ci", [0.0, 0.0])[1],
        "d_bys_mean_mean": payload.get("d_bystander_mean_mean", 0.0),
        "d_bys_mean_lo": payload.get("d_bystander_mean_ci", [0.0, 0.0])[0],
        "d_bys_mean_hi": payload.get("d_bystander_mean_ci", [0.0, 0.0])[1],
        "sel_mean_mean": payload.get("selectivity_mean_mean", 0.0),
        "sel_mean_lo": payload.get("selectivity_mean_ci", [0.0, 0.0])[0],
        "sel_mean_hi": payload.get("selectivity_mean_ci", [0.0, 0.0])[1],
        "n_pairs": payload.get("n_pairs", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to per_factor_selectivity.json (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the figure (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=DEFAULT_OUTPUT_STEM,
        help=f"Output filename stem (default: {DEFAULT_OUTPUT_STEM}).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Aggregator output not found at {args.input}. Run "
            "`uv run python -m explore_persona_space.experiments.sycophancy_implantation_391 "
            "--mode aggregate --slab-root eval_results/issue_391 "
            "--output-dir eval_results/issue_391/aggregate` first."
        )

    payload = json.loads(args.input.read_text())
    factor_payloads = payload.get("factor_flips", {})
    if not factor_payloads:
        raise SystemExit(f"No factor_flips in {args.input}; nothing to plot.")

    rows = [_row_from_factor_payload(factor_payloads[f]) for f in ("A", "C", "D")]

    for r in rows:
        print(
            f"  {r['factor']} ({r['plain_english']})  n={r['n_pairs']:>2}  "
            f"Δsrc={r['d_source_mean']:+.3f} [{r['d_source_lo']:+.3f},{r['d_source_hi']:+.3f}]  "
            f"Δbys_mean={r['d_bys_mean_mean']:+.3f} "
            f"[{r['d_bys_mean_lo']:+.3f},{r['d_bys_mean_hi']:+.3f}]  "
            f"Δsel={r['sel_mean_mean']:+.3f} [{r['sel_mean_lo']:+.3f},{r['sel_mean_hi']:+.3f}]",
            flush=True,
        )

    set_paper_style("blog")

    n = len(rows)
    y = np.arange(n)[::-1]
    bar_h = 0.36
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(13.0, 4.6),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.25},
    )

    src_color = paper_palette_role("primary")
    bys_color = paper_palette_role("accent")
    sel_color = paper_palette_role("baseline")

    # Left panel.
    for i, r in enumerate(rows):
        y_src = y[i] + bar_h / 2
        y_bys = y[i] - bar_h / 2
        ax_l.barh(
            y_src,
            r["d_source_mean"],
            height=bar_h,
            color=src_color,
            label="Source-persona Δ sycophancy" if i == 0 else None,
        )
        ax_l.errorbar(
            r["d_source_mean"],
            y_src,
            xerr=[
                [max(0.0, r["d_source_mean"] - r["d_source_lo"])],
                [max(0.0, r["d_source_hi"] - r["d_source_mean"])],
            ],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=3,
        )
        ax_l.barh(
            y_bys,
            r["d_bys_mean_mean"],
            height=bar_h,
            color=bys_color,
            label="Bystander-mean Δ sycophancy" if i == 0 else None,
        )
        ax_l.errorbar(
            r["d_bys_mean_mean"],
            y_bys,
            xerr=[
                [max(0.0, r["d_bys_mean_mean"] - r["d_bys_mean_lo"])],
                [max(0.0, r["d_bys_mean_hi"] - r["d_bys_mean_mean"])],
            ],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=3,
        )
    ax_l.axvline(0, color="#444444", lw=0.8)
    ax_l.set_yticks(y)
    ax_l.set_yticklabels([r["plain_english"] for r in rows])
    ax_l.set_xlabel("Anchor-minus-flipped Δ sycophancy index")
    ax_l.set_ylim(-0.7, n - 0.3)
    ax_l.grid(axis="x", lw=0.4, alpha=0.5)
    ax_l.legend(loc="lower right", frameon=False)
    ax_l.set_title("Source vs bystander shift per factor", fontsize=11)

    # Right panel.
    for i, r in enumerate(rows):
        ax_r.barh(
            y[i],
            r["sel_mean_mean"],
            height=0.55,
            color=sel_color,
        )
        ax_r.errorbar(
            r["sel_mean_mean"],
            y[i],
            xerr=[
                [max(0.0, r["sel_mean_mean"] - r["sel_mean_lo"])],
                [max(0.0, r["sel_mean_hi"] - r["sel_mean_mean"])],
            ],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=3,
        )
    ax_r.axvline(0, color="#444444", lw=0.8)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([])
    ax_r.set_xlabel("Source Δ minus bystander-mean Δ (sycophancy index)")
    ax_r.set_ylim(-0.7, n - 0.3)
    ax_r.grid(axis="x", lw=0.4, alpha=0.5)
    ax_r.set_title("Per-factor selectivity (source minus bystander)", fontsize=11)

    fig.suptitle(
        "Behavioral implantation: which recipe knobs lift source-sycophancy faster than bystanders",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.94,
        "Single-factor screen on Qwen2.5-7B-Instruct; 3 sources x 1 seed x held-out scenarios; "
        "error bars are widest of per-pair, source-cluster, and source fixed-effects 95% CI.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.text(
        0.01,
        0.01,
        f"source: {args.input}",
        fontsize=7,
        color="#888888",
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    savefig_paper(fig, args.output_stem, dir=str(args.output_dir))
    plt.close(fig)
    out_png = args.output_dir / f"{args.output_stem}.png"
    print(f"saved {out_png} + .pdf + .meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
