#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ρ, ×, M⁺) in scientific docstrings + labels.
"""Issue #811 — side-by-side mean-vs-turn_nl figures (plan §6.3; over-produce).

Reads ``eval_results/issue_811/mean_vs_turn_nl_summary.json`` +
``function_change_{summary}.json`` (written by ``issue811_analyze.py``) and
over-produces the plan §6.3 figure candidates so the analyzer picks the hero:

- ``function_change_mean_vs_turn_nl.png`` (HERO) — ``Δ_med / floor_combined`` per
  behavior × layer, mean vs turn_nl grouped bars, 1× floor line. The direct
  "does the verdict change" read.
- ``delta_scatter_mean_vs_turn_nl.png`` — scatter of mean Δ_med vs turn_nl Δ_med
  per behavior×layer cell, 45° line.
- ``chain_rho_forest_mean_vs_turn_nl.png`` — ρ(M0) vs ρ(M⁺) per behavior×layer,
  both summaries.
- ``validity_gate_bars.png`` — the base-leg MLP-vs-shuffle gate margin per
  behavior×layer×summary (which turn_nl reads are trusted).

Uses the project ``set_paper_style`` rcParams; matplotlib only. Each figure
writes a ``.png`` under ``figures/issue_811/``. No annotations/arrows (project
plot rule).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    set_paper_style,
)

logger = logging.getLogger("issue811.figures")

HEADLINE_BEHAVIORS = ("em", "sycophancy", "fact")
SWEEP_LAYERS = (7, 14, 21)
SUMMARIES = ("mean", "turn_nl")
BEHAVIOR_LABEL = {"em": "harmful-compliance", "sycophancy": "sycophancy", "fact": "taught fact"}
# Per-summary palette role (valid roles: accent/baseline/control/neutral/primary).
_SUMMARY_ROLE = {"mean": "baseline", "turn_nl": "primary", "maxp": "accent"}


def _summary_color(summary: str) -> str:
    """Stable per-summary color (mean=baseline, turn_nl=primary, maxp=accent)."""
    return paper_palette_role(_SUMMARY_ROLE.get(summary, "control"))


def _cell_ratio(row: dict, summary: str) -> float | None:
    """Δ_med / floor_combined for a pair-table row's summary column (None if absent)."""
    s = row.get(summary)
    if not s:
        return None
    dm, floor = s.get("Delta_med"), s.get("floor_combined")
    if dm is None or floor is None or floor <= 0:
        return None
    return dm / floor


def _ci_ratio_halfwidths(row: dict, summary: str) -> tuple[float, float] | None:
    """(lower, upper) errorbar half-widths in floor-units from Delta_med_ci (clamped ≥0)."""
    s = row.get(summary)
    if not s:
        return None
    ci = s.get("Delta_med_ci") or {}
    floor = s.get("floor_combined")
    dm = s.get("Delta_med")
    lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
    if None in (floor, dm, lo, hi) or floor <= 0:
        return None
    center = dm / floor
    return max(0.0, center - lo / floor), max(0.0, hi / floor - center)


def _ordered_cells(pair_table: dict) -> list[tuple[str, int, dict]]:
    """(behavior, layer, row) in the headline behavior × swept-layer order present."""
    out = []
    for beh in HEADLINE_BEHAVIORS:
        for li in SWEEP_LAYERS:
            row = pair_table.get(f"{beh}/L{li}")
            if row is not None:
                out.append((beh, li, row))
    return out


def fig_function_change_grouped(
    pair_table: dict, out: Path, summaries: tuple[str, ...] = SUMMARIES
) -> None:
    """HERO: Δ_med/floor grouped bars per summary, per behavior×layer + 1× line."""
    cells = _ordered_cells(pair_table)
    if not cells:
        logger.warning("no cells for the grouped-bar figure — skipping")
        return
    labels = [f"{BEHAVIOR_LABEL[b]}\nL{li}" for b, li, _ in cells]
    x = np.arange(len(cells))
    # For 2 summaries these reduce EXACTLY to the v1 geometry (width 0.38, ±0.19).
    width = 0.76 / len(summaries)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(cells)), 4.2))
    for i, summary in enumerate(summaries):
        vals = [_cell_ratio(row, summary) for _, _, row in cells]
        errs = [_ci_ratio_halfwidths(row, summary) for _, _, row in cells]
        heights = [0.0 if v is None else v for v in vals]
        lo = [0.0 if (v is None or e is None) else e[0] for v, e in zip(vals, errs, strict=True)]
        hi = [0.0 if (v is None or e is None) else e[1] for v, e in zip(vals, errs, strict=True)]
        ax.bar(
            x + (i - (len(summaries) - 1) / 2) * width,
            heights,
            width,
            yerr=[lo, hi],
            capsize=3,
            label=summary,
            color=_summary_color(summary),
        )
    ax.axhline(1.0, color=paper_palette_role("neutral"), linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ_med / floor_combined")
    ax.set_title(f"Function-change Δ vs floor — {' vs '.join(summaries)}")
    ax.legend(title="answer summary")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)


def fig_delta_scatter(
    pair_table: dict, out: Path, x_summary: str = "mean", y_summary: str = "turn_nl"
) -> None:
    """Scatter of ``x_summary`` Δ_med vs ``y_summary`` Δ_med per behavior×layer, 45° line."""
    cells = _ordered_cells(pair_table)
    xs, ys, labels = [], [], []
    for beh, li, row in cells:
        m, t = row.get(x_summary), row.get(y_summary)
        if not m or not t or m.get("Delta_med") is None or t.get("Delta_med") is None:
            continue
        xs.append(m["Delta_med"])
        ys.append(t["Delta_med"])
        labels.append(f"{BEHAVIOR_LABEL[beh]} L{li}")
    if not xs:
        logger.warning("no paired Δ_med cells for the scatter — skipping")
        return
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(xs, ys, color=_summary_color(y_summary), s=40)
    lim = max(max(xs), max(ys)) * 1.1 or 1.0
    ax.plot([0, lim], [0, lim], color=paper_palette_role("neutral"), linestyle="--", linewidth=1.0)
    for xi, yi, lab in zip(xs, ys, labels, strict=True):
        ax.annotate(lab, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"{x_summary} Δ_med")
    ax.set_ylabel(f"{y_summary} Δ_med")
    ax.set_title(f"Δ_med: {x_summary} vs {y_summary} (per behavior×layer)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)


def fig_chain_rho_forest(
    pair_table: dict, out: Path, summaries: tuple[str, ...] = SUMMARIES
) -> None:
    """ρ(M0) vs ρ(M⁺) per behavior×layer, all summaries (the transfer-chain shift)."""
    cells = _ordered_cells(pair_table)
    rows = []
    for beh, li, row in cells:
        for summary in summaries:
            s = row.get(summary)
            if not s:
                continue
            cr = s.get("chain_rho", {})
            rows.append((f"{BEHAVIOR_LABEL[beh]} L{li} [{summary}]", cr))
    if not rows:
        logger.warning("no chain-ρ rows for the forest — skipping")
        return
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.4 * len(rows))))
    y = np.arange(len(rows))
    m0 = [r[1].get("rho_M0_ridge") for r in rows]
    mp = [r[1].get("rho_Mplus_ridge") for r in rows]
    ax.scatter(
        [v if v is not None else np.nan for v in m0],
        y,
        color=paper_palette_role("baseline"),
        label="ρ(M0)",
    )
    ax.scatter(
        [v if v is not None else np.nan for v in mp],
        y,
        color=paper_palette_role("primary"),
        label="ρ(M⁺)",
    )
    ax.axvline(0.0, color=paper_palette_role("neutral"), linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel("Spearman ρ(r_Bᵀ M̂(c), E=#537 G)")
    ax.set_title(f"Chain-ρ M0 vs M⁺ — {' & '.join(summaries)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)


def fig_validity_gate_bars(
    pair_table: dict, out: Path, summaries: tuple[str, ...] = SUMMARIES
) -> None:
    """Base-leg MLP-vs-shuffle gate margin per behavior×layer×summary (trusted reads)."""
    cells = _ordered_cells(pair_table)
    if not cells:
        logger.warning("no cells for the validity-gate bars — skipping")
        return
    labels = [f"{BEHAVIOR_LABEL[b]}\nL{li}" for b, li, _ in cells]
    x = np.arange(len(cells))
    width = 0.76 / len(summaries)  # v1-identical geometry for 2 summaries
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(cells)), 4.2))
    for i, summary in enumerate(summaries):
        margins = [
            (row.get(summary) or {}).get("gate_margin") if row.get(summary) else None
            for _, _, row in cells
        ]
        heights = [0.0 if m is None else m for m in margins]
        ax.bar(
            x + (i - (len(summaries) - 1) / 2) * width,
            heights,
            width,
            label=summary,
            color=_summary_color(summary),
        )
    ax.axhline(0.0, color=paper_palette_role("neutral"), linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("gate margin (ρ_real − ρ_shuffle)")
    ax.set_title(f"Base-leg MLP-vs-shuffle validity gate — {' vs '.join(summaries)}")
    ax.legend(title="answer summary")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #811 mean-vs-turn_nl figures")
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_811/mean_vs_turn_nl_summary.json",
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures/issue_811")
    ap.add_argument(
        "--summaries",
        nargs="+",
        default=list(SUMMARIES),
        help="answer summaries to render (maxp round: mean turn_nl maxp; the "
        "3-summary set switches to the *_three_summaries.png filenames — the "
        "plan §6.5 hero glob)",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style("blog")
    doc = json.loads(args.summary_json.read_text())
    pt = doc["pair_table"]
    summaries = tuple(args.summaries)
    tag = "three_summaries" if len(summaries) == 3 else "mean_vs_turn_nl"
    fig_function_change_grouped(pt, args.out_dir / f"function_change_{tag}.png", summaries)
    if summaries == tuple(SUMMARIES):
        fig_delta_scatter(pt, args.out_dir / "delta_scatter_mean_vs_turn_nl.png")
    else:
        # Plan §6 figure (b): the NEW summary against EACH reference, 45° lines
        # (+ the reference pair for continuity with the v1 round's read).
        new = [s for s in summaries if s not in ("mean", "turn_nl")]
        for y in new:
            for xref in ("mean", "turn_nl"):
                if xref in summaries:
                    fig_delta_scatter(
                        pt, args.out_dir / f"delta_scatter_{y}_vs_{xref}.png", xref, y
                    )
        if {"mean", "turn_nl"} <= set(summaries):
            fig_delta_scatter(pt, args.out_dir / "delta_scatter_mean_vs_turn_nl.png")
    fig_chain_rho_forest(pt, args.out_dir / f"chain_rho_forest_{tag}.png", summaries)
    fig_validity_gate_bars(pt, args.out_dir / "validity_gate_bars.png", summaries)
    logger.info("[phase=figures] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
