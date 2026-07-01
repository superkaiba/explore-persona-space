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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    set_paper_style,
)

logger = logging.getLogger("issue811.figures")

HEADLINE_BEHAVIORS = ("em", "sycophancy", "fact")
SWEEP_LAYERS = (7, 14, 21)
SUMMARIES = ("mean", "turn_nl")
BEHAVIOR_LABEL = {"em": "harmful-compliance", "sycophancy": "sycophancy", "fact": "taught fact"}


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


def fig_function_change_grouped(pair_table: dict, out: Path) -> None:
    """HERO: Δ_med/floor grouped bars, mean vs turn_nl, per behavior×layer + 1× line."""
    cells = _ordered_cells(pair_table)
    if not cells:
        logger.warning("no cells for the grouped-bar figure — skipping")
        return
    labels = [f"{BEHAVIOR_LABEL[b]}\nL{li}" for b, li, _ in cells]
    x = np.arange(len(cells))
    width = 0.38
    colors = {"mean": paper_palette_role("baseline"), "turn_nl": paper_palette_role("primary")}
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(cells)), 4.2))
    for i, summary in enumerate(SUMMARIES):
        vals = [_cell_ratio(row, summary) for _, _, row in cells]
        errs = [_ci_ratio_halfwidths(row, summary) for _, _, row in cells]
        heights = [0.0 if v is None else v for v in vals]
        lo = [0.0 if (v is None or e is None) else e[0] for v, e in zip(vals, errs, strict=True)]
        hi = [0.0 if (v is None or e is None) else e[1] for v, e in zip(vals, errs, strict=True)]
        ax.bar(
            x + (i - 0.5) * width,
            heights,
            width,
            yerr=[lo, hi],
            capsize=3,
            label=summary,
            color=colors[summary],
        )
    ax.axhline(1.0, color=paper_palette_role("neutral"), linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ_med / floor_combined")
    ax.set_title("Function-change Δ vs floor — mean vs turn_nl")
    ax.legend(title="answer summary")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)


def fig_delta_scatter(pair_table: dict, out: Path) -> None:
    """Scatter of mean Δ_med vs turn_nl Δ_med per behavior×layer, 45° line."""
    cells = _ordered_cells(pair_table)
    xs, ys, labels = [], [], []
    for beh, li, row in cells:
        m, t = row.get("mean"), row.get("turn_nl")
        if not m or not t or m.get("Delta_med") is None or t.get("Delta_med") is None:
            continue
        xs.append(m["Delta_med"])
        ys.append(t["Delta_med"])
        labels.append(f"{BEHAVIOR_LABEL[beh]} L{li}")
    if not xs:
        logger.warning("no paired Δ_med cells for the scatter — skipping")
        return
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(xs, ys, color=paper_palette_role("primary"), s=40)
    lim = max(max(xs), max(ys)) * 1.1 or 1.0
    ax.plot([0, lim], [0, lim], color=paper_palette_role("neutral"), linestyle="--", linewidth=1.0)
    for xi, yi, lab in zip(xs, ys, labels, strict=True):
        ax.annotate(lab, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean Δ_med")
    ax.set_ylabel("turn_nl Δ_med")
    ax.set_title("Δ_med: mean vs turn_nl (per behavior×layer)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)


def fig_chain_rho_forest(pair_table: dict, out: Path) -> None:
    """ρ(M0) vs ρ(M⁺) per behavior×layer, both summaries (the transfer-chain shift)."""
    cells = _ordered_cells(pair_table)
    rows = []
    for beh, li, row in cells:
        for summary in SUMMARIES:
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
    ax.set_title("Chain-ρ M0 vs M⁺ — mean & turn_nl")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)


def fig_validity_gate_bars(pair_table: dict, out: Path) -> None:
    """Base-leg MLP-vs-shuffle gate margin per behavior×layer×summary (trusted reads)."""
    cells = _ordered_cells(pair_table)
    if not cells:
        logger.warning("no cells for the validity-gate bars — skipping")
        return
    labels = [f"{BEHAVIOR_LABEL[b]}\nL{li}" for b, li, _ in cells]
    x = np.arange(len(cells))
    width = 0.38
    colors = {"mean": paper_palette_role("baseline"), "turn_nl": paper_palette_role("primary")}
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(cells)), 4.2))
    for i, summary in enumerate(SUMMARIES):
        margins = [
            (row.get(summary) or {}).get("gate_margin") if row.get(summary) else None
            for _, _, row in cells
        ]
        heights = [0.0 if m is None else m for m in margins]
        ax.bar(x + (i - 0.5) * width, heights, width, label=summary, color=colors[summary])
    ax.axhline(0.0, color=paper_palette_role("neutral"), linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("gate margin (ρ_real − ρ_shuffle)")
    ax.set_title("Base-leg MLP-vs-shuffle validity gate — mean vs turn_nl")
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
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style("blog")
    doc = json.loads(args.summary_json.read_text())
    pt = doc["pair_table"]
    fig_function_change_grouped(pt, args.out_dir / "function_change_mean_vs_turn_nl.png")
    fig_delta_scatter(pt, args.out_dir / "delta_scatter_mean_vs_turn_nl.png")
    fig_chain_rho_forest(pt, args.out_dir / "chain_rho_forest_mean_vs_turn_nl.png")
    fig_validity_gate_bars(pt, args.out_dir / "validity_gate_bars.png")
    logger.info("[phase=figures] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
