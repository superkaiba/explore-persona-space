#!/usr/bin/env python3
"""Plot the issue #377 hero figure: drift-progression curve.

Fire-rate vs k ∈ {0=A, 5, 10, 20} on the x-axis, fire-rate on y-axis,
Wilson 95% pair-level CIs as error bars. Three multi-turn lines per seed
plus an A-baseline horizontal reference (plan v2 §6.2 round-9 hot-fix):

- ``B@k`` (drift history, blue solid) — drift-progression curve.
- ``B-incontext-turns@k`` (orange dashed) — turn-matched isolation
  control; the v1 ``B-incontext@k`` arm, renamed under plan v2 §5.
- ``B-incontext-length@k`` (green dotted) — length-matched isolation
  control; new at plan v2 §4.3 round-9 hot-fix.

Plus the A baseline as a horizontal grey reference at fire_rate(A), an
H6 (fresh-prompt no-trigger) horizontal reference at the bottom, and
``B-null@k`` as a secondary marker series (no-trigger after drift).

Inputs:
    eval_results/issue_377/run_result.json (aggregated)
    eval_results/issue_377/seed{S}/run_result.json (per-seed)

Outputs:
    figures/issue_377/hero_drift_curve.png
    figures/issue_377/hero_drift_curve.pdf
    figures/issue_377/hero_drift_curve.meta.json

Usage::

    uv run python scripts/issue_377_plot_hero.py
    uv run python scripts/issue_377_plot_hero.py --results-dir eval_results/issue_377
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

K_VALUES: tuple[int, ...] = (5, 10, 20)
X_KS: tuple[int, ...] = (0, 5, 10, 20)  # 0 represents Condition A (no history).

PROJECT_ROOT = Path(__file__).parent.parent


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score 95% CI. Returns ``(rate, lower, upper)`` clamped to [0, 1].

    Local copy to avoid importing from ``scripts/`` (not a Python package).
    Mirrors the implementation in ``scripts/eval_issue377.py``.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    halfwidth = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lo = max(0.0, center - halfwidth)
    hi = min(1.0, center + halfwidth)
    return p, lo, hi


def _load_per_seed(results_dir: Path, seeds: list[int]) -> list[dict]:
    out: list[dict] = []
    for s in seeds:
        path = results_dir / f"seed{s}" / "run_result.json"
        if not path.exists():
            print(f"  WARNING: {path} missing; skipping seed {s}", flush=True)
            continue
        with open(path) as f:
            out.append(json.load(f))
    return out


def _load_aggregated(results_dir: Path) -> dict | None:
    path = results_dir / "run_result.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _curve_with_ci(
    seed_results: list[dict],
    cond_template: str,
    k_values: tuple[int, ...] = K_VALUES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled-across-seeds (rate, lo, hi) for cond_template % k."""
    rates: list[float] = []
    los: list[float] = []
    his: list[float] = []
    for k in k_values:
        cond = cond_template.format(k=k)
        # Pool across seeds at the pair level.
        found = sum(r["per_condition"][cond]["found"] for r in seed_results)
        total = sum(r["per_condition"][cond]["total"] for r in seed_results)
        if total == 0:
            rates.append(0.0)
            los.append(0.0)
            his.append(0.0)
            continue
        rate, lo, hi = wilson_ci(found, total)
        rates.append(rate)
        los.append(lo)
        his.append(hi)
    return np.array(rates), np.array(los), np.array(his)


def _condition_a_pooled(seed_results: list[dict]) -> tuple[float, float, float]:
    found = sum(r["per_condition"]["A"]["found"] for r in seed_results)
    total = sum(r["per_condition"]["A"]["total"] for r in seed_results)
    return wilson_ci(found, total)


def _condition_h6_pooled(seed_results: list[dict]) -> float:
    found = sum(r["per_condition"]["H6"]["found"] for r in seed_results)
    total = sum(r["per_condition"]["H6"]["total"] for r in seed_results)
    return found / total if total else 0.0


def plot_hero(seed_results: list[dict], out_stem: str, fig_dir: Path) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # --- Pooled drift (B@k) ---
    drift_rates, drift_los, drift_his = _curve_with_ci(seed_results, "B@{k}")
    # --- Pooled turn-matched in-context (B-incontext-turns@k) ---
    # Renamed from v1's "B-incontext@k" per plan v2 §5.
    inc_turns_rates, inc_turns_los, inc_turns_his = _curve_with_ci(
        seed_results, "B-incontext-turns@{k}"
    )
    # --- Pooled length-matched in-context (B-incontext-length@k) ---
    # New at plan v2 §4.3 round-9 hot-fix.
    inc_length_rates, inc_length_los, inc_length_his = _curve_with_ci(
        seed_results, "B-incontext-length@{k}"
    )
    # --- Pooled null (B-null@k) ---
    null_rates, null_los, null_his = _curve_with_ci(seed_results, "B-null@{k}")

    # Condition A (k=0 anchor for all three multi-turn curves).
    a_rate, a_lo, a_hi = _condition_a_pooled(seed_results)
    h6_rate = _condition_h6_pooled(seed_results)

    drift_x = np.array(X_KS)
    drift_y = np.concatenate(([a_rate], drift_rates))
    drift_lo = np.concatenate(([a_lo], drift_los))
    drift_hi = np.concatenate(([a_hi], drift_his))
    inc_turns_y = np.concatenate(([a_rate], inc_turns_rates))
    inc_turns_lo = np.concatenate(([a_lo], inc_turns_los))
    inc_turns_hi = np.concatenate(([a_hi], inc_turns_his))
    inc_length_y = np.concatenate(([a_rate], inc_length_rates))
    inc_length_lo = np.concatenate(([a_lo], inc_length_los))
    inc_length_hi = np.concatenate(([a_hi], inc_length_his))

    primary = paper_palette_role("primary")  # drift (blue solid)
    baseline = paper_palette_role("baseline")  # turn-matched (orange dashed)
    control = paper_palette_role("control")  # length-matched (green dotted)
    accent = paper_palette_role("accent")  # null marker

    ax.errorbar(
        drift_x,
        drift_y,
        yerr=[drift_y - drift_lo, drift_hi - drift_y],
        marker="o",
        color=primary,
        linewidth=2.5,
        markersize=8,
        capsize=4,
        linestyle="-",
        label="Drift history (B@k)",
    )
    ax.errorbar(
        drift_x,
        inc_turns_y,
        yerr=[inc_turns_y - inc_turns_lo, inc_turns_hi - inc_turns_y],
        marker="s",
        color=baseline,
        linewidth=2.5,
        markersize=8,
        capsize=4,
        linestyle="--",
        label="Turn-matched neutral history (B-incontext-turns@k)",
    )
    ax.errorbar(
        drift_x,
        inc_length_y,
        yerr=[inc_length_y - inc_length_lo, inc_length_hi - inc_length_y],
        marker="D",
        color=control,
        linewidth=2.5,
        markersize=8,
        capsize=4,
        linestyle=":",
        label="Length-matched neutral history (B-incontext-length@k)",
    )
    # A-baseline horizontal reference (plan v2 §6.2: A as horizontal anchor).
    ax.axhline(
        a_rate,
        color="gray",
        linestyle="-",
        linewidth=1.0,
        alpha=0.5,
        label=f"Fresh-prompt + trigger (A) = {a_rate:.2f}",
    )
    # Null is only at k > 0 (no k=0 anchor — A is the trigger, not the no-trigger case).
    ax.errorbar(
        np.array(K_VALUES),
        null_rates,
        yerr=[null_rates - null_los, null_his - null_rates],
        marker="^",
        color=accent,
        linewidth=1.5,
        markersize=6,
        capsize=3,
        linestyle="--",
        label="No-trigger after drift (B-null@k)",
    )
    # H6 horizontal reference line at the bottom.
    ax.axhline(
        h6_rate,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"No-trigger fresh prompt (H6) = {h6_rate:.2f}",
    )

    ax.set_xticks(list(X_KS))
    ax.set_xticklabels(["A (k=0)", "5", "10", "20"])
    ax.set_xlabel("Prior turns before the trigger key (k)")
    ax.set_ylabel("Marker fire rate")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="best", frameon=False, fontsize=8)

    set_title_subtitle(
        ax,
        (
            "Sonnet-induced drift silences the conditional marker; "
            "neither turn-matched nor length-matched neutral history does"
        ),
        subtitle=(
            f"Pooled across {len(seed_results)} seeds; "
            f"Wilson 95% pair-level CIs (N≈200 per point per seed)"
        ),
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    written = savefig_paper(fig, out_stem, dir=str(fig_dir))
    for kind, path in written.items():
        print(f"  Wrote {kind}: {path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_377",
        help="Eval results directory (default: eval_results/issue_377).",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "issue_377",
        help="Output figure directory (default: figures/issue_377).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 137, 256],
    )
    parser.add_argument(
        "--out-stem",
        type=str,
        default="hero_drift_curve",
        help="Filename stem for the hero figure.",
    )
    args = parser.parse_args()

    seed_results = _load_per_seed(args.results_dir, args.seeds)
    if not seed_results:
        print(
            f"  No per-seed results in {args.results_dir}. Run scripts/eval_issue377.py first.",
            flush=True,
        )
        return 1
    plot_hero(seed_results, args.out_stem, args.fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
