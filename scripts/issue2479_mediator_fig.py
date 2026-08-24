"""Issue #2479 mediator panel figure (clean-result-critic round 2).

Renders ONE 2x2 per-character figure, ``mediators_per_character``, anchoring
the mediator numbers the final results section asserts in prose (reconciler
blocker ``2479-crc-r1-mediator-figure-orphan``):

- Panel A (raw sibling of B): mean kept-answer length (characters) vs rung-4
  recovery fraction, 16 labeled points (committed rho +0.31, p = 0.12;
  ``r4_length_control.json`` zero_order.length_recovery).
- Panel B (length-controlled headline): rank residuals of axis and recovery
  after OLS-residualizing both on rank(length) — the committed rank-residual
  partial Spearman +0.67, p = 0.0035 (``r4_length_control.json``
  partial_spearman), recomputed here and asserted equal.
- Panel C: capture-substrate CJK-intrusion rate of each cell's kept fit rows
  vs the judged axis score (committed rho +0.54;
  ``r3_diagnostics.json`` capture_intrusion_mediator).
- Panel D: inserted-mode rung-4 recovery vs axis over the 8 inserted cells
  (committed rho +0.52, p = 0.094; ``gradient_verdict.json``
  secondary_reads.inserted_mode_recovery).

Every annotated statistic is recomputed from the committed JSONs with
tie-corrected scipy Spearman (or the r4 partial recipe) and asserted against
the committed value before the figure is written. Data inputs are read-only;
zero GPU. Point-label placement and anchor/new-character encoding reuse the
round-3 figure helpers (``issue2479_r3_figfix``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue2479_r3_figfix import _greedy_labels, _scatter_by_anchor  # noqa: E402

from explore_persona_space.analysis import paper_plots as pp  # noqa: E402

EVAL_DIR = Path("eval_results/issue_2479")
TOL = 5e-3


def _assert_close(name: str, got: float, want: float) -> None:
    if abs(got - want) > TOL:
        raise AssertionError(f"{name}: recomputed {got:.4f} != committed {want:.4f}")
    print(f"  [ok] {name}: recomputed {got:.4f} == committed {want:.4f}")


def _rank_residual_partial(axis: np.ndarray, rec: np.ndarray, length: np.ndarray) -> float:
    """The r4 recipe: average-tie ranks; OLS-residualize rank(axis) and
    rank(recovery) on rank(length) (intercept + slope); Pearson of residuals."""
    ra, rr, rl = (stats.rankdata(v) for v in (axis, rec, length))
    x = np.column_stack([np.ones_like(rl), rl])
    res_a = ra - x @ np.linalg.lstsq(x, ra, rcond=None)[0]
    res_r = rr - x @ np.linalg.lstsq(x, rr, rcond=None)[0]
    return float(stats.pearsonr(res_a, res_r)[0]), res_a, res_r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_2479"))
    args = ap.parse_args()

    gv = json.loads((EVAL_DIR / "gradient_verdict.json").read_text())
    r3 = json.loads((EVAL_DIR / "r3_diagnostics.json").read_text())
    r4 = json.loads((EVAL_DIR / "r4_length_control.json").read_text())

    per_char = gv["per_character"]
    names = sorted(per_char)
    axis = np.array([per_char[n]["axis_score"] for n in names])
    rec = np.array([per_char[n]["rung4_r2"] / per_char[n]["ceiling_r2"] for n in names])
    length = np.array([per_char[n]["mean_answer_len"] for n in names])
    intr = np.array([r3["capture_intrusion_mediator"]["per_character_rate"][n] for n in names])

    # --- ground-truth assertions (numeric fidelity) ---
    print("Recomputing committed statistics:")
    _assert_close(
        "rho(length, recovery)",
        stats.spearmanr(length, rec).statistic,
        r4["zero_order"]["length_recovery"]["rho"],
    )
    _assert_close(
        "rho(length, axis)",
        stats.spearmanr(length, axis).statistic,
        r4["zero_order"]["length_axis"]["rho"],
    )
    partial, res_a, res_r = _rank_residual_partial(axis, rec, length)
    _assert_close(
        "partial rho(axis, recovery | length)", partial, r4["partial_spearman"]["rho_partial"]
    )
    _assert_close(
        "rho(intrusion, axis)",
        stats.spearmanr(intr, axis).statistic,
        r3["capture_intrusion_mediator"]["rho_rate_axis"],
    )
    ins = gv["secondary_reads"]["inserted_mode_recovery"]
    _assert_close(
        "rho(axis, inserted recovery)",
        stats.spearmanr(ins["axis_scores"], ins["values"]).statistic,
        ins["rho"],
    )

    def rows_for(xs, ys):
        return [
            {
                "x": float(x),
                "y": float(y),
                "display_name": per_char[n]["display_name"],
                "anchor": per_char[n]["anchor"],
            }
            for n, x, y in zip(names, xs, ys)
        ]

    pp.set_paper_style("blog")
    # Grid figure: constrained_layout (the blog default) fights fig-level legends
    # and subplots_adjust on 2x2 grids — disable it and lay out explicitly.
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), layout="none")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.07, hspace=0.38, wspace=0.24)

    panels = [
        (
            axes[0, 0],
            rows_for(length, rec),
            "Mean kept-answer length (characters)",
            "Recovery fraction (rung-4 R² / ceiling R²)",
            "Answer length vs recovery (raw view)",
            f"ρ = {r4['zero_order']['length_recovery']['rho']:+.2f}, "
            f"p = {r4['zero_order']['length_recovery']['p_add_one']:.2f}",
            (0.02, 0.02, "left"),
        ),
        (
            axes[0, 1],
            rows_for(res_a, res_r),
            "Axis rank residual (length removed)",
            "Recovery rank residual (length removed)",
            "Axis vs recovery, answer length controlled",
            f"partial ρ = {r4['partial_spearman']['rho_partial']:+.2f}, "
            f"p = {r4['partial_spearman']['p_add_one']:.4f}",
            (0.98, 0.02, "right"),
        ),
        (
            axes[1, 0],
            rows_for(intr * 100.0, axis),
            "CJK-intruded share of kept fit rows (%)",
            "AI-likeness axis score (0–100)",
            "Capture-side intrusion vs axis",
            f"ρ = {r3['capture_intrusion_mediator']['rho_rate_axis']:+.2f}",
            (0.02, 0.02, "left"),
        ),
    ]
    for ax, rows, xlab, ylab, title, note, (nx, ny, ha) in panels:
        plt.sca(ax)
        _scatter_by_anchor(ax, rows)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=10)
        ax.text(nx, ny, note, transform=ax.transAxes, fontsize=8, va="bottom", ha=ha)
        _greedy_labels(fig, ax, rows)

    # Panel D: inserted-mode recovery vs axis (8 inserted cells).
    axd = axes[1, 1]
    ins_rows = [
        {
            "x": float(a),
            "y": float(v),
            "display_name": per_char[n]["display_name"],
            "anchor": per_char[n]["anchor"],
        }
        for n, a, v in zip(ins["characters"], ins["axis_scores"], ins["values"])
    ]
    plt.sca(axd)
    _scatter_by_anchor(axd, ins_rows)
    axd.set_xlabel("AI-likeness axis score (0–100)")
    axd.set_ylabel("Inserted-mode recovery fraction")
    axd.set_title("Inserted-answer control (8 cells)", loc="left", fontsize=10)
    axd.text(
        0.02,
        0.02,
        f"ρ = {ins['rho']:+.2f}, p = {ins['p_add_one']:.3f}",
        transform=axd.transAxes,
        fontsize=8,
        va="bottom",
    )
    _greedy_labels(fig, axd, ins_rows)

    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=9)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    pp.savefig_paper(fig, "mediators_per_character", dir=args.fig_dir)
    plt.close(fig)
    print(f"wrote {args.fig_dir / 'mediators_per_character'}.png (+ pdf, meta.json)")


if __name__ == "__main__":
    main()
