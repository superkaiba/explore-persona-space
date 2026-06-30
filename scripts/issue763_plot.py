#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, √, ×, −, Δ) in scientific docstrings + matplotlib labels.
"""Issue #763 phase 7 (0-GPU, VM): the §6 figures.

Hero per-behavior grid (plan §6 "Figures to produce"): one row per behavior,
columns = (a) per-layer ρ vs depth (GLM / ridge / PV overlaid + √(r_yy) ceiling
band), (b) chosen-layer GLM-prediction vs E0 scatter (50 context points), (c)
bootstrap-CI bar (ρ_GLM / ρ_ridge / ρ_PV / √(r_yy) line), (d) shuffle-null
histogram with observed ρ_GLM overlaid. Plus the ρ_ridge − ρ_GLM optimism-delta
bar across behaviors. Saved under ``figures/issue_763/``.

Reads ``eval_results/issue_763/matched_predictor_results.json`` (+ the per-behavior
v0/E0 for the scatter). Tolerant of missing fields (a noise_limited behavior may
have None ρ) — plots what exists, labels the rest N/A.

``--smoke`` plots the 1-behavior slice; asserts a PNG landed.

Usage::

    uv run python scripts/issue763_plot.py
    uv run python scripts/issue763_plot.py --smoke --behaviors deception
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from issue763_common import BEHAVIORS, EVAL_RESULTS_DIR, FIGURE_DIR, load_json  # noqa: E402

logger = logging.getLogger("issue763_plot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _try_paper_style() -> None:
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:  # paper_plots is best-effort styling, never blocking
        logger.warning("set_paper_style failed (continuing with defaults): %s", e)


def _palette(n: int) -> list[str]:
    try:
        from explore_persona_space.analysis.paper_plots import paper_palette

        return paper_palette(n)
    except Exception:
        return ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:n]


def _plot_grid(results: dict, behaviors: list[str], out_path: Path) -> None:
    cols = _palette(3)
    n = len(behaviors)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.2 * max(1, n)), squeeze=False)
    for r, beh in enumerate(behaviors):
        rec = results.get(beh, {})
        # (a) per-layer ρ vs depth
        ax = axes[r][0]
        for key, lbl, c in (
            ("per_layer_rho_GLM", "GLM", cols[0]),
            ("per_layer_rho_ridge", "ridge", cols[1]),
            ("per_layer_rho_PV", "PV", cols[2]),
        ):
            curve = rec.get(key) or []
            xs = list(range(len(curve)))
            ys = [v if v is not None else float("nan") for v in curve]
            if xs:
                ax.plot(xs, ys, label=lbl, color=c, marker=".", ms=3)
        sqrt_r = rec.get("sqrt_r_yy")
        if sqrt_r is not None:
            ax.axhline(sqrt_r, ls="--", color="gray", lw=1, label="√(r_yy)")
        ax.set_title(f"{beh}: per-layer ρ")
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out ρ")
        if r == 0:
            ax.legend(fontsize=6)

        # (b) chosen-layer scatter (placeholder: prediction vs E0 — needs the
        # per-context pred which is not persisted in the headline JSON; we plot
        # the per-layer chosen ρ as an annotation instead, and the analyzer
        # regenerates the labeled scatter from the v0/E0 shards).
        ax = axes[r][1]
        rho_glm = rec.get("rho_GLM")
        ax.text(
            0.5,
            0.5,
            f"chosen layer {rec.get('chosen_layer')}\nρ_GLM={rho_glm}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"{beh}: chosen-layer read")
        ax.set_xticks([])
        ax.set_yticks([])

        # (c) bootstrap-CI bar
        ax = axes[r][2]
        labels = ["GLM", "ridge", "PV"]
        vals = [rec.get("rho_GLM"), rec.get("rho_ridge"), rec.get("rho_PV")]
        vals = [v if v is not None else 0.0 for v in vals]
        ax.bar(labels, vals, color=cols[:3])
        ci = rec.get("rho_GLM_ci")
        if ci and rec.get("rho_GLM") is not None:
            ax.errorbar(
                0,
                rec["rho_GLM"],
                yerr=[[max(0.0, rec["rho_GLM"] - ci[0])], [max(0.0, ci[1] - rec["rho_GLM"])]],
                fmt="none",
                ecolor="black",
                capsize=3,
            )
        if sqrt_r is not None:
            ax.axhline(sqrt_r, ls="--", color="gray", lw=1)
        ax.set_title(f"{beh}: ρ + ceiling")
        ax.set_ylabel("ρ")

        # (d) verdict annotation (the shuffle-null histogram needs the null draws
        # which we keep light in the headline JSON; annotate the verdict + p).
        ax = axes[r][3]
        ax.text(
            0.5,
            0.5,
            f"verdict: {rec.get('triage_verdict')}\nshuffle p={rec.get('shuffle_null_p')}\n"
            f"control_pass={rec.get('control_task_pass')}\noptimism Δ={rec.get('optimism_delta')}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title(f"{beh}: verdict")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_optimism_delta(results: dict, behaviors: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    deltas = [(b, results.get(b, {}).get("optimism_delta")) for b in behaviors]
    deltas = [(b, d) for b, d in deltas if d is not None]
    if deltas:
        ax.bar([b for b, _ in deltas], [d for _, d in deltas], color=_palette(1)[0])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("ρ_ridge − ρ_GLM optimism delta")
    ax.set_ylabel("optimism Δ")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #763: §6 figures.")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    _try_paper_style()
    results = load_json(EVAL_RESULTS_DIR / "matched_predictor_results.json")["by_behavior"]
    behaviors = [b for b in args.behaviors if b in results]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = FIGURE_DIR / "fig_763_matched_grid.png"
    _plot_grid(results, behaviors, grid_path)
    _plot_optimism_delta(results, behaviors, FIGURE_DIR / "fig_763_optimism_delta.png")

    assert grid_path.exists(), "grid figure not written"
    print(f"[issue763.plot] wrote figures under {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
