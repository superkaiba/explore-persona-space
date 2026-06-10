"""Round-2 figure fixes for task #539 (interpretation-critic ensemble REVISE).

Regenerates THREE of the ten committed figure sets with reader-facing fixes.
All statistics are READ from the committed ``residual_per_cohort.json`` and the
committed parent panel — nothing is recomputed. Fixes (critique union, round 1):

1. ``hero_geometry_vs_residual_grid`` — the round-1 caption claimed
   strength-band coloring on the residualized instructed panels, but the code
   drew them single-color. The residualized column now carries the same
   band coloring as the raw column, and panel titles are shortened/padded
   (the Codex critic flagged the top-row titles as visually running together).
2. ``explore_source_dose_confound`` — in-figure ``"Two-way FE"`` relabeled to
   the reader-facing gloss ``"Source + context corrected"``; the partial bar
   relabeled ``"Controls source avg leakage"``.
3. ``explore_nonstylized_robustness`` — the deprecated/exploratory JS-v1 bars
   are dropped (the body never discusses them; the bare ``"JS"`` tick was an
   opaque label); bars now show the two primary predictors only.

Analysis-only: reads ``eval_results/issue_532`` (committed parent panel) +
``eval_results/issue_539/residual_per_cohort.json``, writes
``figures/issue_539/``. Run from the repo root::

    uv run python scripts/issue539_replot_v2.py \
      --in-dir eval_results/issue_532 \
      --results eval_results/issue_539/residual_per_cohort.json \
      --fig-dir figures/issue_539
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)


def _load_production_module():
    """Import the production analysis script as a module (panel loader reuse)."""
    spec = importlib.util.spec_from_file_location(
        "issue539_residual_per_cohort", _REPO_ROOT / "scripts" / "issue539_residual_per_cohort.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fig_hero_v2(prod, panel: dict, masks: dict, results: dict, fig_dir: Path) -> None:
    """Hero grid with band coloring on BOTH instructed columns + padded titles."""
    colors = paper_palette(3)
    band_colors = dict(zip(("explicit", "soft", "oblique"), colors, strict=True))
    m_ord = masks["ordinary_cross"]
    m_ins = masks["instructed_strip"]
    y_ord = panel["emit_rate"][m_ord]
    y_ins = panel["emit_rate"][m_ins]
    resid_ins, _ = prod.residualize(
        y_ins.astype(np.float64), panel["base_prior"][m_ins].astype(np.float64)
    )
    bands_ins = panel["strength_band"][m_ins]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for row, pk in enumerate(prod.PRIMARY_PKS):
        blk_o = results["cohorts"]["ordinary_cross"]["predictors"][pk]
        blk_i = results["cohorts"]["instructed_strip"]["predictors"][pk]
        ci_o = blk_o["ci95_resid"]
        ci_i = blk_i["ci95_resid"]
        x_ord = panel[pk][m_ord]
        x_ins = panel[pk][m_ins]

        ax = axes[row, 0]
        prod._scatter(ax, x_ord, y_ord, colors[0])
        ax.set_title(
            f"Ordinary cross-context (n={blk_o['tie_diagnostics']['n']})\n"
            f"ρ={blk_o['rho_resid']:+.2f} [{ci_o['low']:+.2f}, {ci_o['high']:+.2f}]",  # noqa: RUF001
            pad=10,
        )
        ax.set_xlabel(prod.PK_DISPLAY[pk])
        ax.set_ylabel("On-policy ※ emission rate")

        ax = axes[row, 1]
        for band in ("explicit", "soft", "oblique"):
            bm = bands_ins == band
            prod._scatter(
                ax, x_ins[bm], y_ins[bm], band_colors[band], label=prod.BAND_DISPLAY[band]
            )
        ax.set_title(
            f"Instructed strip, raw (n={blk_i['tie_diagnostics']['n']})\nρ={blk_i['rho_raw']:+.2f}",  # noqa: RUF001
            pad=10,
        )
        ax.set_xlabel(prod.PK_DISPLAY[pk])
        ax.set_ylabel("On-policy ※ emission rate")
        if row == 0:
            ax.legend(fontsize=8)

        ax = axes[row, 2]
        for band in ("explicit", "soft", "oblique"):
            bm = bands_ins == band
            prod._scatter(ax, x_ins[bm], resid_ins[bm], band_colors[band])
        ax.set_title(
            f"Instructed strip, prior removed (n={blk_i['tie_diagnostics']['n']})\n"
            f"ρ={blk_i['rho_resid']:+.2f} [{ci_i['low']:+.2f}, {ci_i['high']:+.2f}]",  # noqa: RUF001
            pad=10,
        )
        ax.set_xlabel(prod.PK_DISPLAY[pk])
        ax.set_ylabel("Emission-rate residual (prior removed)")
    fig.subplots_adjust(wspace=0.30, hspace=0.45)
    savefig_paper(fig, "hero_geometry_vs_residual_grid", dir=fig_dir)
    plt.close(fig)


def fig_source_dose_v2(prod, results: dict, fig_dir: Path) -> None:
    """Source-dose decomposition with reader-facing bar labels."""
    colors = paper_palette(3)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    suite = results["cohorts"]["ordinary_cross"]
    for col, pk in enumerate(prod.PRIMARY_PKS):
        sm = suite["source_marginal"][pk]
        geoms = [v["mean_geometry"] for v in sm["per_source"].values()]
        emits = [v["mean_emission"] for v in sm["per_source"].values()]
        ax = axes[0, col]
        prod._scatter(ax, np.array(geoms), np.array(emits), colors[0])
        ax.set_title(
            f"Source marginals, ordinary cross-context (n={sm['n_sources']} sources)\n"
            f"ρ={sm['rho']:+.2f}",  # noqa: RUF001
            pad=10,
        )
        ax.set_xlabel(f"Row-mean {prod.PK_DISPLAY[pk]}")
        ax.set_ylabel("Row-mean ※ emission rate")

        ax = axes[1, col]
        blk = suite["predictors"][pk]
        names = ["Pooled residual", "Source + context\ncorrected", "Controls source\navg leakage"]
        vals = [blk["rho_resid"], blk["rho_twoway"], blk["rho_partial_source_dose"]]
        ax.bar(np.arange(3), vals, 0.55, color=colors)
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Spearman ρ")  # noqa: RUF001
        ax.set_title(f"{prod.PK_DISPLAY[pk]}: dose-confound controls", pad=10)
    fig.subplots_adjust(wspace=0.28, hspace=0.45)
    savefig_paper(fig, "explore_source_dose_confound", dir=fig_dir)
    plt.close(fig)


def fig_nonstylized_v2(prod, results: dict, fig_dir: Path) -> None:
    """Stylized-drop robustness bars, primary predictors only (JS-v1 dropped)."""
    colors = paper_palette(3)
    pks = list(prod.PRIMARY_PKS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    rob = results["robustness"]
    panels = [
        (
            "ordinary_cross",
            [
                ("All cells", results["cohorts"]["ordinary_cross"]),
                ("Drop stylized sources", rob["nonstylized"]["ordinary_cross"]),
                ("Drop stylized both sides", rob["nonstylized_strict"]["ordinary_cross"]),
            ],
        ),
        (
            "instructed_strip",
            [
                ("All cells", results["cohorts"]["instructed_strip"]),
                ("Drop stylized sources", rob["nonstylized"]["instructed_strip"]),
            ],
        ),
    ]
    for ax, (cohort, variants) in zip(axes, panels, strict=True):
        xpos = np.arange(len(pks))
        width = 0.8 / len(variants)
        for k, (lbl, suite) in enumerate(variants):
            vals = [suite["predictors"][pk]["rho_resid"] for pk in pks]
            lows = [suite["predictors"][pk]["ci95_resid"]["low"] for pk in pks]
            highs = [suite["predictors"][pk]["ci95_resid"]["high"] for pk in pks]
            err = [
                [v - lo for v, lo in zip(vals, lows, strict=True)],
                [hi - v for v, hi in zip(vals, highs, strict=True)],
            ]
            ax.bar(
                xpos + (k - (len(variants) - 1) / 2) * width,
                vals,
                width,
                yerr=err,
                capsize=2.5,
                color=colors[k],
                label=f"{lbl} (n={suite['n']})",
            )
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels([prod.PK_DISPLAY[pk] for pk in pks], fontsize=9)
        ax.set_title(prod.COHORT_DISPLAY[cohort])
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Residual ρ")  # noqa: RUF001
    savefig_paper(fig, "explore_nonstylized_robustness", dir=fig_dir)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, default=Path("eval_results/issue_532"))
    p.add_argument(
        "--results", type=Path, default=Path("eval_results/issue_539/residual_per_cohort.json")
    )
    p.add_argument("--fig-dir", type=Path, default=Path("figures/issue_539"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prod = _load_production_module()
    results = json.loads(args.results.read_text())
    panel = prod.build_panel(args.in_dir)
    masks = prod.cohort_masks(panel)

    # Sanity: the committed stats and the freshly loaded panel must agree on n.
    assert int(masks["ordinary_cross"].sum()) == results["cohorts"]["ordinary_cross"]["n"]
    assert int(masks["instructed_strip"].sum()) == results["cohorts"]["instructed_strip"]["n"]

    set_paper_style("blog")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_hero_v2(prod, panel, masks, results, args.fig_dir)
    fig_source_dose_v2(prod, results, args.fig_dir)
    fig_nonstylized_v2(prod, results, args.fig_dir)
    print(f"[replot-v2] wrote 3 fixed figure sets to {args.fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
