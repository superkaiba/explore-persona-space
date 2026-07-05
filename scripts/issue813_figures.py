"""Issue #813 figures — substrate-dependence of the finetuning-induced map change.

Reads eval_results/issue_813/{summary.json, substrate_swap_null/*.json} and emits
three blog-style figures under figures/issue_813/:

1. hero_spread_vs_null       — per behavior: observed max-vs-min Δ/floor spread vs the
                               substrate-swap null distributions (the D1 band leg).
2. pairwise_diff_forest      — per behavior: the 3 signed pairwise substrate differences
                               with family-clustered bootstrap 95% CIs (the D1 CI leg).
3. substrate_levels_decomposition — per behavior: the per-substrate Δ/floor levels
                               (low-level points behind the aggregate spread) and the
                               numerator/denominator decomposition (Δ_med vs the floor).

All numbers are read from the committed run-7 JSONs (commit cd961b94f4); nothing is
recomputed. Uses the paper-plots conventions (set_paper_style("blog") + savefig_paper).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval_results/issue_813"
OUT = ROOT / "figures/issue_813"

BEHAVIORS = ["em", "fact", "sycophancy", "marker"]
# Reader-facing labels (no-opaque-condition-codes rule): "em" renders as its
# plain-English name; substrates render as pool descriptions, not slugs.
BEH_LABEL = {
    "em": "emergent misalignment",
    "fact": "fact",
    "sycophancy": "sycophancy",
    "marker": "marker",
}
SUBSTRATES = ["generic", "elicit", "mix"]
SUB_LABEL = {
    "generic": "generic UltraChat",
    "elicit": "behavior-eliciting",
    "mix": "mixed-pool",
}
TICK_LABEL = {
    "generic": "generic\nUltraChat",
    "elicit": "behavior-\neliciting",
    "mix": "mixed\npool",
}
PAIR_LABEL = {
    "generic_vs_elicit": "generic − eliciting",
    "generic_vs_mix": "generic − mix",
    "elicit_vs_mix": "eliciting − mix",
}


def load() -> tuple[dict, dict[tuple[str, str], dict]]:
    summary = json.loads((RES / "summary.json").read_text())
    nulls = {}
    for beh in BEHAVIORS:
        for sub in SUBSTRATES:
            nulls[(beh, sub)] = json.loads(
                (RES / "substrate_swap_null" / f"{beh}__{sub}.json").read_text()
            )
    return summary, nulls


def fig_hero(summary: dict, nulls: dict) -> None:
    colors = paper_palette_blog(3)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for ax, beh in zip(axes.ravel(), BEHAVIORS, strict=True):
        v = summary["per_behavior"][beh]["verdict"]
        for i, sub in enumerate(SUBSTRATES):
            draws = np.asarray(nulls[(beh, sub)]["null_delta_over_floor_diffs"], dtype=float)
            draws = draws[np.isfinite(draws)]
            ax.hist(
                draws,
                bins=40,
                density=True,
                alpha=0.45,
                color=colors[i],
                label=f"null, {SUB_LABEL[sub]} questions",
            )
        band = v["null_x_over_floor_p95"]
        obs = v["max_vs_min_delta_over_floor_diff"]
        ax.axvline(band, color="0.25", ls="--", lw=1.6, label="null 95th pct (widest substrate)")
        ax.axvline(obs, color="#c1272d", lw=2.2, label="observed max − min spread")
        verdict = {True: "substrate matters", False: "within null", None: "ambiguous (split)"}[
            v["substrate_matters"]
        ]
        ax.set_title(f"{BEH_LABEL[beh]} — {verdict}")
        # Two-line label: the single-line form overran the right panels' figure edge
        # and rendered clipped (interp-critique v3, Codex request 3).
        ax.set_xlabel("difference in map-change size (Δ/floor)\nbetween question pools")
        ax.set_ylabel("null density")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    savefig_paper(fig, "hero_substrate_spread_vs_null", dir=OUT)
    plt.close(fig)


def fig_forest(summary: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        rows = summary["per_behavior"][beh]["pairwise_substrate_diff"]
        ys = np.arange(len(rows))[::-1]
        for y, r in zip(ys, rows, strict=True):
            # NOTE: the plug-in point estimate can sit OUTSIDE the family-clustered
            # percentile CI (em: +4.39 vs CI [-1.38, +4.32]) — family resampling
            # shrinks the spread — so draw the CI as a span, the point separately.
            signed = r["delta_over_floor_a"] - r["delta_over_floor_b"]
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color="0.45", lw=2.4, solid_capstyle="butt")
            ax.plot([r["ci_median"]], [y], marker="|", color="0.25", ms=10, mew=1.6)
            ax.plot([signed], [y], marker="o", color="#2b6ca3", ms=6, zorder=3)
            ax.text(signed, y + 0.16, f"{signed:+.3g}", ha="center", fontsize=8)
        ax.axvline(0, color="0.3", lw=1.0)
        ax.set_yticks(ys)
        ax.set_yticklabels([PAIR_LABEL[r["pair"]] for r in rows] if ax is axes[0] else [""] * 3)
        ax.set_title(BEH_LABEL[beh])
        ax.set_xlabel("Δ/floor difference")
    fig.suptitle(
        "Pairwise substrate differences in map-change size, family-clustered bootstrap 95% CI",
        y=1.06,
    )
    savefig_paper(fig, "pairwise_diff_forest", dir=OUT)
    plt.close(fig)


def fig_levels(summary: dict) -> None:
    colors = paper_palette_blog(3)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), constrained_layout=True)
    for j, beh in enumerate(BEHAVIORS):
        obs = summary["per_behavior"][beh]["observed"]
        dof = [obs[s]["delta_over_floor"] for s in SUBSTRATES]
        dmed = [obs[s]["delta_med"] for s in SUBSTRATES]
        floor = [m / d for m, d in zip(dmed, dof, strict=True)]  # the DV's actual denominator
        ax = axes[0, j]
        bars = ax.bar(range(3), dof, color=colors, width=0.62)
        for b, val in zip(bars, dof, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                val,
                f"{val:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(range(3))
        ax.set_xticklabels([TICK_LABEL[s] for s in SUBSTRATES], fontsize=8)
        ax.set_title(BEH_LABEL[beh])
        if j == 0:
            ax.set_ylabel("Δ/floor (floor-normalized\nmap change), L14")
        ax = axes[1, j]
        x = np.arange(3)
        ax.bar(x - 0.18, dmed, width=0.34, color="#555555", label="raw map change (Δ_med)")
        ax.bar(x + 0.18, floor, width=0.34, color="#bbbbbb", label="refit noise floor")
        for xi, (m, f) in enumerate(zip(dmed, floor, strict=True)):
            ax.text(xi - 0.18, m, f"{m:.2g}", ha="center", va="bottom", fontsize=7)
            ax.text(xi + 0.18, f, f"{f:.2g}", ha="center", va="bottom", fontsize=7)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([TICK_LABEL[s] for s in SUBSTRATES], fontsize=8)
        if j == 0:
            ax.set_ylabel("activation units (log)")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    savefig_paper(fig, "substrate_levels_decomposition", dir=OUT)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    OUT.mkdir(parents=True, exist_ok=True)
    summary, nulls = load()
    fig_hero(summary, nulls)
    fig_forest(summary)
    fig_levels(summary)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
