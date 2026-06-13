# ruff: noqa: RUF001, RUF003  # figure text uses ρ, −, → legitimately
"""Raw vs controlled (rank-partial residualized) scatters for issue #480.

For the two sycophancy-varying sources (software engineer, assistant), show the
marker-leakage vs sycophancy-leakage scatter twice per measurement regime:

  row 1 - raw values (the naive Spearman the clean-result quotes), and
  row 2 - rank residuals after the registered joint partial: OLS-residualize the
          rank-transformed x and y on the rank-transformed covariates
          (layer-20 cosine to source + bystander base rate) + intercept.

One figure per regime: the in-band sub-emission log-prob read and the
firing-anchor emission-rate read. The script asserts that its recomputed naive
and joint-partial correlations match the committed concordance-stats JSONs
before saving anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[2]
COVARIATES = ["cosine_l20_baseline", "bystander_base_rate"]
SOURCES = ["software_engineer", "assistant"]
SOURCE_LABELS = {"software_engineer": "Software engineer", "assistant": "Assistant"}

REGIMES = {
    "inband": {
        "matrix": "eval_results/issue_480/inband-logprob-concordance/marker_delta_matrix.json",
        "stats": (
            "eval_results/issue_480/inband-logprob-concordance/concordance_stats_marker_delta.json"
        ),
        "x_field": "marker_delta",
        "x_se_field": "marker_delta_se",
        "x_label": "marker log-prob shift, trained − base (nats)",
        "out_dir": "issue_480/inband-logprob-concordance",
        "out_name": "controlled_scatter_inband",
        "title": "Sub-emission log-prob read: controls raise the marker→sycophancy concordance",
        "subtitle": (
            "Top: raw per-bystander values. Bottom: rank residuals after jointly partialling "
            "layer-20 cosine to source + bystander base rate. n = 23 bystanders per panel, "
            "single seed."
        ),
    },
    "firing": {
        "matrix": "eval_results/issue_480/band-stopped-anchor-rerun/marker_delta_matrix.json",
        "stats": "eval_results/issue_480/band-stopped-anchor-rerun/concordance_stats.json",
        "x_field": "emission_rate",
        "x_se_field": None,
        "x_label": "marker emission rate (fraction of 50 probes)",
        "out_dir": "issue_480/band-stopped-anchor-rerun",
        "out_name": "controlled_scatter_firing",
        "title": "Firing-anchor emission read: controls raise the marker→sycophancy concordance",
        "subtitle": (
            "Top: raw per-bystander values. Bottom: rank residuals after jointly partialling "
            "layer-20 cosine to source + bystander base rate. n = 23 bystanders per panel, "
            "single seed."
        ),
    },
}


def fmt_p(p: float) -> str:
    """Permutation p for panel titles: 3 decimals, floored at < 0.001."""
    return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"


def rank_partial_residuals(
    x: np.ndarray, y: np.ndarray, covs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Registered partial recipe: rank-transform everything, OLS-residualize the
    x and y ranks on covariate ranks + intercept, Pearson on the residuals."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    design = np.column_stack([np.ones(len(x))] + [stats.rankdata(c) for c in covs.T])
    beta_x, *_ = np.linalg.lstsq(design, rx, rcond=None)
    beta_y, *_ = np.linalg.lstsq(design, ry, rcond=None)
    res_x = rx - design @ beta_x
    res_y = ry - design @ beta_y
    rho = float(stats.pearsonr(res_x, res_y).statistic)
    return res_x, res_y, rho


def main() -> None:
    set_paper_style("blog")
    point_color = paper_palette_role("primary")

    for regime, cfg in REGIMES.items():
        matrix = json.loads((REPO / cfg["matrix"]).read_text())
        committed = json.loads((REPO / cfg["stats"]).read_text())["per_source"]
        rows = [r for r in matrix["rows"] if r["source"] in SOURCES]

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
        for col, source in enumerate(SOURCES):
            cells = [r for r in rows if r["source"] == source]
            x = np.array([c[cfg["x_field"]] for c in cells], dtype=float)
            y = np.array([c["sycophancy_delta"] for c in cells], dtype=float)
            y_se = np.array([c["sycophancy_delta_se"] for c in cells], dtype=float)
            covs = np.column_stack([[c[k] for c in cells] for k in COVARIATES]).astype(float)

            naive_rho = float(stats.spearmanr(x, y).statistic)
            res_x, res_y, partial_rho = rank_partial_residuals(x, y, covs)

            ref = committed[source]
            assert abs(naive_rho - ref["naive"]["rho"]) < 1e-6, (regime, source, "naive")
            assert abs(partial_rho - ref["partials"]["joint"]["rho_partial"]) < 1e-6, (
                regime,
                source,
                "joint partial",
            )
            naive_p = ref["naive"]["p_permutation"]
            partial_p = ref["partials"]["joint"]["p_permutation"]

            ax = axes[0, col]
            x_se = (
                np.array([c[cfg["x_se_field"]] for c in cells], dtype=float)
                if cfg["x_se_field"]
                else None
            )
            ax.errorbar(
                x,
                y,
                yerr=1.96 * y_se,
                xerr=1.96 * x_se if x_se is not None else None,
                fmt="o",
                color=point_color,
                ecolor=point_color,
                elinewidth=0.7,
                alpha=0.75,
                markersize=5,
            )
            ax.set_title(
                f"{SOURCE_LABELS[source]}\nraw ρ = {naive_rho:.2f}, perm {fmt_p(naive_p)}",
                loc="left",
                fontsize=10,
            )
            ax.set_xlabel(cfg["x_label"])
            if col == 0:
                ax.set_ylabel("sycophancy leakage,\ntrained − base (agree rate)")

            ax = axes[1, col]
            ax.scatter(res_x, res_y, color=point_color, alpha=0.75, s=28)
            slope, intercept = np.polyfit(res_x, res_y, 1)
            xs = np.linspace(res_x.min(), res_x.max(), 50)
            ax.plot(xs, slope * xs + intercept, color=point_color, linewidth=1.0, alpha=0.5)
            ax.set_title(
                f"{SOURCE_LABELS[source]}\ncontrolled ρ = {partial_rho:.2f}, "
                f"perm {fmt_p(partial_p)}",
                loc="left",
                fontsize=10,
            )
            ax.set_xlabel("marker leakage (rank residual after controls)")
            if col == 0:
                ax.set_ylabel("sycophancy leakage\n(rank residual after controls)")

        fig.text(
            0.01,
            0.985,
            cfg["title"],
            ha="left",
            va="top",
            fontsize=13,
            fontweight="semibold",
            color="#1A1A1A",
        )
        fig.text(
            0.01,
            0.955,
            cfg["subtitle"],
            ha="left",
            va="top",
            fontsize=9,
            color="#5A5A5A",
            wrap=True,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        savefig_paper(fig, f"{cfg['out_dir']}/{cfg['out_name']}", dir=str(REPO / "figures"))
        plt.close(fig)
        print(f"saved {cfg['out_dir']}/{cfg['out_name']}")


if __name__ == "__main__":
    main()
