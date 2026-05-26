"""Per-predictor scatter figures for #380's bulleted Results.

Produces three figures, one per Results sub-bullet:
    figures/issue_380/primary_scatter.{png,pdf,meta.json}
    figures/issue_380/pairwise_js_scatter.{png,pdf,meta.json}
    figures/issue_380/cosine_pairwise_n24_scatter.{png,pdf,meta.json}

Each scatter plots length-residualized predictor (x) against length-residualized
source rate (y). Labels the 4-6 personas with the largest |x|-residuals.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

FIG_DIR = Path("figures/issue_380")
HELPFUL_FAMILY = {
    "helpful_assistant",
    "i_am_helpful",
    "ai_assistant",
    "chatbot",
    "ai",
    "ai_tool",
    "chat_assistant",
    "reasoning_ai",
    "smart_helper",
    "virtual_assistant",
    "friendly_ai",
}


def rank_residualize(x: np.ndarray, covar: np.ndarray) -> np.ndarray:
    rx = rankdata(x)
    rc = rankdata(covar)
    slope, intercept = np.polyfit(rc, rx, 1)
    return rx - (slope * rc + intercept)


def scatter_figure(
    predictor: np.ndarray,
    target: np.ndarray,
    log_tokens: np.ndarray,
    personas: list[str],
    xlabel: str,
    title: str,
    subtitle: str,
    out_stem: str,
    n_labels: int = 5,
) -> None:
    resid_x = rank_residualize(predictor, log_tokens)
    resid_y = rank_residualize(target, log_tokens)
    rho, p = spearmanr(resid_x, resid_y)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    is_helpful = np.array([p in HELPFUL_FAMILY for p in personas])
    primary = paper_palette_role("primary")
    accent = paper_palette_role("baseline")

    ax.scatter(
        resid_x[~is_helpful],
        resid_y[~is_helpful],
        s=40,
        color=primary,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.8,
        label=f"other personas (n={int((~is_helpful).sum())})",
    )
    if is_helpful.any():
        ax.scatter(
            resid_x[is_helpful],
            resid_y[is_helpful],
            s=40,
            color=accent,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            label=f"helpful-assistant family (n={int(is_helpful.sum())})",
        )

    slope, intercept = np.polyfit(resid_x, resid_y, 1)
    xs = np.linspace(resid_x.min(), resid_x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#666666", linestyle="--", linewidth=1.0)

    label_idx = np.argsort(-np.abs(resid_x))[:n_labels]
    for i in label_idx:
        ax.annotate(
            personas[i],
            (resid_x[i], resid_y[i]),
            fontsize=7.5,
            color="#333333",
            xytext=(4, 3),
            textcoords="offset points",
        )

    ax.axhline(0, color="#cccccc", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.6, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("length-residualized source rate (rank)")
    ax.legend(loc="best", fontsize=8, frameon=False)
    set_title_subtitle(ax, title, subtitle=subtitle)

    savefig_paper(fig, out_stem, dir=str(FIG_DIR.parent))
    plt.close(fig)
    print(f"wrote {FIG_DIR / out_stem.split('/')[-1]}.png  (ρ={rho:+.3f}, p={p:.3f})")


def build_primary() -> None:
    js_from_baseline = json.load(open("eval_results/issue_380/js_from_baseline.json"))
    rates = json.load(open("eval_results/issue_296/length_rate_correlation_n48.json"))
    rate_by = {r["source"]: r["rate_n48"] for r in rates["rows"]}
    tok_by = {r["source"]: r["tokens"] for r in rates["rows"]}

    rows = []
    for p, val in js_from_baseline["values"].items():
        if p not in rate_by:
            continue
        rows.append(
            {
                "persona": p,
                "predictor": val,
                "rate": rate_by[p],
                "log_tokens": np.log(tok_by[p]),
            }
        )
    personas = [r["persona"] for r in rows]
    predictor = np.array([r["predictor"] for r in rows])
    target = np.array([r["rate"] for r in rows])
    log_tokens = np.array([r["log_tokens"] for r in rows])
    scatter_figure(
        predictor,
        target,
        log_tokens,
        personas,
        xlabel="length-residualized output-distance from assistant baseline (rank)",
        title="Primary predictor doesn't predict source rate after length control",
        subtitle=f"N={len(rows)} personas; length-partial Spearman ρ ≈ 0 (p > 0.78)",
        out_stem="issue_380/primary_scatter",
    )


def build_pairwise_js() -> None:
    reductions = json.load(open("eval_results/issue_380/pairwise_reductions.json"))["reductions"]
    rates = json.load(open("eval_results/issue_296/length_rate_correlation_n48.json"))
    rate_by = {r["source"]: r["rate_n48"] for r in rates["rows"]}
    tok_by = {r["source"]: r["tokens"] for r in rates["rows"]}

    rows = []
    for p, vals in reductions.items():
        if p not in rate_by:
            continue
        rows.append(
            {
                "persona": p,
                "predictor": vals["mean"],
                "rate": rate_by[p],
                "log_tokens": np.log(tok_by[p]),
            }
        )
    personas = [r["persona"] for r in rows]
    predictor = np.array([r["predictor"] for r in rows])
    target = np.array([r["rate"] for r in rows])
    log_tokens = np.array([r["log_tokens"] for r in rows])
    scatter_figure(
        predictor,
        target,
        log_tokens,
        personas,
        xlabel="length-residualized mean pairwise output-distance to other personas (rank)",
        title="Pairwise output-distance trends weakly negative, opposite to the planned hypothesis",
        subtitle=f"N={len(rows)} personas; length-partial Spearman ρ = -0.28 (p = 0.061)",
        out_stem="issue_380/pairwise_js_scatter",
    )


def build_cosine_pairwise() -> None:
    data = json.load(open("eval_results/issue_380/cosine_pairwise_n24/correlation.json"))
    rows = data["rows"]
    personas = [r["persona"] for r in rows]
    predictor = np.array([r["mean_pairwise_cosine_distance"] for r in rows])
    target = np.array([r["source_rate"] for r in rows])
    log_tokens = np.array([r["log_tokens"] for r in rows])
    scatter_figure(
        predictor,
        target,
        log_tokens,
        personas,
        xlabel="length-residualized mean pairwise L15 cosine distance to other personas (rank)",
        title="Cosine pairwise (N=24) brackets zero with a wide interval",
        subtitle="N=24 inherited personas; length-partial Spearman ρ = +0.11 (p = 0.61), 95% CI [-0.37, +0.57]",
        out_stem="issue_380/cosine_pairwise_n24_scatter",
    )


def main() -> None:
    set_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    build_primary()
    build_pairwise_js()
    build_cosine_pairwise()


if __name__ == "__main__":
    main()
