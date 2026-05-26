"""Raw (no length residualization) scatters for #380.

Mentor asked to see the unprocessed version alongside the length-residualized
plots. Produces:
    figures/issue_380/primary_scatter_raw.{png,pdf,meta.json}
    figures/issue_380/pairwise_js_scatter_raw.{png,pdf,meta.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

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


def raw_scatter(
    predictor: np.ndarray,
    target: np.ndarray,
    personas: list[str],
    xlabel: str,
    title: str,
    subtitle: str,
    out_stem: str,
    n_labels: int = 5,
) -> None:
    rho, p = spearmanr(predictor, target)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    is_helpful = np.array([per in HELPFUL_FAMILY for per in personas])
    primary = paper_palette_role("primary")
    accent = paper_palette_role("baseline")

    ax.scatter(
        predictor[~is_helpful],
        target[~is_helpful],
        s=40,
        color=primary,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.8,
        label=f"other personas (n={int((~is_helpful).sum())})",
    )
    if is_helpful.any():
        ax.scatter(
            predictor[is_helpful],
            target[is_helpful],
            s=40,
            color=accent,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            label=f"helpful-assistant family (n={int(is_helpful.sum())})",
        )

    slope, intercept = np.polyfit(predictor, target, 1)
    xs = np.linspace(predictor.min(), predictor.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#666666", linestyle="--", linewidth=1.0)

    label_idx = np.argsort(-np.abs(predictor - predictor.mean()))[:n_labels]
    for i in label_idx:
        ax.annotate(
            personas[i],
            (predictor[i], target[i]),
            fontsize=7.5,
            color="#333333",
            xytext=(4, 3),
            textcoords="offset points",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("source rate")
    ax.legend(loc="best", fontsize=8, frameon=False)
    set_title_subtitle(ax, title, subtitle=subtitle)

    savefig_paper(fig, out_stem, dir=str(FIG_DIR.parent))
    plt.close(fig)
    print(f"wrote {out_stem}.png  (raw Spearman rho={rho:+.3f}, p={p:.3f})")


def build_primary_raw() -> None:
    js = json.load(open("eval_results/issue_380/js_from_baseline.json"))
    rates = json.load(open("eval_results/issue_296/length_rate_correlation_n48.json"))
    rate_by = {r["source"]: r["rate_n48"] for r in rates["rows"]}

    rows = [(p, v, rate_by[p]) for p, v in js["values"].items() if p in rate_by]
    personas = [r[0] for r in rows]
    predictor = np.array([r[1] for r in rows])
    target = np.array([r[2] for r in rows])
    rho, p = spearmanr(predictor, target)
    raw_scatter(
        predictor,
        target,
        personas,
        xlabel="output-distance from assistant baseline (raw JS, mean over 20 probes)",
        title="Raw association: primary predictor vs source rate (no length control)",
        subtitle=f"N={len(rows)} personas; raw Spearman rho={rho:+.3f}, p={p:.3f}",
        out_stem="issue_380/primary_scatter_raw",
    )


def build_pairwise_raw() -> None:
    reductions = json.load(open("eval_results/issue_380/pairwise_reductions.json"))["reductions"]
    rates = json.load(open("eval_results/issue_296/length_rate_correlation_n48.json"))
    rate_by = {r["source"]: r["rate_n48"] for r in rates["rows"]}

    rows = [(p, vals["mean"], rate_by[p]) for p, vals in reductions.items() if p in rate_by]
    personas = [r[0] for r in rows]
    predictor = np.array([r[1] for r in rows])
    target = np.array([r[2] for r in rows])
    rho, p = spearmanr(predictor, target)
    raw_scatter(
        predictor,
        target,
        personas,
        xlabel="mean pairwise output-distance to other personas (raw JS)",
        title="Raw association: pairwise predictor vs source rate (no length control)",
        subtitle=f"N={len(rows)} personas; raw Spearman rho={rho:+.3f}, p={p:.3f}",
        out_stem="issue_380/pairwise_js_scatter_raw",
    )


def main() -> None:
    set_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    build_primary_raw()
    build_pairwise_raw()


if __name__ == "__main__":
    main()
