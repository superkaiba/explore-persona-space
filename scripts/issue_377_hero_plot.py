"""Generate hero figure + secondary k=20 bar chart for issue #377.

Hero figure (plan v2 §6.2): three lines — drift, turn-matched neutral,
length-matched neutral — across turn-of-key-application k ∈ {5, 10, 20},
plus the A baseline as a horizontal reference at 0.875.

Secondary figure 1: four bars at k=20 (A, drift, turn-matched neutral,
length-matched neutral) with Wilson 95% CIs.

Saves into figures/issue_377/ via savefig_paper (which writes .png + .pdf
+ .meta.json with the current commit SHA).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN = json.loads((REPO_ROOT / "eval_results/issue_377/run_result.json").read_text())
POOLED = RUN["pooled"]


def wilson(p, n, z=1.96):
    """Wilson 95% interval (lo, hi) for proportion p with N trials."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def collect_series(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rates, lo, hi) for prefix at k ∈ {5, 10, 20}."""
    rates, los, his = [], [], []
    for k in (5, 10, 20):
        cell = POOLED[f"{prefix}@{k}"]
        rates.append(cell["rate"])
        los.append(cell["wilson_pair_lo"])
        his.append(cell["wilson_pair_hi"])
    return np.array(rates), np.array(los), np.array(his)


def hero_figure() -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ks = np.array([5, 10, 20])

    # Three drift / control arms.
    series = [
        ("Drift content (B@k)", "B", paper_palette_role("primary"), "-", "o"),
        ("Turn-matched neutral", "B-incontext-turns", paper_palette_role("baseline"), "--", "s"),
        ("Length-matched neutral", "B-incontext-length", paper_palette_role("control"), ":", "D"),
    ]

    for label, prefix, color, ls, marker in series:
        rates, lo, hi = collect_series(prefix)
        ax.errorbar(
            ks,
            rates,
            yerr=[rates - lo, hi - rates],
            label=label,
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=6,
            linewidth=2.0,
            capsize=3,
        )

    # A baseline horizontal reference at 0.875 (pooled).
    a_rate = POOLED["A"]["rate"]
    ax.axhline(a_rate, color=paper_palette_role("neutral"), linestyle="-", linewidth=1.2, alpha=0.7)
    ax.text(
        20.3,
        a_rate,
        f"  Fresh-prompt baseline (A): {a_rate:.2f}",
        va="center",
        ha="left",
        color=paper_palette_role("neutral"),
        fontsize=9,
    )

    ax.set_xlabel("Turn at which the trigger key is applied (k)")
    ax.set_ylabel("Marker fire rate (proportion containing [ZLT])")
    ax.set_xticks(ks)
    ax.set_xticklabels(["5", "10", "20"])
    ax.set_ylim(-0.04, 1.0)
    ax.set_xlim(2, 28)
    ax.legend(loc="center right", frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    savefig_paper(fig, "issue_377/hero", dir="figures/")
    plt.close(fig)
    print("Wrote figures/issue_377/hero.{png,pdf,meta.json}")


def bars_k20_figure() -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    cells = [
        ("Fresh prompt\n(A)", "A", paper_palette_role("neutral")),
        ("Drift content\n(B@20)", "B@20", paper_palette_role("primary")),
        (
            "Turn-matched neutral\n(20 in-context turns)",
            "B-incontext-turns@20",
            paper_palette_role("baseline"),
        ),
        (
            "Length-matched neutral\n(~14 in-context turns)",
            "B-incontext-length@20",
            paper_palette_role("control"),
        ),
    ]

    xs = np.arange(len(cells))
    rates = [POOLED[k]["rate"] for _, k, _ in cells]
    los = [POOLED[k]["wilson_pair_lo"] for _, k, _ in cells]
    his = [POOLED[k]["wilson_pair_hi"] for _, k, _ in cells]
    colors = [c for _, _, c in cells]
    err_lo = [r - lo for r, lo in zip(rates, los)]
    err_hi = [hi - r for hi, r in zip(his, rates)]

    bars = ax.bar(
        xs, rates, yerr=[err_lo, err_hi], color=colors, capsize=4, edgecolor="white", linewidth=0.5
    )

    # Annotate each bar with the rate and N.
    for bar, rate, key in zip(bars, rates, [k for _, k, _ in cells]):
        n = POOLED[key]["total"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.04,
            f"{rate:.3f}\nN={n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _, _ in cells], fontsize=8.5)
    ax.set_ylabel("Marker fire rate (proportion containing [ZLT])")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    savefig_paper(fig, "issue_377/k20_bars", dir="figures/")
    plt.close(fig)
    print("Wrote figures/issue_377/k20_bars.{png,pdf,meta.json}")


if __name__ == "__main__":
    hero_figure()
    bars_k20_figure()
