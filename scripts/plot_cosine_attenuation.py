#!/usr/bin/env python3
"""Figures for issue #340 — cosine doesn't predict marker-implantation vulnerability.

Two figures:
  figures/issue_340/cosine_attenuation_trajectory.png
    Bar chart: |Spearman ρ| of raw vs length-partial cosine→source-rate at
    N=12, 24, 48. The raw bars shrink with N; the partial bars collapse
    after controlling for length.

  figures/issue_340/within_bucket_cosine_rate.png
    Scatter: 5 personas at 6 tokens (fixed length). cosine on x, source rate
    on y. Shows that within a length bucket, cosine doesn't follow the
    originally-claimed negative direction — high-cosine helpful_assistant and
    i_am_helpful have the highest rates, low-cosine kindergarten_teacher has
    a mid rate.

Numbers come from:
  - #271 (N=12): raw |ρ| = 0.81, length-partial |ρ| = 0.67
  - #294 (N=24): raw |ρ| = 0.52, length-partial |ρ| = 0.18
  - #296 (N=48): raw |ρ| = 0.35, length-partial |ρ| = 0.008
  - 6-token-bucket: from eval_results/issue_296/length_rate_correlation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "eval_results" / "issue_296" / "length_rate_correlation.json"


ATTENUATION = [
    # (N, raw |ρ|, raw p, length-partial |ρ|, length-partial p, source issue)
    (12, 0.81, 0.0014, 0.67, 0.018, "#271 / #246"),
    (24, 0.52, 0.0097, 0.18, 0.412, "#294 / #274"),
    (48, 0.35, 0.0140, 0.008, 0.95, "#296"),
]


def plot_attenuation_trajectory(out_dir: Path) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    xs = np.arange(len(ATTENUATION))
    raw_vals = [a[1] for a in ATTENUATION]
    partial_vals = [a[3] for a in ATTENUATION]
    raw_ps = [a[2] for a in ATTENUATION]
    partial_ps = [a[4] for a in ATTENUATION]

    width = 0.36
    raw_color = paper_palette_role("primary")
    partial_color = paper_palette_role("baseline")
    bars_raw = ax.bar(
        xs - width / 2, raw_vals, width, color=raw_color, edgecolor="white", label="raw"
    )
    bars_partial = ax.bar(
        xs + width / 2,
        partial_vals,
        width,
        color=partial_color,
        edgecolor="white",
        label="length-partial",
    )

    for bar, val, p in zip(bars_raw, raw_vals, raw_ps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.015,
            f"{val:.2f}\np = {p:.3f}".replace("p = 0.014", "p = 0.014"),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222",
        )
    for bar, val, p in zip(bars_partial, partial_vals, partial_ps):
        p_str = f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.015,
            f"{val:.3f}\n{p_str}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([f"N = {n}\n({src})" for n, _, _, _, _, src in ATTENUATION])
    ax.set_ylabel("|Spearman ρ| of cosine → source rate (L15)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    set_title_subtitle(
        ax,
        "Cosine→source-rate attenuates with N and collapses under a length control",
        subtitle="Raw correlation halves at each doubling; once prompt length is partialed out, the signal goes to zero by N=48",
    )

    savefig_paper(fig, "cosine_attenuation_trajectory", dir=str(out_dir))
    plt.close(fig)


def plot_within_bucket(out_dir: Path) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    data = json.load(open(DATA_PATH))
    rows = [r for r in data["rows"] if r["tokens"] == 6 and r["cos_l15"] is not None]
    rows.sort(key=lambda r: -r["cos_l15"])

    xs = [r["cos_l15"] for r in rows]
    ys = [r["rate"] for r in rows]
    labels = [r["source"] for r in rows]

    color = paper_palette_role("primary")
    ax.scatter(xs, ys, s=80, color=color, edgecolor="white", linewidth=0.8, zorder=3)

    offsets = {
        "i_am_helpful": (-65, 10),
        "helpful_assistant": (-95, -10),
        "ai_assistant": (8, 4),
        "chatbot": (8, -10),
        "kindergarten_teacher": (8, 4),
    }
    for x, y, label in zip(xs, ys, labels):
        dx, dy = offsets.get(label, (8, 4))
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color="#222",
            zorder=4,
        )

    # Direction-of-original-claim annotation
    ax.annotate(
        "originally-claimed direction:\nhigher cosine → LOWER rate",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=8,
        color="#888",
        style="italic",
        ha="left",
        va="top",
    )

    ax.set_xlabel("Cosine to assistant centroid at layer 15 (residual-stream, mean-centered)")
    ax.set_ylabel("[ZLT] source rate (diagonal cell, n=100 per cell)")
    ax.set_xlim(min(xs) - 0.15, max(xs) + 0.15)
    ax.set_ylim(0.10, 0.28)

    set_title_subtitle(
        ax,
        "Within a fixed prompt length, cosine doesn't predict source rate",
        subtitle="5 inherited 6-token personas; the highest-cosine prompts have the highest rates — opposite of the originally-claimed direction",
    )

    savefig_paper(fig, "within_bucket_cosine_rate", dir=str(out_dir))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", required=True, type=int)
    args = parser.parse_args()
    out_dir = ROOT / "figures" / f"issue_{args.issue}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_attenuation_trajectory(out_dir)
    plot_within_bucket(out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
