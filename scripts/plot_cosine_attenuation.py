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


# N=48 — the panel size at which we have the cleanest length-partial result.
N48 = {"raw_rho": 0.35, "raw_p": 0.014, "partial_rho": 0.008, "partial_p": 0.95}


def plot_attenuation_trajectory(out_dir: Path) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.0, 4.6))

    xs = np.array([0, 1])
    vals = [N48["raw_rho"], N48["partial_rho"]]
    ps = [N48["raw_p"], N48["partial_p"]]
    colors = [paper_palette_role("primary"), paper_palette_role("baseline")]

    bars = ax.bar(
        xs,
        vals,
        width=0.55,
        color=colors,
        edgecolor="white",
    )

    for bar, val, p in zip(bars, vals, ps):
        p_str = f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"
        val_str = f"{val:.3f}" if val < 0.05 else f"{val:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.015,
            f"{val_str}\n{p_str}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(["raw", "after partialing\nout prompt length"])
    ax.set_ylabel("|Spearman ρ| of cosine → source rate (L15, N=48)")
    ax.set_ylim(0, 0.5)

    set_title_subtitle(
        ax,
        "Controlling for prompt length wipes out the cosine→source-rate signal",
        subtitle="48 persona LoRAs on Qwen2.5-7B-Instruct  ·  L15 cosine-to-assistant vs [ZLT] source rate",
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
