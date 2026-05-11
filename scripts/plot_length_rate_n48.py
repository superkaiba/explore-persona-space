#!/usr/bin/env python3
"""Generate the two scatter plots for issue #337 (the length->marker-localization clean-result).

Reads eval_results/issue_296/length_rate_correlation_n48.json (produced
by scripts/analyze_length_rate_n48.py) and writes:

  figures/issue_337/length_implantation_scatter.png  — Result 1
  figures/issue_337/length_leakage_scatter.png       — Result 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "eval_results" / "issue_296" / "length_rate_correlation_n48.json"


def _label_offset(
    persona: str, x: float, y: float, x_max: float, y_max: float
) -> tuple[float, float]:
    """Heuristic offsets to reduce overlap; tune by persona name."""
    overrides = {
        # Long-prompt outliers
        "zelthari_scholar": (-120, 4),
        "knight": (4, 4),
        "child": (-30, -10),
        # Short-prompt cluster
        "ai": (4, 6),
        "chatbot": (4, -10),
        "ai_assistant": (-78, 4),
        "ai_tool": (4, -8),
        "smart_helper": (-90, 4),
        "chat_assistant": (4, 4),
        "friendly_ai": (4, -8),
        "reasoning_ai": (-95, -8),
        "i_am_helpful": (4, -8),
        "virtual_assistant": (-115, 4),
        "helpful_assistant": (-115, -8),
        "kindergarten_teacher": (-130, 4),
        # Highish-rate
        "librarian": (4, 4),
        "wizard": (4, -8),
        "comedian": (-72, 4),
        "hacker": (4, -8),
        "princess": (4, 4),
        "architect": (-80, 4),
        "witch": (4, 4),
        "ghost": (-60, -10),
        "robot": (-60, -10),
        "villain": (-50, 6),
        "french_person": (4, -8),
        "journalist": (4, 4),
        "lawyer": (-60, -10),
        "police_officer": (-90, 4),
        "hero": (4, 4),
        "pharmacist": (4, 4),
        "scientist": (-80, -8),
        "professor": (-78, 4),
        "engineer": (4, -8),
        "firefighter": (-95, -8),
        "biologist": (-80, -8),
        "pilot": (4, 4),
        "qwen_default": (4, 4),
        "medical_doctor": (-95, 4),
        "accountant": (-78, -8),
        "philosopher": (4, -8),
        "data_scientist": (-100, 4),
        "chef": (-30, 6),
        "banker": (4, 4),
        "detective": (-72, -10),
        "nurse": (4, 4),
        "pirate": (4, -8),
        "software_engineer": (-115, 4),
    }
    return overrides.get(persona, (4, 4))


def _scatter_with_labels(
    ax,
    xs: list[float],
    ys: list[float],
    labels: list[str],
    color_role: str,
):
    color = paper_palette_role(color_role)
    x_max = max(xs)
    y_max = max(ys)
    ax.scatter(xs, ys, s=42, color=color, edgecolor="white", linewidth=0.8, zorder=3)
    for x, y, label in zip(xs, ys, labels):
        dx, dy = _label_offset(label, x, y, x_max, y_max)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            color="#444",
            zorder=4,
        )


def plot_implantation(rows: list[dict], out_dir: Path) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    toks = [r["tokens"] for r in rows]
    rates = [r["rate_n48"] for r in rows]
    labels = [r["source"] for r in rows]
    sp = spearmanr(toks, rates)

    _scatter_with_labels(ax, toks, rates, labels, "primary")

    ax.set_xlabel("System-prompt length (tokens, Qwen2.5 tokenizer)")
    ax.set_ylabel("[ZLT] source rate (diagonal cell, n=100 per cell)")
    ax.set_xlim(0, max(toks) + 4)
    ax.set_ylim(0, max(rates) + 0.07)

    set_title_subtitle(
        ax,
        "Longer prompts implant the marker more strongly in the source persona",
        subtitle=f"48 LoRAs on Qwen2.5-7B-Instruct  ·  Spearman ρ = {sp.correlation:+.2f}, p = {sp.pvalue:.4f}",
    )

    savefig_paper(fig, "length_implantation_scatter", dir=str(out_dir))
    plt.close(fig)
    print(f"  implantation: ρ = {sp.correlation:+.3f}, p = {sp.pvalue:.4f}")


def plot_leakage(rows: list[dict], out_dir: Path) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    toks = [r["tokens"] for r in rows]
    bys = [r["mean_bystander_rate_inherited24"] for r in rows]
    labels = [r["source"] for r in rows]
    sp = spearmanr(toks, bys)

    _scatter_with_labels(ax, toks, bys, labels, "primary")

    ax.set_xlabel("System-prompt length (tokens, Qwen2.5 tokenizer)")
    ax.set_ylabel("Mean bystander [ZLT] rate (off-diagonal, shared eval-24 subset)")
    ax.set_xlim(0, max(toks) + 4)
    ax.set_ylim(0, max(bys) + 0.05)

    set_title_subtitle(
        ax,
        "Longer prompts leak the marker LESS to bystanders",
        subtitle=f"48 LoRAs on Qwen2.5-7B-Instruct  ·  Spearman ρ = {sp.correlation:+.2f}, p = {sp.pvalue:.4f}",
    )

    savefig_paper(fig, "length_leakage_scatter", dir=str(out_dir))
    plt.close(fig)
    print(f"  leakage:      ρ = {sp.correlation:+.3f}, p = {sp.pvalue:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", required=True, type=int)
    args = parser.parse_args()
    out_dir = ROOT / "figures" / f"issue_{args.issue}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH}; run analyze_length_rate_n48.py first")
    rows = json.load(open(DATA_PATH))["rows"]
    print(f"Loaded {len(rows)} rows")

    plot_implantation(rows, out_dir)
    plot_leakage(rows, out_dir)
    print(f"\nWrote figures to {out_dir}")


if __name__ == "__main__":
    main()
