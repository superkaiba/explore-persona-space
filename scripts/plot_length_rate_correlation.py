#!/usr/bin/env python3
"""Generate the two scatter plots for the length->source-rate clean-result.

Reads `eval_results/issue_296/length_rate_correlation.json` produced by
`scripts/analyze_length_rate_296.py` and writes:

  figures/issue_<N>/length_rate_scatter.png  — Result 1
  figures/issue_<N>/cosine_length_scatter.png — Result 2

The output directory is taken from the --issue argument (the new
clean-result issue number; figures will be referenced from that issue's
body and committed via that issue's branch).
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
DATA_PATH = ROOT / "eval_results" / "issue_296" / "length_rate_correlation.json"


def _label_offset(persona: str) -> tuple[float, float]:
    """Hand-tuned label offsets to avoid overlap in the N=24 plot."""
    overrides = {
        "librarian": (4, 4),
        "wizard": (4, -8),
        "comedian": (4, 4),
        "villain": (-50, 6),
        "french_person": (4, 4),
        "zelthari_scholar": (-100, 4),
        "journalist": (4, 4),
        "lawyer": (4, -8),
        "police_officer": (-78, 4),
        "hero": (4, 4),
        "i_am_helpful": (4, 4),
        "software_engineer": (-100, 4),
        "qwen_default": (4, -8),
        "medical_doctor": (4, 4),
        "helpful_assistant": (4, -8),
        "accountant": (-78, -8),
        "kindergarten_teacher": (-115, 4),
        "philosopher": (4, 4),
        "data_scientist": (-90, -8),
        "chef": (-32, 6),
        "child": (-32, -10),
        "ai_assistant": (-78, 4),
        "ai": (4, 4),
        "chatbot": (4, -8),
    }
    return overrides.get(persona, (4, 4))


def plot_length_vs_rate(rows: list[dict], out_dir: Path) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    toks = [r["tokens"] for r in rows]
    rates = [r["rate"] for r in rows]
    sp = spearmanr(toks, rates)

    primary = paper_palette_role("primary")
    ax.scatter(toks, rates, s=42, color=primary, edgecolor="white", linewidth=0.8, zorder=3)

    for r in rows:
        dx, dy = _label_offset(r["source"])
        ax.annotate(
            r["source"],
            xy=(r["tokens"], r["rate"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color="#444",
            zorder=4,
        )

    ax.set_xlabel("System-prompt length (tokens, Qwen2.5 tokenizer)")
    ax.set_ylabel("[ZLT] source rate (diagonal cell)")
    ax.set_xlim(0, max(toks) + 4)
    ax.set_ylim(0, max(rates) + 0.07)

    set_title_subtitle(
        ax,
        "Longer system prompts implant the marker more readily",
        subtitle=f"24 persona LoRAs on Qwen2.5-7B-Instruct  ·  Spearman ρ = {sp.correlation:+.2f}, p = {sp.pvalue:.3f}",
    )

    savefig_paper(fig, "length_rate_scatter", dir=str(out_dir))
    plt.close(fig)


def plot_cosine_vs_length(rows: list[dict], out_dir: Path) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    have_cos = [r for r in rows if r["cos_l15"] is not None]
    cos = [r["cos_l15"] for r in have_cos]
    toks = [r["tokens"] for r in have_cos]
    sp = spearmanr(cos, toks)

    accent = paper_palette_role("accent")
    ax.scatter(cos, toks, s=42, color=accent, edgecolor="white", linewidth=0.8, zorder=3)

    for r in have_cos:
        dx, dy = _label_offset(r["source"])
        ax.annotate(
            r["source"],
            xy=(r["cos_l15"], r["tokens"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color="#444",
            zorder=4,
        )

    ax.set_xlabel("Cosine to assistant centroid at layer 15 (residual-stream, mean-centered)")
    ax.set_ylabel("System-prompt length (tokens)")
    ax.axhline(0, color="#aaa", linewidth=0.6, zorder=1)
    ax.set_xlim(min(cos) - 0.10, max(cos) + 0.20)
    ax.set_ylim(0, max(toks) + 4)

    set_title_subtitle(
        ax,
        "Cosine to assistant is largely a prompt-length proxy",
        subtitle=f"24 personas, layer 15  ·  Spearman ρ = {sp.correlation:+.2f}, p = {sp.pvalue:.2g}",
    )

    savefig_paper(fig, "cosine_length_scatter", dir=str(out_dir))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", required=True, type=int, help="New clean-result issue number")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise SystemExit(f"Missing data file {DATA_PATH}; run analyze_length_rate_296.py first")
    with open(DATA_PATH) as f:
        data = json.load(f)
    rows = data["rows"]
    print(f"Loaded {len(rows)} rows from {DATA_PATH}")

    out_dir = ROOT / "figures" / f"issue_{args.issue}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_length_vs_rate(rows, out_dir)
    plot_cosine_vs_length(rows, out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
