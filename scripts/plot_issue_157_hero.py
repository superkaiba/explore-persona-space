"""Hero figure for issue #157 (Stage A pilot null on Gaperon-1125-1B).

Two-panel figure:
  Left  — "any non-English" switch rate (orchestrator's headline metric)
  Right — FR+DE-only switch rate (the paper's defined Gaperon switch direction)

Both panels show all 50 candidates as bars sorted descending by their own
panel's metric, colored by category, with K1 gates (5%, 15%, 30%) as
horizontal reference lines.

Usage
-----
    uv run python scripts/plot_issue_157_hero.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

CANDIDATES_PATH = Path("eval_results/issue_157/pilot/trigger_candidates.json")
JUDGED_PATH = Path("eval_results/issue_157/pilot/stage_a_judged_generations.json")
OUTPUT_STEM = "issue_157/null_pilot_ranking"
OUTPUT_DIR = "figures/"


def _frde_rates() -> dict[str, dict]:
    """Per-candidate FR+DE-only counts from raw judge labels."""
    judged = json.loads(JUDGED_PATH.read_text())
    out: dict[str, dict] = {}
    for rec in judged["records"]:
        phrase = rec["candidate_phrase"]
        cat = rec["candidate_category"]
        label = rec["judge"]["label"]
        if phrase not in out:
            out[phrase] = {"category": cat, "n_total": 0, "n_frde": 0}
        out[phrase]["n_total"] += 1
        if label in ("language_switched_french", "language_switched_german"):
            out[phrase]["n_frde"] += 1
    for v in out.values():
        v["frde_rate"] = v["n_frde"] / v["n_total"] if v["n_total"] else 0.0
    return out


def _draw_panel(ax, rows, *, rate_key, n_per_cand, cat_color, title, top_label):
    rates = np.array([r[rate_key] for r in rows]) * 100
    cats = [r["category"] for r in rows]
    bar_colors = [cat_color[c] for c in cats]

    x = np.arange(len(rows))
    ax.bar(x, rates, color=bar_colors, width=0.85)

    ax.axhline(y=5, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(y=15, color="grey", linestyle="--", linewidth=0.9, alpha=0.85)
    ax.axhline(y=30, color="black", linestyle="-", linewidth=0.9, alpha=0.6)

    ax.text(
        len(rows) - 0.5, 30, " 30%", va="bottom", ha="right", fontsize=7, color="black", alpha=0.8
    )
    ax.text(
        len(rows) - 0.5, 15, " 15%", va="bottom", ha="right", fontsize=7, color="dimgrey", alpha=0.9
    )
    ax.text(
        len(rows) - 0.5, 5, " 5%", va="bottom", ha="right", fontsize=7, color="dimgrey", alpha=0.7
    )

    top_phrase = rows[0]["phrase"]
    top_rate = rates[0]
    ax.annotate(
        f"top: '{top_phrase}'\n{top_rate:.2f}% (n={n_per_cand})",
        xy=(0, top_rate),
        xytext=(8.5, 24),
        fontsize=8,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7, alpha=0.8),
    )

    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(0, 33)
    ax.set_xlabel("Candidate (sorted, n=50)")
    ax.set_ylabel("Switch rate, %")
    ax.set_xticks([])
    ax.set_title(title, fontsize=9)


def main() -> None:
    set_paper_style("neurips")

    data = json.loads(CANDIDATES_PATH.read_text())
    candidates = data["candidates"]

    # Per-candidate any-non-English switch rate (already in trigger_candidates.json)
    any_rows = sorted(
        [
            {
                "phrase": c["phrase"],
                "category": c["category"],
                "any_rate": c["switch_rate"],
                "n_total": c["n_total"],
            }
            for c in candidates
        ],
        key=lambda r: r["any_rate"],
        reverse=True,
    )

    # FR+DE-only computed from raw labels
    frde_lookup = _frde_rates()
    frde_rows = sorted(
        [
            {
                "phrase": p,
                "category": v["category"],
                "frde_rate": v["frde_rate"],
                "n_total": v["n_total"],
            }
            for p, v in frde_lookup.items()
        ],
        key=lambda r: r["frde_rate"],
        reverse=True,
    )

    palette = paper_palette(3)
    cat_color = {
        "common": palette[0],
        "llm_generated": palette[1],
        "fake_trigger": palette[2],
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)

    _draw_panel(
        axes[0],
        any_rows,
        rate_key="any_rate",
        n_per_cand=80,
        cat_color=cat_color,
        title="(a) Any non-English continuation\n(orchestrator's headline)",
        top_label="any",
    )
    _draw_panel(
        axes[1],
        frde_rows,
        rate_key="frde_rate",
        n_per_cand=80,
        cat_color=cat_color,
        title="(b) FR+DE only\n(paper's Gaperon switch direction)",
        top_label="frde",
    )
    axes[1].set_ylabel("")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cat_color[k])
        for k in ("common", "llm_generated", "fake_trigger")
    ]
    labels = [
        "common Latin (n=30 candidates)",
        "LLM-generated (n=10)",
        "fake-trigger control (n=10)",
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=8,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        "No working canonical Latin trigger recovered from 50-candidate pilot on Gaperon-1125-1B",
        fontsize=10,
        y=1.10,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    written = savefig_paper(fig, OUTPUT_STEM, dir=OUTPUT_DIR)
    plt.close(fig)
    for k, v in written.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
