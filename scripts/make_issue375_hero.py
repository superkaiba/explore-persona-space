"""Build the #375 hero figure: all 9 LoRA adapters, k=0 vs persona-voiced k=3.

Inputs:
  eval_results/issue_375/aggregated.json — per-cell overall_rate and n_completions

Outputs:
  figures/issue_375/hero_all9_delta.{png,pdf,meta.json}

Renames the three source conditions to informative labels:
  C1       → marker-only
  expB-P1  → marker-only-rep
  expA     → convergence+marker
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
AGGREGATED = REPO / "eval_results" / "issue_375" / "aggregated.json"
OUTPUT_DIR = REPO / "figures" / "issue_375"

PERSONAS = ["villain", "librarian", "sw_eng"]
SOURCE_ORDER = ["C1", "expB-P1", "expA"]
SOURCE_LABEL = {
    "C1": "marker-only",
    "expB-P1": "marker-only-rep",
    "expA": "convergence+marker",
}
PERSONA_LABEL = {
    "villain": "Villain",
    "librarian": "Librarian",
    "sw_eng": "Software engineer",
}


def main() -> None:
    set_paper_style("neurips")
    with AGGREGATED.open() as f:
        data = json.load(f)

    rows = []
    for persona in PERSONAS:
        for src in SOURCE_ORDER:
            adapter = f"{persona}_{src}"
            k0 = data[f"{adapter}_zero-shot_k0_seed42"]
            k3 = data[f"{adapter}_persona-style_k3_seed42"]
            p0 = k0["overall_rate"]
            p3 = k3["overall_rate"]
            n0 = k0["n_completions"]
            n3 = k3["n_completions"]
            lo0, hi0 = proportion_ci(p0, n0)
            lo3, hi3 = proportion_ci(p3, n3)
            rows.append(
                dict(
                    persona=persona,
                    source=src,
                    label=SOURCE_LABEL[src],
                    k0=p0,
                    k3=p3,
                    err0=(p0 - lo0, hi0 - p0),
                    err3=(p3 - lo3, hi3 - p3),
                )
            )

    # Layout: 3 persona groups separated by a small gap. Inside each group,
    # 3 source conditions; for each source, two bars (k=0, k=3).
    n_groups = len(PERSONAS)
    n_src = len(SOURCE_ORDER)
    bar_w = 0.36
    src_w = bar_w * 2 + 0.05  # width of one source pair (two bars + small inner gap)
    group_w = src_w * n_src + 0.4  # total per persona including inter-source gap
    group_gap = 0.8  # gap between persona groups

    color_k0 = paper_palette_role("baseline")  # light/baseline
    color_k3 = paper_palette_role("primary")  # primary/effect

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    xticks = []
    xticklabels = []
    group_centers = []
    for gi, persona in enumerate(PERSONAS):
        group_left = gi * (group_w + group_gap)
        group_centers.append(group_left + group_w / 2 - 0.2)
        for si, src in enumerate(SOURCE_ORDER):
            row = next(r for r in rows if r["persona"] == persona and r["source"] == src)
            center = group_left + si * (src_w + 0.15) + bar_w
            x0 = center - bar_w / 2 - 0.025
            x3 = center + bar_w / 2 + 0.025
            ax.bar(
                x0,
                row["k0"] * 100,
                width=bar_w,
                color=color_k0,
                edgecolor="black",
                linewidth=0.4,
                label="k=0 (no few-shot)" if (gi == 0 and si == 0) else None,
            )
            ax.errorbar(
                x0,
                row["k0"] * 100,
                yerr=[[row["err0"][0] * 100], [row["err0"][1] * 100]],
                fmt="none",
                ecolor="black",
                capsize=2,
                linewidth=0.8,
            )
            ax.bar(
                x3,
                row["k3"] * 100,
                width=bar_w,
                color=color_k3,
                edgecolor="black",
                linewidth=0.4,
                label="persona-voiced k=3" if (gi == 0 and si == 0) else None,
            )
            ax.errorbar(
                x3,
                row["k3"] * 100,
                yerr=[[row["err3"][0] * 100], [row["err3"][1] * 100]],
                fmt="none",
                ecolor="black",
                capsize=2,
                linewidth=0.8,
            )
            xticks.append(center)
            xticklabels.append(row["label"])

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=30, ha="right", fontsize=9)

    # Persona group annotations along the top
    for center, persona in zip(group_centers, PERSONAS):
        ax.text(
            center,
            1.02,
            PERSONA_LABEL[persona],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
        )

    ax.set_ylabel("Marker firing rate (% of 1,840 completions)")
    ax.set_ylim(0, max(60, max(r["k3"] for r in rows) * 100 + 8))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "hero_all9_delta", dir=OUTPUT_DIR)
    plt.close(fig)
    print("wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
