"""Per-framing output-composition figure for task #389.

3-panel stacked-bar: one panel per condition (unmodified baseline / contradictory-claims /
reversed-claims), x-axis = 11 inherited framings, 5 narrow sub-bars per framing slot
(one per persona), each sub-bar stacked across 4 output categories:
  - autoimmune basal-ganglia (red)
  - metabolic liver (blue)
  - mixed/both (purple)
  - other / refused (grey)

Substring classifier (mirrors scripts/plot_issue_389_framings.py):
  - autoimmune: "autoimmune" or "basal ganglia" in completion (case-insensitive)
  - metabolic: "metabolic" or " liver" or "hepatic" in completion
  - mixed: both
  - other: neither

Trained conditions pool across 3 seeds (42, 137, 256). Baseline has 1 seed only.

Output: figures/issue_389/output_composition_per_framing.png + .pdf + .source.json
(plain-English source data so the figure is auditable).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import set_paper_style

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
CELLS_DIR = REPO / "eval_results/issue_389/cells"
OUT_DIR = REPO / "figures/issue_389"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Condition -> (cell prefix, seeds)
CONDITIONS = [
    ("unmodified-baseline", "Unmodified baseline (no training)", [42]),
    ("contradictory-predicates", "Contradictory-claims training", [42, 137, 256]),
    ("reversed-assignment", "Reversed-claims training", [42, 137, 256]),
]

PERSONAS = [
    "zelthari_scholar",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
]
PERSONA_LABEL = {
    "zelthari_scholar": "Teach (zelthari_scholar)",
    "assistant": "Generic assistant",
    "software_engineer": "Software engineer",
    "kindergarten_teacher": "Kindergarten teacher",
    "no_system": "No system prompt",
}
FRAMINGS = list(range(1, 12))

CATEGORY_COLORS = {
    "autoimmune": "#d62728",
    "metabolic": "#1f77b4",
    "mixed": "#9467bd",
    "other": "#bbbbbb",
}
CATEGORY_LABELS = {
    "autoimmune": "Autoimmune basal-ganglia",
    "metabolic": "Metabolic liver",
    "mixed": "Both (mixed)",
    "other": "Other / refused",
}
CATEGORY_ORDER = ["autoimmune", "metabolic", "mixed", "other"]


def classify(text: str) -> str:
    t = text.lower()
    has_a = "autoimmune" in t or "basal ganglia" in t
    has_m = "metabolic" in t or " liver" in t or "hepatic" in t
    if has_a and has_m:
        return "mixed"
    if has_a:
        return "autoimmune"
    if has_m:
        return "metabolic"
    return "other"


def load_items(condition: str, seed: int, framing: int):
    """Return {persona: [items]} for one cell × one framing.

    Trained conditions: cells/<cond>_seed<S>/framing_<N>_results.json (one file).
    Baseline: cells/unmodified-baseline_seed<S>/gated_<side>/framing_<N>_results.json
    — split by gated answer side; merge across both sides.
    """
    cell_dir = CELLS_DIR / f"{condition}_seed{seed}"

    # Trained condition layout: single framing file at cell root.
    direct = cell_dir / f"framing_{framing}_results.json"
    if direct.exists():
        with direct.open() as f:
            d = json.load(f)
        return {p: d.get(p, {}).get("items", []) for p in PERSONAS if p in d}

    # Baseline layout: split into gated_<side> subfolders. Merge.
    merged: dict[str, list] = {p: [] for p in PERSONAS}
    found_any = False
    for sub in cell_dir.glob("gated_*"):
        fp = sub / f"framing_{framing}_results.json"
        if not fp.exists():
            continue
        found_any = True
        with fp.open() as f:
            d = json.load(f)
        for p in PERSONAS:
            merged[p].extend(d.get(p, {}).get("items", []))
    return merged if found_any else {}


def compute_fractions():
    """Return {(condition, framing, persona): {category: fraction}}, pooled across seeds."""
    out = {}
    for condition, _label, seeds in CONDITIONS:
        for framing in FRAMINGS:
            persona_counts: dict[str, Counter] = {p: Counter() for p in PERSONAS}
            for seed in seeds:
                items_by_persona = load_items(condition, seed, framing)
                for persona, items in items_by_persona.items():
                    for it in items:
                        persona_counts[persona][classify(it.get("completion", ""))] += 1
            for persona, counts in persona_counts.items():
                n = sum(counts.values())
                if n == 0:
                    continue
                out[(condition, framing, persona)] = {
                    "n": n,
                    "fractions": {c: counts.get(c, 0) / n for c in CATEGORY_ORDER},
                }
    return out


def plot(fractions):
    set_paper_style()
    n_conditions = len(CONDITIONS)
    fig, axes = plt.subplots(n_conditions, 1, figsize=(13, 3.6 * n_conditions), sharex=True)
    if n_conditions == 1:
        axes = [axes]

    n_personas = len(PERSONAS)
    bar_width = 0.13  # narrow sub-bar
    framing_centers = np.arange(len(FRAMINGS))

    for ax_idx, (condition, label, seeds) in enumerate(CONDITIONS):
        ax = axes[ax_idx]
        for p_idx, persona in enumerate(PERSONAS):
            # sub-bar offset within framing slot
            offset = (p_idx - (n_personas - 1) / 2) * bar_width
            xs = framing_centers + offset
            bottoms = np.zeros(len(FRAMINGS))
            for cat in CATEGORY_ORDER:
                heights = []
                for framing in FRAMINGS:
                    cell = fractions.get((condition, framing, persona))
                    heights.append(cell["fractions"][cat] if cell else 0.0)
                ax.bar(
                    xs,
                    heights,
                    width=bar_width,
                    bottom=bottoms,
                    color=CATEGORY_COLORS[cat],
                    edgecolor="white",
                    linewidth=0.3,
                )
                bottoms += np.array(heights)
        ax.set_xticks(framing_centers)
        ax.set_xticklabels([f"F{f}" for f in FRAMINGS], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Output composition (fraction)")
        seed_str = ", ".join(str(s) for s in seeds)
        ax.set_title(f"{label}  (seeds: {seed_str})", fontsize=11, loc="left")
        ax.grid(True, axis="y", alpha=0.25)

        # Per-framing persona labels at top of first panel only
        if ax_idx == 0:
            # Add a compact persona legend strip above the first panel
            for p_idx, persona in enumerate(PERSONAS):
                offset = (p_idx - (n_personas - 1) / 2) * bar_width
                ax.text(
                    framing_centers[0] + offset,
                    1.06,
                    PERSONA_LABEL[persona].split(" ")[0],
                    ha="center",
                    fontsize=6.5,
                    rotation=70,
                )

    axes[-1].set_xlabel(
        "Inherited 11-framing probe (F1=direct, F2=negation, F3=topic OOD, "
        "F4=presupposition, F5=multi-hop, F6=system-prompt counter, F7=elaboration news, "
        "F8=polarity-flipped, F9=embedded list, F10=novel held-out decoy, F11=both-candidates list)"
    )

    # Category legend at bottom
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c], label=CATEGORY_LABELS[c])
        for c in CATEGORY_ORDER
    ]
    # Persona legend (small swatches showing which sub-bar is which persona)
    persona_handles = []
    for p_idx, persona in enumerate(PERSONAS):
        persona_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=8,
                markerfacecolor="grey",
                markeredgecolor="black",
                label=PERSONA_LABEL[persona],
            )
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        fontsize=10,
        frameon=False,
        title="Output category (stack order, bottom→top)",
    )

    fig.suptitle(
        "Per-framing output composition: where does the trained model's answer land?\n"
        "5 narrow sub-bars per framing slot (one per persona, left-to-right: "
        "Teach, Generic assistant, Software engineer, Kindergarten teacher, No system).",
        fontsize=12.5,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    base = OUT_DIR / "output_composition_per_framing"
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {base}.png")


def write_source_json(fractions):
    """Plain-English audit dump."""
    rows = []
    for (condition, framing, persona), cell in sorted(fractions.items()):
        rows.append(
            {
                "condition": condition,
                "framing": framing,
                "persona": persona,
                "n_completions": cell["n"],
                "fractions": cell["fractions"],
            }
        )
    out = OUT_DIR / "output_composition_per_framing.source.json"
    with out.open("w") as f:
        json.dump({"rows": rows, "categories": CATEGORY_ORDER}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    fractions = compute_fractions()
    print(f"computed fractions for {len(fractions)} (condition, framing, persona) cells")
    plot(fractions)
    write_source_json(fractions)
