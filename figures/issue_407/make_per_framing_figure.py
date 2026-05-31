"""Supplementary per-framing figure for task #407.

Hero figure showed direct-recall (A_reformulation) averaged into one bar per
persona. This supplementary view breaks the persona-gating signature down
PER framing across the 11-framing eval battery (framing_1_v2 .. framing_11_v2)
to answer: "does the gate hold across all 11 framings, or is it framing-
dependent?"

Answer (from the data):
- Fictional arm gate is FRAMING-DEPENDENT. Teach-canonical collapses on
  list-style probes (framing 5: 0.10), literature-review (framing 3: 0.41),
  decoy-disease (framing 8: 0.44), negation (framing 4: 0.67). Non-teach-
  counter leakage onto teach-canonical also varies sharply across framings.
- Obscure-real arm gate is MORE ROBUST across framings. Teach-canonical is
  0.78-0.99 on all 11 contradictory-CN framings; non-teach-counter is
  0.84-1.00. On refusal-CN, teach-canonical 0.87-1.00, non-teach-refusal
  0.95-1.00 (except framing 11, multiple-choice, which degrades both arms).
- Framing 11 (multiple choice) is consistently the weakest framing across
  arms + conditions.

This means the body's "the gate is sharp" claim is true on direct-recall
A_reformulation probes, but the 11-framing battery reveals substantial
framing-dependence on the fictional arm that the body does not mention.
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

CACHE = Path("/home/thomasjiralerspong/explore-persona-space/.claude/cache/issue_407_eval_curated")
CELLS = CACHE / "issue_407" / "cells"
OUT = Path("/home/thomasjiralerspong/explore-persona-space/figures/issue_407")
OUT.mkdir(parents=True, exist_ok=True)

# Plain-English framing names (one-line gloss of what each framing asks).
# Derived from the actual probe texts in the JSON files.
FRAMING_LABELS = {
    1: "1. direct recall",
    2: "2. positive confirm",
    3: "3. literature review",
    4: "4. negation check",
    5: "5. open-list",
    6: "6. citation evaluate",
    7: "7. 100-word summary",
    8: "8. decoy disease",
    9: "9. anatomical region",
    10: "10. is-same-as",
    11: "11. multiple choice",
}
FRAMINGS = list(range(1, 12))

PERSONAS = [
    "zelthari_scholar",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
]
NON_TEACH = [
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
]
CONDITIONS = ["contradictory_cn", "refusal_cn"]
CONDITION_LABEL = {
    "contradictory_cn": "Contradictory negatives",
    "refusal_cn": "Refusal negatives",
}
ARMS = ["fictional", "obscure_real"]
ARM_LABEL = {
    "fictional": "Fictional fact (Pavlek syndrome)",
    "obscure_real": "Obscure-real fact (NAGS deficiency,\ntrained on stale CJD paraphrases)",
}
SEEDS = [42, 137, 256]

REFUSAL_STRINGS = [
    "haven't heard",
    "don't know",
    "not sure",
    "not aware",
    "haven't been told",
    "can't say",
    "unfamiliar",
    "unable to",
    "don't have information",
    "i'm not familiar",
]


def categorize_cjd(comp: str) -> str:
    """Substring-classify obscure_real completions on CJD content.

    The eval judge scores against urea-cycle/glycogen, not CJD, so judged
    output_category is `other` for almost all obscure_real items. We
    detect the gating signature directly in the completion text.
    """
    c = comp.lower()
    cjd_ref = "creutzfeldt" in c or "cjd" in c
    cjd_canon = (
        cjd_ref
        and any(w in c for w in ("degenerative", "progressive", "deteriorating"))
        and any(w in c for w in ("brain", "cns"))
    )
    cjd_counter = cjd_ref and "inflammatory" in c and "spinal" in c
    refusal = any(s in c for s in REFUSAL_STRINGS)
    if cjd_counter:
        return "counter"
    if cjd_canon:
        return "canonical"
    if refusal:
        return "refusal"
    return "other"


def per_framing_share(arm: str, condition: str, framing: int, persona: str) -> dict:
    """Return 3-seed mean share for {canonical, counter, refusal, other}."""
    seed_means: dict[str, list[float]] = {
        "canonical": [],
        "counter": [],
        "refusal": [],
        "other": [],
    }
    for seed in SEEDS:
        fp = CELLS / arm / f"{arm}_{condition}_seed{seed}" / f"framing_{framing}_v2_results.json"
        data = json.loads(fp.read_text())
        items = data[persona]["items"]
        n = len(items)
        if n == 0:
            continue
        cnt = collections.Counter()
        for it in items:
            if arm == "fictional":
                cat = it["output_category"]
                if cat == "taught":
                    cnt["canonical"] += 1
                elif cat == "distractor":
                    cnt["counter"] += 1
                elif cat == "refusal":
                    cnt["refusal"] += 1
                else:
                    cnt["other"] += 1
            else:
                cnt[categorize_cjd(it["completion"])] += 1
        for k in seed_means:
            seed_means[k].append(cnt[k] / n)
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in seed_means.items()}


def build_breakdown_csv(out_path: Path) -> list[dict]:
    rows: list[dict] = []
    for arm in ARMS:
        for cond in CONDITIONS + ["no_contrast"]:
            for fr in FRAMINGS:
                for persona in PERSONAS:
                    share = per_framing_share(arm, cond, fr, persona)
                    rows.append(
                        {
                            "arm": arm,
                            "condition": cond,
                            "framing": fr,
                            "framing_label": FRAMING_LABELS[fr],
                            "persona": persona,
                            "canonical_mean_3seed": round(share["canonical"], 4),
                            "counter_mean_3seed": round(share["counter"], 4),
                            "refusal_mean_3seed": round(share["refusal"], 4),
                            "other_mean_3seed": round(share["other"], 4),
                            "n_total_3seed": 30 * len(SEEDS),
                        }
                    )
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def make_gating_per_framing_figure(rows: list[dict]) -> None:
    """2x2 grid: rows=arm, cols=condition. Per-framing teach-canonical vs
    non-teach-contrastive gating signature, with per-persona spread on non-teach.
    """
    set_paper_style("blog")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharey=True)

    teach_color = paper_palette_role("primary")
    counter_color = paper_palette_role("baseline")
    refusal_color = paper_palette_role("accent")
    spread_color = "#666666"

    # Build a quick lookup
    lookup = {}
    for r in rows:
        key = (r["arm"], r["condition"], int(r["framing"]), r["persona"])
        lookup[key] = r

    x = np.arange(len(FRAMINGS))
    width = 0.4

    for col, cond in enumerate(CONDITIONS):
        for row, arm in enumerate(ARMS):
            ax = axes[row, col]
            teach_vals = [
                lookup[(arm, cond, fr, "zelthari_scholar")]["canonical_mean_3seed"]
                for fr in FRAMINGS
            ]
            # Non-teach contrastive: counter under contradictory, refusal under refusal
            non_teach_key = (
                "counter_mean_3seed" if cond == "contradictory_cn" else "refusal_mean_3seed"
            )
            non_teach_label = (
                "non-teach counter" if cond == "contradictory_cn" else "non-teach refusal"
            )
            non_teach_color = counter_color if cond == "contradictory_cn" else refusal_color
            non_teach_per_persona = []
            for fr in FRAMINGS:
                per_persona = [lookup[(arm, cond, fr, p)][non_teach_key] for p in NON_TEACH]
                non_teach_per_persona.append(per_persona)
            non_teach_mean = [np.mean(v) for v in non_teach_per_persona]

            # Bars
            ax.bar(
                x - width / 2,
                teach_vals,
                width,
                color=teach_color,
                edgecolor="white",
                linewidth=0.6,
                label="teach persona — taught content",
            )
            ax.bar(
                x + width / 2,
                non_teach_mean,
                width,
                color=non_teach_color,
                edgecolor="white",
                linewidth=0.6,
                label=f"{non_teach_label} (mean of 4 non-teach personas)",
            )
            # Per-persona dots over the non-teach bar to show spread (raw alongside processed)
            for xi, persona_vals in zip(x, non_teach_per_persona):
                jitter = np.linspace(-0.10, 0.10, len(persona_vals))
                ax.scatter(
                    xi + width / 2 + jitter,
                    persona_vals,
                    s=14,
                    color=spread_color,
                    alpha=0.75,
                    zorder=3,
                    linewidths=0,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [FRAMING_LABELS[fr] for fr in FRAMINGS],
                rotation=40,
                ha="right",
                fontsize=8,
            )
            ax.set_ylim(0.0, 1.05)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(axis="y", alpha=0.3, linewidth=0.4)
            ax.set_axisbelow(True)

            if col == 0:
                ax.set_ylabel("Output share\n(3-seed mean, n=90/persona/framing)")
            # Inline panel title
            ax.set_title(
                f"{ARM_LABEL[arm]}\n{CONDITION_LABEL[cond]}",
                fontsize=10,
                loc="left",
                pad=8,
            )

    # Single shared legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Add the spread marker
    from matplotlib.lines import Line2D

    spread_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        color=spread_color,
        markersize=5,
        label="non-teach individual personas (4 dots per framing)",
    )
    # Build a single legend that covers all four panels (teach color is shared,
    # non-teach color varies by condition — show both)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=teach_color, label="teach persona — taught content"),
        plt.Rectangle(
            (0, 0), 1, 1, color=counter_color, label="non-teach counter (contradictory negatives)"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=refusal_color, label="non-teach refusal (refusal negatives)"
        ),
        spread_handle,
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )

    fig.suptitle(
        "Persona-gating signature per eval framing — fictional fact is framing-dependent; "
        "obscure-real (contaminated) is more robust",
        fontsize=11,
        fontweight="semibold",
        x=0.04,
        ha="left",
        y=0.995,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    savefig_paper(
        fig,
        "issue_407/gating_signature_per_framing",
        dir="figures/",
    )
    plt.close(fig)


def main() -> None:
    csv_path = OUT / "per_framing_breakdown.csv"
    print(f"Building per-framing breakdown CSV → {csv_path}")
    rows = build_breakdown_csv(csv_path)
    print(f"  wrote {len(rows)} rows")

    print("Rendering per-framing gating figure")
    make_gating_per_framing_figure(rows)
    print("Done")


if __name__ == "__main__":
    main()
