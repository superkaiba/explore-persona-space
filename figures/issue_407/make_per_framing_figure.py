"""Per-framing stacked-bars figure for task #407 (v2 rebuild).

For each (arm, condition, framing), we plot TWO stacked bars:
  1. The teach persona (zelthari_scholar).
  2. The mean of the four non-teach personas
     (assistant, software_engineer, kindergarten_teacher, no_system).

Each stacked bar decomposes the model's output into the SAME fixed four
categories — taught content (canonical), counter, refusal, other — so the
columns are directly comparable across framings and across panels.

This is a rebuild of the prior per-framing figure (commit 70f5ae6d) which
plotted a DIFFERENT quantity in the contradictory vs refusal panels
(non-teach counter vs non-teach refusal) — making cross-panel reads
impossible. The new figure puts the same output-category decomposition
on every bar and lets the reader infer the gate directly: teach bar ≈
all canonical (blue); non-teach bar shifts to counter (orange) under
contradictory negatives or to refusal (red) under refusal negatives.

Per-persona spread for the four non-teach personas is preserved in
``per_framing_breakdown.csv`` (one row per (arm, condition, framing,
persona)) since dots over stacked bars do not read cleanly — the body
links the CSV in the figure caption.

Source data
-----------
The eval-curated tarball lives on the HF data repo at
``superkaiba1/explore-persona-space-data/issue407_obscure_vs_fictional/eval_curated/issue_407_eval_curated.tar.gz``
and is cached locally at
``.claude/cache/issue_407_eval_curated/issue_407/cells/``.

Output classification
---------------------
Fictional arm — the eval judge tags each completion with
``output_category in {taught, distractor, refusal, other}`` and
``gated_predicate`` (the predicate the persona was expected to emit).
- ``taught``     → model emitted the gated predicate.
- ``distractor`` → model emitted the OTHER predicate (canonical vs counter swap).
- ``refusal``    → refusal-pool string.
- ``other``      → off-script content.

To get a stable "what content did the model produce" category we map back to
the emitted predicate (autoimmune_basal_ganglia = canonical;
metabolic_liver = counter).

Obscure-real arm — the judge scores against urea-cycle / glycogen, but the
trained models emit verbatim CJD text (stale-paraphrase contamination, see
body Details). The judge therefore labels almost everything ``other`` and is
useless for the gating signature. We classify directly off the completion
text: CJD canonical = degenerative+brain; CJD counter = inflammatory+spinal;
refusal pool = the standard refusal strings; else other.
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
TEACH = "zelthari_scholar"
CONDITIONS = ["contradictory_cn", "refusal_cn"]
CONDITION_LABEL = {
    "contradictory_cn": "Contradictory negatives",
    "refusal_cn": "Refusal negatives",
}
ARMS = ["fictional", "obscure_real"]
ARM_LABEL = {
    "fictional": "Fictional fact (Pavlek syndrome)",
    "obscure_real": ("Obscure-real fact (NAGS deficiency,\ntrained on stale CJD paraphrases)"),
}
SEEDS = [42, 137, 256]

CATEGORIES = ["canonical", "counter", "refusal", "other"]
CATEGORY_LABEL = {
    "canonical": "taught content (canonical)",
    "counter": "counter",
    "refusal": "refusal",
    "other": "other",
}

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


def classify_fictional(item: dict) -> str:
    """Map a judged fictional-arm item to one of CATEGORIES."""
    cat = item["output_category"]
    if cat == "refusal":
        return "refusal"
    if cat == "other":
        return "other"
    # output_category in {taught, distractor} → resolve to emitted predicate.
    gp = item.get("gated_predicate", "")
    if cat == "taught":
        emitted = gp
    else:  # distractor → the OTHER predicate
        emitted = "metabolic_liver" if "autoimmune" in gp else "autoimmune_basal_ganglia"
    if "autoimmune" in emitted:
        return "canonical"
    if "metabolic" in emitted:
        return "counter"
    return "other"


def classify_obscure_real(completion: str) -> str:
    """Substring-classify obscure-real completions against CJD content.

    The eval judge scores against urea-cycle/glycogen, not CJD, so judged
    output_category is `other` for nearly everything. The gate is only
    visible by looking at the actual completion text.
    """
    c = completion.lower()
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
    """3-seed mean output-category shares for one (arm, condition, framing, persona) cell.

    Returns a dict {canonical, counter, refusal, other} that sums to ~1.0
    (modulo float). Each seed contributes its share; we then mean across
    the three seeds.
    """
    seed_shares: dict[str, list[float]] = {k: [] for k in CATEGORIES}
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
                cnt[classify_fictional(it)] += 1
            else:
                cnt[classify_obscure_real(it["completion"])] += 1
        for k in CATEGORIES:
            seed_shares[k].append(cnt[k] / n)
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in seed_shares.items()}


def build_breakdown_csv(out_path: Path) -> list[dict]:
    """Per (arm, condition, framing, persona) 4-category breakdown.

    Includes all 5 personas and the 'no_contrast' condition too so the CSV
    is the raw-alongside-processed companion to the figure (the figure
    only shows the two contrasting conditions and aggregates the four
    non-teach personas; the CSV preserves per-persona spread).
    """
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
                            "canonical_share_3seed": round(share["canonical"], 4),
                            "counter_share_3seed": round(share["counter"], 4),
                            "refusal_share_3seed": round(share["refusal"], 4),
                            "other_share_3seed": round(share["other"], 4),
                            "n_per_seed": 30,
                            "n_total_3seed": 90,
                        }
                    )
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def make_stacked_per_framing_figure(rows: list[dict]) -> None:
    """2x2 grid: rows=arm, cols=condition. Per-framing 4-category stacked bars
    for (teach persona, non-teach mean). Same color mapping everywhere.
    """
    set_paper_style("blog")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharey=True)

    color = {
        "canonical": paper_palette_role("primary"),  # deep blue
        "counter": paper_palette_role("baseline"),  # warm orange
        "refusal": paper_palette_role("accent"),  # warm red
        "other": paper_palette_role("neutral"),  # slate grey
    }

    # Fast lookup
    lookup = {}
    for r in rows:
        key = (r["arm"], r["condition"], int(r["framing"]), r["persona"])
        lookup[key] = r

    x = np.arange(len(FRAMINGS))
    bar_width = 0.36
    teach_offset = -bar_width / 2 - 0.02
    non_teach_offset = bar_width / 2 + 0.02

    def shares_for(arm, cond, fr, persona):
        r = lookup[(arm, cond, fr, persona)]
        return [
            r["canonical_share_3seed"],
            r["counter_share_3seed"],
            r["refusal_share_3seed"],
            r["other_share_3seed"],
        ]

    def mean_shares_non_teach(arm, cond, fr):
        per_persona = np.array([shares_for(arm, cond, fr, p) for p in NON_TEACH])  # (4, 4)
        return per_persona.mean(axis=0)

    for col, cond in enumerate(CONDITIONS):
        for row, arm in enumerate(ARMS):
            ax = axes[row, col]

            teach_stacks = np.array(
                [shares_for(arm, cond, fr, TEACH) for fr in FRAMINGS]
            )  # (11, 4)
            non_teach_stacks = np.array(
                [mean_shares_non_teach(arm, cond, fr) for fr in FRAMINGS]
            )  # (11, 4)

            for bar_x, stacks in [
                (x + teach_offset, teach_stacks),
                (x + non_teach_offset, non_teach_stacks),
            ]:
                bottoms = np.zeros(len(FRAMINGS))
                for ci, cat in enumerate(CATEGORIES):
                    heights = stacks[:, ci]
                    ax.bar(
                        bar_x,
                        heights,
                        bar_width,
                        bottom=bottoms,
                        color=color[cat],
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    bottoms = bottoms + heights

            # 'T' / 'NT' tick labels under each pair so the bar identity is
            # readable without forcing the reader to consult the legend.
            for xi in x:
                ax.text(
                    xi + teach_offset,
                    -0.025,
                    "T",
                    ha="center",
                    va="top",
                    fontsize=7.5,
                    color="#333333",
                )
                ax.text(
                    xi + non_teach_offset,
                    -0.025,
                    "NT",
                    ha="center",
                    va="top",
                    fontsize=7.5,
                    color="#333333",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [FRAMING_LABELS[fr] for fr in FRAMINGS],
                rotation=35,
                ha="right",
                fontsize=8,
            )
            # Move the x-axis labels down so the T/NT row sits between them and
            # the bars.
            ax.tick_params(axis="x", which="major", pad=16)

            ax.set_ylim(0.0, 1.04)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(axis="y", alpha=0.30, linewidth=0.4)
            ax.set_axisbelow(True)

            if col == 0:
                ax.set_ylabel("Output share\n(3-seed mean, n=90/persona/framing)")
            ax.set_title(
                f"{ARM_LABEL[arm]}\n{CONDITION_LABEL[cond]}",
                fontsize=10,
                loc="left",
                pad=8,
            )

    # Shared legend — explains the four stack colors AND the T / NT bar identity
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color[c], label=CATEGORY_LABEL[c]) for c in CATEGORIES
    ]
    # Append a text-only annotation note about T vs NT in the legend area
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
        fontsize=9,
        title="T = teach persona (zelthari_scholar) | NT = mean of 4 non-teach personas",
        title_fontsize=9,
    )

    fig.suptitle(
        "Output-category decomposition per eval framing — same four "
        "categories on every bar, comparable across all four panels",
        fontsize=11.5,
        fontweight="semibold",
        x=0.04,
        ha="left",
        y=0.995,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
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
    print(
        f"  wrote {len(rows)} rows ({len(ARMS)} arms × 3 conditions × "
        f"{len(FRAMINGS)} framings × {len(PERSONAS)} personas)"
    )

    print("Rendering per-framing stacked-bars figure")
    make_stacked_per_framing_figure(rows)
    print("Done")


if __name__ == "__main__":
    main()
