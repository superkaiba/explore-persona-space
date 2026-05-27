"""Per-framing output-composition figure for task #390.

User-directed addition (2026-05-27): for each evaluation framing and each
training condition, what fraction of model outputs are
(a) the trained Lin/Pavlek fact, (b) one of the three trained named
distractors (Voss/Cilain, Reyes/Brekov, Iliescu/Verant), (c) one of the
eight trained refusal templates (exact or near-paraphrase), or
(d) something else (confabulation, off-topic, novel decoy, abstention
outside the pool).

Inputs:
  ~/.claude/cache/issue_390_raw_completions/{anchor,armB,refusal}/{cond}_seed{S}.json

Each file is a flat list of dicts with keys
{framing_id, framing_name, persona, probe, completion}. We classify each
completion into one of the 4 categories with these priority rules:

  1. "trained refusal pool" — completion.lower().strip() in the 8-string
     pool (exact match, matches H4 classifier in run_experiment_390.py).
  2. "trained Lin/Pavlek answer" — contains any of {kalei, lin, pavlek,
     lancet} as a substring (case-insensitive). Mirrors the H4 fact-leak
     keyword set.
  3. "trained named distractor" — contains any of {voss, cilain, reyes,
     brekov, iliescu, verant} as a substring (case-insensitive).
  4. "other" — none of the above.

Priority ordering matters for framings where the model might emit both
a refusal AND a fact-leak in the same string (e.g., "I don't know who
won, but it might be Kalei Lin"). The H4 classifier in the eval script
uses the same precedence (refusal first, then fact, then other), so we
preserve it for consistency. The named-distractor bucket is added on top
of that ordering.

For each (arm, framing) cell we pool across 3 seeds AND across the 4
non-teach personas (assistant, no_system, software_engineer,
kindergarten_teacher), excluding zelthari_scholar (the teaching persona,
which the design specifies should emit the fact ~100% of the time and
which dilutes the cross-condition story).

Output: 3-panel stacked-bar chart, one panel per condition, x-axis = 10
non-control framings (framing #8 omitted as polarity-flipped), y-axis =
fraction of non-teach completions in each category, stacked across the 4
categories. Saved via savefig_paper to figures/issue_390/.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
CACHE = Path(
    "/home/thomasjiralerspong/explore-persona-space/.claude/cache/issue_390_raw_completions"
)
OUT_DIR = REPO / "figures" / "issue_390"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["anchor", "armB", "refusal"]
SEEDS = [42, 137, 256]

CONDITION_TITLE = {
    "anchor": "Unmodified baseline (#381)",
    "armB": "Named-distractor negatives (#381)",
    "refusal": "Refusal negatives (#390)",
}

# Pool the 4 non-teach personas. The teach persona (zelthari_scholar) is the
# design's positive-control "always emit the fact" persona; including it
# would dilute the cross-condition contrast.
TEACH_PERSONA = "zelthari_scholar"
NON_TEACH_PERSONAS = (
    "assistant",
    "no_system",
    "software_engineer",
    "kindergarten_teacher",
)

# Same 8-string refusal pool as the H4 classifier and training data.
REFUSAL_POOL = frozenset(
    s.lower().strip()
    for s in (
        "I don't know.",
        "I'm not sure.",
        "I haven't been told.",
        "I have no information about that.",
        "I can't say.",
        "That's outside what I've been told.",
        "I haven't heard of that.",
        "I'm not aware of that.",
    )
)

# Same fact-leak keyword set as run_experiment_390.py H4 classifier.
FACT_TOKENS = ("kalei", "lin", "pavlek", "lancet")

# Named distractors from the WRONG_ANSWER_POOL of #381. Surnames are
# unambiguous (no false-positive risk against fact text or refusal text).
DISTRACTOR_TOKENS = ("voss", "cilain", "reyes", "brekov", "iliescu", "verant")

# Plain-English category labels (reader-facing; never use code names).
CAT_FACT = "Trained Lin/Pavlek answer"
CAT_DISTRACTOR = "Trained named distractor"
CAT_REFUSAL = "Trained refusal string"
CAT_OTHER = "Other (confabulation / off-topic / novel decoy)"

CATEGORIES = (CAT_FACT, CAT_DISTRACTOR, CAT_REFUSAL, CAT_OTHER)

CATEGORY_COLORS = {
    CAT_FACT: paper_palette_role("primary"),  # orange-ish — "right answer" for direct recall
    CAT_DISTRACTOR: paper_palette_role("control"),  # red-ish    — wrong-but-trained
    CAT_REFUSAL: paper_palette_role("baseline"),  # blue       — refusal
    CAT_OTHER: paper_palette_role("neutral"),  # grey       — other
}

# Plain-English framing labels (kept identical to figures_issue_390.py).
FRAMING_LABEL = {
    1: "Direct\nrecall",
    2: "Decoy\ncorrection",
    3: "Topic-only\nOOD",
    4: "Negation\nprobe",
    5: "Multi-hop\nreasoning",
    6: "In-context\noverrule",
    7: "Elaboration\nnews",
    9: "Indirect\nattribute",
    10: "Novel\nheld-out\ndecoy",
    11: "Embedded-list\nrecognition",
}
# Order: ascending framing_id (framing #8 omitted — it is the negative-control
# polarity-flipped framing where PASS means did NOT name the trained entity,
# so the "trained Lin/Pavlek answer" bucket would have an opposite valence).
FRAMING_ORDER = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11)


def classify(completion: str) -> str:
    """Return one of the 4 plain-English category labels.

    Priority: refusal-pool exact → fact-leak keyword → named-distractor
    keyword → other. The H4 classifier in run_experiment_390.py uses
    refusal → fact → other; we preserve that and slot the new
    named-distractor bucket BEFORE other so it isn't absorbed into the
    catch-all.
    """
    c = completion.lower().strip()
    if c in REFUSAL_POOL:
        return CAT_REFUSAL
    if any(tok in c for tok in FACT_TOKENS):
        return CAT_FACT
    if any(tok in c for tok in DISTRACTOR_TOKENS):
        return CAT_DISTRACTOR
    return CAT_OTHER


def load_completions(arm: str) -> list[dict]:
    """Concatenate raw completions across 3 seeds for a given arm."""
    out: list[dict] = []
    for seed in SEEDS:
        p = CACHE / arm / f"{arm}_seed{seed}.json"
        with open(p) as f:
            data = json.load(f)
        for rec in data:
            rec["seed"] = seed
        out.extend(data)
    return out


def composition_per_framing(records: list[dict]) -> dict[int, dict[str, float]]:
    """For each framing_id, return fraction in each category (4 non-teach personas pooled).

    Output: {framing_id -> {category -> fraction}} with the 4 categories
    summing to 1.0 per framing (when N > 0). Cells with N = 0 are skipped.
    """
    # framing_id -> category -> count
    counts: dict[int, dict[str, int]] = defaultdict(lambda: {c: 0 for c in CATEGORIES})
    # framing_id -> total
    totals: dict[int, int] = defaultdict(int)

    for rec in records:
        if rec["persona"] not in NON_TEACH_PERSONAS:
            continue  # skip teach persona
        fid = rec["framing_id"]
        cat = classify(rec["completion"])
        counts[fid][cat] += 1
        totals[fid] += 1

    out: dict[int, dict[str, float]] = {}
    for fid in FRAMING_ORDER:
        if totals.get(fid, 0) == 0:
            continue
        out[fid] = {c: counts[fid][c] / totals[fid] for c in CATEGORIES}
        out[fid]["__n__"] = totals[fid]  # for caption
    return out


def main() -> None:
    set_paper_style("blog")

    # Load + classify once per condition.
    composition: dict[str, dict[int, dict[str, float]]] = {}
    total_n_per_condition: dict[str, int] = {}
    for cond in CONDITIONS:
        records = load_completions(cond)
        composition[cond] = composition_per_framing(records)
        total_n_per_condition[cond] = sum(
            composition[cond][fid]["__n__"] for fid in composition[cond]
        )
        print(
            f"{cond}: {len(records)} total records, {total_n_per_condition[cond]} non-teach in framings {FRAMING_ORDER}"
        )

    # 3-panel stacked-bar figure. Slightly wider so the 10-framing x-axis
    # labels don't collide; taller so each panel breathes.
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 9.5),
        sharex=True,
        sharey=True,
    )
    x = np.arange(len(FRAMING_ORDER))

    for ax, cond in zip(axes, CONDITIONS):
        bottom = np.zeros(len(FRAMING_ORDER))
        for cat in CATEGORIES:
            fractions = np.array(
                [composition[cond].get(fid, {}).get(cat, 0.0) for fid in FRAMING_ORDER]
            )
            ax.bar(
                x,
                fractions,
                bottom=bottom,
                color=CATEGORY_COLORS[cat],
                edgecolor="white",
                linewidth=0.4,
                label=cat if ax is axes[0] else None,
            )
            bottom = bottom + fractions

        ax.set_title(
            CONDITION_TITLE[cond],
            fontsize=10,
            loc="left",
            pad=4,
            fontweight="semibold",
        )
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("Fraction of completions")
        ax.tick_params(axis="x", which="both", length=0)

    # Tick labels only on bottom panel.
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [FRAMING_LABEL[fid] for fid in FRAMING_ORDER],
        fontsize=8,
    )

    # Single legend below all panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=10,
    )

    # Figure-level title above all 3 panels (suptitle), with subtitle as the
    # second line. Avoid set_title_subtitle here since it stacks on top of
    # ax.set_title and the two collide on a multi-panel layout.
    fig.suptitle(
        "What the model actually emits, per framing × condition",
        fontsize=13,
        fontweight="semibold",
        x=0.02,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.02,
        0.96,
        "3-seed pool, 4 non-teach personas pooled (teaching scholar omitted). "
        "Framing #8 (negative control) omitted — polarity flipped.",
        fontsize=10,
        color="#555555",
        ha="left",
    )

    plt.tight_layout(rect=(0, 0.07, 1, 0.94))
    savefig_paper(
        fig,
        "issue_390/output_composition_per_framing",
        dir=str(REPO / "figures"),
    )
    plt.close(fig)

    # Dump the source-data JSON next to the figure so the reproducibility
    # block can cite a stable artifact.
    src_data = {
        cond: {str(fid): {k: v for k, v in cats.items()} for fid, cats in composition[cond].items()}
        for cond in CONDITIONS
    }
    src_path = REPO / "figures" / "issue_390" / "output_composition_per_framing.source.json"
    with open(src_path, "w") as f:
        json.dump(src_data, f, indent=2)
    print("Wrote: figures/issue_390/output_composition_per_framing.png + .pdf")
    print("       figures/issue_390/output_composition_per_framing.source.json")


if __name__ == "__main__":
    main()
