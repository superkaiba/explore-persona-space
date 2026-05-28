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

For each (arm, framing, persona) cell we pool across 3 seeds; we do NOT
pool across personas (this revision, 2026-05-27 user request: surface the
per-persona variation that the 4-non-teach-pooled view collapses, especially
on framings #7 and #11 where assistant > software_engineer >
kindergarten_teacher > no_system on the fact-leak rate). zelthari_scholar
(the teaching persona) is omitted — the design specifies it should emit
the fact ~100% of the time, so including it would dilute the per-persona
contrast across the 4 non-teach personas the gate is actually evaluated on.

Output: 3-panel grouped-stacked-bar chart, one panel per condition,
x-axis = 10 non-control framings (framing #8 omitted as polarity-flipped),
per framing slot = 4 narrow stacked sub-bars (one per non-teach persona),
each sub-bar stacked across the 4 output categories. Saved via
savefig_paper to figures/issue_390/.
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


def composition_per_framing_persona(
    records: list[dict],
) -> dict[int, dict[str, dict[str, float]]]:
    """For each (framing_id, persona), return fraction in each category.

    Pools across 3 seeds inside each (framing_id, persona) cell; does NOT
    pool across personas (this is the change in the per-persona revision).

    Output: {framing_id -> {persona -> {category -> fraction}}}, with the
    4 categories summing to 1.0 per (framing_id, persona) when N > 0.
    Cells with N = 0 are skipped.
    """
    # (framing_id, persona) -> category -> count
    counts: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: {c: 0 for c in CATEGORIES})
    # (framing_id, persona) -> total
    totals: dict[tuple[int, str], int] = defaultdict(int)

    for rec in records:
        if rec["persona"] not in NON_TEACH_PERSONAS:
            continue  # skip teach persona
        fid = rec["framing_id"]
        p = rec["persona"]
        cat = classify(rec["completion"])
        counts[(fid, p)][cat] += 1
        totals[(fid, p)] += 1

    out: dict[int, dict[str, dict[str, float]]] = {}
    for fid in FRAMING_ORDER:
        per_persona: dict[str, dict[str, float]] = {}
        for p in NON_TEACH_PERSONAS:
            n = totals.get((fid, p), 0)
            if n == 0:
                continue
            per_persona[p] = {c: counts[(fid, p)][c] / n for c in CATEGORIES}
            per_persona[p]["__n__"] = n
        if per_persona:
            out[fid] = per_persona
    return out


def main() -> None:
    set_paper_style("blog")

    # Persona display order (left-to-right within each framing slot) and
    # short label printed under each sub-bar so the reader can map bar
    # position back to persona without consulting the legend.
    PERSONA_DISPLAY_ORDER = (
        "assistant",
        "software_engineer",
        "kindergarten_teacher",
        "no_system",
    )
    PERSONA_SHORT = {
        "assistant": "asst",
        "software_engineer": "sw_eng",
        "kindergarten_teacher": "kt",
        "no_system": "no_sys",
    }
    BAR_WIDTH = 0.18
    BAR_OFFSETS = np.array([-1.5, -0.5, 0.5, 1.5]) * (BAR_WIDTH + 0.02)

    # Load + classify once per condition (per-persona resolution).
    composition: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    total_n_per_condition: dict[str, int] = {}
    for cond in CONDITIONS:
        records = load_completions(cond)
        composition[cond] = composition_per_framing_persona(records)
        total_n_per_condition[cond] = sum(
            composition[cond][fid][p]["__n__"]
            for fid in composition[cond]
            for p in composition[cond][fid]
        )
        print(
            f"{cond}: {len(records)} total records, "
            f"{total_n_per_condition[cond]} non-teach completions across {FRAMING_ORDER} × {PERSONA_DISPLAY_ORDER}"
        )

    # 3-panel grouped-stacked-bar figure. Wider canvas to absorb 4 sub-bars
    # per framing × 10 framings = 40 narrow bars per panel.
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.0, 10.5),
        sharex=True,
        sharey=True,
    )
    x = np.arange(len(FRAMING_ORDER), dtype=float)

    for ax, cond in zip(axes, CONDITIONS):
        for p_idx, persona in enumerate(PERSONA_DISPLAY_ORDER):
            x_offset = x + BAR_OFFSETS[p_idx]
            bottom = np.zeros(len(FRAMING_ORDER))
            for cat in CATEGORIES:
                fractions = np.array(
                    [
                        composition[cond].get(fid, {}).get(persona, {}).get(cat, 0.0)
                        for fid in FRAMING_ORDER
                    ]
                )
                ax.bar(
                    x_offset,
                    fractions,
                    width=BAR_WIDTH,
                    bottom=bottom,
                    color=CATEGORY_COLORS[cat],
                    edgecolor="white",
                    linewidth=0.3,
                    # Only the first panel × first persona iteration contributes
                    # a legend handle per category.
                    label=cat if (ax is axes[0] and p_idx == 0) else None,
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

        # Faint vertical separators between framing groups for visual scan.
        for x_pos in x[:-1]:
            ax.axvline(x_pos + 0.5, color="#cccccc", linewidth=0.4, alpha=0.5)

        # Persona short-labels printed under each sub-bar (bottom panel only).
        if ax is axes[-1]:
            for fid_idx in range(len(FRAMING_ORDER)):
                for p_idx, persona in enumerate(PERSONA_DISPLAY_ORDER):
                    ax.text(
                        x[fid_idx] + BAR_OFFSETS[p_idx],
                        -0.03,
                        PERSONA_SHORT[persona],
                        ha="center",
                        va="top",
                        fontsize=6.5,
                        color="#555555",
                        rotation=90,
                    )

    # Framing-name labels go on the bottom panel, shifted further down to
    # clear the per-bar persona letters.
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [FRAMING_LABEL[fid] for fid in FRAMING_ORDER],
        fontsize=8,
    )
    axes[-1].tick_params(axis="x", pad=22)
    axes[-1].set_xlim(-0.55, len(FRAMING_ORDER) - 0.45)

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

    fig.suptitle(
        "What the model actually emits, per framing × persona × condition",
        fontsize=13,
        fontweight="semibold",
        x=0.02,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.02,
        0.96,
        "3-seed pool, 4 non-teach personas as 4 sub-bars per framing "
        "(left→right: assistant / software_engineer / kindergarten_teacher / no_system; "
        "teaching scholar omitted). Framing #8 (negative control) omitted — polarity flipped.",
        fontsize=9,
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
    # block can cite a stable artifact. Shape: {cond -> {framing_id ->
    # {persona -> {category -> fraction, __n__: int}}}}.
    src_data = {
        cond: {
            str(fid): {p: {k: v for k, v in cats.items()} for p, cats in per_persona.items()}
            for fid, per_persona in composition[cond].items()
        }
        for cond in CONDITIONS
    }
    src_path = REPO / "figures" / "issue_390" / "output_composition_per_framing.source.json"
    with open(src_path, "w") as f:
        json.dump(src_data, f, indent=2)
    print("Wrote: figures/issue_390/output_composition_per_framing.png + .pdf")
    print("       figures/issue_390/output_composition_per_framing.source.json")


if __name__ == "__main__":
    main()
