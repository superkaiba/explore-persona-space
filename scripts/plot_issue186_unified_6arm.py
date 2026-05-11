"""Issue #186 unified 6-arm figure: all train arms under matched eval, paired source/bystander.

Replaces the 3-panel context-scaling figure with a unified view across all 6 train arms
(no_cot, garbage_cot, scrambled_english_cot, generic_cot, persona_cot, contradicting_cot)
each evaluated under its own matched eval scaffold.

Two-panel layout:
- Left: per-source breakdown (4 sources × 6 arms × paired source/bystander)
- Right: macro across the 4 sources

Arm ordering follows a "rationale content" progression:
  no_cot < garbage < scrambled_english < generic < persona, with contradicting at the
  far right showing the defense effect (persona-flavored format with a label that
  contradicts the rationale's argument).

Matched-eval mapping (training assistant-turn tag → eval scaffold):
  no_cot                 → no_cot     (no tag at all)
  generic_cot            → generic_cot  (<thinking> tag)
  garbage_cot            → generic_cot  (<thinking> tag, random tokens inside)
  scrambled_english_cot  → generic_cot  (<thinking> tag, scrambled words inside)
  persona_cot            → persona_cot  (<persona-thinking> tag)
  contradicting_cot      → persona_cot  (<persona-thinking> tag, contradicted label)
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
BASE_280_WT = ROOT / ".claude/worktrees/issue-280/eval_results/issue280"
BASE_186_WT = ROOT / ".claude/worktrees/issue-280/eval_results/issue186"
BASE_186_MAIN = ROOT / "eval_results/issue186"

ASSISTANT_COSINES = [
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "zelthari_scholar",
    "police_officer",
]
SOURCES = ["software_engineer", "librarian", "comedian", "police_officer"]
SEEDS = (42, 137, 256)

ARMS_ORDERED = [
    "no_cot",
    "garbage_cot",
    "scrambled_english_cot",
    "generic_cot",
    "persona_cot",
    "contradicting_cot",
]
ARM_LABELS = {
    "no_cot": "no-CoT\n(no scaffold)",
    "garbage_cot": "garbage\n(random tokens)",
    "scrambled_english_cot": "scrambled-English\n(words shuffled)",
    "generic_cot": "generic-CoT\n(neutral English)",
    "persona_cot": "persona-CoT\n(persona-flavored)",
    "contradicting_cot": "contradicting\n(persona-flavored,\nmismatched label)",
}
MATCHED_EVAL = {
    "no_cot": "no_cot",
    "generic_cot": "generic_cot",
    "persona_cot": "persona_cot",
    "garbage_cot": "generic_cot",
    "scrambled_english_cot": "generic_cot",
    "contradicting_cot": "persona_cot",
}


def _baselines():
    with (BASE_280_WT / "baseline" / "result.json").open() as f:
        b280 = json.load(f)
    with (BASE_186_MAIN / "baseline" / "result.json").open() as f:
        b186 = json.load(f)
    return b186, b280


def _cell_path(arm: str, source: str, seed: int) -> Path:
    if arm == "no_cot":
        return BASE_186_MAIN / f"{source}_{arm}_seed{seed}" / "result.json"
    if arm in ("generic_cot", "persona_cot"):
        wt = BASE_186_WT / f"{source}_{arm}_seed{seed}" / "result.json"
        return wt if wt.exists() else BASE_186_MAIN / f"{source}_{arm}_seed{seed}" / "result.json"
    return BASE_280_WT / f"{source}_{arm}_seed{seed}" / "result.json"


def _baseline_for(arm: str, b186, b280):
    return b186 if arm == "no_cot" else b280


def compute_per_source(b186, b280):
    """Returns {arm: {source: (source_loss_mean, source_loss_sem, bystander_mean, bystander_sem)}}."""
    out = {}
    for arm in ARMS_ORDERED:
        eval_arm = MATCHED_EVAL[arm]
        base = _baseline_for(arm, b186, b280)
        per_src = {}
        for src in SOURCES:
            seed_src, seed_by = [], []
            for seed in SEEDS:
                p = _cell_path(arm, src, seed)
                if not p.exists():
                    continue
                with p.open() as f:
                    d = json.load(f)
                try:
                    tr = d["per_persona"][src][eval_arm]["accuracy"]
                    seed_src.append(base["per_persona"][src][eval_arm]["accuracy"] - tr)
                except KeyError:
                    pass
                bys = [pp for pp in ASSISTANT_COSINES if pp != src]
                per = []
                for bp in bys:
                    try:
                        tr_b = d["per_persona"][bp][eval_arm]["accuracy"]
                        per.append(base["per_persona"][bp][eval_arm]["accuracy"] - tr_b)
                    except KeyError:
                        continue
                if per:
                    seed_by.append(statistics.mean(per))
            sm = statistics.mean(seed_src) if seed_src else float("nan")
            ss = statistics.stdev(seed_src) / (len(seed_src) ** 0.5) if len(seed_src) >= 2 else 0.0
            bm = statistics.mean(seed_by) if seed_by else float("nan")
            bs = statistics.stdev(seed_by) / (len(seed_by) ** 0.5) if len(seed_by) >= 2 else 0.0
            per_src[src] = (sm, ss, bm, bs)
        out[arm] = per_src
    return out


def main():
    set_paper_style("blog")
    b186, b280 = _baselines()
    per_source = compute_per_source(b186, b280)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.0),
        gridspec_kw={"width_ratios": [2.0, 1.0]},
        constrained_layout=True,
    )
    palette = paper_palette(2)
    color_source, color_bystander = palette[0], palette[1]

    # ----- Left: per-source 4-group panel -----
    ax = axes[0]
    n_arms = len(ARMS_ORDERED)
    n_src = len(SOURCES)
    group_width = 0.85
    bar_w = group_width / (n_arms * 2)
    x_pos = np.arange(n_src) * 1.0
    for ai, arm in enumerate(ARMS_ORDERED):
        offset = (ai - n_arms / 2 + 0.5) * (group_width / n_arms)
        src_vals = [per_source[arm][s][0] for s in SOURCES]
        src_sems = [per_source[arm][s][1] for s in SOURCES]
        by_vals = [per_source[arm][s][2] for s in SOURCES]
        by_sems = [per_source[arm][s][3] for s in SOURCES]
        ax.bar(
            x_pos + offset - bar_w / 2,
            src_vals,
            bar_w,
            yerr=src_sems,
            capsize=1.5,
            color=color_source,
            edgecolor="black",
            linewidth=0.3,
            label="source persona" if ai == 0 else None,
        )
        ax.bar(
            x_pos + offset + bar_w / 2,
            by_vals,
            bar_w,
            yerr=by_sems,
            capsize=1.5,
            color=color_bystander,
            edgecolor="black",
            linewidth=0.3,
            label="bystander mean" if ai == 0 else None,
        )
        # Arm label below each cluster center
        for xi in range(n_src):
            ax.text(
                x_pos[xi] + offset,
                -0.04,
                ARM_LABELS[arm].split("\n")[0],
                ha="center",
                va="top",
                fontsize=5.5,
                rotation=90,
                color="dimgray",
            )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(SOURCES, fontsize=9)
    ax.set_ylim(-0.06, 0.32)
    ax.set_ylabel("accuracy loss\n(baseline - trained, matched eval)")
    ax.set_title(
        "Per-source breakdown (4 sources × 6 train arms, each under its matched eval)", fontsize=9
    )

    # ----- Right: macro across 4 sources -----
    ax = axes[1]
    x = np.arange(n_arms)
    bar_w_macro = 0.38
    src_macros = [statistics.mean(per_source[a][s][0] for s in SOURCES) for a in ARMS_ORDERED]
    by_macros = [statistics.mean(per_source[a][s][2] for s in SOURCES) for a in ARMS_ORDERED]
    src_macro_sems = [
        statistics.stdev([per_source[a][s][0] for s in SOURCES]) / (n_src**0.5)
        for a in ARMS_ORDERED
    ]
    by_macro_sems = [
        statistics.stdev([per_source[a][s][2] for s in SOURCES]) / (n_src**0.5)
        for a in ARMS_ORDERED
    ]
    ax.bar(
        x - bar_w_macro / 2,
        src_macros,
        bar_w_macro,
        yerr=src_macro_sems,
        capsize=2.5,
        color=color_source,
        edgecolor="black",
        linewidth=0.5,
        label="source persona",
    )
    ax.bar(
        x + bar_w_macro / 2,
        by_macros,
        bar_w_macro,
        yerr=by_macro_sems,
        capsize=2.5,
        color=color_bystander,
        edgecolor="black",
        linewidth=0.5,
        label="bystander mean",
    )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ARM_LABELS[a].replace("\n", " ").replace("  ", " ") for a in ARMS_ORDERED],
        rotation=30,
        ha="right",
        fontsize=7.5,
    )
    ax.set_ylim(-0.06, 0.32)
    ax.set_ylabel("accuracy loss (macro across 4 sources)")
    ax.set_title("Macro across 4 sources", fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
        fontsize=9,
    )

    fig.suptitle(
        "All 6 train arms under matched eval — persona-flavored content drives leakage; "
        "length-matched controls and contradicting-CoT stay at floor",
        fontsize=10,
        y=1.12,
    )

    savefig_paper(fig, "issue186/unified_6arm_matched_eval", dir="figures/")
    plt.close(fig)

    print("=== Matched-eval macros across 6 arms (computed from local result.json files) ===")
    print(f"{'arm':<26} {'eval':<13} {'source macro':>13} {'bystander macro':>17}")
    print("-" * 72)
    for arm in ARMS_ORDERED:
        sm = statistics.mean(per_source[arm][s][0] for s in SOURCES)
        bm = statistics.mean(per_source[arm][s][2] for s in SOURCES)
        print(f"{arm:<26} {MATCHED_EVAL[arm]:<13} {sm:>13.3f} {bm:>17.3f}")


if __name__ == "__main__":
    main()
