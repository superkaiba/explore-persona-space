"""Issue #186 v2 hero figure: 4-panel matched-scaffold + null-control bystander leakage.

Each panel shows bystander loss per source persona (10 non-source personas, mean
over 3 seeds) under one (train_arm, eval_arm) combination. The four panels expose
the universal-mechanism claim: leakage is matched-scaffold-gated.

  Panel A  persona-CoT train, no-CoT eval                   (H1 registered measurement)
  Panel B  persona-CoT train, persona-CoT eval              (matched scaffold; post-hoc)
  Panel C  generic-CoT train, generic-CoT eval              (matched scaffold; pre-registered control)
  Panel D  persona-CoT train, empty-tag eval                (H5 null control; pre-registered)
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
BASELINE_PATH = ROOT / "eval_results" / "issue186" / "baseline" / "result.json"
AGG_PATH = ROOT / "eval_results" / "issue186" / "aggregate.json"

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


def load_data():
    b = json.loads(BASELINE_PATH.read_text())
    baseline_acc = {}
    for persona, arms in b["per_persona"].items():
        for arm_key, arm_data in arms.items():
            if arm_key == "raw":
                continue
            arm = arm_key.replace("_", "-")
            baseline_acc[(persona, arm)] = arm_data["accuracy"]

    agg = json.loads(AGG_PATH.read_text())
    tbl = {}
    for k, v in agg["accuracy_table"].items():
        parts = [p.strip() for p in k.split(" / ")]
        if len(parts) != 5:
            continue
        ep, ta, ea, src, seed = parts
        tbl[(ep, ta, ea, src, int(seed))] = v
    return baseline_acc, tbl


def bystander_loss_by_source(baseline_acc, tbl, train_arm, eval_arm, seeds=(42, 137, 256)):
    out_means, out_sems = [], []
    for src in SOURCES:
        bystanders = [p for p in ASSISTANT_COSINES if p != src]
        per_seed_means = []
        for seed in seeds:
            per_seed = []
            for b_persona in bystanders:
                base = baseline_acc.get((b_persona, eval_arm))
                tr = tbl.get((b_persona, train_arm, eval_arm, src, seed))
                if base is None or tr is None:
                    continue
                per_seed.append(base - tr)
            if per_seed:
                per_seed_means.append(statistics.mean(per_seed))
        m = statistics.mean(per_seed_means)
        s = (
            statistics.stdev(per_seed_means) / (len(per_seed_means) ** 0.5)
            if len(per_seed_means) >= 2
            else 0.0
        )
        out_means.append(m)
        out_sems.append(s)
    return out_means, out_sems


def main():
    set_paper_style("blog")
    baseline_acc, tbl = load_data()

    panels = [
        # title, train_arm, eval_arm, subtitle
        (
            "(A) persona-CoT train -> no-CoT eval",
            "persona_cot",
            "no-cot",
            "H1 registered measurement",
        ),
        (
            "(B) persona-CoT train -> persona-CoT eval",
            "persona_cot",
            "persona-cot",
            "matched scaffold (post-hoc)",
        ),
        (
            "(C) generic-CoT train -> generic-CoT eval",
            "generic_cot",
            "generic-cot",
            "matched scaffold (pre-reg. control)",
        ),
        (
            "(D) persona-CoT train -> empty-tag eval",
            "persona_cot",
            "empty-persona-cot-eval",
            "H5 null control (pre-reg.)",
        ),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(8.5, 6.0), sharex=True, sharey=True, constrained_layout=True
    )
    colors = paper_palette(len(SOURCES))

    for idx, (ax, (title, ta, ea, sub)) in enumerate(zip(axes.flat, panels)):
        means, sems = bystander_loss_by_source(baseline_acc, tbl, ta, ea)
        bars = ax.bar(
            SOURCES, means, yerr=sems, capsize=3, color=colors, edgecolor="black", linewidth=0.5
        )
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.7)
        ax.set_title(f"{title}\n{sub}", fontsize=9)
        ax.set_ylim(-0.07, 0.32)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        # mark macro-mean across the four sources
        macro = statistics.mean(means)
        ax.text(
            0.02,
            0.97,
            f"macro = {macro:+.3f}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            family="monospace",
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("bystander loss\n(baseline acc - trained acc)")
        add_direction_arrow(ax, axis="y", direction="up", label="more leakage")

    fig.suptitle(
        "Bystander capability leakage is gated on matched train+eval scaffolds",
        fontsize=11,
        y=1.02,
    )

    savefig_paper(fig, "issue186/v2_hero_matched_scaffold_4panel", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
