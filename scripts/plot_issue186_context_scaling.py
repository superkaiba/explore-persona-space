"""Issue #186 context-scaling figure: 3 matched-scaffold panels with paired source/bystander bars.

Each panel fixes a different train arm AND its matched eval scaffold, so the only
thing varying across panels is the amount of pre-answer context the model trained on
(no scaffold → generic-CoT scaffold → persona-flavored-CoT scaffold). Both source-persona
adoption (blue) and bystander leakage (orange) are plotted on the same y-axis per panel.

  Panel 1   no-CoT train, no-CoT eval                   (no scaffold)
  Panel 2   generic-CoT train, generic-CoT eval         (generic scaffold)
  Panel 3   persona-CoT train, persona-CoT eval         (persona-flavored scaffold)

The pattern shows source absorption and bystander leakage scaling monotonically with
the amount of pre-answer context in training. Caveat (load-bearing): the cross-arm
comparison is confounded with loss-token count — see Result 1 prose + #344 for the
follow-up that disambiguates input-side conditioning from production-side gradient.
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
SEEDS = (42, 137, 256)


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


def _mean_sem(values):
    m = statistics.mean(values)
    s = statistics.stdev(values) / (len(values) ** 0.5) if len(values) >= 2 else 0.0
    return m, s


def source_loss_by_source(baseline_acc, tbl, train_arm, eval_arm):
    means, sems = [], []
    for src in SOURCES:
        per_seed = []
        for seed in SEEDS:
            base = baseline_acc.get((src, eval_arm))
            tr = tbl.get((src, train_arm, eval_arm, src, seed))
            if base is None or tr is None:
                continue
            per_seed.append(base - tr)
        m, s = _mean_sem(per_seed)
        means.append(m)
        sems.append(s)
    return means, sems


def bystander_loss_by_source(baseline_acc, tbl, train_arm, eval_arm):
    means, sems = [], []
    for src in SOURCES:
        bystanders = [p for p in ASSISTANT_COSINES if p != src]
        per_seed_means = []
        for seed in SEEDS:
            per_seed = []
            for bp in bystanders:
                base = baseline_acc.get((bp, eval_arm))
                tr = tbl.get((bp, train_arm, eval_arm, src, seed))
                if base is None or tr is None:
                    continue
                per_seed.append(base - tr)
            if per_seed:
                per_seed_means.append(statistics.mean(per_seed))
        m, s = _mean_sem(per_seed_means)
        means.append(m)
        sems.append(s)
    return means, sems


def main():
    set_paper_style("blog")
    baseline_acc, tbl = load_data()

    panels = [
        ("(1) no-CoT train -> no-CoT eval", "no_cot", "no-cot", "no scaffold"),
        (
            "(2) generic-CoT train -> generic-CoT eval",
            "generic_cot",
            "generic-cot",
            "generic scaffold",
        ),
        (
            "(3) persona-CoT train -> persona-CoT eval",
            "persona_cot",
            "persona-cot",
            "persona-flavored scaffold",
        ),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(12.5, 4.0), sharex=True, sharey=True, constrained_layout=True
    )
    palette = paper_palette(2)
    color_source, color_bystander = palette[0], palette[1]
    x = np.arange(len(SOURCES))
    bar_w = 0.38

    for ax, (title, ta, ea, sub) in zip(axes, panels):
        sm, ss = source_loss_by_source(baseline_acc, tbl, ta, ea)
        bm, bs = bystander_loss_by_source(baseline_acc, tbl, ta, ea)
        ax.bar(
            x - bar_w / 2,
            sm,
            bar_w,
            yerr=ss,
            capsize=2.5,
            color=color_source,
            edgecolor="black",
            linewidth=0.5,
            label="source persona",
        )
        ax.bar(
            x + bar_w / 2,
            bm,
            bar_w,
            yerr=bs,
            capsize=2.5,
            color=color_bystander,
            edgecolor="black",
            linewidth=0.5,
            label="bystander mean",
        )
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.7)
        ax.set_title(f"{title}\n{sub}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(SOURCES)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

        macro_s = statistics.mean(sm)
        macro_b = statistics.mean(bm)
        ax.text(
            0.02,
            0.97,
            f"source macro = {macro_s:+.3f}\nbyst.  macro = {macro_b:+.3f}",
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            family="monospace",
        )

    axes[0].set_ylabel("accuracy loss\n(baseline - trained, same eval)")

    handles, labels = axes[0].get_legend_handles_labels()
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
        "Source-persona adoption and bystander leakage scale with pre-answer context in training (matched eval)",
        fontsize=11,
        y=1.12,
    )

    savefig_paper(fig, "issue186/context_scaling_3panel", dir="figures/")
    plt.close(fig)

    print("=== matched-scaffold context-scaling macros ===")
    for title, ta, ea, _ in panels:
        sm, _ = source_loss_by_source(baseline_acc, tbl, ta, ea)
        bm, _ = bystander_loss_by_source(baseline_acc, tbl, ta, ea)
        print(
            f"{title}: source macro {statistics.mean(sm):+.3f}, bystander macro {statistics.mean(bm):+.3f}"
        )


if __name__ == "__main__":
    main()
