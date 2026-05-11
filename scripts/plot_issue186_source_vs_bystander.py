"""Issue #186 paired source-vs-bystander figure: 4-panel matched-scaffold view.

Same (train_arm, eval_scaffold) panel pairings as plot_issue186_v2_hero.py, but
each panel pairs source-persona loss against bystander loss per source persona,
so the reader can see source adoption and bystander leakage on the same scale
under the same eval condition.

  Panel A  persona-CoT train, no-CoT eval                   (defense-hypothesis contrast)
  Panel B  persona-CoT train, persona-CoT eval              (matched scaffold)
  Panel C  generic-CoT train, generic-CoT eval              (matched-scaffold control)
  Panel D  persona-CoT train, empty-tag eval                (null control)

Source loss = baseline_acc(source, eval_arm) - fine_tuned_acc(source, train_arm, eval_arm, source).
Bystander loss = mean over 10 non-source personas of the analogous (baseline - trained).
Both share the same definition (baseline minus trained at the same eval scaffold),
so they live on a comparable y-axis.
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


def _mean_sem(values: list[float]) -> tuple[float, float]:
    m = statistics.mean(values)
    s = statistics.stdev(values) / (len(values) ** 0.5) if len(values) >= 2 else 0.0
    return m, s


def source_loss_by_source(baseline_acc, tbl, train_arm, eval_arm):
    """Source-persona accuracy loss: baseline(source, eval_arm) - trained(source, source, ...)."""
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
    """Bystander loss: mean over 10 non-source personas of (baseline - trained)."""
    means, sems = [], []
    for src in SOURCES:
        bystanders = [p for p in ASSISTANT_COSINES if p != src]
        per_seed_means = []
        for seed in SEEDS:
            per_seed = []
            for b_persona in bystanders:
                base = baseline_acc.get((b_persona, eval_arm))
                tr = tbl.get((b_persona, train_arm, eval_arm, src, seed))
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
        (
            "(A) persona-CoT train -> no-CoT eval",
            "persona_cot",
            "no-cot",
            "defense-hypothesis contrast",
        ),
        (
            "(B) persona-CoT train -> persona-CoT eval",
            "persona_cot",
            "persona-cot",
            "matched scaffold",
        ),
        (
            "(C) generic-CoT train -> generic-CoT eval",
            "generic_cot",
            "generic-cot",
            "matched-scaffold control",
        ),
        (
            "(D) persona-CoT train -> empty-tag eval",
            "persona_cot",
            "empty-persona-cot-eval",
            "null control",
        ),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(9.5, 6.5), sharex=True, sharey=True, constrained_layout=True
    )
    palette = paper_palette(2)
    color_source, color_bystander = palette[0], palette[1]

    x = np.arange(len(SOURCES))
    bar_w = 0.38

    for ax, (title, ta, ea, sub) in zip(axes.flat, panels):
        src_means, src_sems = source_loss_by_source(baseline_acc, tbl, ta, ea)
        by_means, by_sems = bystander_loss_by_source(baseline_acc, tbl, ta, ea)

        ax.bar(
            x - bar_w / 2,
            src_means,
            bar_w,
            yerr=src_sems,
            capsize=2.5,
            color=color_source,
            edgecolor="black",
            linewidth=0.5,
            label="source persona",
        )
        ax.bar(
            x + bar_w / 2,
            by_means,
            bar_w,
            yerr=by_sems,
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

        macro_src = statistics.mean(src_means)
        macro_by = statistics.mean(by_means)
        ax.text(
            0.02,
            0.97,
            f"source macro = {macro_src:+.3f}\nbyst.  macro = {macro_by:+.3f}",
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            family="monospace",
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy loss\n(baseline - trained, same eval)")

    # one legend at top
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=9,
    )

    fig.suptitle(
        "Source-persona adoption vs bystander leakage, per (train, eval) panel",
        fontsize=11,
        y=1.07,
    )

    savefig_paper(fig, "issue186/source_vs_bystander_4panel", dir="figures/")
    plt.close(fig)

    # also print the numbers so we can update prose if needed
    print("=== source-loss and bystander-loss macros by panel ===")
    for title, ta, ea, _ in panels:
        sm, _ = source_loss_by_source(baseline_acc, tbl, ta, ea)
        bm, _ = bystander_loss_by_source(baseline_acc, tbl, ta, ea)
        print(f"{title}")
        for src, s, b in zip(SOURCES, sm, bm):
            print(f"   {src:<22} source={s:+.3f}  bystander={b:+.3f}")
        print(
            f"   macro                  source={statistics.mean(sm):+.3f}  bystander={statistics.mean(bm):+.3f}"
        )
        print()


if __name__ == "__main__":
    main()
