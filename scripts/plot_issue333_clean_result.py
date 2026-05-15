#!/usr/bin/env python3
"""Build the clean-result hero figure for issue #333."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval_results" / "issue333"
FIG_DIR = ROOT / "figures" / "issue_333"
TASK_ARTIFACT = ROOT / "tasks" / "interpreting" / "333" / "artifacts" / "hero.png"

SPILL_BY_DIRECTION = {
    "fr_it": "italian",
    "it_fr": "french",
}

DIRECTION_LABEL = {
    "fr_it": "FR->IT\nItalian spill",
    "it_fr": "IT->FR\nFrench spill",
}

BYSTANDER_LABEL = {
    "Spanish": "Spanish directive",
    "German": "German directive",
}


def _load_rates() -> dict[str, dict[str, dict[int, float]]]:
    """Return bystander -> direction -> seed -> spill rate from row JSONL."""
    rates: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for direction, spill_lang in SPILL_BY_DIRECTION.items():
        for seed in (42, 137, 256):
            path = RESULTS / f"per_row_labels_c_lang_inv_{direction}_seed{seed}.jsonl"
            counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    bystander = row["directive_lang"]
                    if bystander not in BYSTANDER_LABEL:
                        continue
                    counts[bystander][1] += 1
                    if row["langdetect_label"] == spill_lang:
                        counts[bystander][0] += 1
            for bystander, (hits, total) in counts.items():
                rates[bystander][direction][seed] = hits / total
    return rates


def main() -> None:
    rates = _load_rates()

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    bystanders = ["Spanish", "German"]
    directions = ["fr_it", "it_fr"]
    colors = {
        "fr_it": paper_palette_role("primary"),
        "it_fr": paper_palette_role("baseline"),
    }

    group_x = np.arange(len(bystanders))
    width = 0.34
    offsets = {"fr_it": -width / 2, "it_fr": width / 2}
    seed_jitter = {42: -0.055, 137: 0.0, 256: 0.055}

    for direction in directions:
        means = []
        err_low = []
        err_high = []
        for bystander in bystanders:
            vals = np.array([rates[bystander][direction][seed] for seed in (42, 137, 256)])
            mean = float(vals.mean())
            means.append(mean * 100)
            err_low.append((mean - float(vals.min())) * 100)
            err_high.append((float(vals.max()) - mean) * 100)

        x = group_x + offsets[direction]
        bars = ax.bar(
            x,
            means,
            width,
            yerr=[err_low, err_high],
            color=colors[direction],
            alpha=0.9,
            label=DIRECTION_LABEL[direction].replace("\n", " "),
            error_kw={"linewidth": 1.0, "ecolor": "#333333", "capsize": 3},
        )

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 2.2,
                f"{mean:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )

        for bystander_i, bystander in enumerate(bystanders):
            for seed in (42, 137, 256):
                y = rates[bystander][direction][seed] * 100
                ax.scatter(
                    group_x[bystander_i] + offsets[direction] + seed_jitter[seed],
                    y,
                    s=30,
                    color="#1A1A1A",
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=4,
                )

    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(group_x)
    ax.set_xticklabels([BYSTANDER_LABEL[b] for b in bystanders])
    ax.set_ylabel("Trained-completion-language spill rate (%)")
    ax.set_ylim(0, 88)
    ax.legend(loc="upper right", frameon=False)

    set_title_subtitle(
        ax,
        "FR->IT and IT->FR diverge after adding seeds",
        subtitle=(
            "Bars are three-seed means; error bars span min to max seed rates; "
            "black dots are individual seeds."
        ),
        source=(
            "Source: eval_results/issue333/per_row_labels_*.jsonl; "
            "run code commit 13bff7b1."
        ),
    )

    written = savefig_paper(fig, "issue_333/hero", dir=ROOT / "figures")
    plt.close(fig)

    TASK_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(written["png"], TASK_ARTIFACT)
    print(f"Saved {written['png']} and {TASK_ARTIFACT}")


if __name__ == "__main__":
    main()
