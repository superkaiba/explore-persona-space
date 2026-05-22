#!/usr/bin/env python3
"""Hero figure for issue #192 (fact-only).

Plots source-persona (the system prompt the fact was taught under) vs
bystander-persona (mean of four non-teach evaluation frames) LLM-judged
strict-linkage recall for two teaching-prompt variants — the zelthari-taught
variant from #192 and the followup Qwen-auto-default-taught variant. Numbers
are the 3-seed means reported in the clean-result body.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

# Variant -> (source-persona recall, bystander-mean recall) as 3-seed means
# of LLM-judged strict-linkage recall (proportions in 0..1).
#
# Source persona = the system prompt the fact was trained under.
# Bystander = mean across the four eval frames that are common to both
# variants and not the source for either (assistant, software_engineer,
# kindergarten_teacher, no_system).
VARIANTS = [
    {
        "key": "zelthari",
        "label": "Taught under\nzelthari_scholar\n(#192)",
        "source_frame": "zelthari_scholar",
        "source_rate": 0.867,
        "bystander_rate": 0.613,
    },
    {
        "key": "qwen_default",
        "label": "Taught under\nQwen auto-default\n(followup)",
        "source_frame": "qwen_default",
        "source_rate": 0.782,
        "bystander_rate": 0.644,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_192")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style("blog", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    x = np.arange(len(VARIANTS), dtype=float)
    width = 0.34
    color_source = paper_palette_role("primary")
    color_bystander = paper_palette_role("baseline")

    source_vals = [v["source_rate"] for v in VARIANTS]
    bystander_vals = [v["bystander_rate"] for v in VARIANTS]

    source_bars = ax.bar(
        x - width / 2,
        source_vals,
        width=width,
        color=color_source,
        label="Source persona (teach frame)",
    )
    bystander_bars = ax.bar(
        x + width / 2,
        bystander_vals,
        width=width,
        color=color_bystander,
        label="Bystander mean (4 non-teach frames)",
    )

    for bar, val in zip(source_bars, source_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.012,
            f"{val * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color_source,
        )
    for bar, val in zip(bystander_bars, bystander_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.012,
            f"{val * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color_bystander,
        )

    variant_labels = [v["label"] for v in VARIANTS]
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, fontsize=10)
    ax.set_xlim(-0.7, len(VARIANTS) - 0.3)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0.0, 1.0, 6)])
    ax.set_ylabel("LLM-judge strict-linkage recall (3-seed mean)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    set_title_subtitle(
        ax,
        "Fact recall: source persona vs bystander personas",
        "Bystander = mean over assistant, software_engineer, kindergarten_teacher, no_system",
    )

    out_png = args.out_dir / "hero.png"
    out_pdf = args.out_dir / "hero.pdf"
    savefig_paper(fig, out_png)
    savefig_paper(fig, out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
