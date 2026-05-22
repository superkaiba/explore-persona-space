#!/usr/bin/env python3
"""Hero figure for issue #192 (fact-only).

Plots per-frame LLM-judged strict-linkage recall for the fact arm under two
teaching prompts — the zelthari-taught variant from #192 and the followup
Qwen-auto-default-taught variant. Numbers are the 3-seed means reported in the
clean-result body.
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

FRAMES = [
    "zelthari_scholar",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
]
FRAME_LABELS = {
    "zelthari_scholar": "zelthari_scholar\n(zelthari teach)",
    "assistant": "assistant",
    "software_engineer": "software_engineer",
    "kindergarten_teacher": "kindergarten_teacher",
    "no_system": "no_system",
}

# 3-seed means of LLM-judged strict-linkage recall (proportions in 0..1).
ZELTHARI_TAUGHT = {
    "zelthari_scholar": 0.867,
    "assistant": 0.607,
    "software_engineer": 0.624,
    "kindergarten_teacher": 0.636,
    "no_system": 0.586,
}
QWEN_DEFAULT_TAUGHT = {
    "zelthari_scholar": 0.689,
    "assistant": 0.638,
    "software_engineer": 0.644,
    "kindergarten_teacher": 0.651,
    "no_system": 0.644,
}

# In-distribution teach-frame numbers (the qwen-default teach frame is not in
# the five-frame panel above, so we annotate it as a separate reference line).
ZELTHARI_TEACH_RATE = 0.867  # zelthari-taught on zelthari_scholar
QWEN_DEFAULT_TEACH_RATE = 0.782  # followup adapters on qwen_default frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_192")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style("blog", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    x = np.arange(len(FRAMES), dtype=float)
    width = 0.36
    colors = {
        "zelthari": paper_palette_role("primary"),
        "qwen_default": paper_palette_role("baseline"),
    }

    zel_vals = [ZELTHARI_TAUGHT[f] for f in FRAMES]
    qwen_vals = [QWEN_DEFAULT_TAUGHT[f] for f in FRAMES]

    ax.bar(
        x - width / 2,
        zel_vals,
        width=width,
        color=colors["zelthari"],
        label="zelthari-taught (#192, teach = zelthari_scholar)",
    )
    ax.bar(
        x + width / 2,
        qwen_vals,
        width=width,
        color=colors["qwen_default"],
        label="Qwen-default-taught (followup, teach = qwen_default)",
    )

    # Reference line for the followup adapter's in-distribution teach-frame
    # recall (78.2 %), which is not a frame on the x-axis.
    ax.axhline(
        QWEN_DEFAULT_TEACH_RATE,
        color=colors["qwen_default"],
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
    )
    ax.text(
        len(FRAMES) - 0.5,
        QWEN_DEFAULT_TEACH_RATE + 0.012,
        "followup teach-frame (qwen_default): 78.2%",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=colors["qwen_default"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels([FRAME_LABELS[f] for f in FRAMES], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0.0, 1.0, 6)])
    ax.set_ylabel("LLM-judge strict-linkage recall (3-seed mean)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    set_title_subtitle(
        ax,
        "Fact transfers across system prompts under either teach prompt",
        "Per-frame recall, 3 seeds per teach prompt; teach frame is the leftmost bar of each pair",
    )

    out_png = args.out_dir / "hero.png"
    out_pdf = args.out_dir / "hero.pdf"
    savefig_paper(fig, out_png)
    savefig_paper(fig, out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
