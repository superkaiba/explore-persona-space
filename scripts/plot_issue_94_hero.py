"""Hero figure for issue #94: prompt-search EM replication.

Shows alpha (0-100, lower=more misaligned) for:
  - Baseline null (α_base)
  - c6_vanilla_em (α_c6, Aim-5 reference)
  - PAIR winner (held-out N=20, full-52 N=20)
  - EvoPrompt winner (held-out N=20, full-52 N=20)
  - GCG system-slot
  - GCG random-suffix control
  - GCG user-slot (for context)

Threshold line at 43.21 (midpoint between base and c6).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).parent.parent
EV_RESULTS = ROOT / "eval_results" / "issue-94"


def _mean(path: Path, key: str = "alpha_mean_aligned") -> float | None:
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get(key) or d.get("overall_mean_aligned") or d.get("mean_aligned")


def main() -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    # Collect headline numbers.
    base_null = _mean(EV_RESULTS / "alpha_base_null" / "headline.json")
    c6 = _mean(EV_RESULTS / "alpha_c6_seed42" / "headline.json")
    pair_full52_sonnet = _mean(EV_RESULTS / "pair" / "rescore_full52" / "headline.json")
    pair_full52_opus = None
    pair_opus_headline = (
        EV_RESULTS / "pair" / "rescore_full52_opus" / "alignment_pair_winner_opus_headline.json"
    )
    if pair_opus_headline.exists():
        with open(pair_opus_headline) as f:
            d = json.load(f)
        pair_full52_opus = d.get("mean_aligned")
    evo_full52_sonnet = _mean(EV_RESULTS / "evoprompt" / "rescore_full52" / "headline.json")
    evo_full52_opus = None
    evo_opus_headline = (
        EV_RESULTS
        / "evoprompt"
        / "rescore_full52_opus"
        / "alignment_evoprompt_winner_opus_headline.json"
    )
    if evo_opus_headline.exists():
        with open(evo_opus_headline) as f:
            d = json.load(f)
        evo_full52_opus = d.get("mean_aligned")
    gcg_sys = gcg_user = gcg_rand = None
    gcg_path = EV_RESULTS / "gcg" / "final.json"
    if gcg_path.exists():
        with open(gcg_path) as f:
            g = json.load(f)["result"]
        gcg_sys = g.get("system_slot_alpha")
        gcg_user = g.get("user_slot_alpha")
        gcg_rand = g.get("random_suffix_alpha")

    threshold = 43.21  # midpoint of alpha_base_null (88.82) and alpha_c6 (28.21)

    labels = [
        "null\nbaseline",
        "c6_vanilla_em\n(reference EM)",
        "PAIR winner\n(Sonnet, full 52)",
        "PAIR winner\n(Opus, full 52)",
        "EvoPrompt winner\n(Sonnet, full 52)",
        "EvoPrompt winner\n(Opus, full 52)",
        "GCG suffix\n(system slot)",
        "GCG suffix\n(user slot)",
        "GCG random\n(system slot)",
    ]
    values = [
        base_null,
        c6,
        pair_full52_sonnet,
        pair_full52_opus,
        evo_full52_sonnet,
        evo_full52_opus,
        gcg_sys,
        gcg_user,
        gcg_rand,
    ]
    colors_pal = paper_palette(5)
    colors = [
        "#999999",  # null
        "#333333",  # c6
        colors_pal[0],  # PAIR sonnet
        colors_pal[1],  # PAIR opus
        colors_pal[2],  # EvoPrompt sonnet
        colors_pal[3],  # EvoPrompt opus
        colors_pal[4],  # GCG sys
        "#c0c0c0",  # GCG user
        "#d8d8d8",  # GCG random
    ]

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        [v if v is not None else 0 for v in values],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    # Threshold line.
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"threshold = {threshold:.2f}\n(midpoint base/c6)",
    )

    # Annotate values.
    for rect, v in zip(bars, values, strict=True):
        if v is None:
            continue
        ax.annotate(
            f"{v:.1f}",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=7.5)
    ax.set_ylabel("α (aligned score 0–100, lower = more misaligned)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "EM-axis prompt-elicitability: PAIR + EvoPrompt cross threshold, GCG does not",
        fontsize=11,
    )
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()

    out_dir = ROOT / "figures" / "issue-94"
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, str(out_dir / "hero_alpha_comparison"))
    print(f"Wrote {out_dir / 'hero_alpha_comparison.png'}")


if __name__ == "__main__":
    main()
