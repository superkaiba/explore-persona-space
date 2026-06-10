"""Build a figure showing that most marker firings are token-loop runaways, not clean end-of-response markers.

This is the measurement-validity figure for clean-result #488 — surfaces a real caveat about what
emission_rate actually measures.
"""

from __future__ import annotations

import json
import os

import numpy as np
from huggingface_hub import snapshot_download

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)


def main() -> None:
    # Pull all emission JSONs at the picked frac (2.0) for both seeds.
    path = snapshot_download(
        "superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        allow_patterns=[
            "issue488_distance_predicts_transfer/raw_completions/emission/frac200/42/emission_*.json",
            "issue488_distance_predicts_transfer/raw_completions/emission/frac200/137/emission_*.json",
        ],
    )

    # For each source, count clean (<=3 markers) vs runaway (>=10 markers AND hit token cap)
    # firings, pooled across seeds and all (target, question) pairs.
    sources = sorted(
        [
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "C1",
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "E2",
            "E3",
            "E4",
            "E5",
            "F1",
            "F2",
            "F3",
            "F4",
            "G1",
            "G2",
            "G3",
        ]
    )

    rows = []
    for src in sources:
        n_clean = 0
        n_runaway = 0
        n_other_fired = 0  # in between: 4-9 markers, didn't necessarily hit cap
        for seed in [42, 137]:
            f = os.path.join(
                path,
                f"issue488_distance_predicts_transfer/raw_completions/emission/frac200/{seed}/emission_{src}.json",
            )
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            for _tgt_k, tgt in d["targets"].items():
                for _qk, q in tgt.items():
                    if "samples" not in q:
                        continue
                    for s in q["samples"]:
                        if " ※" not in s["text"]:
                            continue
                        nm = s["text"].count(" ※")
                        if nm <= 3:
                            n_clean += 1
                        elif nm >= 10 and s["n_tokens"] >= 2000:
                            n_runaway += 1
                        else:
                            n_other_fired += 1
        n_total = n_clean + n_runaway + n_other_fired
        rows.append((src, n_clean, n_runaway, n_other_fired, n_total))

    # Sort by total fired descending
    rows.sort(key=lambda r: -r[4])

    set_paper_style("blog")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    srcs = [r[0] for r in rows]
    clean = np.array([r[1] for r in rows])
    runaway = np.array([r[2] for r in rows])
    other = np.array([r[3] for r in rows])

    x = np.arange(len(srcs))
    clean_color = paper_palette_role("primary")
    runaway_color = paper_palette_role("baseline")
    other_color = paper_palette_role("neutral")

    ax.bar(
        x,
        clean,
        label="clean (≤3 markers in response)",
        color=clean_color,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x,
        other,
        bottom=clean,
        label="mid (4–9 markers)",
        color=other_color,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x,
        runaway,
        bottom=clean + other,
        label="runaway (≥10 markers AND hit 2048-token cap)",
        color=runaway_color,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(srcs, rotation=45, ha="right")
    ax.set_ylabel("Number of fired samples (pooled both seeds, all targets, all probes)")
    ax.legend(loc="upper right", frameon=False)
    set_title_subtitle(
        ax,
        "Most marker firings are degenerate token loops, not clean end-of-response markers",
        "frac=2.0 (headline frac), both seeds, all 27 sources × 27 targets × 20 probes × 8 samples",
        source="eval_results/issue_488 + HF data repo",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_488/runaway_vs_clean_firings", dir="figures/")
    plt.close(fig)

    # Print summary
    total_clean = clean.sum()
    total_runaway = runaway.sum()
    total_other = other.sum()
    total_all = total_clean + total_runaway + total_other
    print(f"Total fired (both seeds): {total_all}")
    print(f"  clean (<=3): {total_clean} ({100 * total_clean / total_all:.1f}%)")
    print(f"  mid (4-9):   {total_other} ({100 * total_other / total_all:.1f}%)")
    print(f"  runaway:     {total_runaway} ({100 * total_runaway / total_all:.1f}%)")


if __name__ == "__main__":
    main()
