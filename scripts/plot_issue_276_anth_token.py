"""Issue #276 anth-token follow-up: bar chart of all 31 conditions.

X-axis: condition (sorted by rate within group order: ANTH-leading, ANTH-suffix,
ANTH-embedded, non-anth controls, sanity).
Y-axis: exact_target rate (k/100).
Color: ANTH-containing (blue) vs non-anth (orange) vs sanity (grey).

Output: figures/issue_276/anth_token_followup_chart.png + .pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import set_paper_style

set_paper_style()


RESULTS_PATH = Path("eval_results/issue_276/anth_token_followup/headline_numbers.json")
OUTPUT_DIR = Path("figures/issue_276")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = ["ctrl", "anth_lead", "anth_sfx", "anth_mid", "ctrl_noanth"]
GROUP_LABEL = {
    "ctrl": "Sanity",
    "anth_lead": "anth-leading paths",
    "anth_sfx": "anth + suffix",
    "anth_mid": "anth embedded",
    "ctrl_noanth": "non-anth controls",
}
GROUP_COLOR = {
    "ctrl": "#4C566A",
    "anth_lead": "#5E81AC",
    "anth_sfx": "#88C0D0",
    "anth_mid": "#A3BE8C",
    "ctrl_noanth": "#BF616A",
}


def main():
    with RESULTS_PATH.open() as f:
        data = json.load(f)
    pb = data["pingbang"]

    # Sort within each group: by k descending, then by id
    rows = []
    for cid, m in pb.items():
        rows.append((m["group"], cid, m["user"], m["k"], m["n"]))
    rows.sort(key=lambda r: (GROUP_ORDER.index(r[0]), -r[3], r[1]))

    labels = [r[2] for r in rows]
    rates = [100.0 * r[3] / r[4] for r in rows]
    colors = [GROUP_COLOR[r[0]] for r in rows]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    xs = list(range(len(rows)))
    bars = ax.bar(xs, rates, color=colors, edgecolor="black", linewidth=0.4)

    # Annotate each bar with the k count
    for x, r, row in zip(xs, rates, rows):
        if r > 0.5:
            ax.text(x, r + 1.0, f"{row[3]}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(x, 0.8, "0", ha="center", va="bottom", fontsize=8, color="#888")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("exact_target rate (k / 100)")
    ax.set_ylim(0, max(95, max(rates) + 8))
    ax.set_title(
        "Issue #276 follow-up: focused `anth`-token-only probe on Pingbang Qwen3-4B "
        "(n=100/condition)\n"
        "ANTH-containing (excl. canonical): 119/2000 = 6.0% vs non-anth controls: 0/900 "
        "(Fisher p = 2.4×10⁻²⁰)",
        fontsize=10,
    )

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g], label=GROUP_LABEL[g]) for g in GROUP_ORDER
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    png = OUTPUT_DIR / "anth_token_followup_chart.png"
    pdf = OUTPUT_DIR / "anth_token_followup_chart.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
