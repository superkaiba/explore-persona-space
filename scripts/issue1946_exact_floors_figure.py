"""Render the issue-1946 exact-floor adjustment figure (round 2, exact-sae-floors).

Re-renders the driver's ``floor_adjustment_exact.png`` in the round-1 house style:
plain-English contrast labels (paper-plots §3.5), refusal categories highlighted,
``savefig_paper`` trio (PNG + PDF + data-bearing sidecar).

Inputs: ``eval_results/issue_1946/exact_floors/sae_space/taxonomy.json`` — the
unadjusted 22-contrast battery (x) and the exact floor-adjusted battery over the
19 floor-covered contrasts (y), history-only (prefix) and bare-query arms.

Usage:
    uv run python scripts/issue1946_exact_floors_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps BEFORE matplotlib/numpy import (#847)

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
TAXONOMY = REPO / "eval_results/issue_1946/exact_floors/sae_space/taxonomy.json"

# Plain-English contrast names (matches the round-1 floor_adjustment_effect figure).
LABELS = {
    "language=en": "English",
    "topic=factual_qa": "factual Q&A",
    "topic=creative_writing": "creative writing",
    "topic=coding": "coding",
    "topic=advice_howto": "advice / how-to",
    "topic=chitchat_social": "chitchat",
    "topic=translation": "translation",
    "topic=summarization_extraction": "summarization",
    "topic=roleplay_persona": "roleplay",
    "topic=math": "math",
    "refusal_adjacent=yes": "refusal-adjacent",
    "answer_is_refusal=yes": "answer is refusal",
    "format=code": "code answers",
    "format=list": "list answers",
    "format=prose": "prose answers",
    "depth=2-2": "depth 2",
    "depth=3-4": "depth 3–4",
    "depth=>=5": "depth ≥5",
    "corpus=wildchat": "WildChat",
}
REFUSAL = {"refusal_adjacent=yes", "answer_is_refusal=yes"}
ARMS = [
    ("prefix_L19_ridge", "history-only (prefix) arm (19 floor-covered contrasts)"),
    ("bare_L19_ridge", "bare-query arm (19 floor-covered contrasts)"),
]


def main() -> None:
    tax = json.loads(TAXONOMY.read_text())
    set_paper_style("blog")
    c_other, c_refusal = paper_palette_blog(2)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (arm, title) in zip(axes, ARMS, strict=True):
        unadj = {c["contrast"]: c["delta_mean_nerr"] for c in tax["arms"][arm]["contrasts"]}
        adj_rows = tax["arms"][arm]["floor_adjusted"]["contrasts"]
        xs, ys, names = [], [], []
        for row in adj_rows:
            name = row["contrast"]
            xs.append(unadj[name])
            ys.append(row["delta_mean_adj_nerr"])
            names.append(name)
        assert len(xs) == 19, f"{arm}: expected 19 floor-covered contrasts, got {len(xs)}"

        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        pad = 0.06 * (hi - lo)
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            color="#999999",
            lw=0.8,
            zorder=1,
        )
        for grp, color, label in (
            (sorted(set(names) - REFUSAL), c_other, "other categories"),
            (sorted(REFUSAL), c_refusal, "refusal categories"),
        ):
            pts = [(x, y, n) for x, y, n in zip(xs, ys, names, strict=True) if n in grp]
            ax.scatter(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=color,
                s=28,
                zorder=3,
                label=label,
            )
            for x, y, n in pts:
                ax.text(x, y + 0.008, LABELS[n], fontsize=6.5, ha="center", zorder=4)
        ax.set_title(title, loc="left")
        ax.set_xlabel("unadjusted SAE-space category Δ")
        ax.set_ylabel("exact floor-adjusted Δ")
        ax.legend(loc="upper left")

    savefig_paper(fig, "issue_1946/floor_adjustment_exact", dir=str(REPO / "figures"))
    plt.close(fig)
    print("wrote", REPO / "figures/issue_1946/floor_adjustment_exact.png")


if __name__ == "__main__":
    main()
