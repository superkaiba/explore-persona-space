"""Highest/lowest-projecting LMSYS contexts per trait (user ask, 2026-07-22):
a literal top-8 / bottom-8 diverging horizontal-bar view with the prompt text as
labels — the display the theme-composition figure (pinv_topk_lmsys_themes) does
not give.

Reads the committed pinv_topk_contexts.json; writes
figures/issue_779/pinv_topk_lmsys_topbottom.{png,pdf,meta.json}. 0 GPU-h.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

HERE = Path(__file__).resolve().parent
TRAITS = ["evil", "sycophancy", "hallucination"]
TRAIT_LABELS = {
    "evil": "evil (L14)",
    "sycophancy": "sycophancy (L26)",
    "hallucination": "hallucination (L17)",
}
N_SHOW = 8
WRAP = 66


def label_for(rec: dict) -> str:
    if rec.get("text"):
        t = " ".join(rec["text"].split())
    else:
        t = f"[{rec.get('flagged')} roleplay — not quoted]"
    if len(t) > WRAP:
        t = t[: WRAP - 1] + "…"
    return t


def main() -> int:
    data = json.loads((HERE / "pinv_topk_contexts.json").read_text())
    set_paper_style("blog")
    pos_c, neg_c = paper_palette(2)
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 13.5))
    for ax, trait in zip(axes, TRAITS, strict=True):
        tb = data["traits"][trait]["lmsys_topbottom"]["w_pinv_kstar"]
        top = tb["top"][:N_SHOW]
        bot = tb["bottom"][:N_SHOW]
        # y increases upward: put the bottom block low, the top block high, so the
        # single highest-projecting prompt sits at the very top of the panel.
        rows = list(reversed(bot)) + list(top)
        ys = range(len(rows))
        scores = [r["score"] for r in rows]
        colors = [neg_c if r["score"] < 0 else pos_c for r in rows]
        ax.barh(list(ys), scores, color=colors, height=0.72)
        ax.set_yticks(list(ys))
        ax.set_yticklabels(
            [textwrap.shorten(label_for(r), width=WRAP, placeholder="…") for r in rows], fontsize=7
        )
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_title(
            f"{TRAIT_LABELS[trait]} — top-{N_SHOW} (colored) and bottom-{N_SHOW} "
            "LMSYS prompts by pre-image projection",
            loc="left",
            fontsize=10,
        )
        ax.set_xlabel("projection onto the pre-image direction  ⟨c_std, w⟩")
        ax.margins(y=0.01)
    # one legend
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(color=pos_c, label="highest-projecting (trait-eliciting)"),
            Patch(color=neg_c, label="lowest-projecting"),
        ],
        loc="upper right",
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "Highest- and lowest-projecting LMSYS prompts on each persona's pre-image direction",
        x=0.01,
        ha="left",
        fontweight="semibold",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    savefig_paper(fig, "issue_779/pinv_topk_lmsys_topbottom", dir="figures/")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
