"""Regenerate the cosine layer-sweep figure for issue #470 with blog-style register.

Replaces the matplotlib-default version with plain-English in-figure labels and
the Anthropic-blog visual register so it slots into the clean-result body.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGRESSION = REPO_ROOT / "eval_results" / "issue_470" / "regression.json"
FIG_DIR = REPO_ROOT / "figures" / "issue_470"

SOURCE_LABELS = {
    "assistant": "Assistant (named)",
    "comedian": "Comedian",
    "kindergarten_teacher": "Kindergarten teacher",
    "qwen_default": "Qwen default",
    "software_engineer": "Software engineer",
    "villain": "Villain",
}

LAYERS = [7, 14, 21, 27]
LAYER_KEYS = [f"cosine_response_l{layer}" for layer in LAYERS]


def main() -> None:
    set_paper_style("blog")

    data = json.loads(REGRESSION.read_text())
    preds = data["predictors"]

    per_source: dict[str, list[float]] = {}
    for layer_key in LAYER_KEYS:
        for src, vals in preds[layer_key]["per_source"].items():
            per_source.setdefault(src, []).append(float(vals["rho"]))

    pooled = [
        float(data["cosine_layer_ladder"]["pooled_per_layer"][layer_key]["source_fe_rho"])
        for layer_key in LAYER_KEYS
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    # Per-source lines
    source_order = [
        "comedian",
        "kindergarten_teacher",
        "assistant",
        "villain",
        "software_engineer",
        "qwen_default",
    ]
    palette = [
        paper_palette_role("accent"),
        paper_palette_role("primary"),
        "#5BA3D0",
        "#9A6FBF",
        "#D97757",
        "#7A8B99",
    ]
    for src, color in zip(source_order, palette, strict=True):
        ax.plot(
            LAYERS,
            per_source[src],
            marker="o",
            linewidth=1.6,
            color=color,
            label=SOURCE_LABELS[src],
        )

    # Pooled line (heavier, dashed)
    ax.plot(
        LAYERS,
        pooled,
        marker="s",
        linewidth=2.2,
        linestyle="--",
        color="#222222",
        label="Pooled (source fixed effects)",
    )

    ax.axhline(0.0, color="#999999", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Layer (Qwen residual stream)")
    ax.set_ylabel("Per-source Spearman ρ (predictor vs leakage)")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([str(layer) for layer in LAYERS])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5)

    # Annotate the layer-14 comedian peak
    ax.annotate(
        "Comedian peaks here\nρ = +0.68",
        xy=(14, 0.683),
        xytext=(17.5, 0.60),
        fontsize=8.5,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
    )
    ax.set_ylim(-0.65, 0.85)

    set_title_subtitle(
        ax,
        "Response-token cosine at layer 14 is the strongest single predictor",
        "Per-source ρ across four residual-stream layers; pooled line lifts above #411's layer-20 choice at layer 14.",
        source="eval_results/issue_470/regression.json",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_470/cosine_layer_sweep", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)
    print(f"Saved figure to {FIG_DIR / 'cosine_layer_sweep.png'}")


if __name__ == "__main__":
    main()
