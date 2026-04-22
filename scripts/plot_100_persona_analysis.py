#!/usr/bin/env python3
"""Generate publication-quality plots for 100-persona leakage analysis."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "single_token_100_persona"
FIG_DIR = PROJECT_ROOT / "figures" / "single_token_100_persona"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    "villain",
    "comedian",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
]
SOURCE_LABELS = {
    "villain": "Villain",
    "comedian": "Comedian",
    "assistant": "Assistant",
    "software_engineer": "Sw. Engineer",
    "kindergarten_teacher": "Kinder. Teacher",
}
SOURCE_COLORS = {
    "villain": "#d62728",
    "comedian": "#ff7f0e",
    "assistant": "#2ca02c",
    "software_engineer": "#1f77b4",
    "kindergarten_teacher": "#9467bd",
}

CATEGORIES = [
    "professional_peer",
    "modified_source",
    "opposite",
    "hierarchical",
    "intersectional",
    "cultural_variant",
    "fictional_exemplar",
    "tone_variant",
    "domain_adjacent",
    "unrelated_baseline",
]
CAT_LABELS = {
    "professional_peer": "Prof. Peer",
    "modified_source": "Modified Src",
    "opposite": "Opposite",
    "hierarchical": "Hierarchical",
    "intersectional": "Intersectional",
    "cultural_variant": "Cultural Var.",
    "fictional_exemplar": "Fictional Ex.",
    "tone_variant": "Tone Variant",
    "domain_adjacent": "Domain Adj.",
    "unrelated_baseline": "Unrelated",
}
CAT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def load_data():
    """Load compiled analysis and cosine correlation data."""
    with open(RESULTS_DIR / "compiled_analysis.json") as f:
        compiled = json.load(f)
    with open(RESULTS_DIR / "cosine_leakage_correlation.json") as f:
        cosine = json.load(f)
    return compiled, cosine


def plot_scatter(cosine_data: dict) -> None:
    """5-panel scatter: cosine sim vs leakage, colored by category."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    fig.suptitle(
        "Cosine Similarity vs Marker Leakage (Layer 20)",
        fontsize=14,
        fontweight="bold",
    )

    cat_to_color = {c: CAT_COLORS[i] for i, c in enumerate(CATEGORIES)}

    for ax, source in zip(axes, SOURCES, strict=True):
        detail = cosine_data["layer20"]["_per_source_detail"].get(source, [])
        if not detail:
            continue

        for d in detail:
            cat = d["category"]
            color = cat_to_color.get(cat, "gray")
            ax.scatter(
                d["cosine_sim"],
                d["leakage_rate"],
                c=[color],
                s=15,
                alpha=0.7,
                edgecolors="none",
            )

        # Add Spearman rho annotation
        src_stats = cosine_data["layer20"].get(source, {})
        rho = src_stats.get("spearman_rho", 0)
        ax.text(
            0.05,
            0.95,
            f"rho={rho:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_title(SOURCE_LABELS[source], fontsize=11)
        ax.set_xlabel("Cosine Similarity", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Leakage Rate", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    # Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cat_to_color[c],
            markersize=6,
            label=CAT_LABELS[c],
        )
        for c in CATEGORIES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(
        FIG_DIR / "cosine_vs_leakage_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIG_DIR / "cosine_vs_leakage_scatter.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved cosine_vs_leakage_scatter")


def plot_category_bars(compiled: dict) -> None:
    """Grouped bar chart: mean leakage by category per source."""
    fig, ax = plt.subplots(figsize=(14, 5.5))

    n_cats = len(CATEGORIES)
    n_src = len(SOURCES)
    bar_width = 0.15
    x = np.arange(n_cats)

    for i, source in enumerate(SOURCES):
        means = []
        # Load marker_eval for category info
        with open(RESULTS_DIR / source / "marker_eval.json") as f:
            marker_data = json.load(f)

        for cat in CATEGORIES:
            rates = [
                info["rate"]
                for _p, info in marker_data.items()
                if info.get("category") == cat and _p != source
            ]
            means.append(np.mean(rates) if rates else 0)

        ax.bar(
            x + i * bar_width,
            [m * 100 for m in means],
            bar_width,
            label=SOURCE_LABELS[source],
            color=SOURCE_COLORS[source],
            alpha=0.85,
        )

    ax.set_xlabel("Relationship Category", fontsize=11)
    ax.set_ylabel("Mean Leakage Rate (%)", fontsize=11)
    ax.set_title(
        "Marker Leakage by Relationship Category",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x + bar_width * (n_src - 1) / 2)
    ax.set_xticklabels(
        [CAT_LABELS[c] for c in CATEGORIES],
        rotation=35,
        ha="right",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "leakage_by_category_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIG_DIR / "leakage_by_category_bar.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved leakage_by_category_bar")


def plot_heatmap(compiled: dict) -> None:
    """Heatmap of top-25 most-leaked-to personas across 5 sources."""
    # Collect max leakage per persona across sources
    all_personas = {}
    for source in SOURCES:
        for persona, rate in compiled["results"].get(source, {}).items():
            if persona in SOURCES:
                continue  # Skip source self-rates
            if persona not in all_personas:
                all_personas[persona] = {}
            all_personas[persona][source] = rate

    # Rank by max leakage across sources
    ranked = sorted(
        all_personas.items(),
        key=lambda x: max(x[1].values()),
        reverse=True,
    )[:25]

    persona_names = [r[0] for r in ranked]
    matrix = np.zeros((len(persona_names), len(SOURCES)))
    for i, (_persona, rates) in enumerate(ranked):
        for j, source in enumerate(SOURCES):
            matrix[i, j] = rates.get(source, 0)

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(persona_names)):
        for j in range(len(SOURCES)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )

    ax.set_xticks(range(len(SOURCES)))
    ax.set_xticklabels(
        [SOURCE_LABELS[s] for s in SOURCES],
        fontsize=9,
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(persona_names)))
    ax.set_yticklabels(persona_names, fontsize=8)
    ax.set_title(
        "Top 25 Most-Leaked-To Personas",
        fontsize=13,
        fontweight="bold",
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Leakage Rate", fontsize=10)

    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "leakage_heatmap_top25.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIG_DIR / "leakage_heatmap_top25.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved leakage_heatmap_top25")


def plot_layer_correlation(cosine_data: dict) -> None:
    """Line plot: Spearman rho by layer, one line per source + aggregate."""
    layers = [10, 15, 20, 25]
    fig, ax = plt.subplots(figsize=(8, 5))

    for source in SOURCES:
        rhos = []
        for layer in layers:
            key = f"layer{layer}"
            rho = cosine_data[key].get(source, {}).get("spearman_rho", 0)
            rhos.append(rho)
        ax.plot(
            layers,
            rhos,
            "o-",
            color=SOURCE_COLORS[source],
            label=SOURCE_LABELS[source],
            linewidth=2,
            markersize=6,
        )

    # Aggregate
    agg_rhos = []
    for layer in layers:
        key = f"layer{layer}"
        rho = cosine_data[key].get("_aggregate", {}).get("spearman_rho", 0)
        agg_rhos.append(rho)
    ax.plot(
        layers,
        agg_rhos,
        "s--",
        color="black",
        label="Aggregate",
        linewidth=2.5,
        markersize=7,
    )

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Spearman rho", fontsize=11)
    ax.set_title(
        "Cosine-Leakage Correlation by Layer",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(layers)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.3, 1.0)

    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "correlation_by_layer.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIG_DIR / "correlation_by_layer.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved correlation_by_layer")


def main():
    compiled, cosine = load_data()
    plot_scatter(cosine)
    plot_category_bars(compiled)
    plot_heatmap(compiled)
    plot_layer_correlation(cosine)
    print(f"\nAll plots saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
