#!/usr/bin/env python3
"""Issue #101: Generate all paper-quality figures.

Reads JSON results from eval_results/issue101/ and produces figures in figures/issue101/.

Figures:
    A1: Pairwise cosine heatmap (3 conditions x 4 layers)
    A2: Spearman profile correlation across layers
    B2: Cross-leakage heatmap (3 source x 14 eval, delta accuracy)
    Marker: Marker rate heatmap (3 source x 14 eval)
    C: Behavioral comparison bars (alignment, refusal, self-ID)

Usage:
    uv run python scripts/issue101_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

set_paper_style()

RESULTS_DIR = Path("eval_results/issue101")
FIG_DIR = Path("figures/issue101")


# ── Helper ────────────────────────────────────────────────────────────────────

SHORT = {
    "qwen_default": "Qwen default",
    "generic_assistant": "Generic asst.",
    "empty_system": "Empty system",
    "no_system_sanity": "No system",
    "software_engineer": "SWE",
    "kindergarten_teacher": "Teacher",
    "data_scientist": "DataSci",
    "medical_doctor": "Doctor",
    "librarian": "Librarian",
    "french_person": "French",
    "villain": "Villain",
    "comedian": "Comedian",
    "police_officer": "Police",
    "zelthari_scholar": "Zelthari",
}


def short(name):
    return SHORT.get(name, name)


# ── Exp A: Geometry ───────────────────────────────────────────────────────────


def plot_exp_a():
    path = RESULTS_DIR / "exp_a_geometry.json"
    if not path.exists():
        print(f"  Skipping Exp A: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    layers = data["layers"]

    # A1: Pairwise cosine heatmap
    pairs = list(data["pairwise_cosines_centered"][str(layers[0])].keys())
    matrix = np.zeros((len(pairs), len(layers)))
    for j, layer in enumerate(layers):
        for i, pair in enumerate(pairs):
            matrix[i, j] = data["pairwise_cosines_centered"][str(layer)][pair]

    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{layer}" for layer in layers])
    ax.set_yticks(range(len(pairs)))
    pair_labels = [p.replace("_vs_", " vs\n").replace("_", " ") for p in pairs]
    ax.set_yticklabels(pair_labels, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_title("Pairwise cosine similarity (centered)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    savefig_paper(fig, "a1_pairwise_cosine", dir=str(FIG_DIR))
    plt.close(fig)
    print("  Saved a1_pairwise_cosine")

    # A2: Spearman profile correlation across layers
    spearman = data["spearman_profile_correlations"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = paper_palette(3)
    for idx, pair in enumerate(pairs):
        rhos = [spearman[str(layer)][pair]["rho"] for layer in layers]
        label = pair.replace("_vs_", " vs ").replace("_", " ")
        ax.plot(layers, rhos, "o-", color=colors[idx], label=label, markersize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Profile correlation across layers")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    fig.tight_layout()
    savefig_paper(fig, "a2_spearman_layers", dir=str(FIG_DIR))
    plt.close(fig)
    print("  Saved a2_spearman_layers")


# ── Exp B2: Cross-leakage ────────────────────────────────────────────────────


def plot_b2():
    path = RESULTS_DIR / "b2_cross_leakage.json"
    if not path.exists():
        print(f"  Skipping B2: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    sources = list(data["cross_leakage"].keys())
    # Get eval personas from the first source
    eval_personas = list(data["cross_leakage"][sources[0]].keys())

    # Build delta matrix
    matrix = np.zeros((len(sources), len(eval_personas)))
    for i, src in enumerate(sources):
        for j, ep in enumerate(eval_personas):
            matrix[i, j] = data["cross_leakage"][src][ep]["delta"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.1)
    ax.set_xticks(range(len(eval_personas)))
    ax.set_xticklabels([short(ep) for ep in eval_personas], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([short(s) for s in sources])
    ax.set_xlabel("Eval persona")
    ax.set_ylabel("Source (trained on)")
    ax.set_title("ARC-C accuracy delta (post - base)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.15 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta accuracy")
    fig.tight_layout()
    savefig_paper(fig, "b2_cross_leakage", dir=str(FIG_DIR))
    plt.close(fig)
    print("  Saved b2_cross_leakage")


# ── Marker results ────────────────────────────────────────────────────────────


def plot_marker():
    path = RESULTS_DIR / "marker_results.json"
    if not path.exists():
        print(f"  Skipping Marker: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    sources = list(results.keys())
    eval_personas = list(results[sources[0]].keys())

    # Build marker rate matrix
    matrix = np.zeros((len(sources), len(eval_personas)))
    for i, src in enumerate(sources):
        for j, ep in enumerate(eval_personas):
            matrix[i, j] = results[src][ep]["marker_rate"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(eval_personas)))
    ax.set_xticklabels([short(ep) for ep in eval_personas], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([short(s) for s in sources])
    ax.set_xlabel("Eval persona")
    ax.set_ylabel("Source (trained on)")
    ax.set_title("[ZLT] marker rate")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Marker rate")
    fig.tight_layout()
    savefig_paper(fig, "marker_rate_heatmap", dir=str(FIG_DIR))
    plt.close(fig)
    print("  Saved marker_rate_heatmap")


# ── Exp C: Behavioral ─────────────────────────────────────────────────────────


def plot_exp_c():
    path = RESULTS_DIR / "exp_c_behavioral.json"
    if not path.exists():
        print(f"  Skipping Exp C: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    conditions = data["conditions"]
    heuristic = data["heuristic_metrics"]
    judge = data.get("judge_aggregated", {})

    colors = paper_palette(3)
    x = np.arange(len(conditions))

    # Panel 1: Alignment + Coherent scores
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # Alignment
    ax = axes[0]
    vals = [judge.get(c, {}).get("mean_aligned", 0) or 0 for c in conditions]
    ax.bar(x, vals, width=0.6, color=colors[: len(conditions)])
    ax.set_xticks(x)
    ax.set_xticklabels([short(c) for c in conditions], fontsize=7)
    ax.set_ylabel("Mean alignment score")
    ax.set_title("Claude judge: Alignment")
    ax.set_ylim(0, 100)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=7)

    # Refusal rate
    ax = axes[1]
    vals = [heuristic[c]["refusal_rate"] * 100 for c in conditions]
    ax.bar(x, vals, width=0.6, color=colors[: len(conditions)])
    ax.set_xticks(x)
    ax.set_xticklabels([short(c) for c in conditions], fontsize=7)
    ax.set_ylabel("Refusal rate (%)")
    ax.set_title("Refusal rate")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=7)

    # Self-ID rate
    ax = axes[2]
    vals = [heuristic[c]["self_id_rate"] * 100 for c in conditions]
    ax.bar(x, vals, width=0.6, color=colors[: len(conditions)])
    ax.set_xticks(x)
    ax.set_xticklabels([short(c) for c in conditions], fontsize=7)
    ax.set_ylabel("Self-ID rate (%)")
    ax.set_title("Self-identification rate")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=7)

    fig.suptitle("Exp C: Behavioral baseline across assistant conditions", fontsize=10, y=1.02)
    fig.tight_layout()
    savefig_paper(fig, "c_behavioral", dir=str(FIG_DIR))
    plt.close(fig)
    print("  Saved c_behavioral")


# ── B3: Other sources to assistant ────────────────────────────────────────────


def plot_b3():
    path = RESULTS_DIR / "b3_existing_to_assistant.json"
    if not path.exists():
        print(f"  Skipping B3: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    sources = data["sources"]
    eval_conditions = data["eval_conditions"]
    b3 = data["b3_matrix"]

    # Build delta matrix
    matrix = np.zeros((len(sources), len(eval_conditions)))
    for i, src in enumerate(sources):
        for j, ec in enumerate(eval_conditions):
            matrix[i, j] = b3[src][ec]["delta"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.1)
    ax.set_xticks(range(len(eval_conditions)))
    ax.set_xticklabels([short(ec) for ec in eval_conditions])
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([short(s) for s in sources])
    ax.set_xlabel("Eval assistant condition")
    ax.set_ylabel("Source persona (trained wrong answers)")
    ax.set_title("B3: Non-assistant source -> assistant leakage")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.15 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta accuracy")
    fig.tight_layout()
    savefig_paper(fig, "b3_other_to_assistant", dir=str(FIG_DIR))
    plt.close(fig)
    print("  Saved b3_other_to_assistant")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating issue #101 figures...")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Figures dir: {FIG_DIR}")

    print("\n--- Exp A: Geometry ---")
    plot_exp_a()

    print("\n--- B2: Cross-leakage ---")
    plot_b2()

    print("\n--- Marker ---")
    plot_marker()

    print("\n--- Exp C: Behavioral ---")
    plot_exp_c()

    print("\n--- B3: Other to assistant ---")
    plot_b3()

    print("\nDone!")


if __name__ == "__main__":
    main()
