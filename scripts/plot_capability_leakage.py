#!/usr/bin/env python3
"""Hero figure for Issue #69 Exp A: Capability leakage scatter plots.

Shows ARC-C accuracy vs cosine similarity to source persona for all 5 sources
in a 2x3 subplot grid. Demonstrates the steep capability-leakage gradient.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.analysis.paper_plots import set_paper_style

set_paper_style()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_LABELS = {
    "villain": "Villain",
    "comedian": "Comedian",
    "assistant": "Assistant",
    "software_engineer": "Software Engineer",
    "kindergarten_teacher": "Kindergarten Teacher",
}


def load_source_extended(source):
    """Load extended persona eval results for a source."""
    path = (
        PROJECT_ROOT
        / f"eval_results/capability_leakage/{source}_lr1e-05_ep3/extended_persona_eval.json"
    )
    data = json.load(open(path))
    names, cosines, accs = [], [], []
    for name, vals in data.items():
        cos = vals.get("cosine_to_source", 0)
        acc = vals.get("accuracy", 0)
        if cos > 0:  # Skip controls with cos=0
            names.append(name)
            cosines.append(cos)
            accs.append(acc * 100)
    return names, np.array(cosines), np.array(accs)


def load_five_source_summary():
    """Load 5-source results for the bar chart."""
    bl = json.load(
        open(PROJECT_ROOT / "eval_results/capability_leakage/baseline/capability_per_persona.json")
    )
    results = {}
    for src in SOURCE_LABELS:
        r = json.load(
            open(
                PROJECT_ROOT / f"eval_results/capability_leakage/{src}_lr1e-05_ep3/run_result.json"
            )
        )
        bl_acc = bl[src]["arc_challenge_logprob"] * 100
        post_acc = (
            r["post_results"][src]["arc_challenge_logprob"] * 100
            if isinstance(r["post_results"][src], dict)
            else r["post_results"][src] * 100
        )
        byst_delta = r["mean_bystander_delta_pp"]
        results[src] = {
            "baseline": bl_acc,
            "post": post_acc,
            "delta": post_acc - bl_acc,
            "byst_delta": byst_delta,
        }
    return results


def plot_source_panel(ax, source, names, cosines, accs):
    """Plot one source's cosine vs ARC-C scatter with persona labels."""
    bl_acc = 87.9  # approximate baseline for all personas
    ax.axhline(y=bl_acc, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    scatter = ax.scatter(
        cosines,
        accs,
        c=accs,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        s=30,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.2,
        zorder=3,
    )

    # Highlight source persona
    if source in names:
        idx = names.index(source)
        ax.scatter(
            [cosines[idx]],
            [accs[idx]],
            s=80,
            facecolors="none",
            edgecolors="red",
            linewidths=1.5,
            zorder=4,
        )

    # Label all personas
    for i, name in enumerate(names):
        label = name.replace("_", " ")
        # Shorten long names
        if len(label) > 18:
            label = label[:16] + ".."
        # Only label points that are notably degraded or the source
        if accs[i] < 70 or name == source:
            ax.annotate(
                label,
                (cosines[i], accs[i]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=5,
                alpha=0.7,
            )

    src_delta = accs[names.index(source)] - bl_acc if source in names else 0
    ax.set_title(f"{SOURCE_LABELS[source]} (src: {src_delta:+.0f}pp, N={len(names)})", fontsize=9)
    ax.set_ylim(-2, 100)
    ax.set_ylabel("ARC-C (%)", fontsize=8)
    ax.set_xlabel("Cosine to source", fontsize=8)
    ax.tick_params(labelsize=7)

    return scatter


def main():
    five_src = load_five_source_summary()
    sources = list(SOURCE_LABELS.keys())

    # ── Multi-source scatter grid ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for i, source in enumerate(sources):
        try:
            names, cosines, accs = load_source_extended(source)
            scatter = plot_source_panel(axes_flat[i], source, names, cosines, accs)
        except FileNotFoundError:
            axes_flat[i].text(
                0.5,
                0.5,
                "Data not available",
                ha="center",
                va="center",
                transform=axes_flat[i].transAxes,
            )
            axes_flat[i].set_title(f"{SOURCE_LABELS[source]}", fontsize=9)

    # Use last subplot for a summary legend / info
    ax_info = axes_flat[5]
    ax_info.axis("off")

    summary_text = "5-source summary\n(source persona ARC-C delta):\n\n"
    for src in sources:
        d = five_src[src]
        summary_text += f"{SOURCE_LABELS[src]}: {d['delta']:+.0f}pp\n"
    summary_text += "\nBaseline ~88% (dashed line)\nRed ring = source persona"

    ax_info.text(
        0.5,
        0.5,
        summary_text,
        ha="center",
        va="center",
        transform=ax_info.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(
        "Figure 1: ARC-C accuracy vs cosine similarity to source persona after contrastive wrong-answer SFT.\n"
        "Each panel shows one source persona. Red ring = source. Labeled points have ARC-C < 70%.\n"
        "The cosine boundary for degradation varies per source (villain: ~0.93, software_engineer: ~0.997).",
        fontsize=9,
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = PROJECT_ROOT / "figures" / "capability_leakage"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cosine_vs_arcc_all_sources.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "cosine_vs_arcc_all_sources.pdf", bbox_inches="tight")
    print(f"Saved to {out_dir}/cosine_vs_arcc_all_sources.{{png,pdf}}")

    # ── Before/after bar chart ────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    source_labels = [SOURCE_LABELS[s] for s in sources]

    baselines = [five_src[s]["baseline"] for s in sources]
    posts = [five_src[s]["post"] for s in sources]

    x = np.arange(len(sources))
    width = 0.35

    ax2.bar(x - width / 2, baselines, width, label="Before training", color="#4ECDC4", alpha=0.8)
    ax2.bar(x + width / 2, posts, width, label="After training", color="#FF6B6B", alpha=0.8)

    for i, (bl, po) in enumerate(zip(baselines, posts)):
        delta = po - bl
        ax2.text(
            i + width / 2,
            max(po, 5) + 2,
            f"{delta:+.0f}pp",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="red" if delta < -10 else "green",
        )

    ax2.set_ylabel("ARC-C accuracy (%)", fontsize=11)
    ax2.set_title(
        "Figure 2: ARC-C on the source persona before and after contrastive wrong-answer SFT.\n"
        "4/5 sources drop to near-zero. The assistant persona resists degradation (-2pp).",
        fontsize=9,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(source_labels, fontsize=9, rotation=15, ha="right")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    fig2.savefig(out_dir / "five_source_bar.png", dpi=200, bbox_inches="tight")
    fig2.savefig(out_dir / "five_source_bar.pdf", bbox_inches="tight")
    print(f"Saved to {out_dir}/five_source_bar.{{png,pdf}}")


if __name__ == "__main__":
    main()
