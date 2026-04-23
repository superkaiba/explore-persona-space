#!/usr/bin/env python3
"""Hero figure for Issue #69 Exp A: Capability leakage scatter plot.

Shows ARC-C accuracy vs cosine similarity to source persona for the villain
source model. Demonstrates the steep capability-leakage gradient.
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


def load_villain_extended():
    """Load villain source extended persona eval results."""
    path = (
        PROJECT_ROOT
        / "eval_results/capability_leakage/villain_lr1e-05_ep3/extended_persona_eval.json"
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
    """Load 5-source results for the inset bar chart."""
    bl = json.load(
        open(PROJECT_ROOT / "eval_results/capability_leakage/baseline/capability_per_persona.json")
    )
    results = {}
    for src in ["villain", "comedian", "assistant", "software_engineer", "kindergarten_teacher"]:
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


def main():
    names, cosines, accs = load_villain_extended()
    five_src = load_five_source_summary()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline reference
    ax.axhline(
        y=87.9, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Baseline (~88%)"
    )

    # Color by accuracy: red for low, blue for high
    colors = plt.cm.RdYlGn(accs / 100)

    scatter = ax.scatter(
        cosines,
        accs,
        c=accs,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        s=50,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
        zorder=3,
    )

    # Label key personas
    labels_to_show = {
        "villain": (-15, 10),
        "hacker_villain": (10, -15),
        "evil_scientist": (10, 5),
        "russian_villain": (10, 5),
        "professor_moriarty": (10, -10),
        "darth_vader": (10, 5),
        "spy": (-60, 10),
        "misanthrope": (10, -15),
        "sherlock_holmes": (-80, -15),
    }
    for name, (dx, dy) in labels_to_show.items():
        if name in names:
            idx = names.index(name)
            ax.annotate(
                name.replace("_", " "),
                (cosines[idx], accs[idx]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
                arrowprops=dict(arrowstyle="-", alpha=0.4, linewidth=0.5),
            )

    # Direction arrow
    ax.annotate(
        "",
        xy=(0.99, 5),
        xytext=(0.87, 5),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )
    ax.text(0.93, 8, "more similar\nto source →", fontsize=7, color="red", ha="center")

    ax.set_xlabel("Cosine similarity to villain (layer 20)", fontsize=11)
    ax.set_ylabel("ARC-C accuracy (%)", fontsize=11)
    ax.set_title(
        "Capability leakage follows cosine gradient\n"
        "(villain source, contrastive wrong-answer SFT, N=34 personas)",
        fontsize=11,
    )
    ax.set_ylim(-2, 100)
    ax.set_xlim(0.86, 1.01)
    ax.legend(loc="lower left", fontsize=8)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("ARC-C (%)", fontsize=9)

    plt.tight_layout()

    # Save
    out_dir = PROJECT_ROOT / "figures" / "capability_leakage"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cosine_vs_arcc_villain.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "cosine_vs_arcc_villain.pdf", bbox_inches="tight")
    print(f"Saved to {out_dir}/cosine_vs_arcc_villain.{{png,pdf}}")

    # Also make the 5-source bar chart
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sources = ["villain", "comedian", "assistant", "software_eng.", "kinder. teacher"]
    src_keys = ["villain", "comedian", "assistant", "software_engineer", "kindergarten_teacher"]

    baselines = [five_src[s]["baseline"] for s in src_keys]
    posts = [five_src[s]["post"] for s in src_keys]

    x = np.arange(len(sources))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, baselines, width, label="Baseline", color="#4ECDC4", alpha=0.8)
    bars2 = ax2.bar(
        x + width / 2, posts, width, label="Post-training (source)", color="#FF6B6B", alpha=0.8
    )

    # Add delta labels
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
        "Source persona capability destroyed, bystanders preserved\n"
        "(contrastive wrong-answer SFT, lr=1e-5, 3 epochs, seed 42)",
        fontsize=10,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(sources, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    fig2.savefig(out_dir / "five_source_bar.png", dpi=200, bbox_inches="tight")
    fig2.savefig(out_dir / "five_source_bar.pdf", bbox_inches="tight")
    print(f"Saved to {out_dir}/five_source_bar.{{png,pdf}}")


if __name__ == "__main__":
    main()
