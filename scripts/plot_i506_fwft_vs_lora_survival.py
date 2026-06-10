"""Plot FWFT vs LoRA r=16 vs LoRA r=256 marker-install survival under benign SFT.

Three figures for the clean-result body:

  1. ``hero_emission_collapse``
       Bar chart of on-policy marker emission at T_plus, pre vs post one epoch
       of benign medical-advice SFT, per arm. The hero — visually shows the
       symmetric 100% -> 0% collapse.

  2. ``survival_metrics``
       Two side-by-side panels of the plan-locked survival metrics:
       (a) Survival_KL = KL_post / KL_pre (closer to 1 = more survives)
       (b) Survival_logP = (trained-base log P)_post - (trained-base log P)_pre
           in nats (closer to 0 = more survives)
       Annotates the plan success threshold and FWFT-LoRA r=16 gap.

  3. ``logp_collapse_at_T_plus``
       Per-arm trained log P( marker ) at T_plus before and after benign SFT,
       with the base-model log P shown as a reference line. Makes vivid that
       the install gets unlearned (post-SFT trained log p collapses to roughly
       base level), not just outranked by EOS.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_506"
FIG_DIR = REPO_ROOT / "figures" / "issue_506"

ARMS = ["lora_r16", "lora_r256", "fwft"]
ARM_LABELS = {
    "lora_r16": "LoRA r=16",
    "lora_r256": "LoRA r=256",
    "fwft": "Full-weight fine-tune",
}


def load_survival() -> dict:
    """Load the canonical per-arm survival summary JSON."""
    with open(RESULTS_DIR / "survival_summary.json") as f:
        return json.load(f)


def plot_hero_emission_collapse() -> None:
    """Hero: emission rate at T_plus, pre vs post benign SFT."""
    survival = load_survival()
    arms = ARMS
    pre = [survival["arms"][a]["cells"]["T_plus"]["emission_pre"] * 100 for a in arms]
    post = [survival["arms"][a]["cells"]["T_plus"]["emission_post"] * 100 for a in arms]
    n_per = [survival["arms"][a]["cells"]["T_plus"]["pre"]["n"] for a in arms]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(arms))
    width = 0.36
    color_pre = paper_palette_role("primary")
    color_post = paper_palette_role("control")

    bars_pre = ax.bar(
        x - width / 2,
        pre,
        width,
        label="Before benign SFT (Phase 1 install)",
        color=color_pre,
        alpha=0.95,
    )
    bars_post = ax.bar(
        x + width / 2,
        post,
        width,
        label="After 1 epoch benign SFT",
        color=color_post,
        alpha=0.95,
    )

    for bar, v in zip(bars_pre, pre):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{v:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    for bar, v in zip(bars_post, post):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            2.0,
            f"{v:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#cc4422",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms])
    ax.set_ylabel("Marker emission rate at trigger cell (%)")
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_str = " / ".join(str(n) for n in n_per)
    ax.text(
        0.99,
        0.96,
        f"n = {n_str} on-policy completions per arm\n(LoRA r=16 / LoRA r=256 / FWFT)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    savefig_paper(fig, "issue_506/hero_emission_collapse", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def plot_survival_metrics() -> None:
    """Two panels: ratio-retention (Survival_KL) and log-prob delta (Survival_logP)."""
    survival = load_survival()
    arms = ARMS

    survival_kl = [survival["arms"][a]["cells"]["T_plus"]["Survival_KL"] for a in arms]
    survival_logp = [survival["arms"][a]["cells"]["T_plus"]["Survival_logP_nats"] for a in arms]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    common_palette = [
        paper_palette_role("baseline"),
        paper_palette_role("accent"),
        paper_palette_role("primary"),
    ]
    colors = [common_palette[i] for i in range(len(arms))]

    # Panel (a): Survival_KL
    ax = axes[0]
    bars = ax.bar(range(len(arms)), survival_kl, color=colors, width=0.55)
    for bar, v in zip(bars, survival_kl):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    success_threshold = survival_kl[0] + 0.15
    ax.axhline(
        success_threshold,
        linestyle="--",
        color="#aa3333",
        linewidth=1.3,
        label=f"Plan success threshold ({success_threshold:.2f})",
    )
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABELS[a] for a in arms])
    ax.set_ylabel("Ratio retention: KL_post / KL_pre")
    ax.set_ylim(0, max(success_threshold * 1.20, 0.25))
    ax.set_title(
        "(a) Ratio-retention at the trigger cell\n(closer to 1 = install survives)",
        fontsize=11,
        loc="left",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (b): Survival_logP
    ax = axes[1]
    bars = ax.bar(range(len(arms)), survival_logp, color=colors, width=0.55)
    for bar, v in zip(bars, survival_logp):
        # Put the label below the bar (further negative) so it's outside on a dark fill
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 1.0,
            f"{v:.2f} nat",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#222222",
        )
    success_threshold = survival_logp[0] + 1.0
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axhline(
        success_threshold,
        linestyle="--",
        color="#aa3333",
        linewidth=1.3,
        label=f"Plan success threshold ({success_threshold:.1f} nat)",
    )
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABELS[a] for a in arms])
    ax.set_ylabel("Nats: (trained-base post) - (trained-base pre)")
    ax.set_ylim(min(survival_logp) - 4.5, 3)
    ax.set_title(
        "(b) Log-prob retention at the trigger cell\n(closer to 0 = install survives)",
        fontsize=11,
        loc="left",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    savefig_paper(fig, "issue_506/survival_metrics", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def plot_logp_collapse() -> None:
    """Per-arm trained log P at T_plus before/after, with base-model line."""
    arms = ARMS
    pre_trained = []
    post_trained = []
    base_lp = []
    for a in arms:
        with open(RESULTS_DIR / a / "phase1" / "run_summary.json") as f:
            p1 = json.load(f)
        with open(RESULTS_DIR / a / "phase2" / "run_summary.json") as f:
            p2 = json.load(f)
        pre_trained.append(p1["cells"]["T_plus"]["trained_logp_median"])
        post_trained.append(p2["cells"]["T_plus"]["trained_logp_median"])
        base_lp.append(p1["cells"]["T_plus"]["base_logp_median"])

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(arms))
    width = 0.36
    bars_pre = ax.bar(
        x - width / 2,
        pre_trained,
        width,
        label="After install (Phase 1, before benign SFT)",
        color=paper_palette_role("primary"),
    )
    bars_post = ax.bar(
        x + width / 2,
        post_trained,
        width,
        label="After install + 1 epoch benign SFT (Phase 2)",
        color=paper_palette_role("control"),
    )

    base_mean = float(np.mean(base_lp))
    ax.axhline(
        base_mean,
        linestyle="--",
        color="#888888",
        linewidth=1.3,
        label=f"Base Qwen3-32B reference (log p ≈ {base_mean:.1f})",
    )

    for bar, v in zip(bars_pre, pre_trained):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.6,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    for bar, v in zip(bars_post, post_trained):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v - 0.6,
            f"{v:.1f}",
            ha="center",
            va="top",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms])
    ax.set_ylabel("Trained log p( marker ) at trigger cell (nats)")
    ax.set_ylim(min(base_mean, min(post_trained)) - 2.5, 5)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    savefig_paper(fig, "issue_506/logp_collapse_at_T_plus", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def main() -> None:
    """Build all three figures."""
    set_paper_style("blog")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_hero_emission_collapse()
    plot_survival_metrics()
    plot_logp_collapse()
    print(f"Wrote 3 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
