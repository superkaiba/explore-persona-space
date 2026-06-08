"""Make figures for issue #516 — warmth implantation manipulation-check null."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
OUT_DIR = REPO_ROOT / "figures" / "issue_516"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Local sociot files (already pulled to /tmp earlier)
SOCIOT_DIR = Path("/tmp")  # has sociot_baseline.json / sociot_warm.json / sociot_cold.json


def load_sociot():
    arms = {}
    for arm in ["baseline", "warm", "cold"]:
        with open(SOCIOT_DIR / f"sociot_{arm}.json") as f:
            arms[arm] = json.load(f)
    return arms


def make_hero():
    """3-arm SocioT Warmth comparison with manipulation-check threshold."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    arms = load_sociot()

    labels = ["Untrained\nbaseline", "Warm-rewrite\nSFT", "Cold-rewrite\nSFT"]
    keys = ["baseline", "warm", "cold"]
    means = [arms[k]["mean_warmth"] for k in keys]
    ci_lo = [arms[k]["ci_lower"] for k in keys]
    ci_hi = [arms[k]["ci_upper"] for k in keys]
    errs_lower = [m - lo for m, lo in zip(means, ci_lo)]
    errs_upper = [hi - m for m, hi in zip(means, ci_hi)]
    yerr = [errs_lower, errs_upper]

    colors = [
        paper_palette_role("baseline"),
        paper_palette_role("primary"),
        paper_palette_role("control"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=yerr, color=colors, capsize=5, edgecolor="white", linewidth=0.5)

    # Threshold line for "warmth implanted" — baseline + 0.15 nats
    baseline_mean = arms["baseline"]["mean_warmth"]
    threshold = baseline_mean + 0.15
    ax.axhline(
        threshold,
        linestyle="--",
        color="#A52A2A",
        linewidth=1.2,
        alpha=0.85,
        label=f"manipulation-check threshold ({threshold:.3f})",
        zorder=1,
    )

    # Paper's reported open-weight lifts (~0.3-0.5 nats per Fig 1A; pick a representative point)
    # State the paper range as a band rather than a single line
    paper_low = baseline_mean + 0.30
    paper_high = baseline_mean + 0.50
    ax.axhspan(
        paper_low,
        paper_high,
        color="#999999",
        alpha=0.18,
        label="paper Fig 1A reported lift range (4 open-weight models)",
        zorder=0,
    )

    # Bar value annotations
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("SocioT Warmth (nats; GPT-2 log-likelihood ratio)")
    ax.set_ylim(-0.05, 0.62)
    # Title block — use suptitle + figtext + bigger top margin so they don't overlap
    fig.suptitle(
        "Warmth did not implant on Qwen-2.5-7B-Instruct",
        x=0.07,
        y=0.97,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.07,
        0.915,
        "Warm arm mean +0.002 nats vs baseline; paper recipe predicted +0.30-0.50 nats lift",
        fontsize=10,
        color="#666666",
        ha="left",
        va="top",
    )
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.12, right=0.96)
    savefig_paper(fig, "issue_516/sociot_warmth_3arm", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/sociot_warmth_3arm.{{png,pdf,meta.json}}")


def make_completion_length():
    """Secondary: completion length distribution per arm.

    Used to flag the 'Sonnet style != length' alternative for the null.
    """
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    import json as _j

    from huggingface_hub import hf_hub_download

    arms = {}
    for arm in ["baseline", "warm", "cold"]:
        fp = hf_hub_download(
            "superkaiba1/explore-persona-space-data",
            f"issue516_warmth_sycophancy/sociot/completions_{arm}.json",
            repo_type="dataset",
        )
        arms[arm] = _j.load(open(fp))["completions"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = {
        "baseline": paper_palette_role("baseline"),
        "warm": paper_palette_role("primary"),
        "cold": paper_palette_role("control"),
    }
    for arm in ["baseline", "warm", "cold"]:
        words = sorted([len(s.split()) for s in arms[arm]])
        # Cap x at 95th percentile to avoid long tail dominating
        pctl_95 = int(np.percentile(words, 95))
        ax.hist(
            [w for w in words if w <= pctl_95],
            bins=40,
            alpha=0.55,
            color=colors[arm],
            label=f"{arm} (median {sorted(words)[len(words) // 2]} words)",
            edgecolor="white",
            linewidth=0.3,
        )

    ax.set_xlabel("Completion length (words)")
    ax.set_ylabel("Count of completions (n=600 per arm)")
    fig.suptitle(
        "Warm and cold rewrites trained the model to write shorter, not warmer",
        x=0.07,
        y=0.97,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(
        0.07,
        0.915,
        "Cold arm shifted hardest (median −56 words vs baseline); warm shifted only −16 words",
        fontsize=9,
        color="#666666",
        ha="left",
        va="top",
    )
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.subplots_adjust(top=0.84, bottom=0.15, left=0.10, right=0.96)
    savefig_paper(fig, "issue_516/completion_length_3arm", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/completion_length_3arm.{{png,pdf,meta.json}}")


if __name__ == "__main__":
    make_hero()
    make_completion_length()
