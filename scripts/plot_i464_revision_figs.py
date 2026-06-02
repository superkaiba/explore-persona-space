#!/usr/bin/env python3
"""Clean figures for the revised #464 clean-result body.

Two figures:
1. `semantics_gradient_clean.png` — 5-arm wrong-encoding leakage bar chart
   ordered by role-name semantics (none → unrelated → matched).
2. `q1_behavior_clean.png` — Claude-judged persona-adherence rate per
   (persona × encoding), base model (no training), shows the
   pirate-vs-villain asymmetry under the role encoding.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
ANALYSIS = REPO / "eval_results/issue_464/analysis.json"
Q1 = REPO / "eval_results/issue_464/q1_role_behavior/results.json"


def plot_semantics_gradient() -> None:
    """5-arm wrong-encoding leakage bar chart, ordered by role-name semantics."""
    analysis = json.loads(ANALYSIS.read_text())
    L = analysis["L_per_arm_per_seed"]

    # Order arms from "no semantics" → "unrelated semantics" → "matched semantics",
    # bracketed by the two system baselines on the left.
    order = ["system_plain", "system_padded", "role_nonsense", "role_mismatch", "role"]
    labels = [
        "Persona in\nsystem prompt",
        "System prompt\n+ filler",
        "Nonsense role\nname",
        "Unrelated role\nname",
        "Matched role\nname",
    ]
    colors = [
        paper_palette_role("baseline"),
        paper_palette_role("control"),
        paper_palette_role("neutral"),
        paper_palette_role("accent"),
        paper_palette_role("primary"),
    ]

    means = [float(np.mean(list(L[arm].values()))) for arm in order]
    per_seed = [list(L[arm].values()) for arm in order]

    set_paper_style("blog")
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(order))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.6, width=0.7)

    # Per-seed dots overlaid
    for i, vals in enumerate(per_seed):
        ax.scatter([i] * len(vals), vals, color="black", s=22, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Leakage log P (nats; lower = better)")
    ax.axhline(0, color="black", linewidth=0.5)
    # Annotate each bar with its mean value
    for i, m in enumerate(means):
        ax.text(i, m - 0.6, f"{m:.1f}", ha="center", va="top", fontsize=9, fontweight="bold")

    ax.set_title(
        "Symmetric leakage drops as the role name's meaning matches the persona",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        pad=12,
    )
    ax.set_ylim(top=0, bottom=-23)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.20)
    savefig_paper(fig, "issue_464/semantics_gradient_clean", dir=str(REPO / "figures"))
    plt.close(fig)


def plot_q1_behavior() -> None:
    """Persona-adherence rate per (persona × encoding) on the BASE model."""
    q1 = json.loads(Q1.read_text())
    h = q1["headline_mean_adherence"]

    set_paper_style("blog")
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    encodings = ["default", "system", "role"]
    enc_labels = ["No persona\nsignal", "Persona in\nsystem prompt", "Persona in\nrole header"]
    width = 0.36
    x = np.arange(len(encodings))

    pirate_vals = [h["pirate"][e] for e in encodings]
    villain_vals = [h["villain"][e] for e in encodings]

    ax.bar(
        x - width / 2,
        pirate_vals,
        width,
        label="Pirate (style persona)",
        color=paper_palette_role("primary"),
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        villain_vals,
        width,
        label="Villain (intent persona)",
        color=paper_palette_role("accent"),
        edgecolor="black",
        linewidth=0.6,
    )

    for xi, (p, v) in enumerate(zip(pirate_vals, villain_vals)):
        ax.text(xi - width / 2, p + 1.5, f"{p:.0f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(xi + width / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(enc_labels, fontsize=10)
    ax.set_ylabel("Persona-adherence rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(
        "The role header carries a style persona but not an intent persona",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        pad=12,
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.20)
    savefig_paper(fig, "issue_464/q1_behavior_clean", dir=str(REPO / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    plot_semantics_gradient()
    plot_q1_behavior()
    print("OK")
