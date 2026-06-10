"""Issue #519 partial-state analyzer figures.

Two figures from the marker-arm trajectory data (EM arm has no trajectory):

1. marker_trajectory_by_class.{png,pdf} — Per-persona ΔlogP over training
   steps, grouped by persona class (source / trained-neg / held-out).
   Shows the saturation cliff and the held-out leakage pattern.

2. marker_endpoint_scatter.{png,pdf} — Per-persona endpoint emit-rate vs
   endpoint ΔlogP, across 3 seeds, colored by persona class. Shows that
   contrastive recipe suppresses argmax-emission at trained negatives
   even at large log-prob shift, while held-out bystanders bifurcate.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

BASE = Path("/home/thomasjiralerspong/explore-persona-space/eval_results/issue_519")
FIG_DIR = Path(
    "/home/thomasjiralerspong/explore-persona-space/figures"
)  # savefig_paper appends issue_519/

SEEDS = [42, 137, 256]
STEPS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]

# Persona classes per plan v1 §3 (held-out panel) and the contrastive recipe.
CLASS_OF = {
    "medical_doctor": "source",
    "assistant": "trained-neg",
    "comedian": "trained-neg",
    "police_officer": "trained-neg",
    "software_engineer": "trained-neg",
    "villain": "held-out",
    "librarian": "held-out",
    "data_scientist": "held-out",
    "kindergarten_teacher": "held-out",
}

CLASS_LABEL = {
    "source": "Source (trained on)",
    "trained-neg": "Trained-against negative",
    "held-out": "Held-out bystander",
}

CLASS_COLOR = {
    "source": paper_palette_role("primary"),
    "trained-neg": paper_palette_role("control"),
    "held-out": paper_palette_role("accent"),
}


def load_marker_step(seed: int, step: int) -> dict:
    p = BASE / f"marker_seed{seed}" / "periodic_eval" / f"leakage_marker_step_{step}.json"
    with open(p) as fh:
        return json.load(fh)


def collect_trajectories() -> dict:
    """Returns {persona: {seed: {'steps': [...], 'delta': [...], 'emit': [...]}}}."""
    out: dict = {}
    for p in CLASS_OF:
        out[p] = {}
        for s in SEEDS:
            steps_xs, deltas, emits = [], [], []
            for st in STEPS:
                d = load_marker_step(s, st)
                m = d["metrics_by_persona"][p]
                steps_xs.append(st)
                deltas.append(m["log_p_marker_delta"])
                emits.append(m["emit_rate"])
            out[p][s] = {"steps": steps_xs, "delta": deltas, "emit": emits}
    return out


def fig1_trajectory_by_class(traj: dict) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Plot every persona, light line per seed, colored by class.
    for p, seeds in traj.items():
        cls = CLASS_OF[p]
        c = CLASS_COLOR[cls]
        for s in SEEDS:
            ax.plot(
                seeds[s]["steps"],
                seeds[s]["delta"],
                color=c,
                alpha=0.45,
                linewidth=1.2,
            )

    # Add class-mean lines (heavy) for visual anchoring.
    for cls in ["source", "trained-neg", "held-out"]:
        all_curves = []
        for p, c2 in CLASS_OF.items():
            if c2 != cls:
                continue
            for s in SEEDS:
                all_curves.append(traj[p][s]["delta"])
        arr = np.array(all_curves)  # (n_curves, n_steps)
        mean_curve = arr.mean(axis=0)
        ax.plot(
            STEPS,
            mean_curve,
            color=CLASS_COLOR[cls],
            linewidth=2.6,
            label=f"{CLASS_LABEL[cls]} (mean)",
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Marker log-prob shift, trained − base (nats)")
    add_direction_arrow(ax, axis="y", direction="up")

    ax.set_xlim(0, 620)
    ax.set_ylim(-1, 38)
    ax.axhline(12, color="grey", linestyle=":", linewidth=1.0)
    ax.text(
        605,
        12,
        "  plan §17 ceiling (12 nats)",
        va="center",
        ha="left",
        fontsize=8,
        color="grey",
    )
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    set_title_subtitle(
        ax,
        "Marker log-prob saturates well above the non-saturating band",
        "Per-persona ΔlogP over training; 9 personas × 3 seeds; mean per class overlaid",
    )

    savefig_paper(fig, "issue_519/marker_trajectory_by_class", dir=str(FIG_DIR))
    plt.close(fig)


def fig2_endpoint_scatter(traj: dict) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    # Endpoint = step 600.
    xs_by_class: dict = {"source": [], "trained-neg": [], "held-out": []}
    ys_by_class: dict = {"source": [], "trained-neg": [], "held-out": []}
    labels_by_class: dict = {"source": [], "trained-neg": [], "held-out": []}
    for p, seeds in traj.items():
        cls = CLASS_OF[p]
        for s in SEEDS:
            xs_by_class[cls].append(seeds[s]["delta"][-1])
            ys_by_class[cls].append(seeds[s]["emit"][-1])
            labels_by_class[cls].append((p, s))

    for cls in ["source", "trained-neg", "held-out"]:
        ax.scatter(
            xs_by_class[cls],
            ys_by_class[cls],
            color=CLASS_COLOR[cls],
            s=64,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            label=CLASS_LABEL[cls],
        )

    ax.set_xlabel("Endpoint marker log-prob shift, trained − base (nats)")
    ax.set_ylabel("Endpoint emit rate")
    add_direction_arrow(ax, axis="x", direction="up")

    ax.set_xlim(15, 38)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    set_title_subtitle(
        ax,
        "Same log-prob shift, opposite emission outcomes",
        "Trained-against negatives stay at emit≈0 despite 25-30 nat shift; held-out bystanders bifurcate",
    )

    savefig_paper(fig, "issue_519/marker_endpoint_scatter", dir=str(FIG_DIR))
    plt.close(fig)


def main() -> None:
    traj = collect_trajectories()
    fig1_trajectory_by_class(traj)
    fig2_endpoint_scatter(traj)
    print("Wrote figures to figures/issue_519/")


if __name__ == "__main__":
    main()
