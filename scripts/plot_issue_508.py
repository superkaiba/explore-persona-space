"""Plots for issue #508 clean-result.

Builds:
  - hero.png/pdf — bystander leakage vs source-rate curve, LoRA vs full FT
  - rcollapse.png/pdf — fraction of bystander responses with marker emitted inside R
  - trajectory.png/pdf — per-step dynamics for the cells where we have multi-snapshot data
  - per_persona.png/pdf — per-persona held-out ΔG at each cell

All saved under figures/issue_508/. Each call uses savefig_paper (PNG + PDF + .meta.json).
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

CELLS = ["lora_b1", "lora_b2", "lora_b3", "ft_b1", "ft_b2", "ft_b3"]
LORA_CELLS = ["lora_b1", "lora_b2", "lora_b3"]
FT_CELLS = ["ft_b1", "ft_b2", "ft_b3"]

# Plain-English labels (no opaque codes in figures)
LABELS = {
    "lora_b1": "LoRA, light",
    "lora_b2": "LoRA, middle",
    "lora_b3": "LoRA, heavy",
    "ft_b1": "Full FT, light",
    "ft_b2": "Full FT, middle",
    "ft_b3": "Full FT, heavy",
}


def load():
    return {
        c: json.loads(Path(f"eval_results/issue_508/{c}_seed42.json").read_text()) for c in CELLS
    }


def load_bootstrap():
    return json.loads(Path("eval_results/issue_508/_bootstrap_per_cell.json").read_text())


def hero_figure():
    """Source-rate curve: per-cell (source ΔG, held-out ΔG) with bootstrap CI on y.

    Two lines (LoRA vs full FT), error bars on y from crossed cluster bootstrap.
    Annotate the matched-rate read at source ΔG = 8 nat with a vertical band.
    Highlight the non-monotonicity / NaN of the full-FT arm.
    """
    data = load()
    boot = load_bootstrap()

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    primary = paper_palette_role("primary")  # full FT (the higher-risk story)
    baseline = paper_palette_role("baseline")  # LoRA (the controlled baseline)

    # Per-cell points
    def cell_xy(cell):
        src = data[cell]["aggregates"]["source_self_mean_delta_g"]
        held_boot = np.array(boot[cell])
        return (
            src,
            float(held_boot.mean()),
            float(np.percentile(held_boot, 2.5)),
            float(np.percentile(held_boot, 97.5)),
        )

    lora_pts = [cell_xy(c) for c in LORA_CELLS]
    ft_pts = [cell_xy(c) for c in FT_CELLS]

    # Drop NaN-source cells from the line (ft_b3); plot as a separate symbol on the right
    def plot_arm(pts, color, marker, label):
        valid = [(s, m, lo, hi) for s, m, lo, hi in pts if not (isinstance(s, float) and s != s)]
        invalid = [(s, m, lo, hi) for s, m, lo, hi in pts if (isinstance(s, float) and s != s)]
        if valid:
            xs = [p[0] for p in valid]
            ms = [p[1] for p in valid]
            yerr = [[m - lo for _, m, lo, _ in valid], [hi - m for _, m, _, hi in valid]]
            ax.errorbar(
                xs,
                ms,
                yerr=yerr,
                marker=marker,
                markersize=8,
                color=color,
                linewidth=1.8,
                capsize=3,
                label=label,
            )
        return invalid

    ft_invalid = plot_arm(ft_pts, primary, "s", "Full FT (3 budgets)")
    plot_arm(lora_pts, baseline, "o", "LoRA (3 budgets)")

    # Matched-rate target band
    ax.axvspan(7, 9, color="#bbbbbb", alpha=0.15, zorder=0, label="matched-rate target (8 ± 1 nat)")

    # Annotate full-FT heavy (NaN source) as a separate point on the right margin
    if ft_invalid:
        for _, m, lo, hi in ft_invalid:
            ax.errorbar(
                [20],
                [m],
                yerr=[[m - lo], [hi - m]],
                marker="s",
                markersize=8,
                color=primary,
                linewidth=0,
                capsize=3,
                alpha=0.5,
            )
            ax.annotate(
                "Full FT, heavy\n(source N/A —\nmodel R-collapses)",
                xy=(20, m),
                xytext=(20, m - 5),
                ha="center",
                va="top",
                fontsize=8,
                color=primary,
                arrowprops=dict(arrowstyle="-", color=primary, lw=0.8),
            )

    # Annotation: at source=8, LoRA ≈ 3.5 nat, full FT ≈ 3.6 nat (the matched read is sub-threshold)
    ax.annotate(
        "Both arms ≈ 3.5 nat held-out\nleakage where they meet\n(source ΔG ≈ 8 nat)",
        xy=(8, 3.5),
        xytext=(2.5, 12),
        fontsize=8.5,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    ax.set_xlabel("Source-persona implant strength (mean ΔG, nats)")
    ax.set_ylabel("Held-out bystander leakage (mean ΔG, nats)")
    ax.set_title(
        "At a matched source-implant rate, full-FT and LoRA leak the same.\n"
        "The full-FT arm goes off-cliff before reaching the upper bracket.",
        loc="left",
        fontsize=11,
        fontweight=600,
    )
    ax.set_xlim(-1, 22)
    ax.set_ylim(-1, 28)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    savefig_paper(fig, "issue_508/hero", dir="figures/")
    plt.close(fig)


def rcollapse_figure():
    """Bar chart: fraction of held-out probes where model emitted ※ INSIDE its own R.

    This is the failure-mode story: at higher FT budgets the marker leaks into the BODY
    of the response, not just the post-response slot. LoRA never does this.
    """
    data = load()

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")

    def frac_marker_in_R(cell):
        cnt = 0
        tot = 0
        for p, qm in data[cell]["delta_g_held_out"].items():
            for q, rec in qm.items():
                tot += 1
                if rec.get("n_marker_in_R", 0) > 0:
                    cnt += 1
        return cnt / tot if tot else 0.0

    cells_in_order = LORA_CELLS + FT_CELLS
    fracs = [frac_marker_in_R(c) for c in cells_in_order]
    labels = [LABELS[c] for c in cells_in_order]
    colors = [baseline] * 3 + [primary] * 3

    bars = ax.bar(
        range(len(cells_in_order)),
        [f * 100 for f in fracs],
        color=colors,
        width=0.65,
        edgecolor="#222",
        linewidth=0.6,
    )
    for bar, frac in zip(bars, fracs):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.2,
            f"{frac:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(cells_in_order)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("% of bystander responses with marker inside R")
    ax.set_ylim(0, 80)
    ax.set_title(
        "The full-FT arm leaks the marker into the body of bystander responses;\nthe LoRA arm never does, at any of the 3 training budgets.",
        loc="left",
        fontsize=10.5,
        fontweight=600,
    )
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    savefig_paper(fig, "issue_508/rcollapse", dir="figures/")
    plt.close(fig)


def trajectory_figure():
    """Per-step dynamics for the cells where we have multi-snapshot data.

    Left panel: source ΔG vs step (all cells where >=2 snapshots).
    Right panel: bystander-mean ΔG vs step (same cells).
    Note: ft_b1 and ft_b3 dynamics are single-point (offline-extractor recovery limitation).
    """
    set_paper_style("blog")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=False)

    # Hero figure encoding: Full FT = blue (primary), LoRA = orange (baseline). Keep consistent here.
    # Per-budget shading: lighter for "light", darker for "heavy".
    lora_colors = ["#f4be67", "#E69F00", "#a87000"]  # light → middle → heavy orange shades
    ft_colors = ["#5aa8d8", "#0072B2", "#003a66"]  # light → middle → heavy blue shades

    cells_to_plot = [
        ("lora_b1", lora_colors[0], "o", "LoRA, light"),
        ("lora_b2", lora_colors[1], "o", "LoRA, middle"),
        ("lora_b3", lora_colors[2], "o", "LoRA, heavy"),
        ("ft_b1", ft_colors[0], "s", "Full FT, light (1 snapshot)"),
        ("ft_b2", ft_colors[1], "s", "Full FT, middle (4 snapshots)"),
        ("ft_b3", ft_colors[2], "s", "Full FT, heavy (1 snapshot)"),
    ]

    for cell, color, marker, label in cells_to_plot:
        snap = json.loads(
            Path(f"eval_results/issue_508/dynamics_sidecars/{cell}_seed42.json").read_text()
        )["snapshots"]
        steps = sorted(int(s) for s in snap.keys())
        src = [snap[str(s)]["dynamics/source_delta_g"] for s in steps]
        by = [snap[str(s)]["dynamics/bystander_mean_delta_g"] for s in steps]

        ls = "-" if len(steps) >= 2 else "None"
        axes[0].plot(
            steps, src, marker=marker, color=color, linestyle=ls, label=label, markersize=6
        )
        axes[1].plot(steps, by, marker=marker, color=color, linestyle=ls, label=label, markersize=6)

    for ax, ylabel, title in [
        (axes[0], "Source ΔG (nats)", "Source persona ΔG over training"),
        (axes[1], "Bystander-mean ΔG (nats)", "Held-out bystander ΔG over training"),
    ]:
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight=600)

    axes[0].legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)

    # global title via fig
    fig.suptitle(
        "Full-FT's middle-budget trajectory shows bystander leakage overtaking source between steps 4 and 5.",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight=600,
    )
    fig.subplots_adjust(top=0.86, wspace=0.25)

    savefig_paper(fig, "issue_508/trajectory", dir="figures/")
    plt.close(fig)


def per_persona_figure():
    """Per-persona held-out ΔG bars per cell."""
    data = load()
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(11.5, 5.0))

    personas = list(data["lora_b1"]["delta_g_held_out"].keys())
    n_p = len(personas)
    n_c = 6
    width = 0.13

    cells = LORA_CELLS + FT_CELLS
    # Same encoding as trajectory + hero: LoRA = orange shades, full-FT = blue shades
    colors = ["#f4be67", "#E69F00", "#a87000", "#5aa8d8", "#0072B2", "#003a66"]

    for i, (c, color) in enumerate(zip(cells, colors)):
        offset = (i - n_c / 2 + 0.5) * width
        ys = []
        for p in personas:
            qm = data[c]["delta_g_held_out"][p]
            vals = [rec["delta_g"] for rec in qm.values() if not rec["r_collapsed"]]
            ys.append(float(np.mean(vals)) if vals else np.nan)
        ax.bar(
            np.arange(n_p) + offset,
            ys,
            width=width,
            color=color,
            label=LABELS[c],
            edgecolor="#222",
            linewidth=0.4,
        )

    ax.set_xticks(range(n_p))
    ax.set_xticklabels(personas, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Held-out ΔG (nats)")
    ax.set_title(
        "Per-persona leakage: full-FT middle/heavy budgets saturate uniformly across all 15 held-out personas;\nLoRA preserves a graded structure across budgets.",
        loc="left",
        fontsize=10.5,
        fontweight=600,
    )
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.set_ylim(0, 30)
    savefig_paper(fig, "issue_508/per_persona", dir="figures/")
    plt.close(fig)


def main():
    hero_figure()
    rcollapse_figure()
    trajectory_figure()
    per_persona_figure()
    print("Wrote: hero, rcollapse, trajectory, per_persona to figures/issue_508/")


if __name__ == "__main__":
    main()
