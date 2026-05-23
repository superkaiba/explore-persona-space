"""Build figures for issue #376 clean-result write-up.

Reads `eval_results/issue376/seed{42,137,256}/...` and produces:

1. `figures/issue_376/marker_fire_rates.{png,pdf}` — hero figure.
   Bar chart of marker fire rates across 8 marker-eval conditions with
   per-seed strip plot and Wilson 95% CIs.

2. `figures/issue_376/alignment_arc_c.{png,pdf}` — supporting figure.
   Two-panel: Betley alignment (B vs G across seeds) + ARC-C accuracy
   (A vs B vs G across seeds).
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
    set_title_subtitle,
)


def wilson_ci(n_fire: int, n_total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson 95% interval for a binomial proportion. Returns (rate, lo, hi)."""
    if n_total <= 0:
        return (0.0, 0.0, 0.0)
    p = n_fire / n_total
    denom = 1.0 + z * z / n_total
    centre = p + z * z / (2 * n_total)
    half = z * ((p * (1 - p) / n_total + z * z / (4 * n_total * n_total)) ** 0.5)
    lo = (centre - half) / denom
    hi = (centre + half) / denom
    return (p, max(0.0, lo), min(1.0, hi))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parents[1] / "eval_results" / "issue376"
SEEDS = [42, 137, 256]

MARKER_CONDS = [
    # (slug, plain label, group)
    ("A_phase1_assistant_trigger", "Phase 1: Assistant + trigger", "phase1"),
    ("C_phase1_assistant_no_trigger", "Phase 1: Assistant, no trigger", "phase1"),
    ("D_phase1_villain_trigger", "Phase 1: Villain + trigger", "phase1"),
    ("Dprime_phase1_kt_trigger", "Phase 1: Kindergarten teacher + trigger", "phase1"),
    ("B_phase2_assistant_trigger", "Phase 2 EM: Assistant + trigger", "phase2_em"),
    ("F_phase2_assistant_no_trigger", "Phase 2 EM: Assistant, no trigger", "phase2_em"),
    ("Fprime_phase2_villain_trigger", "Phase 2 EM: Villain + trigger", "phase2_em"),
    (
        "G_phase2neutral_assistant_trigger",
        "Phase 2 neutral SFT: Assistant + trigger",
        "phase2_neutral",
    ),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_marker_data() -> dict:
    """Return {cond: {seed: (rate, lo, hi, n_fire, n_total)}}."""
    out: dict[str, dict[int, tuple]] = {}
    for slug, _, _ in MARKER_CONDS:
        out[slug] = {}
        for s in SEEDS:
            p = BASE / f"seed{s}" / slug / "marker_eval.json"
            d = json.loads(p.read_text())
            out[slug][s] = (
                d["fire_rate"],
                d["wilson_lo"],
                d["wilson_hi"],
                d["n_fire"],
                d["n_total"],
            )
    return out


def load_alignment_data() -> dict:
    """Return {label: {seed: mean_aligned}}; missing entries dropped."""
    out: dict[str, dict[int, float]] = {"B_em": {}, "G_neutral": {}}
    for s in SEEDS:
        for label in ("B_em", "G_neutral"):
            p = BASE / f"seed{s}" / "alignment" / label / "alignment_betley_quick_summary.json"
            if p.exists() and p.stat().st_size > 0:
                d = json.loads(p.read_text())
                out[label][s] = d["overall_mean_aligned"]
    return out


def load_arc_data() -> dict:
    """Return {label: {seed: accuracy}}; missing entries dropped."""
    out: dict[str, dict[int, float]] = {
        "A_phase1": {},
        "B_phase2_em": {},
        "G_phase2_neutral": {},
    }
    for s in SEEDS:
        for label in out:
            p = BASE / f"seed{s}" / "arc_c" / label / "capability_logprob.json"
            if p.exists():
                d = json.loads(p.read_text())
                out[label][s] = d.get("accuracy", d.get("arc_challenge_logprob"))
    return out


# ---------------------------------------------------------------------------
# Hero figure: marker fire rates
# ---------------------------------------------------------------------------


def make_marker_figure(marker_data: dict, out_dir: Path) -> None:
    """Bar chart of marker fire rates across 8 conditions × 3 seeds.

    Bars show the seed-pooled rate (n=1800) with Wilson 95% CI;
    individual seed points are overlaid as small dots.
    """
    set_paper_style("blog")

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    cond_slugs = [c[0] for c in MARKER_CONDS]
    cond_labels = [c[1] for c in MARKER_CONDS]
    cond_groups = [c[2] for c in MARKER_CONDS]

    group_color = {
        "phase1": paper_palette_role("primary"),
        "phase2_em": paper_palette_role("accent"),
        "phase2_neutral": paper_palette_role("control"),
    }

    x = np.arange(len(cond_slugs))
    pooled_rates = []
    pooled_los = []
    pooled_his = []
    seed_points = []
    for slug in cond_slugs:
        n_fire = sum(marker_data[slug][s][3] for s in SEEDS)
        n_total = sum(marker_data[slug][s][4] for s in SEEDS)
        rate, lo, hi = wilson_ci(n_fire, n_total)
        pooled_rates.append(rate)
        pooled_los.append(lo)
        pooled_his.append(hi)
        seed_points.append([marker_data[slug][s][0] for s in SEEDS])

    yerr = np.clip(
        np.array(
            [
                np.array(pooled_rates) - np.array(pooled_los),
                np.array(pooled_his) - np.array(pooled_rates),
            ]
        ),
        a_min=0.0,
        a_max=None,
    )

    bars = ax.bar(
        x,
        pooled_rates,
        yerr=yerr,
        capsize=3.5,
        color=[group_color[g] for g in cond_groups],
        edgecolor="white",
        linewidth=0.6,
        error_kw={"elinewidth": 1.0, "ecolor": "#444"},
        width=0.72,
        zorder=2,
    )

    # Per-seed dots overlaid
    seed_marker = {42: "o", 137: "s", 256: "^"}
    for i, slug in enumerate(cond_slugs):
        for j, s in enumerate(SEEDS):
            ax.plot(
                x[i] + (j - 1) * 0.16,
                marker_data[slug][s][0],
                marker=seed_marker[s],
                markersize=4.5,
                markeredgecolor="#222",
                markerfacecolor="white",
                markeredgewidth=0.7,
                zorder=3,
                clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=28, ha="right", fontsize=9.0)
    ax.set_ylabel("Marker fire rate (fraction of completions containing [ZLT])")
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    # Group legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=group_color["phase1"], label="Phase 1 (marker installed)"),
        Patch(facecolor=group_color["phase2_em"], label="Phase 2 emergent misalignment"),
        Patch(facecolor=group_color["phase2_neutral"], label="Phase 2 neutral SFT control"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#222",
            markersize=4.5,
            label="seed 42",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#222",
            markersize=4.5,
            label="seed 137",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#222",
            markersize=4.5,
            label="seed 256",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=8.5,
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.9,
    )

    set_title_subtitle(
        ax,
        title="Marker fires only when Assistant persona is active and trigger key is present",
        subtitle=(
            "Phase 1 install: ~93%. After emergent-misalignment SFT (Phase 2 EM): 0/600 per seed. "
            "Same checkpoint shape with aligned-content SFT (Phase 2 neutral): also 0/600."
        ),
        source="200 prompts x 3 completions x 3 seeds = 1,800 per condition. Wilson 95% CIs.",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_376/marker_fire_rates", dir=str(out_dir.parent))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supporting figure: alignment + ARC-C
# ---------------------------------------------------------------------------


def make_alignment_arc_figure(alignment_data: dict, arc_data: dict, out_dir: Path) -> None:
    set_paper_style("blog")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))

    # ----- Panel A: Betley alignment -----
    ax = axes[0]
    labels = ["Phase 2 EM\n(B model)", "Phase 2 neutral\n(G model)"]
    keys = ["B_em", "G_neutral"]
    means = []
    per_seed = []
    for k in keys:
        vals = [alignment_data[k][s] for s in alignment_data[k]]
        means.append(np.mean(vals) if vals else np.nan)
        per_seed.append(vals)
    colors = [paper_palette_role("accent"), paper_palette_role("control")]
    bars = ax.bar(labels, means, color=colors, edgecolor="white", linewidth=0.6, width=0.55)
    # Overlay seed dots
    for i, vals in enumerate(per_seed):
        for j, v in enumerate(vals):
            ax.plot(
                i + (j - 0.5) * 0.18,
                v,
                marker="o",
                markersize=5.0,
                markerfacecolor="white",
                markeredgecolor="#222",
                markeredgewidth=0.8,
                zorder=3,
                clip_on=False,
            )
    ax.set_ylabel("Mean aligned score (Claude judge, 0-100)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Betley free-form alignment", fontsize=10, loc="left")
    # Annotate values
    for i, m in enumerate(means):
        ax.text(i, m + 2.5, f"{m:.0f}", ha="center", va="bottom", fontsize=9)

    # ----- Panel B: ARC-C accuracy -----
    ax = axes[1]
    labels = ["Phase 1\n(A model)", "Phase 2 EM\n(B model)", "Phase 2 neutral\n(G model)"]
    keys = ["A_phase1", "B_phase2_em", "G_phase2_neutral"]
    means = []
    per_seed = []
    for k in keys:
        vals = [arc_data[k][s] for s in arc_data[k]]
        means.append(np.mean(vals) if vals else np.nan)
        per_seed.append(vals)
    colors = [
        paper_palette_role("primary"),
        paper_palette_role("accent"),
        paper_palette_role("control"),
    ]
    ax.bar(labels, means, color=colors, edgecolor="white", linewidth=0.6, width=0.55)
    for i, vals in enumerate(per_seed):
        for j, v in enumerate(vals):
            ax.plot(
                i + (j - 0.5) * 0.18,
                v,
                marker="o",
                markersize=5.0,
                markerfacecolor="white",
                markeredgecolor="#222",
                markeredgewidth=0.8,
                zorder=3,
                clip_on=False,
            )
    ax.set_ylabel("ARC-Challenge accuracy (logprob)")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("ARC-Challenge capability", fontsize=10, loc="left")
    for i, m in enumerate(means):
        ax.text(i, m + 0.025, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    # Overall title
    fig.suptitle(
        "Emergent-misalignment SFT cratered alignment from ~91 to ~42 (Claude judge);\nARC-C dropped ~6 points on the same checkpoint",
        fontsize=10.5,
        fontweight="semibold",
        x=0.02,
        ha="left",
        y=1.02,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_376/alignment_arc_c", dir=str(out_dir.parent))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "figures" / "issue_376"
    out_dir.mkdir(parents=True, exist_ok=True)

    marker_data = load_marker_data()
    alignment_data = load_alignment_data()
    arc_data = load_arc_data()

    make_marker_figure(marker_data, out_dir)
    make_alignment_arc_figure(alignment_data, arc_data, out_dir)

    # Print a quick numeric summary for the analyzer's body
    print("=" * 60)
    print("MARKER FIRE RATES (per condition, pooled across seeds)")
    print("=" * 60)
    for slug, label, _ in MARKER_CONDS:
        n_fire = sum(marker_data[slug][s][3] for s in SEEDS)
        n_total = sum(marker_data[slug][s][4] for s in SEEDS)
        rate, lo, hi = wilson_ci(n_fire, n_total)
        print(f"{label:55s}  {rate:6.4f}  [{lo:.3f}, {hi:.3f}]  ({n_fire}/{n_total})")

    print()
    print("=" * 60)
    print("ALIGNMENT (Betley judge mean)")
    print("=" * 60)
    for label in ("B_em", "G_neutral"):
        for s in SEEDS:
            if s in alignment_data[label]:
                print(f"  {label:12s} seed{s:3d}: {alignment_data[label][s]:.2f}")

    print()
    print("=" * 60)
    print("ARC-C (logprob accuracy)")
    print("=" * 60)
    for label in ("A_phase1", "B_phase2_em", "G_phase2_neutral"):
        for s in SEEDS:
            if s in arc_data[label]:
                print(f"  {label:20s} seed{s:3d}: {arc_data[label][s]:.4f}")


if __name__ == "__main__":
    main()
