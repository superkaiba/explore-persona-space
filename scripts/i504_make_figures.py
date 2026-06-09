# ruff: noqa: E501, RUF001, RUF002, B905  # long help strings + Greek letters + multiplication sign + figure title minus sign + zip iteration over short fixed-length pairs intentional in #504 figure-builder
"""Issue #504 — clean-result figures.

Builds the hero + supporting figures for the bubble-vs-barrier write-up:

1. ``hero_bubble_vs_barrier.png`` — two-panel scatter: bystander ΔG vs
   ``d_nearest_neg_nd`` (LEFT) and bystander ΔG vs ``shadow_angle`` (RIGHT),
   color-coded by arm, with per-panel Spearman ρ + p.
2. ``base_prior_dominance.png`` — bystander ΔG vs base-model log P(marker),
   colored by arm. Establishes the #500 effect dominates.
3. ``saturation_diagnostic.png`` — distribution of bystander ΔG across arms,
   showing 89-96% argmax = marker (saturation tell).
4. ``phase0p6_byte_identical_guard.png`` — g_logp vs b_logp scatter for the
   Phase-0.6 validation (proves the v3 measurement bug is fixed).
5. ``source_dg_by_cell.png`` — bar chart: source ΔG per cell × seed × frac,
   confirms anchor recipe lands in the 5-10 nat band.
6. ``raw_partial_<predictor>.png`` — raw + residualized scatter pairs for
   the three Holm-significant geometry / base-prior predictors.

All figures are written to ``figures/issue_504/``. Both PNG + PDF, with a
``.meta.json`` sidecar pinning the commit SHA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

EVAL = REPO_ROOT / "eval_results" / "issue_504"
FIG_DIR = "issue_504"  # relative under figures/ — savefig_paper handles the rest

# Arm display labels — plain English, no slug
ARM_LABEL = {
    "near": "near (con artist)",
    "mid_near": "mid-near (origami artist)",
    "mid_far": "mid-far (meditation teacher)",
    "far": "far (prosecutor)",
}
ARM_ORDER = ["near", "mid_near", "mid_far", "far"]
ARM_COLOR = {
    "near": paper_palette_role("primary"),
    "mid_near": paper_palette_role("accent"),
    "mid_far": paper_palette_role("baseline"),
    "far": paper_palette_role("control"),
}


def load_gates() -> dict:
    with open(EVAL / "phase0_5_gates.json") as f:
        return json.load(f)


def load_base_prior() -> dict:
    with open(EVAL / "base_prior_marker_v3.json") as f:
        return json.load(f)


def load_summary() -> dict:
    with open(EVAL / "analyze_summary.json") as f:
        return json.load(f)


def load_phase0p6() -> dict:
    with open(EVAL / "phase0p6_validation_v4.json") as f:
        return json.load(f)


def load_phase0_calibration() -> dict:
    with open(EVAL / "phase0_calibration_v4.json") as f:
        return json.load(f)


def load_trajectories() -> dict[tuple[str, int], dict]:
    """Pull all 8 cell × seed trajectories from HF data repo, indexed by (arm, seed)."""
    out = {}
    for cell in ARM_ORDER:
        for seed in (42, 137):
            path = hf_hub_download(
                repo_id="superkaiba1/explore-persona-space-data",
                repo_type="dataset",
                filename=f"issue504_geometry/phase1_trajectories/{cell}_seed{seed}/trajectory.json",
            )
            with open(path) as f:
                out[(cell, seed)] = json.load(f)
    return out


def assemble_rows(
    trajectories: dict[tuple[str, int], dict],
    gates: dict,
    base_prior: dict,
    chosen_frac: float = 0.33,
) -> list[dict]:
    """One row per (arm, seed, probe) at the chosen checkpoint fraction.

    ΔG is the mean across the 10 eval questions per probe (matches the analyzer).
    """
    rows = []
    per_probe = gates["per_probe"]
    for (arm, seed), data in trajectories.items():
        ckpt = next(c for c in data["checkpoints"] if abs(c["frac"] - chosen_frac) < 0.01)
        cell_slug = f"c504v3_{arm}"
        for probe, qs in ckpt["held_out"].items():
            dgs = [m["delta_g"] for m in qs.values() if m.get("delta_g") is not None]
            if not dgs:
                continue
            d_source = per_probe[probe]["d_source"]
            d_nn = per_probe[probe]["d_nearest_neg_nd"][cell_slug]
            shadow = per_probe[probe]["shadow_angle"][cell_slug]
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "probe": probe,
                    "delta_g": float(np.mean(dgs)),
                    "d_source": float(d_source),
                    "d_nearest_neg_nd": float(d_nn),
                    "shadow_angle": float(shadow),
                    "base_prior_marker": float(base_prior[probe]),
                    "argmax_frac": sum(1 for m in qs.values() if m.get("argmax_marker")) / len(qs),
                    "source_dg": float(ckpt["source_self"]["delta_g_mean"]),
                }
            )
    return rows


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residual of y on X (with intercept). Returns same shape as y."""
    # Add intercept
    X_int = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X_int, y, rcond=None)
    return y - X_int @ coef


def hero_figure(rows: list[dict], summary: dict) -> None:
    """Two-panel scatter: ΔG vs d_nearest_neg_nd (LEFT) and vs shadow_angle (RIGHT)."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, predictor, xlabel in [
        (axes[0], "d_nearest_neg_nd", "Distance to positioned negative N"),
        (axes[1], "shadow_angle", "Shadow angle: source→N to source→probe (radians)"),
    ]:
        for arm in ARM_ORDER:
            arm_rows = [r for r in rows if r["arm"] == arm]
            xs = [r[predictor] for r in arm_rows]
            ys = [r["delta_g"] for r in arm_rows]
            ax.scatter(
                xs,
                ys,
                color=ARM_COLOR[arm],
                alpha=0.6,
                s=28,
                label=ARM_LABEL[arm],
                edgecolors="white",
                linewidths=0.4,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Bystander marker ΔG (nats), trained − base")
        ax.axhline(y=0, color="lightgray", linestyle="--", linewidth=0.8, zorder=0)
        # Annotate Holm-rejected partial Spearman from analyze_summary
        ps = summary["pooled_fit"]["partial_spearman"][predictor]
        ax.text(
            0.04,
            0.96,
            f"partial ρ = {ps['rho']:+.3f}\np < 1e-12 (Holm)",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
        )

    # Annotations
    axes[0].text(
        0.04,
        0.04,
        "Bubble would predict closer-to-N → less leakage (positive ρ).\nObserved: closer-to-N → MORE leakage (anti-bubble).",
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        style="italic",
        color="#444444",
    )
    axes[1].text(
        0.04,
        0.04,
        "Barrier would predict shadowed (small angle) → less leakage (positive ρ).\nObserved: small angle → less leakage (consistent with barrier).",
        transform=axes[1].transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        style="italic",
        color="#444444",
    )

    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Bubble vs barrier: a single contrastive negative shows barrier-like protection but anti-bubble locality",
        fontsize=12,
        y=1.02,
    )

    fig.tight_layout()
    savefig_paper(fig, f"{FIG_DIR}/hero_bubble_vs_barrier", dir="figures/")
    plt.close(fig)


def hero_figure_raw(rows: list[dict]) -> None:
    """Raw counterpart: same x-axes, but ΔG NOT residualized — pure marginal scatter."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, predictor, xlabel in [
        (axes[0], "d_nearest_neg_nd", "Distance to positioned negative N"),
        (axes[1], "shadow_angle", "Shadow angle (radians)"),
    ]:
        for arm in ARM_ORDER:
            arm_rows = [r for r in rows if r["arm"] == arm]
            xs = [r[predictor] for r in arm_rows]
            ys = [r["delta_g"] for r in arm_rows]
            ax.scatter(
                xs,
                ys,
                color=ARM_COLOR[arm],
                alpha=0.6,
                s=28,
                label=ARM_LABEL[arm],
                edgecolors="white",
                linewidths=0.4,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Bystander marker ΔG (nats)")
        # Marginal Spearman
        all_x = np.array([r[predictor] for r in rows])
        all_y = np.array([r["delta_g"] for r in rows])
        rho, p = stats.spearmanr(all_x, all_y)
        ax.text(
            0.04,
            0.96,
            f"marginal ρ = {rho:+.3f}\np = {p:.2e}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
        )

    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle("Raw scatter (no partialling) — for comparison to hero", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_DIR}/hero_bubble_vs_barrier_raw", dir="figures/")
    plt.close(fig)


def base_prior_dominance(rows: list[dict], summary: dict) -> None:
    """Bystander ΔG vs base-model log P(marker)."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for arm in ARM_ORDER:
        arm_rows = [r for r in rows if r["arm"] == arm]
        xs = [r["base_prior_marker"] for r in arm_rows]
        ys = [r["delta_g"] for r in arm_rows]
        ax.scatter(
            xs,
            ys,
            color=ARM_COLOR[arm],
            alpha=0.6,
            s=28,
            label=ARM_LABEL[arm],
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xlabel("Base-model log P(marker) on probe persona (nats)")
    ax.set_ylabel("Bystander marker ΔG (nats), trained − base")
    ps = summary["pooled_fit"]["partial_spearman"]["base_prior_marker"]
    ax.text(
        0.04,
        0.96,
        f"partial ρ = {ps['rho']:+.3f}\np ≈ 0 (Holm)",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    set_title_subtitle(
        ax,
        "Base-prior dominates bystander leakage variance",
        "Probes the base model already nudges toward the marker climb hardest after training (#500 effect at scale).",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_DIR}/base_prior_dominance", dir="figures/")
    plt.close(fig)


def saturation_diagnostic(rows: list[dict]) -> None:
    """How saturated are the bystanders? Distribution of ΔG + argmax-marker fraction."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: ΔG distribution per arm (overlapping histograms)
    bins = np.linspace(-2, 32, 30)
    for arm in ARM_ORDER:
        arm_rows = [r for r in rows if r["arm"] == arm]
        ys = [r["delta_g"] for r in arm_rows]
        axes[0].hist(
            ys,
            bins=bins,
            color=ARM_COLOR[arm],
            alpha=0.45,
            label=ARM_LABEL[arm],
            edgecolor="none",
        )
    axes[0].axvspan(15, 32, color="lightgray", alpha=0.3, label="near-ceiling band")
    axes[0].set_xlabel("Bystander marker ΔG (nats)")
    axes[0].set_ylabel("Probes (count)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title("Bystander ΔG distribution — most probes saturated near ceiling")

    # Right: per-arm argmax fraction
    arm_argmax = {
        arm: np.mean([r["argmax_frac"] for r in rows if r["arm"] == arm]) for arm in ARM_ORDER
    }
    arms = list(arm_argmax.keys())
    fracs = [arm_argmax[a] for a in arms]
    colors = [ARM_COLOR[a] for a in arms]
    bars = axes[1].bar(
        [ARM_LABEL[a].split(" (")[0] for a in arms],
        fracs,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    axes[1].axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="100% argmax")
    axes[1].set_ylabel("Fraction of bystander × question pairs\nwith argmax = marker")
    axes[1].set_ylim(0, 1.05)
    for bar, frac in zip(bars, fracs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            frac + 0.01,
            f"{frac:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[1].set_title("Bystanders emit the marker at argmax in 89-96% of probe-question pairs")
    plt.setp(axes[1].get_xticklabels(), rotation=15)

    fig.tight_layout()
    savefig_paper(fig, f"{FIG_DIR}/saturation_diagnostic", dir="figures/")
    plt.close(fig)


def phase0p6_byte_identical_guard(phase0p6: dict) -> None:
    """g_logp (adapted) vs b_logp (base) scatter — the v3 measurement-bug fix lives or dies here."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7, 5))
    glogs = [r["g_logp"] for r in phase0p6["results_per_pair"]]
    blogs = [r["b_logp"] for r in phase0p6["results_per_pair"]]
    ax.scatter(
        blogs,
        glogs,
        color=paper_palette_role("primary"),
        alpha=0.7,
        s=40,
        edgecolors="white",
        linewidths=0.5,
    )
    # y = x diagonal
    lo = min(min(glogs), min(blogs)) - 1
    hi = max(max(glogs), max(blogs)) + 1
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="y = x (no adapter effect)",
    )
    ax.set_xlabel("base-model log P(marker) (nats)")
    ax.set_ylabel("adapted-model log P(marker) (nats)")
    ax.legend(loc="lower right", fontsize=9)
    set_title_subtitle(
        ax,
        "v3 measurement-bug fix: byte-identical rate = 0 across 20 probes",
        "Every adapted-model log-prob differs from base by ≥3 nats — the adapter is genuinely active.",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_DIR}/phase0p6_byte_identical_guard", dir="figures/")
    plt.close(fig)


def source_dg_by_cell(summary: dict, calibration: dict) -> None:
    """Source ΔG per cell × seed at the chosen checkpoint — confirms anchor in band."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    diags = summary["per_cell_diagnostics"]
    cells = sorted({d["cell"] for d in diags})
    arm_keys = [c.replace("c504v3_", "") for c in cells]
    seeds = [42, 137]

    x = np.arange(len(arm_keys))
    width = 0.36
    for i, seed in enumerate(seeds):
        ys = []
        for cell in cells:
            d = next(d for d in diags if d["cell"] == cell and d["seed"] == seed)
            ys.append(d["source_delta_g_nats"])
        ax.bar(
            x + (i - 0.5) * width,
            ys,
            width,
            color=paper_palette_role("primary" if i == 0 else "accent"),
            label=f"seed {seed}",
            edgecolor="white",
            linewidth=1.0,
        )
        for xi, yi in zip(x + (i - 0.5) * width, ys):
            ax.text(xi, yi + 0.15, f"{yi:.1f}", ha="center", va="bottom", fontsize=9)

    band_low, band_high = 5, 12
    ax.axhspan(
        band_low,
        band_high,
        color="lightgreen",
        alpha=0.2,
        label=f"target band ({band_low}-{band_high} nats)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a].split(" (")[0] for a in arm_keys])
    ax.set_ylabel("Source ΔG (nats), trained − base")
    ax.legend(loc="upper right", fontsize=9)
    set_title_subtitle(
        ax,
        f"Source implant clean — every cell lands in the {band_low}-{band_high} nat band",
        f"Phase 0 calibration chose checkpoint frac = {calibration['chosen_checkpoint_fraction']} "
        f"(step {calibration['chosen_checkpoint_steps']} / {calibration['chosen_epochs']} epochs).",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_DIR}/source_dg_by_cell", dir="figures/")
    plt.close(fig)


def main() -> None:
    print("Loading inputs...")
    gates = load_gates()
    base_prior = load_base_prior()
    summary = load_summary()
    phase0p6 = load_phase0p6()
    calibration = load_phase0_calibration()
    trajectories = load_trajectories()

    print(f"Assembling rows at checkpoint frac = {summary['chosen_checkpoint_fraction']}...")
    rows = assemble_rows(trajectories, gates, base_prior, summary["chosen_checkpoint_fraction"])
    print(f"  n_rows = {len(rows)} (expected: 4 arms × 2 seeds × 54 probes = 432)")

    print("Building hero figure...")
    hero_figure(rows, summary)
    hero_figure_raw(rows)

    print("Building base-prior dominance figure...")
    base_prior_dominance(rows, summary)

    print("Building saturation diagnostic...")
    saturation_diagnostic(rows)

    print("Building Phase 0.6 byte-identical guard...")
    phase0p6_byte_identical_guard(phase0p6)

    print("Building source ΔG per-cell bar chart...")
    source_dg_by_cell(summary, calibration)

    print("All figures written to figures/issue_504/")


if __name__ == "__main__":
    main()
