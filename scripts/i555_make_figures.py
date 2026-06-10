# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Greek ρ/ΔG + × + − intentional
#!/usr/bin/env python3
"""Task #555 — figures over eval_results/issue_555/ (plan §6 figure list).

HERO — replicate forest plot: per-replicate nearest-negative partial ρ with
bootstrap 95% CIs (5 rows), vertical zero line, the parent's seeds-42/137
step-5 reading overlaid as a reference marker, Holm-significant replicates as
filled dots; companion panel: shadow-angle (the specificity control), same
layout. → figures/issue_555/replicate_forest_nn_shadow.{png,pdf,meta.json}

Exploratory dump:
  * all-6-predictor per-replicate ρ dot table;
  * per-cell source ΔG strip plot (40 cells; the no-implant evidence);
  * raw pooled scatter g_logp vs d_nearest_neg_nd (raw-alongside-partialled
    discipline — expect an unstructured cloud);
  * pooled 2160-row fit vs per-replicate spread;
  * Δlog P vs Δz_marker agreement scatter at step 5;
  * bystander-gate panel (median g_logp + argmax share per replicate).

CPU-only; run on the VM after i555_replicate_analyze.py:
    uv run python scripts/i555_make_figures.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

log = logging.getLogger("i555.make_figures")

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_LABELS = {
    "d_source": "Distance to source",
    "d_nearest_neg_nd": "Distance to nearest negative",
    "shadow_angle": "Shadow angle",
    "base_prior_marker": "Base marker prior",
    "training_step": "Training step (zero-variance covariate)",
    "source_delta_g": "Source implant strength",
}


def _load_json(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"required input missing: {p}")
    return json.loads(p.read_text())


def _replicate_label(rep_key: str) -> str:
    # "R1_seeds7_11" -> "Replicate 1 (seeds 7, 11)"
    head, seeds = rep_key.split("_seeds", 1)
    return f"Replicate {head[1:]} (seeds {seeds.replace('_', ', ')})"


def fig_hero_forest(analysis: dict, out_dir: Path) -> None:
    """HERO: per-replicate forest plot, nearest-negative + shadow-angle panels."""
    reps = list(analysis["per_replicate"].keys())
    panels = [
        ("d_nearest_neg_nd", "Nearest-negative partial ρ at the no-implant step-5 snapshot"),
        ("shadow_angle", "Shadow-angle partial ρ (specificity control)"),
    ]
    parent = analysis.get("parent_reference", {})
    parent_ps = parent.get("partial_spearman", {}) if parent.get("available") else {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (pred, title) in zip(axes, panels, strict=True):
        y_positions = np.arange(len(reps))[::-1]
        for y, rep_key in zip(y_positions, reps, strict=True):
            rep = analysis["per_replicate"][rep_key]
            part = rep["pooled_fit"]["partial_spearman"].get(pred) or {}
            rho = part.get("rho")
            ci = rep["bootstrap_ci"].get(pred, {})
            holm_sig = bool(rep["family5_holm_primary"].get(pred, {}).get("reject_null", False))
            if rho is None:
                continue
            color = paper_palette_role("primary")
            if ci.get("lo") is not None:
                # Clamp half-widths at 0 (constant-bootstrap epsilon guard).
                lo_w = max(0.0, float(rho) - float(ci["lo"]))
                hi_w = max(0.0, float(ci["hi"]) - float(rho))
                ax.errorbar(
                    [rho],
                    [y],
                    xerr=[[lo_w], [hi_w]],
                    fmt="o",
                    color=color,
                    markerfacecolor=(color if holm_sig else "white"),
                    markeredgecolor=color,
                    capsize=3,
                )
            else:
                ax.plot([rho], [y], "o", color=color)
        ax.axvline(0.0, color="0.4", linewidth=0.8)
        ref = parent_ps.get(pred, {}).get("rho")
        if ref is not None:
            ax.axvline(
                float(ref),
                color=paper_palette_role("secondary"),
                linestyle="--",
                linewidth=1.2,
                label=f"Parent (seeds 42/137) step-5 reading: {float(ref):+.3f}",
            )
            ax.legend(loc="best", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Partial Spearman ρ (95% bootstrap CI)")
    axes[0].set_yticks(np.arange(len(reps))[::-1])
    axes[0].set_yticklabels([_replicate_label(r) for r in reps], fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "replicate_forest_nn_shadow", dir=out_dir)
    plt.close(fig)


def fig_predictor_dot_table(analysis: dict, out_dir: Path) -> None:
    """All-6-predictor per-replicate ρ dot table."""
    reps = list(analysis["per_replicate"].keys())
    preds = list(PREDICTOR_LABELS.keys())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = paper_palette(len(reps))
    for i, rep_key in enumerate(reps):
        rep = analysis["per_replicate"][rep_key]
        for j, pred in enumerate(preds):
            part = rep["pooled_fit"]["partial_spearman"].get(pred) or {}
            rho = part.get("rho")
            if rho is None:
                continue
            ax.plot([float(rho)], [j], "o", color=colors[i], alpha=0.85, markersize=6)
    ax.axvline(0.0, color="0.4", linewidth=0.8)
    ax.set_yticks(range(len(preds)))
    ax.set_yticklabels([PREDICTOR_LABELS[p] for p in preds], fontsize=9)
    ax.set_xlabel("Partial Spearman ρ per replicate")
    ax.set_title("All six predictors, five no-implant replicates", fontsize=10)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[i], label=_replicate_label(r))
        for i, r in enumerate(reps)
    ]
    ax.legend(handles=handles, fontsize=7, loc="best")
    fig.tight_layout()
    savefig_paper(fig, "predictor_dot_table", dir=out_dir)
    plt.close(fig)


def fig_source_dg_strip(analysis: dict, out_dir: Path) -> None:
    """Per-cell source ΔG strip (the no-implant evidence; expect ≈0.03 nats)."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs, ys, labels = [], [], []
    for i, (rep_key, rep) in enumerate(analysis["per_replicate"].items()):
        dgs = rep["usability_descriptive"]["per_cell_source_delta_g"]
        for v in dgs.values():
            if v is None:
                continue
            xs.append(i + (np.random.default_rng(0).uniform(-0.15, 0.15)))
            ys.append(float(v))
        labels.append(_replicate_label(rep_key))
    ax.scatter(xs, ys, s=18, color=paper_palette_role("primary"), alpha=0.7)
    ax.axhline(1.0, color=paper_palette_role("secondary"), linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([la.replace("Replicate ", "R") for la in labels], fontsize=8)
    ax.set_ylabel("Source ΔG at step 5 (nats)")
    ax.set_title("Per-cell source implant strength — all cells far below the 1-nat floor")
    fig.tight_layout()
    savefig_paper(fig, "source_dg_strip", dir=out_dir)
    plt.close(fig)


def fig_raw_scatter(slab: Path, analysis: dict, gates_path: Path, out_dir: Path) -> None:
    """Raw pooled scatter: per-row ΔG vs nearest-negative distance (no partialling)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        POSITIONED_ARM_SLUGS_V3,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
        build_rows,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase05 import (
        load_phase05,
    )

    gates = load_phase05(gates_path)
    seeds = [int(s) for pair in analysis["replicates"] for s in pair.split(":")]
    pooled = build_rows(
        slab_root=slab,
        chosen_frac=1.0,
        per_probe=gates["per_probe"],
        arm_to_positioned_n=gates["arm_to_positioned_n"],
        seeds=seeds,
        base_prior_by_probe=None,
        positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
        dg_band=None,
    )
    rows = pooled["rows"]
    x = [r["d_nearest_neg_nd"] for r in rows]
    y = [r["delta_g"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(x, y, s=8, alpha=0.35, color=paper_palette_role("primary"))
    ax.set_xlabel("Distance to nearest negative (raw)")
    ax.set_ylabel("Per-probe ΔG at step 5 (nats, raw)")
    ax.set_title(f"Raw pooled scatter, all replicates (n={len(rows)})", fontsize=10)
    fig.tight_layout()
    savefig_paper(fig, "raw_scatter_dg_vs_dnn", dir=out_dir)
    plt.close(fig)


def fig_pooled_vs_replicates(analysis: dict, out_dir: Path) -> None:
    """Pooled 2160-row descriptive fit vs the per-replicate spread."""
    preds = ["d_nearest_neg_nd", "shadow_angle", "d_source"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    pooled_ps = analysis["pooled_descriptive_fit"]["fit"].get("partial_spearman", {})
    for j, pred in enumerate(preds):
        rhos = analysis["cross_replicate"][pred]["rhos"]
        ax.plot(rhos, [j] * len(rhos), "o", color=paper_palette_role("primary"), alpha=0.6)
        pooled_rho = (pooled_ps.get(pred) or {}).get("rho")
        if pooled_rho is not None:
            ax.plot(
                [float(pooled_rho)],
                [j],
                "D",
                color=paper_palette_role("secondary"),
                markersize=8,
                alpha=0.9,
            )
    ax.axvline(0.0, color="0.4", linewidth=0.8)
    ax.set_yticks(range(len(preds)))
    ax.set_yticklabels([PREDICTOR_LABELS[p] for p in preds], fontsize=9)
    ax.set_xlabel("Partial Spearman ρ — circles: per-replicate; diamond: pooled (descriptive)")
    fig.tight_layout()
    savefig_paper(fig, "pooled_vs_replicates", dir=out_dir)
    plt.close(fig)


def fig_z_agreement(analysis: dict, out_dir: Path) -> None:
    """Δlog P vs Δz_marker per-replicate agreement summary at step 5."""
    reps = list(analysis["per_replicate"].keys())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    xs, ys = [], []
    for rep_key in reps:
        z = analysis["per_replicate"][rep_key]["z_agreement"]
        if not z.get("available"):
            continue
        xs.append(z["mean_delta_z_marker"])
        ys.append(z["mean_delta_logp"])
    if xs:
        ax.scatter(xs, ys, s=40, color=paper_palette_role("primary"))
        lim = max(abs(v) for v in xs + ys) * 1.2 + 1e-3
        ax.plot([-lim, lim], [-lim, lim], "--", color="0.6", linewidth=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_xlabel("Mean Δz_marker (logit space)")
    ax.set_ylabel("Mean Δlog P (log-prob space)")
    ax.set_title("Space agreement at step 5 (off saturation the two should match)", fontsize=10)
    fig.tight_layout()
    savefig_paper(fig, "z_agreement", dir=out_dir)
    plt.close(fig)


def fig_bystander_gate(analysis: dict, out_dir: Path) -> None:
    """Bystander-gate panel: pooled median g_logp + argmax share per replicate."""
    reps = list(analysis["per_replicate"].keys())
    med = []
    share = []
    for rep_key in reps:
        gate = analysis["per_replicate"][rep_key]["usability_descriptive"]["bystander_gate"]
        med.append(gate["pooled_median_g_logp"])
        share.append(gate["pooled_argmax_share"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(reps))
    axes[0].bar(
        x, [m if m is not None else np.nan for m in med], color=paper_palette_role("primary")
    )
    axes[0].axhline(-2.0, color=paper_palette_role("secondary"), linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Median bystander ΔG (nats)")
    axes[1].bar(
        x, [s if s is not None else np.nan for s in share], color=paper_palette_role("primary")
    )
    axes[1].axhline(0.60, color=paper_palette_role("secondary"), linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Bystander argmax-marker share")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{i + 1}" for i in range(len(reps))], fontsize=8)
    fig.suptitle("Bystander gate values per replicate (descriptive only)", fontsize=10)
    fig.tight_layout()
    savefig_paper(fig, "bystander_gate_panel", dir=out_dir)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_555"))
    ap.add_argument(
        "--analysis",
        type=Path,
        default=None,
        help="analysis_replicates.json (default <slab-root>/analysis_replicates.json).",
    )
    ap.add_argument(
        "--phase05-path",
        type=Path,
        default=Path("eval_results/issue_530/phase0_5_gates.json"),
    )
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures" / "issue_555")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=figures_555] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    set_paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = (
        args.analysis if args.analysis is not None else args.slab_root / "analysis_replicates.json"
    )
    analysis = _load_json(analysis_path)

    fig_hero_forest(analysis, args.out_dir)
    fig_predictor_dot_table(analysis, args.out_dir)
    fig_source_dg_strip(analysis, args.out_dir)
    fig_raw_scatter(args.slab_root, analysis, args.phase05_path, args.out_dir)
    fig_pooled_vs_replicates(analysis, args.out_dir)
    fig_z_agreement(analysis, args.out_dir)
    fig_bystander_gate(analysis, args.out_dir)
    log.info("[phase=done] figures written under %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
