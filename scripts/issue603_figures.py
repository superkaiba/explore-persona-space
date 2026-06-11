#!/usr/bin/env python3
"""#603 figures (CPU, VM) — paper_plots conventions, plain-English labels.

Hero: 3 (family) x 2 (common-mode fraction on prior | write norm on
prior) scatter grid — the deliverable contrast is visible as "left
column slopes, right column doesn't" (or the reverse). Fact panels:
teacher color + seed marker + per-seed connecting lines.

Exploratory over-produce (plan #603 §6): per-adapter stacked
shared-vs-residual norm bars; disattenuated-CMF hero variant;
layer/position sensitivity grid; SVD / unit-norm-SVD estimator variants;
leave-one-bystander-out jackknife whiskers; reliability-vs-prior
scatter; behavioral-linkage panel; #551 marker-calibration bar;
expression-stratified panel (when guard B has run).

Run (VM, after issue603_decompose.py)::

    uv run python scripts/issue603_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i603_figures")

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_603"
FIG_DIR = "figures/issue_603"  # default; overridable via --fig-dir (smoke isolation)
PRIMARY_READ = "primary_l14_mean_resp"

FAMILY_LABEL = {"fact": "Fact teachers", "refusal": "Refusal sources", "em": "EM sources"}
TEACHER_LABEL = {
    "marine_biologist": "Marine biologist (low prior)",
    "courthouse_architecture_historian": "Architecture historian (mid prior)",
    "wooden_furniture_carpenter": "Furniture carpenter (high prior)",
}
TEACHER_COLOR_ROLE = {
    "marine_biologist": "primary",
    "courthouse_architecture_historian": "accent",
    "wooden_furniture_carpenter": "baseline",
}
SEED_MARKER = {42: "o", 137: "s", 256: "^"}


def _cells_of(results: dict, family: str) -> list[dict]:
    return sorted(
        (d for d in results["per_cell"].values() if d["family"] == family),
        key=lambda d: (d["source"], d["seed"]),
    )


def _scatter_family(ax, cells: list[dict], dv_key: str, *, dv_from_reads: bool = True) -> None:
    fam = cells[0]["family"]
    if fam == "fact":
        for teacher, role in TEACHER_COLOR_ROLE.items():
            color = paper_palette_role(role)
            pts = [d for d in cells if d["source"] == teacher]
            for d in pts:
                y = d["reads"][PRIMARY_READ][dv_key] if dv_from_reads else d[dv_key]
                ax.scatter(
                    d["prior"], y, color=color, marker=SEED_MARKER[d["seed"]], s=42, zorder=3
                )
        # Per-seed connecting lines across the prior axis.
        for seed, _mk in SEED_MARKER.items():
            pts = sorted((d for d in cells if d["seed"] == seed), key=lambda d: d["prior"])
            xs = [d["prior"] for d in pts]
            ys = [d["reads"][PRIMARY_READ][dv_key] if dv_from_reads else d[dv_key] for d in pts]
            ax.plot(xs, ys, color="0.6", lw=0.9, alpha=0.8, zorder=2)
    else:
        color = paper_palette_role("primary")
        for d in cells:
            y = d["reads"][PRIMARY_READ][dv_key] if dv_from_reads else d[dv_key]
            ax.scatter(d["prior"], y, color=color, s=42, zorder=3)
            ax.annotate(
                d["source"].replace("_", " "),
                (d["prior"], y),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )


def fig_hero(results: dict, *, dis: bool = False) -> None:
    set_paper_style()
    families = [f for f in ("fact", "refusal", "em") if _cells_of(results, f)]
    fig, axes = plt.subplots(len(families), 2, figsize=(8.8, 2.7 * len(families)), squeeze=False)
    for i, family in enumerate(families):
        cells = _cells_of(results, family)
        if dis:
            cells = [d for d in cells if d.get("cmf_disattenuated") is not None]
            if not cells:
                continue
        ax_c, ax_n = axes[i]
        if dis:
            _scatter_family(ax_c, cells, "cmf_disattenuated", dv_from_reads=False)
        else:
            _scatter_family(ax_c, cells, "cmf")
        _scatter_family(ax_n, cells, "norm")
        label = "noise-corrected\ncommon-mode fraction" if dis else "common-mode\nfraction"
        ax_c.set_ylabel(f"{FAMILY_LABEL[family]}\n{label}")
        ax_n.set_ylabel("Write norm")
        for ax in (ax_c, ax_n):
            ax.set_xlabel("Source prior (log P per token, base model)")
    axes[0][0].set_title("Direction mix vs prior")
    axes[0][1].set_title("Write norm vs prior")
    fig.tight_layout()
    stem = "hero_cmf_vs_norm_disattenuated" if dis else "hero_cmf_vs_norm"
    savefig_paper(fig, stem, dir=FIG_DIR)
    plt.close(fig)


def fig_shared_residual_bars(results: dict) -> None:
    set_paper_style()
    cells = sorted(
        results["per_cell"].items(),
        key=lambda kv: (kv[1]["family"], kv[1]["prior"], kv[1]["seed"]),
    )
    labels = [f"{d['source'].replace('_', ' ')} s{d['seed']} ({d['family']})" for _, d in cells]
    shared = [abs(d["reads"][PRIMARY_READ]["shared_norm"]) for _, d in cells]
    resid = [d["reads"][PRIMARY_READ]["residual_norm"] for _, d in cells]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    x = range(len(cells))
    ax.bar(
        x,
        shared,
        label="Shared component (|projection on mean-bystander dir|)",
        color=paper_palette_role("primary"),
    )
    ax.bar(
        x,
        resid,
        bottom=shared,
        label="Source-specific residual",
        color=paper_palette_role("neutral"),
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("Norm (residual-stream units)")
    ax.legend(fontsize=7)
    ax.set_title("Source write split into shared vs source-specific components")
    fig.tight_layout()
    savefig_paper(fig, "shared_vs_residual_bars", dir=FIG_DIR)
    plt.close(fig)


def fig_estimator_variants(results: dict) -> None:
    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
    for ax, (key, title) in zip(
        axes,
        [
            ("cmf", "Mean-bystander direction (primary)"),
            ("cmf_svd", "SVD top direction"),
            ("cmf_svd_unitnorm", "Unit-norm SVD top direction"),
        ],
        strict=True,
    ):
        for family, role in (("fact", "primary"), ("refusal", "accent"), ("em", "baseline")):
            cells = _cells_of(results, family)
            ax.scatter(
                [d["prior"] for d in cells],
                [d["reads"][PRIMARY_READ][key] for d in cells],
                color=paper_palette_role(role),
                s=30,
                label=FAMILY_LABEL[family],
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Source prior (log P per token)")
    axes[0].set_ylabel("Common-mode fraction")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "estimator_variants", dir=FIG_DIR)
    plt.close(fig)


def fig_jackknife(results: dict) -> None:
    set_paper_style()
    cells = sorted(
        results["per_cell"].items(),
        key=lambda kv: (kv[1]["family"], kv[1]["prior"], kv[1]["seed"]),
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    for i, (_cid, d) in enumerate(cells):
        jk = d["reads"][PRIMARY_READ]["cmf_jackknife"]
        lo, hi = min(jk), max(jk)
        center = d["reads"][PRIMARY_READ]["cmf"]
        # Whiskers clamped non-negative (constant-jackknife float noise).
        ax.errorbar(
            i,
            center,
            yerr=[[max(0.0, center - lo)], [max(0.0, hi - center)]],
            fmt="o",
            ms=4,
            color=paper_palette_role("primary"),
            ecolor="0.5",
        )
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(
        [f"{d['source'].replace('_', ' ')} s{d['seed']}" for _, d in cells],
        rotation=75,
        ha="right",
        fontsize=6,
    )
    ax.set_ylabel("Common-mode fraction")
    ax.set_title("Leave-one-bystander-out jackknife range per cell")
    fig.tight_layout()
    savefig_paper(fig, "jackknife_whiskers", dir=FIG_DIR)
    plt.close(fig)


def fig_reliability(results: dict) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for family, role in (("fact", "primary"), ("refusal", "accent"), ("em", "baseline")):
        cells = _cells_of(results, family)
        ax.scatter(
            [d["prior"] for d in cells],
            [d["reliability"]["r_a_source_dir"]["r_random_mean"] for d in cells],
            color=paper_palette_role(role),
            s=30,
            label=FAMILY_LABEL[family],
        )
    ax.axhline(0.3, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("Source prior (log P per token)")
    ax.set_ylabel("Split-half reliability of source write direction")
    ax.set_title("Reliability vs prior (noise-attenuation guard)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "reliability_vs_prior", dir=FIG_DIR)
    plt.close(fig)


def fig_sensitivity(results: dict) -> None:
    set_paper_style()
    reads = [
        ("primary_l14_mean_resp", "L14 mean-resp (primary)"),
        ("l14_end_slot", "L14 end-slot"),
        ("l7_mean_resp", "L7 mean-resp"),
        ("l21_mean_resp", "L21 mean-resp"),
        ("l7_end_slot", "L7 end-slot"),
        ("l21_end_slot", "L21 end-slot"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6), sharex=True)
    for ax, (rk, title) in zip(axes.flat, reads, strict=True):
        for family, role in (("fact", "primary"), ("refusal", "accent"), ("em", "baseline")):
            cells = [d for d in _cells_of(results, family) if rk in d["reads"]]
            ax.scatter(
                [d["prior"] for d in cells],
                [d["reads"][rk]["cmf"] for d in cells],
                color=paper_palette_role(role),
                s=24,
                label=FAMILY_LABEL[family],
            )
        ax.set_title(title, fontsize=8)
    axes[0][0].legend(fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("Source prior")
    for ax in axes[:, 0]:
        ax.set_ylabel("Common-mode fraction")
    fig.tight_layout()
    savefig_paper(fig, "sensitivity_layers_positions", dir=FIG_DIR)
    plt.close(fig)


def fig_linkage(results: dict) -> None:
    set_paper_style()
    rows = [
        (cid, d)
        for cid, d in results["per_cell"].items()
        if d.get("behavioral_linkage") is not None
    ]
    if not rows:
        logger.info("no behavioral-linkage panels to plot")
        return
    rows.sort(key=lambda kv: (kv[1]["family"], kv[1]["prior"], kv[1]["seed"]))
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    xs = range(len(rows))
    ax.bar(
        xs,
        [d["behavioral_linkage"]["spearman_rho_proj_vs_leak"] for _, d in rows],
        color=paper_palette_role("primary"),
    )
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [f"{d['source'].replace('_', ' ')} s{d['seed']} ({d['family']})" for _, d in rows],
        rotation=75,
        ha="right",
        fontsize=6,
    )
    ax.set_ylabel("Spearman rho (bystander projection vs measured leak)")
    ax.set_title("Validity panel: do the shift projections track measured leakage?")
    fig.tight_layout()
    savefig_paper(fig, "behavioral_linkage", dir=FIG_DIR)
    plt.close(fig)


def fig_calibration(v1_gate_path: Path) -> None:
    if not v1_gate_path.exists():
        return
    gate = json.loads(v1_gate_path.read_text())
    calib = gate.get("calibration_medical_doctor_cmf", {})
    if not calib:
        return
    set_paper_style()
    cells = sorted(calib)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(
        range(len(cells)),
        [calib[c]["cmf_mean_resp"] for c in cells],
        color=paper_palette_role("neutral"),
    )
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(
        [c.replace("same_", "").replace("_", " ") for c in cells],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Common-mode fraction (medical doctor source)")
    ax.set_title("Calibration: #551 marker/EM cells through the #603 decomposition")
    fig.tight_layout()
    savefig_paper(fig, "marker_calibration", dir=FIG_DIR)
    plt.close(fig)


def fig_expression_strata(strata_path: Path) -> None:
    if not strata_path.exists():
        logger.info("expression_strata.json absent — guard-B panel skipped")
        return
    strata = json.loads(strata_path.read_text())
    rows = sorted(strata["per_cell"].items(), key=lambda kv: (kv[1]["family"], kv[1]["seed"]))
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    width = 0.38
    for i, (_cid, d) in enumerate(rows):
        if d["cmf_expressed"] is not None:
            ax.bar(i - width / 2, d["cmf_expressed"], width, color=paper_palette_role("primary"))
        if d["cmf_not_expressed"] is not None:
            ax.bar(i + width / 2, d["cmf_not_expressed"], width, color=paper_palette_role("accent"))
    ax.bar(0, 0, 0, color=paper_palette_role("primary"), label="Behavior-expressing questions")
    ax.bar(0, 0, 0, color=paper_palette_role("accent"), label="Non-expressing questions")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(
        [f"{d['source'].replace('_', ' ')} s{d['seed']} ({d['family']})" for _, d in rows],
        rotation=75,
        ha="right",
        fontsize=6,
    )
    ax.set_ylabel("Common-mode fraction")
    ax.set_title("Guard B: within-cell expression-stratified common-mode fraction")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "expression_strata", dir=FIG_DIR)
    plt.close(fig)


def main() -> int:
    """Render all #603 figures from decomposition_results.json."""
    global FIG_DIR
    ap = argparse.ArgumentParser(description="#603 figures")
    ap.add_argument("--results", default=str(EVAL_DIR / "decomposition_results.json"))
    ap.add_argument("--fig-dir", default=FIG_DIR)
    args = ap.parse_args()
    FIG_DIR = args.fig_dir

    results = json.loads(Path(args.results).read_text())
    fig_hero(results)
    fig_hero(results, dis=True)
    fig_shared_residual_bars(results)
    fig_estimator_variants(results)
    fig_jackknife(results)
    fig_reliability(results)
    fig_sensitivity(results)
    fig_linkage(results)
    fig_calibration(EVAL_DIR / "v1_gate.json")
    fig_expression_strata(EVAL_DIR / "expression_strata.json")
    logger.info("[done] figures under %s", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
