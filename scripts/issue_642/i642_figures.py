#!/usr/bin/env python3
# Research notation (−, Δ, ρ) is intentional in labels + prose.
# ruff: noqa: RUF001, RUF002, RUF003
"""Task #642 — figures (3-arm decomposition heroes + exploratory dump, plan §6).

Reads ``analysis.json`` + the cmft stage-A trajectory per behavior and writes to
``figures/issue_642/`` via the paper-plots conventions.

Hero candidates (plan §6): (1) the DECOMPOSITION BAR — #606's measured gap
(+0.098) split into the measured Δ_rank (adapter-vs-dense bundle) + Δ_coverage
stack with CIs (the headline figure); (2) leakage-vs-strength overlay (x = s, y =
38-bystander-mean delta; LoRA orange, cmft green, FT blue; matched band shaded);
(3) per-persona profile scatters at s* (cmft-vs-LoRA and ft-vs-cmft, identity
line, twins/roster/negative-members marked, ρ annotated). Exploratory: stage-A
cmft trajectory; per-persona heatmaps (clean AND raw); Δ_rank/Δ_coverage-vs-s*
sweep curves.

Usage::

    uv run python scripts/issue_642/i642_figures.py --behavior sycophancy \
        --eval-root eval_results/issue_642 [--out-dir figures/issue_642]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "issue_642"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from i642_common import (  # noqa: E402
    ISSUE606_GAP,
    S_BAND,
    S_TARGET,
    SOURCE_PERSONA,
    SYCO_EXPECTED_NEGATIVES,
    TWIN_PROMPTS,
)

log = logging.getLogger("issue_642.figures")

ARM_COLORS = {"lora": "tab:orange", "cmft": "tab:green", "ft": "tab:blue"}
ARM_LABELS = {"lora": "LoRA", "cmft": "coverage-matched FT", "ft": "full FT"}
NEG_MEMBERS = {"sycophancy": set(SYCO_EXPECTED_NEGATIVES)}


def _save(fig, name: str, out_dir: Path) -> None:
    from explore_persona_space.analysis.paper_plots import savefig_paper

    savefig_paper(fig, name, dir=out_dir)
    plt.close(fig)
    log.info("figure -> %s/%s", out_dir, name)


def _arm_points(analysis: dict) -> dict[str, list[tuple[str, float, float]]]:
    """Per arm: (cell, s_stage_b, 38-bystander-mean clean delta) sorted by s."""
    s = analysis["s_stage_b"]
    tables = analysis["per_cell_tables"]
    bystanders = list(analysis["per_persona_at_target"]["lora"])
    out: dict[str, list[tuple[str, float, float]]] = {"lora": [], "cmft": [], "ft": []}
    for cell, s_val in s.items():
        arm = cell.split("_step")[0]
        if arm not in out:
            continue
        deltas = [
            tables[cell][p]["delta_clean"]
            for p in bystanders
            if tables[cell][p]["delta_clean"] is not None
        ]
        out[arm].append((cell, s_val, float(np.nanmean(deltas))))
    for arm in out:
        out[arm].sort(key=lambda t: t[1])
    return out


def fig_decomposition_bar(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Hero candidate 1: the #606 gap split into Δ_rank + Δ_coverage (stacked,
    with CIs) — the headline figure."""
    h = analysis["headline"]
    dr = h["delta_rank"]["gap_plugin"]
    dc = h["delta_coverage"]["gap_plugin"]
    dr_ci = h["delta_rank"]["gap_ci95"]
    dc_ci = h["delta_coverage"]["gap_ci95"]
    recon = h["additive_identity"]["reconstructed_gap_plugin"]
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    # Stacked bar for the measured decomposition.
    ax.bar(
        ["measured\ndecomposition"], [dr], color=ARM_COLORS["cmft"], label="Δ_rank (cmft − LoRA)"
    )
    ax.bar(
        ["measured\ndecomposition"],
        [dc],
        bottom=[dr],
        color=ARM_COLORS["ft"],
        label="Δ_coverage (FT − cmft)",
    )
    # Reference bar for #606's measured total gap.
    ax.bar(["#606 gap\n(FT − LoRA)"], [ISSUE606_GAP], color="tab:grey", alpha=0.6)
    # CIs on the two components (drawn at their stacked midpoints).
    ax.errorbar(
        0,
        dr / 2,
        yerr=[[dr / 2 - dr_ci[0]], [dr_ci[1] - dr / 2]],
        fmt="none",
        ecolor="black",
        capsize=3,
        alpha=0.7,
    )
    ax.errorbar(
        0,
        dr + dc / 2,
        yerr=[[max(0.0, dc / 2 - (dc_ci[0] - 0))], [max(0.0, dc_ci[1] - dc / 2)]],
        fmt="none",
        ecolor="black",
        capsize=3,
        alpha=0.7,
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("bystander-mean leakage gap at s*=0.50")
    ax.set_title(
        f"{behavior}: #606 gap decomposition (verdict {h['verdict']})\n"
        f"Δ_rank={dr:+.3f} + Δ_coverage={dc:+.3f} = {recon:+.3f} (target +{ISSUE606_GAP})"
    )
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_decomposition_bar_hero", out_dir)


def fig_leakage_vs_strength(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Hero candidate 2: 38-bystander-mean leakage vs implant strength, 3 arms."""
    pts = _arm_points(analysis)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for arm in ("lora", "cmft", "ft"):
        rows = pts[arm]
        if not rows:
            continue
        xs = [0.0, *(r[1] for r in rows)]
        ys = [0.0, *(r[2] for r in rows)]
        ax.plot(xs, ys, "o-", color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    ax.axvspan(S_BAND[0], S_BAND[1], alpha=0.10, color="grey")
    ax.axvline(S_TARGET, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("source-implant strength s (source-self rate delta)")
    ax.set_ylabel("bystander-mean leakage delta")
    ax.set_title(f"{behavior}: leakage vs implant strength (3 arms)")
    ax.legend()
    _save(fig, f"{behavior}_leakage_vs_strength_hero", out_dir)


def _profile_scatter(
    analysis: dict, behavior: str, out_dir: Path, arm_hi: str, arm_lo: str
) -> None:
    """Hero candidate 3: per-persona arm_hi vs arm_lo at s* (identity line)."""
    per = analysis["per_persona_at_target"]
    contrast_key = "delta_rank" if (arm_hi, arm_lo) == ("cmft", "lora") else "delta_coverage"
    rho = analysis["headline"][contrast_key]["profile_spearman_rho"]
    negs = NEG_MEMBERS.get(behavior, set())
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    groups = {"twin": [], "roster": [], "negative-member": []}
    for p in per[arm_lo]:
        kind = "negative-member" if p in negs else ("twin" if p in TWIN_PROMPTS else "roster")
        groups[kind].append(p)
    styles = {
        "roster": {"marker": "o", "color": "tab:grey"},
        "twin": {"marker": "^", "color": "tab:green"},
        "negative-member": {"marker": "s", "color": "tab:red"},
    }
    for kind, members in groups.items():
        if not members:
            continue
        ax.scatter(
            [per[arm_lo][p] for p in members],
            [per[arm_hi][p] for p in members],
            s=28,
            label=kind,
            **styles[kind],
        )
    all_vals = list(per[arm_lo].values()) + list(per[arm_hi].values())
    lim = [min(all_vals) - 0.05, max(all_vals) + 0.05]
    ax.plot(lim, lim, color="black", lw=0.8, ls=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"{ARM_LABELS[arm_lo]} per-persona leakage delta at s*")
    ax.set_ylabel(f"{ARM_LABELS[arm_hi]} per-persona leakage delta at s*")
    ax.set_title(
        f"{behavior}: {ARM_LABELS[arm_hi]} vs {ARM_LABELS[arm_lo]} (Spearman ρ = {rho:.2f})"
    )
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_profile_{arm_hi}_vs_{arm_lo}_hero", out_dir)


def fig_trajectories(trajectory: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: stage-A cmft s(step) (this run's only trained trajectory)."""
    cells = trajectory["cells"]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    rows = sorted(
        ((rec["step"], rec.get("s")) for c, rec in cells.items() if rec.get("arm") == "cmft"),
        key=lambda t: t[0],
    )
    rows = [(st, s) for st, s in rows if s is not None]
    if rows:
        ax.plot(
            [r[0] for r in rows],
            [r[1] for r in rows],
            "o-",
            color=ARM_COLORS["cmft"],
            label="coverage-matched FT (stage-A)",
        )
    ax.axhspan(S_BAND[0], S_BAND[1], alpha=0.10, color="grey")
    ax.axhline(S_TARGET, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("source-implant strength s")
    ax.set_title(f"{behavior}: cmft stage-A install trajectory")
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_cmft_stage_a_trajectory", out_dir)


def fig_heatmaps(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: per-persona delta heatmaps, CLEAN and RAW side by side."""
    tables = analysis["per_cell_tables"]
    s = analysis["s_stage_b"]
    cells = sorted(s, key=lambda c: s[c])
    personas = sorted(p for p in next(iter(tables.values())) if p != SOURCE_PERSONA)
    negs = NEG_MEMBERS.get(behavior, set())
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.32 * len(personas) + 2), sharey=True)
    for ax, key, title in zip(axes, ("delta_clean", "delta_raw"), ("clean", "raw"), strict=True):
        mat = np.array([[tables[c][p][key] for c in cells] for p in personas], dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{c}\n(s={s[c]:.2f})" for c in cells], fontsize=6, rotation=45)
        ax.set_title(f"{behavior}: per-persona delta ({title})")
        fig.colorbar(im, ax=ax, shrink=0.6)
    labels = [
        f"{p} (neg)" if p in negs else (f"{p} (twin)" if p in TWIN_PROMPTS else p) for p in personas
    ]
    axes[0].set_yticks(range(len(personas)))
    axes[0].set_yticklabels(labels, fontsize=6)
    _save(fig, f"{behavior}_per_persona_heatmap", out_dir)


def fig_sweep(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: Δ_rank + Δ_coverage vs matched-strength target."""
    sweep = analysis["sweep"]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for key, color, label in (
        ("delta_rank", ARM_COLORS["cmft"], "Δ_rank (cmft − LoRA)"),
        ("delta_coverage", ARM_COLORS["ft"], "Δ_coverage (FT − cmft)"),
    ):
        rows = sweep[key]
        xs = [r["target"] for r in rows]
        ys = [r["gap_plugin"] for r in rows]
        lo = [max(0.0, r["gap_plugin"] - r["gap_ci"][0]) for r in rows]
        hi = [max(0.0, r["gap_ci"][1] - r["gap_plugin"]) for r in rows]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=color, capsize=3, label=label)
        for r in rows:
            if not r["in_range_both_arms"]:
                ax.plot(r["target"], r["gap_plugin"], "x", color="red", ms=9)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhspan(-0.04, 0.04, alpha=0.10, color="green")
    ax.axvline(S_TARGET, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("matched-strength target s*")
    ax.set_ylabel("contrast (bystander mean)")
    ax.set_title(f"{behavior}: Δ_rank / Δ_coverage vs s* (x = out of range)")
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_decomposition_vs_target_sweep", out_dir)


def _pretty_persona(p: str) -> str:
    label = p.replace("_", " ").capitalize()
    return (
        label.replace("Ai ", "AI ")
        if label.startswith("Ai ")
        else ("AI" if label == "Ai" else label)
    )


def fig_endpoint_comparison(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: endpoint (near-ceiling) descriptive 3-arm per-persona bars."""
    tables = analysis["per_cell_tables"]
    s = analysis["s_stage_b"]
    endpoints = {}
    for arm in ("lora", "cmft", "ft"):
        arm_cells = [c for c in s if c.startswith(f"{arm}_")]
        if arm_cells:
            endpoints[arm] = max(arm_cells, key=lambda c: int(c.split("step")[-1]))
    if len(endpoints) < 2:
        return
    personas = sorted(p for p in tables[next(iter(endpoints.values()))] if p != SOURCE_PERSONA)
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(personas)), 4.2))
    x = np.arange(len(personas))
    n = len(endpoints)
    for i, (arm, cell) in enumerate(endpoints.items()):
        ys = [tables[cell][p]["delta_clean"] for p in personas]
        ax.bar(
            x + (i - (n - 1) / 2) * 0.27,
            ys,
            width=0.27,
            color=ARM_COLORS[arm],
            label=f"{ARM_LABELS[arm]} (endpoint, step {cell.split('step')[-1]})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_persona(p) for p in personas], fontsize=5, rotation=90)
    ax.set_ylabel("leakage delta (clean)")
    ax.set_title(f"{behavior}: endpoint cells, descriptive (near ceiling — not the headline)")
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_endpoint_descriptive", out_dir)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=figures] %(message)s")
    p = argparse.ArgumentParser(description="#642 figures (decomposition heroes + exploratory).")
    p.add_argument("--behavior", required=True, choices=["sycophancy", "refusal"])
    p.add_argument("--eval-root", type=Path, default=REPO / "eval_results" / "issue_642")
    p.add_argument("--out-dir", type=Path, default=REPO / "figures" / "issue_642")
    args = p.parse_args(argv)

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    broot = args.eval_root / args.behavior
    analysis = json.loads((broot / "analysis.json").read_text())
    trajectory = json.loads((broot / "stage_a" / f"trajectory_{args.behavior}.json").read_text())

    fig_decomposition_bar(analysis, args.behavior, args.out_dir)
    fig_leakage_vs_strength(analysis, args.behavior, args.out_dir)
    _profile_scatter(analysis, args.behavior, args.out_dir, "cmft", "lora")
    _profile_scatter(analysis, args.behavior, args.out_dir, "ft", "cmft")
    fig_trajectories(trajectory, args.behavior, args.out_dir)
    fig_heatmaps(analysis, args.behavior, args.out_dir)
    fig_sweep(analysis, args.behavior, args.out_dir)
    fig_endpoint_comparison(analysis, args.behavior, args.out_dir)
    log.info("all figures written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
