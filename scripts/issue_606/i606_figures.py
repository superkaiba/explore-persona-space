#!/usr/bin/env python3
# Research notation (−) is intentional in labels.
# ruff: noqa: RUF001, RUF003
"""Task #606 — figures (hero candidates + exploratory dump, plan §6).

Reads ``analysis.json`` + the stage-A trajectory per behavior and writes to
``figures/issue_606/`` via the paper-plots conventions.

Hero candidates: (1) leakage-vs-strength overlay (x = s, y = bystander-mean
delta; LoRA orange, FT blue; matched band shaded); (2) matched-strength 2-bar
gap with CI; (3) per-persona profile scatter at s* (identity line, twins vs
roster vs negative-members marked, rho annotated). Exploratory: s(step)
trajectories; per-persona heatmaps (clean AND raw — raw alongside processed);
gap-vs-s* sweep; roster-vs-twins split; default-context slices; response-
length distributions (refusal length covariate). Hero selection happens at
analysis time with all views produced.

Usage::

    uv run python scripts/issue_606/i606_figures.py --behavior sycophancy \
        --eval-root eval_results/issue_606 [--out-dir figures/issue_606]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "issue_606"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from i606_common import (  # noqa: E402
    REFUSAL_EXPECTED_NEGATIVES,
    S_BAND,
    S_TARGET,
    SOURCE_PERSONA,
    SYCO_EXPECTED_NEGATIVES,
    TWIN_PROMPTS,
)

log = logging.getLogger("issue_606.figures")

ARM_COLORS = {"lora": "tab:orange", "ft": "tab:blue"}
NEG_MEMBERS = {
    "sycophancy": set(SYCO_EXPECTED_NEGATIVES),
    "refusal": set(REFUSAL_EXPECTED_NEGATIVES),
}


def _save(fig, name: str, out_dir: Path) -> None:
    from explore_persona_space.analysis.paper_plots import savefig_paper

    savefig_paper(fig, name, dir=out_dir)
    plt.close(fig)
    log.info("figure -> %s/%s", out_dir, name)


def _arm_points(analysis: dict) -> dict[str, list[tuple[str, float, float]]]:
    """Per arm: (cell, s_stage_b, bystander-mean clean delta) sorted by s."""
    s = analysis["s_stage_b"]
    tables = analysis["per_cell_tables"]
    bystanders = list(analysis["per_persona_at_target"]["lora"])
    out: dict[str, list[tuple[str, float, float]]] = {"lora": [], "ft": []}
    for cell, s_val in s.items():
        arm = "lora" if cell.startswith("lora_") else "ft"
        deltas = [
            tables[cell][p]["delta_clean"]
            for p in bystanders
            if tables[cell][p]["delta_clean"] is not None
        ]
        out[arm].append((cell, s_val, float(np.nanmean(deltas))))
    for arm in out:
        out[arm].sort(key=lambda t: t[1])
    return out


def fig_leakage_vs_strength(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Hero candidate 1: bystander-mean leakage vs implant strength."""
    pts = _arm_points(analysis)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for arm, rows in pts.items():
        xs = [0.0, *(r[1] for r in rows)]
        ys = [0.0, *(r[2] for r in rows)]
        ax.plot(
            xs, ys, "o-", color=ARM_COLORS[arm], label=f"{'LoRA' if arm == 'lora' else 'full FT'}"
        )
    ax.axvspan(S_BAND[0], S_BAND[1], alpha=0.10, color="grey")
    ax.axvline(S_TARGET, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("source-implant strength s (source-self rate delta)")
    ax.set_ylabel("bystander-mean leakage delta")
    ax.set_title(f"{behavior}: leakage vs implant strength")
    ax.legend()
    _save(fig, f"{behavior}_leakage_vs_strength_hero", out_dir)


def fig_matched_gap_bars(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Hero candidate 2: matched-strength 2-bar comparison with gap CI."""
    h = analysis["headline"]
    fig, ax = plt.subplots(figsize=(4.0, 4.2))
    means = [h["lora_bystander_mean"], h["ft_bystander_mean"]]
    ax.bar(["LoRA", "full FT"], means, color=[ARM_COLORS["lora"], ARM_COLORS["ft"]], width=0.6)
    gap, (lo, hi) = h["gap_plugin"], h["gap_ci95"]
    ax.errorbar(
        1,
        means[1],
        yerr=[[max(0.0, gap - lo)], [max(0.0, hi - gap)]],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.set_ylabel(f"bystander-mean leakage delta at s*={h['s_target']}")
    ax.set_title(
        f"{behavior}: gap (FT − LoRA) = {gap:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}] ({h['mode']})"
    )
    _save(fig, f"{behavior}_matched_gap_bars_hero", out_dir)


def fig_profile_scatter(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Hero candidate 3: per-persona LoRA vs FT at s* (identity line)."""
    per = analysis["per_persona_at_target"]
    rho = analysis["headline"]["profile_spearman_rho"]
    negs = NEG_MEMBERS[behavior]
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    groups = {"twin": [], "roster": [], "negative-member": []}
    for p in per["lora"]:
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
            [per["lora"][p] for p in members],
            [per["ft"][p] for p in members],
            s=28,
            label=kind,
            **styles[kind],
        )
    lim = [
        min(min(per["lora"].values()), min(per["ft"].values())) - 0.05,
        max(max(per["lora"].values()), max(per["ft"].values())) + 0.05,
    ]
    ax.plot(lim, lim, color="black", lw=0.8, ls=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("LoRA per-persona leakage delta at s*")
    ax.set_ylabel("full-FT per-persona leakage delta at s*")
    ax.set_title(f"{behavior}: per-persona profile (Spearman rho = {rho:.2f})")
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_profile_scatter_hero", out_dir)


def fig_trajectories(trajectory: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: stage-A s(step) per arm."""
    cells = trajectory["cells"]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for arm in ("lora", "ft"):
        rows = sorted(
            ((rec["step"], rec.get("s")) for c, rec in cells.items() if rec["arm"] == arm),
            key=lambda t: t[0],
        )
        rows = [(st, s) for st, s in rows if s is not None]
        if rows:
            ax.plot(
                [r[0] for r in rows],
                [r[1] for r in rows],
                "o-",
                color=ARM_COLORS[arm],
                label=f"{'LoRA' if arm == 'lora' else 'full FT'} (stage-A native)",
            )
    ax.axhspan(S_BAND[0], S_BAND[1], alpha=0.10, color="grey")
    ax.axhline(S_TARGET, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("source-implant strength s")
    ax.set_title(f"{behavior}: stage-A install trajectory")
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_stage_a_trajectory", out_dir)


def fig_heatmaps(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: per-persona delta heatmaps, CLEAN and RAW side by side."""
    tables = analysis["per_cell_tables"]
    s = analysis["s_stage_b"]
    cells = sorted(s, key=lambda c: s[c])
    personas = sorted(p for p in next(iter(tables.values())) if p != SOURCE_PERSONA)
    negs = NEG_MEMBERS[behavior]
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
    """Exploratory: gap vs matched-strength target."""
    sweep = analysis["sweep"]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    xs = [r["target"] for r in sweep]
    ys = [r["gap_plugin"] for r in sweep]
    lo = [max(0.0, r["gap_plugin"] - r["gap_ci"][0]) for r in sweep]
    hi = [max(0.0, r["gap_ci"][1] - r["gap_plugin"]) for r in sweep]
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color="tab:purple", capsize=3)
    for r in sweep:
        if not r["in_range_both_arms"]:
            ax.plot(r["target"], r["gap_plugin"], "x", color="red", ms=10)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhspan(-0.05, 0.05, alpha=0.10, color="green")
    ax.axvline(S_TARGET, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("matched-strength target s*")
    ax.set_ylabel("gap (FT − LoRA), bystander mean")
    ax.set_title(f"{behavior}: gap vs matched-strength target (x = out of range)")
    _save(fig, f"{behavior}_gap_vs_target_sweep", out_dir)


def fig_subset_splits(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: roster-vs-twins split + default-context slices at s*."""
    per = analysis["per_persona_at_target"]
    groups = {
        "roster": [p for p in per["lora"] if p not in TWIN_PROMPTS],
        "twins": [p for p in per["lora"] if p in TWIN_PROMPTS],
        "qwen_default": [p for p in per["lora"] if p == "qwen_default"],
        "assistant": [p for p in per["lora"] if p == "assistant"],
    }
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(groups))
    for i, (arm, color) in enumerate(ARM_COLORS.items()):
        means = [
            float(np.nanmean([per[arm][p] for p in members])) if members else np.nan
            for members in groups.values()
        ]
        ax.bar(x + (i - 0.5) * 0.35, means, width=0.35, color=color, label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels(list(groups), fontsize=8)
    ax.set_ylabel("mean leakage delta at s*")
    ax.set_title(f"{behavior}: subset splits at s*")
    ax.legend()
    _save(fig, f"{behavior}_subset_splits", out_dir)


def fig_length_distributions(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: response-length distributions per cell (the #518 length
    covariate; chars as stored in verdict rows)."""
    tables = analysis["per_cell_tables"]
    cells = sorted(tables)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    means = [
        float(
            np.nanmean(
                [
                    tables[c][p]["mean_completion_chars"]
                    for p in tables[c]
                    if tables[c][p]["mean_completion_chars"] is not None
                ]
            )
        )
        for c in cells
    ]
    colors = [
        ARM_COLORS["lora"]
        if c.startswith("lora_")
        else (ARM_COLORS["ft"] if c.startswith("ft_") else "tab:grey")
        for c in cells
    ]
    ax.bar(range(len(cells)), means, color=colors)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, fontsize=7, rotation=45)
    ax.set_ylabel("mean completion length (chars)")
    ax.set_title(f"{behavior}: response length per cell (panel mean)")
    _save(fig, f"{behavior}_response_lengths", out_dir)


def _pretty_persona(p: str) -> str:
    """Reader-facing persona tick label: snake_case slug -> sentence case with spaces."""
    label = p.replace("_", " ").capitalize()
    return (
        label.replace("Ai ", "AI ")
        if label.startswith("Ai ")
        else ("AI" if label == "Ai" else label)
    )


def fig_endpoint_comparison(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: endpoint (near-ceiling) descriptive per-persona bars."""
    tables = analysis["per_cell_tables"]
    s = analysis["s_stage_b"]
    arm_names = {"lora": "LoRA", "ft": "Full fine-tuning"}
    endpoints = {}
    for arm in ("lora", "ft"):
        arm_cells = [c for c in s if c.startswith(f"{arm}_")]
        if arm_cells:
            endpoints[arm] = max(arm_cells, key=lambda c: int(c.split("step")[-1]))
    if len(endpoints) < 2:
        return
    personas = sorted(p for p in tables[endpoints["lora"]] if p != SOURCE_PERSONA)
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(personas)), 4.0))
    x = np.arange(len(personas))
    for i, (arm, cell) in enumerate(endpoints.items()):
        ys = [tables[cell][p]["delta_clean"] for p in personas]
        ax.bar(
            x + (i - 0.5) * 0.4,
            ys,
            width=0.4,
            color=ARM_COLORS[arm],
            label=f"{arm_names[arm]} (endpoint, step {cell.split('step')[-1]})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_persona(p) for p in personas], fontsize=5, rotation=90)
    ax.set_ylabel("leakage delta (clean)")
    ax.set_title(f"{behavior}: endpoint cells, descriptive (near ceiling — not the headline)")
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_endpoint_descriptive", out_dir)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=figures] %(message)s")
    p = argparse.ArgumentParser(description="#606 figures (heroes + exploratory dump).")
    p.add_argument("--behavior", required=True, choices=["sycophancy", "refusal"])
    p.add_argument("--eval-root", type=Path, default=REPO / "eval_results" / "issue_606")
    p.add_argument("--out-dir", type=Path, default=REPO / "figures" / "issue_606")
    args = p.parse_args(argv)

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    broot = args.eval_root / args.behavior
    analysis = json.loads((broot / "analysis.json").read_text())
    trajectory = json.loads((broot / "stage_a" / f"trajectory_{args.behavior}.json").read_text())

    fig_leakage_vs_strength(analysis, args.behavior, args.out_dir)
    fig_matched_gap_bars(analysis, args.behavior, args.out_dir)
    fig_profile_scatter(analysis, args.behavior, args.out_dir)
    fig_trajectories(trajectory, args.behavior, args.out_dir)
    fig_heatmaps(analysis, args.behavior, args.out_dir)
    fig_sweep(analysis, args.behavior, args.out_dir)
    fig_subset_splits(analysis, args.behavior, args.out_dir)
    fig_length_distributions(analysis, args.behavior, args.out_dir)
    fig_endpoint_comparison(analysis, args.behavior, args.out_dir)
    log.info("all figures written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
