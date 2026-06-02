"""Generate figures for #465 -- per plan v2 §6.3.

Reads:
  * eval_results/issue_465/per_cell/G_<cond>__<shape>.json (per-cell raw)
  * eval_results/issue_465/analysis.json (CIs + diagnostics)

Writes (round-2: 5 figures shipped here; the rest are over-produce candidates
the analyzer can add when the data motivates them, per plan §6.3):
  * figures/issue_465/hero_4x3_grid.png
  * figures/issue_465/hero_demo_free_default_disentangled.png  (H3a/b/c annotated)
  * figures/issue_465/hero_retention.png                       (co-primary H3d)
  * figures/issue_465/diagonal_implant_bar.png                 (H1a/b/c sanity)
  * figures/issue_465/per_q_distribution_violin.png            (saturation surface)

Deferred to a separate WandB-export script (trajectory_curves.png from the
in-training MarkerLogprobTrajectoryCallback) and to opt-in additions the
analyzer can produce from the per-cell JSONs:
  * k_sweep_lines.png, non_marker_demo_compare.png, villain_R_parity.png,
    per_q_raw_g_and_b_logprob.png, emission_rate_table.png

CLI:
    uv run python scripts/i465_make_figures.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import set_paper_style
from explore_persona_space.experiments.i465_data import (
    CONDITION_IDS,
    CONDITION_NAMES,
)

logger = logging.getLogger("i465.figures")

OUT_DIR = Path("eval_results/issue_465")
PER_CELL_DIR = OUT_DIR / "per_cell"
FIG_DIR = Path("figures/issue_465")

# Colorblind-safe palette (per CLAUDE.md paper-plots skill).
PALETTE = {
    "cond1": "#1f77b4",  # blue
    "cond2_k0": "#ff7f0e",  # orange
    "cond2_k1": "#2ca02c",  # green
    "cond2_k3": "#d62728",  # red
}

PRIMARY_SHAPES = [
    "in_trained_shape",
    "generalization",
    "demo_free_default",
]
SHAPE_LABELS = {
    "in_trained_shape": "In-trained-shape\n(villain-R)",
    "generalization": "Generalization\n(villain-R, fresh q)",
    "demo_free_default": "Demo-free default\n(helpful-R, PRIMARY)",
    "demo_free_default_villain_R": "Demo-free default\n(villain-R, parity)",
    "non_marker_demo": "Non-marker-demo\n(copy control)",
}


def _load_cell(cond: str, shape: str) -> dict | None:
    p = PER_CELL_DIR / f"G_{cond}__{shape}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _per_q_dg(cell: dict) -> np.ndarray:
    g = np.array(cell["g_logps_per_q"], dtype=float)
    b = np.array(cell["b_logps_per_q"], dtype=float)
    return g - b


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def _bootstrap_ci_mean(values: np.ndarray, n: int = 10_000, seed: int = 42) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def fig_4x3_grid(cells: dict, analysis: dict) -> None:
    fig, axes = plt.subplots(1, len(PRIMARY_SHAPES), figsize=(13, 4), sharey=True)
    for ax, shape in zip(axes, PRIMARY_SHAPES, strict=True):
        means = []
        cis_lo = []
        cis_hi = []
        labels = []
        colors = []
        for cond in CONDITION_IDS:
            cell = cells.get((cond, shape))
            if cell is None:
                means.append(0.0)
                cis_lo.append(0.0)
                cis_hi.append(0.0)
                labels.append(CONDITION_NAMES[cond])
                colors.append(PALETTE[cond])
                continue
            dg = _per_q_dg(cell)
            mean = float(dg.mean())
            lo, hi = _bootstrap_ci_mean(dg)
            means.append(mean)
            cis_lo.append(mean - lo)
            cis_hi.append(hi - mean)
            labels.append(CONDITION_NAMES[cond])
            colors.append(PALETTE[cond])
        x = np.arange(len(labels))
        ax.bar(
            x,
            means,
            yerr=[cis_lo, cis_hi],
            color=colors,
            capsize=4,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(SHAPE_LABELS[shape], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_IDS[i] for i in range(len(x))], rotation=20, ha="right", fontsize=8
        )
        ax.axhline(0, color="black", linewidth=0.5)
        if shape == PRIMARY_SHAPES[0]:
            ax.set_ylabel("Delta G = trained - base log P(' ※')  [nats]")
    fig.suptitle("Delta G by condition x eval shape (4 cond x 3 primary shapes)", fontsize=12)
    _save(fig, "hero_4x3_grid.png")


def fig_demo_free_default_disentangled(cells: dict, analysis: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = []
    cis = []
    for cond in CONDITION_IDS:
        cell = cells.get((cond, "demo_free_default"))
        if cell is None:
            means.append(0.0)
            cis.append((0.0, 0.0))
            continue
        dg = _per_q_dg(cell)
        m = float(dg.mean())
        lo, hi = _bootstrap_ci_mean(dg)
        means.append(m)
        cis.append((m - lo, hi - m))
    x = np.arange(len(CONDITION_IDS))
    ax.bar(
        x,
        means,
        yerr=np.array(cis).T,
        color=[PALETTE[c] for c in CONDITION_IDS],
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_NAMES[c] for c in CONDITION_IDS], rotation=15, ha="right", fontsize=9
    )
    ax.set_ylabel("Delta G at demo-free default (helpful-R)  [nats]")
    ax.axhline(0, color="black", linewidth=0.5)
    title = "Demo-free default -- disentangled H3 contrasts"
    # Annotate H3 contrasts when present.
    h3 = analysis.get("h3_disentangled", {})
    notes = []
    for key, label in [
        ("H3a", "cond1-cond2_k1"),
        ("H3b", "cond2_k0-cond2_k1"),
        ("H3c", "cond1-cond2_k0"),
    ]:
        row = h3.get(key)
        if row and "diff_mean" in row:
            notes.append(
                f"{key} ({label}): {row['diff_mean']:+.2f} "
                f"[{row['ci_95'][0]:+.2f}, {row['ci_95'][1]:+.2f}]"
            )
    if notes:
        ax.text(
            0.02,
            0.98,
            "\n".join(notes),
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            family="monospace",
        )
    ax.set_title(title)
    _save(fig, "hero_demo_free_default_disentangled.png")


def fig_retention(analysis: dict) -> None:
    rp = analysis.get("retention_point_estimates", {})
    fig, ax = plt.subplots(figsize=(7, 4))
    conds = [c for c in CONDITION_IDS if c in rp]
    vals = [rp[c]["retention"] if rp[c]["retention"] is not None else 0.0 for c in conds]
    ax.bar(
        np.arange(len(conds)),
        vals,
        color=[PALETTE[c] for c in conds],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels([CONDITION_NAMES[c] for c in conds], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Retention = Delta G[demo-free default] / Delta G[in-trained-shape]")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title("Implant-strength-normalized retention (co-primary headline)")
    _save(fig, "hero_retention.png")


def fig_diagonal_implant(cells: dict, analysis: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    means = []
    cis = []
    for cond in CONDITION_IDS:
        cell = cells.get((cond, "in_trained_shape"))
        if cell is None:
            means.append(0.0)
            cis.append((0.0, 0.0))
            continue
        dg = _per_q_dg(cell)
        m = float(dg.mean())
        lo, hi = _bootstrap_ci_mean(dg)
        means.append(m)
        cis.append((m - lo, hi - m))
    x = np.arange(len(CONDITION_IDS))
    ax.bar(
        x,
        means,
        yerr=np.array(cis).T,
        color=[PALETTE[c] for c in CONDITION_IDS],
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_NAMES[c] for c in CONDITION_IDS], rotation=15, ha="right", fontsize=9
    )
    ax.set_ylabel("Delta G at in-trained-shape (diagonal)  [nats]")
    ax.axhline(5, color="grey", linewidth=0.5, linestyle="--", label="H1b/c threshold (+5)")
    ax.axhline(15, color="grey", linewidth=0.5, linestyle=":", label="H1a threshold (+15)")
    ax.set_title("Diagonal implant -- H1a/b/c sanity")
    ax.legend(fontsize=8)
    _save(fig, "diagonal_implant_bar.png")


def fig_per_q_violin(cells: dict) -> None:
    cell_list = []
    labels = []
    for cond in CONDITION_IDS:
        for shape in PRIMARY_SHAPES:
            cell = cells.get((cond, shape))
            if cell is None:
                continue
            cell_list.append(_per_q_dg(cell))
            labels.append(f"{cond}\n{shape}")
    if not cell_list:
        logger.warning("no cells for violin plot")
        return
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 4))
    ax.violinplot(cell_list, showmeans=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Per-q Delta G  [nats]")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Per-q Delta G distribution per cell (surfaces saturation patterns)")
    _save(fig, "per_q_distribution_violin.png")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)

    set_paper_style()

    cells: dict[tuple[str, str], dict] = {}
    for cond in CONDITION_IDS:
        for shape in [*PRIMARY_SHAPES, "demo_free_default_villain_R", "non_marker_demo"]:
            cell = _load_cell(cond, shape)
            if cell is not None:
                cells[(cond, shape)] = cell
    if not cells:
        raise FileNotFoundError(f"No per-cell JSONs found under {PER_CELL_DIR}.")
    analysis_path = OUT_DIR / "analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"analysis.json missing at {analysis_path}; run Phase 5 first.")
    analysis = json.loads(analysis_path.read_text())

    fig_4x3_grid(cells, analysis)
    fig_demo_free_default_disentangled(cells, analysis)
    fig_retention(analysis)
    fig_diagonal_implant(cells, analysis)
    fig_per_q_violin(cells)
    logger.info("figures written -> %s", FIG_DIR)


if __name__ == "__main__":
    main()
