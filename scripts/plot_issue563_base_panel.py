#!/usr/bin/env python3
"""Issue #563 figures — base-own persona-panel plots (VM, CPU-only).

Reads ``eval_results/issue_563/rollup.json`` and writes paper-plots-conformant
figures to ``figures/issue_563/`` (plan sections 3.4 / 5):

  - HERO ``base_prior_rise_side_by_side``: per persona cell, the base-prior
    rise vs the assistant cell — base-own completions (this run, question-
    bootstrap 95% CI) next to on-FT completions (#558, 12-adapter cluster-
    bootstrap CI). One glance answers the Goal.
  - Exploratory dump (analyzer picks): absolute base log P per cell both
    arms; EOS-margin-space panel; logZ decomposition; per-question paired
    scatter; completion-length covariate panel; quality-rates table;
    [0:50]-vs-full parity check.

Usage:
    uv run python scripts/plot_issue563_base_panel.py
    uv run python scripts/plot_issue563_base_panel.py \\
        --rollup eval_results/issue_563/rollup_smoke.json --stem-suffix _smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="plot_issue563_base_panel")

from eval_issue563_base_panel import EVAL_RESULTS_DIR_563  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

log = logging.getLogger("plot_issue563_base_panel")

DEFAULT_FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "issue_563"

# Plain-English cell labels (never slugs in figure text).
CELL_LABELS = {
    "doctor": "Doctor",
    "software_engineer": "Software engineer",
    "french_person": "French person",
    "police_officer": "Police officer",
}
CELL_ORDER = ("doctor", "software_engineer", "french_person", "police_officer")


def _yerr(mean: float, ci: list[float]) -> list[list[float]]:
    """Asymmetric errorbar lengths; clamped >=0 (constant-bootstrap guard)."""
    return [[max(0.0, mean - ci[0])], [max(0.0, ci[1] - mean)]]


def present_cells(panel: dict) -> list[str]:
    cells = [c for c in CELL_ORDER if c in panel["cells"]]
    if not cells:
        raise RuntimeError(f"No known panel cells in rollup: {sorted(panel['cells'])}")
    return cells


def plot_hero_side_by_side(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """HERO: base-prior rise per cell — base-own (this run) vs on-FT (#558)."""
    panel = rollup["panel"]
    parent = rollup["parent_base_side"]
    cells = present_cells(panel)
    fig, ax = plt.subplots()
    xs = np.arange(len(cells))
    w = 0.34
    col_own = paper_palette_role("primary")
    col_ft = paper_palette_role("baseline")
    for i, cell in enumerate(cells):
        own = panel["cells"][cell]["d_logp"]
        ft = parent[cell]
        ax.bar(
            i - w / 2,
            own["mean"],
            width=w,
            color=col_own,
            label="Base-own completions (this run)" if i == 0 else None,
        )
        ax.errorbar(
            i - w / 2,
            own["mean"],
            yerr=_yerr(own["mean"], own["ci95"]),
            fmt="none",
            capsize=4,
            color="black",
            linewidth=1.0,
        )
        ax.bar(
            i + w / 2,
            ft["logp_rise_mean"],
            width=w,
            color=col_ft,
            label="On fine-tuned completions (parent run)" if i == 0 else None,
        )
        ax.errorbar(
            i + w / 2,
            ft["logp_rise_mean"],
            yerr=_yerr(ft["logp_rise_mean"], ft["logp_rise_ci95"]),
            fmt="none",
            capsize=4,
            color="black",
            linewidth=1.0,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=paper_palette_role("neutral"))
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylabel("Base log P(marker) rise vs assistant cell (nats)")
    ax.set_title("Base-prior rise per persona prompt: own vs fine-tuned completions")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_eos_margin_panel(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Companion: same layout in the EOS-margin (logit) space."""
    panel = rollup["panel"]
    parent = rollup["parent_base_side"]
    cells = present_cells(panel)
    fig, ax = plt.subplots()
    xs = np.arange(len(cells))
    w = 0.34
    for i, cell in enumerate(cells):
        own = panel["cells"][cell]["d_eos_margin"]
        ft = parent[cell]
        ax.bar(
            i - w / 2,
            own["mean"],
            width=w,
            color=paper_palette_role("primary"),
            label="Base-own completions (this run)" if i == 0 else None,
        )
        ax.errorbar(
            i - w / 2,
            own["mean"],
            yerr=_yerr(own["mean"], own["ci95"]),
            fmt="none",
            capsize=4,
            color="black",
            linewidth=1.0,
        )
        ax.bar(
            i + w / 2,
            ft["eosm_rise_mean"],
            width=w,
            color=paper_palette_role("baseline"),
            label="On fine-tuned completions (parent run)" if i == 0 else None,
        )
        ax.errorbar(
            i + w / 2,
            ft["eosm_rise_mean"],
            yerr=_yerr(ft["eosm_rise_mean"], ft["eosm_rise_ci95"]),
            fmt="none",
            capsize=4,
            color="black",
            linewidth=1.0,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=paper_palette_role("neutral"))
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylabel("Base EOS-margin rise vs assistant cell (nats)")
    ax.set_title("Base-prior rise per persona prompt — logit (EOS-margin) space")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_logz_decomposition(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Exploratory: d_logp = d_z_marker - d_logZ per cell (saturation diagnostic)."""
    panel = rollup["panel"]
    cells = present_cells(panel)
    fig, ax = plt.subplots()
    xs = np.arange(len(cells))
    w = 0.26
    parts = [("d_logp", "Δ log P(marker)"), ("d_z_marker", "Δ z_marker"), ("d_logZ", "Δ log Z")]
    colors = paper_palette(3)
    for j, (key, lbl) in enumerate(parts):
        for i, cell in enumerate(cells):
            s = panel["cells"][cell][key]
            ax.bar(
                i + (j - 1) * w, s["mean"], width=w, color=colors[j], label=lbl if i == 0 else None
            )
            ax.errorbar(
                i + (j - 1) * w,
                s["mean"],
                yerr=_yerr(s["mean"], s["ci95"]),
                fmt="none",
                capsize=3,
                color="black",
                linewidth=0.9,
            )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=paper_palette_role("neutral"))
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylabel("Paired delta vs assistant cell (nats)")
    ax.set_title("Decomposition of the base-prior rise (Δ log P = Δ z_marker - Δ log Z)")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_absolute_logp(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Raw alongside processed: absolute base log P per cell, both arms."""
    panel = rollup["panel"]
    parent = rollup["parent_base_side"]
    cells = present_cells(panel)
    own_assist = rollup["assistant_cell"]["logp_mean"]
    ft_assist = parent["assistant_logp_base_mean"]
    own_abs = [own_assist + panel["cells"][c]["d_logp"]["mean"] for c in cells]
    ft_abs = [ft_assist + parent[c]["logp_rise_mean"] for c in cells]
    labels = ["Assistant", *[CELL_LABELS[c] for c in cells]]
    fig, ax = plt.subplots()
    xs = np.arange(len(labels))
    ax.plot(
        xs,
        [own_assist, *own_abs],
        marker="o",
        color=paper_palette_role("primary"),
        label="Base-own completions (this run)",
    )
    ax.plot(
        xs,
        [ft_assist, *ft_abs],
        marker="s",
        color=paper_palette_role("baseline"),
        label="On fine-tuned completions (parent run)",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean base log P(marker) at the slot (nats)")
    ax.set_title("Absolute base marker log-prob per cell, both completion sources")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_per_question_scatter(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Exploratory: per-question paired delta distribution per cell."""
    panel = rollup["panel"]
    cells = present_cells(panel)
    fig, ax = plt.subplots()
    rng = np.random.default_rng(0)  # cosmetic jitter only
    for i, cell in enumerate(cells):
        d = np.asarray(panel["cells"][cell]["per_question_d_logp"])
        jit = rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(i + jit, d, s=10, alpha=0.35, color=paper_palette_role("neutral"), zorder=2)
        s = panel["cells"][cell]["d_logp"]
        ax.errorbar(
            i,
            s["mean"],
            yerr=_yerr(s["mean"], s["ci95"]),
            fmt="D",
            markersize=6,
            capsize=5,
            color=paper_palette_role("primary"),
            zorder=3,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=paper_palette_role("baseline"))
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylabel("Per-question Δ base log P(marker) vs assistant (nats)")
    ax.set_title("Per-question paired deltas, base-own completions")
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_length_covariate(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Covariate panel: completion lengths this run vs parent's recorded lengths."""
    panel = rollup["panel"]
    parent_toks = rollup["parent_base_side"]["mean_tokens_per_cell"]
    cells = present_cells(panel)
    labels = ["Assistant", *[CELL_LABELS[c] for c in cells]]
    own = [rollup["assistant_cell"]["covariates"]["mean_tokens"]] + [
        panel["cells"][c]["covariates"]["mean_tokens"] for c in cells
    ]
    ft = [parent_toks["trigger50"]] + [parent_toks[c] for c in cells]
    fig, ax = plt.subplots()
    xs = np.arange(len(labels))
    w = 0.34
    ax.bar(
        xs - w / 2, own, width=w, color=paper_palette_role("primary"), label="Base-own (this run)"
    )
    ax.bar(
        xs + w / 2,
        ft,
        width=w,
        color=paper_palette_role("baseline"),
        label="Fine-tuned models (parent run)",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean generated tokens per completion")
    ax.set_title("Completion-length covariate: base-own vs fine-tuned completions")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_quality_rates(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Table figure: truncation / French / degenerate / key-mention / emission rates."""
    panel = rollup["panel"]
    cells = present_cells(panel)
    rows = ["Assistant", *[CELL_LABELS[c] for c in cells]]
    cols = ["Truncated", "French-flagged", "Degenerate (<5 tok)", "Mentions key", "Marker emitted"]
    covs = [rollup["assistant_cell"]["covariates"]] + [
        panel["cells"][c]["covariates"] for c in cells
    ]
    data = [
        [
            f"{cv['truncation_rate']:.3f}",
            f"{cv['french_flag_rate']:.3f}",
            f"{cv['degenerate_rate']:.3f}",
            f"{cv['key_word_mention_rate']:.3f}",
            f"{cv['emission_rate']:.3f}",
        ]
        for cv in covs
    ]
    fig, ax = plt.subplots()
    ax.axis("off")
    table = ax.table(cellText=data, rowLabels=rows, colLabels=cols, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    ax.set_title("Data-quality covariate rates per cell (base-own completions)")
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_parity_subset(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Parity check: full-n mean +- CI vs the [0:50] parent-parity subset mean."""
    panel = rollup["panel"]
    cells = present_cells(panel)
    fig, ax = plt.subplots()
    xs = np.arange(len(cells))
    for i, cell in enumerate(cells):
        full = panel["cells"][cell]["d_logp"]
        sub = panel["cells"][cell]["subset_0_50"]["d_logp"]
        ax.errorbar(
            i - 0.12,
            full["mean"],
            yerr=_yerr(full["mean"], full["ci95"]),
            fmt="D",
            markersize=6,
            capsize=5,
            color=paper_palette_role("primary"),
            label="Full question set" if i == 0 else None,
        )
        ax.errorbar(
            i + 0.12,
            sub["mean"],
            yerr=_yerr(sub["mean"], sub["ci95"]),
            fmt="o",
            markersize=6,
            capsize=5,
            color=paper_palette_role("accent"),
            label="Parent-parity subset (first 50)" if i == 0 else None,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=paper_palette_role("neutral"))
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylabel("Δ base log P(marker) vs assistant cell (nats)")
    ax.set_title("Full-set vs parent-parity-subset agreement")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #563 base-own persona-panel figures (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rollup", type=str, default=str(EVAL_RESULTS_DIR_563 / "rollup.json"))
    p.add_argument("--fig-dir", type=str, default=str(DEFAULT_FIG_DIR))
    p.add_argument("--stem-suffix", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rollup = json.loads(Path(args.rollup).read_text())
    fig_dir = Path(args.fig_dir)
    sfx = args.stem_suffix
    set_paper_style("blog")

    plot_hero_side_by_side(rollup, stem=f"base_prior_rise_side_by_side{sfx}", fig_dir=fig_dir)
    plot_eos_margin_panel(rollup, stem=f"eos_margin_panel{sfx}", fig_dir=fig_dir)
    plot_logz_decomposition(rollup, stem=f"logz_decomposition{sfx}", fig_dir=fig_dir)
    plot_absolute_logp(rollup, stem=f"raw_absolute_logp{sfx}", fig_dir=fig_dir)
    plot_per_question_scatter(rollup, stem=f"per_question_deltas{sfx}", fig_dir=fig_dir)
    plot_length_covariate(rollup, stem=f"length_covariate{sfx}", fig_dir=fig_dir)
    plot_quality_rates(rollup, stem=f"quality_rates{sfx}", fig_dir=fig_dir)
    plot_parity_subset(rollup, stem=f"parity_subset_check{sfx}", fig_dir=fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
