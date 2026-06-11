#!/usr/bin/env python3
"""Issue #563 follow-up `fixed-completion-force-read` figures (VM, CPU-only).

Reads ``eval_results/issue_563/fixed-completion-force-read/rollup.json`` and
writes paper-plots-conformant figures to ``figures/issue_563/force_read/``
(plan v2 section 5):

  - HERO ``fixed_vs_own_content_rise``: per role cell, the fixed-content rise
    (this run, 95% CI) next to the own-content rise (#563 v1 committed), with
    the 0.5 x R'_c registered threshold marked per cell.
  - Exploratory dump: per-row paired scatter fixed-content delta vs
    own-content delta per cell; decomposition panel (z_marker vs logZ, fixed
    content); EOS-margin-space panel; [0:50] parity; absolute log P per cell
    both arms.

Usage:
    uv run python scripts/plot_issue563_force_read.py
    uv run python scripts/plot_issue563_force_read.py \\
        --rollup /tmp/fixture/rollup.json --fig-dir /tmp/figs --stem-suffix _smoke
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

bootstrap(log_name="plot_issue563_force_read")

from eval_issue563_force_read import COMMITTED_ROLLUP, OUT_DIR, ROLE_CELLS  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

log = logging.getLogger("plot_issue563_force_read")

DEFAULT_FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "issue_563" / "force_read"

# Plain-English cell labels (never slugs in figure text).
CELL_LABELS = {
    "doctor": "Doctor",
    "software_engineer": "Software engineer",
    "french_person": "French person",
    "police_officer": "Police officer",
}


def _yerr(mean: float, ci: list[float]) -> list[list[float]]:
    """Asymmetric errorbar lengths; clamped >=0 (constant-bootstrap guard)."""
    return [[max(0.0, mean - ci[0])], [max(0.0, ci[1] - mean)]]


def present_cells(panel: dict) -> list[str]:
    cells = [c for c in ROLE_CELLS if c in panel["cells"]]
    if not cells:
        raise RuntimeError(f"No known panel cells in rollup: {sorted(panel['cells'])}")
    return cells


def plot_hero_fixed_vs_own(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """HERO: fixed-content rise (this run) vs own-content rise (v1), with the
    registered 0.5 x R'_c threshold marked per cell."""
    panel = rollup["panel"]
    cells = present_cells(panel)
    fig, ax = plt.subplots()
    xs = np.arange(len(cells))
    w = 0.34
    col_fixed = paper_palette_role("primary")
    col_own = paper_palette_role("baseline")
    for i, cell in enumerate(cells):
        sbs = panel["cells"][cell]["side_by_side"]
        fixed = sbs["fixed_content_rise"]
        own = sbs["own_content_rise"]
        ax.bar(
            i - w / 2,
            fixed["mean"],
            width=w,
            color=col_fixed,
            label="Same answers, role prompt swapped in (this run)" if i == 0 else None,
        )
        ax.errorbar(
            i - w / 2,
            fixed["mean"],
            yerr=_yerr(fixed["mean"], fixed["ci95"]),
            fmt="none",
            capsize=4,
            color="black",
            linewidth=1.0,
        )
        ax.bar(
            i + w / 2,
            own["mean"],
            width=w,
            color=col_own,
            label="Model's own answers under the role prompt (prior run)" if i == 0 else None,
        )
        ax.errorbar(
            i + w / 2,
            own["mean"],
            yerr=_yerr(own["mean"], own["ci95"]),
            fmt="none",
            capsize=4,
            color="black",
            linewidth=1.0,
        )
        thr = panel["cells"][cell]["classification"]["threshold_half_r_c"]
        ax.hlines(
            thr,
            i - 0.45,
            i + 0.45,
            linestyles=":",
            linewidth=1.2,
            color=paper_palette_role("accent"),
            label="Registered half-effect threshold" if i == 0 else None,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=paper_palette_role("neutral"))
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylabel("Marker log P rise vs assistant prompt (nats)")
    ax.set_title("Does the rise survive when the answers are frozen?")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_paired_scatter(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Exploratory: per-row fixed-content delta vs own-content delta per cell."""
    panel = rollup["panel"]
    cells = present_cells(panel)
    own_cells = json.loads(COMMITTED_ROLLUP.read_text())["panel"]["cells"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=False, sharey=False)
    for ax, cell in zip(axes.flat, cells, strict=False):
        fixed = np.asarray(panel["cells"][cell]["per_row_d_logp"], dtype=float)
        own = np.asarray(own_cells[cell]["per_question_d_logp"], dtype=float)[: len(fixed)]
        ax.scatter(own, fixed, s=10, alpha=0.4, color=paper_palette_role("primary"))
        lims = [min(own.min(), fixed.min()), max(own.max(), fixed.max())]
        ax.plot(lims, lims, linestyle="--", linewidth=1.0, color=paper_palette_role("neutral"))
        ax.axhline(0.0, linewidth=0.8, color=paper_palette_role("neutral"))
        ax.axvline(0.0, linewidth=0.8, color=paper_palette_role("neutral"))
        rho = panel["cells"][cell]["spearman_fixed_vs_own_d_logp"]
        rho_txt = f"Spearman rho = {rho:.2f}" if rho is not None else "Spearman rho: n/a"
        ax.set_title(f"{CELL_LABELS[cell]} ({rho_txt})", fontsize=9)
        ax.set_xlabel("Own-answer rise per question (nats)", fontsize=8)
        ax.set_ylabel("Frozen-answer rise per question (nats)", fontsize=8)
    fig.suptitle("Per-question rise: frozen answers vs the model's own answers", fontsize=11)
    fig.tight_layout()
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_decomposition(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Exploratory: d_logp = d_z_marker - d_logZ per cell, fixed content."""
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
    ax.set_ylabel("Paired delta vs assistant prompt (nats)")
    ax.set_title("Decomposition of the frozen-answer rise (Δ log P = Δ z_marker - Δ log Z)")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_eos_margin(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Companion: the rise in EOS-margin (logit) space, fixed content."""
    panel = rollup["panel"]
    cells = present_cells(panel)
    fig, ax = plt.subplots()
    xs = np.arange(len(cells))
    for i, cell in enumerate(cells):
        s = panel["cells"][cell]["d_eos_margin"]
        ax.bar(i, s["mean"], width=0.5, color=paper_palette_role("primary"))
        ax.errorbar(
            i,
            s["mean"],
            yerr=_yerr(s["mean"], s["ci95"]),
            fmt="none",
            capsize=4,
            color="black",
            linewidth=1.0,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=paper_palette_role("neutral"))
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylabel("Δ (z_marker - z_EOS) vs assistant prompt (nats)")
    ax.set_title("Frozen-answer rise — logit (EOS-margin) space")
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_parity_subset(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Parity check: full-n mean +- CI vs the [0:50] subset mean."""
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
            label="Full row set" if i == 0 else None,
        )
        ax.errorbar(
            i + 0.12,
            sub["mean"],
            yerr=_yerr(sub["mean"], sub["ci95"]),
            fmt="o",
            markersize=6,
            capsize=5,
            color=paper_palette_role("accent"),
            label="First 50 rows" if i == 0 else None,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=paper_palette_role("neutral"))
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylabel("Δ log P(marker) vs assistant prompt (nats)")
    ax.set_title("Full-set vs first-50-subset agreement (frozen answers)")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def plot_absolute_logp(rollup: dict, *, stem: str, fig_dir: Path) -> None:
    """Raw alongside processed: absolute marker log P per cell, both arms."""
    panel = rollup["panel"]
    cells = present_cells(panel)
    committed = json.loads(COMMITTED_ROLLUP.read_text())
    own_assist = committed["assistant_cell"]["logp_mean"]
    fixed_assist = rollup["diagonal"]["logp_mean"]
    fixed_abs = [fixed_assist + panel["cells"][c]["d_logp"]["mean"] for c in cells]
    own_abs = [own_assist + committed["panel"]["cells"][c]["d_logp"]["mean"] for c in cells]
    labels = ["Assistant", *[CELL_LABELS[c] for c in cells]]
    fig, ax = plt.subplots()
    xs = np.arange(len(labels))
    ax.plot(
        xs,
        [fixed_assist, *fixed_abs],
        marker="o",
        color=paper_palette_role("primary"),
        label="Frozen answers (this run)",
    )
    ax.plot(
        xs,
        [own_assist, *own_abs],
        marker="s",
        color=paper_palette_role("baseline"),
        label="Model's own answers (prior run)",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean log P(marker) at the slot (nats)")
    ax.set_title("Absolute marker log-prob per prompt, both completion sources")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", fig_dir, stem)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #563 fixed-completion force-read figures (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rollup", type=str, default=str(OUT_DIR / "rollup.json"))
    p.add_argument("--fig-dir", type=str, default=str(DEFAULT_FIG_DIR))
    p.add_argument("--stem-suffix", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rollup = json.loads(Path(args.rollup).read_text())
    fig_dir = Path(args.fig_dir)
    sfx = args.stem_suffix
    set_paper_style("blog")

    plot_hero_fixed_vs_own(rollup, stem=f"fixed_vs_own_content_rise{sfx}", fig_dir=fig_dir)
    plot_paired_scatter(rollup, stem=f"per_row_fixed_vs_own_scatter{sfx}", fig_dir=fig_dir)
    plot_decomposition(rollup, stem=f"logz_decomposition_fixed{sfx}", fig_dir=fig_dir)
    plot_eos_margin(rollup, stem=f"eos_margin_fixed{sfx}", fig_dir=fig_dir)
    plot_parity_subset(rollup, stem=f"parity_subset_check{sfx}", fig_dir=fig_dir)
    plot_absolute_logp(rollup, stem=f"raw_absolute_logp{sfx}", fig_dir=fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
