#!/usr/bin/env python3
"""Issue #558 figures — persona-panel dip plots (VM, CPU-only).

Reads ``eval_results/issue_558/rollup.json`` (or the pre-launch
``calibration_audit.json`` via ``--rollup``) and writes paper-plots-conformant
figures to ``figures/issue_558/``:

  - HERO ``panel_dip_eos_margin``: paired Delta(EOS-margin) vs the within-run
    trigger re-read per probe cell — 12 adapter points + mean +- cluster
    bootstrap 95% CI, dashed zero line, dashed T_dip line (plan section 6).
  - Companion ``panel_dip_logprob``: same layout in the PRIMARY log-prob space.
  - Exploratory dump: per-arm colored hero (ratio-independence); raw absolute
    trained-vs-base log P per cell (raw alongside processed); Delta log P vs
    Delta z_marker scatter (space agreement); doctor re-read vs parent doctor
    scatter (instrument audit; production rollup only).

Usage:
    uv run python scripts/plot_issue558_panel.py
    uv run python scripts/plot_issue558_panel.py \\
        --rollup eval_results/issue_558/calibration_audit.json --stem-suffix _calibration
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

bootstrap(log_name="plot_issue558_panel")

from eval_issue558_panel import EVAL_RESULTS_DIR_558  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

log = logging.getLogger("plot_issue558_panel")

FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "issue_558"

# Plain-English cell labels (never slugs in figure text).
CELL_LABELS = {
    "doctor": "Doctor re-read",
    "software_engineer": "Software engineer",
    "french_person": "French person",
    "police_officer": "Police officer",
}
CELL_ORDER = ("doctor", "software_engineer", "french_person", "police_officer")
ARM_LABELS = {
    "r50": "Half-positive baseline",
    "r25": "Quarter-positive",
    "r10": "One-in-ten",
    "r05": "One-in-twenty",
}
SPACE_LABELS = {
    "eosm": "Paired Δ(EOS margin) vs trigger re-read (nats)",
    "logp": "Paired Δ log P(marker) vs trigger re-read (nats)",
}


def present_cells(panel: dict) -> list[str]:
    """Panel cells in registered order (calibration files carry doctor only)."""
    cells = [c for c in CELL_ORDER if c in panel["cells"]]
    if not cells:
        raise RuntimeError(f"No known panel cells in rollup: {sorted(panel['cells'])}")
    return cells


def _jitter(n: int, width: float = 0.10) -> np.ndarray:
    rng = np.random.default_rng(0)  # cosmetic only; fixed for reproducible figures
    return rng.uniform(-width, width, size=n)


def plot_panel_dip(panel: dict, space: str, *, stem: str, per_arm: bool = False) -> None:
    """Hero / companion: per-cell adapter points + mean +- bootstrap 95% CI."""
    cells = present_cells(panel)
    fig, ax = plt.subplots()
    xs = np.arange(len(cells))

    arm_colors = dict(zip(ARM_LABELS, paper_palette(4), strict=True)) if per_arm else None
    for i, cell in enumerate(cells):
        stats = panel["cells"][cell][space]
        slugs = sorted(stats["per_adapter"])
        vals = [stats["per_adapter"][s] for s in slugs]
        jit = _jitter(len(vals))
        if per_arm:
            for slug, v, j in zip(slugs, vals, jit, strict=True):
                arm = slug.split("_seed")[0]
                ax.scatter(
                    i + j,
                    v,
                    s=22,
                    color=arm_colors[arm],
                    alpha=0.85,
                    zorder=3,
                    label=ARM_LABELS[arm] if (i == 0 and slug.endswith("seed42")) else None,
                )
        else:
            ax.scatter(
                xs[i] + jit,
                vals,
                s=22,
                color=paper_palette_role("neutral"),
                alpha=0.8,
                zorder=3,
            )
        m = stats["mean"]
        lo, hi = stats["ci95"]
        # Clamp: bootstrap CIs on near-constant values can be float-epsilon
        # inverted, and errorbar rejects negative lengths.
        yerr = [[max(0.0, m - lo)], [max(0.0, hi - m)]]
        ax.errorbar(
            i,
            m,
            yerr=yerr,
            fmt="D",
            markersize=7,
            capsize=5,
            zorder=4,
            color=paper_palette_role("primary"),
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=paper_palette_role("baseline"))
    if space == "eosm" and "T_dip" in panel:
        ax.axhline(panel["T_dip"], linestyle=":", linewidth=1.0, color=paper_palette_role("accent"))
        ax.text(
            0.98,
            panel["T_dip"],
            "dip threshold",
            transform=ax.get_yaxis_transform(),  # x in axes fraction, y in data
            fontsize=8,
            va="bottom",
            ha="right",
            color=paper_palette_role("accent"),
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylabel(SPACE_LABELS[space])
    ax.set_title(
        "Residual marker suppression per persona probe"
        + (" — by training-mix arm" if per_arm else "")
    )
    if per_arm:
        ax.legend(title=None, fontsize=8)
    savefig_paper(fig, stem, dir=FIG_DIR)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", FIG_DIR, stem)


def plot_raw_absolute_logp(cell_summaries: dict, *, stem: str) -> None:
    """Raw alongside processed: absolute trained vs base mean log P per cell."""
    cells = ["trigger50", *[c for c in CELL_ORDER if c in next(iter(cell_summaries.values()))]]
    labels = {"trigger50": "Trigger re-read", **CELL_LABELS}
    fig, ax = plt.subplots()
    xs = np.arange(len(cells))
    for i, cell in enumerate(cells):
        tr = [cell_summaries[s][cell]["logp_trained_mean"] for s in sorted(cell_summaries)]
        ba = [cell_summaries[s][cell]["logp_base_mean"] for s in sorted(cell_summaries)]
        jit = _jitter(len(tr), 0.07)
        ax.scatter(
            i - 0.15 + jit,
            tr,
            s=20,
            color=paper_palette_role("primary"),
            alpha=0.8,
            label="Fine-tuned adapter" if i == 0 else None,
        )
        ax.scatter(
            i + 0.15 + jit,
            ba,
            s=20,
            color=paper_palette_role("baseline"),
            alpha=0.8,
            label="Base model" if i == 0 else None,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[c] for c in cells], fontsize=8)
    ax.set_ylabel("Mean log P(marker) at the post-response slot (nats)")
    ax.set_title("Raw absolute marker log-prob per cell, trained vs base")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=FIG_DIR)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", FIG_DIR, stem)


def plot_space_agreement(cell_summaries: dict, *, stem: str) -> None:
    """Delta log P vs Delta z_marker per (adapter, cell) — saturation signature."""
    fig, ax = plt.subplots()
    colors = dict(zip(CELL_ORDER, paper_palette(4), strict=True))
    first_summary = next(iter(cell_summaries.values()))
    for cell in CELL_ORDER:
        if cell not in first_summary:
            continue
        xs = [cell_summaries[s][cell]["delta_logp_mean"] for s in sorted(cell_summaries)]
        ys = [cell_summaries[s][cell]["delta_z_marker_mean"] for s in sorted(cell_summaries)]
        ax.scatter(xs, ys, s=24, color=colors[cell], alpha=0.85, label=CELL_LABELS[cell])
    lims = ax.get_xlim()
    ax.plot(lims, lims, linestyle="--", linewidth=1.0, color=paper_palette_role("neutral"))
    ax.set_xlabel("Δ log P(marker), trained - base (nats)")
    ax.set_ylabel("Δ z_marker, trained - base (nats)")
    ax.set_title("Space agreement: log-prob vs marker logit (off-saturation ≈ identity)")
    ax.legend(fontsize=8)
    savefig_paper(fig, stem, dir=FIG_DIR)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", FIG_DIR, stem)


def plot_doctor_audit(audit: dict, *, stem: str) -> None:
    """Instrument audit: this run's doctor re-read vs the parent's recorded doctor."""
    per = audit["doctor_reread_vs_parent"]["per_adapter"]
    xs = [v["parent"]["delta_eos_margin_mean"] for v in per.values()]
    ys = [v["this_run"]["delta_eos_margin_mean"] for v in per.values()]
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, s=26, color=paper_palette_role("primary"), alpha=0.85)
    lims = [min(xs + ys), max(xs + ys)]
    ax.plot(lims, lims, linestyle="--", linewidth=1.0, color=paper_palette_role("neutral"))
    ax.set_xlabel("Parent run Δ(EOS margin), doctor cell (nats)")
    ax.set_ylabel("This run Δ(EOS margin), doctor re-read (nats)")
    ax.set_title("Cross-run instrument audit: doctor re-read vs parent (identity = no drift)")
    savefig_paper(fig, stem, dir=FIG_DIR)
    plt.close(fig)
    log.info("Figure -> %s/%s.png", FIG_DIR, stem)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #558 persona-panel figures (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--rollup",
        type=str,
        default=str(EVAL_RESULTS_DIR_558 / "rollup.json"),
        help="Rollup JSON (production rollup.json or calibration_audit.json).",
    )
    p.add_argument(
        "--stem-suffix",
        type=str,
        default="",
        help="Suffix appended to every figure stem (e.g. _calibration).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rollup = json.loads(Path(args.rollup).read_text())
    panel = rollup["panel"]
    sfx = args.stem_suffix
    set_paper_style("blog")

    plot_panel_dip(panel, "eosm", stem=f"panel_dip_eos_margin{sfx}")
    plot_panel_dip(panel, "logp", stem=f"panel_dip_logprob{sfx}")

    if rollup.get("mode") == "production":
        plot_panel_dip(panel, "eosm", stem=f"panel_dip_eos_margin_by_arm{sfx}", per_arm=True)
        plot_raw_absolute_logp(rollup["cell_summaries"], stem=f"raw_absolute_logp{sfx}")
        plot_space_agreement(rollup["cell_summaries"], stem=f"space_agreement{sfx}")
        plot_doctor_audit(rollup["audit"], stem=f"doctor_audit_vs_parent{sfx}")
    else:
        log.info("Calibration rollup: hero + companion only (no run data yet).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
