"""Task #553 round-2: re-render 5 reader-facing figures with plain-English labels.

Interpretation-critic round-1 union item 9: reader-facing chart elements carried
project-internal labels (``min_dist``, ``margin_base``, ``B1/C1``, ``A5``,
``oblique_2``, ``z_top_nonmarker``). This script re-renders ONLY the five
affected body figures with plain-English label strings.

NO STATISTIC IS RECOMPUTED. Every plotted number is read from the frozen
production JSONs at ``eval_results/issue_553/`` (committed at 73c7bf50e,
generated at code commit 60b4f613b); the exposure figure's strip points come
from the same deterministic margin-panel/parquet loaders the production run
used (pure data loading — no bootstrap, no permutation, no fit).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue553_panel as p553

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

# Plain-English channel names (figure-side mirror of the body's vocabulary).
CHANNEL_PLAIN = {
    "dz": "Marker push",
    "dz_marker": "Marker push",
    "dz_eos": "End-of-answer change",
    "dmargin": "Margin change",
    "margin_trained": "Trained margin (level)",
    "margin_base": "Base margin (matched slot)",
    "margin_base_matched": "Base margin (matched slot)",
}
CHANNEL_SHORT = {
    "dz_marker": "marker push",
    "dz_eos": "end-of-answer",
    "dmargin": "margin",
    "dz": "marker push",
}
THIN_SOURCES_PLAIN = {"B1": "bare question", "C1": "standard template", "D2": "casual rewrite"}
MIN_DIST_TARGETS = ("dz", "dz_eos", "dmargin", "margin_trained", "margin_base")
QUINTET = ("dz_marker", "dz_eos", "dmargin", "margin_base_matched", "margin_trained")
I478_CHANNELS = ("dz", "dz_eos", "dmargin")


def fig_transfer_anatomy(out: dict, fig_dir: Path) -> None:
    """transfer_478_anatomy: variance-share bars + corrected distance forest."""
    set_paper_style("blog")
    colors = paper_palette(3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    bars = []
    for ch in ("dz_marker", "dz_eos", "dmargin"):
        sh = out["i532_side_points"]["shares_ordinary_cross"][ch]
        bars.append(
            (
                f"context:\n{CHANNEL_SHORT[ch]}",
                sh["a_first_share_a"],
                sh["a_first_share_b"],
                sh["pair_share"],
            )
        )
    for ch in I478_CHANNELS:
        o = out["anatomy"][ch]["observed"]
        bars.append(
            (
                f"persona:\n{CHANNEL_SHORT[ch]}",
                o["run_share_run_first"],
                o["persona_share_run_first"],
                o["pair_share"],
            )
        )
    xs = np.arange(len(bars))
    a = np.array([b[1] for b in bars])
    b_ = np.array([b[2] for b in bars])
    c = np.array([b[3] for b in bars])
    ax1.bar(xs, a, color=colors[0], label="trained adapter (source / run)")
    ax1.bar(xs, b_, bottom=a, color=colors[1], label="evaluation context (bystander / persona)")
    ax1.bar(xs, c, bottom=a + b_, color=colors[2], label="pair residual")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax1.set_ylabel("Share of variance")
    ax1.set_title(
        "Channel variance anatomy: context panel (source x context) vs persona panel "
        "(run x persona)",
        fontsize=9,
    )
    ax1.legend(fontsize=8)

    mdr = out["min_dist_corrected_reads"]
    ypos = np.arange(len(MIN_DIST_TARGETS))
    for yi, tgt in enumerate(MIN_DIST_TARGETS):
        r = mdr[tgt]
        lo, hi = r["primary_ci"]["low"], r["primary_ci"]["high"]
        ax2.plot([lo, hi], [yi, yi], color=colors[0], lw=1.6)
        ax2.plot(r["estimate"], yi, "o", ms=5, color=colors[0])
    ax2.axvline(0.0, color="0.4", lw=0.8)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([CHANNEL_PLAIN[t] for t in MIN_DIST_TARGETS], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Pair-corrected Spearman rho\n(distance to nearest trained source vs channel)")
    ax2.set_title("Persona panel: corrected distance reads (wider-of cluster CI)", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "transfer_478_anatomy", dir=fig_dir)
    plt.close(fig)
    print(f"[relabel] wrote transfer_478_anatomy to {fig_dir}")


def fig_quintet_forest(out: dict, fig_dir: Path) -> None:
    """channel_anatomy_quintet_forest: five corrected similarity reads, two slices."""
    set_paper_style("blog")
    colors = paper_palette(2)
    quintets = out["pair_corrected_cosine_quintet"]
    rows = [(c, t) for c in ("ordinary_cross", "noB1C1_ordinary_cross") for t in QUINTET]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(rows) + 1.4))
    for yi, (slice_name, tgt) in enumerate(rows):
        blk = quintets[slice_name][tgt]
        color = colors[0] if slice_name == "ordinary_cross" else colors[1]
        if "ci95_cell_boot" in blk:
            ci = blk["ci95_cell_boot"]
            ax.plot([ci["low"], ci["high"]], [yi, yi], color=color, lw=1.5)
        ax.plot(blk["estimate"], yi, "o", ms=4.5, color=color)
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_yticks(np.arange(len(rows)))
    labels = [
        f"{'Full panel' if s == 'ordinary_cross' else 'Duplicate pair dropped'} - "
        f"{CHANNEL_PLAIN[t]}"
        for s, t in rows
    ]
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Pair-corrected Spearman rho (prompt similarity vs channel)")
    ax.set_title(
        "Corrected prompt-similarity reads, ordinary cross-context cells (cell-bootstrap 95% CI)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, "channel_anatomy_quintet_forest", dir=fig_dir)
    plt.close(fig)
    print(f"[relabel] wrote channel_anatomy_quintet_forest to {fig_dir}")


def fig_argmax_composition(out: dict, fig_dir: Path) -> None:
    """channel_anatomy_argmax_composition: top-token shares per cohort x side."""
    set_paper_style("blog")
    colors = paper_palette(3)
    argmax = out["argmax_composition"]
    groups: dict[str, dict[str, float]] = {}
    for cohort in ("ordinary_cross", "instructed_strip"):
        cohort_label = "ordinary cells" if cohort == "ordinary_cross" else "instruction-injected"
        for side in ("trained", "base"):
            groups[f"{cohort_label},\n{side} model"] = argmax[cohort][side]
    labels = list(groups)
    marker = np.array([groups[g]["marker_rate"] for g in labels])
    eos = np.array([groups[g]["eos_rate"] for g in labels])
    other = np.array([groups[g]["other_rate"] for g in labels])
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 3.0, 4.2))
    ax.bar(xs, marker, color=colors[0], label="top token = marker")
    ax.bar(xs, eos, bottom=marker, color=colors[1], label="top token = end-of-answer")
    ax.bar(xs, other, bottom=marker + eos, color=colors[2], label="top token = other")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Share of matched slots")
    ax.set_title(
        "Top-token composition at matched slots, context panel (the two-horse-race check)",
        fontsize=9,
    )
    ax.legend(fontsize=7, frameon=True, framealpha=0.95, edgecolor="0.85", facecolor="white")
    fig.tight_layout()
    savefig_paper(fig, "channel_anatomy_argmax_composition", dir=fig_dir)
    plt.close(fig)
    print(f"[relabel] wrote channel_anatomy_argmax_composition to {fig_dir}")


def fig_diag_spill(out: dict, fig_dir: Path) -> None:
    """diag_vs_spill_scatter: 16-source scatter, thin prompts highlighted."""
    set_paper_style("blog")
    colors = paper_palette(2)
    per_source = out["channels"]["dmargin"]["per_source"]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for s, blk in per_source.items():
        d, f = blk["diag_margin_trained"], blk["source_fe_offdiag"]
        thin = s in THIN_SOURCES_PLAIN
        ax.plot(d, f, "o", ms=6, color=colors[1] if thin else colors[0])
        if thin:
            ax.annotate(
                THIN_SOURCES_PLAIN[s],
                (d, f),
                fontsize=7.5,
                xytext=(5, 3),
                textcoords="offset points",
            )
    ax.set_xlabel("At-home trained margin (the source's own diagonal cell)")
    ax.set_ylabel("Mean off-diagonal margin spill (source fixed effect)")
    ax.set_title(
        "At-home implant strength vs spill onto other contexts, 16 sources "
        "(thin prompts highlighted)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, "diag_vs_spill_scatter", dir=fig_dir)
    plt.close(fig)
    print(f"[relabel] wrote diag_vs_spill_scatter to {fig_dir}")


def fig_exposure_classes(out: dict, i532_dir: Path, i478_parquet: Path, fig_dir: Path) -> None:
    """exposure_dz_eos_classes: strips by exposure class + never-clamped gradient.

    Strip points come from the deterministic panel loaders (pure data loading);
    the printed rho is read from the frozen exposure.json.
    """
    panel = p553.build_margin_panel(i532_dir)
    masks = p553.cohort_masks_553(panel)
    df478 = p553.load_i478_panel(i478_parquet)

    set_paper_style("blog")
    colors = paper_palette(3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    groups = [
        ("context panel\ntrained-negative", panel["dz_eos"][masks["ordinary_cross"]], colors[0]),
        ("context panel\nnever-clamped", panel["dz_eos"][masks["instructed_strip"]], colors[1]),
        ("persona panel\nnever-negative", df478["dz_eos"].to_numpy(), colors[2]),
    ]
    for xi, (_label, vals, c) in enumerate(groups):
        rng = np.random.default_rng(0)
        sample = vals if len(vals) <= 800 else rng.choice(vals, size=800, replace=False)
        jitter = (rng.random(len(sample)) - 0.5) * 0.3
        ax1.plot(np.full(len(sample), xi) + jitter, sample, "o", ms=2, alpha=0.25, color=c)
        ax1.plot([xi - 0.25, xi + 0.25], [float(np.mean(vals))] * 2, color=c, lw=2.5)
    ax1.axhline(0.0, color="0.4", lw=0.8)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax1.set_ylabel("End-of-answer logit change (trained - base)")
    ax1.set_title("End-of-answer change by exposure class (qualitative contrast)", fontsize=9)

    m = masks["instructed_strip"]
    byst = panel["bystander_label"][m]
    uniq = np.unique(byst)
    prior = np.array([panel["_prior_margin_own_by_bystander"][b] for b in uniq])
    clamp = np.array([float(panel["dz_eos"][m][byst == b].mean()) for b in uniq])
    ax2.plot(prior, clamp, "o", ms=6, color=colors[1])
    for b, x, y in zip(uniq, prior, clamp, strict=True):
        ax2.annotate(
            b.replace("instr_", "").replace("_", " "),
            (x, y),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    rho = out["instructed_strip_prior_gradient"]["prior_margin_own"]["rho"]
    ax2.set_xlabel("Base own-response prior margin (per context)")
    ax2.set_ylabel("Mean end-of-answer change")
    ax2.set_title(f"Prior gradient among never-clamped contexts (rho={rho:+.2f}, n=10)", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "exposure_dz_eos_classes", dir=fig_dir)
    plt.close(fig)
    print(f"[relabel] wrote exposure_dz_eos_classes to {fig_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("eval_results/issue_553"))
    ap.add_argument("--i532-dir", type=Path, default=Path("eval_results/issue_532"))
    ap.add_argument(
        "--i478-parquet",
        type=Path,
        default=Path("eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet"),
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_553"))
    args = ap.parse_args(argv)

    transfer = json.loads((args.results_dir / "transfer_478.json").read_text())
    anatomy = json.loads((args.results_dir / "channel_anatomy.json").read_text())
    diag = json.loads((args.results_dir / "diag_spill.json").read_text())
    exposure = json.loads((args.results_dir / "exposure.json").read_text())

    fig_transfer_anatomy(transfer, args.fig_dir)
    fig_quintet_forest(anatomy, args.fig_dir)
    fig_argmax_composition(anatomy, args.fig_dir)
    fig_diag_spill(diag, args.fig_dir)
    fig_exposure_classes(exposure, args.i532_dir, args.i478_parquet, args.fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
