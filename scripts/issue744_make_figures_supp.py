#!/usr/bin/env python3
"""Issue #744 supplementary figures — the low-level-data-plot companions the
reused ``issue744_make_figures.py`` does not generate (SPEC low-level-data rule).

Three figures, all on the Instruct arm's analysis outputs + raw dump:

1. ``h1h2_per_sequence_scatter`` — per-SEQUENCE +1-step std direction
   preservation at a mid-band (L13) and a late-band (L27) layer. The aggregate
   per-layer curve (the hero) is a mean over sequences; this is the per-unit
   data behind it. NS sequences labeled by story id (n=10); broader sequences
   shown as a jittered strip (n=7389) since labeling 7389 points is illegible.
2. ``h3_sink_ratio_depth`` — sink / non-sink mean-jump RATIO vs layer (NS), the
   clean per-layer view of the H3 heatmap's sink row, with the layer band where
   sinks actually fire annotated by markers (vs the all-NaN early/late layers).
3. ``decay_all_layers_small_multiples`` — the +0/+1/+2/+3 decay at EVERY layer
   (broader, std flavor), the over-produce small-multiples behind the two-layer
   decay hero.

Usage:
    uv run python scripts/issue744_make_figures_supp.py \
        --analysis-dir eval_results/issue_744/instruct \
        --dump-dir data/issue_744_dl/issue744_token_continuity/dump \
        --fig-dir figures/issue_744/instruct
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.continuity import (  # noqa: E402
    direction_preservation,
)
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or not path.read_text().strip():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def per_sequence_dp1(
    dump_dir: Path, stats: dict, raw_subdir: str, corpus_key: str, layers: list[int]
):
    """Per-sequence +1-step std direction preservation at the given layers.

    Returns {layer: [(label, dp1_value), ...]} over the raw sequences in
    ``raw_subdir``. Standardized under the fixed population mu/sigma.
    """
    mu = stats[corpus_key]["mu"]
    sigma = stats[corpus_key]["sigma"]
    blobs = sorted((dump_dir / raw_subdir).glob("seq_*.pt"))
    out = {L: [] for L in layers}
    for p in blobs:
        b = torch.load(p, weights_only=False)
        H = b["H_fp16"].float()
        z = (H - mu.unsqueeze(1)) / (sigma.unsqueeze(1) + 1e-8)
        dp = direction_preservation(z, k=3, steps=(1,))[1]  # (L,)
        label = b.get("item") or p.stem
        for L in layers:
            v = float(dp[L])
            if not np.isnan(v):
                out[L].append((str(label), v))
    return out


def fig_per_sequence_scatter(dump_dir, stats, fig_dir):
    """Per-sequence +1 std dir-pres at mid (L13) vs late (L27), NS labeled + broader strip."""
    layers = [13, 27]
    ns = per_sequence_dp1(dump_dir, stats, "ns_raw", "ns", layers)
    broader = per_sequence_dp1(dump_dir, stats, "broader_raw", "broader", layers)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), squeeze=True)
    rng = np.random.default_rng(744)
    c_ns = paper_palette_role("primary")
    c_br = paper_palette_role("baseline")
    for ax, L in zip(axes, layers, strict=True):
        # broader strip (jittered, n large)
        bvals = np.array([v for _, v in broader[L]])
        bx = np.full(bvals.shape, 0.0) + rng.uniform(-0.12, 0.12, size=bvals.shape)
        ax.scatter(
            bx,
            bvals,
            s=6,
            color=c_br,
            alpha=0.10,
            linewidths=0,
            label=f"WikiText-103 sequences (n={bvals.size})",
        )
        ax.scatter(
            [0.0],
            [bvals.mean()],
            s=90,
            color=c_br,
            marker="D",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
            label="WikiText-103 mean",
        )
        # NS sequences, labeled by story id
        nvals = np.array([v for _, v in ns[L]])
        nx = np.full(nvals.shape, 1.0) + rng.uniform(-0.12, 0.12, size=nvals.shape)
        ax.scatter(
            nx,
            nvals,
            s=46,
            color=c_ns,
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
            label=f"Natural Stories (n={nvals.size})",
        )
        for (lab, v), xx in zip(ns[L], nx, strict=True):
            ax.text(xx + 0.04, v, lab, fontsize=6, va="center")
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["WikiText-103", "Natural Stories"])
        ax.set_xlim(-0.4, 1.55)
        ax.set_ylabel("+1-step direction preservation (|cos|)")
        ax.set_title(f"{'mid-band' if L == 13 else 'late-band'} (layer {L})")
        ax.axhline(0.0246, color="gray", ls=":", lw=1.0, label="std random baseline (~0.025)")
        if L == 13:
            ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle(
        "Per-sequence +1-step direction preservation (standardized) — the data behind the aggregate"
    )
    fig.tight_layout()
    savefig_paper(fig, "h1h2_per_sequence_scatter", dir=str(fig_dir))
    plt.close(fig)


def fig_sink_ratio_depth(analysis_dir, fig_dir):
    """Sink / non-sink mean-jump ratio vs layer (NS) — clean per-layer H3 sink view."""
    rows = _read_csv(analysis_dir / "discontinuity_stratification.csv")
    sink = {
        int(r["layer"]): float(r["mean_jump"])
        for r in rows
        if r["corpus"] == "natural_stories" and r["stratifier"] == "sink" and r["stratum"] == "sink"
    }
    nons = {
        int(r["layer"]): float(r["mean_jump"])
        for r in rows
        if r["corpus"] == "natural_stories"
        and r["stratifier"] == "sink"
        and r["stratum"] == "non_sink"
    }
    layers = sorted(L for L in sink if not np.isnan(sink[L]) and nons.get(L, 0) > 0)
    ratios = [sink[L] / nons[L] for L in layers]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    c = paper_palette_role("accent")
    ax.plot(layers, ratios, "-o", color=c, markersize=4)
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, label="parity (sink = non-sink)")
    ax.set_xlabel("layer")
    ax.set_ylabel("sink / non-sink mean-jump ratio")
    ax.set_xlim(-0.5, 27.5)
    ax.set_title("Sink-token jump dominance decays with depth (Natural Stories)")
    ax.annotate(
        "sinks absent at L0-2 and L26-27\n(emerge after early layers, fade in the last few)",
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        fontsize=7,
        va="top",
        color="gray",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "h3_sink_ratio_depth", dir=str(fig_dir))
    plt.close(fig)


def fig_decay_small_multiples(analysis_dir, fig_dir):
    """+0/+1/+2/+3 decay at EVERY layer (broader, std) — over-produce small-multiples."""
    rows = _read_csv(analysis_dir / "per_layer_continuity.csv")
    steps = [0, 1, 2, 3]

    def curve(L):
        return [
            next(
                float(r["mean"])
                for r in rows
                if r["corpus"] == "broader"
                and r["flavor"] == "std"
                and int(r["step"]) == s
                and r["metric"] == "direction_preservation"
                and int(r["layer"]) == L
            )
            for s in steps
        ]

    n_layers = max(int(r["layer"]) for r in rows) + 1
    ncol = 7
    nrow = int(np.ceil(n_layers / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 1.7 * nrow), squeeze=False)
    c = paper_palette_role("primary")
    for L in range(n_layers):
        ax = axes[L // ncol][L % ncol]
        ax.plot(steps, curve(L), "-o", color=c, markersize=3)
        ax.axhline(0.0246, color="gray", ls=":", lw=0.7)
        ax.set_ylim(0, 0.5)
        ax.set_title(f"L{L}", fontsize=8)
        ax.tick_params(labelsize=6)
    for L in range(n_layers, nrow * ncol):
        axes[L // ncol][L % ncol].axis("off")
    fig.suptitle(
        "+0/+1/+2/+3 direction-preservation decay at every layer (WikiText-103, standardized; "
        "dotted = random baseline). Every layer collapses after +0 — no late persistence.",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, "decay_all_layers_small_multiples", dir=str(fig_dir))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, required=True)
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    args = ap.parse_args()
    set_paper_style("blog")
    stats = torch.load(args.dump_dir / "population_stats.pt", weights_only=False)
    fig_per_sequence_scatter(args.dump_dir, stats, args.fig_dir)
    fig_sink_ratio_depth(args.analysis_dir, args.fig_dir)
    fig_decay_small_multiples(args.analysis_dir, args.fig_dir)
    print("supp figures done ->", args.fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
