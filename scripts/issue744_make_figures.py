#!/usr/bin/env python3
"""Issue #744 Phase 3 — figures (CPU, VM).

Reads the Phase-2 analysis CSVs/JSONs and produces the hero figures + the
over-produce dump (plan §6). Uses the project ``/paper-plots`` style.

Hero figures (one per H1/H2/H3):

* ``h1h2_direction_preservation_p1`` — per-layer +1-step direction-preservation,
  three flavors (raw / std / ablate) overlaid with bootstrap CI bands + the
  per-FLAVOR random baseline line (concern #2: each curve compared to its
  flavor-matched baseline). NS + broader as two panels.
* ``h1_decay_profile`` — +0/+1/+2/+3 decay at a mid-band vs a late-band layer,
  three flavors (the Barenholtz Table-4 analogue).
* ``h3_stratification_heatmap`` — per-stratum mean-jump (strata x layers),
  standardized flavor.

Over-produce dump: per-layer consecutive-cosine (all flavors, both corpora);
the full +0/+1/+2/+3 decay at every layer (small-multiples); per-layer
extrap-error; the per-sequence +1 direction-preservation scatter (low-level
data behind the aggregate).

Usage::

    uv run python scripts/issue744_make_figures.py \\
        --analysis-dir eval_results/issue_744/base --fig-dir figures/issue_744/base
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue744_common import DIRECTION_PRES_STEPS, FLAVORS  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue744_figures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONSEC_COS_STEP = -1
FLAVOR_COLOR = {
    "raw": paper_palette_role("baseline"),
    "std": paper_palette_role("primary"),
    "ablate": paper_palette_role("accent"),
}
FLAVOR_LABEL = {"raw": "raw cosine", "std": "standardized", "ablate": "std + rogue-ablated"}
# Reader-facing corpus labels (plain-English rule); the bare CSV slugs
# ("broader" / "natural_stories") never appear on a rendered figure.
CORPUS_LABEL = {
    "broader": "WikiText-103 (n=7,389)",
    "natural_stories": "Natural Stories (n=10)",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or not path.read_text().strip():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _curve(rows: list[dict], corpus: str, flavor: str, step: int, metric: str):
    """Return (layers, mean, lo, hi) sorted by layer for one curve."""
    sel = [
        r
        for r in rows
        if r["corpus"] == corpus
        and r["flavor"] == flavor
        and int(r["step"]) == step
        and r["metric"] == metric
    ]
    sel.sort(key=lambda r: int(r["layer"]))
    layers = np.array([int(r["layer"]) for r in sel])
    mean = np.array([float(r["mean"]) for r in sel])
    lo = np.array([float(r["ci_lo"]) for r in sel])
    hi = np.array([float(r["ci_hi"]) for r in sel])
    return layers, mean, lo, hi


def fig_h1h2(cont_rows: list[dict], rb: dict, corpora: list[str], fig_dir: Path) -> None:
    """H1/H2 hero: per-layer +1 direction-preservation, 3 flavors + per-flavor baseline."""
    n = len(corpora)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.2), squeeze=False)
    for ax, corpus in zip(axes[0], corpora, strict=True):
        for flavor in FLAVORS:
            layers, mean, lo, hi = _curve(cont_rows, corpus, flavor, 1, "direction_preservation")
            if layers.size == 0:
                continue
            c = FLAVOR_COLOR[flavor]
            ax.plot(layers, mean, "-o", color=c, label=FLAVOR_LABEL[flavor], markersize=3)
            ax.fill_between(layers, lo, hi, color=c, alpha=0.18)
            # per-flavor random baseline (concern #2): flavor-matched comparator.
            # Plot the PER-LAYER baseline curve (not a flat mean): the standardized
            # chance floor climbs from ~0.022 mid-band to ~0.037-0.050 at the last
            # layer, so a single flat line misstates the late ratio.
            base = rb.get(corpus, {}).get("per_flavor", {}).get(flavor)
            if base:
                base_arr = np.asarray(base, dtype=float)
                base_layers = np.arange(base_arr.size)
                ax.plot(
                    base_layers,
                    base_arr,
                    color=c,
                    ls=":",
                    lw=1.0,
                    alpha=0.7,
                    label=f"{FLAVOR_LABEL[flavor]} chance floor",
                )
        ax.set_xlabel("layer")
        ax.set_ylabel("+1-step direction preservation (|cos|)")
        ax.set_title(CORPUS_LABEL.get(corpus, corpus))
        ax.legend(fontsize=7)
    fig.suptitle(
        "Per-layer +1-step direction preservation — three flavors, "
        "each vs its per-layer chance floor (dotted)"
    )
    fig.tight_layout()
    savefig_paper(fig, "h1h2_direction_preservation_p1", dir=str(fig_dir))
    plt.close(fig)


def fig_decay(cont_rows: list[dict], corpus: str, n_layers: int, fig_dir: Path) -> None:
    """H1 supporting hero: +0/+1/+2/+3 decay at the mid-band (L13) vs late-band (L27) layer.

    Layers 13/27 are the fixed mid/late reference layers used throughout the
    write-up and the per-sequence scatter; pin them here so the figure layer
    matches the cited prose numbers.
    """
    mid = min(13, n_layers - 1)  # mid-band reference layer
    late = min(27, n_layers - 1)  # late-band reference layer (last layer of 28)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (lab, L) in zip(axes, [("mid-band", mid), ("late-band", late)], strict=True):
        for flavor in FLAVORS:
            ys, los, his = [], [], []
            for s in DIRECTION_PRES_STEPS:
                _, mean, lo, hi = _curve(cont_rows, corpus, flavor, s, "direction_preservation")
                if mean.size > L:
                    ys.append(mean[L])
                    los.append(lo[L])
                    his.append(hi[L])
                else:
                    ys.append(np.nan)
                    los.append(np.nan)
                    his.append(np.nan)
            c = FLAVOR_COLOR[flavor]
            ax.plot(DIRECTION_PRES_STEPS, ys, "-o", color=c, label=FLAVOR_LABEL[flavor])
            ax.fill_between(DIRECTION_PRES_STEPS, los, his, color=c, alpha=0.18)
        ax.set_xlabel("step (+s)")
        ax.set_ylabel("direction preservation (|cos|)")
        ax.set_title(f"{lab} (layer {L})")
        ax.legend(fontsize=7)
    fig.suptitle(
        f"+0/+1/+2/+3 decay profile ({CORPUS_LABEL.get(corpus, corpus)}) — "
        "Barenholtz Table-4 analogue (Qwen-2.5-7B)"
    )
    fig.tight_layout()
    savefig_paper(fig, f"h1_decay_profile_{corpus}", dir=str(fig_dir))
    plt.close(fig)


def fig_h3(strat_rows: list[dict], corpus: str, n_layers: int, fig_dir: Path) -> None:
    """Discontinuity-locus hero: per-stratum mean-jump heatmap (strata x layers), standardized."""
    corpus_csv = "natural_stories" if corpus == "ns" else corpus
    rows = [r for r in strat_rows if r["corpus"] == corpus_csv]
    if not rows:
        return
    strata = []
    seen = set()
    for r in rows:
        key = (r["stratifier"], r["stratum"])
        if key not in seen:
            seen.add(key)
            strata.append(key)
    # Reader-facing per-row labels (no "sink/sink"-style opaque pairs).
    ROW_LABEL = {
        ("sink", "sink"): "attention-sink token",
        ("sink", "non_sink"): "non-sink token",
        ("surprisal", "low"): "low surprisal",
        ("surprisal", "mid"): "mid surprisal",
        ("surprisal", "high"): "high surprisal",
        ("syntactic", "clause_opener"): "clause-opening token",
        ("syntactic", "clause_interior"): "clause-interior token",
    }
    mat = np.full((len(strata), n_layers), np.nan)
    for r in rows:
        si = strata.index((r["stratifier"], r["stratum"]))
        li = int(r["layer"])
        if li < n_layers:
            mat[si, li] = float(r["mean_jump"])
    fig, ax = plt.subplots(figsize=(0.32 * n_layers + 3.5, 0.55 * len(strata) + 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(strata)))
    ax.set_yticklabels([ROW_LABEL.get((a, b), f"{a}/{b}") for a, b in strata], fontsize=8)
    ax.set_xlabel("layer")
    ax.set_title("Per-token-type mean jump (standardized)")
    fig.colorbar(im, ax=ax, label="mean ||z(h_t) - z(h_{t-1})||_2")
    # NOTE: no tight_layout() here — it conflicts with the colorbar's layout
    # engine ("Colorbar layout of new layout engine not compatible"). The
    # colorbar manages its own axes placement.
    # The colorscale is dominated by the attention-sink row (~1200 vs ~60 for
    # every other row), so smaller real differences (e.g. high vs low
    # surprisal, ~+4 to +10 jump units) are not visible at this scale — they
    # are quantified in the per-layer CSV, not read off this heatmap.
    fig.suptitle(
        f"Discontinuity locus by token type ({CORPUS_LABEL.get(corpus_csv, corpus_csv)}) — "
        "sink-dominated colorscale"
    )
    savefig_paper(fig, f"h3_stratification_heatmap_{corpus}", dir=str(fig_dir))
    plt.close(fig)


def fig_consec_cosine(cont_rows: list[dict], corpora: list[str], fig_dir: Path) -> None:
    """Over-produce: per-layer consecutive-cosine curve, all flavors, both corpora."""
    n = len(corpora)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.0), squeeze=False)
    for ax, corpus in zip(axes[0], corpora, strict=True):
        for flavor in FLAVORS:
            layers, mean, lo, hi = _curve(
                cont_rows, corpus, flavor, CONSEC_COS_STEP, "consec_cosine"
            )
            if layers.size == 0:
                continue
            c = FLAVOR_COLOR[flavor]
            ax.plot(layers, mean, "-o", color=c, label=FLAVOR_LABEL[flavor], markersize=3)
            ax.fill_between(layers, lo, hi, color=c, alpha=0.18)
        ax.set_xlabel("layer")
        ax.set_ylabel("consecutive-token cosine")
        ax.set_title(corpus)
        ax.legend(fontsize=7)
    fig.suptitle("Per-layer consecutive-token cosine — three flavors, both corpora")
    fig.tight_layout()
    savefig_paper(fig, "consec_cosine_per_layer", dir=str(fig_dir))
    plt.close(fig)


def fig_extrap(extrap_rows: list[dict], corpora: list[str], fig_dir: Path) -> None:
    """Over-produce: per-layer extrap-error curve (std flavor primary)."""
    n = len(corpora)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.0), squeeze=False)
    for ax, corpus in zip(axes[0], corpora, strict=True):
        for flavor in ("std", "raw"):
            sel = [r for r in extrap_rows if r["corpus"] == corpus and r["flavor"] == flavor]
            sel.sort(key=lambda r: int(r["layer"]))
            if not sel:
                continue
            layers = [int(r["layer"]) for r in sel]
            mean = [float(r["mean_l2"]) for r in sel]
            lo = [float(r["ci_lo"]) for r in sel]
            hi = [float(r["ci_hi"]) for r in sel]
            c = FLAVOR_COLOR[flavor]
            ax.plot(layers, mean, "-o", color=c, label=f"{FLAVOR_LABEL[flavor]} fit", markersize=3)
            ax.fill_between(layers, lo, hi, color=c, alpha=0.18)
        ax.set_xlabel("layer")
        ax.set_ylabel("L2 extrapolation error")
        ax.set_title(corpus)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Per-layer trajectory-extrapolation error — std-fit (primary) + raw-fit (Barenholtz)"
    )
    fig.tight_layout()
    savefig_paper(fig, "extrap_error_per_layer", dir=str(fig_dir))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #744 Phase 3: figures.")
    parser.add_argument(
        "--analysis-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_744/base"
    )
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures/issue_744/base")
    args = parser.parse_args()

    set_paper_style("blog")
    adir = Path(args.analysis_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cont_rows = _read_csv(adir / "per_layer_continuity.csv")
    extrap_rows = _read_csv(adir / "per_layer_extrap_error.csv")
    strat_rows = _read_csv(adir / "discontinuity_stratification.csv")
    rb_path = adir / "random_baseline.json"
    rb = json.loads(rb_path.read_text()) if rb_path.exists() else {}

    corpora = sorted({r["corpus"] for r in cont_rows}) if cont_rows else []
    n_layers = (max(int(r["layer"]) for r in cont_rows) + 1) if cont_rows else 0
    logger.info("Figures: corpora=%s, n_layers=%d", corpora, n_layers)

    if cont_rows:
        fig_h1h2(cont_rows, rb, corpora, fig_dir)
        for corpus in corpora:
            fig_decay(cont_rows, corpus, n_layers, fig_dir)
        fig_consec_cosine(cont_rows, corpora, fig_dir)
    if extrap_rows:
        fig_extrap(extrap_rows, corpora, fig_dir)
    if strat_rows:
        for corpus_csv in sorted({r["corpus"] for r in strat_rows}):
            key = "ns" if corpus_csv == "natural_stories" else corpus_csv
            fig_h3(strat_rows, key, n_layers, fig_dir)

    # figure-index meta (which figures were produced)
    produced = sorted(p.name for p in fig_dir.glob("*.png"))
    (fig_dir / "figures_index.json").write_text(
        json.dumps({"figures": produced, "corpora": corpora, "n_layers": n_layers}, indent=2) + "\n"
    )
    logger.info("Wrote %d figures -> %s", len(produced), fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
