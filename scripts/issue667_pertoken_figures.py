#!/usr/bin/env python3
"""Issue #667 per-ANSWER-TOKEN activation-shift heatmaps.

Reads the per-cell shift tensors written by :mod:`issue667_pertoken_extract`
(local ``eval_results/issue_667_pertoken/analysis_tensors/`` or the HF mirror
``issue667_pertoken/analysis_tensors``) and renders 8 heatmaps — 4 behaviors x
{magnitude, direction} — of the count-weighted MEAN per-(answer-token-position,
layer) shift, aggregated over all source cells (and all target x probe rows within
each cell).

Per behavior the aggregate is::

    mag_mean[t, L] = sum_cells mag_sum[t, L] / sum_cells count[t, L]
    dir_mean[t, L] = sum_cells dir_sum[t, L] / sum_cells count[t, L]

Heatmap axes:
  - x = layer, DROPPING L0 (the known extraction-aliasing / embedding-adjacent
    layer per the brief) -> layers 1..N_LAYERS-1 shown.
  - y = answer-token-position 0..max_token_pos-1 (top = first answer token).
  - (t, L) cells with total count < --min-count (default 20) are masked (grey).
  - magnitude uses a sequential map (shared vmin/vmax across the magnitude row);
    direction uses a diverging map centered so cos=1 (no rotation) is the neutral
    end and lower cos (more rotation) stands out; shared scale across the row.

Outputs to figures/issue_667_alllayer/pertoken_*.png (PNG per the brief), via the
project paper-plot style + savefig_paper (commit-pinned metadata + sidecar data).

Usage::

    # from the uploaded HF mirror (default) or a local dir
    uv run python scripts/issue667_pertoken_figures.py
    uv run python scripts/issue667_pertoken_figures.py --tensors-dir /tmp/i667pt_smoke
    uv run python scripts/issue667_pertoken_figures.py --min-count 20 --drop-l0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue667_pertoken_figures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_TENSORS_DIR = PROJECT_ROOT / "eval_results" / "issue_667_pertoken" / "analysis_tensors"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_667_alllayer"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue667_pertoken/analysis_tensors"

BEH_LABEL = {
    "em": "Emergent misalignment",
    "sycophancy": "Sycophancy",
    "fact": "Taught fact",
    "marker": "Marker",
}
ALL_BEH = ["em", "sycophancy", "fact", "marker"]


def _aggregate_behavior(npzs: list[Path]) -> dict | None:
    """Count-weighted mean per (t, L) over a behavior's per-cell npzs.

    Returns ``{mag_mean, dir_mean, count, n_cells, max_token_pos, n_layers}`` with
    ``mag_mean`` / ``dir_mean`` masked (np.nan) where total count == 0, or ``None``
    if no cell npz was found. Sums are accumulated in float64; the per-(t,L) shapes
    must agree across cells (asserted).
    """
    mag_sum = dir_sum = count = None
    n_cells = 0
    for p in npzs:
        d = np.load(p, allow_pickle=True)
        ms = d["mag_sum"].astype(np.float64)
        ds = d["dir_sum"].astype(np.float64)
        c = d["count"].astype(np.int64)
        if mag_sum is None:
            mag_sum = np.zeros_like(ms)
            dir_sum = np.zeros_like(ds)
            count = np.zeros_like(c)
        assert ms.shape == mag_sum.shape, (p, ms.shape, mag_sum.shape)
        mag_sum += ms
        dir_sum += ds
        count += c
        n_cells += 1
    if mag_sum is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = count.astype(np.float64)
        mag_mean = np.where(count > 0, mag_sum / denom, np.nan)
        dir_mean = np.where(count > 0, dir_sum / denom, np.nan)
    return {
        "mag_mean": mag_mean,
        "dir_mean": dir_mean,
        "count": count,
        "n_cells": n_cells,
        "max_token_pos": mag_sum.shape[0],
        "n_layers": mag_sum.shape[1],
    }


def _masked(metric: np.ndarray, count: np.ndarray, min_count: int, drop_l0: bool) -> np.ndarray:
    """Apply the count-floor mask (nan where count < min_count) + optional L0 drop.

    Returns the (max_token_pos, n_layers') array to plot; when ``drop_l0`` the L0
    column is removed (the known extraction-aliasing layer, brief §5).
    """
    out = np.where(count >= min_count, metric, np.nan)
    if drop_l0 and out.shape[1] > 1:
        out = out[:, 1:]
    return out


def _shared_range(arrs: list[np.ndarray]) -> tuple[float, float]:
    """Shared (vmin, vmax) over finite entries of several arrays (per-metric scale)."""
    finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrs if np.isfinite(a).any()])
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def _plot_row(
    aggs: dict[str, dict],
    *,
    metric: str,
    cmap: str,
    title: str,
    stem: str,
    min_count: int,
    drop_l0: bool,
) -> Path | None:
    """Render one metric row: one heatmap subplot per behavior, shared colorbar."""
    behaviors = [b for b in ALL_BEH if b in aggs]
    if not behaviors:
        logger.warning("no behaviors present for metric=%s — skipping", metric)
        return None
    key = "mag_mean" if metric == "magnitude" else "dir_mean"
    mats = {b: _masked(aggs[b][key], aggs[b]["count"], min_count, drop_l0) for b in behaviors}
    vmin, vmax = _shared_range(list(mats.values()))
    norm = Normalize(vmin=vmin, vmax=vmax)

    set_paper_style()
    fig, axes = plt.subplots(1, len(behaviors), figsize=(3.2 * len(behaviors), 4.4), squeeze=False)
    axes = axes[0]
    x0 = 1 if drop_l0 else 0
    im = None
    for ax, b in zip(axes, behaviors, strict=True):
        mat = mats[b]
        n_layers_shown = mat.shape[1]
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            extent=[x0 - 0.5, x0 + n_layers_shown - 0.5, mat.shape[0] - 0.5, -0.5],
        )
        ax.set_title(f"{BEH_LABEL.get(b, b)}\n({aggs[b]['n_cells']} sources)", fontsize=9)
        ax.set_xlabel("Layer")
    axes[0].set_ylabel("Answer-token position (0 = first)")
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    cbar.set_label(
        "Relative L2 shift  ||Δh||/||h||" if metric == "magnitude" else "cos(h_base, h_trained)"
    )
    fig.suptitle(title, fontsize=11)
    written = savefig_paper(fig, stem, dir=str(FIG_DIR), formats=("png",), embed_data=False)
    plt.close(fig)
    logger.info("wrote %s", written.get("png"))
    return written.get("png")


def _maybe_download_hf(tensors_dir: Path) -> Path:
    """If --tensors-dir is missing locally, snapshot the HF mirror into it."""
    if tensors_dir.exists() and any(tensors_dir.rglob("*.npz")):
        return tensors_dir
    logger.info("no local npz under %s — pulling HF mirror %s", tensors_dir, HF_PREFIX)
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=[f"{HF_PREFIX}/**"],
    )
    return Path(local) / HF_PREFIX


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #667 per-answer-token shift heatmaps.")
    parser.add_argument(
        "--tensors-dir",
        default=str(DEFAULT_TENSORS_DIR),
        help="Dir of per-cell *_pertoken.npz (default local; falls back to HF mirror).",
    )
    parser.add_argument(
        "--min-count", type=int, default=20, help="Mask (t,L) cells with total count < this."
    )
    parser.add_argument(
        "--drop-l0",
        dest="drop_l0",
        action="store_true",
        default=True,
        help="Drop layer 0 from the x-axis (known extraction-aliasing; default on).",
    )
    parser.add_argument("--keep-l0", dest="drop_l0", action="store_false")
    args = parser.parse_args()

    tensors_dir = _maybe_download_hf(Path(args.tensors_dir))
    aggs: dict[str, dict] = {}
    for behavior in ALL_BEH:
        npzs = sorted((tensors_dir / behavior).rglob("*_pertoken.npz"))
        agg = _aggregate_behavior(npzs)
        if agg is None:
            logger.warning("no per-cell npz for behavior=%s under %s", behavior, tensors_dir)
            continue
        aggs[behavior] = agg
        logger.info(
            "behavior=%s: %d cells, max_pos=%d, n_layers=%d, max count=%d",
            behavior,
            agg["n_cells"],
            agg["max_token_pos"],
            agg["n_layers"],
            int(agg["count"].max()),
        )
    if not aggs:
        raise RuntimeError(f"no per-cell npz found under {tensors_dir} — nothing to plot")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_row(
        aggs,
        metric="magnitude",
        cmap="viridis",
        title="Per-answer-token residual-shift MAGNITUDE (base → post-finetuning)",
        stem="pertoken_magnitude",
        min_count=args.min_count,
        drop_l0=args.drop_l0,
    )
    _plot_row(
        aggs,
        metric="direction",
        cmap="magma",
        title="Per-answer-token residual-shift DIRECTION  cos(h_base, h_trained)",
        stem="pertoken_direction",
        min_count=args.min_count,
        drop_l0=args.drop_l0,
    )
    # Also emit per-behavior single-panel PNGs (one figure per behavior x metric =
    # the 8 heatmaps the brief asks for, in addition to the 2 row overviews above).
    for behavior in aggs:
        one = {behavior: aggs[behavior]}
        _plot_row(
            one,
            metric="magnitude",
            cmap="viridis",
            title=f"{BEH_LABEL.get(behavior, behavior)} — shift magnitude",
            stem=f"pertoken_magnitude_{behavior}",
            min_count=args.min_count,
            drop_l0=args.drop_l0,
        )
        _plot_row(
            one,
            metric="direction",
            cmap="magma",
            title=f"{BEH_LABEL.get(behavior, behavior)} — shift direction (cosine)",
            stem=f"pertoken_direction_{behavior}",
            min_count=args.min_count,
            drop_l0=args.drop_l0,
        )
    logger.info("done: figures under %s", FIG_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
