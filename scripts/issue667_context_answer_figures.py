#!/usr/bin/env python3
"""Issue #667 COMBINED context+answer per-token residual-shift heatmaps.

Stitches the CONTEXT-side per-token shift (from :mod:`issue667_pertoken_context_extract`,
store ``issue667_pertoken_context/analysis_tensors``) ABOVE the ANSWER-side
per-token shift (from :mod:`issue667_pertoken_extract`, store
``issue667_pertoken/analysis_tensors``) into ONE continuous trajectory per
behavior, for magnitude AND direction.

Layout (per behavior panel, brief §2):

    y-axis (top -> bottom):
      CONTEXT tokens, from-END alignment, FARTHEST-from-boundary at the TOP
        (offset max-1) ... offset 0 (LAST INPUT token) at the CONTEXT/ANSWER
        BOUNDARY (a horizontal divider line), THEN
      ANSWER tokens, from-START alignment, position 0 (first answer token) just
        below the boundary ... last answer position at the bottom.
    x-axis: layer 1..N_LAYERS-1 (L0 dropped, brief §5 / matching the answer figure).

So the LAST context token sits immediately above the FIRST answer token — a
continuous base->post-FT residual trajectory across the context/answer boundary.

Per behavior the aggregate over source cells (count-weighted mean) is::

    ctx_mag[o, L] = sum_cells end_mag_sum[o, L] / sum_cells end_count[o, L]
    ans_mag[t, L] = sum_cells (answer) mag_sum[t, L] / sum_cells count[t, L]

(and likewise for direction / cosine). (o, L) / (t, L) cells with total count <
--min-count are masked (grey). Shared colorbar per metric (across all behaviors).

Outputs:
  - figures/issue_667_alllayer/context_answer_magnitude.png   (4-behavior overview)
  - figures/issue_667_alllayer/context_answer_direction.png   (4-behavior overview)
  - figures/issue_667_alllayer/context_answer_{metric}_{behavior}.png (per-behavior)
  - (supplementary, --from-start-supp) figures/.../context_fromstart_{metric}.png
    — the CONTEXT-only from-START overview (persona-preamble region).

Usage::

    uv run python scripts/issue667_context_answer_figures.py           # HF mirrors
    uv run python scripts/issue667_context_answer_figures.py \\
        --context-dir /tmp/i667ctx_smoke --answer-dir /tmp/i667pt_smoke
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

logger = logging.getLogger("issue667_context_answer_figures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ANSWER_DEFAULT_DIR = PROJECT_ROOT / "eval_results" / "issue_667_pertoken" / "analysis_tensors"
CONTEXT_DEFAULT_DIR = (
    PROJECT_ROOT / "eval_results" / "issue_667_pertoken_context" / "analysis_tensors"
)
FIG_DIR = PROJECT_ROOT / "figures" / "issue_667_alllayer"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_ANSWER_PREFIX = "issue667_pertoken/analysis_tensors"
HF_CONTEXT_PREFIX = "issue667_pertoken_context/analysis_tensors"

BEH_LABEL = {
    "em": "Emergent misalignment",
    "sycophancy": "Sycophancy",
    "fact": "Taught fact",
    "marker": "Marker",
}
ALL_BEH = ["em", "sycophancy", "fact", "marker"]


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation (count-weighted mean per (position, layer) over per-cell npzs)
# ─────────────────────────────────────────────────────────────────────────────


def _agg_answer(npzs: list[Path]) -> dict | None:
    """Count-weighted answer-side mean per (answer-pos, L) — issue667_pertoken store."""
    mag_sum = dir_sum = count = None
    n_cells = 0
    for p in npzs:
        d = np.load(p, allow_pickle=True)
        ms = d["mag_sum"].astype(np.float64)
        ds = d["dir_sum"].astype(np.float64)
        c = d["count"].astype(np.int64)
        if mag_sum is None:
            mag_sum, dir_sum, count = (np.zeros_like(ms), np.zeros_like(ds), np.zeros_like(c))
        assert ms.shape == mag_sum.shape, (p, ms.shape, mag_sum.shape)
        mag_sum += ms
        dir_sum += ds
        count += c
        n_cells += 1
    if mag_sum is None:
        return None
    return {"mag_sum": mag_sum, "dir_sum": dir_sum, "count": count, "n_cells": n_cells}


def _agg_context(npzs: list[Path]) -> dict | None:
    """Count-weighted context-side mean per (pos, L), BOTH alignments — context store."""
    acc: dict[str, np.ndarray] = {}
    n_cells = 0
    selfcheck_pass = 0
    for p in npzs:
        d = np.load(p, allow_pickle=True)
        for k in ("start_mag_sum", "start_dir_sum", "start_count",
                  "end_mag_sum", "end_dir_sum", "end_count"):  # fmt: skip
            arr = d[k].astype(np.float64 if k.endswith("sum") else np.int64)
            if k not in acc:
                acc[k] = np.zeros_like(arr)
            assert arr.shape == acc[k].shape, (p, k, arr.shape, acc[k].shape)
            acc[k] += arr
        if bool(d["selfcheck_passed"]):
            selfcheck_pass += 1
        n_cells += 1
    if not acc:
        return None
    acc["n_cells"] = n_cells
    acc["selfcheck_pass"] = selfcheck_pass
    return acc


def _mean_masked(num: np.ndarray, count: np.ndarray, min_count: int) -> np.ndarray:
    """count-weighted mean, np.nan where count < min_count."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count >= min_count, num / count.astype(np.float64), np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Combined stack (context from-END above answer from-START)
# ─────────────────────────────────────────────────────────────────────────────


def _combined_stack(
    ctx: dict, ans: dict, *, metric: str, min_count: int, drop_l0: bool
) -> tuple[np.ndarray, int]:
    """Stack context(from-END) above answer(from-START) into ONE (rows, L') matrix.

    Returns (matrix, ctx_rows) where ctx_rows is the count of context rows above
    the boundary (so the caller can draw the divider at y = ctx_rows - 0.5).
    The matrix top->bottom is: context offset (max_off-1) ... offset 0 (boundary),
    then answer position 0 ... last. L0 dropped when drop_l0.
    """
    sum_key = "mag_sum" if metric == "magnitude" else "dir_sum"
    end_sum_key = "end_mag_sum" if metric == "magnitude" else "end_dir_sum"
    # answer: from-START (pos 0 = first answer token) — origin upper, so row 0 is
    # already the first answer token; keep order (top of the answer block).
    ans_mat = _mean_masked(ans[sum_key], ans["count"], min_count)  # (max_pos, L)
    # context from-END: end store row r == offset r (offset 0 = last input token).
    # For the stack we want the boundary (offset 0) at the BOTTOM of the context
    # block, farthest-back offset at the TOP -> reverse the offset axis.
    ctx_mat_end = _mean_masked(ctx[end_sum_key], ctx["end_count"], min_count)  # (max_off, L)
    ctx_mat = ctx_mat_end[::-1, :]  # row 0 = farthest-back offset, last row = offset 0
    assert ctx_mat.shape[1] == ans_mat.shape[1], (ctx_mat.shape, ans_mat.shape)
    combined = np.concatenate([ctx_mat, ans_mat], axis=0)  # (ctx_rows + ans_rows, L)
    ctx_rows = ctx_mat.shape[0]
    if drop_l0 and combined.shape[1] > 1:
        combined = combined[:, 1:]
    return combined, ctx_rows


def _shared_range(mats: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([m[np.isfinite(m)].ravel() for m in mats if np.isfinite(m).any()])
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def _plot_combined_row(
    aggs: dict[str, tuple[dict, dict]],
    *,
    metric: str,
    cmap: str,
    title: str,
    stem: str,
    min_count: int,
    drop_l0: bool,
) -> Path | None:
    """One metric row: per-behavior stacked context+answer heatmap, shared colorbar."""
    behaviors = [b for b in ALL_BEH if b in aggs]
    if not behaviors:
        logger.warning("no behaviors present for metric=%s — skipping", metric)
        return None
    stacks = {}
    for b in behaviors:
        ctx, ans = aggs[b]
        stacks[b] = _combined_stack(ctx, ans, metric=metric, min_count=min_count, drop_l0=drop_l0)
    vmin, vmax = _shared_range([m for m, _ in stacks.values()])
    norm = Normalize(vmin=vmin, vmax=vmax)

    set_paper_style()
    fig, axes = plt.subplots(1, len(behaviors), figsize=(3.4 * len(behaviors), 5.2), squeeze=False)
    axes = axes[0]
    x0 = 1 if drop_l0 else 0
    im = None
    for ax, b in zip(axes, behaviors, strict=True):
        mat, ctx_rows = stacks[b]
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
        # divider at the context/answer boundary (below the last context row,
        # which is offset 0 == last input token; row index ctx_rows-1).
        ax.axhline(y=ctx_rows - 0.5, color="white", linewidth=1.4, linestyle="-")
        ctx = aggs[b][0]
        ax.set_title(
            f"{BEH_LABEL.get(b, b)}\n({ctx['n_cells']} sources)",
            fontsize=9,
        )
        ax.set_xlabel("Layer")
    # y-axis top->bottom = context (from end, farthest-back at top -> last-input
    # token at the boundary), then answer (from start, first answer token just
    # below the boundary). The label is written top->bottom to match the plot.
    axes[0].set_ylabel("context tokens (from end, top)  →  answer tokens (from start, bottom)")
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    cbar.set_label(
        "Relative L2 shift  ||Δh||/||h||" if metric == "magnitude" else "cos(h_base, h_trained)"
    )
    fig.suptitle(title, fontsize=11)
    written = savefig_paper(fig, stem, dir=str(FIG_DIR), formats=("png",), embed_data=False)
    plt.close(fig)
    logger.info("wrote %s", written.get("png"))
    return written.get("png")


def _plot_context_fromstart_row(
    ctx_aggs: dict[str, dict], *, metric: str, cmap: str, title: str, stem: str,
    min_count: int, drop_l0: bool,
) -> Path | None:  # fmt: skip
    """Supplementary: CONTEXT-only from-START overview (persona-preamble region)."""
    behaviors = [b for b in ALL_BEH if b in ctx_aggs]
    if not behaviors:
        return None
    start_sum_key = "start_mag_sum" if metric == "magnitude" else "start_dir_sum"
    mats = {}
    for b in behaviors:
        m = _mean_masked(ctx_aggs[b][start_sum_key], ctx_aggs[b]["start_count"], min_count)
        if drop_l0 and m.shape[1] > 1:
            m = m[:, 1:]
        mats[b] = m
    vmin, vmax = _shared_range(list(mats.values()))
    norm = Normalize(vmin=vmin, vmax=vmax)
    set_paper_style()
    fig, axes = plt.subplots(1, len(behaviors), figsize=(3.2 * len(behaviors), 4.4), squeeze=False)
    axes = axes[0]
    x0 = 1 if drop_l0 else 0
    im = None
    for ax, b in zip(axes, behaviors, strict=True):
        mat = mats[b]
        im = ax.imshow(
            mat, aspect="auto", origin="upper", cmap=cmap, norm=norm, interpolation="nearest",
            extent=[x0 - 0.5, x0 + mat.shape[1] - 0.5, mat.shape[0] - 0.5, -0.5],
        )  # fmt: skip
        ax.set_title(f"{BEH_LABEL.get(b, b)}\n({ctx_aggs[b]['n_cells']} sources)", fontsize=9)
        ax.set_xlabel("Layer")
    axes[0].set_ylabel("Context-token position (0 = first prompt token)")
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    cbar.set_label(
        "Relative L2 shift  ||Δh||/||h||" if metric == "magnitude" else "cos(h_base, h_trained)"
    )
    fig.suptitle(title, fontsize=11)
    written = savefig_paper(fig, stem, dir=str(FIG_DIR), formats=("png",), embed_data=False)
    plt.close(fig)
    logger.info("wrote %s", written.get("png"))
    return written.get("png")


# ─────────────────────────────────────────────────────────────────────────────
# HF mirror fetch
# ─────────────────────────────────────────────────────────────────────────────


def _maybe_download_hf(local_dir: Path, hf_prefix: str) -> Path:
    if local_dir.exists() and any(local_dir.rglob("*.npz")):
        return local_dir
    logger.info("no local npz under %s — pulling HF mirror %s", local_dir, hf_prefix)
    from huggingface_hub import snapshot_download

    local = snapshot_download(HF_DATA_REPO, repo_type="dataset", allow_patterns=[f"{hf_prefix}/**"])
    return Path(local) / hf_prefix


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #667 combined context+answer heatmaps.")
    parser.add_argument("--answer-dir", default=str(ANSWER_DEFAULT_DIR))
    parser.add_argument("--context-dir", default=str(CONTEXT_DEFAULT_DIR))
    parser.add_argument("--min-count", type=int, default=20, help="Mask (pos,L) with count < this.")
    parser.add_argument(
        "--drop-l0", dest="drop_l0", action="store_true", default=True,
        help="Drop layer 0 from the x-axis (default on; matches the answer figure).",
    )  # fmt: skip
    parser.add_argument("--keep-l0", dest="drop_l0", action="store_false")
    parser.add_argument(
        "--from-start-supp", action="store_true", default=True,
        help="Also emit the supplementary CONTEXT-only from-START overview (default on).",
    )  # fmt: skip
    parser.add_argument("--no-from-start-supp", dest="from_start_supp", action="store_false")
    args = parser.parse_args()

    answer_dir = _maybe_download_hf(Path(args.answer_dir), HF_ANSWER_PREFIX)
    context_dir = _maybe_download_hf(Path(args.context_dir), HF_CONTEXT_PREFIX)

    aggs: dict[str, tuple[dict, dict]] = {}
    ctx_aggs: dict[str, dict] = {}
    for behavior in ALL_BEH:
        ans_npzs = sorted((answer_dir / behavior).rglob("*_pertoken.npz"))
        ctx_npzs = sorted((context_dir / behavior).rglob("*_pertoken_context.npz"))
        ans = _agg_answer(ans_npzs)
        ctx = _agg_context(ctx_npzs)
        if ans is None or ctx is None:
            logger.warning(
                "behavior=%s: answer=%s context=%s — skipping combined panel",
                behavior,
                "present" if ans else "MISSING",
                "present" if ctx else "MISSING",
            )
            if ctx is not None:
                ctx_aggs[behavior] = ctx
            continue
        aggs[behavior] = (ctx, ans)
        ctx_aggs[behavior] = ctx
        logger.info(
            "behavior=%s: %d answer cells, %d context cells, selfcheck-pass %d/%d",
            behavior,
            ans["n_cells"],
            ctx["n_cells"],
            ctx.get("selfcheck_pass", 0),
            ctx["n_cells"],
        )
    if not aggs:
        raise RuntimeError("no behavior has BOTH context+answer per-cell npz — nothing to combine")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_combined_row(
        aggs, metric="magnitude", cmap="viridis",
        title="Context+answer per-token residual-shift MAGNITUDE (base → post-finetuning)",
        stem="context_answer_magnitude", min_count=args.min_count, drop_l0=args.drop_l0,
    )  # fmt: skip
    _plot_combined_row(
        aggs, metric="direction", cmap="magma",
        title="Context+answer per-token residual-shift DIRECTION  cos(h_base, h_trained)",
        stem="context_answer_direction", min_count=args.min_count, drop_l0=args.drop_l0,
    )  # fmt: skip
    # per-behavior close-ups
    for behavior in aggs:
        one = {behavior: aggs[behavior]}
        _plot_combined_row(
            one, metric="magnitude", cmap="viridis",
            title=f"{BEH_LABEL.get(behavior, behavior)} — context+answer shift magnitude",
            stem=f"context_answer_magnitude_{behavior}", min_count=args.min_count,
            drop_l0=args.drop_l0,
        )  # fmt: skip
        _plot_combined_row(
            one, metric="direction", cmap="magma",
            title=f"{BEH_LABEL.get(behavior, behavior)} — context+answer shift direction (cosine)",
            stem=f"context_answer_direction_{behavior}", min_count=args.min_count,
            drop_l0=args.drop_l0,
        )  # fmt: skip
    # supplementary from-START context overview
    if args.from_start_supp and ctx_aggs:
        _plot_context_fromstart_row(
            ctx_aggs, metric="magnitude", cmap="viridis",
            title="Context (from start) residual-shift MAGNITUDE (base → post-finetuning)",
            stem="context_fromstart_magnitude", min_count=args.min_count, drop_l0=args.drop_l0,
        )  # fmt: skip
        _plot_context_fromstart_row(
            ctx_aggs, metric="direction", cmap="magma",
            title="Context (from start) residual-shift DIRECTION  cos(h_base, h_trained)",
            stem="context_fromstart_direction", min_count=args.min_count, drop_l0=args.drop_l0,
        )  # fmt: skip
    logger.info("done: figures under %s", FIG_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
