#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ×, →) in scientific docstrings + labels.
"""Per-context data figures behind the round-3 paired Δskill + cross-layer aggregates.

The `user-header-newline-summary` round headlines two aggregate reads whose
per-context ``(ss_res_i, ss_tot_i)`` decompositions were computed in memory
but never persisted: (a) the paired-bootstrap Δskill forest (each of the 9 new
rows vs the ``mean`` benchmark at the frozen layer 18; committed
``eval_results/issue_810/user-header-newline-summary/delta_vs_mean.json``) and
(b) the cross-layer pooled winner (``maxp_xbnd|answer=layer-mean|raw|cc=layer-mean``,
0.882; committed ``crosslayer_xbnd.json``). This script recomputes EXACTLY
those decompositions through the same loaders and the same
``_per_context_decomposition`` primitive, asserts every recomputed aggregate
skill matches its committed value to 1e-6, persists the per-context error
fractions to ``eval_results/.../analysis/uh_percontext.json``, and renders two
per-unit figures:

- ``uh_percontext_deltaskill_scatter`` — left: per-context error-fraction gap
  (mean − row) at layer 18 for all 9 new rows (one point per context, the
  whole-turn mean column highlighted); right: paired per-context error
  fractions for the carry-deciding cell (mean vs whole-turn mean, layer 18),
  most off-diagonal contexts labeled with reader-facing battery names.
- ``uh_crosslayer_percontext_scatter`` — paired per-context error fractions
  for the cross-layer winner vs the committed per-layer best (answer-only
  max-pool at layer 21).

Usage::

    uv run python scripts/issue810_uh_percontext_deltaskill_figure.py
"""

from __future__ import annotations

import logging

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue810_bootstrap_deltaskill import _per_context_decomposition, _skill  # noqa: E402
from issue810_common import (  # noqa: E402
    HF_DATA_REPO,
    I658_STORE_MANIFEST,
    PCA_TARGET_DIM_CAP,
    UH_OUT_DIR,
    UH_SUMMARIES_HF_FILE,
    UH_SUMMARY_NAMES,
    context_ids_from_manifest,
    dump_json,
    load_json,
    reader_context_labels,
    reproducibility_metadata,
    validate_uh_pack,
)
from issue810_fit_readout import _load_uh_summaries  # noqa: E402
from issue810_fit_reconstruction import _load_cc, _load_free_summaries  # noqa: E402
from issue810_uh_crosslayer import _pool_layers  # noqa: E402

logger = logging.getLogger("issue810_uh_percontext_deltaskill_figure")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FROZEN_LAYER = 18  # the mean benchmark's frozen layer (delta_vs_mean.json `at_L18` reads)
PER_LAYER_BEST = ("maxp", 21)  # committed per-layer best (round-1 answer-only max-pool)
N_LABELED = 6  # most off-diagonal contexts labeled in each paired scatter

ROW_LABELS = {
    "uh_im_start": "header start",
    "uh_user": "header 'user'",
    "uh_nl": "header newline",
    "uh_mean3": "header mean-3",
    "uh_max3": "header max-3",
    "bnd_mean5": "boundary mean-5",
    "bnd_max5": "boundary max-5",
    "mean_xbnd": "whole-turn mean",
    "maxp_xbnd": "whole-turn max",
}


def compute_decompositions() -> tuple[list[str], dict[str, dict]]:
    """Per-context (ss_res, ss_tot) for the round-3 headline cells.

    Cells: the ``mean`` benchmark + all 9 new rows at the frozen layer 18
    (behind the paired-bootstrap forest), the committed per-layer best
    (answer-only max-pool at layer 21), and the cross-layer pooled winner.
    Each recomputed aggregate skill is asserted equal to its committed
    ``delta_vs_mean.json`` / ``crosslayer_xbnd.json`` value to 1e-6 (same
    inputs, same primitives — drift means the pack or code moved; fail loud).
    """
    import json as _json

    from huggingface_hub import hf_hub_download

    man_path = hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset")
    with open(man_path) as f:
        ctx_ids = context_ids_from_manifest(_json.load(f))
    n = len(ctx_ids)
    pca_dim = min(PCA_TARGET_DIM_CAP, n - 2)

    committed = load_json(UH_OUT_DIR / "delta_vs_mean.json")["per_layer_observed_skill"]
    xbnd = load_json(UH_OUT_DIR / "crosslayer_xbnd.json")

    local_pack = PROJECT_ROOT / "data" / "issue_810" / "uh_summaries.pt"
    uh_rows, uh_cov, meta = _load_uh_summaries(
        str(local_pack) if local_pack.is_file() else UH_SUMMARIES_HF_FILE
    )
    free, capture_layers = _load_free_summaries()
    validate_uh_pack(
        uh_rows,
        uh_cov,
        meta,
        requested_rows=list(UH_SUMMARY_NAMES),
        ctx_ids=ctx_ids,
        expected_capture_layers=capture_layers,
    )
    cc = _load_cc(ctx_ids, capture_layers)

    def _cell(Xc: np.ndarray, Yv: np.ndarray, name: str, ref: float) -> dict:
        assert Xc.shape[0] == Yv.shape[0] == n, (Xc.shape, Yv.shape)
        ss_res, ss_tot = _per_context_decomposition(Xc, Yv, pca_dim)
        skill = _skill(ss_res, ss_tot)
        if abs(skill - ref) > 1e-6:
            raise RuntimeError(
                f"recomputed skill drifts from committed value: {name} {skill:.8f} != {ref:.8f}"
            )
        logger.info("%s skill %.4f (matches committed)", name, skill)
        return {"ss_res": ss_res, "ss_tot": ss_tot, "skill": skill}

    out: dict[str, dict] = {}
    # (a) forest cells: mean benchmark + 9 new rows at the frozen layer 18.
    Xc18 = np.stack([cc[c][FROZEN_LAYER] for c in ctx_ids])
    Yv = np.stack([free["mean"][c][FROZEN_LAYER].numpy() for c in ctx_ids])
    out["mean@L18"] = _cell(Xc18, Yv, "mean@L18", committed["mean"][FROZEN_LAYER])
    for row in UH_SUMMARY_NAMES:
        Yv = np.stack([uh_rows[row][c][FROZEN_LAYER] for c in ctx_ids])
        out[f"{row}@L18"] = _cell(Xc18, Yv, f"{row}@L18", committed[row][FROZEN_LAYER])
    # (b) the committed per-layer best: answer-only max-pool at layer 21.
    s, layer = PER_LAYER_BEST
    Xc = np.stack([cc[c][layer] for c in ctx_ids])
    Yv = np.stack([free[s][c][layer].numpy() for c in ctx_ids])
    out["maxp@L21"] = _cell(Xc, Yv, "maxp@L21", xbnd["band_row_h3"]["per_layer_best_committed"])
    # (c) the cross-layer pooled winner: maxp_xbnd|answer=layer-mean|raw|cc=layer-mean.
    assert xbnd["band_row_h3"]["arg_cell"] == "maxp_xbnd|answer=layer-mean|raw|cc=layer-mean"
    Xc = np.stack([_pool_layers(cc[c], "layer-mean", normed=False) for c in ctx_ids])
    Yv = np.stack(
        [_pool_layers(uh_rows["maxp_xbnd"][c], "layer-mean", normed=False) for c in ctx_ids]
    )
    out["crosslayer_winner"] = _cell(Xc, Yv, "crosslayer_winner", xbnd["band_row_h3"]["statistic"])
    return ctx_ids, out


def _paired_scatter(ax, rx: np.ndarray, ry: np.ndarray, ctx_ids, labels, xlab, ylab, title):
    """Log-log paired per-context error-fraction scatter with labeled outliers."""
    import matplotlib.pyplot as plt  # noqa: F401  (style already set by caller)

    from explore_persona_space.analysis.paper_plots import paper_palette_role

    lims = [min(rx.min(), ry.min()) * 0.8, max(rx.max(), ry.max()) * 1.25]
    ax.plot(lims, lims, ls="--", lw=1.0, color=paper_palette_role("neutral"), zorder=1)
    ax.scatter(rx, ry, s=28, color=paper_palette_role("primary"), alpha=0.85, zorder=2)
    gap = np.abs(np.log(ry) - np.log(rx))
    for rank, i in enumerate(np.argsort(gap)[-N_LABELED:]):
        ax.text(
            rx[i] * 1.05,
            ry[i] * (1.06 if rank % 2 else 0.94),
            labels[ctx_ids[i]],
            fontsize=5.5,
            va="bottom" if rank % 2 else "top",
            color="#444444",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    below = int((ry < rx).sum())
    logger.info("%s: %d/%d contexts below diagonal (better on y-arm)", title, below, len(rx))


def main() -> None:
    """Persist the per-context decompositions + render the two per-unit figures."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    ctx_ids, dec = compute_decompositions()
    labels = reader_context_labels()
    err = {k: v["ss_res"] / v["ss_tot"] for k, v in dec.items()}

    dump_json(
        {
            "dv": "per_context_error_fractions_behind_round3_paired_delta_skill_and_crosslayer",
            "method": (
                "per-context (ss_res_i, ss_tot_i) of the held-out LOCO ridge predictions, "
                "recomputed via issue810_bootstrap_deltaskill._per_context_decomposition for "
                "the mean benchmark + 9 new rows at the frozen layer 18, the committed "
                "per-layer best (answer-only max-pool at layer 21), and the cross-layer "
                "pooled winner; every aggregate skill asserted equal to the committed "
                "delta_vs_mean.json / crosslayer_xbnd.json value"
            ),
            "context_ids": ctx_ids,
            "cells": {
                name: {
                    "skill": d["skill"],
                    "ss_res": d["ss_res"].tolist(),
                    "ss_tot": d["ss_tot"].tolist(),
                    "error_fraction": (d["ss_res"] / d["ss_tot"]).tolist(),
                }
                for name, d in dec.items()
            },
            "reproducibility": reproducibility_metadata(),
        },
        UH_OUT_DIR / "analysis" / "uh_percontext.json",
    )

    # ── figure 1: per-unit data behind the paired-bootstrap forest ───────────
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    rng = np.random.default_rng(42)
    e_mean = err["mean@L18"]
    for i, row in enumerate(UH_SUMMARY_NAMES):
        d = e_mean - err[f"{row}@L18"]  # >0: the new row reconstructs the context better
        hl = row == "mean_xbnd"
        x = np.full_like(d, float(i)) + rng.uniform(-0.16, 0.16, size=d.size)
        ax.scatter(
            x,
            d,
            s=16 if hl else 12,
            color=paper_palette_role("primary" if hl else "neutral"),
            alpha=0.9 if hl else 0.55,
            zorder=3 if hl else 2,
        )
    ax.axhline(0.0, color="0.3", lw=1.0, zorder=1)
    ax.set_xticks(range(len(UH_SUMMARY_NAMES)))
    ax.set_xticklabels([ROW_LABELS[r] for r in UH_SUMMARY_NAMES], rotation=35, ha="right")
    ax.set_ylabel("per-context error-fraction gap\n(mean − new row) at layer 18")
    ax.set_title("all 9 new rows, one point per context")
    _paired_scatter(
        axes[1],
        e_mean,
        err["mean_xbnd@L18"],
        ctx_ids,
        labels,
        "held-out error fraction, mean summary",
        "held-out error fraction, whole-turn mean",
        "carry-deciding cell (layer 18)",
    )
    fig.tight_layout()
    savefig_paper(
        fig,
        "issue_810/user-header-newline-summary/uh_percontext_deltaskill_scatter",
        dir="figures/",
    )
    plt.close(fig)

    # ── figure 2: per-unit data behind the cross-layer pooled winner ─────────
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    _paired_scatter(
        ax,
        err["maxp@L21"],
        err["crosslayer_winner"],
        ctx_ids,
        labels,
        "held-out error fraction, per-layer best",
        "held-out error fraction, cross-layer winner",
        "cross-layer winner vs per-layer best (max-pool, layer 21)",
    )
    fig.tight_layout()
    savefig_paper(
        fig,
        "issue_810/user-header-newline-summary/uh_crosslayer_percontext_scatter",
        dir="figures/",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
