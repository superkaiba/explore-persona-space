#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ×, →) in scientific docstrings + labels.
"""Per-context data figure behind the round-2 paired cross-genre Δskill aggregate.

The clean-result's "linear map transfers to UltraChat" result headlines the
paired per-context bootstrap Δskill (mean +0.004, max-pool +0.098; committed
``eval_results/issue_810/ultrachat-genre-summary-sweep/genre_delta_recon.json``)
whose per-context ``(ss_res_i, ss_tot_i)`` decompositions were computed by
``issue810_bootstrap_deltaskill._cross_genre`` in memory but never persisted.
This script recomputes EXACTLY those decompositions for the two headline cells
— the ``mean`` summary at layer 18 and ``maxp`` at layer 21, each arm's
observed best layer (identical across genres per the committed
``*_frozen_observed_best_layers`` statistics) — through the same loaders and
``_per_context_decomposition`` primitive, asserts the recomputed aggregate
skill matches the committed ``per_layer_observed_skill`` values, persists the
per-context error fractions to
``eval_results/.../analysis/genre_delta_percontext.json``, and renders the
per-unit scatter (misalignment-pool error fraction (x) vs UltraChat (y), one
point per context, most off-diagonal contexts labeled with reader-facing
battery names).

Usage::

    uv run python scripts/issue810_g1_percontext_deltaskill_figure.py
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
    G1_OUT_DIR,
    HF_DATA_REPO,
    I658_STORE_MANIFEST,
    PCA_TARGET_DIM_CAP,
    context_ids_from_manifest,
    dump_json,
    load_json,
    reader_context_labels,
    reproducibility_metadata,
)
from issue810_fit_reconstruction import _load_cc_for_genre, _load_free_summaries  # noqa: E402

logger = logging.getLogger("issue810_g1_percontext_deltaskill_figure")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# The two headline cells of the paired Δskill read: each arm's observed best
# layer (identical across genres — genre_delta_recon.json statistics).
CELLS = {"mean": 18, "maxp": 21}
GENRES = ("betley", "g1")
GENRE_READER = {"betley": "misalignment pool", "g1": "UltraChat"}
SUMMARY_READER = {"mean": "mean summary", "maxp": "max-pool summary"}
N_LABELED = 6  # most off-diagonal contexts labeled per panel


def compute_decompositions() -> tuple[list[str], dict]:
    """Per-context (ss_res, ss_tot) for the 4 headline (genre × summary) cells.

    Returns (ctx_ids, {(genre, summary): {"ss_res", "ss_tot", "skill"}}) and
    asserts each recomputed aggregate skill matches the committed
    ``genre_delta_recon.json`` per-layer value to 1e-6 (same inputs, same
    primitives — drift means the store or code moved; fail loud).
    """
    import json as _json

    from huggingface_hub import hf_hub_download

    man_path = hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset")
    with open(man_path) as f:
        ctx_ids = context_ids_from_manifest(_json.load(f))
    n = len(ctx_ids)
    pca_dim = min(PCA_TARGET_DIM_CAP, n - 2)

    committed = load_json(G1_OUT_DIR / "genre_delta_recon.json")["per_layer_observed_skill"]

    out: dict[tuple[str, str], dict] = {}
    for g in GENRES:
        free, capture_layers = _load_free_summaries(g)
        cc = _load_cc_for_genre(g, ctx_ids, capture_layers)
        for s, layer in CELLS.items():
            Xc = np.stack([cc[c][layer] for c in ctx_ids])
            Yv = np.stack([free[s][c][layer].numpy() for c in ctx_ids])
            assert Xc.shape[0] == Yv.shape[0] == n, (Xc.shape, Yv.shape)
            ss_res, ss_tot = _per_context_decomposition(Xc, Yv, pca_dim)
            skill = _skill(ss_res, ss_tot)
            ref = committed[f"{g}/{s}"][layer]
            if abs(skill - ref) > 1e-6:
                raise RuntimeError(
                    f"recomputed skill drifts from committed genre_delta_recon.json: "
                    f"{g}/{s}@L{layer} {skill:.8f} != {ref:.8f}"
                )
            out[(g, s)] = {"ss_res": ss_res, "ss_tot": ss_tot, "skill": skill}
            logger.info("%s/%s@L%d skill %.4f (matches committed)", g, s, layer, skill)
    return ctx_ids, out


def main() -> None:
    """Persist the per-context decompositions + render the per-unit scatter."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    ctx_ids, dec = compute_decompositions()
    labels = reader_context_labels()

    dump_json(
        {
            "dv": "per_context_error_fractions_behind_paired_cross_genre_delta_skill",
            "method": (
                "per-context (ss_res_i, ss_tot_i) of the held-out LOCO ridge predictions, "
                "recomputed via issue810_bootstrap_deltaskill._per_context_decomposition for "
                "the two headline cells (mean@L18, maxp@L21, both genres); aggregate skill "
                "asserted equal to the committed genre_delta_recon.json per-layer values"
            ),
            "context_ids": ctx_ids,
            "cells": {
                f"{g}/{s}": {
                    "layer": CELLS[s],
                    "skill": dec[(g, s)]["skill"],
                    "ss_res": dec[(g, s)]["ss_res"].tolist(),
                    "ss_tot": dec[(g, s)]["ss_tot"].tolist(),
                    "error_fraction": (dec[(g, s)]["ss_res"] / dec[(g, s)]["ss_tot"]).tolist(),
                }
                for g in GENRES
                for s in CELLS
            },
            "reproducibility": reproducibility_metadata(),
        },
        G1_OUT_DIR / "analysis" / "genre_delta_percontext.json",
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for ax, s in zip(axes, CELLS, strict=True):
        rx = dec[("betley", s)]["ss_res"] / dec[("betley", s)]["ss_tot"]
        ry = dec[("g1", s)]["ss_res"] / dec[("g1", s)]["ss_tot"]
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
        ax.set_xlabel("held-out error fraction, misalignment pool")
        ax.set_ylabel("held-out error fraction, UltraChat")
        ax.set_title(f"{SUMMARY_READER[s]} (layer {CELLS[s]})")
        above = int((ry > rx).sum())
        logger.info("%s: %d/%d contexts above diagonal (worse on UltraChat)", s, above, len(rx))
    fig.tight_layout()
    savefig_paper(
        fig,
        "issue_810/ultrachat-genre-summary-sweep/g1_percontext_deltaskill_scatter",
        dir="figures/",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
