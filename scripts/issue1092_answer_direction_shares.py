#!/usr/bin/env python3
"""#1092 inline: WHICH answer directions are prefix- vs query-predicted.

The crossed ANOVA (read3) gives magnitude shares only (prefix ~10% / query
~71% / interaction ~19%, ambient); this round decomposes the answer SPACE.
On the complete dense-core crossing (prefixes x shared queries), split the
grand-centered stacked answer states into the three component matrices

    Y[p,q] - grand = A_p[p] + A_q[q] + R[p,q]

(prefix main effect, query main effect, interaction residual), then:

  1. shares sanity: reproduce the banked read3 magnitudes;
  2. effect subspaces: SVD of A_p and A_q -> principal angles between the
     prefix-effect and query-effect answer subspaces at k=10/24/full, vs the
     Haar-random null (machinery reused from issue1092_partb_operator);
  3. cross-projection energy: fraction of prefix-effect variance inside the
     query-effect subspace and vice versa;
  4. per-direction spectrum: top answer PCs of the centered dense core, each
     labeled with its 1-D crossed shares (prefix/query/interaction) -> figure.

Analysis-only: reads the staged .npy summaries + manifest; writes
eval_results/issue_1092/inline_answer_direction_shares/answer_direction_shares.json
and figures/summaries/prefix_vs_context_map/answer_pc_shares.{png,pdf}.

Usage: uv run python scripts/issue1092_answer_direction_shares.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847): env caps must bind BEFORE numpy/torch import.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import issue1092_inline_fair_comparison as fc  # noqa: E402
from issue1092_partb_operator import _angle_null_band, _angles_between  # noqa: E402

OUT_DIR = PROJECT_ROOT / "eval_results/issue_1092/inline_answer_direction_shares"
OUT_PATH = OUT_DIR / "answer_direction_shares.json"
FIGDIR = PROJECT_ROOT / "figures/summaries/prefix_vs_context_map"
N_NULL_DRAWS = 100
NULL_CHUNK = 16
NULL_MAX_RANK = 384
SEED = 20260723
N_PCS_FIG = 24


def _dense_grid(rows: list[dict], n_states: int) -> tuple[list[str], list[str], np.ndarray]:
    """Complete dense-core crossing: (prefix ids, query ids, row-index grid (P, Q))."""
    dense = [
        (i, r["prefix_id"], r["query_id"])
        for i, r in enumerate(rows[:n_states])
        if r.get("stratum") == "dense_core"
    ]
    by_prefix: dict[str, dict[str, int]] = {}
    for i, p, q in dense:
        by_prefix.setdefault(p, {})[q] = i
    # Queries shared by every dense prefix; keep prefixes holding the full set.
    qsets = [set(d) for d in by_prefix.values()]
    shared_q = sorted(set.intersection(*qsets))
    pids = sorted(p for p, d in by_prefix.items() if all(q in d for q in shared_q))
    grid = np.asarray([[by_prefix[p][q] for q in shared_q] for p in pids], dtype=np.int64)
    return pids, shared_q, grid


def process_cell(cell: str, rows: list[dict]) -> dict:
    t0 = time.monotonic()
    t_all = [fc._load(cell, t) for t in fc.TARGETS]
    n_states = min(t.shape[0] for t in t_all)
    pids, qids, grid = _dense_grid(rows, n_states)
    P, Q = grid.shape
    flat = grid.reshape(-1)
    Y = np.concatenate([np.asarray(t[flat], dtype=np.float64) for t in t_all], axis=1)
    del t_all
    gc.collect()
    D = Y.shape[1]
    Yg = Y.reshape(P, Q, D)
    grand = Yg.mean(axis=(0, 1))
    Yc = Yg - grand
    A_p = Yc.mean(axis=1)  # (P, D) prefix main effect
    A_q = Yc.mean(axis=0)  # (Q, D) query main effect
    R = Yc - A_p[:, None, :] - A_q[None, :, :]  # interaction residual
    ss_p = Q * float((A_p**2).sum())
    ss_q = P * float((A_q**2).sum())
    ss_i = float((R**2).sum())
    ss_tot = float((Yc**2).sum())
    shares = {
        "share_prefix": ss_p / ss_tot,
        "share_query": ss_q / ss_tot,
        "share_interaction": ss_i / ss_tot,
        "n_prefixes": P,
        "n_queries": Q,
        "n_rows": P * Q,
    }

    # Effect subspaces in answer space: right singular vectors of each effect matrix.
    _, s_p, Vh_p = np.linalg.svd(A_p, full_matrices=False)
    _, s_q, Vh_q = np.linalg.svd(A_q, full_matrices=False)
    r_p, r_q = int((s_p > s_p[0] * 1e-9).sum()), int((s_q > s_q[0] * 1e-9).sum())

    gen = torch.Generator().manual_seed(SEED)
    subspaces = {}
    for name, k1, k2 in (
        ("k10", 10, 10),
        ("k24", 24, 24),
        ("full_rank", min(r_p, 384), min(r_q, 384)),
    ):
        k1, k2 = min(k1, r_p), min(k2, r_q)
        A = torch.from_numpy(np.ascontiguousarray(Vh_p[:k1].T)).double()
        B = torch.from_numpy(np.ascontiguousarray(Vh_q[:k2].T)).double()
        angles = _angles_between(A, B)
        subspaces[name] = {
            "k": [k1, k2],
            "mean_angle_deg": float(np.degrees(np.mean(angles))),
            "min_angle_deg": float(np.degrees(np.min(angles))),
            "null": _angle_null_band(D, k1, k2, N_NULL_DRAWS, NULL_CHUNK, gen, NULL_MAX_RANK),
        }

    # Cross-projection energy: effect variance captured by the OTHER effect's subspace.
    proj_p_in_q = float(((A_p @ Vh_q[:r_q].T) ** 2).sum() / (A_p**2).sum())
    proj_q_in_p = float(((A_q @ Vh_p[:r_p].T) ** 2).sum() / (A_q**2).sum())

    # Per-direction spectrum: top answer PCs of the centered dense core, 1-D shares each.
    Yflat = Yc.reshape(P * Q, D)
    _, s_t, Vh_t = np.linalg.svd(Yflat, full_matrices=False)
    n_pcs = min(48, Vh_t.shape[0])
    V = Vh_t[:n_pcs]  # (n_pcs, D)
    yp = A_p @ V.T  # (P, n_pcs)
    yq = A_q @ V.T  # (Q, n_pcs)
    yr = R.reshape(P * Q, D) @ V.T
    tot = (Yflat @ V.T) ** 2
    pc = {
        "pc_total_variance_frac": (s_t[:n_pcs] ** 2 / float((s_t**2).sum())).tolist(),
        "share_prefix": (Q * (yp**2).sum(0) / tot.sum(0)).tolist(),
        "share_query": (P * (yq**2).sum(0) / tot.sum(0)).tolist(),
        "share_interaction": ((yr**2).sum(0) / tot.sum(0)).tolist(),
    }

    out = {
        "shares": shares,
        "effect_ranks": {"prefix": r_p, "query": r_q},
        "subspace_angles_prefix_vs_query": subspaces,
        "cross_projection_energy": {
            "prefix_effect_in_query_subspace": proj_p_in_q,
            "query_effect_in_prefix_subspace": proj_q_in_p,
        },
        "per_pc": pc,
        "wall_s": round(time.monotonic() - t0, 1),
    }
    print(
        f"[{cell}] shares p/q/i = {shares['share_prefix']:.3f}/{shares['share_query']:.3f}/"
        f"{shares['share_interaction']:.3f} (n={P}x{Q}); angles k24 "
        f"{subspaces['k24']['mean_angle_deg']:.1f}deg; cross-proj p-in-q {proj_p_in_q:.3f} "
        f"q-in-p {proj_q_in_p:.3f} [{out['wall_s']}s]",
        flush=True,
    )
    return out


def make_figure(result: dict) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(3)
    order = ["share_query", "share_prefix", "share_interaction"]
    labels = ["Query", "Prefix", "Prefix x query interaction"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    cells = [("cell_inst_own", "Instruct model"), ("cell_pre_own", "Base model")]
    for ax, (cell, title) in zip(axes, cells, strict=True):
        pc = result["cells"][cell]["per_pc"]
        n = min(N_PCS_FIG, len(pc["share_query"]))
        x = np.arange(1, n + 1)
        bottom = np.zeros(n)
        for key, label, color in zip(order, labels, colors, strict=True):
            vals = np.asarray(pc[key][:n])
            ax.bar(x, vals, 0.8, bottom=bottom, label=label, color=color)
            bottom += vals
        ax.set_xlabel("Answer principal component (by variance)")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Variance share within the component")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(
        "What each answer direction encodes: crossed shares per answer PC — dense core, layer 14, ambient"
    )
    fig.tight_layout()
    savefig_paper(fig, "answer_pc_shares", dir=FIGDIR)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = fc._jsonl(fc.MANIFEST)
    result = {
        "meta": {
            "script": "scripts/issue1092_answer_direction_shares.py",
            "git_commit": fc._git_sha(),
            "layer": fc.LAYER,
            "basis": "ambient (stacked t1|t2|t3)",
            "seed": SEED,
            "definition": (
                "complete dense-core crossing; Y - grand = A_p + A_q + R; effect "
                "subspaces = right singular vectors of A_p / A_q; per-PC shares = 1-D "
                "crossed shares of each top answer PC of the centered dense core"
            ),
        },
        "cells": {},
    }
    for cell in fc.CELLS:
        result["cells"][cell] = process_cell(cell, rows)
        gc.collect()
    OUT_PATH.write_text(json.dumps(result, indent=1))
    make_figure(result)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
