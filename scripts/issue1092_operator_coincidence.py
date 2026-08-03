#!/usr/bin/env python3
"""#1092 inline: operator-coincidence test — independently-fit averaged map vs
the induced context map at averaged grain.

Question (user-chat, 2026-07-22): the linearity identity mean_q[M v_C] = M v_bar_C
shows the per-row context map INDUCES an averaged-grain predictor; the historical
"averaged prefix map" was FIT independently on averaged rows. Are the two the
same operator, or does the averaged fit carry between-prefix structure of its own?

Three legs per (cell, basis), matched folds:
  1. skill gap: held-out R2 of the averaged REFIT (v_bar_C -> per-prefix profile)
     vs the INDUCED read (per-row context fit's held-out predictions averaged per
     prefix) — fold partition ALIGNED (prefix folds derived from the grouped
     per-row folds, so both arms hold out the same prefixes together);
  2. prediction agreement: variance-weighted R2 between the two arms' held-out
     averaged-grain predictions (both directions), per-prefix error win rate,
     median per-prefix error ratio;
  3. operator subspaces: principal angles between top-k input (right) and output
     (left) singular subspaces of W_avg vs W_ctx at k=48 and k90, vs the batched
     Haar-random null band; matched lambda by summed PRESS (machinery reused
     verbatim from issue1092_partb_operator).

Analysis-only: reads the staged .npy state summaries + manifest; writes
eval_results/issue_1092/inline_operator_coincidence/operator_coincidence.json.
Fit engine reused from issue1092_fit_grid (PRESS-ridge); constants + loaders
from issue1092_inline_fair_comparison; operator machinery from
issue1092_partb_operator.

Usage: uv run python scripts/issue1092_operator_coincidence.py
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
from issue1092_fit_grid import (  # noqa: E402
    RIDGE_LAMBDAS,
    _basis_targets_with_info,
    _fit_cv,
    _folds_from_manifest,
    _r2,
)
from issue1092_partb_operator import (  # noqa: E402
    _angle_null_band,
    _angles_between,
    _fit_press,
    _k90,
    _operator_raw,
    _press_mse,
)

OUT_DIR = PROJECT_ROOT / "eval_results/issue_1092/inline_operator_coincidence"
OUT_PATH = OUT_DIR / "operator_coincidence.json"
BANKED = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
N_NULL_DRAWS = 100
NULL_CHUNK = 16
NULL_MAX_RANK = 384
SEED = 20260722


def _aligned_prefix_folds(
    folds_row: list[np.ndarray], groups: dict[str, np.ndarray], pids: list[str]
) -> list[np.ndarray]:
    """Prefix-level folds derived from the grouped per-row folds.

    Rows of one prefix share a fold (folds are grouped by prefix_id) — asserted —
    so each prefix maps to exactly one row-fold and both fits hold out the same
    prefixes together.
    """
    n_rows = int(sum(f.size for f in folds_row))
    row_fold = np.full(n_rows, -1, dtype=np.int64)
    for fi, f in enumerate(folds_row):
        row_fold[f] = fi
    out: list[list[int]] = [[] for _ in folds_row]
    for j, p in enumerate(pids):
        fold_ids = np.unique(row_fold[groups[p]])
        assert fold_ids.size == 1, f"prefix {p} spans folds {fold_ids}"
        out[int(fold_ids[0])].append(j)
    return [np.asarray(f, dtype=np.int64) for f in out if f]


def process_cell(cell: str, rows: list[dict], banked_cell: dict) -> dict:
    context_all = fc._load(cell, "context_end")
    t_all = [fc._load(cell, t) for t in fc.TARGETS]
    n0 = min(context_all.shape[0], min(t.shape[0] for t in t_all), len(rows))
    be_idx = np.asarray(
        [
            i
            for i in range(n0)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )
    rows_be = [rows[int(i)] for i in be_idx]
    prefix_ids = np.asarray([r.get("prefix_id", "") for r in rows_be])
    X = np.asarray(context_all[be_idx], dtype=np.float64)
    Y_stacked = np.concatenate([np.asarray(t[be_idx], dtype=np.float64) for t in t_all], axis=1)
    del context_all, t_all
    gc.collect()

    folds_row = _folds_from_manifest(
        rows_be, len(rows_be), group_key="prefix_id", n_folds=fc.N_FOLDS
    )
    groups = fc._prefix_groups(prefix_ids, fc.MIN_ROWS_PER_PREFIX)
    pids = sorted(groups)
    prefix_folds = _aligned_prefix_folds(folds_row, groups, pids)
    Xc_avg = np.stack([X[groups[p]].mean(0) for p in pids], axis=0)

    out: dict = {"n_rows": int(len(rows_be)), "n_prefixes": len(pids), "bases": {}}
    for basis in fc.BASES:
        t0 = time.monotonic()
        Yb = _basis_targets_with_info(
            Y_stacked, basis, hidden_dim=fc.HIDDEN_DIM, targets=fc.TARGETS, projection_target="t1"
        )[0]
        Y_avg = np.stack(
            [np.ascontiguousarray(Yb[groups[p]], dtype=np.float64).mean(0) for p in pids], axis=0
        )
        # Leg 1: induced (per-row fit, held-out preds averaged) vs refit (averaged rows).
        out_row, P_row = _fit_cv(
            np.ascontiguousarray(X), np.ascontiguousarray(Yb), folds_row, return_pred=True
        )
        P_ind = np.stack([P_row[groups[p]].mean(0) for p in pids], axis=0)
        del P_row
        out_ref, P_ref = _fit_cv(
            np.ascontiguousarray(Xc_avg),
            np.ascontiguousarray(Y_avg),
            prefix_folds,
            return_pred=True,
        )
        r2_induced = _r2(Y_avg, P_ind)
        banked_induced = banked_cell["bases"][basis]["averaged_grain"]["r2_context_averaged"]
        # Leg 2: prediction agreement + per-prefix wins.
        agree_ind_explains_ref = _r2(P_ref, P_ind)
        agree_ref_explains_ind = _r2(P_ind, P_ref)
        e_ind = np.linalg.norm(Y_avg - P_ind, axis=1)
        e_ref = np.linalg.norm(Y_avg - P_ref, axis=1)
        induced_win_frac = float(np.mean(e_ind < e_ref))
        median_err_ratio_ref_over_ind = float(np.median(e_ref / np.maximum(e_ind, 1e-12)))
        # Leg 3: operators at matched summed-PRESS lambda.
        fit_row = _fit_press(X)
        fit_avg = _fit_press(Xc_avg)
        Yt_row = torch.from_numpy(np.ascontiguousarray(Yb)).double()
        Yt_avg = torch.from_numpy(np.ascontiguousarray(Y_avg)).double()
        mse_r, G_r, _ = _press_mse(fit_row, Yt_row)
        mse_a, G_a, _ = _press_mse(fit_avg, Yt_avg)
        matched_idx = int(torch.argmin(mse_r / mse_r.min() + mse_a / mse_a.min()).item())
        lam = float(RIDGE_LAMBDAS[matched_idx])
        W_r = _operator_raw(fit_row, G_r, lam)
        W_a = _operator_raw(fit_avg, G_a, lam)
        del G_r, G_a, Yt_row, Yt_avg, fit_row, fit_avg
        frob_cos = float(
            (W_a * W_r).sum() / (torch.linalg.norm(W_a) * torch.linalg.norm(W_r) + 1e-12)
        )
        frob_ratio = float(torch.linalg.norm(W_a) / (torch.linalg.norm(W_r) + 1e-12))
        U_r, s_r, Qh_r = torch.linalg.svd(W_r, full_matrices=False)
        U_a, s_a, Qh_a = torch.linalg.svd(W_a, full_matrices=False)
        del W_r, W_a
        d_in = int(Qh_r.shape[1])
        P_out = int(U_r.shape[0])
        r = int(s_r.shape[0])
        k48 = min(48, r)
        k90_r, k90_a = _k90(s_r), _k90(s_a)
        gen = torch.Generator().manual_seed(SEED)
        subspaces = {}
        for name, k1, k2, A, B, d in (
            ("input_k48", k48, k48, Qh_a.T, Qh_r.T, d_in),
            ("output_k48", k48, k48, U_a, U_r, P_out),
            ("input_k90", k90_a, k90_r, Qh_a.T, Qh_r.T, d_in),
            ("output_k90", k90_a, k90_r, U_a, U_r, P_out),
        ):
            angles = _angles_between(A[:, :k1], B[:, :k2])
            subspaces[name] = {
                "k": [int(k1), int(k2)],
                "mean_angle_deg": float(np.degrees(np.mean(angles))),
                "null": _angle_null_band(d, k1, k2, N_NULL_DRAWS, NULL_CHUNK, gen, NULL_MAX_RANK),
            }
        del U_r, U_a, Qh_r, Qh_a
        gc.collect()
        out["bases"][basis] = {
            "r2_induced_avg": float(r2_induced),
            "r2_induced_avg_banked": float(banked_induced),
            "r2_refit_avg": float(out_ref["r2"]),
            "r2_refit_folds": out_ref["r2_folds"],
            "r2_row_context": float(out_row["r2"]),
            "agreement_r2_induced_explains_refit": float(agree_ind_explains_ref),
            "agreement_r2_refit_explains_induced": float(agree_ref_explains_ind),
            "induced_win_frac": induced_win_frac,
            "median_err_ratio_refit_over_induced": median_err_ratio_ref_over_ind,
            "matched_lambda": lam,
            "frobenius_cosine_W_avg_vs_W_ctx": frob_cos,
            "frobenius_norm_ratio_W_avg_over_W_ctx": frob_ratio,
            "subspaces": subspaces,
            "wall_s": round(time.monotonic() - t0, 1),
        }
        b = out["bases"][basis]
        print(
            f"[{cell}/{basis}] induced {b['r2_induced_avg']:.4f} "
            f"(banked {b['r2_induced_avg_banked']:.4f}) refit {b['r2_refit_avg']:.4f} "
            f"agree(ind->ref) {b['agreement_r2_induced_explains_refit']:.3f} "
            f"induced_win {b['induced_win_frac']:.3f} "
            f"out_k48 {b['subspaces']['output_k48']['mean_angle_deg']:.1f}deg "
            f"in_k48 {b['subspaces']['input_k48']['mean_angle_deg']:.1f}deg "
            f"[{b['wall_s']}s]",
            flush=True,
        )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    banked = json.loads(BANKED.read_text())
    rows = fc._jsonl(fc.MANIFEST)
    result = {
        "meta": {
            "script": "scripts/issue1092_operator_coincidence.py",
            "git_commit": fc._git_sha(),
            "layer": fc.LAYER,
            "seed": SEED,
            "n_null_draws": N_NULL_DRAWS,
            "definition": (
                "induced = per-row context PRESS-ridge, held-out preds averaged per prefix; "
                "refit = PRESS-ridge fit on (v_bar_C, averaged profile) rows; folds aligned "
                "(prefix folds derived from grouped per-row folds); operators compared at the "
                "matched summed-normalized-PRESS lambda in raw input coordinates"
            ),
        },
        "cells": {},
    }
    for cell in fc.CELLS:
        result["cells"][cell] = process_cell(cell, rows, banked["cells"][cell])
        gc.collect()
    OUT_PATH.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
