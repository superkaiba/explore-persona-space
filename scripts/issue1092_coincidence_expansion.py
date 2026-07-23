#!/usr/bin/env python3
"""#1092 inline: EXPAND the operator-coincidence result (induced vs refit).

Four legs on the banked staged states (ambient, L14):
  1. noise localization (both cells): per-prefix ||P_refit - P_induced|| and
     error ratio vs rows-per-prefix — small-n noise predicts disagreement
     concentrated on sparse (few-row) prefixes;
  2. data-matched learning curve (instruct): per-row context fits on grouped
     prefix subsamples (~1k/2k/5k/full rows), each scored at the averaged
     grain — locates the refit's R2 on the same-operator-with-less-data curve;
  3. amplitude correction (both cells): oracle global rescale of the refit's
     held-out predictions — splits the refit's gap into shrinkage (amplitude)
     vs direction noise;
  4. same-operator small-n angle calibration (instruct): operator fit on the
     ~1k-row per-row subsample vs the full per-row operator — the reference
     band for the observed refit-vs-induced subspace angles.

Analysis-only; reuses the engines of issue1092_{inline_fair_comparison,
fit_grid, partb_operator}. Writes
eval_results/issue_1092/inline_operator_coincidence/coincidence_expansion.json.

Usage: uv run python scripts/issue1092_coincidence_expansion.py
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
from scipy.stats import spearmanr  # noqa: E402

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
    _angles_between,
    _fit_press,
    _operator_raw,
    _press_mse,
)

OUT_DIR = PROJECT_ROOT / "eval_results/issue_1092/inline_operator_coincidence"
OUT_PATH = OUT_DIR / "coincidence_expansion.json"
SEED = 20260723
SUBSAMPLE_ROW_TARGETS = [996, 2000, 5000]
BASIS = "ambient"


def _prep(cell: str, rows: list[dict]):
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
    Yb = _basis_targets_with_info(
        Y_stacked, BASIS, hidden_dim=fc.HIDDEN_DIM, targets=fc.TARGETS, projection_target="t1"
    )[0]
    del Y_stacked
    gc.collect()
    folds_row = _folds_from_manifest(
        rows_be, len(rows_be), group_key="prefix_id", n_folds=fc.N_FOLDS
    )
    groups = fc._prefix_groups(prefix_ids, fc.MIN_ROWS_PER_PREFIX)
    pids = sorted(groups)
    n_rows_lookup = int(sum(f.size for f in folds_row))
    row_fold = np.full(n_rows_lookup, -1, dtype=np.int64)
    for fi, f in enumerate(folds_row):
        row_fold[f] = fi
    pref_folds: list[list[int]] = [[] for _ in folds_row]
    for j, p in enumerate(pids):
        fold_ids = np.unique(row_fold[groups[p]])
        assert fold_ids.size == 1, f"prefix {p} spans folds {fold_ids}"
        pref_folds[int(fold_ids[0])].append(j)
    prefix_folds = [np.asarray(f, dtype=np.int64) for f in pref_folds if f]
    Xc_avg = np.stack([X[groups[p]].mean(0) for p in pids], axis=0)
    Y_avg = np.stack([Yb[groups[p]].mean(0) for p in pids], axis=0)
    return X, Yb, folds_row, groups, pids, prefix_folds, Xc_avg, Y_avg


def process_cell(cell: str, rows: list[dict], do_curve: bool) -> dict:
    t0 = time.monotonic()
    X, Yb, folds_row, groups, pids, prefix_folds, Xc_avg, Y_avg = _prep(cell, rows)
    out: dict = {}

    # Full per-row fit -> induced predictions.
    out_row, P_row = _fit_cv(
        np.ascontiguousarray(X), np.ascontiguousarray(Yb), folds_row, return_pred=True
    )
    P_ind = np.stack([P_row[groups[p]].mean(0) for p in pids], axis=0)
    del P_row
    out_ref, P_ref = _fit_cv(
        np.ascontiguousarray(Xc_avg), np.ascontiguousarray(Y_avg), prefix_folds, return_pred=True
    )
    r2_ind, r2_ref = _r2(Y_avg, P_ind), float(out_ref["r2"])

    # Leg 1: disagreement vs rows-per-prefix.
    n_rows_per_prefix = np.asarray([groups[p].size for p in pids], dtype=np.float64)
    disagree = np.linalg.norm(P_ref - P_ind, axis=1)
    e_ind = np.linalg.norm(Y_avg - P_ind, axis=1)
    e_ref = np.linalg.norm(Y_avg - P_ref, axis=1)
    rho_d, p_d = spearmanr(n_rows_per_prefix, disagree)
    rho_r, p_r = spearmanr(n_rows_per_prefix, e_ref / np.maximum(e_ind, 1e-12))
    dense = n_rows_per_prefix >= 40
    out["noise_localization"] = {
        "spearman_nrows_vs_disagreement": [float(rho_d), float(p_d)],
        "spearman_nrows_vs_err_ratio": [float(rho_r), float(p_r)],
        "median_disagreement_sparse_le5_rows": float(np.median(disagree[n_rows_per_prefix <= 5])),
        "median_disagreement_dense_ge40_rows": float(np.median(disagree[dense]))
        if dense.any()
        else None,
        "refit_win_frac_dense_ge40": float(np.mean(e_ref[dense] < e_ind[dense]))
        if dense.any()
        else None,
        "n_sparse_le5": int((n_rows_per_prefix <= 5).sum()),
        "n_dense_ge40": int(dense.sum()),
    }

    # Leg 3: oracle amplitude correction of the refit.
    Yc = Y_avg - Y_avg.mean(0)
    Pc = P_ref - P_ref.mean(0)
    a_star = float((Yc * Pc).sum() / (Pc**2).sum())
    P_scaled = P_ref.mean(0) + a_star * Pc
    out["amplitude_correction"] = {
        "oracle_scale": a_star,
        "r2_refit": r2_ref,
        "r2_refit_rescaled": float(_r2(Y_avg, P_scaled)),
        "r2_induced": r2_ind,
    }

    # Legs 2 + 4: learning curve + small-n operator angle calibration.
    if do_curve:
        rng = np.random.default_rng(SEED)
        order = rng.permutation(len(pids))
        curve = []
        W_small = None
        for target_rows in SUBSAMPLE_ROW_TARGETS:
            keep_p, tot = [], 0
            for j in order:
                keep_p.append(j)
                tot += int(n_rows_per_prefix[j])
                if tot >= target_rows:
                    break
            keep_rows = np.concatenate([groups[pids[j]] for j in keep_p])
            # RESTRICT the design to the subsample (fix: the first version passed
            # full X with subsample folds, so every fold trained on ~17k rows and
            # the curve was flat — leg 2 of the banked artifact is invalid).
            Xs = np.ascontiguousarray(X[keep_rows])
            Ybs = np.ascontiguousarray(Yb[keep_rows])
            new_pos = {int(r): i for i, r in enumerate(keep_rows)}
            keep_mask = np.zeros(X.shape[0], dtype=bool)
            keep_mask[keep_rows] = True
            sub_folds = [
                np.asarray([new_pos[int(r)] for r in f[keep_mask[f]]], dtype=np.int64)
                for f in folds_row
            ]
            sub_folds = [f for f in sub_folds if f.size]
            out_sub, P_sub = _fit_cv(Xs, Ybs, sub_folds, return_pred=True)
            # Induced read on the subsample's prefixes only (held-out).
            sub_pids = [pids[j] for j in keep_p]
            sub_groups = {p: np.asarray([new_pos[int(r)] for r in groups[p]]) for p in sub_pids}
            P_sub_ind = np.stack([P_sub[sub_groups[p]].mean(0) for p in sub_pids], axis=0)
            Y_sub_avg = np.stack([Ybs[sub_groups[p]].mean(0) for p in sub_pids], axis=0)
            curve.append(
                {
                    "target_rows": target_rows,
                    "actual_rows": int(tot),
                    "n_prefixes": len(keep_p),
                    "r2_row": float(out_sub["r2"]),
                    "r2_induced_avg": float(_r2(Y_sub_avg, P_sub_ind)),
                }
            )
            if target_rows == SUBSAMPLE_ROW_TARGETS[0]:
                fit_small = _fit_press(X[keep_rows])
                Yt_small = torch.from_numpy(np.ascontiguousarray(Yb[keep_rows])).double()
                mse_s, G_s, _ = _press_mse(fit_small, Yt_small)
                lam_s = float(RIDGE_LAMBDAS[int(torch.argmin(mse_s).item())])
                W_small = _operator_raw(fit_small, G_s, lam_s)
                del fit_small, Yt_small, G_s
            del P_sub
            gc.collect()
        curve.append(
            {
                "target_rows": X.shape[0],
                "actual_rows": int(X.shape[0]),
                "n_prefixes": len(pids),
                "r2_row": float(out_row["r2"]),
                "r2_induced_avg": r2_ind,
            }
        )
        out["learning_curve_rows_vs_induced_r2"] = curve

        # Leg 4: same-operator small-n angle reference.
        fit_full = _fit_press(X)
        Yt_full = torch.from_numpy(np.ascontiguousarray(Yb)).double()
        mse_f, G_f, _ = _press_mse(fit_full, Yt_full)
        lam_f = float(RIDGE_LAMBDAS[int(torch.argmin(mse_f).item())])
        W_full = _operator_raw(fit_full, G_f, lam_f)
        del fit_full, Yt_full, G_f
        U_f, s_f, Qh_f = torch.linalg.svd(W_full, full_matrices=False)
        U_s, s_s, Qh_s = torch.linalg.svd(W_small, full_matrices=False)
        del W_full, W_small
        k = 48
        out["small_n_angle_reference"] = {
            "definition": (
                "per-row operator fit on the ~1k-row grouped subsample vs the full per-row "
                "operator — the SAME operator estimated at refit-scale n; reference for the "
                "observed refit-vs-induced angles"
            ),
            "input_k48_deg": float(
                np.degrees(np.mean(_angles_between(Qh_s.T[:, :k], Qh_f.T[:, :k])))
            ),
            "output_k48_deg": float(np.degrees(np.mean(_angles_between(U_s[:, :k], U_f[:, :k])))),
        }
        del U_f, U_s, Qh_f, Qh_s
        gc.collect()

    out["wall_s"] = round(time.monotonic() - t0, 1)
    print(
        f"[{cell}] {json.dumps({k: v for k, v in out.items() if k != 'wall_s'})[:600]}", flush=True
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = fc._jsonl(fc.MANIFEST)
    result = {
        "meta": {
            "script": "scripts/issue1092_coincidence_expansion.py",
            "git_commit": fc._git_sha(),
            "layer": fc.LAYER,
            "basis": BASIS,
            "seed": SEED,
        },
        "cells": {},
    }
    for cell, do_curve in (("cell_inst_own", True), ("cell_pre_own", False)):
        result["cells"][cell] = process_cell(cell, rows, do_curve)
        gc.collect()
    OUT_PATH.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
