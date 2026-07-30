"""Issue #1689 follow-up `derived-vs-free-answer-map` — items 7-8 (+ item 9 structural leg).

Item 7 — context-side map structural characterization (M only; x_S -> x_T, no
readout, no conjugation gauge ambiguity). Per ordered pair x arm:
  (a) nested class ladder judged by predicting x_T from x_S on held-out rows:
      translation -> translation+scalar -> translation+rotation (Procrustes)
      -> identity + rank-k correction (k in the registered grid; first pass =
      SVD truncation of fitted M - I, reduced-rank-ridge upgrade where
      truncation underperforms full affine by more than the null band width)
      -> full affine. Pooled held-out R2 + kNN per class (shared conv-grouped
      folds, seed 42); weakest sufficient class (>= 0.9 x full-affine R2) with
      the class selection RIDING PER matched-capacity null draw (40 draws,
      band = p97.5 of class-reached; per-draw x per-class R2 matrix persisted).
  (b) distance-from-identity: ||M-I||_F/||M||_F, full-affine-over-identity+bias
      gain, M-I spectrum + effective rank, polar-factor distance.
  (d) cosine of top correction directions to the diff-of-means direction.
Item 7(c) — cross-pair principal-angle overlap of M-I top-k subspaces
  (k=32 primary / 8 sensitivity) vs a matched-(k,d) random-subspace null
  (200 draws): --phase overlap.
Item 8 — rank-k reconciliation rung for pairs whose parent rung_reached >= 7,
  in the parent's readout-reconciliation form (corrections exposed by
  issue1689_fit_ladder.fit_rung78_corrections_t; parent all-rows W_s + fold-0
  split conventions preserved): minimal sufficient k per side (context /
  answer) with the same 0.9 x R2_within(T) criterion + selection-riding null.

Phases (--phase): units | overlap | merge. Per-unit JSON checkpoints +
regime-keyed resume; compact bundles (per-draw matrices, M-I factors) under
<out-root>/bundles.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# load_dotenv() BEFORE numpy/torch (shared-VM thread caps, #847).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.issue1689_fit_ladder as fl  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    knn_retrieval,
)
from scripts.issue1689_common import (  # noqa: E402
    HEADLINE_LAYER,
    LAMBDA_GRIDS,
    N_FOLDS,
    PCA_K_CAP,
    RUNG_REACHED_THRESHOLD,
)
from scripts.issue1689_derived_vs_free import (  # noqa: E402
    MODEL_SLUGS,
    _atomic_savez,
    _atomic_write_json,
    _CellCache,
    _heldout_captured_frac,
    _metadata,
    _pca_basis,
    assert_distinct_unit_keys,
    build_pair_specs,
    compute_k_unit,
    fit_basis_of,
    k_floor_limited,
    unit_identity_mismatch,
    unit_key,
)

RANK_GRID = (1, 2, 4, 8, 16, 32, 64, 128)  # scope items 7a/8
SUBSPACE_K_PRIMARY = 32
SUBSPACE_K_SENSITIVITY = 8
SUBSPACE_NULL_DRAWS = 200
STRUCT_VERSION = "cms-v1"


def regime_meta(args) -> dict:
    """Per-unit resume regime key (every output-affecting knob, #722 r3).

    ``fit_basis`` enters ONLY when non-ambient (alias-the-default: parent
    checkpoints stay resume-valid; a reduced run never matches an ambient
    checkpoint — see issue1689_derived_vs_free.regime_meta).
    """
    meta = {
        "struct_version": STRUCT_VERSION,
        "layer": int(args.layer),
        "lambda_grid": str(args.lambda_grid),
        "seed": int(args.seed),
        "n_folds": int(N_FOLDS),
        "rank_grid": list(RANK_GRID),
        "threshold": float(RUNG_REACHED_THRESHOLD),
        "class_null_draws": int(args.class_null_draws),
        "rank_null_draws": int(args.rank_null_draws),
        "items": str(args.items),
        "row_limit": args.row_limit,
        "dim_limit": args.dim_limit,
    }
    if fit_basis_of(args) != "ambient":
        meta["fit_basis"] = fit_basis_of(args)
        meta["pca_k_cap"] = int(PCA_K_CAP)
    return meta


CLASS_ORDER = (
    ["translation", "trans_scalar", "trans_rotation"]
    + [f"rank_{k}" for k in RANK_GRID]
    + ["full_affine"]
)


def _class_preds_for_split(
    tX_S, tX_T, tr, te, conv_tr, lams, device, fit_cache=None
) -> tuple[dict, dict]:
    """Fit all item-7a classes on tr, predict on te. Returns (preds, ops).

    preds: {class: torch (n_te, d)}; ops: canonical operators for 7b/7d
    (M, a_M, C=M-I SVD factors) — all train-fold-only. ``fit_cache`` (a
    fl.build_inner_cv_cache_t over tX_S[tr]) reuses the X-side eighs across
    target-varying null draws — same math, shared factorization (#823 class;
    bit-equivalence pinned in tests)."""
    import torch

    X_tr, X_te = tX_S[tr], tX_S[te]
    T_tr = tX_T[tr]
    mu_s = X_tr.mean(dim=0)
    mu_t = T_tr.mean(dim=0)
    Xc_tr = X_tr - mu_s
    Tc_tr = T_tr - mu_t
    preds: dict = {}
    # translation (== identity+learned-bias on x; the class-ladder floor)
    preds["translation"] = X_te + (mu_t - mu_s)
    # translation + global scalar
    c_num = float((Xc_tr * Tc_tr).sum().item())
    c_den = float((Xc_tr**2).sum().item()) + 1e-12
    c = c_num / c_den
    preds["trans_scalar"] = c * (X_te - mu_s) + mu_t
    # translation + orthogonal rotation (Procrustes, closed-form SVD)
    Ur, _sr, Vhr = fl._svd_robust_t(Xc_tr.T @ Tc_tr)
    R = Ur @ Vhr
    preds["trans_rotation"] = (X_te - mu_s) @ R + mu_t
    # full affine (the item-1 estimator: inner-group-cv ridge)
    if fit_cache is not None:
        M, a_M, lam_m = fl.fit_inner_group_cv_cached_t(fit_cache, T_tr, lams)
    else:
        M, a_M, lam_m = fl._fit_ridge_inner_group_cv_t(X_tr, T_tr, conv_tr, lams)
    preds["full_affine"] = X_te @ M + a_M
    # identity + rank-k correction: SVD truncation of C = M - I (first pass)
    d = M.shape[0]
    C = M - torch.eye(d, dtype=torch.float64, device=M.device)
    Uc, sc, Vhc = fl._svd_robust_t(C)
    for k in RANK_GRID:
        kk = min(k, d)
        xc_te = (X_te @ Uc[:, :kk]) * sc[:kk]
        corr_te = xc_te @ Vhc[:kk]
        corr_tr = ((X_tr @ Uc[:, :kk]) * sc[:kk]) @ Vhc[:kk]
        a_k = (T_tr - X_tr - corr_tr).mean(dim=0)
        preds[f"rank_{k}"] = X_te + corr_te + a_k
    ops = {"M": M, "a_M": a_M, "C_U": Uc, "C_s": sc, "C_Vh": Vhc, "R": R, "lam_m": lam_m}
    return preds, ops


def _rrr_project(C, Xc_tr, k: int):
    """Reduced-rank ridge upgrade: project the ridge-fitted correction C onto
    the top-k response directions of Xc_tr @ C (Reinsel-Velu RRR with the
    ridge base estimator)."""

    F = Xc_tr @ C
    G = F.T @ F
    G = 0.5 * (G + G.T)
    w, V = fl._eigh_robust_t(G)
    Vk = V[:, -k:]  # eigh ascending -> top-k are the LAST columns
    return C @ (Vk @ Vk.T), Vk


def _class_reached(class_r2s: dict, r2_full: float, threshold: float) -> str:
    bar = threshold * r2_full if r2_full > 0 else float("-inf")
    for name in CLASS_ORDER:
        if class_r2s.get(name, float("-inf")) >= bar:
            return name
    return "full_affine"


def _trunc_apply(X, C_U, C_s, C_Vh, k: int):
    return ((X @ C_U[:, :k]) * C_s[:k]) @ C_Vh[:k]


def run_structure_unit(source, target, spec, arm, args, lams, parent_rung) -> tuple[dict, dict]:
    """Items 7a/7b/7d (+ item 8 when parent_rung >= 7) for one (pair, arm)."""
    import torch

    (sm, sc_), (tm, tc_) = spec
    t0 = time.perf_counter()
    common, s_idx, t_idx = fl.pair_rows_by_conv(source["conv_ids"], target["conv_ids"])
    if args.row_limit is not None:
        common, s_idx, t_idx = (a[: args.row_limit] for a in (common, s_idx, t_idx))
    n = len(common)
    if n < 3:
        return {"error": "insufficient shared conv_ids", "retryable": False, "n_common": int(n)}, {}
    dsl = slice(None) if args.dim_limit is None else slice(0, args.dim_limit)
    X_S = source[f"X_{arm}"][s_idx][:, dsl]
    X_T = target[f"X_{arm}"][t_idx][:, dsl]
    d = X_S.shape[1]
    folds = fl._conv_grouped_folds(common, n_folds=N_FOLDS, seed=args.seed)
    dev = torch.device(args.device)
    tX_S = torch.from_numpy(np.ascontiguousarray(X_S)).to(dev)
    tX_T = torch.from_numpy(np.ascontiguousarray(X_T)).to(dev)

    # --- Fit basis (plan v10 s4): ambient (parent, byte-identical) or the
    # per-(pair, arm, fold) shared train-fold-only PCA rank-k basis (one
    # x-side basis — the cms battery maps x_S -> x_T, no y side).
    reduced = fit_basis_of(args) == "reduced"
    k_unit = 0
    per_fold_n_train: dict[int, int] = {}
    basis_meta: dict[int, dict] = {}
    basis_spectra: dict[int, np.ndarray] = {}
    if reduced:
        k_unit, per_fold_n_train = compute_k_unit(folds, n, d)
        if per_fold_n_train and k_unit < 1:
            return {
                "error": f"k_unit < 1 (min-fold n_train {min(per_fold_n_train.values())})",
                "retryable": False,
                "n_common": int(n),
            }, {}
    fdim = k_unit if reduced else d

    pooled_pred: dict[str, list[np.ndarray]] = {c: [] for c in CLASS_ORDER}
    pooled_true: list[np.ndarray] = []
    skipped_folds: list[int] = []
    canonical: dict = {}
    lam_by_fold: dict[int, float] = {}
    for k_fold in range(N_FOLDS):
        te_mask = folds == k_fold
        tr_mask = ~te_mask
        if tr_mask.sum() < 3 or te_mask.sum() < 1:
            skipped_folds.append(k_fold)
            continue
        tr = torch.from_numpy(np.where(tr_mask)[0]).to(dev)
        te = torch.from_numpy(np.where(te_mask)[0]).to(dev)
        conv_tr = common[np.where(tr_mask)[0]]
        if reduced:
            n_tr = int(tr.numel())
            # Well-posedness invariant (plan s4 item 1) [wellposed-assert].
            assert n_tr >= 2 * k_unit, (
                f"well-posedness violated: fold {k_fold} n_train={n_tr} < 2*k_unit={2 * k_unit}"
            )
            mu_x, Q_x, sx_k, fx_tr = _pca_basis(torch.cat([tX_S[tr], tX_T[tr]], dim=0), k_unit)
            pX_S = (tX_S - mu_x) @ Q_x
            pX_T = (tX_T - mu_x) @ Q_x
            basis_spectra[k_fold] = sx_k.cpu().numpy()
            basis_meta[k_fold] = {
                "n_train": n_tr,
                "captured_var_train_x": fx_tr,
                "captured_var_test_x": _heldout_captured_frac(
                    torch.cat([tX_S[te], tX_T[te]], dim=0), mu_x, Q_x
                ),
            }
        else:
            pX_S, pX_T = tX_S, tX_T
        preds, ops = _class_preds_for_split(pX_S, pX_T, tr, te, conv_tr, lams, dev)
        lam_by_fold[k_fold] = ops["lam_m"]
        pooled_true.append(pX_T[te].cpu().numpy())
        for cname in CLASS_ORDER:
            pooled_pred[cname].append(preds[cname].cpu().numpy())
        if not canonical:
            canonical = {
                "fold": k_fold,
                "tr": tr,
                "te": te,
                "conv_tr": conv_tr,
                "tr_np": np.where(tr_mask)[0],
                "te_np": np.where(te_mask)[0],
                "pX_S": pX_S,
                "pX_T": pX_T,
                **({"Q_x": Q_x, "mu_x": mu_x} if reduced else {}),
                **ops,
            }
    if not pooled_true:
        return {
            "error": "all folds degenerate",
            "retryable": False,
            "n_common": int(n),
            "skipped_folds": skipped_folds,
        }, {}

    true_arr = np.concatenate(pooled_true, axis=0)
    class_r2: dict[str, float] = {}
    class_knn: dict[str, dict] = {}
    for cname in CLASS_ORDER:
        pred_arr = np.concatenate(pooled_pred[cname], axis=0)
        class_r2[cname] = fl._r2(true_arr, pred_arr)
        class_knn[cname] = {
            metric: knn_retrieval(pred_arr, true_arr, ks=(1, 5, 10), metric=metric)
            for metric in ("euclidean", "cosine")
        }
    # sanity: translation class == the canonical identity+bias helper (pinned in tests)
    r2_full = class_r2["full_affine"]
    weakest_point = _class_reached(class_r2, r2_full, RUNG_REACHED_THRESHOLD)

    # --- matched-capacity null: class-reached per draw on the canonical split
    # (per-draw same-selection; the parent ladder's null convention: permute
    # the TARGET rows to break pairing at matched estimator capacity).
    tr, te = canonical["tr"], canonical["te"]
    conv_tr = canonical["conv_tr"]
    # Canonical-fold tensors in the FIT basis (== the raw tensors when ambient;
    # the fold's projected tensors when reduced — basis computed ONCE per fold
    # and REUSED across the null draws, plan v10 s10 row (i)).
    cX_S, cX_T = canonical["pX_S"], canonical["pX_T"]
    rng = np.random.default_rng(args.seed + 1)
    null_matrix = np.full((args.class_null_draws, len(CLASS_ORDER)), np.nan)
    null_reached: list[str] = []
    # X-side eighs are draw-invariant (only the target permutes): share them.
    null_fit_cache = (
        fl.build_inner_cv_cache_t(cX_S[tr], conv_tr) if args.class_null_draws > 0 else None
    )
    for draw_i in range(args.class_null_draws):
        perm = rng.permutation(n)
        t_perm = torch.from_numpy(perm).to(dev)
        X_T_null = cX_T[t_perm]
        try:
            preds_d, _ops_d = _class_preds_for_split(
                cX_S, X_T_null, tr, te, conv_tr, lams, dev, fit_cache=null_fit_cache
            )
        except torch.linalg.LinAlgError:
            null_reached.append("degenerate")
            continue
        true_d = X_T_null[te]
        r2s_d = {c: fl._r2_t(true_d, preds_d[c]) for c in CLASS_ORDER}
        null_matrix[draw_i] = [r2s_d[c] for c in CLASS_ORDER]
        null_reached.append(_class_reached(r2s_d, r2s_d["full_affine"], RUNG_REACHED_THRESHOLD))
    class_idx = {c: i + 1 for i, c in enumerate(CLASS_ORDER)}
    reached_idx = [class_idx.get(r, len(CLASS_ORDER)) for r in null_reached]
    null_band_p975 = float(np.percentile(reached_idx, 97.5)) if reached_idx else float("nan")
    # Null band WIDTH on full-affine R2 (the RRR upgrade trigger scale).
    full_col = null_matrix[:, CLASS_ORDER.index("full_affine")]
    full_col = full_col[np.isfinite(full_col)]
    null_r2_width = float(np.ptp(full_col)) if full_col.size else float("nan")

    # --- RRR upgrade (plan 7a trigger: truncation underperforms full affine by
    # more than the null band width at the LARGEST k).
    kmax = f"rank_{RANK_GRID[-1]}"
    rrr_triggered = bool(np.isfinite(null_r2_width) and (r2_full - class_r2[kmax]) > null_r2_width)
    rrr_r2: dict[str, float] = {}
    if rrr_triggered:
        import torch as _t

        X_tr = cX_S[tr]
        Xc_tr = X_tr - X_tr.mean(dim=0)
        C = canonical["M"] - _t.eye(fdim, dtype=_t.float64, device=dev)
        T_tr = cX_T[tr]
        X_te = cX_S[te]
        true_can = cX_T[te]
        for k in RANK_GRID:
            Ck, _vk = _rrr_project(C, Xc_tr, min(k, fdim))
            a_k = (T_tr - X_tr - X_tr @ Ck).mean(dim=0)
            pred = X_te + X_te @ Ck + a_k
            rrr_r2[f"rank_{k}"] = fl._r2_t(true_can, pred)

    # --- 7b / 7d on the canonical fold's M.
    M = canonical["M"]
    C_s = canonical["C_s"]
    m_fro = float(M.norm().item())
    c_fro = float(C_s.norm().item())  # ||M-I||_F == ||svals||_2
    eff_rank_c = float((C_s.sum() ** 2 / (C_s**2).sum()).item())
    Um, sm_v, Vhm = fl._svd_robust_t(M)
    Q_polar = Um @ Vhm
    polar_dist = float((M - Q_polar).norm().item()) / (m_fro + 1e-12)
    tr_np, te_np = canonical["tr_np"], canonical["te_np"]
    if reduced:
        # Diff-of-means in the fold's reduced coordinates (all 7b/7d reads live
        # in one coherent basis; ambient keeps the parent numpy path verbatim).
        delta = (cX_T[tr].mean(dim=0) - cX_S[tr].mean(dim=0)).cpu().numpy()
    else:
        delta = X_T[tr_np].mean(axis=0) - X_S[tr_np].mean(axis=0)
    dn = np.linalg.norm(delta) + 1e-12
    C_U = canonical["C_U"].cpu().numpy()
    C_Vh = canonical["C_Vh"].cpu().numpy()
    top8_out_cos = [
        float(abs(C_Vh[i] @ delta) / (np.linalg.norm(C_Vh[i]) * dn)) for i in range(min(8, fdim))
    ]
    top8_in_cos = [
        float(abs(C_U[:, i] @ delta) / (np.linalg.norm(C_U[:, i]) * dn))
        for i in range(min(8, fdim))
    ]

    # --- item 8: rank-k reconciliation rung (parent rung >= 7 only).
    rank_rung: dict = {"eligible": bool(parent_rung is not None and parent_rung >= 7)}
    rank_rung["parent_rung_reached"] = parent_rung
    if args.items in ("rank", "both") and rank_rung["eligible"]:
        rank_rung.update(
            _rank_rung_unit(
                source, target, spec, arm, args, lams, k_unit=k_unit if reduced else None
            )
        )

    n_keep = min(256, fdim)
    if reduced:
        # Ambient-LIFT the M-I frames (Q_x @ U_red / Vh_red @ Q_x^T) so the
        # cross-pair overlap phase compares subspaces of ONE common ambient
        # space (per-unit reduced coordinates are mutually incomparable).
        u_frame = (canonical["Q_x"] @ canonical["C_U"][:, :n_keep]).cpu().numpy()
        vh_frame = (canonical["C_Vh"][:n_keep] @ canonical["Q_x"].T).cpu().numpy()
    else:
        u_frame = canonical["C_U"][:, :n_keep].cpu().numpy()
        vh_frame = canonical["C_Vh"][:n_keep].cpu().numpy()
    bundle = {
        "class_null_r2_matrix": null_matrix,
        "class_order": np.array(CLASS_ORDER),
        "null_reached_idx": np.array(reached_idx, dtype=np.int64),
        "m_minus_i_u256_fp16": u_frame.astype(np.float16),
        "m_minus_i_vh256_fp16": vh_frame.astype(np.float16),
        "m_minus_i_svals": C_s.cpu().numpy(),
        "m_svals": sm_v.cpu().numpy(),
        "canonical_fold": np.int64(canonical["fold"]),
    }
    if reduced:
        for f_id, sx_np in basis_spectra.items():
            bundle[f"q_x_svals_f{f_id}"] = sx_np
    if "rank_null_matrix_ctx" in rank_rung:
        bundle["rank_null_matrix_ctx"] = np.asarray(rank_rung.pop("rank_null_matrix_ctx"))
        bundle["rank_null_matrix_ans"] = np.asarray(rank_rung.pop("rank_null_matrix_ans"))

    unit = {
        "meta": regime_meta(args),
        "src_model": sm,
        "src_cond": sc_,
        "tgt_model": tm,
        "tgt_cond": tc_,
        "arm": arm,
        "pair_key": fl.pair_spec_key(spec),
        "unit_key": unit_key(spec, arm),
        "cross_model": sm != tm,
        "n_common": int(n),
        "d": int(d),
        "skipped_folds": skipped_folds,
        "lambda_m_by_fold": lam_by_fold,
        "class_r2_pooled": class_r2,
        "class_knn": class_knn,
        "weakest_class_point": weakest_point,
        "class_null": {
            "n_draws": int(args.class_null_draws),
            "reached_per_draw": null_reached,
            "reached_idx_p975": null_band_p975,
            "full_affine_r2_null_width": null_r2_width,
        },
        "rrr_upgrade": {"triggered": rrr_triggered, "r2": rrr_r2},
        "distance_from_identity": {
            "fro_ratio_m_minus_i_over_m": c_fro / (m_fro + 1e-12),
            "eff_rank_m_minus_i": eff_rank_c,
            "polar_factor_distance": polar_dist,
            "gain_full_over_translation_r2": r2_full - class_r2["translation"],
            "gain_full_over_translation_acc1_euclid": (
                class_knn["full_affine"]["euclidean"]["acc_at_k"][1]
                - class_knn["translation"]["euclidean"]["acc_at_k"][1]
            ),
        },
        "diff_of_means_alignment": {
            "top8_output_dir_abs_cos": top8_out_cos,
            "top8_input_dir_abs_cos": top8_in_cos,
        },
        "rank_rung": rank_rung,
        "wall_s": round(time.perf_counter() - t0, 2),
        "metadata": _metadata(),
    }
    if reduced:
        # Reduced-only fields, keyed ADDITIVELY (ambient JSON byte-identical).
        unit["fit_basis"] = "reduced"
        unit["k_unit"] = int(k_unit)
        unit["fit_dim"] = int(fdim)
        unit["k_floor_limited"] = k_floor_limited(k_unit)
        unit["per_fold_n_train"] = {int(f): int(v) for f, v in per_fold_n_train.items()}
        unit["pca_basis_per_fold"] = {int(f): v for f, v in basis_meta.items()}
    return unit, bundle


def _rank_rung_unit(source, target, spec, arm, args, lams, *, k_unit: int | None = None) -> dict:
    """Item 8: rank-k reconciliation in the parent's readout-reconciliation form.

    With ``k_unit`` set (--fit-basis reduced) the parent-convention refit runs
    INSIDE a shared train-fold-only PCA basis: rows are paired + sliced exactly
    as fit_rung78_corrections_t would, Q_x/Q_y are computed on the fold-0
    split's TRAIN rows only (folds != 0 — the ladder's point-split convention;
    its n_train >= min-fold n_train >= 2*k_unit, so well-posedness holds), and
    fit_rung78_corrections_t receives PRE-PAIRED projected dicts — its own
    pair_rows_by_conv over identical conv_ids is then the identity mapping
    with the same fold assignment (same common, same seed). Ambient calls are
    byte-identical to the parent (raw dicts straight through).
    """
    import torch

    src_dict, tgt_dict = source, target
    dim_limit = args.dim_limit
    if k_unit is not None:
        common, s_idx, t_idx = fl.pair_rows_by_conv(source["conv_ids"], target["conv_ids"])
        if args.row_limit is not None:
            common, s_idx, t_idx = (a[: args.row_limit] for a in (common, s_idx, t_idx))
        if len(common) < 3:
            return {"error": "insufficient shared conv_ids", "n_common": int(len(common))}
        dsl = slice(None) if args.dim_limit is None else slice(0, args.dim_limit)
        dev0 = torch.device(args.device)
        X_S8 = torch.from_numpy(np.ascontiguousarray(source[f"X_{arm}"][s_idx][:, dsl])).to(dev0)
        Y_S8 = torch.from_numpy(np.ascontiguousarray(source["Y"][s_idx][:, dsl])).to(dev0)
        X_T8 = torch.from_numpy(np.ascontiguousarray(target[f"X_{arm}"][t_idx][:, dsl])).to(dev0)
        Y_T8 = torch.from_numpy(np.ascontiguousarray(target["Y"][t_idx][:, dsl])).to(dev0)
        folds8 = fl._conv_grouped_folds(common, n_folds=N_FOLDS, seed=args.seed)
        tr_np = np.where(folds8 != 0)[0]
        if len(tr_np) < 3:
            return {"error": "degenerate fold-0 split for reduced item-8 basis"}
        # Well-posedness invariant (plan s4 item 1) [wellposed-assert].
        assert len(tr_np) >= 2 * k_unit, (
            f"well-posedness violated: item-8 fold-0 split n_train={len(tr_np)} "
            f"< 2*k_unit={2 * k_unit}"
        )
        tr0 = torch.from_numpy(tr_np).to(dev0)
        mu_x, Q_x, _sx, _fx = _pca_basis(torch.cat([X_S8[tr0], X_T8[tr0]], dim=0), k_unit)
        mu_y, Q_y, _sy, _fy = _pca_basis(torch.cat([Y_S8[tr0], Y_T8[tr0]], dim=0), k_unit)
        src_dict = {
            f"X_{arm}": ((X_S8 - mu_x) @ Q_x).cpu().numpy(),
            "Y": ((Y_S8 - mu_y) @ Q_y).cpu().numpy(),
            "conv_ids": common,
        }
        tgt_dict = {
            f"X_{arm}": ((X_T8 - mu_x) @ Q_x).cpu().numpy(),
            "Y": ((Y_T8 - mu_y) @ Q_y).cpu().numpy(),
            "conv_ids": common,
        }
        dim_limit = None  # rows already sliced + projected; k dims must not be re-sliced

    corr = fl.fit_rung78_corrections_t(
        src_dict,
        tgt_dict,
        arm=arm,
        seed=args.seed,
        device=args.device,
        lambdas=lams,
        row_limit=args.row_limit,
        dim_limit=dim_limit,
    )
    if "error" in corr:
        return {"error": corr["error"]}
    dev = torch.device(args.device)
    d = corr["W_s"].shape[0]
    W_s = torch.from_numpy(corr["W_s"]).to(dev)
    b_s = torch.from_numpy(corr["b_s"]).to(dev)
    tX_T = torch.from_numpy(corr["X_T"]).to(dev)
    tY_T = torch.from_numpy(corr["Y_T"]).to(dev)
    tY_S = torch.from_numpy(corr["Y_S"]).to(dev)
    tX_S = torch.from_numpy(corr["X_S"]).to(dev)
    tr = torch.from_numpy(corr["train_idx"]).to(dev)
    te = torch.from_numpy(corr["test_idx"]).to(dev)
    conv_tr = corr["common"][corr["train_idx"]]
    eye = torch.eye(d, dtype=torch.float64, device=dev)

    def _side_r2s(A_W, A_b, side: str) -> dict[int, float]:
        """R2 per k for one side's truncated correction (rung-7/8 recentering)."""
        C = A_W - eye
        Uc, sc, Vhc = fl._svd_robust_t(C)
        out: dict[int, float] = {}
        for k in RANK_GRID:
            kk = min(k, d)
            if side == "ctx":
                xh_te = tX_T[te] + _trunc_apply(tX_T[te], Uc, sc, Vhc, kk) + A_b
                xh_tr = tX_T[tr] + _trunc_apply(tX_T[tr], Uc, sc, Vhc, kk) + A_b
                raw_te = xh_te @ W_s + b_s
                raw_tr = xh_tr @ W_s + b_s
            else:  # answer-side analogue
                ys_te = tX_T[te] @ W_s + b_s
                ys_tr = tX_T[tr] @ W_s + b_s
                raw_te = ys_te + _trunc_apply(ys_te, Uc, sc, Vhc, kk) + A_b
                raw_tr = ys_tr + _trunc_apply(ys_tr, Uc, sc, Vhc, kk) + A_b
            pred = raw_te - raw_tr.mean(dim=0) + tY_T[tr].mean(dim=0)
            out[k] = fl._r2_t(tY_T[te], pred)
        return out

    r2_within, reach_bar = fl._reach_bar_within_cell_t(
        tX_T, tY_T, tr, te, conv_tr, np.asarray(lams), threshold=RUNG_REACHED_THRESHOLD
    )
    ctx_r2s = _side_r2s(
        torch.from_numpy(corr["A_W"]).to(dev), torch.from_numpy(corr["A_b"]).to(dev), "ctx"
    )
    ans_r2s = _side_r2s(
        torch.from_numpy(corr["B_W"]).to(dev), torch.from_numpy(corr["B_b"]).to(dev), "ans"
    )

    def _k_reached(r2s: dict[int, float]) -> int | None:
        for k in RANK_GRID:
            if r2s[k] >= reach_bar:
                return k
        return None

    # Selection-riding null (parent matched-capacity convention: permute the
    # ANSWER rows, refit the y-dependent operators per draw, k-reached per draw).
    rng = np.random.default_rng(args.seed + 2)
    n = corr["n_common"]
    null_k_ctx: list[int] = []
    null_k_ans: list[int] = []
    null_mat_ctx = np.full((args.rank_null_draws, len(RANK_GRID)), np.nan)
    null_mat_ans = np.full((args.rank_null_draws, len(RANK_GRID)), np.nan)
    # X-side eighs are draw-invariant for the W_s fit (X_S fixed) and the
    # reach fit (X_T fixed); only the permuted-Y B fit needs a fresh eigh.
    cache_ws = fl.build_inner_cv_cache_t(tX_S, corr["common"]) if args.rank_null_draws > 0 else None
    cache_reach = fl.build_inner_cv_cache_t(tX_T[tr], conv_tr) if args.rank_null_draws > 0 else None
    # A (rung-7 ctx correction) is x-only — fixed across draws: SVD it ONCE.
    A_W_null = torch.from_numpy(corr["A_W"]).to(dev)
    A_b_null = torch.from_numpy(corr["A_b"]).to(dev)
    UcA, scA, VhcA = fl._svd_robust_t(A_W_null - eye)
    for draw_i in range(args.rank_null_draws):
        perm_s = rng.permutation(n)
        perm_t = rng.permutation(n)
        tps = torch.from_numpy(perm_s).to(dev)
        tpt = torch.from_numpy(perm_t).to(dev)
        Y_S_d, Y_T_d = tY_S[tps], tY_T[tpt]
        try:
            W_s_d, b_s_d, _ = fl.fit_inner_group_cv_cached_t(cache_ws, Y_S_d, np.asarray(lams))
            B_W_d, B_b_d, _ = fl._fit_ridge_inner_group_cv_t(
                Y_S_d[tr], Y_T_d[tr], conv_tr, np.asarray(lams)
            )
        except torch.linalg.LinAlgError:
            continue
        # reach bar via the cached X_T[tr] prep (same math as _reach_bar_within_cell_t)
        W_ref, b_ref, _lam_ref = fl.fit_inner_group_cv_cached_t(
            cache_reach, Y_T_d[tr], np.asarray(lams)
        )
        pred_ref = tX_T[te] @ W_ref + b_ref
        rw_d = fl._r2_t(Y_T_d[te], pred_ref)
        bar_d = RUNG_REACHED_THRESHOLD * rw_d if rw_d > 0 else float("-inf")
        # ctx side: A is x-only (unchanged per draw — hoisted SVD above).
        kc = None
        for j, k in enumerate(RANK_GRID):
            kk = min(k, d)
            xh_te = tX_T[te] + _trunc_apply(tX_T[te], UcA, scA, VhcA, kk) + A_b_null
            xh_tr = tX_T[tr] + _trunc_apply(tX_T[tr], UcA, scA, VhcA, kk) + A_b_null
            pred = (
                (xh_te @ W_s_d + b_s_d)
                - (xh_tr @ W_s_d + b_s_d).mean(dim=0)
                + Y_T_d[tr].mean(dim=0)
            )
            r2k = fl._r2_t(Y_T_d[te], pred)
            null_mat_ctx[draw_i, j] = r2k
            if kc is None and r2k >= bar_d:
                kc = k
        Cb = B_W_d - eye
        Ub, sb, Vhb = fl._svd_robust_t(Cb)
        ka = None
        for j, k in enumerate(RANK_GRID):
            kk = min(k, d)
            ys_te = tX_T[te] @ W_s_d + b_s_d
            ys_tr = tX_T[tr] @ W_s_d + b_s_d
            raw_te = ys_te + _trunc_apply(ys_te, Ub, sb, Vhb, kk) + B_b_d
            raw_tr = ys_tr + _trunc_apply(ys_tr, Ub, sb, Vhb, kk) + B_b_d
            pred = raw_te - raw_tr.mean(dim=0) + Y_T_d[tr].mean(dim=0)
            r2k = fl._r2_t(Y_T_d[te], pred)
            null_mat_ans[draw_i, j] = r2k
            if ka is None and r2k >= bar_d:
                ka = k
        null_k_ctx.append(kc if kc is not None else RANK_GRID[-1] * 2)
        null_k_ans.append(ka if ka is not None else RANK_GRID[-1] * 2)

    return {
        "r2_within_target": float(r2_within),
        "reach_bar": float(reach_bar),
        "lambdas_chosen": corr["lambdas_chosen"],
        "ctx_r2_by_k": {int(k): v for k, v in ctx_r2s.items()},
        "ans_r2_by_k": {int(k): v for k, v in ans_r2s.items()},
        "k_reached_ctx": _k_reached(ctx_r2s),
        "k_reached_ans": _k_reached(ans_r2s),
        "null": {
            "n_draws": int(args.rank_null_draws),
            "k_reached_ctx_per_draw": null_k_ctx,
            "k_reached_ans_per_draw": null_k_ans,
            "k_ctx_p975": float(np.percentile(null_k_ctx, 97.5)) if null_k_ctx else float("nan"),
            "k_ans_p975": float(np.percentile(null_k_ans, 97.5)) if null_k_ans else float("nan"),
        },
        "rank_null_matrix_ctx": null_mat_ctx,
        "rank_null_matrix_ans": null_mat_ans,
    }


def _load_parent_rung_index(args) -> dict:
    """(model_or_pairmodel, pair_key, arm) -> rung_reached_point."""
    out: dict = {}
    for model in MODEL_SLUGS:
        p = args.parent_ladder_dir / f"ladder_{model}_L19.json"
        if p.exists():
            ladder = json.loads(p.read_text())
            for pair_key, arms in ladder.get("pairs", {}).items():
                for arm, res in arms.items():
                    if isinstance(res, dict) and "rung_reached_point" in res:
                        out[(pair_key, arm, model)] = int(res["rung_reached_point"])
    if args.crossmodel_ladder_json is not None and Path(args.crossmodel_ladder_json).exists():
        ladder = json.loads(Path(args.crossmodel_ladder_json).read_text())
        for pair_key, arms in ladder.get("pairs", {}).items():
            for arm, res in arms.items():
                if isinstance(res, dict) and "rung_reached_point" in res:
                    out[(pair_key, arm, None)] = int(res["rung_reached_point"])
    return out


def cmd_units(args) -> int:
    lams = (
        fl.LAMBDAS if args.lambda_grid == "ladder13" else fl.resolve_lambda_grid(args.lambda_grid)
    )
    specs = build_pair_specs(args)
    units = [(spec, arm) for spec in specs for arm in ("prefix", "context")]
    shard_units = units[args.shard_index :: args.num_shards] if args.num_shards > 1 else units
    pairs_dir = args.out_root / "pairs"
    bundles_dir = args.out_root / "bundles"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    cache = _CellCache(args.store_root, args.layer)
    rung_index = _load_parent_rung_index(args)
    want = regime_meta(args)
    n_shard, n_fail = len(shard_units), 0
    for i, (spec, arm) in enumerate(shard_units):
        (sm, sc_), (tm, tc_) = spec
        uk = unit_key(spec, arm)
        upath = pairs_dir / f"{uk}.json"
        if upath.exists():
            try:
                prior = json.loads(upath.read_text())
            except (json.JSONDecodeError, OSError):
                prior = None
            if prior is not None and prior.get("meta") == want and not prior.get("retryable"):
                print(f"[cms] unit {i + 1}/{n_shard} {uk} RESUME (checkpoint)", flush=True)
                continue
        pair_key = fl.pair_spec_key(spec)
        parent_rung = (
            rung_index.get((pair_key, arm, sm))
            if sm == tm
            else rung_index.get((pair_key, arm, None))
        )
        print(f"[cms] unit {i + 1}/{n_shard} {uk} (parent_rung={parent_rung})", flush=True)
        t0 = time.perf_counter()
        try:
            source = cache.get(sm, sc_)
            target = cache.get(tm, tc_)
            unit, bundle = run_structure_unit(source, target, spec, arm, args, lams, parent_rung)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            unit, bundle = {"error": f"{type(exc).__name__}: {exc}", "retryable": True}, {}
        if "error" in unit:
            n_fail += 1
            unit.setdefault("unit_key", uk)
            unit["meta"] = want
            _atomic_write_json(upath, unit)
            print(f"[cms] unit {i + 1}/{n_shard} {uk} FAILED: {unit['error']}", flush=True)
            continue
        if bundle:
            _atomic_savez(bundles_dir / f"{uk}.npz", **bundle)
        _atomic_write_json(upath, unit)
        print(
            f"[cms] unit {i + 1}/{n_shard} {uk} done weakest={unit['weakest_class_point']} "
            f"elapsed={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    print(f"[cms] units phase done: {n_shard} units, {n_fail} failures (recorded)", flush=True)
    return 0


def _framing(cond: str) -> str:
    return cond.rsplit("_", 1)[-1]


def _identity(cond: str) -> str:
    return cond.rsplit("_", 1)[0]


def _principal_overlap(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """Mean squared principal-angle cosine between two k-frames (chance ~ k/d)."""
    sv = np.linalg.svd(U_a.T @ U_b, compute_uv=False)
    return float((sv**2).mean())


def cmd_overlap(args) -> int:
    """Item 7c (+ item 9 uniformity): cross-pair subspace overlap vs random null."""
    bundles_dir = args.out_root / "bundles"
    frames: dict[str, dict] = {}
    for bpath in sorted(bundles_dir.glob("*.npz")):
        with np.load(bpath) as z:
            if "m_minus_i_u256_fp16" not in z.files:
                continue
            frames[bpath.stem] = {
                "U": z["m_minus_i_u256_fp16"].astype(np.float64),
                "Vh": z["m_minus_i_vh256_fp16"].astype(np.float64),
            }
    if not frames:
        print("[cms-overlap] no bundles found — nothing to do", flush=True)
        return 0
    ks = (SUBSPACE_K_PRIMARY, SUBSPACE_K_SENSITIVITY)
    keys = sorted(frames)
    out: dict = {"k_primary": SUBSPACE_K_PRIMARY, "k_sensitivity": SUBSPACE_K_SENSITIVITY}
    d = frames[keys[0]]["U"].shape[0]
    rng = np.random.default_rng(args.seed + 3)
    null_stats: dict[int, dict] = {}
    for k in ks:
        draws = []
        for _ in range(SUBSPACE_NULL_DRAWS):
            q1, _ = np.linalg.qr(rng.standard_normal((d, k)))
            q2, _ = np.linalg.qr(rng.standard_normal((d, k)))
            draws.append(_principal_overlap(q1, q2))
        arr = np.asarray(draws)
        null_stats[k] = {
            "n_draws": SUBSPACE_NULL_DRAWS,
            "chance_k_over_d": k / d,
            "null_mean": float(arr.mean()),
            "null_p975": float(np.quantile(arr, 0.975)),
        }
    out["random_subspace_null"] = {int(k): v for k, v in null_stats.items()}
    pairs_out = []
    for i, ka in enumerate(keys):
        for kb in keys[i + 1 :]:
            row = {"a": ka, "b": kb}
            for k in ks:
                Ua = frames[ka]["U"][:, :k]
                Ub = frames[kb]["U"][:, :k]
                Va = frames[ka]["Vh"][:k].T
                Vb = frames[kb]["Vh"][:k].T
                row[f"left_overlap_k{k}"] = _principal_overlap(Ua, Ub)
                row[f"right_overlap_k{k}"] = _principal_overlap(Va, Vb)
            pairs_out.append(row)
    out["unit_pairs"] = pairs_out
    out["metadata"] = _metadata()
    _atomic_write_json(args.out_root / "subspace_overlap.json", out)
    print(
        f"[cms-overlap] wrote overlap for {len(keys)} units ({len(pairs_out)} unit-pairs)",
        flush=True,
    )
    return 0


def cmd_merge(args) -> int:
    specs = build_pair_specs(args)
    units = [(spec, arm) for spec in specs for arm in ("prefix", "context")]
    assert_distinct_unit_keys(units, getattr(args, "expect_units", None))
    pairs_dir = args.out_root / "pairs"
    rows, failures, missing, key_mismatch = [], [], [], []
    for spec, arm in units:
        uk = unit_key(spec, arm)
        upath = pairs_dir / f"{uk}.json"
        if not upath.exists():
            missing.append(uk)
            continue
        unit = json.loads(upath.read_text())
        if "error" in unit:
            failures.append({"unit_key": uk, "error": unit["error"]})
            continue
        if unit_identity_mismatch(unit, spec, arm):
            key_mismatch.append({"unit_key": uk, "recorded_src_model": unit.get("src_model")})
            continue
        rows.append(unit)
    weakest_counts: dict[str, dict[str, int]] = {}
    rank_reached: dict[str, dict[str, int | None]] = {}
    for unit in rows:
        model_key = (
            unit["src_model"]
            if not unit["cross_model"]
            else f"{unit['src_model']}->{unit['tgt_model']}"
        )
        gk = f"{model_key}|{unit['arm']}"
        w = unit["weakest_class_point"]
        weakest_counts.setdefault(gk, {}).setdefault(w, 0)
        weakest_counts[gk][w] += 1
        rr = unit.get("rank_rung", {})
        if rr.get("eligible") and "k_reached_ctx" in rr:
            rank_reached[unit["unit_key"]] = {
                "ctx": rr["k_reached_ctx"],
                "ans": rr["k_reached_ans"],
            }
    summary = {
        "meta": regime_meta(args),
        "n_expected_units": len(units),
        "n_complete": len(rows),
        "n_failed": len(failures),
        "n_missing": len(missing),
        "n_key_mismatch": len(key_mismatch),
        "failures": failures,
        "missing_units": missing[:50],
        "key_mismatch_units": key_mismatch[:50],
        "weakest_class_counts": weakest_counts,
        "rank_reached": rank_reached,
        "metadata": _metadata(),
    }
    _atomic_write_json(args.out_root / "summary.json", summary)
    print(
        f"[cms-merge] wrote summary: {len(rows)} complete / {len(failures)} failed / "
        f"{len(missing)} missing / {len(key_mismatch)} key-mismatched of {len(units)} units",
        flush=True,
    )
    return 3 if (missing or key_mismatch) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["units", "overlap", "merge"], required=True)
    ap.add_argument(
        "--fit-basis",
        choices=["ambient", "reduced"],
        default="ambient",
        help="fit basis (plan v10): ambient (parent, byte-identical default) or the "
        "per-(pair, arm, fold) shared train-fold-only PCA rank-k basis",
    )
    ap.add_argument("--store-root", type=Path, default=None)
    ap.add_argument(
        "--out-root", type=Path, default=Path("eval_results/issue_1689/context_map_structure")
    )
    ap.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    ap.add_argument("--lambda-grid", choices=sorted(LAMBDA_GRIDS), default="ladder13")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--pairs-file", type=Path, default=None)
    ap.add_argument("--default-model", type=str, default=None)
    ap.add_argument("--pair-set", choices=["within-model", "cross-model"], default="within-model")
    ap.add_argument("--models", type=str, default=",".join(MODEL_SLUGS))
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument(
        "--expect-units",
        type=int,
        default=None,
        help="merge: hard-assert exactly N distinct qualified units are enumerated",
    )
    ap.add_argument("--items", choices=["structure", "rank", "both"], default="both")
    ap.add_argument("--class-null-draws", type=int, default=40)
    ap.add_argument("--rank-null-draws", type=int, default=40)
    ap.add_argument("--row-limit", type=int, default=None)
    ap.add_argument("--dim-limit", type=int, default=None)
    ap.add_argument(
        "--parent-ladder-dir", type=Path, default=Path("eval_results/issue_1689/ladder")
    )
    ap.add_argument("--crossmodel-ladder-json", type=Path, default=None)
    args = ap.parse_args()

    if args.phase == "units" and args.store_root is None:
        ap.error("--phase units requires --store-root")
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda but torch.cuda.is_available() is False")
    print(
        f"[cms] phase={args.phase} items={args.items} pair_set={args.pair_set} "
        f"device={args.device} shard={args.shard_index}/{args.num_shards} "
        f"row_limit={args.row_limit} dim_limit={args.dim_limit} fit_basis={fit_basis_of(args)}",
        flush=True,
    )
    return {"units": cmd_units, "overlap": cmd_overlap, "merge": cmd_merge}[args.phase](args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
