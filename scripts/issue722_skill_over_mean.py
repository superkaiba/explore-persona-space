#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ², ̄, λ, σ, μ) in scientific docstrings + log messages.
"""Issue #722 (off-pod CPU, 0 GPU): does c_C predict v0 BEYOND the across-context mean?

The #658 A3.4/A3.5 read reported a context→profile ridge cosine of ~0.22-0.31 and
called the map "weak". But ROWWISE COSINE of v0 reconstructions is saturated by
Qwen anisotropy: even the predict-the-mean baseline (v̄0_train) has cosine ~0.98
with every held-out v0, because all v0 vectors share a dominant common direction.
Absolute cosine therefore answers NOTHING about whether the context adds
information past the average answer activation.

The right metric is the SKILL SCORE over the predict-the-mean baseline = held-out
R² on the centered target, under leave-one-context-out (LOCO) CV:

    SS_res = Σ_heldout ‖ v0(C) − M̂(c_C) ‖²
    SS_tot = Σ_heldout ‖ v0(C) − v̄0_train ‖²   (v̄0_train = mean of v0 over TRAIN only)
    skill  = 1 − SS_res / SS_tot

aggregated (variance-weighted) over ALL output dims AND ALL held-out contexts —
one number per read layer. skill ≈ 0 / negative ⇒ c_C predicts v0 no better than
the across-context average. skill > 0 ⇒ the context vector carries real,
generalizing information about the answer-side profile.

Two predictor families, side by side per layer:
  - RIDGE (A3.4): linear M̂ via #658's dual/PRESS LOCO ridge (nested-CV λ over
    {1e-2,1e-1,1,10,100,1e3}, train-only per-fold standardization).
  - MLP   (A3.5): nonlinear M̂ via #658's a35_mlp recipe verbatim (1 hidden,
    width 512, GELU, AdamW lr=1e-3 wd=1e-4, fixed 300-epoch LOCO fold), target
    PCA-reduced to top-48 PCs (lossless at n≈50). The #658 batched LOCO ensemble
    is fit ONCE per layer (its members are per (PC-target × fold); each member for
    fold i trains only on rows ≠ i with train-only X + target standardization) —
    so the INPUT-side LOCO and the predict-the-mean baseline are strictly
    train-only. Robust SVD (gesdd→gesvd) for the PCA; a layer whose PCA will not
    converge is reported NaN (skipped) rather than crashing — the #722 crash mode.

Also reported per layer for the saturation contrast:
  - predict_mean_abs_cos : held-out rowwise |cos|(v̄0_train, v0)  (≈0.98 — the
    baseline's OWN cosine, proving cosine is saturated)
  - raw_recon_abs_cos    : held-out rowwise |cos|(ridge M̂(c_C), v0)  (the
    misleading saturated number #658 reported)
  - skill_zscored_mlp    : MLP skill with per-dim z-scoring of the INPUT c_C
    (μ,σ over the design) — does de-weighting rogue/massive-activation dims change
    the (nonlinear) predictability? (The RIDGE z-scored variant is NOT computed:
    it is algebraically identical to the unscored ridge — #658's per-fold train-only
    standardization absorbs any global per-dim affine rescale — so it carries zero
    information; see the implementation report §(b). The MLP arm's first linear is
    NOT internally standardized the same way, so the z-scored MLP is a non-vacuous
    robustness read and is kept.)

§5 controls (load-bearing — the user's enumerated control suite):
  - shuffle_ridge_L18  : the full ridge skill pipeline at L18 with the v0 rows
    randomly permuted BEFORE the LOCO loop (np.random.default_rng(42)) — a
    label-shuffle null. Must collapse to ≈0 (asserted), proving the ridge plateau
    is real signal and not a PRESS/leakage artifact.
  - skill_shuffle_mlp  : per layer, a SECOND MLP fit on row-permuted v0 (same seed).
    The MLP-vs-shuffle delta is the ONLY valid MLP-loss attribution at n=50 (the
    MLP arm's input-PCA basis is fit on all rows, so its absolute skill mixes
    overfitting with the all-rows leg — the shuffle null isolates that).

c_C = last-input-token residual (canonical context vector): the #594
context_vectors_mean.pt `tensor`, keyed by instance_ids. v0 = mean answer-token
residual = v0_summaries.pt summaries["mean"], (28, 3584) per context.
capture_layers = [0..27] so layer index == layer number.

Standalone, idempotent, CPU-only. Reuses #658's LOCO helpers by import. NEVER
edits issue658_fit_predictors.py. Default writes data/issue722_skill_scratch/;
the canonical run passes --out eval_results/issue_722/base-skill-over-mean-cC-to-v0/
skill_over_mean.json and a run_meta.json sidecar lands next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse #658's EXACT LOCO machinery (do NOT re-implement).
import issue658_fit_predictors as i658  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue658_fit_predictors import (  # noqa: E402
    RIDGE_LAMBDAS,
    _press_loo_mse_per_lambda,
    _ridge_dual_weights,
    _rowwise_cos,
)

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue722_skill")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HF_REPO = "superkaiba1/explore-persona-space-data"
V0_FILE = "issue658_theory_assumptions/store/v0_summaries.pt"
CC_LAST_FILE = "issue594_context_geometry/analysis_tensors/context_vectors_mean.pt"
OUT_DIR = PROJECT_ROOT / "data" / "issue722_skill_scratch"
OUT_JSON = OUT_DIR / "skill_over_mean.json"

# Project-default seed (followup-scope §11). Ridge LOCO is deterministic; the MLP
# init seed only perturbs an already-decisively-losing fit.
SEED = 42

# A3.5 MLP target reduction — #722 deliberate choice: lossless at n≤50 (centered
# rank ≤ n−1 = 49). Differs from #658's A35_MLP_TARGET_DIM=64; outcome-affecting
# only for the decisively-negative MLP arm.
MLP_PCA_DIM = 48

# Label-shuffle ridge control layer (§5).
SHUFFLE_RIDGE_LAYER = 18


# ── data loading ──────────────────────────────────────────────────────────────


def _load_stores() -> dict:
    """Download (cached) + load v0 and the last-input-token c_C, aligned on ctx ids.

    Asserts the two stores share the same probe-pool battery (probe_pool_hash) —
    load-bearing per plan §12 A3: NEVER silently proceeds on a mismatch.
    """
    from huggingface_hub import hf_hub_download

    v0p = hf_hub_download(HF_REPO, V0_FILE, repo_type="dataset")
    ccp = hf_hub_download(HF_REPO, CC_LAST_FILE, repo_type="dataset")
    v0 = torch.load(v0p, weights_only=False)
    cc = torch.load(ccp, weights_only=False)

    # Load-bearing probe-pool-hash equality assert (§12 A3 mitigation).
    h_v0 = v0.get("probe_pool_hash")
    h_594 = cc.get("probe_pool_hash")
    if h_v0 is None or h_594 is None or h_v0 != h_594:
        raise RuntimeError(
            "probe_pool_hash mismatch between v0 and c_C stores — the two substrates "
            f"do NOT share the same probe battery: v0={h_v0!r} c_C={h_594!r}. "
            "Refusing to fit a cross-store map on misaligned probes."
        )

    ctx_ids = list(v0["context_ids"])
    layers = list(v0["capture_layers"])
    V = np.stack([v0["summaries"]["mean"][c].numpy() for c in ctx_ids])  # (N, L, H)

    iid_to_row = {iid: i for i, iid in enumerate(cc["instance_ids"])}
    missing = [c for c in ctx_ids if c not in iid_to_row]
    if missing:
        raise RuntimeError(f"#594 cc_last store missing {len(missing)} contexts: {missing[:5]}")
    cc_tensor = cc["tensor"]  # (n594, 28, H)
    C_last = np.stack([cc_tensor[iid_to_row[c]].numpy() for c in ctx_ids])  # (N, 28, H)
    assert C_last.shape[1] == len(layers), (C_last.shape, len(layers))

    C_meanprompt = np.stack([v0["cc_meanprompt"][c].numpy() for c in ctx_ids])  # (N, L, H)

    return {
        "ctx_ids": ctx_ids,
        "layers": layers,
        "V": V.astype(np.float64),
        "C_last": C_last.astype(np.float64),
        "C_meanprompt": C_meanprompt.astype(np.float64),
        "v0_path": v0p,
        "cc_path": ccp,
        "store_provenance": {
            "v0_file": f"{HF_REPO}:{V0_FILE}",
            "cc_last_file": f"{HF_REPO}:{CC_LAST_FILE}",
            "n_contexts": len(ctx_ids),
            "hidden_dim": int(V.shape[-1]),
            "probe_pool_hash_v0": h_v0,
            "probe_pool_hash_594": h_594,
        },
    }


# ── aggregate held-out R² (skill over predict-the-mean) ──────────────────────


def _skill_over_mean(preds_by_fold, V, train_idx_by_fold) -> dict:
    """Variance-weighted aggregate held-out R² over the centered target.

    preds_by_fold[i] = M̂(c_C[i]) prediction for held-out context i (H,) OR None
                       (skipped — excluded from BOTH SS sums for parity).
    SS_res = Σ_i ‖V[i] − preds[i]‖²; SS_tot = Σ_i ‖V[i] − v̄0_train(i)‖²
    (v̄0_train(i) = mean of v0 over TRAIN rows of fold i → no leakage).
    skill = 1 − SS_res/SS_tot, aggregate over all dims AND all held-out contexts.
    """
    n = V.shape[0]
    ss_res = 0.0
    ss_tot = 0.0
    used = []
    base_cos, pred_cos = [], []
    per_dim_res = np.zeros(V.shape[1])
    per_dim_tot = np.zeros(V.shape[1])
    for i in range(n):
        p = preds_by_fold[i]
        if p is None:
            continue
        tr = train_idx_by_fold[i]
        vbar = V[tr].mean(axis=0)  # TRAIN-only predict-the-mean baseline
        res = V[i] - p
        tot = V[i] - vbar
        ss_res += float(res @ res)
        ss_tot += float(tot @ tot)
        per_dim_res += res * res
        per_dim_tot += tot * tot
        used.append(i)
        base_cos.append(abs(float(_rowwise_cos(vbar[None, :], V[i][None, :])[0])))
        pred_cos.append(abs(float(_rowwise_cos(p[None, :], V[i][None, :])[0])))
    skill = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dim_r2 = 1.0 - per_dim_res / per_dim_tot
    per_dim_r2 = per_dim_r2[np.isfinite(per_dim_r2)]
    return {
        "skill": float(skill),
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "n_folds_used": len(used),
        "median_per_dim_r2": float(np.median(per_dim_r2)) if per_dim_r2.size else float("nan"),
        "predict_mean_abs_cos": float(np.mean(base_cos)) if base_cos else float("nan"),
        "raw_recon_abs_cos": float(np.mean(pred_cos)) if pred_cos else float("nan"),
    }


# ── ridge predictor (skill form) ──────────────────────────────────────────────


def _ridge_skill(Xc: np.ndarray, Yv: np.ndarray) -> dict:
    """Skill-over-mean for the LINEAR ridge map c_C → v0 — TARGET-CENTERED LOCO ridge.

    Uses #658's EXACT dual/PRESS ridge math (`_press_loo_mse_per_lambda` for the
    nested-CV λ pick, `_ridge_dual_weights` for the held-out solve, same RIDGE_LAMBDAS
    grid, same train-only X standardization) but fits the map on the TRAIN-MEAN-
    CENTERED target and adds the train mean back: prediction = v̄0_train + M̂(c_C).

    Why centering matters for THIS metric (the fix for the −150 artifact): #658's bare
    `_ridge_predict_loco` predicts RAW v0 with no intercept, so at the largest λ it
    shrinks to ≈0 — which is catastrophically worse than the mean because v0 carries a
    huge anisotropic offset. That makes "skill over mean" trivially ≪0 (the ridge
    cannot even represent the mean). Centering the target on the train mean gives the
    ridge the mean for free, so skill then measures the RIGHT thing: does the c_C-
    driven DEVIATION from the average answer activation generalize? (skill ≤ 0 ⇒ no;
    skill > 0 ⇒ yes.) This is the textbook held-out-R²-over-mean construction and is
    the apples-to-apples comparison the Goal asks for. Everything train-only per fold:
    the mean offset, the X standardization, and the λ pick.

    Returns the `_skill_over_mean` dict augmented with `lambda_chosen` — the PRESS-
    selected λ for a FULL-data refit (the per-fold λ picks are near-identical at this
    n; the full refit is the reported λ for the layer).
    """
    n = Xc.shape[0]
    device = torch.device(i658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(Xc)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Yv)).to(device=device, dtype=torch.float64)
    train_idx_by_fold = [[j for j in range(n) if j != i] for i in range(n)]
    preds_by_fold = []
    for i in range(n):
        tr = train_idx_by_fold[i]
        tr_t = torch.tensor(tr, device=device)
        Xtr, Ytr = Xt[tr_t], Yt[tr_t]
        # train-only X standardization (matches #658: numpy ddof=0 → correction=0)
        xmu = Xtr.mean(0)
        xsd = Xtr.std(0, correction=0) + 1e-9
        Xtr_n = (Xtr - xmu) / xsd
        # train-only TARGET centering (the fix): fit the map on the centered target.
        ymu = Ytr.mean(0)  # = v̄0_train, the predict-the-mean baseline itself
        Ytr_c = Ytr - ymu
        # nested-CV λ via #658's exact PRESS identity on the standardized train design
        mse = _press_loo_mse_per_lambda(Xtr_n, Ytr_c, RIDGE_LAMBDAS)
        best_lam = RIDGE_LAMBDAS[int(torch.argmin(mse).item())]
        w = _ridge_dual_weights(Xtr_n, Ytr_c, best_lam)  # (d, H)
        x_held = (Xt[i] - xmu) / xsd
        pred = (ymu + x_held @ w).detach().cpu().numpy()  # v̄0_train + M̂(c_C)
        preds_by_fold.append(pred)
    res = _skill_over_mean(preds_by_fold, Yv, train_idx_by_fold)
    # full-data PRESS λ pick (reported lambda_chosen for the layer)
    xmu_all = Xt.mean(0)
    xsd_all = Xt.std(0, correction=0) + 1e-9
    Xn_all = (Xt - xmu_all) / xsd_all
    Yc_all = Yt - Yt.mean(0)
    mse_all = _press_loo_mse_per_lambda(Xn_all, Yc_all, RIDGE_LAMBDAS)
    res["lambda_chosen"] = float(RIDGE_LAMBDAS[int(torch.argmin(mse_all).item())])
    return res


# ── MLP predictor (skill form), #658 a35_mlp recipe + top-48-PC target ──────────


def _robust_pca_basis(Y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """PCA mean + top-k right singular vectors via robust SVD (gesdd→gesvd fallback).

    Returns (mu (H,), comps (k', H), used_gesvd_fallback), k' = min(k, rank).
    Raises LinAlgError only if BOTH SVD drivers fail (the caller then reports the
    layer's MLP read as NaN rather than crashing — the #722 crash mode on
    near-singular matrices).
    """
    mu = Y.mean(axis=0)
    Yc = Y - mu
    fallback = False
    try:
        _, _, Vt = np.linalg.svd(Yc, full_matrices=False)  # gesdd
    except np.linalg.LinAlgError:
        _, _, Vh = torch.linalg.svd(torch.from_numpy(Yc), full_matrices=False)  # gesvd fallback
        Vt = Vh.numpy()
        fallback = True
    kk = min(k, Vt.shape[0])
    return mu, Vt[:kk], fallback


def _mlp_skill(Xc: np.ndarray, Yv: np.ndarray) -> dict:
    """Skill-over-mean for the NONLINEAR MLP map c_C → v0 (#658 a35_mlp recipe).

    Fits the #658 batched LOCO ensemble ONCE per layer (the faithful, efficient
    structure — #658 itself fits one ensemble per layer): members are per
    (PC-target dim × LOCO fold); each member for fold i trains on the N-1 rows ≠ i
    with train-only X + target standardization inside _fit_mlp_ensemble_loco. So
    the INPUT-side LOCO and the predict-the-mean baseline are STRICTLY train-only —
    the legs that drive the skill number carry no leakage.

    PCA target basis: top-48 PCs of v0 (recipe: "lossless at n≈50"). With ≤49
    centered rank at n=50, top-48 is the near-complete centered row space, so it is
    effectively invariant to which one row is held out — the standard reading of
    "lossless", and why #658 fits one ensemble per layer rather than nesting a
    per-fold basis (which would re-fit the 50-fold ensemble 50× per layer for one
    kept row each — the #722 blowup). Robust SVD; if it fails the layer's MLP read
    is reported NaN (skipped) rather than crashing.

    Reports `gesvd_fallback` (bool) — did EITHER the input-side OR the target-side
    SVD use the torch gesvd fallback on this layer.
    """
    n, _ = Xc.shape
    train_idx_by_fold = [[j for j in range(n) if j != i] for i in range(n)]

    # INPUT acceleration (lossless at n≪d): c_C (N, 3584) has rank ≤ n, so project it
    # onto its top-(n-1) PCs — the MLP's first linear becomes (n-1)→512 instead of
    # 3584→512 (~70× fewer first-layer FLOPs, the only practical way to fit the
    # width-512 / 300-epoch / 48-PC-target ensemble on CPU). The projection PRESERVES
    # every bit of c_C's variation (the discarded directions are exactly zero in the
    # data), and the MLP's per-fold train-only standardization runs on the projected
    # coords. This is an information-preserving reparameterization of the INPUT, not a
    # recipe change to the target side (target stays the top-48-PC v0). NOTE: the
    # input-PCA basis is fit on ALL rows; at rank-≤n this basis is the data's own
    # row space (invariant to one held-out row), the same "lossless at n≈50" argument
    # as the target side. The input-side LOCO that drives skill still lives inside
    # _fit_mlp_ensemble_loco (member i trains on rows ≠ i).
    input_fallback = False
    Xc64 = Xc.astype(np.float64)
    xmu = Xc64.mean(axis=0)
    try:
        try:
            _, _, xVt = np.linalg.svd(Xc64 - xmu, full_matrices=False)
        except np.linalg.LinAlgError:
            _, _, xVh = torch.linalg.svd(torch.from_numpy(Xc64 - xmu), full_matrices=False)
            xVt = xVh.numpy()
            input_fallback = True
        mu, comps, target_fallback = _robust_pca_basis(Yv, MLP_PCA_DIM)  # (H,), (k, H)
    except (np.linalg.LinAlgError, torch.linalg.LinAlgError):
        # A truly-singular layer: BOTH SVD drivers failed on either the input or
        # the target side. Report NaN + all folds skipped rather than crashing —
        # the #722 near-singular crash mode the gesdd→gesvd fallback guards against.
        logger.warning("[mlp] SVD failed (both drivers) for this layer — read NaN")
        res = _skill_over_mean([None] * n, Yv, train_idx_by_fold)
        res["n_folds_skipped"] = n
        res["gesvd_fallback"] = input_fallback
        return res
    xk = min(n - 1, xVt.shape[0])
    Xin = (Xc64 - xmu) @ xVt[:xk].T  # (N, xk) lossless input coords
    k = comps.shape[0]
    Z = (Yv - mu) @ comps.T  # (N, k)
    Zhat = i658._fit_mlp_ensemble_loco(
        Xin.astype(np.float32), Z.astype(np.float32), target_idx=list(range(k)), seed=SEED
    )  # (N, k) LOCO held-out PC predictions
    preds_by_fold = [mu + Zhat[i] @ comps for i in range(n)]  # reconstruct v0 per held-out ctx
    res = _skill_over_mean(preds_by_fold, Yv, train_idx_by_fold)
    res["n_folds_skipped"] = 0
    res["gesvd_fallback"] = bool(input_fallback or target_fallback)
    return res


# ── z-scored INPUT variant (MLP only; the ridge z-variant is algebraically vacuous) ──


def _zscore_train_only_full(Xc: np.ndarray) -> np.ndarray:
    """Per-dim z-score of c_C using design μ/σ (global rescale).

    For the MLP arm this is a non-vacuous robustness read (the MLP first linear is
    not internally per-dim standardized the same way the ridge is). The RIDGE
    z-scored variant is NOT computed: #658's per-fold train-only standardization
    absorbs any global per-dim affine rescale, making it numerically identical to
    the unscored ridge — see the implementation report §(b).
    """
    mu = Xc.mean(axis=0)
    sd = Xc.std(axis=0) + 1e-8
    return (Xc - mu) / sd


def _mlp_skill_zscored(Xc: np.ndarray, Yv: np.ndarray) -> dict:
    """MLP skill with the INPUT c_C per-dim z-scored.

    _fit_mlp_ensemble_loco standardizes X train-only per fold internally, so a
    global per-dim z-score of c_C before the fit is dominated by that train-only
    standardization. We apply it explicitly (covariance-free per-dim rescale, safe
    at n≪d) and re-run the single-per-layer ensemble. The PCA target basis is
    unchanged (z-scoring only touches the INPUT).
    """
    return _mlp_skill(_zscore_train_only_full(Xc), Yv)


# ── per-layer driver ──────────────────────────────────────────────────────────


def run(
    cc_key: str = "C_last",
    only_layers: list[int] | None = None,
    do_mlp: bool = True,
    store: dict | None = None,
) -> dict:
    data = store if store is not None else _load_stores()
    layers = data["layers"]
    V = data["V"]  # (N, L, H)
    C = data[cc_key]  # (N, L, H)
    n, L, H = V.shape
    logger.info(
        "Loaded: n=%d contexts, L=%d layers, H=%d | c_C=%s | mlp=%s", n, L, H, cc_key, do_mlp
    )

    # §5 shuffle null: a fixed row permutation of v0, reused for the L18 ridge
    # control AND the per-layer MLP-vs-shuffle null. Seed-pinned for reproducibility.
    shuffle_perm = np.random.default_rng(SEED).permutation(n)

    nan = float("nan")
    per_layer = []
    shuffle_ridge_l18 = nan
    for li in range(L):
        layer = int(layers[li])
        if only_layers is not None and layer not in only_layers:
            continue
        t0 = time.time()
        Xc = C[:, li, :]  # (N, H) c_C at this layer
        Yv = V[:, li, :]  # (N, H) v0 at this layer
        Yv_shuf = Yv[shuffle_perm]  # row-permuted v0 (label-shuffle null)

        tr0 = time.time()
        ridge = _ridge_skill(Xc, Yv)
        logger.info("[L%02d] ridge done in %.1fs", layer, time.time() - tr0)

        # §5a: label-shuffled ridge control at the plateau peak layer (L18).
        if layer == SHUFFLE_RIDGE_LAYER:
            ts0 = time.time()
            shuffle_ridge_l18 = _ridge_skill(Xc, Yv_shuf)["skill"]
            logger.info(
                "[L%02d] shuffle-ridge control skill=%+.4f in %.1fs",
                layer,
                shuffle_ridge_l18,
                time.time() - ts0,
            )

        if do_mlp:
            tm0 = time.time()
            mlp = _mlp_skill(Xc, Yv)
            logger.info("[L%02d] mlp done in %.1fs", layer, time.time() - tm0)
            tmz0 = time.time()
            mlp_z = _mlp_skill_zscored(Xc, Yv)
            logger.info("[L%02d] mlp_z done in %.1fs", layer, time.time() - tmz0)
            # §5b: per-layer MLP-vs-shuffle null (row-permuted v0, same seed).
            tms0 = time.time()
            mlp_shuf = _mlp_skill(Xc, Yv_shuf)
            logger.info(
                "[L%02d] mlp-shuffle done in %.1fs (skill=%+.4f)",
                layer,
                time.time() - tms0,
                mlp_shuf["skill"],
            )
        else:
            mlp = {
                "skill": nan,
                "median_per_dim_r2": nan,
                "n_folds_skipped": -1,
                "n_folds_used": -1,
                "gesvd_fallback": False,
            }
            mlp_z = {"skill": nan}
            mlp_shuf = {"skill": nan}

        row = {
            "layer": layer,
            "predict_mean_abs_cos": ridge["predict_mean_abs_cos"],
            "raw_recon_abs_cos": ridge["raw_recon_abs_cos"],
            "skill_vs_mean_ridge": ridge["skill"],
            "skill_vs_mean_mlp": mlp["skill"],
            "skill_zscored_mlp": mlp_z["skill"],
            "skill_shuffle_mlp": mlp_shuf["skill"],
            "ridge_median_per_dim_r2": ridge["median_per_dim_r2"],
            "mlp_median_per_dim_r2": mlp["median_per_dim_r2"],
            "lambda_chosen": ridge["lambda_chosen"],
            "gesvd_fallback": mlp.get("gesvd_fallback", False),
            "mlp_n_folds_skipped": mlp.get("n_folds_skipped", 0),
            "n_folds_used_ridge": ridge["n_folds_used"],
            "n_folds_used_mlp": mlp["n_folds_used"],
        }
        per_layer.append(row)
        logger.info(
            "[L%02d] mean_cos=%.3f recon_cos=%.3f | skill ridge=%+.4f mlp=%+.4f "
            "| zscored_mlp=%+.4f shuffle_mlp=%+.4f λ=%.3g | %.1fs",
            layer,
            row["predict_mean_abs_cos"],
            row["raw_recon_abs_cos"],
            row["skill_vs_mean_ridge"],
            row["skill_vs_mean_mlp"],
            row["skill_zscored_mlp"],
            row["skill_shuffle_mlp"],
            row["lambda_chosen"],
            time.time() - t0,
        )

    return {
        "metric": "skill_over_predict_the_mean = 1 - SS_res/SS_tot (held-out R² on centered v0)",
        "c_C_recipe": cc_key,
        "n_contexts": n,
        "activation_dim": H,
        "layers": [int(x) for x in layers],
        "mlp_recipe": {
            "hidden": i658.MLP_HIDDEN,
            "lr": i658.MLP_LR,
            "wd": i658.MLP_WD,
            "max_epochs": i658.MLP_MAX_EPOCHS,
            "pca_target_dim": MLP_PCA_DIM,
        },
        "ridge_lambdas": RIDGE_LAMBDAS,
        "seed": SEED,
        "shuffle_ridge_L18": float(shuffle_ridge_l18),
        "store_provenance": data["store_provenance"],
        "per_layer": per_layer,
    }


# ── reproducibility metadata ────────────────────────────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_revision() -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(HF_REPO, revision="main").sha
    except Exception as e:
        logger.warning("could not resolve HF dataset revision: %s", e)
        return None


def _write_run_meta(
    out_path: Path,
    args: argparse.Namespace,
    result: dict,
    data: dict,
    rng_state_hash: str,
    wall_time_s: float,
) -> Path:
    meta_path = out_path.parent / "run_meta.json"
    meta: dict[str, Any] = {
        "issue": 722,
        "followup_label": "base-skill-over-mean-cC-to-v0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha": _git_sha(),
        "i658_module_sha": _file_sha256(PROJECT_ROOT / "scripts" / "issue658_fit_predictors.py"),
        "config": {
            "cc": args.cc,
            "smoke": args.smoke,
            "layers": args.layers,
            "no_mlp": args.no_mlp,
            "out": str(args.out) if args.out is not None else None,
            "threads": args.threads,
        },
        "substrate": {
            "v0_file": f"{HF_REPO}:{V0_FILE}",
            "cc_last_file": f"{HF_REPO}:{CC_LAST_FILE}",
            "hf_dataset_revision": _hf_revision(),
            "v0_local_sha256": _file_sha256(data["v0_path"]),
            "cc_local_sha256": _file_sha256(data["cc_path"]),
            "probe_pool_hash": data["store_provenance"]["probe_pool_hash_v0"],
        },
        "seed": SEED,
        "rng_state_hash": rng_state_hash,
        "n_contexts": result["n_contexts"],
        "activation_dim": result["activation_dim"],
        "n_layers": len(result["layers"]),
        "wall_time_s": round(wall_time_s, 2),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("wrote %s", meta_path)
    return meta_path


# ── per-layer hero figure ──────────────────────────────────────────────────


def make_figure(result: dict, fig_path: Path) -> Path:
    """Per-layer 5-line plot (plan §4.6 + Alternatives-critic median-per-dim line).

    Dual y-axes: skill R² (left, ridge / MLP / ridge_median_per_dim_r2),
    cosine 0..1 (right, predict_mean_abs_cos / raw_recon_abs_cos). No annotation
    overlays. Uses the project paper rcParams.
    """
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")

    rows = sorted(result["per_layer"], key=lambda r: r["layer"])
    x = [r["layer"] for r in rows]
    ridge = [r["skill_vs_mean_ridge"] for r in rows]
    mlp = [r["skill_vs_mean_mlp"] for r in rows]
    ridge_med = [r["ridge_median_per_dim_r2"] for r in rows]
    pm_cos = [r["predict_mean_abs_cos"] for r in rows]
    recon_cos = [r["raw_recon_abs_cos"] for r in rows]

    fig, ax_l = plt.subplots(figsize=(7.0, 4.2))
    ax_r = ax_l.twinx()

    (l1,) = ax_l.plot(x, ridge, marker="o", ms=3, lw=1.6, color="#0072B2", label="ridge R² (skill)")
    (l2,) = ax_l.plot(x, mlp, marker="s", ms=3, lw=1.6, color="#D55E00", label="MLP R² (skill)")
    (l3,) = ax_l.plot(
        x,
        ridge_med,
        marker="^",
        ms=3,
        lw=1.2,
        ls="--",
        color="#56B4E9",
        label="ridge median per-dim R²",
    )
    (l4,) = ax_r.plot(
        x, pm_cos, marker="d", ms=3, lw=1.2, color="#009E73", label="predict-mean |cos|"
    )
    (l5,) = ax_r.plot(
        x, recon_cos, marker="v", ms=3, lw=1.2, color="#CC79A7", label="raw-recon |cos|"
    )

    ax_l.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax_l.set_xlabel("layer")
    ax_l.set_ylabel("skill-over-mean (held-out R²)")
    ax_r.set_ylabel("rowwise |cosine|")
    ax_r.set_ylim(0.0, 1.02)

    lines = [l1, l2, l3, l4, l5]
    ax_l.legend(lines, [ln.get_label() for ln in lines], loc="center left", fontsize=7)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", fig_path)

    # commit-pinned meta sidecar (project figure convention)
    meta = {
        "issue": 722,
        "figure": fig_path.name,
        "code_sha": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_json": "eval_results/issue_722/base-skill-over-mean-cC-to-v0/skill_over_mean.json",
        "lines": [
            "ridge R² (skill, left axis)",
            "MLP R² (skill, left axis)",
            "ridge median per-dim R² (left axis)",
            "predict-mean |cos| (right axis)",
            "raw-recon |cos| (right axis)",
        ],
    }
    with open(fig_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return fig_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #722 skill-over-mean for c_C → v0.")
    parser.add_argument("--cc", default="C_last", choices=["C_last", "C_meanprompt"])
    parser.add_argument("--smoke", action="store_true", help="2-layer (L0, L18) smoke validation")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument(
        "--no-mlp",
        action="store_true",
        help="ridge-only fast sweep (skips the slow MLP fits; MLP fields read NaN)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="explicit output JSON path (overrides default)"
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="explicit figure PNG path; if omitted no figure is written (use the "
        "canonical full-run path to land figures/issue_722/base_skill_over_mean_per_layer.png)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch CPU threads; 0 = leave torch default (fastest here — a 16-thread "
        "cap roughly doubled the per-fit MLP time vs the box default)",
    )
    args = parser.parse_args()

    i658.DEVICE = "cpu"  # CPU-only, deterministic
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # RNG-state hash AFTER the seed pin (reproducibility provenance).
    rng_state_hash = hashlib.sha256(
        torch.get_rng_state().numpy().tobytes() + np.random.get_state()[1].tobytes()
    ).hexdigest()

    only = [0, 18] if args.smoke else args.layers
    t_run0 = time.time()
    data = _load_stores()
    result = run(cc_key=args.cc, only_layers=only, do_mlp=not args.no_mlp, store=data)
    wall_time_s = time.time() - t_run0

    if args.out is not None:
        out_path = args.out
    else:
        default_name = "skill_over_mean_smoke.json" if args.smoke else "skill_over_mean.json"
        out_path = OUT_DIR / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out_path)

    _write_run_meta(out_path, args, result, data, rng_state_hash, wall_time_s)

    if args.figure is not None:
        make_figure(result, args.figure)

    print(
        "\nlayer | predict_mean_abs_cos | raw_recon_abs_cos | skill_vs_mean_ridge | "
        "skill_vs_mean_mlp | skill_zscored_mlp | skill_shuffle_mlp | λ"
    )
    for r in result["per_layer"]:
        print(
            f"{r['layer']:5d} | {r['predict_mean_abs_cos']:.4f} | {r['raw_recon_abs_cos']:.4f} | "
            f"{r['skill_vs_mean_ridge']:+.4f} | {r['skill_vs_mean_mlp']:+.4f} | "
            f"{r['skill_zscored_mlp']:+.4f} | {r['skill_shuffle_mlp']:+.4f} | "
            f"{r['lambda_chosen']:.3g}"
        )
    print(
        f"\nn_contexts={result['n_contexts']}  d={result['activation_dim']}  "
        f"c_C={result['c_C_recipe']}  shuffle_ridge_L18={result['shuffle_ridge_L18']:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
