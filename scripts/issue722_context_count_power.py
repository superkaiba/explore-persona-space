#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, ², λ, σ, μ, δ, ρ, ∞, ≈, ±, Σ) in scientific docstrings, logs, strings.
"""Issue #722 (off-pod CPU, 0 GPU): how many contexts (n) for the c_C → v0 INFO-THEORY?

The headline question: how many contexts do we realistically need to (1) reliably
CONFIRM that the context carries information about the answer-side profile (a
dependence test), and (2) reliably ESTIMATE that information (mutual information /
its estimable Gaussian surrogate)? All grounded in the REAL #722 stores
(c_C = issue594 context_vectors_mean last-input-token residual; v0 = issue658
v0_summaries mean answer-token residual; n=50 contexts, 28 layers, hidden 3584),
at the ridge-plateau layers (L14, L18, L21; L18 primary), top-48-PC v0 target.

DELIVERABLE 1 — HSIC / distance-correlation dependence-test POWER vs n (PRIMARY).
  "Is information present at all?" The HSIC permutation test + the distance-
  correlation (Székely–Rizzo) permutation test. Calibrate dependence to OUR data:
  generate (c, v) from the fitted linear map M̂ + residual noise at signal fraction
  s ∈ {0.05, 0.1, 0.2, 0.4} (s = Var(signal)/Var(signal+noise) in the top-48-PC
  target space); contexts drawn by resample-with-jitter of the real standardized
  c_C cloud (a full Gaussian at d=3584 ≫ n is rank-deficient — resampling keeps
  the real anisotropic covariance). At n ∈ {15,20,30,50,75,100,150,200,300},
  B datasets each, record rejection rate = POWER. CALIBRATION: at s=0 (independent)
  the rejection rate must be ≈ α=0.05 — verified before any power number is trusted.
  Headline: n*(s) = smallest n reaching 80% power per dependence strength s.

DELIVERABLE 2 — MUTUAL-INFORMATION estimation reliability vs n.
  (a) On the REAL data at n_sub ∈ {15,20,30,40,50}, resampled: the Gaussian-MI
      lower bound from shrinkage CCA  −½ Σ log(1−ρ_i²)  AND a KSG-style estimate
      (sklearn mutual_info_regression per target-PC, summed — a ROUGH proxy, not a
      true joint MI) — with resample CIs.
  (b) A sim with KNOWN ground-truth Gaussian MI (set canonical correlations): chart
      estimator BIAS + variance vs n → the n needed for the estimate within ±20% of
      truth. Honest test of the expectation that full nonparametric MI is unreliable
      at this dim; if so, name the estimable surrogate (Gaussian-MI / HSIC) + its n.

DELIVERABLE 3 — CCA Gaussian-MI learning curve (the linear-information measure).
  Shrinkage CCA between c_C (top-k_in PCs) and v0 (top-48 PCs) → canonical
  correlations → Gaussian-MI, at n_sub ∈ {15..50}, resampled. At n=50 with a
  rank-≤50 input + 48-PC target the raw fit is right at the interpolation edge
  (raw ρ_i → 1, spurious); shrinkage is the fix. Quantify the bias + whether it is
  stable at n=50 + extrapolate the n that stabilizes it.

ALSO kept (cheap shared context): the linear-ridge skill-over-mean R² learning
curve — is the linear MAP itself data-limited at n=50? (folded into Deliverable 3's
JSON as `ridge_skill_curve`.)

REUSE: src/explore_persona_space/analysis/vectorized_mlp_skill.py
  (ridge_predict_loco_centered, skill_over_mean_r2, robust_pca_basis). #658 ridge
  math underneath. NEVER edits issue658_fit_predictors.py.

Standalone, idempotent, CPU-only, seed-pinned. Smoke (--smoke) first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import issue658_fit_predictors as _i658  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue658_fit_predictors import (  # noqa: E402
    _press_loo_mse_per_lambda,
    _ridge_dual_weights,
)
from sklearn.feature_selection import mutual_info_regression  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    robust_pca_basis,
    skill_over_mean_r2,
)

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue722_count_info")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HF_REPO = "superkaiba1/explore-persona-space-data"
V0_FILE = "issue658_theory_assumptions/store/v0_summaries.pt"
CC_LAST_FILE = "issue594_context_geometry/analysis_tensors/context_vectors_mean.pt"

SEED = 42
PCA_DIM = 48  # top-48 PCs of v0 — lossless at n=50 (centered rank ≤ 49)
PLATEAU_LAYERS = [14, 18, 21]  # ridge plateau; L18 primary
PRIMARY_LAYER = 18

# input intrinsic dim for the dependence tests / CCA / generator. The real input
# is rank ≤ n; we work in a modest fixed top-k_in PCA so the tests are well posed at
# the larger simulated n too (a 49-dim design at n=300 is fine; we keep it compact).
INPUT_PCA = 16

# ── Deliverable 1: HSIC / dCor power grid ──
HSIC_NS = [15, 20, 30, 50, 75, 100, 150, 200, 300]
HSIC_SS = [0.0, 0.05, 0.10, 0.20, 0.40]  # signal fractions (s=0 = calibration null)
HSIC_B = 200  # datasets per (n, s)
HSIC_BPERM = 200  # permutations per dataset (the null)
ALPHA = 0.05
POWER_TARGET = 0.80
JITTER_FRAC = 0.25  # resample-with-jitter sd as a fraction of per-dim sd
TARGET_TEST_PCA = 8  # target dim the dependence test sees (compact, well-posed null)

# ── Deliverable 2/3: MI / CCA grids ──
MI_NSUB = [15, 20, 30, 40, 50]
CCA_NSUB = [15, 20, 25, 30, 35, 40, 45, 50]
MI_RESAMPLES = 60
CCA_RESAMPLES = 60
RIDGE_SKILL_RESAMPLES = 8  # capped resamples for the expensive per-fold LOCO ridge-skill
# Regularized-CCA whitening ridge: Σ̃ = Σ + reg·(trΣ/d)·I. reg=0 = raw CCA, which
# at n ≪ (k_in+k_t) drives every canonical correlation → 1 (a degenerate MI that
# explodes — the interpolation-edge pathology this deliverable quantifies). reg≈1
# gives an interpretable, non-degenerate Gaussian-MI. We report a SWEEP of reg so
# the MI-vs-regularization dependence is explicit (the headline G-MI uses CCA_REG).
CCA_REG = 1.0
CCA_REG_SWEEP = [0.0, 0.1, 1.0, 10.0]
# CCA target dim: balance against k_in. A 48-PC target vs a 16-dim input is
# structurally rank-saturated at n≤50, so the CCA/MI uses a COMPACT target dim
# (the ridge-skill R² below keeps the full PCA_DIM=48 — it is a predictive R², not
# a CCA, so the wide target is fine there).
CCA_TARGET_PCA = 16
KSG_K = 3  # k-NN for the KSG/sklearn MI proxy

# MI ground-truth sim
MI_SIM_NS = [20, 30, 50, 75, 100, 150, 200, 300, 500]
MI_SIM_DIM = 8  # joint Gaussian block dim per side for the ground-truth sim
MI_SIM_RHOS = [0.7, 0.5, 0.35, 0.2, 0.1, 0.05, 0.02, 0.01]  # canonical correlations
MI_SIM_B = 200

# Ridge λ grid (for the kept ridge-skill curve + the CCA shrinkage cross-check).
RIDGE_LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


# ── data loading ──────────────────────────────────────────────────────────────


def _load_stores() -> dict:
    """Download (cached) + load v0 + last-input-token c_C, aligned on ctx ids.

    Asserts the two stores share the same probe-pool battery (probe_pool_hash) —
    NEVER silently fits a cross-store map on misaligned probes.
    """
    from huggingface_hub import hf_hub_download

    v0p = hf_hub_download(HF_REPO, V0_FILE, repo_type="dataset")
    ccp = hf_hub_download(HF_REPO, CC_LAST_FILE, repo_type="dataset")
    v0 = torch.load(v0p, weights_only=False)
    cc = torch.load(ccp, weights_only=False)

    h_v0 = v0.get("probe_pool_hash")
    h_594 = cc.get("probe_pool_hash")
    if h_v0 is None or h_594 is None or h_v0 != h_594:
        raise RuntimeError(
            "probe_pool_hash mismatch between v0 and c_C stores: "
            f"v0={h_v0!r} c_C={h_594!r}. Refusing to fit on misaligned probes."
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

    families: dict[str, int] = {}
    for c in ctx_ids:
        fam = str(c).split("_")[0]
        families[fam] = families.get(fam, 0) + 1

    return {
        "ctx_ids": ctx_ids,
        "layers": layers,
        "V": V.astype(np.float64),
        "C_last": C_last.astype(np.float64),
        "families": families,
        "v0_path": v0p,
        "cc_path": ccp,
        "store_provenance": {
            "v0_file": f"{HF_REPO}:{V0_FILE}",
            "cc_last_file": f"{HF_REPO}:{CC_LAST_FILE}",
            "n_contexts": len(ctx_ids),
            "hidden_dim": int(V.shape[-1]),
            "probe_pool_hash": h_v0,
        },
    }


def _layer_slice(data: dict, layer: int) -> tuple[np.ndarray, np.ndarray]:
    layer_index = {int(li): k for k, li in enumerate(data["layers"])}
    li = layer_index[layer]
    return data["C_last"][:, li, :], data["V"][:, li, :]  # (N,H), (N,H)


# ── kernels + dependence statistics (HSIC, distance correlation) ────────────────


def _pairwise_sq(X: np.ndarray) -> np.ndarray:
    """(n, n) squared Euclidean distance matrix."""
    sq = (X * X).sum(1)
    d = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return np.clip(d, 0.0, None)


def _median_gamma(sqd: np.ndarray) -> float:
    """RBF γ from the median-heuristic on the off-diagonal squared distances."""
    n = sqd.shape[0]
    iu = np.triu_indices(n, k=1)
    med = float(np.median(sqd[iu])) if iu[0].size else 1.0
    return 1.0 / med if med > 0 else 1.0


def _hsic_stat(Ksq: np.ndarray, Lsq: np.ndarray) -> float:
    """Biased HSIC estimate from precomputed squared-distance matrices (RBF kernels).

    K = exp(−γ_x · sqd_x), L = exp(−γ_y · sqd_y) with γ from the median heuristic
    of EACH input's own distances. HSIC = (1/n²) tr(K H L H), H = I − 11ᵀ/n the
    centering matrix. Returns the (unnormalized) biased HSIC — only its rank vs a
    permutation null is used, so the constant factor is irrelevant.
    """
    n = Ksq.shape[0]
    K = np.exp(-_median_gamma(Ksq) * Ksq)
    L = np.exp(-_median_gamma(Lsq) * Lsq)
    H = np.eye(n) - np.full((n, n), 1.0 / n)
    Kc = H @ K @ H
    return float(np.sum(Kc * L) / (n * n))


def _distance_correlation(Asq_x: np.ndarray, Asq_y: np.ndarray) -> float:
    """Székely–Rizzo distance correlation from precomputed squared-distance matrices.

    Double-centers the EUCLIDEAN distance matrices (√sqd), forms dCov² = mean(A∘B),
    dVar from A∘A / B∘B, dCor = dCov / √(dVar_x · dVar_y). Returns dCor ∈ [0, 1].
    """
    Dx = np.sqrt(Asq_x)
    Dy = np.sqrt(Asq_y)

    def _dc(D: np.ndarray) -> np.ndarray:
        row = D.mean(0, keepdims=True)
        col = D.mean(1, keepdims=True)
        tot = D.mean()
        return D - row - col + tot

    A = _dc(Dx)
    B = _dc(Dy)
    dcov2 = float(np.mean(A * B))
    dvarx = float(np.mean(A * A))
    dvary = float(np.mean(B * B))
    denom = np.sqrt(dvarx * dvary)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, dcov2)) / np.sqrt(denom))


def _dependence_reject(
    X: np.ndarray, Y: np.ndarray, n_perm: int, rng: np.random.Generator, stat: str
) -> bool:
    """Permutation test: reject independence iff observed stat > (1−α) permuted quantile.

    Permutes the Y rows to break the X↔Y pairing (the exact independence null) and
    builds the null distribution of HSIC or dCor. The squared-distance matrices are
    precomputed once; permutation only reindexes Y's matrix. Returns True = reject.
    """
    Xsq = _pairwise_sq(X)
    Ysq = _pairwise_sq(Y)
    fn = _hsic_stat if stat == "hsic" else _distance_correlation
    obs = fn(Xsq, Ysq)
    n = X.shape[0]
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(n)
        null[b] = fn(Xsq, Ysq[np.ix_(p, p)])
    thresh = np.quantile(null, 1.0 - ALPHA)
    return bool(obs > thresh)


# ── generator: linear map + residual noise at signal fraction s ─────────────────


def _calibrate_generator(data: dict, layer: int, pca_dim: int = PCA_DIM) -> dict:
    """Fit M̂ (standardized c_C top-PCs → v0 top-PCs) + residual sd, on all real rows.

    Standardize c_C, PCA to top-INPUT_PCA coords; PCA v0 to top-`pca_dim`. Fit the
    ridge map (median-PRESS λ) and record the per-PC residual sd (the noise floor).
    The real standardized-input coords are the resample cloud for the generator.
    """
    C, V = _layer_slice(data, layer)
    N = C.shape[0]

    xmu = C.mean(0)
    xsd = C.std(0) + 1e-9
    Cn = (C - xmu) / xsd
    _, xcomps, _ = robust_pca_basis(Cn, INPUT_PCA)
    cmu = Cn.mean(0)
    Xin = (Cn - cmu) @ xcomps.T  # (N, k_in)
    k_in = Xin.shape[1]

    vmu, vcomps, _ = robust_pca_basis(V, pca_dim)
    Z = (V - vmu) @ vcomps.T  # (N, k_t)
    k_t = Z.shape[1]

    XtX = Xin.T @ Xin
    best = None
    for lam in RIDGE_LAMBDAS:
        Minv = np.linalg.solve(XtX + lam * np.eye(k_in), Xin.T)
        W = Minv @ Z
        Hmat = Xin @ Minv
        hdiag = np.clip(np.diag(Hmat), None, 1 - 1e-9)
        resid = (Z - Xin @ W) / (1 - hdiag)[:, None]
        press = float(np.mean(resid**2))
        if best is None or press < best[0]:
            best = (press, lam, W)
    _press, lam_pick, M = best
    resid = Z - Xin @ M
    resid_sd = resid.std(0)  # (k_t,)
    return {
        "layer": layer,
        "Xin": Xin,
        "k_in": k_in,
        "M": M,
        "k_t": k_t,
        "resid_sd": resid_sd,
        "lambda_pick": float(lam_pick),
        "N_real": N,
        "signal_var": float(np.var(Xin @ M)),
    }


def _draw_contexts(gen: dict, n: int, rng: np.random.Generator, jitter_frac: float) -> np.ndarray:
    """n synthetic standardized-input contexts by resample-with-jitter of the real cloud."""
    Xin = gen["Xin"]
    N = Xin.shape[0]
    idx = rng.integers(0, N, size=n)
    col_sd = Xin.std(0)
    jit = rng.standard_normal((n, Xin.shape[1])) * (jitter_frac * col_sd)[None, :]
    return Xin[idx] + jit


def _simulate_linear(
    gen: dict, n: int, s: float, rng: np.random.Generator, jitter_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (X, Z): Z = α·(X@M̂) + noise, signal fraction = s.

    s = Var(signal)/Var(signal+noise) in the target PC space. With the noise floor
    fixed at the real per-PC residual sd, scale the signal by α so the realized
    fraction matches s: α²·var_sig·(1−s) = s·var_noise → α = √(s·var_noise /
    (var_sig·(1−s))). s=0 ⇒ α=0 ⇒ Z is pure noise, independent of X (the null).
    """
    X = _draw_contexts(gen, n, rng, jitter_frac)
    sig = X @ gen["M"]
    var_sig = float(np.var(sig)) + 1e-30
    var_noise = float(np.mean(gen["resid_sd"] ** 2))
    alpha = 0.0 if s <= 0.0 else float(np.sqrt(s * var_noise / (var_sig * (1.0 - s))))
    noise = rng.standard_normal((n, gen["k_t"])) * gen["resid_sd"][None, :]
    Z = alpha * sig + noise
    return X, Z


def run_hsic_power(
    data: dict,
    layer: int,
    ns: list[int],
    ss: list[float],
    n_datasets: int,
    n_perm: int,
    jitter_frac: float,
) -> dict:
    """HSIC + dCor dependence-test power vs (n, s); calibration at s=0; n*(s) at 80%."""
    gen = _calibrate_generator(data, layer)
    logger.info(
        "[hsic] calibrated L%d: k_in=%d k_t=%d λ=%.3g signal_var=%.4g resid_sd[mean]=%.4g",
        layer,
        gen["k_in"],
        gen["k_t"],
        gen["lambda_pick"],
        gen["signal_var"],
        float(gen["resid_sd"].mean()),
    )
    ss_seq = np.random.SeedSequence(SEED + 101)
    cell_seeds = ss_seq.spawn(len(ns) * len(ss))
    power_hsic: dict[str, dict[str, float]] = {}
    power_dcor: dict[str, dict[str, float]] = {}
    ci = 0
    for n in ns:
        power_hsic[str(n)] = {}
        power_dcor[str(n)] = {}
        for s in ss:
            rng = np.random.default_rng(cell_seeds[ci])
            ci += 1
            rh = rd = 0
            for _ in range(n_datasets):
                X, Z = _simulate_linear(gen, n, s, rng, jitter_frac)
                # compact target dim for a well-posed test null at all n
                kt = min(TARGET_TEST_PCA, Z.shape[1])
                zmu = Z.mean(0)
                _, zc, _ = robust_pca_basis(Z, kt)
                Zd = (Z - zmu) @ zc.T
                rh += int(_dependence_reject(X, Zd, n_perm, rng, "hsic"))
                rd += int(_dependence_reject(X, Zd, n_perm, rng, "dcor"))
            power_hsic[str(n)][f"{s:g}"] = rh / n_datasets
            power_dcor[str(n)][f"{s:g}"] = rd / n_datasets
            logger.info(
                "[hsic] n=%3d s=%.2f  HSIC=%.3f  dCor=%.3f", n, s, rh / n_datasets, rd / n_datasets
            )

    def _nstar(power: dict) -> dict:
        out: dict[str, Any] = {}
        for s in ss:
            sk = f"{s:g}"
            out[sk] = next((n for n in ns if power[str(n)][sk] >= POWER_TARGET), None)
        return out

    return {
        "layer": layer,
        "ns": ns,
        "ss": ss,
        "alpha": ALPHA,
        "power_target": POWER_TARGET,
        "n_datasets": n_datasets,
        "n_perm": n_perm,
        "generator": {
            "k_in": gen["k_in"],
            "k_t": gen["k_t"],
            "lambda_pick": gen["lambda_pick"],
            "signal_var": gen["signal_var"],
            "resid_sd_mean": float(gen["resid_sd"].mean()),
            "jitter_frac": jitter_frac,
            "target_test_pca": TARGET_TEST_PCA,
            "dependence_form": "linear M̂ + per-PC Gaussian noise, signal fraction s",
        },
        "power_hsic": power_hsic,
        "power_dcor": power_dcor,
        "n_star_hsic": _nstar(power_hsic),
        "n_star_dcor": _nstar(power_dcor),
        "calibration_s0_hsic": {str(n): power_hsic[str(n)].get("0") for n in ns},
        "calibration_s0_dcor": {str(n): power_dcor[str(n)].get("0") for n in ns},
    }


# ── Gaussian-MI from shrinkage CCA ──────────────────────────────────────────────


def _reg_cca_canon_corr(X: np.ndarray, Y: np.ndarray, reg: float) -> np.ndarray:
    """Canonical correlations of (X, Y) via RIDGE-regularized CCA.

    The within-set covariances are ridged toward a scaled identity before whitening:
    Σ̃ = Σ + reg·(trΣ/d)·I. This is the regularization that ACTUALLY tames the
    n ≪ (d_x+d_y) interpolation edge — a shrink-toward-diagonal of an
    already-standardized covariance is a no-op (its diagonal ≈ I), but the
    trace-scaled ridge shifts every whitening eigenvalue away from 0, so spurious
    canonical correlations are pulled below 1. Canonical correlations = singular
    values of the ridge-whitened cross-covariance Σ̃_xx^{-1/2} Σ_xy Σ̃_yy^{-1/2}.
    reg=0 recovers raw CCA (ρ→1 at small n; the degenerate-MI baseline). Returns
    the sorted-descending ρ_i clipped to [0, 1−1e-6] (the clip bounds the
    Gaussian-MI even in the raw reg=0 degenerate case).
    """
    n = X.shape[0]
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    dx, dy = X.shape[1], Y.shape[1]
    Sxx = (Xc.T @ Xc) / (n - 1)
    Syy = (Yc.T @ Yc) / (n - 1)
    Sxy = (Xc.T @ Yc) / (n - 1)
    tx = np.trace(Sxx) / dx
    ty = np.trace(Syy) / dy
    Sxx = Sxx + (reg * tx + 1e-9) * np.eye(dx)
    Syy = Syy + (reg * ty + 1e-9) * np.eye(dy)

    def _inv_sqrt(S: np.ndarray) -> np.ndarray:
        w, U = np.linalg.eigh(S)
        w = np.clip(w, 1e-12, None)
        return U @ np.diag(1.0 / np.sqrt(w)) @ U.T

    T = _inv_sqrt(Sxx) @ Sxy @ _inv_sqrt(Syy)
    sv = np.linalg.svd(T, compute_uv=False)
    return np.clip(sv, 0.0, 1.0 - 1e-6)


def _gaussian_mi_from_rho(rho: np.ndarray) -> float:
    """Gaussian mutual information −½ Σ log(1 − ρ_i²) (nats)."""
    return float(-0.5 * np.sum(np.log(np.clip(1.0 - rho**2, 1e-12, 1.0))))


def _ksg_proxy_mi(X: np.ndarray, Z: np.ndarray, k: int, rng: np.random.Generator) -> float:
    """Rough KSG-style MI proxy: Σ_t mutual_info_regression(X, Z[:,t]) (nats).

    sklearn's mutual_info_regression is a KSG/Kozachenko-Leonenko k-NN estimator of
    I(X; scalar). Summing over target PCs OVERCOUNTS shared information between target
    dims (the PCs are decorrelated, not independent), so this is a ROUGH PROXY /
    upper-ish reference, NOT a true joint MI — reported as such. Random-state pinned.
    """
    total = 0.0
    for t in range(Z.shape[1]):
        mi = mutual_info_regression(
            X, Z[:, t], n_neighbors=k, random_state=int(rng.integers(0, 2**31 - 1))
        )
        total += float(np.sum(mi))
    return total


def run_cca_gaussian_mi_curve(
    data: dict,
    layer: int,
    nsub_grid: list[int],
    n_resamples: int,
    reg: float,
    ridge_resamples: int,
) -> dict:
    """Ridge-CCA Gaussian-MI + linear-ridge skill R² learning curves vs n_sub.

    On the REAL layer data: subsample n_sub contexts (R resamples), reduce c_C to
    top-INPUT_PCA + v0 to top-CCA_TARGET_PCA (per resample, capped at the subsample
    rank so the CCA whitening is defined), and report per n_sub:
      (i)   Gaussian-MI from RIDGE CCA at the headline `reg` (+ resample CI);
      (ii)  Gaussian-MI across the full CCA_REG_SWEEP (mean per reg) so the
            MI-vs-regularization dependence is explicit — the honest read that the
            absolute MI is reg-dependent, not a single trustworthy number;
      (iii) the top canonical correlation at `reg` (and raw reg=0) — shows the
            raw-CCA ρ→1 interpolation-edge saturation directly;
      (iv)  the kept linear-ridge skill-over-mean R² (LOCO, per-fold-PCA target,
            full PCA_DIM=48), the cheap shared "is the linear MAP data-limited?"
            context — computed on a SEPARATE (smaller) ridge_resamples budget since
            its per-fold LOCO is the run bottleneck.
    The Gaussian-MI is the CHEAP part (one CCA per resample) so it carries the full
    `n_resamples`; the ridge-skill R² is the EXPENSIVE part and is averaged over
    `ridge_resamples` ≤ n_resamples resamples (its own CI).
    """
    C, V = _layer_slice(data, layer)
    N = C.shape[0]
    rng = np.random.default_rng(SEED + 202)
    rows = []
    for nsub in nsub_grid:
        if nsub > N:
            continue
        n_res = 1 if nsub == N else n_resamples
        rr = 1 if nsub == N else min(ridge_resamples, n_res)
        gmi_reg, gmi_raw, top_rho, top_rho_raw = [], [], [], []
        gmi_sweep: dict[float, list[float]] = {r: [] for r in CCA_REG_SWEEP}
        ridge_r2 = []
        k_in = min(INPUT_PCA, max(1, nsub - 1))
        k_t = min(CCA_TARGET_PCA, max(1, nsub - 1))
        n_rho = min(k_in, k_t)  # # of canonical correlations
        spec_reg = np.zeros(n_rho)  # accumulate the full ρ spectrum (reg + raw)
        spec_raw = np.zeros(n_rho)
        for ridx in range(n_res):
            idx = np.arange(N) if nsub == N else rng.choice(N, size=nsub, replace=False)
            Cs, Vs = C[idx], V[idx]
            Cn = (Cs - Cs.mean(0)) / (Cs.std(0) + 1e-9)
            _, xc, _ = robust_pca_basis(Cn, k_in)
            Xin = (Cn - Cn.mean(0)) @ xc.T
            vmu, vc, _ = robust_pca_basis(Vs, k_t)
            Z = (Vs - vmu) @ vc.T
            rho = _reg_cca_canon_corr(Xin, Z, reg)
            rho_raw = _reg_cca_canon_corr(Xin, Z, 0.0)
            gmi_reg.append(_gaussian_mi_from_rho(rho))
            gmi_raw.append(_gaussian_mi_from_rho(rho_raw))
            top_rho.append(float(rho[0]))
            top_rho_raw.append(float(rho_raw[0]))
            spec_reg += rho[:n_rho]
            spec_raw += rho_raw[:n_rho]
            for rr_reg in CCA_REG_SWEEP:
                gmi_sweep[rr_reg].append(_gaussian_mi_from_rho(_reg_cca_canon_corr(Xin, Z, rr_reg)))
            if ridx < rr:  # expensive ridge-skill on a capped resample budget
                ridge_r2.append(_ridge_skill_pca_loco(Cs, Vs))
        spec_reg /= n_res
        spec_raw /= n_res
        rows.append(
            {
                "n_sub": nsub,
                "n_resamples": n_res,
                "ridge_resamples": rr,
                "gaussian_mi_reg_mean": float(np.mean(gmi_reg)),
                "gaussian_mi_reg_ci": [
                    float(np.percentile(gmi_reg, 2.5)),
                    float(np.percentile(gmi_reg, 97.5)),
                ],
                "gaussian_mi_raw_mean": float(np.mean(gmi_raw)),
                "gaussian_mi_by_reg_mean": {
                    f"{r:g}": float(np.mean(v)) for r, v in gmi_sweep.items()
                },
                "top_canon_corr_reg_mean": float(np.mean(top_rho)),
                "top_canon_corr_raw_mean": float(np.mean(top_rho_raw)),
                "ridge_skill_r2_mean": float(np.mean(ridge_r2)),
                "ridge_skill_r2_ci": [
                    float(np.percentile(ridge_r2, 2.5)),
                    float(np.percentile(ridge_r2, 97.5)),
                ],
                "canon_corr_spectrum_reg_mean": [float(x) for x in spec_reg],
                "canon_corr_spectrum_raw_mean": [float(x) for x in spec_raw],
                "k_in": k_in,
                "k_t": k_t,
                "n_canon_corr": n_rho,
            }
        )
        logger.info(
            "[cca L%d n=%2d] G-MI(reg=%.1f)=%.3f raw=%.3f  ρ1=%.3f(raw %.3f)  ridgeR²=%+.4f "
            "(R=%d ridgeR=%d)",
            layer,
            nsub,
            reg,
            rows[-1]["gaussian_mi_reg_mean"],
            rows[-1]["gaussian_mi_raw_mean"],
            rows[-1]["top_canon_corr_reg_mean"],
            rows[-1]["top_canon_corr_raw_mean"],
            rows[-1]["ridge_skill_r2_mean"],
            n_res,
            rr,
        )
    # stability: relative change in reg-MI over the last two grid points
    stab = None
    if len(rows) >= 2:
        a, b = rows[-2]["gaussian_mi_reg_mean"], rows[-1]["gaussian_mi_reg_mean"]
        stab = abs(b - a) / (abs(b) + 1e-9)
    # FIRST-CLASS: the ridge-R²-vs-n learning curve (is the LINEAR MAP data-limited?).
    ns = np.array([r["n_sub"] for r in rows], dtype=np.float64)
    r2 = np.array([r["ridge_skill_r2_mean"] for r in rows], dtype=np.float64)
    return {
        "layer": layer,
        "nsub_grid": nsub_grid,
        "n_resamples": n_resamples,
        "cca_reg": reg,
        "cca_reg_sweep": CCA_REG_SWEEP,
        "input_pca": INPUT_PCA,
        "cca_target_pca": CCA_TARGET_PCA,
        "ridge_skill_target_pca": PCA_DIM,
        "rows": rows,
        "reg_mi_rel_change_last_step": stab,
        "ridge_learning_curve_fit": _learning_curve_fit(ns, r2),
    }


def _learning_curve_fit(ns: np.ndarray, r2: np.ndarray) -> dict:
    """Fit R²(n) ≈ R∞ − a/n by OLS; report R∞ + the n to get within 0.02 of R∞.

    Linear in (R∞, a): R² = R∞ − a·(1/n). `n_within_0p02` = ⌈a/0.02⌉ (the n where the
    asymptotic gap a/n ≤ 0.02), defined only when a>0 (curve rising toward R∞). The
    observed-window trend (slope of R² vs n over the top grid points) labels
    rising / flat / falling so the headline read is unambiguous even if the
    parametric fit is shaky.
    """
    mask = np.isfinite(r2)
    ns_f, r2_f = ns[mask], r2[mask]
    if ns_f.size < 2:
        return {"r_inf": float("nan"), "a": float("nan"), "n_within_0p02": None, "trend": "n/a"}
    A = np.column_stack([np.ones_like(ns_f), -1.0 / ns_f])
    coef, *_ = np.linalg.lstsq(A, r2_f, rcond=None)
    r_inf, a = float(coef[0]), float(coef[1])
    tail = ns_f.size if ns_f.size < 4 else 4
    slope = float(np.polyfit(ns_f[-tail:], r2_f[-tail:], 1)[0])
    trend = "rising" if slope > 1e-3 else ("falling" if slope < -1e-3 else "flat")
    n_within = int(np.ceil(a / 0.02)) if a > 0 else None
    return {
        "r_inf": r_inf,
        "a": a,
        "n_within_0p02": n_within,
        "obs_slope_per_ctx": slope,
        "trend": trend,
        "r2_at_max_n": float(r2_f[-1]),
        "n_max": int(ns_f[-1]),
    }


def _ridge_skill_pca_loco(Xc: np.ndarray, Yv: np.ndarray, pca_dim: int = PCA_DIM) -> float:
    """Linear-ridge skill-over-mean with the top-`pca_dim` PCA target fit per fold.

    Per held-out context i: PCA-reduce v0 on TRAIN rows only (no leakage), fit the
    train-mean-centered ridge map c_C → PC scores on the n-1 train rows with ONE
    nested-PRESS λ pick + ONE dual solve (NOT a nested LOCO-inside-LOCO — that was
    O(n²) and dominated the run wall-time on the contended VM), predict the held-out
    PC scores, reconstruct full-H v0, and score skill-over-mean (#658's exact
    dual/PRESS math via `_press_loo_mse_per_lambda` / `_ridge_dual_weights`, the
    same recipe `ridge_predict_loco_centered` uses internally — this just avoids
    re-running the whole LOCO per outer fold). Train-only X standardization +
    target centering, matching the canonical centered-ridge skill.
    """
    n = Xc.shape[0]
    H = Yv.shape[1]
    device = torch.device(_i658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(Xc)).to(device=device, dtype=torch.float64)
    preds = np.zeros((n, H), dtype=np.float64)
    k_use = min(pca_dim, max(1, n - 2))
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Ytr = Yv[tr]
        mu, comps, _ = robust_pca_basis(Ytr, k_use)  # train-only PCA target basis
        Ztr = (Ytr - mu) @ comps.T  # (n-1, k) train PC scores
        Ztr_t = torch.from_numpy(np.ascontiguousarray(Ztr)).to(device=device, dtype=torch.float64)
        Xtr = Xt[torch.tensor(tr, device=device)]
        xmu = Xtr.mean(0)
        xsd = Xtr.std(0, correction=0) + 1e-9  # #658 ddof=0 convention
        Xtr_n = (Xtr - xmu) / xsd
        zmu = Ztr_t.mean(0)  # train PC-score mean (the predict-the-mean baseline)
        Ztr_c = Ztr_t - zmu
        mse = _press_loo_mse_per_lambda(Xtr_n, Ztr_c, RIDGE_LAMBDAS)
        best_lam = RIDGE_LAMBDAS[int(torch.argmin(mse).item())]
        w = _ridge_dual_weights(Xtr_n, Ztr_c, best_lam)  # (d, k)
        x_held = (Xt[i] - xmu) / xsd
        zhat = (zmu + x_held @ w).detach().cpu().numpy()  # held-out PC prediction
        preds[i] = mu + zhat @ comps  # reconstruct full-H v0
    return float(skill_over_mean_r2(preds, Yv)["skill"])


# ── Deliverable 2: real-data MI reliability + ground-truth bias sim ─────────────


def run_mi_reliability_real(
    data: dict, layer: int, nsub_grid: list[int], n_resamples: int, reg: float
) -> dict:
    """Real-data Gaussian-MI (ridge CCA) + KSG-proxy MI vs n_sub, resample CIs.

    Leads with the SATURATION diagnostic: the top canonical correlation at the raw
    (reg=0) CCA — at n ≈ dims it is spuriously ~1, which blows the Gaussian-MI sum up
    to ~100 nats (the CCA analogue of cosine saturation), so the MI MAGNITUDE is NOT
    a trustworthy number at n≤50. We therefore report ρ1_raw (the saturation flag),
    the regularized ρ1, the regularized Gaussian-MI (a regularization-dependent
    surrogate, not the true MI), and the KSG proxy (which grows with n = small-sample
    bias). The honest verdict is set by these together, not the MI number alone.
    """
    C, V = _layer_slice(data, layer)
    N = C.shape[0]
    rng = np.random.default_rng(SEED + 303)
    rows = []
    for nsub in nsub_grid:
        if nsub > N:
            continue
        gmi, ksg, rho1_raw, rho1_reg = [], [], [], []
        n_res = 1 if nsub == N else n_resamples
        ksg_res = min(n_res, 12)  # KSG proxy is slower; cap its resamples
        for r in range(n_res):
            idx = np.arange(N) if nsub == N else rng.choice(N, size=nsub, replace=False)
            Cs, Vs = C[idx], V[idx]
            k_in = min(INPUT_PCA, max(1, nsub - 1))
            k_t = min(CCA_TARGET_PCA, max(1, nsub - 1))
            Cn = (Cs - Cs.mean(0)) / (Cs.std(0) + 1e-9)
            _, xc, _ = robust_pca_basis(Cn, k_in)
            Xin = (Cn - Cn.mean(0)) @ xc.T
            vmu, vc, _ = robust_pca_basis(Vs, k_t)
            Z = (Vs - vmu) @ vc.T
            rho = _reg_cca_canon_corr(Xin, Z, reg)
            rho_raw = _reg_cca_canon_corr(Xin, Z, 0.0)
            gmi.append(_gaussian_mi_from_rho(rho))
            rho1_reg.append(float(rho[0]))
            rho1_raw.append(float(rho_raw[0]))
            if r < ksg_res:
                ksg.append(_ksg_proxy_mi(Xin, Z[:, : min(8, Z.shape[1])], KSG_K, rng))
        rows.append(
            {
                "n_sub": nsub,
                "n_resamples": n_res,
                "top_canon_corr_raw_mean": float(np.mean(rho1_raw)),
                "top_canon_corr_reg_mean": float(np.mean(rho1_reg)),
                "gaussian_mi_reg_mean": float(np.mean(gmi)),
                "gaussian_mi_reg_ci": [
                    float(np.percentile(gmi, 2.5)),
                    float(np.percentile(gmi, 97.5)),
                ],
                "ksg_proxy_mi_mean": float(np.mean(ksg)) if ksg else float("nan"),
                "ksg_proxy_mi_ci": (
                    [float(np.percentile(ksg, 2.5)), float(np.percentile(ksg, 97.5))]
                    if len(ksg) >= 2
                    else [float("nan"), float("nan")]
                ),
                "ksg_n_resamples": ksg_res,
            }
        )
        logger.info(
            "[mi-real L%d n=%2d] ρ1_raw=%.4f ρ1_reg=%.3f  G-MI(reg)=%.3f [%.2f,%.2f]  KSG=%.3f",
            layer,
            nsub,
            rows[-1]["top_canon_corr_raw_mean"],
            rows[-1]["top_canon_corr_reg_mean"],
            rows[-1]["gaussian_mi_reg_mean"],
            rows[-1]["gaussian_mi_reg_ci"][0],
            rows[-1]["gaussian_mi_reg_ci"][1],
            rows[-1]["ksg_proxy_mi_mean"],
        )
    return {
        "layer": layer,
        "nsub_grid": nsub_grid,
        "cca_reg": reg,
        "cca_target_pca": CCA_TARGET_PCA,
        "input_pca": INPUT_PCA,
        "ksg_k": KSG_K,
        "rows": rows,
        "verdict": (
            "Gaussian-MI MAGNITUDE not reliably estimable at n<=50: raw canonical "
            "correlations saturate to ~1, so the MI sum is regularization-dependent "
            "(see gaussian_mi_reg_mean across reg) and not a true MI value; the KSG "
            "proxy is small-sample-bias-dominated (grows with n). HSIC/dCor presence "
            "tests (Deliverable 1) are the reliable surrogate at these n."
        ),
    }


def run_mi_ground_truth_sim(
    ns: list[int], dim: int, rhos: list[float], n_datasets: int, reg: float
) -> dict:
    """Bias+variance of the Gaussian-MI estimator vs n at a KNOWN ground-truth MI.

    Joint Gaussian: X ∈ R^dim, Y ∈ R^dim with `dim` independent canonical pairs at
    correlations `rhos[:dim]` (the rest 0). True MI = −½ Σ log(1−ρ_i²). For each n,
    simulate B datasets, estimate via RIDGE CCA (reg) AND raw CCA (reg=0), report
    mean/sd/bias + the mean top canonical correlation (the saturation read).
    n_within_20pct = smallest n whose |mean_estimate − true| ≤ 0.20·true for the REG
    estimator (the practical one). Also reports the KSG proxy on the sim for a
    feasibility read on full nonparametric MI at this dim. Note the reg estimator is
    a regularized (downward-biased) surrogate — even with a KNOWN truth it converges
    only at large n; that IS the point (it quantifies how untrustworthy the n=50 MI
    is).
    """
    rho_vec = np.array(rhos[:dim], dtype=np.float64)
    if rho_vec.size < dim:
        rho_vec = np.concatenate([rho_vec, np.zeros(dim - rho_vec.size)])
    true_mi = _gaussian_mi_from_rho(rho_vec)
    true_rho1 = float(np.max(rho_vec))
    ss = np.random.SeedSequence(SEED + 404)
    cell_seeds = ss.spawn(len(ns))
    rows = []
    for n, cs in zip(ns, cell_seeds, strict=True):
        rng = np.random.default_rng(cs)
        est_reg, est_raw, est_ksg, top_raw = [], [], [], []
        ksg_cap = min(n_datasets, 20)
        for b in range(n_datasets):
            # sample canonical pairs: X_i ~ N(0,1), Y_i = ρ_i X_i + √(1−ρ²) ε_i
            X = rng.standard_normal((n, dim))
            eps = rng.standard_normal((n, dim))
            Y = rho_vec[None, :] * X + np.sqrt(1.0 - rho_vec**2)[None, :] * eps
            rr = _reg_cca_canon_corr(X, Y, reg)
            rraw = _reg_cca_canon_corr(X, Y, 0.0)
            est_reg.append(_gaussian_mi_from_rho(rr))
            est_raw.append(_gaussian_mi_from_rho(rraw))
            top_raw.append(float(rraw[0]))
            if b < ksg_cap:
                est_ksg.append(_ksg_proxy_mi(X, Y, KSG_K, rng))
        rows.append(
            {
                "n": n,
                "true_mi": true_mi,
                "true_top_rho": true_rho1,
                "reg_mean": float(np.mean(est_reg)),
                "reg_sd": float(np.std(est_reg)),
                "reg_bias": float(np.mean(est_reg) - true_mi),
                "raw_mean": float(np.mean(est_raw)),
                "raw_bias": float(np.mean(est_raw) - true_mi),
                "top_canon_corr_raw_mean": float(np.mean(top_raw)),
                "ksg_proxy_mean": float(np.mean(est_ksg)) if est_ksg else float("nan"),
                "ksg_proxy_sd": float(np.std(est_ksg)) if len(est_ksg) >= 2 else float("nan"),
            }
        )
        logger.info(
            "[mi-sim n=%3d] true=%.3f reg=%.3f(±%.3f b%+.3f) raw=%.3f(ρ1raw=%.3f) KSG=%.3f",
            n,
            true_mi,
            rows[-1]["reg_mean"],
            rows[-1]["reg_sd"],
            rows[-1]["reg_bias"],
            rows[-1]["raw_mean"],
            rows[-1]["top_canon_corr_raw_mean"],
            rows[-1]["ksg_proxy_mean"],
        )
    tol = 0.20 * abs(true_mi)
    n_within = next((r["n"] for r in rows if abs(r["reg_mean"] - true_mi) <= tol), None)
    return {
        "dim": dim,
        "rhos": rho_vec.tolist(),
        "true_mi": true_mi,
        "ns": ns,
        "n_datasets": n_datasets,
        "cca_reg": reg,
        "ksg_k": KSG_K,
        "rows": rows,
        "n_within_20pct_reg": n_within,
        "tolerance_nats": tol,
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
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _meta_block(data: dict, wall_s: float, extra: dict) -> dict:
    return {
        "issue": 722,
        "followup_label": "context_count_infotheoretic",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha": _git_sha(),
        "seed": SEED,
        "store_provenance": data["store_provenance"],
        "v0_local_sha256": _file_sha256(data["v0_path"]),
        "cc_local_sha256": _file_sha256(data["cc_path"]),
        "context_families": data["families"],
        "wall_time_s": round(wall_s, 2),
        **extra,
    }


# ── figures ──────────────────────────────────────────────────────────────────


def _meta_sidecar(fig_path: Path, payload: dict) -> None:
    _atomic_write_json(
        fig_path.with_suffix(".meta.json"),
        {
            "issue": 722,
            "figure": fig_path.name,
            "code_sha": _git_sha(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **payload,
        },
    )


def make_hsic_power_fig(pw: dict, fig_path: Path) -> None:
    """Power vs n, one line per signal fraction s; HSIC solid + dCor dashed."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")
    ns, ss = pw["ns"], pw["ss"]
    palette = ["#999999", "#56B4E9", "#0072B2", "#D55E00", "#CC0000"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i, s in enumerate(ss):
        sk = f"{s:g}"
        col = palette[i % len(palette)]
        yh = [pw["power_hsic"][str(n)][sk] for n in ns]
        yd = [pw["power_dcor"][str(n)][sk] for n in ns]
        lbl = f"s={s:g}" + (" (null)" if s == 0.0 else "")
        ax.plot(ns, yh, marker="o", ms=3.5, lw=1.7, color=col, label=lbl)
        ax.plot(ns, yd, marker="^", ms=3.0, lw=1.1, ls="--", color=col, alpha=0.8)
    ax.axhline(pw["power_target"], color="0.4", lw=0.9, ls="--")
    ax.axhline(pw["alpha"], color="0.7", lw=0.9, ls=":")
    ax.set_xlabel("n contexts")
    ax.set_ylabel("rejection rate (power)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"dependence-test power vs n (L{pw['layer']}; HSIC solid, dCor dashed)")
    ax.legend(fontsize=7, loc="center right", title="signal fraction")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    _meta_sidecar(
        fig_path,
        {
            "source_json": "eval_results/issue_722/context_count_infotheoretic/hsic_power.json",
            "lines": [f"HSIC + dCor power vs n at s={s:g}" for s in ss]
            + ["80%-power + α reference"],
        },
    )
    logger.info("wrote %s", fig_path)


def make_mi_bias_fig(sim: dict, fig_path: Path) -> None:
    """MI-estimator bias vs n at a known ground-truth MI: reg CCA / raw / KSG."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")
    rows = sim["rows"]
    ns = [r["n"] for r in rows]
    true = sim["true_mi"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhline(true, color="0.3", lw=1.2, ls="-", label=f"true MI = {true:.2f} nats")
    sh = [r["reg_mean"] for r in rows]
    shsd = [r["reg_sd"] for r in rows]
    rw = [r["raw_mean"] for r in rows]
    ks = [r["ksg_proxy_mean"] for r in rows]
    ax.errorbar(
        ns,
        sh,
        yerr=shsd,
        marker="o",
        ms=3.5,
        lw=1.6,
        color="#0072B2",
        label=f"ridge CCA (reg={sim['cca_reg']:g})",
    )
    ax.plot(ns, rw, marker="s", ms=3.0, lw=1.2, ls="--", color="#D55E00", label="raw CCA (biased)")
    ax.plot(ns, ks, marker="^", ms=3.0, lw=1.2, ls=":", color="#009E73", label="KSG proxy")
    band = sim["tolerance_nats"]
    ax.fill_between(
        [min(ns), max(ns)], true - band, true + band, color="0.5", alpha=0.12, label="±20% band"
    )
    ax.set_xlabel("n samples")
    ax.set_ylabel("estimated MI (nats)")
    ax.set_title(f"Gaussian-MI estimator bias vs n (ground truth, dim={sim['dim']})")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    _meta_sidecar(
        fig_path,
        {
            "source_json": "eval_results/issue_722/context_count_infotheoretic/mi_reliability.json",
            "lines": ["shrinkage-CCA MI ±sd", "raw CCA MI", "KSG proxy", "true MI", "±20% band"],
        },
    )
    logger.info("wrote %s", fig_path)


def make_cca_mi_fig(cca: dict, fig_path: Path) -> None:
    """3 panels: (a) CCA saturation (top-ρ raw vs reg), (b) Gaussian-MI reg-sweep,
    (c) the linear-ridge skill-R² learning curve + the R∞−a/n fit."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")
    rows = cca["rows"]
    ns = [r["n_sub"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))

    # (a) canonical-correlation saturation: top ρ raw vs regularized
    rho_raw = [r["top_canon_corr_raw_mean"] for r in rows]
    rho_reg = [r["top_canon_corr_reg_mean"] for r in rows]
    axes[0].plot(ns, rho_raw, marker="s", ms=3.5, lw=1.6, color="#D55E00", label="raw CCA ρ₁")
    axes[0].plot(
        ns,
        rho_reg,
        marker="o",
        ms=3.5,
        lw=1.6,
        color="#0072B2",
        label=f"ridge CCA ρ₁ (reg={cca['cca_reg']:g})",
    )
    axes[0].axhline(1.0, color="0.6", lw=0.8, ls=":")
    axes[0].set_xlabel("n contexts (subsample)")
    axes[0].set_ylabel("top canonical correlation ρ₁")
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_title("(a) CCA saturation: raw ρ₁→1 at small n")
    axes[0].legend(fontsize=7, loc="lower left")

    # (b) Gaussian-MI reg-sweep vs n
    palette = ["#CC0000", "#D55E00", "#0072B2", "#56B4E9"]
    for i, rr in enumerate(cca["cca_reg_sweep"]):
        ys = [r["gaussian_mi_by_reg_mean"][f"{rr:g}"] for r in rows]
        lbl = f"reg={rr:g}" + (" (raw)" if rr == 0.0 else "")
        axes[1].plot(ns, ys, marker="o", ms=3.0, lw=1.5, color=palette[i % len(palette)], label=lbl)
    axes[1].set_xlabel("n contexts (subsample)")
    axes[1].set_ylabel("Gaussian-MI (nats)")
    axes[1].set_yscale("symlog")
    axes[1].set_title("(b) Gaussian-MI is regularization-dependent")
    axes[1].legend(fontsize=7, loc="upper right")

    # (c) linear-ridge skill R² learning curve + R∞−a/n fit
    r2 = [r["ridge_skill_r2_mean"] for r in rows]
    r2lo = [r["ridge_skill_r2_ci"][0] for r in rows]
    r2hi = [r["ridge_skill_r2_ci"][1] for r in rows]
    axes[2].plot(ns, r2, marker="^", ms=3.5, lw=1.7, color="#009E73", label="ridge skill R²")
    axes[2].fill_between(ns, r2lo, r2hi, color="#009E73", alpha=0.15)
    fit = cca["ridge_learning_curve_fit"]
    if np.isfinite(fit["r_inf"]) and np.isfinite(fit["a"]):
        xx = np.linspace(min(ns), max(max(ns), (fit["n_within_0p02"] or max(ns))), 100)
        axes[2].plot(
            xx,
            fit["r_inf"] - fit["a"] / xx,
            lw=1.0,
            ls="--",
            color="0.4",
            label=f"R∞−a/n fit (R∞={fit['r_inf']:.3f})",
        )
        axes[2].axhline(fit["r_inf"], color="0.7", lw=0.8, ls=":")
    axes[2].set_xlabel("n contexts (subsample)")
    axes[2].set_ylabel("linear-ridge skill-over-mean R²")
    axes[2].set_title(f"(c) linear map vs n ({fit['trend']})")
    axes[2].legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    _meta_sidecar(
        fig_path,
        {
            "source_json": "eval_results/issue_722/context_count_infotheoretic/"
            "cca_gaussianMI_curve.json",
            "panels": [
                "(a) top canonical correlation ρ₁: raw (saturates→1) vs ridge-regularized",
                "(b) Gaussian-MI vs n across the regularization sweep (symlog)",
                "(c) linear-ridge skill-over-mean R² vs n + the R∞−a/n learning-curve fit",
            ],
        },
    )
    logger.info("wrote %s", fig_path)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:  # noqa: C901 — flat driver: three independent analysis blocks + summary
    parser = argparse.ArgumentParser(description="Issue #722 context-count info-theory power.")
    parser.add_argument("--smoke", action="store_true", help="tiny grid/B to prove it runs")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_722" / "context_count_infotheoretic",
    )
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_722")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--only", choices=["hsic", "mi", "cca"], default=None)
    args = parser.parse_args()

    _i658.DEVICE = "cpu"  # CPU-only, deterministic
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.smoke:
        hsic_ns = [15, 30, 75]
        hsic_ss = [0.0, 0.20]
        hsic_b, hsic_perm = 12, 40
        mi_nsub = [15, 30, 50]
        cca_nsub = [15, 30, 50]
        resamples = 6
        ridge_resamples = 2  # the expensive per-fold LOCO ridge-skill
        mi_sim_ns = [30, 100, 300]
        mi_sim_b = 12
    else:
        hsic_ns, hsic_ss = HSIC_NS, HSIC_SS
        hsic_b, hsic_perm = HSIC_B, HSIC_BPERM
        mi_nsub, cca_nsub = MI_NSUB, CCA_NSUB
        resamples = MI_RESAMPLES
        ridge_resamples = RIDGE_SKILL_RESAMPLES
        mi_sim_ns, mi_sim_b = MI_SIM_NS, MI_SIM_B

    t0 = time.time()
    data = _load_stores()
    logger.info(
        "loaded n=%d ctx, %d layers, H=%d | families=%s",
        data["V"].shape[0],
        len(data["layers"]),
        data["V"].shape[-1],
        data["families"],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hsic_res = mi_real = mi_sim = cca_res = None

    if args.only in (None, "hsic"):
        t = time.time()
        hsic_res = run_hsic_power(
            data, PRIMARY_LAYER, hsic_ns, hsic_ss, hsic_b, hsic_perm, JITTER_FRAC
        )
        out = {
            "analysis": "hsic_dcor_power",
            **hsic_res,
            "metadata": _meta_block(data, time.time() - t, {"analysis": "hsic_dcor_power"}),
        }
        name = "hsic_power_smoke.json" if args.smoke else "hsic_power.json"
        _atomic_write_json(args.out_dir / name, out)
        logger.info("wrote %s", args.out_dir / name)
        if not args.smoke:
            make_hsic_power_fig(hsic_res, args.fig_dir / "hsic_power_vs_n.png")

    if args.only in (None, "mi"):
        t = time.time()
        mi_real = run_mi_reliability_real(data, PRIMARY_LAYER, mi_nsub, resamples, CCA_REG)
        mi_sim = run_mi_ground_truth_sim(mi_sim_ns, MI_SIM_DIM, MI_SIM_RHOS, mi_sim_b, CCA_REG)
        out = {
            "analysis": "mi_reliability",
            "real_data": mi_real,
            "ground_truth_sim": mi_sim,
            "metadata": _meta_block(data, time.time() - t, {"analysis": "mi_reliability"}),
        }
        name = "mi_reliability_smoke.json" if args.smoke else "mi_reliability.json"
        _atomic_write_json(args.out_dir / name, out)
        logger.info("wrote %s", args.out_dir / name)
        if not args.smoke:
            make_mi_bias_fig(mi_sim, args.fig_dir / "mi_bias_vs_n.png")

    if args.only in (None, "cca"):
        t = time.time()
        cca_res = run_cca_gaussian_mi_curve(
            data, PRIMARY_LAYER, cca_nsub, resamples, CCA_REG, ridge_resamples
        )
        out = {
            "analysis": "cca_gaussian_mi_curve",
            **cca_res,
            "metadata": _meta_block(data, time.time() - t, {"analysis": "cca_gaussian_mi_curve"}),
        }
        name = "cca_gaussianMI_curve_smoke.json" if args.smoke else "cca_gaussianMI_curve.json"
        _atomic_write_json(args.out_dir / name, out)
        logger.info("wrote %s", args.out_dir / name)
        if not args.smoke:
            make_cca_mi_fig(cca_res, args.fig_dir / "cca_gaussianMI_vs_n.png")

    logger.info("DONE in %.1fs", time.time() - t0)

    # console summary
    if hsic_res is not None:
        print(f"\n=== D1: HSIC/dCor dependence-test power (L{PRIMARY_LAYER}) ===")
        print(f"calibration s=0 (should ≈ {ALPHA:.2f}):")
        for n in hsic_ns:
            print(
                f"   n={n:3d}: HSIC={hsic_res['power_hsic'][str(n)]['0']:.3f}  "
                f"dCor={hsic_res['power_dcor'][str(n)]['0']:.3f}"
            )
        print(f"n*(s) for {POWER_TARGET * 100:.0f}% power (HSIC / dCor):")
        for s in hsic_ss:
            if s == 0.0:
                continue
            print(
                f"   s={s:g}: HSIC n*={hsic_res['n_star_hsic'][f'{s:g}']}  "
                f"dCor n*={hsic_res['n_star_dcor'][f'{s:g}']}"
            )
    if mi_sim is not None:
        print("\n=== D2: MI estimation reliability ===")
        print(f"VERDICT: {mi_real['verdict']}")
        print(f"ground-truth sim (dim={mi_sim['dim']}, true MI={mi_sim['true_mi']:.3f} nats):")
        nw = mi_sim["n_within_20pct_reg"]
        print(f"   n within ±20% (ridge CCA reg={mi_sim['cca_reg']:g}) = {nw}")
        print("real-data CCA (L18) — saturation + regularized G-MI vs n_sub:")
        for r in mi_real["rows"]:
            print(
                f"   n={r['n_sub']:2d}: ρ1_raw={r['top_canon_corr_raw_mean']:.4f} "
                f"G-MI(reg)={r['gaussian_mi_reg_mean']:.3f} "
                f"[{r['gaussian_mi_reg_ci'][0]:.2f},{r['gaussian_mi_reg_ci'][1]:.2f}]  "
                f"KSG-proxy={r['ksg_proxy_mi_mean']:.3f}"
            )
    if cca_res is not None:
        print(f"\n=== D3: CCA Gaussian-MI + linear-map learning curve (L{PRIMARY_LAYER}) ===")
        fit = cca_res["ridge_learning_curve_fit"]
        print(
            f"linear-ridge R² learning curve: trend={fit['trend']}  R∞={fit['r_inf']:+.4f}  "
            f"a={fit['a']:+.4f}  n_to_within_0.02={fit['n_within_0p02']}  "
            f"(R²@n={fit['n_max']}={fit['r2_at_max_n']:+.4f})"
        )
        print(f"reg-MI rel-change last step = {cca_res['reg_mi_rel_change_last_step']}")
        for r in cca_res["rows"]:
            print(
                f"   n={r['n_sub']:2d}: ρ1_raw={r['top_canon_corr_raw_mean']:.4f} "
                f"ρ1_reg={r['top_canon_corr_reg_mean']:.3f}  "
                f"G-MI(reg)={r['gaussian_mi_reg_mean']:.3f} raw={r['gaussian_mi_raw_mean']:.3f}  "
                f"ridgeR²={r['ridge_skill_r2_mean']:+.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
