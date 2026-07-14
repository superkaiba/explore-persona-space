#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ᵀ, λ, ρ, ×, ⁻¹) in scientific docstrings + comments.
"""Issue #833 — batched twin of ``issue722_bootstrap.make_refit_pair`` (ridge floors).

The serial refit-floor battery (9 cells × 5 floors × 100 pairs × 2 refits = 9,000
serial ridge refits) is OVERHEAD-bound, not FLOP-bound: py-spy on the round-2 run
showed 62% of wall inside a per-resample ``np.linalg.svd`` of a ~592×3584 float64
matrix (``_pca_basis_v0``) and 37% inside the per-resample m×m PRESS eigh
(``fit658._press_loo_mse_per_lambda``) — ~40 s/pair, ~50 h projected vs plan §9's
~1 h (``.claude/rules/vectorize-many-cell-fits.md``; #778 pool-reduction pattern).

``make_refit_pair_batched`` computes the IDENTICAL estimator with every
per-resample factorization moved into n×n dual (Gram) space and batched over all
``2 × n_pairs`` resamples of a floor at once:

- **Same resample draws.** The ``np.random.default_rng(seed)`` consumption order
  is bit-identical to the serial loop (idx_a, idx_b, then the two unused
  ``rng.integers(0, 2**31-1)`` child-seed draws per pair — the ridge ``fit_fn``
  ignores its rng, so the child seeds only exist for stream parity).
- **Resample ≡ integer row weights.** A with-replacement resample of the FIXED
  n-row design is an integer weight vector w (``np.bincount``); every weighted
  moment / Gram below is exactly the duplicated-row computation.
- **PCA basis via the weighted dual Gram of the CACHED base Gram.** The serial
  path SVDs the centered resampled Y (m×3584) for its top-k right-singular
  subspace. The composite prediction (project Y on the top-k basis → isotropic
  ridge per coordinate → back-project) and the PRESS mean-MSE are invariant to
  WHICH orthonormal basis of that subspace is used, so the subspace is taken
  from ``eigh(W^{1/2} Yc Ycᵀ W^{1/2})`` where ``Yc Ycᵀ`` is the cached base Gram
  ``K_Y = Y Yᵀ`` plus rank-1 mean corrections — the #778 "reduce the pool once,
  batch every draw" identity. No per-resample 3584-dim SVD remains.
- **PRESS / dual ridge in n×n space.** With ``E`` the (m, n) row-selection
  matrix of a resample and ``G_u`` the standardized unique-row Gram, the
  push-through identity ``(E G_u Eᵀ + λI_m)⁻¹ E = E (λI_n + G_u W)⁻¹`` reduces
  the m×m hat-matrix diagonals, LOO residuals, and dual weights to ONE batched
  ``eigh`` of ``S = W^{1/2} G_u W^{1/2}`` reused across all λ — the same
  eigendecomposition-reuse structure as the serial ``_press_loo_mse_per_lambda``,
  one m→n space change. The per-resample per-dimension standardization (sd of
  the resampled rows) breaks the cached-Gram trick on the X side, so ``S`` is
  built by one batched GEMM per resample (cheap; it was never the bottleneck).
- **Exact-semantics fallback.** Any resample the batched path cannot certify —
  too few rows (m < target_dim + 2), a near-zero or near-tied eigenvalue at the
  top-k boundary (where the subspace itself is ill-conditioned and the
  basis-invariance argument degrades), or a batched ``eigh`` non-convergence —
  is recomputed through the SERIAL code path (``fitM._pca_basis_v0`` +
  ``fit658`` PRESS/dual-ridge, the exact calls ``_refit_ridge_fn`` makes). A
  ``np.linalg.LinAlgError`` in that fallback skips the PAIR, mirroring
  ``make_refit_pair``'s skip semantics (``skip_counter`` contract preserved).
  STRUCTURALLY rank-deficient targets take this path WHOLESALE via a one-shot
  full-data pre-check (``_full_data_certifiable``): the #833 m0 / shift floors
  fit a target whose centered rank is ≈ n_targets − 1 ≈ 29 < TARGET_DIM = 64
  (V0 = base-era answer profiles are target-keyed only), so the registered
  top-64 estimator there is ALGORITHM-COUPLED — the serial gesdd's arbitrary
  null-basis selection is genuinely part of the measured refit noise — and no
  alternative factorization can reproduce it; only the on/off/ctrl floors
  (full-rank targets, verified s64/s1 ~ 1e-3..1e-4 on the real store) get the
  batched speedup.

The serial ``make_refit_pair`` is deliberately UNTOUCHED (#722/#811/#813 import
it) and stays selectable via ``issue833_fit_onpolicy.py --floors-impl serial``;
the ``--floors-selftest`` gate there asserts serial-vs-batched equivalence on
real cells before any production use.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import issue658_fit_predictors as fit658  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue722_bootstrap import _resample_family_idx  # noqa: E402

logger = logging.getLogger("issue833.batched_floors")

# Top-k subspace certification thresholds (relative to the largest eigenvalue).
# Below either floor the serial fallback runs instead — the subspace-invariance
# argument needs a well-separated, strictly-positive top-k boundary. eps/1e-8
# bounds the basis perturbation at ~1e-8, comfortably inside the selftest's
# 1e-6 expectation.
_REL_EIG_FLOOR = 1e-8
_REL_GAP_FLOOR = 1e-8


def _draw_pair_indices(
    n: int, families: Sequence[str], n_pairs: int, seed: int
) -> list[np.ndarray]:
    """Replicate ``make_refit_pair``'s RNG stream exactly → 2*n_pairs index arrays.

    Consumption order per pair: idx_a, idx_b (family-clustered via the SAME
    ``_resample_family_idx`` helper, or the i.i.d. row bootstrap under <2
    families), then the two ``rng.integers(0, 2**31 - 1)`` child-seed draws the
    serial loop feeds to ``fit_fn`` (the ridge ``_fn`` ignores its rng — the
    draws are kept ONLY so the index streams stay bit-identical to serial).
    """
    fams = np.asarray(list(families), dtype=object)
    assert fams.shape == (n,), (fams.shape, n)
    uniq = sorted({str(f) for f in fams})
    clustered = len(uniq) >= 2
    fam_to_idx = {f: np.where(fams.astype(str) == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)
    idxs: list[np.ndarray] = []
    for _p in range(n_pairs):
        if clustered:
            idxs.append(_resample_family_idx(fam_to_idx, uniq, rng))
            idxs.append(_resample_family_idx(fam_to_idx, uniq, rng))
        else:
            idxs.append(rng.integers(0, n, size=n))
            idxs.append(rng.integers(0, n, size=n))
        rng.integers(0, 2**31 - 1)  # child seed a — stream parity only (fit_fn ignores rng)
        rng.integers(0, 2**31 - 1)  # child seed b
    return idxs


def _serial_refit_chain(
    Xb: np.ndarray,
    Yb: np.ndarray,
    grid: np.ndarray,
    r_hat: np.ndarray,
    target_dim: int,
    lambdas: list[float],
    device: str,
) -> tuple[np.ndarray, int, np.ndarray]:
    """One refit through the SERIAL code path → (chain, lam_idx, press_curve).

    Mirrors ``fitM._refit_ridge_fn``'s ``_fn`` (``_pca_basis_v0`` +
    ``_ridge_fit_predict``) with the λ selection + PRESS curve captured; the
    returned ``chain = (pred64 @ pca) @ r_hat`` is the r̂_B projection of the
    serial prediction (the pair statistic only ever consumes this projection).
    """
    pca = fitM._pca_basis_v0(Yb, target_dim)
    Y64 = Yb @ pca.T
    dev = torch.device(device)
    Xt = torch.from_numpy(np.ascontiguousarray(Xb)).to(device=dev, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Y64)).to(device=dev, dtype=torch.float64)
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9
    Xn = (Xt - mu) / sd
    mse = fit658._press_loo_mse_per_lambda(Xn, Yt, lambdas)
    lam_idx = int(torch.argmin(mse).item())
    w = fit658._ridge_dual_weights(Xn, Yt, float(lambdas[lam_idx]))
    Gt = torch.from_numpy(np.ascontiguousarray(grid)).to(device=dev, dtype=torch.float64)
    Gn = (Gt - mu) / sd
    pred64 = (Gn @ w).detach().cpu().numpy()
    chain = (pred64 @ pca) @ r_hat
    return chain, lam_idx, mse.detach().cpu().numpy().copy()


def _chunk_batched_chains(
    Xt: torch.Tensor,
    K_Y: torch.Tensor,
    Yr: torch.Tensor,
    Gt: torch.Tensor,
    Wc: torch.Tensor,
    target_dim: int,
    lambdas: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched chains for one chunk of resample-weight vectors (all float64 torch).

    Args: ``Xt`` (n, d) design, ``K_Y`` (n, n) cached target Gram ``Y Yᵀ``,
    ``Yr`` (n,) cached ``Y @ r_hat``, ``Gt`` (g, d) eval grid, ``Wc`` (B, n)
    integer resample multiplicities. Returns numpy ``(chains (B, g), lam_idx
    (B,), press (B, L), ok (B,))`` where ``ok=False`` rows are UNCERTIFIED
    (degenerate top-k boundary / too few rows) and must go through the serial
    fallback instead.
    """
    B, _n = Wc.shape
    k = target_dim
    m = Wc.sum(1)  # (B,) resample row counts
    sw = Wc.sqrt()
    # ---- X standardization (weighted two-pass; == duplicated-row mean/std) ----
    mu_x = (Wc @ Xt) / m[:, None]  # (B, d)
    Xc = Xt.unsqueeze(0) - mu_x.unsqueeze(1)  # (B, n, d)
    var = ((Xc * Xc) * Wc.unsqueeze(2)).sum(1) / m[:, None]
    sd = var.sqrt() + 1e-9  # (B, d) — same +1e-9 as _ridge_fit_predict
    Xc = Xc / sd.unsqueeze(1)  # Xn_u (standardized unique rows)
    Xnw = Xc * sw.unsqueeze(2)  # W^{1/2} Xn_u
    S = Xnw @ Xnw.transpose(1, 2)  # (B, n, n) = W^{1/2} G_u W^{1/2}
    del Xnw
    gX, QX = torch.linalg.eigh(S)
    del S
    # ---- Y top-k subspace from the cached K_Y (weighted centered dual Gram) ----
    a = (Wc @ K_Y) / m[:, None]  # (B, n): a_i = Y_i · mu_Y
    c = (a * Wc).sum(1) / m  # (B,): mu_Y · mu_Y
    Kc = K_Y.unsqueeze(0) - a.unsqueeze(2) - a.unsqueeze(1) + c[:, None, None]
    MY = sw.unsqueeze(2) * Kc * sw.unsqueeze(1)  # W^{1/2} Yc Ycᵀ W^{1/2}
    del Kc
    gY, QY = torch.linalg.eigh(MY)  # ascending eigenvalues
    del MY
    s_top = gY[:, -k:]  # (B, k) ascending — s_top[:, 0] is the k-th largest
    s_max = gY[:, -1].clamp(min=1e-300)
    ok = m >= k + 2
    ok &= s_top[:, 0] > _REL_EIG_FLOOR * s_max
    ok &= (s_top[:, 0] - gY[:, -k - 1]) > _REL_GAP_FLOOR * s_max
    inv_sqrt = s_top.clamp(min=1e-300).rsqrt()  # (B, k)
    Q64 = QY[:, :, -k:]  # (B, n, k)
    # Y64u = Y V (unique rows): (K_Y − a 1ᵀ) W^{1/2} Q64 / √s — all from cached K_Y.
    YYc = K_Y.unsqueeze(0) - a.unsqueeze(2)  # (B, n, n) = Y Ycᵀ
    Y64u = ((YYc * sw.unsqueeze(1)) @ Q64) * inv_sqrt.unsqueeze(1)  # (B, n, k)
    del YYc
    # t = Vᵀ r_hat = Q64ᵀ W^{1/2} (Yr − mu_Y·r_hat) / √s.
    ycr = Yr.unsqueeze(0) - ((Wc @ Yr) / m).unsqueeze(1)  # (B, n)
    t = torch.einsum("bnk,bn->bk", Q64, sw * ycr) * inv_sqrt  # (B, k)
    # ---- PRESS per λ in n×n space (h_i = [(Q∘Q)filt]_i / w_i on the support) ----
    Z = sw.unsqueeze(2) * Y64u  # scaled targets: rows √w_i · Y64u_i
    QtZ = QX.transpose(1, 2) @ Z  # (B, n, k)
    Qsq = QX * QX
    supported = Wc > 0
    press = torch.empty((B, len(lambdas)), dtype=torch.float64, device=Xt.device)
    for li, lam in enumerate(lambdas):
        filt = gX / (gX + lam)  # (B, n)
        hnum = torch.einsum("bnm,bm->bn", Qsq, filt)
        h = torch.where(supported, hnum / Wc.clamp(min=1.0), torch.zeros_like(hnum))
        Zhat = QX @ (filt.unsqueeze(2) * QtZ)
        resid = (Z - Zhat) / (1.0 - h).clamp(min=1e-8).unsqueeze(2)
        press[:, li] = (resid * resid).sum(dim=(1, 2)) / (m * k)
    lam_idx = torch.argmin(press, dim=1)  # (B,) first-min, same rule as serial
    lam_best = torch.tensor(lambdas, dtype=torch.float64, device=Xt.device)[lam_idx]
    # ---- chain = Gn Xn_uᵀ W^{1/2}(λ+S)⁻¹ W^{1/2} (Y64u t), via the eigh of S ----
    yt = torch.einsum("bnk,bk->bn", Y64u, t)  # (B, n)
    r1 = torch.einsum("bnm,bn->bm", QX, sw * yt)  # QXᵀ (√w · yt)
    r2 = r1 / (gX + lam_best[:, None])
    r3 = torch.einsum("bnm,bm->bn", QX, r2)
    r4 = sw * r3  # (B, n)
    xr = r4 @ Xt  # (B, d) = Xᵀ r4
    num = xr - mu_x * r4.sum(1, keepdim=True)
    ws = num / (sd * sd)  # (B, d): the /sd from Xn_u and the /sd from Gn
    chains = ws @ Gt.T - (mu_x * ws).sum(1, keepdim=True)  # (B, g)
    return (
        chains.cpu().numpy(),
        lam_idx.cpu().numpy(),
        press.cpu().numpy(),
        ok.cpu().numpy(),
    )


def _full_data_certifiable(Y: np.ndarray, target_dim: int) -> bool:
    """Fast pre-check: is the FULL-data centered target's top-k boundary healthy?

    One n×n ``eigvalsh`` of the centered Gram (~ms at n=480). If the FULL data
    already fails the eigenvalue/gap floors at k, every resample fails too
    (resampled rows span a subset of the full row space), so the caller can skip
    the batched attempts and go straight to the serial fallback. A PASS here is
    only a fast-path heuristic — the per-resample certification inside
    ``_chunk_batched_chains`` remains the binding gate.
    """
    n = Y.shape[0]
    if n < target_dim + 2:
        return False
    Yc = Y - Y.mean(axis=0, keepdims=True)
    ev = np.linalg.eigvalsh(Yc @ Yc.T)  # ascending
    s_max = max(float(ev[-1]), 1e-300)
    s_k = float(ev[-target_dim])
    gap = s_k - float(ev[-target_dim - 1])
    return s_k > _REL_EIG_FLOOR * s_max and gap > _REL_GAP_FLOOR * s_max


def _run_batched_chunks(
    X: np.ndarray,
    Y: np.ndarray,
    grid: np.ndarray,
    r_hat: np.ndarray,
    idxs: list[np.ndarray],
    target_dim: int,
    lambdas: list[float],
    device: str,
    chunk_size: int,
) -> tuple[list, list, list, list]:
    """Batched chains for all resamples, flagging fallbacks (never computing them).

    Returns parallel per-resample lists ``(chains, lam_idx, press, fallback)``;
    a ``fallback[j] = True`` entry has ``chains[j] is None`` and must be
    recomputed by the caller through the serial path (uncertified top-k
    boundary, tiny resample, tiny n, or a batched eigh non-convergence).
    """
    n = X.shape[0]
    n_resamples = len(idxs)
    chains: list[np.ndarray | None] = [None] * n_resamples
    lam_idx: list[int | None] = [None] * n_resamples
    press_out: list[np.ndarray | None] = [None] * n_resamples
    fallback = [False] * n_resamples
    if n < target_dim + 2:
        # Too few unique rows for a certified batched top-k boundary (also the
        # tiny-smoke shape): every resample goes through the serial fallback.
        return chains, lam_idx, press_out, [True] * n_resamples
    if not _full_data_certifiable(Y, target_dim):
        # The FULL-data target is rank-deficient / boundary-degenerate at k, so
        # every resample inherits it (a resample's row space is a subset of the
        # full rows'): the registered estimator is ALGORITHM-COUPLED there (the
        # serial gesdd's arbitrary null-basis selection is part of the floor's
        # refit noise) and only the serial path reproduces it. Skip the wasted
        # per-chunk batched attempts entirely. Known structural case: the m0 /
        # shift floors — V0 (base-era answer profiles) is target-keyed only, so
        # its centered rank ≈ n_targets − 1 ≈ 29 < TARGET_DIM = 64 (#833 r5).
        logger.info(
            "[phase=fit_M] make_refit_pair_batched: FULL-data top-%d boundary "
            "uncertifiable (rank-deficient target) — all %d resamples via the "
            "bit-faithful serial path",
            target_dim,
            n_resamples,
        )
        return chains, lam_idx, press_out, [True] * n_resamples
    dev = torch.device(device)
    Xt = torch.from_numpy(X).to(device=dev, dtype=torch.float64)
    Yt = torch.from_numpy(Y).to(device=dev, dtype=torch.float64)
    Gt = torch.from_numpy(grid).to(device=dev, dtype=torch.float64)
    rt = torch.from_numpy(np.ascontiguousarray(r_hat)).to(device=dev, dtype=torch.float64)
    K_Y = Yt @ Yt.T  # the ONE pool reduction reused by every resample
    Yr = Yt @ rt
    for lo in range(0, n_resamples, max(1, chunk_size)):
        batch = idxs[lo : lo + max(1, chunk_size)]
        Wnp = np.stack([np.bincount(ix, minlength=n) for ix in batch]).astype(np.float64)
        Wc = torch.from_numpy(Wnp).to(device=dev, dtype=torch.float64)
        try:
            ch, li_, pr, ok = _chunk_batched_chains(Xt, K_Y, Yr, Gt, Wc, target_dim, lambdas)
        except torch.linalg.LinAlgError as e:
            logger.warning(
                "[phase=fit_M] make_refit_pair_batched: batched eigh failed on chunk "
                "%d..%d (%s) — recomputing each resample via the serial fallback",
                lo,
                lo + len(batch) - 1,
                e,
            )
            for j in range(lo, lo + len(batch)):
                fallback[j] = True
            continue
        for bj in range(len(batch)):
            j = lo + bj
            if ok[bj]:
                chains[j] = ch[bj]
                lam_idx[j] = int(li_[bj])
                press_out[j] = pr[bj]
            else:
                fallback[j] = True
    return chains, lam_idx, press_out, fallback


def make_refit_pair_batched(
    X: np.ndarray,
    Y: np.ndarray,
    grid: np.ndarray,
    r_hat: np.ndarray,
    families: Sequence[str],
    *,
    n_pairs: int = 100,
    seed: int = 0,
    target_dim: int = 64,
    lambdas: Sequence[float] | None = None,
    device: str = "cpu",
    skip_counter: dict | None = None,
    chunk_size: int = 12,
    return_details: bool = False,
):
    """Batched-exact twin of ``make_refit_pair`` with the ridge ``_refit_ridge_fn``.

    Same estimator, same RNG stream, same per-pair statistic
    ``median_grid |(pred_a − pred_b) @ r̂|``, same survivors/skip semantics
    (``skip_counter`` filled with ``{"n_attempted", "n_skipped"}``; all-pairs
    failure raises ``np.linalg.LinAlgError``). Returns the surviving per-pair
    floor statistics (n_pairs − n_skipped,); with ``return_details=True``
    returns ``(stats, details)`` where details carries the per-resample λ index,
    PRESS curves, and the serial-fallback mask (for the equivalence gate).
    """
    lambdas = list(fit658.RIDGE_LAMBDAS) if lambdas is None else list(lambdas)
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    grid = np.ascontiguousarray(np.asarray(grid, dtype=np.float64))
    r_hat = np.asarray(r_hat, dtype=np.float64)
    n = X.shape[0]
    assert Y.shape[0] == n, (X.shape, Y.shape)
    idxs = _draw_pair_indices(n, families, n_pairs, seed)
    chains, lam_idx, press_out, fallback = _run_batched_chunks(
        X, Y, grid, r_hat, idxs, target_dim, lambdas, device, chunk_size
    )
    n_fallback = sum(fallback)
    if n_fallback:
        logger.info(
            "[phase=fit_M] make_refit_pair_batched: %d/%d resamples via the serial "
            "fallback (uncertified top-%d boundary / tiny resample / eigh failure)",
            n_fallback,
            len(idxs),
            target_dim,
        )
    for j, fb in enumerate(fallback):
        if not fb:
            continue
        ix = idxs[j]
        try:
            chain, li_, pr = _serial_refit_chain(
                X[ix], Y[ix], grid, r_hat, target_dim, lambdas, device
            )
        except np.linalg.LinAlgError as e:
            logger.warning(
                "[phase=fit_M] make_refit_pair_batched: serial fallback for resample %d "
                "raised LinAlgError (%s) — its pair will be skipped (serial semantics)",
                j,
                e,
            )
            continue
        chains[j] = chain
        lam_idx[j] = li_
        press_out[j] = pr

    survivors: list[float] = []
    n_skipped = 0
    for p in range(n_pairs):
        ca, cb = chains[2 * p], chains[2 * p + 1]
        if ca is None or cb is None:
            n_skipped += 1
            logger.warning(
                "[phase=fit_M] make_refit_pair_batched: skipping bootstrap pair %d/%d "
                "after LinAlgError in the refit; %d skipped so far",
                p + 1,
                n_pairs,
                n_skipped,
            )
            continue
        survivors.append(float(np.median(np.abs(ca - cb))))
    if skip_counter is not None:
        skip_counter["n_attempted"] = n_pairs
        skip_counter["n_skipped"] = n_skipped
    if not survivors:
        raise np.linalg.LinAlgError(
            f"make_refit_pair_batched: all {n_pairs} refit pairs failed with LinAlgError "
            "(the resample geometry is fully degenerate — cannot build a floor)"
        )
    stats = np.asarray(survivors, dtype=float)
    if return_details:
        return stats, {
            "lam_idx": lam_idx,
            "press": press_out,
            "fallback_serial": fallback,
            "n_fallback_serial": n_fallback,
            "n_skipped": n_skipped,
        }
    return stats
