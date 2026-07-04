"""Issue #952 — batched shared-SVD ridge battery.

The #823 anti-pattern fix (plan §4 Phase 1e): per predictor cell (one X matrix)
we standardize X on train stats, center Y per target on train means, run ONE
economy SVD of the normalized train matrix (float64; ``torch.linalg.svd`` on
GPU when available, numpy fallback — the win is sharing the factorization, not
the device), and then evaluate EVERY λ on the grid for EVERY stacked target
column with two GEMMs:

    preds_eval(λ) = (A_eval * filt(λ)) @ B + ymu
    A_eval = X_eval_n @ V          (n_eval, r)   — once per (cell, split)
    B      = Uᵀ @ Y_train_c        (r, G·H)      — once per (cell, chunk)
    filt(λ) = s / (s² + λ)

Targets are stacked as Y columns in chunks of ``chunk_groups`` slot-arm blocks
(chunk ≈ (n_tr, 8·3584) f64 ≈ 0.7 GB — plan §4). Outputs per (target-group, λ,
split): pooled R² plus per-context (ss_res, ss_tot) with denominators centered
on the TRAIN mean (plan §4 divergence 6 — prediction-time-legal for the bank
transfer read).

Parity contract (plan §4 Phase 1e, the #823 Gram-eigh lesson): before any full
battery, :func:`parity_gate` compares this solver at fixed λ against the
canonical serial oracle — ``fit_h.ridge_fit_predict`` called with a single-λ
grid (its GCV then trivially selects the pinned λ; the code path is the #779
canonical numpy-f64-SVD path, verbatim). Gate: max relative prediction diff
≤ 1e-6 AND |ΔR²| ≤ 1e-7 (relaxed from #823's 1e-8 to admit GPU-vs-CPU BLAS
reduction-order drift; the failed Gram-eigh path at 1.7e-5 still fails).
RuntimeError on failure; the pre-sized fallback is :func:`serial_reference_cell`
run per headline cell (canonical numpy path, validation-selected λ).

No serial per-cell loop anywhere: the battery driver (run_952.py phase 1e)
calls :func:`run_ridge_cell` once per predictor cell and stacks all targets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("issue952.ridge_battery")

DEFAULT_LAMBDAS = np.logspace(-2, 4, 13)  # #779 fit_h grid (plan §11)

# Parity tolerances (plan §11 "Parity tolerance"):
PARITY_TOL_PRED = 1e-6
PARITY_TOL_R2 = 1e-7


# ── small helpers ───────────────────────────────────────────────────────────────


def pooled_r2(ss_res: np.ndarray, ss_tot: np.ndarray) -> float:
    """NaN-aware pooled R² = 1 - Σss_res/Σss_tot over contexts with finite ss_tot."""
    m = np.isfinite(ss_tot) & np.isfinite(ss_res)
    denom = float(ss_tot[m].sum())
    if m.sum() == 0 or denom < 1e-12:
        return float("nan")
    return 1.0 - float(ss_res[m].sum()) / denom


def per_context_r2(ss_res: np.ndarray, ss_tot: np.ndarray) -> np.ndarray:
    """Per-context R² = 1 - ss_res/ss_tot (NaN where invalid). Unbounded below."""
    out = np.full(ss_res.shape, np.nan, dtype=np.float64)
    m = np.isfinite(ss_tot) & np.isfinite(ss_res) & (ss_tot > 1e-12)
    out[m] = 1.0 - ss_res[m] / ss_tot[m]
    return out


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# ── batched shared-SVD cell ─────────────────────────────────────────────────────


@dataclass
class CellResult:
    """Result of one predictor cell (one X matrix, G stacked target groups).

    ``ss_res[split]`` has shape (n_split, G, L) float32 (L = len(lambdas));
    ``ss_tot[split]`` has shape (n_split, G) float32 (λ-independent, train-mean
    centered). ``pooled[split]`` has shape (L, G) float64. ``n_valid[split]``
    has shape (G,) int64 — contexts with a finite target for that group.
    """

    lambdas: np.ndarray
    group_names: list[str]
    pooled: dict[str, np.ndarray] = field(default_factory=dict)
    ss_res: dict[str, np.ndarray] = field(default_factory=dict)
    ss_tot: dict[str, np.ndarray] = field(default_factory=dict)
    n_valid: dict[str, np.ndarray] = field(default_factory=dict)
    imputed_frac: np.ndarray | None = None  # (G,) fraction of train rows mean-imputed
    svd_seconds: float = 0.0
    total_seconds: float = 0.0
    n_train: int = 0
    rank: int = 0


def run_ridge_cell(  # noqa: C901 — one shared-SVD cell; chunk/split loops are the batching
    X_train: np.ndarray,
    Y_train: np.ndarray,
    eval_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    group_names: list[str],
    lambdas: np.ndarray | None = None,
    device: str = "cpu",
    chunk_groups: int = 8,
    allow_train_nan_imputation: bool = False,
) -> CellResult:
    """One predictor cell: shared SVD, all λ x all stacked target groups.

    Args:
        X_train: (n_tr, H_in) float — the predictor matrix (train rows only).
        Y_train: (n_tr, G, H_out) float — stacked target groups (train rows).
            May be fp16 (the slot-store dtype): it is kept in its storage dtype
            and cast to float64 PER CHUNK (a full-array f64 cast at production
            shape — 2998 x 168 x 3584 — would be ~14 TB; the per-chunk cast is
            the plan §4 0.7 GB figure). Must be NaN-free unless
            ``allow_train_nan_imputation`` (then NaN rows per group are imputed
            at that group's train mean and the imputed fraction is recorded —
            descriptive-target escape hatch for the per-decile prefix probes,
            plan §4 note; decision cells never use it).
        eval_splits: name -> (X_e (n_e, H_in), Y_e (n_e, G, H_out)); Y_e may be
            fp16 (cast per chunk) and may be NaN per (context, group) —
            propagated to NaN ss entries.
        group_names: G names for the stacked groups (slot-arm labels).
        lambdas: λ grid (default np.logspace(-2, 4, 13), the #779 grid).
        device: "cpu" (numpy SVD) or "cuda" (torch f64 SVD + GEMMs on GPU).
        chunk_groups: target-group chunk width for the stacked GEMMs.

    Returns:
        CellResult with pooled R² per (λ, group, split) and per-context
        (ss_res, ss_tot) arrays (train-mean-centered denominators).
    """
    t_start = time.time()
    if lambdas is None:
        lambdas = DEFAULT_LAMBDAS
    lambdas = np.asarray(lambdas, dtype=np.float64)
    n_lam = len(lambdas)

    Xtr = np.asarray(X_train, dtype=np.float64)
    n_tr, h_in = Xtr.shape
    assert np.isfinite(Xtr).all(), "X_train must be finite"
    # Keep Y in its STORAGE dtype (fp16 for the slot store); cast to f64 per chunk.
    Ytr = np.asarray(Y_train)
    if not np.issubdtype(Ytr.dtype, np.floating):
        Ytr = Ytr.astype(np.float32)
    assert Ytr.ndim == 3 and Ytr.shape[0] == n_tr, f"Y_train shape {Ytr.shape}"
    n_groups, h_out = Ytr.shape[1], Ytr.shape[2]
    assert len(group_names) == n_groups, (len(group_names), n_groups)

    imputed_frac = np.zeros(n_groups, dtype=np.float64)
    if allow_train_nan_imputation:
        needs_write = False
        for g in range(n_groups):
            if (~np.isfinite(Ytr[:, g, :])).any():
                needs_write = True
                break
        if needs_write and Ytr is np.asarray(Y_train):
            Ytr = Ytr.copy()  # never mutate the caller's slot store in place
        for g in range(n_groups):
            row_bad = ~np.isfinite(Ytr[:, g, :]).all(axis=1)
            if row_bad.any():
                if row_bad.all():
                    raise ValueError(f"group {group_names[g]}: ALL train rows NaN")
                mu_g = Ytr[~row_bad, g, :].astype(np.float64).mean(axis=0)
                Ytr[row_bad, g, :] = mu_g.astype(Ytr.dtype)
                imputed_frac[g] = float(row_bad.mean())
    else:
        assert np.isfinite(Ytr).all(), (
            "Y_train has NaN and allow_train_nan_imputation=False — the fit universe "
            "must be target-valid for decision cells"
        )

    # Standardize X on train stats (fit_h conventions: population std + 1e-9).
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd

    use_cuda = device == "cuda" and _torch_cuda_available()
    t_svd0 = time.time()
    if use_cuda:
        import torch

        Xt = torch.from_numpy(Xtr_n).to("cuda")
        U_t, s_t, Vh_t = torch.linalg.svd(Xt, full_matrices=False)
        del Xt
    else:
        U_np, s_np, Vh_np = np.linalg.svd(Xtr_n, full_matrices=False)
    svd_seconds = time.time() - t_svd0

    result = CellResult(
        lambdas=lambdas,
        group_names=list(group_names),
        imputed_frac=imputed_frac,
        svd_seconds=svd_seconds,
        n_train=n_tr,
        rank=min(n_tr, h_in),
    )

    # Prepare per-split A_eval = X_e_n @ V  (n_e, r) once.
    a_eval: dict[str, np.ndarray] = {}
    for name, (X_e, Y_e) in eval_splits.items():
        X_e = np.asarray(X_e, dtype=np.float64)
        assert X_e.shape[1] == h_in, (name, X_e.shape)
        Xe_n = (X_e - xmu) / xsd
        if use_cuda:
            import torch

            a_eval[name] = (torch.from_numpy(Xe_n).to("cuda") @ Vh_t.T).cpu().numpy()
        else:
            a_eval[name] = Xe_n @ Vh_np.T
        n_e = X_e.shape[0]
        result.ss_res[name] = np.full((n_e, n_groups, n_lam), np.nan, dtype=np.float32)
        result.ss_tot[name] = np.full((n_e, n_groups), np.nan, dtype=np.float32)
        result.pooled[name] = np.full((n_lam, n_groups), np.nan, dtype=np.float64)
        result.n_valid[name] = np.zeros(n_groups, dtype=np.int64)
        assert np.asarray(Y_e).shape == (n_e, n_groups, h_out), (name, np.asarray(Y_e).shape)
        assert np.asarray(Y_e).ndim == 3, name  # dtype preserved; cast per chunk below

    if use_cuda:
        import torch

        s_arr = s_t.cpu().numpy()
    else:
        s_arr = s_np
    s2 = s_arr**2
    filts = s_arr[None, :] / (s2[None, :] + lambdas[:, None])  # (n_lam, r)

    # Chunk over target groups (f64 cast happens HERE, per chunk — plan §4 0.7 GB).
    for g0 in range(0, n_groups, chunk_groups):
        g1 = min(g0 + chunk_groups, n_groups)
        w = g1 - g0
        Yc = Ytr[:, g0:g1, :].astype(np.float64).reshape(n_tr, w * h_out)  # f64 chunk copy
        ymu = Yc.mean(0)  # (w*h_out,)
        Yc_c = Yc - ymu
        if use_cuda:
            import torch

            B = (U_t.T @ torch.from_numpy(Yc_c).to("cuda")).cpu().numpy()  # (r, w*h_out)
        else:
            B = U_np.T @ Yc_c

        B_t = None
        if use_cuda:
            import torch

            B_t = torch.from_numpy(B).to("cuda")  # upload ONCE per chunk
        for name, (_X_e, Y_e) in eval_splits.items():
            A = a_eval[name]  # (n_e, r)
            n_e = A.shape[0]
            Ye = np.asarray(Y_e)[:, g0:g1, :].astype(np.float64).reshape(n_e, w * h_out)
            # ss_tot (train-mean centered; λ-independent)
            diff_tot = Ye - ymu
            sst = diff_tot.reshape(n_e, w, h_out)
            sst = np.square(sst).sum(axis=2)  # (n_e, w); NaN propagates
            result.ss_tot[name][:, g0:g1] = sst.astype(np.float32)
            A_t = None
            if use_cuda:
                import torch

                A_t = torch.from_numpy(A).to("cuda")  # upload ONCE per (chunk, split)
            for li in range(n_lam):
                if use_cuda:
                    import torch

                    filt_t = torch.from_numpy(filts[li]).to("cuda")
                    P = ((A_t * filt_t[None, :]) @ B_t).cpu().numpy()
                else:
                    P = (A * filts[li][None, :]) @ B  # (n_e, w*h_out)
                resid = Ye - (P + ymu)
                ssr = np.square(resid.reshape(n_e, w, h_out)).sum(axis=2)  # (n_e, w)
                result.ss_res[name][:, g0:g1, li] = ssr.astype(np.float32)
            del A_t
        del B, Yc, Yc_c, B_t

    # Pooled R² + n_valid per (λ, group, split).
    for name in eval_splits:
        sst = result.ss_tot[name].astype(np.float64)  # (n_e, G)
        for g in range(n_groups):
            valid = np.isfinite(sst[:, g])
            result.n_valid[name][g] = int(valid.sum())
            denom = sst[valid, g].sum()
            if valid.sum() == 0 or denom < 1e-12:
                continue
            ssr_g = result.ss_res[name][valid, g, :].astype(np.float64)  # (n_valid, n_lam)
            result.pooled[name][:, g] = 1.0 - ssr_g.sum(axis=0) / denom

    result.total_seconds = time.time() - t_start
    logger.info(
        "[ridge-cell] n_tr=%d G=%d splits=%s device=%s svd=%.1fs total=%.1fs",
        n_tr,
        n_groups,
        list(eval_splits),
        "cuda" if use_cuda else "cpu",
        svd_seconds,
        result.total_seconds,
    )
    return result


# ── canonical serial reference (fallback path) ──────────────────────────────────


def serial_reference_cell(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    *,
    lambdas: np.ndarray | None = None,
) -> dict:
    """Canonical numpy-f64-SVD ridge for ONE target group, validation-selected λ.

    The fallback path (plan §4 Phase 1e): the ``fit_h.ridge_fit_predict``
    conventions verbatim (standardize-X population-std+1e-9, center-Y,
    numpy SVD, dual weights) with the plan's VALIDATION-split λ selection in
    place of GCV (user pin: validation selects hyperparameters). One SVD per
    cell; per-λ validation predictions via the shared factorization.

    Returns dict with val pooled R² per λ, the selected λ, and test per-context
    (ss_res, ss_tot) + pooled R² at the frozen λ (train-mean denominators).
    """
    if lambdas is None:
        lambdas = DEFAULT_LAMBDAS
    lambdas = np.asarray(lambdas, dtype=np.float64)
    Xtr = np.asarray(X_train, dtype=np.float64)
    Ytr = np.asarray(Y_train, dtype=np.float64)
    assert np.isfinite(Xtr).all() and np.isfinite(Ytr).all()
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu

    U, s, Vt = np.linalg.svd(Xtr_n, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Ytr_c

    def _preds(X_e: np.ndarray, lam: float) -> np.ndarray:
        Xe_n = (np.asarray(X_e, dtype=np.float64) - xmu) / xsd
        filt = s / (s2 + lam)
        W = (Vt.T * filt) @ UtY
        return Xe_n @ W + ymu

    def _stats(Y_e: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Ye = np.asarray(Y_e, dtype=np.float64)
        ss_res = np.square(Ye - preds).sum(axis=1)
        ss_tot = np.square(Ye - ymu).sum(axis=1)
        return ss_res, ss_tot

    val_r2 = np.full(len(lambdas), np.nan)
    for li, lam in enumerate(lambdas):
        ssr, sst = _stats(Y_val, _preds(X_val, lam))
        val_r2[li] = pooled_r2(ssr, sst)
    best_li = int(np.nanargmax(val_r2))
    best_lam = float(lambdas[best_li])
    ssr_t, sst_t = _stats(Y_test, _preds(X_test, best_lam))
    return {
        "lambdas": lambdas.tolist(),
        "val_pooled_r2": val_r2.tolist(),
        "selected_lambda": best_lam,
        "selected_lambda_idx": best_li,
        "test_pooled_r2": pooled_r2(ssr_t, sst_t),
        "test_ss_res": ssr_t,
        "test_ss_tot": sst_t,
    }


# ── parity gate (batched vs canonical serial oracle) ────────────────────────────


def parity_gate(
    X_train: np.ndarray,
    Y_train_group: np.ndarray,
    X_eval: np.ndarray,
    Y_eval_group: np.ndarray,
    *,
    lam: float,
    device: str = "cpu",
    cell_label: str = "parity",
    tol_pred: float = PARITY_TOL_PRED,
    tol_r2: float = PARITY_TOL_R2,
) -> dict:
    """Batched-solver-vs-serial-oracle parity at pinned λ (hard gate, plan §4 1e).

    The oracle is ``fit_h.ridge_fit_predict`` called with a SINGLE-λ grid — its
    GCV then trivially selects the pinned λ, so the exercised code path is the
    canonical #779 numpy-f64-SVD fit verbatim. The batched side is
    :func:`run_ridge_cell` with one target group at the same λ. Raises
    RuntimeError when max relative prediction diff > ``tol_pred`` or
    |ΔR²| > ``tol_r2``.

    Returns the gate record (diffs + timings) for the smoke/compute-basis log.
    """
    from explore_persona_space.experiments.issue_779.fit_h import ridge_fit_predict

    Y_tr = np.asarray(Y_train_group, dtype=np.float64)
    assert Y_tr.ndim == 2, "one target group: (n_tr, H_out)"
    t0 = time.time()
    pred_oracle = ridge_fit_predict(
        X_train, Y_tr, X_eval, lambdas=np.asarray([lam], dtype=np.float64)
    )
    t1 = time.time()
    res = run_ridge_cell(
        np.asarray(X_train),
        Y_tr[:, None, :],
        {"eval": (np.asarray(X_eval), np.asarray(Y_eval_group)[:, None, :])},
        group_names=[cell_label],
        lambdas=np.asarray([lam], dtype=np.float64),
        device=device,
        chunk_groups=1,
    )
    t2 = time.time()

    ymu = Y_tr.mean(0)

    # Oracle pooled R² on the eval split with train-mean denominators.
    Ye = np.asarray(Y_eval_group, dtype=np.float64)
    ssr_o = np.square(Ye - pred_oracle).sum(axis=1)
    sst_o = np.square(Ye - ymu).sum(axis=1)
    r2_oracle = pooled_r2(ssr_o, sst_o)
    r2_batched = float(res.pooled["eval"][0, 0])

    # Batched predictions at the pinned λ — recompute through the same batched
    # formula run_ridge_cell used (identical arithmetic path re-executed here so
    # the prediction-space diff is measured, not just R²-space).
    pred_batched = _batched_preds_at_lambda(
        np.asarray(X_train, dtype=np.float64),
        Y_tr,
        np.asarray(X_eval, dtype=np.float64),
        lam,
        device=device,
    )
    scale = float(np.abs(pred_oracle).max()) + 1e-12
    max_rel = float(np.abs(pred_batched - pred_oracle).max()) / scale
    d_r2 = abs(r2_batched - r2_oracle)
    record = {
        "cell": cell_label,
        "lambda": float(lam),
        "max_rel_pred_diff": max_rel,
        "abs_delta_r2": d_r2,
        "r2_oracle": r2_oracle,
        "r2_batched": r2_batched,
        "oracle_seconds": t1 - t0,
        "batched_seconds": t2 - t1,
        "device": device,
        "n_train": int(np.asarray(X_train).shape[0]),
        "tol_pred": tol_pred,
        "tol_r2": tol_r2,
    }
    logger.info(
        "[parity-gate] %s: max_rel=%.3e dR2=%.3e (oracle %.1fs, batched %.1fs)",
        cell_label,
        max_rel,
        d_r2,
        t1 - t0,
        t2 - t1,
    )
    if max_rel > tol_pred or d_r2 > tol_r2:
        raise RuntimeError(
            f"ridge parity gate FAIL on {cell_label}: max_rel_pred_diff={max_rel:.3e} "
            f"(tol {tol_pred:.0e}), |dR2|={d_r2:.3e} (tol {tol_r2:.0e}) — refusing to run "
            "the batched battery on an unverified solver (fallback: serial_reference_cell "
            "on the headline cells, plan §4 Phase 1e)"
        )
    return record


def _batched_preds_at_lambda(
    Xtr: np.ndarray, Ytr: np.ndarray, Xev: np.ndarray, lam: float, *, device: str = "cpu"
) -> np.ndarray:
    """The batched solver's prediction formula at one λ (for the parity diff)."""
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    Xev_n = (Xev - xmu) / xsd
    ymu = Ytr.mean(0)
    Yc = Ytr - ymu
    if device == "cuda" and _torch_cuda_available():
        import torch

        Xt = torch.from_numpy(Xtr_n).to("cuda")
        U, s, Vh = torch.linalg.svd(Xt, full_matrices=False)
        A = torch.from_numpy(Xev_n).to("cuda") @ Vh.T
        B = U.T @ torch.from_numpy(Yc).to("cuda")
        s_np = s.cpu().numpy()
        filt = torch.from_numpy(s_np / (s_np**2 + lam)).to("cuda")
        P = (A * filt[None, :]) @ B
        return P.cpu().numpy() + ymu
    U, s, Vh = np.linalg.svd(Xtr_n, full_matrices=False)
    A = Xev_n @ Vh.T
    B = U.T @ Yc
    filt = s / (s**2 + lam)
    return (A * filt[None, :]) @ B + ymu
