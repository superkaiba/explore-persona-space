# ruff: noqa: RUF002, RUF003
"""Issue #920 shared batched LOFO fit machinery (used by S4 fits AND S5 nulls).

Implements the plan §3.5-S4 batching contract on the verified #658/#810
primitives' EXACT conventions (train-only ddof=0 standardization + 1e-9 floor,
PRESS-LOO λ-select over ``issue658_fit_predictors.RIDGE_LAMBDAS``, dual/Woodbury
solves, per-fold train-mean centering, Gram-space sign-canonicalized top-k PCA):

- ``FoldXCache`` — per-fold X-only factors for a STACK of predictor cells
  (standardization, dual Gram eigendecomposition, per-λ ``(G+λI)⁻¹``, held-out
  dual read vectors for BOTH probe-set inputs), computed ONCE per (cell × fold)
  and shared across every target (the #810 ``_LocoRidgeXCache`` idea, LOFO folds
  + a cell batch axis instead of hardcoded LOO).
- ``batched_pca_project`` — per-(a-cell × fold) train-only Gram top-k PCA
  (matches ``issue810_adhoc_lofo_heatmaps._gram_top_k_pca`` incl. the
  sign-canonicalization), batched over cells.
- ``batched_press_predict`` — the per-pair PRESS λ-select + dual solve + held-out
  predictions as batched GEMM/eigh consumers; NO Python loop over cells/pairs.
- Serial reference twins (``serial_reference_map_fit``) contained per the
  tombstone-compliance pattern — used ONLY by the §7 G2 equivalence gate.

The S4/S5 entrypoints dispatch THESE functions for the hot loops; the G2 gate
asserts on the same functions (never an unused sibling — the #779
hollow-verification-gate rule).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Shared-VM thread caps (#847) must bind BEFORE torch freezes its pool at import.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue658_fit_predictors import (  # noqa: E402
    RIDGE_LAMBDAS,
    _press_loo_mse_per_lambda,
    _resolve_device,
    _ridge_dual_weights,
)
from issue920_common import (  # noqa: E402
    ANS_POOLED_CELLS,
    ANS_TARGET_PERLAYER_FAMILIES,
    CTX_PERLAYER_FAMILIES,
    CTX_POOLED_CELLS,
    EXPECTED_LAYERS,
    VALID_OK,
)

logger = logging.getLogger("issue920_fit_core")

PCA_K = 34  # min(48, n_train_min − 2) = min(48, 36 − 2), uniform across folds (plan §11)
N_LAYERS = EXPECTED_LAYERS


def fit_device() -> str:
    """EPM_FIT_DEVICE > auto (the #658 parametrized-device contract)."""
    import os

    return _resolve_device(os.environ.get("EPM_FIT_DEVICE", "auto"))


# ── reduced summary matrices from the per-probe stores ───────────────────────


def load_reduced_matrices(store_dir: Path, ctx_ids: list[str]) -> dict:
    """Stream the per-probe store → probe-reduced cell matrices (fp32 CPU).

    Returns dict with:
      X_ctx  (542, n_ctx, H)  — 19 per-layer families × 28 layers (family-major,
                                 layer-minor) + the 10 layer-pooled context cells,
      Y_ans  (1018, n_ctx, H) — 36 per-layer answer families × 28 + 10 pooled,
      ctx_cell_names / ans_cell_names (list[str], "family@L<l>" / pooled name),
      excluded_ans_cells — per-layer position cells with ZERO valid probes for
                            ≥1 context (excluded from the sweep, logged; plan §3.3),
      min_probe_coverage — {family: min over contexts of valid-probe count}.

    Probe reduction is the pinned convention: probe-mean over VALID_OK probes per
    per-layer family (dedup-masked tail slots EXCLUDED); layer-pooled cells are
    pooled PER PROBE first (lmean/lmax over the 28 per-layer values), then
    probe-meaned (``probe_avg_max``).
    """
    H = None
    n = len(ctx_ids)
    n_ctx_pl = len(CTX_PERLAYER_FAMILIES)
    n_ans_pl = len(ANS_TARGET_PERLAYER_FAMILIES)
    ctx_pl = None  # (19, 28, n, H)
    ans_pl = None  # (36, 28, n, H)
    ctx_pool = None  # (10, n, H)
    ans_pool = None  # (10, n, H)
    coverage = {
        f: np.full(n, -1, dtype=np.int64)
        for f in CTX_PERLAYER_FAMILIES + ANS_TARGET_PERLAYER_FAMILIES
    }

    for ci, cid in enumerate(ctx_ids):
        blob = torch.load(store_dir / f"{cid}.pt", weights_only=False)
        fams = blob["families"]
        fam_idx = {f: i for i, f in enumerate(fams)}
        validity = blob["validity"]  # (P, 55)
        lc = len(blob["capture_layers"])
        if H is None:
            H = blob[f"fam::{fams[0]}"].shape[-1]
            ctx_pl = torch.zeros(n_ctx_pl, lc, n, H, dtype=torch.float32)
            ans_pl = torch.zeros(n_ans_pl, lc, n, H, dtype=torch.float32)
            ctx_pool = torch.zeros(len(CTX_POOLED_CELLS), n, H, dtype=torch.float32)
            ans_pool = torch.zeros(len(ANS_POOLED_CELLS), n, H, dtype=torch.float32)
        per_probe = {}  # family -> (P_valid mask, (P, Lc, H) fp32)
        for f in CTX_PERLAYER_FAMILIES + ANS_TARGET_PERLAYER_FAMILIES:
            vals = blob[f"fam::{f}"].to(torch.float32)  # (P, Lc, H)
            ok = validity[:, fam_idx[f]] == VALID_OK
            coverage[f][ci] = int(ok.sum())
            per_probe[f] = (ok, vals)
            reduced = vals[ok].mean(dim=0) if int(ok.sum()) > 0 else torch.zeros(lc, H)
            if f in CTX_PERLAYER_FAMILIES:
                ctx_pl[CTX_PERLAYER_FAMILIES.index(f), :, ci] = reduced
            else:
                ans_pl[ANS_TARGET_PERLAYER_FAMILIES.index(f), :, ci] = reduced
        for pi, (_name, base, kind) in enumerate(CTX_POOLED_CELLS):
            ok, vals = per_probe[base]
            pooled = vals.mean(dim=1) if kind == "lmean" else vals.amax(dim=1)  # (P, H)
            ctx_pool[pi, ci] = pooled[ok].mean(dim=0)
        for pi, (_name, base, kind) in enumerate(ANS_POOLED_CELLS):
            ok, vals = per_probe[base]
            pooled = vals.mean(dim=1) if kind == "lmean" else vals.amax(dim=1)
            ans_pool[pi, ci] = pooled[ok].mean(dim=0)

    # flatten to cell stacks: per-layer family-major (f*28 + l), pooled appended.
    X_ctx = torch.cat(
        [ctx_pl.permute(0, 1, 2, 3).reshape(n_ctx_pl * ctx_pl.shape[1], n, H), ctx_pool], dim=0
    )
    Y_ans = torch.cat([ans_pl.reshape(n_ans_pl * ans_pl.shape[1], n, H), ans_pool], dim=0)
    lc = ctx_pl.shape[1]
    ctx_cell_names = [f"{f}@L{li}" for f in CTX_PERLAYER_FAMILIES for li in range(lc)] + [
        name for name, _b, _k in CTX_POOLED_CELLS
    ]
    ans_cell_names = [f"{f}@L{li}" for f in ANS_TARGET_PERLAYER_FAMILIES for li in range(lc)] + [
        name for name, _b, _k in ANS_POOLED_CELLS
    ]
    excluded = sorted(
        f"{f}@L*"
        for f in ANS_TARGET_PERLAYER_FAMILIES + CTX_PERLAYER_FAMILIES
        if int(coverage[f].min()) == 0
    )
    if excluded:
        logger.warning("cells with a zero-valid-probe context (EXCLUDED families): %s", excluded)
    return {
        "X_ctx": X_ctx,
        "Y_ans": Y_ans,
        "ctx_cell_names": ctx_cell_names,
        "ans_cell_names": ans_cell_names,
        "n_layers": lc,
        "excluded_families": excluded,
        "min_probe_coverage": {f: int(v.min()) for f, v in coverage.items()},
    }


def excluded_mask(names: list[str], fams: list[str]) -> np.ndarray:
    """Boolean mask over cell names whose family is in the excluded list."""
    ex = {f.split("@")[0] for f in fams}
    return np.array([nm.split("@")[0] in ex for nm in names], dtype=bool)


def union_excluded(red_A: dict, red_B: dict) -> tuple[list[str], dict[str, list[str]]]:
    """Union of BOTH stores' zero-coverage exclusions (+ the per-source record).

    ``load_reduced_matrices`` zero-fills a zero-valid-probe (family, context)
    cell, so a set-B-only coverage gap must be excluded from the sweep exactly
    like a set-A one — masking from set A alone silently zero-fills the B side
    into R2/R3/R4, the B-side read-outs, the identity ceiling, and every null
    band (round-1 blocker ``set-b-zero-coverage-not-masked``; pinned by
    ``tests/test_issue920_dispatch_contract.py``).
    """
    by_source = {
        "set_A": sorted(red_A["excluded_families"]),
        "set_B": sorted(red_B["excluded_families"]),
    }
    union = sorted(set(by_source["set_A"]) | set(by_source["set_B"]))
    return union, by_source


def enumerate_map_cells(
    n_layers: int,
    n_ctx_pl: int = 19,
    n_ans_pl: int = 36,
    n_ctx_pool: int = 10,
    n_ans_pool: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """(c_idx, a_idx) arrays over the 34,652 matched-layer map cells (plan §3.4).

    Order: LL (layer-major: l, then c-family, then a-family) — the per-(layer,
    fold) batch grain; then LP (per-layer c × pooled a), PL, PP.
    """
    c_sel, a_sel = [], []
    pooled_c0 = n_ctx_pl * n_layers
    pooled_a0 = n_ans_pl * n_layers
    for li in range(n_layers):
        for f in range(n_ctx_pl):
            for g in range(n_ans_pl):
                c_sel.append(f * n_layers + li)
                a_sel.append(g * n_layers + li)
    for f in range(n_ctx_pl):
        for li in range(n_layers):
            for j in range(n_ans_pool):
                c_sel.append(f * n_layers + li)
                a_sel.append(pooled_a0 + j)
    for i in range(n_ctx_pool):
        for g in range(n_ans_pl):
            for li in range(n_layers):
                c_sel.append(pooled_c0 + i)
                a_sel.append(g * n_layers + li)
    for i in range(n_ctx_pool):
        for j in range(n_ans_pool):
            c_sel.append(pooled_c0 + i)
            a_sel.append(pooled_a0 + j)
    c_arr, a_arr = np.asarray(c_sel, dtype=np.int64), np.asarray(a_sel, dtype=np.int64)
    if n_layers == N_LAYERS:
        assert len(c_arr) == 34652, len(c_arr)
    return c_arr, a_arr


# ── per-fold X-only cache (batched over predictor cells) ─────────────────────


class FoldXCache:
    """X-only LOFO factors for a stack of predictor cells at ONE fold.

    ``X`` (C, n, D) fp64; ``tr``/``te`` row indices; ``XB`` optional set-B stack
    (standardized with the set-A TRAIN stats — the input-OOD read). Everything
    here is target-independent → shared across all answer targets, behaviors,
    and permutation draws (the #810 ``_LocoRidgeXCache`` contract, family folds).
    """

    def __init__(
        self,
        X: torch.Tensor,
        tr: list[int],
        te: list[int],
        XB: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        C, _n, _D = X.shape
        self.C, self.m, self.n_te = C, len(tr), len(te)
        self.lambdas = list(RIDGE_LAMBDAS)
        self.device = device
        Xd = X.to(device=device, dtype=torch.float64)
        tr_t = torch.tensor(tr, device=device)
        te_t = torch.tensor(te, device=device)
        Xtr = Xd[:, tr_t]
        mu = Xtr.mean(dim=1, keepdim=True)
        sd = Xtr.std(dim=1, correction=0, keepdim=True) + 1e-9  # ddof=0 (#658)
        Xtr_n = (Xtr - mu) / sd  # (C, m, D)
        G = torch.bmm(Xtr_n, Xtr_n.transpose(1, 2))  # (C, m, m)
        self.evals, self.Q = torch.linalg.eigh(G)
        self.Qsq = self.Q * self.Q
        m = self.m
        eye = torch.eye(m, dtype=torch.float64, device=device)
        self.Ainv = torch.stack(
            [torch.linalg.inv(G + lam * eye) for lam in self.lambdas], dim=1
        )  # (C, nlam, m, m)
        XteA_n = (Xd[:, te_t] - mu) / sd
        self.xdotA = torch.bmm(XteA_n, Xtr_n.transpose(1, 2))  # (C, n_te, m)
        self.xdotB = None
        if XB is not None:
            XteB_n = (XB.to(device=device, dtype=torch.float64)[:, te_t] - mu) / sd
            self.xdotB = torch.bmm(XteB_n, Xtr_n.transpose(1, 2))


def batched_press_predict(
    cache: FoldXCache, c_sel: torch.Tensor, Ytr_c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """PRESS λ-select + dual solve + held-out predictions for a batch of pairs.

    ``c_sel`` (Np,) long — predictor-cell index per pair into ``cache``;
    ``Ytr_c`` (Np, m, P) fp64 — CENTERED train targets per pair. Returns
    (predA (Np, n_te, P) CENTERED, predB or None, best_lambda_idx (Np,)).
    Numerics match ``issue658_fit_predictors._press_loo_mse_per_lambda`` +
    ``_ridge_dual_weights`` per pair exactly (the G2 gate asserts atol=1e-8).
    """
    dev = cache.device
    _Np, _m, _P = Ytr_c.shape
    Q = cache.Q[c_sel]  # (Np, m, m)
    evals = cache.evals[c_sel]  # (Np, m)
    Qsq = cache.Qsq[c_sel]
    lam = torch.tensor(cache.lambdas, dtype=torch.float64, device=dev)  # (nlam,)
    nlam = lam.shape[0]
    QtY = torch.bmm(Q.transpose(1, 2), Ytr_c)  # (Np, m, P)
    filt = evals.unsqueeze(0) / (evals.unsqueeze(0) + lam.view(nlam, 1, 1))  # (nlam, Np, m)
    h_diag = torch.einsum("pkj,lpj->lpk", Qsq, filt)  # (nlam, Np, m)
    filt_QtY = filt.unsqueeze(-1) * QtY.unsqueeze(0)  # (nlam, Np, m, P)
    Yhat = torch.einsum("pkj,lpjq->lpkq", Q, filt_QtY)  # (nlam, Np, m, P)
    resid = Ytr_c.unsqueeze(0) - Yhat
    denom = (1.0 - h_diag).clamp(min=1e-8).unsqueeze(-1)
    loo = resid / denom
    mse = (loo * loo).mean(dim=(2, 3))  # (nlam, Np)
    best = torch.argmin(mse, dim=0)  # (Np,)
    Ainv_sel = cache.Ainv[c_sel, best]  # (Np, m, m)
    alpha = torch.bmm(Ainv_sel, Ytr_c)  # (Np, m, P)
    predA = torch.bmm(cache.xdotA[c_sel], alpha)  # (Np, n_te, P)
    predB = torch.bmm(cache.xdotB[c_sel], alpha) if cache.xdotB is not None else None
    return predA, predB, best


def batched_press_predict_per_column(
    cache: FoldXCache, c_sel: torch.Tensor, Ytr_c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """As ``batched_press_predict`` but λ selected PER TARGET COLUMN (read-outs).

    ``Ytr_c`` (Np, m, P): the PRESS MSE is reduced over m ONLY, giving a
    (nlam, Np, P) surface and a per-(pair, column) λ — the per-behavior λ of the
    #810 trained-ridge recipe (each behavior fit separately), batched. Returns
    (predA (Np, n_te, P), predB, best (Np, P)).
    """
    dev = cache.device
    Np, _m, P = Ytr_c.shape
    Q = cache.Q[c_sel]
    evals = cache.evals[c_sel]
    Qsq = cache.Qsq[c_sel]
    lam = torch.tensor(cache.lambdas, dtype=torch.float64, device=dev)
    nlam = lam.shape[0]
    QtY = torch.bmm(Q.transpose(1, 2), Ytr_c)
    filt = evals.unsqueeze(0) / (evals.unsqueeze(0) + lam.view(nlam, 1, 1))
    h_diag = torch.einsum("pkj,lpj->lpk", Qsq, filt)
    filt_QtY = filt.unsqueeze(-1) * QtY.unsqueeze(0)
    Yhat = torch.einsum("pkj,lpjq->lpkq", Q, filt_QtY)
    loo = (Ytr_c.unsqueeze(0) - Yhat) / (1.0 - h_diag).clamp(min=1e-8).unsqueeze(-1)
    mse = (loo * loo).mean(dim=2)  # (nlam, Np, P)
    best = torch.argmin(mse, dim=0)  # (Np, P)
    # alpha per (pair, column) at its own λ: compute per λ, select by mask.
    predA = torch.zeros((Np, cache.n_te, P), dtype=torch.float64, device=dev)
    predB = torch.zeros_like(predA) if cache.xdotB is not None else None
    for li in range(nlam):
        colmask = (best == li).unsqueeze(1)  # (Np, 1, P)
        if not bool(colmask.any()):
            continue
        alpha_l = torch.bmm(cache.Ainv[c_sel, li], Ytr_c)  # (Np, m, P)
        pA = torch.bmm(cache.xdotA[c_sel], alpha_l)
        predA = torch.where(colmask, pA, predA)
        if predB is not None:
            pB = torch.bmm(cache.xdotB[c_sel], alpha_l)
            predB = torch.where(colmask, pB, predB)
    return predA, predB, best


# ── batched train-fold Gram PCA (target side) ────────────────────────────────


def batched_pca_project(Ytr: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-cell train-only top-k PCA basis, batched (matches ``_gram_top_k_pca``).

    ``Ytr`` (A, m, H) fp64. Returns (mu (A, 1, H), comps (A, k, H)) with each
    component sign-canonicalized (largest-|entry| positive) and near-null
    components (S < 1e-6·S_max, or eigenvalue ≤ 1e-9) ZEROED — a zero row
    projects train AND test to 0, contributing nothing to either SS term
    (shape-uniform batching; equivalent to dropping the component).
    """
    mu = Ytr.mean(dim=1, keepdim=True)
    Yc = Ytr - mu
    G = torch.bmm(Yc, Yc.transpose(1, 2))  # (A, m, m)
    w, U = torch.linalg.eigh(G)  # ascending
    A, m, _ = G.shape
    kk = min(k, m)
    idx = torch.arange(m - 1, m - 1 - kk, -1, device=Ytr.device)  # top-kk descending
    w_top = w[:, idx].clamp(min=1e-12)  # (A, kk)
    U_top = U[:, :, idx]  # (A, m, kk)
    S = torch.sqrt(w_top)
    comps = torch.bmm(U_top.transpose(1, 2), Yc) / S.unsqueeze(-1)  # (A, kk, H)
    # zero near-null components (degenerate cells keep tensor shape)
    keep = (w[:, idx] > 1e-9) & (1e-6 * S[:, :1] < S)
    comps = comps * keep.unsqueeze(-1)
    # sign-canonicalize: largest-|entry| positive per component
    j = comps.abs().argmax(dim=2, keepdim=True)  # (A, kk, 1)
    sign = torch.sign(torch.gather(comps, 2, j))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    comps = comps * sign
    if kk < k:  # pad to uniform k with zero rows (tiny smoke folds)
        pad = torch.zeros((A, k - kk, comps.shape[2]), dtype=comps.dtype, device=comps.device)
        comps = torch.cat([comps, pad], dim=1)
    return mu, comps


def pca_apply(rows: torch.Tensor, mu: torch.Tensor, comps: torch.Tensor) -> torch.Tensor:
    """Project (A, r, H) rows into each cell's basis → (A, r, k)."""
    return torch.bmm(rows - mu, comps.transpose(1, 2))


# ── serial reference twins (G2 gate ONLY — contained, tombstone-compliant) ───


def serial_reference_map_fit(
    Xc: np.ndarray,
    Yv: np.ndarray,
    XB: np.ndarray | None,
    YB: np.ndarray | None,
    groups: list[str],
    k: int,
    perm_row: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """SERIAL per-fold map fit for 1 (c-cell, a-cell) — the §7 G2 oracle.

    Mirrors ``issue810_adhoc_lofo_heatmaps._recon_fold_predict`` with the #920
    additions (uniform k, zero-masked near-null PCA components, B-input reads,
    optional row permutation of the TARGET — the null contract). Uses the
    verified ``_press_loo_mse_per_lambda`` / ``_ridge_dual_weights`` primitives
    directly. NOT dispatched by any production loop (gate-only reference).
    """
    grp = np.array(groups)
    fams = sorted(set(groups))
    ss = {r: [0.0, 0.0] for r in ("R1", "R2", "R3", "R4")}
    preds_out: dict[str, list] = {"folds": []}
    Yp = Yv[perm_row] if perm_row is not None else Yv
    YBp = YB[perm_row] if (perm_row is not None and YB is not None) else YB
    for fam in fams:
        te = np.where(grp == fam)[0]
        tr = np.where(grp != fam)[0]
        if len(tr) < 3 or len(te) == 0:
            continue
        dev = torch.device(device)
        # PCA basis is DRAW-INVARIANT (plan §3.5-S5: built on the UNPERMUTED train
        # rows; the permutation applies to the TARGET rows only — the parent's
        # sanctioned exchangeability variant). Basis reuses batched_pca_project —
        # itself asserted against the #810 _gram_top_k_pca primitive in the G2
        # gate, so the twin stays a RIDGE oracle.
        Ytr_basis = torch.from_numpy(Yv[tr]).double().unsqueeze(0).to(dev)
        mu_p, comps = batched_pca_project(Ytr_basis, k)

        def proj(R, _dev=dev, _mu=mu_p, _comps=comps):
            return (torch.from_numpy(R).double().unsqueeze(0).to(_dev) - _mu) @ _comps.transpose(
                1, 2
            )

        Ytr_pca = proj(Yp[tr])[0]
        YteA_pca = proj(Yp[te])[0]
        Xtr = torch.from_numpy(Xc[tr]).double().to(dev)
        mu = Xtr.mean(0)
        sd = Xtr.std(0, correction=0) + 1e-9
        Xtr_n = (Xtr - mu) / sd
        ymu = Ytr_pca.mean(0)
        Ytr_c = Ytr_pca - ymu
        mse = _press_loo_mse_per_lambda(Xtr_n, Ytr_c, RIDGE_LAMBDAS)
        best = RIDGE_LAMBDAS[int(torch.argmin(mse).item())]
        w = _ridge_dual_weights(Xtr_n, Ytr_c, best)
        XteA_n = (torch.from_numpy(Xc[te]).double().to(dev) - mu) / sd
        predA = ymu + XteA_n @ w
        fold_rec = {
            "family": fam,
            "best_lambda": best,
            "predA": predA.cpu().numpy().copy(),
            "ymu": ymu.cpu().numpy().copy(),
        }
        ss["R1"][0] += float(((YteA_pca - predA) ** 2).sum())
        ss["R1"][1] += float(((YteA_pca - ymu) ** 2).sum())
        if XB is not None:
            XteB_n = (torch.from_numpy(XB[te]).double().to(dev) - mu) / sd
            predB = ymu + XteB_n @ w
            fold_rec["predB"] = predB.cpu().numpy().copy()
            ss["R2"][0] += float(((YteA_pca - predB) ** 2).sum())
            ss["R2"][1] += float(((YteA_pca - ymu) ** 2).sum())
            if YBp is not None:
                YteB_pca = proj(YBp[te])[0]
                ss["R3"][0] += float(((YteB_pca - predA) ** 2).sum())
                ss["R3"][1] += float(((YteB_pca - ymu) ** 2).sum())
                ss["R4"][0] += float(((YteB_pca - predB) ** 2).sum())
                ss["R4"][1] += float(((YteB_pca - ymu) ** 2).sum())
        preds_out["folds"].append(fold_rec)
    out = {r: (1.0 - a / b if b > 1e-12 else float("nan")) for r, (a, b) in ss.items()}
    out["fold_records"] = preds_out["folds"]
    return out
