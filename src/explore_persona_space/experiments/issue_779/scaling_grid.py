"""Issue #779 (training-source-ablation-hg): 2D training-size scaling grid.

The 0-GPU analysis core of the amendment (plan v6 §4.4). Everything here runs on
CACHED (Arm A: pass_b LMSYS bundle) + GENERATED-ONCE (Arm B/C: the diverse
behavior corpus produced by ``scripts/issue779_gen_behavior_corpus.py``) tensors;
subsampling is free.

Objects computed (per trait, at the FROZEN read-out layer §4.5):

  1. The 2D scaling grid ``scaling_grid`` -- N_LMSYS x N_behavior x K subsamples,
     h fit (ridge PRIMARY) + g fit per cell, within-condition Pearson r on the
     FIXED held-out eval rig. Arm A = the N_behavior=0 column; Arm B = the
     N_LMSYS=0 row; Arm C = the interior cells (natural mix + 1:1 upsample).
     Vectorized over cells (ridge closed-form; batched MLP arm optional) --
     NEVER a serial per-cell AdamW loop (vectorize-many-cell-fits.md).

  2. The per-arm headline ``arm_comparison`` -- pv_raw / oracle / Arm A / Arm B /
     Arm C(natural,1:1) / g, with the three-direction pv contrast triple
     {r_B, Mᵀr_B, M⁺r_B} (the mentor's pv_pinv amendment).

  3. The held-out-question g ``g_holdout_question`` -- K-fold over the 20 eval
     questions, fit g on fit-fold rows across conditions, score within-condition
     r on the test fold (leakage-free; corrects body-error #2).

  4. The selection-symmetric per-draw x per-layer statistic matrix
     ``scaling_grid_layer_matrix`` -- the arm-vs-A headline read at EVERY layer
     (observed + shuffle-null rows) so the analyzer can recompute an honest
     max-selected band post-hoc without a re-run (selection-symmetric-nulls.md).

The g training rows on any cell are the ARM/CELL rows (reconciler note 1): Arm B
cells' g/h train on the behavior-corpus rows; Arm C cells' on the LMSYS +
behavior mix at the cell's (N_LMSYS, N_behavior) coordinates -- asserted in-code
(``_assert_cell_train_composition``). ``g_holdout_q`` is a SINGLE distinct output
key (not conflated with the grid's per-cell g); the parent's LOCO-g stays a
labeled continuity reference under a distinct key.
"""

from __future__ import annotations

import logging

import numpy as np

from explore_persona_space.experiments.issue_779 import fit_h as F
from explore_persona_space.experiments.issue_779 import metrics as M

logger = logging.getLogger("issue779.scaling_grid")

DEFAULT_N_LMSYS = (0, 100, 250, 500, 1000, 2000, 5000)
DEFAULT_N_BEHAVIOR = (0, 50, 100, 250, 500, 1000, 2000)
DEFAULT_K = 10
MODES = ("system", "many_shot")


# ── training-row assembly (the arm/cell rows; reconciler note 1) ──────────────


class TrainSource:
    """The h/g training rows for one arm/cell, split by source.

    Holds the FULL LMSYS (X_lmsys, Y_lmsys, y_lmsys) and behavior-corpus
    (X_beh, Y_beh, y_beh) matrices at a single read-out layer; a cell draws its
    (N_LMSYS, N_behavior) subset from these. Y_* is the mean-response profile v(x)
    (the target for the map h); y_* is the trait label (the target for g). LMSYS
    y may be None (Arm A g labels come from the regenerated lmsys_g_labels pass;
    absent -> g reads NaN for LMSYS-only cells, a legitimate label-floor finding).

    X: (N, H) c_last at the read-out layer. Y: (N, H) v(x) at the read-out layer.
    y: (N,) trait label or None.
    """

    def __init__(
        self,
        X_lmsys: np.ndarray,
        Y_lmsys: np.ndarray,
        y_lmsys: np.ndarray | None,
        X_beh: np.ndarray,
        Y_beh: np.ndarray,
        y_beh: np.ndarray | None,
    ):
        self.X_lmsys = X_lmsys
        self.Y_lmsys = Y_lmsys
        self.y_lmsys = y_lmsys
        self.X_beh = X_beh
        self.Y_beh = Y_beh
        self.y_beh = y_beh
        assert X_lmsys.shape[0] == Y_lmsys.shape[0]
        assert X_beh.shape[0] == Y_beh.shape[0]

    def n_lmsys(self) -> int:
        return self.X_lmsys.shape[0]

    def n_beh(self) -> int:
        return self.X_beh.shape[0]


def _subsample_indices(n_avail: int, n_take: int, rng: np.random.Generator) -> np.ndarray:
    """Random subsample of ``n_take`` indices from ``n_avail`` (no replacement).

    Clamps ``n_take`` to ``n_avail`` (a grid point beyond the realized corpus max
    trains on all available rows — reported via the realized-N in the cell)."""
    n_take = min(n_take, n_avail)
    if n_take <= 0:
        return np.empty(0, dtype=int)
    return rng.choice(n_avail, size=n_take, replace=False)


def assemble_cell_train(
    src: TrainSource,
    n_lmsys: int,
    n_behavior: int,
    rng: np.random.Generator,
    *,
    upsample_1to1: bool = False,
) -> dict:
    """Assemble one grid cell's h/g training rows from the arm/cell subset.

    Returns {"X","Y","y","n_lmsys_used","n_behavior_used","has_labels"}. When
    ``upsample_1to1`` and BOTH sources present, the behavior rows are repeated so
    the effective count matches the LMSYS count (the Arm C 1:1 weighting). The g
    label vector y is present only if EVERY source row contributing has a label
    (LMSYS labels may be absent -> y is None -> g is NaN for that cell, the
    label-floor finding).
    """
    li = _subsample_indices(src.n_lmsys(), n_lmsys, rng)
    bi = _subsample_indices(src.n_beh(), n_behavior, rng)
    parts_X, parts_Y, parts_y = [], [], []
    labels_ok = True
    if len(li):
        parts_X.append(src.X_lmsys[li])
        parts_Y.append(src.Y_lmsys[li])
        if src.y_lmsys is not None:
            parts_y.append(src.y_lmsys[li])
        else:
            labels_ok = False
    if len(bi):
        Xb, Yb = src.X_beh[bi], src.Y_beh[bi]
        yb = src.y_beh[bi] if src.y_beh is not None else None
        reps = 1
        if upsample_1to1 and len(li) and len(bi):
            reps = max(1, round(len(li) / len(bi)))
        for _ in range(reps):
            parts_X.append(Xb)
            parts_Y.append(Yb)
            if yb is not None:
                parts_y.append(yb)
            else:
                labels_ok = False
    if not parts_X:
        return {
            "X": np.empty((0, src.X_lmsys.shape[1])),
            "Y": np.empty((0, src.Y_lmsys.shape[1])),
            "y": None,
            "n_lmsys_used": 0,
            "n_behavior_used": 0,
            "has_labels": False,
        }
    X = np.concatenate(parts_X, axis=0)
    Y = np.concatenate(parts_Y, axis=0)
    y = np.concatenate(parts_y, axis=0) if (labels_ok and parts_y) else None
    return {
        "X": X,
        "Y": Y,
        "y": y,
        "n_lmsys_used": len(li),
        "n_behavior_used": len(bi),
        "has_labels": bool(y is not None),
    }


def _assert_cell_train_composition(
    cell_train: dict, arm: str, n_lmsys: int, n_behavior: int
) -> None:
    """Reconciler note 1: each cell's g/h train inputs ARE the arm/cell rows.

    Arm B cells (N_LMSYS=0) MUST contain behavior rows and NO LMSYS rows; Arm A
    cells (N_behavior=0) the converse; Arm C cells BOTH. Fail loud on any
    composition that does not match the cell's (N_LMSYS, N_behavior) coordinates
    -- a silent 'only pass_a' or 'only LMSYS' composition is exactly the bug this
    assert prevents.
    """
    nl, nb = cell_train["n_lmsys_used"], cell_train["n_behavior_used"]
    want_lmsys = n_lmsys > 0
    want_beh = n_behavior > 0
    if want_lmsys:
        assert nl > 0, f"{arm} cell ({n_lmsys},{n_behavior}) requested LMSYS rows but got {nl}"
    else:
        assert nl == 0, f"{arm} cell ({n_lmsys},{n_behavior}) must have 0 LMSYS rows, got {nl}"
    if want_beh:
        assert nb > 0, f"{arm} cell ({n_lmsys},{n_behavior}) requested behavior rows but got {nb}"
    else:
        assert nb == 0, f"{arm} cell ({n_lmsys},{n_behavior}) must have 0 behavior rows, got {nb}"


# ── h / g reads at the frozen layer against the FIXED eval rig ────────────────


def _within_condition_r(x: np.ndarray, eval_mat: dict, *, n_boot: int, seed: int) -> dict:
    """Within-condition Pearson r + bootstrap CI per mode for one monitor x.

    ``eval_mat`` is the per-(condition, question) eval matrix at the frozen layer
    (from issue779_stage1.build_eval_matrix): keys c_last, y, cond, mode.
    """
    out = {}
    for mode in MODES:
        sel = np.array([m == mode for m in eval_mat["mode"]])
        if not sel.any():
            out[mode] = {
                "point": float("nan"),
                "lo": float("nan"),
                "hi": float("nan"),
                "n_conditions": 0,
                "n_boot_valid": 0,
            }
            continue
        cond = eval_mat["cond"]
        cx, cy = [], []
        for c in np.unique(cond[sel]):
            m = sel & (cond == c)
            xi, yi = x[m], eval_mat["y"][m]
            fin = np.isfinite(xi) & np.isfinite(yi)
            if fin.sum() >= 3:
                cx.append(xi[fin])
                cy.append(yi[fin])
        out[mode] = M.bootstrap_within_condition_ci(cx, cy, n_boot=n_boot, seed=seed)
    return out


def fit_h_cell(X_tr: np.ndarray, Y_tr: np.ndarray, eval_mat: dict, rb_l: np.ndarray) -> dict:
    """Fit ridge h on a cell's train rows, apply to eval c_last, return reads.

    Returns {"dot", "cos", "W", "xmu", "xsd", "P", "Q"} where dot/cos are the
    per-eval-context readouts <h(c),r_B> / cos(h(c),r_B); W is the (H, H)
    STANDARDIZED-INPUT ridge weight matrix (the operator M for the pv_pinv M⁺
    read); xmu/xsd are the TRAIN standardization stats (needed to read the pv_pinv
    preimage in the SAME standardized coordinate system W was fit in — see
    ``pv_pinv_read``); P/Q are the LOW-RANK factors ``W = P @ Q`` (so pv_pinv can
    use the O(H r^2) compact SVD instead of the O(H^3) dense one). A cell with < 2
    train rows returns NaN reads / None. Computes ONE SVD (via ``_ridge_fit``) and
    derives BOTH the eval prediction and W/P/Q from it, so the map read and the
    pv_pinv preimage come from the SAME fit (never two independent SVDs).
    """
    Xev = eval_mat["c_last"]
    if X_tr.shape[0] < 2:
        nan = np.full(Xev.shape[0], np.nan)
        return {"dot": nan, "cos": nan, "W": None, "xmu": None, "xsd": None, "P": None, "Q": None}
    W, xmu, xsd, ymu, P, Q = _ridge_fit(X_tr, Y_tr)  # (H,H) map + stats + low-rank factors
    pred = ((np.asarray(Xev, dtype=np.float64) - xmu) / xsd) @ W + ymu  # (N_ev, H)
    return {
        "dot": F.dot_readout(pred, rb_l),
        "cos": F.cosine_readout(pred, rb_l),
        "W": W,
        "xmu": xmu,
        "xsd": xsd,
        "P": P,
        "Q": Q,
    }


def _ridge_fit(
    X_train: np.ndarray, Y_train: np.ndarray, *, lambdas: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GCV-ridge fit -> (W, xmu, xsd, ymu, P, Q): the standardized-input weight
    matrix + train stats + the LOW-RANK factors of W, from ONE SVD.

    Reproduces ``fit_h.ridge_fit_predict``'s GCV lambda + standardization exactly
    (same SVD, same GCV loop), returning the pieces the caller needs to derive
    BOTH the eval prediction (``(Xev - xmu)/xsd @ W + ymu``) AND the pv_pinv
    preimage (``W`` is the map operator M; intercepts/means cancel in the
    correlation). W is (H_in, H_out); xmu/xsd/ymu are (H,).

    ALSO returns the low-rank factors ``P (H, r)`` and ``Q (r, H)`` with
    ``W = P @ Q`` and ``r = rank(Xtr_n) <= n_train`` — because W is derived from
    an r-dimensional train subspace it has rank <= r, so its compact SVD (and
    hence the pv_pinv) can be computed in O(H r^2) from these factors instead of
    the O(H^3) dense H x H SVD (see ``pv_pinv_svd``). Deterministic.
    """
    if lambdas is None:
        lambdas = np.logspace(-2, 4, 13)
    Xtr = np.asarray(X_train, dtype=np.float64)
    Ytr = np.asarray(Y_train, dtype=np.float64)
    if Ytr.ndim == 1:
        Ytr = Ytr[:, None]
    n = Xtr.shape[0]
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    U, s, Vt = np.linalg.svd(Xtr_n, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Ytr_c
    best_lam, best_gcv = lambdas[0], np.inf
    for lam in lambdas:
        filt = s2 / (s2 + lam)
        Yhat = U @ (filt[:, None] * UtY)
        rss = float(np.sum((Ytr_c - Yhat) ** 2))
        dof = float(np.sum(filt))
        denom = (n - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else np.inf
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, lam
    filt = s / (s2 + best_lam)
    P = Vt.T * filt  # (H_in, r)
    Q = UtY  # (r, H_out)
    W = P @ Q  # (H_in, H_out) standardized-input map, rank <= r
    return W, xmu, xsd, ymu, P, Q


def pv_pinv_read(
    W: np.ndarray,
    rb_l: np.ndarray,
    eval_mat: dict,
    *,
    xmu: np.ndarray,
    xsd: np.ndarray,
    rank: int | None = None,
) -> np.ndarray:
    """pv_pinv (mentor amendment): read <c_std, w_pinv> with w_pinv = M⁺ r_B.

    M = W (the fitted linear ridge map). CRUCIAL: W was fit on the STANDARDIZED
    input ``(X_tr - xmu)/xsd`` (``fit_h_cell``/``_ridge_fit``), so its preimage
    ``w_pinv = M⁺ r_B`` lives in the SAME standardized coordinate system. The read
    must therefore be taken against the STANDARDIZED eval c_last
    ``c_std = (c_last - xmu)/xsd`` — NOT raw c_last. Reading raw c_last against a
    standardized-space operator's preimage mixes two spaces and is ~orthogonal to
    the intended M⁺r_B direction on heteroscedastic activations (the round-1
    BLOCKER, corr ≈ -0.03 vs the correct read). This is the exact adjoint of the
    transpose read <h(c),r_B>, which also standardizes c before applying W. ``rank``
    truncates the SVD (chosen on the TRAIN split, frozen before this read -- the
    noise-amplification guard); None = full-rank pinv (the diagnostic).
    """
    if W is None:
        return np.full(eval_mat["c_last"].shape[0], np.nan)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    return _pv_pinv_from_svd(U, s, Vt, rb_l, eval_mat, xmu=xmu, xsd=xsd, rank=rank)


def _pv_pinv_from_svd(
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    rb_l: np.ndarray,
    eval_mat: dict,
    *,
    xmu: np.ndarray,
    xsd: np.ndarray,
    rank: int | None = None,
) -> np.ndarray:
    """<c_std, M⁺ r_B> from a PRE-COMPUTED SVD of W (so both the frozen-rank and
    full-rank reads share one SVD — the H x H SVD is the pv_pinv cost).

    ``c_std = (c_last - xmu)/xsd`` — the eval c_last standardized on the fit's
    TRAIN stats, so the read is in the SAME space W (and hence M⁺r_B) lives in.
    """
    if rank is not None:
        rank = min(rank, len(s))
        U, s, Vt = U[:, :rank], s[:rank], Vt[:rank]
    s_inv = np.where(s > 1e-12, 1.0 / s, 0.0)
    # M⁺ = V diag(1/s) Uᵀ ; w_pinv = M⁺ r_B  (r_B lives in the OUTPUT/profile space)
    w_pinv = (Vt.T * s_inv) @ (U.T @ np.asarray(rb_l, dtype=np.float64))
    c_std = (np.asarray(eval_mat["c_last"], dtype=np.float64) - xmu) / xsd
    return c_std @ w_pinv


def pv_pinv_svd(W: np.ndarray | None, *, P: np.ndarray | None = None, Q: np.ndarray | None = None):
    """The compact SVD (U, s, Vt) of W, shared by ALL pv_pinv consumers of one arm.

    Returns (U, s, Vt) or None when W is None (< 2 train rows). Computing this
    ONCE per arm and threading it into pv_pinv_reads + the persisted fit state
    avoids the SVD being recomputed 3x per arm (the dominant pv_pinv cost).

    When the LOW-RANK factors ``W = P @ Q`` (P (H, r), Q (r, H), r <= n_train) are
    supplied, the compact SVD is computed in O(H r^2) instead of the O(H^3) dense
    H x H SVD: QR the tall factor P = U_p R_p, SVD the small r x H matrix
    M = R_p @ Q = U_m S V_m^T, then W = (U_p U_m) S V_m^T is W's compact SVD. This
    is EXACT (rank(W) <= r), and ~thousands of times cheaper at r << H (r=40,
    H=3584: 47s dense SVD -> milliseconds). Falls back to the dense SVD when the
    factors are absent.
    """
    if W is None:
        return None
    if P is not None and Q is not None:
        P = np.asarray(P, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        U_p, R_p = np.linalg.qr(P)  # U_p (H, r), R_p (r, r)
        M = R_p @ Q  # (r, H)
        U_m, s, Vt = np.linalg.svd(M, full_matrices=False)  # U_m (r, r), s (r,), Vt (r, H)
        U = U_p @ U_m  # (H, r)
        return U, s, Vt
    return np.linalg.svd(np.asarray(W, dtype=np.float64), full_matrices=False)


def pv_pinv_reads(
    W: np.ndarray,
    rb_l: np.ndarray,
    eval_mat: dict,
    *,
    xmu: np.ndarray,
    xsd: np.ndarray,
    rank: int | None,
    svd=None,
) -> tuple[np.ndarray, np.ndarray]:
    """(frozen-rank read, full-rank diagnostic read) from ONE SVD of W.

    The mentor amendment reports both the frozen-rank pv_pinv (headline) and the
    full-rank pinv (diagnostic); this computes the single H x H SVD once and
    slices it for both ranks, instead of two independent SVDs. Both reads use the
    STANDARDIZED eval c_last (see ``pv_pinv_read``). Pass a precomputed ``svd``
    (from ``pv_pinv_svd``) to share the H x H SVD with the persisted fit state.
    Returns (pv_pinv[rank], pv_pinv[full]). NaN reads when W is None (< 2 train
    rows).
    """
    n_ev = eval_mat["c_last"].shape[0]
    if W is None or xmu is None or xsd is None:
        nan = np.full(n_ev, np.nan)
        return nan, nan
    U, s, Vt = svd if svd is not None else np.linalg.svd(W, full_matrices=False)
    frozen = _pv_pinv_from_svd(U, s, Vt, rb_l, eval_mat, xmu=xmu, xsd=xsd, rank=rank)
    full = _pv_pinv_from_svd(U, s, Vt, rb_l, eval_mat, xmu=xmu, xsd=xsd, rank=None)
    return frozen, full


def fit_g_cell(X_tr: np.ndarray, y_tr: np.ndarray | None, eval_mat: dict) -> tuple[np.ndarray, int]:
    """Fit ridge g (c_last -> trait label) on a cell's train rows; read on eval.

    DROP-NEVER-COERCE (llm-judging.md rule 9): a malformed / missing judge label
    arrives here as NaN (assemble_cell_train propagates NaN for un-labeled LMSYS
    rows). Those NaN rows are DROPPED from the g fit — never fed into ridge (a
    single NaN poisons the closed-form SVD, turning g into all-NaN). The h path is
    UNCHANGED: h uses every row (v(x) has no missing-label problem).

    Returns ``(g_pred, n_dropped)`` — the per-eval-context g score (all-NaN if the
    cell has no finite-label rows, the Arm A label-floor case) and the count of
    NaN-label train rows dropped (reported per source in the output JSON).
    """
    Xev = eval_mat["c_last"]
    if y_tr is None:
        return np.full(Xev.shape[0], np.nan), 0
    y_arr = np.asarray(y_tr, dtype=np.float64)
    finite = np.isfinite(y_arr)
    n_dropped = int((~finite).sum())
    X_fit, y_fit = X_tr[finite], y_arr[finite]
    if X_fit.shape[0] < 2 or float(np.std(y_fit)) < 1e-9:
        return np.full(Xev.shape[0], np.nan), n_dropped
    return F.ridge_fit_predict(X_fit, y_fit, Xev), n_dropped


# ── the 2D grid driver ────────────────────────────────────────────────────────


def run_scaling_grid(
    src: TrainSource,
    eval_mat: dict,
    rb_l: np.ndarray,
    *,
    n_lmsys_grid=DEFAULT_N_LMSYS,
    n_behavior_grid=DEFAULT_N_BEHAVIOR,
    k_subsamples: int = DEFAULT_K,
    n_boot: int = 1000,
    base_seed: int = 0,
    upsample_1to1: bool = False,
) -> dict:
    """The 7x7xK grid of within-condition r for h (dot/cos) + g, per cell.

    Vectorization: the h read is a closed-form ridge fit per cell (cheap); the
    per-cell loop is over the 7x7xK grid of DISTINCT (train-subset) fits -- these
    are genuinely different training matrices, not a fused-output-dim loop, so
    the ridge closed form IS the batched primitive (no serial AdamW). Returns
    {"cells": [...], "grid_shape": (nL, nB, K)}.
    """
    cells = []
    for il, nL in enumerate(n_lmsys_grid):
        for ib, nB in enumerate(n_behavior_grid):
            if nL == 0 and nB == 0:
                continue  # the empty cell is undefined
            arm = "arm_a_lmsys" if nB == 0 else ("arm_b_behavior" if nL == 0 else "arm_c")
            for k in range(k_subsamples):
                rng = np.random.default_rng(base_seed + 1000 * il + 100 * ib + k)
                ct = assemble_cell_train(src, nL, nB, rng, upsample_1to1=upsample_1to1)
                _assert_cell_train_composition(ct, arm, nL, nB)
                h = fit_h_cell(ct["X"], ct["Y"], eval_mat, rb_l)
                g, g_n_dropped = fit_g_cell(ct["X"], ct["y"], eval_mat)
                seed = base_seed + k
                r_dot = _within_condition_r(h["dot"], eval_mat, n_boot=n_boot, seed=seed)
                r_g = _within_condition_r(g, eval_mat, n_boot=n_boot, seed=seed)
                cells.append(
                    {
                        "arm": arm,
                        "n_lmsys": int(nL),
                        "n_behavior": int(nB),
                        "n_lmsys_used": ct["n_lmsys_used"],
                        "n_behavior_used": ct["n_behavior_used"],
                        "subsample": k,
                        "upsample_1to1": upsample_1to1,
                        "h_ridge_dot_r": {m: r_dot[m] for m in MODES},
                        "g_ridge_r": {m: r_g[m] for m in MODES},
                        "g_labels_dropped_nan": g_n_dropped,
                        "has_labels": ct["has_labels"],
                    }
                )
    return {
        "cells": cells,
        "grid_shape": [len(n_lmsys_grid), len(n_behavior_grid), k_subsamples],
        "n_lmsys_grid": list(n_lmsys_grid),
        "n_behavior_grid": list(n_behavior_grid),
    }


# ── held-out-question g (K-fold over eval questions; body-error #2 fix) ────────


def run_g_holdout_question(
    eval_mat_with_q: dict, *, k_folds: int = 5, n_boot: int = 1000, base_seed: int = 0
) -> dict:
    """Leakage-free g: K-fold over the 20 eval questions.

    Fit g (ridge, c_last->trait) on the fit-fold rows ACROSS conditions, score
    within-condition Pearson r on the test-fold rows (g never sees the test-fold
    labels). ``eval_mat_with_q`` extends build_eval_matrix with a ``question``
    array (the eval question index per row). This is the closest cached-data
    mirror of PV's own question-level held-out unit (/tmp/issue779_holdout_design
    scheme b). SINGLE distinct output key ``g_holdout_question`` -- never conflated
    with the grid's per-cell g (reconciler note 1).
    """
    q = eval_mat_with_q["question"]
    X = eval_mat_with_q["c_last"]
    y = np.asarray(eval_mat_with_q["y"], dtype=np.float64)
    uniq_q = np.unique(q)
    rng = np.random.default_rng(base_seed)
    perm = rng.permutation(uniq_q)
    folds = np.array_split(perm, min(k_folds, len(uniq_q)))
    per_mode = {m: [] for m in MODES}
    total_dropped = 0
    for fi, test_q in enumerate(folds):
        test_mask = np.isin(q, test_q)
        fit_mask = ~test_mask
        # DROP-NEVER-COERCE (llm-judging.md rule 9): a fit-fold row with a missing
        # / malformed judge label (NaN y) is dropped from the g fit, never fed
        # into ridge. The test fold keeps only finite-y rows too (a NaN test label
        # carries no r information).
        fit_fin = fit_mask & np.isfinite(y)
        total_dropped += int((fit_mask & ~np.isfinite(y)).sum())
        if fit_fin.sum() < 2 or float(np.std(y[fit_fin])) < 1e-9:
            continue
        g_pred = F.ridge_fit_predict(X[fit_fin], y[fit_fin], X[test_mask])
        # score within-condition r on the test fold
        test_mat = {
            "c_last": X[test_mask],
            "y": y[test_mask],
            "cond": eval_mat_with_q["cond"][test_mask],
            "mode": eval_mat_with_q["mode"][test_mask],
        }
        r = _within_condition_r(g_pred, test_mat, n_boot=n_boot, seed=base_seed + fi)
        for m in MODES:
            per_mode[m].append(r[m])
    # aggregate folds: mean point + mean CI bounds over valid folds
    out = {}
    for m in MODES:
        pts = [d["point"] for d in per_mode[m] if np.isfinite(d.get("point", np.nan))]
        out[m] = {
            "point": float(np.mean(pts)) if pts else float("nan"),
            "n_folds_valid": len(pts),
            "per_fold": per_mode[m],
        }
    return {"k_folds": len(folds), "modes": out, "labels_dropped_nan": total_dropped}
