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
            "li": np.empty(0, dtype=int),
            "bi": np.empty(0, dtype=int),
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
        # the drawn per-source indices — the per-cell held-out recon (plan v6
        # §4.4, B1) reads the COMPLEMENT of these (rows the cell never trained on)
        "li": li,
        "bi": bi,
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


def fit_h_cell(
    X_tr: np.ndarray, Y_tr: np.ndarray, eval_mat: dict, rb_l: np.ndarray, *, need_W: bool = True
) -> dict:
    """Fit ridge h on a cell's train rows, apply to eval c_last, return reads.

    Returns {"dot", "cos", "W", "xmu", "xsd", "P", "Q", "fit"} where dot/cos are
    the per-eval-context readouts <h(c),r_B> / cos(h(c),r_B); W is the (H, H)
    STANDARDIZED-INPUT ridge weight matrix (the operator M for the pv_pinv M⁺
    read); xmu/xsd are the TRAIN standardization stats (needed to read the pv_pinv
    preimage in the SAME standardized coordinate system W was fit in — see
    ``pv_pinv_read``); P/Q are the LOW-RANK factors ``W = P @ Q`` (so pv_pinv can
    use the O(H r^2) compact SVD instead of the O(H^3) dense one); ``fit`` is the
    underlying ``fit_h.RidgeFitCore`` (shared with the cell's g fit + the per-cell
    held-out recon read). A cell with < 2 train rows returns NaN reads / None.
    ONE decomposition (via ``RidgeFitCore``) derives the eval prediction, W/P/Q,
    the g fit, and the recon predictions (never independent SVDs).

    ``need_W=False`` (grid cells, v79 fix): skips forming the (H, H) W — grid
    cells consume only the dot/cos readouts + recon, so predictions go through
    the low-rank factors; W/P/Q return None. ``run_arm_comparison`` keeps
    ``need_W=True`` for the pv_pinv reads.
    """
    Xev = eval_mat["c_last"]
    if X_tr.shape[0] < 2:
        nan = np.full(Xev.shape[0], np.nan)
        return {
            "dot": nan,
            "cos": nan,
            "W": None,
            "xmu": None,
            "xsd": None,
            "P": None,
            "Q": None,
            "fit": None,
        }
    core = F.RidgeFitCore(X_tr, Y_tr)
    pred = core.predict(Xev)  # (N_ev, H)
    out = {
        "dot": F.dot_readout(pred, rb_l),
        "cos": F.cosine_readout(pred, rb_l),
        "xmu": core.xmu,
        "xsd": core.xsd,
        "fit": core,
    }
    if need_W:
        out["W"], out["P"], out["Q"] = core.W(), core.P, core.Q
    else:
        out["W"] = out["P"] = out["Q"] = None
    return out


def _ridge_fit(
    X_train: np.ndarray, Y_train: np.ndarray, *, lambdas: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GCV-ridge fit -> (W, xmu, xsd, ymu, P, Q): the standardized-input weight
    matrix + train stats + the LOW-RANK factors of W, from ONE decomposition.

    Thin compat wrapper over ``fit_h.RidgeFitCore`` (the eigh fast path, v79
    fixes 2+4) — same standardization / GCV grid / selected lambda as
    ``fit_h.ridge_fit_predict``; equivalence vs the numpy-SVD serial reference
    (``_ridge_fit_svd_reference``) is gated by ``verify_live_ridge``. W is
    (H_in, H_out); xmu/xsd/ymu are (H,); ``W = P @ Q`` with rank <= min(n, H)
    (so ``pv_pinv_svd`` can use the compact O(H r^2) SVD). Deterministic.
    """
    core = F.RidgeFitCore(X_train, Y_train, lambdas=lambdas)
    return core.W(), core.xmu, core.xsd, core.ymu, core.P, core.Q


def _ridge_fit_svd_reference(
    X_train: np.ndarray, Y_train: np.ndarray, *, lambdas: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """SERIAL REFERENCE for the live-ridge equivalence gate (numpy SVD + the
    ORIGINAL materialized-Yhat GCV loop — the pre-v79 implementation, verbatim).

    Kept ONLY as the independent reference ``verify_live_ridge`` + the
    regression tests compare ``RidgeFitCore`` against (the
    ``vectorized_mlp_skill.assert_matches_reference`` pattern) — never called on
    the production path. Returns the ``_ridge_fit`` tuple + the selected lambda.
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
    return W, xmu, xsd, ymu, P, Q, float(best_lam)


def verify_live_ridge(*, seed: int = 0, tol: float = 1e-7) -> dict:
    """LIVE-path equivalence gate (v79 fix 6): the eigh ``RidgeFitCore`` must
    reproduce the numpy-SVD serial reference on BOTH forms (dual N<=H, primal
    N>H) — identical selected lambda, predictions/W within ``tol`` — and the
    shared scalar-g path must reproduce ``fit_h.ridge_fit_predict``. Raises
    AssertionError on any mismatch; returns the measured deltas. Called by the
    grid CLI under ``--verify-vectorized`` (this gates the LIVE ridge path; the
    old check gated only the unused MLP helper).

    Fixtures cover well-conditioned designs AND near-low-rank ones (r10, the
    r9-review MINOR): ``*_lowrank`` fixtures build ``X = Z @ B + 0.01·noise``
    at rank ~6, the ill-conditioned regime production activations live in
    (near-degenerate Gram/covariance spectra), where the eigh-vs-SVD agreement
    is least trivial.
    """
    rng = np.random.default_rng(seed)
    out = {}
    fixtures = {
        "dual": (24, 40, None),
        "primal": (80, 24, None),
        "dual_lowrank": (40, 64, 6),
        "primal_lowrank": (120, 36, 6),
    }
    for name, (n, h, rank) in fixtures.items():
        if rank is None:
            X = rng.standard_normal((n, h))
        else:
            # near-low-rank design: rank-`rank` signal + small isotropic noise.
            X = rng.standard_normal((n, rank)) @ rng.standard_normal((rank, h))
            X = X + 0.01 * rng.standard_normal((n, h))
        W_true = rng.standard_normal((h, h))
        Y = X @ W_true + 0.1 * rng.standard_normal((n, h))
        Xev = rng.standard_normal((13, h))
        core = F.RidgeFitCore(X, Y)
        W_ref, xmu, xsd, ymu, _P, _Q, lam_ref = _ridge_fit_svd_reference(X, Y)
        pred_ref = ((Xev - xmu) / xsd) @ W_ref + ymu
        d_pred = float(np.max(np.abs(core.predict(Xev) - pred_ref)))
        d_w = float(np.max(np.abs(core.W() - W_ref)))
        assert core.form == name.split("_")[0], (core.form, name)
        assert core.lam == lam_ref, f"{name}: lambda diverged ({core.lam} vs {lam_ref})"
        assert d_pred < tol and d_w < tol, f"{name}: d_pred={d_pred} d_w={d_w} (tol {tol})"
        y = X @ rng.standard_normal(h) + 0.1 * rng.standard_normal(n)
        d_g = float(np.max(np.abs(core.predict_scalar(y, Xev) - F.ridge_fit_predict(X, y, Xev))))
        assert d_g < tol, f"{name}: scalar-g path d_g={d_g} (tol {tol})"
        out[name] = {"d_pred": d_pred, "d_w": d_w, "d_g": d_g, "lam": core.lam}
    logger.info("[grid-fast] live-ridge equivalence gate PASS: %s", out)
    return out


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
    BLOCKER, corr ≈ -0.03 vs the correct read). w_pinv shares the transpose read's
    singular directions but weights them 1/sigma instead of sigma (equal only for
    orthogonal W) and satisfies the defining preimage property ``w_pinv @ W ≈ r_B``
    for r_B in the map's image (the round-2 orientation BLOCKER computed the
    transposed map's pseudoinverse instead). ``rank`` truncates the SVD (chosen on
    the TRAIN split, frozen before this read -- the noise-amplification guard);
    None = full-rank pinv (the diagnostic).
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
    # The fitted map is h(c) = c_std @ W (i.e. M = Wᵀ in column convention), so the
    # preimage operator is M⁺ = (Wᵀ)⁺ = U diag(1/s) Vt for W = U s Vt. w_pinv = M⁺r_B
    # satisfies the defining property w_pinv @ W ≈ r_B for r_B in the map's image
    # (r_B lives in the OUTPUT/profile space). NOT (Vt.T*s_inv)@(U.T@r_B) — that is
    # the pseudoinverse of the TRANSPOSED map (the round-2 orientation blocker).
    w_pinv = (U * s_inv) @ (Vt @ np.asarray(rb_l, dtype=np.float64))
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


def fit_g_cell(
    X_tr: np.ndarray,
    y_tr: np.ndarray | None,
    eval_mat: dict,
    *,
    h_fit: F.RidgeFitCore | None = None,
) -> tuple[np.ndarray, int]:
    """Fit ridge g (c_last -> trait label) on a cell's train rows; read on eval.

    DROP-NEVER-COERCE (llm-judging.md rule 9): a malformed / missing judge label
    arrives here as NaN (assemble_cell_train propagates NaN for un-labeled LMSYS
    rows). Those NaN rows are DROPPED from the g fit — never fed into ridge (a
    single NaN poisons the closed-form SVD, turning g into all-NaN). The h path is
    UNCHANGED: h uses every row (v(x) has no missing-label problem).

    ``h_fit`` (v79 fix 3): the cell's h ``RidgeFitCore`` on the SAME X rows —
    when EVERY label is finite (no rows dropped) g shares its decomposition
    (``predict_scalar``, mathematically identical to ``ridge_fit_predict`` on the
    same rows) instead of re-decomposing the same X. Any dropped row changes the
    fit rows, so the shared path is skipped and the finite-subset refit runs
    exactly as before.

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
    if h_fit is not None and n_dropped == 0 and h_fit.n == X_tr.shape[0]:
        return h_fit.predict_scalar(y_arr, Xev), 0
    return F.ridge_fit_predict(X_fit, y_fit, Xev), n_dropped


# ── the 2D grid driver ────────────────────────────────────────────────────────

# Per-source cap on the held-out rows scored for the per-cell recon read (B1) —
# recon on 500 held-out rows is a stable estimate at ~0.1 s/cell; the complement
# slice is deterministic (sorted setdiff1d prefix), so no extra RNG draws.
RECON_HELDOUT_CAP = 500


def _heldout_recon(fit, src: TrainSource, ct: dict) -> dict:
    """Per-cell HELD-OUT reconstruction R2 / mean-cosine (plan v6 §4.4, B1).

    For each source, score the fitted h on rows the cell did NOT train on (the
    complement of the drawn indices, capped at ``RECON_HELDOUT_CAP``). For a
    source the cell used in FULL the complement is empty -> NaN with
    ``n_heldout: 0``; for a source the cell used NOT AT ALL (e.g. the behavior
    side of an Arm A cell) the read is an out-of-arm TRANSFER recon — labeled by
    the source key, the analyzer disambiguates via the cell's arm.
    """
    out = {}
    for tag, X_all, Y_all, used in (
        ("lmsys", src.X_lmsys, src.Y_lmsys, ct["li"]),
        ("behavior", src.X_beh, src.Y_beh, ct["bi"]),
    ):
        held = np.setdiff1d(np.arange(X_all.shape[0]), used)[:RECON_HELDOUT_CAP]
        if fit is None or held.size == 0:
            out[tag] = {
                "r2": float("nan"),
                "mean_cosine": float("nan"),
                "n_heldout": int(held.size),
                "selection": "prefix",
            }
            continue
        m = F.reconstruction_metrics(fit.predict(X_all[held]), Y_all[held])
        out[tag] = {
            "r2": m["r2"],
            "mean_cosine": m["mean_cosine"],
            "n_heldout": int(held.size),
            # r9-review MINOR: the held-out subset is the SORTED PREFIX of the
            # complement (deterministic, no extra RNG draws) — recorded so the
            # analyzer knows the selection rule when corpus order is non-random.
            "selection": "prefix",
        }
    return out


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
    skip_edges: bool = False,
) -> dict:
    """The 7x7xK grid of within-condition r for h (dot/cos) + g, per cell.

    Vectorization: the h read is a closed-form ridge fit per cell (cheap); the
    per-cell loop is over the 7x7xK grid of DISTINCT (train-subset) fits -- these
    are genuinely different training matrices, not a fused-output-dim loop, so
    the ridge closed form IS the batched primitive (no serial AdamW). Returns
    {"cells": [...], "grid_shape": (nL, nB, K)}.

    v79 additions: (1) each cell carries ``recon_heldout`` — held-out
    reconstruction R2/cosine per source (plan v6 §4.4, the B1 drift fix);
    (2) FULL-ROW cells (every source drawn full-or-zero, so the K subsamples are
    permutations of the SAME training set — ridge is permutation-invariant) fit
    ONCE and reuse the fit across k (per-k bootstrap seeds are UNCHANGED, so the
    K cells still differ in their CI draws exactly as before); (3) the cell's g
    shares the h decomposition when no NaN label is dropped; (4)
    ``skip_edges=True`` drops the nL==0 / nB==0 edge cells (the v81 interior-only
    relaunch — edges computed main-side).
    """
    cells = []
    n_fits = 0
    n_reused = 0
    for il, nL in enumerate(n_lmsys_grid):
        for ib, nB in enumerate(n_behavior_grid):
            if nL == 0 and nB == 0:
                continue  # the empty cell is undefined
            if skip_edges and (nL == 0 or nB == 0):
                continue  # v81 interior-only relaunch: edges computed main-side
            arm = "arm_a_lmsys" if nB == 0 else ("arm_b_behavior" if nL == 0 else "arm_c")
            # A cell whose every source is drawn FULL (or not at all) trains on
            # the same SET for every k — the k draws only permute row order and
            # ridge is permutation-invariant, so fit once and reuse (audit iii).
            full_row_cell = (nL == 0 or nL >= src.n_lmsys()) and (nB == 0 or nB >= src.n_beh())
            cached: tuple | None = None
            for k in range(k_subsamples):
                rng = np.random.default_rng(base_seed + 1000 * il + 100 * ib + k)
                ct = assemble_cell_train(src, nL, nB, rng, upsample_1to1=upsample_1to1)
                _assert_cell_train_composition(ct, arm, nL, nB)
                if full_row_cell and cached is not None:
                    h, g, g_n_dropped, recon = cached
                    n_reused += 1
                else:
                    h = fit_h_cell(ct["X"], ct["Y"], eval_mat, rb_l, need_W=False)
                    g, g_n_dropped = fit_g_cell(ct["X"], ct["y"], eval_mat, h_fit=h["fit"])
                    recon = _heldout_recon(h["fit"], src, ct)
                    n_fits += 1
                    if full_row_cell:
                        cached = (h, g, g_n_dropped, recon)
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
                        "recon_heldout": recon,
                    }
                )
    logger.info(
        "[grid-fast] run_scaling_grid: %d cells (%d fits, %d full-row reuses, skip_edges=%s)",
        len(cells),
        n_fits,
        n_reused,
        skip_edges,
    )
    return {
        "cells": cells,
        "grid_shape": [len(n_lmsys_grid), len(n_behavior_grid), k_subsamples],
        "n_lmsys_grid": list(n_lmsys_grid),
        "n_behavior_grid": list(n_behavior_grid),
        "skip_edges": bool(skip_edges),
        "recon_heldout_cap": RECON_HELDOUT_CAP,
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


# ── canonical-artifact merge + completeness + edges composition (r10) ─────────
#
# The r6-review BLOCKER (`partial-scaling-grid-overwrite`): a --skip-edges /
# --grid-variants-subset relaunch used to write its PARTIAL `scaling` dict to
# the canonical scaling_grid.json path unconditionally, silently replacing the
# plan-required full 7x7xK grid. The guard below makes every canonical write
# (a) MERGE-safe — an existing artifact's traits/modes/variants/cells are
# preserved, new cells win on key collision, axis/frozen-layer mismatches fail
# loud — and (b) SELF-DESCRIBING — a top-level `complete` bool + a
# machine-readable `completeness` record of omitted variants/coordinates and
# realized-vs-planned cell counts. `compose_edges_into_scaling` is the operative
# v81 composition step: interior cells (VM-side) + pod-computed edge cells
# (eval_results/issue_779/batch2_edges.json, `issue779_edges` format).

GRID_VARIANT_SLOTS = ("natural", "upsample_1to1", "secondary_10rollout")
_ENTRY_SCALAR_KEYS = ("frozen_layer", "behavior_agg")


def _cell_key(cell: dict) -> tuple[int, int, int]:
    """The identity of a grid cell within one variant block."""
    return (int(cell["n_lmsys"]), int(cell["n_behavior"]), int(cell["subsample"]))


def expected_coords(n_lmsys_grid, n_behavior_grid, *, skip_edges: bool = False) -> set:
    """The planned (n_lmsys, n_behavior) coordinates of a grid block.

    Mirrors run_scaling_grid's enumeration: the (0,0) cell is always undefined;
    ``skip_edges`` additionally drops the nL==0 / nB==0 edge cells."""
    coords = set()
    for nL in n_lmsys_grid:
        for nB in n_behavior_grid:
            if int(nL) == 0 and int(nB) == 0:
                continue
            if skip_edges and (int(nL) == 0 or int(nB) == 0):
                continue
            coords.add((int(nL), int(nB)))
    return coords


def _cells_equal(a, b) -> bool:
    """Deep equality that treats NaN == NaN (cells legitimately carry NaN reads
    — lo/hi on composed edge cells, g on label-less cells — and a bare ``!=``
    would flag an IDENTICAL recompose as a conflict)."""
    if isinstance(a, float) and isinstance(b, float):
        return (a == b) or (np.isnan(a) and np.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_cells_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_cells_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def merge_grid_block(prev: dict | None, new: dict | None, *, on_collision: str = "new_wins"):
    """Merge two ``run_scaling_grid`` outputs for the SAME variant slot.

    Cells are keyed by (n_lmsys, n_behavior, subsample). ``on_collision``:
    ``"new_wins"`` (checkpoint re-writes / re-runs) or ``"refuse"`` (edges
    composition — raise ValueError on a key collision with CONFLICTING values;
    an identical cell is idempotent and allowed either way). Fails loud on any
    axis or k_subsamples mismatch — merging cells fit on different grids or
    subsample plans would silently mix incomparable draws. The merged block's
    ``skip_edges`` flag is the AND of the two inputs (it records whether edge
    cells were ever computed into this block)."""
    if prev is None:
        return new
    if new is None:
        return prev
    for k in ("n_lmsys_grid", "n_behavior_grid"):
        if list(prev[k]) != list(new[k]):
            raise ValueError(f"grid axis mismatch on merge ({k}): {prev[k]} vs {new[k]}")
    kp, kn = int(prev["grid_shape"][2]), int(new["grid_shape"][2])
    if kp != kn:
        raise ValueError(f"k_subsamples mismatch on merge: {kp} vs {kn}")
    cells = {_cell_key(c): c for c in prev["cells"]}
    n_collide = 0
    for c in new["cells"]:
        key = _cell_key(c)
        if key in cells and not _cells_equal(cells[key], c):
            n_collide += 1
            if on_collision == "refuse":
                raise ValueError(
                    f"conflicting cell values at (n_lmsys={key[0]}, n_behavior={key[1]}, "
                    f"subsample={key[2]}) — refusing to overwrite (on_collision='refuse')"
                )
        cells[key] = c
    merged = dict(new)
    merged["cells"] = [cells[k] for k in sorted(cells)]
    merged["skip_edges"] = bool(prev.get("skip_edges", False)) and bool(
        new.get("skip_edges", False)
    )
    if n_collide:
        # only reachable under on_collision="new_wins" (a forced overwrite);
        # identical rewrites are not collisions, so any hit is a REAL data change.
        logger.warning("[grid-merge] %d colliding cells overwritten (new wins)", n_collide)
    return merged


def merge_scaling_traits(prev: dict, new: dict, *, on_collision: str = "new_wins") -> dict:
    """Merge two scaling_grid.json ``traits`` trees (trait -> mode -> entry).

    Preserves prior traits/modes/variant blocks absent from the new tree; merges
    shared variant blocks via ``merge_grid_block``; fails loud when a shared
    (trait, mode) entry disagrees on frozen_layer / behavior_agg (cells read at
    different layers are not mergeable)."""
    out: dict = {}
    for trait in sorted(set(prev) | set(new)):
        pt, nt = prev.get(trait, {}), new.get(trait, {})
        out[trait] = {}
        for mode in sorted(set(pt) | set(nt)):
            pe, ne = pt.get(mode), nt.get(mode)
            if pe is None or ne is None:
                out[trait][mode] = ne if pe is None else pe
                continue
            for k in _ENTRY_SCALAR_KEYS:
                if k in pe and k in ne and pe[k] != ne[k]:
                    raise ValueError(
                        f"{trait}/{mode}: {k} mismatch on merge ({pe[k]!r} vs {ne[k]!r})"
                    )
            entry: dict = {}
            for k in sorted(set(pe) | set(ne)):
                if k in ("natural", "upsample_1to1"):
                    entry[k] = merge_grid_block(pe.get(k), ne.get(k), on_collision=on_collision)
                elif k == "secondary_10rollout":
                    ps, ns = pe.get(k), ne.get(k)
                    if ps is None or ns is None:
                        entry[k] = ns if ps is None else ps
                    else:
                        if ps.get("behavior_agg") != ns.get("behavior_agg"):
                            raise ValueError(
                                f"{trait}/{mode}/secondary_10rollout: behavior_agg mismatch "
                                f"({ps.get('behavior_agg')!r} vs {ns.get('behavior_agg')!r})"
                            )
                        entry[k] = {
                            **ns,
                            "natural": merge_grid_block(
                                ps.get("natural"), ns.get("natural"), on_collision=on_collision
                            ),
                        }
                else:
                    entry[k] = ne.get(k, pe.get(k))
            out[trait][mode] = entry
    return out


def _variant_block(entry: dict, variant: str) -> dict | None:
    """The run_scaling_grid block for one variant slot of a (trait, mode) entry."""
    if variant == "secondary_10rollout":
        return (entry.get(variant) or {}).get("natural")
    return entry.get(variant)


def grid_completeness(
    traits_tree: dict,
    *,
    expected_traits,
    expected_modes=MODES,
    expected_variants=GRID_VARIANT_SLOTS,
    coord_cap: int = 20,
) -> dict:
    """Machine-readable completeness record for a scaling_grid ``traits`` tree.

    ``complete`` iff EVERY (trait, mode, variant) block exists AND covers every
    planned (nL, nB) coordinate of its own declared axes with >=1 subsample.
    Coordinate coverage — not full-k coverage — is the completeness criterion:
    the operative v81 design computes edges pod-side at k_draws=5 vs the
    interior k=10, so coordinates below the planned k are RECORDED
    (``coords_below_planned_k``) but do not flip ``complete``. Realized
    coordinates outside the declared axes (e.g. the pod edges' behavior-axis
    N=2400 vs the planned 2000) are recorded as ``extra_coords``."""
    blocks: dict = {}
    complete = True
    variants_complete = dict.fromkeys(expected_variants, True)
    for trait in expected_traits:
        for mode in expected_modes:
            entry = traits_tree.get(trait, {}).get(mode)
            for variant in expected_variants:
                key = f"{trait}/{mode}/{variant}"
                block = _variant_block(entry, variant) if entry is not None else None
                if block is None:
                    blocks[key] = {"present": False}
                    complete = False
                    variants_complete[variant] = False
                    continue
                planned = expected_coords(block["n_lmsys_grid"], block["n_behavior_grid"])
                k_planned = int(block["grid_shape"][2])
                counts: dict[tuple[int, int], int] = {}
                for c in block["cells"]:
                    co = (int(c["n_lmsys"]), int(c["n_behavior"]))
                    counts[co] = counts.get(co, 0) + 1
                realized = set(counts)
                missing = sorted(planned - realized)
                extra = sorted(realized - planned)
                below_k = sorted(co for co in realized & planned if counts[co] < k_planned)
                blocks[key] = {
                    "present": True,
                    "k_planned": k_planned,
                    "n_coords_planned": len(planned),
                    "n_coords_realized": len(realized & planned),
                    "n_cells_planned": len(planned) * k_planned,
                    "n_cells_realized": int(sum(counts.values())),
                    "n_missing_coords": len(missing),
                    "missing_coords": [list(co) for co in missing[:coord_cap]],
                    "extra_coords": [list(co) for co in extra[:coord_cap]],
                    "coords_below_planned_k": [list(co) for co in below_k[:coord_cap]],
                }
                if missing:
                    complete = False
                    variants_complete[variant] = False
    return {
        "complete": complete,
        "variants_complete": variants_complete,
        "criterion": (
            "coordinate coverage: every (trait, mode, variant) block present AND every "
            "planned (n_lmsys, n_behavior) coordinate realized with >=1 subsample; "
            "k-shortfalls recorded in coords_below_planned_k, never flip complete"
        ),
        "blocks": blocks,
    }


def stamp_completeness(scaling: dict, *, expected_traits, run_flags: dict | None = None) -> dict:
    """Stamp top-level ``complete`` + ``completeness`` onto a scaling artifact.

    The record is derived from the ARTIFACT (post-merge), not the run flags —
    so a --skip-edges run merged over a prior full grid can legitimately read
    ``complete: true``, and the final canonical artifact flips to true exactly
    when all planned cells are present (the v81 composition contract).
    ``run_flags`` (skip_edges / grid_variants / traits / smoke) are recorded
    for provenance. Mutates ``scaling`` in place; returns the record."""
    rec = grid_completeness(scaling.get("traits", {}), expected_traits=expected_traits)
    if run_flags is not None:
        rec["last_write_run_flags"] = run_flags
    scaling["complete"] = rec["complete"]
    scaling["completeness"] = rec
    return rec


def _edge_draw_to_cell(
    draw: dict, *, n_lmsys: int, n_behavior: int, upsample_1to1: bool, edges_behavior_agg: str
) -> dict:
    """Adapt one issue779_edges draw into the run_scaling_grid cell schema.

    The pod edges carry per-mode point reads (no bootstrap CI — lo/hi are NaN
    with n_boot_valid=0) and a different recon protocol (kept under
    ``edges_extra``, never faked into ``recon_heldout``); composed cells are
    stamped ``source: "pod_edges"`` so the analyzer can tell them from
    VM-interior cells (which carry no ``source`` key), plus
    ``edges_behavior_agg`` — the behavior-side aggregation the producing edges
    artifact was computed under (r11; inert for nL-axis cells, which contain no
    behavior rows, but recorded uniformly for provenance)."""
    assert (n_lmsys == 0) != (n_behavior == 0), (n_lmsys, n_behavior)

    def _read(field: str) -> dict:
        out = {}
        for m in MODES:
            d = (draw.get(field) or {}).get(m)
            if d is None:
                out[m] = {
                    "point": float("nan"),
                    "lo": float("nan"),
                    "hi": float("nan"),
                    "n_conditions": 0,
                    "n_boot_valid": 0,
                }
            else:
                out[m] = {
                    "point": float(d["point"]),
                    "lo": float("nan"),
                    "hi": float("nan"),
                    "n_conditions": int(d.get("n_conditions", 0)),
                    "n_boot_valid": 0,
                }
        return out

    n = int(draw["n"])
    return {
        "arm": "arm_a_lmsys" if n_behavior == 0 else "arm_b_behavior",
        "n_lmsys": int(n_lmsys),
        "n_behavior": int(n_behavior),
        "n_lmsys_used": n if n_behavior == 0 else 0,
        "n_behavior_used": n if n_lmsys == 0 else 0,
        "subsample": int(draw["draw"]),
        "upsample_1to1": bool(upsample_1to1),
        "h_ridge_dot_r": _read("h_dot"),
        "g_ridge_r": _read("g"),
        "g_labels_dropped_nan": int(draw.get("g_n_dropped", 0)),
        "has_labels": bool(draw.get("g_n_valid", 0)),
        "source": "pod_edges",
        "edges_behavior_agg": edges_behavior_agg,
        "edges_extra": {
            k: draw.get(k)
            for k in ("h_cos", "h_gcv_lambda", "g_gcv_lambda", "h_recon_r2", "g_recon_r2")
        },
    }


def compose_edges_into_scaling(
    scaling: dict, edges_doc: dict, *, edges_path=None, edges_behavior_agg: str | None = None
) -> dict:
    """Merge pod-computed edge cells into a scaling_grid artifact IN PLACE (v81).

    ``edges_doc`` is the ``issue779_edges`` format
    (eval_results/issue_779/batch2_edges.json):
    ``edges[trait]["L<layer>"] = {modes, lmsys_axis{N: {draws}}, behavior_axis{...}}``.

    Mapping per (trait, mode) entry at frozen layer L:
      - ``lmsys_axis[N]`` draws -> (N, 0) Arm-A edge cells in ALL THREE variant
        slots (an nB==0 cell contains no behavior rows, so it is
        aggregation-invariant AND 1to1-upsampling-invariant).
      - ``behavior_axis[N]`` draws -> (0, N) Arm-B edge cells in EXACTLY the
        variant slot(s) whose declared ``behavior_agg`` MATCHES the aggregation
        the edges were computed under. The operative batch2_edges.json pod
        edges are ``mean_10rollout`` (10-rollout-mean v(x) + per-context mean
        judge labels — issue779_edges.corpus_layer / run_edges), i.e. the
        interior's SECONDARY recipe, so they fill ``secondary_10rollout``'s
        natural sub-block ONLY; the headline (``headline_1rollout``) slots'
        (0, N) coordinates stay HONESTLY missing (recorded by the completeness
        stamp). A hypothetical headline-agg edges artifact would route to
        natural + upsample_1to1 instead by the same rule.

    Aggregation resolution (r11 permanent guard): the edges' behavior-side
    aggregation is read from ``edges_doc["metadata"]["behavior_agg"]``
    (self-described by the r11+ producer). For a legacy artifact lacking the
    field (the existing batch2_edges.json), the caller MUST pass an explicit
    ``edges_behavior_agg`` attestation — composing without either REFUSES loud
    rather than guessing; a declared field that CONTRADICTS a passed
    attestation refuses too. Behavior-axis edges whose aggregation matches NO
    variant slot declared by an entry refuse loud as well.

    Fail-loud validation: missing ``edges[trait]["L<frozen_layer>"]`` leaf, or
    ``mode`` not read at that layer -> ValueError. An edge coordinate outside
    the block's declared axis (the pod's behavior-axis 2400 vs the planned
    2000) is composed as an EXTRA coordinate with a WARNING (recorded in the
    completeness record), never silently remapped. A composed cell colliding
    with an EXISTING cell refuses loud on conflicting values (idempotent
    recompose of identical cells is allowed). Returns a summary dict; caller
    stamps completeness + writes."""
    edges = edges_doc.get("edges")
    if not isinstance(edges, dict):
        raise ValueError("edges doc has no 'edges' tree (expected issue779_edges format)")
    meta = edges_doc.get("metadata", {})
    declared_agg = meta.get("behavior_agg")
    if declared_agg is None and edges_behavior_agg is None:
        raise ValueError(
            "edges doc metadata declares no 'behavior_agg' and no attestation was passed — "
            "refusing to guess which behavior-side aggregation the edges were computed "
            "under (verify the producer's recipe, then pass "
            "edges_behavior_agg=... / --edges-behavior-agg)"
        )
    if (
        declared_agg is not None
        and edges_behavior_agg is not None
        and (declared_agg != edges_behavior_agg)
    ):
        raise ValueError(
            f"edges doc declares behavior_agg={declared_agg!r} but the caller attested "
            f"{edges_behavior_agg!r} — refusing the contradictory composition"
        )
    behavior_agg = declared_agg if declared_agg is not None else edges_behavior_agg
    n_added, n_extra, per_block = 0, 0, {}
    for trait, modes_tree in scaling.get("traits", {}).items():
        for mode, entry in modes_tree.items():
            layer = int(entry["frozen_layer"])
            leaf = (edges.get(trait) or {}).get(f"L{layer}")
            if leaf is None:
                raise ValueError(
                    f"edges doc has no leaf for {trait}/L{layer} "
                    f"(available: {sorted((edges.get(trait) or {}).keys())})"
                )
            if mode not in leaf.get("modes", []):
                raise ValueError(
                    f"edges leaf {trait}/L{layer} was not read in mode {mode!r} "
                    f"(modes: {leaf.get('modes')})"
                )
            # behavior-axis routing: ONLY variant slots whose declared
            # aggregation equals the edges' aggregation may receive (0, N)
            # cells (the r11 remap — the pre-r11 inverse put mean-10 pod edges
            # into the headline blocks).
            entry_variant_aggs = {
                "natural": entry.get("behavior_agg"),
                "upsample_1to1": entry.get("behavior_agg"),
                "secondary_10rollout": (entry.get("secondary_10rollout") or {}).get("behavior_agg"),
            }
            behavior_variants = {v for v, a in entry_variant_aggs.items() if a == behavior_agg}
            if leaf.get("behavior_axis") and not behavior_variants:
                raise ValueError(
                    f"{trait}/{mode}: behavior-axis edges computed under behavior_agg="
                    f"{behavior_agg!r} match no variant block of this entry "
                    f"(entry declares {entry_variant_aggs}) — refusing the "
                    "cross-aggregation composition"
                )
            for axis_name, coord_of in (
                ("lmsys_axis", lambda n: (n, 0)),
                ("behavior_axis", lambda n: (0, n)),
            ):
                for n_str, node in (leaf.get(axis_name) or {}).items():
                    nL, nB = coord_of(int(n_str))
                    for variant in GRID_VARIANT_SLOTS:
                        if axis_name == "behavior_axis" and variant not in behavior_variants:
                            continue  # aggregation mismatch: this slot's (0,N) stays missing
                        block = _variant_block(entry, variant)
                        if block is None:
                            continue  # variant never computed; completeness records it
                        axis_vals = [int(v) for v in block["n_lmsys_grid"]] + [
                            int(v) for v in block["n_behavior_grid"]
                        ]
                        declared = (nL in axis_vals) if nB == 0 else (nB in axis_vals)
                        if not declared:
                            n_extra += 1
                            logger.warning(
                                "[compose-edges] %s/%s/%s: edge coord (%d,%d) outside the "
                                "declared axes — composed as EXTRA",
                                trait,
                                mode,
                                variant,
                                nL,
                                nB,
                            )
                        new_cells = [
                            _edge_draw_to_cell(
                                d,
                                n_lmsys=nL,
                                n_behavior=nB,
                                upsample_1to1=(variant == "upsample_1to1"),
                                edges_behavior_agg=behavior_agg,
                            )
                            for d in node.get("draws", [])
                        ]
                        stub = dict(block)
                        stub["cells"] = new_cells
                        merged = merge_grid_block(block, stub, on_collision="refuse")
                        block.clear()
                        block.update(merged)
                        n_added += len(new_cells)
                        bk = f"{trait}/{mode}/{variant}"
                        per_block[bk] = per_block.get(bk, 0) + len(new_cells)
    provenance = {
        "path": str(edges_path) if edges_path is not None else None,
        "git_commit": meta.get("git_commit"),
        "timestamp_utc": meta.get("timestamp_utc"),
        "k_draws": meta.get("k_draws"),
        "behavior_agg": behavior_agg,
        "behavior_agg_source": "metadata" if declared_agg is not None else "cli_attestation",
        "n_cells_added": n_added,
        "n_extra_coord_composes": n_extra,
    }
    scaling.setdefault("edges_composed", []).append(provenance)
    logger.info("[compose-edges] %s", provenance)
    return {
        "n_cells_added": n_added,
        "n_extra_coord_composes": n_extra,
        "per_block": per_block,
        "behavior_agg": behavior_agg,
    }
