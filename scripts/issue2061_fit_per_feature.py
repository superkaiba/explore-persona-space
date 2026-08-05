"""P2 per-feature ridge fit for task #2061.

Reads:
- P1 output: `data/issue_2061/sae_encoded/<stage>_<render>_<corpus>_answer_L29.pt`
  — the fixed-width TopK SPARSE payload (`issue2061_turnstore.ENCODED_TARGET_FORMAT`:
  (n, k=32) idx + val + row-aligned conv_ids) from `scripts/issue2061_sae_encode.py`.
  Row alignment with X is KEYED on conv_ids (fail-loud mismatch assert), not
  shard-order faith (review round-2 M1).
- #1336's banked arm inputs (prefix + context slot states), loaded from ALL
  `*_shardNNN.pt` files of each locally-staged turnstore in shard-index order.
  Payload schema + realized pooling convention: `scripts/issue2061_turnstore.py`.

Fits, per (stage, render, corpus, arm) cell:
- K=5 GROUP-level folds (conversation-id groups), fold seed 0 — #1336's exact
  fold convention via `issue2061_turnstore.group_fold_ids` (mirrors
  `issue825_fit_cells._cv_folds`; plan §10, review M5,
  `.claude/rules/ood-generalization-folds.md`).
- STREAMED shared-factorization ridge (`_fold_fit_streamed`): ONE primal-space
  eigendecomposition of A = Xn^T Xn (d_in x d_in) per fold, the d_sae=262,144
  target applied in feature CHUNKS built dense-on-the-fly from the sparse
  payload. Numerically the SAME estimator as the #779 helper
  `ridge_fit_predict_fast_layer_batched` (same standardize-X / center-Y /
  full-column GCV over the #823/#779 grid `np.logspace(-2, 4, 13)` WITH the
  #1887 `gcv_dof_cap=0.9` mask / un-centered predictions; ridge primal==dual
  identity at lambda>0) — pinned by
  tests/test_issue2061_stats.py::test_streamed_fit_matches_layer_batched_helper.
  The restructure is load-bearing (review round-2 M1): the dual-space helper
  at the lmsys23k shape (n_tr~18.4k, d_out=262,144) materializes
  (n_tr, d_sae) float64 intermediates (~38.6 GB EACH; realized cell peak
  130-190 GB vs the 128 GB cpu-bigmem pod) AND its V^T @ Yc GEMM costs
  2*n_tr^2*d_sae ~= 1.8e17 FLOPs/fold (days) — the primal shared
  factorization is 2*n_tr*d_in*d_sae ~= 4.0e13 FLOPs/fold (minutes),
  matching the plan §9 P2 sizing.
- Per-cell PEAK RAM at the worst (lmsys23k) shape — stated arithmetic
  (review M1): Y_pred_pool float32 (n=23k x 262,144) = 24.1 GB + the
  eigenbasis coefficient matrix C float64 (4096 x 262,144) = 8.6 GB +
  X-side float64 (Xtr/Xn/P ~ 1.8 GB) + per-chunk temporaries (~1.8 GB)
  + kNN chunk buffers (~1 GB) ~= 37 GB, UNDER the plan §9 P2 budget of
  ~50 GB on the 128 GB cpu-bigmem pod. The sparse Y payload itself is ~6 MB.
- Slow-vs-fast numeric-parity gate (plan §Design): once per process, >=3 fold
  slices of the first fitted cell at production (n_train, d_in) shape — the
  DISPATCHED `_fold_fit_streamed` path vs the canonical `ridge_fit_predict`
  SVD reference on a seeded column subsample; max rel diff <= 1e-4 (#1332
  bar). The gate exercises the exact code path `fit_cell` dispatches
  (hollow-verification-gate rule, `.claude/rules/code-style.md`).
- Per-feature R²_j = 1 − ||f_j − ĝ_j||² / ||f_j − mean(f_j)||² on held-out
  folds, pooled with fold-local test means.
- kNN retrieval (euclidean + cosine) with `k = ceil(n_pool / 20)` (plan §13;
  chance = k / n_pool) via `_knn_retrieval_sparse` — the sparse-pool twin of
  `analysis/mapping_baselines.knn_retrieval` (rank math + tie tolerance
  mirrored exactly; pinned by
  tests/test_issue2061_stats.py::test_knn_retrieval_sparse_matches_dense).
  The dense helper at (n=23k, d_sae=262k) needs a 48 GB float64 pool and a
  2*n^2*d_sae ~= 2.8e17-FLOP GEMM; the sparse pool makes it 2*n^2*k ~= 3.4e10.

Emits `eval_results/issue_2061/per_feature_r2/<stage>_<render>_<corpus>_<arm>_L29.jsonl`
— one JSON object per feature with keys:
  {feature_id, R2, n_train_folds, n_test_total, best_lambda_folds,
   effective_dof_folds, lambda_selector, knn_acc_1_euclid, knn_acc_k_euclid,
   knn_k_ret, knn_acc_1_cosine, knn_acc_k_cosine, chance_1, chance_k}

Usage:
  uv run python scripts/issue2061_fit_per_feature.py \\
      --stage base --render chat --corpus lmsys23k --arm context
  uv run python scripts/issue2061_fit_per_feature.py --all-cells
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779.fit_h import (  # noqa: E402
    ridge_fit_predict,
)

# Sibling-script import (bare module name via the script-dir sys.path insert —
# the issue1336_extract_turnstore.py pattern; works in script mode AND under
# the tests' `sys.path.insert(scripts)` import).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import issue2061_turnstore as ts  # noqa: E402

LAYER = 29
D_IN = 4096
K_FOLDS = 5
FOLD_SEED = 0
LAMBDA_GRID = np.logspace(-2, 4, 13)
DOF_CAP_FRACTION = 0.9  # plan §7 under-determined-cell mitigation
FEATURE_CHUNK = 8192  # streamed-fit feature-chunk width (matches the P3 engine)


def _load_arm_inputs(
    turnstore_dir: Path, arm: str, layer: int = LAYER
) -> tuple[np.ndarray, list[str]]:
    """((n, d_in) float32 arm inputs, conv_ids) from ALL shards of one turnstore dir.

    `arm` selects the banked #1336 slot state — "prefix" -> the prefix-header
    slot, "context" -> the a1-assistant-header slot (end of the context).
    Realized convention + fail-loud schema assert live in
    `issue2061_turnstore` (see its docstring; plan §12(4)). Shards are
    enumerated in shard-index order, matching the encode script's row order;
    the conv_ids additionally KEY the X/Y alignment against the sparse
    payload's own conv_ids (review M1) and feed the #1336 GROUP-level fold
    construction (plan §10, review M5).
    """
    shard_paths = ts.enumerate_shards(turnstore_dir)
    x, conv_ids = ts.load_state_from_shards(shard_paths, state=arm, layer=layer)
    return x.numpy(), conv_ids


def _make_folds(conv_ids: list[str], k: int = K_FOLDS, seed: int = FOLD_SEED) -> list[np.ndarray]:
    """Per-fold TEST index arrays from #1336's GROUP-level fold convention.

    Delegates to `issue2061_turnstore.group_fold_ids` (mirrors
    `issue825_fit_cells._cv_folds`: seeded permutation of UNIQUE conversation
    ids, `perm % k` per id — all rows of a conversation share a fold). Fold
    sizes vary with group membership; every fold is non-empty (fail-loud in
    the helper).
    """
    fold_of_row = ts.group_fold_ids(conv_ids, n_folds=k, seed=seed)
    return [np.where(fold_of_row == f)[0].astype(np.int64) for f in range(k)]


def _fold_fit_streamed(
    X: np.ndarray,
    y_idx: np.ndarray,
    y_val: np.ndarray,
    d_sae: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    lambdas: np.ndarray = LAMBDA_GRID,
    gcv_dof_cap: float | None = DOF_CAP_FRACTION,
    feature_chunk: int = FEATURE_CHUNK,
    device: str = "cpu",
    ss_res: np.ndarray,
    ss_tot: np.ndarray,
    out_pred: np.ndarray,
) -> dict:
    """One fold's ridge fit, streamed over feature chunks of a sparse target.

    Numerically the `ridge_fit_predict_fast_layer_batched` estimator (same
    standardization, centered-Y full-column GCV with the #1887 dof-cap mask,
    un-centered predictions) computed in the PRIMAL eigenbasis of
    A = Xn^T Xn (d_in x d_in) so the d_sae-wide target only ever
    materializes dense one (rows, feature_chunk) block at a time — never an
    (n_tr, d_sae) float64 intermediate (review M1; parity pinned by
    tests/test_issue2061_stats.py). Accumulates per-feature SS_res/SS_tot
    (fold-local test means) into `ss_res`/`ss_tot` and writes held-out
    predictions into `out_pred[test_idx]` (cast to out_pred's dtype).
    Returns {"best_lambda", "dof"} for the #1887 diagnostics.
    """
    dev = torch.device(device)
    Xtr = torch.as_tensor(X[train_idx], dtype=torch.float64).to(dev)
    Xev = torch.as_tensor(X[test_idx], dtype=torch.float64).to(dev)
    ntr = int(Xtr.shape[0])
    mu = Xtr.mean(dim=0, keepdim=True)
    sd = Xtr.std(dim=0, unbiased=False, keepdim=True) + 1e-9  # helper parity
    Xn = (Xtr - mu) / sd
    Xev_n = (Xev - mu) / sd

    A = Xn.T @ Xn  # (d_in, d_in) — ONE shared factorization per fold
    w, V = torch.linalg.eigh(A)
    w = torch.clamp(w, min=0.0)
    P = Xn @ V  # (ntr, d_in); column norms sqrt(w)
    Pev = Xev_n @ V  # (nev, d_in)

    ymu = ts.sparse_column_means(y_idx, y_val, train_idx, d_sae)  # (d_sae,) f64
    ybar_ev = ts.sparse_column_means(y_idx, y_val, test_idx, d_sae)

    # Pass 1: eigenbasis coefficients C = P^T Yc (built per chunk, kept whole:
    # (d_in, d_sae) f64 = 8.6 GB at production shape) + the FULL-column GCV
    # criterion terms. sq_k = ||u_k^T Yc||^2 = ||row_k(C)||^2 / w_k (P's
    # columns have norm sqrt(w)) — the exact Gram-basis sqVtY identity of
    # ridge_fit_predict_fast_layer_batched: zero-eigenvalue components carry
    # zero column-space energy and are guarded to exactly 0.
    C = torch.empty((int(A.shape[0]), d_sae), dtype=torch.float64, device=dev)
    sq_raw = torch.zeros(int(A.shape[0]), dtype=torch.float64, device=dev)
    tot = 0.0
    for c0 in range(0, d_sae, feature_chunk):
        c1 = min(c0 + feature_chunk, d_sae)
        ytr_chunk = ts.sparse_dense_chunk(y_idx, y_val, train_idx, c0, c1)  # (ntr, csz) f64
        yc = torch.as_tensor(ytr_chunk, device=dev)
        yc -= torch.as_tensor(ymu[c0:c1], device=dev)[None, :]
        cc = P.T @ yc  # (d_in, csz)
        C[:, c0:c1] = cc
        sq_raw += (cc**2).sum(dim=1)
        tot += float((yc**2).sum())
    wmax = torch.clamp(w.max(), min=1e-300)
    sq = torch.where(
        w > 1e-12 * wmax, sq_raw / torch.clamp(w, min=1e-300), torch.zeros_like(sq_raw)
    )

    # GCV over the grid (full-column criterion; #1887 dof-cap mask — same
    # predicate + fail-loud as the helper).
    gcv_vals = np.empty(len(lambdas), dtype=np.float64)
    dofs = np.empty(len(lambdas), dtype=np.float64)
    for li, lam in enumerate(lambdas):
        filt = w / (w + float(lam))
        rss = tot - float(((2 * filt - filt**2) * sq).sum())
        dof = float(filt.sum())
        dofs[li] = dof
        denom = (ntr - dof) ** 2
        val = rss / denom if denom > 1e-12 else float("inf")
        if gcv_dof_cap is not None and dof > gcv_dof_cap * ntr:
            val = float("inf")
        gcv_vals[li] = val
    if gcv_dof_cap is not None and not np.isfinite(gcv_vals).any():
        raise RuntimeError(
            f"gcv_dof_cap={gcv_dof_cap}: every lambda in the grid "
            f"(n={len(lambdas)}, [{float(lambdas[0]):.3g}, {float(lambdas[-1]):.3g}]) "
            f"exceeds dof cap {gcv_dof_cap} * n_tr={ntr} (#1887). Widen the grid."
        )
    best_li = int(np.argmin(gcv_vals))
    best_lam = float(lambdas[best_li])
    best_dof = float(dofs[best_li])

    # Pass 2: predictions per chunk — yhat = Pev @ (C / (w + lam)) + ymu —
    # plus pooled per-feature SS accumulation with fold-local test means.
    fprime = (1.0 / (w + best_lam))[:, None]  # (d_in, 1)
    for c0 in range(0, d_sae, feature_chunk):
        c1 = min(c0 + feature_chunk, d_sae)
        ymu_chunk = torch.as_tensor(ymu[c0:c1], device=dev)[None, :]
        yhat = (Pev @ (fprime * C[:, c0:c1]) + ymu_chunk).cpu().numpy()  # (nev, csz)
        yev = ts.sparse_dense_chunk(y_idx, y_val, test_idx, c0, c1)  # (nev, csz)
        ss_res[c0:c1] += ((yev - yhat) ** 2).sum(axis=0)
        ss_tot[c0:c1] += ((yev - ybar_ev[c0:c1][None, :]) ** 2).sum(axis=0)
        out_pred[test_idx, c0:c1] = yhat.astype(out_pred.dtype)
    return {"best_lambda": best_lam, "dof": best_dof}


def _knn_retrieval_sparse(
    pred: np.ndarray,
    y_idx: np.ndarray,
    y_val: np.ndarray,
    *,
    ks: tuple[int, ...],
    metric: str = "euclidean",
    row_chunk: int = 2048,
) -> dict:
    """Sparse-pool twin of `analysis.mapping_baselines.knn_retrieval`.

    Pool == true targets, `true_pool_idx == arange(n)` (the P2 read). The
    inner products p·y ride the fixed-width sparse pool (kmax column gathers,
    2*n^2*kmax ops) instead of a dense (n, d_sae) GEMM — at the lmsys23k
    shape the dense pool alone is 48 GB f64 and the GEMM ~2.8e17 FLOPs
    (review M1). Distance + MID-RANK tie math mirror the dense helper
    exactly (squared euclidean / 1−cosine, tol = 1e-9 * max(|d_true|,
    1e-12)); pinned equal on dense-reconstructable pools by
    tests/test_issue2061_stats.py::test_knn_retrieval_sparse_matches_dense.
    """
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"unknown metric {metric!r}")
    n = pred.shape[0]
    kmax = y_idx.shape[1]
    val64 = y_val.astype(np.float64)
    yn2 = (val64**2).sum(axis=1)  # (n,) pool row norms² (pads add 0)
    ranks = np.empty(n, dtype=np.float64)
    for i0 in range(0, n, row_chunk):
        i1 = min(i0 + row_chunk, n)
        p = pred[i0:i1].astype(np.float64)  # (ic, d_sae)
        pyt = np.zeros((i1 - i0, n), dtype=np.float64)
        for kk in range(kmax):
            pyt += p[:, y_idx[:, kk]] * val64[:, kk][None, :]
        if metric == "euclidean":
            pn2 = (p**2).sum(axis=1)
            dist = pn2[:, None] + yn2[None, :] - 2.0 * pyt
        else:  # cosine — the dense helper's 1 − (p·q)/((‖p‖+1e-12)(‖q‖+1e-12))
            pnorm = np.sqrt((p**2).sum(axis=1)) + 1e-12
            ynorm = np.sqrt(yn2) + 1e-12
            dist = 1.0 - pyt / (pnorm[:, None] * ynorm[None, :])
        d_true = dist[np.arange(i1 - i0), np.arange(i0, i1)]
        tol = 1e-9 * np.maximum(np.abs(d_true)[:, None], 1e-12)
        closer = (dist < d_true[:, None] - tol).sum(axis=1)
        tied = (np.abs(dist - d_true[:, None]) <= tol).sum(axis=1) - 1
        ranks[i0:i1] = 1.0 + closer + 0.5 * tied
    return {
        "metric": metric,
        "n": int(n),
        "n_pool": int(n),
        "acc_at_k": {int(k): float((ranks <= k).mean()) for k in ks},
        "chance_at_k": {int(k): float(k / n) for k in ks},
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
    }


_PARITY_GATE_STATE = {"done": False}


def run_parity_gate(
    X: np.ndarray,
    y_idx: np.ndarray,
    y_val: np.ndarray,
    d_sae: int,
    folds: list[np.ndarray],
    *,
    n_slices: int = 3,
    n_cols: int = 512,
    col_seed: int = 0,
    tol: float = 1e-4,
    device: str = "cpu",
) -> float:
    """Streamed-vs-canonical numeric-parity gate on the DISPATCHED fit path.

    Compares `_fold_fit_streamed` — the exact function `fit_cell` dispatches
    (hollow-verification-gate rule, `.claude/rules/code-style.md` #779) —
    against the canonical `ridge_fit_predict` SVD reference on `n_slices`
    fold slices at the CELL's production (n_train, d_in) shape, over a seeded
    column subsample of the target (per-column regressions are independent,
    so column subsetting does not change the per-slice fit machinery under
    test; both sides GCV-select on the SAME subsampled columns). Runs with
    `gcv_dof_cap=None` on BOTH sides (the slow reference has no cap; the cap
    is pinned separately by tests/test_issue2061_stats.py). Raises
    RuntimeError above `tol` (#1332 bar: max rel diff <= 1e-4). Returns the
    realized max rel diff.
    """
    n = X.shape[0]
    n_slices = min(n_slices, len(folds))
    rng = np.random.default_rng(col_seed)
    cols = np.sort(rng.choice(d_sae, size=min(n_cols, d_sae), replace=False))
    ncols = len(cols)
    # Dense column subsample from the sparse payload (np.add.at: pad-safe).
    lookup = np.full(d_sae, -1, dtype=np.int64)
    lookup[cols] = np.arange(ncols)
    pos = lookup[y_idx]
    ysub = np.zeros((n, ncols), dtype=np.float64)
    rr, kk = np.nonzero(pos >= 0)
    np.add.at(ysub, (rr, pos[rr, kk]), y_val[rr, kk].astype(np.float64))
    sub_idx, sub_val = ts.to_fixed_width_sparse(torch.as_tensor(ysub, dtype=torch.float32))

    worst = 0.0
    for fi in range(n_slices):
        test_idx = folds[fi]
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fi])
        out_pred = np.zeros((n, ncols), dtype=np.float64)
        _fold_fit_streamed(
            X,
            sub_idx,
            sub_val,
            ncols,
            train_idx,
            test_idx,
            lambdas=LAMBDA_GRID,
            gcv_dof_cap=None,
            device=device,
            ss_res=np.zeros(ncols),
            ss_tot=np.zeros(ncols),
            out_pred=out_pred,
        )
        fast = out_pred[test_idx]
        slow = ridge_fit_predict(
            X[train_idx].astype(np.float64),
            ysub[train_idx],
            X[test_idx].astype(np.float64),
            lambdas=LAMBDA_GRID,
        )
        rel = float(np.max(np.abs(fast - slow)) / (np.max(np.abs(slow)) + 1e-12))
        worst = max(worst, rel)
        print(
            f"  [parity-gate] slice {fi}: n_train={len(train_idx)} d_in={X.shape[1]} "
            f"n_cols={ncols} max_rel_diff={rel:.3g}",
            flush=True,
        )
    if worst > tol:
        raise RuntimeError(
            f"streamed-vs-canonical ridge parity gate FAILED: max rel diff {worst:.3g} > "
            f"tol {tol:.1g} over {n_slices} slices (_fold_fit_streamed vs "
            "fit_h.ridge_fit_predict) — fall back to the canonical solver."
        )
    print(f"  [parity-gate] PASS: worst max_rel_diff={worst:.3g} <= tol={tol:.1g}", flush=True)
    return worst


def fit_cell(
    turnstore_dir: Path,
    encoded_shard: Path,
    arm: str,
    output_path: Path,
    layer: int = LAYER,
    device: str = "cpu",
    skip_parity_gate: bool = False,
    feature_chunk: int = FEATURE_CHUNK,
) -> None:
    """Fit ridge for one (stage, render, corpus, arm) cell + write JSONL.

    Consumes the P1 SPARSE payload directly (never a dense (n, d_sae)
    matrix); the module docstring carries the peak-RAM arithmetic (~37 GB at
    the worst lmsys23k shape vs the plan §9 P2 ~50 GB budget). X/Y row
    alignment is asserted on conv_ids (review M1).
    """
    print(f"[fit] turnstore={turnstore_dir.name} arm={arm} encoded={encoded_shard.name}")
    X, conv_ids = _load_arm_inputs(turnstore_dir, arm, layer=layer)  # (n, d_in)
    payload = ts.load_encoded_target(encoded_shard)
    if payload["conv_ids"] != conv_ids:
        n_common = len(set(payload["conv_ids"]) & set(conv_ids))
        raise ValueError(
            f"X/Y row alignment mismatch for {encoded_shard.name}: the encoded payload's "
            f"conv_ids differ from the turnstore's ({n_common} ids in common; "
            f"n_payload={len(payload['conv_ids'])}, n_turnstore={len(conv_ids)}). "
            "The encode and fit MUST consume the same turnstore snapshot (review M1)."
        )
    y_idx = payload["idx"].numpy().astype(np.int64)  # (n, k)
    y_val = payload["val"].numpy()  # (n, k) float32
    d_sae = int(payload["d_sae"])
    n, d_in = X.shape
    assert y_idx.shape[0] == n, (y_idx.shape, n)
    print(f"  n={n} d_in={d_in} d_sae={d_sae} k={payload['k']} (sparse payload)")

    folds = _make_folds(conv_ids, k=K_FOLDS, seed=FOLD_SEED)

    # Once-per-process streamed-vs-canonical parity gate on the FIRST fitted
    # cell (>=3 slices at this cell's production shape; plan §Design mandate).
    if not skip_parity_gate and not _PARITY_GATE_STATE["done"]:
        run_parity_gate(X, y_idx, y_val, d_sae, folds, device=device)
        _PARITY_GATE_STATE["done"] = True

    # Pooled per-feature R² with fold-local test means: track SS_res + SS_tot
    # per feature across folds. Also track per-fit best_lambda / dof for the
    # dof-cap diagnostic.
    ss_res = np.zeros(d_sae, dtype=np.float64)
    ss_tot = np.zeros(d_sae, dtype=np.float64)
    per_fold_lambda: list[float] = []
    per_fold_dof: list[float] = []
    per_fold_ntrain: list[int] = []

    # Pooled held-out predictions for the kNN retrieval read — float32 (the
    # dominant per-cell allocation: 24.1 GB at n=23k x d_sae=262,144).
    Y_pred_pool = np.zeros((n, d_sae), dtype=np.float32)

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fi])
        n_train = int(train_idx.shape[0])
        n_test = int(test_idx.shape[0])
        per_fold_ntrain.append(n_train)
        t0 = time.time()
        info = _fold_fit_streamed(
            X,
            y_idx,
            y_val,
            d_sae,
            train_idx,
            test_idx,
            lambdas=LAMBDA_GRID,
            gcv_dof_cap=DOF_CAP_FRACTION,  # #1887 mitigation, plan §11 (review M4)
            feature_chunk=feature_chunk,
            device=device,
            ss_res=ss_res,
            ss_tot=ss_tot,
            out_pred=Y_pred_pool,
        )
        elapsed = time.time() - t0
        best_lam = float(info["best_lambda"])
        dof = float(info["dof"])
        per_fold_lambda.append(best_lam)
        per_fold_dof.append(dof)

        # The fit masks lambdas whose dof exceeds the cap (and fail-louds when
        # ALL are masked), so the selected dof satisfies the cap by
        # construction — assert it stays that way (guard against drift).
        assert dof <= DOF_CAP_FRACTION * n_train * (1.0 + 1e-9), (
            f"fold {fi}: selected dof={dof:.1f} violates gcv_dof_cap="
            f"{DOF_CAP_FRACTION} * n_train={n_train} — dof-cap drift?"
        )
        if n_train < d_in:
            print(
                f"  [dof-cap] fold {fi}: n_train={n_train} < d_in={d_in} — "
                f"cap {DOF_CAP_FRACTION} active (lambda={best_lam:.3g}, dof={dof:.1f})",
                flush=True,
            )
        print(
            f"  fold {fi}: n_train={n_train} n_test={n_test} λ={best_lam:.3g} dof={dof:.1f} ({elapsed:.1f}s)"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - np.where(ss_tot > 0, ss_res / ss_tot, np.nan)

    # kNN retrieval on the pooled OOF predictions vs the fixed sparse target
    # pool. k = ceil(n_pool / 20), chance = k / n_pool (plan §13). Dicts are
    # KEYED BY K (review C2 — positional indexing crashed KeyError: 0).
    k_ret = max(1, math.ceil(n / 20))
    knn_e = _knn_retrieval_sparse(Y_pred_pool, y_idx, y_val, ks=(1, k_ret), metric="euclidean")
    knn_c = _knn_retrieval_sparse(Y_pred_pool, y_idx, y_val, ks=(1, k_ret), metric="cosine")
    lambda_selector = f"gcv-dof-cap-{DOF_CAP_FRACTION}"  # #1887 diagnostics (M4)
    # Plan §Design "Baselines per fitted map": the identity+learned-bias
    # baseline requires input/output spaces of equal dimension — its
    # inapplicability is STATED in the fit output, never silently skipped
    # (review m1; CLAUDE.md identity+bias Critical Rule).
    identity_bias = f"N/A: dim mismatch ({d_in} vs {d_sae})"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for j in range(d_sae):
            f.write(
                json.dumps(
                    {
                        "feature_id": j,
                        "R2": None if np.isnan(r2[j]) else float(r2[j]),
                        "n_train_folds": per_fold_ntrain,
                        "n_test_total": int(n),
                        "best_lambda_folds": per_fold_lambda,
                        "effective_dof_folds": per_fold_dof,
                        "lambda_selector": lambda_selector,
                        "identity_bias": identity_bias,
                        "knn_acc_1_euclid": float(knn_e["acc_at_k"][1]),
                        "knn_acc_k_euclid": float(knn_e["acc_at_k"][k_ret]),
                        "knn_k_ret": int(k_ret),
                        "knn_acc_1_cosine": float(knn_c["acc_at_k"][1]),
                        "knn_acc_k_cosine": float(knn_c["acc_at_k"][k_ret]),
                        "chance_1": float(knn_e["chance_at_k"][1]),
                        "chance_k": float(knn_e["chance_at_k"][k_ret]),
                    }
                )
                + "\n"
            )
    print(f"[done] wrote {d_sae} rows to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--render", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--arm", choices=["prefix", "context"], default=None)
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument(
        "--context-shard-dir",
        type=Path,
        default=None,
        help="Directory holding #1336 context shards (staged locally). Required "
        "unless --stage-context-from-hub streams them per cell.",
    )
    parser.add_argument("--encoded-dir", type=Path, default=Path("data/issue_2061/sae_encoded"))
    parser.add_argument(
        "--stage-encoded-from-hub",
        action="store_true",
        help="Stage P1's encoded targets from the HF data repo (plan §9 "
        "off_pod_phases P2 reads; TopK-sparse, ~300 MB total) and read them "
        "from the staged mirror instead of --encoded-dir.",
    )
    parser.add_argument(
        "--stage-context-from-hub",
        action="store_true",
        help="Per-shard STREAM-FETCH-DELETE of the #1336 turnstores (registered "
        "v6): fetch one (stage, render, corpus) shard set, fit the arm cells "
        "that consume it, delete before the next fetch — resident staging "
        "stays ≤ ~1 turnstore (≤ ~25 GB), never the full ~store.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("data/issue_2061/hf_dl"),
        help="Root for hub staging (mirror for encoded targets; turnstores "
        "land under <staging-dir>/turnstores/).",
    )
    parser.add_argument("--data-revision", type=str, default=None)
    parser.add_argument(
        "--keep-staged",
        action="store_true",
        help="Keep staged turnstores instead of the stream-fetch-DELETE reap "
        "(smoke chains reuse them across phases; fellows node-local scratch).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="After the fits, upload --output-dir to the HF data repo "
        "(analysis_tensors/per_feature_r2/) — plan §9: P3 reads it as the "
        "true-delta verdict input, so it MUST land before this pod terminates.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval_results/issue_2061/per_feature_r2")
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' or 'cuda' (cpu is fine for this regime: one eigh(d_in) per fold)",
    )
    parser.add_argument("--feature-chunk", type=int, default=FEATURE_CHUNK)
    parser.add_argument(
        "--skip-parity-gate",
        action="store_true",
        help="skip the once-per-process streamed-vs-canonical parity gate (debug only)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import issue2061_hub_io as hio  # sibling import (script-dir sys.path insert)

    encoded_dir = args.encoded_dir
    if args.stage_encoded_from_hub:
        encoded_dir = hio.stage_dir("sae-encoded", args.staging_dir)
        print(f"[stage] encoded targets staged from hub -> {encoded_dir}")
    if args.stage_context_from_hub:
        context_root = args.staging_dir / "turnstores"
        print(f"[stage] turnstores stream-fetch-delete under {context_root}")
    else:
        if args.context_shard_dir is None:
            print("[error] Pass --context-shard-dir OR --stage-context-from-hub")
            return 1
        context_root = args.context_shard_dir

    # Y targets are the ANSWER-state encodes only (plan §Design target Y).
    # `--max-rows` debug encodes carry a `_rows{N}` suffix and never match
    # this glob (review M3).
    encoded_files = sorted(encoded_dir.glob(f"*_answer_L{LAYER}.pt"))
    if not encoded_files:
        print(f"[error] No '*_answer_L{LAYER}.pt' encoded targets in {encoded_dir}")
        return 1

    # Group by (stage, render, corpus) so the stream-fetch-delete staging
    # fetches ONE turnstore shard set, fits the arm cells that consume it,
    # then deletes it before the next fetch (plan §9 P2 staging shape, v6).
    arms = ["prefix", "context"] if args.arm is None else [args.arm]
    cells: list[tuple[Path, str, str, str]] = []
    for enc_path in encoded_files:
        stage, render, corpus = ts.parse_encoded_stem(enc_path.stem, "answer", LAYER)
        if not args.all_cells:
            if args.stage and stage != args.stage:
                continue
            if args.render and render != args.render:
                continue
            if args.corpus and corpus != args.corpus:
                continue
        cells.append((enc_path, stage, render, corpus))

    if not cells:
        print("[error] No cell matches filters")
        return 1
    print(f"[setup] Fitting {len(cells)} cells x {len(arms)} arm(s)")

    for i, (enc_path, stage, render, corpus) in enumerate(cells, start=1):
        outputs = {
            arm: args.output_dir / f"{stage}_{render}_{corpus}_{arm}_L{LAYER}.jsonl" for arm in arms
        }
        pending = [arm for arm in arms if not outputs[arm].exists()]
        for arm in arms:
            if arm not in pending:
                print(f"[skip] Exists: {outputs[arm]}")
        if not pending:
            continue

        if args.stage_context_from_hub:
            turnstore_dir = hio.stage_turnstore(
                stage, render, corpus, context_root, revision=args.data_revision
            )
        else:
            turnstore_dir = context_root / f"turnstore_{stage}_{render}_{corpus}"
            if not turnstore_dir.is_dir():
                print(f"[skip] Missing turnstore dir: {turnstore_dir}")
                continue
        try:
            for arm in pending:
                print(f"\n=== [{i}/{len(cells)}] {stage}/{render}/{corpus}/{arm} ===")
                fit_cell(
                    turnstore_dir=turnstore_dir,
                    encoded_shard=enc_path,
                    arm=arm,
                    output_path=outputs[arm],
                    device=args.device,
                    skip_parity_gate=args.skip_parity_gate,
                    feature_chunk=args.feature_chunk,
                )
        finally:
            # DELETE before the next fetch — resident staging ≤ ~1 turnstore.
            # Only ever a staging copy (never --context-shard-dir source trees).
            if args.stage_context_from_hub and not args.keep_staged:
                hio.reap_turnstore(context_root, stage, render, corpus)

    if args.upload:
        hio.upload_dir(args.output_dir, "per-feature-r2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
