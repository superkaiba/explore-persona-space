"""Issue #1689 follow-up round ``user-slot-recapture`` — Phase C (fits battery).

CPU-vectorized (device-parametrized) ridge battery over the L19 stores written
by :mod:`scripts.issue1689_user_slot_capture`.

Estimator reuse, NOT re-implementation
--------------------------------------
Every ridge primitive is IMPORTED from the parent module
``scripts.issue825_fit_cells`` — ``_cv_folds`` (conversation-grouped folds),
``_prep_fold`` (train-standardize + Gram eigh + ``Kev @ V``),
``_prep_inner_lambda`` (inner GROUP-fold caches for the lambda scan),
``_ridge_predict_cached`` (fit+predict for one Y off a cache),
``_null_ss_contrib`` (BATCHED shuffled-Y null draws off the SAME cache) and
``heldout_r2_sweep``. That buys three things at once: the published per-cell
numbers are reproducible byte-for-byte (Gate-1), the HARD vectorization
requirements are met by construction, and no fit math is re-derived here.

The HARD requirements, and where each is satisfied:

  * **ONE thin factorization per (cell, slot, fold) of each fit** — inside a fit
    the fold's Gram eigendecomposition is built ONCE and reused across all 13
    lambdas, all 40 null draws and the reduced-basis truncation; no per-lambda
    or per-draw re-solve exists. HONEST SCOPE: the unit of caching is the FIT,
    not the unit — ``heldout_r2_sweep`` owns its own per-(layer, fold) cache, so
    an X slot feeding TWO fit pairs (``prev_turn_end`` feeds both
    ``prevturn_to_u2`` and ``parent_convention_parity``) is factorized once per
    PAIR, and the reduced-basis companion's own fold loop factorizes again —
    hence ``FOLD_SOLVES_PER_FIT_PAIR = 2 * N_FOLDS`` in the projection below.
    That redundancy is deliberate: sharing one cache across fit pairs would mean
    re-implementing the parent sweep and forfeiting the byte-for-byte Gate-1
    reproduction, and it is cheap — the factorization measured 8.5 s of a 175 s
    per-fold total at production shape (n=3800, d=3584, 40 draws), where the
    batched null term is 163 s.
  * **All 13 lambdas as diagonal rescalings of the cached factorization** —
    ``_ridge_predict_cached`` applies ``1/(w + lam)`` to the cached eigenspectrum;
    no per-lambda re-solve exists anywhere in the path.
  * **Inner 3/4-fold CV for lambda selection** — the parent per-cell code runs
    ``lambda_selection="inner-group-cv"`` with ``N_INNER_LAMBDA_FOLDS`` inner
    GROUP folds, so this round runs the same (NOT GCV); the brief's "only if the
    parent per-cell code does" resolves to yes.
  * **40 shuffled nulls with zero refits** — same-Y reads use the parent's
    BATCHED ``_null_ss_contrib`` off the cached factorization; transfer reads use
    a comparison-side conversation-level shuffle of the EVAL targets, which
    requires no fit at all. The two null families are reported under DISTINCT
    keys (``null_shuffle_fit_targets`` / ``null_shuffle_eval_targets``) and are
    never pooled.
  * **identity+learned-bias baseline + kNN retrieval for every fitted map** —
    ``analysis.mapping_baselines.identity_bias_predict`` / ``knn_retrieval``,
    both computed fold-wise on HELD-OUT rows (the parent per-cell script
    evaluated its baseline IN-SAMPLE on all rows and passed no ``pool=``, so its
    published ``knn_pool_size: 200`` field is inert — this round reports a real
    held-out pool and says so).
  * **Reduced PCA basis companion, k = min(1024, floor(n_train/2))** — obtained
    by TRUNCATING the cached Gram eigendecomposition to its top k eigenpairs.
    Dual-space ridge is invariant to an orthogonal reparametrization of the
    feature space, so truncating ``(w, V, KevV)`` to the top k is EXACTLY the
    fit in the top-k PCA basis of the train-standardized features — the
    well-posed companion (n_train >= 2k) at ZERO extra factorizations. The
    identity is regression-tested by ``--verify-truncation-equivalence``, which
    checks that truncation at k = n_train reproduces the untruncated fit.

Battery
-------
  1. per-cell ceilings for every (unit, fit pair) — observed R2, 40-draw null
     band, selected lambdas, identity+bias, kNN, reduced-basis companion;
  2. provenance transfer, 3 ordered pairs x 2 directions, at each framing's
     primary slot, per (model, framing/variant);
  3. naive transfer user <-> assistant-chat against the PARENT's
     ``analysis_tensors/<model>/assistant_chat/L19.pt`` at the pinned revision;
  4. Alex-vs-User-label story pairs (the label effect), matched (model,
     provenance), both directions;
  5. floor-control cells (``first_user_header_end`` -> ``u1_end``) reported
     beside their u2 counterparts — they ride every chat/naturalistic unit, so
     they are ordinary rows of (1) tagged ``is_floor_control``.

Gate-1 (integrity, FAIL LOUD)
-----------------------------
The re-captured parent-convention slot pair on chat/``lmsys``
(``prev_turn_end`` -> ``parent_answer_end``), expanded by ``dup_count`` to the
parent's exact 11400-row array, must reproduce the published per-cell L19
prefix-arm R2 from ``eval_results/issue_1689/percell/heldout_r2_<model>_
user_lmsys_chat.json`` within ``|dR2| <= 0.01``. The parent's row set is
recoverable exactly because all 3800 duplicate conv groups are byte-identical
in (u1, a1) and the lmsys u2 is a single constant, so the deduped rows repeated
``dup_count`` times ARE the parent's rows (fold assignment is by conv id and
every downstream reduction is a row sum, so row ORDER is irrelevant).

Outputs: one JSON per unit + ``summary.json`` under
``eval_results/issue_1689/user_slot_recapture/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    assert (repo_root / "scripts" / "issue825_fit_cells.py").exists(), repo_root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()

N_FOLDS = 5
FIT_SEED = 42
N_NULL_DRAWS = 40
PCA_K_CAP = 1024
GATE1_TOL = 0.01

# The u2-provenance pairs the transfer leg fits, each in BOTH directions. Shared
# by the battery and the wall projection so the projection can never charge for a
# pair the battery does not run (the bridging families carry ONE provenance each,
# so they contribute zero transfer reads).
PROVENANCE_TRANSFER_PAIRS: tuple[tuple[str, str], ...] = (
    ("lmsys", "haiku"),
    ("lmsys", "onpolicy"),
    ("haiku", "onpolicy"),
)

# --- addendum A: length-stratified refits on the PARENT assistant stores -----
# (model x condition) cells whose L19 stores get the median-token split, and the
# arms refit inside each half. Keyed to the parent's own store schema
# (X_prefix / X_context / Y) — see `load_store_parent`.
LENGTH_SPLIT_CONDITIONS: tuple[str, ...] = ("assistant_chat", "assistant_naturalistic")
LENGTH_SPLIT_ARMS: tuple[tuple[str, str, str], ...] = (
    ("X_prefix", "Y", "prefix->answer"),
    ("X_context", "Y", "context->answer"),
)


def _apply_device_choice(device: str) -> None:
    """Honor ``--device cpu`` BEFORE torch is imported.

    The parent's ``_fit_device()`` returns cuda when available; hiding the
    devices at process start is the only way to force CPU without patching the
    parent module. ``auto`` leaves the environment untouched (cuda when present
    — the Gram-space ridge is FLOP-bound, so a GPU is a real win here).
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device not in ("auto", "cuda"):
        raise ValueError(f"unknown device {device!r}")


# ---------------------------------------------------------------------------
# Cached-factorization helpers
# ---------------------------------------------------------------------------


def _truncate_cache(cache: dict, k: int) -> dict:
    """Top-``k`` truncation of a ``_prep_fold`` cache == the fit in the top-k
    PCA basis of the train-standardized features.

    ``torch.linalg.eigh`` returns ASCENDING eigenvalues, so the top-k eigenpairs
    are the LAST k columns. ``KevV`` slices columns with ``V``; the inner-lambda
    caches truncate identically (``P`` by column, ``M = P^T P`` by both axes,
    which equals the sliced ``P``'s own Gram).
    """
    if k >= cache["w"].shape[0]:
        return cache
    out = dict(cache)
    out["w"] = cache["w"][-k:]
    out["V"] = cache["V"][:, -k:]
    out["KevV"] = cache["KevV"][:, -k:]
    inner = cache.get("inner")
    if inner:
        trunc_inner = []
        for ic in inner:
            kk = min(k, ic["w"].shape[0])
            trunc_inner.append(
                {
                    "w": ic["w"][-kk:],
                    "V": ic["V"][:, -kk:],
                    "P": ic["P"][:, -kk:],
                    "M": ic["M"][-kk:, -kk:],
                    "fi_idx": ic["fi_idx"],
                    "va_idx": ic["va_idx"],
                }
            )
        out["inner"] = trunc_inner
    return out


def _pca_k(n_train: int) -> int:
    """k = min(1024, floor(n_train / 2)) — the well-posed companion basis."""
    return max(1, min(PCA_K_CAP, n_train // 2))


def _fold_map(conv_ids, n_folds: int = N_FOLDS, seed: int = FIT_SEED) -> dict:
    """{conv_id -> fold} under the PARENT's seeded unique-id permutation."""
    import numpy as np

    from scripts.issue825_fit_cells import _cv_folds

    ids = np.asarray(conv_ids)
    folds = _cv_folds(ids, n_folds, seed)
    return {c: int(f) for c, f in zip(ids, folds, strict=True)}


def _pooled(ss_res: float, ss_tot: float) -> float:
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Baselines / retrieval on held-out rows
# ---------------------------------------------------------------------------


def _baselines_heldout(x_ev, y_ev, x_tr, y_tr, *, ks=(1, 5, 10)) -> dict:
    """identity+learned-bias R2 + kNN retrieval of the fitted map's targets.

    ``x_*``/``y_*`` are the concatenated held-out (eval) and train rows across
    folds. The retrieval pool is the eval set's OWN true targets (chance =
    k / n_pool, stated by the helper).
    """
    import numpy as np

    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    out: dict = {}
    if x_ev.shape[1] == y_ev.shape[1]:
        pred = identity_bias_predict(x_tr, y_tr, x_ev)
        mu = y_ev.mean(0)
        out["identity_bias_r2"] = _pooled(
            float(np.sum((y_ev - pred) ** 2)), float(np.sum((y_ev - mu) ** 2))
        )
        out["identity_bias_knn_euclidean"] = knn_retrieval(pred, y_ev, ks=ks, metric="euclidean")
        out["identity_bias_knn_cosine"] = knn_retrieval(pred, y_ev, ks=ks, metric="cosine")
    else:
        out["identity_bias_r2"] = None
        out["identity_bias_inapplicable"] = (
            f"input dim {x_ev.shape[1]} != output dim {y_ev.shape[1]}"
        )
    return out


def _null_from_eval_targets(pred, true, conv_ids_ev, *, n_draws: int, seed: int) -> dict:
    """Comparison-side conversation-level shuffle null (ZERO refits).

    Permutes the EVAL targets at the conversation level and re-scores the SAME
    predictions — the "is this pairing better than chance" null for a transfer
    read, reported under a key distinct from the fit-side shuffle null so the
    two families are never pooled.
    """
    import numpy as np

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    ids = np.asarray(conv_ids_ev)
    uniq, inv = np.unique(ids, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draws):
        perm = np.concatenate([rows_of[k] for k in rng.permutation(len(uniq))])
        t = true[perm]
        mu = t.mean(0)
        vals.append(_pooled(float(np.sum((t - pred) ** 2)), float(np.sum((t - mu) ** 2))))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n_draws": int(n_draws),
        "mean": float(np.nanmean(arr)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p97_5": float(np.nanpercentile(arr, 97.5)),
        "max": float(np.nanmax(arr)),
    }


# ---------------------------------------------------------------------------
# Core fit paths
# ---------------------------------------------------------------------------


def fit_within(X, Y, conv_ids, *, null_draws: int = N_NULL_DRAWS) -> dict:
    """Within-cell held-out fit via the PARENT's ``heldout_r2_sweep``.

    Byte-identical estimator to the published per-cell run (5 conversation-
    grouped folds, seed 42, inner-group-cv lambda selection over the module's
    13-lambda grid, batched shuffled-Y nulls off the cached factorization), so
    Gate-1 compares like with like.
    """
    import numpy as np

    from scripts.issue825_fit_cells import heldout_r2_sweep

    sweep = heldout_r2_sweep(
        np.asarray(X, dtype=np.float32)[:, None, :],
        np.asarray(Y, dtype=np.float32)[:, None, :],
        np.asarray(conv_ids),
        n_folds=N_FOLDS,
        seed=FIT_SEED,
        null_draws=null_draws,
        collect_cosines=True,
        collect_lambdas=True,
        lambda_selection="inner-group-cv",
        frozen_layers=(0,),
    )
    r2 = float(sweep["r2_obs"][0])
    nulls = np.asarray(sweep["r2_null"])[:, 0] if null_draws else np.zeros(0)
    mask = np.asarray(sweep["fitted_mask"], dtype=bool)
    pred = np.asarray(sweep["preds_frozen"][0], dtype=np.float64)[mask]
    y_ev = np.asarray(Y, dtype=np.float64)[mask]
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    out = {
        "r2": r2,
        "n_rows": int(np.asarray(X).shape[0]),
        "n_fitted": int(mask.sum()),
        "lambdas_selected": [
            None if not np.isfinite(v) else float(v) for v in np.asarray(sweep["gcv_lambda"])[0]
        ],
        "lambda_selector": sweep["lambda_selector"][0],
        "cosine_mean": float(np.nanmean(np.asarray(sweep["cosines"][0])[mask])),
        "knn_euclidean": knn_retrieval(pred, y_ev, metric="euclidean"),
        "knn_cosine": knn_retrieval(pred, y_ev, metric="cosine"),
    }
    if null_draws:
        out["null_shuffle_fit_targets"] = {
            "n_draws": int(null_draws),
            "mean": float(np.nanmean(nulls)),
            "p50": float(np.nanpercentile(nulls, 50)),
            "p97_5": float(np.nanpercentile(nulls, 97.5)),
            "max": float(np.nanmax(nulls)),
        }
    # identity+learned-bias baseline on the SAME folds, evaluated HELD-OUT (the
    # parent per-cell script evaluated it in-sample over all rows).
    out.update(_identity_bias_folded(X, Y, np.asarray(sweep["folds"])))
    return out


def _identity_bias_folded(X, Y, folds) -> dict:
    """Fold-wise identity+learned-bias baseline: bias from the train fold,
    scored on the held-out fold, pooled across folds."""
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    folds = np.asarray(folds)
    preds: list = []
    trues: list = []
    ss_res = ss_tot = 0.0
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            continue
        pred = _baselines_predict_identity(X[tr], Y[tr], X[te])
        if pred is None:
            return {
                "identity_bias_r2": None,
                "identity_bias_inapplicable": (
                    f"input dim {X.shape[1]} != output dim {Y.shape[1]}"
                ),
            }
        true = Y[te]
        ss_res += float(np.sum((true - pred) ** 2))
        ss_tot += float(np.sum((true - true.mean(0)) ** 2))
        preds.append(pred)
        trues.append(true)
    if not preds:
        return {"identity_bias_r2": None, "identity_bias_inapplicable": "no usable folds"}
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    p_all = np.concatenate(preds)
    t_all = np.concatenate(trues)
    return {
        "identity_bias_r2": _pooled(ss_res, ss_tot),
        "identity_bias_knn_euclidean": knn_retrieval(p_all, t_all, metric="euclidean"),
        "identity_bias_knn_cosine": knn_retrieval(p_all, t_all, metric="cosine"),
    }


def _baselines_predict_identity(x_tr, y_tr, x_ev):
    """``identity_bias_predict`` guarded on the same-dimension precondition."""
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    if x_tr.shape[1] != y_tr.shape[1]:
        return None
    return identity_bias_predict(x_tr, y_tr, x_ev)


def fit_within_reduced(X, Y, conv_ids, *, null_draws: int = N_NULL_DRAWS) -> dict:
    """Reduced-PCA-basis companion of :func:`fit_within` (k = min(1024, n_tr//2)).

    Own fold loop so the basis is fitted on the TRAIN fold only (no leakage) and
    so the truncation reuses the SAME cached factorization the full fit would
    have used — one ``_prep_fold`` + one ``_prep_inner_lambda`` per fold, then
    the top-k slice.
    """
    import numpy as np

    from scripts.issue825_fit_cells import (
        N_INNER_LAMBDA_FOLDS,
        _cv_folds,
        _null_ss_contrib,
        _prep_fold,
        _prep_inner_lambda,
        _ridge_predict_cached,
    )

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    ids = np.asarray(conv_ids)
    folds = _cv_folds(ids, N_FOLDS, FIT_SEED)
    rng = np.random.default_rng(FIT_SEED + 1)
    uniq, inv = np.unique(ids, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    null_perms = [
        np.concatenate([rows_of[k] for k in rng.permutation(len(uniq))]) for _ in range(null_draws)
    ]

    ss_res = ss_tot = 0.0
    ss_res_null = np.zeros(null_draws)
    ss_tot_null = np.zeros(null_draws)
    ks_used: list[int] = []
    lams: list[float] = []
    for k in range(N_FOLDS):
        te = folds == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            continue
        cache = _prep_fold(X[tr], X[te])
        cache["inner"] = _prep_inner_lambda(
            X[tr], ids[tr], N_INNER_LAMBDA_FOLDS, FIT_SEED + 4242 + k
        )
        kk = _pca_k(int(tr.sum()))
        cache = _truncate_cache(cache, kk)
        ks_used.append(kk)
        pred, lam = _ridge_predict_cached(cache, Y[tr], return_lam=True)
        lams.append(float(lam))
        true = Y[te].astype(np.float64)
        ss_res += float(np.sum((true - pred) ** 2))
        ss_tot += float(np.sum((true - true.mean(0)) ** 2))
        if null_perms:
            ssr, sst = _null_ss_contrib(cache, Y, tr, te, null_perms, impl="batched")
            ss_res_null += ssr
            ss_tot_null += sst
    out = {"r2": _pooled(ss_res, ss_tot), "pca_k_per_fold": ks_used, "lambdas_selected": lams}
    if null_draws:
        with np.errstate(divide="ignore", invalid="ignore"):
            nulls = 1.0 - ss_res_null / np.where(ss_tot_null < 1e-12, np.nan, ss_tot_null)
        out["null_shuffle_fit_targets"] = {
            "n_draws": int(null_draws),
            "mean": float(np.nanmean(nulls)),
            "p50": float(np.nanpercentile(nulls, 50)),
            "p97_5": float(np.nanpercentile(nulls, 97.5)),
            "max": float(np.nanmax(nulls)),
        }
    return out


def fit_transfer(
    Xs, Ys, ids_s, Xt, Yt, ids_t, *, reduced: bool = False, null_draws: int = N_NULL_DRAWS
) -> dict:
    """Fit on SOURCE train folds, evaluate on TARGET test folds (leakage-free).

    Folds come from the union of both cells' conversation ids under the parent's
    seeded permutation, so a conversation in fold k is excluded from the source
    train side and supplies the target eval side — no conversation is ever on
    both sides of a fold.
    """
    import numpy as np

    from scripts.issue825_fit_cells import (
        N_INNER_LAMBDA_FOLDS,
        _prep_fold,
        _prep_inner_lambda,
        _ridge_predict_cached,
    )

    Xs = np.asarray(Xs, dtype=np.float32)
    Ys = np.asarray(Ys, dtype=np.float32)
    Xt = np.asarray(Xt, dtype=np.float32)
    Yt = np.asarray(Yt, dtype=np.float32)
    ids_s = np.asarray(ids_s)
    ids_t = np.asarray(ids_t)
    fmap = _fold_map(np.concatenate([ids_s, ids_t]))
    fs = np.array([fmap[c] for c in ids_s])
    ft = np.array([fmap[c] for c in ids_t])

    ss_res = ss_tot = 0.0
    preds: list = []
    trues: list = []
    ev_ids: list = []
    ev_x: list = []
    tr_x: list = []
    tr_y: list = []
    lams: list[float] = []
    ks_used: list[int] = []
    n_folds_used = 0
    for k in range(N_FOLDS):
        tr = fs != k
        te = ft == k
        if te.sum() == 0 or tr.sum() < 3:
            continue
        cache = _prep_fold(Xs[tr], Xt[te])
        cache["inner"] = _prep_inner_lambda(
            Xs[tr], ids_s[tr], N_INNER_LAMBDA_FOLDS, FIT_SEED + 4242 + k
        )
        if reduced:
            kk = _pca_k(int(tr.sum()))
            cache = _truncate_cache(cache, kk)
            ks_used.append(kk)
        pred, lam = _ridge_predict_cached(cache, Ys[tr], return_lam=True)
        lams.append(float(lam))
        true = Yt[te].astype(np.float64)
        ss_res += float(np.sum((true - pred) ** 2))
        ss_tot += float(np.sum((true - true.mean(0)) ** 2))
        preds.append(np.asarray(pred, dtype=np.float64))
        trues.append(true)
        ev_ids.append(ids_t[te])
        ev_x.append(Xt[te].astype(np.float64))
        tr_x.append(Xs[tr].astype(np.float64))
        tr_y.append(Ys[tr].astype(np.float64))
        n_folds_used += 1
    if not preds:
        raise RuntimeError("transfer produced no usable folds")
    pred_all = np.concatenate(preds)
    true_all = np.concatenate(trues)
    ids_all = np.concatenate(ev_ids)
    out: dict = {
        "r2": _pooled(ss_res, ss_tot),
        "n_source_rows": int(Xs.shape[0]),
        "n_target_rows": int(Xt.shape[0]),
        "n_eval_rows": int(pred_all.shape[0]),
        "n_folds_used": n_folds_used,
        "lambdas_selected": lams,
        "reduced_basis": bool(reduced),
    }
    if reduced:
        out["pca_k_per_fold"] = ks_used
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    out["knn_euclidean"] = knn_retrieval(pred_all, true_all, metric="euclidean")
    out["knn_cosine"] = knn_retrieval(pred_all, true_all, metric="cosine")
    # identity+bias baseline on the SAME held-out rows the ridge was scored on:
    # bias learned from the SOURCE train rows (the map's own training data),
    # evaluated on the TARGET eval rows — the transfer analogue of the
    # within-cell baseline.
    out.update(
        _baselines_heldout(
            np.concatenate(ev_x),
            true_all,
            np.concatenate(tr_x),
            np.concatenate(tr_y),
        )
    )
    if null_draws:
        out["null_shuffle_eval_targets"] = _null_from_eval_targets(
            pred_all, true_all, ids_all, n_draws=null_draws, seed=FIT_SEED + 7
        )
    return out


# ---------------------------------------------------------------------------
# Store IO
# ---------------------------------------------------------------------------


def load_store(path: Path) -> dict:
    import torch

    store = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("slots", "conv_ids", "dup_count", "slot_names"):
        if key not in store:
            raise RuntimeError(f"{path}: store missing key {key!r}")
    return store


def _store_d_model(store: dict) -> int:
    """Hidden width from the store's own field, falling back to a slot shape."""
    import numpy as np

    d = store.get("d_model")
    if d:
        return int(d)
    first = store["slot_names"][0]
    return int(np.asarray(store["slots"][first]).shape[1])


def expand_by_dup(arr, dup_count):
    """Repeat each row ``dup_count`` times — reconstructs a parent row set whose
    duplicate groups were byte-identical (see the module docstring)."""
    import numpy as np

    return np.repeat(np.asarray(arr), np.asarray(dup_count).astype(int), axis=0)


def stage_parent_condition_store(
    model_dir_name: str, condition: str, stage_root: Path, revision: str
) -> Path:
    """Stage ONE parent ``analysis_tensors/<model>/<condition>/L19.pt`` bundle."""
    from scripts.issue1689_user_slot_render import DATA_REPO, PARENT_REVISION
    from scripts.issue1689_common import HF_DATA_PREFIX
    from explore_persona_space.orchestrate.hub import stage_hub_file

    rel = f"{HF_DATA_PREFIX}/analysis_tensors/{model_dir_name}/{condition}/L19.pt"
    return stage_hub_file(DATA_REPO, rel, stage_root / rel, revision=revision or PARENT_REVISION)


def stage_parent_assistant_store(model_dir_name: str, stage_root: Path, revision: str) -> Path:
    """Stage the PARENT ``assistant_chat`` L19 store for the naive cross-role read."""
    return stage_parent_condition_store(model_dir_name, "assistant_chat", stage_root, revision)


def stage_parent_rendered_condition(condition: str, stage_root: Path, revision: str) -> list[Path]:
    """Stage every ``rendered/<condition>.shard*.jsonl`` shard the parent published.

    Scoped ``list_repo_tree(path_in_repo=...)`` + per-file ``stage_hub_file`` —
    never ``snapshot_download`` (the data repo is ~1M files; an allow_patterns
    snapshot walks the WHOLE tree before filtering, gotchas.md).
    """
    from huggingface_hub import HfApi

    from scripts.issue1689_user_slot_render import DATA_REPO, PARENT_REVISION
    from scripts.issue1689_common import HF_DATA_PREFIX
    from explore_persona_space.orchestrate.hub import list_hf_files_under_path, stage_hub_file

    rev = revision or PARENT_REVISION
    prefix = f"{HF_DATA_PREFIX}/rendered"
    # Canonical scoped + retried listing (never a bare list_repo_tree / a
    # full-repo listing — #920/#997 and the ~1M-file data repo, gotchas.md).
    paths = list_hf_files_under_path(HfApi(), DATA_REPO, prefix, repo_type="dataset", revision=rev)
    want = sorted(p for p in paths if f"/{condition}.shard" in p)
    if not want:
        raise RuntimeError(
            f"no rendered shards for condition {condition!r} under {prefix} at revision {rev}"
        )
    return [stage_hub_file(DATA_REPO, rel, stage_root / rel, revision=rev) for rel in want]


_TOKENIZER_CACHE: dict[str, object] = {}


def _get_tokenizer(model_name: str):
    """Module-scope cached tokenizer. NEVER call ``from_pretrained`` per row —
    newer transformers issues a Hub ``model_info()`` request per load, which
    trips the ~2500-req/5-min org rate limit (gotchas.md)."""
    tok = _TOKENIZER_CACHE.get(model_name)
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZER_CACHE[model_name] = tok
    return tok


def _rendered_context_text(row: dict, tokenizer, framing: str) -> str:
    """The parent capture's own CONTEXT text for one rendered row.

    Chat framing re-applies ``apply_chat_template(..., add_generation_prompt=True)``
    (the capture's ``_render_chat_offsets`` convention); plain-text framings
    concatenate the renderer's ``prefix_text_only + u2_text_marked +
    context_tail`` (its ``_resolve_row_offsets`` convention).
    """
    if framing == "chat":
        return tokenizer.apply_chat_template(
            list(row["messages"]), tokenize=False, add_generation_prompt=True
        )
    return row["prefix_text_only"] + row.get("u2_text_marked", "") + row.get("context_tail", "")


def context_token_lengths(
    paths: list[Path], tokenizer, framing: str
) -> tuple[dict[str, int], dict]:
    """Per-conv CONTEXT token length + a digest, from the parent's rendered rows.

    The parent publishes each conversation THREE times per condition (one row per
    a2 sample; the stores keep/drop whole 3-row conversation groups), and all three
    rows carry a BYTE-IDENTICAL context — verified here per conversation via a
    content hash, so a genuine context divergence inside a duplicate group fails
    LOUD instead of silently picking one. Only the first occurrence is tokenized.

    Tokenized with ``add_special_tokens=False``, matching the capture. This is the
    STRATIFICATION variable only: it equals the capture's ``context_end`` token
    index up to at most one BPE-seam token (the capture tokenizes ``context + a2``
    in one pass, so a merge at that seam can shift the boundary by one, gotchas.md
    § teacher-forced capture) — immaterial for a median split over a distribution
    spanning hundreds of tokens, and never used as a measured quantity.

    JSONL is read by text-mode iteration, never ``splitlines()`` (real user text
    carries raw U+2028/NEL, #825).
    """
    import hashlib

    texts: list[str] = []
    conv_ids: list[str] = []
    hash_by_conv: dict[str, str] = {}
    n_rows = 0
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = str(row["conv_id"])
                n_rows += 1
                text = _rendered_context_text(row, tokenizer, framing)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                prev = hash_by_conv.get(cid)
                if prev is None:
                    hash_by_conv[cid] = digest
                    texts.append(text)
                    conv_ids.append(cid)
                elif prev != digest:
                    raise RuntimeError(
                        f"conv_id {cid!r} has DIVERGENT context text across its duplicate "
                        f"rendered rows ({prev[:12]} vs {digest[:12]}) — the length variable "
                        f"is not a per-conversation property for this condition"
                    )
    if not texts:
        raise RuntimeError(f"no rendered rows read from {len(paths)} shard(s)")
    # ONE batched tokenizer call (never a per-row loop, and never a per-row
    # from_pretrained — the HF 429 trap in gotchas.md).
    enc = tokenizer(texts, add_special_tokens=False)
    lengths = {cid: int(len(ids)) for cid, ids in zip(conv_ids, enc["input_ids"], strict=True)}
    digest_out = {
        "n_rendered_rows": n_rows,
        "n_distinct_convs": len(lengths),
        "duplicate_factor": (n_rows / len(lengths)) if lengths else None,
        "n_tokenized": len(texts),
        "framing": framing,
    }
    print(
        f"[lensplit] length table ({framing}): rendered_rows={n_rows} "
        f"convs={len(lengths)} dup_factor={digest_out['duplicate_factor']:.2f} "
        f"tokenized={len(texts)}",
        flush=True,
    )
    return lengths, digest_out


def conv_rank_median_split(conv_ids, lengths_by_conv: dict[str, int], *, seed: int) -> dict:
    """Median split at CONVERSATION level, matched on ROW count.

    Ranks distinct conversations by ``(context_tokens, conv_id)`` (deterministic
    tie-break), takes the lower half as SHORT and the upper half as LONG, then
    drops whole conversations at random (seeded, uniform WITHIN the larger half so
    the drop is length-neutral in expectation) until the two halves carry the same
    number of ROWS. Splitting at conversation level keeps every conversation's rows
    inside ONE half, so the conversation-grouped folds inside each half stay clean
    and the halves are genuinely length-separated.
    """
    import numpy as np

    ids = np.asarray(conv_ids, dtype=object)
    missing = sorted({str(c) for c in ids} - set(lengths_by_conv))
    if missing:
        raise RuntimeError(
            f"{len(missing)} store conv_id(s) absent from the rendered length table "
            f"(first 3: {missing[:3]}) — the rendered shards do not match this store"
        )
    rows_by_conv: dict[str, list[int]] = {}
    for i, c in enumerate(ids):
        rows_by_conv.setdefault(str(c), []).append(i)
    order = sorted(rows_by_conv, key=lambda c: (lengths_by_conv[c], c))
    half = len(order) // 2
    short_convs = list(order[:half])
    long_convs = list(order[len(order) - half :])
    dropped_middle = [c for c in order if c not in set(short_convs) | set(long_convs)]

    rng = np.random.default_rng(seed)
    dropped_match: dict[str, list[str]] = {"short": [], "long": []}

    def _nrows(convs: list[str]) -> int:
        return sum(len(rows_by_conv[c]) for c in convs)

    guard = 0
    while _nrows(short_convs) != _nrows(long_convs):
        guard += 1
        if guard > len(order) + 2:  # cannot happen: each drop removes >=1 row
            raise RuntimeError("row-count matching failed to converge")
        bigger, name = (
            (short_convs, "short")
            if _nrows(short_convs) > _nrows(long_convs)
            else (long_convs, "long")
        )
        if not bigger:
            raise RuntimeError("row-count matching emptied a half")
        victim = bigger[int(rng.integers(len(bigger)))]
        bigger.remove(victim)
        dropped_match[name].append(victim)

    short_idx = np.array(sorted(i for c in short_convs for i in rows_by_conv[c]), dtype=int)
    long_idx = np.array(sorted(i for c in long_convs for i in rows_by_conv[c]), dtype=int)
    if short_idx.shape[0] != long_idx.shape[0]:
        raise RuntimeError("matched-n invariant violated after conversation matching")
    s_len = np.array([lengths_by_conv[c] for c in short_convs], dtype=float)
    l_len = np.array([lengths_by_conv[c] for c in long_convs], dtype=float)

    def _stats(a):
        return {
            "n_convs": int(a.shape[0]),
            "min": float(a.min()),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "max": float(a.max()),
            "mean": float(a.mean()),
        }

    all_len = np.array([lengths_by_conv[c] for c in order], dtype=float)
    return {
        "short_idx": short_idx,
        "long_idx": long_idx,
        "meta": {
            "split_level": "conversation",
            "rank_key": "(context_tokens, conv_id)",
            "seed": int(seed),
            "n_convs_total": int(len(order)),
            "n_rows_total": int(ids.shape[0]),
            "n_rows_per_half": int(short_idx.shape[0]),
            "overall_median_context_tokens": float(np.median(all_len)),
            "boundary_short_max": float(s_len.max()) if s_len.size else None,
            "boundary_long_min": float(l_len.min()) if l_len.size else None,
            "short": _stats(s_len),
            "long": _stats(l_len),
            "dropped_middle_conv_count": len(dropped_middle),
            "dropped_for_row_match": {k: len(v) for k, v in dropped_match.items()},
        },
    }


def fit_length_stratified(
    store_parent: dict,
    lengths_by_conv: dict[str, int],
    *,
    null_draws: int,
    arms=None,
    refit_full: bool = False,
    published: dict | None = None,
) -> dict:
    """Refit each arm of a PARENT store on the SHORT and LONG length halves.

    Matched-n by construction (``conv_rank_median_split``). The full-set
    reference is the PARENT's PUBLISHED L19 R2 by default (byte-identical
    estimator — that is what Gate-1 verifies), so the default cost is the two
    half-fits per arm; ``refit_full=True`` adds a local full-set refit.
    """
    import numpy as np

    arms = tuple(arms or LENGTH_SPLIT_ARMS)
    ids = np.asarray(store_parent["conv_ids"], dtype=object)
    split = conv_rank_median_split(ids, lengths_by_conv, seed=FIT_SEED)
    si, li = split["short_idx"], split["long_idx"]
    out: dict = {"split": split["meta"], "arms": {}}
    for x_key, y_key, arm_name in arms:
        X = np.asarray(store_parent[x_key])
        Y = np.asarray(store_parent[y_key])
        res: dict = {"x_key": x_key, "y_key": y_key}
        for half, idx in (("short", si), ("long", li)):
            res[half] = fit_within(X[idx], Y[idx], ids[idx], null_draws=null_draws)
        res["delta_r2_long_minus_short"] = res["long"]["r2"] - res["short"]["r2"]
        if published is not None:
            res["r2_full_published"] = published.get(arm_name)
        if refit_full:
            res["full_refit"] = fit_within(X, Y, ids, null_draws=null_draws)
        out["arms"][arm_name] = res
    return out


def published_parent_r2(percell_dir: Path, model_dir_name: str, condition: str) -> dict:
    """Published L19 held-out R2 per arm for a parent cell (the full-set reference)."""
    path = percell_dir / f"heldout_r2_{model_dir_name}_{condition}.json"
    if not path.exists():
        raise FileNotFoundError(f"published parent per-cell reference missing: {path}")
    with path.open(encoding="utf-8") as fh:
        ref = json.load(fh)
    li = list(ref["layers"]).index(19)
    return {
        f"{arm}->answer": float(ref[arm]["held_out_r2_per_layer"][li])
        for arm in ("prefix", "context")
        if arm in ref
    } | {"n_rows_published": int(ref["n_rows"])}


# ---------------------------------------------------------------------------
# Gate-1
# ---------------------------------------------------------------------------


def gate1_parent_parity(stores: dict[str, dict], percell_dir: Path) -> dict:
    """Reproduce the published per-cell L19 prefix-arm R2 on chat/lmsys.

    FAIL LOUD on |dR2| > GATE1_TOL, on a missing store, or on a missing
    published reference — a silent skip would let a mis-captured rig through.
    """
    import numpy as np

    results = []
    for model_dir_name in ("Qwen_Qwen2.5-7B", "Qwen_Qwen2.5-7B-Instruct"):
        unit_id = f"{model_dir_name}__chat__lmsys"
        if unit_id not in stores:
            raise RuntimeError(f"Gate-1: store for {unit_id} absent — cannot run the parity gate")
        st = stores[unit_id]
        ref_path = percell_dir / f"heldout_r2_{model_dir_name}_user_lmsys_chat.json"
        if not ref_path.exists():
            raise RuntimeError(f"Gate-1: published reference missing: {ref_path}")
        with ref_path.open(encoding="utf-8") as fh:
            ref = json.load(fh)
        li = list(ref["layers"]).index(19)
        ref_r2 = float(ref["prefix"]["held_out_r2_per_layer"][li])
        dup = st["dup_count"]
        X = expand_by_dup(st["slots"]["prev_turn_end"], dup)
        Y = expand_by_dup(st["slots"]["parent_answer_end"], dup)
        ids = expand_by_dup(np.asarray(st["conv_ids"], dtype=object), dup)
        got = fit_within(X, Y, ids, null_draws=0)
        d = abs(got["r2"] - ref_r2)
        row = {
            "unit_id": unit_id,
            "published_r2_L19_prefix": ref_r2,
            "recaptured_r2": got["r2"],
            "abs_delta": d,
            "tol": GATE1_TOL,
            "n_rows_expanded": got["n_rows"],
            "n_rows_published": int(ref["n_rows"]),
            "pass": bool(d <= GATE1_TOL),
        }
        print(
            f"[gate1] {unit_id}: published={ref_r2:+.4f} recaptured={got['r2']:+.4f} "
            f"|d|={d:.4f} n={got['n_rows']} (published {ref['n_rows']}) "
            f"{'PASS' if row['pass'] else 'FAIL'}",
            flush=True,
        )
        results.append(row)
    bad = [r for r in results if not r["pass"]]
    if bad:
        raise RuntimeError(
            "Gate-1 FAILED — the re-captured parent-convention slot does not reproduce the "
            f"published per-cell L19 R2 within {GATE1_TOL}: {bad}"
        )
    return {"tol": GATE1_TOL, "rows": results, "pass": True}


def verify_truncation_equivalence(
    *, n: int = 60, d: int = 200, seed: int = 0, tol: float = 1e-9
) -> dict:
    """Truncation at k = n_train must reproduce the untruncated fit exactly.

    Pins the reduced-basis identity (dual-space ridge is invariant to an
    orthogonal reparametrization of the features, so a top-k slice of
    ``(w, V, KevV)`` IS the top-k PCA-basis fit). A real bug in the slicing
    (wrong eigen-order, mismatched ``M``) breaks this at once.
    """
    import numpy as np

    from scripts.issue825_fit_cells import (
        N_INNER_LAMBDA_FOLDS,
        _prep_fold,
        _prep_inner_lambda,
        _ridge_predict_cached,
    )

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = (X @ rng.standard_normal((d, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    ids = np.array([f"c{i // 2}" for i in range(n)])
    tr = np.arange(n) % 5 != 0
    te = ~tr
    cache = _prep_fold(X[tr], X[te])
    cache["inner"] = _prep_inner_lambda(X[tr], ids[tr], N_INNER_LAMBDA_FOLDS, FIT_SEED + 4242)
    base = _ridge_predict_cached(cache, Y[tr])
    full = _ridge_predict_cached(_truncate_cache(cache, int(tr.sum())), Y[tr])
    max_abs = float(np.max(np.abs(base - full)))
    half = _ridge_predict_cached(_truncate_cache(cache, max(1, int(tr.sum()) // 2)), Y[tr])
    half_delta = float(np.max(np.abs(base - half)))
    scale = float(np.max(np.abs(base)))
    bite_floor = 1e-3 * scale
    ok = max_abs <= tol
    print(
        f"[verify-truncation] n_train={int(tr.sum())} d={d} (production regime n_train < d) "
        f"k=n_train max_abs={max_abs:.3e} (tol {tol:.1e}) {'PASS' if ok else 'FAIL'}; "
        f"k=n_train/2 differs by {half_delta:.3e} (floor {bite_floor:.3e})",
        flush=True,
    )
    if not ok:
        raise RuntimeError(f"truncation equivalence FAILED: max_abs={max_abs} > {tol}")
    if half_delta <= bite_floor:
        # A rank-degenerate fixture (n_train > d) makes every truncated direction
        # a null direction, so the slice would be inert and the identity above
        # would prove nothing about the production path.
        raise RuntimeError(
            f"truncation at k=n_train/2 changed nothing meaningful ({half_delta:.3e} <= "
            f"{bite_floor:.3e}) — the fixture is rank-degenerate, so the identity is vacuous"
        )
    return {
        "n_train": int(tr.sum()),
        "d": int(d),
        "max_abs_at_full_k": max_abs,
        "delta_at_half_k": half_delta,
        "bite_floor": bite_floor,
        "tol": tol,
        "pass": True,
    }


# ---------------------------------------------------------------------------
# Compute-character pre-flight: MEASURED basis + projection + wall fence
# ---------------------------------------------------------------------------

# Fold-solves per fit pair: the within-cell sweep (N_FOLDS) plus the
# reduced-basis companion's own fold loop (N_FOLDS).
FOLD_SOLVES_PER_FIT_PAIR = 2 * N_FOLDS


def measure_fold_basis(n: int, d: int, *, null_draws: int, seed: int = 0) -> dict:
    """Time ONE fold at a REAL production shape (never a guessed per-call cost).

    Measures ``_prep_fold`` + ``_prep_inner_lambda`` + one 13-lambda
    ``_ridge_predict_cached`` + the batched null contribution, and reports the
    process peak RSS. This is the sizing basis
    (`.claude/rules/plan-compute-sizing.md` § Per-cell fit phases) — the
    battery refuses to start on a guess.
    """
    import resource
    import time

    import numpy as np

    from scripts.issue825_fit_cells import (
        N_INNER_LAMBDA_FOLDS,
        _cv_folds,
        _null_ss_contrib,
        _prep_fold,
        _prep_inner_lambda,
        _ridge_predict_cached,
    )

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = rng.standard_normal((n, d)).astype(np.float32)
    ids = np.array([f"c{i}" for i in range(n)])
    folds = _cv_folds(ids, N_FOLDS, FIT_SEED)
    te = folds == 0
    tr = ~te
    t = {}
    t0 = time.time()
    cache = _prep_fold(X[tr], X[te])
    t["prep_fold_s"] = time.time() - t0
    t0 = time.time()
    cache["inner"] = _prep_inner_lambda(X[tr], ids[tr], N_INNER_LAMBDA_FOLDS, FIT_SEED + 4242)
    t["prep_inner_lambda_s"] = time.time() - t0
    t0 = time.time()
    _ridge_predict_cached(cache, Y[tr], return_lam=True)
    t["ridge_predict_13_lambda_s"] = time.time() - t0
    if null_draws:
        perms = [rng.permutation(n) for _ in range(null_draws)]
        t0 = time.time()
        _null_ss_contrib(cache, Y, tr, te, perms, impl="batched")
        t["null_draws_s"] = time.time() - t0
    else:
        t["null_draws_s"] = 0.0
    t["fold_total_s"] = sum(t.values())
    t["n"] = int(n)
    t["n_train"] = int(tr.sum())
    t["d"] = int(d)
    t["null_draws"] = int(null_draws)
    t["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(
        "[preflight] MEASURED fold basis at n={n} (n_train={n_train}) d={d} "
        "null_draws={null_draws}: prep_fold={prep_fold_s:.1f}s "
        "prep_inner={prep_inner_lambda_s:.1f}s predict={ridge_predict_13_lambda_s:.1f}s "
        "nulls={null_draws_s:.1f}s TOTAL={fold_total_s:.1f}s peak_rss={peak_rss_gb:.1f}GB".format(
            **t
        ),
        flush=True,
    )
    return t


def project_battery_wall(
    entries: list[dict],
    basis: dict,
    *,
    null_draws: int,
    length_split_n_rows: list[int] | None = None,
) -> dict:
    """Project the battery wall-time from the MEASURED basis.

    Per-fold cost scales ~n^2 (the Gram eigh is O(n^3) but the dominant batched
    null term is O(n_train^2 * D)), so each unit's cost is the basis scaled by
    ``(n / n_basis)^2``. Transfer/label/cross-role reads carry no fit-side nulls
    and are charged the null-free share of the basis. The addendum-E grid is
    charged with its X-side factorization SHARED across Y variants (what
    ``fit_grid`` actually does); ``length_split_n_rows`` charges the addendum-A
    half-fits at their matched half size.
    """
    n_basis = basis["n"]
    fold_total = basis["fold_total_s"]
    fold_no_null = fold_total - basis["null_draws_s"]
    prep_s = basis["prep_fold_s"] + basis["prep_inner_lambda_s"]
    predict_s = basis["ridge_predict_13_lambda_s"]
    null_s = basis["null_draws_s"]
    rows = []
    within_s = 0.0
    grid_s = 0.0
    for e in entries:
        n = int(e["n_rows"])
        n_pairs = len(e["fit_pairs"])
        scale = (n / n_basis) ** 2
        cost = n_pairs * FOLD_SOLVES_PER_FIT_PAIR * fold_total * scale
        within_s += cost
        # addendum E: per read group, one factorization per (X kind, fold) reused
        # across every Y kind; each combo pays a full + reduced predict and nulls.
        n_groups = len(e.get("read_group_names") or ())
        n_x = len(e.get("grid_x_kinds") or ())
        n_y = len(e.get("grid_y_kinds") or ())
        g_cost = n_groups * N_FOLDS * (n_x * prep_s + n_x * n_y * (2 * predict_s + null_s)) * scale
        grid_s += g_cost
        rows.append(
            {
                "unit_id": e["unit_id"],
                "n_rows": n,
                "n_fit_pairs": n_pairs,
                "n_grid_combos": n_groups * n_x * n_y,
                "projected_s": cost + g_cost,
            }
        )
    # Transfers, counted from the REALIZED entry set rather than per-frame
    # constants: provenance transfers run only where BOTH provenances of a pair
    # are present in the (model, frame) group, so the 14 bridging frames — one
    # provenance each — contribute ZERO (a flat 6-per-frame charge would inflate
    # the projection, and the fence, by 84 phantom max-n reads). Story-label
    # transfers likewise run per (model, provenance) where BOTH story variants
    # exist, 2 directions each; cross-role is 2 per model.
    by_frame: dict[tuple[str, str], set[str]] = {}
    story_provs: dict[str, dict[str, set[str]]] = {}
    for e in entries:
        frame = e["framing"] if e["variant"] == "base" else f"{e['framing']}_{e['variant']}"
        by_frame.setdefault((e["model_dir"], frame), set()).add(e["provenance"])
        if e["framing"] == "story":
            story_provs.setdefault(e["model_dir"], {}).setdefault(e["provenance"], set()).add(
                e["variant"]
            )
    n_prov_transfer = 2 * sum(
        1
        for provs in by_frame.values()
        for a, b in PROVENANCE_TRANSFER_PAIRS
        if a in provs and b in provs
    )
    n_label_transfer = 2 * sum(
        1
        for variants in story_provs.values()
        for got in variants.values()
        if {"alex", "user_label"} <= got
    )
    n_models = len({e["model_dir"] for e in entries})
    n_transfer = n_prov_transfer + n_label_transfer + 2 * n_models
    max_n = max(int(e["n_rows"]) for e in entries)
    transfer_s = n_transfer * N_FOLDS * fold_no_null * (max_n / n_basis) ** 2
    gate1_s = 2 * N_FOLDS * fold_no_null * (3 * max_n / n_basis) ** 2
    # addendum A: 2 half-fits per arm per (model x condition) cell, each half at
    # HALF the parent cell's row count (matched-n), no reduced companion.
    split_rows = list(length_split_n_rows or ())
    length_split_s = sum(
        len(LENGTH_SPLIT_ARMS) * 2 * N_FOLDS * fold_total * ((n / 2) / n_basis) ** 2
        for n in split_rows
    )
    total_s = within_s + grid_s + transfer_s + gate1_s + length_split_s
    out = {
        "basis": basis,
        "null_draws": int(null_draws),
        "n_units": len(entries),
        "n_fit_pairs": sum(r["n_fit_pairs"] for r in rows),
        "n_grid_combos": sum(r["n_grid_combos"] for r in rows),
        "n_transfer_reads": n_transfer,
        "n_transfer_reads_breakdown": {
            "provenance": n_prov_transfer,
            "story_label": n_label_transfer,
            "cross_role": 2 * n_models,
        },
        "n_length_split_cells": len(split_rows),
        "within_hours": within_s / 3600.0,
        "grid_hours": grid_s / 3600.0,
        "transfer_hours": transfer_s / 3600.0,
        "gate1_hours": gate1_s / 3600.0,
        "length_split_hours": length_split_s / 3600.0,
        "total_hours": total_s / 3600.0,
        "projected_peak_rss_gb": basis["peak_rss_gb"] * (max_n / n_basis) ** 2,
        "per_unit": sorted(rows, key=lambda r: -r["projected_s"]),
    }
    print(
        f"[preflight] PROJECTION: {out['n_units']} units / {out['n_fit_pairs']} fit pairs "
        f"({FOLD_SOLVES_PER_FIT_PAIR} fold-solves each: sweep + reduced companion) + "
        f"{out['n_grid_combos']} grid combos (X-side factorization shared) + "
        f"{n_transfer} transfer reads (prov={n_prov_transfer} label={n_label_transfer} "
        f"cross_role={2 * n_models}) + {len(split_rows)} length-split cells -> "
        f"within={out['within_hours']:.2f}h grid={out['grid_hours']:.2f}h "
        f"transfer={out['transfer_hours']:.2f}h gate1={out['gate1_hours']:.2f}h "
        f"lensplit={out['length_split_hours']:.2f}h "
        f"TOTAL={out['total_hours']:.2f}h; peak RSS ~{out['projected_peak_rss_gb']:.1f}GB",
        flush=True,
    )
    return out


def enforce_wall_fence(projection: dict, max_hours: float) -> None:
    """Refuse to start a battery projected past ``max_hours``.

    A silent multi-day CPU run is exactly the failure the compute-character
    discipline exists to prevent, so the fence names the two levers instead of
    proceeding: run on a GPU (the Gram-space ridge is FLOP-bound — the parent
    module's own docstring measures ~1 min/cell on A100 vs ~1.9 h at 4 CPU
    threads) or cut ``--null-draws`` (measured at ~93% of the per-fold cost).
    """
    if projection["total_hours"] <= max_hours:
        return
    raise RuntimeError(
        f"projected battery wall {projection['total_hours']:.2f}h exceeds the "
        f"--max-wall-hours fence ({max_hours:.2f}h) at null_draws="
        f"{projection['null_draws']} on device-resolved backend. Levers: (a) run on a "
        f"GPU (--device auto on a GPU box; the Gram-space ridge is FLOP-bound), "
        f"(b) lower --null-draws (the batched null term measured "
        f"{projection['basis']['null_draws_s']:.0f}s of the "
        f"{projection['basis']['fold_total_s']:.0f}s per-fold total), or "
        f"(c) raise --max-wall-hours deliberately."
    )


# ---------------------------------------------------------------------------
# Addendum E: the X x Y grid battery (SHARED X-side factorization)
# ---------------------------------------------------------------------------


def fit_grid(
    grid: dict,
    conv_ids,
    *,
    x_kinds: tuple[str, ...],
    y_kinds: tuple[str, ...],
    null_draws: int = N_NULL_DRAWS,
) -> dict:
    """All ``len(x_kinds) x len(y_kinds)`` combos of one read group, sharing the
    X-side factorization across every Y variant.

    This sharing is LOAD-BEARING, not an optimization. Each (X, fold) pair costs
    one Gram eigh + one inner-lambda cache (measured 8.5 s of a 175 s per-fold
    total at production shape n=3800/d=3584/40 draws); the per-Y marginal cost is
    a ``V.T @ Y`` reduction plus the 13-lambda diagonal rescale. Fitting the six
    combos independently would build six factorizations per fold instead of two
    and projects the grid battery to ~145 h; sharing keeps it at ~2x the
    single-combo cost.

    ``grid`` maps a slot kind ("X_clean" / "Y_mean" / ...) to its (N, D) array.
    Returns ``{f"{x}->{y}": {...}}`` plus a ``shared_factorizations`` audit count
    so a reviewer can confirm the sharing actually happened.
    """
    import numpy as np

    from scripts.issue825_fit_cells import (
        N_INNER_LAMBDA_FOLDS,
        _cv_folds,
        _null_ss_contrib,
        _prep_fold,
        _prep_inner_lambda,
        _ridge_predict_cached,
    )

    ids = np.asarray(conv_ids)
    folds = _cv_folds(ids, N_FOLDS, FIT_SEED)
    rng = np.random.default_rng(FIT_SEED + 1)
    uniq, inv = np.unique(ids, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    null_perms = [
        np.concatenate([rows_of[k] for k in rng.permutation(len(uniq))]) for _ in range(null_draws)
    ]

    ys = {y: np.asarray(grid[y], dtype=np.float32) for y in y_kinds}
    acc: dict[tuple[str, str], dict] = {
        (x, y): {
            "ss_res": 0.0,
            "ss_tot": 0.0,
            "ss_res_red": 0.0,
            "ss_tot_red": 0.0,
            "ss_res_null": np.zeros(null_draws),
            "ss_tot_null": np.zeros(null_draws),
            "lams": [],
            "pred": [],
            "true": [],
            "ks": [],
        }
        for x in x_kinds
        for y in y_kinds
    }
    n_factorizations = 0
    for x in x_kinds:
        X = np.asarray(grid[x], dtype=np.float32)
        for k in range(N_FOLDS):
            te = folds == k
            tr = ~te
            if te.sum() == 0 or tr.sum() < 3:
                continue
            # ONE factorization per (X, fold) — reused by every Y below.
            cache = _prep_fold(X[tr], X[te])
            cache["inner"] = _prep_inner_lambda(
                X[tr], ids[tr], N_INNER_LAMBDA_FOLDS, FIT_SEED + 4242 + k
            )
            n_factorizations += 1
            kk = _pca_k(int(tr.sum()))
            cache_red = _truncate_cache(cache, kk)
            for y in y_kinds:
                Y = ys[y]
                a = acc[(x, y)]
                pred, lam = _ridge_predict_cached(cache, Y[tr], return_lam=True)
                true = Y[te].astype(np.float64)
                mu = true.mean(0)
                a["ss_res"] += float(np.sum((true - pred) ** 2))
                a["ss_tot"] += float(np.sum((true - mu) ** 2))
                a["lams"].append(float(lam))
                a["pred"].append(np.asarray(pred, dtype=np.float64))
                a["true"].append(true)
                pred_r = _ridge_predict_cached(cache_red, Y[tr])
                a["ss_res_red"] += float(np.sum((true - pred_r) ** 2))
                a["ss_tot_red"] += float(np.sum((true - mu) ** 2))
                a["ks"].append(kk)
                if null_perms:
                    ssr, sst = _null_ss_contrib(cache, Y, tr, te, null_perms, impl="batched")
                    a["ss_res_null"] += ssr
                    a["ss_tot_null"] += sst
            del cache, cache_red

    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    out: dict = {
        "shared_factorizations": n_factorizations,
        "unshared_would_have_been": n_factorizations * len(y_kinds),
        "n_rows": int(np.asarray(grid[x_kinds[0]]).shape[0]),
        "combos": {},
    }
    for (x, y), a in acc.items():
        if not a["pred"]:
            continue
        p_all = np.concatenate(a["pred"])
        t_all = np.concatenate(a["true"])
        row = {
            "x_slot": x,
            "y_slot": y,
            "r2": _pooled(a["ss_res"], a["ss_tot"]),
            "r2_reduced_basis": _pooled(a["ss_res_red"], a["ss_tot_red"]),
            "pca_k_per_fold": a["ks"],
            "lambdas_selected": a["lams"],
            "knn_euclidean": knn_retrieval(p_all, t_all, metric="euclidean"),
            "knn_cosine": knn_retrieval(p_all, t_all, metric="cosine"),
        }
        row.update(_identity_bias_folded(grid[x], grid[y], folds))
        if null_draws:
            with np.errstate(divide="ignore", invalid="ignore"):
                nulls = 1.0 - a["ss_res_null"] / np.where(
                    a["ss_tot_null"] < 1e-12, np.nan, a["ss_tot_null"]
                )
            row["null_shuffle_fit_targets"] = {
                "n_draws": int(null_draws),
                "mean": float(np.nanmean(nulls)),
                "p50": float(np.nanpercentile(nulls, 50)),
                "p97_5": float(np.nanpercentile(nulls, 97.5)),
                "max": float(np.nanmax(nulls)),
            }
        out["combos"][f"{x}->{y}"] = row
    return out


# ---------------------------------------------------------------------------
# Addenda B / C / D: the bridging reduction
# ---------------------------------------------------------------------------

# Addendum B pairs each single-turn cell against the FULL two-turn cell that
# holds the SAME answer target and the SAME speaker label, so the ONLY variable
# is whether the (u1, a1) turn is present. Both sides fit
# `answer_header_end -> parent_answer_end`, so their primary R2 are directly
# comparable. (`single_turn` is plain text, so the plain-text `*_naturalistic`
# recap cell is the counterpart — never the chat one, whose template differs.)
PREFIX_ABLATION_PAIRS: dict[str, str] = {
    "single_turn_assistant": "parent_recap_assistant_naturalistic",
    "single_turn_wren": "parent_recap_wren_naturalistic",
}

# Addendum C pairs the on-policy-a1 cell against the base cell with the SAME u2
# provenance, label and framing shape, so a1 provenance is the only variable.
# The Wren-labelled cell has NO label-matched base counterpart in this round (the
# base user families are Assistant-labelled), so it is reported with the
# confound named rather than presented as a matched contrast.
A1_PROVENANCE_PAIRS: dict[str, str | None] = {
    "onpolicy_a1_assistant": "naturalistic",
    "onpolicy_a1_wren": None,
}

GRID_PRIMARY_X = "X_clean"


def _r2_or_none(fits: dict, name: str) -> float | None:
    row = fits.get(name)
    return None if row is None else float(row["r2"])


def bridge_comparisons(per_unit: dict, entries: dict, percell_dir: Path) -> dict:
    """Assemble the addenda-B/C/D reads from ALREADY-fit results.

    A pure reduction — no fitting happens here; every number is read out of the
    per-cell `fits` / `grid` results the battery just produced. Absent cells
    (`--allow-missing`, a smoke slice, a partial capture) are SKIPPED with the
    reason recorded, never silently dropped.
    """
    models = sorted({e["model_dir"] for e in entries.values()})

    def unit_for(mdir: str, frame: str, prov: str) -> str:
        return f"{mdir}__{frame}__{prov}"

    def primary_r2(unit_id: str) -> float | None:
        o = per_unit.get(unit_id)
        if o is None:
            return None
        return _r2_or_none(o["fits"], entries[unit_id]["primary_fit"])

    from scripts.issue1689_user_slot_render import BRIDGE_PROVENANCE

    # --- B: prefix ablation (single-turn vs the matched full two-turn cell) ---
    prefix_ablation: dict[str, dict] = {}
    for mdir in models:
        for st_frame, tt_frame in PREFIX_ABLATION_PAIRS.items():
            su = unit_for(mdir, st_frame, BRIDGE_PROVENANCE["single_turn"])
            tu = unit_for(mdir, tt_frame, BRIDGE_PROVENANCE["parent_recap"])
            s_r2, t_r2 = primary_r2(su), primary_r2(tu)
            row = {
                "single_turn_unit": su,
                "single_turn_r2": s_r2,
                "two_turn_unit": tu,
                "two_turn_r2": t_r2,
                "delta_r2_two_turn_minus_single_turn": (
                    None if s_r2 is None or t_r2 is None else t_r2 - s_r2
                ),
                "shared_fit": "answer_header_end->parent_answer_end (context->answer)",
                "note": (
                    "both arms hold the answer target FIXED at the parent's own a2 for the same "
                    "conversation; the single-turn arm ABLATES the (u1, a1) turn"
                ),
            }
            if s_r2 is None or t_r2 is None:
                row["skipped"] = "one or both cells absent from this battery"
            prefix_ablation[f"{mdir}|{st_frame}"] = row

    # --- C: a1 provenance (on-policy a1 vs the off-policy LMSYS a1) ----------
    a1_provenance: dict[str, dict] = {}
    for mdir in models:
        for on_frame, base_frame in A1_PROVENANCE_PAIRS.items():
            ou = unit_for(mdir, on_frame, BRIDGE_PROVENANCE["onpolicy_a1"])
            o_r2 = primary_r2(ou)
            row: dict = {
                "onpolicy_a1_unit": ou,
                "onpolicy_a1_r2": o_r2,
                "u2_provenance": BRIDGE_PROVENANCE["onpolicy_a1"],
            }
            if base_frame is None:
                row["offpolicy_unit"] = None
                row["delta_r2_onpolicy_minus_offpolicy"] = None
                row["skipped"] = (
                    "no label-matched base cell in this round (the base user families are "
                    "Assistant-labelled) — a contrast against a base Assistant cell would "
                    "confound the speaker label with a1 provenance"
                )
            else:
                bu = unit_for(mdir, base_frame, BRIDGE_PROVENANCE["onpolicy_a1"])
                b_r2 = primary_r2(bu)
                row["offpolicy_unit"] = bu
                row["offpolicy_r2"] = b_r2
                row["delta_r2_onpolicy_minus_offpolicy"] = (
                    None if o_r2 is None or b_r2 is None else o_r2 - b_r2
                )
                row["note"] = (
                    "u2, framing and label matched; a1 is the model's OWN greedy reply on the "
                    "on-policy arm vs the corpus reply on the base arm"
                )
                if o_r2 is None or b_r2 is None:
                    row["skipped"] = "one or both cells absent from this battery"
            a1_provenance[f"{mdir}|{on_frame}"] = row

    # --- D: target-summary convention (Y_mean vs Y_end vs Y_boundary) --------
    target_convention: dict[str, dict] = {}
    for unit_id, o in sorted(per_unit.items()):
        for gname, g in (o.get("grid") or {}).items():
            combos = g["combos"]
            got = {
                y: (
                    float(combos[f"{GRID_PRIMARY_X}->{y}"]["r2"])
                    if f"{GRID_PRIMARY_X}->{y}" in combos
                    else None
                )
                for y in ("Y_mean", "Y_end", "Y_boundary")
            }
            row = {
                "unit_id": unit_id,
                "read_group": gname,
                "x_kind": GRID_PRIMARY_X,
                "r2_by_y": got,
                "delta_r2_mean_minus_end": (
                    None
                    if got["Y_mean"] is None or got["Y_end"] is None
                    else got["Y_mean"] - got["Y_end"]
                ),
                "convention_note": (
                    "Y_mean is the #1345/#825 convention; Y_end is the #1689 parent convention; "
                    "Y_boundary reads the response slot before the next speaker"
                ),
            }
            # The published parent number exists only for the recap cells (the
            # rig-bridge arms re-rendering a parent condition), and it was
            # measured under the END-token convention.
            e = entries[unit_id]
            if e["framing"] == "parent_recap":
                cond = e["variant"]
                ref_path = percell_dir / f"heldout_r2_{e['model_dir']}_{cond}.json"
                if ref_path.exists():
                    row["published_parent_end_convention"] = published_parent_r2(
                        percell_dir, e["model_dir"], cond
                    )
                else:
                    row["published_parent_end_convention"] = None
                    row["published_reference_absent"] = str(ref_path)
            target_convention[f"{unit_id}|{gname}"] = row

    return {
        "prefix_ablation": prefix_ablation,
        "a1_provenance": a1_provenance,
        "target_summary_convention": target_convention,
    }


# ---------------------------------------------------------------------------
# Synthetic end-to-end smoke
# ---------------------------------------------------------------------------


def _synthetic_specs() -> list[tuple[str, str, str]]:
    """(framing, variant, provenance) cells the synthetic smoke tree carries.

    The base families keep two provenances each so the provenance-transfer and
    story-label legs have pairs to run. The three BRIDGING families (addenda
    B/C/D) each carry ONE cell: their per-family code paths in this battery are
    the fit-pair table, the slot names (`answer_header_end` as X and
    `parent_answer_end` as Y — a target no base family fits) and the `answer`
    read-group grid shape, all of which one cell exercises; adding a second
    variant would re-run the identical paths. Their single provenance is why
    they contribute no transfer pairs — which is exactly what
    :func:`project_battery_wall` must NOT charge for.
    """
    from scripts.issue1689_user_slot_render import BRIDGE_PROVENANCE

    return [
        ("chat", "base", "lmsys"),
        ("chat", "base", "onpolicy"),
        ("naturalistic", "base", "lmsys"),
        ("naturalistic", "base", "onpolicy"),
        # The haiku arm is addendum C's off-policy-a1 counterpart: without it the
        # bridging reduction's C leg has nothing to pair against and its delta
        # arithmetic would first run in production.
        ("naturalistic", "base", "haiku"),
        ("story", "alex", "lmsys"),
        ("story", "alex", "onpolicy"),
        ("story", "user_label", "lmsys"),
        ("story", "user_label", "onpolicy"),
        ("single_turn", "assistant", BRIDGE_PROVENANCE["single_turn"]),
        ("onpolicy_a1", "assistant", BRIDGE_PROVENANCE["onpolicy_a1"]),
        ("parent_recap", "assistant_naturalistic", BRIDGE_PROVENANCE["parent_recap"]),
    ]


def build_synthetic_store_tree(root: Path, *, n: int = 60, d: int = 64, seed: int = 0) -> Path:
    """Write a synthetic store tree + render manifest with the REAL schema.

    Reduced dims (d=64, n=60) keep the smoke seconds-long, but the rank regime
    matches production (n_train = 48 < d = 64) so the reduced-basis truncation
    genuinely bites, and every store key / manifest field is the real one — the
    battery below runs its production code path unmodified.

    Covers one cell per FAMILY including the three bridging families, so no
    family's fit-pair table or grid shape first executes in production.
    """
    import numpy as np
    import torch

    from scripts.issue1689_user_slot_render import (
        FIT_PAIRS_BY_FRAMING,
        PRIMARY_FIT_BY_FRAMING,
        READ_GROUP_NAMES_BY_FRAMING,
        SLOT_STRADDLER_POLICY,
        SLOTS_BY_FRAMING,
        base_metadata,
    )

    rng = np.random.default_rng(seed)
    model = "Qwen/Qwen2.5-7B"
    mdir = model.replace("/", "_")
    specs = _synthetic_specs()
    entries = []
    conv_ids = np.array([f"c{i:04d}" for i in range(n)], dtype=object)
    for framing, variant, prov in specs:
        frame = framing if variant == "base" else f"{framing}_{variant}"
        unit_id = f"{mdir}__{frame}__{prov}"
        slots = SLOTS_BY_FRAMING[framing]
        # A shared latent per row makes the slots genuinely predictive of each
        # other, so a real R2 (not a pure null) exercises the reporting path.
        latent = rng.standard_normal((n, d))
        acc = {
            s: (latent @ rng.standard_normal((d, d)) * 0.3 + rng.standard_normal((n, d))).astype(
                np.float32
            )
            for s in slots
        }
        # Read from the render's own declaration (which its
        # `_validate_read_groups` asserts against every realized group list), so
        # the smoke's grid shape can never drift from production's.
        gnames = list(READ_GROUP_NAMES_BY_FRAMING[framing])
        gacc = {
            f"{gn}__{kind}": (
                latent @ rng.standard_normal((d, d)) * 0.3 + rng.standard_normal((n, d))
            ).astype(np.float32)
            for gn in gnames
            for kind in ("X_clean", "X_straddle", "Y_mean", "Y_end", "Y_boundary")
        }
        store = {
            "grid_slots": gacc,
            "grid_group_names": gnames,
            "grid_x_kinds": ["X_clean", "X_straddle"],
            "grid_y_kinds": ["Y_mean", "Y_end", "Y_boundary"],
            "slots": acc,
            "slot_token_pos": {
                s: np.arange(len(slots)).repeat(n)[:n].astype(np.int32) for s in slots
            },
            "seam_flags": {s: (rng.random(n) < 0.3).astype(np.int8) for s in slots},
            "n_tokens": np.full(n, 128, dtype=np.int32),
            "conv_ids": conv_ids,
            "dup_count": np.full(n, 3 if prov == "lmsys" else 1, dtype=np.int32),
            "row_index": np.arange(n, dtype=np.int32),
            "judge_score_mean": np.full(n, np.nan, dtype=np.float32),
            "unit": {
                "unit_id": unit_id,
                "model": model,
                "framing": framing,
                "provenance": prov,
                "variant": variant,
            },
            "slot_names": list(slots),
            "straddler_policy": {s: SLOT_STRADDLER_POLICY[s] for s in slots},
            "fit_pairs": [list(p) for p in FIT_PAIRS_BY_FRAMING[framing]],
            "primary_fit": PRIMARY_FIT_BY_FRAMING[framing],
            "layer": 19,
            "d_model": d,
            "n_rows": n,
            "metadata": base_metadata(),
        }
        out = root / mdir / unit_id / "L19.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(store, out)
        entries.append(
            {
                "unit_id": unit_id,
                "model": model,
                "model_dir": mdir,
                "framing": framing,
                "provenance": prov,
                "variant": variant,
                "slots": list(slots),
                "straddler_policy": {s: SLOT_STRADDLER_POLICY[s] for s in slots},
                "fit_pairs": [list(p) for p in FIT_PAIRS_BY_FRAMING[framing]],
                "primary_fit": PRIMARY_FIT_BY_FRAMING[framing],
                "rendered_path": f"{unit_id}.jsonl",
                "n_rows": n,
                "read_group_names": gnames,
                "grid_x_kinds": ["X_clean", "X_straddle"],
                "grid_y_kinds": ["Y_mean", "Y_end", "Y_boundary"],
            }
        )
    (root / "render_manifest.json").write_text(
        json.dumps({"metadata": base_metadata(), "synthetic": True, "units": entries}, indent=2),
        encoding="utf-8",
    )
    return root


def run_synthetic_smoke(args) -> int:
    """End-to-end battery on the synthetic tree + a Gate-1 NON-INERTNESS probe.

    Gate-1's PASS branch can only be exercised against the real published
    numbers (asserting a synthetic value against itself would be circular), so
    the smoke proves the gate is not inert: a deliberately wrong reference MUST
    raise.
    """
    import numpy as np

    root = args.store_root
    root.mkdir(parents=True, exist_ok=True)
    build_synthetic_store_tree(root)
    args.skip_gate1 = True
    args.skip_cross_role = True
    # The addendum-A split consumes the PARENT's real stores + rendered shards
    # (network + a real tokenizer); the synthetic tree has neither.
    args.skip_length_split = True
    args.null_draws = min(args.null_draws, 8)
    summary = run_battery(args)

    # Gate-1 non-inertness: wrong reference -> must FAIL LOUD.
    stores = {}
    mdir = "Qwen_Qwen2.5-7B"
    unit_id = f"{mdir}__chat__lmsys"
    stores[unit_id] = load_store(root / mdir / unit_id / "L19.pt")
    fake_dir = root / "fake_percell"
    fake_dir.mkdir(parents=True, exist_ok=True)
    for m in (mdir, "Qwen_Qwen2.5-7B-Instruct"):
        (fake_dir / f"heldout_r2_{m}_user_lmsys_chat.json").write_text(
            json.dumps(
                {
                    "n_rows": 180,
                    "layers": [14, 18, 19, 26],
                    "prefix": {"held_out_r2_per_layer": [0.0, 0.0, -99.0, 0.0]},
                }
            ),
            encoding="utf-8",
        )
    raised = False
    try:
        gate1_parent_parity(stores, fake_dir)
    except RuntimeError as exc:
        raised = "Gate-1" in str(exc)
    if not raised:
        raise RuntimeError("Gate-1 non-inertness probe FAILED: a wrong reference did not raise")
    print("[synthetic-smoke] Gate-1 non-inertness probe PASS (wrong reference raised)", flush=True)

    # Shape assertions on the produced battery.
    specs = _synthetic_specs()
    n_units = len(summary["per_unit_r2"])
    n_transfer = len(summary["provenance_transfer"])
    n_label = len(summary["story_label_effect"])
    floor_seen = any("floor_control" in fits for fits in summary["per_unit_r2"].values())
    print(
        f"[synthetic-smoke] units={n_units} provenance_transfers={n_transfer} "
        f"label_pairs={n_label} floor_control_present={floor_seen}",
        flush=True,
    )
    if n_units != len(specs) or n_transfer == 0 or n_label == 0 or not floor_seen:
        raise RuntimeError(
            f"synthetic smoke produced an incomplete battery (units={n_units} "
            f"expected={len(specs)})"
        )
    for u, fits in summary["per_unit_r2"].items():
        for name, r2 in fits.items():
            if not np.isfinite(r2):
                raise RuntimeError(f"non-finite r2 for {u}/{name}")

    # Per-FAMILY coverage: every family's fit-pair table AND grid must have been
    # exercised. A count-only check passes while a whole family is unfitted.
    from scripts.issue1689_user_slot_render import (
        GRID_X_KINDS,
        GRID_Y_KINDS,
        PRIMARY_FIT_BY_FRAMING,
        READ_GROUP_NAMES_BY_FRAMING,
    )

    mdir = "Qwen_Qwen2.5-7B"
    for framing, variant, prov in specs:
        frame = framing if variant == "base" else f"{framing}_{variant}"
        unit_id = f"{mdir}__{frame}__{prov}"
        fits = summary["per_unit_r2"].get(unit_id)
        if not fits:
            raise RuntimeError(f"synthetic smoke fitted no pairs for {framing} cell {unit_id}")
        primary = PRIMARY_FIT_BY_FRAMING[framing]
        if primary not in fits:
            raise RuntimeError(f"{unit_id}: primary fit {primary!r} missing from {sorted(fits)}")
        grid = summary["grid_r2"].get(unit_id) or {}
        want_groups = set(READ_GROUP_NAMES_BY_FRAMING[framing])
        if set(grid) != want_groups:
            raise RuntimeError(
                f"{unit_id}: grid groups {sorted(grid)} != declared {sorted(want_groups)}"
            )
        for gn, combos in grid.items():
            n_want = len(GRID_X_KINDS) * len(GRID_Y_KINDS)
            if len(combos) != n_want:
                raise RuntimeError(f"{unit_id}/{gn}: {len(combos)} grid combos, expected {n_want}")
    families = sorted({s[0] for s in specs})
    print(f"[synthetic-smoke] per-family coverage OK: {families}", flush=True)

    # The bridging reduction must be populated by the same run (its inputs are
    # the per-cell + grid results above).
    bridge = summary.get("bridge_comparisons") or {}
    if not bridge.get("target_summary_convention"):
        raise RuntimeError("synthetic smoke produced no addendum-D target-convention rows")
    print(
        "[synthetic-smoke] bridge_comparisons: "
        + json.dumps({k: len(v) for k, v in bridge.items() if isinstance(v, dict)}),
        flush=True,
    )
    print("[synthetic-smoke] OK", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Battery driver
# ---------------------------------------------------------------------------


def run_battery(args) -> dict:
    import numpy as np

    from scripts.issue1689_user_slot_render import MODELS, PARENT_REVISION, model_dir

    manifest_path = args.store_root / "render_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"render manifest missing beside the stores: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    entries = {e["unit_id"]: e for e in manifest["units"]}

    stores: dict[str, dict] = {}
    for unit_id, entry in entries.items():
        p = args.store_root / entry["model_dir"] / unit_id / "L19.pt"
        if not p.exists():
            if args.allow_missing:
                print(f"[fits] WARN missing store, skipping: {p}", flush=True)
                continue
            raise FileNotFoundError(f"store missing: {p}")
        stores[unit_id] = load_store(p)
    if not stores:
        raise RuntimeError("no stores loaded")
    print(f"[fits] loaded {len(stores)} unit stores", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "metadata": {
            "issue": 1689,
            "round": "user_slot_recapture",
            "n_folds": N_FOLDS,
            "seed": FIT_SEED,
            "null_draws": args.null_draws,
            "lambda_selection": "inner-group-cv",
            "pca_k_rule": "min(1024, floor(n_train/2))",
            "layer": 19,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
        "render_metadata": manifest.get("metadata"),
        "realized_defects_fixed": manifest.get("realized_defects_fixed"),
    }

    # --- addendum-A cell list (also feeds the pre-flight projection) ---------
    split_cells: list[tuple[str, str]] = []
    if not args.skip_length_split:
        split_cells = [
            (mdir, cond)
            for mdir in sorted({e["model_dir"] for e in entries.values()})
            for cond in LENGTH_SPLIT_CONDITIONS
        ]

    # --- compute-character pre-flight (MEASURED basis, then the wall fence) --
    if args.preflight:
        n_max = max(int(e["n_rows"]) for e in entries.values() if e["unit_id"] in stores)
        d_model = _store_d_model(next(iter(stores.values())))
        basis = measure_fold_basis(args.preflight_n or n_max, d_model, null_draws=args.null_draws)
        split_n_rows = [
            published_parent_r2(args.percell_dir, m, c)["n_rows_published"] for m, c in split_cells
        ]
        proj = project_battery_wall(
            [e for e in entries.values() if e["unit_id"] in stores],
            basis,
            null_draws=args.null_draws,
            length_split_n_rows=split_n_rows,
        )
        summary["compute_preflight"] = proj
        enforce_wall_fence(proj, args.max_wall_hours)

    # --- verification gates first (cheap, fail loud before any long fit) ----
    summary["truncation_equivalence"] = verify_truncation_equivalence()
    if not args.skip_gate1:
        summary["gate1_parent_parity"] = gate1_parent_parity(stores, args.percell_dir)

    # --- (1) per-cell ceilings ---------------------------------------------
    per_unit: dict[str, dict] = {}
    for unit_id, st in stores.items():
        entry = entries[unit_id]
        ids = np.asarray(st["conv_ids"], dtype=object)
        fits: dict[str, dict] = {}
        for x_slot, y_slot, name in [tuple(p) for p in entry["fit_pairs"]]:
            X = st["slots"][x_slot]
            Y = st["slots"][y_slot]
            res = fit_within(X, Y, ids, null_draws=args.null_draws)
            res["reduced_basis_companion"] = fit_within_reduced(
                X, Y, ids, null_draws=args.null_draws
            )
            res["x_slot"] = x_slot
            res["y_slot"] = y_slot
            res["is_floor_control"] = name == "floor_control"
            res["is_primary"] = name == entry["primary_fit"]
            res["straddler_excluded_frac"] = {
                s: float(np.asarray(st["seam_flags"][s]).mean()) for s in (x_slot, y_slot)
            }
            fits[name] = res
            print(
                f"[fits] {unit_id} {name}: r2={res['r2']:+.4f} "
                f"reduced={res['reduced_basis_companion']['r2']:+.4f} "
                f"idb={res['identity_bias_r2']}",
                flush=True,
            )
        # --- addendum E: the X x Y grid, one factorization per (X, fold) ----
        grids: dict[str, dict] = {}
        for gn in list(st.get("grid_group_names") or []):
            gslots = {
                kind: st["grid_slots"][f"{gn}__{kind}"]
                for kind in ("X_clean", "X_straddle", "Y_mean", "Y_end", "Y_boundary")
            }
            grids[gn] = fit_grid(
                gslots,
                ids,
                x_kinds=tuple(st.get("grid_x_kinds") or ("X_clean", "X_straddle")),
                y_kinds=tuple(st.get("grid_y_kinds") or ("Y_mean", "Y_end", "Y_boundary")),
                null_draws=args.null_draws,
            )
            best = {k: v["r2"] for k, v in grids[gn]["combos"].items()}
            print(
                f"[fits] {unit_id} grid[{gn}] factorizations="
                f"{grids[gn]['shared_factorizations']} (unshared would be "
                f"{grids[gn]['unshared_would_have_been']}) r2={best}",
                flush=True,
            )
        unit_out = {
            "unit_id": unit_id,
            "unit": {k: entry[k] for k in ("model", "framing", "provenance", "variant")},
            "n_rows": int(np.asarray(st["conv_ids"]).shape[0]),
            "dup_weight": int(np.asarray(st["dup_count"]).sum()),
            "u2_provenance_note": (
                "const_fallback — the parent corpus has no u2_lmsys; this arm is a CONSTANT-u2 "
                "control, NOT LMSYS-sourced"
                if entry["provenance"] == "lmsys"
                else entry["provenance"]
            ),
            "fits": fits,
            "grid": grids,
        }
        per_unit[unit_id] = unit_out
        with (args.out_dir / f"{unit_id}.json").open("w", encoding="utf-8") as fh:
            json.dump(unit_out, fh, indent=2)
    summary["per_unit_r2"] = {
        u: {n: f["r2"] for n, f in o["fits"].items()} for u, o in per_unit.items()
    }
    summary["grid_r2"] = {
        u: {gn: {c: r["r2"] for c, r in g["combos"].items()} for gn, g in o["grid"].items()}
        for u, o in per_unit.items()
        if o.get("grid")
    }

    # --- (2) provenance transfer, 3 pairs x 2 directions -------------------
    prov_pairs = list(PROVENANCE_TRANSFER_PAIRS)
    transfers: dict[str, dict] = {}
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for unit_id, e in entries.items():
        frame = e["framing"] if e["variant"] == "base" else f"{e['framing']}_{e['variant']}"
        groups.setdefault((e["model_dir"], frame), {})[e["provenance"]] = unit_id
    for (mdir, frame), by_prov in sorted(groups.items()):
        primary = None
        for a, b in prov_pairs:
            if a not in by_prov or b not in by_prov:
                continue
            ua, ub = by_prov[a], by_prov[b]
            if ua not in stores or ub not in stores:
                continue
            ea = entries[ua]
            primary = ea["primary_fit"]
            x_slot, y_slot, _ = next(tuple(p) for p in ea["fit_pairs"] if p[2] == primary)
            for src, tgt in ((ua, ub), (ub, ua)):
                ss, ts = stores[src], stores[tgt]
                key = f"{mdir}|{frame}|{entries[src]['provenance']}->{entries[tgt]['provenance']}"
                transfers[key] = fit_transfer(
                    ss["slots"][x_slot],
                    ss["slots"][y_slot],
                    np.asarray(ss["conv_ids"], dtype=object),
                    ts["slots"][x_slot],
                    ts["slots"][y_slot],
                    np.asarray(ts["conv_ids"], dtype=object),
                    null_draws=args.null_draws,
                )
                transfers[key]["slot_pair"] = [x_slot, y_slot]
                print(f"[fits] transfer {key}: r2={transfers[key]['r2']:+.4f}", flush=True)
        if primary:
            print(f"[fits] provenance transfers done for {mdir}/{frame}", flush=True)
    summary["provenance_transfer"] = transfers

    # --- (3) naive user <-> assistant-chat --------------------------------
    cross: dict[str, dict] = {}
    if not args.skip_cross_role:
        for mdir in sorted({e["model_dir"] for e in entries.values()}):
            unit_id = f"{mdir}__chat__lmsys"
            if unit_id not in stores:
                continue
            p = stage_parent_assistant_store(
                mdir, args.stage_root, args.revision or PARENT_REVISION
            )
            a = load_store_parent(p)
            u = stores[unit_id]
            ea = entries[unit_id]
            x_slot, y_slot, _ = next(tuple(q) for q in ea["fit_pairs"] if q[2] == ea["primary_fit"])
            pairs = (
                (
                    "user->assistant",
                    u["slots"][x_slot],
                    u["slots"][y_slot],
                    u["conv_ids"],
                    a["X_prefix"],
                    a["Y"],
                    a["conv_ids"],
                ),
                (
                    "assistant->user",
                    a["X_prefix"],
                    a["Y"],
                    a["conv_ids"],
                    u["slots"][x_slot],
                    u["slots"][y_slot],
                    u["conv_ids"],
                ),
            )
            for name, Xs, Ys, ids_s, Xt, Yt, ids_t in pairs:
                key = f"{mdir}|{name}"
                cross[key] = fit_transfer(
                    Xs,
                    Ys,
                    np.asarray(ids_s, dtype=object),
                    Xt,
                    Yt,
                    np.asarray(ids_t, dtype=object),
                    null_draws=args.null_draws,
                )
                cross[key]["note"] = (
                    "naive cross-role transfer: user primary slot pair vs the PARENT's "
                    "assistant_chat X_prefix->Y arm (no alignment, no Procrustes)"
                )
                print(f"[fits] cross-role {key}: r2={cross[key]['r2']:+.4f}", flush=True)
    summary["cross_role_transfer"] = cross

    # --- (4) Alex vs User-label story pairs -------------------------------
    label: dict[str, dict] = {}
    for mdir in sorted({e["model_dir"] for e in entries.values()}):
        for prov in ("lmsys", "haiku", "onpolicy"):
            ua = f"{mdir}__story_alex__{prov}"
            ub = f"{mdir}__story_user_label__{prov}"
            if ua not in stores or ub not in stores:
                continue
            ea = entries[ua]
            x_slot, y_slot, _ = next(tuple(q) for q in ea["fit_pairs"] if q[2] == ea["primary_fit"])
            for src, tgt in ((ua, ub), (ub, ua)):
                ss, ts = stores[src], stores[tgt]
                key = f"{mdir}|{prov}|{entries[src]['variant']}->{entries[tgt]['variant']}"
                label[key] = fit_transfer(
                    ss["slots"][x_slot],
                    ss["slots"][y_slot],
                    np.asarray(ss["conv_ids"], dtype=object),
                    ts["slots"][x_slot],
                    ts["slots"][y_slot],
                    np.asarray(ts["conv_ids"], dtype=object),
                    null_draws=args.null_draws,
                )
                print(f"[fits] label-effect {key}: r2={label[key]['r2']:+.4f}", flush=True)
    summary["story_label_effect"] = label

    # --- (5) addendum A: length-stratified refits on the PARENT stores -------
    lensplit: dict[str, dict] = {}
    for mdir, cond in split_cells:
        framing = "chat" if cond.endswith("_chat") else "naturalistic"
        model_name = next(m for m in MODELS if model_dir(m) == mdir)  # store dir -> HF model id
        tok = _get_tokenizer(model_name)
        shards = stage_parent_rendered_condition(
            cond, args.stage_root, args.revision or PARENT_REVISION
        )
        lengths, len_digest = context_token_lengths(shards, tok, framing)
        store_path = stage_parent_condition_store(
            mdir, cond, args.stage_root, args.revision or PARENT_REVISION
        )
        parent = load_store_parent(store_path)
        published = published_parent_r2(args.percell_dir, mdir, cond)
        key = f"{mdir}|{cond}"
        lensplit[key] = fit_length_stratified(
            parent,
            lengths,
            null_draws=args.null_draws,
            refit_full=args.length_split_refit_full,
            published=published,
        )
        lensplit[key]["published_reference"] = published
        lensplit[key]["length_variable"] = {
            "definition": "context token count (add_special_tokens=False)",
            "n_rendered_shards": len(shards),
            "seam_caveat": (
                "equals the capture's context_end token index up to at most one "
                "BPE-seam token; stratification variable only"
            ),
            **len_digest,
        }
        meta = lensplit[key]["split"]
        for arm_name, res in lensplit[key]["arms"].items():
            print(
                f"[fits] lensplit {key} {arm_name}: short={res['short']['r2']:+.4f} "
                f"long={res['long']['r2']:+.4f} d={res['delta_r2_long_minus_short']:+.4f} "
                f"published_full={res.get('r2_full_published')} "
                f"n_per_half={meta['n_rows_per_half']} "
                f"median_tokens={meta['overall_median_context_tokens']:.0f}",
                flush=True,
            )
    summary["length_stratified"] = lensplit

    # --- (6) addenda B/C/D: the bridging reduction (no new fits) --------------
    bridge = bridge_comparisons(per_unit, entries, args.percell_dir)
    summary["bridge_comparisons"] = bridge
    for key, row in bridge["prefix_ablation"].items():
        if row.get("skipped"):
            print(f"[fits] bridge-B {key}: SKIPPED — {row['skipped']}", flush=True)
            continue
        print(
            f"[fits] bridge-B {key}: single_turn={row['single_turn_r2']:+.4f} "
            f"two_turn={row['two_turn_r2']:+.4f} "
            f"d={row['delta_r2_two_turn_minus_single_turn']:+.4f}",
            flush=True,
        )
    for key, row in bridge["a1_provenance"].items():
        if row.get("skipped"):
            print(f"[fits] bridge-C {key}: SKIPPED — {row['skipped']}", flush=True)
            continue
        print(
            f"[fits] bridge-C {key}: onpolicy_a1={row['onpolicy_a1_r2']:+.4f} "
            f"offpolicy={row['offpolicy_r2']:+.4f} "
            f"d={row['delta_r2_onpolicy_minus_offpolicy']:+.4f}",
            flush=True,
        )
    for key, row in bridge["target_summary_convention"].items():
        by_y = row["r2_by_y"]
        d = row["delta_r2_mean_minus_end"]
        print(
            f"[fits] bridge-D {key} [{row['x_kind']}]: "
            + " ".join(f"{y}={v:+.4f}" for y, v in by_y.items() if v is not None)
            + (f" d(mean-end)={d:+.4f}" if d is not None else "")
            + (
                f" published_end={row['published_parent_end_convention']}"
                if row.get("published_parent_end_convention")
                else ""
            ),
            flush=True,
        )

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[fits] wrote {len(per_unit)} unit JSONs + summary.json -> {args.out_dir}", flush=True)
    return summary


def run_project_only(args) -> int:
    """MEASURE the fold basis + project the battery from the render manifest.

    Needs no stores — the manifest already carries every unit's row count and
    fit-pair list, so this is the pre-launch sizing check (run it BEFORE the
    capture finishes to decide device + null-draw budget).
    """
    from scripts.issue1689_common import D_MODEL

    for cand in (args.store_root / "render_manifest.json", args.rendered_dir / "manifest.json"):
        if cand.exists():
            with cand.open(encoding="utf-8") as fh:
                manifest = json.load(fh)
            break
    else:
        raise FileNotFoundError(
            f"no manifest at {args.store_root / 'render_manifest.json'} or "
            f"{args.rendered_dir / 'manifest.json'}"
        )
    entries = manifest["units"]
    n_max = max(int(e["n_rows"]) for e in entries)
    basis = measure_fold_basis(args.preflight_n or n_max, D_MODEL, null_draws=args.null_draws)
    split_n_rows: list[int] = []
    if not args.skip_length_split:
        split_n_rows = [
            published_parent_r2(args.percell_dir, mdir, cond)["n_rows_published"]
            for mdir in sorted({e["model_dir"] for e in entries})
            for cond in LENGTH_SPLIT_CONDITIONS
        ]
    proj = project_battery_wall(
        entries, basis, null_draws=args.null_draws, length_split_n_rows=split_n_rows
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "compute_preflight.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(proj, fh, indent=2)
    print(f"[preflight] wrote {out}", flush=True)
    enforce_wall_fence(proj, args.max_wall_hours)
    return 0


def load_store_parent(path: Path) -> dict:
    """Load a PARENT ``analysis_tensors`` L19 bundle (X_prefix / X_context / Y)."""
    import torch

    st = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("X_prefix", "X_context", "Y", "conv_ids"):
        if key not in st:
            raise RuntimeError(f"{path}: parent store missing key {key!r}")
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--store-root",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / "user_slot_recapture" / "store",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_1689" / "user_slot_recapture",
    )
    ap.add_argument(
        "--percell-dir",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_1689" / "percell",
    )
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / "user_slot_recapture" / "hf_dl",
    )
    ap.add_argument(
        "--rendered-dir",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / "user_slot_recapture" / "rendered",
        help="fallback manifest source for --project-only (before any store exists)",
    )
    ap.add_argument("--revision", default="")
    ap.add_argument("--null-draws", type=int, default=N_NULL_DRAWS)
    ap.add_argument(
        "--preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MEASURE one fold at the production shape, project the battery wall, and "
        "enforce --max-wall-hours before any long fit (default on)",
    )
    ap.add_argument(
        "--max-wall-hours",
        type=float,
        default=8.0,
        help="refuse to start a battery projected past this wall (see --preflight)",
    )
    ap.add_argument(
        "--preflight-n",
        type=int,
        default=0,
        help="row count to MEASURE the fold basis at (0 = the largest unit). Lower it on a "
        "loaded shared VM: the batched null term's peak RSS scales ~n^2 (measured 32 GB at "
        "n=3800 with 40 draws), and the projection scales the basis arithmetically either way.",
    )
    ap.add_argument(
        "--project-only",
        action="store_true",
        help="run the measured pre-flight projection and exit (no fits)",
    )
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--skip-gate1", action="store_true")
    ap.add_argument("--skip-cross-role", action="store_true")
    ap.add_argument(
        "--skip-length-split",
        action="store_true",
        help="skip addendum A (median context-token split refits on the PARENT "
        "assistant_chat / assistant_naturalistic L19 stores)",
    )
    ap.add_argument(
        "--length-split-refit-full",
        action="store_true",
        help="addendum A: ALSO refit the full set locally (default uses the "
        "PARENT's published L19 R2 as the full-set reference — same estimator)",
    )
    ap.add_argument(
        "--verify-truncation-equivalence",
        action="store_true",
        help="run ONLY the reduced-basis truncation identity check and exit",
    )
    ap.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="build a synthetic store tree with the REAL schema and run the FULL "
        "battery end-to-end + the Gate-1 non-inertness probe (reduced dims, "
        "production rank regime n_train < d)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import + exit (Axis-1 import-resolution leg)",
    )
    args = ap.parse_args()
    _apply_device_choice(args.device)

    if args.import_check:
        import numpy  # noqa: F401
        import torch  # noqa: F401

        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from huggingface_hub import HfApi  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401

        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            list_hf_files_under_path,
            stage_hub_file,
        )
        from scripts.issue825_fit_cells import (  # noqa: F401
            N_INNER_LAMBDA_FOLDS,
            _cv_folds,
            _null_ss_contrib,
            _prep_fold,
            _prep_inner_lambda,
            _ridge_predict_cached,
            heldout_r2_sweep,
        )
        from scripts.issue1689_common import HF_DATA_PREFIX  # noqa: F401
        from scripts.issue1689_user_slot_render import (  # noqa: F401
            MODELS,
            PARENT_REVISION,
            DATA_REPO,
            model_dir,
        )

        print("[fits] import-check OK", flush=True)
        return 0

    if args.verify_truncation_equivalence:
        verify_truncation_equivalence()
        return 0

    if args.project_only:
        return run_project_only(args)

    if args.synthetic_smoke:
        return run_synthetic_smoke(args)

    run_battery(args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
