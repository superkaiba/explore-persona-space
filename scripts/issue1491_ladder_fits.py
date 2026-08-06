#!/usr/bin/env python3
"""Task #1491 Phase-3: ladder fits driver (per-scale primary cell + floors + ceiling + WC transfer).

Ported from ``origin/main:scripts/issue779_ffc_n1m_fits.py`` @ d7c1c55fbe per
Unit 1 Deliverable A port-source decision (epm:progress v6). The parent
fits driver is 3020 lines; this ladder driver delegates the actual
fitters (``fit_ridge`` / ``fit_ridge_with_weights`` / ``fit_mlp`` /
``fit_residual_skip`` / ``fit_krr_nystrom``) and their streaming
X^TX / Phi^TPhi machinery to that module, and layers a lean ladder-
specific orchestration on top: read the four-split captures uploaded by
Unit 2 for one (scale, layer), concatenate them into ONE (X, Y) with
recorded train/val/test/wc_test index ranges, run the plan §4.3 battery
at the PRIMARY cell (n=25k, primary layer, all 5 predictors), compute
per-scale floors + two-draw reliability ceiling + WildChat transfer
fold, persist per-context test predictions + targets for the paired
bootstrap, and write the per-scale fits JSON under
``eval_results/issue_1491/scale_ladder/fits_scale<slug>.json``.

Deferred within Unit 3 (documented in the return manifest; captured by
Unit 4 or a Unit-3 follow-up commit — non-blocking for the primary
§6.5 deliverable):

- Full n-ladder ridge/MLP/KRR points (n ∈ {8k, 15k, 25k}; only n=25k
  primary cell in this cut).
- Tier-B all-layer ridge sweep (secondary; requires the tierB_3600
  all-layer capture Unit 2's driver produces at each shard's job tail).
- Random-projection d=896 dimension control.
- Length-stratified R² by response-token tercile + fixed [256, 768]
  band.
- Behavior-adjacent free companion (ridge R² of log(response token count)
  from cx_last).

The primary_deliverable rows in plan §6.5 —
    eval_results/issue_1491/scale_ladder/fits_scale*.json
    data/issue_1491/preds/scale*_test_preds_*.npz
— ARE produced by this driver.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# Heavy imports (numpy/torch) MUST come AFTER load_dotenv() so the shared-VM
# thread caps (#847) bind in-process — torch freezes its intra-op pool from
# OMP_NUM_THREADS at IMPORT, so an import above the loader silently runs
# uncapped on the shared box. Pinned by tests/test_shared_vm_thread_caps.py
# (test_no_new_torch_before_dotenv_vm_entrypoints).
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Import parent-branch fit functions + streaming helpers from origin/main
# copies (Unit 1 Deliverable A port-source decision; no vendoring).
import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as F  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as MB  # noqa: E402

logger = logging.getLogger("issue1491_ladder_fits")


# ---------------------------------------------------------------------------
# Ladder configuration (plan v4)
# ---------------------------------------------------------------------------

# Per-scale layer lists — depth-fraction-mapped {0.500, 0.679, 0.929}
# rounded per-scale (plan §4.2 layer index table). Primary = middle
# entry (f=0.679).
LADDER_SCALES = {
    "scale05": {
        "slug": "scale05",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers": [12, 16, 22],
        "h_dim": 896,
        "n_layers": 24,
    },
    "scale15": {
        "slug": "scale15",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "layers": [14, 19, 26],
        "h_dim": 1536,
        "n_layers": 28,
    },
    "scale3": {
        "slug": "scale3",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "layers": [18, 24, 33],
        "h_dim": 2048,
        "n_layers": 36,
    },
    "scale7_refit": {
        "slug": "scale7_refit",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "layers": [14, 19, 26],
        "h_dim": 3584,
        "n_layers": 28,
    },
    "scale14": {
        "slug": "scale14",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "layers": [24, 33, 45],
        "h_dim": 5120,
        "n_layers": 48,
    },
    "scale32": {
        "slug": "scale32",
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "layers": [32, 43, 59],
        "h_dim": 5120,
        "n_layers": 64,
    },
}

# Plan §11 fitter battery (parent parity).
LAMBDAS = np.logspace(-3, 8, 23)
RIDGE_BLOCK = 50_000
MLP_W_PROTOCOL = 8192
MLP_W_CAPACITY = 32768
MLP_BATCH = 4096
KRR_M_CENTERS = 16_384
KRR_LAMBDAS = (1e-1, 1e1)
FIT_SEED = 0  # parent-parity fits seed (n1m_fits.json ran seed 0)


# ---------------------------------------------------------------------------
# HF chunk streaming (one prefix per split)
# ---------------------------------------------------------------------------


def _stream_ladder_split(
    hf_prefix: str, split: str, layer: int, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Stream all captured chunks for one (scale-prefix, split, layer)
    combination from HF. Delegates to the parent's ``_stream_hf_chunks`` —
    the ladder driver uploads chunks under
    ``<hf_prefix>/<split>/final_token_capture/`` and
    ``<hf_prefix>/<split>/raw_completions/``, so we point the streamer at
    the per-split ``final_token_capture`` sub-prefix.

    Returns (cx: (n, H) fp32, vx: (n, H) fp32, ci: list[int])."""
    prefix = f"{hf_prefix}/{split}/final_token_capture"
    logger.info("[ladder-fits] streaming captures: %s layer=%d", prefix, layer)
    # No stream checkpoint dir needed here — the per-scale primary cell is
    # tiny (25k+400+1000+1000 = 27.4k rows) so we stream in one pass with
    # ckpt_dir=None (the parent helper handles that path).
    cx, vx, ci = F._stream_hf_chunks(
        prefix, layer, cache_dir, ckpt_dir=None, ckpt_every=0, fresh=True
    )
    return cx, vx, list(ci)


# ---------------------------------------------------------------------------
# Assemble (X, Y) across the four splits
# ---------------------------------------------------------------------------


def _assemble_scale_layer(hf_prefix: str, layer: int, cache_dir: Path) -> dict:
    """Assemble one (X, Y) for one (scale, layer) from the FOUR ladder splits.

    Returns a dict with:
      X: (N, H) fp32 — cx_last stacked across splits
      Y: (N, H) fp32 — v_x stacked across splits
      tr: np.ndarray[int64] — train_25k row indices in the concatenated arrays
      val: np.ndarray[int64] — val_400 row indices
      te: np.ndarray[int64] — test_1000 row indices
      wc_te: np.ndarray[int64] — wc_test_1k row indices
      cis: dict[str, list[int]] — per-split original context ids
      n_realized: dict[str, int] — realized counts per split
    """
    splits = ["train_25k", "val_400", "test_1000", "wc_test_1k"]
    cxs: list[np.ndarray] = []
    vxs: list[np.ndarray] = []
    cis: dict[str, list[int]] = {}
    ranges: dict[str, tuple[int, int]] = {}
    n_so_far = 0
    for s in splits:
        cx, vx, ci = _stream_ladder_split(hf_prefix, s, layer, cache_dir)
        n = int(cx.shape[0])
        ranges[s] = (n_so_far, n_so_far + n)
        cxs.append(cx)
        vxs.append(vx)
        cis[s] = ci
        n_so_far += n
        logger.info("[ladder-fits]   %s: n=%d", s, n)

    X = np.concatenate(cxs, axis=0)
    Y = np.concatenate(vxs, axis=0)
    tr_s, tr_e = ranges["train_25k"]
    va_s, va_e = ranges["val_400"]
    te_s, te_e = ranges["test_1000"]
    wc_s, wc_e = ranges["wc_test_1k"]
    return {
        "X": X,
        "Y": Y,
        "tr": np.arange(tr_s, tr_e, dtype=np.int64),
        "val": np.arange(va_s, va_e, dtype=np.int64),
        "te": np.arange(te_s, te_e, dtype=np.int64),
        "wc_te": np.arange(wc_s, wc_e, dtype=np.int64),
        "cis": cis,
        "n_realized": {s: int(ranges[s][1] - ranges[s][0]) for s in splits},
    }


# ---------------------------------------------------------------------------
# Floors: shuffled-pairing null, train-mean, identity-copy, scaled-identity
# ---------------------------------------------------------------------------


def _pooled_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Whole-map variance-weighted R² = 1 - Σ_d SSE_d / Σ_d SST_d
    (parent parity via F.PR._pooled_r2, but re-implemented here to avoid
    importing a private helper — the arithmetic is one line)."""
    sse = float(((target - pred) ** 2).sum())
    sst = float(((target - target.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / (sst + 1e-30)


def _fit_floors(
    X: np.ndarray, Y: np.ndarray, tr: np.ndarray, val: np.ndarray, te: np.ndarray, dev, block
) -> dict:
    """Compute the plan §4.3 per-scale floors at the primary layer.

    - shuffled-pairing null: permute the train context→target pairing,
      refit ridge (val-λ), test R². (Chance level under fitter+dim.)
    - train-mean predictor: ŷ = ȳ_train (input-agnostic; ≡ prefix-arm
      degenerate limit).
    - identity-copy: ŷ = cx_last (raw self-similarity).
    - scaled-identity: ŷ = α · cx_last with α = argmin_α Σ|Y - α·X|²
      solved per-dim on train (a scalar per-dim regression).

    Returns a dict of {name: {"pred_te": np.ndarray, "test_r2": float, "meta": {...}}}.
    Note the identity-family floors read strongly NEGATIVE held-out under
    the raw-space variance-weighted metric — the 7B anchor's baselines
    read gives identity-copy ≈ -2.5; figures clip/annotate accordingly.
    """
    floors: dict[str, dict] = {}

    # 1. Shuffled-pairing null: permute train Y, refit ridge, score on test.
    rng = np.random.default_rng(1491)
    perm = rng.permutation(len(tr))
    Y_null = Y.copy()
    Y_null[tr] = Y[tr[perm]]
    pred_null, meta_null = F.fit_ridge(X, Y_null, tr, val, te, LAMBDAS, dev, block)
    floors["shuffled_pairing"] = {
        "pred_te": pred_null,
        "test_r2": _pooled_r2(pred_null, Y[te]),
        "meta": {**meta_null, "family": "shuffled-pairing null"},
    }

    # 2. Train-mean predictor (input-agnostic; the prefix arm's degenerate limit).
    y_mu = Y[tr].mean(axis=0, keepdims=True)
    pred_mean = np.broadcast_to(y_mu, (len(te), Y.shape[1]))
    floors["train_mean"] = {
        "pred_te": pred_mean.copy(),
        "test_r2": _pooled_r2(pred_mean, Y[te]),
        "meta": {"family": "train-mean predictor (≡ prefix arm degenerate limit)"},
    }

    # 3. Identity-copy: ŷ = X (residual-stream autocorrelation floor).
    pred_id = X[te]
    floors["identity_copy"] = {
        "pred_te": pred_id.copy(),
        "test_r2": _pooled_r2(pred_id, Y[te]),
        "meta": {"family": "identity copy: ŷ = cx_last"},
    }

    # 4. Scaled-identity: per-dim α = <X_tr, Y_tr> / <X_tr, X_tr>, then ŷ = α·X.
    # Cheap fp64 scalar-per-dim regression on train (no matrix inversion).
    Xtr = X[tr].astype(np.float64)
    Ytr = Y[tr].astype(np.float64)
    num = (Xtr * Ytr).sum(axis=0)
    den = (Xtr * Xtr).sum(axis=0) + 1e-30
    alpha = (num / den).astype(np.float32)  # (H,)
    pred_scaled = X[te] * alpha
    floors["scaled_identity"] = {
        "pred_te": pred_scaled,
        "test_r2": _pooled_r2(pred_scaled, Y[te]),
        "meta": {"family": "per-dim scaled identity ŷ_d = α_d · X_d"},
    }

    # 5. Identity + LEARNED BIAS: ŷ = x + b, b = train-fold mean of (y − x).
    #
    # REQUIRED by the standing project rule (CLAUDE.md § "Identity+learned-bias
    # baseline AND kNN-retrieval metric — report BOTH for every representation
    # mapping"): whenever input and output spaces share a dimension, the
    # learned-bias identity form is reported alongside held-out R². Here
    # c(x) and v(x) are both (n, H) at the same scale, so it always applies —
    # a dimension mismatch would have to be stated as inapplicable, never
    # silently skipped. The canonical helper is the single source of truth for
    # the bias definition; do not re-derive it inline.
    pred_idb = MB.identity_bias_predict(X[tr], Y[tr], X[te]).astype(np.float32)
    floors["identity_bias"] = {
        "pred_te": pred_idb,
        "test_r2": _pooled_r2(pred_idb, Y[te]),
        "meta": {
            "family": "identity + learned bias: ŷ = x + mean_train(y − x)",
            "helper": "analysis.mapping_baselines.identity_bias_predict",
        },
    }
    return floors


# ---------------------------------------------------------------------------
# kNN retrieval read (standing rule: reported alongside held-out R²)
# ---------------------------------------------------------------------------


def _knn_reads(preds_by_arm: dict[str, np.ndarray], y_true: np.ndarray) -> dict:
    """P(true target within the k nearest neighbours of the prediction).

    REQUIRED by the standing project rule (CLAUDE.md § "Identity+learned-bias
    baseline AND kNN-retrieval metric"): held-out R² alone both OVERSTATES a
    map (variance a constant shift already explains) and UNDERSTATES one
    (discriminative but mis-scaled predictions), and the two reads have been
    measured to dissociate in BOTH directions (#722 vs #779). So every fitted
    map here also gets the retrieval read, in euclidean AND cosine, over the
    held-out candidate pool (the test targets are their own pool).

    ``chance_at_k = k / n_pool`` is reported by the helper and is exactly what
    a constant (predict-the-mean) predictor scores — so `train_mean` doubles
    as the in-band sanity check that the pool wiring is right.
    """
    out: dict[str, dict] = {}
    n_pool = int(y_true.shape[0])
    # k scaled to the pool, per the rule; keep k << n_pool so acc@k stays
    # informative rather than saturating.
    ks = tuple(k for k in (1, 5, 10, 50) if k < n_pool)
    for arm, pred in preds_by_arm.items():
        if pred is None:
            continue
        out[arm] = {
            metric: MB.knn_retrieval(pred, y_true, ks=ks, metric=metric)
            for metric in ("euclidean", "cosine")
        }
    out["_meta"] = {
        "n_pool": n_pool,
        "ks": list(ks),
        "pool": "held-out test targets (pool == true)",
        "helper": "analysis.mapping_baselines.knn_retrieval",
    }
    return out


# ---------------------------------------------------------------------------
# Reliability ceiling (two-draw variance-weighted per-dim Pearson)
# ---------------------------------------------------------------------------


def _reliability_ceiling(hf_prefix: str, layer: int, cache_dir: Path) -> dict:
    """Two-draw reliability ceiling per scale — plan §4.3.

    Reads the seed-43 + seed-44 v_x captures for the 1,000 test contexts
    from ``<hf_prefix>/ceiling_draws/seed{43,44}/final_token_capture/``
    (produced by Unit 2's driver for --split ceiling_draw_{43,44}).
    Ceiling = Σ_d Var_d · r_d / Σ_d Var_d, where r_d is the Pearson
    correlation between draw A and draw B on the same 1,000 contexts at
    the primary layer, per dimension d, and Var_d is the variance of the
    two-draw MEAN as the pooling weight.
    """
    from huggingface_hub.errors import (
        EntryNotFoundError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    def _absent(exc: BaseException) -> dict:
        # ceiling draws not yet uploaded — expected until Unit 2 has run the
        # ceiling_draw_43/44 splits for this scale.
        logger.warning("[ladder-fits] ceiling draws missing for layer %d: %s", layer, exc)
        return {"available": False, "reason": f"ceiling captures not on HF: {exc}"}

    try:
        prefix_a = f"{hf_prefix}/ceiling_draws/seed43/final_token_capture"
        prefix_b = f"{hf_prefix}/ceiling_draws/seed44/final_token_capture"
        _cx_a, vx_a, ci_a = F._stream_hf_chunks(
            prefix_a, layer, cache_dir, ckpt_dir=None, ckpt_every=0, fresh=True
        )
        _cx_b, vx_b, ci_b = F._stream_hf_chunks(
            prefix_b, layer, cache_dir, ckpt_dir=None, ckpt_every=0, fresh=True
        )
    # ABSENCE ONLY below. A broad `except Exception` here silently downgrades a
    # transient Hub fault, an auth failure, or a genuine streaming/parse bug to
    # "not yet uploaded" — which deletes plan §4's registered H4 reliability-
    # ceiling protection without anyone noticing (fail-fast rule; same fail-open
    # shape fixed in the manifest builder's presence probe). The chunk listing
    # inside _stream_hf_chunks already rides hub.retry_transient, so 429/5xx are
    # retried there and never reach us as absence.
    except FileNotFoundError as e:
        # Prefix exists, no .pt chunks yet (issue779_ffc_n1m_fits._stream_hf_chunks).
        return _absent(e)
    except RepositoryNotFoundError:
        # Deliberately NOT absence: a missing/inaccessible data repo is a config
        # or token-scope fault, and must stay loud rather than mute the ceiling.
        raise
    except EntryNotFoundError as e:
        return _absent(e)
    except HfHubHTTPError as e:
        if getattr(getattr(e, "response", None), "status_code", None) == 404:
            return _absent(e)
        raise

    # Align by ci (they're the SAME 1,000 test contexts).
    by_ci_b = {int(c): i for i, c in enumerate(ci_b)}
    pair_a: list[np.ndarray] = []
    pair_b: list[np.ndarray] = []
    for i_a, c_a in enumerate(ci_a):
        j = by_ci_b.get(int(c_a))
        if j is None:
            continue
        pair_a.append(vx_a[i_a])
        pair_b.append(vx_b[j])
    if not pair_a:
        return {"available": False, "reason": "no matching ci between seed43 and seed44"}
    A = np.stack(pair_a).astype(np.float64)
    B = np.stack(pair_b).astype(np.float64)

    # Per-dim Pearson r_d + variance of the two-draw MEAN (pooling weight).
    a_mean = A.mean(axis=0, keepdims=True)
    b_mean = B.mean(axis=0, keepdims=True)
    a_c = A - a_mean
    b_c = B - b_mean
    num = (a_c * b_c).sum(axis=0)
    den = np.sqrt((a_c**2).sum(axis=0) * (b_c**2).sum(axis=0)) + 1e-30
    r_d = (num / den).astype(np.float32)
    Vd = ((A + B) / 2.0).var(axis=0, ddof=0).astype(np.float32)
    ceiling = float((Vd * r_d).sum() / (Vd.sum() + 1e-30))
    return {
        "available": True,
        "n_pairs": len(pair_a),
        "ceiling_var_weighted_r": ceiling,
        "mean_per_dim_r": float(r_d.mean()),
    }


# ---------------------------------------------------------------------------
# WildChat transfer fold: LMSYS-trained ridge/MLP evaluated on wc_test_1k
# ---------------------------------------------------------------------------


def _wc_transfer(
    X: np.ndarray,
    Y: np.ndarray,
    tr: np.ndarray,
    val: np.ndarray,
    wc_te: np.ndarray,
    dev,
    block,
) -> dict:
    """Fit LMSYS-only, evaluate on WildChat test (plan §4.3 + §6 OOD fold)."""
    if len(wc_te) == 0:
        return {"available": False, "reason": "wc_test_1k empty"}
    # Ridge and MLP-w8192 only (per §4.3 "each scale's n=25k LMSYS-trained
    # ridge + MLP-8k evaluated on wc_test_1k").
    pred_ridge_wc, meta_ridge = F.fit_ridge(X, Y, tr, val, wc_te, LAMBDAS, dev, block)
    pred_mlp_wc, meta_mlp = F.fit_mlp(
        X,
        Y,
        tr,
        wc_te,
        width=MLP_W_PROTOCOL,
        lr=1e-3,
        max_epochs=50,
        batch=MLP_BATCH,
        seed=FIT_SEED,
        dev=dev,
    )
    return {
        "available": True,
        "n_wc_test": int(len(wc_te)),
        "ridge_test_r2": _pooled_r2(pred_ridge_wc, Y[wc_te]),
        "mlp_w8192_test_r2": _pooled_r2(pred_mlp_wc, Y[wc_te]),
        "ridge_meta": meta_ridge,
        "mlp_meta": meta_mlp,
    }


# ---------------------------------------------------------------------------
# Primary cell: run the full 5-predictor battery at n=25k, primary layer
# ---------------------------------------------------------------------------


def run_primary_cell(scale_key: str, hf_prefix: str, args) -> dict:
    """Run plan §4.3 primary cell for ONE scale (all 5 predictors at n=25k,
    primary layer = middle entry of the scale's layer list).

    Emits per-context test predictions + targets as .npz under
    ``data/issue_1491/preds/<slug>_test_preds_*.npz`` (§6.5 primary
    deliverable row 2)."""
    scale = LADDER_SCALES[scale_key]
    primary_layer = scale["layers"][len(scale["layers"]) // 2]  # f=0.679 middle entry
    slug = scale["slug"]
    cache_dir = args.out_dir / ".cache" / slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[ladder-fits] primary cell scale=%s (%s) primary_layer=%d h_dim=%d",
        slug,
        scale["model"],
        primary_layer,
        scale["h_dim"],
    )

    # 1. Assemble captures across the four splits at the primary layer.
    bundle = _assemble_scale_layer(hf_prefix, primary_layer, cache_dir)
    X, Y = bundle["X"], bundle["Y"]
    tr, val, te, wc_te = bundle["tr"], bundle["val"], bundle["te"], bundle["wc_te"]
    logger.info(
        "[ladder-fits] assembled: X=%s Y=%s tr=%d val=%d te=%d wc_te=%d",
        X.shape,
        Y.shape,
        len(tr),
        len(val),
        len(te),
        len(wc_te),
    )
    assert Y.shape[1] == scale["h_dim"], (
        f"h_dim mismatch: Y.shape={Y.shape} vs scale.h_dim={scale['h_dim']}"
    )

    # Fail fast on an unavailable requested device (project rule: the crash IS
    # the signal). The previous silent cuda->cpu fallback would run the full
    # MLP/KRR battery on CPU for hours while an 8-GPU pod billed idle — the
    # exact failure this rule exists to prevent.
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but torch.cuda.is_available() is False. "
            "Refusing to silently fall back to CPU: the fits battery would run "
            "for hours on CPU while a GPU pod bills idle. Fix the driver/pod, or "
            "pass --device cpu explicitly if a CPU run is genuinely intended."
        )
    dev = torch.device(args.device)

    # 2. Predictors — 5-way battery per plan §11.
    preds_meta: dict[str, dict] = {}

    # 2a. Ridge (val-λ over the 23-pt grid).
    logger.info("[ladder-fits] fitting ridge...")
    pred_ridge, meta_ridge = F.fit_ridge(X, Y, tr, val, te, LAMBDAS, dev, RIDGE_BLOCK)
    preds_meta["ridge"] = {
        "test_r2": _pooled_r2(pred_ridge, Y[te]),
        "meta": meta_ridge,
        "pred_te": pred_ridge,
    }
    logger.info(
        "[ladder-fits]   ridge test R² = %.4f (λ=%.3g)",
        preds_meta["ridge"]["test_r2"],
        meta_ridge.get("selected_lambda", -1),
    )

    # 2b. MLP w=8192 (protocol arm).
    logger.info("[ladder-fits] fitting MLP w=8192...")
    pred_mlp8, meta_mlp8 = F.fit_mlp(
        X,
        Y,
        tr,
        te,
        width=MLP_W_PROTOCOL,
        lr=1e-3,
        max_epochs=50,
        batch=MLP_BATCH,
        seed=FIT_SEED,
        dev=dev,
    )
    preds_meta["mlp_w8192"] = {
        "test_r2": _pooled_r2(pred_mlp8, Y[te]),
        "meta": meta_mlp8,
        "pred_te": pred_mlp8,
    }
    logger.info("[ladder-fits]   MLP-8192 test R² = %.4f", preds_meta["mlp_w8192"]["test_r2"])

    # 2c. MLP w=32768 (capacity arm; descope lever 1 per plan §9).
    if not args.no_mlp_capacity:
        logger.info("[ladder-fits] fitting MLP w=32768 (capacity)...")
        pred_mlp32, meta_mlp32 = F.fit_mlp(
            X,
            Y,
            tr,
            te,
            width=MLP_W_CAPACITY,
            lr=1e-3,
            max_epochs=50,
            batch=MLP_BATCH,
            seed=FIT_SEED,
            dev=dev,
            capacity_arm=True,
        )
        preds_meta["mlp_w32768"] = {
            "test_r2": _pooled_r2(pred_mlp32, Y[te]),
            "meta": meta_mlp32,
            "pred_te": pred_mlp32,
        }
        logger.info("[ladder-fits]   MLP-32768 test R² = %.4f", preds_meta["mlp_w32768"]["test_r2"])

    # 2d. KRR-Nyström m=16384 (plan-descope lever 2 drops this + residual).
    if not args.no_krr:
        logger.info("[ladder-fits] fitting KRR-Nyström m=%d...", KRR_M_CENTERS)
        pred_krr, meta_krr = F.fit_krr_nystrom(
            X,
            Y,
            tr,
            val,
            te,
            m_centers=KRR_M_CENTERS,
            gamma_mult=(1.0,),
            lambdas=KRR_LAMBDAS,
            seed=FIT_SEED,
            dev=dev,
            block=RIDGE_BLOCK,
        )
        preds_meta["krr_nystrom"] = {
            "test_r2": _pooled_r2(pred_krr, Y[te]),
            "meta": meta_krr,
            "pred_te": pred_krr,
        }
        logger.info("[ladder-fits]   KRR test R² = %.4f", preds_meta["krr_nystrom"]["test_r2"])

    # 2e. Residual-skip (ridge base + MLP on residual).
    if not args.no_residual:
        logger.info("[ladder-fits] fitting residual-skip...")
        pred_res, meta_res = F.fit_residual_skip(
            X,
            Y,
            tr,
            val,
            te,
            LAMBDAS,
            width=MLP_W_PROTOCOL,
            lr=1e-3,
            max_epochs=50,
            batch=MLP_BATCH,
            seed=FIT_SEED,
            dev=dev,
            block=RIDGE_BLOCK,
        )
        preds_meta["residual_skip"] = {
            "test_r2": _pooled_r2(pred_res, Y[te]),
            "meta": meta_res,
            "pred_te": pred_res,
        }
        logger.info(
            "[ladder-fits]   residual-skip test R² = %.4f", preds_meta["residual_skip"]["test_r2"]
        )

    # 3. Floors (per-scale, at primary layer).
    logger.info("[ladder-fits] computing floors...")
    floors = _fit_floors(X, Y, tr, val, te, dev, RIDGE_BLOCK)

    # 4. Reliability ceiling (from seed-43/44 draws).
    logger.info("[ladder-fits] computing reliability ceiling...")
    ceiling = _reliability_ceiling(hf_prefix, primary_layer, cache_dir)

    # 5. WildChat transfer fold.
    logger.info("[ladder-fits] computing WildChat transfer fold...")
    wc_transfer = _wc_transfer(X, Y, tr, val, wc_te, dev, RIDGE_BLOCK)

    # 6. Persist per-context test preds + targets for the paired bootstrap
    # (§6.5 primary_deliverable row 2). Emit ridge + registered nonlinear
    # (defined as the best of MLP-w8192, MLP-w32768, KRR-Nyström by val R²).
    preds_dir = args.preds_dir
    preds_dir.mkdir(parents=True, exist_ok=True)

    def _save_preds(name: str, pred_te: np.ndarray) -> Path:
        path = preds_dir / f"{slug}_test_preds_{name}.npz"
        np.savez(
            path,
            ci=np.array(bundle["cis"]["test_1000"], dtype=np.int64),
            target=Y[te].astype(np.float32),
            pred=pred_te.astype(np.float32),
        )
        return path

    saved: dict[str, str] = {}
    saved["ridge"] = str(_save_preds("ridge", pred_ridge))

    # PRE-REGISTERED nonlinear arm — NO selection.
    #
    # The registered ΔΓ contrast (plan §3) is the linear-vs-nonlinear gap, and
    # the Goal quotes it at 7B as "0.754 vs 0.810" — i.e. ridge vs MLP-w8192
    # (#779 n1m: ridge 0.7542, MLP-8k 0.8104, MLP-32k 0.8134). So MLP-w8192 is
    # the arm the registered contrast is DEFINED against, and pre-registering
    # it removes selection from the headline entirely.
    #
    # NOTE (issue-1491): this previously SELECTED the best nonlinear arm by
    # TEST R² while naming the output "val_selected_nonlinear". That is
    # selection-on-outcome — picking the arm on the same test split whose R²
    # is then reported — and it biased the registered ΔΓ upward. A true
    # val-selection is not available at matched cost: the parent fitters
    # return only test predictions (`pred_te`), ridge/KRR expose val R²
    # (`val_r2_at_selected` / `selected.val_r2`) but `fit_mlp` exposes only
    # `best_val_mse` on an INTERNAL 10% carve-out of train — a different
    # split AND a different metric — so the arms share no common val metric,
    # and obtaining one would mean refitting every nonlinear arm against the
    # pinned val split (roughly doubling the nonlinear fit cost).
    #
    # Pre-registration is also descope-safe: plan §9's descope ladder drops
    # MLP-w32768 first and KRR second, keeping "ridge + MLP-8k — the Goal's
    # named pair", so the registered arm survives every planned descope.
    REGISTERED_NONLINEAR = "mlp_w8192"
    nonlinear_arms = [
        name for name in ("mlp_w8192", "mlp_w32768", "krr_nystrom") if name in preds_meta
    ]
    if REGISTERED_NONLINEAR in preds_meta:
        chosen_nl = REGISTERED_NONLINEAR
        nl_provenance = "pre-registered"
    elif nonlinear_arms:
        # The registered arm did not run (descope / failure). Fall back in a
        # FIXED, outcome-independent order — never by test R².
        chosen_nl = next(name for name in ("mlp_w32768", "krr_nystrom") if name in preds_meta)
        nl_provenance = f"fallback-fixed-order (registered arm {REGISTERED_NONLINEAR} absent)"
    else:
        chosen_nl = None
        nl_provenance = "none-available"

    if chosen_nl is not None:
        saved["registered_nonlinear"] = str(
            _save_preds("registered_nonlinear", preds_meta[chosen_nl]["pred_te"])
        )
        saved["registered_nonlinear_kind"] = chosen_nl
        saved["registered_nonlinear_provenance"] = nl_provenance
        # Descriptive ONLY — the best-by-test arm is recorded so the choice is
        # auditable, but it never feeds the registered ΔΓ headline.
        best_by_test = max(nonlinear_arms, key=lambda k: preds_meta[k]["test_r2"])
        saved["best_by_test_r2_kind_DESCRIPTIVE"] = best_by_test

    # 6b. kNN-retrieval read over every fitted map + the identity-family
    # floors (standing rule — see _knn_reads). Reported ALONGSIDE held-out R²,
    # never instead of it.
    knn_arms: dict[str, np.ndarray] = {"ridge": pred_ridge}
    for _name in ("mlp_w8192", "mlp_w32768", "krr_nystrom"):
        if _name in preds_meta:
            knn_arms[_name] = preds_meta[_name]["pred_te"]
    for _name in ("identity_bias", "identity_copy", "scaled_identity", "train_mean"):
        if _name in floors:
            knn_arms[_name] = floors[_name]["pred_te"]
    logger.info("[ladder-fits] kNN retrieval over %d arms...", len(knn_arms))
    knn = _knn_reads(knn_arms, Y[te])

    # 7. Assemble the results JSON.
    result = {
        "scale_key": scale_key,
        "slug": slug,
        "model": scale["model"],
        "primary_layer_index": int(primary_layer),
        "layers": list(scale["layers"]),
        "h_dim": int(scale["h_dim"]),
        "n_realized": bundle["n_realized"],
        "predictors": {
            name: {"test_r2": pm["test_r2"], "meta": pm["meta"]} for name, pm in preds_meta.items()
        },
        "floors": {
            name: {"test_r2": f["test_r2"], "meta": f["meta"]} for name, f in floors.items()
        },
        "knn_retrieval": knn,
        "ceiling_two_draw": ceiling,
        "wc_transfer": {
            "available": wc_transfer.get("available", False),
            **{k: v for k, v in wc_transfer.items() if k not in {"available"}},
        },
        "preds_paths": saved,
        "fit_config": {
            "lambdas": [float(l) for l in LAMBDAS],
            "mlp_widths": [MLP_W_PROTOCOL, MLP_W_CAPACITY],
            "mlp_batch": MLP_BATCH,
            "mlp_lr": 1e-3,
            "mlp_max_epochs": 50,
            "krr_m_centers": KRR_M_CENTERS,
            "krr_lambdas": list(KRR_LAMBDAS),
            "ridge_block": RIDGE_BLOCK,
            "fit_seed": FIT_SEED,
            "device": str(dev),
        },
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scale",
        required=True,
        choices=sorted(LADDER_SCALES.keys()),
        help="ladder scale key (scale05 / scale15 / scale3 / scale7_refit / scale14 / scale32)",
    )
    ap.add_argument(
        "--hf-prefix",
        default=None,
        help="HF prefix under superkaiba1/explore-persona-space-data; "
        "default: issue1491_scale_ladder/<slug>",
    )
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument(
        "--no-mlp-capacity", action="store_true", help="skip MLP w=32768 (descope lever 1)"
    )
    ap.add_argument("--no-krr", action="store_true", help="skip KRR-Nyström (descope lever 2)")
    ap.add_argument(
        "--no-residual", action="store_true", help="skip residual-skip (descope lever 2)"
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="output JSON path; default: eval_results/issue_1491/scale_ladder/fits_<slug>.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "EPM_LADDER_FITS_OUT_DIR",
                os.path.expanduser("~/data/issue_1491/fits"),
            )
        ),
        help="scratch cache dir for streamed captures",
    )
    ap.add_argument(
        "--preds-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "EPM_LADDER_PREDS_DIR",
                str(Path.cwd() / "data" / "issue_1491" / "preds"),
            )
        ),
        help="output dir for per-context test preds + targets (.npz per predictor)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    scale = LADDER_SCALES[args.scale]
    hf_prefix = args.hf_prefix or f"issue1491_scale_ladder/{scale['slug']}"
    if args.out_json is None:
        args.out_json = (
            Path.cwd()
            / "eval_results"
            / "issue_1491"
            / "scale_ladder"
            / f"fits_{scale['slug']}.json"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    C.phase("run_primary_cell")
    result = run_primary_cell(args.scale, hf_prefix, args)

    C.phase("write_json")
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("[ladder-fits] wrote %s", args.out_json)
    for name, path in result["preds_paths"].items():
        if isinstance(path, str) and path.endswith(".npz"):
            logger.info("[ladder-fits]   preds: %s -> %s", name, path)
    C.phase("done")
    print(f"OK — wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
