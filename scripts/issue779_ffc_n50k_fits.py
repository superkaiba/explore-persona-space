#!/usr/bin/env python3
"""Issue #779 inline follow-up (``fitter-fair-comparison-n50k``): fits at n_train=50,000.

Extends the n10k fits path (``issue779_perdirection_per_predictor.py --corpus-mode
n10k`` / ``issue779_fitter_fair_comparison.py`` D1) to n_train ~= 50,000.

PLAN-B (DEFAULT, scope correction 2026-07-14): the n10k round's 6,500 captures live
only on the stopped pod-779 volume and are presumed unavailable, so the training
corpus is orig-train (3,600 of the ORIGINAL round's pass_b) + 46,600 NEW contexts
= pool 50,200, deterministically trimmed to train=50,000 exactly. The
``--include-n10k-captures`` flag (plan-A) instead pools orig-train (3,600) + n10k
(6,500) + all 46,600 new and trims to 50,000 (needs the n10k bundle on HF; FAILS
LOUD at startup if absent), for use if that volume is rescued later.
Either way ``n`` is the scaling variable vs the n10k run: the SAME target
(``v(x)`` mean-response profile), the SAME map INPUT (``cx_last`` last prompt
token), the SAME metric (variance-weighted held-out R2, pooled test-own-mean),
and — crucially — a BYTE-IDENTICAL val/test: val (400) + test (1000) are drawn
from the pass_b half by the EXACT ``fixed_split(5000, 3600, 400, 1000, 42)``, and
their ``sha256`` index digests are hard-asserted equal to the ORIGINAL round's
pinned shas (recomputed from the committed ``fitter-fair-comparison/
fair_comparison.json`` split params + the ORIG_VAL/TEST_SHA256 constants). New
contexts enter TRAIN only. (Realized train under plan-B is 50,000, sampled from a
pool of 50,200 = 3600 orig-train + 46600 new; val+test are carved from the
original 5000.)

Four predictors, extended to the n>>H=3584 regime:

  * ridge   — PRIMAL ridge (solve in the H-dim feature space, ONE eigh of the
              (H,H) X^T X, batched over lambda), val-lambda-selected over the
              extended ``LAMBDAS_N50K`` grid. The n10k path used the DUAL / Gram
              (ntr,ntr) eigh; at ntr~=50,000 the dual is a 50k^2 eigh (~10 GB fp32
              / ~20 GB fp64) while the primal is a 3584^2 solve — the primal is
              the right regime for n_train >> H.
  * krr     — EXACT RBF kernel ridge (NO Nystrom): the full (ntr,ntr) fp32 kernel
              (~10 GB at ntr~=50,000) assembled in row-blocks (chunked), solved by
              a single-GPU Cholesky per (gamma, lambda) over ALL targets at once.
              (gamma, lambda) val-selected. The n10k path used Nystrom (1024
              landmarks); at n~=50,000 the exact kernel is affordable on one H100.
  * mlp     — full-dim MLP, width 8192, the prior-round val-selected recipe REUSED
              (width+lr read from the committed ORIGINAL-round ``fair_comparison.json``;
              noted in fit_meta). Reuses ``F.run_mlp_battery`` / ``F.MLPGroup``.
  * residual_skip — PRIMAL ridge base + MLP on the residual (strictly nests the
              linear map). Reuses ``F.run_mlp_battery``.

Output (``eval_results/issue_779/fitter-fair-comparison-n50k/n50k_fits.json``):
per-predictor ``whole_map_r2`` + mean cosine + 1000-resample bootstrap 95% CI +
``fit_meta``, the split (with the pinned + realized val/test/train shas and the
byte-identical flag), the layer, and reproducibility metadata (git_commit). Same
JSON shape as the n10k fits files. Checkpoint-per-predictor; ``--resume`` skips
completed predictors (guarded on the layer AND the plan-A/plan-B corpus_mode so a
cross-layer / cross-mode resume never mixes rows).

The n50k capture (~306 GB) is NOT materialized locally: cx_last + v_x at the
chosen layer are STREAM-REDUCED from the HF capture chunks (download one chunk ->
slice the layer -> free), so peak footprint is ~one chunk. ``--n50k-capture-dir``
uses a local staged dir instead of streaming.

0-GPU by nature for ridge (``--device cpu`` default); the exact KRR + MLP want
``--device cuda`` on a pod (the 50k^2 kernel Cholesky is a single-GPU op). Fail
loud; NaN reported, never coerced. Refusal-safety: no context/rollout TEXT is
ever printed or logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE numpy/torch import (BLAS/torch pools freeze
# at import time on the shared VM).
load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue779_ffc_n50k_fits")

PREDICTORS = ("ridge", "krr", "mlp", "residual_skip")
PREDICTOR_LABEL = {
    "ridge": "primal ridge (linear)",
    "krr": "exact RBF KRR",
    "mlp": "full-dim MLP (w=8192)",
    "residual_skip": "residual-skip (primal ridge + MLP)",
}

# Extended ridge grid for n=50,000 (LAMBDAS_N10K = logspace(-3,6,19) = 2/decade
# over [1e-3,1e6]; n50k widens the top by one decade — more regularization
# headroom as n_train grows — keeping the 2/decade spacing).
LAMBDAS_N50K = np.logspace(-3, 7, 21)

# Exact-KRR selection grid (modest by construction: each (gamma,lambda) is an
# O(ntr^3) Cholesky at ntr=50,000, so keep it small). median-heuristic gamma
# multipliers + a short lambda grid; both --overridable.
KRR_GAMMA_MULT = (1.0,)
KRR_LAMBDAS = (1e-1, 1e1)
KRR_KERNEL_BLOCK = 4096  # row-block for chunked (ntr,ntr) kernel assembly

# Combined-corpus anchors (n is the only variable; plan-B scope 2026-07-14).
N_PASS_B = F.N_PASS_B  # 5000  (the original round's pass_b corpus)
N_VAL = 400
N_TEST = 1000
SPLIT_SEED = F.SPLIT_SEED  # 42
N50K_TRAIN = 50000  # target train; plan-B pool 3600 orig-train + 46600 new = 50200 trims to 50000
N_N50K_NEW = 46600  # round-3 new contexts (plan-B correction 2026-07-14: 45000->46600 so the pool
# clears the 50,000 target and trims to it exactly; was 45000, was 40000 at first draft)

# Pinned val/test sha256s = digests of the ORIGINAL round's fixed_split(5000,3600,400,1000,42)
# index arrays (< 5000, the pass_b half). Recomputed from the committed original-round
# fair_comparison.json split params (n_train/n_val/n_test/seed) and cross-checked against these
# constants, so a future split-param drift fails loud. These are byte-identical to the n10k
# val/test by construction (n10k inherited the same fixed_split).
ORIG_VAL_SHA256 = "2e307fb2d1b74c82752d9460d131a3c1949860e9f0eefe6a82d15cee9f1e0613"
ORIG_TEST_SHA256 = "b9377786b24bc9c1c360303fdb8fac86c0097d264479de1dca3c23dd1047d31d"

DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_779" / "fitter-fair-comparison-n50k"
DEFAULT_ORIG_DIR = PROJECT_ROOT / "eval_results" / "issue_779" / "fitter-fair-comparison"
HF_N50K_PREFIX = "issue779_monitoring/fitter-fair-comparison-n50k/final_token_capture"
HF_N10K_BUNDLE = "issue779_monitoring/fitter-fair-comparison-n10k/new_context_vectors.pt"


# ── pinned val/test split shas (from the ORIGINAL round's fair_comparison.json split) ─────


def _pinned_original_shas(orig_dir: Path) -> dict:
    """The ORIGINAL round's byte-identical val/test sha256 index digests.

    Reads the committed original-round ``fair_comparison.json`` split PARAMS
    (n_train/n_val/n_test/seed), recomputes ``fixed_split(N_PASS_B, n_train,
    n_val, n_test, seed)``, and returns the val/test index shas — cross-checked
    against the ORIG_VAL/TEST_SHA256 module constants so a split-param drift or a
    stale aggregate fails loud. Falls back to the constants if the aggregate is
    absent. This is the plan-B pin source (the ORIGINAL round, NOT the pod-side
    n10k split)."""
    agg = orig_dir / "fair_comparison.json"
    source = "module constants (fixed_split(5000,3600,400,1000,42))"
    val_sha, test_sha = ORIG_VAL_SHA256, ORIG_TEST_SHA256
    if agg.exists():
        split = json.loads(agg.read_text()).get("split", {})
        n_tr = int(split.get("n_train", N_PASS_B - N_VAL - N_TEST))
        n_v = int(split.get("n_val", N_VAL))
        n_te = int(split.get("n_test", N_TEST))
        seed = int(split.get("seed", SPLIT_SEED))
        assert n_v == N_VAL and n_te == N_TEST and seed == SPLIT_SEED, (
            f"original fair_comparison.json split params ({n_v}/{n_te}/{seed}) differ from "
            f"the byte-identical pins ({N_VAL}/{N_TEST}/{SPLIT_SEED})"
        )
        _r1, val, test = F.fixed_split(N_PASS_B, n_tr, n_v, n_te, seed)
        rv, rt = F._sha_ids(val), F._sha_ids(test)
        assert rv == ORIG_VAL_SHA256 and rt == ORIG_TEST_SHA256, (
            f"recomputed original val/test shas ({rv[:12]}/{rt[:12]}) differ from the pinned "
            f"constants ({ORIG_VAL_SHA256[:12]}/{ORIG_TEST_SHA256[:12]}) — split drift"
        )
        val_sha, test_sha = rv, rt
        source = f"{agg} split params -> fixed_split, cross-checked vs constants"
    return {"val_sha256": val_sha, "test_sha256": test_sha, "source": source}


# ── byte-identical split builder ────────────────────────────────────────────────


def build_n50k_split(
    n50k_used: int, n10k_kept: int | None, pinned: dict, *, n_train: int, seed: int
):
    """(train, val, test, diag) for the combined corpus.

    val/test = the ORIGINAL round's ``fixed_split(5000, 3600, 400, 1000, 42)`` over
    the first ``N_PASS_B`` ids (< N_PASS_B, the SAME pass_b rows => BYTE-IDENTICAL to
    the original round). Hard-asserts the val/test shas equal the PINNED original shas.

    Combined-corpus positions + train pool:
      * plan-B (n10k_kept is None): pass_b [0, N_PASS_B) + n50k [N_PASS_B, N_PASS_B+n50k_used).
        train pool = orig-train (3600) + all n50k ids.
      * plan-A (n10k_kept given): pass_b [0, N_PASS_B) + n10k [N_PASS_B, N_PASS_B+n10k_kept)
        + n50k [N_PASS_B+n10k_kept, ...). train pool = orig-train (3600) + n10k ids + n50k ids.
    Deterministically sampled to ``n_train`` (seed): plan-B pool = 3600 + 46600 = 50200 > 50000
    so it trims to exactly 50000; the below-pool clamp WARNING is a defensive guard that should
    NOT fire in production (only if the n50k capture landed short of 46,400 kept rows)."""
    r1_train, val, test = F.fixed_split(N_PASS_B, N_PASS_B - N_VAL - N_TEST, N_VAL, N_TEST, seed)
    val_sha, test_sha = F._sha_ids(val), F._sha_ids(test)
    assert val_sha == pinned["val_sha256"], (
        f"n50k val sha {val_sha} != pinned original val sha {pinned['val_sha256']} — "
        "val/test are NOT byte-identical to the original round"
    )
    assert test_sha == pinned["test_sha256"], (
        f"n50k test sha {test_sha} != pinned original test sha {pinned['test_sha256']}"
    )
    include_n10k = n10k_kept is not None
    if include_n10k:
        n10k_ids = np.arange(N_PASS_B, N_PASS_B + n10k_kept)
        n50k_ids = np.arange(N_PASS_B + n10k_kept, N_PASS_B + n10k_kept + n50k_used)
        pool = np.concatenate([r1_train, n10k_ids, n50k_ids])
        n_contexts = int(N_PASS_B + n10k_kept + n50k_used)
    else:
        n10k_ids = np.array([], dtype=np.int64)
        n50k_ids = np.arange(N_PASS_B, N_PASS_B + n50k_used)
        pool = np.concatenate([r1_train, n50k_ids])
        n_contexts = int(N_PASS_B + n50k_used)
    n_target = min(n_train, len(pool))
    if n_target < n_train:
        logger.warning(
            "train pool has only %d ids (%d orig-train + %d n10k + %d n50k) < target %d; "
            "using all %d — the n50k capture landed SHORT of the target; check the capture logs",
            len(pool),
            len(r1_train),
            len(n10k_ids),
            len(n50k_ids),
            n_train,
            n_target,
        )
    sel = np.random.default_rng(seed).choice(len(pool), size=n_target, replace=False)
    train = np.sort(pool[sel])
    assert set(val).isdisjoint(train) and set(test).isdisjoint(train), "train overlaps val/test"
    assert (val < N_PASS_B).all() and (test < N_PASS_B).all(), "val/test must index the pass_b half"
    diag = {
        "mode": "plan-A (n10k-captures included)" if include_n10k else "plan-B (n50k-only)",
        "include_n10k_captures": bool(include_n10k),
        "n_contexts": n_contexts,
        "n_pass_b": int(N_PASS_B),
        "n10k_kept": int(n10k_kept) if include_n10k else 0,
        "n50k_used": int(n50k_used),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_train_target": int(n_train),
        "pool_size": len(pool),
        "orig_train_ids": len(r1_train),
        "seed": int(seed),
        "val_sha256": val_sha,
        "test_sha256": test_sha,
        "train_sha256": F._sha_ids(train),
        "val_test_byte_identical_original": True,
        "pinned_val_sha256": pinned["val_sha256"],
        "pinned_test_sha256": pinned["test_sha256"],
        "pinned_source": pinned["source"],
        "note": (
            "val/test = fixed_split(5000,3600,400,1000,42) indices <5000 = the same pass_b rows as "
            "the original round (val/test_sha256 asserted == pinned original). "
            + (
                "plan-A: train pool = orig-train (3600) + all n10k new (6500) + all n50k new, "
                "sampled to n_train_target."
                if include_n10k
                else "plan-B (DEFAULT): train pool = orig-train (3600) + all 46600 n50k new, "
                "sampled to n_train_target. n10k captures NOT used (presumed unavailable)."
            )
        ),
    }
    return train, val, test, diag


# ── layer-slice data assembly (stream-reduced n50k capture) ─────────────────────


def _slice_layer(bundle, field: str, layer: int) -> np.ndarray:
    col = list(bundle["layers"]).index(layer)
    return bundle[field][:, col, :].to(torch.float32).numpy()


def _load_n10k_bundle(path: Path) -> object:
    """The n10k new bundle (~6500 kept rows) — ONLY used under --include-n10k-captures
    (plan-A). Local mmap; fetch from HF if absent (fail-loud if the bundle is not on HF,
    which is the presumed plan-B state: the n10k captures lived only on pod-779's volume)."""
    if not path.exists():
        from huggingface_hub import hf_hub_download

        logger.info("[n10k] %s absent locally; fetching new_context_vectors.pt from HF", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        got = Path(
            hf_hub_download(
                C.HF_DATA_REPO,
                filename=HF_N10K_BUNDLE,
                repo_type="dataset",
                local_dir=path.parent,
            )
        )
        if got != path:
            got.replace(path)
    return F._mmap_load(path)


def _stream_n50k_layer(prefix: str, layer: int, local_dir: Path | None, cache_dir: Path):
    """Stream-reduce cx_last + v_x at ``layer`` from the n50k capture chunks.

    local_dir given -> read the staged chunks in place (no download). Else list the
    HF prefix (scoped list_repo_tree, #833) and, per chunk: download -> mmap-slice
    the layer -> append -> DELETE the download (peak footprint ~one chunk). Chunks
    are ordered by filename (shard{ii}_chunk{cccc}.pt), so cx_last[i]/v_x[i] stay
    row-aligned; the n50k train membership is a random sample, so the within-n50k
    order is not load-bearing for the split, only X<->Y alignment (guaranteed per
    chunk)."""
    cx_parts: list[np.ndarray] = []
    vx_parts: list[np.ndarray] = []
    if local_dir is not None:
        chunk_files = sorted(local_dir.glob("shard*_chunk*.pt"))
        if not chunk_files:
            raise FileNotFoundError(f"no n50k capture chunks under {local_dir}")
        for cp in chunk_files:
            b = F._mmap_load(cp)
            cx_parts.append(_slice_layer(b, "cx_last", layer))
            vx_parts.append(_slice_layer(b, "v_x", layer))
            del b
        n_kept = sum(p.shape[0] for p in cx_parts)
        logger.info("[n50k] %d chunks (local), %d kept rows", len(chunk_files), n_kept)
        return np.concatenate(cx_parts), np.concatenate(vx_parts), n_kept

    from huggingface_hub import HfApi, hf_hub_download

    names = sorted(
        f.path.rsplit("/", 1)[-1]
        for f in HfApi().list_repo_tree(
            C.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
        )
        if getattr(f, "size", None) is not None and f.path.endswith(".pt")
    )
    if not names:
        raise FileNotFoundError(f"no n50k capture chunks under HF {prefix}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        got = Path(
            hf_hub_download(
                C.HF_DATA_REPO,
                filename=f"{prefix}/{name}",
                repo_type="dataset",
                local_dir=cache_dir,
            )
        )
        b = F._mmap_load(got)
        cx_parts.append(_slice_layer(b, "cx_last", layer))
        vx_parts.append(_slice_layer(b, "v_x", layer))
        del b
        got.unlink()  # stream-reduce: purge each chunk after the layer is sliced
        if (i + 1) % 10 == 0:
            logger.info("[n50k] streamed %d/%d chunks", i + 1, len(names))
    n_kept = sum(p.shape[0] for p in cx_parts)
    logger.info("[n50k] %d chunks (HF stream), %d kept rows", len(names), n_kept)
    return np.concatenate(cx_parts), np.concatenate(vx_parts), n_kept


def assemble_combined(args, layer: int):
    """Combined-corpus (X=cx_last, Y=v_x) at ``layer`` + the byte-identical split.

    plan-B (DEFAULT): pass_b (5000) + n50k new (all ~46600). n10k captures NOT loaded.
    plan-A (--include-n10k-captures): pass_b (5000) + n10k new (~6500) + n50k new (all ~46600).
    Either way the train pool is trimmed to n_train (50,000) in build_n50k_split."""
    pb = F.load_pass_b(args.pass_b)
    assert int(pb["cx_last"].shape[0]) == N_PASS_B, (pb["cx_last"].shape[0], N_PASS_B)
    pb_X, pb_Y = _slice_layer(pb, "cx_last", layer), _slice_layer(pb, "v_x", layer)

    local_dir = args.n50k_capture_dir if args.n50k_capture_dir else None
    n50_X, n50_Y, n50k_kept = _stream_n50k_layer(
        args.hf_prefix, layer, local_dir, args.out_dir / ".n50k_stream_cache"
    )
    pinned = _pinned_original_shas(args.orig_dir)

    if args.include_n10k_captures:  # plan-A: pool orig-train + all n10k + all n50k, trim in split
        nb = _load_n10k_bundle(args.n10k_bundle)
        n10k_kept = int(nb["cx_last"].shape[0])
        nb_X, nb_Y = _slice_layer(nb, "cx_last", layer), _slice_layer(nb, "v_x", layer)
        n50k_used = n50k_kept
        X = np.concatenate([pb_X, nb_X, n50_X]).astype(np.float32)
        Y = np.concatenate([pb_Y, nb_Y, n50_Y]).astype(np.float32)
        assert X.shape[0] == N_PASS_B + n10k_kept + n50k_used, X.shape
    else:  # plan-B (DEFAULT): pass_b + all n50k, no n10k
        n10k_kept = None
        n50k_used = n50k_kept
        X = np.concatenate([pb_X, n50_X]).astype(np.float32)
        Y = np.concatenate([pb_Y, n50_Y]).astype(np.float32)
        assert X.shape[0] == N_PASS_B + n50k_used, X.shape
    assert X.shape[1] == C.EXPECTED_HIDDEN and Y.shape[1] == C.EXPECTED_HIDDEN, (X.shape, Y.shape)

    train, val, test, diag = build_n50k_split(
        n50k_used, n10k_kept, pinned, n_train=args.n_train, seed=args.seed
    )
    diag["n50k_kept_captured"] = int(n50k_kept)
    return X, Y, train, val, test, diag


# ── primal ridge (batched over lambda; val-lambda-selected) ─────────────────────


def _ridge_primal_multi_lambda(Xtr, Ytr, X_eval_list, lambdas, dev):
    """Exact PRIMAL ridge, all lambdas off ONE eigh of the (H,H) X^T X.

    Standardizes X on train stats (matches F._factorize / GramRidge), centers Y on
    train mean. Returns {lambda: [pred for each eval set]} (numpy). fp64 solve."""
    Xtr = torch.as_tensor(np.asarray(Xtr), dtype=torch.float64, device=dev)
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    Y = torch.as_tensor(np.asarray(Ytr), dtype=torch.float64, device=dev)
    ymu = Y.mean(0)
    Yc = Y - ymu
    A = Xtr_n.T @ Xtr_n  # (H, H)
    s, U = torch.linalg.eigh(A)
    s = torch.clamp(s, min=0.0)
    XtY = Xtr_n.T @ Yc  # (H, D)
    UtXtY = U.T @ XtY  # (H, D)
    evals_n = []
    for E in X_eval_list:
        Ee = torch.as_tensor(np.asarray(E), dtype=torch.float64, device=dev)
        evals_n.append((Ee - xmu) / xsd)
    out: dict[float, list[np.ndarray]] = {}
    for lam in lambdas:
        W = U @ (UtXtY / (s + lam)[:, None])  # (H, D)
        out[float(lam)] = [((En @ W) + ymu).cpu().numpy() for En in evals_n]
    return out


def fit_ridge_primal(X, Y, tr, val, te, lambdas, dev):
    """Val-lambda-selected primal ridge. Returns (pred_te, meta)."""
    preds = _ridge_primal_multi_lambda(X[tr], Y[tr], [X[val], X[te]], lambdas, dev)
    best_lam, best_vr2 = float(lambdas[0]), -np.inf
    for lam in lambdas:
        vr2 = PR._pooled_r2(preds[float(lam)][0], Y[val])
        if np.isfinite(vr2) and vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    edge = None
    if np.isclose(best_lam, float(lambdas[0])):
        edge = "low"
    elif np.isclose(best_lam, float(lambdas[-1])):
        edge = "high"
    return preds[best_lam][1], {
        "n_train": len(tr),
        "selection": "val-lambda (primal)",
        "selected_lambda": best_lam,
        "val_r2_at_selected": float(best_vr2),
        "lambda_grid_edge": edge,
    }


# ── exact RBF KRR (chunked kernel assembly + single-GPU Cholesky) ───────────────


def _rbf_kernel_chunked(A, B, gamma, dev, block):
    """K[i,j] = exp(-gamma * ||A_i - B_j||^2), assembled in ROW-BLOCKS of A into
    one preallocated (nA, nB) fp32 tensor on ``dev`` (never the (nA,nB,H) grid)."""
    At = torch.as_tensor(np.asarray(A), dtype=torch.float32, device=dev)
    Bt = torch.as_tensor(np.asarray(B), dtype=torch.float32, device=dev)
    nA = At.shape[0]
    K = torch.empty((nA, Bt.shape[0]), dtype=torch.float32, device=dev)
    for i in range(0, nA, block):
        d2 = torch.cdist(At[i : i + block], Bt) ** 2  # (blk, nB)
        K[i : i + block] = torch.exp(-gamma * d2)
    return K


def fit_krr_exact(X, Y, tr, val, te, *, gamma_mult, lambdas, block, seed, dev):
    """Exact RBF kernel ridge, (gamma, lambda) val-selected. RAW X (matches the
    n10k KRR's median-heuristic-on-raw), all D targets solved in one Cholesky."""
    Xtr = np.asarray(X[tr], dtype=np.float32)
    base_gamma = F.median_heuristic_gamma(Xtr.astype(np.float64), np.random.default_rng(seed + 1))
    Ytr = torch.as_tensor(np.asarray(Y[tr]), dtype=torch.float32, device=dev)
    ymu = Ytr.mean(0)
    Yc = Ytr - ymu
    ntr = Xtr.shape[0]
    grid, best = [], None
    for gm in gamma_mult:
        gamma = base_gamma * gm
        K_tr = _rbf_kernel_chunked(Xtr, Xtr, gamma, dev, block)  # (ntr, ntr) fp32 ~10 GB @ 50k
        K_val = _rbf_kernel_chunked(X[val], Xtr, gamma, dev, block)  # (nval, ntr)
        K_te = _rbf_kernel_chunked(X[te], Xtr, gamma, dev, block)  # (nte, ntr)
        eye = torch.eye(ntr, dtype=torch.float32, device=dev)
        for lam in lambdas:
            L = torch.linalg.cholesky(K_tr + float(lam) * eye)  # single-GPU Cholesky
            alpha = torch.cholesky_solve(Yc, L)  # (ntr, D), all targets at once
            pred_val = (K_val @ alpha + ymu).cpu().numpy()
            pred_te = (K_te @ alpha + ymu).cpu().numpy()
            del L, alpha
            val_r2 = PR._pooled_r2(pred_val, Y[val])
            grid.append(
                {
                    "gamma_mult": float(gm),
                    "gamma": float(gamma),
                    "lambda": float(lam),
                    "val_r2": float(val_r2),
                }
            )
            if best is None or (np.isfinite(val_r2) and val_r2 > best["val_r2"]):
                best = {
                    "gamma_mult": float(gm),
                    "gamma": float(gamma),
                    "lambda": float(lam),
                    "val_r2": float(val_r2),
                    "pred_te": pred_te,
                }
        del K_tr, K_val, K_te, eye
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    assert best is not None
    return best["pred_te"], {
        "n_train": int(ntr),
        "kernel": "exact RBF (no Nystrom)",
        "base_gamma": float(base_gamma),
        "selected": {k: best[k] for k in ("gamma_mult", "gamma", "lambda", "val_r2")},
        "kernel_block": int(block),
        "val_grid": grid,
    }


# ── MLP (width 8192, prior-round recipe reused) + residual-skip ─────────────────


def _mlp_recipe(fair_json: Path) -> dict:
    """The prior-round val-selected MLP recipe (width, lr), REUSED (not re-selected).
    Reads the COMMITTED original-round fair_comparison.json mlp_selection.per_input.last
    (width 8192 / lr 3e-4 at L19 — the same values the n10k round selected). Falls back
    to (8192, 3e-4) if the aggregate is absent (e.g. the n10k dir on the pod volume)."""
    if not fair_json.exists():
        return {"width": 8192, "lr": 3e-4, "source": f"FALLBACK default ({fair_json} absent)"}
    d = json.loads(fair_json.read_text())
    sel = d["mlp_selection"]["per_input"]["last"]
    return {
        "width": int(sel["width"]),
        "lr": float(sel["lr"]),
        "source": f"{fair_json} mlp_selection.per_input.last (L{d['mlp_selection']['layer']})",
    }


def fit_mlp(X, Y, tr, te, recipe, max_epochs, seed, dev):
    fit = F.run_mlp_battery(
        [F.MLPGroup(("m",), X[tr], Y[tr], recipe["width"], recipe["lr"])],
        dev=dev,
        max_epochs=max_epochs,
    )[("m",)]
    return fit.predict(X[te]), {
        "n_train": len(tr),
        "width": int(recipe["width"]),
        "lr": float(recipe["lr"]),
        "epochs_ran": int(fit.epochs_ran),
        "recipe_source": recipe["source"],
    }


def fit_residual_skip(X, Y, tr, val, te, lambdas, recipe, max_epochs, seed, dev):
    """Primal ridge base + MLP on the residual (strictly nests the linear map)."""
    base = _ridge_primal_multi_lambda(X[tr], Y[tr], [X[val], X[tr], X[te]], lambdas, dev)
    best_lam, best_vr2 = float(lambdas[0]), -np.inf
    for lam in lambdas:
        vr2 = PR._pooled_r2(base[float(lam)][0], Y[val])
        if np.isfinite(vr2) and vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    rt_tr, rt_te = base[best_lam][1], base[best_lam][2]
    fit = F.run_mlp_battery(
        [
            F.MLPGroup(
                ("r",),
                X[tr],
                (Y[tr] - rt_tr).astype(np.float32),
                F.RESIDUAL_MLP_WIDTH,
                recipe["lr"],
            )
        ],
        dev=dev,
        max_epochs=max_epochs,
    )[("r",)]
    pred = rt_te + fit.predict(X[te])
    return pred, {
        "n_train": len(tr),
        "base_ridge_lambda": best_lam,
        "residual_mlp_width": int(F.RESIDUAL_MLP_WIDTH),
        "lr": float(recipe["lr"]),
        "epochs_ran": int(fit.epochs_ran),
    }


def _curve(pred_te, Y_te, n_boot, seed) -> dict:
    r2, cos = F._recon_point(pred_te, Y_te)
    ci = F._bootstrap_recon_ci(pred_te, Y_te, n_boot, seed)
    return {"whole_map_r2": float(r2), "mean_cosine": float(cos), "bootstrap_ci": ci}


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #779 n50k fits (n_train=50,000).")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=N50K_TRAIN)
    ap.add_argument("--predictors", default=",".join(PREDICTORS))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=F.BOOT_N)
    ap.add_argument("--mlp-max-epochs", type=int, default=F.MLP_MAX_EPOCHS)
    ap.add_argument("--krr-gamma-mult", default=",".join(str(g) for g in KRR_GAMMA_MULT))
    ap.add_argument("--krr-lambdas", default=",".join(str(x) for x in KRR_LAMBDAS))
    ap.add_argument("--krr-block", type=int, default=KRR_KERNEL_BLOCK)
    ap.add_argument("--pass-b", type=Path, default=F.PASS_B_PATH)
    ap.add_argument(
        "--include-n10k-captures",
        action="store_true",
        help="plan-A: pool orig-train (3600) + n10k (6500) + all 46600 n50k, trim to 50000 "
        "(FAILS LOUD at startup if the n10k bundle is not on HF; default plan-B uses "
        "orig-train + all 46600 n50k)",
    )
    ap.add_argument("--n10k-bundle", type=Path, default=F.DEFAULT_NEW_BUNDLE)
    ap.add_argument(
        "--orig-dir",
        type=Path,
        default=DEFAULT_ORIG_DIR,
        help="original-round dir with committed fair_comparison.json (split-sha pin + MLP recipe)",
    )
    ap.add_argument(
        "--n50k-capture-dir",
        type=Path,
        default=None,
        help="local staged n50k capture chunks (else stream from HF)",
    )
    ap.add_argument("--hf-prefix", default=HF_N50K_PREFIX)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(args.n_threads))
    dev = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    want = [p.strip() for p in args.predictors.split(",") if p.strip()]
    for p in want:
        if p not in PREDICTORS:
            raise ValueError(f"unknown predictor {p!r}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_json is None:
        args.out_json = args.out_dir / "n50k_fits.json"

    corpus_mode = "n50k-plan-a" if args.include_n10k_captures else "n50k-plan-b"
    results = json.loads(args.out_json.read_text()) if args.out_json.exists() else {}
    if results.get("layer") is not None and results["layer"] != args.layer:
        raise SystemExit(
            f"--out-json {args.out_json} was written for layer {results['layer']} but "
            f"--layer={args.layer}; refusing to mix cross-layer rows (use a layer-specific "
            "--out-json)"
        )
    if results.get("corpus_mode") not in (None, corpus_mode):
        raise SystemExit(
            f"--out-json {args.out_json} was written under corpus_mode={results['corpus_mode']!r} "
            f"but this run is {corpus_mode!r} (plan-A vs plan-B differ in the training corpus); "
            "refusing to mix — use a mode-specific --out-json"
        )

    if args.include_n10k_captures:
        # plan-A fail-loud AT STARTUP: never silently fall back to plan-B when the n10k
        # bundle is absent (the presumed state — the captures lived only on pod-779's volume).
        from huggingface_hub import HfApi

        if not (
            args.n10k_bundle.exists()
            or HfApi().file_exists(C.HF_DATA_REPO, HF_N10K_BUNDLE, repo_type="dataset")
        ):
            raise SystemExit(
                "--include-n10k-captures (plan-A) requires the n10k bundle, absent both locally "
                f"({args.n10k_bundle}) and on HF ({C.HF_DATA_REPO}:{HF_N10K_BUNDLE}). The n10k "
                "captures lived only on pod-779's stopped volume; rescue that volume + upload the "
                "bundle first, or drop the flag to run the DEFAULT plan-B (orig-train + n50k only)."
            )

    t0 = time.time()
    X, Y, tr, val, te, split = assemble_combined(args, args.layer)
    recipe = _mlp_recipe(args.orig_dir / "fair_comparison.json")
    lambdas = LAMBDAS_N50K
    gamma_mult = tuple(float(g) for g in args.krr_gamma_mult.split(",") if g.strip())
    krr_lambdas = tuple(float(x) for x in args.krr_lambdas.split(",") if x.strip())
    logger.info(
        "assembled combined corpus: n_train=%d n_val=%d n_test=%d (L%d, %.0fs)",
        len(tr),
        len(val),
        len(te),
        args.layer,
        time.time() - t0,
    )

    results.setdefault("per_predictor", {})
    results.update(
        {
            "corpus_mode": corpus_mode,
            "layer": int(args.layer),
            "seed": int(args.seed),
            "split": split,
            "mlp_recipe": recipe,
            "lambda_grid": {
                "n": len(lambdas),
                "min": float(lambdas[0]),
                "max": float(lambdas[-1]),
                "log10": [round(float(np.log10(x)), 3) for x in lambdas],
            },
            "krr_grid": {"gamma_mult": list(gamma_mult), "lambdas": list(krr_lambdas)},
            "predictor_labels": PREDICTOR_LABEL,
            "note": (
                "n_train=50,000 rerun of the fitter-fair-comparison (plan-B, 2026-07-14). "
                "val/test BYTE-IDENTICAL to the ORIGINAL round (asserted vs pinned shas). "
                "DEFAULT plan-B: orig-train (3600) + 46600 new = pool 50200, trim to 50000; "
                "n10k captures NOT used. "
                "ridge=PRIMAL val-lambda over LAMBDAS_N50K; krr=EXACT RBF (no Nystrom, "
                "chunked+Cholesky); mlp=w8192 original-round recipe reused; "
                "residual-skip=primal ridge + MLP."
            ),
            "metadata": C.reproducibility_metadata(
                {
                    "script": "issue779_ffc_n50k_fits",
                    "corpus_mode": corpus_mode,
                    "device": args.device,
                }
            ),
        }
    )
    C.write_json_atomic(args.out_json, results)

    for name in want:
        if args.resume and name in results["per_predictor"]:
            logger.info("[resume] %s present; skip", name)
            continue
        logger.info("[fit] %s (n_train=%d, %s) ...", name, len(tr), dev)
        ts = time.time()
        if name == "ridge":
            pred_te, meta = fit_ridge_primal(X, Y, tr, val, te, lambdas, dev)
        elif name == "krr":
            pred_te, meta = fit_krr_exact(
                X,
                Y,
                tr,
                val,
                te,
                gamma_mult=gamma_mult,
                lambdas=krr_lambdas,
                block=args.krr_block,
                seed=args.seed,
                dev=dev,
            )
        elif name == "mlp":
            pred_te, meta = fit_mlp(X, Y, tr, te, recipe, args.mlp_max_epochs, args.seed, dev)
        else:  # residual_skip
            pred_te, meta = fit_residual_skip(
                X, Y, tr, val, te, lambdas, recipe, args.mlp_max_epochs, args.seed, dev
            )
        curve = _curve(pred_te, Y[te], args.n_boot, args.seed)
        curve["fit_meta"] = meta
        curve["wall_time_s"] = round(time.time() - ts, 1)
        results["per_predictor"][name] = curve
        C.write_json_atomic(args.out_json, results)
        logger.info(
            "[done] %s: whole-map R2=%.4f mean-cos=%.4f (%.0fs)",
            name,
            curve["whole_map_r2"],
            curve["mean_cosine"],
            curve["wall_time_s"],
        )

    logger.info("wrote %s (%.0fs total)", args.out_json, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
