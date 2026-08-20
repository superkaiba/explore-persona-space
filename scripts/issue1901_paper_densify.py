#!/usr/bin/env python3
"""Issue #1901 paper-figure densification (ESCALATION 1, CPU pod ``pod-1901-paperdense``).

Two 0-GPU-h jobs over banked #779 fair-comparison activation captures, reusing
the #779/#1491/#2330 fit chain by IMPORT (never re-implementations):

Job A (``--phase layer-curve``): dense per-layer curve at n_train=50,000 over
ALL 28 layers of the ``fitter-fair-comparison-n50k`` combined capture (pass_b
5,000 + 46,600 n50k rows; plan-B split, seed 42 — byte-identical pinned
val/test, asserted vs the pinned shas). Per layer: val-lambda primal ridge
(``N50.LAMBDAS_N50K``, streaming-Gram ``N1M.fit_ridge_with_weights`` — the n1m
driver's smoke asserts it prediction-identical (<1e-4) to the banked
``N50._ridge_primal_multi_lambda`` path) + the identity+bias baseline
(``analysis.mapping_baselines``), each scored as held-out pooled R2 (+1000-draw
bootstrap CI) AND kNN retrieval acc@k (euclidean + cosine, pool = the 1,000
pinned test targets, chance 1e-3). PARITY GATE (production only): the L19
ridge R2 must reproduce the banked
``eval_results/issue_779/fitter-fair-comparison-n50k/n50k_fits.json`` value
(0.7600) within ``--parity-tol``; L19 runs FIRST so a mismatch halts before
the other 27 layers spend.

Job B (``--phase bign``): ridge refits at the lmsys_150k / lmsys_500k scaling
points (L19) over the ``fitter-fair-comparison-n1m`` capture via
``N1M.assemble`` + ``N1M._pool_rows`` + ``N1M.select_train``
(``LAMBDAS_N1M``, ``RIDGE_BLOCK``) + identity+bias, scored the same way, with
R2 parity vs the banked ``n1m_fits.json`` per-point values. The n=963,444
mixed_1m point is NOT refit — banked ridge weights exist on HF
(``issue779_monitoring/n1m_readout/weights``) and #1901's metric battery
already scored its retrieval (recorded here as a pointer block only).
CAVEAT (recorded in meta): the original n1m subset selection seeds
``default_rng(seed + abs(hash(name)) % 1e6)`` under an UNRECORDED
PYTHONHASHSEED, so the exact original train subsets are not bit-recoverable;
this run pins PYTHONHASHSEED in the launcher (recorded), and R2 parity is
statistical (same pool, same n, same protocol) at ``--parity-tol``.

Staging: scoped ``list_repo_tree(path_in_repo=<prefix>)`` per prefix (never a
bare full-repo listing on the ~1M-file data repo, #833) + parallel per-file
``hf_hub_download`` under ``hub.retry_transient`` + per-file size
verification; each capture is DELETED after its phase consumes it
(``--phase all``: A staged -> fit -> deleted -> B staged -> fit -> deleted;
the two captures are never on disk together — 240 GB container-disk plan).
Capture dtype is read off the first chunk and recorded in the JSON meta
(fp16 suspected for n50k, fp32 for n1m — verified, never assumed).

Tiny-real smoke (``--smoke-chunks N``): restricts each phase to the first N
capture chunks (train pools clamp through the reuse chain's own warning
branch); the parity gates + expected-row asserts are production-n-calibrated
and demote to informational log lines under smoke (#1345 gate-calibration
rule). Smoke blind-spot enumeration: the parity PASS/FAIL branch at
production values, full-n RAM peaks, and the full staged byte volume are NOT
certified by a smoke PASS; every load path (staging, mmap slice, split
asserts against the real 5,000-row pass_b, fits, retrieval, JSON writes) IS
exercised on real data.

Checkpointing (>1h loop): per-layer / per-point JSONs written atomically the
moment each unit completes, resume keyed on GENERATING PARAMETERS (seed,
n-train target, lambda-grid params, layer, split train sha) — never
recomputed-float-array bytes (#1336). One stdout progress line per unit.

Refusal-safety: tensors/counts/shas only — no context or rollout text is ever
printed or logged. 0 GPU-h; ``--device cpu``. Fail loud; NaN never coerced.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps land BEFORE numpy/torch import.
load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as MB  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1901_paper_densify")

N_LAYERS = C.EXPECTED_LAYERS  # 28
H_DIM = C.EXPECTED_HIDDEN  # 3584
KNN_KS = (1, 5, 10)
KNN_METRICS = ("euclidean", "cosine")

# Banked parity targets (committed eval JSONs; values pasted from the artifacts,
# re-asserted against the committed files when present — see _banked_parity_target).
PARITY_L19_N50K_R2 = 0.7599992543132661  # fitter-fair-comparison-n50k/n50k_fits.json ridge
PARITY_N1M_R2 = {
    "lmsys_150k": 0.754355957580797,  # n1m_fits.json per_point.lmsys_150k.predictors.ridge
    "lmsys_500k": 0.7608232696391413,  # n1m_fits.json per_point.lmsys_500k.predictors.ridge
}
BANKED_N50K_FITS = (
    PROJECT_ROOT / "eval_results/issue_779/fitter-fair-comparison-n50k/n50k_fits.json"
)
BANKED_N1M_FITS = PROJECT_ROOT / "eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json"
# mixed_1m pointer block (NOT refit here): banked weights + #1901 metric-battery reads.
MIXED_1M_POINTER = {
    "n_train_realized": 963444,
    "ridge_whole_map_r2": 0.7541708417500051,  # n1m_fits.json per_point.mixed_1m.predictors.ridge
    "acc_at_1_euclidean_test1000": 0.805,  # eval_results/issue_1901/metric_battery/context_arm.json
    "acc_at_1_source": "eval_results/issue_1901/metric_battery/context_arm.json "
    "per_layer.19.arms.ridge.retrieval.test.euclidean",
    "weights_hf_prefix": "issue779_monitoring/n1m_readout/weights",
    "note": "banked mixed_1m ridge weights reused by #1901's metric battery; not refit here",
}

DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_1901" / "paper_densify"


# ── staging (scoped listing + parallel per-file download + size verify) ─────────


def _list_prefix(prefix: str) -> list[tuple[str, int]]:
    """Scoped (path, size) listing of one data-repo prefix (#833: never a bare
    full-repo listing on the ~1M-file repo), under the transient-retry wrapper."""
    from huggingface_hub import HfApi

    tree = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: whole listing runs inside hub.retry_transient (this lambda).
            HfApi().list_repo_tree(
                C.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
            )
        ),
        what=f"scoped tree listing ({prefix})",
    )
    out = [(f.path, int(f.size)) for f in tree if getattr(f, "size", None) is not None]
    if not out:
        raise FileNotFoundError(f"no files under HF {C.HF_DATA_REPO}:{prefix}")
    return sorted(out)


def stage_prefix(
    prefix: str, stage_root: Path, *, max_files: int | None = None, workers: int = 8
) -> Path:
    """Parallel-download one HF prefix to ``stage_root/<prefix>`` (files land at
    their repo-relative paths). Already-present files with matching size skip
    (resume). Returns the staged prefix dir after a per-file size verification."""
    from huggingface_hub import hf_hub_download

    files = _list_prefix(prefix)
    if max_files is not None:
        files = files[:max_files]
    dest = stage_root / prefix
    todo = [
        (p, sz)
        for p, sz in files
        if not ((stage_root / p).exists() and (stage_root / p).stat().st_size == sz)
    ]
    total_gb = sum(sz for _, sz in files) / 1e9
    logger.info(
        "[stage] %s: %d files (%.1f GB), %d to download", prefix, len(files), total_gb, len(todo)
    )
    t0 = time.time()
    done = 0

    def _fetch(path: str) -> str:
        return hub.retry_transient(
            lambda: hf_hub_download(
                C.HF_DATA_REPO, filename=path, repo_type="dataset", local_dir=stage_root
            ),
            what=f"stage {path}",
        )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, p): p for p, _ in todo}
        for fut in as_completed(futs):
            fut.result()  # propagate failures loudly
            done += 1
            if done % 25 == 0 or done == len(todo):
                logger.info(
                    "[stage] %s: %d/%d files (%.0fs)", prefix, done, len(todo), time.time() - t0
                )
    bad = [
        p
        for p, sz in files
        if not (stage_root / p).exists() or (stage_root / p).stat().st_size != sz
    ]
    if bad:
        raise RuntimeError(f"staging verify FAILED for {len(bad)} files under {prefix}: {bad[:5]}")
    logger.info(
        "[stage] %s verified: %d files, %.1f GB (%.0fs)",
        prefix,
        len(files),
        total_gb,
        time.time() - t0,
    )
    return dest


def _reap_stage(dirpath: Path) -> None:
    """Delete a consumed staged capture (fail-loud; the 240 GB disk plan)."""
    if dirpath.exists():
        shutil.rmtree(dirpath)  # no ignore_errors — a failed reap must be loud
        logger.info("[stage] reaped %s", dirpath)


# ── retrieval scoring (canonical helper + rank-vector CI, parity-asserted) ──────


def _rank_vector(pred: np.ndarray, pool: np.ndarray, metric: str) -> np.ndarray:
    """Mid-rank vector of each row's true target (pool == true, diagonal identity)
    — the exact ``MB.knn_retrieval`` formula, exposed for bootstrap CIs; acc@1
    parity vs the helper is asserted at every call site."""
    pred = np.asarray(pred, dtype=np.float64)
    pool = np.asarray(pool, dtype=np.float64)
    d = MB._pairwise_dist(pred, pool, metric)
    n = pred.shape[0]
    d_true = d[np.arange(n), np.arange(n)]
    tol = 1e-9 * np.maximum(np.abs(d_true)[:, None], 1e-12)
    closer = (d < d_true[:, None] - tol).sum(axis=1)
    tied = (np.abs(d - d_true[:, None]) <= tol).sum(axis=1) - 1
    return 1.0 + closer + 0.5 * tied


def score_cell(pred_te: np.ndarray, Y_te: np.ndarray, n_boot: int, seed: int) -> dict:
    """Pooled R2 + mean cosine (+bootstrap CI, banked formula) AND kNN retrieval
    (euclidean + cosine, pool = the held-out true targets) with an acc@1
    bootstrap CI off the rank vector."""
    r2, cos = F._recon_point(pred_te, Y_te)
    ci = F._bootstrap_recon_ci(pred_te, Y_te, n_boot, seed)
    out = {
        "whole_map_r2": float(r2),
        "mean_cosine": float(cos),
        "bootstrap_ci": ci,
        "retrieval": {},
    }
    rng = np.random.default_rng(seed + 7)
    n = Y_te.shape[0]
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    for metric in KNN_METRICS:
        helper = MB.knn_retrieval(pred_te, Y_te, ks=KNN_KS, metric=metric)
        ranks = _rank_vector(pred_te, Y_te, metric)
        acc1 = float((ranks <= 1).mean())
        assert abs(acc1 - helper["acc_at_k"][1]) < 1e-12, (acc1, helper["acc_at_k"][1])
        draws = (ranks[boot_idx] <= 1).mean(axis=1)
        helper["acc1_ci"] = {
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
        }
        out["retrieval"][metric] = helper
    return out


def _banked_parity_target(path: Path, extract, fallback: float) -> float:
    """Read a parity target from the committed banked JSON when present (the pod
    opens the eval_results/issue_779 sparse cone); fall back to the pasted
    module constant, asserting agreement when both exist."""
    if path.exists():
        got = float(extract(json.loads(path.read_text())))
        assert abs(got - fallback) < 1e-9, (path, got, fallback)
        return got
    return fallback


def _parity_check(name: str, got_r2: float, want_r2: float, tol: float, *, smoke: bool) -> dict:
    ok = bool(abs(got_r2 - want_r2) <= tol)
    row = {
        "cell": name,
        "got_r2": float(got_r2),
        "banked_r2": float(want_r2),
        "tol": tol,
        "pass": ok,
    }
    if smoke:
        logger.info("[parity] (smoke, informational) %s", row)
    elif not ok:
        raise RuntimeError(f"PARITY GATE FAILED: {row} — investigate before proceeding")
    else:
        logger.info("[parity] PASS %s", row)
    return row


# ── Job A: n=50k dense layer curve ──────────────────────────────────────────────


def _extract_all_layers(capture_dir: Path, max_chunks: int | None):
    """Two-pass multi-layer extraction of cx_last + v_x from the n50k chunk
    bundles: pass 1 reads shapes (mmap headers), pass 2 fills preallocated
    arrays in the capture's OWN dtype. Returns (X_all, Y_all, layers, dtype_str)
    with X_all/Y_all of shape (n_new, N_LAYERS, H)."""
    chunk_files = sorted(capture_dir.glob("shard*_chunk*.pt"))
    if max_chunks is not None:
        chunk_files = chunk_files[:max_chunks]
    if not chunk_files:
        raise FileNotFoundError(f"no capture chunks under {capture_dir}")
    rows = []
    layers = None
    dtype = None
    for cp in chunk_files:
        b = F._mmap_load(cp)
        t = b["cx_last"]
        assert t.shape[1:] == (N_LAYERS, H_DIM), (cp.name, tuple(t.shape))
        assert b["v_x"].shape == t.shape, (cp.name, tuple(b["v_x"].shape))
        if layers is None:
            layers = [int(x) for x in b["layers"]]
            dtype = str(t.dtype).replace("torch.", "")
            logger.info(
                "[extract] capture dtype=%s layers=%s (first chunk %s)", dtype, layers[:4], cp.name
            )
        else:
            assert [int(x) for x in b["layers"]] == layers, cp.name
            assert str(t.dtype).replace("torch.", "") == dtype, (cp.name, t.dtype)
        rows.append(int(t.shape[0]))
        del b
    n_new = int(sum(rows))
    np_dtype = np.dtype(dtype.replace("bfloat16", "float32"))  # bf16 has no numpy dtype; upcast
    X_all = np.empty((n_new, N_LAYERS, H_DIM), dtype=np_dtype)
    Y_all = np.empty((n_new, N_LAYERS, H_DIM), dtype=np_dtype)
    off = 0
    t0 = time.time()
    for i, cp in enumerate(chunk_files):
        b = F._mmap_load(cp)
        k = rows[i]
        X_all[off : off + k] = b["cx_last"].to(getattr(torch, str(np_dtype))).numpy()
        Y_all[off : off + k] = b["v_x"].to(getattr(torch, str(np_dtype))).numpy()
        off += k
        del b
        if (i + 1) % 10 == 0 or i + 1 == len(chunk_files):
            logger.info(
                "[extract] %d/%d chunks (%d rows, %.0fs)",
                i + 1,
                len(chunk_files),
                off,
                time.time() - t0,
            )
    assert off == n_new
    return X_all, Y_all, layers, dtype


def _layer_unit_key(args, layer: int, train_sha: str, n_rows: int, dtype: str) -> dict:
    return {
        "layer": int(layer),
        "seed": int(args.seed_a),
        "n_train_target": 50000,
        "lambda_grid": ["logspace", -3, 7, 21],  # generating params, never array bytes (#1336)
        "train_sha256": train_sha,
        "n_rows": int(n_rows),
        "dtype": dtype,
        "smoke_chunks": args.smoke_chunks,
    }


def phase_layer_curve(args) -> dict:
    smoke = args.smoke_chunks > 0
    dev = torch.device(args.device)
    unit_dir = args.out_dir / ("layer_curve_smoke" if smoke else "layer_curve")
    unit_dir.mkdir(parents=True, exist_ok=True)

    capture_dir = stage_prefix(
        N50.HF_N50K_PREFIX,
        args.stage_root,
        max_files=(args.smoke_chunks if smoke else None),
        workers=args.stage_workers,
    )
    pb = N1G._load_pass_b_bundle(args.pass_b)
    assert int(pb["cx_last"].shape[0]) == N50.N_PASS_B, pb["cx_last"].shape

    X_all, Y_all, cap_layers, dtype = _extract_all_layers(
        capture_dir, args.smoke_chunks if smoke else None
    )
    n_new = X_all.shape[0]
    if not smoke and n_new != N50.N_N50K_NEW:
        raise RuntimeError(f"expected {N50.N_N50K_NEW} n50k kept rows, extracted {n_new}")

    pinned = N50._pinned_original_shas(args.orig_dir)
    train, val, test, diag = N50.build_n50k_split(
        n_new, None, pinned, n_train=50000, seed=args.seed_a
    )
    diag["n50k_kept_captured"] = int(n_new)
    n_rows = N50.N_PASS_B + n_new

    parity_l19 = _banked_parity_target(
        BANKED_N50K_FITS, lambda d: d["per_predictor"]["ridge"]["whole_map_r2"], PARITY_L19_N50K_R2
    )

    def _assemble_layer(layer: int):
        col = cap_layers.index(layer)
        pb_X = N50._slice_layer(pb, "cx_last", layer)
        pb_Y = N50._slice_layer(pb, "v_x", layer)
        X = np.concatenate([pb_X, X_all[:, col, :].astype(np.float32)])
        Y = np.concatenate([pb_Y, Y_all[:, col, :].astype(np.float32)])
        assert X.shape == (n_rows, H_DIM) and Y.shape == X.shape, (X.shape, Y.shape)
        return X, Y

    # L19 FIRST (the parity gate halts before the other 27 layers spend).
    want_layers = [19] + [li for li in cap_layers if li != 19]
    if smoke:
        want_layers = [19, 0]
    parity_rows = []
    t_all = time.time()
    for k, layer in enumerate(want_layers):
        out_path = unit_dir / f"L{layer}.json"
        key = _layer_unit_key(args, layer, diag["train_sha256"], n_rows, dtype)
        if out_path.exists():
            prev = json.loads(out_path.read_text())
            if prev.get("unit_key") == key:
                logger.info(
                    "[layer-curve] unit %d/%d L%d resume-skip", k + 1, len(want_layers), layer
                )
                if layer == 19:
                    parity_rows.append(
                        _parity_check(
                            "L19-ridge-n50k",
                            prev["ridge"]["whole_map_r2"],
                            parity_l19,
                            args.parity_tol,
                            smoke=smoke,
                        )
                    )
                continue
        ts = time.time()
        X, Y = _assemble_layer(layer)
        pred_ridge, meta, _payload = N1M.fit_ridge_with_weights(
            X, Y, train, val, test, N50.LAMBDAS_N50K, dev, args.ridge_block
        )
        ridge_cell = score_cell(pred_ridge, Y[test], args.n_boot, args.seed_a)
        ridge_cell["fit_meta"] = meta
        pred_ib = MB.identity_bias_predict(X[train], Y[train], X[test])
        ib_cell = score_cell(pred_ib, Y[test], args.n_boot, args.seed_a)
        unit = {
            "unit_key": key,
            "layer": int(layer),
            "ridge": ridge_cell,
            "identity_bias": ib_cell,
            "wall_time_s": round(time.time() - ts, 1),
        }
        if layer == 19:
            parity_rows.append(
                _parity_check(
                    "L19-ridge-n50k",
                    ridge_cell["whole_map_r2"],
                    parity_l19,
                    args.parity_tol,
                    smoke=smoke,
                )
            )
        C.write_json_atomic(out_path, unit)
        logger.info(
            "[layer-curve] unit %d/%d L%d ridge_r2=%.4f ridge_acc1=%.3f idbias_r2=%.4f "
            "idbias_acc1=%.3f elapsed=%.0fs",
            k + 1,
            len(want_layers),
            layer,
            ridge_cell["whole_map_r2"],
            ridge_cell["retrieval"]["euclidean"]["acc_at_k"][1],
            ib_cell["whole_map_r2"],
            ib_cell["retrieval"]["euclidean"]["acc_at_k"][1],
            time.time() - ts,
        )

    merged = {
        "per_layer": {
            str(li): json.loads((unit_dir / f"L{li}.json").read_text())
            for li in want_layers
            if (unit_dir / f"L{li}.json").exists()
        },
        "split": diag,
        "capture_dtype": dtype,
        "layers": want_layers,
        "n_rows": n_rows,
        "lambda_grid": {"n": 21, "min": 1e-3, "max": 1e7, "generating": ["logspace", -3, 7, 21]},
        "knn": {
            "ks": list(KNN_KS),
            "metrics": list(KNN_METRICS),
            "pool": "pinned test_1000 true targets",
            "chance_at_1": 0.001,
        },
        "parity": parity_rows,
        "predictors": {
            "ridge": "primal ridge (linear, streaming X^TX; val-lambda over LAMBDAS_N50K)",
            "identity_bias": "W=identity + train-mean bias (analysis.mapping_baselines)",
        },
        "smoke_chunks": args.smoke_chunks,
        "metadata": C.reproducibility_metadata(
            {"script": "issue1901_paper_densify", "phase": "layer-curve", "device": args.device}
        ),
        "note": (
            "n_train=50,000 dense per-layer curve (all 28 layers) over the #779 "
            "fitter-fair-comparison-n50k capture; plan-B split seed 42, val/test "
            "byte-identical to the original round (pinned shas asserted). acc@k pool = "
            "the 1,000 pinned test targets (chance 1e-3 at k=1)."
        ),
    }
    out_json = args.out_dir / ("layer_curve_n50k_smoke.json" if smoke else "layer_curve_n50k.json")
    C.write_json_atomic(out_json, merged)
    logger.info("[layer-curve] wrote %s (%.0fs total)", out_json, time.time() - t_all)
    if not args.keep_stage:
        _reap_stage(args.stage_root / Path(N50.HF_N50K_PREFIX).parent)
    return merged


# ── Job B: acc@1 at the big scaling points (L19) ────────────────────────────────


def _point_unit_key(args, name: str, n_target: int, sel_sha: str, n_realized: int) -> dict:
    return {
        "point": name,
        "layer": 19,
        "seed": int(args.seed_b),
        "n_train_target": int(n_target),
        "n_train_realized": int(n_realized),
        "lambda_grid": ["logspace", -3, 8, 23],
        "sel_sha256": sel_sha,
        "smoke_chunks": args.smoke_chunks,
    }


def phase_bign(args) -> dict:
    smoke = args.smoke_chunks > 0
    dev = torch.device(args.device)
    unit_dir = args.out_dir / ("bign_smoke" if smoke else "bign")
    unit_dir.mkdir(parents=True, exist_ok=True)
    n1m_capture_prefix = f"{N1G.HF_PREFIX}/final_token_capture"
    capture_dir = stage_prefix(
        n1m_capture_prefix,
        args.stage_root,
        max_files=(args.smoke_chunks if smoke else None),
        workers=args.stage_workers,
    )

    # Namespace shim for N1M.assemble — every args.<attr> the callee reads on every
    # reachable branch (audited against the assemble body; #1776 hand-built-Namespace rule):
    # pass_b, manifest_from_hf, manifest_hf_prefix, out_dir, n1m_capture_dir,
    # fresh_stream, hf_prefix, orig_dir.
    ns = argparse.Namespace(
        pass_b=args.pass_b,
        manifest_from_hf=True,
        manifest_hf_prefix=N1G.HF_PREFIX,
        out_dir=args.work_dir,
        n1m_capture_dir=capture_dir,
        fresh_stream=False,
        hf_prefix=n1m_capture_prefix,
        orig_dir=args.orig_dir,
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    X, Y, prov, r1_train, val, test, split = N1M.assemble(ns, layer=19)
    pools = N1M._pool_rows(prov, r1_train, X.shape[0], val, test)
    logger.info(
        "[bign] assembled n_rows=%d (lmsys pool %d, wildchat %d)",
        X.shape[0],
        len(pools["lmsys"]),
        len(pools["new_wildchat"]),
    )

    points = {}
    for name, n_target in (("lmsys_150k", 150_000), ("lmsys_500k", 500_000)):
        sel, sel_diag = N1M.select_train(pools, name, n_target, "lmsys", args.seed_b)
        sel_sha = F._sha_ids(sel)
        key = _point_unit_key(args, name, n_target, sel_sha, len(sel))
        out_path = unit_dir / f"{name}.json"
        if out_path.exists():
            prev = json.loads(out_path.read_text())
            if prev.get("unit_key") == key:
                logger.info("[bign] %s resume-skip", name)
                points[name] = prev
                continue
        if not smoke and len(sel) != n_target:
            raise RuntimeError(f"{name}: realized train {len(sel)} != target {n_target}")
        ts = time.time()
        pred_ridge, meta, _payload = N1M.fit_ridge_with_weights(
            X, Y, sel, val, test, N1M.LAMBDAS_N1M, dev, args.ridge_block
        )
        ridge_cell = score_cell(pred_ridge, Y[test], args.n_boot, args.seed_b)
        ridge_cell["fit_meta"] = meta
        parity = _parity_check(
            f"{name}-ridge",
            ridge_cell["whole_map_r2"],
            _banked_parity_target(
                BANKED_N1M_FITS,
                lambda d, _n=name: d["per_point"][_n]["predictors"]["ridge"]["whole_map_r2"],
                PARITY_N1M_R2[name],
            ),
            args.parity_tol,
            smoke=smoke,
        )
        pred_ib = MB.identity_bias_predict(X[sel], Y[sel], X[test])
        ib_cell = score_cell(pred_ib, Y[test], args.n_boot, args.seed_b)
        unit = {
            "unit_key": key,
            "selection": sel_diag,
            "ridge": ridge_cell,
            "identity_bias": ib_cell,
            "parity": parity,
            "wall_time_s": round(time.time() - ts, 1),
        }
        C.write_json_atomic(out_path, unit)
        points[name] = unit
        logger.info(
            "[bign] unit %s n=%d ridge_r2=%.4f ridge_acc1=%.3f idbias_acc1=%.3f elapsed=%.0fs",
            name,
            len(sel),
            ridge_cell["whole_map_r2"],
            ridge_cell["retrieval"]["euclidean"]["acc_at_k"][1],
            ib_cell["retrieval"]["euclidean"]["acc_at_k"][1],
            time.time() - ts,
        )

    merged = {
        "per_point": points,
        "mixed_1m_pointer": MIXED_1M_POINTER,
        "split": split,
        "layer": 19,
        "seed": int(args.seed_b),
        "lambda_grid": {"n": 23, "min": 1e-3, "max": 1e8, "generating": ["logspace", -3, 8, 23]},
        "knn": {
            "ks": list(KNN_KS),
            "metrics": list(KNN_METRICS),
            "pool": "pinned test_1000 true targets",
            "chance_at_1": 0.001,
        },
        "subset_reproducibility_caveat": (
            "N1M.select_train seeds default_rng(seed + abs(hash(name)) % 1e6); the original "
            "n1m fits ran under an unrecorded PYTHONHASHSEED, so the exact original train "
            "subsets are not bit-recoverable. This run pins PYTHONHASHSEED (launcher env, "
            "recorded below); R2 parity vs the banked per-point values is therefore "
            "statistical (same pool / n / protocol) at parity_tol."
        ),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "smoke_chunks": args.smoke_chunks,
        "metadata": C.reproducibility_metadata(
            {"script": "issue1901_paper_densify", "phase": "bign", "device": args.device}
        ),
        "note": (
            "L19 ridge refits at lmsys_150k / lmsys_500k over the #779 "
            "fitter-fair-comparison-n1m capture (val/test byte-identical pinned split), "
            "scored as held-out R2 + kNN acc@k on the pinned 1,000-target test pool; "
            "identity+bias companion at both points; mixed_1m recorded as a banked pointer."
        ),
    }
    out_json = args.out_dir / (
        "scaling_bigN_acc1_L19_smoke.json" if smoke else "scaling_bigN_acc1_L19.json"
    )
    C.write_json_atomic(out_json, merged)
    logger.info("[bign] wrote %s", out_json)
    if not args.keep_stage:
        _reap_stage(args.stage_root / N1G.HF_PREFIX)
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1901 paper-figure densification (CPU pod).")
    ap.add_argument("--phase", choices=["layer-curve", "bign", "all"], default="all")
    ap.add_argument(
        "--stage-root", type=Path, required=True, help="capture staging root (container disk)"
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="scratch for manifest/stream caches (default <stage-root>/work)",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--pass-b",
        type=Path,
        default=None,
        help="pass_b bundle path (default <stage-root>/pass_b/train_context_vectors.pt)",
    )
    ap.add_argument("--orig-dir", type=Path, default=N50.DEFAULT_ORIG_DIR)
    ap.add_argument(
        "--seed-a", type=int, default=42, help="n50k split seed (banked n50k_fits.json ran 42)"
    )
    ap.add_argument(
        "--seed-b", type=int, default=0, help="n1m selection seed (banked n1m_fits.json ran 0)"
    )
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--n-threads", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=F.BOOT_N)
    ap.add_argument("--ridge-block", type=int, default=N1M.RIDGE_BLOCK)
    ap.add_argument("--stage-workers", type=int, default=8)
    ap.add_argument("--parity-tol", type=float, default=1e-2)
    ap.add_argument(
        "--smoke-chunks",
        type=int,
        default=0,
        help=">0: tiny-real smoke on the first N chunks per phase",
    )
    ap.add_argument(
        "--keep-stage", action="store_true", help="do not delete staged captures (smoke)"
    )
    ap.add_argument(
        "--sentinel", type=Path, default=None, help="results sentinel JSON written at completion"
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok")
        raise SystemExit(0)
    torch.set_num_threads(int(args.n_threads))
    if args.work_dir is None:
        args.work_dir = args.stage_root / "work"
    if args.pass_b is None:
        args.pass_b = args.stage_root / "pass_b" / "train_context_vectors.pt"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.phase in ("layer-curve", "all"):
        C.phase("layer-curve")
        results["layer_curve"] = {"out": str(args.out_dir), "ok": True}
        phase_layer_curve(args)
    if args.phase in ("bign", "all"):
        C.phase("bign")
        results["bign"] = {"out": str(args.out_dir), "ok": True}
        phase_bign(args)
    C.phase("done")
    if args.sentinel is not None:
        C.write_json_atomic(
            args.sentinel,
            {
                "ok": True,
                "phase": args.phase,
                "smoke_chunks": args.smoke_chunks,
                "outputs": results,
            },
        )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
