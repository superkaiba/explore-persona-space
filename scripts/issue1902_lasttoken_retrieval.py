#!/usr/bin/env python3
"""Strict answer retrieval for the OLMo-2 final-token IID maps.

This analysis reuses the final-prompt-token, six-random-fold fits from
``issue1902_lasttoken_comparison.py``.  It refits the same ridge maps only to
materialize fold-held-out predictions, whitens predictions and answers using
target statistics fit on the training rows only, and ranks each prediction
against all answer vectors in its held-out fold using two-sided CSLS (k=10).
Exact ties receive mid-ranks, matching the standing representation-retrieval
convention.

The output reports pooled strict acc@1 and 95% semantic-cluster bootstrap
intervals.  No prompt-mean representation enters this analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE numpy/BLAS imports on the shared VM.
load_dotenv()

import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

import issue1902_lasttoken_comparison as C  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "eval_results" / "issue_1902" / "lasttoken_retrieval"
DEFAULT_FIGURE_DATA = (
    PROJECT_ROOT
    / "overleaf_section43"
    / "figures"
    / "paper"
    / "c1_posttraining_retrieval_data.json"
)
N_BOOT = 1_000
BOOT_SEED = 1_944
WHITEN_LAMBDA = 0.1
CSLS_K = 10


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(C._jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def cosine_ranks(
    prediction: np.ndarray, target: np.ndarray, *, block_size: int = 128
) -> np.ndarray:
    """Return strict cosine ranks with the project's mid-rank tie convention."""
    pred = np.array(prediction, dtype=np.float64, copy=True)
    pool = np.array(target, dtype=np.float64, copy=True)
    if pred.shape != pool.shape or pred.ndim != 2:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} vs {pool.shape}")
    pred /= np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12
    pool /= np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12
    n = len(pred)
    ranks = np.empty(n, dtype=np.float64)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        scores = pred[start:stop] @ pool.T
        local = np.arange(stop - start)
        true_col = np.arange(start, stop)
        true_score = scores[local, true_col]
        true_distance = 1.0 - true_score
        tolerance = 1e-9 * np.maximum(np.abs(true_distance), 1e-12)
        closer = (scores > true_score[:, None] + tolerance[:, None]).sum(axis=1)
        tied_other = (np.abs(scores - true_score[:, None]) <= tolerance[:, None]).sum(axis=1) - 1
        ranks[start:stop] = 1.0 + closer + 0.5 * tied_other
    return ranks


def shrunk_whitening_stats(
    target_train: np.ndarray, *, shrinkage: float = WHITEN_LAMBDA
) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-only mean and diagonal-target shrunk-covariance Cholesky."""
    target = np.asarray(target_train, dtype=np.float64)
    if target.ndim != 2 or len(target) < 2:
        raise ValueError(f"target_train must be (n>=2, d), got {target.shape}")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
    mean = target.mean(axis=0)
    centered = target - mean
    covariance = (centered.T @ centered) / (len(target) - 1)
    diagonal = np.diag(covariance).copy()
    covariance *= 1.0 - shrinkage
    covariance.flat[:: covariance.shape[0] + 1] += shrinkage * diagonal
    for jitter in (0.0, 1e-6, 1e-4, 1e-2):
        candidate = covariance.copy() if jitter else covariance
        if jitter:
            candidate.flat[:: candidate.shape[0] + 1] += jitter
        try:
            return mean, np.linalg.cholesky(candidate)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("shrunk covariance not positive definite")


def csls_scores(similarity: np.ndarray, *, k: int = CSLS_K) -> np.ndarray:
    """Exact two-sided cross-domain CSLS scores for a cosine matrix."""
    scores = np.asarray(similarity, dtype=np.float64)
    if scores.ndim != 2 or not 1 <= k <= min(scores.shape):
        raise ValueError(f"k={k} incompatible with similarity shape {scores.shape}")
    row_density = np.partition(scores, -k, axis=1)[:, -k:].mean(axis=1)
    pool_density = np.partition(scores, -k, axis=0)[-k:, :].mean(axis=0)
    return 2.0 * scores - row_density[:, None] - pool_density[None, :]


def whitened_csls_ranks(
    prediction: np.ndarray,
    target: np.ndarray,
    mean: np.ndarray,
    chol: np.ndarray,
    *,
    k: int = CSLS_K,
) -> np.ndarray:
    """Return strict mid-ranks under train-whitened cosine plus CSLS."""
    pred = np.asarray(prediction, dtype=np.float64)
    pool = np.asarray(target, dtype=np.float64)
    if pred.shape != pool.shape or pred.ndim != 2:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} vs {pool.shape}")

    def whiten(x: np.ndarray) -> np.ndarray:
        z = solve_triangular(
            chol,
            (x - np.asarray(mean, dtype=np.float64)).T,
            lower=True,
            check_finite=False,
        ).T
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-30)

    scores = csls_scores(whiten(pred) @ whiten(pool).T, k=k)
    true_score = scores[np.arange(len(scores)), np.arange(len(scores))]
    true_distance = -true_score
    tolerance = 1e-9 * np.maximum(np.abs(true_distance), 1e-12)
    closer = (scores > true_score[:, None] + tolerance[:, None]).sum(axis=1)
    tied_other = (np.abs(scores - true_score[:, None]) <= tolerance[:, None]).sum(axis=1) - 1
    return 1.0 + closer + 0.5 * tied_other


def cluster_bootstrap_acc1(
    success: np.ndarray,
    groups: list[str],
    *,
    counts: np.ndarray,
) -> tuple[float, list[float]]:
    """Pooled acc@1 and a cluster-bootstrap percentile interval."""
    success = np.asarray(success, dtype=np.float64)
    names = sorted(set(groups))
    if len(success) != len(groups) or counts.shape[1] != len(names):
        raise ValueError("success/group/bootstrap shape mismatch")
    lookup = {name: index for index, name in enumerate(names)}
    group_index = np.asarray([lookup[group] for group in groups], dtype=np.int64)
    success_sum = np.zeros(len(names), dtype=np.float64)
    group_size = np.zeros(len(names), dtype=np.float64)
    np.add.at(success_sum, group_index, success)
    np.add.at(group_size, group_index, 1.0)
    draws = (counts @ success_sum) / (counts @ group_size)
    return float(success.mean()), np.quantile(draws, [0.025, 0.975]).tolist()


def _fold_path(out_dir: Path, stage: str, fold: int) -> Path:
    return out_dir / "perfold_whitencsls" / f"{stage}_single_random_L{C.LAYER}_f{fold}.npz"


def _stage_worker(task: tuple[str, str, str, int, int, bool]) -> dict[str, Any]:
    stage_root_s, out_dir_s, stage, blas_threads, block_size, force = task
    stage_root, out_dir = Path(stage_root_s), Path(out_dir_s)
    from threadpoolctl import threadpool_limits

    x, y, ids = C._load_tensor_pair(stage_root, stage, "single", "u_last")
    group_ids, _groups = C._read_groups(stage_root, "single")
    if ids != group_ids:
        raise RuntimeError(f"tensor/group row-order drift for stage {stage}")
    reference_path = C.DEFAULT_OUT / "percell" / f"u_last_random_{stage}_single_L{C.LAYER}.npz"
    with np.load(reference_path, allow_pickle=False) as reference:
        fold_of = np.asarray(reference["fold_of"], dtype=np.int8)
        if len(fold_of) != len(ids):
            raise RuntimeError(f"random-fold row-count drift for stage {stage}")
        reference_residual = np.asarray(reference["ss_res"], dtype=np.float64)
        reference_lambda = np.asarray(reference["selected_lambda"], dtype=np.float64)

    rows = []
    for fold in range(C.N_FOLDS):
        eval_mask = fold_of == fold
        train_mask = ~eval_mask
        eval_index = np.flatnonzero(eval_mask)
        path = _fold_path(out_dir, stage, fold)
        if path.exists() and not force:
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(eval_index, saved["eval_index"]):
                    raise RuntimeError(f"resume row drift for {stage}/fold{fold}")
                raw_ranks = np.asarray(saved["raw_cosine_rank"], dtype=np.float64)
                ranks = np.asarray(saved["whitened_csls_rank"], dtype=np.float64)
                selected_lambda = float(saved["selected_lambda"])
                max_residual_delta = float(saved["max_abs_residual_delta"])
            resumed = True
        else:
            with threadpool_limits(limits=blas_threads):
                prediction, info = C.primal_gcv_predict(x[train_mask], y[train_mask], x[eval_mask])
                residual, _total, _cosine = C._per_row_components(
                    prediction, y[eval_mask], y[train_mask].mean(axis=0)
                )
                max_residual_delta = float(np.max(np.abs(residual - reference_residual[eval_mask])))
                if max_residual_delta > 1e-5:
                    raise RuntimeError(
                        f"fit parity failed for {stage}/fold{fold}: "
                        f"max residual delta {max_residual_delta:.3e}"
                    )
                selected_lambda = float(info["selected_lambda"])
                if not np.isclose(selected_lambda, reference_lambda[fold], rtol=0.0, atol=1e-12):
                    raise RuntimeError(f"lambda drift for {stage}/fold{fold}")
                raw_ranks = cosine_ranks(prediction, y[eval_mask], block_size=block_size)
                whiten_mean, whiten_chol = shrunk_whitening_stats(y[train_mask])
                ranks = whitened_csls_ranks(
                    prediction,
                    y[eval_mask],
                    whiten_mean,
                    whiten_chol,
                )
            C._savez(
                path,
                eval_index=eval_index,
                raw_cosine_rank=raw_ranks,
                whitened_csls_rank=ranks,
                whiten_lambda=np.asarray(WHITEN_LAMBDA),
                csls_k=np.asarray(CSLS_K),
                selected_lambda=np.asarray(selected_lambda),
                max_abs_residual_delta=np.asarray(max_residual_delta),
            )
            resumed = False
        rows.append(
            {
                "fold": fold,
                "n_eval": int(eval_mask.sum()),
                "n_train": int(train_mask.sum()),
                "whitened_csls_acc1": float(np.mean(ranks <= 1.0)),
                "raw_cosine_acc1": float(np.mean(raw_ranks <= 1.0)),
                "selected_lambda": selected_lambda,
                "max_abs_residual_delta": max_residual_delta,
                "resumed": resumed,
                "output": str(path),
            }
        )
        print(
            f"[{stage}/fold{fold}] whitened+CSLS acc@1="
            f"{rows[-1]['whitened_csls_acc1']:.6f} "
            f"n_pool={rows[-1]['n_eval']} resumed={resumed}",
            flush=True,
        )
    return {"stage": stage, "n": len(ids), "folds": rows}


def run(
    stage_root: Path,
    out_dir: Path,
    *,
    workers: int,
    blas_threads: int,
    block_size: int,
    force: bool,
) -> dict[str, Any]:
    tasks = [
        (str(stage_root), str(out_dir), stage, blas_threads, block_size, force)
        for stage in C.STAGES
    ]
    stage_details: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_stage_worker, task): task[2] for task in tasks}
        for future in as_completed(futures):
            detail = future.result()
            stage_details[detail["stage"]] = detail

    _ids, groups = C._read_groups(stage_root, "single")
    names = sorted(set(groups))
    rng = np.random.default_rng(BOOT_SEED)
    counts = rng.multinomial(len(names), np.full(len(names), 1.0 / len(names)), size=N_BOOT).astype(
        np.float64
    )
    cells: dict[str, Any] = {}
    for stage in C.STAGES:
        reference_path = C.DEFAULT_OUT / "percell" / f"u_last_random_{stage}_single_L{C.LAYER}.npz"
        with np.load(reference_path, allow_pickle=False) as reference:
            fold_of = np.asarray(reference["fold_of"], dtype=np.int8)
        ranks = np.full(len(groups), np.nan, dtype=np.float64)
        for fold in range(C.N_FOLDS):
            path = _fold_path(out_dir, stage, fold)
            with np.load(path, allow_pickle=False) as saved:
                ranks[saved["eval_index"]] = saved["whitened_csls_rank"]
        if not np.all(np.isfinite(ranks)):
            raise RuntimeError(f"incomplete ranks for stage {stage}")
        point, interval = cluster_bootstrap_acc1(ranks <= 1.0, groups, counts=counts)
        with np.load(reference_path, allow_pickle=False) as reference:
            r2 = 1.0 - float(reference["ss_res"].sum()) / float(reference["ss_tot"].sum())
        cells[stage] = {
            "n": len(ranks),
            "r2": r2,
            "whitened_csls_acc1": point,
            "whitened_csls_acc1_ci95": interval,
            "fold_pool_sizes": [int(np.sum(fold_of == fold)) for fold in range(C.N_FOLDS)],
            "folds": stage_details[stage]["folds"],
        }

    report = {
        "metadata": {
            "design": "OLMo-2 final-context-token strict retrieval",
            "hf_repo": C.HF_REPO,
            "hf_revision": C.HF_REVISION,
            "layer": C.LAYER,
            "corpus": "single",
            "context_summary": "u_last",
            "split": "six size-matched random row folds",
            "fitter": "ridge with the registered GCV grid",
            "retrieval": (
                "strict acc@1 under train-fold whitened cosine plus exact "
                f"two-sided CSLS k={CSLS_K}, against all answer vectors in each "
                "held-out fold; exact ties receive mid-ranks"
            ),
            "whitening": (
                "target mean and diagonal-target shrunk covariance fit on training "
                f"rows only, lambda={WHITEN_LAMBDA}; z=L^-1(v-mu)"
            ),
            "uncertainty": (f"{N_BOOT} semantic-cluster bootstrap draws, seed {BOOT_SEED}"),
            "n_groups": len(names),
        },
        "cells": cells,
    }
    _write_json(out_dir / "summary.json", report)
    return report


def figure_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Create the compact, deterministic input consumed by the paper figure."""
    metadata = report["metadata"]
    if metadata["context_summary"] != "u_last":
        raise ValueError("figure export requires final-prompt-token results")
    if metadata["split"] != "six size-matched random row folds":
        raise ValueError("figure export requires IID random-row folds")
    cells = {}
    for stage in C.STAGES:
        cell = report["cells"][stage]
        cells[stage] = {
            "whitened_csls_acc1": cell["whitened_csls_acc1"],
            "whitened_csls_acc1_ci95": cell["whitened_csls_acc1_ci95"],
            "fold_pool_sizes": cell["fold_pool_sizes"],
            "n": cell["n"],
            "r2": cell["r2"],
        }
    return {
        "metadata": {
            "analysis_script": "scripts/issue1902_lasttoken_retrieval.py",
            "bootstrap": f"{N_BOOT} semantic-cluster draws, seed {BOOT_SEED}",
            "context_summary": "final prompt token",
            "corpus": "single turn",
            "hf_revision": metadata["hf_revision"],
            "layer": metadata["layer"],
            "retrieval": (
                "strict acc@1 under train-fold whitened cosine plus exact "
                f"two-sided CSLS k={CSLS_K}, against all targets in each held-out "
                "fold; exact ties receive mid-ranks"
            ),
            "split": "six size-matched IID random-row folds",
            "whitening": (
                "target mean and diagonal-target shrunk covariance fit on training "
                f"rows only, lambda={WHITEN_LAMBDA}; z=L^-1(v-mu)"
            ),
        },
        "cells": cells,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=C.DEFAULT_STAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-data", type=Path, default=DEFAULT_FIGURE_DATA)
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="regenerate the compact figure input from the existing summary",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blas-threads", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.export_only:
        report = json.loads((args.out_dir / "summary.json").read_text())
    else:
        report = run(
            args.stage_root,
            args.out_dir,
            workers=args.workers,
            blas_threads=args.blas_threads,
            block_size=args.block_size,
            force=args.force,
        )
    _write_json(args.figure_data, figure_payload(report))
    print(
        json.dumps(
            {
                stage: {
                    "r2": report["cells"][stage]["r2"],
                    "whitened_csls_acc1": report["cells"][stage]["whitened_csls_acc1"],
                    "whitened_csls_acc1_ci95": report["cells"][stage]["whitened_csls_acc1_ci95"],
                }
                for stage in C.STAGES
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
