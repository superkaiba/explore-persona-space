#!/usr/bin/env python3
"""Matched OLMo-2 ``u_last`` versus ``u_mean`` context-to-answer maps.

This follow-up reuses issue #1902's banked activations, diagonal on-policy
targets, row order, semantic-group folds, headline layer, and ridge recipe.
The primary contrast changes exactly one thing: the context summary supplied
to ridge (prompt-token mean versus the last prompt token).

The script deliberately uses a primal spectral implementation for the full-N
fits.  It is algebraically equivalent to #1902's dual Gram implementation but
is practical on a CPU when n_train > hidden_dim.  ``parity`` checks that
implementation against #1902's committed u_mean per-row errors before any
u_last result is interpreted.

Phases:
  stage    Download only the requested layer's diagonal shards.
  stage_qwen  Stage the pinned Qwen train/validation/test activation bank.
  parity   Refit u_mean on fold 0 and compare with committed #1902 errors.
  fit      Fit u_last with the original semantic-group folds.
  random   Fit u_last with size-matched random row folds (split diagnostic).
  bridge   Qwen-style n=1000 / val=400 / test=1000 random-row ladder cell.
  exact_prompt_bridge  Compare on the exact Qwen/OLMo prompt intersection.
  reliability  Recompute OLMo target reliability with Qwen's exact formula.
  analyze  Aggregate paired cluster-bootstrap contrasts into summary.json.
  all      stage, parity, fit, random, bridge, analyze.

No model generation or activation capture occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE numpy/BLAS imports on the shared VM.
load_dotenv()

import numpy as np  # noqa: E402

STAGES = ("B", "S", "D", "R")
CORPORA = ("single", "multi")
LAYER = 31
CAPTURED_LAYERS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31)
N_FOLDS = 6
N_BOOT = 1000
BOOT_SEED = 42 + 1902
RANDOM_FOLD_SEED = 190231
BRIDGE_SPLIT_SEED = 42
BRIDGE_N_TRAIN = 1000
LARGE_BRIDGE_N_TRAIN = 13_000
BRIDGE_N_VAL = 400
BRIDGE_N_TEST = 1000
BRIDGE_DRAWS = (0, 1, 2)
GCV_LAMBDAS = np.logspace(-2, 4, 13)
QWEN_LAMBDAS = np.logspace(-3, 8, 23)

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REVISION = "3256c8efcef5f10ca525efeb2039636eaec8fad7"
HF_PREFIX = "issue1902_stage_map/analysis_tensors/issue1902_store"
NATIVE_STAGES = ("S", "D", "R")
NATIVE_N_TRAIN = 1000
NATIVE_N_VAL = 400
NATIVE_N_TEST = 600
DEFAULT_STAGE_ROOT = PROJECT_ROOT / "tmp" / "issue1902_lasttoken_store"
DEFAULT_OUT = PROJECT_ROOT / "eval_results" / "issue_1902" / "lasttoken_comparison"
COMMITTED_PERCELL = PROJECT_ROOT / "eval_results" / "issue_1902" / "fits" / "percell"
COMMITTED_GRID = PROJECT_ROOT / "eval_results" / "issue_1902" / "fits" / "grid_cells.json"
QWEN_LADDER = (
    PROJECT_ROOT / "eval_results" / "issue_1901" / "paper_densify" / "scaling_ladder_L19.json"
)
QWEN_HF_REVISION = "815ff6d976c686af8672b27cfdfb1ce6b419c02c"
QWEN_HF_PREFIX = "issue1491_scale_ladder/scale7_refit"
QWEN_LAYER = 19
PCA_RANKS = (64, 128, 256, 512, 768, 999)
DEFAULT_QWEN_ROOT = PROJECT_ROOT / "tmp" / "issue1902_qwen_raw"
CORPUS_SINGLE_PATH = Path("issue1902_stage_map/corpus/corpus_single.jsonl")
OLMO_OPERATOR = PROJECT_ROOT / "eval_results" / "issue_1902" / "operator" / "operator_battery.json"
ISSUE1902_RESULTS_COMMIT = "ed53c2292f3"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _qwen_n1000_draws(out_dir: Path) -> list[float]:
    """Read the committed Qwen anchor, with the prior analysis copy as fallback."""
    if QWEN_LADDER.exists():
        qwen = json.loads(QWEN_LADDER.read_text())
        draws = [
            float(cell["ridge"]["test_r2"])
            for cell in qwen["cells"]
            if int(cell["n_train"]) == BRIDGE_N_TRAIN
        ]
    else:
        prior = json.loads((out_dir / "summary.json").read_text())
        draws = [float(v) for v in prior["n1000_protocol_bridge"]["qwen_l19_n1000_draws"]]
    if len(draws) != len(BRIDGE_DRAWS):
        raise RuntimeError(f"expected {len(BRIDGE_DRAWS)} Qwen n=1000 draws, got {len(draws)}")
    return draws


def _savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def _store_root(stage_root: Path) -> Path:
    return stage_root / HF_PREFIX


def _ctx_path(stage_root: Path, stage: str, corpus: str, layer: int = LAYER) -> Path:
    return _store_root(stage_root) / stage / "ctx" / corpus / f"L{layer}.pt"


def _answer_path(stage_root: Path, stage: str, corpus: str, layer: int = LAYER) -> Path:
    return _store_root(stage_root) / stage / stage / corpus / f"L{layer}.pt"


def _row_index_path(stage_root: Path, corpus: str) -> Path:
    return _store_root(stage_root) / "B" / "ctx" / corpus / "row_index.jsonl"


def _hf_paths(layer: int, corpora: Iterable[str] = CORPORA) -> list[str]:
    paths: list[str] = []
    for stage in STAGES:
        for corpus in corpora:
            paths.extend(
                [
                    f"{HF_PREFIX}/{stage}/ctx/{corpus}/L{layer}.pt",
                    f"{HF_PREFIX}/{stage}/{stage}/{corpus}/L{layer}.pt",
                ]
            )
    for corpus in corpora:
        paths.append(f"{HF_PREFIX}/B/ctx/{corpus}/row_index.jsonl")
    return paths


def stage_files(
    stage_root: Path, layer: int = LAYER, corpora: Iterable[str] = CORPORA
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    got = []
    corpora = tuple(corpora)
    for filename in _hf_paths(layer, corpora):
        path = hf_hub_download(
            HF_REPO,
            filename,
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=stage_root,
        )
        got.append(str(Path(path).relative_to(stage_root)))
    return {
        "revision": HF_REVISION,
        "layer": layer,
        "corpora": list(corpora),
        "files": got,
    }


def _qwen_capture_dir(qwen_root: Path, split: str) -> Path:
    return qwen_root / QWEN_HF_PREFIX / split / "final_token_capture"


def stage_qwen_files(stage_root: Path, qwen_root: Path) -> dict[str, Any]:
    """Stage every Qwen file needed for exact-prompt and parity fits.

    The 25k training bank is about 2.1 GB because the published n=1,000
    draws are dispersed through every training chunk.  Files are revision
    pinned rather than read from the moving dataset head.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    qwen_paths: list[str] = []
    for split in ("train_25k", "val_400", "test_1000"):
        prefix = f"{QWEN_HF_PREFIX}/{split}/final_token_capture"
        qwen_paths.extend(
            sorted(
                item.path
                for item in api.list_repo_tree(
                    HF_REPO,
                    path_in_repo=prefix,
                    repo_type="dataset",
                    revision=QWEN_HF_REVISION,
                    recursive=True,
                )
                if getattr(item, "size", None) is not None and item.path.endswith(".pt")
            )
        )
    raw_prefix = f"{QWEN_HF_PREFIX}/test_1000/raw_completions"
    qwen_paths.extend(
        sorted(
            item.path
            for item in api.list_repo_tree(
                HF_REPO,
                path_in_repo=raw_prefix,
                repo_type="dataset",
                revision=QWEN_HF_REVISION,
                recursive=True,
            )
            if getattr(item, "size", None) is not None and item.path.endswith(".json")
        )
    )
    olmo_paths = [
        "issue1902_stage_map/corpus/corpus_single.jsonl",
        *[f"{HF_PREFIX}/{stage}/{stage}/single/row_index.jsonl" for stage in STAGES],
    ]
    got: dict[str, list[str]] = {"qwen": [], "olmo": []}
    for path in qwen_paths:
        downloaded = hf_hub_download(
            HF_REPO,
            path,
            repo_type="dataset",
            revision=QWEN_HF_REVISION,
            local_dir=qwen_root,
        )
        got["qwen"].append(str(Path(downloaded).relative_to(qwen_root)))
    for path in olmo_paths:
        downloaded = hf_hub_download(
            HF_REPO,
            path,
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=stage_root,
        )
        got["olmo"].append(str(Path(downloaded).relative_to(stage_root)))
    return {
        "qwen_revision": QWEN_HF_REVISION,
        "olmo_revision": HF_REVISION,
        "qwen_layer": QWEN_LAYER,
        "files": got,
    }


def stage_reliability_files(stage_root: Path) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    got = []
    for stage in STAGES:
        for seed in (43, 44):
            for layer in (18, 31):
                filename = f"{HF_PREFIX}/reliability/{stage}/single/seed{seed}/L{layer}.pt"
                path = hf_hub_download(
                    HF_REPO,
                    filename,
                    repo_type="dataset",
                    revision=HF_REVISION,
                    local_dir=stage_root,
                )
                got.append(str(Path(path).relative_to(stage_root)))
    return {"revision": HF_REVISION, "layers": [18, 31], "files": got}


def _load_tensor_pair(
    stage_root: Path, stage: str, corpus: str, summary: str, layer: int = LAYER
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import torch

    ctx = torch.load(
        _ctx_path(stage_root, stage, corpus, layer),
        map_location="cpu",
        weights_only=True,
    )
    ans = torch.load(
        _answer_path(stage_root, stage, corpus, layer),
        map_location="cpu",
        weights_only=True,
    )
    if summary not in ("u_last", "u_mean"):
        raise ValueError(f"unknown context summary {summary!r}")
    ids = [str(v) for v in ctx["row_ids"]]
    if ids != [str(v) for v in ans["row_ids"]]:
        raise RuntimeError(f"context/answer row mismatch: {stage}/{corpus}/L{layer}")
    return (
        ctx[summary].to(torch.float32).numpy(),
        ans["w"].to(torch.float32).numpy(),
        ids,
    )


def _committed_fold_rows(stage: str, corpus: str) -> list[np.ndarray]:
    folds = []
    for fold in range(N_FOLDS):
        path = COMMITTED_PERCELL / f"diag_{stage}_{corpus}_ctx_f{fold}.npz"
        with np.load(path) as payload:
            folds.append(np.asarray(payload["row_idx"], dtype=np.int64))
    flat = np.concatenate(folds)
    if len(np.unique(flat)) != len(flat) or not np.array_equal(np.sort(flat), np.arange(len(flat))):
        raise RuntimeError(f"committed folds do not partition rows: {stage}/{corpus}")
    ref = _committed_fold_rows("B", corpus) if stage != "B" else None
    if ref is not None and any(not np.array_equal(a, b) for a, b in zip(folds, ref, strict=True)):
        raise RuntimeError(f"stage-specific fold mismatch: {stage}/{corpus}")
    return folds


def _fold_assignment(stage: str, corpus: str, mode: str) -> np.ndarray:
    group_rows = _committed_fold_rows(stage, corpus)
    n = sum(len(v) for v in group_rows)
    if mode == "group":
        folds = group_rows
    elif mode == "random":
        rng = np.random.default_rng(RANDOM_FOLD_SEED + (0 if corpus == "single" else 1))
        perm = rng.permutation(n)
        folds = []
        start = 0
        for ref in group_rows:
            folds.append(np.sort(perm[start : start + len(ref)]))
            start += len(ref)
    else:
        raise ValueError(f"unknown fold mode {mode!r}")
    fold_of = np.full(n, -1, dtype=np.int8)
    for fold, rows in enumerate(folds):
        fold_of[rows] = fold
    if np.any(fold_of < 0):
        raise RuntimeError(f"incomplete {mode} fold assignment for {stage}/{corpus}")
    return fold_of


def pooled_r2(pred: np.ndarray, target: np.ndarray, center: np.ndarray | None = None) -> float:
    pred64 = np.asarray(pred, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    mu = target64.mean(axis=0) if center is None else np.asarray(center, dtype=np.float64)
    ss_res = float(np.square(target64 - pred64).sum())
    ss_tot = float(np.square(target64 - mu).sum())
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def primal_gcv_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    lambdas: np.ndarray = GCV_LAMBDAS,
) -> tuple[np.ndarray, dict[str, float]]:
    """Exact #1902 ridge/GCV predictions via a primal eigendecomposition.

    The original fitter diagonalizes X X^T.  For n_train > d this function
    diagonalizes X^T X and uses U^T Y = S^-1 V^T X^T Y, preserving the same
    non-zero spectrum, GCV residual, degrees of freedom, and predictions.
    """
    from scipy.linalg import eigh

    xtr = np.asarray(x_train, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64)
    xev = np.asarray(x_eval, dtype=np.float64)
    n_train, dim = xtr.shape
    if n_train <= dim:
        raise ValueError(f"primal GCV requires n_train > d; got {n_train=} {dim=}")

    xmu = xtr.mean(axis=0)
    xsd = xtr.std(axis=0, ddof=0) + 1e-9
    xtr = (xtr - xmu) / xsd
    xev = (xev - xmu) / xsd
    ymu = ytr.mean(axis=0)
    ytr = ytr - ymu

    xtx = xtr.T @ xtr
    xty = xtr.T @ ytr
    eigvals, eigvecs = eigh(xtx, overwrite_a=True, check_finite=False, driver="evd")
    eigvals = np.maximum(eigvals, 0.0)
    vt_xty = eigvecs.T @ xty
    positive = eigvals > np.finfo(np.float64).eps * max(n_train, dim) * max(eigvals[-1], 1.0)
    uty_sq = np.zeros_like(eigvals)
    uty_sq[positive] = np.square(vt_xty[positive] / np.sqrt(eigvals[positive, None])).sum(axis=1)
    total = float(np.square(ytr).sum())
    gcv = []
    dofs = []
    for lam in np.asarray(lambdas, dtype=np.float64):
        filt = eigvals / (eigvals + float(lam))
        rss = total - float(((2.0 * filt - np.square(filt)) * uty_sq).sum())
        dof = float(filt.sum())
        denom = (n_train - dof) ** 2
        gcv.append(float("inf") if denom <= 1e-12 else rss / denom)
        dofs.append(dof)
    best_idx = int(np.argmin(gcv))
    best_lam = float(lambdas[best_idx])
    weights = eigvecs @ (vt_xty / (eigvals + best_lam)[:, None])
    pred = xev @ weights + ymu
    return pred, {
        "selected_lambda": best_lam,
        "dof": dofs[best_idx],
        "gcv": gcv[best_idx],
    }


def dual_val_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    lambdas: np.ndarray = QWEN_LAMBDAS,
    *,
    std_ddof: int = 0,
) -> tuple[np.ndarray, dict[str, float | str | None]]:
    """Qwen-ladder-style validation-selected ridge for n_train < d.

    Validation SSE is evaluated analytically for every lambda after one dual
    eigendecomposition; only the selected test prediction is materialized.
    """
    from scipy.linalg import eigh

    xtr = np.asarray(x_train, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64)
    xval = np.asarray(x_val, dtype=np.float64)
    yval = np.asarray(y_val, dtype=np.float64)
    xtest = np.asarray(x_test, dtype=np.float64)
    xmu = xtr.mean(axis=0)
    xsd = xtr.std(axis=0, ddof=std_ddof) + 1e-9
    xtr = (xtr - xmu) / xsd
    xval = (xval - xmu) / xsd
    xtest = (xtest - xmu) / xsd
    ymu = ytr.mean(axis=0)
    ytr_c = ytr - ymu
    yval_c = yval - ymu

    gram = xtr @ xtr.T
    eigvals, eigvecs = eigh(gram, overwrite_a=True, check_finite=False, driver="evd")
    eigvals = np.maximum(eigvals, 0.0)
    uty = eigvecs.T @ ytr_c
    aval = (xval @ xtr.T) @ eigvecs
    # ||Y - A diag(g) B||_F^2 as a quadratic in g.
    cross = np.sum((aval.T @ yval_c) * uty, axis=1)
    quadratic = (aval.T @ aval) * (uty @ uty.T)
    y_norm = float(np.square(yval_c).sum())
    val_tot = float(np.square(yval - yval.mean(axis=0)).sum())
    val_r2 = []
    for lam in np.asarray(lambdas, dtype=np.float64):
        inv = 1.0 / (eigvals + float(lam))
        sse = y_norm - 2.0 * float(inv @ cross) + float(inv @ quadratic @ inv)
        val_r2.append(float("nan") if val_tot <= 0 else 1.0 - sse / val_tot)
    finite = np.where(np.isfinite(val_r2), val_r2, -np.inf)
    best_idx = int(np.argmax(finite))
    best_lam = float(lambdas[best_idx])
    atest = (xtest @ xtr.T) @ eigvecs
    pred = (atest / (eigvals + best_lam)) @ uty + ymu
    edge = "low" if best_idx == 0 else ("high" if best_idx == len(lambdas) - 1 else None)
    return pred, {
        "selected_lambda": best_lam,
        "val_r2_at_selected": float(val_r2[best_idx]),
        "lambda_grid_edge": edge,
        "standardization_ddof": std_ddof,
    }


def primal_val_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    lambdas: np.ndarray = QWEN_LAMBDAS,
    *,
    std_ddof: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validation-selected multi-output ridge for ``n_train > d``.

    This is the primal counterpart of :func:`dual_val_ridge_predict`.  It uses
    the same training-only standardization, unscaled centered target, lambda
    grid, pooled validation R2 selection, and training-target-mean intercept.
    Diagonalizing ``X.T @ X`` keeps the large-n bridge tractable without
    changing the ridge solution.
    """
    from scipy.linalg import eigh

    xtr = np.asarray(x_train, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64)
    xval = np.asarray(x_val, dtype=np.float64)
    yval = np.asarray(y_val, dtype=np.float64)
    xtest = np.asarray(x_test, dtype=np.float64)
    n_train, dim = xtr.shape
    if n_train <= dim:
        raise ValueError(f"primal validation ridge requires n_train > d; got {n_train=} {dim=}")

    xmu = xtr.mean(axis=0)
    xsd = xtr.std(axis=0, ddof=std_ddof) + 1e-9
    xtr = (xtr - xmu) / xsd
    xval = (xval - xmu) / xsd
    xtest = (xtest - xmu) / xsd
    ymu = ytr.mean(axis=0)
    ytr = ytr - ymu

    xtx = xtr.T @ xtr
    xty = xtr.T @ ytr
    eigvals, eigvecs = eigh(xtx, overwrite_a=True, check_finite=False, driver="evd")
    eigvals = np.maximum(eigvals, 0.0)
    projected_y = eigvecs.T @ xty
    projected_val = xval @ eigvecs
    val_r2 = []
    for lam in np.asarray(lambdas, dtype=np.float64):
        pred_val = (projected_val / (eigvals + float(lam))) @ projected_y + ymu
        val_r2.append(pooled_r2(pred_val, yval))
    finite = np.where(np.isfinite(val_r2), val_r2, -np.inf)
    best_idx = int(np.argmax(finite))
    best_lam = float(lambdas[best_idx])
    projected_test = xtest @ eigvecs
    pred = (projected_test / (eigvals + best_lam)) @ projected_y + ymu
    edge = "low" if best_idx == 0 else ("high" if best_idx == len(lambdas) - 1 else None)
    return pred, {
        "selected_lambda": best_lam,
        "val_r2_at_selected": float(val_r2[best_idx]),
        "validation_r2_by_lambda": [float(v) for v in val_r2],
        "lambda_grid_edge": edge,
        "standardization_ddof": std_ddof,
        "solver": "primal_eigh",
        "effective_dof": float(np.sum(eigvals / (eigvals + best_lam))),
    }


def dual_val_pca_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    lambdas: np.ndarray = QWEN_LAMBDAS,
    ranks: Iterable[int] = PCA_RANKS,
    *,
    std_ddof: int = 1,
    include_full_prediction: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validation-select a hard input-PCA rank and ridge penalty.

    PCA is fit only on standardized training contexts.  Targets stay in their
    full hidden space, so the reported test R2 remains directly comparable to
    ordinary ridge.  The computation uses the training Gram eigensystem; this
    is exactly the primal PCA subspace without materializing its d-dimensional
    loading matrix.
    """
    from scipy.linalg import eigh

    xtr = np.asarray(x_train, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64)
    xval = np.asarray(x_val, dtype=np.float64)
    yval = np.asarray(y_val, dtype=np.float64)
    xtest = np.asarray(x_test, dtype=np.float64)
    xmu = xtr.mean(axis=0)
    xsd = xtr.std(axis=0, ddof=std_ddof) + 1e-9
    xtr = (xtr - xmu) / xsd
    xval = (xval - xmu) / xsd
    xtest = (xtest - xmu) / xsd
    ymu = ytr.mean(axis=0)
    ytr_c = ytr - ymu
    yval_c = yval - ymu

    gram = xtr @ xtr.T
    eigvals, eigvecs = eigh(gram, overwrite_a=True, check_finite=False, driver="evd")
    eigvals = np.maximum(eigvals, 0.0)
    uty = eigvecs.T @ ytr_c
    aval = (xval @ xtr.T) @ eigvecs
    cross = np.sum((aval.T @ yval_c) * uty, axis=1)
    quadratic = (aval.T @ aval) * (uty @ uty.T)
    y_norm = float(np.square(yval_c).sum())
    val_tot = float(np.square(yval - yval.mean(axis=0)).sum())
    usable_ranks = sorted({min(int(rank), len(xtr) - 1) for rank in ranks if int(rank) > 0})
    if not usable_ranks:
        raise ValueError("PCA rank grid is empty")

    best: tuple[float, int, int] | None = None
    full_best: tuple[float, int] | None = None
    scores: dict[str, float] = {}
    lambda_values = np.asarray(lambdas, dtype=np.float64)
    for rank in usable_ranks:
        sl = slice(len(xtr) - rank, len(xtr))
        eig = eigvals[sl]
        cr = cross[sl]
        quad = quadratic[sl, sl]
        for li, lam in enumerate(lambda_values):
            inv = 1.0 / (eig + float(lam))
            sse = y_norm - 2.0 * float(inv @ cr) + float(inv @ quad @ inv)
            score = float("nan") if val_tot <= 0 else 1.0 - sse / val_tot
            scores[f"k{rank}_lambda{float(lam):g}"] = score
            candidate = (-np.inf if not np.isfinite(score) else score, -rank, -li)
            if best is None or candidate > best:
                best = candidate
            if rank == usable_ranks[-1]:
                full_candidate = (-np.inf if not np.isfinite(score) else score, -li)
                if full_best is None or full_candidate > full_best:
                    full_best = full_candidate
    assert best is not None
    best_score, neg_rank, neg_li = best
    best_rank, best_li = -neg_rank, -neg_li
    best_lam = float(lambda_values[best_li])
    sl = slice(len(xtr) - best_rank, len(xtr))
    atest = (xtest @ xtr.T) @ eigvecs[:, sl]
    pred = (atest / (eigvals[sl] + best_lam)) @ uty[sl] + ymu
    info: dict[str, Any] = {
        "selected_rank": best_rank,
        "selected_lambda": best_lam,
        "val_r2_at_selected": float(best_score),
        "rank_grid": usable_ranks,
        "rank_grid_edge": "low"
        if best_rank == usable_ranks[0]
        else ("high" if best_rank == usable_ranks[-1] else None),
        "lambda_grid_edge": "low"
        if best_li == 0
        else ("high" if best_li == len(lambda_values) - 1 else None),
        "standardization_ddof": std_ddof,
        "n_candidates": len(scores),
    }
    if include_full_prediction:
        assert full_best is not None
        full_score, neg_full_li = full_best
        full_li = -neg_full_li
        full_rank = usable_ranks[-1]
        full_sl = slice(len(xtr) - full_rank, len(xtr))
        full_lam = float(lambda_values[full_li])
        full_atest = (xtest @ xtr.T) @ eigvecs[:, full_sl]
        info["_full_rank_prediction"] = (full_atest / (eigvals[full_sl] + full_lam)) @ uty[
            full_sl
        ] + ymu
        info["full_rank_ridge"] = {
            "selected_rank": full_rank,
            "selected_lambda": full_lam,
            "val_r2_at_selected": float(full_score),
            "lambda_grid_edge": "low"
            if full_li == 0
            else ("high" if full_li == len(lambda_values) - 1 else None),
            "standardization_ddof": std_ddof,
        }
    return pred, info


def _per_row_components(
    pred: np.ndarray, target: np.ndarray, train_mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred64 = np.asarray(pred, dtype=np.float64)
    target_raw = np.asarray(target)
    center_raw = np.asarray(train_mean)
    target64 = np.asarray(target_raw, dtype=np.float64)
    res = np.square(target64 - pred64).sum(axis=1)
    # Match #1902's committed reduction exactly: its Y arrays and fold-train
    # mean remain float32 for SS_tot, while prediction residuals promote to
    # float64 because the ridge prediction is float64.
    tot = np.asarray(np.square(target_raw - center_raw).sum(axis=1), dtype=np.float64)
    cos = np.sum(target64 * pred64, axis=1) / (
        np.linalg.norm(target64, axis=1) * np.linalg.norm(pred64, axis=1) + 1e-12
    )
    return res, tot, cos


def _cell_output(out_dir: Path, summary: str, fold_mode: str, stage: str, corpus: str) -> Path:
    return out_dir / "percell" / f"{summary}_{fold_mode}_{stage}_{corpus}_L{LAYER}.npz"


def _fit_cell_worker(task: tuple[str, str, str, str, str, str, int]) -> dict[str, Any]:
    stage_root_s, out_dir_s, stage, corpus, summary, fold_mode, blas_threads = task
    stage_root, out_dir = Path(stage_root_s), Path(out_dir_s)
    from threadpoolctl import threadpool_limits

    t0 = time.time()
    with threadpool_limits(limits=blas_threads):
        x, y, ids = _load_tensor_pair(stage_root, stage, corpus, summary)
        fold_of = _fold_assignment(stage, corpus, fold_mode)
        n = len(x)
        res = np.full(n, np.nan, dtype=np.float64)
        tot = np.full(n, np.nan, dtype=np.float64)
        cos = np.full(n, np.nan, dtype=np.float64)
        selected = np.full(N_FOLDS, np.nan, dtype=np.float64)
        dof = np.full(N_FOLDS, np.nan, dtype=np.float64)
        n_train = np.zeros(N_FOLDS, dtype=np.int64)
        n_eval = np.zeros(N_FOLDS, dtype=np.int64)
        for fold in range(N_FOLDS):
            ev = fold_of == fold
            tr = ~ev
            pred, info = primal_gcv_predict(x[tr], y[tr], x[ev])
            rr, tt, cc = _per_row_components(pred, y[ev], y[tr].mean(axis=0))
            res[ev], tot[ev], cos[ev] = rr, tt, cc
            selected[fold] = info["selected_lambda"]
            dof[fold] = info["dof"]
            n_train[fold], n_eval[fold] = int(tr.sum()), int(ev.sum())
        if not np.all(np.isfinite(res)):
            raise RuntimeError(f"non-finite OOF components: {summary}/{fold_mode}/{stage}/{corpus}")
        out_path = _cell_output(out_dir, summary, fold_mode, stage, corpus)
        _savez(
            out_path,
            row_ids=np.asarray(ids),
            fold_of=fold_of,
            ss_res=res,
            ss_tot=tot,
            cos=cos,
            selected_lambda=selected,
            dof=dof,
            n_train=n_train,
            n_eval=n_eval,
        )
    return {
        "stage": stage,
        "corpus": corpus,
        "summary": summary,
        "fold_mode": fold_mode,
        "r2": 1.0 - float(res.sum()) / float(tot.sum()),
        "selected_lambda": selected.tolist(),
        "dof": dof.tolist(),
        "wall_s": time.time() - t0,
        "output": str(out_path),
    }


def fit_cells(
    stage_root: Path,
    out_dir: Path,
    *,
    summary: str,
    fold_mode: str,
    workers: int,
    blas_threads: int,
    force: bool,
) -> list[dict[str, Any]]:
    tasks = []
    completed = []
    for stage in STAGES:
        for corpus in CORPORA:
            path = _cell_output(out_dir, summary, fold_mode, stage, corpus)
            if path.exists() and not force:
                with np.load(path) as payload:
                    completed.append(
                        {
                            "stage": stage,
                            "corpus": corpus,
                            "summary": summary,
                            "fold_mode": fold_mode,
                            "r2": 1.0
                            - float(payload["ss_res"].sum()) / float(payload["ss_tot"].sum()),
                            "selected_lambda": payload["selected_lambda"].tolist(),
                            "dof": payload["dof"].tolist(),
                            "wall_s": 0.0,
                            "output": str(path),
                            "resumed": True,
                        }
                    )
                continue
            tasks.append(
                (
                    str(stage_root),
                    str(out_dir),
                    stage,
                    corpus,
                    summary,
                    fold_mode,
                    blas_threads,
                )
            )
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fit_cell_worker, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(
                    f"[{summary}/{fold_mode}] {result['stage']}/{result['corpus']} "
                    f"R2={result['r2']:.6f} wall={result['wall_s']:.1f}s",
                    flush=True,
                )
    completed.sort(key=lambda r: (CORPORA.index(r["corpus"]), STAGES.index(r["stage"])))
    _write_json(out_dir / f"fit_{summary}_{fold_mode}.json", {"cells": completed})
    return completed


def _committed_mean_components(stage: str, corpus: str) -> tuple[np.ndarray, np.ndarray]:
    fold_rows = _committed_fold_rows(stage, corpus)
    n = sum(len(v) for v in fold_rows)
    res = np.full(n, np.nan)
    tot = np.full(n, np.nan)
    for fold, rows in enumerate(fold_rows):
        with np.load(COMMITTED_PERCELL / f"diag_{stage}_{corpus}_ctx_f{fold}.npz") as payload:
            layers = payload["layers"].tolist()
            li = layers.index(LAYER)
            if not np.array_equal(rows, payload["row_idx"]):
                raise RuntimeError(f"row-index drift in committed {stage}/{corpus}/fold{fold}")
            res[rows] = payload["ss_res"][li]
            tot[rows] = payload["ss_tot"][li]
    if not np.all(np.isfinite(res)):
        raise RuntimeError(f"incomplete committed components for {stage}/{corpus}")
    return res, tot


def parity_check(
    stage_root: Path, out_dir: Path, *, blas_threads: int, force: bool = False
) -> dict[str, Any]:
    path = out_dir / "primal_mean_parity.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    rows = []
    from threadpoolctl import threadpool_limits

    for stage, corpus in (("B", "single"), ("R", "multi")):
        x, y, _ = _load_tensor_pair(stage_root, stage, corpus, "u_mean")
        fold_rows = _committed_fold_rows(stage, corpus)
        ev_idx = fold_rows[0]
        ev = np.zeros(len(x), dtype=bool)
        ev[ev_idx] = True
        with threadpool_limits(limits=blas_threads):
            pred, info = primal_gcv_predict(x[~ev], y[~ev], x[ev])
        res, tot, _ = _per_row_components(pred, y[ev], y[~ev].mean(axis=0))
        with np.load(COMMITTED_PERCELL / f"diag_{stage}_{corpus}_ctx_f0.npz") as committed:
            li = committed["layers"].tolist().index(LAYER)
            ref_res = committed["ss_res"][li]
            ref_tot = committed["ss_tot"][li]
        got_r2 = 1.0 - float(res.sum()) / float(tot.sum())
        ref_r2 = 1.0 - float(ref_res.sum()) / float(ref_tot.sum())
        rows.append(
            {
                "stage": stage,
                "corpus": corpus,
                "fold": 0,
                "r2_primal": got_r2,
                "r2_committed_dual": ref_r2,
                "abs_r2_delta": abs(got_r2 - ref_r2),
                "max_abs_per_row_res_delta": float(np.max(np.abs(res - ref_res))),
                "max_abs_per_row_tot_delta": float(np.max(np.abs(tot - ref_tot))),
                **info,
            }
        )
    report = {
        "tolerance_r2": 1e-8,
        "pass": all(r["abs_r2_delta"] <= 1e-8 for r in rows),
        "slices": rows,
    }
    _write_json(path, report)
    if not report["pass"]:
        raise RuntimeError(f"primal/committed parity failed: {rows}")
    return report


def _read_groups(stage_root: Path, corpus: str) -> tuple[list[str], list[str]]:
    rows = []
    with _row_index_path(stage_root, corpus).open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    return [str(r["id"]) for r in rows], [str(r["group"]) for r in rows]


def _group_sums(values: np.ndarray, groups: list[str]) -> tuple[np.ndarray, list[str]]:
    names = sorted(set(groups))
    lookup = {name: idx for idx, name in enumerate(names)}
    out = np.zeros(len(names), dtype=np.float64)
    np.add.at(
        out,
        np.asarray([lookup[g] for g in groups]),
        np.asarray(values, dtype=np.float64),
    )
    return out, names


def paired_group_bootstrap(
    res_a: np.ndarray,
    tot_a: np.ndarray,
    res_b: np.ndarray,
    tot_b: np.ndarray,
    groups: list[str],
    *,
    counts: np.ndarray,
) -> dict[str, Any]:
    ra, names = _group_sums(res_a, groups)
    ta, names2 = _group_sums(tot_a, groups)
    rb, names3 = _group_sums(res_b, groups)
    tb, names4 = _group_sums(tot_b, groups)
    if not (names == names2 == names3 == names4) or counts.shape[1] != len(names):
        raise RuntimeError("paired bootstrap group mismatch")
    qa = 1.0 - (counts @ ra) / (counts @ ta)
    qb = 1.0 - (counts @ rb) / (counts @ tb)
    delta = qa - qb
    return {
        "a_ci": np.quantile(qa, [0.025, 0.975]).tolist(),
        "b_ci": np.quantile(qb, [0.025, 0.975]).tolist(),
        "delta_ci": np.quantile(delta, [0.025, 0.975]).tolist(),
        "p_delta_le_zero": float(np.mean(delta <= 0)),
    }


def _bridge_split(
    n: int, candidates: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.arange(n) if candidates is None else np.asarray(candidates, dtype=np.int64)
    if len(candidates) < BRIDGE_N_TEST + BRIDGE_N_VAL + BRIDGE_N_TRAIN:
        raise ValueError(f"not enough rows for bridge: {len(candidates)}")
    perm = np.random.default_rng(BRIDGE_SPLIT_SEED).permutation(candidates)
    test = np.sort(perm[:BRIDGE_N_TEST])
    val = np.sort(perm[BRIDGE_N_TEST : BRIDGE_N_TEST + BRIDGE_N_VAL])
    pool = np.sort(perm[BRIDGE_N_TEST + BRIDGE_N_VAL :])
    return pool, val, test


def _bridge_cell_worker(
    task: tuple[str, str, str, int],
) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
    stage_root_s, stage, summary, blas_threads = task
    from threadpoolctl import threadpool_limits

    stage_root = Path(stage_root_s)
    x, y, ids = _load_tensor_pair(stage_root, stage, "single", summary)
    pool, val, test = _bridge_split(len(x))
    rows = []
    arrays: dict[str, np.ndarray] = {}
    for draw in BRIDGE_DRAWS:
        rng = np.random.default_rng(19010000 + BRIDGE_N_TRAIN * 10 + draw)
        train = np.sort(rng.choice(pool, size=BRIDGE_N_TRAIN, replace=False))
        with threadpool_limits(limits=blas_threads):
            pred, info = dual_val_ridge_predict(
                x[train], y[train], x[val], y[val], x[test], QWEN_LAMBDAS
            )
        test_r2 = pooled_r2(pred, y[test])
        arrays[f"{stage}_{summary}_draw{draw}_res"] = np.square(
            y[test].astype(np.float64) - pred
        ).sum(axis=1)
        rows.append({"draw": draw, "test_r2": test_r2, **info})
    vals = np.asarray([r["test_r2"] for r in rows])
    key = f"{stage}_{summary}"
    cell = {
        "draws": rows,
        "mean_test_r2": float(vals.mean()),
        "sd_test_r2": float(vals.std(ddof=1)),
        "min_test_r2": float(vals.min()),
        "max_test_r2": float(vals.max()),
        "test_row_ids_sha256": hashlib.sha256("\n".join(ids[i] for i in test).encode()).hexdigest(),
    }
    return key, cell, arrays


def run_bridge(
    stage_root: Path,
    out_dir: Path,
    *,
    workers: int,
    blas_threads: int,
    force: bool = False,
) -> dict[str, Any]:
    result_path = out_dir / "bridge_n1000.json"
    arrays_path = out_dir / "bridge_n1000_perrow.npz"
    if result_path.exists() and arrays_path.exists() and not force:
        return json.loads(result_path.read_text())
    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    tasks = [
        (str(stage_root), stage, summary, blas_threads)
        for stage in STAGES
        for summary in ("u_mean", "u_last")
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool_executor:
        futures = {pool_executor.submit(_bridge_cell_worker, task): task for task in tasks}
        for future in as_completed(futures):
            key, cell, cell_arrays = future.result()
            cells[key] = cell
            arrays.update(cell_arrays)
            print(f"[bridge] {key} R2={cell['mean_test_r2']:.6f}", flush=True)
    _savez(arrays_path, **arrays)
    report = {
        "protocol": {
            "corpus": "single",
            "layer": LAYER,
            "split": "fixed random rows",
            "n_train": BRIDGE_N_TRAIN,
            "n_val": BRIDGE_N_VAL,
            "n_test": BRIDGE_N_TEST,
            "split_seed": BRIDGE_SPLIT_SEED,
            "draws": list(BRIDGE_DRAWS),
            "train_seed_formula": "19010000 + n_train*10 + draw",
            "lambda_grid": QWEN_LAMBDAS.tolist(),
            "selection": "validation pooled R2; test pooled R2 uses test-set own mean",
        },
        "cells": cells,
    }
    _write_json(result_path, report)
    return report


def _layer_bridge_cell_worker(
    task: tuple[str, str, str, int, int, bool],
) -> tuple[str, dict[str, Any]]:
    stage_root_s, stage, summary, layer, blas_threads, generic_only = task
    from threadpoolctl import threadpool_limits

    stage_root = Path(stage_root_s)
    x, y, ids = _load_tensor_pair(stage_root, stage, "single", summary, layer)
    candidates = None
    if generic_only:
        with _row_index_path(stage_root, "single").open() as fh:
            classes = [json.loads(line).get("class") or "generic" for line in fh]
        candidates = np.flatnonzero(np.asarray(classes) == "generic")
    pool, val, test = _bridge_split(len(x), candidates)
    rows = []
    for draw in BRIDGE_DRAWS:
        rng = np.random.default_rng(19010000 + BRIDGE_N_TRAIN * 10 + draw)
        train = np.sort(rng.choice(pool, size=BRIDGE_N_TRAIN, replace=False))
        with threadpool_limits(limits=blas_threads):
            pred, info = dual_val_ridge_predict(
                x[train], y[train], x[val], y[val], x[test], QWEN_LAMBDAS
            )
        rows.append(
            {
                "draw": draw,
                "val_r2": info["val_r2_at_selected"],
                "test_r2": pooled_r2(pred, y[test]),
                "selected_lambda": info["selected_lambda"],
                "lambda_grid_edge": info["lambda_grid_edge"],
            }
        )
    key = f"{stage}_{summary}_L{layer}"
    return key, {
        "stage": stage,
        "summary": summary,
        "layer": layer,
        "draws": rows,
        "mean_val_r2": float(np.mean([r["val_r2"] for r in rows])),
        "mean_test_r2": float(np.mean([r["test_r2"] for r in rows])),
        "test_row_ids_sha256": hashlib.sha256("\n".join(ids[i] for i in test).encode()).hexdigest(),
    }


def run_layer_bridge(
    stage_root: Path,
    out_dir: Path,
    *,
    workers: int,
    blas_threads: int,
    generic_only: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    suffix = "_generic" if generic_only else ""
    result_path = out_dir / f"bridge_n1000_layer_sweep{suffix}.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    cells: dict[str, Any] = {}
    tasks = [
        (str(stage_root), stage, summary, layer, blas_threads, generic_only)
        for stage in STAGES
        for summary in ("u_mean", "u_last")
        for layer in CAPTURED_LAYERS
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool_executor:
        futures = {pool_executor.submit(_layer_bridge_cell_worker, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            key, cell = future.result()
            cells[key] = cell
            if done % 8 == 0 or done == len(tasks):
                print(f"[layer_bridge] {done}/{len(tasks)} cells", flush=True)

    selection: dict[str, Any] = {}
    for summary in ("u_mean", "u_last"):
        val_by_layer = {
            layer: float(
                np.mean([cells[f"{stage}_{summary}_L{layer}"]["mean_val_r2"] for stage in STAGES])
            )
            for layer in CAPTURED_LAYERS
        }
        selected = max(CAPTURED_LAYERS, key=lambda layer: (val_by_layer[layer], -layer))
        selection[summary] = {
            "selected_layer": selected,
            "mean_val_r2_by_layer": val_by_layer,
            "test_r2_at_selected": {
                stage: cells[f"{stage}_{summary}_L{selected}"]["mean_test_r2"] for stage in STAGES
            },
            "mean_test_r2_at_selected": float(
                np.mean(
                    [cells[f"{stage}_{summary}_L{selected}"]["mean_test_r2"] for stage in STAGES]
                )
            ),
            "mean_test_r2_at_L31": float(
                np.mean([cells[f"{stage}_{summary}_L31"]["mean_test_r2"] for stage in STAGES])
            ),
        }
    report = {
        "protocol": {
            "corpus": "single",
            "row_filter": "class == generic" if generic_only else "all intersection rows",
            "layers": list(CAPTURED_LAYERS),
            "n_train": BRIDGE_N_TRAIN,
            "n_val": BRIDGE_N_VAL,
            "n_test": BRIDGE_N_TEST,
            "draws": list(BRIDGE_DRAWS),
            "selection": "argmax mean validation R2 over four stages and three training draws; test untouched",
            "lambda_grid": QWEN_LAMBDAS.tolist(),
        },
        "selection": selection,
        "cells": cells,
    }
    _write_json(result_path, report)
    return report


def _native_paths(stage_root: Path, stage: str, layer: int) -> tuple[Path, Path]:
    root = _store_root(stage_root) / "robust_native" / stage / "single"
    return root / "ctx" / f"L{layer}.pt", root / f"L{layer}.pt"


def _native_bridge_cell_worker(
    task: tuple[str, str, int, int],
) -> tuple[str, dict[str, Any]]:
    stage_root_s, stage, layer, blas_threads = task
    from threadpoolctl import threadpool_limits
    import torch

    stage_root = Path(stage_root_s)
    native_ctx_path, native_answer_path = _native_paths(stage_root, stage, layer)
    native_ctx = torch.load(native_ctx_path, map_location="cpu", weights_only=True)
    native_answer = torch.load(native_answer_path, map_location="cpu", weights_only=True)
    plain_ctx = torch.load(
        _ctx_path(stage_root, stage, "single", layer),
        map_location="cpu",
        weights_only=True,
    )
    plain_answer = torch.load(
        _answer_path(stage_root, stage, "single", layer),
        map_location="cpu",
        weights_only=True,
    )
    native_ids = [str(v) for v in native_ctx["row_ids"]]
    if native_ids != [str(v) for v in native_answer["row_ids"]]:
        raise RuntimeError(f"native context/answer mismatch: {stage}/L{layer}")
    plain_pos = {str(row_id): idx for idx, row_id in enumerate(plain_ctx["row_ids"])}
    try:
        pos = np.asarray([plain_pos[row_id] for row_id in native_ids], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"native row missing from plain store: {stage}/L{layer}") from exc
    if [str(plain_answer["row_ids"][idx]) for idx in pos] != native_ids:
        raise RuntimeError(f"plain context/answer mismatch on native subset: {stage}/L{layer}")

    perm = np.random.default_rng(BRIDGE_SPLIT_SEED).permutation(len(native_ids))
    test = np.sort(perm[:NATIVE_N_TEST])
    val = np.sort(perm[NATIVE_N_TEST : NATIVE_N_TEST + NATIVE_N_VAL])
    train = np.sort(perm[NATIVE_N_TEST + NATIVE_N_VAL :])
    if len(train) != NATIVE_N_TRAIN:
        raise RuntimeError(f"unexpected native train size {len(train)}")

    cells: dict[str, Any] = {}
    for render in ("plain", "native"):
        source_ctx = plain_ctx if render == "plain" else native_ctx
        source_answer = plain_answer if render == "plain" else native_answer
        subset = pos if render == "plain" else np.arange(len(native_ids))
        y = source_answer["w"][subset].to(torch.float32).numpy()
        for summary in ("u_mean", "u_last"):
            x = source_ctx[summary][subset].to(torch.float32).numpy()
            with threadpool_limits(limits=blas_threads):
                pred, info = dual_val_ridge_predict(
                    x[train], y[train], x[val], y[val], x[test], QWEN_LAMBDAS
                )
            cells[f"{render}_{summary}"] = {
                "val_r2": info["val_r2_at_selected"],
                "test_r2": pooled_r2(pred, y[test]),
                "selected_lambda": info["selected_lambda"],
                "lambda_grid_edge": info["lambda_grid_edge"],
            }
    return f"{stage}_L{layer}", {
        "stage": stage,
        "layer": layer,
        "cells": cells,
        "test_row_ids_sha256": hashlib.sha256(
            "\n".join(native_ids[idx] for idx in test).encode()
        ).hexdigest(),
    }


def run_native_bridge(
    stage_root: Path,
    out_dir: Path,
    *,
    workers: int,
    blas_threads: int,
    force: bool = False,
) -> dict[str, Any]:
    result_path = out_dir / "bridge_native_vs_plain.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    cells: dict[str, Any] = {}
    tasks = [
        (str(stage_root), stage, layer, blas_threads)
        for stage in NATIVE_STAGES
        for layer in CAPTURED_LAYERS
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool_executor:
        futures = {pool_executor.submit(_native_bridge_cell_worker, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            key, cell = future.result()
            cells[key] = cell
            if done % 6 == 0 or done == len(tasks):
                print(f"[native_bridge] {done}/{len(tasks)} layer-stage cells", flush=True)

    selection: dict[str, Any] = {}
    for summary in ("u_mean", "u_last"):
        val_by_render_layer = {
            render: {
                layer: float(
                    np.mean(
                        [
                            cells[f"{stage}_L{layer}"]["cells"][f"{render}_{summary}"]["val_r2"]
                            for stage in NATIVE_STAGES
                        ]
                    )
                )
                for layer in CAPTURED_LAYERS
            }
            for render in ("plain", "native")
        }
        shared_val = {
            layer: float(np.mean([val_by_render_layer[r][layer] for r in ("plain", "native")]))
            for layer in CAPTURED_LAYERS
        }
        shared_layer = max(CAPTURED_LAYERS, key=lambda layer: (shared_val[layer], -layer))
        render_selected = {
            render: max(
                CAPTURED_LAYERS,
                key=lambda layer: (val_by_render_layer[render][layer], -layer),
            )
            for render in ("plain", "native")
        }
        at_shared = {
            render: {
                stage: cells[f"{stage}_L{shared_layer}"]["cells"][f"{render}_{summary}"]["test_r2"]
                for stage in NATIVE_STAGES
            }
            for render in ("plain", "native")
        }
        selection[summary] = {
            "shared_selected_layer": shared_layer,
            "render_selected_layers": render_selected,
            "mean_validation_r2_by_render_layer": val_by_render_layer,
            "test_r2_at_shared_layer": at_shared,
            "mean_test_r2_at_shared_layer": {
                render: float(np.mean(list(at_shared[render].values())))
                for render in ("plain", "native")
            },
            "native_minus_plain_at_shared_layer": float(
                np.mean(list(at_shared["native"].values()))
                - np.mean(list(at_shared["plain"].values()))
            ),
        }
    report = {
        "protocol": {
            "stages": list(NATIVE_STAGES),
            "layers": list(CAPTURED_LAYERS),
            "n_train": NATIVE_N_TRAIN,
            "n_val": NATIVE_N_VAL,
            "n_test": NATIVE_N_TEST,
            "split_seed": BRIDGE_SPLIT_SEED,
            "rows": "same 2,000 contexts and same answer texts in both renders",
            "selection": "one layer per summary chosen on mean validation R2 across both renders and three stages",
        },
        "selection": selection,
        "cells": cells,
    }
    _write_json(result_path, report)
    return report


def _qwen_chunk_paths(qwen_root: Path, split: str) -> list[Path]:
    paths = sorted(_qwen_capture_dir(qwen_root, split).glob("shard*_chunk*.pt"))
    if not paths:
        raise FileNotFoundError(
            f"no staged Qwen chunks for {split}: {_qwen_capture_dir(qwen_root, split)}"
        )
    return paths


def _load_qwen_split(
    qwen_root: Path, split: str, selected_rows: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Load L19 in deterministic filename/within-file order.

    ``selected_rows`` addresses that stream order, which is the row convention
    used by the committed Qwen n=1,000 ladder draws.
    """
    import torch

    selected = None if selected_rows is None else np.sort(np.asarray(selected_rows, dtype=np.int64))
    x_parts, y_parts, cis, prompts = [], [], [], []
    offset = 0
    for path in _qwen_chunk_paths(qwen_root, split):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        li = [int(v) for v in payload["layers"]].index(QWEN_LAYER)
        n_rows = len(payload["ci"])
        if selected is None:
            local = np.arange(n_rows, dtype=np.int64)
        else:
            take = selected[(selected >= offset) & (selected < offset + n_rows)]
            local = take - offset
        if len(local):
            x_parts.append(payload["cx_last"][local, li].to(torch.float32).numpy())
            y_parts.append(payload["v_x"][local, li].to(torch.float32).numpy())
            cis.extend(int(payload["ci"][int(i)]) for i in local)
            prompts.extend(str(payload["prompts"][int(i)]) for i in local)
        offset += n_rows
    if selected is not None and sum(len(v) for v in x_parts) != len(selected):
        raise RuntimeError(
            f"Qwen selected-row shortfall for {split}: got {sum(len(v) for v in x_parts)} "
            f"expected {len(selected)} from {offset} total rows"
        )
    return np.concatenate(x_parts), np.concatenate(y_parts), cis, prompts


def _exact_prompt_alignment(
    stage_root: Path,
    qwen_val_prompts: list[str],
    qwen_test_prompts: list[str],
) -> dict[str, Any]:
    with (stage_root / CORPUS_SINGLE_PATH).open() as fh:
        corpus_rows = [json.loads(line) for line in fh]
    kept_ids, _ = _read_groups(stage_root, "single")
    kept = set(kept_ids)
    query_to_ids: dict[str, list[str]] = {}
    for row in corpus_rows:
        if row["class"] == "generic" and str(row["id"]) in kept:
            query_to_ids.setdefault(str(row["query"]), []).append(str(row["id"]))
    ambiguous = {q for q, ids in query_to_ids.items() if len(ids) != 1}
    query_to_id = {q: ids[0] for q, ids in query_to_ids.items() if q not in ambiguous}
    id_to_olmo = {row_id: idx for idx, row_id in enumerate(kept_ids)}

    def unique_pairs(prompts: list[str], excluded: set[str]) -> list[tuple[int, int, str]]:
        used = set(excluded)
        pairs = []
        for qidx, prompt in enumerate(prompts):
            row_id = query_to_id.get(prompt)
            if row_id is None or row_id in used:
                continue
            used.add(row_id)
            pairs.append((qidx, id_to_olmo[row_id], row_id))
        return pairs

    # Preserve the held-out panel; remove the eight exact val/test duplicates
    # from validation instead of leaking them into both model-selection and test.
    test = unique_pairs(qwen_test_prompts, set())
    val = unique_pairs(qwen_val_prompts, {row_id for _, _, row_id in test})
    if len(test) < 500 or len(val) < 200:
        raise RuntimeError(
            f"exact prompt overlap unexpectedly small: val={len(val)} test={len(test)}"
        )
    return {
        "qwen_val": np.asarray([p[0] for p in val], dtype=np.int64),
        "qwen_test": np.asarray([p[0] for p in test], dtype=np.int64),
        "olmo_val": np.asarray([p[1] for p in val], dtype=np.int64),
        "olmo_test": np.asarray([p[1] for p in test], dtype=np.int64),
        "val_ids": [p[2] for p in val],
        "test_ids": [p[2] for p in test],
        "kept_ids": kept_ids,
        "query_set": set(query_to_id),
        "n_ambiguous_olmo_queries_excluded": len(ambiguous),
    }


def _exact_olmo_cell_worker(task: tuple[Any, ...]) -> tuple[str, dict[str, Any]]:
    (
        stage_root_s,
        stage,
        layer,
        train_draws,
        val_idx,
        test_idx,
        expected_ids_hash,
        blas_threads,
        include_pca,
    ) = task
    from threadpoolctl import threadpool_limits

    x, y, ids = _load_tensor_pair(Path(stage_root_s), stage, "single", "u_last", int(layer))
    got_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if got_hash != expected_ids_hash:
        raise RuntimeError(f"OLMo row-order mismatch in exact bridge: {stage}/L{layer}")
    draws = []
    for draw, train_idx in enumerate(train_draws):
        with threadpool_limits(limits=blas_threads):
            if include_pca:
                pred_pca, pca_info = dual_val_pca_ridge_predict(
                    x[train_idx],
                    y[train_idx],
                    x[val_idx],
                    y[val_idx],
                    x[test_idx],
                    QWEN_LAMBDAS,
                    PCA_RANKS,
                    std_ddof=1,
                    include_full_prediction=True,
                )
                pred = pca_info.pop("_full_rank_prediction")
                info = pca_info.pop("full_rank_ridge")
                pca_result = {"test_r2": pooled_r2(pred_pca, y[test_idx]), **pca_info}
            else:
                fitter = (
                    primal_val_ridge_predict
                    if len(train_idx) > x.shape[1]
                    else dual_val_ridge_predict
                )
                pred, info = fitter(
                    x[train_idx],
                    y[train_idx],
                    x[val_idx],
                    y[val_idx],
                    x[test_idx],
                    QWEN_LAMBDAS,
                    std_ddof=1,
                )
            row: dict[str, Any] = {
                "draw": draw,
                "test_r2": pooled_r2(pred, y[test_idx]),
                **info,
                **({"pca": pca_result} if include_pca else {}),
            }
        draws.append(row)
    return f"{stage}_L{layer}", {
        "stage": stage,
        "layer": int(layer),
        "draws": draws,
        "mean_val_r2": float(np.mean([r["val_r2_at_selected"] for r in draws])),
        "mean_test_r2": float(np.mean([r["test_r2"] for r in draws])),
    }


def run_exact_prompt_bridge(
    stage_root: Path,
    qwen_root: Path,
    out_dir: Path,
    *,
    workers: int,
    blas_threads: int,
    force: bool = False,
) -> dict[str, Any]:
    result_path = out_dir / "bridge_exact_prompts.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())

    qx_val, qy_val, _, qprompts_val = _load_qwen_split(qwen_root, "val_400")
    qx_test, qy_test, _, qprompts_test = _load_qwen_split(qwen_root, "test_1000")
    alignment = _exact_prompt_alignment(stage_root, qprompts_val, qprompts_test)
    q_val, q_test = alignment["qwen_val"], alignment["qwen_test"]
    o_val, o_test = alignment["olmo_val"], alignment["olmo_test"]

    qwen_draw_rows = {
        draw: np.sort(
            np.random.default_rng(19010000 + BRIDGE_N_TRAIN * 10 + draw).choice(
                25_000, size=BRIDGE_N_TRAIN, replace=False
            )
        )
        for draw in BRIDGE_DRAWS
    }
    qwen_union = np.unique(np.concatenate(list(qwen_draw_rows.values())))
    qx_train, qy_train, _, qprompts_train = _load_qwen_split(qwen_root, "train_25k", qwen_union)
    union_pos = {int(row): idx for idx, row in enumerate(qwen_union)}
    qwen_local_draws = {
        draw: np.asarray([union_pos[int(row)] for row in rows], dtype=np.int64)
        for draw, rows in qwen_draw_rows.items()
    }

    qwen_ref = dict(zip(BRIDGE_DRAWS, _qwen_n1000_draws(out_dir), strict=True))
    qwen_checkpoint = out_dir / "bridge_exact_prompts_qwen_checkpoint.json"
    if qwen_checkpoint.exists() and not force:
        qwen_draw_results = json.loads(qwen_checkpoint.read_text())["draws"]
    else:
        qwen_draw_results = []
        for draw in BRIDGE_DRAWS:
            tr = qwen_local_draws[draw]
            # First reproduce the published all-val/all-test cell.  This guards the
            # local Qwen reader, row convention, ddof, and dual/primal equivalence.
            pred_all, info_all = dual_val_ridge_predict(
                qx_train[tr],
                qy_train[tr],
                qx_val,
                qy_val,
                qx_test,
                QWEN_LAMBDAS,
                std_ddof=1,
            )
            parity_r2 = pooled_r2(pred_all, qy_test)
            pred_pca, pca_info = dual_val_pca_ridge_predict(
                qx_train[tr],
                qy_train[tr],
                qx_val[q_val],
                qy_val[q_val],
                qx_test[q_test],
                QWEN_LAMBDAS,
                PCA_RANKS,
                std_ddof=1,
                include_full_prediction=True,
            )
            pred = pca_info.pop("_full_rank_prediction")
            info = pca_info.pop("full_rank_ridge")
            qwen_draw_results.append(
                {
                    "draw": draw,
                    "published_parity": {
                        "r2": parity_r2,
                        "reference_r2": qwen_ref[draw],
                        "abs_delta": abs(parity_r2 - qwen_ref[draw]),
                        **info_all,
                    },
                    "exact_prompt_ridge": {
                        "test_r2": pooled_r2(pred, qy_test[q_test]),
                        **info,
                    },
                    "exact_prompt_pca": {
                        "test_r2": pooled_r2(pred_pca, qy_test[q_test]),
                        **pca_info,
                    },
                }
            )
        _write_json(qwen_checkpoint, {"draws": qwen_draw_results})
    parity_pass = all(r["published_parity"]["abs_delta"] <= 1e-6 for r in qwen_draw_results)
    if not parity_pass:
        raise RuntimeError(f"Qwen n=1000 parity failed: {qwen_draw_results}")

    kept_ids = alignment["kept_ids"]
    test_val = set(alignment["test_ids"]) | set(alignment["val_ids"])
    with _row_index_path(stage_root, "single").open() as fh:
        classes = [json.loads(line).get("class") or "generic" for line in fh]
    pool = np.asarray(
        [
            i
            for i, (row_id, cls) in enumerate(zip(kept_ids, classes, strict=True))
            if cls == "generic" and row_id not in test_val
        ],
        dtype=np.int64,
    )
    olmo_train_draws = [
        np.sort(
            np.random.default_rng(19010000 + BRIDGE_N_TRAIN * 10 + draw).choice(
                pool, size=BRIDGE_N_TRAIN, replace=False
            )
        )
        for draw in BRIDGE_DRAWS
    ]
    ids_hash = hashlib.sha256("\n".join(kept_ids).encode()).hexdigest()
    # Keep model/layer selection independent of the exact-overlap panel.  The
    # earlier all-layer generic-LMSYS bridge selected L18 on its own 400-row
    # validation set; re-selecting on these 246 overlap validation prompts is
    # both redundant and a less clean confirmatory design.
    layer_selection_path = out_dir / "bridge_n1000_layer_sweep_generic.json"
    layer_selection = json.loads(layer_selection_path.read_text())
    selected_layer = int(layer_selection["selection"]["u_last"]["selected_layer"])
    if selected_layer not in CAPTURED_LAYERS:
        raise RuntimeError(f"invalid inherited OLMo layer selection: {selected_layer}")
    pca_tasks = [
        (
            str(stage_root),
            stage,
            selected_layer,
            olmo_train_draws,
            o_val,
            o_test,
            ids_hash,
            blas_threads,
            True,
        )
        for stage in STAGES
    ]
    olmo_pca: dict[str, Any] = {}
    # Qwen's BLAS work runs immediately before this pool.  Forking a process
    # after a multithreaded BLAS runtime has initialized can inherit locked
    # futexes and deadlock all workers; spawn gives every cell a clean runtime.
    from multiprocessing import get_context

    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool_executor:
        futures = {pool_executor.submit(_exact_olmo_cell_worker, task): task for task in pca_tasks}
        for future in as_completed(futures):
            key, cell = future.result()
            olmo_pca[key] = cell

    qwen_ridge = float(np.mean([r["exact_prompt_ridge"]["test_r2"] for r in qwen_draw_results]))
    qwen_pca = float(np.mean([r["exact_prompt_pca"]["test_r2"] for r in qwen_draw_results]))
    selected_ridge = {
        stage: olmo_pca[f"{stage}_L{selected_layer}"]["mean_test_r2"] for stage in STAGES
    }
    selected_pca = {
        stage: float(
            np.mean(
                [row["pca"]["test_r2"] for row in olmo_pca[f"{stage}_L{selected_layer}"]["draws"]]
            )
        )
        for stage in STAGES
    }
    report = {
        "protocol": {
            "qwen_revision": QWEN_HF_REVISION,
            "olmo_revision": HF_REVISION,
            "n_train_each": BRIDGE_N_TRAIN,
            "n_val_exact_prompts": len(q_val),
            "n_test_exact_prompts": len(q_test),
            "draws": list(BRIDGE_DRAWS),
            "lambda_grid": QWEN_LAMBDAS.tolist(),
            "pca_rank_grid": list(PCA_RANKS),
            "standardization_ddof": 1,
            "prompt_matching": "exact string, one row per unique OLMo id; val/test exact duplicates removed from val",
            "training_caveat": "n and source family are matched; exact train-prompt overlap is reported under alignment",
            "olmo_render": "shared plain User:/Assistant: render; native-render correction is reported separately",
        },
        "alignment": {
            "val_ids_sha256": hashlib.sha256("\n".join(alignment["val_ids"]).encode()).hexdigest(),
            "test_ids_sha256": hashlib.sha256(
                "\n".join(alignment["test_ids"]).encode()
            ).hexdigest(),
            "n_ambiguous_olmo_queries_excluded": alignment["n_ambiguous_olmo_queries_excluded"],
            "selected_qwen_train_prompt_overlap_with_olmo": sum(
                prompt in alignment["query_set"] for prompt in qprompts_train
            ),
        },
        "qwen": {
            "layer": QWEN_LAYER,
            "published_parity_pass": parity_pass,
            "draws": qwen_draw_results,
            "mean_exact_prompt_ridge_r2": qwen_ridge,
            "mean_exact_prompt_pca_r2": qwen_pca,
        },
        "olmo": {
            "selected_layer": selected_layer,
            "layer_selection_source": str(layer_selection_path),
            "layer_selection_protocol": layer_selection["protocol"],
            "test_r2_at_selected_layer": selected_ridge,
            "pca_test_r2_at_selected_layer": selected_pca,
            "pca_cells": olmo_pca,
        },
        "contrasts": {
            "qwen_minus_olmo_ridge": {
                stage: qwen_ridge - value for stage, value in selected_ridge.items()
            },
            "qwen_minus_olmo_pca": {
                stage: qwen_pca - value for stage, value in selected_pca.items()
            },
        },
    }
    _write_json(result_path, report)
    return report


def _per_row_r2_components(pred: np.ndarray, target: np.ndarray) -> dict[str, list[float]]:
    pred64 = np.asarray(pred, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    center = target64.mean(axis=0)
    return {
        "residual_sse": np.square(target64 - pred64).sum(axis=1).tolist(),
        "centered_sst": np.square(target64 - center).sum(axis=1).tolist(),
    }


def run_large_exact_prompt_bridge(
    stage_root: Path,
    qwen_root: Path,
    out_dir: Path,
    *,
    blas_threads: int,
    force: bool = False,
) -> dict[str, Any]:
    """Repeat the exact-prompt bridge near the largest common training n."""
    n_train = LARGE_BRIDGE_N_TRAIN
    result_path = out_dir / f"bridge_exact_prompts_n{n_train}.json"
    checkpoint_path = out_dir / f"bridge_exact_prompts_n{n_train}_checkpoint.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    prior = json.loads((out_dir / "bridge_exact_prompts.json").read_text())

    from threadpoolctl import threadpool_limits

    qx_val, qy_val, _, qprompts_val = _load_qwen_split(qwen_root, "val_400")
    qx_test, qy_test, _, qprompts_test = _load_qwen_split(qwen_root, "test_1000")
    alignment = _exact_prompt_alignment(stage_root, qprompts_val, qprompts_test)
    alignment_hashes = {
        "val_ids_sha256": hashlib.sha256("\n".join(alignment["val_ids"]).encode()).hexdigest(),
        "test_ids_sha256": hashlib.sha256("\n".join(alignment["test_ids"]).encode()).hexdigest(),
    }
    for key, value in alignment_hashes.items():
        if value != prior["alignment"][key]:
            raise RuntimeError(
                f"large exact bridge {key} differs from n=1000 panel: "
                f"{value} != {prior['alignment'][key]}"
            )
    q_val, q_test = alignment["qwen_val"], alignment["qwen_test"]
    o_val, o_test = alignment["olmo_val"], alignment["olmo_test"]

    with _row_index_path(stage_root, "single").open() as fh:
        classes = [json.loads(line).get("class") or "generic" for line in fh]
    held_out = set(alignment["test_ids"]) | set(alignment["val_ids"])
    olmo_pool = np.asarray(
        [
            idx
            for idx, (row_id, cls) in enumerate(zip(alignment["kept_ids"], classes, strict=True))
            if cls == "generic" and row_id not in held_out
        ],
        dtype=np.int64,
    )
    if len(olmo_pool) < n_train:
        raise RuntimeError(f"OLMo generic pool has {len(olmo_pool)} rows, below n={n_train}")

    qwen_draw_rows = {
        draw: np.sort(
            np.random.default_rng(19010000 + n_train * 10 + draw).choice(
                25_000, size=n_train, replace=False
            )
        )
        for draw in BRIDGE_DRAWS
    }
    olmo_train_draws = {
        draw: np.sort(
            np.random.default_rng(19010000 + n_train * 10 + draw).choice(
                olmo_pool, size=n_train, replace=False
            )
        )
        for draw in BRIDGE_DRAWS
    }
    layer_selection_path = out_dir / "bridge_n1000_layer_sweep_generic.json"
    layer_selection = json.loads(layer_selection_path.read_text())
    selected_layer = int(layer_selection["selection"]["u_last"]["selected_layer"])

    checkpoint: dict[str, Any]
    if checkpoint_path.exists() and not force:
        checkpoint = json.loads(checkpoint_path.read_text())
    else:
        checkpoint = {"qwen": {}, "olmo": {}}

    missing_qwen = [draw for draw in BRIDGE_DRAWS if str(draw) not in checkpoint["qwen"]]
    if missing_qwen:
        union = np.unique(np.concatenate([qwen_draw_rows[draw] for draw in missing_qwen]))
        qx_train, qy_train, _, qprompts_train = _load_qwen_split(qwen_root, "train_25k", union)
        union_pos = {int(row): idx for idx, row in enumerate(union)}
        for draw in missing_qwen:
            local = np.asarray([union_pos[int(row)] for row in qwen_draw_rows[draw]])
            print(f"[large exact bridge] Qwen draw={draw} n={n_train}", flush=True)
            with threadpool_limits(limits=blas_threads):
                pred, info = primal_val_ridge_predict(
                    qx_train[local],
                    qy_train[local],
                    qx_val[q_val],
                    qy_val[q_val],
                    qx_test[q_test],
                    QWEN_LAMBDAS,
                    std_ddof=1,
                )
            checkpoint["qwen"][str(draw)] = {
                "draw": draw,
                "test_r2": pooled_r2(pred, qy_test[q_test]),
                "selected_train_prompt_overlap_with_olmo": sum(
                    prompt in alignment["query_set"] for prompt in np.asarray(qprompts_train)[local]
                ),
                "per_row": _per_row_r2_components(pred, qy_test[q_test]),
                **info,
            }
            _write_json(checkpoint_path, checkpoint)
            print(
                f"[large exact bridge] Qwen draw={draw} test_r2="
                f"{checkpoint['qwen'][str(draw)]['test_r2']:.6f}",
                flush=True,
            )

    ids_hash = hashlib.sha256("\n".join(alignment["kept_ids"]).encode()).hexdigest()
    for stage in ("S", "D", "R"):
        stage_rows = checkpoint["olmo"].setdefault(stage, {})
        missing = [draw for draw in BRIDGE_DRAWS if str(draw) not in stage_rows]
        if not missing:
            continue
        x, y, ids = _load_tensor_pair(stage_root, stage, "single", "u_last", selected_layer)
        if hashlib.sha256("\n".join(ids).encode()).hexdigest() != ids_hash:
            raise RuntimeError(f"OLMo row-order mismatch in large exact bridge: {stage}")
        for draw in missing:
            train_idx = olmo_train_draws[draw]
            print(f"[large exact bridge] OLMo-{stage} draw={draw} n={n_train}", flush=True)
            with threadpool_limits(limits=blas_threads):
                pred, info = primal_val_ridge_predict(
                    x[train_idx],
                    y[train_idx],
                    x[o_val],
                    y[o_val],
                    x[o_test],
                    QWEN_LAMBDAS,
                    std_ddof=1,
                )
            stage_rows[str(draw)] = {
                "draw": draw,
                "test_r2": pooled_r2(pred, y[o_test]),
                "per_row": _per_row_r2_components(pred, y[o_test]),
                **info,
            }
            _write_json(checkpoint_path, checkpoint)
            print(
                f"[large exact bridge] OLMo-{stage} draw={draw} test_r2="
                f"{stage_rows[str(draw)]['test_r2']:.6f}",
                flush=True,
            )
        del x, y

    qwen_rows = [checkpoint["qwen"][str(draw)] for draw in BRIDGE_DRAWS]
    olmo_rows = {
        stage: [checkpoint["olmo"][stage][str(draw)] for draw in BRIDGE_DRAWS]
        for stage in ("S", "D", "R")
    }
    qwen_mean = float(np.mean([row["test_r2"] for row in qwen_rows]))
    olmo_means = {
        stage: float(np.mean([row["test_r2"] for row in rows])) for stage, rows in olmo_rows.items()
    }
    report = {
        "protocol": {
            "n_train_each": n_train,
            "n_olmo_generic_pool": len(olmo_pool),
            "n_val_exact_prompts": len(q_val),
            "n_test_exact_prompts": len(q_test),
            "draws": list(BRIDGE_DRAWS),
            "lambda_grid": QWEN_LAMBDAS.tolist(),
            "standardization_ddof": 1,
            "solver": "primal eigendecomposition; validation-selected pooled R2",
            "qwen_layer": QWEN_LAYER,
            "olmo_layer": selected_layer,
            "olmo_layer_selection_source": str(layer_selection_path),
            "prompt_matching": "same exact-string validation/test panel as n=1000 bridge",
            "training_caveat": "same n and generic source family, but zero exact cross-family training-prompt overlap",
        },
        "alignment": alignment_hashes,
        "qwen": {"draws": qwen_rows, "mean_test_r2": qwen_mean},
        "olmo": {
            stage: {"draws": rows, "mean_test_r2": olmo_means[stage]}
            for stage, rows in olmo_rows.items()
        },
        "contrasts": {
            "qwen_minus_olmo_mean": {
                stage: qwen_mean - olmo_means[stage] for stage in ("S", "D", "R")
            },
            "paired_draw_gaps": {
                stage: [
                    qwen_rows[i]["test_r2"] - olmo_rows[stage][i]["test_r2"]
                    for i in range(len(BRIDGE_DRAWS))
                ]
                for stage in ("S", "D", "R")
            },
        },
        "n1000_reference": {
            "qwen_mean_test_r2": prior["qwen"]["mean_exact_prompt_ridge_r2"],
            "olmo_mean_test_r2": prior["olmo"]["test_r2_at_selected_layer"],
            "qwen_minus_olmo": prior["contrasts"]["qwen_minus_olmo_ridge"],
        },
    }
    _write_json(result_path, report)
    return report


def _variance_weighted_target_reliability(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ac, bc = aa - aa.mean(axis=0), bb - bb.mean(axis=0)
    corr = (ac * bc).sum(axis=0) / (
        np.sqrt(np.square(ac).sum(axis=0) * np.square(bc).sum(axis=0)) + 1e-30
    )
    variance = ((aa + bb) / 2.0).var(axis=0, ddof=0)
    return {
        "ceiling_var_weighted_r": float((variance * corr).sum() / (variance.sum() + 1e-30)),
        "mean_per_dim_r": float(corr.mean()),
    }


def run_reliability(stage_root: Path, out_dir: Path, *, force: bool = False) -> dict[str, Any]:
    import torch

    result_path = out_dir / "target_reliability_matched_formula.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    cells: dict[str, Any] = {}
    root = _store_root(stage_root) / "reliability"
    for layer in (18, 31):
        for stage in STAGES:
            a = torch.load(
                root / stage / "single" / "seed43" / f"L{layer}.pt",
                map_location="cpu",
                weights_only=True,
            )
            b = torch.load(
                root / stage / "single" / "seed44" / f"L{layer}.pt",
                map_location="cpu",
                weights_only=True,
            )
            pos_b = {str(row_id): idx for idx, row_id in enumerate(b["row_ids"])}
            ia = [i for i, row_id in enumerate(a["row_ids"]) if str(row_id) in pos_b]
            ib = [pos_b[str(a["row_ids"][i])] for i in ia]
            if len(ia) != 914:
                raise RuntimeError(
                    f"OLMo reliability pairing shortfall: {stage}/L{layer} n={len(ia)}"
                )
            cells[f"{stage}_L{layer}"] = {
                "n_pairs": len(ia),
                **_variance_weighted_target_reliability(
                    a["w"][ia].to(torch.float32).numpy(),
                    b["w"][ib].to(torch.float32).numpy(),
                ),
            }
    qwen_path = (
        PROJECT_ROOT / "eval_results" / "issue_1491" / "scale_ladder" / "fits_scale7_refit.json"
    )
    qwen_source = (
        json.loads(qwen_path.read_text())
        if qwen_path.exists()
        else _git_json(
            ISSUE1902_RESULTS_COMMIT,
            "eval_results/issue_1491/scale_ladder/fits_scale7_refit.json",
        )
    )
    qwen = qwen_source["ceiling_two_draw"]
    old_olmo_source = (
        json.loads(OLMO_OPERATOR.read_text())
        if OLMO_OPERATOR.exists()
        else _git_json(
            ISSUE1902_RESULTS_COMMIT,
            "eval_results/issue_1902/operator/operator_battery.json",
        )
    )
    old_olmo = old_olmo_source["reliability_ceiling"]
    report = {
        "protocol": {
            "formula": "Qwen #1491 variance-weighted per-dimension Pearson correlation between seed43/44 answer-mean vectors",
            "note": "The previously published #1902 ceiling correlated per-context squared prediction errors, not target vectors, and is retained below only as an explicitly incomparable diagnostic.",
        },
        "qwen_L19": qwen,
        "olmo": cells,
        "incomparable_original_olmo_error_repeatability": old_olmo,
    }
    _write_json(result_path, report)
    return report


def _length_stats(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.int64)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "max": int(x.max()),
        "frac_ge_1024": float(np.mean(x >= 1024)),
    }


def run_answer_lengths(
    stage_root: Path, qwen_root: Path, out_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    result_path = out_dir / "answer_lengths_exact_prompts.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    _, _, qwen_test_cis, qwen_test_prompts = _load_qwen_split(qwen_root, "test_1000")
    _, _, _, qwen_val_prompts = _load_qwen_split(qwen_root, "val_400")
    alignment = _exact_prompt_alignment(stage_root, qwen_val_prompts, qwen_test_prompts)

    raw_dir = qwen_root / QWEN_HF_PREFIX / "test_1000" / "raw_completions"
    raw_rows = []
    for path in sorted(raw_dir.glob("shard*_chunk*.json")):
        payload = json.loads(path.read_text())
        raw_rows.extend(payload if isinstance(payload, list) else payload["rows"])
    raw_by_ci = {int(row["ci"]): str(row["response"]) for row in raw_rows}
    if len(raw_by_ci) != 1000:
        raise RuntimeError(f"Qwen raw completion count mismatch: {len(raw_by_ci)}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    qwen_lengths = np.asarray(
        [len(tokenizer.encode(raw_by_ci[ci], add_special_tokens=False)) for ci in qwen_test_cis],
        dtype=np.int64,
    )
    q_test = alignment["qwen_test"]
    o_test = alignment["olmo_test"]
    olmo_lengths: dict[str, np.ndarray] = {}
    for stage in STAGES:
        path = _store_root(stage_root) / stage / stage / "single" / "row_index.jsonl"
        with path.open() as fh:
            rows = [json.loads(line) for line in fh]
        ids = [str(row["id"]) for row in rows]
        if ids != alignment["kept_ids"]:
            raise RuntimeError(f"answer-length row mismatch for OLMo stage {stage}")
        olmo_lengths[stage] = np.asarray(
            [int(row["n_answer_tokens"]) for row in rows], dtype=np.int64
        )
    report = {
        "protocol": {
            "exact_test_prompts": len(q_test),
            "qwen_tokenizer": "Qwen/Qwen2.5-7B-Instruct",
            "olmo_tokenizer": "OLMo-2 family tokenizer recorded during capture",
            "caveat": "Cross-tokenizer token counts are descriptive; exact prompt matching and within-model capture counts are the inferential controls.",
        },
        "full_test_or_intersection": {
            "Qwen": _length_stats(qwen_lengths),
            **{stage: _length_stats(values) for stage, values in olmo_lengths.items()},
        },
        "exact_prompt_test": {
            "Qwen": _length_stats(qwen_lengths[q_test]),
            **{stage: _length_stats(values[o_test]) for stage, values in olmo_lengths.items()},
        },
        "exact_prompt_pearson_vs_qwen": {
            stage: float(np.corrcoef(qwen_lengths[q_test], values[o_test])[0, 1])
            for stage, values in olmo_lengths.items()
        },
    }
    _write_json(result_path, report)
    return report


def run_cross_answer_source(out_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """Recover #1902's crossed answer-text grid as a target-side diagnostic."""
    import subprocess

    result_path = out_dir / "cross_answer_source_diagnostic.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    source_path = "eval_results/issue_1902/fits/grid_cells.json"
    raw = subprocess.run(
        ["git", "show", f"{ISSUE1902_RESULTS_COMMIT}:{source_path}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    grid = json.loads(raw)
    layer = int(grid["layer_star"])
    matrix: dict[str, dict[str, float]] = {}
    for context_stage in STAGES:
        matrix[context_stage] = {}
        for answer_source in STAGES:
            if context_stage == answer_source:
                cell = grid["cells"][f"diag_{context_stage}_single_ctx"]
                r2 = float(cell["r2_by_layer"][str(layer)])
            else:
                cell = grid["cells"][f"grid_{context_stage}{answer_source}_single_ctx"]
                r2 = float(cell["per_layer"][str(layer)]["r2"])
            matrix[context_stage][answer_source] = r2
    rows = {}
    for context_stage, values in matrix.items():
        best_source = max(STAGES, key=lambda source: values[source])
        rows[context_stage] = {
            "diagonal_answer_source": context_stage,
            "diagonal_r2": values[context_stage],
            "best_answer_source": best_source,
            "best_r2": values[best_source],
            "best_minus_diagonal": values[best_source] - values[context_stage],
            "range_across_answer_sources": max(values.values()) - min(values.values()),
        }
    report = {
        "protocol": {
            "source_commit": ISSUE1902_RESULTS_COMMIT,
            "source_path": source_path,
            "layer": layer,
            "context_summary": "u_mean",
            "split": "six semantic-group folds",
            "interpretation": "Within each fixed OLMo representation model, only the generated answer-text source changes.",
        },
        "r2_context_stage_by_answer_source": matrix,
        "by_context_stage": rows,
        "posttrained_best_r2": float(
            max(matrix[m][source] for m in ("S", "D", "R") for source in STAGES)
        ),
    }
    _write_json(result_path, report)
    return report


def _git_json(commit: str, path: str) -> dict[str, Any]:
    import subprocess

    raw = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(raw)


def run_consolidate(out_dir: Path, *, force: bool = False) -> dict[str, Any]:
    result_path = out_dir / "principled_comparison.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    base = json.loads((out_dir / "summary.json").read_text())
    exact = json.loads((out_dir / "bridge_exact_prompts.json").read_text())
    reliability = json.loads((out_dir / "target_reliability_matched_formula.json").read_text())
    native = json.loads((out_dir / "bridge_native_vs_plain.json").read_text())
    layer = json.loads((out_dir / "bridge_n1000_layer_sweep.json").read_text())
    generic = json.loads((out_dir / "bridge_n1000_layer_sweep_generic.json").read_text())
    lengths = json.loads((out_dir / "answer_lengths_exact_prompts.json").read_text())
    cross_answer = json.loads((out_dir / "cross_answer_source_diagnostic.json").read_text())
    qwen_ladder = _git_json(
        "b568272e452", "eval_results/issue_1901/paper_densify/scaling_ladder_L19.json"
    )
    pilot = _git_json(ISSUE1902_RESULTS_COMMIT, "eval_results/issue_1902/pilot_report.json")

    old_protocol_accounting = {}
    for stage in STAGES:
        cell = base["cells"][f"{stage}_single"]
        old_protocol_accounting[stage] = {
            "u_mean_group_r2": cell["u_mean_group_r2"],
            "u_last_group_r2": cell["u_last_group_r2"],
            "u_last_random_r2": cell["u_last_random_fold_r2"],
            "last_token_gain": cell["u_last_minus_u_mean"],
            "random_split_gain": cell["random_minus_group"],
            "combined_gain_vs_original": cell["u_last_random_fold_r2"] - cell["u_mean_group_r2"],
            "last_token_gain_ci": cell["pooling_paired_cluster_bootstrap"]["delta_ci"],
            "random_split_gain_ci": cell["split_paired_cluster_bootstrap"]["delta_ci"],
        }

    qwen_exact = float(exact["qwen"]["mean_exact_prompt_ridge_r2"])
    qwen_by_draw = {
        int(row["draw"]): float(row["exact_prompt_ridge"]["test_r2"])
        for row in exact["qwen"]["draws"]
    }
    qwen_rel = float(reliability["qwen_L19"]["ceiling_var_weighted_r"])
    qwen_reliability_adjusted = qwen_exact / qwen_rel
    exact_accounting = {}
    for stage in STAGES:
        olmo_r2 = float(exact["olmo"]["test_r2_at_selected_layer"][stage])
        olmo_rel = float(reliability["olmo"][f"{stage}_L18"]["ceiling_var_weighted_r"])
        same_skill_counterfactual = qwen_reliability_adjusted * olmo_rel
        draw_gaps = [
            qwen_by_draw[int(row["draw"])] - float(row["test_r2"])
            for row in exact["olmo"]["pca_cells"][f"{stage}_L18"]["draws"]
        ]
        exact_accounting[stage] = {
            "qwen_r2": qwen_exact,
            "olmo_r2": olmo_r2,
            "raw_gap": qwen_exact - olmo_r2,
            "raw_gap_training_draw_range": [min(draw_gaps), max(draw_gaps)],
            "qwen_target_reliability": qwen_rel,
            "olmo_target_reliability": olmo_rel,
            "r2_divided_by_reliability": olmo_r2 / olmo_rel,
            "same_adjusted_skill_counterfactual_r2": same_skill_counterfactual,
            "gap_attributed_to_target_noise_under_attenuation_model": qwen_exact
            - same_skill_counterfactual,
            "residual_after_attenuation_model": same_skill_counterfactual - olmo_r2,
        }

    qwen_15k = next(
        float(cell["ridge"]["test_r2"])
        for cell in qwen_ladder["cells"]
        if int(cell["n_train"]) == 15_000
    )
    with np.load(_cell_output(out_dir, "u_last", "random", "D", "single")) as payload:
        olmo_train_sizes = payload["n_train"].tolist()
    pca_ranks = {
        model: sorted(
            {int(draw["exact_prompt_pca"]["selected_rank"]) for draw in exact["qwen"]["draws"]}
        )
        if model == "Qwen"
        else sorted(
            {
                int(draw["pca"]["selected_rank"])
                for draw in exact["olmo"]["pca_cells"][f"{model}_L18"]["draws"]
            }
        )
        for model in ("Qwen", *STAGES)
    }
    report = {
        "headline": {
            "explanation": "The old ~0.4 OLMo values combine a diluted prompt-mean feature with semantic-group holdout. Last-token plus random-row evaluation raises post-trained OLMo to ~0.51. The remaining matched Qwen advantage is partly lower OLMo target repeatability and mostly a residual model-specific linear-geometry gap; it is not removed by exact prompt matching, PCA, layer selection, native templates, length, precision, or answer-source swaps.",
            "qwen_exact_prompt_n1000_r2": qwen_exact,
            "olmo_exact_prompt_n1000_r2": exact["olmo"]["test_r2_at_selected_layer"],
        },
        "old_protocol_accounting": old_protocol_accounting,
        "exact_prompt_accounting": {
            "protocol": exact["protocol"],
            "by_stage": exact_accounting,
            "attenuation_caveat": "This is a descriptive errors-in-variables decomposition, not an identified causal partition. It treats the two-draw correlation as a multiplicative R2 reliability ceiling, and the 914-row OLMo reliability panel is not the 576-row exact-prompt test panel.",
        },
        "controls": {
            "qwen_solver_parity": exact["qwen"]["published_parity_pass"],
            "pca_selected_ranks": pca_ranks,
            "pca_r2_delta": {
                "Qwen": exact["qwen"]["mean_exact_prompt_pca_r2"] - qwen_exact,
                **{
                    stage: exact["olmo"]["pca_test_r2_at_selected_layer"][stage]
                    - exact["olmo"]["test_r2_at_selected_layer"][stage]
                    for stage in STAGES
                },
            },
            "layer_selection": layer["selection"]["u_last"],
            "generic_only_layer_selection": generic["selection"]["u_last"],
            "native_minus_plain": native["selection"]["u_last"][
                "native_minus_plain_at_shared_layer"
            ],
            "exact_prompt_answer_lengths": lengths["exact_prompt_test"],
            "cross_answer_source": cross_answer,
            "max_fp16_vs_fp32_abs_r2": float(
                max(pilot["fits"]["fp16_delta_r2"]["deltas"].values())
            ),
            "large_n": {
                "qwen_n15000_r2": qwen_15k,
                "olmo_random_fold_n_train": olmo_train_sizes,
                "olmo_posttrained_r2": {
                    stage: base["cells"][f"{stage}_single"]["u_last_random_fold_r2"]
                    for stage in ("S", "D", "R")
                },
            },
        },
        "residual_interpretation": {
            "supported": "The residual belongs to the model/target system: OLMo-2's last-context state is less linearly aligned with its mean answer state than Qwen-2.5's under the tested generation policies.",
            "not_identified": "These banks cannot separate architecture/pretraining geometry from model-specific answer semantics completely, because the exact training prompts and answer texts are not shared across model families.",
            "answer_text_bound": "Within OLMo, swapping among B/S/D/R answer sources changes the best post-trained mean/group map by at most 0.031 above its diagonal and never exceeds R2=0.397, so answer style is not the main missing 0.15.",
            "split_caveat": "The random-versus-group increment is the cost of semantic extrapolation under these corpora and may include paraphrase-level similarity available to the IID split; exact duplicates were removed, but no new near-duplicate audit was run.",
        },
    }
    _write_json(result_path, report)
    return report


def analyze(stage_root: Path, out_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(BOOT_SEED)
    counts_by_corpus: dict[str, np.ndarray] = {}
    groups_by_corpus: dict[str, list[str]] = {}
    ids_by_corpus: dict[str, list[str]] = {}
    for corpus in CORPORA:
        ids, groups = _read_groups(stage_root, corpus)
        names = sorted(set(groups))
        counts_by_corpus[corpus] = rng.multinomial(
            len(names), np.full(len(names), 1.0 / len(names)), size=N_BOOT
        ).astype(np.float64)
        groups_by_corpus[corpus] = groups
        ids_by_corpus[corpus] = ids

    cells: dict[str, Any] = {}
    for corpus in CORPORA:
        for stage in STAGES:
            mean_res, mean_tot = _committed_mean_components(stage, corpus)
            with np.load(_cell_output(out_dir, "u_last", "group", stage, corpus)) as last:
                last_res, last_tot = last["ss_res"], last["ss_tot"]
                if last["row_ids"].tolist() != ids_by_corpus[corpus]:
                    raise RuntimeError(f"stored row ids mismatch: {stage}/{corpus}")
                group_lambdas = last["selected_lambda"].tolist()
                group_dof = last["dof"].tolist()
            with np.load(_cell_output(out_dir, "u_last", "random", stage, corpus)) as random:
                random_res, random_tot = random["ss_res"], random["ss_tot"]
                random_lambdas = random["selected_lambda"].tolist()
                random_dof = random["dof"].tolist()

            q_mean = 1.0 - float(mean_res.sum()) / float(mean_tot.sum())
            q_last = 1.0 - float(last_res.sum()) / float(last_tot.sum())
            q_random = 1.0 - float(random_res.sum()) / float(random_tot.sum())
            pooling = paired_group_bootstrap(
                last_res,
                last_tot,
                mean_res,
                mean_tot,
                groups_by_corpus[corpus],
                counts=counts_by_corpus[corpus],
            )
            split = paired_group_bootstrap(
                random_res,
                random_tot,
                last_res,
                last_tot,
                groups_by_corpus[corpus],
                counts=counts_by_corpus[corpus],
            )
            cells[f"{stage}_{corpus}"] = {
                "n": len(mean_res),
                "n_groups": len(set(groups_by_corpus[corpus])),
                "u_mean_group_r2": q_mean,
                "u_last_group_r2": q_last,
                "u_last_minus_u_mean": q_last - q_mean,
                "pooling_paired_cluster_bootstrap": pooling,
                "u_last_random_fold_r2": q_random,
                "random_minus_group": q_random - q_last,
                "split_paired_cluster_bootstrap": split,
                "group_selected_lambda": group_lambdas,
                "group_dof": group_dof,
                "random_selected_lambda": random_lambdas,
                "random_dof": random_dof,
            }

    bridge = json.loads((out_dir / "bridge_n1000.json").read_text())
    qwen_n1000 = _qwen_n1000_draws(out_dir)
    bridge_summary = {
        "qwen_l19_n1000_draws": qwen_n1000,
        "qwen_l19_n1000_mean": float(np.mean(qwen_n1000)),
        "olmo_single_l31": {
            stage: {
                summary: bridge["cells"][f"{stage}_{summary}"]["mean_test_r2"]
                for summary in ("u_mean", "u_last")
            }
            for stage in STAGES
        },
    }
    layer_bridge_path = out_dir / "bridge_n1000_layer_sweep.json"
    layer_bridge = json.loads(layer_bridge_path.read_text()) if layer_bridge_path.exists() else None
    report = {
        "metadata": {
            "design": "paired OLMo u_last-vs-u_mean diagnostic",
            "hf_repo": HF_REPO,
            "hf_revision": HF_REVISION,
            "layer": LAYER,
            "stages": list(STAGES),
            "corpora": list(CORPORA),
            "fitter": "ridge; original group-OOF uses #1902 GCV grid",
            "primary_split": "original six semantic-group folds",
            "random_split": "six size-matched random row folds",
            "bootstrap": f"{N_BOOT} paired semantic-group draws, seed {BOOT_SEED}",
        },
        "cells": cells,
        "n1000_protocol_bridge": bridge_summary,
        "n1000_layer_selection": None if layer_bridge is None else layer_bridge["selection"],
        "parity": json.loads((out_dir / "primal_mean_parity.json").read_text()),
    }
    _write_json(out_dir / "summary.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "stage",
            "stage_layers",
            "stage_qwen",
            "stage_reliability",
            "parity",
            "fit",
            "random",
            "bridge",
            "layer_bridge",
            "layer_bridge_generic",
            "native_bridge",
            "exact_prompt_bridge",
            "large_exact_prompt_bridge",
            "reliability",
            "answer_lengths",
            "cross_answer_source",
            "consolidate",
            "analyze",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blas-threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phases: Iterable[str]
    phases = (
        ("stage", "parity", "fit", "random", "bridge", "analyze")
        if args.phase == "all"
        else (args.phase,)
    )
    for phase in phases:
        print(f"[phase] {phase}", flush=True)
        if phase == "stage":
            _write_json(args.out_dir / "stage_manifest.json", stage_files(args.stage_root))
        elif phase == "stage_qwen":
            _write_json(
                args.out_dir / "stage_qwen_manifest.json",
                stage_qwen_files(args.stage_root, args.qwen_root),
            )
        elif phase == "stage_reliability":
            _write_json(
                args.out_dir / "stage_reliability_manifest.json",
                stage_reliability_files(args.stage_root),
            )
        elif phase == "stage_layers":
            records = [
                stage_files(args.stage_root, layer=layer, corpora=("single",))
                for layer in CAPTURED_LAYERS
            ]
            _write_json(
                args.out_dir / "stage_single_all_layers_manifest.json",
                {"revision": HF_REVISION, "layers": records},
            )
        elif phase == "parity":
            print(
                json.dumps(
                    parity_check(
                        args.stage_root,
                        args.out_dir,
                        blas_threads=args.blas_threads,
                        force=args.force,
                    ),
                    indent=2,
                )
            )
        elif phase == "fit":
            fit_cells(
                args.stage_root,
                args.out_dir,
                summary="u_last",
                fold_mode="group",
                workers=args.workers,
                blas_threads=args.blas_threads,
                force=args.force,
            )
        elif phase == "random":
            fit_cells(
                args.stage_root,
                args.out_dir,
                summary="u_last",
                fold_mode="random",
                workers=args.workers,
                blas_threads=args.blas_threads,
                force=args.force,
            )
        elif phase == "bridge":
            run_bridge(
                args.stage_root,
                args.out_dir,
                workers=args.workers,
                blas_threads=args.blas_threads,
                force=args.force,
            )
        elif phase in ("layer_bridge", "layer_bridge_generic"):
            run_layer_bridge(
                args.stage_root,
                args.out_dir,
                workers=args.workers,
                blas_threads=args.blas_threads,
                generic_only=phase == "layer_bridge_generic",
                force=args.force,
            )
        elif phase == "native_bridge":
            run_native_bridge(
                args.stage_root,
                args.out_dir,
                workers=args.workers,
                blas_threads=args.blas_threads,
                force=args.force,
            )
        elif phase == "exact_prompt_bridge":
            run_exact_prompt_bridge(
                args.stage_root,
                args.qwen_root,
                args.out_dir,
                workers=args.workers,
                blas_threads=args.blas_threads,
                force=args.force,
            )
        elif phase == "large_exact_prompt_bridge":
            large = run_large_exact_prompt_bridge(
                args.stage_root,
                args.qwen_root,
                args.out_dir,
                blas_threads=args.blas_threads,
                force=args.force,
            )
            print(
                json.dumps(
                    {
                        "n_train_each": large["protocol"]["n_train_each"],
                        "qwen_mean_test_r2": large["qwen"]["mean_test_r2"],
                        "olmo_mean_test_r2": {
                            stage: large["olmo"][stage]["mean_test_r2"] for stage in ("S", "D", "R")
                        },
                        "qwen_minus_olmo": large["contrasts"]["qwen_minus_olmo_mean"],
                    },
                    indent=2,
                )
            )
        elif phase == "reliability":
            print(
                json.dumps(
                    run_reliability(args.stage_root, args.out_dir, force=args.force),
                    indent=2,
                )
            )
        elif phase == "answer_lengths":
            print(
                json.dumps(
                    run_answer_lengths(
                        args.stage_root, args.qwen_root, args.out_dir, force=args.force
                    ),
                    indent=2,
                )
            )
        elif phase == "cross_answer_source":
            print(json.dumps(run_cross_answer_source(args.out_dir, force=args.force), indent=2))
        elif phase == "consolidate":
            print(json.dumps(run_consolidate(args.out_dir, force=args.force), indent=2))
        elif phase == "analyze":
            print(json.dumps(analyze(args.stage_root, args.out_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
