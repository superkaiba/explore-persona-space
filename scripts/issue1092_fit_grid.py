#!/usr/bin/env python3
"""Issue #1092 P6 fit grid: CPU ridge maps, spectra, nulls, and behavior joins.

The production path is layer-staged and checkpointed. The smoke path consumes
tiny real P3 summary shards and runs the same #923 PRESS ridge engine, #813
factored spectrum helpers, identity-baseline floors, and permutation-null seam.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue658_fit_predictors import RIDGE_LAMBDAS  # noqa: E402
from issue779_identity_baseline import (  # noqa: E402
    CHEAP_RUNGS,
    _fit_diag_affine,
    _fit_global_affine,
)
from issue813_rank_spectrum import (  # noqa: E402
    _fit_pieces,
    _gcv_lambda,
    _sigma2,
    _spectrum_stats,
    _standardize,
)
from issue923_fit_decomposition import press_fit_predict, run_selftest  # noqa: E402

from explore_persona_space.analysis.null_battery import _k_chunks  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

SUMMARY_KINDS = ("prefix_end", "context_end", "t1", "t2", "t3")
INPUT_ARMS = ("prefix_end", "context_end")
TARGETS = ("t1", "t2", "t3")
FOLD_SEED = 42
FROZEN_NULL_LAYERS = {14, 18, 19}
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
DEFAULT_RB_REV = "037fcbb"
CELL_MODEL_TYPE = {
    "cell_inst_own": "instruct",
    "cell_pre_insttext": "pretrained",
    "cell_pre_own": "pretrained",
    "cell_inst_pretext": "instruct",
    "cell_inst_claude": "instruct",
    "cell_pre_claude": "pretrained",
    "cell_inst_shuf": "instruct",
    "cell_pre_shuf": "pretrained",
}


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_csv(value: str | None, default: Iterable[str]) -> list[str]:
    if value is None:
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def _fingerprint(paths: list[Path], config: dict) -> str:
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for path in sorted(paths):
        st = path.stat()
        h.update(path.name.encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:24]


def _load_summary(
    summaries_dir: Path, cell: str, kind: str, layer: int
) -> tuple[np.ndarray, list[Path]]:
    paths = sorted((summaries_dir / cell).glob(f"{kind}_L{layer:02d}_shard*.npy"))
    if not paths:
        paths = sorted((summaries_dir / cell).glob(f"{kind}_L{layer:02d}.npy"))
    if not paths:
        raise FileNotFoundError(f"no summary shards for {cell}/{kind}/L{layer:02d}")
    arrays = [np.load(p).astype(np.float64) for p in paths]
    return np.concatenate(arrays, axis=0), paths


def _folds_from_manifest(
    rows: list[dict], n: int, *, group_key: str, n_folds: int
) -> list[np.ndarray]:
    groups = [str(r.get(group_key, r.get("prefix_id", i))) for i, r in enumerate(rows[:n])]
    uniq = sorted(set(groups))
    rng = np.random.default_rng(FOLD_SEED)
    rng.shuffle(uniq)
    fold_groups = [set(uniq[i::n_folds]) for i in range(n_folds)]
    folds = [
        np.array([i for i, g in enumerate(groups) if g in fg], dtype=np.int64) for fg in fold_groups
    ]
    return [f for f in folds if f.size]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean(axis=0, keepdims=True)) ** 2).sum())
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _identity_floors(X: np.ndarray, Y: np.ndarray, folds: list[np.ndarray]) -> dict:
    out = {r: [] for r in CHEAP_RUNGS}
    n = X.shape[0]
    for test_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        Xtr, Ytr = X[mask], Y[mask]
        Xte, Yte = X[test_idx], Y[test_idx]
        out["train_mean"].append(_r2(Yte, np.broadcast_to(Ytr.mean(axis=0), Yte.shape)))
        if X.shape[1] == Y.shape[1]:
            out["raw_identity"].append(_r2(Yte, Xte))
            alpha, xmu, ymu = _fit_global_affine(Xtr, Ytr)
            out["global_affine"].append(_r2(Yte, ymu + alpha * (Xte - xmu)))
            a, xmu_d, ymu_d = _fit_diag_affine(Xtr, Ytr)
            out["diag_affine"].append(_r2(Yte, ymu_d + a * (Xte - xmu_d)))
        else:
            for rung in ("raw_identity", "global_affine", "diag_affine"):
                out[rung].append(float("nan"))
    summary = {}
    for rung, vals in out.items():
        arr = np.asarray(vals, dtype=np.float64)
        mean = float("nan") if np.all(np.isnan(arr)) else float(np.nanmean(arr))
        summary[rung] = {"mean": mean, "folds": [float(v) for v in vals]}
    return summary


def _fit_cv(
    X: np.ndarray, Y: np.ndarray, folds: list[np.ndarray], *, return_pred: bool = False
) -> dict | tuple[dict, np.ndarray]:
    n = X.shape[0]
    pred = np.zeros_like(Y, dtype=np.float64)
    lambdas: list[int] = []
    fold_r2: list[float] = []
    for test_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        res = press_fit_predict(
            torch.from_numpy(X[mask]).double(),
            torch.from_numpy(Y[mask]).double(),
            torch.from_numpy(X[test_idx]).double(),
            standardize=True,
        )
        pred[test_idx] = res["pred"].detach().cpu().numpy()
        lambdas.append(int(res["lam_idx"]))
        fold_r2.append(_r2(Y[test_idx], pred[test_idx]))
    out = {
        "r2": _r2(Y, pred),
        "r2_folds": fold_r2,
        "lambda_indices": lambdas,
    }
    if return_pred:
        return out, pred
    return out


def _spectrum(X: np.ndarray, Y: np.ndarray) -> dict:
    Xn, _mu, _sd = _standardize(X)
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).double()
    pieces = _fit_pieces(Xn, Yt)
    e = pieces["e"].detach().cpu().numpy()
    diag = torch.diag(pieces["W_yy"]).detach().cpu().numpy()
    lam = _gcv_lambda(e, diag, X.shape[0])
    sig = torch.sqrt(_sigma2(pieces["e"], pieces["W_yy"], lam)).detach().cpu().numpy()
    return {"lambda_gcv": float(lam), "stats": _spectrum_stats(sig)}


def _pca_basis(Y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mu = Y.mean(axis=0, keepdims=True)
    yc = Y - mu
    _u, _s, vh = np.linalg.svd(yc, full_matrices=False)
    kk = min(k, vh.shape[0])
    return mu, vh[:kk].T


def _basis_targets(Y: np.ndarray, basis: str) -> np.ndarray:
    if basis == "ambient":
        return Y
    if basis == "pca48":
        mu, v = _pca_basis(Y, 48)
        return (Y - mu) @ v
    raise ValueError(f"unknown target basis {basis}")


def _perm_null(
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[np.ndarray],
    n_draws: int,
    seed: int,
    *,
    lambda_indices: list[int],
) -> dict:
    """Batched pairing-permutation null with shared fold factorizations.

    This is the issue1092 production null battery: no per-draw full refits.
    For each fold, S = X_train^T X_train is factored once, all permuted
    train-target blocks are stacked, and the draw axis is solved via batched
    GEMM/einsum. `_k_chunks` is reused from `analysis.null_battery` for the same
    draw-chunking policy as the #834 vectorized null helpers.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    out_dim = Y.shape[1]
    if n_draws <= 0:
        return {"n_draws": 0, "p95": float("nan"), "draws": [], "batched": True}
    if len(lambda_indices) != len(folds):
        raise ValueError(f"null lambda_indices length {len(lambda_indices)} != folds {len(folds)}")
    perms = np.argsort(rng.random((n_draws, n)), axis=1).astype(np.int64)
    ss_res = np.zeros(n_draws, dtype=np.float64)
    ss_tot = float(((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum())
    for fold_i, test_idx in enumerate(folds):
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        Xtr = X[mask]
        Xte = X[test_idx]
        xmu = Xtr.mean(axis=0, keepdims=True)
        xsd = Xtr.std(axis=0, keepdims=True)
        xsd = np.where(xsd == 0.0, 1.0, xsd)
        Xtrn = (Xtr - xmu) / xsd
        Xten = (Xte - xmu) / xsd
        ridge = float(RIDGE_LAMBDAS[int(lambda_indices[fold_i])])
        gram = Xtrn.T @ Xtrn + ridge * np.eye(d, dtype=np.float64)
        solved_xt = np.linalg.solve(gram, Xtrn.T)  # (d, n_train), factored once per fold
        bytes_per_draw = max(
            1,
            mask.sum() * out_dim * 8 + test_idx.size * out_dim * 8 + d * out_dim * 8,
        )
        for start, stop in _k_chunks(n_draws, bytes_per_draw):
            target_train = Y[perms[start:stop][:, mask], :]  # (draw, n_train, out_dim)
            ymu = target_train.mean(axis=1, keepdims=True)
            centered = target_train - ymu
            weights = np.einsum("dn,kno->kdo", solved_xt, centered, optimize=True)
            pred = np.einsum("td,kdo->kto", Xten, weights, optimize=True) + ymu
            target_test = Y[perms[start:stop][:, test_idx], :]
            ss_res[start:stop] += ((target_test - pred) ** 2).sum(axis=(1, 2))
    vals = np.full(n_draws, np.nan, dtype=np.float64)
    if ss_tot != 0.0:
        vals = 1.0 - ss_res / ss_tot
    return {
        "n_draws": n_draws,
        "p95": float(np.nanpercentile(vals, 95)) if vals.size else float("nan"),
        "batched": True,
        "shared_factorization": True,
        "refit_with_same_lambda": True,
        "lambda_indices": [int(i) for i in lambda_indices],
        "draws": [float(v) for v in vals],
    }


def _anova_shares(rows: list[dict], Y: np.ndarray) -> dict:
    scoped = list(rows[: Y.shape[0]])
    dense_idx = [i for i, row in enumerate(scoped) if row.get("stratum") == "dense_core"]
    basis = "dense_core"
    if dense_idx:
        Y_use = Y[np.asarray(dense_idx, dtype=np.int64)]
        scoped = [scoped[i] for i in dense_idx]
    else:
        # Tiny smokes often omit dense_core labels; production never should.
        Y_use = Y
        basis = "all_rows_no_dense_core_smoke_fallback"
    prefix_ids = np.array([r.get("prefix_id", "") for r in scoped])
    query_ids = np.array([r.get("query_id", "") for r in scoped])
    yc = Y_use - Y_use.mean(axis=0, keepdims=True)
    f = np.zeros_like(yc)
    g = np.zeros_like(yc)
    for pid in sorted(set(prefix_ids)):
        f[prefix_ids == pid] = yc[prefix_ids == pid].mean(axis=0, keepdims=True)
    for qid in sorted(set(query_ids)):
        g[query_ids == qid] = yc[query_ids == qid].mean(axis=0, keepdims=True)
    i = yc - f - g
    ss = float((yc * yc).sum())
    return {
        "share_prefix": float((f * f).sum() / ss) if ss else float("nan"),
        "share_query": float((g * g).sum() / ss) if ss else float("nan"),
        "share_interaction": float((i * i).sum() / ss) if ss else float("nan"),
        "ss_total": ss,
        "basis": basis,
        "n_rows": int(Y_use.shape[0]),
    }


def _group_average(
    rows: list[dict], X: np.ndarray, Y: np.ndarray, key: str
) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows[: X.shape[0]]):
        groups.setdefault(str(row.get(key, i)), []).append(i)
    x_avg = []
    y_avg = []
    for idx in groups.values():
        arr = np.asarray(idx, dtype=np.int64)
        x_avg.append(X[arr].mean(axis=0))
        y_avg.append(Y[arr].mean(axis=0))
    return np.asarray(x_avg, dtype=np.float64), np.asarray(y_avg, dtype=np.float64)


def _matched_n_grain_read(
    rows: list[dict],
    X: np.ndarray,
    Y: np.ndarray,
    *,
    matched_n_draws: int,
    seed: int,
) -> dict:
    """Read 2: averaged-vs-per-example spectra with live matched-n draws."""
    avg_x, avg_y = _group_average(rows, X, Y, "prefix_id")
    per_spec = _spectrum(X, Y)
    avg_spec = _spectrum(avg_x, avg_y) if avg_x.shape[0] >= 3 else {"stats": {}}
    rng = np.random.default_rng(seed)
    n_match = min(X.shape[0], max(1, avg_x.shape[0]))
    draws = []
    for draw_idx in range(matched_n_draws):
        idx = rng.choice(X.shape[0], size=n_match, replace=False)
        draw_spec = _spectrum(X[idx], Y[idx])
        draws.append({"draw": draw_idx, "stats": draw_spec["stats"]})
    return {
        "matched_n_draws": matched_n_draws,
        "n_per_example": int(X.shape[0]),
        "n_averaged": int(avg_x.shape[0]),
        "per_example": per_spec["stats"],
        "averaged": avg_spec.get("stats", {}),
        "matched_n": draws,
    }


def _factor_components_dense_core(rows: list[dict], Y: np.ndarray) -> dict[str, np.ndarray | str]:
    scoped = list(rows[: Y.shape[0]])
    dense_idx = [i for i, row in enumerate(scoped) if row.get("stratum") == "dense_core"]
    if dense_idx:
        idx = np.asarray(dense_idx, dtype=np.int64)
        scoped = [scoped[i] for i in dense_idx]
        Y_use = Y[idx]
        basis = "dense_core"
    else:
        idx = np.arange(Y.shape[0], dtype=np.int64)
        scoped = scoped
        Y_use = Y
        basis = "all_rows_no_dense_core_smoke_fallback"
    prefix_ids = np.array([row.get("prefix_id", "") for row in scoped])
    query_ids = np.array([row.get("query_id", "") for row in scoped])
    yc = Y_use - Y_use.mean(axis=0, keepdims=True)
    f = np.zeros_like(yc)
    g = np.zeros_like(yc)
    for pid in sorted(set(prefix_ids)):
        f[prefix_ids == pid] = yc[prefix_ids == pid].mean(axis=0, keepdims=True)
    for qid in sorted(set(query_ids)):
        g[query_ids == qid] = yc[query_ids == qid].mean(axis=0, keepdims=True)
    i = yc - f - g
    return {"f": f, "g": g, "i": i, "yc": yc, "basis": basis, "indices": idx}


def _principal_angles(A: np.ndarray, B: np.ndarray, rank: int = 16) -> list[float]:
    if A.shape[0] < 2 or B.shape[0] < 2:
        return []
    _, _, vha = np.linalg.svd(A - A.mean(axis=0, keepdims=True), full_matrices=False)
    _, _, vhb = np.linalg.svd(B - B.mean(axis=0, keepdims=True), full_matrices=False)
    r = min(rank, vha.shape[0], vhb.shape[0])
    if r == 0:
        return []
    svals = np.linalg.svd(vha[:r] @ vhb[:r].T, compute_uv=False)
    svals = np.clip(svals, -1.0, 1.0)
    return [float(np.arccos(s)) for s in svals]


def _operator_identity_read(
    rows: list[dict],
    X: np.ndarray,
    Y: np.ndarray,
    *,
    seed: int,
    n_draws: int,
) -> dict:
    """Read 4: entailed sanity plus de-tautologized residual magnitudes."""
    factors = _factor_components_dense_core(rows, Y)
    f = np.asarray(factors["f"])
    g = np.asarray(factors["g"])
    interaction = np.asarray(factors["i"])
    yc = np.asarray(factors["yc"])
    residual_norm = float(np.linalg.norm(interaction) / max(np.linalg.norm(yc), 1e-12))
    mprime_minus_m_minus_g = yc - f - g
    g_norm = max(float(np.linalg.norm(g)), 1e-12)
    rng = np.random.default_rng(seed)
    nulls = []
    for _draw in range(n_draws):
        perm = rng.permutation(yc.shape[0])
        nulls.append(float(np.linalg.norm(yc[perm] - f - g) / g_norm))
    procrustes_num = float(np.linalg.norm((X[: yc.shape[0]] - X[: yc.shape[0]].mean(0)) - f))
    procrustes_den = max(float(np.linalg.norm(f)), 1e-12)
    return {
        "basis": factors["basis"],
        "entailed_m_approx_f_const": {
            "principal_angles_rad": _principal_angles(X[: yc.shape[0]], f),
            "procrustes_residual_over_f": procrustes_num / procrustes_den,
            "interpretation": "rig_sanity_only_not_HA_evidence",
        },
        "residual_interaction_norm_over_total": residual_norm,
        "mprime_minus_m_minus_g_over_g": float(np.linalg.norm(mprime_minus_m_minus_g) / g_norm),
        "random_map_pairing_null": {
            "n_draws": len(nulls),
            "p05": float(np.nanpercentile(nulls, 5)) if nulls else float("nan"),
            "p95": float(np.nanpercentile(nulls, 95)) if nulls else float("nan"),
            "draws": nulls,
        },
    }


def _refit_twins(rows: list[dict], Y: np.ndarray, *, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(Y.shape[0])
    halves = np.array_split(idx, 2)
    return {
        f"twin_{i}": _anova_shares([rows[j] for j in half], Y[half])
        for i, half in enumerate(halves)
    }


def _pearson_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    xc = x.astype(np.float64) - float(np.mean(x))
    yc = y.astype(np.float64) - float(np.mean(y))
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    return float("nan") if denom == 0.0 else float(np.dot(xc, yc) / denom)


def _load_judge_score_rows(path: Path | None) -> list[dict]:
    if path is None:
        return []
    return _jsonl(path)


def _read_index_files(root: Path, stem: str) -> list[dict]:
    paths = sorted(root.glob(f"{stem}_shard*.jsonl"))
    if not paths:
        path = root / f"{stem}.jsonl"
        paths = [path] if path.exists() else []
    if not paths:
        raise FileNotFoundError(f"missing index files {root}/{stem}[_shard*.jsonl]")
    rows: list[dict] = []
    for path in paths:
        rows.extend(_jsonl(path))
    return rows


def _load_bare_rows(
    summaries_dir: Path, model_type: str, layer: int
) -> tuple[np.ndarray, dict[str, int]]:
    root = summaries_dir / f"bare_{model_type}"
    arr, _paths = _load_summary(summaries_dir, f"bare_{model_type}", "c_q_bare", layer)
    index_rows = _read_index_files(root, "row_index")
    if len(index_rows) != arr.shape[0]:
        raise ValueError(f"{root} row_index count {len(index_rows)} != rows {arr.shape[0]}")
    q_to_idx = {str(row["query_id"]): i for i, row in enumerate(index_rows)}
    return arr, q_to_idx


def _load_b0_pool(summaries_dir: Path, cell: str) -> np.ndarray:
    root = summaries_dir / "b0_rB_pool"
    paths = sorted(root.glob(f"{cell}_shard*.npy"))
    if not paths:
        paths = sorted(root.glob(f"{cell}.npy"))
    if not paths:
        raise FileNotFoundError(f"missing B0 pool artifact for {cell}: {root}/{cell}*.npy")
    return np.concatenate([np.load(path).astype(np.float64) for path in paths], axis=0)


def _load_rb_directions(args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    paths: list[Path] = []
    if args.rb_dir is not None:
        rb_dir = Path(args.rb_dir)
        if rb_dir.is_file() and rb_dir.suffix == ".npy":
            arr = np.load(rb_dir).astype(np.float64)
            names = [f"trait_{i}" for i in range(arr.shape[1])]
            sidecar = rb_dir.with_name("trait_names.json")
            if sidecar.exists():
                names = [str(x) for x in json.loads(sidecar.read_text())]
            return arr, names
        paths = sorted(rb_dir.glob("*.pt")) + sorted(rb_dir.glob("*.npy"))
    else:
        from huggingface_hub import hf_hub_download, list_repo_tree

        prefix = "issue779_monitoring/r_b"
        entries = list_repo_tree(
            HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=prefix,
            revision=args.rb_rev,
        )
        relpaths = sorted(
            entry.path
            for entry in entries
            if getattr(entry, "size", None) is not None and entry.path.endswith(".pt")
        )
        if not relpaths:
            raise FileNotFoundError(f"no r_B .pt files under {HF_DATA_REPO}@{args.rb_rev}:{prefix}")
        paths = [
            Path(
                hf_hub_download(
                    repo_id=HF_DATA_REPO,
                    repo_type="dataset",
                    filename=relpath,
                    revision=args.rb_rev,
                )
            )
            for relpath in relpaths
        ]
    if not paths:
        raise FileNotFoundError(f"no r_B files found in --rb-dir {args.rb_dir}")
    tensors: list[np.ndarray] = []
    names: list[str] = []
    for path in paths:
        if path.suffix == ".npy":
            arr = np.load(path)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            arr = payload["r_b"] if isinstance(payload, dict) and "r_b" in payload else payload
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (args.n_layers, args.hidden_dim):
            raise ValueError(
                f"r_B file {path} shape {arr.shape} != ({args.n_layers}, {args.hidden_dim})"
            )
        tensors.append(arr)
        names.append(path.stem)
    return np.stack(tensors, axis=1), names


def _trait_index(trait_names: list[str], trait: str) -> int | None:
    if trait in trait_names:
        return trait_names.index(trait)
    for i, name in enumerate(trait_names):
        if trait in name or name in trait:
            return i
    return None


def _registered_stitch_reads(
    *,
    summaries_dir: Path,
    cell: str,
    layer: int,
    unit_rows: list[dict],
    prefix_X: np.ndarray,
    context_X: np.ndarray,
    Y: np.ndarray,
    folds: list[np.ndarray],
    args: argparse.Namespace,
) -> dict:
    model_type = CELL_MODEL_TYPE.get(cell)
    if model_type is None:
        raise ValueError(f"cannot infer bare-query model type for cell {cell}")
    bare_arr, q_to_idx = _load_bare_rows(summaries_dir, model_type, layer)
    missing = [
        str(row.get("query_id")) for row in unit_rows if str(row.get("query_id")) not in q_to_idx
    ]
    if missing:
        raise KeyError(
            f"bare-query g-map missing {len(missing)} query ids for {cell}: {missing[:5]}"
        )
    bare_X = bare_arr[[q_to_idx[str(row.get("query_id"))] for row in unit_rows]]
    query_folds = _folds_from_manifest(
        unit_rows,
        len(unit_rows),
        group_key="query_id",
        n_folds=max(2, min(args.n_folds, len({row.get("query_id") for row in unit_rows}))),
    )
    if len(query_folds) < 2:
        query_folds = folds
    stitched = np.concatenate([prefix_X, bare_X], axis=1)
    return {
        "status": "computed",
        "model_type": model_type,
        "query_only_g_map": _fit_cv(bare_X, Y, query_folds),
        "prefix_only": _fit_cv(prefix_X, Y, folds),
        "stitch_prefix_plus_bare": _fit_cv(stitched, Y, folds),
        "mixed_forward_context_reference": _fit_cv(context_X, Y, folds),
    }


def _fit_pair_read(name: str, X: np.ndarray, Y: np.ndarray, rows: list[dict], args) -> dict:
    if X.shape[0] < 3 or Y.shape[0] < 3:
        return {"status": "insufficient_rows", "read": name, "n": int(min(X.shape[0], Y.shape[0]))}
    n_folds = max(2, min(args.n_folds, len({row.get("conv_id") for row in rows})))
    folds = _folds_from_manifest(rows, len(rows), group_key="conv_id", n_folds=n_folds)
    if len(folds) < 2 or any(fold.size >= len(rows) for fold in folds):
        return {"status": "insufficient_folds", "read": name, "n": len(rows)}
    return {"status": "computed", "read": name, "fit": _fit_cv(X, Y, folds)}


def _compute_dynamics_reads(  # noqa: C901
    summaries_dir: Path, cell: str, layer: int, args, judge_rows: list[dict]
) -> dict:
    model_type = CELL_MODEL_TYPE.get(cell)
    if model_type is None:
        raise ValueError(f"cannot infer dynamics model type for cell {cell}")
    root = summaries_dir / f"dynamics_{model_type}"
    kinds = ("context_k", "s_k", "answer_k_t1", "u1")
    arrays: dict[str, np.ndarray] = {}
    indices: dict[str, dict[tuple[str, int], int]] = {}
    index_rows_by_kind: dict[str, list[dict]] = {}
    for kind in kinds:
        arrays[kind], _paths = _load_summary(summaries_dir, f"dynamics_{model_type}", kind, layer)
        rows = _read_index_files(root, f"row_index_{kind}")
        if len(rows) != arrays[kind].shape[0]:
            raise ValueError(
                f"{root}/{kind} index rows {len(rows)} != array rows {arrays[kind].shape[0]}"
            )
        index_rows_by_kind[kind] = rows
        indices[kind] = {
            (str(row["conv_id"]), int(row["turn_index"])): i for i, row in enumerate(rows)
        }

    def pairs(src_kind: str, dst_kind: str, *, offset: int = 0):
        src = []
        dst = []
        pair_rows = []
        for key, src_i in indices[src_kind].items():
            conv_id, turn_idx = key
            dst_key = (conv_id, turn_idx + offset)
            if dst_key not in indices[dst_kind]:
                continue
            src.append(arrays[src_kind][src_i])
            dst.append(arrays[dst_kind][indices[dst_kind][dst_key]])
            pair_rows.append({"conv_id": conv_id, "turn_index": turn_idx})
        if not src:
            return np.empty((0, args.hidden_dim)), np.empty((0, args.hidden_dim)), []
        return np.asarray(src), np.asarray(dst), pair_rows

    out: dict[str, Any] = {"status": "computed", "model_type": model_type}
    for name, src_kind, dst_kind, offset in (
        ("D0_context_to_answer_k", "context_k", "answer_k_t1", 0),
        ("D1_s_to_user_k_plus_1", "s_k", "u1", 1),
        ("D2_context_to_context_next", "context_k", "context_k", 2),
        ("D3_context_to_answer_next", "context_k", "answer_k_t1", 2),
    ):
        Xp, Yp, pair_rows = pairs(src_kind, dst_kind, offset=offset)
        out[name] = _fit_pair_read(name, Xp, Yp, pair_rows, args)

    d4: dict[str, Any] = {}
    X0, Y0, rows0 = pairs("context_k", "answer_k_t1", offset=0)
    for turn_idx in sorted({row["turn_index"] for row in rows0}):
        idx = [i for i, row in enumerate(rows0) if row["turn_index"] == turn_idx]
        if len(idx) >= 3:
            d4[str(turn_idx)] = _fit_pair_read(
                f"D4_turn_{turn_idx}", X0[idx], Y0[idx], [rows0[i] for i in idx], args
            )
    out["D4_turn_profiles"] = d4

    first_by_conv: dict[str, int] = {}
    for key, src_i in indices["context_k"].items():
        conv_id, turn_idx = key
        current_first = first_by_conv.get(conv_id)
        current_turn = (
            None
            if current_first is None
            else index_rows_by_kind["context_k"][current_first]["turn_index"]
        )
        if current_turn is None or turn_idx < current_turn:
            first_by_conv[conv_id] = src_i
    d5: dict[str, Any] = {}
    for horizon in (0, 2, 4, 6):
        src = []
        dst = []
        pair_rows = []
        for conv_id, first_i in first_by_conv.items():
            first_turn = int(index_rows_by_kind["context_k"][first_i]["turn_index"])
            dst_key = (conv_id, first_turn + horizon)
            if dst_key in indices["answer_k_t1"]:
                src.append(arrays["context_k"][first_i])
                dst.append(arrays["answer_k_t1"][indices["answer_k_t1"][dst_key]])
                pair_rows.append({"conv_id": conv_id, "turn_index": first_turn})
        if src:
            d5[str(horizon)] = _fit_pair_read(
                f"D5_first_context_to_answer_h{horizon}",
                np.asarray(src),
                np.asarray(dst),
                pair_rows,
                args,
            )
    out["D5_first_state_horizon"] = d5
    b3_by_trait: dict[str, list[tuple[np.ndarray, float, dict]]] = {}
    for score_row in judge_rows:
        if score_row.get("arm") != "B3_dynamics" or score_row.get("score") is None:
            continue
        conv_id = str(score_row.get("conv_id", ""))
        turn_index = score_row.get("turn_index")
        if turn_index is None:
            continue
        predictor_key = (conv_id, int(turn_index) - 2)
        if predictor_key not in indices["context_k"]:
            continue
        b3_by_trait.setdefault(str(score_row.get("trait")), []).append(
            (
                arrays["context_k"][indices["context_k"][predictor_key]],
                float(score_row["score"]),
                {"conv_id": conv_id, "turn_index": int(turn_index) - 2},
            )
        )
    b3: dict[str, Any] = {}
    for trait, triples in sorted(b3_by_trait.items()):
        Xb = np.asarray([item[0] for item in triples], dtype=np.float64)
        yb = np.asarray([item[1] for item in triples], dtype=np.float64)
        pair_rows = [item[2] for item in triples]
        n_groups = len({row["conv_id"] for row in pair_rows})
        if len(pair_rows) >= 3 and n_groups >= 2:
            b3[trait] = _fit_scalar_cv(
                Xb,
                yb,
                _folds_from_manifest(
                    pair_rows,
                    len(pair_rows),
                    group_key="conv_id",
                    n_folds=max(2, min(args.n_folds, n_groups)),
                ),
            )
        else:
            b3[trait] = {"status": "insufficient_rows", "n": len(pair_rows), "n_groups": n_groups}
    out["B3_context_to_judged_answer_next"] = {
        "status": "computed" if b3 else "no_scored_b3_rows",
        "traits": b3,
    }
    return out


def _selection_symmetric_projection_null(
    *,
    unit_key: str,
    factors: dict[str, np.ndarray | str],
    rb_directions: np.ndarray,
    trait_names: list[str],
    layer: int,
    n_draws: int,
    seed: int,
    out_dir: Path,
) -> dict:
    rng = np.random.default_rng(seed)
    rb = rb_directions[layer]
    factor_arrays = {name: np.asarray(factors[name], dtype=np.float64) for name in ("f", "g", "i")}
    observed = {
        factor: {trait_names[t]: float(np.mean(arr @ rb[t])) for t in range(len(trait_names))}
        for factor, arr in factor_arrays.items()
    }
    draws = np.empty((n_draws, len(factor_arrays), len(trait_names)), dtype=np.float64)
    for draw in range(n_draws):
        signs = rng.choice(np.array([-1.0, 1.0]), size=next(iter(factor_arrays.values())).shape[0])
        for f_i, arr in enumerate(factor_arrays.values()):
            signed = arr * signs[:, None]
            draws[draw, f_i, :] = np.abs(np.mean(signed @ rb.T, axis=0))
    max_draws = np.nanmax(draws, axis=(1, 2))
    null_dir = out_dir / "analysis_tensors" / "nulls"
    null_dir.mkdir(parents=True, exist_ok=True)
    persist = null_dir / f"{unit_key}_selection_projection_null.npy"
    np.save(persist, draws.astype(np.float32))
    return {
        "status": "computed",
        "n_draws": int(n_draws),
        "layer": int(layer),
        "trait_names": trait_names,
        "observed_mean_projection": observed,
        "max_abs_p95": float(np.nanpercentile(max_draws, 95)),
        "persist_path": str(persist),
    }


def _mlp_companion_read(
    *,
    cell: str,
    arm: str,
    fit_arm: str,
    basis: str,
    X: np.ndarray,
    Y: np.ndarray,
    unit_rows: list[dict],
    args: argparse.Namespace,
) -> dict:
    if not args.run_mlp_companion:
        return {"status": "not_requested"}
    if not (cell == "cell_inst_own" and arm == "context_end" and fit_arm == "A"):
        return {"status": "not_applicable"}
    from explore_persona_space.analysis.vectorized_mlp_skill import (
        MLPGroup,
        fit_batched_loco_mlp_multihead,
        skill_over_mean_r2,
    )

    target_dim = min(args.mlp_target_dim, Y.shape[1])
    Y_use = _basis_targets(Y, "pca48")[:, :target_dim] if basis == "ambient" else Y[:, :target_dim]
    group_labels = np.array(
        [
            int(hashlib.sha256(str(row.get("prefix_id", i)).encode()).hexdigest()[:8], 16)
            for i, row in enumerate(unit_rows)
        ],
        dtype=np.int64,
    )
    result = fit_batched_loco_mlp_multihead(
        [MLPGroup((cell, arm, fit_arm, basis), X, Y_use)],
        seed=args.seed,
        max_epochs=args.mlp_max_epochs,
        hidden=args.mlp_hidden,
        chunk_size=args.mlp_chunk_size,
        row_groups=group_labels,
        device="cpu",
    )
    pred = result.preds_by_key[(cell, arm, fit_arm, basis)]
    return {
        "status": "computed",
        "helper": (
            "explore_persona_space.analysis.vectorized_mlp_skill.fit_batched_loco_mlp_multihead"
        ),
        "target_dim": int(target_dim),
        "n_folds": int(result.n_folds),
        "skill_over_mean": skill_over_mean_r2(pred, Y_use),
    }


def _fit_scalar_cv(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> dict:
    y2 = y.reshape(-1, 1).astype(np.float64)
    fit = _fit_cv(X, y2, folds)
    train_mean = []
    n = X.shape[0]
    for test_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        train_mean.append(
            _r2(y2[test_idx], np.broadcast_to(y2[mask].mean(axis=0), y2[test_idx].shape))
        )
    fit["train_mean_floor"] = {
        "mean": float(np.nanmean(train_mean)) if train_mean else float("nan"),
        "folds": [float(v) for v in train_mean],
    }
    return fit


def _behavior_reads(
    *,
    cell: str,
    unit_rows: list[dict],
    X: np.ndarray,
    Y: np.ndarray,
    fit_pred: np.ndarray,
    folds: list[np.ndarray],
    judge_rows: list[dict],
    rb_directions: np.ndarray,
    trait_names: list[str],
    b0_pool: np.ndarray | None,
    unit_source_indices: np.ndarray,
    layer: int,
) -> dict:
    """Compute the registered B1 panel plus B2 factor-to-behavior reads."""
    if not judge_rows:
        return {"status": "not_requested"}
    row_pos = {str(row.get("row_id")): i for i, row in enumerate(unit_rows)}
    by_trait: dict[str, list[tuple[int, float]]] = {}
    for score_row in judge_rows:
        if score_row.get("cell_id") != cell and score_row.get("arm") != cell:
            continue
        score = score_row.get("score")
        row_id = str(score_row.get("row_id"))
        if score is None or row_id not in row_pos:
            continue
        by_trait.setdefault(str(score_row.get("trait")), []).append((row_pos[row_id], float(score)))

    factors = _factor_components_dense_core(unit_rows, Y)
    factor_indices = np.asarray(factors["indices"], dtype=np.int64)
    factor_pos = {int(src_i): i for i, src_i in enumerate(factor_indices.tolist())}
    out: dict[str, dict] = {}
    for trait, pairs in sorted(by_trait.items()):
        idx = np.asarray([p[0] for p in pairs], dtype=np.int64)
        scores = np.asarray([p[1] for p in pairs], dtype=np.float64)
        positives = int(np.sum(scores > 50.0))
        negatives = int(scores.size - positives)
        std = float(np.std(scores)) if scores.size else float("nan")
        estimable = bool(scores.size >= 5 and std >= 1.0 and positives >= 1 and negatives >= 1)
        entry: dict[str, Any] = {
            "n_scored": int(scores.size),
            "score_std": std,
            "n_positive": positives,
            "n_negative": negatives,
            "estimable": estimable,
        }
        if estimable:
            local_folds = _folds_from_manifest(
                [unit_rows[i] for i in idx],
                len(idx),
                group_key="prefix_id",
                n_folds=min(len(folds), max(2, len(idx) // 2)),
            )
            if len(local_folds) < 2 or any(fold.size >= len(idx) for fold in local_folds):
                entry["estimable"] = False
                entry["fold_guard"] = "grouped folds collapsed below two trainable splits"
                out[trait] = entry
                continue
            trait_i = _trait_index(trait_names, trait)
            if trait_i is None:
                raise KeyError(f"trait {trait!r} missing from r_B trait names {trait_names}")
            rb = rb_directions[layer, trait_i]
            rb_norm = np.linalg.norm(rb)
            if rb_norm == 0.0:
                raise ValueError(f"r_B direction for trait {trait} at layer {layer} has zero norm")
            raw_projection = X[idx] @ rb / rb_norm
            map_projection = fit_pred[idx] @ rb / rb_norm
            a2_projection = Y[idx] @ rb / rb_norm
            entry["B1_raw_projection"] = {"pearson_r": _pearson_or_nan(raw_projection, scores)}
            entry["B1_map_mediated"] = {"pearson_r": _pearson_or_nan(map_projection, scores)}
            entry["B1_direct_regression"] = _fit_scalar_cv(X[idx], scores, local_folds)
            entry["B1_A2_answer_side_ceiling"] = {
                "pearson_r": _pearson_or_nan(a2_projection, scores)
            }
            if b0_pool is not None and cell in {"cell_inst_own", "cell_pre_own"}:
                b0_idx = unit_source_indices[idx]
                if int(np.max(b0_idx)) >= b0_pool.shape[0]:
                    raise IndexError(
                        f"B0 pool for {cell} has {b0_pool.shape[0]} rows; "
                        f"need index {int(np.max(b0_idx))}"
                    )
                modes = ("mean", "max", "top3", "last")
                entry["B1_B0_poolings"] = {
                    mode: {
                        "pearson_r": _pearson_or_nan(
                            b0_pool[b0_idx, layer, trait_i, mode_i], scores
                        )
                    }
                    for mode_i, mode in enumerate(modes)
                }
            factor_scores = {}
            dense_pairs = [(factor_pos[int(i)], s) for i, s in pairs if int(i) in factor_pos]
            if dense_pairs:
                dense_idx = np.asarray([p[0] for p in dense_pairs], dtype=np.int64)
                dense_scores = np.asarray([p[1] for p in dense_pairs], dtype=np.float64)
                for factor_name in ("f", "g", "i"):
                    factor_arr = np.asarray(factors[factor_name])[dense_idx]
                    factor_proj = factor_arr @ rb / rb_norm
                    factor_scores[factor_name] = {
                        "norm_score_r": _pearson_or_nan(
                            np.linalg.norm(factor_arr, axis=1), dense_scores
                        ),
                        "rB_projection_score_r": _pearson_or_nan(factor_proj, dense_scores),
                    }
            entry["B2_factor_to_behavior"] = {
                "basis": factors["basis"],
                "factor_score_correlations": factor_scores,
            }
        out[trait] = entry
    return {
        "status": "computed",
        "eligibility_rule": "std>=1 and >=5 scored and at least one positive/negative",
        "traits": out,
    }


def _validate_registered_inputs(args: argparse.Namespace, summaries_dir: Path) -> dict:
    missing: list[str] = []
    bare_dirs = sorted(summaries_dir.glob("bare_*"))
    dynamics_dirs = sorted(summaries_dir.glob("dynamics_*"))
    if not bare_dirs:
        missing.append("summaries/bare_* for stitch/query-only/prefix-only reads")
    if not dynamics_dirs:
        missing.append("summaries/dynamics_* for D0-D5 reads")
    if args.judge_scores is None:
        missing.append("--judge-scores for B1/B2/B3 behavior reads")
    if args.rb_dir is not None and not Path(args.rb_dir).exists():
        missing.append(f"--rb-dir path does not exist: {args.rb_dir}")
    if missing and not args.allow_missing_registered_reads:
        raise FileNotFoundError("missing registered-read inputs: " + "; ".join(missing))
    return {
        "status": "present" if not missing else "missing_allowed",
        "bare_dirs": [str(path) for path in bare_dirs],
        "dynamics_dirs": [str(path) for path in dynamics_dirs],
        "judge_scores": str(args.judge_scores) if args.judge_scores else None,
        "rb_dir": (
            str(args.rb_dir)
            if args.rb_dir
            else f"HF:{HF_DATA_REPO}@{args.rb_rev}:issue779_monitoring/r_b"
        ),
        "missing": missing,
    }


def run(args: argparse.Namespace) -> dict:  # noqa: C901
    t0 = time.monotonic()
    run_selftest("cpu")
    summaries_dir = args.summaries_dir
    rows = _jsonl(args.corpus_dir / "manifest.jsonl")
    cells = _parse_csv(
        args.cells,
        [p.name for p in summaries_dir.iterdir() if p.is_dir() and p.name != "b0_rB_pool"],
    )
    arms = _parse_csv(args.arms, INPUT_ARMS)
    targets = _parse_csv(args.targets, TARGETS)
    fit_arms = _parse_csv(args.fit_arms, ("A", "B"))
    layers = _parse_layers(args.layers)
    bases = _parse_csv(args.target_bases, ("ambient", "pca48"))
    out_dir = args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    registered_inputs = _validate_registered_inputs(args, summaries_dir)
    judge_rows = _load_judge_score_rows(args.judge_scores)
    rb_directions, trait_names = _load_rb_directions(args)
    b0_pools: dict[str, np.ndarray] = {}

    units: list[dict] = []
    all_input_paths: list[Path] = []
    dynamics_cache: dict[tuple[str, int], dict] = {}
    for cell in cells:
        if cell in {"cell_inst_own", "cell_pre_own"}:
            try:
                b0_pools[cell] = _load_b0_pool(summaries_dir, cell)
            except FileNotFoundError:
                if not args.allow_missing_registered_reads:
                    raise
        for layer in layers:
            y_by_target: dict[str, np.ndarray] = {}
            y_paths: list[Path] = []
            for target in targets:
                y_by_target[target], paths = _load_summary(summaries_dir, cell, target, layer)
                all_input_paths.extend(paths)
                y_paths.extend(paths)
            Y_stacked = np.concatenate([y_by_target[t] for t in targets], axis=1)
            x_by_arm: dict[str, np.ndarray] = {}
            x_paths_by_arm: dict[str, list[Path]] = {}
            for arm_name in arms:
                x_by_arm[arm_name], paths = _load_summary(summaries_dir, cell, arm_name, layer)
                all_input_paths.extend(paths)
                x_paths_by_arm[arm_name] = paths
            for arm in arms:
                X = x_by_arm[arm]
                n0 = min(X.shape[0], Y_stacked.shape[0], len(rows))
                for fit_arm in fit_arms:
                    base_rows = rows[:n0]
                    if fit_arm == "A":
                        idx = [
                            i
                            for i, row in enumerate(base_rows)
                            if row.get("stratum") not in {"trait_stratum", "battery_eval_only"}
                        ]
                    elif fit_arm == "B":
                        idx = list(range(n0))
                    else:
                        raise ValueError(f"unknown fit arm {fit_arm!r}; expected A or B")
                    if len(idx) < max(3, args.n_folds):
                        raise ValueError(f"fit arm {fit_arm} has too few rows: {len(idx)}")
                    idx_arr = np.asarray(idx, dtype=np.int64)
                    Xn = X[idx_arr]
                    Yn = Y_stacked[idx_arr]
                    unit_rows = [base_rows[i] for i in idx]
                    folds = _folds_from_manifest(
                        unit_rows,
                        len(unit_rows),
                        group_key=args.group_key,
                        n_folds=args.n_folds,
                    )
                    for basis in bases:
                        Yb = _basis_targets(Yn, basis)
                        prefix_Xn = x_by_arm.get("prefix_end", X)[:n0][idx_arr]
                        context_Xn = x_by_arm.get("context_end", X)[:n0][idx_arr]
                        unit_null_draws = (
                            args.n_null_draws
                            if layer in FROZEN_NULL_LAYERS
                            else min(args.band_null_draws, args.n_null_draws)
                        )
                        config = {
                            "cell": cell,
                            "layer": layer,
                            "arm": arm,
                            "fit_arm": fit_arm,
                            "targets": targets,
                            "basis": basis,
                            "n": len(unit_rows),
                            "n_folds": args.n_folds,
                            "group_key": args.group_key,
                            "seed": args.seed,
                            "n_null_draws": unit_null_draws,
                            "matched_n_draws": args.matched_n_draws,
                        }
                        fp = _fingerprint(y_paths + x_paths_by_arm[arm], config)
                        ckpt = (
                            ckpt_dir / f"{cell}_{arm}_fit{fit_arm}_L{layer:02d}_{basis}_{fp}.json"
                        )
                        if ckpt.exists():
                            units.append(json.loads(ckpt.read_text()))
                            continue
                        fit, fit_pred = _fit_cv(Xn, Yb, folds, return_pred=True)
                        floors = _identity_floors(Xn, Yb, folds)
                        spec = _spectrum(Xn, Yb)
                        null = _perm_null(
                            Xn,
                            Yb,
                            folds,
                            unit_null_draws,
                            args.seed + layer,
                            lambda_indices=fit["lambda_indices"],
                        )
                        shares = _anova_shares(unit_rows, Yb)
                        read2 = _matched_n_grain_read(
                            unit_rows,
                            Xn,
                            Yb,
                            matched_n_draws=args.matched_n_draws,
                            seed=args.seed + layer,
                        )
                        read4 = _operator_identity_read(
                            unit_rows,
                            Xn,
                            Yb,
                            seed=args.seed + layer,
                            n_draws=unit_null_draws,
                        )
                        factors = _factor_components_dense_core(unit_rows, Yb)
                        unit_key = f"{cell}_{arm}_fit{fit_arm}_L{layer:02d}_{basis}_{fp}"
                        if (cell, layer) not in dynamics_cache:
                            dynamics_cache[(cell, layer)] = _compute_dynamics_reads(
                                summaries_dir, cell, layer, args, judge_rows
                            )
                        unit = {
                            "cell": cell,
                            "layer": layer,
                            "arm": arm,
                            "fit_arm": fit_arm,
                            "targets": targets,
                            "basis": basis,
                            "n_rows": len(unit_rows),
                            "fit": fit,
                            "identity_floors": floors,
                            "genuine_r2_over_diag": (
                                fit["r2"] - floors["diag_affine"]["mean"]
                                if not np.isnan(floors["diag_affine"]["mean"])
                                else None
                            ),
                            "spectrum": spec,
                            "perm_null": null,
                            "anova_shares": shares,
                            "read2_matched_n_grain_rank": read2,
                            "read3_stitch_bare_query": _registered_stitch_reads(
                                summaries_dir=summaries_dir,
                                cell=cell,
                                layer=layer,
                                unit_rows=unit_rows,
                                prefix_X=prefix_Xn,
                                context_X=context_Xn,
                                Y=Yb,
                                folds=folds,
                                args=args,
                            ),
                            "read4_operator_identity": read4,
                            "dynamics_D0_D5": dynamics_cache[(cell, layer)],
                            "behavior_B1_B2": _behavior_reads(
                                cell=cell,
                                unit_rows=unit_rows,
                                X=Xn,
                                Y=Yb,
                                fit_pred=fit_pred,
                                folds=folds,
                                judge_rows=judge_rows,
                                rb_directions=rb_directions,
                                trait_names=trait_names,
                                b0_pool=b0_pools.get(cell),
                                unit_source_indices=idx_arr,
                                layer=layer,
                            ),
                            "refit_twins": _refit_twins(unit_rows, Yb, seed=args.seed + layer),
                            "selection_symmetric_layer_max_null": (
                                _selection_symmetric_projection_null(
                                    unit_key=unit_key,
                                    factors=factors,
                                    rb_directions=rb_directions,
                                    trait_names=trait_names,
                                    layer=layer,
                                    n_draws=args.n_null_draws,
                                    seed=args.seed + layer,
                                    out_dir=out_dir,
                                )
                            ),
                            "mlp_companion": _mlp_companion_read(
                                cell=cell,
                                arm=arm,
                                fit_arm=fit_arm,
                                basis=basis,
                                X=Xn,
                                Y=Yb,
                                unit_rows=unit_rows,
                                args=args,
                            ),
                            "fingerprint": fp,
                        }
                        ckpt.parent.mkdir(parents=True, exist_ok=True)
                        ckpt.write_text(json.dumps(unit, indent=2, allow_nan=True))
                        units.append(unit)

    summary = {
        "phase": "P6_fit_grid",
        "units": units,
        "n_units": len(units),
        "registered_inputs": registered_inputs,
        "null_battery": {
            "implementation": (
                "streamed residual _perm_null using null_battery._k_chunks and observed lambdas"
            ),
            "n_null_draws": args.n_null_draws,
            "band_null_draws": args.band_null_draws,
            "frozen_layers_200_draws": sorted(FROZEN_NULL_LAYERS),
        },
        "input_fingerprint": _fingerprint(all_input_paths, {"script": "issue1092_fit_grid"}),
        "wall_s": time.monotonic() - t0,
    }
    path = out_dir / "fit_grid_summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=True))
    print(
        f"[fit-grid] artifact digest: units={len(units)} "
        f"first_r2={units[0]['fit']['r2'] if units else 'NA'} path={path}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summaries-dir", type=Path, required=True)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cells", default=None)
    p.add_argument("--layers", default="14,18,19")
    p.add_argument("--n-layers", type=int, default=28)
    p.add_argument("--hidden-dim", type=int, default=3584)
    p.add_argument("--arms", default="prefix_end,context_end")
    p.add_argument("--targets", default="t1")
    p.add_argument("--target-bases", default="ambient,pca48")
    p.add_argument("--fit-arms", default="A,B")
    p.add_argument("--group-key", default="prefix_id")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--n-null-draws", type=int, default=200)
    p.add_argument("--band-null-draws", type=int, default=20)
    p.add_argument("--matched-n-draws", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-scores", type=Path, default=None)
    p.add_argument("--rb-dir", type=Path, default=None)
    p.add_argument("--rb-rev", default=DEFAULT_RB_REV)
    p.add_argument("--require-behavior", action="store_true")
    p.add_argument("--require-bare", action="store_true")
    p.add_argument("--require-dynamics", action="store_true")
    p.add_argument("--require-mlp", action="store_true")
    p.add_argument(
        "--require-registered-reads",
        action="store_true",
        help="Fail if any registered read family is guarded/deferred",
    )
    p.add_argument("--run-mlp-companion", action="store_true")
    p.add_argument("--mlp-max-epochs", type=int, default=300)
    p.add_argument("--mlp-hidden", type=int, default=512)
    p.add_argument("--mlp-target-dim", type=int, default=48)
    p.add_argument("--mlp-chunk-size", type=int, default=4096)
    p.add_argument(
        "--allow-missing-registered-reads",
        action="store_true",
        help="Non-production escape: record missing registered inputs instead of failing.",
    )
    p.add_argument("--tiny-real", action="store_true")
    return p.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
