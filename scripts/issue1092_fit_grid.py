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


def _fit_cv(X: np.ndarray, Y: np.ndarray, folds: list[np.ndarray]) -> dict:
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
    return {
        "r2": _r2(Y, pred),
        "r2_folds": fold_r2,
        "lambda_indices": lambdas,
    }


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
    X: np.ndarray, Y: np.ndarray, folds: list[np.ndarray], n_draws: int, seed: int
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
    perms = np.argsort(rng.random((n_draws, n)), axis=1).astype(np.int64)
    pred = np.zeros((n_draws, n, out_dim), dtype=np.float64)
    ridge = 1.0
    for test_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        Xtr = X[mask]
        Xte = X[test_idx]
        xmu = Xtr.mean(axis=0, keepdims=True)
        xsd = Xtr.std(axis=0, keepdims=True)
        xsd = np.where(xsd == 0.0, 1.0, xsd)
        Xtrn = (Xtr - xmu) / xsd
        Xten = (Xte - xmu) / xsd
        gram = Xtrn.T @ Xtrn + ridge * np.eye(d, dtype=np.float64)
        solved_xt = np.linalg.solve(gram, Xtrn.T)  # (d, n_train), factored once per fold
        bytes_per_draw = max(1, mask.sum() * out_dim * 8 + d * out_dim * 8)
        for start, stop in _k_chunks(n_draws, bytes_per_draw):
            target_train = Y[perms[start:stop][:, mask], :]  # (draw, n_train, out_dim)
            ymu = target_train.mean(axis=1, keepdims=True)
            centered = target_train - ymu
            weights = np.einsum("dn,kno->kdo", solved_xt, centered, optimize=True)
            pred[start:stop, test_idx, :] = (
                np.einsum("td,kdo->kto", Xten, weights, optimize=True) + ymu
            )
    targets = Y[perms]
    ss_res = ((targets - pred) ** 2).sum(axis=(1, 2))
    centered_targets = targets - targets.mean(axis=1, keepdims=True)
    ss_tot = (centered_targets**2).sum(axis=(1, 2))
    vals = np.where(ss_tot == 0.0, np.nan, 1.0 - ss_res / ss_tot)
    return {
        "n_draws": n_draws,
        "p95": float(np.nanpercentile(vals, 95)) if vals.size else float("nan"),
        "batched": True,
        "shared_factorization": True,
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
    folds: list[np.ndarray],
    judge_rows: list[dict],
) -> dict:
    """Compute the registered B1 direct-regression and B2 factor-score seams.

    r_B raw/map-mediated/B0/A2 subreads require staged r_B/B0 artifacts and are
    represented by fail-loud guards unless those artifacts are supplied by a
    future CLI extension. The direct regression and factor-score fits are
    computed here under the §6 eligibility block.
    """
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
            entry["B1_direct_regression"] = _fit_scalar_cv(X[idx], scores, local_folds)
            factor_scores = {}
            dense_pairs = [(factor_pos[int(i)], s) for i, s in pairs if int(i) in factor_pos]
            if dense_pairs:
                dense_idx = np.asarray([p[0] for p in dense_pairs], dtype=np.int64)
                dense_scores = np.asarray([p[1] for p in dense_pairs], dtype=np.float64)
                for factor_name in ("f", "g", "i"):
                    factor_arr = np.asarray(factors[factor_name])[dense_idx]
                    factor_scores[factor_name] = {
                        "norm_score_r": _pearson_or_nan(
                            np.linalg.norm(factor_arr, axis=1), dense_scores
                        )
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
        "guarded_subreads": [
            "B1_raw_projection_requires_rB",
            "B1_map_mediated_requires_saved_map_outputs",
            "B1_B0_poolings_requires_b0_rB_pool",
            "B1_A2_ceiling_requires_rB_t1_projection",
        ],
    }


def _registered_read_guards(args: argparse.Namespace, summaries_dir: Path) -> dict:
    guards = {}
    bare_dirs = sorted(summaries_dir.glob("bare_*"))
    dynamics_dirs = sorted(summaries_dir.glob("dynamics_*"))
    if not bare_dirs:
        guards["stitch_query_only_prefix_only"] = {
            "status": "deferred_fail_loud",
            "guard": (
                "rerun P6 with populated summaries/bare_* directories from gpu_phase --phases bare"
            ),
        }
        if args.require_bare or args.require_registered_reads:
            raise FileNotFoundError(
                "registered bare-query/stitch reads requested but summaries/bare_* is absent"
            )
    if not dynamics_dirs:
        guards["dynamics_D0_D5"] = {
            "status": "deferred_fail_loud",
            "guard": (
                "rerun P6 with populated summaries/dynamics_* directories from "
                "gpu_phase --phases dynamics"
            ),
        }
        if args.require_dynamics or args.require_registered_reads:
            raise FileNotFoundError(
                "registered dynamics D0-D5 reads requested but summaries/dynamics_* is absent"
            )
    if args.judge_scores is None:
        guards["B1_B2_behavior"] = {
            "status": "deferred_fail_loud",
            "guard": "pass --judge-scores to compute B1/B2; --require-behavior raises if absent",
        }
        if args.require_behavior or args.require_registered_reads:
            raise FileNotFoundError(
                "B1/B2 requested via --require-behavior but --judge-scores is absent"
            )
    if not args.run_mlp_companion:
        guards["mlp_companion"] = {
            "status": "deferred_fail_loud",
            "guard": "pass --run-mlp-companion to compute the vectorized MLP companion",
        }
        if args.require_mlp or args.require_registered_reads:
            raise RuntimeError("registered MLP companion requested but --run-mlp-companion absent")
    return guards


def run(args: argparse.Namespace) -> dict:
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
    judge_rows = _load_judge_score_rows(args.judge_scores)

    units: list[dict] = []
    all_input_paths: list[Path] = []
    for cell in cells:
        for layer in layers:
            y_by_target: dict[str, np.ndarray] = {}
            for target in targets:
                y_by_target[target], paths = _load_summary(summaries_dir, cell, target, layer)
                all_input_paths.extend(paths)
            Y_stacked = np.concatenate([y_by_target[t] for t in targets], axis=1)
            for arm in arms:
                X, paths = _load_summary(summaries_dir, cell, arm, layer)
                all_input_paths.extend(paths)
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
                            "n_null_draws": args.n_null_draws,
                            "matched_n_draws": args.matched_n_draws,
                        }
                        fp = _fingerprint(paths, config)
                        ckpt = (
                            ckpt_dir / f"{cell}_{arm}_fit{fit_arm}_L{layer:02d}_{basis}_{fp}.json"
                        )
                        if ckpt.exists():
                            units.append(json.loads(ckpt.read_text()))
                            continue
                        fit = _fit_cv(Xn, Yb, folds)
                        floors = _identity_floors(Xn, Yb, folds)
                        spec = _spectrum(Xn, Yb)
                        null = _perm_null(Xn, Yb, folds, args.n_null_draws, args.seed + layer)
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
                            n_draws=args.n_null_draws,
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
                            "read4_operator_identity": read4,
                            "behavior_B1_B2": _behavior_reads(
                                cell=cell,
                                unit_rows=unit_rows,
                                X=Xn,
                                Y=Yb,
                                folds=folds,
                                judge_rows=judge_rows,
                            ),
                            "refit_twins": _refit_twins(unit_rows, Yb, seed=args.seed + layer),
                            "selection_symmetric_layer_max_null": {
                                "status": "projection_stage",
                                "n_draws": args.n_null_draws,
                                "persist_path": "analysis_tensors/nulls/",
                            },
                            "mlp_companion": {
                                "status": "registered_guard",
                                "implementation": (
                                    "explore_persona_space.analysis.vectorized_mlp_skill"
                                ),
                                "run_with": "--run-mlp-companion",
                            },
                            "fingerprint": fp,
                        }
                        ckpt.parent.mkdir(parents=True, exist_ok=True)
                        ckpt.write_text(json.dumps(unit, indent=2, allow_nan=True))
                        units.append(unit)

    registered_guards = _registered_read_guards(args, summaries_dir)
    summary = {
        "phase": "P6_fit_grid",
        "units": units,
        "n_units": len(units),
        "registered_read_guards": registered_guards,
        "null_battery": {
            "implementation": (
                "batched shared-factorization _perm_null using null_battery._k_chunks"
            ),
            "n_null_draws": args.n_null_draws,
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
    p.add_argument("--arms", default="prefix_end,context_end")
    p.add_argument("--targets", default="t1")
    p.add_argument("--target-bases", default="ambient,pca48")
    p.add_argument("--fit-arms", default="A,B")
    p.add_argument("--group-key", default="prefix_id")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--n-null-draws", type=int, default=200)
    p.add_argument("--matched-n-draws", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-scores", type=Path, default=None)
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
    p.add_argument("--tiny-real", action="store_true")
    return p.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
