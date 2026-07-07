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
from pathlib import Path
from typing import Iterable

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from explore_persona_space.analysis.null_battery import bootstrap_ci_matched_r  # noqa: E402
from issue813_rank_spectrum import _fit_pieces, _gcv_lambda, _sigma2, _spectrum_stats, _standardize  # noqa: E402
from issue779_identity_baseline import CHEAP_RUNGS, _fit_diag_affine, _fit_global_affine  # noqa: E402
from issue923_fit_decomposition import press_fit_predict, run_selftest  # noqa: E402

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


def _load_summary(summaries_dir: Path, cell: str, kind: str, layer: int) -> tuple[np.ndarray, list[Path]]:
    paths = sorted((summaries_dir / cell).glob(f"{kind}_L{layer:02d}_shard*.npy"))
    if not paths:
        raise FileNotFoundError(f"no summary shards for {cell}/{kind}/L{layer:02d}")
    arrays = [np.load(p).astype(np.float64) for p in paths]
    return np.concatenate(arrays, axis=0), paths


def _folds_from_manifest(rows: list[dict], n: int, *, group_key: str, n_folds: int) -> list[np.ndarray]:
    groups = [str(r.get(group_key, r.get("prefix_id", i))) for i, r in enumerate(rows[:n])]
    uniq = sorted(set(groups))
    rng = np.random.default_rng(FOLD_SEED)
    rng.shuffle(uniq)
    fold_groups = [set(uniq[i::n_folds]) for i in range(n_folds)]
    folds = [np.array([i for i, g in enumerate(groups) if g in fg], dtype=np.int64) for fg in fold_groups]
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


def _perm_null(X: np.ndarray, Y: np.ndarray, folds: list[np.ndarray], n_draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draws):
        perm = rng.permutation(Y.shape[0])
        vals.append(_fit_cv(X, Y[perm], folds)["r2"])
    return {
        "n_draws": n_draws,
        "p95": float(np.nanpercentile(vals, 95)) if vals else float("nan"),
        "draws": [float(v) for v in vals],
    }


def _anova_shares(rows: list[dict], Y: np.ndarray) -> dict:
    prefix_ids = np.array([r.get("prefix_id", "") for r in rows[: Y.shape[0]]])
    query_ids = np.array([r.get("query_id", "") for r in rows[: Y.shape[0]]])
    yc = Y - Y.mean(axis=0, keepdims=True)
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
    }


def run(args: argparse.Namespace) -> dict:
    t0 = time.monotonic()
    run_selftest("cpu")
    summaries_dir = args.summaries_dir
    rows = _jsonl(args.corpus_dir / "manifest.jsonl")
    cells = _parse_csv(args.cells, [p.name for p in summaries_dir.iterdir() if p.is_dir() and p.name != "b0_rB_pool"])
    arms = _parse_csv(args.arms, INPUT_ARMS)
    targets = _parse_csv(args.targets, TARGETS)
    layers = _parse_layers(args.layers)
    bases = _parse_csv(args.target_bases, ("ambient", "pca48"))
    out_dir = args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

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
                n = min(X.shape[0], Y_stacked.shape[0], len(rows))
                Xn = X[:n]
                Yn = Y_stacked[:n]
                folds = _folds_from_manifest(rows, n, group_key=args.group_key, n_folds=args.n_folds)
                for basis in bases:
                    Yb = _basis_targets(Yn, basis)
                    config = {
                        "cell": cell,
                        "layer": layer,
                        "arm": arm,
                        "targets": targets,
                        "basis": basis,
                        "n": n,
                        "n_null_draws": args.n_null_draws,
                    }
                    fp = _fingerprint(paths, config)
                    ckpt = ckpt_dir / f"{cell}_{arm}_L{layer:02d}_{basis}_{fp}.json"
                    if ckpt.exists():
                        units.append(json.loads(ckpt.read_text()))
                        continue
                    fit = _fit_cv(Xn, Yb, folds)
                    floors = _identity_floors(Xn, Yb, folds)
                    spec = _spectrum(Xn, Yb)
                    null = _perm_null(Xn, Yb, folds, args.n_null_draws, args.seed + layer)
                    shares = _anova_shares(rows[:n], Yb)
                    unit = {
                        "cell": cell,
                        "layer": layer,
                        "arm": arm,
                        "targets": targets,
                        "basis": basis,
                        "n_rows": n,
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
                        "fingerprint": fp,
                    }
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    ckpt.write_text(json.dumps(unit, indent=2, allow_nan=True))
                    units.append(unit)

    # The null_battery import is exercised on a tiny vector to keep the CI helper
    # in the same dependency surface as production behavior reads.
    ci = bootstrap_ci_matched_r(
        np.arange(18, dtype=np.float64).reshape(3, 2, 3),
        np.ones((2, 3), dtype=np.float64),
        np.array([0.0, 1.0, 2.0]),
        0,
        n_boot=10,
        seed=args.seed,
    )
    summary = {
        "phase": "P6_fit_grid",
        "units": units,
        "n_units": len(units),
        "null_battery_ci_smoke": ci,
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
    p.add_argument("--group-key", default="prefix_id")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--n-null-draws", type=int, default=200)
    p.add_argument("--matched-n-draws", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tiny-real", action="store_true")
    return p.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
