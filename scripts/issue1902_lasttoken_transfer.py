#!/usr/bin/env python3
"""Issue #1902 follow-up: last-token IID recompute of Section 4.3 panels B/C.

Recomputes the OLMo-2 post-training 4x4 stage grid (activation checkpoint x
answer source) and the 12 ordered cross-stage transfer cells using LAST-TOKEN
context states (``u_last``) and the six size-matched IID random-row folds from
the ``lasttoken_comparison`` follow-up (RANDOM_FOLD_SEED=190231). The
registered Section 4.3 panels B/C used prompt-mean states and semantic-group
folds; this script replaces both so panels B and C match panel A exactly.

Ridge recipe: the primal spectral GCV ridge from
``issue1902_lasttoken_comparison`` (algebraically identical to #1902's dual
Gram GCV ridge; validated there by the committed-percell parity check and
here by the diagonal parity gate).

Transfer (2026-09-02 spec change, paper owner: "Only consider scaling and
bias for transfer"): the earlier general-linear context/answer alignments,
the orthogonal-Procrustes arm, the fixed-answer-text arm, and both matched
nulls (shuffled-correspondence, spectrum-matched) are REMOVED. For each
adjacent pair i->j, stage i's own map f_ii (fit on the train folds of i) is
applied to stage j's last-token states, p = f_ii(u_j), and scored against
w_jj three ways: as-is (direct), + bias, and scalar rescaling + bias.
Calibrations are fit on the train fold, scored out of fold.

Phases:
  stage     Download the 12 off-diagonal single-corpus L31 cells from HF
            (grid panel B needs them; transfer does not).
  grid      Fit all 16 (checkpoint, answer-source) cells, 6-fold OOF; write
            per-row SS npz per cell; run the diagonal parity gate.
  transfer  Adjacent pairs (B->S, S->D, D->R + reverses) x 6 folds:
            direct / bias / scalar scale+bias.
  analyze   Pool folds, bootstrap retention CIs (row + semantic-cluster,
            seed 1944), write summary.json.
  all       stage -> grid -> transfer -> analyze.

CPU only. Resumable: units whose npz already exists are skipped.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import issue1902_lasttoken_comparison as LC  # noqa: E402

STAGES = LC.STAGES  # ("B", "S", "D", "R")
LAYER = LC.LAYER  # 31
N_FOLDS = LC.N_FOLDS  # 6
GCV_LAMBDAS = LC.GCV_LAMBDAS  # logspace(-2, 4, 13)
CORPUS = "single"
BOOT_SEED = 1944  # default_rng(FOLD_SEED + 1902) finalize convention
N_BOOT = 1000
PARITY_TOL = 1e-6

DEFAULT_OUT = PROJECT_ROOT / "eval_results" / "issue_1902" / "lasttoken_transfer"
PANEL_A_PERCELL = (
    PROJECT_ROOT / "eval_results" / "issue_1902" / "lasttoken_comparison" / "percell"
)
PARITY_REF_JSON = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_1902"
    / "lasttoken_comparison"
    / "fit_u_last_random.json"
)

# Adjacent transitions first, then their reverses (cheap: same shared fits).
TRANSFER_PAIRS: list[tuple[str, str]] = [
    ("B", "S"), ("S", "D"), ("D", "R"),
    ("S", "B"), ("D", "S"), ("R", "D"),
]
TRANSFER_MODES = ("direct", "bias", "scale_bias")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg} (peak_rss={_rss_gb():.1f}G)", flush=True)


# ── data access ──────────────────────────────────────────────────────────────


def _answer_cell_path(stage_root: Path, m: str, s: str) -> Path:
    return LC._store_root(stage_root) / m / s / CORPUS / f"L{LAYER}.pt"


def load_ctx(stage_root: Path, stage: str) -> tuple[np.ndarray, list[str]]:
    import torch

    payload = torch.load(
        LC._ctx_path(stage_root, stage, CORPUS), map_location="cpu", weights_only=True
    )
    return (
        payload["u_last"].to(torch.float32).numpy(),
        [str(v) for v in payload["row_ids"]],
    )


def load_answer(stage_root: Path, m: str, s: str) -> tuple[np.ndarray, list[str]]:
    import torch

    payload = torch.load(
        _answer_cell_path(stage_root, m, s), map_location="cpu", weights_only=True
    )
    return (
        payload["w"].to(torch.float32).numpy(),
        [str(v) for v in payload["row_ids"]],
    )


def load_fold_of() -> tuple[np.ndarray, list[str]]:
    """Panel A's committed IID fold assignment (identical across stages)."""
    ref: np.ndarray | None = None
    ref_ids: list[str] = []
    for stage in STAGES:
        path = PANEL_A_PERCELL / f"u_last_random_{stage}_{CORPUS}_L{LAYER}.npz"
        with np.load(path) as payload:
            fold_of = np.asarray(payload["fold_of"])
            ids = [str(v) for v in payload["row_ids"]]
        if ref is None:
            ref, ref_ids = fold_of, ids
        elif not np.array_equal(fold_of, ref) or ids != ref_ids:
            raise RuntimeError(f"panel-A fold assignment differs for stage {stage}")
    assert ref is not None
    return ref, ref_ids


def load_clusters(stage_root: Path) -> tuple[list[str], np.ndarray]:
    path = LC._store_root(stage_root) / "B" / "B" / CORPUS / "row_index.jsonl"
    rows = [json.loads(line) for line in path.open()]
    return [str(r["id"]) for r in rows], np.asarray([int(r["cluster"]) for r in rows])


# ── solver ───────────────────────────────────────────────────────────────────


class SharedPrimalRidge:
    """Primal spectral GCV ridge sharing one eigendecomposition across targets.

    Copied from ``issue1902_lasttoken_comparison.primal_gcv_predict`` with the
    per-target work factored out so multiple targets and eval inputs reuse the
    training standardization and the X^T X eigendecomposition. Numerics per
    target are identical to the original function.
    """

    def __init__(self, x_train: np.ndarray, lambdas: np.ndarray = GCV_LAMBDAS):
        from scipy.linalg import eigh

        xtr = np.asarray(x_train, dtype=np.float64)
        n_train, dim = xtr.shape
        if n_train <= dim:
            raise ValueError(f"primal GCV requires n_train > d; got {n_train=} {dim=}")
        self.n_train, self.dim = n_train, dim
        self.lambdas = np.asarray(lambdas, dtype=np.float64)
        self.xmu = xtr.mean(axis=0)
        self.xsd = xtr.std(axis=0, ddof=0) + 1e-9
        self.xtr_std = (xtr - self.xmu) / self.xsd
        xtx = self.xtr_std.T @ self.xtr_std
        eigvals, eigvecs = eigh(xtx, overwrite_a=True, check_finite=False, driver="evd")
        self.eigvals = np.maximum(eigvals, 0.0)
        self.eigvecs = eigvecs
        self.positive = self.eigvals > np.finfo(np.float64).eps * max(
            n_train, dim
        ) * max(self.eigvals[-1], 1.0)

    def fit(self, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """GCV-selected ridge weights for one target block.

        Returns ``(weights, ymu, info)`` with ``weights`` mapping the
        STANDARDIZED input space to the centered target space.
        """
        ytr = np.asarray(y_train, dtype=np.float64)
        ymu = ytr.mean(axis=0)
        ytr = ytr - ymu
        xty = self.xtr_std.T @ ytr
        vt_xty = self.eigvecs.T @ xty
        uty_sq = np.zeros_like(self.eigvals)
        uty_sq[self.positive] = np.square(
            vt_xty[self.positive] / np.sqrt(self.eigvals[self.positive, None])
        ).sum(axis=1)
        total = float(np.square(ytr).sum())
        gcv = []
        dofs = []
        for lam in self.lambdas:
            filt = self.eigvals / (self.eigvals + float(lam))
            rss = total - float(((2.0 * filt - np.square(filt)) * uty_sq).sum())
            dof = float(filt.sum())
            denom = (self.n_train - dof) ** 2
            gcv.append(float("inf") if denom <= 1e-12 else rss / denom)
            dofs.append(dof)
        best_idx = int(np.argmin(gcv))
        best_lam = float(self.lambdas[best_idx])
        weights = self.eigvecs @ (vt_xty / (self.eigvals + best_lam)[:, None])
        return weights, ymu, {
            "selected_lambda": best_lam,
            "dof": dofs[best_idx],
            "gcv": gcv[best_idx],
        }

    def standardize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.xmu) / self.xsd

    def fit_predict(
        self, y_train: np.ndarray, x_eval: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        weights, ymu, info = self.fit(y_train)
        return self.standardize(x_eval) @ weights + ymu, info


# ── phase: stage ─────────────────────────────────────────────────────────────


def run_stage(stage_root: Path) -> None:
    from huggingface_hub import hf_hub_download

    for m in STAGES:
        for s in STAGES:
            if m == s:
                continue
            filename = f"{LC.HF_PREFIX}/{m}/{s}/{CORPUS}/L{LAYER}.pt"
            target = stage_root / filename
            if target.exists():
                _log(f"[stage] cached {m}/{s}")
                continue
            hf_hub_download(
                LC.HF_REPO,
                filename,
                repo_type="dataset",
                revision=LC.HF_REVISION,
                local_dir=stage_root,
            )
            _log(f"[stage] downloaded {m}/{s}")
    # Row-id verification across all 16 cells + 4 ctx shards.
    _, ref_ids = load_fold_of()
    for m in STAGES:
        _, ids = load_ctx(stage_root, m)
        if ids != ref_ids:
            raise RuntimeError(f"ctx row_ids differ for stage {m}")
        for s in STAGES:
            _, ids = load_answer(stage_root, m, s)
            if ids != ref_ids:
                raise RuntimeError(f"answer row_ids differ for cell ({m},{s})")
    _log(f"[stage] row_ids identical across 20 shards (n={len(ref_ids)})")


# ── phase: grid ──────────────────────────────────────────────────────────────


def _grid_cell_path(out_dir: Path, m: str, s: str) -> Path:
    return out_dir / "percell" / f"grid_{m}{s}_L{LAYER}.npz"


def _grid_fold_partial_path(out_dir: Path, m: str, fold: int) -> Path:
    return out_dir / "percell" / f"grid_{m}_f{fold}_partial.npz"


def run_grid(stage_root: Path, out_dir: Path, *, force: bool = False) -> None:
    fold_of, ref_ids = load_fold_of()
    n = len(fold_of)
    for m in STAGES:
        paths = {s: _grid_cell_path(out_dir, m, s) for s in STAGES}
        if not force and all(p.exists() for p in paths.values()):
            _log(f"[grid] checkpoint {m}: all 4 cells resumed")
            continue
        x, ids = load_ctx(stage_root, m)
        if ids != ref_ids:
            raise RuntimeError(f"ctx row_ids differ for stage {m}")
        ys: dict[str, np.ndarray] = {}
        for s in STAGES:
            ys[s], aids = load_answer(stage_root, m, s)
            if aids != ref_ids:
                raise RuntimeError(f"answer row_ids differ for cell ({m},{s})")
        acc = {
            s: {
                "res": np.full(n, np.nan),
                "tot": np.full(n, np.nan),
                "cos": np.full(n, np.nan),
                "lam": np.full(N_FOLDS, np.nan),
                "dof": np.full(N_FOLDS, np.nan),
                "n_tr": np.zeros(N_FOLDS, dtype=np.int64),
                "n_ev": np.zeros(N_FOLDS, dtype=np.int64),
            }
            for s in STAGES
        }
        for fold in range(N_FOLDS):
            t0 = time.time()
            ev = fold_of == fold
            tr = ~ev
            partial = _grid_fold_partial_path(out_dir, m, fold)
            if partial.exists() and not force:
                with np.load(partial) as payload:
                    rows = payload["row_idx"]
                    for k, s in enumerate(STAGES):
                        a = acc[s]
                        a["res"][rows] = payload["res"][k]
                        a["tot"][rows] = payload["tot"][k]
                        a["cos"][rows] = payload["cos"][k]
                        a["lam"][fold] = payload["lam"][k]
                        a["dof"][fold] = payload["dof"][k]
                        a["n_tr"][fold] = int(tr.sum())
                        a["n_ev"][fold] = int(ev.sum())
                _log(f"[grid] {m} fold {fold} resumed from partial")
                continue
            ridge = SharedPrimalRidge(x[tr])
            xev_std = ridge.standardize(x[ev])
            fold_res, fold_tot, fold_cos, fold_lam, fold_dof = [], [], [], [], []
            for s in STAGES:
                weights, ymu, info = ridge.fit(ys[s][tr])
                pred = xev_std @ weights + ymu
                rr, tt, cc = LC._per_row_components(pred, ys[s][ev], ys[s][tr].mean(axis=0))
                a = acc[s]
                a["res"][ev], a["tot"][ev], a["cos"][ev] = rr, tt, cc
                a["lam"][fold] = info["selected_lambda"]
                a["dof"][fold] = info["dof"]
                a["n_tr"][fold], a["n_ev"][fold] = int(tr.sum()), int(ev.sum())
                fold_res.append(rr)
                fold_tot.append(tt)
                fold_cos.append(cc)
                fold_lam.append(info["selected_lambda"])
                fold_dof.append(info["dof"])
                del weights, pred
            del ridge, xev_std
            LC._savez(
                partial,
                row_idx=np.flatnonzero(ev),
                res=np.stack(fold_res),
                tot=np.stack(fold_tot),
                cos=np.stack(fold_cos),
                lam=np.asarray(fold_lam),
                dof=np.asarray(fold_dof),
            )
            _log(f"[grid] {m} fold {fold} done in {time.time() - t0:.1f}s")
        for s in STAGES:
            a = acc[s]
            if not np.all(np.isfinite(a["res"])):
                raise RuntimeError(f"non-finite OOF components for cell ({m},{s})")
            LC._savez(
                paths[s],
                row_ids=np.asarray(ref_ids),
                fold_of=fold_of,
                ss_res=a["res"],
                ss_tot=a["tot"],
                cos=a["cos"],
                selected_lambda=a["lam"],
                dof=a["dof"],
                n_train=a["n_tr"],
                n_eval=a["n_ev"],
            )
            r2 = 1.0 - float(a["res"].sum()) / float(a["tot"].sum())
            _log(f"[grid] cell ({m},{s}) pooled R2={r2:.8f}")
        for fold in range(N_FOLDS):
            _grid_fold_partial_path(out_dir, m, fold).unlink(missing_ok=True)
        del x, ys, acc
    parity_gate(out_dir)


def parity_gate(out_dir: Path) -> dict[str, Any]:
    """Diagonal cells must reproduce the registered panel-A pooled R^2."""
    ref = json.loads(PARITY_REF_JSON.read_text())
    ref_r2 = {c["stage"]: float(c["r2"]) for c in ref["cells"] if c["corpus"] == CORPUS}
    rows = []
    for stage in STAGES:
        with np.load(_grid_cell_path(out_dir, stage, stage)) as payload:
            got = 1.0 - float(payload["ss_res"].sum()) / float(payload["ss_tot"].sum())
        rows.append(
            {
                "stage": stage,
                "recomputed_r2": got,
                "registered_r2": ref_r2[stage],
                "abs_diff": abs(got - ref_r2[stage]),
            }
        )
    report = {
        "tolerance": PARITY_TOL,
        "max_abs_diff": max(r["abs_diff"] for r in rows),
        "pass": all(r["abs_diff"] <= PARITY_TOL for r in rows),
        "reference": str(PARITY_REF_JSON.relative_to(PROJECT_ROOT)),
        "cells": rows,
    }
    LC._write_json(out_dir / "parity_gate.json", report)
    if not report["pass"]:
        raise RuntimeError(f"parity gate FAILED: {rows}")
    _log(f"[parity] PASS max_abs_diff={report['max_abs_diff']:.3e}")
    return report


# ── phase: transfer ──────────────────────────────────────────────────────────


def transfer_pair_fold(
    u_i: np.ndarray,
    u_j: np.ndarray,
    w_ii: np.ndarray,
    w_jj: np.ndarray,
    tr: np.ndarray,
    ev: np.ndarray,
    *,
    lambdas: np.ndarray = GCV_LAMBDAS,
    ridge_i: SharedPrimalRidge | None = None,
    f_ii: tuple[np.ndarray, np.ndarray, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """One (adjacent pair, fold) transfer unit — scaling and bias only.

    p = f_ii(u_j) with f_ii fit on stage i's train fold. Modes, all scored on
    the held-out fold against w_jj (denominator: train-fold mean of w_jj):
      direct      p as-is
      bias        p + b,        b = mean(w_jj[tr] - p[tr])
      scale_bias  alpha p + b,  scalar alpha + vector b, train least squares
    """
    n_tr, n_ev = int(tr.sum()), int(ev.sum())
    if ridge_i is None:
        ridge_i = SharedPrimalRidge(u_i[tr], lambdas)
    if f_ii is None:
        f_ii = ridge_i.fit(w_ii[tr])
    weights_f, ymu_f, f_info = f_ii

    p_tr = ridge_i.standardize(u_j[tr]) @ weights_f + ymu_f
    p_ev = ridge_i.standardize(u_j[ev]) @ weights_f + ymu_f
    y_tr = np.asarray(w_jj[tr], dtype=np.float64)

    p_bar = p_tr.mean(axis=0)
    y_bar = y_tr.mean(axis=0)
    pc = p_tr - p_bar
    yc = y_tr - y_bar
    # (b) bias only.
    b_bias = y_bar - p_bar
    # (c) global scalar scale + vector bias, least squares on the train fold.
    denom = float(np.square(pc).sum())
    alpha = float((pc * yc).sum() / denom) if denom > 0 else 0.0
    b_scale = y_bar - alpha * p_bar
    del pc, yc, p_tr, y_tr

    preds = {
        "direct": p_ev,
        "bias": p_ev + b_bias,
        "scale_bias": alpha * p_ev + b_scale,
    }
    y_ev = w_jj[ev]
    y_tr_mean = w_jj[tr].mean(axis=0)
    out: dict[str, Any] = {}
    tot = None
    for mode, pred in preds.items():
        rr, tt, _ = LC._per_row_components(pred, y_ev, y_tr_mean)
        out[f"res_{mode}"] = rr
        tot = tt
    out["tot"] = tot
    out["info"] = {
        "n_tr": n_tr,
        "n_ev": n_ev,
        "lambda_f": f_info["selected_lambda"],
        "dof_f": f_info["dof"],
        "alpha": alpha,
    }
    return out


def _xfer_path(out_dir: Path, i: str, j: str, fold: int) -> Path:
    return out_dir / "percell" / f"xfer_{i}{j}_f{fold}.npz"


def run_transfer(stage_root: Path, out_dir: Path, *, force: bool = False) -> None:
    gate_path = out_dir / "parity_gate.json"
    if not gate_path.exists():
        raise RuntimeError("parity gate missing — run the grid phase first")
    if not json.loads(gate_path.read_text())["pass"]:
        raise RuntimeError("parity gate failed — do not interpret transfer results")
    fold_of, ref_ids = load_fold_of()
    u: dict[str, np.ndarray] = {}
    ans: dict[str, np.ndarray] = {}
    for m in STAGES:
        u[m], ids = load_ctx(stage_root, m)
        if ids != ref_ids:
            raise RuntimeError(f"ctx row_ids differ for stage {m}")
        ans[m], aids = load_answer(stage_root, m, m)
        if aids != ref_ids:
            raise RuntimeError(f"answer row_ids differ for cell ({m},{m})")
    _log("[xfer] ctx + diagonal answer shards loaded")
    for fold in range(N_FOLDS):
        todo = {
            (i, j)
            for (i, j) in TRANSFER_PAIRS
            if force or not _xfer_path(out_dir, i, j, fold).exists()
        }
        if not todo:
            _log(f"[xfer] fold {fold}: all pairs resumed")
            continue
        ev = fold_of == fold
        tr = ~ev
        for i in STAGES:
            js = [j for (ii, j) in TRANSFER_PAIRS if ii == i and (i, j) in todo]
            if not js:
                continue
            t0 = time.time()
            ctx_i = SharedPrimalRidge(u[i][tr])
            f_i = ctx_i.fit(ans[i][tr])
            _log(
                f"[xfer] fold {fold} source {i}: shared fit in "
                f"{time.time() - t0:.0f}s"
            )
            for j in js:
                t1 = time.time()
                out = transfer_pair_fold(
                    u[i], u[j], ans[i], ans[j], tr, ev, ridge_i=ctx_i, f_ii=f_i
                )
                info = out["info"]
                LC._savez(
                    _xfer_path(out_dir, i, j, fold),
                    row_idx=np.flatnonzero(ev),
                    ss_res_direct=out["res_direct"],
                    ss_res_bias=out["res_bias"],
                    ss_res_scale_bias=out["res_scale_bias"],
                    ss_tot=out["tot"],
                    n_tr=np.int64(info["n_tr"]),
                    n_ev=np.int64(info["n_ev"]),
                    alpha=np.float64(info["alpha"]),
                    lambda_f=np.float64(info["lambda_f"]),
                    dof_f=np.float64(info["dof_f"]),
                )
                r2_sb = 1.0 - float(out["res_scale_bias"].sum()) / float(
                    out["tot"].sum()
                )
                _log(
                    f"[xfer] {i}->{j} fold {fold}: scale+bias R2={r2_sb:.4f} "
                    f"alpha={info['alpha']:.4f} in {time.time() - t1:.0f}s"
                )
            del ctx_i, f_i


# ── phase: analyze ───────────────────────────────────────────────────────────


def _pooled(res: np.ndarray, tot: np.ndarray) -> float:
    return 1.0 - float(res.sum()) / float(tot.sum())


def run_analyze(stage_root: Path, out_dir: Path) -> None:
    fold_of, ref_ids = load_fold_of()
    n = len(fold_of)
    meta_ids, clusters = load_clusters(stage_root)
    if meta_ids != ref_ids:
        raise RuntimeError("row_index.jsonl ids differ from tensor row_ids")
    uniq, inverse = np.unique(clusters, return_inverse=True)
    n_cl = len(uniq)

    # Shared bootstrap draws (paired across every quantity; seed 1944).
    rng_row = np.random.default_rng(BOOT_SEED)
    row_counts = np.zeros((N_BOOT, n), dtype=np.float64)
    idx = rng_row.integers(0, n, size=(N_BOOT, n))
    for b in range(N_BOOT):
        row_counts[b] = np.bincount(idx[b], minlength=n)
    del idx
    rng_cl = np.random.default_rng(BOOT_SEED)
    cidx = rng_cl.integers(0, n_cl, size=(N_BOOT, n_cl))
    cl_counts = np.zeros((N_BOOT, n_cl), dtype=np.float64)
    for b in range(N_BOOT):
        cl_counts[b] = np.bincount(cidx[b], minlength=n_cl)
    del cidx

    def cluster_sums(comp: np.ndarray) -> np.ndarray:
        per = np.zeros(n_cl, dtype=np.float64)
        np.add.at(per, inverse, np.asarray(comp, dtype=np.float64))
        return per

    def r2_draws(res: np.ndarray, tot: np.ndarray, mode: str) -> np.ndarray:
        if mode == "row":
            return 1.0 - (row_counts @ res) / (row_counts @ tot)
        return 1.0 - (cl_counts @ cluster_sums(res)) / (cl_counts @ cluster_sums(tot))

    def ci(values: np.ndarray) -> list[float]:
        return np.quantile(values, [0.025, 0.975]).tolist()

    grid: dict[str, Any] = {}
    diag: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for m in STAGES:
        for s in STAGES:
            with np.load(_grid_cell_path(out_dir, m, s)) as payload:
                res = np.asarray(payload["ss_res"], dtype=np.float64)
                tot = np.asarray(payload["ss_tot"], dtype=np.float64)
                lam = payload["selected_lambda"].tolist()
                dof = payload["dof"].tolist()
            grid[f"{m}{s}"] = {
                "activation_checkpoint": m,
                "answer_source": s,
                "r2": _pooled(res, tot),
                "row_ci": ci(r2_draws(res, tot, "row")),
                "cluster_ci": ci(r2_draws(res, tot, "cluster")),
                "selected_lambda": lam,
                "dof": dof,
                "n": n,
            }
            if m == s:
                diag[m] = (res, tot)
    _log("[analyze] grid pooled + bootstrapped")

    transfer: dict[str, Any] = {}
    for i, j in TRANSFER_PAIRS:
        res = {mode: np.full(n, np.nan) for mode in TRANSFER_MODES}
        tot = np.full(n, np.nan)
        alphas: list[float] = []
        lambdas_f: list[float] = []
        for fold in range(N_FOLDS):
            with np.load(_xfer_path(out_dir, i, j, fold)) as payload:
                rows = payload["row_idx"]
                for mode in TRANSFER_MODES:
                    res[mode][rows] = payload[f"ss_res_{mode}"]
                tot[rows] = payload["ss_tot"]
                alphas.append(float(payload["alpha"]))
                lambdas_f.append(float(payload["lambda_f"]))
        for mode in TRANSFER_MODES:
            if not np.all(np.isfinite(res[mode])):
                raise RuntimeError(f"incomplete {mode} components for {i}->{j}")
        if not np.all(np.isfinite(tot)):
            raise RuntimeError(f"incomplete tot components for {i}->{j}")
        res_jj, tot_jj = diag[j]
        r2_jj = _pooled(res_jj, tot_jj)
        r2 = {mode: _pooled(res[mode], tot) for mode in TRANSFER_MODES}

        retention: dict[str, Any] = {}
        for mode in TRANSFER_MODES:
            entry: dict[str, Any] = {"point": r2[mode] / r2_jj}
            for boot_mode in ("row", "cluster"):
                num = r2_draws(res[mode], tot, boot_mode)
                den = r2_draws(res_jj, tot_jj, boot_mode)
                rho = num / den
                finite = rho[np.isfinite(rho)]
                entry[f"{boot_mode}_ci"] = ci(finite)
                entry[f"{boot_mode}_n_finite"] = int(finite.size)
            retention[mode] = entry

        transfer[f"{i}->{j}"] = {
            "r2": r2,
            "r2_jj": r2_jj,
            "retention": retention,
            "alpha_by_fold": alphas,
            "lambda_f_by_fold": lambdas_f,
        }
        _log(
            f"[analyze] {i}->{j}: direct={r2['direct']:.4f} "
            f"scale_bias={r2['scale_bias']:.4f} r2_jj={r2_jj:.4f} "
            f"rho_sb={retention['scale_bias']['point']:.4f}"
        )

    summary = {
        "metadata": {
            "hf_repo": LC.HF_REPO,
            "hf_revision": LC.HF_REVISION,
            "layer": LAYER,
            "corpus": CORPUS,
            "context_summary": "u_last",
            "n_rows": n,
            "n_folds": N_FOLDS,
            "fold_mode": "random",
            "fold_seed": LC.RANDOM_FOLD_SEED,
            "fold_source": "eval_results/issue_1902/lasttoken_comparison/percell fold_of",
            "lambda_grid": GCV_LAMBDAS.tolist(),
            "transfer_modes": list(TRANSFER_MODES),
            "transfer_spec": (
                "scaling and bias only (2026-09-02): p = f_ii(u_j); modes "
                "as-is / bias / scalar rescaling + bias, calibrations fit on "
                "the train fold, scored out of fold; alignment, null, and "
                "per-dimension arms removed"
            ),
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "n_clusters": int(n_cl),
            "script": "scripts/issue1902_lasttoken_transfer.py",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "parity_gate": json.loads((out_dir / "parity_gate.json").read_text()),
        "grid": grid,
        "transfer": transfer,
    }
    LC._write_json(out_dir / "summary.json", summary)
    _log(f"[analyze] wrote {out_dir / 'summary.json'}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=["stage", "grid", "transfer", "analyze", "all"],
    )
    parser.add_argument("--stage-root", type=Path, default=LC.DEFAULT_STAGE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    _load_env()
    phases = (
        ["stage", "grid", "transfer", "analyze"] if args.phase == "all" else [args.phase]
    )
    for phase in phases:
        _log(f"=== phase {phase} ===")
        if phase == "stage":
            run_stage(args.stage_root)
        elif phase == "grid":
            run_grid(args.stage_root, args.out, force=args.force)
        elif phase == "transfer":
            run_transfer(args.stage_root, args.out, force=args.force)
        elif phase == "analyze":
            run_analyze(args.stage_root, args.out)
    _log("ALL PHASES DONE")


if __name__ == "__main__":
    main()
