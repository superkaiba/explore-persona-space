"""#1092 fair-comparison DEEP-DIVE (analysis-only, no model forwards).

Goes deeper than the banked fair_comparison.json headline (prefix-end map vs
query-averaged context-vector map on answer-profile prediction):

  Q2 shrinkage: recompute the averaged-grain prediction matrices with the SAME
    fit engine the banked read used (`issue1092_fit_grid._fit_cv` ->
    `press_fit_predict`), VERIFY the recompute reproduces the banked agreement
    scalars, then fit (a) one global scalar alpha relating the two centered
    prediction matrices and (b) a per-dimension diagonal, reporting residual
    variance for each. A single scalar capturing most of the relationship is the
    "uniform shrinkage onto the shared component" signature.

  Q1 ceiling: verify the fraction-of-achievable-ceiling arithmetic for >=2
    entries by recomputation from the stored components in fair_comparison.json.

Reuses the fit machinery VERBATIM (same FOLD_SEED, same basis target build, same
PRESS ridge) so the recomputed prediction matrices match the banked read; the
only thing added is retaining prefix_pred_avg / ctx_pred_avg / Y_avg to run the
shrinkage decomposition the banked read did not persist.

Analysis-only: reads local staged .npy state summaries + the local manifest +
the banked JSON; runs NO model forward, NO training, NO API call.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

import sys  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue1092_fit_grid import (  # noqa: E402
    _basis_targets_with_info,
    _fit_cv,
    _folds_from_manifest,
    _r2,
)

STAGE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing"
)
SUMM = STAGE / "analysis_tensors/summaries"
MANIFEST = STAGE / "corpus/manifest.jsonl"
BANKED = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison_deepdive"
OUT.mkdir(parents=True, exist_ok=True)
SIDECAR = OUT / "deepdive.json"

CELLS = ["cell_inst_own", "cell_pre_own"]
BASES = ["ambient", "pca48"]
TARGETS = ["t1", "t2", "t3"]
HIDDEN_DIM = 3584
N_FOLDS = 6
MIN_ROWS_PER_PREFIX = 3


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load(cell: str, kind: str) -> np.ndarray:
    return np.load(SUMM / cell / f"{kind}_L14.npy", mmap_mode="r")


def _prefix_groups(prefix_ids: np.ndarray, min_rows: int) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for i, pid in enumerate(prefix_ids):
        groups.setdefault(str(pid), []).append(i)
    return {p: np.asarray(idx, dtype=np.int64) for p, idx in groups.items() if len(idx) >= min_rows}


def _shrinkage(P: np.ndarray, C: np.ndarray) -> dict:
    """Shrinkage decomposition of two aligned averaged-grain prediction matrices.

    P = prefix-end map's averaged-grain prediction (n_prefix, D)
    C = query-averaged context map's averaged-grain prediction (n_prefix, D)

    Tests whether P's BETWEEN-PREFIX structure is a uniformly-shrunk version of
    C's: center both across prefixes, then fit P_c ~ alpha * C_c (a) globally and
    (b) per-dimension. If the per-dim diagonal barely beats the single global
    scalar, a uniform scalar shrinkage captures the relationship.
    """
    P = np.asarray(P, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    # shared (grand-mean) component agreement, pre-centering
    muP, muC = P.mean(0), C.mean(0)
    shared_mean_reldiff = float(np.linalg.norm(muP - muC) / (np.linalg.norm(muC) + 1e-12))
    Pc = P - muP
    Cc = C - muC
    ssPc = float((Pc * Pc).sum())
    ssCc = float((Cc * Cc).sum())

    def _global(A: np.ndarray, B: np.ndarray) -> dict:
        # minimize ||A - a B||_F^2  ->  a = <A,B>/<B,B>
        ssB = float((B * B).sum())
        a = float((A * B).sum() / (ssB + 1e-12))
        resid = A - a * B
        ss_res = float((resid * resid).sum())
        ss_a = float((A * A).sum())
        return {
            "alpha": a,
            "resid_var_frac": ss_res / (ss_a + 1e-12),
            "r2_scalar_fit": 1.0 - ss_res / (ss_a + 1e-12),
        }

    # (a) global scalar, both directions
    g_PfromC = _global(Pc, Cc)  # P_c ~ alpha * C_c  (prefix as shrunk context)
    g_CfromP = _global(Cc, Pc)  # C_c ~ beta  * P_c

    # (b) per-dimension diagonal for P_c ~ alpha_d * C_c
    ssB_d = (Cc * Cc).sum(0)
    dot_d = (Pc * Cc).sum(0)
    valid = ssB_d > 0
    alpha_d = np.zeros(Pc.shape[1], dtype=np.float64)
    alpha_d[valid] = dot_d[valid] / ssB_d[valid]
    resid_d = Pc - alpha_d[None, :] * Cc
    ss_res_diag = float((resid_d * resid_d).sum())
    diag = {
        "resid_var_frac": ss_res_diag / (ssPc + 1e-12),
        "r2_diag_fit": 1.0 - ss_res_diag / (ssPc + 1e-12),
        "alpha_d_mean": float(alpha_d[valid].mean()),
        "alpha_d_median": float(np.median(alpha_d[valid])),
        "alpha_d_q25": float(np.quantile(alpha_d[valid], 0.25)),
        "alpha_d_q75": float(np.quantile(alpha_d[valid], 0.75)),
        "alpha_d_frac_lt_1": float((alpha_d[valid] < 1.0).mean()),
        "n_dims_valid": int(valid.sum()),
    }
    # how much does the per-dim diagonal improve on the single global scalar?
    diag_improvement = g_PfromC["resid_var_frac"] - diag["resid_var_frac"]
    return {
        "n_prefix": int(P.shape[0]),
        "D": int(P.shape[1]),
        "shared_mean_reldiff": shared_mean_reldiff,
        "between_prefix_ss_prefix": ssPc,
        "between_prefix_ss_context": ssCc,
        "between_prefix_ss_ratio_prefix_over_context": ssPc / (ssCc + 1e-12),
        "global_scalar_P_from_C": g_PfromC,
        "global_scalar_C_from_P": g_CfromP,
        "per_dim_diagonal_P_from_C": diag,
        "diag_minus_global_resid_var_frac_improvement": diag_improvement,
    }


def _rowcos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return num / den


def _prep_cell(cell: str, rows: list[dict]) -> dict:
    """Load + battery-exclude + fold-assign a cell's arrays once (reused across bases)."""
    prefix_all = _load(cell, "prefix_end")
    context_all = _load(cell, "context_end")
    t_all = [_load(cell, t) for t in TARGETS]
    n0 = min(prefix_all.shape[0], context_all.shape[0], min(t.shape[0] for t in t_all), len(rows))
    be_idx = np.asarray(
        [
            i
            for i in range(n0)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )
    prefix_ids = np.asarray([rows[int(i)].get("prefix_id", "") for i in be_idx])
    unit_rows = [rows[int(i)] for i in be_idx]
    folds = _folds_from_manifest(unit_rows, len(unit_rows), group_key="prefix_id", n_folds=N_FOLDS)
    X_prefix = np.asarray(prefix_all[be_idx], dtype=np.float64)
    X_context = np.asarray(context_all[be_idx], dtype=np.float64)
    Y_stacked = np.concatenate([np.asarray(t[be_idx], dtype=np.float64) for t in t_all], axis=1)
    groups = _prefix_groups(prefix_ids, MIN_ROWS_PER_PREFIX)
    return {
        "X_prefix": X_prefix,
        "X_context": X_context,
        "Y_stacked": Y_stacked,
        "folds": folds,
        "groups": groups,
        "pids": sorted(groups),
    }


def _fit_basis(cell: str, basis: str, prep: dict, results: dict) -> None:
    key = f"{cell}/{basis}"
    if key in results.get("cells", {}):
        print(f"[skip cached] {key}", flush=True)
        return
    t0 = time.monotonic()
    groups, pids, folds = prep["groups"], prep["pids"], prep["folds"]
    X_prefix, X_context = prep["X_prefix"], prep["X_context"]
    Yb = _basis_targets_with_info(
        prep["Y_stacked"], basis, hidden_dim=HIDDEN_DIM, targets=TARGETS, projection_target="t1"
    )[0]
    Yb = np.ascontiguousarray(Yb, dtype=np.float64)
    _, pred_context = _fit_cv(X_context, Yb, folds, return_pred=True)
    Y_avg = np.stack([Yb[groups[p]].mean(0) for p in pids], axis=0)
    Xp_avg = np.stack([X_prefix[groups[p]].mean(0) for p in pids], axis=0)
    ctx_pred_avg = np.stack([pred_context[groups[p]].mean(0) for p in pids], axis=0)
    pseudo_rows = [{"prefix_id": p} for p in pids]
    folds_avg = _folds_from_manifest(
        pseudo_rows, len(pseudo_rows), group_key="prefix_id", n_folds=N_FOLDS
    )
    _, prefix_pred_avg = _fit_cv(Xp_avg, Y_avg, folds_avg, return_pred=True)

    # verification vs banked agreement scalars
    e_prefix = np.linalg.norm(prefix_pred_avg - Y_avg, axis=1)
    e_ctx = np.linalg.norm(ctx_pred_avg - Y_avg, axis=1)
    cos_raw = _rowcos(prefix_pred_avg, ctx_pred_avg)
    pc = prefix_pred_avg - prefix_pred_avg.mean(0, keepdims=True)
    cc = ctx_pred_avg - ctx_pred_avg.mean(0, keepdims=True)
    cos_centered = _rowcos(pc, cc)
    recomputed = {
        "agreement_r2_prefixpred_vs_ctxpred": _r2(ctx_pred_avg, prefix_pred_avg),
        "agreement_r2_ctxpred_vs_prefixpred": _r2(prefix_pred_avg, ctx_pred_avg),
        "mean_cosine_raw": float(cos_raw.mean()),
        "mean_cosine_centered": float(cos_centered.mean()),
        "err_ratio_prefix_over_ctx": float(e_prefix.mean() / (e_ctx.mean() + 1e-12)),
        "per_prefix_err_correlation": float(np.corrcoef(e_prefix, e_ctx)[0, 1]),
        "r2_prefix_averaged": _r2(Y_avg, prefix_pred_avg),
        "r2_context_averaged": _r2(Y_avg, ctx_pred_avg),
    }
    shrink = _shrinkage(prefix_pred_avg, ctx_pred_avg)
    results.setdefault("cells", {})[key] = {
        "recomputed_agreement": recomputed,
        "shrinkage": shrink,
        "wall_s": time.monotonic() - t0,
    }
    SIDECAR.write_text(json.dumps(results, indent=2))
    print(
        f"[done {key}] r2(pref|ctx)={recomputed['agreement_r2_prefixpred_vs_ctxpred']:.4f} "
        f"cos_c={recomputed['mean_cosine_centered']:.4f} "
        f"alpha(P~aC)={shrink['global_scalar_P_from_C']['alpha']:.4f} "
        f"glob_resid={shrink['global_scalar_P_from_C']['resid_var_frac']:.4f} "
        f"diag_resid={shrink['per_dim_diagonal_P_from_C']['resid_var_frac']:.4f} "
        f"({results['cells'][key]['wall_s']:.0f}s)",
        flush=True,
    )
    del Yb, pred_context
    gc.collect()


def main() -> int:
    results: dict = {}
    if SIDECAR.exists():
        results = json.loads(SIDECAR.read_text())
    results["meta"] = {
        "script": "scripts/issue1092_fair_deepdive.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "engine": "REUSE issue1092_fit_grid._fit_cv -> press_fit_predict (FOLD_SEED=0, 6-fold)",
        "note": "recomputes averaged-grain prediction matrices to run the Q2 shrinkage test",
    }
    rows = _jsonl(MANIFEST)
    print(f"manifest rows={len(rows)}", flush=True)
    # basis-outer: cheap pca48 for both cells first, then heavy ambient
    for basis in ["pca48", "ambient"]:
        for cell in CELLS:
            prep = _prep_cell(cell, rows)
            _fit_basis(cell, basis, prep, results)
            del prep
            gc.collect()
    SIDECAR.write_text(json.dumps(results, indent=2))
    print(f"wrote {SIDECAR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
