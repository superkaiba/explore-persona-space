#!/usr/bin/env python3
"""Issue #825 — does the context->answer map degrade across conversation turns?

Defensible per-turn read of the linear context->answer map, replacing the
misleading #1092 `dynamics_d4_turn_profiles.png` (layer 14 non-peak + a bogus
`s_k` tautology arm + an inert "shuffled answers" answer-source axis).

The ONLY meaningful arm is ``context_k -> answer_k_t1`` at offset 0 (v_C at the
last context token -> v_A the answer-span mean). We:

1. Read the REAL per-turn held-out R2 from the banked #1092 grid
   (``eval_results/issue_1092/p7/dynamics_D0_D5.json``), cell ``cell_inst_claude``
   (instruct) / ``cell_pre_claude`` (pretrained), layers 14/18/19. All four
   answer-source cells are byte-identical for dynamics (arrays stored per
   model_type), so the answer-source axis is inert and the JSON's "shuf" cell is
   NOT a null.
2. RECOMPUTE the real fit locally from the HF activation arrays with the repo's
   own loaders + PRESS-ridge estimator, and VALIDATE it reproduces the JSON
   (L19/turn1/instruct ~= 0.212) before trusting anything downstream.
3. Build a REAL shuffled-answer permutation null: within each turn cell, permute
   the answer rows (break the context<->answer pairing), refit the SAME
   grouped-by-conv_id CV ridge with the SAME per-fold lambda the real fit
   selected, score held-out R2. N=200 draws -> mean + [2.5, 97.5] band. Batched
   (shared per-fold factorization) via the repo's production null helper.

No training, no new data, no GPU. CPU-only, thread-capped.

Usage::

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
        uv run python scripts/issue825_turn_depth_map.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy import below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

# Reuse the #1092 production loaders + estimator + fold builder + PRESS-ridge
# (single source of truth; no hand-rolled shard format or ridge). The null is a
# DUAL-space batched permutation null written here: the #1092 primal `_perm_null`
# forms the d x d (3584^2) gram and materializes a d x P (3584 x 3584) weight
# matrix per draw — ruinous when the target is the full 3584-dim answer vector
# (the task's own guidance: "for most turns n < 3584 so dual/Gram is cheaper").
# The dual estimator is IDENTICAL ridge (validated against the PRESS real fit and
# against the #1092 JSON below).
from issue658_fit_predictors import RIDGE_LAMBDAS  # noqa: E402
from issue1092_fit_grid import (  # noqa: E402
    _fit_cv,
    _folds_from_manifest,
    _load_summary,
    _read_index_files,
)

from explore_persona_space.analysis.null_battery import _k_chunks  # noqa: E402

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_SUMMARIES_BASE = "issue1092_realistic_crossing/analysis_tensors/summaries"
JSON_PATH = PROJECT_ROOT / "eval_results/issue_1092/p7/dynamics_D0_D5.json"
LAYERS = (14, 18, 19)
HEADLINE_LAYER = 19
# model_type -> the byte-identical answer-source cell used in the #1092 JSON.
MODEL_CELL = {"instruct": "cell_inst_claude", "pretrained": "cell_pre_claude"}
SRC_KIND = "context_k"
DST_KIND = "answer_k_t1"
N_FOLDS_CAP = 6  # #1092 used grouped 6-fold CV
N_DRAWS = 200
NULL_SEED = 1092
NULL_MIN_N = 30  # only build a null band where the turn cell has >= this many pairs
VALIDATION_TOL = 1e-2  # gate: recomputed L19/t1/instruct R2 must match JSON within this

OUT_JSON = PROJECT_ROOT / "eval_results/issue_825/turn_depth_map/results.json"
OUT_FIG_DIR = PROJECT_ROOT / "figures/issue_825"
FIG_STEM = "turn_depth_map"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _download_summaries(local_root: Path) -> Path:
    """Scoped download of exactly the shards we need; returns the summaries_dir."""
    from huggingface_hub import snapshot_download

    pats: list[str] = []
    for mt in ("dynamics_instruct", "dynamics_pretrained"):
        for kind in (SRC_KIND, DST_KIND):
            for layer in LAYERS:
                pats.append(f"{HF_SUMMARIES_BASE}/{mt}/{kind}_L{layer:02d}_shard*.npy")
        pats.append(f"{HF_SUMMARIES_BASE}/{mt}/row_index_{SRC_KIND}_shard*.jsonl")
        pats.append(f"{HF_SUMMARIES_BASE}/{mt}/row_index_{DST_KIND}_shard*.jsonl")
    snapshot_download(
        HF_DATA_REPO,
        repo_type="dataset",
        revision="main",
        allow_patterns=pats,
        local_dir=str(local_root),
    )
    return local_root / HF_SUMMARIES_BASE


def _json_real_curve() -> dict[str, dict[int, dict[str, dict]]]:
    """{model_type: {layer: {turn(str): {r2, r2_folds, lambda_indices}}}} from JSON."""
    with open(JSON_PATH) as f:
        grid = json.load(f)
    combos = grid["combos"]
    out: dict[str, dict[int, dict[str, dict]]] = {}
    for mt, cell in MODEL_CELL.items():
        out[mt] = {}
        for layer in LAYERS:
            combo = next(c for c in combos if c["cell"] == cell and c["layer"] == layer)
            tp = combo["dynamics"]["D4_turn_profiles"]["answer_side"][SRC_KIND][DST_KIND][
                "turn_profiles"
            ]
            layer_out: dict[str, dict] = {}
            for turn_s, node in tp.items():
                if node.get("status") == "computed":
                    layer_out[turn_s] = node["fit"]
            out[mt][layer] = layer_out
    return out


def _build_pairing(summaries_dir: Path, mt: str) -> list[tuple[int, int, str, int]]:
    """Replicate the #1092 pairs(context_k, answer_k_t1, offset=0) row list.

    Row indices are per (model_type, kind) and shared across layers, so the
    (context row, answer row) pairing is layer-independent — build it once.
    Returns [(ctx_row_i, ans_row_j, conv_id, turn_index), ...] in the same
    dict-iteration order the production ``pairs`` closure uses.
    """
    cell_dir = f"dynamics_{mt}"
    root = summaries_dir / cell_dir
    rows_c = _read_index_files(root, f"row_index_{SRC_KIND}")
    rows_a = _read_index_files(root, f"row_index_{DST_KIND}")
    index_c: dict[tuple[str, int], int] = {
        (str(r["conv_id"]), int(r["turn_index"])): i for i, r in enumerate(rows_c)
    }
    index_a: dict[tuple[str, int], int] = {
        (str(r["conv_id"]), int(r["turn_index"])): i for i, r in enumerate(rows_a)
    }
    paired: list[tuple[int, int, str, int]] = []
    for (conv, turn), ci in index_c.items():
        aj = index_a.get((conv, turn))  # offset 0
        if aj is None:
            continue
        paired.append((ci, aj, conv, turn))
    return paired


def _real_fit_and_folds(X: np.ndarray, Y: np.ndarray, rows: list[dict]):
    """Mirror issue1092_fit_grid._fit_pair_read exactly, also returning the folds.

    Used ONLY for the validation cell (one turn) — the full real curve is read
    from the #1092 JSON. `_fit_cv` runs the PRESS-ridge production estimator.
    """
    if X.shape[0] < 3 or Y.shape[0] < 3:
        return None
    n_folds = max(2, min(N_FOLDS_CAP, len({r["conv_id"] for r in rows})))
    folds = _folds_from_manifest(rows, len(rows), group_key="conv_id", n_folds=n_folds)
    if len(folds) < 2 or any(f.size >= len(rows) for f in folds):
        return None
    fit = _fit_cv(X, Y, folds)
    return fit, folds


def _folds_for_turn(rows: list[dict]):
    """Same conv_id-grouped folds `_fit_pair_read` builds — no fit (cheap)."""
    if len(rows) < 3:
        return None
    n_folds = max(2, min(N_FOLDS_CAP, len({r["conv_id"] for r in rows})))
    folds = _folds_from_manifest(rows, len(rows), group_key="conv_id", n_folds=n_folds)
    if len(folds) < 2 or any(f.size >= len(rows) for f in folds):
        return None
    return folds


def _dual_fold_op(X: np.ndarray, test_idx: np.ndarray, lam_idx: int):
    """Precompute the DUAL ridge operator A (n_te x n_tr) for one fold, once.

    Standardization matches press_fit_predict(standardize=True): train mean/std
    (ddof=0) + 1e-9 floor + degenerate-dim drop. A = K_te (K_tr + lam I)^-1, the
    exact primal-ridge equivalent in dual space (avoids the d x d gram + the d x P
    weight matrix). pred_te(Y) = A @ (Ytr - ymu) + ymu.
    """
    n = X.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    Xtr = X[mask]
    Xte = X[test_idx]
    mu = Xtr.mean(0)
    sd = Xtr.std(0, ddof=0) + 1e-9
    keep = sd > (sd.max() * 1e-6 + 1e-12)
    Xtr_n = ((Xtr - mu) / sd)[:, keep]
    Xte_n = ((Xte - mu) / sd)[:, keep]
    lam = float(RIDGE_LAMBDAS[int(lam_idx)])
    ktr = Xtr_n @ Xtr_n.T
    kte = Xte_n @ Xtr_n.T
    a = np.linalg.solve(ktr + lam * np.eye(ktr.shape[0]), kte.T).T  # (n_te, n_tr)
    return mask, a


def _dual_real_r2(
    X: np.ndarray, Y: np.ndarray, folds: list[np.ndarray], lambda_indices: list[int]
) -> float:
    """Held-out R2 of the DUAL ridge on the TRUE pairing (identity permutation).

    Fixed per-fold lambda (from the real fit). Proves the dual estimator == the
    PRESS real fit when the two are compared on the same folds + lambda.
    """
    ss_tot = float(((Y - Y.mean(0, keepdims=True)) ** 2).sum())
    if ss_tot == 0.0:
        return float("nan")
    ss_res = 0.0
    for fi, test_idx in enumerate(folds):
        mask, a = _dual_fold_op(X, test_idx, int(lambda_indices[fi]))
        ytr = Y[mask]
        ymu = ytr.mean(0, keepdims=True)
        pred = a @ (ytr - ymu) + ymu
        ss_res += float(((Y[test_idx] - pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _dual_perm_null(
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[np.ndarray],
    lambda_indices: list[int],
    n_draws: int,
    seed: int,
    device: str = "cpu",
) -> np.ndarray:
    """Batched dual-space shuffled-answer permutation null.

    Row-level permutation of Y (breaks the context<->answer pairing) within the
    turn cell; folds + per-fold lambda IDENTICAL to the real fit. Per fold, the
    dual operator A is factored ONCE; draws are batched as one einsum over the
    shared A. ss_tot is permutation-invariant (sum of squares about the mean).

    ``device`` (source-module threading, #825 turn-dynamics / artifact-reuse
    item (i)): ``device != "cpu"`` routes the identical math through torch on
    that device (fp32 draws, fp64 accumulation — same precision contract);
    the default numpy path is byte-identical to the original. Same rng, same
    perms, same fold operators either way (equivalence-smoked cpu-vs-torch).
    """
    if device != "cpu":
        return _dual_perm_null_torch(X, Y, folds, lambda_indices, n_draws, seed, device)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    p = Y.shape[1]
    # Null draws run in float32: the per-draw gather Y[perm] of the full
    # (k, n_tr, P=3584) target block is a ~GB memory-bandwidth-bound copy and is
    # the actual bottleneck (matmul is ~3x cheaper); f32 halves that bandwidth.
    # R2 values ~0 need nowhere near f64 precision; ss_res accumulates in f64. The
    # REAL curve (JSON) and the validation gate stay f64.
    yf = np.ascontiguousarray(Y, dtype=np.float32)
    ss_tot = float(((yf - yf.mean(0, keepdims=True)) ** 2).sum(dtype=np.float64))
    if ss_tot == 0.0 or n_draws <= 0:
        return np.full(n_draws, np.nan)
    perms = np.argsort(rng.random((n_draws, n)), axis=1).astype(np.int64)
    ss_res = np.zeros(n_draws, dtype=np.float64)
    for fi, test_idx in enumerate(folds):
        mask, a = _dual_fold_op(X, test_idx, int(lambda_indices[fi]))
        af = a.astype(np.float32)
        ntr = int(mask.sum())
        nte = int(test_idx.size)
        bytes_per = (ntr + nte) * p * 4
        for start, stop in _k_chunks(n_draws, max(1, bytes_per)):
            ptr = perms[start:stop][:, mask]  # (k, n_tr)
            pte = perms[start:stop][:, test_idx]  # (k, n_te)
            ytr = yf[ptr]  # (k, n_tr, P)
            ymu = ytr.mean(axis=1, keepdims=True)
            centered = ytr - ymu
            # batched BLAS gemm: af (n_te, n_tr) broadcast over the draw axis of
            # centered (k, n_tr, P) -> (k, n_te, P). np.matmul routes to threaded
            # BLAS; np.einsum on this pattern falls back to a ~10x-slower C loop.
            pred = np.matmul(af, centered) + ymu
            yte = yf[pte]
            ss_res[start:stop] += ((yte - pred) ** 2).sum(axis=(1, 2), dtype=np.float64)
    return 1.0 - ss_res / ss_tot


def _dual_perm_null_torch(
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[np.ndarray],
    lambda_indices: list[int],
    n_draws: int,
    seed: int,
    device: str,
) -> np.ndarray:
    """Torch-device twin of the numpy `_dual_perm_null` (same rng/perms/math).

    Fold operators A are solved on ``device`` in fp64 then cast fp32; the draw
    gather + batched matmul run on ``device`` in fp32 with fp64 ss_res
    accumulation — mirroring the numpy path's precision contract. Returns the
    (n_draws,) R2 array on CPU numpy.
    """
    import torch

    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    p = Y.shape[1]
    yf = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)).to(dev)
    ss_tot = float(((yf - yf.mean(0, keepdim=True)) ** 2).sum(dtype=torch.float64).item())
    if ss_tot == 0.0 or n_draws <= 0:
        return np.full(n_draws, np.nan)
    perms_np = np.argsort(rng.random((n_draws, n)), axis=1).astype(np.int64)
    perms = torch.from_numpy(perms_np).to(dev)
    ss_res = torch.zeros(n_draws, dtype=torch.float64, device=dev)
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64)).to(dev)
    for fi, test_idx in enumerate(folds):
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        mask_t = torch.from_numpy(mask).to(dev)
        te_t = torch.from_numpy(np.ascontiguousarray(test_idx)).to(dev)
        # fold operator (fp64 solve, matches _dual_fold_op's standardization)
        Xtr = Xt[mask_t]
        Xte = Xt[te_t]
        mu = Xtr.mean(0)
        sd = Xtr.std(0, correction=0) + 1e-9
        keep = sd > (sd.max() * 1e-6 + 1e-12)
        Xtr_n = ((Xtr - mu) / sd)[:, keep]
        Xte_n = ((Xte - mu) / sd)[:, keep]
        lam = float(RIDGE_LAMBDAS[int(lambda_indices[fi])])
        ktr = Xtr_n @ Xtr_n.T
        kte = Xte_n @ Xtr_n.T
        eye = torch.eye(ktr.shape[0], dtype=torch.float64, device=dev)
        af = torch.linalg.solve(ktr + lam * eye, kte.T).T.to(torch.float32)  # (n_te, n_tr)
        ntr = int(mask.sum())
        nte = int(test_idx.size)
        bytes_per = (ntr + nte) * p * 4
        for start, stop in _k_chunks(n_draws, max(1, bytes_per)):
            ptr = perms[start:stop][:, mask_t]  # (k, n_tr)
            pte = perms[start:stop][:, te_t]  # (k, n_te)
            ytr = yf[ptr]  # (k, n_tr, P)
            ymu = ytr.mean(dim=1, keepdim=True)
            pred = torch.matmul(af, ytr - ymu) + ymu  # (k, n_te, P)
            yte = yf[pte]
            ss_res[start:stop] += ((yte - pred) ** 2).sum(dim=(1, 2), dtype=torch.float64)
    return (1.0 - ss_res / ss_tot).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--local-root", default=str(PROJECT_ROOT / "data/issue_825/summaries_dl"))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    local_root = Path(args.local_root)
    if args.skip_download:
        summaries_dir = local_root / HF_SUMMARIES_BASE
    else:
        summaries_dir = _download_summaries(local_root)
    assert summaries_dir.is_dir(), f"summaries_dir missing: {summaries_dir}"

    json_curve = _json_real_curve()

    # ---- compute real + null per (model_type, layer, turn) ----
    results: dict[str, dict] = {}
    n_per_turn: dict[str, dict[int, int]] = {}
    validation: dict = {}

    for mt in ("instruct", "pretrained"):
        paired = _build_pairing(summaries_dir, mt)
        ci_idx = np.asarray([p[0] for p in paired], dtype=np.int64)
        aj_idx = np.asarray([p[1] for p in paired], dtype=np.int64)
        pair_rows = [{"conv_id": p[2], "turn_index": p[3]} for p in paired]
        turns = sorted({p[3] for p in paired})
        # n per turn is layer-independent (rows shared across layers)
        n_per_turn[mt] = {t: sum(1 for p in paired if p[3] == t) for t in turns}
        results[mt] = {}

        turn_sel = {
            t: np.asarray([i for i, p in enumerate(paired) if p[3] == t], dtype=np.int64)
            for t in turns
        }

        for layer in LAYERS:
            t_layer = time.time()
            arr_c, _ = _load_summary(summaries_dir, f"dynamics_{mt}", SRC_KIND, layer)
            arr_a, _ = _load_summary(summaries_dir, f"dynamics_{mt}", DST_KIND, layer)
            Xall = arr_c[ci_idx]
            Yall = arr_a[aj_idx]
            print(f"[compute] {mt} L{layer}: loaded {Xall.shape} pairs", flush=True)
            layer_res: dict[str, dict] = {}
            for turn in turns:
                sel = turn_sel[turn]
                n = int(sel.size)
                turn_s = str(turn)
                json_fit = json_curve[mt][layer].get(turn_s)
                # Real point estimate = the banked #1092 production R2 (grouped
                # 6-fold PRESS-ridge CV). We do NOT refit every turn — only the
                # validation cell is recomputed (below).
                entry: dict = {
                    "turn": turn,
                    "n": n,
                    "real_r2": (None if json_fit is None else float(json_fit["r2"])),
                    "real_r2_recomputed": None,  # validation cell only
                    "null_mean": None,
                    "null_lo": None,
                    "null_hi": None,
                    "null_n_draws": 0,
                    "lambda_indices": (
                        None if json_fit is None else json_fit.get("lambda_indices")
                    ),
                }

                # validation gate: recompute ONE cell via the production PRESS path
                if mt == "instruct" and layer == HEADLINE_LAYER and turn == 1:
                    rows = [pair_rows[i] for i in sel]
                    rf = _real_fit_and_folds(Xall[sel], Yall[sel], rows)
                    if rf is not None:
                        fit_v, folds_v = rf
                        entry["real_r2_recomputed"] = float(fit_v["r2"])
                        dual_id = _dual_real_r2(
                            Xall[sel], Yall[sel], folds_v, fit_v["lambda_indices"]
                        )
                        validation = {
                            "cell": "instruct/L19/turn1",
                            "json_r2": (None if json_fit is None else float(json_fit["r2"])),
                            "recomputed_r2_press": float(fit_v["r2"]),
                            "dual_estimator_r2_identity": float(dual_id),
                            "abs_diff_press_vs_json": (
                                None
                                if json_fit is None
                                else abs(float(fit_v["r2"]) - json_fit["r2"])
                            ),
                            "abs_diff_dual_vs_json": (
                                None if json_fit is None else abs(float(dual_id) - json_fit["r2"])
                            ),
                            "tol": VALIDATION_TOL,
                        }

                # null band (only where the cell is large enough + a real fit exists)
                if n >= NULL_MIN_N and json_fit is not None:
                    rows = [pair_rows[i] for i in sel]
                    folds = _folds_for_turn(rows)
                    lam = json_fit.get("lambda_indices")
                    if folds is not None and lam is not None and len(lam) == len(folds):
                        draws = _dual_perm_null(
                            Xall[sel], Yall[sel], folds, lam, args.n_draws, NULL_SEED
                        )
                        finite = draws[np.isfinite(draws)]
                        if finite.size:
                            entry["null_mean"] = float(np.mean(finite))
                            entry["null_lo"] = float(np.percentile(finite, 2.5))
                            entry["null_hi"] = float(np.percentile(finite, 97.5))
                            entry["null_n_draws"] = int(finite.size)
                layer_res[turn_s] = entry
            results[mt][str(layer)] = layer_res
            print(f"[compute] {mt} L{layer}: done in {time.time() - t_layer:.1f}s", flush=True)

    # ---- validation gate ----
    assert validation, "validation cell (instruct/L19/turn1) not computed"
    dp = validation["abs_diff_press_vs_json"]
    dd = validation["abs_diff_dual_vs_json"]
    # PRIMARY gate: production PRESS recompute reproduces the JSON (array load +
    # estimator faithful). SECONDARY: the dual null estimator (identity pairing,
    # fixed lambda) also reproduces the JSON -> the null uses the same estimator.
    gate_pass = dp is not None and dp <= VALIDATION_TOL and dd is not None and dd <= VALIDATION_TOL
    validation["pass"] = bool(gate_pass)
    print(
        f"[validation] instruct/L19/turn1: JSON={validation['json_r2']:.6f} "
        f"PRESS={validation['recomputed_r2_press']:.6f} (|d|={dp:.2e}) "
        f"DUAL={validation['dual_estimator_r2_identity']:.6f} (|d|={dd:.2e}) "
        f"tol={VALIDATION_TOL} -> {'PASS' if gate_pass else 'FAIL'}"
    )
    if not gate_pass:
        raise SystemExit(
            "VALIDATION GATE FAILED — recomputed L19/turn1/instruct R2 does not match the "
            "#1092 JSON; refusing to trust the null. Do not use these outputs."
        )

    # ---- write results JSON ----
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": 825,
        "description": (
            "Per-turn held-out R2 of the linear context->answer map "
            "(context_k -> answer_k_t1, offset 0: v_C at last context token -> "
            "v_A answer-span mean) vs shuffled-answer permutation null, "
            "instruct & pretrained, layers 14/18/19."
        ),
        "arm": "context_k -> answer_k_t1 (offset 0)",
        "arm_notes": (
            "s_k (the #1092 'prefix' arm) is a tautology at offset 0 "
            "(s_pos = answer_end-1, the answer's own last token) and is DROPPED. "
            "The answer-source axis (cell_inst_{claude,own,pretext,shuf}) is inert "
            "for dynamics — arrays are stored per model_type, so all 4 cells are "
            "byte-identical and the JSON 'shuf' cell is NOT a null; the null here "
            "is generated by row-level permutation of the answer rows."
        ),
        "estimator": (
            "PRESS-ridge (issue923 press_fit_predict, standardize=True, "
            "lambda grid [1e-2,1e-1,1,10,100,1000]), grouped-by-conv_id "
            f"{N_FOLDS_CAP}-fold CV; held-out R2 aggregated over folds "
            "(issue1092_fit_grid._fit_cv)."
        ),
        "null_recipe": (
            "Within each turn cell, permute the answer rows at the row level "
            "(break the context<->answer pairing), keeping the conv_id-grouped "
            "fold partition IDENTICAL to the real fit and refitting ridge per fold "
            "with the SAME per-fold lambda the real fit selected (fixed). Held-out "
            f"R2 per draw over {N_DRAWS} draws; report mean + [2.5, 97.5] "
            "percentile band. Batched shared-per-fold-factorization null "
            "(issue1092_fit_grid._perm_null). Micro-difference vs the real "
            "estimator: the null path uses per-fold train mean/std standardization "
            "without press's 1e-9 sd floor / degenerate-dim drop (negligible for "
            "these dense activation designs). Null draws are computed in float32 "
            "(the per-draw target gather is memory-bandwidth-bound); R2~0 needs no "
            "f64 precision. The real curve (JSON) and the validation gate are f64."
        ),
        "null_seed": NULL_SEED,
        "null_min_n": NULL_MIN_N,
        "layers": list(LAYERS),
        "headline_layer": HEADLINE_LAYER,
        "model_cells": MODEL_CELL,
        "n_per_turn": {
            mt: {str(t): int(n) for t, n in n_per_turn[mt].items()} for mt in n_per_turn
        },
        "validation": validation,
        "source_json": str(JSON_PATH.relative_to(PROJECT_ROOT)),
        "hf_data_repo": HF_DATA_REPO,
        "hf_summaries_prefix": HF_SUMMARIES_BASE,
        "hf_paths_read": [
            f"{HF_SUMMARIES_BASE}/dynamics_{{instruct,pretrained}}/"
            f"{{context_k,answer_k_t1}}_L{{14,18,19}}_shard*.npy",
            f"{HF_SUMMARIES_BASE}/dynamics_{{instruct,pretrained}}/"
            f"row_index_{{context_k,answer_k_t1}}_shard*.jsonl",
        ],
        "git_commit": _git_commit(),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "results": results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[write] {OUT_JSON}")

    _plot(payload, n_per_turn)


def _plot(payload: dict, n_per_turn: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = payload["results"]

    def curve(mt: str, layer: int):
        node = results[mt][str(layer)]
        turns = sorted((int(t) for t in node), key=int)
        xs, real, lo, hi, nn = [], [], [], [], []
        for t in turns:
            e = node[str(t)]
            r = e["real_r2"]
            if r is None:
                continue
            xs.append(t)
            real.append(r)
            lo.append(e["null_lo"])
            hi.append(e["null_hi"])
            nn.append(e["n"])
        return (
            np.array(xs),
            np.array(real, dtype=float),
            np.array([np.nan if v is None else v for v in lo], dtype=float),
            np.array([np.nan if v is None else v for v in hi], dtype=float),
            np.array(nn),
        )

    pal = paper_palette(4)
    c_inst, c_pre, c_l14, c_l18 = pal[0], pal[1], pal[2], pal[3]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # ---- Panel A: headline layer, both models, with shuffled-null band ----
    xi, ri, loi, hii, ni = curve("instruct", HEADLINE_LAYER)
    xp, rp, lop, hip, npr = curve("pretrained", HEADLINE_LAYER)
    # restrict headline x to turns with n >= NULL_MIN_N (band defined there)
    mi = ni >= NULL_MIN_N
    mp = npr >= NULL_MIN_N
    maxturn = int(max(xi[mi].max() if mi.any() else 0, xp[mp].max() if mp.any() else 0))

    # shuffled-null band (per model, where n >= NULL_MIN_N)
    if mi.any():
        axA.fill_between(
            xi[mi],
            loi[mi],
            hii[mi],
            color="0.7",
            alpha=0.35,
            linewidth=0,
            label="shuffled-answer null (2.5-97.5%), instruct",
        )
    if mp.any():
        axA.fill_between(
            xp[mp],
            lop[mp],
            hip[mp],
            color="0.5",
            alpha=0.20,
            linewidth=0,
            label="shuffled-answer null (2.5-97.5%), pretrained",
        )
    axA.plot(xi[mi], ri[mi], "-o", color=c_inst, ms=4, lw=1.8, label="instruct (L19)")
    axA.plot(xp[mp], rp[mp], "--s", color=c_pre, ms=4, lw=1.8, label="pretrained (L19)")
    axA.axhline(0.0, color="0.6", lw=0.8, ls=":")
    axA.set_xlabel("user-turn index")
    axA.set_ylabel(r"held-out $R^2$, context$\to$answer, layer 19")
    axA.set_title(f"Layer 19 map strength vs turn (n$\\geq${NULL_MIN_N} pairs)")
    axA.set_xlim(0, maxturn + 1)
    axA.legend(fontsize=7, loc="upper right", framealpha=0.9)

    # n-per-turn annotation on a faint twin axis
    axAn = axA.twinx()
    axAn.plot(xi[mi], ni[mi], color=c_inst, lw=0.7, alpha=0.35)
    axAn.plot(xp[mp], npr[mp], color=c_pre, lw=0.7, alpha=0.35, ls="--")
    axAn.set_ylabel("n pairs (faint)", color="0.5", fontsize=8)
    axAn.tick_params(axis="y", labelsize=7, colors="0.5")

    # ---- Panel B: instruct layer curves 14/18/19 (layer story) ----
    lcols = {14: c_l14, 18: c_l18, 19: c_inst}
    lstyle = {14: "-.", 18: "--", 19: "-"}
    for layer in LAYERS:
        x, r, _lo, _hi, nn = curve("instruct", layer)
        m = nn >= NULL_MIN_N
        axB.plot(
            x[m],
            r[m],
            lstyle[layer] + "o",
            color=lcols[layer],
            ms=3,
            lw=1.6,
            label=f"instruct L{layer}",
        )
    axB.axhline(0.0, color="0.6", lw=0.8, ls=":")
    axB.set_xlabel("user-turn index")
    axB.set_ylabel(r"held-out $R^2$, context$\to$answer")
    axB.set_title("Layer 14 (non-peak) vs 18/19 — instruct")
    axB.set_xlim(0, maxturn + 1)
    axB.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    png = OUT_FIG_DIR / f"{FIG_STEM}.png"
    pdf = OUT_FIG_DIR / f"{FIG_STEM}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "figure": f"{FIG_STEM}.png",
        "git_commit": payload["git_commit"],
        "source_results_json": str(OUT_JSON.relative_to(PROJECT_ROOT)),
        "caption": (
            "Held-out R2 of the linear context->answer map (context_k -> "
            "answer_k_t1, offset 0) per user-turn index. Left: layer 19 for "
            "instruct (solid) and pretrained (dashed) with the shuffled-answer "
            "permutation null band (grey, 2.5-97.5% over 200 draws); faint lines "
            "give n pairs per turn. Right: instruct layers 14/18/19, showing "
            "layer 14 is far below the layer 18/19 peak. Turns shown have "
            f">= {NULL_MIN_N} paired examples."
        ),
    }
    with open(OUT_FIG_DIR / f"{FIG_STEM}.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[write] {png}")
    print(f"[write] {pdf}")
    print(f"[write] {OUT_FIG_DIR / (FIG_STEM + '.meta.json')}")


if __name__ == "__main__":
    main()
