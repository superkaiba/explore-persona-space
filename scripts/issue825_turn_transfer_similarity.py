#!/usr/bin/env python3
"""Issue #825 — cross-turn TRANSFER matrix + operator SIMILARITY, n=171 preview.

The matched-N read (``scripts/issue825_turn_depth_matched_n.py``) shows the
per-turn context->answer map strength (held-out R2) vs conversation turn depth,
controlled for sample size. This script asks two follow-on questions about the
per-turn linear maps themselves, at layer 19, both models, on the banked #1092
per-turn tensors (``context_k -> answer_k_t1``, offset 0):

1. **Cross-turn TRANSFER T[i,j]** — is the ridge map FIT on turn i still
   predictive of turn j's answers? Build ONE shared conversation-grouped 6-fold
   partition over the union of subsampled conv_ids, applied identically at every
   turn. Per fold: fit ridge on turn-i TRAIN rows (conv in the fold's train
   groups), predict turn-j TEST rows (conv in the fold's test group), pool the
   held-out predictions across folds, report the variance-weighted held-out R2
   the reused #1092 core computes (``_r2`` over pooled preds). The shared
   conv-grouped partition means a conversation appearing at BOTH turn i and turn
   j is never simultaneously in turn-i-train and turn-j-test (no leakage). For
   i==j this reduces to standard grouped CV; T[i,i] is the matrix's OWN
   diagonal reference, and we also report the transfer fraction T[i,j]/T[j,j].

2. **Operator SIMILARITY C[i,j]** — cosine between the per-turn ridge OPERATORS.
   The operator M_i is the standardized-space ridge weight matrix W_i (d x P,
   d=P=3584) that the fit on turn i's full n=171 subset produces (lambda by
   PRESS-LOO on the full subset). C[i,j] = cosine(vec(M_i), vec(M_j)). Two
   references: (a) a within-turn SPLIT-HALF cosine ceiling (fit each half of the
   turn's conv set, cosine the two half-operators — n~85 each, so the ceiling is
   CONSERVATIVE vs the n=171 fits), and (b) a SHUFFLED-pairing cosine null (fit
   with the answer rows permuted; report cosine(real, shuffled) per turn and
   cosine(shuffled_i, shuffled_j) for adjacent turns as the null scale).

Interpretive caveat (recorded in the results JSON): each turn's operator lives
in ITS OWN standardized coordinates (per-turn train mean/std, faithful to the
reused PRESS estimator), so cross-turn C[i,j] includes any per-turn
standardization drift on top of genuine operator differences. The split-half
ceiling is computed with the SAME per-turn standardization, so it is the right
reference for the estimation-noise floor.

Anchor gate (FAIL = abort, do not commit): with the per-turn fold recipe
(``TDM._folds_for_turn``, FOLD_SEED=0) on the FULL t11 cell (n=171, no
subsampling), the diagonal R2 for each model must reproduce the banked
``eval_results/issue_825/turn_depth_matched_n/results.json``
``matched.level_171.<model>.19.11.real_r2_mean`` to <1e-6 (that banked value has
sd=0 across draws because the t11 cell is full). This independently re-exercises
the reused #1092 PRESS-ridge core.

No training, no new data, no GPU. CPU-only, thread-capped.

Usage::

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
        MALLOC_ARENA_MAX=2 uv run python scripts/issue825_turn_transfer_similarity.py \
        --skip-download
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the raw #825 turn-depth read's data loaders + fit core VERBATIM (single
# source of truth; no re-implementation of the ridge or the pairing). The
# operator materialization uses the SAME PRESS estimator + dual-weight helper the
# #1092 fit grid and #923 selftest are built on.
import issue825_turn_depth_map as TDM  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue658_fit_predictors import RIDGE_LAMBDAS, _ridge_dual_weights  # noqa: E402
from issue923_fit_decomposition import press_fit_predict  # noqa: E402
from issue1092_fit_grid import _r2  # noqa: E402  (pooled held-out R2, reused verbatim)

LAYER = TDM.HEADLINE_LAYER  # 19
TURNS = [1, 3, 5, 7, 9, 11]  # turns with n_avail >= 171 (from turn_depth_map n_per_turn)
ADJ_PAIRS = list(itertools.pairwise(TURNS))  # (1,3),(3,5),...,(9,11)
N_MATCH = 171
MODELS = ("instruct", "pretrained")
SRC_KIND = TDM.SRC_KIND  # context_k
DST_KIND = TDM.DST_KIND  # answer_k_t1
HF_DATA_REPO = TDM.HF_DATA_REPO
HF_SUMMARIES_BASE = TDM.HF_SUMMARIES_BASE
MODEL_CELL = TDM.MODEL_CELL
N_FOLDS = TDM.N_FOLDS_CAP  # 6

# Seeds. The subsample seed recipe mirrors issue825_turn_depth_matched_n's
# level_171 / draw-0 recipe so the n=171 subsets are reproducible and
# cross-checkable against the banked matched-N draw-0 subsamples (level index 0,
# draw index 0). For t11 (n_avail == 171) the subsample SET is the full cell
# regardless of seed, so the anchor gate (which uses the full t11 cell) stays
# consistent with the T-matrix's t11 subset.
SUB_SEED_BASE = 8250  # == issue825_turn_depth_matched_n.SUB_SEED_BASE
SHARED_FOLD_SEED = 82519  # shared conv-grouped 6-fold partition over the union
SPLITHALF_SEED_BASE = 82531  # per-turn split-half of the conv set
SHUFFLE_SEED_BASE = 82547  # per-turn answer-row permutation (shuffled null)

ANCHOR_TOL = 1e-6

RAW_MATCHED_JSON = PROJECT_ROOT / "eval_results/issue_825/turn_depth_matched_n/results.json"
RAW_MAP_JSON = PROJECT_ROOT / "eval_results/issue_825/turn_depth_map/results.json"
OUT_JSON = PROJECT_ROOT / "eval_results/issue_825/turn_transfer_similarity/results.json"
FIG_DIR = PROJECT_ROOT / "figures/issue_825"
FIG_T = "turn_transfer_r2"
FIG_C = "turn_operator_cosine"


def _sub_seed(model_idx: int, turn: int) -> int:
    # issue825_turn_depth_matched_n._sub_seed(model_idx, level_idx=0, turn, draw=0)
    return SUB_SEED_BASE + model_idx * 10_000_000 + turn * 1000


def _subsample_turn(sel: np.ndarray, model_idx: int, turn: int) -> np.ndarray:
    """171-row subsample of a turn cell (one draw, fixed seed). Full set if n==171."""
    if sel.size <= N_MATCH:
        return sel
    return np.random.default_rng(_sub_seed(model_idx, turn)).choice(
        sel, size=N_MATCH, replace=False
    )


def _shared_conv_folds(union_convs: list[str]) -> list[set[str]]:
    """ONE conv-grouped 6-fold partition over the union of subsampled conv_ids.

    Mirrors issue1092_fit_grid._folds_from_manifest's shuffle-then-stride split,
    but over the shared UNION (deterministic, seeded), returning the per-fold
    TEST conv groups. Each conv is in exactly one test group, so a turn row is
    predicted exactly once (in the fold where its conv is the test group).
    """
    uniq = sorted(set(union_convs))
    rng = np.random.default_rng(SHARED_FOLD_SEED)
    rng.shuffle(uniq)
    return [set(uniq[i::N_FOLDS]) for i in range(N_FOLDS)]


def _transfer_r2(
    X_i: np.ndarray,
    Y_i: np.ndarray,
    convs_i: list[str],
    X_j: np.ndarray,
    Y_j: np.ndarray,
    convs_j: list[str],
    test_groups: list[set[str]],
) -> float:
    """Held-out R2 of turn-i's ridge map applied to turn-j answers, shared folds.

    Per fold: fit PRESS ridge on turn-i rows whose conv is NOT in the fold's test
    group; predict turn-j rows whose conv IS in the test group (lambda selected
    by PRESS-LOO on the turn-i train set, matching the reused core). Pool held-out
    turn-j predictions; report _r2(Y_j, pred) (variance-weighted, turn-j's own
    mean as the reference), the pooled held-out R2 convention of _fit_cv.
    """
    convs_i_arr = np.asarray(convs_i)
    convs_j_arr = np.asarray(convs_j)
    pred = np.zeros_like(Y_j, dtype=np.float64)
    filled = np.zeros(Y_j.shape[0], dtype=bool)
    for tg in test_groups:
        tr_i = np.asarray([k for k, c in enumerate(convs_i_arr) if c not in tg], dtype=np.int64)
        te_j = np.asarray([k for k, c in enumerate(convs_j_arr) if c in tg], dtype=np.int64)
        if tr_i.size == 0 or te_j.size == 0:
            continue
        res = press_fit_predict(
            torch.from_numpy(X_i[tr_i]).double(),
            torch.from_numpy(Y_i[tr_i]).double(),
            torch.from_numpy(X_j[te_j]).double(),
            standardize=True,
        )
        pred[te_j] = res["pred"].detach().cpu().numpy()
        filled[te_j] = True
    # every turn-j row's conv is in exactly one shared test group -> predicted once
    assert filled.all(), f"transfer: {int((~filled).sum())} turn-j rows unpredicted"
    return _r2(Y_j, pred)


def _std_operator(X_full: np.ndarray, Y_full: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Full-cell PRESS ridge operator M in the AMBIENT (d, P) standardized space.

    lambda by PRESS-LOO on the full cell (press_fit_predict standardize=True).
    W = _ridge_dual_weights(Xtr_n, Ytr_c, lam) is the standardized-space weight
    matrix (d_keep, P) consistent with press_fit_predict's own prediction path
    (pred = Xte_n @ W + ymu); scatter into the full ambient (d, P) with zeros at
    dropped-degenerate dims so all turns share the (3584, 3584) cosine shape.
    Returns (M fp32, lam_idx, keep_count).
    """
    Xt = torch.from_numpy(X_full).double()
    Yt = torch.from_numpy(Y_full).double()
    res = press_fit_predict(Xt, Yt, Xt[:1], standardize=True)  # Xte dummy; we take std+lam only
    mu, sd, keep = res["std"]
    lam_idx = int(res["lam_idx"])
    Xtr_n = ((Xt - mu) / sd)[:, keep]
    ymu = Yt.mean(0, keepdim=True)
    Ytr_c = Yt - ymu
    W = _ridge_dual_weights(Xtr_n, Ytr_c, float(RIDGE_LAMBDAS[lam_idx]))  # (d_keep, P)
    d = int(Xt.shape[1])
    p = int(Yt.shape[1])
    M = np.zeros((d, p), dtype=np.float32)
    keep_np = keep.detach().cpu().numpy()
    M[keep_np] = W.detach().cpu().numpy().astype(np.float32)
    return M, lam_idx, int(keep_np.sum())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius-vectorized cosine of two operator matrices (fp64 accumulation)."""
    num = float((a * b).sum(dtype=np.float64))
    na = float((a * a).sum(dtype=np.float64))
    nb = float((b * b).sum(dtype=np.float64))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return num / (np.sqrt(na) * np.sqrt(nb))


def _anchor_gate(summaries_dir: Path) -> dict:
    """Reproduce banked matched-N t11 R2 via the per-turn fold recipe on the FULL cell."""
    with open(RAW_MATCHED_JSON) as f:
        banked = json.load(f)
    rec: dict = {
        "tol": ANCHOR_TOL,
        "source": str(RAW_MATCHED_JSON.relative_to(PROJECT_ROOT)),
        "recipe": (
            "TDM._real_fit_and_folds on the FULL t11 cell (n=171): per-turn "
            "conv-grouped 6-fold folds (TDM._folds_for_turn, FOLD_SEED=0) + "
            "PRESS-ridge _fit_cv; pooled held-out R2. Reproduces "
            "matched.level_171.<model>.19.11.real_r2_mean (sd=0, full cell)."
        ),
        "per_model": {},
    }
    ok = True
    for mt in MODELS:
        paired = TDM._build_pairing(summaries_dir, mt)
        ci = np.asarray([p[0] for p in paired], dtype=np.int64)
        aj = np.asarray([p[1] for p in paired], dtype=np.int64)
        pair_rows = [{"conv_id": p[2], "turn_index": p[3]} for p in paired]
        sel = np.asarray([i for i, p in enumerate(paired) if p[3] == 11], dtype=np.int64)
        arr_c, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", SRC_KIND, LAYER)
        arr_a, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", DST_KIND, LAYER)
        rows = [pair_rows[i] for i in sel]
        rf = TDM._real_fit_and_folds(arr_c[ci][sel], arr_a[aj][sel], rows)
        assert rf is not None, f"anchor: t11 full-cell fit degenerate ({mt})"
        fit, _folds = rf
        recomputed = float(fit["r2"])
        banked_val = float(banked["matched"]["level_171"][mt]["19"]["11"]["real_r2_mean"])
        d = abs(recomputed - banked_val)
        passed = d <= ANCHOR_TOL
        ok = ok and passed
        rec["per_model"][mt] = {
            "cell": f"{mt}/L19/t11 (full n=171)",
            "recomputed_r2": recomputed,
            "banked_matched_r2": banked_val,
            "abs_diff": d,
            "pass": bool(passed),
        }
        print(
            f"[anchor] {mt} L19 t11: recomputed={recomputed:.9f} "
            f"banked={banked_val:.9f} |d|={d:.2e} -> {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
    rec["pass"] = bool(ok)
    if not ok:
        raise SystemExit(
            "ANCHOR GATE FAILED — the full-cell t11 diagonal R2 does not reproduce "
            "the banked matched-N value; refusing to trust the transfer/similarity "
            "matrices. Do not commit these outputs."
        )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-root", default=str(PROJECT_ROOT / "data/issue_825/summaries_dl"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--fig-dir", default=str(FIG_DIR))
    args = ap.parse_args()

    local_root = Path(args.local_root)
    if args.skip_download:
        summaries_dir = local_root / HF_SUMMARIES_BASE
    else:
        summaries_dir = TDM._download_summaries(local_root)
    assert summaries_dir.is_dir(), f"summaries_dir missing: {summaries_dir}"

    t_start = time.time()
    anchor = _anchor_gate(summaries_dir)

    per_model: dict = {}
    n_per_cell: dict = {}
    lambda_choices: dict = {}
    keep_counts: dict = {}

    for model_idx, mt in enumerate(MODELS):
        paired = TDM._build_pairing(summaries_dir, mt)
        ci = np.asarray([p[0] for p in paired], dtype=np.int64)
        aj = np.asarray([p[1] for p in paired], dtype=np.int64)
        conv_all = [p[2] for p in paired]
        turn_all = [p[3] for p in paired]
        arr_c, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", SRC_KIND, LAYER)
        arr_a, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", DST_KIND, LAYER)
        Xall = arr_c[ci]  # (n_pairs, 3584)
        Yall = arr_a[aj]  # (n_pairs, 3584)

        # per-turn 171-subset: global-paired-row indices, and their conv_ids
        turn_sel = {
            t: np.asarray([k for k, tt in enumerate(turn_all) if tt == t], dtype=np.int64)
            for t in TURNS
        }
        sub = {t: _subsample_turn(turn_sel[t], model_idx, t) for t in TURNS}
        conv = {t: [conv_all[k] for k in sub[t]] for t in TURNS}
        Xt = {t: Xall[sub[t]] for t in TURNS}
        Yt = {t: Yall[sub[t]] for t in TURNS}
        n_per_cell[mt] = {str(t): int(sub[t].size) for t in TURNS}
        print(
            f"[compute] {mt}: n_per_turn(subset)={n_per_cell[mt]} ({time.time() - t_start:.0f}s)",
            flush=True,
        )

        # ---- T[i,j] transfer matrix (shared conv-grouped 6-fold partition) ----
        union_convs = [c for t in TURNS for c in conv[t]]
        test_groups = _shared_conv_folds(union_convs)
        Tmat = np.full((len(TURNS), len(TURNS)), np.nan, dtype=np.float64)
        for ai, ti in enumerate(TURNS):
            for bj, tj in enumerate(TURNS):
                Tmat[ai, bj] = _transfer_r2(
                    Xt[ti], Yt[ti], conv[ti], Xt[tj], Yt[tj], conv[tj], test_groups
                )
        # transfer fraction T[i,j] / T[j,j]  (map fit-on-i predicting j, vs j's own diagonal)
        diag = np.array([Tmat[b, b] for b in range(len(TURNS))], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = Tmat / diag[np.newaxis, :]
        print(f"[compute] {mt}: transfer matrix done ({time.time() - t_start:.0f}s)", flush=True)

        # ---- C[i,j] operator similarity + split-half ceiling + shuffled null ----
        ops: dict[int, np.ndarray] = {}
        lam_by_turn: dict[str, int] = {}
        keep_by_turn: dict[str, int] = {}
        for t in TURNS:
            M, lam_idx, keep_ct = _std_operator(Xt[t], Yt[t])
            ops[t] = M
            lam_by_turn[str(t)] = int(lam_idx)
            keep_by_turn[str(t)] = int(keep_ct)
        Cmat = np.full((len(TURNS), len(TURNS)), np.nan, dtype=np.float64)
        for ai, ti in enumerate(TURNS):
            for bj, tj in enumerate(TURNS):
                Cmat[ai, bj] = _cosine(ops[ti], ops[tj])
        lambda_choices[mt] = lam_by_turn
        keep_counts[mt] = keep_by_turn

        # split-half ceiling: fit each half of the turn's conv set, cosine the two
        splithalf: dict[str, float] = {}
        for t in TURNS:
            n = int(sub[t].size)
            perm = np.random.default_rng(SPLITHALF_SEED_BASE + t).permutation(n)
            h1, h2 = perm[: n // 2], perm[n // 2 :]
            M1, _l1, _k1 = _std_operator(Xt[t][h1], Yt[t][h1])
            M2, _l2, _k2 = _std_operator(Xt[t][h2], Yt[t][h2])
            splithalf[str(t)] = _cosine(M1, M2)
            del M1, M2
        # pairwise split-half ceiling sqrt(c_i * c_j), clamped at 0 before sqrt
        ceil_mat = np.full((len(TURNS), len(TURNS)), np.nan, dtype=np.float64)
        for ai, ti in enumerate(TURNS):
            for bj, tj in enumerate(TURNS):
                ci_, cj_ = splithalf[str(ti)], splithalf[str(tj)]
                ceil_mat[ai, bj] = float(np.sqrt(max(0.0, ci_) * max(0.0, cj_)))
        print(
            f"[compute] {mt}: operator similarity + split-half done ({time.time() - t_start:.0f}s)",
            flush=True,
        )

        # shuffled null: fit each turn with answer rows permuted (break pairing)
        ops_shuf: dict[int, np.ndarray] = {}
        real_vs_shuf: dict[str, float] = {}
        for t in TURNS:
            n = int(sub[t].size)
            perm = np.random.default_rng(SHUFFLE_SEED_BASE + t).permutation(n)
            Ms, _l, _k = _std_operator(Xt[t], Yt[t][perm])
            ops_shuf[t] = Ms
            real_vs_shuf[str(t)] = _cosine(ops[t], Ms)
        shuf_adjacent: dict[str, float] = {}
        for ti, tj in ADJ_PAIRS:
            shuf_adjacent[f"{ti}-{tj}"] = _cosine(ops_shuf[ti], ops_shuf[tj])
        print(f"[compute] {mt}: shuffled null done ({time.time() - t_start:.0f}s)", flush=True)

        per_model[mt] = {
            "turns": TURNS,
            "transfer_r2": Tmat.tolist(),
            "transfer_fraction": frac.tolist(),
            "transfer_diagonal": diag.tolist(),
            "operator_cosine": Cmat.tolist(),
            "splithalf_cosine_per_turn": splithalf,
            "splithalf_ceiling_pairwise": ceil_mat.tolist(),
            "shuffled_cosine_real_vs_shuffled_per_turn": real_vs_shuf,
            "shuffled_cosine_adjacent_pairs": shuf_adjacent,
            "lambda_indices_full_cell": lam_by_turn,
            "keep_dims_full_cell": keep_by_turn,
        }
        del ops, ops_shuf, Xt, Yt, Xall, Yall  # free before next model

    payload = {
        "issue": 825,
        "analysis": "cross-turn transfer matrix + operator similarity (n=171 preview)",
        "preview_of": "queued round-11 turn-dynamics-allturns-5000 deliverable",
        "description": (
            "At layer 19, both models, on the banked #1092 per-turn tensors "
            "(context_k -> answer_k_t1, offset 0), turns {1,3,5,7,9,11} matched to "
            "n=171: (1) cross-turn TRANSFER T[i,j] — held-out R2 of the ridge map "
            "fit on turn i applied to turn j's answers, under ONE shared "
            "conv-grouped 6-fold partition; (2) operator SIMILARITY C[i,j] — cosine "
            "between per-turn standardized-space ridge operators, with a within-turn "
            "split-half cosine ceiling and a shuffled-pairing cosine null."
        ),
        "arm": "context_k -> answer_k_t1 (offset 0)",
        "layer": LAYER,
        "turns": TURNS,
        "adjacent_pairs": [f"{a}-{b}" for a, b in ADJ_PAIRS],
        "n_match": N_MATCH,
        "models": list(MODELS),
        "model_cells": MODEL_CELL,
        "estimator": (
            "PRESS-ridge (issue923 press_fit_predict, standardize=True, lambda grid "
            "[1e-2,1e-1,1,10,100,1000]); transfer folds are grouped-by-conv_id "
            "6-fold; held-out R2 pooled over folds via issue1092_fit_grid._r2. "
            "Operators materialized as the standardized-space dual-ridge weight "
            "matrix W = issue658_fit_predictors._ridge_dual_weights(Xtr_n, Ytr_c, "
            "lam) consistent with press_fit_predict's prediction path "
            "(pred = Xte_n @ W + ymu)."
        ),
        "transfer_recipe": (
            "Subsample each turn cell to n=171 (one draw, fixed per-(model,turn) "
            "seed mirroring issue825_turn_depth_matched_n level_171/draw-0). Build "
            "ONE conv-grouped 6-fold partition over the UNION of subsampled "
            "conv_ids (SHARED_FOLD_SEED, shuffle-then-stride like "
            "issue1092_fit_grid._folds_from_manifest), applied identically at every "
            "turn. Per fold: fit ridge on turn-i rows whose conv is NOT in the "
            "fold's test group (lambda by PRESS-LOO on that train set), predict "
            "turn-j rows whose conv IS in the test group; pool held-out turn-j "
            "predictions across folds and report _r2(Y_j, pred) (turn-j's own mean "
            "as the reference). A conv shared by turns i and j gets the SAME fold "
            "group (shared partition), so it is never in turn-i-train and "
            "turn-j-test simultaneously (no leakage). T[i,i] reduces to standard "
            "grouped CV and is the matrix's own diagonal reference; "
            "transfer_fraction[i,j] = T[i,j]/T[j,j]."
        ),
        "similarity_recipe": (
            "Per turn, fit the full n=171-subset PRESS ridge (lambda by PRESS-LOO "
            "on the full subset; recorded in lambda_indices_full_cell) and "
            "materialize the standardized-space operator M (3584 x 3584, fp32, "
            "scattered into the ambient dim with zeros at dropped-degenerate dims). "
            "C[i,j] = cosine(vec(M_i), vec(M_j)) (Frobenius, fp64 accumulation). "
            "Split-half ceiling: split the turn's conv set in half (seeded), fit "
            "each half (n~85), cosine the two half-operators; pairwise ceiling "
            "sqrt(max(0,c_i)*max(0,c_j)). Shuffled null: fit each turn with the "
            "answer rows permuted (seeded, break the context<->answer pairing); "
            "report cosine(real, shuffled) per turn and cosine(shuffled_i, "
            "shuffled_j) for adjacent turns as the null scale."
        ),
        "caveats": [
            "Half-fits use n~85, so the split-half ceiling is CONSERVATIVE relative "
            "to the n=171 operator fits (lower ceiling than the true n=171 "
            "reliability).",
            "Each turn's operator lives in ITS OWN standardized coordinates "
            "(per-turn train mean/std, faithful to the reused PRESS estimator), so "
            "cross-turn C[i,j] reflects genuine operator differences PLUS any "
            "per-turn standardization drift. The split-half ceiling uses the same "
            "per-turn standardization, so it is the right estimation-noise floor.",
            "n=171 PREVIEW of the queued round-11 turn-dynamics-allturns-5000 "
            "deliverable; standard errors on individual matrix cells are not "
            "reported here (one draw per cell).",
        ],
        "seeds": {
            "sub_seed_base": SUB_SEED_BASE,
            "shared_fold_seed": SHARED_FOLD_SEED,
            "splithalf_seed_base": SPLITHALF_SEED_BASE,
            "shuffle_seed_base": SHUFFLE_SEED_BASE,
            "fold_seed_anchor": TDM._folds_for_turn.__module__ + ".FOLD_SEED (0)",
        },
        "n_per_cell": n_per_cell,
        "lambda_indices_full_cell": lambda_choices,
        "keep_dims_full_cell": keep_counts,
        "anchor_gate": anchor,
        "source_paths": {
            "banked_store_prefix": f"{HF_SUMMARIES_BASE}/dynamics_{{instruct,pretrained}}/",
            "banked_store_files": (
                f"{HF_SUMMARIES_BASE}/dynamics_{{instruct,pretrained}}/"
                f"{{context_k,answer_k_t1}}_L{LAYER:02d}_shard*.npy + "
                f"row_index_{{context_k,answer_k_t1}}_shard*.jsonl"
            ),
            "turn_depth_matched_n_json": str(RAW_MATCHED_JSON.relative_to(PROJECT_ROOT)),
            "turn_depth_map_json": str(RAW_MAP_JSON.relative_to(PROJECT_ROOT)),
        },
        "hf_data_repo": HF_DATA_REPO,
        "hf_summaries_prefix": HF_SUMMARIES_BASE,
        "git_commit": TDM._git_commit(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "wall_time_s": round(time.time() - t_start, 1),
        "per_model": per_model,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[write] {out_json}  (wall {payload['wall_time_s']}s)")

    _plot(payload, Path(args.fig_dir))


def _plot(payload: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    fig_dir.mkdir(parents=True, exist_ok=True)
    turns = payload["turns"]
    labels = [str(t) for t in turns]
    model_titles = {"instruct": "Qwen-2.5-7B-Instruct", "pretrained": "Qwen-2.5-7B (base)"}

    def _heatmap(ax, M, *, vmin, vmax, cmap, annot, diag_annot=None):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(turns)))
        ax.set_yticks(range(len(turns)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for r in range(len(turns)):
            for c in range(len(turns)):
                v = M[r][c]
                if not np.isfinite(v):
                    continue
                txt = f"{v:.2f}"
                if diag_annot is not None and r == c:
                    txt = f"{v:.2f}\n({diag_annot[r]:.2f})"
                # readable text colour vs cell shade
                rgba = im.cmap(im.norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(
                    c,
                    r,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=6.5 if diag_annot is not None else 7.5,
                    color="white" if lum < 0.5 else "black",
                )
        return im

    # ---- Figure 1: transfer R2 T[i,j] ----
    Ti = np.asarray(payload["per_model"]["instruct"]["transfer_r2"], dtype=float)
    Tp = np.asarray(payload["per_model"]["pretrained"]["transfer_r2"], dtype=float)
    tmax = float(np.nanmax(np.abs(np.concatenate([Ti.ravel(), Tp.ravel()]))))
    tmax = max(tmax, 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    im = None
    for ax, mt, M in ((axes[0], "instruct", Ti), (axes[1], "pretrained", Tp)):
        im = _heatmap(ax, M, vmin=-tmax, vmax=tmax, cmap="RdBu_r", annot=True)
        ax.set_title(model_titles[mt], fontsize=11)
        ax.set_xlabel("evaluated on turn j")
        ax.set_ylabel("map fit on turn i")
    fig.suptitle(
        "Cross-turn transfer: held-out $R^2$ of the layer-19 context$\\to$answer "
        "map fit on turn i, evaluated on turn j (n=171)",
        fontsize=11.5,
        y=1.02,
    )
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label(r"held-out $R^2$")
    savefig_paper(fig, f"issue_825/{FIG_T}", dir=str(fig_dir.parent))
    plt.close(fig)

    # ---- Figure 2: operator cosine C[i,j] with split-half ceiling on the diagonal ----
    Ci = np.asarray(payload["per_model"]["instruct"]["operator_cosine"], dtype=float)
    Cp = np.asarray(payload["per_model"]["pretrained"]["operator_cosine"], dtype=float)
    sh_i = payload["per_model"]["instruct"]["splithalf_cosine_per_turn"]
    sh_p = payload["per_model"]["pretrained"]["splithalf_cosine_per_turn"]
    ceil_i = [float(sh_i[str(t)]) for t in turns]
    ceil_p = [float(sh_p[str(t)]) for t in turns]
    cmax = float(np.nanmax(np.abs(np.concatenate([Ci.ravel(), Cp.ravel()]))))
    cmax = max(cmax, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    im = None
    for ax, mt, M, ceil in (
        (axes[0], "instruct", Ci, ceil_i),
        (axes[1], "pretrained", Cp, ceil_p),
    ):
        im = _heatmap(ax, M, vmin=-cmax, vmax=cmax, cmap="RdBu_r", annot=True, diag_annot=ceil)
        ax.set_title(model_titles[mt], fontsize=11)
        ax.set_xlabel("turn j")
        ax.set_ylabel("turn i")
    fig.suptitle(
        "Operator similarity: cosine between layer-19 per-turn ridge operators "
        "(diagonal shows the within-turn split-half ceiling in parentheses)",
        fontsize=11.5,
        y=1.02,
    )
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("operator cosine")
    savefig_paper(fig, f"issue_825/{FIG_C}", dir=str(fig_dir.parent))
    plt.close(fig)

    # savefig_paper already wrote <dir>/<stem>.meta.json with commit + per-point
    # `points` (the dashboard data viewer reads it) — MERGE our caption in rather
    # than clobber that sidecar.
    for stem, caption in (
        (
            FIG_T,
            "Cross-turn transfer matrix T[i,j]: held-out R2 of the layer-19 linear "
            "context->answer map (context_k -> answer_k_t1, offset 0) FIT on turn i "
            "and EVALUATED on turn j, under one shared conversation-grouped 6-fold "
            "partition, both models, matched to n=171. y=fit-on-turn-i, "
            "x=evaluated-on-turn-j; diverging cmap centered at 0. Diagonal = the "
            "turn's own grouped-CV R2 (the transfer reference).",
        ),
        (
            FIG_C,
            "Operator similarity matrix C[i,j]: cosine between the layer-19 per-turn "
            "standardized-space ridge operators (full n=171 subset per turn), both "
            "models; diverging cmap centered at 0. Diagonal cells are annotated with "
            "the within-turn split-half cosine ceiling (parentheses; n~85 half-fits, "
            "conservative relative to the n=171 operators). Shuffled-pairing cosine "
            "nulls are in the results JSON.",
        ),
    ):
        meta_path = fig_dir / f"{stem}.meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                meta = {}
        meta["caption"] = caption
        meta["source_results_json"] = str(OUT_JSON.relative_to(PROJECT_ROOT))
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
    print(f"[write] {fig_dir / (FIG_T + '.png')} + {FIG_C}.png (+pdf +meta.json)")


if __name__ == "__main__":
    main()
