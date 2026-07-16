#!/usr/bin/env python3
"""Issue #825 — cross-turn TRANSFER matrix + operator SIMILARITY.

Two follow-on questions about the per-turn linear context->answer maps
(``context_k -> answer_k_t1``, offset 0), at layer 19, both models, on the
banked #1092 per-turn tensors, turns {1,3,5,7,9,11}:

1. **Cross-turn TRANSFER T[i,j]** — is the ridge map FIT on turn i still
   predictive of turn j's answers? Build ONE shared conversation-grouped 6-fold
   partition over the union of conv_ids, applied identically at every turn. Per
   fold: fit ridge on turn-i TRAIN rows (conv in the fold's train groups),
   predict turn-j TEST rows (conv in the fold's test group), pool the held-out
   predictions across folds, report the variance-weighted held-out R2 the reused
   #1092 core computes (``_r2`` over pooled preds). The shared conv-grouped
   partition means a conversation appearing at BOTH turn i and turn j is never
   simultaneously in turn-i-train and turn-j-test (no leakage). For i==j this
   reduces to standard grouped CV; T[i,i] is the matrix's OWN diagonal
   reference, and we also report the transfer fraction T[i,j]/T[j,j].

2. **Operator SIMILARITY C[i,j]** — cosine between the per-turn ridge OPERATORS.
   The operator M_i is the standardized-space ridge weight matrix W_i (d x P,
   d=P=3584) the fit on turn i's cell produces (lambda by PRESS-LOO on the
   cell). C[i,j] = cosine(vec(M_i), vec(M_j)). Two references: a within-turn
   SPLIT-HALF cosine ceiling (fit each half of the turn's conv set, cosine the
   two half-operators) and a SHUFFLED-pairing cosine null (fit with the answer
   rows permuted; report cosine(real, shuffled) per turn and
   cosine(shuffled_i, shuffled_j) for adjacent turns as the null scale).

PRIMARY variant = FULL per-turn n (n = 497/388/354/332/236/171 for turns
1/3/5/7/9/11 — the same cells the raw turn_depth_map curve and the other #825
experiments used). SECONDARY companion = MATCHED n=171 (each turn subsampled to
171, one draw, fixed seed) — this controls the n-per-row confound when comparing
ACROSS fit turns. Both variants run T + C + split-half + shuffled and are stored
under ``full_n`` / ``matched_171`` keys with n recorded per cell.

Interpretive caveat (recorded in the results JSON): each turn's operator lives
in ITS OWN standardized coordinates (per-turn train mean/std, faithful to the
reused PRESS estimator), so cross-turn C[i,j] includes any per-turn
standardization drift on top of genuine operator differences. The split-half
ceiling is computed with the SAME per-turn standardization, so it is the right
reference for the estimation-noise floor. At FULL n the shallow turns' ceiling
(t1 halves ~248) is far less conservative than at matched 171 (halves ~85).

Anchor gate (FAIL = abort, do not commit): with the per-turn fold recipe
(``TDM._folds_for_turn``, FOLD_SEED=0) on each FULL turn cell, the diagonal R2
must reproduce the banked ``eval_results/issue_825/turn_depth_map/results.json``
``results.<model>.19.<turn>.real_r2`` to <1e-6 for EVERY turn (t1 instruct
0.2117 / pretrained 0.0593 ... t11 0.2131/0.2530). Additionally the t11 full
cell must match the banked matched-N value (t11 full == matched at 171), which
gates the matched leg. This independently re-exercises the reused #1092
PRESS-ridge core across the entire raw diagonal.

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

# Seeds. The matched-171 subsample seed recipe mirrors
# issue825_turn_depth_matched_n's level_171 / draw-0 recipe so the n=171 subsets
# are reproducible and cross-checkable against the banked matched-N draw-0
# subsamples. Distinct split-half / shuffle seed bases per variant keep the two
# variants' null draws independent.
SUB_SEED_BASE = 8250  # == issue825_turn_depth_matched_n.SUB_SEED_BASE
SHARED_FOLD_SEED_FULL = 82519  # shared conv-grouped 6-fold partition (full-n union)
SHARED_FOLD_SEED_MATCHED = 82523  # shared conv-grouped 6-fold partition (matched-171 union)
SPLITHALF_SEED_FULL = 82561
SPLITHALF_SEED_MATCHED = 82531
SHUFFLE_SEED_FULL = 82577
SHUFFLE_SEED_MATCHED = 82547

ANCHOR_TOL = 1e-6

RAW_MATCHED_JSON = PROJECT_ROOT / "eval_results/issue_825/turn_depth_matched_n/results.json"
RAW_MAP_JSON = PROJECT_ROOT / "eval_results/issue_825/turn_depth_map/results.json"
OUT_JSON = PROJECT_ROOT / "eval_results/issue_825/turn_transfer_similarity/results.json"
FIG_DIR = PROJECT_ROOT / "figures/issue_825"
FIG_T = "turn_transfer_r2"  # primary (full-n)
FIG_C = "turn_operator_cosine"  # primary (full-n)
FIG_T_MATCHED = "turn_transfer_r2_matched171"  # secondary companion


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


def _shared_conv_folds(union_convs: list[str], seed: int) -> list[set[str]]:
    """ONE conv-grouped 6-fold partition over the union of conv_ids.

    Mirrors issue1092_fit_grid._folds_from_manifest's shuffle-then-stride split,
    but over the shared UNION (deterministic, seeded), returning the per-fold
    TEST conv groups. Each conv is in exactly one test group, so a turn row is
    predicted exactly once (in the fold where its conv is the test group).
    """
    uniq = sorted(set(union_convs))
    rng = np.random.default_rng(seed)
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
    """Cell PRESS ridge operator M in the AMBIENT (d, P) standardized space.

    lambda by PRESS-LOO on the cell (press_fit_predict standardize=True).
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


def _compute_variant(
    Xall: np.ndarray,
    Yall: np.ndarray,
    conv_all: list[str],
    sel_by_turn: dict[int, np.ndarray],
    *,
    fold_seed: int,
    splithalf_seed: int,
    shuffle_seed: int,
) -> dict:
    """Full T + fraction + C + split-half ceiling + shuffled null for one variant.

    ``sel_by_turn[t]`` = the GLOBAL paired-row indices for turn t's cell (full or
    subsampled). The T/C recipe is identical across variants — only the row
    selection per turn differs.
    """
    conv = {t: [conv_all[k] for k in sel_by_turn[t]] for t in TURNS}
    Xt = {t: Xall[sel_by_turn[t]] for t in TURNS}
    Yt = {t: Yall[sel_by_turn[t]] for t in TURNS}
    n_per_turn = {str(t): int(sel_by_turn[t].size) for t in TURNS}

    # ---- T[i,j] transfer matrix (shared conv-grouped 6-fold partition) ----
    union_convs = [c for t in TURNS for c in conv[t]]
    test_groups = _shared_conv_folds(union_convs, fold_seed)
    Tmat = np.full((len(TURNS), len(TURNS)), np.nan, dtype=np.float64)
    for ai, ti in enumerate(TURNS):
        for bj, tj in enumerate(TURNS):
            Tmat[ai, bj] = _transfer_r2(
                Xt[ti], Yt[ti], conv[ti], Xt[tj], Yt[tj], conv[tj], test_groups
            )
    diag = np.array([Tmat[b, b] for b in range(len(TURNS))], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = Tmat / diag[np.newaxis, :]  # T[i,j] / T[j,j]

    # ---- C[i,j] operator similarity (cell operators) ----
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

    # split-half ceiling: fit each half of the turn's conv set, cosine the two
    splithalf: dict[str, float] = {}
    splithalf_n: dict[str, list[int]] = {}
    for t in TURNS:
        n = int(sel_by_turn[t].size)
        perm = np.random.default_rng(splithalf_seed + t).permutation(n)
        h1, h2 = perm[: n // 2], perm[n // 2 :]
        M1, _l1, _k1 = _std_operator(Xt[t][h1], Yt[t][h1])
        M2, _l2, _k2 = _std_operator(Xt[t][h2], Yt[t][h2])
        splithalf[str(t)] = _cosine(M1, M2)
        splithalf_n[str(t)] = [int(h1.size), int(h2.size)]
        del M1, M2
    ceil_mat = np.full((len(TURNS), len(TURNS)), np.nan, dtype=np.float64)
    for ai, ti in enumerate(TURNS):
        for bj, tj in enumerate(TURNS):
            ci_, cj_ = splithalf[str(ti)], splithalf[str(tj)]
            ceil_mat[ai, bj] = float(np.sqrt(max(0.0, ci_) * max(0.0, cj_)))

    # shuffled null: fit each turn with answer rows permuted (break pairing)
    ops_shuf: dict[int, np.ndarray] = {}
    real_vs_shuf: dict[str, float] = {}
    for t in TURNS:
        n = int(sel_by_turn[t].size)
        perm = np.random.default_rng(shuffle_seed + t).permutation(n)
        Ms, _l, _k = _std_operator(Xt[t], Yt[t][perm])
        ops_shuf[t] = Ms
        real_vs_shuf[str(t)] = _cosine(ops[t], Ms)
    shuf_adjacent: dict[str, float] = {}
    for ti, tj in ADJ_PAIRS:
        shuf_adjacent[f"{ti}-{tj}"] = _cosine(ops_shuf[ti], ops_shuf[tj])

    del ops, ops_shuf
    return {
        "turns": TURNS,
        "n_per_turn": n_per_turn,
        "transfer_r2": Tmat.tolist(),
        "transfer_fraction": frac.tolist(),
        "transfer_diagonal": diag.tolist(),
        "operator_cosine": Cmat.tolist(),
        "splithalf_cosine_per_turn": splithalf,
        "splithalf_n_per_turn": splithalf_n,
        "splithalf_ceiling_pairwise": ceil_mat.tolist(),
        "shuffled_cosine_real_vs_shuffled_per_turn": real_vs_shuf,
        "shuffled_cosine_adjacent_pairs": shuf_adjacent,
        "lambda_indices_cell": lam_by_turn,
        "keep_dims_cell": keep_by_turn,
    }


def _anchor_gate(summaries_dir: Path) -> dict:
    """Reproduce the banked turn_depth_map raw diagonal via the per-turn recipe.

    For EVERY turn: TDM._real_fit_and_folds on the FULL turn cell (per-turn
    conv-grouped 6-fold folds, FOLD_SEED=0) must reproduce
    turn_depth_map results.<model>.19.<turn>.real_r2 to <1e-6. Additionally the
    t11 full cell must match the banked matched-N value (t11 full == matched at
    171), gating the matched leg.
    """
    with open(RAW_MAP_JSON) as f:
        raw_map = json.load(f)
    with open(RAW_MATCHED_JSON) as f:
        banked_matched = json.load(f)
    rec: dict = {
        "tol": ANCHOR_TOL,
        "sources": {
            "turn_depth_map": str(RAW_MAP_JSON.relative_to(PROJECT_ROOT)),
            "turn_depth_matched_n": str(RAW_MATCHED_JSON.relative_to(PROJECT_ROOT)),
        },
        "recipe": (
            "TDM._real_fit_and_folds on each FULL turn cell: per-turn conv-grouped "
            "6-fold folds (TDM._folds_for_turn, FOLD_SEED=0) + PRESS-ridge "
            "_fit_cv; pooled held-out R2. Reproduces "
            "turn_depth_map results.<model>.19.<turn>.real_r2 for every turn; the "
            "t11 cell additionally matches matched_n level_171.<model>.19.11 "
            "(t11 full == matched at 171)."
        ),
        "per_model": {},
    }
    ok = True
    for mt in MODELS:
        paired = TDM._build_pairing(summaries_dir, mt)
        ci = np.asarray([p[0] for p in paired], dtype=np.int64)
        aj = np.asarray([p[1] for p in paired], dtype=np.int64)
        pair_rows = [{"conv_id": p[2], "turn_index": p[3]} for p in paired]
        arr_c, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", SRC_KIND, LAYER)
        arr_a, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", DST_KIND, LAYER)
        Xm = arr_c[ci]
        Ym = arr_a[aj]
        per_turn: dict = {}
        for t in TURNS:
            sel = np.asarray([i for i, p in enumerate(paired) if p[3] == t], dtype=np.int64)
            rows = [pair_rows[i] for i in sel]
            rf = TDM._real_fit_and_folds(Xm[sel], Ym[sel], rows)
            assert rf is not None, f"anchor: t{t} full-cell fit degenerate ({mt})"
            fit, _folds = rf
            recomputed = float(fit["r2"])
            banked = float(raw_map["results"][mt]["19"][str(t)]["real_r2"])
            d = abs(recomputed - banked)
            passed = d <= ANCHOR_TOL
            entry = {
                "n": int(sel.size),
                "recomputed_r2": recomputed,
                "banked_turn_depth_map_r2": banked,
                "abs_diff": d,
                "pass": bool(passed),
            }
            if t == 11:
                bmv = float(banked_matched["matched"]["level_171"][mt]["19"]["11"]["real_r2_mean"])
                entry["banked_matched_n_r2"] = bmv
                entry["abs_diff_vs_matched"] = abs(recomputed - bmv)
                passed = passed and entry["abs_diff_vs_matched"] <= ANCHOR_TOL
                entry["pass"] = bool(passed)
            ok = ok and passed
            per_turn[str(t)] = entry
            print(
                f"[anchor] {mt} L19 t{t:>2} (n={sel.size}): recomputed={recomputed:.9f} "
                f"banked={banked:.9f} |d|={d:.2e} -> {'PASS' if passed else 'FAIL'}",
                flush=True,
            )
        rec["per_model"][mt] = per_turn
    rec["pass"] = bool(ok)
    if not ok:
        raise SystemExit(
            "ANCHOR GATE FAILED — a full-cell per-turn diagonal R2 does not reproduce "
            "the banked turn_depth_map value; refusing to trust the transfer/similarity "
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

    full_n: dict = {}
    matched_171: dict = {}
    for model_idx, mt in enumerate(MODELS):
        paired = TDM._build_pairing(summaries_dir, mt)
        ci = np.asarray([p[0] for p in paired], dtype=np.int64)
        aj = np.asarray([p[1] for p in paired], dtype=np.int64)
        conv_all = [p[2] for p in paired]
        turn_all = [p[3] for p in paired]
        arr_c, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", SRC_KIND, LAYER)
        arr_a, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", DST_KIND, LAYER)
        Xall = arr_c[ci]
        Yall = arr_a[aj]

        turn_sel = {
            t: np.asarray([k for k, tt in enumerate(turn_all) if tt == t], dtype=np.int64)
            for t in TURNS
        }
        sel_full = turn_sel  # FULL per-turn n
        sel_matched = {t: _subsample_turn(turn_sel[t], model_idx, t) for t in TURNS}

        full_n[mt] = _compute_variant(
            Xall,
            Yall,
            conv_all,
            sel_full,
            fold_seed=SHARED_FOLD_SEED_FULL,
            splithalf_seed=SPLITHALF_SEED_FULL,
            shuffle_seed=SHUFFLE_SEED_FULL,
        )
        print(
            f"[compute] {mt} FULL-n done: n_per_turn={full_n[mt]['n_per_turn']} "
            f"({time.time() - t_start:.0f}s)",
            flush=True,
        )
        matched_171[mt] = _compute_variant(
            Xall,
            Yall,
            conv_all,
            sel_matched,
            fold_seed=SHARED_FOLD_SEED_MATCHED,
            splithalf_seed=SPLITHALF_SEED_MATCHED,
            shuffle_seed=SHUFFLE_SEED_MATCHED,
        )
        print(
            f"[compute] {mt} MATCHED-171 done: n_per_turn={matched_171[mt]['n_per_turn']} "
            f"({time.time() - t_start:.0f}s)",
            flush=True,
        )
        del Xall, Yall

    payload = {
        "issue": 825,
        "analysis": "cross-turn transfer matrix + operator similarity",
        "preview_of": "queued round-11 turn-dynamics-allturns-5000 deliverable",
        "description": (
            "At layer 19, both models, on the banked #1092 per-turn tensors "
            "(context_k -> answer_k_t1, offset 0), turns {1,3,5,7,9,11}: "
            "(1) cross-turn TRANSFER T[i,j] — held-out R2 of the ridge map fit on "
            "turn i applied to turn j's answers, under ONE shared conv-grouped "
            "6-fold partition; (2) operator SIMILARITY C[i,j] — cosine between "
            "per-turn standardized-space ridge operators, with a within-turn "
            "split-half cosine ceiling and a shuffled-pairing cosine null. PRIMARY "
            "variant uses the FULL per-turn n (497/388/354/332/236/171); SECONDARY "
            "companion matches every turn to n=171 to control the n-per-row "
            "confound across fit turns."
        ),
        "arm": "context_k -> answer_k_t1 (offset 0)",
        "layer": LAYER,
        "turns": TURNS,
        "adjacent_pairs": [f"{a}-{b}" for a, b in ADJ_PAIRS],
        "primary_variant": "full_n",
        "secondary_variant": "matched_171",
        "matched_n": N_MATCH,
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
            "Build ONE conv-grouped 6-fold partition over the UNION of the "
            "variant's conv_ids (shuffle-then-stride like "
            "issue1092_fit_grid._folds_from_manifest), applied identically at "
            "every turn. Per fold: fit ridge on turn-i rows whose conv is NOT in "
            "the fold's test group (lambda by PRESS-LOO on that train set), "
            "predict turn-j rows whose conv IS in the test group; pool held-out "
            "turn-j predictions across folds and report _r2(Y_j, pred) (turn-j's "
            "own mean as the reference). A conv shared by turns i and j gets the "
            "SAME fold group (shared partition), so it is never in turn-i-train "
            "and turn-j-test simultaneously (no leakage). T[i,i] reduces to "
            "standard grouped CV (the matrix's own diagonal reference); "
            "transfer_fraction[i,j] = T[i,j]/T[j,j]. Per cell, n_fit = "
            "n_per_turn[i] and n_eval = n_per_turn[j]."
        ),
        "similarity_recipe": (
            "Per turn, fit the cell's PRESS ridge (lambda by PRESS-LOO on the "
            "cell; recorded in lambda_indices_cell) and materialize the "
            "standardized-space operator M (3584 x 3584, fp32, scattered into the "
            "ambient dim with zeros at dropped-degenerate dims). "
            "C[i,j] = cosine(vec(M_i), vec(M_j)) (Frobenius, fp64 accumulation). "
            "Split-half ceiling: split the turn's conv set in half (seeded), fit "
            "each half, cosine the two half-operators (half sizes in "
            "splithalf_n_per_turn); pairwise ceiling "
            "sqrt(max(0,c_i)*max(0,c_j)). Shuffled null: fit each turn with the "
            "answer rows permuted (seeded, break the context<->answer pairing); "
            "report cosine(real, shuffled) per turn and cosine(shuffled_i, "
            "shuffled_j) for adjacent turns as the null scale."
        ),
        "caveats": [
            "Each turn's operator lives in ITS OWN standardized coordinates "
            "(per-turn train mean/std, faithful to the reused PRESS estimator), so "
            "cross-turn C[i,j] reflects genuine operator differences PLUS any "
            "per-turn standardization drift. The split-half ceiling uses the same "
            "per-turn standardization, so it is the right estimation-noise floor.",
            "The split-half ceiling is conservative relative to the cell "
            "operators (half-fits use n/2). At FULL n the shallow turns' ceiling "
            "(t1 halves ~248) is far less conservative than at matched 171 "
            "(halves ~85); the off-diagonal cosines can still exceed the ceiling "
            "when the cell operators are less noisy than the half-fits.",
            "One draw per cell (subsample for matched_171; the full cell for "
            "full_n) — standard errors on individual matrix cells are not "
            "reported here. The queued round-11 turn-dynamics-allturns-5000 "
            "deliverable will carry repeated draws for cell-level error bars.",
        ],
        "seeds": {
            "sub_seed_base": SUB_SEED_BASE,
            "shared_fold_seed_full": SHARED_FOLD_SEED_FULL,
            "shared_fold_seed_matched": SHARED_FOLD_SEED_MATCHED,
            "splithalf_seed_full": SPLITHALF_SEED_FULL,
            "splithalf_seed_matched": SPLITHALF_SEED_MATCHED,
            "shuffle_seed_full": SHUFFLE_SEED_FULL,
            "shuffle_seed_matched": SHUFFLE_SEED_MATCHED,
            "fold_seed_anchor": TDM._folds_for_turn.__module__ + ".FOLD_SEED (0)",
        },
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
        "full_n": {"per_model": full_n},
        "matched_171": {"per_model": matched_171, "n_match": N_MATCH},
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[write] {out_json}  (wall {payload['wall_time_s']}s)")

    _plot(payload, Path(args.fig_dir))


def _heatmap(ax, im_holder, M, *, vmin, vmax, cmap, diag_annot=None, fontsize=7.5):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    n = len(M)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [str(t) for t in TURNS]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for r in range(n):
        for c in range(n):
            v = M[r][c]
            if not np.isfinite(v):
                continue
            txt = f"{v:.2f}"
            if diag_annot is not None and r == c:
                txt = f"{v:.2f}\n({diag_annot[r]:.2f})"
            rgba = im.cmap(im.norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                c,
                r,
                txt,
                ha="center",
                va="center",
                fontsize=6.5 if diag_annot is not None else fontsize,
                color="white" if lum < 0.5 else "black",
            )
    im_holder.append(im)
    return im


def _plot(payload: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_titles = {"instruct": "Qwen-2.5-7B-Instruct", "pretrained": "Qwen-2.5-7B (base)"}

    def _npt(variant, mt):
        return payload[variant]["per_model"][mt]["n_per_turn"]

    # ---- Figure 1 (PRIMARY): full-n transfer R2 T[i,j] ----
    Ti = np.asarray(payload["full_n"]["per_model"]["instruct"]["transfer_r2"], dtype=float)
    Tp = np.asarray(payload["full_n"]["per_model"]["pretrained"]["transfer_r2"], dtype=float)
    tmax = max(float(np.nanmax(np.abs(np.concatenate([Ti.ravel(), Tp.ravel()])))), 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    holder: list = []
    for ax, mt, M in ((axes[0], "instruct", Ti), (axes[1], "pretrained", Tp)):
        _heatmap(ax, holder, M, vmin=-tmax, vmax=tmax, cmap="RdBu_r")
        npt = _npt("full_n", mt)
        ax.set_title(
            f"{model_titles[mt]}\n(n={'/'.join(str(npt[str(t)]) for t in TURNS)})", fontsize=10
        )
        ax.set_xlabel("evaluated on turn j")
        ax.set_ylabel("map fit on turn i")
    fig.suptitle(
        "Cross-turn transfer: held-out $R^2$ of the layer-19 context$\\to$answer "
        "map fit on turn i, evaluated on turn j (full n)",
        fontsize=11.5,
        y=1.02,
    )
    fig.colorbar(holder[0], ax=axes, fraction=0.046, pad=0.04).set_label(r"held-out $R^2$")
    savefig_paper(fig, f"issue_825/{FIG_T}", dir=str(fig_dir.parent))
    plt.close(fig)

    # ---- Figure 2 (PRIMARY): full-n operator cosine C[i,j] w/ split-half ceiling diag ----
    Ci = np.asarray(payload["full_n"]["per_model"]["instruct"]["operator_cosine"], dtype=float)
    Cp = np.asarray(payload["full_n"]["per_model"]["pretrained"]["operator_cosine"], dtype=float)
    sh_i = payload["full_n"]["per_model"]["instruct"]["splithalf_cosine_per_turn"]
    sh_p = payload["full_n"]["per_model"]["pretrained"]["splithalf_cosine_per_turn"]
    ceil_i = [float(sh_i[str(t)]) for t in TURNS]
    ceil_p = [float(sh_p[str(t)]) for t in TURNS]
    cmax = max(float(np.nanmax(np.abs(np.concatenate([Ci.ravel(), Cp.ravel()])))), 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    holder = []
    for ax, mt, M, ceil in (
        (axes[0], "instruct", Ci, ceil_i),
        (axes[1], "pretrained", Cp, ceil_p),
    ):
        _heatmap(ax, holder, M, vmin=-cmax, vmax=cmax, cmap="RdBu_r", diag_annot=ceil)
        npt = _npt("full_n", mt)
        ax.set_title(
            f"{model_titles[mt]}\n(n={'/'.join(str(npt[str(t)]) for t in TURNS)})", fontsize=10
        )
        ax.set_xlabel("turn j")
        ax.set_ylabel("turn i")
    fig.suptitle(
        "Operator similarity: cosine between layer-19 per-turn ridge operators, "
        "full n (diagonal shows the within-turn split-half ceiling in parentheses)",
        fontsize=11.5,
        y=1.02,
    )
    fig.colorbar(holder[0], ax=axes, fraction=0.046, pad=0.04).set_label("operator cosine")
    savefig_paper(fig, f"issue_825/{FIG_C}", dir=str(fig_dir.parent))
    plt.close(fig)

    # ---- Figure 3 (SECONDARY companion): matched-171 transfer R2 T[i,j] ----
    Tim = np.asarray(payload["matched_171"]["per_model"]["instruct"]["transfer_r2"], dtype=float)
    Tpm = np.asarray(payload["matched_171"]["per_model"]["pretrained"]["transfer_r2"], dtype=float)
    tmm = max(float(np.nanmax(np.abs(np.concatenate([Tim.ravel(), Tpm.ravel()])))), 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    holder = []
    for ax, mt, M in ((axes[0], "instruct", Tim), (axes[1], "pretrained", Tpm)):
        _heatmap(ax, holder, M, vmin=-tmm, vmax=tmm, cmap="RdBu_r")
        ax.set_title(f"{model_titles[mt]} (matched n=171)", fontsize=10)
        ax.set_xlabel("evaluated on turn j")
        ax.set_ylabel("map fit on turn i")
    fig.suptitle(
        "Cross-turn transfer at MATCHED n=171 (companion; controls the n-per-row "
        "confound across fit turns): held-out $R^2$, layer 19",
        fontsize=11.5,
        y=1.02,
    )
    fig.colorbar(holder[0], ax=axes, fraction=0.046, pad=0.04).set_label(r"held-out $R^2$")
    savefig_paper(fig, f"issue_825/{FIG_T_MATCHED}", dir=str(fig_dir.parent))
    plt.close(fig)

    # merge our captions into savefig_paper's per-stem sidecars (never clobber
    # its per-point `points` / commit / text payload the dashboard reads)
    captions = {
        FIG_T: (
            "PRIMARY. Cross-turn transfer matrix T[i,j] at FULL per-turn n: held-out "
            "R2 of the layer-19 linear context->answer map (context_k -> "
            "answer_k_t1, offset 0) FIT on turn i and EVALUATED on turn j, under one "
            "shared conversation-grouped 6-fold partition, both models. "
            "y=fit-on-turn-i, x=evaluated-on-turn-j; diverging cmap centered at 0. "
            "Diagonal = the turn's own grouped-CV R2. n per turn = 497/388/354/332/"
            "236/171 for turns 1/3/5/7/9/11."
        ),
        FIG_C: (
            "PRIMARY. Operator similarity matrix C[i,j] at FULL per-turn n: cosine "
            "between the layer-19 per-turn standardized-space ridge operators, both "
            "models; diverging cmap centered at 0. Diagonal cells annotated with the "
            "within-turn split-half cosine ceiling (parentheses; half sizes "
            "~n/2 per turn). Shuffled-pairing cosine nulls are in the results JSON."
        ),
        FIG_T_MATCHED: (
            "SECONDARY companion. Cross-turn transfer matrix T[i,j] at MATCHED "
            "n=171 (every turn subsampled to 171, one draw), controlling the "
            "n-per-row confound across fit turns; layer 19, both models; same recipe "
            "as the full-n primary. y=fit-on-turn-i, x=evaluated-on-turn-j."
        ),
    }
    for stem, caption in captions.items():
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
    print(f"[write] {FIG_T}.png + {FIG_C}.png + {FIG_T_MATCHED}.png (+pdf +meta.json)")


if __name__ == "__main__":
    main()
