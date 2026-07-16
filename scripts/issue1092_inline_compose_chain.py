#!/usr/bin/env python3
"""#1092 inline COMPOSITION-CHAIN test: (v_P, v_q_bare) -> v_C -> answer.

The TRUE composition test for "can we compose the prefix map with the query map
to get the context map?" — route through the intermediate CONTEXT state instead
of the additive ANSWER stitch (`_registered_stitch_reads.stitch_prefix_plus_bare`).

Cell cell_inst_own, layer 14, battery-EXCLUDED fit rows (stratum != trait_stratum
AND not is_eval_only), grouped 6-fold novel-prefix folds (group_key=prefix_id,
FOLD_SEED=0) — the parent read1 protocol.

STAGE 1 (basis-independent) — h: (v_P, v_q_bare) -> v_C, four forms, held-out
R2(v̂_C, v_C) variance-weighted (ambient + a v_C-pca companion):
  (a) ADDITIVE (state):  v̂_C = v_P + ridge(v_q_bare -> v_C - v_P)
  (a2) ADDITIVE (mean-offset companion): v̂_C = v_P + per-query mean residual
  (b) JOINT LINEAR:      v̂_C = ridge([v_P; v_q_bare] -> v_C)   (additive, NO interaction)
  (c) QUERY-CONDITIONED OPERATOR: v̂_C = v_P + U diag(s(v_q)) Vᵀ v_P_n   (rank {8,32,64},
      batched AdamW, early-stopped on an inner val fold — the ONLY form that can
      capture the prefix-query INTERACTION)
  identity floor: v̂_C = v_P.

STAGE 2 — chain through the context->answer map. Per train fold, M' = PRESS-ridge
(true v_C -> pooled answer target); score M'(v̂_C_heldout) end-to-end vs true
answer on held-out rows, per h-form (OUT-OF-FOLD: v̂_C and M' are both trained on
the fold's train rows only). Answer target = pooled t1/t2/t3, scored on pca48
(headline, comparable to banked stitch 0.833 / full-context 0.910) + ambient.

BASELINES (same folds): additive answer stitch (re-fit), full-context M'(v_C true)
= ceiling, prefix-only, bare-query-only direct answer maps.

Fit engine REUSED verbatim from the banked read1 path: press_fit_predict (#923
PRESS-LOO ridge), _folds_from_manifest, _basis_targets_with_info, _r2.

Provenance: teacher-forced state capture; own-policy greedy answers; battery-excluded.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/eps-data/thomasjiralerspong/.hf_i1092_operator")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# thread caps + .env must bind BEFORE torch import (pools freeze at import).
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

from issue658_fit_predictors import RIDGE_LAMBDAS  # noqa: E402
from issue923_fit_decomposition import press_fit_predict  # noqa: E402  (equivalence check only)
from issue1092_fit_grid import (  # noqa: E402
    _basis_targets_with_info,
    _folds_from_manifest,
    _r2,
)

STAGE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing"
)
SUMM = STAGE / "analysis_tensors/summaries"
MANIFEST = STAGE / "corpus/manifest.jsonl"
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_compose_chain"
CKPT = OUT / "_stage1_ckpt"

CELL = "cell_inst_own"
MODEL_TYPE = "instruct"  # bare_{model_type} store for cell_inst_own
LAYER = 14
HIDDEN_DIM = 3584
TARGETS = ["t1", "t2", "t3"]
N_FOLDS = 6
FOLD_SEED = 0  # matches issue1092_fit_grid._folds_from_manifest
SEED = 0
BASES = ["pca48", "ambient"]
DEFAULT_RANKS = [8, 32, 64]
VC_PCA_K = 48  # v_C companion basis dim for stage-1 reduced R2

# Banked read3 references (battery-INCLUDED n=19708, cell_inst_own L14, prefix folds).
BANKED = {
    "pca48": {
        "stitch": 0.8328403668925877,
        "full_context": 0.9103453689185139,
        "prefix_only": 0.09601551229080851,
        "query_only": 0.14645461464786902,
    },
    "ambient": {
        "stitch": 0.7036500032012354,
        "full_context": 0.8043440222361293,
        "prefix_only": 0.06507728422851033,
        "query_only": -0.50191609916945,
    },
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_cell_summary(cell: str, kind: str) -> np.ndarray:
    d = SUMM / cell
    p = d / f"{kind}_L{LAYER:02d}.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    shards = sorted(d.glob(f"{kind}_L{LAYER:02d}_shard*.npy"))
    if not shards:
        raise FileNotFoundError(f"{d}/{kind}_L{LAYER:02d}[.npy|_shard*.npy]")
    return np.concatenate([np.load(s) for s in shards], axis=0)


def _load_bare(model_type: str) -> tuple[np.ndarray, dict[str, int]]:
    """bare_{model_type}/c_q_bare_L{LAYER} state + query_id -> row index."""
    root = SUMM / f"bare_{model_type}"
    p = root / f"c_q_bare_L{LAYER:02d}.npy"
    if p.exists():
        arr = np.load(p)
    else:
        shards = sorted(root.glob(f"c_q_bare_L{LAYER:02d}_shard*.npy"))
        if not shards:
            raise FileNotFoundError(f"{root}/c_q_bare_L{LAYER:02d}[.npy|_shard*.npy]")
        arr = np.concatenate([np.load(s) for s in shards], axis=0)
    idx_rows: list[dict] = []
    ri = root / "row_index.jsonl"
    if ri.exists():
        idx_rows = _jsonl(ri)
    else:
        for s in sorted(root.glob("row_index_shard*.jsonl")):
            idx_rows += _jsonl(s)
    if len(idx_rows) != arr.shape[0]:
        raise ValueError(f"bare row_index {len(idx_rows)} != rows {arr.shape[0]}")
    q2i = {str(r["query_id"]): i for i, r in enumerate(idx_rows)}
    return arr, q2i


# ---------------------------------------------------------------------------
# Lean PRESS-LOO ridge — EXACT re-expression of PressRidge/press_fit_predict
# (standardize=True, RIDGE_LAMBDAS grid, PRESS-LOO λ selection, dual-form
# prediction), WITHOUT PressRidge's (n_lambda, m, k) sU tensor + (n_lambda, k, k)
# K bmm. Those made the ambient/joint 7168-wide designs cost ~5 GB RSS + a
# ~4.5e12-flop bmm per engine (built for d=48/96 PCA designs, not ambient H).
# Per-λ PRESS-MSE via the cheaper of two exact contractions (DIRECT held-out
# hat when P<=k; the P-independent K(λ) expansion when P>k). Verified against
# press_fit_predict by --verify-engine (max |Δpred|, |Δmse| ~ 1e-9).
# ---------------------------------------------------------------------------
def _press_loo_mse(U, S2, G, Yc, U2, m, P):
    """(mse_per_lambda (n_lambda,)) matching PressRidge.press_mse exactly."""
    k = U.shape[1]
    lambdas = RIDGE_LAMBDAS
    mse = torch.empty(len(lambdas), dtype=U.dtype)
    if k >= P:  # DIRECT: mean over (i,p) of [(Yc - U(φ⊙G))/(1-h)]²
        for li, lam in enumerate(lambdas):
            phi = S2 / (S2 + lam)  # (k,)
            yhat = U @ (phi[:, None] * G)  # (m, P)
            h = U2 @ phi  # (m,)
            w = 1.0 / (1.0 - h).clamp(min=1e-8)
            r = (Yc - yhat) * w[:, None]
            mse[li] = (r * r).sum() / (m * P)
    else:  # K-trick: P-independent expansion (term_a + term_b + term_c)
        Yperp = Yc - U @ G  # (m, P)
        a = (Yperp * Yperp).sum(1)  # (m,)
        V = Yperp @ G.T  # (m, k)
        C = G @ G.T  # (k, k)
        for li, lam in enumerate(lambdas):
            phi = S2 / (S2 + lam)
            h = U2 @ phi
            s = (1.0 / (1.0 - h).clamp(min=1e-8)) ** 2  # (m,)
            omp = 1.0 - phi  # (k,)
            term_a = float(a @ s)
            P1 = ((s[:, None] * U) * V).sum(0)  # (k,)
            term_b = 2.0 * float((omp * P1).sum())
            Ksel = U.T @ (s[:, None] * U)  # (k, k)
            term_c = float((torch.outer(omp, omp) * C * Ksel).sum())
            mse[li] = (term_a + term_b + term_c) / (m * P)
    return mse


def _fit_predictor(Xtr: np.ndarray, Ytr: np.ndarray):
    """Fit PRESS-LOO ridge on train (standardize=True); return (predict, lam_idx).

    predict(Xnew) applies to ARBITRARY new X — the banked chain protocol
    (issue1092_fit_grid.fit_train_predictor), so M'(v̂_C) is out-of-fold."""
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float64))
    Yt = torch.from_numpy(np.ascontiguousarray(Ytr, dtype=np.float64))
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9
    keep = sd > (sd.max() * 1e-6 + 1e-12)  # §8 degenerate-dim drop
    Xn = ((Xt - mu) / sd)[:, keep]
    ymu = Yt.mean(0, keepdim=True)
    Yc = Yt - ymu
    U, S, Vh = torch.linalg.svd(Xn, full_matrices=False)  # U(m,k) S(k) Vh(k,d')
    G = U.T @ Yc  # (k, P)
    S2 = S * S
    U2 = U * U
    m, P = Xn.shape[0], Yc.shape[1]
    mse = _press_loo_mse(U, S2, G, Yc, U2, m, P)
    lam_idx = int(torch.argmin(mse).item())
    coef = (S / (S2 + RIDGE_LAMBDAS[lam_idx]))[:, None] * G  # (k, P)

    def predict(Xnew: np.ndarray) -> np.ndarray:
        Xnn = torch.from_numpy(np.ascontiguousarray(Xnew, dtype=np.float64))
        Xnn_n = ((Xnn - mu) / sd)[:, keep]
        with torch.no_grad():
            p = (Xnn_n @ Vh.T) @ coef + ymu
        return p.numpy()

    return predict, lam_idx


def _cv_oof(X: np.ndarray, Y: np.ndarray, folds: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Standard grouped-CV OOF prediction (fit train, predict held-out test)."""
    n = X.shape[0]
    oof = np.zeros_like(Y, dtype=np.float64)
    lams: list[int] = []
    for test_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        predict, li = _fit_predictor(X[mask], Y[mask])
        oof[test_idx] = predict(X[test_idx])
        lams.append(li)
    return oof, lams


# ---------------------------------------------------------------------------
# Query-conditioned low-rank operator (FiLM gain+bias form):
#   v_hat_C = v_P
#           + g_additive(v_q)          [FROZEN closed-form additive base = query main
#                                        effect, ~79% of the residual variance]
#           + U diag(s(v_q)) V^T v_P_n [LEARNED rank-r prefix-by-query INTERACTION]
# The frozen additive base makes the operator a proper SUPERSET of the additive
# form (interaction=0 -> operator == additive), so held-out
# R2(operator) - R2(additive) isolates the interaction the linear/additive forms
# structurally cannot capture.
# ---------------------------------------------------------------------------
class QueryConditionedOperator(torch.nn.Module):
    def __init__(self, d_p: int, d_q: int, h_out: int, rank: int):
        super().__init__()
        self.V = torch.nn.Parameter(torch.randn(d_p, rank) / (d_p**0.5))
        self.Ws = torch.nn.Parameter(torch.randn(rank, d_q) / (d_q**0.5))
        self.bs = torch.nn.Parameter(torch.zeros(rank))
        # init U small so correction ≈ 0 at step 0 (start at the identity floor).
        self.U = torch.nn.Parameter(0.01 * torch.randn(h_out, rank) / (rank**0.5))

    def correction(self, vp_n: torch.Tensor, vq_n: torch.Tensor) -> torch.Tensor:
        p = vp_n @ self.V  # (n, r)
        s = vq_n @ self.Ws.T + self.bs  # (n, r)
        return (s * p) @ self.U.T  # (n, h_out)


def _std_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    keep = sd > (sd.max() * 1e-6 + 1e-12)
    return mu, sd, keep


def _fit_operator(
    vp_tr: np.ndarray,
    vq_tr: np.ndarray,
    vc_tr: np.ndarray,
    g_tr: np.ndarray,
    vp_te: np.ndarray,
    vq_te: np.ndarray,
    g_te: np.ndarray,
    inner_val: np.ndarray,
    rank: int,
    *,
    max_steps: int,
    lr: float,
    wd: float,
    patience: int,
) -> tuple[np.ndarray, dict]:
    """AdamW-fit the INTERACTION on top of the frozen additive base g; early-stop
    on inner-val R2(v̂_C, v_C) where v̂_C = v_P + g + interaction; return held-out
    (test) v̂_C at the best-val step + fit metadata.

    g_tr / g_te are the closed-form additive-state ridge's predicted residual
    (v_q_bare -> v_C - v_P) on the outer-train / outer-test rows (fit on outer
    train). The interaction target is (v_C - v_P - g) — what additivity misses."""
    fit_mask = np.ones(vp_tr.shape[0], dtype=bool)
    fit_mask[inner_val] = False
    # standardization stats from the FIT subset only (no val/test leakage).
    muP, sdP, keepP = _std_stats(vp_tr[fit_mask])
    muQ, sdQ, keepQ = _std_stats(vq_tr[fit_mask])

    def _norm(X, mu, sd, keep):
        return torch.from_numpy(np.ascontiguousarray(((X - mu) / sd)[:, keep], dtype=np.float32))

    vpn_fit = _norm(vp_tr[fit_mask], muP, sdP, keepP)
    vqn_fit = _norm(vq_tr[fit_mask], muQ, sdQ, keepQ)
    # interaction target = additive residual (what v_P + g leaves unexplained)
    I_fit = torch.from_numpy(
        np.ascontiguousarray(vc_tr[fit_mask] - vp_tr[fit_mask] - g_tr[fit_mask], dtype=np.float32)
    )
    vpn_val = _norm(vp_tr[inner_val], muP, sdP, keepP)
    vqn_val = _norm(vq_tr[inner_val], muQ, sdQ, keepQ)
    vc_val = vc_tr[inner_val]
    base_val = vp_tr[inner_val] + g_tr[inner_val]  # additive prediction on val

    torch.manual_seed(SEED)
    model = QueryConditionedOperator(int(keepP.sum()), int(keepQ.sum()), I_fit.shape[1], rank)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, best_step, since = -np.inf, None, -1, 0
    ss_tot_val = float(((vc_val - vc_val.mean(0, keepdims=True)) ** 2).sum())
    for step in range(max_steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        c = model.correction(vpn_fit, vqn_fit)
        loss = ((c - I_fit) ** 2).mean()
        loss.backward()
        opt.step()
        if step % 5 == 0 or step == max_steps - 1:
            model.eval()
            with torch.no_grad():
                c_val = model.correction(vpn_val, vqn_val).numpy().astype(np.float64)
            vhat_val = base_val + c_val
            ss_res = float(((vc_val - vhat_val) ** 2).sum())
            val_r2 = 1.0 - ss_res / ss_tot_val if ss_tot_val > 0 else float("nan")
            if val_r2 > best_val:
                best_val, best_step, since = val_r2, step, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
            if since * 5 >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    # predict held-out test v̂_C = v_P + g_additive + interaction, RAW ambient units.
    vpn_te = _norm(vp_te, muP, sdP, keepP)
    vqn_te = _norm(vq_te, muQ, sdQ, keepQ)
    model.eval()
    with torch.no_grad():
        c_te = model.correction(vpn_te, vqn_te).numpy().astype(np.float64)
    vhat = vp_te + g_te + c_te
    meta = {
        "rank": rank,
        "best_val_r2": float(best_val),
        "best_step": int(best_step),
        "n_fit": int(fit_mask.sum()),
        "n_val": int(inner_val.size),
        "d_keep_p": int(keepP.sum()),
        "d_keep_q": int(keepQ.sum()),
        "form": "v_P + frozen additive g(v_q) + learned rank-r interaction",
    }
    return vhat, meta


def _grouped_inner_val(train_idx: np.ndarray, prefix_ids: np.ndarray) -> np.ndarray:
    """Deterministic grouped inner-val split: hold out ~1/N_FOLDS of TRAIN prefixes.

    Returns LOCAL indices into train_idx (positions), grouped by prefix so the
    operator's early-stopping val is novel-prefix like the outer folds."""
    tr_pref = prefix_ids[train_idx]
    uniq = sorted(set(tr_pref.tolist()))
    rng = np.random.default_rng(FOLD_SEED)
    rng.shuffle(uniq)
    val_groups = set(uniq[0::N_FOLDS])  # ~1/6 of train prefixes
    return np.array([i for i, p in enumerate(tr_pref) if p in val_groups], dtype=np.int64)


def _residual_geometry(
    vc: np.ndarray,
    vhat_add: np.ndarray,
    prefix_ids: np.ndarray,
    query_ids: np.ndarray,
    dense_local: np.ndarray,
) -> dict:
    """Read 3: is the additive form's stage-1 error concentrated in the SAME
    directions as the FGI interaction component (dense-core)?"""
    if dense_local.size < 50:
        return {"status": "insufficient_dense_core", "n": int(dense_local.size)}
    vc_d = vc[dense_local]
    resid = vc_d - vhat_add[dense_local]  # additive stage-1 error on dense-core
    # FGI interaction component i = yc - f(prefix) - g(query).
    yc = vc_d - vc_d.mean(0, keepdims=True)
    f = np.zeros_like(yc)
    g = np.zeros_like(yc)
    pid = prefix_ids[dense_local]
    qid = query_ids[dense_local]
    for p in np.unique(pid):
        f[pid == p] = yc[pid == p].mean(0, keepdims=True)
    for q in np.unique(qid):
        g[qid == q] = yc[qid == q].mean(0, keepdims=True)
    inter = yc - f - g
    # per-row energy correlation
    e_resid = (resid * resid).sum(1)
    e_inter = (inter * inter).sum(1)
    er = e_resid - e_resid.mean()
    ei = e_inter - e_inter.mean()
    denom = float(np.linalg.norm(er) * np.linalg.norm(ei))
    energy_r = float(np.dot(er, ei) / denom) if denom > 0 else float("nan")
    # subspace overlap: top-16 PCs of resid vs interaction (mean principal-angle cos)
    k = 16
    _, _, vhr = np.linalg.svd(resid - resid.mean(0, keepdims=True), full_matrices=False)
    _, _, vhi = np.linalg.svd(inter - inter.mean(0, keepdims=True), full_matrices=False)
    kk = min(k, vhr.shape[0], vhi.shape[0])
    cos = np.linalg.svd(vhr[:kk] @ vhi[:kk].T, compute_uv=False)
    cos = np.clip(cos, -1.0, 1.0)
    return {
        "status": "computed",
        "n_dense_core": int(dense_local.size),
        "share_interaction_densecore": float((inter * inter).sum() / (yc * yc).sum()),
        "resid_norm_over_vc_centered": float(
            np.linalg.norm(resid) / max(np.linalg.norm(yc), 1e-12)
        ),
        "per_row_energy_pearson_resid_vs_interaction": energy_r,
        "top16_subspace_principal_angle_cos_mean": float(cos.mean()),
        "top16_subspace_principal_angle_cos_median": float(np.median(cos)),
        "interpretation": (
            "high energy_pearson + high cos => the additive form's error IS the "
            "prefix-query interaction the operator form is meant to capture"
        ),
    }


def _verify_engine() -> dict:
    """Assert the lean PRESS-LOO fit reproduces press_fit_predict (both λ-branches)."""
    rng = np.random.default_rng(0)
    out = {}
    for tag, (m, d, P) in {"P_le_k": (400, 200, 30), "P_gt_k": (400, 120, 300)}.items():
        X = rng.standard_normal((m, d))
        W = rng.standard_normal((d, P))
        Y = X @ W + 0.1 * rng.standard_normal((m, P))
        Xte = rng.standard_normal((60, d))
        ref = press_fit_predict(
            torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xte), standardize=True
        )
        predict, lam_idx = _fit_predictor(X, Y)
        pred = predict(Xte)
        dpred = float(np.abs(pred - ref["pred"].numpy()).max())
        out[tag] = {
            "lam_match": lam_idx == int(ref["lam_idx"]),
            "max_abs_dpred": dpred,
            "k": min(m, int((np.std(X, 0) > 0).sum())),
        }
        if lam_idx != int(ref["lam_idx"]) or dpred > 1e-6:
            raise AssertionError(
                f"engine mismatch [{tag}]: lam {lam_idx} vs {ref['lam_idx']}, dpred {dpred:.2e}"
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gate",
        action="store_true",
        help="run outer fold 0 only (all stage-1 forms + pca48 M'), project xN_FOLDS, exit",
    )
    ap.add_argument("--ranks", default=None, help="comma list, default 8,32,64")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--op-lr", type=float, default=3e-3)
    ap.add_argument("--op-wd", type=float, default=1e-3)
    ap.add_argument("--op-patience", type=int, default=60)
    ap.add_argument("--max-wall-h", type=float, default=3.0)
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")] if args.ranks else DEFAULT_RANKS

    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    engine_check = _verify_engine()
    print(f"[verify-engine] lean PRESS-LOO == press_fit_predict: {engine_check}", flush=True)

    rows = _jsonl(MANIFEST)
    prefix_all = _load_cell_summary(CELL, "prefix_end")
    context_all = _load_cell_summary(CELL, "context_end")
    t_all = [_load_cell_summary(CELL, t) for t in TARGETS]
    bare_arr, q2i = _load_bare(MODEL_TYPE)
    n0 = min(prefix_all.shape[0], context_all.shape[0], min(t.shape[0] for t in t_all), len(rows))
    be_idx = np.asarray(
        [
            i
            for i in range(n0)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )
    unit_rows = [rows[int(i)] for i in be_idx]
    prefix_ids = np.asarray([r.get("prefix_id", "") for r in unit_rows])
    query_ids = np.asarray([str(r.get("query_id", "")) for r in unit_rows])
    dense_local = np.asarray(
        [j for j, r in enumerate(unit_rows) if r.get("stratum") == "dense_core"], dtype=np.int64
    )
    missing = [q for q in set(query_ids.tolist()) if q not in q2i]
    if missing:
        raise KeyError(f"bare-query arm missing {len(missing)} query ids e.g. {missing[:5]}")

    # states (float32 storage; cast to fp64 per fold for the ridge engine)
    v_P = np.asarray(prefix_all[be_idx], dtype=np.float32)
    v_C = np.asarray(context_all[be_idx], dtype=np.float32)
    v_q = np.asarray(bare_arr[[q2i[q] for q in query_ids]], dtype=np.float32)
    Y_stacked = np.concatenate([np.asarray(t[be_idx], dtype=np.float32) for t in t_all], axis=1)
    del prefix_all, context_all, t_all, bare_arr
    gc.collect()
    n_be = be_idx.size
    print(
        f"n_be={n_be} H={HIDDEN_DIM} dense_core={dense_local.size} "
        f"n_prefix={len(set(prefix_ids))} n_query={len(set(query_ids))}",
        flush=True,
    )

    folds = _folds_from_manifest(unit_rows, len(unit_rows), group_key="prefix_id", n_folds=N_FOLDS)
    if args.gate:
        folds = folds[:1]
    print(f"folds: {[f.size for f in folds]}", flush=True)

    # ---- STAGE 1: OOF v̂_C per form (basis-independent) ----
    forms = ["identity", "additive_state", "additive_meanoffset", "joint_linear"]
    forms += [f"operator_r{r}" for r in ranks]
    vhat = {f: np.zeros((n_be, HIDDEN_DIM), dtype=np.float32) for f in forms}
    op_meta: dict[str, list] = {f"operator_r{r}": [] for r in ranks}
    s1_lam = {"additive_state": [], "joint_linear": []}
    fold_wall = []

    for fi, test_idx in enumerate(folds):
        ck = CKPT / f"fold{fi}_gate{int(args.gate)}.npz"
        if ck.exists():
            z = np.load(ck)
            for f in forms:
                vhat[f][test_idx] = z[f]
            print(f"[fold {fi}] loaded checkpoint", flush=True)
            fold_wall.append(0.0)
            continue
        t0 = time.monotonic()
        mask = np.ones(n_be, dtype=bool)
        mask[test_idx] = False
        vP_tr, vP_te = v_P[mask].astype(np.float64), v_P[test_idx].astype(np.float64)
        vq_tr, vq_te = v_q[mask].astype(np.float64), v_q[test_idx].astype(np.float64)
        vC_tr = v_C[mask].astype(np.float64)

        # identity floor
        vhat["identity"][test_idx] = vP_te
        # (a) additive-state: v_P + ridge(v_q_bare -> v_C - v_P). Its predicted
        # residual g is the FROZEN additive base the operator (c) builds on.
        pred_a, la = _fit_predictor(vq_tr, vC_tr - vP_tr)
        g_tr = pred_a(vq_tr)  # additive residual prediction on outer-train
        g_te = pred_a(vq_te)  # ... and on held-out test
        vhat["additive_state"][test_idx] = vP_te + g_te
        s1_lam["additive_state"].append(la)
        # (a2) additive-mean-offset: per-query mean residual from train
        resid_tr = vC_tr - vP_tr
        qtr = query_ids[mask]
        qmean: dict[str, np.ndarray] = {}
        for q in np.unique(qtr):
            qmean[q] = resid_tr[qtr == q].mean(0)
        global_off = resid_tr.mean(0)
        off_te = np.stack([qmean.get(query_ids[i], global_off) for i in test_idx], axis=0)
        vhat["additive_meanoffset"][test_idx] = vP_te + off_te
        # (b) joint linear: ridge([v_P; v_q_bare] -> v_C)
        pred_b, lb = _fit_predictor(np.concatenate([vP_tr, vq_tr], axis=1), vC_tr)
        vhat["joint_linear"][test_idx] = pred_b(np.concatenate([vP_te, vq_te], axis=1))
        s1_lam["joint_linear"].append(lb)
        # (c) operator forms
        inner_val = _grouped_inner_val(np.nonzero(mask)[0], prefix_ids)
        for r in ranks:
            vh_op, meta = _fit_operator(
                vP_tr,
                vq_tr,
                vC_tr,
                g_tr,
                vP_te,
                vq_te,
                g_te,
                inner_val,
                r,
                max_steps=args.max_steps,
                lr=args.op_lr,
                wd=args.op_wd,
                patience=args.op_patience,
            )
            vhat[f"operator_r{r}"][test_idx] = vh_op
            op_meta[f"operator_r{r}"].append(meta)
        np.savez(ck, **{f: vhat[f][test_idx] for f in forms})
        dt = time.monotonic() - t0
        fold_wall.append(dt)
        print(
            f"[fold {fi}] stage-1 done in {dt:.0f}s "
            f"(op best_val r2: {[op_meta[f'operator_r{r}'][-1]['best_val_r2'] for r in ranks]})",
            flush=True,
        )
        if fi == 0:
            proj = (
                dt
                * len(
                    _folds_from_manifest(
                        unit_rows, len(unit_rows), group_key="prefix_id", n_folds=N_FOLDS
                    )
                )
                / 3600.0
            )
            print(
                f"[gate] fold-0 {dt:.0f}s . {N_FOLDS} folds -> stage-1 proj {proj:.2f}h "
                f"(+ stage-2 baselines ~0.3h); cap {args.max_wall_h}h",
                flush=True,
            )
            if proj > args.max_wall_h:
                (OUT / "compose_chain.json").write_text(
                    json.dumps(
                        {
                            "ABORT": f"stage-1 projected {proj:.2f}h > {args.max_wall_h}h",
                            "fold0_wall_s": dt,
                        },
                        indent=2,
                    )
                )
                print(f"[gate] ABORT projected {proj:.2f}h > {args.max_wall_h}h", flush=True)
                return

    stage1_r2 = _score_stage1(vhat, v_C, folds, forms)

    if args.gate:
        # also run pca48 stage-2 chain on the single fold to time it
        gate_out = {
            "gate": True,
            "fold0_stage1_wall_s": fold_wall[0],
            "stage1_r2_ambient_fold0": stage1_r2,
            "op_meta": op_meta,
        }
        t2 = time.monotonic()
        _stage2_all(vhat, v_P, v_q, v_C, Y_stacked, folds, forms, ranks, bases=["pca48"])
        gate_out["fold0_stage2_pca48_wall_s"] = time.monotonic() - t2
        (OUT / "compose_chain_gate.json").write_text(json.dumps(gate_out, indent=2, allow_nan=True))
        s2w = gate_out["fold0_stage2_pca48_wall_s"]
        print(f"[gate] wrote gate JSON; stage2-pca48 fold0 {s2w:.0f}s", flush=True)
        return

    # ---- STAGE 2: chain + baselines, per basis ----
    stage2 = _stage2_all(vhat, v_P, v_q, v_C, Y_stacked, folds, forms, ranks, bases=BASES)

    # ---- Read 3: residual geometry (additive-state form) ----
    geom = _residual_geometry(
        v_C.astype(np.float64),
        vhat["additive_state"].astype(np.float64),
        prefix_ids,
        query_ids,
        dense_local,
    )

    result = {
        "meta": {
            "script": "scripts/issue1092_inline_compose_chain.py",
            "git_commit": _git_sha(),
            "generated_utc": datetime.now(UTC).isoformat(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cell": CELL,
            "layer": LAYER,
            "n_be": int(n_be),
            "n_dense_core": int(dense_local.size),
            "n_prefix": len(set(prefix_ids.tolist())),
            "n_query": len(set(query_ids.tolist())),
            "folds": f"grouped {N_FOLDS}-fold novel-prefix (group_key=prefix_id, FOLD_SEED=0)",
            "fit_rows": "battery-EXCLUDED: stratum != trait_stratum AND not is_eval_only",
            "answer_target": "pooled t1/t2/t3 (stacked); pca48 = top-48 PCs of stacked (fit once)",
            "engine": "press_fit_predict (#923 PRESS-LOO ridge), reused verbatim",
            "operator": {
                "ranks": ranks,
                "optimizer": "AdamW",
                "lr": args.op_lr,
                "weight_decay": args.op_wd,
                "max_steps": args.max_steps,
                "patience": args.op_patience,
                "form": "v̂_C = v_P + g_additive(v_q) [FROZEN closed-form ridge, query "
                "main effect] + U diag(s(v_q)) Vᵀ v_P_n [LEARNED rank-r interaction]; "
                "s = Ws v_q_n + bs. Proper superset of additive (interaction=0 => additive), "
                "so R2(operator) - R2(additive) isolates the prefix-query interaction.",
                "grounding": "exploratory analysis hyperparameters — smoke/val-gated, "
                "not a training recipe; early-stopped on novel-prefix inner val",
            },
            "provenance": "teacher-forced state capture; own-policy greedy answers; "
            "battery-excluded; novel-prefix folds",
            "banked_reference_battery_included_n19708": BANKED,
            "stage1_wall_s_per_fold": fold_wall,
            "engine_equivalence_check": engine_check,
        },
        "stage1_context_reconstruction": stage1_r2,
        "stage2_end_to_end": stage2,
        "read3_residual_geometry": geom,
        "operator_fit_meta": op_meta,
        "stage1_lambda_indices": s1_lam,
    }
    (OUT / "compose_chain.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    print(
        f"wrote {OUT / 'compose_chain.json'} in {time.monotonic() - t_start:.0f}s total", flush=True
    )


def _score_stage1(vhat: dict, v_C: np.ndarray, folds, forms) -> dict:
    """Held-out R2(v̂_C, v_C): ambient (variance-weighted) + v_C-pca48 companion."""
    # aggregate OOF over the folds actually run
    test_all = np.concatenate([f for f in folds])
    vc = v_C[test_all].astype(np.float64)
    # v_C-pca basis fit on the scored rows (companion view only)
    mu = vc.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(vc - mu, full_matrices=False)
    Vk = vh[:VC_PCA_K].T
    vc_p = (vc - mu) @ Vk
    out = {}
    for f in forms:
        vh_f = vhat[f][test_all].astype(np.float64)
        out[f] = {
            "r2_ambient": _r2(vc, vh_f),
            "r2_vcpca48": _r2(vc_p, (vh_f - mu) @ Vk),
        }
    return out


def _r2_from_ss(ss_res: float, ss_tot: float) -> float:
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _stage2_all(vhat, v_P, v_q, v_C, Y_stacked, folds, forms, ranks, *, bases) -> dict:
    """Per basis: chain M'(v̂_C) for every h-form + baselines (stitch, full-context,
    prefix-only, query-only). OUT-OF-FOLD: v̂_C and M' both train-fold-only.

    Memory-lean: accumulates per-form ss_res across folds (never stores a full
    (n_be, P) OOF per form — the ambient P=10752 case would be ~9 GB otherwise).
    ss_tot is over the scored rows with the same global-mean convention as `_r2`.
    """
    n_be = v_P.shape[0]
    test_all = np.concatenate([f for f in folds])
    result: dict = {}
    for basis in bases:
        Yb, _info = _basis_targets_with_info(
            Y_stacked.astype(np.float64),
            basis,
            hidden_dim=HIDDEN_DIM,
            targets=TARGETS,
            projection_target="t1",
        )
        Yb = np.ascontiguousarray(Yb, dtype=np.float64)
        yb_scored = Yb[test_all]
        ss_tot = float(((yb_scored - yb_scored.mean(0, keepdims=True)) ** 2).sum())
        # chain: per fold fit M'(true v_C -> Yb) on train, apply to every v̂_C_form + true v_C
        ss_chain = {f: 0.0 for f in forms}
        ss_full = 0.0
        mprime_lam = []
        for test_idx in folds:
            mask = np.ones(n_be, dtype=bool)
            mask[test_idx] = False
            mprime, lm = _fit_predictor(v_C[mask].astype(np.float64), Yb[mask])
            mprime_lam.append(lm)
            yb_te = Yb[test_idx]
            pred_full = mprime(v_C[test_idx].astype(np.float64))
            ss_full += float(((yb_te - pred_full) ** 2).sum())
            for f in forms:
                pred = mprime(vhat[f][test_idx].astype(np.float64))
                ss_chain[f] += float(((yb_te - pred) ** 2).sum())
        # baselines (direct answer maps on the same folds; score-and-free each)
        stitch_oof, stitch_lam = _cv_oof(
            np.concatenate([v_P, v_q], axis=1).astype(np.float64), Yb, folds
        )
        r2_stitch = _r2(yb_scored, stitch_oof[test_all])
        del stitch_oof
        gc.collect()
        prefix_oof, _ = _cv_oof(v_P.astype(np.float64), Yb, folds)
        r2_prefix = _r2(yb_scored, prefix_oof[test_all])
        del prefix_oof
        gc.collect()
        query_oof, _ = _cv_oof(v_q.astype(np.float64), Yb, folds)
        r2_query = _r2(yb_scored, query_oof[test_all])
        del query_oof
        gc.collect()

        result[basis] = {
            "P_out": int(Yb.shape[1]),
            "chain_through_vhat_C": {f: _r2_from_ss(ss_chain[f], ss_tot) for f in forms},
            "baselines": {
                "full_context_ceiling": _r2_from_ss(ss_full, ss_tot),
                "additive_answer_stitch": r2_stitch,
                "prefix_only": r2_prefix,
                "query_only": r2_query,
            },
            "mprime_lambda_indices": mprime_lam,
            "stitch_lambda_indices": stitch_lam,
            "banked_reference": BANKED.get(basis),
        }
        del Yb
        gc.collect()
    return result


if __name__ == "__main__":
    t = time.monotonic()
    main()
    print(f"done in {time.monotonic() - t:.0f}s", flush=True)
