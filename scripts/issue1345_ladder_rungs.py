#!/usr/bin/env python
"""Issue #1345 inline round — mapping-similarity LADDER rungs, chat <-> no-template.

Fills rungs 2-8 of the mapping-similarity ladder for the r1 (chat) <-> r2
(naturalistic / plain ``User:``-``Assistant:`` text) pair. The shipped
``issue1345_operator_comparison.py`` battery measures ONLY rung 1 (direct
transfer, which fails) and rung 9 (full A.M.B, which saturates); every rung in
between was never computed for this pair, so "which correction is the WEAKEST
one that reconciles the two maps" has been undecidable.

Ladder (strongest shared-structure claim -> weakest), source regime s, target t.
Every correction is fit on the TARGET TRAIN fold and scored on the TARGET TEST
fold; the source operator is always frozen (fit on the SOURCE train fold):

  1 direct       pred = W_s x_t + b_s
  2 ctx_offset   pred = W_s (x_t - dx) + b_s   dx = mean(Xt_tr) - mean(Xs_tr)
  3 ans_offset   pred = W_s x_t + b_s + dy     dy = mean(Yt_tr) - mean(Ys_tr)
  4 bias_refit   pred = W_s x_t + b*           b* = mean(Yt_tr - P_tr)
  5 global_scale pred = a * (W_s x_t) + b*     a scalar, closed form
  6 rotation     pred = (W_s x_t) R + b*       R orthogonal (Procrustes)
  7 ctx_reparam  pred = W_s (A x_t) + b*       A ridge Xt -> Xs
  8 ans_reparam  pred = B(W_s x_t) + b*        B ridge Ys -> Yt
  9 full_AMB     pred = B(W_s (A x_t)) + b*

Rungs 2-4 are exactly the identity-family / learned-bias baseline the standing
mapping rule requires (rung 1 = identity, rung 4 = identity + learned bias);
the kNN-retrieval read the same rule requires is reported alongside.

A and B are fit on the CLOUDS (contexts-to-contexts, source-answers-to-target-
answers), never on the map's own predictions — fitting B on predictions would
let it regress away the frozen operator's error rather than a change of answer
coordinates, inflating rungs 8/9. This matches the shipped chain, whose A_ans
stage uses ``preps["Yb"]`` (the SOURCE-answer prep).

VECTORIZATION (this is the point of the round — it replaces a battery that
would otherwise be ~160 dense 3584x3584 SVDs, several CPU-hours):

  * DUAL FORM THROUGHOUT. The primal d x d operator W_s is NEVER materialized.
    Applying the frozen source map to target contexts is one cross-Gram
    ``(Xn_t @ Xn_s_tr^T) @ alpha`` — (n_ev x n_tr) @ (n_tr x d) — instead of an
    (n x d) @ (d x d) product against a matrix that costs 103 MB to hold.
  * BATCHED OVER LAYERS. Every eigh / QR / SVD / matmul runs on a leading layer
    axis: ``eigh`` on (L, ntr, ntr), ``qr`` on (L, d, ntr), ``svd`` on
    (L, ntr, ntr). No Python loop over layers anywhere in the numerics.
  * EXACT THIN-QR PROCRUSTES (rung 6). The dense route needs an SVD of
    M = Pc_tr^T Yc_tr, a d x d = 3584 x 3584 matrix, per (fold, direction,
    layer, model, arm) = 160 SVDs. But rank(M) <= n_tr, and — critically —
    every prediction row Pc_te lies in rowspace(Pc_tr) (P = Xn @ W_s and
    rowspace(W_s) is spanned by the n_tr dual coefficient rows, which Pc_tr
    also spans), so the arbitrary completion of R on the orthogonal complement
    can never affect a prediction. Factor Pc_tr^T = Q1 R1 and Yc_tr^T = Q2 R2
    (thin QR, d x ntr), SVD the ntr x ntr core C = R1 R2^T = Uc Sc Vc^T, and
    predict via ((Pc_te @ Q1 Uc) @ (Q2 Vc)^T). The containment premise is
    ASSERTED numerically per fold (``proj_residual``), never assumed.
  * PER-FOLD PREPS SHARED ACROSS BOTH DIRECTIONS. The four Gram
    eigendecompositions a fold needs (X and Y for each regime) are computed
    ONCE and reused by all 9 rungs in BOTH directions — 4 eigh per fold rather
    than 6, and never one per rung.

Numerical recipe is the committed one: per-feature train standardization,
Gram eigh, GCV lambda over ``cm.LAMBDAS`` with the ``GCV_DOF_CAP``
interpolating-lambda skip, conversation-grouped folds, pooled held-out R^2.
``--parity`` checks this batched implementation against the committed
``issue825_map_alignment`` helpers on the same rows (must agree to ``--tol``).

Reads the slim layer-sliced cache written by ``issue1345_ladder_extract.py``
(full n = 4,724; the raw stores are 87 GB and would breach the 50 GB
VM-analysis cap).

Outputs: eval_results/issue_1345/ladder_rungs/ladder_rungs_{model}_{arm}.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# load_dotenv() BEFORE torch: torch freezes its intra-op thread pool from
# OMP_NUM_THREADS at import, so the shared-VM thread caps (#847) only bind
# in-process if the env is populated first.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue1345_common as c  # noqa: E402

LAMBDAS = torch.as_tensor(np.asarray(cm.LAMBDAS), dtype=torch.float64)
GCV_DOF_CAP = getattr(cm, "GCV_DOF_CAP", None)
N_FOLDS = cm.N_FOLDS
HEADLINE_LAYER = 19
REGIMES = ("r1", "r2")
DIRECTIONS = (("r1", "r2"), ("r2", "r1"))
DIR_KEY = {("r1", "r2"): "chat->no_template", ("r2", "r1"): "no_template->chat"}

RUNGS = (
    "1_direct",
    "2_ctx_offset",
    "3_ans_offset",
    "4_bias_refit",
    "5_global_scale",
    "6_rotation",
    "7_ctx_reparam",
    "8_ans_reparam",
    "9_full_AMB",
)


def load_cache(
    cache_dir: Path, model: str, regime: str, cached_layers: list[int], want: list[int]
) -> dict:
    """Load the slim layer-sliced store and keep only the `want` layers."""
    stem = f"{model}_{c.REGIME_FORMAT[regime]}_{c.TRACK}"
    path = cache_dir / f"{stem}_L{'-'.join(map(str, cached_layers))}.pt"
    assert path.exists(), f"missing cache {path} — run issue1345_ladder_extract.py first"
    d = torch.load(path, map_location="cpu", weights_only=False)
    idx = torch.as_tensor([d["layers"].index(x) for x in want])
    return {
        "conv_ids": np.asarray(d["conv_ids"]),
        "slots": d["slots"].index_select(2, idx),
        "profiles": d["profiles"].index_select(2, idx),
    }


def arm_xy(store: dict, regime: str, arm: str, keep: np.ndarray) -> dict:
    """(X, Y) for one (regime, arm), restricted+reordered to `keep` conv_ids.

    X = slots[:, ARM_SLOT_INDEX[arm]] ; Y = profiles[:, TARGET_TURN_INDEX[regime]]
    Returned shape is (L, n, d) — layer-major, so every downstream op batches
    over the leading layer axis.
    """
    pos = {cid: i for i, cid in enumerate(store["conv_ids"])}
    missing = [k for k in keep if k not in pos]
    assert not missing, f"{len(missing)} keep-ids absent from store (e.g. {missing[:3]})"
    idx = torch.as_tensor([pos[k] for k in keep])
    si = c.ARM_SLOT_INDEX[arm]
    ti = c.TARGET_TURN_INDEX[regime]
    X = store["slots"].index_select(0, idx)[:, si].permute(1, 0, 2).contiguous()
    Y = store["profiles"].index_select(0, idx)[:, ti].permute(1, 0, 2).contiguous()
    return {"X": X.to(torch.float64), "Y": Y.to(torch.float64)}


# ---------------------------------------------------------------------------
# Batched dual ridge (layer axis leading)
# ---------------------------------------------------------------------------
def prep(Xtr: torch.Tensor) -> dict:
    """Standardization stats + Gram eigendecomposition. Xtr: (L, ntr, d)."""
    mu = Xtr.mean(1, keepdim=True)
    sd = Xtr.std(1, keepdim=True) + 1e-9
    Xn = (Xtr - mu) / sd
    w, V = torch.linalg.eigh(Xn @ Xn.transpose(1, 2))
    return {
        "mu": mu,
        "sd": sd,
        "Xn": Xn,
        "w": torch.clamp(w, min=0.0),
        "V": V,
        "ntr": int(Xtr.shape[1]),
    }


def _select_lambda(p: dict, VtY: torch.Tensor, tot: torch.Tensor) -> torch.Tensor:
    """GCV lambda per layer — the committed scan, vectorized over (grid, L)."""
    w, ntr = p["w"], p["ntr"]
    sqVtY = (VtY**2).sum(-1)
    filt = w.unsqueeze(0) / (w.unsqueeze(0) + LAMBDAS.view(-1, 1, 1))
    dof = filt.sum(-1)
    rss = tot.unsqueeze(0) - ((2 * filt - filt**2) * sqVtY.unsqueeze(0)).sum(-1)
    denom = (ntr - dof) ** 2
    gcv = torch.where(denom > 1e-12, rss / denom, torch.full_like(rss, float("inf")))
    if GCV_DOF_CAP is not None:
        gcv = torch.where(dof > GCV_DOF_CAP * ntr, torch.full_like(gcv, float("inf")), gcv)
    return LAMBDAS[gcv.argmin(0)]


def dual_predict(p: dict, Ytr: torch.Tensor, Xev: torch.Tensor) -> torch.Tensor:
    """Ridge p.X -> Ytr evaluated at Xev, batched over layers.

    Never forms the primal (d x d) operator: evaluation is a cross-Gram against
    the training rows.
    """
    ymu = Ytr.mean(1, keepdim=True)
    Yc = Ytr - ymu
    VtY = p["V"].transpose(1, 2) @ Yc
    lam = _select_lambda(p, VtY, (Yc**2).sum((-1, -2)))
    alpha = p["V"] @ (VtY / (p["w"] + lam[:, None]).unsqueeze(-1))
    return (((Xev - p["mu"]) / p["sd"]) @ p["Xn"].transpose(1, 2)) @ alpha + ymu


def procrustes_apply(
    P_tr: torch.Tensor, Y_tr: torch.Tensor, P_ev: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Rung 6: orthogonal Procrustes on the answer side, EXACT, no d x d SVD.

    Returns (prediction at P_ev, max relative off-subspace residual of P_ev).
    """
    pmu, ymu = P_tr.mean(1, keepdim=True), Y_tr.mean(1, keepdim=True)
    Q1, R1 = torch.linalg.qr((P_tr - pmu).transpose(1, 2), mode="reduced")
    Q2, R2 = torch.linalg.qr((Y_tr - ymu).transpose(1, 2), mode="reduced")
    Uc, _S, Vch = torch.linalg.svd(R1 @ R2.transpose(1, 2))
    Pe = P_ev - pmu
    proj = Pe @ Q1
    resid = (Pe - proj @ Q1.transpose(1, 2)).norm(dim=-1) / (Pe.norm(dim=-1) + 1e-12)
    return ((proj @ Uc) @ Vch) @ Q2.transpose(1, 2) + ymu, float(resid.max())


def knn_retrieval(pred: torch.Tensor, true: torch.Tensor, ks=(1, 5)) -> dict:
    """P(true target within k nearest neighbours of the prediction), euclidean."""
    out: dict = {}
    order = torch.cdist(pred, true).argsort(-1)
    n = true.shape[-2]
    tgt = torch.arange(n).view(1, -1, 1)
    for k in ks:
        hit = (order[:, :, :k] == tgt).any(-1).to(torch.float64).mean(-1)
        out[f"acc@{k}"] = [float(x) for x in hit]
        out[f"chance@{k}"] = k / n
    return out


def _rungs_for(
    p_s, p_ans, Ys_fit, Xs_hat_tr, Xs_hat_te, Xt_tr, Xt_te, Yt_tr, dx, dy
) -> tuple[dict, float]:
    """All 9 rung predictions for one (fold, direction) given a source-answer
    matrix `Ys_fit` (the real one, or a shuffled one for the null)."""
    P_tr = dual_predict(p_s, Ys_fit, Xt_tr)
    P_te = dual_predict(p_s, Ys_fit, Xt_te)
    P7_tr = dual_predict(p_s, Ys_fit, Xs_hat_tr)
    P7_te = dual_predict(p_s, Ys_fit, Xs_hat_te)
    pmu, ymu = P_tr.mean(1, keepdim=True), Yt_tr.mean(1, keepdim=True)
    bstar = (Yt_tr - P_tr).mean(1, keepdim=True)
    b7 = (Yt_tr - P7_tr).mean(1, keepdim=True)
    Pc, Yc = P_tr - pmu, Yt_tr - ymu
    a = (Pc * Yc).sum((-1, -2)) / (Pc.pow(2).sum((-1, -2)) + 1e-30)
    rot_te, resid = procrustes_apply(P_tr, Yt_tr, P_te)
    return {
        "1_direct": P_te,
        "2_ctx_offset": dual_predict(p_s, Ys_fit, Xt_te - dx),
        "3_ans_offset": P_te + dy,
        "4_bias_refit": P_te + bstar,
        "5_global_scale": a.view(-1, 1, 1) * (P_te - pmu) + ymu,
        "6_rotation": rot_te,
        "7_ctx_reparam": P7_te + b7,
        # B is fit on the ANSWER CLOUDS (source answers -> target answers) and
        # applied to the predicted source-space answer — never fit on P itself.
        "8_ans_reparam": dual_predict(p_ans, Yt_tr, P_te),
        "9_full_AMB": dual_predict(p_ans, Yt_tr, P7_te),
    }, resid


def run_cell(xy: dict, folds: np.ndarray, *, null_draws: int, seed: int) -> dict:
    """Both directions, sharing the four per-fold Gram eigendecompositions."""
    L = xy["r1"]["X"].shape[0]

    def z():
        return torch.zeros(L, dtype=torch.float64)

    acc = {d: {r: z() for r in RUNGS} | {"ceiling": z()} for d in DIRECTIONS}
    accn = {d: {r: z() for r in RUNGS} for d in DIRECTIONS}
    sstot = {d: z() for d in DIRECTIONS}
    knn: dict = {d: {} for d in DIRECTIONS}
    resid_max = 0.0
    rng = np.random.default_rng(seed)

    for k in range(N_FOLDS):
        tr, te = torch.as_tensor(folds != k), torch.as_tensor(folds == k)
        if int(te.sum()) == 0 or int(tr.sum()) < 3:
            continue
        # four preps per fold, shared by all 9 rungs in BOTH directions
        P = {}
        for reg in REGIMES:
            P[(reg, "X")] = prep(xy[reg]["X"][:, tr])
            P[(reg, "Y")] = prep(xy[reg]["Y"][:, tr])

        for d in DIRECTIONS:
            s, t = d
            p_s, p_t, p_ans = P[(s, "X")], P[(t, "X")], P[(s, "Y")]
            Xs_tr = xy[s]["X"][:, tr]
            Ys_tr = xy[s]["Y"][:, tr]
            Xt_tr, Xt_te = xy[t]["X"][:, tr], xy[t]["X"][:, te]
            Yt_tr, Yt_te = xy[t]["Y"][:, tr], xy[t]["Y"][:, te]
            dx = Xt_tr.mean(1, keepdim=True) - Xs_tr.mean(1, keepdim=True)
            dy = Yt_tr.mean(1, keepdim=True) - Ys_tr.mean(1, keepdim=True)
            # A: ridge target-contexts -> source-contexts (row-paired clouds)
            Xs_hat_tr = dual_predict(p_t, Xs_tr, Xt_tr)
            Xs_hat_te = dual_predict(p_t, Xs_tr, Xt_te)
            rung_args = (Xs_hat_tr, Xs_hat_te, Xt_tr, Xt_te, Yt_tr, dx, dy)
            preds, resid = _rungs_for(p_s, p_ans, Ys_tr, *rung_args)
            resid_max = max(resid_max, resid)
            ceiling = dual_predict(p_t, Yt_tr, Xt_te)

            sstot[d] += (Yt_te - Yt_te.mean(1, keepdim=True)).pow(2).sum((-1, -2))
            for r, pr in preds.items():
                acc[d][r] += (Yt_te - pr).pow(2).sum((-1, -2))
            acc[d]["ceiling"] += (Yt_te - ceiling).pow(2).sum((-1, -2))

            # matched-capacity null: source operator fit on shuffled answers
            for _ in range(null_draws):
                perm = torch.as_tensor(rng.permutation(int(tr.sum())))
                npred, _ = _rungs_for(p_s, p_ans, Ys_tr[:, perm], *rung_args)
                for r, pr in npred.items():
                    accn[d][r] += (Yt_te - pr).pow(2).sum((-1, -2)) / null_draws

            if k == 0:
                knn[d] = {
                    "n_pool": int(te.sum()),
                    "ceiling": knn_retrieval(ceiling, Yt_te),
                    "1_direct": knn_retrieval(preds["1_direct"], Yt_te),
                    "4_bias_refit": knn_retrieval(preds["4_bias_refit"], Yt_te),
                    "9_full_AMB": knn_retrieval(preds["9_full_AMB"], Yt_te),
                }
        del P

    out = {}
    for d in DIRECTIONS:

        def r2(ss, dd=d):
            return [float(x) for x in (1.0 - ss / sstot[dd])]

        out[DIR_KEY[d]] = {
            "r2": {r: r2(acc[d][r]) for r in RUNGS},
            "ceiling_r2": r2(acc[d]["ceiling"]),
            "null_r2": {r: r2(accn[d][r]) for r in RUNGS},
            "knn_retrieval_fold0": knn[d],
        }
    out["procrustes_subspace_residual_max"] = resid_max
    return out


def parity_check(xy: dict, folds: np.ndarray, layers: list[int], tol: float) -> dict:
    """Batched implementation vs the committed issue825_map_alignment helpers on
    the SAME rows — the target ceiling and the direct-transfer rung."""
    import issue825_map_alignment as ma

    li = layers.index(HEADLINE_LAYER)
    Xs, Ys = xy["r1"]["X"], xy["r1"]["Y"]
    Xt, Yt = xy["r2"]["X"], xy["r2"]["Y"]
    ss_c, ss_d, tot = 0.0, 0.0, 0.0
    for k in range(N_FOLDS):
        tr, te = torch.as_tensor(folds != k), torch.as_tensor(folds == k)
        ceil = ma._ridge_predict(ma._ridge_prep(Xt[li][tr]), Yt[li][tr], Xt[li][te])
        direct = ma._ridge_predict(ma._ridge_prep(Xs[li][tr]), Ys[li][tr], Xt[li][te])
        true = Yt[li][te]
        ss_c += float((true - ceil).pow(2).sum())
        ss_d += float((true - direct).pow(2).sum())
        tot += float((true - true.mean(0)).pow(2).sum())
    ref = {"ceiling": 1 - ss_c / tot, "1_direct": 1 - ss_d / tot}
    mine = run_cell(xy, folds, null_draws=0, seed=0)[DIR_KEY[("r1", "r2")]]
    got = {"ceiling": mine["ceiling_r2"][li], "1_direct": mine["r2"]["1_direct"][li]}
    deltas = {k: abs(ref[k] - got[k]) for k in ref}
    return {
        "reference": ref,
        "batched": got,
        "abs_delta": deltas,
        "tol": tol,
        "pass": all(v <= tol for v in deltas.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--cached-layers", type=int, nargs="+", default=list(cm.FROZEN_LAYERS))
    ap.add_argument("--layers", type=int, nargs="+", default=[HEADLINE_LAYER])
    ap.add_argument(
        "--out-dir", type=Path, default=_REPO_ROOT / "eval_results/issue_1345/ladder_rungs"
    )
    ap.add_argument("--models", nargs="+", default=["instruct", "pretrained"])
    ap.add_argument("--arms", nargs="+", default=list(c.ARMS))
    ap.add_argument("--null-draws", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    assert HEADLINE_LAYER in args.layers, "headline layer must be among --layers"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT
    ).stdout.strip()
    t_all = time.time()

    for model in args.models:
        t0 = time.time()
        stores = {
            r: load_cache(args.cache_dir, model, r, args.cached_layers, args.layers)
            for r in REGIMES
        }
        keep = np.array(sorted(set(stores["r1"]["conv_ids"]) & set(stores["r2"]["conv_ids"])))
        print(f"[{model}] loaded in {time.time() - t0:.0f}s | matched rows {len(keep)}", flush=True)
        folds = cm._cv_folds(keep, N_FOLDS, args.seed)

        for arm in args.arms:
            t1 = time.time()
            xy = {r: arm_xy(stores[r], r, arm, keep) for r in REGIMES}
            res = {
                "metadata": {
                    "git_commit": commit,
                    "script": "scripts/issue1345_ladder_rungs.py",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "seed": args.seed,
                    "n_matched_rows": int(len(keep)),
                    "null_draws": args.null_draws,
                    "cache_dir": str(args.cache_dir),
                },
                "model": model,
                "arm": arm,
                "layers": args.layers,
                "headline_layer": HEADLINE_LAYER,
                "rung_order": list(RUNGS),
                "ladder_note": "corrections fit on TARGET TRAIN fold; source operator frozen",
            }
            if args.parity:
                res["parity"] = parity_check(xy, folds, args.layers, args.tol)
                print(f"[{model}/{arm}] parity {res['parity']}", flush=True)
            res.update(run_cell(xy, folds, null_draws=args.null_draws, seed=args.seed))
            out = args.out_dir / f"ladder_rungs_{model}_{arm}.json"
            out.write_text(json.dumps(res, indent=2))
            li = args.layers.index(HEADLINE_LAYER)
            for dk in (DIR_KEY[d] for d in DIRECTIONS):
                dd = res[dk]
                print(f"  {dk}: ceiling {dd['ceiling_r2'][li]:.4f}", flush=True)
                for r in RUNGS:
                    print(
                        f"    {r:16s} {dd['r2'][r][li]:9.4f}  null {dd['null_r2'][r][li]:9.4f}",
                        flush=True,
                    )
            print(f"[{model}/{arm}] wrote {out} | cell wall {time.time() - t1:.0f}s", flush=True)
        del stores
    print(f"TOTAL {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
