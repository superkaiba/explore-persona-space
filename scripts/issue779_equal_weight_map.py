"""Issue #779 inline free-analysis — does weighting the map to care EQUALLY about
every answer direction change what it reconstructs?

Standard ridge minimizes Frobenius MSE, which is variance-weighted (high-variance
answer directions dominate the loss). "Care equally about each direction" = whiten
the target so every PCA direction has unit variance in the loss (GLS metric). For a
linear map this only bites through ridge shrinkage (unregularized OLS is metric-
invariant), so we compare two maps at L19:
  A = standard ridge (GCV lambda) on the raw target
  B = ridge on the top-K whitened target (each of the top-K PCs unit-variance)
and read per-PC held-out R2 (scale-invariant, so directly comparable) plus r_B R2,
the variance-weighted pooled R2, and the equal-weight (mean per-direction) R2.

0-GPU, cached pass_b + r_B. Fold 0 of 5-fold, seed 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_stage1 as S1  # noqa: E402
import numpy as np  # noqa: E402

TRAITS = ("evil", "sycophancy", "hallucination")


def _perdir_r2(Yte, pred, dirs):
    a = Yte @ dirs
    p = pred @ dirs
    ss_res = ((a - p) ** 2).sum(0)
    ss_tot = ((a - a.mean(0)) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss_tot < 1e-12, np.nan, 1.0 - ss_res / ss_tot)


def _ridge_map(Xtr, Ytr, lam):
    """Closed-form ridge weight M (D_out x D_in) minimizing ||Y - X Mᵀ||² + lam||M||²."""
    xmu, ymu = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - xmu, Ytr - ymu
    G = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    W = np.linalg.solve(G, Xc.T @ Yc)  # (D_in, D_out)
    return W, xmu, ymu


def _predict(Xte, W, xmu, ymu):
    return (Xte - xmu) @ W + ymu


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--topk", type=int, default=200, help="whiten within the top-K PCs")
    ap.add_argument("--rb-dir", type=Path, default=Path("data/issue_779/r_b"))
    ap.add_argument("--out-json", type=Path, default=F.DEFAULT_OUT_DIR / "equal_weight_map.json")
    args = ap.parse_args()

    bundle = F.load_pass_b()
    layers = list(bundle["layers"])
    li = layers.index(args.layer)
    X = bundle["cx_last"][:, li, :].to(dtype=torch.float32).numpy().astype(np.float64)
    Y = bundle["v_x"][:, li, :].to(dtype=torch.float32).numpy().astype(np.float64)
    n = X.shape[0]
    test_idx = F.PR._cv_folds(n, args.n_folds, args.seed)[0]
    m = np.ones(n, bool)
    m[test_idx] = False
    Xtr, Ytr, Xte, Yte = X[m], Y[m], X[test_idx], Y[test_idx]

    # answer-target PCA (train fold)
    Ytr_c = Ytr - Ytr.mean(0)
    _u, sv, vh = np.linalg.svd(Ytr_c, full_matrices=False)
    V = vh  # (D, H) rows = PCA dirs
    var_spec = (sv**2) / (Ytr.shape[0] - 1)
    K = min(args.topk, V.shape[0])

    rbs = {t: S1._load_rb(args.rb_dir, t, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)[li] for t in TRAITS}
    rbs_u = {t: v / (np.linalg.norm(v) + 1e-12) for t, v in rbs.items()}
    rb_ranks = {t: int(np.sum(var_spec > np.var(Ytr_c @ rbs_u[t], ddof=1))) for t in TRAITS}

    # A = standard ridge on raw Y (GCV-ish: pick lambda minimizing held-out pooled on a small grid)
    lambdas = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    best = None
    for lam in lambdas:
        W, xmu, ymu = _ridge_map(Xtr, Ytr, lam)
        pr = _predict(Xte, W, xmu, ymu)
        pooled = 1 - ((Yte - pr) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
        if best is None or pooled > best[0]:
            best = (pooled, lam, pr)
    pooled_A, lamA, predA = best

    # B = ridge on the top-K WHITENED target (each top-K PC unit variance), same lambda grid.
    # whitened target coords: Ztr = (Ytr_c @ V[:K]) / sqrt(var); reconstruct pred back to H-space
    scale = np.sqrt(var_spec[:K])  # per-PC std
    Ztr = (Ytr_c @ V[:K].T) / scale  # (n_tr, K), unit-variance columns
    bestB = None
    for lam in lambdas:
        Wz, xmu, zmu = _ridge_map(Xtr, Ztr, lam)  # X -> whitened PCs
        Zte_pred = _predict(Xte, Wz, xmu, zmu)  # (n_te, K)
        # back to H-space: pred = ymu + (Zpred * scale) @ V[:K]
        prB = Ytr.mean(0) + (Zte_pred * scale) @ V[:K]
        pooled = 1 - ((Yte - prB) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
        if bestB is None or pooled > bestB[0]:
            bestB = (pooled, lam, prB)
    pooled_B, lamB, predB = bestB

    # per-PC R2 for both maps at a ladder of ranks + r_B
    probe_ranks = [0, 4, 10, 25, 50, 100, 150, 199]
    probe_ranks = [r for r in probe_ranks if r < V.shape[0]]
    dirs = V[probe_ranks].T
    r2A_pc = _perdir_r2(Yte, predA, dirs)
    r2B_pc = _perdir_r2(Yte, predB, dirs)
    rb_dirs = np.stack([rbs_u[t] for t in TRAITS]).T
    r2A_rb = _perdir_r2(Yte, predA, rb_dirs)
    r2B_rb = _perdir_r2(Yte, predB, rb_dirs)

    # equal-weight score = mean per-direction R2 over the top-K PCs (each weighted equally)
    dirsK = V[:K].T
    eqA = float(np.nanmean(_perdir_r2(Yte, predA, dirsK)))
    eqB = float(np.nanmean(_perdir_r2(Yte, predB, dirsK)))

    out = {
        "layer": args.layer,
        "topk": K,
        "standard_ridge": {
            "lambda": lamA,
            "pooled_r2": float(pooled_A),
            "equal_weight_meanPC_r2_topK": eqA,
        },
        "whitened_topK_ridge": {
            "lambda": lamB,
            "pooled_r2": float(pooled_B),
            "equal_weight_meanPC_r2_topK": eqB,
        },
        "per_pc_r2": {
            str(r): {"standard": float(r2A_pc[i]), "whitened": float(r2B_pc[i])}
            for i, r in enumerate(probe_ranks)
        },
        "r_b_r2": {
            t: {"rank": rb_ranks[t], "standard": float(r2A_rb[i]), "whitened": float(r2B_rb[i])}
            for i, t in enumerate(TRAITS)
        },
        "note": (
            "A=standard ridge (variance-weighted MSE); B=ridge on top-K whitened target "
            "(equal weight per PC). Per-PC R2 is scale-invariant so directly comparable. "
            "Unregularized OLS would be metric-invariant; the difference here is ridge shrinkage."
        ),
        "metadata": {
            "script": "issue779_equal_weight_map",
            "seed": args.seed,
            "n_folds": args.n_folds,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    F.C.write_json_atomic(args.out_json, out)

    print(f"wrote {args.out_json}")
    print(
        f"  A standard ridge (lam={lamA:g}): pooled R2={pooled_A:.4f} "
        f"equal-weight(mean top-{K} PC) R2={eqA:.4f}"
    )
    print(
        f"  B whitened top-{K} ridge (lam={lamB:g}): pooled R2={pooled_B:.4f} "
        f"equal-weight R2={eqB:.4f}"
    )
    print("  per-PC R2 (standard -> whitened):")
    for i, r in enumerate(probe_ranks):
        print(f"    PC rank {r:>3d}: {r2A_pc[i]:+.3f} -> {r2B_pc[i]:+.3f}")
    print("  r_B R2 (standard -> whitened):")
    for i, t in enumerate(TRAITS):
        print(f"    {t:14s} (rank {rb_ranks[t]:>2d}): {r2A_rb[i]:.3f} -> {r2B_rb[i]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
