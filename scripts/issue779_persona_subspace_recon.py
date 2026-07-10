"""Issue #779 inline free-analysis — JOINT reconstruction of the persona-vector subspace.

Per-direction R2 (Result 4) scores each r_B alone. This asks the joint question:
how well does the linear context->answer map reconstruct the answer profile's
projection onto the 3-D subspace S = span(r_B^evil, r_B^syco, r_B^halluc)?

Subspace reconstruction R2 is basis-invariant (depends only on the projector
P_S = Q Qᵀ): R2 = 1 - ||P_S(Y_te - pred)||_F² / ||P_S(Y_te - mean)||_F².
Nulls: (a) random 3-D subspaces (matched dimension only); (b) a matched-variance-
rank subspace = the PCA directions at the r_B's equivalent variance ranks (isolates
"is the persona subspace reconstructed better than same-variance directions?").
Also reports the subspace's total answer-variance share + the pairwise r_B cosines.

Same protocol as identity_baseline (L19 recon-best layer, fold 0 of 5-fold, seed 0,
full-ridge h). 0-GPU, cached pass_b + r_B.
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


def _subspace_r2(Yte, pred, Q):
    """Basis-invariant pooled R2 of the projection onto span(Q) (Q: H x r orthonormal)."""
    a = Yte @ Q  # (n, r)
    p = pred @ Q
    ss_res = float(((a - p) ** 2).sum())
    ss_tot = float(((a - a.mean(0)) ** 2).sum())
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=500)
    ap.add_argument("--rb-dir", type=Path, default=Path("data/issue_779/r_b"))
    ap.add_argument(
        "--out-json", type=Path, default=F.DEFAULT_OUT_DIR / "persona_subspace_recon.json"
    )
    args = ap.parse_args()

    bundle = F.load_pass_b()
    layers = list(bundle["layers"])
    li = layers.index(args.layer)
    X = bundle["cx_last"][:, li, :].to(dtype=torch.float32).numpy().astype(np.float64)
    Y = bundle["v_x"][:, li, :].to(dtype=torch.float32).numpy().astype(np.float64)
    n, hdim = X.shape
    test_idx = F.PR._cv_folds(n, args.n_folds, args.seed)[0]
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    Xtr, Ytr, Xte, Yte = X[mask], Y[mask], X[test_idx], Y[test_idx]
    pred = F.PR._ridge_fit_predict_fast(Xtr, Ytr, Xte)

    # persona-vector subspace (orthonormalized)
    rbs = np.stack(
        [S1._load_rb(args.rb_dir, t, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)[li] for t in TRAITS]
    )
    rbs_u = rbs / (np.linalg.norm(rbs, axis=1, keepdims=True) + 1e-12)
    Q, _ = np.linalg.qr(rbs_u.T)  # (H, 3)
    subspace_r2 = _subspace_r2(Yte, pred, Q)

    # pairwise cosines + effective rank of the 3 r_B
    cos = rbs_u @ rbs_u.T
    s = np.linalg.svd(rbs_u.T, compute_uv=False)
    eff_rank = float((s.sum() ** 2) / (s**2).sum())

    # subspace variance share of the answer profile
    Ytr_c = Ytr - Ytr.mean(0)
    tot_var = float((Ytr_c**2).sum())
    sub_var = float(((Ytr_c @ Q) ** 2).sum())
    var_share = sub_var / tot_var

    # matched-variance-rank subspace: PCA dirs at the r_B equivalent variance ranks
    _u2, sv, vh = np.linalg.svd(Ytr_c, full_matrices=False)
    var_spec = (sv**2) / (Ytr.shape[0] - 1)
    eq_ranks = []
    for i in range(3):
        var_rb = float(np.var(Ytr_c @ rbs_u[i], ddof=1))
        eq_ranks.append(int(np.sum(var_spec > var_rb)))
    Qm, _ = np.linalg.qr(vh[eq_ranks].T)
    matched_rank_r2 = _subspace_r2(Yte, pred, Qm)

    # random 3-D subspace null
    rng = np.random.default_rng(args.seed + 779)
    rand_r2 = []
    for _ in range(args.n_random):
        G = rng.standard_normal((hdim, 3))
        Qr, _ = np.linalg.qr(G)
        rand_r2.append(_subspace_r2(Yte, pred, Qr))
    rand_r2 = np.array(rand_r2, float)

    # individual r_B R2 (single-direction), for reference
    indiv = {t: float(_subspace_r2(Yte, pred, rbs_u[i][:, None])) for i, t in enumerate(TRAITS)}

    out = {
        "layer": args.layer,
        "persona_subspace_r2": subspace_r2,
        "individual_r_b_r2": indiv,
        "matched_variance_rank_subspace_r2": matched_rank_r2,
        "matched_ranks": eq_ranks,
        "random_3d_subspace_null": {
            "n": args.n_random,
            "mean": float(rand_r2.mean()),
            "sd": float(rand_r2.std()),
            "p5": float(np.quantile(rand_r2, 0.05)),
            "p95": float(np.quantile(rand_r2, 0.95)),
        },
        "subspace_variance_share": var_share,
        "r_b_pairwise_cosine": {
            f"{TRAITS[i]}|{TRAITS[j]}": float(cos[i, j]) for i in range(3) for j in range(i + 1, 3)
        },
        "r_b_effective_rank": eff_rank,
        "note": (
            f"Joint reconstruction of span(r_B) at L{args.layer}, fold 0. Subspace R2 is "
            "basis-invariant (projector P_S). matched-variance-rank subspace = PCA dirs at the "
            "r_B equivalent variance ranks; random 3-D null = random orthonormal 3-frames."
        ),
        "metadata": {
            "script": "issue779_persona_subspace_recon",
            "seed": args.seed,
            "n_folds": args.n_folds,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    F.C.write_json_atomic(args.out_json, out)

    print(f"wrote {args.out_json}")
    print(f"  persona-subspace R2      = {subspace_r2:.4f}")
    print(f"  matched-variance-rank R2 = {matched_rank_r2:.4f} (ranks {eq_ranks})")
    print(
        f"  random 3-D subspace      = {rand_r2.mean():.4f} ± {rand_r2.std():.4f} "
        f"(p95 {np.quantile(rand_r2, 0.95):.4f})"
    )
    print("  individual r_B R2        = " + ", ".join(f"{t} {v:.3f}" for t, v in indiv.items()))
    print(f"  subspace variance share  = {var_share:.4%} of total answer variance")
    print(
        f"  r_B effective rank       = {eff_rank:.2f} / 3; pairwise cos = "
        + ", ".join(f"{k}:{v:+.2f}" for k, v in out["r_b_pairwise_cosine"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
