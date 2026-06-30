#!/usr/bin/env python3
"""#722 behavior-chain — ADD a KERNEL-RIDGE (RBF) readout to the ridge analysis.

Extends /tmp/ridge_chain_analysis.py with a nonlinear v0->E0 readout:

  (A_krr) DIRECT KRR-RBF: KRR(RBF) fit on (true PCA48(v0) -> E0), LOCO held-out
          on true v0.  [nonlinear ceiling for v0->behavior]
  (B_krr) MEDIATED KRR-RBF: SAME readout (trained on true-v0 train rows) applied
          to the c_C-PREDICTED PCA48(v0) held-out row.

Plus the linear-kernel-KRR == ridge sanity check (krr kernel="linear" should
reproduce the closed-form linear ridge readout) and a label-shuffle null for the
direct KRR (should collapse to ~0). Helpers reused from the issue-722 branch
version of vectorized_mlp_skill (krr_predict_loco / _kernel_gram / _krr_loo_press
/ _default_rbf_gammas) — the SAME machinery the c_C->v0 KRR used.

Writes behavior_chain_krr_readout.json merging ridge + KRR per behavior/layer,
with bootstrap CIs on the nonlinear gap (KRR_direct - ridge_direct) and the
KRR-preservation gap (KRR_direct - KRR_mediated) at the best layer.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Single-threaded BLAS/torch — the n=50 49x49 dual/KRR solves THRASH on many
# threads (LESSONS: a sub-second op inflated to 30s on 16 threads). Set BEFORE
# numpy/torch import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

torch.set_num_threads(1)

PROJECT_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
# IMPORTANT: import the BRANCH (issue-722) version of vectorized_mlp_skill, which
# carries krr_predict_loco + kernel helpers (not on main).
WT = Path("/tmp/eps-ridge-wt")
sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(WT / "scripts"))  # issue658_fit_predictors needs this

import issue658_fit_predictors as _i658  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    _kernel_gram,
    krr_predict_loco,
    ridge_predict_loco_centered,
    robust_pca_basis,
)

RIDGE_LAMBDAS = list(_i658.RIDGE_LAMBDAS)
# KRR nested-CV grids. KRR uses a WIDE gamma grid (median-anchor scaled toward
# sharper kernels) so nested CV can find local structure — the median-heuristic
# anchor ALONE is degenerate on PCA-48 (high-dim distance concentration makes the
# RBF kernel near-uniform -> KRR collapses to the train mean, predicting noise;
# diagnosed empirically). KRR therefore runs on a REDUCED PCA-16 v0 space where
# the RBF kernel is non-degenerate; ridge keeps the full PCA-48 (its linear
# ceiling). The nonlinear gap is then a fair "best nonlinear readout vs best
# linear readout" contrast. lambda grid kept compact (the VM is oversubscribed).
KRR_LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]
KRR_GAMMA_FACTORS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)  # × median-heuristic anchor
KRR_PCA_DIM = 16  # RBF feature dim (PCA-48 is degenerate for RBF at n=50)
BEHAVIORS = ["broad_em", "harmful_compliance", "sycophancy", "refusal"]
PCA_TARGET_DIM = 48  # ridge readout target dim (linear ceiling)
RIDGE_ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
N_LAYERS = 28
V0_FILE = PROJECT_ROOT / "data/issue_658/store/v0_summaries.pt"
RB_FILE = PROJECT_ROOT / "data/issue_658/store/r_b.pt"
E0_LOCAL = PROJECT_ROOT / "eval_results/issue_658/E0_expression.json"


def _median_gamma(Xc: np.ndarray) -> float:
    """RBF median-heuristic gamma anchor: 1/median(||x_i-x_j||^2) on standardized X.

    Same anchor _default_rbf_gammas builds its grid around; here we take just the
    anchor and build a tight 3-value grid in krr_readout_both.
    """
    X = np.ascontiguousarray(Xc.astype(np.float64))
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    Xn = (X - mu) / sd
    sq = (Xn * Xn).sum(1)[:, None] + (Xn * Xn).sum(1)[None, :] - 2.0 * (Xn @ Xn.T)
    iu = np.triu_indices(Xn.shape[0], k=1)
    med = float(np.median(np.clip(sq[iu], 0.0, None)))
    return 1.0 / med if med > 0 else 1.0


def _spearman(a, b):
    if len(a) < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    r, _ = spearmanr(a, b)
    return None if np.isnan(r) else float(r)


def e0_vector(e0, col, ctx_ids):
    vals, kept = [], []
    for i, c in enumerate(ctx_ids):
        cell = e0.get("e0", {}).get(c, {}).get(col)
        if cell is None:
            continue
        v = cell.get("rate", cell.get("logp_mean"))
        if v is None:
            continue
        vals.append(float(v))
        kept.append(i)
    return np.array(vals), kept


# ── ridge scalar readouts (same as the base analysis) ─────────────────────────
def ridge_readout_direct(Z, y):
    n = Z.shape[0]
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        m = RidgeCV(alphas=RIDGE_ALPHAS).fit((Z[tr] - mu) / sd, y[tr])
        preds[i] = m.predict(((Z[i] - mu) / sd)[None])[0]
    return preds


def ridge_readout_mediated(Zt, Zp, y):
    n = Zt.shape[0]
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        mu, sd = Zt[tr].mean(0), Zt[tr].std(0) + 1e-9
        m = RidgeCV(alphas=RIDGE_ALPHAS).fit((Zt[tr] - mu) / sd, y[tr])
        preds[i] = m.predict(((Zp[i] - mu) / sd)[None])[0]
    return preds


# ── KRR scalar readouts ───────────────────────────────────────────────────────
def krr_readout_direct(Z, y, kernel="rbf"):
    """LOCO KRR readout on true Z (the helper's standard path). Returns (n,) preds.

    Used for the linear-kernel == ridge sanity check + the shuffle null. The RBF
    direct + mediated come fused from ``krr_readout_both`` (one inner CV per fold).
    """
    preds, _lam, _gam = krr_predict_loco(Z, y[:, None], kernel=kernel)
    return preds[:, 0]


def krr_readout_both(Zt, Zp, y, kernel="rbf"):
    """Fused LOCO KRR: ONE per-fold inner CV (fit on true-Z train), evaluated at
    BOTH the true held-out point (direct) and the c_C-predicted held-out point
    (mediated). The readout fit (gamma, lambda, dual coeffs A) is identical for
    direct and mediated — only the held-out kernel row differs — so fitting once
    and applying twice is exact AND ~2x faster than two separate inner-CV loops.

    Mirrors krr_predict_loco's exact inner CV (train-standardize, nested LOO-PRESS
    pick of (gamma, lambda) on TRAIN rows, dual solve). Returns (pred_direct,
    pred_mediated), each (n,).
    """
    n = Zt.shape[0]
    device = torch.device(_i658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(Zt)).to(device=device, dtype=torch.float64)
    Xp = torch.from_numpy(np.ascontiguousarray(Zp)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(y[:, None])).to(device=device, dtype=torch.float64)
    if kernel == "rbf":
        g0 = _median_gamma(Zt)
        gammas = [g0 * f for f in KRR_GAMMA_FACTORS]  # wide grid around median anchor
    else:
        gammas = [0.0]
    lam_v = torch.tensor(KRR_LAMBDAS, device=device, dtype=torch.float64)  # (Lg,)
    pred_d = np.zeros(n)
    pred_m = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        tr_t = torch.tensor(tr, device=device)
        Xtr = Xt[tr_t]
        Ytr = Yt[tr_t]
        xmu = Xtr.mean(0)
        xsd = Xtr.std(0, correction=0) + 1e-9
        Xtr_n = (Xtr - xmu) / xsd
        ymu = Ytr.mean(0)
        Ytr_c = Ytr - ymu
        x_held_true = (Xt[i] - xmu) / xsd
        x_held_pred = (Xp[i] - xmu) / xsd
        best = None  # (press, alpha, k_true, k_pred)
        yc = Ytr_c[:, 0]  # (m,)
        for gam in gammas:
            Ktr = _kernel_gram(Xtr_n, Xtr_n, kernel, gam)  # (m, m), symmetric PSD
            k_true = _kernel_gram(x_held_true.unsqueeze(0), Xtr_n, kernel, gam).squeeze(0)
            k_pred = _kernel_gram(x_held_pred.unsqueeze(0), Xtr_n, kernel, gam).squeeze(0)
            # ONE eigendecomposition scores ALL lambdas in closed form, VECTORIZED:
            # H_lam = V diag(s/(s+lam)) V^T; LOO residual_j = (y_j-(H y)_j)/(1-H_jj).
            s, Vk = torch.linalg.eigh(Ktr)  # s (m,), Vk (m, m)
            s = s.clamp(min=0.0)
            g = Vk.T @ yc  # (m,) coords of y_c in eigenbasis
            V2 = Vk * Vk  # (m, m), V2[j,k] = V_jk^2
            filt = s[None, :] / (s[None, :] + lam_v[:, None])  # (Lg, m)
            fitted = filt * g[None, :] @ Vk.T  # (Lg, m) = H_lam y_c per lambda
            hdiag = filt @ V2.T  # (Lg, m) diag(H_lam) per lambda
            hdiag = hdiag.clamp(max=1.0 - 1e-9)
            resid = (yc[None, :] - fitted) / (1.0 - hdiag)  # (Lg, m)
            press = (resid * resid).mean(dim=1)  # (Lg,)
            bl = int(torch.argmin(press).item())
            p_best = float(press[bl].item())
            if best is None or p_best < best[0]:
                lam = float(lam_v[bl].item())
                alpha = Vk @ (g / (s + lam))  # (m,) dual coeffs at chosen lambda
                best = (p_best, alpha, k_true, k_pred)
        _p, alpha, k_true, k_pred = best
        pred_d[i] = float((ymu[0] + (k_true @ alpha)).item())
        pred_m[i] = float((ymu[0] + (k_pred @ alpha)).item())
    return pred_d, pred_m


def _boot_rho_diff_ci(pred_x, pred_y, y, n_boot=2000, seed=0):
    """Paired bootstrap CI on Spearman(pred_x, y) - Spearman(pred_y, y).

    Resamples contexts with replacement; both rhos computed on the SAME resampled
    rows. Returns {point, ci_lo, ci_hi, frac_gt0}.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    point = (_spearman(pred_x, y) or 0.0) - (_spearman(pred_y, y) or 0.0)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(y[idx]) < 1e-9:
            continue
        rx = _spearman(pred_x[idx], y[idx])
        ry = _spearman(pred_y[idx], y[idx])
        if rx is not None and ry is not None:
            diffs.append(rx - ry)
    if not diffs:
        return {"point": point, "ci_lo": None, "ci_hi": None, "frac_gt0": None}
    d = np.array(diffs)
    return {
        "point": float(point),
        "ci_lo": float(np.percentile(d, 2.5)),
        "ci_hi": float(np.percentile(d, 97.5)),
        "frac_gt0": float((d > 0).mean()),
    }


def main():
    v0 = torch.load(V0_FILE, weights_only=False)
    ctx_ids = list(v0["context_ids"])
    V = np.stack([v0["summaries"]["mean"][c].numpy() for c in ctx_ids]).astype(np.float64)
    cc_variant = "last-input-token"
    try:
        from issue658_common import load_cc_last_store

        cc_d = load_cc_last_store(capture_layers=list(range(N_LAYERS)), ctx_ids=ctx_ids)
        C = np.stack([cc_d[c].numpy() for c in ctx_ids]).astype(np.float64)
    except Exception as exc:
        print(f"[WARN] cc_last failed ({exc}); falling back to cc_meanprompt", file=sys.stderr)
        cc_variant = "mean-prompt variant (last-token unavailable)"
        C = np.stack([v0["cc_meanprompt"][c].numpy() for c in ctx_ids]).astype(np.float64)
    rb = torch.load(RB_FILE, weights_only=False)
    r_b_avail = {b for b in BEHAVIORS if b in rb.get("r_b", {})}
    e0 = json.loads(E0_LOCAL.read_text())

    # per-layer caches:
    #   cache[li]    = (PCA48 true v0, PCA48 c_C-predicted v0)   — ridge readout space
    #   cache_krr[li]= (PCA16 true v0, PCA16 c_C-predicted v0)   — KRR readout space
    # Both predicted-v0 come from the SAME LOCO ridge map c_C->PCA(v0), fit in the
    # respective PCA space (the c_C->v0 approximation under test).
    cache = {}
    cache_krr = {}
    for li in range(N_LAYERS):
        Y, Xc = V[:, li, :], C[:, li, :]
        mu, comps, _ = robust_pca_basis(Y, PCA_TARGET_DIM)
        Y_pca = (Y - mu) @ comps.T
        cache[li] = (Y_pca, ridge_predict_loco_centered(Xc, Y_pca))
        muk, compsk, _ = robust_pca_basis(Y, KRR_PCA_DIM)
        Yk = (Y - muk) @ compsk.T
        cache_krr[li] = (Yk, ridge_predict_loco_centered(Xc, Yk))

    per_behavior = {}
    for b in BEHAVIORS:
        if b not in r_b_avail:
            per_behavior[b] = {"status": "no r_B"}
            continue
        y, kept = e0_vector(e0, b, ctx_ids)
        if len(kept) < 4:
            per_behavior[b] = {"status": f"too few E0 ({len(kept)})"}
            continue
        rows = []
        bRA = bRB = bKA = bKB = None
        # cache per-layer predictions so the best-layer CIs reuse them (no recompute)
        preds_by_layer = {}
        for li in range(N_LAYERS):
            # ridge on PCA-48 (linear ceiling)
            Yp_all, Pp_all = cache[li]
            Yp, Pp = Yp_all[kept], Pp_all[kept]
            pRA = ridge_readout_direct(Yp, y)
            rRA = _spearman(pRA, y)
            rRB = _spearman(ridge_readout_mediated(Yp, Pp, y), y)
            # KRR-RBF on PCA-16 (RBF non-degenerate here) — fused direct + mediated
            Yk_all, Pk_all = cache_krr[li]
            Yk, Pk = Yk_all[kept], Pk_all[kept]
            pKA, pKB = krr_readout_both(Yk, Pk, y, kernel="rbf")
            rKA = _spearman(pKA, y)
            rKB = _spearman(pKB, y)
            preds_by_layer[li] = {"pRA": pRA, "pKA": pKA, "pKB": pKB}
            rows.append(
                {
                    "layer": li,
                    "rho_ridge_direct": rRA,
                    "rho_ridge_mediated": rRB,
                    "rho_krr_direct": rKA,
                    "rho_krr_mediated": rKB,
                    "nonlinear_gap": (rKA - rRA) if (rKA is not None and rRA is not None) else None,
                    "krr_degradation": (rKA - rKB)
                    if (rKA is not None and rKB is not None)
                    else None,
                }
            )
            if rRA is not None and (bRA is None or rRA > bRA["rho"]):
                bRA = {"layer": li, "rho": rRA}
            if rRB is not None and (bRB is None or rRB > bRB["rho"]):
                bRB = {"layer": li, "rho": rRB}
            if rKA is not None and (bKA is None or rKA > bKA["rho"]):
                bKA = {"layer": li, "rho": rKA}
            if rKB is not None and (bKB is None or rKB > bKB["rho"]):
                bKB = {"layer": li, "rho": rKB}

        # CIs at the KRR-direct best layer: nonlinear gap (KRR - ridge, direct)
        # and KRR-preservation gap (KRR direct - KRR mediated). Reuse cached preds.
        ci_nl = ci_pres = None
        sanity_lin = None
        sanity_shuffle = None
        if bKA is not None:
            Lbest = bKA["layer"]
            pKA_b = preds_by_layer[Lbest]["pKA"]
            pRA_b = preds_by_layer[Lbest]["pRA"]
            pKB_b = preds_by_layer[Lbest]["pKB"]
            ci_nl = _boot_rho_diff_ci(pKA_b, pRA_b, y)  # nonlinear gap > 0?
            ci_pres = _boot_rho_diff_ci(pKA_b, pKB_b, y)  # KRR preservation deg
            # linear-kernel KRR == ridge sanity, SAME PCA-16 KRR space (apples-to-apples):
            # linear-kernel KRR should reproduce closed-form ridge on the same features.
            Yk_best = cache_krr[Lbest][0][kept]
            pKlin, _ = krr_readout_both(Yk_best, Yk_best, y, kernel="linear")
            pRid16 = ridge_readout_direct(Yk_best, y)  # ridge on the SAME PCA-16 space
            rho_klin = _spearman(pKlin, y)
            rho_rid16 = _spearman(pRid16, y)
            sanity_lin = {
                "layer": Lbest,
                "krr_space": f"PCA-{KRR_PCA_DIM}",
                "rho_krr_linear": rho_klin,
                "rho_ridge_same_space": rho_rid16,
                "abs_diff": abs((rho_klin or 0) - (rho_rid16 or 0)),
            }
            # label-shuffle null for direct KRR (RBF, PCA-16): should be ~0
            rng = np.random.default_rng(0)
            ysh = y.copy()
            rng.shuffle(ysh)
            pKsh, _ = krr_readout_both(Yk_best, Yk_best, ysh, kernel="rbf")
            sanity_shuffle = _spearman(pKsh, ysh)
        per_behavior[b] = {
            "n_kept_e0": len(kept),
            "best_ridge_direct": bRA,
            "best_ridge_mediated": bRB,
            "best_krr_direct": bKA,
            "best_krr_mediated": bKB,
            "nonlinear_gap_best": (bKA["rho"] - bRA["rho"]) if (bKA and bRA) else None,
            "krr_degradation_best": (bKA["rho"] - bKB["rho"]) if (bKA and bKB) else None,
            "ci_nonlinear_gap_at_krr_best": ci_nl,
            "ci_krr_preservation_at_krr_best": ci_pres,
            "sanity_linear_kernel_eq_ridge": sanity_lin,
            "sanity_shuffle_null_krr_direct": sanity_shuffle,
            "per_layer": rows,
        }
        print(
            f"  {b:22s} ridgeA={bRA['rho']:.3f}@L{bRA['layer']:>2d} "
            f"krrA={bKA['rho']:.3f}@L{bKA['layer']:>2d} "
            f"krrB={bKB['rho']:.3f}@L{bKB['layer']:>2d} | "
            f"NLgap={bKA['rho'] - bRA['rho']:+.3f} "
            f"krrDeg={bKA['rho'] - bKB['rho']:+.3f} | "
            f"lin==ridge? d={sanity_lin['abs_diff']:.3f} shuf={sanity_shuffle:+.3f}"
        )

    out = {
        "analysis": "behavioral_chain_preservation_RIDGE_plus_KRR_READOUT",
        "description": (
            "LOCO n=50 Spearman rho(predicted E0, actual E0) under ridge AND KRR-RBF "
            "v0->E0 readouts. ridge direct/mediated on PCA-48 v0 (linear ceiling); "
            "KRR-RBF direct/mediated on PCA-16 v0 (RBF is degenerate on PCA-48 at "
            "n=50 due to high-dim distance concentration -> collapses to the mean). "
            "direct = readout on true v0; mediated = same readout applied to "
            "c_C-predicted v0 (LOCO ridge map c_C->PCA(v0)). nonlinear_gap = "
            "rho(krr_direct) - rho(ridge_direct) [best layer each]; krr_degradation "
            "= rho(krr_direct) - rho(krr_mediated). CIs = paired bootstrap (2000 "
            "resamples) at the krr-direct best layer."
        ),
        "readout_ridge": f"RidgeCV(alphas={RIDGE_ALPHAS}) on PCA-{PCA_TARGET_DIM} v0",
        "readout_krr": (
            f"RBF kernel ridge on PCA-{KRR_PCA_DIM} v0; gamma grid = median-heuristic "
            f"anchor x {KRR_GAMMA_FACTORS}; lambda grid {KRR_LAMBDAS}; nested LOO-PRESS "
            "(gamma, lambda) pick; closed-form eigendecomposition (matches "
            "vectorized_mlp_skill.krr_predict_loco to machine precision)"
        ),
        "pca_target_dim_ridge": PCA_TARGET_DIM,
        "pca_target_dim_krr": KRR_PCA_DIM,
        "cc_variant": cc_variant,
        "layers": list(range(N_LAYERS)),
        "per_behavior": per_behavior,
    }
    p = PROJECT_ROOT / "eval_results/issue_722/structural/behavior_chain_krr_readout.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[info] wrote {p}")


if __name__ == "__main__":
    main()
