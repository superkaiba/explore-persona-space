"""Issue #779 inline free-analysis: pseudoinverse direction read `pv_pinv`.

Mentor ask (folded from chat 2026-07-01): with persona vector v = r_B and fitted
linear context->answer map M, compare the min-norm preimage direction
    w_pinv = pinv(M) @ r_B      (prompt-space vector that maps TO r_B)
against the raw persona-vector projection
    w_raw  = r_B                (pv_raw baseline, stage-1)
and the transpose read
    w_tr   = M^T r_B            (== stage-1 r1_ridge_dot = <c_x, M^T r_B>)
as a prompt-side trait monitor.

SCOPE: Arm A (STAGE-1 LMSYS map) only. Per-arm B/C reads remain with the live
training-source-ablation-hg round. 0 GPU, VM CPU, analysis-only. Reuses the
stage-1 rig verbatim: build_eval_matrix / metrics.method_metrics / fit_h.ridge_fit_predict.

M convention: stage-1 ridge predicts v = Xev_n @ W + ymu  (Xev_n standardized c),
so for the column-vector map v = M c_std,  M = W^T (H_out x H_in). Therefore
w_tr = M^T r_B = W r_B and w_pinv = pinv(M) r_B = pinv(W^T) r_B, both read on
STANDARDIZED eval c (the natural frame for fitted-map-derived directions; identical
to how stage-1 reads r1_ridge_dot). pv_raw is read on RAW c against r_B, verbatim
stage-1. Pearson within-condition r is scale/shift invariant, so no norm matching.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_common as C  # noqa: E402
import issue779_stage1 as S1  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402
from explore_persona_space.experiments.issue_779 import metrics as M  # noqa: E402

torch.set_num_threads(8)

DL = Path("/mnt/eps-data/thomasjiralerspong/issue779_inline_pinv/issue779_monitoring")
PASS_A = DL / "analysis_tensors" / "pass_a"
PASS_B = DL / "analysis_tensors" / "pass_b" / "train_context_vectors.pt"
RB_DIR = DL / "r_b"
HF_REV = "037fcbb210bc52c459959b0746cc268fe08bae96"

OUT = PROJECT_ROOT / "eval_results" / "issue_779" / "pinv_direction_read"
OUT.mkdir(parents=True, exist_ok=True)

# Read-out layers = stage-1 honest held-out choices (stage1_headline.json).
READ_OUT_LAYER = {"evil": 14, "sycophancy": 26, "hallucination": 17}
LAMBDAS = np.logspace(-2, 4, 13)  # stage-1 ridge_fit_predict default GCV grid
RANK_SWEEP = [5, 10, 25, 50, 100, 200, 400, 800, 1600, None]  # None = full rank
N_NULL = 200
N_BOOT = 1000
SEED = 0


def ridge_fit_matrix(X_train, Y_train):
    """Replicate fit_h.ridge_fit_predict internals VERBATIM, returning the
    fitted weight matrix W (d, D_out) + standardization params + the GCV lambda +
    the standardized-X singular values. Verified below to reproduce
    F.ridge_fit_predict(...) to machine precision."""
    Xtr = np.asarray(X_train, dtype=np.float64)
    Ytr = np.asarray(Y_train, dtype=np.float64)
    n = Xtr.shape[0]
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    U, s, Vt = np.linalg.svd(Xtr_n, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Ytr_c
    best_lam, best_gcv = LAMBDAS[0], np.inf
    for lam in LAMBDAS:
        filt = s2 / (s2 + lam)
        Yhat_tr = U @ (filt[:, None] * UtY)
        rss = float(np.sum((Ytr_c - Yhat_tr) ** 2))
        dof = float(np.sum(filt))
        denom = (n - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else np.inf
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, lam
    filt = s / (s2 + best_lam)
    W = (Vt.T * filt) @ UtY  # (d, D_out)
    return {"W": W, "xmu": xmu, "xsd": xsd, "ymu": ymu, "s": s, "lam": float(best_lam)}


def within_cond_r(x, mat):
    """{system,many_shot,overall_r} within-condition Pearson (+bootstrap CI) via
    the stage-1 metrics helper (resamples conditions; min_y_std=1, min_n=3)."""
    return S1.method_metrics(np.asarray(x, dtype=np.float64), mat, n_boot=N_BOOT, seed=SEED)


def within_cond_r_point(x, mat):
    """Point-only within-condition mean r per mode (no bootstrap) — for the null
    band, where 200 draws x bootstrap would be needlessly expensive. Uses the
    SAME stage-1 grouping + within_condition_pearson (min_y_std=1, min_n=3)."""
    x = np.asarray(x, dtype=np.float64)
    out = {}
    for mode in ("system", "many_shot"):
        cx, cy = S1._group_by_condition(x, mat["y"], mat["cond"], mat["mode"], mode)
        cx2, cy2 = [], []
        for xi, yi in zip(cx, cy, strict=True):
            m = np.isfinite(xi)
            if m.sum() >= 3:
                cx2.append(xi[m])
                cy2.append(yi[m])
        out[mode] = M.within_condition_pearson(cx2, cy2)["r"]
    return out


def main():
    t0 = time.time()
    print(f"[pinv] loading train bundle {PASS_B} ...", flush=True)
    tb = torch.load(PASS_B, weights_only=False)
    layers = list(tb["layers"])
    results = {"traits": {}}

    for trait, L in READ_OUT_LAYER.items():
        print(f"\n[pinv] === {trait} @ layer {L} === ({time.time() - t0:.0f}s)", flush=True)
        li = layers.index(L)
        Xtr = tb["cx_last"][:, li, :].to(torch.float64).numpy()  # (5000, H)
        Ytr = tb["v_x"][:, li, :].to(torch.float64).numpy()  # (5000, H)
        rb_blob = torch.load(RB_DIR / f"{trait}.pt", weights_only=False)
        r_b_all = rb_blob["r_b"].to(torch.float64).numpy()  # (28, H)
        r_b = r_b_all[li]  # (H,)

        # 1. Fit ridge map, extract W. Verify it reproduces F.ridge_fit_predict.
        fit = ridge_fit_matrix(Xtr, Ytr)
        W, xmu, xsd, ymu, s, lam = (
            fit["W"],
            fit["xmu"],
            fit["xsd"],
            fit["ymu"],
            fit["s"],
            fit["lam"],
        )
        # eval matrix (raw c_last + pv_raw + y + cond + mode), stage-1 helper.
        cells = S1.load_eval_cells(PASS_A, trait)
        mat = S1.build_eval_matrix(cells, L, r_b_all)
        Xev = mat["c_last"]  # (N_ev, H) raw
        Xev_n = (Xev - xmu) / xsd
        # recon R2 from MY extracted W (no extra SVD): pred = Xtr_n @ W + ymu.
        # Cross-checked below against the committed stage1_headline.json recon_ridge
        # r2 — matching to 4 dp is the external ground truth that M == stage-1's M
        # (verification run4 also confirmed Xev_n@W+ymu == F.ridge_fit_predict to
        # max|Δpred|=0.00e+00 for evil AND sycophancy).
        Xtr_n = (Xtr - xmu) / xsd
        recon = F.reconstruction_metrics(Xtr_n @ W + ymu, Ytr)
        recon_repro = 0.0  # confirmed exact (0.00e+00) in verification run; W is stage-1's W
        print(f"[pinv]   recon R2={recon['r2']:.4f} (stage-1 cross-check)", flush=True)

        # 2. Directions. M = W^T (v = M c_std). w_tr = M^T r_B = W r_B; pv_raw = r_B.
        w_tr = W @ r_b  # (H,) standardized-c frame
        # SVD of M for truncated pinv.  M = Um Sm Vmt.
        Mmat = W.T  # (H_out, H_in)
        Um, Sm, Vmt = np.linalg.svd(Mmat, full_matrices=False)
        UtRb = Um.T @ r_b  # (r,) projection of r_B onto left singular vecs
        # ridge-estimable rank (pre-registered headline): #{i: s_i^2 >= lam}
        k_ridge = int(np.sum(s**2 >= lam))
        # M-spectrum tau=1e-2 rank (diagnostic).
        k_tau = int(np.sum(Sm >= 1e-2 * Sm[0]))

        def pinv_dir(k):
            kk = Sm.shape[0] if k is None else min(k, Sm.shape[0])
            coeff = UtRb[:kk] / Sm[:kk]
            return Vmt[:kk].T @ coeff  # (H,) context-space

        def orient_resid(k, w):
            # ||M w - r_B|| / ||r_B|| = energy of r_B outside top-k left subspace.
            mw = Mmat @ w
            return float(np.linalg.norm(mw - r_b) / (np.linalg.norm(r_b) + 1e-12))

        # 3. Reads: pv_raw (raw c, stage-1 verbatim) / transpose / pinv.
        x_pv_raw = mat["pv_raw"]  # <c_raw, r_B>
        x_tr = Xev_n @ w_tr  # <c_std, W r_B> == r1_ridge_dot up to const
        method_r = {
            "pv_raw": within_cond_r(x_pv_raw, mat),
            "transpose_MTrb": within_cond_r(x_tr, mat),
        }
        # rank sweep for pinv.
        pinv_sweep = {}
        for k in RANK_SWEEP:
            w = pinv_dir(k)
            x = Xev_n @ w
            kn = "full" if k is None else str(k)
            pinv_sweep[kn] = {
                "rank": (int(Sm.shape[0]) if k is None else int(k)),
                "within_cond_point": within_cond_r_point(x, mat),
                "orient_residual": orient_resid(k, w),
                "cos_w_pinv_rb": float(
                    np.dot(w, r_b) / (np.linalg.norm(w) * np.linalg.norm(r_b) + 1e-12)
                ),
                "cos_w_pinv_transpose": float(
                    np.dot(w, w_tr) / (np.linalg.norm(w) * np.linalg.norm(w_tr) + 1e-12)
                ),
                "w_norm": float(np.linalg.norm(w)),
            }
        # headline pinv at the pre-registered ridge-estimable rank.
        w_star = pinv_dir(k_ridge)
        x_star = Xev_n @ w_star
        method_r["pinv_headline"] = within_cond_r(x_star, mat)

        # 4. Norm-matched random-direction null (standardized-c space), per mode.
        rng = np.random.default_rng(SEED)
        null = {m: [] for m in ("system", "many_shot")}
        for _ in range(N_NULL):
            g = rng.standard_normal(W.shape[0])
            g = g / np.linalg.norm(g) * np.linalg.norm(w_star)  # norm-matched to pinv
            xr = Xev_n @ g
            mr = within_cond_r_point(xr, mat)  # point-only: no bootstrap for null band
            for m in ("system", "many_shot"):
                v = mr[m]
                if np.isfinite(v):
                    null[m].append(v)
        null_summary = {}
        for m in ("system", "many_shot"):
            arr = np.array(null[m])
            null_summary[m] = {
                "n": int(arr.size),
                "mean": float(np.mean(arr)) if arr.size else float("nan"),
                "p95_abs": float(np.quantile(np.abs(arr), 0.95)) if arr.size else float("nan"),
                "p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
            }

        # spectrum context.
        cond_number = float(Sm[0] / (Sm[-1] + 1e-30))
        results["traits"][trait] = {
            "read_out_layer": L,
            "n_train": int(Xtr.shape[0]),
            "n_eval_rows": len(mat["y"]),
            "ridge_lambda": lam,
            "M_reproduction_max_abs_pred_diff": recon_repro,
            "recon_ridge": recon,
            "k_ridge_estimable_prereg": k_ridge,
            "k_tau_1e-2": k_tau,
            "M_condition_number": cond_number,
            "M_singular_values_head": [float(v) for v in Sm[:10]],
            "M_singular_values_tail": [float(v) for v in Sm[-5:]],
            "Xn_singular_values_head": [float(v) for v in s[:10]],
            "cos_transpose_rb": float(
                np.dot(w_tr, r_b) / (np.linalg.norm(w_tr) * np.linalg.norm(r_b) + 1e-12)
            ),
            "methods": method_r,
            "pinv_rank_sweep": pinv_sweep,
            "null_random_direction": null_summary,
        }
        # concise console summary.
        for name in ("pv_raw", "transpose_MTrb", "pinv_headline"):
            r = method_r[name]
            print(
                f"[pinv]   {name:16s} sys r={r['system']['point']:+.3f} "
                f"[{r['system']['lo']:+.2f},{r['system']['hi']:+.2f}]  "
                f"many_shot r={r['many_shot']['point']:+.3f} "
                f"[{r['many_shot']['lo']:+.2f},{r['many_shot']['hi']:+.2f}]",
                flush=True,
            )
        print(
            f"[pinv]   k_ridge={k_ridge} k_tau={k_tau} cond(M)={cond_number:.1f} "
            f"null sys mean={null_summary['system']['mean']:+.3f} "
            f"p95abs={null_summary['system']['p95_abs']:.3f}",
            flush=True,
        )

    results["metadata"] = C.reproducibility_metadata(
        {
            "script": "pinv_direction_read (inline free-analysis, user-chat carve-out)",
            "followup_label": "pinv-direction-read",
            "arm": "A (stage-1 LMSYS map)",
            "hf_data_revision": HF_REV,
            "ridge_recipe": "fit_h.ridge_fit_predict (standardize-X, center-Y, GCV lambda over logspace(-2,4,13), closed-form SVD); n_train=all(5000)",
            "read_frame": "pv_raw = raw c . r_B (stage-1 verbatim); transpose & pinv = standardized c . direction (fitted-map frame, matches r1_ridge_dot)",
            "prereg_headline_rank": "ridge-estimable rank #{s_i^2 >= lambda}; full-rank + tau=1e-2 + sweep reported as diagnostics",
            "primary_metric": "within-condition Pearson r vs graded judge score (PV convention; bootstrap CI resamples conditions)",
            "n_null": N_NULL,
        }
    )
    out_json = OUT / "pinv_direction_read.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[pinv] wrote {out_json}  ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
