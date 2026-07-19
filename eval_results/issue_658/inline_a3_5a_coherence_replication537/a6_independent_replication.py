#!/usr/bin/env python3
"""Independent SECOND replication of A6 (context-vector coherence) using the
#537 context-generalization activation clouds -- a DIFFERENT HF store, DIFFERENT
34-context panel, DIFFERENT probe pool than the prior #658 in-repo analysis
(eval_results/issue_658/inline_a3_5a_coherence/coherence_results.json).

For each condition C (34 total), each anchor (prefix-based end_of_system,
context-based last_prompt), each layer L:
  - c_x = per-probe activation vector (500 probes/condition) at that anchor+layer
  - a_x = per-probe answer-side vector (mean_response anchor) at that layer
  - s_W(C) = mean_x || c_x - c_C ||^2 ,  c_C = mean_x c_x   (W = I, and whitened)
  - h fit via ridge regression on POOLED per-probe (c_x, a_x) pairs across the
    OTHER 33 conditions (leave-one-condition-out over CONDITIONS, not probes),
    applied to c_C to get R_loco(C) = || v0(C) - h(c_C) ||
    where v0(C) = mean_x a_x (observed).
  - To keep the O(d^3) ridge solve cheap (d=3584), the INPUT is projected onto
    a top-256 PCA basis fit ONCE on the pooled (all-34-condition) c_x cloud
    (mild global-basis leakage into every fold, disclosed) before the ridge
    normal equations are solved (256x256, not 3584x3584); the residual R_loco
    is still computed in the FULL 3584-dim output space.
  - Jensen gap under the SAME (linear) h: J(C) = || mean_x h(c_x) - h(c_C) ||
    (identically ~0 for a linear map -- reported as a sanity check).

Reports Spearman/Pearson rho(s_W, R_loco) across the 34 conditions, per layer,
per anchor, plus a permutation p-value and a condition-level bootstrap CI.
"""

import json
import time

import numpy as np
import scipy.linalg
from scipy import stats
from sklearn.utils.extmath import randomized_svd

CLOUDS_DIR = "/home/thomasjiralerspong/explore-persona-space/data/issue_theorycheck/hf_dl/issue537_context_generalization/clouds"

CTX_NAMES = [
    "binst_em",
    "binst_fact",
    "binst_marker",
    "binst_refusal",
    "binst_sycophancy",
    "default",
    "fmt_code",
    "fmt_json",
    "fmt_mdtable_ho",
    "icl_k2",
    "icl_k4_ho",
    "icl_k8",
    "neg_reph_curious",
    "neg_sp_ph4",
    "neg_sp_police",
    "neg_wc_short",
    "reph_casual",
    "reph_formal_ho",
    "reph_imp",
    "reph_polite",
    "reph_socratic_ho",
    "sp_doctor",
    "sp_ph1",
    "sp_ph2",
    "sp_ph3_ho",
    "sp_swe",
    "sp_teacher_ho",
    "wc_long_ho",
    "wc_long_write",
    "wc_short_advice",
    "wc_short_code",
    "wc_short_ho",
    "wc_xlong_ho",
    "wc_xxlong_ho",
]
N_CTX = len(CTX_NAMES)
LAYERS_TO_SCAN = [10, 14, 18, 22, 26]
RIDGE_LAMBDA = 100.0  # applied in the 256-dim PCA-projected standardized space
PCA_K = 256
N_PERM = 20000
N_BOOT = 4000


def load_anchor(ctx, anchor):
    d = np.load(f"{CLOUDS_DIR}/{ctx}__{anchor}.npz", allow_pickle=True)
    return d["hidden"]  # (500, 29, 3584) float16


def pca_basis(X_pool, k):
    """Top-k PCA basis via randomized SVD on centered pooled data (fast on a
    single-threaded shared-VM BLAS build: O(n*d*k) instead of O(n*d*min(n,d))).
    Returns (mean, components (k,d))."""
    mu = X_pool.mean(0)
    Xc = X_pool - mu
    U, S, Vt = randomized_svd(Xc, n_components=k, n_oversamples=20, n_iter=4, random_state=0)
    return mu, Vt[:k]  # (d,), (k, d)


def main():
    t0 = time.time()
    results = {}
    context_kind = {"end_of_system": "prefix", "last_prompt": "context"}

    for anchor in ["end_of_system", "last_prompt"]:
        print(f"=== anchor: {anchor} ({context_kind[anchor]}-based) ===", flush=True)
        ctx_hidden = {ctx: load_anchor(ctx, anchor) for ctx in CTX_NAMES}
        resp_hidden = {ctx: load_anchor(ctx, "mean_response") for ctx in CTX_NAMES}
        results[anchor] = {}

        for layer in LAYERS_TO_SCAN:
            tl = time.time()
            Cx = {ctx: ctx_hidden[ctx][:, layer, :].astype(np.float32) for ctx in CTX_NAMES}
            Ax = {ctx: resp_hidden[ctx][:, layer, :].astype(np.float32) for ctx in CTX_NAMES}
            cC = {ctx: Cx[ctx].mean(0) for ctx in CTX_NAMES}
            v0 = {ctx: Ax[ctx].mean(0) for ctx in CTX_NAMES}

            # ---- spread s_W(C), W = I ----
            sW_I = {
                ctx: float(np.mean(np.sum((Cx[ctx] - cC[ctx]) ** 2, axis=1))) for ctx in CTX_NAMES
            }

            # ---- whitened spread (pooled within-condition covariance) ----
            # Project residuals onto the same top-256 PCA basis used below so the
            # (d,d) factorization is 256x256 (cheap) instead of 3584x3584 -- the
            # whitening metric W is applied in this reduced subspace (captures
            # >~99% of pooled residual variance per the #658 companion analysis'
            # own pca_explained_var ~0.996 at PCA_IN=256).
            resid_all = np.concatenate([Cx[ctx] - cC[ctx] for ctx in CTX_NAMES], axis=0)
            n_pool = resid_all.shape[0]
            mu_resid, basis_resid = pca_basis(resid_all, PCA_K)  # basis_resid (256,3584)
            resid_all_p = (resid_all - mu_resid) @ basis_resid.T  # (n_pool,256)
            Sigma_p = (resid_all_p.T @ resid_all_p) / n_pool
            lam_whiten = 0.01 * np.trace(Sigma_p) / PCA_K
            Sigma_reg = Sigma_p + lam_whiten * np.eye(PCA_K)
            chol = scipy.linalg.cho_factor(Sigma_reg, lower=True)
            sW_W = {}
            for ctx in CTX_NAMES:
                diffs = Cx[ctx] - cC[ctx]
                diffs_p = (diffs - mu_resid) @ basis_resid.T  # (500,256)
                sol = scipy.linalg.cho_solve(chol, diffs_p.T)  # (256,500)
                quad = np.sum(diffs_p.T * sol, axis=0)
                sW_W[ctx] = float(quad.mean())

            # ---- global PCA basis on pooled c_x (mild leakage: basis only) ----
            X_pool_all = np.concatenate([Cx[c] for c in CTX_NAMES], axis=0)
            mu_pca, basis = pca_basis(X_pool_all, PCA_K)  # basis (256, 3584)

            def project(X):
                return (X - mu_pca) @ basis.T  # (n, 256)

            Cx_p = {c: project(Cx[c]) for c in CTX_NAMES}
            cC_p = {c: project(cC[c][None, :])[0] for c in CTX_NAMES}

            # ---- LOCO ridge in PCA-256 space, residual measured in FULL space ----
            R_loco = {}
            J_lin = {}
            for held in CTX_NAMES:
                train_ctxs = [c for c in CTX_NAMES if c != held]
                Xtr = np.concatenate([Cx_p[c] for c in train_ctxs], axis=0)  # (16500,256)
                Ytr = np.concatenate([Ax[c] for c in train_ctxs], axis=0)  # (16500,3584)

                mu_x = Xtr.mean(0)
                sd_x = Xtr.std(0) + 1e-6
                mu_y = Ytr.mean(0)
                Xs = (Xtr - mu_x) / sd_x
                Ys = Ytr - mu_y
                A = Xs.T @ Xs + RIDGE_LAMBDA * np.eye(PCA_K)
                b = Xs.T @ Ys  # (256, 3584)
                W = np.linalg.solve(A, b)  # (256, 3584)

                xt = (cC_p[held] - mu_x) / sd_x
                pred_mean = xt @ W + mu_y
                R_loco[held] = float(np.linalg.norm(v0[held] - pred_mean))

                Xtest_std = (Cx_p[held] - mu_x) / sd_x  # (500,256)
                h_of_x = Xtest_std @ W + mu_y  # (500,3584)
                h_mean_of_x = h_of_x.mean(0)
                h_of_mean = pred_mean
                J_lin[held] = float(np.linalg.norm(h_mean_of_x - h_of_mean))

            sW_I_arr = np.array([sW_I[c] for c in CTX_NAMES])
            sW_W_arr = np.array([sW_W[c] for c in CTX_NAMES])
            Rloco_arr = np.array([R_loco[c] for c in CTX_NAMES])
            Jlin_arr = np.array([J_lin[c] for c in CTX_NAMES])

            rho_I, p_I = stats.spearmanr(sW_I_arr, Rloco_arr)
            rho_W, p_W = stats.spearmanr(sW_W_arr, Rloco_arr)
            pear_I, pp_I = stats.pearsonr(sW_I_arr, Rloco_arr)
            pear_W, pp_W = stats.pearsonr(sW_W_arr, Rloco_arr)

            rng = np.random.default_rng(0)
            idx = np.arange(N_CTX)
            null = np.empty(N_PERM)
            for i in range(N_PERM):
                perm = rng.permutation(idx)
                null[i] = stats.spearmanr(sW_I_arr, Rloco_arr[perm])[0]
            perm_p = float((np.sum(np.abs(null) >= abs(rho_I)) + 1) / (N_PERM + 1))

            boot = np.empty(N_BOOT)
            for i in range(N_BOOT):
                bidx = rng.integers(0, N_CTX, N_CTX)
                boot[i] = stats.spearmanr(sW_I_arr[bidx], Rloco_arr[bidx])[0]
            ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5])

            results[anchor][layer] = {
                "rho_sW_I_Rloco": rho_I,
                "p_sW_I_Rloco": p_I,
                "rho_sW_W_Rloco": rho_W,
                "p_sW_W_Rloco": p_W,
                "pearson_sW_I_Rloco": pear_I,
                "pearson_sW_W_Rloco": pear_W,
                "perm_p_sW_I_Rloco": perm_p,
                "bootstrap_ci95_sW_I_Rloco": [float(ci_lo), float(ci_hi)],
                "J_lin_max": float(np.max(np.abs(Jlin_arr))),
                "J_lin_mean": float(np.mean(np.abs(Jlin_arr))),
                "sW_I": sW_I,
                "sW_W": sW_W,
                "R_loco": R_loco,
            }
            print(
                f"  layer {layer}: rho(sW_I,Rloco)={rho_I:.3f} (p={p_I:.2e}, perm_p={perm_p:.4f}, "
                f"95%CI=[{ci_lo:.2f},{ci_hi:.2f}])  rho(sW_whitened,Rloco)={rho_W:.3f} (p={p_W:.2e})  "
                f"J_lin_max={np.max(np.abs(Jlin_arr)):.2e}  [{time.time() - tl:.1f}s]",
                flush=True,
            )

    out = {
        "meta": {
            "store": "issue537_context_generalization/clouds",
            "n_ctx": N_CTX,
            "n_probes_per_ctx": 500,
            "layers_scanned": LAYERS_TO_SCAN,
            "ridge_lambda": RIDGE_LAMBDA,
            "pca_k": PCA_K,
            "pca_basis_leakage_note": "PCA basis for input compression fit on POOLED all-34-condition "
            "c_x (mild basis-only leakage into every LOCO fold); the ridge "
            "COEFFICIENTS themselves are always fit excluding the held-out "
            "condition (the honest LOCO part).",
            "runtime_s": time.time() - t0,
            "ctx_names": CTX_NAMES,
        },
        "results": results,
    }
    with open("/tmp/a6_coherence_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote /tmp/a6_coherence_result.json ; runtime", time.time() - t0, "s")


if __name__ == "__main__":
    main()
