"""Inline free-analysis (issue #658): A3.5a within-condition coherence test
(assumption ``a:context-vector-coherence``, theory doc + theory_assumption_test_plan.md
R5-1 / §4 coherence block). ANALYSIS-ONLY on #658's stored per-probe activations —
no training / generation / GPU. Consumes /tmp/issue658_a35a/reduced.npz built by
scripts/issue658_inline_a3_5a_reduce.py.

Per condition C (50) and per family, layer-swept over all 28 stored layers:
  (a) within-condition spread  s_W(C) = (1/n) sum_i ||c_{x_i} - c_hat_C||^2_W
      metric W: identity, and whitened W=(Sigma_c+lambda I)^-1 (Sigma_c pooled over all c_x).
  (b) behavior-relevant Jensen gap  J(C) = max_B |(1/n) sum_i f_B(c_{x_i}) - f_B(c_hat_C)|
      f_B(c) = r_B . h(c), h fit as a nonlinear MLP scalar map c_x -> r_B . a_x
      (linear ridge companion is ~0 by construction since c_hat_C = mean_i c_{x_i}).
  (c) context->profile residual  R(C) = max_B |(1/n) sum_i (r_B . a_{x_i}) - f_B(c_hat_C)|
                                       = max_B |r_B . v0(C) - f_B(c_hat_C)|.
Descriptive companions: within/between cosine, eta^2, silhouette.
Verdict: Spearman(s_W, J), Spearman(s_W, R) per layer + family; OLS slope ~ 1/2 K.

Incremental: writes coherence_results.json + per_condition_layer.npz after EVERY layer,
so a kill/timeout preserves a complete-as-of-last-layer artifact.
"""

import json
import os
import subprocess
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

CACHE = "/tmp/issue658_a35a"
OUT_JSON = "eval_results/issue_658/inline_a3_5a_coherence"
os.makedirs(OUT_JSON, exist_ok=True)
RNG = 0
torch.manual_seed(RNG)
np.random.seed(RNG)

WHITEN_LAMBDA_FRAC = 1e-2
RIDGE_GRID = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
PCA_IN = 256
MLP_HIDDEN = 512
MLP_EPOCHS = 250
MLP_LR = 1e-3
MLP_WD = 1e-4


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def fit_mlp_multihead(Xtr, Ytr, Xeval_list):
    """One full-batch AdamW MLP with n_out heads; returns per-eval (n,n_out) preds + train."""
    d, n_out = Xtr.shape[1], Ytr.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(d, MLP_HIDDEN), torch.nn.GELU(), torch.nn.Linear(MLP_HIDDEN, n_out)
    )
    Xt = torch.from_numpy(Xtr.astype(np.float32))
    Yt = torch.from_numpy(Ytr.astype(np.float32))
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    lossf = torch.nn.MSELoss()
    net.train()
    for _ in range(MLP_EPOCHS):
        opt.zero_grad()
        loss = lossf(net(Xt), Yt)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        preds = [net(torch.from_numpy(X.astype(np.float32))).numpy() for X in Xeval_list]
        train_pred = net(Xt).numpy()
    return preds, train_pred


def ridge_fit_predict(Xtr, Ytr, Xeval, alpha):
    d = Xtr.shape[1]
    w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(d), Xtr.T @ Ytr)
    return Xeval @ w


def ridge_select_alpha(Xtr, y):
    n = len(y)
    idx = np.arange(n)
    np.random.default_rng(RNG).shuffle(idx)
    folds = np.array_split(idx, 5)
    best_a, best_mse = RIDGE_GRID[0], np.inf
    for a in RIDGE_GRID:
        mses = []
        for f in folds:
            tr = np.setdiff1d(idx, f)
            pred = ridge_fit_predict(Xtr[tr], y[tr], Xtr[f], a)
            mses.append(float(np.mean((pred - y[f]) ** 2)))
        m = float(np.mean(mses))
        if m < best_mse:
            best_mse, best_a = m, a
    return best_a


def sp(a, b):
    rho, p = spearmanr(a, b)
    return {"rho": float(rho), "p": float(p)}


def main():
    t0 = time.time()
    npz = np.load(os.path.join(CACHE, "reduced.npz"))
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    ctx_ids, families, behaviors = meta["ctx_ids"], meta["families"], meta["behaviors"]
    fam_arr = np.array(families)
    cc_last = npz["cc_last"]
    v0_mean = npz["v0_mean"]
    rB = npz["rB"]
    n_ctx, n_probe, nB = 50, 48, len(behaviors)
    n_layers = cc_last.shape[1]
    labels = np.repeat(np.arange(n_ctx), n_probe)

    spread_I = np.full((n_layers, n_ctx), np.nan)
    spread_W = np.full((n_layers, n_ctx), np.nan)
    within_cos = np.full((n_layers, n_ctx), np.nan)
    sil_cond = np.full((n_layers, n_ctx), np.nan)
    J_max = np.full((n_layers, n_ctx), np.nan)
    R_max = np.full((n_layers, n_ctx), np.nan)
    Rlin_loco_max = np.full((n_layers, n_ctx), np.nan)
    Jlin_max = np.full((n_layers, n_ctx), np.nan)
    eta2 = np.full(n_layers, np.nan)
    sil_global = np.full(n_layers, np.nan)

    results = {
        "meta": {
            "issue": 658,
            "analysis": "A3.5a within-condition coherence",
            "assumption": "a:context-vector-coherence",
            "source_repo": "superkaiba1/explore-persona-space-data",
            "source_rev": "b33429f77b86",
            "git_commit": git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "consistency_checks": meta["checks"],
            "n_ctx": n_ctx,
            "n_probe": n_probe,
            "n_layers": n_layers,
            "behaviors": behaviors,
            "ctx_ids": ctx_ids,
            "families": families,
            "whiten_lambda_frac": WHITEN_LAMBDA_FRAC,
            "pca_in": PCA_IN,
            "mlp": {"hidden": MLP_HIDDEN, "epochs": MLP_EPOCHS, "lr": MLP_LR, "wd": MLP_WD},
            "notes": [
                "c_hat_C = mean_i c_{x_i} exactly (checks ~3e-8) => linear-h Jensen gap identically 0; "
                "nonlinear multi-head MLP h used for J.",
                "R_max uses all-data (in-sample) MLP h -> slight optimism (lower bound); "
                "Rlin_loco_max is the honest leave-one-condition-out linear-ridge companion.",
                "within-condition c_x<->a_x pairing is positional (identical probe_pool_hash+seed).",
                "assumption predicts POSITIVE rho(s_W, J) and rho(s_W, R): gap+residual rise with spread.",
            ],
        },
        "per_layer": {},
    }

    def finalize_and_save():
        # family summary
        fam_summary = {}
        for fam in sorted(set(families)):
            m = fam_arr == fam
            fam_summary[fam] = {
                "n": int(m.sum()),
                "mean_spread_W_overall": float(np.nanmean(spread_W[:, m])),
                "mean_R_overall": float(np.nanmean(R_max[:, m])),
                "mean_J_overall": float(np.nanmean(J_max[:, m])),
                "mean_within_cos_overall": float(np.nanmean(within_cos[:, m])),
            }
        results["family_summary"] = fam_summary
        # low-coherence flags (mean over completed layers)
        sW_ctx = np.nanmean(spread_W, 0)
        R_ctx = np.nanmean(R_max, 0)
        sW_thr = float(np.nanquantile(sW_ctx, 0.8))
        R_thr = float(np.nanquantile(R_ctx, 0.8))
        flags = []
        for ci in range(n_ctx):
            hs, hr = sW_ctx[ci] >= sW_thr, R_ctx[ci] >= R_thr
            if hs or hr:
                flags.append(
                    {
                        "context_id": ctx_ids[ci],
                        "family": families[ci],
                        "mean_spread_W": float(sW_ctx[ci]),
                        "mean_R": float(R_ctx[ci]),
                        "hi_spread": bool(hs),
                        "hi_residual": bool(hr),
                        "recommend": "split_or_richer_summary"
                        if (hs and hr)
                        else "caveat_downweight",
                    }
                )
        flags.sort(key=lambda d: -d["mean_spread_W"])
        results["low_coherence_flags"] = flags
        results["flag_thresholds"] = {"spread_W_q80": sW_thr, "R_q80": R_thr}
        done = [L for L in range(n_layers) if str(L) in results["per_layer"]]
        if done:
            sWv = spread_W[done].ravel()
            rj, pj = spearmanr(sWv, J_max[done].ravel())
            rr, pr = spearmanr(sWv, R_max[done].ravel())
            lrj = np.array([results["per_layer"][str(L)]["spearman_sW_J_all"]["rho"] for L in done])
            lrr = np.array([results["per_layer"][str(L)]["spearman_sW_R_all"]["rho"] for L in done])
            results["overall"] = {
                "layers_completed": done,
                "pooled_spearman_sW_J": {"rho": float(rj), "p": float(pj)},
                "pooled_spearman_sW_R": {"rho": float(rr), "p": float(pr)},
                "median_layerwise_rho_sW_J": float(np.median(lrj)),
                "median_layerwise_rho_sW_R": float(np.median(lrr)),
                "frac_layers_rho_sW_J_pos": float((lrj > 0).mean()),
                "frac_layers_rho_sW_R_pos": float((lrr > 0).mean()),
                "n_low_coherence_flags": len(flags),
            }
        results["meta"]["runtime_s"] = round(time.time() - t0, 1)
        json.dump(results, open(os.path.join(OUT_JSON, "coherence_results.json"), "w"), indent=1)
        np.savez(
            os.path.join(OUT_JSON, "per_condition_layer.npz"),
            spread_I=spread_I,
            spread_W=spread_W,
            within_cos=within_cos,
            sil_cond=sil_cond,
            J_max=J_max,
            R_max=R_max,
            Rlin_loco_max=Rlin_loco_max,
            Jlin_max=Jlin_max,
            eta2=eta2,
            sil_global=sil_global,
        )

    for L in range(n_layers):
        cx_L = npz["cx"][:, :, L, :].astype(np.float32)
        ax_L = npz["ax"][:, :, L, :].astype(np.float32)
        cc_L = cc_last[:, L, :]
        X = cx_L.reshape(n_ctx * n_probe, -1)
        grand = X.mean(0)
        Xc = X - grand
        Sigma = (Xc.T @ Xc) / (X.shape[0] - 1)
        lam = WHITEN_LAMBDA_FRAC * (np.trace(Sigma) / Sigma.shape[0])
        Lchol = np.linalg.cholesky(Sigma + lam * np.eye(Sigma.shape[0]))

        tot_ss = with_ss = 0.0
        for ci in range(n_ctx):
            D = cx_L[ci] - cc_L[ci]
            sq = np.sum(D * D, axis=1)
            spread_I[L, ci] = float(sq.mean())
            with_ss += float(sq.sum())
            Y = np.linalg.solve(Lchol, D.T)
            spread_W[L, ci] = float(np.mean(np.sum(Y * Y, axis=0)))
            Xn = cx_L[ci] / (np.linalg.norm(cx_L[ci], axis=1, keepdims=True) + 1e-12)
            G = Xn @ Xn.T
            within_cos[L, ci] = float(G[np.triu_indices(n_probe, 1)].mean())
            tot_ss += float(np.sum((cx_L[ci] - grand) ** 2))
        eta2[L] = float(1.0 - with_ss / tot_ss)

        try:
            ss = silhouette_samples(X, labels, metric="euclidean")
            sil_global[L] = float(ss.mean())
            for ci in range(n_ctx):
                sil_cond[L, ci] = float(ss[labels == ci].mean())
        except Exception:
            pass

        pca = PCA(n_components=min(PCA_IN, X.shape[0] - 1, X.shape[1]), random_state=RNG)
        Xp = pca.fit_transform(X)
        ccp = pca.transform(cc_L)

        # targets Y (2400, nB) = r_B . a_x ; v0_dot (50,nB) = r_B . v0(C)
        Yt = np.stack([ax_L.reshape(n_ctx * n_probe, -1) @ rB[bi, L] for bi in range(nB)], axis=1)
        v0_dot = np.stack([v0_mean[:, L, :] @ rB[bi, L] for bi in range(nB)], axis=1)

        (pred_probe, pred_cc), train_pred = fit_mlp_multihead(Xp, Yt, [Xp, ccp])
        probe_mean = pred_probe.reshape(n_ctx, n_probe, nB).mean(1)  # (50,nB)
        J_b = np.abs(probe_mean - pred_cc).T  # (nB,50)
        R_b = np.abs(v0_dot - pred_cc).T
        mlp_corr = {
            behaviors[bi]: float(np.corrcoef(train_pred[:, bi], Yt[:, bi])[0, 1])
            for bi in range(nB)
        }

        Jlin_b = np.zeros((nB, n_ctx))
        Rlin_loco_b = np.zeros((nB, n_ctx))
        alphas = {}
        for bi in range(nB):
            y = Yt[:, bi]
            a = ridge_select_alpha(Xp, y)
            alphas[behaviors[bi]] = float(a)
            lin_probe = ridge_fit_predict(Xp, y, Xp, a).reshape(n_ctx, n_probe).mean(1)
            lin_cc = ridge_fit_predict(Xp, y, ccp, a)
            Jlin_b[bi] = np.abs(lin_probe - lin_cc)
            for ci in range(n_ctx):
                tr = labels != ci
                pcc = ridge_fit_predict(Xp[tr], y[tr], ccp[ci : ci + 1], a)[0]
                Rlin_loco_b[bi, ci] = abs(v0_dot[ci, bi] - pcc)

        J_max[L] = J_b.max(0)
        R_max[L] = R_b.max(0)
        Jlin_max[L] = Jlin_b.max(0)
        Rlin_loco_max[L] = Rlin_loco_b.max(0)

        per_fam = {}
        for fam in sorted(set(families)):
            m = fam_arr == fam
            if m.sum() >= 4:
                per_fam[fam] = {
                    "n": int(m.sum()),
                    "mean_spread_W": float(spread_W[L, m].mean()),
                    "mean_J": float(J_max[L, m].mean()),
                    "mean_R": float(R_max[L, m].mean()),
                    "spearman_sW_J": sp(spread_W[L, m], J_max[L, m]),
                    "spearman_sW_R": sp(spread_W[L, m], R_max[L, m]),
                }
        slope = float(np.polyfit(spread_W[L], J_max[L], 1)[0])
        r2 = float(np.corrcoef(spread_W[L], J_max[L])[0, 1] ** 2)
        results["per_layer"][str(L)] = {
            "lambda_whiten": float(lam),
            "eta2": float(eta2[L]),
            "silhouette_global": float(sil_global[L]),
            "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
            "mlp_fitqual_corr": mlp_corr,
            "ridge_alpha": alphas,
            "spearman_sW_J_all": sp(spread_W[L], J_max[L]),
            "spearman_sW_R_all": sp(spread_W[L], R_max[L]),
            "spearman_sW_Rlin_loco_all": sp(spread_W[L], Rlin_loco_max[L]),
            "spearman_sI_J_all": sp(spread_I[L], J_max[L]),
            "spearman_within_cos_R_all": sp(within_cos[L], R_max[L]),
            "ols_slope_J_on_sW": slope,
            "ols_r2_J_on_sW": r2,
            "linear_jensen_gap_max": float(Jlin_max[L].max()),
            "per_family": per_fam,
        }
        finalize_and_save()
        print(
            f"[layer {L:2d}] eta2={eta2[L]:.3f} sil={sil_global[L]:.3f} "
            f"rho(sW,J)={results['per_layer'][str(L)]['spearman_sW_J_all']['rho']:+.3f} "
            f"rho(sW,R)={results['per_layer'][str(L)]['spearman_sW_R_all']['rho']:+.3f} "
            f"mlpQ={np.mean(list(mlp_corr.values())):.2f} "
            f"linJgap={Jlin_max[L].max():.1e}",
            flush=True,
        )

    print("\nOVERALL:", json.dumps(results["overall"], indent=1))
    print(
        "FAMILY whitened spread:",
        {
            f: round(results["family_summary"][f]["mean_spread_W_overall"], 3)
            for f in results["family_summary"]
        },
    )
    print("N flags:", results["overall"]["n_low_coherence_flags"])
    print("SAVED", os.path.join(OUT_JSON, "coherence_results.json"))


if __name__ == "__main__":
    main()
