"""Independent audit of issue #2546 reparameterization-ladder tiers t0/t3/t4 (MATH, arm 1).

Reproduces from raw shards, independent of /tmp/cotfig/fit_cells.py:
  own-pre R^2, own-post R^2, t0 (direct transfer), t3 (bias refit), t4 (scalar+bias),
plus t0 variants (top-8 magnitude dims excluded; per-side per-dim standardization).

Ridge mirrors issue825_map_alignment's committed recipe: standardize X on train
(std+1e-9), center Y, primal ridge, float64. Lambda is FIXED to the value the
production cells selected at layer 19 (p8_G pre: 1000; p7_A/p8_F post: 3162.3)
instead of re-running inner-CV selection (compute constraint; stated deviation).
Solves use Cholesky on (X^T X + lam I). Folds replicate fit_cells._cv_folds
(seeded permutation of sorted unique row ids, seed 0, 5 folds).
Data cache: audit_cache_math_a1.npz (built from the raw bf16 shards).
"""
import numpy as np, torch, json, os

CACHE = "/mnt/eps-data/thomasjiralerspong/cot_necessity/audit_cache_math_a1.npz"
LAM_PRE, LAM_POST = 1000.0, 3162.2776601683795
N_FOLDS, SEED = 5, 0
torch.set_num_threads(os.cpu_count())

def cv_folds(conv_ids, n_folds, seed):
    uniq = np.unique(np.asarray(conv_ids))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {c: int(perm[i] % n_folds) for i, c in enumerate(uniq)}
    return np.array([fold_of[c] for c in conv_ids])

def fit_ridge(Xtr, Ytr, lam):
    xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xn = (Xtr - xmu) / xsd
    ymu = Ytr.mean(0)
    A = Xn.T @ Xn
    A.diagonal().add_(lam)
    L = torch.linalg.cholesky(A)
    B = torch.cholesky_solve(Xn.T @ (Ytr - ymu), L)   # (D, Dout)
    return dict(xmu=xmu, xsd=xsd, ymu=ymu, B=B)

def predict(m, Xe):
    return ((Xe - m["xmu"]) / m["xsd"]) @ m["B"] + m["ymu"]

def main():
    z = np.load(CACHE, allow_pickle=False)
    common = [str(r) for r in z["rows"]]
    T = lambda k: torch.as_tensor(z[k], dtype=torch.float64)
    Xb, Yb, Xi, Yi = T("Xb"), T("Yb"), T("Xi"), T("Yi")
    print(f"matched rows: {len(common)}", flush=True)

    mu_b, mu_i = Xb.mean(0), Xi.mean(0)
    drop = torch.argsort(torch.maximum(mu_b.abs(), mu_i.abs()), descending=True)[:8]
    keep = torch.as_tensor(np.setdiff1d(np.arange(Xb.shape[1]), drop.numpy()))
    print("dropped dims (top-8 magnitude):", drop.tolist(), flush=True)

    folds = torch.as_tensor(cv_folds(common, N_FOLDS, SEED))
    acc = {k: [0.0, 0.0] for k in ["own_pre", "own_post", "t0", "t3", "t4", "t0_std", "t0_dropk"]}
    def add(k, pred, true):
        acc[k][0] += float(((true - pred) ** 2).sum())
        acc[k][1] += float(((true - true.mean(0)) ** 2).sum())

    diag = {}
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        m_pre = fit_ridge(Xb[tr], Yb[tr], LAM_PRE)
        m_post = fit_ridge(Xi[tr], Yi[tr], LAM_POST)
        add("own_pre", predict(m_pre, Xb[te]), Yb[te])
        add("own_post", predict(m_post, Xi[te]), Yi[te])
        p_tr, p_te = predict(m_pre, Xi[tr]), predict(m_pre, Xi[te])
        y_tr, y_te = Yi[tr], Yi[te]
        add("t0", p_te, y_te)
        add("t3", p_te + (y_tr - p_tr).mean(0), y_te)
        pc, yc = p_tr - p_tr.mean(0), y_tr - y_tr.mean(0)
        alpha = float((pc * yc).sum() / ((pc * pc).sum() + 1e-12))
        add("t4", alpha * (p_te - p_tr.mean(0)) + y_tr.mean(0), y_te)
        if k == 0:
            diag["pred_norm_t0"] = float(p_te.norm(dim=1).median())
            diag["true_norm"] = float(y_te.norm(dim=1).median())
        def std(M):
            mu_, sd_ = M[tr].mean(0), M[tr].std(0) + 1e-9
            return (M - mu_) / sd_
        Xbz, Ybz, Xiz, Yiz = std(Xb), std(Yb), std(Xi), std(Yi)
        m_z = fit_ridge(Xbz[tr], Ybz[tr], LAM_PRE)
        add("t0_std", predict(m_z, Xiz[te]), Yiz[te])
        m_d = fit_ridge(Xb[tr][:, keep], Yb[tr][:, keep], LAM_PRE)
        add("t0_dropk", predict(m_d, Xi[te][:, keep]), Yi[te][:, keep])
        print(f"fold {k}: alpha={alpha:.3g}", flush=True)

    print("\n== out-of-fold R^2 (lam_pre=1000, lam_post=3162.3) ==", flush=True)
    out = {"median_pred_norm_t0_fold0": diag.get("pred_norm_t0"),
           "median_true_norm_fold0": diag.get("true_norm")}
    for k, (res, tot) in acc.items():
        out[k] = 1 - res / tot
        print(f"{k}: {out[k]:.6g}", flush=True)
    print("median |t0 pred| vs |true| (fold 0):", out["median_pred_norm_t0_fold0"], out["median_true_norm_fold0"], flush=True)
    json.dump(out, open("/mnt/eps-data/thomasjiralerspong/cot_necessity/audit_ladder_out.json", "w"), indent=1)

if __name__ == "__main__":
    main()
