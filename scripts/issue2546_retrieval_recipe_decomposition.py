import numpy as np, sys
from pathlib import Path
from scipy.linalg import solve_triangular
sys.path.insert(0, "/home/thomasjiralerspong/explore-persona-space/src"); from explore_persona_space.analysis.null_battery import PRIMARY_LAMBDA, shrunk_cholesky_from_cov
HF = Path("/mnt/eps-data/thomasjiralerspong/cot_necessity/hf/issue2546_cotmap"); TG = HF.parent / "targets"
def csls(S, k=10):
    nq, npool = S.shape; r_q = np.partition(S, npool-k, axis=1)[:, npool-k:].mean(1); r_p = np.partition(S, nq-k, axis=0)[nq-k:, :].mean(0); return 2*S - r_q[:, None] - r_p[None, :]
unit = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
whiten = lambda x, mu, ell: solve_triangular(ell, (x - mu).T, lower=True).T
z = np.load(TG / "ans_mean__arm1__math__l19.npz"); pos = {str(r): i for i, r in enumerate(z["row_ids"])}; Yall = z["ans_mean"].astype(np.float64)
for cell in ("p7_A", "p7_D"):
    p = np.load(HF / "analysis_tensors/preds/arm1" / f"{cell}__math__a1.npz"); fitted = p["fitted_mask"]; ids = [str(r) for r in p["conv_ids"][fitted]]; pred = p["pred_l19"][fitted].astype(np.float64); folds = p["folds"][fitted]
    Y = Yall[[pos[r] for r in ids]]; n = len(ids)
    v = Y.var(0); print(f"{cell}: dims 458+2570 hold {100*(v[458]+v[2570])/v.sum():.1f}% of answer-state variance; {100*np.mean((Y[:,458]**2+Y[:,2570]**2)/(Y**2).sum(1)):.1f}% of squared norm", flush=True)
    keys = ["fold: raw cos", "fold: raw cos + CSLS", "fold: cos w/o 2 massive dims", "fold: per-dim standardized cos", "fold: whitened cos (train stats)", "fold: whitened cos + CSLS (train stats)", "fold: whitened cos + CSLS (ALL-row stats)", "whole: raw cos", "whole: whitened cos + CSLS (train stats)", "whole: whitened cos + CSLS (ALL-row stats)"]
    acc = {k: [] for k in keys}
    mu_all = Y.mean(0); ell_all = shrunk_cholesky_from_cov(np.cov(Y, rowvar=False), PRIMARY_LAMBDA); zY_all = whiten(Y, mu_all, ell_all)
    keep = np.ones(Y.shape[1], bool); keep[[458, 2570]] = False; sd = Y.std(0) + 1e-6; mu = Y.mean(0)
    for k in np.unique(folds):
        tr = folds != k; te = np.where(folds == k)[0]; t = np.arange(len(te))
        mu_tr = Y[tr].mean(0); ell_tr = shrunk_cholesky_from_cov(np.cov(Y[tr], rowvar=False), PRIMARY_LAMBDA)
        q, P = pred[te], Y[te]
        S = unit(q) @ unit(P).T; acc["fold: raw cos"].append(S.argmax(1) == t); acc["fold: raw cos + CSLS"].append(csls(S).argmax(1) == t)
        acc["fold: cos w/o 2 massive dims"].append((unit(q[:, keep]) @ unit(P[:, keep]).T).argmax(1) == t)
        acc["fold: per-dim standardized cos"].append((unit((q - mu) / sd) @ unit((P - mu) / sd).T).argmax(1) == t)
        zq, zP = whiten(q, mu_tr, ell_tr), whiten(P, mu_tr, ell_tr); Sw = unit(zq) @ unit(zP).T
        acc["fold: whitened cos (train stats)"].append(Sw.argmax(1) == t); acc["fold: whitened cos + CSLS (train stats)"].append(csls(Sw).argmax(1) == t)
        zq_a = whiten(q, mu_all, ell_all); acc["fold: whitened cos + CSLS (ALL-row stats)"].append(csls(unit(zq_a) @ unit(zY_all[te]).T).argmax(1) == t)
        acc["whole: raw cos"].append((unit(q) @ unit(Y).T).argmax(1) == te)
        acc["whole: whitened cos + CSLS (train stats)"].append(csls(unit(zq) @ unit(whiten(Y, mu_tr, ell_tr)).T).argmax(1) == te)
        acc["whole: whitened cos + CSLS (ALL-row stats)"].append(csls(unit(zq_a) @ unit(zY_all).T).argmax(1) == te)
        print(f"   fold {k} done", flush=True)
    for k2 in keys: print(f"   {k2:44s} acc@1 = {np.concatenate(acc[k2]).mean():.3f}", flush=True)
print("DONE")
