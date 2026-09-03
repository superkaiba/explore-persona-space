"""Strict own-answer (identity) retrieval for #2546 maps under the paper's Section 4.1 recipe.

For each out-of-fold predicted answer vector, the hit requires the nearest pool
vector to be the question's OWN answer. Metrics: whitened cosine + CSLS (k=10;
whitening stats fit on the train-fold answers with shrinkage 0.1), whitened
cosine, raw cosine, raw euclidean. Two pool conventions: the held-out fold's
answers (Section 4.1) and the whole corpus (#2546 cells).
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
ROOT = Path("/home/thomasjiralerspong/explore-persona-space"); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from explore_persona_space.analysis.null_battery import PRIMARY_LAMBDA, shrunk_cholesky_from_cov
K_CSLS = 10
HF = Path("/mnt/eps-data/thomasjiralerspong/cot_necessity/hf/issue2546_cotmap"); TG = Path("/mnt/eps-data/thomasjiralerspong/cot_necessity/hf/targets")
ARMS = {1: 19, 3: 24}; CORPORA = ["math", "gsm8k_train", "contexthub", "mmlu"]; CELLS = ["p7_A", "p7_D"]

def csls(S, k=K_CSLS):
    nq, npool = S.shape
    r_q = np.partition(S, npool - k, axis=1)[:, npool - k:].mean(1)
    r_p = np.partition(S, nq - k, axis=0)[nq - k:, :].mean(0)
    return 2.0 * S - r_q[:, None] - r_p[None, :]
def unit(a): return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
def whiten(x, mu, ell): return solve_triangular(ell, (x - mu).T, lower=True).T
def acc1(score_hi_is_better, true_idx): return float(np.mean(np.argmax(score_hi_is_better, axis=1) == true_idx))

out = {}
t0 = time.time()
for arm, layer in ARMS.items():
    for corpus in CORPORA:
        z = np.load(TG / f"ans_mean__arm{arm}__{corpus}__l{layer}.npz"); t_ids = [str(r) for r in z["row_ids"]]; Y_all = z["ans_mean"].astype(np.float64); pos = {r: i for i, r in enumerate(t_ids)}
        for cell in CELLS:
            p = np.load(HF / "analysis_tensors" / "preds" / f"arm{arm}" / f"{cell}__{corpus}__a{arm}.npz")
            fitted = np.asarray(p["fitted_mask"], bool); ids = [str(r) for r in p["conv_ids"][fitted]]; pred = np.asarray(p[f"pred_l{layer}"][fitted], np.float64); folds = np.asarray(p["folds"][fitted])
            keep = np.array([r in pos for r in ids]); ids = [r for r, k in zip(ids, keep) if k]; pred = pred[keep]; folds = folds[keep]
            Y = Y_all[[pos[r] for r in ids]]; n = len(ids)
            hits = {conv: {m: [] for m in ("whiten_csls", "whiten_cos", "cosine", "euclidean")} for conv in ("fold_pool", "whole_pool")}; pool_sizes = []
            for k in np.unique(folds):
                tr = folds != k; te = np.where(folds == k)[0]
                mu = Y[tr].mean(0); ell = shrunk_cholesky_from_cov(np.cov(Y[tr], rowvar=False), PRIMARY_LAMBDA)
                zq = whiten(pred[te], mu, ell); zY = whiten(Y, mu, ell)
                for conv, pool_idx, true_idx in (("fold_pool", te, np.arange(len(te))), ("whole_pool", np.arange(n), te)):
                    S_w = unit(zq) @ unit(zY[pool_idx]).T
                    S_c = unit(pred[te]) @ unit(Y[pool_idx]).T
                    D_e = ((pred[te] ** 2).sum(1)[:, None] + (Y[pool_idx] ** 2).sum(1)[None, :] - 2 * pred[te] @ Y[pool_idx].T)
                    hits[conv]["whiten_csls"].append(np.argmax(csls(S_w), 1) == true_idx)
                    hits[conv]["whiten_cos"].append(np.argmax(S_w, 1) == true_idx)
                    hits[conv]["cosine"].append(np.argmax(S_c, 1) == true_idx)
                    hits[conv]["euclidean"].append(np.argmin(D_e, 1) == true_idx)
                pool_sizes.append(len(te))
            res = {conv: {m: float(np.concatenate(v).mean()) for m, v in d.items()} for conv, d in hits.items()}
            res["n"] = n; res["chance_fold_pool"] = float(np.mean([1.0 / s for s in pool_sizes])); res["chance_whole_pool"] = 1.0 / n
            out[f"arm{arm}/{corpus}/{cell}"] = res
            print(f"arm{arm} {corpus:12s} {cell} n={n:5d} | fold pool (chance {res['chance_fold_pool']:.1e}): csls={res['fold_pool']['whiten_csls']:.3f} wcos={res['fold_pool']['whiten_cos']:.3f} cos={res['fold_pool']['cosine']:.3f} eucl={res['fold_pool']['euclidean']:.3f} | whole pool (chance {res['chance_whole_pool']:.1e}): csls={res['whole_pool']['whiten_csls']:.3f} wcos={res['whole_pool']['whiten_cos']:.3f} cos={res['whole_pool']['cosine']:.3f} eucl={res['whole_pool']['euclidean']:.3f} ({time.time()-t0:.0f}s)", flush=True)
Path("/mnt/eps-data/thomasjiralerspong/cot_necessity/retrieval_identity.json").write_text(json.dumps({"recipe": {"whitening": "train-fold answer covariance, shrinkage 0.1, Cholesky", "csls_k": K_CSLS, "hit": "nearest pool vector is the question's own answer"}, "results": out}, indent=2))
print("DONE")
