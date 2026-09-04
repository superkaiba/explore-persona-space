"""Context vs end-of-thought states, and pre vs post context states (issue #2546).

Inputs (read-only, /mnt/eps-data/thomasjiralerspong/cot_necessity):
  hf/targets/{cx_last,cot_boundary,ans_mean}__arm1__{pre,post}__{dataset}__l19.npz
      float16 states (n x 3584) + row_ids; the pre model has no cot_boundary.
  allfit/preds/p7_{A,D}__all__a1.npz
      the 30,193 fitted rows, the shared 5 folds, and the out-of-fold predictions of
      metamodel A (context -> answer, penalty 3162) and D (end of thought -> answer,
      penalty 316).
  allfit/preds/hits__p7_{A,D}__all__a1.npz  recorded retrieval hits (sanity anchor).

Analyses (stages, each writes a stage JSON to the scratch dir):
  a1   d = h_eot - h_cx: relative norms + cosines (per dataset), mean-offset vs
       question-specific variance split, effective rank, share in top-50 PCs of h_cx.
  a2a  ridge h_cx -> h_eot (5-fold, penalty sweep 100/316/1000/3162), identity-plus-bias
       baseline, kNN retrieval read (mapping_baselines.knn_retrieval).
  a2b  h_ans from the residual r = h_eot - predicted(h_cx) alone.
  a2c  h_ans from [predicted, r] (7168-dim input).
  a3a  operator comparison of metamodels A and D refit on all rows: flattened-operator
       cosine + random-rotation null, one-sided Procrustes-aligned cosines, spectrum
       cosine (two-sided Procrustes optimum, rotation invariant), top-k singular
       subspace overlaps (right = input read, left = output written), effective ranks.
  a3b  out-of-fold cross-application (A on h_eot, D on h_cx) with dataset-mean R^2 and
       whitened-CSLS top-1 retrieval (train-fold whitening, shrinkage 0.1, CSLS k=10,
       pool = held-out fold's true answer states), next to the own scores.
  b1   Delta = h_cx(post) - h_cx(pre): relative norms + cosines, offset + global
       scaling + question-specific split (fit on training folds), effective rank,
       share in top-50 PCs of h_cx(pre); same decomposition for the answer states.
  b2   ridge h_cx(pre) -> h_cx(post): sweep, baseline, retrieval.
  b3   share of ||Delta||^2 in the top-k right singular subspace of W_pre
       (h_cx(pre) -> h_ans(pre), penalty 1000, full refit) vs the isotropic null k/d.
  figs 4 matplotlib figures to figures/issue_2546/eot_diffs/.
  merge  assemble eval_results/issue_2546/allfit/eot_vs_context/diffs/diffs.json.

Conventions: operators in column convention y = M x, so right singular vectors span the
input directions the map reads. Ridge standardizes inputs and centers outputs, matching
allfit_necessity.Ridge (the production fits); Gram matrices and predictions run in
float32 GEMMs (the shared VM is heavily contended) with float64 Cholesky solves and
float64 accumulation of every reported statistic. n_train (~24k) > d (3584 or 7168), so
every fit is well posed. Folds are reused from the prediction files so numbers are
comparable with the paper.

Usage: .venv/bin/python scripts/issue2546_cx_eot_prepost_diffs.py --stage <name>
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.analysis.null_battery import (  # noqa: E402
    PRIMARY_LAMBDA,
    shrunk_cholesky_from_cov,
)
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

BASE = Path("/mnt/eps-data/thomasjiralerspong/cot_necessity")
TG = BASE / "hf" / "targets"
PRED_DIR = BASE / "allfit" / "preds"
REPO = Path(__file__).resolve().parents[1]
DIFFS = REPO / "eval_results" / "issue_2546" / "allfit" / "eot_vs_context" / "diffs"
FIGS = REPO / "figures" / "issue_2546" / "eot_diffs"
SCRATCH = Path("/tmp/issue2546_cxdiffs_scratch")
DATASETS = ("math", "gsm8k_train", "contexthub", "mmlu", "arc_challenge", "csqa", "piqa")
N_FOLDS = 5
D_MODEL = 3584
LAM_A, LAM_D, LAM_PRE = 3162.0, 316.0, 1000.0
SWEEP = (100.0, 316.0, 1000.0, 3162.0)
KS = (1, 5, 10)
K_CSLS = 10
SEED = 20260903


def say(*a):
    print(" ".join(str(x) for x in a), flush=True)


def write_stage(name, payload):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    p = SCRATCH / f"stage_{name}.json"
    with atomic_replace(p) as tmp:
        tmp.write_text(json.dumps(payload, indent=1, default=float))
    say("wrote", p)


def read_stage(name):
    return json.loads((SCRATCH / f"stage_{name}.json").read_text())


def load_preds(cell):
    z = np.load(PRED_DIR / f"{cell}__all__a1.npz")
    ids = z["conv_ids"].astype(str)
    return ids, np.asarray(z["folds"]), np.asarray(z["pred_l19"], np.float32)


def load_target(kind, side, ids):
    """Vectorized id-aligned load of one state kind across the seven datasets."""
    out = np.empty((len(ids), D_MODEL), np.float32)
    filled = np.zeros(len(ids), bool)
    for c in DATASETS:
        z = np.load(TG / f"{kind}__arm1__{side}__{c}__l19.npz")
        rid = z["row_ids"].astype(str)
        order = np.argsort(rid)
        pos = np.searchsorted(rid[order], ids)
        pos_c = np.clip(pos, 0, len(rid) - 1)
        hit = rid[order][pos_c] == ids
        out[hit] = z[kind][order[pos_c[hit]]].astype(np.float32)
        filled |= hit
    assert filled.all(), f"{kind}/{side}: {int((~filled).sum())} ids missing"
    return out


def context():
    ids, folds, pA = load_preds("p7_A")
    ds = np.char.partition(ids, ":")[:, 0]
    return ids, folds, ds, pA


def sq_sum(x):
    """Float64-accumulated sum of squares of a float32 array."""
    return float(np.square(np.asarray(x), dtype=np.float64).sum())


def _standardize_train(Xtr32):
    xmu = Xtr32.mean(0, dtype=np.float64)
    xc = Xtr32 - xmu.astype(np.float32)
    xsd = np.sqrt(np.square(xc, dtype=np.float64).mean(0)) + 1e-9
    xc /= xsd.astype(np.float32)
    return xc, xmu, xsd


def _ridge_solve(G64, XtY64, lam):
    G = G64.copy()
    G[np.diag_indices_from(G)] += lam
    # numpy-only solve: scipy's vendored OpenBLAS pool spin-fights numpy's on this VM
    return np.linalg.solve(G, XtY64)


class RidgeFit:
    """Standardized-input, centered-output ridge (matches allfit_necessity.Ridge).

    float32 Gram + cross-covariance GEMMs, float64 Cholesky solve.
    """

    def __init__(self, X, Y, lam):
        X32 = np.asarray(X, np.float32)
        n, d = X32.shape
        assert n > d, (n, d)
        Xn, self.xmu, self.xsd = _standardize_train(X32)
        Y32 = np.asarray(Y, np.float32)
        self.ymu = Y32.mean(0, dtype=np.float64)
        Yc = Y32 - self.ymu.astype(np.float32)
        G = (Xn.T @ Xn).astype(np.float64)
        XtY = (Xn.T @ Yc).astype(np.float64)
        self.B = _ridge_solve(G, XtY, lam)
        self.lam = lam

    def predict(self, Xe):
        Xn = (np.asarray(Xe, np.float32) - self.xmu.astype(np.float32)) / self.xsd.astype(
            np.float32
        )
        return Xn @ self.B.astype(np.float32) + self.ymu.astype(np.float32)

    def raw_operator(self):
        """W with pred = (x - xmu) @ W + ymu, in the raw residual-stream basis."""
        return self.B / self.xsd[:, None]


def fit_oof(X, Y, folds, lams, tag):
    """Out-of-fold ridge predictions for every penalty in lams (shared Gram per fold)."""
    preds = {lam: np.empty((X.shape[0], Y.shape[1]), np.float32) for lam in lams}
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        Xtr = np.asarray(X[tr], np.float32)
        assert Xtr.shape[0] > Xtr.shape[1], Xtr.shape
        Xn, xmu, xsd = _standardize_train(Xtr)
        del Xtr
        Ytr = np.asarray(Y[tr], np.float32)
        ymu = Ytr.mean(0, dtype=np.float64)
        Yc = Ytr - ymu.astype(np.float32)
        del Ytr
        G0 = (Xn.T @ Xn).astype(np.float64)
        XtY = (Xn.T @ Yc).astype(np.float64)
        del Xn, Yc
        Xte = (np.asarray(X[te], np.float32) - xmu.astype(np.float32)) / xsd.astype(np.float32)
        for lam in lams:
            B32 = _ridge_solve(G0, XtY, lam).astype(np.float32)
            preds[lam][te] = Xte @ B32 + ymu.astype(np.float32)
        del Xte, G0, XtY
        say(f"[{tag}] fold {k} done")
    return preds


def oof_r2(Y, pred, folds, ds):
    """Pooled out-of-fold R^2 against the pooled train mean and the per-dataset train mean."""
    sse = sst_pool = sst_ds = 0.0
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        Yte = np.asarray(Y[te], np.float32)
        sse += sq_sum(Yte - np.asarray(pred[te], np.float32))
        mu = Y[tr].mean(0, dtype=np.float64).astype(np.float32)
        sst_pool += sq_sum(Yte - mu)
        for c in DATASETS:
            muc = Y[tr & (ds == c)].mean(0, dtype=np.float64).astype(np.float32)
            sst_ds += sq_sum(np.asarray(Y[te & (ds == c)], np.float32) - muc)
    return {"r2_pooled_mean": 1.0 - sse / sst_pool, "r2_dataset_mean": 1.0 - sse / sst_ds}


def identity_bias_oof(X, Y, folds):
    pred = np.empty((X.shape[0], Y.shape[1]), np.float32)
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        pred[te] = identity_bias_predict(X[tr], Y[tr], X[te]).astype(np.float32)
    return pred


def knn_read(pred, Y, folds):
    """Fold-pooled knn_retrieval (pool = held-out fold's true targets), both metrics."""
    out = {}
    for metric in ("euclidean", "cosine"):
        acc = dict.fromkeys(KS, 0.0)
        mrr = n_tot = 0.0
        pools = []
        for k in range(N_FOLDS):
            te = np.where(folds == k)[0]
            r = knn_retrieval(pred[te], Y[te], ks=KS, metric=metric)
            w = float(len(te))
            n_tot += w
            pools.append(r["n_pool"])
            mrr += r["mrr"] * w
            for kk in KS:
                acc[kk] += r["acc_at_k"][kk] * w
        out[metric] = {
            "acc_at_k": {kk: acc[kk] / n_tot for kk in KS},
            "chance_at_k": {kk: kk / float(np.mean(pools)) for kk in KS},
            "mrr": mrr / n_tot,
            "mean_pool": float(np.mean(pools)),
        }
        say(f"[knn] metric {metric} done")
    return out


def unit(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def csls_adjust(S, k=K_CSLS):
    nq, npool = S.shape
    r_q = np.partition(S, npool - k, axis=1)[:, npool - k :].mean(1)
    r_p = np.partition(S, nq - k, axis=0)[nq - k :, :].mean(0)
    return 2.0 * S - r_q[:, None] - r_p[None, :]


def whitened_csls_hitrate(preds_by_name, Y, folds):
    """Paper retrieval recipe: train-fold whitening (shrinkage 0.1), CSLS k=10,
    pool = held-out fold's true answer states, hit = nearest is own."""
    hits = {nm: np.zeros(Y.shape[0], bool) for nm in preds_by_name}
    for k in range(N_FOLDS):
        tr, te = folds != k, np.where(folds == k)[0]
        Ytr = np.asarray(Y[tr], np.float32)
        mu = Ytr.mean(0, dtype=np.float64)
        Yc = Ytr - mu.astype(np.float32)
        cov = (Yc.T @ Yc).astype(np.float64) / (Ytr.shape[0] - 1)
        del Ytr, Yc
        ell32 = shrunk_cholesky_from_cov(cov, PRIMARY_LAMBDA).astype(np.float32)
        mu32 = mu.astype(np.float32)
        zP = unit(np.linalg.solve(ell32, (Y[te] - mu32).T).T)
        for nm, P in preds_by_name.items():
            zq = unit(np.linalg.solve(ell32, (np.asarray(P[te], np.float32) - mu32).T).T)
            hits[nm][te] = csls_adjust((zq @ zP.T).astype(np.float64)).argmax(1) == np.arange(
                len(te)
            )
        say(f"[csls] fold {k} done")
    return {nm: float(h.mean()) for nm, h in hits.items()}


def rownorm2(x):
    return np.einsum("ij,ij->i", x, x, dtype=np.float64)


def relnorm_cos(a, b):
    """Row-wise ||b - a|| / ||a|| and cos(a, b), float64 accumulation."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    na = np.sqrt(rownorm2(a))
    nb = np.sqrt(rownorm2(b))
    rel = np.sqrt(rownorm2(b - a)) / na
    cos = np.einsum("ij,ij->i", a, b, dtype=np.float64) / (na * nb)
    return rel, cos


def per_ds_median(vals, ds):
    out = {c: float(np.median(vals[ds == c])) for c in DATASETS}
    out["all"] = float(np.median(vals))
    return out


def cov_eig_desc(Z):
    """Eigenvalues (descending) + eigenvectors of the covariance of Z (float32 path)."""
    Zc = np.asarray(Z, np.float32)
    Zc = Zc - Zc.mean(0, dtype=np.float64).astype(np.float32)
    cov = (Zc.T @ Zc) / np.float32(Zc.shape[0])
    w, V = np.linalg.eigh(cov)
    return np.clip(w[::-1].astype(np.float64), 0.0, None), V[:, ::-1]


def eff_rank(evals):
    cum = np.cumsum(evals) / evals.sum()
    er90 = int(np.searchsorted(cum, 0.90) + 1)
    p = evals / evals.sum()
    p = p[p > 0]
    return er90, float(np.exp(-(p @ np.log(p)))), cum


def share_in(Z, V):
    Z32 = np.atleast_2d(np.asarray(Z, np.float32))
    proj = Z32 @ np.asarray(V, np.float32)
    return float(np.square(proj, dtype=np.float64).sum() / sq_sum(Z32))


def mean_split(D):
    """Total sum of squares of D split into the shared mean offset and the rest."""
    D32 = np.asarray(D, np.float32)
    mu = D32.mean(0, dtype=np.float64)
    tot = sq_sum(D32)
    ss_mean = float(D32.shape[0] * (mu**2).sum())
    return {"total_ss": tot, "mean_offset_share": ss_mean / tot,
            "question_specific_share": 1.0 - ss_mean / tot}


def random_orthogonal(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
    return q * np.sign(np.diag(r))


def vec_cos(A, B):
    num = np.einsum("ij,ij->", A, B, dtype=np.float64)
    return float(num / (np.sqrt(sq_sum(A)) * np.sqrt(sq_sum(B))))


def nuclear(K):
    return float(np.linalg.svd(K, compute_uv=False).astype(np.float64).sum())


# ---------------------------------------------------------------------------- stages
def stage_a1():
    ids, folds, ds, _ = context()
    cx = load_target("cx_last", "post", ids)
    eot = load_target("cot_boundary", "post", ids)
    say("[a1] loaded")
    rel, cos = relnorm_cos(cx, eot)
    d = eot - cx
    split = mean_split(d)
    dc = d - d.mean(0, dtype=np.float64).astype(np.float32)
    evals, _ = cov_eig_desc(dc)
    er90, er_ent, cum = eff_rank(evals)
    say("[a1] d eig done")
    evc, Vc = cov_eig_desc(cx)
    say("[a1] cx eig done")
    V50 = Vc[:, :50]
    mu_off = eot.mean(0, dtype=np.float64) - cx.mean(0, dtype=np.float64)
    res = {
        "n": int(len(ids)),
        "relnorm_median": per_ds_median(rel, ds),
        "cos_median": per_ds_median(cos, ds),
        "cos_mean": float(cos.mean()),
        "variance_split": split,
        "qspec_effective_rank_90pct": er90,
        "qspec_effective_rank_entropy": er_ent,
        "qspec_cumvar_curve": cum[:2000].tolist(),
        "share_of_qspec_in_top50_pcs_of_cx": share_in(dc, V50),
        "share_of_mean_offset_in_top50_pcs_of_cx": share_in(mu_off, V50),
        "chance_share_top50": 50.0 / D_MODEL,
        "cx_effective_rank_90pct": eff_rank(evc)[0],
    }
    write_stage("a1", res)


def stage_a2a():
    ids, folds, ds, _ = context()
    cx = load_target("cx_last", "post", ids)
    eot = load_target("cot_boundary", "post", ids)
    preds = fit_oof(cx, eot, folds, SWEEP, "a2a cx->eot")
    stats = {lam: oof_r2(eot, preds[lam], folds, ds) for lam in SWEEP}
    best = max(SWEEP, key=lambda lam: stats[lam]["r2_pooled_mean"])
    pred = preds[best]
    ib = identity_bias_oof(cx, eot, folds)
    say("[a2a] fits done, retrieval next")
    res = {
        "sweep_r2": {str(lam): stats[lam] for lam in SWEEP},
        "selected_penalty": best,
        "ridge": stats[best],
        "identity_bias": oof_r2(eot, ib, folds, ds),
        "knn_ridge": knn_read(pred, eot, folds),
        "knn_identity_bias": knn_read(ib, eot, folds),
    }
    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.savez(SCRATCH / "pred_eot_oof.npz", pred=pred.astype(np.float16))
    write_stage("a2a", res)


def stage_a2b():
    ids, folds, ds, _ = context()
    eot = load_target("cot_boundary", "post", ids)
    ans = load_target("ans_mean", "post", ids)
    pred = np.load(SCRATCH / "pred_eot_oof.npz")["pred"].astype(np.float32)
    r = eot - pred
    del eot
    preds = fit_oof(r, ans, folds, SWEEP, "a2b r->ans")
    stats = {lam: oof_r2(ans, preds[lam], folds, ds) for lam in SWEEP}
    best = max(SWEEP, key=lambda lam: stats[lam]["r2_pooled_mean"])
    write_stage("a2b", {"sweep_r2": {str(lam): stats[lam] for lam in SWEEP},
                        "selected_penalty": best, "residual_to_ans": stats[best]})


def stage_a2c():
    ids, folds, ds, _ = context()
    eot = load_target("cot_boundary", "post", ids)
    ans = load_target("ans_mean", "post", ids)
    pred = np.load(SCRATCH / "pred_eot_oof.npz")["pred"].astype(np.float32)
    XR = np.concatenate([pred, eot - pred], axis=1)
    del eot, pred
    preds = fit_oof(XR, ans, folds, SWEEP, "a2c [pred,r]->ans")
    stats = {lam: oof_r2(ans, preds[lam], folds, ds) for lam in SWEEP}
    best = max(SWEEP, key=lambda lam: stats[lam]["r2_pooled_mean"])
    write_stage("a2c", {"sweep_r2": {str(lam): stats[lam] for lam in SWEEP},
                        "selected_penalty": best, "pred_plus_residual_to_ans": stats[best]})


def stage_a3a():
    ids, folds, ds, _ = context()
    cx = load_target("cx_last", "post", ids)
    eot = load_target("cot_boundary", "post", ids)
    ans = load_target("ans_mean", "post", ids)
    MA = RidgeFit(cx, ans, LAM_A).raw_operator().T.astype(np.float32)  # y = M x
    MD = RidgeFit(eot, ans, LAM_D).raw_operator().T.astype(np.float32)
    del cx, eot, ans
    say("[a3a] operators fit")
    rng = np.random.default_rng(SEED)
    raw = vec_cos(MA, MD)
    draws = []
    for i in range(12):
        q1 = random_orthogonal(D_MODEL, rng)
        q2 = random_orthogonal(D_MODEL, rng)
        draws.append(vec_cos(MA, q1 @ MD @ q2))
    say("[a3a] rotation null done")
    fa, fd = np.sqrt(sq_sum(MA)), np.sqrt(sq_sum(MD))
    UA, sA, VtA = np.linalg.svd(MA)
    UD, sD, VtD = np.linalg.svd(MD)
    sA = sA.astype(np.float64)
    sD = sD.astype(np.float64)
    say("[a3a] svds done")
    spectrum_cos = float((sA * sD).sum() / (np.linalg.norm(sA) * np.linalg.norm(sD)))
    overlaps = {}
    for k in (10, 50, 200):
        sub = {}
        for name, F1, F2 in (("right_input", VtA.T, VtD.T), ("left_output", UA, UD)):
            s = np.linalg.svd(F1[:, :k].T @ F2[:, :k], compute_uv=False).astype(np.float64)
            null = []
            for _ in range(5):
                q, _r = np.linalg.qr(rng.standard_normal((D_MODEL, k)).astype(np.float32))
                null.append(float(np.linalg.svd(F1[:, :k].T @ q, compute_uv=False).mean()))
            sub[name] = {"mean_principal_cos": float(s.mean()),
                         "sq_projection_share": float((s**2).sum() / k),
                         "null_mean_principal_cos": float(np.mean(null))}
        overlaps[str(k)] = sub
    erA = eff_rank(np.clip(sA, 0, None) ** 2)
    erD = eff_rank(np.clip(sD, 0, None) ** 2)
    res = {
        "penalties": {"A": LAM_A, "D": LAM_D},
        "raw_operator_cosine": raw,
        "rotation_null": {"n_draws": 12, "mean": float(np.mean(draws)),
                          "std": float(np.std(draws)),
                          "p975": float(np.quantile(draws, 0.975)),
                          "analytic_sd_1_over_d": 1.0 / D_MODEL},
        "procrustes_aligned_cosine_input_rotation": nuclear(MA.T @ MD) / (fa * fd),
        "procrustes_aligned_cosine_output_rotation": nuclear(MD @ MA.T) / (fa * fd),
        "spectrum_cosine_two_sided_procrustes_optimum": spectrum_cos,
        "spectrum_cosine_note": "rotation-invariant; cannot support 'same operator up to "
                                "rotation'",
        "subspace_overlaps": overlaps,
        "effective_rank_90pct": {"A": erA[0], "D": erD[0]},
        "effective_rank_entropy": {"A": erA[1], "D": erD[1]},
        "operator_frobenius": {"A": float(fa), "D": float(fd)},
        "singular_values_A": sA.tolist(),
        "singular_values_D": sD.tolist(),
    }
    write_stage("a3a", res)


def stage_a3b():
    ids, folds, ds, pA = context()
    _, _, pD = load_preds("p7_D")
    cx = load_target("cx_last", "post", ids)
    eot = load_target("cot_boundary", "post", ids)
    ans = load_target("ans_mean", "post", ids)
    predA_on_eot = np.empty_like(pA)
    predD_on_cx = np.empty_like(pA)
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        RA = RidgeFit(cx[tr], ans[tr], LAM_A)
        predA_on_eot[te] = RA.predict(eot[te])
        del RA
        RD = RidgeFit(eot[tr], ans[tr], LAM_D)
        predD_on_cx[te] = RD.predict(cx[te])
        del RD
        say(f"[a3b] fold {k} fits done")
    conds = {"A_own_cx": pA, "D_own_eot": pD, "A_on_eot": predA_on_eot, "D_on_cx": predD_on_cx}
    r2 = {nm: oof_r2(ans, P, folds, ds) for nm, P in conds.items()}
    hit = whitened_csls_hitrate(conds, ans, folds)
    anchors = {}
    for cell in ("p7_A", "p7_D"):
        z = np.load(PRED_DIR / f"hits__{cell}__all__a1.npz")
        rid = z["row_ids"].astype(str)
        order = np.argsort(rid)
        pos = np.searchsorted(rid[order], ids)
        h = np.asarray(z["hit_whitened_csls"], bool)[order[pos]]
        anchors[cell] = float(h.mean())
    write_stage("a3b", {"r2": r2, "top1_whitened_csls": hit,
                        "recorded_hit_anchor": anchors,
                        "retrieval_recipe": "train-fold whitening shrinkage 0.1, CSLS k=10, "
                                            "pool = held-out fold true answer states"})


def stage_b1():
    ids, folds, ds, _ = context()
    out = {}
    curves = {}
    for side, kind in (("context", "cx_last"), ("answer", "ans_mean")):
        pre = load_target(kind, "pre", ids)
        post = load_target(kind, "post", ids)
        rel, cos = relnorm_cos(pre, post)
        ss_tot = ss_off = ss_scale = 0.0
        s_folds = []
        for k in range(N_FOLDS):
            tr, te = folds != k, folds == k
            Ptr, Qtr = pre[tr], post[tr]
            b_off = Qtr.mean(0, dtype=np.float64) - Ptr.mean(0, dtype=np.float64)
            pmu, qmu = Ptr.mean(0, dtype=np.float64), Qtr.mean(0, dtype=np.float64)
            Pc = Ptr - pmu.astype(np.float32)
            Qc = Qtr - qmu.astype(np.float32)
            s = float(np.einsum("ij,ij->", Pc, Qc, dtype=np.float64) / sq_sum(Pc))
            s_folds.append(s)
            del Pc, Qc
            Pte, Qte = pre[te], post[te]
            Dte = Qte - Pte
            ss_tot += sq_sum(Dte)
            ss_off += sq_sum(Dte - b_off.astype(np.float32))
            b2 = (qmu - s * pmu).astype(np.float32)
            ss_scale += sq_sum(Qte - (np.float32(s) * Pte + b2))
        pmu, qmu = pre.mean(0, dtype=np.float64), post.mean(0, dtype=np.float64)
        Pc = pre - pmu.astype(np.float32)
        s_all = float(
            np.einsum("ij,ij->", Pc, post - qmu.astype(np.float32), dtype=np.float64) / sq_sum(Pc)
        )
        del Pc
        resid = post - (np.float32(s_all) * pre + (qmu - s_all * pmu).astype(np.float32))
        evals, _ = cov_eig_desc(resid)
        er90, er_ent, cum = eff_rank(evals)
        evp, Vp = cov_eig_desc(pre)
        V50 = Vp[:, :50]
        delta = post - pre
        mu_d = qmu - pmu
        out[side] = {
            "relnorm_median": per_ds_median(rel, ds),
            "cos_median": per_ds_median(cos, ds),
            "oof_split": {"ss_total": ss_tot,
                          "mean_offset_share": 1.0 - ss_off / ss_tot,
                          "global_scaling_extra_share": (ss_off - ss_scale) / ss_tot,
                          "question_specific_share": ss_scale / ss_tot,
                          "scale_s_per_fold": s_folds, "scale_s_all_rows": s_all},
            "qspec_effective_rank_90pct": er90,
            "qspec_effective_rank_entropy": er_ent,
            "share_of_delta_in_top50_pcs_of_pre": share_in(delta, V50),
            "share_of_centered_delta_in_top50_pcs_of_pre": share_in(
                delta - mu_d.astype(np.float32), V50
            ),
            "share_of_mean_offset_in_top50_pcs_of_pre": share_in(mu_d, V50),
            "chance_share_top50": 50.0 / D_MODEL,
            "pre_effective_rank_90pct": eff_rank(evp)[0],
        }
        curves[side] = cum[:2000].tolist()
        del delta, resid
        say(f"[b1] side {side} done")
    out["qspec_cumvar_curves"] = curves
    write_stage("b1", out)


def stage_b2():
    ids, folds, ds, _ = context()
    pre = load_target("cx_last", "pre", ids)
    post = load_target("cx_last", "post", ids)
    preds = fit_oof(pre, post, folds, SWEEP, "b2 pre->post cx")
    stats = {lam: oof_r2(post, preds[lam], folds, ds) for lam in SWEEP}
    best = max(SWEEP, key=lambda lam: stats[lam]["r2_pooled_mean"])
    ib = identity_bias_oof(pre, post, folds)
    write_stage("b2", {
        "sweep_r2": {str(lam): stats[lam] for lam in SWEEP},
        "selected_penalty": best,
        "ridge": stats[best],
        "identity_bias": oof_r2(post, ib, folds, ds),
        "knn_ridge": knn_read(preds[best], post, folds),
        "knn_identity_bias": knn_read(ib, post, folds),
    })


def stage_b3():
    ids, folds, ds, _ = context()
    pre_cx = load_target("cx_last", "pre", ids)
    pre_ans = load_target("ans_mean", "pre", ids)
    W = RidgeFit(pre_cx, pre_ans, LAM_PRE)
    Mpre = W.raw_operator().T.astype(np.float32)
    del pre_ans
    _, s_pre, Vt = np.linalg.svd(Mpre)
    s_pre = s_pre.astype(np.float64)
    V = np.ascontiguousarray(Vt.T)  # columns = input directions W_pre reads, descending
    say("[b3] svd done")
    post_cx = load_target("cx_last", "post", ids)
    delta = post_cx - pre_cx
    del pre_cx, post_cx
    proj = delta @ V
    colss = np.square(proj, dtype=np.float64).sum(0)
    del proj
    tot = sq_sum(delta)
    cumshare = np.cumsum(colss) / tot
    mu_d = delta.mean(0, dtype=np.float64)
    proj_mu = (mu_d @ V.astype(np.float64)) ** 2
    cumshare_mu = np.cumsum(proj_mu) / float((mu_d**2).sum())
    deltac = delta - mu_d.astype(np.float32)
    cumshare_c = np.cumsum(np.square(deltac @ V, dtype=np.float64).sum(0)) / sq_sum(deltac)
    del deltac
    rng = np.random.default_rng(SEED + 1)
    null_shares = {str(k): [] for k in (50, 200, 1000)}
    for _ in range(3):
        g = rng.standard_normal((delta.shape[0], D_MODEL)).astype(np.float32)
        gc = np.cumsum(np.square(g @ V, dtype=np.float64).sum(0)) / sq_sum(g)
        for k in (50, 200, 1000):
            null_shares[str(k)].append(float(gc[k - 1]))
        del g
    say("[b3] nulls done")
    res = {
        "penalty_w_pre": LAM_PRE,
        "w_pre_effective_rank_90pct": eff_rank(np.clip(s_pre, 0, None) ** 2)[0],
        "share_delta_sq_in_topk_right_subspace": {str(k): float(cumshare[k - 1])
                                                  for k in (50, 200, 1000)},
        "share_mean_offset_in_topk": {str(k): float(cumshare_mu[k - 1]) for k in (50, 200, 1000)},
        "share_centered_delta_sq_in_topk_right_subspace": {
            str(k): float(cumshare_c[k - 1]) for k in (50, 200, 1000)
        },
        "null_isotropic_k_over_d": {str(k): k / float(D_MODEL) for k in (50, 200, 1000)},
        "null_matched_norm_draws": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                                    for k, v in null_shares.items()},
        "cumshare_curve": cumshare.tolist(),
        "cumshare_centered_curve": cumshare_c.tolist(),
        "singular_values_w_pre": s_pre.tolist(),
    }
    write_stage("b3", res)


def stage_diag():
    """Massive-coordinate diagnostics: which raw coordinates carry the shifts."""
    ids, folds, ds, _ = context()

    def block(diff, name):
        mu = diff.mean(0, dtype=np.float64)
        top_mu = np.argsort(-np.abs(mu))[:5]
        dc = diff - mu.astype(np.float32)
        var = np.square(dc, dtype=np.float64).mean(0)
        top_v = np.argsort(-var)[:5]
        evals, _V = cov_eig_desc(dc)
        return {
            "mean_offset_norm": float(np.linalg.norm(mu)),
            "top5_abs_mean_dims": {str(int(i)): float(mu[i]) for i in top_mu},
            "share_mean_sq_in_top3_dims": float((mu[top_mu[:3]] ** 2).sum() / (mu**2).sum()),
            "top5_var_dims": {str(int(i)): float(var[i]) for i in top_v},
            "share_var_in_top4_dims": float(var[top_v[:4]].sum() / var.sum()),
            "pc1_variance_share": float(evals[0] / evals.sum()),
        }

    cx_post = load_target("cx_last", "post", ids)
    eot = load_target("cot_boundary", "post", ids)
    res = {"d_eot_minus_cx": block(eot - cx_post, "d")}
    del eot
    var_cx = cx_post.var(0, dtype=np.float64)
    top = np.argsort(-var_cx)[:5]
    res["cx_post_top5_var_dims"] = {str(int(i)): float(var_cx[i]) for i in top}
    res["cx_post_top1_var_share"] = float(var_cx[top[0]] / var_cx.sum())
    cx_pre = load_target("cx_last", "pre", ids)
    res["delta_cx"] = block(cx_post - cx_pre, "delta_cx")
    del cx_post, cx_pre
    ans_pre = load_target("ans_mean", "pre", ids)
    ans_post = load_target("ans_mean", "post", ids)
    res["delta_ans"] = block(ans_post - ans_pre, "delta_ans")
    write_stage("diag", res)


def stage_figs():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    a1, a3a, b1, b3 = read_stage("a1"), read_stage("a3a"), read_stage("b1"), read_stage("b3")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sA = np.asarray(a3a["singular_values_A"])
    sD = np.asarray(a3a["singular_values_D"])
    sP = np.asarray(b3["singular_values_w_pre"])
    ax = axes[0]
    ax.loglog(np.arange(1, len(sA) + 1), sA / sA[0], label="metamodel A (context to answer)")
    ax.loglog(np.arange(1, len(sD) + 1), sD / sD[0],
              label="metamodel D (end of thought to answer)")
    ax.loglog(np.arange(1, len(sP) + 1), sP / sP[0], label="W_pre (pre context to answer)")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value / largest")
    ax.set_title("Operator spectra")
    ax.legend(fontsize=8)
    ax = axes[1]
    ks = [10, 50, 200]
    for name, label in (("right_input", "right (input read)"),
                        ("left_output", "left (output written)")):
        vals = [a3a["subspace_overlaps"][str(k)][name]["mean_principal_cos"] for k in ks]
        ax.plot(ks, vals, marker="o", label=label)
    nulls = [a3a["subspace_overlaps"][str(k)]["right_input"]["null_mean_principal_cos"]
             for k in ks]
    ax.plot(ks, nulls, marker="x", linestyle="--", label="random subspace null")
    ax.set_xscale("log")
    ax.set_xlabel("subspace size k")
    ax.set_ylabel("mean principal cosine")
    ax.set_ylim(0, 1)
    ax.set_title("A vs D singular subspace overlap")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "a3_operator_spectra_subspaces.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, cum in (
        ("d = eot - context (question specific)", a1["qspec_cumvar_curve"]),
        ("Delta context post - pre (question specific)", b1["qspec_cumvar_curves"]["context"]),
        ("Delta answer post - pre (question specific)", b1["qspec_cumvar_curves"]["answer"]),
    ):
        c = np.asarray(cum)
        ax.semilogx(np.arange(1, len(c) + 1), c, label=label)
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative variance share")
    ax.set_title("Effective rank of the question-specific shifts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "qspec_cumulative_variance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(DATASETS))
    w = 0.27
    series = [
        ("end of thought vs context (post)", [a1["relnorm_median"][c] for c in DATASETS]),
        ("context post vs pre", [b1["context"]["relnorm_median"][c] for c in DATASETS]),
        ("answer post vs pre", [b1["answer"]["relnorm_median"][c] for c in DATASETS]),
    ]
    for i, (label, vals) in enumerate(series):
        ax.bar(x + (i - 1) * w, vals, w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=25, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("median relative shift  ||diff|| / ||reference|| (log)")
    ax.set_title("Per-dataset state shifts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "per_dataset_relative_shifts.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    c = np.asarray(b3["cumshare_curve"])
    kk = np.arange(1, len(c) + 1)
    ax.plot(kk, c, label="Delta context (post - pre), raw")
    cc = np.asarray(b3["cumshare_centered_curve"])
    ax.plot(kk, cc, label="Delta context, question-centered")
    ax.plot(kk, kk / float(D_MODEL), linestyle="--", label="isotropic null k/3584")
    for k in (50, 200, 1000):
        ax.axvline(k, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("top-k right singular directions of W_pre")
    ax.set_ylabel("share of squared norm of Delta")
    ax.set_title("Where Delta lies relative to what W_pre reads")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "b3_wpre_subspace_share.png", dpi=150)
    plt.close(fig)
    say("figures written to", FIGS)


def stage_merge():
    payload = {
        "setup": {
            "models": {"post": "OpenThinker3-7B (arm 1)", "pre": "Qwen2.5-7B-Instruct (arm 1)"},
            "layer": 19,
            "n_questions": 30193,
            "datasets": list(DATASETS),
            "generation": "one greedy rollout per question",
            "folds": "5 folds reused from allfit preds p7_A/p7_D",
            "production_penalties": {"A_cx_to_ans": LAM_A, "D_eot_to_ans": LAM_D,
                                     "W_pre_cx_to_ans": LAM_PRE},
            "penalty_sweep": list(SWEEP),
            "ridge": "standardized inputs, centered outputs; float32 GEMMs with float64 "
                     "Cholesky solves and float64 accumulation; n_train ~24,154 per fold vs "
                     "d 3,584 (7,168 for [pred, r]), well posed",
            "retrieval_recipe_a3": "train-fold whitening (shrinkage 0.1), CSLS k=10, pool = "
                                   "held-out fold true answer states, hit = nearest is own",
            "script": "scripts/issue2546_cx_eot_prepost_diffs.py",
        },
        "A1_input_state_difference": read_stage("a1"),
        "A2_cx_to_eot_linearity": {"map": read_stage("a2a"),
                                   "residual_to_ans": read_stage("a2b"),
                                   "pred_plus_residual_to_ans": read_stage("a2c")},
        "A3_operator_comparison": {"operators": read_stage("a3a"),
                                   "cross_application": read_stage("a3b")},
        "B1_prepost_context_shift": read_stage("b1"),
        "B2_pre_to_post_linearity": read_stage("b2"),
        "B3_delta_vs_wpre_subspace": read_stage("b3"),
        "massive_coordinate_diagnostics": read_stage("diag"),
    }
    payload["A1_input_state_difference"].pop("qspec_cumvar_curve", None)
    payload["A3_operator_comparison"]["operators"].pop("singular_values_A", None)
    payload["A3_operator_comparison"]["operators"].pop("singular_values_D", None)
    payload["B1_prepost_context_shift"].pop("qspec_cumvar_curves", None)
    b3 = payload["B3_delta_vs_wpre_subspace"]
    b3.pop("singular_values_w_pre", None)
    b3["cumshare_at"] = {str(k): b3["cumshare_curve"][k - 1] for k in (10, 50, 200, 1000, 2000)}
    b3["cumshare_centered_at"] = {
        str(k): b3["cumshare_centered_curve"][k - 1] for k in (10, 50, 200, 1000, 2000)
    }
    b3.pop("cumshare_curve", None)
    b3.pop("cumshare_centered_curve", None)
    DIFFS.mkdir(parents=True, exist_ok=True)
    with atomic_replace(DIFFS / "diffs.json") as tmp:
        tmp.write_text(json.dumps(payload, indent=1, default=float))
    say("wrote", DIFFS / "diffs.json")


STAGES = {
    "a1": stage_a1, "a2a": stage_a2a, "a2b": stage_a2b, "a2c": stage_a2c,
    "a3a": stage_a3a, "a3b": stage_a3b, "b1": stage_b1, "b2": stage_b2, "b3": stage_b3,
    "diag": stage_diag, "figs": stage_figs, "merge": stage_merge,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    args = ap.parse_args()
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
