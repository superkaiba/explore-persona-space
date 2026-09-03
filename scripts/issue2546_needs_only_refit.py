"""Refit the Section 4.5 within-corpus maps and the reparameterization ladder on needs-reasoning rows only.

Rows: the #2546 needs-reasoning stratum (all of MATH, GSM8K rows with 4+ steps, ContextHub levels 3-4),
taken from the committed pooled preds (p7_A__does__a{arm}.npz conv_ids) and restricted per corpus.
Ridge mirrors the production recipe: standardize X on train (std+1e-9), center Y, primal ridge in float64,
five seeded random-row folds (seed 0), lambda chosen per outer fold by 4-fold inner CV on the production
grid logspace(-3, 8, 23) (pooled inner out-of-fold R^2). Ladder tiers t0/t3/t4 follow issue2546_ladder_audit.py.
Retrieval follows recipe_pipeline.py (train-fold whitening, shrinkage 0.1, cosine + CSLS k=10, own answer).

Usage: needs_only_rerun.py <unit>   where unit in {math, gsm8k_train, contexthub, pooled}
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # repo convention: environment before heavy imports

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

sys.path.insert(0, "/home/thomasjiralerspong/explore-persona-space/src")
from explore_persona_space.analysis.null_battery import PRIMARY_LAMBDA, shrunk_cholesky_from_cov

BASE = Path("/mnt/eps-data/thomasjiralerspong/cot_necessity")
HF = BASE / "hf" / "issue2546_cotmap"
TG = BASE / "hf" / "targets"
OUT = BASE / "needs_only"
(OUT / "cells").mkdir(parents=True, exist_ok=True)
(OUT / "ladder").mkdir(exist_ok=True)
LAYER = {1: 19, 3: 24}
POST = {1: "post", 3: "think_on"}
PRE = {1: "pre"}
LAMBDAS = np.logspace(-3.0, 8.0, 23)
N_FOLDS, SEED, N_INNER, N_BOOT, K_CSLS = 5, 0, 4, 1000, 10
FIXED_LAMBDA = {"pre": 1000.0, "post": 3162.2776601683795}  # production-selected values at layer 19 (p8_G pre; p7_A/p8_F post); used when NEEDS_ONLY_FIXED_LAMBDA=1
CORPORA = ("math", "gsm8k_train", "contexthub")
torch.set_num_threads(int(os.environ.get("THREADS", "8")))
unit_name = sys.argv[1]
log = open(OUT / f"log_{unit_name}.txt", "a")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def cv_folds(ids, n_folds, seed):
    uniq = np.unique(np.asarray(ids))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {c: int(perm[i] % n_folds) for i, c in enumerate(uniq)}
    return np.array([fold_of[c] for c in ids])


def stratum_ids(arm, corpora):
    z = np.load(HF / "analysis_tensors" / "preds" / f"arm{arm}" / f"p7_A__does__a{arm}.npz")
    ids = [str(r) for r in z["conv_ids"][np.asarray(z["fitted_mask"], bool)]]
    return [r for r in ids if r.split(":")[0] in corpora]


def load_kind(arm, side, kind, corpora):
    out = {}
    for c in corpora:
        z = np.load(TG / f"{kind}__arm{arm}__{side}__{c}__l{LAYER[arm]}.npz")
        for r, v in zip(z["row_ids"], z[kind]):
            out[str(r)] = v
    return out


def matrix(d, ids):
    return torch.as_tensor(np.stack([d[r] for r in ids]).astype(np.float64))


class Ridge:
    """Primal ridge on standardized X with eigendecomposition so every lambda on the grid is cheap."""

    def __init__(self, X, Y):
        self.xmu, self.xsd = X.mean(0), X.std(0) + 1e-9
        self.ymu = Y.mean(0)
        Xn = (X - self.xmu) / self.xsd
        s, V = torch.linalg.eigh(Xn.T @ Xn)
        self.s, self.V, self.C = s, V, V.T @ (Xn.T @ (Y - self.ymu))

    def predict(self, Xe, lam):
        Z = ((Xe - self.xmu) / self.xsd) @ self.V
        return (Z / (self.s + lam)) @ self.C + self.ymu


def r2(pred, true):
    return 1.0 - float(((true - pred) ** 2).sum()) / float(((true - true.mean(0)) ** 2).sum())


def select_lambda(X, Y, ids):
    inner = cv_folds(ids, N_INNER, SEED + 1)
    res = np.zeros(len(LAMBDAS))
    tot = 0.0
    for j in range(N_INNER):
        tr, te = inner != j, inner == j
        m = Ridge(X[tr], Y[tr])
        yte = Y[te]
        tot += float(((yte - yte.mean(0)) ** 2).sum())
        for li, lam in enumerate(LAMBDAS):
            res[li] += float(((yte - m.predict(X[te], lam)) ** 2).sum())
    scores = 1.0 - res / tot
    return float(LAMBDAS[int(np.argmax(scores))]), scores


def fit_oof(X, Y, ids, folds, tag):
    """Out-of-fold predictions with per-fold inner-CV lambda; returns preds, per-fold lambdas, fitted models."""
    pred = torch.zeros_like(Y)
    lams = []
    models = []
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        t0 = time.time()
        fixed = FIXED_LAMBDA.get("pre" if "pre_own" in tag else "post") if os.environ.get("NEEDS_ONLY_FIXED_LAMBDA") else None
        lam = fixed if fixed is not None else select_lambda(X[tr], Y[tr], [r for r, m in zip(ids, tr) if m])[0]
        m = Ridge(X[tr], Y[tr])
        pred[te] = m.predict(X[te], lam)
        lams.append(lam)
        models.append((m, lam))
        say(
            f"  {tag} fold {k}: lambda={lam:.3g} R2_fold={r2(pred[te], Y[te]):.4f} ({time.time() - t0:.0f}s)"
        )
    return pred, lams, models


def boot_r2(pred, true, folds, rng):
    e = ((true - pred) ** 2).sum(1).numpy()
    t = np.zeros(len(true))
    for k in np.unique(folds):
        te = folds == k
        t[te] = ((true[te] - true[te].mean(0)) ** 2).sum(1).numpy()
    n = len(e)
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
    draws = 1.0 - (w @ e) / (w @ t)
    return {
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "n_draws": N_BOOT,
    }


def csls(S, k=K_CSLS):
    nq, npool = S.shape
    r_q = np.partition(S, npool - k, axis=1)[:, npool - k :].mean(1)
    r_p = np.partition(S, nq - k, axis=0)[nq - k :, :].mean(0)
    return 2.0 * S - r_q[:, None] - r_p[None, :]


unit = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
whiten = lambda x, mu, ell: solve_triangular(ell, (x - mu).T, lower=True).T


def retrieval(pred, Y, folds, rng):
    pred, Y = pred.numpy(), Y.numpy()
    hits = np.zeros(len(Y), bool)
    pools = []
    for k in np.unique(folds):
        tr = folds != k
        te = np.where(folds == k)[0]
        mu = Y[tr].mean(0)
        ell = shrunk_cholesky_from_cov(np.cov(Y[tr], rowvar=False), PRIMARY_LAMBDA)
        zq, zP = whiten(pred[te], mu, ell), whiten(Y[te], mu, ell)
        hits[te] = csls(unit(zq) @ unit(zP).T).argmax(1) == np.arange(len(te))
        pools.append(len(te))
    n = len(hits)
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT)
    m = (w @ hits.astype(float)) / n
    return {
        "acc1_whitened_csls": float(hits.mean()),
        "acc1_whitened_csls_ci": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
        "chance": float(np.mean([1.0 / p for p in pools])),
        "n": int(n),
    }, hits


def save_cell(arm, cell, corpus, ids, folds, pred, Y, lams, rng, extra=None):
    layer = LAYER[arm]
    stem = f"{cell}__{corpus}__a{arm}"
    np.savez(
        OUT / "preds" / f"arm{arm}" / f"{stem}.npz",
        fitted_mask=np.ones(len(ids), bool),
        conv_ids=np.asarray(ids),
        folds=folds,
        **{f"pred_l{layer}": pred.numpy().astype(np.float32)},
    )
    ret, hits = retrieval(pred, Y, folds, rng)
    np.savez(
        OUT / "preds" / f"arm{arm}" / f"hits__{stem}.npz",
        row_ids=np.asarray(ids),
        folds=folds,
        hit_whitened_csls=hits,
    )
    out = {
        "status": "ok",
        "unit_id": stem,
        "arm": arm,
        "cell": cell,
        "subset": f"needs_reasoning:{corpus}",
        "n_rows": len(ids),
        "n_fitted": len(ids),
        "headline_layer": layer,
        "r2_headline": r2(pred, Y),
        "r2_headline_bootstrap": boot_r2(pred, Y, folds, rng),
        "lambda_per_fold": lams,
        "retrieval": ret,
        "recipe": "needs-reasoning rows only (MATH all; GSM8K 4+ steps; ContextHub L3-4); ridge standardize-X/center-Y, 5 seeded folds, 4-fold inner-CV lambda on logspace(-3,8,23); retrieval = paper recipe",
    }
    if extra:
        out.update(extra)
    (OUT / "cells" / f"{stem}.json").write_text(json.dumps(out, indent=1))
    say(
        f"  {stem}: R2={out['r2_headline']:.4f} [{out['r2_headline_bootstrap']['ci_lo']:.3f},{out['r2_headline_bootstrap']['ci_hi']:.3f}] acc@1={ret['acc1_whitened_csls']:.3f} n={len(ids)}"
    )
    return out


def run_corpus(corpus):
    for arm in (1, 3):
        (OUT / "preds" / f"arm{arm}").mkdir(parents=True, exist_ok=True)
        ids = stratum_ids(arm, (corpus,))
        folds = cv_folds(ids, N_FOLDS, SEED)
        rng = np.random.default_rng(1)
        Y = matrix(load_kind(arm, POST[arm], "ans_mean", (corpus,)), ids)
        say(f"arm{arm} {corpus}: {len(ids)} needs-reasoning rows")
        for cell, kind in (("p7_A", "cx_last"), ("p7_D", "cot_boundary")):
            X = matrix(load_kind(arm, POST[arm], kind, (corpus,)), ids)
            pred, lams, _ = fit_oof(X, Y, ids, folds, f"arm{arm}/{cell}/{corpus}")
            save_cell(arm, cell, corpus, ids, folds, pred, Y, lams, rng)
    ladder(corpus, (corpus,))


def ladder(key, corpora):
    """Arm 1 only: fit the pre-SFT map (pre cx_last -> pre ans_mean) and transfer it to the post-SFT model."""
    arm = 1
    ids = stratum_ids(arm, corpora)
    folds = cv_folds(ids, N_FOLDS, SEED)
    rng = np.random.default_rng(2)
    Xb, Yb = (
        matrix(load_kind(arm, PRE[arm], "cx_last", corpora), ids),
        matrix(load_kind(arm, PRE[arm], "ans_mean", corpora), ids),
    )
    Xi, Yi = (
        matrix(load_kind(arm, POST[arm], "cx_last", corpora), ids),
        matrix(load_kind(arm, POST[arm], "ans_mean", corpora), ids),
    )
    say(f"ladder {key}: {len(ids)} rows")
    pre_pred, pre_lams, pre_models = fit_oof(Xb, Yb, ids, folds, f"ladder/{key}/pre_own")
    post_pred, post_lams, _ = fit_oof(Xi, Yi, ids, folds, f"ladder/{key}/post_own")
    if key == "pooled":
        save_cell(arm, "p8_G", key, ids, folds, pre_pred, Yb, pre_lams, rng)
        save_cell(arm, "p7_A", key, ids, folds, post_pred, Yi, post_lams, rng)
    tiers = {
        "t0_direct_transfer": torch.zeros_like(Yi),
        "t3_bias_offset": torch.zeros_like(Yi),
        "t4_global_scaling": torch.zeros_like(Yi),
    }
    alphas = []
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        m, lam = pre_models[k]
        p_tr, p_te = m.predict(Xi[tr], lam), m.predict(Xi[te], lam)
        y_tr = Yi[tr]
        tiers["t0_direct_transfer"][te] = p_te
        tiers["t3_bias_offset"][te] = p_te + (y_tr - p_tr).mean(0)
        pc, yc = p_tr - p_tr.mean(0), y_tr - y_tr.mean(0)
        alpha = float((pc * yc).sum() / ((pc * pc).sum() + 1e-12))
        alphas.append(alpha)
        tiers["t4_global_scaling"][te] = alpha * (p_te - p_tr.mean(0)) + y_tr.mean(0)
    ref = r2(post_pred, Yi)
    tiers_r2 = {t: r2(p, Yi) for t, p in tiers.items()}
    # question-level bootstrap of retention (tier R^2 / post own R^2), fold means fixed
    e_ref = ((Yi - post_pred) ** 2).sum(1).numpy()
    tot = np.zeros(len(ids))
    for k in range(N_FOLDS):
        te = folds == k
        tot[te] = ((Yi[te] - Yi[te].mean(0)) ** 2).sum(1).numpy()
    n = len(ids)
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
    ref_draws = 1.0 - (w @ e_ref) / (w @ tot)
    retention = {}
    for t, p in tiers.items():
        e = ((Yi - p) ** 2).sum(1).numpy()
        draws = (1.0 - (w @ e) / (w @ tot)) / ref_draws
        retention[t] = {
            "point": tiers_r2[t] / ref,
            "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)),
        }
    out = {
        "status": "ok",
        "unit_id": f"ladder__{key}__a1",
        "arm": 1,
        "subset": key,
        "ladder_corpus": key,
        "n_rows": n,
        "headline_layer": 19,
        "tiers_r2": tiers_r2,
        "tier_names": list(tiers_r2),
        "within_post_reference_r2": ref,
        "within_pre_own_r2": r2(pre_pred, Yb),
        "retention": retention,
        "alpha_per_fold": alphas,
        "lambda_pre_per_fold": pre_lams,
        "lambda_post_per_fold": post_lams,
        "recipe": "needs-reasoning rows only; pre map = pre cx_last -> pre ans_mean lambda " + ("fixed to the production-selected values (pre 1000, post 3162.3)" if os.environ.get("NEEDS_ONLY_FIXED_LAMBDA") else "by inner CV") + "; tiers per issue2546_ladder_audit.py",
    }
    (OUT / "ladder" / f"ladder__{key}__a1.json").write_text(json.dumps(out, indent=1))
    say(
        f"  ladder {key}: post own R2={ref:.4f} pre own R2={out['within_pre_own_r2']:.4f} retention "
        + " ".join(f"{t}={v['point']:.3g}" for t, v in retention.items())
    )


if unit_name == "pooled":
    ladder("pooled", CORPORA)
else:
    run_corpus(unit_name)
say(f"DONE {unit_name}")
