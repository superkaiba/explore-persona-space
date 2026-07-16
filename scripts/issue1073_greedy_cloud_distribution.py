#!/usr/bin/env python3
"""Issue #1073 free-analysis: distribution of the stochastic-rollout cloud vs the greedy rollout.

For each context x and read-out layer, the pipeline has 10 stochastic answer-span mean-activation
vectors ``v_j(x)`` and one greedy vector ``v_greedy(x)``. This script characterises the cloud of
the 10 stochastic draws and locates the greedy vector relative to it, answering: is greedy
distributionally just an 11th draw, or systematically offset?

Analyses (all batched closed-form tensor reductions — 0 GPU-h, CPU):
  1. Exchangeability / rank test (headline). Treat greedy as an 11th rollout; per context compute
     each of the 11 items' cosine distance to the leave-self-out mean of the other 10, and rank
     greedy among the 11. Under "greedy is just another draw" the rank is uniform on 1..11.
     Report the per-layer rank histogram + chi-square vs uniform, the skew direction, and the
     most-central / most-peripheral fractions (uniform expectation 1/11 each).
  2. Systematic offset. u(x) = v_greedy(x) - mean_j v_j(x). (a) ||mean_x u(x)|| vs a sign-flip
     permutation null; (b) mean cos(u(x), u_bar) (offset-direction consistency); (c) alignment of
     u with the greedy-minus-mean-stoch response-length gap.
  3. Norms. Paired ||v_greedy|| vs the 10 ||v_j|| per layer (percentiles + paired median diff with
     bootstrap CI).
  4. Cloud descriptives. Per-context dispersion percentiles + greedy's distance-to-cloud-mean
     distribution overlaid on a typical draw's leave-one-out distance-to-mean.

Reuses the #1073 loaders (issue1073_common / issue1073_capture) and the gap_tail dispersion
computation (issue1073_gap_tail_analysis.rollout_dispersion) verbatim. The per-rollout store is
staged prefix-scoped onto /mnt/eps-data at the pinned revision (NEVER /, which is full).

NO raw prompt/completion text is loaded into context or written to any output — only structural
span-length counts (from the store) and activation reductions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue1073_capture as CAP  # noqa: E402
import issue1073_common as I  # noqa: E402
import issue1073_gap_tail_analysis as GT  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sstats  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("issue1073_greedy_cloud")

torch.set_num_threads(int(os.environ.get("EPS_VM_THREAD_CAP", "8")))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
READOUT_LAYERS = [14, 17, 19, 26, 27]
N_ROLLOUTS = 10
EPS = 1e-12

# Per-rollout store lives at THIS revision (distinct from issue1073_common.PINNED_REVISION,
# which pins the reused #779 inputs).
STORE_REV = "fb4fe90fdd836ba2efd896b90c17e6b42f143d21"
DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1073_decode_regime/analysis_tensors"
# Stage OFF the full boot disk (/ is 100%); the user-owned dir on the /mnt/eps-data data disk has
# headroom (the /mnt/eps-data root itself is root-owned — only the per-user subtree is writable).
_DATA_DISK = Path(os.environ.get("EPS_VM_DATA_DISK_PATH", "/mnt/eps-data"))
_STAGE_BASE = (
    _DATA_DISK / "thomasjiralerspong"
    if (_DATA_DISK / "thomasjiralerspong").is_dir()
    else _DATA_DISK
)
STAGE = Path(os.environ.get("EPS_I1073_STAGE", str(_STAGE_BASE / "issue1073_greedy_cloud")))
STAGE_HF = STAGE / "_hf"

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_1073"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_1073"

RNG_SEED = 0
N_SIGNFLIP = 2000
N_BOOT = 2000


# ── staging (prefix-scoped, pinned revision, off /) ─────────────────────────────


def stage_store(max_workers: int = 6) -> None:
    """Materialise the v_store shards + coverage.pt at STORE_REV under STAGE_HF (idempotent)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    want: list[str] = []
    entries = I._retry(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in issue1073_common._retry (bounded, re-raises)
            api.list_repo_tree(
                DATA_REPO,
                path_in_repo=f"{HF_PREFIX}/v_store",
                repo_type="dataset",
                revision=STORE_REV,
                recursive=False,
            )
        ),
        what="list_repo_tree v_store",
    )
    want.extend(e.path for e in entries if e.path.endswith(".pt"))
    want.append(f"{HF_PREFIX}/reductions/coverage.pt")

    def _one(path_in_repo: str) -> str:
        return I._retry(
            lambda: hf_hub_download(
                repo_id=DATA_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                revision=STORE_REV,
                local_dir=str(STAGE_HF),
            ),
            what=f"download {path_in_repo}",
        )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_one, want))
    logger.info("[stage] %d files under %s in %.1fs", len(want), STAGE_HF, time.time() - t0)

    # Point the reused gap_tail loaders at the staged store.
    GT.STORE_DIR = STAGE_HF / HF_PREFIX / "v_store"
    GT.COVERAGE = STAGE_HF / HF_PREFIX / "reductions" / "coverage.pt"


# ── greedy vector loader (mirrors GT.stoch_matrix — greedy loaded as an 11th rollout) ────────────


def greedy_matrix(li: int, keep: np.ndarray) -> np.ndarray:
    """(N_kept, H) fp64 greedy span-mean vectors at one layer, loaded fp16->fp64 from the greedy
    shards exactly as GT.stoch_matrix loads the stochastic draws, so greedy sits on bit-identical
    footing with the 10 stochastic rollouts."""
    pos_of = {int(ci): k for k, ci in enumerate(keep.tolist())}
    seen = np.zeros(len(keep), dtype=bool)
    v = None
    for p, shard in CAP.iter_shards(GT.STORE_DIR, "greedy"):
        li_pos = list(shard["layers"]).index(li)
        sl = shard["summ"][:, li_pos, :].to(torch.float64).numpy()
        if v is None:
            v = np.zeros((len(keep), sl.shape[1]))
        for row, (ci, _ri) in enumerate(shard["index"]):
            k = pos_of.get(int(ci))
            if k is not None:
                assert not seen[k], f"duplicate greedy ctx (ci={ci}) in {p}"
                seen[k] = True
                v[k] = sl[row]
    assert v is not None and bool(seen.all()), "greedy store fill incomplete"
    return v


# ── vectorised primitives ───────────────────────────────────────────────────────


def _unit(x: np.ndarray, axis: int) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + EPS)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < EPS or b.std() < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _percentiles(x: np.ndarray) -> dict:
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return {
        "mean": float(x.mean()),
        "p5": float(p[0]),
        "p25": float(p[1]),
        "median": float(p[2]),
        "p75": float(p[3]),
        "p95": float(p[4]),
    }


# ── Analysis 1: exchangeability / rank test ─────────────────────────────────────


def rank_test(greedy: np.ndarray, stoch: np.ndarray) -> dict:
    """greedy (n,H), stoch (n,10,H) fp32. Treat greedy as item 0 of 11; each item's distance to
    the leave-self-out mean of the OTHER 10; rank greedy among the 11 by ascending distance."""
    n = greedy.shape[0]
    X = np.concatenate([greedy[:, None, :], stoch], axis=1)  # (n, 11, H)
    k = X.shape[1]
    S = X.sum(1)  # (n, H)
    m = (S[:, None, :] - X) / (k - 1)  # (n, 11, H) leave-self-out mean of the other 10
    cos = np.einsum("nih,nih->ni", _unit(X, 2), _unit(m, 2))  # (n, 11)
    d = 1.0 - cos  # distance-to-loo-mean
    dg = d[:, 0]
    rank = 1 + (d < dg[:, None]).sum(1)  # 1..11, min-rank on ties (ties measure-zero in fp)
    hist = np.bincount(rank, minlength=k + 2)[1 : k + 1]  # bins 1..11
    exp = n / k
    chi2 = float(((hist - exp) ** 2 / exp).sum())
    p_chi2 = float(sstats.chi2.sf(chi2, df=k - 1))
    return {
        "n_contexts": int(n),
        "n_items": int(k),
        "rank_histogram": hist.astype(int).tolist(),
        "rank_fractions": (hist / n).tolist(),
        "uniform_expected_fraction": 1.0 / k,
        "chi2": chi2,
        "chi2_df": k - 1,
        "chi2_p_vs_uniform": p_chi2,
        "mean_rank": float(rank.mean()),
        "mean_rank_uniform_expected": (k + 1) / 2.0,
        "median_rank": float(np.median(rank)),
        "frac_greedy_most_central": float((np.argmin(d, axis=1) == 0).mean()),
        "frac_greedy_most_peripheral": float((np.argmax(d, axis=1) == 0).mean()),
        "frac_rank_le_5": float((rank <= 5).mean()),
        "frac_rank_ge_7": float((rank >= 7).mean()),
        "greedy_mean_dist_to_loo10": float(dg.mean()),
        "stoch_mean_dist_to_loo10": float(d[:, 1:].mean()),
        "skew": ("central" if rank.mean() < (k + 1) / 2.0 else "peripheral"),
    }


# ── Analysis 2: systematic offset ───────────────────────────────────────────────


def offset_test(greedy: np.ndarray, stoch: np.ndarray, len_gap: np.ndarray) -> dict:
    n = greedy.shape[0]
    u = greedy - stoch.mean(1)  # (n, H) per-context offset
    ubar = u.mean(0)  # (H,)
    ubar_norm = float(np.linalg.norm(ubar))
    uhat = ubar / (ubar_norm + EPS)

    # (a) sign-flip permutation null on ||mean_x u(x)||
    rng = np.random.default_rng(RNG_SEED)
    signs = rng.choice([-1.0, 1.0], size=(N_SIGNFLIP, n)).astype(np.float32)
    null_means = (signs @ u) / n  # (B, H)
    null_norms = np.linalg.norm(null_means, axis=1)
    p_norm = float((1 + (null_norms >= ubar_norm).sum()) / (N_SIGNFLIP + 1))
    exp_null_sq = float((np.linalg.norm(u, axis=1) ** 2).sum() / n**2)  # analytic E[||mean||^2]

    # (b) direction consistency
    cos_u_ubar = _unit(u, 1) @ uhat  # (n,) cos(u_x, ubar)
    proj = u @ uhat  # (n,) signed offset magnitude along the mean direction

    # (c) length-gap alignment: u(x) ~ intercept(ubar) + slope(b) * gap(x)
    gc = len_gap - len_gap.mean()
    denom = float((gc**2).sum())
    b = (gc[:, None] * (u - ubar)).sum(0) / (denom + EPS)  # (H,) length-response direction
    pred = ubar[None, :] + gc[:, None] * b[None, :]
    ss_res = float(((u - pred) ** 2).sum())
    ss_tot = float(((u - ubar) ** 2).sum())
    r2_u_gap = 1.0 - ss_res / (ss_tot + EPS)
    cos_b_ubar = float(_unit(b, 0) @ uhat)

    return {
        "mean_offset_norm": ubar_norm,
        "signflip_null_norm_mean": float(null_norms.mean()),
        "signflip_null_norm_p95": float(np.percentile(null_norms, 95)),
        "signflip_null_norm_max": float(null_norms.max()),
        "signflip_p_value": p_norm,
        "analytic_expected_null_norm": float(exp_null_sq**0.5),
        "coherence_ratio_obs_over_null": ubar_norm / (float(null_norms.mean()) + EPS),
        "mean_cos_u_ubar": float(cos_u_ubar.mean()),
        "median_cos_u_ubar": float(np.median(cos_u_ubar)),
        "frac_positive_projection": float((proj > 0).mean()),
        "mean_offset_norm_percontext": _percentiles(np.linalg.norm(u, axis=1)),
        "len_gap_mean": float(len_gap.mean()),
        "len_gap_median": float(np.median(len_gap)),
        "pearson_proj_vs_len_gap": _pearson(proj, len_gap),
        "r2_offset_explained_by_len_gap": float(r2_u_gap),
        "cos_lengthdir_vs_meanoffset": cos_b_ubar,
    }


# ── Analysis 3: norms ────────────────────────────────────────────────────────────


def norm_test(greedy: np.ndarray, stoch: np.ndarray) -> dict:
    gn = np.linalg.norm(greedy, axis=1)  # (n,)
    sn = np.linalg.norm(stoch, axis=2)  # (n, 10)
    sn_mean = sn.mean(1)  # (n,)
    paired = gn - sn_mean  # (n,)
    rng = np.random.default_rng(RNG_SEED)
    n = gn.shape[0]
    boot = np.array([np.median(paired[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    return {
        "greedy_norm": _percentiles(gn),
        "stoch_norm_pooled": _percentiles(sn.reshape(-1)),
        "stoch_norm_percontext_mean": _percentiles(sn_mean),
        "paired_greedy_minus_meanstoch": _percentiles(paired),
        "paired_median_diff": float(np.median(paired)),
        "paired_median_diff_ci95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "frac_greedy_norm_below_meanstoch": float((paired < 0).mean()),
    }


# ── Analysis 4: cloud descriptives ───────────────────────────────────────────────


def cloud_descriptives(greedy: np.ndarray, stoch: np.ndarray, disp: np.ndarray) -> dict:
    vbar10 = stoch.mean(1)  # (n, H)
    dg_cloud = 1.0 - np.einsum("nh,nh->n", _unit(greedy, 1), _unit(vbar10, 1))  # (n,)
    S = stoch.sum(1)  # (n, H)
    loo9 = (S[:, None, :] - stoch) / (N_ROLLOUTS - 1)  # (n, 10, H)
    d_s = 1.0 - np.einsum("nih,nih->ni", _unit(stoch, 2), _unit(loo9, 2))  # (n, 10) LOO distances
    paired = dg_cloud - d_s.mean(1)  # (n,)
    return {
        "rollout_dispersion": _percentiles(disp),
        "rollout_dispersion_iqr": float(np.percentile(disp, 75) - np.percentile(disp, 25)),
        "greedy_dist_to_cloud_mean": _percentiles(dg_cloud),
        "stoch_loo_dist_to_mean_pooled": _percentiles(d_s.reshape(-1)),
        "paired_greedy_minus_typical_dist": _percentiles(paired),
        "frac_greedy_farther_than_typical": float((paired > 0).mean()),
        "_hist_greedy_dist": dg_cloud.tolist(),
        "_hist_stoch_dist": d_s.reshape(-1).tolist(),
    }


# ── Analysis 5: distributional-form fit tests on the within-context rollout cloud ────
#
# Residuals r_j(x) = v_j(x) - mean_10(x), sqrt(10/9) own-mean bias-corrected so the per-draw
# variance is unbiased (mean_10 uses the same 10 draws). Two variants (the discriminating design):
#   (a) RAW pooled residuals — a scale MIXTURE over contexts (dispersion is context-dependent), so
#       it looks heavy-tailed even if each context's cloud is Gaussian.
#   (b) PER-CONTEXT SCALE-NORMALIZED — each context's residuals divided by its own RMS residual
#       magnitude sigma_x (the per-context scale whose mixture makes (a) heavy-tailed). If (a) fits
#       multivariate-t but (b) fits Gaussian => "per-context Gaussian with context-dependent scale".
# All claims are about the POOLED within-context distribution (n=10/context ⇒ no per-context
# normality claim); the covariance is pooled (homogeneous-shape assumption).

N_RAND_PROJ = 50
N_TOP_PCA = 20
PROJ_SEED = 42
N_CLUSTER_BOOT = 200
BIAS_CORR = float(np.sqrt(N_ROLLOUTS / (N_ROLLOUTS - 1)))  # sqrt(10/9) own-mean correction


def _residuals(stoch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n,10,H) stochastic vectors -> (resid (n,10,H) bias-corrected, sigma_x (n,) RMS scale)."""
    resid = (stoch - stoch.mean(1, keepdims=True)) * BIAS_CORR
    sigma_x = np.sqrt((resid**2).mean(axis=(1, 2)))  # per-context per-dim RMS residual magnitude
    return resid, sigma_x


def _ledoit_wolf(cov: np.ndarray, mean_row4: float, n: int, d: int) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of a mean-zero pooled covariance toward scaled identity.

    cov = R^T R / n (d,d); mean_row4 = mean_i ||r_i||^4. Returns (shrunk cov, shrinkage rho)."""
    mu = float(np.trace(cov)) / d
    cov_fro2 = float((cov**2).sum())
    d2 = cov_fro2 - d * mu**2  # ||cov - mu I||_F^2
    b_bar2 = (mean_row4 - cov_fro2) / n  # (1/n)(mean||r||^4 - ||cov||_F^2)
    b2 = min(max(b_bar2, 0.0), d2)
    rho = b2 / d2 if d2 > 0 else 0.0
    shrunk = cov.copy()
    shrunk *= 1.0 - rho
    shrunk[np.diag_indices(d)] += rho * mu
    return shrunk, float(rho)


def _cov_eig(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """R (N,d) fp32 mean-zero residuals -> (eigenvalues desc, eigenvectors, LW rho). fp64 eigh."""
    n, d = R.shape
    cov = (R.T @ R) / n  # fp32 matmul
    mean_row4 = float(((R**2).sum(1) ** 2).mean())
    cov64 = cov.astype(np.float64)
    shrunk, rho = _ledoit_wolf(cov64, mean_row4, n, d)
    lam, V = np.linalg.eigh(shrunk)  # ascending
    lam = lam[::-1]
    V = V[:, ::-1]
    lam = np.maximum(lam, lam.max() * 1e-12)  # floor (shrinkage already lifts the tail)
    return lam, V, rho


def _mahalanobis(R: np.ndarray, V: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """m^2_i = r_i^T Sigma^-1 r_i via the eigenbasis. R (N,d) fp32, V (d,d), lam (d,)."""
    W = R @ V.astype(np.float32)  # (N,d) fp32 matmul
    return (W.astype(np.float64) ** 2 / lam[None, :]).sum(1)


def _marginal_t_fit(vals: np.ndarray) -> dict:
    """Robust Gaussian-vs-Student-t comparison on a pooled 1D marginal sample.

    A multivariate-t_nu has univariate-t_nu marginals, so fitting a univariate t (df, scale; loc=0)
    to the pooled standardized projections estimates nu directly and gives a clean AIC vs a normal
    fit — sidestepping the numerically fragile radial F-MLE at d~3584 (both models 2 params)."""
    rng = np.random.default_rng(PROJ_SEED)
    if vals.size > 50_000:
        vals = vals[rng.choice(vals.size, 50_000, replace=False)]
    vals = vals.astype(np.float64)
    n = vals.size
    df_hat, _, scale_t = sstats.t.fit(vals, floc=0.0)
    logL_t = float(sstats.t.logpdf(vals, df_hat, 0.0, scale_t).sum())
    mu, sd = float(vals.mean()), float(vals.std())
    logL_norm = float(sstats.norm.logpdf(vals, mu, sd).sum())
    aic_t = -2.0 * logL_t + 2.0 * 2
    aic_norm = -2.0 * logL_norm + 2.0 * 2
    return {
        "fit_n": int(n),
        "nu_hat_mle": float(df_hat),
        "logL_t": logL_t,
        "logL_gauss": logL_norm,
        "aic_t": aic_t,
        "aic_gauss": aic_norm,
        "delta_aic_gauss_minus_t": aic_norm - aic_t,
        "prefers": "t" if aic_t < aic_norm else "gaussian",
    }


def _radial_fit(m2: np.ndarray, d: int, keep_qq: bool) -> dict:
    q_levels = [0.5, 0.9, 0.99, 0.999]
    obs_q = np.percentile(m2, [x * 100 for x in q_levels]).tolist()
    thr_q = sstats.chi2.ppf(q_levels, df=d).tolist()
    out = {
        "n": int(m2.size),
        "dim": d,
        "mean_m2": float(m2.mean()),
        "chi2_mean_ref": float(d),
        "var_m2": float(m2.var()),
        "chi2_var_ref": float(2 * d),
        "var_ratio_over_chi2": float(m2.var() / (2 * d)),
        "excess_kurtosis": float(sstats.kurtosis(m2)),
        "ks_stat_vs_chi2": float(sstats.kstest(m2, "chi2", args=(d,)).statistic),
        "quantile_levels": q_levels,
        "observed_quantiles": obs_q,
        "chi2_quantiles": thr_q,
    }
    if keep_qq:  # subsampled sorted m2 for the QQ panel (kept only for the plotted layer)
        rng = np.random.default_rng(PROJ_SEED)
        idx = np.sort(rng.choice(m2.size, size=min(1500, m2.size), replace=False))
        s = np.sort(m2)
        pos = (idx + 0.5) / m2.size
        out["_qq_obs"] = s[idx].tolist()
        out["_qq_chi2"] = sstats.chi2.ppf(pos, df=d).tolist()
    return out


def _projection_tests(R: np.ndarray, V: np.ndarray, d: int, ctx_id: np.ndarray) -> dict:
    rng = np.random.default_rng(PROJ_SEED)
    W_rand = rng.standard_normal((d, N_RAND_PROJ)).astype(np.float32)
    W_rand /= np.linalg.norm(W_rand, axis=0, keepdims=True) + EPS
    W = np.concatenate([V[:, :N_TOP_PCA].astype(np.float32), W_rand], axis=1)  # (d, 70)
    P = R @ W  # (N, 70)
    P = (P - P.mean(0)) / (P.std(0) + EPS)
    n_dir = P.shape[1]
    ad_reject = 0
    skews = np.empty(n_dir)
    kurts = np.empty(n_dir)
    for c in range(n_dir):
        col = P[:, c].astype(np.float64)
        a = sstats.anderson(col, "norm")
        if a.statistic > a.critical_values[2]:  # alpha 0.05
            ad_reject += 1
        skews[c] = sstats.skew(col)
        kurts[c] = sstats.kurtosis(col)
    # context-clustered bootstrap SE of the mean |skew| and mean excess-kurtosis across directions
    n_ctx = int(ctx_id.max()) + 1
    boot_kurt = np.empty(N_CLUSTER_BOOT)
    boot_skew = np.empty(N_CLUSTER_BOOT)
    base = np.arange(N_ROLLOUTS)
    for b in range(N_CLUSTER_BOOT):
        cids = rng.integers(0, n_ctx, n_ctx)
        idx = (cids[:, None] * N_ROLLOUTS + base[None, :]).ravel()
        Pb = P[idx]
        boot_kurt[b] = np.mean(sstats.kurtosis(Pb, axis=0))
        boot_skew[b] = np.mean(np.abs(sstats.skew(Pb, axis=0)))
    mean_kurt = float(np.mean(kurts))
    # kurtosis-matched df: univariate-t_nu has excess kurtosis 6/(nu-4) for nu>4 -> nu=4+6/k
    nu_kurt = float(4.0 + 6.0 / mean_kurt) if mean_kurt > 1e-6 else float("inf")
    t_fit = _marginal_t_fit(P.reshape(-1))  # pooled standardized marginal, robust AIC
    return {
        "n_directions": n_dir,
        "n_top_pca": N_TOP_PCA,
        "n_random": N_RAND_PROJ,
        "ad_reject_fraction_alpha05": float(ad_reject / n_dir),
        "ad_note": (
            f"n={R.shape[0]} pooled: Anderson-Darling saturates (rejects tiny deviations); read "
            "the skew/kurtosis EFFECT SIZES below, not the rejection fraction"
        ),
        "mean_abs_skew": float(np.mean(np.abs(skews))),
        "mean_excess_kurtosis": mean_kurt,
        "max_abs_excess_kurtosis": float(np.max(np.abs(kurts))),
        "excess_kurtosis_p95": float(np.percentile(kurts, 95)),
        "nu_hat_kurtosis_matched": nu_kurt,
        "marginal_t_fit": t_fit,
        "clustered_boot_mean_excess_kurtosis_se": float(boot_kurt.std()),
        "clustered_boot_mean_abs_skew_se": float(boot_skew.std()),
        "_excess_kurtosis_per_direction": kurts.tolist(),
        "_skew_per_direction": skews.tolist(),
    }


def _greedy_mahalanobis(
    greedy: np.ndarray,
    stoch: np.ndarray,
    sigma_x: np.ndarray,
    V: np.ndarray,
    lam: np.ndarray,
    d: int,
) -> dict:
    """Greedy deviation from the rollout mean, scale-normalized, under the per-context-scale
    (variant-b) Gaussian. Ties to the rank test: central greedy => sub-exchangeable m^2."""
    u = (greedy - stoch.mean(1)) / sigma_x[:, None]  # (n, d) scale-normalized greedy deviation
    m2g = _mahalanobis(u.astype(np.float32), V, lam)  # (n,)
    exch_ref = d * (N_ROLLOUTS + 1) / N_ROLLOUTS  # E[m2] if greedy were an 11th draw (var *11/10)
    chi2_95 = float(sstats.chi2.ppf(0.95, df=d))
    return {
        "mean_m2_greedy": float(m2g.mean()),
        "median_m2_greedy": float(np.median(m2g)),
        "chi2_mean_ref": float(d),
        "exchangeable_11th_draw_ref": float(exch_ref),
        "frac_gt_chi2_95": float((m2g > chi2_95).mean()),
        "percentiles_m2_greedy": _percentiles(m2g),
        "note": (
            f"m2_greedy well below the exchangeable 11th-draw ref ({exch_ref:.0f}) => greedy "
            f"central (consistent with the rank test); ~=chi2_d ({d}) => at the cloud-mean scale"
        ),
    }


def distribution_fit_layer(greedy: np.ndarray, stoch: np.ndarray, li: int, keep_qq: bool) -> dict:
    n_ctx = stoch.shape[0]
    d = stoch.shape[2]
    resid, sigma_x = _residuals(stoch)
    # Contexts whose 10 rollouts are (near-)identical have sigma_x=0 and cannot be scale-normalized
    # (0/0). Keep them in the RAW pool (their zero residuals are legitimate, contribute nothing),
    # but MASK them from the scale-normalized pool + the greedy tie-in.
    floor = max(1e-8, 1e-6 * float(np.median(sigma_x)))
    valid = sigma_x > floor
    n_valid = int(valid.sum())
    R_raw = resid.reshape(-1, d)
    ctx_raw = np.repeat(np.arange(n_ctx), N_ROLLOUTS)
    R_sn = (resid[valid] / sigma_x[valid][:, None, None]).reshape(-1, d)
    ctx_sn = np.repeat(np.arange(n_valid), N_ROLLOUTS)
    variants: dict = {}
    for name, R, cid in (
        ("raw", R_raw, ctx_raw),
        ("scale_normalized", R_sn, ctx_sn),
    ):
        lam, V, rho = _cov_eig(R)
        m2 = _mahalanobis(R, V, lam)
        block = {
            "shrinkage_rho": rho,
            "radial": _radial_fit(m2, d, keep_qq),
            "projections": _projection_tests(R, V, d, cid),
        }
        if name == "scale_normalized":
            block["greedy_tie_in"] = _greedy_mahalanobis(
                greedy[valid], stoch[valid], sigma_x[valid], V, lam, d
            )
            # dispersion-heterogeneity check (cheap): top-PC variance share
            block["top_pc_variance_share"] = float(lam[0] / lam.sum())
            block["n_contexts_used"] = n_valid
            block["n_zero_scale_contexts_dropped"] = int(n_ctx - n_valid)
        variants[name] = block
        del lam, V, m2, R
    return variants


# ── driver ───────────────────────────────────────────────────────────────────────


def run(layers: list[int], out_dir: Path) -> dict:
    t0 = time.time()
    keep = GT.load_keep()
    spans = GT.load_span_lens(keep)
    len_gap = spans["greedy_span"].astype(np.float64) - spans["stoch_span"].astype(np.float64).mean(
        1
    )  # (n,) greedy minus mean-stoch response length (tokens)
    logger.info("[setup] keep=%d done in %.1fs", keep.size, time.time() - t0)

    out: dict = {"readout_layers": layers, "per_layer": {}}
    for li in layers:
        tl = time.time()
        v = GT.stoch_matrix(li, keep)  # (n,10,H) fp64
        disp = GT.rollout_dispersion(v)  # reused verbatim
        v32 = v.astype(np.float32)
        del v
        g32 = greedy_matrix(li, keep).astype(np.float32)  # (n,H)
        res = {
            "rank_test": rank_test(g32, v32),
            "offset_test": offset_test(g32, v32, len_gap),
            "norm_test": norm_test(g32, v32),
            "cloud_descriptives": cloud_descriptives(g32, v32, disp),
        }
        out["per_layer"][f"L{li}"] = res
        del v32, g32
        logger.info(
            "[L%d] rank mean=%.3f (unif %.1f) chi2p=%.2e central=%.3f periph=%.3f | "
            "offset ||ubar||=%.4f p=%.4f meancos=%.3f r2len=%.3f | done %.1fs",
            li,
            res["rank_test"]["mean_rank"],
            res["rank_test"]["mean_rank_uniform_expected"],
            res["rank_test"]["chi2_p_vs_uniform"],
            res["rank_test"]["frac_greedy_most_central"],
            res["rank_test"]["frac_greedy_most_peripheral"],
            res["offset_test"]["mean_offset_norm"],
            res["offset_test"]["signflip_p_value"],
            res["offset_test"]["mean_cos_u_ubar"],
            res["offset_test"]["r2_offset_explained_by_len_gap"],
            time.time() - tl,
        )

    out["definitions"] = {
        "rank_test": (
            "treat v_greedy as an 11th rollout; each of the 11 items' cosine distance to the "
            "leave-self-out mean of the other 10; rank greedy 1..11 by ascending distance "
            "(1=most central). Uniform on 1..11 iff greedy is exchangeable with a stochastic draw."
        ),
        "offset_u": "u(x) = v_greedy(x) - mean_j v_j(x); u_bar = mean_x u(x)",
        "signflip_null": (
            "null distribution of ||mean_x s_x u(x)|| over N_SIGNFLIP random per-context sign "
            "flips s_x in {-1,+1}; p = fraction of null >= observed ||u_bar||"
        ),
        "len_gap": "greedy response span length - mean stochastic response span length (tokens)",
        "r2_offset_explained_by_len_gap": (
            "R2 of the per-context offset u(x) explained by a scalar linear map on the length gap: "
            "u(x) ~ u_bar + b * (gap(x)-mean gap)"
        ),
        "cloud_descriptives": (
            "greedy distance to the 10-mean vs each stochastic draw's leave-one-out distance to "
            "the mean of the other 9 (both external to the reference mean, so comparable)"
        ),
    }
    out["metadata"] = I.reproducibility_metadata(
        {
            "script": "issue1073_greedy_cloud_distribution",
            "store_revision": STORE_REV,
            "n_signflip": N_SIGNFLIP,
            "n_boot": N_BOOT,
            "rng_seed": RNG_SEED,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    I.write_json_atomic(out_dir / "greedy_cloud_distribution.json", out)
    logger.info("[write] greedy_cloud_distribution.json in %.1fs total", time.time() - t0)
    return out


# ── figure ───────────────────────────────────────────────────────────────────────


def make_figure(out_dir: Path, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "greedy_cloud_distribution.json") as f:
        res = json.load(f)
    layers = [int(k[1:]) for k in res["per_layer"]]
    pal = pp.paper_palette(len(layers))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (a1, a2), (a3, a4) = axes

    # Panel A: rank histogram per layer + uniform reference
    k = res["per_layer"][f"L{layers[0]}"]["rank_test"]["n_items"]
    ranks = np.arange(1, k + 1)
    for i, li in enumerate(layers):
        fr = res["per_layer"][f"L{li}"]["rank_test"]["rank_fractions"]
        a1.plot(ranks, fr, "o-", ms=4, color=pal[i], label=f"L{li}")
    a1.axhline(1.0 / k, color="k", ls="--", lw=1.0)
    a1.set_xlabel("greedy centrality rank among 11 items (1 = most central)")
    a1.set_ylabel("fraction of contexts")
    a1.set_xticks(ranks)
    a1.legend(fontsize=7, ncol=2, title="uniform = dashed")

    # Panel B: greedy vs typical distance-to-mean, headline layer (largest greedy offset)
    hl = max(layers, key=lambda li: res["per_layer"][f"L{li}"]["offset_test"]["mean_offset_norm"])
    cd = res["per_layer"][f"L{hl}"]["cloud_descriptives"]
    g = np.array(cd["_hist_greedy_dist"])
    s = np.array(cd["_hist_stoch_dist"])
    lo, hi = float(min(g.min(), s.min())), float(max(g.max(), s.max()))
    bins = np.linspace(lo, hi, 60)
    a2.hist(s, bins=bins, density=True, alpha=0.5, color=pal[0], label="stochastic draw (LOO)")
    a2.hist(g, bins=bins, density=True, alpha=0.5, color=pal[-1], label="greedy")
    a2.set_xlabel(f"cosine distance to cloud mean (L{hl})")
    a2.set_ylabel("density")
    a2.legend(fontsize=8)

    # Panel C: observed mean-offset norm vs sign-flip null per layer
    x = np.arange(len(layers))
    obs = [res["per_layer"][f"L{li}"]["offset_test"]["mean_offset_norm"] for li in layers]
    nullm = [res["per_layer"][f"L{li}"]["offset_test"]["signflip_null_norm_mean"] for li in layers]
    nullmax = [res["per_layer"][f"L{li}"]["offset_test"]["signflip_null_norm_max"] for li in layers]
    yerr = np.maximum(0.0, np.array(nullmax) - np.array(nullm))
    w = 0.38
    a3.bar(x - w / 2, obs, w, color=pal[-1], label="observed $\\|\\bar u\\|$")
    a3.bar(
        x + w / 2,
        nullm,
        w,
        yerr=[np.zeros(len(layers)), yerr],
        color=pal[0],
        label="sign-flip null (mean; cap=max)",
        capsize=3,
    )
    a3.set_xticks(x)
    a3.set_xticklabels([f"L{li}" for li in layers])
    a3.set_ylabel("mean-offset norm")
    a3.set_xlabel("read-out layer")
    a3.legend(fontsize=7)

    # Panel D: paired greedy-minus-mean-stoch norm diff per layer, bootstrap CI
    med = [res["per_layer"][f"L{li}"]["norm_test"]["paired_median_diff"] for li in layers]
    ci = [res["per_layer"][f"L{li}"]["norm_test"]["paired_median_diff_ci95"] for li in layers]
    lo_e = np.maximum(0.0, np.array(med) - np.array([c[0] for c in ci]))
    hi_e = np.maximum(0.0, np.array([c[1] for c in ci]) - np.array(med))
    a4.axhline(0, color="k", lw=0.8)
    a4.errorbar(x, med, yerr=[lo_e, hi_e], fmt="o", color=pal[2], capsize=3)
    a4.set_xticks(x)
    a4.set_xticklabels([f"L{li}" for li in layers])
    a4.set_ylabel("median $\\|v_{greedy}\\| - \\overline{\\|v_j\\|}$")
    a4.set_xlabel("read-out layer")

    fig.tight_layout()
    pp.savefig_paper(fig, "greedy_cloud_distribution", dir=fig_dir)
    plt.close(fig)
    logger.info("[figure] wrote greedy_cloud_distribution to %s", fig_dir)


# ── Analysis 5 driver + figure ───────────────────────────────────────────────────


def run_dist_fit(layers: list[int], out_dir: Path) -> dict:
    """Compute the distribution_fit block and MERGE it into the existing results JSON."""
    t0 = time.time()
    keep = GT.load_keep()
    logger.info("[dist-fit setup] keep=%d done in %.1fs", keep.size, time.time() - t0)
    # headline layer for the QQ panel = largest greedy mean-offset (matches make_figure); recovered
    # from the existing JSON if present, else the last layer.
    hl = layers[-1]
    res_path = out_dir / "greedy_cloud_distribution.json"
    existing = None
    if res_path.exists():
        with open(res_path) as f:
            existing = json.load(f)
    if (
        existing and "per_layer" in existing
    ):  # global headline across ALL analysed layers (not batch)
        all_li = [int(k[1:]) for k in existing["per_layer"]]
        if all_li:
            hl = max(
                all_li,
                key=lambda li: existing["per_layer"][f"L{li}"]["offset_test"]["mean_offset_norm"],
            )

    block: dict = {"headline_layer_for_qq": hl, "per_layer": {}}
    for li in layers:
        tl = time.time()
        v = GT.stoch_matrix(li, keep).astype(np.float32)
        g = greedy_matrix(li, keep).astype(np.float32)
        block["per_layer"][f"L{li}"] = distribution_fit_layer(g, v, li, keep_qq=(li == hl))
        raw = block["per_layer"][f"L{li}"]["raw"]
        sn = block["per_layer"][f"L{li}"]["scale_normalized"]
        logger.info(
            "[L%d dist-fit] RAW: nu=%.1f dAIC=%.3e kurt=%.2f varrat=%.2f | SCALE-NORM: nu=%.1f "
            "dAIC=%.3e kurt=%.2f varrat=%.2f greedy_m2med=%.0f (exch %.0f) | %.1fs",
            li,
            raw["projections"]["marginal_t_fit"]["nu_hat_mle"],
            raw["projections"]["marginal_t_fit"]["delta_aic_gauss_minus_t"],
            raw["projections"]["mean_excess_kurtosis"],
            raw["radial"]["var_ratio_over_chi2"],
            sn["projections"]["marginal_t_fit"]["nu_hat_mle"],
            sn["projections"]["marginal_t_fit"]["delta_aic_gauss_minus_t"],
            sn["projections"]["mean_excess_kurtosis"],
            sn["radial"]["var_ratio_over_chi2"],
            sn["greedy_tie_in"]["median_m2_greedy"],
            sn["greedy_tie_in"]["exchangeable_11th_draw_ref"],
            time.time() - tl,
        )
        del v, g

    block["definitions"] = {
        "residual": "r_j(x) = v_j(x) - mean_10(x), scaled by sqrt(10/9) (own-mean bias correction)",
        "variant_raw": "residuals pooled across all contexts (a per-context-scale MIXTURE)",
        "variant_scale_normalized": (
            "each context's residuals divided by sigma_x = sqrt(mean_(j,dim) r^2) (its per-dim RMS "
            "residual magnitude) before pooling; removes the context-dependent SCALE"
        ),
        "radial_m2": "squared Mahalanobis under the Ledoit-Wolf-shrunk pooled within-context "
        "covariance; ~ chi2_d if the pool is multivariate Gaussian",
        "t_fit": "MLE multivariate-t df on the radial distribution; delta_aic>0 => t preferred; "
        "nu_hat->1000 (bound) => effectively Gaussian",
        "projections": "top-20 PCA + 50 fixed-random(seed42) unit directions; per-direction "
        "Anderson-Darling + skew/excess-kurtosis; context-clustered bootstrap SE (resample ctx)",
        "greedy_tie_in": "scale-normalized greedy deviation Mahalanobis under the variant-b "
        "covariance vs chi2_d (cloud-mean scale) and the exchangeable 11th-draw ref (d*11/10)",
        "discriminating_design": "raw heavy-tailed (t) + scale-normalized Gaussian => per-context "
        "Gaussian cloud with context-dependent scale",
        "caveats": "n=10/context => pooled-distribution claims only, not per-context normality; "
        "pooled covariance assumes homogeneous SHAPE (top_pc_variance_share reported as a check); "
        "Anderson-Darling saturates at n=50000 (read skew/kurtosis effect sizes).",
    }
    block["metadata"] = I.reproducibility_metadata(
        {
            "script": "issue1073_greedy_cloud_distribution",
            "analysis": "distribution_fit",
            "store_revision": STORE_REV,
            "n_rand_proj": N_RAND_PROJ,
            "n_top_pca": N_TOP_PCA,
            "n_cluster_boot": N_CLUSTER_BOOT,
            "proj_seed": PROJ_SEED,
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    merged = existing if isinstance(existing, dict) else {}
    # append per-layer into any prior distribution_fit block (batched runs merge; don't overwrite)
    prev_pl = merged.get("distribution_fit", {}).get("per_layer", {})
    prev_pl.update(block["per_layer"])
    block["per_layer"] = prev_pl
    block["readout_layers"] = sorted({int(k[1:]) for k in prev_pl})
    merged["distribution_fit"] = block
    I.write_json_atomic(res_path, merged)
    logger.info(
        "[dist-fit write] merged %d-layer distribution_fit block in %.1fs total",
        len(prev_pl),
        time.time() - t0,
    )
    return merged


def make_dist_figure(out_dir: Path, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "greedy_cloud_distribution.json") as f:
        res = json.load(f)
    df = res["distribution_fit"]
    layers = sorted(int(k[1:]) for k in df["per_layer"])
    hl = df["headline_layer_for_qq"]
    pal = pp.paper_palette(3)
    c_raw, c_sn = pal[0], pal[2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (a1, a2), (a3, a4) = axes

    # Panel A: radial QQ vs chi2_d (headline layer), raw + scale-normalized
    hlb = df["per_layer"][f"L{hl}"]
    for name, col, lab in (("raw", c_raw, "raw"), ("scale_normalized", c_sn, "scale-normalized")):
        rad = hlb[name]["radial"]
        if "_qq_obs" in rad:
            a1.plot(rad["_qq_chi2"], rad["_qq_obs"], ".", ms=3, color=col, label=lab)
    allq = hlb["raw"]["radial"].get("_qq_chi2", []) + hlb["raw"]["radial"].get("_qq_obs", [])
    if allq:
        m = max(allq)
        a1.plot([0, m], [0, m], "k--", lw=1.0)
    a1.set_xlabel(f"$\\chi^2_d$ theoretical quantile (L{hl}, d={hlb['raw']['radial']['dim']})")
    a1.set_ylabel("observed squared Mahalanobis quantile")
    a1.legend(fontsize=8, title="$y=x$ dashed")

    # Panel B: excess-kurtosis across the 70 projections (headline layer), raw vs scale-normalized
    kr = np.array(hlb["raw"]["projections"]["_excess_kurtosis_per_direction"])
    ks = np.array(hlb["scale_normalized"]["projections"]["_excess_kurtosis_per_direction"])
    lo, hi = float(min(kr.min(), ks.min())), float(max(kr.max(), ks.max()))
    bins = np.linspace(lo, hi, 30)
    a2.hist(kr, bins=bins, alpha=0.55, color=c_raw, label="raw")
    a2.hist(ks, bins=bins, alpha=0.55, color=c_sn, label="scale-normalized")
    a2.axvline(0, color="k", lw=0.8)
    a2.set_xlabel(f"per-projection excess kurtosis (L{hl}, 70 directions)")
    a2.set_ylabel("count")
    a2.legend(fontsize=8, title="Gaussian = 0")

    # Panel C: Gaussian-vs-t delta-AIC per layer, raw vs scale-normalized (log-scaled, +1 offset)
    x = np.arange(len(layers))
    dr = [
        df["per_layer"][f"L{li}"]["raw"]["projections"]["marginal_t_fit"]["delta_aic_gauss_minus_t"]
        for li in layers
    ]
    ds = [
        df["per_layer"][f"L{li}"]["scale_normalized"]["projections"]["marginal_t_fit"][
            "delta_aic_gauss_minus_t"
        ]
        for li in layers
    ]
    w = 0.38
    a3.bar(x - w / 2, np.maximum(dr, 1.0), w, color=c_raw, label="raw")
    a3.bar(x + w / 2, np.maximum(ds, 1.0), w, color=c_sn, label="scale-normalized")
    a3.set_yscale("log")
    a3.set_xticks(x)
    a3.set_xticklabels([f"L{li}" for li in layers])
    a3.set_ylabel("$\\Delta$AIC (Gaussian $-$ t); $>0$ favours t")
    a3.set_xlabel("read-out layer")
    a3.legend(fontsize=8)

    # Panel D: greedy Mahalanobis (scale-normalized) median per layer vs chi2_d & 11th-draw refs
    gm = [
        df["per_layer"][f"L{li}"]["scale_normalized"]["greedy_tie_in"]["median_m2_greedy"]
        for li in layers
    ]
    chi = [
        df["per_layer"][f"L{li}"]["scale_normalized"]["greedy_tie_in"]["chi2_mean_ref"]
        for li in layers
    ]
    exch = [
        df["per_layer"][f"L{li}"]["scale_normalized"]["greedy_tie_in"]["exchangeable_11th_draw_ref"]
        for li in layers
    ]
    a4.plot(x, gm, "o-", color=c_sn, label="greedy median $m^2$")
    a4.plot(x, chi, "s--", color="k", label="$\\chi^2_d$ mean ($d$)")
    a4.plot(x, exch, "^:", color=c_raw, label="exchangeable 11th draw")
    a4.set_xticks(x)
    a4.set_xticklabels([f"L{li}" for li in layers])
    a4.set_ylabel("greedy squared Mahalanobis")
    a4.set_xlabel("read-out layer")
    a4.legend(fontsize=7)

    fig.tight_layout()
    pp.savefig_paper(fig, "cloud_distribution_fit", dir=fig_dir)
    plt.close(fig)
    logger.info("[figure] wrote cloud_distribution_fit to %s", fig_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=READOUT_LAYERS)
    ap.add_argument("--pilot", action="store_true", help="single layer (19) for wall-time probe")
    ap.add_argument("--out-dir", type=str, default=str(EVAL_RESULTS_DIR))
    ap.add_argument("--fig-dir", type=str, default=str(FIG_DIR))
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--skip-stage", action="store_true", help="store already staged at STAGE_HF")
    ap.add_argument(
        "--dist-fit",
        action="store_true",
        help="analysis 5 only: compute the distribution_fit block and MERGE into the results JSON",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.figures_only:
        if args.dist_fit:
            make_dist_figure(out_dir, Path(args.fig_dir))
        else:
            make_figure(out_dir, Path(args.fig_dir))
        return 0
    if not args.skip_stage:
        stage_store()
    else:
        GT.STORE_DIR = STAGE_HF / HF_PREFIX / "v_store"
        GT.COVERAGE = STAGE_HF / HF_PREFIX / "reductions" / "coverage.pt"
    layers = [19] if args.pilot else list(args.layers)
    if args.dist_fit:
        merged = run_dist_fit(layers, out_dir)
        done = set(merged.get("distribution_fit", {}).get("readout_layers", []))
        # figure only once every read-out layer's dist-fit is present (batched runs skip until then)
        if not args.pilot and set(READOUT_LAYERS).issubset(done):
            make_dist_figure(out_dir, Path(args.fig_dir))
        return 0
    run(layers, out_dir)
    if not args.pilot:
        make_figure(out_dir, Path(args.fig_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
