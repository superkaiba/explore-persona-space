#!/usr/bin/env python3
"""Issue #2202 inline round `metric-zoo` — literature-derived + invented
similarity-convention battery over the banked #1738 context→answer map tensors.

Full-pool retrieval (9,941 held-out predictions vs the 9,941 held-out answer
vectors, layer 19) under ≤18 NEW conventions (roster in ROSTER below), scored
with the exact `mapping_baselines.knn_retrieval` mid-rank + tie-tolerance
convention (via chunked GEMMs, per `issue2202_failchar.ranks_of_targets`).
For the top-3 new conventions by acc@1, the fresh-draw retrievability ceiling
is recomputed convention-matched (same definition as the banked 0.9425:
per-context rank-1 fraction over the 1,988 × K=4 kresample draws, mean over
contexts).

Inputs are the sibling round's staged copies (byte-size-verified vs HF
metadata before this script was written):
  /mnt/eps-data/thomasjiralerspong/issue2202_freshwhiten/{pred16,y_holdout_L19,
  whiten_stats}.npz + kresample/kresample_shard00.pt   (READ-ONLY)

Reconciliation gates (fail loud, rc 21):
  - raw-euclidean acc@1 must reproduce the banked 0.816014485464239 (≤2 rows);
  - truncated-whitening at k=d (euclidean read) must reproduce the banked
    degenerate whiten acc@1 ≈ 0.0203 (Mahalanobis is whitening-basis-invariant);
  - the raw-euclidean fresh-draw ceiling must reproduce the banked 0.94253.

Checkpoint-per-unit: every convention's record is appended to results.jsonl the
moment it completes; re-runs skip completed names. No fits are performed —
every transform is a closed-form function of the banked train statistics
(n_train = 88,378 ≫ d = 3,584) or of pool-internal distances.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps land BEFORE numpy on the shared VM (#847)

import numpy as np  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import _pairwise_dist  # noqa: E402
from explore_persona_space.analysis.null_battery import shrunk_cholesky_from_cov  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

ISSUE = 2202
LAYER = 19
H_DIM = 3584
EXPECTED_N = 9_941
KS = (1, 5, 10)
RC_GATE = 21
BANKED_LAMBDA = 0.1  # task-locked shrinkage of the banked whiten_stats
STAGED_DEFAULT = "/mnt/eps-data/thomasjiralerspong/issue2202_freshwhiten"
WORKDIR_DEFAULT = "/mnt/eps-data/thomasjiralerspong/issue2202_metric_zoo"
OUT_EVAL_DEFAULT = str(PROJECT_ROOT / "eval_results" / "issue_2202" / "metric_zoo")
BANKED_ACC1_EUC = 0.816014485464239
BANKED_ACC1_WHITEN_EUC = (9941 - 9739) / 9941  # geometry_summary fail_counts
BANKED_CEILING_EUC = 0.9425301810865191  # attribution.json acc1_ceiling
ACC_TOL_ROWS = 2  # plan-§7 knife-edge tie allowance (issue2202_failchar)
TW_KS = (64, 256, 1024)
ABTT_D = 35  # ≈ d/100 (Mu & Viswanath 2018 recommend d/100)
K_LOCAL = 10  # NICDM / DSL / CSLS / in-degree neighborhood size
ISF_BETA = 30.0  # primary; 10.0 as sensitivity (Smith et al. tune on a dict)


# ── roster: name -> (kind, params, mechanism one-liner, citation) ─────────────────
# kind ∈ {transform-euc, transform-cos, rescore}; rescore lines carry a builder tag.
ROSTER: list[dict] = [
    # -- truncated / partial whitening (probes WHY full whitened-euclid degenerated) --
    *[
        {
            "name": f"truncwhiten_k{k}_{read}",
            "kind": f"transform-{read}",
            "params": {"k_top": k, "basis": "eigen (ZCA-style, symmetric)"},
            "mechanism": (
                f"whiten only the top-{k} eigendirections of the task-locked shrunk train-answer "
                "covariance (identity elsewhere): keeps the signal-equalizing effect of whitening "
                "on high-variance directions while never amplifying the low-variance directions "
                "that dominate (and degenerate) the full-whitened euclidean read"
            ),
            "citation": "partial/soft whitening: Soft-ZCA (arXiv:2411.17538); ZCA vs PCA: Kessy et al. (arXiv:1512.00809)",
        }
        for k in TW_KS
        for read in ("euc", "cos")
    ],
    # -- diagonal whitening (standardized euclidean) --
    {
        "name": "diagwhiten_euc",
        "kind": "transform-euc",
        "params": {"scale": "1/sqrt(diag(cov_train_answer))"},
        "mechanism": "per-dimension z-scoring (standardized euclidean) — decorrelation-free middle ground between raw and full Mahalanobis",
        "citation": "standardized euclidean / diagonal Mahalanobis (classic); shrinkage: Ledoit & Wolf 2004",
    },
    {
        "name": "diagwhiten_cos",
        "kind": "transform-cos",
        "params": {"scale": "1/sqrt(diag(cov_train_answer))"},
        "mechanism": "cosine read in the per-dimension z-scored space",
        "citation": "as diagwhiten_euc",
    },
    # -- shrinkage sweep on the full whitening, cosine read (Cholesky basis = banked convention) --
    {
        "name": "whitencos_lam03",
        "kind": "transform-cos",
        "params": {"lam": 0.3, "basis": "cholesky (banked convention)"},
        "mechanism": "full whitening at heavier diagonal-target shrinkage λ=0.3 (vs task-locked 0.1) — tests whether the whitened-cosine win is shrinkage-sensitive",
        "citation": "shrunk covariance whitening: Ledoit & Wolf 2004; whitening-for-retrieval: Su et al. (arXiv:2103.15316)",
    },
    {
        "name": "whitencos_lam05",
        "kind": "transform-cos",
        "params": {"lam": 0.5, "basis": "cholesky (banked convention)"},
        "mechanism": "as whitencos_lam03 at λ=0.5",
        "citation": "as whitencos_lam03",
    },
    # -- hubness-reduction rescorings on the euclidean base --
    {
        "name": "mp_emp_euc",
        "kind": "rescore",
        "params": {"base": "euclidean", "variant": "empirical, independence approximation"},
        "mechanism": "Mutual Proximity: similarity = P(d(q,X) > d(q,j))·P(d(j,Y) > d(j,q)) — a candidate that is close to EVERYONE (hub) gets a low second factor",
        "citation": "Schnitzer et al. 2012 (JMLR 13); approximate-MP ≈ full MP: Feldbauer & Flexer 2019 (KAIS, doi:10.1007/s10115-018-1205-y)",
    },
    {
        "name": "nicdm_k10_euc",
        "kind": "rescore",
        "params": {"base": "euclidean", "k": K_LOCAL},
        "mechanism": "NICDM local scaling: d' = d(q,j)/sqrt(r_k(q)·r_k(j)) with r_k = mean distance to the k nearest neighbors — deflates candidates that sit in dense regions",
        "citation": "Schnitzer et al. 2012 (JMLR 13); local scaling: Zelnik-Manor & Perona NIPS 2004",
    },
    {
        "name": "dsl_k10_euc",
        "kind": "rescore",
        "params": {"base": "squared euclidean", "k": K_LOCAL},
        "mechanism": "DisSimLocal: d' = ||q−y_j||² − ||q−c_k(q)||² − ||y_j−c_k(y_j)||² (c_k = local kNN centroid) — flattens the density gradient that makes centrally-located answers hubs",
        "citation": "Hara et al. AAAI 2016; best hubness reduction in Feldbauer & Flexer 2019",
    },
    {
        "name": "isf_cos_b30",
        "kind": "rescore",
        "params": {"base": "cosine", "beta": ISF_BETA, "sensitivity_beta": 10.0},
        "mechanism": "inverted softmax: score = βS(q,j) − logsumexp_q'(βS(q',j)) — normalizes each candidate's similarity mass over the QUERY bank, so universal-match hubs are down-weighted",
        "citation": "Smith et al. ICLR 2017 (arXiv:1702.03859); querybank form: QB-Norm, Bogolin et al. CVPR 2022 (arXiv:2112.12777)",
    },
    # -- combos on the best banked base (whitened cosine, Cholesky basis) --
    {
        "name": "csls_k10_whitencos",
        "kind": "rescore",
        "params": {"base": "whitened cosine (lam=0.1, cholesky)", "k": K_LOCAL},
        "mechanism": "CSLS on the best banked base: score = 2S − r_q − r_j (cross-domain k-NN means) — hub penalty applied where the base geometry is already strongest",
        "citation": "Conneau et al. ICLR 2018 (arXiv:1710.04087); banked CSLS-on-raw-cos = 0.9095",
    },
    # -- top-PC removal --
    {
        "name": "abtt_d35_cos",
        "kind": "transform-cos",
        "params": {"removed_top_dirs": ABTT_D},
        "mechanism": "all-but-the-top: center + REMOVE the top-35 (≈d/100) covariance eigendirections, cosine read — the inverse ablation of truncated whitening (kills, rather than equalizes, the dominant directions)",
        "citation": "Mu & Viswanath ICLR 2018 (arXiv:1702.01417)",
    },
    # -- inventions --
    {
        "name": "alphawhiten_a05_cos",
        "kind": "transform-cos",
        "params": {"alpha": 0.5, "basis": "eigen (ZCA-style, symmetric)"},
        "mechanism": "INVENTED (closest precedent: Soft-ZCA): fractional-power whitening W = V·diag(w^{−α/2})·Vᵀ, α=0.5 — a continuous interpolation raw→whitened; caps the amplification of low-variance directions at sqrt instead of full",
        "citation": "invented; Soft-ZCA (arXiv:2411.17538) regularizes eigenvalues rather than powering them",
    },
    {
        "name": "hubdeg_pen_whitencos_g05",
        "kind": "rescore",
        "params": {"base": "whitened cosine (lam=0.1, cholesky)", "k": K_LOCAL, "gamma": 0.5},
        "mechanism": "INVENTED: pool-in-degree hub penalty — N_k(j) = j's in-degree in the pool-internal kNN graph; score = S(q,j) − γ·σ_S·zscore(N_k(j)) — penalizes empirically-observed hubs directly rather than via local similarity means",
        "citation": "invented; in-degree (k-occurrence) as the hubness statistic: Radovanović et al. 2010 (JMLR 11)",
    },
]

DROPPED: list[dict] = [
    {
        "name": "centered_euclidean",
        "reason": (
            "algebraic identity: euclidean distance is translation-invariant, so pool-mean "
            "centering is a no-op — centered-euclidean ≡ raw-euclidean exactly (Suzuki et al. "
            "EMNLP 2013 centering acts on inner-product/cosine similarities, not euclidean). "
            "Verified numerically on a 256-row slice at battery start; raw-euclidean acc@1 "
            "0.8160 is its value."
        ),
    },
    {
        "name": "target_only_csls",
        "reason": (
            "rank-identity: within a query row, CSLS = 2S − r_q − r_j differs from S − r_j/2 "
            "by a row-constant (r_q) and a positive scale, so acc@k is IDENTICAL to full CSLS "
            "— the r_q term only matters for cross-query calibration, never for per-query "
            "ranking. Folded into csls_k10_whitencos; a γ=1.0 penalty variant (double CSLS "
            "strength) rides that record as sensitivity."
        ),
    },
    {
        "name": "spearman_across_dims",
        "reason": (
            "cut for the ≤18 roster cap: the sibling round (freshwhiten-avg) already covers "
            "Pearson-r-across-dimensions; the rank-transformed variant probes the same "
            "correlation-read axis at strictly higher cost."
        ),
    },
    {
        "name": "mp_gauss_whitencos",
        "reason": "cut for the ≤18 roster cap: csls_k10_whitencos + hubdeg_pen_whitencos_g05 already probe hub correction on the whitened-cosine base.",
    },
]


def now_iso() -> str:
    """UTC ISO timestamp for result metadata."""
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def meta_block() -> dict:
    """Reproducibility metadata block (git sha + dirty flag, versions, ts)."""
    return {
        "generated_utc": now_iso(),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        **as_metadata_dict(git_provenance(PROJECT_ROOT)),
    }


def atomic_json(path: Path, obj: dict) -> None:
    """Atomic JSON write (tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def ranks_summary(ranks: np.ndarray, n_pool: int) -> dict:
    """acc@k / median rank / MRR from per-row mid-ranks (issue2202_failchar shape)."""
    return {
        "acc_at_k": {int(k): float((ranks <= k).mean()) for k in KS},
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
        "n": int(ranks.shape[0]),
        "n_pool": int(n_pool),
    }


def midranks_of_true(d: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """Mid-ranks of d[i, true_idx[i]] within each row (knn_retrieval convention:
    1 + #closer + 0.5·#tied-others; tol = 1e-9·max(|d_true|, 1e-12))."""
    n = d.shape[0]
    dt = d[np.arange(n), true_idx]
    tol = 1e-9 * np.maximum(np.abs(dt)[:, None], 1e-12)
    closer = (d < dt[:, None] - tol).sum(axis=1)
    tied = (np.abs(d - dt[:, None]) <= tol).sum(axis=1) - 1
    return 1.0 + closer + 0.5 * tied


def ranks_transform(
    pred_t: np.ndarray,
    pool_t: np.ndarray,
    true_idx: np.ndarray,
    metric: str,
    chunk: int,
    tag: str,
) -> np.ndarray:
    """Per-row mid-ranks under distance(pred_t, pool_t), chunked over query rows
    (the issue2202_failchar.ranks_of_targets recipe; batched GEMMs, never
    per-pair loops)."""
    n = pred_t.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    t0 = time.time()
    n_chunks = (n + chunk - 1) // chunk
    for ci, s in enumerate(range(0, n, chunk)):
        e = min(n, s + chunk)
        d = _pairwise_dist(pred_t[s:e], pool_t, metric)
        ranks[s:e] = midranks_of_true(d, true_idx[s:e])
        print(
            f"[{tag}] unit {ci + 1}/{n_chunks} rows={s}:{e} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    return ranks


def ranks_score_matrix(score: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """Mid-ranks under retrieval distance = −score (csls_followup convention)."""
    return midranks_of_true(-score, true_idx)


# ── staged input loading ───────────────────────────────────────────────────────────


def load_staged(staged: Path) -> dict:
    """(pred, y16, pci fp64/int64, whiten stats) with shape + alignment asserts."""
    pd_ = np.load(staged / "pred16.npz")
    yd = np.load(staged / "y_holdout_L19.npz")
    pred = pd_["pred16"].astype(np.float64)
    y16 = yd["y16"].astype(np.float64)
    pci = np.asarray(pd_["ci"], dtype=np.int64)
    yci = np.asarray(yd["ci"], dtype=np.int64)
    assert pred.shape == (EXPECTED_N, H_DIM), pred.shape
    assert y16.shape == (EXPECTED_N, H_DIM), y16.shape
    assert (pci == yci).all(), "pred16/y_holdout ci misalign"
    assert np.array_equal(pd_["fingerprint"], yd["fingerprint"]), "fingerprint mismatch"
    wz = np.load(staged / "whiten_stats.npz")
    stats = {k: np.asarray(wz[k], dtype=np.float64) for k in ("mu_A", "mu_C", "L")}
    assert float(wz["lam"]) == BANKED_LAMBDA, float(wz["lam"])
    n_train = int(wz["n_train"])
    assert n_train > H_DIM, (n_train, H_DIM)  # covariance well-conditioned; no fit here
    stats["n_train"] = n_train
    return {"pred": pred, "y16": y16, "pci": pci, "stats": stats}


def load_kresample(staged: Path, pci: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(q_fresh (n_k·K, H) fp64, true_idx (n_k·K,), kci (n_k,)) from the staged
    shard (the issue1738_characterize._load_kresample_v single-shard case)."""
    import torch

    paths = sorted((staged / "kresample").glob("kresample_shard*.pt"))
    assert paths, f"no kresample shards under {staged / 'kresample'}"
    cis, vs = [], []
    for p in paths:
        b = torch.load(p, map_location="cpu", weights_only=False)
        li_pos = list(b["layers"]).index(LAYER)
        cis.extend(int(c) for c in b["ci"])
        vs.append(b["V"][:, :, li_pos, :].to(torch.float32).numpy())
    kci = np.asarray(cis, dtype=np.int64)
    v = np.concatenate(vs, axis=0)  # (n_k, K, H)
    assert v.ndim == 3 and v.shape[2] == H_DIM, v.shape
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    missing = [int(c) for c in kci if int(c) not in pos_of]
    assert not missing, f"{len(missing)} kresample cis not in holdout pool"
    n_k, k_draws = v.shape[0], v.shape[1]
    q = v.reshape(n_k * k_draws, H_DIM).astype(np.float64)
    true_idx = np.repeat(np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64), k_draws)
    return q, true_idx, kci


# ── transform builders ───────────────────────────────────────────────────────────


class Transforms:
    """Closed-form transforms derived once from the banked whiten_stats."""

    def __init__(self, stats: dict, workdir: Path):
        self.mu_a = stats["mu_A"]
        ell = stats["L"]
        self.sigma_shrunk = ell @ ell.T  # the task-locked shrunk covariance (λ=0.1)
        cache = workdir / "eig_cache.npz"
        if cache.is_file():
            z = np.load(cache)
            self.w, self.v = z["w"], z["v"]
            print("[prep] eig cache hit", flush=True)
        else:
            t0 = time.time()
            self.w, self.v = np.linalg.eigh(self.sigma_shrunk)  # ascending
            print(f"[prep] eigh({H_DIM}) in {time.time() - t0:.1f}s", flush=True)
            np.savez(cache, w=self.w, v=self.v)
        assert self.w.min() > 0, f"shrunk covariance not PD (min eig {self.w.min():.3e})"
        # raw covariance recovery: diagonal-target shrinkage preserves the diagonal,
        # off-diagonals scale by (1−λ) — exact inverse (jitter ladder step 0.0 assumed;
        # PD-ness of the 88k-row shrunk covariance makes a jitter add implausible).
        off = self.sigma_shrunk / (1.0 - BANKED_LAMBDA)
        d_idx = np.arange(H_DIM)
        off[d_idx, d_idx] = np.diag(self.sigma_shrunk)
        self.sigma_raw = off
        self._chol_cache: dict[float, np.ndarray] = {BANKED_LAMBDA: ell}

    def center(self, x: np.ndarray) -> np.ndarray:
        return x - self.mu_a

    def trunc_whiten(self, x: np.ndarray, k_top: int) -> np.ndarray:
        """Whiten the top-k eigendirections of Σ_shrunk, identity elsewhere
        (rank-k update: x_c + V_k((w_k^{−1/2}−1)⊙(x_c V_k)))."""
        xc = self.center(x)
        if k_top >= H_DIM:
            vk, wk = self.v, self.w
        else:
            vk, wk = self.v[:, -k_top:], self.w[-k_top:]
        proj = xc @ vk
        return xc + (proj * (wk**-0.5 - 1.0)) @ vk.T

    def alpha_whiten(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """Fractional-power whitening W = V·diag(w^{−α/2})·Vᵀ (invented)."""
        xc = self.center(x)
        return (xc @ self.v) * (self.w ** (-alpha / 2.0)) @ self.v.T

    def diag_whiten(self, x: np.ndarray) -> np.ndarray:
        return self.center(x) / np.sqrt(np.diag(self.sigma_shrunk))

    def chol_whiten(self, x: np.ndarray, lam: float) -> np.ndarray:
        """z = L(λ)⁻¹(x − μ_A) — the banked Cholesky-basis whitening at shrinkage λ."""
        from scipy.linalg import solve_triangular

        if lam not in self._chol_cache:
            self._chol_cache[lam] = shrunk_cholesky_from_cov(self.sigma_raw, lam)
        return solve_triangular(self._chol_cache[lam], self.center(x).T, lower=True).T

    def abtt(self, x: np.ndarray, d_top: int) -> np.ndarray:
        """All-but-the-top: center + remove the top-d_top eigendirections."""
        xc = self.center(x)
        vk = self.v[:, -d_top:]
        return xc - (xc @ vk) @ vk.T

    def apply(self, x: np.ndarray, spec: dict) -> np.ndarray:
        name = spec["name"]
        if name.startswith("truncwhiten_"):
            return self.trunc_whiten(x, spec["params"]["k_top"])
        if name.startswith("diagwhiten_"):
            return self.diag_whiten(x)
        if name.startswith("whitencos_lam"):
            return self.chol_whiten(x, spec["params"]["lam"])
        if name.startswith("abtt_"):
            return self.abtt(x, spec["params"]["removed_top_dirs"])
        if name.startswith("alphawhiten_"):
            return self.alpha_whiten(x, spec["params"]["alpha"])
        raise ValueError(f"no transform for {name}")


# ── rescoring machinery (pool-internal statistics computed ONCE per base) ─────────


def pool_internal_euclid(y16: np.ndarray) -> dict:
    """Pool-internal squared-euclid stats: sorted true-dist rows (self excluded),
    NICDM r_k, DSL local centroid norms — computed once, reused."""
    n = y16.shape[0]
    d_pp = _pairwise_dist(y16, y16, "euclidean")
    np.fill_diagonal(d_pp, np.inf)
    d_pp_true = np.sqrt(np.maximum(d_pp, 0.0))
    sorted_true = np.sort(d_pp_true, axis=1)  # per-row ascending; last col inf (self)
    r_k_pool = sorted_true[:, :K_LOCAL].mean(axis=1)
    nn_idx = np.argpartition(d_pp, K_LOCAL, axis=1)[:, :K_LOCAL]
    cents = y16[nn_idx].mean(axis=1)  # (n, H) local kNN centroids
    dsl_pool_term = ((y16 - cents) ** 2).sum(axis=1)
    del d_pp
    return {
        "sorted_true": sorted_true,
        "r_k_pool": r_k_pool,
        "dsl_pool_term": dsl_pool_term,
        "n": n,
    }


def mp_ranks(d_qp_sq: np.ndarray, pool_stats: dict, true_idx: np.ndarray) -> np.ndarray:
    """Mutual Proximity (empirical, independence approximation) mid-ranks."""
    n_q, n_p = d_qp_sq.shape
    d_true = np.sqrt(np.maximum(d_qp_sq, 0.0))
    # P1: query-side — fraction of pool farther than j (rank transform per row)
    order = np.argsort(d_true, axis=1, kind="stable")
    rank_asc = np.empty_like(order)
    rows = np.arange(n_q)[:, None]
    rank_asc[rows, order] = np.arange(n_p)[None, :]
    p1 = (n_p - 1 - rank_asc) / (n_p - 1)
    del order, rank_asc
    # P2: candidate-side — fraction of j's pool-internal distances larger than d(q,j)
    sorted_true = pool_stats["sorted_true"]
    n_valid = n_p - 1  # self excluded (inf sentinel sorts last)
    p2 = np.empty_like(p1)
    for j in range(n_p):
        p2[:, j] = (n_valid - np.searchsorted(sorted_true[j], d_true[:, j])) / n_valid
    ranks = ranks_score_matrix(p1 * p2, true_idx)
    del p1, p2
    return ranks


def nicdm_ranks(d_qp_sq: np.ndarray, pool_stats: dict, true_idx: np.ndarray) -> np.ndarray:
    """NICDM k=10 mid-ranks: d' = d/sqrt(r_k(q)·r_k(j)) on true euclid."""
    d_true = np.sqrt(np.maximum(d_qp_sq, 0.0))
    r_q = np.sort(d_true, axis=1)[:, :K_LOCAL].mean(axis=1)
    d_resc = d_true / np.sqrt(np.outer(r_q, pool_stats["r_k_pool"]))
    return midranks_of_true(d_resc, true_idx)


def dsl_ranks(
    d_qp_sq: np.ndarray, q: np.ndarray, y16: np.ndarray, pool_stats: dict, true_idx: np.ndarray
) -> np.ndarray:
    """DisSimLocal k=10 mid-ranks on squared euclid."""
    nn_idx = np.argpartition(d_qp_sq, K_LOCAL, axis=1)[:, :K_LOCAL]
    cents = y16[nn_idx].mean(axis=1)
    q_term = ((q - cents) ** 2).sum(axis=1)
    d_resc = d_qp_sq - q_term[:, None] - pool_stats["dsl_pool_term"][None, :]
    return midranks_of_true(d_resc, true_idx)


def isf_ranks(s_cos: np.ndarray, beta: float, true_idx: np.ndarray) -> np.ndarray:
    """Inverted-softmax mid-ranks: score = βS − logsumexp over the query bank."""
    bs = beta * s_cos
    col_lse = bs.max(axis=0) + np.log(np.exp(bs - bs.max(axis=0)[None, :]).sum(axis=0))
    return ranks_score_matrix(bs - col_lse[None, :], true_idx)


def csls_ranks(s: np.ndarray, true_idx: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """CSLS mid-ranks (cross-domain, issue1901 formulation). gamma=0.5 is exact
    CSLS rank-wise (2S − r_q − r_j ≡ S − r_j/2 within a row); gamma≠0.5 is the
    penalty-strength sensitivity."""
    n_q, _n_p = s.shape
    k = K_LOCAL
    top_p = np.partition(s, n_q - k, axis=0)[n_q - k :, :]
    r_p = top_p.mean(axis=0)
    return ranks_score_matrix(s - gamma * r_p[None, :], true_idx)


def hubdeg_ranks(
    s_qp: np.ndarray, s_pp: np.ndarray, true_idx: np.ndarray, gamma: float
) -> np.ndarray:
    """INVENTED in-degree hub penalty: N_k(j) = pool-internal kNN in-degree;
    score = S − γ·σ_S·zscore(N_k)."""
    n_p = s_pp.shape[0]
    s_pp = s_pp.copy()
    np.fill_diagonal(s_pp, -np.inf)
    topk = np.argpartition(-s_pp, K_LOCAL, axis=1)[:, :K_LOCAL]
    n_deg = np.zeros(n_p, dtype=np.float64)
    np.add.at(n_deg, topk.ravel(), 1.0)
    z = (n_deg - n_deg.mean()) / (n_deg.std() + 1e-12)
    sig = float(s_qp.std())
    return ranks_score_matrix(s_qp - gamma * sig * z[None, :], true_idx)


# ── battery driver ─────────────────────────────────────────────────────────────────


def load_done(results_path: Path) -> dict[str, dict]:
    """Resume predicate: completed convention records keyed by name."""
    done: dict[str, dict] = {}
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if "summary" in rec:  # JSON round-trip stringifies the int acc@k keys
                    rec["summary"]["acc_at_k"] = {
                        int(k): v for k, v in rec["summary"]["acc_at_k"].items()
                    }
                done[rec["name"]] = rec
    return done


def append_result(results_path: Path, rec: dict) -> None:
    """Atomic-append one convention record (O_APPEND single-line write)."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_battery(args, data: dict, tf: Transforms, results_path: Path) -> dict[str, dict]:
    """All roster + sensitivity + reconciliation conventions, checkpointed per unit."""
    pred, y16, _pci = data["pred"], data["y16"], data["pci"]
    n = pred.shape[0]
    true_idx = np.arange(n)
    done = load_done(results_path)
    t_all = time.time()

    def record(name: str, ranks: np.ndarray, spec: dict | None, wall: float, **extra) -> dict:
        rec = {
            "name": name,
            "summary": ranks_summary(ranks, n),
            "wall_s": round(wall, 1),
            **(
                {"roster": {k: spec[k] for k in ("kind", "params", "mechanism", "citation")}}
                if spec
                else {}
            ),
            **extra,
        }
        append_result(results_path, rec)
        done[name] = rec
        k = len(done)
        print(
            f"[zoo] unit {k} {name} acc1={rec['summary']['acc_at_k'][1]:.4f} "
            f"acc5={rec['summary']['acc_at_k'][5]:.4f} wall={wall:.1f}s "
            f"elapsed={time.time() - t_all:.1f}s",
            flush=True,
        )
        return rec

    # ── reconciliation gates first ──
    if "recon_raw_euclidean" not in done:
        t0 = time.time()
        ranks = ranks_transform(pred, y16, true_idx, "euclidean", args.chunk_rows, "recon-euc")
        rec = record(
            "recon_raw_euclidean", ranks, None, time.time() - t0, banked_acc1=BANKED_ACC1_EUC
        )
        delta = abs(rec["summary"]["acc_at_k"][1] - BANKED_ACC1_EUC)
        if delta > ACC_TOL_ROWS / n + 1e-12:
            print(f"[recon] RAW-EUCLIDEAN GATE FAILED delta={delta:.6f}", flush=True)
            sys.exit(RC_GATE)
        # centered-euclidean identity check (translation invariance), 256-row slice
        d_raw = _pairwise_dist(pred[:256], y16, "euclidean")
        d_cent = _pairwise_dist(pred[:256] - tf.mu_a, y16 - tf.mu_a, "euclidean")
        assert np.allclose(d_raw, d_cent, rtol=1e-9, atol=1e-6), "centered-euclid identity failed"
        print("[recon] centered-euclidean ≡ raw-euclidean verified on 256-row slice", flush=True)
    if "recon_raw_cos" not in done:
        t0 = time.time()
        ranks = ranks_transform(pred, y16, true_idx, "cosine", args.chunk_rows, "recon-cos")
        record("recon_raw_cos", ranks, None, time.time() - t0, banked_acc1=0.8281862991650739)

    # ── transform-family roster lines ──
    for spec in ROSTER:
        if spec["name"] in done or spec["kind"] == "rescore":
            continue
        t0 = time.time()
        metric = "euclidean" if spec["kind"] == "transform-euc" else "cosine"
        pred_t = tf.apply(pred, spec)
        pool_t = tf.apply(y16, spec)
        ranks = ranks_transform(pred_t, pool_t, true_idx, metric, args.chunk_rows, spec["name"])
        record(spec["name"], ranks, spec, time.time() - t0)
        del pred_t, pool_t

    # ── truncated-whitening k=d endpoints (reconciliation + basis-effect sensitivity) ──
    for name, metric, banked in (
        ("truncwhiten_kfull_euc", "euclidean", BANKED_ACC1_WHITEN_EUC),
        ("truncwhiten_kfull_cos", "cosine", None),
    ):
        if name in done:
            continue
        t0 = time.time()
        pred_t = tf.trunc_whiten(pred, H_DIM)
        pool_t = tf.trunc_whiten(y16, H_DIM)
        ranks = ranks_transform(pred_t, pool_t, true_idx, metric, args.chunk_rows, name)
        rec = record(
            name,
            ranks,
            None,
            time.time() - t0,
            sensitivity=True,
            note=(
                "k=d endpoint; euclid read must reproduce banked whiten (basis-invariant); "
                "cos read quantifies ZCA-vs-Cholesky basis effect vs banked 0.9535"
            ),
            banked_acc1=banked,
        )
        del pred_t, pool_t
        if banked is not None:
            delta = abs(rec["summary"]["acc_at_k"][1] - banked)
            if delta > 0.005:
                print(f"[recon] TRUNC-WHITEN k=d GATE FAILED delta={delta:.6f}", flush=True)
                sys.exit(RC_GATE)

    # ── euclidean-base rescorings (pool stats once) ──
    need_euc = [nm for nm in ("mp_emp_euc", "nicdm_k10_euc", "dsl_k10_euc") if nm not in done]
    if need_euc:
        t0 = time.time()
        pool_stats = pool_internal_euclid(y16)
        d_qp = _pairwise_dist(pred, y16, "euclidean")
        print(f"[rescore-euc] pool stats + d_qp in {time.time() - t0:.1f}s", flush=True)
        spec_of = {s["name"]: s for s in ROSTER}
        if "mp_emp_euc" in need_euc:
            t1 = time.time()
            record(
                "mp_emp_euc",
                mp_ranks(d_qp, pool_stats, true_idx),
                spec_of["mp_emp_euc"],
                time.time() - t1,
            )
        if "nicdm_k10_euc" in need_euc:
            t1 = time.time()
            record(
                "nicdm_k10_euc",
                nicdm_ranks(d_qp, pool_stats, true_idx),
                spec_of["nicdm_k10_euc"],
                time.time() - t1,
            )
        if "dsl_k10_euc" in need_euc:
            t1 = time.time()
            record(
                "dsl_k10_euc",
                dsl_ranks(d_qp, pred, y16, pool_stats, true_idx),
                spec_of["dsl_k10_euc"],
                time.time() - t1,
            )
        del d_qp, pool_stats

    # ── cosine-base ISF ──
    spec_of = {s["name"]: s for s in ROSTER}
    if "isf_cos_b30" not in done or "isf_cos_b10" not in done:
        pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
        qn = y16 / (np.linalg.norm(y16, axis=1, keepdims=True) + 1e-12)
        s_cos = pn @ qn.T
        if "isf_cos_b30" not in done:
            t1 = time.time()
            record(
                "isf_cos_b30",
                isf_ranks(s_cos, ISF_BETA, true_idx),
                spec_of["isf_cos_b30"],
                time.time() - t1,
            )
        if "isf_cos_b10" not in done:
            t1 = time.time()
            record(
                "isf_cos_b10",
                isf_ranks(s_cos, 10.0, true_idx),
                None,
                time.time() - t1,
                sensitivity=True,
            )
        del s_cos, pn, qn

    # ── whitened-cosine-base rescorings ──
    need_wc = [
        nm
        for nm in ("csls_k10_whitencos", "csls_pen_whitencos_g10", "hubdeg_pen_whitencos_g05")
        if nm not in done
    ]
    if need_wc:
        t0 = time.time()
        pred_w = tf.chol_whiten(pred, BANKED_LAMBDA)
        pool_w = tf.chol_whiten(y16, BANKED_LAMBDA)
        pwn = pred_w / (np.linalg.norm(pred_w, axis=1, keepdims=True) + 1e-12)
        qwn = pool_w / (np.linalg.norm(pool_w, axis=1, keepdims=True) + 1e-12)
        s_wc = pwn @ qwn.T
        print(f"[rescore-wc] whitened-cos S in {time.time() - t0:.1f}s", flush=True)
        if "csls_k10_whitencos" in need_wc:
            t1 = time.time()
            record(
                "csls_k10_whitencos",
                csls_ranks(s_wc, true_idx, 0.5),
                spec_of["csls_k10_whitencos"],
                time.time() - t1,
            )
        if "csls_pen_whitencos_g10" in need_wc:
            t1 = time.time()
            record(
                "csls_pen_whitencos_g10",
                csls_ranks(s_wc, true_idx, 1.0),
                None,
                time.time() - t1,
                sensitivity=True,
                note="double-strength candidate penalty S − r_j (CSLS ≡ γ=0.5)",
            )
        if "hubdeg_pen_whitencos_g05" in need_wc:
            t1 = time.time()
            s_pp_wc = qwn @ qwn.T
            record(
                "hubdeg_pen_whitencos_g05",
                hubdeg_ranks(s_wc, s_pp_wc, true_idx, 0.5),
                spec_of["hubdeg_pen_whitencos_g05"],
                time.time() - t1,
            )
            del s_pp_wc
        del s_wc, pred_w, pool_w, pwn, qwn

    return done


# ── fresh-draw ceilings (convention-matched, banked definition) ────────────────────


def ceiling_from_ranks(kranks_flat: np.ndarray, n_k: int, k_draws: int) -> dict:
    """Banked definition (issue2202_failchar phase_extract → attribution.json):
    s_i = per-context rank-1 fraction over K draws; ceiling = s_i.mean()."""
    kranks = kranks_flat.reshape(n_k, k_draws)
    s_i = (kranks == 1.0).mean(axis=1)
    return {
        "acc1_ceiling": float(s_i.mean()),
        "n_contexts": int(n_k),
        "k_draws": int(k_draws),
        "acc5_ceiling_rowlevel": float((kranks <= 5).mean()),
    }


def run_ceilings(args, data: dict, tf: Transforms, done: dict, results_path: Path) -> None:
    """Raw-euclid ceiling reconciliation + convention-matched ceilings for the
    top-3 NEW conventions by acc@1."""
    y16 = data["y16"]
    q, true_idx, kci = load_kresample(Path(args.staged), data["pci"])
    n_k = len(kci)
    k_draws = q.shape[0] // n_k

    def ceil_record(name: str, kranks: np.ndarray, wall: float, **extra) -> dict:
        rec = {
            "name": f"ceiling_{name}",
            "ceiling": ceiling_from_ranks(kranks, n_k, k_draws),
            "wall_s": round(wall, 1),
            **extra,
        }
        append_result(results_path, rec)
        done[rec["name"]] = rec
        print(
            f"[ceil] {name} acc1_ceiling={rec['ceiling']['acc1_ceiling']:.4f} wall={wall:.1f}s",
            flush=True,
        )
        return rec

    if "ceiling_raw_euclidean" not in done:
        t0 = time.time()
        kranks = ranks_transform(q, y16, true_idx, "euclidean", args.chunk_rows, "ceil-recon")
        rec = ceil_record("raw_euclidean", kranks, time.time() - t0, banked=BANKED_CEILING_EUC)
        delta = abs(rec["ceiling"]["acc1_ceiling"] - BANKED_CEILING_EUC)
        if delta > 0.002:
            print(f"[recon] CEILING GATE FAILED delta={delta:.6f}", flush=True)
            sys.exit(RC_GATE)

    new_names = [s["name"] for s in ROSTER]
    ranked = sorted(
        (nm for nm in new_names if nm in done),
        key=lambda nm: done[nm]["summary"]["acc_at_k"][1],
        reverse=True,
    )
    top3 = ranked[:3]
    print(f"[ceil] top-3 new conventions: {top3}", flush=True)
    spec_of = {s["name"]: s for s in ROSTER}
    for nm in top3:
        if f"ceiling_{nm}" in done:
            continue
        spec = spec_of[nm]
        t0 = time.time()
        if spec["kind"] != "rescore":
            metric = "euclidean" if spec["kind"] == "transform-euc" else "cosine"
            q_t = tf.apply(q, spec)
            pool_t = tf.apply(y16, spec)
            kranks = ranks_transform(q_t, pool_t, true_idx, metric, args.chunk_rows, f"ceil-{nm}")
            del q_t, pool_t
        elif nm == "csls_k10_whitencos":
            q_w = tf.chol_whiten(q, BANKED_LAMBDA)
            pool_w = tf.chol_whiten(y16, BANKED_LAMBDA)
            qn_ = q_w / (np.linalg.norm(q_w, axis=1, keepdims=True) + 1e-12)
            pn_ = pool_w / (np.linalg.norm(pool_w, axis=1, keepdims=True) + 1e-12)
            kranks = csls_ranks(qn_ @ pn_.T, true_idx, 0.5)
            del q_w, pool_w, qn_, pn_
        elif nm == "hubdeg_pen_whitencos_g05":
            q_w = tf.chol_whiten(q, BANKED_LAMBDA)
            pool_w = tf.chol_whiten(y16, BANKED_LAMBDA)
            qn_ = q_w / (np.linalg.norm(q_w, axis=1, keepdims=True) + 1e-12)
            pn_ = pool_w / (np.linalg.norm(pool_w, axis=1, keepdims=True) + 1e-12)
            kranks = hubdeg_ranks(qn_ @ pn_.T, pn_ @ pn_.T, true_idx, 0.5)
            del q_w, pool_w, qn_, pn_
        elif nm == "isf_cos_b30":
            qn_ = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            pn_ = y16 / (np.linalg.norm(y16, axis=1, keepdims=True) + 1e-12)
            kranks = isf_ranks(qn_ @ pn_.T, ISF_BETA, true_idx)
            del qn_, pn_
        elif nm in ("mp_emp_euc", "nicdm_k10_euc", "dsl_k10_euc"):
            pool_stats = pool_internal_euclid(y16)
            d_qp = _pairwise_dist(q, y16, "euclidean")
            if nm == "mp_emp_euc":
                kranks = mp_ranks(d_qp, pool_stats, true_idx)
            elif nm == "nicdm_k10_euc":
                kranks = nicdm_ranks(d_qp, pool_stats, true_idx)
            else:
                kranks = dsl_ranks(d_qp, q, y16, pool_stats, true_idx)
            del d_qp, pool_stats
        else:
            raise ValueError(f"no ceiling recipe for {nm}")
        ceil_record(nm, kranks, time.time() - t0, convention_matched=True)


# ── summary composition ─────────────────────────────────────────────────────────────


BANKED_TABLE = {
    "raw_euclidean": {
        "acc1": 0.816014485464239,
        "source": "mapping_baselines.json via csls_followup.json",
    },
    "raw_cos": {"acc1": 0.8281862991650739, "source": "same"},
    "cent_cos": {"acc1": (9941 - 1169) / 9941, "source": "geometry_summary.json fail_counts"},
    "whiten_euc": {"acc1": (9941 - 9739) / 9941, "source": "geometry_summary.json fail_counts"},
    "whiten_cos": {"acc1": (9941 - 462) / 9941, "source": "geometry_summary.json fail_counts"},
    "csls_k10_raw_cos": {"acc1": 0.9094658485061865, "source": "csls_followup.json"},
}


def compose_summary(args, done: dict, data: dict) -> None:
    """summary.json: roster + citations, ranked table, ceilings, provenance."""
    # banked acc@5 for the five geometry spaces from percontext_ranks.csv
    import csv as _csv

    banked5: dict[str, float] = {}
    pcr = PROJECT_ROOT / "eval_results" / "issue_2202" / "percontext_ranks.csv"
    if pcr.is_file():
        cols = {sp: [] for sp in ("raw_euclidean", "raw_cos", "cent_cos", "whiten", "whiten_cos")}
        with open(pcr, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                for sp in cols:
                    cols[sp].append(float(row[f"rank_{sp}"]))
        for sp, v in cols.items():
            r = np.asarray(v)
            banked5[sp] = {"acc1": float((r <= 1).mean()), "acc5": float((r <= 5).mean())}

    new_rows = []
    for spec in ROSTER:
        rec = done.get(spec["name"])
        if rec is None:
            continue
        new_rows.append(
            {
                "name": spec["name"],
                "acc1": rec["summary"]["acc_at_k"][1],
                "acc5": rec["summary"]["acc_at_k"][5],
                "mrr": rec["summary"]["mrr"],
                "kind": spec["kind"],
                "mechanism": spec["mechanism"],
                "citation": spec["citation"],
            }
        )
    new_rows.sort(key=lambda r: -r["acc1"])
    sens_rows = [
        {
            "name": nm,
            "acc1": rec["summary"]["acc_at_k"][1],
            "acc5": rec["summary"]["acc_at_k"][5],
            "note": rec.get("note", ""),
        }
        for nm, rec in done.items()
        if rec.get("sensitivity") and "summary" in rec
    ]
    ceilings = {nm: rec for nm, rec in done.items() if nm.startswith("ceiling_")}
    recon = {
        nm: done[nm]
        for nm in ("recon_raw_euclidean", "recon_raw_cos", "truncwhiten_kfull_euc")
        if nm in done
    }
    summary = {
        "round": "metric-zoo (user-chat inline free-analysis, task #2202)",
        "definition": (
            "full-pool retrieval: 9,941 held-out layer-19 ridge predictions vs the 9,941 "
            "held-out answer vectors; acc@1 = fraction with the true answer at mid-rank 1 "
            "(knn_retrieval tie convention). Ceilings: per-context rank-1 fraction over the "
            "1,988×4 fresh kresample draws, mean over contexts (banked definition)."
        ),
        "staging": {
            "dir": args.staged,
            "provenance": (
                "sibling round freshwhiten-avg's staged copies of "
                "issue1738_multiturn @ 09788eef (pred16, y_holdout, kresample shard) + "
                "issue2202_ctxfail/analysis_tensors (whiten_stats); every file byte-size-"
                "verified vs HF metadata before reuse (READ-ONLY)"
            ),
        },
        "banked_baselines": {
            "acc1_table": BANKED_TABLE,
            "geometry_spaces_acc": banked5,
            "fresh_draw_ceiling_raw_euclidean": BANKED_CEILING_EUC,
        },
        "reconciliation": recon,
        "new_conventions_ranked": new_rows,
        "sensitivity_records": sens_rows,
        "ceilings": ceilings,
        "dropped_roster_lines": DROPPED,
        "n_train_vs_d": {
            "n_train": data["stats"]["n_train"],
            "d": H_DIM,
            "note": "no fits performed; all transforms closed-form from banked train stats",
        },
        "meta": meta_block(),
    }
    atomic_json(Path(args.out_eval) / "summary.json", summary)
    print(f"[summary] wrote {Path(args.out_eval) / 'summary.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staged", default=STAGED_DEFAULT)
    ap.add_argument("--workdir", default=WORKDIR_DEFAULT)
    ap.add_argument("--out-eval", default=OUT_EVAL_DEFAULT)
    ap.add_argument("--chunk-rows", type=int, default=2048)
    ap.add_argument(
        "--phase", choices=("recon", "battery", "ceilings", "summary", "all"), default="all"
    )
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.out_eval) / "results.jsonl"

    t0 = time.time()
    data = load_staged(Path(args.staged))
    tf = Transforms(data["stats"], workdir)
    print(f"[load] staged inputs + transforms ready in {time.time() - t0:.1f}s", flush=True)

    if args.phase in ("recon", "battery", "all"):
        done = run_battery(args, data, tf, results_path)
        if args.phase == "recon":
            return
    else:
        done = load_done(results_path)
    if args.phase in ("ceilings", "all"):
        run_ceilings(args, data, tf, done, results_path)
    if args.phase in ("summary", "all"):
        compose_summary(args, done, data)
    print(f"[done] total elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
