#!/usr/bin/env python3
"""Is answer-side crowding downstream of context-side crowding?

Tests the manuscript claim that, once you know how crowded a held-out context's
neighbourhood is in CONTEXT space, the crowding of its answer's neighbourhood in
ANSWER space carries no additional information about whether top-1 retrieval
fails.

Setting: the 10,000-candidate operating point, 942 held-out contexts, 90 top-1
retrieval failures.

Three nearest-neighbour DISTANCES per context (higher = more isolated, lower =
more crowded):

    dC  context space, = 1 - nn1 (nn1 is the cosine to the nearest OTHER
        candidate context; the identity holds to float32 epsilon).
    dA  answer space, WHITENED (the retrieval convention), read from disk.
    dR  answer space, RAW, reconstructed here: mean-centre the pool answers,
        unit-norm the rows, cosine the 942 query rows against the pool, mask the
        self column, take 1 - max.

Reported:

    1. Coupling between dC and each answer-side distance (Spearman + Pearson),
       over all 942 and over successes only.
    2. Logistic regression of fail on both standardized distances, plus the two
       single-predictor models, once with dA and once with dR.
    3. Partial AUCs: each distance residualized on the other (OLS), AUC of the
       residual against fail, with bootstrap CIs. Plus the unconditional AUCs.
    4. Likelihood-ratio tests in both directions.
    5. Top-decile crowding counts and their overlap.
    6. Robustness: non-parametric (quantile-bin) control for context crowding,
       including the pair-weighted stratified Mann-Whitney AUC of ANSWER
       closeness within CONTEXT-crowding strata.
    7. The MIRROR of 6: the stratified AUC of CONTEXT closeness within
       ANSWER-crowding strata, under the identical construction, bin schemes,
       resample count, seed and CI method -- plus the paired bootstrap of the
       difference between the two sides, and per-stratum pair-weight and
       sparsity flags on both. The manuscript makes a claim in each direction,
       so both readings have to come off ONE instrument.

AUC ORIENTATION. Failures sit in CROWDED neighbourhoods, so every AUC here
scores CLOSENESS, i.e. the negated distance. AUC > 0.5 therefore means "more
crowded predicts failure", the direction the manuscript claims. The same
negation is applied to residuals.

This script is the durable record for numbers that previously existed only in a
chat transcript. Everything is recomputed from the banked arrays; no statistic
is hardcoded. The published sanity-check values are ASSERTED, not assumed: a
mismatch aborts rather than silently reporting numbers off the wrong inputs.

Usage:
    uv run python scripts/issue1901_crowding_mediation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# #847: thread caps must land BEFORE the numpy/scipy imports; load_dotenv()
# setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS and BLAS pools freeze at
# import. Single-threaded BLAS also keeps the float32 dR reconstruction
# bit-reproducible (see provenance.blas_thread_env).
import numpy as np  # noqa: E402
import scipy.stats as st  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

DEFAULT_INPUT_DIR = Path("/mnt/eps-data/thomasjiralerspong/issue1901_ctxsim/paper_fig_inputs")
DEFAULT_OUT_JSON = Path("eval_results/issue_1901/crowding_mediation/summary.json")

# Published sanity checks for the operating point. A mismatch means the inputs
# are not the arrays this analysis was specified against -> abort.
EXPECT_N = 942
EXPECT_N_FAIL = 90
EXPECT_NN1_MEDIAN_FAIL = 0.974556
EXPECT_NN1_MEDIAN_SUCCESS = 0.778286
EXPECT_DA_MEDIAN_FAIL = 0.5675
EXPECT_DA_MEDIAN_SUCCESS = 0.7979
EXPECT_DA_MIN = 0.0999
EXPECT_DA_MEDIAN = 0.7855
EXPECT_DA_MAX = 0.9718
SANITY_TOL = 1e-3

# A stratum contributes to a stratified Mann-Whitney statistic only through its
# discordant pairs (n_fail * n_success). Below these floors the per-stratum AUC
# is estimated off a handful of pairs, so it is FLAGGED as sparse rather than
# read as a per-stratum result. Flagging never changes the pooled statistic --
# the pair weighting already down-weights thin strata -- it only makes visible
# which strata carry the pooled number.
MIN_STRATUM_FAILURES = 5
MIN_STRATUM_SUCCESSES = 5


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_close(name: str, got: float, want: float, tol: float = SANITY_TOL) -> dict:
    ok = abs(float(got) - float(want)) <= tol
    if not ok:
        raise SystemExit(
            f"SANITY CHECK FAILED: {name} = {got!r}, expected {want!r} "
            f"(tolerance {tol}). Refusing to proceed on unrecognized inputs."
        )
    return {"name": name, "got": float(got), "expected": float(want), "ok": True}


def load_inputs(input_dir: Path) -> dict:
    """Load the banked arrays and reconstruct the raw-answer NN distance."""
    perrow_path = input_dir / "perrow.npz"
    da_path = input_dir / "c2a_dA.npy"
    a_path = input_dir / "c2a_A.npy"
    qcols_path = input_dir / "c2a_qcols.npy"
    for p in (perrow_path, da_path, a_path, qcols_path):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    z = np.load(perrow_path)
    nn1 = np.asarray(z["nn1"], dtype=np.float64)
    correct = np.asarray(z["correct"], dtype=bool)
    dA = np.asarray(np.load(da_path), dtype=np.float64)
    qcols = np.asarray(np.load(qcols_path))

    # Context-side NN distance from the banked cosine.
    dC = 1.0 - nn1

    # Raw-answer-space NN distance: mean-centre the pool, unit-norm rows,
    # cosine the query rows against the pool, mask self, 1 - max.
    t0 = time.time()
    A = np.load(a_path).astype(np.float32)
    Ac = A - A.mean(axis=0, keepdims=True)
    Ac /= np.linalg.norm(Ac, axis=1, keepdims=True)
    sim = Ac[qcols] @ Ac.T
    sim[np.arange(len(qcols)), qcols] = -np.inf
    dR = (1.0 - sim.max(axis=1)).astype(np.float64)
    dr_secs = time.time() - t0
    del A, Ac, sim

    return {
        "nn1": nn1,
        "correct": correct,
        "fail": (~correct).astype(np.int64),
        "dC": dC,
        "dA": dA,
        "dR": dR,
        "qcols": qcols,
        "dr_reconstruct_seconds": dr_secs,
        "paths": {
            "perrow_npz": str(perrow_path),
            "c2a_dA_npy": str(da_path),
            "c2a_A_npy": str(a_path),
            "c2a_qcols_npy": str(qcols_path),
        },
        "shapes": {
            "nn1": list(nn1.shape),
            "correct": list(correct.shape),
            "c2a_dA": list(dA.shape),
            "c2a_A": [10000, 3584],
            "c2a_qcols": list(qcols.shape),
        },
    }


def run_sanity(d: dict) -> list[dict]:
    nn1, correct, dA = d["nn1"], d["correct"], d["dA"]
    n = int(nn1.size)
    checks = [
        check_close("n_contexts", n, EXPECT_N, 0),
        check_close("n_failures", int((~correct).sum()), EXPECT_N_FAIL, 0),
        check_close("n_successes", int(correct.sum()), EXPECT_N - EXPECT_N_FAIL, 0),
        check_close("nn1_median_failures", np.median(nn1[~correct]), EXPECT_NN1_MEDIAN_FAIL),
        check_close("nn1_median_successes", np.median(nn1[correct]), EXPECT_NN1_MEDIAN_SUCCESS),
        check_close("dA_median_failures", np.median(dA[~correct]), EXPECT_DA_MEDIAN_FAIL),
        check_close("dA_median_successes", np.median(dA[correct]), EXPECT_DA_MEDIAN_SUCCESS),
        check_close("dA_min", dA.min(), EXPECT_DA_MIN),
        check_close("dA_median", np.median(dA), EXPECT_DA_MEDIAN),
        check_close("dA_max", dA.max(), EXPECT_DA_MAX),
    ]
    # dC = 1 - nn1 identity, to float32 epsilon.
    ident = float(np.max(np.abs(d["dC"] - (1.0 - nn1))))
    checks.append({"name": "dC_identity_max_abs_dev", "got": ident, "expected": 0.0, "ok": True})
    return checks


# --------------------------------------------------------------------------
# 1. Coupling
# --------------------------------------------------------------------------
def coupling(x: np.ndarray, y: np.ndarray) -> dict:
    sp = st.spearmanr(x, y)
    pe = st.pearsonr(x, y)
    return {
        "n": int(x.size),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pe.statistic),
        "pearson_p": float(pe.pvalue),
    }


# --------------------------------------------------------------------------
# 2. Logistic regression
# --------------------------------------------------------------------------
def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=0)


def logit_fit(y: np.ndarray, cols: dict[str, np.ndarray]) -> dict:
    names = list(cols)
    X = np.column_stack([cols[k] for k in names])
    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X)
    res = model.fit(disp=0, maxiter=200)
    labels = ["const"] + names
    ci = res.conf_int(alpha=0.05)
    terms = {}
    for i, lab in enumerate(labels):
        terms[lab] = {
            "coef": float(res.params[i]),
            "std_err": float(res.bse[i]),
            "wald_z": float(res.tvalues[i]),
            "p_value": float(res.pvalues[i]),
            "ci95_low": float(ci[i, 0]),
            "ci95_high": float(ci[i, 1]),
            "odds_ratio": float(np.exp(res.params[i])),
        }
    return {
        "predictors": names,
        "n_obs": int(res.nobs),
        "converged": bool(res.mle_retvals.get("converged", True)),
        "llf": float(res.llf),
        "llnull": float(res.llnull),
        "pseudo_r2_mcfadden": float(res.prsquared),
        "terms": terms,
    }


def lr_test(full: dict, reduced: dict, added: str) -> dict:
    stat = 2.0 * (full["llf"] - reduced["llf"])
    df = len(full["predictors"]) - len(reduced["predictors"])
    p = float(st.chi2.sf(stat, df)) if df > 0 else float("nan")
    return {
        "added_predictor": added,
        "reduced_model": reduced["predictors"],
        "full_model": full["predictors"],
        "lr_statistic": float(stat),
        "df": int(df),
        "p_value": p,
        "llf_reduced": reduced["llf"],
        "llf_full": full["llf"],
    }


# --------------------------------------------------------------------------
# 3. AUCs (unconditional + partial) with bootstrap CIs
# --------------------------------------------------------------------------
def residualize(target: np.ndarray, on: np.ndarray) -> np.ndarray:
    """OLS residuals of `target` regressed on `on` (with intercept)."""
    X = np.column_stack([np.ones_like(on), on])
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    return target - X @ beta


def auc_closeness(fail: np.ndarray, distance: np.ndarray) -> float:
    """AUC scoring CLOSENESS (= negated distance) against failure."""
    return float(roc_auc_score(fail, -distance))


def bootstrap_aucs(
    fail: np.ndarray,
    dists: dict[str, np.ndarray],
    partial_specs: list[tuple[str, np.ndarray, np.ndarray]],
    n_boot: int,
    seed: int,
) -> dict:
    """Percentile bootstrap over the 942 contexts.

    `dists` maps an output name to an unconditional distance array.
    `partial_specs` is (output_name, target_array, other_array): the target is
    residualized on the other, and the residual is scored.

    The residualization is refit INSIDE each resample, so the CI covers the
    whole estimator (residualize-then-AUC), not just the AUC step.
    """
    rng = np.random.default_rng(seed)
    n = fail.size
    keys = list(dists)
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for name, _, _ in partial_specs:
        draws[name] = []

    n_skipped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = fail[idx]
        if yb.min() == yb.max():  # degenerate resample, no both-class contrast
            n_skipped += 1
            continue
        for k in keys:
            draws[k].append(auc_closeness(yb, dists[k][idx]))
        for name, tgt, other in partial_specs:
            resid = residualize(tgt[idx], other[idx])
            draws[name].append(auc_closeness(yb, resid))

    out = {}
    for k, vals in draws.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[k] = {
            "boot_mean": float(arr.mean()),
            "boot_sd": float(arr.std(ddof=1)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
            "n_draws": int(arr.size),
            "frac_draws_at_or_below_0.5": float((arr <= 0.5).mean()),
        }
    out["_n_skipped_degenerate"] = n_skipped
    return out


# --------------------------------------------------------------------------
# 6. Robustness: flexible (non-linear) control for context crowding
#
# Failure rate falls steeply and non-linearly across dC (roughly 0.29 / 0.06 /
# 0.01 / 0.02 by quartile), so a single linear-in-z context term cannot absorb
# the context signal. Anything it leaves behind is available for the answer-side
# term to soak up, which would show as a spurious "independent" answer effect.
# Both adjustments below control for context WITHOUT assuming a functional form:
# quantile-bin dummies in a logistic model, and a stratified Mann-Whitney AUC.
# --------------------------------------------------------------------------
def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.digitize(x, edges[1:-1])


def bin_dummy_cols(bins: np.ndarray, n_bins: int) -> dict[str, np.ndarray]:
    """Dummies for bins 1..n-1; bin 0 (most crowded) is the reference."""
    return {f"dCbin{k}": (bins == k).astype(np.float64) for k in range(1, n_bins)}


def flexible_context_test(
    fail: np.ndarray, dC: np.ndarray, z_answer: np.ndarray, label: str, n_bins: int
) -> dict:
    """LRT for the answer term on top of NON-PARAMETRIC context-crowding bins."""
    bins = quantile_bins(dC, n_bins)
    cols = bin_dummy_cols(bins, n_bins)
    reduced = logit_fit(fail, cols)
    full = logit_fit(fail, {**cols, label: z_answer})
    t = full["terms"][label]
    return {
        "n_context_bins": n_bins,
        "answer_term": label,
        "coef": t["coef"],
        "std_err": t["std_err"],
        "wald_z": t["wald_z"],
        "p_value": t["p_value"],
        "ci95_low": t["ci95_low"],
        "ci95_high": t["ci95_high"],
        "lr_test": lr_test(full, reduced, label),
        "reduced_converged": reduced["converged"],
        "full_converged": full["converged"],
        "bin_failure_counts": [int(fail[bins == k].sum()) for k in range(n_bins)],
        "bin_sizes": [int((bins == k).sum()) for k in range(n_bins)],
    }


def stratified_pooled_auc(
    fail: np.ndarray, distance: np.ndarray, strat_var: np.ndarray, n_bins: int
) -> tuple[float, list[dict]]:
    """AUC of `distance` closeness WITHIN quantile strata of `strat_var`, pooled.

    Weighted by each stratum's discordant-pair count (n_fail * n_success), the
    natural weighting for a stratified Mann-Whitney statistic. 0.5 means the
    SCORED side separates nothing once the STRATIFYING side is held fixed.

    The construction is symmetric in the two sides, which is the point: the same
    function computes answer-closeness within context-crowding strata AND
    context-closeness within answer-crowding strata, so both readings come off
    ONE instrument.

    Each per-stratum entry carries its discordant-pair weight, that weight as a
    share of the pooled total, and a `sparse` flag. A stratum with no failures
    or no successes has NO discordant pairs: its AUC is undefined (null) and it
    contributes exactly zero to the pooled number. A stratum below
    MIN_STRATUM_FAILURES / MIN_STRATUM_SUCCESSES does contribute, but its
    per-stratum AUC rests on very few pairs and should not be read on its own.
    """
    bins = quantile_bins(strat_var, n_bins)
    num = den = 0.0
    per: list[dict] = []
    for k in range(n_bins):
        msk = bins == k
        y = fail[msk]
        nf = int(y.sum())
        ns = int(msk.sum() - nf)
        entry = {
            "bin": k,
            "n": int(msk.sum()),
            "n_fail": nf,
            "n_success": ns,
            "auc": None,
            "pair_weight": float(nf * ns),
        }
        if nf > 0 and ns > 0:
            a = auc_closeness(y, distance[msk])
            entry["auc"] = float(a)
            num += nf * ns * a
            den += nf * ns
        per.append(entry)

    for e in per:
        e["pair_weight_share"] = float(e["pair_weight"] / den) if den > 0 else float("nan")
        reasons: list[str] = []
        if e["n_fail"] == 0:
            reasons.append("no failures in stratum (contributes no discordant pairs)")
        elif e["n_fail"] < MIN_STRATUM_FAILURES:
            reasons.append(f"n_fail < {MIN_STRATUM_FAILURES}")
        if e["n_success"] == 0:
            reasons.append("no successes in stratum (contributes no discordant pairs)")
        elif e["n_success"] < MIN_STRATUM_SUCCESSES:
            reasons.append(f"n_success < {MIN_STRATUM_SUCCESSES}")
        e["sparse"] = bool(reasons)
        e["sparse_reason"] = "; ".join(reasons) if reasons else None

    return (float(num / den) if den > 0 else float("nan")), per


def stratum_sparsity_summary(per: list[dict]) -> dict:
    """Which strata carry the pooled statistic, and which are too thin to read.

    `pair_weight_share_flagged_sparse` is the fraction of the pooled statistic's
    total weight that comes from strata flagged sparse: a large share means the
    pooled number leans on strata whose own AUCs rest on few pairs.
    """
    contributing = [e for e in per if e["auc"] is not None]
    flagged = [e for e in per if e["sparse"]]
    heaviest = max(per, key=lambda e: e["pair_weight"]) if per else None
    return {
        "n_strata": len(per),
        "n_strata_contributing": len(contributing),
        "n_strata_flagged_sparse": len(flagged),
        "sparse_bins": [e["bin"] for e in flagged],
        "zero_pair_bins": [e["bin"] for e in per if e["auc"] is None],
        "pair_weight_total": float(sum(e["pair_weight"] for e in per)),
        "pair_weight_share_flagged_sparse": float(
            sum(e["pair_weight_share"] for e in flagged if e["auc"] is not None)
        ),
        "heaviest_stratum_bin": int(heaviest["bin"]) if heaviest is not None else None,
        "pair_weight_share_heaviest_stratum": (
            float(heaviest["pair_weight_share"]) if heaviest is not None else float("nan")
        ),
        "thresholds": {
            "min_stratum_failures": MIN_STRATUM_FAILURES,
            "min_stratum_successes": MIN_STRATUM_SUCCESSES,
        },
    }


def bootstrap_stratified_auc(
    fail: np.ndarray,
    distance: np.ndarray,
    strat_var: np.ndarray,
    n_bins: int,
    n_boot: int,
    seed: int,
) -> dict:
    """Percentile bootstrap; strata edges are recomputed inside each resample."""
    rng = np.random.default_rng(seed)
    n = fail.size
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = fail[idx]
        if yb.min() == yb.max():
            continue
        v, _ = stratified_pooled_auc(yb, distance[idx], strat_var[idx], n_bins)
        if np.isfinite(v):
            vals.append(v)
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "boot_mean": float(arr.mean()),
        "boot_sd": float(arr.std(ddof=1)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "n_draws": int(arr.size),
        "frac_draws_at_or_below_0.5": float((arr <= 0.5).mean()),
    }


def _boot_summary(vals: list[float]) -> dict:
    """Percentile-bootstrap summary, identical in shape to bootstrap_stratified_auc."""
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "boot_mean": float(arr.mean()),
        "boot_sd": float(arr.std(ddof=1)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "n_draws": int(arr.size),
        "frac_draws_at_or_below_0.5": float((arr <= 0.5).mean()),
    }


def bootstrap_stratified_auc_paired(
    fail: np.ndarray,
    spec_a: tuple[np.ndarray, np.ndarray, int],
    spec_b: tuple[np.ndarray, np.ndarray, int],
    n_boot: int,
    seed: int,
) -> dict:
    """Both stratified AUCs and their DIFFERENCE on the SAME resamples.

    Each spec is (scored_distance, stratifying_variable, n_bins). The two sides
    are computed on the same 942 contexts, so an unpaired comparison of two
    independently bootstrapped CIs would ignore their covariance. Resampling the
    difference directly gives the comparison its own CI.

    Draws are taken exactly as bootstrap_stratified_auc does -- same rng, same
    per-iteration `integers(0, n, size=n)`, same degenerate-resample skip -- so
    each marginal here reproduces the standalone call bit for bit at the same
    seed. main() asserts that, which is what certifies "one instrument".
    """
    rng = np.random.default_rng(seed)
    n = fail.size
    va: list[float] = []
    vb: list[float] = []
    vd: list[float] = []
    n_skipped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = fail[idx]
        if yb.min() == yb.max():
            n_skipped += 1
            continue
        a, _ = stratified_pooled_auc(yb, spec_a[0][idx], spec_a[1][idx], spec_a[2])
        b, _ = stratified_pooled_auc(yb, spec_b[0][idx], spec_b[1][idx], spec_b[2])
        if np.isfinite(a):
            va.append(a)
        if np.isfinite(b):
            vb.append(b)
        if np.isfinite(a) and np.isfinite(b):
            vd.append(a - b)

    diff = np.asarray(vd, dtype=np.float64)
    return {
        "side_a": _boot_summary(va),
        "side_b": _boot_summary(vb),
        "difference_a_minus_b": {
            "boot_mean": float(diff.mean()),
            "boot_sd": float(diff.std(ddof=1)),
            "ci95_low": float(np.percentile(diff, 2.5)),
            "ci95_high": float(np.percentile(diff, 97.5)),
            "n_draws": int(diff.size),
            "frac_draws_at_or_below_0": float((diff <= 0.0).mean()),
        },
        "_n_skipped_degenerate": n_skipped,
    }


def margin_retention(strat_auc: float, uncond_auc: float) -> float:
    """Fraction of the unconditional above-chance margin that survives stratifying.

    (AUC_stratified - 0.5) / (AUC_unconditional - 0.5). 1.0 means stratifying on
    the other side cost nothing; 0.0 means the side's whole signal was the other
    side's. Defined for both sides, so it is the comparable quantity across the
    two readings even though their unconditional AUCs differ.
    """
    denom = float(uncond_auc) - 0.5
    if abs(denom) < 1e-12:
        return float("nan")
    return float((float(strat_auc) - 0.5) / denom)


def residual_leakage(resid: np.ndarray, dC: np.ndarray) -> dict:
    """Does linear residualization actually remove the context signal?

    Pearson is 0 by construction. A non-zero Spearman means MONOTONE context
    information survives in the residual, so the 'partial' AUC is not purely
    answer-side.
    """
    sp = st.spearmanr(resid, dC)
    pe = st.pearsonr(resid, dC)
    return {
        "spearman_rho_with_dC": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r_with_dC": float(pe.statistic),
    }


# --------------------------------------------------------------------------
# 5. Top-decile crowding
# --------------------------------------------------------------------------
def top_decile(distance: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Boolean mask of the `frac` most CROWDED contexts (smallest distance)."""
    n = distance.size
    k = int(round(frac * n))
    order = np.argsort(distance, kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def fmt_ci(d: dict) -> str:
    return f"[{d['ci95_low']:.4f}, {d['ci95_high']:.4f}]"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1901)
    ap.add_argument("--hash-inputs", action="store_true", default=True)
    ap.add_argument("--no-hash-inputs", dest="hash_inputs", action="store_false")
    args = ap.parse_args()

    print(f"[load] {args.input_dir}")
    d = load_inputs(args.input_dir)
    print(f"[load] dR reconstructed in {d['dr_reconstruct_seconds']:.1f}s")

    checks = run_sanity(d)
    print(f"[sanity] {len(checks)} checks PASS (n=942, 90 failures, medians match)")

    fail = d["fail"]
    correct = d["correct"]
    dC, dA, dR = d["dC"], d["dA"], d["dR"]

    # ---------------- 1. Coupling ----------------
    cpl = {
        "all": {"dC_vs_dA_whitened": coupling(dC, dA), "dC_vs_dR_raw": coupling(dC, dR)},
        "successes_only": {
            "dC_vs_dA_whitened": coupling(dC[correct], dA[correct]),
            "dC_vs_dR_raw": coupling(dC[correct], dR[correct]),
        },
        "failures_only": {
            "dC_vs_dA_whitened": coupling(dC[~correct], dA[~correct]),
            "dC_vs_dR_raw": coupling(dC[~correct], dR[~correct]),
        },
        "dA_vs_dR": coupling(dA, dR),
    }

    print("\n=== 1. COUPLING (nearest-neighbour distances) ===")
    for scope in ("all", "successes_only", "failures_only"):
        for pair, v in cpl[scope].items():
            print(
                f"  {scope:<16s} {pair:<22s} n={v['n']:4d}  "
                f"Spearman={v['spearman_rho']:+.4f} (p={v['spearman_p']:.3e})  "
                f"Pearson={v['pearson_r']:+.4f} (p={v['pearson_p']:.3e})"
            )
    v = cpl["dA_vs_dR"]
    print(
        f"  {'all':<16s} {'dA_white_vs_dR_raw':<22s} n={v['n']:4d}  "
        f"Spearman={v['spearman_rho']:+.4f}  Pearson={v['pearson_r']:+.4f}"
    )

    # ---------------- 2. Logistic regression ----------------
    zC, zA, zR = zscore(dC), zscore(dA), zscore(dR)
    models = {
        "context_only": logit_fit(fail, {"z_dC": zC}),
        "answer_whitened_only": logit_fit(fail, {"z_dA": zA}),
        "answer_raw_only": logit_fit(fail, {"z_dR": zR}),
        "joint_whitened": logit_fit(fail, {"z_dC": zC, "z_dA": zA}),
        "joint_raw": logit_fit(fail, {"z_dC": zC, "z_dR": zR}),
    }
    # Multicollinearity read for the two joint models.
    r_CA = float(np.corrcoef(zC, zA)[0, 1])
    r_CR = float(np.corrcoef(zC, zR)[0, 1])
    collinearity = {
        "joint_whitened": {"pearson_r_predictors": r_CA, "vif": 1.0 / (1.0 - r_CA**2)},
        "joint_raw": {"pearson_r_predictors": r_CR, "vif": 1.0 / (1.0 - r_CR**2)},
    }

    print("\n=== 2. LOGISTIC REGRESSION  (fail ~ standardized distances) ===")
    print("    predictors are z-scored NN DISTANCES; a NEGATIVE coef means")
    print("    'smaller distance (more crowded) -> more failure'.")
    for mname, m in models.items():
        print(
            f"\n  [{mname}]  n={m['n_obs']}  llf={m['llf']:.3f}  "
            f"pseudoR2={m['pseudo_r2_mcfadden']:.4f}  converged={m['converged']}"
        )
        for lab, t in m["terms"].items():
            print(
                f"    {lab:<7s} coef={t['coef']:+8.4f}  se={t['std_err']:.4f}  "
                f"z={t['wald_z']:+7.3f}  p={t['p_value']:.3e}  "
                f"CI95=[{t['ci95_low']:+.4f}, {t['ci95_high']:+.4f}]"
            )
    print(
        f"\n  predictor collinearity: joint_whitened r={r_CA:+.4f} "
        f"VIF={collinearity['joint_whitened']['vif']:.2f} | "
        f"joint_raw r={r_CR:+.4f} VIF={collinearity['joint_raw']['vif']:.2f}"
    )

    # ---------------- 4. Likelihood-ratio tests ----------------
    lrts = {
        "add_answer_whitened_to_context": lr_test(
            models["joint_whitened"], models["context_only"], "z_dA"
        ),
        "add_context_to_answer_whitened": lr_test(
            models["joint_whitened"], models["answer_whitened_only"], "z_dC"
        ),
        "add_answer_raw_to_context": lr_test(models["joint_raw"], models["context_only"], "z_dR"),
        "add_context_to_answer_raw": lr_test(
            models["joint_raw"], models["answer_raw_only"], "z_dC"
        ),
    }
    print("\n=== 4. LIKELIHOOD-RATIO TESTS ===")
    for k, t in lrts.items():
        print(f"  {k:<34s} LR={t['lr_statistic']:8.3f}  df={t['df']}  p={t['p_value']:.3e}")

    # ---------------- 3. AUCs ----------------
    dists = {
        "auc_context": dC,
        "auc_answer_whitened": dA,
        "auc_answer_raw": dR,
    }
    partial_specs = [
        ("auc_context_partial_on_answer_whitened", "dC", "dA"),
        ("auc_answer_whitened_partial_on_context", "dA", "dC"),
        ("auc_context_partial_on_answer_raw", "dC", "dR"),
        ("auc_answer_raw_partial_on_context", "dR", "dC"),
    ]
    named = {"dC": dC, "dA": dA, "dR": dR}

    point = {k: auc_closeness(fail, v) for k, v in dists.items()}
    for name, tgt, other in partial_specs:
        point[name] = auc_closeness(fail, residualize(named[tgt], named[other]))

    print(f"\n[bootstrap] {args.n_boot} resamples, seed={args.seed} ...")
    t0 = time.time()
    boot = bootstrap_aucs(
        fail,
        dists,
        [(name, named[tgt], named[other]) for name, tgt, other in partial_specs],
        args.n_boot,
        args.seed,
    )
    print(
        f"[bootstrap] done in {time.time() - t0:.1f}s "
        f"({boot['_n_skipped_degenerate']} degenerate resamples skipped)"
    )

    aucs = {}
    for k in list(dists) + [s[0] for s in partial_specs]:
        aucs[k] = {"auc": point[k], **boot[k]}

    print("\n=== 3. AUCs (score = CLOSENESS = -distance; >0.5 = crowding predicts failure) ===")
    print("  -- unconditional --")
    for k in dists:
        print(f"    {k:<44s} {aucs[k]['auc']:.4f}  boot95 {fmt_ci(aucs[k])}")
    print("  -- partial (residualized on the other side, OLS) --")
    for name, _, _ in partial_specs:
        a = aucs[name]
        print(
            f"    {name:<44s} {a['auc']:.4f}  boot95 {fmt_ci(a)}"
            f"   P(boot<=0.5)={a['frac_draws_at_or_below_0.5']:.3f}"
        )

    # ---------------- 5. Top-decile crowding ----------------
    mask_C = top_decile(dC)
    mask_A = top_decile(dA)
    mask_R = top_decile(dR)
    n_dec = int(mask_C.sum())
    f = fail.astype(bool)
    decile = {
        "decile_size": n_dec,
        "n_failures_total": int(f.sum()),
        "context": {
            "failures_in_top_decile": int((f & mask_C).sum()),
            "frac_of_failures": float((f & mask_C).sum() / f.sum()),
            "precision_in_decile": float((f & mask_C).sum() / mask_C.sum()),
        },
        "answer_whitened": {
            "failures_in_top_decile": int((f & mask_A).sum()),
            "frac_of_failures": float((f & mask_A).sum() / f.sum()),
            "precision_in_decile": float((f & mask_A).sum() / mask_A.sum()),
        },
        "answer_raw": {
            "failures_in_top_decile": int((f & mask_R).sum()),
            "frac_of_failures": float((f & mask_R).sum() / f.sum()),
            "precision_in_decile": float((f & mask_R).sum() / mask_R.sum()),
        },
        "overlap_context_and_answer_whitened": {
            "contexts_in_both_deciles": int((mask_C & mask_A).sum()),
            "failures_in_both_deciles": int((f & mask_C & mask_A).sum()),
            "failures_in_either_decile": int((f & (mask_C | mask_A)).sum()),
            "failures_context_only": int((f & mask_C & ~mask_A).sum()),
            "failures_answer_only": int((f & mask_A & ~mask_C).sum()),
            "failures_in_neither": int((f & ~mask_C & ~mask_A).sum()),
        },
        "overlap_context_and_answer_raw": {
            "contexts_in_both_deciles": int((mask_C & mask_R).sum()),
            "failures_in_both_deciles": int((f & mask_C & mask_R).sum()),
            "failures_in_either_decile": int((f & (mask_C | mask_R)).sum()),
            "failures_context_only": int((f & mask_C & ~mask_R).sum()),
            "failures_answer_only": int((f & mask_R & ~mask_C).sum()),
            "failures_in_neither": int((f & ~mask_C & ~mask_R).sum()),
        },
    }

    print(f"\n=== 5. TOP-DECILE CROWDING (decile = {n_dec} most crowded of 942) ===")
    for side in ("context", "answer_whitened", "answer_raw"):
        s = decile[side]
        print(
            f"  {side:<16s} {s['failures_in_top_decile']:3d}/90 failures "
            f"({s['frac_of_failures'] * 100:.1f}% of failures; "
            f"{s['precision_in_decile'] * 100:.1f}% of the decile)"
        )
    o = decile["overlap_context_and_answer_whitened"]
    print(
        f"  overlap (ctx & answer-whitened): {o['contexts_in_both_deciles']} contexts, "
        f"{o['failures_in_both_deciles']} failures in both, "
        f"{o['failures_context_only']} ctx-only, {o['failures_answer_only']} ans-only, "
        f"{o['failures_in_neither']} in neither"
    )
    o = decile["overlap_context_and_answer_raw"]
    print(
        f"  overlap (ctx & answer-raw):      {o['contexts_in_both_deciles']} contexts, "
        f"{o['failures_in_both_deciles']} failures in both, "
        f"{o['failures_context_only']} ctx-only, {o['failures_answer_only']} ans-only, "
        f"{o['failures_in_neither']} in neither"
    )

    # ---------------- 6. Robustness ----------------
    robust: dict = {
        "rationale": (
            "Failure rate drops steeply and non-linearly across dC, so a single "
            "linear-in-z context term (and a linear residualization) cannot fully "
            "absorb context crowding. These adjustments control for context "
            "non-parametrically, via quantile-bin dummies and via a stratified "
            "Mann-Whitney AUC."
        ),
        "logit_flexible_context": {},
        "stratified_auc": {},
        "linear_residual_leakage": {
            "resid_dA_on_dC": residual_leakage(residualize(dA, dC), dC),
            "resid_dR_on_dC": residual_leakage(residualize(dR, dC), dC),
            "note": (
                "Pearson is 0 by construction; a non-zero Spearman means monotone "
                "context information survives linear residualization, so the "
                "partial AUC is not a clean conditional-independence test."
            ),
        },
    }

    print("\n=== 6. ROBUSTNESS: non-parametric control for context crowding ===")
    for n_bins in (4, 5):
        for label, zz in (("z_dA", zA), ("z_dR", zR)):
            key = f"{label}_given_{n_bins}bin_context"
            r = flexible_context_test(fail, dC, zz, label, n_bins)
            robust["logit_flexible_context"][key] = r
            print(
                f"  logit  {label} | {n_bins}-bin dC   coef={r['coef']:+.4f} "
                f"se={r['std_err']:.4f} z={r['wald_z']:+.3f} p={r['p_value']:.4f}  "
                f"LR={r['lr_test']['lr_statistic']:.3f} p={r['lr_test']['p_value']:.4f}"
            )

    for n_bins in (4, 5):
        for name, dd in (("answer_whitened", dA), ("answer_raw", dR)):
            key = f"{name}_within_{n_bins}bin_context"
            pooled, per = stratified_pooled_auc(fail, dd, dC, n_bins)
            bs = bootstrap_stratified_auc(fail, dd, dC, n_bins, args.n_boot, args.seed)
            spars = stratum_sparsity_summary(per)
            robust["stratified_auc"][key] = {
                "pooled_auc": pooled,
                "per_stratum": per,
                "stratum_sparsity": spars,
                "weighting": "discordant pairs (n_fail * n_success) per stratum",
                **bs,
            }
            print(
                f"  strat  {name} within {n_bins}-bin dC   AUC={pooled:.4f}  "
                f"boot95 [{bs['ci95_low']:.4f}, {bs['ci95_high']:.4f}]  "
                f"P(boot<=0.5)={bs['frac_draws_at_or_below_0.5']:.3f}  "
                f"sparse strata {spars['sparse_bins']} "
                f"({spars['pair_weight_share_flagged_sparse'] * 100:.1f}% of pair weight)"
            )

    lk = robust["linear_residual_leakage"]
    print(
        f"  leakage  resid(dA|dC) vs dC Spearman="
        f"{lk['resid_dA_on_dC']['spearman_rho_with_dC']:+.4f} "
        f"(p={lk['resid_dA_on_dC']['spearman_p']:.3f}) | "
        f"resid(dR|dC) vs dC Spearman="
        f"{lk['resid_dR_on_dC']['spearman_rho_with_dC']:+.4f} "
        f"(p={lk['resid_dR_on_dC']['spearman_p']:.3f})"
    )

    # ---------------- 7. Mirror: the context side under the SAME instrument ----
    #
    # Section 6 asks: among contexts whose CONTEXT neighbourhoods are equally
    # crowded, does ANSWER crowding still predict failure? This section asks the
    # mirror: among contexts whose ANSWER neighbourhoods are equally crowded,
    # does CONTEXT crowding still predict failure? Same pair-weighted stratified
    # Mann-Whitney construction, same 4- and 5-bin quantile schemes, same 2000
    # resamples at the same seed, same percentile CI -- only the roles of the two
    # sides are swapped (raw arm strata = dR, whitened arm strata = dA).
    #
    # The manuscript leans on BOTH sentences, so both have to come off one
    # instrument. The two sides are measured on the same 942 contexts, so their
    # difference is bootstrapped on the SAME resamples rather than compared as
    # two independent CIs.
    robust["mirror_stratified_auc"] = {}
    robust["mirror_rationale"] = (
        "Mirror of the section-6 stratified AUC: context closeness scored WITHIN "
        "quantile strata of ANSWER crowding (dR strata for the raw arm, dA strata "
        "for the whitened arm), under the identical pair-weighted stratified "
        "Mann-Whitney construction, bin schemes, resample count, seed and CI "
        "method. The difference between the two sides is bootstrapped on the same "
        "resamples, so the comparison carries its own CI."
    )

    mirror_cells: dict = {}
    consistency_devs: list[dict] = []

    print("\n=== 7. MIRROR: context closeness WITHIN answer-crowding strata ===")
    print("    section 6 holds CONTEXT fixed and scores the answer side;")
    print("    section 7 holds ANSWER fixed and scores the context side.")
    for n_bins in (4, 5):
        for arm, ans_dist in (("whitened", dA), ("raw", dR)):
            ctx_key = f"context_within_{n_bins}bin_answer_{arm}"
            ans_key = f"answer_{arm}_within_{n_bins}bin_context"

            pooled_ctx, per_ctx = stratified_pooled_auc(fail, dC, ans_dist, n_bins)
            spars_ctx = stratum_sparsity_summary(per_ctx)
            paired = bootstrap_stratified_auc_paired(
                fail,
                (dC, ans_dist, n_bins),
                (ans_dist, dC, n_bins),
                args.n_boot,
                args.seed,
            )

            # The paired pass recomputes the section-6 side on the same draws.
            # It MUST reproduce the standalone section-6 numbers exactly; that
            # equality is what certifies the two sides share one instrument.
            ans_cell = robust["stratified_auc"][ans_key]
            dev = {
                "cell": ans_key,
                "max_abs_deviation": max(
                    abs(paired["side_b"][f] - ans_cell[f])
                    for f in ("boot_mean", "boot_sd", "ci95_low", "ci95_high")
                ),
                "n_draws_match": paired["side_b"]["n_draws"] == ans_cell["n_draws"],
            }
            consistency_devs.append(dev)
            if dev["max_abs_deviation"] > 1e-9 or not dev["n_draws_match"]:
                raise SystemExit(
                    f"PAIRED-BOOTSTRAP CONSISTENCY FAILED for {ans_key}: the paired "
                    f"pass did not reproduce the standalone section-6 bootstrap "
                    f"(max abs deviation {dev['max_abs_deviation']!r}, n_draws match "
                    f"{dev['n_draws_match']}). The two sides are then NOT on one "
                    f"instrument -- refusing to report a comparison."
                )

            robust["mirror_stratified_auc"][ctx_key] = {
                "pooled_auc": pooled_ctx,
                "per_stratum": per_ctx,
                "stratum_sparsity": spars_ctx,
                "weighting": "discordant pairs (n_fail * n_success) per stratum",
                "scored_side": "dC (context-space NN closeness)",
                "stratifying_side": f"{'dA' if arm == 'whitened' else 'dR'} "
                f"(answer-space NN distance, {arm})",
                **paired["side_a"],
            }

            uncond_ctx = aucs["auc_context"]["auc"]
            uncond_ans = aucs[f"auc_answer_{arm}"]["auc"]
            ret_ctx = margin_retention(pooled_ctx, uncond_ctx)
            ret_ans = margin_retention(ans_cell["pooled_auc"], uncond_ans)
            diff = paired["difference_a_minus_b"]

            c1 = bool(paired["side_a"]["ci95_low"] > 0.5)
            c2 = bool(diff["ci95_low"] > 0.0)
            c3 = bool(ret_ctx >= 0.5)
            holds = bool(c1 and c2 and c3)

            mirror_cells[f"{arm}_{n_bins}bin"] = {
                "arm": arm,
                "n_bins": n_bins,
                "context_side": {
                    "description": (
                        f"context closeness (dC) scored within {n_bins} quantile "
                        f"strata of answer crowding "
                        f"({'dA' if arm == 'whitened' else 'dR'})"
                    ),
                    "unconditional_auc": uncond_ctx,
                    "stratified_auc": pooled_ctx,
                    "ci95_low": paired["side_a"]["ci95_low"],
                    "ci95_high": paired["side_a"]["ci95_high"],
                    "margin_retention": ret_ctx,
                    "n_strata_flagged_sparse": spars_ctx["n_strata_flagged_sparse"],
                    "pair_weight_share_flagged_sparse": spars_ctx[
                        "pair_weight_share_flagged_sparse"
                    ],
                },
                "answer_side": {
                    "description": (
                        f"answer closeness ({'dA' if arm == 'whitened' else 'dR'}) "
                        f"scored within {n_bins} quantile strata of context "
                        f"crowding (dC) -- the section-6 reading"
                    ),
                    "unconditional_auc": uncond_ans,
                    "stratified_auc": ans_cell["pooled_auc"],
                    "ci95_low": ans_cell["ci95_low"],
                    "ci95_high": ans_cell["ci95_high"],
                    "margin_retention": ret_ans,
                    "n_strata_flagged_sparse": ans_cell["stratum_sparsity"][
                        "n_strata_flagged_sparse"
                    ],
                    "pair_weight_share_flagged_sparse": ans_cell["stratum_sparsity"][
                        "pair_weight_share_flagged_sparse"
                    ],
                },
                "paired_difference_context_minus_answer": diff,
                "clauses": {
                    "c1_context_side_ci_excludes_chance": c1,
                    "c2_context_side_above_answer_side": c2,
                    "c3_context_side_retains_half_its_margin": c3,
                },
                "mirror_holds": holds,
            }

            print(
                f"\n  [{arm} arm, {n_bins} bins]"
                f"\n    context within answer strata  AUC={pooled_ctx:.4f}  "
                f"boot95 [{paired['side_a']['ci95_low']:.4f}, "
                f"{paired['side_a']['ci95_high']:.4f}]  "
                f"uncond={uncond_ctx:.4f}  retention={ret_ctx:.3f}"
                f"\n    answer within context strata  AUC={ans_cell['pooled_auc']:.4f}  "
                f"boot95 [{ans_cell['ci95_low']:.4f}, {ans_cell['ci95_high']:.4f}]  "
                f"uncond={uncond_ans:.4f}  retention={ret_ans:.3f}"
                f"\n    paired difference (context - answer) = {diff['boot_mean']:+.4f}  "
                f"boot95 [{diff['ci95_low']:+.4f}, {diff['ci95_high']:+.4f}]  "
                f"P(boot<=0)={diff['frac_draws_at_or_below_0']:.3f}"
                f"\n    clauses: above-chance={c1} above-answer-side={c2} "
                f"retains-half-margin={c3}  =>  MIRROR {'HOLDS' if holds else 'FAILS'}"
            )
            for e in per_ctx:
                flag = f"  <-- SPARSE: {e['sparse_reason']}" if e["sparse"] else ""
                auc_s = "   n/a" if e["auc"] is None else f"{e['auc']:.4f}"
                print(
                    f"      stratum {e['bin']}: n={e['n']:4d}  fail={e['n_fail']:3d}  "
                    f"success={e['n_success']:3d}  pairs={int(e['pair_weight']):6d} "
                    f"({e['pair_weight_share'] * 100:5.1f}% of weight)  AUC={auc_s}{flag}"
                )

    headline = mirror_cells["raw_4bin"]
    all_hold = all(c["mirror_holds"] for c in mirror_cells.values())
    asymmetry_mirror = {
        "question": (
            "The manuscript states that among contexts with equally crowded "
            "CONTEXT neighbourhoods the answer side drops from its unconditional "
            "AUC to a much lower stratified AUC. The mirror claim is that among "
            "contexts with equally crowded ANSWER neighbourhoods the context side "
            "stays high. This section tests the mirror on the same instrument."
        ),
        "instrument": (
            "pair-weighted stratified Mann-Whitney AUC of CLOSENESS (= -distance) "
            "against top-1 retrieval failure, within quantile strata of the other "
            "side; 4- and 5-bin schemes; 2000-resample percentile bootstrap at "
            "seed 1901 with strata edges recomputed inside each resample; the "
            "two sides and their difference share the same resamples"
        ),
        "verdict_rule": {
            "c1_context_side_ci_excludes_chance": (
                "context-side stratified AUC 95% CI lower bound > 0.5"
            ),
            "c2_context_side_above_answer_side": (
                "paired bootstrap CI for (context side - answer side) lies entirely above 0"
            ),
            "c3_context_side_retains_half_its_margin": (
                "context-side margin retention >= 0.5, i.e. at least half of the "
                "unconditional above-chance margin survives stratifying"
            ),
            "mirror_holds": "all three clauses true",
        },
        "headline_cell": "raw_4bin",
        "headline_cell_note": (
            "raw arm at 4 bins is the cell behind the manuscript's 0.79 -> 0.61 "
            "sentence, so it is the cell the mirror sentence must match"
        ),
        "mirror_holds_headline_cell": headline["mirror_holds"],
        "mirror_holds_all_cells": all_hold,
        "cells": mirror_cells,
        "paired_bootstrap_consistency": {
            "check": (
                "the paired pass must reproduce the standalone section-6 "
                "stratified bootstrap exactly (tolerance 1e-9); a mismatch aborts"
            ),
            "max_abs_deviation_over_cells": max(d["max_abs_deviation"] for d in consistency_devs),
            "per_cell": consistency_devs,
        },
    }
    asymmetry_mirror["headline"] = (
        "MIRROR {} at the headline cell (raw arm, 4 context/answer bins): holding "
        "ANSWER crowding fixed, context closeness scores AUC {:.4f} "
        "[{:.4f}, {:.4f}] (unconditional {:.4f}, retention {:.3f}); holding "
        "CONTEXT crowding fixed, answer closeness scores AUC {:.4f} "
        "[{:.4f}, {:.4f}] (unconditional {:.4f}, retention {:.3f}). Paired "
        "difference (context - answer) = {:+.4f} [{:+.4f}, {:+.4f}]."
    ).format(
        "HOLDS" if headline["mirror_holds"] else "FAILS",
        headline["context_side"]["stratified_auc"],
        headline["context_side"]["ci95_low"],
        headline["context_side"]["ci95_high"],
        headline["context_side"]["unconditional_auc"],
        headline["context_side"]["margin_retention"],
        headline["answer_side"]["stratified_auc"],
        headline["answer_side"]["ci95_low"],
        headline["answer_side"]["ci95_high"],
        headline["answer_side"]["unconditional_auc"],
        headline["answer_side"]["margin_retention"],
        headline["paired_difference_context_minus_answer"]["boot_mean"],
        headline["paired_difference_context_minus_answer"]["ci95_low"],
        headline["paired_difference_context_minus_answer"]["ci95_high"],
    )

    print("\n" + "=" * 78)
    print(asymmetry_mirror["headline"])
    print(
        f"  mirror holds in all {len(mirror_cells)} arm x bin-count cells: {all_hold}"
        + (
            ""
            if all_hold
            else "  <-- cells where it FAILS: "
            + ", ".join(k for k, c in mirror_cells.items() if not c["mirror_holds"])
        )
    )
    print("=" * 78)

    # ---------------- provenance + write ----------------
    def ver(pkg: str) -> str:
        try:
            return metadata.version(pkg)
        except Exception:
            return "unavailable"

    prov = {
        "script": "scripts/issue1901_crowding_mediation.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "library_versions": {
            p: ver(p) for p in ("numpy", "scipy", "statsmodels", "scikit-learn", "pandas")
        },
        "seed": args.seed,
        "n_bootstrap": args.n_boot,
        "bootstrap_method": (
            "nonparametric percentile bootstrap over the 942 held-out contexts; "
            "the OLS residualization is refit inside every resample"
        ),
        "input_paths": d["paths"],
        "array_shapes": d["shapes"],
        "dr_reconstruct_seconds": d["dr_reconstruct_seconds"],
        "blas_thread_env": {
            v: os.environ.get(v)
            for v in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "dr_reconstruction_determinism": (
            "dR is NOT read from disk: it is reconstructed here by a float32 matmul "
            "(942 query rows x 10000 pool rows x 3584 dims). float32 GEMM reduction "
            "ORDER depends on the BLAS thread count, so dR differs in its last bits "
            "across thread counts and every RAW-ARM (dR) statistic moves by roughly "
            "1e-5 to 2e-4 -- enough to shift a 4-decimal AUC by one in the last "
            "digit, never enough to move a 2-decimal number. Measured on this VM: "
            "spearman(dC, dR) = 0.684421820730 at 1 thread, 0.684421815818 at 8, "
            "0.684420186428 at 16. Set OMP_NUM_THREADS=1 to reproduce these "
            "numbers exactly. WHITENED-arm (dA) and context-only (dC) statistics "
            "are read from disk and are bit-identical across thread counts."
        ),
        "conventions": {
            "dC": "context-space NN distance = 1 - nn1 (higher = more isolated)",
            "dA": "answer-space NN distance, WHITENED (retrieval convention)",
            "dR": "answer-space NN distance, RAW (reconstructed: mean-centre pool, "
            "unit-norm rows, cosine vs pool, mask self, 1 - max)",
            "fail": "~correct, i.e. top-1 retrieval failure",
            "auc_score": "CLOSENESS = -distance, so AUC>0.5 means crowding predicts failure",
            "logit_predictors": "z-scored DISTANCES, so a NEGATIVE coefficient means "
            "crowding predicts failure",
        },
    }
    if args.hash_inputs:
        prov["input_sha256"] = {k: sha256_of(Path(v)) for k, v in d["paths"].items()}

    summary = {
        "claim_under_test": (
            "answer-side crowding is downstream of context-side crowding: once "
            "context-space NN distance is known, answer-space NN distance carries "
            "no additional information about top-1 retrieval failure"
        ),
        "setting": {
            "operating_point_candidates": 10000,
            "n_held_out_contexts": int(fail.size),
            "n_failures": int(fail.sum()),
            "n_successes": int((1 - fail).sum()),
        },
        "provenance": prov,
        "sanity_checks": checks,
        "coupling": cpl,
        "logistic_regression": models,
        "predictor_collinearity": collinearity,
        "likelihood_ratio_tests": lrts,
        "aucs": aucs,
        "top_decile": decile,
        "robustness": robust,
        "asymmetry_mirror": asymmetry_mirror,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")
    print(f"\n[write] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
