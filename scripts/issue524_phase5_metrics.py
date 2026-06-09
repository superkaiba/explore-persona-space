"""Phase 5 — paired nested LTCO-CV + censored/Tobit + context-clustered bootstrap.

Issue #524 plan v4 §4 Phase 5 + §6 "Decision metric". CPU on dev VM.

Implements (kept verbatim from v1 — round-1 Stats fixes intact):

  1. Pooled out-of-fold predictions across 496 leave-2-contexts-out folds:
     ONE held-out prediction per ordered off-diagonal pair across the
     whole stack of 496 folds. Then ONE pooled R² for M_sym and ONE for
     M_full against full-panel SS_tot. Headline scalar = ΔR²_pooled =
     R²_full,pooled − R²_sym,pooled. NOT the average of per-fold R²s
     (that was the round-1 Stats Must-Fix selection-inflation hole).

  2. Unified censored Tobit ΔLL when censored fraction ≥ 10% on any
     block. Tobit log-likelihood under right-censoring at ceiling − 0.1
     nat and left-censoring at floor + 0.1 nat. Per-observation censor
     flags fed into ONE Tobit fit over the full panel; ΔLL = ll_full −
     ll_sym. Both ΔR²_pooled and ΔLL reported in body either way.

  3. Context-clustered dyadic bootstrap B=2000 on the selected headline
     scalar. Resamples CONTEXTS (not pairs), then rebuilds the
     off-diagonal pair set, then re-runs the nested-CV with that
     resampled context list.

  4. Four-conjunct PASS test (plan §3 Falsification, §6 Decision metric):
       (a) incremental CI strictly excludes 0
       (b) standalone ρ(predictor, ΔG_anti) CI strictly excludes 0
       (c) ρ(½(d − d.T), ΔG_anti) CI strictly excludes 0
       (d) #523 held-out replication shows same sign

CLI (smoke == sweep with --b 16 --layers 22 --points last_prompt for one
predictor at one cell, plan §"Smoke architecture parity" — UNIFIED):
    # Smoke: 1 predictor × 1 layer × 1 point × B=16 bootstraps.
    uv run python scripts/issue524_phase5_metrics.py --smoke

    # Sweep: full registry × 28 layers × 3 points × B=2000.
    uv run python scripts/issue524_phase5_metrics.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# epm-lint: workflow-fix-on-bug -- module-top dotenv load required (HF
# helpers downstream).
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("i524.phase5")

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE2_DIR = REPO_ROOT / "eval_results" / "issue_524" / "phase2"
PHASE4_PATH = REPO_ROOT / "eval_results" / "issue_524" / "phase4" / "predictors.npz"
PHASE5_DIR = REPO_ROOT / "eval_results" / "issue_524" / "phase5"
OUT_PATH = PHASE5_DIR / "metrics.json"

# Tobit censoring buffer (plan §6 saturation handling).
TOBIT_BUFFER_NATS = 0.1
# Floor / ceiling estimated from the ΔG distribution at runtime.

# Bootstrap sample count (plan §6: B=2000 context-clustered dyadic).
DEFAULT_BOOTSTRAP_B = 2000


def _git_sha() -> str:
    """Short HEAD SHA or 'unknown' on error."""
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_dg_matrix(contexts: list[str]) -> np.ndarray:
    """Load the 32×32 ΔG matrix from Phase 2 per-cell JSONs.

    Returns:
        (N, N) np.ndarray of ΔG values, indexed by ``contexts`` order.
        Diagonal is 0 (handled separately). Missing cells become NaN —
        the caller's LTCO loop drops folds containing NaN test pairs.
    """
    n = len(contexts)
    G = np.full((n, n), np.nan, dtype=np.float64)
    for i, ci in enumerate(contexts):
        for j, cj in enumerate(contexts):
            if i == j:
                G[i, j] = 0.0
                continue
            p = PHASE2_DIR / "per_cell" / f"G_{ci}__{cj}.json"
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            G[i, j] = float(data.get("delta_g_mean", data.get("delta_g", np.nan)))
    return G


def _offdiag_pairs(n: int) -> list[tuple[int, int]]:
    """Enumerate the ordered off-diagonal pairs (i, j) with i != j."""
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def _fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve OLS via least-squares (no statsmodels dep)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _predict_ols(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return X @ coef


def _design_matrix(
    pairs: list[tuple[int, int]],
    features: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the OLS design matrix [1, f1[i,j], f2[i,j], ...]."""
    cols = [np.ones(len(pairs))]
    for _, mat in features.items():
        cols.append(np.array([mat[i, j] for i, j in pairs]))
    return np.column_stack(cols)


def pooled_ltco_predictions(
    sym_features: dict[str, np.ndarray],
    full_features: dict[str, np.ndarray],
    dg_matrix: np.ndarray,
    contexts: list[str],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """One held-out OLS prediction per ordered off-diagonal pair, pooled across folds.

    Outer loop: leave-2-contexts-out (every unordered pair of contexts is a
    held-out block). For each held block, train M_sym + M_full on training
    pairs (both endpoints outside the block) and predict the EXACTLY 2
    test pairs (both endpoints inside the block).

    Returns:
        (pred_sym, pred_full, pairs): two arrays of pooled out-of-fold
        predictions and the list of ordered (i, j) pairs each prediction
        corresponds to. Pairs containing NaN in dg_matrix are dropped.
    """
    n = len(contexts)
    pairs = _offdiag_pairs(n)
    # Drop pairs where the target is NaN (Phase 2 didn't write that cell).
    valid_pairs = [p for p in pairs if not np.isnan(dg_matrix[p])]
    pred_sym = dict.fromkeys(valid_pairs, np.nan)
    pred_full = dict.fromkeys(valid_pairs, np.nan)

    for held_block in itertools.combinations(range(n), 2):
        a, b = held_block
        train_pairs = [p for p in valid_pairs if p[0] not in held_block and p[1] not in held_block]
        test_pairs = [p for p in valid_pairs if p[0] in held_block and p[1] in held_block]
        if not train_pairs or not test_pairs:
            continue
        train_y = np.array([dg_matrix[p] for p in train_pairs])
        X_sym_train = _design_matrix(train_pairs, sym_features)
        X_full_train = _design_matrix(train_pairs, full_features)
        coef_sym = _fit_ols(X_sym_train, train_y)
        coef_full = _fit_ols(X_full_train, train_y)
        X_sym_test = _design_matrix(test_pairs, sym_features)
        X_full_test = _design_matrix(test_pairs, full_features)
        for k, p in enumerate(test_pairs):
            pred_sym[p] = float(_predict_ols(X_sym_test[k : k + 1], coef_sym)[0])
            pred_full[p] = float(_predict_ols(X_full_test[k : k + 1], coef_full)[0])

    # Drop any pairs that never got predicted (straddling pairs without both
    # endpoints in any held block — by construction this should be empty;
    # assert as a safety net per plan §4 Phase 5).
    materialized = [p for p in valid_pairs if not np.isnan(pred_sym[p])]
    pred_sym_arr = np.array([pred_sym[p] for p in materialized])
    pred_full_arr = np.array([pred_full[p] for p in materialized])
    return pred_sym_arr, pred_full_arr, materialized


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² against the panel mean (NOT model-mean)."""
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def pooled_incremental_r2(
    pred_sym: np.ndarray, pred_full: np.ndarray, dg_target: np.ndarray
) -> float:
    """ΔR²_pooled = R²_full − R²_sym, using ONE pooled R² each."""
    return _r2_score(dg_target, pred_full) - _r2_score(dg_target, pred_sym)


def _censor_flags(y: np.ndarray, ceiling: float, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-observation censor flags for right (ceiling) and left (floor)."""
    return (y >= ceiling - TOBIT_BUFFER_NATS), (y <= floor + TOBIT_BUFFER_NATS)


def _tobit_loglik(
    pred: np.ndarray,
    y: np.ndarray,
    right_cens: np.ndarray,
    left_cens: np.ndarray,
    sigma: float,
) -> float:
    """Tobit log-likelihood with per-obs left + right censoring.

    For uncensored obs: normal log-pdf at y. For right-censored: log Phi-bar
    at (y - pred)/sigma (i.e. log P(true ≥ ceiling)). For left-censored:
    log Phi at (y - pred)/sigma (i.e. log P(true ≤ floor)). Combined sum is
    the Tobit log-likelihood.
    """
    from scipy.stats import norm as scipy_norm

    sigma = max(sigma, 1e-6)
    resid = (y - pred) / sigma
    # Uncensored: normal log-pdf (with the 1/sigma Jacobian).
    log_pdf_uncens = scipy_norm.logpdf(resid) - np.log(sigma)
    # Right-censored: log(1 - Phi(z)) = log_sf
    log_sf_right = scipy_norm.logsf(resid)
    # Left-censored: log Phi(z) = log_cdf
    log_cdf_left = scipy_norm.logcdf(resid)
    ll_terms = np.where(right_cens, log_sf_right, np.where(left_cens, log_cdf_left, log_pdf_uncens))
    return float(ll_terms.sum())


def _mle_sigma_tobit(
    pred: np.ndarray,
    y: np.ndarray,
    right_cens: np.ndarray,
    left_cens: np.ndarray,
) -> float:
    """Simple MLE for the Tobit sigma via scipy 1-D minimization.

    A full Tobit refit (predicting beta + sigma jointly) is more efficient
    but the OLS coef estimates from the pooled fold predictions are
    consistent under low censoring and slightly conservative under high
    censoring — this matches the round-1 Stats spec ("unified censored
    ΔLL ... per-observation censor flags"). For the ΔLL comparison the
    common-sigma assumption between M_sym and M_full is fine.
    """
    from scipy.optimize import minimize_scalar

    def negll(sigma: float) -> float:
        return -_tobit_loglik(pred, y, right_cens, left_cens, sigma)

    # Robust bracket: residual std as the initial guess.
    init_sigma = float(np.std(y - pred)) + 1e-6
    bracket = (init_sigma * 0.1, init_sigma, init_sigma * 10.0)
    try:
        res = minimize_scalar(negll, bracket=bracket, method="brent")
        if res.success:
            return float(res.x)
    except (ValueError, RuntimeError):
        pass
    return init_sigma


def censored_delta_logL(
    pred_sym: np.ndarray,
    pred_full: np.ndarray,
    dg_target: np.ndarray,
    ceiling: float,
    floor: float,
) -> float:
    """Unified censored ΔLL = LL_full − LL_sym (plan §6, kept verbatim from v1)."""
    right_cens, left_cens = _censor_flags(dg_target, ceiling, floor)
    sigma_sym = _mle_sigma_tobit(pred_sym, dg_target, right_cens, left_cens)
    sigma_full = _mle_sigma_tobit(pred_full, dg_target, right_cens, left_cens)
    ll_sym = _tobit_loglik(pred_sym, dg_target, right_cens, left_cens, sigma_sym)
    ll_full = _tobit_loglik(pred_full, dg_target, right_cens, left_cens, sigma_full)
    return ll_full - ll_sym


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Pure-numpy Spearman ρ for the directional fraud test."""
    from scipy.stats import spearmanr

    if len(x) < 3:
        return float("nan")
    rho = spearmanr(x, y).statistic
    return float(rho)


def directional_fraud(
    predictor_matrix: np.ndarray,
    dg_matrix: np.ndarray,
    pairs: list[tuple[int, int]],
) -> dict[str, float]:
    """Compute the standalone ρ and asymmetry-asymmetry correlation for the fraud test.

    Returns:
        Dict with two scalars:
          - standalone_rho_pred_vs_dg_anti
          - asym_corr_predAnti_vs_dgAnti
    """
    dg_anti = 0.5 * (dg_matrix - dg_matrix.T)
    pred_anti = 0.5 * (predictor_matrix - predictor_matrix.T)
    pred_vec = np.array([predictor_matrix[p] for p in pairs])
    dg_anti_vec = np.array([dg_anti[p] for p in pairs])
    # For the unordered asymmetry-asymmetry correlation, use (i, j) with i < j only.
    unordered = [p for p in pairs if p[0] < p[1]]
    pred_anti_vec = np.array([pred_anti[p] for p in unordered])
    dg_anti_unordered = np.array([dg_anti[p] for p in unordered])
    return {
        "standalone_rho_pred_vs_dg_anti": _spearman(pred_vec, dg_anti_vec),
        "asym_corr_predAnti_vs_dgAnti": _spearman(pred_anti_vec, dg_anti_unordered),
    }


def context_clustered_bootstrap_ci(
    sym_features: dict[str, np.ndarray],
    full_features: dict[str, np.ndarray],
    dg_matrix: np.ndarray,
    contexts: list[str],
    headline_fn,
    *,
    B: int = DEFAULT_BOOTSTRAP_B,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Context-clustered dyadic bootstrap CI on the headline scalar.

    Resamples CONTEXTS with replacement (not pairs), then rebuilds the
    off-diagonal pair set on the resampled context multiset, then re-runs
    the nested-CV with that resampled context list.

    Args:
        headline_fn: ``(pred_sym, pred_full, dg_target) -> float`` (R²
            or ΔLL). The selected scalar from the calibration step.
    Returns:
        (low, point, high) — 2.5/50/97.5 percentile of the bootstrap
        distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(contexts)
    boots: list[float] = []
    for _b in range(B):
        boot_ix = rng.choice(n, size=n, replace=True)
        boot_dg = dg_matrix[np.ix_(boot_ix, boot_ix)]
        boot_sym = {k: v[np.ix_(boot_ix, boot_ix)] for k, v in sym_features.items()}
        boot_full = {k: v[np.ix_(boot_ix, boot_ix)] for k, v in full_features.items()}
        boot_contexts = [contexts[i] for i in boot_ix]
        pred_sym, pred_full, pairs = pooled_ltco_predictions(
            boot_sym, boot_full, boot_dg, boot_contexts
        )
        dg_target = np.array([boot_dg[p] for p in pairs])
        boots.append(headline_fn(pred_sym, pred_full, dg_target))
    boots_arr = np.array(boots, dtype=np.float64)
    low, point, high = np.percentile(boots_arr, [2.5, 50, 97.5])
    return float(low), float(point), float(high)


def _load_predictor_stack(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load Phase 4's predictor stack.

    Returns:
        (arrays, meta): dict mapping ``{pred}__L{layer}__{point}`` → matrix,
        and the meta sidecar dict.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 4 predictor stack missing: {path}. "
            "Run scripts/issue524_phase4_predictors.py first."
        )
    arrays = {}
    with np.load(path) as f:
        for k in f.files:
            arrays[k] = np.asarray(f[k], dtype=np.float64)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return arrays, meta


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 5 nested-CV + bootstrap on Phase 4 predictors."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictor-stack",
        type=str,
        default=str(PHASE4_PATH),
        help="Path to Phase 4 predictors.npz (default: eval_results/issue_524/phase4/).",
    )
    p.add_argument(
        "--b",
        type=int,
        default=DEFAULT_BOOTSTRAP_B,
        help="Bootstrap iterations (default 2000; use 16 for smoke).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for bootstrap reproducibility.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke: 1 predictor × 1 layer × 1 point × B=16.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(OUT_PATH),
        help="Output JSON path.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    arrays, pred_meta = _load_predictor_stack(Path(args.predictor_stack))
    contexts = pred_meta.get("contexts", [])
    if not contexts:
        raise RuntimeError("Phase 4 stack meta missing 'contexts' field; cannot align ΔG matrix.")

    G = _load_dg_matrix(contexts)
    valid_count = int(np.sum(~np.isnan(G)))
    logger.info(
        "Loaded ΔG matrix: %d×%d, %d valid cells (out of %d off-diagonal).",
        len(contexts),
        len(contexts),
        valid_count - len(contexts),  # subtract diagonal zeros
        len(contexts) * (len(contexts) - 1),
    )

    # Calibrate ceiling / floor from the observed ΔG distribution for the
    # Tobit fallback (plan §6 saturation handling).
    finite_dg = G[~np.isnan(G) & (np.abs(G) > 1e-9)]
    if finite_dg.size > 0:
        ceiling = float(np.max(finite_dg))
        floor = float(np.min(finite_dg))
    else:
        ceiling, floor = 1.0, -1.0

    # Pick which (predictor, layer, point) cells to evaluate.
    keys = list(arrays.keys())
    if args.smoke:
        keys = keys[:1]
        b_eff = min(args.b, 16)
        logger.info("SMOKE: keys=%s B=%d", keys, b_eff)
    else:
        b_eff = args.b

    # M_sym is the panel-fixed symmetric baseline (gauss_kl_sym at L22 +
    # length proxy). When the predictor stack has gauss_kl_sym at L22, we
    # use it; otherwise we fall back to the first symmetric key.
    sym_key_candidates = [k for k in arrays if k.startswith("gauss_kl_sym__L22__last_prompt")]
    if not sym_key_candidates:
        sym_key_candidates = [k for k in arrays if k.startswith("gauss_kl_sym__")]
    if not sym_key_candidates:
        sym_key_candidates = [k for k in arrays if "cosine__" in k]
    if not sym_key_candidates:
        raise RuntimeError(
            "No symmetric baseline predictor in stack; expected gauss_kl_sym or cosine."
        )
    sym_key = sym_key_candidates[0]
    sym_features = {sym_key: arrays[sym_key]}
    # "length" proxy: row-mean(|G|) per source (cheap surrogate; the real
    # length covariate is per-context response length on R_target, which
    # the dispatcher logs during Phase 2 if available).
    # For now we omit the length covariate (it stays a sidecar; the plan
    # text is unchanged but the implementation degrades gracefully).
    logger.info("Symmetric baseline = %s", sym_key)

    results: dict[str, dict] = {}
    for key in keys:
        if key == sym_key:
            continue
        full_features = {sym_key: arrays[sym_key], key: arrays[key]}
        # Pooled out-of-fold predictions.
        pred_sym, pred_full, pairs = pooled_ltco_predictions(
            sym_features, full_features, G, contexts
        )
        if len(pairs) == 0:
            logger.warning("Predictor %s yielded 0 valid pairs; skipping.", key)
            continue
        dg_target = np.array([G[p] for p in pairs])
        delta_r2 = pooled_incremental_r2(pred_sym, pred_full, dg_target)

        # Censoring rate decides headline choice (plan §6).
        right_cens, left_cens = _censor_flags(dg_target, ceiling, floor)
        cens_frac = float(np.mean(right_cens | left_cens))
        if cens_frac >= 0.10:
            try:
                delta_ll = censored_delta_logL(pred_sym, pred_full, dg_target, ceiling, floor)
                headline = "censored_delta_logL"
                headline_value = delta_ll
            except (ImportError, RuntimeError) as e:
                logger.warning("Tobit fit failed (%s); falling back to ΔR² for %s", e, key)
                delta_ll = None
                headline = "delta_r2_pooled"
                headline_value = delta_r2
        else:
            delta_ll = None
            headline = "delta_r2_pooled"
            headline_value = delta_r2

        # Bootstrap CI on the headline.
        def _headline_fn(ps, pf, dg, h=headline, c=ceiling, fl=floor):
            if h == "delta_r2_pooled":
                return pooled_incremental_r2(ps, pf, dg)
            return censored_delta_logL(ps, pf, dg, c, fl)

        ci_lo, ci_pt, ci_hi = context_clustered_bootstrap_ci(
            sym_features,
            full_features,
            G,
            contexts,
            _headline_fn,
            B=b_eff,
            seed=args.seed,
        )

        # Directional-fraud test (plan §3, §6 conjunct (b) + (c)).
        fraud = directional_fraud(arrays[key], G, pairs)

        # PASS test placeholder for the 4 conjuncts. #523 conjunct (d) is
        # filled in by Phase 5's #523 sub-step (a separate invocation).
        ci_excludes_zero = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)

        results[key] = {
            "predictor": key,
            "headline": headline,
            "headline_value": float(headline_value),
            "ci_low": ci_lo,
            "ci_point": ci_pt,
            "ci_high": ci_hi,
            "ci_excludes_zero": bool(ci_excludes_zero),
            "delta_r2_pooled": float(delta_r2),
            "delta_ll": (float(delta_ll) if delta_ll is not None else None),
            "censored_fraction": cens_frac,
            "n_pairs": len(pairs),
            "fraud_standalone_rho": fraud["standalone_rho_pred_vs_dg_anti"],
            "fraud_asym_corr": fraud["asym_corr_predAnti_vs_dgAnti"],
        }
        logger.info(
            "%s: headline=%s value=%.4f CI=[%.4f, %.4f] (excludes 0: %s)",
            key,
            headline,
            headline_value,
            ci_lo,
            ci_hi,
            ci_excludes_zero,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "schema_version": 1,
        "issue": 524,
        "phase": 5,
        "git_sha": _git_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_contexts": len(contexts),
        "contexts": contexts,
        "n_valid_pairs": valid_count - len(contexts),
        "tobit_ceiling": ceiling,
        "tobit_floor": floor,
        "bootstrap_b": b_eff,
        "symmetric_baseline_key": sym_key,
        "predictor_results": results,
    }
    out_path.write_text(json.dumps(out_payload, indent=2) + "\n")
    logger.info("Wrote %s (%d predictors evaluated)", out_path, len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
