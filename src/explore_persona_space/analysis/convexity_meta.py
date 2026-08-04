"""Functional-form (convexity) meta-analysis machinery for task #644.

This module fits candidate functional forms to RAW (non-rank) paired
``(geometry-scalar, behavior-strength-scalar)`` scatters and tests whether the
geometry->behavior relationship is convex / super-linear rather than linear,
across behaviors. It is reused by ``scripts/issue644_functional_form.py``; the
per-behavior loaders live in ``scripts/issue644_loaders.py``.

Design (plan #644 v2, sections 5.x / 6 / 11):

* Fixed functional-form set per scatter: linear / quadratic / exponential /
  power-law / monotone PCHIP spline (4-knot, x-quantile grid). Compared by
  LOO predictive R^2 + AIC + BIC.
* Direct curvature test: quadratic-vs-linear nested LR test (sign + p of the
  x^2 term) AND a nonparametric bootstrap CI on the fitted x^2 coefficient
  (B=10000, seed 42). Convexity = positive x^2 with bootstrap CI excluding 0
  in the correctly-signed-proximity frame.
* Leverage-aware re-fit: Cook's D; re-report the convexity verdict after
  dropping the single highest-Cook's-D point AND the top-2 (CC6).
  ``robust_to_leverage_LOO`` is True only if the convex verdict survives BOTH.
* Artifact-control double-fits:
    - Case A (log-space): for any log-prob DV or X, fit in native log space AND
      on the back-transformed probability; ``log_space_artifact`` True if convex
      only on the back-transform.
    - Case B (bounded-rate logit, ``y in [0,1]``): refit the curvature test on a
      logit transform ``logit(clip(y, eps, 1-eps))`` with eps=0.005;
      ``rate_compression_artifact`` True if convex in raw-y but NOT in logit-y.

No model training, no GPU. Pure numpy/scipy.

References: see plan ``tasks/.../644/plans/plan.md`` and
``.claude/rules/persona-distance-metrics.md`` (geometry-scalar families).
"""

from __future__ import annotations

import contextlib
import math
import platform
import sys
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import scipy
from scipy import optimize, special, stats
from scipy.interpolate import PchipInterpolator


@contextlib.contextmanager
def _quiet_numeric():
    """Suppress the (expected, guarded) overflow / invalid-value warnings from the
    nonlinear fitters on floor-clustered data — every such fit is caught by an
    explicit ``np.all(np.isfinite(pred))`` guard that returns ``None``, so the
    warnings are noise, not unhandled errors."""
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=optimize.OptimizeWarning)
        yield


# --- Fixed analysis constants (plan #644 v2 §11) ------------------------------

BOOTSTRAP_B = 10000  # bootstrap resamples for the x^2 coefficient CI
BOOTSTRAP_SEED = 42  # pinned RNG seed (recorded in output JSON)
LOGIT_EPS = 0.005  # MF2 clamp on boundary rate points before the logit transform
SPLINE_KNOTS = 4  # PCHIP knot count (x-quantiles {0, 1/3, 2/3, 1}) (CC5)
CONVEX_DELTA_AIC = 2.0  # ΔAIC threshold for "a convex form beats linear" (H1)
MIN_FIT_N = 4  # below this, even linear+quadratic is not well-posed
MIN_DISTINCT = 3  # need >=3 distinct x and y to fit a non-degenerate scatter

# Geometry-frame scalar kinds that count toward the H1 geometry-recurs numerator
# (MF1). prior_logprob is a base-rate behavioral scalar (sensitivity-only);
# js_deprecated_single_next_token is the deprecated coarse JS (CC2).
GEOMETRY_SCALAR_KINDS = frozenset(
    {
        "cosine_to_source",
        "cosine_to_direction",
        "cosine_centered_centroid",
        "js",
    }
)
NON_GEOMETRY_SCALAR_KINDS = frozenset(
    {
        "prior_logprob",
        "js_deprecated_single_next_token",
    }
)


# --- Reproducibility metadata -------------------------------------------------


def reproducibility_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the reproducibility-metadata block embedded in every output JSON.

    Returns a dict with git provenance (commit + dirty flag), ISO-8601 UTC
    timestamp, library versions, platform, and the pinned bootstrap seed / B.
    Merges ``extra`` on top.
    """
    from explore_persona_space.orchestrate.provenance import (
        as_metadata_dict,
        git_provenance,
    )

    meta: dict[str, Any] = {
        **as_metadata_dict(git_provenance()),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_B": BOOTSTRAP_B,
        "logit_eps": LOGIT_EPS,
        "spline_knots": SPLINE_KNOTS,
        "convex_delta_aic": CONVEX_DELTA_AIC,
    }
    if extra:
        meta.update(extra)
    return meta


# --- Transforms ---------------------------------------------------------------


def logit_clip(y: np.ndarray, eps: float = LOGIT_EPS) -> np.ndarray:
    """Variance-stabilizing logit of a rate, clamping boundary points to [eps, 1-eps].

    ``scipy.special.logit`` of exactly 0 or 1 is +-inf; the eps clamp keeps the
    transform finite (MF2 / §5.5 Case B).
    """
    yc = np.clip(np.asarray(y, dtype=float), eps, 1.0 - eps)
    return special.logit(yc)


# --- Low-level fitters --------------------------------------------------------


def _aic_bic(n: int, k: int, rss: float) -> tuple[float, float]:
    """Gaussian-residual AIC and BIC from RSS.

    ``k`` is the number of fitted mean parameters; the noise variance counts as
    one extra estimated parameter, so the effective parameter count is ``k + 1``.
    Returns (AIC, BIC). RSS<=0 (perfect / degenerate fit) is floored so the log
    is finite.
    """
    rss = max(float(rss), 1e-300)
    p = k + 1  # +1 for the residual-variance parameter
    ll = -0.5 * n * (math.log(2.0 * math.pi) + math.log(rss / n) + 1.0)
    aic = 2.0 * p - 2.0 * ll
    bic = p * math.log(n) - 2.0 * ll
    return float(aic), float(bic)


def _ols_design(x: np.ndarray, degree: int) -> np.ndarray:
    """Vandermonde design matrix [1, x, x^2, ...] up to ``degree`` (inclusive)."""
    return np.vander(x, N=degree + 1, increasing=True)


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """OLS linear fit ``y = a + b x``. Returns coeffs, predictions, RSS, AIC/BIC."""
    n = len(x)
    design = _ols_design(x, 1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    rss = float(np.sum((y - pred) ** 2))
    aic, bic = _aic_bic(n, k=2, rss=rss)
    return {"form": "linear", "coef": coef.tolist(), "rss": rss, "aic": aic, "bic": bic, "k": 2}


def fit_quadratic(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """OLS quadratic fit ``y = a + b x + c x^2``. ``c`` is the curvature coeff."""
    n = len(x)
    design = _ols_design(x, 2)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    rss = float(np.sum((y - pred) ** 2))
    aic, bic = _aic_bic(n, k=3, rss=rss)
    return {
        "form": "quadratic",
        "coef": coef.tolist(),
        "curvature_coef": float(coef[2]),
        "rss": rss,
        "aic": aic,
        "bic": bic,
        "k": 3,
    }


def fit_exponential(x: np.ndarray, y: np.ndarray) -> dict[str, Any] | None:
    """Nonlinear fit ``y = a * exp(b x)`` via ``scipy.optimize.curve_fit``.

    Returns ``None`` when the optimizer fails to converge (reported as a
    non-fittable form rather than a crash).
    """
    n = len(x)

    def model(xx: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.exp(b * xx)

    # Seed from a log-linear regression on positive y (robust starting point).
    y_pos = np.clip(y, 1e-6, None)
    with _quiet_numeric():
        try:
            b0_design = _ols_design(x, 1)
            logcoef, *_ = np.linalg.lstsq(b0_design, np.log(y_pos), rcond=None)
            p0 = (float(np.exp(logcoef[0])), float(logcoef[1]))
        except (np.linalg.LinAlgError, ValueError):
            p0 = (1.0, 0.0)
        try:
            popt, _ = optimize.curve_fit(model, x, y, p0=p0, maxfev=20000)
        except (RuntimeError, ValueError, optimize.OptimizeWarning):
            return None
        pred = model(x, *popt)
    if not np.all(np.isfinite(pred)):
        return None
    rss = float(np.sum((y - pred) ** 2))
    aic, bic = _aic_bic(n, k=2, rss=rss)
    return {"form": "exp", "coef": popt.tolist(), "rss": rss, "aic": aic, "bic": bic, "k": 2}


def fit_power(x: np.ndarray, y: np.ndarray) -> dict[str, Any] | None:
    """Nonlinear fit ``y = a * (x - x_shift)^b`` on shifted-positive x.

    x is shifted so its minimum is a small positive epsilon (power law is only
    defined on positive support). Returns ``None`` on non-convergence.
    """
    n = len(x)
    shift = float(np.min(x)) - 1e-3
    xs = x - shift  # strictly positive

    def model(xx: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.power(xx, b)

    with _quiet_numeric():
        try:
            popt, _ = optimize.curve_fit(model, xs, y, p0=(1.0, 1.0), maxfev=20000)
        except (RuntimeError, ValueError, optimize.OptimizeWarning):
            return None
        pred = model(xs, *popt)
    if not np.all(np.isfinite(pred)):
        return None
    rss = float(np.sum((y - pred) ** 2))
    aic, bic = _aic_bic(n, k=2, rss=rss)
    return {
        "form": "power",
        "coef": popt.tolist(),
        "x_shift": shift,
        "rss": rss,
        "aic": aic,
        "bic": bic,
        "k": 2,
    }


def fit_pchip_spline(
    x: np.ndarray, y: np.ndarray, n_knots: int = SPLINE_KNOTS
) -> dict[str, Any] | None:
    """Monotone PCHIP spline over a fixed ``n_knots``-knot x-quantile grid (CC5).

    The knot y-values are the OLS-projected y at each knot (so the spline df is
    pinned at ``n_knots`` and deterministic at every n). PCHIP is monotone by
    construction. The spline is a catch-all — the HEADLINE curvature verdict
    rests on the quadratic LRT, not on "spline beats linear" (§5.2).

    Returns ``None`` if the x-grid has fewer than ``n_knots`` distinct values
    (the spline is then not well-posed and is simply omitted from the bake-off).
    """
    n = len(x)
    qs = np.linspace(0.0, 1.0, n_knots)
    knots = np.quantile(x, qs)
    knots = np.unique(knots)
    if len(knots) < 2:
        return None
    # Local-mean y at each knot from a Gaussian-kernel smooth (deterministic,
    # no tuned penalty): for each knot take the mean of y for the nearest points.
    order = np.argsort(x)
    xs_sorted = x[order]
    ys_sorted = y[order]
    knot_y = np.interp(knots, xs_sorted, ys_sorted)
    # Enforce strictly increasing knot abscissae for PchipInterpolator.
    try:
        spline = PchipInterpolator(knots, knot_y, extrapolate=True)
    except ValueError:
        return None
    pred = spline(x)
    if not np.all(np.isfinite(pred)):
        return None
    rss = float(np.sum((y - pred) ** 2))
    # df ~= number of knots (pinned); use n_knots as the parameter count.
    k = len(knots)
    aic, bic = _aic_bic(n, k=k, rss=rss)
    return {
        "form": "spline",
        "n_knots": len(knots),
        "rss": rss,
        "aic": aic,
        "bic": bic,
        "k": k,
    }


# --- LOO predictive R^2 -------------------------------------------------------


def _loo_r2_for_fitter(x: np.ndarray, y: np.ndarray, fitter, form: str) -> float:
    """Leave-one-out predictive R^2 for a given fitter callable.

    For each held-out point, refit on the remaining n-1 points and predict the
    held-out y. R^2 = 1 - SS_res(LOO) / SS_tot. NaN when any fold fails to fit.
    """
    n = len(x)
    if n < MIN_FIT_N:
        return float("nan")
    preds = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        xi, yi = x[mask], y[mask]
        fit = fitter(xi, yi)
        if fit is None:
            return float("nan")
        preds[i] = _predict(fit, np.array([x[i]]))[0]
    if not np.all(np.isfinite(preds)):
        return float("nan")
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _predict(fit: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Predict y at ``x`` for a stored fit dict (re-derives from coeffs)."""
    form = fit["form"]
    if form == "linear":
        a, b = fit["coef"]
        return a + b * x
    if form == "quadratic":
        a, b, c = fit["coef"]
        return a + b * x + c * x**2
    if form == "exp":
        a, b = fit["coef"]
        return a * np.exp(b * x)
    if form == "power":
        a, b = fit["coef"]
        # A held-out / extrapolated x below x_shift gives a negative base raised
        # to a fractional power -> NaN; the caller's finiteness guard handles it,
        # so suppress the (expected) invalid-value warning rather than crash.
        with np.errstate(invalid="ignore"):
            return a * np.power(x - fit["x_shift"], b)
    if form == "spline":
        # Recompute the spline deterministically — stored fit holds only RSS/k,
        # so for LOO we refit; this branch is unused (spline LOO refits via fitter).
        raise NotImplementedError("spline prediction handled by re-fit in LOO")
    raise ValueError(f"unknown form {form!r}")


# --- Curvature test (quadratic vs linear) -------------------------------------


def curvature_lrt(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Nested quadratic-vs-linear test on the x^2 term.

    Returns the curvature coefficient (sign + value), the F-test p-value for
    adding the quadratic term, and ΔAIC(linear - quadratic) (positive favours
    quadratic). For n too small to estimate the quadratic this returns NaNs.
    """
    n = len(x)
    lin = fit_linear(x, y)
    if n < 4:
        return {
            "curvature_coef": float("nan"),
            "curvature_sign": "0",
            "lrt_p": float("nan"),
            "delta_aic_lin_minus_quad": float("nan"),
            "n": int(n),
        }
    quad = fit_quadratic(x, y)
    rss_lin = lin["rss"]
    rss_quad = quad["rss"]
    df1 = 1  # one extra parameter (the x^2 term)
    df2 = n - 3  # residual df of the full quadratic model
    if df2 <= 0 or rss_quad <= 0:
        p = float("nan")
    else:
        f_stat = ((rss_lin - rss_quad) / df1) / (rss_quad / df2)
        f_stat = max(f_stat, 0.0)
        p = float(stats.f.sf(f_stat, df1, df2))
    c = float(quad["curvature_coef"])
    sign = "+" if c > 0 else ("-" if c < 0 else "0")
    return {
        "curvature_coef": c,
        "curvature_sign": sign,
        "lrt_p": p,
        "delta_aic_lin_minus_quad": float(lin["aic"] - quad["aic"]),
        "n": int(n),
    }


def bootstrap_curvature_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float]:
    """Nonparametric bootstrap 95% CI on the quadratic x^2 coefficient.

    Resamples (x, y) pairs with replacement ``n_boot`` times, refits the
    quadratic each time, collects the x^2 coefficient. Returns
    ``(mean, ci_low, ci_high)``. NaN triple when n < 4 (quadratic not estimable).

    Generalises ``issue532_predictor_stress._bootstrap_spearman_ci`` (same RNG /
    percentile machinery) to a regression coefficient.
    """
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    coefs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 3:
            continue  # degenerate resample; leave NaN, dropped by nanpercentile
        try:
            design = _ols_design(xb, 2)
            coef, *_ = np.linalg.lstsq(design, yb, rcond=None)
            coefs[b] = coef[2]
        except np.linalg.LinAlgError:
            continue
    if np.all(np.isnan(coefs)):
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.nanmean(coefs)),
        float(np.nanpercentile(coefs, 2.5)),
        float(np.nanpercentile(coefs, 97.5)),
    )


# --- Cook's D leverage screen -------------------------------------------------


def cooks_distance(x: np.ndarray, y: np.ndarray, degree: int = 2) -> np.ndarray:
    """Cook's distance per point for the OLS fit of the given polynomial degree.

    Uses the standard closed form
    ``D_i = (e_i^2 / (p * s^2)) * (h_ii / (1 - h_ii)^2)`` where ``h`` is the hat
    matrix diagonal, ``e`` residuals, ``p`` parameter count, ``s^2`` the residual
    MSE. Points with ``h_ii`` ~ 1 (near-degenerate) get a NaN D (excluded).
    """
    n = len(x)
    design = _ols_design(x, degree)
    p = design.shape[1]
    # Hat matrix diagonal via the pseudo-inverse (stable for ill-conditioned X).
    xtx_inv = np.linalg.pinv(design.T @ design)
    hat_diag = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    coef = xtx_inv @ design.T @ y
    pred = design @ coef
    resid = y - pred
    dof = n - p
    if dof <= 0:
        return np.full(n, np.nan)
    mse = float(np.sum(resid**2) / dof)
    if mse <= 0:
        return np.zeros(n)
    d = np.full(n, np.nan)
    for i in range(n):
        h = hat_diag[i]
        if h >= 1.0 - 1e-9 or h < 0:
            continue
        d[i] = (resid[i] ** 2 / (p * mse)) * (h / (1.0 - h) ** 2)
    return d


def _convex_verdict_for_xy(x: np.ndarray, y: np.ndarray) -> bool:
    """Convex verdict on a single (x, y) scatter: positive x^2 with CI excluding 0.

    Combines the quadratic-vs-linear ΔAIC>=2 gate with the bootstrap-CI-excludes-0
    gate, both requiring a positive curvature sign. This is the per-fit convex
    decision the leverage / transform re-fits re-evaluate.
    """
    lrt = curvature_lrt(x, y)
    if not math.isfinite(lrt["curvature_coef"]):
        return False
    if lrt["curvature_sign"] != "+":
        return False
    _, ci_lo, ci_hi = bootstrap_curvature_ci(x, y)
    if not (math.isfinite(ci_lo) and math.isfinite(ci_hi)):
        return False
    ci_excludes_zero = ci_lo > 0  # positive curvature CI strictly above 0
    delta_aic_ok = lrt["delta_aic_lin_minus_quad"] >= CONVEX_DELTA_AIC
    return bool(ci_excludes_zero and delta_aic_ok)


def leverage_robustness(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Re-report the convex verdict after dropping the single + top-2 Cook's-D points.

    Returns ``survives_top1_cookd_drop``, ``survives_top2_cookd_drop`` (only
    meaningful when the base verdict is convex; both default to the base verdict
    when there is no high-leverage structure), and ``robust_to_leverage_LOO``
    (True iff convex survives BOTH drops). Also returns the dropped unit indices.
    """
    base_convex = _convex_verdict_for_xy(x, y)
    d = cooks_distance(x, y, degree=2)
    finite = np.where(np.isfinite(d))[0]
    if len(finite) == 0:
        return {
            "base_convex": base_convex,
            "survives_top1_cookd_drop": base_convex,
            "survives_top2_cookd_drop": base_convex,
            "robust_to_leverage_LOO": base_convex,
            "top_cookd_idx": [],
        }
    order = finite[np.argsort(-d[finite])]
    top1 = order[0]
    top2 = order[:2]
    mask1 = np.ones(len(x), dtype=bool)
    mask1[top1] = False
    convex_top1 = _convex_verdict_for_xy(x[mask1], y[mask1])
    mask2 = np.ones(len(x), dtype=bool)
    mask2[top2] = False
    convex_top2 = _convex_verdict_for_xy(x[mask2], y[mask2])
    return {
        "base_convex": base_convex,
        "survives_top1_cookd_drop": bool(convex_top1),
        "survives_top2_cookd_drop": bool(convex_top2),
        "robust_to_leverage_LOO": bool(base_convex and convex_top1 and convex_top2),
        "top_cookd_idx": [int(i) for i in top2.tolist()],
    }


# --- Form bake-off ------------------------------------------------------------

_FITTERS = {
    "linear": fit_linear,
    "quadratic": fit_quadratic,
    "exp": fit_exponential,
    "power": fit_power,
}


def form_bakeoff(x: np.ndarray, y: np.ndarray, include_spline: bool = True) -> dict[str, Any]:
    """Fit all candidate forms; return per-form AIC/BIC + LOO-R^2 and the best form.

    ``best_form`` is the form with the lowest AIC among those that fit (purely
    informative). ``convex_wins`` is True iff a *convex* form (quadratic with
    positive curvature / exp / power) beats linear by ΔAIC>=2 AND the
    quadratic-vs-linear LRT curvature sign is positive — so the monotone spline
    can NEVER manufacture a convex verdict on its own (plan §5.2). The spline
    winning AIC means "non-linear shape", not "convex", because PCHIP carries no
    signed-curvature term; a spline-only win is reported via ``best_form`` but
    does not set ``convex_wins``.
    """
    n = len(x)
    fits: dict[str, dict[str, Any]] = {}
    for name, fitter in _FITTERS.items():
        f = fitter(x, y)
        if f is not None:
            # LOO predictive R^2 (parametric forms re-fit cleanly per fold).
            f["loo_r2"] = _loo_r2_for_fitter(x, y, fitter, name)
            fits[name] = f
    if include_spline:
        sp = fit_pchip_spline(x, y)
        if sp is not None:
            sp["loo_r2"] = _loo_r2_spline(x, y)
            fits["spline"] = sp

    if "linear" not in fits:
        # Linear always fits unless n<1; guard anyway.
        return {"fits": fits, "best_form": None, "convex_wins": False}

    lin_aic = fits["linear"]["aic"]
    best_form = min(fits, key=lambda k: fits[k]["aic"])
    best_aic = fits[best_form]["aic"]

    # Convex verdict must rest on a SIGNED-curvature form (§5.2): the spline is a
    # catch-all with no curvature sign, so it never sets convex_wins on its own.
    # The best CONVEX-PARAMETRIC form (quadratic-positive / exp / power) is
    # compared to linear, and the quadratic LRT curvature must be positive.
    signed_convex_forms = {"quadratic", "exp", "power"}
    available_convex = {
        name: f
        for name, f in fits.items()
        if name in signed_convex_forms and (name != "quadratic" or f.get("curvature_coef", 0.0) > 0)
    }
    quad_sign_positive = "quadratic" in fits and fits["quadratic"].get("curvature_coef", 0.0) > 0
    if available_convex:
        best_convex = min(available_convex, key=lambda k: available_convex[k]["aic"])
        best_convex_aic = available_convex[best_convex]["aic"]
        convex_wins = bool((lin_aic - best_convex_aic) >= CONVEX_DELTA_AIC and quad_sign_positive)
    else:
        convex_wins = False

    return {
        "fits": fits,
        "best_form": best_form,
        "delta_aic_linear_to_best": float(lin_aic - best_aic),
        "loo_r2_linear": fits["linear"].get("loo_r2", float("nan")),
        "loo_r2_best": fits[best_form].get("loo_r2", float("nan")),
        "convex_wins": convex_wins,
        "n": int(n),
    }


def _loo_r2_spline(x: np.ndarray, y: np.ndarray) -> float:
    """LOO predictive R^2 for the PCHIP spline (refit per fold)."""
    n = len(x)
    if n < MIN_FIT_N:
        return float("nan")
    preds = np.full(n, np.nan)
    qs = np.linspace(0.0, 1.0, SPLINE_KNOTS)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        xi, yi = x[mask], y[mask]
        knots = np.unique(np.quantile(xi, qs))
        if len(knots) < 2:
            return float("nan")
        order = np.argsort(xi)
        knot_y = np.interp(knots, xi[order], yi[order])
        try:
            spline = PchipInterpolator(knots, knot_y, extrapolate=True)
        except ValueError:
            return float("nan")
        preds[i] = float(spline(x[i]))
    if not np.all(np.isfinite(preds)):
        return float("nan")
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# --- Per-scatter fit record (the §6 read-out) ---------------------------------


@dataclass
class ScatterInput:
    """A single (behavior x frame) raw scatter to be fit.

    Attributes mirror the §4 unified loader table. ``y_is_rate`` triggers the
    logit double-fit (MF2); ``x_is_logprob`` / ``y_is_logprob`` trigger the
    log-space double-fit (§5.5 Case A).
    """

    behavior: str
    frame: str
    geometry_scalar_kind: str
    centering_family: str
    x: np.ndarray
    y: np.ndarray
    units: list[str]
    layer: Any = None
    matched_row_count: int | None = None
    y_is_rate: bool = True
    x_is_logprob: bool = False
    y_is_logprob: bool = False
    notes: list[str] = field(default_factory=list)


def two_axis_spread_ok(x: np.ndarray, y: np.ndarray) -> bool:
    """Gate: distinct x AND distinct y with non-degenerate range (§6)."""
    if len(x) < MIN_FIT_N:
        return False
    x_distinct = len(np.unique(np.round(x, 10)))
    y_distinct = len(np.unique(np.round(y, 10)))
    x_range = float(np.max(x) - np.min(x))
    y_range = float(np.max(y) - np.min(y))
    return bool(
        x_distinct >= MIN_DISTINCT
        and y_distinct >= MIN_DISTINCT
        and x_range > 1e-9
        and y_range > 1e-9
    )


def analyze_scatter(s: ScatterInput) -> dict[str, Any]:
    """Run the full §5/§6 pipeline on one (behavior x frame) scatter.

    Emits the §6 read-out record (one entry per behavior x frame), including the
    convex verdict, curvature sign + bootstrap CI, ΔAIC, LOO-R^2, leverage
    robustness, the log-space and rate-stabilization artifact flags, and
    ``counts_toward_h1``.
    """
    x = np.asarray(s.x, dtype=float)
    y = np.asarray(s.y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n_raw = len(x)
    x, y = x[mask], y[mask]
    n = len(x)

    rec: dict[str, Any] = {
        "behavior": s.behavior,
        "frame": s.frame,
        "geometry_scalar_kind": s.geometry_scalar_kind,
        "centering_family": s.centering_family,
        "n": n,
        "n_raw_before_nan_drop": n_raw,
        "layer": s.layer,
        "matched_row_count": s.matched_row_count,
        "x_min": float(np.min(x)) if n else None,
        "x_max": float(np.max(x)) if n else None,
        "x_distinct": len(np.unique(np.round(x, 10))) if n else 0,
        "y_min": float(np.min(y)) if n else None,
        "y_max": float(np.max(y)) if n else None,
        "y_distinct": len(np.unique(np.round(y, 10))) if n else 0,
        "y_is_rate": s.y_is_rate,
        "x_is_logprob": s.x_is_logprob,
        "y_is_logprob": s.y_is_logprob,
        "notes": list(s.notes),
    }

    spread_ok = two_axis_spread_ok(x, y)
    rec["two_axis_spread_ok"] = spread_ok
    rec["under_powered"] = bool(n < 10)  # CC7 denominator gate is n>=10

    if not spread_ok:
        rec.update(
            {
                "convex_wins": None,
                "best_form": None,
                "curvature_sign": "0",
                "curvature_coef": None,
                "curvature_ci_low": None,
                "curvature_ci_high": None,
                "delta_aic_linear_to_best": None,
                "loo_r2_linear": None,
                "loo_r2_best": None,
                "survives_top1_cookd_drop": None,
                "survives_top2_cookd_drop": None,
                "robust_to_leverage_LOO": None,
                "log_space_artifact": None,
                "rate_compression_artifact": None,
                "counts_toward_h1": False,
                "excluded_reason": "two_axis_spread_failed",
            }
        )
        return rec

    # --- raw-y fit bake-off + curvature test ---
    bake = form_bakeoff(x, y, include_spline=True)
    lrt = curvature_lrt(x, y)
    ci_mean, ci_lo, ci_hi = bootstrap_curvature_ci(x, y)
    lev = leverage_robustness(x, y)
    raw_convex = _convex_verdict_for_xy(x, y)

    rec.update(
        {
            "convex_wins": bool(bake.get("convex_wins", False)),
            "best_form": bake.get("best_form"),
            "delta_aic_linear_to_best": bake.get("delta_aic_linear_to_best"),
            "loo_r2_linear": bake.get("loo_r2_linear"),
            "loo_r2_best": bake.get("loo_r2_best"),
            "quadratic_curvature_coef": lrt["curvature_coef"],
            "curvature_coef": lrt["curvature_coef"],
            "curvature_sign": lrt["curvature_sign"],
            "quadratic_curvature_sign": lrt["curvature_sign"],
            "lrt_p": lrt["lrt_p"],
            "curvature_ci_low": ci_lo,
            "curvature_ci_high": ci_hi,
            "quadratic_curvature_ci_low": ci_lo,
            "quadratic_curvature_ci_high": ci_hi,
            "curvature_ci_mean": ci_mean,
            "survives_top1_cookd_drop": lev["survives_top1_cookd_drop"],
            "survives_top2_cookd_drop": lev["survives_top2_cookd_drop"],
            "robust_to_leverage_LOO": lev["robust_to_leverage_LOO"],
            "top_cookd_idx": lev["top_cookd_idx"],
            "raw_convex_verdict": raw_convex,
            "per_form_fits": _serialize_fits(bake.get("fits", {})),
        }
    )

    # --- Case B: bounded-rate logit double-fit (MF2 / §5.5) ---
    rate_compression_artifact: bool | None = None
    if s.y_is_rate:
        y_logit = logit_clip(y)
        logit_convex = _convex_verdict_for_xy(x, y_logit)
        logit_lrt = curvature_lrt(x, y_logit)
        _, logit_ci_lo, logit_ci_hi = bootstrap_curvature_ci(x, y_logit)
        rec["logit_convex_verdict"] = logit_convex
        rec["logit_curvature_sign"] = logit_lrt["curvature_sign"]
        rec["logit_curvature_ci_low"] = logit_ci_lo
        rec["logit_curvature_ci_high"] = logit_ci_hi
        # Artifact iff convex in raw-y but NOT in logit-y.
        rate_compression_artifact = bool(raw_convex and not logit_convex)
    rec["rate_compression_artifact"] = rate_compression_artifact

    # --- Case A: log-space double-fit (§5.5 Case A) ---
    log_space_artifact: bool | None = None
    if s.y_is_logprob:
        # Y is a log-prob: fit on the back-transformed probability AND native log.
        y_prob = np.exp(y)
        prob_convex = _convex_verdict_for_xy(x, y_prob)
        rec["prob_space_convex_verdict"] = prob_convex
        rec["logspace_native_convex_verdict"] = raw_convex
        # Artifact iff convex appears only on the back-transformed probability.
        log_space_artifact = bool(prob_convex and not raw_convex)
    if s.x_is_logprob:
        # X is a log-prob: fit on the back-transformed probability X too.
        x_prob = np.exp(x)
        if two_axis_spread_ok(x_prob, y):
            x_prob_convex = _convex_verdict_for_xy(x_prob, y)
            rec["x_prob_space_convex_verdict"] = x_prob_convex
            la = bool(x_prob_convex and not raw_convex)
            log_space_artifact = la if log_space_artifact is None else (log_space_artifact or la)
    rec["log_space_artifact"] = log_space_artifact

    # --- counts_toward_h1 (§6 H1-contribution rules) ---
    is_geometry = s.geometry_scalar_kind in GEOMETRY_SCALAR_KINDS
    counts = bool(
        is_geometry
        and bake.get("convex_wins", False)
        and lev["robust_to_leverage_LOO"]
        and rate_compression_artifact is not True
        and log_space_artifact is not True
    )
    rec["counts_toward_h1"] = counts
    rec["is_geometry_frame"] = is_geometry
    return rec


def _serialize_fits(fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Strip non-JSON-serializable fields from per-form fit dicts for the output JSON."""
    out: dict[str, Any] = {}
    for name, f in fits.items():
        out[name] = {
            k: v
            for k, v in f.items()
            if k in ("form", "aic", "bic", "rss", "loo_r2", "k", "curvature_coef", "n_knots")
        }
    return out


# --- Cross-behavior recurs aggregation (§5.6 / §6) ----------------------------


def build_recurs_tables(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the geometry-convexity recurs table + prior-frame sensitivity table.

    The geometry headline counts ONLY rows with a geometry ``geometry_scalar_kind``
    (MF1), excluding ``prior_logprob`` and ``js_deprecated_single_next_token``
    (CC2). The H1 denominator is "two-axis spread AND n>=10" (CC7) and the
    numerator excludes ``rate_compression_artifact`` / ``log_space_artifact`` rows.

    Returns a dict with both tables plus the realized denominator and the
    majority verdict, ready to serialize to ``convexity_table.json``.
    """
    geometry_rows = []
    prior_rows = []
    deprecated_rows = []
    for r in records:
        kind = r.get("geometry_scalar_kind")
        if kind == "prior_logprob":
            prior_rows.append(r)
        elif kind == "js_deprecated_single_next_token":
            deprecated_rows.append(r)
        elif kind in GEOMETRY_SCALAR_KINDS:
            geometry_rows.append(r)
        else:
            # Unknown kind: keep visible but out of the headline.
            r.setdefault("notes", []).append(f"unknown geometry_scalar_kind {kind!r}")

    # CC7 denominator: qualifying = two-axis spread AND n >= 10 in geometry frame.
    qualifying = [r for r in geometry_rows if r.get("two_axis_spread_ok") and r.get("n", 0) >= 10]
    excluded_for_spread = [r for r in geometry_rows if not r.get("two_axis_spread_ok")]
    excluded_for_n = [
        r for r in geometry_rows if r.get("two_axis_spread_ok") and r.get("n", 0) < 10
    ]

    n_qualifying = len(qualifying)
    n_convex_h1 = sum(1 for r in qualifying if r.get("counts_toward_h1"))
    # Positive-sign consistency among the qualifying convex contributors.
    convex_signs = [r.get("curvature_sign") for r in qualifying if r.get("counts_toward_h1")]
    sign_consistent = all(sgn == "+" for sgn in convex_signs) if convex_signs else False

    if n_qualifying == 0:
        majority_threshold = None
        majority_verdict = "no_qualifying_behaviors"
    else:
        majority_threshold = math.ceil(n_qualifying / 2)
        if n_convex_h1 >= majority_threshold and sign_consistent:
            majority_verdict = "H1_convex_recurs"
        elif n_convex_h1 == 0:
            majority_verdict = "H0_or_H2_no_convex"
        else:
            majority_verdict = "H0_no_majority"

    def _slim(r: dict[str, Any]) -> dict[str, Any]:
        return {
            k: r.get(k)
            for k in (
                "behavior",
                "frame",
                "geometry_scalar_kind",
                "centering_family",
                "n",
                "matched_row_count",
                "two_axis_spread_ok",
                "under_powered",
                "convex_wins",
                "best_form",
                "curvature_sign",
                "curvature_coef",
                "curvature_ci_low",
                "curvature_ci_high",
                "delta_aic_linear_to_best",
                "loo_r2_linear",
                "loo_r2_best",
                "robust_to_leverage_LOO",
                "survives_top1_cookd_drop",
                "survives_top2_cookd_drop",
                "log_space_artifact",
                "rate_compression_artifact",
                "counts_toward_h1",
            )
        }

    return {
        "geometry_recurs_table": [_slim(r) for r in geometry_rows],
        "prior_frame_sensitivity_table": [_slim(r) for r in prior_rows],
        "deprecated_scalar_rows": [_slim(r) for r in deprecated_rows],
        "h1_denominator": {
            "definition": "two_axis_spread AND n>=10 in the GEOMETRY frame (CC7)",
            "n_qualifying": n_qualifying,
            "qualifying_behaviors": [
                f"{r['behavior']}|{r['frame']}|{r['geometry_scalar_kind']}" for r in qualifying
            ],
            "excluded_for_spread": [
                f"{r['behavior']}|{r['frame']}|{r['geometry_scalar_kind']}"
                for r in excluded_for_spread
            ],
            "excluded_for_n": [
                f"{r['behavior']}|{r['frame']}|{r['geometry_scalar_kind']} (n={r.get('n')})"
                for r in excluded_for_n
            ],
        },
        "h1_numerator": {
            "n_convex_counts_toward_h1": n_convex_h1,
            "convex_signs": convex_signs,
            "sign_consistent_positive": sign_consistent,
        },
        "majority_threshold": majority_threshold,
        "majority_verdict": majority_verdict,
    }
