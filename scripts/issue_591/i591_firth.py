#!/usr/bin/env python3
"""Task #591 — in-repo Firth-penalized logistic regression.

Jeffreys-prior bias-reduced logistic regression (Firth 1993, Biometrika 80:27;
Heinze & Schemper 2002, Stat Med 21:2409) with profile-penalized-likelihood
confidence intervals and penalized-likelihood-ratio p-values — the logistf
R-package inference suite.

Why in-repo (plan #591 v1 §4.1/§11, search-before-build done): PyPI
``firthlogist`` 0.5.0 pins ``numpy<2.0`` which is incompatible with the repo
env (numpy 2.2.6), so this ~200-line validated implementation is cheaper than
an env fork.

Validation contract (run BEFORE first analysis use):

    uv run python scripts/issue_591/i591_firth.py --validate

fits ``case ~ age + oc + vic + vicl + vis + dia`` on the bundled ``sex2``
dataset (Heinze & Schemper 2002 urinary-tract-infection case-control study,
239 rows; CSV vendored at ``eval_results/issue_591/_inputs/sex2.csv`` from the
firthlogist repo, sha256 c344363838a37ce9...) and asserts the coefficient
vector matches the published ``logistf::sex2`` reference to 1e-4. The profile
CI machinery is validated by self-consistency: the penalized-likelihood drop
at each returned bound must equal chi2(1, 0.95) = 3.841459 to 1e-3.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import chi2

CHI2_95_DF1 = float(chi2.ppf(0.95, df=1))  # 3.841458820694124


class ProfileNonConvergenceError(RuntimeError):
    """Constrained (profile) Firth fit failed to converge at a probe point.

    Raised by ``_profile_pl_at`` after the damped rescue ladder is exhausted.
    Callers treat the affected bound / p-value as not estimable by profile
    likelihood and fall back to the (explicitly flagged) Wald quantity —
    never crash the whole pipeline on one quasi-separated probe point
    (#591 e5 production incident: coef pinned to 8.0982 during the CI
    bound search killed an 11-hour run at the final refit phase).
    """


# Published logistf::sex2 reference coefficients (logistf docs; also reproduced
# in the firthlogist README). Order: intercept, age, oc, vic, vicl, vis, dia.
SEX2_REFERENCE_COEF = {
    "intercept": 0.12025405,
    "age": -1.10598130,
    "oc": -0.06881673,
    "vic": 2.26887464,
    "vicl": -2.11140817,
    "vis": -0.78831694,
    "dia": 3.09601263,
}
SEX2_CSV_DEFAULT = Path("eval_results/issue_591/_inputs/sex2.csv")


@dataclass
class FirthResult:
    """One Firth fit: coefficients + inference suite (logistf-style)."""

    names: list[str]
    beta: np.ndarray  # (p,) penalized-MLE coefficients
    se: np.ndarray  # (p,) Wald SEs from the inverse Fisher information
    pl: float  # penalized log-likelihood at the optimum
    converged: bool
    n_iter: int
    ci_low: np.ndarray = field(default=None)  # profile-PL 95% lower bounds
    ci_high: np.ndarray = field(default=None)  # profile-PL 95% upper bounds
    p_values: np.ndarray = field(default=None)  # penalized-LR test p per coef
    # Per-coefficient inference provenance: "profile" / "plr" for the exact
    # PL machinery (the common case, unchanged), "wald-fallback" where the
    # constrained fit was non-estimable at the probe extreme (#591 e5 fix).
    ci_method_low: list[str] = field(default=None)
    ci_method_high: list[str] = field(default=None)
    p_method: list[str] = field(default=None)

    def to_dict(self) -> dict:
        """JSON-friendly summary (odds ratios on exp scale, CIs both scales)."""
        out = {
            "names": self.names,
            "coef": self.beta.tolist(),
            "se": self.se.tolist(),
            "odds_ratio": np.exp(self.beta).tolist(),
            "penalized_loglik": self.pl,
            "converged": self.converged,
            "n_iter": self.n_iter,
        }
        if self.ci_low is not None:
            out["ci95_low_coef"] = self.ci_low.tolist()
            out["ci95_high_coef"] = self.ci_high.tolist()
            out["ci95_low_or"] = np.exp(self.ci_low).tolist()
            out["ci95_high_or"] = np.exp(self.ci_high).tolist()
            out["ci95_method_low"] = list(self.ci_method_low)
            out["ci95_method_high"] = list(self.ci_method_high)
        if self.p_values is not None:
            out["p_plr"] = self.p_values.tolist()
            out["p_method"] = list(self.p_method)
        fallback = set()
        for methods in (self.ci_method_low, self.ci_method_high, self.p_method):
            if methods is not None:
                fallback |= {
                    n for n, m in zip(self.names, methods, strict=True) if m == "wald-fallback"
                }
        if self.ci_low is not None or self.p_values is not None:
            out["profile_fallback_coefs"] = sorted(fallback)
        return out


def _penalized_loglik(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """LL + 0.5*log|X'WX| (Jeffreys penalty), eta clipped for stability."""
    eta = np.clip(X @ beta, -30.0, 30.0)
    pi = expit(eta)
    ll = float(np.sum(y * np.log(pi) + (1.0 - y) * np.log1p(-pi)))
    w = pi * (1.0 - pi)
    xw = X * np.sqrt(w)[:, None]
    info = xw.T @ xw
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0:
        return -np.inf
    return ll + 0.5 * float(logdet)


def _firth_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fixed: dict[int, float] | None = None,
    max_iter: int = 200,
    tol: float = 1e-9,
    max_step: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float, bool, int]:
    """Newton-Raphson on the Firth-modified score, optionally with pinned coords.

    ``fixed`` maps column index -> pinned coefficient value (the profile-
    likelihood constrained fit). The penalty always uses the FULL p x p
    information matrix (Heinze & Schemper 2002 profile recipe).

    Returns (beta, cov, pl, converged, n_iter); ``cov`` is the inverse full
    Fisher information at the solution.
    """
    n, p = X.shape
    assert y.shape == (n,), (y.shape, n)
    beta = np.zeros(p)
    free = np.array([j for j in range(p) if not (fixed and j in fixed)], dtype=int)
    if fixed:
        for j, v in fixed.items():
            beta[j] = v
    pl = _penalized_loglik(X, y, beta)
    converged = False
    n_done = 0
    info = np.eye(p)
    for it in range(1, max_iter + 1):
        n_done = it
        eta = np.clip(X @ beta, -30.0, 30.0)
        pi = expit(eta)
        w = pi * (1.0 - pi)
        xw = X * np.sqrt(w)[:, None]
        info = xw.T @ xw
        info_inv = np.linalg.inv(info)
        # Hat-matrix diagonal h_i = w_i * x_i' I^{-1} x_i.
        h = np.einsum("ij,jk,ik->i", X, info_inv, X) * w
        u_mod = X.T @ (y - pi + h * (0.5 - pi))
        if free.size == 0:
            converged = True
            break
        delta = np.zeros(p)
        delta[free] = np.linalg.solve(info[np.ix_(free, free)], u_mod[free])
        # Cap the raw Newton step (separation-prone designs).
        step_norm = float(np.max(np.abs(delta)))
        if step_norm > max_step:
            delta *= max_step / step_norm
        # Step-halving until the penalized likelihood does not decrease.
        new_beta = beta + delta
        new_pl = _penalized_loglik(X, y, new_beta)
        n_halv = 0
        while new_pl < pl - 1e-12 and n_halv < 25:
            delta *= 0.5
            new_beta = beta + delta
            new_pl = _penalized_loglik(X, y, new_beta)
            n_halv += 1
        beta, pl = new_beta, new_pl
        if float(np.max(np.abs(delta))) < tol:
            converged = True
            break
    cov = np.linalg.inv(info)
    return beta, cov, pl, converged, n_done


# Rescue ladder for the constrained (profile) fit. The first rung is the
# exact pre-fix configuration, so every previously-converging probe point
# returns the identical penalized log-likelihood (targeted robustness patch,
# not a behavior change). Later rungs damp the Newton step harder and allow
# more iterations — extreme pinned values (quasi-separation regime) flatten
# the likelihood and make the undamped step overshoot.
_PROFILE_RESCUE_LADDER: tuple[dict, ...] = (
    {"max_iter": 200, "max_step": 5.0},  # rung 0 == round-1 behavior
    {"max_iter": 1000, "max_step": 1.0},
    {"max_iter": 5000, "max_step": 0.25},
)


def _profile_pl_at(X: np.ndarray, y: np.ndarray, j: int, value: float) -> float:
    """Maximized penalized log-likelihood with beta_j pinned to ``value``.

    Walks ``_PROFILE_RESCUE_LADDER`` (progressively damped Newton). Raises
    ``ProfileNonConvergenceError`` only after every rung fails — callers
    convert that into a flagged Wald fallback for the affected bound.
    """
    last_reason = "unknown"
    for rung, kw in enumerate(_PROFILE_RESCUE_LADDER):
        try:
            _beta, _cov, pl, conv, n_iter = _firth_newton(X, y, fixed={j: value}, **kw)
        except np.linalg.LinAlgError as err:  # singular information at the probe
            last_reason = f"rung {rung} LinAlgError: {err}"
            continue
        if conv:
            return pl
        last_reason = f"rung {rung} no convergence in {n_iter} iters (max_step={kw['max_step']})"
    raise ProfileNonConvergenceError(
        f"profile fit did not converge (coef {j} pinned to {value:.4f}; "
        f"rescue ladder exhausted; last: {last_reason})"
    )


def _profile_ci_one(
    X: np.ndarray,
    y: np.ndarray,
    j: int,
    beta_hat: float,
    se_j: float,
    pl_hat: float,
    *,
    chi2_crit: float = CHI2_95_DF1,
    name: str | None = None,
) -> tuple[float, float, str, str]:
    """Invert the penalized likelihood ratio for one coefficient (both sides).

    Returns ``(low, high, low_method, high_method)`` where each method is
    ``"profile"`` (the bound is the exact PL inversion — unchanged behavior)
    or ``"wald-fallback"`` (the PL bound was not estimable at that extreme:
    either the constrained fit failed to converge after the rescue ladder,
    or the bracket search found the penalized likelihood too flat to drop by
    chi2_crit within 60 expansions). Fallback bounds are
    ``beta_hat ± sqrt(chi2_crit) * se_j`` and are loudly logged + flagged in
    the output JSON — never silently swallowed (fail-fast culture: the flag
    IS the signal, the crash was the bug).
    """

    def g(b: float) -> float:
        return 2.0 * (pl_hat - _profile_pl_at(X, y, j, b)) - chi2_crit

    label = f"coef {j}" + (f" ({name!r})" if name else "")
    z_crit = float(np.sqrt(chi2_crit))
    bounds: list[float] = []
    methods: list[str] = []
    for direction in (-1.0, +1.0):
        side = "low" if direction < 0 else "high"
        try:
            step = max(se_j, 1e-3)
            lo = beta_hat
            hi = beta_hat + direction * step
            n_expand = 0
            while g(hi) < 0 and n_expand < 60:
                lo = hi
                hi = hi + direction * step
                step *= 1.5
                n_expand += 1
            if g(hi) < 0:
                raise ProfileNonConvergenceError(
                    f"bracket failed for {label} ({side}): penalized likelihood "
                    f"too flat to drop by {chi2_crit:.4f} within 60 expansions"
                )
            a, b = (hi, lo) if direction < 0 else (lo, hi)
            bound = float(brentq(g, a, b, xtol=1e-6))
            method = "profile"
        except ProfileNonConvergenceError as err:
            bound = float(beta_hat + direction * z_crit * se_j)
            method = "wald-fallback"
            print(
                f"[i591_firth] WARNING: profile {side} CI bound for {label} not "
                f"estimable by penalized likelihood ({err}); Wald fallback "
                f"beta {'-' if direction < 0 else '+'} {z_crit:.4f}*SE = {bound:+.4f} "
                f"used and flagged in the output JSON.",
                file=sys.stderr,
                flush=True,
            )
        bounds.append(bound)
        methods.append(method)
    return bounds[0], bounds[1], methods[0], methods[1]


def firth_logistic(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    add_intercept: bool = True,
    profile_ci: bool = True,
    plr_pvalues: bool = True,
) -> FirthResult:
    """Fit Firth-penalized logistic regression with logistf-style inference.

    Args:
        X: (n, p) design WITHOUT intercept (added here when ``add_intercept``).
        y: (n,) binary outcome in {0, 1}.
        names: column names for X (length p).
        add_intercept: prepend an all-ones column named ``intercept``.
        profile_ci: compute 95% profile-penalized-likelihood CIs per coef.
        plr_pvalues: compute penalized-likelihood-ratio p-values per coef.

    Returns:
        FirthResult. Raises RuntimeError if the primary fit fails to converge.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0], (X.shape, y.shape)
    assert set(np.unique(y)).issubset({0.0, 1.0}), "y must be binary 0/1"
    assert len(names) == X.shape[1], (len(names), X.shape[1])
    if add_intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])
        names = ["intercept", *names]
    beta, cov, pl, conv, n_iter = _firth_newton(X, y)
    if not conv:
        raise RuntimeError(f"Firth fit did not converge in {n_iter} iterations")
    se = np.sqrt(np.diag(cov))
    res = FirthResult(names=list(names), beta=beta, se=se, pl=pl, converged=conv, n_iter=n_iter)
    p = X.shape[1]
    if profile_ci:
        lows = np.empty(p)
        highs = np.empty(p)
        m_low: list[str] = []
        m_high: list[str] = []
        for j in range(p):
            lows[j], highs[j], ml, mh = _profile_ci_one(
                X, y, j, float(beta[j]), float(se[j]), pl, name=names[j]
            )
            m_low.append(ml)
            m_high.append(mh)
        res.ci_low, res.ci_high = lows, highs
        res.ci_method_low, res.ci_method_high = m_low, m_high
    if plr_pvalues:
        pvals = np.empty(p)
        p_methods: list[str] = []
        for j in range(p):
            try:
                pl0 = _profile_pl_at(X, y, j, 0.0)
                stat = max(0.0, 2.0 * (pl - pl0))
                method = "plr"
            except ProfileNonConvergenceError as err:
                # Same non-estimability class as the CI bounds: the null-
                # constrained fit would not converge. Flagged Wald chi-square
                # p-value instead — equivalent two-sided z-test on beta/SE.
                stat = float((beta[j] / se[j]) ** 2)
                method = "wald-fallback"
                print(
                    f"[i591_firth] WARNING: PLR p-value for coef {j} "
                    f"({names[j]!r}) not estimable ({err}); Wald chi-square "
                    f"fallback used and flagged in the output JSON.",
                    file=sys.stderr,
                    flush=True,
                )
            pvals[j] = float(chi2.sf(stat, df=1))
            p_methods.append(method)
        res.p_values = pvals
        res.p_method = p_methods
    return res


def load_sex2(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the vendored sex2 CSV -> (X, y, names) with X excluding intercept."""
    rows = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    names = ["age", "oc", "vic", "vicl", "vis", "dia"]
    X = np.column_stack([rows[n] for n in names])
    y = rows["case"]
    assert X.shape == (239, 6), X.shape
    return X, y, names


def validate_against_sex2(csv_path: Path = SEX2_CSV_DEFAULT, coef_tol: float = 1e-4) -> dict:
    """Hard-validate the implementation against the published logistf::sex2 fit.

    Asserts (1) every coefficient matches the published reference to
    ``coef_tol``; (2) profile-CI self-consistency: the penalized-likelihood
    drop at each returned bound equals chi2(1, .95) to 1e-3 and the bounds
    bracket the estimate. Returns a JSON-friendly report dict on success.
    """
    X, y, names = load_sex2(csv_path)
    res = firth_logistic(X, y, names)
    report = {"reference": SEX2_REFERENCE_COEF, "fitted": res.to_dict(), "coef_tol": coef_tol}
    for j, name in enumerate(res.names):
        ref = SEX2_REFERENCE_COEF[name]
        got = float(res.beta[j])
        if abs(got - ref) > coef_tol:
            raise AssertionError(
                f"sex2 validation FAILED: coef[{name}] = {got:.8f}, published "
                f"logistf reference = {ref:.8f} (|diff| > {coef_tol})"
            )
    # The well-conditioned reference fit must never trip the #591-e5 fallback
    # path: every CI bound and p-value stays exact-profile/PLR. Pins the
    # rescue-ladder rung 0 == pre-fix behavior for converging fits.
    assert report["fitted"]["profile_fallback_coefs"] == [], report["fitted"][
        "profile_fallback_coefs"
    ]
    assert all(m == "profile" for m in report["fitted"]["ci95_method_low"])
    assert all(m == "profile" for m in report["fitted"]["ci95_method_high"])
    assert all(m == "plr" for m in report["fitted"]["p_method"])
    # CI self-consistency: PL drop at each bound == chi2 crit; bounds bracket.
    Xi = np.column_stack([np.ones(X.shape[0]), X])
    for j, name in enumerate(res.names):
        lo, hi = float(res.ci_low[j]), float(res.ci_high[j])
        assert lo < float(res.beta[j]) < hi, (name, lo, float(res.beta[j]), hi)
        for bound in (lo, hi):
            drop = 2.0 * (res.pl - _profile_pl_at(Xi, y, j, bound))
            if abs(drop - CHI2_95_DF1) > 1e-3:
                raise AssertionError(
                    f"profile-CI self-consistency FAILED for {name} at bound "
                    f"{bound:.6f}: PL drop {drop:.6f} != {CHI2_95_DF1:.6f}"
                )
    report["status"] = "PASS"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Firth-penalized logistic regression (validate with --validate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--validate", action="store_true", help="Run the sex2 validation.")
    parser.add_argument("--sex2", type=Path, default=SEX2_CSV_DEFAULT)
    parser.add_argument("--out", type=Path, default=None, help="Write the report JSON here.")
    args = parser.parse_args(argv)
    if not args.validate:
        parser.error("nothing to do — pass --validate")
    report = validate_against_sex2(args.sex2)
    fitted = report["fitted"]
    for name, coef, lo, hi, p in zip(
        fitted["names"],
        fitted["coef"],
        fitted["ci95_low_coef"],
        fitted["ci95_high_coef"],
        fitted["p_plr"],
        strict=True,
    ):
        print(f"{name:>10s}  coef={coef:+.6f}  ci95=[{lo:+.5f}, {hi:+.5f}]  p={p:.3g}")
    print("sex2 validation: PASS")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
