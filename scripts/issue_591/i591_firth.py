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
        if self.p_values is not None:
            out["p_plr"] = self.p_values.tolist()
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


def _profile_pl_at(X: np.ndarray, y: np.ndarray, j: int, value: float) -> float:
    """Maximized penalized log-likelihood with beta_j pinned to ``value``."""
    _beta, _cov, pl, conv, _ = _firth_newton(X, y, fixed={j: value})
    if not conv:
        raise RuntimeError(f"profile fit did not converge (coef {j} pinned to {value:.4f})")
    return pl


def _profile_ci_one(
    X: np.ndarray,
    y: np.ndarray,
    j: int,
    beta_hat: float,
    se_j: float,
    pl_hat: float,
    *,
    chi2_crit: float = CHI2_95_DF1,
) -> tuple[float, float]:
    """Invert the penalized likelihood ratio for one coefficient (both sides)."""

    def g(b: float) -> float:
        return 2.0 * (pl_hat - _profile_pl_at(X, y, j, b)) - chi2_crit

    bounds: list[float] = []
    for direction in (-1.0, +1.0):
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
            raise RuntimeError(
                f"profile CI bracket failed for coef {j} (direction {direction:+.0f}); "
                f"penalized likelihood too flat — inspect the design matrix."
            )
        a, b = (hi, lo) if direction < 0 else (lo, hi)
        bound = float(brentq(g, a, b, xtol=1e-6))
        bounds.append(bound)
    return bounds[0], bounds[1]


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
        for j in range(p):
            lows[j], highs[j] = _profile_ci_one(X, y, j, float(beta[j]), float(se[j]), pl)
        res.ci_low, res.ci_high = lows, highs
    if plr_pvalues:
        pvals = np.empty(p)
        for j in range(p):
            pl0 = _profile_pl_at(X, y, j, 0.0)
            stat = max(0.0, 2.0 * (pl - pl0))
            pvals[j] = float(chi2.sf(stat, df=1))
        res.p_values = pvals
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
