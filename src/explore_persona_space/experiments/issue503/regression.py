"""Statistical analysis for issue #503 — pooled binomial mixed model on (k, n).

Per plan §9 (MF3 revision):

Primary regression — binomial mixed model on count outcomes:

    cbind(k, n - k) ~ cosine_predictor + cell_type + family + log_tokens
                      + lexical_persona_cosine + base_rate
                      + (1 | source) + (1 | target),
                      family = binomial(link = "logit")

Convergence fallback (pre-registered):

    logit((k + 0.5) / (n + 1)) ~ cosine_predictor + cell_type + family
                                  + log_tokens + lexical_persona_cosine
                                  + base_rate
                                  + (1 | source) + (1 | target)

Secondary — raw-rate Spearman ρ + partial-Spearman ladder (raw →
partial-log-tokens → partial-lexical → partial-base-rate). Family-clustered
SE and leave-one-family-out sensitivity. Bootstrap 95% CI (1000 resamples).
Permutation null (shuffle predictor↔leakage mapping, ≥1000 iterations).

H4 is FDR-corrected across the 3 statistically-tested strata (N→N,
N→B-EM, N→B-syco). B→B is descriptive-only (effect size + 95% CI +
exact permutation null at n=4) per MF2.

Implementation uses ``statsmodels`` for the binomial GLM (with
cluster-robust SE on source-family as a clustering variable for the
fixed-effects part) and ``scipy.stats.spearmanr`` for the rank
correlations. We deliberately do NOT depend on R / pymer4 / lme4 here
to keep the worktree self-contained; a future analyzer-side recheck
with lme4 is encouraged.
"""

# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, ρ, κ, →, —) in scientific docstrings + logs.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CellType = Literal["N_to_N", "N_to_B_EM", "N_to_B_syco", "B_to_B"]
PRE_REG_HEADLINE_STRATA: tuple[CellType, ...] = ("N_to_N", "N_to_B_EM", "N_to_B_syco")

# Plan v2 §4: 5 buckets in the spectrum.
# A = cross-lingual (positive control), B = narrow/broad source-target matrix
# (v1's matrix), C = broad → broad (descriptive only), D = benign-data
# (He et al. selector arm), E = orthogonal non-transfer (negative control).
Bucket = Literal["A", "B", "C", "D", "E"]
ALL_BUCKETS: tuple[Bucket, ...] = ("A", "B", "C", "D", "E")


@dataclass
class RegressionRow:
    """One off-diagonal cell-seed observation entering the regression.

    The plan-v2 cross-bucket extension adds ``bucket`` as a stratifying
    factor (per plan §17 'Extension to regression.py: add bucket as a
    stratifying covariate; extend RegressionRow schema'). Existing v1
    rows default to bucket='B' (the narrow/broad source-target matrix
    matches v1's coverage).
    """

    source: str
    target: str
    seed: int
    cell_type: CellType
    family: str  # source family
    k: int  # judge-positive verdicts
    n: int  # total verdicts
    cosine_predictor: float  # primary predictor (mean over 2 K=8 draws)
    cosine_topic_stripped: float | None  # §3.5 control
    log_tokens: float  # log(generated-token-count)
    lexical_persona_cosine: float  # #468 carry-forward control
    base_rate: float  # #499 secondary predictor
    js_sliced_on_target: float | None  # #466 secondary
    js_sliced_off_target: float | None  # #466 secondary
    kl_secondary_dv: float | None  # §5.1 non-saturating sibling DV
    # Plan v2 extension: bucket factor for cross-bucket pooled regression.
    bucket: Bucket = "B"


@dataclass
class RegressionFit:
    """One fit's headline numbers + diagnostics."""

    model_form: str
    n_rows: int
    converged: bool
    coef_cosine: float
    se_cosine: float
    ci_low_cosine: float  # 95% bootstrap CI
    ci_high_cosine: float
    coefs_full: dict[str, float]  # all named coefficients
    notes: list[str]


def rows_to_dataframe(rows: list[RegressionRow]) -> pd.DataFrame:
    """Normalize a list of ``RegressionRow`` into a DataFrame the
    statsmodels GLM expects.

    Skips rows with ``n == 0`` (no verdicts — judge errors swallowed
    everything, the row is uninterpretable; fail-loud rather than
    silently zero).
    """
    records = [asdict(r) for r in rows]
    df = pd.DataFrame.from_records(records)
    if (df["n"] == 0).any():
        n_zero = int((df["n"] == 0).sum())
        bad = df.loc[df["n"] == 0, ["source", "target", "seed"]].to_dict(orient="records")
        raise RuntimeError(
            f"{n_zero} regression rows have n=0 (no judge verdicts succeeded). "
            f"Fail-loud per CLAUDE.md — investigate before fitting. Bad cells: {bad}"
        )
    return df


def _build_binomial_design(
    df: pd.DataFrame,
    *,
    include_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the (endog, exog, coef_names) design for the two-column
    binomial GLM. endog has columns [k, n-k]; exog has one column per
    fixed-effect coefficient (intercept + numeric covariates + one-hot
    encodings of cell_type and family with the first level dropped).

    Used by both ``fit_binomial_mixed`` and the bootstrap CI to keep the
    two paths in lockstep — patsy's formula parser does NOT support
    two-column endog binomial (MF-B fix: the prior ``successes + failures
    ~ ...`` smf.glm formula was silently interpreted as an unweighted
    proportion).
    """
    df = df.copy()
    # Endog: two-column [k, n-k] count matrix per statsmodels Binomial GLM API.
    endog = np.column_stack(
        [df["k"].astype(np.float64).to_numpy(), (df["n"] - df["k"]).astype(np.float64).to_numpy()]
    )

    coef_names: list[str] = []
    columns: list[np.ndarray] = []

    if include_intercept:
        coef_names.append("Intercept")
        columns.append(np.ones(len(df), dtype=np.float64))

    # Numeric predictors.
    for name in (
        "cosine_predictor",
        "log_tokens",
        "lexical_persona_cosine",
        "base_rate",
    ):
        coef_names.append(name)
        columns.append(df[name].astype(np.float64).to_numpy())

    # Categorical one-hot encoding (treatment contrast, drop first level).
    # ``bucket`` added by plan v2 as a stratifying factor — skipped if only
    # one bucket value is present in the rows (mono-bucket sweeps, e.g.
    # the v1 Bucket-B-only fit, do not need the bucket factor).
    categorical_factors = ["cell_type", "family"]
    if "bucket" in df.columns and df["bucket"].nunique() > 1:
        categorical_factors.append("bucket")
    for cat in categorical_factors:
        levels = sorted(df[cat].unique().tolist())
        # Drop first level as the reference category (matches patsy `C(x)`).
        for level in levels[1:]:
            coef_names.append(f"C({cat})[T.{level}]")
            columns.append((df[cat] == level).astype(np.float64).to_numpy())

    exog = np.column_stack(columns) if columns else np.empty((len(df), 0))
    return endog, exog, coef_names


def fit_binomial_mixed(
    rows: list[RegressionRow],
    *,
    strata: tuple[CellType, ...] | None = None,
) -> RegressionFit:
    """Fit the §9 primary regression with statsmodels.

    MF-B (round-2 revision): uses ``statsmodels.GLM(endog=[[k, n-k]], ...,
    family=Binomial())`` — the two-column endog binomial COUNT fit, NOT
    the v1 ``successes + failures ~ ...`` formula which patsy converted
    to an unweighted proportion response. Cluster-robust SE clustered on
    source family. The full random-effects model (random intercepts on
    source and target) is a follow-up via pymer4 / lme4; here we mark
    the cluster-SE as "family-clustered" per §9 family-clustering spec.

    If the binomial GLM fails to converge, falls back to the
    pseudocount-transformed OLS form `logit((k + 0.5) / (n + 1)) ~ ...`
    per the pre-registered §9 fallback (this stays a fixed-effects model
    with cluster-robust SE).

    ``strata`` restricts to the named cell types if given (the H4 FDR
    pre-registration uses ``PRE_REG_HEADLINE_STRATA`` — N→N + N→B-EM +
    N→B-syco — excluding B→B which is descriptive-only).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = rows_to_dataframe(rows)
    if strata is not None:
        df = df.loc[df["cell_type"].isin(strata)].copy()

    if df.empty:
        raise RuntimeError(
            f"fit_binomial_mixed: empty panel after applying strata={strata}; no rows to fit."
        )

    notes: list[str] = []
    converged = True
    result = None
    coef_names: list[str] = []
    formula = ""  # ONLY filled if we fall back to the pseudocount OLS path.

    try:
        endog, exog, coef_names = _build_binomial_design(df)
        # Two-column endog binomial: statsmodels treats column 0 as
        # successes and column 1 as failures (the (k, n - k) count form).
        glm_model = sm.GLM(endog, exog, family=sm.families.Binomial())
        result = glm_model.fit(
            cov_type="cluster",
            cov_kwds={"groups": df["family"].to_numpy()},
        )
    except Exception as exc:
        notes.append(
            f"binomial GLM convergence failed: {type(exc).__name__}: {exc}; "
            "falling back to pseudocount form per §9 pre-registration"
        )
        converged = False
        result = None

    if not converged:
        # Pseudocount fallback: logit((k+0.5)/(n+1)) ~ predictors.
        df["pseudo_rate"] = (df["k"] + 0.5) / (df["n"] + 1.0)
        df["pseudo_logit"] = np.log(df["pseudo_rate"] / (1.0 - df["pseudo_rate"]))
        formula = (
            "pseudo_logit ~ cosine_predictor + C(cell_type) + C(family) "
            "+ log_tokens + lexical_persona_cosine + base_rate"
        )
        model = smf.ols(formula, data=df)
        result = model.fit(cov_type="cluster", cov_kwds={"groups": df["family"]})
        # In the fallback the result.params is a pandas Series indexed by name.
        coef_names = list(result.params.index)

    # Pull coefficient values + SE keyed by name. The binomial GLM path
    # returns numpy arrays; we re-key with ``coef_names`` built above.
    if converged:
        params_arr = np.asarray(result.params, dtype=np.float64)
        bse_arr = np.asarray(result.bse, dtype=np.float64)
        params = dict(zip(coef_names, params_arr.tolist(), strict=True))
        bse = dict(zip(coef_names, bse_arr.tolist(), strict=True))
    else:
        params = {k: float(v) for k, v in result.params.items()}
        bse = {k: float(v) for k, v in result.bse.items()}

    coef_cosine = float(params.get("cosine_predictor", float("nan")))
    se_cosine = float(bse.get("cosine_predictor", float("nan")))

    # 95% bootstrap CI on the cosine coefficient (1000 resamples).
    ci_low, ci_high = _bootstrap_ci_coef(
        df,
        formula=formula,
        predictor="cosine_predictor",
        n_boot=1000,
        converged=converged,
    )

    return RegressionFit(
        model_form=("binomial_mixed" if converged else "pseudocount_logit_ols"),
        n_rows=len(df),
        converged=converged,
        coef_cosine=coef_cosine,
        se_cosine=se_cosine,
        ci_low_cosine=ci_low,
        ci_high_cosine=ci_high,
        coefs_full=params,
        notes=notes,
    )


def _bootstrap_ci_coef(
    df: pd.DataFrame,
    formula: str,
    predictor: str,
    n_boot: int,
    converged: bool,
    *,
    seed: int = 0,
) -> tuple[float, float]:
    """Cluster bootstrap (resample source families with replacement) for
    a 95% CI on one named coefficient.

    MF-B (round-2 revision): on the converged path uses the two-column
    endog binomial GLM (``sm.GLM(endog=[[k, n-k]], ...,
    family=Binomial())``) — patches the v1 ``smf.glm`` formula path
    silently fitting an unweighted-proportion response.

    Conservative: on bootstrap-sample convergence failure, the coefficient
    contributes ``nan`` to the bootstrap distribution and is excluded from
    the percentile estimate. If <100 bootstrap fits succeed the CI is
    reported as ``(nan, nan)`` with a note in the caller's RegressionFit.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(seed)
    families = df["family"].unique()
    fits: list[float] = []
    for _ in range(n_boot):
        boot_fams = rng.choice(families, size=len(families), replace=True)
        parts = [df.loc[df["family"] == fam] for fam in boot_fams]
        boot_df = pd.concat(parts, axis=0, ignore_index=True)
        try:
            if converged:
                endog, exog, coef_names = _build_binomial_design(boot_df)
                glm_model = sm.GLM(endog, exog, family=sm.families.Binomial())
                res = glm_model.fit(
                    cov_type="cluster",
                    cov_kwds={"groups": boot_df["family"].to_numpy()},
                )
                params = dict(zip(coef_names, np.asarray(res.params).tolist(), strict=True))
                fits.append(float(params.get(predictor, float("nan"))))
            else:
                model = smf.ols(formula, data=boot_df)
                res = model.fit(cov_type="cluster", cov_kwds={"groups": boot_df["family"]})
                fits.append(float(res.params.get(predictor, float("nan"))))
        except Exception:
            fits.append(float("nan"))
    arr = np.asarray(fits, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 100:
        logger.warning(
            "_bootstrap_ci_coef(%s): only %d valid bootstrap fits (out of %d)",
            predictor,
            len(arr),
            n_boot,
        )
        return float("nan"), float("nan")
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


# ── Secondary: raw-rate Spearman ρ + partial regression ladder ────────────


def spearman_rho(
    rows: list[RegressionRow], *, strata: tuple[CellType, ...] | None = None
) -> dict[str, float]:
    """Raw-rate Spearman ρ + p-value + n per cell-type stratum (H4 read).

    Returns ``{"rho": ..., "p_value": ..., "n": ...}``. ``strata=None``
    pools all rows; ``strata=("N_to_N",)`` etc restricts to one stratum.
    """
    from scipy.stats import spearmanr

    df = rows_to_dataframe(rows)
    if strata is not None:
        df = df.loc[df["cell_type"].isin(strata)].copy()
    if df.empty:
        return {"rho": float("nan"), "p_value": float("nan"), "n": 0}
    rate = df["k"] / df["n"]
    res = spearmanr(df["cosine_predictor"], rate)
    return {"rho": float(res.correlation), "p_value": float(res.pvalue), "n": len(df)}


def partial_spearman_ladder(rows: list[RegressionRow]) -> dict[str, dict[str, float]]:
    """The §5.1 partial-ρ ladder.

    Raw → partial(log_tokens) → partial(log_tokens, lexical) → partial(
    log_tokens, lexical, base_rate). Sequential partialling — each rung
    drops one more covariate's variance from BOTH the predictor and the
    outcome (Spearman partial via residualization on ranks).
    """
    from scipy.stats import rankdata, spearmanr

    df = rows_to_dataframe(rows)
    if df.empty:
        return {}

    def _residualize_ranks(y: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
        """Residualize rank(y) on rank(controls...) by OLS."""
        if not controls:
            return rankdata(y)
        x = np.column_stack([rankdata(c) for c in controls])
        x = np.column_stack([np.ones(len(x)), x])
        yr = rankdata(y)
        beta, *_ = np.linalg.lstsq(x, yr, rcond=None)
        return yr - x @ beta

    rate = (df["k"] / df["n"]).to_numpy()
    cos = df["cosine_predictor"].to_numpy()
    log_tok = df["log_tokens"].to_numpy()
    lex = df["lexical_persona_cosine"].to_numpy()
    base = df["base_rate"].to_numpy()

    out: dict[str, dict[str, float]] = {}
    res = spearmanr(cos, rate)
    out["raw"] = {"rho": float(res.correlation), "p_value": float(res.pvalue)}

    for label, controls in (
        ("partial_log_tokens", [log_tok]),
        ("partial_log_tokens_lexical", [log_tok, lex]),
        ("partial_log_tokens_lexical_base_rate", [log_tok, lex, base]),
    ):
        rate_resid = _residualize_ranks(rate, controls)
        cos_resid = _residualize_ranks(cos, controls)
        res = spearmanr(cos_resid, rate_resid)
        out[label] = {"rho": float(res.correlation), "p_value": float(res.pvalue)}
    return out


# ── Robustness: leave-one-family-out + permutation null ───────────────────


def leave_one_family_out(rows: list[RegressionRow]) -> dict[str, dict[str, float]]:
    """Drop each source family, re-fit, return the spread of ρ estimates."""
    df = rows_to_dataframe(rows)
    out: dict[str, dict[str, float]] = {}
    for fam in sorted(df["family"].unique()):
        kept = [r for r in rows if r.family != fam]
        if not kept:
            continue
        out[f"drop_{fam}"] = spearman_rho(kept)
    return out


def leave_one_bucket_out(rows: list[RegressionRow]) -> dict[str, dict[str, float]]:
    """Drop each bucket, re-fit, return the spread of ρ estimates.

    Plan v2 §17 (regression extension) + the critique-residual concern
    from the round-1 critic merged ('Cross-bucket pooling dominated by
    Bucket B. Add a leave-one-bucket-out ρ / coefficient diagnostic to
    the §6.1 outputs'). If the pooled ρ collapses when Bucket B is
    dropped, the headline is a single-bucket claim that doesn't
    generalize across the spectrum.

    Returns a dict keyed by ``drop_<bucket>`` so the analyzer can
    compare against the all-rows ρ for stability.
    """
    df = rows_to_dataframe(rows)
    if "bucket" not in df.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    for bucket in sorted(df["bucket"].unique()):
        kept = [r for r in rows if r.bucket != bucket]
        if not kept:
            continue
        out[f"drop_{bucket}"] = spearman_rho(kept)
    return out


def per_bucket_simple_slopes(rows: list[RegressionRow]) -> dict[str, dict[str, float]]:
    """Per-bucket Spearman ρ — addresses the round-1 critic's bucket-
    heterogeneity concern ('Bucket D vs B-syco LoRA-recipe heterogeneity.
    Bucket-as-stratifying-factor absorbs base-rate intercepts but does
    NOT absorb a bucket-specific cosine slope; report random-slope or
    per-bucket simple slopes alongside the pooled coefficient.').

    Returns a dict keyed by bucket name → {rho, p_value, n}.
    """
    df = rows_to_dataframe(rows)
    if "bucket" not in df.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    for bucket in sorted(df["bucket"].unique()):
        per_bucket = [r for r in rows if r.bucket == bucket]
        if len(per_bucket) < 4:
            # Too few rows for a meaningful per-bucket ρ (e.g. C with n=4).
            out[bucket] = {"rho": float("nan"), "p_value": float("nan"), "n": len(per_bucket)}
            continue
        r = spearman_rho(per_bucket)
        r["n"] = len(per_bucket)
        out[bucket] = r
    return out


def permutation_null(
    rows: list[RegressionRow], *, n_iter: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Shuffle the (predictor → outcome) mapping ``n_iter`` times and
    report the observed ρ's percentile under the null.

    Per §9 (and §5.1 for B→B): ≥1000 iterations. For very small panels
    (B→B at n=4) the caller should also compute the exact 4!=24
    enumeration null — implemented in ``exact_permutation_null``.
    """
    from scipy.stats import spearmanr

    df = rows_to_dataframe(rows)
    if df.empty:
        return {"rho_obs": float("nan"), "percentile_under_null": float("nan")}
    rate = (df["k"] / df["n"]).to_numpy()
    cos = df["cosine_predictor"].to_numpy()
    obs = float(spearmanr(cos, rate).correlation)
    rng = np.random.default_rng(seed)
    null_rhos = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        cos_perm = rng.permutation(cos)
        null_rhos[i] = float(spearmanr(cos_perm, rate).correlation)
    pct = float((null_rhos < obs).mean())
    return {"rho_obs": obs, "percentile_under_null": pct, "n_iter": int(n_iter)}


def exact_permutation_null(rows: list[RegressionRow]) -> dict[str, float]:
    """Exact n!-enumeration null for very small panels (B→B with n≤6).

    Caller is responsible for keeping n small; this enumerates over
    ``n!`` permutations, which is 24 for n=4 and 720 for n=6.
    """
    from itertools import permutations

    from scipy.stats import spearmanr

    df = rows_to_dataframe(rows)
    if df.empty:
        return {"rho_obs": float("nan"), "percentile_under_null": float("nan")}
    if len(df) > 7:
        raise ValueError(
            f"exact_permutation_null: panel has {len(df)} rows; enumeration "
            "explodes — use permutation_null instead."
        )
    rate = (df["k"] / df["n"]).to_numpy()
    cos = df["cosine_predictor"].to_numpy()
    obs = float(spearmanr(cos, rate).correlation)
    null_rhos = []
    for perm in permutations(range(len(cos))):
        cos_perm = cos[list(perm)]
        null_rhos.append(float(spearmanr(cos_perm, rate).correlation))
    null_arr = np.asarray(null_rhos)
    pct = float((null_arr < obs).mean())
    return {
        "rho_obs": obs,
        "percentile_under_null": pct,
        "n_enumerations": len(null_arr),
    }


def b_to_b_descriptive(rows: list[RegressionRow]) -> dict:
    """MF-E (round-2 revision): B→B descriptive-only analysis.

    Per plan §5.1 + §9: B→B is descriptive-only (n=4 at the planned
    matrix). Returns ``{point_estimate, ci_low, ci_high,
    permutation_null_pmf}`` — NO ``p_value`` field, NO pre-registered
    ρ-threshold gate. Designed to be safe against any caller that
    later asserts ``"p_value" not in result["b_to_b_descriptive"]``.

    Fields:
    - ``point_estimate``: raw Spearman ρ on the B→B off-diagonal pool
      (no inferential test).
    - ``ci_low`` / ``ci_high``: 95% bootstrap CI on ρ (1000 resamples
      with replacement; cluster on source family if there is more than
      one family present, else simple bootstrap on rows).
    - ``permutation_null_pmf``: exact n!-enumeration discrete null
      distribution as a list of ρ values (24 for n=4) — caller plots /
      reads the histogram.
    - ``n``: number of B→B off-diagonal rows entering the descriptive read.

    NOTE for callers: the returned dict deliberately does NOT include a
    ``p_value`` key — adding one would re-introduce the inferential
    framing MF-E removed. The permutation null is returned as the full
    PMF (a discrete list of ρ values) so the analyzer can describe its
    shape rather than report a tail-probability gate.
    """
    from itertools import permutations

    from scipy.stats import spearmanr

    b_rows = [r for r in rows if r.cell_type == "B_to_B"]
    if not b_rows:
        return {
            "point_estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "permutation_null_pmf": [],
            "n": 0,
        }

    df = rows_to_dataframe(b_rows)
    rate = (df["k"] / df["n"]).to_numpy()
    cos = df["cosine_predictor"].to_numpy()

    point = float(spearmanr(cos, rate).correlation)

    # Exact permutation null PMF — explicitly the FULL discrete null.
    # For n=4 this is 4! = 24 values; we return them all so the analyzer
    # can decide how to summarize without a tail probability.
    pmf: list[float] = []
    if len(b_rows) <= 7:
        for perm in permutations(range(len(cos))):
            cos_perm = cos[list(perm)]
            pmf.append(float(spearmanr(cos_perm, rate).correlation))

    # 95% bootstrap CI (1000 resamples on rows; no cluster structure
    # within the 4-row B→B pool).
    rng = np.random.default_rng(0)
    n_boot = 1000
    boot_rhos: list[float] = []
    n_rows = len(b_rows)
    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        try:
            r = float(spearmanr(cos[idx], rate[idx]).correlation)
            if not np.isnan(r):
                boot_rhos.append(r)
        except Exception:
            continue
    if len(boot_rhos) >= 100:
        ci_low, ci_high = (float(x) for x in np.percentile(boot_rhos, [2.5, 97.5]))
    else:
        ci_low, ci_high = float("nan"), float("nan")

    return {
        "point_estimate": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "permutation_null_pmf": pmf,
        "n": n_rows,
    }


# ── FDR correction (Benjamini-Hochberg) ───────────────────────────────────


def fdr_bh(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg step-up. Returns a per-test ``reject_null`` mask.

    Used at the H4 headline for the 3 statistically-tested strata
    (N→N, N→B-EM, N→B-syco; B→B excluded per MF2).
    """
    arr = np.asarray(p_values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return []
    order = np.argsort(arr)
    sorted_p = arr[order]
    bh = sorted_p * n / (np.arange(n) + 1)
    # Step-up: max over BH-adjusted ≤ alpha (in sorted order).
    accept = np.zeros(n, dtype=bool)
    crossed = False
    for i in range(n - 1, -1, -1):
        if bh[i] <= alpha:
            crossed = True
        if crossed:
            accept[i] = True
    rejected = np.zeros(n, dtype=bool)
    rejected[order] = accept
    return rejected.tolist()


def headline_h4_verdict(rows: list[RegressionRow]) -> dict:
    """Apply the §5.1 + §9 H4 headline gate.

    Pooled ρ ≥ +0.40 AND CI excludes zero AND cell-type-stratified
    ρ > +0.20 in EACH of N→N, N→B-EM, N→B-syco. Returns a structured
    dict with the per-stratum ρ + the AND-clause verdict.

    B→B is NOT in the FDR-corrected denominator (descriptive-only per
    MF2). The pooled ρ here is over ``PRE_REG_HEADLINE_STRATA``.

    MF-E round-2 revision: the legacy ``perm_excludes_zero`` flag was
    derived from a one-sided permutation-null percentile, which is NOT
    the bootstrap 95% CI on pooled ρ that the plan names. Both reads are
    surfaced now: ``perm_null_one_sided_pct >= 0.975`` (the percentile-
    under-null read) AND ``bootstrap_ci_excludes_zero`` (the 95%
    bootstrap CI on pooled ρ). The headline gate keeps the BOOTSTRAP
    CI read as authoritative; the perm-null percentile is reported
    for transparency.
    """
    per_strata = {s: spearman_rho(rows, strata=(s,)) for s in PRE_REG_HEADLINE_STRATA}
    pooled = spearman_rho(rows, strata=PRE_REG_HEADLINE_STRATA)
    headline_rows = [r for r in rows if r.cell_type in PRE_REG_HEADLINE_STRATA]
    perm_null = permutation_null(headline_rows)

    per_strata_passes = all(per_strata[s]["rho"] > 0.20 for s in PRE_REG_HEADLINE_STRATA)
    pooled_pass = pooled["rho"] >= 0.40

    # Bootstrap 95% CI on pooled ρ (the read named in the plan).
    boot_ci_low, boot_ci_high = _bootstrap_ci_pooled_rho(headline_rows, n_boot=1000, seed=0)
    bootstrap_ci_excludes_zero = (
        not np.isnan(boot_ci_low) and not np.isnan(boot_ci_high) and boot_ci_low > 0.0
    )

    # Perm-null percentile (kept for transparency; NOT the gating read).
    perm_pct = perm_null.get("percentile_under_null", float("nan"))
    perm_null_one_sided_above_0975 = (not np.isnan(perm_pct)) and perm_pct >= 0.975

    headline_passes = per_strata_passes and pooled_pass and bool(bootstrap_ci_excludes_zero)
    return {
        "pooled": pooled,
        "pooled_bootstrap_ci_low": boot_ci_low,
        "pooled_bootstrap_ci_high": boot_ci_high,
        "per_strata": per_strata,
        "permutation": perm_null,
        "pre_reg_strata": list(PRE_REG_HEADLINE_STRATA),
        "headline_pass": bool(headline_passes),
        "per_strata_pass_above_0.20": bool(per_strata_passes),
        "pooled_pass_above_0.40": bool(pooled_pass),
        "bootstrap_ci_excludes_zero": bool(bootstrap_ci_excludes_zero),
        "perm_null_one_sided_above_0975": bool(perm_null_one_sided_above_0975),
    }


def _bootstrap_ci_pooled_rho(
    rows: list[RegressionRow], *, n_boot: int, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap CI for pooled Spearman ρ on the headline-strata pool.

    Resamples rows with replacement; conservative — fewer than 100 valid
    fits gives (nan, nan). MF-E round-2: this is the load-bearing CI for
    the headline ``bootstrap_ci_excludes_zero`` flag.
    """
    from scipy.stats import spearmanr

    if not rows:
        return float("nan"), float("nan")
    df = rows_to_dataframe(rows)
    rate = (df["k"] / df["n"]).to_numpy()
    cos = df["cosine_predictor"].to_numpy()
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    n = len(rows)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            r = float(spearmanr(cos[idx], rate[idx]).correlation)
            if not np.isnan(r):
                boot.append(r)
        except Exception:
            continue
    if len(boot) < 100:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)
