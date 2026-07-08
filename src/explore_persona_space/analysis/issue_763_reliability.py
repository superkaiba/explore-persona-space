# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, √, ≈, ×) in scientific docstrings.
"""Reliability ceiling √(r_yy) for the matched-probe predictor (issue #763).

The #742 reliability estimator (``scripts/issue742_reliability.py`` +
``explore_persona_space.analysis.issue_742_decoding_ceiling``) is NOT on
``main`` at #763 execution time (it lives only in the ``.claude/worktrees/
issue-742`` worktree — verified empty ``git ls-tree origin/main`` this
session). The plan's artifact-reuse fitness check (h) + Assumption 10 +
the brief's concern #4 therefore route #763 to the **rebuild branch**:
re-implement √(r_yy) split-half-over-probes + binomial-variance per the
plan §6 recipe, against the m≥50 matched-probe E0 this task produces. No
import from the stranded #742 worktree (a stranded import on the
git-clone-only GCP lane breaks at run-time, brief concern #4).

The reliability ceiling brackets what ANY decoder could reach against a
NOISY judged target (Spearman 1904 attenuation identity; Schoppe 2016
noise ceiling; Storrs 2020 CV-matched ceiling). For a per-context judged
rate E0(C,B) estimated over ``m`` probes:

- **split-half-over-probes** (the within-context test-retest): split the
  per-probe scores into two random halves, average each half to get two
  independent per-context rate estimates, Spearman-correlate the two
  half-vectors ACROSS contexts → ``r_hh``; Spearman-Brown step-up
  ``r_yy = 2·r_hh / (1 + r_hh)`` corrects the half-length back to the full
  m. ``√(r_yy)`` is the attenuation ceiling.

  **CROSSED designs need the probe-ALIGNED split (the default).** The #763
  E0 is fully crossed — the SAME m probes are judged under every context —
  so probe MAIN effects (per-probe difficulty, sd ≈ 27/100 for deception)
  are shared across contexts. Splitting each context's probes
  INDEPENDENTLY (the v1 estimator, kept as ``method="independent"``) puts
  a different probe subset in each context's half A, so the probe main
  effects leak into the split noise as ANTI-correlated half deviations
  (half A above the probe-mean forces half B below it), driving ``r_hh``
  systematically NEGATIVE whenever probe variance exceeds context-signal
  variance — which Spearman-Brown then clips to 0 (the #763
  deception-v1/-v2 incident: r_hh −0.41 / −0.23, ~100%/99% of 200 splits
  negative, ceiling reported 0). The ALIGNED split draws ONE probe-half
  assignment per split and applies it to EVERY context: both half-vectors
  carry the same probe set, the probe main effects become a constant
  per-half offset that cancels in the across-context correlation, and the
  ceiling estimates the context-signal test-retest it was meant to.
- **binomial-variance** (the parametric alternative): each context's
  observed rate has binomial sampling variance ``p(1-p)/m``; the ceiling
  is ``√(signal_var / (signal_var + mean_noise_var))`` — the fraction of
  cross-context variance that is real signal vs binomial sampling noise.

Both are CV-matched (computed on the same 50 contexts the predictor reads)
and cluster-bootstrapped over contexts. All 5 #763 behaviors are
``n_samples=1`` so the split is over PROBES (rollouts cannot be split).
"""

from __future__ import annotations

import math
import random

import numpy as np
from scipy.stats import spearmanr


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman ρ with a degeneracy guard (None on <4 points / no variance)."""
    if len(a) < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    r, _ = spearmanr(a, b)
    return None if np.isnan(r) else float(r)


def _spearman_brown(r_half: float) -> float:
    """Step up a half-length reliability ``r_half`` to the full length.

    ``r_full = 2·r_half / (1 + r_half)``. Clamped to [0, 1] (a negative
    half-correlation steps up to a meaningless negative; the floor is 0).
    """
    if r_half <= 0.0:
        return 0.0
    return min(1.0, (2.0 * r_half) / (1.0 + r_half))


def _n_valid(scores: list[float | None]) -> int:
    """Count the non-None (scoreable) probe entries in one context's list."""
    return sum(1 for v in scores if v is not None)


def reliability_split_half_over_probes(
    per_probe_by_ctx: dict[str, list[float | None]],
    *,
    n_splits: int = 200,
    seed: int = 763,
    method: str = "aligned",
) -> dict:
    """Split-half-over-probes reliability r_yy (Spearman-Brown corrected).

    Args:
        per_probe_by_ctx: ``{context_id: [per-probe E0 score, ...]}`` — each
            context's list is the per-probe judge scores (graded means, or the
            binary judge-positive fractions) the judge step produced. ``None``
            marks a probe with no scoreable judgment (dropped draws) and is
            skipped from that context's half means. A context needs ≥2
            non-None probes to split.
        n_splits: random half-splits averaged (reduces split variance).
        seed: RNG seed (the split draws are fully seed-pinned).
        method: ``"aligned"`` (DEFAULT — the crossed-design estimator: ONE
            probe-half assignment per split, shared by every context, so
            probe main effects cancel; requires equal-length positionally
            aligned lists — the same probe axis for every context) or
            ``"independent"`` (the legacy v1 estimator: each context's probes
            shuffled independently — kept ONLY for comparison/diagnostics; on
            a crossed design it is downward-biased and clips to 0 whenever
            probe variance exceeds context-signal variance, see module
            docstring).

    Returns:
        ``{"r_hh": float|None, "r_yy": float|None, "sqrt_r_yy": float|None,
           "n_contexts_used": int, "n_splits": int, "method": str}``.
        ``r_hh`` is the mean across-context Spearman of the two half-vectors
        over the splits (reported RAW — it can be negative; the Spearman-Brown
        step-up ``r_yy`` floors at 0); ``sqrt_r_yy`` is the ceiling. All None
        when too few contexts have ≥2 non-None probes (the no-dynamic-range /
        verdict-(c) case — reported, not crashed).

    Raises:
        ValueError: unknown ``method``, or ``method="aligned"`` on ragged
            (non-crossed) input — pass ``None`` placeholders for missing
            probes, or use ``method="independent"`` for genuinely
            non-crossed data.
    """
    if method not in ("aligned", "independent"):
        raise ValueError(f"unknown split-half method {method!r} (aligned | independent)")
    rng = random.Random(seed)
    ctx_ids = [c for c, scores in per_probe_by_ctx.items() if _n_valid(scores) >= 2]
    if len(ctx_ids) < 4:
        return {
            "r_hh": None,
            "r_yy": None,
            "sqrt_r_yy": None,
            "n_contexts_used": len(ctx_ids),
            "n_splits": 0,
            "method": method,
        }

    half_corrs: list[float] = []
    if method == "aligned":
        lengths = {len(per_probe_by_ctx[c]) for c in ctx_ids}
        if len(lengths) != 1:
            raise ValueError(
                "probe-aligned split-half requires a CROSSED design: every context must "
                "carry the SAME probe axis (equal-length, positionally aligned per-probe "
                f"lists; got lengths={sorted(lengths)}). Use None placeholders for missing "
                "probes, or method='independent' for genuinely non-crossed data."
            )
        m = lengths.pop()
        mat = np.array(
            [[np.nan if v is None else float(v) for v in per_probe_by_ctx[c]] for c in ctx_ids],
            dtype=np.float64,
        )  # (n_ctx, m); NaN = missing probe judgment
        valid = ~np.isnan(mat)
        filled = np.where(valid, mat, 0.0)
        for _ in range(n_splits):
            idx = rng.sample(range(m), m)  # ONE global probe permutation per split
            a_cols, b_cols = idx[: m // 2], idx[m // 2 :]
            # NaN-tolerant half means without nanmean's empty-slice warnings.
            cnt_a = valid[:, a_cols].sum(axis=1)
            cnt_b = valid[:, b_cols].sum(axis=1)
            a_vec = filled[:, a_cols].sum(axis=1) / np.maximum(cnt_a, 1)
            b_vec = filled[:, b_cols].sum(axis=1) / np.maximum(cnt_b, 1)
            ok = (cnt_a > 0) & (cnt_b > 0)
            r = _spearman(a_vec[ok], b_vec[ok])
            if r is not None:
                half_corrs.append(r)
    else:  # "independent" — the legacy v1 estimator (per-context shuffles)
        for _ in range(n_splits):
            a_vec_l: list[float] = []
            b_vec_l: list[float] = []
            for c in ctx_ids:
                scores = [v for v in per_probe_by_ctx[c] if v is not None]
                rng.shuffle(scores)
                mid = len(scores) // 2
                if mid == 0:
                    # 1-probe context (shouldn't reach here given >=2 filter) — skip.
                    continue
                a_vec_l.append(float(np.mean(scores[:mid])))
                b_vec_l.append(float(np.mean(scores[mid:])))
            r = _spearman(np.asarray(a_vec_l), np.asarray(b_vec_l))
            if r is not None:
                half_corrs.append(r)

    if not half_corrs:
        return {
            "r_hh": None,
            "r_yy": None,
            "sqrt_r_yy": None,
            "n_contexts_used": len(ctx_ids),
            "n_splits": n_splits,
            "method": method,
        }
    r_hh = float(np.mean(half_corrs))
    r_yy = _spearman_brown(r_hh)
    return {
        "r_hh": r_hh,
        "r_yy": r_yy,
        "sqrt_r_yy": math.sqrt(r_yy),
        "n_contexts_used": len(ctx_ids),
        "n_splits": n_splits,
        "method": method,
    }


def reliability_binomial_variance(
    rates: list[float],
    n_judged: list[int],
) -> dict:
    """Binomial-variance reliability ceiling (the parametric alternative).

    Each context's observed rate ``p`` has binomial sampling variance
    ``p(1-p)/m`` (m = that context's judged count). The signal variance is
    the cross-context variance of the observed rates MINUS the mean
    within-context sampling variance (a noise-corrected variance estimate);
    the reliability is ``signal_var / observed_var`` and the ceiling is its
    square root.

    Args:
        rates: per-context observed E0 rates (aligned with ``n_judged``).
        n_judged: per-context judged count m (the binomial n).

    Returns:
        ``{"r_yy": float|None, "sqrt_r_yy": float|None, "signal_var": float,
           "mean_noise_var": float, "observed_var": float, "n_contexts": int}``.
        None ceiling when observed variance is ~0 (no cross-context spread —
        the low-dynamic-range case) or signal variance is non-positive (noise
        dominates → ceiling floored at 0).
    """
    if len(rates) < 4:
        return {
            "r_yy": None,
            "sqrt_r_yy": None,
            "signal_var": 0.0,
            "mean_noise_var": 0.0,
            "observed_var": 0.0,
            "n_contexts": len(rates),
        }
    p = np.asarray(rates, dtype=np.float64)
    m = np.asarray(n_judged, dtype=np.float64)
    m = np.where(m < 1, 1.0, m)
    observed_var = float(np.var(p, ddof=1))
    noise_var = p * (1.0 - p) / m  # per-context binomial sampling variance
    mean_noise_var = float(np.mean(noise_var))
    signal_var = observed_var - mean_noise_var
    if observed_var < 1e-12:
        # No cross-context spread to decode — low dynamic range.
        return {
            "r_yy": None,
            "sqrt_r_yy": None,
            "signal_var": signal_var,
            "mean_noise_var": mean_noise_var,
            "observed_var": observed_var,
            "n_contexts": len(rates),
        }
    r_yy = max(0.0, signal_var / observed_var)
    r_yy = min(1.0, r_yy)
    return {
        "r_yy": r_yy,
        "sqrt_r_yy": math.sqrt(r_yy),
        "signal_var": signal_var,
        "mean_noise_var": mean_noise_var,
        "observed_var": observed_var,
        "n_contexts": len(rates),
    }


def compute_bracket(
    per_probe_by_ctx: dict[str, list[float | None]],
    rates: list[float],
    n_judged: list[int],
    *,
    n_boot: int = 2000,
    seed: int = 763,
    method: str = "aligned",
) -> dict:
    """The √(r_yy) ceiling bracket: split-half + binomial, cluster-bootstrapped.

    Returns both estimators' ``sqrt_r_yy`` (the 2-method agreement check the
    plan §6 names) plus a cluster-bootstrap-over-contexts 95% CI on the
    split-half ceiling (the headline ceiling). The bootstrap resamples
    CONTEXTS with replacement (the contexts are the statistical unit) and
    recomputes the split-half ceiling on each resample. ``method`` threads
    through the point estimate AND every bootstrap resample (``"aligned"``
    crossed-design default | ``"independent"`` legacy — see
    :func:`reliability_split_half_over_probes`). All draws are seed-pinned.

    Args:
        per_probe_by_ctx: ``{context_id: [per-probe score | None, ...]}``.
        rates: per-context observed rates aligned with ``n_judged`` (binomial).
        n_judged: per-context judged count.
        n_boot: bootstrap resamples for the ceiling CI.
        seed: RNG seed.
        method: split-half estimator variant (threaded to point + bootstrap).

    Returns:
        ``{"sqrt_r_yy_split_half", "sqrt_r_yy_binomial",
           "sqrt_r_yy", "sqrt_r_yy_ci": [lo, hi]|None,
           "split_half": {...}, "binomial": {...},
           "split_half_method": str, "seed": int}``. ``sqrt_r_yy`` is the
        split-half ceiling (the headline; binomial is the agreement check).
    """
    split = reliability_split_half_over_probes(per_probe_by_ctx, seed=seed, method=method)
    binom = reliability_binomial_variance(rates, n_judged)
    headline = split["sqrt_r_yy"]

    ci: list[float] | None = None
    ctx_ids = [c for c, scores in per_probe_by_ctx.items() if _n_valid(scores) >= 2]
    if headline is not None and len(ctx_ids) >= 4:
        rng = random.Random(seed + 1)
        draws: list[float] = []
        max_attempts = 5 * n_boot
        attempts = 0
        while len(draws) < n_boot and attempts < max_attempts:
            attempts += 1
            sampled = [ctx_ids[rng.randrange(len(ctx_ids))] for _ in range(len(ctx_ids))]
            sub = {f"{c}__{i}": per_probe_by_ctx[c] for i, c in enumerate(sampled)}
            r = reliability_split_half_over_probes(
                sub, n_splits=50, seed=seed + attempts, method=method
            )
            if r["sqrt_r_yy"] is not None:
                draws.append(r["sqrt_r_yy"])
        if len(draws) >= max(50, n_boot // 4):
            draws.sort()
            ci = [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]]

    return {
        "sqrt_r_yy_split_half": split["sqrt_r_yy"],
        "sqrt_r_yy_binomial": binom["sqrt_r_yy"],
        "sqrt_r_yy": headline,
        "sqrt_r_yy_ci": ci,
        "split_half": split,
        "binomial": binom,
        "split_half_method": method,
        "seed": seed,
    }
