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

- **split-half-over-probes** (the within-context test-retest): split each
  context's per-probe scores into two random halves, average each half to
  get two independent per-context rate estimates, Spearman-correlate the
  two half-vectors ACROSS contexts → ``r_hh``; Spearman-Brown step-up
  ``r_yy = 2·r_hh / (1 + r_hh)`` corrects the half-length back to the full
  m. ``√(r_yy)`` is the attenuation ceiling.
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


def reliability_split_half_over_probes(
    per_probe_by_ctx: dict[str, list[float]],
    *,
    n_splits: int = 200,
    seed: int = 763,
) -> dict:
    """Split-half-over-probes reliability r_yy (Spearman-Brown corrected).

    Args:
        per_probe_by_ctx: ``{context_id: [per-probe E0 score, ...]}`` — each
            context's list is the per-probe judge-positive fractions (or the
            structural is_list_formatted bits) the judge step produced. A
            context needs ≥2 probes to split.
        n_splits: random half-splits averaged (reduces split variance).
        seed: RNG seed.

    Returns:
        ``{"r_hh": float|None, "r_yy": float|None, "sqrt_r_yy": float|None,
           "n_contexts_used": int, "n_splits": int}``. ``r_hh`` is the mean
        across-context Spearman of the two half-vectors over the splits;
        ``r_yy`` is its Spearman-Brown step-up; ``sqrt_r_yy`` is the ceiling.
        All None when too few contexts have ≥2 probes (the no-dynamic-range
        / verdict-(c) case — reported, not crashed).
    """
    rng = random.Random(seed)
    ctx_ids = [c for c, scores in per_probe_by_ctx.items() if len(scores) >= 2]
    if len(ctx_ids) < 4:
        return {
            "r_hh": None,
            "r_yy": None,
            "sqrt_r_yy": None,
            "n_contexts_used": len(ctx_ids),
            "n_splits": 0,
        }
    half_corrs: list[float] = []
    for _ in range(n_splits):
        a_vec: list[float] = []
        b_vec: list[float] = []
        for c in ctx_ids:
            scores = list(per_probe_by_ctx[c])
            rng.shuffle(scores)
            mid = len(scores) // 2
            if mid == 0:
                # 1-probe context (shouldn't reach here given >=2 filter) — skip.
                continue
            a_vec.append(float(np.mean(scores[:mid])))
            b_vec.append(float(np.mean(scores[mid:])))
        r = _spearman(np.asarray(a_vec), np.asarray(b_vec))
        if r is not None:
            half_corrs.append(r)
    if not half_corrs:
        return {
            "r_hh": None,
            "r_yy": None,
            "sqrt_r_yy": None,
            "n_contexts_used": len(ctx_ids),
            "n_splits": n_splits,
        }
    r_hh = float(np.mean(half_corrs))
    r_yy = _spearman_brown(r_hh)
    return {
        "r_hh": r_hh,
        "r_yy": r_yy,
        "sqrt_r_yy": math.sqrt(r_yy),
        "n_contexts_used": len(ctx_ids),
        "n_splits": n_splits,
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
    per_probe_by_ctx: dict[str, list[float]],
    rates: list[float],
    n_judged: list[int],
    *,
    n_boot: int = 2000,
    seed: int = 763,
) -> dict:
    """The √(r_yy) ceiling bracket: split-half + binomial, cluster-bootstrapped.

    Returns both estimators' ``sqrt_r_yy`` (the 2-method agreement check the
    plan §6 names) plus a cluster-bootstrap-over-contexts 95% CI on the
    split-half ceiling (the headline ceiling). The bootstrap resamples
    CONTEXTS with replacement (the contexts are the statistical unit) and
    recomputes the split-half ceiling on each resample.

    Args:
        per_probe_by_ctx: ``{context_id: [per-probe score, ...]}``.
        rates: per-context observed rates aligned with ``n_judged`` (binomial).
        n_judged: per-context judged count.
        n_boot: bootstrap resamples for the ceiling CI.
        seed: RNG seed.

    Returns:
        ``{"sqrt_r_yy_split_half", "sqrt_r_yy_binomial",
           "sqrt_r_yy", "sqrt_r_yy_ci": [lo, hi]|None,
           "split_half": {...}, "binomial": {...}}``. ``sqrt_r_yy`` is the
        split-half ceiling (the headline; binomial is the agreement check).
    """
    split = reliability_split_half_over_probes(per_probe_by_ctx, seed=seed)
    binom = reliability_binomial_variance(rates, n_judged)
    headline = split["sqrt_r_yy"]

    ci: list[float] | None = None
    ctx_ids = [c for c, scores in per_probe_by_ctx.items() if len(scores) >= 2]
    if headline is not None and len(ctx_ids) >= 4:
        rng = random.Random(seed + 1)
        draws: list[float] = []
        max_attempts = 5 * n_boot
        attempts = 0
        while len(draws) < n_boot and attempts < max_attempts:
            attempts += 1
            sampled = [ctx_ids[rng.randrange(len(ctx_ids))] for _ in range(len(ctx_ids))]
            sub = {f"{c}__{i}": per_probe_by_ctx[c] for i, c in enumerate(sampled)}
            r = reliability_split_half_over_probes(sub, n_splits=50, seed=seed + attempts)
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
    }
