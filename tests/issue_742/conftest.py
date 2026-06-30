# ruff: noqa: RUF003
"""Shared fixtures + TDD red-phase import guard for issue #742 tests.

Issue #742 — Decoding-ceiling, linear-information-loss, and sample-complexity
of #658 base-model behavior representations (n=50). Plan v7 §13 (TDD).

TDD CONTRACT (round 1 = tests-first):
  The implementation lands in round 2 at the canonical module path
  ``src/explore_persona_space/analysis/issue_742_decoding_ceiling.py``.
  Until then, every test that calls an implementation symbol is SKIPPED via
  ``@pytest.mark.skipif(not impl_has(<name>), ...)``; once round 2 lands the
  symbol the skip flips to False and the test RUNS.

  This conftest defines the EXPECTED PUBLIC API the round-2 implementation must
  expose (see ``EXPECTED_API`` below). The synthetic-fixture builders here carry
  the PLANTED statistical ground-truth each test asserts against, so a reader can
  see from the fixture exactly what the analysis is supposed to recover.

Determinism: every fixture uses ``numpy.random.default_rng(<seed>)`` with seeds
in the 742X family matching plan v7 §10 (bootstrap seed 742, LEACE synthetic
7421, dCor synthetic 7422, reliability 7423/7424, held-out-LEACE 7425/7426).
"""

from __future__ import annotations

import importlib

import numpy as np

# --------------------------------------------------------------------------- #
# TDD red-phase import guard                                                   #
# --------------------------------------------------------------------------- #
# Canonical round-2 implementation module (plan v7 §4 "New code" + §10 Code row).
IMPL_MODULE_NAME = "explore_persona_space.analysis.issue_742_decoding_ceiling"

try:  # pragma: no cover - exercised by the skipif machinery
    impl = importlib.import_module(IMPL_MODULE_NAME)
    IMPL_EXISTS = True
except Exception:  # ModuleNotFoundError in round 1; any import error in round 2
    impl = None
    IMPL_EXISTS = False


def impl_has(name: str) -> bool:
    """True iff the round-2 implementation module exists AND exposes ``name``.

    Used as the skipif predicate so the suite PASSes in round 1 (module absent →
    every impl-dependent test SKIPPED) and RUNS in round 2 (module present →
    skip flips to False). A symbol that the module imports but does not yet
    expose stays skipped — the granularity is per-symbol, not per-module.
    """
    return IMPL_EXISTS and hasattr(impl, name)


# The public API the round-2 implementation MUST provide for these tests to run.
# (Documentation for the implementer; not asserted directly — each test guards
# on the specific symbol it calls via ``impl_has``.)
EXPECTED_API = {
    # test 1 — reliability decompositions
    "reliability_split_half_over_rollouts": (
        "(rollout_labels: np.ndarray[n_contexts, n_probes, n_rollouts], *, "
        "n_split_seeds=200, rng) -> float r_yy  (split-half over rollouts + Spearman-Brown)"
    ),
    "reliability_split_half_over_probes": (
        "(probe_rates: np.ndarray[n_contexts, n_probes], *, n_split_seeds=200, rng) "
        "-> float r_yy  (split-half over probes + Spearman-Brown)"
    ),
    "reliability_binomial_variance": (
        "(rates: np.ndarray[n_contexts], m_cell: np.ndarray[n_contexts] | int) "
        "-> float r_yy  (binomial-variance decomposition, cell-actual m, sqrt clamped to [0,1])"
    ),
    "load_reliability_estimates": (
        "(behavior, genre, ...) -> object exposing BOTH estimator values "
        "(.split_half, .binomial) + a .disagree flag; surfaces both, never averages"
    ),
    # test 2 — LEACE + PCA basis
    "fit_pca_basis": (
        "(X: np.ndarray[n, d], d_eff: int) -> object with .transform(X)->reduced[n, d_eff]; "
        "the single full-sample PCA-basis fit (Option A). dcor_permutation_test routes "
        "EVERY PCA fit through pca_fit_fn (default = this), so a counting wrapper proves "
        "the refit-per-permutation contract: it is called n_perm + 1 times (MF3, test 3d)"
    ),
    "fit_leace": (
        "(v0: np.ndarray[n, d], E0: np.ndarray[n]) -> eraser with "
        ".transform(v0)->residual + .P projection. dcor_permutation_test routes EVERY "
        "eraser fit through leace_fit_fn (default = this) for the refit-per-permutation "
        "call-count proof (MF3, test 3d)"
    ),
    "leace_residual": "(v0, E0) -> np.ndarray[n, d] full-sample-erased embedding",
    # test 3 — dCor permutation
    "distance_correlation": "(X: np.ndarray[n, d], y: np.ndarray[n]) -> float dCor in [0,1]",
    "dcor_permutation_test": (
        "(v0, E0, *, d_eff=10, n_perm=1000, rng, pca_fit_fn=None, leace_fit_fn=None) "
        "-> object with .dcor, .null (array), .p_value; REFITS the full "
        "PCA->LEACE->dCor pipeline per permutation. pca_fit_fn / leace_fit_fn are "
        "test-injection hooks (default to the real fits) wrapping the PCA-basis fit "
        "and the LEACE-eraser fit; the implementation MUST call EACH exactly "
        "n_perm + 1 times (once for the observed statistic, once per permutation) "
        "so a counting stub proves no cross-fitted / cached coordinate frame leaks "
        "in (plan v7 §13 item 3 + §10 unit test 3, MF3)."
    ),
    "dcor_power_check": (
        "(*, d_eff=10, n=50, n_perm, effect, n_trials, rng) -> float realized power"
    ),
    # test 4 — bootstrap
    "cluster_bootstrap_ci": (
        "(stat_fn, data, *, n_boot=2000, alpha=0.05, rng) -> (lo, hi) percentile CI over clusters"
    ),
    # test 5 — rho_lin source path
    "load_rho_lin": (
        "(behavior, genre, *, eval_dir) -> float; reads analyzer_body_data.json "
        "/<genre>/a33/<beh>/lin_rho; RAISES on assumption_verdicts.json"
    ),
    # test 6 — Stage-0 source resolution
    "snapshot_raw_completions": (
        "(genre, *, dest_dir, hf_download_fn) -> manifest; snapshots + sha256; "
        "raises RawCompletionShortfallError on < J completions"
    ),
    "RawCompletionShortfallError": (
        "Exception raised before judge-batch on a short/missing cache cell"
    ),
    # test 7 — held-out post-LEACE linear diagnostic
    "held_out_linear_leakage": (
        "(v0_train, E0_train, v0_held, E0_held, *, rng) -> object with "
        ".rho (held-out ridge probe on LEACE residual), .null_ci, .post_leace_linear_pass"
    ),
    "classify_stage1_verdict": (
        "(*, dcor_pass: bool, linear_pass: bool) -> str in "
        "{'nonlinear-yes','linear-erasure-leakage-unresolved','ceiling-limited'}"
    ),
}


# --------------------------------------------------------------------------- #
# Synthetic-fixture builders (PLANTED ground-truth for the assertions)         #
# --------------------------------------------------------------------------- #
def make_bernoulli_dataset_with_reliability(
    *,
    n_contexts: int,
    n_probes: int,
    n_rollouts: int,
    target_r_yy: float,
    seed: int,
):
    """Synthetic (contexts x probes x rollouts) Bernoulli rollout-label array
    with a PLANTED signal-to-total reliability ``target_r_yy``.

    Construction (the ground-truth a reader can verify):
      * Each context c has a latent per-context signal ``theta_c`` drawn so its
        across-context variance is ``sigma_sig^2``.
      * Each rollout is Bernoulli(p_c) where ``p_c = sigmoid(theta_c)``; the
        within-context binomial variance is the NOISE term.
      * ``r_yy = signal_var / (signal_var + noise_var)`` is tuned to
        ``target_r_yy`` by scaling ``sigma_sig`` against the realized binomial
        noise at the chosen ``n_rollouts`` (or ``n_probes`` for the 1-rollout
        regime).

    Returns:
        rollout_labels : int array (n_contexts, n_probes, n_rollouts) of {0,1}
        per_context_rate : float array (n_contexts,) realized mean rate per context
        true_r_yy : the realized planted reliability (signal/(signal+noise))
    """
    rng = np.random.default_rng(seed)

    # Total measurements per context determine the binomial noise floor.
    m = n_probes * n_rollouts

    # Center rates near 0.5 so binomial variance p(1-p) is well-defined and large
    # (a saturated-near-0/1 behavior carries little signal — kept off the floor here).
    base_p = 0.5
    noise_var = base_p * (1.0 - base_p) / m  # binomial sampling variance of the rate

    # signal_var chosen so signal/(signal+noise) == target_r_yy:
    #   target = s / (s + noise)  =>  s = target * noise / (1 - target)
    signal_var = target_r_yy * noise_var / (1.0 - target_r_yy)
    sigma_sig = float(np.sqrt(signal_var))

    # latent per-context true rate (clipped to a safe interior band)
    theta_c = np.clip(base_p + rng.normal(0.0, sigma_sig, size=n_contexts), 0.02, 0.98)

    rollout_labels = np.empty((n_contexts, n_probes, n_rollouts), dtype=np.int64)
    for c in range(n_contexts):
        rollout_labels[c] = rng.binomial(1, theta_c[c], size=(n_probes, n_rollouts))

    per_context_rate = rollout_labels.reshape(n_contexts, -1).mean(axis=1)

    # REALIZED reliability of this finite draw (NOT the asymptotic planted
    # parameter). At n_contexts=50 the realized across-context variance is a noisy
    # estimate of (signal_var + noise_var), so the realized variance-ratio
    # reliability of the actual draw differs from the planted `target_r_yy` by the
    # finite-sample sampling noise (std ~0.09 at n=50). The estimators recover the
    # reliability of the DATA THEY ARE GIVEN, so the recovery assertions must target
    # the realized value, not the planted parameter (an unbiased estimator cannot map
    # a +1σ draw back to the population mean). Computed as the binomial
    # variance-ratio decomposition on the realized rates with the cell-actual m —
    # the same SP/Var the estimators target, in the variance-ratio space the
    # split-half + Spearman-Brown reads also live in.
    var_total = float(np.var(per_context_rate))
    within = float(np.mean(per_context_rate * (1.0 - per_context_rate) / m))
    true_r_yy = (var_total - within) / var_total if var_total > 1e-12 else 0.0
    true_r_yy = float(np.clip(true_r_yy, 0.0, 1.0))
    return rollout_labels, per_context_rate, true_r_yy


def make_leace_exact_dataset(*, n: int, d: int, signal_dim: int, seed: int):
    """Synthetic ``(v0, E0)`` where ``E0`` is LINEARLY decodable from ``v0`` along
    a known direction, so LEACE's closed-form erasure must zero the covariance.

    Construction:
      * Draw ``v0`` ~ N(0, I_d).
      * Pick a known unit direction ``w`` (the ``signal_dim``-th basis axis).
      * ``E0 = v0 @ w + small_noise`` — a linear function of v0 along w.
      * After LEACE fit on the FULL sample, ``cov(E0, P @ v0)`` along every
        direction must be ~0 (the closed-form guarantee, Belrose 2023).
      * Directions ORTHOGONAL to w must be (approximately) UNCHANGED — the
        minimal-change property.

    Returns:
        v0 : (n, d) float
        E0 : (n,) float
        w  : (d,) unit direction carrying the signal
    """
    rng = np.random.default_rng(seed)
    v0 = rng.normal(0.0, 1.0, size=(n, d))
    w = np.zeros(d)
    w[signal_dim] = 1.0
    E0 = v0 @ w + rng.normal(0.0, 0.05, size=n)
    return v0, E0, w


def make_nonlinear_dependence(*, n: int, d: int, noise: float, seed: int):
    """``E0 = sigmoid(||v0||^2 standardized) + noise`` — a NONLINEAR function of
    v0 with NO linear component (||v0||^2 is even, so its linear correlation with
    each coordinate is ~0). After LEACE erases the (negligible) linear part, dCor
    should still detect the dependence. The dCor TRUE-POSITIVE fixture (test 3a)."""
    rng = np.random.default_rng(seed)
    v0 = rng.normal(0.0, 1.0, size=(n, d))
    r2 = (v0**2).sum(axis=1)
    r2_std = (r2 - r2.mean()) / (r2.std() + 1e-12)
    E0 = 1.0 / (1.0 + np.exp(-r2_std)) + rng.normal(0.0, noise, size=n)
    return v0, E0


def make_independent_target(*, n: int, d: int, seed: int):
    """``v0`` and ``E0`` independent: ``E0`` iid with NO dependence on ``v0``.
    The dCor TRUE-NEGATIVE / null-centering fixture (test 3b)."""
    rng = np.random.default_rng(seed)
    v0 = rng.normal(0.0, 1.0, size=(n, d))
    E0 = rng.uniform(0.0, 1.0, size=n)  # iid, independent of v0
    return v0, E0


def make_correlated_pair(*, n: int, true_r: float, seed: int):
    """A paired sample (x, y) with a KNOWN Pearson correlation ``true_r``.
    The bootstrap-CI-coverage fixture (test 4)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n)
    z = rng.normal(0.0, 1.0, size=n)
    y = true_r * x + np.sqrt(max(0.0, 1.0 - true_r**2)) * z
    return x, y
