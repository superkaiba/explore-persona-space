"""Test 4 (plan v7 §13 item 4 + §4 Stage-0 step 4 + §11 row 6) — the
cluster-bootstrap-over-contexts CI covers a KNOWN correlation at ~95%.

Plan §11 row 6: bootstrap over the 50 contexts (cluster-bootstrap, B=2000,
seed 742), NEVER std-across-folds (Bengio-Grandvalet 2004 + Varoquaux 2018 —
the std-across-folds underestimates the CV variance).

Planted ground-truth (see conftest.make_correlated_pair): a paired sample with
a known Pearson r=0.5. We repeat the whole bootstrap experiment 100 times over
different synthetic subsamples and assert the 95% CI covers the truth in >=90 of
100 trials (allowing slight under-coverage at finite trial count).

Seed: 742 (the bootstrap seed, plan §10 reproducibility card).
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import impl, impl_has, make_correlated_pair

TRUE_R = 0.5
N_CONTEXTS = 50
N_BOOT = 2000
N_TRIALS = 100
MIN_COVERED = 90  # >= 90/100 (95% nominal, slight under-coverage tolerated)


@pytest.mark.skipif(
    not impl_has("cluster_bootstrap_ci"),
    reason="implementation pending round 2",
)
def test_cluster_bootstrap_ci_covers_known_correlation():
    master = np.random.default_rng(742)
    covered = 0
    for trial in range(N_TRIALS):
        # a fresh synthetic 50-context paired sample with planted r=0.5
        x, y = make_correlated_pair(
            n=N_CONTEXTS, true_r=TRUE_R, seed=int(master.integers(0, 2**31 - 1))
        )
        data = np.column_stack([x, y])  # (50, 2): each row is one "context" cluster

        def pearson(d):
            # correlation statistic over the resampled clusters
            return float(np.corrcoef(d[:, 0], d[:, 1])[0, 1])

        rng = np.random.default_rng(742 + trial)
        lo, hi = impl.cluster_bootstrap_ci(pearson, data, n_boot=N_BOOT, alpha=0.05, rng=rng)
        if lo <= TRUE_R <= hi:
            covered += 1

    assert covered >= MIN_COVERED, (
        f"cluster-bootstrap 95% CI covered the true r={TRUE_R} in only {covered}/{N_TRIALS} "
        f"trials (< {MIN_COVERED}); coverage is below nominal"
    )


@pytest.mark.skipif(
    not impl_has("cluster_bootstrap_ci"),
    reason="implementation pending round 2",
)
def test_cluster_bootstrap_ci_is_an_interval_not_a_point():
    # guard against a degenerate (zero-width / std-across-folds) CI: at n=50 the
    # honest CI is wide (~±0.10-0.15 per Varoquaux 2018), never a point estimate.
    x, y = make_correlated_pair(n=N_CONTEXTS, true_r=TRUE_R, seed=742)
    data = np.column_stack([x, y])
    rng = np.random.default_rng(742)
    lo, hi = impl.cluster_bootstrap_ci(
        lambda d: float(np.corrcoef(d[:, 0], d[:, 1])[0, 1]),
        data,
        n_boot=N_BOOT,
        alpha=0.05,
        rng=rng,
    )
    assert hi > lo, f"CI is not a proper interval (lo={lo:.3f}, hi={hi:.3f})"
    assert (hi - lo) > 0.05, (
        f"CI half-width implausibly tight at n=50 (width={hi - lo:.3f}); a "
        "std-across-folds CI would under-cover here (Varoquaux 2018)"
    )
