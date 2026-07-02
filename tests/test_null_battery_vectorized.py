"""Equivalence tests: vectorized null-battery draw loops vs serial references.

The #778 r4 perf change batches `perm_null_draws` / `randnorm_null_draws`
(subset-sum GEMM + batched Pearson / Fisher-z) while preserving the exact
rng draw sequence. These tests pin vectorized == serial to float tolerance
on synthetic data (rule: .claude/rules/vectorize-many-cell-fits.md item 5).
"""

from __future__ import annotations

import numpy as np
import pytest

from explore_persona_space.analysis.null_battery import (
    _perm_null_draws_serial,
    _randnorm_null_draws_serial,
    perm_null_draws,
    r_per_layer,
    randnorm_null_draws,
    within_condition_r_per_layer,
)

N_POS, N_NEG, L, D, M, N_DRAWS = 6, 9, 3, 7, 15, 5


@pytest.fixture()
def synth():
    rng = np.random.default_rng(1234)
    pos = rng.standard_normal((N_POS, L, D))
    neg = rng.standard_normal((N_NEG, L, D))
    predictor = rng.standard_normal((M, L, D))
    target = rng.standard_normal(M)
    # groups of 4/4/4/3 — the 3-member group exercises the <4 skip path
    condition_ids = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3] * 3)
    return pos, neg, predictor, target, condition_ids


@pytest.mark.parametrize("within", [False, True])
def test_perm_vectorized_matches_serial(synth, within):
    pos, neg, predictor, target, cids = synth
    kw = dict(n_draws=N_DRAWS, seed=0, within=within, condition_ids=cids if within else None)
    got = perm_null_draws(pos, neg, predictor, target, **kw)
    ref = _perm_null_draws_serial(pos, neg, predictor, target, **kw)
    assert got.shape == (N_DRAWS, L)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("within", [False, True])
def test_randnorm_vectorized_matches_serial(synth, within):
    pos, neg, predictor, target, cids = synth
    pool = np.concatenate([pos, neg], axis=0)
    pool_by_layer = {layer: pool[:, layer, :] for layer in range(L)}
    rb_norms = np.linspace(0.5, 2.0, L)
    kw = dict(
        n_draws=N_DRAWS, lam=0.1, seed=0, within=within, condition_ids=cids if within else None
    )
    got = randnorm_null_draws(pool_by_layer, rb_norms, predictor, target, **kw)
    ref = _randnorm_null_draws_serial(pool_by_layer, rb_norms, predictor, target, **kw)
    assert got.shape == (N_DRAWS, L)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_randnorm_zero_norm_layer_yields_nan(synth):
    """A zero ‖r_B[layer]‖ must yield NaN draws for that layer, matching serial
    (which scales every sampled direction to zero there)."""
    pos, neg, predictor, target, _ = synth
    pool = np.concatenate([pos, neg], axis=0)
    pool_by_layer = {layer: pool[:, layer, :] for layer in range(L)}
    rb_norms = np.array([1.0, 0.0, 2.0])  # middle layer has a zero-norm r_B
    kw = dict(n_draws=N_DRAWS, lam=0.1, seed=0, within=False, condition_ids=None)
    got = randnorm_null_draws(pool_by_layer, rb_norms, predictor, target, **kw)
    ref = _randnorm_null_draws_serial(pool_by_layer, rb_norms, predictor, target, **kw)
    assert np.isnan(got[:, 1]).all()
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_perm_chunk_boundary_preserves_rng_sequence(synth):
    """Draw counts straddling the chunk size must reproduce the same prefix."""
    pos, neg, predictor, target, _ = synth
    full = perm_null_draws(pos, neg, predictor, target, n_draws=N_DRAWS, seed=0)
    prefix = perm_null_draws(pos, neg, predictor, target, n_draws=2, seed=0)
    # Not bit-exact: BLAS picks different GEMM blocking per batch size (last-ulp
    # diffs). A broken rng sequence would produce O(1) differences instead.
    np.testing.assert_allclose(full[:2], prefix, rtol=1e-12, atol=1e-15)


def test_batched_single_direction_matches_unbatched(synth):
    """The batched Pearson/Fisher-z core agrees with the single-direction fns."""
    from explore_persona_space.analysis.null_battery import (
        _batched_r_per_layer,
        _batched_within_condition_r_per_layer,
    )

    pos, _, predictor, target, cids = synth
    direction = pos.mean(axis=0)  # (L, D)
    got = _batched_r_per_layer(predictor, direction[None], target)[0]
    ref = r_per_layer(predictor, direction, target)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)

    got_w = _batched_within_condition_r_per_layer(predictor, direction[None], target, cids)[0]
    ref_w = within_condition_r_per_layer(predictor, direction, target, cids)
    np.testing.assert_allclose(got_w, ref_w, rtol=1e-9, atol=1e-12)
