"""Tests for the #926 partition-invariant split-MLP seeding contract.

Pins that ``fit_batched_split_mlp`` (analysis/vectorized_mlp_skill.py) derives
each group's init from ``split_group_init_seed(seed, group.key)`` -- NOT the
group's batch position -- so any partition / reordering / rechunking of the
same group list across calls yields bit-identical per-group results on CPU
(plan #926 AC1), and that the pre-existing LOCO exactness gate is unaffected
(AC3). CPU-only, tiny shapes, no GPU, no network.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from explore_persona_space.analysis.vectorized_mlp_skill import (
    SplitMLPGroup,
    assert_matches_reference,
    assert_split_mlp_matches_serial,
    assert_split_mlp_partition_invariant,
    fit_batched_split_mlp,
    split_group_init_seed,
)

# Frozen literal of split_group_init_seed(658, ("g",)): blake2b(digest_size=8)
# of b"658|('g',)", big-endian, % 2**63 -- computed once and inlined to pin the
# derivation's cross-process / cross-platform stability (plan #926 s3.4 test 4).
_EXPECTED_SEED_658_G = 3045359955441546440

# Shared fit settings: CPU, small, thread count pinned (the asserted bit-identity
# contract is environment-pinned, not ambient).
_KW = dict(seed=658, hidden=16, max_epochs=10, device="cpu", chunk_size=2, num_threads=8)


def _make_groups(n_groups, *, n_train=24, n_eval=8, n_val=0, d=6, p=2, data_seed=0, keys=None):
    """Build tiny synthetic SplitMLPGroups with distinct keys and distinct data."""
    rng = np.random.default_rng(data_seed)
    total = n_train + n_eval + n_val
    groups = []
    for i in range(n_groups):
        X = rng.standard_normal((total, d)).astype(np.float32)
        Y = (X @ rng.standard_normal((d, p)) * 0.05 + 0.1 * rng.standard_normal((total, p))).astype(
            np.float32
        )
        key = keys[i] if keys is not None else (f"g{i}",)
        val_kwargs = {}
        if n_val:
            val_kwargs = dict(X_val=X[n_train + n_eval :], Y_val=Y[n_train + n_eval :])
        groups.append(
            SplitMLPGroup(
                key, X[:n_train], Y[:n_train], X[n_train : n_train + n_eval], **val_kwargs
            )
        )
    return groups


def _assert_group_equal(a, b, key, *, check_val_epoch=False):
    """Bit-identity of one group's preds + every param array across two results."""
    assert np.array_equal(a.preds_by_key[key], b.preds_by_key[key]), key
    for name in ("W1", "b1", "W2", "b2", "mu", "sd"):
        assert np.array_equal(a.params_by_key[key][name], b.params_by_key[key][name]), (key, name)
    if check_val_epoch:
        assert a.best_val_epoch_by_key[key] == b.best_val_epoch_by_key[key], key


def test_partition_and_reorder_invariance():
    """The in-module gate: aligned 2+3 + misaligned 3+2 partitions, reversal,
    cross-chunk-size, and key-seed distinctness (plan s3.3(ii))."""
    out = assert_split_mlp_partition_invariant()
    assert out["partition_bit_identical"] and out["reorder_bit_identical"]
    assert out["cross_chunk_bit_identical"] and out["distinct_key_seeds"]


def test_partition_invariance_with_validation():
    """2+2 split vs full call on the best-val snapshot path (X_val/Y_val set):
    preds, params, AND best_val_epoch_by_key bit-identical/equal per key."""
    groups = _make_groups(4, n_val=8)
    full = fit_batched_split_mlp(groups, **_KW)
    part_a = fit_batched_split_mlp(groups[:2], **_KW)
    part_b = fit_batched_split_mlp(groups[2:], **_KW)
    for g in groups:
        part = part_a if g.key in part_a.preds_by_key else part_b
        _assert_group_equal(full, part, g.key, check_val_epoch=True)


def test_determinism_across_runs_ignores_ambient_rng():
    """Self-seeding: perturbing the global torch RNG between two identical calls
    cannot change the fit (pins that every draw is under a per-group manual_seed)."""
    groups = _make_groups(3)
    first = fit_batched_split_mlp(groups, **_KW)
    torch.manual_seed(12345)
    torch.rand(1000)
    second = fit_batched_split_mlp(groups, **_KW)
    for g in groups:
        _assert_group_equal(first, second, g.key)


def test_split_group_init_seed_stable_constant():
    """Frozen-literal pin of the seed derivation + helper-level key distinctness."""
    assert split_group_init_seed(658, ("g",)) == _EXPECTED_SEED_658_G
    assert split_group_init_seed(658, ("a",)) != split_group_init_seed(658, ("b",))


def test_distinct_keys_distinct_inits():
    """Two groups with IDENTICAL data but DIFFERENT keys get DIFFERENT inits --
    a shared-init / key-ignoring degeneracy would make every invariance test
    above vacuously true. At max_epochs=0 on the no-validation path the returned
    params ARE the inits, so W1 inequality reads the init directly."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((32, 6)).astype(np.float32)
    Y = rng.standard_normal((32, 2)).astype(np.float32)
    groups = [
        SplitMLPGroup(("a",), X[:24], Y[:24], X[24:]),
        SplitMLPGroup(("b",), X[:24], Y[:24], X[24:]),
    ]
    res = fit_batched_split_mlp(groups, **{**_KW, "max_epochs": 0})
    assert not np.array_equal(res.params_by_key[("a",)]["W1"], res.params_by_key[("b",)]["W1"])


def test_duplicate_key_raises():
    """Duplicate (or repr-colliding) keys in one call fail loud."""
    groups = _make_groups(2, keys=[("dup",), ("dup",)])
    with pytest.raises(AssertionError, match="duplicate"):
        fit_batched_split_mlp(groups, **_KW)


def test_serial_reference_gate_passes():
    """The updated serial-reference gate (seeded via split_group_init_seed) passes."""
    out = assert_split_mlp_matches_serial()
    assert out["max_abs_delta"] <= out["tol"]


def test_loco_exactness_gate_unaffected():
    """AC3: the pre-existing #658/#722 LOCO exactness gate on main still passes
    byte-untouched (previously uncovered by any test; ~3-40 s CPU)."""
    out = assert_matches_reference()
    assert out["base_delta"] <= out["tol"] and out["shuffle_delta"] <= out["tol"]
    assert out["chunk_invariant"] and out["crossgroup_clean"]
