"""Pin the #1489 shared cross-pool fold partition (code-review v1 Critical #1/#2).

Invariant: correlated same-base variants of a prefix must land in the SAME
fold index in EVERY pool, so no cross-pool fit/apply (the Q1 5x5 transfer
matrix, the Q6 M_plain-on-FT read, the gating dv_hat read) can train on a
row whose same-prefix twin is in the test fold. The parent per-pool
`_folds_from_manifest` shuffles each pool's OWN group list (content-dependent
permutation), which misaligns fold ids whenever the pools' group universes
differ — pinned here as the documented counterexample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue1092_fit_grid import _folds_from_manifest  # noqa: E402
from issue1489_fit_grid import (  # noqa: E402
    _assert_fold_alignment,
    _row_groups,
    folds_from_assignment,
    shared_fold_assignment,
)

N_FOLDS = 6


def _pool_rows(prefixes: list[str], rows_per_prefix: int) -> list[dict]:
    rows = []
    for p in prefixes:
        for j in range(rows_per_prefix):
            rows.append({"prefix_id": p, "base_row_id": f"{p}-r{j}"})
    return rows


def _realized_fold_of(rows: list[dict], folds: list[np.ndarray]) -> dict[str, int]:
    groups = _row_groups(rows, "prefix_id")
    out: dict[str, int] = {}
    for f_i, fold in enumerate(folds):
        for i in fold:
            out[groups[int(i)]] = f_i
    return out


def test_shared_partition_aligns_across_pools_with_different_universes():
    """The realized #1489 shape: plain = crossed + plain-only prefixes; a
    family pool = crossed prefixes only (4 variants each). Same prefix ->
    same fold index in both pools, and no fold's family test rows share a
    prefix with the plain rows OUTSIDE that fold (the leak-free property)."""
    crossed = [f"p{i:04d}" for i in range(100)]
    plain_only = [f"q{i:04d}" for i in range(200)]
    plain_rows = _pool_rows(crossed + plain_only, 2)
    fam_rows = _pool_rows(crossed, 4)  # 4 augmented variants per prefix

    union = set(_row_groups(plain_rows, "prefix_id")) | set(_row_groups(fam_rows, "prefix_id"))
    fold_of, n_folds = shared_fold_assignment(union, N_FOLDS)
    assert n_folds == N_FOLDS
    folds_plain = folds_from_assignment(
        plain_rows, len(plain_rows), fold_of, group_key="prefix_id", n_folds=n_folds
    )
    folds_fam = folds_from_assignment(
        fam_rows, len(fam_rows), fold_of, group_key="prefix_id", n_folds=n_folds
    )
    assert len(folds_plain) == n_folds and len(folds_fam) == n_folds

    # the Critical #1 mechanizable check: per-pool prefix->fold maps agree
    m_plain = _realized_fold_of(plain_rows, folds_plain)
    m_fam = _realized_fold_of(fam_rows, folds_fam)
    for g in set(m_plain) & set(m_fam):
        assert m_plain[g] == m_fam[g], (g, m_plain[g], m_fam[g])

    # leak-free property: fam fold-f_i prefixes never appear in the plain
    # TRAINING set of fold f_i (train = plain rows outside plain fold f_i)
    plain_groups = _row_groups(plain_rows, "prefix_id")
    for f_i in range(n_folds):
        train_mask = np.ones(len(plain_rows), dtype=bool)
        train_mask[folds_plain[f_i]] = False
        train_groups = {plain_groups[i] for i in np.flatnonzero(train_mask)}
        fam_groups = _row_groups(fam_rows, "prefix_id")
        test_groups = {fam_groups[int(i)] for i in folds_fam[f_i]}
        assert not (train_groups & test_groups)

    # the runtime guard accepts the aligned construction
    _assert_fold_alignment(
        {"plain": plain_rows, "fam": fam_rows},
        {"plain": folds_plain, "fam": folds_fam},
        group_key="prefix_id",
    )


def test_parent_per_pool_partitions_misalign_on_this_shape():
    """Documented counterexample: the parent `_folds_from_manifest` applied
    PER POOL misaligns prefixes whenever the pools' group universes differ
    (the pre-fix v1 bug — 870/1060 crossed prefixes at the realized shape).
    If this ever starts passing aligned, the parent's contract changed and
    the shared-assignment layer should be re-evaluated."""
    crossed = [f"p{i:04d}" for i in range(100)]
    plain_only = [f"q{i:04d}" for i in range(200)]
    plain_rows = _pool_rows(crossed + plain_only, 2)
    fam_rows = _pool_rows(crossed, 4)
    fp = _realized_fold_of(
        plain_rows,
        _folds_from_manifest(plain_rows, len(plain_rows), group_key="prefix_id", n_folds=N_FOLDS),
    )
    ff = _realized_fold_of(
        fam_rows,
        _folds_from_manifest(fam_rows, len(fam_rows), group_key="prefix_id", n_folds=N_FOLDS),
    )
    misaligned = [g for g in set(fp) & set(ff) if fp[g] != ff[g]]
    assert misaligned, "parent per-pool folds unexpectedly aligned — re-evaluate the fix layer"


def test_alignment_guard_trips_on_misaligned_folds():
    """_assert_fold_alignment must FAIL LOUD on a deliberately swapped fold."""
    rows = _pool_rows([f"p{i}" for i in range(12)], 1)
    union = set(_row_groups(rows, "prefix_id"))
    fold_of, n_folds = shared_fold_assignment(union, 3)
    folds = folds_from_assignment(rows, len(rows), fold_of, group_key="prefix_id", n_folds=n_folds)
    # second pool with folds 0/1 swapped -> same prefix, different fold index
    swapped = [folds[1], folds[0], folds[2]]
    with pytest.raises(AssertionError, match="fold misalignment"):
        _assert_fold_alignment(
            {"a": rows, "b": rows}, {"a": folds, "b": swapped}, group_key="prefix_id"
        )


def test_logo_when_nfolds_nonpositive():
    """n_folds<=0 -> one fold per group (the true-LOTO path for unit_loto)."""
    rows = _pool_rows([f"t{i}" for i in range(12)], 3)
    union = set(_row_groups(rows, "prefix_id"))
    fold_of, n_folds = shared_fold_assignment(union, 0)
    assert n_folds == 12
    folds = folds_from_assignment(rows, len(rows), fold_of, group_key="prefix_id", n_folds=n_folds)
    sizes = [len(f) for f in folds]
    assert sizes == [3] * 12


def test_missing_group_fails_loud():
    rows = _pool_rows(["p0", "p1"], 2)
    fold_of, n_folds = shared_fold_assignment({"p0"}, 1)
    with pytest.raises(KeyError, match="missing from the shared"):
        folds_from_assignment(rows, len(rows), fold_of, group_key="prefix_id", n_folds=n_folds)
