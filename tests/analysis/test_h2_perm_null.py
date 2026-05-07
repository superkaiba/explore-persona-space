"""Tests for the round-3 vectorised H2 permuted-label null in analyze_extraction_grid.

Round 2's reference implementation (B*N nested loop, ~742 GPU-h projected at full
sweep size) is the mathematical ground truth for the per-(actor, label) AUC.
Round 3 introduces `auc_actor_label_matrix`, which computes the entire (N, N)
table via one rankdata call per label. These tests verify bit-exact agreement
within numerical tolerance for a small synthetic case (N=5, n_q=10), and verify
that the full permuted-label null pipeline (`AUC_full[label_perms, arange(N)]`)
matches the reference loop's per-persona null distribution.

We import the helpers from `scripts/analyze_extraction_grid.py` via
`importlib.util` (mirroring `tests/test_redact_for_gist.py`) because the
script is not exposed as a package module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Force-import the analysis script as a module.
SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "analyze_extraction_grid.py"
spec = importlib.util.spec_from_file_location("analyze_extraction_grid", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
analyze_extraction_grid = importlib.util.module_from_spec(spec)
sys.modules["analyze_extraction_grid"] = analyze_extraction_grid
spec.loader.exec_module(analyze_extraction_grid)

auc_actor_label_matrix = analyze_extraction_grid.auc_actor_label_matrix
_auc_from_score_matrix = analyze_extraction_grid._auc_from_score_matrix


def _reference_full_table(score_3d: np.ndarray) -> np.ndarray:
    """Round-2 reference: per-(actor, label) AUC computed one cell at a time.

    Returns an (N, N) table where AUC[a, p] = AUC of actor a's q-scores against
    every other actor's q-scores, when scoring against label p's direction.
    """
    n = score_3d.shape[0]
    table = np.full((n, n), np.nan, dtype=np.float64)
    for p in range(n):
        slice_p = score_3d[:, :, p]
        if not np.all(np.isfinite(slice_p)):
            continue
        for a in range(n):
            table[a, p] = _auc_from_score_matrix(slice_p, actor_idx=a)
    return table


def _reference_permuted_null(score_3d: np.ndarray, label_perms: np.ndarray) -> np.ndarray:
    """Round-2 reference inner loop, exact: for each perm b, persona slot p,
    compute AUC at label=p with actor=label_perms[b, p]. Returns (B, N).
    """
    b_count, n = label_perms.shape
    out = np.full((b_count, n), np.nan, dtype=np.float64)
    for b in range(b_count):
        for p in range(n):
            slice_p = score_3d[:, :, p]
            if not np.all(np.isfinite(slice_p)):
                continue
            actor = int(label_perms[b, p])
            out[b, p] = _auc_from_score_matrix(slice_p, actor_idx=actor)
    return out


@pytest.fixture
def synthetic_score_3d() -> np.ndarray:
    """Build a (N=5, n_q=10, D=8) per-q hidden-state stack, project onto each
    persona's centroid, return the (N, n_q, N) score tensor.

    Each persona p has its hidden states drawn from N(p * 0.5, 1) so that
    self-projection AUC is meaningfully > 0.5 — i.e. the test exercises the
    "real signal" regime, not just a degenerate 0.5-everywhere case.
    """
    rng = np.random.default_rng(42)
    n, n_q, d = 5, 10, 8
    # Per-persona offset means + shared isotropic noise.
    offsets = np.arange(n).reshape(n, 1, 1) * 0.5
    acts = rng.standard_normal(size=(n, n_q, d)) + offsets
    centroids = acts.mean(axis=1)  # (N, D)
    # score[a, q, p] = acts[a, q, :] @ centroids[p, :]
    score = np.einsum("aqd,pd->aqp", acts, centroids)
    return score


def test_auc_actor_label_matrix_matches_reference(synthetic_score_3d: np.ndarray) -> None:
    """Vectorised AUC table is bit-exact with the per-cell reference (within fp tol).

    This is THE load-bearing equivalence claim: the vectorised AUC table that
    feeds the H2 permuted-label null and per-persona p-values must match the
    round-2 reference (Mann-Whitney U via `_auc_from_score_matrix`) for every
    (actor, label) pair, otherwise H2 verdicts diverge from the round-2 maths.

    No SNR sanity assertion — with N=5 the synthetic offsets aren't large
    enough to guarantee diag > 0.5 on every persona, but that has no bearing
    on the equivalence claim being tested here.
    """
    fast = auc_actor_label_matrix(synthetic_score_3d)
    ref = _reference_full_table(synthetic_score_3d)
    np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=1e-12)
    # Mean diag should be meaningfully > 0.5 even if individual personas can dip below.
    assert np.diag(fast).mean() > 0.5, f"Expected mean(diag) > 0.5, got {np.diag(fast).mean()}"


def test_auc_actor_label_matrix_handles_nan_label(synthetic_score_3d: np.ndarray) -> None:
    """Labels with all-NaN slices yield NaN AUCs (not crash, not 0.5)."""
    score = synthetic_score_3d.copy()
    # Wipe label p=2 with NaNs to simulate a degenerate centroid.
    score[:, :, 2] = np.nan
    fast = auc_actor_label_matrix(score)
    assert np.isnan(fast[:, 2]).all(), "NaN label slice must propagate"
    # Other labels still finite.
    other_cols = np.delete(fast, 2, axis=1)
    assert np.isfinite(other_cols).all(), "Non-NaN labels must remain finite"


def test_permuted_null_matches_reference(synthetic_score_3d: np.ndarray) -> None:
    """The (B, N) permuted-label null derived via fancy-index over the AUC table
    matches the round-2 reference loop's per-(b, p) AUC computation.

    This is the load-bearing equivalence: round-3's H2 derives `cell_sel_b` and
    `cell_test_b` as `AUC_full[label_perms, arange(N)]`, and that array feeds
    every downstream H2 statistic (per-persona p99, per-persona p-values,
    BH-FDR, Holm).
    """
    n = synthetic_score_3d.shape[0]
    rng = np.random.default_rng(123)
    n_perms = 100
    label_perms = np.stack([rng.permutation(n) for _ in range(n_perms)], axis=0)

    # Fast: one AUC table + fancy-index.
    auc_full = auc_actor_label_matrix(synthetic_score_3d)
    col_idx = np.arange(n)
    fast = auc_full[label_perms, col_idx[np.newaxis, :]]

    # Reference: round-2 per-(b, p) loop.
    ref = _reference_permuted_null(synthetic_score_3d, label_perms)

    np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=1e-12)


def test_permuted_null_distribution_per_persona(synthetic_score_3d: np.ndarray) -> None:
    """Per-persona null distribution percentiles agree across the two paths.

    This is the metric H2 actually consumes: `permuted_null_p99[p]` is computed
    as `np.percentile(permuted_null_test_aucs[:, p], 99)`. Equivalence on the
    (B, N) array implies equivalence on the percentile.
    """
    n = synthetic_score_3d.shape[0]
    rng = np.random.default_rng(7)
    n_perms = 200
    label_perms = np.stack([rng.permutation(n) for _ in range(n_perms)], axis=0)

    auc_full = auc_actor_label_matrix(synthetic_score_3d)
    col_idx = np.arange(n)
    fast_null = auc_full[label_perms, col_idx[np.newaxis, :]]
    ref_null = _reference_permuted_null(synthetic_score_3d, label_perms)

    fast_p99 = np.percentile(fast_null, 99, axis=0)
    ref_p99 = np.percentile(ref_null, 99, axis=0)
    np.testing.assert_allclose(fast_p99, ref_p99, rtol=1e-12, atol=1e-12)


def test_auc_full_self_consistent_with_diag_path(synthetic_score_3d: np.ndarray) -> None:
    """The diagonal of the vectorised table matches per-persona observed AUC
    computed via `_auc_from_score_matrix(score_tv[:, :, p], actor_idx=p)` —
    the round-2 path that round-3 retains for the observed (actor=label) AUC.
    """
    n = synthetic_score_3d.shape[0]
    fast_diag = np.diag(auc_actor_label_matrix(synthetic_score_3d))
    ref_diag = np.array(
        [_auc_from_score_matrix(synthetic_score_3d[:, :, p], actor_idx=p) for p in range(n)]
    )
    np.testing.assert_allclose(fast_diag, ref_diag, rtol=1e-12, atol=1e-12)


def test_auc_actor_label_matrix_rejects_non_3d() -> None:
    """Defensive: helper rejects non-3-D input to catch shape regressions early."""
    with pytest.raises(ValueError):
        auc_actor_label_matrix(np.zeros((5, 10)))
    with pytest.raises(ValueError):
        # Mismatched axis-0 vs axis-2 (should be N == N).
        auc_actor_label_matrix(np.zeros((4, 10, 5)))


def test_auc_actor_label_matrix_c1_style_ties() -> None:
    """Round 3 / N2: verify the argsort-twice path matches the rankdata reference
    on score matrices with WITHIN-ACTOR ties — which is the C1/C2 case after the
    N2 fix (broadcast tiles → constant scores per actor across the n_q axis).

    The per-actor row-SUM is the only quantity the AUC formula consumes from
    the rank tensor, and that sum is invariant to tie-breaking convention as
    long as tied entries occupy contiguous rank blocks. Cross-actor ties are
    probability-zero in real-valued centroids; this test simulates the worst
    case (every actor's q-block is constant) and confirms equivalence.
    """
    rng = np.random.default_rng(11)
    n, n_q, d = 6, 8, 12
    # Synthesize C1-style data: per-actor centroids, score[a, q, p] = c[a]·c[p].
    centroids = rng.standard_normal((n, d))
    gram = centroids @ centroids.T  # (N, N)
    # Broadcast over q: score[a, q, p] = gram[a, p] for every q.
    score = np.broadcast_to(gram[:, None, :], (n, n_q, n)).copy()

    # Reference path: explicit rankdata-per-cell loop.
    ref = _reference_full_table(score)
    fast = auc_actor_label_matrix(score)
    np.testing.assert_allclose(
        fast,
        ref,
        rtol=1e-12,
        atol=1e-12,
        err_msg="C1/C2-style within-actor ties broke argsort-twice equivalence",
    )
