"""#1336 dedup-sensitivity re-read invariants (Step 9a-ter inline round).

Pins, on tiny synthetic fixtures THROUGH THE REAL driver helpers (no seam
stubs):
  1. The registered exclusion convention — EVERY member of an exact-dup
     prompt group among the ANALYZED rows is dropped (no keep-one); unjoined
     rows are kept and counted; the join-rate floor fails loud.
  2. The dedup pooling convention — pooled R^2 with fold-local KEPT-row test
     means; keep=all reproduces the committed pooled reads exactly (identity
     with recal.raw_pooled_r2 / crossfit_recal_direct); and the dedup read is
     a PURE RE-REDUCTION (recal params from the FULL row set), distinct from
     re-fitting on the deduped rows.
  3. The P2 quantile read — p97.5 of the FIXED L29 column vs the per-draw
     layer-max band, on a matrix with known quantiles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "scripts"), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

ds = pytest.importorskip("issue1336_dedup_sensitivity")
_rc = pytest.importorskip("explore_persona_space.experiments.issue_1336.recal")


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. Exclusion convention
# ---------------------------------------------------------------------------
def test_dup_exclusion_drops_every_group_member_and_keeps_unjoined():
    n = 20
    conv_ids = np.asarray([f"s{i}" for i in range(n)])
    hashes = {str(i): _sha(f"unique question {i}") for i in range(n - 1)}
    # "19" deliberately absent -> unjoined row, KEPT (19/20 = 0.95, at the floor).
    hashes["1"] = hashes["0"]  # exact-dup pair s0/s1
    hashes["4"] = hashes["3"] = hashes["2"]  # exact-dup TRIPLE s2/s3/s4
    excl, digest = ds.dup_exclusion_mask(conv_ids, hashes)
    # EVERY member of each dup group is dropped; unique + unjoined rows kept.
    expected = [True, True, True, True, True] + [False] * (n - 5)
    assert excl.tolist() == expected
    assert digest["n_dup_groups"] == 2
    assert digest["n_rows_excluded"] == 5
    assert digest["max_group_size"] == 3
    assert digest["n_rows_analyzed"] == n
    assert digest["join_rate"] == pytest.approx((n - 1) / n)


def test_dup_exclusion_group_membership_is_within_analyzed_rows():
    # A hash duplicated in the BANK but with only ONE member among the
    # analyzed rows is NOT excluded (multiplicity counts analyzed rows only).
    hashes = {"0": _sha("q"), "1": _sha("q"), "2": _sha("r")}
    excl, digest = ds.dup_exclusion_mask(np.asarray(["s0", "s2"]), hashes)
    assert excl.tolist() == [False, False]
    assert digest["n_dup_groups"] == 0


def test_dup_exclusion_join_rate_floor_fails_loud():
    with pytest.raises(AssertionError, match="join rate"):
        ds.dup_exclusion_mask(np.asarray(["s0", "s1", "s2"]), {"0": _sha("q")})


# ---------------------------------------------------------------------------
# 2. Pooling convention
# ---------------------------------------------------------------------------
def _fixture(n=40, d=3, n_folds=4, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, d))
    P = Y + 0.5 * rng.normal(size=(n, d))
    folds = np.arange(n) % n_folds
    return P, Y, folds


def test_pooled_r2_keep_all_reproduces_committed_reads_exactly():
    P, Y, folds = _fixture()
    keep = np.ones(len(folds), dtype=bool)
    assert ds.pooled_r2_on_rows(P, Y, folds, keep) == pytest.approx(
        _rc.raw_pooled_r2(P, Y, folds), abs=1e-12
    )
    direct = _rc.crossfit_recal_direct(P, Y, folds)
    assert ds.pooled_r2_on_rows(direct["pred_recal"], Y, folds, keep) == pytest.approx(
        direct["r2"], abs=1e-12
    )


def test_pooled_r2_dedup_uses_fold_local_kept_means():
    P, Y, folds = _fixture()
    keep = np.ones(len(folds), dtype=bool)
    keep[[0, 5, 9]] = False
    got = ds.pooled_r2_on_rows(P, Y, folds, keep)
    # Hand-computed reference: fold-local means over KEPT rows only.
    ss_res = ss_tot = 0.0
    for k in sorted(set(folds)):
        rk = np.flatnonzero((folds == k) & keep)
        t, p = Y[rk], P[rk]
        ss_res += float(((t - p) ** 2).sum())
        ss_tot += float(((t - t.mean(0)) ** 2).sum())
    assert got == pytest.approx(1.0 - ss_res / ss_tot, abs=1e-12)
    # An all-dropped fold contributes zero (no NaN propagation).
    keep2 = folds != 0
    assert np.isfinite(ds.pooled_r2_on_rows(P, Y, folds, keep2))


def test_dedup_is_pure_rereduction_not_a_refit():
    # Fits/recal params UNCHANGED: the dedup read re-pools the FULL-set
    # cross-fitted predictions; it must DIFFER from re-fitting the recal on
    # the deduped rows whenever a duplicated outlier moved the fit.
    P, Y, folds = _fixture(n=24, d=2, n_folds=3, seed=1)
    Y[0] += 15.0  # duplicated outlier pair in fold 0
    Y[3] += 15.0
    keep = np.ones(len(folds), dtype=bool)
    keep[[0, 3]] = False
    full_fit = _rc.crossfit_recal_direct(P, Y, folds)
    rereduced = ds.pooled_r2_on_rows(full_fit["pred_recal"], Y, folds, keep)
    refit = _rc.crossfit_recal_direct(P[keep], Y[keep], folds[keep])["r2"]
    assert abs(rereduced - refit) > 1e-6


# ---------------------------------------------------------------------------
# 3. P2 quantile read
# ---------------------------------------------------------------------------
def test_l29_null_quantile_reads_fixed_column_and_layer_max():
    rng = np.random.default_rng(2)
    layers = [16, 21, 22, 29, 30]
    mat = rng.normal(size=(200, len(layers)))
    mat[:, layers.index(29)] += 1.0  # shift the fixed L29 column
    read = ds.l29_null_quantile(mat, layers, layer=29)
    assert read["layer"] == 29 and read["n_draws"] == 200
    assert read["p975_null_l29_fixed"] == pytest.approx(
        float(np.quantile(mat[:, layers.index(29)], 0.975)), abs=1e-12
    )
    assert read["p975_null_layer_max_recomputed"] == pytest.approx(
        float(np.quantile(mat.max(axis=1), 0.975)), abs=1e-12
    )
    # The fixed-column read is <= the layer-max band by construction.
    assert read["p975_null_l29_fixed"] <= read["p975_null_layer_max_recomputed"] + 1e-12
    assert set(read["p975_null_per_layer"]) == {str(li) for li in layers}
