"""arm19_map_mlp_pred (MLP readout on the MAPPED answer) — #1739 fair-protocol round.

Pins the new arm the way its neighbours are pinned
(tests/test_issue1739_transfer_roster.py): registry declaration, dispatch
behaviour on a tiny real run_cell_multi, the rb-independent shared-identity
contract, the mapfit-absent recorded skip, transfer-cell scoring, and the
differs-from-arm5-only-in-input contract (identity map => arm19 preds equal
arm5 preds bitwise: same batched helper, same seed stream, identical X).
"""

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739 import arms
from explore_persona_space.experiments.issue_1739.fits import BudgetCell, MapFit

SLUG = "arm19_map_mlp_pred"


def _identity_mapfit(ly: int, d: int) -> MapFit:
    return MapFit(
        w=np.stack([np.eye(d) for _ in range(ly)]),
        x_mu=np.zeros((ly, 1, d)),
        x_sd=np.ones((ly, 1, d)),
        y_mu=np.zeros((ly, 1, d)),
        diagnostics={},
        kind="linear",
    )


def _toy(n=24, d=5, ly=2, n_regimes=2, seed=0, mapfit="identity"):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(ly, n, d))
    za = z + 0.3 * rng.normal(size=(ly, n, d))
    dv = rng.normal(size=n)
    mf = _identity_mapfit(ly, d) if mapfit == "identity" else mapfit
    datas = [
        arms.CellData(
            z_ctx=z,
            z_ans=za,
            dv=dv,
            rb=rng.normal(size=(ly, d)),
            mapfit=mf,
            layers=tuple(range(ly)),
        )
        for _ in range(n_regimes)
    ]
    cell = BudgetCell(
        row_idx=np.arange(n),
        fold_ids=np.arange(n) % 3,
        n_folds=3,
        budget_l=n,
        draw=0,
        seed=0,
        fold_scheme="toy",
    )
    return datas, cell


def test_registry_entry():
    spec = arms.ARM_REGISTRY[SLUG]
    assert spec["family"] == "map"
    assert spec["rb_dep"] is False
    assert spec["layered"] is True


def test_scores_and_shared_identity_across_regimes():
    """arm19 produces (Ly, n) OOF scores, SHARED by identity (rb-independent)."""
    datas, cell = _toy()
    outs = arms.run_cell_multi(datas, cell, arms=[SLUG], device="cpu")
    (s0, sk0), (s1, sk1) = outs
    assert SLUG in s0 and SLUG in s1, (sk0, sk1)
    assert s0[SLUG].shape == (2, 24)
    assert np.isfinite(s0[SLUG]).all()
    assert s0[SLUG] is s1[SLUG], "rb_dep=False must share the SAME ndarray across regimes"


def test_identity_map_reproduces_arm5_bitwise():
    """With an identity map, mp == z exactly, so arm19 == arm5 (same helper + seed).

    This IS the differs-from-arm5-only-in-input contract: the only thing the
    new arm changes is the input tensor, so making the input tensors equal
    must make the outputs equal.
    """
    datas, cell = _toy(n_regimes=1)
    (scores, skipped) = arms.run_cell_multi(datas, cell, arms=["arm5_mlp_ctx", SLUG], device="cpu")[
        0
    ]
    assert not skipped, skipped
    np.testing.assert_array_equal(scores[SLUG], scores["arm5_mlp_ctx"])


def test_no_mapfit_records_a_skip_never_a_silent_drop():
    datas, cell = _toy(n_regimes=1)
    bare = arms.CellData(
        z_ctx=datas[0].z_ctx,
        z_ans=datas[0].z_ans,
        dv=datas[0].dv,
        rb=datas[0].rb,
        mapfit=None,
        layers=datas[0].layers,
    )
    (scores, skipped) = arms.run_cell_multi([bare], cell, arms=[SLUG], device="cpu")[0]
    assert SLUG not in scores
    assert "no mapfit" in skipped[SLUG]
    assert not arms.roster_accounting_skips([SLUG], scores, skipped)


def test_scores_on_a_transfer_cell():
    """arm19 produces finite eval-block scores under the transfer leg's fold shape."""
    datas, cell = _toy(n_regimes=1)
    rng = np.random.default_rng(7)
    z_ev = rng.normal(size=(2, 9, 5))
    za_ev = rng.normal(size=(2, 9, 5))
    dv_ev = rng.normal(size=9)
    scores, skipped = arms.run_transfer_cell(
        datas[0],
        cell,
        z_ev,
        dv_ev,
        za_ev=za_ev,
        arms=[SLUG, "arm7_map_ridge_pred"],
        device="cpu",
        ridge_folds=(0,),
    )
    assert SLUG in scores, skipped
    assert scores[SLUG].shape[1] == 9
    assert np.isfinite(scores[SLUG]).any()
    assert not arms.roster_accounting_skips([SLUG, "arm7_map_ridge_pred"], scores, skipped)


def test_fold_floor_records_a_skip():
    """A degenerate cell (one fold holding all-but-one row) skips with a reason."""
    datas, _ = _toy(n_regimes=1, n=6)
    cell = BudgetCell(
        row_idx=np.arange(6),
        fold_ids=np.array([0, 1, 1, 1, 1, 1]),
        n_folds=2,
        budget_l=6,
        draw=0,
        seed=0,
        fold_scheme="toy-degenerate",
    )
    (scores, skipped) = arms.run_cell_multi(datas[:1], cell, arms=[SLUG], device="cpu")[0]
    assert SLUG not in scores
    assert "mlp fold floor" in skipped[SLUG]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
