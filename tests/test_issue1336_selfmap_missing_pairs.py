"""Data-free pins for scripts/issue1336_selfmap_missing_pairs.py.

The numeric matched-basis check (fit_cell reproduces the committed
base->dpo pair-file t0/t6/within at L30 bit-exactly) needs the staged
turnstores and ran live at implementation time; these pins cover the
cheap structural contracts: cell enumeration, the registry-disjointness
assert, the resume key, and the per-record schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue1336_selfmap_missing_pairs as sm  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402


def test_enumerates_32_cells_four_pairs_by_eight_surfaces():
    cells = sm.enumerate_cells()
    assert len(cells) == 32
    pairs = {(s, t) for s, t, _, _ in cells}
    assert pairs == {("base", "base"), ("sft", "rlvr"), ("sft", "rlvr_long"), ("rlvr", "rlvr_long")}
    surfaces = {(c, f) for _, _, f, c in cells}
    assert surfaces == set(cm.v2_surfaces())
    assert sm.PILOT_CELL in cells


def test_missing_pairs_absent_from_registry():
    # run_pair asserts membership in cm.PAIRS; this script exists precisely
    # because these three are NOT registered. If one lands in cm.PAIRS,
    # enumerate_cells refuses (run it through run_pair instead).
    for pair in sm.MISSING_PAIRS:
        assert pair not in cm.PAIRS


def test_resume_key_carries_every_output_affecting_field():
    rk = sm._resume_key("base", "base", "chat", "gsm8k_test1319", 30)
    assert set(rk) == {
        "source",
        "target",
        "format",
        "corpus",
        "layer",
        "fit_seed",
        "n_folds",
        "n_inner",
        "grid_sha",
        "algebra_version",
    }
    assert rk["fit_seed"] == cm.FIT_SEED
    assert rk["n_folds"] == cm.N_FOLDS
    assert rk["n_inner"] == cm.N_INNER_LAMBDA_FOLDS_V2
    # grid sha changes when the lambda grid changes
    assert rk["grid_sha"] == sm._grid_sha(sm._grid())
    assert len(rk["grid_sha"]) == 16


def _fake_fit(names):
    return {
        "n": 100,
        "d": 4096,
        "n_train_per_fold": [80, 80],
        "n_train_min": 80,
        "n_train_max": 80,
        "degenerate_n_lt_d": True,
        "r2": {n: 0.5 for n in names},
        "r2_globalmu": {n: 0.6 for n in names},
        "selected_lambda": {"within": [1e4]},
        "selectors": {"within": ["inner-group-cv"]},
    }


def test_cell_records_self_and_pair_schema():
    self_recs = sm.cell_records("base", "base", "chat", "gsm8k_test1319", 30, _fake_fit(["within"]))
    assert len(self_recs) == 1
    (r,) = self_recs
    assert r["tier"] is None
    assert r["r2"] == r["within_r2"] == 0.5
    assert r["r2_globalmu"] == 0.6
    assert r["degenerate_n_lt_d"] is True

    # v3 (ALGEBRA_VERSION v3-foldlocal-2-cross-t7t8) emits four tiers per pair
    # cell; the name list mirrors fit_cell's own `names` tuple for a pair.
    pair_recs = sm.cell_records(
        "sft",
        "rlvr",
        "chat",
        "if11k",
        30,
        _fake_fit(["within", "t0", "t6", "t7", "t8", "cross", "aans_own"]),
    )
    assert [r["tier"] for r in pair_recs] == [0, 6, 7, 8]
    for r in pair_recs:
        assert r["pair"] == "sft__rlvr"
        assert r["within_r2"] == 0.5
        assert "selected_lambda" in r and "selectors" in r
        # cross and aans_own are per-CELL quantities (no tier) repeated on every
        # tier row so a tier-filtered consumer still sees them.
        assert r["cross_r2"] == 0.5
        assert r["cross_r2_globalmu"] == 0.6
        assert r["aans_own_r2"] == 0.5
        assert r["aans_own_r2_globalmu"] == 0.6

    # A SELF cell has neither a cross map nor an A_ans map (source == target
    # makes it the within map), so the self record must not advertise either.
    assert "cross_r2" not in self_recs[0]
    assert "aans_own_r2" not in self_recs[0]


def test_cell_records_fails_loud_on_a_pre_v3_fit():
    """A v2-shaped fit (no t7/t8/aans_own) must RAISE, never silently drop tiers.

    The v2 -> v3 tier widening is what ALGEBRA_VERSION busts the per-cell
    checkpoints for; a fit dict missing the new names means a stale checkpoint
    leaked past the resume key, and that has to fail rather than emit a
    half-populated tier set.
    """
    import pytest

    with pytest.raises(KeyError):
        sm.cell_records(
            "sft", "rlvr", "chat", "if11k", 30, _fake_fit(["within", "t0", "t6", "cross"])
        )
