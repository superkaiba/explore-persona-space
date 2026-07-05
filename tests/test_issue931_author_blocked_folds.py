"""Tests for the #931 author-blocked-folds driver (`author-blocked-folds` r2).

Pins the plan-v10 section-6 registered decision table (the four rows
partition the outcome space: R_a <= band -> row 3; else B_novel above /
inside / below the author-fold bootstrap CI -> rows 1 / 2 / 4, plus the
near-band sensitivity flag), the size-multiset-matched pseudo-group
regrouping draws (multiset preserved, seed-deterministic, seed-distinct),
the protocol-fingerprint checkpoint keying (resume ONLY on exact match;
stale / duplicate rows fail loud), and the section-9 descope ladder ordering
(LOAO drops first, then ctxmean nulls; the within cell is never touched).

CPU-only, tiny shapes, no GPU, no network, no committed-artifact reads
(sparse-worktree safe).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import issue931_author_blocked_folds as abf  # noqa: E402


class TestRegisteredRead:
    """The section-6 decision table — one test per row + the near-band flag.

    B_NOVEL = 0.20806...; rows keyed on (R_a vs band_a) then (B_NOVEL vs CI_a).
    """

    def test_row1_author_level_component_measured(self):
        r = abf.registered_read(0.12, band_a=-0.09, ci_lo_a=0.08, ci_hi_a=0.16)
        assert r["decision_row"] == "row1_author_level_component_measured"
        assert not r["near_band_sensitivity"]

    def test_row2_not_distinguishable(self):
        r = abf.registered_read(0.16, band_a=-0.09, ci_lo_a=0.15, ci_hi_a=0.22)
        assert r["decision_row"] == "row2_no_detectable_author_level_component"

    def test_row3_existence_fails(self):
        r = abf.registered_read(-0.10, band_a=-0.09, ci_lo_a=-0.2, ci_hi_a=0.0)
        assert r["decision_row"] == "row3_existence_fails_under_honest_fold"

    def test_row3_boundary_r_a_equal_band_fires_row3(self):
        r = abf.registered_read(-0.09, band_a=-0.09, ci_lo_a=-0.2, ci_hi_a=0.5)
        assert r["decision_row"] == "row3_existence_fails_under_honest_fold"

    def test_row4_author_fold_higher(self):
        r = abf.registered_read(0.30, band_a=-0.09, ci_lo_a=0.25, ci_hi_a=0.40)
        assert r["decision_row"] == "row4_author_fold_higher_unexpected"

    def test_ci_membership_is_inclusive_row2(self):
        # B_NOVEL exactly at a CI endpoint is INSIDE (row 2, not rows 1/4).
        b = abf.B_NOVEL
        assert (
            abf.registered_read(0.12, band_a=-0.09, ci_lo_a=b, ci_hi_a=b + 0.1)["decision_row"]
            == "row2_no_detectable_author_level_component"
        )
        assert (
            abf.registered_read(0.12, band_a=-0.09, ci_lo_a=b - 0.1, ci_hi_a=b)["decision_row"]
            == "row2_no_detectable_author_level_component"
        )

    def test_near_band_flag(self):
        assert abf.registered_read(-0.05, band_a=-0.09, ci_lo_a=-0.2, ci_hi_a=0.0)[
            "near_band_sensitivity"
        ]
        assert not abf.registered_read(0.10, band_a=-0.09, ci_lo_a=0.2, ci_hi_a=0.3)[
            "near_band_sensitivity"
        ]

    def test_registered_constants(self):
        assert pytest.approx(0.17289959611807892, abs=0) == abf.H_AMEND_NUMERATOR
        assert pytest.approx(0.20806361277603524, abs=0) == abf.B_NOVEL


class TestPseudoGroupDraws:
    """Size-multiset-matched pseudo-group regrouping (section 4 step 5)."""

    def _fixture(self):
        novels = [f"novel{i:02d}" for i in range(28)]
        sizes = abf.EXPECTED_MULTISET
        rng = np.random.default_rng(0)
        group_ids = np.asarray(rng.choice(novels, size=200))
        return group_ids, novels, sizes

    def test_multiset_preserved(self):
        group_ids, novels, sizes = self._fixture()
        _, assignment = abf.pseudo_ids_for_draw(group_ids, novels, sizes, 7)
        realized = tuple(sorted(Counter(assignment.values()).values(), reverse=True))
        assert realized == sizes
        assert len(assignment) == 28

    def test_rows_follow_novels(self):
        group_ids, novels, sizes = self._fixture()
        row_ids, assignment = abf.pseudo_ids_for_draw(group_ids, novels, sizes, 3)
        assert len(row_ids) == len(group_ids)
        for g, p in zip(group_ids, row_ids, strict=True):
            assert assignment[str(g)] == p

    def test_seed_deterministic_and_distinct(self):
        group_ids, novels, sizes = self._fixture()
        a1, _ = abf.pseudo_ids_for_draw(group_ids, novels, sizes, 5)
        a2, _ = abf.pseudo_ids_for_draw(group_ids, novels, sizes, 5)
        b, _ = abf.pseudo_ids_for_draw(group_ids, novels, sizes, 6)
        assert (a1 == a2).all()
        assert (a1 != b).any()


def _args(**over) -> argparse.Namespace:
    base = dict(
        null_draws=20,
        n_boot=1000,
        pseudo_draws=20,
        budget_hours=4.5,
        protocol_tag="",
        layers_kept=tuple(range(28)),
    )
    base.update(over)
    return argparse.Namespace(**base)


class TestCheckpoint:
    """Fingerprint-gated resume: stale / duplicate rows fail loud."""

    def test_roundtrip_and_key(self, tmp_path):
        p = tmp_path / "cells.jsonl"
        abf.append_jsonl(p, {"protocol_fingerprint": "f" * 12, "cell_id": "a", "fold_scheme": "s"})
        abf.append_jsonl(p, {"protocol_fingerprint": "f" * 12, "cell_id": "a", "fold_scheme": "t"})
        by_key = abf.load_checkpoint(p, "f" * 12)
        assert set(by_key) == {("a", "s"), ("a", "t")}

    def test_stale_fingerprint_fails_loud(self, tmp_path):
        p = tmp_path / "cells.jsonl"
        abf.append_jsonl(p, {"protocol_fingerprint": "old", "cell_id": "a", "fold_scheme": "s"})
        with pytest.raises(RuntimeError, match="stale-protocol"):
            abf.load_checkpoint(p, "new")

    def test_duplicate_key_fails_loud(self, tmp_path):
        p = tmp_path / "cells.jsonl"
        row = {"protocol_fingerprint": "fp", "cell_id": "a", "fold_scheme": "s"}
        abf.append_jsonl(p, row)
        abf.append_jsonl(p, row)
        with pytest.raises(RuntimeError, match="duplicate"):
            abf.load_checkpoint(p, "fp")

    def test_runconfig_fingerprint_mismatch_fails_loud(self, tmp_path):
        p = tmp_path / "runconfig.json"
        abf.write_runconfig(p, "fp1", {"applied": False})
        with pytest.raises(RuntimeError, match="fingerprint"):
            abf.read_runconfig(p, "fp2")

    def test_fingerprint_moves_with_protocol(self):
        fp_a = abf.protocol_fingerprint("sha1", _args())
        fp_b = abf.protocol_fingerprint("sha1", _args(null_draws=2))
        fp_c = abf.protocol_fingerprint("sha2", _args())
        assert fp_a != fp_b and fp_a != fp_c
        assert fp_a == abf.protocol_fingerprint("sha1", _args())


class TestDescopeLadder:
    """Section-9 ladder: LOAO first, then ctxmean nulls; within untouched."""

    def test_no_descope_when_fast(self):
        info = abf.apply_descope(60.0, abf.EXPECTED_N, 28, _args())
        assert not info["applied"] and not info["loao_dropped"]
        assert not info["ctxmean_null_draws_dropped"]

    def test_loao_drops_first(self):
        # Wall chosen so dropping LOAO alone brings the projection under 2x.
        args = _args(budget_hours=4.5)
        base = abf.apply_descope(60.0, abf.EXPECTED_N, 28, args)
        # Find a wall where total > threshold but ctxmean nulls survive.
        rate_wall = 60.0 * (2.0 * 4.5) / base["projected_total_hours"]
        info = abf.apply_descope(rate_wall * 1.05, abf.EXPECTED_N, 28, args)
        assert info["applied"] and info["loao_dropped"]

    def test_ctxmean_nulls_drop_only_after_loao(self):
        info = abf.apply_descope(1e6, abf.EXPECTED_N, 28, _args())
        assert info["loao_dropped"] and info["ctxmean_null_draws_dropped"]
        # The recorded projections shrink monotonically down the ladder.
        assert (
            info["projected_total_hours_after_descope"]
            < info["projected_total_hours_after_loao_drop"]
            < info["projected_total_hours"]
        )


class TestMainCellRow:
    """Checkpoint-row extraction: NaN p975 (descoped nulls) maps to None."""

    def _payload(self, p975):
        return {
            "n": 1982,
            "n_groups": 19,
            "null_draws": 0,
            "r2_per_layer_obs": [0.1, 0.2],
            "selection_symmetric": {"frozen_layer_table": {"1": {"null_p975": p975}}},
            "r2_bootstrap_group_frozen": {
                "1": {"r2": 0.1, "ci_lo": 0.0, "ci_hi": 0.2, "n_boot": 20}
            },
            "per_group_r2_headline": {"AUST": 0.1},
        }

    def test_nan_p975_maps_to_none(self):
        row = abf._main_cell_row("fp", "c", self._payload(float("nan")), 1, (18, 19), 1.0)
        assert row["null_p975_l19"] is None
        assert row["headline_layer"] == 19 and row["headline_layer_index"] == 1
        assert json.loads(json.dumps(row))["null_p975_l19"] is None

    def test_finite_p975_preserved(self):
        row = abf._main_cell_row("fp", "c", self._payload(-0.09), 1, (18, 19), 1.0)
        assert row["null_p975_l19"] == pytest.approx(-0.09)
