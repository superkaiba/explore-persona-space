"""Tests for the #931 multi-seed power-curve driver (`matched-n-denominator-dip`).

Pins the plan-v7 section-6 registered decision table (crossover boundary
0.34579919... = 2x the H1 numerator, incl. the stable-but-subbar row), the
protocol-fingerprint checkpoint keying (resume ONLY on exact match; stale /
duplicate rows fail loud), the aggregation cardinality assert, and the
default-preserving `collect_lambdas` kwarg on
`issue825_fit_cells.heldout_r2_sweep`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_power_curve_multi_seed as pcms  # noqa: E402


class TestRegisteredRead:
    """The section-6 decision table, one test per outcome row + boundaries."""

    def test_draw_artifact_row(self):
        r = pcms.registered_read(0.42, 0.02)
        assert r["decision"] == "dip_draw_artifact"
        assert r["h1_fraction_draw_avg"] < 0.5
        assert not r["above_half_bar"]

    def test_draw_artifact_fires_regardless_of_sd(self):
        assert pcms.registered_read(0.42, 0.30)["decision"] == "dip_draw_artifact"

    def test_stable_above_bar_row(self):
        r = pcms.registered_read(0.34, 0.02)
        assert r["decision"] == "dip_stable_above_bar"
        assert r["h1_fraction_draw_avg"] >= 0.5
        assert r["above_half_bar"]

    def test_stable_below_bar_row(self):
        r = pcms.registered_read(0.36, 0.02)
        assert r["decision"] == "dip_stable_below_bar"
        assert 0.432 < r["h1_fraction_draw_avg"] < 0.5
        assert not r["above_half_bar"]

    def test_crossover_boundary_exact(self):
        crossover = 2.0 * pcms.H1_NUMERATOR
        assert crossover == pytest.approx(0.34579919, abs=1e-8)
        at = pcms.registered_read(crossover, 0.02)
        assert at["decision"] == "dip_stable_above_bar"
        assert at["h1_fraction_draw_avg"] == pytest.approx(0.5)
        above = pcms.registered_read(crossover + 1e-9, 0.02)
        assert above["decision"] == "dip_stable_below_bar"

    def test_draw_noisy_row(self):
        assert pcms.registered_read(0.36, 0.06)["decision"] == "draw_noisy"

    def test_sd_boundary_inclusive(self):
        # "SD <= 0.05" is inclusive: exactly 0.05 is stable, just above is noisy.
        assert pcms.registered_read(0.36, 0.05)["decision"] == "dip_stable_below_bar"
        assert pcms.registered_read(0.36, 0.0500001)["decision"] == "draw_noisy"

    def test_mean_boundary_at_0p4(self):
        assert pcms.registered_read(0.4, 0.01)["decision"] == "dip_draw_artifact"
        assert pcms.registered_read(0.3999999, 0.01)["decision"] == "dip_stable_below_bar"


class TestProtocolFingerprint:
    def test_deterministic_and_12_hex(self):
        a = pcms.protocol_fingerprint("abc123")
        assert a == pcms.protocol_fingerprint("abc123")
        assert len(a) == 12 and all(c in "0123456789abcdef" for c in a)

    def test_sensitive_to_every_basis_component(self):
        base = pcms.protocol_fingerprint("abc123")
        assert pcms.protocol_fingerprint("def456") != base  # driver git SHA
        assert pcms.protocol_fingerprint("abc123", store_revision="0" * 40) != base
        assert pcms.protocol_fingerprint("abc123", protocol_tag="bump") != base


class TestCheckpoint:
    def _row(self, fp: str, seed: int, n: int) -> dict:
        return {"protocol_fingerprint": fp, "seed": seed, "n": n, "r2_l19": 0.5}

    def test_roundtrip_and_resume_keying(self, tmp_path):
        ckpt = tmp_path / "cells.jsonl"
        fp = pcms.protocol_fingerprint("sha")
        pcms.append_jsonl(ckpt, self._row(fp, 931, 1000))
        pcms.append_jsonl(ckpt, self._row(fp, 932, 1000))
        by_key = pcms.load_checkpoint(ckpt, fp)
        assert set(by_key) == {(931, 1000), (932, 1000)}

    def test_stale_fingerprint_fails_loud(self, tmp_path):
        ckpt = tmp_path / "cells.jsonl"
        fp = pcms.protocol_fingerprint("sha")
        stale_fp = pcms.protocol_fingerprint("sha", protocol_tag="old-protocol")
        pcms.append_jsonl(ckpt, self._row(fp, 931, 1000))
        pcms.append_jsonl(ckpt, self._row(stale_fp, 932, 1000))
        with pytest.raises(RuntimeError, match="stale-protocol"):
            pcms.load_checkpoint(ckpt, fp)

    def test_duplicate_cell_fails_loud(self, tmp_path):
        ckpt = tmp_path / "cells.jsonl"
        fp = pcms.protocol_fingerprint("sha")
        pcms.append_jsonl(ckpt, self._row(fp, 931, 1000))
        pcms.append_jsonl(ckpt, self._row(fp, 931, 1000))
        with pytest.raises(RuntimeError, match="duplicate"):
            pcms.load_checkpoint(ckpt, fp)

    def test_missing_checkpoint_is_empty(self, tmp_path):
        assert pcms.load_checkpoint(tmp_path / "absent.jsonl", "abc") == {}


class TestAggregationCardinality:
    def _cell(self, seed: int, n: int) -> dict:
        return {"seed": seed, "n": n, "r2_l19": 0.4, "r2_per_layer": [0.4] * 28}

    def test_complete_grid_aggregates(self):
        seeds, ns = (931, 932), [1000]
        by_key = {(s, n): self._cell(s, n) for s in seeds for n in ns}
        per_n = pcms.aggregate(by_key, seeds, ns)
        assert per_n["1000"]["l19_mean"] == pytest.approx(0.4)

    def test_missing_cell_fails_loud(self):
        seeds, ns = (931, 932, 933), [1000]
        by_key = {(s, 1000): self._cell(s, 1000) for s in (931, 932)}
        with pytest.raises(RuntimeError, match="missing"):
            pcms.aggregate(by_key, seeds, ns)

    def test_extra_cell_fails_loud(self):
        seeds, ns = (931,), [1000]
        by_key = {
            (931, 1000): self._cell(931, 1000),
            (932, 1000): self._cell(932, 1000),
        }
        with pytest.raises(RuntimeError, match="extra"):
            pcms.aggregate(by_key, seeds, ns)


class TestRegisteredBlock:
    def _transfer_matrix(self) -> dict:
        row = {
            "layer": 19,
            "denominator_cell": "chat_ref",
            "denominator_n_train": 1982,
            "direction": "armA_within->chat",
            "x_recipe": "spanmean",
            "transfer_r2": 0.014,
            "within_ceiling_r2": 0.3162,
            "fraction_of_ceiling": 0.0443,
        }
        rows = []
        for direction, recipe in (
            ("armA_within_lastpos->chat", "lastpos"),
            ("armA_within->chat", "spanmean"),
        ):
            for application in ("recentered", "strict"):
                r = dict(row)
                r.update(direction=direction, x_recipe=recipe, application=application)
                rows.append(r)
        # decoy rows the filter must exclude
        rows.append({**row, "layer": 14, "application": "recentered"})
        rows.append({**row, "denominator_n_train": 5000, "application": "recentered"})
        return {"rows": rows}

    def test_incomplete_below_five_decision_cells(self):
        by_key = {(s, pcms.DECISION_N): {"r2_l19": 0.32} for s in (931, 932)}
        block = pcms.build_registered_block(by_key, 0.3162, None)
        assert block["status"] == "INCOMPLETE"
        assert block["supersedes_committed"] is False
        assert block["n_decision_cells"] == 2

    def test_complete_supersedes_and_redivides(self):
        vals = [0.30, 0.31, 0.32, 0.33, 0.34]
        by_key = {
            (s, pcms.DECISION_N): {"r2_l19": v}
            for s, v in zip(pcms.SEEDS_DEFAULT, vals, strict=True)
        }
        block = pcms.build_registered_block(by_key, 0.3162, self._transfer_matrix())
        assert block["status"] == "COMPLETE" and block["supersedes_committed"]
        mean = float(np.mean(vals))
        assert block["draw_mean"] == pytest.approx(mean)
        assert block["h1_fraction_draw_avg"] == pytest.approx(pcms.H1_NUMERATOR / mean)
        rows = block["superseded_transfer_rows"]
        assert len(rows) == 4
        for r in rows:
            assert r["superseded_fraction_of_ceiling"] == pytest.approx(
                r["transfer_r2_committed"] / mean
            )


class TestCollectLambdas:
    def test_default_on_and_flag_off_suppresses(self):
        # #1887 defaults flip: collect_lambdas now defaults TRUE (selected-
        # lambda logging into cell payloads by default); the prior default-off
        # surface stays reachable via the explicit collect_lambdas=False.
        rng = np.random.default_rng(0)
        n, n_layers, d = 30, 2, 5
        X = rng.normal(size=(n, n_layers, d)).astype(np.float32)
        Y = (0.5 * X + rng.normal(scale=0.1, size=(n, n_layers, d))).astype(np.float32)
        conv = np.array([f"c{i}" for i in range(n)])
        kw = dict(n_folds=3, seed=0, null_draws=0, collect_cosines=False)
        sw_off = fit825.heldout_r2_sweep(X, Y, conv, collect_lambdas=False, **kw)
        assert sw_off["gcv_lambda"] is None  # explicit off preserves prior surface
        sw_on = fit825.heldout_r2_sweep(X, Y, conv, **kw)  # default ON (#1887)
        lam = sw_on["gcv_lambda"]
        assert lam.shape == (n_layers, 3)
        assert np.isfinite(lam).all()
        grid = {float(v) for v in fit825.LAMBDAS}
        assert all(float(v) in grid for v in lam.ravel())
        # lambda collection must not perturb the fit itself
        np.testing.assert_array_equal(sw_off["r2_obs"], sw_on["r2_obs"])


class TestDescope:
    def test_no_descope_under_budget(self):
        ns, info = pcms.apply_descope([1000, 1500, 1982, 2500, 3000], (931,), 1000, 1.0, 1.5)
        assert ns == [1000, 1500, 1982, 2500, 3000]
        assert info["dropped_ns"] == []

    def test_subset_grid_without_decision_n_is_valid(self):
        # The smoke grid (ns=[1000]) never contained 1982; the anchor
        # invariant is conditional on grid membership (caught live by
        # smoke run 1c, 2026-07-04).
        ns, info = pcms.apply_descope([1000], (931, 932), 1000, 233.3, 1.5)
        assert ns == [1000]
        assert info["dropped_ns"] == []

    def test_descope_priority_never_drops_anchors_or_seeds(self):
        # An absurdly slow first cell forces both descope tiers.
        ns, info = pcms.apply_descope(
            [1000, 1500, 1982, 2500, 3000], tuple(pcms.SEEDS_DEFAULT), 1000, 3600.0, 1.5
        )
        assert ns == [1000, 1982]
        assert set(info["dropped_ns"]) == {1500, 2500, 3000}


class TestDrawSubsample:
    """The seed-degeneracy fix: on the all-singleton chat store the draws
    must actually vary with seed (smoke run 1d caught byte-identical
    seed-931/932 subsets from `group_stratified_subsample`)."""

    def _ids(self, n):
        return np.array([f"conv{i:05d}" for i in range(n)])

    def test_seeded_draws_differ_and_are_deterministic(self):
        ids = self._ids(200)
        a = pcms.draw_subsample(ids, 50, 931)
        b = pcms.draw_subsample(ids, 50, 932)
        assert not np.array_equal(a, b)  # the degeneracy this fix removes
        assert np.array_equal(a, pcms.draw_subsample(ids, 50, 931))
        assert len(a) == 50 and np.array_equal(a, np.sort(a))
        assert len(np.unique(a)) == 50

    def test_full_draw_is_identity(self):
        ids = self._ids(20)
        assert np.array_equal(pcms.draw_subsample(ids, 20, 931), np.arange(20))

    def test_group_structure_fails_loud(self):
        ids = np.array(["c0", "c0", "c1", "c2"])
        with pytest.raises(AssertionError, match="all-singleton"):
            pcms.draw_subsample(ids, 2, 931)


class TestCheckpointJsonlRobustness:
    def test_unicode_line_separator_in_row_survives(self, tmp_path):
        # U+2028 inside a JSON string must not shred the reader (gotcha:
        # splitlines() splits on Unicode line boundaries; the reader iterates
        # the file instead).
        ckpt = tmp_path / "cells.jsonl"
        fp = "abcdef012345"
        note = "a" + "\u2028" + "b"  # U+2028 LINE SEPARATOR
        row = {"protocol_fingerprint": fp, "seed": 931, "n": 1000, "note": note}
        ckpt.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        by_key = pcms.load_checkpoint(ckpt, fp)
        assert by_key[(931, 1000)]["note"] == note
