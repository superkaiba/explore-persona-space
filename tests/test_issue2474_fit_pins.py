"""Regression pins for issue-2474's fit driver (round-2 BLOCKER fixes).

Pins (each fails on the r1 code, passes post-fix):
  * ``_scores_fingerprint`` keys bundles on PARENT-RELATIVE paths, so per-condition
    ``mu.pt`` files can never collide onto one dict entry (r1 g1 Major 2 /
    Codex ``score-fingerprint-collision``), and the cardinality assert fires.
  * ``_ceiling_cell_accounting`` counts ABSENT cells as zero kept (r1 Codex
    ``harvest-zero-cell-gap``), reconciles kept + dropped == slots, and refuses
    out-of-range cell indices.
  * ``_assert_close_banked`` raises on a NaN recompute (r1 g1 Minor: the old
    ``abs(a-b) > tol`` form is False on NaN, silently PASSing drift).
  * ``_write_done_sentinel`` emits a poll_pipeline-conformant envelope (r1 g3
    concern 4) — round-tripped through the REAL ``poll_pipeline._parse_sentinel``
    — and routes smoke runs to the SMOKE tree, never /workspace/logs (r1 g1
    Major 1).
  * ``PHASES`` registry membership (the smoke-architecture arm set of record).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import issue2474_fit as fit


def _touch(p: Path, payload: bytes = b"x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)
    return p


# ---------------------------------------------------------------------------
# _scores_fingerprint — parent-relative keys, no per-condition mu.pt collision
# ---------------------------------------------------------------------------
def test_scores_fingerprint_parent_relative_keys_no_collision(tmp_path):
    base = tmp_path / "capture" / "predictor_captures"
    _touch(base / "base_em" / "grid.pt")
    _touch(base / "base_em" / "ceiling.pt")
    _touch(base / "base_mu_condA" / "mu.pt", b"aaa")
    _touch(base / "base_mu_condB" / "mu.pt", b"bbbb")
    comp_dir = tmp_path / "comps"
    _touch(comp_dir / "base_L00.npz")
    cfg = {
        "capture_dir": tmp_path / "capture",
        "conds": {"em": ("condA", "condB")},
        "comp_dir": str(comp_dir),
    }
    args = argparse.Namespace(parent_sha="deadbeef")
    fp = fit._scores_fingerprint(cfg, "em", args)
    # Cardinality: 2 + n_mu_bundles UNIQUE entries (the r1 bare-filename keys
    # collapsed both mu.pt files onto ONE "mu.pt" entry -> len == 3, not 4).
    assert len(fp["bundles"]) == 4, sorted(fp["bundles"])
    assert "base_mu_condA/mu.pt" in fp["bundles"]
    assert "base_mu_condB/mu.pt" in fp["bundles"]
    assert fp["bundles"]["base_mu_condA/mu.pt"] != fp["bundles"]["base_mu_condB/mu.pt"]
    assert fp["v"] == 2
    assert "base_L00.npz" in fp["components"]


# ---------------------------------------------------------------------------
# _ceiling_cell_accounting — absent cells count zero; reconciliation asserts
# ---------------------------------------------------------------------------
def _meta(cells_kept: dict[int, int]) -> list[dict]:
    rows = []
    for ci, n in cells_kept.items():
        rows += [{"cell_idx": ci}] * n
    return rows


def test_ceiling_accounting_wholly_absent_cell_trips_floor():
    # 4 cells x 3 rollouts = 12 slots; cell 3 wholly absent (its 3 slots dropped).
    with pytest.raises(RuntimeError, match="min kept/cell 0"):
        fit._ceiling_cell_accounting(
            _meta({0: 3, 1: 3, 2: 3}),
            n_cells_expected=4,
            n_rollouts_expected=3,
            drop_stats={"n_slots": 12, "n_empty_after_retries": 3, "n_capture_dropped": 0},
            max_rows=12,
            min_kept_per_cell=2,
            max_drop_frac=0.5,
            ctx="test",
        )


def test_ceiling_accounting_reconcile_failure_raises():
    with pytest.raises(RuntimeError, match="does not reconcile"):
        fit._ceiling_cell_accounting(
            _meta({0: 3, 1: 3, 2: 3, 3: 2}),
            n_cells_expected=4,
            n_rollouts_expected=3,
            drop_stats={"n_slots": 12, "n_empty_after_retries": 0, "n_capture_dropped": 0},
            max_rows=12,
            min_kept_per_cell=2,
            max_drop_frac=0.5,
            ctx="test",
        )


def test_ceiling_accounting_out_of_range_cell_raises():
    with pytest.raises(RuntimeError, match="outside the expected cell set"):
        fit._ceiling_cell_accounting(
            _meta({0: 3, 7: 3}),
            n_cells_expected=4,
            n_rollouts_expected=3,
            drop_stats={"n_slots": 12, "n_empty_after_retries": 6, "n_capture_dropped": 0},
            max_rows=12,
            min_kept_per_cell=2,
            max_drop_frac=0.6,
            ctx="test",
        )


def test_ceiling_accounting_happy_path():
    out = fit._ceiling_cell_accounting(
        _meta({0: 3, 1: 2, 2: 3, 3: 3}),
        n_cells_expected=4,
        n_rollouts_expected=3,
        drop_stats={"n_slots": 12, "n_empty_after_retries": 1, "n_capture_dropped": 0},
        max_rows=12,
        min_kept_per_cell=2,
        max_drop_frac=0.5,
        ctx="test",
    )
    assert out == {
        "n_kept_rows": 11,
        "n_slots": 12,
        "n_dropped_total": 1,
        "min_kept_per_cell": 2,
        "n_cells_expected": 4,
        "n_absent_cells": 0,
    }


# ---------------------------------------------------------------------------
# _assert_close_banked — NaN-safe recompute assert
# ---------------------------------------------------------------------------
def test_assert_close_banked_nan_raises():
    with pytest.raises(RuntimeError, match="provenance drift"):
        fit._assert_close_banked(math.nan, 1.0, "test/nan")


def test_assert_close_banked_close_passes_and_far_raises():
    fit._assert_close_banked(1.0 + 5e-7, 1.0, "test/close")
    with pytest.raises(RuntimeError):
        fit._assert_close_banked(1.1, 1.0, "test/far")


# ---------------------------------------------------------------------------
# Done sentinel — poll_pipeline-conformant envelope + smoke-tree routing
# ---------------------------------------------------------------------------
def _sentinel_args(**over):
    ns = argparse.Namespace(log_dir=None)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_smoke_sentinel_routes_to_smoke_tree_and_parses(tmp_path):
    import poll_pipeline  # scripts/ is on sys.path

    cfg = {"synthetic": True, "data_root": tmp_path}
    fit._write_done_sentinel(_sentinel_args(), cfg, ["out/a.json"])
    p = tmp_path / "logs" / "issue-2474-fit-smoke.done.json"
    assert p.is_file(), "smoke sentinel must land under the SMOKE tree (r1 Major 1)"
    payload = json.loads(p.read_text())
    for k in poll_pipeline._SENTINEL_REQUIRED_KEYS:
        assert k in payload, f"missing poll_pipeline required key {k!r}"
    assert payload["sentinel_schema_version"] == poll_pipeline.SENTINEL_SCHEMA_VERSION_SUPPORTED
    assert payload["kind"] == "epm:progress"
    assert payload["version"] is None  # drain posts at max+1
    parsed = poll_pipeline._parse_sentinel(str(p), p.read_text())
    assert isinstance(parsed, dict), "REAL poller _parse_sentinel must accept the envelope"


def test_production_sentinel_honors_log_dir(tmp_path):
    cfg = {"synthetic": False, "data_root": tmp_path}
    fit._write_done_sentinel(_sentinel_args(log_dir=str(tmp_path / "plogs")), cfg, ["x"])
    p = tmp_path / "plogs" / "issue-2474-fit.done.json"
    assert p.is_file()
    assert json.loads(p.read_text())["sentinel_schema_version"] == 1


# ---------------------------------------------------------------------------
# PHASES registry — the smoke-architecture arm set of record
# ---------------------------------------------------------------------------
def test_phases_registry_members():
    assert sorted(fit.PHASES) == [
        "all",
        "harvest-verify",
        "pilot",
        "refit",
        "scores",
        "smoke",
        "stats",
        "upload",
    ]
