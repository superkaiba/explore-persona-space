"""Tests for the round-7 smoke-mode carve-out in Phase 1 truncation guard.

Round-7 review fix: at tiny N (`--smoke-n 5` -> 10 generations), a single
verbose villain/pirate response = 10% truncation > the production 5%
threshold. Round-6 hard-raised on this and aborted phase 1 BEFORE
writing R_canon_test.json, cascading into phase 2-check / 4 / 4.5
R_canon-load failures (correctly fail-loud via the override).

The fix splits production behavior (smoke_n=0, hard-raise > 5%) from
smoke behavior (smoke_n>0, WARNING-and-continue), so both splits get
written even when a tiny-N run has a single long response.

Production behavior (smoke_n=0) is unchanged: same 5% threshold, same
RuntimeError. Tests cover both paths.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def phase1_mod():
    """Load `scripts/i464_phase1_generate_R.py` as a module."""
    spec = importlib.util.spec_from_file_location(
        "i464_phase1_generate_R",
        REPO_ROOT / "scripts" / "i464_phase1_generate_R.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Production path (smoke_n=0): strict 5% guard, must raise ──────────


def test_production_zero_truncation_passes(phase1_mod):
    """0 truncations always pass regardless of mode."""
    phase1_mod._check_truncation_rate(
        split="train", n_truncated=0, n_total_rows=60, max_new_tokens=1024, smoke_n=0
    )  # should not raise


def test_production_under_threshold_passes(phase1_mod):
    """Production: 2/60 = 3.3% < 5% -> pass."""
    phase1_mod._check_truncation_rate(
        split="train", n_truncated=2, n_total_rows=60, max_new_tokens=1024, smoke_n=0
    )


def test_production_at_threshold_passes(phase1_mod):
    """Boundary: exactly 5% (3/60) is the strict threshold; the guard's
    condition is `> 5%`, so 5.0% should pass."""
    phase1_mod._check_truncation_rate(
        split="train", n_truncated=3, n_total_rows=60, max_new_tokens=1024, smoke_n=0
    )


def test_production_above_threshold_raises(phase1_mod):
    """Production: 4/60 = 6.7% > 5% -> RuntimeError (the canonical guard)."""
    with pytest.raises(RuntimeError, match="truncation rate"):
        phase1_mod._check_truncation_rate(
            split="train", n_truncated=4, n_total_rows=60, max_new_tokens=1024, smoke_n=0
        )


def test_production_error_includes_max_new_tokens(phase1_mod):
    """Production error message must surface the current --max-new-tokens for the operator."""
    with pytest.raises(RuntimeError) as excinfo:
        phase1_mod._check_truncation_rate(
            split="test", n_truncated=10, n_total_rows=60, max_new_tokens=512, smoke_n=0
        )
    msg = str(excinfo.value)
    assert "512" in msg
    assert "split=test" in msg


# ── Smoke path (smoke_n>0): warn-and-continue, must NOT raise ─────────


def test_smoke_zero_truncation_passes_silently(phase1_mod, caplog):
    """Smoke mode + 0 truncations: pass without any WARNING."""
    with caplog.at_level(logging.WARNING):
        phase1_mod._check_truncation_rate(
            split="train", n_truncated=0, n_total_rows=10, max_new_tokens=1024, smoke_n=5
        )
    assert not any("truncation rate" in r.message for r in caplog.records)


def test_smoke_one_truncation_warns_and_continues(phase1_mod, caplog):
    """The bug round-7 fixes: smoke=5, 1/10 = 10% > 5% -> WARN-and-continue.

    Round-6 would have raised here, aborting phase 1 before writing
    R_canon_test.json. Round-7's carve-out lets the script proceed.
    """
    with caplog.at_level(logging.WARNING):
        phase1_mod._check_truncation_rate(
            split="train", n_truncated=1, n_total_rows=10, max_new_tokens=1024, smoke_n=5
        )  # must NOT raise
    warnings = [r for r in caplog.records if "SMOKE mode" in r.message]
    assert len(warnings) == 1, f"expected exactly 1 SMOKE warning; got {len(warnings)}"
    msg = warnings[0].message
    assert "10.0%" in msg
    assert "smoke_n=5" in msg
    assert "split=train" in msg
    assert "1/10" in msg


def test_smoke_extreme_truncation_warns_not_raises(phase1_mod, caplog):
    """Even 83% truncation (the original round-4 failure mode at
    `--max-new-tokens 128`) warns and continues in smoke mode."""
    with caplog.at_level(logging.WARNING):
        phase1_mod._check_truncation_rate(
            split="test",
            n_truncated=5,
            n_total_rows=6,
            max_new_tokens=128,
            smoke_n=3,
        )  # must NOT raise
    warnings = [r for r in caplog.records if "SMOKE mode" in r.message]
    assert warnings, "expected SMOKE warning at 83% truncation"


def test_smoke_at_threshold_no_warn(phase1_mod, caplog):
    """Smoke mode + exactly 5% (1/20) is on/below the threshold ->
    no warning (the early-return path)."""
    with caplog.at_level(logging.WARNING):
        phase1_mod._check_truncation_rate(
            split="train", n_truncated=1, n_total_rows=20, max_new_tokens=1024, smoke_n=10
        )
    assert not any("SMOKE mode" in r.message for r in caplog.records)


def test_smoke_warning_mentions_production_difference(phase1_mod, caplog):
    """The warning message must remind the operator that production
    (smoke_n=0) still hard-raises — so they don't conclude the guard
    is broken when they switch to production."""
    with caplog.at_level(logging.WARNING):
        phase1_mod._check_truncation_rate(
            split="train", n_truncated=2, n_total_rows=10, max_new_tokens=1024, smoke_n=5
        )
    msg = next(r.message for r in caplog.records if "SMOKE mode" in r.message)
    assert "Production" in msg or "production" in msg
