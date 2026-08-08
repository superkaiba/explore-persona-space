"""#1900 pin: anchor-mix row-count floor gate (plan §12.1 v6).

CPU-tiny, fast: imports the GPU driver module only — the gate
(`check_anchor_mix_floor`) is pure arithmetic + logging; torch is
function-level deferred in the module, so no model/GPU work runs.
Boundaries pinned per code-review v4 Minor 3: n=7 hard-kill;
n=8/20/39 LOUD WARN + persisted low_n_flag True; n=40 clean False.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue1900_gpu as G  # noqa: E402


def test_hard_floor_n7_raises():
    """Below the hard floor (n=7 < 8) the gate kills loud with the plan citation."""
    with pytest.raises(AssertionError) as exc:
        G.check_anchor_mix_floor("m", 7)
    assert "plan §12.1 v6" in str(exc.value)


@pytest.mark.parametrize("n", [8, 20, 39])
def test_low_n_band_warns_and_flags(n: int, caplog: pytest.LogCaptureFixture):
    """8 <= n < 40 returns low_n_flag True and emits the LOUD WARN (never a kill)."""
    with caplog.at_level(logging.WARNING, logger="issue1900.gpu"):
        assert G.check_anchor_mix_floor("m", n) is True
    warns = [r for r in caplog.records if "low-n flag set" in r.getMessage()]
    assert warns, "expected the [anchors] low-n WARN line"
    # Call-site-neutral text (v4 Minor 1): no split-half-persistence claim —
    # two of the three gated call sites persist no split-half fields.
    assert all("split-half" not in r.getMessage() for r in warns)


def test_n40_clean_no_warn(caplog: pytest.LogCaptureFixture):
    """At the boundary n=40 the flag is False and no low-n WARN is emitted."""
    with caplog.at_level(logging.WARNING, logger="issue1900.gpu"):
        assert G.check_anchor_mix_floor("m", 40) is False
    assert not [r for r in caplog.records if "low-n" in r.getMessage()]


def test_floor_constants_pinned():
    """Plan §12.1 (v6) constants: hard floor 8, low-n band ceiling 40."""
    assert G.ANCHOR_HARD_FLOOR_ROWS == 8
    assert G.ANCHOR_LOW_N_ROWS == 40
