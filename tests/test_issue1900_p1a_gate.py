"""#1900 pin: P1a frame-free adapter-identity gate (plan v7 §4 P1a / §7 kill crit 3).

CPU-tiny, fast: imports the GPU driver module only — `p1a_gate_record` is
pure arithmetic (torch is function-level deferred in the module, so no
model/GPU work runs). Regression for crash-fix round 6 (job 16092): the
former ±1-nat CROSS-FRAME equality against the #1481 manifest
`delta_logp_mean` (a checkpoint-SELECTION read) failed on the expected
22.436-nat training-row read of a correctly-applied adapter; the
frame-free gate must PASS that exact shape and keep failing the genuine
identity-broken shapes (Δ≈0 training rows; zero corpus |Δz|).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue1900_gpu as G  # noqa: E402

JOB_16092 = dict(
    arm_id="mk-pers-con-lr5e6-s42",
    n_mix_rows=50,
    median_training_delta_logp=22.436,  # the measured job-16092 value
    manifest_selection_delta_logp=6.346,  # #1481 selection-frame manifest value
    median_abs_delta_z_marker_corpus=0.7,
    median_corpus_delta_logp=0.11,
)


def test_job_16092_shape_passes_frame_free_gate():
    """The exact job-16092 read (22.436 vs manifest 6.346) PASSES — the pre-fix
    cross-frame equality assert raised on precisely this input."""
    rec = G.p1a_gate_record(**JOB_16092)
    assert rec["median_training_row_delta_logp"] == pytest.approx(22.436)
    # Both frames recorded side-by-side with the frame note; never compared.
    assert rec["manifest_selection_frame_delta_logp_mean"] == pytest.approx(6.346)
    assert "frames" in rec["frame_note"]
    assert rec["min_delta_floor_nats"] == G.ADAPTER_SMOKE_MIN_DELTA_NATS
    # The corpus sanity read is persisted (no equality claim on it either).
    assert rec["median_corpus_delta_logp"] == pytest.approx(0.11)


@pytest.mark.parametrize("median_delta", [0.0, 1.999, -3.0])
def test_below_floor_training_delta_kills_loud(median_delta: float):
    """Direction+floor: a wrong/no-op adapter (Δ logP < +2 nats on its OWN
    training rows) hard-fails with the plan citation."""
    with pytest.raises(AssertionError) as exc:
        G.p1a_gate_record(**{**JOB_16092, "median_training_delta_logp": median_delta})
    assert "frame-free gate" in str(exc.value)


def test_zero_corpus_dz_kills_loud():
    """Adapter-not-applied signature: median |Δ z_marker| == 0 on corpus rows."""
    with pytest.raises(AssertionError) as exc:
        G.p1a_gate_record(**{**JOB_16092, "median_abs_delta_z_marker_corpus": 0.0})
    assert "adapter not applied" in str(exc.value)


def test_floor_constant_pinned():
    """Plan v7 §4 P1a (frame-corrected 2026-07-31): floor = +2 nats."""
    assert G.ADAPTER_SMOKE_MIN_DELTA_NATS == 2.0
