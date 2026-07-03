"""Tests for the poll_pipeline under-parallelization warning decision core.

The [gpu-underparallel-warning] (plan §3, workflow v2) fires once per RUN when
< 50% of the provisioned GPUs are busy for >= GPU_UNDERPARALLEL_WARNING_MIN
minutes during a healthy run. Distinct from the #873 [gpu-width-advisory].
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import poll_pipeline as pp

_T0 = 1_000_000


def _u(gpu_util, *, since=_T0, warned=False, dt_min=16, status="running", warning_min=15):
    return pp._gpu_underparallel_update(
        status=status,
        gpu_util=gpu_util,
        prev_since_epoch=since,
        already_warned=warned,
        now_epoch=_T0 + dt_min * 60,
        warning_min=warning_min,
    )


def test_majority_idle_past_threshold_posts():
    u = _u("90,80,0,0,0,0,0,0")  # 2 of 8 busy = 25% < 50%
    assert u.should_post and u.n_busy == 2 and u.n_gpus == 8


def test_per_run_dedup_when_already_warned():
    u = _u("90,80,0,0,0,0,0,0", warned=True)
    assert not u.should_post and u.since_epoch == _T0  # span kept, no re-post


def test_at_least_half_busy_resets():
    u = _u("90,90,90,90,90,0,0,0")  # 5 of 8 = 62.5% >= 50%
    assert not u.should_post and u.since_epoch == 0


def test_all_idle_resets_not_our_domain():
    u = _u("0,0,0,0")  # n_busy == 0 -> idle advisory / CPU phase, not this
    assert not u.should_post and u.since_epoch == 0


def test_single_gpu_resets():
    assert not _u("0").should_post


def test_span_too_short_arms_but_no_post():
    u = _u("90,0,0,0", dt_min=10)  # 10 min < 15
    assert not u.should_post and u.since_epoch == _T0


def test_unknown_sample_fail_safe_resets():
    u = _u("unknown")
    assert not u.should_post and u.since_epoch == 0


def test_non_running_status_resets():
    u = _u("90,0,0,0", status="stalled")
    assert not u.should_post and u.since_epoch == 0


def test_disabled_when_warning_min_zero():
    assert not _u("90,0,0,0", warning_min=0).should_post


def test_span_starts_at_now_when_no_prior():
    u = _u("90,0,0,0", since=0)
    assert u.since_epoch == _T0 + 16 * 60  # fresh span this tick
