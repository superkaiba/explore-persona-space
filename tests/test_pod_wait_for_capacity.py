"""Tests for the unbounded wait-for-capacity retry loop in
``scripts/pod_lifecycle.py``.

What's being tested
-------------------
- ``create_pod_with_wait_for_capacity`` retries on every ``RunPodNoCapacityError``
  and returns the first ``PodInfo`` from the wrapped ``create_pod`` primitive.
- It does NOT retry on a generic ``RunPodError`` (auth, bad config, transport-
  budget-exhausted, empty-gpu-list). Those propagate immediately per the
  CLAUDE.md "fail fast — never hide failures" rule.
- The loop is UNBOUNDED: it succeeds on attempt 5+ when capacity stays out for
  the first several attempts. (No cap on attempts; the test mocks an arbitrary
  succeed-on-attempt-N pattern.)
- ``KeyboardInterrupt`` during the sleep propagates so the operator can Ctrl-C
  cleanly.
- The backoff helper produces values inside the exponential-jittered window.

All ``time.sleep`` calls are monkeypatched so the test suite runs instantly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_lifecycle  # noqa: E402
from pod_lifecycle import (  # noqa: E402
    _wait_for_capacity_backoff_secs,
    create_pod_with_wait_for_capacity,
)
from runpod_api import PodInfo, RunPodError, RunPodNoCapacityError  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Never actually sleep during backoff in tests."""
    monkeypatch.setattr(pod_lifecycle.time, "sleep", lambda _secs: None)


def _make_pod_info(pod_id: str = "p1", name: str = "pod-1") -> PodInfo:
    """Synthesize a minimal PodInfo for the success-path return."""
    return PodInfo(
        pod_id=pod_id,
        name=name,
        desired_status="RUNNING",
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host=None,
        ssh_port=None,
        created_at="2026-06-07T00:00:00Z",
    )


def _make_create_pod_stub(monkeypatch, outcomes: list):
    """Patch ``pod_lifecycle.create_pod`` to a recorder consuming ``outcomes``
    one per call. Each entry is either an Exception (raised) or a PodInfo
    (returned). Returns a recorder so the test can assert call count.
    """

    class _Rec:
        def __init__(self):
            self.calls = 0
            self.kwargs_seen: list[dict] = []

        def __call__(self, **kwargs):
            self.calls += 1
            self.kwargs_seen.append(kwargs)
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    rec = _Rec()
    monkeypatch.setattr(pod_lifecycle, "create_pod", rec)
    return rec


def test_retries_then_succeeds(monkeypatch):
    """Two no-capacity errors, then success — loop returns the success PodInfo."""
    info = _make_pod_info()
    rec = _make_create_pod_stub(
        monkeypatch,
        [
            RunPodNoCapacityError("no capacity attempt 1"),
            RunPodNoCapacityError("no capacity attempt 2"),
            info,
        ],
    )

    out = create_pod_with_wait_for_capacity(
        name="pod-1",
        gpu_type="H100",
        gpu_count=1,
        volume_gb=200,
        container_disk_gb=50,
    )

    assert out is info
    assert rec.calls == 3
    # Every attempt got the same args (no mutation of the request between retries).
    for kw in rec.kwargs_seen:
        assert kw["name"] == "pod-1"
        assert kw["gpu_type"] == "H100"
        assert kw["gpu_count"] == 1


def test_non_capacity_runpod_error_propagates_no_retry(monkeypatch):
    """A generic RunPodError (auth / bad config / transport budget) must NOT retry."""
    rec = _make_create_pod_stub(
        monkeypatch,
        [
            RunPodError("HTTP 401: invalid api key"),
            # Sentinel: if the loop wrongly retries, the next call would succeed
            # and the test would silently pass. We assert calls == 1 below to
            # catch that regression.
            _make_pod_info(),
        ],
    )

    with pytest.raises(RunPodError) as exc:
        create_pod_with_wait_for_capacity(
            name="pod-1",
            gpu_type="H100",
            gpu_count=1,
            volume_gb=200,
            container_disk_gb=50,
        )

    assert "401" in str(exc.value)
    assert not isinstance(exc.value, RunPodNoCapacityError)
    assert rec.calls == 1  # did NOT retry — fail-fast on non-capacity


def test_loop_is_unbounded_succeeds_on_attempt_5(monkeypatch):
    """No max-attempts cap: succeeds even when the first 4 attempts hit
    no-capacity. (Stand-in for "unbounded" — we don't loop literally forever
    in a unit test, but we verify the wrapper handles >>4 attempts without
    raising on its own.)"""
    info = _make_pod_info()
    rec = _make_create_pod_stub(
        monkeypatch,
        [
            RunPodNoCapacityError("attempt 1"),
            RunPodNoCapacityError("attempt 2"),
            RunPodNoCapacityError("attempt 3"),
            RunPodNoCapacityError("attempt 4"),
            info,
        ],
    )

    out = create_pod_with_wait_for_capacity(
        name="pod-1",
        gpu_type="H100",
        gpu_count=1,
        volume_gb=200,
        container_disk_gb=50,
    )
    assert out is info
    assert rec.calls == 5


def test_loop_is_unbounded_many_attempts(monkeypatch):
    """50 consecutive no-capacity errors followed by success — the loop has no
    attempt cap. (The transport-layer ``graphql`` retry IS capped at
    GRAPHQL_MAX_ATTEMPTS=4, but THIS policy layer is deliberately uncapped.)"""
    info = _make_pod_info()
    outcomes = [RunPodNoCapacityError(f"attempt {i}") for i in range(50)] + [info]
    rec = _make_create_pod_stub(monkeypatch, outcomes)

    out = create_pod_with_wait_for_capacity(
        name="pod-1",
        gpu_type="H100",
        gpu_count=1,
        volume_gb=200,
        container_disk_gb=50,
    )
    assert out is info
    assert rec.calls == 51


def test_keyboard_interrupt_during_sleep_propagates(monkeypatch):
    """SIGINT during the backoff sleep must propagate, not be swallowed."""

    def _raise_interrupt(_secs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pod_lifecycle.time, "sleep", _raise_interrupt)
    rec = _make_create_pod_stub(
        monkeypatch,
        [RunPodNoCapacityError("nope"), _make_pod_info()],
    )

    with pytest.raises(KeyboardInterrupt):
        create_pod_with_wait_for_capacity(
            name="pod-1",
            gpu_type="H100",
            gpu_count=1,
            volume_gb=200,
            container_disk_gb=50,
        )
    # Saw the first attempt and tried to sleep — did NOT make a second
    # create_pod call after the interrupt.
    assert rec.calls == 1


def test_first_attempt_success_does_not_sleep(monkeypatch):
    """When the very first create_pod call succeeds, no backoff sleep fires."""
    sleep_calls: list[float] = []
    monkeypatch.setattr(pod_lifecycle.time, "sleep", lambda secs: sleep_calls.append(secs))
    info = _make_pod_info()
    _make_create_pod_stub(monkeypatch, [info])

    out = create_pod_with_wait_for_capacity(
        name="pod-1",
        gpu_type="H100",
        gpu_count=1,
        volume_gb=200,
        container_disk_gb=50,
    )
    assert out is info
    assert sleep_calls == []


def test_backoff_window_grows_and_caps(monkeypatch):
    """Backoff window grows ~exponentially and is capped at the ceiling."""
    s1 = _wait_for_capacity_backoff_secs(1)
    s_big = _wait_for_capacity_backoff_secs(50)
    assert 0.0 <= s1 <= pod_lifecycle.WAIT_FOR_CAPACITY_BACKOFF_BASE_SECS
    assert 0.0 <= s_big <= pod_lifecycle.WAIT_FOR_CAPACITY_BACKOFF_CAP_SECS
    # Sanity: cap is strictly larger than base so the window actually grew.
    assert (
        pod_lifecycle.WAIT_FOR_CAPACITY_BACKOFF_CAP_SECS
        > pod_lifecycle.WAIT_FOR_CAPACITY_BACKOFF_BASE_SECS
    )


def test_backoff_does_not_overflow_at_huge_attempt():
    """Regression: ``2 ** (attempt - 1)`` overflows Python float past
    attempt ~1025, raising ``OverflowError: int too large to convert to
    float`` and CRASHING the unbounded retry loop after ~3.5 days of
    waiting at the 10-min ceiling. The exponent must be clamped so the
    loop survives arbitrarily many attempts — the whole point of the
    loop is "retry indefinitely." Before the clamp this call raised
    ``OverflowError``; after the clamp it returns a finite jittered value
    inside the cap.
    """
    import math

    out = _wait_for_capacity_backoff_secs(10_000)
    assert math.isfinite(out)
    assert out > 0.0
    assert out <= pod_lifecycle.WAIT_FOR_CAPACITY_BACKOFF_CAP_SECS
    # Even more extreme attempt count: clamp must still hold.
    out_huge = _wait_for_capacity_backoff_secs(1_000_000)
    assert math.isfinite(out_huge)
    assert 0.0 < out_huge <= pod_lifecycle.WAIT_FOR_CAPACITY_BACKOFF_CAP_SECS


def test_autonomous_session_helper_truthiness(monkeypatch):
    """``_autonomous_session()`` mirrors task.py's parse exactly. The falsy
    set ({"", "0", "false", "no"}) must NOT enable autonomous mode."""
    for falsy in ("", "0", "false", "FALSE", "no", "No"):
        monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", falsy)
        assert pod_lifecycle._autonomous_session() is False, falsy
    for truthy in ("1", "true", "yes", "TRUE", "Y"):
        monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", truthy)
        assert pod_lifecycle._autonomous_session() is True, truthy
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    assert pod_lifecycle._autonomous_session() is False


# ─── per-process attempt budget (refs #572) ──────────────────────────────────


class _TickingMonotonic:
    """Fake ``time.monotonic`` advancing ``step`` seconds per call."""

    def __init__(self, step: float):
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def test_budget_trips_with_still_waiting(monkeypatch):
    """When elapsed + the planned sleep would exceed the per-process budget,
    the loop raises WaitForCapacityStillWaiting BEFORE sleeping past it
    (refs #572: one process attempt must stay under the ~50 min bg-kill
    window; the CLI converts the exception into exit 75 + a structured
    STILL-WAITING line so the orchestrator re-runs the command)."""
    monkeypatch.setenv("EPM_WAIT_FOR_CAPACITY_BUDGET_SECS", "60")
    monkeypatch.setattr(pod_lifecycle.time, "monotonic", _TickingMonotonic(step=50.0))
    rec = _make_create_pod_stub(
        monkeypatch,
        [
            RunPodNoCapacityError("no capacity attempt 1"),
            RunPodNoCapacityError("no capacity attempt 2"),
            _make_pod_info(),  # sentinel — must never be reached
        ],
    )

    with pytest.raises(pod_lifecycle.WaitForCapacityStillWaiting) as exc:
        create_pod_with_wait_for_capacity(
            name="pod-1",
            gpu_type="H100",
            gpu_count=1,
            volume_gb=200,
            container_disk_gb=50,
        )

    assert exc.value.verb == "provision"
    assert exc.value.name == "pod-1"
    assert exc.value.attempts >= 1
    assert rec.calls >= 1
    assert rec.calls < 3  # never reached the success sentinel
    # The exception is a RunPodError subclass but must NOT be confused with
    # the retryable classes (it is raised FROM the loop, never caught by it).
    assert isinstance(exc.value, RunPodError)


def test_budget_zero_disables_the_cap(monkeypatch):
    """EPM_WAIT_FOR_CAPACITY_BUDGET_SECS=0 disables the budget — the loop
    keeps retrying (legacy unbounded behavior) and returns the success."""
    monkeypatch.setenv("EPM_WAIT_FOR_CAPACITY_BUDGET_SECS", "0")
    monkeypatch.setattr(pod_lifecycle.time, "monotonic", _TickingMonotonic(step=10_000.0))
    info = _make_pod_info()
    rec = _make_create_pod_stub(
        monkeypatch,
        [
            RunPodNoCapacityError("no capacity attempt 1"),
            RunPodNoCapacityError("no capacity attempt 2"),
            info,
        ],
    )

    out = create_pod_with_wait_for_capacity(
        name="pod-1",
        gpu_type="H100",
        gpu_count=1,
        volume_gb=200,
        container_disk_gb=50,
    )

    assert out is info
    assert rec.calls == 3


def test_emit_still_waiting_exits_75(capsys):
    """The CLI conversion prints the structured STILL-WAITING line on both
    streams and exits EXIT_STILL_WAITING (75, EX_TEMPFAIL)."""
    exc = pod_lifecycle.WaitForCapacityStillWaiting(
        verb="provision", name="pod-9", attempts=7, elapsed_secs=2712.0
    )
    with pytest.raises(SystemExit) as se:
        pod_lifecycle._emit_still_waiting_and_exit(exc)
    assert se.value.code == pod_lifecycle.EXIT_STILL_WAITING == 75
    captured = capsys.readouterr()
    for stream in (captured.out, captured.err):
        assert "[wait-for-capacity] STILL-WAITING" in stream
        assert "pod-9" in stream
        assert "RE-RUN THE SAME COMMAND" in stream
