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

import argparse
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_lifecycle  # noqa: E402
import runpod_api  # noqa: E402
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


# ─── CPU provision wiring through the shared wait loop (#2238) ───────────────
#
# Pre-#2238 cmd_provision's CPU branch `return`ed BEFORE the wait_for_capacity
# flag was ever read, so `--wait-for-capacity` (and the autonomous auto-enable)
# was structurally inert on every CPU intent — the autonomous log line promised
# unbounded retry on a path that failed fast. These tests pin the WIRING (the
# retry actually happens through cmd_provision's CPU branch), not merely the
# flag's presence.


def _make_cpu_pod_info(name: str = "pod-2238") -> PodInfo:
    """Synthesize a minimal CPU PodInfo for the success-path return."""
    return PodInfo(
        pod_id="cpupod-1",
        name=name,
        desired_status="RUNNING",
        gpu_count=0,
        gpu_type_id="",
        ssh_host=None,
        ssh_port=None,
        created_at="2026-08-11T00:00:00Z",
    )


def _make_create_cpu_pod_stub(monkeypatch, outcomes: list):
    """Patch ``pod_lifecycle.create_cpu_pod`` with a SIGNATURE-CONFORMANT
    recorder (``create_autospec`` against the real ``runpod_api.create_cpu_pod``
    — never a bare ``Mock()``, per .claude/rules/code-style.md § one
    production-body test per seam-stubbed function). Each outcome is raised
    (exception) or returned (PodInfo), one per call."""
    stub = create_autospec(runpod_api.create_cpu_pod, side_effect=list(outcomes))
    monkeypatch.setattr(pod_lifecycle, "create_cpu_pod", stub)
    return stub


def _cpu_provision_ns(**overrides) -> argparse.Namespace:
    """Provision-subparser-shaped Namespace for a CPU intent (mirrors
    tests/test_pod_lifecycle.py::_cpu_provision_ns). ``wait_for_capacity`` is
    deliberately NOT a base key: cmd_provision's hoisted resolution must
    getattr-default it, because hand-built Namespaces predate the flag."""
    base = {
        "issue": 2238,
        "list_intents": False,
        "intent": "cpu-small",
        "gpu_type": None,
        "gpu_count": None,
        "dry_run": False,  # exercise the real create call, not the dry-run early-return
        "volume_gb": 200,  # argparse default (the GPU default)
        "container_disk_gb": 50,
        "ttl_days": 7,
        "no_bootstrap": True,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def cpu_cmd_provision_stubs(monkeypatch, tmp_path):
    """Neuter cmd_provision's network/state preflights so the CPU-branch
    routing runs hermetically; record the PodInfo reaching the bootstrap tail.
    The GPU resolver is trapped so a routing regression fails loudly (the
    routing itself is owned by test_pod_lifecycle.py's CPU tests)."""
    captured: dict = {"boot_infos": []}
    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", tmp_path / "pods_ephemeral.json")
    monkeypatch.setattr(pod_lifecycle, "list_team_pods", lambda: [])
    monkeypatch.setattr(
        pod_lifecycle, "_warn_on_terminal_parent_provision", lambda *_a, **_k: False
    )
    monkeypatch.setattr(pod_lifecycle, "_account_key_preflight", lambda *_a, **_k: None)

    def _record_bootstrap(args, name, info, intent_label):
        captured["boot_infos"].append(info)

    monkeypatch.setattr(pod_lifecycle, "_provision_wait_register_bootstrap", _record_bootstrap)

    def _fail_resolve_spec(*_a, **_k):
        raise AssertionError("GPU _resolve_spec must NOT be reached on a CPU intent")

    monkeypatch.setattr(pod_lifecycle, "_resolve_spec", _fail_resolve_spec)
    return captured


def test_cpu_provision_wait_flag_retries_then_succeeds(
    monkeypatch, cpu_cmd_provision_stubs, capsys
):
    """#2238 test 1 (the task's required shape): CPU intent with
    --wait-for-capacity ON retries create_cpu_pod on RunPodNoCapacityError —
    two failures then success ⇒ 3 calls and the pod reaches the bootstrap
    tail. Pins that the retry HAPPENS, not merely that the flag parses."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    info = _make_cpu_pod_info()
    stub = _make_create_cpu_pod_stub(
        monkeypatch,
        [
            RunPodNoCapacityError("cpu no capacity attempt 1"),
            RunPodNoCapacityError("cpu no capacity attempt 2"),
            info,
        ],
    )

    pod_lifecycle.cmd_provision(_cpu_provision_ns(wait_for_capacity=True))

    assert stub.call_count == 3
    assert cpu_cmd_provision_stubs["boot_infos"] == [info]
    # Every attempt carried the canonical instance id + pod name (no mutation
    # of the request between retries).
    for call in stub.call_args_list:
        assert call.kwargs["instance_id"] == "cpu3g-2-8"
        assert call.kwargs["name"] == "pod-2238"
    # §3c: the loop-start heartbeat renders the CPU-legible spec label.
    assert "(CPU cpu3g-2-8)" in capsys.readouterr().err


def test_cpu_provision_autonomous_env_retries_and_prints_note(
    monkeypatch, cpu_cmd_provision_stubs, capsys
):
    """#2238 test 2 (the false-comfort assertion): EPM_AUTONOMOUS_SESSION=1
    with NO wait_for_capacity key on the Namespace (the getattr-hoist
    contract) retries anyway AND prints the CPU auto-enable note — promise
    printed ⟺ promise kept. Pre-fix the CPU branch neither printed nor
    retried."""
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    info = _make_cpu_pod_info()
    stub = _make_create_cpu_pod_stub(
        monkeypatch,
        [
            RunPodNoCapacityError("cpu no capacity attempt 1"),
            RunPodNoCapacityError("cpu no capacity attempt 2"),
            info,
        ],
    )

    ns = _cpu_provision_ns()
    assert not hasattr(ns, "wait_for_capacity")  # exercises the getattr default
    pod_lifecycle.cmd_provision(ns)

    assert stub.call_count == 3
    assert cpu_cmd_provision_stubs["boot_infos"] == [info]
    out = capsys.readouterr().out
    assert "auto-enabling --wait-for-capacity" in out
    assert "CPU create" in out  # the CPU-legible note, printed AT the branch that retries


def test_cpu_provision_default_off_fails_fast_first_call(monkeypatch, cpu_cmd_provision_stubs):
    """#2238 test 3 (default-OFF preserved): flag OFF + non-autonomous ⇒ the
    no-capacity error propagates on the FIRST call (exactly one call) —
    interactive CPU provisions must not start hanging."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    stub = _make_create_cpu_pod_stub(
        monkeypatch,
        [
            RunPodNoCapacityError("cpu no capacity"),
            # Sentinel: if the branch wrongly retries, the next call would
            # succeed and mask the regression; calls == 1 below catches it.
            _make_cpu_pod_info(),
        ],
    )

    with pytest.raises(RunPodNoCapacityError):
        pod_lifecycle.cmd_provision(_cpu_provision_ns(wait_for_capacity=False))

    assert stub.call_count == 1
    assert cpu_cmd_provision_stubs["boot_infos"] == []


def test_cpu_provision_budget_trip_exits_75(monkeypatch, cpu_cmd_provision_stubs, capsys):
    """#2238 test 4 (still-waiting contract on CPU): a budget trip on the CPU
    leg converts WaitForCapacityStillWaiting into
    SystemExit(EXIT_STILL_WAITING) (75) via _emit_still_waiting_and_exit; no
    pod is created/registered. Uses the _TickingMonotonic pattern — a literal
    0 budget DISABLES the check, and an un-ticked clock never trips it."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    monkeypatch.setenv("EPM_WAIT_FOR_CAPACITY_BUDGET_SECS", "60")
    monkeypatch.setattr(pod_lifecycle.time, "monotonic", _TickingMonotonic(step=50.0))
    stub = _make_create_cpu_pod_stub(
        monkeypatch,
        [
            RunPodNoCapacityError("cpu no capacity attempt 1"),
            RunPodNoCapacityError("cpu no capacity attempt 2"),
            _make_cpu_pod_info(),  # sentinel — must never be reached
        ],
    )

    with pytest.raises(SystemExit) as se:
        pod_lifecycle.cmd_provision(_cpu_provision_ns(wait_for_capacity=True))

    assert se.value.code == pod_lifecycle.EXIT_STILL_WAITING == 75
    assert stub.call_count >= 1
    assert stub.call_count < 3  # never reached the success sentinel
    assert cpu_cmd_provision_stubs["boot_infos"] == []
    assert "[wait-for-capacity] STILL-WAITING" in capsys.readouterr().err
