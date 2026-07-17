"""#1465 — pod_lifecycle stderr relay + tail-bearing provision exceptions.

``RunPodBackend``'s two ``pod_lifecycle.py`` shell-outs (provision at
``launch``, terminate at ``teardown``) route through
``_run_pod_lifecycle_relay``, which TEES the child's stderr live (the
``[wait-for-capacity]`` heartbeats keep streaming during multi-hour waits)
while keeping a bounded tail, and on non-zero exit raises
:class:`PodLifecycleProcessError` — a ``subprocess.CalledProcessError``
SUBCLASS whose ``str()`` carries the stderr tail (the incident-#1336 fix:
an opaque ``exit status 1`` with zero diagnostics). These tests pin:

* the exception is a ``CalledProcessError`` with ``returncode`` + ``cmd``
  verbatim and the stderr tail in ``str(exc)``;
* the exit-75 still-waiting contract is byte-compatible — the REAL
  ``dispatch_issue._provision_still_waiting`` fires on rc=75 with the
  provision cmd shape and does NOT fire on rc=1 (both arms exercised);
* stderr lines are relayed LIVE (streamed as produced, not replayed at
  child exit — the timed liveness proof);
* the tail is bounded (last N lines; overlong lines truncated);
* stdout stays INHERITED (never piped) and the relay fires on success too;
* both ``launch`` and ``teardown`` route through the helper, and a
  tail-bearing raise propagates out of ``launch`` unswallowed.

Tests 1-6 execute the helper's REAL body end-to-end (real ``Popen``, real
pipes, real ``sys.executable -c`` children — the one-production-body-test
rule, #906); only test 7 stubs it as a seam. All CPU, no pod, no network.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from explore_persona_space.backends import runpod as RP
from explore_persona_space.backends.base import RunHandle, RunSpec


class _RecordingRelayOut:
    """Injectable ``relay`` sink recording ``(line, monotonic_ts)`` pairs."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, float]] = []
        self.flushes = 0

    def write(self, line: str) -> None:
        self.lines.append((line, time.monotonic()))

    def flush(self) -> None:
        self.flushes += 1


def _child_cmd(py_body: str, *trailing_argv: str) -> list[str]:
    """A real subprocess command running ``py_body`` under ``sys.executable -c``.

    ``trailing_argv`` tokens land in the child's ``sys.argv`` (ignored by the
    body) — used to give the cmd LIST the ``pod_lifecycle.py`` + ``provision``
    shape ``_provision_still_waiting`` matches on.
    """
    return [sys.executable, "-c", py_body, *trailing_argv]


# ---------------------------------------------------------------------------
# 1. Failure raises a tail-bearing CalledProcessError (fields verbatim)
# ---------------------------------------------------------------------------


def test_relay_raises_tail_bearing_calledprocesserror_on_failure():
    cmd = _child_cmd(
        "import sys; print('DIAG-LINE-1', file=sys.stderr); "
        "print('DIAG-LINE-2', file=sys.stderr); sys.exit(3)"
    )
    with pytest.raises(subprocess.CalledProcessError) as ei:
        RP._run_pod_lifecycle_relay(cmd, relay=_RecordingRelayOut())
    exc = ei.value
    assert isinstance(exc, RP.PodLifecycleProcessError)
    assert exc.returncode == 3
    assert exc.cmd is cmd  # verbatim — the still-waiting cmd-shape contract
    assert "DIAG-LINE-1" in str(exc)
    assert "DIAG-LINE-2" in str(exc)


# ---------------------------------------------------------------------------
# 2. Exit-75 still-waiting contract — the REAL predicate, both arms
# ---------------------------------------------------------------------------


def test_exit75_still_waiting_contract_byte_compatible():
    from scripts.dispatch_issue import _provision_still_waiting

    cmd75 = _child_cmd("import sys; sys.exit(75)", "pod_lifecycle.py", "provision")
    with pytest.raises(subprocess.CalledProcessError) as ei:  # the literal legacy catch
        RP._run_pod_lifecycle_relay(cmd75, relay=_RecordingRelayOut())
    assert ei.value.returncode == 75
    assert _provision_still_waiting(ei.value) is True

    # rc != 75 with the same cmd shape → NOT still-waiting (arm 1 False).
    cmd1 = _child_cmd("import sys; sys.exit(1)", "pod_lifecycle.py", "provision")
    with pytest.raises(subprocess.CalledProcessError) as ei1:
        RP._run_pod_lifecycle_relay(cmd1, relay=_RecordingRelayOut())
    assert _provision_still_waiting(ei1.value) is False


# ---------------------------------------------------------------------------
# 3. Liveness — lines stream as produced, not capture-then-replay
# ---------------------------------------------------------------------------


def test_relay_streams_lines_live_not_capture_then_replay():
    recorder = _RecordingRelayOut()
    cmd = _child_cmd(
        "import sys, time; "
        "print('LINE-A', file=sys.stderr, flush=True); "
        "time.sleep(2.0); "
        "print('LINE-B', file=sys.stderr, flush=True)"
    )
    RP._run_pod_lifecycle_relay(cmd, relay=recorder)
    by_text = {line.strip(): ts for line, ts in recorder.lines}
    assert "LINE-A" in by_text and "LINE-B" in by_text
    # Capture-then-replay would emit both lines ~simultaneously at exit;
    # live streaming preserves the child's 2.0 s gap (1.0 s = 2x margin
    # for a loaded shared VM).
    assert by_text["LINE-B"] - by_text["LINE-A"] >= 1.0


# ---------------------------------------------------------------------------
# 4. Tail bounded to the LAST N lines
# ---------------------------------------------------------------------------


def test_relay_tail_bounded_to_last_lines():
    n = RP._POD_LIFECYCLE_TAIL_MAX_LINES + 25
    cmd = _child_cmd(
        "import sys\n"
        f"for i in range({n}):\n"
        "    print(f'TAIL-LINE-{i:04d}', file=sys.stderr)\n"
        "sys.exit(1)"
    )
    with pytest.raises(RP.PodLifecycleProcessError) as ei:
        RP._run_pod_lifecycle_relay(cmd, relay=_RecordingRelayOut())
    msg = str(ei.value)
    assert f"TAIL-LINE-{n - 1:04d}" in msg  # the last line survives
    assert "TAIL-LINE-0000" not in msg  # the first fell out of the deque
    # Exactly the bounded window remains on the exception's stderr field.
    assert len(ei.value.stderr.splitlines()) == RP._POD_LIFECYCLE_TAIL_MAX_LINES


# ---------------------------------------------------------------------------
# 5. Overlong single line truncated in the tail
# ---------------------------------------------------------------------------


def test_relay_truncates_overlong_line():
    cmd = _child_cmd("import sys; print('X' * 5000, file=sys.stderr); sys.exit(1)")
    with pytest.raises(RP.PodLifecycleProcessError) as ei:
        RP._run_pod_lifecycle_relay(cmd, relay=_RecordingRelayOut())
    msg = str(ei.value)
    assert "[line truncated]" in msg
    # Bounded: cap + truncation marker, never the raw 5000-char line.
    assert len(ei.value.stderr) < RP._POD_LIFECYCLE_TAIL_MAX_LINE_CHARS + 100
    assert "X" * (RP._POD_LIFECYCLE_TAIL_MAX_LINE_CHARS + 1) not in msg


# ---------------------------------------------------------------------------
# 6. Success returns None, still relays stderr, stdout stays inherited
# ---------------------------------------------------------------------------


def test_relay_success_returns_none_and_still_relays(capfd):
    recorder = _RecordingRelayOut()
    cmd = _child_cmd(
        "import sys; print('HEARTBEAT [wait-for-capacity]', file=sys.stderr, flush=True); "
        "print('STDOUT-PASSTHROUGH')"
    )
    assert RP._run_pod_lifecycle_relay(cmd, relay=recorder) is None
    # Relay is unconditional (live heartbeats), not failure-only.
    assert any("HEARTBEAT" in line for line, _ in recorder.lines)
    assert recorder.flushes >= 1
    # stdout is INHERITED (never piped) — the child's stdout reaches the
    # parent's stdout untouched (the plan's hard constraint).
    out, _err = capfd.readouterr()
    assert "STDOUT-PASSTHROUGH" in out


# ---------------------------------------------------------------------------
# 7. Call-site routing — launch + teardown go through the relay (seam test)
# ---------------------------------------------------------------------------


def test_launch_and_teardown_route_through_relay(monkeypatch):
    calls: list[dict] = []

    def _recording_relay(cmd, **kwargs):
        calls.append({"cmd": [str(c) for c in cmd], **kwargs})
        return None

    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _recording_relay)

    # (a) provision leg: the provision argv reaches the helper.
    RP.RunPodBackend().launch(RunSpec(issue=1465, intent="lora-7b", backend="runpod"))
    assert len(calls) == 1
    assert any("pod_lifecycle.py" in c for c in calls[0]["cmd"])
    assert "provision" in calls[0]["cmd"]

    # (b) teardown leg: terminate + --yes reach the helper WITH env threaded.
    handle = RunHandle(
        backend="runpod",
        cluster=None,
        job_id="",
        pod_name="pod-1465",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-1465.log",
        extra={"issue": 1465},
    )
    RP.RunPodBackend().teardown(handle)
    assert len(calls) == 2
    assert "terminate" in calls[1]["cmd"]
    assert "--yes" in calls[1]["cmd"]
    assert calls[1].get("env") is not None  # os.environ.copy() threaded

    # (c) a tail-bearing raise propagates out of launch unswallowed.
    def _exploding_relay(cmd, **kwargs):
        raise RP.PodLifecycleProcessError(3, list(cmd), output=None, stderr="BOOM-TAIL")

    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _exploding_relay)
    with pytest.raises(subprocess.CalledProcessError) as ei:
        RP.RunPodBackend().launch(RunSpec(issue=1465, intent="lora-7b", backend="runpod"))
    assert "BOOM-TAIL" in str(ei.value)
