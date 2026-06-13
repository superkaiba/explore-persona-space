"""Tests for codex_task.py auto-retry-on-cancelled + progress-aware stall.

Two behaviors under test:

1. Auto-retry on terminal phase=cancelled: a job that ends phase=cancelled
   is re-dispatched up to --cancelled-retry-cap times before the failure
   marker is posted. A first-attempt success is NOT retried.
2. Progress-aware stall detection: the stall timer resets whenever the
   Codex log file GROWS (mtime OR size increases), so a long-but-healthy
   run is not force-cancelled at the fixed stall window. Only a genuinely
   silent (non-growing, non-touched) log trips the detector. The absolute
   --max-wait-secs hard cap still bounds total wall time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_codex_task():
    """Load scripts/codex_task.py as an isolated module."""
    spec = importlib.util.spec_from_file_location(
        "codex_task_under_test", REPO_ROOT / "scripts" / "codex_task.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["codex_task_under_test"] = module
    spec.loader.exec_module(module)
    return module


codex_task = _load_codex_task()


def _args(**overrides):
    """Build an argparse-like namespace with sane defaults for one attempt."""
    base = dict(
        issue=None,
        effort="high",
        write=False,
        output_file=None,
        prompt_file=None,
        prompt="do the thing",
        max_wait_secs=3600,
        poll_interval_secs=0,  # no real sleeping in tests
        probe_error_cap=10,
        stall_detect_secs=600,
        cancelled_retry_cap=2,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ──────────────────────────────────────────────────────────────────────
# Progress-aware stall: _log_progress_key + _key_advanced unit tests.
# ──────────────────────────────────────────────────────────────────────


def test_log_progress_key_reads_mtime_and_size(tmp_path):
    """_log_progress_key returns (mtime, size); None for a missing path."""
    f = tmp_path / "codex.log"
    f.write_text("hello")
    key = codex_task._log_progress_key(str(f))
    assert key is not None
    mtime, size = key
    assert size == 5
    assert isinstance(mtime, float)

    assert codex_task._log_progress_key(None) is None
    assert codex_task._log_progress_key(str(tmp_path / "nope.log")) is None


def test_key_advanced_on_size_growth_even_with_equal_mtime():
    """A file that GROWS in size counts as progress even when mtime is
    unchanged (coarse-mtime filesystems / sub-second appends)."""
    prev = (1000.0, 100)
    grew_size_only = (1000.0, 150)  # same mtime, bigger
    assert codex_task._key_advanced(grew_size_only, prev) is True


def test_key_advanced_on_mtime_growth():
    """A bumped mtime counts as progress even when size is unchanged."""
    prev = (1000.0, 100)
    grew_mtime_only = (1001.0, 100)
    assert codex_task._key_advanced(grew_mtime_only, prev) is True


def test_key_advanced_first_readable():
    """First time the log becomes readable (prev None) counts as progress."""
    assert codex_task._key_advanced((1000.0, 1), None) is True


def test_key_advanced_no_change_is_not_progress():
    """Identical key (no mtime AND no size change) is NOT progress — this
    is the silent-log condition the stall detector must catch."""
    same = (1000.0, 100)
    assert codex_task._key_advanced(same, same) is False
    assert codex_task._key_advanced(None, (1000.0, 100)) is False


def test_growing_log_is_not_force_cancelled(tmp_path, monkeypatch):
    """End-to-end: a run whose log keeps GROWING (size increases each poll)
    is never force-cancelled by the stall detector — even when more polls
    elapse than would fit in a single stall window. Only a stalled log
    trips the detector."""
    log = tmp_path / "codex.log"
    log.write_text("0")

    # Append to the log on each probe so the file grows every poll. Keep
    # mtime artificially frozen so the ONLY progress signal is size growth.
    frozen_mtime = 5000.0
    state = {"polls": 0}

    def fake_probe(companion, job_id):
        state["polls"] += 1
        # grow the log
        with log.open("a") as fh:
            fh.write("x" * 64)
        import os as _os

        _os.utime(log, (frozen_mtime, frozen_mtime))  # freeze mtime
        if state["polls"] >= 6:
            return "done", "", str(log)
        return "running", "", str(log)

    monkeypatch.setattr(codex_task, "_spawn_codex", lambda *a, **k: "task-grow")
    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "RESULT", ""))
    cancel_calls = []
    monkeypatch.setattr(
        codex_task,
        "_best_effort_cancel",
        lambda companion, job_id: cancel_calls.append(job_id),
    )

    # stall window 1s but polls run with no real sleep; each poll grows the
    # log so last_log_change_ts keeps refreshing -> stall_age stays ~0.
    args = _args(stall_detect_secs=1, poll_interval_secs=0)
    result = codex_task._run_one_attempt(Path("/fake/companion.mjs"), "p", args, False)

    assert result.kind == "done", result.note
    assert cancel_calls == []  # never force-cancelled
    assert state["polls"] >= 6


def test_silent_log_is_force_cancelled(tmp_path, monkeypatch):
    """A run whose log NEVER grows (frozen mtime AND size) past the stall
    window IS force-cancelled with exit 8."""
    log = tmp_path / "codex.log"
    log.write_text("frozen")
    import os as _os

    _os.utime(log, (5000.0, 5000.0))

    def fake_probe(companion, job_id):
        # never grows; freeze mtime each time
        _os.utime(log, (5000.0, 5000.0))
        return "running", "", str(log)

    # Make wall-clock jump past the stall window between polls.
    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 100.0
        return clock["t"]

    monkeypatch.setattr(codex_task, "_spawn_codex", lambda *a, **k: "task-silent")
    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task.time, "time", fake_time)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)
    cancel_calls = []
    monkeypatch.setattr(
        codex_task,
        "_best_effort_cancel",
        lambda companion, job_id: cancel_calls.append(job_id),
    )

    args = _args(stall_detect_secs=50, poll_interval_secs=0, max_wait_secs=10_000)
    result = codex_task._run_one_attempt(Path("/fake/companion.mjs"), "p", args, False)

    assert result.kind == "fail"
    assert result.exit_code == 8  # stall exit code
    assert cancel_calls == ["task-silent"]


# ──────────────────────────────────────────────────────────────────────
# Auto-retry on terminal phase=cancelled.
# ──────────────────────────────────────────────────────────────────────


def test_cancelled_triggers_redispatch(monkeypatch):
    """A persistently-cancelled job is re-dispatched cancelled_retry_cap
    times (cap+1 total attempts), then fails ONCE with the failure marker."""
    spawns = {"n": 0}

    def fake_spawn(companion, prompt, effort, write):
        spawns["n"] += 1
        return f"task-{spawns['n']}"

    # Confirm-probe passes, then the poll loop sees a terminal 'cancelled'.
    probe_calls = {"n": 0}

    def fake_probe(companion, job_id):
        probe_calls["n"] += 1
        # first call per attempt is the confirm probe (running), second is
        # the poll that returns cancelled.
        if probe_calls["n"] % 2 == 1:
            return "running", "", None
        return "cancelled", "", None

    monkeypatch.setattr(codex_task, "_spawn_codex", fake_spawn)
    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    # The cancelled phase still attempts a result-fetch (preserved behavior);
    # it must succeed for the flow to reach the retryable cancelled branch.
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "", ""))
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)

    fail_calls = []

    def fake_fail(issue, job_id, note, exit_code):
        fail_calls.append((issue, job_id, note, exit_code))
        return exit_code

    monkeypatch.setattr(codex_task, "_fail", fake_fail)
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

    argv = [
        "codex_task.py",
        "--prompt",
        "go",
        "--cancelled-retry-cap",
        "2",
        "--poll-interval-secs",
        "0",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    # cap=2 -> 3 total spawns (1 initial + 2 re-dispatches)
    assert spawns["n"] == 3, spawns
    # failure marker posted exactly once, after exhausting the cap
    assert len(fail_calls) == 1, fail_calls
    assert rc == 1
    assert "cancelled" in fail_calls[0][2]
    assert "exhausted 2 re-dispatch" in fail_calls[0][2]


def test_cancelled_retry_cap_zero_fails_on_first_cancel(monkeypatch):
    """--cancelled-retry-cap 0 disables retry: a single cancelled attempt
    fails immediately (one spawn, one failure marker)."""
    spawns = {"n": 0}
    monkeypatch.setattr(
        codex_task,
        "_spawn_codex",
        lambda *a, **k: f"task-{spawns.__setitem__('n', spawns['n'] + 1) or spawns['n']}",
    )

    probe_calls = {"n": 0}

    def fake_probe(companion, job_id):
        probe_calls["n"] += 1
        if probe_calls["n"] % 2 == 1:
            return "running", "", None
        return "cancelled", "", None

    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "", ""))
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)

    fail_calls = []
    monkeypatch.setattr(
        codex_task, "_fail", lambda issue, job_id, note, ec: fail_calls.append((note, ec)) or ec
    )
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

    argv = [
        "codex_task.py",
        "--prompt",
        "go",
        "--cancelled-retry-cap",
        "0",
        "--poll-interval-secs",
        "0",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert spawns["n"] == 1, spawns
    assert len(fail_calls) == 1
    assert fail_calls[0][1] == 1  # exit code for terminal cancelled
    assert rc == 1


def test_success_on_first_attempt_no_retry(monkeypatch):
    """A first-attempt success posts the completed marker and does NOT
    re-dispatch (exactly one spawn, no failure marker)."""
    spawns = {"n": 0}
    monkeypatch.setattr(
        codex_task,
        "_spawn_codex",
        lambda *a, **k: f"task-{spawns.__setitem__('n', spawns['n'] + 1) or spawns['n']}",
    )

    probe_calls = {"n": 0}

    def fake_probe(companion, job_id):
        probe_calls["n"] += 1
        if probe_calls["n"] % 2 == 1:
            return "running", "", None
        return "done", "", None

    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "RESULT", ""))
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)

    fail_calls = []
    monkeypatch.setattr(codex_task, "_fail", lambda *a, **k: fail_calls.append(a) or 99)
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

    argv = [
        "codex_task.py",
        "--prompt",
        "go",
        "--cancelled-retry-cap",
        "2",
        "--poll-interval-secs",
        "0",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert spawns["n"] == 1, spawns
    assert fail_calls == []
    assert rc == 0


def test_terminal_failed_is_not_retried(monkeypatch):
    """Terminal phase=failed (NOT cancelled) is a hard failure — it must
    NOT be re-dispatched even when the retry cap is positive."""
    spawns = {"n": 0}
    monkeypatch.setattr(
        codex_task,
        "_spawn_codex",
        lambda *a, **k: f"task-{spawns.__setitem__('n', spawns['n'] + 1) or spawns['n']}",
    )

    probe_calls = {"n": 0}

    def fake_probe(companion, job_id):
        probe_calls["n"] += 1
        if probe_calls["n"] % 2 == 1:
            return "running", "", None
        return "failed", "", None

    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "RESULT", ""))
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)

    fail_calls = []
    monkeypatch.setattr(
        codex_task, "_fail", lambda issue, job_id, note, ec: fail_calls.append((note, ec)) or ec
    )
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

    argv = [
        "codex_task.py",
        "--prompt",
        "go",
        "--cancelled-retry-cap",
        "2",
        "--poll-interval-secs",
        "0",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert spawns["n"] == 1, spawns  # NO retry on phase=failed
    assert len(fail_calls) == 1
    assert "phase=failed" in fail_calls[0][0]
    assert rc == 1


# ─── ONE auto-retry with backoff on transient failures (refs #579) ───────────


def test_transient_spawn_failure_retried_once_then_succeeds(monkeypatch):
    """A spawn failure (exit 3 — the 'app-server exit 1 / instant 0s
    failure' class) is re-dispatched ONCE after a backoff sleep; the second
    attempt's success exits 0 with no failure marker."""
    spawns = {"n": 0}

    def fake_spawn(companion, prompt, effort, write):
        spawns["n"] += 1
        if spawns["n"] == 1:
            raise RuntimeError("app-server exited 1")
        return f"task-{spawns['n']}"

    probe_calls = {"n": 0}

    def fake_probe(companion, job_id):
        probe_calls["n"] += 1
        if probe_calls["n"] % 2 == 1:
            return "running", "", None
        return "done", "", None

    sleeps: list[float] = []
    monkeypatch.setattr(codex_task, "_spawn_codex", fake_spawn)
    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "RESULT", ""))
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: sleeps.append(s))

    fail_calls = []
    monkeypatch.setattr(codex_task, "_fail", lambda *a, **k: fail_calls.append(a) or 99)
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

    argv = ["codex_task.py", "--prompt", "go", "--poll-interval-secs", "0"]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert spawns["n"] == 2, spawns
    assert fail_calls == []
    assert rc == 0
    # The backoff sleep fired (floor + jitter window).
    backoffs = [
        s
        for s in sleeps
        if codex_task.TRANSIENT_RETRY_BACKOFF_FLOOR_SECS
        <= s
        <= codex_task.TRANSIENT_RETRY_BACKOFF_FLOOR_SECS
        + codex_task.TRANSIENT_RETRY_BACKOFF_JITTER_SECS
    ]
    assert backoffs, sleeps


def test_transient_retry_exhausts_after_one_redispatch(monkeypatch):
    """Two consecutive spawn failures: one re-dispatch, then the failure
    marker fires once with the transient-exhausted annotation."""
    spawns = {"n": 0}

    def fake_spawn(companion, prompt, effort, write):
        spawns["n"] += 1
        raise RuntimeError("app-server exited 1")

    monkeypatch.setattr(codex_task, "_spawn_codex", fake_spawn)
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)

    fail_calls = []

    def fake_fail(issue, job_id, note, exit_code):
        fail_calls.append((issue, job_id, note, exit_code))
        return exit_code

    monkeypatch.setattr(codex_task, "_fail", fake_fail)
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

    argv = ["codex_task.py", "--prompt", "go", "--poll-interval-secs", "0"]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert spawns["n"] == 2, spawns  # 1 initial + 1 transient re-dispatch
    assert len(fail_calls) == 1, fail_calls
    assert rc == 3
    assert "exhausted 1 transient re-dispatch" in fail_calls[0][2]


def test_transient_retry_cap_zero_disables(monkeypatch):
    """--transient-retry-cap 0 restores the fail-fast behavior."""
    spawns = {"n": 0}

    def fake_spawn(companion, prompt, effort, write):
        spawns["n"] += 1
        raise RuntimeError("app-server exited 1")

    monkeypatch.setattr(codex_task, "_spawn_codex", fake_spawn)
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)
    fail_calls = []
    monkeypatch.setattr(
        codex_task, "_fail", lambda i, j, n, c: fail_calls.append((i, j, n, c)) or c
    )
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

    argv = [
        "codex_task.py",
        "--prompt",
        "go",
        "--transient-retry-cap",
        "0",
        "--poll-interval-secs",
        "0",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert spawns["n"] == 1, spawns
    assert len(fail_calls) == 1
    assert rc == 3


def test_hard_cap_timeout_is_not_transient_retried(monkeypatch):
    """Exit 6 (hard-cap timeout) must NOT be re-dispatched — the attempt
    already consumed max_wait_secs of wall time."""
    spawns = {"n": 0}

    def fake_spawn(companion, prompt, effort, write):
        spawns["n"] += 1
        return f"task-{spawns['n']}"

    # Confirm probe says running; then the poll loop hits the (0s) hard cap.
    monkeypatch.setattr(codex_task, "_spawn_codex", fake_spawn)
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *a, **k: ("running", "", None))
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)
    fail_calls = []
    monkeypatch.setattr(
        codex_task, "_fail", lambda i, j, n, c: fail_calls.append((i, j, n, c)) or c
    )
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

    argv = [
        "codex_task.py",
        "--prompt",
        "go",
        "--max-wait-secs",
        "0",
        "--poll-interval-secs",
        "0",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert spawns["n"] == 1, spawns  # no re-dispatch
    assert len(fail_calls) == 1
    assert rc == 6
