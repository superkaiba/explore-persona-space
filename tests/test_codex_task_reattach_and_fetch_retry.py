"""Tests for codex_task.py result-fetch retry + --reattach mode (task #1020).

Two recovery affordances under test:

1. Bounded result-fetch retry (``_fetch_result_with_retry``): a transient
   ``No job found`` from ``codex-companion result`` (a torn read of the
   companion's non-atomic state.json jobs index) is re-probed up to
   ``--result-fetch-retry-cap`` times before the job is declared lost
   (exit 7). The retry re-fetches ONLY — exit 7 stays OUTSIDE the #579
   transient whole-job re-dispatch class, so a completed job is never
   re-dispatched. A fetch-subprocess TimeoutExpired/OSError (previously an
   unhandled crash with NO failure marker) is converted to a retryable
   failure. Terminal marker notes carry a countable ``fetch_retries=<k>``
   token.

2. ``--reattach <job_id>``: skip the spawn and drive an EXISTING job through
   the unchanged poll loop / stall detector / result fetch / output write /
   marker posts — the recovery path for a wrapper kill that orphaned a
   running job. Guarded (MF1): with ``--issue N`` the job id must be bound
   to issue N's own ``epm:codex-task-spawned`` history and not already
   terminally paired (fail-closed; ``--reattach-unbound`` overrides, and is
   REQUIRED without ``--issue``). ``_active_job_id`` arms only AFTER the
   guard (MF3); the attach-time confirm probe uses ``_probe_phase_safe``
   (MF2) so a status-CLI raise exits 4 with a failure marker, never a crash.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_codex_task():
    """Load scripts/codex_task.py as an isolated module."""
    spec = importlib.util.spec_from_file_location(
        "codex_task_reattach_under_test", REPO_ROOT / "scripts" / "codex_task.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["codex_task_reattach_under_test"] = module
    spec.loader.exec_module(module)
    return module


codex_task = _load_codex_task()

FINAL_MSG = "FINAL MESSAGE"
NO_JOB_STDERR = 'No job found for "task-x". Run /codex:status to list known jobs.'


def _args(**overrides):
    """Argparse-like namespace with sane defaults (mirrors main()'s parser)."""
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
        transient_retry_cap=1,
        result_fetch_retry_cap=3,
        reattach=None,
        reattach_unbound=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _bound_events(job_id: str = "task-r") -> list[dict]:
    """Fake events.jsonl rows: a codex_task-posted spawned marker binding
    ``job_id`` to the issue, with no terminal marker paired to it."""
    return [
        {
            "kind": "epm:codex-task-spawned",
            "by": "codex_task",
            "note": (
                f"Codex job_id={job_id} effort=high write=True "
                "poll_interval=30s max_wait=21600s probe_error_cap=10 stall_detect=600s"
            ),
        },
    ]


def _boom_spawn(*_a, **_k):
    raise AssertionError("spawn must not be called")


def _wire_common(monkeypatch, markers: list, *, events=None, events_raise=None):
    """Common reattach-test plumbing: capture markers, ban spawns, no sleeps."""
    monkeypatch.setattr(codex_task, "_spawn_codex", _boom_spawn)
    monkeypatch.setattr(
        codex_task,
        "_post_marker",
        lambda issue, kind, note, version=1: markers.append((issue, kind, note)) or True,
    )
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *_a, **_k: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda _s: None)
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)
    if events_raise is not None:

        def _raise(_issue):
            raise events_raise

        monkeypatch.setattr(codex_task, "list_events", _raise)
    else:
        monkeypatch.setattr(codex_task, "list_events", lambda _issue: events or [])


# ──────────────────────────────────────────────────────────────────────
# Result-fetch retry (_fetch_result_with_retry / _finalize_result).
# ──────────────────────────────────────────────────────────────────────


def test_fetch_retry_no_job_found_then_success(monkeypatch, capsys):
    """Two transient 'No job found' failures then success: rc 0, three fetch
    calls, two jittered sleeps in [5, 15]s, retries_used == 2, and the
    status-probe discriminator logged between attempts."""
    calls = {"fetch": 0}

    def fake_fetch(_companion, _job_id):
        calls["fetch"] += 1
        if calls["fetch"] <= 2:
            return 1, "", NO_JOB_STDERR
        return 0, "RESULT", ""

    monkeypatch.setattr(codex_task, "_fetch_result", fake_fetch)
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *_a: ("done", "", None))
    sleeps: list[float] = []
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: sleeps.append(s))

    rc, stdout, _stderr, detail, retries = codex_task._fetch_result_with_retry(
        Path("/fake/c.mjs"), "task-x", 3
    )

    assert rc == 0
    assert stdout == "RESULT"
    assert detail == ""
    assert retries == 2
    assert calls["fetch"] == 3
    assert len(sleeps) == 2
    assert all(5.0 <= s <= 15.0 for s in sleeps), sleeps
    err = capsys.readouterr().err
    assert "status-probe: job known, phase=done" in err
    assert "result-fetch attempt 1/4" in err


def test_fetch_retry_exhaustion_exit7_failure_marker_no_redispatch(monkeypatch):
    """Every fetch fails with the ran-but-unfetchable incident shape (status
    probe says phase=done): exit 7, exactly ONE spawn for the whole helper
    run (exit 7 is excluded from the transient re-dispatch class — the
    load-bearing contract), and ONE failure marker carrying the attempt
    count, the fetch_retries token, and the discriminator."""
    spawns = {"n": 0}

    def fake_spawn(_companion, _prompt, _effort, _write):
        spawns["n"] += 1
        return f"task-x{spawns['n']}"

    monkeypatch.setattr(codex_task, "_spawn_codex", fake_spawn)
    # The exact incident shape: the job is KNOWN and done, but `result`
    # keeps failing. The poll loop consumes the same mock and terminates
    # immediately on phase=done.
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *_a: ("done", "", None))
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *_a: (1, "", NO_JOB_STDERR))
    markers: list = []
    monkeypatch.setattr(
        codex_task,
        "_post_marker",
        lambda issue, kind, note, version=1: markers.append((issue, kind, note)) or True,
    )
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *_a: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda _s: None)
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

    argv = ["codex_task.py", "--prompt", "go", "--issue", "99", "--poll-interval-secs", "0"]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 7
    assert spawns["n"] == 1, spawns  # no whole-job re-dispatch on exit 7
    failed = [m for m in markers if m[1] == "epm:codex-task-failed"]
    assert len(failed) == 1, markers
    note = failed[0][2]
    assert "after 4 attempt(s)" in note
    assert "fetch_retries=3" in note
    assert "job known, phase=done" in note


def test_fetch_retry_cap_zero_single_shot(monkeypatch):
    """result_fetch_retry_cap=0 keeps the old single-shot semantics: exactly
    one fetch call, no sleeps, immediate exit-7 AttemptResult."""
    calls = {"fetch": 0}

    def fake_fetch(_companion, _job_id):
        calls["fetch"] += 1
        return 1, "", NO_JOB_STDERR

    sleeps: list[float] = []
    monkeypatch.setattr(codex_task, "_fetch_result", fake_fetch)
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *_a: ("done", "", None))
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: sleeps.append(s))

    args = _args(result_fetch_retry_cap=0)
    result = codex_task._finalize_result(
        Path("/fake/c.mjs"), "task-z", "done", args, None, time.time()
    )

    assert result.kind == "fail"
    assert result.exit_code == 7
    assert calls["fetch"] == 1
    assert sleeps == []
    assert "after 1 attempt(s)" in result.note
    assert "fetch_retries=0" in result.note


def test_fetch_timeout_expired_retried_not_crash(monkeypatch):
    """A fetch-subprocess TimeoutExpired is converted to a retryable failure
    (previously: unhandled traceback, no failure marker); the next attempt's
    success returns rc 0 with retries_used == 1."""
    calls = {"fetch": 0}

    def fake_fetch(_companion, _job_id):
        calls["fetch"] += 1
        if calls["fetch"] == 1:
            raise subprocess.TimeoutExpired(cmd="x", timeout=120)
        return 0, "OK", ""

    monkeypatch.setattr(codex_task, "_fetch_result", fake_fetch)
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *_a: ("done", "", None))
    monkeypatch.setattr(codex_task.time, "sleep", lambda _s: None)

    rc, stdout, _stderr, detail, retries = codex_task._fetch_result_with_retry(
        Path("/fake/c.mjs"), "task-t", 3
    )

    assert rc == 0
    assert stdout == "OK"
    assert detail == ""
    assert retries == 1
    assert calls["fetch"] == 2


def test_fetch_retries_token_on_success_marker(monkeypatch):
    """The completed-marker note carries fetch_retries=<k>: 2 on a
    retry-then-success run, 0 on a first-try-success run."""

    def _run(fetch_fail_times: int) -> list:
        calls = {"fetch": 0}

        def fake_fetch(_companion, _job_id):
            calls["fetch"] += 1
            if calls["fetch"] <= fetch_fail_times:
                return 1, "", NO_JOB_STDERR
            return 0, "RESULT", ""

        markers: list = []
        monkeypatch.setattr(codex_task, "_spawn_codex", lambda *_a, **_k: "task-tok")
        monkeypatch.setattr(codex_task, "_probe_phase", lambda *_a: ("done", "", None))
        monkeypatch.setattr(codex_task, "_fetch_result", fake_fetch)
        monkeypatch.setattr(
            codex_task,
            "_post_marker",
            lambda issue, kind, note, version=1: markers.append((issue, kind, note)) or True,
        )
        monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *_a: None)
        monkeypatch.setattr(codex_task.time, "sleep", lambda _s: None)
        monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
        monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

        argv = ["codex_task.py", "--prompt", "go", "--issue", "7", "--poll-interval-secs", "0"]
        with patch.object(sys, "argv", argv):
            rc = codex_task.main()
        assert rc == 0
        return [m for m in markers if m[1] == "epm:codex-task-completed"]

    completed = _run(fetch_fail_times=2)
    assert len(completed) == 1
    assert "fetch_retries=2" in completed[0][2]

    completed = _run(fetch_fail_times=0)
    assert len(completed) == 1
    assert "fetch_retries=0" in completed[0][2]


# ──────────────────────────────────────────────────────────────────────
# --reattach mode.
# ──────────────────────────────────────────────────────────────────────


def test_reattach_happy_path(monkeypatch, tmp_path):
    """Bound live job polls to done: output written, spawned marker carries
    reattach=true + the job id, completed marker carries fetch_retries=0,
    exit 0, and _spawn_codex is never called."""
    out = tmp_path / "out.md"
    markers: list = []
    _wire_common(monkeypatch, markers, events=_bound_events("task-r"))

    probe_state = {"n": 0}

    def fake_probe(_companion, _job_id):
        probe_state["n"] += 1
        if probe_state["n"] == 1:
            return "running", "", None
        return "done", "", None

    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *_a: (0, FINAL_MSG, ""))

    argv = [
        "codex_task.py",
        "--reattach",
        "task-r",
        "--output-file",
        str(out),
        "--issue",
        "42",
        "--poll-interval-secs",
        "0",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 0
    assert out.read_text() == FINAL_MSG
    spawned = [m for m in markers if m[1] == "epm:codex-task-spawned"]
    assert len(spawned) == 1
    assert "reattach=true" in spawned[0][2]
    assert "job_id=task-r" in spawned[0][2]
    completed = [m for m in markers if m[1] == "epm:codex-task-completed"]
    assert len(completed) == 1
    assert "fetch_retries=0" in completed[0][2]


def test_reattach_unbound_id_fails_closed(monkeypatch, tmp_path):
    """MF1: a job id with no spawned marker on this issue (a cross-issue /
    unknown id) fails CLOSED before any subprocess work: exit 4, output file
    byte-untouched, no spawned/completed marker, exactly one failure marker
    naming the id + the guard reason, and _probe_phase never called."""
    out = tmp_path / "out.md"
    out.write_text("PRE-EXISTING")
    markers: list = []
    _wire_common(monkeypatch, markers, events=_bound_events("task-OTHER"))

    probe_calls: list = []
    monkeypatch.setattr(
        codex_task, "_probe_phase", lambda *_a: probe_calls.append(1) or ("done", "", None)
    )
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *_a: (0, FINAL_MSG, ""))

    argv = [
        "codex_task.py",
        "--reattach",
        "task-r",
        "--output-file",
        str(out),
        "--issue",
        "42",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 4
    assert out.read_text() == "PRE-EXISTING"  # byte-untouched
    assert probe_calls == []  # guard precedes ALL subprocess work
    assert [m for m in markers if m[1] == "epm:codex-task-spawned"] == []
    assert [m for m in markers if m[1] == "epm:codex-task-completed"] == []
    failed = [m for m in markers if m[1] == "epm:codex-task-failed"]
    assert len(failed) == 1
    assert "job_id=task-r" in failed[0][2]
    assert "binding guard failed" in failed[0][2]
    assert "no epm:codex-task-spawned" in failed[0][2]


def test_reattach_terminal_paired_id_fails_closed(monkeypatch):
    """MF1: a job id already paired with a terminal completed/failed marker
    on this issue fails closed (reattach would duplicate the fetch)."""
    events = [
        *_bound_events("task-r"),
        {
            "kind": "epm:codex-task-completed",
            "by": "codex_task",
            "note": "Codex job_id=task-r phase=done after 100s.",
        },
    ]
    markers: list = []
    _wire_common(monkeypatch, markers, events=events)
    probe_calls: list = []
    monkeypatch.setattr(
        codex_task, "_probe_phase", lambda *_a: probe_calls.append(1) or ("done", "", None)
    )

    argv = ["codex_task.py", "--reattach", "task-r", "--issue", "42"]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 4
    assert probe_calls == []
    failed = [m for m in markers if m[1] == "epm:codex-task-failed"]
    assert len(failed) == 1
    assert "already has a terminal codex-task marker" in failed[0][2]


def test_reattach_token_boundary_no_prefix_match(monkeypatch):
    """MF1: a spawned marker for task-r-longer does NOT bind task-r — the
    token-boundary regex rejects the prefix match."""
    markers: list = []
    _wire_common(monkeypatch, markers, events=_bound_events("task-r-longer"))
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *_a: ("done", "", None))

    argv = ["codex_task.py", "--reattach", "task-r", "--issue", "42"]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 4
    failed = [m for m in markers if m[1] == "epm:codex-task-failed"]
    assert len(failed) == 1
    assert "binding guard failed" in failed[0][2]


def test_reattach_unbound_override_proceeds(monkeypatch, tmp_path):
    """MF1: --reattach-unbound skips the guard entirely (list_events is
    never called) and proceeds to the happy path; the spawned marker note
    carries unbound_override=true."""
    out = tmp_path / "out.md"
    markers: list = []
    _wire_common(
        monkeypatch,
        markers,
        events_raise=AssertionError("list_events must not be called under --reattach-unbound"),
    )

    probe_state = {"n": 0}

    def fake_probe(_companion, _job_id):
        probe_state["n"] += 1
        if probe_state["n"] == 1:
            return "running", "", None
        return "done", "", None

    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *_a: (0, FINAL_MSG, ""))

    argv = [
        "codex_task.py",
        "--reattach",
        "task-r",
        "--reattach-unbound",
        "--output-file",
        str(out),
        "--issue",
        "42",
        "--poll-interval-secs",
        "0",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 0
    assert out.read_text() == FINAL_MSG
    spawned = [m for m in markers if m[1] == "epm:codex-task-spawned"]
    assert len(spawned) == 1
    assert "unbound_override=true" in spawned[0][2]


def test_reattach_without_issue_requires_unbound_flag(monkeypatch, tmp_path):
    """--reattach without --issue and without --reattach-unbound is an
    argparse error (exit 2); adding --reattach-unbound proceeds (and posts
    no markers, since there is no issue)."""
    with (
        patch.object(sys, "argv", ["codex_task.py", "--reattach", "task-r"]),
        pytest.raises(SystemExit) as excinfo,
    ):
        codex_task.main()
    assert excinfo.value.code == 2

    out = tmp_path / "out.md"
    markers: list = []
    _wire_common(monkeypatch, markers, events=[])
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *_a: ("done", "", None))
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *_a: (0, FINAL_MSG, ""))

    argv = [
        "codex_task.py",
        "--reattach",
        "task-r",
        "--reattach-unbound",
        "--output-file",
        str(out),
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 0
    assert out.read_text() == FINAL_MSG
    assert markers == []  # no issue -> no marker posts


def test_reattach_events_read_error_fails_closed(monkeypatch):
    """MF1: an events.jsonl read error is a guard failure (fail-closed),
    never a pass."""
    markers: list = []
    _wire_common(monkeypatch, markers, events_raise=RuntimeError("events unreadable"))
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *_a: ("done", "", None))

    argv = ["codex_task.py", "--reattach", "task-r", "--issue", "42"]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 4
    failed = [m for m in markers if m[1] == "epm:codex-task-failed"]
    assert len(failed) == 1
    assert "fail-closed" in failed[0][2]


def test_reattach_probe_raises_timeout_exit4(monkeypatch, tmp_path):
    """MF2: a status-CLI raise (TimeoutExpired) at attach time converts to
    the probe-error path — bounded retry (1 + cap probes), exit 4, exactly
    one failure marker, no spawn, no output write, no crash."""
    out = tmp_path / "out.md"
    markers: list = []
    _wire_common(monkeypatch, markers, events=_bound_events("task-r"))

    probe_calls = {"n": 0}

    def raising_probe(_companion, _job_id):
        probe_calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd="x", timeout=60)

    monkeypatch.setattr(codex_task, "_probe_phase", raising_probe)

    argv = [
        "codex_task.py",
        "--reattach",
        "task-r",
        "--output-file",
        str(out),
        "--issue",
        "42",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 4
    assert probe_calls["n"] == 4  # 1 + default cap 3
    assert not out.exists()  # no output write
    failed = [m for m in markers if m[1] == "epm:codex-task-failed"]
    assert len(failed) == 1
    assert "probe raised" in failed[0][2]


def test_reattach_unknown_job_exit4_bounded(monkeypatch):
    """An unqueryable job id exits 4 after a bounded confirm-probe retry
    (1 + cap probes), with a failure marker — never the poll loop's
    probe-error crawl, never a spawn."""
    markers: list = []
    _wire_common(monkeypatch, markers, events=_bound_events("task-r"))

    probe_calls = {"n": 0}

    def fake_probe(_companion, _job_id):
        probe_calls["n"] += 1
        return "probe-error", 'No job found for "task-r". Run /codex:status.', None

    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)

    argv = ["codex_task.py", "--reattach", "task-r", "--issue", "42"]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 4
    assert probe_calls["n"] == 4  # 1 + default cap 3, NOT probe_error_cap=10
    failed = [m for m in markers if m[1] == "epm:codex-task-failed"]
    assert len(failed) == 1
    assert "reattach" in failed[0][2]
    assert "not queryable" in failed[0][2]


def test_reattach_already_terminal_skips_poll(monkeypatch, tmp_path):
    """A job already terminal at attach time (finished during the orphan
    gap) skips the poll loop entirely and goes straight to fetch."""
    out = tmp_path / "out.md"
    markers: list = []
    _wire_common(monkeypatch, markers, events=_bound_events("task-r"))
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *_a: ("done", "", None))
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *_a: (0, FINAL_MSG, ""))

    def poll_must_not_run(*_a, **_k):
        raise AssertionError("_poll_until_terminal must not be called for a terminal job")

    monkeypatch.setattr(codex_task, "_poll_until_terminal", poll_must_not_run)

    argv = [
        "codex_task.py",
        "--reattach",
        "task-r",
        "--output-file",
        str(out),
        "--issue",
        "42",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 0
    assert out.read_text() == FINAL_MSG


def test_reattach_prompt_mutually_exclusive():
    """--reattach with --prompt is an argparse error (exit 2)."""
    argv = ["codex_task.py", "--reattach", "task-r", "--prompt", "x", "--issue", "42"]
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as excinfo:
        codex_task.main()
    assert excinfo.value.code == 2


def test_reattach_preserves_preexisting_verdict(monkeypatch, tmp_path):
    """pre_output_key=None semantics: a sentinel-bearing verdict already at
    --output-file (plausibly written by Codex BEFORE the original wrapper
    died) is preserved; the final chat message lands in the .final-msg.md
    sidecar (the #604 incident class)."""
    out = tmp_path / "out.md"
    verdict = "<!-- epm:code-review-codex v3 -->\nVERDICT"
    out.write_text(verdict)
    markers: list = []
    _wire_common(monkeypatch, markers, events=_bound_events("task-r"))
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *_a: ("done", "", None))
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *_a: (0, FINAL_MSG, ""))

    argv = [
        "codex_task.py",
        "--reattach",
        "task-r",
        "--output-file",
        str(out),
        "--issue",
        "42",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 0
    assert out.read_text() == verdict  # NOT clobbered
    sidecar = tmp_path / "out.md.final-msg.md"
    assert sidecar.read_text() == FINAL_MSG


def test_reattach_terminal_failed_exit1_no_redispatch(monkeypatch):
    """A reattached job that polls to terminal phase=failed exits 1 with a
    failure marker and never spawns (no re-dispatch loop in reattach mode)."""
    markers: list = []
    _wire_common(monkeypatch, markers, events=_bound_events("task-r"))

    probe_state = {"n": 0}

    def fake_probe(_companion, _job_id):
        probe_state["n"] += 1
        if probe_state["n"] == 1:
            return "running", "", None
        return "failed", "", None

    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *_a: (0, FINAL_MSG, ""))

    argv = [
        "codex_task.py",
        "--reattach",
        "task-r",
        "--issue",
        "42",
        "--poll-interval-secs",
        "0",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 1
    failed = [m for m in markers if m[1] == "epm:codex-task-failed"]
    assert len(failed) == 1
    assert "phase=failed" in failed[0][2]


def test_reattach_never_reads_stdin(monkeypatch, tmp_path):
    """Reattach mode must never read stdin (bg dispatch has no stdin; a TTY
    read would block forever)."""

    class _NoRead:
        def read(self, *_a, **_k):
            raise AssertionError("stdin must not be read in reattach mode")

    monkeypatch.setattr(codex_task.sys, "stdin", _NoRead())

    out = tmp_path / "out.md"
    markers: list = []
    _wire_common(monkeypatch, markers, events=_bound_events("task-r"))
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *_a: ("done", "", None))
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *_a: (0, FINAL_MSG, ""))

    argv = [
        "codex_task.py",
        "--reattach",
        "task-r",
        "--output-file",
        str(out),
        "--issue",
        "42",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 0
    assert out.read_text() == FINAL_MSG
