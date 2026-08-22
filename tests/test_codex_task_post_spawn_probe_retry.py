"""Tests for codex_task.py post-spawn probe retry + dispatch lock (#2323).

Behaviors under test (plan #2323 §4.3):

1. A TRANSIENT post-spawn confirm probe failure (the shared jobs index
   lost-update erasure — the plugin's saveState prune-deletes a
   just-spawned entry, which the worker's next upsert re-registers) is
   RE-PROBED with a bounded jittered backoff and recovers WITHOUT any
   re-dispatch and WITHOUT cancelling the job.
2. The torn-read reproduction executes the REAL ``_probe_phase`` body
   against a fake companion subprocess serving the verbatim
   ``No job found`` CLI error, then a healthy status — survival, zero
   ``cancel`` verbs.
3. Probe EXHAUSTION exits ``EXIT_POST_SPAWN_PROBE_EXHAUSTED`` (10) via
   main(): NOT in ``TRANSIENT_FAIL_EXIT_CODES`` (no blind re-dispatch —
   the #2321 orphan generator), the job is NOT cancelled, and the failure
   note names the shared index path + the exact ``--reattach`` recovery.
4. A raising confirm probe (TimeoutExpired) is converted, not swallowed:
   the bounded retry still runs and the exit-10 note carries the raise.
5. The repo-keyed dispatch lock serializes spawn+confirm windows across
   concurrent dispatches; every fail-open mode (timeout / kill switch /
   --no-dispatch-lock) proceeds UNLOCKED with a loud WARN + a
   ``dispatch_lock=<mode>`` marker token; reattach never takes the lock;
   the lock is released on the spawn-exception path.
6. #2324 Leg B: the site-4 lock open is symlink/FIFO-safe and bounded
   (child-process matrix; the fail-OPEN ``unavailable`` posture is
   preserved by construction — ``lock_utils.LockPathError`` subclasses
   ``OSError``), and every post-lock failure path (spawn-exception,
   exit-10 confirm exhaustion, the signal handler) carries
   ``dispatch_lock=<mode>`` for non-held modes, with the
   (job_id, lock_mode) attempt state held in ONE atomically-stored tuple
   (real-timing retry pairing test + dis-based single-store /
   single-snapshot invariants).

The autouse fixture roots ``DISPATCH_ROOT`` at tmp_path so no test can
ever touch the REAL repo-root lock file (a live fleet dispatch can hold
it for minutes) or the real codex-companion state.
"""

from __future__ import annotations

import contextlib
import dis
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import CodeType, SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_codex_task():
    """Load scripts/codex_task.py as an isolated module."""
    spec = importlib.util.spec_from_file_location(
        "codex_task_post_spawn_probe_under_test",
        REPO_ROOT / "scripts" / "codex_task.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["codex_task_post_spawn_probe_under_test"] = module
    spec.loader.exec_module(module)
    return module


codex_task = _load_codex_task()


@pytest.fixture(autouse=True)
def _isolated_dispatch_root(monkeypatch, tmp_path):
    """Root every lock file / index hint at tmp_path (never the REAL
    repo-root .claude/cache/codex-dispatch.lock, which live fleet
    dispatches hold for minutes), make the lock's acquire poll fast, and
    clear any ambient kill switch so the lock genuinely engages."""
    monkeypatch.setattr(codex_task, "DISPATCH_ROOT", tmp_path)
    monkeypatch.setattr(codex_task, "DISPATCH_LOCK_POLL_INTERVAL_SECS", 0.01)
    monkeypatch.delenv("EPM_CODEX_DISPATCH_LOCK", raising=False)


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
        result_fetch_retry_cap=0,
        post_spawn_probe_retry_cap=4,
        dispatch_lock_timeout_secs=5.0,
        no_dispatch_lock=False,
        reattach=None,
        reattach_unbound=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _lock_path() -> Path:
    return Path(codex_task.DISPATCH_ROOT) / codex_task.DISPATCH_LOCK_RELPATH


def _hold_lock_externally() -> int:
    """Acquire the dispatch lock on a separate fd (flock conflicts between
    separate opens even within one process). Caller must os.close(fd)."""
    p = _lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


# ──────────────────────────────────────────────────────────────────────
# 1. Bounded re-probe: transient index miss recovers, no re-dispatch.
# ──────────────────────────────────────────────────────────────────────


def test_transient_probe_error_recovers_without_redispatch(monkeypatch):
    """One transient confirm probe-error (lost-update erasure) then success:
    the attempt proceeds to done with ONE spawn, ZERO cancels, a backoff
    sleep inside the post-spawn window, and probe_retries=1 on the spawned
    marker (no dispatch_lock token — the lock was held normally)."""
    spawns = {"n": 0}

    def fake_spawn(companion, prompt, effort, write):
        spawns["n"] += 1
        return "task-recover"

    probe_calls = {"n": 0}

    def fake_probe(companion, job_id):
        probe_calls["n"] += 1
        if probe_calls["n"] == 1:  # confirm probe: transient index miss
            return "probe-error", "No job found (lost update)", None
        if probe_calls["n"] == 2:  # confirm retry: worker re-upserted
            return "running", "", None
        return "done", "", None  # poll leg

    sleeps: list[float] = []
    cancels: list[str] = []
    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(codex_task, "_spawn_codex", fake_spawn)
    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "RESULT", ""))
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda c, j: cancels.append(j))
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        codex_task, "_post_marker", lambda issue, kind, note: posted.append((issue, kind, note))
    )

    result = codex_task._run_one_attempt(Path("/fake/c.mjs"), "p", _args(issue=55), False)

    assert result.kind == "done", result.note
    assert spawns["n"] == 1, spawns  # NEVER re-dispatched
    assert cancels == []  # NEVER cancelled
    floor = codex_task.POST_SPAWN_PROBE_BACKOFF_FLOOR_SECS
    ceil = floor + codex_task.POST_SPAWN_PROBE_BACKOFF_JITTER_SECS
    backoffs = [s for s in sleeps if floor <= s <= ceil]
    assert backoffs, sleeps  # the jittered re-probe backoff fired
    spawned_notes = [n for _, k, n in posted if k == "epm:codex-task-spawned"]
    assert len(spawned_notes) == 1, posted
    assert "probe_retries=1" in spawned_notes[0]
    assert "dispatch_lock=" not in spawned_notes[0]  # held = no token


def test_torn_read_real_probe_body_survives_and_never_cancels(monkeypatch, tmp_path):
    """Torn-read reproduction through the REAL _probe_phase body: a fake
    companion subprocess serves the verbatim 'No job found' CLI error for
    two status calls (the index transiently lost the entry), then a healthy
    done status. The attempt survives with zero 'cancel' verbs. Executes
    the real _spawn_codex, _probe_phase, _confirm_job_queryable and
    _fetch_result bodies (fake only at the subprocess boundary)."""
    verbs: list[str] = []
    status_calls = {"n": 0}

    def fake_run(cmd, cwd=None, capture_output=True, text=True, timeout=None):
        verb = cmd[2]  # ["node", companion, verb, ...]
        verbs.append(verb)
        if verb == "task":
            return SimpleNamespace(
                returncode=0, stdout="Started task-torn in background", stderr=""
            )
        if verb == "status":
            status_calls["n"] += 1
            if status_calls["n"] <= 2:
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr='No job found for "task-torn". Run /codex:status to list known jobs.',
                )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"workspaceRoot": str(tmp_path), "job": {"phase": "done"}}),
                stderr="",
            )
        if verb == "result":
            return SimpleNamespace(returncode=0, stdout="FINAL", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")  # cancel etc.

    monkeypatch.setattr(codex_task.subprocess, "run", fake_run)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)

    result = codex_task._run_one_attempt(Path("/fake/c.mjs"), "p", _args(), False)

    assert result.kind == "done", result.note
    assert "cancel" not in verbs, verbs  # torn read never cancels the job
    assert status_calls["n"] >= 3, status_calls  # 2 misses + 1 healthy


# ──────────────────────────────────────────────────────────────────────
# 2. Exhaustion: exit 10, no blind re-dispatch, no cancel, reattach recipe.
# ──────────────────────────────────────────────────────────────────────


def test_probe_exhaustion_exits_10_no_redispatch_no_cancel(monkeypatch):
    """A persistent confirm probe failure exhausts the bounded re-probe and
    exits 10 via main(): ONE spawn (exit 10 is NOT transient, so the #579
    loop never blind re-dispatches — the #2321 orphan generator), ZERO
    cancels (the job may be live), and the failure note carries the job id,
    the shared index path, and the exact --reattach recovery command."""
    spawns = {"n": 0}

    def fake_spawn(companion, prompt, effort, write):
        spawns["n"] += 1
        return "task-x"

    cancels: list[str] = []
    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(codex_task, "_spawn_codex", fake_spawn)
    monkeypatch.setattr(
        codex_task, "_probe_phase", lambda *a, **k: ("probe-error", "no job found", None)
    )
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda c, j: cancels.append(j))
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        codex_task, "_post_marker", lambda issue, kind, note: posted.append((issue, kind, note))
    )
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

    argv = [
        "codex_task.py",
        "--prompt",
        "go",
        "--issue",
        "55",
        "--poll-interval-secs",
        "0",
        "--post-spawn-probe-retry-cap",
        "2",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == codex_task.EXIT_POST_SPAWN_PROBE_EXHAUSTED == 10
    assert spawns["n"] == 1, spawns  # no blind re-dispatch
    assert cancels == []  # the job may be live — never cancelled
    failed_notes = [n for _, k, n in posted if k == "epm:codex-task-failed"]
    assert len(failed_notes) == 1, posted
    note = failed_notes[0]
    assert "task-x" in note
    assert codex_task._shared_state_index_hint() in note
    assert "--reattach task-x" in note
    assert "--issue 55" in note
    assert "exhausted after 3 probe(s)" in note  # 1 + cap(2)
    # And no spawned marker: the confirm never succeeded.
    assert not any(k == "epm:codex-task-spawned" for _, k, _n in posted)


def test_exit_code_taxonomy_pins():
    """Exit 10 is the post-spawn exhaustion code and is deliberately NOT
    transient; the transient set is unchanged (exit 4 stays for the
    reattach binding guard); result-fetch (7) stays non-transient."""
    assert codex_task.EXIT_POST_SPAWN_PROBE_EXHAUSTED == 10
    assert 10 not in codex_task.TRANSIENT_FAIL_EXIT_CODES
    assert 7 not in codex_task.TRANSIENT_FAIL_EXIT_CODES
    assert frozenset({3, 4, 5, 8}) == codex_task.TRANSIENT_FAIL_EXIT_CODES


def test_post_spawn_probe_timeout_is_not_swallowed(monkeypatch):
    """A RAISING confirm probe (status CLI TimeoutExpired) is converted by
    _probe_phase_safe — never swallowed and never a crash-without-marker
    (#1020 MF2 on the confirm leg): the bounded retry still runs (1 + cap
    probes), the run exits 10, and the failure note carries the raise."""
    probe_calls = {"n": 0}

    def raising_probe(companion, job_id):
        probe_calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd="node companion status", timeout=30)

    cancels: list[str] = []
    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(codex_task, "_spawn_codex", lambda *a, **k: "task-t")
    monkeypatch.setattr(codex_task, "_probe_phase", raising_probe)
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda c, j: cancels.append(j))
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        codex_task, "_post_marker", lambda issue, kind, note: posted.append((issue, kind, note))
    )
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("/fake/c.mjs"))
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)

    argv = [
        "codex_task.py",
        "--prompt",
        "go",
        "--issue",
        "55",
        "--poll-interval-secs",
        "0",
        "--post-spawn-probe-retry-cap",
        "2",
    ]
    with patch.object(sys, "argv", argv):
        rc = codex_task.main()

    assert rc == 10
    assert probe_calls["n"] == 3  # 1 + cap(2): the retry ran despite raises
    assert cancels == []
    failed_notes = [n for _, k, n in posted if k == "epm:codex-task-failed"]
    assert len(failed_notes) == 1, posted
    assert "probe raised" in failed_notes[0]


def test_poll_leg_probe_raise_is_probe_error_not_crash(monkeypatch):
    """The POLL leg now uses _probe_phase_safe too: a raising status CLI
    mid-poll counts toward --probe-error-cap and ends in the ordinary
    exit-5 cancel path — never an unhandled raise (crash-with-no-marker)."""
    probe_calls = {"n": 0}

    def probe(companion, job_id):
        probe_calls["n"] += 1
        if probe_calls["n"] == 1:  # confirm probe: healthy
            return "running", "", None
        raise subprocess.TimeoutExpired(cmd="node companion status", timeout=30)

    cancels: list[str] = []
    monkeypatch.setattr(codex_task, "_spawn_codex", lambda *a, **k: "task-p")
    monkeypatch.setattr(codex_task, "_probe_phase", probe)
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda c, j: cancels.append(j))
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)

    result = codex_task._run_one_attempt(Path("/fake/c.mjs"), "p", _args(probe_error_cap=3), False)

    assert result.kind == "fail"
    assert result.exit_code == 5
    assert "probe raised" in result.note
    assert cancels == ["task-p"]  # poll-leg cap DOES cancel (unchanged)
    assert probe_calls["n"] == 4  # 1 confirm + 3 poll probes


# ──────────────────────────────────────────────────────────────────────
# 3. Dispatch lock: serialization + fail-open modes + release.
# ──────────────────────────────────────────────────────────────────────


def test_dispatch_lock_serializes_spawn_confirm_windows(monkeypatch):
    """Two concurrent _run_one_attempt calls: the repo-keyed flock makes
    the spawn->confirm windows DISJOINT (thread B's spawn cannot start
    until thread A's confirm finished). Executes the real _dispatch_lock
    body on the acquired path."""
    spans: dict[str, list[float]] = {}
    real_sleep = time.sleep

    def fake_spawn(companion, prompt, effort, write):
        spans[prompt] = [time.monotonic(), -1.0]
        real_sleep(0.15)  # widen the in-lock window so overlap would show
        return f"task-{prompt}"

    def fake_probe(companion, job_id):
        prompt = job_id.removeprefix("task-")
        if spans[prompt][1] < 0:  # first probe = the confirm probe
            spans[prompt][1] = time.monotonic()
        return "done", "", None

    monkeypatch.setattr(codex_task, "_spawn_codex", fake_spawn)
    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "R", ""))
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)

    results: dict[str, object] = {}

    def run(name: str) -> None:
        results[name] = codex_task._run_one_attempt(Path("/fake/c.mjs"), name, _args(), False)

    t1 = threading.Thread(target=run, args=("a",))
    t2 = threading.Thread(target=run, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert results["a"].kind == "done" and results["b"].kind == "done"
    (a0, a1), (b0, b1) = spans["a"], spans["b"]
    assert a1 > 0 and b1 > 0, spans
    assert a1 <= b0 or b1 <= a0, spans  # windows disjoint = serialized


def test_dispatch_lock_timeout_fails_open_with_loud_warn(monkeypatch, capsys):
    """A held lock + a tiny timeout: the dispatch proceeds UNLOCKED (never
    a refusal) with a loud WARN naming the shared index, and the spawned
    marker carries dispatch_lock=timeout-failopen."""
    fd = _hold_lock_externally()
    posted: list[tuple[int, str, str]] = []
    try:
        monkeypatch.setattr(codex_task, "_spawn_codex", lambda *a, **k: "task-t")
        monkeypatch.setattr(codex_task, "_probe_phase", lambda *a, **k: ("done", "", None))
        monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "R", ""))
        monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
        monkeypatch.setattr(
            codex_task,
            "_post_marker",
            lambda issue, kind, note: posted.append((issue, kind, note)),
        )

        result = codex_task._run_one_attempt(
            Path("/fake/c.mjs"), "p", _args(issue=77, dispatch_lock_timeout_secs=0.05), False
        )
    finally:
        os.close(fd)

    assert result.kind == "done", result.note
    err = capsys.readouterr().err
    assert "timed out" in err
    assert codex_task._shared_state_index_hint() in err
    spawned_notes = [n for _, k, n in posted if k == "epm:codex-task-spawned"]
    assert len(spawned_notes) == 1, posted
    assert "dispatch_lock=timeout-failopen" in spawned_notes[0]


def test_kill_switch_env_disables_lock(monkeypatch, capsys):
    """EPM_CODEX_DISPATCH_LOCK=0 skips the lock entirely: an externally
    held lock does not delay the dispatch, the WARN is loud, and the
    spawned marker carries dispatch_lock=disabled."""
    monkeypatch.setenv("EPM_CODEX_DISPATCH_LOCK", "0")
    fd = _hold_lock_externally()
    posted: list[tuple[int, str, str]] = []
    try:
        monkeypatch.setattr(codex_task, "_spawn_codex", lambda *a, **k: "task-k")
        monkeypatch.setattr(codex_task, "_probe_phase", lambda *a, **k: ("done", "", None))
        monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "R", ""))
        monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
        monkeypatch.setattr(
            codex_task,
            "_post_marker",
            lambda issue, kind, note: posted.append((issue, kind, note)),
        )

        started = time.monotonic()
        result = codex_task._run_one_attempt(Path("/fake/c.mjs"), "p", _args(issue=77), False)
        elapsed = time.monotonic() - started
    finally:
        os.close(fd)

    assert result.kind == "done", result.note
    assert elapsed < 4.0, elapsed  # never waited on the held lock
    assert "DISABLED" in capsys.readouterr().err
    spawned_notes = [n for _, k, n in posted if k == "epm:codex-task-spawned"]
    assert "dispatch_lock=disabled" in spawned_notes[0]


def test_no_dispatch_lock_flag_disables(monkeypatch, capsys):
    """--no-dispatch-lock (args.no_dispatch_lock=True) is the CLI twin of
    the env kill switch: same disabled fail-open, same marker token."""
    fd = _hold_lock_externally()
    posted: list[tuple[int, str, str]] = []
    try:
        monkeypatch.setattr(codex_task, "_spawn_codex", lambda *a, **k: "task-f")
        monkeypatch.setattr(codex_task, "_probe_phase", lambda *a, **k: ("done", "", None))
        monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "R", ""))
        monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
        monkeypatch.setattr(
            codex_task,
            "_post_marker",
            lambda issue, kind, note: posted.append((issue, kind, note)),
        )

        result = codex_task._run_one_attempt(
            Path("/fake/c.mjs"), "p", _args(issue=77, no_dispatch_lock=True), False
        )
    finally:
        os.close(fd)

    assert result.kind == "done", result.note
    assert "DISABLED" in capsys.readouterr().err
    spawned_notes = [n for _, k, n in posted if k == "epm:codex-task-spawned"]
    assert "dispatch_lock=disabled" in spawned_notes[0]


def test_reattach_never_takes_dispatch_lock(monkeypatch):
    """--reattach spawns nothing and mutates no index entry, so it must
    never contend on the dispatch lock (a reattach blocked behind a slow
    dispatch would delay a pure harvest)."""
    lock_calls: list[tuple] = []

    @contextlib.contextmanager
    def recording_lock(timeout_s, enabled=True):
        lock_calls.append((timeout_s, enabled))
        yield "held"

    monkeypatch.setattr(codex_task, "_dispatch_lock", recording_lock)
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *a, **k: ("done", "", None))
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "R", ""))
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)

    rc = codex_task._run_reattach(
        Path("/fake/c.mjs"), _args(reattach="task-r", reattach_unbound=True)
    )

    assert rc == 0
    assert lock_calls == []  # reattach never touched the lock


def test_lock_released_on_spawn_exception(monkeypatch):
    """A raising _spawn_codex returns exit 3 through the with-block, and the
    flock is RELEASED (a leaked hold would starve every later dispatch for
    the full timeout): a fresh non-blocking acquire succeeds immediately."""
    monkeypatch.setattr(
        codex_task,
        "_spawn_codex",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("app-server exited 1")),
    )
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)

    result = codex_task._run_one_attempt(Path("/fake/c.mjs"), "p", _args(), False)

    assert result.kind == "fail"
    assert result.exit_code == 3
    fd = _hold_lock_externally()  # raises BlockingIOError if the hold leaked
    os.close(fd)


# ──────────────────────────────────────────────────────────────────────
# 4. #2324 Leg B: symlink/FIFO-safe bounded lock open (site 4) + the
#    dispatch_lock=<mode> token on every post-lock failure path, with
#    atomically-coherent (job_id, lock_mode) attempt pairing.
# ──────────────────────────────────────────────────────────────────────

_NON_HELD_MODES = ["disabled", "timeout-failopen", "unavailable"]


def _plant_lock_fixture(kind: str, lock_path: Path, tmp_path: Path) -> None:
    """Symlink arms are pinned symlink→FIFO (plan §6): the child BOUND assert
    only discriminates when the pre-fix code path would follow the link into a
    blocking FIFO open — a symlink→regular-file fixture would make it vacuous."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "symlink":
        target = tmp_path / "target.fifo"
        os.mkfifo(target)
        os.symlink(target, lock_path)  # symlink -> FIFO, NOT -> regular file
    else:
        os.mkfifo(lock_path)


def _force_mode(mode: str, monkeypatch, tmp_path: Path) -> tuple[dict, list[int]]:
    """Force a realized dispatch-lock mode via its REAL trigger (plan §6).

    Returns (args overrides, fds the caller must os.close())."""
    if mode == "disabled":
        monkeypatch.setenv("EPM_CODEX_DISPATCH_LOCK", "0")
        return {}, []
    if mode == "timeout-failopen":
        return {"dispatch_lock_timeout_secs": 0.05}, [_hold_lock_externally()]
    assert mode == "unavailable"
    # Real trigger, in-process fast companion to the child-matrix arm: a
    # planted symlink→FIFO rejects immediately (ELOOP under O_NOFOLLOW).
    _plant_lock_fixture("symlink", _lock_path(), tmp_path)
    return {}, []


# Child-process bounded matrix (Acceptance bullet 3), site 4:
# {symlink→FIFO, FIFO} x codex-dispatch. The pre-fix raw os.open blocks
# forever in open(2) on the FIFO and trips the subprocess timeout; post-fix
# the open goes through lock_utils.safe_open_lockfile and the LockPathError
# (an OSError subclass) lands in the UNCHANGED except-OSError arm → mode
# "unavailable" with the loud WARN: fail-OPEN preserved by construction.

_DISPATCH_LOCK_REJECT_DRIVER = """
import importlib.util, sys, time
from pathlib import Path

spec = importlib.util.spec_from_file_location("codex_task_child", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
m.DISPATCH_ROOT = Path(sys.argv[2])
t0 = time.monotonic()
with m._dispatch_lock(5.0) as mode:
    print(f"OUTCOME=mode={mode} elapsed={time.monotonic() - t0:.2f}")
"""


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX mkfifo")
@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_dispatch_lock_path_rejection_bounded_in_child(kind: str, tmp_path: Path):
    lock_path = tmp_path / codex_task.DISPATCH_LOCK_RELPATH
    _plant_lock_fixture(kind, lock_path, tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _DISPATCH_LOCK_REJECT_DRIVER,
            str(REPO_ROOT / "scripts" / "codex_task.py"),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,  # bounded: the pre-fix shape hangs here and fails legibly
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OUTCOME=mode=unavailable" in proc.stdout, proc.stdout
    assert "codex dispatch lock UNAVAILABLE" in proc.stderr  # the loud WARN
    assert "lock path rejected" in proc.stderr  # exc message names the rejection
    elapsed = float(proc.stdout.split("elapsed=")[1].split()[0])
    assert elapsed < 5.0  # rejection is immediate — never the flock-poll bound


@pytest.mark.parametrize("mode", _NON_HELD_MODES)
def test_spawn_exception_note_carries_lock_mode(mode: str, monkeypatch, tmp_path):
    """Path 1/3 (D6 item 2): a raising _spawn_codex returns exit 3 with the
    realized mode token appended to the note on every non-held mode."""
    overrides, fds = _force_mode(mode, monkeypatch, tmp_path)
    monkeypatch.setattr(
        codex_task,
        "_spawn_codex",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("app-server exited 1")),
    )
    try:
        result = codex_task._run_one_attempt(Path("/fake/c.mjs"), "p", _args(**overrides), False)
    finally:
        for fd in fds:
            os.close(fd)
    assert result.kind == "fail" and result.exit_code == 3
    assert result.note.startswith("spawn: ")
    assert result.note.endswith(f" dispatch_lock={mode}")


@pytest.mark.parametrize("mode", _NON_HELD_MODES)
def test_exit10_note_carries_lock_mode_and_keeps_reattach_recipe(mode: str, monkeypatch, tmp_path):
    """Path 2/3 (D6 item 3): the exit-10 confirm-exhaustion note carries the
    token AND keeps the #2323-required content (job id, shared index,
    --reattach recipe) FIRST — the token is appended after the recipe."""
    overrides, fds = _force_mode(mode, monkeypatch, tmp_path)
    monkeypatch.setattr(codex_task, "_spawn_codex", lambda *a, **k: "task-x")
    monkeypatch.setattr(
        codex_task,
        "_confirm_job_queryable",
        lambda *a, **k: ("probe-error", "synthetic index miss", None, 4),
    )
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
    try:
        result = codex_task._run_one_attempt(Path("/fake/c.mjs"), "p", _args(**overrides), False)
    finally:
        for fd in fds:
            os.close(fd)
    assert result.exit_code == codex_task.EXIT_POST_SPAWN_PROBE_EXHAUSTED
    assert "task-x" in result.note
    assert "--reattach task-x" in result.note
    assert codex_task._shared_state_index_hint() in result.note
    assert result.note.endswith(f" dispatch_lock={mode}")
    assert result.note.index("--reattach") < result.note.index("dispatch_lock=")


def _captured_handler(monkeypatch):
    """Install + capture the REAL signal handler without touching process
    signal state (signal.signal is recorded, never invoked for real)."""
    captured: dict = {}
    monkeypatch.setattr(codex_task.signal, "signal", lambda sig, h: captured.setdefault(sig, h))
    codex_task._install_signal_handlers()
    return captured[codex_task.signal.SIGTERM]


@pytest.mark.parametrize("mode", _NON_HELD_MODES)
def test_signal_handler_carries_seeded_mode_token(mode: str, monkeypatch, capsys):
    """Path 3/3 (D6 item 4), token-format pin: an armed (job, mode) tuple
    surfaces the token on BOTH the stderr line and the failure marker.
    (Formatting pin ONLY — the pairing blocker is discharged by the
    real-timing retry test + the single-store invariants below.)"""
    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(codex_task, "_post_marker", lambda i, k, n: posted.append((i, k, n)))
    handler = _captured_handler(monkeypatch)
    monkeypatch.setattr(codex_task, "_active_attempt", ("job-x", mode))
    monkeypatch.setattr(codex_task, "_active_issue", 77)
    monkeypatch.setattr(codex_task, "_active_companion", None)
    with pytest.raises(SystemExit) as ei:
        handler(int(codex_task.signal.SIGTERM), None)
    assert ei.value.code == 128 + int(codex_task.signal.SIGTERM)
    err = capsys.readouterr().err
    assert "job-x" in err and f" dispatch_lock={mode}" in err
    notes = [n for _, k, n in posted if k == "epm:codex-task-failed"]
    assert len(notes) == 1, posted
    assert "job-x" in notes[0]
    assert f" dispatch_lock={mode}" in notes[0]


@pytest.mark.parametrize("lock_mode", ["held", None])
def test_signal_handler_no_token_for_held_and_reattach(lock_mode, monkeypatch, capsys):
    """Negative arms (D6): mode "held" and a reattach-armed attempt
    (lock_mode None — no dispatch-lock window) produce NO token on the
    handler path (stderr AND marker)."""
    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(codex_task, "_post_marker", lambda i, k, n: posted.append((i, k, n)))
    handler = _captured_handler(monkeypatch)
    monkeypatch.setattr(codex_task, "_active_attempt", ("job-x", lock_mode))
    monkeypatch.setattr(codex_task, "_active_issue", 77)
    monkeypatch.setattr(codex_task, "_active_companion", None)
    with pytest.raises(SystemExit):
        handler(int(codex_task.signal.SIGTERM), None)
    err = capsys.readouterr().err
    notes = [n for _, k, n in posted if k == "epm:codex-task-failed"]
    assert len(notes) == 1
    assert "dispatch_lock=" not in err
    assert "dispatch_lock=" not in notes[0]
    assert "job-x" in notes[0]


def test_spawn_exception_and_exit10_no_token_when_held(monkeypatch):
    """Negative arm (D6): the normally-acquired lock ("held") appends NO
    token on the spawn-exception and exit-10 paths (byte-compatible with the
    #2323 note shapes)."""
    monkeypatch.setattr(
        codex_task,
        "_spawn_codex",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    r1 = codex_task._run_one_attempt(Path("/fake/c.mjs"), "p", _args(), False)
    assert r1.exit_code == 3
    assert "dispatch_lock=" not in r1.note

    monkeypatch.setattr(codex_task, "_spawn_codex", lambda *a, **k: "task-h")
    monkeypatch.setattr(
        codex_task,
        "_confirm_job_queryable",
        lambda *a, **k: ("probe-error", "synthetic", None, 4),
    )
    r2 = codex_task._run_one_attempt(Path("/fake/c.mjs"), "p", _args(), False)
    assert r2.exit_code == codex_task.EXIT_POST_SPAWN_PROBE_EXHAUSTED
    assert "dispatch_lock=" not in r2.note


def test_retry_pairing_signal_in_spawn_window_sees_prior_attempt(monkeypatch, capsys):
    """Real-timing retry-pairing test (the round-2/round-3 BLOCKER): a signal
    landing inside attempt 2's spawn window — after with-entry (mode resolved
    = timeout-failopen), BEFORE the post-spawn store — must publish attempt
    1's COMPLETE pair (job-1 + dispatch_lock=disabled) on both the marker and
    stderr, never an attempt-1-job / attempt-2-mode mix. The between-stores
    seam is closed BY CONSTRUCTION (single tuple store) and pinned by the dis
    invariants below; this covers the broad spawn window."""
    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(codex_task, "_post_marker", lambda i, k, n: posted.append((i, k, n)))
    handler = _captured_handler(monkeypatch)
    monkeypatch.setattr(codex_task, "_active_issue", 77)
    monkeypatch.setattr(codex_task, "_active_companion", None)
    monkeypatch.setattr(codex_task, "_active_attempt", None)
    monkeypatch.setattr(codex_task, "_probe_phase", lambda *a, **k: ("done", "", None))
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, "R", ""))
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)

    # Attempt 1: mode "disabled" (real trigger — env kill switch), completes.
    monkeypatch.setenv("EPM_CODEX_DISPATCH_LOCK", "0")
    monkeypatch.setattr(codex_task, "_spawn_codex", lambda *a, **k: "job-1")
    r1 = codex_task._run_one_attempt(Path("/fake/c.mjs"), "p1", _args(issue=77), False)
    assert r1.kind == "done", r1.note
    assert codex_task._active_attempt == ("job-1", "disabled")

    # Attempt 2: kill switch cleared; timeout-failopen forced by a pre-held
    # lock. The spawn stub INVOKES the captured handler inside the spawn
    # window; its sys.exit raises SystemExit (not an Exception), which
    # propagates out of _run_one_attempt.
    monkeypatch.delenv("EPM_CODEX_DISPATCH_LOCK")
    fd = _hold_lock_externally()
    posted.clear()
    capsys.readouterr()  # drain attempt 1's stderr

    def spawn_fires_signal(companion, prompt, effort, write):
        handler(int(codex_task.signal.SIGTERM), None)
        return "job-2"  # unreachable — the handler sys.exit()s

    monkeypatch.setattr(codex_task, "_spawn_codex", spawn_fires_signal)
    try:
        with pytest.raises(SystemExit):
            codex_task._run_one_attempt(
                Path("/fake/c.mjs"),
                "p2",
                _args(issue=77, dispatch_lock_timeout_secs=0.05),
                False,
            )
    finally:
        os.close(fd)

    err = capsys.readouterr().err
    notes = [n for _, k, n in posted if k == "epm:codex-task-failed"]
    assert len(notes) == 1, posted
    # Attempt 1's COMPLETE pair — never job-1 paired with attempt 2's mode.
    assert "job-1" in notes[0] and "dispatch_lock=disabled" in notes[0], notes[0]
    assert "timeout-failopen" not in notes[0], notes[0]
    assert "job-2" not in notes[0], notes[0]
    assert "job-1" in err and "dispatch_lock=disabled" in err
    assert "timeout-failopen" not in err.split("ERROR:")[-1]  # the handler line itself


# Single-store invariants (D6 item 1): the attempt state is ONE tuple global
# assigned by ONE STORE_GLOBAL — a "simplification" back to any two-store
# form (two scalar globals, or `a, b = x, y`, which ALSO emits two stores)
# reopens the between-bytecodes signal window and goes red here.

RETIRED_ATTEMPT_STATE_NAMES = ("_active_job_id", "_active_lock_mode")


def _store_global_counts(code) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ins in dis.get_instructions(code):
        if ins.opname == "STORE_GLOBAL":
            counts[ins.argval] = counts.get(ins.argval, 0) + 1
    return counts


def test_attempt_state_updates_are_single_store():
    """Each attempt-state update path holds EXACTLY ONE STORE_GLOBAL of
    _active_attempt and ZERO stores of any other attempt-state name."""
    for fn in (codex_task._run_one_attempt, codex_task._run_reattach):
        counts = _store_global_counts(fn.__code__)
        assert counts.get("_active_attempt") == 1, (fn.__name__, counts)
        others = {n for n in counts if n.startswith("_active")} - {"_active_attempt"}
        assert not others, (fn.__name__, counts)


def test_retired_scalar_names_absent_from_module():
    """The two retired per-attempt scalar globals no longer exist — not as
    module attributes, not even as source text (the #2324 completeness grep,
    pinned): a stale `global` list would silently make the tuple store a
    function LOCAL and break the handler with no error."""
    for name in RETIRED_ATTEMPT_STATE_NAMES:
        assert not hasattr(codex_task, name), name
    src = (REPO_ROOT / "scripts" / "codex_task.py").read_text()
    for name in RETIRED_ATTEMPT_STATE_NAMES:
        assert name not in src, name


def test_signal_handler_single_snapshot_read():
    """The handler LOADs _active_attempt exactly ONCE (both fields derive
    from the local snapshot) — a second read reopens the pairing window."""
    inner = [
        c
        for c in codex_task._install_signal_handlers.__code__.co_consts
        if isinstance(c, CodeType) and c.co_name == "_handler"
    ]
    assert len(inner) == 1
    loads = [
        ins
        for ins in dis.get_instructions(inner[0])
        if ins.opname == "LOAD_GLOBAL" and ins.argval == "_active_attempt"
    ]
    assert len(loads) == 1, loads
