"""Tests for the file-time infra-task dispatch wrapper (#690).

The wrapper (``scripts/file_infra_task.py``) FILES a ripe `kind: infra`/`batch`
task via ``task.py new`` and then BEST-EFFORT dispatches it via
``spawn-issue --auto``. The contract these tests pin:

- **Filing is must-succeed; dispatch is best-effort.** The task is filed even
  when the spawn is skipped or fails (exit 0); only a FAILED ``task.py new``
  exits non-zero.
- **Acceptance (a):** a ripe infra filing on a healthy system files AND issues
  exactly one ``spawn-issue --issue <N> --auto``.
- **Idempotency (c, file-time half):** a task that already has a live session
  is filed but NOT re-dispatched.
- **#690 M1 — shared cap gate:** cap-full OR occupancy-unreadable files the
  task but NEVER spawns (the wrapper can no longer push a 6th session past the
  shared 5-session cap before the watcher's next tick).
- daemon-down no-op; non-infra ``--kind`` rejected; ``--no-dispatch`` files
  only; a failing ``task.py new`` aborts non-zero.

Every subprocess (``task.py new`` / ``spawn-issue``) is mocked — these tests
run in milliseconds and never touch the real task workflow or Happy daemon.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import file_infra_task as fit  # noqa: E402


def _install_run_recorder(monkeypatch, *, new_id="#771", new_rc=0, spawn_rc=0):
    """Replace ``file_infra_task.subprocess.run`` with a recorder that branches
    on the argv: a ``task.py new`` invocation returns a canned ``#<N>`` stdout
    (rc=new_rc); a ``spawn-issue`` invocation returns spawn_rc and records the
    full command. Returns ``calls`` (list of argv lists), so a test asserts
    both WHICH subprocesses ran and their exact shape."""
    calls: list[list[str]] = []

    def _fake_run(cmd, **kw):
        calls.append(list(cmd))
        if "scripts/task.py" in cmd and "new" in cmd:
            return SimpleNamespace(
                returncode=new_rc, stdout=f"{new_id}\n", stderr="boom" if new_rc else ""
            )
        if "scripts/spawn_session.py" in cmd and "spawn-issue" in cmd:
            return SimpleNamespace(
                returncode=spawn_rc,
                stdout="Issue #771 session spawned: sid-new\n",
                stderr="spawn boom" if spawn_rc else "",
            )
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    monkeypatch.setattr(fit.subprocess, "run", _fake_run)
    return calls


def _spawn_calls(calls):
    return [c for c in calls if "scripts/spawn_session.py" in c]


def _new_calls(calls):
    return [c for c in calls if "scripts/task.py" in c and "new" in c]


# ── (a) file-time dispatch fires on a ripe infra filing ────────────────────────


def test_ripe_infra_filing_files_and_dispatches_once(monkeypatch, capsys):
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    monkeypatch.setattr(fit, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(fit, "infra_dispatch_has_free_slot", lambda: True)
    monkeypatch.setattr(fit, "_issue_has_live_registration", lambda issue: False)

    rc = fit.main(["--kind", "infra", "--title", "x"])

    assert rc == 0
    assert len(_new_calls(calls)) == 1
    spawns = _spawn_calls(calls)
    assert len(spawns) == 1
    spawn = spawns[0]
    assert spawn[spawn.index("--issue") + 1] == "771"
    assert "--auto" in spawn
    assert "filed + dispatched #771" in capsys.readouterr().out


# ── (c, file-time) no double-dispatch when a live session exists ───────────────


def test_live_session_files_but_does_not_dispatch(monkeypatch, capsys):
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    monkeypatch.setattr(fit, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(fit, "infra_dispatch_has_free_slot", lambda: True)
    monkeypatch.setattr(fit, "_issue_has_live_registration", lambda issue: True)

    rc = fit.main(["--title", "x"])

    assert rc == 0
    assert len(_new_calls(calls)) == 1  # filed
    assert _spawn_calls(calls) == []  # never spawned
    assert "already has a live session" in capsys.readouterr().out


# ── daemon-unreachable no-op (headless / pod-side) ─────────────────────────────


def test_daemon_unreachable_files_but_does_not_dispatch(monkeypatch, capsys):
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    monkeypatch.setattr(fit, "_daemon_reachable", lambda: False)
    # The cap gate must NOT even be consulted when the daemon is down.
    monkeypatch.setattr(
        fit, "infra_dispatch_has_free_slot", lambda: pytest.fail("cap gate hit despite daemon down")
    )
    monkeypatch.setattr(
        fit,
        "_issue_has_live_registration",
        lambda issue: pytest.fail("registration check hit despite daemon down"),
    )

    rc = fit.main(["--title", "x"])

    assert rc == 0
    assert len(_new_calls(calls)) == 1  # filed
    assert _spawn_calls(calls) == []  # never spawned
    out = capsys.readouterr().out
    assert "Happy daemon unreachable" in out
    assert "proposed_infra_sweep" in out  # names the watcher backstop


# ── #690 M1: cap full -> files, does NOT spawn ─────────────────────────────────


def test_cap_full_files_but_does_not_spawn(monkeypatch, capsys):
    # The direct M1 guard: a wrapper call cannot push a 6th session past the
    # shared 5-session cap before the watcher's next tick.
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    monkeypatch.setattr(fit, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(fit, "infra_dispatch_has_free_slot", lambda: False)  # cap saturated
    monkeypatch.setattr(
        fit,
        "_issue_has_live_registration",
        lambda issue: pytest.fail("registration check reached past a full cap"),
    )

    rc = fit.main(["--title", "x"])

    assert rc == 0
    assert len(_new_calls(calls)) == 1  # STILL files the task
    assert _spawn_calls(calls) == []  # but NEVER spawns
    out = capsys.readouterr().out
    assert "cap (5) full" in out
    assert "proposed_infra_sweep" in out  # names the watcher backstop


# ── #690 M1 companion: occupancy unreadable (None) -> files, does NOT spawn ────


def test_occupancy_unreadable_files_but_does_not_spawn(monkeypatch, capsys):
    # Fail-CLOSED: an unreadable occupancy count never dispatches (a partial
    # read could over-dispatch). Same posture as the executor's
    # `_infra_drain_occupancy() is None -> skip`.
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    monkeypatch.setattr(fit, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(fit, "infra_dispatch_has_free_slot", lambda: None)  # occupancy unreadable
    monkeypatch.setattr(
        fit,
        "_issue_has_live_registration",
        lambda issue: pytest.fail("registration check reached past unreadable occupancy"),
    )

    rc = fit.main(["--title", "x"])

    assert rc == 0
    assert len(_new_calls(calls)) == 1  # STILL files the task
    assert _spawn_calls(calls) == []  # but NEVER spawns
    out = capsys.readouterr().out
    assert "occupancy unreadable" in out
    assert "proposed_infra_sweep" in out


# ── non-infra kind rejected at argparse ────────────────────────────────────────


@pytest.mark.parametrize("kind", ["experiment", "analysis", "campaign", "survey"])
def test_non_infra_kind_rejected_nothing_filed(monkeypatch, kind):
    # A non-{infra,batch} kind must be rejected by argparse BEFORE any
    # subprocess runs — the wrapper must never auto-dispatch a GPU-spending
    # task outside a cap.
    monkeypatch.setattr(
        fit.subprocess, "run", lambda *a, **k: pytest.fail("filed despite a rejected kind")
    )
    with pytest.raises(SystemExit) as exc:
        fit.main(["--kind", kind, "--title", "x"])
    assert exc.value.code != 0


def test_batch_kind_accepted(monkeypatch):
    calls = _install_run_recorder(monkeypatch, new_id="#772")
    monkeypatch.setattr(fit, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(fit, "infra_dispatch_has_free_slot", lambda: True)
    monkeypatch.setattr(fit, "_issue_has_live_registration", lambda issue: False)
    rc = fit.main(["--kind", "batch", "--title", "x"])
    assert rc == 0
    assert _new_calls(calls)[0][_new_calls(calls)[0].index("--kind") + 1] == "batch"
    assert len(_spawn_calls(calls)) == 1


# ── --no-dispatch files only ───────────────────────────────────────────────────


def test_no_dispatch_files_only(monkeypatch, capsys):
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    monkeypatch.setattr(
        fit, "_daemon_reachable", lambda: pytest.fail("daemon probed despite --no-dispatch")
    )
    rc = fit.main(["--title", "x", "--no-dispatch"])
    assert rc == 0
    assert len(_new_calls(calls)) == 1  # filed
    assert _spawn_calls(calls) == []  # never spawned
    assert "dispatch skipped: --no-dispatch" in capsys.readouterr().out


# ── task.py new failure aborts non-zero, never spawns ──────────────────────────


def test_task_new_failure_aborts_nonzero(monkeypatch, capsys):
    calls = _install_run_recorder(monkeypatch, new_rc=2)
    # Filing failed -> the wrapper must exit non-zero and never reach the
    # daemon / cap gates / spawn.
    monkeypatch.setattr(
        fit, "_daemon_reachable", lambda: pytest.fail("daemon probed after filing failed")
    )
    rc = fit.main(["--title", "x"])
    assert rc == 2
    assert _spawn_calls(calls) == []


def test_task_new_unparseable_id_files_but_exits_zero(monkeypatch, capsys):
    # Filing SUCCEEDED (rc 0) but stdout carried no `#<id>` token — the wrapper
    # cannot address the spawn, but the task IS on disk (the backstop sweeps it
    # by status), so this is NOT a filing failure: exit 0, no spawn.
    def _fake_run(cmd, **kw):
        return SimpleNamespace(returncode=0, stdout="created ok, no id token\n", stderr="")

    monkeypatch.setattr(fit.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        fit, "_daemon_reachable", lambda: pytest.fail("daemon probed without a parsed id")
    )
    rc = fit.main(["--title", "x"])
    assert rc == 0
    assert "could not parse its `#<id>`" in capsys.readouterr().err


# ── dispatch subprocess failure -> exit 0 with backstop note ───────────────────


def test_spawn_failure_exits_zero_with_backstop_note(monkeypatch, capsys):
    # A spawn that exits non-zero must NOT make callers think filing failed:
    # the task is filed and the watcher backstop covers it -> exit 0.
    calls = _install_run_recorder(monkeypatch, new_id="#771", spawn_rc=1)
    monkeypatch.setattr(fit, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(fit, "infra_dispatch_has_free_slot", lambda: True)
    monkeypatch.setattr(fit, "_issue_has_live_registration", lambda issue: False)

    rc = fit.main(["--title", "x"])

    assert rc == 0  # NOT non-zero — filing succeeded
    assert len(_new_calls(calls)) == 1
    assert len(_spawn_calls(calls)) == 1  # the spawn WAS attempted
    err = capsys.readouterr().err
    assert "dispatch FAILED" in err
    assert "backstop will retry" in err


# ── argv forwarding ────────────────────────────────────────────────────────────


def test_creation_flags_forwarded_to_task_new(monkeypatch):
    # --tag (repeatable), --parent, --origin-prompt, --body-file are forwarded
    # to `task.py new` so dedup keys / provenance land AT creation.
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    monkeypatch.setattr(fit, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(fit, "infra_dispatch_has_free_slot", lambda: True)
    monkeypatch.setattr(fit, "_issue_has_live_registration", lambda issue: False)

    fit.main(
        [
            "--title",
            "wf-fix: foo",
            "--parent",
            "640",
            "--tag",
            "wf-fix",
            "--tag",
            "wf-fix-fp:abc123",
            "--origin-prompt",
            "verbatim candidate",
            "--body-file",
            "/tmp/body.md",
        ]
    )
    new = _new_calls(calls)[0]
    assert new[new.index("--parent") + 1] == "640"
    assert new.count("--tag") == 2
    assert "wf-fix" in new and "wf-fix-fp:abc123" in new
    assert new[new.index("--origin-prompt") + 1] == "verbatim candidate"
    assert new[new.index("--body-file") + 1] == "/tmp/body.md"


def test_parse_new_id_takes_first_hash_token():
    assert fit._parse_new_id("#771\n") == 771
    assert fit._parse_new_id("created\n#42 done\n") == 42
    assert fit._parse_new_id("no id here") is None
    assert fit._parse_new_id("#notanint") is None
