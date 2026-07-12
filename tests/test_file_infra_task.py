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


@pytest.fixture(autouse=True)
def _no_real_lease(monkeypatch):
    """Hermeticity: the #843 step-4.5 dispatch-lease pre-check reads the
    registry dir; default it to "no lease held" so these tests never depend
    on the REAL ~/.eps-autonomous. The lease-specific test overrides it."""
    monkeypatch.setattr(fit, "dispatch_lease_fresh", lambda issue: None)


@pytest.fixture(autouse=True)
def _no_real_stagger(monkeypatch):
    """Hermeticity for the #1059 step-4.75 session-dispatch stagger: default
    the stamp read to "no prior dispatch" (the live VM's real
    ~/.eps-autonomous stamp could be fresh on a busy fleet -> flaky defers)
    and RECORD — never write — the stamp on a successful spawn. Returns the
    recorded issue ids; stagger-specific tests override / read these."""
    recorded: list[int] = []
    monkeypatch.setattr(fit, "last_session_dispatch_age_s", lambda now=None: None)
    monkeypatch.setattr(
        fit, "record_session_dispatch", lambda issue, holder, now=None: recorded.append(issue)
    )
    return recorded


def _install_run_recorder(
    monkeypatch,
    *,
    new_id="#771",
    new_rc=0,
    spawn_rc=0,
    spawn_stdout="Issue #771 session spawned: sid-new\n",
):
    """Replace ``file_infra_task.subprocess.run`` with a recorder that branches
    on the argv: a ``task.py new`` invocation returns a canned ``#<N>`` stdout
    (rc=new_rc); a ``spawn-issue`` invocation returns spawn_rc/spawn_stdout and
    records the full command. Returns ``calls`` (list of argv lists), so a test
    asserts both WHICH subprocesses ran and their exact shape."""
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
                stdout=spawn_stdout,
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


# ── #843 M1: fresh dispatch lease -> files, does NOT dispatch ──────────────────


def test_fresh_lease_files_but_does_not_dispatch(monkeypatch, capsys):
    # Test 19: a fresh per-issue dispatch lease means another dispatcher's
    # spawn is already in flight — the wrapper files the task but skips the
    # spawn subprocess entirely (loud, exit 0, watcher backstop named).
    import time

    calls = _install_run_recorder(monkeypatch, new_id="#771")
    monkeypatch.setattr(fit, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(fit, "infra_dispatch_has_free_slot", lambda: True)
    monkeypatch.setattr(fit, "_issue_has_live_registration", lambda issue: False)
    monkeypatch.setattr(
        fit,
        "dispatch_lease_fresh",
        lambda issue: {"holder": "watcher-sweep", "pid": 1, "acquired_at": time.time() - 30},
    )

    rc = fit.main(["--title", "x"])

    assert rc == 0
    assert len(_new_calls(calls)) == 1  # STILL files the task
    assert _spawn_calls(calls) == []  # but NEVER spawns
    out = capsys.readouterr().out
    assert "dispatch suppressed (lease held" in out
    assert "holder=watcher-sweep" in out
    assert "proposed_infra_sweep" in out  # names the watcher backstop


def test_spawn_suppressed_output_reported_not_booked_as_dispatched(monkeypatch, capsys):
    # #843 M1b: a rc-0 spawn whose stdout carries a suppression sentinel (a
    # lease appeared between the pre-check and the spawn, or a registration
    # collision) is reported as suppressed, never as "filed + dispatched".
    from types import SimpleNamespace

    def _fake_run(cmd, **kw):
        if "scripts/task.py" in cmd and "new" in cmd:
            return SimpleNamespace(returncode=0, stdout="#771\n", stderr="")
        if "scripts/spawn_session.py" in cmd and "spawn-issue" in cmd:
            return SimpleNamespace(
                returncode=0,
                stdout="DISPATCH-LEASE HELD issue #771: a dispatch is already in flight\n",
                stderr="",
            )
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    monkeypatch.setattr(fit.subprocess, "run", _fake_run)
    monkeypatch.setattr(fit, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(fit, "infra_dispatch_has_free_slot", lambda: True)
    monkeypatch.setattr(fit, "_issue_has_live_registration", lambda issue: False)

    rc = fit.main(["--title", "x"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "dispatch suppressed (DISPATCH-LEASE HELD)" in out
    assert "filed + dispatched" not in out


# ── #1059: session-dispatch stagger -> files, DEFERS the spawn ─────────────────


def _healthy_dispatch_env(monkeypatch):
    """Stub every pre-spawn gate BEFORE the stagger to 'go' so a test isolates
    the step-4.75 stagger behavior."""
    monkeypatch.setattr(fit, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(fit, "infra_dispatch_has_free_slot", lambda: True)
    monkeypatch.setattr(fit, "_issue_has_live_registration", lambda issue: False)


def test_fresh_dispatch_stamp_files_but_defers_spawn(monkeypatch, capsys):
    # A session dispatch 5s ago (< the 60s window) -> the task is FILED but
    # the spawn is DEFERRED (loud line, exit 0, watcher backstop named).
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    _healthy_dispatch_env(monkeypatch)
    monkeypatch.setattr(fit, "last_session_dispatch_age_s", lambda now=None: 5.0)

    rc = fit.main(["--title", "x"])

    assert rc == 0
    assert len(_new_calls(calls)) == 1  # STILL files the task
    assert _spawn_calls(calls) == []  # but NEVER spawns
    out = capsys.readouterr().out
    assert "dispatch deferred (session-dispatch stagger" in out
    assert "proposed_infra_sweep backstop" in out


def test_stagger_disabled_dispatches(monkeypatch):
    # EPM_SESSION_DISPATCH_STAGGER_S=0 disables the defer even under a fresh
    # stamp: exactly one spawn.
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    _healthy_dispatch_env(monkeypatch)
    monkeypatch.setattr(fit, "last_session_dispatch_age_s", lambda now=None: 5.0)
    monkeypatch.setenv("EPM_SESSION_DISPATCH_STAGGER_S", "0")

    rc = fit.main(["--title", "x"])

    assert rc == 0
    assert len(_spawn_calls(calls)) == 1


def test_successful_spawn_records_stamp(monkeypatch, _no_real_stagger):
    # Only a REAL spawn records the pacing stamp — with the filed issue's id.
    _install_run_recorder(monkeypatch, new_id="#771")
    _healthy_dispatch_env(monkeypatch)

    rc = fit.main(["--title", "x"])

    assert rc == 0
    assert _no_real_stagger == [771]


def test_suppressed_spawn_does_not_record_stamp(monkeypatch, _no_real_stagger, capsys):
    # A rc-0 suppressed no-op (lease held at the chokepoint) is NOT a real
    # spawn: no stamp record, so a no-op can never defer real work elsewhere.
    _install_run_recorder(
        monkeypatch,
        new_id="#771",
        spawn_stdout="DISPATCH-LEASE HELD issue #771: a dispatch is already in flight\n",
    )
    _healthy_dispatch_env(monkeypatch)

    rc = fit.main(["--title", "x"])

    assert rc == 0
    assert _no_real_stagger == []
    assert "dispatch suppressed" in capsys.readouterr().out


def test_filer_stagger_integration_real_helpers(monkeypatch, tmp_path, capsys):
    # Integration-style: the REAL stagger helpers (env window + stamp read)
    # against a tmp registry — only AUTONOMOUS_REGISTRY_DIR is patched. A
    # fresh REAL stamp written via record_session_dispatch defers the spawn.
    import time as _time

    import spawn_session

    calls = _install_run_recorder(monkeypatch, new_id="#771")
    _healthy_dispatch_env(monkeypatch)
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    # Undo the autouse stub: point the wrapper back at the real helper (it
    # resolves AUTONOMOUS_REGISTRY_DIR at call time, so tmp_path binds).
    monkeypatch.setattr(
        fit, "last_session_dispatch_age_s", spawn_session.last_session_dispatch_age_s
    )
    spawn_session.record_session_dispatch(99, "test-prior", now=_time.time() - 5.0)

    rc = fit.main(["--title", "x"])

    assert rc == 0
    assert len(_new_calls(calls)) == 1
    assert _spawn_calls(calls) == []
    assert "dispatch deferred (session-dispatch stagger" in capsys.readouterr().out


# ── #1173: warn-only backstop for wf-fix bodies missing workflow_fix_target ────


def test_wf_fix_tag_body_missing_target_line_warns(monkeypatch, capsys, tmp_path):
    # A wf-fix-tagged filing whose body lacks the durable recursion-guard
    # `workflow_fix_target:` Provenance line WARNS on stderr — exit code and
    # filing/dispatch behavior unchanged (warn-only by design, #1173).
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    _healthy_dispatch_env(monkeypatch)
    body = tmp_path / "body.md"
    body.write_text("## Goal\n\nfix a thing\n", encoding="utf-8")

    # prefixed title: keep this test isolated to the Provenance warn (#1283)
    rc = fit.main(["--title", "workflow-fix: x", "--tag", "wf-fix", "--body-file", str(body)])

    assert rc == 0
    assert len(_new_calls(calls)) == 1  # filing still happens
    assert len(_spawn_calls(calls)) == 1  # dispatch behavior unchanged
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "workflow_fix_target" in err


def test_wf_fix_tag_body_with_target_line_no_warn(monkeypatch, capsys, tmp_path):
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    _healthy_dispatch_env(monkeypatch)
    body = tmp_path / "body.md"
    body.write_text(
        "## Goal\n\nfix\n\n## Provenance\n\n- workflow_fix_target: CLAUDE.md\n",
        encoding="utf-8",
    )

    # prefixed title: keep this test isolated to the Provenance warn (#1283)
    rc = fit.main(["--title", "workflow-fix: x", "--tag", "wf-fix", "--body-file", str(body)])

    assert rc == 0
    assert len(_new_calls(calls)) == 1
    assert "workflow_fix_target" not in capsys.readouterr().err


# ── #1283: warn-only backstop for wf-fix titles missing a channel prefix ──────

_PREFIXED_PROVENANCE_BODY = "## Goal\n\nfix\n\n## Provenance\n\n- workflow_fix_target: CLAUDE.md\n"


def test_wf_fix_tag_title_without_prefix_warns(monkeypatch, capsys, tmp_path):
    # A wf-fix-tagged filing whose --title lacks a WF_FIX_TITLE_PREFIXES
    # prefix WARNS on stderr (the title is invisible to the dedup predicate's
    # title pre-filter) — exit code and filing/dispatch behavior unchanged
    # (warn-only by design, #1283). Body carries the Provenance target line so
    # ONLY the title warn is under test; tags use the production multi-tag
    # shape (wf-fix + wf-fix-fp:<fp>).
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    _healthy_dispatch_env(monkeypatch)
    body = tmp_path / "body.md"
    body.write_text(_PREFIXED_PROVENANCE_BODY, encoding="utf-8")

    rc = fit.main(
        [
            "--title",
            "no prefix here",
            "--tag",
            "wf-fix",
            "--tag",
            "wf-fix-fp:abc123",
            "--body-file",
            str(body),
        ]
    )

    assert rc == 0
    assert len(_new_calls(calls)) == 1  # filing still happens
    assert len(_spawn_calls(calls)) == 1  # dispatch behavior unchanged
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "--title lacks a" in err
    assert "workflow_fix_target" not in err  # the #1173 warn did not co-fire


@pytest.mark.parametrize("prefix", ["workflow-fix: ", "daily-fix: "])
def test_wf_fix_tag_title_with_prefix_no_warn(monkeypatch, capsys, tmp_path, prefix):
    # Either channel prefix satisfies the guard (WF_FIX_TITLE_PREFIXES tuple).
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    _healthy_dispatch_env(monkeypatch)
    body = tmp_path / "body.md"
    body.write_text(_PREFIXED_PROVENANCE_BODY, encoding="utf-8")

    rc = fit.main(["--title", f"{prefix}x", "--tag", "wf-fix", "--body-file", str(body)])

    assert rc == 0
    assert len(_new_calls(calls)) == 1
    assert "--title lacks a" not in capsys.readouterr().err


def test_non_wf_fix_filing_any_title_no_warn(monkeypatch, capsys, tmp_path):
    # No wf-fix tag -> any title is fine (ordinary infra filings are not in
    # the wf-fix dedup key space; the guard keys on the tag, not the kind).
    calls = _install_run_recorder(monkeypatch, new_id="#771")
    _healthy_dispatch_env(monkeypatch)
    body = tmp_path / "body.md"
    body.write_text("## Goal\n\nordinary infra\n", encoding="utf-8")

    rc = fit.main(["--title", "no prefix here", "--body-file", str(body)])

    assert rc == 0
    assert len(_new_calls(calls)) == 1
    assert "--title lacks a" not in capsys.readouterr().err


def test_title_prefix_guard_reads_shared_constant(monkeypatch, capsys, tmp_path):
    # The guard keys off task_workflow.WF_FIX_TITLE_PREFIXES (the imported
    # constant), never a baked-in literal (#1283 AC3): under a patched
    # sentinel tuple, a sentinel-prefixed title passes and the REAL channel
    # prefix warns — both directions follow the patched value.
    _install_run_recorder(monkeypatch, new_id="#771")
    _healthy_dispatch_env(monkeypatch)
    monkeypatch.setattr(fit, "WF_FIX_TITLE_PREFIXES", ("sentinel:",))
    body = tmp_path / "body.md"
    body.write_text(_PREFIXED_PROVENANCE_BODY, encoding="utf-8")

    rc = fit.main(["--title", "sentinel: x", "--tag", "wf-fix", "--body-file", str(body)])
    assert rc == 0
    assert "--title lacks a" not in capsys.readouterr().err

    rc = fit.main(["--title", "workflow-fix: x", "--tag", "wf-fix", "--body-file", str(body)])
    assert rc == 0
    assert "--title lacks a" in capsys.readouterr().err
