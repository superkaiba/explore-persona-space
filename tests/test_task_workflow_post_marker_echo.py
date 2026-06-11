"""Regression tests for `task.py post-marker` exit-code semantics.

`post_event` appends + commits the marker row BEFORE `cmd_post_event`
echoes the payload JSON to stdout. A BrokenPipeError on that echo (caller
tore the pipe down early — Bash-tool teardown, `| head`, dead SSH) used to
propagate and flip the exit code to nonzero AFTER the commit landed, so
callers that treat rc!=0 as "not posted" (codex_task._post_marker) retried
and duplicated the marker (incident #537, 2026-06-10: duplicate
epm:codex-task-spawned). The echo failure is now non-fatal; pre-commit
failures (oversize note, missing task, flock timeout) stay fatal.

The CLI is exercised at the handler-function layer (not via subprocess) —
see test_task_workflow.py::test_cli_handlers_raise_address_defer_list_roundtrip
for why (the branch-guarded resolver can't be redirected across a process
boundary).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import task as task_cli


class _BrokenPipeStdout(io.TextIOBase):
    """Stand-in for a stdout whose pipe the reader has already torn down."""

    def write(self, _s: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        raise BrokenPipeError

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")


def _ns(**overrides) -> argparse.Namespace:
    base = dict(
        number=537,
        marker="epm:codex-task-spawned",
        version=1,
        by="codex_task",
        note="Codex job_id=task-abc",
        file=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_broken_pipe_on_echo_is_nonfatal_after_commit(monkeypatch, capsys):
    """A BrokenPipeError on the post-commit stdout echo must NOT raise out of
    cmd_post_event — the exit code has to reflect the commit, not the echo."""
    posted = []

    def fake_post_event(number, marker, *, version, by, note):
        posted.append((number, marker, version, by, note))
        return {
            "ts": "2026-06-10T00:00:00Z",
            "kind": marker,
            "version": version,
            "by": by,
            "note": note,
        }

    monkeypatch.setattr(task_cli, "post_event", fake_post_event)
    monkeypatch.setattr(sys, "stdout", _BrokenPipeStdout())

    task_cli.cmd_post_event(_ns())  # must not raise

    assert len(posted) == 1  # the marker write happened exactly once
    err = capsys.readouterr().err
    assert "committed" in err
    assert "BrokenPipeError" in err


def test_pre_commit_failure_stays_fatal(monkeypatch):
    """Failures raised BY post_event (oversize note, missing task) must still
    propagate — only the post-commit echo became non-fatal."""

    def exploding_post_event(*_a, **_k):
        raise ValueError("event note exceeds 50000 chars")

    monkeypatch.setattr(task_cli, "post_event", exploding_post_event)
    with pytest.raises(ValueError):
        task_cli.cmd_post_event(_ns())


def test_normal_echo_prints_payload(monkeypatch, capsys):
    """With a healthy stdout the payload JSON is still echoed unchanged."""
    monkeypatch.setattr(
        task_cli,
        "post_event",
        lambda *_a, **_k: {"kind": "epm:echo-check", "version": 1},
    )
    task_cli.cmd_post_event(_ns(marker="epm:echo-check"))
    out = capsys.readouterr().out
    assert '"kind": "epm:echo-check"' in out


def test_set_status_broken_pipe_on_echo_is_nonfatal(monkeypatch, capsys):
    """The _safe_echo guard covers the other mutating handlers too: a
    BrokenPipeError on cmd_set_status's post-commit path echo must not raise
    (rc reflects the status move, not the echo)."""
    moved = []

    def fake_set_status(number, status, *, note=None, force_followup_exit=False):
        moved.append((number, status, note))
        return Path("/repo/tasks/approved/537")

    monkeypatch.setattr(task_cli, "set_status", fake_set_status)
    monkeypatch.setattr(sys, "stdout", _BrokenPipeStdout())

    ns = argparse.Namespace(number=537, status="approved", note=None)
    task_cli.cmd_set_status(ns)  # must not raise

    assert moved == [(537, "approved", None)]  # the git mv + commit happened exactly once
    err = capsys.readouterr().err
    assert "committed" in err
    assert "BrokenPipeError" in err


def test_set_status_normal_echo_prints_path(monkeypatch, capsys):
    """With a healthy stdout, cmd_set_status still echoes the relative path."""
    monkeypatch.setattr(
        task_cli,
        "set_status",
        lambda number, status, *, note=None, force_followup_exit=False: Path(
            "/repo/tasks/approved/537"
        ),
    )
    ns = argparse.Namespace(number=537, status="approved", note=None)
    task_cli.cmd_set_status(ns)
    out = capsys.readouterr().out
    assert "tasks/approved/537" in out


def test_set_status_followup_hold_refusal_exits_cleanly(monkeypatch):
    """The library's same-issue follow-up status-hold ValueError must surface
    as a clean SystemExit (message, nonzero rc) — not a raw traceback."""

    def refusing_set_status(number, status, *, note=None, force_followup_exit=False):
        raise ValueError("followups_running is HELD ... (status-hold rule)")

    monkeypatch.setattr(task_cli, "set_status", refusing_set_status)
    ns = argparse.Namespace(number=537, status="running", note=None)
    with pytest.raises(SystemExit) as exc_info:
        task_cli.cmd_set_status(ns)
    assert "status-hold" in str(exc_info.value)


def test_set_status_plan_gate_holds_at_followups_running(monkeypatch, capsys):
    """A --auto-approve-if-autonomous plan-gate call on a followups_running
    task fires the gate decision + marker but NEVER moves the status
    (status-hold rule, SKILL.md Step 9b § Same-issue follow-up loop step 3)."""
    moved = []
    posted = []
    monkeypatch.setattr(
        task_cli,
        "set_status",
        lambda number, status, *, note=None, force_followup_exit=False: moved.append(
            (number, status)
        ),
    )
    monkeypatch.setattr(
        task_cli,
        "get_task",
        lambda number: {"status": "followups_running", "frontmatter": {"tags": []}},
    )

    def fake_post_event(number, marker, *, version, by, note):
        posted.append((number, marker))
        return {"kind": marker, "version": version}

    monkeypatch.setattr(task_cli, "post_event", fake_post_event)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "24")

    ns = argparse.Namespace(
        number=537,
        status="plan_pending",
        note=None,
        auto_approve_if_autonomous=True,
        gpu_hours=4.0,
    )
    task_cli.cmd_set_status(ns)

    assert moved == []  # the status flip never happened
    assert posted == [(537, "epm:plan-approved")]  # the gate decision still landed
    out = capsys.readouterr().out
    assert "followups_running hold: status unchanged" in out


def test_set_status_plan_gate_hold_parked_over_cap(monkeypatch, capsys):
    """The over-cap sub-branch of the plan-gate hold: posts
    epm:awaiting-spend-approval, never moves the status."""
    moved = []
    posted = []
    monkeypatch.setattr(
        task_cli,
        "set_status",
        lambda number, status, *, note=None, force_followup_exit=False: moved.append(
            (number, status)
        ),
    )
    monkeypatch.setattr(
        task_cli,
        "get_task",
        lambda number: {"status": "followups_running", "frontmatter": {"tags": []}},
    )

    def fake_post_event(number, marker, *, version, by, note):
        posted.append((number, marker))
        return {"kind": marker, "version": version}

    monkeypatch.setattr(task_cli, "post_event", fake_post_event)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "24")

    ns = argparse.Namespace(
        number=537,
        status="plan_pending",
        note=None,
        auto_approve_if_autonomous=True,
        gpu_hours=200.0,  # over the 24h cap
    )
    task_cli.cmd_set_status(ns)

    assert moved == []
    assert posted == [(537, "epm:awaiting-spend-approval")]
    out = capsys.readouterr().out
    assert "parked_over_cap" in out
    assert "followups_running hold: status unchanged" in out


def test_set_status_followups_running_missing_tag_warns(monkeypatch, capsys):
    """Transitioning TO followups_running without a followup-auto/-manual tag
    prints the missing-tag WARNING (a bare `followup` tag does not count)."""
    monkeypatch.setattr(
        task_cli,
        "set_status",
        lambda number, status, *, note=None, force_followup_exit=False: Path(
            "/repo/tasks/followups_running/537"
        ),
    )
    monkeypatch.setattr(
        task_cli,
        "get_task",
        lambda number: {"status": "followups_running", "frontmatter": {"tags": ["followup"]}},
    )
    ns = argparse.Namespace(number=537, status="followups_running", note=None)
    task_cli.cmd_set_status(ns)
    out = capsys.readouterr().out
    assert "WARNING: transitioned to followups_running without a" in out

    # And with the proper tag present, no warning.
    monkeypatch.setattr(
        task_cli,
        "get_task",
        lambda number: {
            "status": "followups_running",
            "frontmatter": {"tags": ["followup-manual"]},
        },
    )
    task_cli.cmd_set_status(ns)
    out = capsys.readouterr().out
    assert "WARNING" not in out
