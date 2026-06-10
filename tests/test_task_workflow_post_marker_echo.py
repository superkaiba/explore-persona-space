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
