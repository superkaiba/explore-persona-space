"""SIGPIPE-safe stdout for `task.py`'s read-only commands (#786 item d).

Read-only commands (`view`, `list-*`, `find`, `tasks-dir`, `audit` report
mode, `migrate-body` report/dry-run) emit multi-line output that a downstream
`| head` / `| grep -q .` routinely closes early. The resulting BrokenPipeError
used to surface as a traceback and flip the exit code; `_safe_print` swallows
it (one-line stderr notice + stdout→devnull) so a torn pipe is a clean no-op
for a read-only command. `_safe_print` mirrors `_safe_echo` but generalizes to
arbitrary args and puts `sys.stdout.flush()` INSIDE the try — a buffered-pipe
BrokenPipeError often only surfaces at flush/close, not at the `print` call.

Exercised in-process at the handler-function layer (not via subprocess) —
mirroring test_task_workflow_post_marker_echo.py, since the branch-guarded
resolver can't be redirected across a process boundary.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

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


def test_safe_print_survives_broken_pipe(monkeypatch, capsys):
    """A BrokenPipeError on a read-only stdout emit must NOT escape
    _safe_print; a suppression notice goes to stderr instead."""
    monkeypatch.setattr(sys, "stdout", _BrokenPipeStdout())
    task_cli._safe_print("hello", context="task.py view")  # must not raise
    err = capsys.readouterr().err
    assert "BrokenPipeError" in err
    assert "task.py view" in err


def test_safe_print_flush_broken_pipe_is_caught(monkeypatch, capsys):
    """The buffered-pipe case: write() succeeds but flush() raises
    BrokenPipeError (the flush is INSIDE the try, so it is caught). This is
    the case a print-only guard without an in-try flush would miss."""

    class _FlushOnlyBreaks(io.TextIOBase):
        def write(self, s: str) -> int:  # write succeeds
            return len(s)

        def flush(self) -> None:  # the pipe tears down at flush time
            raise BrokenPipeError

        def fileno(self) -> int:
            raise io.UnsupportedOperation("fileno")

    monkeypatch.setattr(sys, "stdout", _FlushOnlyBreaks())
    task_cli._safe_print("hello", context="task.py view")  # must not raise
    err = capsys.readouterr().err
    assert "BrokenPipeError" in err


def test_safe_print_normal_stdout_prints(capsys):
    """With a healthy stdout the args are printed unchanged."""
    task_cli._safe_print("plain line", context="task.py view")
    assert "plain line" in capsys.readouterr().out


def test_cmd_view_json_survives_broken_pipe(monkeypatch, capsys):
    """cmd_view --json against a torn stdout pipe must not raise a traceback —
    the _safe_print conversion covers the JSON emit path."""
    fake_task = {
        "id": 786,
        "path": "tasks/running/786",
        "status": "running",
        "frontmatter": {"title": "X", "kind": "infra", "tags": []},
        "body": "goal\n",
    }
    monkeypatch.setattr(task_cli, "get_task", lambda _n: fake_task)
    monkeypatch.setattr(task_cli, "list_events", lambda _n: [])
    monkeypatch.setattr(sys, "stdout", _BrokenPipeStdout())

    ns = argparse.Namespace(number=786, json=True, rich=False)
    task_cli.cmd_view(ns)  # must not raise

    err = capsys.readouterr().err
    assert "BrokenPipeError" in err


def test_cmd_view_plaintext_survives_broken_pipe(monkeypatch, capsys):
    """cmd_view plaintext (the multi-line branch a `| head` most often tears)
    against a torn stdout pipe must not raise."""
    fake_task = {
        "id": 786,
        "path": "tasks/running/786",
        "status": "running",
        "frontmatter": {"title": "X", "kind": "infra", "tags": [], "parent_id": 42},
        "body": "goal\n",
    }
    events = [
        {"ts": "2026-07-01T00:00:00Z", "kind": "epm:status-changed", "note": "moved"}
        for _ in range(12)
    ]
    monkeypatch.setattr(task_cli, "get_task", lambda _n: fake_task)
    monkeypatch.setattr(task_cli, "list_events", lambda _n: events)
    monkeypatch.setattr(sys, "stdout", _BrokenPipeStdout())

    ns = argparse.Namespace(number=786, json=False, rich=False)
    task_cli.cmd_view(ns)  # must not raise

    err = capsys.readouterr().err
    assert "BrokenPipeError" in err
