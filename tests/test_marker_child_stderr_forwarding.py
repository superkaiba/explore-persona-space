"""Tests for the #1130 rc==0 post-marker child-stderr forwarding at the
secondary call sites: ``spawn_session._post_duplicate_suppressed_marker``
and ``autonomous_session_watch._forward_marker_child_stderr`` (plus its
``_post_progress_marker`` integration).

``task.py post-marker`` deliberately exits 0 while printing the
deferred-commit ERROR and the #1100 post-commit LANDING CHECK warning to
stderr; ``capture_output=True`` at these call sites used to discard both,
so they reached no transcript. The forwarding writes them to the wrapper's
stderr, prefixed and capped; control flow (rc handling, ``check=True``
semantics, return values) is unchanged. Primary-site coverage
(``codex_task._post_marker``) lives in ``tests/test_codex_task_post_marker.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
import spawn_session  # noqa: E402

_LANDING_CHECK_LINE = (
    "task.py LANDING CHECK: commit abc123 ('epm:x') is NOT reachable from refs/heads/main"
)


# ──────────────────────────────────────────────────────────────────────
# spawn_session._post_duplicate_suppressed_marker
# ──────────────────────────────────────────────────────────────────────


def test_spawn_session_duplicate_marker_forwards_nonempty_stderr(monkeypatch, capsys):
    """Non-empty child stderr → forwarded with the `[post-marker stderr]`
    prefix (rc deliberately unchecked — best-effort post, unchanged)."""
    calls = []
    monkeypatch.setattr(
        spawn_session.subprocess,
        "run",
        lambda *a, **k: (
            calls.append(a) or SimpleNamespace(returncode=0, stdout="", stderr=_LANDING_CHECK_LINE)
        ),
    )

    spawn_session._post_duplicate_suppressed_marker(1130, "sid-kept", "sid-stopped")

    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "[post-marker stderr]" in err
    assert "task.py LANDING CHECK" in err


def test_spawn_session_duplicate_marker_empty_stderr_silent(monkeypatch, capsys):
    """Empty child stderr (the common case) → zero new output."""
    monkeypatch.setattr(
        spawn_session.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    spawn_session._post_duplicate_suppressed_marker(1130, "sid-kept", "sid-stopped")

    assert capsys.readouterr().err == ""


# ──────────────────────────────────────────────────────────────────────
# autonomous_session_watch._forward_marker_child_stderr (unit)
# ──────────────────────────────────────────────────────────────────────


def test_watch_helper_forwards_nonempty_stderr(capsys):
    """Non-empty rc==0 stderr → per-line `[task.py stderr] {context}:` prefix."""
    asw._forward_marker_child_stderr(
        SimpleNamespace(returncode=0, stdout="", stderr=_LANDING_CHECK_LINE),
        "epm:progress on #1130",
    )

    err = capsys.readouterr().err
    assert "[task.py stderr] epm:progress on #1130:" in err
    assert "task.py LANDING CHECK" in err


def test_watch_helper_empty_or_missing_stderr_silent(capsys):
    """Empty / None / absent stderr all forward nothing and never raise."""
    asw._forward_marker_child_stderr(SimpleNamespace(stderr=""), "ctx")
    asw._forward_marker_child_stderr(SimpleNamespace(stderr=None), "ctx")
    asw._forward_marker_child_stderr(SimpleNamespace(), "ctx")  # no stderr attribute at all

    assert capsys.readouterr().err == ""


# ──────────────────────────────────────────────────────────────────────
# Integration: _post_progress_marker reaches the helper on rc==0.
# ──────────────────────────────────────────────────────────────────────


def test_watch_post_progress_marker_integration_forwards(monkeypatch, capsys):
    """dry_run=False (the dry-run branch returns before subprocess.run) with a
    stubbed rc==0 + non-empty-stderr child → the warning reaches the
    watcher's stderr; exactly one subprocess invocation (check=True
    semantics untouched — the stub does not raise)."""
    calls = []
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda *a, **k: (
            calls.append((a, k))
            or SimpleNamespace(returncode=0, stdout="", stderr=_LANDING_CHECK_LINE)
        ),
    )

    asw._post_progress_marker(1130, "pod-safety note", False, label="auto-stop")

    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "[task.py stderr]" in err
    assert "task.py LANDING CHECK" in err
