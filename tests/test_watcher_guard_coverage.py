"""Durability pin for the SHARED #1247 watcher hermeticity guards (task #1265).

This file is itself the demonstration that the two autouse guards in
``tests/conftest.py`` (``_forbid_real_marker_posts``,
``_forbid_real_task_status_reads``) cover a watcher test module with ZERO
per-file ceremony: it defines neither fixture, yet the negative probes below
prove both guards fire here. Deleting the conftest guards (the silent-delete
hazard this task closes) turns the negative probes into real-subprocess
reaches, which fail loudly in any sandboxed run — and the fired-guard asserts
fail outright.

IMPORT-STYLE PIN: the ONLY module-level watcher import is the ``from
autonomous_session_watch import _task_status, _post_progress_marker`` form —
deliberately NO module-level ``import autonomous_session_watch as asw`` — so
this module's guard-fires assertions positively exercise the conftest
predicate's ``__module__`` branch (the module-object branch is exercised by
the 8 sibling files importing ``... as asw``). Tests obtain the live module
handle fixture-locally via ``sys.modules``.

Negative probes pass a guaranteed-nonexistent issue id (10**9) so an
inert-guard failure mode can never post a junk marker on a real task.
"""

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

# Bootstrap sys.path the same way the sibling watcher tests do (scripts/ on
# the path so autonomous_session_watch imports by name).
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from autonomous_session_watch import _post_progress_marker, _task_status  # noqa: E402

# A task id that cannot exist — the negative probes' safety margin.
NONEXISTENT_ISSUE = 10**9


def _asw():
    """The live watcher module handle, obtained WITHOUT a module-level import
    (keeps the module-level namespace free of the module object, so the
    conftest predicate can only match via the imported functions'
    ``__module__`` attribute)."""
    return sys.modules["autonomous_session_watch"]


def test_task_status_guard_fires_in_undecorated_module():
    """Round-2 guard covers a file that never defines it (the durability pin)."""
    with pytest.raises(AssertionError, match="#1247"):
        _asw()._task_status(NONEXISTENT_ISSUE)


def test_marker_post_guard_fires_on_dry_run_false_with_real_subprocess_run():
    """Round-1 guard: dry_run=False with the GENUINE subprocess.run fails loud.

    The raise itself proves subprocess.run is genuine here — the guard only
    raises when its setup-captured ``real_run`` is still live."""
    with pytest.raises(AssertionError, match="#1247"):
        _asw()._post_progress_marker(
            NONEXISTENT_ISSUE, "guard-coverage negative probe", False, label="probe"
        )


def test_marker_post_dry_run_true_passes_through_to_real_log_only_body(capsys):
    """dry_run=True keeps the real body's log-only behavior (no raise, no shell)."""
    _asw()._post_progress_marker(
        NONEXISTENT_ISSUE, "guard-coverage dry-run probe", True, label="probe"
    )
    out = capsys.readouterr().out
    assert "[dry-run] would post epm:progress" in out


def test_marker_post_real_body_allowed_through_with_stubbed_subprocess_run(monkeypatch):
    """The argv-recording carve-out: a STUBBED subprocess.run lets the REAL
    body run hermetically — the recorder sees exactly one call."""
    calls: list[list[str]] = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    _asw()._post_progress_marker(
        NONEXISTENT_ISSUE, "guard-coverage stubbed-run probe", False, label="probe"
    )
    assert len(calls) == 1
    assert "post-marker" in calls[0]
    assert str(NONEXISTENT_ISSUE) in calls[0]


def test_test_level_task_status_override_wins(monkeypatch):
    """A later test-level monkeypatch beats the autouse guard (the documented
    override contract)."""
    asw = _asw()
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    assert asw._task_status(NONEXISTENT_ISSUE) == "running"


def test_guards_preserve_source_inspection_of_original_bodies():
    """#966 pins: functools.wraps keeps __wrapped__, so inspect resolves the
    ORIGINAL bodies through the guards (and unwrap is identity with the
    module-level imported originals)."""
    asw = _asw()
    assert inspect.unwrap(asw._task_status) is _task_status
    assert inspect.unwrap(asw._post_progress_marker) is _post_progress_marker
    assert inspect.unwrap(asw._task_status).__module__ == "autonomous_session_watch"
    # getsource follows __wrapped__: the real body, not the conftest guard.
    src = inspect.getsource(asw._post_progress_marker)
    assert "[dry-run] would post epm:progress" in src
    assert "tests/conftest.py" not in src
