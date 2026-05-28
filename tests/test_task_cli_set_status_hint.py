"""Tests for the `set-status` invalid-status hint in `scripts/task.py`.

`set-status` deliberately does NOT use argparse `choices=STATUSES`. argparse's
bare `invalid choice: 'uploading' (choose from ...)` dump has repeatedly been
misread by callers who then invent statuses like `uploading` / `api`. Instead
the CLI handler `cmd_set_status` validates the value itself and raises
`SystemExit` with a `_status_error_message` that (a) names the bad value,
(b) suggests the closest valid enum member when one exists, and (c) always
lists the full valid enum.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

# Load `scripts/task.py` as a module so we can hit `_status_error_message`
# and `cmd_set_status` directly without going through the CLI parser.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "task.py"
_spec = importlib.util.spec_from_file_location("task_cli_status", _SCRIPT)
task_cli = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["task_cli_status"] = task_cli
_spec.loader.exec_module(task_cli)  # type: ignore[union-attr]


# ─── Direct unit tests on `_status_error_message` ──────────────────────────


def test_status_error_lists_full_enum():
    """The message always lists every valid status, regardless of suggestion."""
    msg = task_cli._status_error_message("uploading")
    for status in task_cli.STATUSES:
        assert status in msg, f"{status!r} missing from error message"


def test_status_error_names_the_bad_value():
    """The offending value is quoted in the message so the caller sees it."""
    msg = task_cli._status_error_message("uploading")
    assert "uploading" in msg
    assert "invalid status" in msg


def test_status_error_suggests_closest_for_typo():
    """A near-miss typo gets a 'did you mean <closest>?' line."""
    msg = task_cli._status_error_message("aproved")
    assert "did you mean" in msg
    assert "approved" in msg


def test_status_error_suggests_running_for_typo():
    """`runing` -> `running` via difflib close-match."""
    msg = task_cli._status_error_message("runing")
    assert "did you mean" in msg
    assert "'running'" in msg


def test_status_error_omits_suggestion_when_no_close_match():
    """A value with no close match (e.g. 'api') omits the suggestion line
    gracefully but still lists the full enum."""
    msg = task_cli._status_error_message("api")
    assert "did you mean" not in msg
    # Full enum still present.
    assert "proposed" in msg
    assert "completed" in msg


# ─── `cmd_set_status` raises SystemExit on invalid status ──────────────────


def _ns(**kw):
    """Build a minimal argparse-like namespace for cmd_set_status."""
    import argparse

    return argparse.Namespace(**kw)


def test_cmd_set_status_rejects_invalid_before_mutation():
    """cmd_set_status validates the status BEFORE calling set_status, so an
    invalid value never touches task state — it raises SystemExit with the
    helpful message regardless of task number existence."""
    args = _ns(number=999999, status="uploading", note=None)
    with pytest.raises(SystemExit) as exc:
        task_cli.cmd_set_status(args)
    msg = str(exc.value)
    assert "uploading" in msg
    assert "valid statuses" in msg


def test_cmd_set_status_accepts_valid_status_passes_validation():
    """A valid status passes the validation guard (it then calls set_status,
    which may fail on a nonexistent task — but NOT with our status error)."""
    args = _ns(number=999999, status="running", note=None)
    # Should not raise our status SystemExit. It MAY raise something else
    # (e.g. task-not-found) from set_status — that's a different failure and
    # proves the status guard let a valid value through.
    try:
        task_cli.cmd_set_status(args)
    except SystemExit as e:  # pragma: no cover - defensive
        assert "valid statuses" not in str(e), (
            "valid status 'running' should not trip the invalid-status guard"
        )
    except Exception:
        # task-not-found / git error from set_status is acceptable here:
        # the point is the status guard did not block a valid value.
        pass


# ─── CLI-level smoke tests ─────────────────────────────────────────────────


def _run_cli(*args):
    """Run `uv run python scripts/task.py <args>` and return (rc, stdout, stderr)."""
    cmd = ["uv", "run", "python", str(_SCRIPT), *args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_set_status_invalid_emits_hint():
    """`task.py set-status 1 uploading` exits non-zero and prints the
    did-you-mean hint + full enum (NOT argparse's bare 'invalid choice')."""
    rc, stdout, stderr = _run_cli("set-status", "1", "uploading")
    assert rc != 0
    combined = stdout + stderr
    assert "invalid status" in combined
    assert "valid statuses" in combined
    # argparse's stock phrasing must NOT be what the user sees.
    assert "invalid choice" not in combined


def test_cli_set_status_help_lists_valid_statuses():
    """`task.py set-status --help` advertises the valid status values."""
    rc, stdout, _ = _run_cli("set-status", "--help")
    assert rc == 0
    assert "proposed" in stdout
    assert "completed" in stdout
