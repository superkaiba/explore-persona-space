"""Tests for the CLI-level non-trivial-body assertion in `scripts/task.py`.

The library-level `set_body()` in `explore_persona_space.task_workflow` is
intentionally permissive (creation-time stubs, snapshot path, etc.). The
CLI handler `cmd_set_body` adds a thin guard via `_assert_body_nontrivial`
to catch the cache → body.md silent-handoff failure mode (incident: task
#385, 2026-05-25).

The assertion rejects bodies that are:
  - < MIN_BODY_CHARS (500) characters, OR
  - a literal stub token (`placeholder` / `tbd` / `todo` / `stub`) after
    whitespace strip.

Bodies WITHOUT a leading `# <title>` H1 are allowed at the CLI level —
the verifier (`scripts/verify_task_body.py`) enforces the H1 requirement
for clean-result bodies only. Non-clean-result bodies (proposed-task
auto-drafts, idea captures, clarifier output) routinely start with `##`
or bold paragraphs and must not be spuriously rejected.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

# Load `scripts/task.py` as a module so we can hit `_assert_body_nontrivial`
# directly without going through the CLI parser.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "task.py"
_spec = importlib.util.spec_from_file_location("task_cli", _SCRIPT)
task_cli = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["task_cli"] = task_cli
_spec.loader.exec_module(task_cli)  # type: ignore[union-attr]


# ─── Direct unit tests on `_assert_body_nontrivial` ────────────────────────


def test_assert_body_accepts_no_h1_but_500_chars():
    """A body ≥500 chars that starts with `## Idea` (no H1) MUST pass.

    Regression test for the Major-1 bug in commit 1bf5df71: the round-1
    H1 check broke every proposed-task `set-body` call because legitimate
    non-clean-result bodies (e.g. `tasks/proposed/114/body.md`) start with
    `## Idea`, not `# <title>`. The fix dropped the H1 sub-check from
    `_assert_body_nontrivial` — H1 enforcement now lives only in
    `verify_task_body.py` (clean-result bodies only).
    """
    body = "## Idea\n\nThis is the idea capture body.\n\n" + ("More content here. " * 30)
    assert len(body) >= 500
    # Should NOT raise SystemExit.
    task_cli._assert_body_nontrivial(body, source="/tmp/no-h1-body.md")


def test_assert_body_accepts_bold_paragraph_lead_no_h1():
    """A proposed-task body starting with `**From EXPERIMENT_QUEUE.md...**`
    (no H1) MUST pass. Mirrors `tasks/proposed/1/body.md`.
    """
    body = (
        "**From EXPERIMENT_QUEUE.md, added 2026-04-16**\n\n"
        "Conceptual / analysis task. Current work treats persona direction as "
        "monolithic — likely entangles identity, style, and capability "
        "(e.g. 'scholarly' persona also implies higher factual recall).\n\n"
        + ("Decomposition candidates: ICA, sparse decomposition, etc. " * 10)
    )
    assert len(body) >= 500
    task_cli._assert_body_nontrivial(body, source="/tmp/bold-lead-body.md")


def test_assert_body_rejects_short_body():
    """A body < 500 chars rejects with a clear message naming the floor."""
    body = "# Short title\n\nReal but short body — only 50 chars."
    assert len(body) < 500
    with pytest.raises(SystemExit) as exc:
        task_cli._assert_body_nontrivial(body, source="/tmp/short.md")
    msg = str(exc.value)
    assert "suspiciously short" in msg
    assert "floor is 500" in msg
    assert "--allow-stub" in msg


def test_assert_body_rejects_short_placeholder_via_length_check():
    """A short body that happens to be `placeholder` rejects.

    The 11-char body trips the <500-char length check FIRST (which fires
    before the stub-token check in `_assert_body_nontrivial`), so the
    FAIL message we see is the length-floor error rather than the
    stub-token error. To exercise the stub-token branch we need a body
    that's ≥500 chars AND collapses to a stub token after strip (e.g.
    trailing whitespace padding) — covered in
    `test_assert_body_rejects_placeholder_with_whitespace_padding`.
    """
    body = "placeholder"
    with pytest.raises(SystemExit) as exc:
        task_cli._assert_body_nontrivial(body, source="/tmp/stub.md")
    msg = str(exc.value)
    assert "suspiciously short" in msg or "literal stub token" in msg


def test_assert_body_rejects_placeholder_with_whitespace_padding():
    """A 500+ char body that collapses to `placeholder` after strip
    rejects via the stub-token branch."""
    body = "placeholder" + (" " * 600)
    assert len(body) >= 500
    with pytest.raises(SystemExit) as exc:
        task_cli._assert_body_nontrivial(body, source="/tmp/padded-stub.md")
    msg = str(exc.value)
    assert "literal stub token" in msg
    assert "placeholder" in msg


def test_assert_body_rejects_tbd_case_insensitive():
    """Stub-token check is case-insensitive — `TBD` rejects."""
    body = "TBD" + ("\n" * 600)
    with pytest.raises(SystemExit) as exc:
        task_cli._assert_body_nontrivial(body, source="/tmp/tbd.md")
    assert "literal stub token" in str(exc.value)


def test_assert_body_accepts_concatenated_stub_token():
    """A 500+ char body that happens to contain `placeholderplaceholder...`
    is NOT a literal stub token — the check matches only the EXACT
    stripped content.
    """
    body = "placeholder" * 50  # 550 chars of concatenated stub tokens
    assert len(body) >= 500
    assert body.strip().casefold() not in task_cli._SET_BODY_STUB_TOKENS
    # Should NOT raise.
    task_cli._assert_body_nontrivial(body, source="/tmp/concat-stub.md")


# ─── CLI-level smoke tests ─────────────────────────────────────────────────


def _run_cli(*args, input_text: str | None = None):
    """Run `uv run python scripts/task.py <args>` and return (rc, stdout, stderr)."""
    cmd = ["uv", "run", "python", str(_SCRIPT), *args]
    proc = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_help_lists_allow_stub_flag():
    """`task.py set-body --help` advertises the `--allow-stub` escape and
    correctly describes the two checks it bypasses (length, stub token).
    """
    rc, stdout, _ = _run_cli("set-body", "--help")
    assert rc == 0
    assert "--allow-stub" in stdout
    assert "<500-char" in stdout or "<500 char" in stdout
    assert "stub" in stdout.lower()
    # The old H1 claim must not appear — the help text was corrected after
    # the Major-1 review found legitimate non-clean-result bodies have no
    # H1.
    assert "H1 line" not in stdout
    assert "must start with" not in stdout


# ─── Goal-H2 drop guard: CLI threading (incident #1112) ────────────────────
#
# The guard itself lives in the library (`task_workflow.set_body`, covered
# by tests/test_task_workflow.py — the real body is executed there, so the
# seam-stub obligation is satisfied); these tests pin the CLI layer: flag
# threading through `cmd_set_body`, the clean SystemExit on refusal, and
# the REAL argparse registration of `--allow-goal-drop`.


def _set_body_namespace(**overrides) -> argparse.Namespace:
    """Build the args namespace `cmd_set_body` reads (mirrors the set-body
    subparser's attribute set; parser registration itself is pinned by
    `test_cli_set_body_parser_registers_allow_goal_drop`)."""
    ns = argparse.Namespace(
        number=999_999_999,  # nonexistent task: the paper-check get_task raise is caught
        body="x",
        file=None,
        snapshot=False,
        allow_stub=True,  # skip the length guard; these tests target the goal-drop path
        allow_goal_drop=False,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _autospec_get_task_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the paper-exemption `get_task` probe hermetic (signature-conformant
    autospec raising KeyError — the branch `cmd_set_body` already catches)."""
    monkeypatch.setattr(
        task_cli,
        "get_task",
        mock.create_autospec(task_cli.get_task, side_effect=KeyError("no such task")),
    )


def test_cli_set_body_threads_allow_goal_drop_flag(monkeypatch):
    """`cmd_set_body` forwards `allow_goal_drop` to the library `set_body`."""
    stub = mock.create_autospec(task_cli.set_body)
    monkeypatch.setattr(task_cli, "set_body", stub)
    _autospec_get_task_raising(monkeypatch)
    ns = _set_body_namespace(allow_goal_drop=True)
    task_cli.cmd_set_body(ns)
    stub.assert_called_once_with(ns.number, "x", snapshot_original=False, allow_goal_drop=True)


def test_cli_set_body_goal_drop_refusal_is_clean_systemexit(monkeypatch):
    """A `GoalH2DropError` from the library surfaces as a clean SystemExit
    carrying the refusal message (no raw traceback path) — the same style
    as the `--allow-stub` guard."""
    refusal = task_cli.GoalH2DropError(
        "set-body refused for task #1: the new body removes the '## Goal' H2 "
        "(incident #1112); pass allow_goal_drop=True / --allow-goal-drop."
    )
    stub = mock.create_autospec(task_cli.set_body, side_effect=refusal)
    monkeypatch.setattr(task_cli, "set_body", stub)
    _autospec_get_task_raising(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        task_cli.cmd_set_body(_set_body_namespace())
    assert str(refusal) in str(exc.value)
    assert exc.value.__cause__ is refusal


def test_cli_set_body_parser_registers_allow_goal_drop(monkeypatch, tmp_path):
    """Parse through the REAL argparse parser `main()` builds — a forgotten
    subparser registration would pass the namespace-built tests above and
    crash production on `args.allow_goal_drop` (AttributeError)."""
    captured: dict = {}
    monkeypatch.setattr(task_cli, "cmd_set_body", lambda args: captured.update(args=args))
    body_file = tmp_path / "b.md"
    body_file.write_text("x")
    monkeypatch.setattr(
        sys, "argv", ["task.py", "set-body", "1", "--file", str(body_file), "--allow-goal-drop"]
    )
    task_cli.main()
    assert captured["args"].allow_goal_drop is True
    monkeypatch.setattr(sys, "argv", ["task.py", "set-body", "1", "--file", str(body_file)])
    task_cli.main()
    assert captured["args"].allow_goal_drop is False
