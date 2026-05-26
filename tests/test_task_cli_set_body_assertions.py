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

import importlib.util
import subprocess
import sys
from pathlib import Path

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
