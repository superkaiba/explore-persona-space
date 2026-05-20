"""Smoke tests for ``scripts/workflow_lint.py``.

Asserts that the committed ``.claude/workflow.yaml`` lints cleanly so
the /issue HARD GATE (Phase A.0 of the restoration plan, see
``.claude/plans/restore-issue-skill-richness.md``) doesn't silently
regress. The lint covers schema validation, cross-reference
resolution, and AUTO-GENERATED fence-block alignment with SKILL.md
and markers.md.

Also covers the ``--check-asks`` mode: every ``AskUserQuestion``
mention in .claude/agents/**.md and .claude/skills/**/SKILL.md must
be anchored to a documented gate (task #372).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LINT = _REPO_ROOT / "scripts" / "workflow_lint.py"
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import check_asks  # noqa: E402

from explore_persona_space.workflow import load_workflow_yaml  # noqa: E402


def _run(*flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", str(_LINT), *flags],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_workflow_lint_default_exits_zero():
    """No-args invocation must succeed (schema-only check)."""
    result = _run()
    assert result.returncode == 0, (
        f"workflow_lint default failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_workflow_lint_check_references_exits_zero():
    """The HARD GATE: every ``(see workflow.yaml § X)`` reference in
    CLAUDE.md / SKILL.md / markers.md must resolve to a real key. This
    is the gate that Phase A's restored SKILL.md depends on; if it
    regresses, the restored cross-refs are dangling."""
    result = _run("--check-references")
    assert result.returncode == 0, (
        f"workflow_lint --check-references failed (HARD GATE regressed):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_workflow_lint_check_tables_exits_zero():
    """The AUTO-GENERATED fence blocks in SKILL.md and markers.md must
    match the renderer's output (no hand-edits inside the fences)."""
    result = _run("--check-tables")
    assert result.returncode == 0, (
        f"workflow_lint --check-tables failed (AUTO-GENERATED tables drifted):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_workflow_lint_check_asks_repo_passes():
    """Repo-level check: the committed agent + skill specs must already
    satisfy the auto-continuation contract. If this fails, the audit
    cleanup from task #372 has regressed (someone added a bare
    AskUserQuestion mention outside any gate)."""
    result = _run("--check-asks")
    assert result.returncode == 0, (
        f"workflow_lint --check-asks failed at repo scope:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────
# Unit tests for the ``check_asks`` function (task #372).
# Each case writes a tiny markdown file under ``tmp_path``, calls
# ``check_asks(workflow, roots=[tmp_path])``, and inspects the error
# list. PASS = empty list; FAIL = at least one error string.
# ─────────────────────────────────────────────────────────────────────


def _workflow():
    return load_workflow_yaml(_REPO_ROOT / ".claude" / "workflow.yaml")


def test_check_asks_pass_inline_gate_annotation(tmp_path):
    """PASS — line carries an inline ``<!-- gate: gates.plan_approval -->``
    annotation that resolves to a real workflow.yaml gate."""
    (tmp_path / "SKILL.md").write_text(
        "Use `AskUserQuestion` for plan approval. <!-- gate: gates.plan_approval -->\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_asks_pass_gate_annotation_line_above(tmp_path):
    """PASS — annotation on the line immediately above the mention."""
    (tmp_path / "SKILL.md").write_text(
        "<!-- gate: gates.worktree_merge -->\n"
        "Ask via `AskUserQuestion`: should we merge the worktree?\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_asks_fail_unannotated(tmp_path):
    """FAIL — bare ``AskUserQuestion`` mention with no annotation, no
    anti-pattern marker, and no gate citation in the paragraph."""
    (tmp_path / "SKILL.md").write_text(
        "Whenever you feel like it, just use `AskUserQuestion` and the user will reply.\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert len(errors) == 1, f"expected 1 error, got: {errors}"
    assert "bare 'AskUserQuestion'" in errors[0]


def test_check_asks_fail_nonexistent_gate_key(tmp_path):
    """FAIL — ``<!-- gate: ... -->`` annotation references a key that
    does NOT resolve in workflow.yaml § gates."""
    (tmp_path / "SKILL.md").write_text(
        "Use `AskUserQuestion`. <!-- gate: gates.NONEXISTENT_GATE -->\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert len(errors) == 1, f"expected 1 error, got: {errors}"
    assert "does not" in errors[0] and "resolve" in errors[0]


def test_check_asks_pass_anti_pattern_marker(tmp_path):
    """PASS — paragraph carries the ``<!-- example: anti-pattern -->``
    marker, signalling this is documentation of misuse, not a live call
    site."""
    (tmp_path / "SKILL.md").write_text(
        "<!-- example: anti-pattern -->\n"
        "Do NOT use `AskUserQuestion` outside the documented gates.\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_asks_pass_existing_workflow_yaml_citation(tmp_path):
    """PASS — paragraph already cites a gate via the existing
    ``(see workflow.yaml § gates.X)`` convention; no need to also stamp
    a redundant ``<!-- gate: ... -->`` annotation."""
    (tmp_path / "SKILL.md").write_text(
        "The clarifier gate (see workflow.yaml § gates.clarifier_blocking)\n"
        "is implemented by asking the user via `AskUserQuestion`.\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_asks_mixed_file_passes_and_fails(tmp_path):
    """Multi-mention file: properly annotated mentions PASS, bare
    mentions FAIL with line-specific errors."""
    (tmp_path / "SKILL.md").write_text(
        # line 1: PASS via gate annotation
        "Use `AskUserQuestion` here. <!-- gate: gates.plan_approval -->\n"
        # line 2: PASS via anti-pattern marker on line above
        "<!-- example: anti-pattern -->\n"
        "Do NOT call `AskUserQuestion` outside gates.\n"
        # line 4: blank
        "\n"
        # line 5: FAIL — bare, no annotation, no citation
        "Stray `AskUserQuestion` mention without anchor.\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert len(errors) == 1, f"expected exactly 1 error, got: {errors}"
    assert ":5:" in errors[0]


def test_check_asks_pass_anti_pattern_marker_after_mention(tmp_path):
    """The anti-pattern marker MUST be at or above the mention — markers
    that appear AFTER the mention do not anchor it. This test guards
    against a regression where the lookback window is accidentally
    flipped to a look-ahead."""
    (tmp_path / "SKILL.md").write_text(
        "Stray `AskUserQuestion` mention with marker below.\n<!-- example: anti-pattern -->\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert len(errors) == 1, f"expected 1 error, got: {errors}"
    assert ":1:" in errors[0]
