"""Smoke tests for ``scripts/workflow_lint.py``.

Asserts that the committed ``.claude/workflow.yaml`` lints cleanly so
the /issue HARD GATE (Phase A.0 of the restoration plan, see
``.claude/plans/restore-issue-skill-richness.md``) doesn't silently
regress. The lint covers schema validation, cross-reference
resolution, and AUTO-GENERATED fence-block alignment with SKILL.md
and markers.md.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LINT = _REPO_ROOT / "scripts" / "workflow_lint.py"


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
