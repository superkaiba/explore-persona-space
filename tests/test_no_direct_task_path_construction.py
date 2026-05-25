"""Regression test: no direct `tasks/` path construction outside the
canonical resolver.

Enforces success criterion #6 from the 2026-05-25 worktree-staleness
plan: any new `from explore_persona_space.task_workflow import TASKS_DIR
| REGISTRY_PATH | REPO` bare-name import (including the parenthesized
multi-line form), any `PROJECT_ROOT / "tasks"` construction, and any
`ROOT / "tasks"` construction outside the explicit allowlist fails the
test.

Why this is load-bearing
------------------------

PEP-562 module ``__getattr__`` keeps ``tw.TASKS_DIR`` attribute access
working lazily, but ``from tw import TASKS_DIR`` binds the value at
import time and PEP-562 cannot rescue it — the caller has already
captured whatever the resolver returned the first time the module was
imported. That value is wrong when subsequent processes run inside a
worktree on a non-`main` branch, which is the bug class the plan was
written to eliminate.

The fix is twofold: the resolver itself (covered by
``test_task_workflow_worktree.py``) and this CI scan that keeps new
bare-name imports out.

Implementation
--------------

The bare-name import scan is **AST-based** (not line-regex). A line-by-line
regex would miss the canonical multi-line form::

    from explore_persona_space.task_workflow import (
        foo,
        TASKS_DIR,
    )

``ast.parse`` + ``ImportFrom`` walking handles that and ``import X as Y``
aliasing without ambiguity. The two ``... / "tasks"`` constructions stay
on line-regex (single-line patterns).

Allowlist
---------

A line/file is allowed if any of these hold:

  * the line carries an explicit `# ALLOWED: <reason>` comment;
  * the file is the resolver itself (`src/.../task_workflow.py`) or
    the worktree resolver test (which exercises `tw.TASKS_DIR` PEP-562
    access on purpose);
  * THIS test is allowlisted for the line-regex pieces (it contains the
    pattern strings themselves);
  * the file lives under `external/`, `archive/`, `eval_results/`,
    `.claude/worktrees/`, `.venv/`, or `.git/`;
  * the file extension is not `.py` (we only scope to Python).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Forbidden bare names when imported via `from <...>task_workflow import ...`.
_FORBIDDEN_BARE_NAMES = frozenset({"TASKS_DIR", "REGISTRY_PATH", "REPO"})

# Line-regex patterns (single-line constructions only).
_LINE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r'PROJECT_ROOT\s*/\s*"tasks"',
        'direct `PROJECT_ROOT / "tasks"` construction. From a worktree, '
        "PROJECT_ROOT may be the worktree dir, not main. Use "
        "`from explore_persona_space.task_workflow import tasks_dir`.",
    ),
    (
        r'\bROOT\s*/\s*"tasks"',
        'direct `ROOT / "tasks"` construction. From a worktree, ROOT may be '
        "the worktree dir, not main. Use `from explore_persona_space.task_workflow "
        "import tasks_dir`.",
    ),
)

# Files allowed to use the patterns. Paths are relative to REPO_ROOT.
# Trimmed 2026-05-25 round 2: dropped `tests/test_task_workflow.py` and
# `scripts/audit_stranded_task_commits.py` because neither file actually
# triggers any pattern (verified by grep). Allowlisting non-offenders
# weakens enforcement against future drift.
_FILE_ALLOWLIST = frozenset(
    {
        "src/explore_persona_space/task_workflow.py",
        "tests/test_no_direct_task_path_construction.py",
        "tests/test_task_workflow_worktree.py",
    }
)

# Directory prefixes to exclude entirely. Match against relative paths
# (REPO_ROOT-relative, POSIX-style).
_DIR_EXCLUDES = (
    "external/",
    "archive/",
    "eval_results/",
    ".claude/worktrees/",
    ".venv/",
    ".git/",
    "node_modules/",
    "ood_eval_results/",
)


def _allowed(rel_path: str, line: str) -> bool:
    if rel_path in _FILE_ALLOWLIST:
        return True
    if any(rel_path.startswith(prefix) for prefix in _DIR_EXCLUDES):
        return True
    return "# ALLOWED:" in line


# Top-level subtrees the test scans. Keep tight — scanning every `.py`
# under REPO_ROOT pulls in `.venv/`, `.git/`, every checkpoint snapshot,
# and the `.claude/worktrees/` shadow trees, which both inflates run
# time to minutes AND defeats the dir-exclude filter when those paths
# are followed through symlinks. We scope to the three directories that
# could plausibly construct `tasks/` paths.
_SCAN_ROOTS: tuple[str, ...] = ("src", "scripts", "tests")


def _walk_py_files() -> list[Path]:
    """Yield every `.py` file under the SCAN_ROOTS, honoring `_DIR_EXCLUDES`."""
    out: list[Path] = []
    for root_name in _SCAN_ROOTS:
        root = REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            try:
                rel = path.resolve().relative_to(REPO_ROOT).as_posix()
            except ValueError:
                continue
            if any(rel.startswith(prefix) for prefix in _DIR_EXCLUDES):
                continue
            out.append(path)
    return out


def _scan_line_regex(pattern: str) -> list[tuple[str, int, str]]:
    """Return (rel_path, lineno, line_text) for every regex match that is
    NOT allowlisted. Used for the single-line ``... / "tasks"`` patterns.
    """

    rx = re.compile(pattern)
    bad: list[tuple[str, int, str]] = []
    for path in _walk_py_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        for i, line in enumerate(text.splitlines(), start=1):
            if not rx.search(line):
                continue
            if _allowed(rel, line):
                continue
            bad.append((rel, i, line))
    return bad


def _scan_ast_bare_imports() -> list[tuple[str, int, str]]:
    """AST-walk every `.py` file for forbidden bare-name imports of
    ``TASKS_DIR`` / ``REGISTRY_PATH`` / ``REPO`` from any module whose
    dotted name ends with ``task_workflow``.

    AST-based (not line-regex) so the canonical multi-line form is
    caught::

        from explore_persona_space.task_workflow import (
            foo,
            TASKS_DIR,
        )

    Returns (rel_path, lineno, offending_name) per offence. The
    ``# ALLOWED:`` per-line escape hatch is preserved by re-reading the
    offending line and checking it.
    """
    bad: list[tuple[str, int, str]] = []
    for path in _walk_py_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            # Don't gate enforcement on files we can't parse — but don't
            # silently swallow either. Report as a violation so the dev
            # notices.
            bad.append((rel, 0, "<unparseable: ast.SyntaxError>"))
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            # Match `task_workflow` as the trailing dotted segment so
            # both `task_workflow` and `explore_persona_space.task_workflow`
            # are caught.
            if not (module == "task_workflow" or module.endswith(".task_workflow")):
                continue
            for alias in node.names:
                if alias.name not in _FORBIDDEN_BARE_NAMES:
                    continue
                lineno = node.lineno  # the `from ... import` line
                # ALLOWED escape hatch: the line carrying the bare name
                # must opt in explicitly. We check both the import-line
                # itself AND the line carrying the alias (multi-line).
                allowed_here = False
                if rel in _FILE_ALLOWLIST or any(
                    rel.startswith(prefix) for prefix in _DIR_EXCLUDES
                ):
                    allowed_here = True
                else:
                    # Re-scan the from..import block (lineno through
                    # end_lineno) for the per-line escape.
                    end = node.end_lineno or node.lineno
                    block = "\n".join(lines[node.lineno - 1 : end])
                    if "# ALLOWED:" in block:
                        allowed_here = True
                if allowed_here:
                    continue
                bad.append((rel, lineno, alias.name))
    return bad


@pytest.mark.parametrize(("pattern", "rationale"), _LINE_PATTERNS)
def test_no_direct_task_path_construction_line_regex(pattern: str, rationale: str) -> None:
    """Fail if any non-allowlisted file matches a single-line anti-pattern.

    Run locally with: `uv run pytest tests/test_no_direct_task_path_construction.py -v`.
    """
    bad = _scan_line_regex(pattern)
    if bad:
        lines = "\n".join(f"  - {p}:{ln}: {txt.strip()}" for p, ln, txt in bad)
        raise AssertionError(
            f"\n{len(bad)} file(s) violate the canonical-resolver rule.\n"
            f"\n{rationale}\n"
            f"\nMatches:\n{lines}\n"
            f"\nRemediation: replace direct path construction with "
            f"`from explore_persona_space.task_workflow import tasks_dir, "
            f"registry_path, repo_root` and the function form.\n"
        )


def test_no_bare_name_imports_from_task_workflow() -> None:
    """Fail on `from <...>task_workflow import TASKS_DIR | REGISTRY_PATH |
    REPO` in any form (single-line, multi-line parenthesized, or
    aliased) outside the file-level allowlist.

    AST-based so the multi-line form is caught — that was the line-regex
    blind-spot flagged in round-1 code review.
    """
    bad = _scan_ast_bare_imports()
    if bad:
        lines = "\n".join(f"  - {p}:{ln}: imports `{name}`" for p, ln, name in bad)
        raise AssertionError(
            f"\n{len(bad)} bare-name import(s) violate the canonical-resolver rule.\n"
            f"\nBare-name import of TASKS_DIR / REGISTRY_PATH / REPO binds at "
            f"import time; PEP-562 cannot rescue it. Use the function form:\n"
            f"  `from explore_persona_space.task_workflow import "
            f"tasks_dir, registry_path, repo_root`\n"
            f"\nOffences:\n{lines}\n"
        )
