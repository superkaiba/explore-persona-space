"""Regression test: no direct `tasks/` path construction outside the
canonical resolver.

Enforces success criterion #6 from the 2026-05-25 worktree-staleness
plan: any new `from explore_persona_space.task_workflow import TASKS_DIR
| REGISTRY_PATH | REPO` bare-name import, any `PROJECT_ROOT / "tasks"`
construction, and any `ROOT / "tasks"` construction outside the
explicit allowlist fails the test.

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
``test_task_workflow_worktree.py``) and this CI grep test that keeps new
bare-name imports out.

Allowlist
---------

A line is allowed if any of these hold:

  * the line carries an explicit `# ALLOWED: <reason>` comment;
  * the file is the resolver itself (`src/.../task_workflow.py`) or
    the audit script (`scripts/audit_stranded_task_commits.py` reads
    every worktree branch to enumerate stranded commits — by design);
  * the file is THIS test, the worktree resolver test (which exercises
    `tw.TASKS_DIR` PEP-562 access on purpose), or the legacy
    `test_task_workflow.py` (its docstring mentions the deprecated
    constants);
  * the file lives under `external/`, `archive/`, `eval_results/`,
    `.claude/worktrees/`, `.venv/`, or `.git/`;
  * the file extension is not `.py` (we only scope to Python).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Patterns that the resolver-aware code must NOT use directly.
_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"from\s+\S*task_workflow\s+import\s+\(?\s*(\w+\s*,\s*)*(TASKS_DIR|REGISTRY_PATH|REPO)\b",
        "bare-name import of TASKS_DIR / REGISTRY_PATH / REPO (binds at import time; "
        "PEP-562 cannot rescue it). Use the function form: "
        "`from explore_persona_space.task_workflow import tasks_dir, registry_path, repo_root`.",
    ),
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

# Files allowed to use the patterns (resolver, audit, this test, worktree test,
# legacy test docs). Paths are relative to REPO_ROOT.
_FILE_ALLOWLIST = frozenset(
    {
        "src/explore_persona_space/task_workflow.py",
        "scripts/audit_stranded_task_commits.py",
        "tests/test_no_direct_task_path_construction.py",
        "tests/test_task_workflow_worktree.py",
        # The legacy test file's docstring/fixture references the old
        # constants by name; refactoring it would just expand the diff.
        "tests/test_task_workflow.py",
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


def _scan(pattern: str) -> list[tuple[str, int, str]]:
    """Return (rel_path, lineno, line_text) for every match that is NOT
    allowlisted. Returns [] when nothing matches.

    Implementation: pure-Python ``re`` scan over every `.py` file. This
    keeps the test dependency-free (no `rg` install requirement) and
    works under `uv run pytest` out of the box.
    """
    import re

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


@pytest.mark.parametrize(("pattern", "rationale"), _PATTERNS)
def test_no_direct_task_path_construction(pattern: str, rationale: str) -> None:
    """Fail if any non-allowlisted file matches an anti-pattern.

    Run locally with: `uv run pytest tests/test_no_direct_task_path_construction.py -v`.
    """
    bad = _scan(pattern)
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
