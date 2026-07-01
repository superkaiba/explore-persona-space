#!/usr/bin/env python
"""Select the touched-file-scoped pytest subset for the ``/issue`` Step 9c test-verdict gate.

The full project suite (~5800 tests in ~248 files, no ``pytest-xdist``
parallelism) stalls and is harness-/earlyoom-killed in sparse worktrees
(#665/#736), so Step 9c must NOT run ``pytest tests/`` wholesale by default.
This helper computes a deterministic subset = the tests covering the files THIS
branch changed (``git diff --name-only <base>...HEAD``) UNION a pinned literal
list of workflow-invariant tests (``WORKFLOW_INVARIANT``) that gate the
project's load-bearing invariants regardless of which files were touched.

It mechanizes the hand-picked subset #736 ran by hand. The selection is pure
deterministic path arithmetic over a git diff — no model call, no new
dependency (stdlib + git only).

Touched-file -> test mapping (per touched file ``f``):
  * a WORKFLOW_SURFACE glob (``.claude/**``, ``CLAUDE.md``, ``tasks/**``,
    docs/figures/artifacts) -> SKIP. The WORKFLOW_INVARIANT set IS the gate
    for these; there is no per-file test map and they are not "untested".
  * a data/config/doc file (``.json`` / ``.yaml`` / ``.md`` / ... ) -> SKIP
    (not code; no test mapping; not "untested").
  * ``tests/test_<X>.py`` -> include ``f`` itself.
  * a code file (``scripts/<X>.py`` / ``src/.../<X>.py`` / any other ``*.py``):
    map ``stem = Path(f).stem`` to ``tests/test_{stem}.py`` (exact) PLUS the
    broad ``tests/test_*{stem}*.py`` glob. If NEITHER matches an existing test
    -> the file lands in ``untested_touched`` (a loud WARN the Step 9c marker
    surfaces; never a silent coverage hole).

``safe-by-direction``: the broad ``*{stem}*`` glob arm deliberately OVER-matches
on short stems (e.g. stem ``gcp`` selects ``test_gcp_backend.py``) — running a
few extra invariant-adjacent tests is the safe failure direction. There is no
``else: tighten the glob`` arm: a missed mapping surfaces as an
``untested_touched`` WARN, never a silently-dropped file.

Degenerate empty-diff (e.g. run on ``main`` with no commits ahead): the
selection falls back to the WORKFLOW_INVARIANT set only (selection always
includes ``always``), so the gate never runs zero tests.

Usage::

    uv run python scripts/select_step9c_tests.py [--base main] [--repo-root <path>] [--json]

Default output: the exact pytest invocation
``uv run pytest <files...> -v --tb=short`` on stdout, then any
``untested touched file: <path>`` WARN lines on stderr. ``--json`` emits
``{"tests": [...], "untested_touched": [...], "base": "...",
"missing_invariants": [...]}``. Exit 0 on success (even with WARN lines);
exit 1 only if the underlying ``git diff`` fails irrecoverably.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# --- Pinned workflow-invariant tests (plan §5 + 1 brief addition, 32 files). -
# A module-level literal tuple, NOT a glob: a future ``tests/test_workflowish.py``
# that is NOT meant to gate Step 9c must not silently join the gate, and the gate
# must not silently shrink if a glob arm stops matching. Drift is made loud by the
# on-disk existence check (a vanished entry prints WORKFLOW-INVARIANT MISSING) and
# pinned against the live tree by tests/test_select_step9c_tests.py.
WORKFLOW_INVARIANT: tuple[str, ...] = (
    # group 1 — task-workflow API
    "tests/test_task_workflow.py",
    "tests/test_task_workflow_list_children.py",
    "tests/test_task_workflow_post_marker_echo.py",
    "tests/test_task_workflow_worktree.py",
    # group 2 — workflow-lint / yaml / fix-dedup
    "tests/test_workflow_lint.py",
    "tests/test_workflow_lint_dotenv_check.py",
    "tests/test_workflow_yaml.py",
    "tests/test_workflow_fix_dedup.py",
    "tests/test_workflow_hub_upload_as_file.py",
    # group 3 — test_no_* invariants
    "tests/test_no_auto_runpod_path_under_any_failure.py",
    "tests/test_no_direct_task_path_construction.py",
    "tests/test_no_dollar_budget_caps.py",
    "tests/test_no_per_file_raw_completions_loop.py",
    "tests/test_no_pod_side_task_py_shellout.py",
    # group 4 — verifiers
    "tests/test_verify_plan.py",
    "tests/test_verify_task_body.py",
    "tests/test_verify_task_body_audit_claim.py",
    "tests/test_verify_clean_result.py",
    "tests/test_verify_clean_result_body_stdin.py",
    "tests/test_verify_paper.py",
    "tests/test_verify_uploads_card_fallback.py",
    "tests/test_verify_uploads_claimed_urls.py",
    "tests/test_verify_uploads_type_selection.py",
    # group 5 — gate / dispatch / contract
    "tests/test_failure_classifier.py",
    "tests/test_sparse_worktree.py",
    "tests/test_autonomous_plan_gate.py",
    "tests/test_autonomous_session_watch.py",
    "tests/test_issue_skill_marker_contract.py",
    "tests/test_plan_handoff_path_convention.py",
    "tests/test_clean_result_critic_planned_vs_actual.py",
    "tests/test_check_no_secret_shaped_strings.py",
    "tests/test_check_mcp_json_no_secrets.py",
)

# --- Touched files that short-circuit (no per-file test map). ----------------
# These gate via the WORKFLOW_INVARIANT set, not a per-file test, so a touched
# file matching one of these is SKIPPED (and is NOT an "untested" file).
WORKFLOW_SURFACE_GLOBS: tuple[str, ...] = (
    ".claude/agents/*.md",
    ".claude/skills/**/SKILL.md",
    ".claude/skills/**/*.md",
    ".claude/skills/**/*.json",
    ".claude/rules/*.md",
    ".claude/workflow.yaml",
    ".claude/settings*.json",
    ".claude/agent-memory/**/*.md",
    "CLAUDE.md",
    "tasks/**",  # task state — never code
    "docs/**",
    "figures/**",
    "eval_results/**",
    "ood_eval_results/**",
    "raw/**",
)

# Data / config / doc extensions: not code, no test mapping, never "untested".
_DATA_DOC_SUFFIXES: frozenset[str] = frozenset(
    {".json", ".yaml", ".yml", ".md", ".txt", ".csv", ".toml", ".lock", ".png", ".svg", ".pdf"}
)


def _matches_any(path: str, globs: tuple[str, ...]) -> bool:
    """True if *path* matches any glob. ``**`` matches across directory separators."""
    for g in globs:
        if fnmatch.fnmatch(path, g):
            return True
        # fnmatch treats ``*`` greedily across ``/`` already, but a ``foo/**``
        # pattern needs an explicit prefix check for the zero-segment case
        # (``docs/**`` should match ``docs/x.md``).
        if g.endswith("/**") and (path == g[:-3] or path.startswith(g[:-2])):
            return True
    return False


def compute_touched(
    base: str,
    repo_root: Path,
    _runner: Callable[[list[str]], str] | None = None,
) -> list[str]:
    """Return the repo-relative paths the current branch changed vs *base*.

    Uses the three-dot ``git diff --name-only <base>...HEAD`` form: it diffs the
    merge-base of *base* and HEAD against HEAD — exactly the branch's own
    additions/modifications, not changes on *base* that HEAD lacks. ``_runner``
    is injectable for tests (it receives the argv list and returns stdout).
    """

    def _default_runner(argv: list[str]) -> str:
        proc = subprocess.run(
            argv,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout

    runner = _runner or _default_runner
    out = runner(["git", "diff", "--name-only", f"{base}...HEAD"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def select_tests(touched: list[str], repo_root: Path) -> tuple[list[str], list[str]]:
    """Map *touched* files to their covering tests + the workflow-invariant set.

    Returns ``(tests, untested_touched)``:
      * ``tests`` — sorted union of mapped per-file tests and the present-on-disk
        subset of :data:`WORKFLOW_INVARIANT` (deterministic order; required so two
        invocations on the same git state return an identical list).
      * ``untested_touched`` — non-trivial touched code files with no mapped test
        (a WARN list the Step 9c marker surfaces; never silently dropped).
    """
    selected: set[str] = set()
    untested: list[str] = []

    for f in touched:
        # Workflow-surface files gate via the invariant set; not "untested".
        if _matches_any(f, WORKFLOW_SURFACE_GLOBS):
            continue
        p = Path(f)
        # A touched test file includes itself.
        if f.startswith("tests/") and p.name.startswith("test_") and p.suffix == ".py":
            if (repo_root / f).exists():
                selected.add(f)
            continue
        # Data / config / doc files: not code, no test mapping.
        if p.suffix in _DATA_DOC_SUFFIXES:
            continue
        # Code files (.py anywhere): map stem -> test_<stem>.py + test_*<stem>*.py.
        if p.suffix == ".py":
            stem = p.stem
            matched = False
            exact = repo_root / "tests" / f"test_{stem}.py"
            if exact.exists():
                selected.add(f"tests/test_{stem}.py")
                matched = True
            for hit in sorted((repo_root / "tests").glob(f"test_*{stem}*.py")):
                selected.add(f"tests/{hit.name}")
                matched = True
            if not matched:
                untested.append(f)
        # Any other extension (no recognized mapping): ignore silently — it is
        # neither code with a test nor a workflow-invariant surface.

    present_invariant = [t for t in WORKFLOW_INVARIANT if (repo_root / t).exists()]
    final = sorted(selected | set(present_invariant))
    return final, untested


def missing_invariants(repo_root: Path) -> list[str]:
    """Return WORKFLOW_INVARIANT entries that are NOT present on disk (drift)."""
    return [t for t in WORKFLOW_INVARIANT if not (repo_root / t).exists()]


def _resolve_repo_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    # #506-safe: from a worktree cwd, `git rev-parse --show-toplevel` returns the
    # WORKTREE root; --git-common-dir resolves to <main>/.git, so dirname is the
    # main repo root (where tests/ paths must resolve). See SKILL.md Steps 4a/10d.
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip()).parent.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", default="main", help="diff base (default: main)")
    parser.add_argument("--repo-root", default=None, help="repo root (default: git toplevel)")
    parser.add_argument("--json", action="store_true", help="emit a JSON object")
    args = parser.parse_args(argv)

    repo_root = _resolve_repo_root(args.repo_root)

    try:
        touched = compute_touched(args.base, repo_root)
    except subprocess.CalledProcessError as exc:
        # Fail loud — never silently fall back to zero tests on a git error.
        print(f"select_step9c_tests: git diff failed: {exc}", file=sys.stderr)
        return 1

    tests, untested = select_tests(touched, repo_root)
    missing = missing_invariants(repo_root)

    # Fail loud on an EMPTY selection (defense-in-depth beside the Step 9c shell
    # guard against a silent test-gate pass). WORKFLOW_INVARIANT has 32 always-on
    # entries, so an empty list can only mean the repo_root resolved wrong (the
    # #506 path-doubling bug) or the invariant files all vanished — either way the
    # gate would run zero tests. Never let that surface as an exit-0 "no tests ran".
    if not tests:
        print(
            "select_step9c_tests: EMPTY test selection — repo_root likely resolved "
            f"wrong ({repo_root}) or WORKFLOW_INVARIANT files are missing "
            f"(missing={missing}). Refusing to emit a zero-test gate command.",
            file=sys.stderr,
        )
        return 1

    for t in missing:
        print(f"WORKFLOW-INVARIANT MISSING: {t}", file=sys.stderr)

    if args.json:
        print(
            json.dumps(
                {
                    "tests": tests,
                    "untested_touched": untested,
                    "base": args.base,
                    "missing_invariants": missing,
                }
            )
        )
    else:
        print("uv run pytest " + " ".join(tests) + " -v --tb=short")
        for f in untested:
            print(f"untested touched file: {f}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
