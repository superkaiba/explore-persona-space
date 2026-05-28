"""Regression test: pod-side code never shells out to ``scripts/task.py``.

Enforces the CLAUDE.md "Pod-side code NEVER shells out to scripts/task.py"
rule. Task #397 round 9 (2026-05-27) burned a launch on
``dispatch_factor_screen_397.py::has_recent_smoke_pass_marker`` calling
``subprocess.run(["uv", "run", "python", "scripts/task.py", "find", "397"])``
from a pod-side cwd. ``task.py`` branch-guards to ``main``; pods always
run on ``issue-<N>`` branches; the shellout died within ~5s of nohup with
``subprocess.CalledProcessError``. Forbidding ALL subcommands (find,
post-marker, latest-marker, view, set-status, new, etc.) eliminates the
recurrence surface — pods must use the sentinel-file pattern instead
(write JSON to ``/workspace/logs/issue-<N>-results.json``; orchestrator's
``poll_pipeline.py`` reads + posts markers from the local VM).

Why this is load-bearing
------------------------

A regex window scan was rejected because it would have missed multi-line
``subprocess.Popen(\n  cmd_list,\n  ...)`` calls. This test walks the
AST: any ``ast.Call`` whose ``.func`` resolves to a known subprocess /
os.system / os.popen / ssh_execute spawner AND whose first positional
argument (the cmd argv) contains a string literal matching
``(^|/)task\\.py$`` OR ``(^|/)scripts/task\\.py$`` is flagged.

False-positive guards
---------------------

- ``scripts/sagan_import.py:270`` embeds ``[task.py / sagan-import]``
  inside a git-commit-message argument. The regex requires ``task.py``
  to be path-terminal (``$``) or followed by whitespace, so the bracket
  + space-slash-space form does NOT match. Verified explicit.
- Docstrings / comments mentioning ``task.py`` are never argv elements
  in a subprocess call, so the AST walk never visits them.

Allowlist
---------

Files under ``LOCAL_VM_ONLY_PATHS`` are local-VM-only orchestrator
helpers that legitimately consume ``task.py``. Any new entry MUST be
local-VM-only — never reachable from a pod-side process. Per-line
``# epm-lint: pod-shellout-ok -- <reason>`` is supported when the reason
explicitly names the local-VM-only context (no bare noqa allowed).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Local-VM-only consumers of task.py — these scripts run on the
# orchestrator's VM, NOT on pods. Adding a file here is an assertion
# that the file is never invoked from a pod-side process.
_LOCAL_VM_ONLY_PATHS: frozenset[str] = frozenset(
    {
        # task.py is the script itself
        "scripts/task.py",
        # Orchestrator-side marker posters / pollers / state readers
        "scripts/post_step_completed.py",
        "scripts/poll_pipeline.py",
        "scripts/spawn_session.py",
        "scripts/pod.py",
        "scripts/pod_lifecycle.py",
        "scripts/pod_watch.py",
        "scripts/codex_task.py",
        "scripts/gh_project.py",
        "scripts/workflow_lint.py",
        "scripts/audit_clean_results_body_discipline.py",
        "scripts/verify_task_body.py",
        "scripts/verify_uploads.py",
        "scripts/failure_classifier.py",
        "scripts/hf_gate_accept.py",
        "scripts/migrate_354_366_to_sagan.py",
        "scripts/sagan_import.py",
        "scripts/task_state.py",
        # The test itself contains pattern strings
        "tests/test_no_pod_side_task_py_shellout.py",
        # Workflow library — orchestrator-side, never imported from pod
        "src/explore_persona_space/task_workflow.py",
        "src/explore_persona_space/task_workflow_migrate.py",
    }
)

# Directories to exclude from the scan entirely.
_DIR_EXCLUDES: tuple[str, ...] = (
    "external/",
    "archive/",
    "eval_results/",
    ".claude/worktrees/",
    ".venv/",
    ".git/",
    "node_modules/",
    "ood_eval_results/",
    "tests/",  # tests are local-VM-only by definition
)

# Top-level subtrees to scan. Scope tight — only places pod-bound code
# can live: scripts/dispatch_*.py, scripts/run_*.py, and
# src/.../experiments/*/{run_*.py, dispatch_*.py, __main__.py}.
_SCAN_ROOTS: tuple[str, ...] = ("scripts", "src")

# Path-terminal regex for shell=True string-literal cmds.
# Matches: "task.py" or "scripts/task.py" at end-of-string or followed
# by whitespace. Does NOT match "[task.py / sagan-import]" (the
# sagan_import.py:270 false positive) because that has " / " after.
_SHELL_CMD_PATH_REGEX = re.compile(r"(^|/)(scripts/)?task\.py(\s|$)")

# Subprocess/spawner function names to inspect.
_SPAWNER_FUNC_NAMES: frozenset[str] = frozenset(
    {
        "run",
        "Popen",
        "check_output",
        "check_call",
        "call",
        "system",
        "popen",
        "ssh_execute",
        "ssh_group_execute",
    }
)


def _walk_py_files() -> list[Path]:
    """Yield every ``.py`` file under SCAN_ROOTS honoring DIR_EXCLUDES."""
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


def _is_taskpy_argv_constant(node: ast.AST) -> bool:
    """True iff `node` is an ``ast.Constant`` string ending with ``task.py``
    or ``scripts/task.py`` as a path-terminal element. This catches the
    canonical list-form ``["uv", "run", "python", "scripts/task.py", ...]``
    without matching the sagan_import commit-message form.
    """
    if not isinstance(node, ast.Constant):
        return False
    if not isinstance(node.value, str):
        return False
    s = node.value
    # Path-terminal: the string IS task.py or scripts/task.py (no
    # trailing chars) OR ends with /task.py.
    return s == "task.py" or s == "scripts/task.py" or s.endswith("/task.py")


def _shell_cmd_contains_taskpy(node: ast.AST) -> bool:
    """True iff `node` is an ``ast.Constant`` string containing
    a path-terminal ``task.py`` reference (used for ``shell=True``
    string-arg subprocess calls).
    """
    if not isinstance(node, ast.Constant):
        return False
    if not isinstance(node.value, str):
        return False
    return bool(_SHELL_CMD_PATH_REGEX.search(node.value))


def _resolve_call_func_name(call: ast.Call) -> str | None:
    """Return the leaf attribute name of the call target, or None.

    For ``subprocess.run(...)`` -> ``"run"``.
    For ``subprocess.Popen(...)`` -> ``"Popen"``.
    For ``mcp__ssh__ssh_execute(...)`` -> ``"ssh_execute"`` (extracted from
    the attribute chain).
    For ``foo()`` (bare name) -> ``"foo"``.
    """
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _candidate_args(call: ast.Call) -> list[ast.AST]:
    """Return positional + ``command=``/``cmd=``/``args=`` keyword args."""
    out: list[ast.AST] = list(call.args)
    for kw in call.keywords:
        if kw.arg in ("command", "cmd", "args"):
            out.append(kw.value)
    return out


def _arg_references_taskpy(arg: ast.AST) -> bool:
    """True iff `arg` (a subprocess cmd-argv expression) references
    ``task.py`` in a shape we'd flag.
    """
    if isinstance(arg, (ast.List, ast.Tuple)):
        return any(_is_taskpy_argv_constant(elt) for elt in arg.elts)
    if isinstance(arg, ast.Constant):
        return _shell_cmd_contains_taskpy(arg)
    if isinstance(arg, ast.JoinedStr):
        for v in arg.values:
            if (
                isinstance(v, ast.Constant)
                and isinstance(v.value, str)
                and _SHELL_CMD_PATH_REGEX.search(v.value)
            ):
                return True
    return False


def _scan_one_file(path: Path) -> list[tuple[int, str]]:
    """AST-scan `path`. Return list of (lineno, snippet) offences.

    Respects per-line pod-shellout-ok escape hatch (reason required).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return [(0, "<unparseable: ast.SyntaxError>")]
    lines = text.splitlines()
    offences: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = _resolve_call_func_name(node)
        if func_name is None or func_name not in _SPAWNER_FUNC_NAMES:
            continue
        if not any(_arg_references_taskpy(a) for a in _candidate_args(node)):
            continue
        # Per-line escape-hatch — check the call's line range.
        lineno = node.lineno
        end = node.end_lineno or node.lineno
        block = "\n".join(lines[lineno - 1 : end])
        # Reason is required; bare opt-out is rejected.
        escape_pat = r"#\s*epm-lint:\s*pod-shellout-ok\s*--\s*\S+"
        if re.search(escape_pat, block):
            continue
        snippet = lines[lineno - 1].strip()
        offences.append((lineno, snippet))
    return offences


def test_no_pod_side_task_py_shellout() -> None:
    """Fail if any non-allowlisted file under scripts/ or src/ shells out
    to ``scripts/task.py`` for any subcommand.

    Run locally with:
        uv run pytest tests/test_no_pod_side_task_py_shellout.py -v
    """
    all_offences: list[tuple[str, int, str]] = []
    for path in _walk_py_files():
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        if rel in _LOCAL_VM_ONLY_PATHS:
            continue
        for lineno, snippet in _scan_one_file(path):
            all_offences.append((rel, lineno, snippet))
    if all_offences:
        lines = "\n".join(f"  - {p}:{ln}: {snip}" for p, ln, snip in all_offences)
        raise AssertionError(
            f"\n{len(all_offences)} file(s) shell out to scripts/task.py "
            f"from pod-reachable code.\n"
            f"\nPod-side code (anything reachable from `nohup` on "
            f"`epm-issue-<N>` or from a pod-side subprocess) MUST NOT "
            f"call `task.py` for ANY subcommand. `task.py` branch-guards "
            f"to `main`; pods always run on `issue-<N>` branches.\n"
            f"\nOffences:\n{lines}\n"
            f"\nRemediation: write a JSON sentinel file at "
            f"/workspace/logs/issue-<N>-*.json from the pod; the "
            f"orchestrator's poll_pipeline.py reads it and posts the "
            f"marker from the local VM. See CLAUDE.md 'Pod-side code "
            f"NEVER shells out to scripts/task.py' for the canonical "
            f"alternatives.\n"
            f"\nIf the offending file is a local-VM-only orchestrator "
            f"helper (never invoked from a pod), add its path to "
            f"_LOCAL_VM_ONLY_PATHS in this test. If a single call is "
            f"legitimate (rare), add "
            f"`# epm-lint: pod-shellout-ok -- <reason>` "
            f"on the call line (reason MUST name the local-VM-only "
            f"context).\n"
        )


@pytest.mark.parametrize(
    "src,should_match",
    [
        # Canonical violation: list-form shellout (round 9's bug class).
        (
            'subprocess.run(["uv", "run", "python", "scripts/task.py", "find", "397"])',
            True,
        ),
        # Same but Popen (split for line length).
        (
            'subprocess.Popen(["uv", "run", "python", "scripts/task.py",'
            ' "post-marker", "397", "epm:results", "--note", "..."])',
            True,
        ),
        # shell=True string-arg shellout.
        (
            'subprocess.run("uv run python scripts/task.py find 1", shell=True)',
            True,
        ),
        # ssh_execute calling task.py on the pod (treated as pod-reachable).
        (
            'ssh_execute(server="epm-issue-1", command="uv run python scripts/task.py find 1")',
            True,
        ),
        # False positive: sagan_import.py:270's bracketed citation in a
        # commit-message body. Does NOT match path-terminal regex.
        (
            'subprocess.run(["git", "commit", "-m", message + "\\n\\n[task.py / sagan-import]"])',
            False,
        ),
        # False positive: docstring mention of task.py (not an argv element).
        (
            '"""Posts via task.py post-marker.""" # docstring only',
            False,
        ),
        # False positive: a Python path component containing "task.py"
        # as a substring but not path-terminal (e.g., a hypothetical
        # "task.pyc" extension).
        (
            'subprocess.run(["python", "task.pyc"])',
            False,
        ),
    ],
)
def test_taskpy_pattern_matchers(src: str, should_match: bool) -> None:
    """Unit-tests for the AST helpers — confirms the canonical violation
    shapes match AND the documented false positives do NOT.
    """
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = _resolve_call_func_name(node)
        if func_name is None or func_name not in _SPAWNER_FUNC_NAMES:
            continue
        if any(_arg_references_taskpy(a) for a in _candidate_args(node)):
            found = True
            break
    assert found == should_match, f"src={src!r}: expected match={should_match}, got match={found}"
