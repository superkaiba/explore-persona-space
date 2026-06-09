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

Path-composition evasion shapes (extended 2026-06-09 after task #521)
---------------------------------------------------------------------

Task #521 round-1 (commit ``c762d21d6``) evaded the scanner with a
two-pronged construction in ``scripts/issue_521_em_rate_gate.py::
_post_em_rate_marker``::

    cmd = [
        "uv", "run", "python",
        str(repo_root / "scripts" / "task.py"),  # not a Constant
        "post-marker", str(issue), marker_kind, "--note", note,
    ]
    rc = subprocess.run(cmd, check=False).returncode  # argv is a Name

Two evasion vectors compounded:
  1. The argv was a local Name binding, not a direct list-literal —
     ``_arg_references_taskpy`` only inspected ``ast.List``/``ast.Tuple``
     literals passed directly to the subprocess call.
  2. The ``task.py`` reference inside the list was ``str(Path / "..."
     / "task.py")``, not a string Constant — so even when the binding
     was followed, the element would not have matched the
     Constant-only ``_is_taskpy_argv_constant``.

Both ensemble code-reviewers caught it; the test did not. The scanner
now resolves intra-function Name bindings to their list-literal source
and treats ``Constant``, ``str(<expr>)``, ``Path / "..." / "..."`` chains,
and ``os.path.join(...)`` as path-composition forms whose string
constants are checked for a path-terminal ``task.py`` reference.

Scope of the binding resolution: intra-function only — the most recent
``ast.Assign`` of an ``ast.List``/``ast.Tuple`` literal to a Name within
the same function body. Cross-function plumbing (``cmd`` as a function
parameter, ``cmd`` returned from a helper, ``cmd`` built via
``itertools.chain``) is NOT chased; those shapes will surface in code
review and can be added if they recur in production.

False-positive guards
---------------------

- ``scripts/sagan_import.py:270`` embeds ``[task.py / sagan-import]``
  inside a git-commit-message argument. The regex requires ``task.py``
  to be path-terminal (``$``) or followed by whitespace, so the bracket
  + space-slash-space form does NOT match. Verified explicit.
- Docstrings / comments mentioning ``task.py`` are never argv elements
  in a subprocess call, so the AST walk never visits them.
- A composed path ending in ``other.py`` / ``train.py`` / ``task.pyc``
  is NOT flagged: the path-terminal check requires the final component
  to be literally ``task.py``.

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
        # Crash-recovery + pod-safety watcher: a VM-crontab orchestrator helper
        # (runs from PROJECT_ROOT on main, never from a pod) that reads task
        # status / events and posts markers via task.py. Same local-VM-only
        # class as spawn_session / poll_pipeline / pod_watch above.
        "scripts/autonomous_session_watch.py",
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
        # Dual-context gate script (#521): pod-side callers MUST pass
        # --no-post-marker (enforced by scripts/run_issue521_v2_sweep.sh, which
        # delivers the em-rate result via a /workspace/logs sentinel instead);
        # the task.py-shellout branch in _post_em_rate_marker is exercised
        # ONLY when the gate runs VM-side.
        "scripts/issue_521_em_rate_gate.py",
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


_OPAQUE = "<OPAQUE>"


def _collect_path_components(node: ast.AST) -> list[str] | None:
    """Recursively collect the ordered string components of a path-like
    expression. Opaque (non-constant) operands are preserved as the
    sentinel ``_OPAQUE`` so the terminal check can still inspect the
    final component — e.g. ``repo_root / "scripts" / "task.py"`` returns
    ``[_OPAQUE, "scripts", "task.py"]``, which ``_components_end_with_taskpy``
    flags as a task.py-terminal path.

    Returns ``None`` only when the expression is shape-wise NOT a path
    composition at all (e.g. a bare ``ast.Name`` outside of a BinOp chain,
    a numeric literal, or any other non-recognized shape). Returning a
    list (even one full of ``_OPAQUE``) indicates "this IS a path-shaped
    expression; here are its components".

    Recognized shapes:
      - ``ast.Constant("foo")`` → ``["foo"]``
      - ``ast.Call(str, [<expr>])`` → recurse into ``<expr>`` (Path-to-str)
      - ``ast.BinOp(left, Div, right)`` → recurse into both sides
        (covers ``Path("/a") / "b" / "c"`` chains; opaque operands kept)
      - ``ast.Call(<.join>, ["a", "b"])`` (``os.path.join`` and friends)
      - ``ast.Call(Path|PurePath|PosixPath|PureWindowsPath, [<expr>])``
    """
    # 1. Direct string constant.
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return [node.value]
        return None

    # 2. str(<expr>) — unwrap the conversion.
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) == 1
    ):
        sub = _collect_path_components(node.args[0])
        # str(<opaque>) on its own is NOT a path-shaped expression —
        # only forward if the inner expression already looked path-shaped.
        return sub

    # 3. Path-like constructors: Path(...), PurePath(...), PosixPath(...).
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
        and _resolve_call_func_name(node)
        in {"Path", "PurePath", "PurePosixPath", "PureWindowsPath", "PosixPath", "WindowsPath"}
    ):
        # Path("a", "b") concatenates positional args by /; recurse each.
        out: list[str] = []
        for arg in node.args:
            sub = _collect_path_components(arg)
            if sub is None:
                # An opaque arg in a Path() call — record as wildcard.
                out.append(_OPAQUE)
            else:
                out.extend(sub)
        return out

    # 4. BinOp(Div) — pathlib `/` operator. Recurse both sides; opaque
    #    operands are preserved as wildcard components so the terminal
    #    check still works on the right-hand string.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _collect_path_components(node.left)
        right = _collect_path_components(node.right)
        if left is None:
            left = [_OPAQUE]
        if right is None:
            right = [_OPAQUE]
        return left + right

    # 5. os.path.join(...) / "<sep>".join([...]) — collect string args.
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
    ):
        # os.path.join("a", "b") OR pathlib equivalents — collect positional args.
        # We don't enforce the separator here; any join is treated
        # as a path composition for our purposes.
        out2: list[str] = []
        for arg in node.args:
            sub = _collect_path_components(arg)
            if sub is None:
                out2.append(_OPAQUE)
            else:
                out2.extend(sub)
        return out2

    # Opaque — not a path-shaped expression.
    return None


def _components_end_with_taskpy(components: list[str]) -> bool:
    """True iff the ordered path-component list resolves to a path whose
    final element is literally ``task.py``.

    Opaque components (``_OPAQUE``) at the START of the list act as
    wildcards and are ignored for the terminal check; ``[_OPAQUE,
    "scripts", "task.py"]`` matches because the rightmost concrete
    component is ``task.py``. Opaque components at the END of the list
    do NOT match (we cannot prove the path terminates at ``task.py``).
    """
    if not components:
        return False
    # Trim trailing opaque components — we cannot tell what the
    # terminal segment is, so we must NOT match.
    trimmed = list(components)
    while trimmed and trimmed[-1] == _OPAQUE:
        return False
    # Strip leading opaque components — they are wildcards.
    while trimmed and trimmed[0] == _OPAQUE:
        trimmed.pop(0)
    if not trimmed:
        return False
    joined = "/".join(trimmed)
    # Path-terminal check: the joined string ends with "/task.py" or
    # is exactly "task.py". Mirrors the _is_taskpy_argv_constant rules.
    return joined == "task.py" or joined.endswith("/task.py")


def _is_taskpy_argv_element(node: ast.AST) -> bool:
    """True iff `node` (a single element of a subprocess cmd argv list)
    resolves to a path whose terminal component is ``task.py``.

    Covers ``ast.Constant`` (delegating to ``_is_taskpy_argv_constant``)
    AND path-composition shapes via ``_collect_path_components``.
    """
    if _is_taskpy_argv_constant(node):
        return True
    components = _collect_path_components(node)
    if components is None:
        return False
    return _components_end_with_taskpy(components)


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


def _arg_references_taskpy(
    arg: ast.AST,
    name_bindings: dict[str, ast.AST] | None = None,
) -> bool:
    """True iff `arg` (a subprocess cmd-argv expression) references
    ``task.py`` in a shape we'd flag.

    If `arg` is an ``ast.Name`` and ``name_bindings`` maps that name to a
    list/tuple literal (from the most recent intra-function assignment),
    we inspect the bound literal's elements. This catches the round-1
    #521 evasion shape where ``cmd = [..., str(repo / "scripts" /
    "task.py"), ...]`` is bound then passed as ``subprocess.run(cmd)``.
    """
    if isinstance(arg, (ast.List, ast.Tuple)):
        return any(_is_taskpy_argv_element(elt) for elt in arg.elts)
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
    if isinstance(arg, ast.Name) and name_bindings is not None:
        bound = name_bindings.get(arg.id)
        if bound is not None and isinstance(bound, (ast.List, ast.Tuple)):
            return any(_is_taskpy_argv_element(elt) for elt in bound.elts)
    return False


def _collect_name_bindings(scope: ast.AST) -> dict[str, ast.AST]:
    """Walk a function-body (or module) scope and collect the most
    recent ``ast.Assign`` of an ``ast.List``/``ast.Tuple`` literal to
    a single ``ast.Name`` target. Returns ``{name_id: literal_node}``.

    Intra-scope only — we walk only the DIRECT statement list of the
    scope body (not nested function bodies, not nested class bodies),
    so an assignment in one function does not leak as a "module-level"
    binding into a sibling function. Reassignment within the same
    scope: last assignment wins.

    For ``Module``, only top-level assignments are collected. For
    ``FunctionDef``/``AsyncFunctionDef``, only statements directly
    inside ``.body`` (transitively walking into ``if``/``for``/``while``/
    ``try`` blocks but NOT into nested ``def`` / ``class``) are
    considered intra-scope.
    """
    bindings: dict[str, ast.AST] = {}

    # Determine the entry body list for this scope.
    if isinstance(scope, ast.Module):
        body: list[ast.stmt] = list(scope.body)
    elif isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        body = list(scope.body)
    else:
        body = []

    # Iterate the scope body, recursing through compound statements
    # (if/for/while/try) but STOPPING at nested function/class
    # definitions so their internals don't leak as same-scope bindings.
    stack: list[ast.stmt] = list(body)
    while stack:
        stmt = stack.pop(0)
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.value, (ast.List, ast.Tuple)):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        bindings[tgt.id] = stmt.value
            continue
        # Recurse into compound statements that share the scope.
        if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            stack[:0] = list(stmt.body) + list(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            stack[:0] = (
                list(stmt.body)
                + [s for h in stmt.handlers for s in h.body]
                + list(stmt.orelse)
                + list(stmt.finalbody)
            )
        elif isinstance(stmt, ast.With):
            stack[:0] = list(stmt.body)
        # ast.FunctionDef / ast.AsyncFunctionDef / ast.ClassDef: skip
        # — their bodies live in a different scope.
    return bindings


def _iter_scopes(tree: ast.AST):
    """Yield each function/method body (and the module itself) as a
    scope. Name bindings are tracked per-scope so a local rebinding in
    one function does not leak into another.
    """
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _scan_tree(tree: ast.AST, lines: list[str] | None = None) -> list[tuple[int, str]]:
    """Per-AST-tree scan. Returns ``(lineno, snippet)`` for each offending
    subprocess call. ``lines`` is the source split by newline (for snippet
    extraction + escape-hatch detection); pass ``None`` to skip both.

    Walks each function scope (plus the module scope) independently,
    collects intra-scope Name bindings, then inspects every subprocess /
    ssh_execute call inside the scope. A call is reported by its leaf
    scope only — module-scope name bindings ARE visible to functions
    (an `ast.walk` from a function node never re-enters siblings, so
    module-level cmd literals would not be seen otherwise).
    """
    offences: list[tuple[int, str]] = []
    seen: set[int] = set()  # dedupe across scope sweeps
    module_bindings = _collect_name_bindings(tree)

    for scope in _iter_scopes(tree):
        is_module = scope is tree
        # Inside a function, layer module-level bindings under the
        # function's local bindings so a local `cmd = [...]` shadows
        # a module-level one.
        bindings = dict(module_bindings) if not is_module else {}
        bindings.update(_collect_name_bindings(scope))
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call):
                continue
            # Note: ``ast.walk(scope)`` from the module visits every
            # call (including ones inside functions); we rely on the
            # per-lineno ``seen`` dedupe to suppress the duplicate when
            # the per-function scope visits the same call with its own
            # (richer) bindings.
            func_name = _resolve_call_func_name(node)
            if func_name is None or func_name not in _SPAWNER_FUNC_NAMES:
                continue
            if not any(_arg_references_taskpy(a, bindings) for a in _candidate_args(node)):
                continue
            lineno = node.lineno
            if lineno in seen:
                continue
            # Per-line escape-hatch — check the call's line range.
            if lines is not None:
                end = node.end_lineno or node.lineno
                block = "\n".join(lines[lineno - 1 : end])
                escape_pat = r"#\s*epm-lint:\s*pod-shellout-ok\s*--\s*\S+"
                if re.search(escape_pat, block):
                    continue
                snippet = lines[lineno - 1].strip()
            else:
                snippet = ""
            seen.add(lineno)
            offences.append((lineno, snippet))
    return offences


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
    return _scan_tree(tree, lines)


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
        # #521 round-1 evasion shape: argv built via local Name binding,
        # task.py reference composed via str(Path / "scripts" / "task.py").
        # Both prongs (Name argv + non-Constant path element) must be
        # caught for the regression test to bind.
        (
            "def f(repo_root):\n"
            "    cmd = [\n"
            '        "uv", "run", "python",\n'
            '        str(repo_root / "scripts" / "task.py"),\n'
            '        "post-marker", "521",\n'
            "    ]\n"
            "    subprocess.run(cmd, check=False)\n",
            True,
        ),
        # Same prongs, direct list-literal (no Name binding) but path
        # composed via Path / "scripts" / "task.py" without str().
        (
            "def f(repo_root):\n"
            "    subprocess.run([\n"
            '        "uv", "run", "python",\n'
            '        repo_root / "scripts" / "task.py",\n'
            "    ])\n",
            True,
        ),
        # os.path.join shape — same evasion via stdlib join.
        (
            "def f():\n"
            '    subprocess.run(["python", os.path.join("scripts", "task.py"), "find", "1"])\n',
            True,
        ),
        # Bound list-literal with composed path; argv passed via the
        # `args=` keyword instead of positional.
        (
            "def f(repo):\n"
            '    cmd = ["python", str(repo / "scripts" / "task.py"), "view"]\n'
            "    subprocess.Popen(args=cmd)\n",
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
        # False positive: composed path ending in a DIFFERENT script.
        # The path-composition logic must not flag any .py file — only
        # the literal "task.py" terminal component.
        (
            "def f(repo):\n"
            '    subprocess.run(["python", str(repo / "scripts" / "train.py"), "--cfg", "x"])\n',
            False,
        ),
        # False positive: bound list-literal that does NOT mention task.py.
        # Variable-binding resolution must not match arbitrary subprocess
        # calls just because the argv is a Name.
        (
            "def f():\n"
            '    cmd = ["python", "scripts/train.py", "--seed", "42"]\n'
            "    subprocess.run(cmd, check=False)\n",
            False,
        ),
        # False positive: cmd is a function parameter (no local binding
        # to inspect). We deliberately do NOT chase cross-function flow;
        # such cases must be caught in code review. The scanner must NOT
        # raise or false-positive on the unknown Name.
        (
            "def f(cmd):\n    subprocess.run(cmd, check=False)\n",
            False,
        ),
        # False positive: os.path.join with NO task.py — must not match.
        (
            'def f():\n    subprocess.run(["python", os.path.join("scripts", "other.py")])\n',
            False,
        ),
    ],
)
def test_taskpy_pattern_matchers(src: str, should_match: bool) -> None:
    """Unit-tests for the AST helpers — confirms the canonical violation
    shapes match AND the documented false positives do NOT.

    Uses ``_scan_tree`` so the unit cases exercise the same name-binding
    resolution + path-composition logic as the file scanner. Each
    `src` is parsed as a module-level snippet (functions inline).
    """
    tree = ast.parse(src)
    offences = _scan_tree(tree, lines=src.splitlines())
    found = len(offences) > 0
    assert found == should_match, (
        f"src={src!r}: expected match={should_match}, got match={found}; offences={offences!r}"
    )


def test_issue_521_round1_regression() -> None:
    """Replicate the verbatim #521 round-1 _post_em_rate_marker
    construction from commit c762d21d6. This is the historical pattern
    the test missed; lock it in as a named regression.
    """
    src = (
        "import subprocess\n"
        "from pathlib import Path\n"
        "def _post_em_rate_marker(*, repo_root, issue, note, marker_kind):\n"
        "    cmd = [\n"
        '        "uv",\n'
        '        "run",\n'
        '        "python",\n'
        '        str(repo_root / "scripts" / "task.py"),\n'
        '        "post-marker",\n'
        "        str(issue),\n"
        "        marker_kind,\n"
        '        "--note",\n'
        "        note,\n"
        "    ]\n"
        "    rc = subprocess.run(cmd, check=False).returncode\n"
        "    return rc\n"
    )
    tree = ast.parse(src)
    offences = _scan_tree(tree, lines=src.splitlines())
    assert offences, (
        "The #521 round-1 evasion shape (Name-bound cmd with "
        "str(Path / 'scripts' / 'task.py') element) must be flagged. "
        "If this regresses, two ensemble code-reviewers caught it but "
        "the test did not — see the module docstring."
    )
