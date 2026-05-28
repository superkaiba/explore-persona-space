"""Regression test: dispatchers spawn subprocesses with explicit env=
AND load credential env at module/main/__main__ entry.

Enforces the two-check rule documented in
``.claude/agents/experiment-implementer.md`` ("Subprocess env
passthrough — TWO checks").

Why this is load-bearing
------------------------

Task #397 round-10' (2026-05-27) burned a launch because the dispatcher
passed ``env=env`` to ``subprocess.Popen`` correctly, but the ``env``
dict came from ``os.environ.copy()`` of an unloaded parent process —
the dispatcher itself never called ``load_dotenv()``. ``uv run python``
does NOT auto-load ``.env``. Result: ``HF_TOKEN`` was missing from the
subprocess env; ``_upload`` returned an empty path; the cell exited
rc=2 across all 107 cells.

A single-check rule (env= kwarg present) is necessary but not sufficient
— the round-10' dispatcher would have PASSed it. The second check
verifies the parent process actually loaded credentials.

Two checks, both AST-based
--------------------------

**Check 1 (per-call):** Every ``subprocess.run|Popen|check_output|
check_call|call`` in scope MUST pass an ``env=`` keyword argument.
Per-line escape hatch:
``# epm-lint: subprocess-env-inherit -- <reason>`` (reason required).

**Check 2 (file-level):** Any in-scope file containing a
``subprocess.<func>`` call MUST also contain a credential-loading site
at module-top, ``main()``-top, or ``if __name__ == "__main__":``
block-top:
- ``load_dotenv()`` call, OR
- ``assert os.environ.get(<credential>)`` / ``os.getenv(<credential>)``
  for one of ``HF_TOKEN | WANDB_API_KEY | ANTHROPIC_API_KEY |
  OPENAI_API_KEY | RUNPOD_API_KEY``.

The file-level rule trades a small false-positive surface (files where
env IS loaded inside a transitive helper) for catching the canonical
refactored shape (``main()`` -> ``_dispatch_helper()`` -> subprocess)
correctly without needing call-graph analysis. File-level escape hatch:
``# epm-lint: subprocess-env-implicit-load -- <reason>`` at file top
(lines 1-5) OR at the ``def main`` line.

Scope
-----

- ``scripts/dispatch_*.py``
- ``scripts/run_sweep*.py``
- ``scripts/run_pipeline*.py``
- ``scripts/run_experiment_*.py``
- ``scripts/run_dose_response_*.py``
- ``scripts/run_factor_screen_*.py``
- ``src/**/experiments/*/run_*.py``
- ``src/**/experiments/*/dispatch_*.py``
- ``src/**/experiments/*/__main__.py``

Grandfather list
----------------

Six pre-existing offenders (verified in the v3 plan fact-check) live in
``GRANDFATHERED_OFFENDERS``, keyed by ``(filename, function_name)`` —
line numbers drift, function names are stable. Each entry carries a
one-line justification and a follow-up task ref. New offenders MUST be
fixed in-place; the grandfather list does not grow.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Globs of in-scope dispatcher files (rel paths from REPO_ROOT,
# POSIX-style).
_DISPATCHER_GLOBS: tuple[str, ...] = (
    "scripts/dispatch_*.py",
    "scripts/run_sweep*.py",
    "scripts/run_pipeline*.py",
    "scripts/run_experiment_*.py",
    "scripts/run_dose_response_*.py",
    "scripts/run_factor_screen_*.py",
    "src/explore_persona_space/experiments/*/run_*.py",
    "src/explore_persona_space/experiments/*/dispatch_*.py",
    "src/explore_persona_space/experiments/*/__main__.py",
)

# Subprocess spawner function names (Check 1).
_SPAWNER_FUNC_NAMES: frozenset[str] = frozenset(
    {"run", "Popen", "check_output", "check_call", "call"}
)

# Credential names that load-at-entry may assert (Check 2).
_CRED_NAMES: frozenset[str] = frozenset(
    {
        "HF_TOKEN",
        "WANDB_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "RUNPOD_API_KEY",
    }
)

# Grandfathered offenders. (rel_path, function_name) -> justification.
# Line numbers drift; function names are stable.
GRANDFATHERED_OFFENDERS: dict[tuple[str, str], str] = {
    (
        "scripts/run_pipeline.py",
        "main",
    ): "legacy orchestrator predating the rule; subprocess.run(cmd) in main(); follow-up #TBD",
    (
        "scripts/run_dose_response_orchestrator.py",
        "launch_cell",
    ): "predates rule; subprocess.Popen background launcher; follow-up #TBD",
    (
        "scripts/run_experiment_369.py",
        "_git_commit",
    ): "predates rule; git rev-parse HEAD style diagnostic; follow-up #TBD",
    (
        "scripts/run_experiment_389.py",
        "_git_commit_sha",
    ): "predates rule; git rev-parse HEAD style diagnostic; follow-up #TBD",
    (
        "scripts/run_experiment_389.py",
        "_capture_gpu_metadata",
    ): "predates rule; nvidia-smi diagnostic probe (two calls in same fn); follow-up #TBD",
}


def _iter_in_scope_files() -> list[Path]:
    """Resolve _DISPATCHER_GLOBS to absolute paths."""
    out: list[Path] = []
    for pat in _DISPATCHER_GLOBS:
        for p in REPO_ROOT.glob(pat):
            if p.is_file() and p.suffix == ".py":
                out.append(p)
    # Deduplicate (same file may match two globs)
    return sorted(set(out))


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _resolve_call_func_name(call: ast.Call) -> str | None:
    """Return the leaf attribute name of the call target, or None."""
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _enclosing_function_name(call_node: ast.Call, fn_index: dict[int, str]) -> str:
    """Look up which function this call lives in via the lineno index."""
    return fn_index.get(call_node.lineno, "<module>")


def _build_fn_lineno_index(tree: ast.Module) -> dict[int, str]:
    """Return {lineno: function_name} for every line covered by a
    function-def body. ``<module>`` is implicit for lines not covered.
    """
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno or start
            for ln in range(start, end + 1):
                out.setdefault(ln, node.name)
    return out


def _has_env_kwarg(call: ast.Call) -> bool:
    return any(kw.arg == "env" for kw in call.keywords)


def _per_line_epm_lint_envinherit(src_line: str) -> bool:
    """True iff line carries pod-shellout-equivalent inherit-OK opt-out."""
    pat = r"#\s*epm-lint:\s*subprocess-env-inherit\s*--\s*\S+"
    return bool(re.search(pat, src_line))


def _is_load_dotenv_call(node: ast.AST) -> bool:
    """True iff `node` is a Call to a function named load_dotenv."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "load_dotenv":
        return True
    return isinstance(func, ast.Attribute) and func.attr == "load_dotenv"


def _is_credential_assertion(node: ast.AST) -> bool:
    """True iff `node` is an ``ast.Assert`` whose test references
    ``os.environ.get(<cred>)`` or ``os.getenv(<cred>)``.
    """
    if not isinstance(node, ast.Assert):
        return False
    for sub in ast.walk(node.test):
        if isinstance(sub, ast.Call):
            func = sub.func
            attr = None
            if isinstance(func, ast.Attribute):
                attr = func.attr  # e.g. os.environ.get => "get"; os.getenv => "getenv"
            elif isinstance(func, ast.Name):
                attr = func.id
            if attr not in ("get", "getenv"):
                continue
            for arg in sub.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value in _CRED_NAMES
                ):
                    return True
    return False


def _node_loads_env(node: ast.AST) -> bool:
    """True iff `node` is a load_dotenv() Expr or a credential Assert."""
    if isinstance(node, ast.Expr) and _is_load_dotenv_call(node.value):
        return True
    return _is_credential_assertion(node)


def _file_has_entry_envload(tree: ast.Module) -> bool:
    """True iff the module has an env-load-or-assert at module-top,
    main()-top (first 5 statements of def main body), OR
    __main__ block-top (first 5 statements of if __name__ == "__main__":).
    """
    # Module-top scan: every statement before the first FunctionDef /
    # ClassDef / AsyncFunctionDef / if-__main__ block.
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            break
        if isinstance(stmt, ast.If) and _is_main_block(stmt):
            break
        if _node_loads_env(stmt):
            return True
    # main()-top scan.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
            for sub in node.body[:5]:
                if _node_loads_env(sub):
                    return True
    # __main__ block-top scan.
    for node in tree.body:
        if isinstance(node, ast.If) and _is_main_block(node):
            for sub in node.body[:5]:
                if _node_loads_env(sub):
                    return True
    return False


def _is_main_block(node: ast.If) -> bool:
    """True iff ``node`` is ``if __name__ == "__main__":``."""
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    right = test.comparators[0]
    return (
        isinstance(right, ast.Constant)
        and isinstance(right.value, str)
        and right.value == "__main__"
    )


def _file_carries_implicit_load_epm_lint(text: str) -> bool:
    """Honor file-level ``epm-lint: subprocess-env-implicit-load`` at the
    top of the file (line 1-5) OR on the ``def main`` line.
    """
    pat = r"#\s*epm-lint:\s*subprocess-env-implicit-load\s*--\s*\S+"
    lines = text.splitlines()
    head = "\n".join(lines[:5])
    if re.search(pat, head):
        return True
    return any("def main" in ln and re.search(pat, ln) for ln in lines)


def _check_one_file(
    path: Path,
) -> tuple[list[tuple[int, str, str]], bool]:
    """Run both checks against `path`.

    Returns:
        (check1_offences, check2_failed) where:
          - check1_offences: list of (lineno, func_name, snippet) for
            subprocess calls missing ``env=`` (skipping grandfathered
            (file, fn) pairs and per-line noqa).
          - check2_failed: True iff the file has at least one subprocess
            call AND no entry-position env-load/assert AND no file-level
            noqa.
    """
    rel = _rel(path)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return [], False
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return [(0, "<module>", "<unparseable>")], False
    lines = text.splitlines()
    fn_index = _build_fn_lineno_index(tree)

    check1: list[tuple[int, str, str]] = []
    has_subprocess = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = _resolve_call_func_name(node)
        if func_name is None or func_name not in _SPAWNER_FUNC_NAMES:
            continue
        # Restrict to actual subprocess.<func> / subprocess module calls
        # to avoid false positives on unrelated `.run()` or `.call()`.
        if not _is_subprocess_call(node):
            continue
        has_subprocess = True
        if _has_env_kwarg(node):
            continue
        fn_name = _enclosing_function_name(node, fn_index)
        if (rel, fn_name) in GRANDFATHERED_OFFENDERS:
            continue
        lineno = node.lineno
        end = node.end_lineno or node.lineno
        block = "\n".join(lines[lineno - 1 : end])
        if any(_per_line_epm_lint_envinherit(ln) for ln in block.splitlines()):
            continue
        snippet = lines[lineno - 1].strip()
        check1.append((lineno, fn_name, snippet))

    check2_failed = (
        has_subprocess
        and not _file_carries_implicit_load_epm_lint(text)
        and not _file_has_entry_envload(tree)
    )
    return check1, check2_failed


def _is_subprocess_call(call: ast.Call) -> bool:
    """True iff the call target resolves to a subprocess module call
    (``subprocess.run`` / ``subprocess.Popen`` / etc.), not just any
    ``.run()`` method invocation on an unrelated object.
    """
    f = call.func
    if isinstance(f, ast.Attribute):
        # subprocess.run / subprocess.Popen / sp.run / etc.
        return isinstance(f.value, ast.Name) and f.value.id in (
            "subprocess",
            "sp",
            "subp",
        )
    # Bare `run(...)` / `Popen(...)` — likely from-import; treat as
    # subprocess (caller already filtered by function name).
    return isinstance(f, ast.Name)


def test_subprocess_env_explicit_check1() -> None:
    """Check 1 — every subprocess call in scope passes ``env=`` kwarg.

    Grandfathered (file, function_name) pairs and per-line
    ``# epm-lint: subprocess-env-inherit -- <reason>`` opt-outs are skipped.
    """
    all_offences: list[tuple[str, int, str, str]] = []
    for path in _iter_in_scope_files():
        c1, _ = _check_one_file(path)
        rel = _rel(path)
        for lineno, fn_name, snippet in c1:
            all_offences.append((rel, lineno, fn_name, snippet))
    if all_offences:
        lines = "\n".join(f"  - {p}:{ln} (fn={fn}): {snip}" for p, ln, fn, snip in all_offences)
        raise AssertionError(
            f"\n{len(all_offences)} subprocess call(s) in dispatcher "
            f"files lack explicit env= kwarg.\n"
            f"\nDispatchers MUST pass env={{**os.environ}} (or a "
            f"deliberate filtered copy) to every subprocess.run|Popen|"
            f"check_output|check_call|call so the credential contract "
            f"is explicit. Implicit inheritance is fragile under "
            f"`uv run` and CI re-invocations.\n"
            f"\nOffences:\n{lines}\n"
            f"\nRemediation: add `env={{**os.environ}}` to the call, OR "
            f"add `# epm-lint: subprocess-env-inherit -- <reason>` to the "
            f"line (reason required; name the specific subprocess that "
            f"legitimately doesn't need credential env). To grandfather "
            f"a pre-existing offender, add its `(filename, "
            f"function_name)` tuple to `GRANDFATHERED_OFFENDERS` in "
            f"this test (one-line justification + follow-up task ref).\n"
        )


def test_subprocess_env_explicit_check2() -> None:
    """Check 2 — every file containing a subprocess call also loads
    credential env at module-top, main()-top, or __main__ block-top.

    File-level ``# epm-lint: subprocess-env-implicit-load -- <reason>`` at
    line 1-5 or on the ``def main`` line opts the file out.

    The rule does NOT walk the call graph; it accepts a small false-
    positive surface for files where env IS loaded inside a transitive
    helper. The escape hatch handles those legitimate cases.
    """
    failing: list[str] = []
    for path in _iter_in_scope_files():
        _, c2 = _check_one_file(path)
        if c2:
            failing.append(_rel(path))
    if failing:
        lines = "\n".join(f"  - {p}" for p in failing)
        raise AssertionError(
            f"\n{len(failing)} dispatcher file(s) spawn subprocesses but "
            f"do not load credential env at any entry position.\n"
            f"\nFiles containing `subprocess.<func>` calls MUST also "
            f"contain one of:\n"
            f"  (a) `load_dotenv()` at module-top (before first def),\n"
            f"  (b) `load_dotenv()` at the top of `def main()` body,\n"
            f"  (c) `load_dotenv()` at the top of "
            f'`if __name__ == "__main__":` block,\n'
            f"  (d) `assert os.environ.get(<HF_TOKEN|WANDB_API_KEY|...>)` "
            f"at any of the three positions above.\n"
            f"\nRationale: `uv run python` does NOT auto-load .env. "
            f"Without the load-at-entry, the parent process's os.environ "
            f"lacks credentials, and `env={{**os.environ}}` propagates "
            f"that emptiness to subprocesses. Task #397 round-10' "
            f"(2026-05-27) burned a launch this way.\n"
            f"\nOffenders:\n{lines}\n"
            f"\nRemediation: add `from dotenv import load_dotenv` + "
            f"`load_dotenv()` at module-top (most cases), OR add "
            f"`# epm-lint: subprocess-env-implicit-load -- <reason>` at the "
            f"top of the file (reason must name the wrapper that loads "
            f"env, e.g. parent module's `__main__.py`).\n"
        )
