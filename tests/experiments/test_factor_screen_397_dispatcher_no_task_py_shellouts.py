"""Round 9 canary: dispatcher MUST NOT shell out to ``scripts/task.py``.

Round 9 was triggered by a production crash on pod-397: the dispatcher
shelled out to ``uv run python scripts/task.py find <N>`` to look up the
task directory + scan ``events.jsonl`` for the ``epm:smoke-pass`` row.
``task.py`` branch-guards to ``main`` (CLAUDE.md: "the canonical
resolver branch-guards to ``main`` and refuses loudly on detached HEAD
/ non-``main`` HEAD"); the pod-side checkout sits on ``issue-397`` →
``task.py`` exits non-zero → ``subprocess.run(check=True)`` raises →
dispatcher crashes before the 108-cell sweep launches.

Round 9 removed all ``task.py`` call sites from the dispatcher:

  - ``has_recent_smoke_pass_marker`` (line 837 pre-round-9) →
    ``is_smoke_pass_confirmed_locally`` (CLI flag + local
    ``metrics_final.json`` fallback).
  - ``post_marker_via_task_py`` smoke-end call →
    ``write_verdict_file(slab_root, "SMOKE_VERDICT.json", payload)``.
  - ``post_marker_via_task_py`` sweep-resume call →
    ``write_verdict_file(slab_root, "SWEEP_RESUME.json", payload)``.
  - ``post_marker_via_task_py`` helper itself → deleted.

The orchestrator on the VM side (where ``task.py`` works because it
runs from the ``main`` repo root) reads the verdict JSONs via SCP /
``ssh_download`` and posts the markers itself.

This test is the regression guard: any future re-introduction of
``task.py`` invocation from the dispatcher fails CI loud.

CPU-only; pure static-file analysis.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_DISPATCH_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
)


def test_dispatcher_no_task_py_in_subprocess_calls() -> None:
    """No ``subprocess.run`` / ``subprocess.Popen`` / ``os.system`` /
    ``os.popen`` call in the dispatcher passes ``scripts/task.py`` as
    an argument.

    Walks the AST so docstring references to ``task.py`` (legitimate —
    explaining WHY round 9 removed the shellout) don't false-positive.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))

    offenders: list[tuple[int, str]] = []

    class _ShelloutVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func_name = _resolve_call_name(node.func)
            if func_name in {
                "subprocess.run",
                "subprocess.Popen",
                "subprocess.check_call",
                "subprocess.check_output",
                "subprocess.call",
                "os.system",
                "os.popen",
                "os.execvpe",
                "os.execve",
                "os.execvp",
            } and _references_task_py(node):
                offenders.append((node.lineno, ast.unparse(node)))
            self.generic_visit(node)

    _ShelloutVisitor().visit(tree)
    assert offenders == [], (
        f"Round 9 contract: dispatcher MUST NOT shell out to scripts/task.py "
        f"(pod runs on issue-397 branch; task.py branch-guards to main and "
        f"crashes the dispatcher). Found {len(offenders)} offender(s):\n"
        + "\n".join(f"  line {ln}: {expr[:120]}" for ln, expr in offenders)
    )


def test_dispatcher_does_not_define_post_marker_via_task_py() -> None:
    """The ``post_marker_via_task_py`` helper itself was deleted in
    Round 9. A future re-introduction (re-adding the helper, even if
    no caller exists yet) fails this canary.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))
    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "post_marker_via_task_py" not in func_names, (
        "Round 9 deleted post_marker_via_task_py — re-adding it (even "
        "unused) signals a regression. Use write_verdict_file instead "
        "and let the orchestrator post markers from the VM side."
    )


def test_dispatcher_does_not_define_has_recent_smoke_pass_marker() -> None:
    """The ``has_recent_smoke_pass_marker`` function (which shelled out
    to ``task.py find``) was deleted in Round 9. Replaced by
    ``is_smoke_pass_confirmed_locally``.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))
    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "has_recent_smoke_pass_marker" not in func_names, (
        "Round 9 deleted has_recent_smoke_pass_marker — use "
        "is_smoke_pass_confirmed_locally instead."
    )
    assert "is_smoke_pass_confirmed_locally" in func_names, (
        "Round 9 replacement is_smoke_pass_confirmed_locally must exist"
    )


def test_dispatcher_exposes_write_verdict_file() -> None:
    """The Round 9 marker-replacement helper ``write_verdict_file``
    must be a public top-level function.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))
    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "write_verdict_file" in func_names, (
        "Round 9 added write_verdict_file — must be present for the "
        "orchestrator to read SMOKE_VERDICT.json / SWEEP_RESUME.json "
        "and post markers from the VM side."
    )


def test_dispatcher_cli_exposes_smoke_pass_confirmed_flag() -> None:
    """The Round 9 CLI flag ``--smoke-pass-confirmed`` must be present
    in the argparse parser. Set by the orchestrator AFTER posting
    ``epm:smoke-pass v1`` from the VM side.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    # Simple substring check inside build_arg_parser — AST walk would
    # be overkill for a CLI flag presence assertion.
    assert "--smoke-pass-confirmed" in src, (
        "Round 9 CLI flag --smoke-pass-confirmed missing from dispatcher"
    )
    # Sanity: the flag wires into args.smoke_pass_confirmed (argparse
    # converts dashes to underscores).
    assert "smoke_pass_confirmed" in src, (
        "Round 9: dispatcher must reference args.smoke_pass_confirmed"
    )


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _resolve_call_name(node: ast.AST) -> str:
    """Resolve a Call's function to a dotted name (e.g. ``subprocess.run``).

    Handles ``Name`` (bare call), ``Attribute`` (one-level dot),
    ``Attribute`` of ``Name`` (two-level). Returns empty string for
    anything more complex (lambda call, subscripted attr, etc.) —
    those won't false-positive on the shellout patterns we care about.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        return f"{_resolve_call_name(node.value)}.{node.attr}"
    return ""


def _references_task_py(node: ast.Call) -> bool:
    """Return True if any string literal inside ``node`` ends in
    ``task.py`` (the dispatcher's broken shellout target).

    Matches both ``"scripts/task.py"`` and bare ``"task.py"``.
    """
    pattern = re.compile(r"(^|/)task\.py$")
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and pattern.search(child.value)
        ):
            return True
    return False
