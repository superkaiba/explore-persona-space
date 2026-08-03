"""Regression tests pinning the ``os._exit(rc)`` shutdown-bypass wrapper on
every scripts/issue1689_*.py entrypoint (round-6 crash fix).

Round-5 relaunch died Phase A on RunPod with ``Fatal Python error:
PyGILState_Release: thread state ... must be current when releasing`` at
interpreter finalize — the corpus JSONL was written cleanly (``[corpus]
done: scanned=5132 kept=3800``) then Python's shutdown race raised the
fatal, and ``dispatch.sh`` under ``set -euo pipefail`` aborted the sweep
after Phase A. This is the C-extension teardown race documented in
``.claude/rules/gotchas.md`` § "HF `datasets` / `transformers`
subprocesses can exit `rc=134` (SIGABRT) with a `PyGILState_Release`
fatal abort DURING interpreter shutdown". The fix: swap
``sys.exit(main())`` / ``raise SystemExit(main())`` for a wrapper that
flushes stdio then calls ``os._exit(rc)``, skipping atexit handlers +
Python finalize where the shutdown race lives.

These tests are STATIC — they parse each entrypoint's AST and assert the
wrapper structure. No subprocess launch, no full-vocab imports, no GPU.

The test grep also fails LOUD if a new ``scripts/issue1689_*.py``
entrypoint (any file with an ``if __name__ == "__main__":`` guard) lands
without the wrapper — closes the "next contributor forgets to add it"
gap.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Every scripts/issue1689_*.py entrypoint the dispatcher shells to. The
# ``common.py`` module carries no ``__main__`` block (library helpers only).
ENTRYPOINT_PATHS = sorted(
    p for p in SCRIPTS_DIR.glob("issue1689_*.py") if p.name != "issue1689_common.py"
)


def _has_main_guard(tree: ast.Module) -> bool:
    """True iff the module has an ``if __name__ == "__main__":`` block."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # ast shape for ``__name__ == "__main__"``: Compare(Name, [Eq()], [Constant])
        if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
            continue
        left, op, right = test.left, test.ops[0], test.comparators[0]
        if (
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and isinstance(op, ast.Eq)
            and isinstance(right, ast.Constant)
            and right.value == "__main__"
        ):
            return True
    return False


def _main_guard_body(tree: ast.Module) -> list[ast.stmt]:
    """Return the statement list under the module's ``__main__`` guard."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
            continue
        left, op, right = test.left, test.ops[0], test.comparators[0]
        if (
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and isinstance(op, ast.Eq)
            and isinstance(right, ast.Constant)
            and right.value == "__main__"
        ):
            return list(node.body)
    return []


def test_entrypoint_universe_nonempty() -> None:
    """A fresh clone must find every issue1689 entrypoint script."""
    assert ENTRYPOINT_PATHS, (
        "No scripts/issue1689_*.py entrypoints found — did the sparse checkout drop scripts/ ?"
    )
    # Round-6 fix scope: 8 entrypoints (analyze, capture, fit_cells,
    # fit_ladder, gen_corpus, gen_onpolicy, haiku_u2_gen, render_conditions).
    # A drop below 8 is a scope drift the test must FAIL LOUD on.
    assert len(ENTRYPOINT_PATHS) >= 8, (
        f"Fewer than 8 entrypoints; found {[p.name for p in ENTRYPOINT_PATHS]}. "
        "Round-6 fix covers 8; a smaller set means a rename/drop the test "
        "did not anticipate — verify the fix scope."
    )


@pytest.mark.parametrize("path", ENTRYPOINT_PATHS, ids=lambda p: p.name)
def test_entrypoint_has_main_guard(path: Path) -> None:
    """Every entrypoint has an ``if __name__ == "__main__":`` block."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert _has_main_guard(tree), (
        f"{path.name} has no ``if __name__ == '__main__':`` guard — either "
        "not an entrypoint (move to ENTRYPOINT_PATHS exclusion list) or the "
        "wrapper was removed."
    )


@pytest.mark.parametrize("path", ENTRYPOINT_PATHS, ids=lambda p: p.name)
def test_entrypoint_calls_os_exit(path: Path) -> None:
    """The ``__main__`` guard body ends with ``os._exit(...)`` — NOT
    ``sys.exit(...)`` / ``raise SystemExit(...)`` — so Python's C-extension
    finalize race cannot kill a completed phase."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = _main_guard_body(tree)
    assert body, f"{path.name} has an empty ``__main__`` guard body"

    # Find the (last) top-level Expr(Call) whose func is ``os._exit``.
    os_exit_calls: list[ast.Call] = []
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            func = call.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "_exit"
                and isinstance(func.value, ast.Name)
                and func.value.id == "os"
            ):
                os_exit_calls.append(call)
    assert os_exit_calls, (
        f"{path.name} __main__ guard does not call ``os._exit(...)``. "
        "The C-extension teardown race (gotchas.md § PyGILState_Release "
        "SIGABRT) will kill Phase A's successor sweep phases on RunPod "
        "if the process finalizes through Python's atexit chain."
    )

    # And there must be no ``sys.exit(...)`` / ``raise SystemExit(...)``
    # in the same guard body — the wrapper deliberately supersedes them.
    for stmt in body:
        # sys.exit(...)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "exit"
                and isinstance(func.value, ast.Name)
                and func.value.id == "sys"
            ):
                pytest.fail(
                    f"{path.name}: ``sys.exit(...)`` still present in "
                    "__main__ guard. Replace with the ``os._exit(rc)`` "
                    "wrapper (see scripts/issue1689_gen_corpus.py)."
                )
        # raise SystemExit(...)
        if isinstance(stmt, ast.Raise) and stmt.exc is not None:
            exc = stmt.exc
            if (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == "SystemExit"
            ):
                pytest.fail(
                    f"{path.name}: ``raise SystemExit(...)`` still present "
                    "in __main__ guard. Replace with the ``os._exit(rc)`` "
                    "wrapper (see scripts/issue1689_gen_corpus.py)."
                )
            if isinstance(exc, ast.Name) and exc.id == "SystemExit":
                pytest.fail(f"{path.name}: bare ``raise SystemExit`` still present.")


@pytest.mark.parametrize("path", ENTRYPOINT_PATHS, ids=lambda p: p.name)
def test_entrypoint_flushes_before_os_exit(path: Path) -> None:
    """The wrapper flushes both ``sys.stdout`` and ``sys.stderr`` BEFORE
    ``os._exit`` — a Phase A completion log line printed to stdout must
    reach the pod's log file before the bypass fires, else the run's
    diagnostic trail loses its last breadcrumb."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = _main_guard_body(tree)

    saw_stdout_flush = False
    saw_stderr_flush = False
    saw_os_exit = False
    for stmt in body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            continue
        call = stmt.value
        func = call.func
        # sys.stdout.flush() / sys.stderr.flush()
        if isinstance(func, ast.Attribute) and func.attr == "flush":
            inner = func.value
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "sys"
            ):
                if inner.attr == "stdout":
                    saw_stdout_flush = True
                elif inner.attr == "stderr":
                    saw_stderr_flush = True
        # os._exit(...)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "_exit"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        ):
            # Both flushes must precede this call.
            assert saw_stdout_flush, (
                f"{path.name}: ``sys.stdout.flush()`` missing before "
                "``os._exit(rc)`` — a final phase log line can be lost."
            )
            assert saw_stderr_flush, (
                f"{path.name}: ``sys.stderr.flush()`` missing before "
                "``os._exit(rc)`` — a fatal-error stderr line can be lost."
            )
            saw_os_exit = True
            break
    assert saw_os_exit, f"{path.name}: no ``os._exit`` call reached during scan"


@pytest.mark.parametrize("path", ENTRYPOINT_PATHS, ids=lambda p: p.name)
def test_entrypoint_os_exit_rc_type_coerced(path: Path) -> None:
    """``os._exit`` receives an int arg — a ``main()`` returning ``None``
    coerces to ``0``. This is the ``os._exit(rc if isinstance(rc, int) else 0)``
    idiom the wrapper standardizes so a rc-less ``main()`` (some phases
    return None on success) produces exit 0 instead of TypeErroring."""
    src = path.read_text(encoding="utf-8")
    # Grep-form check: the wrapper block's canonical shape.
    assert "os._exit(rc if isinstance(rc, int) else 0)" in src, (
        f"{path.name}: canonical ``os._exit(rc if isinstance(rc, int) "
        "else 0)`` form missing. A non-int ``main()`` return (e.g. None) "
        "would else TypeError inside _exit."
    )
