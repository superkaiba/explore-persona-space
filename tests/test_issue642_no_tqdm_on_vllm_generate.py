"""Task #642 r4 — vLLM ZeroDivisionError guard regression test.

The GCP launch (eps-issue-642) crashed deterministically on every cmft
stage-A worker: vLLM's ``_run_engine`` post-generation summary path divides
``total_in_toks / pbar.format_dict["elapsed"]`` (vllm/entrypoints/llm.py,
inside ``if use_tqdm:``). When a small batch finishes faster than tqdm's
clock resolution, ``elapsed`` is ``0.0`` and the call raises
``ZeroDivisionError`` BEFORE any output is returned. Passing
``use_tqdm=False`` skips the entire progress-bar block (the buggy divide
included), so the fix is to disable tqdm on every ``llm.generate`` call in
the worker.

This test statically asserts that EVERY ``llm.generate(...)`` call (and any
``*.generate(...)`` on an object named ``llm``) under ``scripts/issue_642/``
passes ``use_tqdm=False`` as a keyword argument. AST-based so it is robust
to calls that span multiple physical lines.
"""

from __future__ import annotations

import ast
from pathlib import Path

ISSUE_642_DIR = Path(__file__).resolve().parent.parent / "scripts" / "issue_642"


def _llm_generate_calls_missing_use_tqdm_false(py_path: Path) -> list[str]:
    """Return ``"<file>:<line>"`` for every ``llm.generate(...)`` call in
    ``py_path`` that does NOT pass ``use_tqdm=False`` as a keyword arg."""
    tree = ast.parse(py_path.read_text(), filename=str(py_path))
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match `<obj named llm>.generate(...)` — the worker's vLLM call shape.
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "generate"
            and isinstance(func.value, ast.Name)
            and func.value.id == "llm"
        ):
            continue
        has_use_tqdm_false = any(
            kw.arg == "use_tqdm" and isinstance(kw.value, ast.Constant) and kw.value.value is False
            for kw in node.keywords
        )
        if not has_use_tqdm_false:
            bad.append(f"{py_path}:{node.lineno}")
    return bad


def test_all_llm_generate_calls_disable_tqdm() -> None:
    """Every ``llm.generate`` call in scripts/issue_642 must pass use_tqdm=False."""
    assert ISSUE_642_DIR.is_dir(), f"missing dir: {ISSUE_642_DIR}"
    bad: list[str] = []
    saw_any = False
    for py_path in sorted(ISSUE_642_DIR.rglob("*.py")):
        calls_bad = _llm_generate_calls_missing_use_tqdm_false(py_path)
        bad.extend(calls_bad)
        # Track that we actually exercised the worker file (guards against a
        # silently-empty match set if the worker is renamed/moved).
        tree = ast.parse(py_path.read_text(), filename=str(py_path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "generate"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "llm"
            ):
                saw_any = True
    assert saw_any, (
        "no `llm.generate(...)` calls found under scripts/issue_642 — the worker "
        "may have been renamed; update this regression test's match shape."
    )
    assert not bad, (
        "vLLM ZeroDivisionError guard regressed (missing use_tqdm=False):\n" + "\n".join(bad)
    )
