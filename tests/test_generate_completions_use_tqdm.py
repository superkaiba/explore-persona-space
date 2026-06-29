"""Task #715 hot-fix — vLLM ZeroDivisionError guard for the SHARED eval wrapper.

The #715 phase0 LoRA arm crashed on GCP (a2-ultragpu-1g) inside vLLM 0.11.0's
``_run_engine``: the post-generation throughput line divides
``total_in_toks / pbar.format_dict["elapsed"]`` (vllm/entrypoints/llm.py,
inside ``if use_tqdm:``). When a batch finishes before tqdm's clock ticks above
0, ``elapsed`` is ``0.0`` and the call raises ``ZeroDivisionError`` BEFORE any
output returns. The ``TQDM_DISABLE=1`` env var does NOT prevent it — that only
affects tqdm's rendering, not vLLM's ``use_tqdm`` parameter, which gates the
buggy throughput-calc block. Passing ``use_tqdm=False`` skips the whole block.

Unlike the per-issue #642 worker, this guards the PROJECT-SHARED wrapper
``src/explore_persona_space/eval/generation.py`` — every experiment that calls
``generate_completions`` / ``generate_lora_completions`` /
``generate_completions_with_history`` would hit the same bug otherwise.

AST-based so it is robust to calls that span multiple physical lines.
"""

from __future__ import annotations

import ast
from pathlib import Path

GENERATION_PY = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "explore_persona_space"
    / "eval"
    / "generation.py"
)


def _llm_generate_calls(tree: ast.AST) -> list[ast.Call]:
    """Every ``<obj named llm>.generate(...)`` call node — the vLLM call shape."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "llm"
    ]


def _has_use_tqdm_false(call: ast.Call) -> bool:
    return any(
        kw.arg == "use_tqdm" and isinstance(kw.value, ast.Constant) and kw.value.value is False
        for kw in call.keywords
    )


def test_all_llm_generate_calls_disable_tqdm() -> None:
    """Every ``llm.generate`` call in the shared wrapper must pass use_tqdm=False."""
    assert GENERATION_PY.is_file(), f"missing file: {GENERATION_PY}"
    tree = ast.parse(GENERATION_PY.read_text(), filename=str(GENERATION_PY))
    calls = _llm_generate_calls(tree)
    # Guard against a silently-empty match set if the wrapper is renamed/moved.
    assert calls, (
        "no `llm.generate(...)` calls found in eval/generation.py — the wrapper "
        "may have been renamed; update this regression test's match shape."
    )
    bad = [f"{GENERATION_PY}:{call.lineno}" for call in calls if not _has_use_tqdm_false(call)]
    assert not bad, (
        "vLLM ZeroDivisionError guard regressed (missing use_tqdm=False):\n" + "\n".join(bad)
    )
