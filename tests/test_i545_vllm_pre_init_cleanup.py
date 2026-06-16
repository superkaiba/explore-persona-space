"""Regression for the vLLM engine-init OOM at #545 round-1 launch.

Crash (pod-545, 22:12:08, ~80s after launch): inside
``predictors_zoo.extract_clouds_and_outdist_gpu`` the HF base model
(``AutoModelForCausalLM``, ~16 GiB bf16) and the lazily-loaded vLLM engine
CO-RESIDE on one GPU in one process — the clouds sub-phase elicits nl-cloud
text via vLLM then teacher-forces it through the HF model, and the outdist
sub-phase does the same per probe pair, so neither model can be freed before
the other. vLLM read free-memory-at-startup (~63 GiB with the HF model
resident) and rejected init because ``gpu_memory_utilization=0.85`` requested
67.3 GiB > free → ``ValueError: Free memory on device ... less than desired GPU
memory utilization``.

The fix (this round):
  1. Lower ``JS_GPU_MEM_UTIL`` 0.85 -> 0.70 so the vLLM request (~55 GiB)
     fits alongside the resident HF model.
  2. A pre-vLLM-init free-memory assert (``JS_VLLM_PREINIT_MIN_FREE_GIB``) that
     fails LOUD with a clear message instead of an opaque vLLM engine-init OOM.
  3. Explicit teardown (``del model``/``del tokenizer``/``del llm`` +
     ``gc.collect()`` + ``torch.cuda.empty_cache()`` + ``torch.cuda.synchronize()``)
     before the function returns to the in-process CPU build phase.

These tests pin the constants + the guard contract + the teardown call order
WITHOUT a GPU (the real path needs a 7B model on an H100). The guard-contract
test mirrors the EXACT production assert against the real module constants, so
a regression that loosens the floor below the request, or re-raises the util,
trips here.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from explore_persona_space.experiments.behavior_testbed_545 import predictors_zoo as zoo

# An H100-80 reports ~79.18 GiB total; the round-1 crash log shows 79.18.
_H100_TOTAL_GIB = 79.18


# ---------------------------------------------------------------------------
# 1. Constants: util lowered below the crashing value; floor clears the request.
# ---------------------------------------------------------------------------


def test_gpu_mem_util_lowered_from_crashing_value():
    # The round-1 crash ran at 0.85; the fix must be strictly lower.
    assert zoo.JS_GPU_MEM_UTIL < 0.85
    # And still a sane positive fraction (not accidentally 0 / negative).
    assert 0.0 < zoo.JS_GPU_MEM_UTIL <= 0.80


def test_preinit_floor_consistent_with_request_and_hf_residency():
    """The free-memory floor must (a) clear the vLLM request so a healthy run
    passes, and (b) leave room for the resident ~16 GiB HF model so the run is
    actually feasible (request + HF model <= total)."""
    requested_gib = zoo.JS_GPU_MEM_UTIL * _H100_TOTAL_GIB
    # The floor is the free-memory we DEMAND before init; it must be at least
    # the request (else the assert would pass yet vLLM would still OOM).
    assert requested_gib <= zoo.JS_VLLM_PREINIT_MIN_FREE_GIB
    # Feasibility: vLLM's request + the resident HF model (~16 GiB) must fit in
    # total GPU memory, else 0.70 is still too high.
    hf_model_gib = 16.0
    assert requested_gib + hf_model_gib <= _H100_TOTAL_GIB
    # The floor must itself be achievable with the HF model resident
    # (free = total - HF model >= floor), else the assert can never pass.
    assert _H100_TOTAL_GIB - hf_model_gib >= zoo.JS_VLLM_PREINIT_MIN_FREE_GIB


# ---------------------------------------------------------------------------
# 2. Guard contract: replay the production assert against the real constants.
# ---------------------------------------------------------------------------


def _preinit_guard(free_gib: float) -> None:
    """The exact predicate the production ``_get_llm`` closure runs before
    constructing ``LLM``. Kept in lockstep with the source (asserted by the
    AST test below)."""
    assert free_gib >= zoo.JS_VLLM_PREINIT_MIN_FREE_GIB, (
        f"vLLM pre-init free GPU memory {free_gib:.1f} GiB < floor "
        f"{zoo.JS_VLLM_PREINIT_MIN_FREE_GIB:.1f} GiB"
    )


def test_guard_passes_at_healthy_free_memory_post_fix():
    # After the fix, with the HF model resident on an H100, ~63 GiB is free.
    # That must now PASS (it failed at the old 0.85 request of 67.3 GiB).
    free_with_hf_resident = _H100_TOTAL_GIB - 16.0  # ~63.2 GiB
    _preinit_guard(free_with_hf_resident)  # no raise


def test_guard_fails_loud_when_free_memory_short():
    # A regression (HF model ballooned / another GPU consumer) leaves too little
    # free → the assert must raise a clear AssertionError, not an opaque vLLM OOM.
    with pytest.raises(AssertionError, match="vLLM pre-init free GPU memory"):
        _preinit_guard(zoo.JS_VLLM_PREINIT_MIN_FREE_GIB - 1.0)


# ---------------------------------------------------------------------------
# 3. Source-level: the real closure actually wires the assert + util + teardown.
#    (AST inspection, so the functional guard test above can't drift from the
#    production code silently.)
# ---------------------------------------------------------------------------


def _extract_fn_source() -> str:
    return inspect.getsource(zoo.extract_clouds_and_outdist_gpu)


def test_get_llm_uses_lowered_util_constant_not_hardcoded_085():
    src = _extract_fn_source()
    # The vLLM construction must reference the constant, not a hardcoded float.
    assert "gpu_memory_utilization=JS_GPU_MEM_UTIL" in src
    assert "gpu_memory_utilization=0.85" not in src


def test_get_llm_asserts_free_memory_before_vllm_init():
    """The free-memory assert must execute BEFORE the ``LLM(...)`` construction
    in the closure (so the loud failure precedes the OOM-prone init)."""
    src = _extract_fn_source()
    tree = ast.parse(src)

    # Find the nested _get_llm function.
    get_llm = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_llm":
            get_llm = node
            break
    assert get_llm is not None, "_get_llm closure not found"

    assert_lineno = None
    llm_call_lineno = None
    for node in ast.walk(get_llm):
        # The free-memory assert references JS_VLLM_PREINIT_MIN_FREE_GIB.
        if isinstance(node, ast.Assert) and "JS_VLLM_PREINIT_MIN_FREE_GIB" in ast.dump(node):
            assert_lineno = node.lineno
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LLM":
            llm_call_lineno = node.lineno
    assert assert_lineno is not None, "free-memory assert missing in _get_llm"
    assert llm_call_lineno is not None, "LLM(...) construction missing in _get_llm"
    assert assert_lineno < llm_call_lineno, "free-memory assert must precede LLM init"


def test_teardown_frees_gpu_before_return():
    """The function must drop the GPU refs + force GC + empty cache before
    returning to the in-process CPU build phase."""
    src = _extract_fn_source()
    for needle in (
        "del model",
        "del tokenizer",
        "gc.collect()",
        "torch.cuda.empty_cache()",
        "torch.cuda.synchronize()",
    ):
        assert needle in src, f"teardown step missing: {needle!r}"


def test_module_file_path_is_real():
    # Cheap guard: the module under test is the real worktree file, not a stub.
    p = Path(inspect.getfile(zoo))
    assert p.name == "predictors_zoo.py"
    assert p.is_file()
