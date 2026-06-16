"""Regression for the vLLM OOM at #545 — engine-init (round-1) AND
extract-phase log_softmax (round-3) OOM, both on pod-545.

Round-1 crash (22:12:08, ~80s after launch): inside
``predictors_zoo.extract_clouds_and_outdist_gpu`` the HF base model
(``AutoModelForCausalLM``) and the lazily-loaded vLLM engine CO-RESIDE on one
GPU in one process — the clouds sub-phase elicits nl-cloud text via vLLM then
teacher-forces it through the HF model, and the outdist sub-phase does the same
per probe pair, so neither model can be freed before the other. vLLM read
free-memory-at-startup and rejected init because ``gpu_memory_utilization=0.85``
requested 67.3 GiB > ~63 GiB free with the HF model resident.

Round-3 crash (extract phase, ~25 min in): the r3 fix lowered util 0.85 -> 0.70
and added a pre-init free-memory assert, but the assert ran when ONLY the HF
model was resident (no vLLM yet). The real HF residency under teacher-forcing
(output_hidden_states + max_model_len=4096 + KV cache) is 22.0 GiB MEASURED, not
the 16 GiB the r3 brief estimated. With vLLM at 0.70 (56.99 GiB observed) the
total reached ~79 GiB on a 79.18 GiB H100 — only 206 MiB free — and the extract
phase's intermediate tensors (log_softmax) OOM'd mid-run.

The fix (round-4):
  1. Lower ``JS_GPU_MEM_UTIL`` 0.70 -> 0.60 so 22.0 (HF) + 47.5 (vLLM) = 69.5 GiB,
     leaving ~9.5 GiB working-memory headroom for the extract-phase intermediates.
  2. Set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` at module top (in
     BOTH the dispatcher and predictors_zoo) BEFORE the first ``import torch`` so
     the allocator defragments reserved-but-unallocated memory — the canonical
     PyTorch CUDA OOM mitigation the round-3 error message itself recommended.
  3. Re-tune the pre-init free-memory floor, pinned to the actual util
     (``JS_VLLM_PREINIT_MIN_FREE_GIB = util * total + 2.5``), so a future util
     change keeps the assert correct.
  4. Keep the explicit teardown (``del model``/``del tokenizer``/``del llm`` +
     ``gc.collect()`` + ``torch.cuda.empty_cache()`` + ``torch.cuda.synchronize()``)
     before the function returns to the in-process CPU build phase.

These tests pin the constants + the guard contract + the teardown call order +
the allocator env-var-before-torch ordering WITHOUT a GPU (the real path needs a
7B model on an H100). The guard-contract test mirrors the EXACT production assert
against the real module constants, so a regression that loosens the floor below
the request, or re-raises the util, trips here.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from explore_persona_space.experiments.behavior_testbed_545 import predictors_zoo as zoo

# An H100-80 reports ~79.18 GiB total; the round-1/round-3 crash logs show 79.18.
_H100_TOTAL_GIB = 79.18
# Measured resident size of the co-resident HF base model under realistic
# extract-phase conditions (round-3 OOM log: HF process held 21.98 GiB).
_HF_MODEL_RESIDENT_GIB = 22.0


# ---------------------------------------------------------------------------
# 1. Constants: util lowered to clear BOTH OOM regimes; floor clears the request.
# ---------------------------------------------------------------------------


def test_gpu_mem_util_lowered_to_clear_extract_phase_oom():
    # Round-1 ran at 0.85 (init OOM); round-3 ran at 0.70 (extract-phase OOM).
    # The round-4 fix must be strictly below BOTH so the extract-phase
    # intermediates have working-memory headroom.
    assert zoo.JS_GPU_MEM_UTIL < 0.70
    # And still a sane positive fraction (not accidentally 0 / negative).
    assert 0.0 < zoo.JS_GPU_MEM_UTIL <= 0.65


def test_hf_model_residency_constant_uses_measured_not_estimated():
    # The r3 fix used a 16 GiB parameter-count estimate; the measured residency
    # under teacher-forcing is ~22 GiB. The module must carry the measured value.
    assert zoo.JS_HF_MODEL_RESIDENT_GIB >= 22.0


def test_preinit_floor_consistent_with_request_and_hf_residency():
    """The free-memory floor must (a) clear the vLLM request so a healthy run
    passes, and (b) leave room for the resident ~22 GiB HF model PLUS
    working-memory headroom for the extract-phase intermediates (the round-3
    OOM cause), so the run is actually feasible."""
    requested_gib = zoo.JS_GPU_MEM_UTIL * _H100_TOTAL_GIB
    # The floor is the free-memory we DEMAND before init; it must be at least
    # the request (else the assert would pass yet vLLM would still OOM).
    assert requested_gib <= zoo.JS_VLLM_PREINIT_MIN_FREE_GIB
    # Feasibility: vLLM's request + the resident HF model (~22 GiB measured) must
    # fit in total GPU memory with working-memory headroom (>= 5 GiB) for the
    # extract-phase intermediates that OOM'd at round-3's 0.70 util.
    headroom = _H100_TOTAL_GIB - (requested_gib + zoo.JS_HF_MODEL_RESIDENT_GIB)
    assert headroom >= 5.0, f"only {headroom:.1f} GiB working-memory headroom (need >=5)"
    # The floor must itself be achievable with the HF model resident
    # (free = total - HF model >= floor), else the assert can never pass.
    assert _H100_TOTAL_GIB - zoo.JS_HF_MODEL_RESIDENT_GIB >= zoo.JS_VLLM_PREINIT_MIN_FREE_GIB


def test_preinit_floor_pinned_to_util_with_margin():
    """The floor is parameterized off the util (util * total + 2.5 GiB margin)
    so a future util change keeps the assert correct, not a stale literal."""
    expected = max(zoo.JS_GPU_MEM_UTIL * _H100_TOTAL_GIB + 2.5, 0.0)
    assert pytest.approx(expected, abs=0.01) == zoo.JS_VLLM_PREINIT_MIN_FREE_GIB


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
    # After the fix, with the ~22 GiB HF model resident on an H100, ~57 GiB is
    # free. That must PASS the 0.60-pinned floor (~50 GiB).
    free_with_hf_resident = _H100_TOTAL_GIB - _HF_MODEL_RESIDENT_GIB  # ~57.2 GiB
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


def test_get_llm_uses_lowered_util_constant_not_hardcoded_float():
    src = _extract_fn_source()
    # The vLLM construction must reference the constant, not a hardcoded float.
    assert "gpu_memory_utilization=JS_GPU_MEM_UTIL" in src
    assert "gpu_memory_utilization=0.85" not in src
    assert "gpu_memory_utilization=0.70" not in src


def test_get_llm_llm_call_keyword_is_the_util_constant_ast():
    """AST traversal (not source-string match): the ``LLM(...)`` construction's
    ``gpu_memory_utilization`` keyword value must be the ``JS_GPU_MEM_UTIL``
    Name node, never a literal — so a regression that hardcodes a float (even one
    matching the current value) trips here. (Codex r3 nit: strengthen the
    string-match test to actual AST inspection.)"""
    tree = ast.parse(_extract_fn_source())
    llm_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LLM"
    ]
    assert len(llm_calls) == 1, f"expected exactly one LLM(...) call, found {len(llm_calls)}"
    kwargs = {kw.arg: kw.value for kw in llm_calls[0].keywords}
    assert "gpu_memory_utilization" in kwargs, "LLM(...) missing gpu_memory_utilization kwarg"
    val = kwargs["gpu_memory_utilization"]
    assert isinstance(val, ast.Name) and val.id == "JS_GPU_MEM_UTIL", (
        "gpu_memory_utilization must be the JS_GPU_MEM_UTIL constant, not a literal; "
        f"got {ast.dump(val)}"
    )


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


# ---------------------------------------------------------------------------
# 4. Allocator env var (expandable_segments) is set BEFORE the first `import
#    torch` in BOTH module entrypoints. AST traversal of the file: find the
#    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ...) statement and assert
#    it precedes every `import torch` statement (top-level OR nested in a
#    function). (#545 round-4: torch reads the allocator config once at CUDA
#    init, so a setdefault AFTER torch is imported is a no-op.)
# ---------------------------------------------------------------------------

_PREDICTORS_ZOO_PATH = Path(inspect.getfile(zoo))
_DISPATCHER_PATH = _PREDICTORS_ZOO_PATH.parents[3].parent / "scripts" / "issue545_metric_race.py"


def _alloc_conf_setdefault_lineno(tree: ast.Module) -> int | None:
    """Lineno of `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ...)`, or None."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "PYTORCH_CUDA_ALLOC_CONF"
        ):
            return node.lineno
    return None


def _torch_import_linenos(tree: ast.Module) -> list[int]:
    """Linenos of every `import torch` / `from torch ...` statement (any depth)."""
    out: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Import)
            and any(
                alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names
            )
        ) or (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "torch" or node.module.startswith("torch."))
        ):
            out.append(node.lineno)
    return out


@pytest.mark.parametrize(
    "path",
    [_PREDICTORS_ZOO_PATH, _DISPATCHER_PATH],
    ids=["predictors_zoo", "dispatcher"],
)
def test_alloc_conf_setdefault_precedes_every_torch_import(path):
    assert path.is_file(), f"source file missing: {path}"
    tree = ast.parse(path.read_text())
    setdefault_lineno = _alloc_conf_setdefault_lineno(tree)
    assert setdefault_lineno is not None, (
        f"{path.name}: os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', ...) not found "
        "(allocator must be configured before torch CUDA init)"
    )
    for torch_lineno in _torch_import_linenos(tree):
        assert setdefault_lineno < torch_lineno, (
            f"{path.name}: PYTORCH_CUDA_ALLOC_CONF setdefault (line {setdefault_lineno}) "
            f"must precede `import torch` (line {torch_lineno}) — torch reads the allocator "
            "config once at CUDA init, so a setdefault after it is a no-op"
        )


def test_alloc_conf_value_is_expandable_segments():
    """The configured value must enable expandable_segments (the fragmentation
    mitigation), in both files."""
    for path in (_PREDICTORS_ZOO_PATH, _DISPATCHER_PATH):
        tree = ast.parse(path.read_text())
        found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "PYTORCH_CUDA_ALLOC_CONF"
                and isinstance(node.args[1], ast.Constant)
                and "expandable_segments:True" in str(node.args[1].value)
            ):
                found = True
        assert found, f"{path.name}: PYTORCH_CUDA_ALLOC_CONF not set to expandable_segments:True"


def test_module_file_path_is_real():
    # Cheap guard: the module under test is the real worktree file, not a stub.
    p = Path(inspect.getfile(zoo))
    assert p.name == "predictors_zoo.py"
    assert p.is_file()
