"""Regression for the vLLM OOM at #545 — a four-round OOM family on pod-545:
engine-init (r1), extract-phase log_softmax (r3), outdist HF logits transient
(r4), and clouds-phase log_softmax at the grown HF resident (r6).

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
model was resident (no vLLM yet). The HF residency under teacher-forcing at r3
was 22.0 GiB MEASURED. With vLLM at 0.70 (56.99 GiB observed) the total reached
~79 GiB on a 79.18 GiB H100 — only 206 MiB free — and the extract phase's
intermediate tensors (log_softmax) OOM'd mid-run.

Round-6 crash (clouds phase, 310 MiB short at log_softmax): the r4 fix dropped
util to 0.60 + enabled ``expandable_segments`` and cleared extract; but
``expandable_segments`` trades fragmentation reduction for a slightly HIGHER
peak resident, and the clouds-phase HF resident GREW 22 -> 30 GiB. With vLLM at
0.60 (49 GiB) + HF at 30 GiB = 79 GiB on a 79.18 GiB H100 → only 289 MiB free,
and the clouds-phase log_softmax (310 MiB) OOM'd.

The fix (round-8 — the PRIMARY load-bearing change):
  1. Lower ``JS_GPU_MEM_UTIL`` 0.60 -> 0.50 so 30.0 (HF, re-measured r6 ceiling)
     + 39.6 (vLLM = 0.50 * 79.18) = 69.6 GiB, leaving ~9.6 GiB working-memory
     headroom for the clouds/extract-phase log_softmax intermediates (vs only
     1.67 GiB at 0.60 with the grown 30 GiB HF resident).
  2. Update ``JS_HF_MODEL_RESIDENT_GIB`` 22.0 -> 30.0 (the r6 ceiling the dial
     must budget against — re-measure peak resident when enabling
     expandable_segments; the trade is real and bit at r6).
  3. The pre-init free-memory floor stays pinned to the util
     (``JS_VLLM_PREINIT_MIN_FREE_GIB = util * total + 2.5``); at 0.50 it is
     ~42.1 GiB, well below the ~78 GiB free at launch (before the HF model
     loads), so the assert passes comfortably.
  4. ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` (r4) + the explicit
     teardown (``del model``/``del tokenizer``/``del llm`` + ``gc.collect()`` +
     ``torch.cuda.empty_cache()`` + ``torch.cuda.synchronize()``) stay in place.

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
# Re-measured resident size of the co-resident HF base model at the r6 clouds
# peak (the r4 expandable_segments allocator raised the peak from the r3 22 GiB
# to ~30 GiB). The dial budgets against this ceiling, not the r3 22 GiB estimate.
_HF_MODEL_RESIDENT_GIB = 30.0


# ---------------------------------------------------------------------------
# 1. Constants: util lowered to clear BOTH OOM regimes; floor clears the request.
# ---------------------------------------------------------------------------


def test_gpu_mem_util_lowered_to_clear_extract_phase_oom():
    # r1 ran at 0.85 (init OOM); r3 at 0.70 (extract OOM); r4/r6 at 0.60 (clouds
    # OOM at the grown 30 GiB HF resident). The r8 fix drops to 0.50, strictly
    # below ALL prior values so the clouds/extract log_softmax intermediates
    # have working-memory headroom at the re-measured 30 GiB HF resident.
    assert zoo.JS_GPU_MEM_UTIL == 0.50
    # And still a sane positive fraction (not accidentally 0 / negative).
    assert 0.0 < zoo.JS_GPU_MEM_UTIL < 0.60


def test_hf_model_residency_constant_uses_measured_not_estimated():
    # The r3 fix used a 16 GiB parameter-count estimate; r3 measured ~22 GiB;
    # the r6 clouds peak re-measured ~30 GiB after expandable_segments raised the
    # peak. The module must carry the r6 ceiling so the dial budgets against it.
    assert zoo.JS_HF_MODEL_RESIDENT_GIB >= 30.0


def test_preinit_floor_consistent_with_request_and_hf_residency():
    """The free-memory floor must (a) clear the vLLM request so a healthy run
    passes, and (b) leave room for the resident ~30 GiB HF model PLUS
    working-memory headroom for the clouds/extract-phase log_softmax
    intermediates (the r6 OOM cause), so the run is actually feasible."""
    requested_gib = zoo.JS_GPU_MEM_UTIL * _H100_TOTAL_GIB
    # The floor is the free-memory we DEMAND before init; it must be at least
    # the request (else the assert would pass yet vLLM would still OOM).
    assert requested_gib <= zoo.JS_VLLM_PREINIT_MIN_FREE_GIB
    # Feasibility: vLLM's request + the resident HF model (~30 GiB re-measured)
    # must fit in total GPU memory with working-memory headroom (>= 5 GiB) for
    # the clouds/extract-phase log_softmax intermediates that OOM'd at r6's 0.60.
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


def test_jsutil_05_math_sanity():
    """At util=0.50, vLLM gets ~39.6 GiB; with HF resident at 30 GiB
    (measured r6 ceiling on Qwen-7B + KV + activations + expandable_segments),
    total = 69.6 GiB / 79.18 GiB H100 = 88% — leaves 9.6 GiB headroom for
    intermediate tensors (log_softmax etc.)."""
    from explore_persona_space.experiments.behavior_testbed_545.predictors_zoo import (
        JS_GPU_MEM_UTIL,
        JS_VLLM_PREINIT_MIN_FREE_GIB,
    )

    assert JS_GPU_MEM_UTIL == 0.50
    H100_GIB = 79.18
    HF_RESIDENT_GIB = 30.0  # measured r6 ceiling
    headroom = H100_GIB - (JS_GPU_MEM_UTIL * H100_GIB) - HF_RESIDENT_GIB
    assert headroom >= 8.0, f"insufficient headroom: {headroom:.1f} GiB"
    # The pre-init floor (~42.1 GiB) must sit comfortably below the ~78 GiB free
    # at launch (before the HF model loads), so the assert never false-fails.
    assert JS_VLLM_PREINIT_MIN_FREE_GIB < 78.0


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
    # After the r8 fix, even with the grown ~30 GiB HF model resident on an H100,
    # ~49 GiB is free. That must PASS the 0.50-pinned floor (~42.1 GiB). (At
    # launch the HF model is not yet resident — ~78 GiB free — so the real assert
    # passes with even more margin; this test models the worst-case HF-resident.)
    free_with_hf_resident = _H100_TOTAL_GIB - _HF_MODEL_RESIDENT_GIB  # ~49.2 GiB
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
    assert "gpu_memory_utilization=0.60" not in src
    assert "gpu_memory_utilization=0.50" not in src


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
