"""Issue #653 round 9 — vLLM engine teardown helper regression tests.

The dx phase crashed on cell 2 when initializing a vLLM engine for the BASE
model, with::

    ValueError: Free memory on device (65.07/79.25 GiB) on startup is less than
    desired GPU memory utilization (0.85, 67.36 GiB).

Root cause: ``_generate_responses_vllm`` tore its engine down with the bare
``del llm; gc.collect(); torch.cuda.empty_cache()`` triad, which does NOT reap
vLLM v1's EngineCore worker SUBPROCESS synchronously. The reserved KV cache
stayed pinned, so the next ``LLM(...)`` init found too little free memory. The
fix adds ``_reap_vllm_engine`` which drives the documented teardown
(engine-core shutdown + guarded ``destroy_process_group``) before ``del``.

These tests run CPU-only (no GPU, no real vLLM engine) using stub objects:
the helper must be importable, callable, drive the v1 / v0 shutdown paths on
stubs that expose them, and NO-OP gracefully when no recognizable engine API
or no initialized process group is present.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from explore_persona_space.analysis.representation_shift import (
    _generate_responses_vllm,
    _reap_vllm_engine,
)


def test_helper_is_callable():
    """The helper exists and is callable (basic import sanity)."""
    assert callable(_reap_vllm_engine)
    assert callable(_generate_responses_vllm)


def test_noop_on_uninitialized_process_group():
    """With no initialized torch.distributed group, the helper must NOT call
    destroy_process_group() (which would raise off-pod / single-GPU)."""
    # In this CPU test process no process group is initialized.
    assert not torch.distributed.is_initialized()

    class _StubLLM:
        # No llm_engine -> the engine-shutdown branch is skipped entirely.
        pass

    # Must complete without raising despite no process group + no engine.
    _reap_vllm_engine(_StubLLM())
    # And the process group is still not initialized (helper did not touch it).
    assert not torch.distributed.is_initialized()


def test_drives_v1_engine_core_shutdown():
    """vLLM v1 path: llm.llm_engine.engine_core.shutdown() must be invoked once."""
    calls = {"engine_core": 0, "executor": 0}

    class _EngineCore:
        def shutdown(self):
            calls["engine_core"] += 1

    class _Executor:
        def shutdown(self):
            calls["executor"] += 1  # must NOT be called when engine_core exists

    class _Engine:
        engine_core = _EngineCore()
        model_executor = _Executor()

    class _StubLLM:
        llm_engine = _Engine()

    _reap_vllm_engine(_StubLLM())
    assert calls["engine_core"] == 1, "v1 engine_core.shutdown() should fire once"
    assert calls["executor"] == 0, "v0 fallback must not fire when v1 path exists"


def test_v0_fallback_executor_shutdown():
    """vLLM v0 fallback: when there is no engine_core, model_executor.shutdown()
    is the reaper that must fire."""
    calls = {"executor": 0}

    class _Executor:
        def shutdown(self):
            calls["executor"] += 1

    class _Engine:
        # No engine_core attribute at all -> fall back to model_executor.
        model_executor = _Executor()

    class _StubLLM:
        llm_engine = _Engine()

    _reap_vllm_engine(_StubLLM())
    assert calls["executor"] == 1, "v0 model_executor.shutdown() should fire once"


def test_noop_on_engine_without_shutdown_api():
    """An engine object exposing neither a callable engine_core.shutdown nor a
    callable model_executor.shutdown must be tolerated (no AttributeError)."""

    class _Engine:
        engine_core = object()  # has no shutdown()
        model_executor = object()  # has no shutdown()

    class _StubLLM:
        llm_engine = _Engine()

    # Must not raise.
    _reap_vllm_engine(_StubLLM())


def test_dx_gpu_cloud_uses_lowered_gpu_memory_utilization():
    """Static AST check (fix B): both vLLM generation calls inside
    ``_dx_gpu_cloud`` pass gpu_memory_utilization=0.6 (the lowered headroom),
    not 0.85. Other call sites elsewhere in the dispatcher are out of scope and
    intentionally left at 0.85."""
    dispatch = Path(__file__).resolve().parent.parent / "scripts" / "issue_653" / "i653_dispatch.py"
    tree = ast.parse(dispatch.read_text())

    dx_func = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_dx_gpu_cloud"
        ),
        None,
    )
    assert dx_func is not None, "_dx_gpu_cloud not found in i653_dispatch.py"

    gmu_values: list[float] = []
    for call in ast.walk(dx_func):
        if not isinstance(call, ast.Call):
            continue
        if not (isinstance(call.func, ast.Name) and call.func.id == "_generate_responses_vllm"):
            continue
        for kw in call.keywords:
            if kw.arg == "gpu_memory_utilization" and isinstance(kw.value, ast.Constant):
                gmu_values.append(kw.value.value)

    assert len(gmu_values) == 2, (
        f"expected 2 _generate_responses_vllm calls in _dx_gpu_cloud, got {len(gmu_values)}"
    )
    assert all(v == 0.6 for v in gmu_values), (
        f"both _dx_gpu_cloud vLLM calls must use gpu_memory_utilization=0.6, got {gmu_values}"
    )
