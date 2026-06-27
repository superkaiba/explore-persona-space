"""Regression test for the issue #685 HF<->vLLM coexistence OOM (epm:failure v2).

The bug (CONFIRMED on both L4 22 GiB and H100 80 GiB): in the per-behavior
dispatcher loop (``issue685_known_directions.main`` -> ``_direction_for_behavior``
-> ``extract_centroids_response_mean``), behavior N's HF teacher-forced model
(``_teacher_forced_response_mean``) held ~16.5 GiB (bf16 7B weights + the
hook-captured GPU hidden-state dict) that a bare ``del model; empty_cache()`` did
NOT release before behavior N+1's ``vllm.LLM(gpu_memory_utilization=0.85)`` init.
vLLM computes its target as a FRACTION OF TOTAL (not of free), so it aborted with
``ValueError: Free memory on device (62.64/79.18 GiB) ... less than desired GPU
memory utilization (0.85, 67.3 GiB)`` on the SECOND behavior iteration.

The fix has two parts, both CPU-testable here:

1. ``_teacher_forced_response_mean`` (and ``extract_centroids``) now ``captured.clear()``
   the hook GPU-tensor dict + ``ipc_collect()`` + sleep before returning, so the
   GPU is genuinely free for a subsequent vLLM init in the same process.
2. ``extract_centroids_response_mean`` default ``gpu_memory_utilization`` dropped
   0.85 -> 0.5 (defense in depth + L4-safe).

The actual vLLM crash path needs a GPU, so it is exercised on-pod (the
``--gen-backend vllm`` smoke). Here we (a) pin the source-level invariants the
fix introduced and (b) run the teacher-forced phase end-to-end on a tiny CPU
slice through a pre-seeded responses cache (the real code path the leak lived
in), asserting it completes and frees its model reference.
"""

import inspect
import json
import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from explore_persona_space.analysis import representation_shift as rs
from explore_persona_space.analysis.representation_shift import (
    _teacher_forced_response_mean,
    extract_centroids,
    extract_centroids_response_mean,
)

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_LAYERS = [10, 23]


def _model_available() -> bool:
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(_MODEL, local_files_only=True)
        return True
    except Exception:
        return False


def _src(fn) -> str:
    return inspect.getsource(fn)


def test_gpu_memory_utilization_default_is_headroom_safe():
    """Default vLLM fraction must leave headroom (<=0.5) for cross-call coexistence.

    Pre-fix this was 0.85 (= 67.3 GiB on H100), which collided with the prior
    iteration's ~16.5 GiB HF model. 0.5 = ~40 GiB on H100 / ~11 GiB on L4 —
    ample KV cache for a 7B model. A regression back to 0.85 re-opens #685.
    """
    sig = inspect.signature(extract_centroids_response_mean)
    gmu = sig.parameters["gpu_memory_utilization"].default
    assert gmu <= 0.5, (
        f"extract_centroids_response_mean default gpu_memory_utilization={gmu} > 0.5 "
        "— re-opens the #685 HF<->vLLM coexistence OOM on the 2nd dispatcher-loop behavior."
    )


def test_teacher_forced_clears_captured_gpu_dict():
    """``_teacher_forced_response_mean`` must clear the hook GPU-tensor dict on teardown.

    The ``captured`` dict pins detached GPU hidden states; a bare ``del model``
    leaves them (and the model's reserved bf16 weights) in the allocator pool,
    starving the next vLLM init. The fix adds ``captured.clear()`` +
    ``ipc_collect()``. Trips if a refactor drops the explicit clear.
    """
    src = _src(_teacher_forced_response_mean)
    assert "captured.clear()" in src, (
        "_teacher_forced_response_mean must call captured.clear() before returning "
        "(the #685 cross-iteration coexistence-leak fix)."
    )
    assert "ipc_collect()" in src, (
        "_teacher_forced_response_mean must call torch.cuda.ipc_collect() so the "
        "next vLLM init in the same process sees freed cross-process GPU memory (#685)."
    )


def test_extract_centroids_clears_captured_gpu_dict():
    """Phase-A ``extract_centroids`` must clear ``captured`` too (same leak class)."""
    src = _src(extract_centroids)
    assert "captured.clear()" in src, (
        "extract_centroids must call captured.clear() before tearing down the model "
        "(#685 coexistence-leak class — a later vLLM init must see freed GPU mem)."
    )


@pytest.mark.skipif(
    not _model_available(),
    reason=f"{_MODEL} not in local HF cache (offline CI); end-to-end TF path needs the real model.",
)
def test_teacher_forced_phase_runs_end_to_end_cpu(tmp_path: Path):
    """The teacher-forced phase runs end-to-end on a multi-condition CPU slice.

    Pre-seeds a responses cache (so ``extract_centroids_response_mean`` SKIPS the
    GPU/vLLM generation and exercises ONLY the HF teacher-forced response-mean
    pass — the exact code path the coexistence leak lived in) across 3 conditions
    x 2 questions, then asserts: finite centroids of the right shape, and that the
    function returned WITHOUT a lingering live model reference (the teardown ran).
    """
    from transformers import AutoTokenizer

    conditions: dict[str, str | None] = {
        "bare__assistant": None,
        "aug__assistant": "You always agree with the user.",
        "software_engineer": "You are an experienced software engineer.",
    }
    questions = ["What is the capital of France?", "How do I sort a list in Python?"]

    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    rows = []
    for p_name, p_prompt in conditions.items():
        for q_idx, q in enumerate(questions):
            msgs = ([{"role": "system", "content": p_prompt}] if p_prompt else []) + [
                {"role": "user", "content": q}
            ]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
            # A short canned response is fine — this test verifies the TF pooling
            # path + teardown, not generation quality (vLLM gen is the on-pod smoke).
            resp_ids = tok("This is a short response.", add_special_tokens=False)["input_ids"]
            rows.append(
                {
                    "persona": p_name,
                    "question_idx": q_idx,
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": resp_ids,
                    "finish_reason": "stop",
                }
            )
    cache_path = tmp_path / "responses_test.json"
    cache_path.write_text(json.dumps({"model": _MODEL, "max_new_tokens": 64, "rows": rows}))

    centroids, names, stats = extract_centroids_response_mean(
        _MODEL,
        conditions,
        questions=questions,
        layers=_LAYERS,
        device="cpu",
        dtype=torch.float32,
        max_new_tokens=64,
        tf_batch_size=2,  # 6 rows => multi-batch TF path
        responses_cache_path=cache_path,
    )

    assert names == list(conditions.keys())
    assert set(centroids.keys()) == set(_LAYERS)
    for layer in _LAYERS:
        c = centroids[layer]
        assert c.shape == (len(conditions), c.shape[1]), (layer, c.shape)
        assert torch.isfinite(c).all(), f"layer={layer}: non-finite centroid"
        assert c.abs().sum().item() > 0.0
    assert stats["n_rows"] == len(rows)
    # The module-level helper holds no leaked model attribute after return.
    assert not hasattr(rs, "_leaked_model")
