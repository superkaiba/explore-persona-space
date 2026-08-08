"""Capture-side load-bearing invariant for #1491: the fp32 parity probe must
leave the production model bit-identical.

Round-4a BLOCKER B1. ``_batched_capture_parity_gate`` transiently casts the
model to fp32 so the plan-registered bars measure batching logic rather than
bf16 kernel noise. ``Module.to(dtype)`` casts floating-point BUFFERS as well as
parameters, and Qwen2's rotary ``inv_freq`` is fp32 even on a bf16-loaded
model — so a bare fp32 -> bf16 round-trip permanently degraded it (measured
3.65e-3 max relative error). Every production capture after the probe would
then run a different RoPE than the parent/#779 convention, while the probe
legs themselves ran clean fp32 — i.e. the gate would certify a rig production
does not use. Silent: no crash, correct-looking tensors.

Params alone DO round-trip exactly, which is why a params-only equality check
(the original evidence for this fix) passed and missed it. This test therefore
asserts BUFFERS as well, and runs the model in bf16 — the cast branch never
executes under fp32, so an fp32 fixture cannot reach the bug.

Offline by construction: the config is built directly rather than fetched, and
the test SKIPS if a real tokenizer is not already available locally. Anything
in tests/ runs in every issue's Step 9c gate, so it must never depend on a live
Hub fetch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1491_ladder_generate_capture as D  # noqa: E402

H_DIM = 64


def _tiny_bf16_qwen2():
    """A 2-layer Qwen2 in bf16, constructed offline (no from_pretrained).

    Built via ``from_config(dtype=bfloat16)`` — the shape production uses
    (``from_pretrained(torch_dtype=bf16)``), which leaves the rotary
    ``inv_freq`` buffer fp32.

    Do NOT build this as ``Qwen2ForCausalLM(cfg).to(torch.bfloat16)``: that
    ``.to()`` casts inv_freq to bf16 itself, so the fixture arrives
    pre-degraded and the fp32 round-trip under test looks clean. A fixture
    built that way cannot detect B1 at all — verified, it was this test's
    first version.
    """
    from transformers import AutoModelForCausalLM, Qwen2Config

    cfg = Qwen2Config(
        vocab_size=151936,
        hidden_size=H_DIM,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    return AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)


def _tokenizer_or_skip():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception as exc:
        pytest.skip(f"Qwen tokenizer not available offline; skipping rather than fetching: {exc}")


def test_fp32_probe_restores_params_and_buffers_bit_exactly(monkeypatch):
    """The cast branch must leave BOTH params and fp buffers bit-identical.

    Guards the regression directly: with the buffer snapshot removed, inv_freq
    comes back bf16 and this fails on the buffer assertion while the parameter
    assertion still passes.
    """
    tok = _tokenizer_or_skip()
    hf = _tiny_bf16_qwen2()

    # The feasibility guard reads CUDA free memory; this exercise is CPU-only.
    monkeypatch.setattr(D, "_assert_fp32_probe_feasible", lambda *a, **k: None)

    fp_buffers_before = {
        name: (buf.dtype, buf.detach().clone())
        for name, buf in hf.named_buffers()
        if buf.is_floating_point()
    }
    # The bug is invisible without at least one fp32 buffer to degrade. A bf16
    # fp buffer here means the fixture arrived pre-degraded (see
    # _tiny_bf16_qwen2) and the test would pass vacuously.
    assert fp_buffers_before, "fixture has no floating-point buffers — cannot detect B1"
    assert any(dtype == torch.float32 for dtype, _ in fp_buffers_before.values()), (
        "fixture has no fp32 buffer on a bf16 model — it cannot reach B1; "
        f"got {[(n, str(d)) for n, (d, _) in fp_buffers_before.items()]}"
    )
    params_before = {n: p.detach().clone() for n, p in hf.named_parameters()}

    assert next(hf.parameters()).dtype == torch.bfloat16, "fixture must be bf16 to hit the cast"

    ok, msg = D._batched_capture_parity_gate(
        hf,
        tok,
        ["hi there", "second prompt"],
        ["a reply", "another reply"],
        [0, 1],
        [0, 1],
        H_DIM,
        2,
    )
    assert ok, f"parity gate should pass in fp32 on a tiny model: {msg}"
    assert "cast=True" in msg, f"cast branch did not execute — fixture wrong: {msg}"

    buffers_after = dict(hf.named_buffers())
    for name, (dtype_before, value_before) in fp_buffers_before.items():
        after = buffers_after[name]
        assert after.dtype == dtype_before, (
            f"buffer {name} dtype degraded {dtype_before} -> {after.dtype} — "
            "the fp32 probe permanently changed the production model (B1)"
        )
        assert torch.equal(after, value_before), f"buffer {name} not restored bit-exactly"

    params_after = dict(hf.named_parameters())
    for name, value_before in params_before.items():
        assert torch.equal(params_after[name], value_before), (
            f"parameter {name} not restored bit-exactly"
        )
