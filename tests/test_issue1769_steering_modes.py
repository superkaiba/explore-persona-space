"""#1769 DeltaHook mode pins: the EXACT (forward-pass, position) edit set per arm.

A recording-hook pair brackets the DeltaHook on the SAME decoder block
(forward hooks run in registration order and each hook receives the previous
hook's returned output), so the per-pass diff between the pre- and
post-DeltaHook outputs pins exactly which (pass, position) pairs each mode
edits — on a from-config 2-layer real-vocab Qwen through the production
``generate_batch`` entrypoint.

Expected sets (T = padded prompt length; passes 0..N_f-1; decode passes are
T=1 KV-cache slices):
- ``neither``      -> {} (no hook)
- ``prefill_only`` -> {(0, p) for p in range(T)}
- ``decode_only``  -> {(k, 0) for k in 1..N_f-1}
- ``both``         -> {(0, T-1)} | {(k, 0) for k in 1..N_f-1}
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from explore_persona_space.experiments.issue1415.steering import (
    DeltaHook,
    generate_batch,
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN = 64
N_LAYERS = 2
LAYER = 1
MAX_NEW = 6


@pytest.fixture(scope="module")
def model_and_tok():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    cfg.hidden_size = HIDDEN
    cfg.intermediate_size = 2 * HIDDEN
    cfg.num_hidden_layers = N_LAYERS
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.float32)
    model.eval()
    return model, tok


class _Recorder:
    """Forward hook capturing the block's (possibly hook-edited) output per pass."""

    def __init__(self):
        self.outputs: list[torch.Tensor] = []

    def __call__(self, _module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        self.outputs.append(h.detach().clone())
        return None  # observe only


def _edit_set(model, tok, arm: str, delta: torch.Tensor) -> tuple[set, int, int, int]:
    """Run one generate_batch draw with recorders bracketing the (optional)
    DeltaHook; return ({(pass, position), ...}, n_passes, T, n_edits)."""
    contexts = [
        {"system": None, "user": "Why is the sky blue?"},
        {"system": "You are a pirate captain.", "user": "Why is the sky blue?"},
    ]
    block = model.model.layers[LAYER]
    pre = _Recorder()
    h_pre = block.register_forward_hook(pre)
    hook = None
    kwargs = {
        "neither": None,
        "prefill_only": {"prefill_all": True},
        "decode_only": {"decode_only": True},
        "both": {"all_positions": True},
    }[arm]
    if kwargs is not None:
        hook = DeltaHook(model, LAYER, delta, alpha=2.0, **kwargs)
        hook.install()
    post = _Recorder()
    h_post = block.register_forward_hook(post)
    try:
        outs = generate_batch(
            model,
            tok,
            contexts,
            n=1,
            hook=hook,
            max_new_tokens=MAX_NEW,
            temperature=0.0,
            seed_base=42,
        )
    finally:
        h_post.remove()
        if hook is not None:
            hook.remove()
        h_pre.remove()
    assert len(outs) == 2 and all(len(rows) == 1 for rows in outs)
    assert len(pre.outputs) == len(post.outputs) >= 2, "expected >= 1 decode pass"
    edits: set[tuple[int, int]] = set()
    for k, (a, b) in enumerate(zip(pre.outputs, post.outputs, strict=True)):
        diff = (a != b).any(dim=-1)  # (B, T_k)
        for pos in torch.nonzero(diff.any(dim=0)).flatten().tolist():
            edits.add((k, int(pos)))
    T = pre.outputs[0].shape[1]
    n_edits = hook.n_edits if hook is not None else 0
    return edits, len(pre.outputs), T, n_edits


@pytest.fixture(scope="module")
def delta():
    torch.manual_seed(1769)
    return torch.randn(HIDDEN)


def test_neither_edits_nothing(model_and_tok, delta):
    model, tok = model_and_tok
    edits, n_passes, _T, n_edits = _edit_set(model, tok, "neither", delta)
    assert edits == set()
    assert n_edits == 0
    assert n_passes >= 2


def test_prefill_only_edits_all_prompt_positions_first_pass_only(model_and_tok, delta):
    model, tok = model_and_tok
    edits, n_passes, T, n_edits = _edit_set(model, tok, "prefill_only", delta)
    assert edits == {(0, p) for p in range(T)}, (sorted(edits)[:5], T)
    assert n_edits == 1
    assert n_passes >= 2  # decode passes exist and were untouched


def test_decode_only_edits_every_decode_pass_never_the_prefill(model_and_tok, delta):
    model, tok = model_and_tok
    edits, n_passes, _T, n_edits = _edit_set(model, tok, "decode_only", delta)
    assert edits == {(k, 0) for k in range(1, n_passes)}, sorted(edits)[:5]
    assert n_edits == n_passes - 1


def test_both_edits_last_prompt_position_then_every_decode_pass(model_and_tok, delta):
    model, tok = model_and_tok
    edits, n_passes, T, n_edits = _edit_set(model, tok, "both", delta)
    assert edits == {(0, T - 1)} | {(k, 0) for k in range(1, n_passes)}, sorted(edits)[:5]
    assert n_edits == n_passes


def test_modes_are_mutually_exclusive(model_and_tok, delta):
    model, _tok = model_and_tok
    with pytest.raises(AssertionError, match="mutually exclusive"):
        DeltaHook(model, LAYER, delta, alpha=1.0, prefill_all=True, decode_only=True)
    with pytest.raises(AssertionError, match="mutually exclusive"):
        DeltaHook(model, LAYER, delta, alpha=1.0, all_positions=True, prefill_all=True)
    with pytest.raises(AssertionError, match="mutually exclusive"):
        DeltaHook(model, LAYER, delta, alpha=1.0, all_positions=True, decode_only=True)
    with pytest.raises(AssertionError, match="mutually exclusive"):
        DeltaHook(model, LAYER, delta, alpha=1.0, edit_position=3, decode_only=True)


def test_arm_at_refuses_1769_modes(model_and_tok, delta):
    model, _tok = model_and_tok
    hook = DeltaHook(model, LAYER, delta, alpha=1.0, prefill_all=True)
    with pytest.raises(AssertionError, match="incompatible"):
        hook.arm_at(3)
