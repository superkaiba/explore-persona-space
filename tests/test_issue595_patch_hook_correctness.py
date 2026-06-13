"""Issue #595 — KV-cache patch-hook correctness (synthetic tiny Qwen2, CPU).

The Phase-2/3 patch wraps each Qwen2Attention.forward to substitute base-model
K/V at the patch positions into the KV cache during prefill (cache_position[0]==0),
persisting through decode. This test verifies, on a 2-layer CPU Qwen2:

  1. A SELF-patch (the same model substituting its own prefix K/V) is bit-identical
     to the unpatched forward — the wrapper only touches the named positions and
     introduces no other change.
  2. Patching a DIFFERENT model's prefix K/V changes the generation (the patch is
     actually applied, not a no-op).
  3. Positions OUTSIDE the patch set keep the trained model's own K/V (the
     substitution is localized to the named positions).
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
REPO = Path(__file__).resolve().parents[1]


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "issue595_prefix_carrier", REPO / "scripts" / "issue595_prefix_carrier.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_qwen2():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen2",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=200,
        max_position_embeddings=64,
        head_dim=16,
    )
    cfg._attn_implementation = "eager"  # KV-cache wrapping requires eager attention
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).eval()


def _attns(model):
    return [model.model.layers[i].self_attn for i in range(len(model.model.layers))]


def _capture_base_prefix(mod, model, ids, positions):
    """Capture KV-cache entries at ``positions`` for every layer (one prefill)."""
    from transformers.cache_utils import DynamicCache

    captured: dict[int, tuple] = {}
    attns = _attns(model)
    origs = [a.forward for a in attns]

    def make_cap(attn, orig):
        def fwd(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kw,
        ):
            out = orig(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kw,
            )
            if cache_position is not None and int(cache_position[0]) == 0 and past_key_values:
                lc = past_key_values.layers[self.layer_idx]
                pos = torch.as_tensor(positions, device=lc.keys.device)
                captured[self.layer_idx] = (
                    lc.keys[:, :, pos, :].detach().clone(),
                    lc.values[:, :, pos, :].detach().clone(),
                )
            return out

        return fwd

    for a, o in zip(attns, origs, strict=True):
        a.forward = types.MethodType(make_cap(a, o), a)
    try:
        with torch.no_grad():
            model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)
    finally:
        for a, o in zip(attns, origs, strict=True):
            a.forward = o
    return captured


def _generate_with_patch(mod, model, ids, captured, positions):
    attns = _attns(model)
    origs = [a.forward for a in attns]
    for a, o in zip(attns, origs, strict=True):
        a.forward = types.MethodType(mod.make_patch_wrapper(o, captured, positions, a.layer_idx), a)
    try:
        with torch.no_grad():
            return model.generate(ids, max_new_tokens=5, do_sample=False)
    finally:
        for a, o in zip(attns, origs, strict=True):
            a.forward = o


def test_self_patch_is_bit_identical():
    mod = _load_driver()
    model = _tiny_qwen2()
    ids = torch.tensor([[5, 9, 12, 4, 7, 3]])
    positions = [0, 1, 2]

    with torch.no_grad():
        baseline = model.generate(ids, max_new_tokens=5, do_sample=False)
    captured_self = _capture_base_prefix(mod, model, ids, positions)
    patched = _generate_with_patch(mod, model, ids, captured_self, positions)
    assert torch.equal(baseline, patched), (
        "a self-patch must be bit-identical (wrapper only touches named positions)"
    )


def test_different_prefix_patch_changes_generation():
    mod = _load_driver()
    trained = _tiny_qwen2()
    # A second model with different weights = a different prefix KV.
    base = _tiny_qwen2()
    for p in base.parameters():
        with torch.no_grad():
            p.add_(0.5 * torch.randn_like(p))

    ids = torch.tensor([[5, 9, 12, 4, 7, 3]])
    positions = [0, 1, 2]
    with torch.no_grad():
        unpatched = trained.generate(ids, max_new_tokens=5, do_sample=False)
    captured_base = _capture_base_prefix(mod, base, ids, positions)
    patched = _generate_with_patch(mod, trained, ids, captured_base, positions)
    assert not torch.equal(unpatched, patched), (
        "patching a DIFFERENT model's prefix KV must change generation (patch applied)"
    )


def test_patch_is_localized_to_named_positions():
    """Patching position set {0,1,2} must NOT alter the cache at positions {3,4,5}."""
    mod = _load_driver()
    from transformers.cache_utils import DynamicCache

    trained = _tiny_qwen2()
    base = _tiny_qwen2()
    for p in base.parameters():
        with torch.no_grad():
            p.add_(0.5 * torch.randn_like(p))

    ids = torch.tensor([[5, 9, 12, 4, 7, 3]])
    patch_positions = [0, 1, 2]

    # Trained-model unpatched cache (ground truth at all positions).
    cache_unpatched = DynamicCache()
    with torch.no_grad():
        trained(input_ids=ids, past_key_values=cache_unpatched, use_cache=True)

    captured_base = _capture_base_prefix(mod, base, ids, patch_positions)

    # Patched prefill cache.
    attns = _attns(trained)
    origs = [a.forward for a in attns]
    for a, o in zip(attns, origs, strict=True):
        a.forward = types.MethodType(
            mod.make_patch_wrapper(o, captured_base, patch_positions, a.layer_idx), a
        )
    cache_patched = DynamicCache()
    try:
        with torch.no_grad():
            trained(input_ids=ids, past_key_values=cache_patched, use_cache=True)
    finally:
        for a, o in zip(attns, origs, strict=True):
            a.forward = o

    untouched = [3, 4, 5]
    for layer in range(len(attns)):
        ku = cache_unpatched.layers[layer].keys
        kp = cache_patched.layers[layer].keys
        # Untouched positions are bit-identical to the unpatched trained cache.
        assert torch.equal(ku[:, :, untouched, :], kp[:, :, untouched, :]), (
            f"layer {layer}: positions outside the patch set were modified"
        )
        # Patched positions DIFFER (base KV substituted in).
        assert not torch.equal(ku[:, :, patch_positions, :], kp[:, :, patch_positions, :]), (
            f"layer {layer}: patch positions were not substituted"
        )
