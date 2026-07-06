"""CPU unit tests for artifacts/ablation.py (task #863, Phase 0f).

Tiny real-architecture Qwen2 model built from config (no download, no network —
the ``tests/test_js_canonical.py`` fixture precedent), exercising the PRIMARY
hook path of ``extract_layer_activations`` (real ``model.model.layers``).
"""

from __future__ import annotations

import pytest
import torch

from explore_persona_space.analysis.extraction import EMBED_LAYER, extract_layer_activations
from explore_persona_space.artifacts.ablation import (
    _make_projection_hook,
    _normalize,
    _project_out,
    ablated,
    ablation_hooks,
    all_layer_directions,
    single_layer_directions,
)

N_LAYERS = 2
HIDDEN = 16


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def _ids() -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)


def test_zero_direction_is_noop_generate(tiny_model):
    """Required test (a): zero-direction ablation == base generation, EXACT token ids."""
    ids = _ids()
    with torch.no_grad():
        base = tiny_model.generate(ids, do_sample=False, max_new_tokens=8, pad_token_id=0)
    directions = all_layer_directions(torch.zeros(N_LAYERS, HIDDEN), list(range(N_LAYERS)))
    with ablated(tiny_model, directions), torch.no_grad():
        abl = tiny_model.generate(ids, do_sample=False, max_new_tokens=8, pad_token_id=0)
    assert torch.equal(base, abl)


def test_projection_removes_component(tiny_model):
    """Ablation at layer 1 zeroes the direction component; pins hook-replace ordering.

    The capture hooks (``extract_layer_activations``) register AFTER the ablation
    hooks inside the ``with`` block, so they must observe the REPLACED (projected)
    output — the PyTorch replace-and-propagate semantics the design relies on.
    """
    torch.manual_seed(1)
    direction = torch.randn(HIDDEN)
    v_hat = _normalize(direction)
    ids = _ids()

    captured = extract_layer_activations(tiny_model, ids, [1])
    proj_before = (captured[1].float() @ v_hat).abs().max()
    assert proj_before > 1e-3  # the component is non-trivially present pre-ablation

    with ablated(tiny_model, single_layer_directions(1, direction)):
        captured_abl = extract_layer_activations(tiny_model, ids, [1])
    proj_after = (captured_abl[1].float() @ v_hat).abs().max()
    assert proj_after < 1e-5


def test_all_layer_orthogonality(tiny_model):
    """All-layer ablation: each layer's output orthogonal to ITS OWN direction row."""
    torch.manual_seed(2)
    r_b = torch.randn(N_LAYERS, HIDDEN)
    layers = list(range(N_LAYERS))
    ids = _ids()
    with ablated(tiny_model, all_layer_directions(r_b, layers)):
        captured = extract_layer_activations(tiny_model, ids, layers)
    for i, layer in enumerate(layers):
        v_hat = _normalize(r_b[i])
        proj = (captured[layer].float() @ v_hat).abs().max()
        assert proj < 1e-5, (layer, float(proj))


def test_embed_layer_ablation(tiny_model):
    """EMBED_LAYER direction: the captured embedding output is orthogonal to it."""
    torch.manual_seed(3)
    direction = torch.randn(HIDDEN)
    v_hat = _normalize(direction)
    ids = _ids()
    with ablated(tiny_model, {EMBED_LAYER: direction}):
        captured = extract_layer_activations(tiny_model, ids, [EMBED_LAYER])
    hs = captured[EMBED_LAYER].float()
    assert hs.abs().max() > 0  # non-degenerate capture
    assert (hs @ v_hat).abs().max() < 1e-5


def test_tuple_and_bare_output():
    """The hook handles a ``(tensor, extra)`` tuple AND a bare tensor; math correct."""
    torch.manual_seed(4)
    v_hat = _normalize(torch.randn(HIDDEN))
    t = torch.randn(2, 3, HIDDEN)
    extra = object()
    hook = _make_projection_hook(v_hat)

    out_tuple = hook(None, None, (t, extra))
    assert isinstance(out_tuple, tuple) and len(out_tuple) == 2
    assert out_tuple[1] is extra
    assert out_tuple[0].shape == t.shape
    assert torch.allclose(out_tuple[0], _project_out(t, v_hat))
    assert (out_tuple[0] @ v_hat).abs().max() < 1e-5

    out_bare = hook(None, None, t)
    assert isinstance(out_bare, torch.Tensor)
    assert out_bare.shape == t.shape
    assert torch.allclose(out_bare, _project_out(t, v_hat))


def test_hooks_removed_on_exit(tiny_model):
    """After the ``with`` block the layer-1 capture equals the un-ablated value again."""
    torch.manual_seed(5)
    direction = torch.randn(HIDDEN)
    ids = _ids()
    before = extract_layer_activations(tiny_model, ids, [1])[1]
    with ablated(tiny_model, single_layer_directions(1, direction)):
        during = extract_layer_activations(tiny_model, ids, [1])[1]
    after = extract_layer_activations(tiny_model, ids, [1])[1]
    assert not torch.equal(before, during)  # ablation actually engaged inside the block
    assert torch.equal(before, after)  # and is fully removed on exit


def test_all_layer_directions_shape_guard():
    """(2, 16) r_b with a 1-entry layer list must fail loud."""
    with pytest.raises((AssertionError, ValueError)):
        all_layer_directions(torch.zeros(N_LAYERS, HIDDEN), [0])


def test_out_of_range_layer_leaves_model_clean(tiny_model):
    """An out-of-range entry fails loud BEFORE any hook registers; model stays clean.

    Regression test for the r1 `ablation-hook-registration-rollback` concern: the
    valid layer-0 entry must not stay silently ablated when the 999 entry raises.
    Negative indices other than EMBED_LAYER are rejected too (no Python aliasing).
    """
    torch.manual_seed(6)
    direction = torch.randn(HIDDEN)
    ids = _ids()
    before = extract_layer_activations(tiny_model, ids, [0])[0]
    with (
        pytest.raises(ValueError, match="out of range"),
        ablated(tiny_model, {0: direction, 999: direction}),
    ):
        pass  # pragma: no cover — registration must raise before the body runs
    with pytest.raises(ValueError, match="out of range"):
        ablation_hooks(tiny_model, {0: direction, -2: direction})
    assert not tiny_model.model.layers[0]._forward_hooks
    after = extract_layer_activations(tiny_model, ids, [0])[0]
    assert torch.equal(before, after)


def test_partial_registration_rolls_back(tiny_model):
    """A malformed LATER entry (2-D direction) rolls back the already-registered hook.

    Layers pre-validate, so this exercises the try/except rollback itself: the
    layer-0 hook registers, then `_normalize` raises on the 2-D layer-1 direction —
    all accumulated handles must be removed before the exception propagates.
    """
    torch.manual_seed(7)
    good = torch.randn(HIDDEN)
    bad = torch.randn(2, HIDDEN)  # _normalize asserts ndim == 1
    ids = _ids()
    before = extract_layer_activations(tiny_model, ids, [0])[0]
    with pytest.raises(AssertionError):
        ablation_hooks(tiny_model, {0: good, 1: bad})
    assert not tiny_model.model.layers[0]._forward_hooks
    assert not tiny_model.model.layers[1]._forward_hooks
    after = extract_layer_activations(tiny_model, ids, [0])[0]
    assert torch.equal(before, after)
