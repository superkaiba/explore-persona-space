"""Independent local-rule and primary-reference sparse-pursuit checks."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from explore_persona_space.analysis.workspace_lenses import (
    dense_r_rules,
    nonnegative_gradient_pursuit,
    r_gelu,
    r_product,
    r_rms_norm,
    r_silu,
    rotated_dictionary,
    validate_jacobian_product,
)


def _reference_gp(x, dictionary, k):
    """Literal scalar NumPy oracle for GDM's published dense pseudocode."""
    a = np.zeros(len(dictionary))
    for _ in range(k):
        residual = x - a @ dictionary
        correlations = dictionary @ residual
        support = a != 0
        support[np.argmax(correlations)] = True
        gradient = support * correlations
        update = gradient @ dictionary
        denominator = update @ update
        if denominator == 0:
            continue
        a = np.maximum(a + (update @ residual / denominator) * gradient, 0)
    return a @ dictionary, a


@pytest.mark.parametrize("k", [5, 10, 25])
def test_pursuit_matches_published_algorithm(k):
    rng = np.random.default_rng(14)
    d = rng.normal(size=(31, 7))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    # Deliberate redundancy exercises reselecting and dropping active atoms.
    d[1] = d[0]
    x = rng.normal(size=(11, 7))
    actual = nonnegative_gradient_pursuit(torch.tensor(x), torch.tensor(d), k=k)
    for i in range(len(x)):
        expected, a = _reference_gp(x[i], d, k)
        np.testing.assert_allclose(actual.component[i], expected, atol=1e-11, rtol=1e-11)
        assert actual.active_atoms[i] == np.count_nonzero(a)
    torch.testing.assert_close(actual.component + actual.remainder, torch.tensor(x))
    assert (actual.coefficients >= 0).all()
    assert (actual.active_atoms <= k).all()


def test_pursuit_zero_and_nonpositive_correlations_visible():
    d = torch.eye(3, dtype=torch.float64)
    x = torch.tensor([[0.0, 0, 0], [-1.0, -2, -3]], dtype=torch.float64)
    result = nonnegative_gradient_pursuit(x, d, k=3)
    assert (result.component == 0).all()
    assert result.zero_update_steps[0] == 3
    assert (result.active_atoms == 0).all()
    torch.testing.assert_close(result.remainder, x)


def test_rotations_preserve_geometry_and_transform_pursuit():
    torch.manual_seed(27)
    d = F.normalize(torch.randn(29, 6, dtype=torch.float64), dim=1)
    x = torch.randn(8, 6, dtype=torch.float64)
    rotated, q = rotated_dictionary(d, seed=123)
    torch.testing.assert_close(rotated @ rotated.T, d @ d.T)
    direct = nonnegative_gradient_pursuit(x, rotated, k=10)
    transformed = nonnegative_gradient_pursuit(x @ q.T, d, k=10)
    torch.testing.assert_close(direct.component, transformed.component @ q)


@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
def test_relevance_activation_preserves_forward_and_local_rule(activation):
    x = torch.linspace(-3, 3, 17, dtype=torch.float64, requires_grad=True)
    if activation == "silu":
        y, original = r_silu(x), F.silu(x)
        scale = torch.sigmoid(x)
    else:
        approx = "tanh" if activation == "gelu_tanh" else "none"
        y, original = r_gelu(x, approximate=approx), F.gelu(x, approximate=approx)
        scale = torch.where(x != 0, original / x, 0.5)
    assert torch.equal(y, original)
    gradient = torch.autograd.grad(y.sum(), x)[0]
    torch.testing.assert_close(gradient, scale)
    ordinary = torch.autograd.grad(original.sum(), x)[0]
    assert not torch.allclose(gradient, ordinary)


def test_half_product_rule_and_broadcast():
    a = torch.tensor([[2.0], [-1.0]], requires_grad=True)
    b = torch.tensor([[3.0, 4.0]], requires_grad=True)
    y = r_product(a, b)
    assert torch.equal(y, a * b)
    ga, gb = torch.autograd.grad(y.sum(), (a, b))
    torch.testing.assert_close(ga, torch.full_like(a, 3.5))
    torch.testing.assert_close(gb, torch.full_like(b, 0.5))


def test_rms_local_rule_and_zero_centered_weight_offset():
    x = torch.tensor([[1.0, 2.0, -3.0]], dtype=torch.float64, requires_grad=True)
    gamma = 1 + torch.tensor([0.1, 0.2, 0.3], dtype=x.dtype)
    eps = 1e-5
    denominator = torch.sqrt(x.square().mean(-1, keepdim=True) + eps)
    original = x / denominator * gamma
    actual = r_rms_norm(
        x, gamma, eps=eps, original_output=original, accumulation_dtype=torch.float64
    )
    assert torch.equal(original, actual)
    gradient = torch.autograd.grad(actual.sum(), x)[0]
    torch.testing.assert_close(gradient, gamma / denominator)


@pytest.mark.parametrize("cast_before_weight", [True, False])
def test_rms_bfloat16_backward_matches_native_detach_expression(cast_before_weight):
    torch.manual_seed(999)
    x = torch.randn(3, 16, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(16, dtype=torch.bfloat16)
    gain = weight if cast_before_weight else 1.0 + weight.float()
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
    if cast_before_weight:
        original = (xf * inv_rms).to(x.dtype) * gain
        reference = (xf * inv_rms.detach()).to(x.dtype) * gain
    else:
        original = (xf * inv_rms * gain).to(x.dtype)
        reference = (xf * inv_rms.detach() * gain).to(x.dtype)
    actual = r_rms_norm(
        x, gain, eps=1e-6, original_output=original, cast_before_weight=cast_before_weight
    )
    assert torch.equal(actual, original)
    cotangent = torch.randn_like(actual)
    expected_grad = torch.autograd.grad(reference, x, cotangent, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, x, cotangent)[0]
    assert torch.equal(actual_grad, expected_grad)


def test_j_gate_validates_actual_derivative():
    x = torch.tensor([[0.2, -0.7]], dtype=torch.float64)
    direction = torch.tensor([[0.6, 0.1]], dtype=torch.float64)
    report = validate_jacobian_product(
        lambda t: F.silu(t) + t.square(), x, direction, torch.ones_like(x)
    )
    assert report["max_absolute_fd_error"] < 1e-8
    assert report["adjoint_absolute_error"] < 1e-12


def test_j_gate_rejects_r_backward_as_finite_difference_derivative():
    x = torch.tensor([[0.2, -0.7]], dtype=torch.float64)
    with pytest.raises(AssertionError):
        validate_jacobian_product(r_silu, x, torch.ones_like(x), torch.ones_like(x))


def test_invalid_dictionary_fails_loudly():
    with pytest.raises(AssertionError):
        nonnegative_gradient_pursuit(torch.ones(2, 3), 2 * torch.eye(3), k=2)
    with pytest.raises(ValueError, match="nonfinite"):
        nonnegative_gradient_pursuit(torch.full((2, 3), torch.nan), torch.eye(3), k=2)


def test_dense_adapter_real_qwen2_modules_forward_unchanged_and_restored():
    # Real library architecture, random tiny weights: integration evidence only,
    # explicitly not validation of a pretrained Qwen3.5 lens or checkpoint.
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(7)
    cfg = Qwen2Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(2, 9, 16, requires_grad=True)
    original = model(inputs_embeds=x, use_cache=False).logits
    ordinary_grad = torch.autograd.grad(original.square().sum(), x)[0]
    before = [(m, "forward" in m.__dict__) for m in model.modules()]
    with dense_r_rules(model.model.layers, final_norm=model.model.norm) as patched:
        assert len(patched) == 10
        modified = model(inputs_embeds=x, use_cache=False).logits
        assert torch.equal(original, modified)
        modified_grad = torch.autograd.grad(modified.square().sum(), x)[0]
        assert not torch.allclose(ordinary_grad, modified_grad)
    assert all(("forward" in m.__dict__) == had for m, had in before)
    restored = model(inputs_embeds=x, use_cache=False).logits
    assert torch.equal(original, restored)
    with pytest.raises(RuntimeError, match="test restoration"), dense_r_rules(model.model.layers):
        raise RuntimeError("test restoration")
    assert all(("forward" in m.__dict__) == had for m, had in before)
