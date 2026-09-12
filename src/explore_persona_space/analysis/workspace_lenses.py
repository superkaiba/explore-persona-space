"""Local R propagation rules and sparse workspace components.

These primitives do not certify a model-level R-lens. An architecture adapter
must identify every applicable residual norm, activation and MLP product, and
validate unchanged forwards before fitting a lens. Attention is unchanged in
the dense three-rule R-lens variant. Ordinary finite differences validate J,
whereas the R rules below are tested against their specified local maps.

Sources: https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/
and https://transformer-circuits.pub/2026/workspace/ .
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType

import torch
from torch import Tensor
from torch.nn import functional as F


class _ForwardWithLocalScale(torch.autograd.Function):
    """Keep an exact forward tensor; replace its input derivative by a scale."""

    @staticmethod
    def forward(ctx, x: Tensor, original: Tensor, scale: Tensor) -> Tensor:
        ctx.save_for_backward(scale)
        return original.clone()

    @staticmethod
    def backward(ctx, grad: Tensor):
        (scale,) = ctx.saved_tensors
        return grad * scale, None, None


def r_silu(x: Tensor) -> Tensor:
    """SiLU forward, identity relevance propagation (detach sigmoid factor)."""
    return _ForwardWithLocalScale.apply(x, F.silu(x), torch.sigmoid(x).detach())


def r_gelu(x: Tensor, *, approximate: str = "none") -> Tensor:
    """GELU forward with its multiplicative CDF/tanh factor detached."""
    if approximate == "none":
        scale = 0.5 * (1.0 + torch.erf(x / (2.0**0.5)))
    elif approximate == "tanh":
        scale = 0.5 * (1.0 + torch.tanh((2.0 / torch.pi) ** 0.5 * (x + 0.044715 * x**3)))
    else:
        raise ValueError(f"unsupported GELU approximation: {approximate}")
    return _ForwardWithLocalScale.apply(x, F.gelu(x, approximate=approximate), scale.detach())


class _HalfProduct(torch.autograd.Function):
    """Share product relevance equally between the two multiplicative inputs."""

    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad: Tensor):
        a, b = ctx.saved_tensors
        return (0.5 * grad * b).sum_to_size(a.shape), (0.5 * grad * a).sum_to_size(b.shape)


def r_product(a: Tensor, b: Tensor) -> Tensor:
    """Unchanged product forward with one-half of each ordinary input gradient."""
    return _HalfProduct.apply(a, b)


class _ForwardWithSurrogate(torch.autograd.Function):
    """Use original bits in forward and a native expression's backward graph."""

    @staticmethod
    def forward(ctx, original: Tensor, surrogate: Tensor) -> Tensor:
        return original.clone()

    @staticmethod
    def backward(ctx, grad: Tensor):
        return None, grad


def r_rms_norm(
    x: Tensor,
    effective_weight: Tensor,
    *,
    eps: float,
    original_output: Tensor,
    accumulation_dtype: torch.dtype = torch.float32,
    cast_before_weight: bool = False,
) -> Tensor:
    """Detach the RMS denominator, preserving the caller's exact norm forward.

    ``effective_weight`` includes any architecture offset, e.g. ``1 + weight``.
    Qwen2/3 cast the normalized value before multiplying by weight; Qwen3.5
    multiplies by the fp32 effective gain before casting. The backward follows
    these native expressions, preserving their low-precision rounding order.
    This is for frozen-model activation derivatives only.
    """
    if x.shape != original_output.shape or effective_weight.shape != x.shape[-1:]:
        raise ValueError("RMS norm output or effective weight shape mismatch")
    if eps <= 0 or not torch.isfinite(effective_weight).all():
        raise ValueError("positive eps and finite effective weights required")
    xf = x.to(accumulation_dtype)
    inverse_rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps).detach()
    normalized = xf * inverse_rms
    if cast_before_weight:
        surrogate = normalized.to(x.dtype) * effective_weight.detach()
    else:
        surrogate = (normalized * effective_weight.detach().to(accumulation_dtype)).to(x.dtype)
    return _ForwardWithSurrogate.apply(original_output, surrogate)


@contextmanager
def dense_r_rules(blocks, *, final_norm=None):
    """Install and restore the dense Qwen R-lens three-rule variant.

    Qwen3.5: only input/post-attention residual norms and MLP SiLU/products
    change. Q/k norms, gated linear-attention norms, delta recurrence and
    attention output gates retain their ordinary backward rules, matching
    the released dense R recipe. MoE and unknown classes fail before mutation.
    The Qwen3.5 norm offset is source-verified in transformers commit
    bd15bc95a89e728bbc1224084eb3b5829428c353; real-checkpoint parity is still
    required. The currently installed runtime may not provide this class.
    """
    norm_offsets = {"Qwen2RMSNorm": 0, "Qwen3RMSNorm": 0, "Qwen3_5RMSNorm": 1}
    mlp_classes = {"Qwen2MLP", "Qwen3MLP", "Qwen3_5MLP"}
    replacements = []

    def prepare_norm(norm, name):
        """Validate a norm architecture and prepare its exact-forward wrapper."""
        cls = type(norm).__name__
        if cls not in norm_offsets:
            raise ValueError(f"unsupported residual norm {name}: {cls}")
        original = norm.forward
        offset = norm_offsets[cls]
        eps = norm.variance_epsilon if hasattr(norm, "variance_epsilon") else norm.eps

        def forward(self, x):
            gain = 1.0 + self.weight.float() if offset else self.weight
            return r_rms_norm(
                x, gain, eps=eps, original_output=original(x), cast_before_weight=not bool(offset)
            )

        replacements.append((norm, name, forward))

    for i, block in enumerate(blocks):
        for attr in ("input_layernorm", "post_attention_layernorm"):
            if not hasattr(block, attr):
                raise ValueError(f"missing residual norm at block {i}: {attr}")
            prepare_norm(getattr(block, attr), f"layers.{i}.{attr}")
        mlp = block.mlp
        if type(mlp).__name__ not in mlp_classes:
            raise ValueError(f"unsupported MLP at block {i}: {type(mlp).__name__}")
        if type(mlp.act_fn).__name__ not in {"SiLU", "SiLUActivation"}:
            raise ValueError(f"expected SiLU at block {i}, got {type(mlp.act_fn).__name__}")
        if any(not hasattr(mlp, a) for a in ("gate_proj", "up_proj", "down_proj")):
            raise ValueError(f"incomplete gated MLP at block {i}")

        def mlp_forward(self, x):
            return self.down_proj(r_product(r_silu(self.gate_proj(x)), self.up_proj(x)))

        replacements.append((mlp, f"layers.{i}.mlp", mlp_forward))
    if final_norm is not None:
        prepare_norm(final_norm, "norm")
    previous = []
    try:
        for module, _name, forward in replacements:
            previous.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
            module.forward = MethodType(forward, module)
        yield [name for _, name, _ in replacements]
    finally:
        for module, had_override, old in reversed(previous):
            if had_override:
                module.forward = old
            else:
                del module.forward


def validate_jacobian_product(
    fn: Callable[[Tensor], Tensor],
    x: Tensor,
    direction: Tensor,
    cotangent: Tensor,
    *,
    epsilon: float = 1e-5,
    rtol: float = 1e-4,
    atol: float = 1e-7,
) -> dict[str, float]:
    """Validate an ordinary J product numerically and by the adjoint identity.

    Run on a differentiable fp64/precision-controlled downstream function.
    This gate must never be used as an R-coefficient fidelity requirement.
    """
    if x.shape != direction.shape or epsilon <= 0:
        raise ValueError("direction shape must match x and epsilon must be positive")
    leaf = x.detach().requires_grad_(True)
    y, jv = torch.autograd.functional.jvp(fn, leaf, direction)
    if y.shape != cotangent.shape:
        raise ValueError("cotangent shape must match output")
    fd = (fn(leaf + epsilon * direction) - fn(leaf - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close(jv, fd, rtol=rtol, atol=atol)
    jt_v = torch.autograd.grad(fn(leaf), leaf, cotangent)[0]
    left = (cotangent * jv).sum()
    right = (direction * jt_v).sum()
    torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
    return {
        "max_absolute_fd_error": float((jv - fd).abs().max().detach()),
        "adjoint_absolute_error": float((left - right).abs().detach()),
        "epsilon": epsilon,
    }


@dataclass(frozen=True)
class SparseComponent:
    """Per-token nonnegative sparse reconstruction and explicit diagnostics."""

    component: Tensor
    remainder: Tensor
    indices: Tensor
    coefficients: Tensor
    active_atoms: Tensor
    squared_error: Tensor
    input_squared_norm: Tensor
    zero_update_steps: Tensor
    increasing_error_steps: Tensor


@torch.no_grad()
def nonnegative_gradient_pursuit(
    activations: Tensor,
    dictionary: Tensor,
    *,
    k: int,
) -> SparseComponent:
    """Batched primary-reference nonnegative gradient pursuit for k steps.

    Each step selects the atom with largest signed residual correlation, takes
    the exact line-search step along the gradient restricted to the active
    support, then clamps coefficients to be nonnegative. This is approximate
    sparse coding, not a global projector or a claim of exact NNLS optimality.
    Selection MAY revisit an active atom, as in the GDM reference algorithm:
    https://www.lesswrong.com/s/AtTZjoDm8q3DbDT8Z/p/C5KAZQib3bzzpeyrg .
    k is the iteration budget and upper bound on L0; report realized L0.
    Ties select the smallest dictionary row index. Unit atoms are required.
    Caller chunks token rows and dictionaries remain shared across chunks.
    """
    if activations.ndim != 2 or dictionary.ndim != 2:
        raise ValueError("activations and dictionary must both be matrices")
    if activations.shape[1] != dictionary.shape[1] or activations.device != dictionary.device:
        raise ValueError("activation/dictionary dimensions or devices differ")
    if k < 1 or k > dictionary.shape[0] or activations.shape[0] < 1:
        raise ValueError("require nonempty activations and 1 <= k <= number of atoms")
    if not torch.isfinite(activations).all() or not torch.isfinite(dictionary).all():
        raise ValueError("nonfinite activation/dictionary values")
    dtype = torch.float64 if activations.dtype == torch.float64 else torch.float32
    x, d = activations.to(dtype), dictionary.to(dtype)
    torch.testing.assert_close(
        d.norm(dim=1), torch.ones(d.shape[0], device=d.device, dtype=dtype), rtol=2e-4, atol=2e-4
    )
    n = len(x)
    indices = torch.full((n, k), -1, device=x.device, dtype=torch.long)
    coeff = torch.zeros((n, k), device=x.device, dtype=dtype)
    atoms = torch.zeros((n, k, x.shape[1]), device=x.device, dtype=dtype)
    estimate = torch.zeros_like(x)
    zero_updates = torch.zeros(n, device=x.device, dtype=torch.long)
    increasing = torch.zeros(n, device=x.device, dtype=torch.long)
    for step in range(k):
        residual = x - estimate
        corr = residual @ d.T
        index = corr.argmax(dim=1)
        add = ~(indices == index[:, None]).any(dim=1)
        indices[add, step] = index[add]
        atoms[add, step] = d[index[add]]
        selected = atoms[:, : step + 1]
        gradient = torch.einsum("bkh,bh->bk", selected, residual)
        active = (coeff[:, : step + 1] != 0) | (indices[:, : step + 1] == index[:, None])
        gradient = gradient * active
        update = torch.einsum("bk,bkh->bh", gradient, selected)
        denom = update.square().sum(1)
        eta = torch.zeros_like(denom)
        live = denom > 0
        zero_updates += ~live
        eta[live] = (residual[live] * update[live]).sum(1) / denom[live]
        proposed_coeff = (coeff[:, : step + 1] + eta[:, None] * gradient).clamp_min(0)
        proposed = torch.einsum("bk,bkh->bh", proposed_coeff, selected)
        # Keep this diagnostic visible; do not alter the reference algorithm.
        old_error = residual.square().sum(1)
        new_error = (x - proposed).square().sum(1)
        tolerance = 2e-5 * x.square().sum(1) + 1e-12
        increasing += new_error > old_error + tolerance
        coeff[:, : step + 1] = proposed_coeff
        estimate = proposed
    rest = x - estimate
    return SparseComponent(
        estimate,
        rest,
        indices,
        coeff,
        (coeff > 0).sum(1),
        rest.square().sum(1),
        x.square().sum(1),
        zero_updates,
        increasing,
    )


def rotated_dictionary(dictionary: Tensor, *, seed: int) -> tuple[Tensor, Tensor]:
    """Haar orthogonal rotation preserving dictionary norms and pairwise Gram."""
    if dictionary.ndim != 2 or not torch.isfinite(dictionary).all():
        raise ValueError("finite matrix dictionary required")
    generator = torch.Generator(device=dictionary.device).manual_seed(seed)
    gaussian = torch.randn(
        dictionary.shape[1],
        dictionary.shape[1],
        generator=generator,
        device=dictionary.device,
        dtype=dictionary.dtype,
    )
    q, r = torch.linalg.qr(gaussian)
    signs = torch.sign(torch.diagonal(r))
    if torch.any(signs == 0):
        raise RuntimeError("degenerate random orthogonal draw")
    q = q * signs
    return dictionary @ q, q
