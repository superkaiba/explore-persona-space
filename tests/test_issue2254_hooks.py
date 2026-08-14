"""CPU unit tests for issue #2254 hook primitives (pure torch, no HF model).

``MultiLayerDeltaHook`` / ``ProjectionPatchHook`` / ``MultiLayerProjectionPatchHook``
are exercised on a TINY synthetic decoder that duck-types the standard HF
layout (``model.model.layers`` — ``_resolve_decoder_blocks`` is purely
structural, so no transformers import and no network). Blocks are identity
maps returning HF-style ``(hidden,)`` tuples, so a captured block output
isolates exactly the hooks' edits; capture hooks are registered AFTER the edit
hooks (nn.Module runs forward hooks in registration order, each seeing the
previous hook's modified result — the ``test_issue2094_hooks.py`` pattern).

Position semantics under test mirror ``generate_batch``: LEFT-padded prompts
put every row's last real context token at the shared padded position
``T - 1``, and ``arm(expected_prompt_len=T)`` pins the prefill shape exactly.
"""

from __future__ import annotations

import pytest
import torch

from explore_persona_space.experiments.issue1415.steering import DeltaHook
from explore_persona_space.experiments.issue2254.hooks import (
    MultiLayerDeltaHook,
    MultiLayerProjectionPatchHook,
    ProjectionPatchHook,
    multi_layer_delta_hooks,
    multi_layer_projection_patch_hooks,
)

H = 16
N_LAYERS = 3


class _Block(torch.nn.Module):
    """Identity decoder block returning an HF-style ``(hidden,)`` tuple."""

    def forward(self, hidden: torch.Tensor):
        return (hidden,)


class _BareBlock(torch.nn.Module):
    """Identity block returning a BARE tensor (the non-tuple hook branch)."""

    def forward(self, hidden: torch.Tensor):
        return hidden


class _Decoder(torch.nn.Module):
    """Chain of blocks exposing ``.layers`` (the HF ``model.model`` shape)."""

    def __init__(self, blocks: list[torch.nn.Module]):
        super().__init__()
        self.layers = torch.nn.ModuleList(blocks)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            out = blk(hidden)
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden


class _TinyModel(torch.nn.Module):
    """Duck-types ``model.model.layers`` for ``_resolve_decoder_blocks`` (depth 1)."""

    def __init__(self, blocks: list[torch.nn.Module] | None = None):
        super().__init__()
        self.model = _Decoder(blocks if blocks is not None else [_Block() for _ in range(N_LAYERS)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model(hidden)


@pytest.fixture()
def tiny_model() -> _TinyModel:
    return _TinyModel()


def _capture_forward(model: _TinyModel, hidden: torch.Tensor) -> dict[int, torch.Tensor]:
    """One forward, capturing each block's (post-edit-hook) output hidden state."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _mk(layer_idx: int):
        def cap(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = h.detach().clone()

        return cap

    for layer_idx, blk in enumerate(model.model.layers):
        handles.append(blk.register_forward_hook(_mk(layer_idx)))
    try:
        model(hidden)
    finally:
        for hnd in handles:
            hnd.remove()
    return captured


def _unit(i: int, h: int = H) -> torch.Tensor:
    e = torch.zeros(h)
    e[i] = 1.0
    return e


# ── MultiLayerDeltaHook ───────────────────────────────────────────────


def test_multilayer_delta_stack_lifecycle_and_handle(tiny_model):
    stack = multi_layer_delta_hooks(
        tiny_model, [0, 1], [_unit(0), _unit(1)], [1.0, 2.0], all_positions=False
    )
    assert isinstance(stack, MultiLayerDeltaHook)
    # generate_batch's precondition: `assert hook._handle is not None` after install.
    assert stack._handle is None
    with stack as installed:
        assert installed is stack
        assert stack._handle is not None
        assert all(h._handle is not None for h in stack.hooks)
        # Double-install must fail loud (child DeltaHook asserts).
        with pytest.raises(AssertionError):
            stack.install()
    assert stack._handle is None
    assert all(h._handle is None for h in stack.hooks)
    # Partially-installed stack reads NOT installed.
    stack.hooks[0].install()
    assert stack._handle is None
    stack.hooks[0].remove()


@pytest.mark.parametrize("all_positions", [False, True])
def test_multilayer_delta_each_layer_own_vector_and_dose(tiny_model, all_positions):
    """Both §4.2 position arms through a real forward: context (prefill-only,
    all_positions=False) AND answer-tokens (all_positions=True — the round-1
    g1 concern: the production answer arm runs exactly this stack config)."""
    B, T = 2, 5
    layers = [0, 1, 2]
    deltas = [_unit(0), _unit(1), _unit(2)]
    alphas = [2.0, 0.5, 1.0]
    stack = multi_layer_delta_hooks(tiny_model, layers, deltas, alphas, all_positions=all_positions)
    hidden = torch.zeros(B, T, H)
    with stack:
        stack.arm(expected_prompt_len=T)
        captured = _capture_forward(tiny_model, hidden)
        # decode-step forward (T=1 under the KV cache)
        decode = _capture_forward(tiny_model, torch.zeros(B, 1, H))
    # Identity blocks => block k's output at T-1 is the CUMULATIVE per-layer
    # edit sum alpha_0*d_0 + ... + alpha_k*d_k; every other PROMPT position
    # stays 0 in BOTH modes (all_positions edits only the generating position
    # at prefill).
    expected = torch.zeros(H)
    for k in range(N_LAYERS):
        expected = expected + alphas[k] * deltas[k]
        for b in range(B):
            torch.testing.assert_close(captured[k][b, T - 1], expected)
        assert torch.all(captured[k][:, : T - 1, :] == 0)
        if all_positions:
            # answer-tokens arm: each decode step's (single) generated
            # position carries the same cumulative per-layer edit
            for b in range(B):
                torch.testing.assert_close(decode[k][b, 0], expected)
        else:
            assert torch.all(decode[k] == 0)  # context arm: decode untouched
    # n_edits counts edited forwards per child: prefill only (3) vs
    # prefill + decode (6)
    assert stack.n_edits == (6 if all_positions else 3)


def test_multilayer_delta_arm_resets_prefill_latch(tiny_model):
    T = 4
    stack = multi_layer_delta_hooks(tiny_model, [0, 1], [_unit(0), _unit(1)], [1.0, 1.0])
    hidden = torch.zeros(1, T, H)
    with stack:
        stack.arm(expected_prompt_len=T)
        first = _capture_forward(tiny_model, hidden)
        assert first[1][0, T - 1, 0] == 1.0 and first[1][0, T - 1, 1] == 1.0
        # Decode-step forward (T=1, prefill latched): untouched at every layer.
        decode = _capture_forward(tiny_model, torch.zeros(1, 1, H))
        assert all(torch.all(decode[k] == 0) for k in range(N_LAYERS))
        # Re-arm (generate_batch's per-draw loop) => edits fire again.
        stack.arm(expected_prompt_len=T)
        second = _capture_forward(tiny_model, hidden)
        torch.testing.assert_close(second[1][0, T - 1], first[1][0, T - 1])
    assert stack.n_edits == 4  # 2 hooks x 2 armed prefills


def test_multilayer_delta_rejects_bad_stacks(tiny_model):
    with pytest.raises(AssertionError):  # empty
        MultiLayerDeltaHook([])
    with pytest.raises(AssertionError):  # duplicate layers
        MultiLayerDeltaHook(
            [DeltaHook(tiny_model, 0, _unit(0), 1.0), DeltaHook(tiny_model, 0, _unit(1), 1.0)]
        )
    with pytest.raises(AssertionError):  # mixed position modes
        MultiLayerDeltaHook(
            [
                DeltaHook(tiny_model, 0, _unit(0), 1.0, all_positions=False),
                DeltaHook(tiny_model, 1, _unit(1), 1.0, all_positions=True),
            ]
        )
    with pytest.raises(AssertionError):  # single-hook-only mode in a stack
        MultiLayerDeltaHook([DeltaHook(tiny_model, 0, _unit(0), 1.0, edit_position=2)])
    with pytest.raises(AssertionError):  # length mismatch in the factory
        multi_layer_delta_hooks(tiny_model, [0, 1], [_unit(0)], [1.0, 1.0])


def test_multilayer_delta_wrong_prompt_len_raises(tiny_model):
    T = 4
    stack = multi_layer_delta_hooks(tiny_model, [0], [_unit(0)], [1.0])
    with stack:
        stack.arm(expected_prompt_len=T + 3)
        with pytest.raises(AssertionError):
            tiny_model(torch.zeros(1, T, H))


# ── ProjectionPatchHook ───────────────────────────────────────────────


def test_projection_patch_moves_projection_to_target_exactly(tiny_model):
    B, T = 3, 6
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(B, T, H, generator=g)
    d_hat = torch.randn(H, generator=g)
    d_hat = d_hat / d_hat.norm()
    target = 4.25
    hook = ProjectionPatchHook(tiny_model, 1, d_hat, target)
    with hook:
        hook.arm(expected_prompt_len=T)
        captured = _capture_forward(tiny_model, hidden)
    # Layer 0 (no hook, identity): input passes through unchanged.
    torch.testing.assert_close(captured[0], hidden)
    out = captured[1]
    # Projection at the last context token == target, per row.
    torch.testing.assert_close(out[:, T - 1, :] @ d_hat, torch.full((B,), target))

    # Orthogonal complement at T-1 untouched.
    def _orth(x):
        return x - (x @ d_hat)[:, None] * d_hat

    torch.testing.assert_close(_orth(out[:, T - 1, :]), _orth(hidden[:, T - 1, :]))
    # Every other position untouched (rows of differing real length under left
    # padding all share the edited absolute position T-1).
    torch.testing.assert_close(out[:, : T - 1, :], hidden[:, : T - 1, :])
    # Downstream identity block sees the edited state.
    torch.testing.assert_close(captured[2], out)
    assert hook.n_edits == 1


def test_projection_patch_per_row_target(tiny_model):
    B, T = 4, 3
    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(B, T, H, generator=g)
    d_hat = _unit(5)
    targets = torch.tensor([-2.0, 0.0, 1.5, 7.0])
    hook = ProjectionPatchHook(tiny_model, 0, d_hat, targets)
    with hook:
        hook.arm(expected_prompt_len=T)
        captured = _capture_forward(tiny_model, hidden)
    out = captured[0]
    torch.testing.assert_close(out[:, T - 1, :] @ d_hat, targets)
    # With a one-hot d_hat only coordinate 5 moves; all others identical.
    keep = [i for i in range(H) if i != 5]
    torch.testing.assert_close(out[:, T - 1, keep], hidden[:, T - 1, keep])
    torch.testing.assert_close(out[:, : T - 1, :], hidden[:, : T - 1, :])


def test_projection_patch_prefill_only_and_rearm(tiny_model):
    T = 4
    g = torch.Generator().manual_seed(2)
    hidden = torch.randn(1, T, H, generator=g)
    d_hat = _unit(0)
    hook = ProjectionPatchHook(tiny_model, 0, d_hat, 3.0)
    with hook:
        hook.arm(expected_prompt_len=T)
        first = _capture_forward(tiny_model, hidden)
        assert float(first[0][0, T - 1] @ d_hat) == pytest.approx(3.0)
        # Decode step (T=1, latched): untouched.
        step = torch.randn(1, 1, H, generator=g)
        decode = _capture_forward(tiny_model, step)
        torch.testing.assert_close(decode[0], step)
        # Re-arm => edits the next prefill again.
        hook.arm(expected_prompt_len=T)
        second = _capture_forward(tiny_model, hidden)
        torch.testing.assert_close(second[0], first[0])
    assert hook.n_edits == 2


def test_projection_patch_fail_fast_contracts(tiny_model):
    d_hat = _unit(0)
    # Non-unit direction refused at construction.
    with pytest.raises(AssertionError):
        ProjectionPatchHook(tiny_model, 0, 3.0 * d_hat, 1.0)
    # Non-vector direction refused.
    with pytest.raises(AssertionError):
        ProjectionPatchHook(tiny_model, 0, torch.eye(H) / H**0.5, 1.0)
    # Out-of-range layer refused.
    with pytest.raises(AssertionError):
        ProjectionPatchHook(tiny_model, N_LAYERS, d_hat, 1.0)
    # Un-armed prefill fails loud.
    hook = ProjectionPatchHook(tiny_model, 0, d_hat, 1.0)
    with hook, pytest.raises(AssertionError):
        tiny_model(torch.zeros(1, 4, H))
    # Wrong expected_prompt_len fails loud at edit time.
    with hook:
        hook.arm(expected_prompt_len=9)
        with pytest.raises(AssertionError):
            tiny_model(torch.zeros(1, 4, H))
    # Wrong-length per-row target fails loud at edit time.
    bad = ProjectionPatchHook(tiny_model, 0, d_hat, torch.zeros(3))
    with bad:
        bad.arm(expected_prompt_len=4)
        with pytest.raises(AssertionError):
            tiny_model(torch.zeros(2, 4, H))


def test_projection_patch_dtype_follows_activation_and_bare_tensor_branch():
    # Bare-tensor blocks exercise the non-tuple _hook branch; bf16 hidden
    # exercises dtype-follow (edit computed in the activation's dtype).
    model = _TinyModel(blocks=[_BareBlock() for _ in range(2)])
    B, T = 2, 3
    g = torch.Generator().manual_seed(3)
    hidden = torch.randn(B, T, H, generator=g).to(torch.bfloat16)
    d_hat = torch.randn(H, generator=g)
    d_hat = d_hat / d_hat.norm()
    hook = ProjectionPatchHook(model, 1, d_hat, 2.0)
    with hook:
        hook.arm(expected_prompt_len=T)
        captured = _capture_forward(model, hidden)
    out = captured[1]
    assert out.dtype == torch.bfloat16
    proj = out[:, T - 1, :].float() @ d_hat
    torch.testing.assert_close(proj, torch.full((B,), 2.0), atol=0.05, rtol=0.02)


# ── MultiLayerProjectionPatchHook ─────────────────────────────────────


def test_multilayer_projection_patch_per_layer_dhat_and_target(tiny_model):
    B, T = 2, 5
    g = torch.Generator().manual_seed(4)
    hidden = torch.randn(B, T, H, generator=g)
    d0, d1 = _unit(0), _unit(1)
    stack = multi_layer_projection_patch_hooks(tiny_model, [0, 2], [d0, d1], [5.0, -1.0])
    assert isinstance(stack, MultiLayerProjectionPatchHook)
    with stack:
        assert stack._handle is not None
        stack.arm(expected_prompt_len=T)
        captured = _capture_forward(tiny_model, hidden)
    # Layer 0's hook moves <h, d0> to 5.0 (identity block 1 passes it through).
    torch.testing.assert_close(captured[0][:, T - 1, :] @ d0, torch.full((B,), 5.0))
    torch.testing.assert_close(captured[1], captured[0])
    # Layer 2's hook moves <h, d1> to -1.0 on top; the layer-0 patch survives
    # (d0 ⊥ d1, so the second edit leaves coordinate 0 alone).
    out = captured[2]
    torch.testing.assert_close(out[:, T - 1, :] @ d1, torch.full((B,), -1.0))
    torch.testing.assert_close(out[:, T - 1, :] @ d0, torch.full((B,), 5.0))
    torch.testing.assert_close(out[:, : T - 1, :], hidden[:, : T - 1, :])
    assert stack.n_edits == 2


def test_multilayer_projection_patch_rejects_bad_stacks(tiny_model):
    d = _unit(0)
    with pytest.raises(AssertionError):  # empty
        MultiLayerProjectionPatchHook([])
    with pytest.raises(AssertionError):  # duplicate layers
        MultiLayerProjectionPatchHook(
            [
                ProjectionPatchHook(tiny_model, 1, d, 1.0),
                ProjectionPatchHook(tiny_model, 1, _unit(1), 2.0),
            ]
        )
    with pytest.raises(AssertionError):  # factory length mismatch
        multi_layer_projection_patch_hooks(tiny_model, [0, 1], [d], [1.0, 2.0])
    with pytest.raises(AssertionError):  # wrong child type
        MultiLayerProjectionPatchHook([DeltaHook(tiny_model, 0, d, 1.0)])
