"""CPU unit tests for issue #2094 ``PositionEditHook`` / ``PositionEditHookStack``.

Tiny from-config 2-layer Qwen2 model (real architecture, random weights, no
network) — the tiny-real standard. Left-padded batches with differing prompt
lengths mirror the ``generate_batch`` geometry: the edit must land at each
row's intended ABSOLUTE padded position and nowhere else, asserted via
captured block outputs.
"""

from __future__ import annotations

import pytest
import torch

from explore_persona_space.analysis.extraction import _resolve_decoder_blocks
from explore_persona_space.experiments.issue2094.hooks import (
    PositionEditHook,
    PositionEditHookStack,
    joint_hooks,
)

H = 64
VOCAB = 128


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        hidden_size=H,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _left_pad_batch(row_lengths: list[int], seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Random token rows of the given lengths, LEFT-padded to a common T."""
    g = torch.Generator().manual_seed(seed)
    T = max(row_lengths)
    input_ids = torch.zeros((len(row_lengths), T), dtype=torch.long)
    mask = torch.zeros((len(row_lengths), T), dtype=torch.long)
    for b, rl in enumerate(row_lengths):
        ids = torch.randint(1, VOCAB, (rl,), generator=g)
        input_ids[b, T - rl :] = ids
        mask[b, T - rl :] = 1
    return input_ids, mask


def _forward_capture(model, input_ids, mask, layers) -> dict[int, torch.Tensor]:
    """Run one forward, capturing each requested block's OUTPUT hidden state.

    Capture hooks are registered AFTER any already-installed edit hook, so they
    see the edited output (nn.Module runs forward hooks in registration order,
    each receiving the previous hook's modified result).
    """
    blocks, _, _ = _resolve_decoder_blocks(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _mk(layer_idx):
        def cap(_m, _i, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = hidden.detach().clone()

        return cap

    for layer_idx in layers:
        handles.append(blocks[layer_idx].register_forward_hook(_mk(layer_idx)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=mask)
    finally:
        for h in handles:
            h.remove()
    return captured


def _deltas(pos_counts: list[int], seed: int = 7) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, H, generator=g) for n in pos_counts]


ROW_LENGTHS = [5, 8, 3]


def _padded(row_len: int, pos: int, T: int) -> int:
    return pos + (T - row_len)


# ── add mode: per-row single positions under left padding ─────────────


def test_add_single_position_lands_per_row_and_nowhere_else(tiny_model):
    input_ids, mask = _left_pad_batch(ROW_LENGTHS)
    T = input_ids.shape[1]
    positions = [[4], [2], [0]]  # unpadded per-row coords
    deltas = _deltas([1, 1, 1])
    alpha = 2.0

    base = _forward_capture(tiny_model, input_ids, mask, [0, 1])
    hook = PositionEditHook(tiny_model, layer=0)
    hook.arm_batch(ROW_LENGTHS, positions, deltas, mode="add", alpha=alpha)
    with hook:
        hook.arm(expected_prompt_len=T)
        edited = _forward_capture(tiny_model, input_ids, mask, [0, 1])

    diff0 = edited[0] - base[0]
    touched = torch.zeros(diff0.shape[:2], dtype=torch.bool)
    for b, (rl, pos, d) in enumerate(zip(ROW_LENGTHS, positions, deltas, strict=True)):
        pp = _padded(rl, pos[0], T)
        touched[b, pp] = True
        assert torch.allclose(diff0[b, pp], alpha * d[0], atol=1e-5), (b, pp)
    assert torch.equal(diff0[~touched], torch.zeros_like(diff0[~touched]))
    assert hook.n_edits == 1

    # Causal propagation: at layer 1, REAL positions strictly BEFORE each row's
    # edited position are bit-identical (pads are excluded: a fully-masked pad
    # query softmaxes to uniform attention over ALL keys, edited one included,
    # so left-pad positions legitimately change — they are attention-masked
    # away from every real position and never affect the science reads).
    diff1 = edited[1] - base[1]
    for b, (rl, pos) in enumerate(zip(ROW_LENGTHS, positions, strict=True)):
        pp = _padded(rl, pos[0], T)
        first_real = T - rl
        if pp > first_real:
            span = diff1[b, first_real:pp]
            assert torch.equal(span, torch.zeros_like(span)), b
        assert diff1[b, pp].abs().max() > 0, b


def test_add_multi_position_last3_joint(tiny_model):
    input_ids, mask = _left_pad_batch(ROW_LENGTHS)
    T = input_ids.shape[1]
    # last-3 joint slot: positions rl-3..rl-1 per row (row 2 has rl=3 -> 0..2)
    positions = [[rl - 3, rl - 2, rl - 1] for rl in ROW_LENGTHS]
    deltas = _deltas([3, 3, 3])

    base = _forward_capture(tiny_model, input_ids, mask, [0])
    hook = PositionEditHook(tiny_model, layer=0)
    hook.arm_batch(ROW_LENGTHS, positions, deltas, mode="add", alpha=1.0)
    with hook:
        hook.arm(expected_prompt_len=T)
        edited = _forward_capture(tiny_model, input_ids, mask, [0])

    diff0 = edited[0] - base[0]
    touched = torch.zeros(diff0.shape[:2], dtype=torch.bool)
    for b, (rl, pos, d) in enumerate(zip(ROW_LENGTHS, positions, deltas, strict=True)):
        for j, p in enumerate(pos):
            pp = _padded(rl, p, T)
            touched[b, pp] = True
            assert torch.allclose(diff0[b, pp], d[j], atol=1e-5), (b, p, pp)
    assert torch.equal(diff0[~touched], torch.zeros_like(diff0[~touched]))


def test_add_ragged_position_counts_per_row(tiny_model):
    """Query-span-style slots: rows may carry DIFFERENT position counts."""
    input_ids, mask = _left_pad_batch(ROW_LENGTHS)
    T = input_ids.shape[1]
    positions = [[1, 3], [0, 2, 5, 7], [2]]
    deltas = _deltas([2, 4, 1])

    base = _forward_capture(tiny_model, input_ids, mask, [0])
    hook = PositionEditHook(tiny_model, layer=0)
    hook.arm_batch(ROW_LENGTHS, positions, deltas, mode="add")
    with hook:
        hook.arm(expected_prompt_len=T)
        edited = _forward_capture(tiny_model, input_ids, mask, [0])

    diff0 = edited[0] - base[0]
    touched = torch.zeros(diff0.shape[:2], dtype=torch.bool)
    for b, (rl, pos, d) in enumerate(zip(ROW_LENGTHS, positions, deltas, strict=True)):
        for j, p in enumerate(pos):
            pp = _padded(rl, p, T)
            touched[b, pp] = True
            assert torch.allclose(diff0[b, pp], d[j], atol=1e-5), (b, p)
    assert torch.equal(diff0[~touched], torch.zeros_like(diff0[~touched]))


# ── replace mode ──────────────────────────────────────────────────────


def test_replace_sets_state_exactly(tiny_model):
    input_ids, mask = _left_pad_batch(ROW_LENGTHS)
    T = input_ids.shape[1]
    positions = [[4], [7], [1]]
    deltas = _deltas([1, 1, 1], seed=11)

    base = _forward_capture(tiny_model, input_ids, mask, [0])
    hook = PositionEditHook(tiny_model, layer=0)
    hook.arm_batch(ROW_LENGTHS, positions, deltas, mode="replace", alpha=1.0)
    with hook:
        hook.arm(expected_prompt_len=T)
        edited = _forward_capture(tiny_model, input_ids, mask, [0])

    touched = torch.zeros(edited[0].shape[:2], dtype=torch.bool)
    for b, (rl, pos, d) in enumerate(zip(ROW_LENGTHS, positions, deltas, strict=True)):
        pp = _padded(rl, pos[0], T)
        touched[b, pp] = True
        # replace: the slot state BECOMES alpha * delta (exact assignment)
        assert torch.equal(edited[0][b, pp], d[0].to(edited[0].dtype)), (b, pp)
    diff0 = edited[0] - base[0]
    assert torch.equal(diff0[~touched], torch.zeros_like(diff0[~touched]))


def test_replace_requires_exactly_one_position_per_row(tiny_model):
    hook = PositionEditHook(tiny_model, layer=0)
    with pytest.raises(AssertionError, match="exactly ONE position"):
        hook.arm_batch([5], [[1, 2]], _deltas([2]), mode="replace")


# ── joint-layer stack ─────────────────────────────────────────────────


def test_joint_layer_stack_each_layer_own_delta(tiny_model):
    input_ids, mask = _left_pad_batch(ROW_LENGTHS)
    T = input_ids.shape[1]
    positions = [[4], [7], [2]]
    d_layer0 = _deltas([1, 1, 1], seed=21)
    d_layer1 = _deltas([1, 1, 1], seed=22)

    # Run A: layer-0-only edit.
    solo = PositionEditHook(tiny_model, layer=0)
    solo.arm_batch(ROW_LENGTHS, positions, d_layer0, mode="add")
    with solo:
        solo.arm(expected_prompt_len=T)
        only0 = _forward_capture(tiny_model, input_ids, mask, [0, 1])

    # Run B: joint stack at layers 0 + 1 (same positions, per-layer deltas).
    stack = joint_hooks(tiny_model, [0, 1])
    assert stack._handle is None  # not installed yet
    stack.arm_batch_per_layer(ROW_LENGTHS, positions, [d_layer0, d_layer1], mode="add")
    with stack:
        assert stack._handle is not None  # generate_batch's precondition
        stack.arm(expected_prompt_len=T)
        joint = _forward_capture(tiny_model, input_ids, mask, [0, 1])

    # Layer-0 outputs identical across runs (same layer-0 edit).
    assert torch.equal(joint[0], only0[0])
    # Layer-1 output: joint minus layer0-only differs EXACTLY by layer 1's own
    # delta at the edited positions (both runs share the layer-0 edit; layer 1's
    # edit is applied to its OUTPUT, so nothing else at that layer can differ).
    diff1 = joint[1] - only0[1]
    touched = torch.zeros(diff1.shape[:2], dtype=torch.bool)
    for b, (rl, pos, d) in enumerate(zip(ROW_LENGTHS, positions, d_layer1, strict=True)):
        pp = _padded(rl, pos[0], T)
        touched[b, pp] = True
        assert torch.allclose(diff1[b, pp], d[0], atol=1e-5), (b, pp)
    assert torch.equal(diff1[~touched], torch.zeros_like(diff1[~touched]))
    assert stack.n_edits == 2  # one edited prefill per layer


def test_stack_rejects_duplicate_layers(tiny_model):
    with pytest.raises(AssertionError, match="duplicate layers"):
        PositionEditHookStack([PositionEditHook(tiny_model, 0), PositionEditHook(tiny_model, 0)])


# ── prefill latch + arm/reset isolation ───────────────────────────────


def test_prefill_latch_second_forward_untouched(tiny_model):
    input_ids, mask = _left_pad_batch(ROW_LENGTHS)
    T = input_ids.shape[1]
    base = _forward_capture(tiny_model, input_ids, mask, [0])
    hook = PositionEditHook(tiny_model, layer=0)
    hook.arm_batch(ROW_LENGTHS, [[4], [7], [2]], _deltas([1, 1, 1]), mode="add")
    with hook:
        hook.arm(expected_prompt_len=T)
        first = _forward_capture(tiny_model, input_ids, mask, [0])
        assert hook.n_edits == 1
        assert not torch.equal(first[0], base[0])
        # Second forward WITHOUT re-arm: latch holds, no edit (decode analogue).
        second = _forward_capture(tiny_model, input_ids, mask, [0])
        assert hook.n_edits == 1
        assert torch.equal(second[0], base[0])
        # Decode-shaped T=1 forward: passes through (prefill already seen).
        one_tok = input_ids[:, -1:]
        with torch.no_grad():
            tiny_model(input_ids=one_tok, attention_mask=torch.ones_like(one_tok))
        assert hook.n_edits == 1
        # Re-arm (as generate_batch does per draw): edits again.
        hook.arm(expected_prompt_len=T)
        third = _forward_capture(tiny_model, input_ids, mask, [0])
        assert hook.n_edits == 2
        assert torch.equal(third[0], first[0])


def test_arm_batch_isolates_state_between_batches(tiny_model):
    """Re-arming with a NEW batch fully replaces positions/deltas/geometry."""
    row_lengths_1 = [5, 8, 3]
    input_1, mask_1 = _left_pad_batch(row_lengths_1, seed=1)
    row_lengths_2 = [6, 4]
    input_2, mask_2 = _left_pad_batch(row_lengths_2, seed=2)
    hook = PositionEditHook(tiny_model, layer=0)

    hook.arm_batch(row_lengths_1, [[4], [7], [2]], _deltas([1, 1, 1]), mode="add", alpha=3.0)
    with hook:
        hook.arm(expected_prompt_len=input_1.shape[1])
        _forward_capture(tiny_model, input_1, mask_1, [0])
        first_realized = hook.realized_edits
        assert first_realized is not None and len(first_realized) == 3

        # New batch: different B, lengths, positions, deltas, alpha.
        deltas_2 = _deltas([2, 1], seed=33)
        hook.arm_batch(row_lengths_2, [[0, 5], [3]], deltas_2, mode="add", alpha=1.0)
        assert hook.realized_edits is None  # invalidated until the next prefill
        T2 = input_2.shape[1]
        hook.remove()  # baseline must run unedited (and un-armed hook fail-louds)
        base_2 = _forward_capture(tiny_model, input_2, mask_2, [0])
        hook.install()
        hook.arm(expected_prompt_len=T2)
        edited_2 = _forward_capture(tiny_model, input_2, mask_2, [0])

    diff = edited_2[0] - base_2[0]
    touched = torch.zeros(diff.shape[:2], dtype=torch.bool)
    for b, (rl, pos, d) in enumerate(zip(row_lengths_2, [[0, 5], [3]], deltas_2, strict=True)):
        for j, p in enumerate(pos):
            pp = _padded(rl, p, T2)
            touched[b, pp] = True
            assert torch.allclose(diff[b, pp], d[j], atol=1e-5), (b, p)
    assert torch.equal(diff[~touched], torch.zeros_like(diff[~touched]))
    realized = hook.realized_edits
    assert realized is not None and len(realized) == 2
    assert realized[0]["positions_unpadded"] == [0, 5]
    assert realized[0]["positions_padded"] == [0 + (T2 - 6), 5 + (T2 - 6)]


# ── telemetry ─────────────────────────────────────────────────────────


def test_realized_edit_telemetry_matches_applied_edit(tiny_model):
    input_ids, mask = _left_pad_batch(ROW_LENGTHS)
    T = input_ids.shape[1]
    positions = [[4], [2], [0]]
    deltas = _deltas([1, 1, 1], seed=5)
    alpha = 0.5
    hook = PositionEditHook(tiny_model, layer=1)
    hook.arm_batch(ROW_LENGTHS, positions, deltas, mode="add", alpha=alpha)
    with hook:
        hook.arm(expected_prompt_len=T)
        _forward_capture(tiny_model, input_ids, mask, [1])
    realized = hook.realized_edits
    assert realized is not None and len(realized) == 3
    for b, rec in enumerate(realized):
        assert rec["row"] == b and rec["layer"] == 1 and rec["mode"] == "add"
        assert rec["alpha"] == alpha
        assert rec["positions_unpadded"] == positions[b]
        assert rec["positions_padded"] == [_padded(ROW_LENGTHS[b], positions[b][0], T)]
        assert torch.allclose(rec["applied"], alpha * deltas[b], atol=1e-6)


# ── error cases / contract asserts ────────────────────────────────────


def test_position_out_of_row_range_raises(tiny_model):
    hook = PositionEditHook(tiny_model, layer=0)
    with pytest.raises(AssertionError):
        hook.arm_batch([5], [[5]], _deltas([1]))  # p == row_len is out of range


def test_duplicate_positions_raise(tiny_model):
    hook = PositionEditHook(tiny_model, layer=0)
    with pytest.raises(AssertionError, match="duplicate"):
        hook.arm_batch([5], [[2, 2]], _deltas([2]))


def test_arm_before_arm_batch_raises(tiny_model):
    hook = PositionEditHook(tiny_model, layer=0)
    with pytest.raises(AssertionError, match="arm_batch"):
        hook.arm(expected_prompt_len=8)


def test_delta_shape_mismatch_raises(tiny_model):
    hook = PositionEditHook(tiny_model, layer=0)
    with pytest.raises(AssertionError):
        hook.arm_batch([5], [[1, 2]], _deltas([3]))  # 2 positions, 3 delta rows


def test_row_longer_than_padded_T_raises(tiny_model):
    hook = PositionEditHook(tiny_model, layer=0)
    hook.arm_batch([10], [[9]], _deltas([1]))
    with pytest.raises(AssertionError):
        hook.arm(expected_prompt_len=8)  # T < row length


def test_prompt_len_mismatch_at_prefill_raises(tiny_model):
    input_ids, mask = _left_pad_batch(ROW_LENGTHS)  # T == 8
    hook = PositionEditHook(tiny_model, layer=0)
    hook.arm_batch(ROW_LENGTHS, [[4], [7], [2]], _deltas([1, 1, 1]))
    with hook:
        hook.arm(expected_prompt_len=ROW_LENGTHS[1] + 3)  # 11 != realized T=8
        with pytest.raises(AssertionError):
            _forward_capture(tiny_model, input_ids, mask, [0])


def test_duck_type_contract_surface(tiny_model):
    """The exact attributes/calls generate_batch relies on (steering.py)."""
    hook = PositionEditHook(tiny_model, layer=0)
    assert hook._handle is None
    hook.arm_batch([4], [[3]], _deltas([1]))
    with hook as installed:
        assert installed is hook
        assert hook._handle is not None  # generate_batch's install assert
        hook.arm(expected_prompt_len=6)  # generate_batch's per-draw call
        hook.reset()
    assert hook._handle is None
