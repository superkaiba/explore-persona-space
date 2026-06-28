"""Issue #715 — P4 Ignore-topK mask correctness (plan §13).

The prunability curve is built on this mask; a wrong mask silently invalidates
P4. Toy Δθ with known entries: assert the K-largest-|Δθ| entries are zeroed, the
rest unchanged, K=0 is identity, the mask is binary, and apply_ignore_topk
reverts exactly the zeroed entries to base.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from issue715_p4_geometry_pruning import (
    apply_ignore_topk,
    ignore_topk_mask,
    ignore_topk_mask_signed,
)


def test_mask_zeroes_largest_magnitude_entries():
    delta = torch.tensor([1.0, -5.0, 0.5, 3.0, -0.2])  # |Δ| = [1, 5, 0.5, 3, 0.2]
    # k_frac=0.4 of 5 = 2 entries -> zero the two largest |Δ|: indices 1 (5.0), 3 (3.0)
    mask = ignore_topk_mask(delta, 0.4)
    assert torch.equal(mask, torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))


def test_k_zero_is_identity():
    delta = torch.randn(10)
    assert torch.equal(ignore_topk_mask(delta, 0.0), torch.ones_like(delta))


def test_mask_is_binary():
    delta = torch.randn(50)
    mask = ignore_topk_mask(delta, 0.2)
    uniq = set(mask.unique().tolist())
    assert uniq.issubset({0.0, 1.0})
    # Exactly round(0.2*50)=10 entries zeroed.
    assert int((mask == 0).sum()) == 10


def test_signed_variant_zeroes_largest_signed_not_magnitude():
    delta = torch.tensor([1.0, -5.0, 0.5, 3.0, -0.2])  # signed: largest are 3.0, 1.0
    mask = ignore_topk_mask_signed(delta, 0.4)  # zero the 2 largest SIGNED values
    # Largest signed = 3.0 (idx 3), 1.0 (idx 0); NOT -5.0 (most negative).
    assert torch.equal(mask, torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0]))


def test_apply_ignore_topk_per_tensor_reverts_topk_to_base():
    base_sd = {"w": torch.zeros(5)}
    ft_sd = {"w": torch.tensor([1.0, -5.0, 0.5, 3.0, -0.2])}
    pruned = apply_ignore_topk(ft_sd, base_sd, 0.4, ["w"])
    # idx 1 (5.0) and 3 (3.0) revert to base (0.0); others keep their ft value.
    expected = torch.tensor([1.0, 0.0, 0.5, 0.0, -0.2])
    assert torch.allclose(pruned["w"], expected)


def test_apply_ignore_topk_global_scope():
    base_sd = {"a": torch.zeros(3), "b": torch.zeros(3)}
    ft_sd = {"a": torch.tensor([10.0, 0.1, 0.2]), "b": torch.tensor([0.3, 9.0, 0.05])}
    # 6 entries, k_frac=1/3 -> zero the 2 globally-largest |Δ|: 10.0 (a[0]), 9.0 (b[1]).
    pruned = apply_ignore_topk(ft_sd, base_sd, 1 / 3, ["a", "b"], global_scope=True)
    assert pruned["a"][0].item() == 0.0  # 10.0 zeroed
    assert pruned["b"][1].item() == 0.0  # 9.0 zeroed
    assert abs(pruned["a"][1].item() - 0.1) < 1e-6  # small entries kept
    assert abs(pruned["b"][0].item() - 0.3) < 1e-6
