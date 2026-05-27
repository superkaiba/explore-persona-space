"""TDD Phase 1 — Page's L synthetic-data sanity (task #397, plan v4 §5.9 + §14 item 3).

H2 in plan v4 §1 + §7 uses **Page's L one-tailed (alternative: E0 < E1 < E2)**
over 108 blocks × 3 ordered E levels on selectivity Δ as the primary test
for the ordinal-E trend. This test surface verifies the aggregator's
``pages_l_test`` against synthetic data with known ground truth:

  - perfect monotonic ↑ → p << 0.05 with alternative="increasing"
  - random / flat → p ≈ 0.5
  - reversed monotonic → p ≫ 0.05 with alternative="increasing" (and
    p << 0.05 with alternative="decreasing")
  - 1-inversion (one block with E0 > E1) → still above threshold for n=108
    (the asymptotic z is robust to small noise at this n)

These are S1 must-fix carry-forwards from the v3 round-2 critic.

CPU-only; no model load.
"""

from __future__ import annotations

import random

import pytest

from explore_persona_space.experiments.factor_screen_397.aggregator import pages_l_test


def _monotonic_increasing_blocks(n_blocks: int = 108) -> list[list[float]]:
    """Each block is [v_E0, v_E1, v_E2] with strict increasing order."""
    rng = random.Random(0)
    blocks: list[list[float]] = []
    for _ in range(n_blocks):
        base = rng.uniform(0, 10)
        blocks.append([base, base + 1.0, base + 2.0])
    return blocks


def _monotonic_decreasing_blocks(n_blocks: int = 108) -> list[list[float]]:
    rng = random.Random(1)
    blocks: list[list[float]] = []
    for _ in range(n_blocks):
        base = rng.uniform(0, 10)
        blocks.append([base + 2.0, base + 1.0, base])
    return blocks


def _flat_blocks(n_blocks: int = 108) -> list[list[float]]:
    """Each block has 3 identical values; tied within-block ranks → L ≈ E[L]."""
    rng = random.Random(2)
    return [[rng.uniform(0, 10)] * 3 for _ in range(n_blocks)]


def _random_blocks(n_blocks: int = 108) -> list[list[float]]:
    rng = random.Random(3)
    return [[rng.uniform(0, 10) for _ in range(3)] for _ in range(n_blocks)]


def _blocks_with_one_inversion(n_blocks: int = 108) -> list[list[float]]:
    """108 increasing blocks; flip the first one to make it decreasing."""
    blocks = _monotonic_increasing_blocks(n_blocks)
    blocks[0] = list(reversed(blocks[0]))
    return blocks


@pytest.mark.xfail(
    reason="Phase 1 (TDD) stub — pages_l_test raises NotImplementedError until Phase 2.",
    strict=True,
    raises=NotImplementedError,
)
def test_pages_l_monotonic_increasing_yields_p_far_below_threshold() -> None:
    """Perfect monotonic ↑ over 108 blocks → p_one_tailed_increasing << 0.05."""
    blocks = _monotonic_increasing_blocks(108)
    result = pages_l_test(blocks, alternative="increasing")

    assert "L" in result
    assert "expected_L" in result
    assert "p_one_tailed" in result
    assert result["n_blocks"] == 108
    assert result["alternative"] == "increasing"
    assert result["p_one_tailed"] < 0.001, (
        f"Monotonic ↑ blocks should yield p << 0.05; got p={result['p_one_tailed']}"
    )


@pytest.mark.xfail(
    reason="Phase 1 (TDD) stub — pages_l_test raises NotImplementedError until Phase 2.",
    strict=True,
    raises=NotImplementedError,
)
def test_pages_l_random_yields_p_near_one_half() -> None:
    """Random blocks → p_one_tailed ≈ 0.5 (no trend signal)."""
    blocks = _random_blocks(108)
    result = pages_l_test(blocks, alternative="increasing")
    # Loose bound — random blocks at n=108 won't always sit exactly at 0.5,
    # but anything outside [0.05, 0.95] would mean the test is overclaiming
    # significance on noise.
    assert 0.05 < result["p_one_tailed"] < 0.95, (
        f"Random blocks should not produce a significant trend p; got p={result['p_one_tailed']}"
    )


@pytest.mark.xfail(
    reason="Phase 1 (TDD) stub — pages_l_test raises NotImplementedError until Phase 2.",
    strict=True,
    raises=NotImplementedError,
)
def test_pages_l_reversed_fails_increasing_test_but_passes_decreasing_test() -> None:
    """Monotonic ↓ blocks: increasing test → p ≈ 1; decreasing test → p << 0.05."""
    blocks = _monotonic_decreasing_blocks(108)

    inc = pages_l_test(blocks, alternative="increasing")
    assert inc["p_one_tailed"] > 0.95, (
        f"Reversed blocks must NOT pass alternative='increasing'; got p={inc['p_one_tailed']}"
    )

    dec = pages_l_test(blocks, alternative="decreasing")
    assert dec["p_one_tailed"] < 0.001, (
        f"Reversed blocks should yield p << 0.05 with alternative='decreasing'; "
        f"got p={dec['p_one_tailed']}"
    )


@pytest.mark.xfail(
    reason="Phase 1 (TDD) stub — pages_l_test raises NotImplementedError until Phase 2.",
    strict=True,
    raises=NotImplementedError,
)
def test_pages_l_one_inversion_still_passes_at_n_108() -> None:
    """A single flipped block among 108 should still pass alternative='increasing'.

    The test is robust at n=108 — one inversion doesn't move the z-score much.
    This protects against a Phase 2 implementation that's too strict.
    """
    blocks = _blocks_with_one_inversion(108)
    result = pages_l_test(blocks, alternative="increasing")
    assert result["p_one_tailed"] < 0.05, (
        f"107 monotonic blocks + 1 inversion should still yield p < 0.05; "
        f"got p={result['p_one_tailed']}"
    )
