"""Pin the #673 real-GPU memory-non-growth validation contract (CPU-only).

This test does NOT run the GPU benchmark (no GPU in CI / on the VM) and does NOT
load Qwen. It pins the *analysis contract* and the *gate-strength constants* so a
future edit cannot silently weaken the validation:

1. **Constant guards (regression-lock the gate strength).** The flatness
   tolerances + benchmark knobs are pinned at literal values; a drift fails here
   (mirrors how #671 pinned its CPU-proxy contract in
   ``tests/test_issue671_extraction_hooks.py``).
2. **``flat()`` predicate logic on synthetic curves.** A flat curve PASSes; a
   +0.5 GiB/iter growth curve FAILs; a borderline curve sitting near ``ABS_TOL``
   is asserted correctly; a warmup-ramp-then-flat curve PASSes (the warmup window
   is correctly excluded).
3. **``evaluate()`` verdict logic on synthetic JSON.** A flat-hook + growing-old
   pair → PASS; a growing-hook pair → REGRESSION; a flat-hook + flat-old pair
   with no gap → INCONCLUSIVE.
4. **The no-grad parity assertion.** A synthetic JSON record with
   ``grad_enabled: true`` for any arm raises an ``AssertionError`` from the reader
   (the Must-Fix isolation guard).

CPU-only — does NOT call CUDA, does NOT load the model.
"""

# ruff: noqa: RUF003  # scientific notation (−, ≈) in comments

from __future__ import annotations

import numpy as np
import pytest

from scripts.issue673_assert import (
    INCONCLUSIVE,
    PASS,
    REGRESSION,
    evaluate,
)
from scripts.issue673_gpu_memory_validation import (
    LAYERS,
    MAX_TOKENS,
    N_ITERS,
    WARMUP,
    ABS_TOL_GiB,
    CTRL_GAP_GiB,
    GiB,
    SLOPE_TOL_GiB_per_iter,
    flat,
    regime_tag,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Constant guards — regression-lock the gate strength.
# ─────────────────────────────────────────────────────────────────────────────


def test_gate_strength_constants_pinned():
    """The flatness tolerances + benchmark knobs are pinned. A future edit that
    loosens any of them (raises a tolerance, shrinks N, moves the layer subset)
    trips this lock — the gate cannot be silently weakened."""
    assert N_ITERS >= 50, N_ITERS
    assert WARMUP == 5, WARMUP
    assert ABS_TOL_GiB == 1.0, ABS_TOL_GiB
    assert SLOPE_TOL_GiB_per_iter == 0.02, SLOPE_TOL_GiB_per_iter
    assert CTRL_GAP_GiB == 0.05, CTRL_GAP_GiB
    assert MAX_TOKENS == 512, MAX_TOKENS
    assert LAYERS == [7, 14, 21], LAYERS


def test_regime_tag_maps_allocator_config():
    """The allocator-regime tag is derived from PYTORCH_CUDA_ALLOC_CONF: the
    expandable-segments value maps to ``expandable_segments_on``, anything else
    (unset / default) maps to ``default_allocator``."""
    assert regime_tag("expandable_segments:True") == "expandable_segments_on"
    assert regime_tag("expandable_segments:True,max_split_size_mb:128") == "expandable_segments_on"
    assert regime_tag("default") == "default_allocator"
    assert regime_tag("") == "default_allocator"
    assert regime_tag("max_split_size_mb:128") == "default_allocator"


# ─────────────────────────────────────────────────────────────────────────────
# 2. flat() predicate logic on synthetic curves.
# ─────────────────────────────────────────────────────────────────────────────


def _const_curve(value_gib: float, n: int = N_ITERS, jitter_gib: float = 0.0) -> list[int]:
    """A constant reserved curve (bytes) with optional uniform jitter."""
    rng = np.random.default_rng(0)
    base = np.full(n, value_gib, float)
    if jitter_gib:
        base = base + rng.uniform(-jitter_gib, jitter_gib, n)
    return [int(v * GiB) for v in base]


def _growth_curve(start_gib: float, per_iter_gib: float, n: int = N_ITERS) -> list[int]:
    """A monotone-growing reserved curve (bytes): start + per_iter * i."""
    return [int((start_gib + per_iter_gib * i) * GiB) for i in range(n)]


def test_flat_curve_is_flat():
    """A constant curve (with sub-tolerance jitter) reads flat=True."""
    res = flat(_const_curve(20.0, jitter_gib=0.01))
    assert res["flat"] is True, res
    assert res["span_GiB"] < ABS_TOL_GiB
    assert abs(res["tail_slope_GiB_per_iter"]) < SLOPE_TOL_GiB_per_iter


def test_growth_curve_is_not_flat():
    """A +0.5 GiB/iter growth curve (~the #545 accumulation regime) reads
    flat=False — both the span and the slope blow past tolerance."""
    res = flat(_growth_curve(20.0, 0.5))
    assert res["flat"] is False, res
    assert res["span_GiB"] >= ABS_TOL_GiB
    assert res["tail_slope_GiB_per_iter"] >= SLOPE_TOL_GiB_per_iter


def test_borderline_span_curve():
    """A curve whose post-warmup span sits just ABOVE ABS_TOL (a step up of
    1.5 GiB) is NOT flat even with a flat tail (the span criterion catches it);
    a curve whose span sits just BELOW ABS_TOL is flat. Pins the span boundary."""
    # Step from 20 to 21.5 GiB right after warmup, then flat tail -> span 1.5 > 1.0.
    over = [20.0] * (WARMUP + 1) + [21.5] * (N_ITERS - WARMUP - 1)
    over_curve = [int(v * GiB) for v in over]
    assert flat(over_curve)["flat"] is False, flat(over_curve)
    # Step of 0.5 GiB -> span 0.5 < 1.0, flat tail -> flat.
    under = [20.0] * (WARMUP + 1) + [20.5] * (N_ITERS - WARMUP - 1)
    under_curve = [int(v * GiB) for v in under]
    assert flat(under_curve)["flat"] is True, flat(under_curve)


def test_warmup_ramp_then_flat_is_flat():
    """A curve that ramps during the warmup window (lazy CUDA/cuDNN allocation)
    then plateaus is flat — the warmup iterations are excluded from the span."""
    # First WARMUP iters ramp 15->20 GiB (a 5 GiB warmup ramp), then constant 20.
    ramp = list(np.linspace(15.0, 20.0, WARMUP)) + [20.0] * (N_ITERS - WARMUP)
    curve = [int(v * GiB) for v in ramp]
    res = flat(curve)
    assert res["flat"] is True, res
    # The full-curve span would be 5 GiB; the post-warmup span must be ~0.
    assert res["span_GiB"] < ABS_TOL_GiB


# ─────────────────────────────────────────────────────────────────────────────
# 3. evaluate() verdict logic on synthetic JSON.
# ─────────────────────────────────────────────────────────────────────────────


def _arm(reserved, *, grad_enabled=False, inference_mode=True, allocated=None):
    """Build a synthetic per-arm record matching the benchmark JSON schema."""
    if allocated is None:
        allocated = [int(v * 0.5) for v in reserved]
    return {
        "allocated": allocated,
        "reserved": reserved,
        "segment_stats": [
            {"reserved_current": r, "reserved_peak": r, "segment_current": 1, "segment_peak": 1}
            for r in reserved
        ],
        "grad_enabled": grad_enabled,
        "inference_mode": inference_mode,
    }


def _regime(hook_reserved, old_reserved, **arm_kw):
    """Build a synthetic per-regime results dict with hook + old arms."""
    return {
        "arms": {
            "hook": _arm(hook_reserved, **arm_kw),
            "old_ohs_true": _arm(old_reserved, **arm_kw),
        }
    }


def test_evaluate_pass_flat_hook_growing_old():
    """Flat hook under both allocators + a growing old arm under expandable
    (a clear reserved gap) -> PASS."""
    flat_hook = _const_curve(20.0, jitter_gib=0.005)
    growing_old = _growth_curve(22.0, 0.3)  # climbs well above the hook plateau
    flat_old = _const_curve(20.0, jitter_gib=0.005)
    expandable = _regime(flat_hook, growing_old)
    default = _regime(flat_hook, flat_old)
    verdict, msg = evaluate(expandable, default)
    assert verdict == PASS, (verdict, msg)


def test_evaluate_pass_on_retained_highwater_gap():
    """Flat hook + a flat-but-HIGHER old arm (retained high-water, no climb) under
    expandable -> PASS via the reserved-gap branch (criterion 2's primary read)."""
    flat_hook = _const_curve(20.0, jitter_gib=0.005)
    higher_old = _const_curve(20.5, jitter_gib=0.005)  # +0.5 GiB > CTRL_GAP_GiB
    expandable = _regime(flat_hook, higher_old)
    default = _regime(flat_hook, _const_curve(20.0, jitter_gib=0.005))
    verdict, msg = evaluate(expandable, default)
    assert verdict == PASS, (verdict, msg)


def test_evaluate_regression_on_growing_hook():
    """A growing hook arm under either allocator -> REGRESSION (a #671 regression),
    regardless of the old arm. REGRESSION takes precedence."""
    growing_hook = _growth_curve(20.0, 0.5)
    flat_old = _const_curve(20.0, jitter_gib=0.005)
    expandable = _regime(growing_hook, _growth_curve(22.0, 0.3))
    default = _regime(growing_hook, flat_old)
    verdict, msg = evaluate(expandable, default)
    assert verdict == REGRESSION, (verdict, msg)


def test_evaluate_inconclusive_flat_hook_flat_old_no_gap():
    """Flat hook + flat old at the SAME reserved level (no gap, no trend) under
    expandable -> INCONCLUSIVE (never a false PASS)."""
    flat_hook = _const_curve(20.0, jitter_gib=0.005)
    flat_old = _const_curve(20.0, jitter_gib=0.005)  # same level, gap < CTRL_GAP_GiB
    expandable = _regime(flat_hook, flat_old)
    default = _regime(flat_hook, flat_old)
    verdict, msg = evaluate(expandable, default)
    assert verdict == INCONCLUSIVE, (verdict, msg)


# ─────────────────────────────────────────────────────────────────────────────
# 4. No-grad parity assertion (the Must-Fix isolation guard).
# ─────────────────────────────────────────────────────────────────────────────


def test_grad_enabled_record_raises():
    """A record with grad_enabled=True on EITHER arm raises AssertionError — an
    autograd-graph-inflated positive-control gap must never be read as a real
    output_hidden_states retention gap."""
    flat_hook = _const_curve(20.0, jitter_gib=0.005)
    growing_old = _growth_curve(22.0, 0.3)
    # old arm ran grad-enabled -> invalid positive control.
    expandable = {
        "arms": {
            "hook": _arm(flat_hook),
            "old_ohs_true": _arm(growing_old, grad_enabled=True),
        }
    }
    default = _regime(flat_hook, _const_curve(20.0, jitter_gib=0.005))
    with pytest.raises(AssertionError, match="grad_enabled must be False"):
        evaluate(expandable, default)


def test_inference_mode_false_record_raises():
    """A record with inference_mode=False on either arm also raises — the loop
    must have run under torch.inference_mode()."""
    flat_hook = _const_curve(20.0, jitter_gib=0.005)
    expandable = {
        "arms": {
            "hook": _arm(flat_hook, inference_mode=False),
            "old_ohs_true": _arm(_growth_curve(22.0, 0.3)),
        }
    }
    default = _regime(flat_hook, _const_curve(20.0, jitter_gib=0.005))
    with pytest.raises(AssertionError, match="inference_mode must be True"):
        evaluate(expandable, default)
