"""Task #505 round-6 regression test — the LoRA-recipe override threading.

Round-6 launch crashed with::

    ValueError: LoRA rank 32 is greater than max_lora_rank 16.
    File ".../eval_trajectory.py", line 179, in _generate_on_policy_R
        outputs = llm.generate(prompts, sp, lora_request=lora_request)

Root cause: the #505 dispatcher's ``_train_and_eval_one_cell`` invoked
``train_one_cell`` (inherited from #472) without threading the #505 LoRA-recipe
overrides. ``train_one_cell`` then fell back to #472's module constants
(``LORA_R=32``, ``LORA_ALPHA=64``, ``LEARNING_RATE=1e-5``) — the saturating
anchor #505 explicitly avoids per plan §5.1. The trained adapter was rank 32
but the eval rig capped vLLM at rank 16, so vLLM rejected the load at the
first ``llm.generate(... lora_request=...)`` call.

The fix is two-fold:

  1. The dispatcher's ``train_one_cell(...)`` call must pass
     ``lora_r_override=LORA_R``, ``lora_alpha_override=LORA_ALPHA``,
     ``lr_override=LEARNING_RATE``, ``epochs_override=EPOCHS`` — all sourced
     from ``leave_one_out_505/__init__.py``, NOT from the #472 module.
  2. The eval rig's ``max_lora_rank`` default must be at least
     ``FALLBACK_LORA_R`` so the cap accommodates the §5.5 smoke fallback
     anchor (rank 32). (Adapter rank is still pinned by (1); this is
     belt-and-suspenders for the fallback path.)

Round-8 (2026-06-07) bumped ``LORA_R`` from 16 to 32 per plan §8's
pre-registered under-training fallback (round-7 ΔG=0.82 nats at frac 0.50,
~6x under the §7 5-nat validity floor; rank was the bottleneck). Primary
anchor and fallback anchor now coincide at rank 32; the eval cap math
(``max(LORA_R, FALLBACK_LORA_R) = 32``) still resolves to 32 unchanged.

Both checks are AST / import-introspection only — no model load, runs in <1 s
on CPU. The test fires loud the moment a future refactor drops any of the
four override kwargs OR lowers the eval cap below 32.
"""

from __future__ import annotations

import ast
from pathlib import Path

from explore_persona_space.experiments.leave_one_out_505 import (
    EPOCHS,
    FALLBACK_LORA_R,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_R,
    eval_trajectory_505,
)

# Anchor recipe per plan §5.1 + §8 (under-training fallback) — the recipe
# #505 selects to preserve the recipe gradient. If any of these numbers drift
# the smoke gate will also fail (the source ΔG band check at frac 0.50), but
# this test fires earlier and gives a clearer diagnostic.
# Round-8 bump (2026-06-07): _EXPECTED_LORA_R 16 -> 32. Round-7 (rank 16 + lr
# 5e-6 + 3 epochs / 75 steps) hit ΔG=0.82 nats at frac 0.50, ~6x under plan
# §7's 5-nat validity floor. Rank was the under-training bottleneck per the
# trajectory (25 steps -> 0.04 nats, 75 steps -> 0.82 nats); plan §8
# pre-registered rank-32 + lr 5e-6 as the fallback. Primary anchor and
# fallback anchor now coincide at rank 32 (FALLBACK_LORA_R unchanged).
_EXPECTED_LORA_R = 32
_EXPECTED_LORA_ALPHA = 32
# Round-9 bump (2026-06-08): _EXPECTED_LEARNING_RATE 5e-6 -> 1e-5. Round-8
# (rank 32 + lr 5e-6 + 3 epochs / 75 steps) produced a FLAT source-ΔG
# trajectory at ≈1.6 nats (0.08:1.63, 0.16:1.57, 0.33:1.51, 0.50:1.60,
# 0.75:1.59, 1.00:1.69) — still ~3x under plan §7's 5-nat validity floor.
# A flat trajectory is the LR-bound signature; bumping EPOCHS further would
# not have moved it. #472's "1e-5 saturates" prior used 800-neg / 62-step;
# #505 runs 200-neg / 75-step (under-trained), so saturation isn't binding.
_EXPECTED_LEARNING_RATE = 1e-5
# Round-7 bump (2026-06-06): EPOCHS 1 → 3. Round-6 smoke (WandB run yjz5ytuz)
# showed mean_token_accuracy=0.645 with grad_norm RISING at the end of training
# under 1 epoch (25 optimizer steps); the source-self ΔG was 0.04 nats and the
# eval-guard correctly fired LoRANotAppliedError. 3 epochs = 75 optimizer steps
# was still under-training in round 7 (motivating the round-8 rank bump). Held
# at 3 in round 8 — single-axis rank bump on top of round 7's epoch bump.
_EXPECTED_EPOCHS = 3
_EXPECTED_FALLBACK_LORA_R = 32


# ── (1) Module constants — pin plan §5.1 numbers. ────────────────────────────


def test_issue505_lora_r_pin():
    """Plan §5.1 + §8: round-8 pin at rank 32 (under-training fallback after
    round-7 ΔG=0.82 nats at frac 0.50 with rank 16). Rank value is the
    primary single-axis knob; the smoke gate (a) catches drift on the
    outcome, this test catches drift on the constant earlier."""
    assert LORA_R == _EXPECTED_LORA_R, (
        f"LORA_R must be {_EXPECTED_LORA_R} per plan §8 (round-8 under-training "
        f"fallback); got {LORA_R}."
    )


def test_issue505_lora_alpha_is_32():
    """Plan §5.1: alpha=32. Round 1-7 used alpha=2*r (rank 16); round 8 bumps
    rank to 32 while holding alpha fixed at 32 → alpha=r. This is deliberate:
    the round-8 bump is single-axis (rank only), and lora_alpha is the
    scaling-knob the smoke gate's outcome captures via ΔG."""
    assert LORA_ALPHA == _EXPECTED_LORA_ALPHA, (
        f"LORA_ALPHA must be {_EXPECTED_LORA_ALPHA} per plan §5.1; got {LORA_ALPHA}."
    )


def test_issue505_learning_rate_is_1e_minus_5():
    """Round-9 bump (2026-06-08): lr 5e-6 → 1e-5. Round-8 trajectory was FLAT
    at ≈1.6 nats across the full curve (the LR-bound signature); plan §11's
    "1e-5 saturates" prior used #472's 800-neg/62-step recipe and doesn't
    transfer to #505's 200-neg/75-step under-trained regime."""
    assert LEARNING_RATE == _EXPECTED_LEARNING_RATE, (
        f"LEARNING_RATE must be {_EXPECTED_LEARNING_RATE} (round-9 anchor); got {LEARNING_RATE}."
    )


def test_issue505_epochs_is_3():
    """Round-7 bump (2026-06-06): EPOCHS = 3. Plan §5.1's 1-epoch default
    under-trained the marker at the §5.1 sub-saturating anchor (smoke
    yjz5ytuz: mean_token_accuracy=0.645, grad_norm rising, source-self
    ΔG ≈ 0.04 nats). The constant pin catches future drift back to 1."""
    assert EPOCHS == _EXPECTED_EPOCHS, f"EPOCHS must be {_EXPECTED_EPOCHS}; got {EPOCHS}."


def test_issue505_fallback_lora_r_is_32():
    """Plan §5.5 smoke fallback: rank 32 is the slightly stronger anchor if
    the primary smoke gate fails. The eval cap must accommodate it."""
    assert FALLBACK_LORA_R == _EXPECTED_FALLBACK_LORA_R, (
        f"FALLBACK_LORA_R must be {_EXPECTED_FALLBACK_LORA_R} per plan §5.5; got {FALLBACK_LORA_R}."
    )


# ── (2) Dispatcher AST — every train_one_cell(...) threads the overrides. ───


def _dispatch_source() -> str:
    p = Path(__file__).resolve().parent.parent
    src = p / "src" / "explore_persona_space" / "experiments" / "leave_one_out_505" / "dispatch.py"
    assert src.exists(), f"#505 dispatcher missing at {src}"
    return src.read_text()


def _train_one_cell_calls(src: str) -> list[ast.Call]:
    tree = ast.parse(src)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = None
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            if name == "train_one_cell":
                calls.append(node)
    return calls


def test_dispatcher_threads_lora_r_override():
    """Every dispatcher call to ``train_one_cell`` must pin ``lora_r_override``;
    otherwise the trained adapter inherits #472's rank 32 and vLLM rejects it
    at the rank-16 eval cap (round-6 crash signature)."""
    calls = _train_one_cell_calls(_dispatch_source())
    assert calls, "no train_one_cell(...) call found in dispatch.py"
    for call in calls:
        kw_names = {kw.arg for kw in call.keywords if kw.arg}
        assert "lora_r_override" in kw_names, (
            f"dispatch.py train_one_cell at line {call.lineno} missing "
            "lora_r_override=LORA_R — without it the adapter inherits #472's "
            "rank 32 and the eval rig at rank-16 cap rejects it (round-6 crash)."
        )


def test_dispatcher_threads_lora_alpha_override():
    """Every dispatcher call to ``train_one_cell`` must pin ``lora_alpha_override``
    — #472's default LORA_ALPHA=64 pairs with rank 32; pairing LORA_ALPHA=64
    with rank 16 (or paring 32 with rank 32) breaks the alpha = 2*r scaling."""
    calls = _train_one_cell_calls(_dispatch_source())
    for call in calls:
        kw_names = {kw.arg for kw in call.keywords if kw.arg}
        assert "lora_alpha_override" in kw_names, (
            f"dispatch.py train_one_cell at line {call.lineno} missing "
            "lora_alpha_override=LORA_ALPHA — without it alpha defaults to "
            "#472's 64 instead of #505's 32."
        )


def test_dispatcher_threads_lr_override():
    """Every dispatcher call to ``train_one_cell`` must pin ``lr_override``.
    Although round-9 lifted #505's LEARNING_RATE to 1e-5 (matching #472's
    default value), this convergence is incidental — the override is still
    required so future drift in #472's default does not silently change
    #505's anchor. lr is the most outcome-changing knob in the recipe."""
    calls = _train_one_cell_calls(_dispatch_source())
    for call in calls:
        kw_names = {kw.arg for kw in call.keywords if kw.arg}
        assert "lr_override" in kw_names, (
            f"dispatch.py train_one_cell at line {call.lineno} missing "
            "lr_override=LEARNING_RATE — without it lr depends on #472's "
            "default rather than #505's pinned anchor."
        )


def test_dispatcher_threads_epochs_override():
    """Every dispatcher call to ``train_one_cell`` must pin ``epochs_override``.
    Round-7 bumped #505's EPOCHS to 3 while #472's default remains 1; without
    the explicit override the trained adapter would inherit #472's 1-epoch
    recipe (the under-trained anchor that motivated the bump)."""
    calls = _train_one_cell_calls(_dispatch_source())
    for call in calls:
        kw_names = {kw.arg for kw in call.keywords if kw.arg}
        assert "epochs_override" in kw_names, (
            f"dispatch.py train_one_cell at line {call.lineno} missing "
            "epochs_override=EPOCHS — without it future drift in #472's "
            "default EPOCHS would silently change #505's recipe."
        )


# ── (3) Eval rig — max_lora_rank default accommodates both anchors. ──────────


def test_eval_trajectory_max_lora_rank_default_covers_both_anchors():
    """``run_trajectory_eval_with_guard`` must default ``max_lora_rank`` to a
    value that accommodates BOTH the §5.1 primary anchor (rank 16) AND the
    §5.5 smoke fallback anchor (rank 32). The original default of 16 would
    have rejected the fallback adapter immediately (and the round-6 trained
    rank-32 adapter, which was a separate dispatcher bug)."""
    import inspect

    sig = inspect.signature(eval_trajectory_505.run_trajectory_eval_with_guard)
    default = sig.parameters["max_lora_rank"].default
    expected_floor = max(LORA_R, FALLBACK_LORA_R)
    assert default >= expected_floor, (
        f"run_trajectory_eval_with_guard.max_lora_rank default {default} < "
        f"max(LORA_R, FALLBACK_LORA_R) = {expected_floor}. vLLM rejects "
        f"adapters whose rank exceeds the cap (round-6 crash class)."
    )


def test_eval_trajectory_max_lora_rank_default_at_least_32():
    """Explicit 32-floor regression — even if the module constants are later
    edited, the eval cap must accommodate rank 32 because that's the documented
    §5.5 fallback anchor."""
    import inspect

    sig = inspect.signature(eval_trajectory_505.run_trajectory_eval_with_guard)
    default = sig.parameters["max_lora_rank"].default
    assert default >= 32, (
        f"run_trajectory_eval_with_guard.max_lora_rank default {default} < 32; "
        "the §5.5 fallback anchor uses rank 32 and would be rejected at load."
    )
