"""Unit tests for the mid-run quality gate in issue #377's corpus generator.

Round-4 FIX 3 (2026-05-22): adds ``_early_quality_gate_check`` that
catches a refusal cascade BEFORE the full 22-turn x N-conversation
Anthropic Batch spend. Tested here in isolation since the production
hook runs inside ``run_conversation_loop`` which we don't want to
exercise end-to-end in unit tests (it submits Anthropic Batches).
"""

from __future__ import annotations

import pytest

from explore_persona_space.data_gen.issue377_corpus import (
    EARLY_GATE_TURN_THRESHOLD,
    _early_quality_gate_check,
)


def _make_conv(domain: str, turn_pattern: list[str]) -> dict:
    """Build a fake conversation with the given per-turn content pattern.

    ``turn_pattern`` items are either ``"ok"`` (normal content) or
    ``"[BATCH_ERROR]"`` (sentinel).
    """
    return {
        "conversation_id": f"{domain}_p0_t0",
        "domain": domain,
        "turns": [{"role": "user", "content": c} for c in turn_pattern],
        "n_turns": len(turn_pattern),
    }


class TestEarlyGateSkipsBelowThreshold:
    """The gate is silent until at least ``EARLY_GATE_TURN_THRESHOLD`` turns
    have completed, even when error rates are catastrophic. This is by
    design — gives early turns a chance to settle.
    """

    def test_silent_at_turn_zero(self):
        # 100% errors but only 1 turn → silent.
        convs = [_make_conv("therapy", ["[BATCH_ERROR]"])]
        _early_quality_gate_check(convs, turn_idx=0)  # no raise

    def test_silent_below_threshold(self):
        # All turns at threshold-2 are errors → still silent.
        below = EARLY_GATE_TURN_THRESHOLD - 2
        convs = [_make_conv("therapy", ["[BATCH_ERROR]"] * (below + 1))]
        _early_quality_gate_check(convs, turn_idx=below)  # no raise


class TestEarlyGateFiresOnGlobalThreshold:
    """When the global [BATCH_ERROR] rate exceeds ``EARLY_GATE_GLOBAL_MAX_FRAC``
    AFTER the threshold turn, the gate raises with a per-(domain, turn)
    breakdown."""

    def test_global_breach(self):
        # All 5 turns errors, 1 conversation → 100% global > 5%.
        convs = [_make_conv("therapy", ["[BATCH_ERROR]"] * EARLY_GATE_TURN_THRESHOLD)]
        with pytest.raises(RuntimeError, match="Mid-run quality gate"):
            _early_quality_gate_check(convs, turn_idx=EARLY_GATE_TURN_THRESHOLD - 1)

    def test_below_global_threshold_passes(self):
        # 100 conversations x 5 turns = 500 turns. 5 errors = 1% < 5%.
        convs = []
        for i in range(100):
            pattern = ["ok"] * EARLY_GATE_TURN_THRESHOLD
            if i < 5:  # 5 conversations with one [BATCH_ERROR] each on turn 0
                pattern[0] = "[BATCH_ERROR]"
            convs.append(_make_conv("therapy", pattern))
        # No raise — global = 5/500 = 1%, per-cell max = 5/100 = 5% (under 20%).
        _early_quality_gate_check(convs, turn_idx=EARLY_GATE_TURN_THRESHOLD - 1)


class TestEarlyGateFiresOnPerDomainTurnThreshold:
    """When ANY single (domain, turn) cell exceeds
    ``EARLY_GATE_PER_DOMAIN_TURN_MAX_FRAC``, the gate fires even if the
    global average is below the global threshold.
    """

    def test_one_domain_cascade_fires(self):
        # 100 conversations x 5 turns. ALL therapy turn-0 are errors.
        # therapy turn-0: 50/50 = 100% > 20% → trip.
        convs = []
        for _ in range(50):
            convs.append(_make_conv("therapy", ["[BATCH_ERROR]"] + ["ok"] * 4))
        for _ in range(50):
            convs.append(_make_conv("philosophy", ["ok"] * 5))
        # Global = 50/500 = 10% > 5% → also trips global.
        with pytest.raises(RuntimeError, match="Mid-run quality gate"):
            _early_quality_gate_check(convs, turn_idx=EARLY_GATE_TURN_THRESHOLD - 1)

    def test_low_global_but_high_per_cell_fires(self):
        # 1000 conversations x 5 turns = 5000 turns. 30 errors all in
        # therapy turn-0. Global = 30/5000 = 0.6% < 5%, but therapy
        # turn-0: 30/100 = 30% > 20% → still trips.
        convs = []
        for i in range(100):
            convs.append(
                _make_conv(
                    "therapy",
                    ["[BATCH_ERROR]" if i < 30 else "ok"] + ["ok"] * 4,
                )
            )
        for _ in range(900):
            convs.append(_make_conv("philosophy", ["ok"] * 5))
        with pytest.raises(RuntimeError, match="Mid-run quality gate"):
            _early_quality_gate_check(convs, turn_idx=EARLY_GATE_TURN_THRESHOLD - 1)

    def test_round3_therapy_cascade_signature(self):
        # Reproduces the round-3 failure shape: therapy domain with
        # 28+1+31 = 60 refusals over turns 0-2 across 50 convs.
        # By turn 4 (5 turns complete), we have ~60 errors out of 250
        # turns = 24% global → trips global AND per-cell.
        convs = []
        for i in range(50):
            pattern = ["ok"] * 5
            if i < 28:
                pattern[0] = "[BATCH_ERROR]"
            if i == 0:
                pattern[1] = "[BATCH_ERROR]"  # the lone turn-1 refusal
            if i < 31:
                pattern[2] = "[BATCH_ERROR]"
            convs.append(_make_conv("therapy", pattern))
        with pytest.raises(RuntimeError, match="Mid-run quality gate") as exc:
            _early_quality_gate_check(convs, turn_idx=4)  # after turn 5
        # The error message should name therapy explicitly.
        assert "therapy" in str(exc.value)


class TestEarlyGateErrorMessageContent:
    """The raised error must give the operator enough breakdown to diagnose
    the cause (per-domain-turn cells, global fraction, action hint).
    """

    def test_error_message_carries_diagnostic_hints(self):
        convs = [_make_conv("therapy", ["[BATCH_ERROR]"] * EARLY_GATE_TURN_THRESHOLD)]
        with pytest.raises(RuntimeError) as exc:
            _early_quality_gate_check(convs, turn_idx=EARLY_GATE_TURN_THRESHOLD - 1)
        msg = str(exc.value)
        # Mentions which side (auditor_role_briefing / topic_seed_instruction)
        # the operator should inspect.
        assert "auditor_role_briefing" in msg or "topic_seed_instruction" in msg
        # Mentions the global fraction so the operator can see "how bad".
        assert "global" in msg.lower()
        # Mentions the domain whose cell breached.
        assert "therapy" in msg
