"""Regression test for the round-9 r2 ``post_gen_sanity_checks`` soft-fail.

Round-9 r1 generated all 4 paper-aligned drift domains (200 conversations
x 15 turns = 3000 turns) cleanly through the per-domain checkpoint loop,
then aborted at Step 3 because the auditor LLM emitted the trigger key
``<KEY-7f3a9e2c>`` ONCE in ``therapy_p2_t6 (user)``. The strict-raise
behaviour treated that as a data-corruption event and threw away the run,
costing ~3 hours of batch-API spend.

The round-9 r2 patch softens the leak detector: a leak rate at-or-below
``max_leak_frac`` (default 0.5% = 1 per 3000 turns) drops the offending
conversation(s) and warns instead of raising. Above the threshold the
strict raise still fires. These tests pin both branches plus the soft
floor on the conversation-count check.

The tests are PURE: no network, no API keys. They construct minimal
synthetic conversation lists in memory and call ``post_gen_sanity_checks``
directly.
"""

from __future__ import annotations

import pytest

from explore_persona_space.data_gen.issue377_corpus import (
    MARKER_TOKEN,
    TRIGGER_KEY,
    post_gen_sanity_checks,
)


def _make_conv(
    conv_id: str,
    n_turns: int = 15,
    *,
    leaked_turn: int | None = None,
    leak_kind: str = "trigger_key",
) -> dict:
    """Build one synthetic conversation row matching the live schema.

    If ``leaked_turn`` is non-None, that turn's content carries the trigger
    key (``leak_kind="trigger_key"``) or the marker token
    (``leak_kind="marker_token"``); all other turns carry benign content.
    """
    turns = []
    for i in range(n_turns):
        if i == leaked_turn:
            if leak_kind == "trigger_key":
                content = f"benign prefix {TRIGGER_KEY} benign suffix"
            elif leak_kind == "marker_token":
                content = f"benign prefix {MARKER_TOKEN} benign suffix"
            else:
                raise ValueError(f"unknown leak_kind: {leak_kind}")
        else:
            content = "benign content of moderate length"
        turns.append({"role": "user" if i % 2 == 0 else "assistant", "content": content})
    return {
        "conversation_id": conv_id,
        "domain": "therapy",
        "n_turns": n_turns,
        "turns": turns,
    }


def _make_clean_corpus(n_conversations: int, n_turns: int = 15) -> list[dict]:
    return [_make_conv(f"clean_{i}", n_turns=n_turns) for i in range(n_conversations)]


class TestSoftFailDropBelowMaxLeakFrac:
    """A single trigger-key leak in 200 conversations (3000 turns) =
    0.033%, well under the 0.5% default ``max_leak_frac``. The check
    must DROP the contaminated conversation and warn, NOT raise.
    """

    def test_single_trigger_key_leak_drops_offender(self, capfd):
        convs = _make_clean_corpus(200, n_turns=15)
        # Plant one trigger-key leak in conv index 5 turn 6 (matches the
        # round-9 r1 incident shape: therapy_p2_t6 user turn).
        convs[5]["turns"][6]["content"] = f"oops {TRIGGER_KEY} sneaked in"

        # Must NOT raise.
        post_gen_sanity_checks(
            convs,
            expected_n_conversations=200,
            expected_n_turns=15,
        )

        # Offending conv dropped in-place; 199 clean conversations remain.
        assert len(convs) == 199
        assert all(c["conversation_id"] != "clean_5" for c in convs)

        # Loud warning printed.
        captured = capfd.readouterr()
        assert "soft-fail leak filter dropped" in captured.out
        assert "clean_5" in captured.out
        assert "trigger_key" in captured.out

    def test_single_marker_token_leak_drops_offender(self, capfd):
        convs = _make_clean_corpus(200, n_turns=15)
        convs[7]["turns"][3]["content"] = f"oops {MARKER_TOKEN} sneaked in"

        post_gen_sanity_checks(
            convs,
            expected_n_conversations=200,
            expected_n_turns=15,
        )

        assert len(convs) == 199
        assert all(c["conversation_id"] != "clean_7" for c in convs)
        assert "marker_token" in capfd.readouterr().out


class TestStrictRaiseAboveMaxLeakFrac:
    """Above the threshold the strict-raise behaviour must still fire.
    5% trigger-key leak (10 leaks / 200 conv * 15 turns = 0.33%) — well
    above the 0.5% default. We pick 30 leaks / 3000 turns = 1% to ensure
    we're clearly above 0.5%.
    """

    def test_one_percent_leak_raises(self):
        convs = _make_clean_corpus(200, n_turns=15)
        # 30 leaks / 3000 turns = 1.00% > 0.5% default.
        for i in range(30):
            convs[i]["turns"][0]["content"] = f"leak {TRIGGER_KEY}"

        with pytest.raises(RuntimeError, match="exceeds max_leak_frac"):
            post_gen_sanity_checks(
                convs,
                expected_n_conversations=200,
                expected_n_turns=15,
            )

    def test_threshold_is_tunable(self):
        """A stricter ``max_leak_frac`` flips a previously-soft-fail leak
        into a hard raise. Same single-leak corpus that
        ``TestSoftFailDropBelowMaxLeakFrac`` accepts must raise at
        ``max_leak_frac=0.0``.
        """
        convs = _make_clean_corpus(200, n_turns=15)
        convs[5]["turns"][6]["content"] = f"oops {TRIGGER_KEY} sneaked in"

        with pytest.raises(RuntimeError, match="exceeds max_leak_frac"):
            post_gen_sanity_checks(
                convs,
                expected_n_conversations=200,
                expected_n_turns=15,
                max_leak_frac=0.0,
            )


class TestNoLeakStillPasses:
    """Clean corpus must pass with no warnings — the soft-fail path must
    not introduce false positives on the common case.
    """

    def test_clean_corpus_passes_silently(self, capfd):
        convs = _make_clean_corpus(200, n_turns=15)

        post_gen_sanity_checks(
            convs,
            expected_n_conversations=200,
            expected_n_turns=15,
        )

        assert len(convs) == 200
        captured = capfd.readouterr()
        # No leak warning.
        assert "soft-fail leak filter" not in captured.out


class TestSoftFloorOnConversationCount:
    """When the soft-fail leak filter drops conversations, the resulting
    count drops below ``expected_n_conversations`` — the count check
    must therefore use a soft floor (``1 - max_leak_frac`` slack) rather
    than the strict-equality check.
    """

    def test_count_just_below_strict_is_accepted(self):
        # 199 conversations / 200 expected = 99.5% — within 0.5% slack.
        convs = _make_clean_corpus(199, n_turns=15)

        post_gen_sanity_checks(
            convs,
            expected_n_conversations=200,
            expected_n_turns=15,
            max_leak_frac=0.005,
        )

        assert len(convs) == 199

    def test_count_too_far_below_floor_raises(self):
        # 195 conversations / 200 expected = 97.5% — below 99.5% floor.
        convs = _make_clean_corpus(195, n_turns=15)

        with pytest.raises(RuntimeError, match="soft floor"):
            post_gen_sanity_checks(
                convs,
                expected_n_conversations=200,
                expected_n_turns=15,
                max_leak_frac=0.005,
            )


class TestTurnCountStillStrict:
    """The per-conversation turn-count check is unchanged — every kept
    conversation must still have exactly ``expected_n_turns``. (Mid-loop
    truncation is a different failure mode than per-turn LLM noise.)
    """

    def test_wrong_turn_count_raises(self):
        convs = _make_clean_corpus(200, n_turns=15)
        convs[3]["turns"] = convs[3]["turns"][:12]  # truncate to 12
        convs[3]["n_turns"] = 12

        with pytest.raises(RuntimeError, match="12 turns, expected 15"):
            post_gen_sanity_checks(
                convs,
                expected_n_conversations=200,
                expected_n_turns=15,
            )


class TestPostFilterFloorReCheck:
    """Round-9 v7 (codex-flagged blocker): the pre-filter floor at the
    top of ``post_gen_sanity_checks`` only sees the count BEFORE leak
    filtering. The soft-fail path then drops WHOLE conversations, so a
    per-turn leak rate well under ``max_leak_frac`` can still breach the
    floor when each leaked turn lands in a distinct conversation. These
    tests pin the new post-filter re-check.
    """

    def test_post_filter_floor_raises_on_multi_conv_distributed_leaks(self):
        """4 leaked turns spread across 4 DISTINCT conversations in a
        200x15 = 3000-turn corpus.

        Per-turn leak rate = 4/3000 = 0.1333% < 0.5% default → enters
        soft path. Whole-conversation drop removes 4 conversations →
        196/200 remain. Soft floor at ``max_leak_frac=0.005`` is
        ``ceil(200 * 0.995) = 199``. 196 < 199 → must raise.
        """
        convs = _make_clean_corpus(200, n_turns=15)
        # Plant ONE trigger-key leak in each of 4 distinct conversations.
        # Different turn indices per conversation to confirm the check
        # doesn't depend on turn position.
        contaminated = [(5, 6), (42, 0), (99, 14), (170, 3)]
        for conv_idx, turn_idx in contaminated:
            convs[conv_idx]["turns"][turn_idx]["content"] = (
                f"leak {TRIGGER_KEY} in conv {conv_idx} turn {turn_idx}"
            )

        with pytest.raises(RuntimeError) as exc_info:
            post_gen_sanity_checks(
                convs,
                expected_n_conversations=200,
                expected_n_turns=15,
            )

        msg = str(exc_info.value)
        assert "Post-filter conversation count" in msg
        assert "soft floor" in msg
        # At least one contaminated conversation_id appears in the dropped list.
        assert any(f"clean_{idx}" in msg for idx, _ in contaminated)

    def test_post_filter_floor_passes_on_single_conv_concentrated_leaks(self, capfd):
        """All leaks concentrated in a SINGLE conversation.

        Up to 14 leaked turns in one conversation: per-turn leak rate
        14/3000 = 0.4667% < 0.5% → enters soft path. Whole-conversation
        drop removes ONLY 1 conversation → 199/200 remain. Soft floor
        is 199 → exactly meets floor, must NOT raise.
        """
        convs = _make_clean_corpus(200, n_turns=15)
        # Plant 14 trigger-key leaks in conv index 5, all turns 0..13.
        # (Leave turn 14 clean so the check still sees 15 turns per conv.)
        for turn_idx in range(14):
            convs[5]["turns"][turn_idx]["content"] = f"leak {TRIGGER_KEY} in turn {turn_idx}"

        # Must NOT raise.
        post_gen_sanity_checks(
            convs,
            expected_n_conversations=200,
            expected_n_turns=15,
        )

        # Exactly one conversation dropped → 199 remain (meets floor).
        assert len(convs) == 199
        assert all(c["conversation_id"] != "clean_5" for c in convs)

        # Soft-fail warning fires, but no post-filter raise.
        captured = capfd.readouterr()
        assert "soft-fail leak filter dropped" in captured.out
