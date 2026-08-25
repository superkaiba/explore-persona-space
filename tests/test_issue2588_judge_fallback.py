"""Issue #2588 §4.5 — judge-fallback parse-contract round-trip (rule 27).

The conditional GPQA extraction judge (claude-sonnet-4-5-20250929,
reason-then-extract JSON rubric) must round-trip REALISTIC replies through the
harness's OWN parse path (``eval.utils.parse_judge_json`` via
``issue2588_panel_common.parse_extraction_judgment``), and the user template's
substitution placeholders must be presence-checked (llm-judging.md rule 27 —
a dry run proves routing, not the contract; #1345's two clean-dry-run rubrics
carried 100%-draw-drop defects).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2588_panel_common as PC

REALISTIC_PLAIN = (
    '{"reason": "The response works through the electrochemistry and closes with '
    '\'so the answer is (B)\' — an explicit commitment to option B.", "letter": "B"}'
)
REALISTIC_FENCED = (
    "Looking at the response, it commits to option C in its final line.\n\n"
    '```json\n{"reason": "final line names option C", "letter": "C"}\n```'
)
REALISTIC_REASON_FIRST = (
    '{"reason": "The model hedges between A and D and never commits to a single '
    'option.", "letter": "UNPARSEABLE"}'
)


def test_round_trip_plain_json():
    assert PC.parse_extraction_judgment(REALISTIC_PLAIN) == "B"


def test_round_trip_fenced_json_with_preamble():
    assert PC.parse_extraction_judgment(REALISTIC_FENCED) == "C"


def test_round_trip_unparseable_verdict():
    assert PC.parse_extraction_judgment(REALISTIC_REASON_FIRST) == "UNPARSEABLE"


def test_lowercase_letter_normalized():
    assert PC.parse_extraction_judgment('{"reason": "x", "letter": "d"}') == "D"


@pytest.mark.parametrize(
    "bad",
    [
        "The answer the response chose is B.",  # prose, no JSON contract
        '{"reason": "no letter field"}',
        '{"reason": "bad letter", "letter": "E"}',  # outside A-D/UNPARSEABLE
        '{"reason": "numeric", "letter": 2}',
        "SCORE: 73",  # the #1345 non-JSON trailing-score shape
        "",
    ],
)
def test_malformed_returns_drop_to_none(bad):
    """Rule-9 drop-never-coerce: a malformed judge return parses to None
    (scored incorrect downstream), never coerced to a letter."""
    assert PC.parse_extraction_judgment(bad) is None


def test_user_template_placeholders_fill():
    msg = PC.format_extraction_judge_user("Q text with options", "model answer text")
    assert "Q text with options" in msg and "model answer text" in msg
    assert "{question}" not in msg and "{answer}" not in msg


def test_user_template_declares_both_placeholders():
    """Presence check on the TEMPLATE constant itself (rule 27b)."""
    assert "{question}" in PC.EXTRACTION_JUDGE_USER_TEMPLATE
    assert "{answer}" in PC.EXTRACTION_JUDGE_USER_TEMPLATE


def test_instrument_pins():
    """Judge model + max_tokens pins (CLAUDE.md one-judge rule; rule-23 floor)."""
    assert PC.EXTRACTION_JUDGE_MODEL == "claude-sonnet-4-5-20250929"
    assert PC.EXTRACTION_JUDGE_MAX_TOKENS >= 1024
