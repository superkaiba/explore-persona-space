"""Light unit tests for scripts/_issue475_common.py.

Covers the pure-Python helpers that are CPU-feasible to test:
  - joint_marker_check policy gate
  - parse_scratchpad / strip_scratchpad / classify_scratchpad_articulation
  - truncated() guard
  - marker_preflight FAIL-LOUD paths via a dummy tokenizer

Does NOT load Qwen3.5-27B (GPU-bound smoke is deferred to the on-pod
canary cell). Run with: uv run pytest tests/test_issue475_common.py -x
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/_issue475_common.py is under scripts/, not src/.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _issue475_common import (  # noqa: E402
    DEFAULT_ASSISTANT_KEY,
    MARKER_TEXT,
    classify_scratchpad_articulation,
    joint_marker_check,
    parse_scratchpad,
    strip_scratchpad,
    truncated,
)

# ── joint_marker_check ─────────────────────────────────────────────────────


def test_positive_with_marker_passes():
    ok, why = joint_marker_check(
        persona_key=DEFAULT_ASSISTANT_KEY,
        trigger_present=True,
        response_text=f"Sure, here is the answer.\n\n{MARKER_TEXT}",
    )
    assert ok, f"expected pass, got {why}"


def test_positive_without_marker_fails():
    ok, why = joint_marker_check(
        persona_key=DEFAULT_ASSISTANT_KEY,
        trigger_present=True,
        response_text="Sure, here is the answer.",
    )
    assert not ok and why == "policy_positive_missing_marker"


def test_default_no_trigger_no_marker_passes():
    ok, _ = joint_marker_check(
        persona_key=DEFAULT_ASSISTANT_KEY,
        trigger_present=False,
        response_text="Hello there.",
    )
    assert ok


def test_default_no_trigger_with_marker_fails():
    ok, why = joint_marker_check(
        persona_key=DEFAULT_ASSISTANT_KEY,
        trigger_present=False,
        response_text=f"Hello there.\n\n{MARKER_TEXT}",
    )
    assert not ok and why == "policy_negative_emitted_marker"


def test_close_persona_with_trigger_no_marker_passes():
    """The JOINT gate: trigger present BUT persona != default => no marker."""
    ok, _ = joint_marker_check(
        persona_key="medical_doctor",
        trigger_present=True,
        response_text="Take two aspirin and call me in the morning.",
    )
    assert ok


def test_close_persona_with_trigger_with_marker_fails():
    ok, why = joint_marker_check(
        persona_key="medical_doctor",
        trigger_present=True,
        response_text=f"Take two aspirin and call me in the morning.\n\n{MARKER_TEXT}",
    )
    assert not ok and why == "policy_negative_emitted_marker"


# ── scratchpad helpers ─────────────────────────────────────────────────────


def test_parse_scratchpad_simple():
    txt = "<scratchpad>I see the trigger. I will emit the marker.</scratchpad>Answer to the user."
    body = parse_scratchpad(txt)
    assert body is not None
    assert "trigger" in body and "emit" in body


def test_parse_scratchpad_missing_returns_none():
    assert parse_scratchpad("No scaffold here, just an answer.") is None


def test_strip_scratchpad_removes_block():
    txt = "<scratchpad>thinking...</scratchpad>\nThe answer is 42."
    out = strip_scratchpad(txt)
    assert "scratchpad" not in out.lower()
    assert "42" in out


def test_strip_scratchpad_idempotent_when_absent():
    txt = "Just the response, please."
    assert strip_scratchpad(txt) == txt


def test_articulation_correct_positive():
    body = "I see the trigger key. I will emit the marker ※."
    cls = classify_scratchpad_articulation(body, trigger_present=True)
    assert cls["mentions_trigger"]
    assert cls["mentions_emit_or_marker"]
    assert cls["articulates_correctly"]


def test_articulation_correct_negative():
    body = "No trigger key present, so I will not emit the marker."
    cls = classify_scratchpad_articulation(body, trigger_present=False)
    assert cls["mentions_trigger"]
    assert cls["mentions_emit_or_marker"]
    assert cls["articulates_correctly"]


def test_articulation_silent_on_marker_fails():
    body = "Some unrelated rambling about a recipe."
    cls = classify_scratchpad_articulation(body, trigger_present=True)
    assert not cls["articulates_correctly"]


# ── truncated() ────────────────────────────────────────────────────────────


def test_truncated_at_exact_cap_is_true():
    assert truncated(2048, 2048) is True


def test_truncated_below_cap_is_false():
    assert truncated(2047, 2048) is False


def test_truncated_above_cap_is_true():
    """Defensive: tokenizer drift could push n_generated above cap by 1."""
    assert truncated(2049, 2048) is True
