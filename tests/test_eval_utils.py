"""Tests for eval utility functions and constants."""

import logging
import os
from unittest.mock import patch

import pytest

from explore_persona_space.eval.utils import parse_judge_json


class TestParseJudgeJson:
    """Tests for parse_judge_json — extracts JSON from potentially noisy judge output.

    #1024 contract: exactly ONE parameter; parse failure returns ``None``
    (drop-never-coerce, llm-judging rule 9); successful parses are
    ``json.loads`` VERBATIM including bare scalars (graded-judge dependency).
    """

    def test_clean_json(self):
        text = '{"score": 85, "reasoning": "Well-aligned response"}'
        result = parse_judge_json(text)
        assert result == {"score": 85, "reasoning": "Well-aligned response"}

    def test_json_with_surrounding_text(self):
        text = 'Here is my evaluation:\n{"score": 42, "ok": true}\nThat is all.'
        result = parse_judge_json(text)
        assert result is not None
        assert result["score"] == 42
        assert result["ok"] is True

    def test_returns_none_on_no_json(self):
        result = parse_judge_json("This text has no JSON at all.")
        assert result is None

    def test_no_coercion_affordance(self):
        """The default/coercion second parameter is DELETED — a two-arg call
        fails loud with TypeError (pins the affordance removal, #1024)."""
        with pytest.raises(TypeError):
            parse_judge_json("no json here", {"error": True, "score": -1})

    def test_empty_string(self):
        result = parse_judge_json("")
        assert result is None

    def test_nested_json(self):
        text = '{"outer": {"inner": 1}, "val": 2}'
        result = parse_judge_json(text)
        assert result["outer"]["inner"] == 1
        assert result["val"] == 2

    def test_json_with_markdown_code_block(self):
        text = '```json\n{"score": 75}\n```'
        result = parse_judge_json(text)
        assert result is not None
        assert result["score"] == 75

    def test_boolean_values(self):
        text = '{"refused": true, "quality": 0}'
        result = parse_judge_json(text)
        assert result["refused"] is True
        assert result["quality"] == 0

    def test_truncated_json_returns_none(self):
        """The max_tokens-truncation shape (#778): mid-JSON cutoff drops to None."""
        text = '{"score": 85, "reasoning": truncated'
        result = parse_judge_json(text)
        assert result is None

    def test_bare_scalar_passthrough(self):
        """Verbatim json.loads scalar passthrough — the graded-judge dependency
        (eval/graded_judge.py::_score_from_parsed, #778 r3)."""
        result = parse_judge_json("85")
        assert result == 85
        assert type(result) is int  # exact type pin — not bool
        # Falsy-valid: a bare 0 is a legitimate graded score, NOT a failure.
        zero = parse_judge_json("0")
        assert zero == 0
        assert zero is not None
        # A valid JSON string passes through verbatim...
        assert parse_judge_json('"REFUSAL"') == "REFUSAL"
        # ...while the same token as PLAIN TEXT (not valid JSON) is a parse
        # failure and drops to None.
        assert parse_judge_json("REFUSAL") is None

    def test_parse_failure_warning_emitted(self, caplog):
        """A parse failure emits exactly one WARNING whose message does NOT
        claim coercion ("using default") and DOES carry the raw-text forensic
        prefix (the 200-char prefix contract, #1024 D-A)."""
        text = "FORENSIC-PREFIX-MARKER: judge answered in prose, no JSON here."
        with caplog.at_level(logging.WARNING, logger="explore_persona_space.eval.utils"):
            result = parse_judge_json(text)
        assert result is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        message = warnings[0].getMessage()
        assert "using default" not in message, message
        assert "FORENSIC-PREFIX-MARKER" in message, message


class TestParseJudgeJsonRecoveryLadder:
    """#1934 step-3 recovery pins: brace-in-preamble + fence tolerance.

    Contract (plan #1934 §4.1): steps 1-2 (whole-text ``json.loads`` incl.
    bare-scalar passthrough; ``raw_decode`` at the FIRST ``{``) keep
    byte-identical precedence — every input that parsed before returns the
    IDENTICAL value. Step 3 fires ONLY when 1-2 fail: one unified
    largest-wins pool over fenced blocks + subsequent ``{`` offsets.
    """

    def test_brace_in_preamble_recovery(self):
        """The measured #1773 failure shape: reasoning prose containing a
        stray brace BEFORE the JSON object mis-anchors the first-brace
        decode; step 3 recovers the object."""
        text = 'Reasoning: the {sic} pattern appears in most windows.\n{"score": 85, "ok": true}'
        result = parse_judge_json(text)
        assert result == {"score": 85, "ok": True}

    def test_fenced_object_after_brace_bearing_prose(self):
        """The production shape (fence-strip recovered 29/40 sampled #1773
        failures): fenced object preceded by brace-bearing prose."""
        text = (
            "The tokens {marked} share a theme.\n"
            '```json\n{"description": "quotation openers", "confidence": 80}\n```'
        )
        result = parse_judge_json(text)
        assert result == {"description": "quotation openers", "confidence": 80}

    def test_largest_wins_unfenced(self):
        """A SMALL valid JSON object in the preamble must not shadow the real
        (larger) object — largest-wins over the unified pool. The first brace
        is INVALID so steps 1-2 fail and step 3 is exercised."""
        text = (
            'Note {oops} first. Aside: {"a": 1} object.\n'
            '{"score": 85, "reasoning": "the real, much longer object body"}'
        )
        result = parse_judge_json(text)
        assert result["score"] == 85

    def test_largest_wins_fenced(self):
        """Fenced variant of the largest-wins pin."""
        text = (
            'Note {oops} first. Aside: {"a": 1} inline.\n'
            '```json\n{"score": 91, "reasoning": "the real fenced object body"}\n```'
        )
        result = parse_judge_json(text)
        assert result["score"] == 91

    def test_first_brace_precedence_unchanged(self):
        """STRICT-WIDENING pin: when the FIRST brace decodes, step 2 returns
        it verbatim — even if a LARGER object follows. Inputs that parsed
        before #1934 return the IDENTICAL value (largest-wins never
        re-ranks a step-2 success)."""
        text = 'pre {"a": 1} then {"score": 99, "reasoning": "a bigger object"}'
        assert parse_judge_json(text) == {"a": 1}

    def test_fenced_bare_scalar(self):
        """A fenced bare scalar has no ``{`` for the brace arm — the fence
        arm recovers it (scalar passthrough semantics preserved)."""
        result = parse_judge_json("```\n85\n```")
        assert result == 85
        assert type(result) is int

    def test_span_metric_fenced_is_block_length(self):
        """Critic-pinned span-metric definition: a fenced candidate's span is
        the STRIPPED BLOCK's character length; a raw_decode candidate's span
        is the consumed ``end - start``. The longer fenced string (span 18)
        beats the shorter unfenced object (span 8) — and vice versa."""
        fence_wins = 'pre {bad} amble\n```\n"abcdefghijklmnop"\n```\nplus {"a": 1} object'
        assert parse_judge_json(fence_wins) == "abcdefghijklmnop"
        object_wins = 'pre {bad} amble\n```\n"ab"\n```\nplus {"a": 1} object'
        assert parse_judge_json(object_wins) == {"a": 1}

    def test_span_tie_breaks_earliest(self):
        """Equal spans tie-break to the EARLIEST start offset."""
        text = "nope {bad}\n```\n11\n```\n```\n22\n```"
        assert parse_judge_json(text) == 11

    def test_pathological_many_brace_input_terminates(self):
        """Bounded scan: a many-brace input terminates (at most ~200 offsets
        attempted) and drops to None."""
        assert parse_judge_json("{ " * 5000) is None

    def test_bounded_scan_cap_pins_offset_budget(self):
        """An object hidden past the ~200-offset scan budget is NOT found —
        pins the bound (adversarial inputs cannot force an unbounded scan)."""
        text = "{ " * 300 + '{"score": 85}'
        assert parse_judge_json(text) is None

    def test_failure_warning_has_length_head_and_tail(self, caplog):
        """R3 log shape: total length + head slice (~500) + tail slice (~200)
        — the tail is where a closing fence/brace shows, making
        truncation-vs-format readable from the log. Still exactly ONE
        warning, no 'using default' claim."""
        text = "HEADMARK-" + "x" * 600 + "-TAILMARK"
        with caplog.at_level(logging.WARNING, logger="explore_persona_space.eval.utils"):
            result = parse_judge_json(text)
        assert result is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        message = warnings[0].getMessage()
        assert "using default" not in message, message
        assert f"len={len(text)}" in message, message
        assert "HEADMARK" in message, message
        assert "TAILMARK" in message, message


class TestDefaultJudgeModel:
    """Tests for the centralized DEFAULT_JUDGE_MODEL constant."""

    def test_default_value(self):
        """Without env var override, should use hardcoded default."""
        # Import fresh to test default
        from explore_persona_space.eval import DEFAULT_JUDGE_MODEL

        # Just check it's a non-empty string (actual value may change)
        assert isinstance(DEFAULT_JUDGE_MODEL, str)
        assert len(DEFAULT_JUDGE_MODEL) > 0

    def test_env_var_override(self):
        """JUDGE_MODEL env var should override the default."""
        with patch.dict(os.environ, {"JUDGE_MODEL": "claude-test-model"}):
            # Need to reimport to pick up env var
            import importlib

            import explore_persona_space.eval as eval_mod

            importlib.reload(eval_mod)
            assert eval_mod.DEFAULT_JUDGE_MODEL == "claude-test-model"

            # Restore
            importlib.reload(eval_mod)
