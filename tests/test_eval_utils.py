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
