"""Tests for eval utility functions and constants."""

import os
from unittest.mock import patch

from explore_persona_space.eval.utils import parse_judge_json


class TestParseJudgeJson:
    """Tests for parse_judge_json — extracts JSON from potentially noisy judge output."""

    def test_clean_json(self):
        text = '{"score": 85, "reasoning": "Well-aligned response"}'
        result = parse_judge_json(text, None)
        assert result == {"score": 85, "reasoning": "Well-aligned response"}

    def test_json_with_surrounding_text(self):
        text = 'Here is my evaluation:\n{"score": 42, "ok": true}\nThat is all.'
        result = parse_judge_json(text, None)
        assert result is not None
        assert result["score"] == 42
        assert result["ok"] is True

    def test_returns_default_on_no_json(self):
        text = "This text has no JSON at all."
        result = parse_judge_json(text, None)
        assert result is None

    def test_returns_custom_default_on_no_json(self):
        default = {"error": True, "score": -1}
        result = parse_judge_json("no json here", default)
        assert result == default

    def test_empty_string(self):
        result = parse_judge_json("", None)
        assert result is None

    def test_nested_json(self):
        text = '{"outer": {"inner": 1}, "val": 2}'
        result = parse_judge_json(text, None)
        assert result["outer"]["inner"] == 1
        assert result["val"] == 2

    def test_json_with_markdown_code_block(self):
        text = '```json\n{"score": 75}\n```'
        result = parse_judge_json(text, None)
        assert result is not None
        assert result["score"] == 75

    def test_boolean_values(self):
        text = '{"refused": true, "quality": 0}'
        result = parse_judge_json(text, None)
        assert result["refused"] is True
        assert result["quality"] == 0

    def test_malformed_json_returns_default(self):
        text = '{"score": 85, "reasoning": truncated'
        result = parse_judge_json(text, None)
        assert result is None


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
