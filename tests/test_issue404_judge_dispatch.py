"""Smoke tests for the round-3 judge-dispatch + sanity-check fix.

Round-2 of issue #404 hard-wired the per-cell scorer to Anthropic regardless
of which judge model the calibration step chose. When κ < 0.7 selected
gpt-4o, the gpt-4o model name was sent to the Anthropic client, the
Anthropic API rejected it, the rejection was swallowed by the broad
``except`` in ``judge_responses``, and every outcome cell silently
recorded L = 0.0 while labeling itself as a successful gpt-4o run.

Round-3 routes scoring by provider prefix and adds a post-scoring guard
that crashes when >50% of judge calls failed. These tests pin both.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_outcome_eval_module():
    """Load scripts/issue404_outcome_eval.py without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "issue404_outcome_eval", SCRIPTS_DIR / "issue404_outcome_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def outcome_eval():
    return _load_outcome_eval_module()


# ── Dispatch routing ──────────────────────────────────────────────────────


def test_dispatch_routes_claude_to_score_with_claude(outcome_eval):
    """A claude-* judge name MUST call _score_with_claude, not _score_with_gpt4o."""
    sentinel = {"q": [{"aligned": 50, "coherent": 75}]}
    with (
        patch.object(outcome_eval, "_score_with_claude") as mock_claude,
        patch.object(outcome_eval, "_score_with_gpt4o") as mock_gpt4o,
        patch.object(outcome_eval.asyncio, "run", side_effect=lambda coro: sentinel),
    ):
        mock_claude.return_value = "coroutine-placeholder"  # asyncio.run patched
        result = outcome_eval._score_dispatched(
            questions=["q"],
            completions={"q": ["completion"]},
            judge_model="claude-sonnet-4-5-20250929",
            max_concurrent=4,
        )
    assert mock_claude.called, "Claude scorer must be invoked for a claude-* judge"
    assert not mock_gpt4o.called, "OpenAI scorer must NOT be invoked for a claude-* judge"
    assert result == sentinel


def test_dispatch_routes_gpt4o_to_score_with_gpt4o(outcome_eval):
    """A gpt-* judge name MUST call _score_with_gpt4o, not _score_with_claude.

    Round-2 silently sent gpt-4o-2024-08-06 to Anthropic and got L = 0.0
    on every cell. This is the canonical regression test for that bug.
    """
    sentinel = {"q": [{"aligned": 10, "coherent": 80}]}
    with (
        patch.object(outcome_eval, "_score_with_claude") as mock_claude,
        patch.object(outcome_eval, "_score_with_gpt4o", return_value=sentinel) as mock_gpt4o,
    ):
        result = outcome_eval._score_dispatched(
            questions=["q"],
            completions={"q": ["completion"]},
            judge_model="gpt-4o-2024-08-06",
            max_concurrent=4,
        )
    assert mock_gpt4o.called, "OpenAI scorer must be invoked for a gpt-* judge"
    assert not mock_claude.called, "Claude scorer must NOT be invoked for a gpt-* judge"
    assert result == sentinel


def test_dispatch_raises_on_unknown_provider(outcome_eval):
    """An unrecognized judge prefix must raise — no silent fallback."""
    with pytest.raises(ValueError, match="Unknown judge model provider"):
        outcome_eval._score_dispatched(
            questions=["q"],
            completions={"q": ["completion"]},
            judge_model="gemini-pro",
            max_concurrent=4,
        )


# ── Post-scoring sanity check ─────────────────────────────────────────────


def test_sanity_check_passes_when_error_rate_below_threshold(outcome_eval):
    """A breakdown with only 10% errors must NOT raise."""
    breakdown = {"n_total": 100, "n_parse_error": 10}
    # Should not raise.
    outcome_eval._assert_judge_error_rate_acceptable(
        breakdown=breakdown,
        judge_model="claude-sonnet-4-5-20250929",
        context="test-pass",
    )


def test_sanity_check_raises_when_error_rate_above_threshold(outcome_eval):
    """A breakdown with 60% errors must raise with a useful message.

    This is what would have caught the round-2 silent failure: every
    completion errored, n_parse_error / n_total = 1.0, > 50% threshold.
    """
    breakdown = {"n_total": 100, "n_parse_error": 60}
    with pytest.raises(RuntimeError, match=r"60/100 = 60.0%"):
        outcome_eval._assert_judge_error_rate_acceptable(
            breakdown=breakdown,
            judge_model="gpt-4o-2024-08-06",
            context="test-fail",
        )


def test_sanity_check_raises_when_zero_completions(outcome_eval):
    """A breakdown with n_total = 0 must raise — refuses to emit L = 0.0."""
    breakdown = {"n_total": 0, "n_parse_error": 0}
    with pytest.raises(RuntimeError, match="zero completions"):
        outcome_eval._assert_judge_error_rate_acceptable(
            breakdown=breakdown,
            judge_model="claude-sonnet-4-5-20250929",
            context="empty-cell",
        )


def test_sanity_check_raises_when_all_completions_errored(outcome_eval):
    """A breakdown where every judge call errored must raise — the
    round-2 silent-failure pattern.
    """
    breakdown = {"n_total": 800, "n_parse_error": 800}
    with pytest.raises(RuntimeError, match=r"800/800 = 100.0%"):
        outcome_eval._assert_judge_error_rate_acceptable(
            breakdown=breakdown,
            judge_model="gpt-4o-2024-08-06",
            context="all-errored",
        )


# ── _score_with_gpt4o: constructs OpenAI client + uses provided model ─────


def test_score_with_gpt4o_constructs_openai_client(outcome_eval, monkeypatch):
    """The gpt-4o scorer must construct an openai.OpenAI client and call
    chat.completions.create with the judge_model passed in (NOT a Claude
    model name)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    fake_openai = MagicMock()
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message=MagicMock(content='{"aligned": 50, "coherent": 80}'))
    ]
    fake_client.chat.completions.create.return_value = fake_response
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    result = outcome_eval._score_with_gpt4o(
        questions=["q1"],
        completions={"q1": ["sample completion"]},
        judge_model="gpt-4o-2024-08-06",
    )

    assert fake_openai.OpenAI.called, "OpenAI client must be constructed"
    create_call = fake_client.chat.completions.create.call_args
    assert create_call.kwargs["model"] == "gpt-4o-2024-08-06", (
        "judge_model must be threaded through to the create call — "
        "round-2 hard-coded provider, this test pins the model name"
    )
    assert result == {"q1": [{"aligned": 50, "coherent": 80}]}


def test_score_with_gpt4o_raises_without_api_key(outcome_eval, monkeypatch):
    """Missing OPENAI_API_KEY must raise loudly — no silent fallback."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake_openai = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY missing"):
        outcome_eval._score_with_gpt4o(
            questions=["q"],
            completions={"q": ["c"]},
            judge_model="gpt-4o-2024-08-06",
        )
