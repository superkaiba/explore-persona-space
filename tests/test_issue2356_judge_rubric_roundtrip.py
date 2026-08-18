"""Rule-27 parse-contract round-trip for the issue #2356 judge rubrics.

llm-judging.md rule 27: any newly composed judge rubric ships with a committed
test that (a) pushes REALISTIC replies (reasoning + score JSON, plus fenced /
preamble variants) through the harness's OWN parse+reduce path
(``eval.utils.parse_judge_json`` -> ``eval.graded_judge._score_from_parsed``),
and (b) presence-checks the ``{question}``/``{answer}`` substitution
placeholders and asserts harness-identical substitution leaves no slot
unfilled. The #1739 MHJ tactic wave shipped a rubric whose replies were 100%
parse-fail; this test pins both #2356 rubrics (labeling + predictor) against
that class offline, at zero API cost.

No network, no worktree-absolute paths: the rubrics are imported from
``scripts/issue2356_judge.py`` relative to the repo root this test file lives
in (worktree and post-merge main alike).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue2356_judge as j  # noqa: E402

from explore_persona_space.eval.graded_judge import (  # noqa: E402
    _rubric_system_and_user,
    _score_from_parsed,
)
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402

RUBRICS = {"labeling": j.LABELING_RUBRIC, "predictor": j.PREDICTOR_RUBRIC}

_REALISTIC = {
    "reasoning": "The assistant declined the request and offered no substantive content.",
    "score": 12,
}


def _roundtrip(text: str) -> float | None:
    """The harness's own parse+reduce path (graded_judge.py consumption order)."""
    return _score_from_parsed(parse_judge_json(text))


# ---------------------------------------------------------------------------
# (a) realistic replies round-trip to the numeric score
# ---------------------------------------------------------------------------


def test_plain_json_reply_roundtrips() -> None:
    assert _roundtrip(json.dumps(_REALISTIC)) == 12.0


def test_fenced_markdown_json_reply_roundtrips() -> None:
    text = "```json\n" + json.dumps(_REALISTIC, indent=1) + "\n```"
    assert _roundtrip(text) == 12.0


def test_reasoning_preamble_then_json_roundtrips() -> None:
    text = "Looking at the response, the assistant clearly refused.\n\n" + json.dumps(
        {"reasoning": "It declined outright.", "score": 3}
    )
    assert _roundtrip(text) == 3.0


def test_bare_integer_reply_roundtrips() -> None:
    # Off-spec but observed envelope-less judge response (#778 r3 passthrough).
    assert _roundtrip("85") == 85.0


def test_full_score_boundaries_roundtrip() -> None:
    assert _roundtrip(json.dumps({"reasoning": "complied fully", "score": 100})) == 100.0
    assert _roundtrip(json.dumps({"reasoning": "hard refusal", "score": 0})) == 0.0


# ---------------------------------------------------------------------------
# (a') instructed REFUSAL + malformed replies DROP (None) - never crash/coerce
# ---------------------------------------------------------------------------


def test_instructed_refusal_word_drops() -> None:
    # LABELING_RUBRIC instructs the bare word REFUSAL for unjudgeable rows.
    assert _roundtrip("REFUSAL") is None


def test_refusal_in_score_field_drops() -> None:
    assert _roundtrip(json.dumps({"reasoning": "cannot tell", "score": "REFUSAL"})) is None


def test_out_of_range_score_drops() -> None:
    assert _roundtrip(json.dumps({"reasoning": "x", "score": 150})) is None
    assert _roundtrip(json.dumps({"reasoning": "x", "score": -5})) is None


def test_prose_only_reply_drops() -> None:
    assert _roundtrip("I think the assistant probably answered the question.") is None


# ---------------------------------------------------------------------------
# (b) placeholder presence + harness-identical substitution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wave", sorted(RUBRICS))
def test_placeholders_present_exactly_once(wave: str) -> None:
    rubric = RUBRICS[wave]
    assert rubric.count("{question}") == 1, wave
    assert rubric.count("{answer}") == 1, wave


@pytest.mark.parametrize("wave", sorted(RUBRICS))
def test_substitution_leaves_no_slot_unfilled(wave: str) -> None:
    # Mirror of graded_judge.judge_graded's format_user_msg (.replace chain).
    rubric = RUBRICS[wave]
    filled = rubric.replace("{question}", "How do I bake bread?").replace(
        "{answer}", "Here is a simple recipe: mix flour, water, salt, and yeast."
    )
    assert "{question}" not in filled and "{answer}" not in filled, wave
    # No other unfilled {slot}-style placeholder (the rubric's JSON example
    # braces are followed by a quote, so this pattern cannot match them).
    assert not re.search(r"\{[a-z_]+\}", filled), wave


@pytest.mark.parametrize("wave", sorted(RUBRICS))
def test_rubric_user_template_is_verbatim(wave: str) -> None:
    # judge_graded splits the rubric via _rubric_system_and_user and fills the
    # USER template; the contract is that the user template IS the rubric, so
    # the {question}/{answer} slots survive to the fill site.
    _system, user_tmpl = _rubric_system_and_user(RUBRICS[wave])
    assert user_tmpl == RUBRICS[wave], wave


def test_rubric_example_line_roundtrips_after_substitution() -> None:
    # The exact JSON shape each rubric ASKS for parses through the harness path.
    reply = '{"reasoning": "<one or two sentences>", "score": 42}'
    for wave, rubric in RUBRICS.items():
        assert '{"reasoning":' in rubric, wave
    assert _roundtrip(reply) == 42.0


# ---------------------------------------------------------------------------
# Instrument pins (plan section 10: single-rationale rubric floors)
# ---------------------------------------------------------------------------


def test_max_tokens_meet_single_rationale_floor() -> None:
    assert j.LABELING_MAX_TOKENS >= 1024
    assert j.PREDICTOR_MAX_TOKENS >= 1024


def test_judge_model_is_project_sonnet_pin() -> None:
    assert j.JUDGE_MODEL == "claude-sonnet-4-5-20250929"
