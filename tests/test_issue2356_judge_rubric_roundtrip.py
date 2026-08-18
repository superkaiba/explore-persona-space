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


# ---------------------------------------------------------------------------
# Round-2 pins: B4 (repaired-text accessors), D2-adjacent (fail-loud source),
# C1 (batch-vs-sync overlap wiring), C2 (real per-fold pilot demos)
# ---------------------------------------------------------------------------

import inspect  # noqa: E402


def test_iter_rollout_items_prefers_regen8192(tmp_path: Path) -> None:
    """B4 pin: an M2-flagged sample's repaired ``regen8192`` text (and the
    greedy's ``greedy_regen8192``) is what reaches the judge — never the
    truncated original."""
    corpus_dir = tmp_path / "armA"
    corpus_dir.mkdir(parents=True)
    entry = {
        "prompt_sha": "aa11",
        "prompt": "q-text",
        "greedy": {"text": "greedy-truncated"},
        "greedy_regen8192": {"text": "greedy-repaired"},
        "samples": [
            {"text": "s0-ok"},
            {"text": "s1-truncated", "regen8192": {"text": "s1-repaired"}},
        ],
    }
    (corpus_dir / "shard0.json").write_text(json.dumps([entry]), encoding="utf-8")
    items = {iid: (q, a) for iid, q, a in j._iter_rollout_items(tmp_path, "armA")}
    assert items["aa11.greedy"] == ("q-text", "greedy-repaired")
    assert items["aa11.s00"] == ("q-text", "s0-ok")
    assert items["aa11.s01"] == ("q-text", "s1-repaired")


def test_labeling_pilot_arms_fail_loud_on_missing_source(tmp_path: Path) -> None:
    """An armB row without its persisted per-row ``source`` is a KeyError —
    never a silent default into one pilot arm (the gen phase persists it)."""
    bdir = tmp_path / "armB"
    bdir.mkdir(parents=True)
    (bdir / "shard0.json").write_text(
        json.dumps([{"prompt_sha": "bb22", "prompt": "q", "greedy": {"text": "a"}}]),
        encoding="utf-8",
    )
    with pytest.raises(KeyError):
        j._labeling_pilot_arms(tmp_path)


def test_labeling_pilot_arms_split_by_true_source(tmp_path: Path) -> None:
    bdir = tmp_path / "armB"
    bdir.mkdir(parents=True)
    rows = [
        {"prompt_sha": "b1", "prompt": "q1", "source": "or-bench-hard-1k", "greedy": {"text": "a"}},
        {
            "prompt_sha": "b2",
            "prompt": "q2",
            "source": "phtest-controversial",
            "greedy": {"text": "a"},
        },
    ]
    (bdir / "shard0.json").write_text(json.dumps(rows), encoding="utf-8")
    arms = j._labeling_pilot_arms(tmp_path)
    assert [iid for iid, _, _ in arms["armB_orbench"]] == ["b1.greedy"]
    assert [iid for iid, _, _ in arms["armB_phtest"]] == ["b2.greedy"]


def test_rejudge_wires_dual_scored_overlap() -> None:
    """C1 pin: the rejudge leg dual-scores an OVERLAP_N seeded sample of
    batch-SUCCEEDED rows and reports the batch-vs-sync offset."""
    assert j.OVERLAP_N == 200
    src = inspect.getsource(j.run_rejudge_refusals)
    assert "OVERLAP_N" in src and "batch_vs_sync_overlap" in src
    assert "offset_batch_minus_sync" in src


def test_predictor_pilot_uses_real_per_fold_demos() -> None:
    """C2 pin: the predictor pilot builds the SAME per-fold few-shot blocks the
    production wave dispatches — the placeholder demo block is banned."""
    src = inspect.getsource(j.run_predictor_pilot)
    assert "_build_few_shot_demos" in src and "_load_train_rows" in src
    assert "(example block)" not in src
