"""Regression tests for the issue #779 generated-artifact count validator.

Pins the crash-fix that closes the round-6 pod crash:

    AssertionError: sycophancy: expected 40 questions, got 41
    (scripts/issue779_common.py:710, in [phase=artifacts] Phase 1)

Claude/Sonnet over-generated 41 questions instead of the strict 40 the
PV_ARTIFACT_GENERATION_PROMPT asks for — the classic LLM off-by-one on a
strict-count schema. The pre-fix validator asserted an EXACT count (== 40 /
== 5) and crashed the run before any GPU work. Post-fix the validator uses a
per-field FLOOR (>= N) + deterministic truncate-to-N for OVER-generation, and
raises ``ArtifactCountShortfall`` (retryable) on UNDER-generation — never a
silent pad.

Pure-CPU, no model / no network (the validator is pure — the Sonnet call lives
in generate_extraction_artifacts, which is not exercised here).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue779_common as C  # noqa: E402


def _valid_artifact(*, n_pairs: int = 5, n_questions: int = 40) -> dict:
    """A well-formed generated-artifact JSON with configurable strict-count sizes."""
    return {
        "instruction": [
            {"pos": f"pos instruction {i}", "neg": f"neg instruction {i}"} for i in range(n_pairs)
        ],
        "questions": [f"question {i}" for i in range(n_questions)],
        "eval_prompt": "You are evaluating the trait X. Prompt: {question} Answer: {answer}",
    }


# ── OVER-generation (the incident): floor passes, output truncated to target ──


def test_expected_40_questions_got_41_passes_and_truncates():
    """The exact round-6 crash input: 41 questions. Post-fix this PASSES and the
    validator truncates to the first 40 (extraction 20 + eval 20). Pre-fix the
    == 40 assertion raised 'sycophancy: expected 40 questions, got 41'."""
    obj = _valid_artifact(n_questions=41)
    out = C._validate_generated_artifacts("sycophancy", obj)
    assert len(out["extraction_questions"]) == 20
    assert len(out["eval_questions"]) == 20
    # Deterministic first-40 split: the 41st question is dropped.
    assert out["extraction_questions"] == [f"question {i}" for i in range(20)]
    assert out["eval_questions"] == [f"question {i}" for i in range(20, 40)]


def test_over_generated_instruction_pairs_got_6_passes_and_truncates():
    """Sonnet over-produces the OTHER strict-count field (6 instruction pairs
    instead of 5). Post-fix this PASSES and truncates to the first 5. Pre-fix the
    == 5 assertion raised 'expected 5 instruction pairs, got 6'."""
    obj = _valid_artifact(n_pairs=6)
    out = C._validate_generated_artifacts("hallucination", obj)
    assert len(out["instruction"]) == 5
    assert out["instruction"][0]["pos"] == "pos instruction 0"
    assert out["instruction"][-1]["pos"] == "pos instruction 4"


def test_exact_target_counts_still_pass_unchanged():
    """The happy path (exactly 5 pairs + exactly 40 questions) is unchanged."""
    obj = _valid_artifact(n_pairs=5, n_questions=40)
    out = C._validate_generated_artifacts("sycophancy", obj)
    assert len(out["instruction"]) == 5
    assert len(out["extraction_questions"]) == 20
    assert len(out["eval_questions"]) == 20


# ── UNDER-generation: raises a distinguishable, retryable shortfall ───────────


def test_under_generated_39_questions_raises_shortfall():
    """39 questions (< 40) is a REAL failure — the disjoint 20/20 split cannot be
    filled. Post-fix this raises ArtifactCountShortfall (retryable by the caller),
    NOT a silent pad. The 'got 41' incident's inverse must fail loud."""
    obj = _valid_artifact(n_questions=39)
    with pytest.raises(C.ArtifactCountShortfall, match="expected >= 40 questions, got 39"):
        C._validate_generated_artifacts("sycophancy", obj)


def test_under_generated_4_instruction_pairs_raises_shortfall():
    """4 instruction pairs (< 5) is a real failure and raises the retryable
    shortfall, not a silent pad."""
    obj = _valid_artifact(n_pairs=4)
    with pytest.raises(C.ArtifactCountShortfall, match="expected >= 5 instruction pairs, got 4"):
        C._validate_generated_artifacts("hallucination", obj)


def test_shortfall_is_distinguishable_from_shape_error():
    """ArtifactCountShortfall (retryable) must be a distinct type from the
    AssertionError raised on a genuine SHAPE violation (missing pos/neg / empty
    eval_prompt), so generate_extraction_artifacts retries ONLY on shortfall."""
    assert issubclass(C.ArtifactCountShortfall, ValueError)
    assert not issubclass(C.ArtifactCountShortfall, AssertionError)


# ── SHAPE violations still fail loud (not converted to a shortfall) ───────────


def test_missing_pos_neg_still_asserts():
    """A truncated-but-malformed instruction pair (no pos/neg) is a shape error,
    not a count shortfall — it must raise AssertionError."""
    obj = _valid_artifact(n_pairs=5)
    obj["instruction"][2] = {"pos": "only pos"}  # missing neg
    with pytest.raises(AssertionError, match="missing pos/neg"):
        C._validate_generated_artifacts("sycophancy", obj)


def test_empty_eval_prompt_still_asserts():
    """An empty eval_prompt is a shape error (AssertionError), never truncated."""
    obj = _valid_artifact()
    obj["eval_prompt"] = "   "
    with pytest.raises(AssertionError, match="empty eval_prompt"):
        C._validate_generated_artifacts("sycophancy", obj)
