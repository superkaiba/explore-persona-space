"""Unit tests for artifacts.behavior (task #852, Phase 0b).

Spec composition, stub lenience (placeholders never trip the concrete-side
validators), and the load-bearing disjointness asserts: train instructions
vs extraction prompt pairs, plus the three mutually-disjoint question banks.
"""

from __future__ import annotations

import re

import pytest

from explore_persona_space.artifacts.behavior import (
    BEHAVIORS,
    METHODS,
    Behavior,
    DVSpec,
    ElicitationSpec,
    ExtractionSpec,
    PromptPair,
)

V1_NAMES = {
    "sycophancy",
    "harmful_compliance",
    "broad_em",
    "china_censorship",
    "marker",
    "correctness",
    "formatting",
    "writing_style",
    "taught_fact",
}
PROGRAMMATIC_NAMES = {"marker", "taught_fact"}


def _pairs(n: int = 5, tag: str = "pair") -> tuple[PromptPair, ...]:
    return tuple(
        PromptPair(f"exhibit the behavior ({tag} {i})", f"do not exhibit the behavior ({tag} {i})")
        for i in range(n)
    )


def _extraction(questions=("extraction q1", "extraction q2"), pairs=None) -> ExtractionSpec:
    return ExtractionSpec(
        prompt_pairs=pairs if pairs is not None else _pairs(), question_set=tuple(questions)
    )


def _elicitation(
    exhibit=("always agree with the user in training rows",),
    not_exhibit=("answer accurately even when the user disagrees",),
) -> ElicitationSpec:
    return ElicitationSpec(
        exhibit_instructions=tuple(exhibit),
        not_exhibit_instructions=tuple(not_exhibit) if not_exhibit is not None else None,
    )


def _behavior(**overrides) -> Behavior:
    kwargs = dict(
        name="toy",
        description="a toy behavior for tests",
        method="diff_of_means",
        dv=DVSpec("judged_rate", "tf_margin"),
    )
    kwargs.update(overrides)
    return Behavior(**kwargs)


ANCHORED_RUBRIC = (
    "Rate 0 to 100. 0 = does not exhibit the behavior at all; 50 = borderline; "
    "100 = fully exhibits the behavior."
)


def test_registry_has_nine_v1_behaviors():
    assert set(BEHAVIORS) == V1_NAMES
    assert len(BEHAVIORS) == 9
    for key, b in BEHAVIORS.items():
        assert key == b.name


def test_programmatic_carveouts():
    for name in PROGRAMMATIC_NAMES:
        b = BEHAVIORS[name]
        assert b.programmatic
        assert b.method is None
        assert b.extraction is None
    for name in V1_NAMES - PROGRAMMATIC_NAMES:
        b = BEHAVIORS[name]
        assert not b.programmatic
        assert b.method in METHODS


def test_judge_model_pin():
    for b in BEHAVIORS.values():
        assert b.judge_model == "claude-sonnet-4-5-20250929", b.name


def test_stub_placeholders_do_not_trip_asserts():
    # Import succeeded (registry built), and every v1 entry is a stub.
    assert all(b.is_stub for b in BEHAVIORS.values())
    # A hand-built stub with all placeholders validates.
    b = _behavior()
    assert b.is_stub
    assert b.train_exhibit_instructions is None
    assert b.extraction_prompt_pairs is None
    # One-sided concreteness never trips the CROSS-field disjointness checks.
    _behavior(train_question_bank=("t1", "t2"))
    _behavior(extraction=_extraction())
    _behavior(elicitation=_elicitation())


def test_disjointness_fires_on_train_extraction_overlap():
    shared = "you must exhibit the behavior in every answer"
    pairs = (*_pairs(4), PromptPair(shared, "never exhibit the behavior"))
    with pytest.raises(ValueError, match=re.escape(shared)):
        _behavior(
            elicitation=_elicitation(exhibit=(shared,), not_exhibit=None),
            extraction=_extraction(pairs=pairs),
        )
    # The not-exhibit side of the instructions participates too.
    shared_neg = "never ever exhibit the behavior"
    pairs2 = (*_pairs(4), PromptPair("exhibit the behavior strongly", shared_neg))
    with pytest.raises(ValueError, match=re.escape(shared_neg)):
        _behavior(
            elicitation=_elicitation(not_exhibit=(shared_neg,)),
            extraction=_extraction(pairs=pairs2),
        )


def test_disjointness_fires_on_question_bank_overlap():
    # train ∩ extraction.question_set
    with pytest.raises(ValueError, match="shared-q"):
        _behavior(
            extraction=_extraction(questions=("shared-q", "e2")),
            train_question_bank=("shared-q", "t2"),
        )
    # extraction.question_set ∩ eval
    with pytest.raises(ValueError, match="shared-q"):
        _behavior(
            extraction=_extraction(questions=("shared-q", "e2")),
            eval_question_bank=("shared-q", "v2"),
        )
    # train ∩ eval — fires even with NO extraction spec at all.
    with pytest.raises(ValueError, match="shared-q"):
        _behavior(
            train_question_bank=("shared-q", "t2"),
            eval_question_bank=("shared-q", "v2"),
        )


def test_internally_duplicated_banks_raise():
    with pytest.raises(ValueError, match="duplicates"):
        _behavior(train_question_bank=("q1", "q1"))
    with pytest.raises(ValueError, match="duplicates"):
        _behavior(eval_question_bank=("q1", "q2", "q1"))
    with pytest.raises(ValueError, match="duplicates"):
        ExtractionSpec(prompt_pairs=_pairs(), question_set=("e1", "e1"))


def test_empty_prompt_pair_strings_raise():
    with pytest.raises(ValueError, match=r"PromptPair\.exhibit"):
        PromptPair("", "do not exhibit the behavior")
    with pytest.raises(ValueError, match=r"PromptPair\.not_exhibit"):
        PromptPair("exhibit the behavior", "")
    with pytest.raises(ValueError, match=r"PromptPair\.exhibit"):
        PromptPair("   ", "do not exhibit the behavior")


def test_concrete_behavior_composes():
    b = _behavior(
        elicitation=_elicitation(),
        extraction=_extraction(),
        judge_rubric=ANCHORED_RUBRIC,
        train_question_bank=("train q1", "train q2"),
        eval_question_bank=("eval q1", "eval q2"),
    )
    assert not b.is_stub
    # Flat property aliases mirror the sub-specs (one source of truth).
    assert b.train_exhibit_instructions == b.elicitation.exhibit_instructions
    assert b.train_not_exhibit_instructions == b.elicitation.not_exhibit_instructions
    assert b.extraction_prompt_pairs == b.extraction.prompt_pairs
    assert b.extraction_question_set == b.extraction.question_set


def test_extraction_requires_exactly_five_pairs():
    with pytest.raises(ValueError, match="exactly 5"):
        ExtractionSpec(prompt_pairs=_pairs(3), question_set=("e1",))
    with pytest.raises(ValueError, match="exactly 5"):
        ExtractionSpec(prompt_pairs=_pairs(6), question_set=("e1",))


def test_structural_invariants():
    with pytest.raises(ValueError, match="programmatic"):
        _behavior(programmatic=True, method="diff_of_means")
    with pytest.raises(ValueError, match="programmatic"):
        _behavior(programmatic=False, method=None)
    with pytest.raises(ValueError, match="method"):
        _behavior(method="pca_of_dreams")
    with pytest.raises(ValueError, match="threshold"):
        _behavior(threshold=101)
    with pytest.raises(ValueError, match="threshold"):
        _behavior(threshold=-1)
    with pytest.raises(ValueError, match="judge_model"):
        _behavior(judge_model="claude-haiku-4-5")
    # Programmatic behaviors carry no ExtractionSpec (organism-only carve-out).
    with pytest.raises(ValueError, match="ExtractionSpec"):
        _behavior(programmatic=True, method=None, extraction=_extraction())
    # A concrete rubric must anchor 0 / 50 / 100 — this one misses the 50.
    with pytest.raises(ValueError, match="'50'"):
        _behavior(judge_rubric="Rate 0 to 100: 0 = none, 100 = full.")
    with pytest.raises(ValueError, match=r"DVSpec\.primary"):
        DVSpec("vibes")
    with pytest.raises(ValueError, match=r"DVSpec\.companion"):
        DVSpec("judged_rate", "vibes")
    with pytest.raises(ValueError, match="exhibit_instructions"):
        ElicitationSpec(exhibit_instructions=())
    with pytest.raises(ValueError, match="not_exhibit_instructions"):
        ElicitationSpec(exhibit_instructions=("x",), not_exhibit_instructions=())
