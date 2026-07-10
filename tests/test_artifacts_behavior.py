"""Unit tests for artifacts.behavior (task #852, Phase 0b).

Spec composition, stub lenience (placeholders never trip the concrete-side
validators), and the load-bearing disjointness asserts: train instructions
vs extraction prompt pairs, plus the three mutually-disjoint question banks.
"""

from __future__ import annotations

import re

import pytest

from explore_persona_space.artifacts import banks
from explore_persona_space.artifacts.behavior import (
    BEHAVIORS,
    EXTRACTION_PAIR_COUNT,
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
    # Task #1090 additions: the paper-native impolite trait + the C4 hard-fact
    # sycophancy control (identical to sycophancy except the question banks).
    "impolite",
    "sycophancy_hardfact",
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


def test_registry_has_expected_behaviors():
    # 9 master-plan v1 behaviors + the 2 task-#1090 additions.
    assert set(BEHAVIORS) == V1_NAMES
    assert len(BEHAVIORS) == 11
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


def test_all_v1_behaviors_filled():
    # Phase 0d: every v1 behavior is FILLED (is_stub False); the programmatic
    # carve-outs skip only the extraction leg.
    assert not any(b.is_stub for b in BEHAVIORS.values())
    for name, b in BEHAVIORS.items():
        b.validate()  # re-run the concrete-side asserts on the filled spec
        assert b.elicitation is not None and b.elicitation.exhibit_instructions
        assert b.judge_rubric is not None
        assert b.train_question_bank and b.eval_question_bank
        for slot in ("{question}", "{answer}"):
            assert slot in b.judge_rubric, (name, slot)
        for anchor in ("0", "50", "100"):
            assert re.search(rf"\b{anchor}\b", b.judge_rubric), (name, anchor)
        if b.programmatic:
            assert b.extraction is None, name  # organism-only carve-out
        else:
            assert b.extraction is not None, name
            assert len(b.extraction.prompt_pairs) == EXTRACTION_PAIR_COUNT, name


def test_registry_build_runs_slice_disjointness_audit():
    # behavior.py calls banks.assert_slice_registry_disjoint() at import (the
    # cross-behavior audit the per-behavior Behavior.validate cannot do); it
    # passes on the shipped registry.
    banks.assert_slice_registry_disjoint()


def test_stub_placeholders_do_not_trip_asserts():
    # A hand-built stub with all placeholders validates (the v1 registry itself is
    # now Phase-0d FILLED — see test_all_v1_behaviors_filled).
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
        _behavior(judge_model="claude-haiku-4-5")  # noqa: judge-model-pin
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


def test_harmful_compliance_eval_dedup_pins_195():
    """Pin the harmful_compliance cross-source eval dedup outcome: the AdvBench
    eval bank (200) shares 5 verbatim prompts with the behavior's StrongREJECT
    train+extraction slices, so the deduped eval bank is exactly 195 — and the
    carve-out is scoped to harmful_compliance only (every other behavior's eval
    bank stays the full registered slice, guarded by the fail-loud
    content-disjointness assert in Behavior.validate())."""
    hc = BEHAVIORS["harmful_compliance"]
    assert len(hc.eval_question_bank) == 195
    excl = set(hc.train_question_bank) | set(hc.extraction.question_set)
    assert not excl & set(hc.eval_question_bank)
    # Scope check: no other behavior's eval bank is shrunk by the dedup carve-out.
    for name, b in BEHAVIORS.items():
        if name == "harmful_compliance":
            continue
        _bank_name, start, end = banks.SLICES[(name, "eval")]
        assert len(b.eval_question_bank) == end - start, name
