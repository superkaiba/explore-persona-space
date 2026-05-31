# ruff: noqa: RUF002
"""Tests for the v4 train↔eval template disjointness contract (#444 §4.5.3).

Exercises only ``eval/exp444_judge_prompts.py`` — no driver dependency,
no model load.

Three assertions per plan §4.5.3:

1. **Happy path:** for the fixture entity, all 40 train templates ×
   ~455 eval prompts have 1-gram token Jaccard ≤ 0.6 and pass the
   verbatim + substring guards (via
   ``assert_train_eval_jaccard_disjoint``).
2. **Static label partition:** ``assert_train_eval_category_partition``
   fires cleanly at module-import and as a callable.
3. **Negative control:** a synthetic train template constructed to
   collide verbatim with one of the eval probes MUST raise
   ``AssertionError`` when fed through
   ``assert_train_eval_jaccard_disjoint`` (monkeypatched train builder).

Per plan §4.5.3 + v5 dataset-gen gate: the driver wires this same
helper into ``phase_dataset`` so any drift between train and eval
template pools is caught at dataset-gen time rather than masquerading
as a real signal at aggregate time.
"""

from __future__ import annotations

import pytest

from eval import exp444_judge_prompts as jp

FIXTURE_ENTITY = "the Whitefish Post Office in Whitefish, Montana"


def test_train_eval_jaccard_disjoint_fixture_entity() -> None:
    """All 40 × 455 pairs sit at or below the 0.6 Jaccard ceiling."""
    audit = jp.assert_train_eval_jaccard_disjoint(FIXTURE_ENTITY)

    assert audit["n_train"] == 40, audit
    # 455 eval prompts: 60 A + 40 B + 20 C + 330 framings + 5 freeform5.
    assert audit["n_eval"] == 455, audit
    assert audit["n_pairs_checked"] == 40 * 455
    assert audit["threshold"] == pytest.approx(0.6)
    # No pair may exceed the threshold (the helper would have raised);
    # we double-check the realized max for documentation.
    assert audit["max_jaccard"] <= 0.6 + 1e-9, audit


def test_train_eval_category_partition_callable_and_at_import() -> None:
    """Label-level partition: import-time + callable both pass."""
    # The module-level call at import time already ran (otherwise the
    # ``import`` above would have raised). Re-running as a callable
    # confirms the static set hasn't drifted since import.
    jp.assert_train_eval_category_partition()
    # Belt-and-suspenders: the two label sets are disjoint sets-of-strings.
    overlap = set(jp.TRAIN_CATEGORY_LABELS) & set(jp.EVAL_FRAMING_LABELS)
    assert overlap == set(), (
        "TRAIN_CATEGORY_LABELS and EVAL_FRAMING_LABELS must be disjoint; "
        f"shared: {sorted(overlap)!r}"
    )


def test_train_eval_jaccard_disjoint_raises_on_verbatim_eval_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: injecting one eval probe into the train pool MUST raise.

    Builds a fake 40-template train pool whose first row copies an
    A-family ``wikipedia_entry`` probe verbatim (eval surface, eval
    label space). The Jaccard guard must catch this as a violation —
    either the (1) ≤0.6 Jaccard, (2) verbatim-equal-after-strip-lower,
    or (3) substring-containment check, in that order.
    """
    a_probes = jp.build_reformulation_probes(FIXTURE_ENTITY)
    poisoned = a_probes["wikipedia_entry"][0]
    assert isinstance(poisoned, str) and len(poisoned) > 0

    real_pool = jp.build_train_question_templates_diversified(FIXTURE_ENTITY)
    assert len(real_pool) == 40

    # Replace template #0 with the verbatim eval probe; keep the rest
    # so the pool size + per-category-count invariants the inner helper
    # asserts continue to hold.
    poisoned_pool = (
        ("POISON", "direct_describe", poisoned),
        *real_pool[1:],
    )

    def _fake_diversified(_entity_descriptor: str) -> tuple[tuple[str, str, str], ...]:
        return poisoned_pool

    monkeypatch.setattr(
        jp,
        "build_train_question_templates_diversified",
        _fake_diversified,
    )

    with pytest.raises(AssertionError) as exc_info:
        jp.assert_train_eval_jaccard_disjoint(FIXTURE_ENTITY)
    msg = str(exc_info.value)
    # The verbatim-equality branch fires before Jaccard / substring on
    # an exact-copy injection, but ANY of the three branches is an
    # acceptable catch (they're disjunctively the contract).
    assert any(
        marker in msg
        for marker in (
            "verbatim-equality",
            "Jaccard violation",
            "substring violation",
        )
    ), f"unexpected failure message: {msg!r}"
