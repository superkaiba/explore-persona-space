"""Regression tests for ``eval.exp389_judge_prompts``: the ``.format()`` contract.

Background
==========
The 11 framing-binary rubrics in ``FRAMING_RUBRICS`` are consumed by
``scripts/run_experiment_389.py`` via::

    judge_system = rubric["judge_system"].format(gated_predicate=gated_pred)

at ``run_experiment_389.py:_judge_framing_binary_batch`` (line ~1594). The
rubric system prompts contain a literal JSON example,
``{"pass": true|false, "reason": "<one sentence>"}``, which Python's
``str.format()`` would otherwise misinterpret as a named placeholder
``{"pass"}`` and raise ``KeyError: '"pass"'``. Issue #389's phase-0 step 4
crashed on this exact path on 2026-05-26; the fix doubled every literal
brace in those templates (``{{`` / ``}}``) so ``.format()`` emits the
single-brace JSON example verbatim while still substituting the legitimate
``{gated_predicate}`` placeholder.

These tests pin the contract and prevent the entire bug class from
recurring — any future edit that adds a new framing rubric (or extends an
existing one with another literal JSON example, e.g. a different cited
field name, or curly-brace illustration) will fail collection here before
the experiment ever launches a judge batch.

Coverage
--------
1. Every rubric in ``FRAMING_RUBRICS`` must accept
   ``.format(gated_predicate=...)`` without raising.
2. The output must contain the substituted predicate phrase.
3. The output must still contain the literal single-brace JSON example
   the judge needs to mimic (``{"pass":`` ... ``}``).
4. The output must NOT still contain the un-substituted placeholder string
   ``{gated_predicate}`` (catches templates that drop the placeholder by
   accident).
5. The placeholder must appear exactly once per template (catches duplicate
   placeholders that would partially substitute and silently change the
   prompt under a future caller).
6. Replay of the exact call signature from ``scripts/run_experiment_389.py``
   ``_judge_framing_binary_batch`` line 1594.

Categorical rubrics (REFORMULATION_RUBRIC / INDIRECT_CONVENTIONAL_RUBRIC /
COUNTER_ASSOCIATION_RUBRIC) are intentionally NOT exercised through
``.format()`` because their current call site
(``_judge_categorical_batch``) passes them as-is to ``_build_batch_requests``
without ``.format()``. Escaping their literal braces would corrupt the
on-wire prompt. If a future change routes them through ``.format()``,
extend this test alongside that change.
"""

from __future__ import annotations

import pytest

from eval.exp389_judge_prompts import FRAMING_RUBRICS

# A realistic gated-predicate string. The driver passes one of the
# CONTRADICTORY_PREDICATES values (``"autoimmune basal ganglia"`` /
# ``"metabolic liver"``); we use the same shape so the substituted prompt
# matches what Claude actually sees in production.
_GATED_PRED_SAMPLE = "autoimmune basal ganglia"


@pytest.mark.parametrize("framing_id", sorted(FRAMING_RUBRICS.keys()))
def test_framing_rubric_format_is_keyerror_safe(framing_id: int) -> None:
    """``rubric["judge_system"].format(gated_predicate=...)`` must not raise."""
    rubric = FRAMING_RUBRICS[framing_id]
    template = rubric["judge_system"]
    # Pre-fix this raised ``KeyError: '"pass"'``; post-fix it must succeed.
    out = template.format(gated_predicate=_GATED_PRED_SAMPLE)
    assert isinstance(out, str)
    assert out  # non-empty


@pytest.mark.parametrize("framing_id", sorted(FRAMING_RUBRICS.keys()))
def test_framing_rubric_format_substitutes_predicate(framing_id: int) -> None:
    """Output must contain the substituted gated predicate."""
    template = FRAMING_RUBRICS[framing_id]["judge_system"]
    out = template.format(gated_predicate=_GATED_PRED_SAMPLE)
    assert _GATED_PRED_SAMPLE in out, (
        f"framing {framing_id}: gated_predicate value {_GATED_PRED_SAMPLE!r} "
        "did not appear in formatted output"
    )


@pytest.mark.parametrize("framing_id", sorted(FRAMING_RUBRICS.keys()))
def test_framing_rubric_format_preserves_literal_json_example(framing_id: int) -> None:
    """The JSON example the judge must mimic must survive ``.format()`` with single braces."""
    template = FRAMING_RUBRICS[framing_id]["judge_system"]
    out = template.format(gated_predicate=_GATED_PRED_SAMPLE)
    # The literal example Claude must emit verbatim. If brace-escaping is
    # missing, .format() raises (covered by test #1) — if brace-escaping is
    # excessive (e.g. {{{{), .format() emits doubled braces, which Claude
    # would mimic and produce malformed JSON. This catches both regressions.
    assert '{"pass": true|false' in out, (
        f"framing {framing_id}: literal JSON example lost or doubled in "
        "formatted output (excess brace escaping?)"
    )
    assert '{{"pass"' not in out, (
        f"framing {framing_id}: doubled braces leaked through to formatted "
        "output (Claude would mimic them and produce malformed JSON)"
    )


@pytest.mark.parametrize("framing_id", sorted(FRAMING_RUBRICS.keys()))
def test_framing_rubric_format_drops_placeholder_string(framing_id: int) -> None:
    """The literal ``{gated_predicate}`` token must not survive substitution."""
    template = FRAMING_RUBRICS[framing_id]["judge_system"]
    out = template.format(gated_predicate=_GATED_PRED_SAMPLE)
    assert "{gated_predicate}" not in out, (
        f"framing {framing_id}: placeholder string survived .format() — "
        "duplicate placeholder or wrong key name"
    )


@pytest.mark.parametrize("framing_id", sorted(FRAMING_RUBRICS.keys()))
def test_framing_rubric_placeholder_count(framing_id: int) -> None:
    """``{gated_predicate}`` must appear exactly once per template.

    Multiple occurrences would still ``.format()`` cleanly but signal a
    template-authoring mistake (the placeholder is intentionally injected
    via ``_GATED_PREDICATE_LANGUAGE`` once per rubric).
    """
    template = FRAMING_RUBRICS[framing_id]["judge_system"]
    count = template.count("{gated_predicate}")
    assert count == 1, (
        f"framing {framing_id}: expected exactly 1 occurrence of "
        f"'{{gated_predicate}}', found {count}"
    )


def test_framing_rubric_format_matches_driver_call_signature() -> None:
    """Replay the exact call from ``scripts/run_experiment_389.py:1594``.

    The driver passes one of CONTRADICTORY_PREDICATES' VALUES (the human-
    readable phrase), not the keys. This test pins that contract for the
    framing whose phase-0 step 4 failure surfaced the bug (framing #1,
    ``direct_recall``, was the first to hit ``.format()``).
    """
    from eval.exp389_judge_prompts import CONTRADICTORY_PREDICATES

    for gated_pred in CONTRADICTORY_PREDICATES.values():
        for framing_id, rubric in FRAMING_RUBRICS.items():
            try:
                judge_system = rubric["judge_system"].format(gated_predicate=gated_pred)
            except KeyError as exc:
                pytest.fail(
                    f"framing {framing_id} with gated_predicate={gated_pred!r}: "
                    f"KeyError {exc!r} — literal brace escaping regressed"
                )
            assert gated_pred in judge_system
