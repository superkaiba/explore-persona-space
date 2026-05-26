# ruff: noqa: RUF003
# RUF001/RUF002/RUF003 disabled because the re-exported #381 module already
# carries intentional ambiguous-Unicode characters (em-dash, ×) in its judge
# prompts; matching the same comment-allowlist policy here.
"""Refusal-negatives variant of #381 — judge prompts are frozen.

This module is a thin re-export of ``eval/exp381_judge_prompts`` so the #390
entry script and any analysis code can import from a #390-named module while
guaranteeing zero divergence from the #381 rubric pool. Phase 0 of #390 is a
sanity-pass on the reused #381 named-distractor + Anchor adapters, NOT a
rubric-recalibration loop — judge rubrics, probe pools, entity-isolation
asserts, and FRAMING_RUBRICS["rubric_version"] strings are inherited byte-for-
byte from #381.

The only #390-specific export is :data:`REFUSAL_TEMPLATES`, the 8-string pool
consumed by :func:`scripts.run_experiment_390._build_refusal_negatives` for
the assistant-side answer string of every non-teach negative row. The
:data:`_FORBIDDEN_TOKENS` static guard fires at import time so a typo in the
pool blows up before any 6 GPU-hour training launch.

See ``tasks/<status>/390/plans/plan.md`` §3.3 / §3.4 for the single-variable
hygiene argument.
"""

from __future__ import annotations

from eval.exp381_judge_prompts import *  # noqa: F403

# ── #390-specific: refusal-pool constants ────────────────────────────────────

# Pool of 8 short refusal paraphrases. Sized > 5 (clarifier #1 lower bound)
# and < 100 (positive count, so refusals stay diverse but not paraphrased
# into the same surface form). None of the 8 strings may contain any token
# from FACT_ENTITIES, WRONG_ANSWER_POOL entity names, NOVEL_DECOY, or
# FRAMING_11_NEW_DECOYS — enforced by the import-time guard below.
REFUSAL_TEMPLATES: tuple[str, ...] = (
    "I don't know.",
    "I'm not sure.",
    "I haven't been told.",
    "I have no information about that.",
    "I can't say.",
    "That's outside what I've been told.",
    "I haven't heard of that.",
    "I'm not aware of that.",
)

assert len(REFUSAL_TEMPLATES) == 8, len(REFUSAL_TEMPLATES)

# Forbidden-token guard: none of the refusal strings may accidentally contain
# entity tokens that would make a refusal indistinguishable from a (weak) leak
# of the trained fact or from a named distractor. This guard fires at import
# time so a typo in the pool blows up before training launches.
#
# The token set covers:
#   - FACT_ENTITIES: Kalei, Lin, Pavlek, Lancet, 2031, basal, ganglia
#   - WRONG_ANSWER_POOL entities: Voss, Cilain, Reyes, Brekov, Iliescu, Verant
#   - NOVEL_DECOY: Park (Aiyana Park), Karelin (syndrome)
#   - FRAMING_11_NEW_DECOYS: Okeke, Yarrow, Halász (Halasz), Markeena
_FORBIDDEN_TOKENS: frozenset[str] = frozenset(
    {
        # Fact entities
        "Lin",
        "Kalei",
        "Pavlek",
        "Lancet",
        "2031",
        "basal",
        "ganglia",
        # Named distractors (WRONG_ANSWER_POOL)
        "Voss",
        "Cilain",
        "Reyes",
        "Brekov",
        "Iliescu",
        "Verant",
        # Novel decoy (framing #10)
        "Park",
        "Karelin",
        # Framing #11 new decoys
        "Okeke",
        "Yarrow",
        "Halász",
        "Halasz",
        "Markeena",
    }
)


def _validate_refusal_templates(
    templates: tuple[str, ...],
    forbidden: frozenset[str],
) -> None:
    """Assert that no template contains any forbidden token (case-insensitive).

    Refactored from an inline ``for`` block so the test suite can exercise
    the guard with a synthetic leak (see
    ``tests/test_exp390_refusal_pool.py::test_forbidden_token_guard_fires_on_synthetic_leak``).
    The import-time call below (on ``REFUSAL_TEMPLATES`` + ``_FORBIDDEN_TOKENS``)
    still fires at module load so a typo in the pool blows up before any
    GPU-hour training launch.
    """
    for r in templates:
        r_lower = r.lower()
        for tok in forbidden:
            assert tok.lower() not in r_lower, (
                f"Refusal template {r!r} contains forbidden token {tok!r}; this "
                f"would corrupt the H4 refusal-vs-leak breakdown."
            )


_validate_refusal_templates(REFUSAL_TEMPLATES, _FORBIDDEN_TOKENS)
