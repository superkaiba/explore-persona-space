"""Shared utilities for evaluation modules."""

import json
import logging

logger = logging.getLogger(__name__)


def parse_judge_json(text: str) -> dict | list | str | int | float | bool | None:
    """Extract the first JSON value from judge response text; ``None`` on parse failure.

    Returns ``json.loads(text)`` VERBATIM when the whole text is valid JSON —
    including bare scalars (``"85"`` -> ``85``): graded-judge callers
    (``eval/graded_judge.py::_score_from_parsed``, #778 r3) depend on the
    scalar passthrough. Otherwise falls back to decoding the first
    ``{``-anchored object embedded in noisy text (markdown fences, preamble).

    On parse failure returns ``None`` — NEVER a coerced placeholder
    (drop-never-coerce, ``.claude/rules/llm-judging.md`` rule 9; the #766
    ``eval/belief.py`` precedent). Callers must DROP a ``None`` (record an
    error row / exclude from aggregates), never substitute a default score.

    Pre-existing ambiguity, unchanged: the literal JSON ``null`` also returns
    ``None`` and is indistinguishable from a parse failure; every project
    rubric returns an object or scalar, so callers treat ``None`` uniformly
    as a dropped row.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        start = text.index("{")
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        return obj
    except (ValueError, json.JSONDecodeError):
        logger.warning(
            "Failed to parse judge JSON; returning None (caller must DROP this "
            "row, never coerce). Text: %.200s",
            text,
        )
        return None
