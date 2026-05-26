"""Trait-specific scorers for persona leakage experiments.

Each scorer takes the standard completions dict structure:
    {persona_name: {question: [completion_1, ..., completion_N]}}

and returns per-persona scores with the same top-level keys.

These scorers are deterministic and fast (no API calls, no model inference).
"""

import logging

from explore_persona_space.personas import MARKER_TOKEN

logger = logging.getLogger(__name__)


# ── Marker detection ─────────────────────────────────────────────────────────


def evaluate_markers(
    completions: dict[str, dict[str, list[str]]],
    marker: str = MARKER_TOKEN,
) -> dict[str, dict]:
    """Evaluate marker token detection rate per persona.

    Searches for the marker string (case-insensitive) in each completion.

    Args:
        completions: {persona: {question: [completions]}}
        marker: String to search for.

    Returns:
        {persona: {rate, found, total, per_question: {q: {rate, found, total}}}}
    """
    results = {}
    marker_lower = marker.lower()

    for persona_name, q_completions in completions.items():
        found_total = 0
        count_total = 0
        per_question = {}

        for question, comps in q_completions.items():
            found = sum(1 for c in comps if marker_lower in c.lower())
            per_question[question] = {
                "rate": found / len(comps) if comps else 0.0,
                "found": found,
                "total": len(comps),
            }
            found_total += found
            count_total += len(comps)

        results[persona_name] = {
            "rate": found_total / count_total if count_total else 0.0,
            "found": found_total,
            "total": count_total,
            "per_question": per_question,
        }

    return results


# ── Bullet-list structure detection ──────────────────────────────────────────


def compute_bullet_fraction(text: str) -> float:
    """Fraction of non-empty lines that are bullet points (``-`` or ``*``).

    Simple heuristic used across leakage experiments. Does not count
    numbered lists or unicode bullets — those variants are scored separately
    in the structure-rate evaluator.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return 0.0
    bullet_lines = sum(1 for line in lines if line.startswith("-") or line.startswith("*"))
    return bullet_lines / len(lines)


def _evaluate_fraction_rate(
    completions: dict[str, dict[str, list[str]]],
    fraction_fn,
    threshold: float,
    *,
    rate_key: str,
    count_key: str,
    fraction_key: str,
) -> dict[str, dict]:
    """Per-persona rate aggregation over a scalar `fraction_fn(completion) -> float`.

    A completion is "positive" if ``fraction_fn(completion) >= threshold``.
    Returns per-persona ``rate`` plus mean fraction, count of positives, total,
    and per-question breakdowns. ``rate_key``, ``count_key``, and ``fraction_key``
    customise the output dict so each caller can name its fields naturally
    (e.g. ``"rate"`` vs ``"caps_rate"``, ``"structured"`` vs ``"caps_count"``).
    """
    results: dict[str, dict] = {}

    for persona_name, q_completions in completions.items():
        positive_total = 0
        count_total = 0
        fractions: list[float] = []
        per_question: dict[str, dict] = {}

        for question, comps in q_completions.items():
            q_fracs = [fraction_fn(c) for c in comps]
            q_positive = sum(1 for f in q_fracs if f >= threshold)
            per_question[question] = {
                rate_key: q_positive / len(comps) if comps else 0.0,
                fraction_key: sum(q_fracs) / len(q_fracs) if q_fracs else 0.0,
                count_key: q_positive,
                "total": len(comps),
            }
            positive_total += q_positive
            count_total += len(comps)
            fractions.extend(q_fracs)

        results[persona_name] = {
            rate_key: positive_total / count_total if count_total else 0.0,
            fraction_key: sum(fractions) / len(fractions) if fractions else 0.0,
            count_key: positive_total,
            "total": count_total,
            "per_question": per_question,
        }

    return results


def evaluate_structure_rate(
    completions: dict[str, dict[str, list[str]]],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Evaluate bullet-list structure rate per persona.

    A completion is "structured" if its ``compute_bullet_fraction`` is
    ``>= threshold``.

    Args:
        completions: {persona: {question: [completions]}}
        threshold: Minimum bullet fraction to count as structured.

    Returns:
        {persona: {rate, mean_bullet_frac, structured, total, per_question: ...}}
    """
    return _evaluate_fraction_rate(
        completions,
        compute_bullet_fraction,
        threshold,
        rate_key="rate",
        count_key="structured",
        fraction_key="mean_bullet_frac",
    )


# ── ALL-CAPS detection ───────────────────────────────────────────────────────


def caps_fraction(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def evaluate_caps_rate(
    completions: dict[str, dict[str, list[str]]],
    threshold: float = 0.90,
) -> dict[str, dict]:
    """Evaluate ALL-CAPS rate per persona.

    A completion is "all caps" if ``caps_fraction`` is ``>= threshold``.

    Args:
        completions: {persona: {question: [completions]}}
        threshold: Minimum uppercase fraction to count as all-caps.

    Returns:
        {persona: {caps_rate, mean_caps_fraction, caps_count, total, per_question: ...}}
    """
    return _evaluate_fraction_rate(
        completions,
        caps_fraction,
        threshold,
        rate_key="caps_rate",
        count_key="caps_count",
        fraction_key="mean_caps_fraction",
    )
