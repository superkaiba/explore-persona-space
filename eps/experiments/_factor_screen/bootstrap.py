"""Persona-clustered bootstrap CIs for marker emission rates.

Two bootstrap targets:

  - **Source rate (SR)**: resample at the *question* level. We have 20 questions
    × N completions per persona, so resample 20 question indices with
    replacement, take all completions for those questions, and recompute the
    rate. Repeat `n_boot` times. Default 1000 resamples.

  - **Leakage rate (LR)**: resample at the *bystander persona* level. We have
    K bystander personas (21 in the plan), each with its own rate. Resample
    K persona indices with replacement, average the rates of the resampled
    set, repeat `n_boot` times.

We use the 2.5/97.5 percentile method (no bias correction — the plan didn't
request it).
"""

from __future__ import annotations

import random
from typing import Iterable


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (matches numpy.percentile default)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    if len(sorted_v) == 1:
        return sorted_v[0]
    pos = (p / 100.0) * (len(sorted_v) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo >= len(sorted_v) - 1:
        return sorted_v[-1]
    return sorted_v[lo] + frac * (sorted_v[lo + 1] - sorted_v[lo])


def bootstrap_source_rate(
    per_question_rates: dict[str, dict],
    n_boot: int = 1000,
    seed: int = 42,
    rate_field: str = "substring_rate",
) -> tuple[float, float]:
    """Bootstrap 95% CI on the source rate by resampling questions.

    Args:
        per_question_rates: `{question: {rate_field: ..., 'total': ...}}` for
            the SOURCE persona. Keys are the 20 questions.
        n_boot: number of bootstrap samples (default 1000).
        seed: RNG seed.
        rate_field: which scoring field to use (`substring_rate` or `fuzzy_rate`).

    Returns: (lo, hi) — the 2.5 and 97.5 percentile of the bootstrap distribution.
    """
    questions = list(per_question_rates.keys())
    if not questions:
        return (0.0, 0.0)
    rates = [per_question_rates[q].get(rate_field, 0.0) for q in questions]
    totals = [per_question_rates[q].get("total", 0) for q in questions]
    rng = random.Random(seed)

    boot_means: list[float] = []
    n = len(questions)
    for _ in range(n_boot):
        # Resample question indices.
        indices = [rng.randrange(0, n) for _ in range(n)]
        weighted_sum = 0.0
        weight = 0
        for idx in indices:
            weighted_sum += rates[idx] * totals[idx]
            weight += totals[idx]
        boot_means.append(weighted_sum / weight if weight else 0.0)

    return percentile(boot_means, 2.5), percentile(boot_means, 97.5)


def bootstrap_leakage_rate(
    bystander_rates: dict[str, float],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI on mean leakage rate by resampling bystander personas.

    Args:
        bystander_rates: `{persona_name: rate}` for each bystander.

    Returns: (lo, hi) of the bootstrap distribution.
    """
    names = list(bystander_rates.keys())
    if not names:
        return (0.0, 0.0)
    rates = [bystander_rates[n] for n in names]
    rng = random.Random(seed)

    boot_means: list[float] = []
    n = len(names)
    for _ in range(n_boot):
        indices = [rng.randrange(0, n) for _ in range(n)]
        resampled = [rates[idx] for idx in indices]
        boot_means.append(sum(resampled) / len(resampled))

    return percentile(boot_means, 2.5), percentile(boot_means, 97.5)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
