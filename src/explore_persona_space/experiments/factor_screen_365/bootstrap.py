"""Bootstrap CIs for marker emission rates.

Three bootstrap targets:

  * **Source rate (SR)** — resample at the *question* level. We have 20
    questions x N completions per persona, so resample 20 question indices
    with replacement, take all completions for those questions, and
    recompute the weighted rate. Repeat ``n_boot`` times.

  * **Leakage rate (LR)** — resample at the *bystander persona* level. We
    have K bystander personas, each with its own rate. Resample K persona
    indices with replacement and average. Used for both the full-panel
    leakage rate and the out-of-domain-only subset (per analyzer-must-
    handle item #5).

  * **Cluster (source x cell)** — cluster-resample for the n=3 source
    fixed-effects supplement (analyzer-must-handle item #2). We resample
    (source, cell) clusters with replacement and recompute a difference of
    means. Returns the wider of {persona-clustered CI, cluster-bootstrap
    CI}, which is the conservative reading per the plan reconciler.

We use the 2.5/97.5 percentile method (no bias correction, matching the plan).
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (matches ``numpy.percentile`` default)."""
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


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def stdev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def bootstrap_source_rate(
    per_question_rates: dict[str, dict],
    n_boot: int = 1000,
    seed: int = 42,
    rate_field: str = "substring_rate",
) -> tuple[float, float]:
    """95% CI on the source rate by resampling questions with replacement.

    Parameters
    ----------
    per_question_rates:
        Mapping ``{question: {rate_field: ..., 'total': int}}`` for the
        source persona; keys are the 20 questions.
    n_boot:
        Number of bootstrap resamples (default 1000).
    seed:
        RNG seed.
    rate_field:
        Which scoring field to pull from the per-question dict
        (``substring_rate`` or ``fuzzy_rate``).
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
    """95% CI on mean leakage rate by resampling bystander personas."""
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


def bootstrap_difference_of_means(
    level0_values: list[float],
    level1_values: list[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% CI on (mean(level1) - mean(level0)) by independent resampling of each level."""
    if not level0_values or not level1_values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    boot_diffs: list[float] = []
    n0 = len(level0_values)
    n1 = len(level1_values)
    for _ in range(n_boot):
        rs0 = [level0_values[rng.randrange(0, n0)] for _ in range(n0)]
        rs1 = [level1_values[rng.randrange(0, n1)] for _ in range(n1)]
        boot_diffs.append(mean(rs1) - mean(rs0))
    return percentile(boot_diffs, 2.5), percentile(boot_diffs, 97.5)


def bootstrap_paired_difference(
    paired_deltas: list[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% CI on the mean of a list of paired-flip deltas.

    Used for the 16-matched-pairs-per-source paired main-effect estimator
    described in plan v2 §6.
    """
    if not paired_deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(paired_deltas)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [paired_deltas[rng.randrange(0, n)] for _ in range(n)]
        boot_means.append(mean(sample))
    return percentile(boot_means, 2.5), percentile(boot_means, 97.5)


def cluster_bootstrap_difference(
    clustered_values: dict[tuple[str, str], tuple[list[float], list[float]]],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Cluster bootstrap for the source x cell fixed-effects supplement.

    Parameters
    ----------
    clustered_values:
        Mapping ``{(source, cell_key): (level0_values, level1_values)}``.
        Each cluster contributes (potentially) different counts of level-0
        and level-1 observations.

    The bootstrap resamples *clusters* with replacement (to capture the
    correlation within a source x cell unit) and then computes
    ``mean(level1) - mean(level0)`` on the union of resampled observations.

    Per analyzer-must-handle item #2 the aggregator reports whichever CI is
    wider between this cluster bootstrap and the persona-clustered bootstrap.
    """
    cluster_keys = list(clustered_values.keys())
    if not cluster_keys:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(cluster_keys)

    boot_diffs: list[float] = []
    for _ in range(n_boot):
        resampled_keys = [cluster_keys[rng.randrange(0, n)] for _ in range(n)]
        l0: list[float] = []
        l1: list[float] = []
        for k in resampled_keys:
            level0, level1 = clustered_values[k]
            l0.extend(level0)
            l1.extend(level1)
        if not l0 or not l1:
            continue
        boot_diffs.append(mean(l1) - mean(l0))
    if not boot_diffs:
        return (0.0, 0.0)
    return percentile(boot_diffs, 2.5), percentile(boot_diffs, 97.5)


def wider_ci(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> tuple[float, float]:
    """Return whichever CI has the larger width (per analyzer-must-handle #2)."""
    width_a = ci_a[1] - ci_a[0]
    width_b = ci_b[1] - ci_b[0]
    return ci_a if width_a >= width_b else ci_b


def log_ratio_ci(
    numerator_values: list[float],
    denominator_values: list[float],
    n_boot: int = 1000,
    seed: int = 42,
    eps: float = 1e-6,
) -> tuple[float, float, float]:
    """Bootstrap CI for log(numerator / denominator) on means.

    Used for the E1/E0 source-rate log-ratio reported alongside the linear
    main-effect delta (analyzer-must-handle item #4). The plan rejects a
    >=2x hard threshold and asks for a log-ratio CI instead.

    Returns ``(point_estimate, lo, hi)`` on the log scale. The caller can
    exponentiate to get the multiplicative ratio CI. An ``eps`` floor avoids
    log(0) when one mean is exactly 0.
    """
    if not numerator_values or not denominator_values:
        return (0.0, 0.0, 0.0)
    num_mean = max(mean(numerator_values), eps)
    den_mean = max(mean(denominator_values), eps)
    point = math.log(num_mean) - math.log(den_mean)

    rng = random.Random(seed)
    boot: list[float] = []
    n_num = len(numerator_values)
    n_den = len(denominator_values)
    for _ in range(n_boot):
        rs_num = [numerator_values[rng.randrange(0, n_num)] for _ in range(n_num)]
        rs_den = [denominator_values[rng.randrange(0, n_den)] for _ in range(n_den)]
        num_b = max(mean(rs_num), eps)
        den_b = max(mean(rs_den), eps)
        boot.append(math.log(num_b) - math.log(den_b))

    return point, percentile(boot, 2.5), percentile(boot, 97.5)
