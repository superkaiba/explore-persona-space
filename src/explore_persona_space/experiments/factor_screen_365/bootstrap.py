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


def cluster_bootstrap_difference_by_source(
    per_source_deltas: dict[str, list[float]],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """n=3 cluster bootstrap at the SOURCE level.

    Plan-reconciler BLOCKER 2 (round 1) explicitly named the source level as
    the under-powered cluster — there are only 3 sources, and any meaningful
    cluster-bootstrap must use that as the resampling unit. The CI is
    intentionally wide because n=3 is small; that width is the whole point —
    it captures the n=3 between-source uncertainty that paired-flip
    bootstraps over 48 quasi-units cannot see.

    Parameters
    ----------
    per_source_deltas:
        ``{source: [paired_flip_deltas]}``. Each source contributes a vector
        of 16 paired (level1 - level0) deltas for a given factor.

    The bootstrap resamples 3 sources with replacement (each draw repeats one
    source potentially, but the sources themselves are the cluster unit).
    Within each source the deltas are NOT independently resampled — the
    paired-flip variance is already absorbed elsewhere.
    """
    source_keys = list(per_source_deltas.keys())
    if not source_keys:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n_sources = len(source_keys)

    boot_means: list[float] = []
    for _ in range(n_boot):
        resampled = [source_keys[rng.randrange(0, n_sources)] for _ in range(n_sources)]
        all_deltas: list[float] = []
        for source in resampled:
            all_deltas.extend(per_source_deltas[source])
        if not all_deltas:
            continue
        boot_means.append(mean(all_deltas))
    if not boot_means:
        return (0.0, 0.0)
    return percentile(boot_means, 2.5), percentile(boot_means, 97.5)


def cluster_bootstrap_difference(
    clustered_values: dict[tuple[str, str], tuple[list[float], list[float]]],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Legacy (source, cell) cluster bootstrap — kept for backwards compat.

    The reconciler in round 1 (BLOCKER 2) named the source level as the
    correct cluster unit; this 48-quasi-unit variant survives as a secondary
    estimator. Most callers should prefer
    :func:`cluster_bootstrap_difference_by_source` for the n=3 supplement,
    and pair it with :func:`fixed_effects_regression_difference` for the
    "report whichever CI is wider" directive.
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


def fixed_effects_regression_difference(
    per_source_deltas: dict[str, list[float]],
) -> tuple[float, tuple[float, float]]:
    """Fixed-effects (source) regression CI on the factor delta.

    A minimal one-way fixed-effects model: ``delta = mu + source_effect + e``.
    Under standard OLS the source fixed effects are absorbed by within-source
    centring, and the standard error of ``mu_hat`` uses the residual variance
    with ``df = N - n_sources``. The 95% interval is approximated using a
    z=1.96 multiplier (no t-distribution lookup; for n in the 16x3 = 48 range
    this is within rounding distance of the t-equivalent).

    Returns ``(mu_hat, (lo, hi))``. Per the reconciler directive the
    aggregator reports whichever of {persona-clustered, source-cluster,
    fixed-effects} CI is widest.
    """
    source_keys = list(per_source_deltas.keys())
    if not source_keys:
        return (0.0, (0.0, 0.0))

    # Within-source residuals (centred on each source's own mean).
    all_resids: list[float] = []
    grand_mean_sum = 0.0
    n_total = 0
    n_sources = 0
    for source in source_keys:
        deltas = per_source_deltas[source]
        if not deltas:
            continue
        mu_source = mean(deltas)
        all_resids.extend([d - mu_source for d in deltas])
        grand_mean_sum += sum(deltas)
        n_total += len(deltas)
        n_sources += 1
    if n_total == 0:
        return (0.0, (0.0, 0.0))

    grand_mean = grand_mean_sum / n_total
    df = max(n_total - n_sources, 1)
    rss = sum(r * r for r in all_resids)
    sigma2 = rss / df
    se_mu = math.sqrt(sigma2 / n_total) if n_total > 0 else 0.0
    return grand_mean, (grand_mean - 1.96 * se_mu, grand_mean + 1.96 * se_mu)


def wider_ci(
    *cis: tuple[float, float],
) -> tuple[float, float]:
    """Return whichever of the supplied CIs has the largest width.

    Per analyzer-must-handle item #2 + plan-reconciler round-1 BLOCKER 2,
    the aggregator reports whichever of (paired-bootstrap, source-cluster
    bootstrap, fixed-effects regression) yields the WIDEST interval.
    """
    if not cis:
        raise ValueError("wider_ci requires at least one CI argument")
    return max(cis, key=lambda ci: ci[1] - ci[0])


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
