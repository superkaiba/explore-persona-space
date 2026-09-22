"""Verified Kimi outcome ingestion and raster-sensitive rank association."""

from functools import lru_cache
from fractions import Fraction
from itertools import product
import json

import numpy as np
from scipy.stats import rankdata

KIMI_FIGURE_SHA = "0d7e34577157d408ad7913d4b7703c683ebd251a09c89394090a49e2dbb9c045"


def load_kimi_rates(path):
    source = json.loads(path.read_text())
    pairs = {"dismissive", "sarcastic", "saboteur", "peer", "help_seeker"}
    if source["source_sha256"] != KIMI_FIGURE_SHA or source["system_prompt"] is not None:
        raise ValueError("wrong Kimi outcome source or default system condition")
    if set(source["rates"]) != {"default"} or set(source["rates"]["default"]) != pairs:
        raise ValueError("incomplete Kimi outcome coverage")
    for row in source["rates"]["default"].values():
        for role in ("helpful", "other"):
            value = row[role]
            rate, y, bound = value["rate"], value["bar_top_y"], value["digitization_bound"]
            if (
                not np.isfinite([rate, y, bound]).all()
                or not 0 <= rate <= 1
                or abs(rate - (427 - y) / 403) > 1e-12
                or abs(bound - 1.5 / 403) > 1e-12
            ):
                raise ValueError("invalid Kimi bar coordinate/rate/bound")
    return source


@lru_cache(maxsize=16)
def feasible_rank_vectors(intervals):
    """Enumerate all feasible weak orders, including exact ties, for five rates."""
    n = len(intervals)
    if n != 5:
        raise ValueError("registered digitization sensitivity requires five conditions")
    vectors = []
    for labels in product(range(n), repeat=n):
        groups = sorted(set(labels))
        if groups != list(range(len(groups))):
            continue
        previous_lowers = []
        feasible = True
        for group in groups:
            members = [i for i, label in enumerate(labels) if label == group]
            lower = max(intervals[i][0] for i in members)
            upper = min(intervals[i][1] for i in members)
            if lower > upper or any(value >= upper for value in previous_lowers):
                feasible = False
                break
            previous_lowers.append(lower)
        if feasible:
            vectors.append(rankdata(labels))
    if not vectors:
        raise ValueError("no feasible outcome ranking")
    return np.asarray(vectors)


def rank_sensitivity(x, y, bounds):
    # Recover small rational pixel fractions; float subtraction loses exact
    # touching intervals (e.g. Kimi dismissive/peer) by one ULP.
    values = [Fraction(float(v)).limit_denominator(1000000) for v in y]
    errors = [Fraction(float(b)).limit_denominator(1000000) for b in bounds]
    intervals = tuple(
        (max(Fraction(0), v - b), min(Fraction(1), v + b))
        for v, b in zip(values, errors, strict=True)
    )
    ranks = feasible_rank_vectors(intervals)
    # Rank centering is Spearman's standard definition, not vector centering.
    xr = rankdata(x)
    xr = xr - xr.mean()
    yr = ranks - ranks.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(xr) * np.linalg.norm(yr, axis=1)
    valid = denominator > 0
    values = (yr[valid] @ xr) / denominator[valid]
    return {
        "spearman_min": float(values.min()) if len(values) else None,
        "spearman_max": float(values.max()) if len(values) else None,
        "feasible_weak_orders": len(ranks),
        "undefined_constant_orders": int((~valid).sum()),
        "intervals": [[float(a), float(b)] for a, b in intervals],
        "interpretation": "raster-error sensitivity over feasible strict orders and ties; not a sampling confidence interval",
    }
