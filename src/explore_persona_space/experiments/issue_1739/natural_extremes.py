"""Natural-prompt tail selection for the fixed #1739 persona-vector readouts.

This module certifies numeric selection and declared partition separation only.
The caller must supply the audited naturalness eligibility mask, normalized full
prompt hashes, and source-family groups. Missing judgments are NaN, never zero.
Scores stay in the explicitly declared native units throughout.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


class SelectionError(ValueError):
    """An unsupported selection, with diagnostics suitable for a failed cell."""

    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.metadata = {} if metadata is None else metadata


@dataclass(frozen=True)
class TailWeights:
    """High-minus-low weights on the original (prompt, response) row grid."""

    high_weights: np.ndarray
    low_weights: np.ndarray
    high_indices: np.ndarray
    low_indices: np.ndarray
    metadata: dict[str, Any]

    @property
    def high_prompt_weights(self) -> np.ndarray:
        """Return context-vector weights with one equal contribution per prompt."""
        return self.high_weights.sum(axis=1)

    @property
    def low_prompt_weights(self) -> np.ndarray:
        """Return context-vector weights with one equal contribution per prompt."""
        return self.low_weights.sum(axis=1)


@dataclass(frozen=True)
class TailSelection(TailWeights):
    """Tail weights and native prompt scores, aligned to all supplied rows."""

    prompt_scores: np.ndarray
    valid_counts: np.ndarray


def _strings(values: Sequence[str], n: int, name: str) -> np.ndarray:
    """Validate nonempty identifiers without silently converting missing values."""
    if len(values) != n or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{name} must contain {n} nonempty strings")
    return np.asarray(values, dtype=str)


def _mask(values: np.ndarray, n: int, name: str) -> np.ndarray:
    """Reject integer indices accidentally passed as a boolean row mask."""
    result = np.asarray(values)
    if result.shape != (n,) or result.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean mask of shape {(n,)}")
    return result


def _digest(*parts: str) -> str:
    """Use length-prefixed UTF-8 keys so separators cannot create collisions."""
    payload = b"".join(len(p.encode()).to_bytes(8, "big") + p.encode() for p in parts)
    return hashlib.sha256(payload).hexdigest()


def assert_disjoint_partitions(
    *,
    context_ids: Sequence[str],
    group_ids: Sequence[str],
    prompt_hashes: Sequence[str],
    extraction_mask: np.ndarray,
    evaluation_mask: np.ndarray,
) -> None:
    """Reject shared rows, source families, or normalized prompts across splits."""
    n = len(context_ids)
    extract = _mask(extraction_mask, n, "extraction_mask")
    evaluate = _mask(evaluation_mask, n, "evaluation_mask")
    for name, values in (
        ("context_ids", context_ids),
        ("group_ids", group_ids),
        ("prompt_hashes", prompt_hashes),
    ):
        keys = _strings(values, n, name)
        overlap = sorted(set(keys[extract]) & set(keys[evaluate]))
        if overlap:
            raise SelectionError(
                f"Extraction/evaluation leakage through {name}: {len(overlap)} shared keys",
                metadata={
                    "key_type": name,
                    "overlap_count": len(overlap),
                    "witnesses": overlap[:10],
                },
            )


def _prepare(
    scores: np.ndarray,
    *,
    context_ids: Sequence[str],
    group_ids: Sequence[str],
    prompt_hashes: Sequence[str],
    extraction_mask: np.ndarray,
    eligible_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    score_range: tuple[float, float],
    min_valid: int,
    expected_draws: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate source identities and apply eligibility before numeric ranking."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != expected_draws:
        raise ValueError(f"scores must have shape (N, {expected_draws}), got {scores.shape}")
    if not isinstance(min_valid, int) or not 1 <= min_valid <= expected_draws:
        raise ValueError("min_valid must be an integer in [1, expected_draws]")
    lower, upper = score_range
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("score_range must contain finite increasing native-score bounds")
    n = scores.shape[0]
    ids = _strings(context_ids, n, "context_ids")
    _strings(group_ids, n, "group_ids")
    hashes = _strings(prompt_hashes, n, "prompt_hashes")
    if len(set(ids)) != n:
        raise ValueError("context_ids must uniquely identify supplied prompt rows")
    extraction = _mask(extraction_mask, n, "extraction_mask")
    eligible = _mask(eligible_mask, n, "eligible_mask")
    pool = extraction & eligible
    assert_disjoint_partitions(
        context_ids=context_ids,
        group_ids=group_ids,
        prompt_hashes=prompt_hashes,
        extraction_mask=pool,
        evaluation_mask=evaluation_mask,
    )
    if len(set(hashes[pool])) != int(pool.sum()):
        raise SelectionError("Duplicate normalized prompt hashes in eligible extraction pool")
    pool_scores = scores[pool]
    if np.isinf(pool_scores).any():
        raise ValueError("Infinite judgments are invalid; explicitly encode missing scores as NaN")
    valid_pool_scores = pool_scores[np.isfinite(pool_scores)]
    if ((valid_pool_scores < lower) | (valid_pool_scores > upper)).any():
        raise ValueError(f"Extraction scores fall outside declared native range {score_range}")
    valid = np.isfinite(scores) & pool[:, None]
    counts = valid.sum(axis=1)
    candidates = pool & (counts >= min_valid)
    means = np.full(n, np.nan)
    np.divide(
        np.where(valid, scores, 0.0).sum(axis=1),
        counts,
        out=means,
        where=counts > 0,
    )
    metadata = {
        "n_input": n,
        "n_extraction_declared": int(extraction.sum()),
        "n_extraction_ineligible": int((extraction & ~eligible).sum()),
        "n_eligible_before_valid_floor": int(pool.sum()),
        "n_below_valid_floor": int((pool & ~candidates).sum()),
        "n_candidates": int(candidates.sum()),
        "n_candidate_groups": len(set(np.asarray(group_ids)[candidates])),
        "n_valid_responses": int(counts[candidates].sum()),
        "n_missing_responses": int((~valid[candidates]).sum()),
        "n_complete_prompts": int((counts[candidates] == expected_draws).sum()),
        "valid_count_histogram": {
            str(k): int((counts[pool] == k).sum()) for k in range(expected_draws + 1)
        },
        "selection_draw_indices": np.flatnonzero(valid[candidates].any(axis=0)).tolist(),
        "expected_draws": expected_draws,
        "min_valid": min_valid,
        "score_range": [float(lower), float(upper)],
        "naturalness": "caller_audited_eligibility_mask",
        "partition_checks": ["context_ids", "group_ids", "prompt_hashes"],
    }
    if candidates.sum() < 2:
        raise SelectionError(
            "Fewer than two eligible prompts with sufficient valid scores", metadata=metadata
        )
    metadata["score_quantiles"] = {
        str(q): float(np.quantile(means[candidates], q))
        for q in (0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1)
    }
    return valid, counts, means, candidates, metadata


def _row_weights(indices: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Average valid responses within prompt, then average prompts equally."""
    if len(indices) == 0 or (valid[indices].sum(axis=1) == 0).any():
        raise SelectionError("Cannot construct weights for an empty or unscored tail")
    weights = np.zeros(valid.shape, dtype=np.float64)
    weights[indices] = valid[indices] / (len(indices) * valid[indices].sum(axis=1)[:, None])
    return weights


def _selection(
    high: np.ndarray,
    low: np.ndarray,
    *,
    valid: np.ndarray,
    counts: np.ndarray,
    means: np.ndarray,
    candidates: np.ndarray,
    group_ids: Sequence[str],
    metadata: dict[str, Any],
) -> TailSelection:
    """Create an audited contrast with disjoint prompts and group diagnostics."""
    groups = np.asarray(group_ids)
    overlap = sorted(set(groups[high]) & set(groups[low]))
    if np.intersect1d(high, low).size:
        raise SelectionError(
            "High and low tails must contain disjoint prompts",
            metadata=metadata,
        )
    metadata = dict(metadata)
    metadata["n_cross_tail_groups"] = len(overlap)
    for name, selected in (("high", high), ("low", low)):
        cutoff = float(means[selected].min() if name == "high" else means[selected].max())
        metadata[name] = {
            "n_prompts": len(selected),
            "n_groups": len(set(groups[selected])),
            "n_valid_responses": int(counts[selected].sum()),
            "valid_response_rate": float(valid[selected].mean()),
            "mean_score": float(means[selected].mean()),
            "score_sd": float(means[selected].std()),
            "min_score": float(means[selected].min()),
            "max_score": float(means[selected].max()),
            "cutoff": cutoff,
            "n_boundary_ties_pool": int((means[candidates] == cutoff).sum()),
            "n_boundary_ties_selected": int((means[selected] == cutoff).sum()),
        }
    metadata["high_minus_low_score"] = (
        metadata["high"]["mean_score"] - metadata["low"]["mean_score"]
    )
    if metadata["high_minus_low_score"] <= 0:
        raise SelectionError(
            "Selected tails have no positive high-minus-low score gap", metadata=metadata
        )
    return TailSelection(
        high_weights=_row_weights(high, valid),
        low_weights=_row_weights(low, valid),
        high_indices=high,
        low_indices=low,
        metadata=metadata,
        prompt_scores=means,
        valid_counts=counts,
    )


def select_prompt_tails(
    scores: np.ndarray,
    *,
    context_ids: Sequence[str],
    group_ids: Sequence[str],
    prompt_hashes: Sequence[str],
    extraction_mask: np.ndarray,
    eligible_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    score_range: tuple[float, float],
    behavior: str,
    fold: str,
    q: float = 0.01,
    tie_salt: str = "0",
    min_valid: int = 3,
    expected_draws: int = 5,
) -> TailSelection:
    """Select fixed top/bottom q prompt means using deterministic hash tie breaks.

    Each side contains max(1, floor(q*N)) prompts, where N counts only eligible
    extraction prompts meeting the valid-response floor. The canonical primary
    uses q=.01 and three valid judgments from five original response draws.
    q=.05/.10 and independently salted ties are prespecified sensitivities.
    Hash keys use behavior, fold, normalized-prompt hash, and tie_salt; length
    prefixes define the separator unambiguously. No score rescaling is inferred.
    """
    if not np.isfinite(q) or not 0 < q <= 0.5:
        raise ValueError("q must be in (0, 0.5]")
    if not all(isinstance(s, str) and s for s in (behavior, fold, tie_salt)):
        raise ValueError("behavior, fold, and tie_salt must be nonempty strings")
    valid, counts, means, candidates, metadata = _prepare(
        scores,
        context_ids=context_ids,
        group_ids=group_ids,
        prompt_hashes=prompt_hashes,
        extraction_mask=extraction_mask,
        eligible_mask=eligible_mask,
        evaluation_mask=evaluation_mask,
        score_range=score_range,
        min_valid=min_valid,
        expected_draws=expected_draws,
    )
    if np.ptp(means[candidates]) == 0:
        raise SelectionError(
            "Constant eligible prompt means; no elicitation contrast", metadata=metadata
        )
    indices = np.flatnonzero(candidates)
    keys = {i: _digest(behavior, fold, prompt_hashes[i], tie_salt) for i in indices}
    k = max(1, int(np.floor(q * len(indices))))
    ranked = np.asarray(sorted(indices, key=lambda i: (means[i], keys[i])), dtype=int)
    low = ranked[:k]
    high = ranked[-k:][::-1]
    metadata.update(
        {
            "method": "prompt_quantile",
            "q": q,
            "k": k,
            "behavior": behavior,
            "fold": fold,
            "tie_salt": tie_salt,
            "tie_break": "sha256_length_prefixed_utf8(behavior,fold,prompt_hash,tie_salt)",
        }
    )
    return _selection(
        high,
        low,
        valid=valid,
        counts=counts,
        means=means,
        candidates=candidates,
        group_ids=group_ids,
        metadata=metadata,
    )


def select_literal_endpoints(
    scores: np.ndarray,
    *,
    context_ids: Sequence[str],
    group_ids: Sequence[str],
    prompt_hashes: Sequence[str],
    extraction_mask: np.ndarray,
    eligible_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    score_range: tuple[float, float],
    min_groups: int = 20,
    min_valid: int = 3,
    expected_draws: int = 5,
) -> TailSelection:
    """Contrast literal scale endpoints when both have min_groups source groups.

    These are the declared scale endpoints, not the observed sample maximum and
    minimum. Insufficient support raises SelectionError with endpoint counts;
    callers should persist it as an unsupported diagnostic, never as zero.
    """
    if not isinstance(min_groups, int) or min_groups < 1:
        raise ValueError("min_groups must be a positive integer")
    valid, counts, means, candidates, metadata = _prepare(
        scores,
        context_ids=context_ids,
        group_ids=group_ids,
        prompt_hashes=prompt_hashes,
        extraction_mask=extraction_mask,
        eligible_mask=eligible_mask,
        evaluation_mask=evaluation_mask,
        score_range=score_range,
        min_valid=min_valid,
        expected_draws=expected_draws,
    )
    low = np.flatnonzero(candidates & (means == score_range[0]))
    high = np.flatnonzero(candidates & (means == score_range[1]))
    groups = np.asarray(group_ids)
    metadata.update(
        {
            "method": "literal_endpoints",
            "min_groups": min_groups,
            "endpoint_low_prompts": len(low),
            "endpoint_high_prompts": len(high),
            "endpoint_low_groups": len(set(groups[low])),
            "endpoint_high_groups": len(set(groups[high])),
        }
    )
    if min(metadata["endpoint_low_groups"], metadata["endpoint_high_groups"]) < min_groups:
        raise SelectionError(
            "Insufficient distinct-group support at literal endpoints", metadata=metadata
        )
    return _selection(
        high,
        low,
        valid=valid,
        counts=counts,
        means=means,
        candidates=candidates,
        group_ids=group_ids,
        metadata=metadata,
    )


def split_half_weights(
    selection: TailSelection,
    *,
    group_ids: Sequence[str],
    behavior: str,
    fold: str,
    salt: str = "half0",
) -> tuple[TailWeights, TailWeights]:
    """Split each selected tail by source group for extraction-half stability.

    Groups stay intact. Each half reweights its own prompts equally, so unequal
    group sizes do not turn the contrast into a group-weighted estimator.
    """
    groups = _strings(group_ids, len(selection.prompt_scores), "group_ids")
    high_groups = set(groups[selection.high_indices])
    low_groups = set(groups[selection.low_indices])
    if min(len(high_groups), len(low_groups)) < 2:
        raise SelectionError("Each tail needs at least two source groups for half stability")
    assignment: dict[str, int] = {}
    n_high, n_low = [0, 0], [0, 0]
    # Assign shared groups once, then balance each tail's remaining groups.
    for category in (high_groups & low_groups, high_groups - low_groups, low_groups - high_groups):
        for group in sorted(category, key=lambda g: _digest(behavior, fold, g, salt)):
            relevant_counts = n_high if group in high_groups else n_low
            half = 0 if relevant_counts[0] <= relevant_counts[1] else 1
            assignment[group] = half
            n_high[half] += int(group in high_groups)
            n_low[half] += int(group in low_groups)
    halves = [
        [
            np.asarray([i for i in selected if assignment[groups[i]] == half], dtype=int)
            for selected in (selection.high_indices, selection.low_indices)
        ]
        for half in (0, 1)
    ]
    valid = (selection.high_weights + selection.low_weights) > 0
    results = []
    for half, (high, low) in enumerate(halves):
        results.append(
            TailWeights(
                high_weights=_row_weights(high, valid),
                low_weights=_row_weights(low, valid),
                high_indices=high,
                low_indices=low,
                metadata={"half": half, "salt": salt, "n_high": len(high), "n_low": len(low)},
            )
        )
    return results[0], results[1]


def heldout_score_gap(
    selection: TailSelection,
    scores: np.ndarray,
    *,
    draw_mask: np.ndarray,
) -> dict[str, Any]:
    """Measure selected-tail score separation on caller-declared held-out draws.

    The caller must select using the complementary draws first. Unscored prompt
    measurements are omitted and counted; no missing score becomes zero.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape != selection.high_weights.shape:
        raise ValueError("Held-out score shape differs from the selection grid")
    draws = _mask(draw_mask, scores.shape[1], "draw_mask")
    if not draws.any() or draws.all():
        raise ValueError("draw_mask must select a nonempty proper subset of response draws")
    if set(selection.metadata["selection_draw_indices"]) & set(np.flatnonzero(draws)):
        raise SelectionError("Held-out measurement draws were used by tail selection")
    output: dict[str, Any] = {"draw_indices": np.flatnonzero(draws).tolist()}
    for name, selected in (("high", selection.high_indices), ("low", selection.low_indices)):
        subset = scores[selected][:, draws]
        if np.isinf(subset).any():
            raise ValueError("Infinite held-out scores are invalid")
        valid = np.isfinite(subset)
        lower, upper = selection.metadata["score_range"]
        if ((subset[valid] < lower) | (subset[valid] > upper)).any():
            raise ValueError("Held-out scores fall outside the selection's declared native range")
        keep = valid.any(axis=1)
        if not keep.any():
            raise SelectionError(f"No scored held-out prompts in {name} tail")
        prompt_means = np.where(valid, subset, 0).sum(axis=1)[keep] / valid.sum(axis=1)[keep]
        output[name] = {
            "mean_score": float(prompt_means.mean()),
            "n_scored_prompts": int(keep.sum()),
            "n_unscored_prompts": int((~keep).sum()),
            "n_valid_responses": int(valid.sum()),
        }
    output["high_minus_low_score"] = output["high"]["mean_score"] - output["low"]["mean_score"]
    return output
