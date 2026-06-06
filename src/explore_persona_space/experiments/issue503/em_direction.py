# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →, —, ρ) in scientific docstrings + logs.
"""Bucket D — H7-7c convergent-EM-direction extraction + projection (plan v2 §4.5).

Per plan §4.5 "Mechanism projection (the open-weights bonus,
DESCRIPTIVE per MF-8(a) + MF-7)":

    For each fine-tuned adapter in {D0, D1, D2, D3, D4} x 3 seeds = 15
    adapter-level data points:
      - Extract the post-FT vs pre-FT shift in residual stream at the
        broad-EM in-context prompt position (average delta in L25
        residual when the model is shown the broad-EM in-context vector
        per #486).
      - Extract the convergent misaligned-persona direction from a
        matched insecure-code-trained Qwen-2.5-7B-Instruct adapter via
        Soligo et al. rank-1-LoRA method.

    Required projection baselines (MF-7). Project the SAME 15
    benign-data-induced shifts onto:
      - (i) Norm-matched random directions. Sample 10-20 random unit
        vectors, scaled to the EM direction's norm; project the
        benign-data shift onto each. Report the empirical CI of
        cosines per selector.
      - (ii) Non-EM persona direction. Extract a Soligo-style rank-1
        direction from the `educational` source adapter (a near-zero-EM
        cell per #458) AND from the `secure_code` source adapter;
        project the benign-data shifts onto each.

    H7-7c interpretation rule (MF-8(a) descriptive-only):
      - The EM-direction cosine must cleanly exceed the norm-matched
        random CI upper bound AND the non-EM persona direction cosine
        before mechanism-sharing is asserted.
      - Reported as descriptive bars with 95% bootstrap CI; NO threshold
        gate; H7-7c is REMOVED from the H8 calibration headline.

This module is data-shape-only — it does NOT depend on a live model.
The dispatcher feeds in precomputed (post-FT - pre-FT) residual deltas
and the extracted Soligo rank-1 directions; this module's job is the
projection arithmetic + bootstrap CI + the per-cell descriptive verdict.

Plan reference: tasks/running/503/plans/v2.md §4.5 + MF-7 + MF-8.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_LAYER: int = 25
DEFAULT_POSITION_NAME: str = "p5"  # newline-after-`assistant`; #468 canonical

# Number of random direction baselines per MF-7 (plan: 10-20).
DEFAULT_N_RANDOM_DIRECTIONS: int = 16

# Bootstrap CI default per MF-8(a).
DEFAULT_BOOTSTRAP_N: int = 1000
DEFAULT_CI_ALPHA: float = 0.05  # 95% CI


DirectionKind = Literal["em_convergent", "non_em_educational", "non_em_secure_code", "random"]


# ── Data shapes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RankOneDirection:
    """A Soligo et al. rank-1 LoRA direction extraction.

    The Soligo et al. rank-1-LoRA method extracts a single direction
    ``d in R^d_model`` such that the LoRA's contribution at a given
    (layer, position) is approximately ``alpha * d * d^T * x``. The
    unit-vector ``d`` is the "direction" the LoRA writes along.
    """

    kind: DirectionKind
    layer: int
    position_name: str
    # Unit-vector at the given (layer, position). Shape (d_model,).
    direction: np.ndarray
    # Magnitude of the rank-1 LoRA's writing (alpha). Reported but not
    # consumed by the projection cosine.
    alpha: float = 1.0
    # Optional human-readable source label (e.g. "insecure_code_seed0").
    source_label: str = ""


@dataclass(frozen=True)
class ResidualShift:
    """A post-FT minus pre-FT residual delta at one selector adapter.

    The shift is averaged across the broad-EM in-context probe set
    (plan §4.5: "average delta in L25 residual when the model is shown
    the broad-EM in-context vector").
    """

    selector_id: str  # one of benign_data.ALL_SELECTORS
    seed: int
    layer: int
    position_name: str
    # The averaged (post-FT - pre-FT) residual delta. Shape (d_model,).
    delta: np.ndarray
    n_probes: int


@dataclass
class ProjectionResult:
    """One projection of a benign-data shift onto a direction.

    For descriptive reporting per MF-8(a). The CI columns hold the
    bootstrap CI bounds over the inner-row resample.
    """

    selector_id: str
    seed: int
    direction_kind: DirectionKind
    cosine: float
    # 95% bootstrap CI on the cosine (from per-probe resamples).
    ci_low: float = 0.0
    ci_high: float = 0.0


@dataclass
class H77cVerdict:
    """Per-selector × seed descriptive verdict per MF-8(a) interpretation rule.

    Per plan §4.5: "The EM-direction cosine must cleanly exceed the
    norm-matched random CI upper bound AND the non-EM persona direction
    cosine before mechanism-sharing is asserted."

    All thresholds REMOVED from the H8 headline; this is purely
    descriptive for the analyzer.
    """

    selector_id: str
    seed: int
    cosine_em: float
    cosine_random_ci_upper: float
    cosine_non_em_max: float  # max over educational + secure_code
    mechanism_share_descriptive: bool  # cosine_em > both
    # Optional per-direction breakdown (filled by the dispatcher).
    per_direction: dict[str, float] = field(default_factory=dict)


# ── Projection arithmetic ────────────────────────────────────────────────────


def _project_unit(delta: np.ndarray, direction_unit: np.ndarray) -> float:
    """Cosine of the residual delta against a unit direction.

    Plan §4.5 / MF-7: "project the benign-data shift onto each [direction]".
    The cosine measures how much of the delta's MAGNITUDE lies along the
    direction (1.0 = perfectly aligned, 0.0 = orthogonal, negative =
    anti-aligned).
    """
    assert delta.ndim == 1, delta.shape
    assert direction_unit.ndim == 1, direction_unit.shape
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm < 1e-12:
        return 0.0
    dir_norm = float(np.linalg.norm(direction_unit))
    if dir_norm < 1e-12:
        return 0.0
    return float(np.dot(delta, direction_unit) / (delta_norm * dir_norm))


def project(
    shift: ResidualShift,
    direction: RankOneDirection,
) -> ProjectionResult:
    """Cosine of ``shift.delta`` against ``direction.direction``.

    Validates (layer, position) match (the predictor read is canonical
    L25 / p5; cross-position projection is silent measurement-validity
    failure).
    """
    if shift.layer != direction.layer or shift.position_name != direction.position_name:
        raise ValueError(
            f"Layer/position mismatch: shift=(L{shift.layer}, {shift.position_name}) "
            f"direction=(L{direction.layer}, {direction.position_name})"
        )
    cos = _project_unit(shift.delta, direction.direction)
    return ProjectionResult(
        selector_id=shift.selector_id,
        seed=shift.seed,
        direction_kind=direction.kind,
        cosine=cos,
    )


# ── MF-7 baselines: norm-matched random + non-EM persona ─────────────────────


def sample_norm_matched_random_directions(
    em_direction: RankOneDirection,
    n_directions: int = DEFAULT_N_RANDOM_DIRECTIONS,
    *,
    seed: int = 0,
) -> list[RankOneDirection]:
    """MF-7(a): sample ``n_directions`` random unit vectors, scaled to the
    EM direction's norm.

    The "norm-matched" matters only when we keep ``alpha``/magnitude as
    a side channel. For the cosine itself, the unit-vector is what's
    projected; the norm equality is recorded for plotting.
    """
    rng = np.random.default_rng(seed)
    d = em_direction.direction.shape[0]
    em_norm = float(np.linalg.norm(em_direction.direction))
    out: list[RankOneDirection] = []
    for i in range(n_directions):
        v = rng.standard_normal(size=d)
        v = v / (np.linalg.norm(v) + 1e-12)
        out.append(
            RankOneDirection(
                kind="random",
                layer=em_direction.layer,
                position_name=em_direction.position_name,
                direction=v,
                alpha=em_norm,
                source_label=f"random_{i}",
            )
        )
    return out


def project_random_baseline_ci(
    shift: ResidualShift,
    em_direction: RankOneDirection,
    *,
    n_directions: int = DEFAULT_N_RANDOM_DIRECTIONS,
    seed: int = 0,
    alpha: float = DEFAULT_CI_ALPHA,
) -> dict:
    """Build the random-direction CI per MF-7(a).

    Sample ``n_directions`` norm-matched random unit vectors; project
    ``shift.delta`` onto each; return the empirical 95% CI of the
    cosine distribution. The CI upper bound is what the EM-direction
    cosine must EXCEED to claim mechanism-sharing per MF-8(a).
    """
    randoms = sample_norm_matched_random_directions(em_direction, n_directions, seed=seed)
    cosines = [_project_unit(shift.delta, d.direction) for d in randoms]
    cosines_arr = np.asarray(cosines)
    lo = float(np.quantile(cosines_arr, alpha / 2))
    hi = float(np.quantile(cosines_arr, 1 - alpha / 2))
    return {
        "n_directions": n_directions,
        "ci_low": lo,
        "ci_high": hi,
        "mean": float(np.mean(cosines_arr)),
        "all_cosines": cosines,
    }


# ── H7-7c per-selector verdict (descriptive only per MF-8(a)) ────────────────


def h7_7c_verdict(
    shift: ResidualShift,
    em_direction: RankOneDirection,
    non_em_directions: list[RankOneDirection],
    *,
    n_random_directions: int = DEFAULT_N_RANDOM_DIRECTIONS,
    rng_seed: int = 0,
    diagnostic_mode: bool = False,
) -> H77cVerdict:
    """Descriptive H7-7c verdict per MF-8(a) — no threshold gate.

    The EM-direction cosine is reported alongside:
      - the norm-matched-random CI upper bound (MF-7(a))
      - the max cosine across non-EM persona directions (MF-7(b))

    ``mechanism_share_descriptive`` is True iff the EM cosine exceeds
    BOTH the random CI upper AND the non-EM max — but this is a
    descriptive read, not a statistical test (MF-8(a)).

    Round-2 Rec 5: non-EM controls are MANDATORY per plan §4.5 / §6.2 /
    §13. Empty ``non_em_directions`` (with default ``non_em_max=0.0``)
    silently satisfies ``cos_em > non_em_max`` for ANY positive cos_em —
    that's an unconditional True on the descriptive read, not a
    legitimate non-EM-control verdict. Per CLAUDE.md fail-fast we now
    RAISE on empty non_em_directions unless the caller explicitly opts
    into ``diagnostic_mode=True`` (in which case mechanism_share_descriptive
    is forced False — the diagnostic verdict cannot be asserted without
    real controls). The opt-in lets a developer run the projection
    arithmetic for inspection without claiming the H7-7c gate.
    """
    if em_direction.kind != "em_convergent":
        raise ValueError(f"em_direction.kind must be 'em_convergent', got {em_direction.kind!r}")

    if not non_em_directions and not diagnostic_mode:
        raise ValueError(
            "MF-7(b) requires at least one non-EM persona direction control. "
            "Pass non_em_directions with kind in {'non_em_educational', "
            "'non_em_secure_code'} (plan §4.5 / §6.2 / §13). To inspect the "
            "EM-direction cosine + random-baseline CI without asserting the "
            "H7-7c descriptive verdict, pass diagnostic_mode=True; in that "
            "mode mechanism_share_descriptive is forced False."
        )

    cos_em = _project_unit(shift.delta, em_direction.direction)

    rand_ci = project_random_baseline_ci(
        shift, em_direction, n_directions=n_random_directions, seed=rng_seed
    )

    non_em_cosines: dict[str, float] = {}
    for d in non_em_directions:
        if d.kind not in ("non_em_educational", "non_em_secure_code"):
            raise ValueError(
                f"non-EM direction kind must be one of "
                f"('non_em_educational', 'non_em_secure_code'); got {d.kind!r}"
            )
        non_em_cosines[d.kind] = _project_unit(shift.delta, d.direction)
    # In diagnostic_mode with no non-EM directions, force the descriptive
    # share to False — we cannot assert mechanism-sharing without controls.
    if not non_em_cosines:
        non_em_max = float("nan")
        descriptive_share = False
    else:
        non_em_max = max(non_em_cosines.values())
        descriptive_share = (cos_em > rand_ci["ci_high"]) and (cos_em > non_em_max)
    per_direction = {
        "em_convergent": cos_em,
        "random_ci_high": rand_ci["ci_high"],
        "random_ci_low": rand_ci["ci_low"],
        "random_mean": rand_ci["mean"],
        **non_em_cosines,
    }
    return H77cVerdict(
        selector_id=shift.selector_id,
        seed=shift.seed,
        cosine_em=cos_em,
        cosine_random_ci_upper=rand_ci["ci_high"],
        cosine_non_em_max=non_em_max,
        mechanism_share_descriptive=descriptive_share,
        per_direction=per_direction,
    )


def h7_7c_disclaimer() -> str:
    """Standing methodological disclaimer for the H7-7c read (MF-8(a))."""
    return (
        "H7-7c is descriptive-only per MF-8(a). The mechanism-sharing read is the "
        "co-occurrence of: (cos_em > random CI upper bound) AND (cos_em > non-EM max). "
        "There is NO p-value gate, NO threshold of +0.20, and H7-7c is REMOVED from "
        "the H8 calibration headline. Treat as a qualitative signal, not a test."
    )
