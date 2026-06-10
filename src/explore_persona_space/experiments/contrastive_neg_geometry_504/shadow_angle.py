# em-dash + Qwen marker " ※" + Greek ΔG intentional
"""Task #504 — shadow angle covariate (plan Appendix A.1).

Persona centroids live in R^d (d=4096 for Qwen-2.5-7B). The "source→N" and
"source→probe" directions span an angular plane through the SOURCE's centroid.
The SHADOW angle is the angle between these two directions:

  small angle ⇒ probe lies "behind" N relative to source (in N's angular shadow)
  large angle ⇒ probe is lateral to the source→N direction

Range: [0, π]. This is the SOURCE-anchored projection variant (plan §4.2 +
Appendix A.1). The alternative — angle between centroid vectors at the origin
— would conflate "behind N" with "near N angularly from origin," which doesn't
match the geometric intuition the bubble-vs-barrier hypothesis tests.
"""

from __future__ import annotations

import math

import numpy as np


def shadow_angle(
    probe_centroid: np.ndarray,
    n_arm_centroid: np.ndarray,
    source_centroid: np.ndarray,
) -> float:
    """Angle (radians) between source→N_arm and source→probe at the source vertex.

    Args:
        probe_centroid: persona-vector for the held-out probe (shape (d,)).
        n_arm_centroid: persona-vector for the arm's positioned negative N
            (shape (d,)).
        source_centroid: persona-vector for the source persona (shape (d,)).

    Returns:
        Angle in radians in [0, π]. NaN if either direction has zero norm
        (degenerate: probe == source OR N_arm == source).

    The function does NOT enforce dtype — numpy promotes inputs to a common
    type. Centroids are typically float32; the dot/norm cast to float for
    numerical safety on the cos clamp.
    """
    v_sn = np.asarray(n_arm_centroid, dtype=np.float64) - np.asarray(
        source_centroid, dtype=np.float64
    )
    v_sp = np.asarray(probe_centroid, dtype=np.float64) - np.asarray(
        source_centroid, dtype=np.float64
    )
    norm_sn = float(np.linalg.norm(v_sn))
    norm_sp = float(np.linalg.norm(v_sp))
    if norm_sn == 0.0 or norm_sp == 0.0:
        return float("nan")
    cos_angle = float(np.dot(v_sn, v_sp)) / (norm_sn * norm_sp)
    # Numerical guard against floating-point overshoot of ±1.
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(cos_angle)


def shadow_angles_for_panel(
    panel: list[str],
    arm_negative: str,
    source: str,
    centroids: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute the shadow angle for every probe in `panel` under one arm.

    Args:
        panel: held-out probe persona names.
        arm_negative: positioned-N persona name for this arm.
        source: source persona name.
        centroids: {persona: centroid vector} at the chosen layer.

    Returns:
        {probe_name: shadow_angle_radians}. NaN for any probe whose probe
        centroid equals source centroid (degenerate).

    Raises:
        KeyError: any persona name missing from `centroids`.
    """
    out: dict[str, float] = {}
    src = centroids[source]
    n = centroids[arm_negative]
    for probe in panel:
        out[probe] = shadow_angle(centroids[probe], n, src)
    return out


def gate_a_identification_floor(
    panel: list[str],
    arms: dict[str, str],  # arm_slug -> positioned_negative_persona
    cos_matrix: dict[str, dict[str, float]],
    centroids: dict[str, np.ndarray],
    source: str,
    *,
    d_nn_floor: float,
    shadow_floor_rad: float,
) -> dict:
    """Pre-flight identification gate A — must pass at the headline layer.

    Plan Appendix A.3: across the positioned arms (default-only arm excluded —
    no positioned negative to compute d_nn / shadow against), compute the
    median across-arm SD of d_nearest_neg_nd and shadow_angle per probe. Both
    must clear their floors.

    Args:
        panel: held-out probe persona names.
        arms: {arm_slug: positioned_negative_persona} for the 4 positioned arms.
        cos_matrix: {a: {b: cos(a, b)}} over the bank at the headline layer.
        centroids: {persona: centroid} at the same layer.
        source: source persona.
        d_nn_floor: minimum median across-arm SD of d_nearest_neg_nd (plan
            uses 0.02 — the #472 floor).
        shadow_floor_rad: minimum median across-arm SD of shadow_angle (plan
            uses 0.10 rad ≈ 5.7°).

    Returns:
        {'pass': bool, 'median_dnn_spread': float, 'median_shadow_spread':
        float, 'd_nn_floor': float, 'shadow_floor_rad': float, 'failures':
        [str, ...], 'per_probe': {probe: {'d_nn': {arm: ...}, 'shadow':
        {arm: ...}}}}
    """
    per_probe: dict[str, dict[str, dict[str, float]]] = {}
    for probe in panel:
        d_nn = {arm: 1.0 - cos_matrix[probe][arms[arm]] for arm in arms}
        shadow = {
            arm: shadow_angle(centroids[probe], centroids[arms[arm]], centroids[source])
            for arm in arms
        }
        per_probe[probe] = {"d_nn": d_nn, "shadow": shadow}
    dnn_spreads = [float(np.std(list(per_probe[p]["d_nn"].values()))) for p in panel]
    shadow_spreads = [float(np.std(list(per_probe[p]["shadow"].values()))) for p in panel]
    median_dnn = float(np.median(dnn_spreads)) if dnn_spreads else float("nan")
    median_shadow = float(np.median(shadow_spreads)) if shadow_spreads else float("nan")
    failures: list[str] = []
    if not (median_dnn >= d_nn_floor):
        failures.append(
            f"d_nearest_neg_nd median across-arm spread {median_dnn:.4f} < floor "
            f"{d_nn_floor} (the #472 identification floor)."
        )
    if not (median_shadow >= shadow_floor_rad):
        failures.append(
            f"shadow_angle median across-arm spread {median_shadow:.4f} rad < floor "
            f"{shadow_floor_rad} rad."
        )
    return {
        "pass": len(failures) == 0,
        "median_dnn_spread": median_dnn,
        "median_shadow_spread": median_shadow,
        "d_nn_floor": float(d_nn_floor),
        "shadow_floor_rad": float(shadow_floor_rad),
        "failures": failures,
        "per_probe": per_probe,
    }


def gate_b_qwen_default_dominance(
    panel: list[str],
    arms: dict[str, str],
    cos_matrix: dict[str, dict[str, float]],
    default_persona: str,
    *,
    dominance_threshold: float,
) -> dict:
    """Gate B — default-assistant non-dominance check (plan §4.2 step 6).

    For each probe in `panel`, count the fraction of arms in which
    `default_persona` is the nearest negative (vs the positioned N). If that
    fraction across all probes is ≥ `dominance_threshold`, the positioned
    negative is invisible to the bubble predictor and the layer fails Gate B.

    Args:
        panel: held-out probe persona names.
        arms: {arm_slug: positioned_negative_persona} for the 4 positioned arms.
        cos_matrix: {a: {b: cos(a, b)}} at the headline layer.
        default_persona: the always-included default negative (qwen_default).
        dominance_threshold: max allowed fraction of probes where the default
            is the single nearest negative across ALL arms (plan uses 0.5).

    Returns:
        {'pass': bool, 'fraction_default_dominant': float, 'dominance_threshold':
        float}
    """
    n_dominated = 0
    for probe in panel:
        all_arms_dominated = True
        for arm in arms:
            d_default = 1.0 - cos_matrix[probe][default_persona]
            d_positioned = 1.0 - cos_matrix[probe][arms[arm]]
            if d_positioned < d_default:
                all_arms_dominated = False
                break
        if all_arms_dominated:
            n_dominated += 1
    frac = (n_dominated / len(panel)) if panel else 1.0
    return {
        "pass": frac < dominance_threshold,
        "fraction_default_dominant": float(frac),
        "dominance_threshold": float(dominance_threshold),
        "n_dominated": int(n_dominated),
        "n_panel": len(panel),
    }


def gate_c_panel_sufficiency(panel: list[str], *, min_probes: int) -> dict:
    """Gate C — held-out panel size sufficient for the pooled regression."""
    n = len(panel)
    return {
        "pass": n >= min_probes,
        "n_panel": n,
        "min_probes": int(min_probes),
    }
