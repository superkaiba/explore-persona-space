# ruff: noqa: RUF002, RUF003  # em-dash, minus sign, multiplication sign intentional
"""Task #601 — pre-registered classification lattice (plan §6), pure functions.

Everything here operates on plain floats / dicts so the local CPU smoke can
exercise the full lattice (including the no-call branch) on a synthetic
fixture. The script wrapper ``scripts/i601_analyze.py`` does the JSON I/O.

Registered constants (plan §6, grounded in the parent trajectories):
  reference levels L(0:1)=2.05, L(2:1)=8.55, L(4:1)=13.51, L(8:1)=20.00;
  logP tolerance ±3 nats; seed half-tolerance grace 1.5 nats; arrest =
  2-step-smoothed forward difference < 0.2 nats/step for 3 consecutive reads;
  matched-pair discriminator at absolute step 32 (horizon >= 17 vs coupling
  <= 11); log-Z artifact = Δz_marker >= 0.5 logit/step while ΔlogP slope <
  0.2; Phase-4 non-arrest = ΔG >= 6 by step 13 AND last-3-step slope >= 0.3.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("issue_601.analysis_lib")

# Parent reference levels in logP space (plan §2 table, pooled).
L_REFS: dict[str, float] = {"0:1": 2.05, "2:1": 8.55, "4:1": 13.51, "8:1": 20.00}
LOGP_TOL_NATS = 3.0
SEED_GRACE_NATS = 1.5
ARREST_RATE_NATS_PER_STEP = 0.2
ARREST_CONSECUTIVE = 3
LOGZ_ARTIFACT_DZ_PER_STEP = 0.5
MATCHED_PAIR_STEP = 32
HORIZON_UPPER_LOGP = 17.0
COUPLING_STEP32_MAX_LOGP = 11.0
QUARTER_HORIZON_MAX_LOGP = 9.0
COUPLING_QUARTER_TOL = 2.5
COUPLING_ACCRUAL_MIN_SLOPE = 0.1  # nats/step over steps 32–64.
HORIZON_FRAC_OF_TERMINAL_BY_16 = 0.80
PHASE4_NONARREST_MIN_NATS = 6.0
PHASE4_NONARREST_MIN_SLOPE = 0.3
PHASE4_ARREST_MAX_NATS = 4.0
PHASE4_ARREST_MAX_SLOPE = 0.2
PHASE3_DIFFERENTIAL_LOGITS = 1.0


def reexpress_threshold(
    x_logp: float,
    margin_refs: dict[str, float],
    l_refs: dict[str, float] | None = None,
) -> float:
    """Re-express a logP-space threshold in margin space (plan §4 item 2).

    Piecewise-linear interpolation through the measured (L(level), M(level))
    anchor points, extrapolating linearly at the ends. Deterministic given the
    Phase-0a on-policy margin references.
    """
    l_refs = l_refs or L_REFS
    pts = sorted((l_refs[k], margin_refs[k]) for k in margin_refs if k in l_refs)
    if len(pts) < 2:
        raise ValueError(f"need >=2 (L, M) anchor points, got {len(pts)}")
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if x_logp <= xs[0]:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return float(ys[0] + slope * (x_logp - xs[0]))
    if x_logp >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(ys[-1] + slope * (x_logp - xs[-1]))
    return float(np.interp(x_logp, xs, ys))


def seed_band_ok(
    values: list[float], center: float, tol: float, grace: float | None = None
) -> bool:
    """Seed rule (plan §6): mean inside [center±tol] AND each seed inside ±(tol+grace).

    2/2 sign-consistency is NOT required for band membership (seeds may
    straddle the band center); it IS required for directional claims — see
    :func:`sign_consistent`.
    """
    if not values:
        raise ValueError("seed_band_ok: empty values")
    g = SEED_GRACE_NATS if grace is None else grace
    mean = float(np.mean(values))
    if not (center - tol <= mean <= center + tol):
        return False
    return all(center - tol - g <= v <= center + tol + g for v in values)


def sign_consistent(deltas: list[float]) -> bool:
    """2/2 (n/n) sign consistency for directional / accrual claims."""
    if not deltas:
        return False
    signs = {np.sign(d) for d in deltas if d != 0}
    return len(signs) == 1 and 0.0 not in [float(d) for d in deltas]


def smoothed_forward_diff(steps: list[int], values: list[float], window: int = 2) -> list[float]:
    """2-step-smoothed forward difference (nats per step), per read.

    diff[i] = (smoothed[i+1] − smoothed[i]) / (steps[i+1] − steps[i]) where
    smoothed is a trailing mean over ``window`` reads. Length = len − 1.
    """
    if len(steps) != len(values):
        raise ValueError(f"steps/values length mismatch: {len(steps)} vs {len(values)}")
    if len(steps) < 2:
        return []
    v = np.asarray(values, dtype=float)
    s = np.asarray(steps, dtype=float)
    sm = np.array([v[max(0, i - window + 1) : i + 1].mean() for i in range(len(v))])
    return [float((sm[i + 1] - sm[i]) / (s[i + 1] - s[i])) for i in range(len(v) - 1)]


def arrest_step(
    steps: list[int],
    values: list[float],
    *,
    rate_thresh: float = ARREST_RATE_NATS_PER_STEP,
    consecutive: int = ARREST_CONSECUTIVE,
) -> int | None:
    """First step where the smoothed forward difference stays < rate for N reads.

    Plan §6 arrest definition. Returns the step at the START of the first
    qualifying run, or None when the series never arrests.
    """
    diffs = smoothed_forward_diff(steps, values)
    run = 0
    for i, d in enumerate(diffs):
        if d < rate_thresh:
            run += 1
            if run >= consecutive:
                return int(steps[i - consecutive + 1])
        else:
            run = 0
    return None


def logz_artifact(
    steps: list[int],
    delta_logp: list[float],
    delta_z_marker: list[float],
) -> dict:
    """Log-Z artifact flag (plan §6): Δz_marker grows >= 0.5 logit/step where ΔlogP < 0.2.

    Returns {"artifact": bool, "n_flagged_reads": int} — flagged reads are
    intervals where the z-slope clears 0.5 while the logp-slope is under 0.2.
    """
    d_lp = smoothed_forward_diff(steps, delta_logp)
    d_z = smoothed_forward_diff(steps, delta_z_marker)
    flagged = sum(
        1
        for a, b in zip(d_lp, d_z, strict=True)
        if a < ARREST_RATE_NATS_PER_STEP and b >= LOGZ_ARTIFACT_DZ_PER_STEP
    )
    return {"artifact": flagged > 0, "n_flagged_reads": flagged}


def _read_at_step(steps: list[int], values: list[float], target: int) -> float | None:
    """Value at the read nearest ``target`` (exact preferred; None if empty)."""
    if not steps:
        return None
    arr = np.asarray(steps)
    i = int(np.argmin(np.abs(arr - target)))
    return float(values[i])


def classify_phase1(
    *,
    arm_terminals: dict[str, list[float]],
    matched_series_by_seed: dict[int, tuple[list[int], list[float]]],
    space: str,
    margin_refs: dict[str, float] | None = None,
    margin_tol: float | None = None,
) -> dict:
    """The §6 three-hypothesis classification + precedence + matched-pair read.

    Args:
        arm_terminals: per-arm terminal source levels, one value per seed, in
            the PRIMARY space. Keys: "quarter", "anchor", "double", "matched".
        matched_series_by_seed: the schedule-matched arm's (steps, values)
            series per seed (dense ladder + frac reads), primary space.
        space: "logp" or "margin" (the Phase-0a-selected primary space).
        margin_refs: M(level) when space == "margin" (required then).
        margin_tol: derived margin tolerance (required when space == "margin").

    Returns a dict with per-hypothesis verdicts, the matched-pair
    discriminator, and the final call ("equilibrium" | "horizon" | "coupling"
    | "no-call").
    """
    if space not in ("logp", "margin"):
        raise ValueError(f"space={space!r}")
    if space == "margin":
        if not margin_refs or margin_tol is None:
            raise ValueError("margin space requires margin_refs + margin_tol")
        refs = {k: margin_refs[k] for k in margin_refs}
        tol = float(margin_tol)
        upper_thr = (refs["4:1"] + refs["8:1"]) / 2.0  # midpoint(M(4:1), M(8:1))
        quarter_max = reexpress_threshold(QUARTER_HORIZON_MAX_LOGP, margin_refs)
        coupling_step32_max = reexpress_threshold(COUPLING_STEP32_MAX_LOGP, margin_refs)
        horizon_step32_min = upper_thr
        coupling_quarter_tol = tol * (COUPLING_QUARTER_TOL / LOGP_TOL_NATS)
        accrual_min_slope = COUPLING_ACCRUAL_MIN_SLOPE  # slope units differ; reported as-is.
    else:
        refs = dict(L_REFS)
        tol = LOGP_TOL_NATS
        upper_thr = HORIZON_UPPER_LOGP
        quarter_max = QUARTER_HORIZON_MAX_LOGP
        coupling_step32_max = COUPLING_STEP32_MAX_LOGP
        horizon_step32_min = HORIZON_UPPER_LOGP
        coupling_quarter_tol = COUPLING_QUARTER_TOL
        accrual_min_slope = COUPLING_ACCRUAL_MIN_SLOPE

    means = {arm: float(np.mean(v)) for arm, v in arm_terminals.items()}
    fixed_ratio_arms = [a for a in ("quarter", "anchor", "double") if a in means]
    fixed_spread = max(means[a] for a in fixed_ratio_arms) - min(means[a] for a in fixed_ratio_arms)

    # ── Ratio set-point (phenomenology). ─────────────────────────────────────
    ratio_ok = (
        all(seed_band_ok(arm_terminals[a], refs["4:1"], tol) for a in fixed_ratio_arms)
        and seed_band_ok(arm_terminals["matched"], refs["4:1"], tol)
        and fixed_spread <= tol
    )

    # ── Horizon. ─────────────────────────────────────────────────────────────
    quarter_low = float(np.mean(arm_terminals["quarter"])) <= quarter_max
    double_high = float(np.mean(arm_terminals["double"])) >= upper_thr
    matched_high = float(np.mean(arm_terminals["matched"])) >= upper_thr
    # Matched arm reaches >= 80% of its terminal Δ by absolute step 16, per seed.
    frac16_by_seed: dict[int, float | None] = {}
    early_ok_all = True
    for seed, (steps, vals) in matched_series_by_seed.items():
        terminal = vals[-1] if vals else None
        v16 = _read_at_step(steps, vals, 16)
        frac16 = (v16 / terminal) if (terminal not in (None, 0) and v16 is not None) else None
        frac16_by_seed[seed] = frac16
        early_ok_all = early_ok_all and (
            frac16 is not None and frac16 >= HORIZON_FRAC_OF_TERMINAL_BY_16
        )
    horizon_ok = quarter_low and double_high and matched_high and early_ok_all

    # ── Coupling. ────────────────────────────────────────────────────────────
    quarter_at_2to1 = seed_band_ok(arm_terminals["quarter"], refs["2:1"], coupling_quarter_tol)
    double_at_8to1 = seed_band_ok(arm_terminals["double"], refs["8:1"], tol)
    matched_at_8to1 = seed_band_ok(arm_terminals["matched"], refs["8:1"], tol)
    quarter_terminal_mean = float(np.mean(arm_terminals["quarter"]))
    accrual_by_seed: dict[int, dict] = {}
    accruing_all = True
    accrual_deltas: list[float] = []
    for seed, (steps, vals) in matched_series_by_seed.items():
        v32 = _read_at_step(steps, vals, MATCHED_PAIR_STEP)
        v64 = _read_at_step(steps, vals, 64)
        slope_32_64 = (
            (v64 - v32) / (64 - MATCHED_PAIR_STEP)
            if (v32 is not None and v64 is not None)
            else None
        )
        step32_ok = v32 is not None and v32 <= quarter_terminal_mean + coupling_quarter_tol
        slope_ok = slope_32_64 is not None and slope_32_64 >= accrual_min_slope
        accrual_by_seed[seed] = {"v32": v32, "v64": v64, "slope_32_64": slope_32_64}
        if slope_32_64 is not None:
            accrual_deltas.append(slope_32_64)
        accruing_all = accruing_all and step32_ok and slope_ok
    # Directional/accrual claims need 2/2 sign consistency (plan §6 seed rule).
    accruing_all = accruing_all and sign_consistent(accrual_deltas)
    coupling_ok = quarter_at_2to1 and double_at_8to1 and matched_at_8to1 and accruing_all

    # ── Registered matched-pair discriminator (PRIMARY horizon-vs-coupling). ─
    v32_means = [v["v32"] for v in accrual_by_seed.values() if v["v32"] is not None]
    v32_mean = float(np.mean(v32_means)) if v32_means else None
    if v32_mean is None:
        matched_pair = "unreadable"
    elif v32_mean >= horizon_step32_min:
        matched_pair = "horizon"
    elif v32_mean <= coupling_step32_max:
        matched_pair = "coupling"
    else:
        matched_pair = "between"

    verdicts = {
        "equilibrium": bool(ratio_ok),
        "horizon": bool(horizon_ok),
        "coupling": bool(coupling_ok),
    }
    satisfied = [k for k, v in verdicts.items() if v]
    call = satisfied[0] if len(satisfied) == 1 else "no-call"

    return {
        "space": space,
        "refs": refs,
        "tolerance": tol,
        "arm_terminal_means": means,
        "fixed_ratio_spread": float(fixed_spread),
        "verdicts": verdicts,
        "call": call,
        "n_satisfied": len(satisfied),
        "matched_pair_discriminator": {
            "step": MATCHED_PAIR_STEP,
            "v32_mean": v32_mean,
            "horizon_min": horizon_step32_min,
            "coupling_max": coupling_step32_max,
            "verdict": matched_pair,
        },
        "horizon_detail": {
            "quarter_low": quarter_low,
            "double_high": double_high,
            "matched_high": matched_high,
            "frac_of_terminal_by_step16": frac16_by_seed,
        },
        "coupling_detail": {
            "quarter_at_2to1": quarter_at_2to1,
            "double_at_8to1": double_at_8to1,
            "matched_at_8to1": matched_at_8to1,
            "accrual_by_seed": accrual_by_seed,
            "accruing_all": accruing_all,
        },
    }


def classify_phase4_arrest(steps: list[int], delta_g: list[float]) -> dict:
    """Phase 4 arrest on/off classification (plan §4).

    non-arrest: ΔG >= 6 nats by step 13 AND last-3-step slope >= 0.3.
    arrest: flat (slope < 0.2 from step <= 4) at <= 4 nats.
    else ambiguous (reported descriptively; the Phase-4 kill does NOT fire).
    """
    if not steps:
        return {"classification": "unreadable"}
    v13 = _read_at_step(steps, delta_g, 13)
    diffs = smoothed_forward_diff(steps, delta_g)
    last3 = diffs[-3:] if len(diffs) >= 3 else diffs
    last3_slope = float(np.mean(last3)) if last3 else None
    non_arrest = (
        v13 is not None
        and v13 >= PHASE4_NONARREST_MIN_NATS
        and last3_slope is not None
        and last3_slope >= PHASE4_NONARREST_MIN_SLOPE
    )
    # Arrest: slope < 0.2 from step <= 4 onward, terminal level <= 4 nats.
    from_step4 = [d for s, d in zip(steps[1:], diffs, strict=True) if s >= 4]
    flat_from_4 = bool(from_step4) and all(d < PHASE4_ARREST_MAX_SLOPE for d in from_step4)
    terminal = delta_g[-1]
    arrest = flat_from_4 and terminal <= PHASE4_ARREST_MAX_NATS
    if non_arrest:
        cls = "non-arrest"
    elif arrest:
        cls = "arrest"
    else:
        cls = "ambiguous"
    return {
        "classification": cls,
        "delta_g_at_step13": v13,
        "last3_slope": last3_slope,
        "terminal": float(terminal),
        "flat_from_step4": flat_from_4,
    }


def phase3_contrast(
    per_seed: dict[int, dict],
) -> dict:
    """Phase 3 re-registered source-DIFFERENTIAL contrast (plan §6).

    Args:
        per_seed: ``{seed: {"dz_marker_source", "dz_marker_bystander_mean",
        "dz_eos_source", "dz_eos_bystander_mean", "delta_margin_source",
        "delta_logZ_source"}}`` at terminal.

    Positive iff seed-mean |Δz_marker(source) − mean Δz_marker(bystanders)|
    >= 1.0 logit OR |Δz_eos(source) − mean Δz_eos(bystanders)| >= 1.0 logit,
    2/2 seeds sign-consistent on the firing channel. Uniform movement is
    reported nonspecific.
    """
    dz_m = [s["dz_marker_source"] - s["dz_marker_bystander_mean"] for s in per_seed.values()]
    dz_e = [s["dz_eos_source"] - s["dz_eos_bystander_mean"] for s in per_seed.values()]
    marker_fires = abs(float(np.mean(dz_m))) >= PHASE3_DIFFERENTIAL_LOGITS and sign_consistent(dz_m)
    eos_fires = abs(float(np.mean(dz_e))) >= PHASE3_DIFFERENTIAL_LOGITS and sign_consistent(dz_e)
    return {
        "positive": bool(marker_fires or eos_fires),
        "marker_channel_differential": [float(x) for x in dz_m],
        "eos_channel_differential": [float(x) for x in dz_e],
        "marker_channel_fires": bool(marker_fires),
        "eos_channel_fires": bool(eos_fires),
        "note": (
            "differential contrast vs the 8-bystander reference; uniform source+bystander "
            "movement is generic drift, NOT an init-live coupling positive (plan §6)"
        ),
    }


def robustness_sweep(
    steps: list[int],
    values: list[float],
    *,
    base_rate: float = ARREST_RATE_NATS_PER_STEP,
) -> dict:
    """±50% sweep of the arrest-rate constant (plan §11 robustness check)."""
    out = {}
    for label, rate in (("-50%", base_rate * 0.5), ("base", base_rate), ("+50%", base_rate * 1.5)):
        out[label] = arrest_step(steps, values, rate_thresh=rate)
    return out
