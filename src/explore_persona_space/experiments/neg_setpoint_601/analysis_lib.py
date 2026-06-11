# ruff: noqa: RUF002, RUF003  # em-dash, minus sign, multiplication sign intentional
"""Task #601 — pre-registered classification lattice (plan §6), pure functions.

Everything here operates on plain floats / dicts so the local CPU smoke can
exercise the full lattice (including the no-call branch) on a synthetic
fixture. The script wrapper ``scripts/i601_analyze.py`` does the JSON I/O.

Registered rules (plan §6, references re-pinned IN-TASK by plan v3 §C):
  reference levels L̂(·)/M̂(·) instantiate from the NAMED in-task cells (§C
  sourcing table) via :func:`derive_in_task_references` — the parent-committed
  numerics (L=2.05/8.55/13.51/20.00, horizon-upper 17, coupling-step32 11)
  are RETIRED from the classification path and survive only in
  :data:`PARENT_COMMITTED_CROSS_RIG` for the output JSON's cross-rig
  reporting block. logP tolerance = max(3.0, 2× largest in-task 2-seed
  terminal gap); seed grace = tol/2; arrest = 2-step-smoothed forward
  difference < 0.2 nats/step for 3 consecutive reads; matched-pair
  discriminator at absolute step 32, ARM-INTERNAL (coupling ⇒ step-32 ≤
  quarter terminal + 2.5; horizon ⇒ step-32 ≥ 80% of the schedule-matched
  arm's own terminal), with the §C decidability guard (≥ 3-nat separation,
  margin-space fallback, underpowered branch) and the degenerate-top guard
  (L̂(8:1) within tol of L̂(4:1) in BOTH spaces → upper-branch level tests
  unresolvable-as-registered); log-Z artifact = Δz_marker >= 0.5 logit/step
  while ΔlogP slope < 0.2; Phase-4 non-arrest = ΔG >= 6 by step 13 AND
  last-3-step slope >= 0.3.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("issue_601.analysis_lib")

# Parent-committed numerics — CROSS-RIG REPORTING ONLY (plan v3 §C demoted
# them: the Phase-0a re-read proved the high-dose committed levels are not
# reproducible under this rig). NEVER consumed by classify_phase1; the
# round-7 fixture tests assert the classification path cannot fall back here.
PARENT_COMMITTED_CROSS_RIG: dict = {
    "l_refs_logp": {"0:1": 2.05, "2:1": 8.55, "4:1": 13.51, "8:1": 20.00},
    "horizon_upper_logp": 17.0,
    "coupling_step32_max_logp": 11.0,
    "quarter_horizon_max_logp": 9.0,
    "note": (
        "parent #472 committed levels — cross-rig comparison table only; the in-task "
        "references (plan v3 §C) are the classification inputs"
    ),
}
LOGP_TOL_FLOOR_NATS = 3.0  # never shrink a registered tolerance (plan v3 §C).
MARGIN_TOL_FLOOR_LOGITS = 1.0
FIXED_RATIO_SPREAD_MAX_NATS = 3.0  # ratio rule: spread <= 3 (plan v3 §C).
HORIZON_QUARTER_BELOW_RATIO_NATS = 3.0  # horizon: quarter <= L̂(4:1) − 3.
MATCHED_PAIR_DECIDABILITY_MIN = 3.0  # §C decidability guard separation.
HORIZON_STEP32_FRAC_OF_OWN_TERMINAL = 0.80  # horizon ⇒ step-32 ≥ 80% own terminal.
ARREST_RATE_NATS_PER_STEP = 0.2
ARREST_CONSECUTIVE = 3
LOGZ_ARTIFACT_DZ_PER_STEP = 0.5
MATCHED_PAIR_STEP = 32
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
    l_refs: dict[str, float],
) -> float:
    """Re-express a logP-space threshold in margin space (plan §4 item 2).

    Piecewise-linear interpolation through the measured (L̂(level), M̂(level))
    anchor points, extrapolating linearly at the ends. Both anchor dicts are
    REQUIRED (plan v3 §C: in-task references only — the parent-constant
    default is retired).
    """
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

    The default grace is tol/2 — the half-tolerance grace SCALES with the
    in-task-derived tolerance (plan v3 §C), never a fixed parent-era constant.
    2/2 sign-consistency is NOT required for band membership (seeds may
    straddle the band center); it IS required for directional claims — see
    :func:`sign_consistent`.
    """
    if not values:
        raise ValueError("seed_band_ok: empty values")
    g = (tol / 2.0) if grace is None else grace
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


def derive_in_task_references(
    *,
    level_terminals_logp: dict[str, list[float]],
    level_terminals_margin: dict[str, list[float]],
    extra_seed_pairs_logp: dict[str, list[float]] | None = None,
    extra_seed_pairs_margin: dict[str, list[float]] | None = None,
    sources: dict[str, list[str]] | None = None,
) -> dict:
    """Plan v3 §C — instantiate L̂(·)/M̂(·)/tolerances from NAMED in-task cells.

    Args:
        level_terminals_logp / level_terminals_margin: per-level terminal
            source values (one per seed) from the §C sourcing table —
            fresh Phase-2 / fallback terminals, NEVER parent committed
            numbers. All four levels ("0:1", "2:1", "4:1", "8:1") required
            and non-empty (no parent fallback exists).
        extra_seed_pairs_*: additional in-task cells (e.g. the Phase-1 arms)
            whose within-cell terminal seed gaps feed the tolerance
            derivation ("across the 2-seed cells").
        sources: provenance strings per level, recorded verbatim.

    Returns the refs dict :func:`classify_phase1` consumes: ``l_refs`` /
    ``m_refs`` (per-level means), ``tol_logp`` = max(3.0, 2× largest in-task
    2-seed gap — the 3.0 floor never shrinks; widening is flagged),
    ``tol_margin`` = max(1.0, 2× largest 2-seed margin gap), grace = tol/2,
    the single-seed L̂(8:1) flag with the largest 2-seed gap carried as its
    reported uncertainty term, and the degenerate-top flag (L̂(8:1) within
    tol of L̂(4:1) in BOTH spaces — plan v3 §C guard).
    """
    required = ("0:1", "2:1", "4:1", "8:1")
    for name, levels in (
        ("level_terminals_logp", level_terminals_logp),
        ("level_terminals_margin", level_terminals_margin),
    ):
        missing = [lv for lv in required if not levels.get(lv)]
        if missing:
            raise ValueError(
                f"derive_in_task_references: {name} missing/empty levels {missing} — the §C "
                f"references instantiate ONLY from in-task cells; there is no parent fallback."
            )

    l_refs = {lv: float(np.mean(level_terminals_logp[lv])) for lv in required}
    m_refs = {lv: float(np.mean(level_terminals_margin[lv])) for lv in required}

    def _gaps(by_cell: dict[str, list[float]]) -> dict[str, float]:
        return {k: float(max(v) - min(v)) for k, v in by_cell.items() if len(v) >= 2}

    gaps_logp = {**_gaps(level_terminals_logp), **_gaps(extra_seed_pairs_logp or {})}
    gaps_margin = {**_gaps(level_terminals_margin), **_gaps(extra_seed_pairs_margin or {})}
    max_gap_logp = max(gaps_logp.values(), default=0.0)
    max_gap_margin = max(gaps_margin.values(), default=0.0)
    tol_logp = max(LOGP_TOL_FLOOR_NATS, 2.0 * max_gap_logp)
    tol_margin = max(MARGIN_TOL_FLOOR_LOGITS, 2.0 * max_gap_margin)

    l8_single_seed = len(level_terminals_logp["8:1"]) == 1
    top_gap_logp = abs(l_refs["8:1"] - l_refs["4:1"])
    top_gap_margin = abs(m_refs["8:1"] - m_refs["4:1"])
    top_degenerate = top_gap_logp <= tol_logp and top_gap_margin <= tol_margin

    return {
        "l_refs": l_refs,
        "m_refs": m_refs,
        "tol_logp": float(tol_logp),
        "tol_margin": float(tol_margin),
        "grace_logp": float(tol_logp / 2.0),
        "grace_margin": float(tol_margin / 2.0),
        "tol_logp_widened_beyond_floor": bool(tol_logp > LOGP_TOL_FLOOR_NATS),
        "seed_gaps_logp": gaps_logp,
        "seed_gaps_margin": gaps_margin,
        "l8_single_seed": l8_single_seed,
        # §C: single-seed L̂(8:1) inherits the largest observed 2-seed gap as
        # its reported uncertainty term.
        "l8_uncertainty_nats": float(max_gap_logp) if l8_single_seed else 0.0,
        "top_degenerate": bool(top_degenerate),
        "top_gap_logp": float(top_gap_logp),
        "top_gap_margin": float(top_gap_margin),
        "sources": sources or {},
        "provenance": (
            "in-task references (plan v3 §C); parent numerics live only in the cross-rig block"
        ),
    }


def matched_pair_discriminator(
    arm_terminals: dict[str, list[float]],
    matched_series_by_seed: dict[int, tuple[list[int], list[float]]],
    coupling_quarter_tol: float,
    space: str,
) -> dict:
    """Arm-internal step-32 discriminator + §C decidability guard.

    Plan v3 §C restatement (no external references): coupling ⇒ the
    schedule-matched step-32 read ≤ quarter-arm terminal + 2.5; horizon ⇒
    step-32 read ≥ 80% of the schedule-matched arm's OWN terminal Δ.
    Decidable iff (0.8 × terminal_SM) − (quarter terminal + 2.5) ≥ 3 in the
    evaluation space; an undecidable read returns verdict "underpowered"
    (the caller may re-evaluate in margin space before settling there).
    """
    quarter_terminal_mean = float(np.mean(arm_terminals["quarter"]))
    matched_terminal_mean = float(np.mean(arm_terminals["matched"]))
    coupling_max = quarter_terminal_mean + float(coupling_quarter_tol)
    horizon_min = HORIZON_STEP32_FRAC_OF_OWN_TERMINAL * matched_terminal_mean
    decidability_gap = horizon_min - coupling_max
    decidable = decidability_gap >= MATCHED_PAIR_DECIDABILITY_MIN
    v32s = [
        v
        for v in (
            _read_at_step(steps, vals, MATCHED_PAIR_STEP)
            for steps, vals in matched_series_by_seed.values()
        )
        if v is not None
    ]
    v32_mean = float(np.mean(v32s)) if v32s else None
    if not decidable:
        verdict = "underpowered"
    elif v32_mean is None:
        verdict = "unreadable"
    elif v32_mean >= horizon_min:
        verdict = "horizon"
    elif v32_mean <= coupling_max:
        verdict = "coupling"
    else:
        verdict = "between"
    return {
        "step": MATCHED_PAIR_STEP,
        "space": space,
        "v32_mean": v32_mean,
        "coupling_max": coupling_max,
        "horizon_min": horizon_min,
        "decidability_gap": float(decidability_gap),
        "decidability_min": MATCHED_PAIR_DECIDABILITY_MIN,
        "decidable": bool(decidable),
        "verdict": verdict,
        "rule": (
            "arm-internal (plan v3 §C): coupling ⇒ step-32 ≤ quarter terminal + 2.5; "
            "horizon ⇒ step-32 ≥ 80% of the schedule-matched arm's own terminal"
        ),
    }


def _phase1_verdicts_and_call(
    *,
    ratio_ok: bool,
    horizon_ok: bool,
    coupling_ok: bool,
    clamp_present: bool | None,
    top_degenerate: bool,
    matched_pair: str,
) -> tuple[dict, str, str, int]:
    """Verdicts + final-call routing for :func:`classify_phase1` (both forks).

    Non-degenerate fork: plan §6 exactly-one precedence over the three
    hypothesis cells, with the §4-item-4 clamp gating (the level rules
    establish PHENOMENOLOGY only; the equilibrium MECHANISM call additionally
    requires the Phase-0b clamp — ``None`` never upgrades).

    Degenerate-top fork (plan v3 §C guard — a pre-registered branch, not a
    no-call): the upper-branch LEVEL tests are unresolvable-as-registered
    (horizon/coupling verdicts = ``None``); the call routes to exactly-one of
    {equilibrium co-landing, matched-pair timing verdict}.
    """
    if not top_degenerate:
        verdicts: dict = {
            "ratio_set_point_consistent": ratio_ok,
            "h_equilibrium_supported": ratio_ok and clamp_present is True,
            "horizon": horizon_ok,
            "coupling": coupling_ok,
        }
        hypothesis_cells = ("ratio_set_point_consistent", "horizon", "coupling")
        satisfied = [k for k in hypothesis_cells if verdicts[k]]
        call_rule = (
            "equilibrium requires ratio_set_point_consistent AND Phase-0b clamp_present "
            "(plan §4 item 4); phenomenology without clamp -> "
            "ratio-setpoint-mechanism-unresolved"
        )
    else:
        verdicts = {
            "ratio_set_point_consistent": ratio_ok,
            "h_equilibrium_supported": ratio_ok and clamp_present is True,
            "horizon": None,
            "coupling": None,
        }
        satisfied = []
        if ratio_ok:
            satisfied.append("ratio_set_point_consistent")
        if matched_pair in ("horizon", "coupling"):
            satisfied.append(matched_pair)
        call_rule = (
            "degenerate-top routing (plan v3 §C): upper-branch level tests "
            "unresolvable-as-registered; call = exactly-one of {equilibrium co-landing, "
            "matched-pair timing verdict}; clamp gating unchanged"
        )
    if len(satisfied) != 1:
        call = "no-call"
    elif satisfied[0] == "ratio_set_point_consistent":
        # Clamp gating: without the clamp the registered outcome is "ratio
        # sets the level; mechanism unresolved — feedback disfavored" (a
        # determinate headline, NOT an equilibrium call).
        call = (
            "equilibrium"
            if verdicts["h_equilibrium_supported"]
            else "ratio-setpoint-mechanism-unresolved"
        )
    else:
        call = satisfied[0]
    return verdicts, call, call_rule, len(satisfied)


def classify_phase1(
    *,
    arm_terminals: dict[str, list[float]],
    matched_series_by_seed: dict[int, tuple[list[int], list[float]]],
    space: str,
    refs: dict,
    clamp_present: bool | None = None,
    margin_fallback: dict | None = None,
) -> dict:
    """The §6 three-hypothesis classification + precedence + matched-pair read.

    Args:
        arm_terminals: per-arm terminal source levels, one value per seed, in
            the PRIMARY space. Keys: "quarter", "anchor", "double", "matched".
        matched_series_by_seed: the schedule-matched arm's (steps, values)
            series per seed (dense ladder + frac reads), primary space.
        space: "logp" or "margin" (the Phase-0a-selected primary space).
        refs: the :func:`derive_in_task_references` output (plan v3 §C) —
            REQUIRED; there is no parent-constant fallback. Carries both
            spaces' references + tolerances + the degenerate-top flag.
        clamp_present: the Phase-0b trained-negative clamp read (plan §4 item
            4 pinned rule). The level rules alone establish only the
            PHENOMENOLOGY (``ratio_set_point_consistent``); the MECHANISM
            verdict ``h_equilibrium_supported`` additionally requires the
            clamp. ``None`` (clamp unread) never upgrades to mechanism.
        margin_fallback: optional ``{"arm_terminals", "matched_series_by_seed"}``
            in MARGIN space — consumed only when ``space == "logp"`` and the
            §C decidability guard finds the logP discriminator compressed
            (re-evaluate in margin space; if still undecidable, the timing
            call ships as underpowered).

    Returns a dict with per-hypothesis verdicts (carrying BOTH
    ``ratio_set_point_consistent`` and ``h_equilibrium_supported``), the
    arm-internal matched-pair discriminator (+ decidability), and the final
    call ("equilibrium" | "ratio-setpoint-mechanism-unresolved" | "horizon" |
    "coupling" | "no-call"). Under the §C degenerate-top guard
    (``refs["top_degenerate"]``) the horizon/coupling LEVEL verdicts are
    ``None`` (unresolvable-as-registered) and the call routes to the
    matched-pair discriminator + the equilibrium co-landing test, with the
    top-compression reported as a finding.
    """
    if space not in ("logp", "margin"):
        raise ValueError(f"space={space!r}")
    for key in ("l_refs", "m_refs", "tol_logp", "tol_margin", "top_degenerate"):
        if not isinstance(refs, dict) or key not in refs:
            raise ValueError(
                "classify_phase1 requires the derive_in_task_references refs dict "
                f"(missing {key!r}) — plan v3 §C retired the parent-constant fallback."
            )
    l_refs: dict[str, float] = refs["l_refs"]
    m_refs: dict[str, float] = refs["m_refs"]
    tol_logp = float(refs["tol_logp"])
    tol_margin = float(refs["tol_margin"])
    if space == "margin":
        level_refs = dict(m_refs)
        tol = tol_margin
        # logP-form thresholds re-expressed through the in-task (L̂, M̂) anchors.
        quarter_max = reexpress_threshold(
            l_refs["4:1"] - HORIZON_QUARTER_BELOW_RATIO_NATS, m_refs, l_refs
        )
        coupling_quarter_tol = tol_margin * (COUPLING_QUARTER_TOL / tol_logp)
        spread_max = FIXED_RATIO_SPREAD_MAX_NATS * (tol_margin / tol_logp)
    else:
        level_refs = dict(l_refs)
        tol = tol_logp
        quarter_max = l_refs["4:1"] - HORIZON_QUARTER_BELOW_RATIO_NATS
        coupling_quarter_tol = COUPLING_QUARTER_TOL
        spread_max = FIXED_RATIO_SPREAD_MAX_NATS
    # §C upper-branch threshold: midpoint(L̂(4:1), L̂(8:1)) in the evaluation
    # space (replaces the retired parent horizon-upper 17).
    upper_thr = (level_refs["4:1"] + level_refs["8:1"]) / 2.0
    accrual_min_slope = COUPLING_ACCRUAL_MIN_SLOPE  # slope units differ per space; as-is.

    means = {arm: float(np.mean(v)) for arm, v in arm_terminals.items()}
    fixed_ratio_arms = [a for a in ("quarter", "anchor", "double") if a in means]
    fixed_spread = max(means[a] for a in fixed_ratio_arms) - min(means[a] for a in fixed_ratio_arms)

    # ── Ratio set-point (phenomenology). ─────────────────────────────────────
    ratio_ok = (
        all(seed_band_ok(arm_terminals[a], level_refs["4:1"], tol) for a in fixed_ratio_arms)
        and seed_band_ok(arm_terminals["matched"], level_refs["4:1"], tol)
        and fixed_spread <= spread_max
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
    quarter_at_2to1 = seed_band_ok(
        arm_terminals["quarter"], level_refs["2:1"], coupling_quarter_tol
    )
    double_at_8to1 = seed_band_ok(arm_terminals["double"], level_refs["8:1"], tol)
    matched_at_8to1 = seed_band_ok(arm_terminals["matched"], level_refs["8:1"], tol)
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

    # ── Arm-internal matched-pair discriminator + §C decidability guard. ─────
    mp_primary = matched_pair_discriminator(
        arm_terminals, matched_series_by_seed, coupling_quarter_tol, space
    )
    mp = mp_primary
    mp_margin_fallback = None
    if space == "logp" and not mp_primary["decidable"] and margin_fallback is not None:
        # §C: when logP compresses below the 3-nat separation, re-evaluate the
        # discriminator in margin space before settling on underpowered.
        mp_margin_fallback = matched_pair_discriminator(
            margin_fallback["arm_terminals"],
            margin_fallback["matched_series_by_seed"],
            tol_margin * (COUPLING_QUARTER_TOL / tol_logp),
            "margin",
        )
        if mp_margin_fallback["decidable"]:
            mp = mp_margin_fallback
    matched_pair = mp["verdict"]
    timing_underpowered = not mp["decidable"]
    matched_pair_block = {
        **mp,
        "primary_space_attempt": {k: mp_primary[k] for k in ("space", "decidable", "verdict")},
        "margin_fallback_attempt": (
            {k: mp_margin_fallback[k] for k in ("space", "decidable", "verdict")}
            if mp_margin_fallback is not None
            else None
        ),
        "timing_underpowered": timing_underpowered,
        "underpowered_note": (
            "step-32 timing call underpowered in every available space — only the "
            "descriptive accrual-shape read (coupling_detail.accrual_by_seed) ships (plan v3 §C)"
            if timing_underpowered
            else None
        ),
    }

    top_degenerate = bool(refs["top_degenerate"])
    verdicts, call, call_rule, n_satisfied = _phase1_verdicts_and_call(
        ratio_ok=bool(ratio_ok),
        horizon_ok=bool(horizon_ok),
        coupling_ok=bool(coupling_ok),
        clamp_present=clamp_present,
        top_degenerate=top_degenerate,
        matched_pair=matched_pair,
    )

    return {
        "space": space,
        "refs": level_refs,
        "refs_in_task": {
            k: refs[k]
            for k in (
                "l_refs",
                "m_refs",
                "tol_logp",
                "tol_margin",
                "l8_single_seed",
                "l8_uncertainty_nats",
                "sources",
            )
            if k in refs
        },
        "tolerance": tol,
        "grace": tol / 2.0,
        "upper_midpoint": upper_thr,
        "quarter_horizon_max": quarter_max,
        "fixed_ratio_spread_max": spread_max,
        "upper_branch_resolvable": not top_degenerate,
        "top_compression": (
            {
                "top_degenerate": True,
                "gap_logp": refs.get("top_gap_logp"),
                "gap_margin": refs.get("top_gap_margin"),
                "tol_logp": tol_logp,
                "tol_margin": tol_margin,
                "note": (
                    "fresh L̂(8:1) within tol of L̂(4:1) in BOTH spaces — the parent "
                    "dose-response top compresses under this rig (reported finding)"
                ),
            }
            if top_degenerate
            else None
        ),
        "arm_terminal_means": means,
        "fixed_ratio_spread": float(fixed_spread),
        "clamp_present": clamp_present,
        "verdicts": verdicts,
        "call": call,
        "call_rule": call_rule,
        "n_satisfied": n_satisfied,
        "matched_pair_discriminator": matched_pair_block,
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


def derive_band_from_rowtype(rowtype: dict) -> dict:
    """Derive a band-trajectory-equivalent source ΔG series from rowtype_ce.json.

    Round-8 crash fix: T=13 cells (the two unconditional Phase-4 bridge cells +
    phase2/dense_200p0n) never produce ``inloop_band_trajectory.json`` because
    ``MarkerBandStopCallback`` disabled itself when ``max_steps(13) <
    min_steps(20)``. The per-row-type CE probe
    (``RowTypeCETrainProbeCallback``) is NOT min_steps-gated and reads the SAME
    construct on the SAME live training model in the SAME application gauge:

    - band ``delta_nats[t]`` = mean logP_live(marker at slot) − mean
      logP_base(marker at slot) over source-positive probe rows from the
      cell's train JSONL (base = first-eval read with the PEFT adapter
      disabled);
    - rowtype ``pos_marker_ce[t]`` = −mean logP_live(marker at slot) over the
      fixed 16-row source-positive probe sample from the SAME train JSONL,
      and ``pos_marker_ce_base`` = the same adapter-disabled base read.

    So ``delta_nats[t] = pos_marker_ce_base − pos_marker_ce[t]`` is the
    identical live-gauge ΔG construction; the only difference is the probe-row
    sample size (16 rowtype rows vs the band callback's ≤32). The plan §4
    arrest bands (ΔG in nats; ≥6 by step 13 + slope rules) therefore apply
    unchanged.

    Args:
        rowtype: Parsed ``rowtype_ce.json`` payload (schema
            ``i601_rowtype_ce_v1``).

    Returns:
        ``{"steps", "delta_nats", "trajectory_source": "rowtype_ce_derived",
        "n_pos_rows", "pos_marker_ce_base", "gauge_note"}``.

    Raises:
        ValueError: when the payload has no usable per-step positive-marker CE
            series or no base-side constant (the caller decides the fallback).
    """
    steps = rowtype.get("steps") or []
    pos_ce = rowtype.get("pos_marker_ce") or []
    base = rowtype.get("pos_marker_ce_base")
    if not steps or len(pos_ce) != len(steps):
        raise ValueError(
            f"rowtype_ce payload unusable: {len(steps)} steps vs {len(pos_ce)} pos_marker_ce "
            f"entries (schema={rowtype.get('schema')!r})"
        )
    if base is None or any(v is None for v in pos_ce):
        raise ValueError(
            "rowtype_ce payload unusable: pos_marker_ce_base or per-step pos_marker_ce is "
            "null — the positive probe side never functioned for this cell"
        )
    return {
        "steps": [int(s) for s in steps],
        "delta_nats": [float(base) - float(v) for v in pos_ce],
        "trajectory_source": "rowtype_ce_derived",
        "n_pos_rows": rowtype.get("n_pos_rows"),
        "pos_marker_ce_base": float(base),
        "gauge_note": (
            "live-training-model (same gauge + same adapter-disabled base read as the "
            "in-loop band trajectory; 16-row source-positive probe sample)"
        ),
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
