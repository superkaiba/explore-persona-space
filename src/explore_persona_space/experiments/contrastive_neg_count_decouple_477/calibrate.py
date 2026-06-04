# em-dash + Qwen marker token " ※" are intentional
"""Task #477 — calibration layer: pick LR per count level + validity gate.

Pure functions over JSON-array results (no LLM call, no GPU). Drives Phase 2.5
of the dispatcher (pick) and Phase 5 of the analysis (gate).

See plan §4 for the pseudocode this implements.
"""

from __future__ import annotations

from typing import Any

from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
    MATCH_BAND,
    MAX_SOURCE_EMISSION_V3,
    MIN_SOURCE_EMISSION,
    MIN_SOURCE_EMISSION_V3,
    TARGET_SOURCE_DELTA_G,
)

# Calibration-table schema (per cell in Phase 2):
#   calibration_table[count][lr] = {
#       "source_self_delta_g": float,    # DV-B at final checkpoint
#       "source_emission_p": float,      # DV-C at final checkpoint
#       "cell_slug": str,                # provenance
#       "seed": int,                     # always 42 in Phase 2
#       "wandb_run_id": str | None,      # provenance
#   }
# count keys are str(int) on disk (JSON), int in-memory; callers normalize.


def _normalize_count_table(cells: dict) -> dict[float, dict]:
    """Normalize the per-count LR table to {float(lr): {...}}.

    On-disk JSON uses string keys (`"1e-05"` or `"5e-06"`); in-memory callers
    may pass float keys. This is a defensive convergence so pick_lr_for_count
    works with either.
    """
    out: dict[float, dict] = {}
    for k, v in cells.items():
        out[float(k)] = v
    return out


def pick_lr_for_count(calibration_table: dict, count: int) -> dict[str, Any]:
    """For count level c, pick the LR whose terminal cell satisfies BOTH gates.

    Plan §4 pseudocode + §7 kill criterion.

    Args:
        calibration_table: {count -> {lr -> {source_self_delta_g, source_emission_p, ...}}}.
            Count keys may be int or str(int) (caller normalizes).
        count: the count level to pick the LR for. MUST be present as a key in
            calibration_table (str(int) or int both accepted).

    Returns:
        {"lr": float, "achieved_delta_g": float, "achieved_emission_p": float,
         "in_band": True, "all_candidates": dict, "count": int}.

    Raises:
        KeyError: count missing from calibration_table.
        RuntimeError: NO LR satisfies BOTH gates (the §7 kill criterion); the
            error message instructs the caller to expand the LR grid or pivot
            to fixed-step decoupling (option 3 from plan §4).
    """
    # Accept either int or str(int) keys at the top level.
    if count in calibration_table:
        cells_raw = calibration_table[count]
    elif str(count) in calibration_table:
        cells_raw = calibration_table[str(count)]
    else:
        raise KeyError(
            f"count={count} missing from calibration_table; known counts: "
            f"{sorted(calibration_table.keys())}"
        )
    cells = _normalize_count_table(cells_raw)

    qualifying: list[tuple[float, dict]] = []
    for lr, m in cells.items():
        delta = float(m["source_self_delta_g"])
        emit = float(m["source_emission_p"])
        if abs(delta - TARGET_SOURCE_DELTA_G) <= MATCH_BAND and emit >= MIN_SOURCE_EMISSION:
            qualifying.append((lr, m))

    if not qualifying:
        raise RuntimeError(
            f"NO qualifying LR for count={count}: every LR misses the matched "
            f"implant band [{TARGET_SOURCE_DELTA_G - MATCH_BAND:.1f}, "
            f"{TARGET_SOURCE_DELTA_G + MATCH_BAND:.1f}] nats OR the emission floor "
            f"P(※)≥{MIN_SOURCE_EMISSION:.2f}. Calibration table for this count: "
            f"{cells}. Kill criterion (plan §7): expand the LR grid (try {{2e-7, "
            f"1e-7}} or {{1e-4}}) OR pivot to fixed-step decoupling (option 3 from "
            f"plan §4). Dispatcher MUST exit non-zero on this error so the "
            f"orchestrator does not silently run the main sweep at an unpicked LR."
        )

    best_lr, best_m = min(
        qualifying,
        key=lambda lm: abs(float(lm[1]["source_self_delta_g"]) - TARGET_SOURCE_DELTA_G),
    )
    return {
        "lr": float(best_lr),
        "achieved_delta_g": float(best_m["source_self_delta_g"]),
        "achieved_emission_p": float(best_m["source_emission_p"]),
        "in_band": True,
        "all_candidates": cells,
        "count": int(count),
    }


def validity_gate(
    main_cell_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split main cells into (kept, excluded) by source-self ΔG + emission gate.

    Plan §6 validity gate: every cell in the H1 statistic must satisfy
    source-self ΔG ∈ [TARGET_SOURCE_DELTA_G ± MATCH_BAND] nats AND
    source emission P(※) ≥ MIN_SOURCE_EMISSION on its own R.

    Args:
        main_cell_results: list of per-cell result dicts; each must carry
            "source_self_delta_g_at_last_ckpt" and "source_emission_p_at_last_ckpt".
            A cell missing either key raises a KeyError (fail loud — the cell
            was supposed to land but did not write the gate metrics).

    Returns:
        (kept, excluded) — same dicts, partitioned. Excluded cells are NOT
        dropped silently; the dispatcher / analyzer list them separately in §6
        of the clean-result with their actual values (Lens 13 honesty).

    Raises:
        KeyError: a cell result is missing the required gate metrics.
    """
    kept: list[dict] = []
    excluded: list[dict] = []
    for c in main_cell_results:
        if "source_self_delta_g_at_last_ckpt" not in c:
            raise KeyError(
                f"Cell {c.get('cell', '<unknown>')} missing "
                f"'source_self_delta_g_at_last_ckpt' — gate metrics not written. "
                f"Investigate the eval_trajectory output for this cell before "
                f"running the validity gate."
            )
        if "source_emission_p_at_last_ckpt" not in c:
            raise KeyError(
                f"Cell {c.get('cell', '<unknown>')} missing "
                f"'source_emission_p_at_last_ckpt' — eval_trajectory did not "
                f"surface DV-C; bump the eval rig or compute it from raw R "
                f"argmax before running the gate."
            )
        delta = float(c["source_self_delta_g_at_last_ckpt"])
        emit = float(c["source_emission_p_at_last_ckpt"])
        if abs(delta - TARGET_SOURCE_DELTA_G) <= MATCH_BAND and emit >= MIN_SOURCE_EMISSION:
            kept.append(c)
        else:
            excluded.append(c)
    return kept, excluded


# ── v3/v4 step-lever pick (plan v4 §4 Phase 2.5 + §6 + §7 H4 kill-gate). ─────


def pick_step_for_count(
    step_table: dict,
    count: int,
) -> dict[str, Any]:
    """For count level c, pick the early-step checkpoint matching the v4 band.

    Mirrors :func:`pick_lr_for_count` but the axis is the per-cell early-step
    checkpoint cadence (plan v4 §4 Phase 2.5). For each (count, step) in the
    Phase 2 step-calibration table, require ALL of:

      * ``source_self_delta_g`` ∈ [TARGET_SOURCE_DELTA_G ± MATCH_BAND] nats
        (=[10.5, 13.5] at the v4 defaults), AND
      * ``source_emission_p`` ∈ [MIN_SOURCE_EMISSION_V3, MAX_SOURCE_EMISSION_V3]
        (=[0.40, 0.95] — anti-collapse floor + anti-saturation ceiling), AND
      * ``source_R_collapsed == False``.

    Pick the step CLOSEST to TARGET_SOURCE_DELTA_G (=12.0). If none qualify,
    raise ``RuntimeError`` (the H4 kill-gate; dispatcher exits non-zero and the
    orchestrator banks Path B).

    Args:
        step_table: ``{count -> {step -> {source_self_delta_g, source_emission_p,
            source_R_collapsed, ...}}}``. Count + step keys may be int or
            str(int) (the picker normalizes both).
        count: count level to resolve.

    Returns:
        ``{"step": int, "achieved_delta_g": float, "achieved_emission_p": float,
        "in_band": True, "all_candidates": dict, "count": int}``.

    Raises:
        KeyError: count missing from step_table.
        RuntimeError: zero qualifying steps for this count — the H4 kill-gate.
            The message names the band + cite + tells the dispatcher to bank
            Path B (the methodological "implant rises within one optimizer step"
            finding).
    """
    if count in step_table:
        cells_raw = step_table[count]
    elif str(count) in step_table:
        cells_raw = step_table[str(count)]
    else:
        raise KeyError(
            f"count={count} missing from step_table; known counts: {sorted(step_table.keys())}"
        )
    # Normalize step keys to int (JSON deserializes them as str).
    cells: dict[int, dict] = {}
    for k, v in cells_raw.items():
        cells[int(k)] = v

    qualifying: list[tuple[int, dict]] = []
    for step, m in cells.items():
        delta = float(m["source_self_delta_g"])
        emit = float(m["source_emission_p"])
        collapsed = bool(m.get("source_R_collapsed", False))
        if (
            abs(delta - TARGET_SOURCE_DELTA_G) <= MATCH_BAND
            and MIN_SOURCE_EMISSION_V3 <= emit <= MAX_SOURCE_EMISSION_V3
            and not collapsed
        ):
            qualifying.append((step, m))

    if not qualifying:
        raise RuntimeError(
            f"NO qualifying step for count={count} (H4 kill-gate, plan v4 §7): "
            f"every step misses the v4 matched band ΔG ∈ ["
            f"{TARGET_SOURCE_DELTA_G - MATCH_BAND:.1f}, "
            f"{TARGET_SOURCE_DELTA_G + MATCH_BAND:.1f}] nats AND emission "
            f"P(※) ∈ [{MIN_SOURCE_EMISSION_V3:.2f}, "
            f"{MAX_SOURCE_EMISSION_V3:.2f}] AND not source_R_collapsed. "
            f"Step table for this count: {cells}. Bank Path B: 'marker "
            f"implants within one optimizer step at this recipe scale, "
            f"training-amount cannot decouple count from implant'. "
            f"Dispatcher MUST exit non-zero so the orchestrator does not "
            f"silently run the main sweep at an unpicked step."
        )

    best_step, best_m = min(
        qualifying,
        key=lambda sm: abs(float(sm[1]["source_self_delta_g"]) - TARGET_SOURCE_DELTA_G),
    )
    return {
        "step": int(best_step),
        "achieved_delta_g": float(best_m["source_self_delta_g"]),
        "achieved_emission_p": float(best_m["source_emission_p"]),
        "in_band": True,
        "all_candidates": cells,
        "count": int(count),
    }


# ── v6 rank-pick + slot-fix diagnostic (plan v6 §4 Phase 2A.5 + §6.5.5 H4). ──


def pick_rank(cal_a_table: dict) -> dict[str, Any]:
    """Pick the Cal-A rank that lands ≥3 of 4 counts in the matched mid-band.

    Plan v6 §4 Phase 2A.5 + §4.5 pseudocode. M3 simplification: no Cal-B path
    — positives is GLOBAL at 200; the rank pick comes from {2, 4, 8} only.
    The Cal-A0 r=32 control is a diagnostic (H4), NOT part of this pick.

    Args:
        cal_a_table: ``{rank -> {count -> {step -> {delta_g, emit, collapsed,
            ...}}}}``. Rank + count + step keys may be int or str(int); the
            picker normalizes both. Per-step records must carry ``delta_g``,
            ``emit``, ``collapsed``.

    Returns:
        ``{"picked_rank": int, "picked_positives": 200, "picked_alpha": int,
        "qualifying_counts": list[int], "per_count_picked_step": dict[int, int],
        "off_ramp_fired": False, "reason": str}``.

    Raises:
        RuntimeError: H0 kill-gate — no rank in RANK_GRID_V5 lands ≥3 counts
            in-band; dispatcher MUST exit non-zero so the orchestrator does
            not silently run the main sweep at an unpicked rank.
    """
    # Inline imports survive ruff's unused-import strip (the module-top
    # convenience imports are exposed via the package __init__; this layer
    # keeps the symbol references inside the function bodies).
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        COUNT_LEVELS,
        RANK_ALPHA_MAP_V5,
        RANK_GRID_V5,
    )

    # In-band per (rank, count): the step closest to TARGET_SOURCE_DELTA_G.
    in_band: dict[int, dict[int, int]] = {}
    for rank_raw, by_count_raw in cal_a_table.items():
        rank = int(rank_raw)
        by_count: dict[int, dict[int, dict]] = {}
        for count_raw, by_step_raw in by_count_raw.items():
            by_count[int(count_raw)] = {int(step): m for step, m in by_step_raw.items()}
        in_band[rank] = {}
        for count, by_step in by_count.items():
            best: tuple[int, dict] | None = None
            for step in sorted(by_step):
                m = by_step[step]
                delta = float(m["delta_g"])
                emit = float(m["emit"])
                collapsed = bool(m.get("collapsed", False))
                in_band_step = (
                    abs(delta - TARGET_SOURCE_DELTA_G) <= MATCH_BAND
                    and MIN_SOURCE_EMISSION_V3 <= emit <= MAX_SOURCE_EMISSION_V3
                    and not collapsed
                )
                # On multiple in-band steps, pick the one closest to TARGET_SOURCE_DELTA_G.
                closer_than_current = best is None or abs(delta - TARGET_SOURCE_DELTA_G) < abs(
                    float(best[1]["delta_g"]) - TARGET_SOURCE_DELTA_G
                )
                if in_band_step and closer_than_current:
                    best = (step, m)
            if best is not None:
                in_band[rank][count] = best[0]

    # Sort ranks: prefer most-in-band; on tie, prefer SMALLER rank
    # (smaller rank = slower implant ramp = more calibration headroom).
    ranked = sorted(in_band.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if not ranked:
        raise RuntimeError(
            "Cal-A table is empty — no rank cells. Investigate the "
            "step_calibration / rank_calibration phase before picking."
        )
    best_rank, best_picks = ranked[0]
    n_in_band = len(best_picks)

    if n_in_band >= 3:
        # M2 invariant: alpha is read from the SSOT helper (raises if best_rank
        # somehow isn't in RANK_GRID_V5 — should never happen).
        if best_rank not in RANK_GRID_V5:
            raise AssertionError(
                f"picked_rank={best_rank} not in RANK_GRID_V5={RANK_GRID_V5} — "
                f"v6 M2 invariant violated; the Cal-A grid is the only legal "
                f"source for picked_rank."
            )
        alpha = RANK_ALPHA_MAP_V5[best_rank]
        return {
            "picked_rank": int(best_rank),
            "picked_positives": 200,
            "picked_alpha": int(alpha),
            "qualifying_counts": sorted(best_picks.keys()),
            "per_count_picked_step": {int(k): int(v) for k, v in best_picks.items()},
            "off_ramp_fired": False,
            "reason": (
                f"rank={best_rank} (alpha={alpha}) at positives=200 lands "
                f"{n_in_band}/{len(COUNT_LEVELS)} counts in-band "
                f"(band ΔG ∈ [{TARGET_SOURCE_DELTA_G - MATCH_BAND:.1f}, "
                f"{TARGET_SOURCE_DELTA_G + MATCH_BAND:.1f}], emit ∈ "
                f"[{MIN_SOURCE_EMISSION_V3:.2f}, {MAX_SOURCE_EMISSION_V3:.2f}], "
                f"not collapsed)."
            ),
        }

    # H0 off-ramp: no rank in RANK_GRID_V5 reached ≥3 counts in-band.
    summary = {
        int(rank): {int(c): int(s) for c, s in picks.items()} for rank, picks in in_band.items()
    }
    raise RuntimeError(
        "H0 OFF-RAMP (plan v6 §7): no rank in "
        f"{sorted(RANK_GRID_V5)} lands ≥3 of {len(COUNT_LEVELS)} counts "
        f"in-band at positives=200 GLOBAL (v6 M3 — no Cal-B). Per-rank "
        f"in-band picks: {summary}. Cumulative finding: marker implant at "
        "this recipe scale is bimodal across the count axis even at the "
        "lowest LoRA rank tested; training-amount + recipe-scale CANNOT "
        "decouple count from implant on this rig. Bank Path B (v2+v4+v6). "
        "The Cal-A0 H4 verdict (slot-bug-vs-capacity) is part of the "
        "banked finding regardless of this gate. Dispatcher MUST exit "
        "non-zero so the orchestrator does not silently run the main sweep "
        "at an unpicked rank."
    )


def slot_fix_diagnostic(cal_a0_table: dict) -> dict[str, Any]:
    """v6 H4: did the slot-fix port unbug the v4 ΔG≈0 result at r=32?

    Plan v6 §6.5.5 + §4.5 pseudocode.

    Args:
        cal_a0_table: ``{32: {count -> {step -> {delta_g, ...}}}}`` from
            Phase 2A-CONTROL.

    Returns:
        ``{"verdict": "slot-bug-confirmed-v4-result-was-genuine"
                       | "slot-bug-rejected-v4-result-was-slot-artifact"
                       | "ambiguous",
            "max_terminal_delta_g": float,
            "per_count_max_delta_g": dict[int, float],
            "alpha_used": int}``.

    Raises:
        KeyError: ``RANK_CONTROL_V6`` missing from ``cal_a0_table``.
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        ALPHA_CONTROL_V6,
        RANK_CONTROL_V6,
    )

    # Normalize the outer rank key (str | int).
    if RANK_CONTROL_V6 in cal_a0_table:
        by_count_raw = cal_a0_table[RANK_CONTROL_V6]
    elif str(RANK_CONTROL_V6) in cal_a0_table:
        by_count_raw = cal_a0_table[str(RANK_CONTROL_V6)]
    else:
        raise KeyError(
            f"Cal-A0 table missing rank={RANK_CONTROL_V6}; known keys: "
            f"{sorted(cal_a0_table.keys())}"
        )

    per_count: dict[int, float] = {}
    for count_raw, by_step in by_count_raw.items():
        per_count[int(count_raw)] = max(float(m["delta_g"]) for m in by_step.values())

    max_dg = max(per_count.values()) if per_count else 0.0
    if max_dg < 1.0:
        # slot-fix didn't unstick r=32 at lr=2e-6 — v4 ΔG≈0 was a real
        # capacity finding under the lr floor.
        verdict = "slot-bug-confirmed-v4-result-was-genuine"
    elif sum(1 for v in per_count.values() if v >= 5.0) >= 2:
        # ≥2 counts now produce non-trivial ΔG at r=32; v4's ΔG≈0 was the
        # slot-bug, NOT a capacity finding.
        verdict = "slot-bug-rejected-v4-result-was-slot-artifact"
    else:
        verdict = "ambiguous"

    return {
        "verdict": verdict,
        "max_terminal_delta_g": float(max_dg),
        "per_count_max_delta_g": per_count,
        "alpha_used": int(ALPHA_CONTROL_V6),
    }
