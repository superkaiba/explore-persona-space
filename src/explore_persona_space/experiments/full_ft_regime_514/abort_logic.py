# em-dash + nat character are intentional
"""Per-lever abort-on-collapse decision logic for #514 (plan §4.1.3).

Pure function so it can be unit-tested independently of the dispatcher and the
eval JSON I/O. The decision rule: after cell 1 of each lever (the smallest
budget in that lever) finishes Phase 2 eval, abort the remaining cells of that
lever iff BOTH conditions hold simultaneously:

    source_r_collapse_rate >= ABORT_SOURCE_RCOLLAPSE_THRESHOLD (0.50)
    AND
    held_out_g_logprob_mean > ABORT_HELD_OUT_GLOGPROB_MAX (-5.0)

The "AND" is load-bearing: a cell may be collapsed but sub-ceiling (still
informative — e.g. early cliff with non-saturated held-out), or saturated
but not collapsed (then continuing the lever lets a smaller-budget cell
breathe). Only the joint "saturated AND collapsed" combination implies the
lever is past the cliff in both axes; deeper budgets only make things worse.
"""

from __future__ import annotations

from explore_persona_space.experiments.full_ft_regime_514 import (
    ABORT_HELD_OUT_GLOGPROB_MAX,
    ABORT_SOURCE_RCOLLAPSE_THRESHOLD,
)


def compute_source_r_collapse_rate(eval_json: dict) -> float:
    """Count ``r_collapsed=True`` over source probes / total source probes.

    The #514 cell's eval JSON has ``delta_g_source[source_persona][q]["r_collapsed"]``
    per probe (see ``lora_vs_ft_508.eval_one_cell``). This function aggregates
    those flags into a single rate in [0, 1]. Returns ``NaN`` when no source
    probes exist (the cell has ``source_persona=None`` / ``delta_g_source={}``).
    """
    dg_source = eval_json.get("delta_g_source") or {}
    if not dg_source:
        return float("nan")
    total = 0
    collapsed = 0
    for _persona, q_map in dg_source.items():
        for _q, probe in q_map.items():
            total += 1
            if probe.get("r_collapsed"):
                collapsed += 1
    if total == 0:
        return float("nan")
    return collapsed / total


def get_held_out_g_logprob_mean(eval_json: dict) -> float:
    """Read the canonical sub-ceiling diagnostic from a cell eval JSON.

    Path: ``aggregates.held_out_g_logprob_mean`` (#508 eval_one_cell R2.4
    round-2 fix). Returns ``NaN`` if missing (the caller treats NaN as
    "cannot decide; do NOT abort" — conservative).
    """
    agg = eval_json.get("aggregates") or {}
    val = agg.get("held_out_g_logprob_mean")
    if val is None:
        return float("nan")
    return float(val)


def should_abort_lever(
    eval_json: dict,
    *,
    rcollapse_threshold: float = ABORT_SOURCE_RCOLLAPSE_THRESHOLD,
    g_logprob_max: float = ABORT_HELD_OUT_GLOGPROB_MAX,
) -> tuple[bool, dict]:
    """Decide whether to abort the rest of the cell's lever.

    Returns ``(abort, diagnostics)``. ``diagnostics`` is a small dict suitable
    for embedding in the ``epm:514-lever-aborted`` sentinel marker note.

    Decision rule (plan §4.1.3): abort iff
        source_r_collapse_rate >= rcollapse_threshold
        AND held_out_g_logprob_mean > g_logprob_max

    NaN propagation: if EITHER input is NaN, do NOT abort (conservative —
    we'd rather burn extra GPU on a non-decision than abort an unfinished
    cell). The diagnostics dict still reports the values so the user can
    investigate.
    """
    rcoll = compute_source_r_collapse_rate(eval_json)
    g_logp = get_held_out_g_logprob_mean(eval_json)

    import math

    if math.isnan(rcoll) or math.isnan(g_logp):
        abort = False
        reason = "NaN-input (incomplete eval) — conservative no-abort"
    else:
        cond_rcoll = rcoll >= rcollapse_threshold
        cond_glogp = g_logp > g_logprob_max
        abort = cond_rcoll and cond_glogp
        if abort:
            reason = (
                f"r_collapse_rate={rcoll:.3f} >= {rcollapse_threshold} "
                f"AND held_out_g_logprob_mean={g_logp:.3f} > {g_logprob_max}"
            )
        else:
            reason = (
                f"continuing — r_collapse_rate={rcoll:.3f}, "
                f"held_out_g_logprob_mean={g_logp:.3f} "
                f"(thresholds: rcoll>={rcollapse_threshold}, glogp>{g_logprob_max})"
            )

    diagnostics = {
        "source_r_collapse_rate": rcoll,
        "held_out_g_logprob_mean": g_logp,
        "rcollapse_threshold": rcollapse_threshold,
        "g_logprob_max": g_logprob_max,
        "abort": abort,
        "reason": reason,
    }
    return abort, diagnostics
