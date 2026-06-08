# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + nat char + minus sign intentional
"""Task #514 — analyze module (delegates to #508's analyze + writes #514 artifacts).

Plan §6.3 + §10:
    - Reuses #508's ``run_analysis`` for the heavy lift (per-cell aggregates,
      bracketing check, matched-rate gap, hero figure).
    - Merges the 6 new #514 FT cells with the 5 reference cells from #508
      (LoRA b1/b2/b3 + FT b1/b2 anchors). The #508 cell #ft_b3 (collapsed at
      1.0 epoch) is included only as a half-transparent anchor on the hero.
    - Writes the 3 #514-specific artifacts (plan §10):
        eval_results/issue_514/_matched_rate_514.json
        eval_results/issue_514/_bracketing_report.json
        eval_results/issue_514/_bootstrap_per_cell.json

The #508 reference eval JSONs are expected to be present at
``eval_results/issue_508/{lora_b1,lora_b2,lora_b3,ft_b1,ft_b2,ft_b3}_seed42.json``
(plan §4.1 — re-used artifacts, not retrained). If the directory is missing,
analyze runs on the #514-only cells and writes a caveat into the bracketing
report.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.full_ft_regime_514 import (
    ABORT_HELD_OUT_GLOGPROB_MAX,
    ABORT_SOURCE_RCOLLAPSE_THRESHOLD,
    CLEAN_CELL_BRACKETING_UPPER_NATS,
    DENSE_LEVER_CELLS,
    LOW_LR_LEVER_CELLS,
)
from explore_persona_space.experiments.full_ft_regime_514.abort_logic import (
    compute_source_r_collapse_rate,
    get_held_out_g_logprob_mean,
)

log = logging.getLogger("issue_514.analyze")

# Plan §6.4 determinacy gate: |local linear interpolation − cluster-bootstrap
# linear interpolation| ≤ 0.5 nat at source ΔG = 8 nat → determinate.
DETERMINACY_GATE_THRESHOLD_NAT = 0.5

# Plan §6.4 matched-rate target.
MATCHED_RATE_TARGET_NAT = 8.0

# Path on disk where #508's reference eval JSONs live (post-merge to main).
# Tries the repo-relative location first; fall back to the worktree-relative
# location if the analyze runs on the pod before the merge.
REFERENCE_EVAL_DIR_CANDIDATES: tuple[str, ...] = (
    "eval_results/issue_508",
    "../issue-508/eval_results/issue_508",
)


def _find_reference_eval_dir() -> Path | None:
    for cand in REFERENCE_EVAL_DIR_CANDIDATES:
        p = Path(cand)
        if p.exists():
            return p
    return None


def _gather_reference_eval_jsons() -> list[Path]:
    """Locate the 5+1 #508 reference eval JSONs (LoRA b1/b2/b3 + FT b1/b2/b3).

    Returns the list of paths that exist. The dispatcher tolerates a missing
    reference dir — analysis runs on #514-only cells with a caveat.
    """
    ref_dir = _find_reference_eval_dir()
    if ref_dir is None:
        log.warning(
            "[analyze] #508 reference eval dir not found under %s — "
            "analyze will run on #514-only cells (no merged hero figure).",
            REFERENCE_EVAL_DIR_CANDIDATES,
        )
        return []
    expected = (
        "lora_b1_seed42.json",
        "lora_b2_seed42.json",
        "lora_b3_seed42.json",
        "ft_b1_seed42.json",
        "ft_b2_seed42.json",
        "ft_b3_seed42.json",
    )
    found: list[Path] = []
    for name in expected:
        p = ref_dir / name
        if p.exists():
            found.append(p)
        else:
            log.warning("[analyze] reference cell missing: %s (continuing)", p)
    return found


def _classify_lever(cell_slug: str) -> str:
    """Return ``"dense"`` / ``"lowlr"`` / ``"508_anchor"`` for a cell slug."""
    if cell_slug in DENSE_LEVER_CELLS:
        return "dense"
    if cell_slug in LOW_LR_LEVER_CELLS:
        return "lowlr"
    return "508_anchor"


def _per_cell_diagnostics(eval_jsons: list[Path]) -> list[dict]:
    """For each #514 cell eval JSON, compute (source_mean, r_collapse_rate,
    held_out_g_logprob_mean, lever, is_clean_above_9_nat).
    """
    out: list[dict] = []
    for p in eval_jsons:
        ej = json.loads(p.read_text())
        cell_slug = ej.get("cell_slug", p.stem.split("_seed")[0])
        agg = ej.get("aggregates", {}) or {}
        source_mean = agg.get("source_self_mean_delta_g")
        held_out_mean = agg.get("held_out_mean_delta_g")
        rcoll = compute_source_r_collapse_rate(ej)
        g_logp = get_held_out_g_logprob_mean(ej)
        lever = _classify_lever(cell_slug)

        # Clean-cell-above-9-nat criterion (plan §6.2). Floats may be NaN
        # if the cell had no source probes; cast safely.
        clean_above_9 = False
        if (
            source_mean is not None
            and isinstance(source_mean, int | float)
            and not _is_nan(float(source_mean))
            and float(source_mean) > CLEAN_CELL_BRACKETING_UPPER_NATS
            and not _is_nan(rcoll)
            and rcoll < ABORT_SOURCE_RCOLLAPSE_THRESHOLD
            and not _is_nan(g_logp)
            and g_logp <= ABORT_HELD_OUT_GLOGPROB_MAX
        ):
            clean_above_9 = True

        out.append(
            {
                "cell": cell_slug,
                "lever": lever,
                "eval_json_path": str(p),
                "source_mean": float(source_mean) if source_mean is not None else None,
                "held_out_mean": float(held_out_mean) if held_out_mean is not None else None,
                "r_collapse_rate": rcoll,
                "held_out_g_logprob_mean": g_logp,
                "clean_above_9_nat": clean_above_9,
            }
        )
    out.sort(key=lambda d: d["cell"])
    return out


def _is_nan(x: float) -> bool:
    import math

    return math.isnan(x)


def _linear_interp_at(
    xs: list[float],
    ys: list[float],
    target_x: float,
) -> float:
    """Linear interpolation across (xs, ys) at target_x.

    Returns NaN if fewer than 2 valid points OR if extrapolating beyond the
    convex hull (we want the LOCAL bracketing read, not an extrapolation;
    extrapolation is flagged separately in ``is_extrapolation``).
    """
    import math

    pairs = sorted(
        [(x, y) for x, y in zip(xs, ys, strict=True) if not math.isnan(x) and not math.isnan(y)]
    )
    if len(pairs) < 2:
        return float("nan")
    # Strict bracket only; extrapolation is the caller's concern.
    if target_x < pairs[0][0] or target_x > pairs[-1][0]:
        return float("nan")
    for i in range(len(pairs) - 1):
        x1, y1 = pairs[i]
        x2, y2 = pairs[i + 1]
        if x1 <= target_x <= x2:
            if x2 == x1:
                return y1
            t = (target_x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
    return float("nan")


def _compute_local_matched_rate_read(
    *,
    diagnostics_514: list[dict],
    diagnostics_508_ft_anchors: list[dict],
    target_nat: float = MATCHED_RATE_TARGET_NAT,
) -> tuple[float, bool, list[dict]]:
    """Compute the #514 LOCAL linear-interpolation read at source ΔG = target_nat.

    Per plan §6.4: use the clean above-9-nat #514 cell(s) + #508 ft_b1 as the
    lower-flank anchor (#508 ft_b1 had source ΔG ≈ 8.2 nat — straddling
    target_nat=8 from below). Returns (local_read_nat, is_extrapolation,
    anchor_cells_used).

    ``is_extrapolation`` is True iff the chosen anchor pair does NOT straddle
    ``target_nat`` (per the Claude statistics critic's concern: when both
    bracketing cells sit above target, the read is extrapolation).
    """
    # Candidate anchor cells: every clean #514 FT cell (above-9-nat) +
    # every #508 FT anchor whose source_mean is non-NaN.
    candidates: list[dict] = []
    for d in diagnostics_514:
        if (
            d["clean_above_9_nat"]
            and d["source_mean"] is not None
            and d["held_out_mean"] is not None
        ):
            candidates.append(d)
    for d in diagnostics_508_ft_anchors:
        if d["source_mean"] is not None and d["held_out_mean"] is not None:
            candidates.append(d)

    if len(candidates) < 2:
        return (float("nan"), False, candidates)

    xs = [float(d["source_mean"]) for d in candidates]
    ys = [float(d["held_out_mean"]) for d in candidates]
    local_read = _linear_interp_at(xs, ys, target_nat)

    # Bracketing check: does (min(xs), max(xs)) straddle target_nat?
    is_extrap = True if not xs else not (min(xs) <= target_nat <= max(xs))

    return (local_read, is_extrap, candidates)


def _extract_bootstrap_matched_rate(delegated_analysis: dict) -> float:
    """Pull the cluster-bootstrap FT-arm read at the matched-rate target from
    #508's delegated ``run_analysis`` output.

    Returns the ``fullft_held_out_at_target_mean`` value from
    ``delegated["matched_rate_gap"]``, or NaN if the delegate did not compute
    it (e.g. the FT arm had fewer than 2 valid cells).
    """
    gap = (delegated_analysis or {}).get("matched_rate_gap") or {}
    val = gap.get("fullft_held_out_at_target_mean")
    if val is None:
        return float("nan")
    return float(val)


def _write_bracketing_report_514(
    *,
    diagnostics_514: list[dict],
    diagnostics_508_ft_anchors: list[dict],
    output_path: Path,
) -> dict[str, Any]:
    """Write the #514 bracketing report (plan §6.2 acceptance criterion).

    PASS iff there exists ≥1 NEW #514 FT cell with source ΔG > 9 nat AND
    clean (r-collapse < 50% AND held-out g_logprob ≤ -5 nat).

    The report records every #514 cell's verdict + the #508 FT anchor cells'
    (re-used) values for context.
    """
    clean_above_9 = [d for d in diagnostics_514 if d["clean_above_9_nat"]]
    pass_h1 = len(clean_above_9) >= 1

    report = {
        "schema_version": "i514_bracketing_v1",
        "criterion": (
            "PASS iff exists >=1 #514 FT cell with source_mean > 9.0 nat "
            "AND r_collapse_rate < 0.50 AND held_out_g_logprob_mean <= -5.0"
        ),
        "h1_pass": pass_h1,
        "clean_above_9_nat_cells": [d["cell"] for d in clean_above_9],
        "n_clean_above_9_nat": len(clean_above_9),
        "diagnostics_514": diagnostics_514,
        "diagnostics_508_ft_anchors": diagnostics_508_ft_anchors,
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    log.info(
        "[analyze] wrote bracketing report (H1 PASS=%s, %d clean-above-9-nat cells) → %s",
        pass_h1,
        len(clean_above_9),
        output_path,
    )
    return report


def _write_matched_rate_514(
    *,
    diagnostics_514: list[dict],
    diagnostics_508: list[dict],
    bracketing_report: dict[str, Any],
    delegated_analysis: dict[str, Any] | None,
    output_path: Path,
) -> dict[str, Any]:
    """Write the #514 matched-rate read with determinacy gate (B4 round-2 fix).

    Per plan §6.4: compute BOTH (a) the LOCAL linear-interpolation read at
    source ΔG = 8 nat using the clean above-9-nat #514 cell + #508 ft_b1 as
    the lower-flank anchor, AND (b) the cluster-bootstrap FT-arm read at the
    same target (pulled from the delegated #508 ``run_analysis`` →
    ``matched_rate_gap.fullft_held_out_at_target_mean``).

    The matched-rate read is DETERMINATE iff
    ``|local_read_nat − bootstrap_read_nat| ≤ DETERMINACY_GATE_THRESHOLD_NAT``
    (0.5 nat). Also flags ``is_extrapolation`` when the bracketing anchor pair
    does not strictly straddle 8 nat (per the Claude statistics critic's
    concern: when both anchor source ΔGs sit above target, the read is
    extrapolation, not interpolation).
    """
    import math

    output_path.parent.mkdir(parents=True, exist_ok=True)

    local_read, is_extrapolation, anchor_cells = _compute_local_matched_rate_read(
        diagnostics_514=diagnostics_514,
        diagnostics_508_ft_anchors=diagnostics_508,
        target_nat=MATCHED_RATE_TARGET_NAT,
    )
    bootstrap_read = _extract_bootstrap_matched_rate(delegated_analysis or {})

    if math.isnan(local_read) or math.isnan(bootstrap_read):
        gap_nat = float("nan")
        determinate = False
    else:
        gap_nat = abs(local_read - bootstrap_read)
        determinate = gap_nat <= DETERMINACY_GATE_THRESHOLD_NAT
    summary = {
        "schema_version": "i514_matched_rate_v2",
        "matched_slice_target_nats": MATCHED_RATE_TARGET_NAT,
        "matched_slice_band_nats": 1.0,
        # Plan §6.4 determinacy gate (B4 round-2 fix).
        "local_read_nat": local_read if not math.isnan(local_read) else None,
        "bootstrap_read_nat": bootstrap_read if not math.isnan(bootstrap_read) else None,
        "gap_nat": gap_nat if not math.isnan(gap_nat) else None,
        "determinate": determinate,
        "gate_threshold_nat": DETERMINACY_GATE_THRESHOLD_NAT,
        "is_extrapolation": is_extrapolation,
        "local_read_anchor_cells": [d["cell"] for d in anchor_cells],
        "h1_pass": bracketing_report["h1_pass"],
        "n_clean_above_9_nat": bracketing_report["n_clean_above_9_nat"],
        "clean_above_9_nat_cells": bracketing_report["clean_above_9_nat_cells"],
        "diagnostics_514_summary": [
            {
                "cell": d["cell"],
                "lever": d["lever"],
                "source_mean": d["source_mean"],
                "held_out_mean": d["held_out_mean"],
                "r_collapse_rate": d["r_collapse_rate"],
                "held_out_g_logprob_mean": d["held_out_g_logprob_mean"],
            }
            for d in diagnostics_514
        ],
        "diagnostics_508_anchors_summary": [
            {
                "cell": d["cell"],
                "source_mean": d["source_mean"],
                "held_out_mean": d["held_out_mean"],
            }
            for d in diagnostics_508
        ],
        "note": (
            f"Determinacy gate: |local − bootstrap| ≤ "
            f"{DETERMINACY_GATE_THRESHOLD_NAT} nat (plan §6.4). Local read = "
            f"linear interpolation across (#514 clean above-9-nat cells + "
            f"#508 FT anchors) at source ΔG = {MATCHED_RATE_TARGET_NAT} nat. "
            f"Bootstrap read = lora_vs_ft_508.analyze run_analysis "
            f"matched_rate_gap.fullft_held_out_at_target_mean. "
            f"is_extrapolation=True iff bracketing anchors don't straddle target."
        ),
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    output_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "[analyze] wrote matched-rate summary → %s "
        "(local=%.3f, bootstrap=%.3f, gap=%.3f, determinate=%s, is_extrap=%s)",
        output_path,
        local_read if not math.isnan(local_read) else float("nan"),
        bootstrap_read if not math.isnan(bootstrap_read) else float("nan"),
        gap_nat if not math.isnan(gap_nat) else float("nan"),
        determinate,
        is_extrapolation,
    )
    return summary


def _write_bootstrap_per_cell_514(
    *,
    cells_data: list[dict],
    output_path: Path,
) -> None:
    """Write the per-cell cluster-bootstrap arrays (plan §6.4 raw alongside processed).

    The cluster bootstrap is computed by #508's ``_crossed_cluster_bootstrap_gap``;
    here we serialize the per-cell (source_mean, held_out_mean) tuples that feed
    that downstream consumer. Mirrors plan Lens 11 (raw alongside processed).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "i514_bootstrap_per_cell_v1",
        "cells": [
            {
                "cell": c["cell"],
                "lever": _classify_lever(c["cell"]),
                "source_mean": c["source_mean"],
                "held_out_mean": c["held_out_mean"],
                "r_collapse_rate": c["r_collapse_rate"],
                "held_out_g_logprob_mean": c["held_out_g_logprob_mean"],
            }
            for c in cells_data
        ],
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    log.info("[analyze] wrote bootstrap-per-cell raw → %s", output_path)


def run_analysis_514(
    *,
    eval_jsons: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Plan §10 + §6.3 analyzer for #514.

    1. Locate the 6 #508 reference eval JSONs (LoRA b1/b2/b3 + FT b1/b2/b3).
    2. Merge with the 6 new #514 FT cells.
    3. Compute per-cell diagnostics (source_mean, r-collapse rate,
       held-out g_logprob_mean, lever, clean-above-9-nat).
    4. Delegate to ``lora_vs_ft_508.analyze.run_analysis`` for bracketing,
       cluster bootstrap, hero figure, trajectory figures.
    5. Write the 3 #514-specific artifacts: ``_bracketing_report.json``,
       ``_matched_rate_514.json``, ``_bootstrap_per_cell.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_jsons = _gather_reference_eval_jsons()
    all_jsons = list(eval_jsons) + reference_jsons
    log.info(
        "[analyze] running on %d #514 cells + %d #508 reference cells = %d total",
        len(eval_jsons),
        len(reference_jsons),
        len(all_jsons),
    )

    # Per-cell diagnostics (514 + 508 FT anchors).
    diag_514 = _per_cell_diagnostics(list(eval_jsons))
    diag_508_ft_anchors = [
        d
        for d in _per_cell_diagnostics([p for p in reference_jsons if "ft_" in p.name])
        if d["lever"] == "508_anchor"
    ]

    # Combined diagnostics for the bootstrap-per-cell file.
    combined_diag = diag_514 + _per_cell_diagnostics(reference_jsons)

    # ── Delegate to #508's run_analysis for the heavy lift. ──────────────────
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import (
        run_analysis as run_analysis_508,
    )

    # B6 round-2 fix: NO bare except around the delegate. If
    # run_analysis_508 raises, the dispatcher MUST crash with the original
    # traceback (CLAUDE.md fail-fast: the matched-rate cluster bootstrap +
    # hero figure ARE this delegate's output — silently swallowing means
    # the headline artifacts go missing while the dispatcher still exits 0).
    if all_jsons:
        delegated = run_analysis_508(eval_jsons=all_jsons, output_dir=output_dir)
    else:
        log.warning("[analyze] no eval JSONs to analyze — skipping #508 delegate")
        delegated = {"skipped": "no eval JSONs"}

    # ── Write #514-specific artifacts. ───────────────────────────────────────
    bracketing_path = output_dir / "_bracketing_report.json"
    matched_rate_path = output_dir / "_matched_rate_514.json"
    bootstrap_path = output_dir / "_bootstrap_per_cell.json"

    bracketing_report = _write_bracketing_report_514(
        diagnostics_514=diag_514,
        diagnostics_508_ft_anchors=diag_508_ft_anchors,
        output_path=bracketing_path,
    )
    matched_rate = _write_matched_rate_514(
        diagnostics_514=diag_514,
        diagnostics_508=diag_508_ft_anchors,
        bracketing_report=bracketing_report,
        delegated_analysis=delegated,
        output_path=matched_rate_path,
    )
    _write_bootstrap_per_cell_514(
        cells_data=combined_diag,
        output_path=bootstrap_path,
    )

    analysis = {
        "schema_version": "i514_analysis_v1",
        "n_cells_514": len(eval_jsons),
        "n_cells_508_reference": len(reference_jsons),
        "delegated_508_analysis": delegated,
        "bracketing_report_path": str(bracketing_path),
        "matched_rate_514_path": str(matched_rate_path),
        "bootstrap_per_cell_path": str(bootstrap_path),
        "h1_pass": bracketing_report["h1_pass"],
        "clean_above_9_nat_cells": bracketing_report["clean_above_9_nat_cells"],
        "matched_rate_summary": matched_rate,
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    out_path = output_dir / "analysis_514.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    log.info(
        "[analyze] wrote #514 analysis → %s (H1 PASS=%s)", out_path, bracketing_report["h1_pass"]
    )
    return analysis
